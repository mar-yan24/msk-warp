"""Compare SHAC's autodiff policy gradient against finite differences of the same objective.

The actor gradient is what SHAC actually descends, so it is the quantity worth validating before
committing to a training run. Per-step bridge gradients can be near-exact while the aggregate
policy gradient is still useless, and the reverse is possible too.

The measurement projects both gradients onto N random unit directions in actor parameter space::

    ad_proj[i] = g_AD . d_i
    fd_proj[i] = (L(theta + eps d_i) - L(theta - eps d_i)) / (2 eps)

and reports the cosine between those two N-vectors, with a bootstrap CI95. Only ``mu_net``
parameters are used; the policy's ``logstd`` has zero gradient under a deterministic rollout and
would only pad the dimension.

Every rollout is run under a determinism freeze: the physics state, both RNG streams, the
observation normaliser and all episode meters are snapshotted and restored, so three rollouts at
the same parameters are bit-identical. A precondition check runs first and aborts if two rollouts
at identical parameters disagree, because the finite differences would then be measuring noise.

Usage::

    .venv/Scripts/python.exe scripts/policy_gradient_cosine.py --cfg configs/hopper_muscle_shac.yaml
    .venv/Scripts/python.exe scripts/policy_gradient_cosine.py --cfg configs/hopper_motor_shac.yaml \
        --directions 30 --num-envs 16 --steps-num 16 --out cosine.json
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import warp as wp
import yaml

from msk_warp import PACKAGE_ROOT
from msk_warp.algorithms.shac import SHAC


def build(cfg_path, logdir, seed, num_envs, steps_num, episode_length):
    """SHAC instance with the determinism freeze applied: no reset noise, no early termination."""
    if not os.path.isabs(cfg_path) and (PACKAGE_ROOT / cfg_path).exists():
        cfg_path = str(PACKAGE_ROOT / cfg_path)
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["params"]["general"]["logdir"] = logdir
    cfg["params"]["general"]["seed"] = seed
    cfg["params"]["env"]["num_actors"] = num_envs
    cfg["params"]["env"]["episode_length"] = episode_length
    cfg["params"]["env"]["stochastic_init"] = False
    cfg["params"]["env"]["early_termination"] = False
    cfg["params"]["config"]["num_actors"] = num_envs
    cfg["params"]["config"]["steps_num"] = steps_num
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["save_interval"] = 0
    cfg["params"]["config"]["eval_interval"] = 0
    return SHAC(cfg), cfg_path


def snapshot(shac):
    """Everything a rollout mutates, so the next rollout starts from the same place."""
    d = shac.env.warp_data
    snap = {
        "qpos": wp.clone(d.qpos), "qvel": wp.clone(d.qvel), "time": wp.clone(d.time),
        "act": wp.clone(d.act) if d.act.shape[1] > 0 else None,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "progress_buf": shac.env.progress_buf.clone(),
        "reset_buf": shac.env.reset_buf.clone(),
        "termination_buf": shac.env.termination_buf.clone(),
        "env_actions": shac.env.actions.clone(),
        "ret": shac.ret.clone(),
        "episode_loss": shac.episode_loss.clone(),
        "episode_discounted_loss": shac.episode_discounted_loss.clone(),
        "episode_gamma": shac.episode_gamma.clone(),
        "episode_length": shac.episode_length.clone(),
        "meters": copy.deepcopy((shac.episode_loss_meter, shac.episode_discounted_loss_meter,
                                 shac.episode_length_meter)),
        "his": (list(shac.episode_loss_his), list(shac.episode_discounted_loss_his),
                list(shac.episode_length_his)),
        "step_count": shac.step_count,
    }
    if shac.obs_rms is not None:
        snap["obs_rms"] = (shac.obs_rms.mean.clone(), shac.obs_rms.var.clone(), shac.obs_rms.count)
    if shac.ret_rms is not None:
        snap["ret_rms"] = (shac.ret_rms.mean.clone(), shac.ret_rms.var.clone(), shac.ret_rms.count)
    return snap


def restore(shac, snap):
    d = shac.env.warp_data
    wp.copy(d.qpos, snap["qpos"])
    wp.copy(d.qvel, snap["qvel"])
    wp.copy(d.time, snap["time"])
    if snap["act"] is not None:
        wp.copy(d.act, snap["act"])
    wp.synchronize()
    torch.set_rng_state(snap["torch_rng"])
    torch.cuda.set_rng_state_all(snap["cuda_rng"])
    if shac.obs_rms is not None:
        shac.obs_rms.mean, shac.obs_rms.var, shac.obs_rms.count = (
            snap["obs_rms"][0].clone(), snap["obs_rms"][1].clone(), snap["obs_rms"][2])
    if shac.ret_rms is not None:
        shac.ret_rms.mean, shac.ret_rms.var, shac.ret_rms.count = (
            snap["ret_rms"][0].clone(), snap["ret_rms"][1].clone(), snap["ret_rms"][2])
    shac.env.progress_buf.copy_(snap["progress_buf"])
    shac.env.reset_buf.copy_(snap["reset_buf"])
    shac.env.termination_buf.copy_(snap["termination_buf"])
    shac.env.actions.copy_(snap["env_actions"])
    shac.ret.copy_(snap["ret"])
    shac.episode_loss.copy_(snap["episode_loss"])
    shac.episode_discounted_loss.copy_(snap["episode_discounted_loss"])
    shac.episode_gamma.copy_(snap["episode_gamma"])
    shac.episode_length.copy_(snap["episode_length"])
    (shac.episode_loss_meter, shac.episode_discounted_loss_meter,
     shac.episode_length_meter) = copy.deepcopy(snap["meters"])
    (shac.episode_loss_his, shac.episode_discounted_loss_his,
     shac.episode_length_his) = (list(x) for x in snap["his"])
    shac.step_count = snap["step_count"]


def theta(shac):
    return torch.nn.utils.parameters_to_vector(shac.actor.mu_net.parameters()).detach().clone()


def set_theta(shac, vec):
    torch.nn.utils.vector_to_parameters(vec, shac.actor.mu_net.parameters())


def loss_only(shac):
    with torch.no_grad():
        return float(shac.compute_actor_loss(deterministic=True))


def loss_and_grad(shac):
    """Forward plus backward. The Warp backward is destructive, so the state is saved around it."""
    shac.actor_optimizer.zero_grad()
    loss = shac.compute_actor_loss(deterministic=True)
    d = shac.env.warp_data
    with torch.no_grad():
        saved = (wp.clone(d.qpos), wp.clone(d.qvel), wp.clone(d.time),
                 wp.clone(d.act) if d.act.shape[1] > 0 else None)
    loss.backward()
    wp.copy(d.qpos, saved[0])
    wp.copy(d.qvel, saved[1])
    wp.copy(d.time, saved[2])
    if saved[3] is not None:
        wp.copy(d.act, saved[3])
    wp.synchronize()
    g = torch.cat([
        p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=p.device)
        for p in shac.actor.mu_net.parameters()
    ]).detach().clone()
    return float(loss), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/hopper_muscle_shac.yaml")
    ap.add_argument("--directions", type=int, default=30)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--steps-num", type=int, default=16)
    ap.add_argument("--episode-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", default="logs/_pg_cosine")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shac, cfg_path = build(args.cfg, args.logdir, args.seed, args.num_envs,
                           args.steps_num, args.episode_length)
    shac.env.begin_epoch(epoch=0, max_epochs=1)
    shac.initialize_env()
    shac.episode_loss.zero_()
    shac.episode_discounted_loss.zero_()
    shac.episode_length.zero_()
    shac.episode_gamma.fill_(1.0)
    snap = snapshot(shac)

    l1 = loss_only(shac)
    restore(shac, snap)
    l2 = loss_only(shac)
    drift = abs(l1 - l2)
    print(f"determinism freeze: L1={l1:.8f} L2={l2:.8f} |dL|={drift:.3e}")
    if drift > 1e-5:
        raise SystemExit(f"determinism freeze broken (|dL|={drift:.3e}); finite differences "
                         "would measure noise, not curvature")

    restore(shac, snap)
    l0, g_ad = loss_and_grad(shac)
    t0 = theta(shac)
    n = t0.numel()
    print(f"theta dim {n}, |theta| {float(t0.norm()):.4f}, |g_AD| {float(g_ad.norm()):.6f}, L0 {l0:.6f}")
    if float(g_ad.norm()) == 0.0:
        raise SystemExit("the autodiff policy gradient is identically zero")

    gen = torch.Generator(device=t0.device).manual_seed(args.seed)
    ad_proj, fd_proj = np.zeros(args.directions), np.zeros(args.directions)
    for i in range(args.directions):
        d = torch.randn(n, generator=gen, device=t0.device, dtype=t0.dtype)
        d /= d.norm()
        ad_proj[i] = float(g_ad @ d)
        set_theta(shac, t0 + args.eps * d)
        restore(shac, snap)
        lp = loss_only(shac)
        set_theta(shac, t0 - args.eps * d)
        restore(shac, snap)
        lm = loss_only(shac)
        fd_proj[i] = (lp - lm) / (2.0 * args.eps)
        set_theta(shac, t0)

    cos = float(ad_proj @ fd_proj / (np.linalg.norm(ad_proj) * np.linalg.norm(fd_proj)))
    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(2000):
        k = rng.integers(0, args.directions, args.directions)
        a, b = ad_proj[k], fd_proj[k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            boot.append(a @ b / (na * nb))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    scale = float(np.polyfit(ad_proj, fd_proj, 1)[0]) if args.directions > 2 else float("nan")

    print(f"cosine(AD, FD) over {args.directions} directions: {cos:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]")
    print(f"FD/AD slope {scale:+.3f}   |ad_proj| {np.linalg.norm(ad_proj):.5f}   "
          f"|fd_proj| {np.linalg.norm(fd_proj):.5f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "cfg": cfg_path, "directions": args.directions, "eps": args.eps,
                "num_envs": args.num_envs, "steps_num": args.steps_num, "seed": args.seed,
                "determinism_drift": drift, "loss": l0, "theta_dim": int(n),
                "g_ad_norm": float(g_ad.norm()), "cosine": cos,
                "ci95": [float(lo), float(hi)], "fd_over_ad_slope": scale,
                "ad_proj": ad_proj.tolist(), "fd_proj": fd_proj.tolist(),
            }, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
