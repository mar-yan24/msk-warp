"""Behavioural evaluation of a trained hopper policy: velocity, fall rate, return.

The Phase 3 training gates are behavioural, never the SHAC loss, and the loss is a poor proxy: a
policy that stands still scores well on the posture terms while going nowhere.

Forward velocity is accumulated from the observation each step rather than read from the final
state. Reading the final state gives nonsense, because the environment resets a world in the same
step it reports done, so the position that survives to the end of the loop is the fresh one. That
mistake once turned a policy walking 43 m into a reported 0.01 m.

Usage::

    .venv/Scripts/python.exe scripts/evaluate_hopper.py --cfg configs/hopper_motor_shac.yaml \
        --checkpoint logs/phase3/motor_ad_seed0/best_policy.pt --episodes 16
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from msk_warp import PACKAGE_ROOT
from msk_warp.envs import ENV_MAP


def load_cfg(cfg_path):
    if not os.path.isabs(cfg_path) and (PACKAGE_ROOT / cfg_path).exists():
        cfg_path = str(PACKAGE_ROOT / cfg_path)
    with open(cfg_path) as fh:
        return yaml.safe_load(fh), cfg_path


def build_env(cfg, episodes, device):
    env_cfg = dict(cfg["params"]["env"])
    name = env_cfg.pop("name")
    env_cfg.pop("num_actors", None)
    env_cfg.pop("backward_mode", None)
    return ENV_MAP[name](
        num_envs=episodes,
        device=device,
        no_grad=True,
        episode_length=env_cfg.pop("episode_length", 1000),
        **env_cfg,
    )


@torch.no_grad()
def evaluate(env, actor, obs_rms, deterministic=True):
    """One deterministic episode per world, scored only over the steps a world is still alive."""
    obs = env.reset()
    n = env.num_envs
    device = obs.device
    returns = torch.zeros(n, device=device)
    lengths = torch.zeros(n, dtype=torch.long, device=device)
    vel_sum = torch.zeros(n, device=device)
    height_sum = torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)

    actor.eval()
    for _ in range(env.episode_length):
        obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
        actions = torch.tanh(actor(obs_in, deterministic=deterministic))
        obs, rew, done, _extras = env.step(actions)[:4]
        a = alive.float()
        returns += rew * a
        vel_sum += obs[:, 5] * a          # qvel[0], forward velocity
        height_sum += obs[:, 0] * a
        lengths += alive.long()
        alive &= done == 0
        if not alive.any():
            break

    steps = lengths.clamp(min=1).float()
    dt = env.substeps * float(env.mjm.opt.timestep)
    return {
        "episodes": int(n),
        "mean_return": float(returns.mean()),
        "mean_length": float(lengths.float().mean()),
        "fall_rate": float((lengths < env.episode_length).float().mean()),
        "mean_velocity": float((vel_sum / steps).mean()),
        "median_velocity": float((vel_sum / steps).median()),
        "mean_height": float((height_sum / steps).mean()),
        "mean_distance_m": float((vel_sum * dt).mean()),
        "control_dt": dt,
        "per_episode_velocity": (vel_sum / steps).cpu().numpy().round(3).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, cfg_path = load_cfg(args.cfg)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    actor, obs_rms = ckpt[0].to(args.device), (ckpt[3].to(args.device) if ckpt[3] is not None else None)

    env = build_env(cfg, args.episodes, args.device)
    res = evaluate(env, actor, obs_rms, deterministic=not args.stochastic)
    res["cfg"] = cfg_path
    res["checkpoint"] = args.checkpoint

    print(f"{os.path.basename(os.path.dirname(args.checkpoint))}: "
          f"return {res['mean_return']:8.1f}   velocity {res['mean_velocity']:+5.2f} m/s   "
          f"fall {res['fall_rate']:5.1%}   length {res['mean_length']:6.1f}   "
          f"distance {res['mean_distance_m']:6.1f} m   height {res['mean_height']:+.3f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
