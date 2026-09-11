"""Periodic-orbit trajectory optimisation through the differentiable simulator.

Phase 3 left one thing unsettled: whether *any* muscle gait exists for ``hopper_muscle.xml``.
Seven optimiser, reward and actuator combinations all landed in one of two attractors, and PPO
landed where SHAC did, so the question is about the model rather than the optimiser. This driver
asks it without a policy at all.

It optimises an open-loop control sequence over one cycle, plus the initial state, directly through
the adjoint, and asks for a **periodic orbit that advances**. That formulation does the
discriminating work, and stage 0 measured that it does
(``docs/research/phase4-capability/protocol.md``)::

    real motor gait   residual 0.234   advance +1.803 m/cycle
    standing          residual 0.482   advance -0.006 m/cycle   -> excluded by the objective
    diving            residual 1.007   advance +0.128 m/cycle   -> excluded by the periodicity term

Both Phase 3 attractors are therefore outside the feasible set by construction, which is what no
reward setting in Phase 3 achieved.

Every world is an independent random restart, optimised simultaneously in one backward pass, so
512 restarts cost one rollout's worth of wall clock rather than 512.

Gates, thresholds and the response to each outcome are pre-registered in the protocol. Run the
motor positive control first: **if it fails, the muscle result is void.**

Usage::

    # stage 1a, cost probe
    .venv/Scripts/python.exe scripts/trajopt_hopper.py --cfg configs/hopper_motor_shac.yaml \\
        --scales docs/research/phase4-capability/gait_motor_seed2.json \\
        --cycle 32 --worlds 256 --iters 20 --probe

    # a real cell
    .venv/Scripts/python.exe scripts/trajopt_hopper.py --cfg configs/hopper_muscle_shac.yaml \\
        --scales docs/research/phase4-capability/gait_motor_seed2.json \\
        --cycle 27 --worlds 512 --iters 500 --out logs/phase4/muscle_T27.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from msk_warp import PACKAGE_ROOT
from msk_warp.bridge import WarpSimStep
from msk_warp.envs import ENV_MAP
from msk_warp.utils.gait import periodicity_residual, shape_vector

# Bounds on the free initial state, from the protocol. Height and pitch are symmetric about the
# reset pose; thigh and leg are mapped into the interior of their (-2.618, 0) range so a restart
# never starts exactly on a joint limit, where the constraint derivative is undefined (G3.2).
HEIGHT_BOUND = 0.15       # m, about the reset height
PITCH_BOUND = 0.30        # rad, against a termination angle of pi/6
KNEE_HIP_SPAN = 1.20      # rad, span available to thigh and leg below the rest pose
FOOT_BOUND = 0.70         # rad, inside the (-0.785, 0.785) range
# Inset every limited joint by the same 0.05 rad G3.2 uses to move a test pose off a limit.
# Without it a saturated sigmoid underflows to exactly 0.0 in float32 and a restart begins on the
# thigh or leg upper limit, where the constraint switches on and the derivative is undefined.
JOINT_MARGIN = 0.05
QVEL_BOUND = (2.0, 2.0, 3.0, 5.0, 5.0, 5.0)

TERMINATION_HEIGHT = -0.45


def load_cfg(cfg_path):
    if not os.path.isabs(cfg_path) and (PACKAGE_ROOT / cfg_path).exists():
        cfg_path = str(PACKAGE_ROOT / cfg_path)
    with open(cfg_path) as fh:
        return yaml.safe_load(fh), cfg_path


def build_env(cfg, worlds, device, episode_length):
    """Differentiable environment with the episode machinery out of the way.

    Early termination and stochastic init are off and the episode is longer than any rollout here,
    so no world is ever reset mid-cycle. The environment is used for its model, its substep count
    and its action map; the rollout is driven through ``WarpSimStep`` directly rather than through
    ``env.step``, which would apply reward, termination and reset logic this objective does not want.
    """
    env_cfg = dict(cfg["params"]["env"])
    name = env_cfg.pop("name")
    env_cfg.pop("num_actors", None)
    env_cfg.pop("episode_length", None)
    env_cfg.pop("stochastic_init", None)
    env_cfg.pop("early_termination", None)
    return ENV_MAP[name](
        num_envs=worlds,
        device=device,
        no_grad=False,
        episode_length=episode_length,
        stochastic_init=False,
        early_termination=False,
        **env_cfg,
    )


def _span():
    """Interior span for thigh and leg, both margins removed."""
    return KNEE_HIP_SPAN - 2.0 * JOINT_MARGIN


def initial_state(z, n_act_state, device):
    """Map unconstrained parameters onto a bounded, physical initial ``(qpos, qvel, act)``.

    ``qpos[0]`` is always 0: displacement is measured from wherever the cycle starts, so the
    absolute track position is not a degree of freedom.
    """
    qvel_bound = torch.tensor(QVEL_BOUND, device=device)
    zero = torch.zeros(z.shape[0], 1, device=device)
    qpos = torch.cat([
        zero,                                                   # x
        HEIGHT_BOUND * torch.tanh(z[:, 0:1]),                   # height, relative to the reset pose
        PITCH_BOUND * torch.tanh(z[:, 1:2]),                    # torso pitch
        -(JOINT_MARGIN + _span() * torch.sigmoid(z[:, 2:3])),   # thigh, strictly inside its range
        -(JOINT_MARGIN + _span() * torch.sigmoid(z[:, 3:4])),   # leg
        FOOT_BOUND * torch.tanh(z[:, 4:5]),                     # foot
    ], dim=1)
    qvel = qvel_bound * torch.tanh(z[:, 5:11])
    act = torch.sigmoid(z[:, 11:11 + n_act_state]) if n_act_state else torch.zeros(z.shape[0], 0, device=device)
    return qpos, qvel, act


def rollout(env, u_logits, z, scales, cycle, n_act_state, device):
    """One cycle from the parameterised initial state. Returns the objective's three terms.

    Kept differentiable end to end: the adjoint runs from the objective back through every substep
    of every control step to both the control sequence and the initial state.
    """
    qpos, qvel, act = initial_state(z, n_act_state, device)
    start_x = qpos[:, 0]
    # Mechanical state only: activation is excluded so the motor and muscle models are scored
    # on one literal 11-component ruler (protocol amendment 1). Activation closure is enforced
    # by the open-loop verification instead, which diverges if act does not return.
    start_shape = shape_vector(qpos, qvel)

    violation = torch.zeros(qpos.shape[0], device=device)
    for t in range(cycle):
        ctrl = env._to_ctrl(torch.tanh(u_logits[:, t]))
        qpos, qvel, act = WarpSimStep.apply(ctrl, qpos, qvel, act, env)
        # Same guard SHAC's rollout uses: a diverging restart must not poison the tape.
        qpos = qpos.clamp(-100.0, 100.0)
        qvel = qvel.clamp(-100.0, 100.0)
        violation = violation + torch.relu(TERMINATION_HEIGHT - qpos[:, 1])

    dt = env.substeps * float(env.mjm.opt.timestep)
    velocity = (qpos[:, 0] - start_x) / (cycle * dt)
    residual = periodicity_residual(shape_vector(qpos, qvel), start_shape, scales)
    return velocity, residual, violation / cycle


@torch.no_grad()
def verify(env, u_logits, z, scales, cycle, n_act_state, device, cycles):
    """Re-simulate open loop for ``cycles`` consecutive cycles. Survival and speed, no gradients.

    An optimum is not yet a gait. This is the check the Phase 3 open-loop sweep lacked.
    """
    qpos, qvel, act = initial_state(z, n_act_state, device)
    start_x = qpos[:, 0].clone()
    alive = torch.ones(qpos.shape[0], dtype=torch.bool, device=device)
    survived = torch.zeros(qpos.shape[0], device=device)
    for _ in range(cycles):
        for t in range(cycle):
            ctrl = env._to_ctrl(torch.tanh(u_logits[:, t]))
            qpos, qvel, act = WarpSimStep.apply(ctrl, qpos, qvel, act, env)
            qpos = qpos.clamp(-100.0, 100.0)
            qvel = qvel.clamp(-100.0, 100.0)
            alive &= qpos[:, 1] >= TERMINATION_HEIGHT
            survived += alive.float()
    dt = env.substeps * float(env.mjm.opt.timestep)
    velocity = (qpos[:, 0] - start_x) / (cycles * cycle * dt)
    return alive, velocity, survived / (cycles * cycle)


def summarise(tag, velocity, residual, violation, objective):
    v, r, h, j = (t.detach().cpu().numpy() for t in (velocity, residual, violation, objective))
    best = int(np.argmax(j))
    return {
        "tag": tag,
        "best_world": best,
        "best_objective": float(j[best]),
        "best_velocity": float(v[best]),
        "best_residual": float(r[best]),
        "best_violation": float(h[best]),
        "velocity_median": float(np.median(v)),
        "velocity_max": float(v.max()),
        "residual_median": float(np.median(r)),
        "residual_min": float(r.min()),
        "violation_median": float(np.median(h)),
        "objective_median": float(np.median(j)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--scales", required=True,
                    help="stage-0 JSON; its component scales are the single ruler both models use")
    ap.add_argument("--cycle", type=int, required=True, help="cycle length T_c in control steps")
    ap.add_argument("--worlds", type=int, default=512, help="independent random restarts")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.02)
    # lambda = 4.0 is derived from stage 0's three reference points on the shared ruler, not tuned:
    # real gait (v 4.01, R 0.234), standing (v -0.01, R 0.088), diving (v 1.28, R 0.485). It is the
    # smallest round value for which J(gait) 3.07 > J(standing) -0.36 > J(diving) -0.66. Below 3.25
    # the dive outranks standing, which would point the optimiser at the worse of the two attractors.
    ap.add_argument("--lam", type=float, default=4.0, help="periodicity weight; see the protocol")
    ap.add_argument("--mu", type=float, default=20.0, help="height-violation weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-cycles", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--probe", action="store_true",
                    help="report term magnitudes and seconds per iteration, then stop (stage 1a)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg, cfg_path = load_cfg(args.cfg)
    env = build_env(cfg, args.worlds, args.device, args.cycle * (args.verify_cycles + 2))

    with open(args.scales) as fh:
        stage0 = json.load(fh)
    scales_list = stage0["per_world"][0]["scales"]
    scales = torch.tensor(scales_list, device=args.device)

    n_act_state = env.warp_data.act.shape[1] if env.has_act else 0
    expected = (env.mjm.nq - 1) + env.mjm.nv
    if len(scales_list) != expected:
        raise SystemExit(
            f"--scales has {len(scales_list)} components but this model's shape vector has "
            f"{expected}. The stage-0 ruler must come from a model with the same mechanical layout."
        )

    u_logits = torch.randn(args.worlds, args.cycle, env.num_actions, device=args.device, requires_grad=True)
    z = torch.randn(args.worlds, 11 + n_act_state, device=args.device, requires_grad=True)
    optimiser = torch.optim.Adam([u_logits, z], lr=args.lr)

    def objective():
        velocity, residual, violation = rollout(
            env, u_logits, z, scales, args.cycle, n_act_state, args.device,
        )
        return velocity, residual, violation, velocity - args.lam * residual - args.mu * violation

    history = []
    grad_norms = []
    started = time.time()

    for iteration in range(args.iters):
        env.clear_grad()
        optimiser.zero_grad(set_to_none=True)
        velocity, residual, violation, obj = objective()
        (-obj.sum()).backward()
        grad_norms.append(float(torch.cat([u_logits.grad.flatten(), z.grad.flatten()]).norm()))
        optimiser.step()

        if args.probe or iteration % args.log_every == 0 or iteration == args.iters - 1:
            row = summarise(f"iter{iteration}", velocity, residual, violation, obj)
            row["iteration"] = iteration
            row["grad_norm"] = grad_norms[-1]
            row["seconds"] = time.time() - started
            history.append(row)
            print(f"iter {iteration:4d}  J {row['best_objective']:+7.3f}  "
                  f"v {row['best_velocity']:+6.2f}  R {row['best_residual']:.3f}  "
                  f"H {row['best_violation']:.3f}  |  median v {row['velocity_median']:+6.2f} "
                  f"R {row['residual_median']:.3f}  |g| {row['grad_norm']:9.2e}  "
                  f"{row['seconds']:6.1f}s", flush=True)

        if args.probe and iteration == min(args.iters, 20) - 1:
            elapsed = time.time() - started
            print(f"\nPROBE: {elapsed / (iteration + 1):.2f} s per iteration at "
                  f"worlds={args.worlds} cycle={args.cycle}")
            print(f"PROBE: term magnitudes at iteration 0 -> "
                  f"|v| {abs(history[0]['velocity_median']):.3f}   "
                  f"lambda*R {args.lam * history[0]['residual_median']:.3f}   "
                  f"mu*H {args.mu * history[0]['violation_median']:.3f}")
            return

    velocity, residual, violation, obj = objective()
    alive, verified_velocity, alive_fraction = verify(
        env, u_logits.detach(), z.detach(), scales, args.cycle, n_act_state, args.device,
        args.verify_cycles,
    )

    # A candidate counts only if it survives every verification cycle and its open-loop speed is
    # within 25% of what the optimiser reported (protocol, "Verification").
    consistent = (verified_velocity - velocity).abs() <= 0.25 * velocity.abs().clamp(min=1e-6)
    qualifies = alive & consistent
    order = torch.argsort(torch.where(qualifies, verified_velocity, torch.full_like(verified_velocity, -1e9)),
                          descending=True)

    result = {
        "cfg": cfg_path,
        "scales_source": args.scales,
        "cycle": args.cycle,
        "worlds": args.worlds,
        "iters": args.iters,
        "lr": args.lr,
        "lambda": args.lam,
        "mu": args.mu,
        "seed": args.seed,
        "verify_cycles": args.verify_cycles,
        "seconds_total": time.time() - started,
        "seconds_per_iter": (time.time() - started) / args.iters,
        "grad_norm_median": float(np.median(grad_norms)),
        "grad_norm_p90": float(np.percentile(grad_norms, 90)),
        "grad_norm_max": float(np.max(grad_norms)),
        "final": summarise("final", velocity, residual, violation, obj),
        "verified": {
            "survived_all_cycles": int(alive.sum()),
            "speed_consistent": int(consistent.sum()),
            "qualifying": int(qualifies.sum()),
            "best_verified_velocity": float(verified_velocity[order[0]]) if int(qualifies.sum()) else None,
            "best_verified_residual": float(residual[order[0]]) if int(qualifies.sum()) else None,
            "top": [
                {
                    "world": int(w),
                    "optimiser_velocity": float(velocity[w]),
                    "verified_velocity": float(verified_velocity[w]),
                    "residual": float(residual[w]),
                    "alive_fraction": float(alive_fraction[w]),
                }
                for w in order[:10].tolist()
            ],
        },
        "history": history,
        # Standing versus diving in trajectory space rather than policy space.
        "restarts_near_zero_velocity": int((velocity.abs() < 0.1).sum()),
        "restarts_above_gate_residual": int((residual > 0.4684).sum()),
    }

    print(f"\nverified: {result['verified']['qualifying']} of {args.worlds} restarts qualify "
          f"(survived all {args.verify_cycles} cycles and speed within 25%)")
    if result["verified"]["best_verified_velocity"] is not None:
        print(f"best verified: {result['verified']['best_verified_velocity']:+.3f} m/s at "
              f"residual {result['verified']['best_verified_residual']:.4f}")
    else:
        print("best verified: none -- no restart survived verification")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        npz = os.path.splitext(args.out)[0] + "_params.npz"
        np.savez_compressed(npz, u_logits=u_logits.detach().cpu().numpy(), z=z.detach().cpu().numpy(),
                            order=order.cpu().numpy())
        print(f"wrote {args.out} and {npz}")


if __name__ == "__main__":
    main()
