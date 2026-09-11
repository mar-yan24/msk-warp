"""Extract the limit cycle of a trained policy: period, periodicity residual, advance per cycle.

Phase 4 scores candidate trajectories on how well a cycle closes. That score is meaningless
without a reference, so it is calibrated here against a gait that is known to exist: the motor
hopper policy that G3.4 evaluated at 3.83 m/s with a 0% fall rate. The numbers this prints are the
``R_ref`` and ``T_ref`` that ``docs/research/phase4-capability/protocol.md`` writes its gates
against, and the component scales it produces are the single ruler the motor and muscle models are
both measured on.

This is instrument calibration, not a test of a hypothesis. It also doubles as the extractor for a
reference trajectory, which is what a tracking or distillation step would consume later.

Usage::

    .venv/Scripts/python.exe scripts/extract_gait_cycle.py \
        --cfg configs/hopper_motor_shac.yaml \
        --checkpoint logs/phase3/motor_ad_seed2/best_policy.pt \
        --out docs/research/phase4-capability/gait_motor_seed2.json
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from msk_warp import PACKAGE_ROOT
from msk_warp.envs import ENV_MAP
from msk_warp.utils.gait import component_scales, find_period, shape_vector


def load_cfg(cfg_path):
    if not os.path.isabs(cfg_path) and (PACKAGE_ROOT / cfg_path).exists():
        cfg_path = str(PACKAGE_ROOT / cfg_path)
    with open(cfg_path) as fh:
        return yaml.safe_load(fh), cfg_path


def build_env(cfg, worlds, device, stochastic_init, episode_length):
    """Environment for a plain rollout: no gradients, no early termination, no mid-trace reset."""
    env_cfg = dict(cfg["params"]["env"])
    name = env_cfg.pop("name")
    env_cfg.pop("num_actors", None)
    env_cfg.pop("backward_mode", None)
    env_cfg.pop("episode_length", None)
    env_cfg.pop("stochastic_init", None)
    env_cfg.pop("early_termination", None)
    return ENV_MAP[name](
        num_envs=worlds,
        device=device,
        no_grad=True,
        episode_length=episode_length,
        stochastic_init=stochastic_init,
        early_termination=False,
        **env_cfg,
    )


@torch.no_grad()
def roll(env, actor, obs_rms, steps):
    """Roll the deterministic policy and return the state and action traces.

    ``env.step`` returns ``None`` for the state in no-grad mode, so the state is read back from
    the environment after each step. Early termination is off, so no world resets mid-trace.
    """
    obs = env.reset()
    qpos_t, qvel_t, act_t, action_t = [], [], [], []
    actor.eval()
    for _ in range(steps):
        obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
        actions = torch.tanh(actor(obs_in, deterministic=True))
        obs = env.step(actions)[0]
        qpos, qvel, act = env.state_tensors()
        qpos_t.append(qpos)
        qvel_t.append(qvel)
        act_t.append(act)
        action_t.append(actions)
    stack = lambda parts: torch.stack(parts, dim=0)  # noqa: E731  -> (steps, worlds, dim)
    return stack(qpos_t), stack(qvel_t), stack(act_t), stack(action_t)


def analyse_world(qpos, qvel, act, transient, dt, min_period, max_period, tolerance,
                  mechanical_only=False, external_scales=None):
    """Period, residual and advance per cycle for one world's ``(steps, dim)`` traces.

    ``mechanical_only`` drops muscle activation from the shape vector, which is what the Phase 4
    gate uses: it makes the motor and muscle models share one literal 11-component ruler
    (protocol amendment 1). ``external_scales`` scores this trace against another run's scales,
    which is how the muscle attractors are put on the motor gait's ruler.
    """
    qpos, qvel, act = qpos[transient:], qvel[transient:], act[transient:]
    trace = shape_vector(qpos, qvel, None if mechanical_only else act)
    scales = component_scales(trace) if external_scales is None else external_scales
    period, residual, periods, curve = find_period(
        trace, scales, min_period=min_period, max_period=max_period, tolerance=tolerance,
    )
    x = qpos[:, 0]
    advance = float((x[period:] - x[:-period]).mean())
    return {
        "period_steps": period,
        "period_seconds": period * dt,
        "residual": residual,
        "advance_per_cycle_m": advance,
        "cycle_velocity_mps": advance / (period * dt),
        "trace_velocity_mps": float((x[-1] - x[0]) / ((len(x) - 1) * dt)),
        "scales": [round(float(s), 6) for s in scales],
        "residual_curve": [[int(p), round(float(r), 6)] for p, r in zip(periods, curve)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=400, help="control steps to roll")
    ap.add_argument("--transient", type=int, default=100, help="leading steps discarded")
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--stochastic-init", action="store_true",
                    help="off by default; the protocol calibrates on the deterministic start")
    ap.add_argument("--min-period", type=int, default=6)
    ap.add_argument("--max-period", type=int, default=80)
    ap.add_argument("--tolerance", type=float, default=1.2)
    ap.add_argument("--mechanical-only", action="store_true",
                    help="drop activation from the shape vector, so motor and muscle share one ruler")
    ap.add_argument("--scales-from", default=None,
                    help="score against another run's component scales instead of this trace's own")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="JSON summary; a .npz of traces is written beside it")
    args = ap.parse_args()

    if args.transient >= args.steps:
        raise SystemExit(f"--transient {args.transient} leaves no trace out of --steps {args.steps}")

    cfg, cfg_path = load_cfg(args.cfg)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    actor, obs_rms = ckpt[0].to(args.device), (ckpt[3].to(args.device) if ckpt[3] is not None else None)

    env = build_env(cfg, args.worlds, args.device, args.stochastic_init, args.steps + 1)
    qpos, qvel, act, actions = roll(env, actor, obs_rms, args.steps)
    dt = env.substeps * float(env.mjm.opt.timestep)

    external_scales = None
    if args.scales_from:
        with open(args.scales_from) as fh:
            external_scales = torch.tensor(json.load(fh)["per_world"][0]["scales"], device=args.device)

    worlds = [
        analyse_world(qpos[:, w], qvel[:, w], act[:, w], args.transient, dt,
                      args.min_period, args.max_period, args.tolerance,
                      mechanical_only=args.mechanical_only, external_scales=external_scales)
        for w in range(args.worlds)
    ]

    for w, r in enumerate(worlds):
        print(f"world {w}: period {r['period_steps']:3d} steps ({r['period_seconds']:.3f} s)   "
              f"residual {r['residual']:.4f}   advance {r['advance_per_cycle_m']:+.3f} m/cycle   "
              f"cycle velocity {r['cycle_velocity_mps']:+.2f} m/s   "
              f"trace velocity {r['trace_velocity_mps']:+.2f} m/s")

    summary = {
        "cfg": cfg_path,
        "checkpoint": args.checkpoint,
        "steps": args.steps,
        "transient": args.transient,
        "worlds": args.worlds,
        "stochastic_init": bool(args.stochastic_init),
        "control_dt": dt,
        "period_search": {"min": args.min_period, "max": args.max_period, "tolerance": args.tolerance},
        "mechanical_only": bool(args.mechanical_only),
        "scales_from": args.scales_from,
        "per_world": worlds,
        "median_period_steps": int(np.median([r["period_steps"] for r in worlds])),
        "median_residual": float(np.median([r["residual"] for r in worlds])),
        "median_cycle_velocity_mps": float(np.median([r["cycle_velocity_mps"] for r in worlds])),
    }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        npz = os.path.splitext(args.out)[0] + "_trace.npz"
        np.savez_compressed(
            npz,
            qpos=qpos.cpu().numpy(), qvel=qvel.cpu().numpy(),
            act=act.cpu().numpy(), actions=actions.cpu().numpy(),
        )
        print(f"wrote {args.out} and {npz}")


if __name__ == "__main__":
    main()
