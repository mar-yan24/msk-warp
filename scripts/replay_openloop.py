"""Replay a recorded gait's own actions open loop, to separate "is it a gait" from "is it stable".

Phase 4's verification asks a candidate orbit to survive ten consecutive open-loop cycles. Zero of
256 motor candidates did, despite closing their cycle to a residual of 0.076 against a real gait's
0.234. Before concluding anything about the optimiser, this asks whether the criterion is even
achievable: it takes the trained motor policy's *own* trajectory -- 3.83 m/s, 0% falls over 16
episodes, the thing G4.1 is supposed to rediscover -- and replays its recorded actions without
feedback.

Two modes, and the difference between them is the whole point:

``sequence``  replay the recorded actions in order from the recorded state. The simulator is
              deterministic, so this must reproduce the recorded trajectory. It is the control on
              the control: if this diverges, the harness is wrong, not the physics.
``cycled``    replay one period's worth of actions over and over. This is exactly what the Phase 4
              verification does to a candidate. If the real gait cannot survive it either, the
              criterion is testing open-loop stability rather than periodicity, and no candidate of
              any kind can pass.

Usage::

    .venv/Scripts/python.exe scripts/replay_openloop.py \\
        --cfg configs/hopper_motor_shac.yaml \\
        --trace docs/research/phase4-capability/gait_motor_seed2_trace.npz \\
        --period 27 --start 150 --cycles 10
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

TERMINATION_HEIGHT = -0.45


def _load_trajopt():
    path = Path(__file__).resolve().parent / "trajopt_hopper.py"
    spec = importlib.util.spec_from_file_location("trajopt_hopper_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def replay(trajopt, env, qpos0, qvel0, act0, actions, device):
    """Drive the model with a fixed action sequence and record height and x."""
    qpos = torch.tensor(qpos0, device=device).unsqueeze(0)
    qvel = torch.tensor(qvel0, device=device).unsqueeze(0)
    act = torch.tensor(act0, device=device).unsqueeze(0)
    heights, xs = [], []
    for step in range(actions.shape[0]):
        a = torch.tensor(actions[step], device=device).unsqueeze(0)
        ctrl = env._to_ctrl(a)
        qpos, qvel, act = trajopt.WarpSimStep.apply(ctrl, qpos, qvel, act, env)
        qpos = qpos.clamp(-100.0, 100.0)
        qvel = qvel.clamp(-100.0, 100.0)
        heights.append(float(qpos[0, 1]))
        xs.append(float(qpos[0, 0]))
    return np.array(heights), np.array(xs)


def survival(heights):
    """Index of the first step below the termination height, or the full length."""
    fallen = np.nonzero(heights < TERMINATION_HEIGHT)[0]
    return int(fallen[0]) if fallen.size else len(heights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--trace", required=True, help="npz from scripts/extract_gait_cycle.py")
    ap.add_argument("--period", type=int, required=True)
    ap.add_argument("--start", type=int, default=150, help="step in the trace to begin from")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--world", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trajopt = _load_trajopt()
    data = np.load(args.trace)
    qpos, qvel, act, actions = (data[k][:, args.world] for k in ("qpos", "qvel", "act", "actions"))

    total = args.period * args.cycles
    if args.start + total > len(qpos):
        raise SystemExit(f"trace has {len(qpos)} steps; need {args.start + total}")

    cfg, _ = trajopt.load_cfg(args.cfg)
    env = trajopt.build_env(cfg, 1, args.device, total + 2)

    # The recorded state at `start` is the state *after* the action at `start`, so the replay must
    # begin from the previous step's state to line the two up.
    s = args.start - 1
    start_state = (qpos[s], qvel[s], act[s])

    recorded_height = qpos[args.start: args.start + total, 1]
    recorded_x = qpos[args.start: args.start + total, 0]

    sequence_actions = actions[args.start: args.start + total]
    cycled_actions = np.tile(actions[args.start: args.start + args.period], (args.cycles, 1))

    results = {}
    for name, seq in (("sequence", sequence_actions), ("cycled", cycled_actions)):
        heights, xs = replay(trajopt, env, *start_state, seq, args.device)
        steps = survival(heights)
        results[name] = {
            "steps_survived": steps,
            "cycles_survived": steps / args.period,
            "fell": steps < len(heights),
            "distance_m": float(xs[min(steps, len(xs)) - 1] - xs[0]),
            "max_height_error_vs_recorded": float(
                np.abs(heights[:steps] - recorded_height[:steps]).max()) if steps else 0.0,
        }

    dt = env.substeps * float(env.mjm.opt.timestep)
    print(f"trained motor gait, period {args.period}, replayed for {args.cycles} cycles "
          f"({total} steps, {total * dt:.1f} s)")
    print(f"  recorded (with feedback):  {total} steps, {recorded_x[-1] - recorded_x[0]:+.2f} m, no fall")
    for name, r in results.items():
        verdict = "FELL" if r["fell"] else "survived"
        print(f"  {name:9s} (open loop): {r['steps_survived']:4d} steps "
              f"({r['cycles_survived']:.1f} cycles), {r['distance_m']:+.2f} m, {verdict}"
              f"   max height error vs recorded {r['max_height_error_vs_recorded']:.4f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"period": args.period, "start": args.start, "cycles": args.cycles,
                       "results": results}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
