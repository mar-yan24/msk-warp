"""Inspect one trajectory-optimisation candidate: is it actually hopping?

The Phase 4 gates ask for a periodic orbit that advances and survives ten cycles. Three behaviours
satisfy the arithmetic without being a gait, and each has a signature this script exposes:

* **sliding** -- the pose barely changes while the body translates. Periodic by definition,
  nonzero velocity, no flight phase. Signature: torso height range near zero.
* **ballistic drift** -- a long unbroken flight. Cannot close in ``qvel`` under gravity, so the
  residual should already exclude it, but the flight fraction makes it obvious if it slips through.
* **a genuine hop** -- height rises and falls once or twice per cycle, the foot leaves the ground
  and returns, and the pattern repeats cycle after cycle.

The residual and the survival count cannot tell these apart. Nothing should be claimed about a
candidate without running this on it.

Usage::

    .venv/Scripts/python.exe scripts/inspect_trajopt_candidate.py \\
        --cfg configs/hopper_motor_shac.yaml --result logs/phase4/motor_T27.json --cycles 4
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

# Thresholds calibrated against the Phase 3 references rather than guessed
# (docs/research/phase4-capability/results.md):
#   real motor gait   height range 0.214 m   advance drift 0.022   advance +1.80 m/cycle
#   standing          height range 0.146 m                         advance -0.007 m over 300 steps
#   diving            height range 0.175 m
# Standing bobs 0.146 m, so height range alone cannot separate standing from hopping; advance does.
# The sliding threshold is a tripwire rather than a fitted value: nothing measured so far sits
# below it, so it has never been exercised against a real slide.
SLIDING_HEIGHT_RANGE = 0.05   # m
STANDING_ADVANCE = 0.05       # m per cycle
DRIFT_TOLERANCE = 0.25        # 11x the real gait's 0.022


def _load_trajopt():
    path = Path(__file__).resolve().parent / "trajopt_hopper.py"
    spec = importlib.util.spec_from_file_location("trajopt_hopper_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def trace(trajopt, env, u_logits, z, cycle, n_act_state, device, cycles):
    """Roll one candidate open loop and record the full state trace."""
    qpos, qvel, act = trajopt.initial_state(z, n_act_state, device)
    rows = []
    for c in range(cycles):
        for t in range(cycle):
            ctrl = env._to_ctrl(torch.tanh(u_logits[:, t]))
            qpos, qvel, act = trajopt.WarpSimStep.apply(ctrl, qpos, qvel, act, env)
            qpos = qpos.clamp(-100.0, 100.0)
            qvel = qvel.clamp(-100.0, 100.0)
            rows.append(torch.cat([qpos[0], qvel[0]]).cpu().numpy())
    return np.stack(rows)


def classify(states, cycle, dt):
    """Height range, flight fraction and per-cycle advance -- enough to name the behaviour."""
    height = states[:, 1]
    x = states[:, 0]
    cycles = len(states) // cycle
    advances = [float(x[(c + 1) * cycle - 1] - x[c * cycle - 1 if c else 0]) for c in range(cycles)]
    return {
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "height_range": float(height.max() - height.min()),
        "mean_velocity": float((x[-1] - x[0]) / ((len(x) - 1) * dt)),
        "advance_per_cycle": advances,
        "advance_drift": float(np.std(advances) / (abs(np.mean(advances)) + 1e-9)),
        "fell": bool((height < -0.45).any()),
    }


def verdict(summary):
    mean_advance = float(np.mean(summary["advance_per_cycle"]))
    if summary["fell"]:
        return "FELL -- not a gait"
    if abs(mean_advance) < STANDING_ADVANCE:
        return "STANDING -- periodic but going nowhere; the Phase 3 basin"
    if summary["height_range"] < SLIDING_HEIGHT_RANGE:
        return "SLIDING -- the body translates but the torso never rises; not a hop"
    if summary["advance_drift"] > DRIFT_TOLERANCE:
        return "DRIFTING -- advance per cycle is not repeatable; not yet a limit cycle"
    return "HOPPING -- torso oscillates and advance per cycle is repeatable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--result", required=True, help="the run's JSON; its _params.npz is read beside it")
    ap.add_argument("--world", type=int, default=None, help="default: the best verified world")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trajopt = _load_trajopt()
    with open(args.result) as fh:
        result = json.load(fh)
    params = np.load(os.path.splitext(args.result)[0] + "_params.npz")

    world = args.world
    if world is None:
        top = result["verified"]["top"]
        if not top:
            raise SystemExit("no candidates in the result; pass --world explicitly")
        world = int(top[0]["world"])

    cfg, _ = trajopt.load_cfg(args.cfg)
    cycle = int(result["cycle"])
    env = trajopt.build_env(cfg, 1, args.device, cycle * (args.cycles + 2))
    n_act_state = env.warp_data.act.shape[1] if env.has_act else 0

    u = torch.tensor(params["u_logits"][world: world + 1], device=args.device)
    z = torch.tensor(params["z"][world: world + 1], device=args.device)
    dt = env.substeps * float(env.mjm.opt.timestep)

    states = trace(trajopt, env, u, z, cycle, n_act_state, args.device, args.cycles)
    summary = classify(states, cycle, dt)
    summary["world"] = world
    summary["verdict"] = verdict(summary)

    print(f"world {world}, {args.cycles} cycles of {cycle} steps")
    print(f"  height   min {summary['height_min']:+.3f}  max {summary['height_max']:+.3f}  "
          f"range {summary['height_range']:.3f} m")
    print(f"  velocity {summary['mean_velocity']:+.3f} m/s")
    print(f"  advance per cycle {[round(a, 3) for a in summary['advance_per_cycle']]}  "
          f"(drift {summary['advance_drift']:.3f})")
    print(f"  {summary['verdict']}")

    print("\n  step   height      vx     pitch   thigh     leg    foot")
    for i in range(0, len(states), max(1, cycle // 9)):
        s = states[i]
        print(f"  {i:4d}  {s[1]:+7.3f} {s[6]:+7.2f}  {s[2]:+7.3f} {s[3]:+7.3f} {s[4]:+7.3f} {s[5]:+7.3f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "states": states.round(5).tolist()}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
