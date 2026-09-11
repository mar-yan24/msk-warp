"""Export a qualifying trajectory-optimisation candidate as a per-phase reference gait.

Phase 4 found a forward periodic gait for the unmodified muscle hopper at a 16-step cycle
(+2.355 m/s, residual 0.3174, confirmed hopping). The orbit is open-loop unstable -- it survives
three cycles and fails at four, as does its motor control and the trained policy's own gait -- so
realising it needs feedback. This writes it out as a reference a tracking policy can be trained
against.

The exported reference is the cycle the optimiser actually solved for: ``qpos[p]``, ``qvel[p]`` and
``act[p]`` are the state *before* control ``ctrl[p]`` is applied, with phase 0 the optimised initial
state. Storing the state and the control together lets a tracking reward score either.

The file is written under ``msk_warp/assets/references/`` rather than to ``logs/`` because it stops
being a result at this point and becomes an input to training, so it needs to be versioned.

Usage::

    .venv/Scripts/python.exe scripts/export_reference_gait.py \\
        --cfg configs/hopper_muscle_shac.yaml --result logs/phase4/muscle_T16.json \\
        --out msk_warp/assets/references/hopper_muscle_T16_gait.npz
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def _load_trajopt():
    path = Path(__file__).resolve().parent / "trajopt_hopper.py"
    spec = importlib.util.spec_from_file_location("trajopt_hopper_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def roll_cycle(trajopt, env, u_logits, z, cycle, n_act_state, device):
    """One cycle, recording the pre-step state at each phase and the control applied."""
    qpos, qvel, act = trajopt.initial_state(z, n_act_state, device)
    states, controls = [], []
    for phase in range(cycle):
        actions = torch.tanh(u_logits[:, phase])
        ctrl = env._to_ctrl(actions)
        states.append((qpos[0].cpu().numpy(), qvel[0].cpu().numpy(), act[0].cpu().numpy()))
        controls.append(ctrl[0].cpu().numpy())
        qpos, qvel, act = trajopt.WarpSimStep.apply(ctrl, qpos, qvel, act, env)
        qpos = qpos.clamp(-100.0, 100.0)
        qvel = qvel.clamp(-100.0, 100.0)
    closing = (qpos[0].cpu().numpy(), qvel[0].cpu().numpy(), act[0].cpu().numpy())
    return states, controls, closing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--world", type=int, default=None, help="default: the best qualifying world")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="docs/research/phase4-capability/gait_motor_seed2.json",
                    help="stage-0 ruler, stored in the asset so a tracking env is self-contained")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    trajopt = _load_trajopt()
    with open(args.result) as fh:
        result = json.load(fh)
    params = np.load(os.path.splitext(args.result)[0] + "_params.npz")

    world = args.world
    if world is None:
        qualifying = [t for t in result["verified"]["top"] if t.get("qualifies")]
        if not qualifying:
            raise SystemExit("no qualifying candidate in this result; pass --world to force one")
        world = int(qualifying[0]["world"])

    cycle = int(result["cycle"])
    cfg, _ = trajopt.load_cfg(args.cfg)
    env = trajopt.build_env(cfg, 1, args.device, cycle * 4)
    n_act_state = env.warp_data.act.shape[1] if env.has_act else 0

    with open(args.scales) as fh:
        scales = np.array(json.load(fh)["per_world"][0]["scales"], dtype=np.float32)
    expected_dim = (env.mjm.nq - 1) + env.mjm.nv
    if scales.shape[0] != expected_dim:
        raise SystemExit(f"--scales has {scales.shape[0]} components, expected {expected_dim}")

    u = torch.tensor(params["u_logits"][world: world + 1], device=args.device)
    z = torch.tensor(params["z"][world: world + 1], device=args.device)
    states, controls, closing = roll_cycle(trajopt, env, u, z, cycle, n_act_state, args.device)

    qpos = np.stack([s[0] for s in states])
    qvel = np.stack([s[1] for s in states])
    act = np.stack([s[2] for s in states])
    ctrl = np.stack(controls)
    dt = env.substeps * float(env.mjm.opt.timestep)
    advance = float(closing[0][0] - qpos[0, 0])

    # The gate was measured on the verified open-loop replay, so the export must reproduce it.
    matched = [t for t in result["verified"]["top"] if int(t["world"]) == world]
    expected = float(matched[0]["verified_velocity"]) if matched else float("nan")
    velocity = advance / (cycle * dt)

    print(f"world {world}, cycle {cycle} steps ({cycle * dt:.3f} s)")
    print(f"  advance {advance:+.4f} m  ->  {velocity:+.4f} m/s over one cycle")
    print(f"  verified velocity in the result: {expected:+.4f} m/s over 3 cycles")
    print(f"  height range over the cycle: {qpos[:, 1].max() - qpos[:, 1].min():.4f} m")
    print(f"  activation range: {act.min():.3f} to {act.max():.3f}")
    print(f"  closing error, shape components: "
          f"qpos {np.abs(closing[0][1:] - qpos[0, 1:]).max():.4f}  "
          f"qvel {np.abs(closing[1] - qvel[0]).max():.4f}  "
          f"act {np.abs(closing[2] - act[0]).max():.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        qpos=qpos, qvel=qvel, act=act, ctrl=ctrl,
        closing_qpos=closing[0], closing_qvel=closing[1], closing_act=closing[2],
        shape_scales=scales,
        cycle=np.array(cycle), control_dt=np.array(dt),
        advance_per_cycle=np.array(advance), velocity=np.array(velocity),
        source=np.array(f"{args.result}#world{world}"),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
