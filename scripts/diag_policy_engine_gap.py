"""Roll a saved ant policy through CPU MuJoCo and through mujoco_warp and compare behaviour.

Backend-agnostic on purpose: it calls ``mujoco_warp`` directly (``put_model``/``make_data``/``step``)
and borrows only ``AntEnv._compute_obs``, so the same script runs on any mujoco_warp checkout and
isolates engine differences from the environment wiring.

    .venv/Scripts/python.exe scripts/diag_policy_engine_gap.py --policy ~/checkpoints/<ckpt>.pt
    .venv-fork/Scripts/python.exe scripts/diag_policy_engine_gap.py --policy ...   # older backend
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from msk_warp import resolve_model_path
from msk_warp.envs.ant import AntEnv

START_QPOS = np.array([0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1], dtype=np.float64)
TERMINATION_HEIGHT = 0.27


def _obs(qpos, qvel, actions, device):
    qp = torch.tensor(np.asarray(qpos)[None], dtype=torch.float32, device=device)
    qv = torch.tensor(np.asarray(qvel)[None], dtype=torch.float32, device=device)
    targets = torch.tensor([[10000.0, 0.0, 0.0]], device=device)
    up = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    heading = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    return AntEnv._compute_obs(qp, qv, actions, targets, up, heading, 0.1)


def rollout_native(mjm, actor, obs_rms, steps, substeps, device):
    d = mujoco.MjData(mjm)
    d.qpos[:] = START_QPOS
    actions = torch.zeros(1, 8, device=device)
    x0, t = d.qpos[0], 0
    with torch.no_grad():
        for t in range(steps):
            obs = _obs(d.qpos, d.qvel, actions, device)
            actions = torch.tanh(actor(obs_rms.normalize(obs) if obs_rms is not None else obs, deterministic=True))
            d.ctrl[:] = actions.cpu().numpy()[0]
            for _ in range(substeps):
                mujoco.mj_step(mjm, d)
            if d.qpos[2] < TERMINATION_HEIGHT:
                break
    return float(d.qpos[0] - x0), t + 1, float(d.qpos[2])


def rollout_warp(mjm, actor, obs_rms, steps, substeps, device, njmax=512):
    import mujoco_warp as mjw
    import warp as wp

    wp.init()
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=1, njmax=njmax)
    mjw.reset_data(m, d)
    wp.copy(d.qpos, wp.array(START_QPOS[None].astype(np.float32), dtype=wp.float32))
    wp.synchronize()
    actions = torch.zeros(1, 8, device=device)
    x0, t = float(d.qpos.numpy()[0][0]), 0
    with torch.no_grad():
        for t in range(steps):
            qpos, qvel = d.qpos.numpy()[0], d.qvel.numpy()[0]
            obs = _obs(qpos, qvel, actions, device)
            actions = torch.tanh(actor(obs_rms.normalize(obs) if obs_rms is not None else obs, deterministic=True))
            wp.copy(d.ctrl, wp.array(actions.cpu().numpy().astype(np.float32), dtype=wp.float32))
            for _ in range(substeps):
                mjw.step(m, d)
            wp.synchronize()
            if d.qpos.numpy()[0][2] < TERMINATION_HEIGHT:
                break
    q = d.qpos.numpy()[0]
    return float(q[0] - x0), t + 1, float(q[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--model", default="assets/ant_soft.xml")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-warp", action="store_true")
    args = ap.parse_args()

    checkpoint = torch.load(os.path.expanduser(args.policy), map_location=args.device, weights_only=False)
    actor = checkpoint[0].to(args.device)
    actor.eval()
    obs_rms = checkpoint[3].to(args.device) if checkpoint[3] is not None else None
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path(args.model))

    import mujoco_warp
    import warp

    print(f"policy    {Path(args.policy).name}")
    print(f"model     {args.model}  substeps={args.substeps}  steps={args.steps}")
    print(f"versions  mujoco {mujoco.__version__} | warp {warp.__version__} | mujoco_warp {Path(mujoco_warp.__file__).parents[1]}")
    x, t, z = rollout_native(mjm, actor, obs_rms, args.steps, args.substeps, args.device)
    print(f"native    x_disp={x:+8.2f} m  steps={t:4d}  final_z={z:.3f}")
    if not args.skip_warp:
        x, t, z = rollout_warp(mjm, actor, obs_rms, args.steps, args.substeps, args.device)
        print(f"warp      x_disp={x:+8.2f} m  steps={t:4d}  final_z={z:.3f}")


if __name__ == "__main__":
    main()
