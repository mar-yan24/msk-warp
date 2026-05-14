"""Phase 4 cross-engine harness — DiffRL/dflex side.

Mirrors phase4_harness_mskwarp.py. Same action protocol, same initial pose,
same output schema.

DiffRL uses y-up convention; the equivalent of msk-warp's standing pose
(qpos[2]=z=0.75) is DiffRL's qpos[1]=y=0.75 with start_rot = -π/2 around x.

Run from DiffRL .venv (Python 3.8 + cu118):
    python scripts/phase4_harness_diffrl.py \
        --output C:/projects/msk-warp/tests/data/phase4_diffrl_traj.npz \
        --num-envs 64 --num-steps 1000 --seed 42 \
        --action-std 0.3 --action-mode gaussian

Initial pose: standing (y=0.75), start_rot=quat(-π/2 around x), canonical joints, qvel=0.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# DiffRL repo (must be on sys.path before importing envs/dflex)
DIFFRL_ROOT = Path(r"C:\Projects\DiffRL")
sys.path.insert(0, str(DIFFRL_ROOT))

import numpy as np
import torch

import dflex as df  # noqa: F401
from envs.ant import AntEnv

DEVICE = "cuda:0"
TERMINATION_HEIGHT = 0.27


def build_env(num_envs: int):
    return AntEnv(
        render=False,
        device=DEVICE,
        num_envs=num_envs,
        seed=42,
        episode_length=10000,
        no_grad=True,
        stochastic_init=False,
        MM_caching_frequency=16,
        early_termination=False,  # we handle termination logging ourselves
    )


def set_standing_pose(env, num_envs: int):
    """Standing init in y-up: y=0.75, start_rot=quat(-π/2 around x), canonical joints."""
    env.reset()
    q = env.state.joint_q.detach().cpu().numpy().reshape(env.num_envs, -1).copy()
    qd = env.state.joint_qd.detach().cpu().numpy().reshape(env.num_envs, -1).copy()
    start_rot = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)
    canonical_joints = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
    for i in range(num_envs):
        q[i, 0] = 0.0
        q[i, 1] = 0.75  # height (y-up)
        q[i, 2] = 0.0
        q[i, 3:7] = start_rot
        q[i, 7:15] = canonical_joints
        qd[i, :] = 0.0
    env.state.joint_q = torch.from_numpy(q.flatten().astype(np.float32)).to(DEVICE)
    env.state.joint_qd = torch.from_numpy(qd.flatten().astype(np.float32)).to(DEVICE)


def make_action_sequence(num_steps: int, num_envs: int, num_actions: int,
                         seed: int, mode: str, std: float) -> np.ndarray:
    """Identical to msk-warp side. Generates on CPU for reproducibility."""
    rng = np.random.default_rng(seed)
    if mode == "gaussian":
        actions = rng.normal(0.0, std, size=(num_steps, num_envs, num_actions)).astype(np.float32)
    elif mode == "zero":
        actions = np.zeros((num_steps, num_envs, num_actions), dtype=np.float32)
    elif mode == "adversarial_hip":
        actions = np.zeros((num_steps, num_envs, num_actions), dtype=np.float32)
        for t in range(num_steps):
            sign = 1.0 if (t // 5) % 2 == 0 else -1.0
            actions[t, :, [0, 2, 4, 6]] = 0.8 * sign
    else:
        raise ValueError(f"unknown action mode {mode}")
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-std", type=float, default=0.3)
    parser.add_argument("--action-mode", type=str, default="gaussian",
                        choices=["gaussian", "zero", "adversarial_hip"])
    args = parser.parse_args()

    env = build_env(args.num_envs)
    set_standing_pose(env, args.num_envs)

    num_actions = env.num_actions
    num_joint_q = env.num_joint_q
    num_joint_qd = env.num_joint_qd
    actions_np = make_action_sequence(args.num_steps, args.num_envs, num_actions,
                                       args.seed, args.action_mode, args.action_std)
    T, N, A = actions_np.shape

    heights = np.zeros((T, N), dtype=np.float32)
    rewards = np.zeros((T, N), dtype=np.float32)
    fell_step = np.full((N,), -1, dtype=np.int32)
    save_steps = np.arange(0, T, max(1, T // 50))
    qpos_snap = np.zeros((len(save_steps), N, num_joint_q), dtype=np.float32)
    qvel_snap = np.zeros((len(save_steps), N, num_joint_qd), dtype=np.float32)

    print(f"[diffrl-harness] running T={T} N={N} mode={args.action_mode} std={args.action_std}")
    snap_idx = 0
    with torch.no_grad():
        for t in range(T):
            a = torch.from_numpy(actions_np[t]).to(DEVICE)
            obs, rew, done, extras = env.step(a)
            # DiffRL's obs[:, 0] is y-height (after start_rot transform, lib uses obs[:, 0]).
            # But the raw env.state.joint_q[:, 1] is the y position. Use the raw.
            q = env.state.joint_q.detach().cpu().numpy().reshape(N, num_joint_q)
            qd = env.state.joint_qd.detach().cpu().numpy().reshape(N, num_joint_qd)
            heights[t] = q[:, 1]  # y-axis is up in DiffRL ant
            rewards[t] = rew.detach().cpu().numpy()
            below = heights[t] < TERMINATION_HEIGHT
            fresh_falls = below & (fell_step < 0)
            fell_step[fresh_falls] = t
            if t in save_steps:
                qpos_snap[snap_idx] = q
                qvel_snap[snap_idx] = qd
                snap_idx += 1
            if t % 100 == 0:
                fallen = int((fell_step >= 0).sum())
                mean_h = float(heights[t].mean())
                print(f"  step={t:4d}  mean_h={mean_h:.3f}  cumulative_fallen={fallen}/{N}")

    fallen = int((fell_step >= 0).sum())
    mean_fall_step = float(fell_step[fell_step >= 0].mean()) if fallen > 0 else -1.0
    print(f"\n[diffrl-harness] DONE  fallen={fallen}/{N} ({100*fallen/N:.1f}%)  "
          f"mean_fall_step={mean_fall_step:.1f}  final_mean_h={float(heights[-1].mean()):.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    cfg_json = json.dumps({
        "engine": "diffrl",
        "num_envs": N, "num_steps": T,
        "seed": args.seed,
        "action_std": args.action_std, "action_mode": args.action_mode,
        "substeps": 16,
        "termination_height": TERMINATION_HEIGHT,
    })
    np.savez_compressed(
        out_path,
        engine="diffrl",
        heights=heights,
        rewards=rewards,
        fell_step=fell_step,
        qpos_snap=qpos_snap,
        qvel_snap=qvel_snap,
        save_steps=save_steps,
        actions=actions_np,
        config_json=cfg_json,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
