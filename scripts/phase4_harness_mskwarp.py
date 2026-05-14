"""Phase 4 cross-engine harness — msk-warp side.

Runs N envs forward for T steps from a controlled initial state with a fixed
action sequence. Logs per-step per-env height/qpos/qvel/reward/terminated.

Saves output as NPZ. Paired with phase4_harness_diffrl.py.

Run from msk-warp .venv:
    python scripts/phase4_harness_mskwarp.py \
        --output tests/data/phase4_mskwarp_traj.npz \
        --num-envs 64 --num-steps 1000 --seed 42 \
        --action-std 0.3 --action-mode gaussian

Initial pose: standing (z=0.75), identity quat, canonical joints, qvel=0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from msk_warp.envs.ant import AntEnv  # noqa: E402

DEVICE = "cuda:0"
SUBSTEPS = 4
ACTION_STRENGTH = 1.0
TERMINATION_HEIGHT = 0.27


def build_env(num_envs: int, model_path: str):
    return AntEnv(
        num_envs=num_envs,
        device=DEVICE,
        episode_length=10000,  # long; we'll handle termination ourselves
        no_grad=True,
        stochastic_init=False,  # we set state explicitly
        substeps=SUBSTEPS,
        model_path=model_path,
        action_strength=ACTION_STRENGTH,
        early_termination=False,  # never auto-reset; we want raw trajectories
        tape_per_substep=True,
    )


def set_standing_pose(env, num_envs: int, init_noise: float = 0.0, seed: int = 0):
    """Standing init: torso at z=0.75, identity quat, canonical joints, qvel=0.

    If init_noise > 0, perturb starting state by that scale:
      qvel linear: ±init_noise (m/s) on x,y; ±init_noise/2 on z
      qvel angular: ±init_noise (rad/s) on all axes
      qpos joints: ±0.1*init_noise (rad)
    """
    qpos = env.start_qpos.clone()
    qvel = torch.zeros((num_envs, env.num_joint_qd), device=DEVICE, dtype=torch.float32)
    if init_noise > 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        v = (torch.rand((num_envs, env.num_joint_qd), generator=g) - 0.5).to(DEVICE) * 2.0
        # qvel layout (msk-warp): [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z, joint(8)]
        qvel[:, 0] = v[:, 0] * init_noise
        qvel[:, 1] = v[:, 1] * init_noise
        qvel[:, 2] = v[:, 2] * init_noise * 0.5
        qvel[:, 3:6] = v[:, 3:6] * init_noise
        qvel[:, 6:] = v[:, 6:] * init_noise * 0.5
        jq = (torch.rand((num_envs, 8), generator=g) - 0.5).to(DEVICE) * 2.0
        qpos[:, 7:15] = qpos[:, 7:15] + jq * init_noise * 0.1
    wp.copy(env.warp_data.qpos, wp.from_torch(qpos.contiguous()))
    wp.copy(env.warp_data.qvel, wp.from_torch(qvel.contiguous()))
    wp.synchronize()


def make_action_sequence(num_steps: int, num_envs: int, num_actions: int,
                         seed: int, mode: str, std: float) -> np.ndarray:
    """Generate (T, N, A) action sequence on CPU. Reproducible across engines.

    Modes:
        gaussian: N(0, std) for each (t, n, a). Will be clipped to [-1, 1] before applying.
        zero: all zeros (sanity baseline).
        adversarial_hip: alternating ±0.8 on hip joints (0, 2, 4, 6); ankles 0.
    """
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
    parser.add_argument("--model", type=str, default=str(REPO_ROOT / "msk_warp" / "assets" / "ant_soft.xml"))
    parser.add_argument("--init-noise", type=float, default=0.0,
                        help="If > 0, perturb starting state by that scale (qvel ~ N(0,init_noise)).")
    args = parser.parse_args()

    env = build_env(args.num_envs, args.model)
    set_standing_pose(env, args.num_envs, init_noise=args.init_noise, seed=args.seed)

    actions_np = make_action_sequence(args.num_steps, args.num_envs, env.num_actions,
                                       args.seed, args.action_mode, args.action_std)

    T, N, A = actions_np.shape
    heights = np.zeros((T, N), dtype=np.float32)
    rewards = np.zeros((T, N), dtype=np.float32)
    fell_step = np.full((N,), -1, dtype=np.int32)  # first step where height < 0.27
    # Storage for full qpos/qvel only at a subset of steps to save space
    save_steps = np.arange(0, T, max(1, T // 50))  # ~50 snapshots
    qpos_snap = np.zeros((len(save_steps), N, env.num_joint_q), dtype=np.float32)
    qvel_snap = np.zeros((len(save_steps), N, env.num_joint_qd), dtype=np.float32)

    print(f"[mskwarp-harness] running T={T} N={N} mode={args.action_mode} std={args.action_std}")
    snap_idx = 0
    with torch.no_grad():
        for t in range(T):
            a = torch.from_numpy(actions_np[t]).to(DEVICE)
            obs, rew, done, extras, _qpos_out, _qvel_out = env.step(a)
            qpos = wp.to_torch(env.warp_data.qpos).clone()  # (N, 15)
            qvel = wp.to_torch(env.warp_data.qvel).clone()  # (N, 14)
            heights[t] = qpos[:, 2].detach().cpu().numpy()
            rewards[t] = rew.detach().cpu().numpy()
            # Track fall events (only set on first fall per env)
            below = heights[t] < TERMINATION_HEIGHT
            fresh_falls = below & (fell_step < 0)
            fell_step[fresh_falls] = t
            if t in save_steps:
                qpos_snap[snap_idx] = qpos.detach().cpu().numpy()
                qvel_snap[snap_idx] = qvel.detach().cpu().numpy()
                snap_idx += 1
            if t % 100 == 0:
                fallen = int((fell_step >= 0).sum())
                mean_h = float(heights[t].mean())
                print(f"  step={t:4d}  mean_h={mean_h:.3f}  cumulative_fallen={fallen}/{N}")

    fallen = int((fell_step >= 0).sum())
    mean_fall_step = float(fell_step[fell_step >= 0].mean()) if fallen > 0 else -1.0
    print(f"\n[mskwarp-harness] DONE  fallen={fallen}/{N} ({100*fallen/N:.1f}%)  "
          f"mean_fall_step={mean_fall_step:.1f}  final_mean_h={float(heights[-1].mean()):.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_json = json.dumps({
        "engine": "mskwarp",
        "model": args.model,
        "num_envs": N, "num_steps": T,
        "seed": args.seed,
        "action_std": args.action_std, "action_mode": args.action_mode,
        "substeps": SUBSTEPS, "action_strength": ACTION_STRENGTH,
        "termination_height": TERMINATION_HEIGHT,
    })
    np.savez_compressed(
        out_path,
        engine="mskwarp",
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
