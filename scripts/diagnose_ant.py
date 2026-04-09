"""Diagnose ant locomotion training issues.

Runs episodes with detailed per-step reward breakdowns, episode statistics,
and optional gradient diagnostics.

Usage:
    # Analyze a trained policy
    python scripts/diagnose_ant.py --cfg logs/ant/cfg.yaml --policy logs/ant/best_policy.pt

    # Analyze with random actions (no policy needed)
    python scripts/diagnose_ant.py --cfg logs/ant/cfg.yaml --random

    # Include gradient diagnostics (requires CUDA + Warp)
    python scripts/diagnose_ant.py --cfg logs/ant/cfg.yaml --policy logs/ant/best_policy.pt --grad

    # More episodes for better statistics
    python scripts/diagnose_ant.py --cfg logs/ant/cfg.yaml --policy logs/ant/best_policy.pt --episodes 50
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import mujoco
import yaml

from msk_warp import resolve_model_path
from msk_warp.utils.running_mean_std import RunningMeanStd
import msk_warp.utils.torch_utils as tu


def load_policy(path, device='cpu'):
    """Load actor and obs_rms from a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    actor = checkpoint[0].to(device)
    obs_rms = checkpoint[3]
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    return actor, obs_rms


class AntDiagAdapter:
    """Lightweight ant adapter for native MuJoCo rollouts (no Warp needed)."""

    def __init__(self, env_cfg, device='cpu'):
        self.action_strength = env_cfg.get('action_strength', 1.0)
        self.substeps = env_cfg.get('substeps', 16)
        self.episode_length = env_cfg.get('episode_length', 1000)
        self.early_termination = env_cfg.get('early_termination', True)
        self.termination_height = 0.27
        self.vel_scale = 0.1
        self.action_penalty = env_cfg.get('action_penalty', 0.0)
        self.forward_vel_weight = env_cfg.get('forward_vel_weight', 1.0)
        self.heading_weight = env_cfg.get('heading_weight', 1.0)
        self.up_weight = env_cfg.get('up_weight', 0.1)
        self.height_weight = env_cfg.get('height_weight', 1.0)
        self.joint_vel_penalty = env_cfg.get('joint_vel_penalty', 0.0)
        self.push_reward_weight = env_cfg.get('push_reward_weight', 0.0)
        self.device = device

        self.targets = torch.tensor([[10000.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        self.up_vec = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        self.heading_vec = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        self.last_actions = torch.zeros(1, 8, device=device, dtype=torch.float32)

        self.start_joint_q = np.array([0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])

    def compute_obs(self, qpos, qvel):
        """Compute 37D ant observation (matches AntEnv._compute_obs)."""
        qpos_t = torch.tensor(qpos, dtype=torch.float32, device=self.device).unsqueeze(0)
        qvel_t = torch.tensor(qvel, dtype=torch.float32, device=self.device).unsqueeze(0)

        torso_pos = qpos_t[:, 0:3]
        torso_quat = qpos_t[:, 3:7]
        joint_q = qpos_t[:, 7:15]

        lin_vel = qvel_t[:, 0:3]
        ang_vel = qvel_t[:, 3:6]
        joint_vel = qvel_t[:, 6:14]

        height = torso_pos[:, 2:3]

        to_target = self.targets - torso_pos
        to_target = torch.cat([to_target[:, 0:1], to_target[:, 1:2],
                               torch.zeros_like(to_target[:, 2:3])], dim=-1)
        target_dirs = tu.normalize(to_target)

        up_proj = tu.quat_rotate(torso_quat, self.up_vec)
        heading_proj = tu.quat_rotate(torso_quat, self.heading_vec)

        obs = torch.cat([
            height,                                                     # 0
            torso_quat,                                                 # 1:5
            lin_vel,                                                    # 5:8
            ang_vel,                                                    # 8:11
            joint_q,                                                    # 11:19
            joint_vel * self.vel_scale,                                 # 19:27
            up_proj[:, 2:3],                                            # 27
            (heading_proj * target_dirs).sum(dim=-1, keepdim=True),     # 28
            self.last_actions,                                          # 29:37
        ], dim=-1)
        return obs

    def decompose_reward(self, obs, action_np):
        """Return weighted reward components matching AntEnv._compute_reward."""
        forward_vel_raw = obs[0, 5].item()
        up_raw = obs[0, 27].item()
        heading_raw = obs[0, 28].item()
        height_raw = obs[0, 0].item()
        joint_vel = (obs[0, 19:27] * 10.0).detach().cpu().numpy()

        action_arr = np.asarray(action_np, dtype=np.float32)

        forward_vel = self.forward_vel_weight * forward_vel_raw
        up_reward = self.up_weight * up_raw
        heading_reward = self.heading_weight * heading_raw
        height_reward = self.height_weight * (height_raw - 0.27)
        action_cost = self.action_penalty * float(np.sum(action_arr ** 2))
        joint_vel_cost = -self.joint_vel_penalty * float(np.sum(joint_vel ** 2))
        push_reward = 0.0
        if self.push_reward_weight != 0.0 and action_arr.shape[0] >= 7:
            push = (-action_arr[0] - action_arr[2] + action_arr[4] + action_arr[6])
            push_reward = self.push_reward_weight * float(push)

        total = (
            forward_vel
            + up_reward
            + heading_reward
            + height_reward
            + action_cost
            + joint_vel_cost
            + push_reward
        )
        return {
            'forward_vel': forward_vel,
            'up_reward': up_reward,
            'heading_reward': heading_reward,
            'height_reward': height_reward,
            'action_cost': action_cost,
            'joint_vel_cost': joint_vel_cost,
            'push_reward': push_reward,
            'total': total,
        }

    def reset_state(self, mjm, mjd):
        """Reset ant to standing pose with stochastic perturbation."""
        mujoco.mj_resetData(mjm, mjd)
        mjd.qpos[0] = 0.0
        mjd.qpos[1] = 0.0
        mjd.qpos[2] = 0.75
        mjd.qpos[3] = 1.0
        mjd.qpos[4:7] = 0.0
        mjd.qpos[7:15] = self.start_joint_q
        mjd.qvel[:] = 0.0

        # Stochastic perturbation (matches training reset)
        mjd.qpos[0:3] += 0.1 * (np.random.rand(3) - 0.5) * 2.0
        mjd.qpos[7:15] += 0.2 * (np.random.rand(8) - 0.5) * 2.0
        mjd.qvel[:] = 0.5 * (np.random.rand(mjd.qvel.shape[0]) - 0.5)

        # Small orientation perturbation
        angle = (np.random.rand() - 0.5) * (math.pi / 12.0)
        axis = np.random.rand(3) - 0.5
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        half = angle / 2.0
        rand_quat = np.array([np.cos(half), *(axis * np.sin(half))])
        w1, x1, y1, z1 = mjd.qpos[3], mjd.qpos[4], mjd.qpos[5], mjd.qpos[6]
        w2, x2, y2, z2 = rand_quat
        mjd.qpos[3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        mjd.qpos[4] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        mjd.qpos[5] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        mjd.qpos[6] = w1*z2 + x1*y2 - y1*x2 + z1*w2

        self.last_actions.zero_()
        mujoco.mj_forward(mjm, mjd)

    def check_done(self, obs):
        if not self.early_termination:
            return False
        return obs[0, 0].item() < self.termination_height


def run_rollout_diagnostics(args, cfg):
    """Run N episodes and report detailed reward/episode statistics."""
    env_cfg = cfg['params']['env']
    adapter = AntDiagAdapter(env_cfg, device=args.device)

    # Load native MuJoCo model (strip Warp-only flags)
    model_path = resolve_model_path(env_cfg['model_path'])
    xml_string = open(model_path).read()
    xml_vis = xml_string.replace('jacobian="sparse"', '')
    xml_vis = xml_vis.replace(
        '<flag contact="disable" constraint="disable" eulerdamp="disable"/>', ''
    )
    mjm = mujoco.MjModel.from_xml_string(xml_vis)
    mjd = mujoco.MjData(mjm)

    # Load policy (or use random)
    actor, obs_rms = None, None
    if not args.random:
        actor, obs_rms = load_policy(args.policy, device=args.device)
        actor.eval()

    # Storage
    all_episode_rewards = []
    all_episode_lengths = []
    all_termination_reasons = []
    all_final_x = []
    all_max_fwd_vel = []
    all_step_components = {
        'forward_vel': [], 'up_reward': [], 'heading_reward': [],
        'height_reward': [], 'action_cost': [], 'joint_vel_cost': [],
        'push_reward': [], 'total': [],
    }

    for ep in range(args.episodes):
        adapter.reset_state(mjm, mjd)
        start_x = mjd.qpos[0]
        ep_reward = 0.0
        ep_max_fwd_vel = 0.0
        steps = 0

        for t in range(adapter.episode_length):
            # Compute observation
            obs = adapter.compute_obs(mjd.qpos.copy(), mjd.qvel.copy())

            # Get action
            if actor is not None:
                obs_norm = obs_rms.normalize(obs) if obs_rms is not None else obs
                with torch.no_grad():
                    action = actor(obs_norm, deterministic=True)
                    action = torch.tanh(action)
                action_np = action.squeeze(0).cpu().numpy()
            else:
                action_np = np.random.uniform(-1, 1, 8).astype(np.float32)

            adapter.last_actions = torch.tensor(
                action_np, dtype=torch.float32, device=args.device
            ).unsqueeze(0)

            # Apply control
            ctrl = action_np * adapter.action_strength
            mjd.ctrl[:len(ctrl)] = ctrl

            # Step physics
            for _ in range(adapter.substeps):
                mujoco.mj_step(mjm, mjd)

            # Post-step observation and reward
            obs_after = adapter.compute_obs(mjd.qpos.copy(), mjd.qvel.copy())
            components = adapter.decompose_reward(obs_after, action_np)

            ep_reward += components['total']
            ep_max_fwd_vel = max(ep_max_fwd_vel, components['forward_vel'])
            steps += 1

            # Accumulate per-step stats
            for key in all_step_components:
                all_step_components[key].append(components[key])

            # Check termination
            done = adapter.check_done(obs_after)
            if done:
                break

        # Episode summary
        reason = 'fell' if done else 'timeout'
        final_x = mjd.qpos[0] - start_x
        all_episode_rewards.append(ep_reward)
        all_episode_lengths.append(steps)
        all_termination_reasons.append(reason)
        all_final_x.append(final_x)
        all_max_fwd_vel.append(ep_max_fwd_vel)

        print(f"  Episode {ep+1:3d}: reward={ep_reward:8.1f}  len={steps:4d}  "
              f"x_disp={final_x:+6.2f}m  max_fwd_vel={ep_max_fwd_vel:5.2f}  ({reason})")

    # Summary
    ep_rewards = np.array(all_episode_rewards)
    ep_lengths = np.array(all_episode_lengths)
    final_xs = np.array(all_final_x)
    max_fvs = np.array(all_max_fwd_vel)
    fell_pct = sum(1 for r in all_termination_reasons if r == 'fell') / len(all_termination_reasons) * 100
    timeout_pct = 100.0 - fell_pct

    print()
    print("=" * 70)
    print("ANT DIAGNOSTIC REPORT")
    print("=" * 70)
    print()
    policy_desc = f"Policy: {args.policy}" if not args.random else "Policy: RANDOM"
    print(f"  {policy_desc}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Early termination: {env_cfg.get('early_termination', True)}")
    print()

    print("--- Episode Statistics ---")
    print(f"  Mean episode length:    {ep_lengths.mean():7.1f} +/- {ep_lengths.std():6.1f}")
    print(f"  Mean total reward:      {ep_rewards.mean():7.1f} +/- {ep_rewards.std():6.1f}")
    print(f"  Termination:            {fell_pct:.0f}% fell, {100-fell_pct:.0f}% timeout")
    print(f"  Mean final x-disp:      {final_xs.mean():+7.2f} +/- {final_xs.std():5.2f} m")
    print(f"  Mean max forward_vel:   {max_fvs.mean():7.2f} +/- {max_fvs.std():5.2f}")
    print()

    print("--- Reward Component Means (per step) ---")
    component_means = {
        key: float(np.array(all_step_components[key]).mean())
        for key in all_step_components
    }
    component_stds = {
        key: float(np.array(all_step_components[key]).std())
        for key in all_step_components
    }
    for key in ['forward_vel', 'up_reward', 'heading_reward', 'height_reward', 'action_cost', 'joint_vel_cost', 'push_reward', 'total']:
        print(f"  {key:<18s} {component_means[key]:+8.4f} +/- {component_stds[key]:7.4f}")
    print()

    # Auto-diagnosis
    print("--- Diagnosis ---")
    issues = []
    fwd_vel_mean = np.array(all_step_components['forward_vel']).mean()
    if abs(fwd_vel_mean) < 0.1:
        issues.append("STANDING STILL: Mean forward_vel is near zero ({:.4f}). "
                       "Policy produces no locomotion.".format(fwd_vel_mean))
    if fell_pct > 50 and ep_lengths.mean() < 500:
        issues.append("INSTABILITY: {:.0f}% of episodes end in falls (mean length {:.0f}). "
                       "Policy attempts to move but can't stay upright.".format(fell_pct, ep_lengths.mean()))
    # Estimate a standing-local-optimum baseline from non-locomotion terms on
    # the current policy rollouts. This avoids hardcoding torso-height priors.
    standing_step = (
        component_means['up_reward']
        + component_means['heading_reward']
        + component_means['height_reward']
    )
    standing_baseline = standing_step * float(ep_lengths.mean())
    if standing_baseline > 1e-6:
        if abs(ep_rewards.mean() - standing_baseline) / standing_baseline < 0.15:
            issues.append(
                "LOCAL OPTIMUM: Total reward ({:.0f}) is within 15% of standing-still "
                "baseline (~{:.0f}). Policy is stuck.".format(ep_rewards.mean(), standing_baseline)
            )
    if np.array(all_step_components['heading_reward']).mean() < 0.16:
        issues.append("ORIENTATION: Mean heading_reward ({:.2f}) suggests ant is not "
                       "facing forward consistently.".format(
                           np.array(all_step_components['heading_reward']).mean()))
    if max_fvs.mean() < 0.3:
        issues.append("NO VELOCITY: Mean max forward velocity ({:.2f}) is very low. "
                       "The ant never achieves meaningful forward movement.".format(max_fvs.mean()))

    if issues:
        for issue in issues:
            print(f"  [!] {issue}")
    else:
        print("  [OK] No obvious issues detected. Policy appears to be walking.")
    print()

    summary = {
        'policy': None if args.random else args.policy,
        'episodes': int(args.episodes),
        'episode_length_mean': float(ep_lengths.mean()),
        'episode_length_std': float(ep_lengths.std()),
        'reward_mean': float(ep_rewards.mean()),
        'reward_std': float(ep_rewards.std()),
        'final_x_disp_mean': float(final_xs.mean()),
        'final_x_disp_std': float(final_xs.std()),
        'max_forward_vel_mean': float(max_fvs.mean()),
        'max_forward_vel_std': float(max_fvs.std()),
        'forward_vel_mean': float(np.array(all_step_components['forward_vel']).mean()),
        'fall_rate': float(fell_pct / 100.0),
        'timeout_rate': float(timeout_pct / 100.0),
        'standing_baseline_reward': float(standing_baseline),
        'issue_count': int(len(issues)),
        'issues': issues,
        'component_means': component_means,
        'component_stds': component_stds,
        'reward_weights': {
            'forward_vel_weight': float(adapter.forward_vel_weight),
            'heading_weight': float(adapter.heading_weight),
            'up_weight': float(adapter.up_weight),
            'height_weight': float(adapter.height_weight),
            'action_penalty': float(adapter.action_penalty),
            'joint_vel_penalty': float(adapter.joint_vel_penalty),
            'push_reward_weight': float(adapter.push_reward_weight),
        },
    }

    return summary


def run_gradient_diagnostics(args, cfg):
    """Run gradient diagnostics via WarpSimStep (requires CUDA + Warp)."""
    import warnings
    warnings.filterwarnings('ignore')

    import warp as wp
    wp.init()

    import mujoco_warp as mjw
    from msk_warp.bridge import WarpSimStep

    print("=" * 70)
    print("GRADIENT DIAGNOSTICS")
    print("=" * 70)
    print()

    device = args.device

    # --- Setup minimal env ---
    model_path = resolve_model_path(cfg['params']['env']['model_path'])
    mjm_native = mujoco.MjModel.from_xml_path(model_path)
    substeps = cfg['params']['env'].get('substeps', 16)

    class MinimalEnv:
        def __init__(self, use_fd_jacobian=False, tape_per_substep=False):
            self.device = device
            self.substeps = substeps
            self.nworld = 1
            self._njmax = 512
            self.use_fd_jacobian = use_fd_jacobian
            self.tape_per_substep = tape_per_substep
            self.warp_model = mjw.put_model(mjm_native)

        def set_state(self, qpos_np, qvel_np):
            self.warp_data = mjw.make_diff_data(
                mjm_native, nworld=1, njmax=self._njmax,
            )
            mjw.reset_data(self.warp_model, self.warp_data)
            wp.copy(self.warp_data.qpos, wp.array(qpos_np.astype(np.float32), dtype=wp.float32))
            wp.copy(self.warp_data.qvel, wp.array(qvel_np.astype(np.float32), dtype=wp.float32))
            wp.synchronize()

    # Standing state
    nq, nv = 15, 14
    qpos0 = np.zeros((1, nq), dtype=np.float64)
    qpos0[0, :] = [0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    qvel0 = np.zeros((1, nv), dtype=np.float64)

    np.random.seed(42)
    ctrl_val = np.random.uniform(-0.3, 0.3, (1, 8))

    # --- Test 1: d(forward_vel)/d(ctrl) via tape-all ---
    print("Test 1: d(forward_vel)/d(ctrl) via tape-all backward")
    print("-" * 50)
    env = MinimalEnv(use_fd_jacobian=False, tape_per_substep=False)
    env.set_state(qpos0, qvel0)

    ctrl_torch = torch.tensor(ctrl_val, dtype=torch.float32, device=device, requires_grad=True)
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().requires_grad_(True)

    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos_in, qvel_in, env)
    forward_vel = qvel_out[:, 0]
    forward_vel.sum().backward()

    g_ctrl = ctrl_torch.grad.cpu().numpy().flatten()
    print(f"  d(fwd_vel)/d(ctrl): {g_ctrl}")
    print(f"  |grad|:             {np.linalg.norm(g_ctrl):.6e}")
    print(f"  max|g_i|:           {np.abs(g_ctrl).max():.6e}")
    print(f"  NaN:                {np.any(np.isnan(g_ctrl))}")
    status1 = "NON-ZERO" if np.abs(g_ctrl).max() > 1e-8 else "ZERO (BROKEN!)"
    print(f"  Status:             {status1}")
    print()

    # --- Test 2: d(total_reward)/d(ctrl) via tape-all ---
    print("Test 2: d(total_reward)/d(ctrl) via tape-all backward")
    print("-" * 50)
    env2 = MinimalEnv(use_fd_jacobian=False, tape_per_substep=False)
    env2.set_state(qpos0, qvel0)

    ctrl_torch2 = torch.tensor(ctrl_val, dtype=torch.float32, device=device, requires_grad=True)
    qpos_in2 = wp.to_torch(env2.warp_data.qpos).clone().requires_grad_(True)
    qvel_in2 = wp.to_torch(env2.warp_data.qvel).clone().requires_grad_(True)

    qpos_out2, qvel_out2 = WarpSimStep.apply(ctrl_torch2, qpos_in2, qvel_in2, env2)

    # Compute full reward from obs (matching AntEnv._compute_reward)
    from msk_warp.envs.ant import AntEnv
    targets = torch.tensor([[10000.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    up_vec = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    heading_vec = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    actions = torch.tensor(ctrl_val, dtype=torch.float32, device=device)

    obs = AntEnv._compute_obs(qpos_out2, qvel_out2, actions, targets, up_vec, heading_vec, 0.1)
    reward = AntEnv._compute_reward(obs, actions, -0.001)
    reward.sum().backward()

    g_rew = ctrl_torch2.grad.cpu().numpy().flatten()
    print(f"  d(reward)/d(ctrl):  {g_rew}")
    print(f"  |grad|:             {np.linalg.norm(g_rew):.6e}")
    print(f"  max|g_i|:           {np.abs(g_rew).max():.6e}")
    print(f"  NaN:                {np.any(np.isnan(g_rew))}")
    status2 = "NON-ZERO" if np.abs(g_rew).max() > 1e-8 else "ZERO (BROKEN!)"
    print(f"  Status:             {status2}")
    print()

    # --- Test 3: d(forward_vel)/d(ctrl) via native MuJoCo FD (ground truth) ---
    print("Test 3: d(forward_vel)/d(ctrl) via native MuJoCo float64 FD (ground truth)")
    print("-" * 50)
    mjm_fd = mujoco.MjModel.from_xml_path(model_path)
    mjd_fd = mujoco.MjData(mjm_fd)

    eps = 1e-6
    fd_grad = np.zeros(8)
    for j in range(8):
        vals = []
        for sign in [1, -1]:
            mujoco.mj_resetData(mjm_fd, mjd_fd)
            mjd_fd.qpos[:] = qpos0.flatten()
            mjd_fd.qvel[:] = qvel0.flatten()
            mjd_fd.ctrl[:] = ctrl_val.flatten()
            mjd_fd.ctrl[j] += sign * eps
            for _ in range(substeps):
                mujoco.mj_step(mjm_fd, mjd_fd)
            vals.append(mjd_fd.qvel[0])
        fd_grad[j] = (vals[0] - vals[1]) / (2 * eps)

    print(f"  d(fwd_vel)/d(ctrl): {fd_grad}")
    print(f"  |grad|:             {np.linalg.norm(fd_grad):.6e}")
    print(f"  max|g_i|:           {np.abs(fd_grad).max():.6e}")
    print()

    # --- Comparison ---
    print("=" * 70)
    print("GRADIENT COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<22s} {'|grad|':<14s} {'max|g_i|':<14s} {'Status'}")
    print("-" * 70)
    for name, g in [("tape-all AD", g_ctrl), ("native MuJoCo FD", fd_grad)]:
        norm = np.linalg.norm(g)
        maxg = np.abs(g).max()
        status = "OK" if maxg > 1e-6 else "ZERO"
        if np.any(np.isnan(g)):
            status = "NaN"
        print(f"  {name:<20s} {norm:<14.6e} {maxg:<14.6e} {status}")

    nonzero = np.abs(fd_grad) > 1e-8
    if nonzero.any():
        rel_err = np.abs(g_ctrl[nonzero] - fd_grad[nonzero]) / (np.abs(fd_grad[nonzero]) + 1e-10)
        print(f"\n  Relative error (tape-all vs native FD): "
              f"max={rel_err.max():.4f}, mean={rel_err.mean():.4f}")

    print()

    # --- Test 4: Policy-in-loop gradient (if policy provided) ---
    if not args.random and args.policy:
        print("Test 4: d(reward)/d(actor_params) with policy in loop")
        print("-" * 50)
        actor, obs_rms = load_policy(args.policy, device=device)
        actor.train()  # Need gradients

        env4 = MinimalEnv(use_fd_jacobian=False, tape_per_substep=False)
        env4.set_state(qpos0, qvel0)

        qpos_in4 = wp.to_torch(env4.warp_data.qpos).clone()
        qvel_in4 = wp.to_torch(env4.warp_data.qvel).clone()

        # Compute obs for policy
        obs4 = AntEnv._compute_obs(qpos_in4, qvel_in4, torch.zeros(1, 8, device=device),
                                    targets, up_vec, heading_vec, 0.1)
        if obs_rms is not None:
            obs4_norm = obs_rms.normalize(obs4)
        else:
            obs4_norm = obs4

        actions4 = actor(obs4_norm, deterministic=True)
        ctrl4 = torch.tanh(actions4)

        qpos_out4, qvel_out4 = WarpSimStep.apply(ctrl4, qpos_in4, qvel_in4, env4)

        obs_after4 = AntEnv._compute_obs(qpos_out4, qvel_out4, ctrl4,
                                          targets, up_vec, heading_vec, 0.1)
        reward4 = AntEnv._compute_reward(obs_after4, ctrl4, -0.001)
        reward4.sum().backward()

        total_grad_norm = 0.0
        num_params = 0
        for p in actor.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
                num_params += p.numel()
        total_grad_norm = total_grad_norm ** 0.5

        print(f"  |d(reward)/d(actor_params)|: {total_grad_norm:.6e}")
        print(f"  Num parameters:              {num_params}")
        status4 = "FLOWS" if total_grad_norm > 1e-8 else "BROKEN"
        print(f"  Gradient through physics:    {status4}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Diagnose ant locomotion training')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to training config YAML')
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to policy checkpoint')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of rollout episodes (default: 20)')
    parser.add_argument('--grad', action='store_true',
                        help='Run gradient diagnostics (requires CUDA + Warp)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for policy inference (default: cpu)')
    parser.add_argument('--random', action='store_true',
                        help='Use random actions instead of a policy')
    parser.add_argument('--json-out', type=str, default=None,
                        help='Optional path to write a JSON summary report')
    args = parser.parse_args()

    if not args.random and args.policy is None:
        parser.error("--policy is required unless --random is specified")

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print("ANT LOCOMOTION DIAGNOSTIC TOOL")
    print("=" * 70)
    print()

    # --- Rollout diagnostics (always runs) ---
    print("--- Running rollout episodes ---")
    rollout_summary = run_rollout_diagnostics(args, cfg)

    # --- Gradient diagnostics (optional, needs CUDA) ---
    if args.grad:
        if args.device == 'cpu':
            print("NOTE: Switching to cuda:0 for gradient diagnostics")
            args.device = 'cuda:0'
        run_gradient_diagnostics(args, cfg)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = {
            'cfg': args.cfg,
            'policy': None if args.random else args.policy,
            'rollout': rollout_summary,
            'grad_enabled': bool(args.grad),
        }
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON report to: {args.json_out}")

    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
