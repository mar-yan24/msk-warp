"""Visualize a trained SHAC policy in the MuJoCo viewer."""

import argparse
import os
import math
import time

import numpy as np
import torch
import mujoco
import mujoco.viewer

from msk_warp import resolve_model_path
from msk_warp.utils.running_mean_std import RunningMeanStd


def load_policy(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    actor = checkpoint[0].to(device)
    obs_rms = checkpoint[3]
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    return actor, obs_rms


def compute_obs(qpos, qvel):
    """Compute observation vector from MuJoCo state."""
    x = qpos[0]
    theta = qpos[1]
    xdot = qvel[0]
    theta_dot = qvel[1]
    return np.array([x, xdot, np.sin(theta), np.cos(theta), theta_dot], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Visualize trained policy')
    parser.add_argument('--policy', type=str, default='logs/cartpole_v3/best_policy.pt')
    parser.add_argument('--model', type=str, default='assets/cartpole.xml')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--episode-length', type=int, default=240)
    parser.add_argument('--action-strength', type=float, default=20.0)
    parser.add_argument('--substeps', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-render', action='store_true', help='Run without viewer, print stats only')
    parser.add_argument('--save-frames', type=str, default=None, help='Directory to save rendered frames')
    args = parser.parse_args()

    # Resolve policy path relative to CWD
    policy_path = args.policy
    if not os.path.isabs(policy_path):
        policy_path = os.path.abspath(policy_path)

    # Resolve model path via package assets
    model_path = resolve_model_path(args.model)

    print(f"Loading policy from: {policy_path}")
    actor, obs_rms = load_policy(policy_path, device=args.device)
    actor.eval()

    # Load MuJoCo model (native, not Warp) for visualization
    # We need a version without the disabled flags for proper rendering
    xml_string = open(model_path).read()
    # Remove the sparse/disable flags for visualization (they affect rendering)
    xml_vis = xml_string.replace('jacobian="sparse"', '')
    xml_vis = xml_vis.replace('<flag contact="disable" constraint="disable" eulerdamp="disable"/>', '')
    mjm = mujoco.MjModel.from_xml_string(xml_vis)
    mjd = mujoco.MjData(mjm)

    # Stats
    episode_rewards = []
    episode_lengths = []

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        renderer = mujoco.Renderer(mjm, width=640, height=480)

    def run_episode(ep_idx):
        """Run a single episode and return total reward."""
        mujoco.mj_resetData(mjm, mjd)
        # Set initial state: pole hanging down
        mjd.qpos[0] = 0.0
        mjd.qpos[1] = math.pi
        mjd.qvel[:] = 0.0
        # Add stochastic perturbation (separated cart vs pole)
        mjd.qpos[0] += (np.random.rand() - 0.5) * 1.0      # cart: +-0.5m
        mjd.qpos[1] += (np.random.rand() - 0.5) * math.pi  # pole: +-pi/2
        mjd.qvel[0] += (np.random.rand() - 0.5) * 0.5
        mjd.qvel[1] += (np.random.rand() - 0.5) * 0.5
        mujoco.mj_forward(mjm, mjd)

        total_reward = 0.0
        step_count = 0

        for t in range(args.episode_length):
            obs = compute_obs(mjd.qpos, mjd.qvel)
            obs_torch = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)

            if obs_rms is not None:
                obs_torch = obs_rms.normalize(obs_torch)

            with torch.no_grad():
                action = actor(obs_torch, deterministic=True)
                action = torch.tanh(action)

            ctrl = action.squeeze(0).cpu().numpy() * args.action_strength
            mjd.ctrl[:] = ctrl

            for _ in range(args.substeps):
                mujoco.mj_step(mjm, mjd)

            # Compute reward (matching training weights)
            theta = np.arctan2(np.sin(mjd.qpos[1]), np.cos(mjd.qpos[1]))
            action_val = ctrl[0] / args.action_strength
            reward = -(theta**2 * 1.0
                       + mjd.qvel[1]**2 * 0.1
                       + mjd.qpos[0]**2 * 0.5
                       + mjd.qvel[0]**2 * 0.1
                       + action_val**2 * 0.01)
            boundary = max(abs(mjd.qpos[0]) - 3.0, 0.0) ** 2
            reward -= boundary * 50.0
            total_reward += reward
            step_count += 1

            if args.save_frames:
                renderer.update_scene(mjd)
                frame = renderer.render()
                frame_path = os.path.join(args.save_frames, f"ep{ep_idx:02d}_frame{t:04d}.png")
                import PIL.Image
                PIL.Image.fromarray(frame).save(frame_path)

        return total_reward, step_count

    if args.no_render or args.save_frames:
        for ep in range(args.episodes):
            reward, length = run_episode(ep)
            episode_rewards.append(reward)
            episode_lengths.append(length)
            print(f"Episode {ep+1}: reward = {reward:.2f}, length = {length}")

        print(f"\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
        print(f"Mean length: {np.mean(episode_lengths):.1f}")
    else:
        # Interactive viewer
        print("Launching MuJoCo viewer...")
        print("Press 'R' in viewer to reset episode")

        ep_idx = 0
        step_in_ep = 0
        total_reward = 0.0

        # Initialize
        mujoco.mj_resetData(mjm, mjd)
        mjd.qpos[0] = 0.0
        mjd.qpos[1] = math.pi
        mjd.qpos[0] += (np.random.rand() - 0.5) * 1.0
        mjd.qpos[1] += (np.random.rand() - 0.5) * math.pi
        mjd.qvel[0] += (np.random.rand() - 0.5) * 0.5
        mjd.qvel[1] += (np.random.rand() - 0.5) * 0.5
        mujoco.mj_forward(mjm, mjd)

        with mujoco.viewer.launch_passive(mjm, mjd) as viewer:
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -15.0
            viewer.cam.lookat[:] = [0, 0, 0.5]

            while viewer.is_running() and ep_idx < args.episodes:
                step_start = time.time()

                # Get observation and action
                obs = compute_obs(mjd.qpos, mjd.qvel)
                obs_torch = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                if obs_rms is not None:
                    obs_torch = obs_rms.normalize(obs_torch)

                with torch.no_grad():
                    action = actor(obs_torch, deterministic=True)
                    action = torch.tanh(action)

                ctrl = action.squeeze(0).cpu().numpy() * args.action_strength
                mjd.ctrl[:] = ctrl

                for _ in range(args.substeps):
                    mujoco.mj_step(mjm, mjd)

                # Reward (matching training weights)
                theta = np.arctan2(np.sin(mjd.qpos[1]), np.cos(mjd.qpos[1]))
                action_val = ctrl[0] / args.action_strength
                reward = -(theta**2 * 1.0 + mjd.qvel[1]**2 * 0.1
                           + mjd.qpos[0]**2 * 0.5 + mjd.qvel[0]**2 * 0.1
                           + action_val**2 * 0.01)
                boundary = max(abs(mjd.qpos[0]) - 3.0, 0.0) ** 2
                reward -= boundary * 50.0
                total_reward += reward
                step_in_ep += 1

                viewer.sync()

                # Timing for real-time playback
                dt = mjm.opt.timestep * args.substeps
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

                # Episode end
                if step_in_ep >= args.episode_length:
                    episode_rewards.append(total_reward)
                    episode_lengths.append(step_in_ep)
                    print(f"Episode {ep_idx+1}: reward = {total_reward:.2f}, length = {step_in_ep}")
                    ep_idx += 1
                    step_in_ep = 0
                    total_reward = 0.0

                    if ep_idx < args.episodes:
                        mujoco.mj_resetData(mjm, mjd)
                        mjd.qpos[0] = 0.0
                        mjd.qpos[1] = math.pi
                        mjd.qpos[0] += (np.random.rand() - 0.5) * 1.0
                        mjd.qpos[1] += (np.random.rand() - 0.5) * math.pi
                        mjd.qvel[0] += (np.random.rand() - 0.5) * 0.5
                        mjd.qvel[1] += (np.random.rand() - 0.5) * 0.5
                        mujoco.mj_forward(mjm, mjd)

            if episode_rewards:
                print(f"\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")


if __name__ == '__main__':
    main()
