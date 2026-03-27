"""Visualize a trained SHAC policy in the MuJoCo viewer.

Config-driven: reads env name, model path, action strength, substeps, etc.
from the training config YAML saved alongside the checkpoint.

Usage:
    python scripts/visualize.py --cfg logs/ant2/cfg.yaml --policy logs/ant2/best_policy.pt
    python scripts/visualize.py --cfg logs/ant2/cfg.yaml --policy logs/ant2/best_policy.pt --no-render
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import mujoco
import mujoco.viewer
import yaml

from msk_warp import resolve_model_path
from msk_warp.utils.running_mean_std import RunningMeanStd
import msk_warp.utils.torch_utils as tu


def load_policy(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    actor = checkpoint[0].to(device)
    obs_rms = checkpoint[3]
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    return actor, obs_rms


# ---------------------------------------------------------------------------
# Env adapters -- lightweight per-env logic for native MuJoCo visualization.
# No Warp imports; uses torch_utils (pure torch) for quaternion math.
# ---------------------------------------------------------------------------

class CartPoleAdapter:
    def __init__(self, env_cfg, device):
        self.action_strength = env_cfg.get('action_strength', 20.0)
        self.substeps = env_cfg.get('substeps', 4)
        self.episode_length = env_cfg.get('episode_length', 240)
        self.device = device

    def compute_obs(self, qpos, qvel):
        x, theta = qpos[0], qpos[1]
        xdot, theta_dot = qvel[0], qvel[1]
        return torch.tensor(
            [[x, xdot, np.sin(theta), np.cos(theta), theta_dot]],
            dtype=torch.float32, device=self.device,
        )

    def compute_reward(self, obs, action_np):
        x = obs[0, 0].item()
        xdot = obs[0, 1].item()
        sin_t = obs[0, 2].item()
        cos_t = obs[0, 3].item()
        theta_dot = obs[0, 4].item()
        theta = math.atan2(sin_t, cos_t)
        action_val = action_np[0]
        reward = -(theta**2 * 1.0 + theta_dot**2 * 0.1
                   + x**2 * 0.5 + xdot**2 * 0.1
                   + action_val**2 * 0.01)
        boundary = max(abs(x) - 3.0, 0.0) ** 2
        reward -= boundary * 50.0
        return reward

    def reset_state(self, mjm, mjd):
        mujoco.mj_resetData(mjm, mjd)
        mjd.qpos[0] = 0.0
        mjd.qpos[1] = math.pi
        mjd.qvel[:] = 0.0
        mjd.qpos[0] += (np.random.rand() - 0.5) * 1.0
        mjd.qpos[1] += (np.random.rand() - 0.5) * math.pi
        mjd.qvel[0] += (np.random.rand() - 0.5) * 0.5
        mjd.qvel[1] += (np.random.rand() - 0.5) * 0.5
        mujoco.mj_forward(mjm, mjd)

    def check_done(self, obs):
        return False

    def update_actions(self, action_np):
        pass

    def camera_setup(self, viewer):
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -15.0
        viewer.cam.lookat[:] = [0, 0, 0.5]


class AntAdapter:
    def __init__(self, env_cfg, device):
        self.action_strength = env_cfg.get('action_strength', 1.0)
        self.substeps = env_cfg.get('substeps', 16)
        self.episode_length = env_cfg.get('episode_length', 1000)
        self.early_termination = env_cfg.get('early_termination', True)
        self.termination_height = 0.27
        self.vel_scale = 0.1
        self.device = device

        self.targets = torch.tensor([[10000.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        self.up_vec = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        self.heading_vec = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        self.last_actions = torch.zeros(1, 8, device=device, dtype=torch.float32)

        self.start_joint_q = np.array([0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])

    def compute_obs(self, qpos, qvel):
        """Compute 37D ant observation. Replicates AntEnv._compute_obs logic."""
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

    def compute_reward(self, obs, action_np):
        forward_vel = obs[0, 5].item()
        up_reward = 0.1 * obs[0, 27].item()
        heading_reward = obs[0, 28].item()
        height_reward = obs[0, 0].item() - 0.27
        return forward_vel + up_reward + heading_reward + height_reward

    def reset_state(self, mjm, mjd):
        mujoco.mj_resetData(mjm, mjd)
        mjd.qpos[0] = 0.0
        mjd.qpos[1] = 0.0
        mjd.qpos[2] = 0.75
        mjd.qpos[3] = 1.0
        mjd.qpos[4:7] = 0.0
        mjd.qpos[7:15] = self.start_joint_q
        mjd.qvel[:] = 0.0

        # Stochastic perturbation (matches training)
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

    def update_actions(self, action_np):
        self.last_actions = torch.tensor(
            action_np, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

    def camera_setup(self, viewer):
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = [0, 0, 0.5]


ADAPTER_MAP = {
    'CartPoleSwingUp': CartPoleAdapter,
    'Ant': AntAdapter,
}


def main():
    parser = argparse.ArgumentParser(description='Visualize trained SHAC policy')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to training config YAML (e.g. logs/ant2/cfg.yaml)')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to policy checkpoint (e.g. logs/ant2/best_policy.pt)')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-render', action='store_true',
                        help='Run without viewer, print stats only')
    parser.add_argument('--save-frames', type=str, default=None,
                        help='Directory to save rendered frames')
    args = parser.parse_args()

    # Load config
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg['params']['env']
    env_name = env_cfg['name']

    # Resolve paths
    policy_path = os.path.abspath(args.policy)
    model_path = resolve_model_path(env_cfg['model_path'])

    print(f"Env: {env_name}")
    print(f"Policy: {policy_path}")
    print(f"Model: {model_path}")

    # Create adapter
    if env_name not in ADAPTER_MAP:
        raise ValueError(
            f"No visualization adapter for env '{env_name}'. "
            f"Available: {list(ADAPTER_MAP.keys())}"
        )
    adapter = ADAPTER_MAP[env_name](env_cfg, args.device)

    # Load policy
    actor, obs_rms = load_policy(policy_path, device=args.device)
    actor.eval()

    # Load native MuJoCo model (strip Warp-only flags)
    xml_string = open(model_path).read()
    xml_vis = xml_string.replace('jacobian="sparse"', '')
    xml_vis = xml_vis.replace(
        '<flag contact="disable" constraint="disable" eulerdamp="disable"/>', ''
    )
    mjm = mujoco.MjModel.from_xml_string(xml_vis)
    mjd = mujoco.MjData(mjm)

    episode_rewards = []
    episode_lengths = []

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        renderer = mujoco.Renderer(mjm, width=640, height=480)

    def run_step():
        """Run one control step: obs -> policy -> physics -> reward."""
        # Obs for policy (uses previous actions)
        obs = adapter.compute_obs(mjd.qpos.copy(), mjd.qvel.copy())
        obs_norm = obs_rms.normalize(obs) if obs_rms is not None else obs

        with torch.no_grad():
            action = actor(obs_norm, deterministic=True)
            action = torch.tanh(action)

        action_np = action.squeeze(0).cpu().numpy()
        adapter.update_actions(action_np)

        ctrl = action_np * adapter.action_strength
        mjd.ctrl[:len(ctrl)] = ctrl

        for _ in range(adapter.substeps):
            mujoco.mj_step(mjm, mjd)

        # Post-step obs for reward and termination
        obs_after = adapter.compute_obs(mjd.qpos.copy(), mjd.qvel.copy())
        reward = adapter.compute_reward(obs_after, action_np)
        done = adapter.check_done(obs_after)
        return reward, done

    if args.no_render or args.save_frames:
        for ep in range(args.episodes):
            adapter.reset_state(mjm, mjd)
            total_reward = 0.0
            steps = 0
            for t in range(adapter.episode_length):
                reward, done = run_step()
                total_reward += reward
                steps += 1
                if args.save_frames:
                    renderer.update_scene(mjd)
                    frame = renderer.render()
                    frame_path = os.path.join(
                        args.save_frames, f"ep{ep:02d}_frame{t:04d}.png"
                    )
                    import PIL.Image
                    PIL.Image.fromarray(frame).save(frame_path)
                if done:
                    break
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episode {ep+1}: reward = {total_reward:.2f}, length = {steps}")

        print(f"\nMean reward: {np.mean(episode_rewards):.2f} "
              f"+/- {np.std(episode_rewards):.2f}")
        print(f"Mean length: {np.mean(episode_lengths):.1f}")
    else:
        # Interactive viewer
        print("Launching MuJoCo viewer...")

        ep_idx = 0
        step_in_ep = 0
        total_reward = 0.0

        adapter.reset_state(mjm, mjd)

        with mujoco.viewer.launch_passive(mjm, mjd) as viewer:
            adapter.camera_setup(viewer)

            while viewer.is_running() and ep_idx < args.episodes:
                step_start = time.time()

                reward, done = run_step()
                total_reward += reward
                step_in_ep += 1

                viewer.sync()

                # Real-time playback
                dt = mjm.opt.timestep * adapter.substeps
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

                # Episode end
                if done or step_in_ep >= adapter.episode_length:
                    episode_rewards.append(total_reward)
                    episode_lengths.append(step_in_ep)
                    reason = "terminated" if done else "completed"
                    print(f"Episode {ep_idx+1}: reward = {total_reward:.2f}, "
                          f"length = {step_in_ep} ({reason})")
                    ep_idx += 1
                    step_in_ep = 0
                    total_reward = 0.0

                    if ep_idx < args.episodes:
                        adapter.reset_state(mjm, mjd)

            if episode_rewards:
                print(f"\nMean reward: {np.mean(episode_rewards):.2f} "
                      f"+/- {np.std(episode_rewards):.2f}")
                print(f"Mean length: {np.mean(episode_lengths):.1f}")


if __name__ == '__main__':
    main()
