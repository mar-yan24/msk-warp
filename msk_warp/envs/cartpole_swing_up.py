"""CartPole Swing-Up environment using MuJoCo Warp."""

import math

import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from msk_warp.envs.base_env import MjWarpEnv
from msk_warp.bridge import WarpSimStep
from msk_warp.utils import torch_utils as tu


class CartPoleSwingUpEnv(MjWarpEnv):
    def __init__(
        self,
        num_envs=64,
        device='cuda:0',
        episode_length=240,
        no_grad=False,
        stochastic_init=True,
        substeps=4,
        model_path='assets/cartpole.xml',
        action_strength=20.0,
        **kwargs,
    ):
        num_obs = 5
        num_act = 1

        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=num_act,
            episode_length=episode_length,
            model_path=model_path,
            device=device,
            no_grad=no_grad,
            substeps=substeps,
        )

        self.stochastic_init = stochastic_init
        self.action_strength = action_strength

        # Reward weights (matching DiffRL)
        self.pole_angle_penalty = 1.0
        self.pole_velocity_penalty = 0.1
        self.cart_position_penalty = 0.05
        self.cart_velocity_penalty = 0.1
        self.cart_action_penalty = 0.0

        self.num_joint_q = 2  # slider x, hinge theta
        self.num_joint_qd = 2

        # Reward weights dict (reused each step)
        self._reward_weights = {
            'pole_angle': self.pole_angle_penalty,
            'pole_vel': self.pole_velocity_penalty,
            'cart_pos': self.cart_position_penalty,
            'cart_vel': self.cart_velocity_penalty,
            'action': self.cart_action_penalty,
        }

        # Save default start state (pole hanging down at theta=pi)
        self._save_start_state()

        # Initial reset
        self.reset()

    def _save_start_state(self):
        """Save the keyframe state as start state."""
        wp.synchronize()
        self.start_qpos = wp.to_torch(self.warp_data.qpos).clone()
        self.start_qvel = wp.to_torch(self.warp_data.qvel).clone()
        # Set pole to hanging down (theta = pi)
        self.start_qpos[:, 0] = 0.0
        self.start_qpos[:, 1] = math.pi
        self.start_qvel[:, :] = 0.0

    @staticmethod
    def _compute_obs(qpos, qvel):
        """Compute observations from qpos/qvel tensors (differentiable in PyTorch)."""
        x = qpos[:, 0:1]
        theta = qpos[:, 1:2]
        xdot = qvel[:, 0:1]
        theta_dot = qvel[:, 1:2]
        return torch.cat(
            [x, xdot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1
        )

    @staticmethod
    def _compute_reward(qpos, qvel, actions, weights):
        """Compute reward from qpos/qvel tensors (differentiable in PyTorch)."""
        x = qpos[:, 0]
        theta = tu.normalize_angle(qpos[:, 1])
        xdot = qvel[:, 0]
        theta_dot = qvel[:, 1]

        reward = (
            -torch.pow(theta, 2.0) * weights['pole_angle']
            - torch.pow(theta_dot, 2.0) * weights['pole_vel']
            - torch.pow(x, 2.0) * weights['cart_pos']
            - torch.pow(xdot, 2.0) * weights['cart_vel']
            - torch.sum(actions ** 2, dim=-1) * weights['action']
        )
        return reward

    def step(self, actions, qpos_in=None, qvel_in=None):
        """Run one control step.

        Args:
            actions: (num_envs, num_actions) action tensor (already tanh'd)
            qpos_in: Optional differentiable qpos input (for state gradient flow)
            qvel_in: Optional differentiable qvel input (for state gradient flow)

        Returns:
            obs, rew, done, extras, qpos_out, qvel_out
        """
        actions = actions.view(self.num_envs, self.num_actions)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions

        # Scale actions to ctrl
        ctrl = actions * self.action_strength

        if self.no_grad:
            # Non-differentiable path
            ctrl_wp = wp.from_torch(ctrl.detach().contiguous())
            wp.copy(self.warp_data.ctrl, ctrl_wp)
            for _ in range(self.substeps):
                mjw.step(self.warp_model, self.warp_data)
            wp.synchronize()

            qpos = wp.to_torch(self.warp_data.qpos)
            qvel = wp.to_torch(self.warp_data.qvel)
            self.obs_buf = self._compute_obs(qpos, qvel)
            self.rew_buf = self._compute_reward(qpos, qvel, actions, self._reward_weights)
            qpos_out, qvel_out = None, None
        else:
            # Differentiable path: state flows through WarpSimStep
            if qpos_in is None:
                qpos_in = wp.to_torch(self.warp_data.qpos).clone()
            if qvel_in is None:
                qvel_in = wp.to_torch(self.warp_data.qvel).clone()

            qpos_out, qvel_out = WarpSimStep.apply(ctrl, qpos_in, qvel_in, self)

            # Clamp state to prevent extreme values causing NaN gradients
            qpos_out = qpos_out.clamp(-20.0, 20.0)
            qvel_out = qvel_out.clamp(-50.0, 50.0)

            self.obs_buf = self._compute_obs(qpos_out, qvel_out)
            self.rew_buf = self._compute_reward(qpos_out, qvel_out, actions, self._reward_weights)

        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.progress_buf += 1

        # Save obs before reset for critic bootstrap
        if not self.no_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf,
            }

        # Check for episode end
        self.reset_buf = torch.where(
            self.progress_buf >= self.episode_length,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_warp_state(env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras, qpos_out, qvel_out

    def _reset_warp_state(self, env_ids):
        """Reset Warp state for specified environments (no gradient)."""
        with torch.no_grad():
            qpos_torch = wp.to_torch(self.warp_data.qpos)
            qvel_torch = wp.to_torch(self.warp_data.qvel)

            qpos_torch[env_ids, :] = self.start_qpos[env_ids, :].clone()
            qvel_torch[env_ids, :] = self.start_qvel[env_ids, :].clone()

            if self.stochastic_init:
                n = len(env_ids)
                qpos_torch[env_ids, 0] += 1.0 * (torch.rand(n, device=self.device) - 0.5)
                qpos_torch[env_ids, 1] += math.pi * (torch.rand(n, device=self.device) - 0.5)
                qvel_torch[env_ids, 0] += 0.5 * (torch.rand(n, device=self.device) - 0.5)
                qvel_torch[env_ids, 1] += 0.5 * (torch.rand(n, device=self.device) - 0.5)

        self.progress_buf[env_ids] = 0

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            self._reset_warp_state(env_ids)

            # Recompute obs for all worlds (non-differentiable, used for initialization)
            with torch.no_grad():
                qpos_view = wp.to_torch(self.warp_data.qpos)
                qvel_view = wp.to_torch(self.warp_data.qvel)
                self.obs_buf = self._compute_obs(qpos_view, qvel_view)

        return self.obs_buf

    def calculateObservations(self):
        """Non-differentiable obs computation (used by initialize_trajectory)."""
        wp.synchronize()
        qpos = wp.to_torch(self.warp_data.qpos)
        qvel = wp.to_torch(self.warp_data.qvel)
        self.obs_buf = self._compute_obs(qpos, qvel)

    def calculateReward(self):
        """Non-differentiable reward computation (unused in diff path)."""
        pass
