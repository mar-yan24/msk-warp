"""Ant locomotion environment using MuJoCo Warp.

Ported from DiffRL's envs/ant.py. Key translation from DiffRL:
  - Coordinate system: z-up (MuJoCo native) instead of y-up (dflex)
  - Quaternion convention: [w, x, y, z] (MuJoCo) instead of [x, y, z, w] (dflex)
  - qvel layout: [lin(3), ang(3), joints] (MuJoCo) instead of [ang(3), lin(3), joints] (dflex)
"""

import math

import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from msk_warp.envs.base_env import MjWarpEnv
from msk_warp.bridge import WarpSimStep
import msk_warp.utils.torch_utils as tu


class AntEnv(MjWarpEnv):
    def __init__(
        self,
        num_envs=64,
        device='cuda:0',
        episode_length=1000,
        no_grad=False,
        stochastic_init=True,
        substeps=16,
        model_path='assets/ant.xml',
        action_strength=1.0,
        early_termination=True,
        njmax=512,
        **kwargs,
    ):
        num_obs = 37
        num_act = 8

        super().__init__(
            num_envs=num_envs,
            num_obs=num_obs,
            num_act=num_act,
            episode_length=episode_length,
            model_path=model_path,
            device=device,
            no_grad=no_grad,
            substeps=substeps,
            njmax=njmax,
        )

        self.stochastic_init = stochastic_init
        self.action_strength = action_strength
        self.early_termination = early_termination

        self.termination_height = 0.27
        self.termination_height_max = 1.0
        self.joint_vel_obs_scaling = 0.1
        self.action_penalty = 0.0

        # DOF layout: free joint (7 qpos / 6 qvel) + 8 hinge joints
        self.num_joint_q = 15
        self.num_joint_qd = 14

        # Target direction (forward along +x in z-up world)
        self.targets = torch.tensor(
            [10000.0, 0.0, 0.0], device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(num_envs, -1)

        # Basis vectors (z-up)
        self.up_vec = torch.tensor(
            [0.0, 0.0, 1.0], device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(num_envs, -1)
        self.heading_vec = torch.tensor(
            [1.0, 0.0, 0.0], device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(num_envs, -1)

        # Default joint angles (from DiffRL)
        self.start_joint_q = torch.tensor(
            [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
            device=device, dtype=torch.float32,
        )

        self._save_start_state()
        self.reset()

    def _save_start_state(self):
        """Save the default standing state."""
        wp.synchronize()
        self.start_qpos = torch.zeros(
            (self.num_envs, self.num_joint_q), device=self.device, dtype=torch.float32,
        )
        self.start_qvel = torch.zeros(
            (self.num_envs, self.num_joint_qd), device=self.device, dtype=torch.float32,
        )

        # Torso position (z-up)
        self.start_qpos[:, 0] = 0.0   # x
        self.start_qpos[:, 1] = 0.0   # y
        self.start_qpos[:, 2] = 0.75  # z (height)

        # Torso quaternion [w, x, y, z] = identity (no rotation needed in z-up)
        self.start_qpos[:, 3] = 1.0
        self.start_qpos[:, 4] = 0.0
        self.start_qpos[:, 5] = 0.0
        self.start_qpos[:, 6] = 0.0

        # Joint angles (from DiffRL default pose)
        self.start_qpos[:, 7:15] = self.start_joint_q.unsqueeze(0)

    @staticmethod
    def _compute_obs(qpos, qvel, actions, targets, up_vec, heading_vec, vel_scale):
        """Compute 37D observation from state tensors (differentiable in PyTorch).

        All in z-up MuJoCo convention with [w,x,y,z] quaternions.
        """
        torso_pos = qpos[:, 0:3]
        torso_quat = qpos[:, 3:7]   # [w, x, y, z]
        joint_q = qpos[:, 7:15]     # 8 joint angles

        # MuJoCo qvel: [linear(3), angular(3), joints(8)]
        lin_vel = qvel[:, 0:3]
        ang_vel = qvel[:, 3:6]
        joint_vel = qvel[:, 6:14]

        # Height
        height = torso_pos[:, 2:3]

        # Direction to target (projected to ground plane)
        to_target = targets - torso_pos
        to_target = torch.cat([to_target[:, 0:1], to_target[:, 1:2],
                               torch.zeros_like(to_target[:, 2:3])], dim=-1)
        target_dirs = tu.normalize(to_target)

        # Rotate basis vectors by torso orientation
        up_proj = tu.quat_rotate(torso_quat, up_vec)
        heading_proj = tu.quat_rotate(torso_quat, heading_vec)

        obs = torch.cat([
            height,                                                          # 0
            torso_quat,                                                      # 1:5
            lin_vel,                                                         # 5:8
            ang_vel,                                                         # 8:11
            joint_q,                                                         # 11:19
            joint_vel * vel_scale,                                           # 19:27
            up_proj[:, 2:3],                                                 # 27
            (heading_proj * target_dirs).sum(dim=-1, keepdim=True),          # 28
            actions,                                                         # 29:37
        ], dim=-1)
        return obs

    @staticmethod
    def _compute_reward(obs, actions, action_penalty):
        """Compute reward from observation tensor (differentiable in PyTorch).

        Matches DiffRL's ant reward:
          forward_vel + 0.1 * up + heading + (height - 0.27) + action_penalty
        """
        height = obs[:, 0]
        forward_vel = obs[:, 5]        # lin_vel_x
        up_reward = 0.1 * obs[:, 27]   # up-vector z-component
        heading_reward = obs[:, 28]     # heading alignment
        height_reward = height - 0.27

        reward = (
            forward_vel
            + up_reward
            + heading_reward
            + height_reward
            + action_penalty * (actions ** 2).sum(dim=-1)
        )
        return reward

    def compute_obs(self, qpos, qvel):
        """Instance method wrapper for SHAC compatibility."""
        return self._compute_obs(
            qpos, qvel, self.actions,
            self.targets, self.up_vec, self.heading_vec,
            self.joint_vel_obs_scaling,
        )

    def step(self, actions, qpos_in=None, qvel_in=None):
        """Run one control step.

        Args:
            actions: (num_envs, 8) action tensor (already tanh'd)
            qpos_in: Optional differentiable qpos input (for state gradient flow)
            qvel_in: Optional differentiable qvel input (for state gradient flow)

        Returns:
            obs, rew, done, extras, qpos_out, qvel_out
        """
        actions = actions.view(self.num_envs, self.num_actions)
        actions = torch.clamp(actions, -1.0, 1.0)
        # Clone + detach stored actions to prevent cross-epoch graph references
        # and avoid in-place modification (by _reset_warp_state) of shared storage
        # that would invalidate the gradient-tracked actions tensor.
        self.actions = actions.detach().clone()

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
            self.obs_buf = self._compute_obs(
                qpos, qvel, actions,
                self.targets, self.up_vec, self.heading_vec,
                self.joint_vel_obs_scaling,
            )
            self.rew_buf = self._compute_reward(self.obs_buf, actions, self.action_penalty)
            qpos_out, qvel_out = None, None
        else:
            # Differentiable path: state flows through WarpSimStep
            if qpos_in is None:
                qpos_in = wp.to_torch(self.warp_data.qpos).clone()
            if qvel_in is None:
                qvel_in = wp.to_torch(self.warp_data.qvel).clone()

            qpos_out, qvel_out = WarpSimStep.apply(ctrl, qpos_in, qvel_in, self)

            # Clamp state to prevent extreme values
            qpos_out = qpos_out.clamp(-100.0, 100.0)
            qvel_out = qvel_out.clamp(-100.0, 100.0)

            # NaN-to-zero gradient hooks to prevent stray NaN from poisoning the batch
            if qpos_out.requires_grad:
                qpos_out.register_hook(lambda g: torch.nan_to_num(g, 0.0, 0.0, 0.0))
            if qvel_out.requires_grad:
                qvel_out.register_hook(lambda g: torch.nan_to_num(g, 0.0, 0.0, 0.0))

            self.obs_buf = self._compute_obs(
                qpos_out, qvel_out, actions,
                self.targets, self.up_vec, self.heading_vec,
                self.joint_vel_obs_scaling,
            )
            self.rew_buf = self._compute_reward(self.obs_buf, actions, self.action_penalty)

        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.progress_buf += 1

        # Early termination: height below threshold
        if self.early_termination:
            too_low = self.obs_buf[:, 0] < self.termination_height
            too_high = self.obs_buf[:, 0] > self.termination_height_max
            self.termination_buf = torch.where(
                too_low | too_high,
                torch.ones_like(self.termination_buf),
                torch.zeros_like(self.termination_buf),
            )
            self.reset_buf = torch.where(
                self.termination_buf > 0,
                torch.ones_like(self.reset_buf),
                self.reset_buf,
            )

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

            # Restore start state
            qpos_torch[env_ids, :] = self.start_qpos[env_ids, :].clone()
            qvel_torch[env_ids, :] = self.start_qvel[env_ids, :].clone()

            if self.stochastic_init:
                n = len(env_ids)
                # Perturb position (x, y, z)
                qpos_torch[env_ids, 0:3] += 0.1 * (
                    torch.rand(n, 3, device=self.device) - 0.5
                ) * 2.0

                # Perturb orientation with small random rotation
                angle = (torch.rand(n, device=self.device) - 0.5) * (math.pi / 12.0)
                axis = torch.nn.functional.normalize(
                    torch.rand(n, 3, device=self.device) - 0.5, dim=-1
                )
                rand_quat = tu.quat_from_angle_axis(angle, axis)
                cur_quat = qpos_torch[env_ids, 3:7].clone()
                qpos_torch[env_ids, 3:7] = tu.quat_mul(cur_quat, rand_quat)

                # Perturb joint angles
                qpos_torch[env_ids, 7:15] += 0.2 * (
                    torch.rand(n, 8, device=self.device) - 0.5
                ) * 2.0

                # Small random velocities
                qvel_torch[env_ids, :] = 0.5 * (
                    torch.rand(n, self.num_joint_qd, device=self.device) - 0.5
                )

            # Clear actions for reset envs
            self.actions[env_ids, :] = 0.0

        self.progress_buf[env_ids] = 0

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            self._reset_warp_state(env_ids)

            # Recompute obs (non-differentiable, for initialization)
            with torch.no_grad():
                qpos_view = wp.to_torch(self.warp_data.qpos)
                qvel_view = wp.to_torch(self.warp_data.qvel)
                self.obs_buf = self._compute_obs(
                    qpos_view, qvel_view, self.actions,
                    self.targets, self.up_vec, self.heading_vec,
                    self.joint_vel_obs_scaling,
                )

        return self.obs_buf

    def calculateObservations(self):
        """Non-differentiable obs computation (used by initialize_trajectory)."""
        wp.synchronize()
        qpos = wp.to_torch(self.warp_data.qpos)
        qvel = wp.to_torch(self.warp_data.qvel)
        self.obs_buf = self._compute_obs(
            qpos, qvel, self.actions,
            self.targets, self.up_vec, self.heading_vec,
            self.joint_vel_obs_scaling,
        )

    def calculateReward(self):
        """Non-differentiable reward computation (unused in diff path)."""
        pass
