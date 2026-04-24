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


RESET_MODE_CANONICAL = "canonical"
RESET_MODE_STRIDE_LEFT = "stride_left"
RESET_MODE_STRIDE_RIGHT = "stride_right"

# qpos joint order follows joint definition order in the MJCF:
# [hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4].
STRIDE_LEFT_JOINT_BIAS = (
    -0.20,
    0.30,
    0.20,
    -0.30,
    0.20,
    -0.30,
    -0.20,
    0.30,
)
STRIDE_RIGHT_JOINT_BIAS = tuple(-value for value in STRIDE_LEFT_JOINT_BIAS)


def normalize_reward_curriculum_cfg(reward_curriculum):
    """Return a normalized reward-curriculum config dictionary."""
    cfg = dict(reward_curriculum or {})
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "anneal_epochs": max(0, int(cfg.get("anneal_epochs", 0))),
        "target_speed": float(cfg.get("target_speed", 0.0)),
        "low_speed_penalty_weight_init": float(cfg.get("low_speed_penalty_weight_init", 0.0)),
        "forward_scale_init": float(cfg.get("forward_scale_init", 1.0)),
        "heading_scale_init": float(cfg.get("heading_scale_init", 1.0)),
        "height_scale_init": float(cfg.get("height_scale_init", 1.0)),
        "up_scale_init": float(cfg.get("up_scale_init", 1.0)),
    }


def compute_reward_curriculum_state(
    reward_curriculum,
    *,
    epoch,
    base_forward_vel_weight,
    base_heading_weight,
    base_height_weight,
    base_up_weight,
):
    """Compute epoch-dependent effective reward weights for Ant training."""
    cfg = normalize_reward_curriculum_cfg(reward_curriculum)
    out = {
        "progress": 1.0,
        "target_speed": 0.0,
        "low_speed_penalty_weight": 0.0,
        "forward_vel_weight": float(base_forward_vel_weight),
        "heading_weight": float(base_heading_weight),
        "height_weight": float(base_height_weight),
        "up_weight": float(base_up_weight),
    }
    if not cfg["enabled"]:
        return out

    anneal_epochs = cfg["anneal_epochs"]
    if anneal_epochs <= 0:
        progress = 1.0
    else:
        progress = min(max(float(epoch) / float(anneal_epochs), 0.0), 1.0)

    def _anneal_scale(init_scale):
        return (1.0 - progress) * float(init_scale) + progress

    out["progress"] = progress
    out["target_speed"] = float(cfg["target_speed"])
    out["low_speed_penalty_weight"] = (1.0 - progress) * float(
        cfg["low_speed_penalty_weight_init"]
    )
    out["forward_vel_weight"] = float(base_forward_vel_weight) * _anneal_scale(
        cfg["forward_scale_init"]
    )
    out["heading_weight"] = float(base_heading_weight) * _anneal_scale(cfg["heading_scale_init"])
    out["height_weight"] = float(base_height_weight) * _anneal_scale(cfg["height_scale_init"])
    out["up_weight"] = float(base_up_weight) * _anneal_scale(cfg["up_scale_init"])
    return out


def _normalize_range(cfg, key, default):
    value = cfg.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = value
    else:
        lo, hi = default
    lo = float(lo)
    hi = float(hi)
    return (min(lo, hi), max(lo, hi))


def normalize_reset_curriculum_cfg(reset_curriculum):
    """Return a normalized reset-curriculum config dictionary."""
    cfg = dict(reset_curriculum or {})
    canonical_prob_init = float(cfg.get("canonical_prob_init", 1.0))
    canonical_prob_init = min(max(canonical_prob_init, 0.0), 1.0)
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "anneal_epochs": max(0, int(cfg.get("anneal_epochs", 0))),
        "canonical_prob_init": canonical_prob_init,
        "x_vel_range": _normalize_range(cfg, "x_vel_range", (0.2, 0.5)),
        "y_vel_abs_max": abs(float(cfg.get("y_vel_abs_max", 0.1))),
        "yaw_deg_abs_max": abs(float(cfg.get("yaw_deg_abs_max", 12.0))),
        "yaw_rate_abs_max": abs(float(cfg.get("yaw_rate_abs_max", 0.4))),
    }


def compute_reset_curriculum_state(reset_curriculum, *, epoch):
    """Compute epoch-dependent reset-mode probabilities for Ant training."""
    cfg = normalize_reset_curriculum_cfg(reset_curriculum)
    out = {
        "progress": 1.0,
        "canonical_prob": 1.0,
        "stride_left_prob": 0.0,
        "stride_right_prob": 0.0,
        "x_vel_range": tuple(cfg["x_vel_range"]),
        "y_vel_abs_max": float(cfg["y_vel_abs_max"]),
        "yaw_deg_abs_max": float(cfg["yaw_deg_abs_max"]),
        "yaw_rate_abs_max": float(cfg["yaw_rate_abs_max"]),
    }
    if not cfg["enabled"]:
        return out

    anneal_epochs = cfg["anneal_epochs"]
    if anneal_epochs <= 0:
        progress = 1.0
    else:
        progress = min(max(float(epoch) / float(anneal_epochs), 0.0), 1.0)

    canonical_prob = (
        (1.0 - progress) * float(cfg["canonical_prob_init"])
        + progress * 1.0
    )
    residual = max(0.0, 1.0 - canonical_prob)

    out["progress"] = progress
    out["canonical_prob"] = canonical_prob
    out["stride_left_prob"] = 0.5 * residual
    out["stride_right_prob"] = 0.5 * residual
    return out


def get_reset_joint_bias(reset_mode):
    """Return the qpos-order joint bias for a named reset mode."""
    if reset_mode == RESET_MODE_CANONICAL:
        return (0.0,) * 8
    if reset_mode == RESET_MODE_STRIDE_LEFT:
        return STRIDE_LEFT_JOINT_BIAS
    if reset_mode == RESET_MODE_STRIDE_RIGHT:
        return STRIDE_RIGHT_JOINT_BIAS
    raise ValueError(f"Unsupported reset mode: {reset_mode}")


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
        use_fd_jacobian=False,
        tape_per_substep=False,
        forward_vel_weight=1.0,
        heading_weight=1.0,
        up_weight=0.1,
        height_weight=1.0,
        joint_vel_penalty=0.0,
        push_reward_weight=0.0,
        **kwargs,
    ):
        num_obs = 37
        num_act = 8

        # Extract smooth adjoint params from kwargs before passing to super
        smooth_adjoint = kwargs.pop('smooth_adjoint', False)
        smooth_friction_viscosity = kwargs.pop('smooth_friction_viscosity', 10.0)
        smooth_friction_scale = kwargs.pop('smooth_friction_scale', 0.01)
        friction_bypass_kf = kwargs.pop('friction_bypass_kf', 0.0)
        free_body_adjoint = kwargs.pop('free_body_adjoint', False)
        penalty_damping_alpha = kwargs.pop('penalty_damping_alpha', 0.0)
        friction_surrogate_adjoint = kwargs.pop('friction_surrogate_adjoint', False)
        friction_surrogate_alpha = kwargs.pop('friction_surrogate_alpha', 0.0)
        cfd_width = float(kwargs.pop('cfd_width', 0.0))
        reward_curriculum = normalize_reward_curriculum_cfg(
            kwargs.pop('reward_curriculum', None)
        )
        reset_curriculum = normalize_reset_curriculum_cfg(
            kwargs.pop('reset_curriculum', None)
        )

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
            use_fd_jacobian=use_fd_jacobian,
            tape_per_substep=tape_per_substep,
            smooth_adjoint=smooth_adjoint,
            smooth_friction_viscosity=smooth_friction_viscosity,
            smooth_friction_scale=smooth_friction_scale,
            friction_bypass_kf=friction_bypass_kf,
            free_body_adjoint=free_body_adjoint,
            penalty_damping_alpha=penalty_damping_alpha,
            friction_surrogate_adjoint=friction_surrogate_adjoint,
            friction_surrogate_alpha=friction_surrogate_alpha,
            cfd_width=cfd_width,
        )

        self.stochastic_init = stochastic_init
        self.action_strength = action_strength
        self.early_termination = early_termination
        self.forward_vel_weight = forward_vel_weight
        self.heading_weight = heading_weight
        self.up_weight = up_weight
        self.height_weight = height_weight
        self.joint_vel_penalty = joint_vel_penalty
        self.push_reward_weight = push_reward_weight
        self.reward_curriculum = reward_curriculum
        self.reset_curriculum = reset_curriculum
        self.current_forward_vel_weight = float(self.forward_vel_weight)
        self.current_heading_weight = float(self.heading_weight)
        self.current_up_weight = float(self.up_weight)
        self.current_height_weight = float(self.height_weight)
        self.current_target_speed = 0.0
        self.current_low_speed_penalty_weight = 0.0
        self.current_reset_curriculum_active = False
        self.current_reset_canonical_prob = 1.0
        self.current_reset_stride_left_prob = 0.0
        self.current_reset_stride_right_prob = 0.0
        self.current_reset_x_vel_range = tuple(self.reset_curriculum["x_vel_range"])
        self.current_reset_y_vel_abs_max = float(self.reset_curriculum["y_vel_abs_max"])
        self.current_reset_yaw_deg_abs_max = float(self.reset_curriculum["yaw_deg_abs_max"])
        self.current_reset_yaw_rate_abs_max = float(self.reset_curriculum["yaw_rate_abs_max"])
        self._reset_joint_bias = {
            RESET_MODE_CANONICAL: torch.tensor(
                get_reset_joint_bias(RESET_MODE_CANONICAL),
                device=device,
                dtype=torch.float32,
            ),
            RESET_MODE_STRIDE_LEFT: torch.tensor(
                get_reset_joint_bias(RESET_MODE_STRIDE_LEFT),
                device=device,
                dtype=torch.float32,
            ),
            RESET_MODE_STRIDE_RIGHT: torch.tensor(
                get_reset_joint_bias(RESET_MODE_STRIDE_RIGHT),
                device=device,
                dtype=torch.float32,
            ),
        }

        self.termination_height = 0.27
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
    def _compute_reward(obs, actions, action_penalty,
                        forward_vel_weight=1.0, heading_weight=1.0,
                        up_weight=0.1, height_weight=1.0,
                        joint_vel_penalty=0.0, push_reward_weight=0.0,
                        target_speed=0.0, low_speed_penalty_weight=0.0):
        """Compute reward from observation tensor (differentiable in PyTorch).

        Default matches DiffRL's ant reward:
          forward_vel + 0.1 * up + heading + (height - 0.27) + action_penalty
        Weights can be adjusted to change the reward landscape.

        joint_vel_penalty adds a viscous friction-like term: -kf * sum(joint_vel^2).
        push_reward_weight adds a shaping reward for forward-pushing actions.
        low_speed_penalty_weight adds a temporary training penalty for staying
        below a target forward speed.
        """
        height = obs[:, 0]
        forward_vel = obs[:, 5]        # lin_vel_x
        up_reward = obs[:, 27]         # up-vector z-component
        heading_reward = obs[:, 28]    # heading alignment
        height_reward = height - 0.27

        reward = (
            forward_vel_weight * forward_vel
            + up_weight * up_reward
            + heading_weight * heading_reward
            + height_weight * height_reward
            + action_penalty * (actions ** 2).sum(dim=-1)
        )

        # Viscous friction penalty: penalizes joint velocity
        if joint_vel_penalty != 0.0:
            joint_vel = obs[:, 19:27] * 10.0
            reward = reward - joint_vel_penalty * (joint_vel ** 2).sum(dim=-1)

        if low_speed_penalty_weight != 0.0:
            speed_shortfall = torch.relu(float(target_speed) - forward_vel)
            reward = reward - low_speed_penalty_weight * (speed_shortfall ** 2)

        # Push shaping: reward actions that push the ant forward.
        # Bypasses the constraint solver gradient entirely — provides
        # a direct gradient from actions to reward for locomotion.
        # Action layout: [hip4, ank4, hip1, ank1, hip2, ank2, hip3, ank3]
        # Ant leg geometry (z-up, forward = +x):
        #   hip4 (back-right at +x,-y): -torque → forward push
        #   hip1 (front-left at +x,+y): -torque → forward push
        #   hip2 (front-right at -x,+y): +torque → forward push
        #   hip3 (back-left at -x,-y): +torque → forward push
        if push_reward_weight != 0.0:
            push = (-actions[:, 0] - actions[:, 2]
                    + actions[:, 4] + actions[:, 6])
            reward = reward + push_reward_weight * push

        return reward

    def begin_epoch(self, epoch: int, max_epochs: int) -> dict[str, float]:
        """Apply optional training curricula for the next epoch."""
        metrics = {}

        reward_state = compute_reward_curriculum_state(
            self.reward_curriculum,
            epoch=epoch,
            base_forward_vel_weight=self.forward_vel_weight,
            base_heading_weight=self.heading_weight,
            base_height_weight=self.height_weight,
            base_up_weight=self.up_weight,
        )
        self.current_forward_vel_weight = float(reward_state["forward_vel_weight"])
        self.current_heading_weight = float(reward_state["heading_weight"])
        self.current_up_weight = float(reward_state["up_weight"])
        self.current_height_weight = float(reward_state["height_weight"])
        self.current_target_speed = float(reward_state["target_speed"])
        self.current_low_speed_penalty_weight = float(
            reward_state["low_speed_penalty_weight"]
        )

        if self.reward_curriculum["enabled"]:
            metrics.update(
                {
                    "reward_curriculum_progress": float(reward_state["progress"]),
                    "reward_curriculum_target_speed": self.current_target_speed,
                    "reward_curriculum_low_speed_penalty_weight": self.current_low_speed_penalty_weight,
                    "reward_curriculum_forward_vel_weight": self.current_forward_vel_weight,
                    "reward_curriculum_heading_weight": self.current_heading_weight,
                    "reward_curriculum_height_weight": self.current_height_weight,
                    "reward_curriculum_up_weight": self.current_up_weight,
                }
            )

        reset_state = compute_reset_curriculum_state(
            self.reset_curriculum,
            epoch=epoch,
        )
        self.current_reset_curriculum_active = bool(self.reset_curriculum["enabled"])
        self.current_reset_canonical_prob = float(reset_state["canonical_prob"])
        self.current_reset_stride_left_prob = float(reset_state["stride_left_prob"])
        self.current_reset_stride_right_prob = float(reset_state["stride_right_prob"])
        self.current_reset_x_vel_range = tuple(reset_state["x_vel_range"])
        self.current_reset_y_vel_abs_max = float(reset_state["y_vel_abs_max"])
        self.current_reset_yaw_deg_abs_max = float(reset_state["yaw_deg_abs_max"])
        self.current_reset_yaw_rate_abs_max = float(reset_state["yaw_rate_abs_max"])

        if self.reset_curriculum["enabled"]:
            metrics.update(
                {
                    "reset_curriculum_progress": float(reset_state["progress"]),
                    "reset_curriculum_canonical_prob": self.current_reset_canonical_prob,
                    "reset_curriculum_stride_left_prob": self.current_reset_stride_left_prob,
                    "reset_curriculum_stride_right_prob": self.current_reset_stride_right_prob,
                }
            )

        return metrics

    def _sample_reset_mode_codes(self, count: int) -> torch.Tensor:
        """Sample reset modes for a batch of env ids."""
        if (
            count <= 0
            or not self.current_reset_curriculum_active
            or not self.reset_curriculum["enabled"]
        ):
            return torch.zeros(count, dtype=torch.long, device=self.device)

        draws = torch.rand(count, device=self.device)
        canonical_threshold = float(self.current_reset_canonical_prob)
        left_threshold = canonical_threshold + float(self.current_reset_stride_left_prob)

        mode_codes = torch.full((count,), 2, dtype=torch.long, device=self.device)
        mode_codes[draws < left_threshold] = 1
        mode_codes[draws < canonical_threshold] = 0
        return mode_codes

    def _apply_canonical_reset_noise(self, qpos_torch, qvel_torch, env_ids):
        n = len(env_ids)

        qpos_torch[env_ids, 0:3] += 0.1 * (
            torch.rand(n, 3, device=self.device) - 0.5
        ) * 2.0

        angle = (torch.rand(n, device=self.device) - 0.5) * (math.pi / 12.0)
        axis = torch.nn.functional.normalize(
            torch.rand(n, 3, device=self.device) - 0.5, dim=-1
        )
        rand_quat = tu.quat_from_angle_axis(angle, axis)
        cur_quat = qpos_torch[env_ids, 3:7].clone()
        qpos_torch[env_ids, 3:7] = tu.quat_mul(cur_quat, rand_quat)

        qpos_torch[env_ids, 7:15] += 0.2 * (
            torch.rand(n, 8, device=self.device) - 0.5
        ) * 2.0

        qvel_torch[env_ids, :] = 0.5 * (
            torch.rand(n, self.num_joint_qd, device=self.device) - 0.5
        )

    def _apply_stride_reset_noise(self, qpos_torch, qvel_torch, env_ids, *, reset_mode):
        n = len(env_ids)
        if n <= 0:
            return

        qpos_torch[env_ids, 0:3] += 0.05 * (
            torch.rand(n, 3, device=self.device) - 0.5
        ) * 2.0

        yaw_limit_rad = math.radians(self.current_reset_yaw_deg_abs_max)
        yaw = (torch.rand(n, device=self.device) - 0.5) * (2.0 * yaw_limit_rad)
        yaw_axis = torch.zeros(n, 3, device=self.device, dtype=torch.float32)
        yaw_axis[:, 2] = 1.0
        yaw_quat = tu.quat_from_angle_axis(yaw, yaw_axis)
        cur_quat = qpos_torch[env_ids, 3:7].clone()
        qpos_torch[env_ids, 3:7] = tu.quat_mul(cur_quat, yaw_quat)

        qpos_torch[env_ids, 7:15] += self._reset_joint_bias[reset_mode].unsqueeze(0)
        qpos_torch[env_ids, 7:15] += 0.1 * (
            torch.rand(n, 8, device=self.device) - 0.5
        ) * 2.0

        qvel_torch[env_ids, :] = 0.25 * (
            torch.rand(n, self.num_joint_qd, device=self.device) - 0.5
        )
        x_vel_lo, x_vel_hi = self.current_reset_x_vel_range
        qvel_torch[env_ids, 0] = x_vel_lo + (x_vel_hi - x_vel_lo) * torch.rand(
            n, device=self.device
        )
        qvel_torch[env_ids, 1] = (
            (torch.rand(n, device=self.device) - 0.5)
            * 2.0
            * self.current_reset_y_vel_abs_max
        )
        qvel_torch[env_ids, 5] = (
            (torch.rand(n, device=self.device) - 0.5)
            * 2.0
            * self.current_reset_yaw_rate_abs_max
        )

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
            self.rew_buf = self._compute_reward(
                self.obs_buf, actions, self.action_penalty,
                self.current_forward_vel_weight, self.current_heading_weight,
                self.current_up_weight, self.current_height_weight,
                self.joint_vel_penalty, self.push_reward_weight,
                self.current_target_speed, self.current_low_speed_penalty_weight)
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

            self.obs_buf = self._compute_obs(
                qpos_out, qvel_out, actions,
                self.targets, self.up_vec, self.heading_vec,
                self.joint_vel_obs_scaling,
            )
            self.rew_buf = self._compute_reward(
                self.obs_buf, actions, self.action_penalty,
                self.current_forward_vel_weight, self.current_heading_weight,
                self.current_up_weight, self.current_height_weight,
                self.joint_vel_penalty, self.push_reward_weight,
                self.current_target_speed, self.current_low_speed_penalty_weight)

        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.progress_buf += 1

        # Early termination: height below threshold
        if self.early_termination:
            too_low = self.obs_buf[:, 0] < self.termination_height
            self.termination_buf = torch.where(
                too_low,
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
                mode_codes = self._sample_reset_mode_codes(len(env_ids))

                canonical_ids = env_ids[mode_codes == 0]
                stride_left_ids = env_ids[mode_codes == 1]
                stride_right_ids = env_ids[mode_codes == 2]

                if len(canonical_ids) > 0:
                    self._apply_canonical_reset_noise(
                        qpos_torch, qvel_torch, canonical_ids
                    )
                if len(stride_left_ids) > 0:
                    self._apply_stride_reset_noise(
                        qpos_torch,
                        qvel_torch,
                        stride_left_ids,
                        reset_mode=RESET_MODE_STRIDE_LEFT,
                    )
                if len(stride_right_ids) > 0:
                    self._apply_stride_reset_noise(
                        qpos_torch,
                        qvel_torch,
                        stride_right_ids,
                        reset_mode=RESET_MODE_STRIDE_RIGHT,
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
