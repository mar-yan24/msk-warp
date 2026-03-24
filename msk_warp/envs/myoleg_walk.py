"""MyoLeg walking environment using MuJoCo Warp.

Scaffolding for musculoskeletal locomotion training with MyoSuite leg models.
The model is loaded from the myosuite installation (auto-discovered or explicit path).

Note: This is an initial setup. MuJoCo Warp support for muscle/tendon actuators
should be verified before full training. The FD Jacobian cost scales with DOF count,
making training significantly slower than simpler environments.
"""

import os

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from msk_warp.envs.base_env import MjWarpEnv
from msk_warp.bridge import WarpSimStep
import msk_warp.utils.torch_utils as tu


class MyoLegWalkEnv(MjWarpEnv):
    def __init__(
        self,
        num_envs=16,
        device='cuda:0',
        episode_length=500,
        no_grad=False,
        stochastic_init=False,
        substeps=4,
        model_path=None,
        action_strength=1.0,
        **kwargs,
    ):
        # Auto-discover model path if not provided
        if model_path is None:
            model_path = self._find_myosuite_model()

        # Pre-load model to discover dimensions
        _mjm = mujoco.MjModel.from_xml_path(model_path)
        nq = _mjm.nq
        nv = _mjm.nv
        nu = _mjm.nu

        # Observation: height(1) + quat(4) + lin_vel(3) + ang_vel(3)
        #            + joint_q(nq-7) + joint_v(nv-6) + up_z(1)
        num_obs = 1 + 4 + 3 + 3 + (nq - 7) + (nv - 6) + 1
        num_act = nu

        print(f'MyoLeg model: nq={nq}, nv={nv}, nu={nu}, num_obs={num_obs}')
        del _mjm

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
        self.termination_height = 0.5

        # Store actual DOF counts from loaded model
        self.nq = self.mjm.nq
        self.nv = self.mjm.nv
        self.nu = self.mjm.nu
        self.n_joint_q = self.nq - 7   # exclude free joint pos
        self.n_joint_v = self.nv - 6   # exclude free joint vel

        # Basis vectors (z-up)
        self.up_vec = torch.tensor(
            [0.0, 0.0, 1.0], device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(num_envs, -1)

        self._save_start_state()
        self.reset()

    @staticmethod
    def _find_myosuite_model():
        """Find myolegs.xml from myosuite installation."""
        # Try importing myosuite
        try:
            import myosuite
            base = os.path.dirname(myosuite.__file__)
            path = os.path.join(base, 'simhive', 'myo_sim', 'leg', 'myolegs.xml')
            if os.path.exists(path):
                return path
        except ImportError:
            pass

        # Fallback to known installation path
        fallback = os.path.join(
            os.path.expanduser('~'),
            'Documents', 'MyoAssist', 'myoassist', '.my_venv',
            'Lib', 'site-packages', 'myosuite',
            'simhive', 'myo_sim', 'leg', 'myolegs.xml',
        )
        if os.path.exists(fallback):
            return fallback

        raise FileNotFoundError(
            "Could not find myolegs.xml. Install myosuite or provide model_path explicitly."
        )

    def _save_start_state(self):
        """Save the default standing state from the model's reset configuration."""
        wp.synchronize()
        self.start_qpos = wp.to_torch(self.warp_data.qpos).clone()
        self.start_qvel = wp.to_torch(self.warp_data.qvel).clone()
        self.start_qvel[:, :] = 0.0

    @staticmethod
    def _compute_obs(qpos, qvel, up_vec, n_joint_q, n_joint_v):
        """Compute observation from state tensors (differentiable in PyTorch).

        Observation layout:
          height(1) + quat(4) + lin_vel(3) + ang_vel(3) +
          joint_q(nq-7) + joint_v(nv-6) + up_z(1)
        """
        # Free joint: pos(3) + quat(4), vel: lin(3) + ang(3)
        torso_pos = qpos[:, 0:3]
        torso_quat = qpos[:, 3:7]   # [w, x, y, z]
        joint_q = qpos[:, 7:7 + n_joint_q]

        lin_vel = qvel[:, 0:3]
        ang_vel = qvel[:, 3:6]
        joint_vel = qvel[:, 6:6 + n_joint_v]

        height = torso_pos[:, 2:3]

        # Up-vector projection (z-component)
        up_proj = tu.quat_rotate(torso_quat, up_vec)

        obs = torch.cat([
            height,
            torso_quat,
            lin_vel,
            ang_vel,
            joint_q,
            joint_vel * 0.1,
            up_proj[:, 2:3],
        ], dim=-1)
        return obs

    @staticmethod
    def _compute_reward(obs, actions):
        """Compute locomotion reward (differentiable in PyTorch).

        Reward = forward_vel + upright_bonus + alive_bonus - energy_cost
        """
        height = obs[:, 0]
        forward_vel = obs[:, 5]        # lin_vel_x
        up_z = obs[:, -1]              # up-vector z-component

        forward_reward = forward_vel
        upright_reward = 0.1 * up_z
        alive_reward = 0.1 * torch.ones_like(height)
        # Muscle activation cost (activations mapped to [0,1])
        energy_cost = 0.01 * (actions ** 2).sum(dim=-1)

        reward = forward_reward + upright_reward + alive_reward - energy_cost
        return reward

    def compute_obs(self, qpos, qvel):
        """Instance method wrapper for SHAC compatibility."""
        return self._compute_obs(
            qpos, qvel, self.up_vec,
            self.n_joint_q, self.n_joint_v,
        )

    def step(self, actions, qpos_in=None, qvel_in=None):
        """Run one control step."""
        actions = actions.view(self.num_envs, self.num_actions)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions.detach()

        # Remap tanh output [-1,1] to muscle activation [0,1]
        activation = 0.5 * (actions + 1.0)
        ctrl = activation * self.action_strength

        if self.no_grad:
            ctrl_wp = wp.from_torch(ctrl.detach().contiguous())
            wp.copy(self.warp_data.ctrl, ctrl_wp)
            for _ in range(self.substeps):
                mjw.step(self.warp_model, self.warp_data)
            wp.synchronize()

            qpos = wp.to_torch(self.warp_data.qpos)
            qvel = wp.to_torch(self.warp_data.qvel)
            self.obs_buf = self._compute_obs(
                qpos, qvel, self.up_vec,
                self.n_joint_q, self.n_joint_v,
            )
            self.rew_buf = self._compute_reward(self.obs_buf, actions)
            qpos_out, qvel_out = None, None
        else:
            if qpos_in is None:
                qpos_in = wp.to_torch(self.warp_data.qpos).clone()
            if qvel_in is None:
                qvel_in = wp.to_torch(self.warp_data.qvel).clone()

            qpos_out, qvel_out = WarpSimStep.apply(ctrl, qpos_in, qvel_in, self)

            qpos_out = qpos_out.clamp(-100.0, 100.0)
            qvel_out = qvel_out.clamp(-100.0, 100.0)

            self.obs_buf = self._compute_obs(
                qpos_out, qvel_out, self.up_vec,
                self.n_joint_q, self.n_joint_v,
            )
            self.rew_buf = self._compute_reward(self.obs_buf, actions)

        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.progress_buf += 1

        # Termination: pelvis height below threshold
        self.termination_buf = torch.where(
            self.obs_buf[:, 0] < self.termination_height,
            torch.ones_like(self.termination_buf),
            torch.zeros_like(self.termination_buf),
        )
        self.reset_buf = torch.where(
            self.termination_buf > 0,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

        if not self.no_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf,
            }

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
                # Small perturbation on position
                qpos_torch[env_ids, 0:3] += 0.05 * (
                    torch.rand(n, 3, device=self.device) - 0.5
                ) * 2.0

        self.progress_buf[env_ids] = 0

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            self._reset_warp_state(env_ids)

            with torch.no_grad():
                qpos_view = wp.to_torch(self.warp_data.qpos)
                qvel_view = wp.to_torch(self.warp_data.qvel)
                self.obs_buf = self._compute_obs(
                    qpos_view, qvel_view, self.up_vec,
                    self.n_joint_q, self.n_joint_v,
                )

        return self.obs_buf

    def calculateObservations(self):
        """Non-differentiable obs computation (used by initialize_trajectory)."""
        wp.synchronize()
        qpos = wp.to_torch(self.warp_data.qpos)
        qvel = wp.to_torch(self.warp_data.qvel)
        self.obs_buf = self._compute_obs(
            qpos, qvel, self.up_vec,
            self.n_joint_q, self.n_joint_v,
        )

    def calculateReward(self):
        """Non-differentiable reward computation (unused in diff path)."""
        pass
