"""MyoLeg26 walking environment using MuJoCo Warp.

Bilateral human gait model with 26 Hill-type muscle actuators routed through
spatial tendons.  Based on the OpenSim Gait2392 reduced model (14 DOF,
22 muscles), extended with EDL/FDL and cosmetic arms.

Key differences from AntEnv:
  - Pelvis uses 3 slide + 3 hinge joints (NOT a free joint), so nq == nv.
  - Muscle actuators with dyntype/gaintype/biastype = "muscle".
  - Pelvis orientation computed differentiably from the three hinge coordinates.
  - Action space maps tanh[-1,1] -> muscle activation [0,1].
"""

import logging

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from msk_warp.envs.base_env import MjWarpEnv
from msk_warp.bridge import WarpSimStep
import msk_warp.utils.torch_utils as tu


class MyoLeg26WalkEnv(MjWarpEnv):
    def __init__(
        self,
        num_envs=64,
        device='cuda:0',
        episode_length=1000,
        no_grad=False,
        stochastic_init=True,
        substeps=4,
        model_path='assets/myoleg/myoLeg26_BASELINE.xml',
        action_strength=1.0,
        early_termination=True,
        njmax=1000,
        **kwargs,
    ):
        # Pre-load model to discover dimensions
        from msk_warp import resolve_model_path
        _path = resolve_model_path(model_path)
        _mjm = mujoco.MjModel.from_xml_path(_path)
        nq = _mjm.nq
        nv = _mjm.nv
        nu = _mjm.nu
        na = _mjm.na

        # Observation: height(1) + xquat(4) + lin_vel(3) + ang_vel(3)
        #            + joint_q(nq-6) + joint_v(nv-6)*0.1 + act(na)
        #            + up_z(1) + heading(1) + actions(nu)
        n_joint_q = nq - 6   # exclude 6 pelvis DOFs
        n_joint_v = nv - 6
        num_obs = 1 + 4 + 3 + 3 + n_joint_q + n_joint_v + na + 1 + 1 + nu
        num_act = nu

        logging.info(f'MyoLeg26 model: nq={nq}, nv={nv}, nu={nu}, na={na}, num_obs={num_obs}')
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
            njmax=njmax,
            **kwargs,
        )

        self.stochastic_init = stochastic_init
        self.action_strength = action_strength
        self.early_termination = early_termination
        self.termination_height = 0.5

        # DOF counts
        self.nq = self.mjm.nq
        self.nv = self.mjm.nv
        self.nu = self.mjm.nu
        self.na = self.mjm.na
        self.n_joint_q = self.nq - 6
        self.n_joint_v = self.nv - 6

        self._init_pelvis_kinematics()

        self._save_start_state()
        self.reset()

    def _save_start_state(self):
        """Save keyframe states for initialization."""
        wp.synchronize()

        # Parse keyframes from MuJoCo model
        self.keyframe_qpos = {}
        self.keyframe_qvel = {}
        for name in ('stand', 'walk_left', 'walk_right'):
            kid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_KEY, name)
            if kid < 0:
                continue
            qp = torch.tensor(
                self.mjm.key_qpos[kid].copy(),
                device=self.device, dtype=torch.float32,
            )
            qv = torch.tensor(
                self.mjm.key_qvel[kid].copy(),
                device=self.device, dtype=torch.float32,
            )
            self.keyframe_qpos[name] = qp
            self.keyframe_qvel[name] = qv

        # Default: stand keyframe broadcast to all envs
        self.start_qpos = self.keyframe_qpos['stand'].unsqueeze(0).expand(
            self.num_envs, -1,
        ).clone()
        self.start_qvel = self.keyframe_qvel['stand'].unsqueeze(0).expand(
            self.num_envs, -1,
        ).clone()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _init_pelvis_kinematics(self):
        """Cache the compiled root body's rotation and ordered hinge definition."""
        self.pelvis_body_id = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'pelvis',
        )
        bid = self.pelvis_body_id
        if bid < 0 or self.mjm.body_parentid[bid] != 0:
            raise ValueError('MyoLeg26 observations require pelvis to be a child of world')
        start = int(self.mjm.body_jntadr[bid])
        joints = np.arange(start, start + int(self.mjm.body_jntnum[bid]))
        expected = [mujoco.mjtJoint.mjJNT_SLIDE] * 3 + [mujoco.mjtJoint.mjJNT_HINGE] * 3
        if not np.array_equal(self.mjm.jnt_type[joints], expected):
            raise ValueError('MyoLeg26 observations require three pelvis slides followed by three hinges')
        hinges = joints[3:]
        self._pelvis_hinge_qpos = tuple(int(i) for i in self.mjm.jnt_qposadr[hinges])
        # Keep the model's precision; conversion in _compute_pelvis_xquat follows
        # the state dtype (float32 in training, float64 for native-MuJoCo checks).
        self._pelvis_base_quat = torch.tensor(self.mjm.body_quat[bid].copy(), device=self.device)
        self._pelvis_hinge_axes = torch.tensor(self.mjm.jnt_axis[hinges].copy(), device=self.device)
        self._pelvis_hinge_refs = torch.tensor(
            self.mjm.qpos0[list(self._pelvis_hinge_qpos)].copy(), device=self.device,
        )
        # Anatomical axes in the pelvis's local frame. Its fixed base rotation
        # maps local +Y (up) onto world +Z; local +X is forward.
        self.up_vec = torch.tensor(
            [0.0, 1.0, 0.0], device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(self.num_envs, -1)
        self.heading_vec = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(self.num_envs, -1)

    def _compute_pelvis_xquat(self, qpos):
        """World quaternion [w, x, y, z] at the supplied state, without simulator reads.

        MuJoCo composes same-body hinge rotations in model order, each around
        its local axis, after the fixed body quaternion. Angles are relative
        to qpos0; these are not interchangeable with a generic Euler convention.
        """
        quat = self._pelvis_base_quat.to(qpos).expand(qpos.shape[0], -1)
        axes = self._pelvis_hinge_axes.to(qpos)
        refs = self._pelvis_hinge_refs.to(qpos)
        for i, adr in enumerate(self._pelvis_hinge_qpos):
            rotation = tu.quat_from_angle_axis(
                qpos[:, adr] - refs[i], axes[i].expand(qpos.shape[0], -1),
            )
            quat = tu.quat_mul(quat, rotation)
        return quat

    @staticmethod
    def _compute_obs(qpos, qvel, act, pelvis_xquat, actions, up_vec, heading_vec,
                     n_joint_q, n_joint_v):
        """Compute observation from state tensors.

        Differentiable in qpos, qvel, act, pelvis_xquat, and actions.

        Observation layout (nq + nv + na + nu + 1 values):
          height(1) + xquat(4) + lin_vel(3) + ang_vel(3)
          + joint_q(nq-6) + joint_v(nv-6)*0.1 + act(na)
          + up_z(1) + heading(1) + actions(nu)

        Activation is distinct from the previous action: it determines the
        current muscle force after the excitation command has changed.
        """
        # Pelvis DOFs: qpos[0:6] = [tx, ty, tz, tilt, list, rotation]
        # Due to body quat (90deg x-rotation): ty maps to world Z (height)
        height = qpos[:, 1:2]

        # Pelvis velocities
        lin_vel = qvel[:, 0:3]
        ang_vel = qvel[:, 3:6]

        # All joint positions/velocities beyond pelvis
        joint_q = qpos[:, 6:6 + n_joint_q]
        joint_v = qvel[:, 6:6 + n_joint_v]

        # Orientation-derived features from xquat
        up_proj = tu.quat_rotate(pelvis_xquat, up_vec)
        up_z = up_proj[:, 2:3]

        heading_proj = tu.quat_rotate(pelvis_xquat, heading_vec)
        heading = heading_proj[:, 0:1]  # x-component = forward alignment

        obs = torch.cat([
            height,               # 1
            pelvis_xquat,         # 4
            lin_vel,              # 3
            ang_vel,              # 3
            joint_q,              # n_joint_q
            joint_v * 0.1,        # n_joint_v (scaled)
            act,                  # na
            up_z,                 # 1
            heading,              # 1
            actions,              # nu
        ], dim=-1)
        return obs

    @staticmethod
    def _compute_reward(obs, actions):
        """Compute locomotion reward (differentiable in obs and actions).

        Reward = forward_vel + 0.1*upright + heading + 0.2*alive
                 + 0.5*(height - 0.6) - 0.005*energy
        """
        height = obs[:, 0]
        # lin_vel starts at obs index 5; lin_vel[0] = pelvis_tx_dot = forward vel
        forward_vel = obs[:, 5]
        # up_z and heading are at fixed offsets from the end
        # up_z = obs[:, -(nu+2)], heading = obs[:, -(nu+1)]
        nu = actions.shape[-1]
        up_z = obs[:, -(nu + 2)]
        heading = obs[:, -(nu + 1)]

        forward_reward = forward_vel
        upright_reward = 0.1 * up_z
        heading_reward = heading
        alive_reward = 0.2
        height_reward = 0.5 * (height - 0.6)
        energy_cost = 0.005 * (actions ** 2).sum(dim=-1)

        reward = (
            forward_reward
            + upright_reward
            + heading_reward
            + alive_reward
            + height_reward
            - energy_cost
        )
        return reward

    def compute_obs(self, qpos, qvel, act=None):
        """Pure observation computation from the complete supplied tracked state.

        Activation is required explicitly, so a supplied mechanical state is
        never silently combined with activation from another simulator state.
        """
        if act is None:
            raise ValueError('MyoLeg26 observations require explicit muscle activation state (act)')
        if act.shape != (qpos.shape[0], self.mjm.na):
            raise ValueError(f'Expected act shape {(qpos.shape[0], self.mjm.na)}, got {tuple(act.shape)}')
        pelvis_xquat = self._compute_pelvis_xquat(qpos)
        return self._compute_obs(
            qpos, qvel, act, pelvis_xquat, self.actions,
            self.up_vec, self.heading_vec,
            self.n_joint_q, self.n_joint_v,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions, qpos_in=None, qvel_in=None, act_in=None):
        """Run one control step.

        Args:
            actions: (num_envs, 26) action tensor from the policy (tanh'd)
            qpos_in, qvel_in, act_in: optional differentiable state inputs (BPTT);
                act is the muscle activation state and must be threaded

        Returns:
            obs, rew, done, extras, qpos_out, qvel_out, act_out
        """
        actions = actions.view(self.num_envs, self.num_actions)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions.detach().clone()

        # Map tanh[-1,1] -> muscle activation [0,1]
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
            act = wp.to_torch(self.warp_data.act)
            pelvis_xquat = self._compute_pelvis_xquat(qpos)
            self.obs_buf = self._compute_obs(
                qpos, qvel, act, pelvis_xquat, actions,
                self.up_vec, self.heading_vec,
                self.n_joint_q, self.n_joint_v,
            )
            self.rew_buf = self._compute_reward(self.obs_buf, actions)
            qpos_out, qvel_out, act_out = None, None, None
        else:
            qpos_in, qvel_in, act_in = self._state_inputs(qpos_in, qvel_in, act_in)
            qpos_out, qvel_out, act_out = WarpSimStep.apply(ctrl, qpos_in, qvel_in, act_in, self)

            qpos_out = qpos_out.clamp(-100.0, 100.0)
            qvel_out = qvel_out.clamp(-100.0, 100.0)

            # Sanitize gradients flowing INTO bridge backward from downstream
            # obs/reward computation. (bridge._sanitize_and_clamp handles
            # gradients flowing OUT of bridge backward from the Warp tape.)
            if qpos_out.requires_grad:
                qpos_out.register_hook(lambda g: torch.nan_to_num(g, 0.0, 0.0, 0.0))
            if qvel_out.requires_grad:
                qvel_out.register_hook(lambda g: torch.nan_to_num(g, 0.0, 0.0, 0.0))

            # Derived Warp kinematics still describe the last substep's input.
            # Compute orientation from the returned post-integration state.
            pelvis_xquat = self._compute_pelvis_xquat(qpos_out)

            self.obs_buf = self._compute_obs(
                qpos_out, qvel_out, act_out, pelvis_xquat, actions,
                self.up_vec, self.heading_vec,
                self.n_joint_q, self.n_joint_v,
            )
            self.rew_buf = self._compute_reward(self.obs_buf, actions)

        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.progress_buf += 1

        # Early termination: pelvis height below threshold
        if self.early_termination:
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

        # Save obs before reset for critic bootstrap
        if not self.no_grad:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'episode_end': self.termination_buf,
            }

        # Episode length termination
        self.reset_buf = torch.where(
            self.progress_buf >= self.episode_length,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_warp_state(env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras, qpos_out, qvel_out, act_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_warp_state(self, env_ids):
        """Reset Warp state for specified environments (no gradient)."""
        with torch.no_grad():
            qpos_torch = wp.to_torch(self.warp_data.qpos)
            qvel_torch = wp.to_torch(self.warp_data.qvel)
            act_torch = wp.to_torch(self.warp_data.act)

            n = len(env_ids)

            if self.stochastic_init and 'walk_left' in self.keyframe_qpos:
                # Randomly choose walk_left or walk_right keyframe per env
                choices = torch.randint(0, 2, (n,), device=self.device)
                kf_names = ['walk_left', 'walk_right']
                for i in range(n):
                    eid = env_ids[i]
                    kf = kf_names[choices[i].item()]
                    qpos_torch[eid, :] = self.keyframe_qpos[kf].clone()
                    qvel_torch[eid, :] = self.keyframe_qvel[kf].clone()

                # Small noise on joint angles (indices 6:nq)
                qpos_torch[env_ids, 6:self.nq] += 0.02 * (
                    torch.rand(n, self.nq - 6, device=self.device) - 0.5
                ) * 2.0

                # Small perturbation on pelvis position
                qpos_torch[env_ids, 0] += 0.05 * (
                    torch.rand(n, device=self.device) - 0.5
                ) * 2.0   # forward
                qpos_torch[env_ids, 1] += 0.02 * (
                    torch.rand(n, device=self.device) - 0.5
                ) * 2.0   # height

                # Small random pelvis velocities
                qvel_torch[env_ids, :6] = 0.1 * (
                    torch.rand(n, 6, device=self.device) - 0.5
                )
            else:
                qpos_torch[env_ids, :] = self.start_qpos[env_ids, :].clone()
                qvel_torch[env_ids, :] = self.start_qvel[env_ids, :].clone()

            # Reset muscle activations and stored actions
            act_torch[env_ids, :] = 0.0
            self.actions[env_ids, :] = 0.0

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
                act_view = wp.to_torch(self.warp_data.act)

                self.obs_buf = self.compute_obs(qpos_view, qvel_view, act_view)

        return self.obs_buf

    def calculateObservations(self):
        """Non-differentiable obs computation (used by initialize_trajectory)."""
        wp.synchronize()
        qpos = wp.to_torch(self.warp_data.qpos)
        qvel = wp.to_torch(self.warp_data.qvel)
        act = wp.to_torch(self.warp_data.act)

        self.obs_buf = self.compute_obs(qpos, qvel, act)

    def calculateReward(self):
        """Non-differentiable reward computation (unused in diff path)."""
        pass
