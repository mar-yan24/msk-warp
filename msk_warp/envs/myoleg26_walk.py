"""MyoLeg26 locomotion with explicit model and observation contracts.

The official model has a free root and a rigidly attached pelvis; the historical
slide/hinge root is available only with ``model_contract='legacy'``. Pelvis pose,
world-frame velocities and activation observations use the supplied state.
"""

import logging

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from msk_warp.envs.base_env import MjWarpEnv
from msk_warp.backend import GradContractError
from msk_warp.bridge import WarpSimStep
from msk_warp.envs.myoleg26_task import MyoLegTaskContract, excitation_from_action
from msk_warp.models.myoleg26 import COLLISION_PREFIX, FOOT_BODIES, GROUND_NAME
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
        model_path='assets/myoleg26/flat_boxes.xml',
        model_contract='official',
        allow_unvalidated_gradients=False,
        action_strength=1.0,
        target_speed=1.0,
        early_termination=True,
        njmax=1000,
        **kwargs,
    ):
        if model_contract not in ('official', 'legacy'):
            raise ValueError("model_contract must be 'official' or 'legacy'")
        if model_contract == 'official' and action_strength != 1.0:
            raise ValueError('Official MyoLeg26 requires action_strength=1.0 for excitation in [0, 1]')
        if model_contract == 'official' and not no_grad and not allow_unvalidated_gradients:
            raise GradContractError(
                'Official MyoLeg26 free-root and tendon gradients are unvalidated. '
                'Use no_grad=True for forward evaluation; '
                'allow_unvalidated_gradients=True is reserved for explicit gradient diagnostics.'
            )
        self.model_contract = model_contract
        self.allow_unvalidated_gradients = allow_unvalidated_gradients
        self.root_qpos_size = 7 if model_contract == 'official' else 6
        # Pre-load model to discover dimensions
        from msk_warp import resolve_model_path
        _path = resolve_model_path(model_path)
        _mjm = mujoco.MjModel.from_xml_path(_path)
        nq = _mjm.nq
        nv = _mjm.nv
        nu = _mjm.nu
        na = _mjm.na

        # Observation: height(1) + xquat(4) + lin_vel(3) + ang_vel(3)
        #            + joint_q(nq-root_qpos_size) + joint_v(nv-6)*0.1 + act(na)
        #            + up_z(1) + heading(1) + actions(nu)
        n_joint_q = nq - self.root_qpos_size
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
        self.n_joint_q = self.nq - self.root_qpos_size
        self.n_joint_v = self.nv - 6

        self._init_pelvis_kinematics()
        self.control_dt = self.substeps * float(self.mjm.opt.timestep)
        self.task_contract = None
        if self.model_contract == 'official':
            self._init_task_contract(target_speed)

        self._save_start_state()
        self.reset()

    def _init_task_contract(self, target_speed=1.0):
        """Bind the versioned objective and ground-contact body classification."""
        self.task_contract = MyoLegTaskContract(target_speed=target_speed)
        self.termination_height = self.task_contract.termination_height
        self._ground_geom = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, GROUND_NAME)
        if self._ground_geom < 0:
            raise ValueError(f'Official MyoLeg26 task requires ground geom {GROUND_NAME!r}')
        nonfoot = np.zeros(self.mjm.ngeom, dtype=bool)
        for geom in range(self.mjm.ngeom):
            body = int(self.mjm.geom_bodyid[geom])
            body_name = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY, body)
            name = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
            collidable = bool(self.mjm.geom_contype[geom] | self.mjm.geom_conaffinity[geom])
            # Include enabled original meshes in diagnostic collision variants;
            # the main task enables only the named body collision proxies.
            nonfoot[geom] = body != 0 and body_name not in FOOT_BODIES and (
                name.startswith(COLLISION_PREFIX) or collidable
            )
        self._nonfoot_geoms = torch.tensor(nonfoot, device=self.device)

    @torch.no_grad()
    def _nonfoot_ground_contacts(self):
        """Per-world physical contacts from the last physics solve's valid prefix.

        Warp packs contacts across worlds into one buffer. Slots after nacon
        retain old data. A contact is active geometrically when its distance is
        below includemargin, independent of stale or unused pyramid addresses.
        """
        d = self.warp_data
        count = int(wp.to_torch(d.nacon).reshape(-1)[0].item())
        capacity = d.contact.geom.shape[0]
        if count < 0 or count > capacity:
            raise RuntimeError(f'MyoLeg26 contact buffer overflow or invalid count: {count} contacts, capacity {capacity}')
        failures = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if count == 0:
            return failures
        geoms = wp.to_torch(d.contact.geom)[:count].long()
        worlds = wp.to_torch(d.contact.worldid)[:count].long()
        dist = wp.to_torch(d.contact.dist)[:count]
        margin = wp.to_torch(d.contact.includemargin)[:count]
        valid = ((worlds >= 0) & (worlds < self.num_envs)
                 & (geoms >= 0).all(dim=-1) & (geoms < self.mjm.ngeom).all(dim=-1))
        ground = (geoms == self._ground_geom).any(dim=-1)
        nonfoot = self._nonfoot_geoms[geoms.clamp(0, self.mjm.ngeom - 1)].any(dim=-1)
        failed_worlds = worlds[valid & ground & nonfoot & (dist < margin)]
        failures[failed_worlds] = True
        return failures

    @staticmethod
    def _require_finite(**fields):
        for name, value in fields.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'MyoLeg26 nonfinite {name}; refusing to continue this rollout')

    def _save_start_state(self):
        """Use the task standing keyframe, falling back to compiled qpos0.

        The task builder aligns the standing heading and clears the floor by a
        rigid root transform, preserving official internal reference positions.
        """
        kid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_KEY, 'stand')
        qpos = self.mjm.key_qpos[kid] if kid >= 0 else self.mjm.qpos0
        qvel = self.mjm.key_qvel[kid] if kid >= 0 else np.zeros(self.mjm.nv)
        self.start_qpos = torch.tensor(qpos.copy(), device=self.device, dtype=torch.float32)[None].repeat(self.num_envs, 1)
        self.start_qvel = torch.tensor(qvel.copy(), device=self.device, dtype=torch.float32)[None].repeat(self.num_envs, 1)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _init_pelvis_kinematics(self):
        """Cache compiled transforms for the explicitly selected root topology."""
        self.pelvis_body_id = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'pelvis',
        )
        bid = self.pelvis_body_id
        if bid < 0:
            raise ValueError('MyoLeg26 requires a body named pelvis')

        def tensor(array):
            return torch.tensor(np.asarray(array).copy(), device=self.device)

        if self.model_contract == 'official':
            root = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'Full Body')
            if root < 0 or self.mjm.body_parentid[root] != 0:
                raise ValueError('Official MyoLeg26 requires a world-child body named Full Body')
            j = int(self.mjm.body_jntadr[root])
            if (self.mjm.body_jntnum[root] != 1 or self.mjm.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
                    or self.mjm.jnt_qposadr[j] != 0 or self.mjm.jnt_dofadr[j] != 0):
                raise ValueError('Official MyoLeg26 requires the first joint to be the Full Body free root')
            path = []
            child = bid
            while child != root and child > 0:
                if self.mjm.body_jntnum[child] != 0:
                    raise ValueError('Official MyoLeg26 pelvis must be rigidly attached to the free root')
                path.append(child)
                child = int(self.mjm.body_parentid[child])
            if child != root:
                raise ValueError('Official MyoLeg26 pelvis must descend from Full Body')
            offset = torch.zeros((1, 3), dtype=torch.float64, device=self.device)
            rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64, device=self.device)
            for child in reversed(path):
                offset = offset + tu.quat_rotate(rotation, tensor(self.mjm.body_pos[child])[None])
                rotation = tu.quat_mul(rotation, tensor(self.mjm.body_quat[child])[None])
            self._pelvis_fixed_pos = offset
            self._pelvis_fixed_quat = rotation
        else:
            if self.mjm.body_parentid[bid] != 0:
                raise ValueError('Legacy MyoLeg26 requires pelvis to be a child of world')
            start = int(self.mjm.body_jntadr[bid])
            joints = np.arange(start, start + int(self.mjm.body_jntnum[bid]))
            expected = [mujoco.mjtJoint.mjJNT_SLIDE] * 3 + [mujoco.mjtJoint.mjJNT_HINGE] * 3
            if (not np.array_equal(self.mjm.jnt_type[joints], expected)
                    or not np.array_equal(self.mjm.jnt_qposadr[joints], np.arange(6))
                    or not np.array_equal(self.mjm.jnt_dofadr[joints], np.arange(6))):
                raise ValueError('Legacy MyoLeg26 requires three pelvis slides followed by three hinges first')
            self._pelvis_base_pos = tensor(self.mjm.body_pos[bid])
            self._pelvis_base_quat = tensor(self.mjm.body_quat[bid])
            self._pelvis_joint_axes = tensor(self.mjm.jnt_axis[joints])
            self._pelvis_joint_pos = tensor(self.mjm.jnt_pos[joints])
            self._pelvis_joint_refs = tensor(self.mjm.qpos0[:6])
        self.root_qpos_size = 7 if self.model_contract == 'official' else 6
        self.n_joint_q = self.mjm.nq - self.root_qpos_size
        self.n_joint_v = self.mjm.nv - 6
        # Anatomical local +Y is up; local +X is forward in both model contracts.
        self.up_vec = torch.tensor(
            [0.0, 1.0, 0.0], device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(self.num_envs, -1)
        self.heading_vec = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(self.num_envs, -1)

    def _compute_pelvis_xquat(self, qpos):
        """World quaternion [w, x, y, z] from the supplied state only."""
        return self._compute_pelvis_state(qpos, qpos.new_zeros((qpos.shape[0], self.mjm.nv)))[1]

    def _compute_pelvis_position(self, qpos):
        """World pelvis origin from the supplied state only."""
        return self._compute_pelvis_state(qpos, qpos.new_zeros((qpos.shape[0], self.mjm.nv)))[0]

    @staticmethod
    def _normalize_root_quat(quat):
        norm = quat.norm(dim=-1, keepdim=True)
        identity = torch.zeros_like(quat)
        identity[:, 0] = 1.0
        return torch.where(norm > 1e-12, quat / norm.clamp_min(1e-12), identity)

    def _compute_pelvis_state(self, qpos, qvel):
        """Return pelvis position, quaternion, linear and angular world velocities.

        MuJoCo free-joint translational velocity is world-frame, but angular
        velocity is in the free body's local frame. A fixed descendant's origin
        adds omega cross its rotated offset to the root translational velocity.
        """
        n = qpos.shape[0]
        if self.model_contract == 'official':
            root_quat = self._normalize_root_quat(qpos[:, 3:7])
            offset = tu.quat_rotate(root_quat, self._pelvis_fixed_pos.to(qpos).expand(n, -1))
            quat = tu.quat_mul(root_quat, self._pelvis_fixed_quat.to(qpos).expand(n, -1))
            angular = tu.quat_rotate(root_quat, qvel[:, 3:6])
            linear = qvel[:, :3] + torch.cross(angular, offset, dim=-1)
            return qpos[:, :3] + offset, quat, linear, angular

        pos = self._pelvis_base_pos.to(qpos).expand(n, -1)
        quat = self._pelvis_base_quat.to(qpos).expand(n, -1)
        linear, angular = torch.zeros_like(pos), torch.zeros_like(pos)
        axes = self._pelvis_joint_axes.to(qpos)
        anchors = self._pelvis_joint_pos.to(qpos)
        refs = self._pelvis_joint_refs.to(qpos)
        for j in range(6):
            axis = tu.quat_rotate(quat, axes[j].expand(n, -1))
            angle = qpos[:, j] - refs[j]
            if j < 3:
                shift = axis * angle[:, None]
                pos = pos + shift
                linear = linear + axis * qvel[:, j:j + 1] + torch.cross(angular, shift, dim=-1)
            else:
                old_arm = tu.quat_rotate(quat, anchors[j].expand(n, -1))
                anchor_pos = pos + old_arm
                anchor_velocity = linear + torch.cross(angular, old_arm, dim=-1)
                quat = tu.quat_mul(quat, tu.quat_from_angle_axis(angle, axes[j].expand(n, -1)))
                angular = angular + axis * qvel[:, j:j + 1]
                new_arm = tu.quat_rotate(quat, anchors[j].expand(n, -1))
                pos = anchor_pos - new_arm
                linear = anchor_velocity - torch.cross(angular, new_arm, dim=-1)
        return pos, quat, linear, angular

    @staticmethod
    def _compute_obs(qpos, qvel, act, pelvis_position, pelvis_xquat, linear_velocity,
                     angular_velocity, actions, up_vec, heading_vec, root_qpos_size):
        """Compute observation from state tensors.

        Differentiable in qpos, qvel, act, pelvis_xquat, and actions.

        Observation layout (145 values for the official model, 173 for legacy):
          height(1) + xquat(4) + lin_vel(3) + ang_vel(3)
          + joint_q(nq-root_qpos_size) + joint_v(nv-6)*0.1 + act(na)
          + up_z(1) + heading(1) + actions(nu)

        Activation is distinct from the previous action: it determines the
        current muscle force after the excitation command has changed.
        """
        height = pelvis_position[:, 2:3]
        joint_q = qpos[:, root_qpos_size:]
        joint_v = qvel[:, 6:]

        # Orientation-derived features from xquat
        up_proj = tu.quat_rotate(pelvis_xquat, up_vec)
        up_z = up_proj[:, 2:3]

        heading_proj = tu.quat_rotate(pelvis_xquat, heading_vec)
        heading = heading_proj[:, 0:1]  # x-component = forward alignment

        obs = torch.cat([
            height,               # 1
            pelvis_xquat,         # 4
            linear_velocity,      # 3, world frame at pelvis origin
            angular_velocity,     # 3, world frame
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
        # World-frame pelvis forward velocity starts at obs index 5.
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
        return self._obs_from_state(qpos, qvel, act, self.actions)

    def _obs_from_state(self, qpos, qvel, act, actions):
        position, quat, linear, angular = self._compute_pelvis_state(qpos, qvel)
        return self._compute_obs(
            qpos, qvel, act, position, quat, linear, angular, actions,
            self.up_vec, self.heading_vec, self.root_qpos_size,
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
        if self.model_contract == 'official':
            self._require_finite(actions=actions)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions.detach().clone()

        # The task charges effort on the exact excitation sent to the model.
        ctrl = excitation_from_action(actions) * self.action_strength

        if self.no_grad:
            ctrl_wp = wp.from_torch(ctrl.detach().contiguous())
            wp.copy(self.warp_data.ctrl, ctrl_wp)
            for _ in range(self.substeps):
                mjw.step(self.warp_model, self.warp_data)
            wp.synchronize()

            qpos = wp.to_torch(self.warp_data.qpos)
            qvel = wp.to_torch(self.warp_data.qvel)
            act = wp.to_torch(self.warp_data.act)
            if self.model_contract == 'official':
                self._require_finite(qpos=qpos, qvel=qvel, act=act)
            self.obs_buf = self._obs_from_state(qpos, qvel, act, actions)
            pelvis_position = self._compute_pelvis_state(qpos, qvel)[0]
            qpos_out, qvel_out, act_out = None, None, None
        else:
            qpos_in, qvel_in, act_in = self._state_inputs(qpos_in, qvel_in, act_in)
            if self.model_contract == 'official':
                self._require_finite(qpos_input=qpos_in, qvel_input=qvel_in, act_input=act_in)
            qpos_out, qvel_out, act_out = WarpSimStep.apply(ctrl, qpos_in, qvel_in, act_in, self)
            if self.model_contract == 'official':
                self._require_finite(qpos=qpos_out, qvel=qvel_out, act=act_out)
            else:
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
            self.obs_buf = self._obs_from_state(qpos_out, qvel_out, act_out, actions)
            pelvis_position = self._compute_pelvis_state(qpos_out, qvel_out)[0]

        if self.model_contract == 'official':
            self._require_finite(observation=self.obs_buf)
            reward_terms = self.task_contract.reward_terms(self.obs_buf, ctrl, self.control_dt)
            self.rew_buf = reward_terms['locomotion_reward'] - reward_terms['effort_cost']
            self._require_finite(reward=self.rew_buf)
            failure_flags = self.task_contract.failures(self.obs_buf, self.num_actions)
            failure_flags['nonfoot_ground_contact'] = self._nonfoot_ground_contacts()
            failed = torch.stack(tuple(failure_flags.values())).any(dim=0)
        else:
            reward_terms = {}
            self.rew_buf = self._compute_reward(self.obs_buf, actions)
            failed = self.obs_buf[:, 0] < self.termination_height
            failure_flags = {'height': failed}

        self.progress_buf += 1
        terminated = (failed if self.early_termination
                      else torch.zeros_like(self.progress_buf, dtype=torch.bool))
        truncated = (self.progress_buf >= self.episode_length) & ~terminated
        self.termination_buf = terminated.long()
        self.reset_buf = (terminated | truncated).long()
        self.obs_buf_before_reset = self.obs_buf.clone()
        self.extras = {
            'obs_before_reset': self.obs_buf_before_reset,
            'pelvis_position_before_reset': pelvis_position.clone(),
            'episode_end': self.termination_buf.clone(),
            'terminated': terminated,
            'truncated': truncated,
            'failure_flags': failure_flags,
            'reward_terms': {name: value.detach().clone() for name, value in reward_terms.items()},
        }

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_warp_state(env_ids)
            with torch.no_grad():
                reset_obs = self.compute_obs(
                    wp.to_torch(self.warp_data.qpos), wp.to_torch(self.warp_data.qvel),
                    wp.to_torch(self.warp_data.act),
                )
            # The policy sees a reset state immediately; terminal observations
            # and bridge outputs remain available for reward/bootstrap/BPTT.
            self.obs_buf = torch.where(self.reset_buf[:, None].bool(), reset_obs, self.obs_buf)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras, qpos_out, qvel_out, act_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_warp_state(self, env_ids):
        """Reset complete integration state without perturbing dependent joints."""
        with torch.no_grad():
            qpos_torch = wp.to_torch(self.warp_data.qpos)
            qvel_torch = wp.to_torch(self.warp_data.qvel)
            act_torch = wp.to_torch(self.warp_data.act)

            n = len(env_ids)

            qpos_torch[env_ids, :] = self.start_qpos[env_ids, :].to(qpos_torch)
            qvel_torch[env_ids, :] = self.start_qvel[env_ids, :].to(qvel_torch)
            if self.stochastic_init:
                # Only root coordinates are independently perturbed. Internal
                # coordinates include joints constrained by tendon/via equalities.
                qpos_torch[env_ids, 0] += 0.1 * (torch.rand(n, device=self.device) - 0.5)
                height_index = 2 if self.model_contract == 'official' else 1
                qpos_torch[env_ids, height_index] += 0.02 * torch.rand(n, device=self.device)
                qvel_torch[env_ids, :6] += 0.1 * (torch.rand(n, 6, device=self.device) - 0.5)
                if self.model_contract == 'official':
                    yaw = 0.1 * (torch.rand(n, device=self.device) - 0.5)
                    axis = qpos_torch.new_tensor([0.0, 0.0, 1.0]).expand(n, -1)
                    rotation = tu.quat_from_angle_axis(yaw, axis)
                    qpos_torch[env_ids, 3:7] = tu.quat_mul(rotation, qpos_torch[env_ids, 3:7])
            if self.model_contract == 'official':
                qpos_torch[env_ids, 3:7] = self._normalize_root_quat(qpos_torch[env_ids, 3:7])

            act_torch[env_ids, :] = 0.0
            wp.to_torch(self.warp_data.ctrl)[env_ids, :] = 0.0
            wp.to_torch(self.warp_data.qacc_warmstart)[env_ids, :] = 0.0
            wp.to_torch(self.warp_data.time)[env_ids] = 0.0
            self.actions[env_ids, :] = -1.0  # signed action for zero excitation

        self.progress_buf[env_ids] = 0

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None:
            if force_reset:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            self._reset_warp_state(env_ids)
            self.reset_buf[env_ids] = 0
            self.termination_buf[env_ids] = 0

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
