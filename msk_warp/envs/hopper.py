"""Hopper locomotion environments on MuJoCo Warp, motor- and muscle-driven.

Ported from DiffRL's ``envs/hopper.py`` (dflex). The port keeps DiffRL's joint layout, so the
observation, reward and termination definitions transfer element for element::

    qpos = [x, z, pitch, thigh, leg, foot]      qvel = [vx, vz, omega, dthigh, dleg, dfoot]
    obs  = [qpos[1:], qvel]                     (11 dims; the muscle variant appends act)

``rootz`` carries no ``ref`` in either asset, so the torso sits at its XML height of 1.25 m when
``qpos[1] = 0`` and ``qpos[1]`` is height relative to the start pose. That is the quantity DiffRL's
``termination_height = -0.45`` and its height reward are written against, and it starts at exactly
0 as in DiffRL (``start_height = 0.0``). DiffRL's own asset writes ``ref="1.25"``, which its dflex
loader ignores; honouring it in MuJoCo would silently turn ``qpos[1]`` into absolute height.

Differences from DiffRL, all forced by the engine change from dflex to MuJoCo:

  * Contacts are MuJoCo constraints, not dflex penalty springs (``contact_ke=2e4``). The
    post-mortem's finding on the ant was that MuJoCo constraint contacts can create a statically
    stable equilibrium the agent never falls out of, so the fall rate is measured before training
    rather than assumed (``scripts/measure_fall_rate.py``).
  * Semi-implicit Euler at 1/240 s with 4 substeps per 1/60 s control step, in place of dflex's
    16 substeps of its own integrator. DiffRL's own asset asks for RK4, which its loader ignores.
  * Reset noise can place ``thigh`` and ``leg`` up to 0.05 rad past their upper limit of 0,
    exactly as in DiffRL. MuJoCo resolves this with a soft limit constraint over the first few
    steps rather than dflex's limit spring; the noise is left unclamped so that the start
    distribution stays identical.

The two variants differ only in how force reaches the joints. ``HopperMotorEnv`` drives three
torque motors at ``gear=200``; ``HopperMuscleEnv`` drives six Hill-type muscles whose peak forces
are calibrated to the same 200 N.m of joint torque (``scripts/calibrate_hopper_muscle.py``). For
the muscle variant the control reaches the dynamics only through the activation state, so ``act``
is both part of the observation and the carrier of the whole control gradient.
"""

import math

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from msk_warp import resolve_model_path
from msk_warp.bridge import WarpSimStep
from msk_warp.envs.base_env import MjWarpEnv
from msk_warp.utils.gait import periodicity_residual


class HopperBaseEnv(MjWarpEnv):
    """Shared hopper dynamics, observation, reward and termination.

    Subclasses supply the action-to-control map and the quantity the action penalty is charged on.
    """

    #: Whether the activation state is appended to the observation. Set by subclasses.
    obs_includes_act = False

    #: Extra observation columns a subclass appends after the state (e.g. a gait phase).
    num_extra_obs = 0

    def __init__(
        self,
        num_envs=64,
        device='cuda:0',
        episode_length=1000,
        no_grad=False,
        stochastic_init=True,
        substeps=4,
        model_path='assets/hopper_motor.xml',
        num_act=3,
        early_termination=True,
        terminate_on_angle=False,
        njmax=128,
        action_strength=1.0,
        height_rew_scale=1.0,
        angle_rew_scale=1.0,
        action_penalty=-1e-1,
        **kwargs,
    ):
        self.num_joint_q = 6
        self.num_joint_qd = 6
        num_obs = (self.num_joint_q - 1) + self.num_joint_qd
        if self.obs_includes_act:
            num_obs += num_act
        num_obs += self.num_extra_obs

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
        self.early_termination = early_termination
        self.terminate_on_angle = terminate_on_angle
        self.action_strength = action_strength

        # DiffRL envs/hopper.py:45-51
        self.termination_height = -0.45
        self.termination_angle = math.pi / 6.0
        self.termination_height_tolerance = 0.15
        self.termination_angle_tolerance = 0.05
        self.height_rew_scale = height_rew_scale
        # DiffRL fixes the posture weights at 1.0. They are exposed here because a
        # velocity-dominant reward is the pre-declared response to a standing basin, and the
        # muscle hopper turned out to have one (docs/research/phase3-hopper/protocol.md).
        self.angle_rew_scale = angle_rew_scale
        self.action_penalty = action_penalty

        self._save_start_state()
        self.reset()

    def _save_start_state(self):
        """DiffRL's fixed start: the asset rest pose, at rest, with qpos[1] == 0."""
        wp.synchronize()
        self.start_qpos = torch.zeros(
            (self.num_envs, self.num_joint_q), device=self.device, dtype=torch.float32,
        )
        self.start_qvel = torch.zeros(
            (self.num_envs, self.num_joint_qd), device=self.device, dtype=torch.float32,
        )

    # ------------------------------------------------------------ observations and reward

    @staticmethod
    def _compute_obs(qpos, qvel, act=None):
        """DiffRL ``calculateObservations``: drop the x coordinate, keep everything else."""
        parts = [qpos[:, 1:], qvel]
        if act is not None and act.shape[-1] > 0:
            parts.append(act)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _compute_reward(obs, effort, termination_height, termination_height_tolerance,
                        termination_angle, height_rew_scale, angle_rew_scale, action_penalty):
        """DiffRL ``calculateReward`` (envs/hopper.py:262-272).

        ``effort`` is the quantity the action penalty is charged on: raw actions for the motor
        variant, activation commands for the muscle variant. The two posture scales are 1.0 in
        DiffRL; lowering them makes forward progress dominate the return. Note that only the
        positive branch of the height term is scaled, so a fall is punished either way.
        """
        height_diff = obs[:, 0] - (termination_height + termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, 0.3)
        height_reward = torch.where(
            height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward,
        )
        height_reward = torch.where(
            height_reward > 0.0, height_rew_scale * height_reward, height_reward,
        )

        angle_reward = angle_rew_scale * (-obs[:, 1] ** 2 / (termination_angle ** 2) + 1.0)
        progress_reward = obs[:, 5]

        return (
            progress_reward
            + height_reward
            + angle_reward
            + torch.sum(effort ** 2, dim=-1) * action_penalty
        )

    def compute_obs(self, qpos, qvel, act=None):
        """Instance wrapper used by SHAC to recompute obs from the tracked state."""
        return self._compute_obs(qpos, qvel, act if self.obs_includes_act else None)

    def _reward(self, obs, actions, ctrl):
        return self._compute_reward(
            obs, self._effort(actions, ctrl),
            self.termination_height, self.termination_height_tolerance,
            self.termination_angle, self.height_rew_scale, self.angle_rew_scale,
            self.action_penalty,
        )

    # ------------------------------------------------------------ actuation (subclass hooks)

    def _to_ctrl(self, actions):
        """Map policy actions in [-1, 1] onto the model's ctrl range."""
        raise NotImplementedError

    def _effort(self, actions, ctrl):
        """Quantity the action penalty is charged on."""
        raise NotImplementedError

    def passive_action(self):
        """The action that maps to zero control, i.e. zero torque or zero activation."""
        return torch.zeros(self.num_envs, self.num_actions, device=self.device)

    def _advance_phase(self):
        """Hook for phase-indexed subclasses; no phase exists in the base environment."""

    # ------------------------------------------------------------ stepping

    def step(self, actions, qpos_in=None, qvel_in=None, act_in=None):
        """Run one control step.

        Returns ``obs, rew, done, extras, qpos_out, qvel_out, act_out``.
        """
        actions = actions.view(self.num_envs, self.num_actions)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions.detach().clone()

        ctrl = self._to_ctrl(actions)

        # The policy has acted on the state at the current phase; everything computed below
        # describes the state one step later, so the phase advances here, before the observation
        # and reward are formed. A base env has no phase and this is a no-op.
        self._advance_phase()

        if self.no_grad:
            wp.copy(self.warp_data.ctrl, wp.from_torch(ctrl.detach().contiguous()))
            for _ in range(self.substeps):
                mjw.step(self.warp_model, self.warp_data)
            wp.synchronize()

            qpos, qvel, act = self.state_tensors()
            self.obs_buf = self.compute_obs(qpos, qvel, act)
            self.rew_buf = self._reward(self.obs_buf, actions, ctrl)
            qpos_out, qvel_out, act_out = None, None, None
        else:
            qpos_in, qvel_in, act_in = self._state_inputs(qpos_in, qvel_in, act_in)
            qpos_out, qvel_out, act_out = WarpSimStep.apply(ctrl, qpos_in, qvel_in, act_in, self)

            # Bound the state so a diverging rollout cannot poison the tape with inf or NaN.
            qpos_out = qpos_out.clamp(-100.0, 100.0)
            qvel_out = qvel_out.clamp(-100.0, 100.0)

            self.obs_buf = self.compute_obs(qpos_out, qvel_out, act_out)
            self.rew_buf = self._reward(self.obs_buf, actions, ctrl)

        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.progress_buf += 1

        # DiffRL terminates on height only. The angle test is a pre-declared response to a
        # standing basin and stays off by default.
        if self.early_termination:
            fell = self.obs_buf[:, 0] < self.termination_height
            if self.terminate_on_angle:
                fell = fell | (self.obs_buf[:, 1].abs() > self.termination_angle)
            self.termination_buf = torch.where(
                fell, torch.ones_like(self.termination_buf), torch.zeros_like(self.termination_buf),
            )
            self.reset_buf = torch.where(
                self.termination_buf > 0, torch.ones_like(self.reset_buf), self.reset_buf,
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

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras, qpos_out, qvel_out, act_out

    # ------------------------------------------------------------ reset

    def _reset_warp_state(self, env_ids):
        """DiffRL reset noise (envs/hopper.py:203-207), applied in place without gradient."""
        with torch.no_grad():
            qpos_torch = wp.to_torch(self.warp_data.qpos)
            qvel_torch = wp.to_torch(self.warp_data.qvel)

            qpos_torch[env_ids, :] = self.start_qpos[env_ids, :].clone()
            qvel_torch[env_ids, :] = self.start_qvel[env_ids, :].clone()

            if self.stochastic_init:
                n = len(env_ids)
                qpos_torch[env_ids, 0:2] += 0.05 * (
                    torch.rand(n, 2, device=self.device) - 0.5
                ) * 2.0
                qpos_torch[env_ids, 2] = (torch.rand(n, device=self.device) - 0.5) * 0.1
                qpos_torch[env_ids, 3:] += 0.05 * (
                    torch.rand(n, self.num_joint_q - 3, device=self.device) - 0.5
                ) * 2.0
                qvel_torch[env_ids, :] = 0.05 * (
                    torch.rand(n, self.num_joint_qd, device=self.device) - 0.5
                ) * 2.0

            if self.has_act:
                wp.to_torch(self.warp_data.act)[env_ids, :] = 0.0

            self.actions[env_ids, :] = 0.0

        self.progress_buf[env_ids] = 0

    def reset(self, env_ids=None, force_reset=True):
        if env_ids is None and force_reset:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            self._reset_warp_state(env_ids)
            self.calculateObservations()

        return self.obs_buf

    def calculateObservations(self):
        """Non-differentiable obs, used by ``reset`` and ``initialize_trajectory``."""
        wp.synchronize()
        qpos, qvel, act = self.state_tensors()
        self.obs_buf = self.compute_obs(qpos, qvel, act)

    def calculateReward(self):
        """Reward is computed inside ``step`` on the differentiable state."""


class HopperMotorEnv(HopperBaseEnv):
    """Three torque motors at ``gear=200``; actions in [-1, 1] pass straight through to ctrl."""

    obs_includes_act = False

    def __init__(self, model_path='assets/hopper_motor.xml', **kwargs):
        super().__init__(model_path=model_path, num_act=3, **kwargs)

    def _to_ctrl(self, actions):
        return actions * self.action_strength

    def _effort(self, actions, ctrl):
        return actions


class HopperMuscleEnv(HopperBaseEnv):
    """Six Hill-type muscles; actions in [-1, 1] map onto activation commands in [0, 1].

    ``ctrl`` reaches the dynamics only through ``act_dot -> act``, so the observation carries
    ``act`` and the action penalty is charged on the activation command rather than on the raw
    action. Charging it on the action would penalise the zero-activation corner (``a = -1``) the
    most, which inverts the intended effort incentive.
    """

    obs_includes_act = True

    def __init__(self, model_path='assets/hopper_muscle.xml', **kwargs):
        super().__init__(model_path=model_path, num_act=6, **kwargs)

    def _to_ctrl(self, actions):
        return 0.5 * (actions + 1.0) * self.action_strength

    def _effort(self, actions, ctrl):
        return ctrl

    def passive_action(self):
        """Zero activation sits at the bottom of the action range, not at its centre."""
        return -torch.ones(self.num_envs, self.num_actions, device=self.device)


class HopperMuscleTrackEnv(HopperMuscleEnv):
    """Muscle hopper rewarded for tracking a reference gait cycle, phase by phase.

    Phase 3 failed this model under seven reward, optimiser and actuator settings, and the reason
    was a reward landscape with two attractors: standing collected 98.7% of the return, and
    whenever the hopper could move the nearest optimum was a lunge. Phase 4 then found a forward
    periodic trajectory by trajectory optimisation, which says the landscape rather than the
    model's capability is what blocked the optimisers.

    A tracking reward removes both attractors at once, because neither standing nor diving
    resembles the reference. It is the pre-registered response to Phase 4's pass.

    Three choices, each with a reason:

    * **Only the mechanical state is tracked** (``qpos[1:]`` and ``qvel``, the same 11-component
      ruler Phase 4 scored on). The reference's *activation* does not close over the cycle -- the
      knee antagonists effectively swap, with per-muscle closing errors up to 0.886 -- so a
      phase-indexed activation target would demand a jump at the phase wrap that no muscle can
      follow, at exactly the deactivation limit Phase 3 measured. Activation is left to the policy,
      which is also what kinematic imitation methods do.
    * **The phase is in the observation**, as ``(sin, cos)``. Without it the policy would have to
      infer where in the cycle it is, and Phase 3 noted the observation carries no clock while the
      plant needs about six steps of anticipation.
    * **Episodes start on the reference orbit at a random phase.** The rest pose is not on the
      orbit, so from there a policy must solve a transient before it can track anything. The cost
      is that the return is no longer comparable with Phase 3's numbers; velocity and fall rate,
      which are the gate, still are.
    """

    #: ``(sin, cos)`` of the gait phase.
    num_extra_obs = 2

    #: Number of leading observation columns forming the mechanical shape vector.
    shape_dims = 11

    def __init__(
        self,
        reference_path='assets/references/hopper_muscle_T16_gait.npz',
        track_sharpness=2.0,
        task_weight=0.0,
        reset_state_noise=0.02,
        num_envs=64,
        device='cuda:0',
        **kwargs,
    ):
        reference = np.load(resolve_model_path(reference_path))
        self.reference_path = reference_path
        self.cycle = int(reference['cycle'])
        self.track_sharpness = float(track_sharpness)
        self.task_weight = float(task_weight)
        self.reset_state_noise = float(reset_state_noise)

        to_torch = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.ref_qpos = to_torch(reference['qpos'])
        self.ref_qvel = to_torch(reference['qvel'])
        self.ref_act = to_torch(reference['act'])
        # x is excluded because a gait translates; this is the ruler Phase 4's residual used, and
        # it travels with the asset so the environment does not carry a magic constant.
        self.ref_shape = torch.cat([self.ref_qpos[:, 1:], self.ref_qvel], dim=-1)
        self.track_scales = to_torch(reference['shape_scales'])
        if self.ref_shape.shape[-1] != self.shape_dims or self.track_scales.shape[-1] != self.shape_dims:
            raise ValueError(
                f"reference {reference_path} has a {self.ref_shape.shape[-1]}-component shape and "
                f"{self.track_scales.shape[-1]} scales; this env tracks {self.shape_dims}"
            )

        self.phase_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        angle = 2.0 * math.pi * torch.arange(self.cycle, device=device, dtype=torch.float32) / self.cycle
        self.phase_features = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

        super().__init__(num_envs=num_envs, device=device, **kwargs)

    # ------------------------------------------------------------ phase and observation

    def _advance_phase(self):
        self.phase_buf = (self.phase_buf + 1) % self.cycle

    def compute_obs(self, qpos, qvel, act=None):
        base = self._compute_obs(qpos, qvel, act if self.obs_includes_act else None)
        return torch.cat([base, self.phase_features[self.phase_buf]], dim=-1)

    # ------------------------------------------------------------ reward

    def tracking_error(self, obs):
        """Scaled RMS distance from the reference at each world's current phase.

        The mechanical shape vector is the leading ``shape_dims`` observation columns by
        construction (``qpos[1:]`` then ``qvel``), so this reads straight off the observation and
        stays differentiable for SHAC.
        """
        return periodicity_residual(
            obs[:, :self.shape_dims], self.ref_shape[self.phase_buf], self.track_scales,
        )

    def _reward(self, obs, actions, ctrl):
        reward = torch.exp(-self.track_sharpness * self.tracking_error(obs))
        if self.task_weight != 0.0:
            reward = reward + self.task_weight * super()._reward(obs, actions, ctrl)
        return reward

    # ------------------------------------------------------------ reset

    def _reset_warp_state(self, env_ids):
        """Reference-state initialisation: begin on the orbit at a random phase."""
        with torch.no_grad():
            n = len(env_ids)
            if self.stochastic_init:
                phase = torch.randint(self.cycle, (n,), device=self.device)
            else:
                phase = torch.zeros(n, dtype=torch.long, device=self.device)
            self.phase_buf[env_ids] = phase

            qpos_torch = wp.to_torch(self.warp_data.qpos)
            qvel_torch = wp.to_torch(self.warp_data.qvel)
            qpos_torch[env_ids, :] = self.ref_qpos[phase]
            qpos_torch[env_ids, 0] = 0.0  # every world starts at the track origin
            qvel_torch[env_ids, :] = self.ref_qvel[phase]

            if self.stochastic_init and self.reset_state_noise > 0.0:
                spread = self.reset_state_noise
                qpos_torch[env_ids, 1:] += spread * (
                    torch.rand(n, self.num_joint_q - 1, device=self.device) - 0.5) * 2.0
                qvel_torch[env_ids, :] += spread * (
                    torch.rand(n, self.num_joint_qd, device=self.device) - 0.5) * 2.0

            # Reference-state initialisation includes the actuator state; only the tracking
            # *target* excludes activation.
            if self.has_act:
                wp.to_torch(self.warp_data.act)[env_ids, :] = self.ref_act[phase]

            self.actions[env_ids, :] = 0.0

        self.progress_buf[env_ids] = 0
