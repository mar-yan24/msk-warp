"""Base MuJoCo Warp environment for differentiable RL."""

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from msk_warp import backend, resolve_model_path
from msk_warp.bridge import MODES, _state_tensors


class MjWarpEnv:
    """Base class for MuJoCo Warp environments used with SHAC.

    State carried through the gradient bridge is ``(qpos, qvel, act)``; ``act`` is empty for
    models without activation dynamics. Subclasses implement ``step(actions, qpos_in, qvel_in,
    act_in) -> (obs, rew, done, extras, qpos_out, qvel_out, act_out)``.
    """

    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        model_path,
        device='cuda:0',
        no_grad=False,
        substeps=4,
        njmax=None,
        backward_mode=None,
        use_fd_jacobian=False,
        tape_per_substep=False,
        rerun_after_backward=True,
        grad_contract='strict',
        fd_eps=1e-3,
        clear_grad_rebuild=False,
    ):
        self.device = device
        self.num_environments = num_envs
        self.num_observations = num_obs
        self.num_actions = num_act
        self.episode_length = episode_length
        self.no_grad = no_grad
        self.substeps = substeps
        # Backward mode: explicit `backward_mode` wins; the two legacy flags map onto it.
        if backward_mode is None:
            backward_mode = 'fd' if use_fd_jacobian else ('tape_per_substep' if tape_per_substep else 'tape')
        if backward_mode not in MODES:
            raise ValueError(f"backward_mode must be one of {MODES}, got {backward_mode!r}")
        self.backward_mode = backward_mode
        self.use_fd_jacobian = backward_mode == 'fd'
        self.tape_per_substep = backward_mode == 'tape_per_substep'
        self.rerun_after_backward = rerun_after_backward
        self.fd_eps = fd_eps
        # Per-epoch graph cut: zero the gradient buffers (default) or rebuild Data from scratch,
        # which also resets solver warm-start state. Kept switchable to A/B training dynamics.
        self.clear_grad_rebuild = clear_grad_rebuild
        self._njmax = njmax

        model_path = resolve_model_path(model_path)
        self.mjm = mujoco.MjModel.from_xml_path(model_path)

        self.warp_model = backend.put_model(self.mjm)
        self.warp_data = backend.make_data(self.mjm, self.warp_model, num_envs, njmax, grad=not no_grad)
        mjw.reset_data(self.warp_model, self.warp_data)

        # Gradient contract: refuse models whose gradients would be silently wrong.
        self.grad_contract_report = None
        if not no_grad and grad_contract != 'off':
            self.grad_contract_report = backend.assert_grad_contract(
                self.mjm, self.warp_model, self.warp_data, strict=(grad_contract == 'strict'), njmax=njmax,
            )

        # Run one step to initialize derived quantities, then reset.
        mjw.step(self.warp_model, self.warp_data)
        mjw.reset_data(self.warp_model, self.warp_data)
        wp.synchronize()

        self.obs_buf = torch.zeros((num_envs, num_obs), device=device, dtype=torch.float32)
        self.rew_buf = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.reset_buf = torch.ones(num_envs, device=device, dtype=torch.long)
        self.termination_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.progress_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.actions = torch.zeros((num_envs, num_act), device=device, dtype=torch.float32)
        self.extras = {}

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_obs(self):
        return self.num_observations

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def has_act(self):
        return self.warp_data.act.size > 0

    def state_tensors(self):
        """Detached copies of the current ``(qpos, qvel, act)`` as torch tensors."""
        return _state_tensors(self.warp_data)

    def clear_grad(self, rebuild=None):
        """Per-epoch graph cut: zero the Warp gradient buffers.

        Every backward pass records and disposes its own tape, so nothing accumulates across
        epochs on the Warp side. ``rebuild=True`` reproduces the old behaviour (fresh Data with
        the state copied over); it exists so a test can show both give the same gradients.
        """
        if rebuild is None:
            rebuild = self.clear_grad_rebuild
        if not rebuild:
            backend.zero_grad(self.warp_data)
            return
        with torch.no_grad():
            qpos_save = wp.clone(self.warp_data.qpos)
            qvel_save = wp.clone(self.warp_data.qvel)
            time_save = wp.clone(self.warp_data.time)
            act_save = wp.clone(self.warp_data.act) if self.has_act else None
        self.warp_data = backend.make_data(self.mjm, self.warp_model, self.num_environments, self._njmax, grad=not self.no_grad)
        mjw.reset_data(self.warp_model, self.warp_data)
        wp.copy(self.warp_data.qpos, qpos_save)
        wp.copy(self.warp_data.qvel, qvel_save)
        wp.copy(self.warp_data.time, time_save)
        if act_save is not None:
            wp.copy(self.warp_data.act, act_save)
        wp.synchronize()

    def initialize_trajectory(self):
        """Cut gradient graph and return initial observations."""
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def begin_epoch(self, epoch: int, max_epochs: int) -> dict[str, float]:
        """Optional epoch hook for env-specific schedules."""
        return {}

    def _state_inputs(self, qpos_in, qvel_in, act_in):
        """Fill in any missing differentiable state input from the current Data."""
        if qpos_in is None or qvel_in is None or act_in is None:
            qpos_cur, qvel_cur, act_cur = self.state_tensors()
            qpos_in = qpos_cur if qpos_in is None else qpos_in
            qvel_in = qvel_cur if qvel_in is None else qvel_in
            act_in = act_cur if act_in is None else act_in
        return qpos_in, qvel_in, act_in

    def step(self, actions, qpos_in=None, qvel_in=None, act_in=None):
        raise NotImplementedError

    def reset(self, env_ids=None, force_reset=True):
        raise NotImplementedError

    def calculateObservations(self):
        raise NotImplementedError

    def calculateReward(self):
        raise NotImplementedError
