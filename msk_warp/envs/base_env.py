"""Base MuJoCo Warp environment for differentiable RL."""

import os

import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw


class MjWarpEnv:
    """Base class for MuJoCo Warp environments used with SHAC."""

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
    ):
        self.device = device
        self.num_environments = num_envs
        self.num_observations = num_obs
        self.num_actions = num_act
        self.episode_length = episode_length
        self.no_grad = no_grad
        self.substeps = substeps

        # Load MuJoCo model
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path)
        self.mjm = mujoco.MjModel.from_xml_path(model_path)

        # Determine njmax for contact constraint buffers (per world)
        self._njmax = njmax
        diff_data_kwargs = {}
        if njmax is not None:
            diff_data_kwargs['njmax'] = njmax

        # Create Warp model and data
        self.warp_model = mjw.put_model(self.mjm)
        self.warp_data = mjw.make_diff_data(self.mjm, nworld=num_envs, **diff_data_kwargs)
        mjw.reset_data(self.warp_model, self.warp_data)

        # Run kinematics once to initialize derived quantities
        mjw.step(self.warp_model, self.warp_data)
        mjw.reset_data(self.warp_model, self.warp_data)
        wp.synchronize()

        # Allocate PyTorch buffers
        self.obs_buf = torch.zeros(
            (num_envs, num_obs), device=device, dtype=torch.float32
        )
        self.rew_buf = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.reset_buf = torch.ones(num_envs, device=device, dtype=torch.long)
        self.termination_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.progress_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.actions = torch.zeros(
            (num_envs, num_act), device=device, dtype=torch.float32
        )
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

    def clear_grad(self):
        """Cut off the gradient graph by recreating diff data and restoring state."""
        with torch.no_grad():
            # Save current state
            qpos_save = wp.clone(self.warp_data.qpos)
            qvel_save = wp.clone(self.warp_data.qvel)
            time_save = wp.clone(self.warp_data.time)

        # Recreate diff data (fresh gradient graph)
        diff_data_kwargs = {}
        if self._njmax is not None:
            diff_data_kwargs['njmax'] = self._njmax
        self.warp_data = mjw.make_diff_data(self.mjm, nworld=self.num_environments, **diff_data_kwargs)
        mjw.reset_data(self.warp_model, self.warp_data)

        # Restore state
        wp.copy(self.warp_data.qpos, qpos_save)
        wp.copy(self.warp_data.qvel, qvel_save)
        wp.copy(self.warp_data.time, time_save)
        wp.synchronize()

    def initialize_trajectory(self):
        """Cut gradient graph and return initial observations."""
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf

    def step(self, actions):
        raise NotImplementedError

    def reset(self, env_ids=None, force_reset=True):
        raise NotImplementedError

    def calculateObservations(self):
        raise NotImplementedError

    def calculateReward(self):
        raise NotImplementedError
