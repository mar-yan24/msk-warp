"""Tests for the explicit friction-surrogate adjoint branch."""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

wp.init()

from msk_warp.bridge import WarpSimStep
from msk_warp.envs.ant import AntEnv


DEVICE = "cuda:0"


def _make_env(model_path="assets/ant_soft.xml", substeps=4, **kwargs):
    return AntEnv(
        num_envs=1,
        device=DEVICE,
        stochastic_init=False,
        substeps=substeps,
        model_path=model_path,
        smooth_adjoint=True,
        **kwargs,
    )


def _settle_env(env, steps=100, ctrl_val=0.3):
    with torch.no_grad():
        ctrl = torch.full((env.num_envs, env.num_actions), ctrl_val, device=DEVICE)
        ctrl_wp = wp.from_torch(ctrl.contiguous())
        for _ in range(steps):
            wp.copy(env.warp_data.ctrl, ctrl_wp)
            wp.synchronize()
            mjw.step(env.warp_model, env.warp_data)
        wp.synchronize()


def _reward_grad(env, ctrl):
    env.clear_grad()
    qpos0 = wp.to_torch(env.warp_data.qpos).clone()
    qvel0 = wp.to_torch(env.warp_data.qvel).clone()

    ctrl_torch = ctrl.clone().requires_grad_(True)
    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos0, qvel0, env)
    qpos_out = qpos_out.clamp(-100.0, 100.0)
    qvel_out = qvel_out.clamp(-100.0, 100.0)

    actions = torch.zeros(env.num_envs, env.num_actions, device=DEVICE)
    obs = AntEnv._compute_obs(
        qpos_out,
        qvel_out,
        actions,
        env.targets,
        env.up_vec,
        env.heading_vec,
        env.joint_vel_obs_scaling,
    )
    rew = AntEnv._compute_reward(
        obs,
        actions,
        env.action_penalty,
        env.forward_vel_weight,
        env.heading_weight,
        env.up_weight,
        env.height_weight,
        env.joint_vel_penalty,
        env.push_reward_weight,
    )
    rew.sum().backward()
    return ctrl_torch.grad.detach().cpu().numpy().copy()


def test_surrogate_does_not_affect_forward_physics():
    ctrl = torch.full((1, 8), 0.2, device=DEVICE)

    env_smooth = _make_env()
    env_surrogate = _make_env(
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.9,
    )

    qpos0 = wp.to_torch(env_smooth.warp_data.qpos).clone()
    qvel0 = wp.to_torch(env_smooth.warp_data.qvel).clone()
    wp.copy(env_surrogate.warp_data.qpos, wp.from_torch(qpos0.contiguous()))
    wp.copy(env_surrogate.warp_data.qvel, wp.from_torch(qvel0.contiguous()))
    wp.synchronize()

    ctrl_wp = wp.from_torch(ctrl.contiguous())
    wp.copy(env_smooth.warp_data.ctrl, ctrl_wp)
    wp.copy(env_surrogate.warp_data.ctrl, ctrl_wp)
    wp.synchronize()

    for _ in range(env_smooth.substeps):
        mjw.step(env_smooth.warp_model, env_smooth.warp_data)
        mjw.step(env_surrogate.warp_model, env_surrogate.warp_data)
    wp.synchronize()

    qpos_diff = np.max(np.abs(env_smooth.warp_data.qpos.numpy() - env_surrogate.warp_data.qpos.numpy()))
    qvel_diff = np.max(np.abs(env_smooth.warp_data.qvel.numpy() - env_surrogate.warp_data.qvel.numpy()))

    assert qpos_diff < 1e-6
    assert qvel_diff < 1e-6


def test_surrogate_produces_nonzero_finite_reward_gradient():
    torch.manual_seed(42)
    ctrl = torch.randn(1, 8, device=DEVICE) * 0.3

    env = _make_env(
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.9,
    )
    _settle_env(env)

    grad = _reward_grad(env, ctrl)
    grad_norm = np.linalg.norm(grad)

    assert np.isfinite(grad).all()
    assert grad_norm > 1e-6


def test_surrogate_reduces_soft_contact_free_body_gradient_scale():
    torch.manual_seed(42)
    ctrl = torch.randn(1, 8, device=DEVICE) * 0.3

    env_free = _make_env(free_body_adjoint=True)
    env_sur = _make_env(
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.9,
    )
    _settle_env(env_free)

    qpos = wp.to_torch(env_free.warp_data.qpos).clone()
    qvel = wp.to_torch(env_free.warp_data.qvel).clone()
    wp.copy(env_sur.warp_data.qpos, wp.from_torch(qpos.contiguous()))
    wp.copy(env_sur.warp_data.qvel, wp.from_torch(qvel.contiguous()))
    wp.synchronize()

    grad_free = _reward_grad(env_free, ctrl)
    grad_sur = _reward_grad(env_sur, ctrl)

    norm_free = np.linalg.norm(grad_free)
    norm_sur = np.linalg.norm(grad_sur)

    assert norm_sur > 1e-6
    assert norm_sur < norm_free


def test_surrogate_keeps_16_substep_gradient_below_free_body():
    torch.manual_seed(42)
    ctrl = torch.randn(1, 8, device=DEVICE) * 0.3

    env_free4 = _make_env(substeps=4, free_body_adjoint=True)
    env_free16 = _make_env(substeps=16, free_body_adjoint=True)
    env_sur4 = _make_env(
        substeps=4,
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.9,
    )
    env_sur16 = _make_env(
        substeps=16,
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.9,
    )

    _settle_env(env_free4)
    qpos = wp.to_torch(env_free4.warp_data.qpos).clone()
    qvel = wp.to_torch(env_free4.warp_data.qvel).clone()
    for env in (env_free16, env_sur4, env_sur16):
        wp.copy(env.warp_data.qpos, wp.from_torch(qpos.contiguous()))
        wp.copy(env.warp_data.qvel, wp.from_torch(qvel.contiguous()))
    wp.synchronize()

    free4 = np.linalg.norm(_reward_grad(env_free4, ctrl))
    free16 = np.linalg.norm(_reward_grad(env_free16, ctrl))
    sur16 = np.linalg.norm(_reward_grad(env_sur16, ctrl))
    assert free4 > 1e-6
    assert sur16 < free16
