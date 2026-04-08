"""Regression tests for AntEnv smooth-adjoint parameter wiring."""

import warnings

import warp as wp

warnings.filterwarnings("ignore")
wp.init()

from msk_warp.envs.ant import AntEnv


def _make_env(**kwargs):
    return AntEnv(
        num_envs=1,
        device="cuda:0",
        stochastic_init=False,
        substeps=4,
        smooth_adjoint=True,
        **kwargs,
    )


def _assert_smooth_params(
    env, bypass_kf, free_body, penalty_alpha, surrogate, surrogate_alpha
):
    assert env.friction_bypass_kf == bypass_kf
    assert env.free_body_adjoint is free_body
    assert env.penalty_damping_alpha == penalty_alpha
    assert env.friction_surrogate_adjoint is surrogate
    assert env.friction_surrogate_alpha == surrogate_alpha

    data = env.warp_data
    assert getattr(data, "smooth_friction_bypass_kf", None) == bypass_kf
    assert getattr(data, "smooth_free_body_adjoint", None) is free_body
    assert getattr(data, "smooth_penalty_damping_alpha", None) == penalty_alpha
    assert getattr(data, "smooth_friction_surrogate_adjoint", None) is surrogate
    assert getattr(data, "smooth_friction_surrogate_alpha", None) == surrogate_alpha


def test_ant_env_forwards_smooth_adjoint_branch_params():
    env = _make_env(
        friction_bypass_kf=1.5,
        free_body_adjoint=True,
        penalty_damping_alpha=0.2,
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.3,
    )

    _assert_smooth_params(
        env,
        bypass_kf=1.5,
        free_body=True,
        penalty_alpha=0.2,
        surrogate=True,
        surrogate_alpha=0.3,
    )


def test_ant_env_clear_grad_preserves_smooth_adjoint_branch_params():
    env = _make_env(
        friction_bypass_kf=0.75,
        free_body_adjoint=False,
        penalty_damping_alpha=0.1,
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.15,
    )

    env.clear_grad()

    _assert_smooth_params(
        env,
        bypass_kf=0.75,
        free_body=False,
        penalty_alpha=0.1,
        surrogate=True,
        surrogate_alpha=0.15,
    )
