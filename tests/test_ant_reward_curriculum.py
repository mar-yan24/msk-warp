"""Unit tests for Ant reward curriculum helpers."""

from __future__ import annotations

import pytest

from msk_warp.envs.ant import compute_reward_curriculum_state
from msk_warp.envs.ant import compute_reset_curriculum_state
from msk_warp.envs.ant import get_reset_joint_bias
from msk_warp.envs.ant import RESET_MODE_CANONICAL
from msk_warp.envs.ant import RESET_MODE_STRIDE_LEFT
from msk_warp.envs.ant import RESET_MODE_STRIDE_RIGHT


def test_reward_curriculum_disabled_keeps_base_weights():
    state = compute_reward_curriculum_state(
        {"enabled": False},
        epoch=0,
        base_forward_vel_weight=1.0,
        base_heading_weight=1.0,
        base_height_weight=1.0,
        base_up_weight=0.1,
    )
    assert state["progress"] == pytest.approx(1.0)
    assert state["target_speed"] == pytest.approx(0.0)
    assert state["low_speed_penalty_weight"] == pytest.approx(0.0)
    assert state["forward_vel_weight"] == pytest.approx(1.0)
    assert state["heading_weight"] == pytest.approx(1.0)
    assert state["height_weight"] == pytest.approx(1.0)
    assert state["up_weight"] == pytest.approx(0.1)


def test_reward_curriculum_epoch_zero_uses_initial_scales():
    state = compute_reward_curriculum_state(
        {
            "enabled": True,
            "anneal_epochs": 20,
            "target_speed": 0.5,
            "low_speed_penalty_weight_init": 1.0,
            "forward_scale_init": 3.0,
            "heading_scale_init": 0.4,
            "height_scale_init": 0.4,
            "up_scale_init": 0.5,
        },
        epoch=0,
        base_forward_vel_weight=1.0,
        base_heading_weight=1.0,
        base_height_weight=1.0,
        base_up_weight=0.1,
    )
    assert state["progress"] == pytest.approx(0.0)
    assert state["target_speed"] == pytest.approx(0.5)
    assert state["low_speed_penalty_weight"] == pytest.approx(1.0)
    assert state["forward_vel_weight"] == pytest.approx(3.0)
    assert state["heading_weight"] == pytest.approx(0.4)
    assert state["height_weight"] == pytest.approx(0.4)
    assert state["up_weight"] == pytest.approx(0.05)


def test_reward_curriculum_anneals_back_to_canonical_weights():
    state = compute_reward_curriculum_state(
        {
            "enabled": True,
            "anneal_epochs": 20,
            "target_speed": 0.5,
            "low_speed_penalty_weight_init": 1.0,
            "forward_scale_init": 3.0,
            "heading_scale_init": 0.4,
            "height_scale_init": 0.4,
            "up_scale_init": 0.5,
        },
        epoch=20,
        base_forward_vel_weight=1.0,
        base_heading_weight=1.0,
        base_height_weight=1.0,
        base_up_weight=0.1,
    )
    assert state["progress"] == pytest.approx(1.0)
    assert state["low_speed_penalty_weight"] == pytest.approx(0.0)
    assert state["forward_vel_weight"] == pytest.approx(1.0)
    assert state["heading_weight"] == pytest.approx(1.0)
    assert state["height_weight"] == pytest.approx(1.0)
    assert state["up_weight"] == pytest.approx(0.1)


def test_reset_curriculum_disabled_stays_canonical():
    state = compute_reset_curriculum_state(
        {"enabled": False},
        epoch=0,
    )
    assert state["progress"] == pytest.approx(1.0)
    assert state["canonical_prob"] == pytest.approx(1.0)
    assert state["stride_left_prob"] == pytest.approx(0.0)
    assert state["stride_right_prob"] == pytest.approx(0.0)
    assert state["x_vel_range"] == pytest.approx((0.2, 0.5))
    assert state["y_vel_abs_max"] == pytest.approx(0.1)
    assert state["yaw_deg_abs_max"] == pytest.approx(12.0)
    assert state["yaw_rate_abs_max"] == pytest.approx(0.4)


def test_reset_curriculum_epoch_zero_uses_initial_stride_mix():
    state = compute_reset_curriculum_state(
        {
            "enabled": True,
            "anneal_epochs": 20,
            "canonical_prob_init": 0.4,
            "x_vel_range": [0.5, 0.2],
            "y_vel_abs_max": 0.2,
            "yaw_deg_abs_max": 9.0,
            "yaw_rate_abs_max": 0.6,
        },
        epoch=0,
    )
    assert state["progress"] == pytest.approx(0.0)
    assert state["canonical_prob"] == pytest.approx(0.4)
    assert state["stride_left_prob"] == pytest.approx(0.3)
    assert state["stride_right_prob"] == pytest.approx(0.3)
    assert state["x_vel_range"] == pytest.approx((0.2, 0.5))
    assert state["y_vel_abs_max"] == pytest.approx(0.2)
    assert state["yaw_deg_abs_max"] == pytest.approx(9.0)
    assert state["yaw_rate_abs_max"] == pytest.approx(0.6)


def test_reset_curriculum_anneals_back_to_canonical():
    state = compute_reset_curriculum_state(
        {
            "enabled": True,
            "anneal_epochs": 20,
            "canonical_prob_init": 0.4,
        },
        epoch=20,
    )
    assert state["progress"] == pytest.approx(1.0)
    assert state["canonical_prob"] == pytest.approx(1.0)
    assert state["stride_left_prob"] == pytest.approx(0.0)
    assert state["stride_right_prob"] == pytest.approx(0.0)


def test_reset_joint_bias_modes_are_mirrored():
    canonical = get_reset_joint_bias(RESET_MODE_CANONICAL)
    left = get_reset_joint_bias(RESET_MODE_STRIDE_LEFT)
    right = get_reset_joint_bias(RESET_MODE_STRIDE_RIGHT)

    assert canonical == pytest.approx((0.0,) * 8)
    assert right == pytest.approx(tuple(-value for value in left))
