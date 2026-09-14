"""Analytical reward/effort examples for the declared walking objective."""

import math

import pytest
import torch

from msk_warp.envs.myoleg26_task import MyoLegTaskContract, excitation_from_action


def _obs(vx=1.0, vy=0.0, upright=1.0, heading=1.0):
    obs = torch.zeros((1, 145), dtype=torch.float64)
    obs[:, 0] = 1.0
    obs[:, 5:7] = torch.tensor([vx, vy], dtype=torch.float64)
    obs[:, -28:-26] = torch.tensor([upright, heading], dtype=torch.float64)
    return obs


def test_command_effort_orders_passive_neutral_and_full_excitation():
    task = MyoLegTaskContract()
    actions = torch.tensor([[-1.0] * 26, [0.0] * 26, [1.0] * 26], dtype=torch.float64)
    excitation = excitation_from_action(actions)
    terms = task.reward_terms(_obs().expand(3, -1), excitation, 0.008)
    torch.testing.assert_close(terms['excitation_effort'], torch.tensor([0., .25, 1.], dtype=torch.float64))
    expected = torch.tensor([1., .9975, .99], dtype=torch.float64) * .008
    torch.testing.assert_close(task.reward(_obs().expand(3, -1), excitation, .008), expected)
    assert excitation_from_action(torch.tensor([-10., 10.])).tolist() == [0., 1.]


def test_tracking_orders_walking_above_standing_and_lateral_motion():
    task = MyoLegTaskContract()
    passive = torch.zeros((1, 26), dtype=torch.float64)
    assert task.reward(_obs(), passive, 1).item() == pytest.approx(1)
    assert task.reward(_obs(vx=0), passive, 1).item() == pytest.approx(math.exp(-4))
    assert task.reward(_obs(vy=1), passive, 1).item() == pytest.approx(math.exp(-4))
    assert task.reward(_obs(heading=-1), passive, 1).item() == 0
    assert task.reward(_obs(upright=-1), passive, 1).item() == 0


def test_reward_scales_with_elapsed_time_and_not_actuator_count():
    task = MyoLegTaskContract()
    excitation = torch.full((1, 26), .4, dtype=torch.float64)
    short = task.reward(_obs(), excitation, .004)
    torch.testing.assert_close(2 * short, task.reward(_obs(), excitation, .008))
    # Equal mean squared commands represent the same proxy effort across sizes.
    obs2 = torch.cat([_obs()[:, :-26], torch.zeros((1, 52), dtype=torch.float64)], dim=-1)
    torch.testing.assert_close(short, task.reward(obs2, excitation.repeat(1, 2), .004))


def test_effort_derivative_uses_excitation_not_signed_action():
    task = MyoLegTaskContract()
    actions = torch.zeros((1, 26), dtype=torch.float64, requires_grad=True)
    reward = task.reward(_obs(), excitation_from_action(actions), .008)
    gradient = torch.autograd.grad(reward.sum(), actions)[0]
    torch.testing.assert_close(gradient, torch.full_like(actions, -.008 * .01 * .5 / 26))
    assert torch.autograd.gradcheck(lambda a: task.reward(_obs(), excitation_from_action(a), .008), (actions,))


def test_failure_flags_distinguish_posture_and_nonfinite():
    task = MyoLegTaskContract()
    obs = _obs().repeat(4, 1)
    obs[1, 0] = .54
    obs[2, -28] = .49
    obs[3, 40] = float('nan')
    flags = task.failures(obs, 26)
    assert flags['low_pelvis'].tolist() == [False, True, False, False]
    assert flags['low_upright'].tolist() == [False, False, True, False]
    assert flags['nonfinite'].tolist() == [False, False, False, True]


@pytest.mark.parametrize('kwargs', [{'velocity_variance': 0}, {'effort_weight': -1}, {'target_speed': float('nan')}])
def test_invalid_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MyoLegTaskContract(**kwargs)
