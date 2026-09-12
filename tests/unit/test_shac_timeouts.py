"""SHAC final-state bootstrap semantics, without a simulator or GPU."""

import torch

from msk_warp.algorithms.shac import SHAC
from msk_warp.utils.running_mean_std import RunningMeanStd


class _Critic:
    def __call__(self, obs):
        return obs[:, :1]


def _shac(lengths):
    shac = object.__new__(SHAC)
    shac.episode_length = torch.tensor(lengths)
    shac.max_episode_length = 10
    shac.target_critic = _Critic()
    return shac


def test_explicit_timeouts_override_lengths_and_termination_wins_coincidence():
    shac = _shac([1, 10, 10, 10])
    next_values = torch.tensor([100., 200., 300., 400.], requires_grad=True)
    final_obs = torch.tensor([[2.], [4.], [6.], [8.]], requires_grad=True)
    values = shac._bootstrap_reset_values(next_values, torch.tensor([True, True, True, False]), {
        'terminated': torch.tensor([False, True, True, False]),
        'truncated': torch.tensor([True, False, True, False]),
        'obs_before_reset': final_obs,
    }, None)
    torch.testing.assert_close(values, torch.tensor([2., 0., 0., 400.]))
    values.sum().backward()
    # A timeout keeps its final-state gradient; reset values and true terminals do not.
    torch.testing.assert_close(next_values.grad, torch.tensor([0., 0., 0., 1.]))
    torch.testing.assert_close(final_obs.grad, torch.tensor([[1.], [0.], [0.], [0.]]))


def test_legacy_environments_keep_episode_length_inference():
    shac = _shac([9, 10, 11, 10])
    values = shac._bootstrap_reset_values(torch.tensor([100., 200., 300., 400.]),
        torch.tensor([True, True, True, False]),
        {'obs_before_reset': torch.tensor([[2.], [4.], [6.], [8.]])}, None)
    torch.testing.assert_close(values, torch.tensor([0., 4., 6., 400.]))


def test_invalid_timeout_final_states_keep_zero_bootstrap():
    shac = _shac([10] * 4)
    values = shac._bootstrap_reset_values(torch.ones(4), torch.ones(4, dtype=torch.bool), {
        'terminated': torch.zeros(4, dtype=torch.bool),
        'truncated': torch.ones(4, dtype=torch.bool),
        'obs_before_reset': torch.tensor([[float('nan')], [float('inf')], [1e7], [3.]]),
    }, None)
    torch.testing.assert_close(values, torch.tensor([0., 0., 0., 3.]))


def test_timeout_uses_provided_frozen_rms_for_final_state():
    shac = _shac([10])
    rms = RunningMeanStd(shape=(1,), device='cpu')
    rms.mean = torch.tensor([10.])
    rms.var = torch.tensor([4.])
    obs = torch.tensor([[18.]])
    values = shac._bootstrap_reset_values(torch.tensor([-100.]), torch.tensor([True]), {
        'terminated': torch.tensor([False]),
        'truncated': torch.tensor([True]),
        'obs_before_reset': obs,
    }, rms)
    torch.testing.assert_close(values, rms.normalize(obs).squeeze(-1))
