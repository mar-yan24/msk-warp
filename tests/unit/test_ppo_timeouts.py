"""CPU checks for final-state bootstrapping at PPO time limits."""

from __future__ import annotations

import copy

import pytest
import torch

from msk_warp.algorithms.ppo import PPO
from msk_warp.utils.average_meter import AverageMeter
from msk_warp.utils.running_mean_std import RunningMeanStd


class _Actor:
    def forward_with_dist(self, obs):
        zeros = torch.zeros((obs.shape[0], 1))
        return zeros, zeros, torch.ones_like(zeros)


class _Critic:
    def __init__(self):
        self.inputs = []

    def __call__(self, obs):
        self.inputs.append(obs.clone())
        return obs[:, :1]


class _Env:
    def __init__(self, transitions):
        self.transitions = iter(transitions)

    def step(self, action):
        return next(self.transitions)


def _ppo(initial_obs, transitions, obs_rms=None):
    """Build the rollout-only algorithm without a simulator or log writer."""
    ppo = object.__new__(PPO)
    ppo.device = 'cpu'
    ppo.steps_num = len(transitions)
    ppo.num_envs, ppo.num_obs = initial_obs.shape
    ppo.gamma = 0.9
    ppo.gae_lambda = 0.5
    ppo.obs_rms = obs_rms
    ppo.actor = _Actor()
    ppo.critic = _Critic()
    ppo.env = _Env(transitions)
    ppo._current_obs = initial_obs
    ppo.step_count = 0
    shape = (ppo.steps_num, ppo.num_envs)
    ppo.buf_obs = torch.zeros((*shape, ppo.num_obs))
    ppo.buf_actions = torch.zeros((*shape, 1))
    for name in ('log_probs', 'rewards', 'dones', 'values', 'timeout_values',
                 'advantages', 'returns'):
        setattr(ppo, 'buf_' + name, torch.zeros(shape))
    ppo.episode_loss = torch.zeros(ppo.num_envs)
    ppo.episode_length = torch.zeros(ppo.num_envs, dtype=torch.long)
    ppo.episode_loss_his = []
    ppo.episode_length_his = []
    ppo.episode_loss_meter = AverageMeter(1, 100)
    ppo.episode_length_meter = AverageMeter(1, 100)
    return ppo


def test_mixed_timeouts_terminals_and_coincidence_cut_gae_keep_raw_rewards():
    """Only a pure timeout bootstraps; the following reset episode is excluded."""
    reset_obs = torch.tensor([[100.], [200.], [300.]])
    last_obs = reset_obs + 1
    rewards = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    transitions = [
        (reset_obs, rewards[0], torch.ones(3, dtype=torch.bool), {
            'terminated': torch.tensor([False, True, True]),
            'truncated': torch.tensor([True, False, True]),
            'obs_before_reset': torch.tensor([[10.], [20.], [30.]]),
        }),
        (last_obs, rewards[1], torch.zeros(3, dtype=torch.bool), {
            'terminated': torch.zeros(3, dtype=torch.bool),
            'truncated': torch.zeros(3, dtype=torch.bool),
        }),
    ]
    ppo = _ppo(torch.tensor([[1.], [2.], [3.]]), transitions)
    # Stale values from an earlier rollout must not survive non-timeout steps.
    ppo.buf_timeout_values.fill_(999.)
    ppo.collect_rollout()

    torch.testing.assert_close(ppo.buf_returns, torch.tensor([
        [10., 2., 3.], [94.9, 185.9, 276.9],
    ]))
    torch.testing.assert_close(ppo.buf_timeout_values, torch.tensor([
        [10., 0., 0.], [0., 0., 0.],
    ]))
    torch.testing.assert_close(ppo.buf_rewards, rewards)
    assert ppo.episode_loss_his == [-1., -2., -3.]
    assert ppo.episode_length_his == [1, 1, 1]
    torch.testing.assert_close(ppo.episode_loss, -rewards[1])
    torch.testing.assert_close(ppo._current_obs, last_obs)


@pytest.mark.parametrize('extras', [{}, {'obs_before_reset': torch.tensor([[50.]])}])
def test_legacy_done_without_termination_metadata_keeps_zero_bootstrap(extras):
    ppo = _ppo(torch.tensor([[2.]]), [
        (torch.tensor([[100.]]), torch.tensor([3.]), torch.tensor([True]), extras),
    ])
    ppo.collect_rollout()
    torch.testing.assert_close(ppo.buf_returns, torch.tensor([[3.]]))
    torch.testing.assert_close(ppo.buf_advantages, torch.tensor([[1.]]))


def test_timeout_at_rollout_end_uses_final_observation_and_frozen_rms():
    rms = RunningMeanStd(shape=(1,), device='cpu')
    rms.mean = torch.tensor([10.])
    rms.var = torch.tensor([4.])
    rms.count = 1.
    frozen = copy.deepcopy(rms)
    initial = torch.tensor([[100.]])
    final = torch.tensor([[18.]])
    reset = torch.tensor([[-100.]])
    ppo = _ppo(initial, [
        (reset, torch.tensor([2.]), torch.tensor([True]), {
            'terminated': torch.tensor([False]),
            'truncated': torch.tensor([True]),
            'obs_before_reset': final,
        }),
    ], rms)
    ppo.collect_rollout()

    final_value = frozen.normalize(final).squeeze(-1)
    torch.testing.assert_close(ppo.buf_returns[0], 2. + 0.9 * final_value)
    torch.testing.assert_close(ppo.buf_obs[0], frozen.normalize(initial))
    torch.testing.assert_close(ppo.critic.inputs[1], frozen.normalize(final))
    assert not torch.allclose(rms.mean, frozen.mean)
    torch.testing.assert_close(ppo.buf_rewards, torch.tensor([[2.]]))
    assert ppo.episode_loss_his == [-2.]


def test_nonterminal_gae_continues_within_episode():
    ppo = _ppo(torch.tensor([[1.]]), [
        (torch.tensor([[2.]]), torch.tensor([3.]), torch.tensor([False]), {}),
        (torch.tensor([[4.]]), torch.tensor([5.]), torch.tensor([False]), {}),
    ])
    ppo.collect_rollout()
    # Last delta = 5 + .9*4 - 2 = 6.6; first = 3 + .9*2 - 1 = 3.8.
    torch.testing.assert_close(ppo.buf_advantages, torch.tensor([[6.77], [6.6]]))
    torch.testing.assert_close(ppo.buf_returns, torch.tensor([[7.77], [8.6]]))
