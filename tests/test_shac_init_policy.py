"""Unit tests for SHAC checkpoint warm-start support."""

from __future__ import annotations

import torch

from msk_warp.algorithms.shac import SHAC
from msk_warp.utils.running_mean_std import RunningMeanStd


def _linear(in_features, out_features, weight_value, bias_value):
    layer = torch.nn.Linear(in_features, out_features)
    with torch.no_grad():
        layer.weight.fill_(weight_value)
        layer.bias.fill_(bias_value)
    return layer


def test_shac_load_rebuilds_optimizers_and_restores_rms(tmp_path):
    checkpoint_path = tmp_path / "warm_start.pt"

    actor = _linear(3, 2, weight_value=0.25, bias_value=-0.5)
    critic = _linear(3, 1, weight_value=0.75, bias_value=0.1)
    target_critic = _linear(3, 1, weight_value=-0.5, bias_value=0.3)

    obs_rms = RunningMeanStd(shape=(3,), device="cpu")
    obs_rms.mean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    obs_rms.var = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    obs_rms.count = 123.0

    ret_rms = RunningMeanStd(shape=(), device="cpu")
    ret_rms.mean = torch.tensor(7.0, dtype=torch.float32)
    ret_rms.var = torch.tensor(8.0, dtype=torch.float32)
    ret_rms.count = 45.0

    torch.save([actor, critic, target_critic, obs_rms, ret_rms], checkpoint_path)

    shac = object.__new__(SHAC)
    shac.device = "cpu"
    shac.actor_lr = 1e-3
    shac.critic_lr = 2e-3
    shac.betas = (0.7, 0.95)
    shac.actor = _linear(3, 2, weight_value=9.0, bias_value=9.0)
    shac.critic = _linear(3, 1, weight_value=9.0, bias_value=9.0)
    shac.target_critic = _linear(3, 1, weight_value=9.0, bias_value=9.0)

    shac.load(str(checkpoint_path))

    assert torch.allclose(shac.actor.weight, actor.weight)
    assert torch.allclose(shac.actor.bias, actor.bias)
    assert torch.allclose(shac.critic.weight, critic.weight)
    assert torch.allclose(shac.critic.bias, critic.bias)
    assert torch.allclose(shac.target_critic.weight, target_critic.weight)
    assert torch.allclose(shac.target_critic.bias, target_critic.bias)
    assert torch.allclose(shac.obs_rms.mean, obs_rms.mean)
    assert torch.allclose(shac.obs_rms.var, obs_rms.var)
    assert shac.obs_rms.count == obs_rms.count
    assert torch.allclose(shac.ret_rms.mean, ret_rms.mean)
    assert torch.allclose(shac.ret_rms.var, ret_rms.var)
    assert shac.ret_rms.count == ret_rms.count

    actor_params = list(shac.actor.parameters())
    critic_params = list(shac.critic.parameters())
    assert shac.actor_optimizer.param_groups[0]["params"][0] is actor_params[0]
    assert shac.critic_optimizer.param_groups[0]["params"][0] is critic_params[0]


def test_shac_load_sets_bootstrap_reference_when_enabled(tmp_path):
    checkpoint_path = tmp_path / "warm_start.pt"

    actor = _linear(3, 2, weight_value=0.25, bias_value=-0.5)
    critic = _linear(3, 1, weight_value=0.75, bias_value=0.1)
    target_critic = _linear(3, 1, weight_value=-0.5, bias_value=0.3)

    torch.save([actor, critic, target_critic, None, None], checkpoint_path)

    shac = object.__new__(SHAC)
    shac.device = "cpu"
    shac.actor_lr = 1e-3
    shac.critic_lr = 2e-3
    shac.betas = (0.7, 0.95)
    shac.bootstrap_reg_enabled = True
    shac.actor = _linear(3, 2, weight_value=9.0, bias_value=9.0)
    shac.critic = _linear(3, 1, weight_value=9.0, bias_value=9.0)
    shac.target_critic = _linear(3, 1, weight_value=9.0, bias_value=9.0)

    shac.load(str(checkpoint_path))

    assert shac.bootstrap_ref_actor is not None
    for ref_param, actor_param in zip(shac.bootstrap_ref_actor.parameters(), shac.actor.parameters()):
        assert torch.allclose(ref_param, actor_param)
        assert ref_param.requires_grad is False
