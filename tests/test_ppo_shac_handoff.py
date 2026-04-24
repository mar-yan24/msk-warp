"""Test that PPO checkpoints are SHAC-compatible for warm-start transfer."""

from __future__ import annotations

import torch

from msk_warp.algorithms.shac import SHAC
from msk_warp.networks.actor import ActorStochasticMLP
from msk_warp.networks.critic import CriticMLP
from msk_warp.utils.running_mean_std import RunningMeanStd


def _make_actor(obs_dim=37, action_dim=8, device="cpu"):
    cfg = {
        'actor_mlp': {'units': [128, 64, 32], 'activation': 'elu'},
        'actor_logstd_init': -1.0,
    }
    return ActorStochasticMLP(obs_dim, action_dim, cfg, device=device)


def _make_critic(obs_dim=37, device="cpu"):
    cfg = {'critic_mlp': {'units': [64, 64], 'activation': 'elu'}}
    return CriticMLP(obs_dim, cfg, device=device)


def test_ppo_checkpoint_loads_into_shac(tmp_path):
    """A PPO-format checkpoint can be loaded by SHAC.load()."""
    checkpoint_path = tmp_path / "ppo_policy.pt"

    actor = _make_actor()
    critic = _make_critic()

    obs_rms = RunningMeanStd(shape=(37,), device="cpu")
    obs_rms.mean = torch.randn(37)
    obs_rms.var = torch.rand(37) + 0.1
    obs_rms.count = 5000.0

    ret_rms = RunningMeanStd(shape=(), device="cpu")
    ret_rms.mean = torch.tensor(1.5)
    ret_rms.var = torch.tensor(0.8)
    ret_rms.count = 5000.0

    # Save in SHAC-compatible format (PPO saves critic twice for target_critic slot)
    import copy
    torch.save(
        [actor, critic, copy.deepcopy(critic), obs_rms, ret_rms],
        checkpoint_path,
    )

    # Create a mock SHAC instance (bypass __init__)
    shac = object.__new__(SHAC)
    shac.device = "cpu"
    shac.actor_lr = 5e-4
    shac.critic_lr = 2e-3
    shac.betas = (0.7, 0.95)
    shac.actor = _make_actor()
    shac.critic = _make_critic()
    shac.target_critic = _make_critic()

    # Load PPO checkpoint
    shac.load(str(checkpoint_path), reset_optimizers=True)

    # Verify actor weights transferred
    for p_loaded, p_orig in zip(shac.actor.parameters(), actor.parameters()):
        assert torch.allclose(p_loaded, p_orig), "Actor weights mismatch"

    # Verify critic weights transferred
    for p_loaded, p_orig in zip(shac.critic.parameters(), critic.parameters()):
        assert torch.allclose(p_loaded, p_orig), "Critic weights mismatch"

    # Verify obs_rms transferred
    assert torch.allclose(shac.obs_rms.mean, obs_rms.mean)
    assert torch.allclose(shac.obs_rms.var, obs_rms.var)
    assert shac.obs_rms.count == obs_rms.count

    # Verify ret_rms transferred
    assert torch.allclose(shac.ret_rms.mean, ret_rms.mean)
    assert torch.allclose(shac.ret_rms.var, ret_rms.var)

    # Verify optimizers were rebuilt with correct parameters
    actor_params = list(shac.actor.parameters())
    assert shac.actor_optimizer.param_groups[0]["params"][0] is actor_params[0]


def test_evaluate_actions_matches_forward():
    """evaluate_actions log_prob is consistent with forward sampling."""
    actor = _make_actor()
    obs = torch.randn(4, 37)

    # Sample actions using forward
    pre_tanh = actor(obs, deterministic=False)

    # Evaluate log_prob of those actions
    log_prob, entropy = actor.evaluate_actions(obs, pre_tanh)

    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert not torch.isnan(log_prob).any()
    assert not torch.isnan(entropy).any()
    # Entropy should be positive for Normal distribution
    assert (entropy > 0).all()


def test_evaluate_actions_gradient_flows():
    """Gradients flow through evaluate_actions to actor parameters."""
    actor = _make_actor()
    obs = torch.randn(4, 37)
    actions = torch.randn(4, 8)

    log_prob, entropy = actor.evaluate_actions(obs, actions)
    loss = -log_prob.mean() - 0.01 * entropy.mean()
    loss.backward()

    # mu_net should have gradients
    for p in actor.mu_net.parameters():
        assert p.grad is not None, "mu_net param missing gradient"
    # logstd should have gradient
    assert actor.logstd.grad is not None, "logstd missing gradient"
