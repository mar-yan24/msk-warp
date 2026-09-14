"""Official MyoLeg26 PPO optimizer/checkpoint plumbing, not a learning benchmark."""

import copy
import math
from pathlib import Path

import torch
import yaml

from msk_warp.algorithms.ppo import PPO


def test_myoleg26_ppo_updates_timeouts_and_checkpoint(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "msk_warp/configs/experiments/myoleg26_ppo.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["params"]["general"].update(seed=0, device="cuda:0", logdir=str(tmp_path))
    cfg["params"]["env"].update(
        num_actors=2, episode_length=2, stochastic_init=False, early_termination=False,
    )
    cfg["params"]["config"].update(
        steps_num=8, max_epochs=2, ppo_epochs=1, num_minibatches=2, save_interval=0,
    )
    cfg["params"]["network"]["actor_mlp"]["units"] = [16, 16]
    cfg["params"]["network"]["critic_mlp"]["units"] = [16, 16]

    algo = PPO(cfg)
    try:
        assert algo.env.model_contract == "official"
        assert algo.env.task_contract.version == "myoleg26-walk-v1"
        assert algo.env.no_grad
        assert algo.num_obs == 145 and algo.num_actions == 26
        assert algo.obs_rms is not None
        assert algo.ret_rms is None
        algo.env.begin_epoch(epoch=0, max_epochs=2)
        algo._current_obs = algo.env.reset()
        reset_obs = algo._current_obs.clone()
        assert torch.isfinite(reset_obs).all()
        assert (reset_obs[:, -26:] == -1).all()
        assert (reset_obs[:, -54:-28] == 0).all()

        for iteration in range(2):
            algo.env.begin_epoch(epoch=iteration, max_epochs=2)
            # collect_rollout freezes normalization for this entire rollout.
            rollout_rms = copy.deepcopy(algo.obs_rms)
            mean_parameters = [p.detach().clone() for p in algo.actor.mu_net.parameters()]
            algo.collect_rollout()

            for name in (
                "buf_obs", "buf_actions", "buf_log_probs", "buf_rewards",
                "buf_values", "buf_timeout_values", "buf_advantages", "buf_returns",
            ):
                assert torch.isfinite(getattr(algo, name)).all(), name
            expected_dones = torch.tensor(
                [0, 1] * 4, device=algo.device, dtype=algo.buf_dones.dtype,
            )[:, None].expand_as(algo.buf_dones)
            torch.testing.assert_close(algo.buf_dones, expected_dones, rtol=0, atol=0)
            # Every completed two-step episode returns the same deterministic
            # reset observation, including zero activation and passive commands.
            expected_resets = rollout_rms.normalize(reset_obs)[None].expand_as(algo.buf_obs[::2])
            torch.testing.assert_close(algo.buf_obs[::2], expected_resets, rtol=0, atol=0)
            torch.testing.assert_close(algo._current_obs, reset_obs, rtol=0, atol=0)
            qpos, qvel, act = algo.env.state_tensors()
            torch.testing.assert_close(algo._current_obs, algo.env.compute_obs(qpos, qvel, act))
            assert (algo.env.progress_buf == 0).all()
            assert algo.env.extras["truncated"].all()
            assert not algo.env.extras["terminated"].any()
            terminal_obs = algo.env.extras["obs_before_reset"]
            assert torch.isfinite(terminal_obs).all()
            assert (terminal_obs[:, -54:-28] > 0).any()
            assert not torch.equal(terminal_obs, algo._current_obs)
            # The timeout bootstrap must use the pre-reset state and the same
            # frozen normalizer as the collected values, before any PPO update.
            with torch.no_grad():
                expected_timeout_value = algo.critic(rollout_rms.normalize(terminal_obs)).squeeze(-1)
            torch.testing.assert_close(algo.buf_timeout_values[-1], expected_timeout_value)
            assert (algo.buf_timeout_values[::2] == 0).all()

            losses = algo.update()
            assert losses and all(math.isfinite(value) for value in losses.values())
            for network in (algo.actor, algo.critic):
                assert all(torch.isfinite(parameter).all() for parameter in network.parameters())
            # Check the policy mean network, so entropy-only changes to logstd
            # cannot satisfy the assertion that the actor was actually updated.
            assert any(
                not torch.equal(before, after)
                for before, after in zip(mean_parameters, algo.actor.mu_net.parameters())
            )

        assert algo.step_count == 32
        assert len(algo.episode_length_his) == 16
        assert all(length == 2 for length in algo.episode_length_his)
        assert algo.obs_rms.count > 32
        probe_obs = torch.cat((reset_obs, terminal_obs), dim=0).clone()
        with torch.no_grad():
            saved_output = algo.actor(algo.obs_rms.normalize(probe_obs), deterministic=True).clone()
        saved_rms = copy.deepcopy(algo.obs_rms)
        saved_actor = copy.deepcopy(algo.actor.state_dict())
        algo.save("plumbing_warmstart")
        checkpoint_path = tmp_path / "plumbing_warmstart.pt"
        assert checkpoint_path.is_file()

        with torch.no_grad():
            for parameter in algo.actor.parameters():
                parameter.add_(1.0)
            algo.obs_rms.mean.add_(2.0)
            algo.obs_rms.var.mul_(3.0)
            algo.obs_rms.count += 100
        # Existing load() restores policy/normalizers and rebinds fresh
        # optimizers. This tests a warm start, not full training-state resume.
        algo.load(str(checkpoint_path))
        for name, expected in saved_actor.items():
            torch.testing.assert_close(algo.actor.state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(algo.obs_rms.mean, saved_rms.mean, rtol=0, atol=0)
        torch.testing.assert_close(algo.obs_rms.var, saved_rms.var, rtol=0, atol=0)
        assert algo.obs_rms.count == saved_rms.count
        assert algo.ret_rms is None
        with torch.no_grad():
            restored_output = algo.actor(algo.obs_rms.normalize(probe_obs), deterministic=True)
        torch.testing.assert_close(restored_output, saved_output, rtol=0, atol=0)
        assert {id(p) for group in algo.actor_optimizer.param_groups for p in group["params"]} == {
            id(p) for p in algo.actor.parameters()
        }
    finally:
        algo.close()
