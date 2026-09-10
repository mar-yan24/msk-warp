"""PPO rollout collection against a live environment.

Threading activation state through the bridge changed ``env.step`` from six return values to
seven, and PPO kept unpacking six. It raised ``ValueError`` on its first rollout and nothing
caught it, because the only PPO coverage in the suite loads a saved ant checkpoint and never runs
PPO training. This exercises the path that broke.
"""

import pytest
import torch

from msk_warp.algorithms.ppo import PPO


def _cfg(env_name, model_path, num_act, **env_extra):
    return {
        'params': {
            'general': {'seed': 0, 'device': 'cuda:0', 'logdir': None},
            'env': {
                'name': env_name,
                'model_path': model_path,
                'num_actors': 4,
                'episode_length': 32,
                'stochastic_init': True,
                'substeps': 2,
                **env_extra,
            },
            'network': {
                'actor': 'ActorStochasticMLP',
                'actor_mlp': {'units': [16, 16], 'activation': 'elu'},
                'critic': 'CriticMLP',
                'critic_mlp': {'units': [16, 16], 'activation': 'elu'},
            },
            'config': {
                'algorithm': 'ppo',
                'name': 'ppo_rollout_test',
                'steps_num': 8,
                'max_epochs': 1,
                'actor_learning_rate': 3e-4,
                'critic_learning_rate': 3e-4,
                'obs_rms': True,
                'save_interval': 0,
            },
        }
    }


@pytest.mark.parametrize("env_name,model_path,num_act", [
    ("HopperMotor", "assets/hopper_motor.xml", 3),
    ("HopperMuscle", "assets/hopper_muscle.xml", 6),
])
def test_ppo_collect_rollout_runs(tmp_path, env_name, model_path, num_act):
    """One rollout on each hopper variant, including the muscle model whose act is what broke it."""
    cfg = _cfg(env_name, model_path, num_act)
    cfg['params']['general']['logdir'] = str(tmp_path)
    algo = PPO(cfg)
    # train() seeds _current_obs before its first rollout; do the same minimal setup here.
    algo.env.begin_epoch(epoch=0, max_epochs=1)
    algo._current_obs = algo.env.reset()

    algo.collect_rollout()

    assert algo.buf_rewards.shape[0] == 8
    assert torch.isfinite(algo.buf_rewards).all()
    assert torch.isfinite(algo.buf_values).all()
    assert torch.isfinite(algo.buf_log_probs).all()
    assert algo.buf_actions.shape[-1] == num_act
