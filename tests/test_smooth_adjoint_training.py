"""Integration test: SHAC training with smooth adjoint doesn't crash."""

import yaml
import warp as wp
wp.init()

from msk_warp.algorithms.shac import SHAC


def test_smooth_adjoint_short_training():
    """Run 5 epochs of SHAC with smooth adjoint to verify pipeline integrity."""
    cfg = {
        'params': {
            'general': {
                'seed': 42,
                'device': 'cuda:0',
                'logdir': 'logs/test_smooth_adjoint_ci',
            },
            'env': {
                'name': 'Ant',
                'model_path': 'assets/ant_substeps4.xml',
                'num_actors': 4,  # small for fast test
                'episode_length': 100,
                'stochastic_init': False,
                'substeps': 4,
                'action_strength': 1.0,
                'early_termination': False,
                'use_fd_jacobian': False,
                'tape_per_substep': True,
                'smooth_adjoint': True,
                'smooth_friction_viscosity': 10.0,
                'smooth_friction_scale': 0.01,
            },
            'network': {
                'actor': 'ActorStochasticMLP',
                'actor_mlp': {'units': [32, 32], 'activation': 'elu'},
                'critic': 'CriticMLP',
                'critic_mlp': {'units': [32, 32], 'activation': 'elu'},
            },
            'config': {
                'name': 'test_smooth_adjoint',
                'gamma': 0.99,
                'steps_num': 4,  # short horizon for test
                'max_epochs': 5,
                'actor_learning_rate': 2e-3,
                'critic_learning_rate': 2e-3,
                'lr_schedule': 'linear',
                'betas': [0.7, 0.95],
                'target_critic_alpha': 0.2,
                'obs_rms': True,
                'ret_rms': False,
                'critic_iterations': 2,
                'critic_method': 'td-lambda',
                'lambda': 0.95,
                'num_batch': 1,
                'num_actors': 4,
                'truncate_grads': True,
                'grad_norm': 1.0,
                'state_bptt': True,
                'state_grad_clip': 5.0,
                'obs_grad_clip': 0.0,
                'save_interval': 999,
            },
        }
    }

    shac = SHAC(cfg)
    shac.train()
    print(f"Training completed. best_policy_loss: {shac.best_policy_loss:.4f}")


if __name__ == '__main__':
    test_smooth_adjoint_short_training()
