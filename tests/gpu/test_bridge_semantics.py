"""Bridge/environment semantics on the current backend: no tape warnings, graph-cut equivalence, determinism."""

from __future__ import annotations

import torch

from msk_warp.envs.cartpole_swing_up import CartPoleSwingUpEnv


def _cartpole_env(**kw):
    env = CartPoleSwingUpEnv(num_envs=8, device="cuda:0", substeps=4, action_strength=100.0, stochastic_init=False, **kw)
    env.reset()
    return env


def _actor_grad(env, seed=0):
    """Gradient of a fixed quadratic loss over a 3-step rollout w.r.t. a constant action."""
    torch.manual_seed(seed)
    actions = torch.full((env.num_envs, env.num_actions), 0.3, device="cuda:0", requires_grad=True)
    qpos, qvel, act = env.state_tensors()
    total = 0.0
    for _ in range(3):
        obs, rew, done, extras, qpos, qvel, act = env.step(torch.tanh(actions), qpos, qvel, act)
        total = total + rew.sum()
    total.backward()
    return actions.grad.clone(), qpos.detach().clone()


def test_no_tape_warning_is_emitted(capfd):
    env = _cartpole_env()
    _actor_grad(env)
    out, err = capfd.readouterr()
    assert "may produce incorrect gradients" not in out + err


def test_clear_grad_without_rebuild_matches_rebuild():
    env = _cartpole_env()
    g1, _ = _actor_grad(env)
    env.clear_grad(rebuild=True)
    env.reset()
    g2, _ = _actor_grad(env)
    env.clear_grad(rebuild=False)
    env.reset()
    g3, _ = _actor_grad(env)
    assert torch.allclose(g1, g2, rtol=1e-5, atol=1e-7)
    assert torch.allclose(g1, g3, rtol=1e-5, atol=1e-7)


def test_backward_is_deterministic():
    env = _cartpole_env()
    g1, q1 = _actor_grad(env)
    env.reset()
    g2, q2 = _actor_grad(env)
    assert torch.equal(q1, q2)
    assert torch.allclose(g1, g2, rtol=1e-6, atol=0.0)


def test_rerun_after_backward_leaves_post_step_state():
    """After one step's backward, Data holds that step's post-step state.

    (Over a multi-step rollout the backwards run in reverse, so Data ends at the post-state of the
    first step; SHAC saves and restores the rollout state around ``actor_loss.backward()``.)
    """
    env = _cartpole_env()
    actions = torch.full((env.num_envs, env.num_actions), 0.3, device="cuda:0", requires_grad=True)
    qpos, qvel, act = env.state_tensors()
    obs, rew, done, extras, qpos_out, qvel_out, act_out = env.step(torch.tanh(actions), qpos, qvel, act)
    q_end = qpos_out.detach().clone()
    rew.sum().backward()
    q_now, _, _ = env.state_tensors()
    assert torch.allclose(q_now, q_end, atol=1e-6)


def test_legacy_flags_map_to_backward_modes():
    assert CartPoleSwingUpEnv(num_envs=2, use_fd_jacobian=True).backward_mode == "fd"
    assert CartPoleSwingUpEnv(num_envs=2, tape_per_substep=True).backward_mode == "tape_per_substep"
    assert CartPoleSwingUpEnv(num_envs=2).backward_mode == "tape"
