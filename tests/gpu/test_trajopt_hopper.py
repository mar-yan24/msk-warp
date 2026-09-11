"""Gradient and consistency gates for the periodic-orbit trajectory optimiser.

The Phase 4 capability claim rests on two things being true of this driver, and neither is obvious
from reading it: the adjoint must reach *both* the control sequence and the free initial state, and
the no-gradient verification rollout must reproduce the differentiable one. If either fails, a
"no muscle gait exists" result would be a statement about the driver.

For the muscle model the control path runs only through ``act_dot -> act``, so a driver that lost
the activation gradient would silently optimise nothing. That is the gradient contract v2 exists to
enforce, checked here at the level the optimiser actually uses.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from msk_warp.envs.hopper import HopperMotorEnv, HopperMuscleEnv


def _load_trajopt():
    path = Path(__file__).resolve().parents[2] / "scripts" / "trajopt_hopper.py"
    spec = importlib.util.spec_from_file_location("trajopt_hopper_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _env(kind, worlds=4):
    cls = HopperMotorEnv if kind == "motor" else HopperMuscleEnv
    return cls(
        num_envs=worlds, device="cuda:0", no_grad=False, episode_length=1000,
        stochastic_init=False, early_termination=False, substeps=2, njmax=128,
        backward_mode="tape_per_substep",
    )


def _params(env, worlds, cycle, n_act_state, seed=0):
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    u = torch.randn(worlds, cycle, env.num_actions, device="cuda:0", generator=generator,
                    requires_grad=True)
    z = torch.randn(worlds, 11 + n_act_state, device="cuda:0", generator=generator,
                    requires_grad=True)
    return u, z


@pytest.mark.parametrize("kind,n_act_state", [("motor", 0), ("muscle", 6)])
def test_objective_gradient_reaches_controls_and_initial_state(kind, n_act_state):
    trajopt = _load_trajopt()
    worlds, cycle = 4, 3
    env = _env(kind, worlds)
    u, z = _params(env, worlds, cycle, n_act_state)
    scales = torch.ones(11, device="cuda:0")

    velocity, residual, violation = trajopt.rollout(env, u, z, scales, cycle, n_act_state, "cuda:0")
    (velocity - residual - violation).sum().backward()

    for name, grad in (("controls", u.grad), ("initial state", z.grad)):
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().max() > 0, f"{name} gradient is identically zero"


def test_muscle_activation_parameters_receive_gradient():
    """``ctrl`` reaches muscle force only through ``act``, so the initial-activation block of the
    parameter vector must carry signal. A zero there is the silent failure v2 was built to catch."""
    trajopt = _load_trajopt()
    worlds, cycle, n_act_state = 4, 3, 6
    env = _env("muscle", worlds)
    u, z = _params(env, worlds, cycle, n_act_state)

    velocity, residual, _ = trajopt.rollout(env, u, z, torch.ones(11, device="cuda:0"),
                                            cycle, n_act_state, "cuda:0")
    (velocity - residual).sum().backward()

    activation_block = z.grad[:, 11:11 + n_act_state]
    assert torch.isfinite(activation_block).all()
    assert activation_block.abs().max() > 0, "initial activation received no gradient"


@pytest.mark.parametrize("kind,n_act_state", [("motor", 0), ("muscle", 6)])
def test_verification_reproduces_the_differentiable_rollout_over_one_cycle(kind, n_act_state):
    """One verification cycle is the same trajectory the objective scored, so the velocities must
    agree. If they diverge, the two paths differ and the gate is measuring something else."""
    trajopt = _load_trajopt()
    worlds, cycle = 4, 4
    env = _env(kind, worlds)
    u, z = _params(env, worlds, cycle, n_act_state, seed=1)
    scales = torch.ones(11, device="cuda:0")

    with torch.no_grad():
        velocity, _, _ = trajopt.rollout(env, u, z, scales, cycle, n_act_state, "cuda:0")
    _, verified, _ = trajopt.verify(env, u, z, scales, cycle, n_act_state, "cuda:0", cycles=1)

    assert torch.allclose(velocity, verified, atol=1e-4), f"{velocity} vs {verified}"


def test_verification_marks_a_collapsed_world_as_not_surviving():
    """A restart driven to zero activation collapses; it must not be reported as alive."""
    trajopt = _load_trajopt()
    worlds, cycle, n_act_state = 2, 8, 6
    env = _env("muscle", worlds)
    # Large negative logits: zero activation, which G3.1 measured falls 100% of the time.
    u = torch.full((worlds, cycle, env.num_actions), -8.0, device="cuda:0")
    z = torch.zeros(worlds, 11 + n_act_state, device="cuda:0")
    z[:, 11:] = -8.0

    alive, _, alive_fraction = trajopt.verify(
        env, u, z, torch.ones(11, device="cuda:0"), cycle, n_act_state, "cuda:0", cycles=20,
    )
    assert not alive.any(), "a passive muscle hopper must fall within 160 control steps"
    assert (alive_fraction < 1.0).all()
