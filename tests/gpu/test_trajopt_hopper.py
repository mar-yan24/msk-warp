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
    # 11 for both models: the initial activation left the parameterisation (VALIDITY CL-02).
    z = torch.randn(worlds, 11, device="cuda:0", generator=generator, requires_grad=True)
    return u, z


@pytest.mark.parametrize("kind,n_act_state", [("motor", 0), ("muscle", 6)])
def test_objective_gradient_reaches_controls_and_initial_state(kind, n_act_state):
    trajopt = _load_trajopt()
    worlds, cycle = 4, 3
    env = _env(kind, worlds)
    u, z = _params(env, worlds, cycle, n_act_state)
    scales = torch.ones(11, device="cuda:0")

    velocity, error, residual, violation = trajopt.rollout(
        env, u, z, scales, cycle, n_act_state, "cuda:0")
    (velocity - residual - violation).sum().backward()

    for name, grad in (("controls", u.grad), ("initial state", z.grad)):
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().max() > 0, f"{name} gradient is identically zero"


def test_initial_activation_is_not_a_decision_variable():
    """The replacement for a test that asserted the opposite, and was asserting the wrong thing.

    It used to read: "``ctrl`` reaches muscle force only through ``act``, so the initial-activation
    block of the parameter vector must carry signal. A zero there is the silent failure v2 was built
    to catch." The premise is true and the conclusion does not follow. Under a ``T_c``-periodic
    control the activation subsystem is autonomous -- ``dyntype=muscle`` computes ``act_dot`` from
    ``(ctrl, act)`` alone -- so it has a unique globally attracting periodic solution ``act*`` fixed
    by the control. Those six parameters were redundant *and* inconsistent with periodicity, and the
    optimiser spent them: the ``T_c`` 16 candidate's per-muscle "activation closing errors"
    ``[0.166, 0.427, 0.858, 0.886, 0.369, 0.212]`` are exactly ``|act_0 - act*|``
    (``docs/VALIDITY.md`` CL-02).

    So the parameter vector is 11 wide for both models, activation starts neutral, and the warm-up
    cycle carries it onto ``act*``. ``ctrl`` still reaches force through ``act`` -- that is checked
    where it belongs, on the control gradient.
    """
    trajopt = _load_trajopt()
    worlds, cycle, n_act_state = 4, 3, 6
    env = _env("muscle", worlds)
    u, z = _params(env, worlds, cycle, n_act_state)
    assert z.shape[1] == 11, "activation must not be back in the parameter vector"

    _, _, act = trajopt.initial_state(z, n_act_state, "cuda:0")
    assert act.shape == (worlds, n_act_state)
    assert torch.allclose(act, torch.full_like(act, 0.5)), "activation starts neutral"

    velocity, _, residual, _ = trajopt.rollout(env, u, z, torch.ones(11, device="cuda:0"),
                                               cycle, n_act_state, "cuda:0")
    (velocity - residual).sum().backward()
    assert torch.isfinite(u.grad).all() and u.grad.abs().max() > 0, (
        "the control must still carry the whole actuation gradient")


def test_warmup_puts_activation_on_its_limit_cycle():
    """After the warm-up the scored cycle starts *and ends* on ``act*``, so closure is structural.

    Checked against the independent float64 CPU implementation in
    :func:`msk_warp.analysis.orbit.activation_limit_cycle`, which integrates the activation ODE
    alone. Agreement means the warm-up is doing what the residual now assumes.
    """
    import numpy as np
    from msk_warp.analysis.orbit import activation_limit_cycle

    trajopt = _load_trajopt()
    worlds, cycle, n_act_state = 2, 8, 6
    env = _env("muscle", worlds)
    u, z = _params(env, worlds, cycle, n_act_state)

    with torch.no_grad():
        qpos, qvel, act = trajopt.initial_state(z, n_act_state, "cuda:0")
        violation = torch.zeros(worlds, device="cuda:0")
        for _ in range(trajopt.WARMUP_CYCLES):
            qpos, qvel, act, violation = trajopt._cycle(env, u, qpos, qvel, act, cycle, violation)
        ctrl = torch.stack([env._to_ctrl(torch.tanh(u[:, t])) for t in range(cycle)], dim=1)

    for w in range(worlds):
        want, _, _ = activation_limit_cycle(
            env.mjm, ctrl[w].double().cpu().numpy(), substeps=env.substeps)
        got = act[w].double().cpu().numpy()
        assert np.abs(got - want).max() < 5e-3, (w, got, want)


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
        velocity, _, _, _ = trajopt.rollout(env, u, z, scales, cycle, n_act_state, "cuda:0")
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


def test_rollout_returns_the_error_vector_the_constraint_needs():
    """The augmented Lagrangian constrains the 11-component error, not its norm.

    ``R = ||e||`` has an infinite-derivative kink at ``e = 0`` -- exactly where convergence has to
    happen -- so constraining the scalar would place a non-differentiable point at the solution.
    ``||e||^2`` is smooth there. This pins both that the vector is returned and that its RMS still
    reproduces the scalar every Phase 4 number was measured with.
    """
    trajopt = _load_trajopt()
    worlds, cycle, n_act_state = 4, 3, 6
    env = _env("muscle", worlds)
    u, z = _params(env, worlds, cycle, n_act_state)
    scales = torch.rand(11, device="cuda:0") + 0.5

    velocity, error, residual, violation = trajopt.rollout(
        env, u, z, scales, cycle, n_act_state, "cuda:0")
    assert error.shape == (worlds, 11)
    assert torch.allclose(error.pow(2).mean(dim=-1).sqrt(), residual, atol=1e-6)

    from msk_warp.utils.gait import periodicity_residual
    # the scalar must equal what the historical ruler would have produced
    assert torch.allclose(
        periodicity_residual(error, torch.zeros_like(error), torch.ones(11, device="cuda:0")),
        residual, atol=1e-6)
    error.pow(2).sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all() and u.grad.abs().max() > 0
