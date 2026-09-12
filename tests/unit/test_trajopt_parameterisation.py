"""The trajectory optimiser's initial-state parameterisation must stay inside the model.

Every restart begins from a free initial state. If that state can leave the joint ranges or the
floor, a restart can start on a constraint boundary, where G3.2 measured the derivative to be
undefined and the finite difference to swing by 20-100x. These bounds are what keeps the 512
restarts on the smooth interior, so they are pinned rather than trusted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

# Ranges as written in msk_warp/assets/hopper_muscle.xml and hopper_motor.xml.
THIGH_RANGE = (-2.61799, 0.0)
LEG_RANGE = (-2.61799, 0.0)
FOOT_RANGE = (-0.785398, 0.785398)


def _load_trajopt():
    path = Path(__file__).resolve().parents[2] / "scripts" / "trajopt_hopper.py"
    spec = importlib.util.spec_from_file_location("trajopt_hopper_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_parameters_give_a_crouched_rest_pose():
    trajopt = _load_trajopt()
    qpos, qvel, act = trajopt.initial_state(torch.zeros(1, 17), 6, "cpu")
    assert torch.allclose(qpos[0, :3], torch.zeros(3))          # x, height, pitch at the rest pose
    assert torch.allclose(qpos[0, 3:5], torch.full((2,), -0.6))  # thigh and leg mid-range
    assert qpos[0, 5].item() == 0.0                              # foot neutral
    assert torch.allclose(qvel, torch.zeros_like(qvel))
    assert torch.allclose(act, torch.full_like(act, 0.5))


def test_extreme_parameters_stay_strictly_inside_every_joint_range():
    trajopt = _load_trajopt()
    for sign in (-1.0, 1.0):
        qpos, qvel, act = trajopt.initial_state(sign * torch.full((4, 17), 100.0), 6, "cpu")
        assert abs(qpos[:, 1]).max() <= trajopt.HEIGHT_BOUND
        assert abs(qpos[:, 2]).max() <= trajopt.PITCH_BOUND
        assert (qpos[:, 3] > THIGH_RANGE[0]).all() and (qpos[:, 3] < THIGH_RANGE[1]).all()
        assert (qpos[:, 4] > LEG_RANGE[0]).all() and (qpos[:, 4] < LEG_RANGE[1]).all()
        assert (qpos[:, 5] > FOOT_RANGE[0]).all() and (qpos[:, 5] < FOOT_RANGE[1]).all()
        assert (act >= 0.0).all() and (act <= 1.0).all()
        bound = torch.tensor(trajopt.QVEL_BOUND)
        assert (qvel.abs() <= bound).all()


def test_the_track_coordinate_is_never_a_degree_of_freedom():
    """Displacement is measured from wherever the cycle starts, so x must always begin at 0."""
    trajopt = _load_trajopt()
    qpos, _, _ = trajopt.initial_state(torch.randn(32, 11), 6, "cpu")
    assert torch.equal(qpos[:, 0], torch.zeros(32))


def test_a_motor_model_gets_an_empty_activation_vector():
    trajopt = _load_trajopt()
    _, _, act = trajopt.initial_state(torch.randn(3, 11), 0, "cpu")
    assert act.shape == (3, 0)


def test_the_parameterisation_is_differentiable():
    """All eleven free parameters must carry signal -- and there are now eleven, not seventeen.

    The initial activation left the parameter vector because it is determined by the control rather
    than chosen (``docs/VALIDITY.md`` CL-02), so this used to pass a width of 17 and assert that all
    seventeen columns received gradient. Six of them no longer exist.
    """
    trajopt = _load_trajopt()
    z = torch.randn(5, 11, requires_grad=True)
    qpos, qvel, act = trajopt.initial_state(z, 6, "cpu")
    (qpos.sum() + qvel.sum() + act.sum()).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    # x is constant, so column 0 of qpos contributes nothing; every free parameter still gets signal
    assert (z.grad.abs() > 0).all()


def test_activation_is_constant_and_carries_no_gradient():
    """The other half of CL-02: activation must not depend on the parameter vector at all."""
    trajopt = _load_trajopt()
    z = torch.randn(4, 11, requires_grad=True)
    _, _, act = trajopt.initial_state(z, 6, "cpu")
    assert torch.allclose(act, torch.full_like(act, 0.5))
    # Stronger than "the gradient is zero": activation is not on the graph at all, so it has no
    # grad_fn and backward() through it raises. That is the property CL-02 asks for.
    assert act.grad_fn is None and not act.requires_grad
