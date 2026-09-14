"""Unit tests for the periodic-gait metrics.

The period search is the instrument Phase 4's gates are written against, so it is tested on
signals whose period is known by construction rather than only on recorded rollouts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from msk_warp.utils.gait import (
    SCALE_FLOOR,
    component_scales,
    find_period,
    periodicity_residual,
    residual_curve,
    shape_vector,
)


def _sinusoid_trace(period, steps=400, dim=4, seed=0):
    """A trace that is exactly periodic with ``period``, with per-component phase and amplitude."""
    generator = torch.Generator().manual_seed(seed)
    phase = torch.rand(dim, generator=generator) * 2.0 * math.pi
    amplitude = 0.5 + torch.rand(dim, generator=generator)
    t = torch.arange(steps, dtype=torch.float32).unsqueeze(-1)
    return amplitude * torch.sin(2.0 * math.pi * t / period + phase)


# ------------------------------------------------------------------ shape vector


def test_shape_vector_drops_the_translating_coordinate():
    qpos = torch.tensor([[1.0, 2.0, 3.0]])
    qvel = torch.tensor([[4.0, 5.0, 6.0]])
    assert torch.equal(shape_vector(qpos, qvel), torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]))


def test_shape_vector_appends_activation_when_present():
    qpos, qvel = torch.zeros(1, 3), torch.zeros(1, 3)
    act = torch.tensor([[0.25, 0.75]])
    assert shape_vector(qpos, qvel, act).shape == (1, 7)
    assert torch.equal(shape_vector(qpos, qvel, act)[:, -2:], act)


def test_shape_vector_ignores_empty_activation():
    """A motor model carries a zero-width act array; it must not change the shape vector."""
    qpos, qvel = torch.zeros(1, 3), torch.zeros(1, 3)
    empty = torch.zeros(1, 0)
    assert shape_vector(qpos, qvel, empty).shape == shape_vector(qpos, qvel).shape


def test_shape_vector_broadcasts_over_batch_dimensions():
    qpos, qvel = torch.zeros(7, 5, 6), torch.zeros(7, 5, 6)
    assert shape_vector(qpos, qvel).shape == (7, 5, 11)


# ------------------------------------------------------------------ scales


def test_component_scales_floor_a_component_that_never_moves():
    trace = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([1.0, 10.0])])
    scales = component_scales(trace)
    assert scales[0] == pytest.approx(SCALE_FLOOR)
    assert scales[1] > 1.0


def test_component_scales_rejects_a_single_sample():
    with pytest.raises(ValueError, match="at least 2 samples"):
        component_scales(torch.zeros(1, 3))


# ------------------------------------------------------------------ residual


def test_residual_is_zero_for_identical_states():
    a = torch.randn(5, 4)
    assert torch.allclose(periodicity_residual(a, a, torch.ones(4)), torch.zeros(5), atol=1e-7)


def test_residual_is_invariant_to_rescaling_a_component():
    """Doubling a component and its scale must leave the residual unchanged."""
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    scales = torch.rand(4) + 0.5
    factor = torch.tensor([1.0, 2.0, 1.0, 5.0])
    base = periodicity_residual(a, b, scales)
    rescaled = periodicity_residual(a * factor, b * factor, scales * factor)
    assert torch.allclose(base, rescaled, atol=1e-6)


def test_residual_is_the_rms_of_scaled_differences():
    a = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    b = torch.zeros(1, 4)
    # one component off by 3, scale 1, over 4 components: sqrt(9/4) = 1.5
    assert periodicity_residual(a, b, torch.ones(4)).item() == pytest.approx(1.5)


def test_residual_is_differentiable():
    a = torch.randn(4, 3, requires_grad=True)
    periodicity_residual(a, torch.randn(4, 3), torch.ones(3)).sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


# ------------------------------------------------------------------ period search


def test_residual_curve_marks_periods_that_do_not_fit_as_nan():
    trace = torch.randn(10, 3)
    curve = residual_curve(trace, torch.ones(3), [2, 9, 10, 50])
    assert np.isfinite(curve[0]) and np.isfinite(curve[1])
    assert np.isnan(curve[2]) and np.isnan(curve[3])


@pytest.mark.parametrize("period", [8, 13, 21, 32])
def test_find_period_recovers_a_known_sinusoid(period):
    trace = _sinusoid_trace(period)
    found, residual, _, _ = find_period(trace, component_scales(trace), min_period=6, max_period=80)
    assert found == period
    assert residual < 1e-3


def test_find_period_returns_the_fundamental_not_a_multiple():
    """A signal with period 10 is equally periodic at 20 and 30; the smallest must win."""
    trace = _sinusoid_trace(10)
    scales = component_scales(trace)
    found, _, periods, curve = find_period(trace, scales, min_period=6, max_period=80)
    assert found == 10
    # the multiples really are near-minimal, which is what makes the tie-break necessary
    assert curve[list(periods).index(20)] < 1e-3


def test_find_period_reports_a_high_residual_for_an_aperiodic_trace():
    trace = torch.cumsum(torch.randn(400, 4, generator=torch.Generator().manual_seed(3)), dim=0)
    _, residual, _, _ = find_period(trace, component_scales(trace), min_period=6, max_period=80)
    assert residual > 0.05


def test_find_period_on_a_constant_trace_is_degenerate_by_design():
    """A standing policy has no period. The search returns min_period at ~0 residual; the caller
    is responsible for rejecting it on displacement, which is what the protocol's gates do."""
    trace = torch.ones(200, 4)
    found, residual, _, _ = find_period(trace, component_scales(trace), min_period=6, max_period=80)
    assert found == 6
    assert residual == pytest.approx(0.0, abs=1e-7)


def test_find_period_rejects_a_trace_shorter_than_every_candidate_period():
    with pytest.raises(ValueError, match="too short"):
        find_period(torch.randn(5, 3), torch.ones(3), min_period=6, max_period=80)
