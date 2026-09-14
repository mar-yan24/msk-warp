"""Periodic-gait metrics: shape vectors, component scales, periodicity residual, period search.

A gait is a periodic orbit of the state *modulo translation*: after one cycle the hopper is in the
same configuration, moving the same way, just further down the track. So every quantity here is
computed on a **shape vector** that drops ``qpos[0]``, the translating coordinate, and keeps
everything else including muscle activation.

The residual is scaled component-wise by the standard deviation of each component over a reference
trace. That matters because the raw components are in incomparable units -- metres, radians,
radians per second, dimensionless activation -- and any fixed weighting of them would be a hidden
tuning knob. Dividing by the gait's own variation removes the knob: a residual of 1.0 means the
cycle fails to close by about as much as the gait itself moves.

The same scales are reused across models so that a motor gait and a muscle gait are measured on one
ruler (``docs/research/phase4-capability/protocol.md``, stage 0).
"""

import numpy as np
import torch

#: ``qpos[0]`` is the translating coordinate and is never part of a shape vector.
TRANSLATING_INDEX = 0

#: Floor on a component scale, so a component that never moves cannot divide by zero.
SCALE_FLOOR = 1e-3


def shape_vector(qpos, qvel, act=None):
    """State with the translating coordinate removed: ``(qpos[1:], qvel, act)``.

    Accepts any leading batch shape. ``act`` is appended only when the model has activation
    state, which keeps the motor and muscle variants on the same code path.
    """
    parts = [qpos[..., TRANSLATING_INDEX + 1:], qvel]
    if act is not None and act.shape[-1] > 0:
        parts.append(act)
    return torch.cat(parts, dim=-1)


def component_scales(trace, floor=SCALE_FLOOR):
    """Per-component standard deviation over a ``(T, D)`` trace, floored at ``floor``."""
    if trace.shape[0] < 2:
        raise ValueError(f"need at least 2 samples to estimate scales, got {trace.shape[0]}")
    return trace.std(dim=0).clamp_min(floor)


def periodicity_residual(a, b, scales):
    """Scaled RMS difference between two shape vectors.

    ``|| (a - b) / scales ||_2 / sqrt(D)``, which is the root-mean-square of the scaled
    per-component differences. Broadcasts over any leading batch dimensions and stays
    differentiable, so the same function scores a recorded trace and drives the optimiser.
    """
    return (((a - b) / scales) ** 2).mean(dim=-1).sqrt()


def residual_curve(trace, scales, periods):
    """``R(T)`` averaged over every valid start step, for each ``T`` in ``periods``.

    Periods that do not fit in the trace come back as NaN rather than raising, so a caller can
    sweep a fixed range against a short trace.
    """
    out = np.empty(len(periods), dtype=np.float64)
    total = trace.shape[0]
    for i, period in enumerate(periods):
        period = int(period)
        if period <= 0 or period >= total:
            out[i] = np.nan
            continue
        out[i] = float(periodicity_residual(trace[period:], trace[:-period], scales).mean())
    return out


def find_period(trace, scales, min_period=6, max_period=80, tolerance=1.2):
    """Fundamental period of a trace, as the smallest near-minimal residual.

    Returns ``(period, residual, periods, curve)``.

    A trace periodic with period ``P`` is also near-periodic at ``2P``, ``3P`` and so on, so the
    global minimum of ``R(T)`` is not necessarily the fundamental. Taking the *smallest* ``T``
    whose residual is within ``tolerance`` of the minimum picks the fundamental instead.

    A trace that does not oscillate at all -- a policy that stands still -- has a near-zero
    residual at every period and will return ``min_period`` with a residual near zero. That is not
    a bug and not a gait; the caller is expected to check displacement separately.
    """
    periods = np.arange(int(min_period), int(max_period) + 1)
    curve = residual_curve(trace, scales, periods)
    if not np.isfinite(curve).any():
        raise ValueError(f"trace of length {trace.shape[0]} is too short for periods {min_period}..{max_period}")
    best = float(np.nanmin(curve))
    near_minimal = np.isfinite(curve) & (curve <= tolerance * best)
    index = int(np.argmax(near_minimal))
    return int(periods[index]), float(curve[index]), periods, curve
