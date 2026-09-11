"""Orbit analysis: Poincare return maps, shooting and Floquet spectra, in float64 CPU MuJoCo.

Deliberately imports **numpy and mujoco only** -- no torch, no warp. Two reasons:

1. It runs in ``tests/unit`` with no GPU and no Warp kernel compilation.
2. It is independent of the Warp adjoint, and therefore of backend defects BE-01, BE-02 and BE-06
   (``docs/VALIDITY.md``). That independence is the whole point: BE-02 corrupts ``dL/dqpos`` on any
   model where two tendons share a kinematic chain, which is every muscle hopper. Nothing in this
   package touches a gradient the backend computed.

The licence for using CPU MuJoCo to describe a Warp candidate is BE-08: replaying the exported
reference's own controls in float64 CPU reproduces the Warp-recorded closing state to
``|dqpos| = 8.4e-07``, ``|dqvel| = 3.8e-06``, ``|dact| = 4.4e-08`` over a full 16-step cycle.
That parity is re-asserted by ``tests/unit/test_orbit_parity.py`` on every commit, because the
package is worthless if it drifts.
"""

from msk_warp.analysis.orbit import (
    CONTROL_SUBSTEPS,
    TERMINATION_HEIGHT,
    CycleResult,
    OrbitTerminated,
    ReturnMap,
    activation_limit_cycle,
    assert_state_layout,
    shape_state,
    state_dim,
)
from msk_warp.analysis.stability import (
    Bounds,
    EpsSweep,
    Jacobian,
    LinearModelCheck,
    Outcome,
    ShootResult,
    Spectrum,
    StructuralChecks,
    cycles_to_amplitude,
    eps_sweep,
    jacobian_composed,
    jacobian_fd,
    linear_model_check,
    multistart,
    shoot,
    spectrum,
    structural_checks,
)

__all__ = [
    "Bounds", "CONTROL_SUBSTEPS", "CycleResult", "EpsSweep", "Jacobian", "LinearModelCheck",
    "OrbitTerminated", "Outcome", "ReturnMap", "ShootResult", "Spectrum", "StructuralChecks",
    "TERMINATION_HEIGHT", "activation_limit_cycle", "assert_state_layout",
    "cycles_to_amplitude", "eps_sweep", "jacobian_composed", "jacobian_fd",
    "linear_model_check", "multistart", "shape_state", "shoot", "spectrum",
    "state_dim", "structural_checks",
]
