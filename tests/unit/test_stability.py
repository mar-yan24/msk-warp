"""Unit tests for :mod:`msk_warp.analysis.stability`.

Split deliberately into two halves.

**Analytic fixtures** validate the linear algebra -- Jacobian recovery, eigenvalues, the shooting
solver, the discontinuity slope test -- against answers that can be written down in closed form, on
a map with no MuJoCo in it at all. If these fail, nothing measured on the hopper means anything.

**Hopper fixtures** pin the values measured on 2026-09-11, so a regression surfaces as a test
failure rather than as a surprising number in a results table. Recorded in ``docs/VALIDITY.md``.
"""

import types

import numpy as np
import mujoco
import pytest

from msk_warp import resolve_model_path
from msk_warp.analysis import orbit as orb
from msk_warp.analysis import stability as st

REFERENCE = "assets/references/hopper_muscle_T16_gait.npz"

# Measured 2026-09-11 at the T_c 16 candidate with activation pinned to act*.
CANDIDATE_RHO = 2.16734
CANDIDATE_RESIDUAL = 0.40997
LM_FLOOR = 0.076091


# --------------------------------------------------------------------------- analytic fixtures

class LinearMap:
    """``P(x) = A x + b``, duck-typed as a :class:`~msk_warp.analysis.orbit.ReturnMap`.

    Enough surface for :func:`~msk_warp.analysis.stability.jacobian_fd`,
    :func:`~msk_warp.analysis.stability.eps_sweep` and
    :func:`~msk_warp.analysis.stability.shoot`, and no more. ``kink`` adds a jump discontinuity at
    ``x[0] == 0`` so the slope test can be validated on a map whose non-smoothness is exact.
    """

    def __init__(self, A, b=None, kink=0.0):
        self.A = np.asarray(A, dtype=np.float64)
        self.b = np.zeros(self.A.shape[0]) if b is None else np.asarray(b, dtype=np.float64)
        self.kink = float(kink)
        self.dim = self.A.shape[0]
        self.mjm = None
        self.include_activation = False
        self.termination_height = -np.inf
        self.cycle = 1
        self.substeps = 1

    def _out(self, x):
        y = self.A @ x + self.b
        if self.kink:
            y = y + self.kink * np.sign(x[0])
        return y

    def roll(self, x, cycles=1, record_contacts=True):
        x = np.asarray(x, dtype=np.float64)
        return types.SimpleNamespace(
            state=self._out(x), activation=np.zeros(0), advance=1.0,
            height_min=0.0, height_max=0.0, height_range=0.0,
            contacts=(), contact_pairs=(), terminated=False, terminated_at=None,
            unstable=False, act_excursions=0,
        )

    def __call__(self, x, cycles=1):
        return self._out(np.asarray(x, dtype=np.float64))


def _unbounded(n):
    return st.Bounds(lo=np.full(n, -np.inf), hi=np.full(n, np.inf))


def test_fd_jacobian_recovers_a_known_matrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    jac = st.jacobian_fd(LinearMap(A), np.zeros(5), scales=np.ones(5), eps_rel=1e-4)
    np.testing.assert_allclose(jac.J, A, rtol=1e-8, atol=1e-8)
    assert jac.contact_stable


def test_spectrum_matches_closed_form_eigenvalues():
    """Similarity-transformed diagonal, so the eigenvalues are known but the matrix is not trivial."""
    rng = np.random.default_rng(1)
    want = np.array([2.0, -1.5, 0.5, 0.0])
    T = rng.normal(size=(4, 4))
    A = T @ np.diag(want) @ np.linalg.inv(T)
    sp = st.spectrum(A)
    np.testing.assert_allclose(np.sort(np.abs(sp.eigenvalues)), np.sort(np.abs(want)), atol=1e-8)
    assert sp.spectral_radius == pytest.approx(2.0, abs=1e-8)
    assert sp.unstable_count == 2, "both 2.0 and |-1.5| exceed 1"
    assert sp.zero_count == 1


def test_spectrum_flags_a_period_doubling_multiplier():
    sp = st.spectrum(np.diag([-1.0 + 1e-4, 0.3]))
    assert sp.period_doubling_margin < 1e-3


def test_shoot_solves_a_linear_map_to_machine_precision():
    rng = np.random.default_rng(2)
    A = 0.3 * rng.normal(size=(4, 4))
    b = rng.normal(size=4)
    want = np.linalg.solve(np.eye(4) - A, b)
    res = st.shoot(LinearMap(A, b), np.zeros(4), scales=np.ones(4),
                   bounds=_unbounded(4), classify_terminal=False)
    assert res.converged, (res.outcome, res.residual)
    # A handful of iterations rather than one: lambda starts at 1e-4 and only decays by 0.25 per
    # accepted step, so the first steps are damped. That is the deliberate trade -- LM is robust to
    # the near-singular (J - I) this hopper actually has (two exact-zero Floquet multipliers), and a
    # plain Newton step there would be arbitrarily large.
    assert res.iterations <= 6
    np.testing.assert_allclose(res.x, want, atol=1e-8)


def test_shoot_never_reports_converged_when_no_fixed_point_exists():
    """``P(x) = x + b`` has no fixed point. The solver must say so, not find one."""
    res = st.shoot(LinearMap(np.eye(3), np.ones(3)), np.zeros(3), scales=np.ones(3),
                   bounds=_unbounded(3), classify_terminal=False)
    assert not res.converged
    assert res.outcome in (st.Outcome.STALLED, st.Outcome.MAX_ITERATIONS)


def test_eps_sweep_slope_is_zero_on_a_smooth_map():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 4))
    sw = st.eps_sweep(LinearMap(A), np.ones(4), scales=np.ones(4))
    assert sw.verdict == "stable"
    assert abs(sw.slope) <= 0.01


def test_eps_sweep_slope_is_minus_one_on_a_synthetic_discontinuity():
    """A jump of fixed size divided by ``2h`` scales as ``1/eps``, so the slope is exactly -1.

    This is the one-number discriminator that separates "the solver stalled at a kink" from "the
    solver stalled at a genuine local minimum", and it is validated here on a map where the answer
    is exact rather than on the hopper where it is not.
    """
    rng = np.random.default_rng(4)
    A = rng.normal(size=(4, 4))
    x = np.zeros(4)  # sits exactly on the kink surface x[0] == 0
    sw = st.eps_sweep(LinearMap(A, kink=1.0), x, scales=np.ones(4))
    assert sw.slope == pytest.approx(-1.0, abs=0.02)
    assert sw.verdict == "discontinuous"


def test_eps_sweep_window_survives_a_single_bad_point_at_one_end():
    """Regression: one outlying step must not veto the window the rest of the grid agrees on.

    The first implementation only considered maximal runs, so the reference candidate -- whose
    spectral radius reads 2.14303 at eps 1e-2 against 2.16734 everywhere below -- was reported
    ``no_window`` despite seven decades of exact agreement. Now every contiguous sub-window is
    scanned. Reproduced here with a map whose Jacobian is deliberately eps-dependent at large eps.
    """

    class CoarseAtLargeEps(LinearMap):
        def _out(self, x):
            y = super()._out(x)
            return y + 3.0 * x[0] ** 3  # cubic term, only visible at a coarse step

    sw = st.eps_sweep(CoarseAtLargeEps(np.diag([1.7, 0.4, 0.2])), np.zeros(3),
                      scales=np.ones(3), grid=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6))
    assert sw.verdict == "stable", (sw.slope, sw.window, sw.spectral_radius)
    assert sw.window_decades >= 3


def test_cycles_to_amplitude_is_nan_for_a_stable_map():
    assert np.isnan(st.cycles_to_amplitude(0.8, 0.1))
    assert st.cycles_to_amplitude(2.0, 0.5, 2.0) == pytest.approx(2.0, abs=1e-9)


# ------------------------------------------------------------------------------ hopper fixtures

@pytest.fixture(scope="module")
def reference():
    d = np.load(resolve_model_path(REFERENCE))
    return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def candidate(reference):
    m = mujoco.MjModel.from_xml_path(resolve_model_path("assets/hopper_muscle.xml"))
    rmap = orb.ReturnMap(m, reference["ctrl"].astype(np.float64))  # activation pinned to act*
    x0 = orb.shape_state(reference["qpos"][0], reference["qvel"][0])
    return rmap, x0, reference["shape_scales"].astype(np.float64)


def test_structural_checks_license_the_state_convention(candidate):
    """Two invariants this project otherwise assumes, now asserted from the cycle Jacobian.

    Dropping ``qpos[0]`` from the state is valid only if the dynamics are translation-invariant, and
    pinning ``act`` to ``act*`` is valid only if activation does not depend on the mechanical state.
    Measured 3.9e-10 and exactly 0.0.
    """
    rmap, x0, _ = candidate
    checks = st.structural_checks(rmap, x0)
    assert checks.translation_column_error <= 1e-7
    assert checks.activation_to_mechanics == 0.0
    assert checks.activation_radius <= 1e-4  # measured 6.5e-06
    assert checks.ok()


def test_motor_hopper_structural_checks_hold_without_activation():
    """The positive control shares the code path; ``na == 0`` must not be a special case."""
    m = mujoco.MjModel.from_xml_path(resolve_model_path("assets/hopper_motor.xml"))
    rmap = orb.ReturnMap(m, np.zeros((16, m.nu)))
    checks = st.structural_checks(rmap, np.zeros(11))
    assert checks.translation_column_error <= 1e-7
    assert checks.activation_to_mechanics is None
    assert checks.ok()


def test_two_jacobian_routes_agree_at_the_candidate(candidate):
    """Full-cycle differences against composed ``mjd_transitionFD``: 5.5e-09 relative Frobenius.

    Two structurally independent estimators. The composed route cannot see a grazing contact; the
    full-cycle route can. Their agreement is the gate, and it is not close.
    """
    rmap, x0, scales = candidate
    fd = st.jacobian_fd(rmap, x0, scales=scales, eps_rel=1e-4)
    composed = st.jacobian_composed(rmap, x0)
    assert fd.contact_stable and composed.contact_stable
    rel = np.linalg.norm(composed.J - fd.J) / np.linalg.norm(fd.J)
    assert rel <= 1e-5, rel
    assert abs(composed.spectral_radius - fd.spectral_radius) / fd.spectral_radius <= 1e-4


def test_candidate_spectral_radius_and_unstable_dimension_are_pinned(candidate):
    """rho = 2.16734 with a **two**-dimensional unstable subspace out of eleven.

    The unstable dimension is the number that matters for stabilisability: two unstable directions
    against 16 phases x 6 muscles of control authority is a lot of margin.
    """
    rmap, x0, scales = candidate
    sp = st.spectrum(st.jacobian_fd(rmap, x0, scales=scales, eps_rel=1e-4).J)
    assert sp.spectral_radius == pytest.approx(CANDIDATE_RHO, abs=1e-4)
    assert sp.unstable_count == 2
    assert sp.period_doubling_margin > 0.5, "no period-doubling multiplier nearby"


def test_eps_sweep_is_stable_at_the_candidate(candidate):
    """Four and a half decades of exact agreement, zero contact-sequence changes, slope 0."""
    rmap, x0, scales = candidate
    sw = st.eps_sweep(rmap, x0, scales=scales)
    assert sw.verdict == "stable", (sw.slope, sw.window, sw.changed_columns)
    assert sw.window_decades >= 3.0
    assert abs(sw.slope) <= 0.05


def test_linear_model_holds_at_the_candidate(candidate):
    """N = 16 directions at two magnitudes. Measured cosine 1.000000, relative error 2e-5 to 6e-4.

    Two magnitudes because a bug that made ``J`` constant would still give cosine 1.0 at one step.
    """
    rmap, x0, scales = candidate
    J = st.jacobian_fd(rmap, x0, scales=scales, eps_rel=1e-4).J
    check = st.linear_model_check(rmap, x0, J, scales=scales, directions=16)
    assert check.ok()
    assert min(check.cosine_min.values()) >= 0.9999
    assert max(check.relative_error_max.values()) <= 5e-3


def test_shoot_from_the_candidate_stalls_at_a_smooth_point(candidate):
    """The headline negative, with the caveat that would have weakened it measured and removed.

    Levenberg-Marquardt improves the candidate's scaled residual from 0.410 to 0.076 -- 5.4x -- and
    then stops. The question that decides what that means is whether it stopped at a genuine local
    minimum or on a contact-event surface, where the merit function has a kink and any solver would
    stall. The eps sweep at the terminal iterate answers it: slope 0.0000 over three clean decades,
    verdict ``stable``. So this is a real local minimum of ``||F||``, not an artefact.

    What it licenses: no periodic orbit in **this start's basin**. Not "no periodic orbit" -- that
    needs multistart, and the searched radius has to be reported with the claim.
    """
    rmap, x0, scales = candidate
    res = st.shoot(rmap, x0, scales=scales)
    assert res.residual_history[0] == pytest.approx(CANDIDATE_RESIDUAL, abs=1e-3)
    assert res.residual == pytest.approx(LM_FLOOR, rel=0.05)
    assert res.residual < res.residual_history[0] / 5.0
    assert res.outcome is st.Outcome.STALLED
    assert res.terminal_sweep is not None
    assert res.terminal_sweep.verdict == "stable", "the stall is not on a contact-event surface"
    assert res.outcome is not st.Outcome.ON_EVENT_BOUNDARY


def test_bounds_read_the_joint_ranges_and_the_termination_height(candidate):
    rmap, _, _ = candidate
    b = st.Bounds.from_model(rmap.mjm)
    # qpos[1:] = height, pitch, thigh, leg, foot -> indices 0..4
    assert b.lo[0] == pytest.approx(orb.TERMINATION_HEIGHT)
    assert b.lo[2] == pytest.approx(-2.61799, abs=1e-4) and b.hi[2] == pytest.approx(0.0)
    assert b.lo[4] == pytest.approx(-0.785398, abs=1e-4)
    assert np.all(np.isinf(b.lo[5:]))  # qvel is unbounded


def test_bounds_project_and_report_active_constraints(candidate):
    rmap, _, _ = candidate
    b = st.Bounds.from_model(rmap.mjm)
    x = np.zeros(11)
    x[2] = -5.0  # past the thigh lower limit
    projected = b.project(x)
    assert projected[2] == pytest.approx(-2.61799, abs=1e-4)
    assert 2 in b.active(projected)


def test_shoot_reports_an_infeasible_start(candidate):
    """A start whose own rollout falls is not a failed solve; it is a start that cannot be scored."""
    rmap, _, scales = candidate
    x = np.zeros(11)
    x[0] = -1.0  # well below the termination height, so cycle 1 cannot complete
    assert rmap.roll(x).terminated
    res = st.shoot(rmap, x, scales=scales, classify_terminal=False)
    assert res.outcome is st.Outcome.START_INFEASIBLE
    assert res.iterations == 0


def test_engine_instability_terminates_a_rollout(candidate):
    """Found by a test, then fixed: a finite state is not the same as a trustworthy one.

    A shooting step reached a region where MuJoCo warned "Nan, Inf or huge value in QACC" while
    ``qpos`` stayed finite, so the rollout was accepted and reported an advance of -918389 m.
    ``roll`` now reads the engine's own instability flags, and such a rollout is terminated.
    """
    rmap, _, _ = candidate
    x = np.zeros(11)
    x[5] = 1e6  # an absurd forward velocity; the solver cannot integrate it
    out = rmap.roll(x)
    assert out.terminated
    assert out.unstable, "the engine's bad-QACC warning must be what stops this, not the height"
    with pytest.raises(orb.OrbitTerminated, match="numerical instability"):
        rmap(x)
