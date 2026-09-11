"""Linearisation, shooting and Floquet spectra on top of a :class:`~msk_warp.analysis.orbit.ReturnMap`.

## The one thing this module exists to get right

A hopper's return map is only **piecewise** smooth: touchdown and lift-off are events, and at a
point sitting on such an event surface the map has a kink. That matters far more than it sounds,
because a Levenberg-Marquardt solve will happily converge *onto* the kink -- the merit function has a
local minimum there -- and then report "did not converge". Read naively, that becomes "no periodic
orbit exists", which is a claim about the model made from an artefact of the solver.

So every Jacobian carries a validity certificate, and it is cheap:

* **Contact-sequence invariance.** Each perturbed rollout records `ncon` and the sorted contacting
  geom pairs at every control step. If any perturbation changes that sequence, the finite difference
  straddled an event and the column is flagged.
* **The eps-sweep slope test.** A jump discontinuity of fixed size divided by `2h` scales as
  `1/eps`, so `d log||J||_F / d log eps` is **-1** at a discontinuity and **0** on a smooth map.
  Measured on this hopper: 0 at a clean point, and `||J||_F` climbing from 4.7e3 at eps 1e-5 to
  4.3e5 at 1e-7 with 11 of 11 columns changing the contact sequence at a bad one.

Without those, a negative result is uninterpretable. With them, `Outcome.ON_EVENT_BOUNDARY` is a
distinct and honest verdict: *inconclusive about the orbit*, rather than evidence against it.

## Two Jacobian routes, and the gate is their agreement

:func:`jacobian_fd` differences the whole nonlinear cycle; :func:`jacobian_composed` multiplies
`mjd_transitionFD`'s per-substep linearisations. They cost about the same and they fail differently:
the composed route perturbs inside one substep so it rarely flips an active set, but it therefore
*assumes* the active set is frozen and will return a smooth-looking matrix at a grazing point. Only
the full-cycle route can see that. Two structurally independent estimators agreeing is much stronger
evidence than either alone, so :func:`jacobian` computes both and refuses to report a spectrum when
they disagree.

The composed route also yields two structural assertions for free, both of which this project
otherwise takes on faith:

* the **translation column** of the full cycle matrix must equal `e_0`, which is what licenses
  dropping `qpos[0]` from the state at all; and
* the **activation-to-mechanics block** must be zero, which is what licenses pinning `act` to
  `act*` and shooting the mechanical state alone (``orbit.py``, and ``docs/VALIDITY.md`` CL-02).

## What the spectrum does and does not license

`rho > 1` means the orbit is real but not realisable open loop; it needs feedback. It does **not**
give a reliable cycles-to-fall figure: on the one case where the answer is known -- the trained motor
gait, which falls at 3.74 cycles of cycled open-loop replay -- the naive `ln(D/d0)/ln(rho)` arithmetic
predicts 1.8 to 2.1 and is short by a factor of ~1.8 (``docs/VALIDITY.md`` CL-09).
:func:`cycles_to_amplitude` is therefore provided and is **reported, never gated**.

numpy only: `scipy` is deliberately absent from this venv (``docs/VALIDITY.md`` IN-14).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional, Sequence

import mujoco
import numpy as np

from msk_warp.analysis.orbit import (
    HEIGHT_INDEX,
    OrbitTerminated,
    ReturnMap,
    TERMINATION_HEIGHT,
)

#: Finite-difference steps swept when certifying a Jacobian, in units of component scale.
DEFAULT_EPS_GRID = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6, 1e-7)

#: Below this slope of ``log||J||_F`` against ``log eps``, the point is on a discontinuity.
DISCONTINUITY_SLOPE = -0.5

#: Advance per cycle below which a converged orbit is the standing basin, not a gait.
#: Inherited from ``scripts/inspect_trajopt_candidate.py``'s calibrated ``STANDING_ADVANCE``.
STANDING_ADVANCE = 0.05


# ----------------------------------------------------------------------------------- Jacobians

@dataclasses.dataclass(frozen=True)
class Jacobian:
    """A cycle Jacobian with the evidence needed to decide whether to believe it."""

    J: np.ndarray
    method: str
    eps_rel: Optional[float]
    changed_columns: tuple
    """Columns whose perturbed rollout changed the contact sequence. Non-empty means the finite
    difference straddled a contact event and this matrix describes two different dynamics."""
    terminated_columns: tuple

    @property
    def contact_stable(self) -> bool:
        return not self.changed_columns and not self.terminated_columns

    @property
    def frobenius(self) -> float:
        return float(np.linalg.norm(self.J))

    @property
    def spectral_radius(self) -> float:
        return float(np.abs(np.linalg.eigvals(self.J)).max())


def jacobian_fd(rmap: ReturnMap, x, *, scales, eps_rel: float = 1e-4) -> Jacobian:
    """Full-cycle central differences, ``2d`` rollouts, step ``eps_rel * scales[j]`` per component.

    Per-component scaling matters: the hopper's components span 0.059 (torso height) to 8.84 (foot
    angular rate), so one absolute step would be far too coarse for some and inside solver noise for
    others.
    """
    x = np.asarray(x, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    n = rmap.dim
    nominal = rmap.roll(x)
    J = np.zeros((n, n))
    changed, terminated = [], []

    for j in range(n):
        h = eps_rel * scales[j]
        col_ok = True
        outs = []
        for sign in (+1.0, -1.0):
            xp = x.copy()
            xp[j] += sign * h
            out = rmap.roll(xp)
            if out.terminated:
                terminated.append(j)
                col_ok = False
                break
            if out.contact_pairs != nominal.contact_pairs:
                col_ok = False
                if j not in changed:
                    changed.append(j)
            outs.append(out.state)
        if len(outs) == 2:
            J[:, j] = (outs[0] - outs[1]) / (2.0 * h)
        else:
            J[:, j] = np.nan
        del col_ok

    return Jacobian(J=J, method="fd", eps_rel=eps_rel,
                    changed_columns=tuple(sorted(set(changed))),
                    terminated_columns=tuple(sorted(set(terminated))))


def _transition_indices(mjm, include_activation: bool):
    """Map ``mjd_transitionFD``'s ``(qpos-tangent, qvel, act)`` order onto the state vector.

    Valid only because ``nq == nv`` with slide/hinge joints only, so the tangent equals ``qpos``
    element for element -- asserted by :func:`~msk_warp.analysis.orbit.assert_state_layout`.
    """
    nv, na = mjm.nv, mjm.na
    idx = list(range(1, nv)) + list(range(nv, 2 * nv))
    if include_activation:
        idx += list(range(2 * nv, 2 * nv + na))
    return np.array(idx, dtype=int)


def jacobian_composed(rmap: ReturnMap, x, *, eps: float = 1e-6) -> Jacobian:
    """Product of ``mjd_transitionFD`` linearisations over every substep of the cycle.

    Also returns the two structural diagnostics in :func:`structural_checks`, which read off the
    *full* ``(2nv + na)`` product rather than the reduced state.
    """
    full, _, changed, terminated = _composed_full(rmap, x, eps=eps)
    idx = _transition_indices(rmap.mjm, rmap.include_activation)
    J = full[np.ix_(idx, idx)] if full is not None else np.full((rmap.dim, rmap.dim), np.nan)
    return Jacobian(J=J, method="composed", eps_rel=None,
                    changed_columns=tuple(changed), terminated_columns=tuple(terminated))


def _composed_full(rmap: ReturnMap, x, *, eps: float = 1e-6):
    """``(A_full, per_step_A, changed, terminated)`` for the full ``(2nv + na)`` state."""
    mjm = rmap.mjm
    d = rmap._data
    nstate = 2 * mjm.nv + mjm.na
    qpos, qvel, act = rmap._unpack(x)

    mujoco.mj_resetData(mjm, d)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    if mjm.na:
        d.act[:] = act
    mujoco.mj_forward(mjm, d)

    total = np.eye(nstate)
    per_step = []
    A = np.zeros((nstate, nstate))
    B = np.zeros((nstate, mjm.nu))
    for t in range(rmap.cycle):
        d.ctrl[:] = rmap.ctrl_seq[t]
        for _ in range(rmap.substeps):
            mujoco.mjd_transitionFD(mjm, d, eps, True, A, B, None, None)
            total = A @ total
            mujoco.mj_step(mjm, d)
            if not np.isfinite(d.qpos).all():
                return None, per_step, (), tuple(range(rmap.dim))
        per_step.append(A.copy())
        if float(d.qpos[HEIGHT_INDEX]) < rmap.termination_height:
            return None, per_step, (), tuple(range(rmap.dim))
    return total, per_step, (), ()


@dataclasses.dataclass(frozen=True)
class StructuralChecks:
    """The two invariants that license this module's state convention."""

    translation_column_error: float
    """``|A_full[:, 0] - e_0|_inf``. Non-zero means the dynamics are not translation-invariant in
    ``qpos[0]``, so quotienting it out of the state is invalid. Measured 4.2e-10; gate 1e-7."""

    activation_to_mechanics: Optional[float]
    """``|A_full[act rows, mech cols]|_inf``. Non-zero means activation depends on the mechanical
    state, so pinning ``act`` to ``act*`` is invalid. Measured bitwise 0.0."""

    activation_radius: Optional[float]
    """Spectral radius of the activation diagonal block -- how fast ``act`` forgets its start."""

    def ok(self, *, translation_tol: float = 1e-7, coupling_tol: float = 1e-12) -> bool:
        if self.translation_column_error > translation_tol:
            return False
        if self.activation_to_mechanics is not None:
            return self.activation_to_mechanics <= coupling_tol
        return True


def structural_checks(rmap: ReturnMap, x, *, eps: float = 1e-6) -> StructuralChecks:
    full, _, _, _ = _composed_full(rmap, x, eps=eps)
    if full is None:
        return StructuralChecks(np.inf, None, None)
    nv, na = rmap.mjm.nv, rmap.mjm.na
    e0 = np.zeros(full.shape[0])
    e0[0] = 1.0
    trans = float(np.abs(full[:, 0] - e0).max())
    if na == 0:
        return StructuralChecks(trans, None, None)
    a = slice(2 * nv, 2 * nv + na)
    m = slice(0, 2 * nv)
    return StructuralChecks(
        translation_column_error=trans,
        activation_to_mechanics=float(np.abs(full[a, m]).max()),
        activation_radius=float(np.abs(np.linalg.eigvals(full[a, a])).max()),
    )


# -------------------------------------------------------------------------------- certification

@dataclasses.dataclass(frozen=True)
class EpsSweep:
    """Is this point smooth, and over what range of finite-difference step?"""

    eps_rel: tuple
    spectral_radius: tuple
    frobenius: tuple
    changed_columns: tuple
    slope: float
    window: Optional[tuple]
    window_decades: float
    verdict: str

    @property
    def ok(self) -> bool:
        return self.verdict == "stable"


def eps_sweep(rmap: ReturnMap, x, *, scales, grid: Sequence[float] = DEFAULT_EPS_GRID,
              rho_tol: float = 1e-3, min_decades: int = 3,
              slope_tol: float = 0.05) -> EpsSweep:
    """Sweep the finite-difference step and decide whether ``x`` sits on a contact-event surface.

    ``verdict`` is one of

    * ``"stable"`` -- a run of at least ``min_decades`` consecutive steps over which the spectral
      radius varies by at most ``rho_tol`` relatively, with no contact-sequence change, and
      ``|slope|`` within ``slope_tol``;
    * ``"discontinuous"`` -- ``slope <= DISCONTINUITY_SLOPE``, the ``1/eps`` signature of a jump;
    * ``"no_window"`` -- smooth-looking but no stable run, so nothing here is trustworthy.
    """
    eps_list, rhos, frobs, changed = [], [], [], []
    for e in grid:
        jac = jacobian_fd(rmap, x, scales=scales, eps_rel=e)
        eps_list.append(float(e))
        frobs.append(jac.frobenius if np.isfinite(jac.J).all() else np.nan)
        rhos.append(jac.spectral_radius if np.isfinite(jac.J).all() else np.nan)
        changed.append(len(jac.changed_columns) + len(jac.terminated_columns))

    e_arr = np.array(eps_list)
    r_arr = np.array(rhos)
    f_arr = np.array(frobs)
    c_arr = np.array(changed)
    order = np.argsort(e_arr)  # ascending eps

    # Scan EVERY contiguous sub-window, not just maximal clean runs. A single bad point at one end
    # of the grid must not veto the window the rest of it agrees on -- measured: at the reference
    # candidate, eps 1e-2 reads rho 2.14303 against 2.16734 everywhere below it, which is a 1.1%
    # spread and would fail rho_tol for the whole 8-point range while a 7-point sub-window is exact.
    clean = np.isfinite(r_arr) & (r_arr > 0) & (c_arr == 0)
    window, decades, best = None, 0.0, None
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            idx = order[a:b + 1]
            if not clean[idx].all():
                continue
            r = r_arr[idx]
            if (r.max() / r.min() - 1.0) > rho_tol:
                continue
            span = float(np.log10(e_arr[idx].max()) - np.log10(e_arr[idx].min()))
            if span > decades:
                decades, window, best = span, (float(e_arr[idx].min()), float(e_arr[idx].max())), idx

    # Fit the slope on the widest CLEAN run only. Including points whose perturbation crossed a
    # contact event mixes two regimes and the fit becomes meaningless -- measured at an LM terminal
    # iterate: ||J||_F of [16.5, 75.6, 215.3, 623.4, 7.92, 7.92, 7.92, 7.92] across the grid, where
    # the first four columns changed the contact sequence and the last four did not. Fitting all
    # eight gives +0.24 and hides the fact that the small-eps end is perfectly smooth.
    fit = best if best is not None and len(best) >= 3 else order[clean[order]]
    slope = float("nan")
    if len(fit) >= 3 and np.all(f_arr[fit] > 0):
        slope = float(np.polyfit(np.log10(e_arr[fit]), np.log10(f_arr[fit]), 1)[0])

    if np.isfinite(slope) and slope <= DISCONTINUITY_SLOPE:
        verdict = "discontinuous"
    elif decades >= min_decades and (not np.isfinite(slope) or abs(slope) <= slope_tol):
        verdict = "stable"
    else:
        verdict = "no_window"

    return EpsSweep(tuple(eps_list), tuple(rhos), tuple(frobs), tuple(changed),
                    slope, window, decades, verdict)


@dataclasses.dataclass(frozen=True)
class LinearModelCheck:
    """Does ``P(x + d) - P(x)`` actually look like ``J d``? Project rule: N >= 10 directions."""

    directions: int
    eps_rel: tuple
    cosine_mean: dict
    cosine_min: dict
    relative_error_mean: dict
    relative_error_max: dict

    def ok(self, *, cosine_floor: float = 0.999, error_ceiling: float = 5e-3) -> bool:
        return (min(self.cosine_min.values()) >= cosine_floor
                and max(self.relative_error_mean.values()) <= error_ceiling)


def linear_model_check(rmap: ReturnMap, x, J, *, scales, directions: int = 16,
                       eps_rel: Sequence[float] = (1e-4, 1e-3), seed: int = 0) -> LinearModelCheck:
    """Compare the Jacobian against the nonlinear map over random directions, at two magnitudes.

    Two magnitudes, not one: a bug that made ``J`` constant would still give cosine 1.0 at a single
    step, but could not also produce first-order error scaling between decades.
    """
    x = np.asarray(x, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    rng = np.random.default_rng(seed)
    P0 = rmap(x)
    cos_mean, cos_min, err_mean, err_max = {}, {}, {}, {}

    for e in eps_rel:
        cosines, errors = [], []
        for _ in range(directions):
            u = rng.normal(size=rmap.dim)
            u /= np.linalg.norm(u / scales)
            delta = e * u * scales
            try:
                actual = rmap(x + delta) - P0
            except OrbitTerminated:
                continue
            predicted = J @ delta
            na, np_ = np.linalg.norm(actual), np.linalg.norm(predicted)
            if na == 0 or np_ == 0:
                continue
            cosines.append(float(actual @ predicted / (na * np_)))
            errors.append(float(np.linalg.norm(actual - predicted) / na))
        cos_mean[float(e)] = float(np.mean(cosines)) if cosines else float("nan")
        cos_min[float(e)] = float(np.min(cosines)) if cosines else float("nan")
        err_mean[float(e)] = float(np.mean(errors)) if errors else float("nan")
        err_max[float(e)] = float(np.max(errors)) if errors else float("nan")

    return LinearModelCheck(directions, tuple(float(e) for e in eps_rel),
                            cos_mean, cos_min, err_mean, err_max)


# ------------------------------------------------------------------------------------- spectrum

@dataclasses.dataclass(frozen=True)
class Spectrum:
    eigenvalues: np.ndarray
    spectral_radius: float
    unstable_count: int
    unit_count: int
    zero_count: int
    period_doubling_margin: float
    """``min |lambda - (-1)|``. Small means a period-doubling bifurcation is nearby, i.e. the true
    period may be a multiple of this one."""
    condition: float

    def stable(self, tol: float = 1.0) -> bool:
        return self.spectral_radius <= tol


def spectrum(J, *, unit_tol: float = 1e-3, zero_tol: float = 1e-10) -> Spectrum:
    J = np.asarray(J, dtype=np.float64)
    ev = np.linalg.eigvals(J)
    ev = ev[np.argsort(-np.abs(ev))]
    mag = np.abs(ev)
    return Spectrum(
        eigenvalues=ev,
        spectral_radius=float(mag.max()),
        unstable_count=int((mag > 1.0 + unit_tol).sum()),
        unit_count=int((np.abs(mag - 1.0) <= unit_tol).sum()),
        zero_count=int((mag <= zero_tol).sum()),
        period_doubling_margin=float(np.abs(ev + 1.0).min()),
        condition=float(np.linalg.cond(J)),
    )


def cycles_to_amplitude(rho: float, initial: float, target: float = 1.0) -> float:
    """``ln(target/initial) / ln(rho)``; NaN when ``rho <= 1``.

    **Reported, never gated.** On the trained motor gait, the only case where the answer is known,
    this underestimates the measured 3.74 cycles by a factor of ~1.8 (``docs/VALIDITY.md`` CL-09).
    """
    if not np.isfinite(rho) or rho <= 1.0 or initial <= 0:
        return float("nan")
    return float(np.log(target / initial) / np.log(rho))


# ------------------------------------------------------------------------------------- shooting

class Outcome(str, enum.Enum):
    CONVERGED = "converged"
    STANDING = "standing"
    ON_EVENT_BOUNDARY = "on_event_boundary"
    BOUND_ACTIVE = "bound_active"
    TERMINATED = "terminated"
    STALLED = "stalled"
    MAX_ITERATIONS = "max_iterations"
    START_INFEASIBLE = "start_infeasible"


@dataclasses.dataclass(frozen=True)
class Bounds:
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def from_model(cls, mjm, *, include_activation: bool = False,
                   termination_height: float = TERMINATION_HEIGHT,
                   margin: float = 0.0) -> "Bounds":
        """Joint ranges for ``qpos[1:]``, the termination height as a floor, ``act`` in [0, 1].

        Projection rather than a ``tanh`` reparameterisation, deliberately: a reparameterisation
        hides the boundary, whereas projection plus :meth:`active` reports it. A fixed point sitting
        on a joint limit is physically real but its Jacobian there is one-sided, and that has to be
        visible rather than smoothed away.
        """
        nq, nv, na = mjm.nq, mjm.nv, mjm.na
        lo = np.full(nq - 1, -np.inf)
        hi = np.full(nq - 1, np.inf)
        for j in range(mjm.njnt):
            adr = mjm.jnt_qposadr[j]
            if adr == 0 or not mjm.jnt_limited[j]:
                continue
            lo[adr - 1] = mjm.jnt_range[j, 0] + margin
            hi[adr - 1] = mjm.jnt_range[j, 1] - margin
        lo[HEIGHT_INDEX - 1] = max(lo[HEIGHT_INDEX - 1], termination_height + margin)
        lo = np.concatenate([lo, np.full(nv, -np.inf)])
        hi = np.concatenate([hi, np.full(nv, np.inf)])
        if include_activation and na:
            lo = np.concatenate([lo, np.zeros(na)])
            hi = np.concatenate([hi, np.ones(na)])
        return cls(lo=lo, hi=hi)

    def project(self, x) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float64), self.lo, self.hi)

    def active(self, x, tol: float = 1e-9) -> tuple:
        x = np.asarray(x, dtype=np.float64)
        return tuple(int(i) for i in range(len(x))
                     if x[i] - self.lo[i] <= tol or self.hi[i] - x[i] <= tol)


@dataclasses.dataclass
class ShootResult:
    x: np.ndarray
    x0: np.ndarray
    residual: float
    residual_history: tuple
    outcome: Outcome
    iterations: int
    lam_final: float
    active_bounds: tuple
    advance: float
    height_range: float
    terminal_sweep: Optional[EpsSweep] = None

    @property
    def converged(self) -> bool:
        return self.outcome is Outcome.CONVERGED


def shoot(rmap: ReturnMap, x0, *, scales, bounds: Optional[Bounds] = None,
          tol: float = 1e-10, max_iter: int = 60, lam0: float = 1e-4,
          lam_down: float = 0.25, lam_up: float = 10.0, max_backtracks: int = 16,
          stall_window: int = 5, stall_rel: float = 1e-3,
          standing_advance: float = STANDING_ADVANCE,
          classify_terminal: bool = True, cycles: int = 1) -> ShootResult:
    """Levenberg-Marquardt on ``F(x) = (P(x) - x) / scales``, seeking a periodic orbit.

    LM rather than damped Newton: with two exact-zero Floquet multipliers measured on this hopper,
    ``J - I`` is near-singular in a couple of directions and a plain Newton step there is
    arbitrarily large. ``lambda`` *is* the line search -- large ``lambda`` gives a short gradient
    step -- so there is no separate step-length ladder. One knob.

    A trial step is **rejected outright** if its rollout terminates, rather than penalised, which
    keeps the residual well defined throughout.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if bounds is None:
        bounds = Bounds.from_model(rmap.mjm, include_activation=rmap.include_activation)
    n = rmap.dim
    ident = np.eye(n)

    def merit(x):
        try:
            r = (rmap(x, cycles) - x) / scales
        except OrbitTerminated:
            return None
        return float(np.sqrt(np.mean(r ** 2))), r

    start = merit(x0)
    if start is None:
        result = rmap.roll(x0, cycles)
        return ShootResult(x0.copy(), x0.copy(), float("inf"), (), Outcome.START_INFEASIBLE,
                           0, lam0, bounds.active(x0), result.advance, result.height_range)

    x = x0.copy()
    cost, F = start
    history = [cost]
    lam = lam0
    outcome = Outcome.MAX_ITERATIONS
    iterations = 0

    for iterations in range(1, max_iter + 1):
        if cost <= tol:
            outcome = Outcome.CONVERGED
            break
        jac = jacobian_fd(rmap, x, scales=scales, eps_rel=1e-4)
        if not np.isfinite(jac.J).all():
            outcome = Outcome.TERMINATED
            break
        A = (jac.J - ident) / scales[:, None]
        JTJ = A.T @ A
        g = A.T @ F
        diag = np.diag(np.diag(JTJ)) + 1e-14 * np.eye(n)

        improved = False
        for _ in range(max_backtracks):
            try:
                step = np.linalg.solve(JTJ + lam * diag, -g)
            except np.linalg.LinAlgError:
                lam *= lam_up
                continue
            trial = bounds.project(x + step)
            got = merit(trial)
            if got is not None and got[0] < cost:
                x, (cost, F) = trial, got
                lam = max(lam * lam_down, 1e-14)
                improved = True
                break
            lam *= lam_up
        history.append(cost)
        if not improved:
            outcome = Outcome.STALLED
            break
        if len(history) > stall_window:
            recent = history[-(stall_window + 1):]
            if recent[0] > 0 and (recent[0] - recent[-1]) / recent[0] < stall_rel:
                outcome = Outcome.STALLED
                break
    else:
        outcome = Outcome.CONVERGED if cost <= tol else Outcome.MAX_ITERATIONS

    final = rmap.roll(x, cycles)
    active = bounds.active(x)

    sweep = None
    if classify_terminal and outcome is not Outcome.CONVERGED:
        sweep = eps_sweep(rmap, x, scales=scales)
        if sweep.verdict == "discontinuous":
            outcome = Outcome.ON_EVENT_BOUNDARY

    if outcome is Outcome.CONVERGED and abs(final.advance) < standing_advance:
        outcome = Outcome.STANDING
    elif outcome is Outcome.CONVERGED and active:
        outcome = Outcome.BOUND_ACTIVE

    return ShootResult(x=x, x0=x0.copy(), residual=cost, residual_history=tuple(history),
                       outcome=outcome, iterations=iterations, lam_final=lam,
                       active_bounds=active, advance=final.advance,
                       height_range=final.height_range, terminal_sweep=sweep)


def multistart(rmap: ReturnMap, starts, **kwargs) -> list:
    """Shoot from each start. Returns one :class:`ShootResult` per start, infeasible ones included.

    A negative result is only as strong as the number and spread of starts, so nothing is dropped:
    the outcome histogram is the claim.
    """
    return [shoot(rmap, s, **kwargs) for s in starts]
