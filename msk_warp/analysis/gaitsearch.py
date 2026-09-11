"""Joint ``(x, u)`` shooting: search over the control as well as the state, by continuation.

## Why the sweep could not answer the question, and this can

:mod:`msk_warp.analysis.stability`'s :func:`~msk_warp.analysis.stability.shoot` takes a control as
given and solves ``P(x; u) = x`` for ``x`` alone -- eleven equations in eleven unknowns, a square
root-find. That answered "is *this candidate* a periodic orbit?" for all eleven Phase 4 cells, and
nine times the answer was no. It cannot answer "does a forward-travelling muscle orbit *exist*",
because it never varies the control.

Letting ``u`` free makes it eleven equations in ``11 + T_c * nu`` unknowns -- 203 for the ``T_c`` 32
muscle hopper. Massively underdetermined, so periodicity alone has a solution *manifold* rather than
isolated roots, and the problem stops being a root-find and becomes an optimisation over that
manifold.

The formulation here stacks a velocity target onto the residual::

    F(x, u) = [ (P(x; u) - x) / scales        # 11, periodicity
                w * (v(x, u) - v_target) ]    # 1,  how fast the orbit travels

and then **continues**: start from an orbit that is known to exist, step ``v_target``, re-solve,
repeat. Each solve lands on the periodic manifold *and* hits the speed, so a successful run traces a
one-parameter family of genuine limit cycles and reports how each one's Floquet spectrum, flight
phase and hop amplitude evolve along it. A failed solve is informative too: it is a fold, or the
edge of feasibility, and it says where the capability actually ends.

That is a far better instrument than a single pass/fail. "How fast can a physiological muscle hopper
go on a periodic orbit, and what does it cost in stability?" is answered by a curve.

## Two things the parameterisation has to get right

**The control box is a hard constraint, unlike joint limits.** ``ctrlrange`` is ``[0, 1]`` with
``ctrllimited`` true and ``mjDSBL_CLAMPCTRL`` unset, and the engine clamps: measured, ``ctrl = 1.0``
and ``ctrl = 5.0`` give identical ``act_dot``. So projecting onto the box is correct -- the opposite
of the soft ``jnt_range`` trap that broke the first solver (``docs/VALIDITY.md`` IN-15). It also
means a **saturated cell has exactly zero derivative**, so the control is held strictly inside the
box by at least the finite-difference step, or a central difference at the edge would silently
report half the true slope.

**Activation is not an unknown.** ``act*`` is determined by ``u`` (``docs/VALIDITY.md`` CL-02), so it
is recomputed whenever the control changes -- warm-started from the previous value, which converges
in one or two cycles because the activation map contracts at a measured rate of about 1e-6 per cycle.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence

import numpy as np

from msk_warp.analysis import stability as st
from msk_warp.analysis.orbit import OrbitTerminated, ReturnMap, activation_limit_cycle

#: Keep every control cell at least this far inside ``ctrlrange``. Must exceed the finite-difference
#: step, or a central difference at a clamped cell reports half the one-sided slope.
CTRL_MARGIN = 1e-3

#: Finite-difference step for a control cell, absolute (``ctrl`` is dimensionless on [0, 1]).
CTRL_EPS = 1e-4

#: Finite-difference step for a state component, in units of that component's scale.
STATE_EPS_REL = 1e-4


@dataclasses.dataclass
class GaitProblem:
    """The joint residual ``F(x, u)`` and its Jacobian, for one model at one cycle length."""

    rmap: ReturnMap
    scales: np.ndarray
    velocity_weight: float = 1.0
    ctrl_margin: float = CTRL_MARGIN
    ctrl_scale: float = 1.0
    ctrl_lo: float = 0.0
    ctrl_hi: float = 1.0

    def __post_init__(self):
        self.scales = np.asarray(self.scales, dtype=np.float64)
        self.n_state = self.rmap.dim
        self.n_ctrl = self.rmap.cycle * self.rmap.mjm.nu
        self.dim = self.n_state + self.n_ctrl
        # The metric the minimum-norm step is measured in. Without it, "smallest step" is taken in
        # raw units where the state components span 0.059 (torso height) to 8.84 (foot angular
        # rate), so the notion of small is meaningless and the solver systematically under-uses the
        # large-scale directions. Measured effect on the reachable velocity fraction at the muscle
        # bob orbit: 0.0055 raw against 0.0114 scaled, a factor of two.
        self.theta_scale = np.concatenate(
            [self.scales, np.full(self.n_ctrl, float(self.ctrl_scale))]
        )
        rng = self.rmap.mjm.actuator_ctrlrange
        if bool(self.rmap.mjm.actuator_ctrllimited.all()):
            self.ctrl_lo = float(rng[:, 0].max())
            self.ctrl_hi = float(rng[:, 1].min())

    # ------------------------------------------------------------------ packing

    def pack(self, x, ctrl) -> np.ndarray:
        return np.concatenate([np.asarray(x, dtype=np.float64).ravel(),
                               np.asarray(ctrl, dtype=np.float64).ravel()])

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        x = theta[: self.n_state]
        ctrl = theta[self.n_state:].reshape(self.rmap.cycle, self.rmap.mjm.nu)
        return x, ctrl

    def project(self, theta) -> np.ndarray:
        """Clamp the control strictly inside its range. The box is real; see the module docstring."""
        theta = np.array(theta, dtype=np.float64, copy=True)
        lo, hi = self.ctrl_lo + self.ctrl_margin, self.ctrl_hi - self.ctrl_margin
        np.clip(theta[self.n_state:], lo, hi, out=theta[self.n_state:])
        return theta

    def saturated(self, theta, tol=None) -> int:
        tol = self.ctrl_margin * 1.001 if tol is None else tol
        _, ctrl = self.unpack(theta)
        return int(((ctrl - self.ctrl_lo <= tol) | (self.ctrl_hi - ctrl <= tol)).sum())

    # ------------------------------------------------------------------ residual

    def _apply(self, ctrl, warm=True):
        self.rmap.set_control(ctrl, warm_start=warm)

    def evaluate(self, theta, v_target):
        """``(residual, info)``, or ``(None, info)`` if the rollout did not survive the cycle."""
        x, ctrl = self.unpack(theta)
        self._apply(ctrl)
        out = self.rmap.roll(x)
        info = {
            "terminated": out.terminated,
            "advance": out.advance,
            "velocity": out.advance / (self.rmap.cycle * self.rmap.control_dt),
            "height_range": out.height_range,
            "flight_steps": int(sum(1 for n in out.contacts if n == 0)),
            "stance_steps": int(sum(1 for n in out.contacts if n > 0)),
        }
        if out.terminated:
            return None, info
        residual = np.empty(self.n_state + 1)
        residual[: self.n_state] = (out.state - x) / self.scales
        residual[self.n_state] = self.velocity_weight * (info["velocity"] - v_target)
        info["periodicity"] = float(np.sqrt(np.mean(residual[: self.n_state] ** 2)))
        info["velocity_error"] = float(info["velocity"] - v_target)
        return residual, info

    def cost(self, residual) -> float:
        return float(np.sqrt(np.mean(np.asarray(residual) ** 2)))

    def jacobian(self, theta, v_target, *, state_eps_rel=STATE_EPS_REL, ctrl_eps=CTRL_EPS):
        """Central differences over all ``11 + T_c * nu`` unknowns.

        The dominant cost is ``2 * dim`` rollouts. Perturbing a state component leaves ``act*``
        untouched, so only the control columns pay for its (warm-started) recomputation.
        """
        A = np.zeros((self.n_state + 1, self.dim))
        steps = np.empty(self.dim)
        steps[: self.n_state] = state_eps_rel * self.scales
        steps[self.n_state:] = ctrl_eps
        for j in range(self.dim):
            h = steps[j]
            plus, minus = theta.copy(), theta.copy()
            plus[j] += h
            minus[j] -= h
            fp, _ = self.evaluate(plus, v_target)
            fm, _ = self.evaluate(minus, v_target)
            if fp is None or fm is None:
                A[:, j] = np.nan
                continue
            A[:, j] = (fp - fm) / (2.0 * h)
        return A


@dataclasses.dataclass
class MultipleShootingProblem:
    """Periodic-orbit search split into segments, with the segment-start states as unknowns.

    **Single shooting does not work on this hopper, and the reason is measured.** At the ``T_c`` 16
    muscle candidate the joint periodicity Jacobian is well conditioned (singular values 8.20 down
    to 0.111, condition 74), but the Gauss-Newton step it implies needs ``max |dctrl| = 2.1`` while
    the linear model holds only out to about ``0.02``: at ``k = 0.01`` of the full step predicted
    and actual residuals agree (1.3461 against 1.3513), at ``k = 1`` they are 1.6e-13 against 6.55.
    So each admissible step buys about 0.6%, and 40 iterations moved the residual 0.410 to 0.322.

    The cause is amplification rather than conditioning: sixteen steps of a map with ``rho > 2`` turn
    any control change into a touchdown-timing change and the contact sequence moves. Splitting the
    cycle caps that at ``L = T_c / S`` steps per segment, which is the standard remedy and is also
    **cheaper** here, because the Jacobian becomes local -- perturbing one unknown re-rolls one
    segment rather than the whole cycle.

    The segment state is the **full** state ``(qpos[1:], qvel, act)``. Carrying activation as an
    unknown with its own continuity residual is what keeps the Jacobian local; otherwise every
    control cell reaches every segment through ``act*``. It also sidesteps CL-01 entirely, because
    activation continuity here is an equation solved to zero rather than a quantity weighed against
    a ruler, so no activation scale has to be chosen.
    """

    rmap: ReturnMap
    scales: np.ndarray
    segments: int = 4
    velocity_weight: float = 0.0
    ctrl_margin: float = CTRL_MARGIN
    ctrl_scale: float = 1.0
    act_scale: float = 1.0

    def __post_init__(self):
        self.scales = np.asarray(self.scales, dtype=np.float64)
        if self.rmap.cycle % self.segments:
            raise ValueError(
                f"cycle {self.rmap.cycle} is not divisible by segments {self.segments}")
        self.length = self.rmap.cycle // self.segments
        self.na = int(self.rmap.mjm.na)
        self.n_mech = self.rmap.mjm.nq - 1 + self.rmap.mjm.nv
        self.n_node = self.n_mech + self.na
        self.n_ctrl = self.rmap.cycle * self.rmap.mjm.nu
        self.dim = self.segments * self.n_node + self.n_ctrl
        self.n_res = self.segments * self.n_node + (1 if self.velocity_weight else 0)
        self.node_scale = np.concatenate([self.scales, np.full(self.na, self.act_scale)])
        self.theta_scale = np.concatenate(
            [np.tile(self.node_scale, self.segments), np.full(self.n_ctrl, self.ctrl_scale)])
        #: Subtracted from the residual. Zero is the true problem; see :meth:`set_homotopy`.
        self.offset = np.zeros(self.n_res)
        rng = self.rmap.mjm.actuator_ctrlrange
        self.ctrl_lo, self.ctrl_hi = 0.0, 1.0
        if bool(self.rmap.mjm.actuator_ctrllimited.all()):
            self.ctrl_lo = float(rng[:, 0].max())
            self.ctrl_hi = float(rng[:, 1].min())

    def pack(self, nodes, ctrl) -> np.ndarray:
        return np.concatenate([np.asarray(nodes, dtype=np.float64).ravel(),
                               np.asarray(ctrl, dtype=np.float64).ravel()])

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        cut = self.segments * self.n_node
        return (theta[:cut].reshape(self.segments, self.n_node),
                theta[cut:].reshape(self.rmap.cycle, self.rmap.mjm.nu))

    def project(self, theta) -> np.ndarray:
        theta = np.array(theta, dtype=np.float64, copy=True)
        cut = self.segments * self.n_node
        lo, hi = self.ctrl_lo + self.ctrl_margin, self.ctrl_hi - self.ctrl_margin
        np.clip(theta[cut:], lo, hi, out=theta[cut:])
        return theta

    def saturated(self, theta, tol=None) -> int:
        tol = self.ctrl_margin * 1.001 if tol is None else tol
        _, ctrl = self.unpack(theta)
        return int(((ctrl - self.ctrl_lo <= tol) | (self.ctrl_hi - ctrl <= tol)).sum())

    def seed(self, x0, ctrl) -> np.ndarray:
        """Starting guess: simulate the cycle once and sample the segment boundaries."""
        self.rmap.set_control(ctrl, warm_start=False)
        act = self.rmap.activation if self.na else np.zeros(0)
        nodes = np.empty((self.segments, self.n_node))
        x = np.asarray(x0, dtype=np.float64)
        for s in range(self.segments):
            nodes[s] = np.concatenate([x, act])
            x, act, _, _, _ = self.rmap.segment(x, act, s * self.length, self.length)
        return self.project(self.pack(nodes, ctrl))

    def _segment(self, node, s):
        return self.rmap.segment(node[: self.n_mech], node[self.n_mech:],
                                 s * self.length, self.length)

    def set_homotopy(self, theta, alpha: float, v_target: float = 0.0) -> None:
        """Demand only a fraction ``alpha`` of the closing gap, measured at ``theta``.

        Needed because the direct problem is far outside any Newton basin. Seeding from a simulated
        cycle puts the entire error in the wraparound block -- 1.36 RMS scale-units for the ``T_c``
        16 muscle candidate -- and no amount of segmenting or box relaxation closes that: with the
        control box widened to [-3, 4] and **zero** cells saturated, the solve stalls at exactly the
        same 1.9431e-02 it reaches inside [0, 1], so the obstacle is the size of the gap, not control
        authority.

        The homotopy replaces ``F`` with ``F - (1 - alpha) * F_0``, which is identically zero at
        ``alpha = 0`` and the true problem at ``alpha = 1``. Stepping ``alpha`` up and re-solving is
        a standard predictor-corrector continuation, and a failure part-way is itself the answer: it
        locates the fraction of closure that is reachable.
        """
        base, _ = self.evaluate(theta, v_target, use_offset=False)
        if base is None:
            raise ValueError("cannot set a homotopy from a state whose rollout terminates")
        self.offset = (1.0 - float(alpha)) * base

    def evaluate(self, theta, v_target, *, use_offset: bool = True):
        nodes, ctrl = self.unpack(theta)
        self.rmap.ctrl_seq = ctrl
        residual = np.empty(self.n_res)
        advance, contacts = 0.0, []
        for s in range(self.segments):
            x_out, act_out, adv, con, term = self._segment(nodes[s], s)
            if term:
                return None, {"terminated": True, "segment": s}
            got = np.concatenate([x_out, act_out])
            residual[s * self.n_node:(s + 1) * self.n_node] = (
                got - nodes[(s + 1) % self.segments]) / self.node_scale
            advance += adv
            contacts.extend(con)
        velocity = advance / (self.rmap.cycle * self.rmap.control_dt)
        if self.velocity_weight:
            residual[-1] = self.velocity_weight * (velocity - v_target)
        n_cont = self.segments * self.n_node
        # "periodicity" is always the TRUE closing error, never the homotopy-relaxed one, so a
        # continuation log cannot flatter itself.
        info = {
            "terminated": False, "advance": advance, "velocity": velocity,
            "flight_steps": int(sum(1 for n in contacts if n == 0)),
            "stance_steps": int(sum(1 for n in contacts if n > 0)),
            "periodicity": float(np.sqrt(np.mean(residual[:n_cont] ** 2))),
            "velocity_error": float(velocity - v_target),
        }
        if use_offset:
            residual = residual - self.offset
            info["relaxed_periodicity"] = float(np.sqrt(np.mean(residual[:n_cont] ** 2)))
        return residual, info

    def cost(self, residual) -> float:
        return float(np.sqrt(np.mean(np.asarray(residual) ** 2)))

    def jacobian(self, theta, v_target, *, state_eps_rel=STATE_EPS_REL, ctrl_eps=CTRL_EPS):
        """Local finite differences: each unknown touches exactly one segment.

        The continuity block ``-I / node_scale`` against the *next* node is analytic and free, which
        is the other half of why this costs less than single shooting.
        """
        nodes, ctrl = self.unpack(theta)
        self.rmap.ctrl_seq = ctrl
        A = np.zeros((self.n_res, self.dim))
        if any(self._segment(nodes[s], s)[4] for s in range(self.segments)):
            return np.full((self.n_res, self.dim), np.nan)
        dt_cycle = self.rmap.cycle * self.rmap.control_dt

        def rows(s):
            return slice(s * self.n_node, (s + 1) * self.n_node)

        node_steps = np.concatenate([state_eps_rel * self.scales,
                                     np.full(self.na, state_eps_rel)])
        for s in range(self.segments):
            prev = (s - 1) % self.segments
            A[rows(prev), s * self.n_node:(s + 1) * self.n_node] += np.diag(-1.0 / self.node_scale)
            for j in range(self.n_node):
                h = node_steps[j]
                out = []
                for sign in (+1.0, -1.0):
                    node = nodes[s].copy()
                    node[j] += sign * h
                    x_out, act_out, adv, _, term = self._segment(node, s)
                    if term:
                        out = None
                        break
                    out.append((np.concatenate([x_out, act_out]), adv))
                col = s * self.n_node + j
                if out is None:
                    A[:, col] = np.nan
                    continue
                A[rows(s), col] += (out[0][0] - out[1][0]) / self.node_scale / (2.0 * h)
                if self.velocity_weight:
                    A[-1, col] += self.velocity_weight * (out[0][1] - out[1][1]) / (2.0 * h) / dt_cycle

        cut = self.segments * self.n_node
        nu = self.rmap.mjm.nu
        for t in range(self.rmap.cycle):
            s = t // self.length
            for i in range(nu):
                col = cut + t * nu + i
                out = []
                for sign in (+1.0, -1.0):
                    probe = ctrl.copy()
                    probe[t, i] += sign * ctrl_eps
                    self.rmap.ctrl_seq = probe
                    x_out, act_out, adv, _, term = self._segment(nodes[s], s)
                    if term:
                        out = None
                        break
                    out.append((np.concatenate([x_out, act_out]), adv))
                self.rmap.ctrl_seq = ctrl
                if out is None:
                    A[:, col] = np.nan
                    continue
                A[rows(s), col] += (out[0][0] - out[1][0]) / self.node_scale / (2.0 * ctrl_eps)
                if self.velocity_weight:
                    A[-1, col] += (self.velocity_weight * (out[0][1] - out[1][1])
                                   / (2.0 * ctrl_eps) / dt_cycle)
        return A


@dataclasses.dataclass
class SolveResult:
    theta: np.ndarray
    cost: float
    periodicity: float
    velocity: float
    velocity_target: float
    converged: bool
    iterations: int
    info: dict
    reason: str = ""


def solve(problem: GaitProblem, theta0, v_target, *, tol=1e-10, max_iter=40,
          lam0=1e-6, lam_down=0.3, lam_up=8.0, max_backtracks=14,
          stall_rel=1e-4, stall_window=4, verbose=False) -> SolveResult:
    """Minimum-norm damped Gauss-Newton on the underdetermined system ``F(theta) = 0``.

    With 12 residuals and up to 203 unknowns, ``A A^T`` is 12x12 while ``A^T A`` is 203x203 and rank
    deficient, so the step is taken in the dual form

        ``delta = -A^T (A A^T + lambda I)^-1 F``

    which is both the cheaper solve and the minimum-norm one. Minimum norm is what continuation
    wants: it keeps each solution near its predecessor rather than wandering across the manifold.
    """
    theta = problem.project(np.asarray(theta0, dtype=np.float64))
    residual, info = problem.evaluate(theta, v_target)
    if residual is None:
        return SolveResult(theta, np.inf, np.inf, info.get("velocity", np.nan), v_target,
                           False, 0, info, "start_infeasible")

    cost = problem.cost(residual)
    history = [cost]
    lam = lam0
    reason = "max_iterations"

    for iteration in range(1, max_iter + 1):
        if cost <= tol:
            reason = "converged"
            break
        A = problem.jacobian(theta, v_target)
        if not np.isfinite(A).all():
            reason = "jacobian_infeasible"
            break
        # Work in the scaled metric: A_s = A diag(theta_scale), step = theta_scale * step_s.
        A_s = A * problem.theta_scale[None, :]
        gram = A_s @ A_s.T
        improved = False
        for _ in range(max_backtracks):
            try:
                step = -problem.theta_scale * (
                    A_s.T @ np.linalg.solve(gram + lam * np.eye(gram.shape[0]), residual)
                )
            except np.linalg.LinAlgError:
                lam *= lam_up
                continue
            trial = problem.project(theta + step)
            r_trial, i_trial = problem.evaluate(trial, v_target)
            if r_trial is not None and problem.cost(r_trial) < cost:
                theta, residual, info = trial, r_trial, i_trial
                cost = problem.cost(residual)
                lam = max(lam * lam_down, 1e-14)
                improved = True
                break
            lam *= lam_up
        history.append(cost)
        if verbose:
            print(f"      it {iteration:2d} cost {cost:.3e} per {info['periodicity']:.3e} "
                  f"v {info['velocity']:+.4f} lam {lam:.1e}")
        if not improved:
            reason = "stalled"
            break
        if len(history) > stall_window:
            window = history[-(stall_window + 1):]
            if window[0] > 0 and (window[0] - window[-1]) / window[0] < stall_rel:
                reason = "stalled"
                break
    else:
        reason = "converged" if cost <= tol else "max_iterations"

    return SolveResult(theta, cost, info["periodicity"], info["velocity"], v_target,
                       cost <= tol, len(history) - 1, info, reason)


def continuation(problem: GaitProblem, theta0, targets: Sequence[float], *,
                 on_step: Optional[Callable] = None, **solve_kwargs):
    """Trace a branch of periodic orbits by stepping the velocity target.

    Each solve is warm-started from the previous solution, which is what makes the branch a branch
    rather than a sequence of unrelated solves. Stops at the first target that fails, because past a
    fold the continuation is no longer following the same family.
    """
    results = []
    theta = np.asarray(theta0, dtype=np.float64)
    for target in targets:
        result = solve(problem, theta, target, **solve_kwargs)
        results.append(result)
        if on_step is not None:
            on_step(result)
        if not result.converged:
            break
        theta = result.theta
    return results


def velocity_reach(problem: GaitProblem, theta, *, jacobian=None):
    """How much velocity is reachable **without leaving the periodic manifold**.

    Projects the velocity gradient onto the null space of the periodicity Jacobian, in a physical
    metric (state in units of its own scale, control in units of its own range). ``dv_per_unit`` is
    the achievable velocity change per unit step along the manifold, and it is the operationally
    meaningful number: ``1 / dv_per_unit`` is how far you must move to gain 1 m/s.

    Measured across four orbits spanning both actuators, this tracks **flight fraction**, not
    actuator type::

        motor trained gait T27   flight 22/27   dv_per_unit 7.80e-01
        motor backwards hop T16  flight  6/16   dv_per_unit 6.11e-01
        muscle bob T32           flight  1/32   dv_per_unit 2.52e-02
        muscle standing T16      flight  0/16   dv_per_unit 7.97e-10

    Which is mechanically plain: in flight the body is ballistic and horizontal velocity is a free
    constant of the motion, while in stance the foot is anchored by contact and changing velocity
    means working against the constraint. An orbit with no flight phase has no lever, whatever
    drives it. That is why continuing in velocity from a bob cannot work, and it is a statement
    about gait type rather than about muscle.
    """
    A = problem.jacobian(theta, 0.0) if jacobian is None else jacobian
    metric = np.concatenate([problem.scales, np.ones(problem.n_ctrl)])
    A_s = A * metric[None, :]
    periodicity, velocity = A_s[: problem.n_state], A_s[problem.n_state]
    _, _, vt = np.linalg.svd(periodicity, full_matrices=True)
    projected = vt[problem.n_state:] @ velocity
    free = float(np.linalg.norm(velocity))
    reachable = float(np.linalg.norm(projected))
    return {
        "dv_per_unit": reachable,
        "dv_per_unit_unconstrained": free,
        "retained_fraction": reachable / free if free > 0 else 0.0,
        "units_to_gain_1mps": float("inf") if reachable == 0 else 1.0 / reachable,
    }


def characterise(problem: GaitProblem, theta, scales=None):
    """Floquet spectrum, flight phase and eps certificate of a converged orbit."""
    scales = problem.scales if scales is None else np.asarray(scales, dtype=np.float64)
    x, ctrl = problem.unpack(theta)
    problem._apply(ctrl)
    jac = st.jacobian_fd(problem.rmap, x, scales=scales, eps_rel=STATE_EPS_REL)
    spec = st.spectrum(jac.J)
    sweep = st.eps_sweep(problem.rmap, x, scales=scales)
    out = problem.rmap.roll(x)
    return {
        "spectral_radius": spec.spectral_radius,
        "unstable_count": spec.unstable_count,
        "period_doubling_margin": spec.period_doubling_margin,
        "eps_verdict": sweep.verdict,
        "eps_window_decades": sweep.window_decades,
        "flight_steps": int(sum(1 for n in out.contacts if n == 0)),
        "stance_steps": int(sum(1 for n in out.contacts if n > 0)),
        "height_range_m": out.height_range,
        "height_min_m": out.height_min,
        "advance_per_cycle_m": out.advance,
        "velocity_mps": out.advance / (problem.rmap.cycle * problem.rmap.control_dt),
        "saturated_ctrl_cells": problem.saturated(theta),
        "act_excursions": out.act_excursions,
    }
