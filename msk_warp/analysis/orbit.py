"""The Poincare return map of a periodic control, and the activation trajectory it implies.

## Why this exists

Phase 4 asked "does a periodic muscle gait exist?" and answered it with a *heuristic*: a scaled RMS
periodicity residual over the mechanical state, gated at twice a real gait's residual. That
instrument cannot answer the question, for a reason that is measured rather than argued
(``docs/VALIDITY.md`` CL-01): the residual excludes muscle activation, and every way of putting
activation back changes the verdict. The same ``T_c`` 16 candidate scores 0.4216 with unit
activation scales and 12.1 with the standing trace's scales, against a 0.4684 bar -- pass or fail by
a factor of 26, on a choice for which no principled value exists.

The exact question has an exact instrument. A control sequence of length ``T_c``, replayed forever,
admits a periodic orbit **iff** its ``T_c``-step return map ``P`` has a fixed point:

    x* = P(x*)

So: build ``P``, solve ``P(x) - x = 0``, and read the orbit's stability off the eigenvalues of
``dP/dx`` at the root. No ruler, no bar, no transferability argument. This module is ``P``;
``stability.py`` is everything built on top of it.

## The activation subsystem is not part of the unknown

MuJoCo's ``dyntype=muscle`` computes ``act_dot`` from ``(ctrl, act)`` alone -- there is no dependence
on tendon length or velocity. Under a ``T_c``-periodic control the activation subsystem is therefore
**autonomous** and, being a per-muscle contraction, has a unique globally attracting periodic
solution ``act*``. Measured (``docs/VALIDITY.md`` CL-02): from a different mechanical state *and* a
different initial activation, the end-of-cycle activation converges to the same ``act*`` with
**exactly 0.0** difference, reaching machine precision in three cycles.

That has a consequence worth stating plainly, because Phase 4 read it the other way round. The
candidate's much-discussed per-muscle "activation closing errors"
``[0.166, 0.427, 0.858, 0.886, 0.369, 0.212]`` are exactly ``|act_0 - act*|``. They are not a
property of the trajectory; they are a **mis-specified initial condition**, handed to the optimiser
as six free parameters (``scripts/trajopt_hopper.py:290``) that the periodicity requirement already
determines. So the right treatment is to *pin* activation to ``act*`` and shoot the mechanical state
only -- a smaller problem, not a larger one.

``ReturnMap`` supports both: ``include_activation=False`` (default) pins ``act`` and gives an
11-component map for the hopper, and ``include_activation=True`` carries ``act`` as a coordinate for
models or questions where that reasoning does not apply. A test asserts the two agree.

## Translation is quotiented out, and that is checked rather than assumed

A gait is periodic *modulo translation*: after one cycle the hopper is in the same configuration
moving the same way, just further down the track. So ``qpos[0]`` is dropped from the state -- the
same convention as ``msk_warp.utils.gait.shape_vector`` -- and returned separately as ``advance``.
That is only legitimate if the dynamics are invariant in ``qpos[0]``, which
``stability.py`` asserts from the cycle Jacobian rather than taking on faith.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import mujoco
import numpy as np

#: Physics substeps per control step. Matches ``substeps: 4`` in every ``hopper_*.yaml``.
CONTROL_SUBSTEPS = 4

#: Torso height below which ``HopperBaseEnv`` terminates the episode (``hopper.py:105``).
TERMINATION_HEIGHT = -0.45

#: ``qpos[0]`` is the translating coordinate and is never part of a state vector.
TRANSLATING_INDEX = 0

#: ``qpos[1]`` is torso height on both hopper assets.
HEIGHT_INDEX = 1


class OrbitTerminated(RuntimeError):
    """A rollout fell below the termination height, so the return map is undefined there."""


def shape_state(qpos, qvel, act=None) -> np.ndarray:
    """State with the translating coordinate removed: ``(qpos[1:], qvel, act)``.

    The numpy sibling of :func:`msk_warp.utils.gait.shape_vector`, component for component, so a
    state produced here can be scored by the residual every Phase 4 number was measured on.
    ``act`` is appended only when it is non-empty, which keeps the motor model (``na = 0``) on the
    same code path. ``tests/unit/test_orbit.py`` asserts the two implementations agree on random
    input -- two rulers drifting apart is how a project loses the ability to compare two phases.
    """
    parts = [np.asarray(qpos, dtype=np.float64)[..., TRANSLATING_INDEX + 1:],
             np.asarray(qvel, dtype=np.float64)]
    if act is not None:
        act = np.asarray(act, dtype=np.float64)
        if act.shape[-1] > 0:
            parts.append(act)
    return np.concatenate(parts, axis=-1)


def state_dim(mjm, *, include_activation: bool = False) -> int:
    """Width of the state vector: ``(nq - 1) + nv``, plus ``na`` when activation is carried."""
    return (mjm.nq - 1) + mjm.nv + (mjm.na if include_activation else 0)


def assert_state_layout(mjm) -> None:
    """Fail loudly if the ``qpos``-index convention does not hold for this model.

    :func:`shape_state` indexes ``qpos`` directly, and :mod:`msk_warp.analysis.stability` maps that
    onto ``mjd_transitionFD``'s ``(qpos-tangent, qvel, act)`` ordering by deleting index 0. Both are
    valid only when ``nq == nv`` with no free or ball joint, so that the tangent equals ``qpos``
    element for element. True for both hoppers; **false for MyoLeg26**, which is exactly why this
    raises rather than mis-indexing silently.
    """
    if mjm.nq != mjm.nv:
        raise ValueError(
            f"nq ({mjm.nq}) != nv ({mjm.nv}): qpos is not its own tangent space, so the "
            "shape-vector / transitionFD index map in msk_warp.analysis does not apply"
        )
    bad = [int(t) for t in mjm.jnt_type
           if t in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL))]
    if bad:
        raise ValueError(
            "model has free or ball joints, whose qpos is not its own tangent space; "
            "msk_warp.analysis assumes slide/hinge joints only"
        )


def activation_limit_cycle(
    mjm,
    ctrl_seq,
    *,
    substeps: int = CONTROL_SUBSTEPS,
    start: Optional[np.ndarray] = None,
    tol: float = 1e-14,
    max_cycles: int = 32,
):
    """The unique ``T_c``-periodic activation trajectory implied by a periodic control.

    Integrates the activation ODE **alone**. ``act_dot = mju_muscleDynamics(ctrl, act, dynprm[:3])``
    depends only on ``(ctrl, act)``, and MuJoCo's Euler integrator advances it as exactly
    ``act += dt * act_dot`` (verified to 0.0 difference against ``mj_step``). So this needs no
    mechanical simulation, which also makes it immune to a diverging or falling rollout -- the
    reason it is done here rather than by reading ``act`` off cycle boundaries of a full rollout.
    A test cross-checks it against that full-rollout route.

    Returns ``(act_star, cycles_used, final_delta)``. ``final_delta`` is the infinity norm between
    the last two cycle boundaries, so a caller can tell convergence from exhaustion.

    Raises ``ValueError`` if the model has no activation state or a non-muscle ``dyntype``, because
    the autonomy argument is specific to ``mjDYN_MUSCLE`` -- ``mjDYN_FILTER`` with a length-dependent
    input would couple back to the mechanics and this function would be wrong.
    """
    ctrl_seq = np.asarray(ctrl_seq, dtype=np.float64)
    if mjm.na == 0:
        raise ValueError("model has no activation state (na == 0); nothing to solve for")
    dyn = set(int(t) for t in mjm.actuator_dyntype)
    if dyn != {int(mujoco.mjtDyn.mjDYN_MUSCLE)}:
        raise ValueError(
            f"activation_limit_cycle assumes every actuator is mjDYN_MUSCLE, got dyntypes {dyn}. "
            "The autonomy argument (act_dot depends only on ctrl and act) does not hold otherwise."
        )

    dt = float(mjm.opt.timestep)
    prm = np.asarray(mjm.actuator_dynprm[:, :3], dtype=np.float64)
    act = np.full(mjm.na, 0.5) if start is None else np.asarray(start, dtype=np.float64).copy()

    delta = np.inf
    cycles = 0
    for cycles in range(1, max_cycles + 1):
        previous = act.copy()
        for u in ctrl_seq:
            for _ in range(substeps):
                act = act + dt * np.array(
                    [mujoco.mju_muscleDynamics(float(u[i]), float(act[i]), prm[i])
                     for i in range(mjm.na)]
                )
        delta = float(np.abs(act - previous).max())
        if delta <= tol:
            break
    return act, cycles, delta


@dataclasses.dataclass(frozen=True)
class CycleResult:
    """Everything one rollout of the return map observed, whether or not it survived."""

    state: np.ndarray
    """State after the rollout, in :func:`shape_state` order. Full float64, never rounded."""

    activation: np.ndarray
    """Activation after the rollout, always reported even when it is not a state coordinate."""

    advance: float
    """``qpos[0]`` travelled, the coordinate the state quotients out."""

    height_min: float
    height_max: float

    contacts: tuple
    """``ncon`` per control step. The contact *sequence*, which a finite difference must not change."""

    contact_pairs: tuple
    """Sorted ``(geom1, geom2)`` pairs per control step. Finer than ``ncon``: a swap of which geoms
    touch at constant count is still a different branch of the dynamics."""

    terminated: bool
    terminated_at: Optional[int]

    act_excursions: int
    """Substeps in which any activation left [0, 1]. This model does not clamp ``act``
    (``actuator_actlimited`` is False, ``docs/VALIDITY.md`` MF-08), and an excursion paired with a
    saturated ``ctrl`` gives ``act_dot == 0`` and a spurious unit Floquet multiplier."""

    @property
    def height_range(self) -> float:
        """Peak-to-peak torso height **over the window actually rolled**.

        Read this only over a single cycle. Over several cycles of a diverging orbit it measures
        divergence, not hop amplitude -- the defect that had 0.508 m reported for a candidate whose
        within-cycle range is 0.0714 m (``docs/VALIDITY.md`` IN-01).
        """
        return self.height_max - self.height_min


class ReturnMap:
    """``P: x -> x`` after one or more replays of a fixed ``T_c``-periodic control.

    ``x`` is :func:`shape_state`: ``(qpos[1:], qvel)``, plus ``act`` when ``include_activation``.
    ``qpos[0]`` is forced to zero on entry and returned as ``advance``.

    One ``MjData`` is allocated per instance and reset per call, which is what gets the measured
    0.40 ms per 16-step cycle (~2850 cycles/s) rather than allocating in the inner loop.
    """

    def __init__(
        self,
        mjm,
        ctrl_seq,
        *,
        substeps: int = CONTROL_SUBSTEPS,
        activation: Optional[np.ndarray] = None,
        include_activation: bool = False,
        termination_height: float = TERMINATION_HEIGHT,
    ):
        assert_state_layout(mjm)
        self.mjm = mjm
        self.ctrl_seq = np.asarray(ctrl_seq, dtype=np.float64)
        if self.ctrl_seq.ndim != 2 or self.ctrl_seq.shape[1] != mjm.nu:
            raise ValueError(
                f"ctrl_seq must be (T_c, nu={mjm.nu}), got {self.ctrl_seq.shape}"
            )
        self.substeps = int(substeps)
        self.include_activation = bool(include_activation)
        self.termination_height = float(termination_height)
        self._data = mujoco.MjData(mjm)

        if self.include_activation or mjm.na == 0:
            self.activation = np.zeros(mjm.na)
            self.activation_cycles = 0
        elif activation is not None:
            self.activation = np.asarray(activation, dtype=np.float64).copy()
            self.activation_cycles = 0
        else:
            self.activation, self.activation_cycles, _ = activation_limit_cycle(
                mjm, self.ctrl_seq, substeps=self.substeps
            )

    @property
    def cycle(self) -> int:
        return int(self.ctrl_seq.shape[0])

    @property
    def dim(self) -> int:
        return state_dim(self.mjm, include_activation=self.include_activation)

    @property
    def control_dt(self) -> float:
        return self.substeps * float(self.mjm.opt.timestep)

    def _unpack(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(f"state must be ({self.dim},), got {x.shape}")
        nq, nv = self.mjm.nq, self.mjm.nv
        qpos = np.concatenate([[0.0], x[: nq - 1]])
        qvel = x[nq - 1: nq - 1 + nv]
        act = x[nq - 1 + nv:] if self.include_activation else self.activation
        return qpos, qvel, np.asarray(act, dtype=np.float64)

    def roll(self, x, cycles: int = 1, *, record_contacts: bool = True) -> CycleResult:
        """Simulate ``cycles * T_c`` control steps. Never raises; report the outcome instead."""
        qpos, qvel, act = self._unpack(x)
        d = self._data
        mujoco.mj_resetData(self.mjm, d)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        if self.mjm.na:
            d.act[:] = act
        mujoco.mj_forward(self.mjm, d)

        h = float(d.qpos[HEIGHT_INDEX])
        hmin = hmax = h
        contacts, pairs = [], []
        terminated, terminated_at, excursions = False, None, 0
        steps = cycles * self.cycle

        for t in range(steps):
            d.ctrl[:] = self.ctrl_seq[t % self.cycle]
            for _ in range(self.substeps):
                mujoco.mj_step(self.mjm, d)
                if self.mjm.na:
                    a = d.act
                    if np.any(a < 0.0) or np.any(a > 1.0):
                        excursions += 1
            if record_contacts:
                contacts.append(int(d.ncon))
                pairs.append(tuple(sorted(tuple(p) for p in d.contact.geom[: d.ncon].tolist())))
            h = float(d.qpos[HEIGHT_INDEX])
            hmin, hmax = min(hmin, h), max(hmax, h)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                terminated, terminated_at = True, t
                break
            if h < self.termination_height:
                terminated, terminated_at = True, t
                break

        return CycleResult(
            state=shape_state(d.qpos, d.qvel, d.act if self.include_activation else None),
            activation=np.array(d.act, dtype=np.float64, copy=True),
            advance=float(d.qpos[TRANSLATING_INDEX]),
            height_min=hmin,
            height_max=hmax,
            contacts=tuple(contacts),
            contact_pairs=tuple(pairs),
            terminated=terminated,
            terminated_at=terminated_at,
            act_excursions=excursions,
        )

    def __call__(self, x, cycles: int = 1) -> np.ndarray:
        """``P(x)``. Raises :class:`OrbitTerminated` if the rollout did not survive the window."""
        result = self.roll(x, cycles, record_contacts=False)
        if result.terminated:
            raise OrbitTerminated(
                f"rollout terminated at control step {result.terminated_at} of "
                f"{cycles * self.cycle} (height fell below {self.termination_height})"
            )
        return result.state

    def residual(self, x, cycles: int = 1) -> np.ndarray:
        """``P(x) - x``, the function whose root is a periodic orbit."""
        return self(x, cycles) - np.asarray(x, dtype=np.float64)

    def scaled_residual(self, x, scales, cycles: int = 1) -> float:
        """``|| (P(x) - x) / scales ||_2 / sqrt(D)``.

        The identical formula to :func:`msk_warp.utils.gait.periodicity_residual`, so a convergence
        number here is directly comparable to every Phase 4 figure -- the candidate's 0.3174, the
        0.4684 gate, a real gait's 0.2342.
        """
        r = self.residual(x, cycles) / np.asarray(scales, dtype=np.float64)
        return float(np.sqrt(np.mean(r ** 2)))

    def state_at_phase(self, qpos, qvel, act=None) -> np.ndarray:
        """Build a state vector from raw MuJoCo arrays, honouring ``include_activation``."""
        return shape_state(qpos, qvel, act if self.include_activation else None)
