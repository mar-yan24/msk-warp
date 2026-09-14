"""Backend seam: everything msk-warp assumes about mujoco_warp's autodiff, in one place.

Backend: google-deepmind/mujoco_warp PR #1423 (johnnynunez, ``feature/differentiability``),
pinned at ``C:/Projects/mujoco_warp_pr1423`` on branch ``mark/pr1423-fixes`` stacked on
``eeac6f2`` (see ``docs/decisions/ADR-0001-backend.md``, kept outside git).

Gradient contract, asserted at environment construction by :func:`assert_grad_contract`:

* Newton solver (constraint forces are differentiated implicitly through the retained Hessian)
* Euler, RK4 or implicitfast integrator
* every colliding geom-type pair appears in the pinned backend's proxy registry
  (including plane-box, sphere-box and capsule-box); mesh/hfield and unsupported
  pairs are rejected here. Registry support does not establish derivative accuracy.
* ``qpos``, ``qvel``, ``ctrl``, ``act`` (and the contact/efc arrays) carry ``.grad``
* a one-step AD-vs-FD tripwire on the actuation path (``ctrl`` for motors, ``act`` for muscles)

Upstream behaviour the bridge relies on: under gradient tracking ``mjw.step`` rebinds
``d.qpos``, ``d.qvel``, ``d.act`` and its intermediates to fresh arrays every step (per-step
isolation), so array handles must be re-read from ``Data`` after each step and gradient
handles captured *before* a tape is recorded.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

BACKEND_NAME = "pr1423"

STATE_FIELDS = ("qpos", "qvel", "act")
INPUT_FIELDS = ("qpos", "qvel", "act", "ctrl")
CONTACT_GRAD_FIELDS = ("contact.dist", "contact.pos", "contact.frame", "efc.J", "efc.pos", "efc.D", "efc.aref", "efc.vel")
SUPPORTED_INTEGRATORS = {
    mujoco.mjtIntegrator.mjINT_EULER,
    mujoco.mjtIntegrator.mjINT_RK4,
    mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}

_GEOM_NAMES = {int(v): k.replace("mjGEOM_", "").lower() for k, v in mujoco.mjtGeom.__members__.items()}

# The constraint solver and the collision pipeline are compiled with ``enable_backward=False`` and
# bridged by analytic adjoints injected with ``tape.record_func`` (implicit differentiation through
# the Newton solve, smooth proxies for contact geometry). Warp cannot see those custom adjoints, so
# it warns once per recorded solver kernel on every backward pass ("_linesearch...", "_JTDAJ...",
# "_solve_...", "_update_constraint...", "_nograd_copy", the time update, ...). The bridge silences
# this class around ``tape.backward``; the AD-vs-FD tests are what actually validate the gradients.
NOGRAD_KERNEL_WARNING = r"Running the tape backwards may produce incorrect gradients"


@contextlib.contextmanager
def quiet_nograd_kernels():
    """Suppress the by-design ``enable_backward=False`` tape warnings around a ``tape.backward``."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=NOGRAD_KERNEL_WARNING)
        yield


class GradContractError(RuntimeError):
    """The model or Data violates the gradient contract; gradients would be silently wrong."""


_prepared = False


def prepare() -> None:
    """One-time process setup: Warp init and backward-kernel compilation for the AD modules.

    Must run before the first kernel launch; ``enable_grad``/``make_diff_data`` call it too.
    """
    global _prepared
    if not _prepared:
        wp.init()
        mjw.enable_ad()
        _prepared = True


def put_model(mjm):
    prepare()
    return mjw.put_model(mjm)


def make_data(mjm, m, nworld: int, njmax=None, grad: bool = True):
    """Data with gradient tracking on the smooth, solver and contact fields (backend defaults)."""
    prepare()
    kw = {} if njmax is None else {"njmax": int(njmax)}
    if not grad:
        return mjw.make_data(mjm, nworld=nworld, **kw)
    return mjw.make_diff_data(mjm, nworld=nworld, **kw)


def _resolve(d, name: str):
    obj = d
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def zero_grad(d) -> None:
    """Zero every gradient buffer on Data (the per-epoch graph cut; no reallocation)."""
    for name in INPUT_FIELDS + CONTACT_GRAD_FIELDS:
        arr = _resolve(d, name)
        if isinstance(arr, wp.array) and arr.requires_grad and arr.grad is not None:
            arr.grad.zero_()


def unsupported_pairs(m) -> list[str]:
    from mujoco_warp._src.collision_smooth import unsupported_geom_pairs

    return sorted(f"{_GEOM_NAMES[int(a)]}-{_GEOM_NAMES[int(b)]}" for a, b in unsupported_geom_pairs(m))


@dataclass
class ContractReport:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tripwire: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_model(mjm, m) -> ContractReport:
    """Static checks on the compiled model (no kernels launched)."""
    rep = ContractReport()
    if mjm.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
        rep.problems.append(
            f"solver={mujoco.mjtSolver(mjm.opt.solver).name}: constraint gradients need solver=\"Newton\""
        )
    if mjm.opt.integrator not in SUPPORTED_INTEGRATORS:
        rep.problems.append(f"integrator={mujoco.mjtIntegrator(mjm.opt.integrator).name} is not differentiable upstream")
    pairs = unsupported_pairs(m)
    if pairs:
        rep.problems.append("colliding geom pairs without a differentiable contact proxy: " + ", ".join(pairs))
    # A sparse Jacobian only matters where constraint rows exist; models with no contacts, no
    # equalities and no joint limits never build one.
    collidable = bool(np.any((mjm.geom_contype | mjm.geom_conaffinity) != 0)) and not (mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
    has_rows = (collidable or mjm.neq > 0 or bool(np.any(mjm.jnt_limited))) and not (mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    if bool(getattr(m, "is_sparse", False)) and has_rows:
        rep.warnings.append("sparse Jacobian with active constraint rows: the contact/equality gradient path was validated dense-only in the bake-off")
    if mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP == 0 and mjm.nv > 0 and np.any(mjm.dof_damping > 0):
        rep.warnings.append("implicit Euler damping active (eulerdamp): validated on the bake-off models, keep an eye on it")
    return rep


def check_data(mjm, d) -> list[str]:
    problems = []
    for name in ("qpos", "qvel", "ctrl") + (("act",) if mjm.na > 0 else ()):
        arr = getattr(d, name)
        if arr.size > 0 and not arr.requires_grad:
            problems.append(f"Data.{name} has no gradient tracking")
    if d.njmax > 0:
        for name in CONTACT_GRAD_FIELDS:
            arr = _resolve(d, name)
            if isinstance(arr, wp.array) and arr.size > 0 and not arr.requires_grad:
                problems.append(f"Data.{name} has no gradient tracking (contact chain would be inert)")
    return problems


@wp.kernel
def _sum_qvel_kernel(qvel: wp.array2d(dtype=float), loss: wp.array(dtype=float)):
    w, j = wp.tid()
    wp.atomic_add(loss, 0, qvel[w, j])


def gradient_tripwire(mjm, m, njmax=None, eps: float = 1e-3, seed: int = 0) -> dict:
    """One-step ``d(sum qvel_next)/d(input)`` by AD vs batched fp32 central differences.

    The input is ``ctrl`` for motor models and ``act`` for muscle models (``ctrl`` cannot reach the
    next state within one step for muscles: it only enters through ``act_dot``). Uses a scratch
    Data at the model's ``qpos0`` with no contact so it measures the actuation chain only.
    """
    prepare()
    kw = {} if njmax is None else {"njmax": int(njmax)}
    use_act = mjm.na > 0
    n_in = int(mjm.na if use_act else mjm.nu)
    if n_in == 0:
        return {"field": None, "cos": float("nan"), "rel": float("nan"), "skipped": "model has no actuators"}
    rng = np.random.default_rng(seed)
    qpos0 = np.asarray(mjm.qpos0, np.float32)
    ctrl0 = rng.uniform(-0.3, 0.3, mjm.nu).astype(np.float32)
    if mjm.nu:
        lo = np.where(mjm.actuator_ctrllimited.astype(bool), mjm.actuator_ctrlrange[:, 0], -1.0)
        hi = np.where(mjm.actuator_ctrllimited.astype(bool), mjm.actuator_ctrlrange[:, 1], 1.0)
        ctrl0 = np.clip(0.5 * (lo + hi) + 0.3 * (hi - lo) * rng.uniform(-0.5, 0.5, mjm.nu), lo, hi).astype(np.float32)
    act0 = rng.uniform(0.2, 0.6, mjm.na).astype(np.float32)
    field_name = "act" if use_act else "ctrl"
    base = {"qpos": qpos0, "qvel": np.zeros(mjm.nv, np.float32), "act": act0, "ctrl": ctrl0}

    def _set(d, batch):
        for f in INPUT_FIELDS:
            arr = getattr(d, f)
            if arr.size == 0:
                continue
            wp.copy(arr, wp.array(np.ascontiguousarray(np.broadcast_to(batch[f], arr.shape), np.float32), dtype=wp.float32, device=arr.device))
        d.time.zero_()

    # batched finite differences: world 0 base, worlds 2k+1 / 2k+2 = +/- eps on coordinate k
    nworld = 1 + 2 * n_in
    batch = {f: np.repeat(base[f][None], nworld, axis=0) for f in INPUT_FIELDS}
    for k in range(n_in):
        batch[field_name][1 + 2 * k, k] += eps
        batch[field_name][2 + 2 * k, k] -= eps
    d_fd = mjw.make_data(mjm, nworld=nworld, **kw)
    mjw.reset_data(m, d_fd)
    _set(d_fd, batch)
    mjw.step(m, d_fd)
    wp.synchronize()
    L = d_fd.qvel.numpy().astype(np.float64).sum(axis=1)
    fd = np.array([(L[1 + 2 * k] - L[2 + 2 * k]) / (2 * eps) for k in range(n_in)])

    d_ad = mjw.make_diff_data(mjm, nworld=1, **kw)
    mjw.reset_data(m, d_ad)
    _set(d_ad, base)
    ref = getattr(d_ad, field_name)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        mjw.step(m, d_ad)
        wp.launch(_sum_qvel_kernel, dim=(1, mjm.nv), inputs=[d_ad.qvel, loss])
    with quiet_nograd_kernels():
        tape.backward(loss=loss)
    wp.synchronize()
    ad = ref.grad.numpy()[0].astype(np.float64)
    tape.zero()
    nfd, nad = float(np.linalg.norm(fd)), float(np.linalg.norm(ad))
    cos = float(ad @ fd / (nad * nfd)) if nad > 0 and nfd > 0 else float("nan")
    rel = float(np.linalg.norm(ad - fd) / nfd) if nfd > 0 else float("nan")
    return {"field": field_name, "cos": cos, "rel": rel, "ad_norm": nad, "fd_norm": nfd, "nan": int(np.isnan(ad).sum())}


def assert_grad_contract(mjm, m, d, *, strict: bool = True, tripwire: bool = True, njmax=None, min_cos: float = 0.9) -> ContractReport:
    """Raise :class:`GradContractError` (strict) or warn on any contract violation."""
    rep = check_model(mjm, m)
    rep.problems += check_data(mjm, d)
    if tripwire and not rep.problems:
        rep.tripwire = gradient_tripwire(mjm, m, njmax=njmax)
        tw = rep.tripwire
        if "skipped" not in tw and (tw["nan"] > 0 or not (tw["cos"] >= min_cos)):
            rep.problems.append(
                f"AD-vs-FD tripwire failed on d(sum qvel)/d({tw['field']}): cos={tw['cos']:.3f} rel={tw['rel']:.2e} nan={tw['nan']}"
            )
    for w in rep.warnings:
        warnings.warn(f"[msk_warp.backend] {w}", stacklevel=2)
    if rep.problems:
        msg = "gradient contract violated:\n  - " + "\n  - ".join(rep.problems)
        if strict:
            raise GradContractError(msg)
        warnings.warn("[msk_warp.backend] " + msg, stacklevel=2)
    return rep
