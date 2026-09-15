"""Bounded PPO continuation segments: immutable capture, validated restore.

Why this exists
---------------
``PPO.save``/``PPO.load`` (``msk_warp/algorithms/ppo.py:428``/``:438``) are an
**inference** checkpoint: they pickle live objects and rebuild fresh optimizers
(``reset_optimizers=True``). Resuming training from one silently drops the Adam
moments, the observation-normalisation count, the episode counters and history,
the rollout buffers, the environment's physics state and every RNG stream. That
pair is deliberately left untouched here; this module is additive and no
existing caller is rerouted through it.

What this helper binds, and what it does not
--------------------------------------------
``read_segment(path)`` with the default ``expect=None`` is **inspection mode**:
schema version, both checksums, payload structure. It does **not** bind epoch,
seed, segment index or parent-segment lineage — those are recorded in every
segment and compared only against what a caller names in ``expect``. Source
hashes (``metadata["sources"]``) and any freeze hashes in ``extra`` are
**recorded and never compared** — recorded *completely*, though: a missing
declared source is refused at capture rather than omitted. ``restore_state``
independently enforces ``algo.env is env``, the compiled model identity,
``nq/nv/na/nu``, integrator, disableflags, the ``eq_active`` invariant and
``RECIPE_IDENTITY_KEYS``. A real resume must
therefore call ``read_segment(path, expect=strict_expect(...))``; see
``strict_expect``. ``include_git_identity=False`` is likewise an inspection
default. Nothing here is "fully bound" on its own.

Scope: ``PPO`` plus ``MyoLeg26WalkEnv`` (and one analytic CPU test adapter).
This is not a generic checkpointing framework. It trains nothing and claims no
numerical equivalence of continued trajectories — see the report for the limits.

**Not re-exported from** ``msk_warp/analysis/__init__.py``: that package is
asserted torch/warp-free by ``tests/unit/test_orbit.py:304``. Import this
submodule directly, as ``msk_warp/analysis/myoleg26_baseline.py`` is imported.

State ownership
---------------
Captured and restored:

* learned parameters (actor, critic) and **both** Adam ``state_dict``s,
  including ``param_groups`` (``train()``'s linear schedule mutates ``lr``);
* ``obs_rms``/``ret_rms`` mean, var and **count** (a Python float, the stateful
  normalisation count; ``ret_rms`` is ``None`` under the v1 config);
* both ``AverageMeter``s: the ``mean`` buffer **and** ``current_size``, which
  ``state_dict()`` omits and which weights the next update
  (``msk_warp/utils/average_meter.py:19-22``);
* ``iter_count``, ``step_count``, ``_current_obs``, ``episode_loss``,
  ``episode_length``, both ``*_his`` lists, ``best_policy_loss`` (may be a
  ``±inf`` sentinel or the 0-d NumPy array ``AverageMeter.get_mean()`` returns);
* every ``buf_*`` rollout buffer — an unknown ``buf_*`` attribute is refused
  rather than silently dropped;
* the environment's torch buffers, including the Task 2 previous-action history;
* the canonical Warp integration inputs ``qpos, qvel, act, ctrl,
  qacc_warmstart, time``, each at **its own backing dtype and shape as read**
  (never through ``obs``/``start_*`` proxies, never hard-coded to 32 or 64 bit),
  re-asserted on restore;
* the derived solver scratch ``qacc`` and ``act_dot``, held in a **separate**
  ``derived`` section (``WARP_DERIVED_FIELDS``; a CPU adapter declares its own
  through ``RESUME_PLAIN_DERIVED_FIELDS``) rather than in ``WARP_STATE_FIELDS``,
  because the environment never writes them: ``_reset_warp_state``
  (``myoleg26_walk.py:553-579``) writes only the six inputs above, and the
  backend owns these two. Preserved at their own backing dtype and shape, and
  asserted **bitwise** on restore;
* all four RNG streams. CUDA RNG is read **only** when CUDA is already
  initialised (``torch.cuda.is_initialized()``, the ``isolated_rng`` precedent),
  so a CPU run never initialises a device just to record an unused stream.

Deliberately **not** restored, because the caller owns them when it builds the
target object: the resolved recipe (``steps_num``, ``num_minibatches``, ``gamma``
…), ``max_epochs``/``lr_schedule``/``name``/``log_dir``, the ``SummaryWriter``
and ``TimeReport``. These are not silently assumed — the segment records the
recipe and ``restore_state`` refuses a target whose ``RECIPE_IDENTITY_KEYS``
differ, so a changed recipe cannot be mistaken for a faithful resume. Logging
and wall-clock accounting are segment-local by design and are the driver's job.

Refused rather than silently approximated (fail closed at capture):

* ``algo.env is not env`` — a mismatched or missing binding. Same-shaped
  environments pass every dtype/shape/model/recipe gate, so identity is checked
  in **both** directions of the round trip, before anything is read or written;
* a declared source file in ``SOURCE_FILES`` that is missing or is not a regular
  file. The raw-bytes source map has no optional entries: a partial map must
  never be published as a complete one;
* solver warm-start **enabled** (the state would then be numerically live under
  ``solver.py:4034``/``:1568``);
* nonzero ``qfrc_applied``/``xfrc_applied`` — dynamic external inputs this env
  never writes and this schema does not model;
* ``nhistory``/``nplugin``/``nuserdata``/``nmocap`` nonzero, or any
  ``mjDYN_USER`` actuator (a ``act_dyn_callback`` dependency);
* an unsupported integrator;
* nonfinite learned or physical state.

Previously omitted as derived — that claim is WITHDRAWN (2026-09-15)
-------------------------------------------------------------------
An earlier version of this schema omitted ``qacc`` and ``act_dot``. The argument
was a single static reading of the pinned backend (``solver.py:1568``/``:1570``
overwrite ``qacc`` at solve init; ``forward.py:1030`` assigns ``act_dot`` for
every actuator each substep, and for ``DynType.MUSCLE`` it is a pure function of
``(ctrl, act, dynprm)``, ``:930-933``), and the empirical case rested on a GPU
test that required two **independent** physics runs to agree bitwise. That test
**failed** in the full GPU suite on ``7bd388da…`` after passing in isolation,
and a preregistered 24-trial diagnostic then showed the instrument itself was
confounded: six *unperturbed* control repeats alone split across two post-step
``qvel`` values, one entry apart by one float32 ULP.

The failed result stands exactly as recorded — it is not relabelled a pass, not
xfailed, and no tolerance was introduced anywhere. What is withdrawn is the
*claim* that the omission was qualified. Both fields are therefore captured and
restored, and the tests assert exact **restore fidelity**, which is a property
of this helper and does not depend on the simulator reproducing a step. Two more
small arrays of state and I/O are the cost. This is **not** a causal explanation
of that failure, and it does **not** make continued trajectories reproducible:
continued-trajectory equivalence remains **UNVALIDATED**.

``qacc_warmstart`` stays with the env-written inputs: it is written every step
from ``d.qacc`` (``forward.py:386-394``) and read only when warm-start is
enabled (``solver.py:4034`` -> ``:1568``), which capture asserts off.

The global claim is withdrawn too
---------------------------------
The version at ``7bd388da…`` cited that one perturbation test as the empirical
enforcement for **every** other omitted derived field. That is withdrawn as
well: the test only ever perturbed ``qacc`` and ``act_dot``, it never touched
any other field, and its instrument is now known to be confounded. **No
untested field inherits validation from it.**

So every other derived Data field — kinematics, ``qfrc_*``, ``sensordata``,
``energy``, ``qLD``/factorisation caches, contact/``efc``/solver scratch and the
solver counters — is omitted with **neither a per-field source argument nor any
empirical control**, and none may be cited as validated. What is retained, with
its sources, is narrower and different in kind:

* the **static write-before-read reading** above, which covers ``qacc`` and
  ``act_dot`` only, and is a code-path argument on one reading of one pinned
  commit — not a measurement, and not a statement about kernel reproducibility;
* the **model-option assertions** enforced at capture (``_assert_model_contract``
  and ``_reject_external_inputs``): warm-start disabled, a supported integrator,
  ``nhistory``/``nplugin``/``nuserdata``/``nmocap`` zero, no ``mjDYN_USER``
  actuator, zero ``qfrc_applied``/``xfrc_applied``, and the ``eq_active``
  invariant. These bound **which** dynamic inputs can exist at all; they are
  refusals, not evidence that an omitted array is irrelevant.

Faithful continuation on the real simulator therefore remains **unvalidated**,
and continued-trajectory numerical equivalence stays **UNVALIDATED**.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping
import uuid

import numpy as np
import torch

import msk_warp

# Bumped from ``…-v1`` when ``qacc``/``act_dot`` became **required** captured
# state. A ``-v1`` payload genuinely lacks required fields, so it is refused
# outright (``read_segment``/``restore_state`` both compare this string) rather
# than migrated or completed with defaults. No segment of the previous version
# exists outside test temporary directories, and no science artifact uses this
# schema, so there is nothing to migrate and nothing historical is rewritten.
SCHEMA_VERSION = "myoleg26-ppo-resume-v2"

ROOT = Path(msk_warp.PACKAGE_ROOT).resolve().parent

ALGO_BUFFERS = (
    "buf_obs", "buf_actions", "buf_log_probs", "buf_rewards", "buf_dones",
    "buf_values", "buf_timeout_values", "buf_advantages", "buf_returns",
)
ENV_TORCH_BUFFERS = (
    "obs_buf", "rew_buf", "reset_buf", "termination_buf", "progress_buf", "actions",
)
# Created lazily by ``step()`` (``myoleg26_walk.py:474``) and by nothing in
# ``MjWarpEnv.__init__`` (``base_env.py:80-86``) or ``reset()``. A never-stepped
# target therefore does not have the attribute at all, and an epoch-0 boundary
# legitimately has no value for it. Absence is recorded as state, not skipped.
ENV_LAZY_BUFFERS = ("obs_buf_before_reset",)
# The persistent integration inputs the env itself writes
# (``myoleg26_walk.py:553-579``); dtypes are read from the arrays, not assumed.
WARP_STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "time")
# Derived solver scratch the env never writes (the backend does), preserved in a
# section of its own so WARP_STATE_FIELDS keeps meaning "env-written inputs".
# Required, not optional: see the withdrawn-omission note in the module
# docstring. Dtypes and shapes are read from the backing arrays, never from an
# ``obs``/``start_*`` proxy and never hard-coded.
WARP_DERIVED_FIELDS = ("qacc", "act_dot")
# A CPU adapter declares its own derived section under this attribute, exactly
# as it declares its integration inputs under RESUME_PLAIN_STATE_FIELDS.
PLAIN_DERIVED_FIELDS_ATTR = "RESUME_PLAIN_DERIVED_FIELDS"
# Persistent solver inputs the env never writes: captured and required equal.
WARP_INVARIANT_FIELDS = ("eq_active",)
# Dynamic external inputs this schema does not model: required zero.
WARP_REJECT_NONZERO_FIELDS = ("qfrc_applied", "xfrc_applied")

# Construction inputs this module does NOT restore, because the caller owns them
# (they come from the config when the target object is built). They are not
# silently assumed: the segment records them and restore refuses a target whose
# values differ, so an accidental recipe change cannot be mistaken for a resume.
RECIPE_IDENTITY_KEYS = (
    "steps_num", "num_envs", "num_obs", "num_actions", "ppo_epochs",
    "num_minibatches", "gamma", "gae_lambda", "clip_range", "entropy_coef",
    "value_coef", "max_grad_norm", "truncate_grad", "normalize_advantages",
    "max_episode_length", "obs_rms", "ret_rms",
)

SOURCE_FILES = (
    "msk_warp/algorithms/ppo.py",
    "msk_warp/envs/myoleg26_walk.py",
    "msk_warp/envs/base_env.py",
    "msk_warp/networks/actor.py",
    "msk_warp/networks/critic.py",
    "msk_warp/utils/average_meter.py",
    "msk_warp/utils/running_mean_std.py",
    "msk_warp/analysis/ppo_resume.py",
)

_PLAIN_TYPES = (bool, int, float, str, bytes, type(None))

_MISSING = object()


class ResumeError(Exception):
    """Base class for continuation-segment failures."""


class ResumeValidationError(ResumeError):
    """Refusal raised before the restore target is mutated."""


class ResumeIntegrityError(ResumeValidationError):
    """A stored segment is missing, truncated, corrupt or self-inconsistent."""


@dataclass(frozen=True)
class ResumeState:
    """In-memory completed-epoch boundary snapshot.

    Held by value so Unit 4 can keep the last valid boundary in RAM through a
    partial rollout/update and publish only at segment end. Nothing here aliases
    the live algorithm, optimizer, normaliser or simulator.
    """

    metadata: Mapping[str, Any]
    algo: Mapping[str, Any]
    env: Mapping[str, Any]
    extra: Mapping[str, Any]


# ----------------------------------------------------------------------
# Copy helpers — ``.cpu()`` alone can alias, so every copy is explicit
# ----------------------------------------------------------------------


def _copy_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", copy=True)


def _copy_plain(value):
    """Deep copy restricted to types ``torch.load(weights_only=True)`` accepts."""
    if torch.is_tensor(value):
        return _copy_tensor(value)
    if isinstance(value, _PLAIN_TYPES):
        return value
    if isinstance(value, dict):
        return {key: _copy_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        copied = [_copy_plain(item) for item in value]
        return tuple(copied) if isinstance(value, tuple) else copied
    if isinstance(value, np.ndarray):
        raise ResumeValidationError(
            "unsupported NumPy array in captured state; encode it as tensors or plain types")
    raise ResumeValidationError(f"unsupported captured value of type {type(value).__name__!r}")


def _require_finite(label: str, tensor: torch.Tensor) -> None:
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ResumeValidationError(f"nonfinite values in {label}; refusing to capture")


def _describe(tensor: torch.Tensor) -> dict:
    return {"dtype": str(tensor.dtype), "shape": tuple(int(dim) for dim in tensor.shape)}


# ----------------------------------------------------------------------
# Metric sentinel (``best_policy_loss``) encoding
# ----------------------------------------------------------------------


_METRIC_KINDS = ("float", "int", "numpy_scalar", "numpy_0d")


def _encode_metric(value) -> dict:
    """``±inf`` is a legitimate unset sentinel; a 0-d array is what ``train()`` stores."""
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ResumeValidationError("best-metric sentinel must be scalar")
        return {"kind": "numpy_0d", "dtype": str(value.dtype), "value": float(value)}
    if isinstance(value, np.generic):
        return {"kind": "numpy_scalar", "dtype": str(value.dtype), "value": float(value)}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResumeValidationError(
            f"unsupported best-metric sentinel of type {type(value).__name__!r}")
    return {"kind": "int" if isinstance(value, int) else "float", "dtype": "", "value": float(value)}


def _decode_metric(record: Mapping[str, Any]):
    kind, value = record["kind"], record["value"]
    if kind == "numpy_0d":
        return np.array(value, dtype=np.dtype(record["dtype"]))
    if kind == "numpy_scalar":
        return np.dtype(record["dtype"]).type(value)
    if kind == "int":
        return int(value)
    return float(value)


# ----------------------------------------------------------------------
# RNG
# ----------------------------------------------------------------------


def _capture_rng() -> dict:
    python_state = random.getstate()
    numpy_state = np.random.get_state(legacy=True)
    initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    return {
        "python": _copy_plain(python_state),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "key": [int(item) for item in numpy_state[1]],
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        # Reading CUDA RNG unguarded lazily initialises CUDA; do not.
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if initialized else None,
        "cuda_initialized": initialized,
        # The inventory is all-device by construction, recorded so a subset can
        # never be mistaken for a legitimate capture.
        "cuda_device_count": int(torch.cuda.device_count()) if initialized else 0,
    }


def _validate_rng(record, problems) -> None:
    """Check capability, format and host compatibility BEFORE anything is written.

    RNG is restored last, so without this every malformed or host-incompatible
    stream would surface only after the target — and the process-global Python
    and NumPy generators — had already been overwritten. Each stream is validated
    against an **isolated** generator object; nothing here touches a global.

    This guarantees fail-closed behaviour for malformed or incompatible *input*.
    It does not, and cannot, cover a hardware or driver fault raised mid-restore.

    The CUDA entry is validated in three steps, all inside the
    already-initialised branch so validation never initialises a device and the
    capture-side "a CPU run must not initialise CUDA" rule is untouched: the
    inventory must be exactly one entry per device (capture is all-device), each
    entry's length must equal the live state length of its device, and the
    content must load into a **private** per-device ``torch.Generator``. That
    private generator is the runtime's own complete format oracle — the format is
    never encoded here — and loading a state into it mutates no default stream.
    """
    if not isinstance(record, Mapping):
        problems.append("rng: segment carries no RNG record")
        return
    for key in ("python", "numpy", "torch_cpu", "cuda", "cuda_initialized"):
        if key not in record:
            problems.append(f"rng: missing stream {key!r}")
    if problems:
        return

    try:
        random.Random().setstate(tuple(record["python"]))
    except Exception as error:
        problems.append(f"rng: python state is malformed ({error})")

    numpy_state = record["numpy"]
    try:
        np.random.RandomState().set_state((
            numpy_state["bit_generator"],
            np.array(numpy_state["key"], dtype=np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ))
    except Exception as error:
        problems.append(f"rng: numpy state is malformed ({error})")

    cpu_state = record["torch_cpu"]
    if not torch.is_tensor(cpu_state) or cpu_state.dtype != torch.uint8 or cpu_state.dim() != 1:
        problems.append("rng: torch_cpu state must be a 1-D uint8 tensor")
    else:
        try:
            torch.Generator().set_state(cpu_state.cpu())
        except Exception as error:
            problems.append(f"rng: torch_cpu state is malformed ({error})")

    cuda_state = record["cuda"]
    if cuda_state is not None:
        if not isinstance(cuda_state, (list, tuple)) or not all(
                torch.is_tensor(item) and item.dtype == torch.uint8 and item.dim() == 1
                for item in cuda_state):
            problems.append("rng: CUDA state must be a list of 1-D uint8 tensors")
        elif not torch.cuda.is_available():
            problems.append(
                "rng: segment carries CUDA RNG state but CUDA is unavailable on this host")
        elif not torch.cuda.is_initialized():
            # There is no isolated CUDA generator to validate against without
            # touching a device, and ``torch.cuda.set_rng_state_all`` goes through
            # ``_lazy_call``: on an uninitialised host it would DEFER, surfacing a
            # malformed state at an arbitrary later point instead of here. So the
            # caller's compatibility requirement is explicit — initialise CUDA
            # (normally by building the target on its device) before restoring a
            # CUDA-carrying segment — and this refuses before any mutation.
            problems.append(
                "rng: segment carries CUDA RNG state but CUDA is not initialised on this host, "
                "so its length cannot be validated and the restore would be deferred; build the "
                "restore target on its CUDA device first")
        elif torch.cuda.device_count() != len(cuda_state):
            # capture_state always records one entry per device
            # (``torch.cuda.get_rng_state_all``), so a subset or an extra entry is
            # not a legitimate segment. Accepting a short list would let
            # ``set_rng_state_all`` restore only part of the host while the
            # restore report still claimed the CUDA stream was restored.
            problems.append(
                f"rng: segment carries CUDA RNG state for {len(cuda_state)} devices but this "
                f"host has {torch.cuda.device_count()} CUDA devices; an all-device capture "
                "cannot be restored from a subset")
        elif record.get("cuda_device_count") not in (None, len(cuda_state)):
            problems.append(
                f"rng: CUDA stream inventory is inconsistent: {len(cuda_state)} entries but the "
                f"segment records {record['cuda_device_count']} devices")
        else:
            # The dtype and the checksums say nothing about length: both checksums
            # only prove the bytes were not altered after publication, so a record
            # malformed BEFORE write_segment loads cleanly. Compare each entry
            # against this runtime's own state length — never a hard-coded size.
            for index, item in enumerate(cuda_state):
                expected = int(torch.cuda.get_rng_state(index).numel())
                if int(item.numel()) != expected:
                    problems.append(
                        f"rng: CUDA device {index} state is {int(item.numel())} bytes but this "
                        f"runtime's generator state is {expected} bytes")
                    continue
                # Correct length is not correct content: a Philox offset that is
                # not a multiple of 4 is rejected by torch at restore time. Ask a
                # **private** per-device generator instead of encoding the format
                # here — it is the runtime's own complete oracle, and loading a
                # state into it mutates no default stream (verified: the review's
                # content probe recorded `private_generator_streams_changed: []`).
                # This branch is already behind ``torch.cuda.is_initialized()``,
                # so no device is initialised to validate.
                try:
                    torch.Generator(device=f"cuda:{index}").set_state(item.cpu().clone())
                except Exception as error:
                    problems.append(f"rng: CUDA device {index} state is malformed ({error})")


def _restore_rng(record: Mapping[str, Any]) -> None:
    random.setstate(tuple(record["python"]))
    numpy_state = record["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        np.array(numpy_state["key"], dtype=np.uint32),
        numpy_state["pos"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(record["torch_cpu"].to(dtype=torch.uint8, device="cpu").clone())
    if record["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ResumeValidationError("segment carries CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([state.clone() for state in record["cuda"]])


# ----------------------------------------------------------------------
# Algorithm capture
# ----------------------------------------------------------------------


def _capture_module(label: str, module) -> dict:
    out = {}
    for key, value in module.state_dict().items():
        _require_finite(f"{label}.{key}", value)
        out[key] = _copy_tensor(value)
    return out


def _capture_optimizer(label: str, optimizer) -> dict:
    """``optimizer.state_dict()`` returns live references; copy every leaf."""
    raw = optimizer.state_dict()
    state = {}
    for index, entry in raw["state"].items():
        copied = {}
        for key, value in entry.items():
            if torch.is_tensor(value):
                _require_finite(f"{label}.state.{index}.{key}", value)
                copied[key] = _copy_tensor(value)
            else:
                copied[key] = _copy_plain(value)
        state[int(index)] = copied
    return {"state": state, "param_groups": _copy_plain(raw["param_groups"])}


def _capture_rms(label: str, rms) -> dict | None:
    if rms is None:
        return None
    _require_finite(f"{label}.mean", rms.mean)
    _require_finite(f"{label}.var", rms.var)
    count = float(rms.count)
    if not np.isfinite(count):
        raise ResumeValidationError(f"nonfinite values in {label}.count; refusing to capture")
    return {"mean": _copy_tensor(rms.mean), "var": _copy_tensor(rms.var), "count": count}


def _capture_meter(label: str, meter) -> dict:
    """``state_dict()`` carries only ``mean``; ``current_size`` weights the next update."""
    _require_finite(f"{label}.mean", meter.mean)
    return {
        "mean": _copy_tensor(meter.mean),
        "current_size": int(meter.current_size),
        "max_size": int(meter.max_size),
    }


def _capture_algo(algo) -> dict:
    unknown = sorted(name for name in vars(algo)
                     if name.startswith("buf_") and name not in ALGO_BUFFERS)
    if unknown:
        raise ResumeValidationError(
            f"unknown rollout buffer attribute(s) {unknown}: extend ALGO_BUFFERS "
            "rather than dropping state")
    buffers = {}
    for name in ALGO_BUFFERS:
        tensor = getattr(algo, name)
        _require_finite(name, tensor)
        buffers[name] = _copy_tensor(tensor)

    current_obs = algo._current_obs
    if current_obs is None:
        raise ResumeValidationError(
            "_current_obs is None: capture at a completed epoch boundary, after reset()")
    _require_finite("_current_obs", current_obs)
    _require_finite("episode_loss", algo.episode_loss)

    return {
        "actor": _capture_module("actor", algo.actor),
        "critic": _capture_module("critic", algo.critic),
        "actor_optimizer": _capture_optimizer("actor_optimizer", algo.actor_optimizer),
        "critic_optimizer": _capture_optimizer("critic_optimizer", algo.critic_optimizer),
        "obs_rms": _capture_rms("obs_rms", algo.obs_rms),
        "ret_rms": _capture_rms("ret_rms", algo.ret_rms),
        "episode_loss_meter": _capture_meter("episode_loss_meter", algo.episode_loss_meter),
        "episode_length_meter": _capture_meter("episode_length_meter", algo.episode_length_meter),
        "buffers": buffers,
        "iter_count": int(algo.iter_count),
        "step_count": int(algo.step_count),
        "_current_obs": _copy_tensor(current_obs),
        "episode_loss": _copy_tensor(algo.episode_loss),
        "episode_length": _copy_tensor(algo.episode_length),
        "episode_loss_his": [float(value) for value in algo.episode_loss_his],
        "episode_length_his": [int(value) for value in algo.episode_length_his],
        "best_policy_loss": _encode_metric(algo.best_policy_loss),
        "rng": _capture_rng(),
    }


# ----------------------------------------------------------------------
# Environment capture
# ----------------------------------------------------------------------


def _env_kind(env) -> str:
    if getattr(env, "warp_data", None) is not None:
        return "warp"
    if getattr(env, "RESUME_PLAIN_STATE_FIELDS", None):
        return "plain"
    raise ResumeValidationError(
        f"{type(env).__name__} exposes no supported state protocol: expected "
        "'warp_data' or 'RESUME_PLAIN_STATE_FIELDS'")


def _assert_model_contract(env) -> dict:
    """Model-option refusals that bound which dynamic inputs can exist.

    These are gates, not evidence: they establish that no external force, plugin,
    history, mocap, userdata or ``mjDYN_USER`` input is live and that warm-start
    is off, which is what the schema's *scope* relies on. They do **not**
    validate any omitted derived array — see the module docstring.
    """
    import mujoco

    mjm = env.mjm
    problems = []
    disableflags = int(mjm.opt.disableflags)
    if not disableflags & int(mujoco.mjtDisableBit.mjDSBL_WARMSTART):
        problems.append(
            "solver warm-start is enabled: qacc_warmstart becomes a live solver input "
            "(solver.py:4034 -> :1568) and this schema's derived-field omissions do not hold")
    integrator = int(mjm.opt.integrator)
    supported = {int(mujoco.mjtIntegrator.mjINT_EULER),
                 int(mujoco.mjtIntegrator.mjINT_RK4),
                 int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)}
    if integrator not in supported:
        problems.append(f"unsupported integrator {integrator}")
    sizes = {name: int(getattr(mjm, name, 0))
             for name in ("nplugin", "nuserdata", "nmocap", "nhistory")}
    for name, value in sizes.items():
        if value:
            problems.append(
                f"{name}={value}: dynamic external state this segment schema does not model")
    dyntypes = np.asarray(mjm.actuator_dyntype, dtype=int)
    if dyntypes.size and int(mujoco.mjtDyn.mjDYN_USER) in set(dyntypes.tolist()):
        problems.append("mjDYN_USER actuator: act_dot depends on act_dyn_callback")
    if problems:
        raise ResumeValidationError("; ".join(problems))

    buffer = np.empty(mujoco.mj_sizeModel(mjm), dtype=np.uint8)
    mujoco.mj_saveModel(mjm, buffer=buffer)
    record = {
        "disableflags": disableflags,
        "integrator": integrator,
        "warmstart_disabled": True,
        "compiled_model_sha256": hashlib.sha256(buffer.tobytes()).hexdigest(),
        "nq": int(mjm.nq), "nv": int(mjm.nv), "na": int(mjm.na), "nu": int(mjm.nu),
    }
    record.update(sizes)
    return record


def _warp_view(data, name):
    import warp as wp

    array = getattr(data, name, None)
    if array is None:
        return None
    return wp.to_torch(array)


def _capture_warp(env) -> dict:
    import warp as wp

    wp.synchronize()
    data = env.warp_data
    fields = {}
    for name in WARP_STATE_FIELDS:
        view = _warp_view(data, name)
        if view is None:
            raise ResumeValidationError(f"Warp Data has no field {name!r}")
        _require_finite(f"warp.{name}", view)
        fields[name] = {"values": _copy_tensor(view), **_describe(view)}
    invariants = {}
    for name in WARP_INVARIANT_FIELDS:
        view = _warp_view(data, name)
        if view is not None:
            invariants[name] = {"values": _copy_tensor(view), **_describe(view)}
    _reject_external_inputs(data)
    return {"fields": fields, "invariants": invariants, "model": _assert_model_contract(env)}


def _reject_external_inputs(data) -> None:
    for name in WARP_REJECT_NONZERO_FIELDS:
        view = _warp_view(data, name)
        if view is not None and view.numel() and bool(view.abs().sum().item()):
            raise ResumeValidationError(
                f"{name} is nonzero: externally applied forces are a dynamic input this "
                "segment schema does not model")


def _derived_names(env, kind) -> tuple:
    """The declared derived section for this environment protocol."""
    if kind == "warp":
        return WARP_DERIVED_FIELDS
    return tuple(getattr(env, PLAIN_DERIVED_FIELDS_ATTR, ()) or ())


def _derived_view(env, kind, name):
    """A live, aliasing view of one derived field, or ``None`` when absent."""
    if kind == "warp":
        return _warp_view(env.warp_data, name)
    return getattr(env, name, None)


def _capture_derived(env, kind) -> dict:
    """Preserve the derived scratch by value, with its own dtype and shape.

    Refuses rather than publishing a partial section: a declared field that the
    environment does not expose would otherwise be silently dropped and then
    silently defaulted on restore.
    """
    out = {}
    for name in _derived_names(env, kind):
        view = _derived_view(env, kind, name)
        if view is None:
            raise ResumeValidationError(
                f"declared derived field {name!r} is missing on the environment; the "
                "preserved-scratch section must never be published incomplete")
        _require_finite(f"derived.{name}", view)
        out[name] = {"values": _copy_tensor(view), **_describe(view)}
    return out


def _capture_env(env) -> dict:
    kind = _env_kind(env)
    buffers = {}
    for name in ENV_TORCH_BUFFERS:
        tensor = getattr(env, name, None)
        if tensor is None:
            raise ResumeValidationError(f"env.{name} is missing: not a supported environment")
        _require_finite(f"env.{name}", tensor)
        buffers[name] = _copy_tensor(tensor)
    absent_lazy = []
    for name in ENV_LAZY_BUFFERS:
        tensor = getattr(env, name, None)
        if tensor is None:
            absent_lazy.append(name)
            continue
        _require_finite(f"env.{name}", tensor)
        buffers[name] = _copy_tensor(tensor)
    record = {
        "kind": kind,
        "buffers": buffers,
        "absent_lazy_buffers": absent_lazy,
        "num_envs": int(env.num_envs),
        "num_obs": int(env.num_obs),
        "num_actions": int(env.num_actions),
        "episode_length": int(env.episode_length),
    }
    if kind == "warp":
        record["warp"] = _capture_warp(env)
    else:
        plain = {}
        for name in env.RESUME_PLAIN_STATE_FIELDS:
            tensor = getattr(env, name)
            _require_finite(f"env.{name}", tensor)
            plain[name] = _copy_tensor(tensor)
        record["plain"] = plain
    # Read after the kind-specific capture: ``_capture_warp`` has already called
    # ``wp.synchronize()`` and nothing has run in between, so this view is
    # coherent with the six inputs recorded above.
    record["derived"] = _capture_derived(env, kind)
    return record


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_hashes(root=None, files=None) -> dict:
    """Raw working-tree bytes (IN-25: these are NOT git blob ids).

    **Every declared entry is required.** There are no optional sources: each
    path in ``SOURCE_FILES`` is production code whose bytes define what a resumed
    run executes, and the record is published under
    ``kind: raw_working_tree_bytes`` and consumed as a complete source map (the
    v2 freeze is the intended reader). Silently omitting a renamed, moved or
    deleted file would therefore present a *partial* map as a complete one, so a
    missing path — or a path that is not a regular file — is a refusal naming it,
    not a dropped key.

    ``files=None`` resolves ``SOURCE_FILES`` at call time, so this record and
    ``_git_identity``'s always describe the same declared set.
    """
    root = Path(root or ROOT)
    files = SOURCE_FILES if files is None else tuple(files)
    out, missing = {}, []
    for relative in files:
        path = root / relative
        if path.is_file():
            out[relative] = sha256_bytes(path.read_bytes())
        else:
            missing.append(relative)
    if missing:
        raise ResumeValidationError(
            "required source files are missing or are not regular files, so the "
            "raw-bytes source record would be incomplete: " + ", ".join(sorted(missing)))
    return out


def _git_identity(root=None, files=None) -> dict:
    """Git object identity, kept in a record of its own, separate from raw bytes.

    Three deliberately distinct things (IN-25):

    * ``sources.sha256`` — the **raw executed bytes** on disk. Checkout-dependent
      under ``core.autocrlf``; this is what actually ran.
    * ``working_tree_blob_sha1`` — ``git hash-object`` of the working-tree file,
      i.e. git's object id **after git's configured filters** (so EOL
      normalisation applies). Not the raw byte hash, and not proof of a commit.
    * ``head_blob_sha1`` / ``head_status`` — the id of the file **as committed at
      HEAD**, or ``None`` with ``"not_in_head"`` when it is uncommitted.

    ``git ls-files -s`` is deliberately **not** used: it reports the *index*
    entry, which under ``git add -N`` is the empty blob
    ``e69de29bb2d1d6434b8b29ae775ad8c2e48c5391`` — a value with no relationship
    to the file's content.

    A key is never omitted: every declared path gets all three entries, and
    ``head_status`` distinguishes ``"in_head"``, ``"not_in_head"``,
    ``"missing_in_working_tree"`` (the file itself is absent — not to be confused
    with git being unusable) and ``"git_unavailable"`` (git returned nonzero for a
    file that is present). ``files=None`` resolves ``SOURCE_FILES`` at call time
    so this record covers the same declared set as ``_source_hashes``, which
    refuses a missing required source outright.
    """
    root = Path(root or ROOT)
    files = SOURCE_FILES if files is None else tuple(files)
    working, head, status = {}, {}, {}

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True)

    for relative in files:
        content = git("hash-object", "--", relative)
        working[relative] = content.stdout.strip() if content.returncode == 0 else None
        committed = git("rev-parse", "--verify", "--quiet", f"HEAD:{relative}")
        if committed.returncode == 0 and committed.stdout.strip():
            head[relative] = committed.stdout.strip()
            status[relative] = "in_head"
        else:
            head[relative] = None
            status[relative] = "not_in_head"
        if working[relative] is None:
            status[relative] = ("missing_in_working_tree"
                                if not (root / relative).is_file() else "git_unavailable")
    return {
        "kind": "git_object_identity",
        "note": ("working_tree_blob_sha1 is git's filtered object id, NOT the raw "
                 "execution-byte hash in sources.sha256 and NOT proof of a commit"),
        "working_tree_blob_sha1": working,
        "head_blob_sha1": head,
        "head_status": status,
    }


def _recipe(algo) -> dict:
    keys = ("gamma", "gae_lambda", "steps_num", "ppo_epochs", "num_minibatches",
            "clip_range", "entropy_coef", "value_coef", "max_grad_norm",
            "truncate_grad", "normalize_advantages", "actor_lr", "critic_lr",
            "lr_schedule", "num_envs", "num_obs", "num_actions", "max_episode_length")
    recipe = {}
    for key in keys:
        value = getattr(algo, key, None)
        recipe[key] = value if isinstance(value, _PLAIN_TYPES) else _copy_plain(value)
    recipe["betas"] = [float(value) for value in getattr(algo, "betas", ())]
    recipe["obs_rms"] = algo.obs_rms is not None
    recipe["ret_rms"] = algo.ret_rms is not None
    return recipe


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, _PLAIN_TYPES):
        return value if not isinstance(value, bytes) else value.hex()
    if torch.is_tensor(value):
        return {"__tensor__": _describe(value), "sha256": sha256_bytes(
            value.detach().cpu().contiguous().numpy().tobytes())}
    return [type(value).__name__, repr(value)]


def _metadata_digest(metadata: Mapping[str, Any]) -> str:
    """Binding used to commit the payload to its metadata; never written as JSON."""
    canonical = json.dumps(_jsonable(dict(metadata)), sort_keys=True, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def _build_metadata(algo, env, *, epoch, extra, include_git_identity) -> dict:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "epoch": None if epoch is None else int(epoch),
        "iter_count": int(algo.iter_count),
        "step_count": int(algo.step_count),
        "recipe": _recipe(algo),
        "env_class": type(env).__name__,
        "sources": {"kind": "raw_working_tree_bytes", "sha256": _source_hashes()},
        "extra": _copy_plain(dict(extra or {})),
    }
    if include_git_identity:
        # Deliberately a separate record: raw working-tree bytes, git's filtered
        # object id and the committed HEAD blob are three different things under
        # core.autocrlf (IN-25) and must never be conflated.
        metadata["sources_git"] = _git_identity()
    return metadata


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def capture_state(algo, env, extra=None, *, epoch=None, include_git_identity=False) -> ResumeState:
    """Immutable snapshot of a **completed epoch boundary**.

    Args:
        algo: a live ``PPO`` (or the CPU analytic adapter with the same surface).
        env: ``MyoLeg26WalkEnv`` (``warp_data``) or an adapter declaring
            ``RESUME_PLAIN_STATE_FIELDS``. Must be **the same object** as
            ``algo.env`` (``_require_bound_env``), and every declared source file
            must exist (``_source_hashes``), or capture refuses.
        extra: caller metadata (seed, lineage, segment index, freeze hashes).
        epoch: completed epoch index recorded in the metadata.
        include_git_identity: also record committed blob ids, kept distinct from
            the raw working-tree hashes (IN-25).

    Returns:
        ``ResumeState`` holding only copies: the caller may keep training the
        supplied objects, or mutate them in place, without changing it.
    """
    _require_bound_env(algo, env)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    env_record = _capture_env(env)
    algo_record = _capture_algo(algo)
    metadata = _build_metadata(algo, env, epoch=epoch, extra=extra,
                               include_git_identity=include_git_identity)
    return ResumeState(metadata=metadata, algo=algo_record, env=env_record,
                       extra=metadata["extra"])


def _require_bound_env(algo, env) -> None:
    """Refuse a mismatched ``(algo, env)`` pair before anything is read or written.

    ``PPO`` owns its environment (``ppo.py:48``) and steps that one object; every
    captured field is either algorithm state or the physics of the environment
    that produced it. Two same-shaped environments pass every dtype, shape,
    world-count, model-identity and recipe gate, so without an **identity** check
    a caller could pair one run's parameters, Adam moments, normaliser, counters
    and RNG with another run's simulator state and get a segment that looks like a
    faithful boundary. Shape agreement is not interchangeability.

    The algorithm's own binding is never rebound to match the argument: the
    caller's mistake is reported, not silently repaired. Raised before any target
    or process-global mutation.
    """
    bound = getattr(algo, "env", _MISSING)
    if bound is _MISSING:
        raise ResumeValidationError(
            "algo.env is missing: the algorithm carries no environment binding, so "
            "it cannot be confirmed to be the one that produced this state")
    if bound is not env:
        raise ResumeValidationError(
            "algo.env is not the supplied environment: refusing to combine "
            f"algorithm state with a different {type(env).__name__} instance "
            f"(algo.env is a {type(bound).__name__}); identity, not matching "
            "shapes, is the requirement")


def _check(problems, condition, message):
    if not condition:
        problems.append(message)


def _check_like(problems, label, captured, target):
    if torch.is_tensor(captured) and torch.is_tensor(target):
        _check(problems, captured.dtype == target.dtype,
               f"{label}: dtype {captured.dtype} does not match target {target.dtype}")
        _check(problems, tuple(captured.shape) == tuple(target.shape),
               f"{label}: shape {tuple(captured.shape)} does not match target {tuple(target.shape)}")


def _validate_recipe(state, algo, problems) -> None:
    captured = state.metadata.get("recipe")
    if not isinstance(captured, dict):
        problems.append("segment metadata carries no recipe")
        return
    target = _recipe(algo)
    for key in RECIPE_IDENTITY_KEYS:
        if captured.get(key) != target.get(key):
            problems.append(
                f"recipe mismatch on {key}: segment {captured.get(key)!r}, "
                f"target {target.get(key)!r}")


def _validate_algo(state, algo, problems) -> None:
    record = state.algo
    for label, module in (("actor", algo.actor), ("critic", algo.critic)):
        captured, target = record[label], module.state_dict()
        _check(problems, set(captured) == set(target),
               f"{label}: parameter key mismatch "
               f"{sorted(set(captured) ^ set(target))}")
        for key in sorted(set(captured) & set(target)):
            _check_like(problems, f"{label}.{key}", captured[key], target[key])
    for label, optimizer in (("actor_optimizer", algo.actor_optimizer),
                             ("critic_optimizer", algo.critic_optimizer)):
        captured = record[label]
        target = optimizer.state_dict()
        _check(problems, len(captured["param_groups"]) == len(target["param_groups"]),
               f"{label}: param_group count mismatch")
        known = set()
        for index, (left, right) in enumerate(zip(captured["param_groups"], target["param_groups"])):
            _check(problems, list(left["params"]) == list(right["params"]),
                   f"{label}.param_groups[{index}]: parameter index mismatch")
            known.update(int(item) for item in right["params"])
        # ``Optimizer.load_state_dict`` would raise on an unknown index only after
        # the parameters had already been overwritten.
        for index in sorted(captured["state"]):
            _check(problems, int(index) in known,
                   f"{label}: state index {index} is outside the target's parameter groups")
    metric = record.get("best_policy_loss")
    _check(problems,
           isinstance(metric, Mapping) and metric.get("kind") in _METRIC_KINDS,
           f"best_policy_loss: unsupported record {metric!r}")
    _validate_rng(record.get("rng"), problems)
    for name, tensor in record["buffers"].items():
        _check_like(problems, name, tensor, getattr(algo, name))
    for label in ("obs_rms", "ret_rms"):
        captured, target = record[label], getattr(algo, label)
        _check(problems, (captured is None) == (target is None),
               f"{label}: presence mismatch (captured={captured is not None}, "
               f"target={target is not None})")
        if captured is not None and target is not None:
            _check_like(problems, f"{label}.mean", captured["mean"], target.mean)
            _check_like(problems, f"{label}.var", captured["var"], target.var)
    for label in ("episode_loss_meter", "episode_length_meter"):
        captured, target = record[label], getattr(algo, label)
        _check_like(problems, f"{label}.mean", captured["mean"], target.mean)
        _check(problems, captured["max_size"] == int(target.max_size),
               f"{label}: max_size {captured['max_size']} does not match target "
               f"{int(target.max_size)}")
    _check_like(problems, "episode_loss", record["episode_loss"], algo.episode_loss)
    _check_like(problems, "episode_length", record["episode_length"], algo.episode_length)
    obs = record["_current_obs"]
    _check(problems, tuple(obs.shape) == (int(algo.num_envs), int(algo.num_obs)),
           f"_current_obs: shape {tuple(obs.shape)} does not match target "
           f"{(int(algo.num_envs), int(algo.num_obs))}")


def _validate_derived(record: Mapping[str, Any], env, kind, problems) -> None:
    """Gate the preserved-scratch section before anything is written.

    Fail closed on a missing section, a missing or unexpected field, a payload
    that is not a tensor, a dtype or shape that disagrees with the target's own
    backing array (or with the record's own declaration), or a nonfinite stored
    value. Nothing is ever defaulted or zero-filled: a payload that cannot supply
    a required field is refused, which is also how a previous-schema segment is
    handled — by refusal, not migration.
    """
    captured = record.get("derived")
    expected = _derived_names(env, kind)
    if not isinstance(captured, Mapping):
        problems.append(
            f"env.derived: the segment carries no preserved-scratch section, so the "
            f"required fields {sorted(expected)} would be silently defaulted; a payload "
            f"older than {SCHEMA_VERSION} is refused, never migrated")
        return
    if set(captured) != set(expected):
        problems.append(
            f"env.derived: segment records {sorted(captured)} but the target declares "
            f"{sorted(expected)}")
    for name in sorted(set(captured) & set(expected)):
        field = captured[name]
        values = field.get("values") if isinstance(field, Mapping) else None
        if not torch.is_tensor(values):
            problems.append(f"env.derived.{name}: payload carries no tensor")
            continue
        view = _derived_view(env, kind, name)
        if view is None:
            problems.append(f"env.derived.{name}: missing on target")
            continue
        target_shape = tuple(int(dim) for dim in view.shape)
        _check(problems, field.get("dtype") == str(view.dtype),
               f"env.derived.{name}: recorded dtype {field.get('dtype')} does not match "
               f"the target backing array {view.dtype}")
        _check(problems, str(values.dtype) == field.get("dtype"),
               f"env.derived.{name}: payload dtype {values.dtype} does not match the "
               f"recorded dtype {field.get('dtype')}")
        _check(problems, tuple(field.get("shape", ())) == target_shape,
               f"env.derived.{name}: recorded shape {tuple(field.get('shape', ()))} does "
               f"not match the target backing array {target_shape}")
        _check(problems, tuple(values.shape) == target_shape,
               f"env.derived.{name}: payload shape {tuple(values.shape)} does not match "
               f"the target backing array {target_shape}")
        if values.is_floating_point() and not torch.isfinite(values).all():
            problems.append(f"env.derived.{name}: nonfinite values in the stored payload")


def _validate_env(state, env, problems) -> None:
    record = state.env
    try:
        kind = _env_kind(env)
    except ResumeValidationError as error:
        problems.append(str(error))
        return
    _check(problems, kind == record["kind"],
           f"environment state protocol mismatch: segment {record['kind']}, target {kind}")
    _check(problems, int(env.num_envs) == record["num_envs"],
           f"env.num_envs: shape/world count {record['num_envs']} does not match target "
           f"{int(env.num_envs)}")
    _check(problems, int(env.num_actions) == record["num_actions"],
           f"env.num_actions: {record['num_actions']} does not match target "
           f"{int(env.num_actions)}")
    for name, tensor in record["buffers"].items():
        target = getattr(env, name, None)
        if target is not None:
            _check_like(problems, f"env.{name}", tensor, target)
        elif name in ENV_LAZY_BUFFERS:
            # The target has never stepped, so there is no tensor to compare
            # against. Validate the captured buffer against the environment's own
            # observation contract instead, and only then create it on restore.
            reference = env.obs_buf
            _check(problems, tensor.dtype == reference.dtype,
                   f"env.{name}: dtype {tensor.dtype} does not match the target's "
                   f"observation dtype {reference.dtype}")
            expected = (int(env.num_envs), int(env.num_obs))
            _check(problems, tuple(tensor.shape) == expected,
                   f"env.{name}: shape {tuple(tensor.shape)} does not match the target's "
                   f"observation contract {expected}")
        else:
            problems.append(f"env.{name}: missing on target")
    if kind != record["kind"]:
        return
    _validate_derived(record, env, kind, problems)
    if kind == "plain":
        for name, tensor in record["plain"].items():
            target = getattr(env, name, None)
            if target is None:
                problems.append(f"env.{name}: missing on target")
            else:
                _check_like(problems, f"env.{name}", tensor, target)
        return

    warp_record = record["warp"]
    data = env.warp_data
    for name, field in warp_record["fields"].items():
        view = _warp_view(data, name)
        if view is None:
            problems.append(f"warp.{name}: missing on target")
            continue
        _check(problems, str(view.dtype) == field["dtype"],
               f"warp.{name}: dtype {field['dtype']} does not match target backing array "
               f"{view.dtype}")
        _check(problems, tuple(int(dim) for dim in view.shape) == tuple(field["shape"]),
               f"warp.{name}: shape {tuple(field['shape'])} does not match target "
               f"{tuple(view.shape)}")
    for name, field in warp_record["invariants"].items():
        view = _warp_view(data, name)
        if view is None:
            problems.append(f"warp.{name}: missing on target")
            continue
        if tuple(int(dim) for dim in view.shape) != tuple(field["shape"]):
            problems.append(f"warp.{name}: shape mismatch on an invariant field")
        elif not torch.equal(view.detach().cpu(), field["values"]):
            problems.append(
                f"warp.{name}: invariant solver input differs between segment and target")
    try:
        _reject_external_inputs(data)
        target_model = _assert_model_contract(env)
    except ResumeValidationError as error:
        problems.append(str(error))
        return
    for key in ("compiled_model_sha256", "nq", "nv", "na", "nu", "integrator", "disableflags"):
        _check(problems, target_model[key] == warp_record["model"][key],
               f"model identity mismatch on {key}")


def restore_state(state: ResumeState, algo, env) -> dict:
    """Write a captured boundary into ``algo``/``env``; validate first, RNG last.

    Objects are **not** constructed here: the caller supplies a freshly built
    algorithm and environment, so the real ``PPO`` and the CPU adapter share one
    path and module construction's own RNG draws happen before the RNG restore.
    ``env`` must be ``algo.env`` itself — see ``_require_bound_env``.
    """
    if not isinstance(state, ResumeState):
        raise ResumeValidationError("restore_state expects a ResumeState")
    # Before every other gate: physics must not be written into one environment
    # while the algorithm steps another.
    _require_bound_env(algo, env)
    problems = []
    if state.metadata.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {state.metadata.get('schema_version')!r} "
                        f"is not {SCHEMA_VERSION!r}")
    if not problems:
        _validate_recipe(state, algo, problems)
        _validate_algo(state, algo, problems)
        _validate_env(state, env, problems)
    if problems:
        raise ResumeValidationError("; ".join(problems))

    record = state.algo
    with torch.no_grad():
        # 1. learned parameters before optimizer state (Adam casts to param dtype/device)
        algo.actor.load_state_dict({key: value.clone() for key, value in record["actor"].items()})
        algo.critic.load_state_dict({key: value.clone() for key, value in record["critic"].items()})
        for label, optimizer in (("actor_optimizer", algo.actor_optimizer),
                                 ("critic_optimizer", algo.critic_optimizer)):
            optimizer.load_state_dict(_copy_plain(record[label]))
        # 2. normalisers, meters, counters
        for label in ("obs_rms", "ret_rms"):
            captured, target = record[label], getattr(algo, label)
            if captured is not None:
                target.mean = captured["mean"].to(device=target.mean.device).clone()
                target.var = captured["var"].to(device=target.var.device).clone()
                target.count = float(captured["count"])
        for label in ("episode_loss_meter", "episode_length_meter"):
            captured, meter = record[label], getattr(algo, label)
            meter.mean.copy_(captured["mean"].to(device=meter.mean.device))
            meter.current_size = int(captured["current_size"])
        algo.iter_count = int(record["iter_count"])
        algo.step_count = int(record["step_count"])
        algo.episode_loss.copy_(record["episode_loss"].to(device=algo.episode_loss.device))
        algo.episode_length.copy_(record["episode_length"].to(device=algo.episode_length.device))
        algo.episode_loss_his = list(record["episode_loss_his"])
        algo.episode_length_his = list(record["episode_length_his"])
        algo.best_policy_loss = _decode_metric(record["best_policy_loss"])
        algo._current_obs = record["_current_obs"].to(
            device=torch.device(algo.device) if isinstance(algo.device, str) else algo.device).clone()
        # 3. rollout buffers, in place so aliases stay valid
        for name, tensor in record["buffers"].items():
            target = getattr(algo, name)
            target.copy_(tensor.to(device=target.device))
        # 4. environment
        _restore_env(state.env, env)
    # 5. RNG last: everything above (module load, clones) may draw or allocate.
    #    Its capability/format checks already ran in the validation phase above.
    _restore_rng(record["rng"])
    streams = ("python", "numpy", "torch_cpu")
    cuda_restored = record["rng"]["cuda"] is not None
    return {"schema_version": SCHEMA_VERSION, "epoch": state.metadata.get("epoch"),
            "kind": state.env["kind"], "iter_count": algo.iter_count,
            "rng_streams_restored": streams + (("cuda",) if cuda_restored else ()),
            # A CPU-captured segment carries no CUDA stream; the target's own CUDA
            # generator is therefore left alone and continuation is NOT bitwise for
            # any CUDA draw. Reported rather than implied.
            "cuda_rng_restored": cuda_restored,
            "lazy_buffers_cleared": tuple(state.env.get("absent_lazy_buffers", ()))}


def _restore_env(record: Mapping[str, Any], env) -> None:
    for name, tensor in record["buffers"].items():
        target = getattr(env, name, None)
        if name == "actions":
            # Task 2 reassigns this tensor every step; identity is not load-bearing.
            env.actions = tensor.to(device=target.device).clone()
        elif name in ENV_LAZY_BUFFERS:
            # Reassigned wholesale by step(); created here only after validation.
            device = target.device if target is not None else env.obs_buf.device
            setattr(env, name, tensor.to(device=device).clone())
        else:
            target.copy_(tensor.to(device=target.device))
    for name in record.get("absent_lazy_buffers", ()):
        # The segment was taken before the env ever stepped. Leaving another run's
        # buffer in place would silently import foreign state.
        if hasattr(env, name):
            delattr(env, name)
    kind = record["kind"]
    if kind == "plain":
        for name, tensor in record["plain"].items():
            target = getattr(env, name)
            target.copy_(tensor.to(device=target.device))
        _restore_derived(record, env, kind)
        return
    import warp as wp

    data = env.warp_data
    for name, field in record["warp"]["fields"].items():
        view = _warp_view(data, name)
        # Validation already proved dtype and shape equality, so this copy
        # never casts the backing array.
        view.copy_(field["values"].to(device=view.device))
    _restore_derived(record, env, kind)
    wp.synchronize()


def _restore_derived(record: Mapping[str, Any], env, kind) -> None:
    """Write the preserved scratch back in place, byte for byte."""
    for name, field in record["derived"].items():
        view = _derived_view(env, kind, name)
        # Validation already compared dtype and shape against this very view, so
        # the copy never casts and never rebinds the backing array.
        view.copy_(field["values"].to(device=view.device))


# ----------------------------------------------------------------------
# Publication and reading
# ----------------------------------------------------------------------


def _container(state: ResumeState) -> dict:
    metadata = dict(state.metadata)
    inner = {
        "metadata": metadata,
        "algo": dict(state.algo),
        "env": dict(state.env),
        "extra": dict(state.extra),
        "metadata_sha256": _metadata_digest(metadata),
    }
    buffer = io.BytesIO()
    torch.save(inner, buffer)
    payload = buffer.getvalue()
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "payload_sha256": sha256_bytes(payload),
        "payload": payload,
    }


def _publish(temporary: Path, final: Path) -> str:
    """Atomic, non-overwriting publication: the file is absent or complete.

    ``os.replace`` is never used: it would overwrite an existing checkpoint, and
    an exclusive create of the final path followed by a replace would expose a
    zero-byte final file. ``os.link`` publishes the already-complete temporary
    under the final name in one atomic step and fails with ``FileExistsError``
    if anything already holds that name. Windows ``os.rename`` has the same
    no-replace semantics and is the fallback where hard links are unavailable.
    """
    try:
        os.link(temporary, final)
        return "hardlink"
    except FileExistsError:
        raise
    except OSError:
        if os.name != "nt":
            raise
        os.rename(temporary, final)
        return "rename_no_replace"


def write_segment(state: ResumeState, path, *, label: str | None = None) -> dict:
    """Serialise and publish one segment; never overwrites an existing file."""
    if not isinstance(state, ResumeState):
        raise ResumeValidationError("write_segment expects a ResumeState")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"segment destination already exists: {path}")
    container = _container(state)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(temporary, "xb") as handle:
            torch.save(container, handle)
            handle.flush()
            os.fsync(handle.fileno())
        mechanism = _publish(temporary, path)
    finally:
        if temporary.exists():
            os.unlink(temporary)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_bytes(path.read_bytes()),
        "payload_sha256": container["payload_sha256"],
        "mechanism": mechanism,
        "label": label,
        "published_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


REQUIRED_IDENTITY_KEYS = (
    "schema_version", "epoch", "seed", "segment_index", "parent_segment_sha256",
)


def strict_expect(*, epoch, seed, segment_index, parent_segment_sha256) -> dict:
    """The identity binding a **real resume** must pass to ``read_segment``.

    ``read_segment(path)`` with the default ``expect=None`` is **inspection
    mode**: it verifies the schema version, both checksums and the payload
    structure, and it will happily return a segment from another seed, epoch or
    lineage. Epoch, seed, segment index and parent-segment lineage are recorded
    in every segment but are compared **only** against what the caller names
    here. Likewise ``metadata["sources"]`` and any freeze hashes the caller puts
    in ``extra`` are recorded and **never** compared on restore; what restore
    does enforce by itself is ``algo.env is env``, the compiled model identity,
    ``nq/nv/na/nu``, integrator, disableflags, ``eq_active`` and
    ``RECIPE_IDENTITY_KEYS``. The recorded source map is *complete* — capture
    refuses a missing declared source — but completeness is not a comparison:
    the caller still has to check the values.

    So: Unit 4's training driver must call
    ``read_segment(path, expect=strict_expect(...))``. Passing ``expect=None``
    there would resume from an unbound file. ``include_git_identity=False`` is
    likewise the inspection default — a campaign that needs source provenance
    must pass ``True`` and compare the record itself.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "seed": seed,
        "segment_index": segment_index,
        "parent_segment_sha256": parent_segment_sha256,
    }


def _expected_value(metadata: Mapping[str, Any], key: str):
    if key in metadata:
        return metadata[key]
    extra = metadata.get("extra") or {}
    if key in extra:
        return extra[key]
    raise ResumeValidationError(f"segment metadata has no field {key!r} to check")


def read_segment(path, *, expect: Mapping[str, Any] | None = None) -> ResumeState:
    """Load and fully validate a segment. Fails closed before any restore."""
    path = Path(path)
    if not path.is_file():
        raise ResumeValidationError(f"segment file is missing: {path}")
    try:
        container = torch.load(path, weights_only=True)
    except Exception as error:  # truncated, malformed, or not a segment at all
        raise ResumeIntegrityError(f"segment file could not be read: {path}: {error}") from error
    if not isinstance(container, dict) or "payload" not in container:
        raise ResumeIntegrityError(f"segment file is malformed: {path}")
    if container.get("schema_version") != SCHEMA_VERSION:
        raise ResumeValidationError(
            f"schema_version {container.get('schema_version')!r} is not {SCHEMA_VERSION!r}")
    payload = bytes(container["payload"])
    if sha256_bytes(payload) != container.get("payload_sha256"):
        raise ResumeIntegrityError(f"payload checksum mismatch: {path}")
    try:
        inner = torch.load(io.BytesIO(payload), weights_only=True)
    except Exception as error:
        raise ResumeIntegrityError(f"segment payload could not be read: {path}: {error}") from error
    if not isinstance(inner, dict) or set(inner) != {"metadata", "algo", "env", "extra",
                                                     "metadata_sha256"}:
        raise ResumeIntegrityError(f"segment payload is malformed: {path}")
    digest = _metadata_digest(inner["metadata"])
    if digest != inner["metadata_sha256"]:
        raise ResumeIntegrityError(f"metadata checksum mismatch: {path}")
    if _metadata_digest(container["metadata"]) != digest:
        raise ResumeIntegrityError(
            f"metadata commit mismatch: the published metadata differs from the payload's: {path}")
    state = ResumeState(metadata=inner["metadata"], algo=inner["algo"], env=inner["env"],
                        extra=inner["extra"])
    for key, wanted in (expect or {}).items():
        found = _expected_value(state.metadata, key)
        if found != wanted:
            raise ResumeValidationError(
                f"segment identity mismatch on {key!r}: expected {wanted!r}, found {found!r}")
    return state
