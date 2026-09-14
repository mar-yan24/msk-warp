"""Bounded four-arm forward Data-mode probe on one frozen MyoLeg26 visited state.

Preregistered source-isolation probe, never a qualification, never an AD
experiment, and never a tolerance revision. Four arms only:

    A1  native MuJoCo            data mode n/a      float64
    A2  direct Warp step         no-grad Data       float32
    A3  direct Warp step         diff Data          float32
    A4  bridge WarpSimStep       diff Data          float32

The Data-allocation flag is a property of the arm and is handled independently
of the step wrapper, so A2/A3 isolate Data mode at a fixed wrapper and A3/A4
isolate the wrapper at fixed diff Data. The bridge / no-grad corner of that 2x2
is deliberately NOT measured and NOT probed: there is no fifth arm and no
capability probe. Its absence is a stated design limitation, so an A2-vs-A4
difference is never a single-factor result.

Forward only. Every Warp arm runs under ``torch.no_grad()``: no backward pass,
no tape playback, no VJP, no gradient of any kind, in any arm.

Two tolerances live here and must never be confused. The state tolerance
(``STATE_ATOL``/``STATE_RTOL``) is the preserved physics gate. The control
rounding bound (``CONTROL_ATOL``/``CONTROL_RTOL``) is one float32 epsilon and
validates *inputs* only; it may not be reused for any state, feature, reward or
gradient quantity.

Three repetitions at one state cannot establish a noise floor, a causal
explanation, a population qualification, or the absence of an effect. Producing
-source equivalence with the historical untracked replay is unproven, so every
number here is a cross-implementation observation.

Disclosure: the reused environment builder runs with
``allow_unvalidated_gradients=True``, ``grad_contract="off"``, ``no_grad=False``.
That is the flag's sanctioned diagnostic use, pre-existing and not introduced
here. It authorises nothing else.

Usage (output is opened exclusively and never overwrites):
  .venv/Scripts/python.exe scripts/diag_myoleg26_forward_data_mode.py \
      --out logs/myoleg26_forward_data_mode_20260914.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, NamedTuple

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECK_PATH = ROOT / "scripts/check_myoleg26_derivatives.py"
_spec = importlib.util.spec_from_file_location("myoleg26_data_mode_target", CHECK_PATH)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

SCHEMA_VERSION = "myoleg26-forward-data-mode-v1"
SAMPLE_ID = "seed3_episode30013_step60"
HORIZON = 4
REPETITIONS = 3
SUBSTEPS = 4
STATE_SIZE = check.STATE_SIZE

FIELDS = ("qpos", "qvel", "act", "features", "reward_components")
REQUIRED_KEYS = FIELDS + ("ctrl", "observed_data_mode", "warning_status")
# Only the native engine exposes a warning channel this probe reads. Warp
# warnings are NOT queried, so no count may be asserted for a Warp arm.
WARNING_MEASURED_ENGINES = ("native_mujoco",)
WARNING_NOT_QUERIED_ENGINES = ("direct_warp", "bridge_warp")

# Section 5b: preserved original_tolerance, applied to states, features, rewards.
STATE_ATOL = 1e-4
STATE_RTOL = 1e-3
# Section 5a clause 2: one float32 epsilon (2**-23). An INPUT representation
# limit. Never reuse for a state, feature, reward or gradient quantity.
CONTROL_ATOL = 1.1920928955078125e-07
CONTROL_RTOL = 0.0

EXPECTED_HASHES = {
    "dataset": "d009b935bc50b4ae201363dafe2496401033e7ec61585cf1bdb2fb8c758a96bb",
    "original_report": "b3a91b6f2d4054b7ef233b89d80f9d13758e2c495025b1bc48ce053e2ef7a0a6",
    "compiled_model": "863ad1753c0e2cb11d475e1640d321ccd8fc7ecf52ae13690b8c2e282c4153a5",
}

DRIVER_SOURCES = (
    "scripts/diag_myoleg26_forward_data_mode.py",
    "scripts/check_myoleg26_derivatives.py",
    "scripts/check_trajopt_gradients.py",
    "tests/unit/test_myoleg26_forward_data_mode.py",
)

WRAPPERS = ("direct", "bridge")
DATA_MODES = ("no_grad", "diff")

# Sections dropped, in order, if the report will not serialise: everything whose
# content is measured (and therefore the possible source of a nonfinite value).
DEGRADABLE_SECTIONS = ("sample", "trajectories", "control_gate_result", "comparisons",
                       "repeat_comparisons", "state_parity", "completeness")
# Sections kept in a degraded report: identity, declarations, provenance and the
# durable error coordinates. All are JSON-safe by construction.
PRESERVED_SECTIONS = ("schema_version", "scope", "status", "qualification_status",
                      "qualification_reason", "sample_id", "horizon", "repetitions",
                      "substeps", "arms", "design", "state_tolerance",
                      "state_tolerance_note", "control_input_gate",
                      "reference_orientation", "input_precision",
                      "diagnostic_override_disclosure",
                      "comparison_values_not_acceptance_bands", "interpretation_limits",
                      "warning_count_semantics", "output", "provenance", "error", "seconds")

EXIT_OK = 0
EXIT_INTEGRITY = 2
EXIT_NONFINITE = 3
EXIT_INCOMPLETE = 4
EXIT_OUTPUT_EXISTS = 5
EXIT_REPORTING_FAULT = 6


class ProbeFailure(Exception):
    """Terminal measurement failure carrying durable JSON-safe coordinates."""

    code = 1

    def __init__(self, reason, *, arm=None, repetition=None, field=None, index=None, detail=None):
        super().__init__(detail or reason)
        self.reason = str(reason)
        self.detail = detail
        self.coordinates = {"arm": arm, "repetition": repetition, "field": field, "index": index}


class IntegrityFailure(ProbeFailure):
    code = EXIT_INTEGRITY


class NonfiniteData(ProbeFailure):
    code = EXIT_NONFINITE


class IncompleteMeasurement(ProbeFailure):
    code = EXIT_INCOMPLETE


class ArmSpec(NamedTuple):
    """One preregistered arm. `wrapper` and `grad_data` are independent axes."""

    arm: str
    name: str
    engine: str
    data_mode: str
    precision: str
    wrapper: str | None
    grad_data: bool | None


_ARM_SPECS = (
    ArmSpec("A1", "native_float64", "native_mujoco", "n/a", "float64", None, None),
    ArmSpec("A2", "direct_nograd_warp", "direct_warp", "no_grad", "float32", "direct", False),
    ArmSpec("A3", "direct_diff_warp", "direct_warp", "diff", "float32", "direct", True),
    ArmSpec("A4", "bridge_diff_warp", "bridge_warp", "diff", "float32", "bridge", True),
)


def register_arms(specs):
    """Validate the preregistered design. Duplicate arms or cells are rejected."""
    specs = tuple(specs)
    # Duplicates are diagnosed before the count, so a duplicated arm is never
    # reported merely as "wrong number of arms".
    for label, values in (("arm", [s.arm for s in specs]), ("name", [s.name for s in specs]),
                          ("cell", [(s.engine, s.data_mode, s.wrapper, s.grad_data) for s in specs])):
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate arm {label} in the preregistered design: {values}")
    if len(specs) != 4:
        raise ValueError("expected exactly four preregistered arms")
    if sum(s.engine == "native_mujoco" for s in specs) != 1:
        raise ValueError("expected exactly one native reference arm")
    return specs


def arm_specs():
    return register_arms(_ARM_SPECS)


def plan_invocation(spec):
    """Independent (wrapper, grad_data) plan for one Warp arm."""
    if spec.engine == "native_mujoco":
        raise ValueError("the native arm has no Warp step wrapper or Data mode")
    if spec.wrapper not in WRAPPERS or spec.data_mode not in DATA_MODES:
        raise ValueError(f"unknown Warp arm {spec.arm}")
    return {"wrapper": spec.wrapper, "grad_data": bool(spec.grad_data)}


def unmeasured_design_cells(specs=None):
    """Cells of the wrapper x data-mode design this probe deliberately omits."""
    specs = arm_specs() if specs is None else tuple(specs)
    measured = {(s.wrapper, s.data_mode) for s in specs if s.engine != "native_mujoco"}
    return tuple((wrapper, mode) for wrapper in WRAPPERS for mode in DATA_MODES
                 if (wrapper, mode) not in measured)


def declared_pairs():
    """Section 5c: the complete emitted between-arm pair set, actual vs reference."""
    return (("A2", "A1"), ("A3", "A1"), ("A4", "A1"), ("A3", "A2"), ("A4", "A3"))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

class ProbeInputs(NamedTuple):
    """Everything the measurement needs, all of it CPU-resident and hashable to disk.

    A NamedTuple, not a dataclass: this module is also loaded by file spec
    (tests, and the same pattern the sibling checkers use), where
    `dataclasses.dataclass` cannot resolve its own module.
    """

    model: Any
    sample: dict
    metadata: dict
    hashes: dict
    names: dict
    provenance: dict
    horizon: int = HORIZON
    repetitions: int = REPETITIONS
    model_path: str | None = None


def coordinate_names(model):
    """Per-field coordinate names, so a worst index can be named, not just numbered."""
    joint_type = mujoco.mjtJoint
    qwidth = {joint_type.mjJNT_FREE: 7, joint_type.mjJNT_BALL: 4,
              joint_type.mjJNT_SLIDE: 1, joint_type.mjJNT_HINGE: 1}
    vwidth = {joint_type.mjJNT_FREE: 6, joint_type.mjJNT_BALL: 3,
              joint_type.mjJNT_SLIDE: 1, joint_type.mjJNT_HINGE: 1}
    qpos = [f"qpos{i}" for i in range(model.nq)]
    qvel = [f"qvel{i}" for i in range(model.nv)]
    for index in range(model.njnt):
        label = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint{index}"
        kind = joint_type(int(model.jnt_type[index]))
        for offset in range(qwidth[kind]):
            qpos[int(model.jnt_qposadr[index]) + offset] = (
                label if qwidth[kind] == 1 else f"{label}[{offset}]")
        for offset in range(vwidth[kind]):
            qvel[int(model.jnt_dofadr[index]) + offset] = (
                label if vwidth[kind] == 1 else f"{label}[{offset}]")
    actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}"
                 for i in range(model.nu)]
    act = actuators[:model.na] if model.na <= model.nu else [f"act{i}" for i in range(model.na)]
    free_root = model.njnt and joint_type(int(model.jnt_type[0])) == joint_type.mjJNT_FREE
    head = ([f"root_pos_{axis}" for axis in "xyz"] + [f"root_rotmat_{i}" for i in range(9)]
            if free_root else [])
    features = head + qpos[7:] + qvel + act if free_root else qpos + qvel + act
    return {"qpos": qpos, "qvel": qvel, "act": act, "features": features,
            "reward_components": ["net_task_reward", "locomotion_reward"]}


def expected_widths(model):
    """Insertion order is the validation order: states first, controls last."""
    return {"qpos": int(model.nq), "qvel": int(model.nv), "act": int(model.na),
            "features": STATE_SIZE, "reward_components": 2, "ctrl": int(model.nu)}


def warning_record(engine, count=None):
    """Warning provenance for one arm: measured, or explicitly not queried.

    An unqueried channel yields `None` with `warning_status: "not_queried"`,
    never `0`. A fabricated zero would sit in a provenance-bearing artifact under
    the same key as the native arm's genuinely measured count.
    """
    if engine in WARNING_MEASURED_ENGINES:
        return {"warning_count": int(count), "warning_status": "measured"}
    return {"warning_count": None, "warning_status": "not_queried"}


def first_nonfinite(value):
    finite = np.isfinite(np.asarray(value, dtype=np.float64))
    if bool(finite.all()):
        return None
    return [int(i) for i in np.unravel_index(int(np.argmin(finite)), finite.shape)]


def validate_inputs(inputs):
    """Integrity gate. Runs before the rollout factory, so before any GPU allocation."""
    for field, expected in EXPECTED_HASHES.items():
        if inputs.hashes.get(field) != expected:
            raise IntegrityFailure("hash_mismatch", field=field,
                                   detail=f"{field} hash {inputs.hashes.get(field)!r} != {expected!r}")
    model = inputs.model
    if (int(model.nq), int(model.nv), int(model.na), int(model.nu)) != (47, 46, 26, 26):
        raise IntegrityFailure("model_dimensions",
                               detail="expected official MyoLeg26 nq47/nv46/na26/nu26")
    if inputs.horizon != HORIZON or inputs.repetitions != REPETITIONS:
        raise IntegrityFailure("design_shape",
                               detail=f"expected H{HORIZON} x {REPETITIONS} repetitions")
    shapes = {"qpos": (int(model.nq),), "qvel": (int(model.nv),), "act": (int(model.na),),
              "action": (inputs.horizon, int(model.nu))}
    for field, shape in shapes.items():
        value = np.asarray(inputs.sample.get(field))
        if value.shape != shape:
            raise IntegrityFailure("wrong_shape", field=field,
                                   detail=f"initial state {field} shape {value.shape} != {shape}")
        index = first_nonfinite(value)
        if index is not None:
            raise NonfiniteData("nonfinite_value", field=field, index=index)
    if inputs.metadata.get("sample_id") != SAMPLE_ID:
        raise IntegrityFailure("sample_identity", field="sample_id",
                               detail=f"expected sample {SAMPLE_ID}")


def validate_environment(env):
    """Pin the environment's substep count to the one the probe compares.

    The direct arm steps the module constant `SUBSTEPS`; the bridge arm steps
    `env.substeps` inside `WarpSimStep.forward`. If those ever diverge, A3-vs-A4
    would silently compare different amounts of physics, which is exactly the
    quantity under test. Both are 4 today, so this only closes a confound.
    """
    observed = getattr(env, "substeps", None)
    if observed is None or int(observed) != SUBSTEPS:
        raise IntegrityFailure("substeps_mismatch", field="substeps",
                               detail=f"environment reports substeps={observed!r}, the probe "
                                      f"compares {SUBSTEPS} physics steps per control step")
    return int(observed)


def validate_trajectory(spec, repetition, rows, inputs):
    if not isinstance(rows, dict):
        raise IncompleteMeasurement("malformed_trajectory", arm=spec.arm, repetition=repetition,
                                    detail=f"expected a mapping, got {type(rows).__name__}")
    for key in REQUIRED_KEYS:
        if key not in rows:
            raise IncompleteMeasurement("missing_field", arm=spec.arm, repetition=repetition, field=key)
    if rows["observed_data_mode"] != spec.data_mode:
        raise IntegrityFailure("data_mode_mismatch", arm=spec.arm, repetition=repetition,
                               field="observed_data_mode",
                               detail=f"{spec.arm} allocated {rows['observed_data_mode']!r}, "
                                      f"preregistered {spec.data_mode!r}")
    expected_status = warning_record(spec.engine, rows.get("warning_count") or 0)["warning_status"]
    if rows["warning_status"] != expected_status:
        raise IntegrityFailure("warning_status_unsupported", arm=spec.arm, repetition=repetition,
                               field="warning_status",
                               detail=f"{spec.engine} reported warning_status "
                                      f"{rows['warning_status']!r}, expected {expected_status!r}")
    if expected_status == "not_queried" and rows.get("warning_count") is not None:
        raise IntegrityFailure("warning_status_unsupported", arm=spec.arm, repetition=repetition,
                               field="warning_count",
                               detail=f"{spec.engine} has no queried warning channel, so "
                                      f"{rows.get('warning_count')!r} is not a measured value")
    for field, width in expected_widths(inputs.model).items():
        value = np.asarray(rows[field])
        if value.shape != (inputs.horizon, width):
            raise IntegrityFailure("wrong_shape", arm=spec.arm, repetition=repetition, field=field,
                                   detail=f"shape {value.shape} != {(inputs.horizon, width)}")
        index = first_nonfinite(value)
        if index is not None:
            raise NonfiniteData("nonfinite_value", arm=spec.arm, repetition=repetition,
                                field=field, index=index)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def canonical_excitation(actions, dtype):
    """The declared action-to-excitation mapping, evaluated wholly in one precision.

    Action -1 is passive and action 0 is 50% excitation, per the frozen contract.
    """
    dtype = np.dtype(dtype)
    values = np.asarray(actions, dtype=dtype)
    one, half = dtype.type(1), dtype.type(0.5)
    return (half * (np.clip(values, -one, one) + one)).astype(dtype)


def field_metrics(actual, reference, *, atol=STATE_ATOL, rtol=STATE_RTOL, names=None):
    """Asymmetric scaled metric: |actual - reference| / (atol + rtol*|reference|)."""
    raw_actual, raw_reference = np.asarray(actual), np.asarray(reference)
    a = raw_actual.astype(np.float64)
    r = raw_reference.astype(np.float64)
    if a.shape != r.shape:
        raise ValueError(f"comparison shapes differ: {a.shape} != {r.shape}")
    difference = np.abs(a - r)
    scaled = difference / (atol + rtol * np.abs(r))
    finite = bool(np.isfinite(a).all() and np.isfinite(r).all())
    worst = np.unravel_index(int(np.argmax(scaled)), scaled.shape)
    coordinate = int(worst[-1])
    name = names[coordinate] if names is not None and coordinate < len(names) else None
    return {"finite": finite,
            "bitwise_equal": bool(np.array_equal(raw_actual, raw_reference)),
            "passed_state_tolerance": bool(finite and np.all(scaled <= 1)),
            "max_absolute_error": float(difference.max()),
            "max_scaled_error": float(scaled.max()),
            "worst_scaled_index": [int(i) for i in worst],
            "worst_coordinate_name": name,
            "actual_at_worst_scaled": float(a[worst]),
            "reference_at_worst_scaled": float(r[worst])}


def compare_trajectories(actual, reference, names):
    """Every field separately; never pooled into one aggregate number."""
    return {field: field_metrics(actual[field], reference[field], names=names.get(field))
            for field in FIELDS}


def control_gate(actions, native_ctrl, warp_ctrl, *, atol_ctrl=CONTROL_ATOL, rtol_ctrl=CONTROL_RTOL):
    """Section 5a input-precision gate. Validates controls only, never states.

    Clause 1: controls are bitwise equal across the Warp arms.
    Clause 2: the same action-to-excitation mapping is verified on both sides and
    native float64 agrees with Warp float32 within one float32 epsilon, plus the
    canonical-cast check reported additionally.
    """
    warp_arms = sorted(warp_ctrl)
    reference_arm = warp_arms[0]
    reference = np.asarray(warp_ctrl[reference_arm])
    warp_pairs = []
    for arm in warp_arms[1:]:
        current = np.asarray(warp_ctrl[arm])
        difference = np.abs(current.astype(np.float64) - reference.astype(np.float64))
        warp_pairs.append({"actual_arm": arm, "reference_arm": reference_arm,
                           # A control-INPUT pair permitted by section 5a clause 1. It is not a
                           # section 5c state comparison: A4-vs-A2 appears here by design and
                           # must not be read as the historical bridge_vs_direct state pair.
                           "pair_kind": "control_input_clause1",
                           "clause": 1, "requirement": "bitwise_equal",
                           "bitwise_equal": bool(np.array_equal(current, reference)),
                           "max_absolute_error": float(difference.max())})
    clause1 = all(pair["bitwise_equal"] for pair in warp_pairs)

    native = np.asarray(native_ctrl).astype(np.float64)
    canonical64 = canonical_excitation(actions, np.float64)
    canonical32 = canonical_excitation(actions, np.float32)
    native_mapping = bool(np.array_equal(np.asarray(native_ctrl), canonical64))
    mapping = native_mapping
    cast = True
    native_pairs = []
    for arm in warp_arms:
        current = np.asarray(warp_ctrl[arm])
        arm_mapping = bool(np.array_equal(current, canonical32))
        arm_cast = bool(np.array_equal(current, canonical64.astype(np.float32)))
        mapping = mapping and arm_mapping
        cast = cast and arm_cast
        difference = np.abs(current.astype(np.float64) - native)
        allowed = atol_ctrl + rtol_ctrl * np.abs(native)
        worst = np.unravel_index(int(np.argmax(difference)), difference.shape)
        native_pairs.append({"actual_arm": arm, "reference_arm": "A1", "clause": 2,
                             "requirement": "same mapping, then |error| <= atol_ctrl",
                             "mapping_equivalent": arm_mapping,
                             "canonical_cast_bitwise_equal": arm_cast,
                             "within_bound": bool(np.all(difference <= allowed)),
                             "max_absolute_error": float(difference.max()),
                             "worst_index": [int(i) for i in worst]})
    clause2_bound = all(pair["within_bound"] for pair in native_pairs)
    return {"atol_ctrl": atol_ctrl, "rtol_ctrl": rtol_ctrl,
            "applies_to": "control/excitation inputs only",
            "never_reuse_for": ["qpos", "qvel", "act", "features", "reward", "gradient"],
            "warp_reference_arm": reference_arm,
            "clause1_bitwise_across_warp_arms": clause1,
            "clause2_within_rounding_bound": clause2_bound,
            "native_mapping_equivalent": native_mapping,
            "mapping_equivalent": bool(mapping),
            "canonical_cast_bitwise_equal": bool(cast),
            "warp_pairs": warp_pairs, "clause2_native_pairs": native_pairs,
            "passed": bool(clause1 and clause2_bound and mapping)}


# --------------------------------------------------------------------------- #
# Comparison assembly
# --------------------------------------------------------------------------- #

def _indexed(trajectories):
    return {(t["arm"], t["repetition"]): t for t in trajectories}


def between_arm_comparisons(trajectories, names, repetitions):
    indexed = _indexed(trajectories)
    result = []
    for repetition in range(repetitions):
        for actual_arm, reference_arm in declared_pairs():
            actual, reference = indexed[(actual_arm, repetition)], indexed[(reference_arm, repetition)]
            result.append({"scope": "between_arm", "repetition": repetition,
                           "actual_arm": actual_arm, "reference_arm": reference_arm,
                           "actual_name": actual["name"], "reference_name": reference["name"],
                           "actual_engine": actual["engine"], "reference_engine": reference["engine"],
                           "actual_data_mode": actual["data_mode"],
                           "reference_data_mode": reference["data_mode"],
                           "fields": compare_trajectories(actual, reference, names)})
    return result


def within_arm_comparisons(trajectories, names, repetitions):
    """Repeat spread is a property of one arm; never subtracted from a between-arm error."""
    indexed = _indexed(trajectories)
    result = []
    for spec in arm_specs():
        base = indexed[(spec.arm, 0)]
        for repetition in range(1, repetitions):
            fields = compare_trajectories(indexed[(spec.arm, repetition)], base, names)
            result.append({"scope": "within_arm_repeat", "arm": spec.arm, "name": spec.name,
                           "actual_arm": spec.arm, "reference_arm": spec.arm,
                           "repetition": repetition, "reference_repetition": 0,
                           "bitwise_repeat_stable": all(fields[f]["bitwise_equal"] for f in FIELDS),
                           "fields": fields})
    return result


def summarize_parity(comparisons):
    failures = [{"actual_arm": c["actual_arm"], "reference_arm": c["reference_arm"],
                 "repetition": c["repetition"],
                 "failing_fields": [f for f in FIELDS if not c["fields"][f]["passed_state_tolerance"]]}
                for c in comparisons
                if any(not c["fields"][f]["passed_state_tolerance"] for f in FIELDS)]
    return {"tolerance": {"atol": STATE_ATOL, "rtol": STATE_RTOL},
            "passed": not failures,
            "note": "A recorded parity failure is a physics result, not a run error; "
                    "the exit status is unaffected by it.",
            "failures": failures}


def assess_completeness(report, repetitions):
    """Promotion gate: 12 trajectories, finite declared comparisons, valid provenance."""
    expected = len(arm_specs()) * repetitions
    count = len(report["trajectories"])
    if count != expected:
        raise IncompleteMeasurement("trajectory_count",
                                    detail=f"{count} trajectories recorded, expected {expected}")
    for entry in list(report["comparisons"]) + list(report["repeat_comparisons"]):
        for field, metrics in entry["fields"].items():
            if not metrics["finite"] or not np.isfinite(metrics["max_scaled_error"]):
                raise NonfiniteData("nonfinite_comparison", arm=entry["actual_arm"],
                                    repetition=entry["repetition"], field=field)
    if not report["control_gate_result"]["passed"]:
        raise IntegrityFailure("control_input_gate", field="ctrl",
                               detail="section 5a control/excitation input gate failed")
    missing = [key for key in DRIVER_SOURCES if not report["provenance"]["driver_sources"].get(key)]
    if missing or not report["provenance"]["runtime"]:
        raise IntegrityFailure("provenance", field="provenance",
                               detail=f"missing provenance entries: {missing}")
    return {"expected_trajectory_count": expected, "trajectory_count": count,
            "expected_arm_count": len(arm_specs()), "repetitions": repetitions,
            "comparison_count": len(report["comparisons"]),
            "repeat_comparison_count": len(report["repeat_comparisons"]),
            "provenance_validated": True,
            "promoted_after": "all 12 trajectories present and complete, all declared "
                              "comparisons finite, control input gate passed, provenance validated"}


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #

def native_trajectory(model, sample, *, substeps=SUBSTEPS, contract=None):
    contract = check.TASK_CONTRACT if contract is None else contract
    data = mujoco.MjData(model)
    for name in ("qpos", "qvel", "act"):
        getattr(data, name)[:] = sample[name]
    rows = {key: [] for key in FIELDS + ("ctrl",)}
    for action in sample["action"]:
        excitation = canonical_excitation(action, np.float64)
        data.ctrl[:] = excitation
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        rows["qpos"].append(data.qpos.copy())
        rows["qvel"].append(data.qvel.copy())
        rows["act"].append(data.act.copy())
        rows["ctrl"].append(np.array(excitation, dtype=np.float64, copy=True))
        rows["features"].append(check.native_state_features(data.qpos, data.qvel, data.act))
        rows["reward_components"].append(
            check.native_reward_components(model, data, excitation, contract, substeps))
    result = {key: np.asarray(value, dtype=np.float64) for key, value in rows.items()}
    result["observed_data_mode"] = "n/a"
    result.update(warning_record("native_mujoco", sum(int(w.number) for w in data.warning)))
    return result


def warp_trajectory(env, sample, *, wrapper, grad_data, substeps=SUBSTEPS):
    """One Warp arm. `wrapper` and `grad_data` are independent, never derived from each other."""
    import torch
    import warp as wp
    import mujoco_warp as mjw
    from msk_warp import backend, bridge

    env.warp_data = backend.make_data(env.mjm, env.warp_model, 1, env._njmax, grad=bool(grad_data))
    mjw.reset_data(env.warp_model, env.warp_data)
    observed = "diff" if bool(getattr(env.warp_data.qpos, "requires_grad", False)) else "no_grad"
    qpos, qvel, act = (torch.tensor(sample[key][None], dtype=torch.float32, device=env.device)
                       for key in ("qpos", "qvel", "act"))
    bridge._write_state(env.warp_data, qpos, qvel, act)
    rows = {key: [] for key in FIELDS + ("ctrl",)}
    with torch.no_grad():
        for command in sample["action"]:
            action = torch.tensor(command[None], dtype=torch.float32, device=env.device)
            excitation = .5 * (action.clamp(-1, 1) + 1)
            if wrapper == "bridge":
                qpos, qvel, act = bridge.WarpSimStep.apply(excitation, qpos, qvel, act, env)
            elif wrapper == "direct":
                # .detach() matches the codebase's _wp_from and makes this independent of
                # the surrounding torch.no_grad() context.
                wp.copy(env.warp_data.ctrl, wp.from_torch(excitation.detach().contiguous()))
                for _ in range(substeps):
                    mjw.step(env.warp_model, env.warp_data)
                wp.synchronize()
                qpos, qvel, act = bridge._state_tensors(env.warp_data)
            else:
                raise ValueError(f"unknown step wrapper {wrapper!r}")
            obs = env._obs_from_state(qpos, qvel, act, action)
            reward = env.task_contract.reward_terms(obs, excitation, env.control_dt)
            components = torch.stack((reward["locomotion_reward"] - reward["effort_cost"],
                                      reward["locomotion_reward"]), dim=-1)
            for name, value in (("qpos", qpos), ("qvel", qvel), ("act", act), ("ctrl", excitation),
                                ("features", check.torch_state_features(qpos, qvel, act)),
                                ("reward_components", components)):
                rows[name].append(value.detach().cpu().numpy()[0].copy())
    result = {key: np.asarray(value, dtype=np.float32) for key, value in rows.items()}
    result["observed_data_mode"] = observed
    # No Warp warning channel is queried here, so no count is asserted.
    result.update(warning_record("direct_warp" if wrapper == "direct" else "bridge_warp"))
    return result


def load_inputs(args):
    """CPU-only. Nothing here allocates on the GPU."""
    source, original = Path(args.states), Path(args.original_report)
    arrays, metadata, model, model_path = check.load_dataset(source)
    index = next((i for i, row in enumerate(metadata["samples"]) if row["sample_id"] == SAMPLE_ID), None)
    if index is None:
        raise IntegrityFailure("sample_identity", field="sample_id",
                               detail=f"{SAMPLE_ID} absent from {source}")
    sample = {key: value[index] for key, value in arrays.items()}
    sample["action"] = np.asarray(sample["action"])[:HORIZON]
    return ProbeInputs(model=model, sample=sample, metadata=metadata["samples"][index],
                       hashes={"dataset": check.sha256(source),
                               "original_report": check.sha256(original),
                               "compiled_model": metadata["compiled_model_sha256"]},
                       names=coordinate_names(model),
                       provenance=check.runtime_provenance(model),
                       horizon=HORIZON, repetitions=REPETITIONS, model_path=str(model_path))


def make_rollout_factory(args):
    """Deferred GPU allocation: the environment is built only when this is called."""

    def factory(inputs):
        env = check.build_warp_environment(
            inputs.model_path, SimpleNamespace(device=args.device, backward_mode=args.backward_mode))
        validate_environment(env)
        if check.compiled_model_sha256(env.mjm) != EXPECTED_HASHES["compiled_model"]:
            raise IntegrityFailure("hash_mismatch", field="compiled_model",
                                   detail="the Warp environment compiled a different model")

        def rollout(spec, repetition):
            if spec.engine == "native_mujoco":
                return native_trajectory(inputs.model, inputs.sample)
            return warp_trajectory(env, inputs.sample, **plan_invocation(spec))

        return rollout

    return factory


def collect_trajectories(inputs, rollout):
    trajectories = []
    for spec in arm_specs():
        for repetition in range(inputs.repetitions):
            rows = rollout(spec, repetition)
            if rows is None:
                raise IncompleteMeasurement("missing_trajectory", arm=spec.arm, repetition=repetition)
            validate_trajectory(spec, repetition, rows, inputs)
            trajectories.append({"arm": spec.arm, "name": spec.name, "engine": spec.engine,
                                 "data_mode": spec.data_mode, "precision": spec.precision,
                                 "wrapper": spec.wrapper, "grad_data": spec.grad_data,
                                 "observed_data_mode": rows["observed_data_mode"],
                                 "repetition": repetition,
                                 # Passed through, never coerced: None must stay None.
                                 "warning_count": rows.get("warning_count"),
                                 "warning_status": rows["warning_status"],
                                 **{field: rows[field] for field in FIELDS + ("ctrl",)}})
    return trajectories


# --------------------------------------------------------------------------- #
# Report and driver
# --------------------------------------------------------------------------- #

def driver_source_hashes():
    """Section 5e: the reused provenance helper does not hash its caller, so do it here."""
    return {relative: check.sha256(ROOT / relative) for relative in DRIVER_SOURCES}


def json_safe(value):
    """NumPy-safe conversion that deliberately preserves nonfinite floats.

    Nonfinite values are NOT silently mapped to null: the integrity gate is
    reached first, and `allow_nan=False` must remain able to refuse anything
    that slipped past it.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def error_payload(failure):
    return {"reason": failure.reason, "exit_code": int(failure.code),
            "coordinates": json_safe(failure.coordinates),
            "detail": None if failure.detail is None else str(failure.detail)}


def initial_report(path):
    specs = arm_specs()
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "One frozen visited state, H4, four preregistered arms, three repetitions each; "
                 "forward only, no backward pass, no VJP, no gradient of any kind. "
                 "Source-isolation probe, not a qualification and not a tolerance revision.",
        "status": "running",
        "qualification_status": "not_assessed",
        "qualification_reason": "A one-state, three-repetition forward probe cannot qualify a "
                                "gradient, a policy, a population or a backend. No verdict is recorded.",
        "sample_id": SAMPLE_ID,
        "horizon": HORIZON,
        "repetitions": REPETITIONS,
        "substeps": SUBSTEPS,
        "arms": [dict(spec._asdict()) for spec in specs],
        "design": {
            "axes": {"wrapper": list(WRAPPERS), "data_mode": list(DATA_MODES)},
            "measured_cells": [[s.wrapper, s.data_mode] for s in specs if s.wrapper],
            "unmeasured_cells": [list(cell) for cell in unmeasured_design_cells(specs)],
            "limitation": "The bridge / no-grad corner is not measured and not probed: no fifth "
                          "arm, no capability probe, no extra invocation. An A2-vs-A4 difference "
                          "is therefore never a single-factor result.",
        },
        "state_tolerance": {"atol": STATE_ATOL, "rtol": STATE_RTOL},
        "state_tolerance_note": "Identical to the preserved original_tolerance; applied to states, "
                                "features and rewards. Not revisable in this probe.",
        "control_input_gate": {
            "atol_ctrl": CONTROL_ATOL, "rtol_ctrl": CONTROL_RTOL,
            "basis": "one float32 epsilon (2**-23); an input representation limit",
            "applies_to": "control/excitation inputs only",
            "never_reuse_for": ["qpos", "qvel", "act", "features", "reward", "gradient"],
            "clauses": ["bitwise equality across the Warp arms",
                        "verified identical action-to-excitation mapping, then agreement with "
                        "native float64 within atol_ctrl, plus the canonical-cast check"],
        },
        "reference_orientation": {
            "metric": "scaled = |actual - reference| / (atol + rtol*|reference|)",
            "between_arm_pairs": [{"actual_arm": a, "reference_arm": r} for a, r in declared_pairs()],
            "within_arm_repeats": "repetition 0",
            "note": "The emitted pair set is fixed in advance and exposes actual/reference "
                    "identity as explicit fields. No other pair is reported.",
        },
        "input_precision": "Captured state and actions are float32 values promoted to float64 for "
                           "the native arm; each Warp arm casts back to float32. Native excitation "
                           "is computed in float64, Warp excitation in float32.",
        "diagnostic_override_disclosure": "The reused environment builder runs with "
                                          "allow_unvalidated_gradients=True, grad_contract='off', "
                                          "no_grad=False. Pre-existing sanctioned diagnostic use; "
                                          "authorises no backward pass, VJP, tape, gradient "
                                          "qualification or estimator training.",
        "comparison_values_not_acceptance_bands": {
            "bridge_native_qvel_absolute_error": 0.0010227001164902216,
            "bridge_native_qvel_scaled_error": 9.426057707333818,
            "direct_repeat_qvel_spread": 5.054473876953125e-04,
            "excitation_direct_vs_native_absolute_error": 2.9802322387695312e-08,
            "note": "Historical comparison values only. Not acceptance bands, not an error floor, "
                    "not targets. Different fresh results remain valid observations.",
        },
        "interpretation_limits": [
            "One state and three repetitions support no population inference.",
            "Three repetitions cannot establish a noise floor, a causal explanation, a population "
            "qualification, or the absence of an effect. Three observations are not a null result.",
            "No no-effect claim and no causal claim is made, regardless of outcome; the native/Warp "
            "gap is not localised and no specific backend kernel, atomic, contact ordering or "
            "solver conditioning is claimed as a cause.",
            "The bridge / no-grad corner is unmeasured, so an A2-vs-A4 difference is not a "
            "single-factor result.",
            "Within-arm nondeterminism can be of the same order as the effects being compared, so "
            "each arm's repeat stability is reported before any between-arm reading.",
            "Execution is arm-major (A1 x3, A2 x3, A3 x3, A4 x3), so first-touch kernel "
            "compilation and allocator warm-up are confounded with arm identity. The approved "
            "protocol fixes no execution order; repetition-major interleaving would be stronger "
            "and was not used, so an ordering effect cannot be separated from an arm effect.",
            "A3-vs-A4 does not isolate the step wrapper alone. WarpSimStep additionally performs a "
            "redundant per-control-step write-back of the previous output state and four clone "
            "allocations per step, which the direct path does not. The written-back values are "
            "bit-identical, so the arms stay mathematically equivalent, but the comparison bundles "
            "call path plus write-back plus a different per-step allocation pattern.",
            "features embeds qvel verbatim at columns 52:98, so a qvel failure and a features "
            "failure at the same element are one quantity reported twice, not two independent "
            "failures. Do not count them as separate evidence.",
            "The state atol of 1e-4 is large relative to the magnitude of reward_components at "
            "this state, so a reward_components pass is weakly informative at this scale. This is "
            "a statement about sensitivity, not a reason to change the frozen tolerance.",
            "Producing-source equivalence with the historical untracked replay is only partly "
            "established: the native arm is bitwise identical to the preserved native arm on all "
            "six fields at this one state and repetition, while the Warp arms remain unproven. "
            "Warp-side comparisons are cross-implementation and no Warp difference from "
            "2026-09-13 is a like-for-like delta.",
            "The frozen v1 forward result is not relabelled a pass and no preregistered tolerance "
            "is relaxed.",
        ],
        "warning_count_semantics": {
            "measured_for": list(WARNING_MEASURED_ENGINES),
            "not_queried_for": list(WARNING_NOT_QUERIED_ENGINES),
            "note": "Only the native arm's warning counter is read. Warp arms record "
                    "warning_count: null with warning_status: 'not_queried' - an explicit "
                    "absence, never a measured zero. Do not read a Warp null as 'no warnings'.",
        },
        "output": {"path": str(path), "mode": "exclusive_create"},
        "provenance": None,
        "sample": None,
        "trajectories": [],
        "control_gate_result": None,
        "comparisons": [],
        "repeat_comparisons": [],
        "state_parity": None,
        "completeness": None,
        "error": None,
        "seconds": None,
    }


def measure(report, inputs, rollout):
    """All physics and all comparisons. Raises ProbeFailure on any fail-closed gate."""
    report["provenance"] = {"runtime": inputs.provenance, "driver_sources": driver_source_hashes()}
    report["sample"] = {"sample_id": inputs.metadata.get("sample_id"), "metadata": inputs.metadata,
                        "hashes": dict(inputs.hashes),
                        "initial_state": {key: np.asarray(inputs.sample[key])
                                          for key in ("qpos", "qvel", "act")},
                        "action_window": np.asarray(inputs.sample["action"])}
    trajectories = collect_trajectories(inputs, rollout)
    indexed = _indexed(trajectories)
    gates = []
    for repetition in range(inputs.repetitions):
        gate = control_gate(inputs.sample["action"],
                            indexed[("A1", repetition)]["ctrl"],
                            {arm: indexed[(arm, repetition)]["ctrl"] for arm in ("A2", "A3", "A4")})
        gate["repetition"] = repetition
        gates.append(gate)
    report["trajectories"] = trajectories
    report["control_gate_result"] = {
        "atol_ctrl": CONTROL_ATOL, "rtol_ctrl": CONTROL_RTOL,
        "passed": all(gate["passed"] for gate in gates), "per_repetition": gates}
    report["comparisons"] = between_arm_comparisons(trajectories, inputs.names, inputs.repetitions)
    report["repeat_comparisons"] = within_arm_comparisons(trajectories, inputs.names, inputs.repetitions)
    report["state_parity"] = summarize_parity(report["comparisons"])
    report["completeness"] = assess_completeness(report, inputs.repetitions)
    return report


def execute(out_path, *, inputs_factory, rollout_factory, printer=print, clock=time.monotonic):
    """Open the output exclusively first, then load inputs, then allocate the GPU.

    Exit 0 means a complete, valid measurement was recorded, regardless of whether
    forward parity passed. A non-zero code is an integrity, nonfinite,
    completeness, output-exclusivity or reporting fault, never a parity result.
    """
    path = Path(out_path)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        # Nothing opened, nothing written, nothing truncated. No inputs loaded and
        # no GPU allocation: both factories are still untouched.
        try:
            printer(f"output_exists: refusing to write {path}")
        except Exception:
            pass
        return EXIT_OUTPUT_EXISTS
    started = clock()
    report = initial_report(path)

    def write(payload):
        text = json.dumps(json_safe(payload), indent=2, allow_nan=False)
        handle.seek(0)
        handle.write(text + "\n")
        handle.truncate()
        handle.flush()

    def save():
        """Write the report, degrading to an error-only payload if it will not serialise.

        A nonfinite value anywhere in the measured sections must never strand the
        artifact mid-write or escape as an uncaught ValueError: section 5d requires
        the durable JSON-safe coordinates to reach disk and the coded exit to be
        returned. So on a serialisation failure the measured sections are dropped,
        the identity/provenance/error sections are kept, and what was dropped is
        named. Returns True when the full report was written.
        """
        try:
            write(report)
            return True
        except ValueError as error:
            dropped = [key for key in DEGRADABLE_SECTIONS if report.get(key)]
            degraded = {key: report.get(key) for key in PRESERVED_SECTIONS}
            degraded["serialisation_fault"] = {
                "reason": "report_not_json_serialisable",
                "detail": "".join(traceback.format_exception_only(type(error), error)).strip(),
                "dropped_sections": dropped,
                "note": "A nonfinite or non-JSON value was present in the measured "
                        "sections. They are dropped rather than written partially; the "
                        "status, error coordinates and provenance above stand. This is "
                        "never a complete measurement.",
            }
            try:
                write(degraded)
            except ValueError:
                # Last resort: primitives only, so this cannot fail in turn. The
                # coded exit and the failure reason still reach disk.
                write({"schema_version": SCHEMA_VERSION, "status": str(report.get("status")),
                       "qualification_status": "not_assessed",
                       "sample_id": SAMPLE_ID, "output": {"path": str(path)},
                       "error": {"reason": str((report.get("error") or {}).get("reason")),
                                 "exit_code": int((report.get("error") or {}).get("exit_code") or
                                                  EXIT_NONFINITE)},
                       "serialisation_fault": {"reason": "report_not_json_serialisable",
                                               "dropped_sections": ["all measured and "
                                                                    "declarative sections"]}})
            return False

    def fail(status_error, code):
        report["status"] = "measurement_incomplete"
        report["error"] = status_error
        report["seconds"] = float(clock() - started)
        save()
        try:
            printer(f"measurement_incomplete [{status_error['reason']}] "
                    f"{status_error['coordinates']}: {path}")
        except Exception:
            pass
        return code

    with handle:
        save()
        try:
            inputs = inputs_factory()
            validate_inputs(inputs)
            rollout = rollout_factory(inputs)
            measure(report, inputs, rollout)
        except ProbeFailure as failure:
            return fail(error_payload(failure), failure.code)
        except Exception as error:  # durable evidence for an unanticipated fault
            return fail({"reason": "measurement_error", "exit_code": EXIT_INCOMPLETE,
                         "coordinates": {"arm": None, "repetition": None,
                                         "field": None, "index": None},
                         "detail": "".join(traceback.format_exception_only(type(error), error)).strip()},
                        EXIT_INCOMPLETE)
        report["status"] = "measurement_complete"
        report["seconds"] = float(clock() - started)
        if not save():
            # An unserialisable payload is never a successful measurement, even
            # though every gate passed: the recorded metrics cannot be trusted.
            return fail({"reason": "serialisation_fault", "exit_code": EXIT_NONFINITE,
                         "coordinates": {"arm": None, "repetition": None,
                                         "field": None, "index": None},
                         "detail": "the completed report contained a value that is not "
                                   "JSON-serialisable with allow_nan=False"},
                        EXIT_NONFINITE)
        try:
            printer(f"measurement_complete state_parity_passed="
                    f"{report['state_parity']['passed']} qualification_status="
                    f"{report['qualification_status']}: {path}")
        except Exception as error:
            # A reporting fault never relabels a recorded physics result.
            report["reporting_fault"] = {
                "reason": "final_print_failed",
                "detail": "".join(traceback.format_exception_only(type(error), error)).strip(),
                "note": "The measurement above is complete and stands; only final printing failed."}
            save()
            return EXIT_REPORTING_FAULT
        return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh exclusive artifact path; an existing path is refused")
    parser.add_argument("--states", default=str(ROOT / "logs/myoleg26_visited_v1.npz"))
    parser.add_argument("--original-report", default=str(ROOT / "logs/myoleg26_derivatives_v1.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backward-mode", choices=("tape_per_substep", "tape"),
                        default="tape_per_substep",
                        help="environment construction only; no backward pass is ever run")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return execute(args.out, inputs_factory=lambda: load_inputs(args),
                   rollout_factory=make_rollout_factory(args))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
