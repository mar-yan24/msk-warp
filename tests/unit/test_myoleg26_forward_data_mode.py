"""CPU controls for the bounded four-arm forward Data-mode probe; no GPU launches.

Every test runs the real driver module against controlled inputs. The only
injected boundary is the trajectory rollout callable, which stands in for the
native MuJoCo and Warp engines; the integrity, comparison, control-gate, status,
exit-code and serialisation logic under test is the production logic. Expected
values are hand-derived (analytic float32 rounding steps, literal coordinate
names taken from the frozen model, stdlib hashlib) and never computed with the
code under test.

The probe measures four arms only. The bridge / no-grad corner of the
wrapper x data-mode design is deliberately unmeasured; `test_unmeasured_design
_corner_is_declared` pins that as a stated limitation, not an omission.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "scripts/diag_myoleg26_forward_data_mode.py"
_spec = importlib.util.spec_from_file_location("myoleg26_forward_data_mode", DRIVER_PATH)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)

MODEL_XML = ROOT / "msk_warp/assets/myoleg26/flat_boxes.xml"

# Hand-derived float32 rounding fixture, independent of the driver.
# action = 2**-24 exactly.
#   float64 mapping: 0.5 * (2**-24 + 1)            = 0.5 + 2**-25   (exact)
#   float32 mapping: fl32(fl32(2**-24) + 1) * 0.5  = 0.5            (ties-to-even)
#   difference: 2**-25 = 2.9802322387695312e-08, the recorded prior.
ROUNDING_ACTION = 2.0 ** -24
NATIVE_ROUNDING_CTRL = 0.5 + 2.0 ** -25
WARP_ROUNDING_CTRL = 0.5
ONE_FLOAT32_EPSILON = 1.1920928955078125e-07


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_XML))


@pytest.fixture(scope="module")
def names(model):
    return diag.coordinate_names(model)


def sample_for(model, *, action_value=None, horizon=4):
    actions = (np.full((horizon, model.nu), action_value, dtype=np.float64) if action_value is not None
               else np.linspace(-.9, -.1, horizon * model.nu).reshape(horizon, model.nu)
               .astype(np.float32).astype(np.float64))
    return {"qpos": model.key_qpos[0].astype(np.float32).astype(np.float64),
            "qvel": np.linspace(-.02, .02, model.nv).astype(np.float32).astype(np.float64),
            "act": np.full(model.na, .2, dtype=np.float32).astype(np.float64),
            "action": actions}


def inputs_for(base_model, base_names, **overrides):
    """`overrides` replace ProbeInputs fields by name, including `model` itself."""
    sample = overrides.pop("sample", None) or sample_for(base_model)
    kwargs = {"model": base_model, "sample": sample, "names": base_names,
              "hashes": dict(diag.EXPECTED_HASHES), "metadata": {"sample_id": diag.SAMPLE_ID},
              "provenance": {"packages": {}}, "horizon": diag.HORIZON,
              "repetitions": diag.REPETITIONS}
    kwargs.update(overrides)
    return diag.ProbeInputs(**kwargs)


def trajectory(model, spec, sample, *, horizon=4, drift=0.0, ctrl=None, data_mode=None):
    """Deterministic synthetic trajectory with the frozen field widths.

    `drift` is added to every qvel entry so a controlled, hand-known parity
    difference can be injected without touching any other field.
    """
    width = {"qpos": model.nq, "qvel": model.nv, "act": model.na,
             "features": diag.STATE_SIZE, "reward_components": 2}
    rows = {}
    for field, size in width.items():
        base = np.arange(horizon * size, dtype=np.float64).reshape(horizon, size) * 1e-3
        rows[field] = base + (drift if field == "qvel" else 0.0)
    dtype = np.float64 if spec.precision == "float64" else np.float32
    rows["ctrl"] = (np.asarray(ctrl, dtype=dtype) if ctrl is not None
                    else diag.canonical_excitation(sample["action"][:horizon], dtype))
    rows["observed_data_mode"] = spec.data_mode if data_mode is None else data_mode
    # Literal, not built with the code under test: only the native engine has a
    # warning channel this probe queries.
    native = spec.engine == "native_mujoco"
    rows["warning_count"] = 0 if native else None
    rows["warning_status"] = "measured" if native else "not_queried"
    return rows


def rollout_from(model, sample, *, drifts=None, controls=None, data_modes=None, omit=(), horizon=4):
    """Rollout callable returning one synthetic trajectory per (arm, repetition)."""
    drifts = drifts or {}
    controls = controls or {}
    data_modes = data_modes or {}

    def rollout(spec, repetition):
        if (spec.arm, repetition) in omit:
            return None
        return trajectory(model, spec, sample, horizon=horizon,
                          drift=drifts.get((spec.arm, repetition), drifts.get(spec.arm, 0.0)),
                          ctrl=controls.get(spec.arm), data_mode=data_modes.get(spec.arm))

    return rollout


def run(tmp_path, base_model, base_names, *, rollout, filename="probe.json", printer=None, **overrides):
    """`overrides` are forwarded to ProbeInputs, so a test may replace `model` itself."""
    out = tmp_path / filename
    code = diag.execute(out, inputs_factory=lambda: inputs_for(base_model, base_names, **overrides),
                        rollout_factory=lambda _inputs: rollout,
                        printer=(printer if printer is not None else (lambda *a, **k: None)))
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return code, report


# --------------------------------------------------------------------------- #
# Arm registry and the preregistered design
# --------------------------------------------------------------------------- #

def test_exactly_four_preregistered_arms_with_the_declared_engine_and_precision():
    """Catches a fifth arm, a dropped arm, or a changed engine/precision assignment."""
    specs = diag.arm_specs()
    assert [s.arm for s in specs] == ["A1", "A2", "A3", "A4"]
    assert [(s.engine, s.data_mode, s.precision) for s in specs] == [
        ("native_mujoco", "n/a", "float64"),
        ("direct_warp", "no_grad", "float32"),
        ("direct_warp", "diff", "float32"),
        ("bridge_warp", "diff", "float32"),
    ]


def test_grad_data_allocation_is_independent_of_the_step_wrapper():
    """Catches the historical coupling `grad=use_bridge`, which cannot express A3.

    A2 and A3 share the direct wrapper and differ only in Data mode; A3 and A4
    share diff Data and differ only in wrapper. A registry that derived the grad
    flag from the wrapper would give A3 no-grad Data and fail here.
    """
    by_arm = {s.arm: s for s in diag.arm_specs()}
    assert (by_arm["A2"].wrapper, by_arm["A2"].grad_data) == ("direct", False)
    assert (by_arm["A3"].wrapper, by_arm["A3"].grad_data) == ("direct", True)
    assert (by_arm["A4"].wrapper, by_arm["A4"].grad_data) == ("bridge", True)
    assert by_arm["A1"].wrapper is None and by_arm["A1"].grad_data is None
    assert diag.plan_invocation(by_arm["A3"]) == {"wrapper": "direct", "grad_data": True}
    assert diag.plan_invocation(by_arm["A4"]) == {"wrapper": "bridge", "grad_data": True}


def test_unmeasured_design_corner_is_declared():
    """Catches silently measuring or silently forgetting the bridge/no-grad cell."""
    assert diag.unmeasured_design_cells() == (("bridge", "no_grad"),)


def test_duplicate_arm_registration_raises():
    """Catches a duplicated arm silently overwriting or double-counting a cell."""
    specs = diag.arm_specs()
    with pytest.raises(ValueError, match="duplicate"):
        diag.register_arms(specs + (specs[2],))
    with pytest.raises(ValueError, match="duplicate"):
        diag.register_arms(specs[:3] + (specs[1]._replace(arm="A4"),))


def test_registry_requires_all_four_arms():
    """Catches a probe that runs a subset and still reports the four-arm design."""
    with pytest.raises(ValueError, match="four"):
        diag.register_arms(diag.arm_specs()[:3])


# --------------------------------------------------------------------------- #
# Metrics: section 5b state tolerance, reference orientation, worst coordinate
# --------------------------------------------------------------------------- #

def test_identity_comparison_reports_every_metric_exactly_zero():
    """Catches a metric with a spurious offset, or a scaled error that cannot reach 0."""
    values = np.array([[0.0, 1.5, -2.25], [3.0, 4.0, 5.0]])
    result = diag.field_metrics(values, values.copy())
    assert result["max_absolute_error"] == 0.0
    assert result["max_scaled_error"] == 0.0
    assert result["bitwise_equal"] is True
    assert result["passed_state_tolerance"] is True
    assert result["finite"] is True


def test_two_atol_against_a_zero_reference_fails_with_scaled_error_exactly_two():
    """Catches a wrong scaling denominator or a rtol term leaking into a zero reference.

    scaled = |2e-4 - 0| / (1e-4 + 1e-3 * 0) = 2 exactly.
    """
    reference = np.zeros((1, 1))
    actual = np.full((1, 1), 2 * diag.STATE_ATOL)
    result = diag.field_metrics(actual, reference)
    assert result["max_scaled_error"] == 2.0
    assert result["max_absolute_error"] == pytest.approx(2e-4, rel=0, abs=0)
    assert result["passed_state_tolerance"] is False


def test_scaled_metric_uses_the_declared_reference_side():
    """Catches a symmetric denominator; the scaled metric is deliberately asymmetric."""
    small, large = np.array([[1.0]]), np.array([[100.0]])
    forward = diag.field_metrics(large, small)["max_scaled_error"]
    backward = diag.field_metrics(small, large)["max_scaled_error"]
    assert forward == pytest.approx(99.0 / (1e-4 + 1e-3 * 1.0))
    assert backward == pytest.approx(99.0 / (1e-4 + 1e-3 * 100.0))
    assert forward > backward


def test_state_tolerance_and_control_rounding_bound_are_separate(names):
    """Catches reuse of the float32 control epsilon as a state tolerance, or vice versa.

    1e-6 is inside the state atol (1e-4) and far outside the control bound
    (1.19e-07). One gate must pass and the other must fail on the same number.
    """
    assert diag.field_metrics(np.full((1, 1), 1e-6), np.zeros((1, 1)))["passed_state_tolerance"] is True
    assert diag.CONTROL_ATOL == ONE_FLOAT32_EPSILON
    assert diag.CONTROL_RTOL == 0.0
    assert diag.STATE_ATOL == 1e-4 and diag.STATE_RTOL == 1e-3
    native = np.full((4, 26), 0.5)
    warp = {"A2": np.full((4, 26), 0.5 + 1e-6, dtype=np.float32)}
    warp["A3"] = warp["A4"] = warp["A2"]
    gate = diag.control_gate(np.zeros((4, 26)), native, warp)
    assert gate["passed"] is False


def test_worst_coordinate_index_and_name_locate_the_failing_dof(model, names):
    """Catches a flattened/transposed argmax or a name table off by the root offset.

    qvel[11] is knee_angle_r in the frozen official MyoLeg26 (HANDOFF worst case).
    """
    assert names["qvel"][11] == "knee_angle_r"
    reference = np.zeros((4, model.nv))
    actual = np.zeros((4, model.nv))
    actual[2, 11] = 5e-3
    result = diag.field_metrics(actual, reference, names=names["qvel"])
    assert result["worst_scaled_index"] == [2, 11]
    assert result["worst_coordinate_name"] == "knee_angle_r"
    assert result["actual_at_worst_scaled"] == 5e-3
    assert result["reference_at_worst_scaled"] == 0.0


def test_every_state_field_is_reported_separately_and_never_pooled(model, names):
    """Catches pooling qpos/qvel/act/features/reward into one aggregate number."""
    spec = diag.arm_specs()[0]
    sample = sample_for(model)
    a = trajectory(model, spec, sample)
    b = trajectory(model, spec, sample, drift=5e-3)
    result = diag.compare_trajectories(a, b, names)
    assert sorted(result) == sorted(diag.FIELDS)
    assert result["qvel"]["max_absolute_error"] == pytest.approx(5e-3)
    for field in ("qpos", "act", "features", "reward_components"):
        assert result[field]["max_absolute_error"] == 0.0


# --------------------------------------------------------------------------- #
# Section 5a: control / excitation input gate, two clauses
# --------------------------------------------------------------------------- #

def test_clause1_fails_on_any_bitwise_control_difference_across_warp_arms():
    """Catches a clause-1 implementation that uses a tolerance instead of bitwise equality."""
    native = np.full((4, 26), 0.5)
    identical = np.full((4, 26), 0.5, dtype=np.float32)
    controls = {"A2": identical, "A3": identical.copy(), "A4": identical.copy()}
    assert diag.control_gate(np.zeros((4, 26)), native, controls)["clause1_bitwise_across_warp_arms"] is True
    nudged = identical.copy()
    nudged[3, 25] = np.nextafter(np.float32(0.5), np.float32(1.0))
    controls["A4"] = nudged
    gate = diag.control_gate(np.zeros((4, 26)), native, controls)
    assert gate["clause1_bitwise_across_warp_arms"] is False
    assert gate["passed"] is False
    failing = [c for c in gate["warp_pairs"] if not c["bitwise_equal"]]
    assert [(c["actual_arm"], c["reference_arm"]) for c in failing] == [("A4", "A2")]


def test_clause2_passes_at_one_float32_rounding_step():
    """Catches a bound tighter than a real float32 half-ULP, which would fail valid inputs.

    The recorded prior 2.9802322387695312e-08 must sit inside the declared bound.
    """
    actions = np.full((4, 26), ROUNDING_ACTION)
    native = np.full((4, 26), NATIVE_ROUNDING_CTRL)
    warp = np.full((4, 26), WARP_ROUNDING_CTRL, dtype=np.float32)
    gate = diag.control_gate(actions, native, {"A2": warp, "A3": warp.copy(), "A4": warp.copy()})
    assert gate["clause2_native_pairs"][0]["max_absolute_error"] == 2.9802322387695312e-08
    assert gate["clause2_within_rounding_bound"] is True
    assert gate["mapping_equivalent"] is True
    assert gate["canonical_cast_bitwise_equal"] is True
    assert gate["passed"] is True


def test_clause2_fails_at_five_hundredths_even_when_warp_arms_agree():
    """Catches a gate that admits a corrupted control because the Warp arms match.

    Clause 1 passes here by construction, so only clause 2 can reject 0.05.
    """
    actions = np.zeros((4, 26))
    native = np.full((4, 26), 0.5)
    corrupted = np.float32(0.5) + np.float32(0.05)
    warp = np.full((4, 26), corrupted, dtype=np.float32)
    gate = diag.control_gate(actions, native, {"A2": warp, "A3": warp.copy(), "A4": warp.copy()})
    assert gate["clause1_bitwise_across_warp_arms"] is True
    assert gate["clause2_within_rounding_bound"] is False
    assert gate["clause2_native_pairs"][0]["max_absolute_error"] == pytest.approx(0.05, abs=1e-7)
    assert gate["passed"] is False


def test_clause2_requires_a_verified_action_to_excitation_mapping():
    """Catches accepting a numerically close control produced by a different mapping."""
    actions = np.zeros((4, 26))
    wrong_mapping = np.full((4, 26), np.float32(0.5) + np.float32(6e-08), dtype=np.float32)
    gate = diag.control_gate(actions, np.full((4, 26), 0.5),
                             {"A2": wrong_mapping, "A3": wrong_mapping.copy(), "A4": wrong_mapping.copy()})
    assert gate["clause2_within_rounding_bound"] is True
    assert gate["mapping_equivalent"] is False
    assert gate["passed"] is False


def test_canonical_excitation_clips_and_maps_the_declared_action_range():
    """Catches a mapping change: action -1 must be passive, action 0 must be 50% excitation."""
    actions = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    assert diag.canonical_excitation(actions, np.float64).tolist() == [[0.0, 0.0, 0.5, 1.0, 1.0]]
    assert diag.canonical_excitation(actions, np.float32).dtype == np.float32


# --------------------------------------------------------------------------- #
# Warning provenance: measured vs not queried
# --------------------------------------------------------------------------- #

def test_warning_record_marks_warp_counts_not_queried_and_native_measured():
    """Catches a fabricated Warp warning_count=0 presented as a measured value.

    No warning channel is queried on the Warp side, so the only truthful value is
    an explicit absence. The native count IS measured and must stay distinct.
    """
    assert diag.warning_record("native_mujoco", 3) == {"warning_count": 3,
                                                       "warning_status": "measured"}
    assert diag.warning_record("native_mujoco", 0) == {"warning_count": 0,
                                                       "warning_status": "measured"}
    for engine in ("direct_warp", "bridge_warp"):
        assert diag.warning_record(engine) == {"warning_count": None,
                                              "warning_status": "not_queried"}


def test_warp_arm_claiming_a_measured_warning_status_fails_integrity(tmp_path, model, names):
    """Catches a Warp arm asserting a warning count it never queried."""
    sample = sample_for(model)

    def rollout(spec, repetition):
        rows = trajectory(model, spec, sample)
        if spec.arm == "A3":
            rows["warning_count"] = 0
            rows["warning_status"] = "measured"
        return rows

    code, report = run(tmp_path, model, names, rollout=rollout)
    assert code == diag.EXIT_INTEGRITY
    assert report["error"]["reason"] == "warning_status_unsupported"
    assert report["error"]["coordinates"]["arm"] == "A3"


def test_artifact_never_stores_a_zero_warning_count_for_a_warp_arm(tmp_path, model, names):
    """Catches an unmeasured 0 being persisted into the provenance-bearing artifact."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    stored = {t["arm"]: (t["warning_count"], t["warning_status"]) for t in report["trajectories"]}
    for arm in ("A2", "A3", "A4"):
        assert stored[arm] == (None, "not_queried")
    assert stored["A1"] == (0, "measured")
    semantics = report["warning_count_semantics"]
    assert sorted(semantics["not_queried_for"]) == ["bridge_warp", "direct_warp"]
    assert semantics["measured_for"] == ["native_mujoco"]


# --------------------------------------------------------------------------- #
# Section 5c: preregistered reference orientation, exposed in JSON
# --------------------------------------------------------------------------- #

def test_declared_pair_set_is_exactly_the_preregistered_five():
    """Catches an added post-hoc pair, e.g. the historical A4-vs-A2 comparison."""
    assert diag.declared_pairs() == (
        ("A2", "A1"), ("A3", "A1"), ("A4", "A1"), ("A3", "A2"), ("A4", "A3"))


def test_every_emitted_comparison_exposes_actual_and_reference_as_json_fields(tmp_path, model, names):
    """Catches reference identity left implicit in a key name, per section 5c."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    emitted = [(c["actual_arm"], c["reference_arm"]) for c in report["comparisons"]
               if c["repetition"] == 0]
    assert emitted == list(diag.declared_pairs())
    for entry in report["comparisons"]:
        assert set(entry) >= {"actual_arm", "reference_arm", "repetition", "fields"}
        assert entry["reference_arm"] in {"A1", "A2", "A3"}
    assert report["reference_orientation"]["within_arm_repeats"] == "repetition 0"


def test_within_arm_repeats_are_reported_separately_against_repetition_zero(tmp_path, model, names):
    """Catches repeat spread pooled into, or subtracted from, between-arm error."""
    sample = sample_for(model)
    drifts = {("A4", 1): 7.743998430669308e-04}
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample, drifts=drifts))
    assert code == 0
    repeats = {(r["arm"], r["repetition"]): r for r in report["repeat_comparisons"]}
    assert sorted(repeats) == [(a, r) for a in ("A1", "A2", "A3", "A4") for r in (1, 2)]
    assert repeats[("A4", 1)]["reference_repetition"] == 0
    assert repeats[("A4", 1)]["fields"]["qvel"]["max_absolute_error"] == pytest.approx(7.743998430669308e-04)
    assert repeats[("A4", 2)]["fields"]["qvel"]["max_absolute_error"] == 0.0
    assert repeats[("A2", 1)]["fields"]["qvel"]["max_absolute_error"] == 0.0
    between = [c for c in report["comparisons"] if (c["actual_arm"], c["reference_arm"]) == ("A4", "A3")]
    assert between[1]["fields"]["qvel"]["max_absolute_error"] == pytest.approx(7.743998430669308e-04)
    assert between[0]["fields"]["qvel"]["max_absolute_error"] == 0.0


# --------------------------------------------------------------------------- #
# Section 5d: status, exit codes, durable coordinates
# --------------------------------------------------------------------------- #

def test_complete_measurement_records_exactly_twelve_trajectories_and_exits_zero(tmp_path, model, names):
    """Catches a promoted status with a missing arm or repetition."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    assert report["status"] == "measurement_complete"
    assert report["completeness"]["trajectory_count"] == 12
    assert report["completeness"]["expected_trajectory_count"] == 12
    assert sorted((t["arm"], t["repetition"]) for t in report["trajectories"]) == [
        (a, r) for a in ("A1", "A2", "A3", "A4") for r in (0, 1, 2)]
    assert report["error"] is None


def test_forward_parity_failure_still_exits_zero_with_measurement_complete(tmp_path, model, names):
    """Catches reporting a genuine physics finding as a run failure (section 5d)."""
    sample = sample_for(model)
    drifts = {"A4": 1.0227001164902216e-03}
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample, drifts=drifts))
    assert code == 0
    assert report["status"] == "measurement_complete"
    assert report["state_parity"]["passed"] is False
    failing = [c for c in report["comparisons"]
               if not c["fields"]["qvel"]["passed_state_tolerance"]]
    assert failing, "the injected 1.02e-03 qvel difference must fail atol=1e-4"
    assert failing[0]["fields"]["qvel"]["max_scaled_error"] > 1.0


def test_status_is_running_before_any_arm_executes(tmp_path, model, names):
    """Catches a report that never records the running state, losing crash evidence."""
    observed = []
    sample = sample_for(model)
    out = tmp_path / "probe.json"

    def rollout(spec, repetition):
        observed.append(json.loads(out.read_text(encoding="utf-8"))["status"])
        return trajectory(model, spec, sample)

    code = diag.execute(out, inputs_factory=lambda: inputs_for(model, names),
                        rollout_factory=lambda _i: rollout, printer=lambda *a, **k: None)
    assert code == 0
    assert observed[0] == "running"
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "measurement_complete"


def test_qualification_is_never_asserted_and_carries_a_stated_reason(tmp_path, model, names):
    """Catches a hardcoded verdict field of the kind the old replay driver wrote."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    assert report["qualification_status"] == "not_assessed"
    assert report["qualification_reason"]
    assert "qualification_passed" not in report
    assert all("passed" != key for key in report)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_value_fails_closed_with_the_coded_exit_and_coordinates(tmp_path, model, names, bad):
    """Catches a nonfinite value reaching serialisation, or losing its coordinates."""
    sample = sample_for(model)

    def rollout(spec, repetition):
        rows = trajectory(model, spec, sample)
        if spec.arm == "A3" and repetition == 1:
            rows["qvel"][2, 11] = bad
        return rows

    code, report = run(tmp_path, model, names, rollout=rollout)
    assert code == diag.EXIT_NONFINITE
    assert code != 0
    assert report["status"] == "measurement_incomplete"
    assert report["error"]["reason"] == "nonfinite_value"
    assert report["error"]["coordinates"] == {"arm": "A3", "repetition": 1, "field": "qvel", "index": [2, 11]}
    assert json.dumps(report, allow_nan=False)


def test_wrong_field_shape_fails_the_integrity_check(tmp_path, model, names):
    """Catches a truncated or mis-widened field silently compared against a full one."""
    sample = sample_for(model)

    def rollout(spec, repetition):
        rows = trajectory(model, spec, sample)
        if spec.arm == "A2":
            rows["qpos"] = rows["qpos"][:, :-1]
        return rows

    code, report = run(tmp_path, model, names, rollout=rollout)
    assert code == diag.EXIT_INTEGRITY
    assert report["status"] == "measurement_incomplete"
    assert report["error"]["reason"] == "wrong_shape"
    assert report["error"]["coordinates"]["arm"] == "A2"
    assert report["error"]["coordinates"]["field"] == "qpos"


def test_wrong_input_hash_fails_the_integrity_check_before_any_rollout(tmp_path, model, names):
    """Catches measuring against a dataset/report/model other than the frozen one."""
    called = []
    sample = sample_for(model)
    hashes = dict(diag.EXPECTED_HASHES)
    hashes["dataset"] = "0" * 64

    def rollout(spec, repetition):
        called.append(spec.arm)
        return trajectory(model, spec, sample)

    code, report = run(tmp_path, model, names, rollout=rollout, hashes=hashes)
    assert code == diag.EXIT_INTEGRITY
    assert report["error"]["reason"] == "hash_mismatch"
    assert report["error"]["coordinates"]["field"] == "dataset"
    assert called == []


def test_wrong_model_dimensions_fail_the_integrity_check(tmp_path, model, names):
    """Catches running the probe against a model that is not official MyoLeg26."""
    other = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint type='hinge' axis='0 0 1'/>"
        "<geom size='.1'/></body></worldbody></mujoco>")
    code, report = run(tmp_path, model, names, rollout=lambda s, r: None, model=other)
    assert code == diag.EXIT_INTEGRITY
    assert report["error"]["reason"] == "model_dimensions"


def test_arm_reporting_a_data_mode_other_than_its_own_fails_integrity(tmp_path, model, names):
    """Catches the grad/wrapper coupling at measurement time, not only in the registry.

    A3 must allocate diff Data. If the rollout actually allocated no-grad Data,
    the arm is not the preregistered A3 and the measurement must not complete.
    """
    sample = sample_for(model)
    code, report = run(tmp_path, model, names,
                       rollout=rollout_from(model, sample, data_modes={"A3": "no_grad"}))
    assert code == diag.EXIT_INTEGRITY
    assert report["error"]["reason"] == "data_mode_mismatch"
    assert report["error"]["coordinates"]["arm"] == "A3"


def test_missing_trajectory_yields_measurement_incomplete_with_the_coded_exit(tmp_path, model, names):
    """Catches a promoted status when an arm silently produced nothing."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names,
                       rollout=rollout_from(model, sample, omit={("A4", 2)}))
    assert code == diag.EXIT_INCOMPLETE
    assert report["status"] == "measurement_incomplete"
    assert report["error"]["reason"] == "missing_trajectory"
    assert report["error"]["coordinates"]["arm"] == "A4"
    assert report["error"]["coordinates"]["repetition"] == 2


def test_malformed_trajectory_yields_measurement_incomplete_with_the_coded_exit(tmp_path, model, names):
    """Catches a payload missing a required field being compared as if complete."""
    sample = sample_for(model)

    def rollout(spec, repetition):
        rows = trajectory(model, spec, sample)
        if spec.arm == "A2" and repetition == 0:
            rows.pop("features")
        return rows

    code, report = run(tmp_path, model, names, rollout=rollout)
    assert code == diag.EXIT_INCOMPLETE
    assert report["status"] == "measurement_incomplete"
    assert report["error"]["reason"] == "missing_field"
    assert report["error"]["coordinates"] == {"arm": "A2", "repetition": 0,
                                              "field": "features", "index": None}


def test_error_payloads_are_json_safe_with_allow_nan_false(tmp_path, model, names):
    """Catches numpy scalars or NaN in the durable error coordinates.

    `json.dumps(..., allow_nan=False)` on the written artifact is the real check:
    numpy int64 raises TypeError and NaN raises ValueError.
    """
    sample = sample_for(model)

    def rollout(spec, repetition):
        rows = trajectory(model, spec, sample)
        if spec.arm == "A4" and repetition == 2:
            rows["features"][1, 7] = np.float32("nan")
        return rows

    out = tmp_path / "probe.json"
    code = diag.execute(out, inputs_factory=lambda: inputs_for(model, names),
                        rollout_factory=lambda _i: rollout, printer=lambda *a, **k: None)
    assert code == diag.EXIT_NONFINITE
    raw = out.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    payload = json.loads(raw)
    assert json.dumps(payload, allow_nan=False)
    coordinates = payload["error"]["coordinates"]
    assert coordinates == {"arm": "A4", "repetition": 2, "field": "features", "index": [1, 7]}
    assert all(isinstance(v, (str, int, list, type(None))) for v in coordinates.values())


def _inject_metric(monkeypatch, key, value):
    """Force one metric field to `value`, leaving every other metric real.

    The secondary (comparison-level) nonfinite gate is unreachable through
    trajectory inputs, because `validate_trajectory` rejects a nonfinite
    trajectory before any comparison exists and the scaled denominator is
    strictly positive. Injecting one metric is the only way to enter that
    branch, and everything downstream of it stays production code.
    """
    real = diag.field_metrics

    def patched(actual, reference, **kwargs):
        metrics = real(actual, reference, **kwargs)
        metrics[key] = value
        return metrics

    monkeypatch.setattr(diag, "field_metrics", patched)


def test_nonfinite_comparison_metric_yields_the_coded_exit_and_a_json_safe_report(
        tmp_path, model, names, monkeypatch):
    """Catches the reviewer's D1 escape: the failure report must not retain inf.

    Before the fix, `assess_completeness` raised after `report["comparisons"]`
    already held the nonfinite metric, so the failure handler's
    `json.dumps(..., allow_nan=False)` raised ValueError out of `execute`: exit 1
    with a traceback, artifact stranded at `status: "running"`, `error: null`, and
    none of the durable coordinates section 5d mandates.
    """
    _inject_metric(monkeypatch, "max_scaled_error", float("inf"))
    sample = sample_for(model)
    out = tmp_path / "probe.json"
    code = diag.execute(out, inputs_factory=lambda: inputs_for(model, names),
                        rollout_factory=lambda _i: rollout_from(model, sample),
                        printer=lambda *a, **k: None)
    assert code == diag.EXIT_NONFINITE
    assert code != 0
    raw = out.read_text(encoding="utf-8")
    assert "Infinity" not in raw and "NaN" not in raw
    report = json.loads(raw)
    assert json.dumps(report, allow_nan=False)
    assert report["status"] == "measurement_incomplete"
    assert report["error"]["reason"] == "nonfinite_comparison"
    assert report["error"]["exit_code"] == diag.EXIT_NONFINITE
    assert report["error"]["coordinates"]["field"] in diag.FIELDS
    assert report["error"]["coordinates"]["arm"] in {"A1", "A2", "A3", "A4"}


def test_unserialisable_payload_never_reports_a_complete_measurement(
        tmp_path, model, names, monkeypatch):
    """Catches an invalid metric being promoted to a successful measurement.

    `max_absolute_error` is deliberately NOT inspected by the completeness gate,
    so injecting inf there passes every gate and only breaks serialisation. The
    run must still refuse to report `measurement_complete` or exit 0.
    """
    _inject_metric(monkeypatch, "max_absolute_error", float("inf"))
    sample = sample_for(model)
    out = tmp_path / "probe.json"
    code = diag.execute(out, inputs_factory=lambda: inputs_for(model, names),
                        rollout_factory=lambda _i: rollout_from(model, sample),
                        printer=lambda *a, **k: None)
    assert code != 0
    assert code == diag.EXIT_NONFINITE
    raw = out.read_text(encoding="utf-8")
    assert "Infinity" not in raw
    report = json.loads(raw)
    assert json.dumps(report, allow_nan=False)
    assert report["status"] == "measurement_incomplete"
    assert report["error"]["reason"] == "serialisation_fault"
    assert report["error"]["exit_code"] == diag.EXIT_NONFINITE


def test_degraded_failure_report_keeps_identity_and_names_what_it_dropped(
        tmp_path, model, names, monkeypatch):
    """Catches a degraded report that silently loses provenance or hides the loss."""
    _inject_metric(monkeypatch, "max_scaled_error", float("-inf"))
    sample = sample_for(model)
    out = tmp_path / "probe.json"
    diag.execute(out, inputs_factory=lambda: inputs_for(model, names),
                 rollout_factory=lambda _i: rollout_from(model, sample),
                 printer=lambda *a, **k: None)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == diag.SCHEMA_VERSION
    assert report["sample_id"] == diag.SAMPLE_ID
    assert report["qualification_status"] == "not_assessed"
    assert report["output"]["mode"] == "exclusive_create"
    assert report["serialisation_fault"]["dropped_sections"]
    assert "comparisons" in report["serialisation_fault"]["dropped_sections"]
    assert set(report["provenance"]["driver_sources"]) == set(diag.driver_source_hashes())


def test_printing_fault_cannot_overturn_a_recorded_physics_result(tmp_path, model, names):
    """Catches a post-save print failure relabelling a complete measurement."""
    sample = sample_for(model)

    def printer(*args, **kwargs):
        raise OSError("stdout closed")

    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample), printer=printer)
    assert code == diag.EXIT_REPORTING_FAULT
    assert code != 0
    assert report["status"] == "measurement_complete"
    assert report["completeness"]["trajectory_count"] == 12
    assert report["reporting_fault"]["reason"] == "final_print_failed"


# --------------------------------------------------------------------------- #
# Section 5g: pre-existing output sentinel
# --------------------------------------------------------------------------- #

def test_pre_existing_output_is_refused_before_any_gpu_allocation(tmp_path, model, names):
    """Catches a driver that truncates or overwrites a preserved artifact.

    The failure must happen before the inputs are loaded and before the rollout
    factory (the only GPU allocation site) is touched.
    """
    out = tmp_path / "preserved.json"
    original = '{"status": "measurement_complete", "keep": true}\n'
    out.write_text(original, encoding="utf-8")
    before = out.stat().st_size

    def forbidden_inputs():
        raise AssertionError("inputs must not be loaded when the output already exists")

    def forbidden_rollout(_inputs):
        raise AssertionError("no GPU allocation may happen when the output already exists")

    code = diag.execute(out, inputs_factory=forbidden_inputs,
                        rollout_factory=forbidden_rollout, printer=lambda *a, **k: None)
    assert code == diag.EXIT_OUTPUT_EXISTS
    assert code != 0
    assert out.read_text(encoding="utf-8") == original
    assert out.stat().st_size == before


def test_fresh_output_path_is_opened_exclusively(tmp_path, model, names):
    """Catches an implementation that opens with 'w' and would clobber on a rerun."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    assert report["output"]["mode"] == "exclusive_create"
    code2, _ = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code2 == diag.EXIT_OUTPUT_EXISTS


# --------------------------------------------------------------------------- #
# Section 5e: provenance
# --------------------------------------------------------------------------- #

def test_provenance_hashes_the_driver_its_helpers_and_this_test_module():
    """Catches an unauditable artifact: the reused helper does not hash its caller."""
    hashes = diag.driver_source_hashes()
    expected = {
        "scripts/diag_myoleg26_forward_data_mode.py",
        "scripts/check_myoleg26_derivatives.py",
        "scripts/check_trajopt_gradients.py",
        "tests/unit/test_myoleg26_forward_data_mode.py",
    }
    assert set(hashes) == expected
    for relative, recorded in hashes.items():
        assert recorded == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_report_carries_schema_version_and_the_reused_runtime_provenance(tmp_path, model, names):
    """Catches an artifact that cannot be bound to the pre-U2 tree."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    assert report["schema_version"] == diag.SCHEMA_VERSION
    assert report["provenance"]["runtime"] == {"packages": {}}
    assert set(report["provenance"]["driver_sources"]) == set(diag.driver_source_hashes())


def test_report_states_input_precision_and_the_interpretation_limits(tmp_path, model, names):
    """Catches dropping the mandated no-cause / no-absence / cross-implementation labels."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    assert "float32" in report["input_precision"] and "float64" in report["input_precision"]
    limits = " ".join(report["interpretation_limits"]).lower()
    for phrase in ("three repetitions", "absence", "caus", "cross-implementation", "noise floor",
                   # D4/D7: confounds and sensitivities that must travel with the numbers
                   "arm-major", "write-back", "columns 52:98", "reward_components",
                   "allocation pattern", "bitwise identical"):
        assert phrase in limits
    assert report["comparison_values_not_acceptance_bands"]["bridge_native_qvel_absolute_error"] == \
        0.0010227001164902216


def test_control_input_pairs_are_labelled_so_a4_vs_a2_is_not_read_as_a_state_pair(
        tmp_path, model, names):
    """Catches an auditor finding A4-vs-A2 in the artifact and reading section 5c as violated.

    A4-vs-A2 is a legitimate control-INPUT pair under section 5a clause 1, and is
    not among the five declared state pairs. The artifact must say which it is.
    """
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    pairs = report["control_gate_result"]["per_repetition"][0]["warp_pairs"]
    assert [(p["actual_arm"], p["reference_arm"]) for p in pairs] == [("A3", "A2"), ("A4", "A2")]
    assert {p["pair_kind"] for p in pairs} == {"control_input_clause1"}
    state_pairs = {(c["actual_arm"], c["reference_arm"]) for c in report["comparisons"]}
    assert ("A4", "A2") not in state_pairs


def test_state_tolerance_recorded_in_the_artifact_is_the_preserved_original(tmp_path, model, names):
    """Catches a silently relaxed preregistered tolerance."""
    sample = sample_for(model)
    code, report = run(tmp_path, model, names, rollout=rollout_from(model, sample))
    assert code == 0
    assert report["state_tolerance"] == {"atol": 1e-4, "rtol": 1e-3}
    assert report["control_input_gate"]["atol_ctrl"] == ONE_FLOAT32_EPSILON
    assert report["control_input_gate"]["rtol_ctrl"] == 0.0


# --------------------------------------------------------------------------- #
# The A1 reference arm: native_trajectory (reference for three of five pairs)
# --------------------------------------------------------------------------- #

def native_sample(model, action_value, horizon=1):
    return {"qpos": model.key_qpos[0].astype(np.float32).astype(np.float64),
            "qvel": np.zeros(model.nv),
            "act": np.full(model.na, .2, dtype=np.float32).astype(np.float64),
            "action": np.full((horizon, model.nu), action_value, dtype=np.float64)}


@pytest.mark.parametrize("action_value,expected_excitation", [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)])
def test_native_trajectory_applies_the_declared_excitation_anchors(
        model, action_value, expected_excitation):
    """Catches a changed or mis-scaled mapping in the control the native arm ACTUALLY applies.

    Anchors are hand-derived from the frozen contract, not from
    `canonical_excitation`: action -1 is passive, action 0 is 50% excitation,
    action +1 is full. This is the independent half of the section 5a check that
    the reviewer's D5 found tautological on the native side.
    """
    rows = diag.native_trajectory(model, native_sample(model, action_value))
    assert rows["ctrl"].shape == (1, model.nu)
    assert rows["ctrl"].dtype == np.float64
    assert np.array_equal(rows["ctrl"], np.full((1, model.nu), expected_excitation))


def test_native_trajectory_records_state_after_all_four_substeps(model):
    """Catches a wrong substep count or state captured before the substep loop.

    The reference is an independent MjData stepped four times in this test, not a
    value produced by the driver.
    """
    sample = native_sample(model, 0.0)
    rows = diag.native_trajectory(model, sample)

    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:] = sample["qpos"], sample["qvel"], sample["act"]
    data.ctrl[:] = 0.5
    for _ in range(4):
        mujoco.mj_step(model, data)
    assert np.array_equal(rows["qpos"][0], data.qpos)
    assert np.array_equal(rows["qvel"][0], data.qvel)
    assert np.array_equal(rows["act"][0], data.act)

    one = mujoco.MjData(model)
    one.qpos[:], one.qvel[:], one.act[:] = sample["qpos"], sample["qvel"], sample["act"]
    one.ctrl[:] = 0.5
    mujoco.mj_step(model, one)
    assert not np.array_equal(rows["qvel"][0], one.qvel), "four substeps must not equal one"


def test_native_trajectory_reward_component_order_is_net_then_locomotion(model):
    """Catches a swapped reward component order, which would silently transpose A1.

    effort = dt * effort_weight * mean(excitation**2), hand-computed here:
    (4 * 0.002) * 0.01 * 0.5**2.
    """
    rows = diag.native_trajectory(model, native_sample(model, 0.0))
    net, locomotion = rows["reward_components"][0]
    expected_effort = (4 * 0.002) * 0.01 * 0.5 ** 2
    assert locomotion - net == pytest.approx(expected_effort, rel=1e-9)
    assert locomotion > net, "the net term must carry the effort cost, the second must not"


def test_native_trajectory_has_the_frozen_field_widths_and_step_count(model):
    """Catches a dropped field or a trajectory that does not span the whole window."""
    rows = diag.native_trajectory(model, native_sample(model, -0.4, horizon=3))
    assert set(rows) >= set(diag.REQUIRED_KEYS)
    for field, width in ((("qpos", model.nq)), ("qvel", model.nv), ("act", model.na),
                         ("features", diag.STATE_SIZE), ("reward_components", 2),
                         ("ctrl", model.nu)):
        assert np.asarray(rows[field]).shape == (3, width), field
    assert rows["observed_data_mode"] == "n/a"
    assert rows["warning_status"] == "measured"
    assert isinstance(rows["warning_count"], int)


def test_probe_mapping_matches_the_production_excitation_function():
    """Catches the probe's float32 mapping drifting from production's.

    `canonical_excitation` re-implements `excitation_from_action`; if the two ever
    diverge, the section 5a mapping-equivalence clause would verify the wrong
    mapping. Compared bitwise against the real production function.
    """
    import torch  # CPU only; kept out of module scope so collection stays torch-free
    from msk_warp.envs.myoleg26_task import excitation_from_action
    actions = np.array([[-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0]], dtype=np.float32)
    production = excitation_from_action(torch.from_numpy(actions)).numpy()
    assert np.array_equal(diag.canonical_excitation(actions, np.float32), production)


# --------------------------------------------------------------------------- #
# The input path: load_inputs, on tracked fixtures only
# --------------------------------------------------------------------------- #

def write_synthetic_dataset(path, model, *, sample_id, horizon_available=6, count=2):
    """A synthetic myoleg26-visited-v1 archive built from the TRACKED model asset.

    Deliberately not the preserved ignored `logs/myoleg26_visited_v1.npz`: a unit
    test must not require ignored evidence to exist.
    """
    meta = {"schema_version": "myoleg26-visited-v1", "held_out": True,
            "model_path": "msk_warp/assets/myoleg26/flat_boxes.xml", "substeps": 4,
            "compiled_model_sha256": diag.check.compiled_model_sha256(model),
            "task_contract": dict(diag.check.TASK_CONTRACT), "primary_strata": ["double_support"],
            "samples": [{"sample_id": (sample_id if i == 0 else f"other{i}"), "seed": i,
                         "episode_id": i, "checkpoint_sha256": f"seed{i}",
                         "contact_stratum": "double_support"} for i in range(count)]}
    rows = {"qpos": model.key_qpos[0].astype(np.float32),
            "qvel": np.zeros(model.nv, dtype=np.float32),
            "act": np.full(model.na, .2, dtype=np.float32),
            "previous_action": np.full(model.nu, -.5, dtype=np.float32)}
    arrays = {k: np.repeat(v[None], count, axis=0) for k, v in rows.items()}
    window = np.linspace(-.9, -.1, horizon_available * model.nu, dtype=np.float32)
    arrays["action"] = np.repeat(window.reshape(1, horizon_available, model.nu), count, axis=0)
    np.savez(path, **arrays, metadata_json=json.dumps(meta))
    return arrays


def synthetic_args(tmp_path, model, *, sample_id=None):
    states = tmp_path / "states.npz"
    arrays = write_synthetic_dataset(states, model, sample_id=sample_id or diag.SAMPLE_ID)
    original = tmp_path / "original.json"
    original.write_text('{"gate_status": "forward_mismatch"}', encoding="utf-8")
    return SimpleNamespace(states=str(states), original_report=str(original)), arrays


def test_load_inputs_slices_the_action_window_to_the_preregistered_horizon(tmp_path, model):
    """Catches a missing or wrong `[:HORIZON]` slice feeding a longer window to every arm."""
    args, arrays = synthetic_args(tmp_path, model)
    inputs = diag.load_inputs(args)
    assert np.asarray(inputs.sample["action"]).shape == (diag.HORIZON, model.nu)
    assert np.array_equal(inputs.sample["action"],
                          arrays["action"][0][:diag.HORIZON].astype(np.float64))
    assert inputs.horizon == diag.HORIZON and inputs.repetitions == diag.REPETITIONS
    assert inputs.metadata["sample_id"] == diag.SAMPLE_ID
    assert np.asarray(inputs.sample["qpos"]).shape == (model.nq,)


def test_load_inputs_assembles_the_three_pinned_hashes_from_real_bytes(tmp_path, model):
    """Catches a hash read from metadata instead of the file, or a missing hash key."""
    args, _ = synthetic_args(tmp_path, model)
    inputs = diag.load_inputs(args)
    assert set(inputs.hashes) == set(diag.EXPECTED_HASHES)
    assert inputs.hashes["dataset"] == hashlib.sha256(Path(args.states).read_bytes()).hexdigest()
    assert inputs.hashes["original_report"] == hashlib.sha256(
        Path(args.original_report).read_bytes()).hexdigest()
    assert inputs.hashes["compiled_model"] == diag.check.compiled_model_sha256(model)


def test_load_inputs_refuses_a_dataset_without_the_preregistered_sample(tmp_path, model):
    """Catches silently measuring whatever sample happens to be first in the archive."""
    args, _ = synthetic_args(tmp_path, model, sample_id="some_other_state")
    with pytest.raises(diag.IntegrityFailure) as raised:
        diag.load_inputs(args)
    assert raised.value.reason == "sample_identity"
    assert raised.value.code == diag.EXIT_INTEGRITY


def test_a_non_frozen_dataset_is_rejected_by_the_pinned_hash_gate(tmp_path, model):
    """Catches a probe that would measure against any dataset carrying the sample id.

    End-to-end through `execute`: `load_inputs` accepts the synthetic archive, and
    the pinned-hash gate is what refuses it.
    """
    args, _ = synthetic_args(tmp_path, model)
    out = tmp_path / "probe.json"
    code = diag.execute(out, inputs_factory=lambda: diag.load_inputs(args),
                        rollout_factory=lambda _i: (lambda spec, rep: None),
                        printer=lambda *a, **k: None)
    assert code == diag.EXIT_INTEGRITY
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["error"]["reason"] == "hash_mismatch"
    assert report["error"]["coordinates"]["field"] == "dataset"


# --------------------------------------------------------------------------- #
# Substep cross-check (D6)
# --------------------------------------------------------------------------- #

def test_environment_substeps_must_equal_the_probes_substep_count():
    """Catches a silent confound on the very quantity the probe compares.

    The direct arm steps the module constant `SUBSTEPS`; the bridge arm steps
    `env.substeps` inside `WarpSimStep`. If they ever differ, A3-vs-A4 would
    compare different amounts of physics.
    """
    assert diag.validate_environment(SimpleNamespace(substeps=diag.SUBSTEPS)) == diag.SUBSTEPS
    for bad in (1, 2, 8):
        with pytest.raises(diag.IntegrityFailure) as raised:
            diag.validate_environment(SimpleNamespace(substeps=bad))
        assert raised.value.reason == "substeps_mismatch"
        assert raised.value.code == diag.EXIT_INTEGRITY
    with pytest.raises(diag.IntegrityFailure):
        diag.validate_environment(SimpleNamespace())


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #

def test_cli_requires_an_output_path_and_exposes_no_overwrite_switch():
    """Catches an --overwrite escape hatch past the exclusive-output requirement."""
    parser = diag.build_parser()
    actions = {a.dest for a in parser._actions}
    assert "out" in actions
    assert "overwrite" not in actions
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--out", "x.json"])
    assert args.device == "cuda:0"
