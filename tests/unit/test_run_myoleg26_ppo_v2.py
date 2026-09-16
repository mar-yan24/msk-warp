"""Unit tests for the segmented MyoLeg26 v2 PPO campaign runner.

CPU only. Fake clocks, fake child processes, fake ledgers and the Task 3 analytic
CPU adapter. **No simulator, no CUDA, no Warp, no real subprocess, no sleeping on
a wall clock and no training.** Nothing here consumes the 28,800 s training
budget or the 1,800 s diagnostic budget; test-suite time is separate accounting.

The runner is machinery only: none of these tests qualifies a physics result, a
gradient, a behaviour claim or an acceleration claim.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from msk_warp.analysis import myoleg26_selection_v2 as S
from msk_warp.analysis import ppo_v2_budget as B
from msk_warp.analysis import ppo_v2_protocol as P
from scripts import run_myoleg26_ppo_v2 as R


# ==========================================================================
# Shared fixtures and helpers
# ==========================================================================

def _freeze(tmp_path) -> Path:
    """A minimal frozen-manifest stand-in with the fields the runner binds."""
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({
        "schema_version": "myoleg26-baseline-freeze-v1",
        "model_sha256": "a" * 64,
        "compiled_model_sha256": "b" * 64,
        "files": {"msk_warp/algorithms/ppo.py": "c" * 64},
    }, sort_keys=True), encoding="utf-8")
    return path


def _launch_argv(tmp_path, **overrides):
    values = {
        "--stage": "screen",
        "--recipe": "g990_e010",
        "--seed": "1001",
        "--segment-index": "0",
        "--freeze": str(_freeze(tmp_path)),
        "--ledger": str(tmp_path / "budget_ledger.jsonl"),
        "--run-dir": str(tmp_path / "run"),
    }
    # ``None`` means "do not supply this flag at all", which is how the negative
    # controls drop a required argument.
    for key, value in overrides.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = str(value)
    argv = ["launch"]
    for key, value in values.items():
        argv += [key, value]
    return argv


def _parse(argv):
    return R.build_parser().parse_args(argv)


# ==========================================================================
# Unit 1 -- CLI surface and launch-plan validation
# ==========================================================================

@pytest.mark.parametrize("mode", ["launch", "worker", "select"])
def test_each_mode_has_help_and_exits_zero(mode, capsys):
    """Behavioural. Every declared mode is reachable and self-describing."""
    with pytest.raises(SystemExit) as raised:
        R.build_parser().parse_args([mode, "--help"])
    assert raised.value.code == 0
    assert mode in capsys.readouterr().out


def test_a_missing_mode_is_refused_rather_than_defaulted():
    with pytest.raises(SystemExit) as raised:
        R.build_parser().parse_args([])
    assert raised.value.code != 0


def test_plan_derives_its_epoch_span_from_the_sealed_segment_rounds(tmp_path):
    """Behavioural. The sealed grid is authoritative; the CLI does not invent one."""
    plan = R.resolve_plan(_parse(_launch_argv(tmp_path)))
    first = P.segment_rounds(P.stage("screen").epoch_cap_per_seed)[0]
    assert (plan.start_epoch, plan.end_epoch) == (first.start_epoch, first.end_epoch)
    assert plan.epochs == first.epochs <= P.SEGMENT_MAX_EPOCHS


def test_plan_places_output_under_the_run_directory_by_segment_index(tmp_path):
    args = _parse(_launch_argv(tmp_path, **{"--segment-index": "1",
                                            "--parent-segment": str(tmp_path / "p.ptc"),
                                            "--parent-result": str(tmp_path / "p.json")}))
    plan = R.resolve_plan(args)
    assert plan.out_dir == Path(args.run_dir) / "segment_0001"
    assert plan.out_dir.parent == Path(args.run_dir)


@pytest.mark.parametrize("recipe", ["g990_e010", "g998_e010", "g990_e000", "g998_e000"])
def test_every_sealed_recipe_is_accepted(recipe, tmp_path):
    plan = R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--recipe": recipe})))
    assert plan.recipe == recipe


def test_an_unsealed_recipe_is_refused(tmp_path):
    with pytest.raises(R.RunnerRefusal, match="recipe"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--recipe": "g950_e050"})))


def test_a_seed_outside_the_stage_schedule_is_refused(tmp_path):
    """Behavioural. Per-seed caps are keyed (stage, recipe, seed); a stray seed
    would silently open a fresh unbudgeted allowance."""
    with pytest.raises(R.RunnerRefusal, match="seed"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--seed": "9999"})))


def test_a_segment_index_past_the_stage_epoch_cap_is_refused(tmp_path):
    rounds = P.segment_rounds(P.stage("screen").epoch_cap_per_seed)
    with pytest.raises(R.RunnerRefusal, match="segment"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{
            "--segment-index": str(len(rounds)),
            "--parent-segment": str(tmp_path / "p.ptc"),
            "--parent-result": str(tmp_path / "p.json")})))


def test_a_resume_segment_requires_both_parent_files(tmp_path):
    for missing in ("--parent-segment", "--parent-result"):
        supplied = {"--segment-index": "1", "--parent-segment": str(tmp_path / "p.ptc"),
                    "--parent-result": str(tmp_path / "p.json")}
        supplied[missing] = None
        with pytest.raises(R.RunnerRefusal, match="parent"):
            R.resolve_plan(_parse(_launch_argv(tmp_path, **supplied)))


def test_segment_zero_refuses_a_parent_rather_than_ignoring_it(tmp_path):
    """Behavioural. Silently dropping a supplied parent would start from scratch
    while the caller believed it was resuming."""
    with pytest.raises(R.RunnerRefusal, match="parent"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{
            "--segment-index": "0", "--parent-segment": str(tmp_path / "p.ptc"),
            "--parent-result": str(tmp_path / "p.json")})))


@pytest.mark.parametrize("epochs", ["0", "-1", "65"])
def test_an_epoch_budget_outside_the_segment_ceiling_is_refused(epochs, tmp_path):
    with pytest.raises(R.RunnerRefusal, match="epoch"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--epochs": epochs})))


def test_a_shorter_epoch_budget_is_allowed_and_shortens_the_span(tmp_path):
    plan = R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--epochs": "8"})))
    assert plan.epochs == 8
    assert plan.end_epoch == plan.start_epoch + 8


def test_a_work_deadline_above_the_sealed_ceiling_is_refused(tmp_path):
    with pytest.raises(R.RunnerRefusal, match="deadline"):
        R.resolve_plan(_parse(_launch_argv(
            tmp_path, **{"--work-deadline-s": str(P.SEGMENT_WORK_DEADLINE_S + 1)})))


@pytest.mark.parametrize("allowance", ["0", "-5"])
def test_a_launch_without_a_shutdown_allowance_is_refused(allowance, tmp_path):
    """Behavioural. A segment that cannot fund its own checkpoint must not start."""
    with pytest.raises(R.RunnerRefusal, match="allowance"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--shutdown-allowance-s": allowance})))


def test_bound_plus_allowance_may_not_exceed_the_sealed_call_bound(tmp_path):
    with pytest.raises(R.RunnerRefusal, match="call bound"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{
            "--reserved-bound-s": str(P.SEGMENT_CALL_BOUND_S), "--shutdown-allowance-s": "60"})))


def test_a_bound_that_cannot_cover_the_work_deadline_is_refused(tmp_path):
    """Behavioural. The child is killed at the bound, so a bound below the work
    deadline would guarantee a mid-epoch kill instead of a clean boundary stop."""
    with pytest.raises(R.RunnerRefusal, match="deadline"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{
            "--reserved-bound-s": "100", "--work-deadline-s": "460"})))


def test_the_default_bound_is_the_tightest_remaining_cap_minus_the_allowance(tmp_path):
    """Behavioural. Global, stage and seed caps bind together, tightest first."""
    remaining = B.Remaining(global_s=28800.0, stage_s=500.0, seed_s=1200.0)
    plan = R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--work-deadline-s": "100"})), remaining)
    assert remaining.binding == "stage"
    assert plan.reserved_bound_s == pytest.approx(remaining.call_cap - plan.shutdown_allowance_s)
    assert plan.reserved_bound_s + plan.shutdown_allowance_s <= P.SEGMENT_CALL_BOUND_S


def test_a_remaining_cap_too_small_to_fund_a_bounded_call_is_refused(tmp_path):
    remaining = B.Remaining(global_s=28800.0, stage_s=10.0, seed_s=1200.0)
    with pytest.raises(R.RunnerRefusal, match="remaining"):
        R.resolve_plan(_parse(_launch_argv(tmp_path)), remaining)


def test_a_missing_freeze_manifest_is_refused(tmp_path):
    with pytest.raises(R.RunnerRefusal, match="freeze"):
        R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--freeze": str(tmp_path / "absent.json")})))


# ==========================================================================
# Unit 2 -- launcher budget lifecycle, spawn marker and settlement
# ==========================================================================

class FakeClock:
    """A monotonic fake. No test ever sleeps on a real wall clock."""

    def __init__(self, start=1_000_000.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


class FakeProcess:
    """A child stand-in. Never a real process, never a real pid probe."""

    def __init__(self, returncode=0, hangs=False):
        self.returncode = returncode
        self.pid = 4242
        self.killed = False
        self._hangs = hangs

    def wait(self, timeout=None):
        if self._hangs and not self.killed:
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return self.returncode

    def kill(self):
        self.killed = True


class RecordingSpawner:
    """Records argv and the ledger rows visible at the instant of the spawn."""

    def __init__(self, process=None, on_spawn=None, raises=None):
        self.process = process or FakeProcess()
        self.calls = []
        self.rows_at_spawn = None
        self.on_spawn = on_spawn
        self.raises = raises
        self.ledger = None

    def bind(self, ledger):
        self.ledger = ledger
        return self

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        self.rows_at_spawn = tuple(self.ledger.rows)
        if self.on_spawn is not None:
            self.on_spawn(argv, kwargs)
        if self.raises is not None:
            raise self.raises
        return self.process


def _new_ledger(tmp_path, clock, probe=None):
    return B.BudgetLedger.create(tmp_path / "budget_ledger.jsonl",
                                 protocol_digest=P.protocol_digest(),
                                 clock=clock, contention_probe=probe)


def _counters(**overrides):
    values = {"attempted_control_transitions": 8192, "completed_control_transitions": 8192,
              "replayed_control_transitions": 0, "attempted_physics_steps": 32768,
              "completed_physics_steps": 32768, "replayed_physics_steps": 0,
              "partial_update_cost_s": 0.0}
    values.update(overrides)
    return values


def _charge_seed(ledger, seconds, *, stage="screen", recipe="g990_e010", seed=1001):
    """Burn budget through the real append-only API, never by editing the file."""
    reservation = ledger.reserve(stage=stage, recipe=recipe, seed=seed, segment_index=0,
                                 start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
                                 shutdown_allowance_s=60.0)
    ledger.mark_spawn(reservation)
    ledger.settle(reservation, actual_wall_s=float(seconds), returncode=0,
                  counters=B.SegmentCounters(**_counters()))


def _write_result(out_dir, **overrides):
    payload = {"schema_version": R.RESULT_SCHEMA, "training_began": True,
               "accounting": _counters(**overrides)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def _kinds(ledger):
    return [row["kind"] for row in ledger.rows]


def _launch(tmp_path, ledger, spawner, clock, **overrides):
    args = _parse(_launch_argv(tmp_path, **overrides))
    return R.run_launch(args, ledger=ledger, spawn=spawner.bind(ledger), clock=clock)


# -- the happy path ---------------------------------------------------------

def test_a_completed_child_settles_the_measured_wall_and_the_child_return_code(tmp_path):
    """Behavioural. The wrapper's own status is not the child's; the child's is."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    out_dir = tmp_path / "run" / "segment_0000"

    def child(argv, kwargs):
        clock.advance(123.5)
        _write_result(out_dir)

    spawner = RecordingSpawner(on_spawn=child)
    code = _launch(tmp_path, ledger, spawner, clock)

    assert code == R.EXIT_OK
    assert _kinds(ledger) == ["header", "reserve", "spawn", "settle"]
    settle = ledger.rows[-1]
    assert settle["charge_basis"] == B.CHARGE_ACTUAL
    assert settle["charged_s"] == pytest.approx(123.5)
    assert settle["returncode"] == 0
    assert settle["spawned"] is True
    assert settle["completed_control_transitions"] == 8192


def test_the_child_argv_is_this_script_in_worker_mode_with_the_bound_epoch_span(tmp_path):
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    spawner = RecordingSpawner(on_spawn=lambda a, k: _write_result(tmp_path / "run" / "segment_0000"))
    _launch(tmp_path, ledger, spawner, clock, **{"--epochs": "8"})

    argv = spawner.calls[0][0]
    assert argv[1].endswith("run_myoleg26_ppo_v2.py")
    assert argv[2] == "worker"
    assert argv[argv.index("--epochs") + 1] == "8"
    assert argv[argv.index("--out-dir") + 1] == str(tmp_path / "run" / "segment_0000")
    assert "launch" not in argv


def test_a_nonzero_child_that_still_reported_counters_settles_the_measured_wall(tmp_path):
    """Behavioural. A censored child measured real seconds; they are charged as
    actual, never as an unknown bound and never as zero."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)

    def child(argv, kwargs):
        clock.advance(40.0)
        _write_result(tmp_path / "run" / "segment_0000")

    spawner = RecordingSpawner(process=FakeProcess(returncode=R.EXIT_CENSORED), on_spawn=child)
    code = _launch(tmp_path, ledger, spawner, clock)

    assert code == R.EXIT_CENSORED
    assert ledger.rows[-1]["charge_basis"] == B.CHARGE_ACTUAL
    assert ledger.rows[-1]["charged_s"] == pytest.approx(40.0)
    assert ledger.rows[-1]["returncode"] == R.EXIT_CENSORED


# -- the spawn marker -------------------------------------------------------

def test_the_spawn_marker_is_the_immediately_preceding_ledger_write(tmp_path):
    """Binding. Nothing may sit between ``mark_spawn`` and the spawn."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    spawner = RecordingSpawner(on_spawn=lambda a, k: _write_result(tmp_path / "run" / "segment_0000"))
    _launch(tmp_path, ledger, spawner, clock)

    assert spawner.rows_at_spawn is not None
    assert spawner.rows_at_spawn[-1]["kind"] == B.SPAWN_KIND
    assert [row["kind"] for row in spawner.rows_at_spawn] == ["header", "reserve", "spawn"]


def test_exactly_one_spawn_marker_is_written_per_reservation(tmp_path):
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    spawner = RecordingSpawner(on_spawn=lambda a, k: _write_result(tmp_path / "run" / "segment_0000"))
    _launch(tmp_path, ledger, spawner, clock)
    assert _kinds(ledger).count(B.SPAWN_KIND) == 1


# -- pre-spawn failures settle at zero --------------------------------------

def test_an_existing_output_directory_aborts_before_any_spawn_at_a_zero_charge(tmp_path):
    """Binding. Reserve row plus settle row, and **no** spawn row."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    (tmp_path / "run" / "segment_0000").mkdir(parents=True)
    spawner = RecordingSpawner()

    with pytest.raises(R.RunnerRefusal):
        _launch(tmp_path, ledger, spawner, clock)

    assert spawner.calls == []
    assert _kinds(ledger) == ["header", "reserve", "settle"]
    assert B.SPAWN_KIND not in _kinds(ledger)
    assert ledger.rows[-1]["charged_s"] == 0.0
    assert ledger.rows[-1]["charge_basis"] == B.CHARGE_ABORTED_PRESPAWN
    assert ledger.rows[-1]["spawned"] is False


def test_a_pre_existing_child_log_aborts_before_the_spawn_at_a_zero_charge(tmp_path):
    """Behavioural. The runner's own ``settle_aborted`` call site: the
    reservation exists, no marker has been written, nothing was started."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    spawner = RecordingSpawner()
    args = _parse(_launch_argv(tmp_path))

    # Plant the exclusive child log the instant the output directory appears, so
    # the failure lands after ``reserve`` returns and before ``mark_spawn``.
    original_mkdir = Path.mkdir

    def mkdir_then_plant(self, *a, **kw):
        original_mkdir(self, *a, **kw)
        if self.name == "segment_0000":
            (self / "child.stdout.log").write_bytes(b"")

    Path.mkdir = mkdir_then_plant
    try:
        with pytest.raises(R.RunnerRefusal):
            R.run_launch(args, ledger=ledger, spawn=spawner.bind(ledger), clock=clock)
    finally:
        Path.mkdir = original_mkdir

    assert spawner.calls == []
    assert _kinds(ledger) == ["header", "reserve", "settle"]
    assert ledger.rows[-1]["charged_s"] == 0.0
    assert ledger.rows[-1]["charge_basis"] == B.CHARGE_ABORTED_PRESPAWN


# -- post-spawn failures never settle at zero -------------------------------

def test_a_timed_out_child_settles_the_full_reserved_bound_never_zero(tmp_path):
    """Binding. Marker present, outcome unknown: the full bound is charged."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    spawner = RecordingSpawner(process=FakeProcess(hangs=True))
    code = _launch(tmp_path, ledger, spawner, clock)

    assert code == R.EXIT_INTERRUPTED
    assert spawner.process.killed is True
    assert _kinds(ledger) == ["header", "reserve", "spawn", "settle"]
    settle = ledger.rows[-1]
    assert settle["charge_basis"] == B.CHARGE_RESERVED_BOUND
    assert settle["charged_s"] == settle["reserved_bound_s"]
    assert settle["charged_s"] > 0.0
    assert settle["completed_control_transitions"] is None


def test_a_spawn_that_itself_fails_after_the_marker_charges_the_bound(tmp_path):
    """Binding, conservative direction. Marking early may overcharge; marking
    late would be a budget bypass, so a failed spawn after the marker is never
    retried into a zero charge."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    spawner = RecordingSpawner(raises=OSError("CreateProcess failed"))

    with pytest.raises(R.RunnerRefusal):
        _launch(tmp_path, ledger, spawner, clock)

    assert _kinds(ledger) == ["header", "reserve", "spawn", "settle"]
    assert ledger.rows[-1]["charge_basis"] == B.CHARGE_RESERVED_BOUND
    assert ledger.rows[-1]["charged_s"] > 0.0


def test_a_missing_child_result_settles_unknown_rather_than_zero(tmp_path):
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    spawner = RecordingSpawner()          # child writes no result.json
    code = _launch(tmp_path, ledger, spawner, clock)

    assert code != R.EXIT_OK
    assert ledger.rows[-1]["charge_basis"] == B.CHARGE_RESERVED_BOUND
    assert ledger.rows[-1]["charged_s"] > 0.0


def test_a_malformed_child_result_settles_unknown_rather_than_guessing(tmp_path):
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)

    def child(argv, kwargs):
        out = tmp_path / "run" / "segment_0000"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text("{not json", encoding="utf-8")

    spawner = RecordingSpawner(on_spawn=child)
    _launch(tmp_path, ledger, spawner, clock)
    assert ledger.rows[-1]["charge_basis"] == B.CHARGE_RESERVED_BOUND


def test_a_zero_settle_after_the_marker_is_refused_and_not_retried_into_zero(tmp_path):
    """Binding. The ledger refuses it; the runner must not catch and retry."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    reservation = ledger.reserve(stage="screen", recipe="g990_e010", seed=1001, segment_index=0,
                                 start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
                                 shutdown_allowance_s=60.0)
    ledger.mark_spawn(reservation)
    with pytest.raises(B.SettlementError):
        ledger.settle_aborted(reservation, detail="attempted zero after a spawn")
    assert _kinds(ledger) == ["header", "reserve", "spawn"]


# -- the zero/bound implication, both directions ----------------------------

@pytest.mark.parametrize("hangs,writes_result", [(False, True), (True, False)])
def test_a_charge_after_a_spawn_marker_is_never_zero(hangs, writes_result, tmp_path):
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)

    def child(argv, kwargs):
        clock.advance(11.0)
        if writes_result:
            _write_result(tmp_path / "run" / "segment_0000")

    spawner = RecordingSpawner(process=FakeProcess(hangs=hangs), on_spawn=child)
    _launch(tmp_path, ledger, spawner, clock)

    settle = ledger.rows[-1]
    assert B.SPAWN_KIND in _kinds(ledger)
    assert settle["charged_s"] > 0.0
    assert settle["charge_basis"] != B.CHARGE_ABORTED_PRESPAWN


def test_a_zero_charge_occurs_only_when_no_spawn_marker_is_present(tmp_path):
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    (tmp_path / "run" / "segment_0000").mkdir(parents=True)
    with pytest.raises(R.RunnerRefusal):
        _launch(tmp_path, ledger, RecordingSpawner(), clock)

    zeros = [row for row in ledger.rows
             if row["kind"] == B.SETTLE_KIND and row["charged_s"] == 0.0]
    assert zeros
    for row in zeros:
        assert row["charge_basis"] == B.CHARGE_ABORTED_PRESPAWN
        assert not any(other["kind"] == B.SPAWN_KIND
                       and other["reservation_id"] == row["reservation_id"]
                       for other in ledger.rows)


# -- caps bind before anything heavy ----------------------------------------

def test_an_exhausted_seed_cap_is_refused_before_any_reservation_or_spawn(tmp_path):
    """Binding. Refusal precedes simulator creation; it leaves no new row."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    _charge_seed(ledger, P.stage("screen").per_seed_safety_cap_s - 5.0)
    before = len(ledger.rows)
    spawner = RecordingSpawner()

    with pytest.raises(R.RunnerRefusal, match="remaining"):
        _launch(tmp_path, ledger, spawner, clock)

    assert spawner.calls == []
    assert len(ledger.rows) == before
    assert not (tmp_path / "run" / "segment_0000").exists()


def test_the_tightest_of_the_three_caps_binds(tmp_path):
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    _charge_seed(ledger, 900.0)
    remaining = ledger.remaining("screen", "g990_e010", 1001)
    assert remaining.binding == "seed"
    assert remaining.least == pytest.approx(300.0)
    plan = R.resolve_plan(_parse(_launch_argv(tmp_path, **{"--work-deadline-s": "200"})), remaining)
    assert plan.reserved_bound_s + plan.shutdown_allowance_s <= 300.0
    assert plan.reserved_bound_s >= plan.work_deadline_s


def test_a_tight_remaining_cap_refuses_rather_than_shrinking_the_work_deadline(tmp_path):
    """Behavioural. The runner never silently retunes a declared deadline to fit
    a cap: it refuses and makes the caller choose the shorter deadline."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    _charge_seed(ledger, 900.0)
    remaining = ledger.remaining("screen", "g990_e010", 1001)
    with pytest.raises(R.RunnerRefusal, match="work deadline"):
        R.resolve_plan(_parse(_launch_argv(tmp_path)), remaining)


# -- the ledger is append-only in every path --------------------------------

@pytest.mark.parametrize("scenario", ["clean", "timeout", "prespawn", "spawn_error"])
def test_no_path_ever_retracts_truncates_or_replaces_a_ledger_row(scenario, tmp_path):
    """Binding. A hash chain protects history, not the tail; the runner must
    never rewrite the tail on any path, including error and cleanup paths."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    path = ledger.path
    before = path.read_bytes()

    spawner = RecordingSpawner()
    if scenario == "clean":
        spawner.on_spawn = lambda a, k: _write_result(tmp_path / "run" / "segment_0000")
    elif scenario == "timeout":
        spawner.process = FakeProcess(hangs=True)
    elif scenario == "prespawn":
        (tmp_path / "run" / "segment_0000").mkdir(parents=True)
    else:
        spawner.raises = OSError("CreateProcess failed")

    try:
        _launch(tmp_path, ledger, spawner, clock)
    except R.RunnerRefusal:
        pass

    after = path.read_bytes()
    assert after.startswith(before)
    assert len(after) >= len(before)
    assert len(list(tmp_path.glob("*.jsonl"))) == 1
    B.BudgetLedger.open(path)          # the chain still verifies end to end


# -- contention is provenance only ------------------------------------------

def test_contention_is_recorded_at_launch_without_threshold_or_refusal(tmp_path):
    clock = FakeClock()
    observation = {"contended": None, "compute_apps": ["1234, game.exe, 900 MiB"],
                   "gpu": ["40 %, 3000 MiB, 8188 MiB"]}
    ledger = _new_ledger(tmp_path, clock, probe=lambda: observation)
    spawner = RecordingSpawner(on_spawn=lambda a, k: _write_result(tmp_path / "run" / "segment_0000"))
    code = _launch(tmp_path, ledger, spawner, clock)

    assert code == R.EXIT_OK
    reserve = ledger.rows[1]
    assert reserve["contention_at_launch"]["contended"] is None
    assert reserve["contention_at_launch"]["compute_apps"] == observation["compute_apps"]


def test_a_failing_contention_probe_never_costs_a_launch(tmp_path):
    clock = FakeClock()

    def boom():
        raise RuntimeError("nvidia-smi missing")

    ledger = _new_ledger(tmp_path, clock, probe=boom)
    spawner = RecordingSpawner(on_spawn=lambda a, k: _write_result(tmp_path / "run" / "segment_0000"))
    assert _launch(tmp_path, ledger, spawner, clock) == R.EXIT_OK
    assert ledger.rows[1]["contention_at_launch"]["status"] == B.CONTENTION_PROBE_FAILED


def test_the_builtin_probe_makes_no_threshold_and_no_causal_claim():
    """Behavioural. ``contended`` stays ``None``: declaring it would need a
    threshold, and this observation has none."""
    class _Result:
        returncode = 0
        stdout = "1234, python.exe, 900 MiB\n"

    record = R.gpu_contention_probe(run=lambda *a, **k: _Result())
    assert record["contended"] is None
    assert record["compute_apps"] == ["1234, python.exe, 900 MiB"]


def test_the_builtin_probe_survives_a_missing_nvidia_smi():
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    record = R.gpu_contention_probe(run=boom)
    assert record["contended"] is None
    assert "error" in record


# -- process control --------------------------------------------------------

def test_the_runner_never_uses_a_pid_liveness_probe_or_signals_another_job():
    """Binding. ``os.kill(pid, 0)`` is not a Windows liveness probe, and this
    runner owns only its own child's handle."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert "taskkill" not in source
    assert "psutil" not in source
    assert source.count(".kill()") == 1


# ==========================================================================
# Unit 3 -- instrumented training: deadlines, censoring, events, neutrality
# ==========================================================================
#
# The Task 3 analytic CPU adapter is reused rather than reinvented: it is a real
# ``PPO`` object graph with closed-form CPU dynamics and the MyoLeg26 reset and
# action-history boundary shape, and it never touches CUDA.

import math
import torch

from msk_warp.analysis import ppo_diagnostics as D

from tests.unit.test_ppo_resume_state import (  # noqa: E402
    _build, _differences, _fingerprint, _seed_cpu, _start,
)

CONTROL_DT = 0.008


class TickingClock:
    """A fake clock that advances a fixed amount on every read.

    Deadlines therefore arrive deterministically after a known number of counted
    steps, with no sleeping and no dependence on real wall time.
    """

    def __init__(self, start=0.0, tick=1.0):
        self.now = float(start)
        self.tick = float(tick)
        self.reads = 0

    def __call__(self):
        self.reads += 1
        value = self.now
        self.now += self.tick
        return value


def _recorder(env):
    return R.EpisodeEventRecorder(env.num_envs, CONTROL_DT, device="cpu")


def _run(algo, env, *, epochs=2, start_epoch=0, clock=None, deadline=1e9,
         capture_reserve_s=0.0, resumed_from_boundary=False, **kwargs):
    clock = clock or TickingClock(tick=0.0)
    return R.train_segment(
        algo, env, start_epoch=start_epoch, end_epoch=start_epoch + epochs,
        max_epochs_total=128, clock=clock, work_deadline=deadline,
        recorder=_recorder(env), resumed_from_boundary=resumed_from_boundary,
        capture_reserve_s=capture_reserve_s, **kwargs)


def _cuda_state():
    """CUDA initialisation state, used as a *change* detector.

    Measured fact, not an assumption: ``torch.optim.Adam.step()`` initialises
    CUDA even when every parameter is CPU-only, because it probes device
    capability to choose its fused/foreach implementation. That happens inside
    PPO's own ``update()``, upstream of this runner, so a blanket "CUDA is never
    initialised" assertion after training would be a claim about torch rather
    than about this unit. The two guards that do bind are therefore: the runner
    module imports and plans a launch **without** CUDA, checked in a clean
    subprocess below; and the instrumentation never *changes* this state.
    """
    return torch.cuda.is_initialized()


# -- the clean path ---------------------------------------------------------

def test_a_clean_segment_completes_every_epoch_at_a_live_boundary(tmp_path):
    algo, env = _build()
    _start(algo, env)
    outcome = _run(algo, env, epochs=3)

    assert outcome.completed_epochs == 3
    assert outcome.completed_epoch == 3
    assert outcome.boundary_is_live is True
    assert outcome.censored is False
    assert outcome.stop_reason == "epoch_budget"
    assert len(outcome.losses) == 3


def test_counted_steps_match_the_epochs_that_actually_completed():
    algo, env = _build(steps_num=4)
    _start(algo, env)
    outcome = _run(algo, env, epochs=3)
    counters = outcome.counters

    assert counters.attempted_control_calls == 3 * algo.steps_num
    assert counters.returned_control_calls == counters.attempted_control_calls
    assert counters.boundary_control_calls == counters.attempted_control_calls


def test_the_instrumented_step_is_removed_again_afterwards():
    algo, env = _build()
    _start(algo, env)
    before = env.step
    _run(algo, env, epochs=1)
    assert env.step == before or env.step.__func__ is before.__func__


# -- deadlines --------------------------------------------------------------

def test_a_deadline_before_an_epoch_stops_at_the_previous_live_boundary():
    """Behavioural. Stopping *between* epochs keeps the boundary publishable."""
    algo, env = _build(steps_num=4)
    _start(algo, env)
    clock = TickingClock(tick=1.0)
    outcome = _run(algo, env, epochs=8, clock=clock, deadline=40.0, capture_reserve_s=5.0)

    assert outcome.boundary_is_live is True
    assert outcome.stop_reason == "work_deadline_epoch_boundary"
    assert 0 < outcome.completed_epochs < 8
    assert outcome.completed_epoch == outcome.completed_epochs
    assert outcome.counters.boundary_control_calls == outcome.counters.attempted_control_calls


def test_a_deadline_inside_a_rollout_censors_the_partial_epoch():
    """Binding. A single epoch cannot overrun unchecked: the deadline is tested
    on every counted step, not only at epoch boundaries."""
    algo, env = _build(steps_num=8)
    _start(algo, env)
    clock = TickingClock(tick=1.0)
    # The deadline lands part-way through the first rollout, so only the counted
    # step check can catch it -- an epoch-boundary check alone would not.
    outcome = _run(algo, env, epochs=4, clock=clock, deadline=6.0)

    assert outcome.boundary_is_live is False
    assert outcome.stop_reason == "work_deadline_partial_epoch"
    assert outcome.counters.attempted_control_calls > outcome.counters.boundary_control_calls
    assert outcome.censored is False          # censoring a partial epoch is not a nonfinite censor


def test_a_partial_epoch_is_never_published_as_a_completed_boundary():
    """Binding. The completed-epoch count never advances for an epoch whose
    update did not run."""
    algo, env = _build(steps_num=8)
    _start(algo, env)
    clock = TickingClock(tick=1.0)
    # The deadline lands part-way through the first rollout, so only the counted
    # step check can catch it -- an epoch-boundary check alone would not.
    outcome = _run(algo, env, epochs=4, clock=clock, deadline=6.0)

    assert outcome.completed_epoch == outcome.completed_epochs
    assert outcome.boundary_is_live is False
    # Discarded work is the whole segment: nothing may be published from it.
    assert outcome.counters.attempted_control_calls > 0


def test_an_update_is_never_started_without_its_shutdown_allowance():
    """Binding. Rollout finished, but the remaining budget cannot fund an update
    plus the checkpoint, so the epoch stops rather than half-updating."""
    algo, env = _build(steps_num=2)
    _start(algo, env)
    clock = TickingClock(tick=1.0)
    # The epoch starts (clock 0 + 5 s reserve fits), the rollout finishes, and
    # only then does the remaining budget fail to cover an update plus the
    # checkpoint.
    outcome = _run(algo, env, epochs=4, clock=clock, deadline=8.0, capture_reserve_s=5.0)

    assert outcome.boundary_is_live is False
    assert outcome.stop_reason == "work_deadline_partial_epoch"
    assert "update" in (outcome.partial_detail or "")


def test_an_interrupted_update_records_its_partial_cost_separately():
    """Behavioural. An update that started and did not finish is real spent wall
    and is recorded on its own, never netted off."""
    algo, env = _build(steps_num=2)
    _start(algo, env)
    clock = TickingClock(tick=1.0)
    original_update = algo.update

    calls = {"n": 0}

    def failing_update():
        calls["n"] += 1
        if calls["n"] == 2:
            raise R.WorkDeadlineExceeded("deadline reached inside the update")
        return original_update()

    algo.update = failing_update
    outcome = _run(algo, env, epochs=4, clock=clock, deadline=1e9)

    assert outcome.completed_epochs == 1
    assert outcome.boundary_is_live is False
    assert outcome.counters.partial_update_cost_s > 0.0
    assert outcome.stop_reason == "work_deadline_partial_epoch"


# -- nonfinite censoring ----------------------------------------------------

@pytest.mark.parametrize("field", ["reward", "observation"])
def test_a_nonfinite_step_output_censors_the_run_immediately(field):
    """Binding. No retry, no clamp, no skipped world: the run stops for
    diagnosis at the first nonfinite value."""
    algo, env = _build(steps_num=8)
    _start(algo, env)
    original_step = env.step
    seen = {"n": 0}

    def poisoned(action, *a, **k):
        obs, rew, done, extras = original_step(action, *a, **k)
        seen["n"] += 1
        if seen["n"] == 3:
            if field == "reward":
                rew = rew.clone()
                rew[0] = float("nan")
            else:
                obs = obs.clone()
                obs[0, 0] = float("inf")
        return obs, rew, done, extras

    env.step = poisoned
    outcome = _run(algo, env, epochs=2)

    assert outcome.censored is True
    assert field in (outcome.censor_reason or "")
    assert outcome.boundary_is_live is False
    assert seen["n"] == 3            # stopped at the first bad step, never retried


def test_a_nonfinite_parameter_is_censored_before_the_environment_is_stepped():
    """Behavioural, and a corrected expectation. A nonfinite actor parameter
    never reaches the step wrapper as a nonfinite *action*: PPO builds a
    ``torch.distributions.Normal`` from it first, which raises on a nonfinite
    location. The censor therefore fires before the rollout and names the
    parameter, which is the real cause."""
    algo, env = _build(steps_num=4)
    _start(algo, env)
    with torch.no_grad():
        algo.actor.mu_net[0].bias.fill_(float("nan"))
    outcome = _run(algo, env, epochs=1)

    assert outcome.censored is True
    assert "parameter" in (outcome.censor_reason or "")
    assert outcome.counters.attempted_control_calls == 0
    assert outcome.counters.returned_control_calls == 0


def test_the_step_wrapper_censors_a_nonfinite_action_before_stepping():
    """Binding. The action guard is exercised directly, since a nonfinite
    parameter is intercepted earlier by torch itself."""
    algo, env = _build(steps_num=4)
    _start(algo, env)
    counters = R.SegmentCounters()
    stepped = {"n": 0}
    original = env.step

    def counting_step(action, *a, **k):
        stepped["n"] += 1
        return original(action, *a, **k)

    env.step = counting_step
    remove = R._install_counted_step(env, _recorder(env), counters,
                                     clock=TickingClock(tick=0.0), work_deadline=1e9)
    try:
        bad = torch.full((env.num_envs, env.num_actions), float("nan"))
        with pytest.raises(R.CensorRun, match="action"):
            env.step(bad)
    finally:
        remove()

    assert stepped["n"] == 0
    assert counters.attempted_control_calls == 0
    assert counters.returned_control_calls == 0


def test_a_nonfinite_update_loss_censors_the_run():
    algo, env = _build(steps_num=4)
    _start(algo, env)
    algo.update = lambda: {"actor_loss": float("nan"), "value_loss": 0.0, "entropy": 0.0}
    outcome = _run(algo, env, epochs=2)

    assert outcome.censored is True
    assert "loss" in (outcome.censor_reason or "")
    assert outcome.completed_epochs == 0


def test_a_nonfinite_parameter_censors_the_run():
    algo, env = _build(steps_num=4)
    _start(algo, env)
    original_update = algo.update

    def poison():
        losses = original_update()
        with torch.no_grad():
            algo.critic.critic[0].weight[0, 0] = float("inf")
        return losses

    algo.update = poison
    outcome = _run(algo, env, epochs=2)

    assert outcome.censored is True
    assert "parameter" in (outcome.censor_reason or "")
    assert outcome.completed_epochs == 0


# -- episode events ---------------------------------------------------------

def test_events_carry_the_declared_fields_with_a_reward_sum_return():
    algo, env = _build(steps_num=8, num_envs=4)
    _start(algo, env)
    outcome = _run(algo, env, epochs=2)

    events = [event for epoch in outcome.epoch_events for event in epoch["events"]]
    assert events, "the analytic env terminates often enough to complete episodes"
    for event in events:
        assert set(event) == set(D.EPISODE_EVENT_FIELDS)
        assert isinstance(event["world"], int) and event["world"] >= 0
        assert isinstance(event["length_controls"], int) and event["length_controls"] > 0
        assert event["duration_s"] == pytest.approx(event["length_controls"] * CONTROL_DT)
        assert math.isfinite(event["return"])
        assert isinstance(event["end_reason"], str)
        assert isinstance(event["failure_flags"], dict)


def test_the_event_return_is_the_reward_sum_and_not_ppos_negated_episode_loss():
    """Binding. ``PPO.episode_loss`` accumulates **negated** reward; supplying it
    would silently flip the sign of every reported episode return."""
    algo, env = _build(steps_num=8, num_envs=4)
    _start(algo, env)
    outcome = _run(algo, env, epochs=1)

    events = outcome.epoch_events[0]["events"]
    assert events
    ours = sorted(round(event["return"], 6) for event in events)
    ppo_negated = sorted(round(value, 6) for value in algo.episode_loss_his)
    assert len(ours) == len(ppo_negated)
    assert ours == sorted(round(-value, 6) for value in ppo_negated)
    assert ours != ppo_negated       # anti-vacuity: the two are genuinely opposite


def test_event_lengths_agree_with_ppos_own_episode_lengths():
    algo, env = _build(steps_num=8, num_envs=4)
    _start(algo, env)
    outcome = _run(algo, env, epochs=1)
    ours = sorted(event["length_controls"] for event in outcome.epoch_events[0]["events"])
    assert ours == sorted(int(value) for value in algo.episode_length_his)


def test_the_event_list_resets_each_epoch_but_the_accumulators_do_not():
    """Binding. The per-epoch list describes *this* epoch; the per-world running
    sums are episode state and must survive an epoch and a segment boundary."""
    algo, env = _build(steps_num=2, num_envs=4)
    _start(algo, env)
    recorder = R.EpisodeEventRecorder(env.num_envs, CONTROL_DT, device="cpu")
    outcome = R.train_segment(algo, env, start_epoch=0, end_epoch=3, max_epochs_total=128,
                              clock=TickingClock(tick=0.0), work_deadline=1e9,
                              recorder=recorder, resumed_from_boundary=False,
                              capture_reserve_s=0.0)

    assert len(outcome.epoch_events) == 3
    for epoch in outcome.epoch_events:
        assert epoch["events"] is not outcome.epoch_events[0]["events"] or epoch is outcome.epoch_events[0]
    # An episode longer than one epoch is still counted once, at its full length.
    assert any(event["length_controls"] > 2
               for epoch in outcome.epoch_events for event in epoch["events"])


def test_the_recorder_state_round_trips_through_the_resume_extra():
    algo, env = _build(steps_num=2, num_envs=4)
    _start(algo, env)
    recorder = R.EpisodeEventRecorder(env.num_envs, CONTROL_DT, device="cpu")
    R.train_segment(algo, env, start_epoch=0, end_epoch=1, max_epochs_total=128,
                    clock=TickingClock(tick=0.0), work_deadline=1e9, recorder=recorder,
                    resumed_from_boundary=False, capture_reserve_s=0.0)
    state = recorder.state()
    assert set(state) == {"returns", "lengths"}
    assert all(isinstance(value, float) for value in state["returns"])
    assert all(isinstance(value, int) for value in state["lengths"])

    revived = R.EpisodeEventRecorder(env.num_envs, CONTROL_DT, device="cpu")
    revived.load(state)
    assert revived.state() == state


def test_the_recorder_never_reads_ppos_meters_or_history_lists():
    """Binding. None of those describes the current epoch, and two of them are
    a capped 100-sample rolling mean."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    for forbidden in ("episode_length_his", "episode_loss_his",
                      "episode_length_meter", "episode_loss_meter"):
        assert f"algo.{forbidden}" not in source
        assert f".{forbidden}[" not in source


def test_the_event_summary_is_produced_by_the_task_4a_helper():
    algo, env = _build(steps_num=8, num_envs=4)
    _start(algo, env)
    outcome = _run(algo, env, epochs=1)
    summary = outcome.epoch_events[0]["summary"]
    assert summary["status"] == D.STATUS_OK
    assert summary["completed_episodes"] == len(outcome.epoch_events[0]["events"])


# -- instrumentation neutrality ---------------------------------------------

def _clone_state(value):
    """A deep, tensor-aware copy, so the snapshot cannot alias the live object."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_state(item) for item in value)
    return value


def _same_state(left, right) -> bool:
    """Bitwise for tensors, structural elsewhere. Mirrors the Task 4a precedent."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right))
    return left == right


def _neutrality_snapshot(algo):
    """Task 4a's declared set: parameters, every grad, both optimizer state_dicts
    **in full** (``param_groups`` alone would miss every Adam moment), the
    normaliser, the training flag and the CPU RNG stream."""
    return {
        "actor": {k: v.detach().clone() for k, v in algo.actor.state_dict().items()},
        "critic": {k: v.detach().clone() for k, v in algo.critic.state_dict().items()},
        "actor_grads": [None if p.grad is None else p.grad.detach().clone()
                        for p in algo.actor.parameters()],
        "critic_grads": [None if p.grad is None else p.grad.detach().clone()
                         for p in algo.critic.parameters()],
        "actor_opt": _clone_state(algo.actor_optimizer.state_dict()),
        "critic_opt": _clone_state(algo.critic_optimizer.state_dict()),
        "obs_rms": (algo.obs_rms.mean.clone(), algo.obs_rms.var.clone(), float(algo.obs_rms.count)),
        "training": algo.actor.training,
        "cpu_rng": torch.get_rng_state().clone(),
    }


def _assert_snapshots_equal(before, after):
    assert torch.equal(before["cpu_rng"], after["cpu_rng"])
    assert before["training"] == after["training"]
    for key in ("actor_opt", "critic_opt"):
        assert _same_state(before[key], after[key]), key
    for key in ("actor", "critic"):
        assert set(before[key]) == set(after[key])
        for name in before[key]:
            assert torch.equal(before[key][name], after[key][name]), name
    for key in ("actor_grads", "critic_grads"):
        assert len(before[key]) == len(after[key])
        for left, right in zip(before[key], after[key]):
            assert (left is None) == (right is None)
            if left is not None:
                assert torch.equal(left, right)
    assert torch.equal(before["obs_rms"][0], after["obs_rms"][0])
    assert torch.equal(before["obs_rms"][1], after["obs_rms"][1])
    assert before["obs_rms"][2] == after["obs_rms"][2]


def test_installing_and_removing_the_instrumentation_changes_nothing():
    algo, env = _build()
    _start(algo, env)
    before = _neutrality_snapshot(algo)
    cuda_before = _cuda_state()
    remove = R._install_counted_step(env, _recorder(env), R.SegmentCounters(),
                                     clock=TickingClock(tick=0.0), work_deadline=1e9)
    remove()
    _assert_snapshots_equal(before, _neutrality_snapshot(algo))
    assert _cuda_state() == cuda_before


def test_an_instrumented_segment_is_bitwise_identical_to_an_uninstrumented_one():
    """Binding. The strongest neutrality control available on CPU: the same seed
    trained with and without the instrumentation must agree on everything a
    faithful continuation reproduces."""
    reference_algo, reference_env = _build(seed=11)
    _start(reference_algo, reference_env)
    for _ in range(3):
        reference_algo.collect_rollout()
        reference_algo.update()
        reference_algo.iter_count += 1
    expected = _fingerprint(reference_algo, reference_env)
    expected_rng = torch.get_rng_state().clone()

    _seed_cpu(11)
    instrumented_algo, instrumented_env = _build(seed=11)
    _start(instrumented_algo, instrumented_env)
    outcome = _run(instrumented_algo, instrumented_env, epochs=3)

    assert outcome.completed_epochs == 3
    assert _differences(expected, _fingerprint(instrumented_algo, instrumented_env)) == []
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_the_diagnostics_pass_is_rng_neutral():
    algo, env = _build(steps_num=4)
    _start(algo, env)
    _run(algo, env, epochs=1)
    before = torch.get_rng_state().clone()
    cuda_before = _cuda_state()
    report = R.epoch_diagnostics(algo)
    assert torch.equal(before, torch.get_rng_state())
    assert report["kl"]["status"] in {D.STATUS_OK, "no_samples"}
    assert report["logstd"]["dimensions"] == algo.num_actions
    assert _cuda_state() == cuda_before


def test_diagnostics_do_not_disturb_parameters_grads_or_the_optimizer():
    algo, env = _build(steps_num=4)
    _start(algo, env)
    _run(algo, env, epochs=1)
    before = _neutrality_snapshot(algo)
    R.epoch_diagnostics(algo)
    _assert_snapshots_equal(before, _neutrality_snapshot(algo))


def test_the_runner_module_imports_and_plans_without_initialising_cuda(tmp_path):
    """Binding, in a clean process. The launcher must be able to reserve budget
    before anything touches a GPU, so importing it and planning a launch must not
    initialise CUDA on their own."""
    program = textwrap.dedent(
        """
        import json, sys, torch
        assert not torch.cuda.is_initialized(), "torch itself initialised CUDA on import"
        from scripts import run_myoleg26_ppo_v2 as R
        args = R.build_parser().parse_args(json.loads(sys.argv[1]))
        plan = R.resolve_plan(args)
        print(json.dumps({"cuda": torch.cuda.is_initialized(),
                          "torch_in_runner": "torch" in dir(R),
                          "epochs": plan.epochs}))
        """
    )
    argv = json.dumps(_launch_argv(tmp_path))
    finished = subprocess.run([sys.executable, "-c", program, argv], capture_output=True,
                              text=True, cwd=str(Path(R.__file__).resolve().parents[1]),
                              timeout=300)
    assert finished.returncode == 0, finished.stderr
    payload = json.loads(finished.stdout.strip().splitlines()[-1])
    assert payload["cuda"] is False
    assert payload["torch_in_runner"] is False
    assert payload["epochs"] == 64


# ==========================================================================
# Unit 4 -- resume identity binding and provenance comparison
# ==========================================================================
#
# ``ppo_resume`` records completely and never compares. Every comparison below
# is therefore the runner's own obligation, and each identity class is compared
# separately: raw execution bytes, git's filtered working-tree blob id and the
# committed HEAD blob id are three different objects under ``core.autocrlf``
# (IN-25) and are never conflated or normalised.

from msk_warp.analysis import ppo_resume  # noqa: E402


def _live_resume():
    """The module object a fresh ``import`` resolves to, right now.

    Upstream hazard, observed rather than assumed:
    ``test_ppo_resume_state.py::test_analysis_package_does_not_import_ppo_resume``
    deletes every ``msk_warp.analysis*`` key from ``sys.modules`` and does not
    put them back. Any reference imported before it ran is then stale for
    *patching* purposes, because the runner's own lazy import -- deliberate, so
    that ``launch`` stays torch-free -- resolves a brand new module object that a
    patch on the stale one would never reach. Resolving the module here, at patch
    time, makes these tests order-independent without modifying that test.
    """
    import importlib

    return importlib.import_module("msk_warp.analysis.ppo_resume")


def test_a_sys_modules_purge_yields_a_different_module_object(monkeypatch):
    """Pins the property that makes late patch resolution necessary.

    A purge of ``msk_warp.analysis*`` makes the next import return a *different*
    module object, so a patch applied to a reference captured earlier would land
    on a module the runner never touches — a silently vacuous patch. The Task 3
    test that used to leak such a purge is fixed in this round; this keeps the
    underlying property pinned, and unlike the original it restores what it
    removes.
    """
    import importlib
    import sys as _sys

    live = _live_resume()
    assert live is _sys.modules["msk_warp.analysis.ppo_resume"]
    for name in [key for key in _sys.modules if key.startswith("msk_warp.analysis")]:
        monkeypatch.delitem(_sys.modules, name)
    revived = importlib.import_module("msk_warp.analysis.ppo_resume")
    assert revived is not live                      # a genuinely different object
    assert revived.SCHEMA_VERSION == live.SCHEMA_VERSION


def _bindings(tmp_path):
    return R.freeze_bindings(_freeze(tmp_path))


def _extra(tmp_path, *, seed=1001, segment_index=0, parent_segment_sha256=None,
           recorder_state=None, replayed=None):
    return R.segment_extra(
        stage="screen", recipe="g990_e010", seed=seed, segment_index=segment_index,
        parent_segment_sha256=parent_segment_sha256, bindings=_bindings(tmp_path),
        recorder_state=recorder_state or {"returns": [0.0], "lengths": [0]},
        replayed=replayed or {"control_transitions": 0, "physics_steps": 0})


def _published(tmp_path, *, epoch=2, seed=1001, segment_index=0,
               parent_segment_sha256=None, name="segment.ptc"):
    algo, env = _build()
    _start(algo, env)
    for _ in range(epoch):
        algo.collect_rollout()
        algo.update()
        algo.iter_count += 1
    extra = _extra(tmp_path, seed=seed, segment_index=segment_index,
                   parent_segment_sha256=parent_segment_sha256,
                   recorder_state={"returns": [0.0] * env.num_envs,
                                   "lengths": [0] * env.num_envs})
    result = R.publish_boundary(algo, env, tmp_path / name, epoch=epoch, extra=extra)
    return result, algo, env


def _resume(tmp_path, result, *, epoch=2, seed=1001, segment_index=0,
            parent_segment_sha256=None, bindings=None, expected_sha=None):
    fresh_algo, fresh_env = _build(seed=999)
    return R.resume_boundary(
        result["path"], fresh_algo, fresh_env, epoch=epoch, seed=seed,
        segment_index=segment_index, parent_segment_sha256=parent_segment_sha256,
        bindings=bindings if bindings is not None else _bindings(tmp_path),
        expected_segment_sha256=expected_sha)


# -- the positive control ---------------------------------------------------

def test_a_published_boundary_resumes_under_the_full_five_key_binding(tmp_path):
    result, _algo, _env = _published(tmp_path)
    report, state = _resume(tmp_path, result, expected_sha=result["sha256"])

    assert report["schema_version"] == ppo_resume.SCHEMA_VERSION
    assert report["epoch"] == 2
    assert state.metadata["epoch"] == 2
    # ``cuda_rng_restored`` describes whether the *capturing* process had a CUDA
    # stream to record, which on this host Adam initialises even for CPU-only
    # parameters. So the contract is that the runner **reports** the flag, not
    # that it has any particular value: when it is False, continuation is not
    # bitwise for any CUDA draw. Continued-trajectory numerical equivalence
    # stays UNVALIDATED either way, and this unit declares no tolerance.
    assert "cuda_rng_restored" in report
    assert isinstance(report["cuda_rng_restored"], bool)
    assert "cuda" in report["rng_streams_restored"] or not report["cuda_rng_restored"]


def test_the_capture_site_always_requests_the_git_identity(tmp_path, monkeypatch):
    """Binding. ``include_git_identity=False`` is the inspection default; a
    campaign segment must record the committed blob ids too."""
    live = _live_resume()
    seen = {}
    original = live.capture_state

    def spy(algo, env, extra=None, **kwargs):
        seen.update(kwargs)
        return original(algo, env, extra, **kwargs)

    monkeypatch.setattr(live, "capture_state", spy)
    _published(tmp_path)
    assert seen["include_git_identity"] is True
    assert seen["epoch"] == 2


def test_every_resume_binds_all_five_required_identity_keys(tmp_path, monkeypatch):
    """Binding. ``expect=None`` is inspection mode and would resume from an
    unbound file, so the runner must name all five keys."""
    result, _algo, _env = _published(tmp_path)
    live = _live_resume()
    seen = {}
    original = live.read_segment

    def spy(path, *, expect=None):
        seen["expect"] = expect
        return original(path, expect=expect)

    monkeypatch.setattr(live, "read_segment", spy)
    _resume(tmp_path, result)

    assert seen["expect"] is not None
    assert set(seen["expect"]) == set(live.REQUIRED_IDENTITY_KEYS)
    assert len(live.REQUIRED_IDENTITY_KEYS) == 5


# -- identity negative controls --------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("epoch", 3), ("seed", 1002), ("segment_index", 7),
    ("parent_segment_sha256", "f" * 64),
])
def test_a_mismatch_on_any_bound_identity_key_is_refused(field, value, tmp_path):
    result, _algo, _env = _published(tmp_path)
    kwargs = {"epoch": 2, "seed": 1001, "segment_index": 0, "parent_segment_sha256": None}
    kwargs[field] = value
    with pytest.raises(R.RunnerRefusal, match="identity|mismatch"):
        _resume(tmp_path, result, **kwargs)


def test_inspection_mode_would_have_accepted_a_foreign_segment(tmp_path):
    """Negative control for ``expect=None``. This is what the runner must never
    do: the same file loads happily with no identity binding at all."""
    result, _algo, _env = _published(tmp_path, seed=1002, segment_index=5)
    unbound = ppo_resume.read_segment(result["path"])          # inspection mode
    assert unbound.extra["seed"] == 1002
    assert unbound.extra["segment_index"] == 5
    # ... while the bound resume refuses exactly that file for this run.
    with pytest.raises(R.RunnerRefusal):
        _resume(tmp_path, result, seed=1001, segment_index=0)


@pytest.mark.parametrize("dropped", ["epoch", "seed", "segment_index",
                                     "parent_segment_sha256"])
def test_dropping_one_identity_key_would_admit_a_mismatched_segment(dropped, tmp_path):
    """Negative control. Each of the five keys is load-bearing on its own: with
    one omitted, a segment that differs only in that key is accepted."""
    result, _algo, _env = _published(tmp_path, epoch=2, seed=1001, segment_index=0)
    expect = ppo_resume.strict_expect(epoch=2, seed=1001, segment_index=0,
                                      parent_segment_sha256=None)
    wrong = {"epoch": 3, "seed": 1002, "segment_index": 9,
             "parent_segment_sha256": "e" * 64}[dropped]
    weakened = {key: value for key, value in expect.items() if key != dropped}
    weakened_wrong = dict(weakened)
    # The weakened binding cannot see the difference it no longer names.
    assert dropped not in weakened_wrong
    ppo_resume.read_segment(result["path"], expect=weakened)
    with pytest.raises(ppo_resume.ResumeValidationError):
        ppo_resume.read_segment(result["path"], expect=dict(expect, **{dropped: wrong}))


# -- provenance: three separate identity classes ---------------------------

def test_a_raw_execution_byte_mismatch_is_refused_on_its_own(tmp_path, monkeypatch):
    """Binding. The raw-bytes map is compared independently of git's ids: a
    mismatch there is never excused by agreement on the blob id."""
    result, _algo, _env = _published(tmp_path)
    current = R.source_identity()
    poisoned = dict(current)
    poisoned["raw_sha256"] = dict(current["raw_sha256"])
    first = sorted(poisoned["raw_sha256"])[0]
    poisoned["raw_sha256"][first] = "0" * 64
    monkeypatch.setattr(R, "source_identity", lambda *a, **k: poisoned)

    with pytest.raises(R.RunnerRefusal) as raised:
        _resume(tmp_path, result)
    message = str(raised.value)
    assert "raw_working_tree_bytes" in message
    assert "working_tree_blob_sha1" not in message
    assert first in message


@pytest.mark.parametrize("key", ["working_tree_blob_sha1", "head_blob_sha1", "head_status"])
def test_each_git_identity_class_is_refused_on_its_own(key, tmp_path, monkeypatch):
    result, _algo, _env = _published(tmp_path)
    current = R.source_identity()
    poisoned = dict(current)
    poisoned[key] = dict(current[key])
    first = sorted(poisoned[key])[0]
    poisoned[key][first] = "tampered"
    monkeypatch.setattr(R, "source_identity", lambda *a, **k: poisoned)

    with pytest.raises(R.RunnerRefusal) as raised:
        _resume(tmp_path, result)
    message = str(raised.value)
    assert key in message
    assert "raw_working_tree_bytes" not in message


def test_the_three_identity_classes_are_recorded_separately_and_never_normalised():
    """Binding, IN-25. Raw bytes, the filtered blob id and the HEAD blob id are
    three different objects; the runner keeps three maps and never reconciles
    them, and never touches ``.gitattributes``."""
    identity = R.source_identity()
    assert set(identity) == {"raw_sha256", "working_tree_blob_sha1",
                             "head_blob_sha1", "head_status"}
    assert set(identity["raw_sha256"]) == set(ppo_resume.SOURCE_FILES)
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "gitattributes" not in source
    assert "autocrlf" not in source.replace("core.autocrlf", "")


def test_a_segment_without_a_git_identity_record_is_refused(tmp_path, monkeypatch):
    """Behavioural. A segment captured in inspection mode carries no
    ``sources_git``, so its provenance cannot be compared and it is refused
    rather than resumed on a partial record."""
    algo, env = _build()
    _start(algo, env)
    algo.collect_rollout()
    algo.update()
    algo.iter_count += 1
    state = ppo_resume.capture_state(algo, env, _extra(tmp_path), epoch=1,
                                     include_git_identity=False)
    result = ppo_resume.write_segment(state, tmp_path / "inspection.ptc")

    with pytest.raises(R.RunnerRefusal, match="sources_git"):
        fresh_algo, fresh_env = _build(seed=5)
        R.resume_boundary(result["path"], fresh_algo, fresh_env, epoch=1, seed=1001,
                          segment_index=0, parent_segment_sha256=None,
                          bindings=_bindings(tmp_path))


# -- provenance: the freeze bindings ---------------------------------------

def test_the_freeze_bindings_are_recorded_in_the_extra(tmp_path):
    result, _algo, _env = _published(tmp_path)
    state = ppo_resume.read_segment(result["path"])
    for key in R.FREEZE_BINDING_KEYS:
        assert key in state.extra, key
    assert state.extra["freeze_sha256"] == R.sha256_file(_freeze(tmp_path))


@pytest.mark.parametrize("key", list(R.FREEZE_BINDING_KEYS))
def test_a_changed_freeze_config_or_model_hash_is_refused(key, tmp_path):
    result, _algo, _env = _published(tmp_path)
    bindings = dict(_bindings(tmp_path))
    bindings[key] = "9" * 64
    with pytest.raises(R.RunnerRefusal, match=key):
        _resume(tmp_path, result, bindings=bindings)


# -- lineage ----------------------------------------------------------------

def test_the_lineage_chains_each_segments_published_sha256(tmp_path):
    """Binding. ``parent_segment_sha256`` for segment n+1 is the previous
    ``write_segment`` result's ``sha256``."""
    first, algo, env = _published(tmp_path, epoch=2, segment_index=0, name="s0.ptc")
    extra = _extra(tmp_path, segment_index=1, parent_segment_sha256=first["sha256"],
                   recorder_state={"returns": [0.0] * env.num_envs,
                                   "lengths": [0] * env.num_envs})
    algo.collect_rollout()
    algo.update()
    algo.iter_count += 1
    second = R.publish_boundary(algo, env, tmp_path / "s1.ptc", epoch=3, extra=extra)

    assert second["sha256"] != first["sha256"]
    state = ppo_resume.read_segment(second["path"])
    assert state.extra["parent_segment_sha256"] == first["sha256"]

    report, _state = R.resume_boundary(
        second["path"], *_build(seed=77), epoch=3, seed=1001, segment_index=1,
        parent_segment_sha256=first["sha256"], bindings=_bindings(tmp_path),
        expected_segment_sha256=second["sha256"])
    assert report["epoch"] == 3


def test_a_parent_file_whose_bytes_changed_is_refused(tmp_path):
    """Behavioural. The recorded lineage sha256 is checked against the file on
    disk, so a swapped parent is caught before anything is restored."""
    result, _algo, _env = _published(tmp_path)
    with pytest.raises(R.RunnerRefusal, match="sha256"):
        _resume(tmp_path, result, expected_sha="a" * 64)


def test_write_segment_never_overwrites_an_existing_boundary(tmp_path):
    result, algo, env = _published(tmp_path)
    extra = _extra(tmp_path, recorder_state={"returns": [0.0] * env.num_envs,
                                             "lengths": [0] * env.num_envs})
    with pytest.raises(FileExistsError):
        R.publish_boundary(algo, env, Path(result["path"]), epoch=2, extra=extra)


def test_the_recorder_accumulators_survive_a_segment_boundary(tmp_path):
    """Binding. An episode may span a segment boundary, so the per-world running
    return and length are carried in the ``extra`` and restored."""
    algo, env = _build()
    _start(algo, env)
    recorder = R.EpisodeEventRecorder(env.num_envs, CONTROL_DT, device="cpu")
    R.train_segment(algo, env, start_epoch=0, end_epoch=1, max_epochs_total=128,
                    clock=TickingClock(tick=0.0), work_deadline=1e9, recorder=recorder,
                    resumed_from_boundary=False, capture_reserve_s=0.0)
    saved = recorder.state()
    extra = _extra(tmp_path, recorder_state=saved)
    result = R.publish_boundary(algo, env, tmp_path / "with_accumulators.ptc",
                                epoch=1, extra=extra)

    state = ppo_resume.read_segment(result["path"])
    assert state.extra["episode_accumulators"] == saved
    revived = R.EpisodeEventRecorder(env.num_envs, CONTROL_DT, device="cpu")
    revived.load(state.extra["episode_accumulators"])
    assert revived.state() == saved


# ==========================================================================
# Unit 5 -- selection: the cause split, exclusion and the two refusal classes
# ==========================================================================
#
# The two refusal classes are handled entirely separately and are never merged
# into one ``except ValueError``. The split between them is keyed on the CAUSE,
# not on the class: ``InvalidEvaluationError`` carries only a free-text reason,
# so the discriminator comes from the record this runner wrote. A wall-cap
# truncation is a scheduling artifact and excludes one checkpoint; every other
# cause is a numerical/structural failure and censors the run.

def _evaluation(*, survival=1.0, duration=4.0, rmse=0.1, band=400, length=500,
                episodes=4, complete=True, control_dt=0.008, end_reason="horizon"):
    return {
        "summary": {"survival_fraction": survival,
                    "mean_first_episode_duration_s": duration,
                    "mean_episode_velocity_2d_rmse_mps": rmse},
        "episodes": [{"length": length, "speed_band_steps": band,
                      "end_reason": end_reason} for _ in range(episodes)],
        "control_dt": control_dt,
        "complete": complete,
    }


def _truncated_evaluation(**overrides):
    """What ``evaluate_policy`` actually produces when its deadline fires: the
    surviving worlds get ``end_reason == "wall_cap"`` and ``complete`` is then
    derived as ``not any(end_reason == "wall_cap")``."""
    evaluation = _evaluation(complete=False, end_reason="wall_cap", **overrides)
    return evaluation


def _candidate(epoch, evaluation, **extra):
    return {"epoch": epoch, "evaluation": evaluation,
            "checkpoint": f"epoch_{epoch:04d}.pt", **extra}


# -- classification ---------------------------------------------------------

def test_a_valid_evaluation_classifies_as_valid():
    status, reason = R.classify_evaluation(_evaluation())
    assert status == R.EVAL_VALID
    assert reason is None


def test_a_wall_cap_truncation_classifies_as_truncated_not_invalid():
    """Binding. A scheduling artifact is not a numerical failure."""
    status, reason = R.classify_evaluation(_truncated_evaluation())
    assert status == R.EVAL_TRUNCATED
    assert "complete" in reason


@pytest.mark.parametrize("evaluation,label", [
    (_evaluation(rmse=float("nan")), "nonfinite metric"),
    (_evaluation(duration=float("inf")), "nonfinite metric"),
    (_evaluation(control_dt=0.0), "non-positive control_dt"),
    (_evaluation(episodes=0), "empty episodes"),
    (_evaluation(length=0), "non-positive episode length"),
    (_evaluation(band=600, length=500), "band steps above length"),
])
def test_every_other_cause_classifies_as_invalid(evaluation, label):
    status, reason = R.classify_evaluation(evaluation)
    assert status == R.EVAL_INVALID, label
    assert reason


def test_a_missing_required_key_classifies_as_invalid():
    evaluation = _evaluation()
    del evaluation["summary"]
    assert R.classify_evaluation(evaluation)[0] == R.EVAL_INVALID


def test_a_record_that_is_both_truncated_and_malformed_classifies_as_invalid():
    """Binding. Truncation is claimed only when it is the **sole** defect, so a
    second real fault is never hidden behind a scheduling artifact."""
    evaluation = _truncated_evaluation()
    evaluation["summary"]["mean_episode_velocity_2d_rmse_mps"] = float("nan")
    assert R.classify_evaluation(evaluation)[0] == R.EVAL_INVALID


def test_classification_never_pattern_matches_the_reason_prose():
    """Binding. The upstream reason is free text; the discriminator is the
    record's own ``end_reason``/``complete`` data."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "wall cap truncated it" not in source
    assert "is not finite" not in source
    assert 'end_reason") == "wall_cap"' in source or 'end_reason"] == "wall_cap"' in source


def test_the_copy_probe_does_not_mutate_the_supplied_evaluation():
    evaluation = _truncated_evaluation()
    before = json.dumps(evaluation, sort_keys=True)
    R.classify_evaluation(evaluation)
    assert json.dumps(evaluation, sort_keys=True) == before


# -- selection over one run -------------------------------------------------

def test_selection_delegates_to_select_best_checkpoint(monkeypatch):
    """Binding. The deterministic earliest-epoch exact-tie rule exists only
    inside ``select_best_checkpoint``; a runner-side ``max`` would resolve an
    exact tie by arrival order."""
    called = {}
    sentinel = _candidate(64, _evaluation())

    def spy(candidates):
        called["candidates"] = list(candidates)
        return sentinel

    monkeypatch.setattr(S, "select_best_checkpoint", spy)
    record = R.select_run([_candidate(0, _evaluation()), _candidate(64, _evaluation())])

    assert record["selected"] is sentinel
    assert len(called["candidates"]) == 2


def test_the_runner_implements_no_ranking_of_its_own():
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "behavior_rank_v2" in source          # used only as a validator
    assert "max(" not in source.replace("max(update_cost", "").replace("max(epoch_cost", "")
    assert "sorted(candidates" not in source
    assert "select_best_checkpoint" in source


def test_the_earliest_epoch_wins_an_exact_tie():
    """Behavioural, through the real helper: identical keys, smaller epoch wins,
    independent of the order the candidates arrive in."""
    late, early = _candidate(64, _evaluation()), _candidate(0, _evaluation())
    assert R.select_run([late, early])["selected"]["epoch"] == 0
    assert R.select_run([early, late])["selected"]["epoch"] == 0


def test_a_better_checkpoint_beats_an_earlier_one():
    weak = _candidate(0, _evaluation(survival=0.25))
    strong = _candidate(64, _evaluation(survival=1.0))
    assert R.select_run([weak, strong])["selected"]["epoch"] == 64


# -- the truncation branch: exclude and continue ----------------------------

def test_a_truncated_checkpoint_is_excluded_and_the_run_continues():
    """Binding, amended ruling. The remaining valid checkpoints still select."""
    truncated = _candidate(0, _truncated_evaluation())
    valid = _candidate(64, _evaluation(survival=0.5))
    record = R.select_run([truncated, valid])

    assert record["status"] == "selected"
    assert record["selected"]["epoch"] == 64
    assert record["wall_cap_truncations"] == 1
    assert [entry["epoch"] for entry in record["excluded"]] == [0]
    assert record["excluded"][0]["status"] == R.EVAL_TRUNCATED


def test_an_exclusion_is_legible_as_an_exclusion_not_a_full_field():
    """Binding. A selection made from 2 of 5 must never read as if all 5
    competed."""
    candidates = [_candidate(0, _truncated_evaluation()),
                  _candidate(64, _truncated_evaluation()),
                  _candidate(128, _truncated_evaluation()),
                  _candidate(192, _evaluation(survival=0.5)),
                  _candidate(256, _evaluation(survival=0.9))]
    record = R.select_run(candidates)

    assert record["candidates_considered"] == 5
    assert record["candidates_compared"] == 2
    assert record["excluded_count"] == 3
    assert sorted(entry["epoch"] for entry in record["excluded"]) == [0, 64, 128]
    assert all(entry["reason"] for entry in record["excluded"])
    assert record["selected"]["epoch"] == 256


def test_no_excluded_checkpoint_is_scored_zero_defaulted_or_silently_dropped():
    candidates = [_candidate(0, _truncated_evaluation()), _candidate(64, _evaluation())]
    record = R.select_run(candidates)
    statuses = {entry["epoch"]: entry for entry in record["checkpoint_status"]}

    assert set(statuses) == {0, 64}                       # nothing dropped
    assert statuses[0]["status"] == R.EVAL_TRUNCATED
    assert "score" not in statuses[0]
    assert "rank" not in statuses[0]
    assert statuses[0].get("excluded") is True


def test_every_checkpoint_excluded_means_an_incomplete_run_not_a_zero():
    """Binding. No fallback to an excluded checkpoint, and no zero score. This
    also never becomes a set-level fault: an empty survivor list is a legitimate
    outcome, unlike an empty candidate list."""
    candidates = [_candidate(0, _truncated_evaluation()),
                  _candidate(64, _truncated_evaluation())]
    record = R.select_run(candidates)

    assert record["status"] == "incomplete"
    assert record["selected"] is None
    assert record["candidates_compared"] == 0
    assert record["wall_cap_truncations"] == 2
    assert "fault" not in record


def test_the_truncation_count_is_surfaced_without_touching_any_deadline():
    record = R.select_run([_candidate(0, _truncated_evaluation()),
                           _candidate(64, _evaluation())])
    assert record["wall_cap_truncations"] == 1
    # Structural: a declared deadline is computed once and never retuned.
    _assert_no_deadline_retuning(ast.parse(Path(R.__file__).read_text(encoding="utf-8")))


# -- the censoring branch ---------------------------------------------------

def test_a_nonfinite_metric_censors_the_run_and_names_the_checkpoint():
    """Binding, amended ruling. A genuinely invalid evaluation is a censoring
    event, not a candidate to be quietly skipped."""
    candidates = [_candidate(0, _evaluation()),
                  _candidate(64, _evaluation(rmse=float("nan")))]
    with pytest.raises(R.CensorRun) as raised:
        R.select_run(candidates)
    message = str(raised.value)
    assert "64" in message
    assert "epoch_0064.pt" in message


def test_no_valid_checkpoint_is_selected_while_an_invalid_sibling_exists():
    """Binding. The abort is not "select among the survivors"."""
    candidates = [_candidate(0, _evaluation(survival=1.0)),
                  _candidate(64, _evaluation(control_dt=-1.0))]
    with pytest.raises(R.CensorRun):
        R.select_run(candidates)


def test_the_censor_is_not_reused_as_the_partial_epoch_label():
    algo, env = _build(steps_num=8)
    _start(algo, env)
    clock = TickingClock(tick=1.0)
    outcome = _run(algo, env, epochs=4, clock=clock, deadline=6.0)
    assert outcome.stop_reason == "work_deadline_partial_epoch"
    assert outcome.censored is False


# -- the set-level fault ----------------------------------------------------

def test_pooling_two_seeds_epoch_zero_is_a_runner_fault():
    """Binding. Every seed and arm has its own epoch 0, so pooling produces a
    duplicate epoch. That refusal is the guard working: it is never routed
    around by renumbering, suffixing a composite key or deduplicating."""
    pooled = [_candidate(0, _evaluation(), seed=1001), _candidate(0, _evaluation(), seed=1002)]
    with pytest.raises(R.RunnerFault) as raised:
        R.select_run(pooled)
    assert "duplicate" in str(raised.value).lower()


@pytest.mark.parametrize("candidates,label", [
    ([{"evaluation": _evaluation()}], "missing epoch"),
    ([{"epoch": "0", "evaluation": _evaluation()}], "non-integer epoch"),
    ([{"epoch": -1, "evaluation": _evaluation()}], "negative epoch"),
    (["not a mapping"], "non-mapping candidate"),
    ([{"epoch": 0}], "candidate without an evaluation"),
    ([], "empty set"),
    ("string", "non-sequence collection"),
    ({"epoch": 0}, "mapping instead of a sequence"),
])
def test_every_set_level_fault_aborts_as_a_runner_fault(candidates, label):
    with pytest.raises(R.RunnerFault):
        R.select_run(candidates)


def test_a_set_level_fault_never_appears_in_a_checkpoint_status_field():
    """Binding. Filing a runner bug as "invalid checkpoint" would silently drop a
    real checkpoint from the campaign record while the selection looked
    complete."""
    pooled = [_candidate(0, _evaluation(), seed=1001), _candidate(0, _evaluation(), seed=1002)]
    with pytest.raises(R.RunnerFault) as raised:
        R.select_run(pooled)

    record = raised.value.record
    assert record["status"] == "runner_fault"
    assert record["fault"]["reason"]
    assert record["selected"] is None
    statuses = record["checkpoint_status"]
    assert len(statuses) == 2                     # neither checkpoint was dropped
    for entry in statuses:
        assert entry["status"] != R.EVAL_INVALID
        assert entry.get("reason") != record["fault"]["reason"]
        assert "duplicate" not in str(entry.get("reason") or "").lower()


def test_the_two_refusal_classes_are_not_interchangeable():
    """Binding. Neither is a subclass of the other, and they are never caught in
    one ``except ValueError``."""
    assert not issubclass(S.InvalidCandidateSetError, S.InvalidEvaluationError)
    assert not issubclass(S.InvalidEvaluationError, S.InvalidCandidateSetError)

    with pytest.raises(S.InvalidCandidateSetError):
        try:
            S.select_best_checkpoint([])
        except S.InvalidEvaluationError:                     # must NOT catch it
            raise AssertionError("a set-level fault was caught as an evaluation fault")

    with pytest.raises(S.InvalidEvaluationError):
        try:
            S.behavior_rank_v2(_evaluation(rmse=float("nan")))
        except S.InvalidCandidateSetError:                   # must NOT catch it
            raise AssertionError("an evaluation fault was caught as a set-level fault")

    # Structural, not textual: every except handler in the runner is inspected,
    # so a prose mention of the forbidden pattern cannot pass or fail this.
    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    handlers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            targets = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            handlers.append([ast.unparse(target) for target in targets])
    assert handlers
    for names in handlers:
        assert "ValueError" not in names, names
        assert not {"S.InvalidEvaluationError", "S.InvalidCandidateSetError"} <= set(names), names


# -- candidates come from one run only -------------------------------------

def test_candidates_are_collected_from_a_single_run_directory(tmp_path):
    run = tmp_path / "run"
    for index, epoch in enumerate((0, 64)):
        segment = run / f"segment_{index:04d}"
        segment.mkdir(parents=True)
        R.write_json_exclusive(segment / f"selection_{epoch:04d}.json", {
            "epoch": epoch, "kind": "selection", "evaluation": _evaluation(),
            "checkpoint": f"epoch_{epoch:04d}.pt"})
        R.write_json_exclusive(segment / "result.json", {
            "schema_version": R.RESULT_SCHEMA, "stage": "screen",
            "recipe": "g990_e010", "seed": 1001, "accounting": _counters()})

    candidates = R.collect_candidates(run)
    assert [candidate["epoch"] for candidate in candidates] == [0, 64]
    assert all(candidate["evaluation"]["complete"] for candidate in candidates)
    assert R.select_run(candidates)["selected"]["epoch"] == 0


def test_a_run_directory_mixing_two_seeds_is_a_runner_fault(tmp_path):
    run = tmp_path / "run"
    for index, (epoch, seed) in enumerate(((0, 1001), (64, 1002))):
        segment = run / f"segment_{index:04d}"
        segment.mkdir(parents=True)
        R.write_json_exclusive(segment / f"selection_{epoch:04d}.json", {
            "epoch": epoch, "kind": "selection", "evaluation": _evaluation(),
            "checkpoint": f"epoch_{epoch:04d}.pt"})
        R.write_json_exclusive(segment / "result.json", {
            "schema_version": R.RESULT_SCHEMA, "stage": "screen",
            "recipe": "g990_e010", "seed": seed, "accounting": _counters()})

    with pytest.raises(R.RunnerFault, match="seed"):
        R.collect_candidates(run)


def test_select_mode_writes_its_record_exclusively(tmp_path):
    run = tmp_path / "run"
    segment = run / "segment_0000"
    segment.mkdir(parents=True)
    R.write_json_exclusive(segment / "selection_0000.json", {
        "epoch": 0, "kind": "selection", "evaluation": _evaluation(),
        "checkpoint": "epoch_0000.pt"})
    R.write_json_exclusive(segment / "result.json", {
        "schema_version": R.RESULT_SCHEMA, "stage": "screen", "recipe": "g990_e010",
        "seed": 1001, "accounting": _counters()})

    args = _parse(["select", "--run-dir", str(run)])
    assert R.run_select(args) == R.EXIT_OK
    record = R.read_json(run / "selection.json")
    assert record["schema_version"] == R.SELECTION_SCHEMA
    assert record["selected"]["epoch"] == 0

    assert R.run_select(args) == R.EXIT_REFUSED       # never overwritten


# ==========================================================================
# Unit 6 -- worker wiring: the publish decision, the result record, replay
# ==========================================================================
#
# The worker is driven through an injected build seam, so these tests exercise
# the whole worker contract on the analytic CPU adapter with no simulator, no
# CUDA device and no training seconds. The real GPU build path is deliberately
# not exercised here; that is a declared limitation of this unit.

def _worker_argv(tmp_path, out_dir, **overrides):
    values = {
        "--stage": "screen", "--recipe": "g990_e010", "--seed": "1001",
        "--segment-index": "0", "--freeze": str(_freeze(tmp_path)),
        "--out-dir": str(out_dir), "--epochs": "3",
        "--work-deadline-s": "460", "--capture-reserve-s": "0",
    }
    for key, value in overrides.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = str(value)
    argv = ["worker"]
    for key, value in values.items():
        argv += [key, value]
    return argv


def _build_seam(*, seed=7, steps_num=4, num_envs=4, evaluate=True, mutate=None):
    def build(args, plan):
        algo, env = _build(seed=seed, steps_num=steps_num, num_envs=num_envs)
        if mutate is not None:
            mutate(algo, env)

        def evaluation_hook(epoch, deadline=None):
            record = {"epoch": int(epoch), "kind": "selection",
                      "evaluation": _evaluation(),
                      "checkpoint": f"epoch_{epoch:04d}.pt"}
            counter = {"attempted_calls": 10, "completed_calls": 10, "worlds": 16}
            return record, counter

        return R.WorkerContext(
            algo=algo, env=env, control_dt=CONTROL_DT, max_epochs_total=128,
            bindings=R.freeze_bindings(args.freeze),
            evaluate=evaluation_hook if evaluate else None)

    return build


def _run_worker(tmp_path, out_dir, *, clock=None, build=None, **overrides):
    out_dir.mkdir(parents=True, exist_ok=True)
    args = _parse(_worker_argv(tmp_path, out_dir, **overrides))
    return R.run_worker(args, build=build or _build_seam(),
                        clock=clock or TickingClock(tick=0.0))


# -- the clean path ---------------------------------------------------------

def test_a_clean_worker_publishes_one_boundary_and_one_result(tmp_path):
    out = tmp_path / "run" / "segment_0000"
    code = _run_worker(tmp_path, out)

    assert code == R.EXIT_OK
    result = R.read_json(out / "result.json")
    assert result["schema_version"] == R.RESULT_SCHEMA
    assert result["completed_epochs"] == 3
    assert result["completed_epoch"] == 3
    assert result["boundary_is_live"] is True
    assert result["published"] is True
    assert (out / "segment.ptc").is_file()
    assert result["lineage"]["segment_sha256"] == R.sha256_file(out / "segment.ptc")
    assert result["lineage"]["parent_segment_sha256"] is None
    assert result["lineage"]["valid_boundary_epoch"] == 3


def test_the_result_records_the_identity_needed_to_scope_selection(tmp_path):
    out = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, out)
    result = R.read_json(out / "result.json")
    assert (result["stage"], result["recipe"], result["seed"]) == ("screen", "g990_e010", 1001)
    assert result["segment_index"] == 0


def test_the_accounting_adds_up_and_separates_every_class(tmp_path):
    out = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, out)
    accounting = R.read_json(out / "result.json")["accounting"]

    assert accounting["attempted_control_transitions"] == (
        accounting["completed_control_transitions"]
        + accounting["discarded_control_transitions"])
    assert accounting["discarded_control_transitions"] == 0
    assert accounting["replayed_control_transitions"] == 0
    assert accounting["attempted_physics_steps"] == (
        accounting["attempted_control_transitions"] * P.PHYSICS_SUBSTEPS)
    assert accounting["partial_update_cost_s"] == 0.0
    # Training and evaluation are both real physics and both counted.
    assert accounting["training_control_transitions"] == 3 * 4 * 4
    assert accounting["evaluation_control_transitions"] > 0


def test_the_worker_never_touches_the_budget_ledger(tmp_path):
    """Binding. Exactly one process writes the ledger, and it is not this one."""
    out = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, out)
    assert list(tmp_path.rglob("*.jsonl")) == []
    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    worker = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "run_worker")
    text = ast.unparse(worker)
    for forbidden in ("BudgetLedger", "reserve(", "mark_spawn", "settle"):
        assert forbidden not in text, forbidden


def test_an_evaluation_is_written_for_each_sealed_evaluation_epoch(tmp_path):
    out = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, out)
    written = sorted(path.name for path in out.glob("selection_*.json"))
    assert written == ["selection_0000.json"]
    record = R.read_json(out / "selection_0000.json")
    assert record["epoch"] == 0
    assert record["evaluation"]["complete"] is True


def test_the_stage_selection_reset_block_is_used_and_the_locked_blocks_are_not(tmp_path):
    """Binding. Confirmation (21000-21031) and audit (31000-31015) stay unused
    until checkpoint selection is locked."""
    out = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, out)
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "selection_reset_block" in source
    for locked in ("21000", "31000", "confirmation", "audit"):
        assert f'V2_RESET_BLOCKS["{locked}"]' not in source
    assert "CONFIRMATION_EPISODES" not in source
    assert "AUDIT_EPISODES" not in source


def test_outputs_are_exclusive_and_a_second_worker_run_is_refused(tmp_path):
    out = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, out)
    with pytest.raises(R.RunnerRefusal, match="result"):
        _run_worker(tmp_path, out)


def test_a_missing_output_directory_is_refused_since_the_launcher_owns_it(tmp_path):
    args = _parse(_worker_argv(tmp_path, tmp_path / "absent"))
    with pytest.raises(R.RunnerRefusal, match="output directory"):
        R.run_worker(args, build=_build_seam(), clock=TickingClock(tick=0.0))


# -- the partial path -------------------------------------------------------

def test_a_partial_epoch_publishes_nothing_and_keeps_the_parent_boundary(tmp_path):
    """Binding. On interruption the prior valid boundary is retained, the
    partial work is censored, and no segment file appears at all."""
    out = tmp_path / "run" / "segment_0000"
    code = _run_worker(tmp_path, out, clock=TickingClock(tick=1.0),
                       build=_build_seam(steps_num=8),
                       **{"--work-deadline-s": "6", "--epochs": "4", "--capture-reserve-s": "0"})

    assert code == R.EXIT_INTERRUPTED
    result = R.read_json(out / "result.json")
    assert result["published"] is False
    assert result["boundary_is_live"] is False
    assert not (out / "segment.ptc").exists()
    assert result["lineage"]["segment_sha256"] is None
    assert result["lineage"]["valid_boundary_epoch"] == 0
    assert result["stop_reason"] == "work_deadline_partial_epoch"

    accounting = result["accounting"]
    # No training epoch survives, so no training transition is retained. The
    # epoch-0 evaluation was written and IS retained, so it is not discarded.
    assert accounting["training_boundary_control_transitions"] == 0
    assert accounting["discarded_control_transitions"] == (
        accounting["training_control_transitions"])
    assert accounting["discarded_control_transitions"] > 0
    assert accounting["completed_control_transitions"] == (
        accounting["evaluation_control_transitions"])


def test_the_next_segment_records_the_discarded_work_as_replayed(tmp_path):
    """Binding. Work redone after an interruption is real cost: it is recorded
    by the segment that redoes it, never netted off."""
    first = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, first, clock=TickingClock(tick=1.0),
                build=_build_seam(steps_num=8),
                **{"--work-deadline-s": "6", "--epochs": "4", "--capture-reserve-s": "0"})
    discarded = R.read_json(first / "result.json")["accounting"]
    assert discarded["discarded_control_transitions"] > 0

    # A published parent for the follow-on segment to resume from. It runs the
    # full sealed round 0, because segment 1 starts at epoch 64 by construction.
    parent = tmp_path / "run" / "segment_0000b"
    _run_worker(tmp_path, parent, build=_build_seam(), **{"--epochs": "64"})
    parent_result = parent / "result.json"

    second = tmp_path / "run" / "segment_0001"
    code = _run_worker(
        tmp_path, second, build=_build_seam(),
        **{"--segment-index": "1", "--epochs": "2",
           "--parent-segment": str(parent / "segment.ptc"),
           "--parent-result": str(parent_result),
           "--replayed-control-transitions": str(discarded["discarded_control_transitions"]),
           "--replayed-physics-steps": str(discarded["discarded_physics_steps"])})

    assert code == R.EXIT_OK
    accounting = R.read_json(second / "result.json")["accounting"]
    assert accounting["replayed_control_transitions"] == discarded["discarded_control_transitions"]
    assert accounting["replayed_physics_steps"] == discarded["discarded_physics_steps"]


def test_a_resumed_segment_chains_its_parents_published_sha256(tmp_path):
    parent = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, parent, **{"--epochs": "64"})
    parent_sha = R.read_json(parent / "result.json")["lineage"]["segment_sha256"]

    second = tmp_path / "run" / "segment_0001"
    code = _run_worker(tmp_path, second, **{
        "--segment-index": "1", "--epochs": "2",
        "--parent-segment": str(parent / "segment.ptc"),
        "--parent-result": str(parent / "result.json")})

    assert code == R.EXIT_OK
    result = R.read_json(second / "result.json")
    assert result["lineage"]["parent_segment_sha256"] == parent_sha
    assert result["resume"]["epoch"] == 64
    assert "cuda_rng_restored" in result["resume"]


def test_a_resume_whose_parent_bytes_changed_is_refused_before_training(tmp_path):
    parent = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, parent, **{"--epochs": "64"})
    (parent / "segment.ptc").write_bytes(b"not a segment")

    second = tmp_path / "run" / "segment_0001"
    with pytest.raises(R.RunnerRefusal):
        _run_worker(tmp_path, second, **{
            "--segment-index": "1", "--epochs": "2",
            "--parent-segment": str(parent / "segment.ptc"),
            "--parent-result": str(parent / "result.json")})
    assert not (second / "segment.ptc").exists()


# -- the censor path --------------------------------------------------------

def test_a_censored_worker_records_the_reason_and_publishes_nothing(tmp_path):
    def poison(algo, env):
        algo.update = lambda: {"actor_loss": float("nan"), "value_loss": 0.0, "entropy": 0.0}

    out = tmp_path / "run" / "segment_0000"
    code = _run_worker(tmp_path, out, build=_build_seam(mutate=poison))

    assert code == R.EXIT_CENSORED
    result = R.read_json(out / "result.json")
    assert result["censored"] is True
    assert "loss" in result["censor_reason"]
    assert result["published"] is False
    assert not (out / "segment.ptc").exists()
    # The result still exists, so the launcher can settle the measured wall.
    assert result["accounting"]["attempted_control_transitions"] > 0


def test_a_censored_worker_still_reports_counters_for_settlement(tmp_path):
    def poison(algo, env):
        algo.update = lambda: {"actor_loss": float("inf"), "value_loss": 0.0, "entropy": 0.0}

    out = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, out, build=_build_seam(mutate=poison))
    assert R._counters_from_result(out / "result.json") is not None


# -- end to end through the launcher ---------------------------------------

def test_the_launcher_settles_a_real_worker_result(tmp_path, monkeypatch):
    """Behavioural. The two modes meet: a fake spawn runs the worker in-process
    on the CPU adapter, and the launcher settles its measured counters."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    out = tmp_path / "run" / "segment_0000"

    def fake_spawn(argv, **kwargs):
        clock.advance(7.5)
        args = R.build_parser().parse_args(argv[2:])
        code = R.run_worker(args, build=_build_seam(), clock=TickingClock(tick=0.0))
        return FakeProcess(returncode=code)

    args = _parse(_launch_argv(tmp_path, **{"--epochs": "2"}))
    code = R.run_launch(args, ledger=ledger, spawn=fake_spawn, clock=clock)

    assert code == R.EXIT_OK
    assert _kinds(ledger) == ["header", "reserve", "spawn", "settle"]
    settle = ledger.rows[-1]
    assert settle["charge_basis"] == B.CHARGE_ACTUAL
    assert settle["charged_s"] == pytest.approx(7.5)
    assert settle["completed_control_transitions"] > 0
    assert settle["replayed_control_transitions"] == 0
    assert (out / "segment.ptc").is_file()
    assert R.read_json(out / "child.json")["returncode"] == 0


# ==========================================================================
# Fix round 1, F1 -- a pre-training refusal must not cost the full bound
# ==========================================================================
#
# A child that refuses deterministically in about a second, having provably done
# no training, was charged the full reserved bound (~540 s). Roughly 13 such
# refusals would exhaust the 7,200 s stage-1 cap, and early-campaign refusals --
# bad arguments, identity mismatches, cap boundaries -- are the expected case.
#
# The cheap charge is gated exactly like the spawn marker: it needs POSITIVE
# evidence that no training ran. The worker writes an explicit marker with every
# counter zero; the parent then charges its OWN measured wall. Absent, or present
# without a consistent marker, still charges the full reserved bound.

def _refusal_result(out_dir, **overrides):
    payload = {"schema_version": R.RESULT_SCHEMA,
               "status": R.REFUSED_BEFORE_TRAINING,
               "training_began": False,
               "refusal": "RunnerRefusal: deliberate",
               "accounting": _counters(**{key: 0 for key in _counters()})}
    payload["accounting"]["partial_update_cost_s"] = 0.0
    payload.update(overrides)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


# -- the worker side --------------------------------------------------------

def test_a_pre_training_refusal_writes_a_zero_counter_marker(tmp_path):
    """Behavioural. The refusal record is the positive evidence the parent needs."""
    parent = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, parent, **{"--epochs": "64"})
    # A parent result that published nothing is a refusal cause reached well
    # before any training in the follow-on segment.
    broken = tmp_path / "run" / "broken.json"
    broken.write_text(json.dumps({"schema_version": R.RESULT_SCHEMA, "segment_index": 0,
                                  "lineage": {"segment_sha256": None}}), encoding="utf-8")

    second = tmp_path / "run" / "segment_0001"
    with pytest.raises(R.RunnerRefusal):
        _run_worker(tmp_path, second, **{
            "--segment-index": "1", "--epochs": "2",
            "--parent-segment": str(parent / "segment.ptc"),
            "--parent-result": str(broken)})

    record = R.read_json(second / "result.json")
    assert record["status"] == R.REFUSED_BEFORE_TRAINING
    assert record["training_began"] is False
    assert record["refusal"]
    accounting = record["accounting"]
    for key in ("attempted_control_transitions", "completed_control_transitions",
                "replayed_control_transitions", "attempted_physics_steps",
                "completed_physics_steps", "replayed_physics_steps"):
        assert accounting[key] == 0, key
    assert accounting["partial_update_cost_s"] == 0.0
    assert not (second / "segment.ptc").exists()


def test_a_refusal_after_training_began_never_carries_the_cheap_marker(tmp_path):
    """Binding. The cheap path must be unreachable once any epoch has run."""
    def poison(algo, env):
        original = algo.update

        def refuse():
            original()
            raise R.RunnerRefusal("a refusal raised after training began")

        algo.update = refuse

    out = tmp_path / "run" / "segment_0000"
    code = _run_worker(tmp_path, out, build=_build_seam(mutate=poison))

    record = R.read_json(out / "result.json")
    assert code == R.EXIT_CENSORED
    assert record["training_began"] is True
    assert record.get("status") != R.REFUSED_BEFORE_TRAINING
    assert record["accounting"]["attempted_control_transitions"] > 0
    assert R._settlement_from_result(out / "result.json")[1] == "trained"


def test_a_completed_worker_declares_that_training_began(tmp_path):
    out = tmp_path / "run" / "segment_0000"
    _run_worker(tmp_path, out)
    record = R.read_json(out / "result.json")
    assert record["training_began"] is True
    assert record.get("status") != R.REFUSED_BEFORE_TRAINING


def test_a_refusal_with_no_output_directory_writes_nothing(tmp_path):
    """Behavioural. Nothing can be written there, so the parent charges the bound."""
    args = _parse(_worker_argv(tmp_path, tmp_path / "absent"))
    with pytest.raises(R.RunnerRefusal, match="output directory"):
        R.run_worker(args, build=_build_seam(), clock=TickingClock(tick=0.0))
    assert not (tmp_path / "absent").exists()


# -- the parent's settlement decision --------------------------------------

def test_a_marked_refusal_settles_the_parents_measured_wall(tmp_path):
    """Binding. The known-cost refusal costs what it cost, not the bound."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)

    def child(argv, kwargs):
        clock.advance(1.25)
        _refusal_result(tmp_path / "run" / "segment_0000")

    spawner = RecordingSpawner(process=FakeProcess(returncode=R.EXIT_REFUSED),
                               on_spawn=child)
    code = _launch(tmp_path, ledger, spawner, clock)

    settle = ledger.rows[-1]
    assert code == R.EXIT_REFUSED
    assert settle["charge_basis"] == B.CHARGE_ACTUAL
    assert settle["charged_s"] == pytest.approx(1.25)
    assert settle["charged_s"] < settle["reserved_bound_s"]
    assert settle["completed_control_transitions"] == 0
    assert settle["attempted_control_transitions"] == 0


def test_thirteen_marked_refusals_no_longer_exhaust_a_stage(tmp_path):
    """Behavioural, the exposure the review quantified: ~13 refusals used to
    consume the whole 7,200 s stage-1 cap."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)
    for index in range(13):
        out = tmp_path / f"run{index}" / "segment_0000"

        def child(argv, kwargs, out=out):
            clock.advance(1.0)
            _refusal_result(out)

        spawner = RecordingSpawner(process=FakeProcess(returncode=R.EXIT_REFUSED),
                                   on_spawn=child)
        _launch(tmp_path, ledger, spawner, clock,
                **{"--run-dir": str(tmp_path / f"run{index}")})

    charged = ledger.charged()
    assert charged["settled_rows"] == 13
    assert charged["global_s"] == pytest.approx(13.0)
    assert charged["global_s"] < P.stage("screen").wall_cap_s
    remaining = ledger.remaining("screen", "g990_e010", 1001)
    assert remaining.least > 1000.0


@pytest.mark.parametrize("overrides,label", [
    ({"training_began": True}, "marker contradicted by training_began"),
    ({"accounting": _counters(attempted_control_transitions=8192)},
     "marker contradicted by nonzero counters"),
    ({"status": "something_else"}, "marker replaced"),
    ({"status": None}, "marker absent"),
])
def test_a_marker_contradicted_by_its_own_record_cannot_buy_the_cheap_settle(
        overrides, label, tmp_path):
    """Binding. A forged or inconsistent marker settles the full reserved bound."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)

    def child(argv, kwargs):
        clock.advance(1.0)
        _refusal_result(tmp_path / "run" / "segment_0000", **overrides)

    spawner = RecordingSpawner(process=FakeProcess(returncode=R.EXIT_REFUSED),
                               on_spawn=child)
    _launch(tmp_path, ledger, spawner, clock)

    settle = ledger.rows[-1]
    assert settle["charge_basis"] == B.CHARGE_RESERVED_BOUND, label
    assert settle["charged_s"] == settle["reserved_bound_s"]


def test_a_trained_record_without_the_training_began_declaration_charges_the_bound(tmp_path):
    """Binding. The two admissible shapes are disjoint; anything else is unknown."""
    clock = FakeClock()
    ledger = _new_ledger(tmp_path, clock)

    def child(argv, kwargs):
        clock.advance(5.0)
        out = tmp_path / "run" / "segment_0000"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(
            json.dumps({"schema_version": R.RESULT_SCHEMA, "accounting": _counters()}),
            encoding="utf-8")

    spawner = RecordingSpawner(on_spawn=child)
    _launch(tmp_path, ledger, spawner, clock)
    assert ledger.rows[-1]["charge_basis"] == B.CHARGE_RESERVED_BOUND


@pytest.mark.parametrize("shape,expected", [
    ("trained", "trained"),
    ("refused", "refused_before_training"),
])
def test_the_settlement_classifier_recognises_exactly_two_shapes(shape, expected, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    if shape == "trained":
        _write_result(out)
    else:
        _refusal_result(out)
    counters, kind = R._settlement_from_result(out / "result.json")
    assert kind == expected
    assert counters is not None


def test_an_absent_or_unparseable_result_is_unknown(tmp_path):
    assert R._settlement_from_result(tmp_path / "missing.json") == (None, "unknown")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert R._settlement_from_result(broken) == (None, "unknown")


# ==========================================================================
# Fix round 1, F2 -- the neutrality snapshot must cover Adam's own state
# ==========================================================================
#
# Capturing only ``param_groups`` would let a mutation of ``exp_avg``,
# ``exp_avg_sq`` or ``step`` pass undetected, which is a genuine weakening
# relative to both the brief ("both optimizer state_dicts") and the Task 4a
# precedent. The snapshot now deep-copies the whole state_dict and compares it
# tensor-aware, and the two tests below keep that non-vacuous.

def test_the_neutrality_snapshot_captures_the_full_optimizer_state_dicts():
    """Binding, with anti-vacuity: the captured Adam state must be non-empty."""
    algo, env = _build()
    _start(algo, env)
    _run(algo, env, epochs=1)
    snapshot = _neutrality_snapshot(algo)

    for key in ("actor_opt", "critic_opt"):
        captured = snapshot[key]
        assert isinstance(captured, dict), f"{key} must keep the whole state_dict"
        assert {"state", "param_groups"} <= set(captured), key
        state = captured["state"]
        assert state, f"{key} Adam state is empty, so the comparison would be vacuous"
        names = {name for entry in state.values() for name in entry}
        assert {"exp_avg", "exp_avg_sq"} <= names, names
        assert any(torch.is_tensor(value) and value.numel()
                   for entry in state.values() for value in entry.values())


@pytest.mark.parametrize("which", ["actor", "critic"])
def test_the_comparator_detects_an_injected_adam_moment_perturbation(which):
    """Anti-vacuity for the comparator itself, not just for the snapshot."""
    algo, env = _build()
    _start(algo, env)
    _run(algo, env, epochs=1)
    before = _neutrality_snapshot(algo)
    optimizer = algo.actor_optimizer if which == "actor" else algo.critic_optimizer
    with torch.no_grad():
        next(iter(optimizer.state.values()))["exp_avg"] += 1.0
    with pytest.raises(AssertionError):
        _assert_snapshots_equal(before, _neutrality_snapshot(algo))


def test_the_comparator_detects_an_injected_adam_step_perturbation():
    algo, env = _build()
    _start(algo, env)
    _run(algo, env, epochs=1)
    before = _neutrality_snapshot(algo)
    entry = next(iter(algo.actor_optimizer.state.values()))
    with torch.no_grad():
        entry["step"] = entry["step"] + 1
    with pytest.raises(AssertionError):
        _assert_snapshots_equal(before, _neutrality_snapshot(algo))


def test_the_snapshot_is_a_copy_so_later_training_cannot_backdate_it():
    """Behavioural. A shallow capture would compare the live object with itself."""
    algo, env = _build()
    _start(algo, env)
    _run(algo, env, epochs=1)
    before = _neutrality_snapshot(algo)
    _run(algo, env, epochs=1)
    with pytest.raises(AssertionError):
        _assert_snapshots_equal(before, _neutrality_snapshot(algo))


def _assert_no_deadline_retuning(tree):
    """Reject every shape an auto-adjusted deadline could take.

    Applied both to the runner (which must pass) and to a synthetic forbidden
    snippet (which must fail), so the guard cannot quietly become vacuous.
    """
    augmented = 0
    assigned = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign):
            augmented += 1
            assert "deadline" not in ast.unparse(node.target), ast.unparse(node)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = ast.unparse(target)
                if "deadline" in name:
                    assigned += 1
                    assert "work_deadline" in name, name
                    # A deadline may be *computed* from the declared budget, but
                    # never from itself: `work_deadline = work_deadline + 60` is
                    # the same auto-adjustment as `+=`, spelled differently.
                    rhs = ast.unparse(node.value)
                    referenced = {
                        child.id for child in ast.walk(node.value)
                        if isinstance(child, ast.Name) and "deadline" in child.id}
                    assert not referenced, f"{name} = {rhs}"
    assert augmented > 0 and assigned > 0, "the guard must have something to check"


def test_the_deadline_guard_rejects_a_self_referential_reassignment():
    """RED for F3: ``work_deadline = work_deadline + 60`` is exactly the
    auto-adjustment the rule forbids, and the guard must catch it."""
    forbidden = ast.parse(textwrap.dedent(
        """
        counter += 1
        work_deadline = work_deadline + 60
        """))
    with pytest.raises(AssertionError):
        _assert_no_deadline_retuning(forbidden)


def test_the_deadline_guard_rejects_an_augmented_deadline_assignment():
    forbidden = ast.parse(textwrap.dedent(
        """
        work_deadline = 1.0
        work_deadline += 60
        """))
    with pytest.raises(AssertionError):
        _assert_no_deadline_retuning(forbidden)


def test_the_deadline_guard_is_not_vacuous_on_an_empty_module():
    with pytest.raises(AssertionError):
        _assert_no_deadline_retuning(ast.parse(""))


# ==========================================================================
# Fix round 1, authorized upstream fix -- the Task 3 sys.modules leak
# ==========================================================================
#
# DELIBERATE edit to a merged Task 3 file, authorized by the parent in fix round
# 1 and called out as such: it is test-only, three lines, and it disarms a
# landmine whose failure mode is the worst a suite can have -- a silently vacuous
# patch. Any later test that lazily imports an ``msk_warp.analysis`` module and
# patches it would otherwise patch a stale object the code under test never
# touches, and pass while asserting nothing.

def test_the_upstream_purge_test_restores_sys_modules():
    """Binding. The Task 3 purge test must put back what it removes."""
    import inspect
    import sys as _sys

    import tests.unit.test_ppo_resume_state as upstream

    target = upstream.test_analysis_package_does_not_import_ppo_resume
    before = {name for name in _sys.modules if name.startswith("msk_warp.analysis")}
    assert before, "the analysis package must be imported for this to mean anything"

    patch = pytest.MonkeyPatch()
    try:
        if inspect.signature(target).parameters:
            target(patch)
        else:
            target()
    finally:
        patch.undo()

    after = {name for name in _sys.modules if name.startswith("msk_warp.analysis")}
    missing = before - after
    assert not missing, f"the purge was never restored; still absent: {sorted(missing)}"


def test_the_late_patch_resolution_still_holds_after_the_upstream_fix(tmp_path, monkeypatch):
    """Defence in depth. The upstream leak is fixed, but late resolution is kept:
    it is correct regardless, and it is what makes these tests order-independent."""
    live = _live_resume()
    seen = {}
    original = live.capture_state

    def spy(algo, env, extra=None, **kwargs):
        seen.update(kwargs)
        return original(algo, env, extra, **kwargs)

    monkeypatch.setattr(live, "capture_state", spy)
    _published(tmp_path)
    assert seen["include_git_identity"] is True


# ==========================================================================
# Fix round 2 -- a refusal record must not poison selection for its run
# ==========================================================================
#
# The F1 fix created a ``result.json`` where none previously existed. That record
# declares no ``(stage, recipe, seed)`` ON PURPOSE -- ``resolve_plan`` may be the
# very thing that refused, so any identity written there would be unvalidated,
# and a *wrong* identity is worse than none. But ``collect_candidates`` added one
# identity per result file, so a refusal contributed ``(None, None, None)``, the
# cross-run guard fired as a false positive, and building its message over a set
# mixing ``str`` and ``None`` raised an uncaught ``TypeError``.
#
# The fix is consumer-side: a record that positively declares itself a
# pre-training refusal contributes no identity and no candidate, and is recorded
# as a non-candidate. The guard stays strict for everything else.

def _trained_segment(run, index, epoch, *, stage="screen", recipe="g990_e010",
                     seed=1001, **result_overrides):
    segment = run / f"segment_{index:04d}"
    segment.mkdir(parents=True)
    R.write_json_exclusive(segment / f"selection_{epoch:04d}.json", {
        "epoch": epoch, "kind": "selection", "evaluation": _evaluation(),
        "checkpoint": f"epoch_{epoch:04d}.pt"})
    record = {"schema_version": R.RESULT_SCHEMA, "training_began": True,
              "stage": stage, "recipe": recipe, "seed": seed,
              "segment_index": index, "accounting": _counters()}
    record.update(result_overrides)
    R.write_json_exclusive(segment / "result.json", record)
    return segment


def _refusal_segment(run, index):
    """Exactly what ``_write_refusal_result`` produces: no identity triple."""
    segment = run / f"segment_{index:04d}"
    segment.mkdir(parents=True)
    R.write_json_exclusive(segment / "result.json", {
        "schema_version": R.RESULT_SCHEMA, "runner_schema": R.SCHEMA_VERSION,
        "status": R.REFUSED_BEFORE_TRAINING, "training_began": False,
        "refusal": "RunnerRefusal: deliberate", "stop_reason": R.REFUSED_BEFORE_TRAINING,
        "published": False, "boundary_is_live": False,
        "censored": False, "censor_reason": None,
        "completed_epoch": None, "completed_epochs": 0,
        "accounting": R._zero_accounting(), "wall_seconds": 1.0})
    return segment


def test_a_refusal_record_does_not_poison_identity_collection(tmp_path):
    """RED reproduction. Pre-fix this raised
    ``TypeError: '<' not supported between instances of 'str' and 'NoneType'``
    from inside the cross-run guard's own message."""
    run = tmp_path / "run"
    _trained_segment(run, 0, 0)
    _refusal_segment(run, 1)

    try:
        candidates = R.collect_candidates(run)
    except TypeError as error:
        raise AssertionError(
            f"a refusal record poisoned identity collection: {error}") from error
    except R.RunnerFault as error:
        raise AssertionError(
            f"a refusal record was mistaken for cross-run mixing: {error}") from error

    assert [candidate["epoch"] for candidate in candidates] == [0]


def test_a_refusal_record_is_never_a_checkpoint_candidate(tmp_path):
    run = tmp_path / "run"
    _trained_segment(run, 0, 0)
    _refusal_segment(run, 1)
    candidates, refused = R.collect_run_records(run)

    assert len(candidates) == 1
    assert candidates[0]["epoch"] == 0
    assert len(refused) == 1
    assert "segment_0001" in refused[0]


def test_selection_over_a_run_with_a_refusal_uses_the_trained_segments_alone(tmp_path):
    run = tmp_path / "run"
    _trained_segment(run, 0, 0)
    segment = run / "segment_0002"
    segment.mkdir(parents=True)
    R.write_json_exclusive(segment / "selection_0064.json", {
        "epoch": 64, "kind": "selection", "evaluation": _evaluation(survival=0.5),
        "checkpoint": "epoch_0064.pt"})
    R.write_json_exclusive(segment / "result.json", {
        "schema_version": R.RESULT_SCHEMA, "training_began": True, "stage": "screen",
        "recipe": "g990_e010", "seed": 1001, "segment_index": 2,
        "accounting": _counters()})
    _refusal_segment(run, 1)

    args = _parse(["select", "--run-dir", str(run)])
    assert R.run_select(args) == R.EXIT_OK
    record = R.read_json(run / "selection.json")
    assert record["status"] == "selected"
    assert record["selected"]["epoch"] == 0          # the stronger evaluation
    assert record["candidates_considered"] == 2
    assert len(record["refused_segments"]) == 1
    assert all("segment_0001" in path for path in record["refused_segments"])


def test_a_run_of_nothing_but_refusals_is_recorded_as_incomplete(tmp_path):
    """Binding. Nothing ever trained, so there is no valid selection -- but that
    is an operational outcome, not a runner bug, and it must be recorded rather
    than crashing or silently selecting nothing."""
    run = tmp_path / "run"
    _refusal_segment(run, 0)
    _refusal_segment(run, 1)

    args = _parse(["select", "--run-dir", str(run)])
    assert R.run_select(args) == R.EXIT_OK
    record = R.read_json(run / "selection.json")
    assert record["status"] == "incomplete"
    assert record["selected"] is None
    assert record["candidates_considered"] == 0
    assert len(record["refused_segments"]) == 2
    assert "fault" not in record


def test_an_empty_run_directory_is_still_a_runner_fault(tmp_path):
    """Binding. With no refusals to explain it, an empty candidate set is a
    runner bug by construction: every run has at least an epoch 0 checkpoint."""
    run = tmp_path / "run"
    run.mkdir(parents=True)
    with pytest.raises(R.RunnerFault):
        R.select_run(R.collect_candidates(run))


def test_genuine_cross_run_mixing_still_aborts(tmp_path):
    """Binding. The guard must not be made permissive to get past the refusal
    case: real contamination still aborts."""
    run = tmp_path / "run"
    _trained_segment(run, 0, 0, seed=1001)
    _trained_segment(run, 1, 64, seed=1002)
    with pytest.raises(R.RunnerFault, match="single run/arm"):
        R.collect_candidates(run)


@pytest.mark.parametrize("missing", ["stage", "recipe", "seed"])
def test_a_trained_record_missing_an_identity_key_still_aborts_without_a_typeerror(
        missing, tmp_path):
    """Binding, and the order-safety case. A trained record lacking an identity
    key still contributes ``None`` and still trips the guard -- but the message
    must be buildable over a set mixing ``str`` and ``None``."""
    run = tmp_path / "run"
    _trained_segment(run, 0, 0)
    _trained_segment(run, 1, 64, **{missing: None})
    try:
        with pytest.raises(R.RunnerFault, match="single run/arm"):
            R.collect_candidates(run)
    except TypeError as error:
        raise AssertionError(f"the fault message was not order-safe: {error}") from error


def test_the_cross_run_fault_message_is_deterministic(tmp_path):
    """Behavioural. The message must not depend on set iteration order."""
    messages = set()
    for attempt in range(4):
        run = tmp_path / f"run{attempt}"
        _trained_segment(run, 0, 0, seed=1001)
        _trained_segment(run, 1, 64, seed=1002)
        _trained_segment(run, 2, 128, recipe=None)
        with pytest.raises(R.RunnerFault) as raised:
            R.collect_candidates(run)
        messages.add(str(raised.value))
    assert len(messages) == 1


def test_a_refusal_record_is_recognised_only_on_a_positive_declaration():
    """Binding. A trained record must never be mistaken for a refusal, so both
    the marker and ``training_began is False`` are required."""
    assert R._is_refusal_record({"status": R.REFUSED_BEFORE_TRAINING,
                                 "training_began": False}) is True
    assert R._is_refusal_record({"status": R.REFUSED_BEFORE_TRAINING,
                                 "training_began": True}) is False
    assert R._is_refusal_record({"status": R.REFUSED_BEFORE_TRAINING}) is False
    assert R._is_refusal_record({"training_began": False}) is False
    assert R._is_refusal_record({"status": "other", "training_began": False}) is False
    assert R._is_refusal_record("not a mapping") is False
    assert R._is_refusal_record(None) is False


# -- 6b: an absent out_dir must never write into the working tree -----------

@pytest.mark.parametrize("value", [None, ""])
def test_a_refusal_with_no_declared_output_directory_writes_nothing_to_the_cwd(
        value, tmp_path, monkeypatch):
    """Binding. ``Path("")`` is the current directory, so the earlier fallback
    deposited a refusal record into the working tree."""
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(out_dir=value)
    R._write_refusal_result(args, R.RunnerRefusal("no directory was declared"),
                            clock=lambda: 1.0, started=0.0,
                            wrote={"training_began": False})
    assert not (tmp_path / "result.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_a_refusal_with_a_missing_out_dir_attribute_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    R._write_refusal_result(argparse.Namespace(), R.RunnerRefusal("no attribute"),
                            clock=lambda: 1.0, started=0.0,
                            wrote={"training_began": False})
    assert list(tmp_path.iterdir()) == []


# ==========================================================================
# Unit 7 -- one epoch, one evaluation (the IN-28 boundary-epoch defect)
# ==========================================================================
#
# The accounted smoke launch found that every multi-segment run evaluated its
# boundary epoch TWICE: segment N evaluated epoch 64 as the epoch it completed,
# and segment N+1 evaluated that same epoch 64 again as its start epoch.
# ``select`` then aborted with child rc 4 and ``duplicate candidate 'epoch' 64``,
# so all nine scheduled runs would have produced no selection and no promotion
# after the whole 28,800 s budget was spent.
#
# The ruling: **the segment that COMPLETES an epoch evaluates it, exactly once.**
# A resuming segment must not re-evaluate the boundary it resumed from, because
# the previous segment already completed and evaluated it. Evaluation belongs to
# the epoch, not to the process, which is what makes epoch-to-evaluation a
# function -- precisely the property ``select_best_checkpoint`` requires. The
# duplicate-epoch refusal itself is correct and is NOT weakened here; what is
# fixed is the runner producing two evaluations for one epoch.

def _two_segment_run(tmp_path, *, first_epochs=64, second_epochs=2):
    """A real two-segment run: publish a boundary, then really resume from it.

    ``screen`` evaluates at epochs 0, 64 and 128 and its first sealed round is
    0..64, so 64 is the only boundary this can reach without inventing a
    protocol. Both segments go through ``run_worker``, so the publish, the
    lineage binding, the five-key ``strict_expect``, both provenance comparison
    classes and ``restore_state`` are all real; only the runtime is the Task 3
    analytic CPU adapter.
    """
    run = tmp_path / "run"
    first = run / "segment_0000"
    assert _run_worker(tmp_path, first, **{"--epochs": first_epochs}) == R.EXIT_OK
    parent = R.read_json(first / "result.json")
    assert parent["published"] is True
    assert parent["lineage"]["valid_boundary_epoch"] == first_epochs
    second = run / "segment_0001"
    code = _run_worker(tmp_path, second, **{
        "--segment-index": 1, "--epochs": second_epochs,
        "--parent-segment": first / "segment.ptc",
        "--parent-result": first / "result.json"})
    return run, first, second, code


def _written_epochs(segment):
    return sorted(int(path.name[len("selection_"):-len(".json")])
                  for path in Path(segment).glob("selection_*.json"))


def test_select_over_a_real_two_segment_run_reaches_a_selection(tmp_path):
    """RED reproduction of the smoke's exact failure. Pre-fix this returned
    ``EXIT_RUNNER_FAULT`` (child rc 4) with ``duplicate candidate 'epoch' 64``
    and wrote a ``runner_fault`` record carrying two epoch-64 statuses."""
    run, first, second, code = _two_segment_run(tmp_path)
    assert code == R.EXIT_OK

    rc = R.main(["select", "--run-dir", str(run)])
    record = R.read_json(run / "selection.json")
    assert rc == R.EXIT_OK, f"select faulted on a two-segment run: {record.get('fault')}"
    assert record["status"] == "selected"
    assert "fault" not in record
    assert record["selected"]["epoch"] in (0, 64)


def test_a_resumed_segment_schedules_no_evaluation_for_its_resume_boundary(tmp_path):
    """The ruling, directly. Epoch 64 was completed by segment 0, so segment 0
    evaluates it; segment 1 resumed from it and completed no scheduled epoch of
    its own, so it writes no evaluation at all."""
    run, first, second, code = _two_segment_run(tmp_path)
    assert code == R.EXIT_OK
    assert _written_epochs(first) == [0, 64]
    assert _written_epochs(second) == []
    assert R.read_json(second / "result.json")["evaluations"] == []


def test_each_epoch_is_evaluated_at_most_once_across_a_multi_segment_run(tmp_path):
    """Behavioural, and the property selection actually needs: across the whole
    run directory, epoch -> evaluation is a function."""
    run, first, second, code = _two_segment_run(tmp_path)
    assert code == R.EXIT_OK
    epochs = [candidate["epoch"] for candidate in R.collect_candidates(run)]
    assert epochs == sorted(set(epochs)), f"an epoch was evaluated twice: {epochs}"
    assert epochs == [0, 64]


def test_the_surviving_boundary_evaluation_is_the_completing_segments_own(tmp_path):
    """Binding. The one epoch-64 evaluation is the one written by the segment
    that completed epoch 64 -- not a copy relabelled after the fact, and not the
    resumed segment's second measurement of somebody else's epoch."""
    run, first, second, code = _two_segment_run(tmp_path)
    assert code == R.EXIT_OK
    candidates = {candidate["epoch"]: candidate for candidate in R.collect_candidates(run)}
    assert Path(candidates[64]["evaluation_path"]).parent == first
    assert R.read_json(first / "result.json")["evaluations"] == [
        str(first / "selection_0000.json"), str(first / "selection_0064.json")]


def test_a_first_segment_still_evaluates_its_own_start_epoch(tmp_path):
    """Binding, the other half of the ruling. Epoch 0 is the initial policy: no
    segment completes it, so the segment that starts the run owns it. Dropping
    it would lose the initial-actor checkpoint v1's selection turned on."""
    out = tmp_path / "run" / "segment_0000"
    assert _run_worker(tmp_path, out, **{"--epochs": 3}) == R.EXIT_OK
    assert _written_epochs(out) == [0]


# -- the derived schedule: structural, not a conditional at the call site ---

def test_the_permitted_evaluation_set_is_derived_from_the_round():
    """Structural. The set a segment may evaluate is a function of its own round,
    so a resumed segment's schedule cannot contain its start epoch however the
    global schedule is spelled."""
    schedule = P.evaluation_epochs(128)
    assert 64 in schedule and 0 in schedule

    resumed = R.segment_evaluation_epochs(schedule, start_epoch=64, end_epoch=128,
                                          resumed=True)
    assert resumed == frozenset({128})
    first = R.segment_evaluation_epochs(schedule, start_epoch=0, end_epoch=64,
                                        resumed=False)
    assert first == frozenset({0, 64})
    # Nothing beyond this segment's own end is ever schedulable here.
    assert R.segment_evaluation_epochs(schedule, start_epoch=0, end_epoch=3,
                                       resumed=False) == frozenset({0})
    # Idempotent: re-deriving over an already-derived set changes nothing, so the
    # two enforcement points cannot disagree.
    assert R.segment_evaluation_epochs(resumed, start_epoch=64, end_epoch=128,
                                       resumed=True) == resumed
    # Union over the sealed rounds is exactly the sealed schedule, once each.
    rounds = P.segment_rounds(384)
    covered = []
    for round_ in rounds:
        covered += sorted(R.segment_evaluation_epochs(
            P.evaluation_epochs(384), start_epoch=round_.start_epoch,
            end_epoch=round_.end_epoch, resumed=round_.index > 0))
    assert covered == list(P.evaluation_epochs(384))


def test_train_segment_makes_every_caller_declare_whether_it_resumed():
    """Structural. There is no default, so the defect cannot come back by a
    caller forgetting one keyword: omitting it is a TypeError, not a silent
    second evaluation of an epoch this segment did not complete."""
    import inspect

    parameter = inspect.signature(R.train_segment).parameters["resumed_from_boundary"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_train_segment_schedules_no_start_epoch_evaluation_when_it_resumed():
    """Behavioural at the training seam, with no filesystem in the way."""
    algo, env = _build(steps_num=2, num_envs=4)
    _start(algo, env)
    seen = []
    outcome = _run(algo, env, epochs=2, start_epoch=64,
                   evaluate=lambda epoch: seen.append(int(epoch)) or {"epoch": epoch},
                   evaluation_epochs=(0, 64, 128), resumed_from_boundary=True)
    assert seen == []
    assert outcome.evaluations == []
    assert outcome.completed_epoch == 66


def test_train_segment_evaluates_a_start_epoch_no_other_segment_completed():
    """The contrast. Same schedule, same start epoch, not resumed -> evaluated
    exactly once, and the evaluation is recorded."""
    algo, env = _build(steps_num=2, num_envs=4)
    _start(algo, env)
    seen = []
    outcome = _run(algo, env, epochs=2, start_epoch=64,
                   evaluate=lambda epoch: seen.append(int(epoch)) or {"epoch": epoch},
                   evaluation_epochs=(0, 64, 128), resumed_from_boundary=False)
    assert seen == [64]
    assert outcome.evaluations == [{"epoch": 64}]


def test_train_segment_still_evaluates_every_completed_scheduled_epoch():
    """Anti-vacuity. The fix must not silence the completed-epoch evaluations
    that are the campaign's whole point."""
    algo, env = _build(steps_num=2, num_envs=4)
    _start(algo, env)
    seen = []
    _run(algo, env, epochs=4, start_epoch=0,
         evaluate=lambda epoch: seen.append(int(epoch)) or {"epoch": epoch},
         evaluation_epochs=(0, 1, 3, 64), resumed_from_boundary=False)
    assert seen == [0, 1, 3]


# -- the writer fence: no file can be written for a non-owned epoch ---------

def test_the_evaluation_writer_refuses_an_epoch_this_segment_did_not_complete(tmp_path):
    """Structural backstop. Even if a future edit re-introduced a call for a
    non-owned epoch, no ``selection_*.json`` could be written for it: the fault
    is raised instead, and the evaluation is never even run."""
    calls = []

    def evaluate(epoch, deadline=None):
        calls.append(int(epoch))
        return ({"epoch": int(epoch), "kind": "selection", "evaluation": _evaluation(),
                 "checkpoint": f"epoch_{epoch:04d}.pt"},
                {"attempted_calls": 1, "completed_calls": 1, "worlds": 16})

    totals = {"attempted": 0, "completed": 0}
    paths = []
    writer = R._evaluation_writer(evaluate, tmp_path, own_epochs=frozenset({0}),
                                  deadline=1e9, totals=totals, paths=paths)
    writer(0)
    with pytest.raises(R.RunnerFault, match="did not complete"):
        writer(64)

    assert calls == [0], "the refused epoch must not be evaluated at all"
    assert _written_epochs(tmp_path) == [0]
    assert paths == [str(tmp_path / "selection_0000.json")]
    assert totals == {"attempted": 16, "completed": 16}


def test_the_writer_is_the_only_place_a_selection_record_is_written():
    """Binding, read off the source. One fenced writer, so the fence cannot be
    bypassed by a second write site."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert source.count('f"selection_{') == 1
    tree = ast.parse(source)
    writer = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_evaluation_writer")
    assert 'selection_{' in ast.unparse(writer)


# -- the duplicate guard itself is untouched --------------------------------

def test_a_genuine_duplicate_epoch_still_aborts_selection(tmp_path):
    """Binding. The duplicate-epoch refusal is NOT weakened by this fix: real
    cross-run contamination -- two evaluations of one epoch inside one run
    directory, which is what a copied-in or re-run segment produces -- still
    aborts with a set-level fault, and the record still names both."""
    run = tmp_path / "run"
    _trained_segment(run, 0, 64)
    _trained_segment(run, 1, 64)

    rc = R.main(["select", "--run-dir", str(run)])
    assert rc == R.EXIT_RUNNER_FAULT
    record = R.read_json(run / "selection.json")
    assert record["status"] == "runner_fault"
    assert record["fault"]["kind"] == "InvalidCandidateSetError"
    assert "duplicate candidate 'epoch' 64" in record["fault"]["reason"]
    assert record["selected"] is None
    assert [entry["epoch"] for entry in record["checkpoint_status"]] == [64, 64]


def test_the_duplicate_guard_is_not_routed_around_in_the_selection_layer():
    """Binding, read off the source. The fix is upstream, in scheduling; the
    selection layer gained no de-duplication, no preference rule and no
    tolerance for two candidates sharing an epoch."""
    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    for name in ("collect_run_records", "select_run", "_delegate_set_fault",
                 "_selection_record"):
        node = next(item for item in ast.walk(tree)
                    if isinstance(item, ast.FunctionDef) and item.name == name)
        text = ast.unparse(node)
        for forbidden in ("dedup", "seen_epoch", "unique", "prefer", "discard",
                          "sorted(candidates", "candidates[-1]"):
            assert forbidden not in text, f"{name} gained {forbidden!r}"


# -- what this fix hides rather than fixes, and the truncation caveat -------

def test_the_in29_cross_process_divergence_is_recorded_not_papered_over():
    """Binding. Removing the duplicate stops the ~1e-3 cross-process evaluation
    divergence being VISIBLE; it does not resolve it. The consequence for the
    ranking key must stay written down in the source a campaign operator reads."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "IN-29" in source
    assert "1e-3" in source


def test_the_shared_deadline_truncation_consequence_is_recorded():
    """Binding, requirement 5's caveat. The work deadline and the evaluation
    deadline are one clock, so a genuine ``wall_cap`` truncation always ends that
    segment. Acceptable, but it must not be mistaken later for a fault."""
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "wall_cap" in source
    tree = ast.parse(source)
    writer = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_evaluation_writer")
    assert "wall_cap" in (ast.get_docstring(writer) or "")
