"""Unit tests for the v2 append-only budget ledger and the campaign lock files.

Fake clocks and fake process records only. **No simulator, no CUDA, no Warp, no
torch, no subprocess, no sleep, no real segment, no GPU query.** Every wall time
below is an injected number, so this file consumes **zero** of the 28,800 s
training budget and none of the 1,800 s diagnostic budget. Test-suite time is
separate accounting.

The fake clock returns injected POSIX timestamps; the fake "process" is a plain
duration plus a return code. Nothing here waits on a real clock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msk_warp.analysis import ppo_v2_budget as B
from msk_warp.analysis import ppo_v2_protocol as P


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeClock:
    """A monotonically advancing injected clock. Never sleeps, never reads time."""

    def __init__(self, start=1_700_000_000.0, step=1.0):
        self.now = float(start)
        self.step = float(step)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        value = self.now
        self.now += self.step
        return value


class FakeProbe:
    """An injected contention observation. No GPU is queried."""

    def __init__(self, observation=None, raises=None):
        self.observation = observation
        self.raises = raises
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.observation


def _counters(**overrides):
    values = dict(
        attempted_control_transitions=8192,
        completed_control_transitions=8192,
        replayed_control_transitions=0,
        attempted_physics_steps=32768,
        completed_physics_steps=32768,
        replayed_physics_steps=0,
        partial_update_cost_s=0.0,
    )
    values.update(overrides)
    return B.SegmentCounters(**values)


def _ledger(tmp_path, *, probe=None, clock=None, name="budget_ledger.jsonl"):
    return B.BudgetLedger.create(
        tmp_path / name, protocol_digest=P.protocol_digest(),
        clock=clock or FakeClock(), contention_probe=probe)


def _reserve(ledger, *, stage="screen", recipe="g990_e010", seed=1001,
             segment_index=0, bound=400.0, allowance=60.0, create=None,
             start_epoch=0, end_epoch=64):
    return ledger.reserve(
        stage=stage, recipe=recipe, seed=seed, segment_index=segment_index,
        start_epoch=start_epoch, end_epoch=end_epoch,
        reserved_bound_s=bound, shutdown_allowance_s=allowance, create=create)


def _charge(ledger, *, stage, recipe, seed, actual, segment_index=0, bound=599.0,
            allowance=1.0):
    """Settle one fake segment so its actual wall is charged. No process runs.

    ``bound + allowance`` is exactly the 600 s call bound, which is the largest a
    single reservation may be; ``actual`` is an injected number and may overrun
    it, because an overrun is charged rather than forbidden.
    """
    reservation = ledger.reserve(
        stage=stage, recipe=recipe, seed=seed, segment_index=segment_index,
        start_epoch=0, end_epoch=64, reserved_bound_s=bound,
        shutdown_allowance_s=allowance, create=None)
    return ledger.settle(reservation, actual_wall_s=actual, returncode=0,
                         counters=_counters())


# --------------------------------------------------------------------------
# Ledger creation: no fresh-ledger bypass
# --------------------------------------------------------------------------

def test_open_refuses_a_missing_ledger_instead_of_creating_one(tmp_path):
    with pytest.raises(B.LedgerIntegrityError, match="does not exist"):
        B.BudgetLedger.open(tmp_path / "budget_ledger.jsonl")


def test_create_refuses_to_overwrite_an_existing_ledger(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    _ledger(tmp_path)
    assert path.exists()
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.create(path, protocol_digest=P.protocol_digest(),
                              clock=FakeClock())
    # The original bytes survive the refused create.
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_header_binds_the_schema_and_the_protocol_digest(tmp_path):
    ledger = _ledger(tmp_path)
    header = ledger.header
    assert header["kind"] == B.HEADER_KIND
    assert header["schema_version"] == B.SCHEMA_VERSION
    assert header["protocol_digest"] == P.protocol_digest()
    assert header["prev_sha256"] == B.GENESIS_PREV_SHA256
    assert header["total_wall_cap_s"] == 28800
    assert header["index"] == 0


def test_a_corrupt_ledger_cannot_be_replaced_by_a_fresh_one(tmp_path):
    """The only way past a broken ledger is human reconciliation."""
    path = tmp_path / "budget_ledger.jsonl"
    _ledger(tmp_path)
    path.write_text("{not json at all}\n", encoding="utf-8")
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(path)
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.create(path, protocol_digest=P.protocol_digest(),
                              clock=FakeClock())


# --------------------------------------------------------------------------
# Integrity: malformed, truncated, tampered, non-monotonic
# --------------------------------------------------------------------------

def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def _rewrite(path, lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_truncated_final_record_fails_closed(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    _reserve(ledger)
    lines = _lines(path)
    assert len(lines) == 2
    _rewrite(path, [lines[0], lines[1][: len(lines[1]) // 2]])
    with pytest.raises(B.LedgerIntegrityError, match="not valid JSON"):
        B.BudgetLedger.open(path)


def test_tampered_payload_breaks_its_own_record_hash(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=400.0)
    ledger.settle(reservation, actual_wall_s=390.0, returncode=0,
                  counters=_counters())
    lines = _lines(path)
    row = json.loads(lines[2])
    assert row["actual_wall_s"] == 390.0
    row["actual_wall_s"] = 1.0  # a cheaper-looking segment
    lines[2] = json.dumps(row, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(B.LedgerIntegrityError, match="record_sha256"):
        B.BudgetLedger.open(path)


def test_broken_hash_link_fails_closed(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    _reserve(ledger)
    lines = _lines(path)
    row = json.loads(lines[1])
    row["prev_sha256"] = "f" * 64
    row["record_sha256"] = B.record_digest(
        {k: v for k, v in row.items() if k != "record_sha256"})
    lines[1] = json.dumps(row, sort_keys=True)
    _rewrite(path, lines)
    # The row now hashes itself correctly but no longer links to its parent.
    with pytest.raises(B.LedgerIntegrityError, match="prev_sha256"):
        B.BudgetLedger.open(path)


def test_dropped_record_breaks_the_chain_not_just_the_index(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    ledger.settle(reservation, actual_wall_s=100.0, returncode=0,
                  counters=_counters())
    lines = _lines(path)
    _rewrite(path, [lines[0], lines[2]])  # excise the reservation
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(path)


def test_non_monotonic_index_fails_closed(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    _reserve(ledger)
    lines = _lines(path)
    row = json.loads(lines[1])
    row["index"] = 7
    row["record_sha256"] = B.record_digest(
        {k: v for k, v in row.items() if k != "record_sha256"})
    lines[1] = json.dumps(row, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(B.LedgerIntegrityError, match="index"):
        B.BudgetLedger.open(path)


def test_non_monotonic_timestamp_fails_closed(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    _reserve(ledger)
    lines = _lines(path)
    row = json.loads(lines[1])
    row["posix_time"] = 0.0
    row["record_sha256"] = B.record_digest(
        {k: v for k, v in row.items() if k != "record_sha256"})
    lines[1] = json.dumps(row, sort_keys=True)
    _rewrite(path, lines)
    with pytest.raises(B.LedgerIntegrityError, match="time"):
        B.BudgetLedger.open(path)


def test_wrong_schema_or_protocol_digest_fails_closed(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    _ledger(tmp_path)
    lines = _lines(path)
    row = json.loads(lines[0])
    row["schema_version"] = "some-other-schema"
    row["record_sha256"] = B.record_digest(
        {k: v for k, v in row.items() if k != "record_sha256"})
    _rewrite(path, [json.dumps(row, sort_keys=True)])
    with pytest.raises(B.LedgerIntegrityError, match="schema"):
        B.BudgetLedger.open(path)


def test_an_empty_ledger_file_is_invalid_not_empty(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(path)


# --------------------------------------------------------------------------
# Reservation ordering, pending blocking and creation ordering
# --------------------------------------------------------------------------

def test_reservation_is_appended_before_the_launch_side_effect(tmp_path):
    """The reserve row must be on disk before anything expensive is created."""
    path = tmp_path / "budget_ledger.jsonl"
    observed = []

    ledger = _ledger(tmp_path)

    def create():
        # Whatever this represents (an output directory, and later the child's
        # backend import, model compile and env creation) it runs strictly after
        # the reservation exists on disk.
        observed.append(len(_lines(path)))

    reservation = _reserve(ledger, create=create)
    assert observed == [2], "create ran before the reservation was durable"
    assert reservation.index == 1
    assert ledger.pending is not None


def test_a_pending_reservation_blocks_the_next_job(tmp_path):
    ledger = _ledger(tmp_path)
    first = _reserve(ledger, segment_index=0)
    with pytest.raises(B.PendingReservationError, match=first.reservation_id):
        _reserve(ledger, segment_index=1)
    ledger.settle(first, actual_wall_s=10.0, returncode=0, counters=_counters())
    second = _reserve(ledger, segment_index=1)
    assert second.index == 3


def test_a_pending_reservation_survives_a_reopen_and_still_blocks(tmp_path):
    """Foreground-sequential enforcement must not depend on in-process memory."""
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    first = _reserve(ledger)
    reopened = B.BudgetLedger.open(path, clock=FakeClock(start=1_800_000_000.0))
    assert reopened.pending is not None
    assert reopened.pending.reservation_id == first.reservation_id
    with pytest.raises(B.PendingReservationError):
        _reserve(reopened, segment_index=1)


def test_exhaustion_refuses_before_any_creation_happens(tmp_path):
    """Budget exhaustion is detected before the expensive create callable runs."""
    ledger = _ledger(tmp_path)
    _charge(ledger, stage="screen", recipe="g990_e010", seed=1001, actual=1200.0)
    created = []
    with pytest.raises(B.BudgetExhaustedError):
        _reserve(ledger, bound=400.0, allowance=60.0,
                 create=lambda: created.append(1))
    assert created == [], "the create callable must not run on a refused reserve"
    assert ledger.pending is None


def test_a_failing_exclusive_create_aborts_at_zero_and_does_not_wedge(tmp_path):
    """Exclusive-create refusal must not permanently block the campaign.

    The reservation is durable before ``create`` runs, so an append-only ledger
    cannot retract it. But ``create`` fails **before any child exists**, and the
    absence of a spawn marker is positive evidence of that, so the reservation is
    settled as aborted at a **zero** charge and the next job may start. Both rows
    stay in the hash chain forever, so the episode remains auditable.
    """
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    outdir = tmp_path / "segment_0000"
    outdir.mkdir()
    with pytest.raises(B.PrelaunchError, match="segment_0000"):
        _reserve(ledger, create=outdir.mkdir)  # exists -> FileExistsError
    lines = _lines(path)
    assert len(lines) == 3, "reserve row, then the auto-abort settlement"
    assert json.loads(lines[1])["kind"] == B.RESERVE_KIND
    aborted = json.loads(lines[2])
    assert aborted["kind"] == B.SETTLE_KIND
    assert aborted["charge_basis"] == B.CHARGE_ABORTED_PRESPAWN
    assert aborted["charged_s"] == 0.0
    assert aborted["spawned"] is False
    assert ledger.charged()["global_s"] == 0.0
    assert ledger.pending is None

    # The next job starts with the full budget intact.
    fresh = tmp_path / "segment_0001"
    assert _reserve(ledger, create=fresh.mkdir) is not None
    assert fresh.is_dir()
    assert B.BudgetLedger.open(path).rows[1]["kind"] == B.RESERVE_KIND


# --------------------------------------------------------------------------
# Ruling 3: the spawn marker separates "never spawned" from "outcome unknown"
# --------------------------------------------------------------------------

def test_no_spawn_marker_exists_until_mark_spawn_is_called(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    assert ledger.spawned(reservation) is False
    row = ledger.mark_spawn(reservation)
    assert row["kind"] == B.SPAWN_KIND
    assert row["reservation_id"] == reservation.reservation_id
    assert row["reservation_index"] == reservation.index
    assert ledger.spawned(reservation) is True
    # It is written once per reservation.
    with pytest.raises(B.SettlementError, match="already"):
        ledger.mark_spawn(reservation)


def test_the_spawn_marker_survives_a_reopen_and_is_read_from_the_file(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    ledger.mark_spawn(reservation)
    reopened = B.BudgetLedger.open(path, clock=FakeClock(start=1_800_000_000.0))
    assert reopened.spawned(reopened.pending) is True
    with pytest.raises(B.SettlementError, match="spawn"):
        reopened.settle_aborted(reopened.pending, detail="trying to dodge the charge")


def test_pre_spawn_abort_settles_at_zero(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    assert ledger.spawned(reservation) is False
    row = ledger.settle_aborted(reservation, detail="bad output path; nothing started")
    assert row["charged_s"] == 0.0
    assert row["charge_basis"] == B.CHARGE_ABORTED_PRESPAWN
    assert row["actual_wall_s"] is None
    assert row["spawned"] is False
    assert row["reserved_bound_s"] == 500.0
    assert "nothing started" in row["detail"]
    # No child ran, so completed work is a known zero rather than unknown.
    assert row["attempted_control_transitions"] == 0
    assert row["completed_control_transitions"] == 0
    assert ledger.charged()["global_s"] == 0.0


def test_post_spawn_unknown_settles_at_the_full_reserved_bound(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    ledger.mark_spawn(reservation)
    row = ledger.settle_unknown(reservation, detail="child vanished after spawn")
    assert row["charged_s"] == 500.0
    assert row["charge_basis"] == B.CHARGE_RESERVED_BOUND
    assert row["spawned"] is True
    # The unknown counters stay null, not zero: a child did run.
    assert row["completed_control_transitions"] is None


def test_a_spawned_run_can_never_be_settled_as_a_zero_charge_abort(tmp_path):
    """The whole point of the marker: no zero charge without evidence."""
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    ledger.mark_spawn(reservation)
    with pytest.raises(B.SettlementError, match="spawn"):
        ledger.settle_aborted(reservation, detail="pretend nothing happened")
    assert ledger.pending is not None
    # The only routes left both charge: the actual wall, or the full bound.
    row = ledger.settle_unknown(reservation, detail="outcome unknown")
    assert row["charged_s"] == 500.0


def test_a_forged_reservation_cannot_claim_a_zero_charge(tmp_path):
    """The marker is read from the file, never from the caller's object."""
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    ledger.mark_spawn(reservation)
    forged = B.Reservation(
        index=reservation.index, reservation_id=reservation.reservation_id,
        stage="screen", recipe="g990_e010", seed=1001, segment_index=0,
        start_epoch=0, end_epoch=64, reserved_bound_s=1.0,
        shutdown_allowance_s=1.0, contention=B.unobserved_contention())
    with pytest.raises(B.SettlementError, match="spawn"):
        ledger.settle_aborted(forged, detail="forged, and it must not work")
    # Even the bound comes from the file, not from the forged object.
    row = ledger.settle_unknown(forged, detail="settled from the recorded bound")
    assert row["charged_s"] == 500.0


def test_excising_a_spawn_marker_from_history_breaks_the_chain(tmp_path):
    """A marker removed from history cannot turn a spawned run into an abort."""
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    ledger.mark_spawn(reservation)
    ledger.settle(reservation, actual_wall_s=120.0, returncode=0,
                  counters=_counters())
    lines = _lines(path)
    assert len(lines) == 4 and json.loads(lines[2])["kind"] == B.SPAWN_KIND
    _rewrite(path, [lines[0], lines[1], lines[3]])  # excise the marker
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(path)


def test_tail_truncation_is_caught_inside_a_live_ledger(tmp_path):
    """A live ledger refuses a file that shrank or whose observed rows changed."""
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    ledger.mark_spawn(reservation)
    lines = _lines(path)
    _rewrite(path, [lines[0], lines[1]])  # drop the trailing marker
    with pytest.raises(B.LedgerIntegrityError, match="shrank"):
        ledger.settle_aborted(reservation, detail="trying to dodge the charge")


def test_the_tail_truncation_gap_is_disclosed_not_claimed_detected():
    """Honest limitation: a chain protects history, not the last line.

    A fresh process cannot prove from the file alone that no further record ever
    existed, so the limitation is written down instead of being claimed away.
    """
    note = B.LEDGER_TAIL_NOTE.lower()
    assert "tail" in note
    assert "cannot prove" in note
    assert "disclosed" in note
    assert "shrunk" in note or "shrank" in note


def test_a_spawn_marker_without_an_open_reservation_fails_closed(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    ledger.settle(reservation, actual_wall_s=5.0, returncode=0, counters=_counters())
    with pytest.raises(B.SettlementError):
        ledger.mark_spawn(reservation)
    # And a hand-forged marker appended after a settlement is refused on read.
    lines = _lines(path)
    row = json.loads(lines[2])
    forged = {"index": 3, "kind": B.SPAWN_KIND, "prev_sha256": row["record_sha256"],
              "posix_time": row["posix_time"] + 1.0, "utc": row["utc"],
              "clock_clamped": False, "reservation_id": reservation.reservation_id,
              "reservation_index": reservation.index, "stage": "screen",
              "recipe": "g990_e010", "seed": 1001, "segment_index": 0,
              "detail": None}
    forged["record_sha256"] = B.record_digest(forged)
    _rewrite(path, lines + [json.dumps(forged, sort_keys=True)])
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(path)


def test_an_aborted_reservation_unblocks_the_next_reserve_but_stays_in_the_chain(tmp_path):
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    first = _reserve(ledger, segment_index=0, bound=500.0, allowance=60.0)
    with pytest.raises(B.PendingReservationError):
        _reserve(ledger, segment_index=1)
    ledger.settle_aborted(first, detail="aborted before any spawn")
    second = _reserve(ledger, segment_index=1)
    assert second.index == 3
    # Both the reservation and its abort are permanent, and the file still verifies.
    reopened = B.BudgetLedger.open(path)
    kinds = [row["kind"] for row in reopened.rows]
    assert kinds == [B.HEADER_KIND, B.RESERVE_KIND, B.SETTLE_KIND, B.RESERVE_KIND]
    assert reopened.rows[2]["charge_basis"] == B.CHARGE_ABORTED_PRESPAWN
    assert reopened.charged()["global_s"] == 0.0
    assert reopened.charged()["settled_rows"] == 1


def test_abort_needs_an_explicit_detail_and_refuses_double_settlement(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    with pytest.raises(B.SettlementError, match="detail"):
        ledger.settle_aborted(reservation, detail="   ")
    ledger.settle_aborted(reservation, detail="aborted")
    with pytest.raises(B.DoubleSettlementError):
        ledger.settle_aborted(reservation, detail="again")


def test_a_normal_settlement_records_that_the_child_was_spawned(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    ledger.mark_spawn(reservation)
    row = ledger.settle(reservation, actual_wall_s=312.5, returncode=0,
                        counters=_counters())
    assert row["spawned"] is True
    assert row["charged_s"] == 312.5
    assert row["charge_basis"] == B.CHARGE_ACTUAL


def test_zero_charge_is_reachable_only_through_the_abort_path(tmp_path):
    """No other settlement route can produce a zero charge for a spawned run."""
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    ledger.mark_spawn(reservation)
    # settle() charges whatever wall is reported, and a zero wall for a spawned
    # child is a measurement the caller asserts, not an abort: it is still
    # recorded as an actual-wall charge with spawned true, so it is auditable.
    row = ledger.settle(reservation, actual_wall_s=0.0, returncode=0,
                        counters=_counters())
    assert row["charge_basis"] == B.CHARGE_ACTUAL
    assert row["spawned"] is True
    assert row["charged_s"] == 0.0
    assert row["charge_basis"] != B.CHARGE_ABORTED_PRESPAWN


# --------------------------------------------------------------------------
# Settlement
# --------------------------------------------------------------------------

def test_settlement_charges_the_actual_full_process_wall(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    row = ledger.settle(reservation, actual_wall_s=473.25, returncode=0,
                        counters=_counters())
    assert row["charged_s"] == 473.25
    assert row["charge_basis"] == B.CHARGE_ACTUAL
    assert row["overran_bound"] is False
    assert ledger.charged()["global_s"] == 473.25


def test_an_overrun_is_charged_in_full_and_never_authorised_in_advance(tmp_path):
    """A forecast is not an authorisation: the real wall is charged even past the bound."""
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    row = ledger.settle(reservation, actual_wall_s=612.5, returncode=0,
                        counters=_counters())
    assert row["charged_s"] == 612.5
    assert row["overran_bound"] is True
    assert row["reserved_bound_s"] == 500.0


def test_a_failed_segment_is_charged_its_actual_wall(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0)
    row = ledger.settle(reservation, actual_wall_s=31.5, returncode=1,
                        counters=_counters(completed_control_transitions=0,
                                           completed_physics_steps=0))
    assert row["charged_s"] == 31.5
    assert row["returncode"] == 1
    assert row["completed_control_transitions"] == 0
    assert row["attempted_control_transitions"] == 8192


def test_replayed_and_partial_update_work_is_recorded(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    row = ledger.settle(reservation, actual_wall_s=120.0, returncode=0,
                        counters=_counters(replayed_control_transitions=4096,
                                           replayed_physics_steps=16384,
                                           partial_update_cost_s=2.5))
    assert row["replayed_control_transitions"] == 4096
    assert row["replayed_physics_steps"] == 16384
    assert row["partial_update_cost_s"] == 2.5


def test_settled_row_carries_every_required_field(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=500.0, allowance=60.0, segment_index=2)
    row = ledger.settle(reservation, actual_wall_s=100.0, returncode=0,
                        counters=_counters())
    for field in ("reserved_bound_s", "actual_wall_s", "charged_s", "charge_basis",
                  "stage", "recipe", "seed", "segment_index", "returncode",
                  "attempted_control_transitions", "completed_control_transitions",
                  "replayed_control_transitions", "attempted_physics_steps",
                  "completed_physics_steps", "replayed_physics_steps",
                  "partial_update_cost_s", "contended", "contention_at_launch",
                  "reservation_id", "reservation_index", "overran_bound"):
        assert field in row, field
    assert row["segment_index"] == 2
    assert row["stage"] == "screen" and row["recipe"] == "g990_e010"
    assert row["seed"] == 1001


def test_unknown_duration_is_charged_at_the_full_reserved_bound(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger, bound=555.0, allowance=45.0)
    row = ledger.settle_unknown(reservation, detail="child vanished; no wall recorded")
    assert row["charged_s"] == 555.0
    assert row["charge_basis"] == B.CHARGE_RESERVED_BOUND
    assert row["actual_wall_s"] is None
    assert row["returncode"] is None
    assert "vanished" in row["detail"]
    assert ledger.charged()["global_s"] == 555.0
    assert ledger.pending is None


def test_double_settlement_is_refused(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    ledger.settle(reservation, actual_wall_s=10.0, returncode=0, counters=_counters())
    with pytest.raises(B.DoubleSettlementError):
        ledger.settle(reservation, actual_wall_s=10.0, returncode=0,
                      counters=_counters())
    with pytest.raises(B.DoubleSettlementError):
        ledger.settle_unknown(reservation, detail="second attempt")


def test_settling_a_reservation_that_is_not_pending_is_refused(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    forged = B.Reservation(
        index=99, reservation_id="forged", stage="screen", recipe="g990_e010",
        seed=1001, segment_index=0, start_epoch=0, end_epoch=64,
        reserved_bound_s=600.0, shutdown_allowance_s=60.0,
        contention=B.unobserved_contention())
    with pytest.raises(B.SettlementError, match="pending"):
        ledger.settle(forged, actual_wall_s=1.0, returncode=0, counters=_counters())
    assert ledger.pending.reservation_id == reservation.reservation_id


def test_settlement_refuses_a_negative_or_nonfinite_wall(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(B.BudgetError):
            ledger.settle(reservation, actual_wall_s=bad, returncode=0,
                          counters=_counters())
    assert ledger.pending is not None  # still unsettled, still blocking


# --------------------------------------------------------------------------
# Caps: global, stage and seed, each binding in isolation
# --------------------------------------------------------------------------

def test_remaining_documents_the_stage_recipe_seed_keying_and_the_reserve(tmp_path):
    """Ruling 2, recorded rather than 'fixed': the arithmetic is intended."""
    doc = B.Remaining.__doc__
    assert "(recipe, seed)" in doc
    assert "stage and global caps dominate" in doc
    ledger = _ledger(tmp_path)
    # The per-seed cap really is per run: charging one run leaves its sibling whole.
    _charge(ledger, stage="screen", recipe="g990_e010", seed=1001, actual=900.0)
    assert ledger.remaining("screen", "g990_e010", 1001).seed_s == 300.0
    assert ledger.remaining("screen", "g990_e010", 1002).seed_s == 1200.0
    assert ledger.remaining("screen", "g998_e010", 1001).seed_s == 1200.0
    # ... while the stage cap is shared by all of them.
    for recipe, seed in (("g990_e010", 1002), ("g998_e010", 1001)):
        assert ledger.remaining("screen", recipe, seed).stage_s == 6300.0


def test_fresh_remaining_is_the_sealed_caps(tmp_path):
    ledger = _ledger(tmp_path)
    remaining = ledger.remaining("screen", "g990_e010", 1001)
    assert remaining.global_s == 28800
    assert remaining.stage_s == 7200
    assert remaining.seed_s == 1200
    assert remaining.binding == "seed"
    assert remaining.least == 1200
    assert remaining.call_cap == 600  # min(600, 1200)


def test_seed_cap_binds_in_isolation_and_wins_as_the_tightest(tmp_path):
    ledger = _ledger(tmp_path)
    _charge(ledger, stage="screen", recipe="g990_e010", seed=1001, actual=1000.0)
    remaining = ledger.remaining("screen", "g990_e010", 1001)
    assert (remaining.global_s, remaining.stage_s, remaining.seed_s) == (
        27800.0, 6200.0, 200.0)
    assert remaining.binding == "seed"
    # A different run of the same stage is untouched by this seed's charge.
    other = ledger.remaining("screen", "g990_e010", 1002)
    assert other.seed_s == 1200 and other.stage_s == 6200

    # bound + allowance == 200 exactly fits; one second more does not.
    reservation = _reserve(ledger, bound=150.0, allowance=50.0)
    ledger.settle(reservation, actual_wall_s=1.0, returncode=0, counters=_counters())
    with pytest.raises(B.BudgetExhaustedError, match="seed"):
        _reserve(ledger, bound=160.0, allowance=50.0)


def test_stage_cap_binds_in_isolation_and_wins_as_the_tightest(tmp_path):
    ledger = _ledger(tmp_path)
    # Seven of the eight screen runs consume 1,000 s each: stage charged 7,000 s.
    runs = [(recipe, seed) for recipe in P.CANONICAL_RECIPE_ORDER
            for seed in (1001, 1002)][:7]
    for recipe, seed in runs:
        _charge(ledger, stage="screen", recipe=recipe, seed=seed, actual=1000.0)
    remaining = ledger.remaining("screen", "g998_e000", 1002)  # the untouched run
    assert remaining.stage_s == 200.0
    assert remaining.seed_s == 1200.0
    assert remaining.global_s == 21800.0
    assert remaining.binding == "stage"
    with pytest.raises(B.BudgetExhaustedError, match="stage"):
        _reserve(ledger, recipe="g998_e000", seed=1002, bound=200.0, allowance=50.0)
    # Within the stage remainder it is allowed.
    assert _reserve(ledger, recipe="g998_e000", seed=1002, bound=150.0,
                    allowance=50.0) is not None


def test_global_cap_binds_in_isolation_once_overruns_eat_the_reserve(tmp_path):
    """The 900 s reserve is the only slack between the stage caps and 28,800 s.

    Stage caps sum to 27,900 s, so the global cap can only become the tightest
    once actual walls have overrun their stage allocations. That is exactly what
    the reserve exists to absorb, and it must still be enforced.
    """
    ledger = _ledger(tmp_path)
    # Two overrunning segments charge 14,350 s each: 28,700 s of 28,800 s.
    _charge(ledger, stage="confirm", recipe="g990_e010", seed=3001, actual=14350.0)
    _charge(ledger, stage="refine", recipe="g990_e010", seed=2001, actual=14350.0)
    assert ledger.charged()["global_s"] == 28700.0
    remaining = ledger.remaining("screen", "g990_e010", 1001)
    assert remaining.global_s == 100.0
    assert remaining.stage_s == 7200.0   # nothing was charged to the screen stage
    assert remaining.seed_s == 1200.0    # nor to this run
    assert remaining.binding == "global"
    assert remaining.call_cap == 100.0   # min(600, 100), not the 600 s call bound
    with pytest.raises(B.BudgetExhaustedError, match="global"):
        _reserve(ledger, bound=100.0, allowance=50.0)  # 150 > 100
    assert _reserve(ledger, bound=50.0, allowance=40.0) is not None


def test_remaining_refuses_an_unknown_stage_or_recipe(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(P.ProtocolError):
        ledger.remaining("screen_extra", "g990_e010", 1001)
    with pytest.raises(P.ProtocolError):
        ledger.remaining("screen", "g000_e000", 1001)
    with pytest.raises(B.BudgetError, match="scheduled"):
        ledger.remaining("screen", "g990_e010", 4242)


# --------------------------------------------------------------------------
# Shutdown allowance and the hard call bound
# --------------------------------------------------------------------------

def test_no_launch_without_an_explicit_positive_shutdown_allowance(tmp_path):
    ledger = _ledger(tmp_path)
    for bad in (0.0, -5.0, float("nan")):
        with pytest.raises(B.ShutdownAllowanceError):
            _reserve(ledger, bound=400.0, allowance=bad)
    assert ledger.pending is None


def test_bound_plus_allowance_must_fit_inside_the_600_second_call_bound(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(B.ShutdownAllowanceError, match="600"):
        _reserve(ledger, bound=600.0, allowance=1.0)
    # 460 s of work plus a 140 s shutdown allowance is exactly the call bound.
    reservation = _reserve(ledger, bound=float(P.SEGMENT_WORK_DEADLINE_S),
                           allowance=140.0)
    assert reservation.reserved_bound_s == 460.0
    assert (reservation.reserved_bound_s + reservation.shutdown_allowance_s
            == P.SEGMENT_CALL_BOUND_S)


def test_reserved_bound_must_be_positive_and_within_the_call_bound(tmp_path):
    ledger = _ledger(tmp_path)
    for bad in (0.0, -1.0, 601.0, float("inf")):
        with pytest.raises(B.BudgetError):
            _reserve(ledger, bound=bad, allowance=10.0)
    assert ledger.pending is None


def test_reserve_refuses_a_segment_longer_than_the_epoch_ceiling(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(B.BudgetError, match="64"):
        _reserve(ledger, start_epoch=0, end_epoch=65)


def test_reserve_refuses_an_unscheduled_seed_or_recipe(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(B.BudgetError, match="scheduled"):
        _reserve(ledger, seed=9999)
    with pytest.raises(P.ProtocolError):
        _reserve(ledger, recipe="g111_e111")


# --------------------------------------------------------------------------
# GPU contention: recorded provenance, never a gate
# --------------------------------------------------------------------------

def test_contention_is_observed_at_launch_and_mirrored_onto_the_settled_row(tmp_path):
    probe = FakeProbe({"contended": True, "utilization_percent": 40,
                       "compute_processes": ["a_desktop_application"]})
    ledger = _ledger(tmp_path, probe=probe)
    reservation = _reserve(ledger)
    assert probe.calls == 1
    assert reservation.contention["contended"] is True
    assert reservation.contention["utilization_percent"] == 40
    row = ledger.settle(reservation, actual_wall_s=100.0, returncode=0,
                        counters=_counters())
    assert row["contended"] is True
    assert row["contention_at_launch"]["compute_processes"] == [
        "a_desktop_application"]
    # Recording it changes neither the charge nor the decision to launch.
    assert row["charged_s"] == 100.0


def test_an_uncontended_observation_is_recorded_as_false_not_missing(tmp_path):
    probe = FakeProbe({"contended": False, "utilization_percent": 0})
    ledger = _ledger(tmp_path, probe=probe)
    reservation = _reserve(ledger)
    row = ledger.settle(reservation, actual_wall_s=1.0, returncode=0,
                        counters=_counters())
    assert row["contended"] is False


def test_no_probe_records_unobserved_and_unknown_never_false(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = _reserve(ledger)
    assert reservation.contention["status"] == B.CONTENTION_UNOBSERVED
    assert reservation.contention["contended"] is None
    row = ledger.settle(reservation, actual_wall_s=1.0, returncode=0,
                        counters=_counters())
    assert row["contended"] is None, "unknown contention must not read as absent"


def test_a_failing_or_malformed_probe_never_blocks_a_launch(tmp_path):
    failing = FakeProbe(raises=RuntimeError("no device query available"))
    ledger = _ledger(tmp_path, probe=failing)
    reservation = _reserve(ledger)
    assert reservation.contention["status"] == B.CONTENTION_PROBE_FAILED
    assert reservation.contention["contended"] is None
    assert "no device query" in reservation.contention["error"]
    ledger.settle(reservation, actual_wall_s=1.0, returncode=0, counters=_counters())

    for observation in ({"utilization_percent": 5}, {"contended": "yes"}, "nope", None):
        malformed = FakeProbe(observation)
        other = _ledger(tmp_path, probe=malformed,
                        name=f"ledger_{abs(hash(str(observation)))}.jsonl")
        held = _reserve(other)
        assert held.contention["status"] == B.CONTENTION_PROBE_MALFORMED
        assert held.contention["contended"] is None


def test_an_unencodable_payload_keeps_a_validly_reported_verdict(tmp_path):
    """A sibling key that will not serialise must not erase the verdict itself."""
    probe = FakeProbe({"contended": True, "handle": object()})
    ledger = _ledger(tmp_path, probe=probe)
    reservation = _reserve(ledger)
    assert reservation.contention["status"] == B.CONTENTION_PROBE_MALFORMED
    assert reservation.contention["contended"] is True
    assert "object object at" in reservation.contention["observation"]
    row = ledger.settle(reservation, actual_wall_s=1.0, returncode=0,
                        counters=_counters())
    assert row["contended"] is True
    # The row is still canonically encodable, which is what the chain needs.
    assert B.record_digest({k: v for k, v in row.items()
                            if k != "record_sha256"}) == row["record_sha256"]


def test_a_backwards_clock_is_clamped_and_recorded_not_hidden(tmp_path):
    """A host clock step must not break the chain or silently vanish."""
    clock = FakeClock(start=1_700_000_000.0, step=-50.0)  # steps backwards
    ledger = _ledger(tmp_path, clock=clock)
    reservation = _reserve(ledger)
    row = ledger.settle(reservation, actual_wall_s=7.0, returncode=0,
                        counters=_counters())
    assert row["clock_clamped"] is True
    assert row["posix_time"] == ledger.header["posix_time"]
    # The charge is the injected duration and is untouched by the clock.
    assert row["charged_s"] == 7.0
    # And the file still verifies, which a non-monotonic timestamp would not.
    assert B.BudgetLedger.open(ledger.path).charged()["global_s"] == 7.0


def test_contention_is_provenance_only_with_no_threshold_and_no_cause_claim(tmp_path):
    note = B.CONTENTION_NOTE.lower()
    assert "no cause" in note or "cause is claimed" in note
    assert "threshold" in note
    # There is no contention threshold anywhere in the module's public surface.
    assert not [name for name in dir(B) if "THRESHOLD" in name.upper()]
    # And a contended launch is never refused: contention appears in no refusal path.
    probe = FakeProbe({"contended": True})
    ledger = _ledger(tmp_path, probe=probe)
    assert _reserve(ledger) is not None


# --------------------------------------------------------------------------
# Process control: the ledger owns none of it
# --------------------------------------------------------------------------

def test_neither_module_contains_a_process_liveness_or_kill_pattern():
    """os.kill(pid, 0) is not a harmless liveness probe on Windows, and killing
    another job is never this module's business. The runner owns child timeouts
    through its own Popen handle."""
    for module in (B, P):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in ("os.kill", "signal.", "taskkill", "OpenProcess",
                          "psutil", "terminate(", "Popen", "pid, 0"):
            assert forbidden not in source, f"{forbidden} in {module.__name__}"
    assert "never kill" in B.PROCESS_CONTROL_NOTE.lower()


# --------------------------------------------------------------------------
# Locks
# --------------------------------------------------------------------------

_H1 = "a" * 64
_H2 = "b" * 64
_H3 = "c" * 64


def test_recipe_lock_binds_the_protocol_and_stage_two_evidence(tmp_path):
    path = tmp_path / "recipe.lock.json"
    record = B.write_lock(path, B.RECIPE_LOCK_KIND, {
        "protocol_sha256": P.protocol_digest(),
        "stage2_evidence_sha256": _H1,
        "promoted_recipe": "g998_e000",
    }, clock=FakeClock())
    assert record["kind"] == B.RECIPE_LOCK_KIND
    assert record["schema_version"] == B.LOCK_SCHEMA_VERSION
    assert len(record["lock_sha256"]) == 64
    assert B.read_lock(path) == record


def test_recipe_lock_must_not_require_future_stage_three_checkpoint_hashes(tmp_path):
    with pytest.raises(B.LockError, match="stage-3"):
        B.write_lock(tmp_path / "bad.lock.json", B.RECIPE_LOCK_KIND, {
            "protocol_sha256": P.protocol_digest(),
            "stage2_evidence_sha256": _H1,
            "selected_checkpoints": {"3001": _H2},
        }, clock=FakeClock())
    assert not (tmp_path / "bad.lock.json").exists()


def test_recipe_lock_refuses_a_missing_required_input(tmp_path):
    with pytest.raises(B.LockError, match="stage2_evidence_sha256"):
        B.write_lock(tmp_path / "bad.lock.json", B.RECIPE_LOCK_KIND, {
            "protocol_sha256": P.protocol_digest(),
        }, clock=FakeClock())


def test_selection_lock_binds_each_checkpoint_and_evaluation(tmp_path):
    path = tmp_path / "selection.lock.json"
    record = B.write_lock(path, B.SELECTION_LOCK_KIND, {
        "protocol_sha256": P.protocol_digest(),
        "selected_checkpoints": {"3001": _H1, "3002": _H2},
        "evaluations": {"3001": _H3, "3002": _H1},
    }, clock=FakeClock())
    flat = B.lock_input_digests(record)
    assert flat["selected_checkpoints.3002"] == _H2
    assert flat["evaluations.3001"] == _H3
    assert "protocol_sha256" in flat


def test_selection_lock_refuses_missing_evaluations(tmp_path):
    with pytest.raises(B.LockError, match="evaluations"):
        B.write_lock(tmp_path / "bad.lock.json", B.SELECTION_LOCK_KIND, {
            "protocol_sha256": P.protocol_digest(),
            "selected_checkpoints": {"3001": _H1},
        }, clock=FakeClock())


def test_lock_files_are_exclusively_created(tmp_path):
    path = tmp_path / "recipe.lock.json"
    inputs = {"protocol_sha256": P.protocol_digest(), "stage2_evidence_sha256": _H1}
    B.write_lock(path, B.RECIPE_LOCK_KIND, inputs, clock=FakeClock())
    with pytest.raises(B.LockError, match="exists"):
        B.write_lock(path, B.RECIPE_LOCK_KIND, inputs, clock=FakeClock())


def test_a_tampered_lock_file_is_refused(tmp_path):
    path = tmp_path / "recipe.lock.json"
    B.write_lock(path, B.RECIPE_LOCK_KIND, {
        "protocol_sha256": P.protocol_digest(),
        "stage2_evidence_sha256": _H1,
    }, clock=FakeClock())
    record = json.loads(path.read_text(encoding="utf-8"))
    record["inputs"]["stage2_evidence_sha256"] = _H2
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(B.LockError, match="lock_sha256"):
        B.read_lock(path)


def test_verify_lock_recomputes_every_hashed_input_before_confirmation(tmp_path):
    path = tmp_path / "selection.lock.json"
    B.write_lock(path, B.SELECTION_LOCK_KIND, {
        "protocol_sha256": P.protocol_digest(),
        "selected_checkpoints": {"3001": _H1},
        "evaluations": {"3001": _H2},
    }, clock=FakeClock())
    truth = {"protocol_sha256": P.protocol_digest(),
             "selected_checkpoints.3001": _H1,
             "evaluations.3001": _H2}
    assert B.verify_lock(path, truth.__getitem__)["kind"] == B.SELECTION_LOCK_KIND

    drifted = dict(truth, **{"selected_checkpoints.3001": _H3})
    with pytest.raises(B.LockError, match="selected_checkpoints.3001"):
        B.verify_lock(path, drifted.__getitem__)

    def missing(name):
        raise KeyError(name)

    with pytest.raises(B.LockError):
        B.verify_lock(path, missing)


def test_a_required_input_supplied_as_a_scalar_is_refused_at_write_time(tmp_path):
    """F1. A declared required input must never be present-but-unbound.

    ``_flatten`` binds a value only when it is a mapping of digests or its name
    ends ``_sha256``. A required input supplied as any other scalar was
    therefore written into the lock while escaping the verified digest set, so
    the lock verified clean without that input ever being checked -- a fail-open
    in the one mechanism whose entire job is tamper evidence.
    """
    for index, value in enumerate(("epoch_96.pt", 7, ["epoch_96.pt"], None)):
        target = tmp_path / f"selection_{index}.json"
        with pytest.raises(B.LockError, match="selected_checkpoints"):
            B.write_lock(target, B.SELECTION_LOCK_KIND, {
                "protocol_sha256": P.protocol_digest(),
                "selected_checkpoints": value,
                "evaluations": {"3001": _H2},
            }, clock=FakeClock())
        assert not target.exists(), "a refused lock must leave no file behind"


def test_an_unbound_required_input_can_never_reach_a_passing_verify_lock(tmp_path):
    """F1, stated as the postcondition: no lock exists to verify, so none passes."""
    path = tmp_path / "selection.lock.json"
    with pytest.raises(B.LockError):
        B.write_lock(path, B.SELECTION_LOCK_KIND, {
            "protocol_sha256": P.protocol_digest(),
            "selected_checkpoints": "epoch_96.pt",
            "evaluations": {"3001": _H2},
        }, clock=FakeClock())
    # Previously this wrote a lock whose input_names omitted selected_checkpoints
    # and which then verified clean against a resolver that never saw it.
    truth = {"protocol_sha256": P.protocol_digest(), "evaluations.3001": _H2}
    with pytest.raises(B.LockError):
        B.verify_lock(path, truth.__getitem__)


def test_a_scalar_required_input_is_refused_for_the_recipe_lock_too(tmp_path):
    path = tmp_path / "recipe.lock.json"
    with pytest.raises(B.LockError, match="stage2_evidence_sha256"):
        B.write_lock(path, B.RECIPE_LOCK_KIND, {
            "protocol_sha256": P.protocol_digest(),
            "stage2_evidence_sha256": "epoch_96.pt",
        }, clock=FakeClock())
    assert not path.exists()


#: Every declared required input, with a value that is legitimately bindable.
#: A newly declared required input that is missing here fails the enumeration
#: test below, so it cannot silently escape the digest.
_BINDABLE_REQUIRED = {
    "protocol_sha256": "a" * 64,
    "stage2_evidence_sha256": "b" * 64,
    "selected_checkpoints": {"3001": "c" * 64, "3002": "d" * 64},
    "evaluations": {"3001": "e" * 64, "3002": "f" * 64},
}


def test_every_declared_required_input_is_provably_bound_in_the_digest(tmp_path):
    """F1's postcondition, enumerated so a future added input cannot escape."""
    kinds = {
        B.RECIPE_LOCK_KIND: B.RECIPE_LOCK_REQUIRED_INPUTS,
        B.SELECTION_LOCK_KIND: B.SELECTION_LOCK_REQUIRED_INPUTS,
    }
    declared = {name for names in kinds.values() for name in names}
    assert declared <= set(_BINDABLE_REQUIRED), (
        "a required lock input has no bindable fixture value, so its boundness "
        f"is untested: {sorted(declared - set(_BINDABLE_REQUIRED))}")

    for kind, required in kinds.items():
        assert required, f"{kind} declares no required inputs"
        inputs = {name: _BINDABLE_REQUIRED[name] for name in required}
        record = B.write_lock(tmp_path / f"{kind}.json", kind, inputs,
                              clock=FakeClock())
        bound = set(B.lock_input_digests(record))
        for name in required:
            assert name in bound or any(key.startswith(f"{name}.") for key in bound), (
                f"{kind} lock declares {name!r} but it is absent from the digest")
        # Nothing bound is invented, either: every bound name traces to an input.
        for key in bound:
            assert key.split(".", 1)[0] in inputs


def test_free_form_metadata_is_still_allowed_when_it_is_not_required(tmp_path):
    """The F1 guard must not turn optional metadata into a refusal."""
    record = B.write_lock(tmp_path / "recipe.lock.json", B.RECIPE_LOCK_KIND, {
        "protocol_sha256": P.protocol_digest(),
        "stage2_evidence_sha256": _H1,
        "promoted_recipe": "g998_e000",
        "segment_count": 6,
    }, clock=FakeClock())
    bound = set(B.lock_input_digests(record))
    assert bound == {"protocol_sha256", "stage2_evidence_sha256"}
    assert record["inputs"]["promoted_recipe"] == "g998_e000"


def test_call_cap_docstring_states_the_reservable_bound_accurately(tmp_path):
    """F2. call_cap is not by itself a reservable bound.

    ``reserve`` also requires ``bound + allowance`` to fit, so the largest
    admissible ``reserved_bound_s`` is ``call_cap - shutdown_allowance_s``.
    """
    doc = B.Remaining.call_cap.__doc__
    assert "shutdown allowance" in doc
    assert "reserved_bound_s" in doc
    assert "the most a single call may reserve" not in doc

    # And the docstring's claim is true of the enforcement.
    ledger = _ledger(tmp_path)
    remaining = ledger.remaining("screen", "g990_e010", 1001)
    assert remaining.call_cap == 600.0
    with pytest.raises(B.ShutdownAllowanceError):
        _reserve(ledger, bound=remaining.call_cap, allowance=1.0)
    largest = remaining.call_cap - 1.0
    assert _reserve(ledger, bound=largest, allowance=1.0).reserved_bound_s == 599.0


def test_lock_refuses_a_non_hex_digest_and_an_unknown_kind(tmp_path):
    with pytest.raises(B.LockError, match="sha256"):
        B.write_lock(tmp_path / "a.json", B.RECIPE_LOCK_KIND, {
            "protocol_sha256": "not-a-hash", "stage2_evidence_sha256": _H1,
        }, clock=FakeClock())
    with pytest.raises(B.LockError, match="kind"):
        B.write_lock(tmp_path / "b.json", "some_other_lock", {
            "protocol_sha256": P.protocol_digest(),
        }, clock=FakeClock())


# --------------------------------------------------------------------------
# Accounting shape
# --------------------------------------------------------------------------

def test_charged_totals_are_keyed_by_global_stage_and_run(tmp_path):
    ledger = _ledger(tmp_path)
    _charge(ledger, stage="screen", recipe="g990_e010", seed=1001, actual=100.0)
    _charge(ledger, stage="screen", recipe="g990_e010", seed=1002, actual=250.0)
    _charge(ledger, stage="refine", recipe="g998_e010", seed=2001, actual=400.0)
    charged = ledger.charged()
    assert charged["global_s"] == 750.0
    assert charged["stage_s"] == {"screen": 350.0, "refine": 400.0}
    assert charged["run_s"]["screen|g990_e010|1002"] == 250.0
    assert charged["settled_rows"] == 3


def test_run_directory_and_ledger_name_are_under_an_ignored_path():
    assert B.DEFAULT_RUN_DIR.startswith("logs/")
    assert B.LEDGER_FILENAME.endswith(".jsonl")
    ignored = Path(__file__).resolve().parents[2] / ".gitignore"
    assert "logs/" in ignored.read_text(encoding="utf-8").splitlines()


def test_the_ledger_file_is_append_only_in_practice(tmp_path):
    """Earlier bytes are never rewritten: the prefix is stable across appends."""
    path = tmp_path / "budget_ledger.jsonl"
    ledger = _ledger(tmp_path)
    snapshots = [path.read_bytes()]
    reservation = _reserve(ledger)
    snapshots.append(path.read_bytes())
    ledger.settle(reservation, actual_wall_s=5.0, returncode=0, counters=_counters())
    snapshots.append(path.read_bytes())
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later.startswith(earlier)
        assert len(later) > len(earlier)
