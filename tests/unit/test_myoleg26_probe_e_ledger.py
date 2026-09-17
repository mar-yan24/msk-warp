"""Unit tests for the probe-E amendment's SEPARATE, carried-forward ledger.

Fake clocks and injected walls only. No simulator, no CUDA, no Warp, no torch,
no subprocess, no sleep, no GPU query, no real segment. Every wall time below is
an injected number, so this file consumes zero of the 28,800 s training budget
and none of the 1,800 s diagnostic budget.

The properties under test
-------------------------
* A ledger created **without** carry-forward behaves exactly as before: same
  header keys, same ``charged()`` shape. Nothing about stage 1's sealed ledger
  changes.
* The amendment's ledger is a **separate file** with its own protocol digest, so
  each ledger refuses the other's protocol and no job can be charged to the
  wrong one.
* Its opening balance is **carried forward** from the sealed ledger, so no cap
  is reset. A fresh ledger is refused for a protocol that requires carry-forward,
  and carry-forward from a ledger of the *same* protocol digest is refused too --
  that would be a second ledger for one protocol, i.e. exactly the escape route
  a new ledger must never be.
* The append-only hash chain, the pending-reservation trap, the spawn marker and
  the two settlement shapes all keep working on a carried ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msk_warp.analysis import ppo_v2_budget as B
from msk_warp.analysis import ppo_v2_protocol as P
from msk_warp.analysis import ppo_v2_protocol_e as E


class FakeClock:
    """A monotonically advancing injected clock. Never sleeps, never reads time."""

    def __init__(self, start=1_700_000_000.0, step=1.0):
        self.now = float(start)
        self.step = float(step)

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


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


class _CountingProbe:
    """Counts observations, so a refusal can be shown to precede the probe."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return B.unobserved_contention()


def _sealed(tmp_path, *, name="sealed_ledger.jsonl", charges=()):
    """A stand-in for stage 1's sealed ledger, with injected settled charges."""
    ledger = B.BudgetLedger.create(tmp_path / name,
                                   protocol_digest=P.protocol_digest(),
                                   clock=FakeClock())
    for stage, recipe, seed, actual in charges:
        reservation = ledger.reserve(
            stage=stage, recipe=recipe, seed=seed, segment_index=0,
            start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
            shutdown_allowance_s=60.0, create=None)
        ledger.settle(reservation, actual_wall_s=actual, returncode=0,
                      counters=_counters())
    return ledger


STAGE_1_LIKE = (("screen", "g990_e010", 1001, 400.0),
                ("screen", "g990_e010", 1002, 350.0),
                ("screen", "g998_e000", 1001, 300.0))


def _amended(tmp_path, sealed, *, name="probe_e_ledger.jsonl"):
    carried = B.carry_forward(sealed.path)
    return B.BudgetLedger.create(
        tmp_path / name, protocol_digest=E.protocol_digest(), protocol=E,
        carried=carried, provenance=E.ledger_provenance(), clock=FakeClock())


# --------------------------------------------------------------------------
# Unit 1 -- nothing changes for a ledger without carry-forward
# --------------------------------------------------------------------------

def test_a_ledger_without_carry_forward_has_exactly_the_old_header_keys(tmp_path):
    """Behavioural. Stage 1's sealed header shape is not altered by this work."""
    ledger = _sealed(tmp_path)
    header = json.loads(ledger.path.read_text(encoding="utf-8").splitlines()[0])
    assert set(header) == {
        "index", "kind", "prev_sha256", "posix_time", "utc", "schema_version",
        "protocol_digest", "total_wall_cap_s", "reserve_wall_cap_s",
        "segment_max_epochs", "segment_work_deadline_s", "segment_call_bound_s",
        "record_sha256"}
    assert ledger.carried_forward is None


def test_charged_keeps_its_exact_shape_without_carry_forward(tmp_path):
    ledger = _sealed(tmp_path, charges=STAGE_1_LIKE)
    charged = ledger.charged()
    assert set(charged) == {"global_s", "stage_s", "run_s", "settled_rows"}
    assert charged["global_s"] == 1050.0
    assert charged["settled_rows"] == 3
    assert ledger.carried_totals() == {"global_s": 0.0, "stage_s": {},
                                       "run_s": {}, "settled_rows": 0}


def test_the_default_protocol_is_the_sealed_one(tmp_path):
    ledger = _sealed(tmp_path)
    assert ledger.protocol is P
    ledger.remaining("screen", "g990_e010", 1001)
    with pytest.raises(P.ProtocolError):
        ledger.remaining("screen", E.PRIMARY_RECIPE, 1001)


# --------------------------------------------------------------------------
# Unit 2 -- carry_forward reads the sealed ledger and totals it
# --------------------------------------------------------------------------

def test_carry_forward_records_the_source_identity_and_its_totals(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    carried = B.carry_forward(sealed.path)
    assert carried.source_path == str(sealed.path.resolve())
    assert carried.source_sha256 == B.sha256_bytes(sealed.path.read_bytes())
    assert carried.source_rows == len(sealed.rows)
    assert carried.source_protocol_digest == P.protocol_digest()
    assert carried.global_s == 1050.0
    assert carried.stage_s == {"screen": 1050.0}
    assert carried.run_s["screen|g990_e010|1002"] == 350.0
    assert carried.settled_rows == 3


def test_carry_forward_verifies_the_source_chain(tmp_path):
    """Behavioural. A tampered source is refused, not silently carried."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    lines = sealed.path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[-1])
    row["charged_s"] = 1.0
    lines[-1] = json.dumps(row, sort_keys=True)
    sealed.path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(B.LedgerIntegrityError):
        B.carry_forward(sealed.path)


def test_carry_forward_refuses_a_source_with_an_unsettled_reservation(tmp_path):
    """Behavioural. An open hold means the source's spend is not yet known, so
    carrying it forward would understate it."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    sealed.reserve(stage="screen", recipe="g998_e010", seed=1001,
                   segment_index=0, start_epoch=0, end_epoch=64,
                   reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    with pytest.raises(B.LedgerIntegrityError, match="unsettled"):
        B.carry_forward(sealed.path)


def test_carry_forward_never_writes_to_the_source(tmp_path):
    """Behavioural, and the guarantee the whole amendment rests on: the sealed
    ledger is read and stays byte-identical."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    before = sealed.path.read_bytes()
    B.carry_forward(sealed.path)
    _amended(tmp_path, sealed)
    assert sealed.path.read_bytes() == before


# --------------------------------------------------------------------------
# Unit 3 -- the new ledger opens at the carried spend: no cap is reset
# --------------------------------------------------------------------------

def test_the_amended_ledger_opens_at_the_sealed_spend(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    charged = amended.charged()
    assert charged["global_s"] == 1050.0
    assert charged["stage_s"] == {"screen": 1050.0}
    assert charged["settled_rows"] == 0          # no local settlement yet
    assert amended.carried_forward["settled_rows"] == 3


def test_the_global_and_stage_remaining_are_unchanged_by_the_new_ledger(tmp_path):
    """Behavioural. The budget must not move: the amended ledger reports the
    same global and stage headroom the sealed one does."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    sealed_remaining = sealed.remaining("screen", "g990_e010", 1001)
    amended_remaining = amended.remaining("screen", E.PRIMARY_RECIPE, 1001)
    assert amended_remaining.global_s == sealed_remaining.global_s
    assert amended_remaining.stage_s == sealed_remaining.stage_s
    assert amended_remaining.global_s == P.TOTAL_WALL_CAP_S - 1050.0


def test_only_the_never_run_cell_opens_at_a_fresh_per_seed_allowance(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    fresh = amended.remaining("screen", E.PRIMARY_RECIPE, 1001)
    assert fresh.seed_s == P.stage("screen").per_seed_safety_cap_s
    # The sealed cell's own per-seed spend is still carried, so the amendment
    # cannot reopen a stage-1 cell's allowance either.
    assert amended.carried_forward["run_s"]["screen|g990_e010|1001"] == 400.0


def test_a_new_ledger_cannot_reset_a_cap(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    assert amended.charged()["global_s"] > 0.0
    assert amended.remaining("screen", E.PRIMARY_RECIPE,
                             1001).global_s < P.TOTAL_WALL_CAP_S


def test_a_settlement_on_the_amended_ledger_adds_to_the_carried_total(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    reservation = amended.reserve(
        stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001, segment_index=0,
        start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0)
    amended.mark_spawn(reservation, detail="argv_sha256=" + "0" * 64)
    assert amended.spawned(reservation) is True
    amended.settle(reservation, actual_wall_s=320.0, returncode=0,
                   counters=_counters())
    charged = amended.charged()
    assert charged["global_s"] == 1370.0
    assert charged["settled_rows"] == 1
    assert charged["run_s"]["screen|" + E.PRIMARY_RECIPE + "|1001"] == 320.0
    assert "screen|g990_e010|1001" not in charged["run_s"] or (
        charged["run_s"]["screen|g990_e010|1001"] == 400.0)


# --------------------------------------------------------------------------
# Unit 4 -- carry-forward is required, and is not an escape route
# --------------------------------------------------------------------------

def test_a_protocol_that_requires_carry_forward_refuses_a_fresh_ledger(tmp_path):
    with pytest.raises(B.LedgerIntegrityError, match="carried forward"):
        B.BudgetLedger.create(tmp_path / "fresh.jsonl",
                              protocol_digest=E.protocol_digest(), protocol=E,
                              clock=FakeClock())


def test_carry_forward_from_the_same_protocol_digest_is_refused(tmp_path):
    """Behavioural. A second ledger for one protocol would be a migration or a
    rewrite of the first, which is exactly what is forbidden."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    carried = B.carry_forward(sealed.path)
    with pytest.raises(B.LedgerIntegrityError, match="same protocol"):
        B.BudgetLedger.create(tmp_path / "second_sealed.jsonl",
                              protocol_digest=P.protocol_digest(),
                              carried=carried, clock=FakeClock())


def test_a_ledger_never_carries_itself_forward(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    carried = B.carry_forward(sealed.path)
    with pytest.raises(B.LedgerIntegrityError, match="itself"):
        B.BudgetLedger.create(sealed.path.parent / sealed.path.name,
                              protocol_digest=E.protocol_digest(), protocol=E,
                              carried=carried, clock=FakeClock())


def test_the_sealed_ledger_is_still_never_overwritten(tmp_path):
    """Behavioural. The source is a DIFFERENT file here, so the self-carry guard
    cannot fire and the refusal really is the overwrite refusal."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    other = _sealed(tmp_path, name="other_sealed.jsonl", charges=STAGE_1_LIKE)
    before = sealed.path.read_bytes()
    with pytest.raises(B.LedgerIntegrityError,
                       match="never overwritten, replaced or started fresh"):
        B.BudgetLedger.create(sealed.path, protocol_digest=E.protocol_digest(),
                              protocol=E, carried=B.carry_forward(other.path),
                              clock=FakeClock())
    assert sealed.path.read_bytes() == before


# --------------------------------------------------------------------------
# Unit 5 -- each ledger refuses the other protocol
# --------------------------------------------------------------------------

def test_the_sealed_ledger_refuses_the_amendment_digest(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    sealed.assert_protocol_unchanged(P.protocol_digest())
    with pytest.raises(B.LedgerIntegrityError):
        sealed.assert_protocol_unchanged(E.protocol_digest())


def test_the_amended_ledger_refuses_the_sealed_digest(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    amended.assert_protocol_unchanged(E.protocol_digest())
    with pytest.raises(B.LedgerIntegrityError):
        amended.assert_protocol_unchanged(P.protocol_digest())
    assert amended.protocol_digest != sealed.protocol_digest


def test_the_amended_ledger_refuses_a_sealed_stage_1_arm(tmp_path):
    """Behavioural. A stage-1 arm cannot be charged to the amendment's ledger."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    with pytest.raises(P.ProtocolError, match="sealed"):
        amended.remaining("screen", "g990_e010", 1001)
    with pytest.raises(P.ProtocolError, match="sealed"):
        amended.reserve(stage="screen", recipe="g990_e010", seed=1001,
                        segment_index=0, start_epoch=0, end_epoch=64,
                        reserved_bound_s=540.0, shutdown_allowance_s=60.0)


def test_the_probe_e_arm_can_never_be_charged_to_a_stage_1_run_key(tmp_path):
    """Behavioural. The run key carries the recipe, and the names are disjoint."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    reservation = amended.reserve(
        stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001, segment_index=0,
        start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0)
    amended.settle(reservation, actual_wall_s=320.0, returncode=0,
                   counters=_counters())
    keys = set(amended.charged()["run_s"])
    assert "screen|" + E.PRIMARY_RECIPE + "|1001" in keys
    for name in P.CANONICAL_RECIPE_ORDER:
        assert amended.charged()["run_s"].get("screen|" + name + "|1001",
                                              0.0) != 320.0


# --------------------------------------------------------------------------
# Unit 6 -- the header records its descent, and the chain still verifies
# --------------------------------------------------------------------------

def test_the_amended_header_records_the_source_and_the_descent(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    header = json.loads(amended.path.read_text(encoding="utf-8").splitlines()[0])
    assert header["protocol_digest"] == E.protocol_digest()
    assert header["carried_forward"]["source_protocol_digest"] == P.protocol_digest()
    assert header["carried_forward"]["source_sha256"] == B.sha256_bytes(
        sealed.path.read_bytes())
    assert header["protocol_provenance"]["amendment_id"] == E.AMENDMENT_ID
    assert header["protocol_provenance"]["parent_protocol_digest"] == (
        P.protocol_digest())
    assert header["total_wall_cap_s"] == P.TOTAL_WALL_CAP_S


def test_the_amended_ledger_reopens_and_reverifies(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    reopened = B.BudgetLedger.open(amended.path, protocol=E)
    assert reopened.protocol_digest == E.protocol_digest()
    assert reopened.charged()["global_s"] == 1050.0
    assert reopened.carried_forward == amended.carried_forward


def test_a_tampered_carried_header_is_refused(tmp_path):
    """Behavioural. The carried block is inside the hash-chained header row, so
    editing it breaks the record digest."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    lines = amended.path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["carried_forward"]["global_s"] = 0.0
    lines[0] = json.dumps(header, sort_keys=True)
    amended.path.write_text("\n".join(lines) + "\n", encoding="utf-8",
                            newline="\n")
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(amended.path, protocol=E)


def test_the_pending_reservation_trap_still_holds_on_a_carried_ledger(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    amended.reserve(stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001,
                    segment_index=0, start_epoch=0, end_epoch=64,
                    reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    with pytest.raises(B.PendingReservationError):
        amended.reserve(stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001,
                        segment_index=0, start_epoch=0, end_epoch=64,
                        reserved_bound_s=540.0, shutdown_allowance_s=60.0)


def test_the_amended_ledger_is_append_only(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    prefix = amended.path.read_bytes()
    reservation = amended.reserve(
        stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001, segment_index=0,
        start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0)
    amended.settle_aborted(reservation, detail="nothing was started")
    after = amended.path.read_bytes()
    assert after.startswith(prefix)
    assert len(after) > len(prefix)


def test_the_real_sealed_ledger_path_is_the_amendments_declared_parent():
    """Transcription. The amendment names the canonical sealed ledger, and its
    own ledger is a different file under the same ignored tree."""
    assert E.PARENT_LEDGER == "logs/myoleg26_ppo_v2/budget_ledger.jsonl"
    assert Path(E.LEDGER).name == B.LEDGER_FILENAME
    assert E.LEDGER != E.PARENT_LEDGER
# --------------------------------------------------------------------------
# Unit 7 -- fix round 1: the parent balance is re-checked at EVERY reserve
#
# A carried opening balance is a snapshot of the parent ledger at one instant.
# If the parent settles anything afterwards, the descendant's balance is stale
# and it can over-grant the global cap -- it would be drawing against money
# already spent. The snapshot alone is a convention; these tests make it
# structural. Every reserve re-reads the parent, recomputes its charge from its
# own rows, and refuses on any divergence. A divergence is never reconciled by
# the descendant: it takes an explicit, recorded re-snapshot.
# --------------------------------------------------------------------------

def _advance_parent(sealed, *, seconds=500.0, recipe="g998_e010", seed=1001):
    """Settle one more fake segment on the parent, after the snapshot was taken."""
    reservation = sealed.reserve(
        stage="screen", recipe=recipe, seed=seed, segment_index=0,
        start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0, create=None)
    return sealed.settle(reservation, actual_wall_s=seconds, returncode=0,
                         counters=_counters())


def _reserve(ledger, *, segment_index=0):
    return ledger.reserve(
        stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001,
        segment_index=segment_index, start_epoch=0, end_epoch=64,
        reserved_bound_s=540.0, shutdown_allowance_s=60.0)


def test_a_parent_that_has_not_moved_reserves_normally(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    amended.assert_parent_unmoved()
    reservation = _reserve(amended)
    assert amended.pending.reservation_id == reservation.reservation_id


def test_a_parent_whose_charge_advanced_refuses_the_next_reserve(tmp_path):
    """Behavioural, and the whole point of this round. The refusal names both
    totals and both sha256s, so a reader sees exactly what moved."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    before_sha = B.sha256_bytes(sealed.path.read_bytes())
    _advance_parent(sealed, seconds=500.0)
    after_sha = B.sha256_bytes(sealed.path.read_bytes())

    with pytest.raises(B.ParentLedgerMovedError) as raised:
        _reserve(amended)
    message = str(raised.value)
    assert "1050.0" in message          # the carried snapshot total
    assert "1550.0" in message          # the parent's recomputed total
    assert before_sha in message
    assert after_sha in message
    assert "re-snapshot" in message

    with pytest.raises(B.ParentLedgerMovedError):
        amended.assert_parent_unmoved()


def test_the_drift_refusal_is_fail_closed_and_durable_free(tmp_path):
    """Behavioural. The refusal precedes the reservation row and the contention
    observation, so it commits nothing and costs nothing."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    probe = _CountingProbe()
    carried = B.carry_forward(sealed.path)
    amended = B.BudgetLedger.create(
        tmp_path / "probe_e_ledger.jsonl", protocol_digest=E.protocol_digest(),
        protocol=E, carried=carried, provenance=E.ledger_provenance(),
        clock=FakeClock(), contention_probe=probe)
    before = amended.path.read_bytes()
    _advance_parent(sealed)
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve(amended)
    assert amended.path.read_bytes() == before
    assert probe.calls == 0


def test_a_parent_holding_a_new_unsettled_reservation_refuses_the_reserve(tmp_path):
    """Behavioural. An open hold on the parent is drift in progress: its charge
    is not yet known, so the descendant must not commit against it."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    sealed.reserve(stage="screen", recipe="g998_e010", seed=1001,
                   segment_index=0, start_epoch=0, end_epoch=64,
                   reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve(amended)


def test_an_unreadable_parent_refuses_rather_than_falling_back(tmp_path):
    """Behavioural. Fail closed: a missing parent is never excused by the
    snapshot the descendant happens to be carrying."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    sealed.path.unlink()
    with pytest.raises(B.LedgerIntegrityError, match="parent ledger"):
        _reserve(amended)


def test_a_chain_broken_parent_refuses_rather_than_falling_back(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    lines = sealed.path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[-1])
    row["charged_s"] = 1.0
    lines[-1] = json.dumps(row, sort_keys=True)
    sealed.path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(B.LedgerIntegrityError, match="parent ledger"):
        _reserve(amended)


def test_a_shrunken_parent_refuses_rather_than_falling_back(tmp_path):
    """Behavioural. Truncating the LAST line of an append-only file leaves a
    self-consistent chain -- the gap disclosed in ``LEDGER_TAIL_NOTE`` -- so a
    fresh cross-process read cannot call it an integrity break. It is caught as
    a MOVED parent instead, on the recomputed total, the byte sha256 AND the row
    count, which is the fail-closed outcome that matters: the snapshot is never
    accepted in its place."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    lines = sealed.path.read_text(encoding="utf-8").splitlines()
    sealed.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8",
                           newline="\n")
    with pytest.raises(B.BudgetError) as raised:
        _reserve(amended)
    message = str(raised.value)
    assert "parent ledger" in message
    assert "1050.0" in message and "750.0" in message
    assert "7 rows against 6" in message


def test_the_check_runs_at_every_reserve_not_only_at_snapshot_time(tmp_path):
    """Behavioural. The first reserve succeeds; the parent then moves; the
    SECOND reserve refuses. A one-time check at snapshot time would miss this."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    first = _reserve(amended)
    amended.settle(first, actual_wall_s=320.0, returncode=0, counters=_counters())
    assert amended.charged()["global_s"] == 1370.0
    _advance_parent(sealed, seconds=500.0)
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve(amended, segment_index=0)


def test_a_ledger_without_a_carried_balance_is_unaffected(tmp_path):
    """Behavioural. The sealed campaign ledger carries nothing, so it has no
    parent to re-check and its reserve path is untouched."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    sealed.assert_parent_unmoved()
    reservation = sealed.reserve(
        stage="screen", recipe="g990_e000", seed=1001, segment_index=0,
        start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0)
    assert sealed.pending.reservation_id == reservation.reservation_id


# -- the explicit re-snapshot ----------------------------------------------

def test_a_resnapshot_records_the_new_balance_beside_the_old_one(tmp_path):
    """Behavioural. The audit trail shows what changed and when: one appended
    row carrying BOTH the superseded block and the new one."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    old = amended.carried_forward
    _advance_parent(sealed, seconds=500.0)

    row = amended.resnapshot_parent(
        detail="the user authorised further stage-1 work after the snapshot")
    assert row["kind"] == B.RESNAPSHOT_KIND
    assert row["previous_carried"]["global_s"] == 1050.0
    assert row["previous_carried"]["source_sha256"] == old["source_sha256"]
    assert row["carried_forward"]["global_s"] == 1550.0
    assert row["carried_forward"]["source_sha256"] == B.sha256_bytes(
        sealed.path.read_bytes())
    assert row["detail"].startswith("the user authorised")
    assert row["utc"] and row["posix_time"]

    # And the reserve it was blocking now proceeds, against the NEW balance.
    amended.assert_parent_unmoved()
    assert amended.charged()["global_s"] == 1550.0
    assert amended.remaining("screen", E.PRIMARY_RECIPE, 1001).global_s == (
        P.TOTAL_WALL_CAP_S - 1550.0)
    _reserve(amended)


def test_the_resnapshot_history_keeps_every_superseded_balance(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    _advance_parent(sealed, seconds=500.0)
    amended.resnapshot_parent(detail="first authorised advance")
    _advance_parent(sealed, seconds=300.0, recipe="g998_e000")
    amended.resnapshot_parent(detail="second authorised advance")
    history = amended.carried_history()
    assert [entry["global_s"] for entry in history] == [1050.0, 1550.0, 1850.0]
    assert amended.charged()["global_s"] == 1850.0


def test_a_resnapshot_of_an_unmoved_parent_is_refused(tmp_path):
    """Behavioural. A re-snapshot is only meaningful when the parent moved; a
    no-op row would be audit noise."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    with pytest.raises(B.LedgerIntegrityError, match="has not moved"):
        amended.resnapshot_parent(detail="nothing happened")


def test_a_resnapshot_is_refused_while_a_reservation_is_pending(tmp_path):
    """Behavioural. Re-snapshotting under an open hold would move the balance
    the hold was checked against."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    _reserve(amended)
    _advance_parent(sealed, seconds=500.0)
    with pytest.raises(B.PendingReservationError):
        amended.resnapshot_parent(detail="mid-flight")


def test_a_resnapshot_cannot_lower_the_carried_total(tmp_path):
    """Behavioural. An append-only parent's charge only grows; a lower total
    means the parent was rewritten, and adopting it would hand back spend."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    smaller = _sealed(tmp_path, name="smaller.jsonl",
                      charges=(("screen", "g990_e010", 1001, 10.0),))
    header = json.loads(amended.path.read_text(encoding="utf-8").splitlines()[0])
    assert header["carried_forward"]["global_s"] == 1050.0
    with pytest.raises(B.LedgerIntegrityError, match="never decrease"):
        amended.resnapshot_parent(detail="pointing at a smaller ledger",
                                  path=smaller.path)


def test_a_resnapshot_requires_an_explicit_detail(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    _advance_parent(sealed, seconds=500.0)
    with pytest.raises(B.LedgerIntegrityError, match="detail"):
        amended.resnapshot_parent(detail="  ")


def test_there_is_no_override_for_a_moved_parent():
    """Behavioural, read off the source: no flag, keyword or environment switch
    lets a stale snapshot be good enough."""
    source = Path(B.__file__).read_text(encoding="utf-8")
    for forbidden in ("allow_stale", "ignore_parent", "force_carried",
                      "skip_parent", "os.environ"):
        assert forbidden not in source


def test_the_resnapshot_row_keeps_the_hash_chain_and_reopens(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    prefix = amended.path.read_bytes()
    _advance_parent(sealed, seconds=500.0)
    amended.resnapshot_parent(detail="authorised advance")
    raw = amended.path.read_bytes()
    assert raw.startswith(prefix) and len(raw) > len(prefix)
    reopened = B.BudgetLedger.open(amended.path, protocol=E)
    assert reopened.charged()["global_s"] == 1550.0
    assert reopened.carried_forward["global_s"] == 1550.0
    assert [entry["global_s"] for entry in reopened.carried_history()] == [
        1050.0, 1550.0]
    reopened.assert_parent_unmoved()


def test_a_tampered_resnapshot_row_is_refused(tmp_path):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    _advance_parent(sealed, seconds=500.0)
    amended.resnapshot_parent(detail="authorised advance")
    lines = amended.path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[-1])
    row["carried_forward"]["global_s"] = 0.0
    lines[-1] = json.dumps(row, sort_keys=True)
    amended.path.write_text("\n".join(lines) + "\n", encoding="utf-8",
                            newline="\n")
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(amended.path, protocol=E)


def test_the_parent_ledger_stays_byte_identical_through_all_of_it(tmp_path):
    """Behavioural. The descendant READS the parent and never writes it -- the
    only writes below are the test's own, through the parent's public API."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed)
    quiet = sealed.path.read_bytes()
    amended.assert_parent_unmoved()
    reservation = _reserve(amended)
    amended.mark_spawn(reservation, detail="argv_sha256=" + "0" * 64)
    amended.settle(reservation, actual_wall_s=320.0, returncode=0,
                   counters=_counters())
    assert sealed.path.read_bytes() == quiet

    _advance_parent(sealed, seconds=500.0)
    moved = sealed.path.read_bytes()
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve(amended, segment_index=1)
    assert sealed.path.read_bytes() == moved
    amended.resnapshot_parent(detail="authorised advance")
    assert sealed.path.read_bytes() == moved
