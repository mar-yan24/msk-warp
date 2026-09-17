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


#: The sealed digest the amendment descends from, transcribed from the stage-1
#: ledger header and the committed v2 manifest.
SEALED_DIGEST = "4c01fdc94117bd190cd44c6b6a5f411a5c7cfc7b7d3e0462a344dda43baf2169"


class _ReboundProtocol:
    """The amendment with a DIFFERENT declared parent and a stable digest.

    The real amendment's ``PARENT_LEDGER`` is part of its provenance and
    therefore of its digest, so it cannot be re-pointed in place without the
    ledger it already wrote refusing to open -- which is itself a guarantee, and
    is asserted separately. This stand-in is the only way to reach the
    re-snapshot's own declared-parent check directly.
    """

    def __init__(self, declared_parent, digest):
        self.PARENT_LEDGER = str(declared_parent)
        self.PARENT_PROTOCOL_DIGEST = P.protocol_digest()
        self._digest = str(digest)

    def protocol_digest(self) -> str:
        return self._digest

    def __getattr__(self, name):
        return getattr(E, name)


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


def _amended(tmp_path, sealed, monkeypatch, *, name="probe_e_ledger.jsonl"):
    """A descendant ledger whose DECLARED parent is this temp sealed ledger.

    ``PARENT_LEDGER`` is part of the amendment's provenance and therefore of its
    digest, so patching it moves ``E.protocol_digest()`` consistently for the
    whole test -- which is exactly the property residual B describes. Pointing it
    at a temp file is what lets the binding be exercised without a test reaching
    into ignored ``logs/``.
    """
    monkeypatch.setattr(E, "PARENT_LEDGER", str(sealed.path))
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


def test_carry_forward_never_writes_to_the_source(tmp_path, monkeypatch):
    """Behavioural, and the guarantee the whole amendment rests on: the sealed
    ledger is read and stays byte-identical."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    before = sealed.path.read_bytes()
    B.carry_forward(sealed.path)
    _amended(tmp_path, sealed, monkeypatch)
    assert sealed.path.read_bytes() == before


# --------------------------------------------------------------------------
# Unit 3 -- the new ledger opens at the carried spend: no cap is reset
# --------------------------------------------------------------------------

def test_the_amended_ledger_opens_at_the_sealed_spend(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    charged = amended.charged()
    assert charged["global_s"] == 1050.0
    assert charged["stage_s"] == {"screen": 1050.0}
    assert charged["settled_rows"] == 0          # no local settlement yet
    assert amended.carried_forward["settled_rows"] == 3


def test_the_global_and_stage_remaining_are_unchanged_by_the_new_ledger(tmp_path, monkeypatch):
    """Behavioural. The budget must not move: the amended ledger reports the
    same global and stage headroom the sealed one does."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    sealed_remaining = sealed.remaining("screen", "g990_e010", 1001)
    amended_remaining = amended.remaining("screen", E.PRIMARY_RECIPE, 1001)
    assert amended_remaining.global_s == sealed_remaining.global_s
    assert amended_remaining.stage_s == sealed_remaining.stage_s
    assert amended_remaining.global_s == P.TOTAL_WALL_CAP_S - 1050.0


def test_only_the_never_run_cell_opens_at_a_fresh_per_seed_allowance(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    fresh = amended.remaining("screen", E.PRIMARY_RECIPE, 1001)
    assert fresh.seed_s == P.stage("screen").per_seed_safety_cap_s
    # The sealed cell's own per-seed spend is still carried, so the amendment
    # cannot reopen a stage-1 cell's allowance either.
    assert amended.carried_forward["run_s"]["screen|g990_e010|1001"] == 400.0


def test_a_new_ledger_cannot_reset_a_cap(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    assert amended.charged()["global_s"] > 0.0
    assert amended.remaining("screen", E.PRIMARY_RECIPE,
                             1001).global_s < P.TOTAL_WALL_CAP_S


def test_a_settlement_on_the_amended_ledger_adds_to_the_carried_total(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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


def test_the_sealed_ledger_is_still_never_overwritten(tmp_path, monkeypatch):
    """Behavioural. The source is a DIFFERENT file here, so the self-carry guard
    cannot fire and the refusal really is the overwrite refusal."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    other = _sealed(tmp_path, name="other_sealed.jsonl", charges=STAGE_1_LIKE)
    monkeypatch.setattr(E, "PARENT_LEDGER", str(other.path))
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


def test_the_amended_ledger_refuses_the_sealed_digest(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    amended.assert_protocol_unchanged(E.protocol_digest())
    with pytest.raises(B.LedgerIntegrityError):
        amended.assert_protocol_unchanged(P.protocol_digest())
    assert amended.protocol_digest != sealed.protocol_digest


def test_the_amended_ledger_refuses_a_sealed_stage_1_arm(tmp_path, monkeypatch):
    """Behavioural. A stage-1 arm cannot be charged to the amendment's ledger."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    with pytest.raises(P.ProtocolError, match="sealed"):
        amended.remaining("screen", "g990_e010", 1001)
    with pytest.raises(P.ProtocolError, match="sealed"):
        amended.reserve(stage="screen", recipe="g990_e010", seed=1001,
                        segment_index=0, start_epoch=0, end_epoch=64,
                        reserved_bound_s=540.0, shutdown_allowance_s=60.0)


def test_the_probe_e_arm_can_never_be_charged_to_a_stage_1_run_key(tmp_path, monkeypatch):
    """Behavioural. The run key carries the recipe, and the names are disjoint."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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

def test_the_amended_header_records_the_source_and_the_descent(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    header = json.loads(amended.path.read_text(encoding="utf-8").splitlines()[0])
    assert header["protocol_digest"] == E.protocol_digest()
    assert header["carried_forward"]["source_protocol_digest"] == P.protocol_digest()
    assert header["carried_forward"]["source_sha256"] == B.sha256_bytes(
        sealed.path.read_bytes())
    assert header["protocol_provenance"]["amendment_id"] == E.AMENDMENT_ID
    assert header["protocol_provenance"]["parent_protocol_digest"] == (
        P.protocol_digest())
    assert header["total_wall_cap_s"] == P.TOTAL_WALL_CAP_S


def test_the_amended_ledger_reopens_and_reverifies(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    reopened = B.BudgetLedger.open(amended.path, protocol=E)
    assert reopened.protocol_digest == E.protocol_digest()
    assert reopened.charged()["global_s"] == 1050.0
    assert reopened.carried_forward == amended.carried_forward


def test_a_tampered_carried_header_is_refused(tmp_path, monkeypatch):
    """Behavioural. The carried block is inside the hash-chained header row, so
    editing it breaks the record digest."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    lines = amended.path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["carried_forward"]["global_s"] = 0.0
    lines[0] = json.dumps(header, sort_keys=True)
    amended.path.write_text("\n".join(lines) + "\n", encoding="utf-8",
                            newline="\n")
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(amended.path, protocol=E)


def test_the_pending_reservation_trap_still_holds_on_a_carried_ledger(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    amended.reserve(stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001,
                    segment_index=0, start_epoch=0, end_epoch=64,
                    reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    with pytest.raises(B.PendingReservationError):
        amended.reserve(stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001,
                        segment_index=0, start_epoch=0, end_epoch=64,
                        reserved_bound_s=540.0, shutdown_allowance_s=60.0)


def test_the_amended_ledger_is_append_only(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    prefix = amended.path.read_bytes()
    reservation = amended.reserve(
        stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001, segment_index=0,
        start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0)
    amended.settle_aborted(reservation, detail="nothing was started")
    after = amended.path.read_bytes()
    assert after.startswith(prefix)
    assert len(after) > len(prefix)


def test_the_amendment_ledger_name_and_parent_are_distinct_files():
    """Transcription only -- the RUNTIME binding of these two literals is tested
    by ``test_the_declared_parent_literals_are_read_at_runtime_not_merely_spelled``,
    which fails if the binding is removed. A spelling assertion on its own was
    vacuous and is no longer the whole of the coverage."""
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


def test_a_parent_that_has_not_moved_reserves_normally(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    amended.assert_parent_unmoved()
    reservation = _reserve(amended)
    assert amended.pending.reservation_id == reservation.reservation_id


def test_a_parent_whose_charge_advanced_refuses_the_next_reserve(tmp_path, monkeypatch):
    """Behavioural, and the whole point of this round. The refusal names both
    totals and both sha256s, so a reader sees exactly what moved."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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


def test_the_drift_refusal_is_fail_closed_and_durable_free(tmp_path, monkeypatch):
    """Behavioural. The refusal precedes the reservation row and the contention
    observation, so it commits nothing and costs nothing."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    probe = _CountingProbe()
    monkeypatch.setattr(E, "PARENT_LEDGER", str(sealed.path))
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


def test_a_parent_holding_a_new_unsettled_reservation_refuses_the_reserve(tmp_path, monkeypatch):
    """Behavioural. An open hold on the parent is drift in progress: its charge
    is not yet known, so the descendant must not commit against it."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    sealed.reserve(stage="screen", recipe="g998_e010", seed=1001,
                   segment_index=0, start_epoch=0, end_epoch=64,
                   reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve(amended)


def test_an_unreadable_parent_refuses_rather_than_falling_back(tmp_path, monkeypatch):
    """Behavioural. Fail closed: a missing parent is never excused by the
    snapshot the descendant happens to be carrying."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    sealed.path.unlink()
    with pytest.raises(B.LedgerIntegrityError, match="parent ledger"):
        _reserve(amended)


def test_a_chain_broken_parent_refuses_rather_than_falling_back(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    lines = sealed.path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[-1])
    row["charged_s"] = 1.0
    lines[-1] = json.dumps(row, sort_keys=True)
    sealed.path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(B.LedgerIntegrityError, match="parent ledger"):
        _reserve(amended)


def test_a_shrunken_parent_refuses_rather_than_falling_back(tmp_path, monkeypatch):
    """Behavioural. Truncating the LAST line of an append-only file leaves a
    self-consistent chain -- the gap disclosed in ``LEDGER_TAIL_NOTE`` -- so a
    fresh cross-process read cannot call it an integrity break. It is caught as
    a MOVED parent instead, on the recomputed total, the byte sha256 AND the row
    count, which is the fail-closed outcome that matters: the snapshot is never
    accepted in its place."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    lines = sealed.path.read_text(encoding="utf-8").splitlines()
    sealed.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8",
                           newline="\n")
    with pytest.raises(B.BudgetError) as raised:
        _reserve(amended)
    message = str(raised.value)
    assert "parent ledger" in message
    assert "1050.0" in message and "750.0" in message
    assert "7 rows against 6" in message


def test_the_check_runs_at_every_reserve_not_only_at_snapshot_time(tmp_path, monkeypatch):
    """Behavioural. The first reserve succeeds; the parent then moves; the
    SECOND reserve refuses. A one-time check at snapshot time would miss this."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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

def test_a_resnapshot_records_the_new_balance_beside_the_old_one(tmp_path, monkeypatch):
    """Behavioural. The audit trail shows what changed and when: one appended
    row carrying BOTH the superseded block and the new one."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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


def test_the_resnapshot_history_keeps_every_superseded_balance(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    _advance_parent(sealed, seconds=500.0)
    amended.resnapshot_parent(detail="first authorised advance")
    _advance_parent(sealed, seconds=300.0, recipe="g998_e000")
    amended.resnapshot_parent(detail="second authorised advance")
    history = amended.carried_history()
    assert [entry["global_s"] for entry in history] == [1050.0, 1550.0, 1850.0]
    assert amended.charged()["global_s"] == 1850.0


def test_a_resnapshot_of_an_unmoved_parent_is_refused(tmp_path, monkeypatch):
    """Behavioural. A re-snapshot is only meaningful when the parent moved; a
    no-op row would be audit noise."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    with pytest.raises(B.LedgerIntegrityError, match="has not moved"):
        amended.resnapshot_parent(detail="nothing happened")


def test_a_resnapshot_is_refused_while_a_reservation_is_pending(tmp_path, monkeypatch):
    """Behavioural. Re-snapshotting under an open hold would move the balance
    the hold was checked against."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    _reserve(amended)
    _advance_parent(sealed, seconds=500.0)
    with pytest.raises(B.PendingReservationError):
        amended.resnapshot_parent(detail="mid-flight")


def test_a_resnapshot_cannot_lower_the_carried_total(tmp_path, monkeypatch):
    """Behavioural. An append-only parent's charge only grows, so a lower total
    means the parent was rewritten -- and adopting it would hand back spend. The
    parent here is truncated on disk, which is the realistic way this happens;
    there is no path= argument to point the re-snapshot elsewhere."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    assert amended.carried_forward["global_s"] == 1050.0
    # Two lines: the reserve/settle pair of the parent's last charge, so what
    # is left is a clean chain with no open hold and a genuinely lower total.
    lines = sealed.path.read_text(encoding="utf-8").splitlines()
    sealed.path.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8",
                           newline="\n")
    assert B.carry_forward(sealed.path).global_s == 750.0
    with pytest.raises(B.LedgerIntegrityError, match="never decrease"):
        amended.resnapshot_parent(detail="the parent lost its last settlement")


def test_a_resnapshot_requires_an_explicit_detail(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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


def test_the_resnapshot_row_keeps_the_hash_chain_and_reopens(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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


def test_a_tampered_resnapshot_row_is_refused(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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


def test_the_parent_ledger_stays_byte_identical_through_all_of_it(tmp_path, monkeypatch):
    """Behavioural. The descendant READS the parent and never writes it -- the
    only writes below are the test's own, through the parent's public API."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
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
# --------------------------------------------------------------------------
# Unit 8 -- fix round 2, DEFECT 1: the carried parent must BE the declared parent
#
# The review carried from a brand-new zero-spend sealed ledger and got
# carried global_s 0.0, remaining 28,800.0 and a successful reserve: the user's
# whole eight-hour authorization re-granted, and then guarded faithfully for the
# rest of that ledger's life, which is the worst shape of bug because everything
# downstream looks correct. PARENT_LEDGER and PARENT_PROTOCOL_DIGEST both
# existed and nothing read either of them.
# --------------------------------------------------------------------------

def _zero_spend_parent(tmp_path, *, name="empty_sealed.jsonl"):
    """The review's decoy: a real sealed-protocol ledger with nothing settled."""
    return B.BudgetLedger.create(tmp_path / name,
                                 protocol_digest=P.protocol_digest(),
                                 clock=FakeClock())


def test_the_reviews_zero_spend_decoy_is_refused(tmp_path):
    """Behavioural, the review's exact reproduction. Carrying from a ledger that
    is NOT the amendment's declared parent must refuse, not re-grant the cap."""
    decoy = _zero_spend_parent(tmp_path)
    carried = B.carry_forward(decoy.path)
    assert carried.global_s == 0.0                  # the decoy really is empty
    with pytest.raises(B.LedgerIntegrityError) as raised:
        B.BudgetLedger.create(tmp_path / "probe_e_ledger.jsonl",
                              protocol_digest=E.protocol_digest(), protocol=E,
                              carried=carried, clock=FakeClock())
    message = str(raised.value)
    assert E.PARENT_LEDGER in message               # the expected parent
    assert str(decoy.path) in message               # the supplied one
    assert not (tmp_path / "probe_e_ledger.jsonl").exists()


def test_a_parent_at_the_wrong_path_is_refused_even_with_the_right_digest(tmp_path):
    """Behavioural. The digest of a sealed-protocol decoy matches by
    construction, so path identity has to be checked on its own."""
    decoy = _sealed(tmp_path, name="wrong_place.jsonl", charges=STAGE_1_LIKE)
    carried = B.carry_forward(decoy.path)
    assert carried.source_protocol_digest == E.PARENT_PROTOCOL_DIGEST
    with pytest.raises(B.LedgerIntegrityError, match="declared parent"):
        B.BudgetLedger.create(tmp_path / "probe_e_ledger.jsonl",
                              protocol_digest=E.protocol_digest(), protocol=E,
                              carried=carried, clock=FakeClock())


def test_a_parent_with_the_wrong_protocol_digest_is_refused(tmp_path, monkeypatch):
    """Behavioural. With the path pointed at the right place, the parent's own
    header digest must still be the one the amendment descends from."""
    foreign = B.BudgetLedger.create(tmp_path / "foreign.jsonl",
                                    protocol_digest="f" * 64, clock=FakeClock())
    monkeypatch.setattr(E, "PARENT_LEDGER", str(foreign.path))
    carried = B.carry_forward(foreign.path)
    with pytest.raises(B.LedgerIntegrityError) as raised:
        B.BudgetLedger.create(tmp_path / "probe_e_ledger.jsonl",
                              protocol_digest=E.protocol_digest(), protocol=E,
                              carried=carried, clock=FakeClock())
    message = str(raised.value)
    assert E.PARENT_PROTOCOL_DIGEST in message
    assert "f" * 64 in message


def test_the_declared_parent_literals_are_read_at_runtime_not_merely_spelled(
        tmp_path, monkeypatch):
    """Behavioural, and this replaces the vacuous spelling assertion: it FAILS if
    the binding is removed, because then the decoy below would be accepted."""
    assert E.PARENT_LEDGER == "logs/myoleg26_ppo_v2/budget_ledger.jsonl"
    assert E.PARENT_PROTOCOL_DIGEST == SEALED_DIGEST
    decoy = _sealed(tmp_path, name="decoy.jsonl", charges=STAGE_1_LIKE)
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.create(tmp_path / "a.jsonl",
                              protocol_digest=E.protocol_digest(), protocol=E,
                              carried=B.carry_forward(decoy.path),
                              clock=FakeClock())
    # ...and the SAME call succeeds once that literal names the decoy, which is
    # the proof that the literal, not the path's spelling, is what decided.
    monkeypatch.setattr(E, "PARENT_LEDGER", str(decoy.path))
    B.BudgetLedger.create(tmp_path / "b.jsonl",
                          protocol_digest=E.protocol_digest(), protocol=E,
                          carried=B.carry_forward(decoy.path),
                          clock=FakeClock())


def test_a_protocol_that_declares_no_parent_is_unaffected(tmp_path):
    """Behavioural. The sealed campaign protocol declares no parent, so the
    binding cannot constrain it."""
    assert not hasattr(P, "PARENT_LEDGER")
    ledger = _sealed(tmp_path, charges=STAGE_1_LIKE)
    assert ledger.carried_forward is None


def test_a_resnapshot_cannot_escape_the_declared_parent(tmp_path):
    """Behavioural. The re-snapshot path re-checks the same binding, so it is not
    a second way in. Reached through a stand-in protocol, because the real
    amendment's declared parent is part of its digest (see the test below)."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    bound = _ReboundProtocol(sealed.path, "d" * 64)
    amended = B.BudgetLedger.create(
        tmp_path / "probe_e_ledger.jsonl", protocol_digest=bound.protocol_digest(),
        protocol=bound, carried=B.carry_forward(sealed.path), clock=FakeClock())
    _advance_parent(sealed, seconds=500.0)

    elsewhere = _ReboundProtocol(tmp_path / "somewhere_else.jsonl", "d" * 64)
    reopened = B.BudgetLedger.open(amended.path, protocol=elsewhere)
    with pytest.raises(B.LedgerIntegrityError, match="declared parent"):
        reopened.resnapshot_parent(detail="pointing somewhere else")


def test_repointing_the_real_declared_parent_invalidates_its_own_ledger(
        tmp_path, monkeypatch):
    """Behavioural, and stronger than the test above: because PARENT_LEDGER is
    part of the amendment's provenance and so of its digest, re-pointing it moves
    the digest and the ledger it already wrote refuses to open at all. The
    declared parent cannot be quietly changed after the fact."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    written_digest = amended.protocol_digest
    monkeypatch.setattr(E, "PARENT_LEDGER",
                        "logs/myoleg26_ppo_v2/budget_ledger.jsonl")
    assert E.protocol_digest() != written_digest
    with pytest.raises(B.LedgerIntegrityError, match="never interchangeable"):
        B.BudgetLedger.open(amended.path, protocol=E)



def test_resnapshot_takes_no_path_argument(tmp_path, monkeypatch):
    """Behavioural. Re-pointing the parent by argument would be exactly the
    override this unit refuses to have."""
    import inspect as _inspect
    signature = _inspect.signature(B.BudgetLedger.resnapshot_parent)
    assert set(signature.parameters) == {"self", "detail"}


# --------------------------------------------------------------------------
# Unit 9 -- fix round 2, DEFECT 2: the loader binds the header to the protocol
#
# The review opened a COPY of the real sealed ledger under the amendment
# protocol and appended a probe-E reservation row to it. The isolation rested
# entirely on run_launch happening to call assert_protocol_unchanged before
# reserve. It is now enforced by the loader, so it holds no matter who calls it
# and in what order.
# --------------------------------------------------------------------------

def test_opening_a_sealed_ledger_under_the_amendment_protocol_is_refused(tmp_path):
    """Behavioural, the review's exact reproduction."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    before = sealed.path.read_bytes()
    with pytest.raises(B.LedgerIntegrityError) as raised:
        B.BudgetLedger.open(sealed.path, protocol=E)
    message = str(raised.value)
    assert P.protocol_digest() in message
    assert E.protocol_digest() in message
    assert sealed.path.read_bytes() == before


def test_the_refused_open_cannot_append_a_probe_e_row(tmp_path):
    """Behavioural. The whole point: the sealed chain never gains a probe-E row,
    whatever the caller's statement order."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    before = sealed.path.read_bytes()
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(sealed.path, protocol=E).reserve(
            stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001,
            segment_index=0, start_epoch=0, end_epoch=64,
            reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    assert sealed.path.read_bytes() == before
    assert all(row["recipe"] != E.PRIMARY_RECIPE
               for row in B.BudgetLedger.open(sealed.path).rows
               if "recipe" in row)


def test_opening_an_amended_ledger_under_the_sealed_protocol_is_refused(
        tmp_path, monkeypatch):
    """Behavioural, the mirror direction: a sealed arm must not be appendable to
    the descendant's ledger either."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    before = amended.path.read_bytes()
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(amended.path, protocol=P)
    with pytest.raises(B.LedgerIntegrityError):
        B.BudgetLedger.open(amended.path)          # the sealed default
    assert amended.path.read_bytes() == before


def test_each_ledger_still_opens_under_its_own_protocol(tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    assert B.BudgetLedger.open(sealed.path, protocol=P).protocol_digest == (
        P.protocol_digest())
    assert B.BudgetLedger.open(sealed.path).protocol_digest == P.protocol_digest()
    assert B.BudgetLedger.open(amended.path, protocol=E).protocol_digest == (
        E.protocol_digest())


def test_inspection_reads_a_foreign_ledger_and_can_never_write_to_it(tmp_path):
    """Behavioural. Reading a foreign ledger to total it is legitimate -- that is
    what carry-forward is -- so it has its own mode, and that mode refuses every
    mutating call rather than being an escape hatch."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    before = sealed.path.read_bytes()
    reading = B.BudgetLedger.open(sealed.path, protocol=E, inspect=True)
    assert reading.charged()["global_s"] == 1050.0
    assert reading.protocol_digest == P.protocol_digest()
    with pytest.raises(B.LedgerIntegrityError, match="inspection"):
        reading.reserve(stage="screen", recipe=E.PRIMARY_RECIPE, seed=1001,
                        segment_index=0, start_epoch=0, end_epoch=64,
                        reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    with pytest.raises(B.LedgerIntegrityError, match="inspection"):
        reading.resnapshot_parent(detail="not from here")
    assert sealed.path.read_bytes() == before


def test_carry_forward_reads_its_foreign_parent_through_inspection(tmp_path):
    """Behavioural. carry_forward must keep working across protocols -- the
    parent's header digest is by definition not the descendant's."""
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    before = sealed.path.read_bytes()
    carried = B.carry_forward(sealed.path)
    assert carried.global_s == 1050.0
    assert sealed.path.read_bytes() == before


def test_the_parent_recheck_reads_its_foreign_parent_through_inspection(
        tmp_path, monkeypatch):
    sealed = _sealed(tmp_path, charges=STAGE_1_LIKE)
    amended = _amended(tmp_path, sealed, monkeypatch)
    before = sealed.path.read_bytes()
    amended.assert_parent_unmoved()
    _advance_parent(sealed, seconds=500.0)
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve(amended)
    assert sealed.path.read_bytes() != before      # the TEST advanced it
    assert B.BudgetLedger.open(sealed.path).charged()["global_s"] == 1550.0


def test_the_note_records_the_settle_row_design_intent_for_a_later_descendant():
    """Recorded design intent, not code: if a later descendant ever authorises
    more than one cell, the answer is a recorded field on the settle row, never a
    refusal that would wedge the ledger with a pending reservation."""
    assert "parent_moved_during_segment" in B.PARENT_RECHECK_NOTE
    assert "wedge" in B.PARENT_RECHECK_NOTE
