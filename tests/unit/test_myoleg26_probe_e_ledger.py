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
