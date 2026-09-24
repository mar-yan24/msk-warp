"""Unit tests for Direction F's ledger, the third level of the v2 ledger chain.

Fake clocks and injected walls only. There is no simulator, CUDA, Warp or torch,
no subprocess, no sleep, no GPU query and no real segment. Every wall time below
is an injected number, so this file spends none of the 28,800 s training budget
and none of the 1,800 s diagnostic budget.

What is under test
------------------
The chain is sealed, then probe E, then Direction F. F's ledger opens at E's
settled spend, which already includes the sealed spend E carried. The budget
module adds four guarantees for F, all of them CPU-only and structural:

(a) The parent re-check is **transitive**. A moved sealed grandparent refuses
    F's reserve even though E's bytes never changed. A loop in the chain is a
    ``LedgerIntegrityError``, never a ``RecursionError``. ``create`` refuses a
    stale E.
(b) ``reserve`` and ``create`` call the protocol's launch lock, and a refusal is
    an ``UnauthorisedLaunchError(BudgetError)`` that writes nothing.
(c) With ``BIND_LEDGER_PATH``, the ledger may be created only at
    ``ROOT/LEDGER``.
(e) With ``BIND_LEDGER_PATH``, ``create`` refuses if ``RUN_ROOT`` already holds
    any ``launch.json`` or segment directory. This is the deleted-ledger re-grant
    guard.

The design-C section 3.7 ledger list maps onto ``test_NN_...`` (1-31; the
optional 32 is deferred along with change (d)). The run-root guard tests are
``test_eNN_...``, and supplementary checks are ``test_sNN_...``.

Isolation
---------
Every test builds a **temporary** three-level chain. ``E.PARENT_LEDGER`` is
patched (it is inside E's digest) and ``F.PARENT_PROTOCOL_DIGEST`` is set to the
patched E digest. ``F.PARENT_LEDGER``, ``F.LEDGER``, ``F.RUN_ROOT``,
``F.AUTHORISATION_RECORD``, ``F.PREREGISTRATION_PATH``,
``F.PREREGISTRATION_SHA256`` and ``F.AUTHORISED`` all point at temporary values.
An autouse guard refuses any ``open`` or directory listing of a path under the
real ``logs/`` or ``docs/``.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import inspect
import io
import json
import os
from pathlib import Path

import pytest

from msk_warp.analysis import ppo_v2_budget as B
from msk_warp.analysis import ppo_v2_protocol as P
from msk_warp.analysis import ppo_v2_protocol_e as E
from msk_warp.analysis import ppo_v2_protocol_f as F

ROOT = Path(__file__).resolve().parents[2]
BUDGET_SOURCE = Path(B.__file__)

#: Transcribed ancestor digests (see the Direction F protocol tests).
SEALED_DIGEST = "4c01fdc94117bd190cd44c6b6a5f411a5c7cfc7b7d3e0462a344dda43baf2169"
E_DIGEST = "7c7892554cd79fb48832b1c431652f71776ef7e05b5359d4ccfc60f8c39ab68e"

F_RECIPE = "g990_e010_f512"
F_SEEDS = (4001, 4002)
E_ARM = "g990_e010_c040"
E_ALT_ARM = "g990_e010_k001"
SEALED_ARMS = ("g990_e010", "g998_e010", "g990_e000", "g998_e000")


# --------------------------------------------------------------------------
# Isolation guard: no test may read or list the real logs/ or docs/
# --------------------------------------------------------------------------

_GUARDED_ROOTS = tuple(os.path.normcase(str((ROOT / name).resolve()))
                       for name in ("logs", "docs"))


def _under_guarded_root(target) -> bool:
    if isinstance(target, int):
        return False
    try:
        resolved = os.path.normcase(str(Path(os.fsdecode(target)).resolve()))
    except (TypeError, ValueError, OSError):
        return False
    return any(resolved == root or resolved.startswith(root + os.sep)
               for root in _GUARDED_ROOTS)


@pytest.fixture(autouse=True)
def _no_real_logs_or_docs(monkeypatch):
    real_open = io.open
    real_scandir = os.scandir
    real_listdir = os.listdir

    def guarded_open(file, *args, **kwargs):
        if _under_guarded_root(file):
            raise AssertionError(
                f"a Direction F ledger test tried to open the real {file!r}")
        return real_open(file, *args, **kwargs)

    def guarded_scandir(path=".", *args, **kwargs):
        if _under_guarded_root(path):
            raise AssertionError(
                f"a Direction F ledger test tried to list the real {path!r}")
        return real_scandir(path, *args, **kwargs)

    def guarded_listdir(path=".", *args, **kwargs):
        if _under_guarded_root(path):
            raise AssertionError(
                f"a Direction F ledger test tried to list the real {path!r}")
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "scandir", guarded_scandir)
    monkeypatch.setattr(os, "listdir", guarded_listdir)
    yield


def test_s00_the_isolation_guard_has_teeth():
    """Anti-vacuity for the guard. Real ledgers, the real run roots and docs/
    are refused before any byte is read, and an ordinary tracked file is not."""
    with pytest.raises(AssertionError, match="real"):
        open(ROOT / "logs" / "myoleg26_ppo_v2" / "budget_ledger.jsonl", "rb")
    with pytest.raises(AssertionError, match="real"):
        (ROOT / "logs" / "myoleg26_ppo_v2_probe_e"
         / "budget_ledger.jsonl").read_bytes()
    with pytest.raises(AssertionError, match="real"):
        list(os.walk(ROOT / "logs" / "myoleg26_ppo_v2_probe_e"))
    with pytest.raises(AssertionError, match="real"):
        os.listdir(ROOT / "docs")
    with pytest.raises(AssertionError, match="real"):
        B.BudgetLedger.open(ROOT / "logs" / "myoleg26_ppo_v2"
                            / "budget_ledger.jsonl", inspect=True)
    assert BUDGET_SOURCE.read_bytes()


# --------------------------------------------------------------------------
# Fakes and helpers
# --------------------------------------------------------------------------

class FakeClock:
    """A monotonically advancing injected clock. Never sleeps, never reads time."""

    def __init__(self, start=1_700_000_000.0, step=1.0):
        self.now = float(start)
        self.step = float(step)

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


class _CountingProbe:
    """Counts observations, so a refusal can be shown to precede the probe."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return B.unobserved_contention()


def _counters():
    return B.SegmentCounters(
        attempted_control_transitions=8192, completed_control_transitions=8192,
        replayed_control_transitions=0, attempted_physics_steps=32768,
        completed_physics_steps=32768, replayed_physics_steps=0,
        partial_update_cost_s=0.0)


def _charge(ledger, *, stage, recipe, seed, actual, segment_index=0,
            start_epoch=0, end_epoch=64):
    """Reserve one fake segment and settle it at an injected wall."""
    reservation = ledger.reserve(
        stage=stage, recipe=recipe, seed=seed, segment_index=segment_index,
        start_epoch=start_epoch, end_epoch=end_epoch, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0)
    return ledger.settle(reservation, actual_wall_s=actual, returncode=0,
                         counters=_counters())


def _reserve_f(ledger, *, seed=4001, segment_index=0, start_epoch=0,
               end_epoch=64, recipe=F_RECIPE, stage="refine", create=None):
    return ledger.reserve(
        stage=stage, recipe=recipe, seed=seed, segment_index=segment_index,
        start_epoch=start_epoch, end_epoch=end_epoch, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0, create=create)


def _charge_f(ledger, *, seed, actual, segment_index=0):
    start = 64 * segment_index
    reservation = _reserve_f(ledger, seed=seed, segment_index=segment_index,
                             start_epoch=start, end_epoch=start + 64)
    return ledger.settle(reservation, actual_wall_s=actual, returncode=0,
                         counters=_counters())


def _header(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])


def _resolved(path) -> str:
    return str(Path(path).resolve())


def _is_unauthorised(error) -> bool:
    """True when ``error`` is the budget module's UnauthorisedLaunchError.

    Looked up by name at call time, so a test run before the class exists fails
    on its own assertion rather than as a collection error.
    """
    cls = getattr(B, "UnauthorisedLaunchError", None)
    return cls is not None and isinstance(error, cls)


class _WithoutLock:
    """Direction F with its launch lock hidden, and nothing else changed.

    Stands in for an F ledger that already exists while the build is locked --
    for example one written by a build before the lock existed. The ledger is
    then opened and used under the real, locked F.
    """

    def __getattr__(self, name):
        if name == "assert_authorised":
            raise AttributeError(name)
        return getattr(F, name)


class _WithoutRunRoot:
    """Direction F declaring BIND_LEDGER_PATH but no RUN_ROOT."""

    def __getattr__(self, name):
        if name == "RUN_ROOT":
            raise AttributeError(name)
        return getattr(F, name)


class Chain:
    """A temporary sealed -> E -> F chain with a temporary prereg and record."""

    PREREG_BYTES = (b"# Direction F temporary preregistration\n\n"
                    b"line two of the frozen text\nline three\n")
    #: Non-round walls, so every carried total below is an exact float check.
    SEALED_CHARGES = (("screen", "g990_e010", 1001, 400.125),
                      ("screen", "g990_e010", 1002, 350.0625),
                      ("screen", "g998_e000", 1001, 300.3))
    E_CHARGE_S = 320.7

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        for name in ("sealed", "probe_e", "prereg"):
            (tmp_path / name).mkdir()
        self.sealed_path = tmp_path / "sealed" / "budget_ledger.jsonl"
        self.e_path = tmp_path / "probe_e" / "budget_ledger.jsonl"
        self.run_root = tmp_path / "f_run_root"
        self.run_root.mkdir()
        self.f_path = self.run_root / "budget_ledger.jsonl"
        self.record_path = self.run_root / "user_authorisation.json"
        self.prereg_path = tmp_path / "prereg" / "direction-f-preregistration.md"
        self.prereg_path.write_bytes(self.PREREG_BYTES)
        self.prereg_sha256 = hashlib.sha256(self.PREREG_BYTES).hexdigest()
        self.sealed = None
        self.e = None
        self.f = None

        monkeypatch.setattr(E, "PARENT_LEDGER", str(self.sealed_path))
        monkeypatch.setattr(F, "PARENT_PROTOCOL_DIGEST", E.protocol_digest())
        monkeypatch.setattr(F, "PARENT_LEDGER", str(self.e_path))
        monkeypatch.setattr(F, "LEDGER", str(self.f_path))
        monkeypatch.setattr(F, "RUN_ROOT", str(self.run_root))
        monkeypatch.setattr(F, "AUTHORISATION_RECORD", str(self.record_path))
        monkeypatch.setattr(F, "PREREGISTRATION_PATH", str(self.prereg_path))
        monkeypatch.setattr(F, "PREREGISTRATION_SHA256", self.prereg_sha256)
        monkeypatch.setattr(F, "AUTHORISED", False)

    # -- the three levels ---------------------------------------------------

    def build_sealed(self):
        self.sealed = B.BudgetLedger.create(
            self.sealed_path, protocol_digest=P.protocol_digest(),
            clock=FakeClock())
        for stage, recipe, seed, actual in self.SEALED_CHARGES:
            _charge(self.sealed, stage=stage, recipe=recipe, seed=seed,
                    actual=actual)
        return self.sealed

    def build_e(self):
        self.e = B.BudgetLedger.create(
            self.e_path, protocol_digest=E.protocol_digest(), protocol=E,
            carried=B.carry_forward(self.sealed_path),
            provenance=E.ledger_provenance(), clock=FakeClock())
        _charge(self.e, stage="screen", recipe=E_ARM, seed=1001,
                actual=self.E_CHARGE_S)
        return self.e

    def create_f(self, path=None, *, carried=None, protocol=F, probe=None):
        return B.BudgetLedger.create(
            self.f_path if path is None else path,
            protocol_digest=F.protocol_digest(), protocol=protocol,
            carried=B.carry_forward(self.e_path) if carried is None else carried,
            provenance=F.ledger_provenance(), clock=FakeClock(),
            contention_probe=probe)

    def build(self, *, probe=None):
        self.build_sealed()
        self.build_e()
        self.authorise()
        self.f = self.create_f(probe=probe)
        return self.f

    def advance_sealed(self, seconds=500.0):
        """Settle one more fake stage-1 segment after the snapshots were taken."""
        _charge(self.sealed, stage="screen", recipe="g998_e010", seed=1001,
                actual=seconds)

    def advance_e(self, seconds=250.5):
        _charge(self.e, stage="screen", recipe=E_ALT_ARM, seed=1001,
                actual=seconds)

    # -- the launch lock ----------------------------------------------------

    def valid_record(self) -> dict:
        """A record bound to the build as it stands at the time of the call."""
        return {
            "schema_version": "myoleg26-direction-f-authorisation-v1",
            "authorised": True,
            "protocol": "direction-f",
            "amendment_id": "direction-f-horizon-v1",
            "protocol_digest": F.protocol_digest(),
            "preregistration_sha256": self.prereg_sha256,
            "funding_cap": "refine",
            "budget_s": F.FUNDING_CAP_S,
            "seeds": [4001, 4002],
            "authorised_by": "Temporary Test Signatory",
            "decided_utc": "2026-09-24T12:00:00Z",
            "statement": "I authorise Direction F on the refine funding line.",
        }

    def write_record(self, record=None) -> None:
        record = self.valid_record() if record is None else record
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_path.write_bytes(json.dumps(record).encode("utf-8"))

    def authorise(self) -> None:
        """Open both layers: the source constant, then a record at the live digest."""
        self.monkeypatch.setattr(F, "AUTHORISED", True)
        self.write_record()

    def relocate(self, *, run_root, ledger_dir, record_dir) -> None:
        """Point RUN_ROOT, LEDGER and the record at separate temporary places."""
        self.run_root = Path(run_root)
        self.f_path = Path(ledger_dir) / "budget_ledger.jsonl"
        self.record_path = Path(record_dir) / "user_authorisation.json"
        self.monkeypatch.setattr(F, "RUN_ROOT", str(self.run_root))
        self.monkeypatch.setattr(F, "LEDGER", str(self.f_path))
        self.monkeypatch.setattr(F, "AUTHORISATION_RECORD", str(self.record_path))


@pytest.fixture
def chain(tmp_path, monkeypatch) -> Chain:
    return Chain(tmp_path, monkeypatch)


SEALED_TOTAL = 0.0 + 400.125 + 350.0625 + 300.3
E_TOTAL = SEALED_TOTAL + Chain.E_CHARGE_S


# --------------------------------------------------------------------------
# 1-8: the opening balance, the header and the refusals at create
# --------------------------------------------------------------------------

def test_01_f_opens_at_e_settled_spend_including_the_sealed_carry(chain):
    f = chain.build()
    assert chain.sealed.charged()["global_s"] == SEALED_TOTAL
    assert chain.e.charged()["global_s"] == E_TOTAL
    charged = f.charged()
    assert charged["global_s"] == E_TOTAL
    assert charged["stage_s"] == {"screen": E_TOTAL}
    assert charged["stage_s"].get("refine", 0.0) == 0.0
    assert charged["run_s"]["screen|g990_e010|1001"] == 400.125
    assert charged["run_s"]["screen|g998_e000|1001"] == 300.3
    assert charged["run_s"][f"screen|{E_ARM}|1001"] == Chain.E_CHARGE_S
    # Own settled rows and carried settled rows are reported apart: E settled
    # one segment and carried the sealed three.
    assert charged["settled_rows"] == 0
    assert f.carried_totals()["settled_rows"] == 1 + 3
    assert f.carried_forward["source_rows"] == len(chain.e.rows)
    assert f.carried_forward["source_path"] == _resolved(chain.e_path)
    _charge_f(f, seed=4001, actual=318.3)
    assert f.charged()["settled_rows"] + f.carried_totals()["settled_rows"] == 5
    assert f.charged()["global_s"] == E_TOTAL + 318.3


def test_02_remaining_is_global_minus_carry_funding_cap_and_per_seed(chain):
    f = chain.build()
    carried = f.carried_forward["global_s"]
    assert carried == E_TOTAL
    for seed in F_SEEDS:
        remaining = f.remaining("refine", F_RECIPE, seed)
        assert remaining.global_s == 28_800 - carried
        assert remaining.global_s == P.TOTAL_WALL_CAP_S - carried
        assert remaining.stage_s == F.FUNDING_CAP_S == 7200
        assert remaining.seed_s == F.PER_SEED_SAFETY_CAP_S == 4800


def test_03_the_header_records_the_provenance_and_the_prereg_hash(chain):
    chain.build()
    header = _header(chain.f_path)
    assert header["protocol_digest"] == F.protocol_digest()
    provenance = header["protocol_provenance"]
    assert provenance == json.loads(json.dumps(F.ledger_provenance()))
    assert provenance["preregistration"] == {
        "path": str(chain.prereg_path), "sha256": chain.prereg_sha256}
    assert provenance["parent_protocol_digest"] == E.protocol_digest()
    assert provenance["grandparent_protocol_digest"] == SEALED_DIGEST
    assert provenance["parent_ledger"] == str(chain.e_path)
    assert provenance["funding"]["cap"] == "refine"
    carried = header["carried_forward"]
    assert carried["source_protocol_digest"] == E.protocol_digest()
    assert carried["source_sha256"] == B.sha256_bytes(chain.e_path.read_bytes())


def test_04_f_is_refused_without_a_carried_balance(chain):
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    with pytest.raises(B.LedgerIntegrityError, match="carried forward"):
        B.BudgetLedger.create(chain.f_path, protocol_digest=F.protocol_digest(),
                              protocol=F, provenance=F.ledger_provenance(),
                              clock=FakeClock())
    assert not chain.f_path.exists()


def test_05a_carrying_from_the_sealed_ledger_directly_fails_the_path_clause(chain):
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    carried = B.carry_forward(chain.sealed_path)
    with pytest.raises(B.LedgerIntegrityError,
                       match="declares its parent ledger") as raised:
        chain.create_f(carried=carried)
    message = str(raised.value)
    assert _resolved(chain.e_path) in message       # the declared parent
    assert _resolved(chain.sealed_path) in message  # the supplied one
    assert not chain.f_path.exists()


def test_05b_carrying_from_the_sealed_ledger_directly_fails_the_digest_clause(chain):
    """With the path clause satisfied on purpose, the digest clause alone must
    still refuse: the sealed header digest is not E's."""
    chain.build_sealed()
    chain.build_e()
    chain.monkeypatch.setattr(F, "PARENT_LEDGER", str(chain.sealed_path))
    chain.authorise()                    # the record binds the moved digest
    carried = B.carry_forward(chain.sealed_path)
    with pytest.raises(B.LedgerIntegrityError,
                       match="descends from parent protocol digest") as raised:
        chain.create_f(carried=carried)
    message = str(raised.value)
    assert F.PARENT_PROTOCOL_DIGEST == E.protocol_digest()
    assert E.protocol_digest() in message
    assert P.protocol_digest() in message
    assert not chain.f_path.exists()


def test_06_a_decoy_ledger_with_e_digest_at_another_path_is_refused(chain):
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    decoy_path = chain.tmp / "decoy_e" / "budget_ledger.jsonl"
    decoy_path.parent.mkdir()
    B.BudgetLedger.create(decoy_path, protocol_digest=E.protocol_digest(),
                          protocol=E, carried=B.carry_forward(chain.sealed_path),
                          provenance=E.ledger_provenance(), clock=FakeClock())
    carried = B.carry_forward(decoy_path)
    assert carried.source_protocol_digest == F.PARENT_PROTOCOL_DIGEST
    assert carried.global_s == SEALED_TOTAL        # a cheaper opening balance
    with pytest.raises(B.LedgerIntegrityError, match="declares its parent ledger"):
        chain.create_f(carried=carried)
    assert not chain.f_path.exists()


def test_07_a_second_f_ledger_at_another_path_is_refused(chain):
    """Path binding (c). The same call at the declared path then succeeds, which
    proves the binding, not something else, decided."""
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    elsewhere = chain.tmp / "second_f" / "budget_ledger.jsonl"
    elsewhere.parent.mkdir()
    with pytest.raises(B.LedgerIntegrityError) as raised:
        chain.create_f(elsewhere)
    message = str(raised.value)
    assert _resolved(elsewhere) in message
    assert _resolved(chain.f_path) in message
    assert not elsewhere.exists()
    chain.create_f()
    # With the one F ledger in place, a second one elsewhere is still refused.
    with pytest.raises(B.LedgerIntegrityError):
        chain.create_f(elsewhere)
    assert not elsewhere.exists()


def test_08_an_existing_f_ledger_is_never_overwritten(chain):
    chain.build()
    before = chain.f_path.read_bytes()
    with pytest.raises(B.LedgerIntegrityError,
                       match="never overwritten, replaced or started fresh"):
        chain.create_f()
    assert chain.f_path.read_bytes() == before


# --------------------------------------------------------------------------
# 9-16: the transitive parent re-check
# --------------------------------------------------------------------------

def test_09_an_unmoved_chain_reserves_normally(chain):
    f = chain.build()
    f.assert_parent_unmoved()
    reservation = _reserve_f(f)
    assert f.pending.reservation_id == reservation.reservation_id
    f.settle(reservation, actual_wall_s=318.3, returncode=0, counters=_counters())
    assert f.charged()["stage_s"]["refine"] == 318.3


def test_10_a_moved_e_refuses_reserve_with_zero_rows(chain):
    probe = _CountingProbe()
    f = chain.build(probe=probe)
    before = chain.f_path.read_bytes()
    chain.advance_e()
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve_f(f)
    assert chain.f_path.read_bytes() == before
    assert probe.calls == 0
    assert B.BudgetLedger.open(chain.f_path, protocol=F).pending is None


def test_11_a_moved_sealed_grandparent_refuses_f_reserve_with_zero_rows(chain):
    """RED before change (a). E's bytes never change here, so only a transitive
    re-check can see that the sealed spend under E's snapshot has moved."""
    probe = _CountingProbe()
    f = chain.build(probe=probe)
    f_before = chain.f_path.read_bytes()
    e_before = chain.e_path.read_bytes()
    chain.advance_sealed(500.0)
    assert chain.e_path.read_bytes() == e_before     # the direct parent is intact
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve_f(f)
    assert chain.f_path.read_bytes() == f_before
    assert probe.calls == 0
    # A fresh process sees the same thing: the refusal is derived from files.
    reopened = B.BudgetLedger.open(chain.f_path, protocol=F)
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve_f(reopened)
    with pytest.raises(B.ParentLedgerMovedError):
        reopened.assert_parent_unmoved()
    assert chain.f_path.read_bytes() == f_before
    assert reopened.pending is None


def test_12_the_refusal_names_the_chain_and_the_remedy(chain):
    f = chain.build()
    chain.advance_sealed(500.0)
    with pytest.raises(B.ParentLedgerMovedError) as raised:
        _reserve_f(f)
    message = str(raised.value)
    sealed_s, e_s, f_s = (_resolved(path) for path in
                          (chain.sealed_path, chain.e_path, chain.f_path))
    for name in (sealed_s, e_s, f_s):
        assert name in message
    assert f"re-snapshot {e_s}, then {f_s}" in message
    # The moved ledger's own numbers travel with it.
    assert repr(SEALED_TOTAL) in message
    assert repr(SEALED_TOTAL + 500.0) in message


def test_13_resnapshotting_e_then_f_adopts_the_new_sealed_charge(chain):
    f = chain.build()
    chain.advance_sealed(500.0)
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve_f(f)
    chain.e.resnapshot_parent(
        detail="the sealed ledger advanced with the user's authority")
    with pytest.raises(B.ParentLedgerMovedError):
        _reserve_f(f)                    # now it is the direct parent that moved
    f.resnapshot_parent(detail="probe E re-snapshotted the sealed advance")
    expected = SEALED_TOTAL + 500.0 + Chain.E_CHARGE_S
    assert chain.e.charged()["global_s"] == expected
    assert f.carried_forward["global_s"] == expected
    assert f.charged()["global_s"] == expected
    assert f.remaining("refine", F_RECIPE, 4001).global_s == (
        P.TOTAL_WALL_CAP_S - expected)
    f.assert_parent_unmoved()
    reservation = _reserve_f(f)
    assert f.pending.reservation_id == reservation.reservation_id


def test_14_create_refuses_a_stale_e_and_writes_no_file(chain):
    """E's own bytes match its carry here; it is E's parent that moved. The
    carry is taken after the move, as an operator would take it."""
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    chain.advance_sealed(500.0)
    carried = B.carry_forward(chain.e_path)
    with pytest.raises(B.ParentLedgerMovedError) as raised:
        chain.create_f(carried=carried)
    message = str(raised.value)
    assert _resolved(chain.e_path) in message
    assert _resolved(chain.sealed_path) in message
    assert "re-snapshot" in message
    assert not chain.f_path.exists()
    # The remedy works: re-snapshot E, then F can be created.
    chain.e.resnapshot_parent(detail="the sealed ledger advanced")
    f = chain.create_f()
    assert f.carried_forward["global_s"] == SEALED_TOTAL + 500.0 + Chain.E_CHARGE_S


def test_15_probe_e_behaviour_is_unchanged(tmp_path, monkeypatch):
    """E declares no ROOT, no BIND_LEDGER_PATH and no lock. It still creates at
    any temporary path, keeps its header shape, reserves without any record, and
    refuses a moved sealed parent with the same direct message."""
    assert not hasattr(E, "BIND_LEDGER_PATH")
    assert not hasattr(E, "assert_authorised")
    sealed_path = tmp_path / "anywhere" / "sealed.jsonl"
    sealed_path.parent.mkdir()
    sealed = B.BudgetLedger.create(sealed_path, protocol_digest=P.protocol_digest(),
                                   clock=FakeClock())
    for stage, recipe, seed, actual in Chain.SEALED_CHARGES:
        _charge(sealed, stage=stage, recipe=recipe, seed=seed, actual=actual)
    monkeypatch.setattr(E, "PARENT_LEDGER", str(sealed_path))
    e_path = tmp_path / "not_e_ledger_path" / "whatever.jsonl"
    e_path.parent.mkdir()
    e = B.BudgetLedger.create(e_path, protocol_digest=E.protocol_digest(),
                              protocol=E, carried=B.carry_forward(sealed_path),
                              provenance=E.ledger_provenance(), clock=FakeClock())
    assert set(_header(e_path)) == {
        "index", "kind", "prev_sha256", "posix_time", "utc", "schema_version",
        "protocol_digest", "total_wall_cap_s", "reserve_wall_cap_s",
        "segment_max_epochs", "segment_work_deadline_s", "segment_call_bound_s",
        "carried_forward", "protocol_provenance", "record_sha256"}
    _charge(e, stage="screen", recipe=E_ARM, seed=1001, actual=Chain.E_CHARGE_S)
    assert e.charged()["global_s"] == E_TOTAL
    _charge(sealed, stage="screen", recipe="g998_e010", seed=1001, actual=500.0)
    before = e_path.read_bytes()
    with pytest.raises(B.ParentLedgerMovedError) as raised:
        e.reserve(stage="screen", recipe=E_ARM, seed=1002, segment_index=0,
                  start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
                  shutdown_allowance_s=60.0)
    message = str(raised.value)
    assert message.startswith(f"the parent ledger {_resolved(sealed_path)} has MOVED")
    assert repr(SEALED_TOTAL) in message and "re-snapshot" in message
    assert e_path.read_bytes() == before


def test_16_a_loop_in_the_chain_raises_integrity_error_not_recursion(tmp_path):
    """A -> B -> A. The file at A is replaced after B carried from it, which is
    the only way a loop can arise: every genuine descendant is created after its
    parent, so a loop is never a real lineage."""
    a_path = tmp_path / "a.jsonl"
    b_path = tmp_path / "b.jsonl"
    a_next = tmp_path / "a_next.jsonl"
    B.BudgetLedger.create(a_path, protocol_digest="a" * 64, clock=FakeClock())
    B.BudgetLedger.create(b_path, protocol_digest="b" * 64,
                          carried=B.carry_forward(a_path), clock=FakeClock())
    B.BudgetLedger.create(a_next, protocol_digest="a" * 64,
                          carried=B.carry_forward(b_path), clock=FakeClock())
    os.replace(a_next, a_path)
    head = B.BudgetLedger.open(a_path, inspect=True)
    with pytest.raises(B.LedgerIntegrityError, match="loop") as raised:
        head.assert_parent_unmoved()
    assert not isinstance(raised.value, RecursionError)
    message = str(raised.value)
    assert _resolved(a_path) in message and _resolved(b_path) in message


# --------------------------------------------------------------------------
# 17-21: the launch lock in reserve and create
# --------------------------------------------------------------------------

def test_17_reserve_is_refused_while_unauthorised_with_zero_rows(chain):
    chain.build_sealed()
    chain.build_e()                          # AUTHORISED stays False
    probe = _CountingProbe()
    chain.create_f(protocol=_WithoutLock())
    f = B.BudgetLedger.open(chain.f_path, protocol=F, contention_probe=probe)
    before = chain.f_path.read_bytes()
    with pytest.raises(B.BudgetError) as raised:
        _reserve_f(f)
    assert _is_unauthorised(raised.value), repr(raised.value)
    assert isinstance(raised.value.__cause__, F.AuthorisationError)
    assert "AUTHORISED" in str(raised.value)
    assert chain.f_path.read_bytes() == before
    assert probe.calls == 0
    assert f.pending is None


def test_18_create_is_refused_while_unauthorised_with_no_file(chain):
    chain.build_sealed()
    chain.build_e()
    # Layer 1: the source constant is False.
    with pytest.raises(B.BudgetError) as raised:
        chain.create_f()
    assert _is_unauthorised(raised.value), repr(raised.value)
    assert not chain.f_path.exists()
    # Layer 2: the constant is True but no record exists.
    chain.monkeypatch.setattr(F, "AUTHORISED", True)
    with pytest.raises(B.BudgetError) as raised:
        chain.create_f()
    assert _is_unauthorised(raised.value), repr(raised.value)
    assert not chain.f_path.exists()
    # Layer 2: a record bound to some other digest.
    chain.write_record(dict(chain.valid_record(), protocol_digest="0" * 64))
    with pytest.raises(B.BudgetError) as raised:
        chain.create_f()
    assert _is_unauthorised(raised.value), repr(raised.value)
    assert not chain.f_path.exists()


def test_19_the_refusal_is_a_budget_error_subclass():
    cls = getattr(B, "UnauthorisedLaunchError", None)
    assert cls is not None, "ppo_v2_budget declares no UnauthorisedLaunchError"
    assert issubclass(cls, B.BudgetError)
    assert cls is not B.BudgetError


def test_20_sealed_and_e_declare_no_lock_and_reserve_without_a_record(chain):
    assert not hasattr(P, "assert_authorised")
    assert not hasattr(E, "assert_authorised")
    assert callable(F.assert_authorised)
    chain.build_sealed()
    chain.build_e()                            # both reserved and settled
    assert not chain.record_path.exists()
    assert chain.e.charged()["settled_rows"] == 1
    assert chain.sealed.charged()["settled_rows"] == 3


def test_21_revoking_the_record_after_creation_refuses_the_next_reserve(chain):
    f = chain.build()
    _charge_f(f, seed=4001, actual=318.3)
    chain.record_path.unlink()
    before = chain.f_path.read_bytes()
    with pytest.raises(B.BudgetError) as raised:
        _reserve_f(f, seed=4002)
    assert _is_unauthorised(raised.value), repr(raised.value)
    assert chain.f_path.read_bytes() == before
    chain.write_record(dict(chain.valid_record(), authorised=False))
    with pytest.raises(B.BudgetError) as raised:
        _reserve_f(f, seed=4002)
    assert _is_unauthorised(raised.value), repr(raised.value)
    assert chain.f_path.read_bytes() == before
    assert f.pending is None


# --------------------------------------------------------------------------
# 22-25: the funding line
# --------------------------------------------------------------------------

def test_22_charges_accrue_to_refine_only(chain):
    f = chain.build()
    screen_before = f.charged()["stage_s"]["screen"]
    _charge_f(f, seed=4001, actual=318.3)
    _charge_f(f, seed=4002, actual=441.0)
    charged = f.charged()
    assert set(charged["stage_s"]) == {"screen", "refine"}
    assert charged["stage_s"]["refine"] == 318.3 + 441.0
    assert charged["stage_s"]["screen"] == screen_before
    assert charged["run_s"][f"refine|{F_RECIPE}|4001"] == 318.3
    assert charged["run_s"][f"refine|{F_RECIPE}|4002"] == 441.0
    before = chain.f_path.read_bytes()
    for stage, seed in (("screen", 1001), ("confirm", 3001), ("screen", 4001)):
        with pytest.raises(P.ProtocolError):
            _reserve_f(f, stage=stage, seed=seed)
    assert chain.f_path.read_bytes() == before


def _fifteen_cells(f, wall):
    cells = F.run_order()
    assert len(cells) == 16
    for segment_index, seed in cells[:15]:
        _charge_f(f, seed=seed, actual=wall, segment_index=segment_index)
    return cells[15]


def test_23a_the_16th_reservation_is_fundable_after_15_x_318_3(chain):
    f = chain.build()
    segment_index, seed = _fifteen_cells(f, 318.3)
    assert (segment_index, seed) == (7, 4002)
    remaining = f.remaining("refine", F_RECIPE, seed)
    assert remaining.stage_s == pytest.approx(7200 - 15 * 318.3)
    assert remaining.least >= 600
    reservation = _reserve_f(f, seed=seed, segment_index=7, start_epoch=448,
                             end_epoch=512)
    f.settle(reservation, actual_wall_s=318.3, returncode=0, counters=_counters())
    assert f.charged()["stage_s"]["refine"] == pytest.approx(16 * 318.3)


def test_23b_the_16th_reservation_is_refused_after_15_x_441_naming_the_stage_cap(chain):
    f = chain.build()
    segment_index, seed = _fifteen_cells(f, 441.0)
    remaining = f.remaining("refine", F_RECIPE, seed)
    assert remaining.stage_s == pytest.approx(585.0)
    assert remaining.binding == "stage"
    before = chain.f_path.read_bytes()
    with pytest.raises(B.BudgetExhaustedError, match="the stage cap"):
        _reserve_f(f, seed=seed, segment_index=segment_index, start_epoch=448,
                   end_epoch=512)
    assert chain.f_path.read_bytes() == before


def test_24_seeds_other_than_4001_and_4002_are_refused(chain):
    f = chain.build()
    before = chain.f_path.read_bytes()
    for seed in (4000, 4003, 1001, 2001, 3001):
        with pytest.raises(B.BudgetError, match="not a scheduled seed"):
            _reserve_f(f, seed=seed)
    assert chain.f_path.read_bytes() == before


def test_25_end_epoch_above_512_is_refused(chain):
    f = chain.build()
    before = chain.f_path.read_bytes()
    for start, end in ((512, 576), (480, 544)):
        with pytest.raises(B.BudgetError, match="epoch cap 512"):
            _reserve_f(f, start_epoch=start, end_epoch=end)
    assert chain.f_path.read_bytes() == before


# --------------------------------------------------------------------------
# 26-31: no interchange, digest binding and no bypass
# --------------------------------------------------------------------------

def test_26_f_ledger_is_refused_under_e_or_sealed(chain):
    chain.build()
    before = chain.f_path.read_bytes()
    for protocol in (E, P, None):
        with pytest.raises(B.LedgerIntegrityError, match="never interchangeable"):
            B.BudgetLedger.open(chain.f_path, protocol=protocol)
    assert chain.f_path.read_bytes() == before


def test_27_e_ledger_is_refused_under_f(chain):
    chain.build()
    before = chain.e_path.read_bytes()
    with pytest.raises(B.LedgerIntegrityError, match="never interchangeable"):
        B.BudgetLedger.open(chain.e_path, protocol=F)
    with pytest.raises(B.LedgerIntegrityError, match="never interchangeable"):
        B.BudgetLedger.open(chain.sealed_path, protocol=F)
    assert chain.e_path.read_bytes() == before


def test_28_sealed_and_e_arms_cannot_reserve_on_f_ledger(chain):
    f = chain.build()
    before = chain.f_path.read_bytes()
    for name in SEALED_ARMS:
        with pytest.raises(P.ProtocolError, match="sealed stage-1 arm"):
            _reserve_f(f, recipe=name)
    for name in E.RECIPES:
        with pytest.raises(P.ProtocolError, match="ppo_v2_protocol_e"):
            _reserve_f(f, recipe=name)
    assert chain.f_path.read_bytes() == before


def test_29_f_arm_cannot_reserve_on_e_or_the_sealed_ledger(chain):
    chain.build()
    e_before = chain.e_path.read_bytes()
    sealed_before = chain.sealed_path.read_bytes()
    with pytest.raises(P.ProtocolError, match="unknown amendment recipe"):
        chain.e.reserve(stage="screen", recipe=F_RECIPE, seed=1001,
                        segment_index=0, start_epoch=0, end_epoch=64,
                        reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    for stage, seed in (("screen", 1001), ("refine", 2001)):
        with pytest.raises(P.ProtocolError, match="unknown recipe"):
            chain.sealed.reserve(stage=stage, recipe=F_RECIPE, seed=seed,
                                 segment_index=0, start_epoch=0, end_epoch=64,
                                 reserved_bound_s=540.0,
                                 shutdown_allowance_s=60.0)
    assert chain.e_path.read_bytes() == e_before
    assert chain.sealed_path.read_bytes() == sealed_before


def test_30_digest_drift_after_creation_refuses_the_ledger(chain):
    f = chain.build()
    written = f.protocol_digest
    before = chain.f_path.read_bytes()
    chain.monkeypatch.setattr(F, "FUNDING_CAP_S", 7000)
    chain.monkeypatch.setattr(F, "STAGE",
                              dataclasses.replace(F.STAGE, wall_cap_s=7000))
    assert F.stage("refine").wall_cap_s == 7000     # a coherent patched build
    assert F.protocol_digest() != written
    with pytest.raises(B.LedgerIntegrityError, match="never interchangeable"):
        B.BudgetLedger.open(chain.f_path, protocol=F)
    with pytest.raises(B.LedgerIntegrityError):
        _reserve_f(f)
    assert chain.f_path.read_bytes() == before


def test_31_the_budget_source_contains_no_bypass_strings():
    source = BUDGET_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("allow_stale", "ignore_parent", "force_carried",
                      "skip_parent", "os.environ"):
        assert forbidden not in source, forbidden


# --------------------------------------------------------------------------
# (e): the deleted-ledger re-grant guard on RUN_ROOT
# --------------------------------------------------------------------------

def _prepared(chain):
    chain.build_sealed()
    chain.build_e()
    chain.authorise()


def _listing(root) -> list:
    return sorted(os.path.relpath(os.path.join(current, name), root)
                  for current, dirs, files in os.walk(root)
                  for name in (*dirs, *files))


def test_e01_create_is_refused_when_run_root_holds_a_launch_json(chain):
    _prepared(chain)
    (chain.run_root / "launch.json").write_text("{}", encoding="utf-8")
    before = _listing(chain.run_root)
    with pytest.raises(B.LedgerIntegrityError) as raised:
        chain.create_f()
    message = str(raised.value)
    assert "launch.json" in message
    assert str(chain.run_root) in message
    assert "re-grant" in message
    assert not chain.f_path.exists()
    assert _listing(chain.run_root) == before


@pytest.mark.parametrize("layout", [
    f"{F_RECIPE}_s4001/segment_0000/launch.json",   # the runner's real layout
    f"{F_RECIPE}_s4002/segment_0003/",              # a segment dir, nothing in it
    "segment_0000/",                                # a segment dir at the top
    "a/b/c/launch.json",                            # launch.json at any depth
    f"{F_RECIPE}_s4001/launch.json/",               # a directory named launch.json
])
def test_e02_create_is_refused_for_a_nested_segment_dir_or_launch_json(chain, layout):
    _prepared(chain)
    target = chain.run_root / layout
    if layout.endswith("/"):
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True)
        target.write_text("{}", encoding="utf-8")
    before = _listing(chain.run_root)
    with pytest.raises(B.LedgerIntegrityError) as raised:
        chain.create_f()
    leaf = Path(layout.rstrip("/")).name
    assert leaf in str(raised.value)
    assert not chain.f_path.exists()
    assert _listing(chain.run_root) == before


def test_e03a_create_is_allowed_when_run_root_is_absent(chain):
    absent = chain.tmp / "absent_run_root"
    ledger_dir = chain.tmp / "ledger_home"
    record_dir = chain.tmp / "auth"
    ledger_dir.mkdir()
    chain.relocate(run_root=absent, ledger_dir=ledger_dir, record_dir=record_dir)
    _prepared(chain)
    f = chain.create_f()
    assert f.path.is_file()
    assert not absent.exists()              # the guard creates nothing


def test_e03b_create_is_allowed_when_run_root_is_empty(chain):
    empty = chain.tmp / "empty_run_root"
    empty.mkdir()
    ledger_dir = chain.tmp / "ledger_home"
    ledger_dir.mkdir()
    chain.relocate(run_root=empty, ledger_dir=ledger_dir,
                   record_dir=chain.tmp / "auth")
    _prepared(chain)
    chain.create_f()
    assert os.listdir(empty) == []


def test_e03c_create_is_allowed_beside_the_record_and_unrelated_files(chain):
    """The authorisation record lives INSIDE the run root, so the guard cannot
    demand an empty directory."""
    _prepared(chain)
    (chain.run_root / "notes.txt").write_text("free text", encoding="utf-8")
    (chain.run_root / "plots").mkdir()
    (chain.run_root / "plots" / "summary.json").write_text("{}", encoding="utf-8")
    assert chain.record_path.is_file()
    f = chain.create_f()
    assert f.path == chain.f_path and f.path.is_file()


def test_e04_no_ledger_file_is_written_on_any_refusal(chain):
    _prepared(chain)
    segment = chain.run_root / f"{F_RECIPE}_s4001" / "segment_0000"
    segment.mkdir(parents=True)
    probe = _CountingProbe()
    before = _listing(chain.run_root)
    for _ in range(2):
        with pytest.raises(B.LedgerIntegrityError):
            chain.create_f(probe=probe)
    assert _listing(chain.run_root) == before
    assert not chain.f_path.exists()
    assert probe.calls == 0


def test_e05_probe_e_and_the_sealed_ledger_are_unaffected(tmp_path, monkeypatch):
    """E declares RUN_ROOT but not BIND_LEDGER_PATH, and its real run root does
    hold launch.json files, so the guard must stay opt-in."""
    assert not hasattr(P, "RUN_ROOT")
    assert hasattr(E, "RUN_ROOT") and not hasattr(E, "BIND_LEDGER_PATH")
    sealed_dir = tmp_path / "sealed_root"
    (sealed_dir / "segment_0000").mkdir(parents=True)
    (sealed_dir / "segment_0000" / "launch.json").write_text("{}", encoding="utf-8")
    sealed_path = sealed_dir / "budget_ledger.jsonl"
    B.BudgetLedger.create(sealed_path, protocol_digest=P.protocol_digest(),
                          clock=FakeClock())
    e_root = tmp_path / "e_root"
    (e_root / f"{E_ARM}_s1001" / "segment_0000").mkdir(parents=True)
    (e_root / f"{E_ARM}_s1001" / "segment_0000" / "launch.json").write_text(
        "{}", encoding="utf-8")
    monkeypatch.setattr(E, "PARENT_LEDGER", str(sealed_path))
    monkeypatch.setattr(E, "RUN_ROOT", str(e_root))
    monkeypatch.setattr(E, "LEDGER", str(e_root / "budget_ledger.jsonl"))
    e = B.BudgetLedger.create(e_root / "budget_ledger.jsonl",
                              protocol_digest=E.protocol_digest(), protocol=E,
                              carried=B.carry_forward(sealed_path),
                              provenance=E.ledger_provenance(), clock=FakeClock())
    _charge(e, stage="screen", recipe=E_ARM, seed=1001, actual=100.0)


def test_e06_run_root_that_is_not_a_directory_is_refused(chain):
    not_a_dir = chain.tmp / "run_root_file"
    not_a_dir.write_text("not a directory", encoding="utf-8")
    ledger_dir = chain.tmp / "ledger_home"
    ledger_dir.mkdir()
    chain.relocate(run_root=not_a_dir, ledger_dir=ledger_dir,
                   record_dir=chain.tmp / "auth")
    _prepared(chain)
    with pytest.raises(B.LedgerIntegrityError, match="not a directory"):
        chain.create_f()
    assert not chain.f_path.exists()


def test_e07_a_bound_protocol_without_a_run_root_is_refused(chain):
    """Fail closed: a protocol that binds its ledger path must also say where
    its launches live, or the guard could not be applied."""
    _prepared(chain)
    with pytest.raises(B.LedgerIntegrityError, match="RUN_ROOT"):
        chain.create_f(protocol=_WithoutRunRoot())
    assert not chain.f_path.exists()


def test_e08_deleting_the_ledger_after_a_launch_cannot_re_grant_the_cap(chain):
    """The realistic re-grant. A launch writes launch.json inside its segment
    directory within reserve, exactly as the runner's create callable does. The
    ledger is then deleted, and re-creating it would reopen the whole funding
    line at zero spend."""
    f = chain.build()

    def _create_segment():
        out_dir = chain.run_root / f"{F_RECIPE}_s4001" / "segment_0000"
        out_dir.mkdir(parents=True)
        (out_dir / "launch.json").write_text("{}", encoding="utf-8")

    reservation = _reserve_f(f, create=_create_segment)
    f.settle(reservation, actual_wall_s=441.0, returncode=0, counters=_counters())
    assert f.charged()["stage_s"]["refine"] == 441.0
    chain.f_path.unlink()
    with pytest.raises(B.LedgerIntegrityError, match="launch.json"):
        chain.create_f()
    assert not chain.f_path.exists()


# --------------------------------------------------------------------------
# Supplementary
# --------------------------------------------------------------------------

def test_s01_resnapshotting_f_before_e_is_refused_naming_the_order(chain):
    """E moved (its own charge) and then went stale (the sealed ledger moved).
    Adopting E's balance now would carry a stale total into F, so F's re-snapshot
    refuses and names the order: re-snapshot E, then F."""
    f = chain.build()
    chain.advance_e(250.5)
    chain.advance_sealed(500.0)
    before = chain.f_path.read_bytes()
    with pytest.raises(B.ParentLedgerMovedError) as raised:
        f.resnapshot_parent(detail="trying F first")
    message = str(raised.value)
    assert f"re-snapshot {_resolved(chain.e_path)}, then {_resolved(chain.f_path)}" in message
    assert chain.f_path.read_bytes() == before
    chain.e.resnapshot_parent(detail="the sealed ledger advanced")
    f.resnapshot_parent(detail="probe E advanced and re-snapshotted")
    assert f.charged()["global_s"] == chain.e.charged()["global_s"]
    assert f.charged()["global_s"] == pytest.approx(E_TOTAL + 250.5 + 500.0)
    _reserve_f(f)


def test_s02_revoking_the_record_never_wedges_a_pending_settlement(chain):
    """The lock gates new reservations only. A charge already committed must
    still land, or the ledger would wedge with a pending reservation."""
    f = chain.build()
    reservation = _reserve_f(f)
    chain.record_path.unlink()
    f.mark_spawn(reservation, detail="argv_sha256=" + "0" * 64)
    f.settle(reservation, actual_wall_s=441.0, returncode=0, counters=_counters())
    assert f.charged()["stage_s"]["refine"] == 441.0
    assert f.pending is None


def test_s03_an_unreadable_grandparent_refuses_rather_than_falling_back(chain):
    f = chain.build()
    before = chain.f_path.read_bytes()
    chain.sealed_path.unlink()
    with pytest.raises(B.LedgerIntegrityError) as raised:
        _reserve_f(f)
    assert _resolved(chain.sealed_path) in str(raised.value)
    assert chain.f_path.read_bytes() == before


def test_s04_an_over_deep_chain_is_refused_as_an_integrity_error(tmp_path):
    """The recursion depth is bounded structurally, so even a long chain of
    distinct ledgers ends in LedgerIntegrityError, never RecursionError."""
    parent = tmp_path / "parent.jsonl"
    child = tmp_path / "child.jsonl"
    B.BudgetLedger.create(parent, protocol_digest="a" * 64, clock=FakeClock())
    ledger = B.BudgetLedger.create(child, protocol_digest="b" * 64,
                                   carried=B.carry_forward(parent),
                                   clock=FakeClock())
    visited = tuple(str(tmp_path / f"ancestor_{index}.jsonl")
                    for index in range(B.MAX_CARRY_CHAIN_DEPTH))
    with pytest.raises(B.LedgerIntegrityError, match="deep"):
        ledger.assert_parent_unmoved(_seen=visited)
    ledger.assert_parent_unmoved()


def test_s05_create_and_reserve_gain_no_switch_parameters():
    """No keyword was added that could disable a check."""
    create = set(inspect.signature(B.BudgetLedger.create).parameters)
    assert create == {"path", "protocol_digest", "clock", "contention_probe",
                      "protocol", "carried", "provenance"}
    reserve = set(inspect.signature(B.BudgetLedger.reserve).parameters)
    assert reserve == {"self", "stage", "recipe", "seed", "segment_index",
                       "start_epoch", "end_epoch", "reserved_bound_s",
                       "shutdown_allowance_s", "create"}
    recheck = inspect.signature(B.BudgetLedger.assert_parent_unmoved).parameters
    assert set(recheck) <= {"self", "_seen"}


def test_s06_the_note_describes_the_transitive_rule_and_keeps_the_intent():
    note = B.PARENT_RECHECK_NOTE
    assert "transitive" in note.lower()
    assert "ancestor" in note.lower()
    assert "loop" in note.lower()
    assert "parent_moved_during_segment" in note
    assert "wedge" in note
