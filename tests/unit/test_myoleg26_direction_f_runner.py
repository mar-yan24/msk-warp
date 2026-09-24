"""Unit tests for running Direction F through the campaign runner.

CPU only. Fake clocks, fake child processes, real temporary ledgers and the
analytic CPU PPO adapter. There is no simulator, no CUDA, no Warp backend, no
real subprocess and no GPU query. Nothing here spends the 28,800 s training
budget or the 1,800 s diagnostic budget.

What is under test
------------------
* Direction F is registered by name. The sealed protocol stays the default, and
  a sealed launch's worker argv and ``launch.json`` keys are unchanged.
* The launch lock runs at the very start of ``run_launch``. While either layer
  is closed, no ledger file, run directory or child comes into existence, and
  the budget module is never even reached.
* The worker is locked too, before its build.
* The latent defect is fixed. The worker used to read the sealed stage record,
  so an F segment would have trained to cap 256, evaluated on 12000-12015 and
  never evaluated epochs 320..512. ``worker_stage`` now reads the launch
  protocol.
* ``effective_config`` refuses a protocol that declares another geometry.
* An F launch against the wrong ledger is refused, and so are the other
  protocols' launches against F's ledger.
* The sealed summarizer treats an F run as foreign.
* The committed manifest records F (tests 21-25).

The design-C section 3.7 runner list maps onto ``test_NN_...`` (1-25).
Supplementary checks are ``test_sNN_...``.

Isolation
---------
Every test that uses F's lock or ledger builds a temporary sealed -> E -> F
chain with the same patches as the Direction F ledger tests: ``E.PARENT_LEDGER``
and every F path (ledger, run root, authorisation record, preregistration)
point at temporary values. An autouse guard refuses any ``open`` or directory
listing of a path under the real ``logs/`` or ``docs/``.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import inspect
import io
import json
import os
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from msk_warp.analysis import ppo_v2_budget as B
from msk_warp.analysis import ppo_v2_protocol as P
from msk_warp.analysis import ppo_v2_protocol_e as E
from msk_warp.analysis import ppo_v2_protocol_f as F
from scripts import run_myoleg26_ppo_v2 as R
from scripts import summarize_myoleg26_ppo_v2 as Z
from tests.unit.test_ppo_resume_state import _build

ROOT = Path(__file__).resolve().parents[2]
V2_MANIFEST = ROOT / "msk_warp/configs/experiments/myoleg26_ppo_v2.json"
V1_MANIFEST = ROOT / "msk_warp/configs/experiments/myoleg26_baseline_v1.json"
F_MODULE = "msk_warp/analysis/ppo_v2_protocol_f.py"

#: Transcribed ancestor digests and the frozen preregistration hash.
SEALED_DIGEST = "4c01fdc94117bd190cd44c6b6a5f411a5c7cfc7b7d3e0462a344dda43baf2169"
E_DIGEST = "7c7892554cd79fb48832b1c431652f71776ef7e05b5359d4ccfc60f8c39ab68e"
PREREG_SHA256 = "acb777df3a7e40b42ae4571f7b715b56e069ac1e5bd3e8d6053f756d55095679"
PREREG_PATH = "docs/research/2026-09-23-session/orchestration/direction-f-preregistration.md"
V1_MANIFEST_SHA256 = "9e6d19e586d5ec1560e6dc2a95cadfd3295966d871315c67f6ab99f341ce04e1"

F_RECIPE = "g990_e010_f512"
E_ARM = "g990_e010_c040"
CONTROL_DT = 0.008

#: The sealed ``launch.json`` key set, which Direction F must not change.
SEALED_LAUNCH_KEYS = frozenset({
    "schema_version", "runner_schema", "protocol", "protocol_digest", "stage",
    "recipe", "seed", "segment_index", "start_epoch", "end_epoch", "epochs",
    "work_deadline_s", "reserved_bound_s", "shutdown_allowance_s",
    "remaining_before", "freeze", "freeze_sha256", "parent_segment",
    "parent_result", "device"})


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
                f"a Direction F runner test tried to open the real {file!r}")
        return real_open(file, *args, **kwargs)

    def guarded_scandir(path=".", *args, **kwargs):
        if _under_guarded_root(path):
            raise AssertionError(
                f"a Direction F runner test tried to list the real {path!r}")
        return real_scandir(path, *args, **kwargs)

    def guarded_listdir(path=".", *args, **kwargs):
        if _under_guarded_root(path):
            raise AssertionError(
                f"a Direction F runner test tried to list the real {path!r}")
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "scandir", guarded_scandir)
    monkeypatch.setattr(os, "listdir", guarded_listdir)
    yield


def test_s00_the_isolation_guard_has_teeth():
    """Anti-vacuity for the guard. The real ledgers, F's real run root and
    docs/ are refused before any byte is read; a tracked file is not."""
    with pytest.raises(AssertionError, match="real"):
        open(ROOT / "logs" / "myoleg26_ppo_v2" / "budget_ledger.jsonl", "rb")
    with pytest.raises(AssertionError, match="real"):
        (ROOT / "logs" / "myoleg26_ppo_v2_probe_e"
         / "budget_ledger.jsonl").read_bytes()
    with pytest.raises(AssertionError, match="real"):
        list(os.walk(ROOT / "logs"))
    with pytest.raises(AssertionError, match="real"):
        os.listdir(ROOT / "docs")
    with pytest.raises(AssertionError, match="real"):
        R.sha256_file(ROOT / F.AUTHORISATION_RECORD)
    assert Path(R.__file__).read_bytes()


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeClock:
    """An injected clock. Never sleeps, never reads time."""

    def __init__(self, start=1_700_000_000.0, step=0.0):
        self.now = float(start)
        self.step = float(step)

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


class TickingClock:
    """A fake clock that advances a fixed amount on every read."""

    def __init__(self, start=0.0, tick=1.0):
        self.now = float(start)
        self.tick = float(tick)

    def __call__(self):
        value = self.now
        self.now += self.tick
        return value


class FakeProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.pid = 4242

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


class Spawner:
    def __init__(self, on_spawn=None):
        self.calls = []
        self.on_spawn = on_spawn

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if self.on_spawn is not None:
            self.on_spawn(argv, kwargs)
        return FakeProcess()


class _CountingProbe:
    """Counts contention observations. Never queries a GPU."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return B.unobserved_contention()


class _Proxy:
    """Direction F with some attributes replaced, and nothing else changed."""

    def __init__(self, **overrides):
        self.__dict__["_overrides"] = dict(overrides)

    def __getattr__(self, name):
        overrides = self.__dict__["_overrides"]
        if name in overrides:
            return overrides[name]
        return getattr(F, name)


def _segment_counters():
    return B.SegmentCounters(
        attempted_control_transitions=8192, completed_control_transitions=8192,
        replayed_control_transitions=0, attempted_physics_steps=32768,
        completed_physics_steps=32768, replayed_physics_steps=0,
        partial_update_cost_s=0.0)


def _counter_dict():
    return {"attempted_control_transitions": 8192,
            "completed_control_transitions": 8192,
            "replayed_control_transitions": 0,
            "attempted_physics_steps": 32768,
            "completed_physics_steps": 32768,
            "replayed_physics_steps": 0,
            "partial_update_cost_s": 0.0}


def _charge(ledger, *, stage, recipe, seed, actual):
    reservation = ledger.reserve(
        stage=stage, recipe=recipe, seed=seed, segment_index=0, start_epoch=0,
        end_epoch=64, reserved_bound_s=540.0, shutdown_allowance_s=60.0)
    return ledger.settle(reservation, actual_wall_s=actual, returncode=0,
                         counters=_segment_counters())


def _refusing(name, calls):
    """A stand-in that records it was reached and then fails the test."""
    def refuse(*args, **kwargs):
        calls.append(name)
        raise AssertionError(f"{name} was reached while Direction F is locked")
    return refuse


def _tree(root) -> list:
    """Every path under ``root``, relative, so a refusal can be shown to write nothing."""
    found = []
    for current, dirs, files in os.walk(root):
        for name in (*dirs, *files):
            found.append(os.path.relpath(os.path.join(current, name), root))
    return sorted(found)


def _child(clock, out_dir, seconds):
    """A fake child: advances the clock and writes a completed-training result."""
    def child(argv, kwargs):
        clock.advance(seconds)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps({
            "schema_version": R.RESULT_SCHEMA, "training_began": True,
            "accounting": _counter_dict()}), encoding="utf-8")
    return child


# --------------------------------------------------------------------------
# The temporary chain
# --------------------------------------------------------------------------

class Chain:
    """A temporary sealed -> E -> F chain with a temporary prereg and record."""

    PREREG_BYTES = (b"# Direction F temporary preregistration\n\n"
                    b"line two of the frozen text\nline three\n")
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

    def build_sealed(self):
        self.sealed = B.BudgetLedger.create(
            self.sealed_path, protocol_digest=P.protocol_digest(),
            clock=FakeClock(step=1.0))
        for stage, recipe, seed, actual in self.SEALED_CHARGES:
            _charge(self.sealed, stage=stage, recipe=recipe, seed=seed,
                    actual=actual)
        return self.sealed

    def build_e(self):
        self.e = B.BudgetLedger.create(
            self.e_path, protocol_digest=E.protocol_digest(), protocol=E,
            carried=B.carry_forward(self.sealed_path),
            provenance=E.ledger_provenance(), clock=FakeClock(step=1.0))
        _charge(self.e, stage="screen", recipe=E_ARM, seed=1001,
                actual=self.E_CHARGE_S)
        return self.e

    def create_f(self):
        return B.BudgetLedger.create(
            self.f_path, protocol_digest=F.protocol_digest(), protocol=F,
            carried=B.carry_forward(self.e_path),
            provenance=F.ledger_provenance(), clock=FakeClock(step=1.0))

    def build(self):
        self.build_sealed()
        self.build_e()
        self.authorise()
        self.f = self.create_f()
        return self.f

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

    def run_dir(self, seed=4001) -> Path:
        return self.run_root / f"{F_RECIPE}_s{seed}"


@pytest.fixture
def chain(tmp_path, monkeypatch) -> Chain:
    return Chain(tmp_path, monkeypatch)


SEALED_TOTAL = 0.0 + 400.125 + 350.0625 + 300.3
E_TOTAL = SEALED_TOTAL + Chain.E_CHARGE_S


# --------------------------------------------------------------------------
# Argument helpers
# --------------------------------------------------------------------------

def _freeze(tmp_path) -> Path:
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({
        "schema_version": "myoleg26-baseline-freeze-v1",
        "model_sha256": "a" * 64,
        "compiled_model_sha256": "b" * 64,
        "files": {"msk_warp/algorithms/ppo.py": "c" * 64},
    }, sort_keys=True), encoding="utf-8")
    return path


def _compose(mode, values, overrides, flags=()) -> list:
    values = dict(values)
    for key, value in overrides.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = str(value)
    argv = [mode]
    for key, value in values.items():
        argv += [key, value]
    return argv + list(flags)


def _parse(argv):
    return R.build_parser().parse_args(argv)


def _f_launch(chain, *flags, **overrides):
    values = {
        "--protocol": "direction-f", "--stage": "refine", "--recipe": F_RECIPE,
        "--seed": "4001", "--segment-index": "0",
        "--freeze": str(_freeze(chain.tmp)), "--ledger": str(chain.f_path),
        "--run-dir": str(chain.run_dir()),
    }
    return _parse(_compose("launch", values, overrides, flags))


def _f_worker(chain, out_dir, **overrides):
    values = {
        "--protocol": "direction-f", "--stage": "refine", "--recipe": F_RECIPE,
        "--seed": "4001", "--segment-index": "0",
        "--freeze": str(_freeze(chain.tmp)), "--out-dir": str(out_dir),
        "--epochs": "64", "--work-deadline-s": "460", "--capture-reserve-s": "0",
    }
    return _parse(_compose("worker", values, overrides))


def _sealed_launch(tmp_path, ledger_path, **overrides):
    values = {
        "--stage": "screen", "--recipe": "g990_e010", "--seed": "1001",
        "--segment-index": "0", "--freeze": str(_freeze(tmp_path)),
        "--ledger": str(ledger_path), "--run-dir": str(tmp_path / "sealed_run"),
    }
    return _parse(_compose("launch", values, overrides))


def _e_launch(tmp_path, ledger_path, **overrides):
    values = {
        "--protocol": "probe-e", "--stage": "screen", "--recipe": E.PRIMARY_RECIPE,
        "--seed": "1001", "--segment-index": "0",
        "--freeze": str(_freeze(tmp_path)), "--ledger": str(ledger_path),
        "--run-dir": str(tmp_path / "e_run"),
    }
    return _parse(_compose("launch", values, overrides))


def _parents(tmp_path) -> dict:
    return {"--parent-segment": str(tmp_path / "parent.ptc"),
            "--parent-result": str(tmp_path / "parent.json")}


# ==========================================================================
# 1-3: registration, and the sealed path left alone
# ==========================================================================

def test_01_direction_f_is_registered_and_the_default_stays_sealed():
    assert set(R.PROTOCOLS) == {"sealed", "probe-e", "direction-f"}
    assert R.DEFAULT_PROTOCOL == "sealed"
    assert R.resolve_protocol("direction-f") is F
    assert R.resolve_protocol("sealed") is P
    assert R.resolve_protocol("probe-e") is E
    with pytest.raises(R.RunnerRefusal, match="direction-f"):
        R.resolve_protocol("direction-g")


def test_02_a_launch_that_never_names_a_protocol_is_the_unchanged_sealed_one(tmp_path):
    """Behavioural. The worker argv carries no --protocol, and ``launch.json``
    keeps exactly the sealed key set: no authorisation key appears."""
    ledger = B.BudgetLedger.create(tmp_path / "budget_ledger.jsonl",
                                   protocol_digest=P.protocol_digest(),
                                   clock=FakeClock(step=1.0))
    args = _sealed_launch(tmp_path, ledger.path)
    assert args.protocol == "sealed"
    assert R.launch_protocol(args) is P
    argv = R._worker_argv(args, R.resolve_plan(args))
    assert "--protocol" not in argv
    assert argv[2] == "worker"

    clock = FakeClock(start=0.0)
    out_dir = tmp_path / "sealed_run" / "segment_0000"
    code = R.run_launch(args, ledger=ledger,
                        spawn=Spawner(on_spawn=_child(clock, out_dir, 300.0)),
                        clock=clock)
    assert code == R.EXIT_OK
    launch = json.loads((out_dir / "launch.json").read_text(encoding="utf-8"))
    assert set(launch) == SEALED_LAUNCH_KEYS
    assert launch["protocol"] == "sealed"
    assert launch["protocol_digest"] == P.protocol_digest() == SEALED_DIGEST


def test_s02_a_probe_e_launch_json_gains_no_authorisation_key(tmp_path, monkeypatch):
    sealed = B.BudgetLedger.create(tmp_path / "sealed.jsonl",
                                   protocol_digest=P.protocol_digest(),
                                   clock=FakeClock(step=1.0))
    monkeypatch.setattr(E, "PARENT_LEDGER", str(sealed.path))
    ledger = B.BudgetLedger.create(
        tmp_path / "probe_e.jsonl", protocol_digest=E.protocol_digest(),
        protocol=E, carried=B.carry_forward(sealed.path),
        provenance=E.ledger_provenance(), clock=FakeClock(step=1.0))
    clock = FakeClock(start=0.0)
    out_dir = tmp_path / "e_run" / "segment_0000"
    args = _e_launch(tmp_path, ledger.path)
    assert R.run_launch(args, ledger=ledger,
                        spawn=Spawner(on_spawn=_child(clock, out_dir, 320.0)),
                        clock=clock) == R.EXIT_OK
    launch = json.loads((out_dir / "launch.json").read_text(encoding="utf-8"))
    assert set(launch) == SEALED_LAUNCH_KEYS
    assert launch["protocol"] == "probe-e"


def test_03_the_worker_argv_carries_the_direction_f_protocol(chain):
    chain.authorise()
    args = _f_launch(chain)
    plan = R.resolve_plan(args, protocol=F)
    argv = R._worker_argv(args, plan)
    assert argv.count("--protocol") == 1
    assert argv[argv.index("--protocol") + 1] == "direction-f"
    assert argv[argv.index("--stage") + 1] == "refine"
    assert argv[argv.index("--recipe") + 1] == F_RECIPE
    # The argv the child receives parses back to Direction F.
    parsed = R.build_parser().parse_args(argv[2:])
    assert parsed.mode == "worker" and parsed.protocol == "direction-f"
    assert R.launch_protocol(parsed) is F


# ==========================================================================
# 4-7: the launch lock
# ==========================================================================

@pytest.mark.parametrize("layer", ["constant_closed", "record_missing"])
def test_04_an_unauthorised_launch_creates_no_ledger_and_no_run_directory(
        chain, monkeypatch, layer):
    """Behavioural. The lock runs before the create-ledger branch, so the budget
    module is never reached: carry_forward, create and open are replaced by
    stand-ins that fail the test if called."""
    chain.build_sealed()
    chain.build_e()
    if layer == "constant_closed":
        chain.write_record()          # a well-formed record; the constant is False
        assert F.AUTHORISED is False
    else:
        monkeypatch.setattr(F, "AUTHORISED", True)
        assert not chain.record_path.exists()
    args = _f_launch(chain, "--create-ledger",
                     **{"--carry-forward-from": str(chain.e_path)})
    before = _tree(chain.tmp)
    sealed_bytes = chain.sealed_path.read_bytes()
    e_bytes = chain.e_path.read_bytes()

    calls = []
    monkeypatch.setattr(B, "carry_forward", _refusing("carry_forward", calls))
    monkeypatch.setattr(B.BudgetLedger, "create", _refusing("create", calls))
    monkeypatch.setattr(B.BudgetLedger, "open", _refusing("open", calls))
    probe = _CountingProbe()
    spawner = Spawner()

    with pytest.raises(R.RunnerRefusal) as caught:
        R.run_launch(args, spawn=spawner, clock=FakeClock(), probe=probe)

    assert isinstance(caught.value.__cause__, F.AuthorisationError)
    assert calls == []
    assert spawner.calls == []
    assert probe.calls == 0
    assert not chain.f_path.exists()
    assert not chain.run_dir().exists()
    assert _tree(chain.tmp) == before
    assert chain.sealed_path.read_bytes() == sealed_bytes
    assert chain.e_path.read_bytes() == e_bytes


@pytest.mark.parametrize("route", ["ledger_path", "injected_ledger"])
@pytest.mark.parametrize("revoke", ["constant_closed", "record_deleted"])
def test_05_a_revoked_launch_on_an_existing_f_ledger_leaves_it_byte_identical(
        chain, monkeypatch, route, revoke):
    ledger = chain.build()
    before = chain.f_path.read_bytes()
    if revoke == "constant_closed":
        monkeypatch.setattr(F, "AUTHORISED", False)
    else:
        chain.record_path.unlink()
    args = _f_launch(chain)
    tree = _tree(chain.tmp)
    spawner = Spawner()
    calls = []
    if route == "ledger_path":
        monkeypatch.setattr(B.BudgetLedger, "open", _refusing("open", calls))
        kwargs = {"probe": _CountingProbe()}
    else:
        kwargs = {"ledger": ledger}

    with pytest.raises(R.RunnerRefusal) as caught:
        R.run_launch(args, spawn=spawner, clock=FakeClock(), **kwargs)

    assert isinstance(caught.value.__cause__, F.AuthorisationError)
    assert calls == []
    assert spawner.calls == []
    assert chain.f_path.read_bytes() == before
    assert not chain.run_dir().exists()
    assert _tree(chain.tmp) == tree


@pytest.mark.parametrize("layer", ["constant_closed", "record_missing"])
def test_06_an_unauthorised_worker_is_refused_before_its_build(chain, monkeypatch, layer):
    if layer == "constant_closed":
        chain.write_record()
    else:
        monkeypatch.setattr(F, "AUTHORISED", True)
    out = chain.tmp / "worker" / "segment_0000"
    out.mkdir(parents=True)
    built = []

    def spy(args, plan):
        built.append(plan)
        raise AssertionError("the build ran while Direction F is locked")

    with pytest.raises(R.RunnerRefusal) as caught:
        R.run_worker(_f_worker(chain, out), build=spy, clock=TickingClock(tick=0.0))
    assert isinstance(caught.value.__cause__, F.AuthorisationError)
    assert built == []
    # What remains is the pre-training refusal record, with zero counters.
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["training_began"] is False
    assert result["status"] == R.REFUSED_BEFORE_TRAINING
    assert sorted(path.name for path in out.iterdir()) == ["result.json"]


def test_07_resolve_plan_refuses_a_valid_cell_until_authorised(chain):
    args = _f_launch(chain)
    with pytest.raises(R.RunnerRefusal) as caught:
        R.resolve_plan(args, protocol=F)
    assert isinstance(caught.value.__cause__, F.AuthorisationError)
    chain.authorise()
    plan = R.resolve_plan(args, protocol=F)
    assert (plan.stage, plan.recipe, plan.seed) == ("refine", F_RECIPE, 4001)
    assert (plan.start_epoch, plan.end_epoch, plan.epochs) == (0, 64, 64)


def test_s03_the_lock_is_the_first_thing_run_launch_does_with_the_protocol():
    """Structural. In ``run_launch`` the lock call precedes the create-or-open
    branch, so no future edit to that branch can run ahead of the lock."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(R.run_launch)))
    body = tree.body[0].body
    lock_at = branch_at = None
    for index, node in enumerate(body):
        text = ast.unparse(node)
        if lock_at is None and "_assert_launch_authorised(protocol)" in text:
            lock_at = index
        if (branch_at is None and isinstance(node, ast.If)
                and ast.unparse(node.test) == "ledger is None"):
            branch_at = index
    assert lock_at is not None and branch_at is not None
    assert lock_at < branch_at
    assert "protocol = launch_protocol(args)" in ast.unparse(body[lock_at - 1])


def test_s04_the_lock_helper_skips_unlocked_protocols_and_refuses_a_broken_lock(chain):
    assert R._assert_launch_authorised(P) is None
    assert R._assert_launch_authorised(E) is None
    with pytest.raises(R.RunnerRefusal, match="not callable"):
        R._assert_launch_authorised(_Proxy(assert_authorised=True))
    chain.authorise()
    assert R._assert_launch_authorised(F) == hashlib.sha256(
        chain.record_path.read_bytes()).hexdigest()


# ==========================================================================
# 8: an authorised launch
# ==========================================================================

def test_08_an_authorised_launch_reserves_spawns_once_settles_and_charges_refine(chain):
    ledger = chain.build()
    clock = FakeClock(start=0.0)
    out_dir = chain.run_dir() / "segment_0000"
    spawner = Spawner(on_spawn=_child(clock, out_dir, 318.3))

    code = R.run_launch(_f_launch(chain), ledger=ledger, spawn=spawner, clock=clock)

    assert code == R.EXIT_OK
    assert len(spawner.calls) == 1
    argv = spawner.calls[0]
    assert argv[argv.index("--protocol") + 1] == "direction-f"
    settled = [row for row in ledger.rows if row["kind"] == "settle"]
    assert len(settled) == 1
    assert settled[0]["stage"] == "refine"
    assert settled[0]["recipe"] == F_RECIPE
    assert settled[0]["seed"] == 4001
    assert settled[0]["charged_s"] == 318.3
    charged = ledger.charged()
    assert charged["stage_s"]["refine"] == 318.3
    assert charged["global_s"] == pytest.approx(E_TOTAL + 318.3, abs=1e-9)
    launch = json.loads((out_dir / "launch.json").read_text(encoding="utf-8"))
    assert launch["protocol"] == "direction-f"
    assert launch["protocol_digest"] == F.protocol_digest()
    assert launch["stage"] == "refine" and launch["recipe"] == F_RECIPE
    assert set(launch) == SEALED_LAUNCH_KEYS | {"authorisation_record_sha256"}
    assert launch["authorisation_record_sha256"] == hashlib.sha256(
        chain.record_path.read_bytes()).hexdigest()


def test_08b_the_cli_create_path_opens_the_f_ledger_at_e_settled_spend(chain):
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    clock = FakeClock(start=0.0)
    out_dir = chain.run_dir() / "segment_0000"
    args = _f_launch(chain, "--create-ledger",
                     **{"--carry-forward-from": str(chain.e_path)})

    code = R.run_launch(args, spawn=Spawner(on_spawn=_child(clock, out_dir, 318.3)),
                        clock=clock, probe=_CountingProbe())

    assert code == R.EXIT_OK
    created = B.BudgetLedger.open(chain.f_path, protocol=F)
    assert created.protocol_digest == F.protocol_digest()
    assert created.carried_forward["source_protocol_digest"] == E.protocol_digest()
    assert created.carried_forward["global_s"] == E_TOTAL
    header = json.loads(chain.f_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["protocol_provenance"] == json.loads(json.dumps(F.ledger_provenance()))
    assert created.charged()["stage_s"]["refine"] == 318.3


def test_08c_a_deleted_f_ledger_cannot_be_recreated_through_the_runner(chain):
    """Behavioural. After a launch, deleting the ledger and asking the runner to
    create it again is refused by the run-root guard, and nothing is written."""
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    clock = FakeClock(start=0.0)
    out_dir = chain.run_dir() / "segment_0000"
    create = ("--create-ledger",)
    carry = {"--carry-forward-from": str(chain.e_path)}
    assert R.run_launch(_f_launch(chain, *create, **carry),
                        spawn=Spawner(on_spawn=_child(clock, out_dir, 318.3)),
                        clock=clock, probe=_CountingProbe()) == R.EXIT_OK
    chain.f_path.unlink()
    args = _f_launch(chain, *create, **carry,
                     **{"--seed": "4002", "--run-dir": str(chain.run_dir(4002))})
    tree = _tree(chain.tmp)
    spawner = Spawner()

    with pytest.raises(R.RunnerRefusal, match="re-grant") as caught:
        R.run_launch(args, spawn=spawner, clock=FakeClock(), probe=_CountingProbe())

    assert "launch.json" in str(caught.value)
    assert not chain.f_path.exists()
    assert spawner.calls == []
    assert not chain.run_dir(4002).exists()
    assert _tree(chain.tmp) == tree


# ==========================================================================
# 9-12: the worker reads the launch protocol's stage
# ==========================================================================

def test_09_worker_stage_returns_the_protocols_own_cap_and_reset_block():
    spec, block = R.worker_stage(F, "refine")
    assert spec.epoch_cap_per_seed == 512
    assert block == tuple(range(14000, 14016))
    spec, block = R.worker_stage(P, "refine")
    assert spec.epoch_cap_per_seed == 256
    assert block == tuple(range(12000, 12016))
    spec, block = R.worker_stage(E, "screen")
    assert spec.epoch_cap_per_seed == 128
    assert block == tuple(range(11000, 11016))
    with pytest.raises(P.ProtocolError, match="screen"):
        R.worker_stage(F, "screen")


def _stub_resume(calls):
    """Stands in for ``resume_boundary``: re-arms the live objects, restores nothing."""
    def resume(path, algo, env, *, epoch, seed, segment_index,
               parent_segment_sha256, bindings, expected_segment_sha256=None):
        calls.append({"epoch": int(epoch), "seed": int(seed),
                      "segment_index": int(segment_index),
                      "expected_segment_sha256": expected_segment_sha256})
        algo._current_obs = env.reset()
        algo.iter_count = int(epoch)
        worlds = int(env.num_envs)
        state = SimpleNamespace(extra={"episode_accumulators": {
            "returns": [0.0] * worlds, "lengths": [0] * worlds}})
        return {"epoch": int(epoch), "stub": True}, state
    return resume


def _f_seam(evaluated):
    def build(args, plan):
        algo, env = _build(seed=7, steps_num=4, num_envs=4)

        def hook(epoch, deadline=None):
            evaluated.append(int(epoch))
            record = {"epoch": int(epoch), "kind": "selection",
                      "checkpoint": f"epoch_{int(epoch):04d}.pt"}
            return record, {"attempted_calls": 10, "completed_calls": 10, "worlds": 16}

        return R.WorkerContext(
            algo=algo, env=env, control_dt=CONTROL_DT,
            max_epochs_total=F.EPOCH_CAP_PER_SEED,
            bindings=R.freeze_bindings(args.freeze), evaluate=hook)
    return build


@pytest.mark.parametrize("segment, expected", [(4, 320), (7, 512)])
def test_10_a_late_f_segment_evaluates_the_epoch_it_completes(
        chain, monkeypatch, segment, expected):
    """Behavioural, and red before the fix. Under the sealed refine record the
    evaluation schedule stops at 256, so segment 4 (256-320) and segment 7
    (448-512) owned no evaluation at all and wrote no selection file."""
    chain.authorise()
    resumed = []
    monkeypatch.setattr(R, "resume_boundary", _stub_resume(resumed))
    parent = chain.tmp / "parent" / f"segment_{segment - 1:04d}"
    parent.mkdir(parents=True)
    (parent / "segment.ptc").write_bytes(b"stub parent segment")
    (parent / "result.json").write_text(json.dumps({
        "segment_index": segment - 1,
        "lineage": {"segment_sha256": "d" * 64,
                    "valid_boundary_epoch": 64 * segment,
                    "parent_segment_sha256": None}}), encoding="utf-8")
    out = chain.tmp / "worker" / f"segment_{segment:04d}"
    out.mkdir(parents=True)
    evaluated = []
    args = _f_worker(chain, out, **{
        "--segment-index": str(segment),
        "--parent-segment": str(parent / "segment.ptc"),
        "--parent-result": str(parent / "result.json")})

    code = R.run_worker(args, build=_f_seam(evaluated), clock=TickingClock(tick=0.0))

    assert code == R.EXIT_OK
    assert resumed == [{"epoch": 64 * segment, "seed": 4001,
                        "segment_index": segment - 1,
                        "expected_segment_sha256": "d" * 64}]
    assert evaluated == [expected]
    assert sorted(path.name for path in out.glob("selection_*.json")) == [
        f"selection_{expected:04d}.json"]
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["protocol"] == "direction-f"
    assert (result["start_epoch"], result["end_epoch"]) == (64 * segment, 64 * segment + 64)
    assert result["completed_epoch"] == 64 * segment + 64
    assert result["evaluations"] == [str(out / f"selection_{expected:04d}.json")]


def test_s10_the_real_build_path_trains_to_512_and_evaluates_on_14000(chain, monkeypatch):
    """Behavioural, and red before the fix. ``_build_runtime`` runs with its
    three heavy imports replaced by CPU fakes. The context it returns must carry
    F's cap, and its evaluation must use F's reset block."""
    chain.authorise()
    policy_calls = []
    evaluation_envs = []

    class FakeTrainEnv:
        num_envs = P.NUM_WORLDS
        control_dt = CONTROL_DT

    class FakePPO:
        def __init__(self, cfg):
            self.cfg = cfg
            self.num_envs = cfg["params"]["env"]["num_actors"]
            self.steps_num = cfg["params"]["config"]["steps_num"]
            self.env = FakeTrainEnv()
            self.device = "cpu"
            self.actor = object()
            self.obs_rms = None
            self._out = Path(cfg["params"]["general"]["logdir"])

        def save(self, name):
            (self._out / f"{name}.pt").write_bytes(b"fake checkpoint " + name.encode())

        def close(self):
            pass

    class FakeEvaluationEnv:
        def __init__(self, num_envs, **kwargs):
            self.num_envs = num_envs
            evaluation_envs.append(self)

    def evaluate_policy(env, actor, obs_rms, seeds, **kwargs):
        policy_calls.append(list(seeds))
        return {"fake": True}, None

    @contextlib.contextmanager
    def isolated_rng():
        yield

    fakes = {
        "msk_warp.algorithms.ppo": {"PPO": FakePPO},
        "msk_warp.analysis.myoleg26_baseline": {"evaluate_policy": evaluate_policy,
                                                "isolated_rng": isolated_rng},
        "msk_warp.envs.myoleg26_walk": {"MyoLeg26WalkEnv": FakeEvaluationEnv},
    }
    for name, members in fakes.items():
        module = types.ModuleType(name)
        for key, value in members.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    out = chain.tmp / "worker" / "segment_0000"
    out.mkdir(parents=True)
    args = _f_worker(chain, out)
    plan = R.resolve_plan(args, protocol=F)
    context = R._build_runtime(args, plan)

    assert context.max_epochs_total == 512
    assert [env.num_envs for env in evaluation_envs] == [16]
    config = json.loads((out / "effective_config.json").read_text(encoding="utf-8"))
    assert config["params"]["config"]["max_epochs"] == 512
    record, counter = context.evaluate(64)
    assert policy_calls == [list(range(14000, 14016))]
    assert record["reset_block"] == list(range(14000, 14016))
    assert record["stage"] == "refine" and counter["worlds"] == 16


def test_11_the_worker_source_reads_the_stage_through_worker_stage():
    for function in (R._build_runtime, R._worker_segment):
        text = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(function))))
        assert "worker_stage(" in text, function.__name__
        assert "P.stage(plan.stage)" not in text, function.__name__
        assert "P.selection_reset_block(" not in text, function.__name__


def test_12_segments_0_to_7_resolve_and_segment_8_is_refused(chain):
    chain.authorise()
    for index in range(8):
        extra = {} if index == 0 else _parents(chain.tmp)
        args = _f_launch(chain, **{"--segment-index": str(index)}, **extra)
        plan = R.resolve_plan(args, protocol=F)
        assert (plan.start_epoch, plan.end_epoch, plan.epochs) == (
            64 * index, 64 * index + 64, 64)
        assert plan.out_dir == chain.run_dir() / f"segment_{index:04d}"
    args = _f_launch(chain, **{"--segment-index": "8"}, **_parents(chain.tmp))
    with pytest.raises(R.RunnerRefusal, match=r"outside 0\.\.7"):
        R.resolve_plan(args, protocol=F)


# ==========================================================================
# 13-15: the configuration F builds
# ==========================================================================

def test_13_f_effective_config_is_the_sealed_g990_e010_one_except_max_epochs(tmp_path):
    sealed = R.effective_config(protocol=P, stage_key="screen", recipe_name="g990_e010",
                                seed=4001, device="cuda:0", logdir=str(tmp_path))
    amended = R.effective_config(protocol=F, stage_key="refine", recipe_name=F_RECIPE,
                                 seed=4001, device="cuda:0", logdir=str(tmp_path))
    config_s, config_f = sealed["params"]["config"], amended["params"]["config"]
    differing = {key for key in set(config_s) | set(config_f)
                 if config_s.get(key) != config_f.get(key)}
    assert differing == {"max_epochs"}
    assert (config_s["max_epochs"], config_f["max_epochs"]) == (128, 512)
    assert set(sealed["params"]) == set(amended["params"])
    for key in sealed["params"]:
        if key != "config":
            assert sealed["params"][key] == amended["params"][key], key
    assert {key: value for key, value in sealed.items() if key != "params"} == {
        key: value for key, value in amended.items() if key != "params"}
    assert amended["params"]["env"]["num_actors"] == P.NUM_WORLDS == 64


@pytest.mark.parametrize("overrides", [
    {"NUM_WORLDS": 32},
    {"CONTROLS_PER_WORLD_PER_EPOCH": P.CONTROLS_PER_WORLD_PER_EPOCH * 2},
])
def test_14_a_protocol_declaring_another_geometry_is_refused(tmp_path, overrides):
    """Behavioural, and red before the fix: the configuration used to be built
    with the sealed geometry whatever the protocol declared."""
    with pytest.raises(R.RunnerRefusal, match="geometry"):
        R.effective_config(protocol=_Proxy(**overrides), stage_key="refine",
                           recipe_name=F_RECIPE, seed=4001, device="cuda:0",
                           logdir=str(tmp_path))
    # The negative control: the same proxy with nothing replaced builds.
    config = R.effective_config(protocol=_Proxy(), stage_key="refine",
                                recipe_name=F_RECIPE, seed=4001, device="cuda:0",
                                logdir=str(tmp_path))
    assert config["params"]["config"]["max_epochs"] == 512


def test_15_the_built_arm_check_accepts_the_f_arm():
    class Algo:
        def __init__(self, **values):
            self.num_envs = P.NUM_WORLDS
            self.steps_num = P.CONTROLS_PER_WORLD_PER_EPOCH
            for key, value in values.items():
                setattr(self, key, value)

    arm = F.recipe(F_RECIPE)
    R.assert_built_arm(Algo(), protocol=F, arm=arm)
    with pytest.raises(R.RunnerRefusal, match="worlds"):
        R.assert_built_arm(Algo(num_envs=8), protocol=F, arm=arm)


# ==========================================================================
# 16-19: each launch against its own ledger only
# ==========================================================================

#: The two refusals a cross-protocol launch can meet: the loader's header-digest
#: binding (a ledger opened by path) and ``assert_protocol_unchanged`` (a ledger
#: object injected into ``run_launch``). Either one pins the reason.
INTERCHANGE_REFUSAL = "never interchangeable|but the current protocol digest is"


@pytest.mark.parametrize("route", ["ledger_path", "injected_ledger"])
@pytest.mark.parametrize("target", ["probe_e", "sealed"])
def test_16_17_an_f_launch_against_an_ancestor_ledger_is_refused(chain, route, target):
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    ledger = chain.e if target == "probe_e" else chain.sealed
    before = ledger.path.read_bytes()
    spawner = Spawner()
    args = _f_launch(chain, **{"--ledger": str(ledger.path)})
    kwargs = ({"ledger": ledger} if route == "injected_ledger"
              else {"probe": _CountingProbe()})
    with pytest.raises(R.RunnerRefusal, match=INTERCHANGE_REFUSAL):
        R.run_launch(args, spawn=spawner, clock=FakeClock(), **kwargs)
    assert ledger.path.read_bytes() == before
    assert spawner.calls == []
    assert not chain.run_dir().exists()
    assert not chain.f_path.exists()


@pytest.mark.parametrize("route", ["ledger_path", "injected_ledger"])
@pytest.mark.parametrize("protocol", ["sealed", "probe-e"])
def test_18_sealed_and_probe_e_launches_against_the_f_ledger_are_refused(
        chain, route, protocol):
    ledger = chain.build()
    before = chain.f_path.read_bytes()
    spawner = Spawner()
    if protocol == "sealed":
        args = _sealed_launch(chain.tmp, chain.f_path)
        run_dir = chain.tmp / "sealed_run"
    else:
        args = _e_launch(chain.tmp, chain.f_path)
        run_dir = chain.tmp / "e_run"
    kwargs = ({"ledger": ledger} if route == "injected_ledger"
              else {"probe": _CountingProbe()})
    with pytest.raises(R.RunnerRefusal, match=INTERCHANGE_REFUSAL):
        R.run_launch(args, spawn=spawner, clock=FakeClock(), **kwargs)
    assert chain.f_path.read_bytes() == before
    assert spawner.calls == []
    assert not run_dir.exists()


def test_19_the_f_arm_is_refused_under_sealed_and_probe_e_and_theirs_under_f(chain):
    sealed = _sealed_launch(chain.tmp, chain.tmp / "unused.jsonl",
                            **{"--recipe": F_RECIPE})
    with pytest.raises(R.RunnerRefusal, match="unknown recipe"):
        R.resolve_plan(sealed)
    amended = _e_launch(chain.tmp, chain.tmp / "unused.jsonl", **{"--recipe": F_RECIPE})
    with pytest.raises(R.RunnerRefusal, match="unknown amendment recipe"):
        R.resolve_plan(amended, protocol=E)
    chain.authorise()
    with pytest.raises(R.RunnerRefusal, match="sealed stage-1 arm"):
        R.resolve_plan(_f_launch(chain, **{"--recipe": "g990_e010"}), protocol=F)
    with pytest.raises(R.RunnerRefusal, match="ppo_v2_protocol_e"):
        R.resolve_plan(_f_launch(chain, **{"--recipe": E_ARM}), protocol=F)


# ==========================================================================
# 20: the sealed summarizer cannot absorb an F run
# ==========================================================================

def _run_tree(root, *, recipe, seed, stage):
    run_dir = root / f"{stage}_{recipe}_s{seed}"
    segment = run_dir / "segment_0000"
    segment.mkdir(parents=True)
    (segment / "launch.json").write_text(json.dumps({
        "schema_version": R.LAUNCH_SCHEMA, "stage": stage, "recipe": recipe,
        "seed": seed, "segment_index": 0}), encoding="utf-8")
    (segment / "result.json").write_text(json.dumps({
        "schema_version": R.RESULT_SCHEMA, "training_began": True,
        "stage": stage, "recipe": recipe, "seed": seed, "segment_index": 0,
        "completed_epoch": 64, "completed_epochs": 64, "censored": False,
        "censor_reason": None, "stop_reason": "epoch_budget", "published": True,
        "boundary_is_live": True, "evaluations": [],
        "accounting": dict(_counter_dict(), completed_control_transitions=524288),
    }), encoding="utf-8")
    return run_dir


def test_20_the_sealed_summarizer_reports_an_f_run_as_foreign_and_stops(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    _run_tree(root, recipe="g990_e010", seed=1001, stage="screen")
    _run_tree(root, recipe=F_RECIPE, seed=4001, stage="refine")
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps({"schema_version": R.FREEZE_SCHEMA, "files": {}}),
                        encoding="utf-8")

    report = Z.summarize(root, manifest)

    foreign = report["campaign"]["foreign_recipe_runs"]
    assert [(entry["recipe"], entry["stage"], entry["seed"]) for entry in foreign] == [
        (F_RECIPE, "refine", 4001)]
    assert "foreign_recipe_runs" in report["review_stop_reasons"]
    assert report["review_stop"] is True
    assert F_RECIPE not in report["recipes"].get("refine", {})
    assert report["campaign_cost_totals"]["completed_control_transitions"] == 524288
    assert report["foreign_cost_totals"]["completed_control_transitions"] == 524288


# ==========================================================================
# 21-25: the freeze records F
# ==========================================================================

def _frozen() -> dict:
    return json.loads(V2_MANIFEST.read_text(encoding="utf-8"))


def test_21_the_committed_manifest_records_f_locked_with_its_parent_and_prereg():
    entry = _frozen()["amendments"][F.AMENDMENT_ID]
    assert entry["digest"] == F.protocol_digest()
    assert entry["parent_protocol_digest"] == E_DIGEST == E.protocol_digest()
    assert entry["grandparent_protocol_digest"] == SEALED_DIGEST
    assert entry["authorised"] is False
    assert entry["preregistration"]["sha256"] == PREREG_SHA256
    assert entry["preregistration"]["path"] == PREREG_PATH
    assert entry == json.loads(json.dumps(
        {**F.amendment_snapshot(), "digest": F.protocol_digest()}))


def test_22_the_sealed_and_probe_e_digests_are_unchanged_in_the_manifest():
    frozen = _frozen()
    assert frozen["protocol"]["digest"] == SEALED_DIGEST == P.protocol_digest()
    assert frozen["amendments"][E.AMENDMENT_ID]["digest"] == E_DIGEST
    assert E.protocol_digest() == E_DIGEST


def test_23_the_f_module_is_pinned_in_head_at_its_on_disk_bytes():
    frozen = _frozen()
    assert F_MODULE in frozen["files"]
    assert F_MODULE in frozen["files_eol"]
    assert frozen["files_git"]["head_status"][F_MODULE] == "in_head"
    assert frozen["files"][F_MODULE] == R.sha256_file(ROOT / F_MODULE)
    assert set(frozen["amendments"]) == {E.AMENDMENT_ID, F.AMENDMENT_ID}


def test_24_the_freeze_checks_f_reset_block_and_records_its_entry():
    text = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(R.freeze_record_v2))))
    assert "F.assert_reset_blocks_disjoint()" in text
    assert "F.AMENDMENT_ID" in text
    assert "F.amendment_snapshot()" in text
    assert text.index("P.assert_reset_blocks_disjoint()") < text.index(
        "F.assert_reset_blocks_disjoint()")


def test_25_the_v1_manifest_is_untouched():
    assert hashlib.sha256(V1_MANIFEST.read_bytes()).hexdigest() == V1_MANIFEST_SHA256


# ==========================================================================
# Review round 1: the ledger copy, the run directory and the record sha
# ==========================================================================

def test_s11_a_copied_f_ledger_named_by_ledger_is_refused(chain):
    """Adversarial review A through the CLI. ``--ledger`` naming a byte copy of
    the F ledger is refused before any row, directory or child."""
    chain.build()
    fork = chain.tmp / "fork" / "budget_ledger.jsonl"
    fork.parent.mkdir()
    fork.write_bytes(chain.f_path.read_bytes())
    before, bound_before = fork.read_bytes(), chain.f_path.read_bytes()
    spawner = Spawner()
    args = _f_launch(chain, **{"--ledger": str(fork)})
    with pytest.raises(R.RunnerRefusal, match="binds its one ledger"):
        R.run_launch(args, spawn=spawner, clock=FakeClock(), probe=_CountingProbe())
    assert fork.read_bytes() == before
    assert chain.f_path.read_bytes() == bound_before
    assert spawner.calls == []
    assert not chain.run_dir().exists()


def _outside_run_dir(chain, monkeypatch, where):
    """A --run-dir that does not resolve inside F's run root, and its top."""
    outside = chain.tmp / "outside_runs"
    if where == "absolute":
        return outside / "x", outside
    if where == "dotdot":
        return chain.run_root / ".." / "outside_runs" / "x", outside
    elsewhere = chain.tmp / "some_cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return Path("runs") / "x", elsewhere / "runs"


@pytest.mark.parametrize("where", ["absolute", "relative", "dotdot"])
def test_s12_an_f_launch_is_refused_outside_its_run_root(chain, monkeypatch, where):
    """Adversarial review B. A launch directory outside RUN_ROOT would hide the
    launch from the deleted-ledger guard, so it is refused before any row,
    directory or child. A relative --run-dir resolves against the working
    directory and is judged by where it lands."""
    chain.build()
    before = chain.f_path.read_bytes()
    run_dir, top = _outside_run_dir(chain, monkeypatch, where)
    spawner = Spawner()
    args = _f_launch(chain, **{"--run-dir": str(run_dir)})
    with pytest.raises(R.RunnerRefusal, match="RUN_ROOT") as caught:
        R.run_launch(args, spawn=spawner, clock=FakeClock(), probe=_CountingProbe())
    assert str(chain.run_root.resolve()) in str(caught.value)
    assert chain.f_path.read_bytes() == before
    assert spawner.calls == []
    assert not top.exists()


def test_s13_an_outside_run_dir_is_refused_before_the_ledger_is_created(chain):
    """The run-directory check precedes the create-or-open branch: with
    --create-ledger, an outside --run-dir leaves no ledger and no directory."""
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    outside = chain.tmp / "outside_runs"
    args = _f_launch(chain, "--create-ledger",
                     **{"--carry-forward-from": str(chain.e_path),
                        "--run-dir": str(outside / "x")})
    spawner = Spawner()
    with pytest.raises(R.RunnerRefusal, match="RUN_ROOT"):
        R.run_launch(args, spawn=spawner, clock=FakeClock(), probe=_CountingProbe())
    assert not chain.f_path.exists()
    assert not outside.exists()
    assert spawner.calls == []
    assert _tree(chain.run_root) == ["user_authorisation.json"]


def _link_directory(kind, target, link):
    if kind == "symlink":
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError) as error:
            pytest.skip(f"directory symlinks unavailable here: {error}")
        return
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, AttributeError, OSError) as error:
        pytest.skip(f"junctions unavailable here: {error}")


@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_s14_a_linked_run_dir_that_leaves_the_run_root_is_refused(chain, kind):
    """Adversarial review B3. A link inside RUN_ROOT that points outside it is
    judged by its resolved target, so launch output cannot hide behind a link
    the run-root walk does not follow."""
    chain.build()
    outside = chain.tmp / "outside_runs"
    outside.mkdir()
    link = chain.run_root / "linked"
    _link_directory(kind, outside, link)
    before = chain.f_path.read_bytes()
    spawner = Spawner()
    args = _f_launch(chain, **{"--run-dir": str(link / "x")})
    with pytest.raises(R.RunnerRefusal, match="RUN_ROOT"):
        R.run_launch(args, spawn=spawner, clock=FakeClock(), probe=_CountingProbe())
    assert chain.f_path.read_bytes() == before
    assert spawner.calls == []
    assert list(outside.iterdir()) == []


def test_s15_the_recorded_sha_is_of_the_bytes_the_lock_verified(chain, monkeypatch):
    """Adversarial review J. The sha the launch records must be of the bytes the
    lock parsed, not of a second read that could see a swapped record."""
    chain.authorise()
    verified = chain.record_path.read_bytes()
    real = F.assert_authorised

    def lock_then_swap():
        result = real()
        chain.record_path.write_bytes(b'{"swapped": true}')
        return result

    monkeypatch.setattr(F, "assert_authorised", lock_then_swap)
    assert R._assert_launch_authorised(F) == hashlib.sha256(verified).hexdigest()


@pytest.mark.parametrize("returned", [None, "", "not-a-sha", "A" * 64, 7])
def test_s16_a_lock_that_names_no_verified_record_is_refused(chain, returned):
    """A protocol that declares an authorisation record must have its lock
    return the sha256 of the record it verified; anything else is refused
    rather than filled in by a second, unverified read."""
    chain.authorise()
    lock = _Proxy(assert_authorised=lambda: returned)
    with pytest.raises(R.RunnerRefusal, match="sha256"):
        R._assert_launch_authorised(lock)


# ==========================================================================
# Review round 2: the segment-directory bound, and a re-created ledger
# ==========================================================================

@pytest.mark.parametrize("route", ["attribute", "junction", "symlink"])
def test_s17_a_segment_dir_that_leaves_the_run_root_is_refused(chain, route):
    """Review x05. The run directory is inside RUN_ROOT but the segment
    directory is not: an out_dir handed to run_launch, or a link where the
    segment directory would be. The segment-directory bound refuses before the
    reservation, so the ledger is byte-identical, no child is spawned and
    nothing appears outside the run root."""
    ledger = chain.build()
    outside = chain.tmp / "outside"
    args = _f_launch(chain)
    if route == "attribute":
        args.out_dir = str(outside / "segment_0000")
    else:
        outside.mkdir()
        chain.run_dir().mkdir()
        _link_directory(route, outside, chain.run_dir() / "segment_0000")
    before = chain.f_path.read_bytes()
    spawner = Spawner()
    with pytest.raises(R.RunnerRefusal,
                       match=r"the segment directory .* not inside this "
                             r"protocol's RUN_ROOT"):
        R.run_launch(args, ledger=ledger, spawn=spawner, clock=FakeClock())
    assert chain.f_path.read_bytes() == before
    assert spawner.calls == []
    assert B.BudgetLedger.open(chain.f_path, inspect=True).pending is None
    assert not outside.exists() or list(outside.iterdir()) == []


def test_s18_a_protocol_omitted_re_create_is_refused_by_the_runner(chain):
    """Adversarial review round 2, G1, through the CLI. After a launch the
    ledger is deleted and written again by create with ``protocol`` omitted.
    A launch that merely opens it must be refused before any row, directory or
    child."""
    chain.build_sealed()
    chain.build_e()
    chain.authorise()
    clock = FakeClock(start=0.0)
    out_dir = chain.run_dir() / "segment_0000"
    assert R.run_launch(_f_launch(chain, "--create-ledger",
                                  **{"--carry-forward-from": str(chain.e_path)}),
                        spawn=Spawner(on_spawn=_child(clock, out_dir, 318.3)),
                        clock=clock, probe=_CountingProbe()) == R.EXIT_OK
    chain.f_path.unlink()
    B.BudgetLedger.create(chain.f_path, protocol_digest=F.protocol_digest(),
                          carried=B.carry_forward(chain.e_path),
                          provenance=F.ledger_provenance(),
                          clock=FakeClock(step=1.0))
    before = chain.f_path.read_bytes()
    tree = _tree(chain.tmp)
    spawner = Spawner()
    probe = _CountingProbe()
    args = _f_launch(chain, **{"--seed": "4002",
                               "--run-dir": str(chain.run_dir(4002))})
    with pytest.raises(R.RunnerRefusal):
        R.run_launch(args, spawn=spawner, clock=FakeClock(), probe=probe)
    assert spawner.calls == []
    assert probe.calls == 0
    assert chain.f_path.read_bytes() == before
    assert not chain.run_dir(4002).exists()
    assert _tree(chain.tmp) == tree
