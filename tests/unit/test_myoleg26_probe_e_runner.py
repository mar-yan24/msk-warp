"""Unit tests for running the probe-E amendment through the campaign runner.

CPU only. Fake clocks, fake child processes and real temp ledgers. **No
simulator, no CUDA, no Warp, no real subprocess, no training.** Nothing here
consumes the 28,800 s training budget or the 1,800 s diagnostic budget.

The properties under test
-------------------------
* A **sealed** launch is untouched: the default protocol is the sealed one and
  the worker argv is byte-identical to what stage 1 executed, so a future
  stage-1 replication runs the same command it always did.
* The amendment is selected explicitly, recorded in ``launch.json`` and
  ``result.json``, and refused against the wrong ledger.
* The amendment arm is refused under the sealed protocol and the sealed arms are
  refused under the amendment, so **no run can be charged to a stage-1 cell**.
* The trust-region value reaches the built configuration, and the worker refuses
  if the built algorithm does not actually carry it.
* The sealed summarizer cannot quietly absorb a probe-E run into a stage-1
  table or a stage-1 cost total.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msk_warp.analysis import ppo_v2_budget as B
from msk_warp.analysis import ppo_v2_protocol as P
from msk_warp.analysis import ppo_v2_protocol_e as E
from scripts import run_myoleg26_ppo_v2 as R
from scripts import summarize_myoleg26_ppo_v2 as Z

ROOT = Path(__file__).resolve().parents[2]
V2_MANIFEST = ROOT / "msk_warp/configs/experiments/myoleg26_ppo_v2.json"


class FakeClock:
    def __init__(self, start=1000.0, step=0.0):
        self.now = float(start)
        self.step = float(step)

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


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


def _freeze(tmp_path) -> Path:
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({
        "schema_version": "myoleg26-baseline-freeze-v1",
        "model_sha256": "a" * 64,
        "compiled_model_sha256": "b" * 64,
        "files": {"msk_warp/algorithms/ppo.py": "c" * 64},
    }, sort_keys=True), encoding="utf-8")
    return path


def _argv(tmp_path, **overrides):
    values = {
        "--stage": "screen",
        "--recipe": "g990_e010",
        "--seed": "1001",
        "--segment-index": "0",
        "--freeze": str(_freeze(tmp_path)),
        "--ledger": str(tmp_path / "budget_ledger.jsonl"),
        "--run-dir": str(tmp_path / "run"),
    }
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


def _counters():
    return {"attempted_control_transitions": 8192,
            "completed_control_transitions": 8192,
            "replayed_control_transitions": 0,
            "attempted_physics_steps": 32768,
            "completed_physics_steps": 32768,
            "replayed_physics_steps": 0,
            "partial_update_cost_s": 0.0}


def _sealed_ledger(tmp_path, clock, *, name="budget_ledger.jsonl"):
    return B.BudgetLedger.create(tmp_path / name,
                                 protocol_digest=P.protocol_digest(), clock=clock)


def _amended_ledger(tmp_path, clock, *, source=None):
    source = source or _sealed_ledger(tmp_path, FakeClock(), name="sealed.jsonl")
    return B.BudgetLedger.create(
        tmp_path / "probe_e_ledger.jsonl", protocol_digest=E.protocol_digest(),
        protocol=E, carried=B.carry_forward(source.path),
        provenance=E.ledger_provenance(), clock=clock)


# ==========================================================================
# Unit 1 -- protocol selection, and the sealed path left alone
# ==========================================================================

def test_the_default_protocol_is_the_sealed_one():
    assert R.DEFAULT_PROTOCOL == "sealed"
    assert R.resolve_protocol("sealed") is P
    assert R.resolve_protocol(R.DEFAULT_PROTOCOL) is P
    assert R.resolve_protocol("probe-e") is E
    assert set(R.PROTOCOLS) == {"sealed", "probe-e"}


def test_an_unknown_protocol_is_refused_by_name():
    with pytest.raises(R.RunnerRefusal, match="probe-e"):
        R.resolve_protocol("whatever")


def test_a_launch_that_never_mentions_the_protocol_gets_the_sealed_one(tmp_path):
    args = _parse(_argv(tmp_path))
    assert args.protocol == "sealed"
    assert R.resolve_protocol(args.protocol) is P


@pytest.mark.parametrize("mode", ["launch", "worker"])
def test_both_identity_modes_accept_the_protocol_flag(mode, tmp_path):
    argv = _argv(tmp_path)
    argv[0] = mode
    if mode == "worker":
        argv = [item for item in argv]
        for flag in ("--ledger", "--run-dir"):
            index = argv.index(flag)
            del argv[index:index + 2]
        argv += ["--out-dir", str(tmp_path / "out")]
    args = R.build_parser().parse_args(argv + ["--protocol", "probe-e"])
    assert args.protocol == "probe-e"


def test_a_sealed_worker_argv_is_unchanged(tmp_path):
    """Behavioural. Stage 1's executed command must not move: the flag is only
    appended when the protocol is NOT the sealed default."""
    args = _parse(_argv(tmp_path))
    plan = R.resolve_plan(args)
    argv = R._worker_argv(args, plan)
    assert "--protocol" not in argv
    assert argv[2] == "worker"


def test_an_amended_worker_argv_carries_the_protocol(tmp_path):
    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                                     "--protocol": "probe-e"}))
    plan = R.resolve_plan(args, protocol=E)
    argv = R._worker_argv(args, plan)
    assert argv[argv.index("--protocol") + 1] == "probe-e"


# ==========================================================================
# Unit 2 -- plan resolution keeps the two protocols apart
# ==========================================================================

def test_the_amendment_arm_is_refused_under_the_sealed_protocol(tmp_path):
    """Behavioural. This is the guarantee that a probe-E arm can never be
    charged to a stage-1 cell: the refusal precedes every reservation."""
    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE}))
    with pytest.raises(R.RunnerRefusal, match="unknown recipe"):
        R.resolve_plan(args)


def test_a_sealed_arm_is_refused_under_the_amendment(tmp_path):
    args = _parse(_argv(tmp_path, **{"--protocol": "probe-e"}))
    with pytest.raises(R.RunnerRefusal, match="sealed"):
        R.resolve_plan(args, protocol=E)


def test_the_authorised_cell_resolves(tmp_path):
    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                                     "--protocol": "probe-e"}))
    plan = R.resolve_plan(args, protocol=E)
    assert (plan.stage, plan.recipe, plan.seed) == ("screen", E.PRIMARY_RECIPE, 1001)
    assert (plan.start_epoch, plan.end_epoch, plan.epochs) == (0, 64, 64)


def test_a_resume_segment_is_refused_under_the_amendment(tmp_path):
    """Behavioural. One 64-epoch segment was authorised; a second would double
    the charge and was never preregistered."""
    args = _parse(_argv(tmp_path, **{
        "--recipe": E.PRIMARY_RECIPE, "--protocol": "probe-e",
        "--segment-index": "1", "--parent-segment": str(tmp_path / "p.pt"),
        "--parent-result": str(tmp_path / "p.json")}))
    with pytest.raises(R.RunnerRefusal, match="one 64-epoch segment"):
        R.resolve_plan(args, protocol=E)


def test_a_second_seed_is_refused_under_the_amendment(tmp_path):
    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                                     "--protocol": "probe-e", "--seed": "1002"}))
    with pytest.raises(R.RunnerRefusal, match="authorised seed"):
        R.resolve_plan(args, protocol=E)


def test_a_shortened_epoch_budget_is_refused_under_the_amendment(tmp_path):
    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                                     "--protocol": "probe-e", "--epochs": "32"}))
    with pytest.raises(R.RunnerRefusal, match="64 epochs"):
        R.resolve_plan(args, protocol=E)


def test_the_sealed_cell_still_resolves_exactly_as_before(tmp_path):
    """Behavioural. The amendment's guards are reached through the protocol
    object, so nothing new constrains a sealed launch."""
    for seed in P.stage("screen").seeds:
        for segment in (0, 1):
            extra = {} if segment == 0 else {
                "--parent-segment": str(tmp_path / "p.pt"),
                "--parent-result": str(tmp_path / "p.json")}
            args = _parse(_argv(tmp_path, **{"--seed": str(seed),
                                             "--segment-index": str(segment)},
                                **extra))
            plan = R.resolve_plan(args)
            assert plan.seed == seed and plan.segment_index == segment


# ==========================================================================
# Unit 3 -- launch against the right ledger, and only the right ledger
# ==========================================================================

def test_a_probe_e_launch_is_refused_against_the_sealed_ledger(tmp_path):
    clock = FakeClock()
    ledger = _sealed_ledger(tmp_path, clock)
    before = ledger.path.read_bytes()
    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                                     "--protocol": "probe-e"}))
    with pytest.raises(R.RunnerRefusal):
        R.run_launch(args, ledger=ledger, spawn=Spawner(), clock=clock)
    assert ledger.path.read_bytes() == before


def test_a_sealed_launch_is_refused_against_the_amended_ledger(tmp_path):
    clock = FakeClock()
    ledger = _amended_ledger(tmp_path, clock)
    before = ledger.path.read_bytes()
    args = _parse(_argv(tmp_path, **{"--ledger": str(ledger.path)}))
    with pytest.raises(R.RunnerRefusal):
        R.run_launch(args, ledger=ledger, spawn=Spawner(), clock=clock)
    assert ledger.path.read_bytes() == before


def test_a_probe_e_launch_records_its_protocol_and_digest(tmp_path):
    clock = FakeClock()
    ledger = _amended_ledger(tmp_path, clock)
    out_dir = tmp_path / "run" / "segment_0000"

    def child(argv, kwargs):
        clock.advance(320.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps({
            "schema_version": R.RESULT_SCHEMA, "training_began": True,
            "accounting": _counters()}), encoding="utf-8")

    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                                     "--protocol": "probe-e",
                                     "--ledger": str(ledger.path)}))
    code = R.run_launch(args, ledger=ledger, spawn=Spawner(on_spawn=child),
                        clock=clock)
    assert code == R.EXIT_OK
    launch = json.loads((out_dir / "launch.json").read_text(encoding="utf-8"))
    assert launch["protocol"] == "probe-e"
    assert launch["protocol_digest"] == E.protocol_digest()
    assert launch["recipe"] == E.PRIMARY_RECIPE
    settled = [row for row in ledger.rows if row["kind"] == "settle"]
    assert len(settled) == 1
    assert settled[0]["recipe"] == E.PRIMARY_RECIPE
    assert settled[0]["charged_s"] == 320.0


def test_a_sealed_launch_still_records_the_sealed_protocol(tmp_path):
    clock = FakeClock()
    ledger = _sealed_ledger(tmp_path, clock)
    out_dir = tmp_path / "run" / "segment_0000"

    def child(argv, kwargs):
        clock.advance(300.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps({
            "schema_version": R.RESULT_SCHEMA, "training_began": True,
            "accounting": _counters()}), encoding="utf-8")

    args = _parse(_argv(tmp_path))
    assert R.run_launch(args, ledger=ledger, spawn=Spawner(on_spawn=child),
                        clock=clock) == R.EXIT_OK
    launch = json.loads((out_dir / "launch.json").read_text(encoding="utf-8"))
    assert launch["protocol"] == "sealed"
    assert launch["protocol_digest"] == P.protocol_digest()


def test_a_probe_e_launch_needs_a_carried_ledger_to_be_created(tmp_path):
    """Behavioural. ``--create-ledger`` alone cannot open a fresh amended ledger:
    without a carried opening balance it would reset every cap."""
    argv = _argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                              "--protocol": "probe-e"}) + ["--create-ledger"]
    args = _parse(argv)
    with pytest.raises(R.RunnerRefusal, match="carried forward"):
        R.run_launch(args, spawn=Spawner(), clock=FakeClock(),
                     probe=lambda: B.unobserved_contention())
    assert not (tmp_path / "budget_ledger.jsonl").exists()


def test_carry_forward_from_is_wired_to_the_ledger_creation(tmp_path):
    source = _sealed_ledger(tmp_path, FakeClock(), name="sealed.jsonl")
    argv = _argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                              "--protocol": "probe-e",
                              "--ledger": str(tmp_path / "probe_e.jsonl"),
                              "--carry-forward-from": str(source.path)})
    argv += ["--create-ledger"]
    args = _parse(argv)
    clock = FakeClock()
    out_dir = tmp_path / "run" / "segment_0000"

    def child(argv_, kwargs):
        clock.advance(320.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps({
            "schema_version": R.RESULT_SCHEMA, "training_began": True,
            "accounting": _counters()}), encoding="utf-8")

    assert R.run_launch(args, spawn=Spawner(on_spawn=child), clock=clock,
                        probe=lambda: B.unobserved_contention()) == R.EXIT_OK
    created = B.BudgetLedger.open(tmp_path / "probe_e.jsonl", protocol=E)
    assert created.protocol_digest == E.protocol_digest()
    assert created.carried_forward["source_protocol_digest"] == P.protocol_digest()
    assert source.path.read_bytes() == B.BudgetLedger.open(source.path).path.read_bytes()


# ==========================================================================
# Unit 4 -- the override reaches the configuration, and is verified there
# ==========================================================================

def test_the_sealed_effective_config_overrides_exactly_the_nine_keys(tmp_path):
    """Behavioural. The refactor must not change what a sealed arm builds."""
    cfg = R.effective_config(protocol=P, stage_key="screen", recipe_name="g990_e010",
                             seed=1001, device="cuda:0", logdir=str(tmp_path))
    assert cfg["params"]["general"] == {"seed": 1001, "device": "cuda:0",
                                        "logdir": str(tmp_path)}
    assert cfg["params"]["env"]["num_actors"] == P.NUM_WORLDS
    assert cfg["params"]["config"]["max_epochs"] == 128
    assert cfg["params"]["config"]["steps_num"] == P.CONTROLS_PER_WORLD_PER_EPOCH
    assert cfg["params"]["config"]["save_interval"] == 0
    assert cfg["params"]["config"]["gamma"] == 0.990
    assert cfg["params"]["config"]["entropy_coef"] == 0.01
    # Untouched by a sealed arm, and therefore still the pinned values.
    assert cfg["params"]["config"]["clip_range"] == 0.2
    assert cfg["params"]["config"]["ppo_epochs"] == 5


def test_the_trust_region_override_reaches_the_effective_config(tmp_path):
    cfg = R.effective_config(protocol=E, stage_key="screen",
                             recipe_name=E.PRIMARY_RECIPE, seed=1001,
                             device="cuda:0", logdir=str(tmp_path))
    assert cfg["params"]["config"]["clip_range"] == 0.4
    assert cfg["params"]["config"]["ppo_epochs"] == 5          # still the pin
    assert cfg["params"]["config"]["gamma"] == 0.990
    assert cfg["params"]["config"]["entropy_coef"] == 0.01

    alt = R.effective_config(protocol=E, stage_key="screen",
                             recipe_name=E.DECLARED_ALTERNATIVE_RECIPE, seed=1001,
                             device="cuda:0", logdir=str(tmp_path))
    assert alt["params"]["config"]["ppo_epochs"] == 1
    assert alt["params"]["config"]["clip_range"] == 0.2        # still the pin


def test_nothing_but_the_trust_region_keys_moves_between_the_two_configs(tmp_path):
    sealed = R.effective_config(protocol=P, stage_key="screen",
                                recipe_name="g990_e010", seed=1001,
                                device="cuda:0", logdir=str(tmp_path))
    amended = R.effective_config(protocol=E, stage_key="screen",
                                 recipe_name=E.PRIMARY_RECIPE, seed=1001,
                                 device="cuda:0", logdir=str(tmp_path))
    differing = {key for key in set(sealed["params"]["config"])
                 | set(amended["params"]["config"])
                 if sealed["params"]["config"].get(key)
                 != amended["params"]["config"].get(key)}
    assert differing == {"clip_range"}
    assert sealed["params"]["env"] == amended["params"]["env"]
    assert sealed["params"]["network"] == amended["params"]["network"]


def test_the_built_arm_is_verified_against_the_requested_override():
    """Behavioural. Positive verification that the configuration reached the
    algorithm, in the same spirit as the existing worlds/steps check."""

    class Algo:
        def __init__(self, **values):
            self.num_envs = P.NUM_WORLDS
            self.steps_num = P.CONTROLS_PER_WORLD_PER_EPOCH
            self.clip_range = 0.4
            self.ppo_epochs = 5
            for key, value in values.items():
                setattr(self, key, value)

    arm = E.recipe(E.PRIMARY_RECIPE)
    R.assert_built_arm(Algo(), protocol=E, arm=arm)
    with pytest.raises(R.RunnerRefusal, match="clip_range"):
        R.assert_built_arm(Algo(clip_range=0.2), protocol=E, arm=arm)
    with pytest.raises(R.RunnerRefusal, match="worlds"):
        R.assert_built_arm(Algo(num_envs=8), protocol=E, arm=arm)
    sealed_arm = P.recipe("g990_e010")
    R.assert_built_arm(Algo(), protocol=P, arm=sealed_arm)


# ==========================================================================
# Unit 5 -- the freeze records the amendment
# ==========================================================================

def test_the_freeze_schema_declares_the_amended_shape():
    assert R.FREEZE_SCHEMA == "myoleg26-ppo-v2-freeze-v2"


def test_the_committed_manifest_keeps_the_sealed_protocol_untouched():
    frozen = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    assert frozen["protocol"]["digest"] == P.protocol_digest()
    assert frozen["protocol"]["digest"] == (
        "4c01fdc94117bd190cd44c6b6a5f411a5c7cfc7b7d3e0462a344dda43baf2169")


def test_the_committed_manifest_records_the_amendment_and_its_parent():
    frozen = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    amendments = frozen["amendments"]
    assert set(amendments) == {E.AMENDMENT_ID}
    entry = amendments[E.AMENDMENT_ID]
    assert entry["digest"] == E.protocol_digest()
    assert entry["parent_protocol_digest"] == P.protocol_digest()
    assert entry["digest"] != frozen["protocol"]["digest"]


def test_the_committed_manifest_pins_the_amendment_module():
    frozen = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    relative = "msk_warp/analysis/ppo_v2_protocol_e.py"
    assert relative in frozen["files"]
    assert relative in frozen["files_eol"]
    assert frozen["files_git"]["head_status"][relative] == "in_head"
    assert frozen["schema_version"] == R.FREEZE_SCHEMA


def test_the_v1_manifest_is_still_untouched():
    import hashlib

    raw = (ROOT / "msk_warp/configs/experiments/myoleg26_baseline_v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "9e6d19e586d5ec1560e6dc2a95cadfd3295966d871315c67f6ab99f341ce04e1")


# ==========================================================================
# Unit 6 -- the sealed summarizer cannot absorb a probe-E run
# ==========================================================================

def _run_tree(root, *, recipe, seed=1001, stage="screen"):
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
        "accounting": dict(_counters(), completed_control_transitions=524288),
    }), encoding="utf-8")
    return run_dir


def _manifest(tmp_path) -> Path:
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"schema_version": R.FREEZE_SCHEMA,
                                "files": {}}), encoding="utf-8")
    return path


def test_a_foreign_recipe_run_is_named_and_forces_a_review_stop(tmp_path):
    """Behavioural. A probe-E run dropped into a stage-1 run root is reported by
    name and stops the summary for review; it is never silently dropped."""
    root = tmp_path / "runs"
    root.mkdir()
    _run_tree(root, recipe="g990_e010")
    _run_tree(root, recipe=E.PRIMARY_RECIPE)
    report = Z.summarize(root, _manifest(tmp_path))
    foreign = report["campaign"]["foreign_recipe_runs"]
    assert [entry["recipe"] for entry in foreign] == [E.PRIMARY_RECIPE]
    assert "foreign_recipe_runs" in report["review_stop_reasons"]
    assert report["review_stop"] is True
    assert E.PRIMARY_RECIPE not in report["recipes"].get("screen", {})


def test_a_foreign_recipe_cost_is_not_added_to_the_campaign_total(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    _run_tree(root, recipe="g990_e010")
    _run_tree(root, recipe=E.PRIMARY_RECIPE)
    report = Z.summarize(root, _manifest(tmp_path))
    assert report["campaign_cost_totals"]["completed_control_transitions"] == 524288
    assert report["foreign_cost_totals"]["completed_control_transitions"] == 524288


def test_a_sealed_only_run_root_reports_no_foreign_recipe(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    _run_tree(root, recipe="g990_e010")
    report = Z.summarize(root, _manifest(tmp_path))
    assert report["campaign"]["foreign_recipe_runs"] == []
    assert report["foreign_cost_totals"]["completed_control_transitions"] == 0
    assert "foreign_recipe_runs" not in report["review_stop_reasons"]


def test_the_summarizer_schema_declares_the_amended_shape():
    assert Z.SCHEMA_VERSION == "myoleg26-ppo-v2-summary-v2"
    assert Z.FOREIGN_RECIPE_NOTE.strip()
# ==========================================================================
# Unit 7 -- fix round 1: a moved parent refuses the launch, not just the reserve
#
# Recorded honestly: this is an INTEGRATION GUARD, not a RED. It passed on its
# first run, because ``ParentLedgerMovedError`` is a ``BudgetError`` and
# ``run_launch`` already converts that class into a clean refusal. The point of
# keeping it is that the conversion, and the "no durable row" property, are now
# asserted at the launch layer too.
# ==========================================================================

def test_a_moved_parent_refuses_the_probe_e_launch_and_writes_nothing(tmp_path):
    clock = FakeClock()
    source = _sealed_ledger(tmp_path, FakeClock(), name="sealed.jsonl")
    ledger = B.BudgetLedger.create(
        tmp_path / "probe_e_ledger.jsonl", protocol_digest=E.protocol_digest(),
        protocol=E, carried=B.carry_forward(source.path),
        provenance=E.ledger_provenance(), clock=clock)
    before = ledger.path.read_bytes()

    # The parent settles one more segment AFTER the snapshot was taken.
    reservation = source.reserve(
        stage="screen", recipe="g990_e010", seed=1001, segment_index=0,
        start_epoch=0, end_epoch=64, reserved_bound_s=540.0,
        shutdown_allowance_s=60.0)
    source.settle(reservation, actual_wall_s=400.0, returncode=0,
                  counters=B.SegmentCounters(**_counters()))
    parent_bytes = source.path.read_bytes()

    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                                     "--protocol": "probe-e",
                                     "--ledger": str(ledger.path)}))
    spawner = Spawner()
    with pytest.raises(R.RunnerRefusal, match="has MOVED"):
        R.run_launch(args, ledger=ledger, spawn=spawner, clock=clock)

    assert spawner.calls == []                      # no child was started
    assert ledger.path.read_bytes() == before       # no durable row
    assert source.path.read_bytes() == parent_bytes  # the parent was only read
    assert not (tmp_path / "run" / "segment_0000").exists()


def test_an_unmoved_parent_lets_the_probe_e_launch_through(tmp_path):
    """The negative control for the test above: same setup, parent untouched."""
    clock = FakeClock()
    source = _sealed_ledger(tmp_path, FakeClock(), name="sealed.jsonl")
    ledger = B.BudgetLedger.create(
        tmp_path / "probe_e_ledger.jsonl", protocol_digest=E.protocol_digest(),
        protocol=E, carried=B.carry_forward(source.path),
        provenance=E.ledger_provenance(), clock=clock)
    parent_bytes = source.path.read_bytes()
    out_dir = tmp_path / "run" / "segment_0000"

    def child(argv, kwargs):
        clock.advance(320.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps({
            "schema_version": R.RESULT_SCHEMA, "training_began": True,
            "accounting": _counters()}), encoding="utf-8")

    args = _parse(_argv(tmp_path, **{"--recipe": E.PRIMARY_RECIPE,
                                     "--protocol": "probe-e",
                                     "--ledger": str(ledger.path)}))
    assert R.run_launch(args, ledger=ledger, spawn=Spawner(on_spawn=child),
                        clock=clock) == R.EXIT_OK
    assert source.path.read_bytes() == parent_bytes
