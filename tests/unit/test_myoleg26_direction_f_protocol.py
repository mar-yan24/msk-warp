"""Unit tests for the Direction F protocol: the third level of the v2 lineage.

CPU only. No simulator, no CUDA, no Warp, no torch, no subprocess, no training
and no launch. Nothing here spends the 28,800 s training budget or the 1,800 s
diagnostic budget; test-suite time is accounted for separately.

What this file is for
---------------------
Direction F descends from probe E, which descends from the sealed v2 protocol.
The chain is sealed ``4c01fdc9...2169``, then probe E ``7c789255...b68e``, then F.
F spends the sealed ``refine`` cap as a **funding line only**. It is not a stage,
it promotes nothing, and nothing it produces is a stage-2 result. Its own seeds
(4001, 4002), its own reset block (14000-14015) and its own 512-epoch cap keep
it apart from sealed stage 2's unseen material.

The launch is locked twice over. The first lock is ``AUTHORISED = False``, a
source constant inside the digest. The second is the user-written authorisation
record. The snapshot and the digest never read the preregistration file or the
record.

The numbers in the design section 3.7 list map onto the test names as
``test_NN_...``. Supplementary checks, which that list does not number, are
named ``test_sNN_...``.

Isolation
---------
* Tests that touch the lock build a three-level **temporary** chain. They patch
  ``E.PARENT_LEDGER`` (it is inside E's digest) and set
  ``F.PARENT_PROTOCOL_DIGEST = E.protocol_digest()``. They also point
  ``F.PARENT_LEDGER``, ``F.LEDGER``, ``F.RUN_ROOT``, ``F.AUTHORISATION_RECORD``,
  ``F.PREREGISTRATION_PATH``, ``F.PREREGISTRATION_SHA256`` and ``F.AUTHORISED``
  at temporary values.
* An autouse guard refuses any ``open`` of a path under the real ``logs/`` or
  ``docs/`` directories, so no test here can read the real ledgers, the real
  preregistration or a real authorisation record.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import inspect
import io
import json
import math
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

from msk_warp.analysis import ppo_v2_protocol as P
from msk_warp.analysis import ppo_v2_protocol_e as E
from msk_warp.analysis import ppo_v2_protocol_f as F

ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = ROOT / "msk_warp/configs/experiments/myoleg26_ppo_v2.yaml"

#: Transcribed digests of the two ancestors. The sealed one comes from the
#: stage-1 ledger header and the committed v2 manifest; probe E's comes from the
#: E ledger header and the regenerated v2 manifest.
SEALED_DIGEST = "4c01fdc94117bd190cd44c6b6a5f411a5c7cfc7b7d3e0462a344dda43baf2169"
E_DIGEST = "7c7892554cd79fb48832b1c431652f71776ef7e05b5359d4ccfc60f8c39ab68e"

#: Transcribed from the preregistration (r2): its LF-normalised sha256 and path.
PREREG_SHA256 = "acb777df3a7e40b42ae4571f7b715b56e069ac1e5bd3e8d6053f756d55095679"
PREREG_PATH = (
    "docs/research/2026-09-23-session/orchestration/"
    "direction-f-preregistration.md")

F_RECIPE = "g990_e010_f512"
F_SEEDS = (4001, 4002)
F_BLOCK = tuple(range(14000, 14016))
RECORD_PATH = "logs/myoleg26_ppo_v2_direction_f/user_authorisation.json"
RECORD_KEYS = frozenset({
    "schema_version", "authorised", "protocol", "amendment_id",
    "protocol_digest", "preregistration_sha256", "funding_cap", "budget_s",
    "seeds", "authorised_by", "decided_utc", "statement"})


def _pinned_config() -> dict:
    return yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))["params"]["config"]


def _source() -> str:
    return Path(F.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Isolation guard: no test may read the real logs/ or docs/
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

    def guarded_open(file, *args, **kwargs):
        if _under_guarded_root(file):
            raise AssertionError(
                f"a Direction F unit test tried to open the real {file!r}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    yield


def test_s00_the_isolation_guard_has_teeth():
    """Anti-vacuity for the guard itself: a real logs/ or docs/ path is refused
    and an ordinary tracked file is not."""
    with pytest.raises(AssertionError, match="real"):
        open(ROOT / "logs" / "does-not-matter.txt", "rb")
    with pytest.raises(AssertionError, match="real"):
        (ROOT / "docs" / "HANDOFF.md").read_bytes()
    assert V2_CONFIG.read_bytes()


# --------------------------------------------------------------------------
# The temporary three-level chain
# --------------------------------------------------------------------------

class Chain:
    """A temporary sealed -> E -> F chain with a temporary prereg and record."""

    PREREG_BYTES = (b"# Direction F temporary preregistration\n\n"
                    b"line two of the frozen text\nline three\n")

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.run_root = tmp_path / "f_run_root"
        self.run_root.mkdir()
        self.record_path = self.run_root / "user_authorisation.json"
        prereg_dir = tmp_path / "prereg"
        prereg_dir.mkdir()
        self.prereg_path = prereg_dir / "direction-f-preregistration.md"
        self.prereg_path.write_bytes(self.PREREG_BYTES)
        self.prereg_sha256 = hashlib.sha256(self.PREREG_BYTES).hexdigest()

        monkeypatch.setattr(E, "PARENT_LEDGER",
                            str(tmp_path / "sealed" / "budget_ledger.jsonl"))
        monkeypatch.setattr(F, "PARENT_PROTOCOL_DIGEST", E.protocol_digest())
        monkeypatch.setattr(F, "PARENT_LEDGER",
                            str(tmp_path / "probe_e" / "budget_ledger.jsonl"))
        monkeypatch.setattr(F, "LEDGER", str(self.run_root / "budget_ledger.jsonl"))
        monkeypatch.setattr(F, "RUN_ROOT", str(self.run_root))
        monkeypatch.setattr(F, "AUTHORISATION_RECORD", str(self.record_path))
        monkeypatch.setattr(F, "PREREGISTRATION_PATH", str(self.prereg_path))
        monkeypatch.setattr(F, "PREREGISTRATION_SHA256", self.prereg_sha256)
        monkeypatch.setattr(F, "AUTHORISED", False)

    def authorise(self, value=True) -> None:
        self.monkeypatch.setattr(F, "AUTHORISED", value)

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

    def write_record(self, record=None, *, raw: bytes | None = None) -> None:
        if raw is None:
            record = self.valid_record() if record is None else record
            raw = json.dumps(record).encode("utf-8")
        self.record_path.write_bytes(raw)


@pytest.fixture
def chain(tmp_path, monkeypatch) -> Chain:
    return Chain(tmp_path, monkeypatch)


def _replace_stage(monkeypatch, **changes) -> None:
    monkeypatch.setattr(F, "STAGE", dataclasses.replace(F.STAGE, **changes))


# --------------------------------------------------------------------------
# 1-8: descent, snapshot and digest
# --------------------------------------------------------------------------

def test_01_the_parent_literal_is_the_live_unpatched_probe_e_digest():
    """Transcription + behavioural. The literal is E's digest, and the live,
    unpatched E still hashes to it."""
    assert F.PARENT_PROTOCOL_DIGEST == E_DIGEST
    assert E.protocol_digest() == E_DIGEST
    assert F.PARENT_MODULE == "msk_warp.analysis.ppo_v2_protocol_e"
    assert F.PARENT_SCHEMA_VERSION == E.SCHEMA_VERSION
    assert F.SCHEMA_VERSION == "myoleg26-ppo-v2f-protocol-v1"
    assert F.AMENDMENT_ID == "direction-f-horizon-v1"
    assert F.PROTOCOL_NAME == "direction-f"
    F.assert_parent_unchanged()


def test_02_the_grandparent_digest_is_the_sealed_digest(monkeypatch):
    assert F.GRANDPARENT_PROTOCOL_DIGEST == SEALED_DIGEST
    assert E.PARENT_PROTOCOL_DIGEST == SEALED_DIGEST
    assert P.protocol_digest() == SEALED_DIGEST
    amendment = F.amendment_snapshot()
    assert amendment["grandparent_protocol_digest"] == SEALED_DIGEST
    assert amendment["parent_protocol_digest"] == E_DIGEST
    monkeypatch.setattr(F, "GRANDPARENT_PROTOCOL_DIGEST", "0" * 64)
    with pytest.raises(P.ProtocolError, match="grandparent"):
        F.protocol_snapshot()


def test_03_the_snapshot_embeds_the_probe_e_snapshot_verbatim():
    snapshot = F.protocol_snapshot()
    assert snapshot["inherited_protocol"] == E.protocol_snapshot()
    assert snapshot["inherited_protocol"]["inherited_protocol"] == P.protocol_snapshot()
    assert snapshot["parent_protocol_digest"] == E_DIGEST
    assert snapshot["schema_version"] == F.SCHEMA_VERSION
    assert snapshot["schema_version"] not in (E.SCHEMA_VERSION, P.SCHEMA_VERSION)
    assert snapshot["amendment"] == F.amendment_snapshot()


def test_04_re_parenting_is_refused(monkeypatch):
    monkeypatch.setattr(F, "PARENT_PROTOCOL_DIGEST", "0" * 64)
    with pytest.raises(P.ProtocolError, match="direction-f-horizon-v1 descends from"):
        F.assert_parent_unchanged()
    with pytest.raises(P.ProtocolError, match="descends from"):
        F.protocol_snapshot()
    with pytest.raises(P.ProtocolError, match="descends from"):
        F.protocol_digest()


def test_04b_a_moved_probe_e_ledger_path_moves_e_and_f_refuses(tmp_path, monkeypatch):
    """Anti-vacuity. E's parent ledger path is inside E's digest, so patching it
    really moves E, and F then refuses until its parent literal follows."""
    monkeypatch.setattr(E, "PARENT_LEDGER", str(tmp_path / "decoy.jsonl"))
    moved = E.protocol_digest()
    assert moved != E_DIGEST
    with pytest.raises(P.ProtocolError, match="descends from"):
        F.protocol_snapshot()
    monkeypatch.setattr(F, "PARENT_PROTOCOL_DIGEST", moved)
    assert F.protocol_snapshot()["parent_protocol_digest"] == moved


def test_05_a_moved_sealed_digest_is_refused_through_probe_e(monkeypatch):
    monkeypatch.setattr(P, "protocol_digest", lambda: "0" * 64)
    with pytest.raises(P.ProtocolError, match=re.escape(E.AMENDMENT_ID)):
        F.protocol_snapshot()
    with pytest.raises(P.ProtocolError):
        F.protocol_digest()


def test_06_the_f_digest_differs_from_the_sealed_and_probe_e_digests():
    digest = F.protocol_digest()
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest not in (SEALED_DIGEST, E_DIGEST)


def test_07_the_snapshot_is_canonical_and_stable_with_no_nan():
    assert F.canonical_json is P.canonical_json
    first = P.canonical_json(F.protocol_snapshot())
    second = P.canonical_json(F.protocol_snapshot())
    assert first == second
    assert b"NaN" not in first and b"Infinity" not in first
    assert json.loads(first) == F.protocol_snapshot()
    assert F.protocol_digest() == hashlib.sha256(first).hexdigest()
    assert first != P.canonical_json(E.protocol_snapshot())


def test_08_the_source_assigns_nothing_to_probe_e_or_the_sealed_module():
    tree = ast.parse(_source())
    module_aliases = {"_parent", "_sealed", "P", "E"}
    offending = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            for sub in ast.walk(target):
                if (isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id in module_aliases):
                    offending.append(ast.unparse(sub))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"setattr", "delattr"}):
            offending.append(ast.unparse(node))
    assert offending == []
    assert "__dict__" not in _source()
    assert "globals()" not in _source()


# --------------------------------------------------------------------------
# 9-16: the funding line and the cell
# --------------------------------------------------------------------------

def test_09_the_stage_record_is_a_refine_funding_line():
    spec = F.stage("refine")
    sealed_refine = P.stage("refine")
    assert spec is F.STAGE
    assert isinstance(spec, P.Stage)
    assert spec.key == "refine" == F.FUNDING_CAP
    assert spec.index == sealed_refine.index == 2
    assert spec.epoch_cap_per_seed == 512 == F.EPOCH_CAP_PER_SEED
    assert spec.epoch_cap_per_seed != sealed_refine.epoch_cap_per_seed
    assert spec.seeds == F_SEEDS
    assert spec.seeds != sealed_refine.seeds
    assert spec.selection_reset_block == F_BLOCK
    assert spec.selection_reset_block != sealed_refine.selection_reset_block
    assert spec.wall_cap_s == F.FUNDING_CAP_S
    assert spec.per_seed_safety_cap_s == 4800
    assert spec.max_recipes == 1
    assert spec.max_promoted == 0
    assert F.selection_reset_block("refine") == F_BLOCK
    assert dict(F.STAGES) == {"refine": F.STAGE}
    assert F.STAGE_ORDER == ("refine",)
    stage_snapshot = F.amendment_snapshot()["stage"]
    assert stage_snapshot == {
        "key": "refine", "index": 2, "epoch_cap_per_seed": 512,
        "seeds": [4001, 4002], "wall_cap_s": F.FUNDING_CAP_S,
        "per_seed_safety_cap_s": 4800, "max_recipes": 1, "max_promoted": 0,
        "selection_reset_block": list(F_BLOCK)}


@pytest.mark.parametrize("key", ["screen", "confirm"])
def test_10_screen_and_confirm_are_refused_by_name(key):
    with pytest.raises(F.AmendmentError, match=repr(key)):
        F.stage(key)
    with pytest.raises(F.AmendmentError, match=repr(key)):
        F.selection_reset_block(key)


@pytest.mark.parametrize("key", ["stage2", "", None, 2, "REFINE"])
def test_10b_unknown_stage_keys_are_refused(key):
    with pytest.raises(P.ProtocolError):
        F.stage(key)


def test_11_the_funding_cap_sits_inside_the_preregistered_window(monkeypatch):
    assert F.FUNDING_CAP_S == 7200
    assert F.MIN_FUNDING_CAP_S == 6196
    assert F.MAX_FUNDING_CAP_S == 7200
    assert F.MIN_FUNDING_CAP_S == math.ceil(15 * 373.00624229999084 + 600)
    assert F.MAX_FUNDING_CAP_S == P.stage("refine").wall_cap_s
    assert F.MIN_FUNDING_CAP_S <= F.FUNDING_CAP_S <= F.MAX_FUNDING_CAP_S
    F.assert_funding_line()
    funding = F.amendment_snapshot()["funding"]
    assert funding == {"cap": "refine", "cap_s": 7200, "min_cap_s": 6196,
                       "max_cap_s": 7200, "is_stage": False, "promotes": False}


@pytest.mark.parametrize("cap, match", [
    (6195, r"window.*6196"),
    (7201, r"window.*7200"),
    (0, r"window"),
])
def test_11b_a_cap_outside_the_window_refuses_the_snapshot(monkeypatch, cap, match):
    monkeypatch.setattr(F, "FUNDING_CAP_S", cap)
    _replace_stage(monkeypatch, wall_cap_s=cap)
    with pytest.raises(F.AmendmentError, match=match):
        F.assert_funding_line()
    with pytest.raises(F.AmendmentError, match=match):
        F.protocol_snapshot()
    with pytest.raises(F.AmendmentError, match=match):
        F.stage("refine")


@pytest.mark.parametrize("cap", [True, 7000.5, 7200.0, "7200", None])
def test_11c_a_cap_that_is_not_an_int_is_refused(monkeypatch, cap):
    monkeypatch.setattr(F, "FUNDING_CAP_S", cap)
    with pytest.raises(F.AmendmentError, match="FUNDING_CAP_S must be an int"):
        F.assert_funding_line()


@pytest.mark.parametrize("cap", [6196, 7000, 7200])
def test_11d_a_cap_inside_the_window_is_accepted_and_moves_the_digest(
        monkeypatch, cap):
    baseline = F.protocol_digest()
    monkeypatch.setattr(F, "FUNDING_CAP_S", cap)
    _replace_stage(monkeypatch, wall_cap_s=cap)
    F.assert_funding_line()
    assert (F.protocol_digest() == baseline) is (cap == 7200)


def test_11e_the_stage_record_must_agree_with_the_funding_cap(monkeypatch):
    monkeypatch.setattr(F, "FUNDING_CAP_S", 7000)
    with pytest.raises(F.AmendmentError, match="STAGE"):
        F.assert_funding_line()


def test_11f_the_cap_never_exceeds_the_live_sealed_refine_cap(monkeypatch):
    """The sealed refine cap is read at call time, not copied at import."""
    real_stage = P.stage
    shrunk = dataclasses.replace(real_stage("refine"), wall_cap_s=7000)
    monkeypatch.setattr(P, "stage",
                        lambda key: shrunk if key == "refine" else real_stage(key))
    with pytest.raises(F.AmendmentError, match="sealed refine"):
        F.assert_funding_line()


def test_12_the_per_seed_cap_is_eight_full_segment_reservations():
    assert F.PER_SEED_SAFETY_CAP_S == 4800 == F.STAGE.per_seed_safety_cap_s
    assert F.PER_SEED_SAFETY_CAP_S == 8 * P.SEGMENT_CALL_BOUND_S
    assert F.PER_SEED_SAFETY_CAP_S == (
        len(P.segment_rounds(F.EPOCH_CAP_PER_SEED)) * P.SEGMENT_CALL_BOUND_S)


def test_13_f_is_not_a_stage_promotes_nothing_and_is_not_a_stage_2_result():
    assert F.IS_STAGE is False
    assert F.PROMOTES is False
    assert F.STAGE.max_promoted == 0
    note = F.NOT_A_STAGE_2_RESULT
    assert isinstance(note, str) and note.strip()
    assert "stage-2" in note.lower() or "stage 2" in note.lower()
    amendment = F.amendment_snapshot()
    assert amendment["not_a_stage_2_result"] == note
    assert amendment["funding"]["is_stage"] is False
    assert amendment["funding"]["promotes"] is False
    assert F.ledger_provenance()["not_a_stage_2_result"] == note
    source = _source()
    for forbidden in ("REFINE_MEDIAN_DURATION_S", "refine_eligible", ".promote(",
                      "REFINE_FALLBACK"):
        assert forbidden not in source


def test_14_segment_rounds_and_evaluation_epochs_at_512():
    rounds = F.segment_rounds(F.STAGE.epoch_cap_per_seed)
    assert [(r.index, r.start_epoch, r.end_epoch) for r in rounds] == [
        (k, 64 * k, 64 * (k + 1)) for k in range(8)]
    assert len(rounds) == F.MAX_SEGMENT_INDEX + 1 == 8
    assert all(r.epochs == F.SEGMENT_EPOCHS == 64 for r in rounds)
    epochs = F.evaluation_epochs(F.STAGE.epoch_cap_per_seed)
    assert epochs == (0, 64, 128, 192, 256, 320, 384, 448, 512)
    assert F.amendment_snapshot()["cell"]["evaluation_epochs"] == list(epochs)


def test_15_run_order_interleaves_sixteen_cells_round_outer_seed_inner():
    order = F.run_order()
    assert order == tuple((r, s) for r in range(8) for s in (4001, 4002))
    assert len(order) == 16
    assert order[:4] == ((0, 4001), (0, 4002), (1, 4001), (1, 4002))
    assert order[-1] == (7, 4002)
    assert F.amendment_snapshot()["cell"]["run_order"] == [list(c) for c in order]


def test_16_the_cell_arithmetic_matches_the_preregistration():
    arithmetic = F.cell_arithmetic()
    assert arithmetic == {
        "epochs_per_seed": 512,
        "seeds": 2,
        "segments_per_seed": 8,
        "segments_total": 16,
        "evaluations_per_seed": 9,
        "evaluations_total": 18,
        "train_control_transitions_per_seed": 4_194_304,
        "train_physics_steps_per_seed": 16_777_216,
        "train_control_transitions_total": 8_388_608,
        "train_physics_steps_total": 33_554_432,
    }
    assert arithmetic["train_control_transitions_per_seed"] == (
        P.train_control_transitions(512))
    assert arithmetic["train_physics_steps_per_seed"] == P.train_physics_steps(512)
    assert F.amendment_snapshot()["cell"]["arithmetic"] == arithmetic


# --------------------------------------------------------------------------
# 17-20: the recipe
# --------------------------------------------------------------------------

def test_17_the_recipe_is_the_pinned_pair_with_no_overrides():
    arm = F.recipe(F_RECIPE)
    assert arm is F.RECIPES[F_RECIPE]
    assert arm.name == F_RECIPE
    assert arm.gamma == 0.99
    assert arm.entropy_coef == 0.01
    assert arm.overrides == {}
    assert arm.epochs_per_seed == 512
    assert F.TRUST_REGION_KEYS == ()
    assert F.PRIMARY_RECIPE == F_RECIPE
    assert F.CANONICAL_RECIPE_ORDER == (F_RECIPE,)
    assert set(F.RECIPES) == {F_RECIPE}
    sealed = P.recipe("g990_e010")
    assert (arm.gamma, arm.entropy_coef) == (sealed.gamma, sealed.entropy_coef)
    recipes = F.amendment_snapshot()["recipes"]
    assert recipes[F_RECIPE]["overrides"] == {}
    assert F.amendment_snapshot()["trust_region_keys"] == []


def test_18_recipe_names_are_disjoint_from_sealed_and_probe_e():
    assert set(F.RECIPES).isdisjoint(P.RECIPES)
    assert set(F.RECIPES).isdisjoint(E.RECIPES)
    F.assert_recipe_names_disjoint()


@pytest.mark.parametrize("colliding", ["g990_e010", "g990_e010_c040"])
def test_18b_the_disjointness_guard_has_teeth(monkeypatch, colliding):
    recipes = dict(F.RECIPES)
    recipes[colliding] = F.RECIPES[F_RECIPE]
    monkeypatch.setattr(F, "RECIPES", recipes)
    with pytest.raises(P.ProtocolError, match=colliding):
        F.assert_recipe_names_disjoint()
    with pytest.raises(P.ProtocolError, match=colliding):
        F.protocol_snapshot()


@pytest.mark.parametrize("name", sorted(P.RECIPES))
def test_19_every_sealed_arm_is_refused_naming_its_owner(name):
    with pytest.raises(F.AmendmentError, match="sealed stage-1 arm") as info:
        F.recipe(name)
    assert re.search(r"msk_warp\.analysis\.ppo_v2_protocol(?!_e)", str(info.value))
    assert E.AMENDMENT_ID not in str(info.value)


@pytest.mark.parametrize("name", sorted(E.RECIPES))
def test_19b_every_probe_e_arm_is_refused_naming_its_owner(name):
    with pytest.raises(F.AmendmentError, match=re.escape(E.AMENDMENT_ID)) as info:
        F.recipe(name)
    assert "msk_warp.analysis.ppo_v2_protocol_e" in str(info.value)


def test_19c_sealed_and_probe_e_refuse_the_f_arm_and_unknown_names_are_refused():
    with pytest.raises(P.ProtocolError):
        P.recipe(F_RECIPE)
    with pytest.raises(P.ProtocolError):
        E.recipe(F_RECIPE)
    with pytest.raises(F.AmendmentError, match=F_RECIPE):
        F.recipe("nope")
    with pytest.raises(F.AmendmentError):
        F.recipe(None)


def test_20_the_pinned_yaml_still_carries_the_recipe_values():
    config = _pinned_config()
    arm = F.recipe(F_RECIPE)
    assert config["gamma"] == 0.99 == arm.gamma
    assert config["entropy_coef"] == 0.01 == arm.entropy_coef
    assert config["lr_schedule"] == "constant"


# --------------------------------------------------------------------------
# 21-24: seeds and reset block
# --------------------------------------------------------------------------

def test_21_the_seeds_are_preregistered_and_named():
    assert F.SEEDS == F_SEEDS == F.STAGE.seeds
    assert F.amendment_snapshot()["seed_preregistered"] is True
    assert F.ledger_provenance()["seed_preregistered"] is True
    assert "4001" in F.SEED_CHOICE_NOTE and "4002" in F.SEED_CHOICE_NOTE
    assert "preregistered" in F.SEED_CHOICE_NOTE.lower()
    assert F.amendment_snapshot()["seed_choice"] == F.SEED_CHOICE_NOTE


def test_22_the_seeds_are_disjoint_from_every_sealed_stage_and_probe_seed():
    for spec in P.STAGES.values():
        assert set(F.SEEDS).isdisjoint(spec.seeds), spec.key
    assert set(F.SEEDS).isdisjoint(E.AMENDMENT_SEEDS)


def test_23_the_block_is_new_and_disjoint_from_v2_and_v1():
    assert F.SELECTION_RESET_BLOCK == F_BLOCK
    for name, ids in P.V2_RESET_BLOCKS.items():
        assert set(F_BLOCK).isdisjoint(ids), name
    assert set(F_BLOCK).isdisjoint(P.v1_reset_ids())
    F.assert_reset_blocks_disjoint()
    assert "14000" in F.RESET_BLOCK_NOTE
    assert F.amendment_snapshot()["reset_block"] == F.RESET_BLOCK_NOTE


@pytest.mark.parametrize("block, match", [
    (tuple(range(11000, 11016)), "selection_screen"),
    (tuple(range(13008, 13024)), "selection_confirm"),
])
def test_23b_the_block_guard_refuses_an_overlap_with_a_v2_block(
        monkeypatch, block, match):
    monkeypatch.setattr(F, "SELECTION_RESET_BLOCK", block)
    _replace_stage(monkeypatch, selection_reset_block=block)
    with pytest.raises(P.ProtocolError, match=match):
        F.assert_reset_blocks_disjoint()


def test_23c_the_block_guard_refuses_a_v1_id(monkeypatch):
    block = tuple(sorted(P.v1_reset_ids())[:16])
    monkeypatch.setattr(F, "SELECTION_RESET_BLOCK", block)
    _replace_stage(monkeypatch, selection_reset_block=block)
    with pytest.raises(P.ProtocolError, match="v1"):
        F.assert_reset_blocks_disjoint()


def test_24_the_source_names_no_locked_or_refine_block():
    source = _source()
    for forbidden in ("12000", "12015", "21000", "21031", "31000", "31015",
                      "selection_refine", 'V2_RESET_BLOCKS["confirmation"]',
                      "V2_RESET_BLOCKS['confirmation']", 'V2_RESET_BLOCKS["audit"]',
                      "V2_RESET_BLOCKS['audit']"):
        assert forbidden not in source, forbidden


# --------------------------------------------------------------------------
# 25-26: cell refusals
# --------------------------------------------------------------------------

_VALID_CELL = {"stage": "refine", "recipe": F_RECIPE, "seed": 4001,
               "segment_index": 0, "epochs": 64}

_CELL_REFUSALS = [
    ("stage_screen", {"stage": "screen"}, r"'refine' funding line only"),
    ("stage_confirm", {"stage": "confirm"}, r"'refine' funding line only"),
    ("recipe_sealed", {"recipe": "g990_e010"}, r"sealed stage-1 arm"),
    ("recipe_probe_e", {"recipe": "g990_e010_c040"}, re.escape(E.AMENDMENT_ID)),
    ("recipe_unknown", {"recipe": "g990_e010_f256"}, r"unknown"),
    ("seed_1001", {"seed": 1001}, r"seed 1001"),
    ("seed_2001", {"seed": 2001}, r"seed 2001"),
    ("seed_float", {"seed": 4001.0}, r"seed 4001\.0"),
    ("seed_bool", {"seed": True}, r"seed True"),
    ("segment_minus_1", {"segment_index": -1}, r"segment -1"),
    ("segment_8", {"segment_index": 8}, r"segment 8"),
    ("segment_bool", {"segment_index": False}, r"segment False"),
    ("epochs_32", {"epochs": 32}, r"epochs 32"),
    ("epochs_none", {"epochs": None}, r"epochs None"),
    ("epochs_float", {"epochs": 64.0}, r"epochs 64\.0"),
]


@pytest.mark.parametrize("changes, match",
                         [case[1:] for case in _CELL_REFUSALS],
                         ids=[case[0] for case in _CELL_REFUSALS])
def test_25_cell_refusals_each_name_their_own_cause(changes, match):
    """Each cell clause refuses before the lock, with its own message."""
    kwargs = {**_VALID_CELL, **changes}
    with pytest.raises(F.AmendmentError, match=match) as info:
        F.assert_launchable(**kwargs)
    assert "AUTHORISED" not in str(info.value)


def test_25b_the_cell_refusal_messages_are_pairwise_distinct(chain):
    chain.authorise()
    chain.write_record()
    messages = []
    for _, changes, _ in _CELL_REFUSALS:
        with pytest.raises(F.AmendmentError) as info:
            F.assert_launchable(**{**_VALID_CELL, **changes})
        messages.append(str(info.value))
    assert len(set(messages)) == len(messages)


@pytest.mark.parametrize("segment_index, seed",
                         [(r, s) for r in range(8) for s in (4001, 4002)])
def test_26_a_valid_cell_is_still_refused_while_unauthorised(segment_index, seed):
    with pytest.raises(F.AuthorisationError, match="AUTHORISED is False"):
        F.assert_launchable(stage="refine", recipe=F_RECIPE, seed=seed,
                            segment_index=segment_index, epochs=64)


# --------------------------------------------------------------------------
# 27-29: the preregistration hash and what the digest never reads
# --------------------------------------------------------------------------

def test_27_the_prereg_hash_is_a_sha256_literal_inside_the_digest(monkeypatch):
    assert F.PREREGISTRATION_SHA256 == PREREG_SHA256
    assert F.PREREGISTRATION_PATH == PREREG_PATH
    assert re.fullmatch(r"[0-9a-f]{64}", F.PREREGISTRATION_SHA256)
    assert f'PREREGISTRATION_SHA256 = (\n    "{PREREG_SHA256}")' in _source() or (
        f'PREREGISTRATION_SHA256 = "{PREREG_SHA256}"' in _source())
    prereg = F.amendment_snapshot()["preregistration"]
    assert prereg["sha256"] == PREREG_SHA256
    assert prereg["path"] == PREREG_PATH
    assert F.ledger_provenance()["preregistration"] == {
        "path": PREREG_PATH, "sha256": PREREG_SHA256}
    baseline = F.protocol_digest()
    monkeypatch.setattr(F, "PREREGISTRATION_SHA256", "a" * 64)
    assert F.protocol_digest() != baseline


@pytest.mark.parametrize("value", ["", None, "A" * 64, "a" * 63, "a" * 65,
                                   "g" * 64, b"a" * 64, 0])
def test_28_a_missing_or_malformed_prereg_hash_refuses_the_digest(monkeypatch, value):
    monkeypatch.setattr(F, "PREREGISTRATION_SHA256", value)
    with pytest.raises(F.AmendmentError, match="preregistration"):
        F.assert_preregistered()
    with pytest.raises(F.AmendmentError, match="preregistration"):
        F.protocol_digest()


@pytest.mark.parametrize("value", ["", None, 0])
def test_28b_a_missing_prereg_path_refuses_the_digest(monkeypatch, value):
    monkeypatch.setattr(F, "PREREGISTRATION_PATH", value)
    with pytest.raises(F.AmendmentError, match="preregistration"):
        F.protocol_digest()


def _refuse_file_access(context) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("the snapshot or digest touched the filesystem")

    for target, name in ((builtins, "open"), (io, "open"), (Path, "open"),
                         (Path, "read_bytes"), (Path, "read_text"),
                         (Path, "exists"), (Path, "is_file"), (Path, "stat"),
                         (os, "stat")):
        context.setattr(target, name, refuse)


def test_29_the_real_digest_reads_no_file(monkeypatch):
    baseline = F.protocol_digest()
    with monkeypatch.context() as context:
        _refuse_file_access(context)
        digest = F.protocol_digest()
        F.amendment_snapshot()
        F.ledger_provenance()
    assert digest == baseline


def test_29b_nonexistent_prereg_and_record_still_produce_a_digest(chain, monkeypatch):
    baseline = F.protocol_digest()
    assert not chain.record_path.exists()
    chain.prereg_path.unlink()
    assert F.protocol_digest() == baseline
    chain.prereg_path.write_bytes(b"entirely different content\n")
    chain.write_record(raw=b"not even json")
    assert F.protocol_digest() == baseline
    with monkeypatch.context() as context:
        _refuse_file_access(context)
        assert F.protocol_digest() == baseline


# --------------------------------------------------------------------------
# 30-36: the two-layer lock
# --------------------------------------------------------------------------

def test_30_authorised_is_false_in_source_and_in_the_snapshot():
    source = _source()
    assert len(re.findall(r"^AUTHORISED = False$", source, re.MULTILINE)) == 1
    tree = ast.parse(source)
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        and any(isinstance(sub, ast.Name) and sub.id == "AUTHORISED"
                for target in (node.targets if isinstance(node, ast.Assign)
                               else [node.target])
                for sub in ast.walk(target))]
    assert len(assignments) == 1
    assert assignments[0] in tree.body
    assert "global AUTHORISED" not in source
    assert F.AUTHORISED is False
    assert F.amendment_snapshot()["authorised"] is False
    assert F.protocol_snapshot()["amendment"]["authorised"] is False


def test_30b_flipping_authorised_moves_the_digest(monkeypatch):
    baseline = F.protocol_digest()
    monkeypatch.setattr(F, "AUTHORISED", True)
    assert F.protocol_digest() != baseline


@pytest.mark.parametrize("value", [False, 0, 1, "True", None])
def test_31_a_valid_record_is_refused_unless_authorised_is_true(chain, value):
    chain.authorise(True)
    chain.write_record()
    F.assert_authorised()                       # positive control
    chain.authorise(value)
    with pytest.raises(F.AuthorisationError, match="AUTHORISED is "):
        F.assert_authorised()


def test_31b_the_source_lock_refuses_before_any_record_is_read(chain, monkeypatch):
    assert not chain.record_path.exists()
    with monkeypatch.context() as context:
        _refuse_file_access(context)
        with pytest.raises(F.AuthorisationError, match="AUTHORISED is False"):
            F.assert_authorised()


def test_32_authorised_with_a_valid_record_passes(chain):
    chain.authorise()
    chain.write_record()
    # The lock returns the sha256 of the very bytes it parsed and verified, so a
    # caller can name them without a second read of the record.
    assert F.assert_authorised() == hashlib.sha256(
        chain.record_path.read_bytes()).hexdigest()
    for segment_index, seed in F.run_order():
        assert F.assert_launchable(stage="refine", recipe=F_RECIPE, seed=seed,
                                   segment_index=segment_index, epochs=64) is None


def _with(**changes):
    def mutate(chain):
        record = chain.valid_record()
        record.update(changes)
        chain.write_record(record)
    return mutate


def _without(key):
    def mutate(chain):
        record = chain.valid_record()
        del record[key]
        chain.write_record(record)
    return mutate


def _raw(data: bytes):
    def mutate(chain):
        chain.write_record(raw=data)
    return mutate


def _duplicate_key(chain):
    text = json.dumps(chain.valid_record())
    chain.write_record(raw=('{"statement": "an earlier statement", '
                            + text[1:]).encode("utf-8"))


def _nan_budget(chain):
    text = json.dumps(chain.valid_record())
    text = text.replace(f'"budget_s": {F.FUNDING_CAP_S}', '"budget_s": NaN')
    assert "NaN" in text
    chain.write_record(raw=text.encode("utf-8"))


def _absent(chain):
    assert not chain.record_path.exists()


def _prereg_missing(chain):
    chain.write_record()
    chain.prereg_path.unlink()


def _prereg_drifted(chain):
    chain.write_record()
    chain.prereg_path.write_bytes(chain.PREREG_BYTES + b"an appended edit\n")


_RECORD_REFUSALS = [
    ("absent", _absent, r"does not exist"),
    ("not_json", _raw(b"{not json"), r"not valid JSON"),
    ("not_utf8", _raw(b"\xff\xfe{}"), r"not valid UTF-8"),
    ("nan", _nan_budget, r"non-finite"),
    ("not_object", _raw(b"[1, 2]"), r"JSON object"),
    ("duplicate_key", _duplicate_key, r"duplicate key 'statement'"),
    ("missing_key", _without("statement"), r"missing.*'statement'"),
    ("extra_key", _with(note="an extra key"), r"unexpected.*'note'"),
    ("schema_version", _with(schema_version="v0"), r"'schema_version'"),
    ("authorised_false", _with(authorised=False), r"'authorised'"),
    ("authorised_string", _with(authorised="true"), r"'authorised'"),
    ("authorised_one", _with(authorised=1), r"'authorised'"),
    ("protocol", _with(protocol="probe-e"), r"'protocol'"),
    ("amendment_id", _with(amendment_id=E.AMENDMENT_ID), r"'amendment_id'"),
    ("digest_drift", _with(protocol_digest="0" * 64), r"'protocol_digest'"),
    ("digest_type", _with(protocol_digest=None), r"'protocol_digest'"),
    ("prereg_sha_empty", _with(preregistration_sha256=""),
     r"'preregistration_sha256'"),
    ("prereg_sha_none", _with(preregistration_sha256=None),
     r"'preregistration_sha256'"),
    ("prereg_sha_mismatch", _with(preregistration_sha256="f" * 64),
     r"'preregistration_sha256'"),
    ("prereg_file_missing", _prereg_missing,
     r"preregistration file .*does not exist"),
    ("prereg_bytes_drifted", _prereg_drifted, r"drift"),
    ("funding_cap_screen", _with(funding_cap="screen"), r"'funding_cap'"),
    ("funding_cap_other", _with(funding_cap="direction_f"), r"'funding_cap'"),
    ("budget_s_wrong", _with(budget_s=7199), r"'budget_s'"),
    ("budget_s_string", _with(budget_s="7200"), r"'budget_s'"),
    ("budget_s_bool", _with(budget_s=True), r"'budget_s'"),
    ("seeds_one", _with(seeds=[4001]), r"'seeds'"),
    ("seeds_order", _with(seeds=[4002, 4001]), r"'seeds'"),
    ("seeds_refine", _with(seeds=[2001, 2002]), r"'seeds'"),
    ("seeds_float", _with(seeds=[4001.0, 4002]), r"'seeds'"),
    ("seeds_string", _with(seeds="4001,4002"), r"'seeds'"),
    ("authorised_by_empty", _with(authorised_by=""), r"'authorised_by'"),
    ("authorised_by_blank", _with(authorised_by="   "), r"'authorised_by'"),
    ("authorised_by_none", _with(authorised_by=None), r"'authorised_by'"),
    ("decided_utc_words", _with(decided_utc="yesterday"), r"'decided_utc'"),
    ("decided_utc_naive", _with(decided_utc="2026-09-24T12:00:00"),
     r"'decided_utc'"),
    ("decided_utc_offset", _with(decided_utc="2026-09-24T12:00:00+02:00"),
     r"'decided_utc'"),
    ("decided_utc_invalid", _with(decided_utc="2026-13-40T12:00:00Z"),
     r"'decided_utc'"),
    ("decided_utc_number", _with(decided_utc=20260924), r"'decided_utc'"),
    ("statement_empty", _with(statement=""), r"'statement'"),
    ("statement_blank", _with(statement="  \n "), r"'statement'"),
]


@pytest.mark.parametrize("mutate, match",
                         [case[1:] for case in _RECORD_REFUSALS],
                         ids=[case[0] for case in _RECORD_REFUSALS])
def test_33_record_refusals(chain, mutate, match):
    chain.authorise()
    mutate(chain)
    with pytest.raises(F.AuthorisationError, match=match):
        F.assert_authorised()
    with pytest.raises(F.AuthorisationError, match=match):
        F.assert_launchable(**_VALID_CELL)


@pytest.mark.parametrize("decided_utc", ["2026-09-24T12:00:00Z",
                                         "2026-09-24T12:00:00.123456Z",
                                         "2026-09-24T12:00:00+00:00"])
def test_33b_utc_timestamps_are_accepted(chain, decided_utc):
    chain.authorise()
    _with(decided_utc=decided_utc)(chain)
    F.assert_authorised()


def test_34_a_record_bound_to_the_unauthorised_build_is_refused_after_the_flip(chain):
    unauthorised_digest = F.protocol_digest()
    chain.write_record()                        # bound while AUTHORISED is False
    chain.authorise()
    assert F.protocol_digest() != unauthorised_digest
    with pytest.raises(F.AuthorisationError, match="'protocol_digest'"):
        F.assert_authorised()


def test_35_the_lock_takes_no_parameters_and_the_source_has_no_escape_hatch():
    signature = inspect.signature(F.assert_authorised)
    assert list(signature.parameters) == []
    lowered = _source().lower()
    for forbidden in ("os.environ", "getenv", "bypass", "force", "skip", "allow_"):
        assert forbidden not in lowered, forbidden


def test_36_a_crlf_resave_is_not_drift_but_a_content_edit_is(chain):
    chain.authorise()
    chain.write_record()
    crlf = chain.PREREG_BYTES.replace(b"\n", b"\r\n")
    assert hashlib.sha256(crlf).hexdigest() != chain.prereg_sha256
    assert F.lf_normalised_sha256(crlf) == chain.prereg_sha256
    chain.prereg_path.write_bytes(crlf)
    F.assert_authorised()
    edited = crlf.replace(b"line two", b"line 2")
    assert edited != crlf
    chain.prereg_path.write_bytes(edited)
    with pytest.raises(F.AuthorisationError, match="drift"):
        F.assert_authorised()


# --------------------------------------------------------------------------
# 37-38: hygiene and ledger separation
# --------------------------------------------------------------------------

def test_37_the_module_is_standard_library_only():
    source = _source()
    for forbidden in ("import numpy", "import yaml", "import torch", "os.system",
                      "Popen", "subprocess", "allow_unvalidated_gradients"):
        assert forbidden not in source, forbidden
    allowed_local = {"msk_warp.analysis.ppo_v2_protocol",
                     "msk_warp.analysis.ppo_v2_protocol_e"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            if node.module == "msk_warp.analysis":
                assert {f"msk_warp.analysis.{a.name}" for a in node.names} <= (
                    allowed_local)
                continue
            assert node.module.split(".")[0] in sys.stdlib_module_names, node.module


def test_38_the_ledger_paths_are_separate_from_probe_e_and_sealed():
    sealed_ledger = E.PARENT_LEDGER
    assert F.LEDGER == "logs/myoleg26_ppo_v2_direction_f/budget_ledger.jsonl"
    assert F.RUN_ROOT == "logs/myoleg26_ppo_v2_direction_f"
    assert F.PARENT_LEDGER == E.LEDGER == (
        "logs/myoleg26_ppo_v2_probe_e/budget_ledger.jsonl")
    assert F.LEDGER not in (E.LEDGER, sealed_ledger)
    assert F.RUN_ROOT not in (E.RUN_ROOT, "logs/myoleg26_ppo_v2")
    assert F.LEDGER.startswith(F.RUN_ROOT + "/")
    assert F.AUTHORISATION_RECORD == RECORD_PATH
    assert F.AUTHORISATION_RECORD.startswith(F.RUN_ROOT + "/")
    assert F.REQUIRES_CARRY_FORWARD is True
    assert F.BIND_LEDGER_PATH is True
    assert F.ROOT == P.ROOT
    ledger = F.amendment_snapshot()["ledger"]
    assert ledger["path"] == F.LEDGER
    assert ledger["run_root"] == F.RUN_ROOT
    assert ledger["separate_from"] == F.PARENT_LEDGER
    assert ledger["requires_carry_forward"] is True
    assert ledger["bind_ledger_path"] is True
    assert F.ledger_provenance()["parent_ledger"] == F.PARENT_LEDGER


# --------------------------------------------------------------------------
# Supplementary checks (not numbered in the design list)
# --------------------------------------------------------------------------

def test_s01_error_classes_keep_the_runner_refusal_path():
    assert F.ProtocolError is P.ProtocolError
    assert issubclass(F.AmendmentError, P.ProtocolError)
    assert issubclass(F.AuthorisationError, F.AmendmentError)
    assert F.AmendmentError is not E.AmendmentError


def test_s02_the_official_task_and_limits_are_inherited_by_reference():
    for name in ("TOTAL_WALL_CAP_S", "RESERVE_WALL_CAP_S", "SEGMENT_MAX_EPOCHS",
                 "SEGMENT_WORK_DEADLINE_S", "SEGMENT_CALL_BOUND_S",
                 "TOOL_TIMEOUT_MAX_S", "NUM_WORLDS", "CONTROLS_PER_WORLD_PER_EPOCH",
                 "PHYSICS_SUBSTEPS", "FORWARD_SPEED_TARGET_MPS", "IMITATION",
                 "EVALUATION_HORIZON_CONTROLS", "EVALUATION_HORIZON_SECONDS",
                 "EVALUATION_EPOCH_INTERVAL", "V2_CONFIG", "V2_RESET_BLOCKS",
                 "WALKING_GATE"):
        assert getattr(F, name) is getattr(P, name), name
    for name in ("canonical_json", "evaluation_epochs", "segment_rounds"):
        assert getattr(F, name) is getattr(P, name), name


def test_s03_the_notes_travel_in_the_snapshot():
    amendment = F.amendment_snapshot()
    assert "fewer worlds" in F.GEOMETRY_NOTE.lower()
    assert "batch" in F.GEOMETRY_NOTE.lower()
    assert amendment["geometry"] == F.GEOMETRY_NOTE
    assert amendment["funding_note"] == F.FUNDING_NOTE
    assert "refine" in F.FUNDING_NOTE
    assert amendment["base_recipe"] == F.BASE_RECIPE_NOTE
    assert amendment["held_fixed"] == F.HELD_FIXED_NOTE
    assert amendment["why"] == F.WHY
    assert amendment["descends_from"] == F.DESCENDS_FROM
    assert amendment["differs_from_parent"] == list(F.DIFFERS_FROM_PARENT)
    assert len(F.DIFFERS_FROM_PARENT) >= 3
    for note in (F.WHY, F.DESCENDS_FROM, F.FUNDING_NOTE, F.NOT_A_STAGE_2_RESULT,
                 F.SEED_CHOICE_NOTE, F.RESET_BLOCK_NOTE, F.GEOMETRY_NOTE,
                 F.BASE_RECIPE_NOTE, F.HELD_FIXED_NOTE):
        assert isinstance(note, str) and note.strip()


def test_s04_the_preregistered_bars_and_labels_are_not_restated():
    source = _source()
    for forbidden in ("1.2065", "0.0434", "0.0456", "0.9944", "F-DOMINATES",
                      "F-BELOW", "F-PARTIAL", "F-INCOMPLETE"):
        assert forbidden not in source, forbidden


def test_s05_the_record_schema_is_declared_and_carried_in_the_snapshot():
    assert F.AUTHORISATION_RECORD_SCHEMA_VERSION == (
        "myoleg26-direction-f-authorisation-v1")
    assert set(F.AUTHORISATION_RECORD_KEYS) == RECORD_KEYS
    assert len(F.AUTHORISATION_RECORD_KEYS) == 12
    record = F.amendment_snapshot()["authorisation_record"]
    assert record == {"path": RECORD_PATH,
                      "schema_version": "myoleg26-direction-f-authorisation-v1",
                      "keys": sorted(RECORD_KEYS)}


def test_s06_the_geometry_note_labels_the_latency_premise_as_assumed():
    """Review fix. Prereg section 3.3: that per-step cost is latency-bound is
    **assumed, not measured on MyoLeg26**, the conclusion is conditional on it,
    and the only support is the Ant scaling. The note is inside F's digest, so
    it must not call the premise measured."""
    lowered = F.GEOMETRY_NOTE.lower()
    assert "assumed, not measured on myoleg26" in lowered
    assert "measured per-step cost" not in lowered
    assert "if so" in lowered
    assert "ant scaling" in lowered
    assert F.amendment_snapshot()["geometry"] == F.GEOMETRY_NOTE


def test_s07_the_base_recipe_note_names_every_read_of_max_epochs():
    """Review note. PPO reads max_epochs as the begin_epoch argument (a no-op
    for MyoLeg26), as the loop bound, and in the linear-schedule branch, which
    lr_schedule constant never enters. The note says so rather than 'only'."""
    lowered = F.BASE_RECIPE_NOTE.lower()
    assert "acts only as a loop bound" not in lowered
    for phrase in ("begin_epoch", "no-op", "loop bound", "linear"):
        assert phrase in lowered, phrase
    assert F.amendment_snapshot()["base_recipe"] == F.BASE_RECIPE_NOTE
