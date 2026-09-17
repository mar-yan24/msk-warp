"""Unit tests for the probe-E trust-region PROTOCOL AMENDMENT.

CPU only. No simulator, no CUDA, no Warp, no torch, no subprocess, no training.
Nothing here consumes the 28,800 s training budget or the 1,800 s diagnostic
budget; test-suite time is separate accounting.

What this file is for
---------------------
The amendment is a **descendant** of the sealed v2 protocol, not a replacement
for it. The load-bearing properties, each asserted below:

* the sealed protocol module is **not edited**, so its digest is still
  ``4c01fdc9...2169`` and stage 1's sealed ledger keeps validating;
* the amendment's digest **differs** from the sealed one, so the sealed ledger
  refuses every amendment job and the amendment's own ledger refuses every
  sealed job;
* the descent is recorded **in the amendment itself** -- which protocol it comes
  from, what differs, and why -- so a probe-E result cannot be read as a stage-1
  result;
* the recipe name sets are **disjoint**, so no launch is ambiguous about which
  protocol and which ledger it belongs to;
* the trust region is the **single varied factor**: gamma 0.990 and entropy 0.01
  are held at their v1-equivalent values and the variation is expressed as a
  configuration override of the pinned yaml, not as new algorithm code;
* the official reward, model, reset distribution, 1.0 m/s target, evaluation
  horizon and walking gate are **inherited unchanged**.

Two kinds of assertion appear, labelled as such: **transcription** checks
re-state a preregistered number (``probes-DE-preregistration.md`` E.2) against a
module constant and detect drift of it, and **behavioural** checks exercise a
derivation or a refusal path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from msk_warp.analysis import ppo_v2_protocol as P
from msk_warp.analysis import ppo_v2_protocol_e as E

ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = ROOT / "msk_warp/configs/experiments/myoleg26_ppo_v2.yaml"

#: The sealed digest the amendment descends from, transcribed from the stage-1
#: ledger header and from the committed v2 manifest.
SEALED_DIGEST = "4c01fdc94117bd190cd44c6b6a5f411a5c7cfc7b7d3e0462a344dda43baf2169"


def _pinned_config() -> dict:
    return yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))["params"]["config"]


# --------------------------------------------------------------------------
# Unit 1 -- descent: which protocol this is an amendment OF
# --------------------------------------------------------------------------

def test_the_amendment_names_the_protocol_it_descends_from():
    """Transcription + behavioural. The parent is named by module, schema and
    digest, so the relationship is auditable without importing anything."""
    assert E.PARENT_MODULE == "msk_warp.analysis.ppo_v2_protocol"
    assert E.PARENT_SCHEMA_VERSION == P.SCHEMA_VERSION
    assert E.PARENT_PROTOCOL_DIGEST == SEALED_DIGEST


def test_the_sealed_protocol_is_not_edited_by_this_amendment():
    """Behavioural, and the single most load-bearing check in this file: the
    live sealed protocol still hashes to the digest stage 1's ledger asserts."""
    assert P.protocol_digest() == SEALED_DIGEST
    E.assert_parent_unchanged()


def test_a_moved_parent_digest_refuses_the_amendment(monkeypatch):
    """Behavioural. The amendment descends from ONE sealed protocol. If the
    parent moves, the amendment refuses by name instead of silently re-parenting."""
    monkeypatch.setattr(P, "protocol_digest", lambda: "0" * 64)
    with pytest.raises(P.ProtocolError, match="descends from"):
        E.assert_parent_unchanged()
    with pytest.raises(P.ProtocolError, match="descends from"):
        E.protocol_snapshot()
    with pytest.raises(P.ProtocolError, match="descends from"):
        E.protocol_digest()


def test_the_amendment_digest_differs_from_the_sealed_digest():
    """Behavioural. A separate digest is what makes the two ledgers disjoint."""
    assert E.protocol_digest() != P.protocol_digest()
    assert E.protocol_digest() != SEALED_DIGEST
    assert len(E.protocol_digest()) == 64
    assert E.protocol_digest() == E.protocol_digest()


def test_the_snapshot_embeds_the_sealed_snapshot_verbatim():
    """Behavioural. Descent is cryptographic, not merely documentary: the parent
    snapshot is carried byte-for-byte, so any parent change moves this digest."""
    snapshot = E.protocol_snapshot()
    assert snapshot["inherited_protocol"] == P.protocol_snapshot()
    assert snapshot["schema_version"] == E.SCHEMA_VERSION
    assert snapshot["schema_version"] != P.SCHEMA_VERSION


def test_the_amendment_records_what_differs_and_why():
    """Behavioural. A reader must be able to see the delta and the reason from
    the artefact alone."""
    amendment = E.amendment_snapshot()
    assert amendment["parent_protocol_digest"] == SEALED_DIGEST
    assert amendment["amendment_id"] == E.AMENDMENT_ID
    differs = amendment["differs_from_parent"]
    assert isinstance(differs, list) and len(differs) >= 3
    joined = " ".join(differs).lower()
    assert "clip_range" in joined
    assert "ppo_epochs" in joined
    assert "ledger" in joined
    assert "0.584" in amendment["why"]
    assert amendment["not_a_stage_1_result"].strip()
    assert "probes-DE-preregistration" in amendment["preregistration"]


def test_the_amendment_does_not_restate_the_preregistered_criteria():
    """Behavioural. The E.3 criteria were fixed before any data and must not be
    copied, paraphrased or re-derived here, where they could drift."""
    source = Path(E.__file__).read_text(encoding="utf-8")
    for forbidden in ("E-MECHANISM-MOVED", "E-MECHANISM-INERT",
                      "E-BEHAVIOUR-MOVED", "E-BEHAVIOUR-NULL", "0.30", "0.45"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# Unit 2 -- the trust-region arm: one varied factor, expressed as config
# --------------------------------------------------------------------------

def test_the_amendment_declares_a_primary_arm_and_a_declared_alternative():
    """Transcription of E.2: clip_range is primary, ppo_epochs is the declared
    alternative, and exactly one cell may be launched."""
    assert E.PRIMARY_RECIPE == "g990_e010_c040"
    assert E.DECLARED_ALTERNATIVE_RECIPE == "g990_e010_k001"
    assert E.CANONICAL_RECIPE_ORDER == (E.PRIMARY_RECIPE,
                                        E.DECLARED_ALTERNATIVE_RECIPE)
    assert set(E.RECIPES) == set(E.CANONICAL_RECIPE_ORDER)
    assert E.recipe(E.PRIMARY_RECIPE).role == "primary"
    assert E.recipe(E.DECLARED_ALTERNATIVE_RECIPE).role == "declared_alternative"
    assert E.MAX_CELLS == 1


def test_the_trust_region_is_the_single_varied_factor():
    """Behavioural. Every arm holds gamma and entropy_coef at the v1-equivalent
    values of the sealed g990_e010 arm, and varies exactly one trust-region key."""
    sealed = P.recipe("g990_e010")
    for name in E.CANONICAL_RECIPE_ORDER:
        arm = E.recipe(name)
        assert arm.gamma == sealed.gamma == 0.990
        assert arm.entropy_coef == sealed.entropy_coef == 0.01
        varied = [key for key, value in arm.overrides.items() if value is not None]
        assert len(varied) == 1, f"{name} varies {varied}"
    assert E.recipe(E.PRIMARY_RECIPE).overrides == {"clip_range": 0.4}
    assert E.recipe(E.DECLARED_ALTERNATIVE_RECIPE).overrides == {"ppo_epochs": 1}


def test_each_override_really_differs_from_the_pinned_yaml():
    """Behavioural, against the freeze-pinned yaml itself: an override that
    matched the pinned value would vary nothing at all."""
    config = _pinned_config()
    assert config["clip_range"] == 0.2
    assert config["ppo_epochs"] == 5
    for name in E.CANONICAL_RECIPE_ORDER:
        for key, value in E.recipe(name).overrides.items():
            assert config[key] != value, f"{name} override {key} matches the pin"


def test_everything_else_is_held_at_the_pinned_value():
    """Behavioural. The arms name no key outside the two trust-region keys, so
    no other hyperparameter can move under this amendment."""
    assert E.TRUST_REGION_KEYS == ("clip_range", "ppo_epochs")
    config = _pinned_config()
    for name in E.CANONICAL_RECIPE_ORDER:
        assert set(E.recipe(name).overrides) <= set(E.TRUST_REGION_KEYS)
    for key in ("gae_lambda", "num_minibatches", "max_grad_norm", "obs_rms",
                "actor_learning_rate", "critic_learning_rate", "lr_schedule"):
        assert key in config


def test_the_variation_is_configuration_not_new_algorithm_code():
    """Behavioural. The amendment module imports no algorithm, no torch and no
    simulator: it can only hand numbers to the pinned configuration."""
    source = Path(E.__file__).read_text(encoding="utf-8")
    for forbidden in ("import torch", "mujoco", "import warp", "subprocess",
                      "from msk_warp.algorithms", "num_minibatches"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# Unit 3 -- disjointness: a probe-E run can never be a stage-1 cell
# --------------------------------------------------------------------------

def test_the_recipe_name_sets_are_disjoint():
    """Behavioural. Disjoint names are what make every launch unambiguous."""
    assert set(E.RECIPES).isdisjoint(set(P.RECIPES))
    E.assert_recipe_names_disjoint()


def test_the_disjointness_guard_has_teeth(monkeypatch):
    """Anti-vacuity. The check above would pass against any name set at all, so
    the guard is separately shown to refuse a colliding one."""
    colliding = dict(E.RECIPES)
    colliding["g990_e010"] = E.RECIPES[E.PRIMARY_RECIPE]
    monkeypatch.setattr(E, "RECIPES", colliding)
    with pytest.raises(P.ProtocolError, match="g990_e010"):
        E.assert_recipe_names_disjoint()


def test_the_amendment_refuses_a_sealed_stage_1_arm_by_name():
    """Behavioural. Running g990_e010 on the amendment's ledger would put a
    stage-1 arm under a probe-E digest, so it is refused and told where to go."""
    for name in P.CANONICAL_RECIPE_ORDER:
        with pytest.raises(P.ProtocolError, match="sealed"):
            E.recipe(name)


def test_the_sealed_protocol_refuses_the_amendment_arms():
    """Behavioural. The sealed protocol is untouched, so it does not know the
    amendment arms: a sealed launch cannot charge one to a stage-1 cell."""
    for name in E.CANONICAL_RECIPE_ORDER:
        with pytest.raises(P.ProtocolError):
            P.recipe(name)


def test_the_amendment_error_is_a_protocol_error_subclass():
    """Behavioural. Every existing ``except P.ProtocolError`` handler in the
    runner must keep producing a clean refusal rather than a traceback."""
    assert issubclass(E.AmendmentError, P.ProtocolError)
    assert issubclass(E.AmendmentError, ValueError)
    assert E.ProtocolError is P.ProtocolError


def test_an_unknown_name_is_refused_and_names_the_declared_arms():
    with pytest.raises(P.ProtocolError, match="g990_e010_c040"):
        E.recipe("nope")
    with pytest.raises(P.ProtocolError):
        E.recipe(None)


# --------------------------------------------------------------------------
# Unit 4 -- the official task is inherited unchanged
# --------------------------------------------------------------------------

def test_the_official_task_and_the_walking_gate_are_inherited_unchanged():
    """Behavioural. Identity, not equality, for the mappings the sealed protocol
    owns: the amendment cannot hold a divergent copy of the gate."""
    assert E.WALKING_GATE is P.WALKING_GATE
    assert E.V2_RESET_BLOCKS is P.V2_RESET_BLOCKS
    assert E.STAGES is P.STAGES
    assert E.V2_CONFIG == P.V2_CONFIG
    assert E.stage is P.stage
    assert E.selection_reset_block is P.selection_reset_block
    assert E.segment_rounds is P.segment_rounds


def test_the_fixed_arm_and_the_caps_are_inherited_unchanged():
    for name in ("NUM_WORLDS", "CONTROLS_PER_WORLD_PER_EPOCH", "PHYSICS_SUBSTEPS",
                 "FORWARD_SPEED_TARGET_MPS", "IMITATION",
                 "EVALUATION_HORIZON_CONTROLS", "EVALUATION_HORIZON_SECONDS",
                 "EVALUATION_EPOCH_INTERVAL", "SEGMENT_MAX_EPOCHS",
                 "SEGMENT_WORK_DEADLINE_S", "SEGMENT_CALL_BOUND_S",
                 "TOOL_TIMEOUT_MAX_S", "TOTAL_WALL_CAP_S", "RESERVE_WALL_CAP_S"):
        assert getattr(E, name) == getattr(P, name), name
    assert E.FORWARD_SPEED_TARGET_MPS == 1.0
    assert E.IMITATION is False


# --------------------------------------------------------------------------
# Unit 5 -- the authorised cell, and the guards around it
# --------------------------------------------------------------------------

def test_the_authorised_cell_is_one_64_epoch_screen_segment():
    """Transcription of E.2: screen stage, one segment, 64 epochs, selection
    block 11000-11015."""
    cell = E.amendment_cell()
    assert cell.stage == "screen" == E.AMENDMENT_STAGE
    assert cell.recipe == E.PRIMARY_RECIPE
    assert cell.seed == 1001
    assert cell.segment_index == 0 == E.MAX_SEGMENT_INDEX
    assert (cell.start_epoch, cell.end_epoch) == (0, 64)
    assert cell.epochs == 64 == E.AMENDMENT_EPOCHS
    assert cell.selection_reset_block == tuple(range(11000, 11016))
    assert E.AMENDMENT_SEEDS == (1001,)


def test_the_cell_seed_is_a_scheduled_seed_of_the_inherited_stage():
    """Behavioural. The amendment may not open an unbudgeted seed allowance."""
    for seed in E.AMENDMENT_SEEDS:
        assert seed in P.stage(E.AMENDMENT_STAGE).seeds


def test_assert_launchable_accepts_the_authorised_cell():
    cell = E.amendment_cell()
    E.assert_launchable(stage=cell.stage, recipe=cell.recipe, seed=cell.seed,
                        segment_index=cell.segment_index, epochs=cell.epochs)


@pytest.mark.parametrize("field, value, match", [
    ("stage", "refine", "screen"),
    ("seed", 1002, "1001"),
    ("segment_index", 1, "one 64-epoch segment"),
    ("epochs", 32, "64"),
    ("recipe", "g990_e010", "sealed"),
])
def test_assert_launchable_refuses_anything_outside_the_authorised_cell(
        field, value, match):
    """Behavioural. Each refusal is its own path: a resume segment would double
    the charge, a second seed would be an unpreregistered replication, and a
    sealed arm belongs to the sealed ledger."""
    cell = E.amendment_cell()
    kwargs = {"stage": cell.stage, "recipe": cell.recipe, "seed": cell.seed,
              "segment_index": cell.segment_index, "epochs": cell.epochs}
    kwargs[field] = value
    with pytest.raises(P.ProtocolError, match=match):
        E.assert_launchable(**kwargs)


def test_the_declared_alternative_is_launchable_but_is_not_the_default_cell():
    """Behavioural. The alternative was preregistered, so it is expressible; it
    is not the cell the amendment authorises by default."""
    alternative = E.amendment_cell(E.DECLARED_ALTERNATIVE_RECIPE)
    assert alternative.recipe == E.DECLARED_ALTERNATIVE_RECIPE
    assert E.amendment_cell().recipe == E.PRIMARY_RECIPE
    E.assert_launchable(stage=alternative.stage, recipe=alternative.recipe,
                        seed=alternative.seed,
                        segment_index=alternative.segment_index,
                        epochs=alternative.epochs)


# --------------------------------------------------------------------------
# Unit 6 -- the separate ledger, and the carry-forward requirement
# --------------------------------------------------------------------------

def test_the_amendment_declares_a_separate_ledger_under_an_ignored_path():
    assert E.LEDGER != E.PARENT_LEDGER
    assert E.LEDGER.startswith("logs/")
    assert E.PARENT_LEDGER.startswith("logs/")
    assert E.RUN_ROOT.startswith("logs/")
    assert E.RUN_ROOT not in E.PARENT_LEDGER


def test_the_amendment_requires_a_carried_forward_opening_balance():
    """Behavioural. A fresh ledger with no carried spend would reset the caps,
    which is exactly what a new ledger must not be used for."""
    assert E.REQUIRES_CARRY_FORWARD is True
    provenance = E.ledger_provenance()
    assert provenance["parent_protocol_digest"] == SEALED_DIGEST
    assert provenance["amendment_id"] == E.AMENDMENT_ID
    assert provenance["not_a_stage_1_result"].strip()
    assert "carried forward" in " ".join(provenance["differs_from_parent"]).lower()
    assert E.PARENT_LEDGER in provenance["carry_forward_note"] or (
        "sealed ledger" in provenance["carry_forward_note"])
    assert provenance["parent_ledger"] == E.PARENT_LEDGER


def test_the_ledger_and_the_cell_appear_in_the_snapshot():
    amendment = E.amendment_snapshot()
    assert amendment["ledger"]["path"] == E.LEDGER
    assert amendment["ledger"]["separate_from"] == E.PARENT_LEDGER
    assert amendment["ledger"]["requires_carry_forward"] is True
    assert amendment["cell"]["stage"] == "screen"
    assert amendment["cell"]["epochs"] == 64
    assert amendment["cell"]["max_cells"] == 1


# --------------------------------------------------------------------------
# Unit 7 -- hygiene
# --------------------------------------------------------------------------

def test_the_module_is_standard_library_only_and_grants_no_budget():
    source = Path(E.__file__).read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import yaml", "os.system", "Popen",
                      "allow_unvalidated_gradients"):
        assert forbidden not in source


def test_the_snapshot_is_json_canonical_and_stable():
    first = P.canonical_json(E.protocol_snapshot())
    second = P.canonical_json(E.protocol_snapshot())
    assert first == second
    assert b"NaN" not in first
    assert first != P.canonical_json(P.protocol_snapshot())
# --------------------------------------------------------------------------
# Unit 8 -- fix round 2: the seed is labelled as a post-hoc, data-informed
# choice, in the provenance and therefore in the digest, the ledger header and
# the manifest. It is NOT restated as a neutral choice anywhere.
# --------------------------------------------------------------------------

def test_the_seed_is_labelled_as_a_post_hoc_data_informed_choice():
    note = E.SEED_CHOICE_NOTE
    assert "NOT PREREGISTERED" in note
    assert "POST-HOC" in note and "DATA-INFORMED" in note
    assert "STAGE-1 OUTCOME" in note
    assert "not threshold-seeking" in note
    assert "seed-independent" in note
    assert "BEFORE the launch" in note


def test_the_seed_label_travels_with_the_provenance():
    amendment = E.amendment_snapshot()
    assert amendment["seed_preregistered"] is False
    assert amendment["seed_choice"] == E.SEED_CHOICE_NOTE
    provenance = E.ledger_provenance()
    assert provenance["seed_preregistered"] is False
    assert provenance["seed_choice"] == E.SEED_CHOICE_NOTE
    assert E.SEED_CHOICE_NOTE in P.canonical_json(E.protocol_snapshot()).decode()


def test_the_module_cites_the_correct_preregistration_section():
    """The two upstream documents cite section 2.3; section 2 has no
    subsections and probe E is section 4. The module has it right and a later
    edit must not 'fix' it back."""
    assert "section 4 (E.1-E.4)" in E.PREREGISTRATION
    assert "2.3" not in E.PREREGISTRATION
