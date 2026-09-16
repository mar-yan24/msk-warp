"""Unit tests for the sealed MyoLeg26 v2 PPO campaign protocol.

CPU only. No simulator, no CUDA, no Warp, no torch, no real clock dependence, no
training. Nothing here consumes the 28,800 s training budget or the 1,800 s
diagnostic budget; test-suite time is separate accounting.

Two kinds of assertion appear below and are labelled as such:

* **Transcription checks** re-state a number from ``task-4b-brief.md`` against the
  module constant. They corroborate nothing physical; they detect drift of a
  sealed value, which is their whole purpose.
* **Behavioural checks** exercise a derivation (rotation, medians, eligibility,
  ranking, promotion, disjointness) against a hand-computed expected value
  written out in the test body.

The reset-block disjointness test reads the **committed v1 freeze manifest**
rather than a hard-coded copy of the v1 blocks, so it is a real check against the
frozen artefact and not a restatement of this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from msk_warp.analysis import ppo_v2_protocol as P

ROOT = Path(__file__).resolve().parents[2]
V1_MANIFEST = ROOT / "msk_warp/configs/experiments/myoleg26_baseline_v1.json"
V2_CONFIG = ROOT / "msk_warp/configs/experiments/myoleg26_ppo_v2.yaml"
V1_CONFIG = ROOT / "msk_warp/configs/experiments/myoleg26_ppo.yaml"


# --------------------------------------------------------------------------
# Transcription checks: the sealed fixed arm and the segment limits
# --------------------------------------------------------------------------

def test_fixed_arm_constants_are_the_sealed_values():
    """Transcription. Fixed for every arm; invented or rounded values are drift."""
    assert P.NUM_WORLDS == 64
    assert P.CONTROLS_PER_WORLD_PER_EPOCH == 128
    assert P.PHYSICS_SUBSTEPS == 4
    assert P.FORWARD_SPEED_TARGET_MPS == 1.0
    assert P.IMITATION is False
    assert P.EVALUATION_HORIZON_CONTROLS == 500
    assert P.EVALUATION_HORIZON_SECONDS == 4.0


def test_segment_limits_are_the_declared_constants():
    """Transcription: <=64 epochs/segment, <=460 s work deadline, <=600 s call bound."""
    assert P.SEGMENT_MAX_EPOCHS == 64
    assert P.SEGMENT_WORK_DEADLINE_S == 460
    assert P.SEGMENT_CALL_BOUND_S == 600
    assert P.TOOL_TIMEOUT_MAX_S == 600
    # The work deadline must sit strictly inside the call bound, leaving the
    # shutdown/checkpoint room the brief requires.
    assert P.SEGMENT_WORK_DEADLINE_S < P.SEGMENT_CALL_BOUND_S
    assert P.TOOL_TIMEOUT_MAX_S <= P.SEGMENT_CALL_BOUND_S


def test_evaluation_cadence_is_epoch_zero_then_every_64_completed_epochs():
    assert P.EVALUATION_EPOCH_INTERVAL == 64
    assert P.evaluation_epochs(128) == (0, 64, 128)
    assert P.evaluation_epochs(256) == (0, 64, 128, 192, 256)
    assert P.evaluation_epochs(384) == (0, 64, 128, 192, 256, 320, 384)


def test_derived_per_epoch_work_matches_the_fixed_arm():
    """Behavioural: 64 worlds x 128 controls, 4 substeps each."""
    assert P.controls_per_epoch() == 8192
    assert P.physics_steps_per_epoch() == 32768
    # The 128-epoch screen cap reproduces the v1 per-seed training volume.
    assert P.train_control_transitions(128) == 1048576
    assert P.train_physics_steps(128) == 4194304


# --------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------

def test_four_recipes_with_the_sealed_gamma_entropy_pairs():
    """Transcription of the four (gamma, entropy_coef) arms."""
    assert P.CANONICAL_RECIPE_ORDER == (
        "g990_e010", "g998_e010", "g990_e000", "g998_e000",
    )
    assert tuple(P.RECIPES) == P.CANONICAL_RECIPE_ORDER
    expected = {
        "g990_e010": (0.990, 0.01),
        "g998_e010": (0.998, 0.01),
        "g990_e000": (0.990, 0),
        "g998_e000": (0.998, 0),
    }
    for name, (gamma, entropy) in expected.items():
        assert P.recipe(name).gamma == gamma
        assert P.recipe(name).entropy_coef == entropy
        assert P.recipe(name).name == name


def test_recipe_table_is_immutable_and_unknown_names_are_refused():
    with pytest.raises(TypeError):
        P.RECIPES["g990_e010"] = None  # type: ignore[index]
    with pytest.raises(P.ProtocolError):
        P.recipe("g999_e999")


# --------------------------------------------------------------------------
# Stages and budgets
# --------------------------------------------------------------------------

def test_stage_table_is_the_sealed_table():
    """Transcription of the stage table, row by row."""
    assert P.STAGE_ORDER == ("screen", "refine", "confirm")

    screen = P.stage("screen")
    assert (screen.index, screen.epoch_cap_per_seed) == (1, 128)
    assert screen.seeds == (1001, 1002)
    assert (screen.wall_cap_s, screen.per_seed_safety_cap_s) == (7200, 1200)
    assert screen.max_promoted == 2
    assert screen.max_recipes == 4

    refine = P.stage("refine")
    assert (refine.index, refine.epoch_cap_per_seed) == (2, 256)
    assert refine.seeds == (2001, 2002)
    assert (refine.wall_cap_s, refine.per_seed_safety_cap_s) == (7200, 2400)
    assert refine.max_promoted == 1
    assert refine.max_recipes == 2

    confirm = P.stage("confirm")
    assert (confirm.index, confirm.epoch_cap_per_seed) == (3, 384)
    assert confirm.seeds == (3001, 3002, 3003, 3004, 3005)
    assert len(confirm.seeds) == 5
    assert (confirm.wall_cap_s, confirm.per_seed_safety_cap_s) == (13500, 3600)
    assert confirm.max_promoted == 0
    assert confirm.max_recipes == 1

    with pytest.raises(P.ProtocolError):
        P.stage("screen_extra")


def test_wall_budget_decomposition_is_exact():
    """Behavioural: 7,200 + 7,200 + 13,500 + 900 reserve == 28,800."""
    assert P.TOTAL_WALL_CAP_S == 28800
    assert P.RESERVE_WALL_CAP_S == 900
    stage_caps = [P.stage(key).wall_cap_s for key in P.STAGE_ORDER]
    assert stage_caps == [7200, 7200, 13500]
    assert sum(stage_caps) + P.RESERVE_WALL_CAP_S == P.TOTAL_WALL_CAP_S
    assert sum(stage_caps) == 27900


def test_per_seed_safety_caps_are_dominated_by_the_stage_caps():
    """Behavioural: the caps are not additional allocations.

    Every stage's (seeds x recipes x per-seed cap) exceeds its stage wall cap, so
    the stage cap, not the per-seed cap, is what binds. Finishing under a cap
    authorises no extra recipe.
    """
    for key, runs in (("screen", 4 * 2), ("refine", 2 * 2), ("confirm", 1 * 5)):
        spec = P.stage(key)
        assert runs == spec.max_recipes * len(spec.seeds)
        assert runs * spec.per_seed_safety_cap_s > spec.wall_cap_s


def test_reserve_is_overhead_only_and_authorises_no_recipe():
    assert "no extra recipes" in P.RESERVE_PURPOSE
    assert P.RESERVE_AUTHORISES_EXTRA_RECIPES is False


# --------------------------------------------------------------------------
# Reset blocks
# --------------------------------------------------------------------------

def test_v2_reset_blocks_are_the_sealed_ranges():
    """Transcription of the five v2 blocks."""
    assert P.V2_RESET_BLOCKS["selection_screen"] == tuple(range(11000, 11016))
    assert P.V2_RESET_BLOCKS["selection_refine"] == tuple(range(12000, 12016))
    assert P.V2_RESET_BLOCKS["selection_confirm"] == tuple(range(13000, 13016))
    assert P.V2_RESET_BLOCKS["confirmation"] == tuple(range(21000, 21032))
    assert P.V2_RESET_BLOCKS["audit"] == tuple(range(31000, 31016))
    assert len(P.V2_RESET_BLOCKS) == 5
    assert P.selection_reset_block("screen") == tuple(range(11000, 11016))
    assert P.selection_reset_block("refine") == tuple(range(12000, 12016))
    assert P.selection_reset_block("confirm") == tuple(range(13000, 13016))


def test_v2_reset_blocks_are_mutually_disjoint_and_disjoint_from_committed_v1():
    """Behavioural, against the committed v1 freeze manifest -- not a copy of it."""
    blocks = {name: set(ids) for name, ids in P.V2_RESET_BLOCKS.items()}
    # Anti-vacuity: an empty or short block set would make every pairwise
    # disjointness check below trivially true.
    assert len(blocks) == 5
    assert all(len(ids) >= 16 for ids in blocks.values())
    names = sorted(blocks)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not blocks[left] & blocks[right], f"{left} overlaps {right}"

    v1_protocol = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))["protocol"]
    v1_ids = set()
    for key in ("selection_episode_seeds", "confirmation_episode_seeds",
                "audit_episode_seeds"):
        ids = v1_protocol[key]
        assert ids, f"v1 manifest lost {key}"
        v1_ids |= set(int(value) for value in ids)
    # Guard against a vacuous comparison: the v1 blocks really are the recorded ones.
    assert v1_ids == set(range(10000, 10016)) | set(range(20000, 20032)) | set(range(30000, 30016))

    for name, ids in blocks.items():
        assert not ids & v1_ids, f"v2 block {name} reuses a v1 reset id"

    # assert_reset_blocks_disjoint reads the same manifest and must agree.
    P.assert_reset_blocks_disjoint()


def test_confirmation_and_audit_blocks_are_locked_until_selection_is_locked():
    assert P.CONFIRMATION_BLOCKED_UNTIL_SELECTION_LOCKED is True
    with pytest.raises(P.ProtocolError, match="selection"):
        P.assert_confirmation_unlocked(selection_locked=False)
    P.assert_confirmation_unlocked(selection_locked=True)


# --------------------------------------------------------------------------
# Run order (R08)
# --------------------------------------------------------------------------

def test_segment_rounds_respect_the_64_epoch_segment_ceiling():
    assert [(r.index, r.start_epoch, r.end_epoch) for r in P.segment_rounds(128)] == [
        (0, 0, 64), (1, 64, 128),
    ]
    assert len(P.segment_rounds(256)) == 4
    assert len(P.segment_rounds(384)) == 6
    for cap in (128, 256, 384):
        rounds = P.segment_rounds(cap)
        assert all(r.epochs <= P.SEGMENT_MAX_EPOCHS for r in rounds)
        assert sum(r.epochs for r in rounds) == cap
        assert rounds[0].start_epoch == 0 and rounds[-1].end_epoch == cap


def test_rotation_is_the_canonical_order_rotated_by_seed_plus_segment_index():
    """Behavioural, against a hand-written expected rotation.

    A, B, C, D = g990_e010, g998_e010, g990_e000, g998_e000.
    shift = (seed_index + segment_index) % 4.
    """
    A, B, C, D = P.CANONICAL_RECIPE_ORDER
    assert P.rotated_recipes(0, 0) == (A, B, C, D)
    assert P.rotated_recipes(1, 0) == (B, C, D, A)
    assert P.rotated_recipes(0, 1) == (B, C, D, A)
    assert P.rotated_recipes(1, 1) == (C, D, A, B)
    assert P.rotated_recipes(2, 1) == (D, A, B, C)
    assert P.rotated_recipes(2, 2) == (A, B, C, D)  # shift 4 wraps to 0
    # Deterministic and pure: same inputs, same output, no hidden state.
    assert P.rotated_recipes(1, 0) == P.rotated_recipes(1, 0)
    assert P.rotated_recipes(5, 3) == P.rotated_recipes(0, 0)  # shift 8 -> 0


def test_rotation_of_a_subset_keeps_canonical_relative_order():
    A, _B, C, _D = P.CANONICAL_RECIPE_ORDER
    assert P.rotated_recipes(0, 0, (A, C)) == (A, C)
    assert P.rotated_recipes(1, 0, (A, C)) == (C, A)
    assert P.rotated_recipes(0, 0, (C,)) == (C,)
    assert P.rotated_recipes(7, 3, (C,)) == (C,)
    with pytest.raises(P.ProtocolError):
        P.rotated_recipes(0, 0, (C, A))  # not in canonical relative order
    with pytest.raises(P.ProtocolError):
        P.rotated_recipes(0, 0, (A, A))
    with pytest.raises(P.ProtocolError):
        P.rotated_recipes(0, 0, ())


def test_screen_run_order_is_segment_outer_then_seed_then_rotated_recipes():
    """Behavioural, against a fully hand-written expected order (16 jobs)."""
    A, B, C, D = P.CANONICAL_RECIPE_ORDER
    expected = [
        # segment round 0
        (0, 1001, A), (0, 1001, B), (0, 1001, C), (0, 1001, D),
        (0, 1002, B), (0, 1002, C), (0, 1002, D), (0, 1002, A),
        # segment round 1
        (1, 1001, B), (1, 1001, C), (1, 1001, D), (1, 1001, A),
        (1, 1002, C), (1, 1002, D), (1, 1002, A), (1, 1002, B),
    ]
    order = P.run_order("screen")
    assert [(j.segment_index, j.seed, j.recipe) for j in order] == expected
    assert len(order) == 16
    assert all(job.stage == "screen" for job in order)
    assert all(job.end_epoch - job.start_epoch == 64 for job in order)
    assert {(j.segment_index, j.start_epoch, j.end_epoch) for j in order} == {
        (0, 0, 64), (1, 64, 128),
    }
    # Every (recipe, seed, segment) pair appears exactly once.
    assert len({(j.recipe, j.seed, j.segment_index) for j in order}) == 16
    # Determinism: recomputed identically.
    assert P.run_order("screen") == order


def test_screen_rotation_balance_is_reported_not_claimed_eliminated():
    """Behavioural: residual ordering bias exists and is quantified, per R08."""
    order = P.run_order("screen")
    positions: dict[str, list[int]] = {name: [] for name in P.CANONICAL_RECIPE_ORDER}
    for index, job in enumerate(order):
        positions[job.recipe].append(index % 4 + 1)
    # Hand-computed 1-based within-row positions across the four (segment, seed) rows.
    assert positions == {
        "g990_e010": [1, 4, 4, 3],
        "g998_e010": [2, 1, 1, 4],
        "g990_e000": [3, 2, 2, 1],
        "g998_e000": [4, 3, 3, 2],
    }
    sums = {name: sum(values) for name, values in positions.items()}
    assert sums == {"g990_e010": 12, "g998_e010": 8, "g990_e000": 8, "g998_e000": 12}
    # Not balanced: the rotation reduces but does not eliminate ordering bias.
    assert len(set(sums.values())) > 1


def test_refine_and_confirm_run_orders_require_an_explicit_recipe_set():
    A, _B, C, _D = P.CANONICAL_RECIPE_ORDER
    refine = P.run_order("refine", (A, C))
    assert len(refine) == 4 * 2 * 2  # 4 segment rounds, 2 seeds, 2 recipes
    assert [(j.segment_index, j.seed, j.recipe) for j in refine[:4]] == [
        (0, 2001, A), (0, 2001, C), (0, 2002, C), (0, 2002, A),
    ]

    confirm = P.run_order("confirm", (C,))
    assert len(confirm) == 6 * 5 * 1
    assert {job.seed for job in confirm} == {3001, 3002, 3003, 3004, 3005}
    assert {job.recipe for job in confirm} == {C}

    with pytest.raises(P.ProtocolError, match="at most"):
        P.run_order("refine", P.CANONICAL_RECIPE_ORDER)  # 4 > refine max_recipes
    with pytest.raises(P.ProtocolError, match="at most"):
        P.run_order("confirm", (A, C))  # 2 > confirm max_recipes


# --------------------------------------------------------------------------
# Promotion thresholds, eligibility and ranking
# --------------------------------------------------------------------------

def test_promotion_thresholds_are_the_sealed_values():
    """Transcription. survival_fraction is a FRACTION, not a survivor count."""
    assert P.SCREEN_MEDIAN_DURATION_S == 1.25
    assert P.SCREEN_ANY_SURVIVAL_FRACTION == 0.0625  # 1/16, exactly representable
    assert P.SCREEN_ANY_SURVIVAL_FRACTION == 1 / 16
    assert P.SCREEN_MAX_PROMOTED == 2
    assert P.REFINE_MEDIAN_DURATION_S == 2.0
    assert P.REFINE_FALLBACK_MEDIAN_DURATION_S == 1.5
    assert P.REFINE_FALLBACK_ANY_SURVIVAL_FRACTION == 0.125  # 2/16
    assert P.REFINE_FALLBACK_ANY_SURVIVAL_FRACTION == 2 / 16
    assert P.REFINE_MAX_PROMOTED == 1
    assert P.SELECTION_EPISODES == 16 and P.CONFIRMATION_EPISODES == 32


def _seed(seed, duration, survival, *, in_band=0.0, rmse=1.0, complete=True):
    return P.SeedOutcome(seed=seed, complete=complete, duration_s=duration,
                         survival_fraction=survival, in_band_seconds=in_band,
                         rmse_mps=rmse)


def _screen(name, durations, survivals, **kwargs):
    seeds = P.stage("screen").seeds
    return P.RecipeOutcome(recipe=name, stage="screen", seeds=tuple(
        _seed(s, d, v, **kwargs) for s, d, v in zip(seeds, durations, survivals)))


def _refine(name, durations, survivals, **kwargs):
    seeds = P.stage("refine").seeds
    return P.RecipeOutcome(recipe=name, stage="refine", seeds=tuple(
        _seed(s, d, v, **kwargs) for s, d, v in zip(seeds, durations, survivals)))


def test_median_of_two_scheduled_seeds_is_their_mean():
    assert P.median((1.0, 1.5)) == 1.25
    assert P.median((1.0, 1.25)) == 1.125
    assert P.median((1.5, 2.5)) == 2.0
    assert P.median((1.5, 2.0)) == 1.75
    assert P.median((1.0, 2.0, 5.0)) == 2.0
    with pytest.raises(P.ProtocolError):
        P.median(())
    with pytest.raises(P.ProtocolError):
        P.median((1.0, math.nan))


def test_screen_eligibility_hand_computed_counterexamples():
    """Behavioural. median duration >= 1.25 OR any survival >= 1/16."""
    # median (1.0 + 1.5) / 2 == 1.25 exactly -> eligible on the boundary
    assert P.screen_eligible(_screen("g990_e010", (1.0, 1.5), (0.0, 0.0))) is True
    # median 1.125 < 1.25 and no survival -> ineligible
    assert P.screen_eligible(_screen("g998_e010", (1.0, 1.25), (0.0, 0.0))) is False
    # short episodes but one seed survives 1/16 -> eligible on the OR branch
    assert P.screen_eligible(_screen("g990_e000", (0.5, 0.5), (0.0625, 0.0))) is True
    # 1/32 survival is below the 1/16 threshold -> ineligible
    assert P.screen_eligible(_screen("g998_e000", (1.0, 1.25), (0.03125, 0.0))) is False


def test_refine_eligibility_hand_computed_counterexamples():
    """Behavioural. median >= 2.0 OR (median >= 1.5 AND any survival >= 2/16)."""
    # median 2.0 exactly -> eligible on the boundary
    assert P.refine_eligible(_refine("g990_e010", (1.5, 2.5), (0.0, 0.0))) is True
    # median 1.75 with a 2/16 survivor -> eligible on the AND branch
    assert P.refine_eligible(_refine("g998_e010", (1.5, 2.0), (0.125, 0.0))) is True
    # median 1.75 but only 1/16 survival -> the AND branch fails
    assert P.refine_eligible(_refine("g990_e000", (1.5, 2.0), (0.0625, 0.0))) is False
    # high survival cannot rescue a median below 1.5
    assert P.refine_eligible(_refine("g998_e000", (1.0, 1.5), (0.5, 0.0))) is False


def test_recipe_rank_puts_duration_first_unlike_the_checkpoint_level_key():
    """Behavioural, and a deliberate divergence.

    The recipe-level key is four maximised components: median duration, median
    survival, median in-band seconds, -median RMSE. Task 4a's *checkpoint*-level
    behavior_rank_v2 puts survival first. Different levels; not a bug.
    """
    assert P.RECIPE_RANK_COMPONENTS == (
        "median_duration_s", "median_survival_fraction", "median_in_band_seconds",
        "negative_median_rmse_mps",
    )
    low_duration_high_survival = _screen("g990_e010", (1.0, 1.0), (1.0, 1.0))
    high_duration_no_survival = _screen("g998_e010", (3.0, 3.0), (0.0, 0.0))
    assert P.recipe_rank(high_duration_no_survival) > P.recipe_rank(low_duration_high_survival)
    assert len(P.recipe_rank(low_duration_high_survival)) == 4


def test_recipe_rank_tie_breaks_component_by_component():
    """Behavioural: each component breaks a tie in the one below it."""
    base = dict(durations=(2.0, 2.0), survivals=(0.25, 0.25))
    survival_wins = P.recipe_rank(_screen("g990_e010", (2.0, 2.0), (0.5, 0.5)))
    survival_loses = P.recipe_rank(_screen("g998_e010", **base))
    assert survival_wins > survival_loses

    in_band_wins = P.recipe_rank(_screen("g990_e010", in_band=3.0, **base))
    in_band_loses = P.recipe_rank(_screen("g998_e010", in_band=1.0, **base))
    assert in_band_wins > in_band_loses

    rmse_low = P.recipe_rank(_screen("g990_e010", rmse=0.2, **base))
    rmse_high = P.recipe_rank(_screen("g998_e010", rmse=0.9, **base))
    assert rmse_low > rmse_high  # lower median RMSE ranks higher

    # Fully tied on all four metrics: recipe_rank itself no longer separates them.
    assert (P.recipe_rank(_screen("g990_e010", **base))
            == P.recipe_rank(_screen("g998_e000", **base)))


# --------------------------------------------------------------------------
# Ruling 1: the final tie-break is LEXICOGRAPHIC on the recipe id string
# --------------------------------------------------------------------------

def test_final_tie_break_is_lexicographic_on_the_recipe_id():
    """Behavioural, and it must NOT be the declaration index.

    A declaration-order index silently changes meaning if anyone reorders
    CANONICAL_RECIPE_ORDER; a lexicographic rule on the id string is
    reproducible by a reader from the ids alone and is stable under reordering.
    """
    assert P.RECIPE_TIE_BREAK == "lexicographically_least_recipe_id"

    base = dict(durations=(2.0, 2.0), survivals=(0.25, 0.25))
    # g990_e000 is declared THIRD but is lexicographically FIRST.
    declared_first = _screen("g990_e010", **base)
    lexicographically_first = _screen("g990_e000", **base)
    assert P.CANONICAL_RECIPE_ORDER.index("g990_e010") == 0
    assert P.CANONICAL_RECIPE_ORDER.index("g990_e000") == 2
    assert "g990_e000" < "g990_e010"
    # sorted() ascending on the order key, so the winner sorts first.
    assert (P.recipe_order_key(lexicographically_first)
            < P.recipe_order_key(declared_first))

    # The two rules genuinely disagree: lexicographic order is not declaration order.
    assert sorted(P.CANONICAL_RECIPE_ORDER) != list(P.CANONICAL_RECIPE_ORDER)
    assert sorted(P.CANONICAL_RECIPE_ORDER) == [
        "g990_e000", "g990_e010", "g998_e000", "g998_e010",
    ]


def test_a_fully_degenerate_screen_promotes_the_lexicographically_least_pair():
    """Behavioural, hand computed: the case where the two rules disagree.

    All four recipes tie on every metric, so only the tie-break decides. The
    lexicographic rule promotes g990_e000 and g990_e010; a declaration-index
    rule would have promoted g990_e010 and g998_e010.
    """
    outcomes = [_screen(name, (2.0, 2.0), (0.25, 0.25))
                for name in P.CANONICAL_RECIPE_ORDER]
    decision = P.promote("screen", outcomes)
    assert decision.promoted == ("g990_e000", "g990_e010")
    assert decision.promoted != P.CANONICAL_RECIPE_ORDER[:2]
    assert [name for name, _ in decision.ranked] == [
        "g990_e000", "g990_e010", "g998_e000", "g998_e010",
    ]
    # The metric components really are identical, so the tie-break is load-bearing.
    keys = {P.recipe_rank(row) for row in outcomes}
    assert len(keys) == 1


def test_the_tie_break_does_not_override_any_metric_component():
    """A lexicographically later id still wins on a better metric."""
    worse_name_better_metric = _screen("g998_e010", (3.0, 3.0), (0.25, 0.25))
    better_name_worse_metric = _screen("g990_e000", (2.0, 2.0), (0.25, 0.25))
    assert (P.recipe_order_key(worse_name_better_metric)
            < P.recipe_order_key(better_name_worse_metric))
    decision = P.promote("refine", [
        _refine("g998_e010", (3.0, 3.0), (0.25, 0.25)),
        _refine("g990_e000", (2.0, 2.0), (0.25, 0.25)),
    ])
    assert decision.promoted == ("g998_e010",)


def test_order_key_refuses_an_incomplete_recipe_like_the_rank_does():
    seeds = P.stage("screen").seeds
    censored = P.RecipeOutcome(recipe="g990_e010", stage="screen", seeds=(
        _seed(seeds[0], 3.0, 0.5), _seed(seeds[1], 3.0, 0.5, complete=False)))
    with pytest.raises(P.IncompleteRecipeError):
        P.recipe_order_key(censored)


# --------------------------------------------------------------------------
# Ruling 2: the per-seed cap is keyed (stage, recipe, seed) -- recorded, not fixed
# --------------------------------------------------------------------------

def test_cap_keying_is_documented_as_stage_recipe_seed_with_the_stage_operative():
    note = P.CAP_KEYING_NOTE
    assert "(stage, recipe, seed)" in note
    assert "7,200" in note and "1,200" in note
    assert "900" in note
    lowered = note.lower()
    assert "stage cap" in lowered
    assert "arithmetic" in lowered and "defect" in lowered
    # And the arithmetic the note asserts is actually true of the sealed table.
    screen = P.stage("screen")
    assert len(screen.seeds) * screen.max_recipes * screen.per_seed_safety_cap_s == 9600
    assert 9600 > screen.wall_cap_s == 7200
    stage_total = sum(P.stage(key).wall_cap_s for key in P.STAGE_ORDER)
    assert P.TOTAL_WALL_CAP_S - stage_total == P.RESERVE_WALL_CAP_S == 900


def test_incomplete_recipe_is_never_scored_as_zero():
    """Behavioural. A censored or missing seed makes the comparison incomplete."""
    seeds = P.stage("screen").seeds
    censored = P.RecipeOutcome(recipe="g990_e010", stage="screen", seeds=(
        _seed(seeds[0], 3.0, 0.5),
        _seed(seeds[1], 0.0, 0.0, complete=False),
    ))
    assert censored.complete is False
    with pytest.raises(P.IncompleteRecipeError):
        P.recipe_rank(censored)
    with pytest.raises(P.IncompleteRecipeError):
        P.screen_eligible(censored)

    missing = P.RecipeOutcome(recipe="g998_e010", stage="screen",
                              seeds=(_seed(seeds[0], 3.0, 0.5),))
    assert missing.complete is False
    assert missing.scheduled_seeds == seeds and missing.reported_seeds == (seeds[0],)
    with pytest.raises(P.IncompleteRecipeError):
        P.recipe_rank(missing)


def test_screen_promotion_promotes_at_most_two_ranked_by_duration_first():
    """Behavioural, hand computed.

    Eligible: g998_e000 median 1.5, g990_e010 median 1.25, g990_e000 median 0.5.
    Ineligible: g998_e010 median 1.125 with no survival.
    Cap of two drops the weakest eligible recipe (g990_e000).
    """
    outcomes = [
        _screen("g990_e010", (1.0, 1.5), (0.0, 0.0)),    # median 1.25 -> eligible
        _screen("g998_e010", (1.0, 1.25), (0.0, 0.0)),   # median 1.125 -> ineligible
        _screen("g990_e000", (0.5, 0.5), (0.0625, 0.0)),  # survival branch -> eligible
        _screen("g998_e000", (1.5, 1.5), (0.0, 0.0)),    # median 1.5 -> eligible
    ]
    decision = P.promote("screen", outcomes)
    assert decision.promoted == ("g998_e000", "g990_e010")
    assert len(decision.promoted) <= P.SCREEN_MAX_PROMOTED
    assert decision.stop_for_review is False
    assert decision.decision == "promote"
    assert decision.ineligible == ("g998_e010",)
    assert decision.incomplete == ()
    assert [name for name, _ in decision.ranked] == [
        "g998_e000", "g990_e010", "g990_e000",
    ]


def test_refine_promotion_promotes_exactly_one_winner():
    outcomes = [
        _refine("g990_e010", (1.5, 2.5), (0.0, 0.0)),   # median 2.0 -> eligible
        _refine("g998_e010", (1.5, 2.0), (0.125, 0.0)),  # median 1.75 -> eligible
    ]
    decision = P.promote("refine", outcomes)
    assert decision.promoted == ("g990_e010",)
    assert len(decision.promoted) == P.REFINE_MAX_PROMOTED
    assert decision.stop_for_review is False


def test_no_eligible_recipe_stops_for_review_and_never_pivots():
    """Behavioural: the sealed no-eligible outcome is a REVIEW stop, not a pivot."""
    outcomes = [_screen(name, (1.0, 1.0), (0.0, 0.0))
                for name in P.CANONICAL_RECIPE_ORDER]
    decision = P.promote("screen", outcomes)
    assert decision.promoted == ()
    assert decision.stop_for_review is True
    assert decision.decision == P.REVIEW_STOP
    assert set(decision.ineligible) == set(P.CANONICAL_RECIPE_ORDER)
    assert "review" in decision.reason.lower()
    # No substitute action is offered anywhere in the decision.
    blob = json.dumps(decision.as_dict()).lower()
    for forbidden in ("pivot", "fallback", "deeper", "shaped", "retry", "relabel"):
        assert forbidden not in blob


def test_incomplete_recipes_cannot_be_promoted_and_are_reported_separately():
    seeds = P.stage("screen").seeds
    outcomes = [
        P.RecipeOutcome(recipe="g990_e010", stage="screen", seeds=(
            _seed(seeds[0], 9.0, 1.0), _seed(seeds[1], 9.0, 1.0, complete=False))),
        _screen("g998_e010", (1.0, 1.0), (0.0, 0.0)),
    ]
    decision = P.promote("screen", outcomes)
    assert decision.incomplete == ("g990_e010",)
    assert "g990_e010" not in decision.promoted
    assert decision.stop_for_review is True  # no complete eligible recipe remains
    assert [name for name, _ in decision.ranked] == []


def test_promote_refuses_an_outcome_whose_seeds_are_not_the_scheduled_seeds():
    bad = P.RecipeOutcome(recipe="g990_e010", stage="screen", seeds=(
        _seed(9999, 3.0, 0.5), _seed(1002, 3.0, 0.5)))
    with pytest.raises(P.ProtocolError, match="scheduled"):
        P.promote("screen", [bad])
    with pytest.raises(P.ProtocolError):
        P.promote("screen", [_screen("g990_e010", (1.0, 1.5), (0.0, 0.0)),
                             _screen("g990_e010", (1.0, 1.5), (0.0, 0.0))])
    with pytest.raises(P.ProtocolError, match="confirm"):
        P.promote("confirm", [])


def test_survival_fraction_is_documented_as_a_fraction_not_a_count():
    assert "fraction" in P.SURVIVAL_FRACTION_NOTE.lower()
    assert P.SURVIVAL_FRACTION_NOTE.count("0.0625") >= 1
    # The selection block has 16 episodes, so a count of 1 is a fraction of 1/16.
    assert P.survival_fraction(1, P.SELECTION_EPISODES) == 0.0625
    assert P.survival_fraction(2, P.SELECTION_EPISODES) == 0.125
    with pytest.raises(P.ProtocolError):
        P.survival_fraction(17, P.SELECTION_EPISODES)


def test_in_band_seconds_has_no_summary_field_and_is_documented_as_derived():
    note = P.IN_BAND_SECONDS_NOTE
    assert "mean_in_band_seconds" in note
    assert "no" in note.lower()
    # This module holds no opinion on how 4a derives it and imports neither 4a module.
    source = Path(P.__file__).read_text(encoding="utf-8")
    assert "myoleg26_selection_v2" not in source
    assert "ppo_diagnostics" not in source


# --------------------------------------------------------------------------
# Walking gate (unchanged from v1)
# --------------------------------------------------------------------------

def test_walking_gate_is_unchanged_from_the_committed_v1_manifest():
    """Behavioural, against the frozen v1 success criteria."""
    gate = P.WALKING_GATE
    assert gate["selection_survivors"] == 15
    assert gate["confirmation_survivors"] == 29
    assert gate["mean_episode_vx_min"] == 0.8
    assert gate["mean_episode_vx_max"] == 1.2
    assert gate["mean_episode_velocity_2d_rmse_max"] == 0.3

    v1 = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))["protocol"]["success"]
    assert dict(gate) == dict(v1), "the walking gate must stay unchanged from v1"


def test_walking_gate_predicates_are_per_seed_and_separate():
    assert P.selection_gate_pass(15) is True
    assert P.selection_gate_pass(14) is False
    assert P.confirmation_gate_pass(29, 1.0, 0.2) is True
    assert P.confirmation_gate_pass(28, 1.0, 0.2) is False
    assert P.confirmation_gate_pass(32, 0.79, 0.2) is False
    assert P.confirmation_gate_pass(32, 1.21, 0.2) is False
    assert P.confirmation_gate_pass(32, 1.0, 0.31) is False
    assert P.confirmation_gate_pass(32, 0.8, 0.3) is True  # inclusive bounds


def test_episode_seed_is_retained_for_crossed_cluster_reporting():
    assert "episode_seed" in P.EMITTED_EPISODE_KEYS
    assert "crossed" in P.CLUSTER_NOTE.lower()
    assert "160" in P.CLUSTER_NOTE


# --------------------------------------------------------------------------
# The v2 arm config
# --------------------------------------------------------------------------

def test_v2_config_matches_v1_except_for_the_declared_deliberate_changes():
    """Behavioural: all other hyperparameters stay at v1."""
    v2 = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))["params"]
    v1 = yaml.safe_load(V1_CONFIG.read_text(encoding="utf-8"))["params"]

    assert v2["env"] == v1["env"]
    assert v2["network"] == v1["network"]
    assert v2["general"]["seed"] == v1["general"]["seed"]
    assert v2["general"]["device"] == v1["general"]["device"]

    differing = {key for key in set(v2["config"]) | set(v1["config"])
                 if v2["config"].get(key) != v1["config"].get(key)}
    assert differing == {"name", "max_epochs"}, differing
    assert v2["config"]["name"] == "myoleg26_ppo_v2"
    assert v2["config"]["max_epochs"] == 384
    assert v2["config"]["max_epochs"] == max(
        P.stage(key).epoch_cap_per_seed for key in P.STAGE_ORDER)
    assert v2["general"]["logdir"] == "logs/myoleg26_ppo_v2"

    # The config's own gamma/entropy defaults must name an actual sealed recipe.
    default = (v2["config"]["gamma"], v2["config"]["entropy_coef"])
    assert default in {(r.gamma, r.entropy_coef) for r in P.RECIPES.values()}
    # The fixed arm in the config agrees with the protocol constants.
    assert v2["env"]["num_actors"] == P.NUM_WORLDS
    assert v2["config"]["steps_num"] == P.CONTROLS_PER_WORLD_PER_EPOCH
    assert v2["env"]["substeps"] == P.PHYSICS_SUBSTEPS
    assert v2["env"]["episode_length"] == P.EVALUATION_HORIZON_CONTROLS


def test_protocol_digest_is_stable_and_covers_the_sealed_numbers():
    digest = P.protocol_digest()
    assert digest == P.protocol_digest()
    assert len(digest) == 64 and int(digest, 16) >= 0
    snapshot = P.protocol_snapshot()
    assert snapshot["schema_version"] == P.SCHEMA_VERSION
    assert snapshot["total_wall_cap_s"] == 28800
    assert snapshot["recipes"]["g998_e000"]["gamma"] == 0.998
    assert snapshot["stages"]["confirm"]["wall_cap_s"] == 13500
    assert json.dumps(snapshot, sort_keys=True, allow_nan=False)


def test_the_yaml_binding_gate_cannot_skip():
    """F4. PyYAML is a pinned measured dependency, so this gate must not skip.

    A skipped test that binds the sealed constants to the shipped YAML is a
    silently missing gate, not a portability courtesy. ``yaml`` is imported at
    module scope so a missing PyYAML fails loudly at collection.
    """
    assert yaml.__name__ == "yaml"
    assert "yaml" in globals()
    # The needle is split so that this assertion does not itself put the
    # forbidden call name into the file it scans.
    needle = "import" + "orskip"
    source = Path(__file__).read_text(encoding="utf-8")
    assert needle not in source, (
        "the YAML binding test must fail loudly rather than skip")
    assert "safe_load" in source


def test_module_is_stdlib_only_and_declares_no_process_control():
    source = Path(P.__file__).read_text(encoding="utf-8")
    for forbidden in ("import torch", "import warp", "import numpy",
                      "mujoco", "os.kill", "subprocess", "taskkill"):
        assert forbidden not in source, f"{forbidden} must not appear"
