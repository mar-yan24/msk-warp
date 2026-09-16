"""Sealed constants and pure derivations for the MyoLeg26 v2 PPO campaign.

This module is **data plus arithmetic**. It trains nothing, measures nothing,
qualifies nothing and claims nothing. Every number below is transcribed from the
preregistered v2 campaign brief; none is derived from a result, tuned, rounded or
invented. Changing a value here changes the preregistration, which is exactly why
the values live in one place with a stable digest (:func:`protocol_digest`).

Standard library only -- no torch, no Warp, no array library, no simulator, no
process control. It therefore imports cleanly on CPU in ``tests/unit`` and is
independent of every open backend derivative defect. It is deliberately **not**
re-exported from ``msk_warp/analysis/__init__.py``; import the submodule directly.

Three recorded source corrections are carried in the API and the docstrings,
because each one is an easy and consequential mistake:

1. ``survival_fraction`` is a **fraction**, not a survivor count. The sealed
   screen threshold of "1/16" is ``0.0625`` and the refine threshold "2/16" is
   ``0.125``; both are exactly representable in binary floating point, so the
   ``>=`` comparisons at those boundaries are exact. See
   :data:`SURVIVAL_FRACTION_NOTE`.
2. There is **no** ``mean_in_band_seconds`` field in an evaluation summary. It is
   a derived quantity; Task 4a derives it from per-episode in-band step counts.
   This module only consumes the resulting number. See
   :data:`IN_BAND_SECONDS_NOTE`.
3. The **recipe-level** rank in :func:`recipe_rank` puts *median duration first*.
   The **checkpoint-level** key ``behavior_rank_v2`` from Task 4a puts *survival
   first*. These are two different keys at two different levels of the campaign
   and the difference is deliberate, preregistered and not a defect. This module
   imports neither Task 4a module and takes plain per-seed floats instead.

Nothing here decides anything on its own: a campaign driver supplies measured
numbers and acts on the returned decision. In particular, when no recipe is
eligible, :func:`promote` returns :data:`REVIEW_STOP` and the campaign stops for
the user to decide. There is no automatic substitute action of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

SCHEMA_VERSION = "myoleg26-ppo-v2-protocol-v1"

#: Repository root, resolved from this file's location (``msk_warp/analysis/``).
ROOT = Path(__file__).resolve().parents[2]

#: The committed v1 freeze manifest, read only to re-verify reset-block disjointness.
V1_FREEZE_MANIFEST = ROOT / "msk_warp/configs/experiments/myoleg26_baseline_v1.json"

#: The v2 arm configuration. Every recipe overrides only gamma and entropy_coef.
V2_CONFIG = ROOT / "msk_warp/configs/experiments/myoleg26_ppo_v2.yaml"


# ---------------------------------------------------------------------------
# The fixed arm: identical for every recipe, seed, stage and segment
# ---------------------------------------------------------------------------

NUM_WORLDS = 64
CONTROLS_PER_WORLD_PER_EPOCH = 128
PHYSICS_SUBSTEPS = 4
FORWARD_SPEED_TARGET_MPS = 1.0
IMITATION = False
EVALUATION_HORIZON_CONTROLS = 500
EVALUATION_HORIZON_SECONDS = 4.0
EVALUATION_EPOCH_INTERVAL = 64

#: The official reward, model, reset distribution, speed target and evaluation
#: horizon are unchanged from v1, and every hyperparameter other than gamma and
#: entropy_coef stays at its v1 value.
FIXED_ARM_NOTE = (
    "64 worlds, 128 controls/world/epoch, 4 physics substeps, unchanged official "
    "reward / model / reset distribution / 1.0 m/s target / no imitation, and a "
    "500-control 4 s evaluation horizon. All other hyperparameters stay at v1."
)


# ---------------------------------------------------------------------------
# Segment limits
# ---------------------------------------------------------------------------

SEGMENT_MAX_EPOCHS = 64
SEGMENT_WORK_DEADLINE_S = 460
SEGMENT_CALL_BOUND_S = 600
TOOL_TIMEOUT_MAX_S = 600


# ---------------------------------------------------------------------------
# Wall budget
# ---------------------------------------------------------------------------

TOTAL_WALL_CAP_S = 28800
RESERVE_WALL_CAP_S = 900

#: The whole training-process wall is charged: imports and setup, rollout,
#: updates, evaluation, checkpoint IO, failed attempts and replayed work.
WALL_ACCOUNTING_NOTE = (
    "28,800 s counts whole training-process wall: imports/setup, rollout, "
    "updates, evaluation, checkpoint IO, failed attempts and replayed work. "
    "Per-seed safety caps are not additional allocations; the stage and global "
    "caps dominate. Finishing below a cap authorizes no extra recipe."
)

RESERVE_PURPOSE = (
    "reconciliation and interrupted-work overhead only; no extra recipes"
)
RESERVE_AUTHORISES_EXTRA_RECIPES = False


# ---------------------------------------------------------------------------
# Recipes: the only two hyperparameters that vary
# ---------------------------------------------------------------------------

class ProtocolError(ValueError):
    """A protocol violation. Never downgraded to a default or a warning."""


class IncompleteRecipeError(ProtocolError):
    """A recipe whose scheduled seeds are not all complete.

    Raised instead of returning a zero-valued or partially populated comparison.
    A censored or missing seed makes a comparison **incomplete, not zero**, so
    there is no code path on which a censored recipe is silently ranked last.
    """


@dataclass(frozen=True)
class Recipe:
    """One (gamma, entropy_coef) arm. Nothing else differs between arms."""

    name: str
    gamma: float
    entropy_coef: float


CANONICAL_RECIPE_ORDER = ("g990_e010", "g998_e010", "g990_e000", "g998_e000")

RECIPES = MappingProxyType({
    "g990_e010": Recipe("g990_e010", 0.990, 0.01),
    "g998_e010": Recipe("g998_e010", 0.998, 0.01),
    "g990_e000": Recipe("g990_e000", 0.990, 0),
    "g998_e000": Recipe("g998_e000", 0.998, 0),
})


def recipe(name) -> Recipe:
    """The sealed recipe called ``name``, or :class:`ProtocolError`."""
    try:
        return RECIPES[name]
    except (KeyError, TypeError):
        raise ProtocolError(
            f"unknown recipe {name!r}; the sealed set is {CANONICAL_RECIPE_ORDER}"
        ) from None


# ---------------------------------------------------------------------------
# Reset blocks
# ---------------------------------------------------------------------------

SELECTION_EPISODES = 16
CONFIRMATION_EPISODES = 32
AUDIT_EPISODES = 16

V2_RESET_BLOCKS = MappingProxyType({
    "selection_screen": tuple(range(11000, 11016)),
    "selection_refine": tuple(range(12000, 12016)),
    "selection_confirm": tuple(range(13000, 13016)),
    "confirmation": tuple(range(21000, 21032)),
    "audit": tuple(range(31000, 31016)),
})

#: Confirmation and audit blocks stay unused until checkpoint selection is locked.
CONFIRMATION_BLOCKED_UNTIL_SELECTION_LOCKED = True

_STAGE_SELECTION_BLOCK = MappingProxyType({
    "screen": "selection_screen",
    "refine": "selection_refine",
    "confirm": "selection_confirm",
})


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    """One sealed campaign stage. ``max_promoted == 0`` means the stage promotes
    nothing (stage 3 is the last stage; its winner is already chosen).

    ``per_seed_safety_cap_s`` is the cap for one scheduled **run**, keyed
    ``(stage, recipe, seed)``. It is not an additional allocation: the stage cap
    is the operative constraint, and the global cap only binds after overruns
    eat the reserve. See :data:`CAP_KEYING_NOTE` for the arithmetic.
    """

    key: str
    index: int
    epoch_cap_per_seed: int
    seeds: tuple[int, ...]
    wall_cap_s: int
    per_seed_safety_cap_s: int
    max_recipes: int
    max_promoted: int
    selection_reset_block: tuple[int, ...]


STAGE_ORDER = ("screen", "refine", "confirm")

STAGES = MappingProxyType({
    "screen": Stage(
        key="screen", index=1, epoch_cap_per_seed=128, seeds=(1001, 1002),
        wall_cap_s=7200, per_seed_safety_cap_s=1200, max_recipes=4, max_promoted=2,
        selection_reset_block=V2_RESET_BLOCKS["selection_screen"]),
    "refine": Stage(
        key="refine", index=2, epoch_cap_per_seed=256, seeds=(2001, 2002),
        wall_cap_s=7200, per_seed_safety_cap_s=2400, max_recipes=2, max_promoted=1,
        selection_reset_block=V2_RESET_BLOCKS["selection_refine"]),
    "confirm": Stage(
        key="confirm", index=3, epoch_cap_per_seed=384,
        seeds=(3001, 3002, 3003, 3004, 3005),
        wall_cap_s=13500, per_seed_safety_cap_s=3600, max_recipes=1, max_promoted=0,
        selection_reset_block=V2_RESET_BLOCKS["selection_confirm"]),
})


def _stage_spec(key) -> Stage:
    try:
        return STAGES[key]
    except (KeyError, TypeError):
        raise ProtocolError(
            f"unknown stage {key!r}; the sealed stages are {STAGE_ORDER}"
        ) from None


def stage(key) -> Stage:
    """The sealed stage called ``key``, or :class:`ProtocolError`."""
    return _stage_spec(key)


def selection_reset_block(stage_key) -> tuple[int, ...]:
    """The stage's selection reset block. Never the confirmation or audit block."""
    return stage(stage_key).selection_reset_block


# ---------------------------------------------------------------------------
# Walking gate: unchanged from v1
# ---------------------------------------------------------------------------

WALKING_GATE = MappingProxyType({
    "selection_survivors": 15,
    "confirmation_survivors": 29,
    "mean_episode_vx_min": .8,
    "mean_episode_vx_max": 1.2,
    "mean_episode_velocity_2d_rmse_max": .3,
})


# ---------------------------------------------------------------------------
# Promotion thresholds
# ---------------------------------------------------------------------------

SCREEN_MEDIAN_DURATION_S = 1.25
SCREEN_ANY_SURVIVAL_FRACTION = 1 / 16
SCREEN_MAX_PROMOTED = 2

REFINE_MEDIAN_DURATION_S = 2.0
REFINE_FALLBACK_MEDIAN_DURATION_S = 1.5
REFINE_FALLBACK_ANY_SURVIVAL_FRACTION = 2 / 16
REFINE_MAX_PROMOTED = 1

#: The four **maximised** components of the recipe-level rank, in comparison order.
RECIPE_RANK_COMPONENTS = (
    "median_duration_s",
    "median_survival_fraction",
    "median_in_band_seconds",
    "negative_median_rmse_mps",
)

#: How a complete tie on all four components is broken.
#:
#: **Lexicographically least recipe id, not declaration order.** A declaration
#: index would silently change meaning if anyone reordered
#: :data:`CANONICAL_RECIPE_ORDER`, whereas a reader can reproduce a lexicographic
#: rule from the ids alone and it is stable under reordering. The two rules do
#: disagree: sorted(CANONICAL_RECIPE_ORDER) is
#: ``g990_e000, g990_e010, g998_e000, g998_e010``, which is not the declared
#: order. In a fully degenerate all-tied outcome this promotes ``g990_e000``
#: where a declaration-index rule would have promoted ``g990_e010``.
#:
#: An exact tie on four continuous medians is near-impossible outside fully
#: degenerate data, so this rule only ever bites when every metric is identical.
RECIPE_TIE_BREAK = "lexicographically_least_recipe_id"

CAP_KEYING_NOTE = (
    "The per-seed safety cap is keyed (stage, recipe, seed) -- per scheduled "
    "*run*, not per seed across recipes. That is the intended reading and it "
    "makes the stage cap the operative constraint: stage 1 schedules 8 runs at "
    "1,200 s each, which is 9,600 s against a 7,200 s stage cap, so a run can "
    "never spend its whole per-seed cap unless the rest of the stage spends "
    "almost nothing. It also means the global cap can only become the tightest "
    "constraint after actual walls have overrun their stage allocations and "
    "eaten into the 900 s reserve, because the three stage caps sum to 27,900 s "
    "against a 28,800 s global cap. Both facts are arithmetic consequences of "
    "the sealed table, not a defect, and they are recorded here rather than "
    "worked around."
)

#: Returned by :func:`promote` when nothing is eligible. The campaign stops and
#: the user decides. There is no automatic alternative course of action.
REVIEW_STOP = "STOP_FOR_REVIEW"

SURVIVAL_FRACTION_NOTE = (
    "survival_fraction is a fraction of the block's episodes, not a survivor "
    "count. With 16 selection episodes the sealed screen threshold 1/16 is "
    "0.0625 and the refine threshold 2/16 is 0.125; both are exactly "
    "representable, so the >= comparisons are exact at the boundary."
)

IN_BAND_SECONDS_NOTE = (
    "There is no mean_in_band_seconds field in an evaluation summary. It is a "
    "derived quantity, computed elsewhere from per-episode in-band step counts "
    "times the control timestep. This module consumes the number only."
)

RANK_LEVEL_NOTE = (
    "The recipe-level rank below puts median duration first. The separate "
    "checkpoint-level key behavior_rank_v2 puts survival first. Two different "
    "keys at two different levels; the difference is deliberate, not a defect."
)

EMITTED_EPISODE_KEYS = (
    "episode_seed", "world", "length", "survived",
    "mean_forward_velocity_mps", "velocity_2d_rmse_mps", "speed_band_steps",
)

CLUSTER_NOTE = (
    "Shared reset ids are crossed clusters in the final report, so episode_seed "
    "is retained in everything emitted. 160 episodes are never 160 independent "
    "runs."
)


# ---------------------------------------------------------------------------
# Pure derivations of the fixed arm
# ---------------------------------------------------------------------------

def controls_per_epoch() -> int:
    """Control transitions attempted in one epoch across all worlds."""
    return NUM_WORLDS * CONTROLS_PER_WORLD_PER_EPOCH


def physics_steps_per_epoch() -> int:
    """Physics steps in one epoch: every control transition is 4 substeps."""
    return controls_per_epoch() * PHYSICS_SUBSTEPS


def _require_epochs(epochs) -> int:
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ProtocolError(f"epochs must be a positive int, got {epochs!r}")
    return epochs


def train_control_transitions(epochs: int) -> int:
    """Control transitions for ``epochs`` epochs of one seed."""
    return _require_epochs(epochs) * controls_per_epoch()


def train_physics_steps(epochs: int) -> int:
    """Physics steps for ``epochs`` epochs of one seed."""
    return _require_epochs(epochs) * physics_steps_per_epoch()


def evaluation_epochs(epoch_cap: int) -> tuple[int, ...]:
    """Deterministic evaluation at epoch 0 and every 64 **completed** epochs."""
    cap = _require_epochs(epoch_cap)
    if cap % EVALUATION_EPOCH_INTERVAL:
        raise ProtocolError(
            f"epoch cap {cap} is not a multiple of {EVALUATION_EPOCH_INTERVAL}"
        )
    return tuple(range(0, cap + 1, EVALUATION_EPOCH_INTERVAL))


@dataclass(frozen=True)
class SegmentRound:
    """One resumable training segment of at most 64 epochs."""

    index: int
    start_epoch: int
    end_epoch: int

    @property
    def epochs(self) -> int:
        return self.end_epoch - self.start_epoch


def segment_rounds(epoch_cap: int) -> tuple[SegmentRound, ...]:
    """Partition ``epoch_cap`` epochs into <=64-epoch segments, in order."""
    cap = _require_epochs(epoch_cap)
    rounds = []
    start = 0
    index = 0
    while start < cap:
        end = min(start + SEGMENT_MAX_EPOCHS, cap)
        rounds.append(SegmentRound(index=index, start_epoch=start, end_epoch=end))
        start = end
        index += 1
    return tuple(rounds)


# ---------------------------------------------------------------------------
# Run order (R08): balanced rotation, written down before the first launch
# ---------------------------------------------------------------------------

R08_NOTE = (
    "Canonical recipe order rotated by (seed index + segment index), with the "
    "epoch-segment round outer, then seed, then the rotated recipes. A pure "
    "function of its arguments, fixed before the first stage-1 launch. Residual "
    "ordering and budget bias is reported, not eliminated."
)


def _require_index(value, label) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProtocolError(f"{label} must be a non-negative int, got {value!r}")
    return value


def _validated_recipes(recipes) -> tuple[str, ...]:
    if recipes is None:
        return CANONICAL_RECIPE_ORDER
    names = tuple(recipes)
    if not names:
        raise ProtocolError("recipe set must not be empty")
    if len(set(names)) != len(names):
        raise ProtocolError(f"duplicate recipe in {names}")
    for name in names:
        recipe(name)
    indices = [CANONICAL_RECIPE_ORDER.index(name) for name in names]
    if indices != sorted(indices):
        raise ProtocolError(
            f"recipe set {names} is not in canonical relative order; canonical "
            f"order is {CANONICAL_RECIPE_ORDER}"
        )
    return names


def rotated_recipes(seed_index, segment_index, recipes=None) -> tuple[str, ...]:
    """The R08 rotation: canonical order rotated by (seed + segment) index.

    Pure and deterministic. A supplied subset keeps canonical relative order and
    is rotated within itself, so a single-recipe stage-3 set is the identity.
    """
    names = _validated_recipes(recipes)
    shift = (_require_index(seed_index, "seed_index")
             + _require_index(segment_index, "segment_index")) % len(names)
    return names[shift:] + names[:shift]


@dataclass(frozen=True)
class SegmentJob:
    """One scheduled (stage, segment, seed, recipe) training call."""

    stage: str
    segment_index: int
    start_epoch: int
    end_epoch: int
    seed: int
    seed_index: int
    recipe: str

    @property
    def epochs(self) -> int:
        return self.end_epoch - self.start_epoch


def run_order(stage_key, recipes=None) -> tuple[SegmentJob, ...]:
    """The stage's full launch order, epoch-segment round outermost.

    ``recipes`` must be given explicitly for the refine and confirm stages,
    whose arms are chosen by measurement rather than sealed in advance.
    """
    spec = stage(stage_key)
    if recipes is None:
        if spec.max_recipes != len(CANONICAL_RECIPE_ORDER):
            raise ProtocolError(
                f"stage {spec.key!r} needs an explicit recipe set; its arms are "
                "decided by the preceding stage, not sealed in advance"
            )
        names = CANONICAL_RECIPE_ORDER
    else:
        names = _validated_recipes(recipes)
    if len(names) > spec.max_recipes:
        raise ProtocolError(
            f"stage {spec.key!r} runs at most {spec.max_recipes} recipes, "
            f"got {len(names)}"
        )
    jobs = []
    for segment in segment_rounds(spec.epoch_cap_per_seed):
        for seed_index, seed in enumerate(spec.seeds):
            for name in rotated_recipes(seed_index, segment.index, names):
                jobs.append(SegmentJob(
                    stage=spec.key, segment_index=segment.index,
                    start_epoch=segment.start_epoch, end_epoch=segment.end_epoch,
                    seed=seed, seed_index=seed_index, recipe=name))
    return tuple(jobs)


# ---------------------------------------------------------------------------
# Measured outcomes, eligibility and ranking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedOutcome:
    """One seed's selected-checkpoint behaviour, as plain floats.

    ``duration_s`` is the selected checkpoint's mean first-episode duration and
    ``rmse_mps`` its mean per-episode 2D velocity RMSE. ``survival_fraction`` is
    a **fraction** (see :data:`SURVIVAL_FRACTION_NOTE`). ``in_band_seconds`` is a
    **derived** quantity with no summary field of its own (see
    :data:`IN_BAND_SECONDS_NOTE`).

    ``complete`` records whether the seed ran to its scheduled epoch cap. A seed
    with ``complete=False`` is censored; its numbers are never imputed, defaulted
    or treated as zero.
    """

    seed: int
    complete: bool
    duration_s: float
    survival_fraction: float
    in_band_seconds: float
    rmse_mps: float


@dataclass(frozen=True)
class RecipeOutcome:
    """One recipe's seeds at one stage. Every scheduled seed must be complete.

    ``stage`` is required, not convenience metadata: completeness is only
    definable against a schedule. A recipe reporting one of two scheduled seeds
    is **incomplete**, and without the stage there is nothing to notice the
    absent seed against. This is the difference between an incomplete comparison
    and a zero-valued one.
    """

    recipe: str
    stage: str
    seeds: tuple[SeedOutcome, ...]

    @property
    def scheduled_seeds(self) -> tuple[int, ...]:
        return _stage_spec(self.stage).seeds

    @property
    def reported_seeds(self) -> tuple[int, ...]:
        return tuple(row.seed for row in self.seeds)

    @property
    def complete(self) -> bool:
        """Every scheduled seed present exactly once and every one complete."""
        reported = self.reported_seeds
        return bool(
            len(set(reported)) == len(reported)
            and set(reported) == set(self.scheduled_seeds)
            and all(row.complete for row in self.seeds))


def _finite(value, label) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"{label} must be finite, got {number!r}")
    return number


def median(values) -> float:
    """Median of finite reals; the mean of the middle two for an even count.

    Both scheduled-seed stages have exactly two seeds, so this is their mean.
    Refuses an empty sequence and any non-finite value instead of propagating a
    NaN through a comparison.
    """
    numbers = sorted(_finite(value, "median input") for value in values)
    if not numbers:
        raise ProtocolError("median of an empty sequence is undefined")
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2


def _require_complete(outcome) -> RecipeOutcome:
    if not isinstance(outcome, RecipeOutcome):
        raise ProtocolError(f"expected a RecipeOutcome, got {outcome!r}")
    if not outcome.complete:
        raise IncompleteRecipeError(
            f"recipe {outcome.recipe!r} has a censored or missing scheduled "
            "seed; the comparison is incomplete, not zero-valued"
        )
    return outcome


def recipe_medians(outcome) -> dict:
    """The four cross-seed medians used by eligibility and ranking."""
    complete = _require_complete(outcome)
    return {
        "median_duration_s": median([row.duration_s for row in complete.seeds]),
        "median_survival_fraction": median(
            [row.survival_fraction for row in complete.seeds]),
        "median_in_band_seconds": median(
            [row.in_band_seconds for row in complete.seeds]),
        "median_rmse_mps": median([row.rmse_mps for row in complete.seeds]),
        "max_survival_fraction": max(
            _finite(row.survival_fraction, "survival_fraction")
            for row in complete.seeds),
    }


def recipe_rank(outcome) -> tuple:
    """The maximised recipe-level rank key: four components, no tie-break.

    ``(median duration, median survival, median in-band seconds, -median RMSE)``.
    Duration first -- see :data:`RANK_LEVEL_NOTE`; the checkpoint-level key
    orders its components differently on purpose.

    This is deliberately **not** a total order: two recipes that tie on all four
    metrics compare equal here. The tie-break lives in :func:`recipe_order_key`
    so that it is visible at one named site instead of being hidden as a fifth
    numeric component.
    """
    medians = recipe_medians(outcome)
    return (
        medians["median_duration_s"],
        medians["median_survival_fraction"],
        medians["median_in_band_seconds"],
        -medians["median_rmse_mps"],
    )


def recipe_order_key(outcome) -> tuple:
    """The total ordering key for promotion, for use with ascending ``sorted``.

    The four maximised components are negated so that ascending sort puts the
    better recipe first, and the **recipe id string** is appended last, so that
    a complete four-way tie is broken by the **lexicographically least id**
    (:data:`RECIPE_TIE_BREAK`).

    Lexicographic rather than declaration order: a reader can reproduce it from
    the ids alone, and it does not change meaning if
    :data:`CANONICAL_RECIPE_ORDER` is ever reordered. The id is the **last**
    component, so it can never override a metric -- a lexicographically later id
    still wins on any better metric.
    """
    name = recipe(outcome.recipe).name
    return tuple(-value for value in recipe_rank(outcome)) + (name,)


def screen_eligible(outcome) -> bool:
    """Stage-1 eligibility: median duration >= 1.25 s **or** any survival >= 1/16."""
    medians = recipe_medians(outcome)
    return bool(medians["median_duration_s"] >= SCREEN_MEDIAN_DURATION_S
                or medians["max_survival_fraction"] >= SCREEN_ANY_SURVIVAL_FRACTION)


def refine_eligible(outcome) -> bool:
    """Stage-2 eligibility: median >= 2.0 s **or** (median >= 1.5 s **and** any
    survival >= 2/16)."""
    medians = recipe_medians(outcome)
    duration = medians["median_duration_s"]
    return bool(
        duration >= REFINE_MEDIAN_DURATION_S
        or (duration >= REFINE_FALLBACK_MEDIAN_DURATION_S
            and medians["max_survival_fraction"]
            >= REFINE_FALLBACK_ANY_SURVIVAL_FRACTION))


_ELIGIBILITY = MappingProxyType({
    "screen": screen_eligible,
    "refine": refine_eligible,
})


def eligible(stage_key, outcome) -> bool:
    """Dispatch to the stage's sealed eligibility rule."""
    spec = stage(stage_key)
    try:
        rule = _ELIGIBILITY[spec.key]
    except KeyError:
        raise ProtocolError(
            f"stage {spec.key!r} promotes nothing, so it has no eligibility rule"
        ) from None
    return rule(outcome)


@dataclass(frozen=True)
class PromotionDecision:
    """The outcome of one promotion step. Records; it does not act."""

    stage: str
    decision: str
    promoted: tuple[str, ...]
    ranked: tuple
    ineligible: tuple[str, ...]
    incomplete: tuple[str, ...]
    reason: str

    @property
    def stop_for_review(self) -> bool:
        return self.decision == REVIEW_STOP

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "decision": self.decision,
            "promoted": list(self.promoted),
            "ranked": [[name, list(key)] for name, key in self.ranked],
            "ineligible": list(self.ineligible),
            "incomplete": list(self.incomplete),
            "reason": self.reason,
            "stop_for_review": self.stop_for_review,
        }


def promote(stage_key, outcomes) -> PromotionDecision:
    """Apply the stage's sealed eligibility, ranking and promotion cap.

    A recipe needs **both** scheduled seeds complete for automatic promotion. A
    censored or missing seed puts the recipe in ``incomplete``; it is never
    ranked, never imputed and never scored as zero.

    Ranking is by :func:`recipe_order_key`, so a complete four-way metric tie is
    broken by the lexicographically least recipe id (:data:`RECIPE_TIE_BREAK`).

    If no complete recipe is eligible the decision is :data:`REVIEW_STOP` and the
    campaign stops for the user. This function offers no substitute course of
    action, and none may be synthesised from its output.
    """
    spec = stage(stage_key)
    if spec.max_promoted <= 0:
        raise ProtocolError(
            f"stage {spec.key!r} promotes nothing; it is the last stage"
        )
    rows = list(outcomes)
    names = [row.recipe for row in rows]
    if len(set(names)) != len(names):
        raise ProtocolError(f"duplicate recipe outcome in {names}")
    for row in rows:
        recipe(row.recipe)
        if row.stage != spec.key:
            raise ProtocolError(
                f"recipe {row.recipe!r} carries stage {row.stage!r}, not {spec.key!r}")
        reported = row.reported_seeds
        if len(set(reported)) != len(reported):
            raise ProtocolError(
                f"recipe {row.recipe!r} reports a duplicated seed in {reported}")
        unknown = [seed for seed in reported if seed not in spec.seeds]
        if unknown:
            raise ProtocolError(
                f"recipe {row.recipe!r} reports seed(s) {unknown}, which are not "
                f"among the scheduled seeds {spec.seeds} of stage {spec.key!r}"
            )
        ordered = tuple(seed for seed in spec.seeds if seed in set(reported))
        if reported != ordered:
            raise ProtocolError(
                f"recipe {row.recipe!r} reports seeds {reported} out of the "
                f"scheduled order {ordered}"
            )

    def canonical(name):
        return CANONICAL_RECIPE_ORDER.index(name)

    incomplete = tuple(sorted((row.recipe for row in rows if not row.complete),
                              key=canonical))
    complete = [row for row in rows if row.complete]
    passed = [row for row in complete if eligible(spec.key, row)]
    failed = tuple(sorted((row.recipe for row in complete if row not in passed),
                          key=canonical))
    ranked = tuple((row.recipe, recipe_rank(row))
                   for row in sorted(passed, key=recipe_order_key))
    promoted = tuple(name for name, _ in ranked[:spec.max_promoted])

    if not promoted:
        reason = (
            f"no complete recipe met the sealed {spec.key} eligibility rule, so "
            f"the campaign STOPS FOR REVIEW and the user decides. "
            f"{len(incomplete)} recipe(s) were incomplete and were not compared."
        )
        return PromotionDecision(stage=spec.key, decision=REVIEW_STOP, promoted=(),
                                 ranked=ranked, ineligible=failed,
                                 incomplete=incomplete, reason=reason)
    reason = (
        f"promoted {len(promoted)} of {len(ranked)} eligible recipe(s) at the "
        f"{spec.key} stage cap of {spec.max_promoted}"
    )
    return PromotionDecision(stage=spec.key, decision="promote", promoted=promoted,
                             ranked=ranked, ineligible=failed,
                             incomplete=incomplete, reason=reason)


# ---------------------------------------------------------------------------
# Reset-block and gate checks
# ---------------------------------------------------------------------------

def v1_reset_ids() -> frozenset:
    """Every reset id recorded in the committed v1 freeze manifest."""
    protocol = json.loads(
        V1_FREEZE_MANIFEST.read_text(encoding="utf-8"))["protocol"]
    ids = set()
    for key in ("selection_episode_seeds", "confirmation_episode_seeds",
                "audit_episode_seeds"):
        block = protocol[key]
        if not block:
            raise ProtocolError(f"v1 manifest has no {key}")
        ids |= {int(value) for value in block}
    return frozenset(ids)


def assert_reset_blocks_disjoint() -> None:
    """Re-verify all five v2 blocks: mutually disjoint, and disjoint from v1.

    Read from the committed v1 manifest rather than a copy of it, so a change to
    the frozen artefact cannot pass unnoticed.
    """
    blocks = {name: set(ids) for name, ids in V2_RESET_BLOCKS.items()}
    if len(blocks) != 5:
        raise ProtocolError(f"expected five v2 reset blocks, got {len(blocks)}")
    names = sorted(blocks)
    for position, left in enumerate(names):
        for right in names[position + 1:]:
            shared = blocks[left] & blocks[right]
            if shared:
                raise ProtocolError(
                    f"v2 reset blocks {left} and {right} share {sorted(shared)[:4]}")
    v1 = v1_reset_ids()
    for name, ids in blocks.items():
        shared = ids & v1
        if shared:
            raise ProtocolError(
                f"v2 reset block {name} reuses v1 ids {sorted(shared)[:4]}")


def assert_confirmation_unlocked(*, selection_locked: bool) -> None:
    """Refuse the confirmation and audit blocks until selection is locked."""
    if CONFIRMATION_BLOCKED_UNTIL_SELECTION_LOCKED and not selection_locked:
        raise ProtocolError(
            "the confirmation and audit reset blocks stay unused until "
            "checkpoint selection is locked"
        )


def survival_fraction(survivors: int, episodes: int) -> float:
    """Convert a survivor count to the fraction the thresholds compare against."""
    for value, label in ((survivors, "survivors"), (episodes, "episodes")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ProtocolError(f"{label} must be a non-negative int, got {value!r}")
    if episodes == 0:
        raise ProtocolError("episodes must be positive")
    if survivors > episodes:
        raise ProtocolError(f"{survivors} survivors of only {episodes} episodes")
    return survivors / episodes


def selection_gate_pass(survivors) -> bool:
    """The unchanged selection walking gate: >= 15 of 16 survive 4 s."""
    if not isinstance(survivors, int) or isinstance(survivors, bool):
        raise ProtocolError(f"survivors must be an int, got {survivors!r}")
    return survivors >= WALKING_GATE["selection_survivors"]


def confirmation_gate_pass(survivors, mean_vx, mean_rmse) -> bool:
    """The unchanged confirmation walking gate, for one seed on its own."""
    if not isinstance(survivors, int) or isinstance(survivors, bool):
        raise ProtocolError(f"survivors must be an int, got {survivors!r}")
    vx = _finite(mean_vx, "mean_vx")
    rmse = _finite(mean_rmse, "mean_rmse")
    return bool(survivors >= WALKING_GATE["confirmation_survivors"]
                and WALKING_GATE["mean_episode_vx_min"] <= vx
                <= WALKING_GATE["mean_episode_vx_max"]
                and rmse <= WALKING_GATE["mean_episode_velocity_2d_rmse_max"])


# ---------------------------------------------------------------------------
# Snapshot and digest
# ---------------------------------------------------------------------------

def protocol_snapshot() -> dict:
    """A JSON-serialisable snapshot of every sealed number in this module."""
    return {
        "schema_version": SCHEMA_VERSION,
        "fixed_arm": {
            "num_worlds": NUM_WORLDS,
            "controls_per_world_per_epoch": CONTROLS_PER_WORLD_PER_EPOCH,
            "physics_substeps": PHYSICS_SUBSTEPS,
            "forward_speed_target_mps": FORWARD_SPEED_TARGET_MPS,
            "imitation": IMITATION,
            "evaluation_horizon_controls": EVALUATION_HORIZON_CONTROLS,
            "evaluation_horizon_seconds": EVALUATION_HORIZON_SECONDS,
            "evaluation_epoch_interval": EVALUATION_EPOCH_INTERVAL,
        },
        "segment": {
            "max_epochs": SEGMENT_MAX_EPOCHS,
            "work_deadline_s": SEGMENT_WORK_DEADLINE_S,
            "call_bound_s": SEGMENT_CALL_BOUND_S,
            "tool_timeout_max_s": TOOL_TIMEOUT_MAX_S,
        },
        "total_wall_cap_s": TOTAL_WALL_CAP_S,
        "reserve_wall_cap_s": RESERVE_WALL_CAP_S,
        "recipes": {name: {"gamma": arm.gamma, "entropy_coef": arm.entropy_coef}
                    for name, arm in RECIPES.items()},
        "canonical_recipe_order": list(CANONICAL_RECIPE_ORDER),
        "stages": {
            key: {
                "index": spec.index,
                "epoch_cap_per_seed": spec.epoch_cap_per_seed,
                "seeds": list(spec.seeds),
                "wall_cap_s": spec.wall_cap_s,
                "per_seed_safety_cap_s": spec.per_seed_safety_cap_s,
                "max_recipes": spec.max_recipes,
                "max_promoted": spec.max_promoted,
                "selection_reset_block": list(spec.selection_reset_block),
            }
            for key, spec in STAGES.items()
        },
        "reset_blocks": {name: list(ids) for name, ids in V2_RESET_BLOCKS.items()},
        "promotion": {
            "screen_median_duration_s": SCREEN_MEDIAN_DURATION_S,
            "screen_any_survival_fraction": SCREEN_ANY_SURVIVAL_FRACTION,
            "screen_max_promoted": SCREEN_MAX_PROMOTED,
            "refine_median_duration_s": REFINE_MEDIAN_DURATION_S,
            "refine_fallback_median_duration_s": REFINE_FALLBACK_MEDIAN_DURATION_S,
            "refine_fallback_any_survival_fraction":
                REFINE_FALLBACK_ANY_SURVIVAL_FRACTION,
            "refine_max_promoted": REFINE_MAX_PROMOTED,
            "rank_components": list(RECIPE_RANK_COMPONENTS),
            "tie_break": RECIPE_TIE_BREAK,
            "no_eligible_recipe": REVIEW_STOP,
        },
        "walking_gate": dict(WALKING_GATE),
        "selection_episodes": SELECTION_EPISODES,
        "confirmation_episodes": CONFIRMATION_EPISODES,
        "audit_episodes": AUDIT_EPISODES,
    }


def canonical_json(value) -> bytes:
    """Deterministic UTF-8 JSON bytes: sorted keys, no NaN, no spare whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def protocol_digest() -> str:
    """SHA256 of the canonical snapshot: the preregistration's identity."""
    return hashlib.sha256(canonical_json(protocol_snapshot())).hexdigest()
