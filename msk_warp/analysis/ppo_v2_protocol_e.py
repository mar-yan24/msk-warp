"""The probe-E trust-region AMENDMENT to the sealed MyoLeg26 v2 PPO protocol.

This module is a **descendant** of :mod:`msk_warp.analysis.ppo_v2_protocol`, not
a replacement for it and not a second copy of it. It is data plus arithmetic:
standard library only, it trains nothing, measures nothing, qualifies nothing,
grants no budget and claims no result.

What it is for
--------------
The direction review measured a median ``clip_fraction`` of 0.584 with 976 of
1,024 updates above 0.5, which would mean roughly half of every minibatch
carries no gradient. Whether that is a real obstacle or an artifact of clipping a
26-dimensional joint ratio is **inferred, not settled**, and probe E is the
discriminator. Probe E could not be expressed inside the sealed protocol: the two
keys it needs to vary live only in the freeze-pinned yaml, and adding a fifth arm
to the sealed module would move the sealed digest that stage 1's ledger asserts
on every launch. This amendment is the authorised way out of that: a **new
protocol version, with its own digest and its own ledger**, leaving the sealed
protocol module unedited and stage 1's sealed ledger byte-intact and valid.

Descent, stated so that no reader can confuse the two
-----------------------------------------------------
* :data:`PARENT_PROTOCOL_DIGEST` is the one sealed protocol this amendment
  descends from. :func:`assert_parent_unchanged` refuses if the live sealed
  module no longer hashes to it, and :func:`protocol_snapshot` calls that check
  before it will produce a snapshot at all.
* The snapshot **embeds the parent snapshot verbatim**, so the descent is
  cryptographic rather than documentary: any change to the sealed protocol moves
  this amendment's digest too, and the amendment's own ledger then refuses.
* The recipe name sets are **disjoint** (:func:`assert_recipe_names_disjoint`).
  The sealed protocol does not know the amendment's arms, so a sealed launch
  cannot charge one to a stage-1 cell; and this module refuses the sealed four by
  name, so a stage-1 arm cannot be run on the amendment's ledger.
* The amendment has its own ledger (:data:`LEDGER`) and **requires a
  carried-forward opening balance** from the sealed one
  (:data:`REQUIRES_CARRY_FORWARD`). A fresh ledger would reset the caps, and a
  new ledger must never be a way of escaping a charge already made.

What is NOT changed, and may not be changed here
------------------------------------------------
The official reward, the official model, the reset distribution, the 1.0 m/s
speed target, the 4 s / 500-control evaluation horizon, the five reset blocks,
the segment limits, the wall caps, the stage table and the walking gate are all
**inherited by reference** from the sealed module. gamma stays 0.990 and
entropy_coef stays 0.01 -- the v1-equivalent pair, identical to the sealed
``g990_e010`` arm -- so the trust region is the single varied factor. The
variation is expressed as an **override of the pinned configuration**
(:data:`TRUST_REGION_KEYS`), never as new algorithm code.

The preregistered advance/stop criteria are deliberately **not** restated in this
module. They were fixed before any data in
``probes-DE-preregistration.md`` **section 4 (E.1-E.4)** -- not section 2.3, which
two upstream documents cite in error -- and are read from there.

One parameter here is **not** preregistered: the cell's seed. It is a post-hoc,
data-informed choice and is labelled as such in :data:`SEED_CHOICE_NOTE`, which
the amendment snapshot, the ledger header provenance and the freeze manifest all
carry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType

from msk_warp.analysis import ppo_v2_protocol as _parent

SCHEMA_VERSION = "myoleg26-ppo-v2e-protocol-v1"

#: The amendment's stable identity. It appears in the amended ledger's header,
#: in the v2 freeze manifest and in every report, so that a probe-E artefact
#: names itself.
AMENDMENT_ID = "probe-e-trust-region-v1"


# ---------------------------------------------------------------------------
# Descent from the sealed protocol
# ---------------------------------------------------------------------------

PARENT_MODULE = "msk_warp.analysis.ppo_v2_protocol"
PARENT_SCHEMA_VERSION = _parent.SCHEMA_VERSION

#: The ONE sealed protocol digest this amendment descends from, transcribed from
#: the stage-1 ledger header and from the committed v2 freeze manifest. It is a
#: literal on purpose: a computed value would re-parent itself silently.
PARENT_PROTOCOL_DIGEST = (
    "4c01fdc94117bd190cd44c6b6a5f411a5c7cfc7b7d3e0462a344dda43baf2169")

#: The sealed ledger the amendment's opening balance is carried forward from. It
#: is read, never written, extended, migrated or rewritten.
PARENT_LEDGER = "logs/myoleg26_ppo_v2/budget_ledger.jsonl"

#: The amendment's own ledger and run root. Separate files, so stage 1's sealed
#: ledger stays byte-intact.
LEDGER = "logs/myoleg26_ppo_v2_probe_e/budget_ledger.jsonl"
RUN_ROOT = "logs/myoleg26_ppo_v2_probe_e"

#: A new ledger must not reset a cap. The amended ledger opens at the sealed
#: ledger's settled spend, so every remaining figure is unchanged by this
#: amendment.
REQUIRES_CARRY_FORWARD = True

DESCENDS_FROM = (
    "the sealed MyoLeg26 v2 PPO campaign protocol "
    f"{PARENT_MODULE} (schema {PARENT_SCHEMA_VERSION}, digest "
    f"{PARENT_PROTOCOL_DIGEST}), which is NOT edited by this amendment and whose "
    "sealed ledger stays byte-intact and continues to validate")

DIFFERS_FROM_PARENT = (
    "adds one trust-region arm beyond the sealed four, varying exactly one of "
    "clip_range or ppo_epochs as an override of the freeze-pinned yaml, with "
    "gamma 0.990 and entropy_coef 0.01 held at their v1-equivalent values",
    "carries its own protocol digest, so every launch under it is refused by the "
    "sealed ledger and every sealed launch is refused by this amendment's ledger",
    "runs on a SEPARATE ledger whose opening balance is carried forward from the "
    "sealed ledger, so no cap is reset and no prior charge is escaped",
    "authorises exactly one cell -- one 64-epoch screen segment on one seed -- "
    "and refuses a resume segment, a second seed, another stage and every sealed "
    "stage-1 arm by name",
)

WHY = (
    "The direction review measured clip_fraction median 0.584 with 976 of 1,024 "
    "updates above 0.5, replicated independently and worse in probe D (64 of 64 "
    "updates above 0.5). Whether that is gradient starvation or an artifact of "
    "clipping a 26-dimensional joint ratio is INFERRED, not settled, and it is "
    "the only measured optimiser anomaly on file. The sealed harness cannot vary "
    "either trust-region key, so the question needs a protocol amendment rather "
    "than a code change.")

AUTHORISATION = (
    "Explicitly authorised by the user as the PROBE-E PROTOCOL AMENDMENT, after "
    "the direction-C forensics refuted the premise on which probe E had been "
    "demoted. This module authorises no launch by itself: it grants no budget, "
    "promotes nothing and starts nothing.")

NOT_A_STAGE_1_RESULT = (
    "Any result produced under this amendment is a PROBE-E result, never a "
    "stage-1 result. It is one cell, one seed, one 64-epoch segment on a reset "
    "block crossed with stage 1's, so it supports no interval, no ordering and "
    "no promotion, and it is not comparable to a completed 128-epoch stage-1 "
    "run. The sealed stage-1 screen result -- four recipes, two seeds, zero "
    "eligible, STOP_FOR_REVIEW -- is untouched by it.")

PREREGISTRATION = (
    "docs/research/2026-09-14-myoleg-session/orchestration/"
    "probes-DE-preregistration.md section 4 (E.1-E.4), written before any data. "
    "The mechanism and behaviour criteria live there and are deliberately not "
    "restated, paraphrased or re-derived in this module.")

HELD_FIXED_NOTE = (
    "Inherited by reference from the sealed protocol and unchanged: the official "
    "reward, the official model, the reset distribution, the 1.0 m/s forward "
    "speed target, no imitation, the 500-control 4 s evaluation horizon, the five "
    "reset blocks, the stage table and its seeds and caps, the segment limits, "
    "the wall caps and the walking gate.")

SEED_CHOICE_NOTE = (
    "SEED 1001 WAS NOT PREREGISTERED, and it is a POST-HOC, DATA-INFORMED "
    "choice, labelled as such rather than presented as neutral. E.2 fixed 'one "
    "seed' and did not name it; 1001 was chosen at implementation time, after "
    "stage 1 had run, on two stated reasons -- it matches probe D's cell, and it "
    "is the seed whose selected duration rose with training in 4 of 4 stage-1 "
    "recipes. THE SECOND REASON IS A STAGE-1 OUTCOME: a property measured after "
    "the data existed. It is not threshold-seeking -- no criterion moved, "
    "nothing was relaxed, and probe E's primary readout (the clip-fraction "
    "median) is seed-independent, so the exposure is bounded -- but it is an "
    "outcome-informed selection and must be read as one. It is open to "
    "objection BEFORE the launch; changing it AFTER the launch would be "
    "selection on an outcome and is refused. One seed supports no interval and "
    "no ordering, which is a named confound of E.2 and is not repaired by this "
    "choice."
)

CARRY_FORWARD_NOTE = (
    "The amended ledger's header records the sealed ledger's path, sha256, row "
    "count and protocol digest together with its settled totals, and every "
    "remaining figure is computed against those totals. Creating the amended "
    "ledger therefore changes no budget: it reproduces the sealed ledger's "
    "remaining global and stage figures exactly, and only the amendment's own "
    "(stage, recipe, seed) run key opens at a fresh per-seed allowance, because "
    "it is a cell that has never run.")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

#: Re-exported so a caller can duck-type this module in place of the sealed one.
ProtocolError = _parent.ProtocolError


class AmendmentError(ProtocolError):
    """An amendment violation.

    A subclass of the sealed :class:`~msk_warp.analysis.ppo_v2_protocol.
    ProtocolError` on purpose: every existing ``except P.ProtocolError`` handler
    in the campaign runner keeps producing a clean refusal rather than a
    traceback.
    """


# ---------------------------------------------------------------------------
# The trust-region arm: one varied factor, expressed as configuration
# ---------------------------------------------------------------------------

#: The only two keys an amendment arm may override in the freeze-pinned yaml.
#: Both are read by PPO straight from that configuration, so varying them needs
#: no algorithm change at all.
TRUST_REGION_KEYS = ("clip_range", "ppo_epochs")


@dataclass(frozen=True)
class TrustRegionRecipe:
    """One amendment arm: the sealed v1-equivalent pair plus one override.

    ``clip_range`` and ``ppo_epochs`` are ``None`` when the arm leaves the pinned
    value alone. Exactly one of them is set, which is what makes the trust region
    the single varied factor.
    """

    name: str
    gamma: float
    entropy_coef: float
    clip_range: float | None
    ppo_epochs: int | None
    role: str
    varies: str

    @property
    def overrides(self) -> dict:
        """The configuration keys this arm overrides. Never any other key."""
        values = {"clip_range": self.clip_range, "ppo_epochs": self.ppo_epochs}
        return {key: value for key, value in values.items() if value is not None}

    def as_dict(self) -> dict:
        return {"gamma": self.gamma, "entropy_coef": self.entropy_coef,
                "clip_range": self.clip_range, "ppo_epochs": self.ppo_epochs,
                "role": self.role, "varies": self.varies,
                "overrides": self.overrides}


#: The primary arm, and the declared alternative. Both were preregistered in E.2;
#: the primary is the one the amendment's default cell names.
PRIMARY_RECIPE = "g990_e010_c040"
DECLARED_ALTERNATIVE_RECIPE = "g990_e010_k001"

CANONICAL_RECIPE_ORDER = (PRIMARY_RECIPE, DECLARED_ALTERNATIVE_RECIPE)

RECIPES = MappingProxyType({
    PRIMARY_RECIPE: TrustRegionRecipe(
        name=PRIMARY_RECIPE, gamma=0.990, entropy_coef=0.01,
        clip_range=0.4, ppo_epochs=None, role="primary",
        varies="clip_range 0.2 -> 0.4: widen the trust region"),
    DECLARED_ALTERNATIVE_RECIPE: TrustRegionRecipe(
        name=DECLARED_ALTERNATIVE_RECIPE, gamma=0.990, entropy_coef=0.01,
        clip_range=None, ppo_epochs=1, role="declared_alternative",
        varies="ppo_epochs 5 -> 1: re-enter the trust region once, not five "
               "times. NOT the same intervention as widening it."),
})

#: At most one cell may be launched under this amendment. The alternative exists
#: because the primary intervention may turn out to be ineffective; it is not a
#: second run to be added to the first.
MAX_CELLS = 1


def recipe(name) -> TrustRegionRecipe:
    """The amendment arm called ``name``, or a refusal that says where to go."""
    if isinstance(name, str) and name in RECIPES:
        return RECIPES[name]
    if isinstance(name, str) and name in _parent.RECIPES:
        raise AmendmentError(
            f"{name!r} is a sealed stage-1 arm of {PARENT_MODULE}, not an arm of "
            f"amendment {AMENDMENT_ID}. Run it under the sealed protocol and the "
            f"sealed ledger {PARENT_LEDGER}; running it here would put a stage-1 "
            "arm under a probe-E digest, where a reader could mistake the result "
            "for a stage-1 result")
    raise AmendmentError(
        f"unknown amendment recipe {name!r}; amendment {AMENDMENT_ID} declares "
        f"{CANONICAL_RECIPE_ORDER}")


#: Module-level alias, because :func:`assert_launchable` takes a ``recipe``
#: keyword that would otherwise shadow the lookup.
_lookup_recipe = recipe


def assert_recipe_names_disjoint() -> None:
    """Refuse if any amendment arm shares a name with a sealed stage-1 arm."""
    collisions = sorted(set(RECIPES) & set(_parent.RECIPES))
    if collisions:
        raise AmendmentError(
            f"amendment arms {collisions} collide with sealed stage-1 arm names; "
            "disjoint names are what make every launch unambiguous about which "
            "protocol and which ledger it belongs to")


# ---------------------------------------------------------------------------
# Inherited by reference: the official task, the stages and the limits
# ---------------------------------------------------------------------------

stage = _parent.stage
selection_reset_block = _parent.selection_reset_block
segment_rounds = _parent.segment_rounds
evaluation_epochs = _parent.evaluation_epochs
assert_reset_blocks_disjoint = _parent.assert_reset_blocks_disjoint
canonical_json = _parent.canonical_json

STAGES = _parent.STAGES
STAGE_ORDER = _parent.STAGE_ORDER
V2_RESET_BLOCKS = _parent.V2_RESET_BLOCKS
WALKING_GATE = _parent.WALKING_GATE
V2_CONFIG = _parent.V2_CONFIG

NUM_WORLDS = _parent.NUM_WORLDS
CONTROLS_PER_WORLD_PER_EPOCH = _parent.CONTROLS_PER_WORLD_PER_EPOCH
PHYSICS_SUBSTEPS = _parent.PHYSICS_SUBSTEPS
FORWARD_SPEED_TARGET_MPS = _parent.FORWARD_SPEED_TARGET_MPS
IMITATION = _parent.IMITATION
EVALUATION_HORIZON_CONTROLS = _parent.EVALUATION_HORIZON_CONTROLS
EVALUATION_HORIZON_SECONDS = _parent.EVALUATION_HORIZON_SECONDS
EVALUATION_EPOCH_INTERVAL = _parent.EVALUATION_EPOCH_INTERVAL

SEGMENT_MAX_EPOCHS = _parent.SEGMENT_MAX_EPOCHS
SEGMENT_WORK_DEADLINE_S = _parent.SEGMENT_WORK_DEADLINE_S
SEGMENT_CALL_BOUND_S = _parent.SEGMENT_CALL_BOUND_S
TOOL_TIMEOUT_MAX_S = _parent.TOOL_TIMEOUT_MAX_S
TOTAL_WALL_CAP_S = _parent.TOTAL_WALL_CAP_S
RESERVE_WALL_CAP_S = _parent.RESERVE_WALL_CAP_S


# ---------------------------------------------------------------------------
# The one authorised cell
# ---------------------------------------------------------------------------

#: The inherited stage the cell charges. Probe E is a screening-scale diagnostic
#: of the optimiser, so it charges the screen stage, not refine or confirm.
AMENDMENT_STAGE = "screen"

#: The one seed. **NOT preregistered** -- see :data:`SEED_CHOICE_NOTE`, which
#: labels it as a post-hoc, data-informed choice rather than a neutral one.
AMENDMENT_SEEDS = (1001,)

AMENDMENT_SEGMENT_INDEX = 0
MAX_SEGMENT_INDEX = 0
AMENDMENT_EPOCHS = 64


@dataclass(frozen=True)
class AmendmentCell:
    """The one (stage, recipe, seed, segment) cell this amendment authorises."""

    stage: str
    recipe: str
    seed: int
    segment_index: int
    start_epoch: int
    end_epoch: int
    selection_reset_block: tuple

    @property
    def epochs(self) -> int:
        return self.end_epoch - self.start_epoch

    def as_dict(self) -> dict:
        return {"stage": self.stage, "recipe": self.recipe, "seed": self.seed,
                "segment_index": self.segment_index,
                "start_epoch": self.start_epoch, "end_epoch": self.end_epoch,
                "epochs": self.epochs,
                "selection_reset_block": list(self.selection_reset_block),
                "seeds": list(AMENDMENT_SEEDS),
                "max_cells": MAX_CELLS,
                "max_segment_index": MAX_SEGMENT_INDEX}


def amendment_cell(recipe_name=None) -> AmendmentCell:
    """The authorised cell for one amendment arm. Defaults to the primary arm."""
    arm = recipe(PRIMARY_RECIPE if recipe_name is None else recipe_name)
    return AmendmentCell(
        stage=AMENDMENT_STAGE, recipe=arm.name, seed=AMENDMENT_SEEDS[0],
        segment_index=AMENDMENT_SEGMENT_INDEX, start_epoch=0,
        end_epoch=AMENDMENT_EPOCHS,
        selection_reset_block=selection_reset_block(AMENDMENT_STAGE))


def assert_launchable(*, stage, recipe, seed, segment_index, epochs=None) -> None:
    """Refuse anything outside the one authorised cell.

    Called by the campaign runner's plan resolution before any reservation, so a
    refusal here leaves no ledger row at all. Each clause is its own path: a
    resume segment would double the charge, a second seed would be an
    unpreregistered replication, another stage would open an untouched cap, and a
    sealed arm belongs to the sealed protocol and the sealed ledger.
    """
    if stage != AMENDMENT_STAGE:
        raise AmendmentError(
            f"amendment {AMENDMENT_ID} charges the {AMENDMENT_STAGE!r} stage "
            f"only, not {stage!r}; the refine and confirm caps stay untouched")
    arm = _lookup_recipe(recipe)
    if seed not in AMENDMENT_SEEDS:
        raise AmendmentError(
            f"seed {seed!r} is not the amendment's authorised seed "
            f"{AMENDMENT_SEEDS}; a second seed would be an unpreregistered "
            "replication with its own charge")
    if segment_index != AMENDMENT_SEGMENT_INDEX:
        raise AmendmentError(
            f"amendment {AMENDMENT_ID} authorises one 64-epoch segment at index "
            f"{AMENDMENT_SEGMENT_INDEX}, not segment {segment_index!r}; a resume "
            "segment would double the charge and was not preregistered")
    if epochs is not None and int(epochs) != AMENDMENT_EPOCHS:
        raise AmendmentError(
            f"the authorised cell is {AMENDMENT_EPOCHS} epochs, not {epochs!r}; "
            "the behavioural readout is defined at that boundary")
    if arm.name not in RECIPES:            # unreachable; kept as a hard floor
        raise AmendmentError(f"{arm.name!r} is not an amendment arm")


# ---------------------------------------------------------------------------
# Snapshot, provenance and digest
# ---------------------------------------------------------------------------

def assert_parent_unchanged() -> None:
    """Refuse unless the live sealed protocol is the one this descends from."""
    actual = _parent.protocol_digest()
    if actual != PARENT_PROTOCOL_DIGEST:
        raise AmendmentError(
            f"amendment {AMENDMENT_ID} descends from sealed protocol digest "
            f"{PARENT_PROTOCOL_DIGEST}, but {PARENT_MODULE} now hashes to "
            f"{actual}. The amendment is a descendant of ONE sealed protocol and "
            "will not silently re-parent itself onto another")


def amendment_snapshot() -> dict:
    """Everything that makes this amendment what it is, and nothing measured."""
    cell = amendment_cell()
    return {
        "amendment_id": AMENDMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "parent_module": PARENT_MODULE,
        "parent_schema_version": PARENT_SCHEMA_VERSION,
        "parent_protocol_digest": PARENT_PROTOCOL_DIGEST,
        "descends_from": DESCENDS_FROM,
        "differs_from_parent": list(DIFFERS_FROM_PARENT),
        "why": WHY,
        "authorisation": AUTHORISATION,
        "not_a_stage_1_result": NOT_A_STAGE_1_RESULT,
        "preregistration": PREREGISTRATION,
        "seed_choice": SEED_CHOICE_NOTE,
        "seed_preregistered": False,
        "held_fixed": HELD_FIXED_NOTE,
        "trust_region_keys": list(TRUST_REGION_KEYS),
        "recipes": {name: arm.as_dict() for name, arm in RECIPES.items()},
        "canonical_recipe_order": list(CANONICAL_RECIPE_ORDER),
        "primary_recipe": PRIMARY_RECIPE,
        "declared_alternative_recipe": DECLARED_ALTERNATIVE_RECIPE,
        "cell": cell.as_dict(),
        "ledger": {
            "path": LEDGER,
            "run_root": RUN_ROOT,
            "separate_from": PARENT_LEDGER,
            "requires_carry_forward": REQUIRES_CARRY_FORWARD,
            "carry_forward_note": CARRY_FORWARD_NOTE,
        },
    }


def ledger_provenance() -> dict:
    """What the amended ledger's own header records about its descent."""
    return {
        "amendment_id": AMENDMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "parent_module": PARENT_MODULE,
        "parent_protocol_digest": PARENT_PROTOCOL_DIGEST,
        "parent_ledger": PARENT_LEDGER,
        "descends_from": DESCENDS_FROM,
        "differs_from_parent": list(DIFFERS_FROM_PARENT),
        "why": WHY,
        "not_a_stage_1_result": NOT_A_STAGE_1_RESULT,
        "seed_choice": SEED_CHOICE_NOTE,
        "seed_preregistered": False,
        "carry_forward_note": CARRY_FORWARD_NOTE,
    }


def protocol_snapshot() -> dict:
    """The amendment's snapshot, embedding the sealed snapshot verbatim.

    The two invariants are checked here rather than trusted, so no digest can be
    produced for a re-parented or a colliding amendment.
    """
    assert_parent_unchanged()
    assert_recipe_names_disjoint()
    return {
        "schema_version": SCHEMA_VERSION,
        "amendment": amendment_snapshot(),
        "parent_protocol_digest": PARENT_PROTOCOL_DIGEST,
        "inherited_protocol": _parent.protocol_snapshot(),
    }


def protocol_digest() -> str:
    """SHA256 of the canonical amended snapshot: this amendment's identity."""
    return hashlib.sha256(canonical_json(protocol_snapshot())).hexdigest()
