"""Direction F: the third level of the MyoLeg26 PPO v2 protocol lineage.

The lineage is the sealed v2 protocol (:mod:`msk_warp.analysis.ppo_v2_protocol`),
then probe E (:mod:`msk_warp.analysis.ppo_v2_protocol_e`), then this module.
Each level embeds its parent's snapshot verbatim and refuses to hash if the
parent has moved, so a Direction F digest names exactly one probe-E protocol,
which names exactly one sealed protocol.

What Direction F is
-------------------
One recipe, ``g990_e010_f512``. It is the pinned yaml pair (gamma 0.99, entropy
0.01) with no overrides, trained for 512 epochs per seed on the sealed geometry
(64 worlds, 128 controls per world per epoch, 4 physics substeps). It runs on
two preregistered seeds, 4001 and 4002, as 16 interleaved 64-epoch segments,
and is evaluated at every 64-epoch boundary on its own reset block,
14000-14015.

Where it is charged
-------------------
Direction F spends the sealed ``refine`` cap as a **funding line only**. It is
not a stage and promotes nothing, and nothing it produces is a stage-2 result.
It therefore declares its own :class:`~msk_warp.analysis.ppo_v2_protocol.Stage`
record, keyed ``"refine"`` so that the budget ledger debits the refine line,
with its own seeds, reset block and epoch cap. Nothing here evaluates the
sealed refine promotion thresholds.

The launch lock
---------------
Two layers, both required:

1. :data:`AUTHORISED` is a source constant and is part of the snapshot. It ships
   ``False``. While it is anything but ``True``, :func:`assert_authorised`
   refuses before it reads a single byte. Turning it on is a reviewed source
   commit that moves the digest.
2. The user-written record :data:`AUTHORISATION_RECORD` must exist, parse, carry
   exactly the declared keys, and bind the live digest, the funding cap, the
   seeds and the preregistration hash.

The snapshot and the digest never read the preregistration file or the
authorisation record: identity is a function of source constants alone.

Standard library only. CPU only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from msk_warp.analysis import ppo_v2_protocol as _sealed
from msk_warp.analysis import ppo_v2_protocol_e as _parent

# ---------------------------------------------------------------------------
# Identity and descent
# ---------------------------------------------------------------------------

#: Bumped only by a reviewed change to what this module declares.
SCHEMA_VERSION = "myoleg26-ppo-v2f-protocol-v1"

#: The amendment's own name, recorded in the snapshot and the ledger header.
AMENDMENT_ID = "direction-f-horizon-v1"

#: The protocol name the authorisation record must quote.
PROTOCOL_NAME = "direction-f"

SEALED_MODULE = "msk_warp.analysis.ppo_v2_protocol"
PARENT_MODULE = "msk_warp.analysis.ppo_v2_protocol_e"
PARENT_SCHEMA_VERSION = _parent.SCHEMA_VERSION

#: Probe E's protocol digest, transcribed as a literal. Direction F descends
#: from this one probe-E protocol and refuses to hash under any other.
PARENT_PROTOCOL_DIGEST = (
    "7c7892554cd79fb48832b1c431652f71776ef7e05b5359d4ccfc60f8c39ab68e")

#: The sealed protocol digest that probe E descends from, transcribed as a
#: literal so that a re-parented probe E is refused by name.
GRANDPARENT_PROTOCOL_DIGEST = (
    "4c01fdc94117bd190cd44c6b6a5f411a5c7cfc7b7d3e0462a344dda43baf2169")

#: The ledger Direction F carries its opening balance from: probe E's ledger.
PARENT_LEDGER = "logs/myoleg26_ppo_v2_probe_e/budget_ledger.jsonl"

#: Direction F's own ledger and run root. Neither is shared with an ancestor.
LEDGER = "logs/myoleg26_ppo_v2_direction_f/budget_ledger.jsonl"
RUN_ROOT = "logs/myoleg26_ppo_v2_direction_f"

#: The ledger must open at the settled spend of :data:`PARENT_LEDGER`.
REQUIRES_CARRY_FORWARD = True

#: The one ledger lives at :data:`LEDGER`. The budget module refuses to create
#: it anywhere else and refuses every non-inspecting open of it anywhere else,
#: so a copied ledger is refused; the runner also keeps every launch directory
#: inside :data:`RUN_ROOT`.
BIND_LEDGER_PATH = True

#: Repository root that relative paths resolve against.
ROOT = _sealed.ROOT

DESCENDS_FROM = (
    f"probe E ({PARENT_MODULE}, digest {PARENT_PROTOCOL_DIGEST}), which descends "
    f"from the sealed v2 protocol ({SEALED_MODULE}, digest "
    f"{GRANDPARENT_PROTOCOL_DIGEST}). Both ancestors' snapshots are embedded "
    "verbatim, and neither ancestor module is modified")

DIFFERS_FROM_PARENT = (
    "one recipe, g990_e010_f512: the pinned yaml pair with no overrides and no "
    "trust-region keys, where probe E varied clip_range or ppo_epochs",
    "512 epochs per seed, run as eight 64-epoch segments per seed, where probe E "
    "ran one 64-epoch segment",
    "two preregistered seeds, 4001 and 4002, where probe E reused a stage-1 seed",
    "its own reset block, 14000-14015, where probe E reused the screen block",
    "funded from the sealed refine cap as a funding line only, where probe E was "
    "funded from the screen cap",
    "a two-layer launch lock: a source constant inside the digest and a "
    "user-written authorisation record",
)

WHY = (
    "Stage 1 of the sealed campaign produced no walking at 128 epochs per seed. "
    "Direction F asks whether that null is partly a budget null for the pinned "
    "recipe: it keeps every sealed setting except depth, and trains the pinned "
    "pair for four times as many epochs on fresh seeds and a fresh reset block. "
    "The outcome labels and their bars live in the hashed preregistration, not "
    "in this module")

AUTHORISATION = (
    "Two layers, both required. The source constant AUTHORISED ships False and "
    "is part of the snapshot, so turning it on is a reviewed commit that moves "
    "the digest. The user-written authorisation record must then bind that live "
    "digest, the refine funding line and its seconds, the two seeds and the "
    "preregistration hash. Spending the refine line is the user's decision "
    "alone; no code path here can make it")

#: How :data:`PREREGISTRATION_SHA256` is computed. The hash is transcribed, not
#: read, so identity never depends on a file being present.
PREREGISTRATION = (
    "sha256 of the preregistration bytes with every CRLF normalised to LF, "
    "transcribed here as a literal before any code for this module was written. "
    "The snapshot and digest carry the literal; only the launch lock reads the "
    "file, to refuse a drifted copy")

PREREGISTRATION_PATH = (
    "docs/research/2026-09-23-session/orchestration/"
    "direction-f-preregistration.md")
PREREGISTRATION_SHA256 = (
    "acb777df3a7e40b42ae4571f7b715b56e069ac1e5bd3e8d6053f756d55095679")

NOT_A_STAGE_2_RESULT = (
    "Direction F spends the sealed refine cap as a funding line only. It is not "
    "sealed stage 2 and not a stage-2 result: it runs a recipe nobody promoted, "
    "on seeds and a reset block that sealed stage 2 does not use, and it "
    "promotes nothing. Any output is a Direction F result under this digest")

FUNDING_NOTE = (
    "The budget ledger debits the refine line, because the budget keys every "
    "charge by stage. The Stage record declared here carries that key for "
    "funding only: its wall cap is FUNDING_CAP_S, inside the preregistered "
    "window and never above the sealed refine cap, and it promotes nothing")

SEED_CHOICE_NOTE = (
    "Seeds 4001 and 4002 are named in the preregistration before any launch, so "
    "they are preregistered rather than chosen after the fact. They are disjoint "
    "from every sealed stage's seeds and from probe E's seed")

RESET_BLOCK_NOTE = (
    "Selection reset block 14000-14015 is a new descendant block, disjoint from "
    "all five sealed v2 blocks and from every v1 reset id. Sealed stage 2's "
    "unseen selection block is not used, so a later genuine stage 2 keeps it "
    "uncontaminated. Any comparison with the stage-1 bars is cross-block and is "
    "reported as such")

GEOMETRY_NOTE = (
    "The research handoff names this direction 'fewer worlds, more epochs'. "
    "Direction F departs from that name and keeps the sealed 64 worlds. Assumed, "
    "not measured on MyoLeg26: per-step cost is latency-bound and nearly flat in "
    "world count; the only support is the Ant scaling in the SHAC speedup "
    "recipe. If so, fewer worlds would not buy more epochs at a fixed wall time; "
    "they would only shrink the batch, from 1,024 to 256 transitions per "
    "minibatch at 16 worlds, and add batch noise as a second varied factor. "
    "Depth is bought with wall time at the sealed geometry instead")

BASE_RECIPE_NOTE = (
    "g990_e010 is the pinned yaml pair: the only arm with no deviation from the "
    "frozen config, and probe E's base. It was not chosen by any stage-1 "
    "ranking. PPO reads max_epochs only as the argument to env.begin_epoch, "
    "which is a no-op for MyoLeg26, as the loop bound, and in the linear-schedule "
    "branch, which lr_schedule constant never enters. So the effective config "
    "differs from sealed g990_e010 in depth alone")

HELD_FIXED_NOTE = (
    "Held fixed: the official 26-muscle model without arms, 1.0 m/s walking "
    "without imitation, the pinned yaml and freeze, 64 worlds, 128 controls per "
    "world per epoch, 4 physics substeps, the 500-control evaluation horizon, "
    "the 64-epoch evaluation interval, the segment sizing and the walking gate")

CARRY_FORWARD_NOTE = (
    "Direction F's ledger opens at the settled spend carried from probe E's "
    "ledger, which itself carried the sealed ledger's settled spend, so no cap "
    "is reset. The sealed and probe-E ledgers cannot see Direction F's spend; "
    "no sealed or probe-E launch may run while a Direction F ledger exists")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

ProtocolError = _sealed.ProtocolError


class AmendmentError(ProtocolError):
    """A request outside what Direction F declares."""


class AuthorisationError(AmendmentError):
    """The launch lock refused: Direction F has not been authorised to run."""


# ---------------------------------------------------------------------------
# Inherited by reference: the official task and the limits
# ---------------------------------------------------------------------------

canonical_json = _sealed.canonical_json
evaluation_epochs = _sealed.evaluation_epochs
segment_rounds = _sealed.segment_rounds

V2_CONFIG = _sealed.V2_CONFIG
V2_RESET_BLOCKS = _sealed.V2_RESET_BLOCKS
WALKING_GATE = _sealed.WALKING_GATE

NUM_WORLDS = _sealed.NUM_WORLDS
CONTROLS_PER_WORLD_PER_EPOCH = _sealed.CONTROLS_PER_WORLD_PER_EPOCH
PHYSICS_SUBSTEPS = _sealed.PHYSICS_SUBSTEPS
FORWARD_SPEED_TARGET_MPS = _sealed.FORWARD_SPEED_TARGET_MPS
IMITATION = _sealed.IMITATION
EVALUATION_HORIZON_CONTROLS = _sealed.EVALUATION_HORIZON_CONTROLS
EVALUATION_HORIZON_SECONDS = _sealed.EVALUATION_HORIZON_SECONDS
EVALUATION_EPOCH_INTERVAL = _sealed.EVALUATION_EPOCH_INTERVAL

SEGMENT_MAX_EPOCHS = _sealed.SEGMENT_MAX_EPOCHS
SEGMENT_WORK_DEADLINE_S = _sealed.SEGMENT_WORK_DEADLINE_S
SEGMENT_CALL_BOUND_S = _sealed.SEGMENT_CALL_BOUND_S
TOOL_TIMEOUT_MAX_S = _sealed.TOOL_TIMEOUT_MAX_S
TOTAL_WALL_CAP_S = _sealed.TOTAL_WALL_CAP_S
RESERVE_WALL_CAP_S = _sealed.RESERVE_WALL_CAP_S


# ---------------------------------------------------------------------------
# The funding line
# ---------------------------------------------------------------------------

#: The sealed cap Direction F is funded from.
FUNDING_CAP = "refine"

#: Seconds on the funding line. A placeholder ceiling: the chosen figure must
#: sit inside [MIN_FUNDING_CAP_S, MAX_FUNDING_CAP_S].
FUNDING_CAP_S = 7200

#: ceil(15 x 373.00624229999084 + 600): fifteen probe-D-level segments plus one
#: full segment reservation, the least cap that funds all sixteen segments at
#: the worst measured segment wall.
MIN_FUNDING_CAP_S = 6196

#: The sealed refine cap. Also checked live against the sealed module.
MAX_FUNDING_CAP_S = 7200

IS_STAGE = False
PROMOTES = False

EPOCH_CAP_PER_SEED = 512
SEEDS = (4001, 4002)
SEGMENT_EPOCHS = 64
MAX_SEGMENT_INDEX = 7

#: Eight 64-epoch segments per seed, each reserving the full segment call bound.
PER_SEED_SAFETY_CAP_S = 4800

SELECTION_RESET_BLOCK = tuple(range(14000, 14016))

#: Direction F's own Stage record, keyed "refine" for funding only.
STAGE = _sealed.Stage(
    key=FUNDING_CAP, index=2, epoch_cap_per_seed=EPOCH_CAP_PER_SEED, seeds=SEEDS,
    wall_cap_s=FUNDING_CAP_S, per_seed_safety_cap_s=PER_SEED_SAFETY_CAP_S,
    max_recipes=1, max_promoted=0, selection_reset_block=SELECTION_RESET_BLOCK)

STAGES = MappingProxyType({FUNDING_CAP: STAGE})
STAGE_ORDER = (FUNDING_CAP,)


def assert_funding_line() -> None:
    """Refuse unless the funding cap and the Stage record agree and are in bounds.

    Checked at call time, never at import, so every snapshot, digest and stage
    lookup re-checks the cap against the live sealed refine cap.
    """
    cap = FUNDING_CAP_S
    if isinstance(cap, bool) or not isinstance(cap, int):
        raise AmendmentError(
            f"FUNDING_CAP_S must be an int number of seconds, got {cap!r}")
    if not MIN_FUNDING_CAP_S <= cap <= MAX_FUNDING_CAP_S:
        raise AmendmentError(
            f"FUNDING_CAP_S {cap} is outside the preregistered window "
            f"[{MIN_FUNDING_CAP_S}, {MAX_FUNDING_CAP_S}] s")
    sealed = _sealed.stage(FUNDING_CAP)
    if cap > sealed.wall_cap_s:
        raise AmendmentError(
            f"FUNDING_CAP_S {cap} exceeds the sealed refine cap "
            f"{sealed.wall_cap_s} s; a funding line may never hold more than the "
            "cap it is drawn from")
    expected = {
        "key": FUNDING_CAP,
        "index": sealed.index,
        "epoch_cap_per_seed": EPOCH_CAP_PER_SEED,
        "seeds": SEEDS,
        "wall_cap_s": cap,
        "per_seed_safety_cap_s": PER_SEED_SAFETY_CAP_S,
        "max_recipes": 1,
        "max_promoted": 0,
        "selection_reset_block": SELECTION_RESET_BLOCK,
    }
    for name, value in expected.items():
        actual = getattr(STAGE, name)
        if actual != value:
            raise AmendmentError(
                f"the Direction F STAGE record disagrees with the funding line: "
                f"{name} is {actual!r}, but the funding line declares {value!r}")


def stage(key) -> _sealed.Stage:
    """Direction F's Stage record for ``"refine"``; every other key is refused."""
    if isinstance(key, str) and key == FUNDING_CAP:
        assert_funding_line()
        return STAGE
    if isinstance(key, str) and key in _sealed.STAGES:
        raise AmendmentError(
            f"stage {key!r} is a sealed stage of {SEALED_MODULE}; amendment "
            f"{AMENDMENT_ID} charges the {FUNDING_CAP!r} funding line only and "
            "opens no other cap")
    raise AmendmentError(
        f"unknown stage {key!r}; amendment {AMENDMENT_ID} declares only "
        f"{STAGE_ORDER}")


def selection_reset_block(stage_key) -> tuple[int, ...]:
    """Direction F's selection reset block, through :func:`stage`."""
    return stage(stage_key).selection_reset_block


def assert_reset_blocks_disjoint() -> None:
    """Re-verify the sealed blocks, then Direction F's block against them and v1.

    Reads the committed v1 freeze manifest (through the sealed module), so it is
    a launch-time check, not part of the snapshot.
    """
    _sealed.assert_reset_blocks_disjoint()
    block = tuple(STAGE.selection_reset_block)
    if block != tuple(SELECTION_RESET_BLOCK):
        raise AmendmentError(
            "the Direction F STAGE record's reset block differs from "
            "SELECTION_RESET_BLOCK")
    ids = set(block)
    if len(ids) != len(block) or len(ids) != _sealed.SELECTION_EPISODES:
        raise AmendmentError(
            f"the Direction F reset block must hold {_sealed.SELECTION_EPISODES} "
            f"distinct ids, got {len(block)} entries and {len(ids)} distinct")
    for name, sealed_ids in V2_RESET_BLOCKS.items():
        shared = ids & set(sealed_ids)
        if shared:
            raise AmendmentError(
                f"the Direction F reset block overlaps the v2 block {name} at "
                f"{sorted(shared)[:4]}")
    shared = ids & _sealed.v1_reset_ids()
    if shared:
        raise AmendmentError(
            f"the Direction F reset block reuses v1 ids {sorted(shared)[:4]}")


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------

#: Direction F varies no trust-region key; the arm declares no overrides.
TRUST_REGION_KEYS = ()


@dataclass(frozen=True)
class HorizonRecipe:
    """The one Direction F arm: the pinned yaml pair, deeper."""

    name: str
    gamma: float
    entropy_coef: float
    epochs_per_seed: int
    role: str
    varies: str

    @property
    def overrides(self) -> dict:
        """No yaml key is overridden: the arm is the pinned config."""
        return {}

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "overrides": self.overrides,
            "epochs_per_seed": self.epochs_per_seed,
            "role": self.role,
            "varies": self.varies,
        }


PRIMARY_RECIPE = "g990_e010_f512"
CANONICAL_RECIPE_ORDER = (PRIMARY_RECIPE,)

RECIPES = MappingProxyType({
    PRIMARY_RECIPE: HorizonRecipe(
        name=PRIMARY_RECIPE, gamma=0.990, entropy_coef=0.01,
        epochs_per_seed=EPOCH_CAP_PER_SEED, role="primary",
        varies="max_epochs 128 -> 512 at the sealed geometry; nothing else"),
})


def recipe(name) -> HorizonRecipe:
    """The Direction F arm called ``name``, or a refusal naming the owner."""
    if isinstance(name, str) and name in RECIPES:
        return RECIPES[name]
    if isinstance(name, str) and name in _sealed.RECIPES:
        raise AmendmentError(
            f"{name!r} is a sealed stage-1 arm of {SEALED_MODULE}, not an arm of "
            f"amendment {AMENDMENT_ID}. Running it here would put a stage-1 arm "
            "under a Direction F digest, where a reader could mistake the result "
            "for a stage-1 result")
    if isinstance(name, str) and name in _parent.RECIPES:
        raise AmendmentError(
            f"{name!r} is an arm of amendment {_parent.AMENDMENT_ID} "
            f"({PARENT_MODULE}), not an arm of amendment {AMENDMENT_ID}. It "
            "belongs to probe E's protocol and probe E's ledger")
    raise AmendmentError(
        f"unknown amendment recipe {name!r}; amendment {AMENDMENT_ID} declares "
        f"{CANONICAL_RECIPE_ORDER}")


#: Module-level alias, because :func:`assert_launchable` takes a ``recipe``
#: keyword that would otherwise shadow the lookup.
_lookup_recipe = recipe


def assert_recipe_names_disjoint() -> None:
    """Refuse if any Direction F arm shares a name with a sealed or probe-E arm."""
    collisions = sorted(set(RECIPES) & (set(_sealed.RECIPES) | set(_parent.RECIPES)))
    if collisions:
        raise AmendmentError(
            f"amendment arms {collisions} collide with sealed or probe-E arm "
            "names; disjoint names are what make every launch unambiguous about "
            "which protocol and which ledger it belongs to")


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------

def run_order() -> tuple[tuple[int, int], ...]:
    """The 16 (segment round, seed) cells: round outer, seed inner."""
    return tuple((segment.index, seed)
                 for segment in segment_rounds(STAGE.epoch_cap_per_seed)
                 for seed in STAGE.seeds)


def cell_arithmetic() -> dict:
    """Epochs, segments, evaluations and training work for the whole cell."""
    epochs = STAGE.epoch_cap_per_seed
    seeds = len(STAGE.seeds)
    segments = len(segment_rounds(epochs))
    evaluations = len(evaluation_epochs(epochs))
    controls = _sealed.train_control_transitions(epochs)
    steps = _sealed.train_physics_steps(epochs)
    return {
        "epochs_per_seed": epochs,
        "seeds": seeds,
        "segments_per_seed": segments,
        "segments_total": segments * seeds,
        "evaluations_per_seed": evaluations,
        "evaluations_total": evaluations * seeds,
        "train_control_transitions_per_seed": controls,
        "train_physics_steps_per_seed": steps,
        "train_control_transitions_total": controls * seeds,
        "train_physics_steps_total": steps * seeds,
    }


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def assert_launchable(*, stage, recipe, seed, segment_index, epochs=None) -> None:
    """Refuse anything outside the 16 declared cells, then apply the launch lock.

    Called by the campaign runner's plan resolution before any reservation, so a
    refusal here leaves no ledger row. Each clause has its own message. The lock
    comes last, so a malformed request is named for what is wrong with it.
    """
    if stage != FUNDING_CAP:
        raise AmendmentError(
            f"amendment {AMENDMENT_ID} charges the {FUNDING_CAP!r} funding line "
            f"only, not {stage!r}; the screen and confirm caps stay untouched")
    arm = _lookup_recipe(recipe)
    if not _is_int(seed) or seed not in SEEDS:
        raise AmendmentError(
            f"seed {seed!r} is not one of the preregistered Direction F seeds "
            f"{SEEDS}; another seed would be an unpreregistered replication")
    if not _is_int(segment_index) or not 0 <= segment_index <= MAX_SEGMENT_INDEX:
        raise AmendmentError(
            f"segment {segment_index!r} is outside the Direction F segments "
            f"0..{MAX_SEGMENT_INDEX}; there are "
            f"{len(segment_rounds(EPOCH_CAP_PER_SEED))} segments per seed")
    if not _is_int(epochs) or epochs != SEGMENT_EPOCHS:
        raise AmendmentError(
            f"epochs {epochs!r} is not the full {SEGMENT_EPOCHS}-epoch segment "
            "every Direction F cell runs; the evaluation boundaries are defined "
            "at whole segments")
    if arm.name not in RECIPES:            # unreachable; kept as a hard floor
        raise AmendmentError(f"{arm.name!r} is not a Direction F arm")
    assert_authorised()


# ---------------------------------------------------------------------------
# The launch lock
# ---------------------------------------------------------------------------

AUTHORISED = False

AUTHORISATION_RECORD = "logs/myoleg26_ppo_v2_direction_f/user_authorisation.json"
AUTHORISATION_RECORD_SCHEMA_VERSION = "myoleg26-direction-f-authorisation-v1"
AUTHORISATION_RECORD_KEYS = frozenset({
    "schema_version", "authorised", "protocol", "amendment_id",
    "protocol_digest", "preregistration_sha256", "funding_cap", "budget_s",
    "seeds", "authorised_by", "decided_utc", "statement",
})

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_UTC_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(\.[0-9]{1,6})?(Z|\+00:00)")


def _resolve(path) -> Path:
    return Path(ROOT) / str(path)


def lf_normalised_sha256(data: bytes) -> str:
    """sha256 of ``data`` with every CRLF normalised to LF."""
    return hashlib.sha256(bytes(data).replace(b"\r\n", b"\n")).hexdigest()


def live_preregistration_sha256() -> str:
    """The LF-normalised sha256 of the preregistration file as it is now."""
    target = _resolve(PREREGISTRATION_PATH)
    if not target.is_file():
        raise AuthorisationError(
            f"preregistration file {target} does not exist; the launch lock "
            "compares its bytes with the transcribed hash")
    return lf_normalised_sha256(target.read_bytes())


def _record_error(message: str) -> AuthorisationError:
    return AuthorisationError(
        f"authorisation record {AUTHORISATION_RECORD}: {message}")


def _field_error(key: str, requirement: str, value) -> AuthorisationError:
    return AuthorisationError(
        f"authorisation record field {key!r} {requirement}, got {value!r} "
        f"(in {AUTHORISATION_RECORD})")


def _refuse_duplicate_keys(pairs) -> dict:
    record = {}
    for key, value in pairs:
        if key in record:
            raise _record_error(
                f"duplicate key {key!r}; a record that says two things says "
                "nothing")
        record[key] = value
    return record


def _refuse_non_finite(name):
    raise _record_error(f"non-finite constant {name} is not a JSON number")


def _is_blank(value) -> bool:
    return not isinstance(value, str) or not value.strip()


def assert_authorised() -> str:
    """Refuse unless both layers of the launch lock are open.

    Takes no arguments. The source constant is checked first, before any read,
    so an unauthorised build never touches the record at all. On success it
    returns the sha256 of the exact record bytes it read and verified, so a
    caller can name them without a second, unverified read of the file.
    """
    if AUTHORISED is not True:
        raise AuthorisationError(
            f"Direction F is locked: AUTHORISED is {AUTHORISED!r} in the source. "
            "Turning it on is a reviewed source commit that moves the protocol "
            "digest, and a user-written authorisation record must then bind the "
            "new digest")

    target = _resolve(AUTHORISATION_RECORD)
    if not target.is_file():
        raise _record_error(
            f"{target} does not exist; only the user writes this record, and "
            "without it nothing launches")
    raw = target.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _record_error(f"not valid UTF-8 ({error})") from None
    try:
        record = json.loads(text, object_pairs_hook=_refuse_duplicate_keys,
                            parse_constant=_refuse_non_finite)
    except json.JSONDecodeError as error:
        raise _record_error(f"not valid JSON ({error})") from None
    if not isinstance(record, dict):
        raise _record_error(
            f"must be a JSON object, got {type(record).__name__}")

    keys = set(record)
    missing = sorted(AUTHORISATION_RECORD_KEYS - keys)
    if missing:
        raise _record_error(f"missing keys {missing}")
    unexpected = sorted(keys - AUTHORISATION_RECORD_KEYS)
    if unexpected:
        raise _record_error(f"unexpected keys {unexpected}")

    value = record["schema_version"]
    if not isinstance(value, str) or value != AUTHORISATION_RECORD_SCHEMA_VERSION:
        raise _field_error("schema_version",
                           f"must be {AUTHORISATION_RECORD_SCHEMA_VERSION!r}", value)
    value = record["authorised"]
    if value is not True:
        raise _field_error("authorised", "must be the JSON literal true", value)
    value = record["protocol"]
    if not isinstance(value, str) or value != PROTOCOL_NAME:
        raise _field_error("protocol", f"must be {PROTOCOL_NAME!r}", value)
    value = record["amendment_id"]
    if not isinstance(value, str) or value != AMENDMENT_ID:
        raise _field_error("amendment_id", f"must be {AMENDMENT_ID!r}", value)
    value = record["funding_cap"]
    if not isinstance(value, str) or value != FUNDING_CAP:
        raise _field_error("funding_cap", f"must be {FUNDING_CAP!r}", value)
    value = record["budget_s"]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value != FUNDING_CAP_S):
        raise _field_error("budget_s",
                           f"must be the funding cap {FUNDING_CAP_S!r} s", value)
    value = record["seeds"]
    if (not isinstance(value, list) or not all(_is_int(seed) for seed in value)
            or tuple(value) != SEEDS):
        raise _field_error("seeds", f"must be exactly {list(SEEDS)}", value)
    value = record["authorised_by"]
    if _is_blank(value):
        raise _field_error("authorised_by", "must be a non-blank string", value)
    value = record["decided_utc"]
    if not isinstance(value, str) or not _UTC_PATTERN.fullmatch(value):
        raise _field_error("decided_utc", "must be an ISO-8601 UTC timestamp",
                           value)
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise _field_error("decided_utc", "must be a real ISO-8601 UTC time",
                           value) from None
    value = record["statement"]
    if _is_blank(value):
        raise _field_error("statement", "must be a non-blank string", value)
    value = record["preregistration_sha256"]
    if not isinstance(value, str) or value != PREREGISTRATION_SHA256:
        raise _field_error("preregistration_sha256",
                           f"must be the transcribed {PREREGISTRATION_SHA256!r}",
                           value)
    value = record["protocol_digest"]
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise _field_error("protocol_digest",
                           "must be a 64-character lowercase hex string", value)

    assert_preregistered()
    live_sha = live_preregistration_sha256()
    if live_sha != PREREGISTRATION_SHA256:
        raise AuthorisationError(
            f"preregistration file {_resolve(PREREGISTRATION_PATH)} has drifted: "
            f"its LF-normalised sha256 is {live_sha}, but Direction F transcribes "
            f"{PREREGISTRATION_SHA256}")

    live_digest = protocol_digest()
    if record["protocol_digest"] != live_digest:
        raise _field_error(
            "protocol_digest",
            f"must bind the live Direction F digest {live_digest}",
            record["protocol_digest"])
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Snapshot, provenance and digest
# ---------------------------------------------------------------------------

def assert_parent_unchanged() -> None:
    """Refuse unless the live probe E is the one this descends from."""
    actual = _parent.protocol_digest()
    if actual != PARENT_PROTOCOL_DIGEST:
        raise AmendmentError(
            f"amendment {AMENDMENT_ID} descends from probe-E protocol digest "
            f"{PARENT_PROTOCOL_DIGEST}, but {PARENT_MODULE} now hashes to "
            f"{actual}. Direction F descends from ONE probe-E protocol and will "
            "not silently re-parent itself onto another")
    if _parent.PARENT_PROTOCOL_DIGEST != GRANDPARENT_PROTOCOL_DIGEST:
        raise AmendmentError(
            f"amendment {AMENDMENT_ID} records the sealed grandparent digest "
            f"{GRANDPARENT_PROTOCOL_DIGEST}, but {PARENT_MODULE} declares "
            f"{_parent.PARENT_PROTOCOL_DIGEST} as its parent")


def assert_preregistered() -> None:
    """Refuse unless the transcribed preregistration hash and path are well formed."""
    sha = PREREGISTRATION_SHA256
    if not isinstance(sha, str) or not _SHA256_PATTERN.fullmatch(sha):
        raise AmendmentError(
            f"the Direction F preregistration hash must be 64 lowercase hex "
            f"characters, got {sha!r}")
    path = PREREGISTRATION_PATH
    if not isinstance(path, str) or not path.strip():
        raise AmendmentError(
            f"the Direction F preregistration path must be a non-empty string, "
            f"got {path!r}")


def _funding() -> dict:
    return {
        "cap": FUNDING_CAP,
        "cap_s": FUNDING_CAP_S,
        "min_cap_s": MIN_FUNDING_CAP_S,
        "max_cap_s": MAX_FUNDING_CAP_S,
        "is_stage": IS_STAGE,
        "promotes": PROMOTES,
    }


def _stage_dict() -> dict:
    return {
        "key": STAGE.key,
        "index": STAGE.index,
        "epoch_cap_per_seed": STAGE.epoch_cap_per_seed,
        "seeds": list(STAGE.seeds),
        "wall_cap_s": STAGE.wall_cap_s,
        "per_seed_safety_cap_s": STAGE.per_seed_safety_cap_s,
        "max_recipes": STAGE.max_recipes,
        "max_promoted": STAGE.max_promoted,
        "selection_reset_block": list(STAGE.selection_reset_block),
    }


def amendment_snapshot() -> dict:
    """Everything that makes Direction F what it is, and nothing measured."""
    assert_preregistered()
    assert_funding_line()
    return {
        "amendment_id": AMENDMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "parent_module": PARENT_MODULE,
        "parent_schema_version": PARENT_SCHEMA_VERSION,
        "parent_protocol_digest": PARENT_PROTOCOL_DIGEST,
        "grandparent_protocol_digest": GRANDPARENT_PROTOCOL_DIGEST,
        "descends_from": DESCENDS_FROM,
        "differs_from_parent": list(DIFFERS_FROM_PARENT),
        "why": WHY,
        "authorisation": AUTHORISATION,
        "authorised": AUTHORISED,
        "authorisation_record": {
            "path": AUTHORISATION_RECORD,
            "schema_version": AUTHORISATION_RECORD_SCHEMA_VERSION,
            "keys": sorted(AUTHORISATION_RECORD_KEYS),
        },
        "not_a_stage_2_result": NOT_A_STAGE_2_RESULT,
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": PREREGISTRATION_SHA256,
            "hashing": PREREGISTRATION,
        },
        "seed_choice": SEED_CHOICE_NOTE,
        "seed_preregistered": True,
        "held_fixed": HELD_FIXED_NOTE,
        "funding_note": FUNDING_NOTE,
        "funding": _funding(),
        "stage": _stage_dict(),
        "reset_block": RESET_BLOCK_NOTE,
        "geometry": GEOMETRY_NOTE,
        "base_recipe": BASE_RECIPE_NOTE,
        "trust_region_keys": list(TRUST_REGION_KEYS),
        "recipes": {name: arm.as_dict() for name, arm in RECIPES.items()},
        "canonical_recipe_order": list(CANONICAL_RECIPE_ORDER),
        "primary_recipe": PRIMARY_RECIPE,
        "cell": {
            "stage": FUNDING_CAP,
            "recipe": PRIMARY_RECIPE,
            "seeds": list(STAGE.seeds),
            "segments": len(segment_rounds(STAGE.epoch_cap_per_seed)),
            "max_segment_index": MAX_SEGMENT_INDEX,
            "segment_epochs": SEGMENT_EPOCHS,
            "selection_reset_block": list(STAGE.selection_reset_block),
            "run_order": [list(cell) for cell in run_order()],
            "evaluation_epochs": list(evaluation_epochs(STAGE.epoch_cap_per_seed)),
            "arithmetic": cell_arithmetic(),
        },
        "ledger": {
            "path": LEDGER,
            "run_root": RUN_ROOT,
            "separate_from": PARENT_LEDGER,
            "requires_carry_forward": REQUIRES_CARRY_FORWARD,
            "bind_ledger_path": BIND_LEDGER_PATH,
            "carry_forward_note": CARRY_FORWARD_NOTE,
        },
    }


def ledger_provenance() -> dict:
    """What Direction F's ledger header records about its descent."""
    assert_preregistered()
    assert_funding_line()
    return {
        "amendment_id": AMENDMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "parent_module": PARENT_MODULE,
        "parent_protocol_digest": PARENT_PROTOCOL_DIGEST,
        "grandparent_protocol_digest": GRANDPARENT_PROTOCOL_DIGEST,
        "parent_ledger": PARENT_LEDGER,
        "descends_from": DESCENDS_FROM,
        "differs_from_parent": list(DIFFERS_FROM_PARENT),
        "why": WHY,
        "not_a_stage_2_result": NOT_A_STAGE_2_RESULT,
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "sha256": PREREGISTRATION_SHA256,
        },
        "funding": _funding(),
        "seed_choice": SEED_CHOICE_NOTE,
        "seed_preregistered": True,
        "carry_forward_note": CARRY_FORWARD_NOTE,
    }


def protocol_snapshot() -> dict:
    """Direction F's snapshot, embedding probe E's snapshot verbatim.

    The invariants are checked here rather than trusted, so no digest can be
    produced for a re-parented, colliding, unpreregistered or out-of-window
    Direction F.
    """
    assert_parent_unchanged()
    assert_recipe_names_disjoint()
    assert_preregistered()
    assert_funding_line()
    return {
        "schema_version": SCHEMA_VERSION,
        "amendment": amendment_snapshot(),
        "parent_protocol_digest": PARENT_PROTOCOL_DIGEST,
        "inherited_protocol": _parent.protocol_snapshot(),
    }


def protocol_digest() -> str:
    """SHA256 of the canonical Direction F snapshot: this protocol's identity."""
    return hashlib.sha256(canonical_json(protocol_snapshot())).hexdigest()
