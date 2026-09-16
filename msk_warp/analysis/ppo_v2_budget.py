"""One canonical append-only, hash-linked wall-time ledger for the v2 campaign.

The 28,800 s training budget is the scarcest thing in the project and it cannot
be recovered once spent, so this module is written to **fail closed** in every
ambiguous case rather than to keep a campaign moving.

The contract
------------

1. **The reservation is taken before anything expensive.** ``reserve`` appends a
   durable row, and only then runs the caller's ``create`` callable. Nothing a
   caller might do -- creating an output directory, importing the backend,
   compiling the model, building the environment, starting the child -- can
   happen before the reservation exists on disk.
2. **A pending, unsettled reservation blocks the next job.** Work is foreground
   and sequential; there are never two open reservations. The block is derived
   from the file, not from memory, so it survives a process restart.
3. **Settlement charges the actual full process wall** -- setup, IO, evaluation,
   failed segments and replayed work alike. A forecast is not an authorisation:
   an actual wall past the reserved bound is charged in full and flagged
   ``overran_bound``.
4. **An unknown or crashed duration is charged at the full reserved bound**
   (:func:`BudgetLedger.settle_unknown`). Unknown work is never charged as zero.
4b. **A second durable marker is written immediately before the child starts**
   (:func:`BudgetLedger.mark_spawn`), so the ledger distinguishes *reserved but
   never spawned* from *spawned, outcome unknown*. A **zero** charge
   (:func:`BudgetLedger.settle_aborted`) is admissible **only** when no spawn
   marker exists -- positive evidence that no process was ever started. Once the
   marker is present, the full reserved bound is the floor.
5. **There is no fresh-ledger bypass.** A missing, empty, malformed, truncated,
   non-monotonic or hash-broken ledger refuses every launch, and ``create``
   refuses to overwrite whatever is already there. The way past a broken ledger
   is a human reconciling it, not this module inventing a recovery.
6. **Global, stage and per-seed caps are all enforced**, not a per-process
   timeout alone, and the tightest one binds.
7. **No launch without an explicit shutdown allowance** that still fits inside
   the remaining caps and inside the hard call bound.

What this module does not do
----------------------------

It owns **no process control at all**: see :data:`PROCESS_CONTROL_NOTE`. It also
makes **no timing claim and no causal claim** from a contention observation: see
:data:`CONTENTION_NOTE`.

Standard library only -- no torch, no Warp, no array library, no simulator. It is
deliberately **not** re-exported from ``msk_warp/analysis/__init__.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time

from msk_warp.analysis import ppo_v2_protocol as P

SCHEMA_VERSION = "myoleg26-ppo-v2-budget-v1"
LOCK_SCHEMA_VERSION = "myoleg26-ppo-v2-lock-v1"

#: The ledger lives under ``logs/``, which is git-ignored: budget evidence is
#: preserved on disk and never committed.
DEFAULT_RUN_DIR = "logs/myoleg26_ppo_v2"
LEDGER_FILENAME = "budget_ledger.jsonl"

GENESIS_PREV_SHA256 = "0" * 64

HEADER_KIND = "header"
RESERVE_KIND = "reserve"

#: Written immediately before the child is actually started. Its presence is the
#: only admissible evidence that a process existed; its absence is the only
#: admissible evidence that none did.
SPAWN_KIND = "spawn"
SETTLE_KIND = "settle"
_KINDS = (HEADER_KIND, RESERVE_KIND, SPAWN_KIND, SETTLE_KIND)

CHARGE_ACTUAL = "actual_process_wall"
CHARGE_RESERVED_BOUND = "reserved_bound_unknown_duration"

#: A zero charge, admissible **only** when no spawn marker exists for the
#: reservation -- that is, with positive evidence that no child ever started.
CHARGE_ABORTED_PRESPAWN = "aborted_before_spawn_zero_charge"

LEDGER_TAIL_NOTE = (
    "A hash chain protects history, not the tail. Removing the LAST line of an "
    "append-only file leaves a self-consistent chain, so a fresh process cannot "
    "prove from the file alone that no further record ever existed. Two things "
    "narrow this: a live ledger object refuses a file that has shrunk or whose "
    "already-observed rows changed, and excising a spawn marker from the middle "
    "of history breaks the index and the chain. The residual gap -- tail "
    "truncation observed only across processes -- is disclosed here rather than "
    "claimed to be detected."
)

CONTENTION_OBSERVED = "observed"
CONTENTION_UNOBSERVED = "unobserved"
CONTENTION_PROBE_FAILED = "probe_failed"
CONTENTION_PROBE_MALFORMED = "probe_malformed"

CONTENTION_NOTE = (
    "A contention observation is recorded provenance and nothing else. The same "
    "unchanged GPU suite self-reported 68.32 s and 60.48 s on this host with no "
    "code change while ordinary desktop applications shared the device, so a "
    "wall time measured under contention is contaminated and must be visible as "
    "such in the record instead of being averaged into a result silently. This "
    "module therefore applies no threshold of its own, refuses no launch on "
    "contention, and makes no timing claim: no cause is claimed for any observed "
    "spread, and two observations establish no distribution. The probe supplies "
    "its own contended verdict; an unobserved, failed or malformed probe records "
    "contended as null, which means unknown and never means absent."
)

PROCESS_CONTROL_NOTE = (
    "This module owns no process control. It starts nothing, waits on nothing "
    "and will never kill, enumerate or signal any process. In particular it does "
    "not probe liveness by sending a null operating-system signal to a process "
    "id: on Windows that is not the harmless no-op it is on POSIX. Child "
    "timeouts belong to the runner, which acts only through its own child "
    "handle and never touches another job."
)

RECIPE_LOCK_KIND = "recipe"
SELECTION_LOCK_KIND = "checkpoint_selection"

RECIPE_LOCK_REQUIRED_INPUTS = ("protocol_sha256", "stage2_evidence_sha256")

#: A recipe lock is written *before* stage 3 exists, so it must not be able to
#: demand a stage-3 checkpoint or evaluation hash that cannot exist yet.
RECIPE_LOCK_FORBIDDEN_INPUTS = (
    "selected_checkpoints", "stage3_checkpoints", "stage3_checkpoint_sha256",
    "evaluations",
)

SELECTION_LOCK_REQUIRED_INPUTS = (
    "protocol_sha256", "selected_checkpoints", "evaluations",
)

_LOCK_KINDS = (RECIPE_LOCK_KIND, SELECTION_LOCK_KIND)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


# ---------------------------------------------------------------------------
# Errors. Every one of them is a refusal, never a warning or a default.
# ---------------------------------------------------------------------------

class BudgetError(Exception):
    """Base class. Any of these means the launch or settlement did not happen."""


class LedgerIntegrityError(BudgetError):
    """The ledger is missing, unreadable or does not verify. Fails closed."""


class BudgetExhaustedError(BudgetError):
    """A global, stage or per-seed cap cannot accommodate the reservation."""


class PendingReservationError(BudgetError):
    """An earlier reservation is still unsettled, so no new job may start."""


class SettlementError(BudgetError):
    """A settlement that does not correspond to the pending reservation."""


class DoubleSettlementError(SettlementError):
    """A reservation that has already been settled. Refused, never merged."""


class ShutdownAllowanceError(BudgetError):
    """No explicit shutdown/checkpoint allowance, or none that fits."""


class PrelaunchError(BudgetError):
    """The caller's exclusive pre-launch creation step failed.

    The reservation is already durable by this point and an append-only ledger
    cannot retract a row, so the reservation stays **pending** and blocks every
    further launch until it is settled explicitly. That is the fail-closed
    outcome; the alternative would be a launch the ledger never recorded.
    """


class LockError(BudgetError):
    """A lock file that is absent, already present, malformed or drifted."""


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------

def canonical_json(value) -> bytes:
    """Deterministic UTF-8 JSON bytes: sorted keys, no NaN, no spare whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def record_digest(body) -> str:
    """SHA256 over a record body, which must exclude its own digest field."""
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _utc(posix_time: float) -> str:
    return datetime.datetime.fromtimestamp(
        posix_time, datetime.timezone.utc).isoformat()


def _finite(value, label, *, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError(f"{label} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise BudgetError(f"{label} must be finite, got {value!r}")
    if minimum is not None and number < minimum:
        raise BudgetError(f"{label} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise BudgetError(f"{label} must be <= {maximum}, got {number}")
    return number


def _count(value, label) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BudgetError(f"{label} must be a non-negative int, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Contention: recorded, never gated
# ---------------------------------------------------------------------------

def unobserved_contention() -> dict:
    """The observation recorded when no contention probe was supplied."""
    return {"status": CONTENTION_UNOBSERVED, "contended": None,
            "detail": "no contention probe supplied at launch"}


def _observe_contention(probe) -> dict:
    """Run the injected probe. Never raises, never refuses, never thresholds."""
    if probe is None:
        return unobserved_contention()
    try:
        observation = probe()
    except Exception as exc:  # a probe failure must not cost a launch
        return {"status": CONTENTION_PROBE_FAILED, "contended": None,
                "error": f"{type(exc).__name__}: {exc}"}
    contended = None
    if isinstance(observation, Mapping) and "contended" in observation:
        value = observation["contended"]
        if isinstance(value, bool) or value is None:
            record = dict(observation)
            record["status"] = CONTENTION_OBSERVED
            record["contended"] = value
            try:
                canonical_json(record)
            except (TypeError, ValueError):
                # The verdict itself was well formed, so it is kept; only the
                # surrounding payload could not be encoded, and it is recorded
                # as a repr. A validly reported verdict is never downgraded to
                # unknown just because a sibling key was unserialisable.
                contended = value
            else:
                return record
    return {"status": CONTENTION_PROBE_MALFORMED, "contended": contended,
            "observation": repr(observation)}


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentCounters:
    """What a segment actually did. Attempted, completed and replayed separately.

    ``replayed_*`` is work redone after an interruption; it is real cost and is
    recorded, not netted off. ``partial_update_cost_s`` is wall spent inside an
    update that did not complete.
    """

    attempted_control_transitions: int
    completed_control_transitions: int
    replayed_control_transitions: int
    attempted_physics_steps: int
    completed_physics_steps: int
    replayed_physics_steps: int
    partial_update_cost_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "attempted_control_transitions":
                _count(self.attempted_control_transitions,
                       "attempted_control_transitions"),
            "completed_control_transitions":
                _count(self.completed_control_transitions,
                       "completed_control_transitions"),
            "replayed_control_transitions":
                _count(self.replayed_control_transitions,
                       "replayed_control_transitions"),
            "attempted_physics_steps":
                _count(self.attempted_physics_steps, "attempted_physics_steps"),
            "completed_physics_steps":
                _count(self.completed_physics_steps, "completed_physics_steps"),
            "replayed_physics_steps":
                _count(self.replayed_physics_steps, "replayed_physics_steps"),
            "partial_update_cost_s":
                _finite(self.partial_update_cost_s, "partial_update_cost_s",
                        minimum=0.0),
        }


@dataclass(frozen=True)
class Reservation:
    """A durable, exclusive claim on wall time, taken before anything runs."""

    index: int
    reservation_id: str
    stage: str
    recipe: str
    seed: int
    segment_index: int
    start_epoch: int
    end_epoch: int
    reserved_bound_s: float
    shutdown_allowance_s: float
    contention: dict = field(default_factory=unobserved_contention)

    @property
    def held_s(self) -> float:
        """The conservative hold: the bound plus the shutdown allowance."""
        return self.reserved_bound_s + self.shutdown_allowance_s


_CAP_ORDER = ("global", "stage", "seed")


@dataclass(frozen=True)
class Remaining:
    """Remaining wall under each cap. The tightest one binds.

    ``seed_s`` is the per-seed safety cap of one scheduled (recipe, seed) run,
    keyed ``(stage, recipe, seed)``, which is what the sealed table allocates.
    It is not an additional allocation: the stage and global caps dominate it.
    Stage 1 schedules eight runs at 1,200 s against a 7,200 s stage cap, so the
    stage cap is the operative constraint; and because the three stage caps sum
    to 27,900 s against 28,800 s, the global cap only becomes the tightest once
    actual walls have overrun their stages and eaten the 900 s reserve. That is
    arithmetic from the sealed table, not a defect.
    """

    global_s: float
    stage_s: float
    seed_s: float

    @property
    def least(self) -> float:
        return min(self.global_s, self.stage_s, self.seed_s)

    @property
    def binding(self) -> str:
        least = self.least
        for name in _CAP_ORDER:
            if getattr(self, f"{name}_s") == least:
                return name
        raise BudgetError("unreachable: no cap equals the minimum")

    @property
    def call_cap(self) -> float:
        """min(600 s, every remaining cap): a ceiling on bound **plus** allowance.

        This is **not** itself a reservable ``reserved_bound_s``. ``reserve``
        requires ``reserved_bound_s + shutdown_allowance_s`` to fit inside both
        this value and the 600 s call bound, so the shutdown allowance must come
        out of it: the largest admissible bound is
        ``call_cap - shutdown_allowance_s``. A caller passing ``call_cap``
        straight through as ``reserved_bound_s`` is refused.
        """
        return min(float(P.SEGMENT_CALL_BOUND_S), self.least)

    def as_dict(self) -> dict:
        return {"global_s": self.global_s, "stage_s": self.stage_s,
                "seed_s": self.seed_s, "binding": self.binding}


def _run_key(stage: str, recipe: str, seed: int) -> str:
    return f"{stage}|{recipe}|{seed}"


def _reservation_id(stage, recipe, seed, segment_index, index) -> str:
    return f"{stage}-{recipe}-s{seed}-g{segment_index}-r{index}"


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

class BudgetLedger:
    """The one canonical append-only hash-linked ledger for a campaign.

    Every mutating call re-reads and re-verifies the whole file first, so any
    external edit between calls is caught before more budget is committed.
    """

    def __init__(self, path, *, clock=time.time, contention_probe=None):
        self.path = Path(path)
        self._clock = clock
        self._probe = contention_probe
        self._rows: tuple = ()

    # -- construction -----------------------------------------------------

    @classmethod
    def create(cls, path, *, protocol_digest, clock=time.time,
               contention_probe=None) -> "BudgetLedger":
        """Create the one ledger, refusing to touch anything already present."""
        ledger = cls(path, clock=clock, contention_probe=contention_probe)
        digest = str(protocol_digest)
        if not _SHA256_RE.match(digest):
            raise LedgerIntegrityError(
                f"protocol_digest must be a sha256 hex digest, got {digest!r}")
        posix_time = _finite(ledger._clock(), "clock")
        body = {
            "index": 0, "kind": HEADER_KIND, "prev_sha256": GENESIS_PREV_SHA256,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "schema_version": SCHEMA_VERSION, "protocol_digest": digest,
            "total_wall_cap_s": P.TOTAL_WALL_CAP_S,
            "reserve_wall_cap_s": P.RESERVE_WALL_CAP_S,
            "segment_max_epochs": P.SEGMENT_MAX_EPOCHS,
            "segment_work_deadline_s": P.SEGMENT_WORK_DEADLINE_S,
            "segment_call_bound_s": P.SEGMENT_CALL_BOUND_S,
        }
        row = dict(body, record_sha256=record_digest(body))
        try:
            with ledger.path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        except FileExistsError as exc:
            raise LedgerIntegrityError(
                f"a ledger already exists at {ledger.path}; it is never "
                "overwritten, replaced or started fresh"
            ) from exc
        ledger._rows = (row,)
        return ledger

    @classmethod
    def open(cls, path, *, clock=time.time, contention_probe=None) -> "BudgetLedger":
        """Open and fully verify an existing ledger, or refuse."""
        ledger = cls(path, clock=clock, contention_probe=contention_probe)
        ledger._load()
        return ledger

    # -- verification -----------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            raise LedgerIntegrityError(
                f"ledger {self.path} does not exist; a missing ledger refuses "
                "every launch and is never created implicitly")
        text = self.path.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise LedgerIntegrityError(
                f"ledger {self.path} is empty, which is invalid rather than new")
        rows = []
        previous = GENESIS_PREV_SHA256
        previous_time = None
        pending_id = None
        spawned_id = None
        for number, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise LedgerIntegrityError(
                    f"ledger {self.path} line {number + 1} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} is not a JSON object")
            if row.get("index") != number:
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} has index {row.get('index')!r}, "
                    f"expected {number}: the index is not monotonic")
            if row.get("prev_sha256") != previous:
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} prev_sha256 {row.get('prev_sha256')!r} "
                    f"does not link to {previous!r}: the hash chain is broken")
            stored = row.get("record_sha256")
            body = {key: value for key, value in row.items()
                    if key != "record_sha256"}
            try:
                recomputed = record_digest(body)
            except (TypeError, ValueError) as exc:
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} cannot be re-encoded: {exc}") from exc
            if stored != recomputed:
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} record_sha256 {stored!r} does not "
                    f"match its content ({recomputed!r}): the row was altered")
            posix_time = row.get("posix_time")
            if not isinstance(posix_time, (int, float)) or isinstance(posix_time, bool):
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} has no usable posix_time")
            if previous_time is not None and posix_time < previous_time:
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} posix_time {posix_time} precedes "
                    f"{previous_time}: the record time is not monotonic")
            kind = row.get("kind")
            if kind not in _KINDS:
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} has unknown kind {kind!r}")
            if number == 0:
                if kind != HEADER_KIND:
                    raise LedgerIntegrityError(
                        "the first ledger record must be the header")
                if row.get("schema_version") != SCHEMA_VERSION:
                    raise LedgerIntegrityError(
                        f"ledger schema_version {row.get('schema_version')!r} is "
                        f"not {SCHEMA_VERSION!r}")
                if not _SHA256_RE.match(str(row.get("protocol_digest"))):
                    raise LedgerIntegrityError(
                        "the ledger header has no valid protocol_digest")
            elif kind == HEADER_KIND:
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} is a second header")
            elif kind == RESERVE_KIND:
                if pending_id is not None:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} opens a reservation while "
                        f"{pending_id!r} is still unsettled")
                pending_id = row.get("reservation_id")
                spawned_id = None
            elif kind == SPAWN_KIND:
                if pending_id is None:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} marks a spawn with no open "
                        "reservation")
                if row.get("reservation_id") != pending_id:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} marks a spawn for "
                        f"{row.get('reservation_id')!r}, not the open "
                        f"{pending_id!r}")
                if spawned_id is not None:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} is a second spawn marker for "
                        f"{pending_id!r}")
                spawned_id = pending_id
            else:
                if pending_id is None:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} settles nothing that is open")
                if row.get("reservation_id") != pending_id:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} settles "
                        f"{row.get('reservation_id')!r}, not the open {pending_id!r}")
                pending_id = None
                spawned_id = None
            previous = stored
            previous_time = posix_time
            rows.append(row)
        observed = self._rows
        if len(rows) < len(observed):
            raise LedgerIntegrityError(
                f"ledger {self.path} shrank from {len(observed)} to {len(rows)} "
                "records; an append-only ledger never loses a row")
        for position, prior in enumerate(observed):
            if rows[position] != prior:
                raise LedgerIntegrityError(
                    f"ledger {self.path} record {position} changed after it was "
                    "observed; an append-only ledger never rewrites a row")
        self._rows = tuple(rows)

    # -- reading ----------------------------------------------------------

    @property
    def rows(self) -> tuple:
        return self._rows

    @property
    def header(self) -> dict:
        return dict(self._rows[0])

    @property
    def protocol_digest(self) -> str:
        return self.header["protocol_digest"]

    def assert_protocol_unchanged(self, expected) -> None:
        """Refuse if the sealed protocol has moved under a running campaign."""
        if self.protocol_digest != expected:
            raise LedgerIntegrityError(
                f"ledger was opened under protocol {self.protocol_digest!r} but "
                f"the current protocol digest is {expected!r}")

    @property
    def pending(self):
        """The one open reservation, reconstructed from the file, or ``None``."""
        for row in reversed(self._rows):
            if row["kind"] == SETTLE_KIND:
                return None
            if row["kind"] == RESERVE_KIND:
                return Reservation(
                    index=row["index"], reservation_id=row["reservation_id"],
                    stage=row["stage"], recipe=row["recipe"], seed=row["seed"],
                    segment_index=row["segment_index"],
                    start_epoch=row["start_epoch"], end_epoch=row["end_epoch"],
                    reserved_bound_s=row["reserved_bound_s"],
                    shutdown_allowance_s=row["shutdown_allowance_s"],
                    contention=dict(row["contention_at_launch"]))
        return None

    def spawned(self, reservation) -> bool:
        """Whether a spawn marker exists for this reservation, read from the file.

        Never from the caller's object: a forged :class:`Reservation` cannot
        assert that no child was started.
        """
        target = getattr(reservation, "reservation_id", reservation)
        return any(row["kind"] == SPAWN_KIND
                   and row["reservation_id"] == target for row in self._rows)

    def charged(self) -> dict:
        """Settled charges, by global total, stage and (stage, recipe, seed) run."""
        totals = {"global_s": 0.0, "stage_s": {}, "run_s": {}, "settled_rows": 0}
        for row in self._rows:
            if row["kind"] != SETTLE_KIND:
                continue
            amount = float(row["charged_s"])
            totals["global_s"] += amount
            totals["stage_s"][row["stage"]] = (
                totals["stage_s"].get(row["stage"], 0.0) + amount)
            key = _run_key(row["stage"], row["recipe"], row["seed"])
            totals["run_s"][key] = totals["run_s"].get(key, 0.0) + amount
            totals["settled_rows"] += 1
        return totals

    def _checked_run(self, stage, recipe, seed):
        spec = P.stage(stage)
        arm = P.recipe(recipe)
        if seed not in spec.seeds:
            raise BudgetError(
                f"seed {seed!r} is not a scheduled seed of stage {spec.key!r} "
                f"({spec.seeds})")
        return spec, arm

    def remaining(self, stage, recipe, seed) -> Remaining:
        """Remaining wall under the global, stage and per-seed caps."""
        spec, _arm = self._checked_run(stage, recipe, seed)
        totals = self.charged()
        return Remaining(
            global_s=P.TOTAL_WALL_CAP_S - totals["global_s"],
            stage_s=spec.wall_cap_s - totals["stage_s"].get(spec.key, 0.0),
            seed_s=(spec.per_seed_safety_cap_s
                    - totals["run_s"].get(_run_key(spec.key, recipe, seed), 0.0)),
        )

    # -- writing ----------------------------------------------------------

    def _append(self, body) -> dict:
        row = dict(body, record_sha256=record_digest(body))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        self._rows = self._rows + (row,)
        return row

    def _next(self) -> tuple:
        """Next (index, timestamp, parent digest), keeping time monotonic.

        A host clock can step backwards (an NTP correction, a manual change). The
        chain invariant requires non-decreasing timestamps, so a backwards clock
        is clamped to the previous row's time rather than refused -- a clock
        adjustment must not cost a launch. The clamp is recorded as
        ``clock_clamped`` so the affected row is visible instead of silent. Note
        that charges never come from this clock: every duration is supplied by
        the caller, so a clamp cannot alter any budget arithmetic.
        """
        index = len(self._rows)
        posix_time = _finite(self._clock(), "clock")
        last = self._rows[-1]
        clamped = posix_time < last["posix_time"]
        if clamped:
            posix_time = float(last["posix_time"])
        return index, posix_time, last["record_sha256"], clamped

    def reserve(self, *, stage, recipe, seed, segment_index, start_epoch, end_epoch,
                reserved_bound_s, shutdown_allowance_s, create=None) -> Reservation:
        """Claim wall time exclusively, before anything expensive is started.

        Order is load-bearing: verify, refuse if a reservation is still open,
        check every cap, observe contention, **append the row**, and only then
        run ``create``. A caller's ``create`` may be arbitrarily expensive -- an
        output directory, a backend import, a model compile, an environment, a
        child process -- so nothing in it may precede the durable reservation.
        """
        self._load()
        open_reservation = self.pending
        if open_reservation is not None:
            raise PendingReservationError(
                f"reservation {open_reservation.reservation_id} is still "
                "unsettled; work is foreground and sequential, so no further "
                "job may start until it is settled")

        spec, _arm = self._checked_run(stage, recipe, seed)
        segment = _count(segment_index, "segment_index")
        first = _count(start_epoch, "start_epoch")
        last = _count(end_epoch, "end_epoch")
        if last <= first:
            raise BudgetError(f"end_epoch {last} must exceed start_epoch {first}")
        if last - first > P.SEGMENT_MAX_EPOCHS:
            raise BudgetError(
                f"segment spans {last - first} epochs, above the {P.SEGMENT_MAX_EPOCHS}"
                "-epoch ceiling")
        if last > spec.epoch_cap_per_seed:
            raise BudgetError(
                f"end_epoch {last} exceeds the stage {spec.key!r} epoch cap "
                f"{spec.epoch_cap_per_seed}")

        try:
            allowance = _finite(shutdown_allowance_s, "shutdown_allowance_s")
        except BudgetError as exc:
            raise ShutdownAllowanceError(str(exc)) from exc
        if allowance <= 0:
            raise ShutdownAllowanceError(
                "a launch needs an explicit positive shutdown/checkpoint "
                f"allowance, got {allowance}")
        bound = _finite(reserved_bound_s, "reserved_bound_s",
                        maximum=float(P.SEGMENT_CALL_BOUND_S))
        if bound <= 0:
            raise BudgetError(f"reserved_bound_s must be positive, got {bound}")
        if bound + allowance > P.SEGMENT_CALL_BOUND_S:
            raise ShutdownAllowanceError(
                f"a reserved bound of {bound} s plus a {allowance} s shutdown "
                f"allowance exceeds the {P.SEGMENT_CALL_BOUND_S} s call bound")

        remaining = self.remaining(spec.key, recipe, seed)
        if bound + allowance > remaining.least:
            raise BudgetExhaustedError(
                f"the {remaining.binding} cap has {remaining.least} s left, which "
                f"cannot hold a {bound} s bound plus a {allowance} s shutdown "
                f"allowance; remaining is {remaining.as_dict()}")

        contention = _observe_contention(self._probe)
        index, posix_time, previous, clamped = self._next()
        reservation_id = _reservation_id(spec.key, recipe, seed, segment, index)
        row = self._append({
            "index": index, "kind": RESERVE_KIND, "prev_sha256": previous,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "clock_clamped": clamped,
            "reservation_id": reservation_id, "stage": spec.key, "recipe": recipe,
            "seed": seed, "segment_index": segment,
            "start_epoch": first, "end_epoch": last,
            "reserved_bound_s": bound, "shutdown_allowance_s": allowance,
            "work_deadline_s": P.SEGMENT_WORK_DEADLINE_S,
            "remaining_before": remaining.as_dict(),
            "contention_at_launch": contention,
        })
        reservation = Reservation(
            index=row["index"], reservation_id=reservation_id, stage=spec.key,
            recipe=recipe, seed=seed, segment_index=segment, start_epoch=first,
            end_epoch=last, reserved_bound_s=bound, shutdown_allowance_s=allowance,
            contention=contention)
        if create is not None:
            try:
                create()
            except Exception as exc:
                # The failure provably preceded any child: this frame appended
                # the reservation and has not marked a spawn. So settle it at
                # zero rather than wedging every later job, and keep both rows.
                try:
                    self.settle_aborted(reservation, detail=(
                        "pre-launch creation failed before any spawn: "
                        f"{type(exc).__name__}: {exc}"))
                except BudgetError as abort_failure:
                    raise PrelaunchError(
                        f"the pre-launch creation step for {reservation_id} "
                        f"failed ({type(exc).__name__}: {exc}) and the zero-charge "
                        f"abort also failed ({abort_failure}); the reservation "
                        "stays pending until it is settled explicitly"
                    ) from exc
                raise PrelaunchError(
                    f"the pre-launch creation step for {reservation_id} failed "
                    f"({type(exc).__name__}: {exc}); nothing was started, so the "
                    "reservation was settled as aborted at a zero charge and "
                    "both rows remain in the chain"
                ) from exc
        return reservation

    def _settling(self, reservation):
        self._load()
        settled = {row["reservation_id"] for row in self._rows
                   if row["kind"] == SETTLE_KIND}
        if reservation.reservation_id in settled:
            raise DoubleSettlementError(
                f"reservation {reservation.reservation_id} is already settled; a "
                "second settlement is refused, never merged or averaged")
        open_reservation = self.pending
        if open_reservation is None:
            raise DoubleSettlementError(
                f"there is no open reservation to settle for "
                f"{reservation.reservation_id}")
        if open_reservation.reservation_id != reservation.reservation_id:
            raise SettlementError(
                f"{reservation.reservation_id} is not the pending reservation "
                f"{open_reservation.reservation_id}")
        return open_reservation

    def _returncode(self, returncode):
        if returncode is None:
            return None
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            raise SettlementError(
                f"returncode must be an int or None, got {returncode!r}")
        return returncode

    def mark_spawn(self, reservation, *, detail=None) -> dict:
        """Record that a child is about to be started. Call it immediately before.

        This is the second durable marker of a segment, and it is what lets the
        ledger distinguish **reserved but never spawned** from **spawned,
        outcome unknown**. Only the first case may ever be settled at zero.

        Written once per reservation; a second marker is refused.
        """
        held = self._settling(reservation)
        if self.spawned(held):
            raise SettlementError(
                f"reservation {held.reservation_id} already has a spawn marker")
        index, posix_time, previous, clamped = self._next()
        return self._append({
            "index": index, "kind": SPAWN_KIND, "prev_sha256": previous,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "clock_clamped": clamped,
            "reservation_id": held.reservation_id,
            "reservation_index": held.index,
            "stage": held.stage, "recipe": held.recipe, "seed": held.seed,
            "segment_index": held.segment_index,
            "detail": None if detail is None else str(detail),
        })

    def settle_aborted(self, reservation, *, detail) -> dict:
        """Settle at a **zero** charge, permitted only if no child ever started.

        This is the one settlement that charges nothing, and it is admissible
        **only** with positive evidence: the absence of a spawn marker for this
        reservation in the append-only chain. If a spawn marker is present the
        settlement is refused, and the caller must use :meth:`settle` with the
        measured wall or :meth:`settle_unknown` at the full reserved bound.

        It exists because the alternative wedges the campaign. A reservation is
        durable before the caller's ``create`` runs, so a bad path or a typo
        would otherwise leave a pending reservation blocking every later job with
        no auditable way forward. Fail-closed is right for a dangerous case, but
        a failure that provably preceded any child is not one -- and both rows
        stay in the chain forever, so nothing is erased.

        This is **not** a fresh-ledger bypass: no row is retracted, no ledger is
        replaced, and the zero charge is unavailable the moment a spawn marker
        exists.
        """
        held = self._settling(reservation)
        if self.spawned(held):
            raise SettlementError(
                f"reservation {held.reservation_id} has a spawn marker, so a "
                "child was started and a zero charge is inadmissible; settle the "
                "measured wall or settle_unknown at the full reserved bound")
        text = str(detail).strip()
        if not text:
            raise SettlementError(
                "settle_aborted needs an explicit detail recording why nothing "
                "was started")
        index, posix_time, previous, clamped = self._next()
        return self._append({
            "index": index, "kind": SETTLE_KIND, "prev_sha256": previous,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "clock_clamped": clamped,
            "reservation_id": held.reservation_id,
            "reservation_index": held.index,
            "stage": held.stage, "recipe": held.recipe, "seed": held.seed,
            "segment_index": held.segment_index,
            "start_epoch": held.start_epoch, "end_epoch": held.end_epoch,
            "reserved_bound_s": held.reserved_bound_s,
            "shutdown_allowance_s": held.shutdown_allowance_s,
            "actual_wall_s": None,
            "charged_s": 0.0,
            "charge_basis": CHARGE_ABORTED_PRESPAWN,
            "spawned": False,
            "overran_bound": False,
            "returncode": None,
            "detail": text,
            "contended": held.contention.get("contended"),
            "contention_at_launch": dict(held.contention),
            # No child existed, so this is a known zero, not an unknown.
            "attempted_control_transitions": 0,
            "completed_control_transitions": 0,
            "replayed_control_transitions": 0,
            "attempted_physics_steps": 0,
            "completed_physics_steps": 0,
            "replayed_physics_steps": 0,
            "partial_update_cost_s": 0.0,
        })

    def settle(self, reservation, *, actual_wall_s, returncode, counters) -> dict:
        """Charge the segment's **actual full process wall** and close the claim.

        Everything the process spent is charged: imports and setup, rollout,
        updates, evaluation, checkpoint IO, a failed attempt and replayed work.
        An actual wall beyond the reserved bound is charged in full; the bound
        was a forecast, and a forecast authorises nothing.
        """
        held = self._settling(reservation)
        spawned = self.spawned(held)
        try:
            actual = _finite(actual_wall_s, "actual_wall_s", minimum=0.0)
        except BudgetError as exc:
            raise SettlementError(str(exc)) from exc
        if not isinstance(counters, SegmentCounters):
            raise SettlementError(
                f"counters must be SegmentCounters, got {counters!r}")
        index, posix_time, previous, clamped = self._next()
        return self._append({
            "index": index, "kind": SETTLE_KIND, "prev_sha256": previous,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "clock_clamped": clamped,
            "reservation_id": held.reservation_id,
            "reservation_index": held.index,
            "stage": held.stage, "recipe": held.recipe, "seed": held.seed,
            "segment_index": held.segment_index,
            "start_epoch": held.start_epoch, "end_epoch": held.end_epoch,
            "reserved_bound_s": held.reserved_bound_s,
            "shutdown_allowance_s": held.shutdown_allowance_s,
            "actual_wall_s": actual,
            "charged_s": actual,
            "charge_basis": CHARGE_ACTUAL,
            "spawned": spawned,
            "overran_bound": bool(actual > held.reserved_bound_s),
            "returncode": self._returncode(returncode),
            "detail": None,
            "contended": held.contention.get("contended"),
            "contention_at_launch": dict(held.contention),
            **counters.as_dict(),
        })

    def settle_unknown(self, reservation, *, detail, returncode=None) -> dict:
        """Charge the **full reserved bound** for a crash or an unknown duration.

        Used when no trustworthy wall was measured. The counters are recorded as
        ``null``, meaning unknown: an unknown amount of completed work is never
        written down as zero, just as an unknown contention is never written
        down as absent.
        """
        held = self._settling(reservation)
        spawned = self.spawned(held)
        text = str(detail).strip()
        if not text:
            raise SettlementError(
                "settle_unknown needs an explicit detail describing what was "
                "not measured")
        index, posix_time, previous, clamped = self._next()
        return self._append({
            "index": index, "kind": SETTLE_KIND, "prev_sha256": previous,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "clock_clamped": clamped,
            "reservation_id": held.reservation_id,
            "reservation_index": held.index,
            "stage": held.stage, "recipe": held.recipe, "seed": held.seed,
            "segment_index": held.segment_index,
            "start_epoch": held.start_epoch, "end_epoch": held.end_epoch,
            "reserved_bound_s": held.reserved_bound_s,
            "shutdown_allowance_s": held.shutdown_allowance_s,
            "actual_wall_s": None,
            "charged_s": held.reserved_bound_s,
            "charge_basis": CHARGE_RESERVED_BOUND,
            "spawned": spawned,
            "overran_bound": False,
            "returncode": self._returncode(returncode),
            "detail": text,
            "contended": held.contention.get("contended"),
            "contention_at_launch": dict(held.contention),
            "attempted_control_transitions": None,
            "completed_control_transitions": None,
            "replayed_control_transitions": None,
            "attempted_physics_steps": None,
            "completed_physics_steps": None,
            "replayed_physics_steps": None,
            "partial_update_cost_s": None,
        })


# ---------------------------------------------------------------------------
# Lock files: exclusive, hash-bound, ignored evidence
# ---------------------------------------------------------------------------

def _flatten(inputs) -> dict:
    """Digest-bearing inputs, flattened to dotted names.

    A nested mapping is a per-key digest set (``selected_checkpoints.3001``), and
    every one of its leaves must be a sha256. A top-level scalar is a digest only
    when its name ends in ``_sha256``; any other scalar is free-form metadata,
    recorded in the lock but not treated as a hashed input.
    """
    if not isinstance(inputs, Mapping) or not inputs:
        raise LockError("lock inputs must be a non-empty mapping")
    flat = {}
    for name, value in inputs.items():
        if isinstance(value, Mapping):
            if not value:
                raise LockError(f"lock input {name!r} is an empty mapping")
            for key, digest in value.items():
                flat[f"{name}.{key}"] = digest
        elif str(name).endswith("_sha256"):
            flat[str(name)] = value
    if not flat:
        raise LockError("a lock must bind at least one sha256 input")
    for name, digest in flat.items():
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise LockError(
                f"lock input {name!r} is not a sha256 hex digest: {digest!r}")
    return flat


def lock_input_digests(record) -> dict:
    """The lock's hashed inputs, flattened to dotted names."""
    return _flatten(record["inputs"])


def write_lock(path, kind, inputs, *, clock=time.time) -> dict:
    """Write one exclusive, hash-bound lock file, or refuse.

    The **recipe lock** is written before stage 3 and binds the frozen protocol
    plus the stage-2 evidence; it must not demand a stage-3 checkpoint or
    evaluation hash, because none exists yet. The **checkpoint-selection lock**
    is written after stage-3 training and selection and before confirmation, and
    binds every selected checkpoint and its evaluation, so that selection can be
    recomputed from the lock's own hashed inputs (:func:`verify_lock`).

    **Postcondition:** no required name can be present in the lock while absent
    from its digest. A required input whose value cannot be bound as a digest is
    refused at write time rather than written unbound.

    Locks are evidence under an ignored path and are never committed.
    """
    if kind not in _LOCK_KINDS:
        raise LockError(f"unknown lock kind {kind!r}; expected one of {_LOCK_KINDS}")
    if not isinstance(inputs, Mapping):
        raise LockError("lock inputs must be a mapping")
    required = (RECIPE_LOCK_REQUIRED_INPUTS if kind == RECIPE_LOCK_KIND
                else SELECTION_LOCK_REQUIRED_INPUTS)
    for name in required:
        if name not in inputs:
            raise LockError(f"a {kind} lock must bind {name!r}")
    if kind == RECIPE_LOCK_KIND:
        present = [name for name in RECIPE_LOCK_FORBIDDEN_INPUTS if name in inputs]
        if present:
            raise LockError(
                f"a recipe lock must not require stage-3 artefacts {present}: it "
                "is written before stage 3 runs, so those hashes cannot exist yet")
    digests = _flatten(inputs)
    # F1: every declared required input must be provably BOUND, not merely
    # present. `_flatten` binds only a mapping of digests or a name ending
    # `_sha256`, so a required input supplied as any other scalar would
    # otherwise be written into the lock while escaping the verified digest set
    # -- and the lock would then verify clean without that input ever being
    # checked. That is a fail-open in the one mechanism whose whole job is
    # tamper evidence, so it is refused at write time instead.
    for name in required:
        if not (name in digests
                or any(key.startswith(f"{name}.") for key in digests)):
            raise LockError(
                f"a {kind} lock must bind {name!r} as a sha256 digest or as a "
                f"non-empty mapping of digests, got {inputs[name]!r}; a required "
                "input that is present but unbound would leave the lock "
                "verifying clean without it")
    posix_time = _finite(clock(), "clock")
    body = {"schema_version": LOCK_SCHEMA_VERSION, "kind": kind,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "inputs": json.loads(json.dumps(inputs, sort_keys=True,
                                            allow_nan=False)),
            "input_names": sorted(digests)}
    record = dict(body, lock_sha256=record_digest(body))
    try:
        with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise LockError(
            f"a lock already exists at {path}; a lock is written exactly once "
            "and never replaced") from exc
    return record


def read_lock(path) -> dict:
    """Read a lock and re-verify its own digest, or refuse."""
    target = Path(path)
    if not target.is_file():
        raise LockError(f"lock {target} does not exist")
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, TypeError) as exc:
        raise LockError(f"lock {target} is not valid JSON: {exc}") from exc
    if not isinstance(record, dict) or "lock_sha256" not in record:
        raise LockError(f"lock {target} has no lock_sha256")
    body = {key: value for key, value in record.items() if key != "lock_sha256"}
    try:
        recomputed = record_digest(body)
    except (TypeError, ValueError) as exc:
        raise LockError(f"lock {target} cannot be re-encoded: {exc}") from exc
    if record["lock_sha256"] != recomputed:
        raise LockError(
            f"lock {target} lock_sha256 {record['lock_sha256']!r} does not match "
            f"its content ({recomputed!r}): the lock was altered")
    if record.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise LockError(
            f"lock {target} schema_version {record.get('schema_version')!r} is "
            f"not {LOCK_SCHEMA_VERSION!r}")
    return record


def verify_lock(path, resolver) -> dict:
    """Recompute every hashed input from ``resolver`` and refuse any drift.

    ``resolver(dotted_name) -> sha256``. Call this before confirmation so that
    selection is recomputed from the lock's own hashed inputs rather than trusted
    from a summary.
    """
    record = read_lock(path)
    for name, digest in sorted(lock_input_digests(record).items()):
        try:
            actual = resolver(name)
        except Exception as exc:
            raise LockError(
                f"lock input {name!r} could not be recomputed "
                f"({type(exc).__name__}: {exc})") from exc
        if actual != digest:
            raise LockError(
                f"lock input {name!r} has drifted: locked {digest!r}, "
                f"recomputed {actual!r}")
    return record
