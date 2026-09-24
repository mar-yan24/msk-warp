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
import os
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

#: An explicit, recorded adoption of a moved parent ledger's balance. It carries
#: BOTH the superseded opening balance and the new one, so the audit trail shows
#: what changed and when. It is never written automatically: see
#: :data:`PARENT_RECHECK_NOTE`.
RESNAPSHOT_KIND = "resnapshot"
_KINDS = (HEADER_KIND, RESERVE_KIND, SPAWN_KIND, SETTLE_KIND, RESNAPSHOT_KIND)

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


class ParentLedgerMovedError(BudgetError):
    """The parent ledger moved after this ledger's opening balance was taken.

    Raised at **every** reserve, before anything durable or expensive, so a
    descendant can never commit wall time against a stale balance. It is not
    downgraded to a warning and there is no flag that suppresses it; the only
    way forward is an explicit :meth:`BudgetLedger.resnapshot_parent`. See
    :data:`PARENT_RECHECK_NOTE`.

    The re-check is transitive, so the ledger that moved need not be the direct
    parent. ``moved_ledger`` names the ledger whose charge actually moved, and
    ``stale_chain`` names every ledger whose carried balance is stale because of
    it, nearest the moved ledger first -- which is the order in which they must
    be re-snapshotted.
    """

    def __init__(self, message, *, stale_chain=(), moved_ledger=None):
        super().__init__(message)
        self.stale_chain = tuple(str(item) for item in stale_chain)
        self.moved_ledger = None if moved_ledger is None else str(moved_ledger)


class UnauthorisedLaunchError(BudgetError):
    """The ledger's protocol declares a launch lock, and the lock is closed.

    Raised by :meth:`BudgetLedger.create` before the file is written, and by
    :meth:`BudgetLedger.reserve` after the parent re-check but before any cap
    check, contention observation or durable row -- so a refusal commits
    nothing. The protocol's own refusal is chained as ``__cause__``. A protocol
    that declares no ``assert_authorised`` (the sealed campaign protocol and
    probe E) is unaffected. Settlement is never gated: revoking an
    authorisation stops the next launch, it never strands a charge already
    committed.
    """


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


def sha256_bytes(raw) -> str:
    """SHA256 of exact bytes. Used to bind a carried-forward source ledger."""
    return hashlib.sha256(bytes(raw)).hexdigest()


#: The keys a header's ``carried_forward`` block must carry, exactly.
CARRIED_FORWARD_KEYS = ("source_path", "source_sha256", "source_rows",
                        "source_protocol_digest", "global_s", "stage_s",
                        "run_s", "settled_rows")

CARRY_FORWARD_NOTE = (
    "A protocol amendment gets its OWN ledger so the ledger it descends from "
    "stays byte-intact, and that new ledger opens at the settled spend of the "
    "one it descends from. The opening balance is not a convenience: without it "
    "a new ledger would reset every cap, which would make a new protocol version "
    "a way of escaping a charge already made. The carried block lives inside the "
    "hash-chained header row, so it cannot be edited after the fact, and it "
    "records the source ledger's path, byte sha256, row count and protocol "
    "digest beside its totals. The source is READ ONLY: it is never extended, "
    "migrated, rewritten or re-hashed."
)


PARENT_RECHECK_NOTE = (
    "A carried opening balance is a snapshot of the parent ledger at ONE "
    "instant. If the parent settles anything afterwards, the descendant's "
    "balance is stale and it can over-grant the global cap -- it would be "
    "drawing against wall time already spent. Relying on the parent being "
    "quiescent is a convention, not a guarantee, so the check is structural "
    "instead: EVERY reserve re-reads the parent, re-verifies its whole hash "
    "chain, recomputes its settled charge from its own rows, and refuses on any "
    "divergence of that total, of the parent's byte sha256 or of its row count. "
    "The refusal names both totals and both sha256s. The descendant NEVER "
    "reconciles a divergence on its own and never falls back to the snapshot: "
    "an unreadable, shrunken, rewritten or chain-broken parent refuses too, and "
    "there is no override flag. Adopting a moved parent's balance takes an "
    "explicit ``resnapshot_parent`` call carrying a human-supplied reason, which "
    "appends a row holding the superseded balance beside the new one. The parent "
    "is READ ONLY throughout: it is never written, extended, migrated or "
    "re-hashed by the descendant. "
    "The re-check is TRANSITIVE. Once the direct parent matches its snapshot, "
    "the parent's own carried balance is re-checked against ITS parent, and so "
    "on up the chain: a parent whose ancestor moved is itself stale even though "
    "its own bytes never changed, so a descendant three levels down refuses "
    "when the sealed grandparent settles. A moved ancestor raises "
    "ParentLedgerMovedError naming the chain from the moved ledger down and "
    "the remedy, which is to re-snapshot every stale ledger in order, nearest "
    "the moved ledger first (for Direction F: re-snapshot probe E, then F). "
    "``create`` and ``resnapshot_parent`` refuse, in the same way, to adopt a "
    "balance from a source ledger that is itself stale. A chain that loops "
    "back on itself, or runs deeper than MAX_CARRY_CHAIN_DEPTH ledgers, is a "
    "LedgerIntegrityError and never a RecursionError: a genuine lineage is "
    "created parent-first, so it cannot loop. "
    "RECORDED DESIGN INTENT for a successor, so nobody reaches for the wrong "
    "tool: the check fires at the reserve, which is sufficient while a "
    "descendant authorises ONE cell at segment 0, because there is then no next "
    "reserve to protect. If a later descendant ever authorises more than one "
    "cell, the right addition is NOT a refusal at settle -- that would wedge the "
    "ledger with a pending reservation and lose a real charge -- but a recorded "
    "``parent_moved_during_segment`` field on the settle row, so the charge "
    "still lands and the divergence stays visible in the chain."
)

#: The deepest carried-forward chain the transitive re-check will walk. The
#: real chain is three ledgers deep (sealed, probe E, Direction F); the bound
#: exists so that any chain, looping or merely long, ends in a
#: LedgerIntegrityError rather than a RecursionError.
MAX_CARRY_CHAIN_DEPTH = 32

#: The runner's per-segment launch record and segment directory prefix. Their
#: presence under a bound protocol's RUN_ROOT is evidence that a launch already
#: happened there (see :func:`_assert_run_root_unlaunched`).
LAUNCH_RECORD_FILENAME = "launch.json"
SEGMENT_DIR_PREFIX = "segment_"


@dataclass(frozen=True)
class CarriedForward:
    """One ledger's settled spend, carried into a descendant ledger's header.

    ``settled_rows`` counts the settlements the source *represents*, including
    any it had itself carried, so a chain of amendments stays honest about how
    many settlements stand behind the opening balance.
    """

    source_path: str
    source_sha256: str
    source_rows: int
    source_protocol_digest: str
    global_s: float
    stage_s: dict
    run_s: dict
    settled_rows: int

    def as_dict(self) -> dict:
        if not _SHA256_RE.match(str(self.source_sha256)):
            raise LedgerIntegrityError(
                f"carried source_sha256 {self.source_sha256!r} is not a sha256")
        if not _SHA256_RE.match(str(self.source_protocol_digest)):
            raise LedgerIntegrityError(
                "carried source_protocol_digest "
                f"{self.source_protocol_digest!r} is not a sha256")
        return {
            "source_path": str(self.source_path),
            "source_sha256": str(self.source_sha256),
            "source_rows": _count(self.source_rows, "source_rows"),
            "source_protocol_digest": str(self.source_protocol_digest),
            "global_s": _finite(self.global_s, "carried global_s", minimum=0.0),
            "stage_s": {str(key): _finite(value, f"carried stage_s[{key}]",
                                          minimum=0.0)
                        for key, value in dict(self.stage_s).items()},
            "run_s": {str(key): _finite(value, f"carried run_s[{key}]",
                                        minimum=0.0)
                      for key, value in dict(self.run_s).items()},
            "settled_rows": _count(self.settled_rows, "carried settled_rows"),
            "note": CARRY_FORWARD_NOTE,
        }


def validate_carried_forward(block) -> dict:
    """Refuse a header's carried block unless it is complete and well formed."""
    if not isinstance(block, Mapping):
        raise LedgerIntegrityError(
            "the ledger header's carried_forward block is not an object: "
            f"{block!r}")
    missing = [key for key in CARRIED_FORWARD_KEYS if key not in block]
    if missing:
        raise LedgerIntegrityError(
            f"the ledger header's carried_forward block is missing {missing}")
    for key in ("source_sha256", "source_protocol_digest"):
        if not _SHA256_RE.match(str(block[key])):
            raise LedgerIntegrityError(
                f"the carried_forward {key} {block[key]!r} is not a sha256")
    _count(block["source_rows"], "carried source_rows")
    _count(block["settled_rows"], "carried settled_rows")
    _finite(block["global_s"], "carried global_s", minimum=0.0)
    for field_name in ("stage_s", "run_s"):
        mapping = block[field_name]
        if not isinstance(mapping, Mapping):
            raise LedgerIntegrityError(
                f"the carried_forward {field_name} is not an object")
        for key, value in mapping.items():
            _finite(value, f"carried {field_name}[{key}]", minimum=0.0)
    return dict(block)


def _assert_declared_parent(spec, carried) -> None:
    """Refuse unless ``carried`` really came from the protocol's declared parent.

    Both clauses are checked separately and each names the expected value beside
    the supplied one. A protocol that declares neither literal -- the sealed
    campaign protocol declares no parent at all -- is unconstrained by this.
    """
    declared_path = getattr(spec, "PARENT_LEDGER", None)
    if declared_path is not None:
        root = Path(getattr(spec, "ROOT", P.ROOT))
        expected = (root / str(declared_path)).resolve()
        supplied = Path(carried.source_path).resolve()
        if supplied != expected:
            raise LedgerIntegrityError(
                f"this protocol declares its parent ledger as {declared_path!r} "
                f"(resolved {expected}), but the carried opening balance came "
                f"from {supplied}. A descendant may open only at the settled "
                "spend of the declared parent: carrying from any other ledger "
                "would re-grant wall time that has already been spent, and the "
                "mistake would then be guarded faithfully for the rest of this "
                "ledger's life")
    declared_digest = getattr(spec, "PARENT_PROTOCOL_DIGEST", None)
    if declared_digest is not None:
        supplied_digest = str(carried.source_protocol_digest)
        if supplied_digest != str(declared_digest):
            raise LedgerIntegrityError(
                f"this protocol descends from parent protocol digest "
                f"{declared_digest}, but the carried opening balance came from a "
                f"ledger whose header digest is {supplied_digest}. A descendant "
                "may open only at the settled spend of the protocol it declares "
                "as its parent")


def _assert_source_chain_current(source_path, *, descendant, action) -> None:
    """Refuse to adopt a carried balance from a source ledger that is stale.

    A source whose OWN carried balance no longer matches its parent has a
    settled total that omits spend beneath it, even though its bytes never
    changed. Carrying that total into a descendant would re-grant wall time
    already spent, so :meth:`BudgetLedger.create` and
    :meth:`BudgetLedger.resnapshot_parent` refuse it and name the order that
    fixes it: re-snapshot the stale source first, then the descendant.
    """
    source = Path(source_path)
    here = str(Path(descendant).resolve())
    try:
        ledger = BudgetLedger.open(source, inspect=True)
    except BudgetError as exc:
        raise LedgerIntegrityError(
            f"the carried source ledger {source} could not be read and "
            f"re-verified, so the {action} for {here} is refused: {exc}"
        ) from exc
    try:
        ledger.assert_parent_unmoved(_seen=(here,))
    except ParentLedgerMovedError as exc:
        stale = exc.stale_chain + (here,)
        remedy = "re-snapshot " + ", then ".join(stale)
        raise ParentLedgerMovedError(
            f"the carried opening balance for {here} would come from {source}, "
            "which is itself STALE: the ledger "
            f"{exc.moved_ledger} has moved beneath it. Adopting that balance "
            "would carry a total that omits wall time already spent, so this "
            f"{action} is refused and nothing is written. Remedy, in order: "
            f"{remedy} -- the last step being this {action}. The source's own "
            f"refusal: {exc}",
            stale_chain=stale, moved_ledger=exc.moved_ledger) from exc
    except LedgerIntegrityError as exc:
        raise LedgerIntegrityError(
            f"the carried-forward chain above {source} could not be confirmed, "
            f"so the {action} for {here} is refused: {exc}"
        ) from exc


def _assert_bound_ledger_path(spec, path) -> None:
    """With ``BIND_LEDGER_PATH``, the one ledger may be created only at LEDGER.

    Opt-in, so the sealed protocol and probe E -- which declare no such flag --
    are unaffected. Without it, a second ledger for the same protocol written
    anywhere else would open every cap afresh beside the first.
    """
    if not getattr(spec, "BIND_LEDGER_PATH", False):
        return
    declared = getattr(spec, "LEDGER", None)
    if declared is None:
        raise LedgerIntegrityError(
            "this protocol binds its ledger path (BIND_LEDGER_PATH) but declares "
            "no LEDGER, so the one place its ledger may live is unknown; refused")
    expected = (Path(getattr(spec, "ROOT", P.ROOT)) / str(declared)).resolve()
    supplied = Path(path).resolve()
    if supplied != expected:
        raise LedgerIntegrityError(
            f"this protocol binds its one ledger to {expected} (LEDGER "
            f"{str(declared)!r}), but create was asked to write {supplied}. A "
            "second ledger for one protocol, anywhere else, would open every cap "
            "afresh beside the first, so it is refused and nothing is written")


def _assert_run_root_unlaunched(spec) -> None:
    """With ``BIND_LEDGER_PATH``, refuse a ledger for a run root already used.

    The deleted-ledger re-grant guard. The path binding pins WHERE the one
    ledger lives, but a ledger that was deleted could be created again at that
    same path at zero spend. Launches leave durable traces the ledger does not
    own -- the runner writes ``launch.json`` inside a ``segment_*`` directory
    under the run root -- so any such entry, at any depth, means launches
    already happened and a fresh ledger would re-grant their spend. Other files
    (the authorisation record, notes) are allowed. The walk never follows a
    symlink, creates nothing and changes nothing. Opt-in, so probe E, whose run
    root legitimately holds its launches, is unaffected.
    """
    if not getattr(spec, "BIND_LEDGER_PATH", False):
        return
    declared = getattr(spec, "RUN_ROOT", None)
    if declared is None:
        raise LedgerIntegrityError(
            "this protocol binds its ledger path (BIND_LEDGER_PATH) but declares "
            "no RUN_ROOT, so its run root cannot be checked for earlier launches; "
            "a ledger is never created without that check")
    root = Path(getattr(spec, "ROOT", P.ROOT)) / str(declared)
    if not os.path.lexists(root):
        return
    if not os.path.isdir(root):
        raise LedgerIntegrityError(
            f"the run root {root} exists but is not a directory, so it cannot be "
            "checked for earlier launches; refused")
    launch_name = os.path.normcase(LAUNCH_RECORD_FILENAME)
    segment_prefix = os.path.normcase(SEGMENT_DIR_PREFIX)
    found = []

    def _refuse_walk_error(error):
        raise error

    try:
        for current, dirs, files in os.walk(root, onerror=_refuse_walk_error,
                                            followlinks=False):
            for name in (*dirs, *files):
                key = os.path.normcase(name)
                if key == launch_name or key.startswith(segment_prefix):
                    found.append(os.path.join(current, name))
    except OSError as exc:
        raise LedgerIntegrityError(
            f"the run root {root} could not be walked to check for earlier "
            f"launches ({exc}); refused"
        ) from exc
    if found:
        found.sort()
        shown = ", ".join(found[:6])
        more = "" if len(found) <= 6 else f", and {len(found) - 6} more"
        raise LedgerIntegrityError(
            f"the run root {root} already holds {len(found)} launch record(s) "
            f"or segment directories ({shown}{more}). A bound ledger is created "
            "once, before the first launch, so these mean launches already "
            "happened; creating a ledger now -- for example after the first one "
            "was deleted -- would re-grant their spend at a zero opening "
            "balance. That deleted-ledger re-grant is refused and nothing is "
            "written. The launch history is never cleared to make room")


def carry_forward(path) -> CarriedForward:
    """Read a ledger and total its settlements, for a descendant's header.

    The source is opened through :meth:`BudgetLedger.open`, so its whole hash
    chain is re-verified before a single number is carried. It is **never**
    written to. A source holding an unsettled reservation is refused rather than
    carried: its charge is not yet known, so carrying it would understate the
    spend and hand the descendant headroom that may not exist.

    This is a snapshot at one instant, and that is **not** relied upon:
    :meth:`BudgetLedger.assert_parent_unmoved` re-checks it at every reserve.
    See :data:`PARENT_RECHECK_NOTE`.
    """
    # Read across the protocol boundary on purpose: a parent's header digest is
    # by definition not the descendant's. Inspection grants nothing and cannot
    # write, and the whole chain is still re-verified.
    source = BudgetLedger.open(path, inspect=True)
    open_reservation = source.pending
    if open_reservation is not None:
        raise LedgerIntegrityError(
            f"the source ledger {source.path} holds the unsettled reservation "
            f"{open_reservation.reservation_id!r}; its charge is not yet known, "
            "so carrying it forward would understate the spend")
    totals = source.charged()
    already = source.carried_forward or {}
    return CarriedForward(
        source_path=str(source.path.resolve()),
        source_sha256=sha256_bytes(source.path.read_bytes()),
        source_rows=len(source.rows),
        source_protocol_digest=source.protocol_digest,
        global_s=float(totals["global_s"]),
        stage_s={key: float(value) for key, value in totals["stage_s"].items()},
        run_s={key: float(value) for key, value in totals["run_s"].items()},
        settled_rows=int(totals["settled_rows"])
        + int(already.get("settled_rows", 0)))


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

    def __init__(self, path, *, clock=time.time, contention_probe=None,
                 protocol=None):
        #: The protocol this ledger is keyed to. The sealed campaign protocol by
        #: default; an amendment module passes itself, which is what lets its own
        #: arms be reserved while every sealed arm stays refused.
        self.protocol = P if protocol is None else protocol
        #: True only for a ledger opened to be READ. An inspecting ledger asserts
        #: no protocol binding, because reading a foreign ledger to total it is
        #: exactly what carry-forward is -- and in exchange it refuses every
        #: mutating call, so it is a read mode and not an escape hatch.
        self._inspect_only = False
        self._carried = None
        self._carried_chain: tuple = ()
        self.path = Path(path)
        self._clock = clock
        self._probe = contention_probe
        self._rows: tuple = ()

    # -- construction -----------------------------------------------------

    @classmethod
    def create(cls, path, *, protocol_digest, clock=time.time,
               contention_probe=None, protocol=None, carried=None,
               provenance=None) -> "BudgetLedger":
        """Create the one ledger, refusing to touch anything already present."""
        ledger = cls(path, clock=clock, contention_probe=contention_probe,
                     protocol=protocol)
        spec = ledger.protocol
        digest = str(protocol_digest)
        if not _SHA256_RE.match(digest):
            raise LedgerIntegrityError(
                f"protocol_digest must be a sha256 hex digest, got {digest!r}")
        # A protocol that descends from another one must open at that one's
        # settled spend. Refusing here is what stops a new protocol version from
        # being used as a way of escaping a charge already made.
        if getattr(spec, "REQUIRES_CARRY_FORWARD", False) and carried is None:
            raise LedgerIntegrityError(
                f"protocol {digest} requires a carried forward opening balance "
                f"from {getattr(spec, 'PARENT_LEDGER', 'its parent ledger')}: a "
                "fresh ledger would reset every cap, and a new ledger is never a "
                "way of escaping a charge already made")
        carried_block = None
        if carried is not None:
            if not isinstance(carried, CarriedForward):
                raise LedgerIntegrityError(
                    f"carried must be a CarriedForward, got {carried!r}")
            if Path(carried.source_path) == ledger.path.resolve():
                raise LedgerIntegrityError(
                    f"a ledger never carries itself forward: {ledger.path}")
            if carried.source_protocol_digest == digest:
                raise LedgerIntegrityError(
                    "the carried source ledger has the same protocol digest "
                    f"{digest}: a second ledger for one protocol would be a "
                    "migration or a rewrite of the first, which is refused. A "
                    "carried opening balance is only for a DESCENDANT protocol")
            # The parent a descendant opens against must be the ONE its protocol
            # declares. Without this, carrying from any other sealed-protocol
            # ledger -- a fresh, zero-spend one, say -- re-grants the whole cap
            # and is then guarded faithfully for the rest of this ledger's life,
            # which is the worst shape of the bug because everything downstream
            # looks correct.
            _assert_declared_parent(spec, carried)
            # Transitive: the declared parent must itself be current against
            # ITS parent, or its settled total omits spend beneath it.
            _assert_source_chain_current(carried.source_path,
                                         descendant=ledger.path, action="create")
            carried_block = carried.as_dict()
        # Opt-in bindings for a protocol that declares BIND_LEDGER_PATH: one
        # ledger, at one path, and never again once launches have happened.
        _assert_bound_ledger_path(spec, ledger.path)
        _assert_run_root_unlaunched(spec)
        # The protocol's launch lock, if it declares one, before the file exists.
        ledger._assert_protocol_authorised("create")
        posix_time = _finite(ledger._clock(), "clock")
        body = {
            "index": 0, "kind": HEADER_KIND, "prev_sha256": GENESIS_PREV_SHA256,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "schema_version": SCHEMA_VERSION, "protocol_digest": digest,
            "total_wall_cap_s": spec.TOTAL_WALL_CAP_S,
            "reserve_wall_cap_s": spec.RESERVE_WALL_CAP_S,
            "segment_max_epochs": spec.SEGMENT_MAX_EPOCHS,
            "segment_work_deadline_s": spec.SEGMENT_WORK_DEADLINE_S,
            "segment_call_bound_s": spec.SEGMENT_CALL_BOUND_S,
        }
        # Both optional blocks are written INSIDE the hash-chained header row, so
        # neither can be edited after the ledger exists. A ledger with neither is
        # byte-identical to one written before this facility existed.
        if carried_block is not None:
            body["carried_forward"] = carried_block
        if provenance is not None:
            if not isinstance(provenance, Mapping):
                raise LedgerIntegrityError(
                    f"provenance must be a mapping, got {provenance!r}")
            body["protocol_provenance"] = dict(provenance)
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
        ledger._carried = carried_block
        ledger._carried_chain = () if carried_block is None else (dict(carried_block),)
        return ledger

    @classmethod
    def open(cls, path, *, clock=time.time, contention_probe=None,
             protocol=None, inspect=False) -> "BudgetLedger":
        """Open and fully verify an existing ledger, or refuse.

        The ledger's header digest is bound to ``protocol`` **here**, in the
        durable authority, rather than by a caller remembering to ask: opening a
        ledger under a protocol that is not its own is refused outright, so the
        two protocols are not interchangeable at any level and in any statement
        order. ``protocol=None`` means the sealed campaign protocol, so a
        descendant's ledger is equally refused under the default.

        ``inspect=True`` opens the ledger to be **read** across that boundary --
        totalling a foreign parent's charge, summarising a ledger a report did
        not create -- and asserts no binding. In exchange an inspecting ledger
        refuses every mutating call (:meth:`_assert_writable`), so it grants
        nothing and is not an override.
        """
        ledger = cls(path, clock=clock, contention_probe=contention_probe,
                     protocol=protocol)
        ledger._inspect_only = bool(inspect)
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
        # The opening balance the file declares, and every superseded one before
        # it. The header's block is the first; each resnapshot row supersedes it.
        effective_carried = None
        carried_chain = []
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
                # The binding, enforced by the loader rather than by whoever
                # happens to call assert_protocol_unchanged first. Without it a
                # sealed ledger opened under an amendment protocol would accept
                # an amendment arm and append it to the sealed chain, which is
                # append-only and therefore permanent.
                if not self._inspect_only:
                    declared = getattr(self.protocol, "protocol_digest", None)
                    expected = None if declared is None else declared()
                    if expected is not None and row["protocol_digest"] != expected:
                        raise LedgerIntegrityError(
                            f"ledger {self.path} has header protocol digest "
                            f"{row['protocol_digest']} but was opened under the "
                            f"protocol whose digest is {expected}; the two are "
                            "never interchangeable, so this open is refused. To "
                            "READ a ledger of another protocol -- to total a "
                            "parent's charge, for instance -- open it with "
                            "inspect=True, which grants nothing and refuses "
                            "every write")
                if "carried_forward" in row:
                    effective_carried = validate_carried_forward(
                        row["carried_forward"])
                    carried_chain.append(dict(effective_carried))
            elif kind == HEADER_KIND:
                raise LedgerIntegrityError(
                    f"ledger line {number + 1} is a second header")
            elif kind == RESNAPSHOT_KIND:
                # An explicit adoption of a moved parent's balance. Validated
                # here rather than trusted: it is inside the hash chain, so a
                # later edit of it is caught, and the balances must form a
                # contiguous, non-decreasing sequence.
                if effective_carried is None:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} re-snapshots a parent, but "
                        "this ledger declares no carried opening balance")
                if pending_id is not None:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} re-snapshots the parent while "
                        f"reservation {pending_id!r} is still open; that would "
                        "move the balance the open hold was checked against")
                for field_name in ("previous_carried", "carried_forward"):
                    if field_name not in row:
                        raise LedgerIntegrityError(
                            f"ledger line {number + 1} is a re-snapshot with no "
                            f"{field_name}")
                superseded = validate_carried_forward(row["previous_carried"])
                adopted = validate_carried_forward(row["carried_forward"])
                if not str(row.get("detail") or "").strip():
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} is a re-snapshot with no "
                        "detail; the reason is part of the record")
                if superseded != effective_carried:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} re-snapshots from a balance "
                        "this ledger was not holding; the carried balances must "
                        "form one contiguous sequence")
                if adopted["global_s"] < effective_carried["global_s"]:
                    raise LedgerIntegrityError(
                        f"ledger line {number + 1} lowers the carried opening "
                        f"balance from {effective_carried['global_s']!r} to "
                        f"{adopted['global_s']!r}; a settled charge never "
                        "decreases")
                effective_carried = adopted
                carried_chain.append(dict(adopted))
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
        self._carried = effective_carried
        self._carried_chain = tuple(carried_chain)

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

    @property
    def carried_forward(self):
        """The header's carried-forward opening balance, or ``None``.

        ``None`` means this ledger opens every cap at its full value, which is
        the sealed campaign ledger's case. See :data:`CARRY_FORWARD_NOTE`.
        """
        return None if self._carried is None else dict(self._carried)

    def carried_history(self) -> list:
        """Every carried opening balance this ledger has held, oldest first.

        One entry for a ledger that has never re-snapshotted, and empty for one
        that carries nothing at all.
        """
        return [dict(block) for block in self._carried_chain]

    def assert_parent_unmoved(self, *, _seen=()) -> None:
        """Refuse unless the parent ledger is exactly what was carried forward.

        Called by :meth:`reserve` on **every** reservation, immediately after the
        pending-reservation trap and before any cap check, contention
        observation or appended row -- so a refusal commits nothing. Fail closed
        in every direction: a parent that cannot be read and re-verified refuses
        just as loudly as one whose charge advanced, and the snapshot is never
        accepted in its place. See :data:`PARENT_RECHECK_NOTE`.

        The check is **transitive**: once the direct parent matches its
        snapshot, the parent's own carried balance is re-checked the same way,
        up to a ledger that carries nothing. A moved ancestor is a
        :class:`ParentLedgerMovedError` naming the chain and the re-snapshot
        order; a loop, or a chain deeper than :data:`MAX_CARRY_CHAIN_DEPTH`, is
        a :class:`LedgerIntegrityError`.

        ``_seen`` is internal: the resolved paths already visited below this
        ledger, oldest descendant first. It can only add refusals -- the loop
        and depth guards -- and never skips a check.

        A ledger with no carried balance has no parent, and returns.
        """
        carried = self._carried
        if carried is None:
            return
        here = str(self.path.resolve())
        visited = tuple(str(item) for item in _seen)
        path = Path(carried["source_path"])
        if len(visited) >= MAX_CARRY_CHAIN_DEPTH:
            raise LedgerIntegrityError(
                f"the carried-forward chain below {here} is more than "
                f"{MAX_CARRY_CHAIN_DEPTH} ledgers deep, which no genuine lineage "
                "is; the chain cannot be confirmed, so this is refused")
        parent_key = os.path.normcase(str(path.resolve()))
        seen_keys = {os.path.normcase(item) for item in visited}
        if (os.path.normcase(here) in seen_keys
                or parent_key in seen_keys | {os.path.normcase(here)}):
            loop = " -> ".join(visited + (here, str(path.resolve())))
            raise LedgerIntegrityError(
                f"the carried-forward chain loops back on itself: {loop} (each "
                "ledger carried from the next). A genuine lineage never loops, "
                "because every descendant is created after its parent; a loop "
                "means a ledger was replaced after a descendant carried from it. "
                "Nothing is reconciled, so this is refused")
        try:
            parent = BudgetLedger.open(path, inspect=True)
        except BudgetError as exc:
            raise LedgerIntegrityError(
                f"the parent ledger {path} could not be read and re-verified, so "
                "this ledger's carried opening balance cannot be confirmed: "
                f"{exc}. A carried balance is never trusted in place of its "
                "source, so this reserve is refused"
            ) from exc
        raw = path.read_bytes()
        actual_sha = sha256_bytes(raw)
        totals = parent.charged()
        actual_global = float(totals["global_s"])
        carried_global = float(carried["global_s"])
        open_reservation = parent.pending
        if (actual_sha == carried["source_sha256"]
                and actual_global == carried_global
                and len(parent.rows) == int(carried["source_rows"])
                and open_reservation is None):
            self._assert_ancestors_unmoved(parent, visited + (here,))
            return
        held = ("" if open_reservation is None else
                f" The parent also holds the unsettled reservation "
                f"{open_reservation.reservation_id!r}, so its charge is not even "
                "final yet.")
        raise ParentLedgerMovedError(
            f"the parent ledger {path} has MOVED since this ledger's carried "
            f"opening balance was taken. Carried global charge {carried_global!r} "
            f"against the parent's recomputed {actual_global!r}; carried sha256 "
            f"{carried['source_sha256']} against the parent's {actual_sha}; "
            f"carried {int(carried['source_rows'])} rows against "
            f"{len(parent.rows)}.{held} Reserving against a stale balance would "
            "draw against wall time already spent, so this reserve is refused. "
            "Nothing is reconciled automatically and the stale snapshot is not "
            "accepted: adopting the new balance takes an explicit re-snapshot "
            "(resnapshot_parent) with a stated reason, which records the "
            "superseded balance beside the new one",
            stale_chain=(here,), moved_ledger=str(path))

    def _assert_ancestors_unmoved(self, parent, visited) -> None:
        """The transitive step: re-check the direct parent's own carried balance.

        ``parent`` has just matched this ledger's snapshot. Its own parent may
        still have moved since the parent's balance was taken, and then this
        ledger's balance -- which includes the parent's -- is stale too. The
        refusal is re-raised here naming the whole chain and the remedy order.
        """
        here = visited[-1]
        try:
            parent.assert_parent_unmoved(_seen=visited)
        except ParentLedgerMovedError as exc:
            stale = exc.stale_chain + (here,)
            chain = " -> ".join((f"{exc.moved_ledger} (moved)",)
                                + tuple(f"{item} (stale)" for item in stale))
            remedy = "re-snapshot " + ", then ".join(stale)
            raise ParentLedgerMovedError(
                f"an ANCESTOR ledger has MOVED beneath this ledger's carried "
                f"opening balance, although the direct parent {parent.path} still "
                f"matches its snapshot. The chain, from the moved ledger down: "
                f"{chain}. Every stale ledger must adopt the new balance in "
                f"order, nearest the moved ledger first: {remedy} "
                "(resnapshot_parent on each, with a stated reason). Reserving "
                "against a stale balance would draw against wall time already "
                "spent, so this is refused and nothing is reconciled "
                f"automatically. The ancestor's own refusal: {exc}",
                stale_chain=stale, moved_ledger=exc.moved_ledger) from exc
        except LedgerIntegrityError as exc:
            raise LedgerIntegrityError(
                f"the carried-forward chain above {here} could not be "
                f"confirmed: its direct parent {parent.path} matches its "
                "snapshot, but the parent's own carried balance does not "
                f"verify: {exc}. A carried balance is never trusted in place of "
                "its source, so this is refused"
            ) from exc

    def _assert_protocol_authorised(self, action) -> None:
        """Call the protocol's launch lock, if it declares one, or refuse.

        A protocol without ``assert_authorised`` (the sealed campaign protocol,
        probe E) is unaffected. A protocol refusal becomes an
        :class:`UnauthorisedLaunchError` chained to it.
        """
        hook = getattr(self.protocol, "assert_authorised", None)
        if hook is None:
            return
        if not callable(hook):
            raise UnauthorisedLaunchError(
                f"the protocol of ledger {self.path} declares a launch lock that "
                f"is not callable ({hook!r}), so this {action} is refused")
        try:
            hook()
        except P.ProtocolError as exc:
            raise UnauthorisedLaunchError(
                f"the protocol of ledger {self.path} declares a launch lock and "
                f"it is closed, so this {action} is refused before anything is "
                f"written: {exc}"
            ) from exc

    def resnapshot_parent(self, *, detail) -> dict:
        """Adopt a moved parent's balance, explicitly and on the record.

        A deliberate human act, not a repair the ledger performs for itself: it
        requires a stated reason, refuses while a reservation is open, refuses a
        parent that has not actually moved, and refuses any balance lower than
        the one it supersedes. The appended row carries **both** balances, so the
        audit trail shows what changed, when, and why.

        There is deliberately **no** way to re-point it at a different parent:
        it re-reads the ledger the header names and re-checks the same declared
        parent binding, so it is not a second route around it.
        """
        self._assert_writable()
        self._load()
        open_reservation = self.pending
        if open_reservation is not None:
            raise PendingReservationError(
                f"reservation {open_reservation.reservation_id} is still "
                "unsettled; re-snapshotting the parent now would move the "
                "balance that open hold was checked against")
        carried = self._carried
        if carried is None:
            raise LedgerIntegrityError(
                f"ledger {self.path} carries no opening balance, so it has no "
                "parent to re-snapshot")
        reason = "" if detail is None else str(detail).strip()
        if not reason:
            raise LedgerIntegrityError(
                "a re-snapshot needs an explicit detail recording why the parent "
                "moved and on whose authority; it is a human decision, and the "
                "reason is part of the record")
        source = Path(carried["source_path"])
        fresh = carry_forward(source)
        _assert_declared_parent(self.protocol, fresh)
        # Transitive: never adopt a balance from a parent that is itself stale.
        # The order is fixed -- the stale parent re-snapshots first, then this.
        _assert_source_chain_current(fresh.source_path, descendant=self.path,
                                     action="re-snapshot")
        if fresh.source_protocol_digest == self.protocol_digest:
            raise LedgerIntegrityError(
                "the named parent has the same protocol digest "
                f"{self.protocol_digest}: a carried balance is only for a "
                "DESCENDANT protocol")
        if Path(fresh.source_path) == self.path.resolve():
            raise LedgerIntegrityError(
                f"a ledger never carries itself forward: {self.path}")
        block = fresh.as_dict()
        if block["global_s"] < float(carried["global_s"]):
            raise LedgerIntegrityError(
                "a re-snapshot may never decrease the carried opening balance: "
                f"{carried['global_s']!r} -> {block['global_s']!r}. An "
                "append-only parent's settled charge only grows, so a lower "
                "total means the parent was rewritten or a different ledger was "
                "named, and adopting it would hand back spend")
        if (block["source_sha256"] == carried["source_sha256"]
                and block["global_s"] == float(carried["global_s"])
                and block["source_rows"] == int(carried["source_rows"])):
            raise LedgerIntegrityError(
                f"the parent ledger {source} has not moved: sha256 "
                f"{block['source_sha256']} and global charge "
                f"{block['global_s']!r} are what this ledger already carries. A "
                "re-snapshot that changes nothing would be audit noise")
        index, posix_time, previous, clamped = self._next()
        row = self._append({
            "index": index, "kind": RESNAPSHOT_KIND, "prev_sha256": previous,
            "posix_time": posix_time, "utc": _utc(posix_time),
            "clock_clamped": clamped,
            "previous_carried": dict(carried),
            "carried_forward": block,
            "detail": reason,
            "note": PARENT_RECHECK_NOTE,
        })
        self._carried = block
        self._carried_chain = self._carried_chain + (dict(block),)
        return row

    def carried_totals(self) -> dict:
        """The opening balance alone, separated from this file's own rows."""
        carried = self._carried or {}
        return {
            "global_s": float(carried.get("global_s", 0.0)),
            "stage_s": {key: float(value)
                        for key, value in (carried.get("stage_s") or {}).items()},
            "run_s": {key: float(value)
                      for key, value in (carried.get("run_s") or {}).items()},
            "settled_rows": int(carried.get("settled_rows", 0)),
        }

    def charged(self) -> dict:
        """Settled charges, by global total, stage and (stage, recipe, seed) run.

        The three amounts INCLUDE any carried-forward opening balance, because
        they are what every cap is measured against. ``settled_rows`` counts the
        settlements in **this** file only; the carried row count is reported
        separately by :meth:`carried_totals`, so the two are never conflated.
        """
        totals = self.carried_totals()
        totals["settled_rows"] = 0
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

    def _assert_writable(self) -> None:
        """Refuse a mutating call on a ledger that was opened to be read."""
        if self._inspect_only:
            raise LedgerIntegrityError(
                f"ledger {self.path} was opened for inspection only, so it "
                "grants nothing and is never written. Reopen it under its own "
                "protocol to reserve, mark, settle or re-snapshot")

    def _checked_run(self, stage, recipe, seed):
        spec = self.protocol.stage(stage)
        arm = self.protocol.recipe(recipe)
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
            global_s=self.protocol.TOTAL_WALL_CAP_S - totals["global_s"],
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
        re-check the carried parent chain, call the protocol's launch lock if
        it declares one, check every cap, observe contention,
        **append the row**, and only then run ``create``. A caller's ``create``
        may be arbitrarily expensive -- an output directory, a backend import, a
        model compile, an environment, a child process -- so nothing in it may
        precede the durable reservation.
        """
        self._assert_writable()
        self._load()
        open_reservation = self.pending
        if open_reservation is not None:
            raise PendingReservationError(
                f"reservation {open_reservation.reservation_id} is still "
                "unsettled; work is foreground and sequential, so no further "
                "job may start until it is settled")
        # Structural, not conventional: the carried opening balance is re-checked
        # against the live parent on EVERY reserve, in the same position as the
        # pending trap -- before any cap check, before the contention
        # observation, and before a single durable row.
        self.assert_parent_unmoved()
        # The protocol's launch lock, if it declares one, in the same position.
        self._assert_protocol_authorised("reserve")

        spec, _arm = self._checked_run(stage, recipe, seed)
        segment = _count(segment_index, "segment_index")
        first = _count(start_epoch, "start_epoch")
        last = _count(end_epoch, "end_epoch")
        if last <= first:
            raise BudgetError(f"end_epoch {last} must exceed start_epoch {first}")
        if last - first > self.protocol.SEGMENT_MAX_EPOCHS:
            raise BudgetError(
                f"segment spans {last - first} epochs, above the "
                f"{self.protocol.SEGMENT_MAX_EPOCHS}-epoch ceiling")
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
                        maximum=float(self.protocol.SEGMENT_CALL_BOUND_S))
        if bound <= 0:
            raise BudgetError(f"reserved_bound_s must be positive, got {bound}")
        if bound + allowance > self.protocol.SEGMENT_CALL_BOUND_S:
            raise ShutdownAllowanceError(
                f"a reserved bound of {bound} s plus a {allowance} s shutdown "
                f"allowance exceeds the {self.protocol.SEGMENT_CALL_BOUND_S} s "
                "call bound")

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
            "work_deadline_s": self.protocol.SEGMENT_WORK_DEADLINE_S,
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
        self._assert_writable()
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
