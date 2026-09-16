#!/usr/bin/env python
"""Segmented MyoLeg26 v2 PPO campaign runner: one bounded segment per call.

Three modes, deliberately separated by process:

``launch``
    The budget-owning parent. Imports **no** torch, no backend and no
    environment: it takes the Task 4b exclusive reservation *before* any
    GPU-heavy initialisation can possibly happen, spawns exactly one child, owns
    only that child's timeout through its own handle, and settles the ledger.
``worker``
    The child that actually trains. It never touches the ledger, so exactly one
    process ever writes it.
``select``
    Checkpoint selection over the checkpoints of **one** ``(stage, recipe,
    seed)`` run. Never pools runs: every seed and arm has its own epoch 0.

This module launches nothing by itself and grants no budget. It writes the
machinery only; no campaign seed is started here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CPU-only, torch-free imports. ``ppo_resume``/``ppo_diagnostics``/``PPO`` pull in
# torch and are imported lazily inside the worker, so ``launch`` and ``select``
# stay light and ``launch`` cannot accidentally initialise a GPU before it has
# reserved the wall time for doing so.
from msk_warp.analysis import myoleg26_selection_v2 as S
from msk_warp.analysis import ppo_v2_budget as B
from msk_warp.analysis import ppo_v2_protocol as P

SCHEMA_VERSION = "myoleg26-ppo-v2-runner-v1"
RESULT_SCHEMA = "myoleg26-ppo-v2-segment-result-v1"
SELECTION_SCHEMA = "myoleg26-ppo-v2-run-selection-v1"
LAUNCH_SCHEMA = "myoleg26-ppo-v2-launch-v1"
CHILD_SCHEMA = "myoleg26-ppo-v2-child-v1"

#: Process exit codes. The launcher inspects the **child's** code, never a
#: wrapper's.
EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_CENSORED = 3
EXIT_RUNNER_FAULT = 4
EXIT_INTERRUPTED = 5

#: The one marker that lets a parent charge its **own measured wall** instead of
#: the full reserved bound. Written only on a path that provably precedes
#: training, and honoured only when the record agrees with itself: the marker
#: present, ``training_began`` exactly ``False``, and every counter exactly zero.
#: This is the same discipline as the spawn marker -- the cheap charge needs
#: positive evidence, never merely the absence of contrary evidence.
REFUSED_BEFORE_TRAINING = "refused_before_training_began"

#: Evaluation classification (see :func:`classify_evaluation`).
EVAL_VALID = "valid"
EVAL_TRUNCATED = "wall_cap_truncated"
EVAL_INVALID = "invalid_evaluation"

#: Default shutdown/checkpoint allowance held back from the 600 s call bound.
DEFAULT_SHUTDOWN_ALLOWANCE_S = 60.0

#: Conservative multiple of the measured epoch cost required before another
#: epoch is started. This is a **stop** policy, not a capture policy: the runner
#: captures once per segment, at the boundary it stops on.
EPOCH_SAFETY_FACTOR = 2.0

#: Reserved wall inside the work deadline for the single capture + publish.
DEFAULT_CAPTURE_RESERVE_S = 30.0


class RunnerRefusal(Exception):
    """A refusal: nothing was started, or nothing may continue. Exit 2."""


class RunnerFault(Exception):
    """A bug in this runner, never a statement about a checkpoint. Exit 4.

    ``record`` carries the diagnostic record so the fault can be written into a
    field of its own, without ever being filed as a checkpoint's status.
    """

    def __init__(self, message, record=None):
        super().__init__(message)
        self.reason = str(message)
        self.record = record


class CensorRun(Exception):
    """A numerical/structural failure that censors the run for diagnosis. Exit 3."""


class WorkDeadlineExceeded(Exception):
    """Internal: the work deadline was reached inside a counted step or update."""


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_exclusive(path, value) -> str:
    """Write JSON to a path that must not exist, and return its sha256."""
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return sha256_file(path)


def _positive(value, label):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RunnerRefusal(f"{label} must be a positive finite number, got {value!r}")
    return number


# ---------------------------------------------------------------------------
# Provenance: contention and source identity
# ---------------------------------------------------------------------------

def gpu_contention_probe(run=subprocess.run) -> dict:
    """Record GPU contention at launch as **provenance only** (VALIDITY OP-05).

    Deliberately reports ``contended: None``. Declaring ``True``/``False`` would
    require a threshold, and this observation carries no threshold, no refusal,
    no timing claim and no causal claim. It is an inventory plus two numbers,
    recorded so a later wall time can be audited, and nothing else.
    """
    record = {"contended": None, "source": "nvidia-smi"}
    try:
        apps = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        usage = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except Exception as error:  # a probe failure must never cost a launch
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    record["compute_apps_returncode"] = int(apps.returncode)
    record["gpu_returncode"] = int(usage.returncode)
    record["compute_apps"] = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    record["gpu"] = [line.strip() for line in usage.stdout.splitlines() if line.strip()]
    return record


def _git(root, *args, run=subprocess.run):
    return run(["git", "-C", str(root), *args], capture_output=True, text=True)


def source_identity(root=None, files=None, *, run=subprocess.run) -> dict:
    """Three *separate* identities for the declared resume sources (IN-25).

    Raw execution bytes, git's filtered working-tree blob id and the committed
    HEAD blob id are three different objects under ``core.autocrlf``. They are
    recorded and compared separately and are never conflated or normalised.
    """
    from msk_warp.analysis import ppo_resume

    root = Path(root or ppo_resume.ROOT)
    names = tuple(ppo_resume.SOURCE_FILES if files is None else files)
    raw, working, head, status = {}, {}, {}, {}
    for relative in names:
        path = root / relative
        raw[relative] = sha256_file(path) if path.is_file() else None
        blob = _git(root, "hash-object", "--", relative, run=run)
        working[relative] = blob.stdout.strip() if blob.returncode == 0 else None
        committed = _git(root, "rev-parse", "--verify", "--quiet", f"HEAD:{relative}", run=run)
        if committed.returncode == 0 and committed.stdout.strip():
            head[relative] = committed.stdout.strip()
            status[relative] = "in_head"
        else:
            head[relative] = None
            status[relative] = "not_in_head"
    return {"raw_sha256": raw, "working_tree_blob_sha1": working,
            "head_blob_sha1": head, "head_status": status}


def compare_source_identity(metadata, current) -> list:
    """Compare recorded provenance against ``current``, one class at a time.

    ``ppo_resume`` records and never compares, so this comparison is the
    runner's obligation. The raw-bytes map, the working-tree blob map and the
    HEAD blob map are compared **independently**: a mismatch in one is reported
    as that class's mismatch and never excused by agreement in another.
    """
    problems = []
    recorded_raw = ((metadata.get("sources") or {}).get("sha256")) or {}
    if recorded_raw != current["raw_sha256"]:
        problems.append("raw_working_tree_bytes: " + _diff_names(recorded_raw, current["raw_sha256"]))
    recorded_git = metadata.get("sources_git")
    if not recorded_git:
        problems.append("sources_git: the segment carries no git identity record, so it was "
                        "captured with include_git_identity=False and cannot be resumed")
        return problems
    for key in ("working_tree_blob_sha1", "head_blob_sha1", "head_status"):
        if (recorded_git.get(key) or {}) != current[key]:
            problems.append(f"{key}: " + _diff_names(recorded_git.get(key) or {}, current[key]))
    return problems


FREEZE_BINDING_KEYS = ("freeze_sha256", "model_sha256", "compiled_model_sha256",
                       "config_sha256")


def freeze_bindings(freeze_path) -> dict:
    """The freeze-derived hashes this runner places in ``extra`` and compares.

    ``ppo_resume`` records these and never compares them, so they are worth
    nothing until the runner checks them itself on the way back in.
    """
    manifest = read_json(freeze_path)
    return {
        "freeze_path": str(Path(freeze_path)),
        "freeze_sha256": sha256_file(freeze_path),
        "model_sha256": manifest.get("model_sha256"),
        "compiled_model_sha256": manifest.get("compiled_model_sha256"),
        "config_sha256": sha256_file(P.V2_CONFIG),
    }


def compare_freeze_bindings(extra, bindings) -> list:
    problems = []
    for key in FREEZE_BINDING_KEYS:
        recorded = (extra or {}).get(key)
        current = (bindings or {}).get(key)
        if recorded != current:
            problems.append(f"{key}: segment recorded {recorded!r}, current is {current!r}")
    return problems


def segment_extra(*, stage, recipe, seed, segment_index, parent_segment_sha256,
                  bindings, recorder_state, replayed) -> dict:
    """The caller metadata a segment carries.

    ``seed``, ``segment_index`` and ``parent_segment_sha256`` sit at the top
    level because that is where ``ppo_resume._expected_value`` looks for them
    when ``strict_expect`` binds them.
    """
    extra = {
        "runner_schema": SCHEMA_VERSION,
        "stage": str(stage),
        "recipe": str(recipe),
        "seed": int(seed),
        "segment_index": int(segment_index),
        "parent_segment_sha256": parent_segment_sha256,
        "episode_accumulators": {
            "returns": [float(value) for value in recorder_state["returns"]],
            "lengths": [int(value) for value in recorder_state["lengths"]],
        },
        "replayed": {key: int(value) for key, value in dict(replayed).items()},
    }
    extra.update({key: (bindings or {}).get(key) for key in FREEZE_BINDING_KEYS})
    extra["freeze_path"] = (bindings or {}).get("freeze_path")
    return extra


def publish_boundary(algo, env, path, *, epoch, extra, label=None) -> dict:
    """Capture a completed epoch boundary and publish it exactly once.

    This is the runner's **only** ``capture_state`` call site, and it always
    requests the git identity: a published segment is the one artifact that
    crosses a process boundary and gets resumed later, so it is exactly what
    needs full provenance. There is deliberately no second, cheaper capture
    category that could be mistaken for a resumable segment.
    """
    from msk_warp.analysis import ppo_resume

    state = ppo_resume.capture_state(algo, env, extra, epoch=int(epoch),
                                     include_git_identity=True)
    return ppo_resume.write_segment(state, path, label=label)


def resume_boundary(path, algo, env, *, epoch, seed, segment_index,
                    parent_segment_sha256, bindings, expected_segment_sha256=None) -> tuple:
    """Bind all five identity keys, compare provenance, then restore.

    Order is load-bearing: the lineage bytes, then the five-key identity, then
    every provenance class, and only then any mutation of the target.
    """
    from msk_warp.analysis import ppo_resume

    path = Path(path)
    if expected_segment_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_segment_sha256:
            raise RunnerRefusal(
                f"parent segment sha256 mismatch: the lineage records "
                f"{expected_segment_sha256!r} but {path} hashes to {actual!r}")

    expect = ppo_resume.strict_expect(
        epoch=int(epoch), seed=int(seed), segment_index=int(segment_index),
        parent_segment_sha256=parent_segment_sha256)
    if set(expect) != set(ppo_resume.REQUIRED_IDENTITY_KEYS):
        raise RunnerFault(
            "strict_expect did not bind every required identity key: "
            f"{sorted(set(ppo_resume.REQUIRED_IDENTITY_KEYS) - set(expect))}")
    try:
        state = ppo_resume.read_segment(path, expect=expect)
    except ppo_resume.ResumeError as error:
        raise RunnerRefusal(f"segment refused: {error}") from error

    problems = compare_source_identity(state.metadata, source_identity())
    problems += compare_freeze_bindings(state.extra, bindings)
    if problems:
        raise RunnerRefusal(
            "provenance mismatch, refusing to resume (each identity class is compared "
            "separately and never normalised): " + "; ".join(problems))

    try:
        report = ppo_resume.restore_state(state, algo, env)
    except ppo_resume.ResumeError as error:
        raise RunnerRefusal(f"restore refused: {error}") from error
    return report, state


def _diff_names(recorded, current) -> str:
    names = sorted(set(recorded) | set(current))
    return ", ".join(name for name in names if recorded.get(name) != current.get(name))


# ---------------------------------------------------------------------------
# Evaluation cause split
# ---------------------------------------------------------------------------

def classify_evaluation(evaluation) -> tuple:
    """Split a rejected evaluation by **cause**, never by parsing prose.

    ``InvalidEvaluationError`` carries only a free-text ``reason``, so the class
    cannot discriminate. The discriminator comes from the record itself, which
    this runner wrote: ``evaluate_policy`` sets ``complete`` from
    ``not any(end_reason == "wall_cap")``, and that ``end_reason`` is written
    only on its deadline branch.

    A wall-cap truncation is a **scheduling artifact**, so it excludes that one
    checkpoint and selection continues. Every other cause — nonfinite metric,
    missing key, non-positive ``control_dt``, empty or malformed ``episodes`` —
    is a numerical/structural failure and censors the run.

    Truncation is only claimed when it is the **sole** defect: a copy with
    ``complete=True`` must validate. The copy's rank is discarded and never
    compared, so selection still happens only inside ``select_best_checkpoint``.
    """
    try:
        S.behavior_rank_v2(evaluation)
    except S.InvalidEvaluationError as error:
        reason = error.reason
    else:
        return EVAL_VALID, None

    if not isinstance(evaluation, Mapping):
        return EVAL_INVALID, reason
    episodes = evaluation.get("episodes")
    if not isinstance(episodes, (list, tuple)):
        return EVAL_INVALID, reason
    # The deadline branch of ``evaluate_policy`` is the only writer of this
    # value, and ``complete`` is derived from exactly this predicate.
    truncated = any(isinstance(row, Mapping) and row.get("end_reason") == "wall_cap"
                    for row in episodes)
    if not truncated or evaluation.get("complete") is not False:
        return EVAL_INVALID, reason

    probe = dict(evaluation)
    probe["complete"] = True
    try:
        S.behavior_rank_v2(probe)       # validator only; the returned key is discarded
    except S.InvalidEvaluationError:
        return EVAL_INVALID, reason     # truncation was not the only defect
    return EVAL_TRUNCATED, reason


# ---------------------------------------------------------------------------
# Launch planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaunchPlan:
    stage: str
    recipe: str
    seed: int
    segment_index: int
    start_epoch: int
    end_epoch: int
    work_deadline_s: float
    shutdown_allowance_s: float
    reserved_bound_s: float
    out_dir: Path
    parent_segment: Path | None
    parent_result: Path | None

    @property
    def epochs(self) -> int:
        return self.end_epoch - self.start_epoch


def resolve_plan(args, remaining=None) -> LaunchPlan:
    """Validate the whole launch identity before anything durable happens.

    Every refusal here precedes the reservation, so it leaves **no** ledger row
    at all. ``remaining``, when supplied, is the live
    :class:`~msk_warp.analysis.ppo_v2_budget.Remaining` triple; the tightest of
    the global, stage and per-seed caps binds, never a timeout alone.
    """
    freeze = Path(args.freeze)
    if not freeze.is_file():
        raise RunnerRefusal(f"freeze manifest is missing or is not a regular file: {freeze}")

    try:
        spec = P.stage(args.stage)
        P.recipe(args.recipe)
    except P.ProtocolError as error:
        raise RunnerRefusal(str(error)) from error
    if args.seed not in spec.seeds:
        raise RunnerRefusal(
            f"seed {args.seed} is not a scheduled seed of stage {spec.key!r} {spec.seeds}; "
            "per-seed caps are keyed (stage, recipe, seed), so an unscheduled seed would "
            "open a fresh unbudgeted allowance")

    rounds = P.segment_rounds(spec.epoch_cap_per_seed)
    if not 0 <= args.segment_index < len(rounds):
        raise RunnerRefusal(
            f"segment index {args.segment_index} is outside 0..{len(rounds) - 1} for stage "
            f"{spec.key!r} (epoch cap {spec.epoch_cap_per_seed})")
    round_ = rounds[args.segment_index]

    epochs = round_.epochs if args.epochs is None else int(args.epochs)
    if epochs <= 0 or epochs > round_.epochs or epochs > P.SEGMENT_MAX_EPOCHS:
        raise RunnerRefusal(
            f"epoch budget {epochs} must be 1..{round_.epochs} for this segment, and never "
            f"above the sealed {P.SEGMENT_MAX_EPOCHS}-epoch segment ceiling")
    start_epoch = round_.start_epoch
    end_epoch = start_epoch + epochs

    parent_segment = args.parent_segment
    parent_result = args.parent_result
    if args.segment_index == 0:
        if parent_segment or parent_result:
            raise RunnerRefusal(
                "segment 0 trains from scratch, so a parent segment is refused rather than "
                "ignored: silently dropping it would start over while the caller believed "
                "it was resuming")
        parent_segment = parent_result = None
    else:
        if not parent_segment or not parent_result:
            raise RunnerRefusal(
                "a resume segment requires both --parent-segment and --parent-result: the "
                "lineage sha256 and the identity to bind come from the parent's result")
        parent_segment = Path(parent_segment)
        parent_result = Path(parent_result)

    work_deadline_s = _positive(args.work_deadline_s, "--work-deadline-s (work deadline)")
    if work_deadline_s > P.SEGMENT_WORK_DEADLINE_S:
        raise RunnerRefusal(
            f"work deadline {work_deadline_s} s is above the sealed "
            f"{P.SEGMENT_WORK_DEADLINE_S} s ceiling")

    allowance = _positive(getattr(args, "shutdown_allowance_s", DEFAULT_SHUTDOWN_ALLOWANCE_S),
                          "--shutdown-allowance-s (shutdown/checkpoint allowance)")
    if remaining is not None and remaining.least <= allowance:
        raise RunnerRefusal(
            f"the {remaining.binding} cap has {remaining.least} s remaining, which cannot fund "
            f"a bounded call plus a {allowance} s shutdown allowance")

    call_cap = float(P.SEGMENT_CALL_BOUND_S) if remaining is None else remaining.call_cap
    supplied = getattr(args, "reserved_bound_s", None)
    bound = (call_cap - allowance) if supplied is None else _positive(supplied, "--reserved-bound-s")
    if bound + allowance > P.SEGMENT_CALL_BOUND_S:
        raise RunnerRefusal(
            f"a {bound} s reserved bound plus a {allowance} s shutdown allowance exceeds the "
            f"sealed {P.SEGMENT_CALL_BOUND_S} s call bound")
    if remaining is not None and bound + allowance > remaining.least:
        raise RunnerRefusal(
            f"the {remaining.binding} cap has {remaining.least} s remaining, which cannot hold "
            f"a {bound} s bound plus a {allowance} s shutdown allowance")
    if bound < work_deadline_s:
        raise RunnerRefusal(
            f"reserved bound {bound} s is below the {work_deadline_s} s work deadline: the "
            "child would be killed mid-epoch instead of stopping at a completed boundary")

    out_dir = getattr(args, "out_dir", None)
    out_dir = (Path(out_dir) if out_dir
               else Path(args.run_dir) / f"segment_{args.segment_index:04d}")
    return LaunchPlan(
        stage=spec.key, recipe=args.recipe, seed=int(args.seed),
        segment_index=int(args.segment_index), start_epoch=start_epoch, end_epoch=end_epoch,
        work_deadline_s=work_deadline_s, shutdown_allowance_s=allowance,
        reserved_bound_s=bound, out_dir=out_dir,
        parent_segment=parent_segment, parent_result=parent_result)


# ---------------------------------------------------------------------------
# Worker: episode events and counted stepping
# ---------------------------------------------------------------------------

_TORCH = None


def _torch():
    """Import torch on first use, so ``launch`` and ``select`` stay torch-free."""
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def _censor_nonfinite(label, tensor) -> None:
    torch = _torch()
    if tensor is None:
        return
    if not bool(torch.isfinite(tensor).all()):
        raise CensorRun(
            f"nonfinite {label}: the run is censored for diagnosis at the first nonfinite "
            "value, with no retry, no clamp and no skipped world")


def _mask_list(value, count):
    if value is None:
        return [False] * count
    return [bool(item) for item in value.detach().reshape(-1).tolist()]


class EpisodeEventRecorder:
    """Per-world running episode accumulators plus this epoch's completed events.

    PPO's own ``episode_loss`` accumulates **negated** reward, its ``*_his`` lists
    are unbounded run-lifetime histories and its meters are a capped 100-sample
    rolling mean, so none of them describes this epoch and none of them is read
    here. These accumulators are the runner's own, and because an episode may
    span an epoch *and* a segment boundary they are carried across a resume in
    the segment ``extra``.

    Accumulation stays on the device and only a completed episode is read back,
    so the instrumentation adds no per-world host synchronisation.
    """

    def __init__(self, num_worlds, control_dt, *, device="cpu"):
        torch = _torch()
        self.num_worlds = int(num_worlds)
        self.control_dt = float(control_dt)
        self.device = device
        self._returns = torch.zeros(self.num_worlds, dtype=torch.float64, device=device)
        self._lengths = torch.zeros(self.num_worlds, dtype=torch.long, device=device)
        self.events = []

    def reset_epoch(self) -> None:
        """Start a new epoch's event list. The running accumulators persist."""
        self.events = []

    def observe(self, rewards, dones, extras) -> None:
        rewards = rewards.detach().reshape(-1)
        self._returns += rewards.to(self._returns.dtype)
        self._lengths += 1
        done = dones.detach().reshape(-1).bool()
        if not bool(done.any()):
            return
        worlds = done.nonzero(as_tuple=False).reshape(-1).tolist()
        returns = self._returns[worlds].tolist()
        lengths = self._lengths[worlds].tolist()
        extras = extras or {}
        terminated = _mask_list(extras.get("terminated"), self.num_worlds)
        truncated = _mask_list(extras.get("truncated"), self.num_worlds)
        flags = {name: _mask_list(value, self.num_worlds)
                 for name, value in (extras.get("failure_flags") or {}).items()}
        for position, world in enumerate(worlds):
            length = int(lengths[position])
            self.events.append({
                "world": int(world),
                "length_controls": length,
                "duration_s": float(length * self.control_dt),
                # The episode's reward SUM. Never PPO's negated episode_loss.
                "return": float(returns[position]),
                "end_reason": ("task_failure" if terminated[world]
                               else "timeout" if truncated[world] else "horizon"),
                "failure_flags": {name: bool(row[world]) for name, row in flags.items()},
            })
        self._returns[worlds] = 0.0
        self._lengths[worlds] = 0

    def state(self) -> dict:
        """Plain data for the resume ``extra``; no tensors, no NumPy arrays."""
        return {"returns": [float(value) for value in self._returns.tolist()],
                "lengths": [int(value) for value in self._lengths.tolist()]}

    def load(self, state) -> None:
        returns = list(state["returns"])
        lengths = list(state["lengths"])
        if len(returns) != self.num_worlds or len(lengths) != self.num_worlds:
            raise RunnerRefusal(
                f"the recorded episode accumulators cover {len(returns)} worlds but this "
                f"environment has {self.num_worlds}")
        torch = _torch()
        self._returns = torch.tensor(returns, dtype=torch.float64, device=self.device)
        self._lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)


@dataclass
class SegmentCounters:
    """Counted control calls, kept apart by what actually happened to them.

    ``attempted`` is incremented **before** the environment is stepped and
    ``returned`` after it comes back, so a call that failed inside the simulator
    is visible rather than absorbed. ``boundary`` counts only the calls belonging
    to epochs whose update also completed: it is the work a published boundary
    can honestly claim.
    """

    attempted_control_calls: int = 0
    returned_control_calls: int = 0
    boundary_control_calls: int = 0
    evaluation_attempted_calls: int = 0
    evaluation_returned_calls: int = 0
    partial_update_cost_s: float = 0.0


@dataclass
class SegmentOutcome:
    completed_epoch: int = 0
    completed_epochs: int = 0
    boundary_is_live: bool = False
    stop_reason: str = "stub"
    censored: bool = False
    censor_reason: str | None = None
    partial_detail: str | None = None
    losses: list = field(default_factory=list)
    epoch_events: list = field(default_factory=list)
    evaluations: list = field(default_factory=list)
    counters: SegmentCounters = field(default_factory=SegmentCounters)


def _install_counted_step(env, recorder, counters, *, clock, work_deadline):
    """Wrap ``env.step`` with counting, the deadline and the nonfinite censor.

    The wrapper draws no random numbers, touches no parameter, gradient,
    optimizer or normaliser, and returns the environment's own result object
    unchanged, so PPO's behaviour is preserved bitwise. The single preregistered
    exception to that neutrality is the nonfinite censor, which stops the run.
    """
    torch = _torch()
    original = env.step

    def counted_step(action, *args, **kwargs):
        if clock() >= work_deadline:
            raise WorkDeadlineExceeded(
                "the work deadline was reached inside a counted training step, so this "
                "epoch is censored rather than allowed to overrun unchecked")
        _censor_nonfinite("action", action)
        counters.attempted_control_calls += 1
        result = original(action, *args, **kwargs)
        counters.returned_control_calls += 1
        observation, reward, done, extras = result[0], result[1], result[2], result[3]
        # One fused device-side reduction, then a single host read.
        if not bool(torch.isfinite(observation).all() & torch.isfinite(reward).all()):
            _censor_nonfinite("observation", observation)
            _censor_nonfinite("reward", reward)
        recorder.observe(reward, done, extras)
        return result

    env.step = counted_step

    def restore():
        env.step = original

    return restore


def epoch_diagnostics(algo) -> dict:
    """Task 4a's stateless observability helpers, applied to this epoch's buffers.

    The RNG state is deliberately **not** saved and restored around this call:
    restoring would hide a draw rather than prove there is none. Neutrality is
    asserted by the tests instead.
    """
    from msk_warp.analysis import ppo_diagnostics as diagnostics
    return {
        "kl": diagnostics.approx_kl_clip(algo.actor, algo.buf_obs, algo.buf_actions,
                                         algo.buf_log_probs, clip_range=algo.clip_range),
        "logstd": diagnostics.logstd_summary(algo.actor),
        "excitation": diagnostics.near_bound_excitation(algo.buf_actions),
    }


def _episode_event_summary(events) -> dict:
    from msk_warp.analysis import ppo_diagnostics as diagnostics
    return diagnostics.episode_event_summary(events)


def _censor_nonfinite_losses(losses) -> None:
    for name, value in dict(losses).items():
        if not math.isfinite(float(value)):
            raise CensorRun(
                f"nonfinite update loss {name!r}: the run is censored for diagnosis rather "
                "than retried")


def _censor_nonfinite_parameters(algo) -> None:
    torch = _torch()
    for label, network in (("actor", algo.actor), ("critic", algo.critic)):
        for name, parameter in network.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise CensorRun(
                    f"nonfinite parameter {label}.{name}: the run is censored for diagnosis "
                    "rather than retried")


def train_segment(algo, env, *, start_epoch, end_epoch, max_epochs_total, clock,
                  work_deadline, recorder, capture_reserve_s=DEFAULT_CAPTURE_RESERVE_S,
                  evaluate=None, evaluation_epochs=(), diagnostics=None) -> SegmentOutcome:
    """Train one bounded segment, stopping only at completed epoch boundaries.

    The deadline is tested in three places, because any one of them alone leaves
    a hole: before each epoch (so a clean boundary is reachable), on every
    counted step (so one epoch cannot overrun unchecked) and before each update
    (so an update never starts without the wall to finish it and checkpoint).

    ``boundary_is_live`` is the whole contract with the caller. It is ``True``
    only when the live objects correspond exactly to ``completed_epoch``. The
    moment an epoch is entered and not completed it becomes ``False``, and the
    caller must then publish nothing and fall back to the parent segment: a
    partially updated epoch is never published or labelled as a boundary.
    """
    wanted_evaluations = {int(value) for value in evaluation_epochs}
    outcome = SegmentOutcome(completed_epoch=int(start_epoch), boundary_is_live=True,
                             stop_reason="epoch_budget")
    counters = outcome.counters
    restore = _install_counted_step(env, recorder, counters, clock=clock,
                                    work_deadline=work_deadline)
    begin_epoch = getattr(env, "begin_epoch", None)
    epoch_cost = 0.0
    update_cost = 0.0
    try:
        # Checked once before the first rollout, then after every update (which
        # is the same boundary for every later epoch). A nonfinite parameter
        # never reaches the step wrapper as a nonfinite action: PPO builds a
        # ``torch.distributions.Normal`` from it first, and that raises on a
        # nonfinite location. Catching it here names the real cause.
        _censor_nonfinite_parameters(algo)
        if evaluate is not None and int(start_epoch) in wanted_evaluations:
            outcome.evaluations.append(evaluate(int(start_epoch)))
        for epoch in range(int(start_epoch), int(end_epoch)):
            if clock() + EPOCH_SAFETY_FACTOR * epoch_cost + capture_reserve_s > work_deadline:
                outcome.stop_reason = "work_deadline_epoch_boundary"
                break
            recorder.reset_epoch()
            epoch_started = clock()
            if callable(begin_epoch):
                begin_epoch(epoch=epoch, max_epochs=max_epochs_total)
            algo.collect_rollout()
            if clock() + EPOCH_SAFETY_FACTOR * update_cost + capture_reserve_s > work_deadline:
                raise WorkDeadlineExceeded(
                    "the remaining budget cannot fund another update plus the checkpoint, so "
                    "the update was never started and this epoch is censored")
            update_started = clock()
            try:
                losses = algo.update()
            except WorkDeadlineExceeded:
                counters.partial_update_cost_s += float(clock() - update_started)
                raise
            update_cost = max(update_cost, float(clock() - update_started))
            _censor_nonfinite_losses(losses)
            _censor_nonfinite_parameters(algo)

            algo.iter_count += 1
            completed = epoch + 1
            outcome.completed_epoch = completed
            outcome.completed_epochs += 1
            counters.boundary_control_calls = counters.returned_control_calls
            outcome.losses.append(
                {"epoch": completed, **{key: float(value) for key, value in losses.items()}})
            record = {"epoch": completed, "events": list(recorder.events),
                      "summary": _episode_event_summary(recorder.events)}
            if diagnostics:
                record["diagnostics"] = epoch_diagnostics(algo)
            outcome.epoch_events.append(record)
            epoch_cost = max(epoch_cost, float(clock() - epoch_started))
            if evaluate is not None and completed in wanted_evaluations:
                outcome.evaluations.append(evaluate(completed))
    except WorkDeadlineExceeded as error:
        outcome.boundary_is_live = False
        outcome.stop_reason = "work_deadline_partial_epoch"
        outcome.partial_detail = str(error)
    except CensorRun as error:
        outcome.boundary_is_live = False
        outcome.censored = True
        outcome.censor_reason = str(error)
        outcome.stop_reason = "censored_nonfinite"
    finally:
        restore()
    return outcome


# ---------------------------------------------------------------------------
# Selection over one run/arm
# ---------------------------------------------------------------------------

def _is_refusal_record(payload) -> bool:
    """Whether a result record positively declares itself a pre-training refusal.

    **Both** keys are required, so a trained record can never be mistaken for a
    refusal and quietly dropped from the campaign record.
    """
    return (isinstance(payload, Mapping)
            and payload.get("status") == REFUSED_BEFORE_TRAINING
            and payload.get("training_began") is False)


def collect_run_records(run_dir) -> tuple:
    """``(candidates, refused)`` for ONE ``(stage, recipe, seed)`` run.

    Never pools runs: every seed and every arm has its own epoch 0, so a pooled
    list would collide on ``epoch`` by construction.

    A **pre-training refusal** declares no identity triple *on purpose*:
    ``resolve_plan`` may be the very thing that refused, so an identity written
    there would be unvalidated, and a wrong identity is worse than none. It also
    produced no checkpoint and no evaluation. Such a record therefore contributes
    **neither** an identity **nor** a candidate, and is reported separately as a
    non-candidate rather than silently ignored.

    The cross-run guard stays strict for everything else: a *trained* record
    missing an identity key still contributes ``None`` and still aborts.
    """
    run_dir = Path(run_dir)
    identities, candidates, refused = set(), [], []
    for segment in sorted(run_dir.glob("segment_*")):
        result_path = segment / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            if _is_refusal_record(result):
                refused.append(str(result_path))
            else:
                identities.add((result.get("stage"), result.get("recipe"),
                                result.get("seed")))
        for path in sorted(segment.glob("selection_*.json")):
            record = read_json(path)
            candidates.append({
                "epoch": int(record["epoch"]),
                "evaluation": record["evaluation"],
                "checkpoint": record.get("checkpoint"),
                "evaluation_path": str(path),
            })
    if len(identities) > 1:
        # ``key=repr`` keeps this buildable and deterministic over a set that
        # mixes strings, ints and ``None``; a bare ``sorted`` raises TypeError.
        raise RunnerFault(
            "this run directory mixes more than one (stage, recipe, seed) identity: "
            f"{sorted(identities, key=repr)}. Candidates must be scoped to a single "
            "run/arm, because every seed and every arm has its own epoch 0")
    return candidates, refused


def collect_candidates(run_dir) -> list:
    """Every evaluated checkpoint of ONE run. See :func:`collect_run_records`."""
    return collect_run_records(run_dir)[0]


def _selection_record(status, *, statuses, excluded, considered, selected=None,
                      fault=None, refused=()):
    record = {
        "schema_version": SELECTION_SCHEMA,
        "status": status,
        "selected": selected,
        "candidates_considered": int(considered),
        "candidates_compared": int(considered) - len(excluded),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "wall_cap_truncations": sum(1 for entry in excluded
                                    if entry["status"] == EVAL_TRUNCATED),
        "checkpoint_status": statuses,
        # Segments that refused before training. Not candidates, never
        # scored, and never silently dropped from the record either.
        "refused_segments": [str(path) for path in refused],
    }
    if fault is not None:
        record["fault"] = fault
    return record


def _classifiable(candidate) -> bool:
    """Whether this candidate's shape lets it be classified at all."""
    if not isinstance(candidate, Mapping):
        return False
    if "evaluation" not in candidate or "epoch" not in candidate:
        return False
    epoch = candidate["epoch"]
    return isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0


def _delegate_set_fault(candidates, statuses, refused=()):
    """Let the upstream guard rule on the SET, so its refusal is the authority."""
    try:
        S.select_best_checkpoint(candidates)
    except S.InvalidCandidateSetError as error:
        raise RunnerFault(error.reason, record=_selection_record(
            "runner_fault", statuses=statuses, excluded=[], considered=len(statuses),
            refused=refused,
            fault={"kind": "InvalidCandidateSetError", "reason": error.reason})) from error
    raise RunnerFault(
        "the candidate set was rejected here but accepted upstream, which is itself a "
        "runner bug", record=_selection_record(
            "runner_fault", statuses=statuses, excluded=[], considered=len(statuses),
            refused=refused,
            fault={"kind": "inconsistent_set_validation",
                   "reason": "runner and upstream disagree on this candidate set"}))


def select_run(candidates, *, refused=()) -> dict:
    """Classify by cause, exclude truncations, censor real faults, then select.

    The two upstream refusal classes are handled entirely separately and never
    merged into one ``except ValueError``:

    * a **wall-cap truncation** excludes that one checkpoint, visibly, and
      selection continues over the rest;
    * any other invalid-evaluation cause **censors the run** for diagnosis; and
    * a **set-level** fault aborts as a runner fault, recorded in a field of its
      own and never written into any checkpoint's status.
    """
    refused = [str(path) for path in refused]
    if isinstance(candidates, (str, bytes)) or isinstance(candidates, Mapping):
        _delegate_set_fault(candidates, [], refused)
    listed = list(candidates)
    if not listed and refused:
        # Every segment refused before training, so nothing was ever produced
        # to select from. That is an operational outcome, not a runner bug: it
        # is recorded as incomplete, never invented and never crashed on.
        return _selection_record("incomplete", statuses=[], excluded=[],
                                 considered=0, refused=refused)
    if not listed or not all(_classifiable(candidate) for candidate in listed):
        _delegate_set_fault(candidates, [], refused)

    statuses, survivors, excluded = [], [], []
    for candidate in listed:
        status, reason = classify_evaluation(candidate["evaluation"])
        entry = {"epoch": int(candidate["epoch"]), "status": status,
                 "checkpoint": candidate.get("checkpoint"), "reason": reason}
        if status == EVAL_INVALID:
            raise CensorRun(
                f"invalid evaluation at epoch {candidate['epoch']} "
                f"({candidate.get('checkpoint')}): {reason}. The run is censored for "
                "diagnosis; no valid sibling checkpoint is selected in its place")
        statuses.append(entry)
        if status == EVAL_TRUNCATED:
            entry["excluded"] = True
            excluded.append(dict(entry))
        else:
            survivors.append(candidate)

    if not survivors:
        # A legitimate outcome, not a set-level fault: there is simply no valid
        # selection. Never a fallback to an excluded checkpoint, never a zero.
        return _selection_record("incomplete", statuses=statuses, excluded=excluded,
                                 considered=len(listed), refused=refused)
    try:
        best = S.select_best_checkpoint(survivors)
    except S.InvalidCandidateSetError as error:
        raise RunnerFault(error.reason, record=_selection_record(
            "runner_fault", statuses=statuses, excluded=excluded, considered=len(listed),
            refused=refused,
            fault={"kind": "InvalidCandidateSetError", "reason": error.reason})) from error
    return _selection_record("selected", statuses=statuses, excluded=excluded,
                             considered=len(listed), selected=best, refused=refused)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _argv_digest(argv) -> str:
    return hashlib.sha256("\x00".join(str(item) for item in argv).encode("utf-8")).hexdigest()


def _worker_argv(args, plan) -> list:
    argv = [sys.executable, str(_HERE), "worker",
            "--stage", plan.stage, "--recipe", plan.recipe, "--seed", str(plan.seed),
            "--segment-index", str(plan.segment_index),
            "--freeze", str(Path(args.freeze)),
            "--epochs", str(plan.epochs),
            "--work-deadline-s", str(plan.work_deadline_s),
            "--capture-reserve-s", str(getattr(args, "capture_reserve_s",
                                               DEFAULT_CAPTURE_RESERVE_S)),
            "--replayed-control-transitions",
            str(getattr(args, "replayed_control_transitions", 0)),
            "--replayed-physics-steps", str(getattr(args, "replayed_physics_steps", 0)),
            "--device", str(args.device),
            "--out-dir", str(plan.out_dir)]
    if plan.parent_segment is not None:
        argv += ["--parent-segment", str(plan.parent_segment),
                 "--parent-result", str(plan.parent_result)]
    return argv


_COUNTER_KEYS = ("attempted_control_transitions", "completed_control_transitions",
                 "replayed_control_transitions", "attempted_physics_steps",
                 "completed_physics_steps", "replayed_physics_steps")


def _zero_accounting() -> dict:
    """Explicit zeros for a segment that provably did nothing.

    These are a **known** zero, not an unknown written down as zero: they are
    only ever emitted beside the :data:`REFUSED_BEFORE_TRAINING` marker.
    """
    accounting = {key: 0 for key in _COUNTER_KEYS}
    accounting.update({
        "worlds": 0, "substeps": int(P.PHYSICS_SUBSTEPS),
        "training_control_transitions": 0,
        "training_boundary_control_transitions": 0,
        "evaluation_control_transitions": 0,
        "discarded_control_transitions": 0,
        "discarded_physics_steps": 0,
        "partial_update_cost_s": 0.0,
        "attempted_control_calls": 0, "returned_control_calls": 0,
        "exact_completed_calls": True,
    })
    return accounting


def _counters_from_payload(payload):
    try:
        accounting = payload["accounting"]
        counters = B.SegmentCounters(
            attempted_control_transitions=int(accounting["attempted_control_transitions"]),
            completed_control_transitions=int(accounting["completed_control_transitions"]),
            replayed_control_transitions=int(accounting["replayed_control_transitions"]),
            attempted_physics_steps=int(accounting["attempted_physics_steps"]),
            completed_physics_steps=int(accounting["completed_physics_steps"]),
            replayed_physics_steps=int(accounting["replayed_physics_steps"]),
            partial_update_cost_s=float(accounting["partial_update_cost_s"]))
        counters.as_dict()      # the ledger's own validation, applied before settling
        return counters
    except Exception:
        return None


def _counters_from_result(path):
    """The child's own measured counters, or ``None`` if they are not trustworthy."""
    try:
        return _counters_from_payload(read_json(path))
    except Exception:
        return None


def _settlement_from_result(path):
    """``(counters, kind)`` for a child's record; ``(None, "unknown")`` charges the bound.

    Exactly **two** shapes are admissible, and they are disjoint by construction:

    * a **trained** record declares ``training_began: True`` and carries its own
      measured counters;
    * a **refusal** record carries the :data:`REFUSED_BEFORE_TRAINING` marker with
      ``training_began: False`` and every counter exactly zero.

    Anything else -- absent, unparseable, counters missing, a marker contradicted
    by ``training_began`` or by a nonzero counter, or a record that declares
    neither shape -- is *unknown* and settles the **full reserved bound**. So a
    forged or self-inconsistent marker cannot buy the cheap charge, and a refusal
    raised after training began cannot reach it either, because that path writes
    the real record with ``training_began: True``.
    """
    try:
        payload = read_json(path)
    except Exception:
        return None, "unknown"
    if not isinstance(payload, Mapping):
        return None, "unknown"
    counters = _counters_from_payload(payload)
    if counters is None:
        return None, "unknown"

    began = payload.get("training_began")
    recorded = counters.as_dict()
    if payload.get("status") == REFUSED_BEFORE_TRAINING:
        if began is not False:
            return None, "unknown"
        if any(recorded[key] for key in _COUNTER_KEYS):
            return None, "unknown"
        if recorded["partial_update_cost_s"]:
            return None, "unknown"
        return counters, "refused_before_training"
    if began is not True:
        return None, "unknown"
    return counters, "trained"


def run_launch(args, *, ledger=None, spawn=subprocess.Popen, clock=time.perf_counter,
               probe=None) -> int:
    """Reserve, spawn exactly one child, own its timeout, settle. No GPU here.

    This function imports no torch, no backend and no environment, so the
    reservation provably precedes every GPU-heavy initialisation: all of that
    happens in the child, after the wall time for it has been claimed.
    """
    ledger_path = Path(args.ledger)
    if ledger is None:
        if probe is None and not getattr(args, "no_contention_probe", False):
            probe = gpu_contention_probe
        try:
            if getattr(args, "create_ledger", False) and not ledger_path.exists():
                ledger = B.BudgetLedger.create(ledger_path, protocol_digest=P.protocol_digest(),
                                               contention_probe=probe)
            else:
                ledger = B.BudgetLedger.open(ledger_path, contention_probe=probe)
        except B.BudgetError as error:
            raise RunnerRefusal(str(error)) from error

    try:
        ledger.assert_protocol_unchanged(P.protocol_digest())
        remaining = ledger.remaining(args.stage, args.recipe, args.seed)
    except (B.BudgetError, P.ProtocolError) as error:
        raise RunnerRefusal(str(error)) from error

    # Every refusal up to here leaves the ledger byte-identical.
    plan = resolve_plan(args, remaining)

    launch_record = {
        "schema_version": LAUNCH_SCHEMA, "runner_schema": SCHEMA_VERSION,
        "stage": plan.stage, "recipe": plan.recipe, "seed": plan.seed,
        "segment_index": plan.segment_index, "start_epoch": plan.start_epoch,
        "end_epoch": plan.end_epoch, "epochs": plan.epochs,
        "work_deadline_s": plan.work_deadline_s,
        "reserved_bound_s": plan.reserved_bound_s,
        "shutdown_allowance_s": plan.shutdown_allowance_s,
        "remaining_before": remaining.as_dict(),
        "freeze": str(Path(args.freeze)), "freeze_sha256": sha256_file(args.freeze),
        "parent_segment": None if plan.parent_segment is None else str(plan.parent_segment),
        "parent_result": None if plan.parent_result is None else str(plan.parent_result),
        "device": str(args.device),
    }

    def _create():
        """Runs inside ``reserve``: cheap, and provably before any child."""
        plan.out_dir.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(plan.out_dir / "launch.json", launch_record)
        if plan.parent_segment is not None:
            for label, path in (("parent segment", plan.parent_segment),
                                ("parent result", plan.parent_result)):
                if not path.is_file():
                    raise RunnerRefusal(f"{label} is missing: {path}")

    try:
        reservation = ledger.reserve(
            stage=plan.stage, recipe=plan.recipe, seed=plan.seed,
            segment_index=plan.segment_index, start_epoch=plan.start_epoch,
            end_epoch=plan.end_epoch, reserved_bound_s=plan.reserved_bound_s,
            shutdown_allowance_s=plan.shutdown_allowance_s, create=_create)
    except B.PrelaunchError as error:
        raise RunnerRefusal(
            "pre-launch setup failed before anything was started; the reservation was "
            f"settled at a zero charge and both rows stay in the chain: {error}") from error
    except (B.BudgetError, P.ProtocolError) as error:
        raise RunnerRefusal(str(error)) from error

    handles = []
    try:
        handles.append((plan.out_dir / "child.stdout.log").open("xb"))
        handles.append((plan.out_dir / "child.stderr.log").open("xb"))
        argv = _worker_argv(args, plan)
    except Exception as error:
        for handle in handles:
            handle.close()
        # Provably pre-spawn: this frame holds the reservation and has written no
        # marker. ``settle_aborted`` gives the failure an auditable forward path
        # instead of wedging every later job; no row is retracted.
        ledger.settle_aborted(reservation, detail=(
            "pre-spawn failure, no child was started: "
            f"{type(error).__name__}: {error}"))
        raise RunnerRefusal(
            f"nothing was started: {type(error).__name__}: {error}") from error

    out_handle, err_handle = handles
    try:
        started = clock()
        ledger.mark_spawn(reservation, detail=f"argv_sha256={_argv_digest(argv)}")
        process = spawn(argv, stdout=out_handle, stderr=err_handle, cwd=str(ROOT))
    except Exception as error:
        for handle in handles:
            handle.close()
        # The marker is read back from the file, never from the caller's object.
        # With a marker a child may have run, so the full reserved bound is
        # charged; the zero path is unavailable and is never retried.
        if ledger.spawned(reservation):
            ledger.settle_unknown(reservation, detail=(
                "the spawn call failed after the marker was written, so whether a child "
                f"ran is unknown: {type(error).__name__}: {error}"))
        else:
            ledger.settle_aborted(reservation, detail=(
                "the spawn marker was never written, so no child was started: "
                f"{type(error).__name__}: {error}"))
        raise RunnerRefusal(f"spawn failed: {type(error).__name__}: {error}") from error

    timed_out = False
    returncode = None
    try:
        returncode = process.wait(timeout=plan.reserved_bound_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Only ever this runner's own child, through the handle it opened. No pid
        # enumeration, no liveness probe, and no other job is ever signalled.
        process.kill()
        try:
            returncode = process.wait(timeout=30)
        except Exception:
            returncode = None
    finally:
        wall = float(clock() - started)
        for handle in handles:
            handle.close()

    counters, settlement = None, "unknown"
    if not timed_out and returncode is not None:
        counters, settlement = _settlement_from_result(plan.out_dir / "result.json")

    if counters is None:
        detail = ("the child exceeded the reserved bound and was killed, so its duration and "
                  "completed work are unknown" if timed_out else
                  f"the child returned {returncode!r} without a result record that declares "
                  "either a completed-training shape or a consistent pre-training refusal")
        ledger.settle_unknown(reservation, detail=detail, returncode=returncode)
        code = EXIT_INTERRUPTED if timed_out else EXIT_REFUSED
    else:
        # Both admissible shapes charge the wall THIS process measured. A refusal
        # that provably preceded training therefore costs what it cost, instead
        # of a ~540 s bound that would exhaust a stage in about 13 mistakes.
        ledger.settle(reservation, actual_wall_s=wall, returncode=returncode, counters=counters)
        code = EXIT_OK if returncode == 0 else int(returncode)

    try:
        write_json_exclusive(plan.out_dir / "child.json", {
            "schema_version": CHILD_SCHEMA, "argv": [str(item) for item in argv],
            "argv_sha256": _argv_digest(argv), "returncode": returncode,
            "timed_out": timed_out, "measured_process_wall_s": wall,
            "settlement_shape": settlement,
            "reserved_bound_s": plan.reserved_bound_s,
            "reservation_id": reservation.reservation_id,
            "settled_counters": None if counters is None else counters.as_dict(),
        })
    except FileExistsError:
        pass
    return code


@dataclass
class WorkerContext:
    """What the build step hands the worker. Injected in tests, so the whole
    worker contract is exercised on a CPU adapter with no simulator."""

    algo: object
    env: object
    control_dt: float
    max_epochs_total: int
    bindings: dict
    evaluate: object = None
    close: object = None


def _build_runtime(args, plan) -> WorkerContext:
    """The real GPU build: backend import, model compile, environment, PPO.

    Everything expensive lives here, and it runs only in the child, after the
    launcher has already claimed the wall time for it.
    """
    import copy
    import yaml
    from msk_warp.algorithms.ppo import PPO
    from msk_warp.analysis.myoleg26_baseline import evaluate_policy, isolated_rng
    from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv

    spec = P.stage(plan.stage)
    arm = P.recipe(plan.recipe)
    cfg = copy.deepcopy(yaml.safe_load(Path(P.V2_CONFIG).read_text(encoding="utf-8")))
    cfg["params"]["general"].update(seed=plan.seed, device=args.device,
                                    logdir=str(plan.out_dir))
    cfg["params"]["env"]["num_actors"] = P.NUM_WORLDS
    # Only gamma and entropy_coef vary between the sealed arms.
    cfg["params"]["config"].update(
        max_epochs=spec.epoch_cap_per_seed, steps_num=P.CONTROLS_PER_WORLD_PER_EPOCH,
        save_interval=0, gamma=arm.gamma, entropy_coef=arm.entropy_coef)
    write_json_exclusive(plan.out_dir / "effective_config.json", cfg)

    algo = PPO(cfg)
    env = algo.env
    if algo.num_envs != P.NUM_WORLDS or algo.steps_num != P.CONTROLS_PER_WORLD_PER_EPOCH:
        raise RunnerRefusal(
            f"the built arm has {algo.num_envs} worlds and {algo.steps_num} controls per "
            f"epoch, but the sealed arm is {P.NUM_WORLDS} and "
            f"{P.CONTROLS_PER_WORLD_PER_EPOCH}")

    # The stage's own selection block. The confirmation and audit blocks stay
    # unused until checkpoint selection is locked, so they are never named here.
    seeds = list(P.selection_reset_block(plan.stage))
    with isolated_rng():
        kwargs = {key: value for key, value in cfg["params"]["env"].items()
                  if key not in ("name", "num_actors")}
        kwargs.update(no_grad=True, stochastic_init=False, device=algo.device,
                      episode_length=P.EVALUATION_HORIZON_CONTROLS)
        evaluation_env = MyoLeg26WalkEnv(num_envs=len(seeds), **kwargs)

    def evaluate(epoch, deadline=None):
        counter = {"kind": "selection", "epoch": int(epoch),
                   "worlds": evaluation_env.num_envs,
                   "attempted_calls": 0, "completed_calls": 0}
        name = f"epoch_{int(epoch):04d}"
        algo.save(name)
        checkpoint = plan.out_dir / f"{name}.pt"
        # Evaluation is RNG-neutral: it runs inside ``isolated_rng`` and is
        # deterministic, so it cannot perturb the training stream.
        with isolated_rng():
            evaluation, _traces = evaluate_policy(
                evaluation_env, algo.actor, algo.obs_rms, seeds,
                horizon=P.EVALUATION_HORIZON_CONTROLS, deterministic=True,
                deadline=deadline, transition_counter=counter)
        record = {"epoch": int(epoch), "kind": "selection", "stage": plan.stage,
                  "recipe": plan.recipe, "seed": plan.seed,
                  "checkpoint": checkpoint.name,
                  "checkpoint_sha256": sha256_file(checkpoint),
                  "reset_block": seeds, "evaluation": evaluation}
        return record, counter

    def close():
        algo.close()

    return WorkerContext(algo=algo, env=env, control_dt=float(env.control_dt),
                         max_epochs_total=spec.epoch_cap_per_seed,
                         bindings=freeze_bindings(args.freeze),
                         evaluate=evaluate, close=close)


def _write_refusal_result(args, error, *, clock, started, wrote) -> None:
    """Record a refusal that provably preceded training, with explicit zeros.

    This is the positive evidence the parent needs to charge the ~1 s a refusal
    really costs instead of the full reserved bound. It is written **only** from
    the pre-training refusal path, so the marker cannot appear on a record whose
    training had begun; and the parent re-checks the record against itself, so a
    marker contradicted by its own counters buys nothing.
    """
    if wrote["training_began"]:
        return
    declared = getattr(args, "out_dir", None)
    if not declared:
        # Path("") is the *current directory*, so a falsy out_dir must never
        # be turned into one: that would deposit a refusal record into the
        # working tree. With no declared directory nothing is recorded
        # anywhere, and the parent charges the full reserved bound.
        return
    out_dir = Path(str(declared))
    path = out_dir / "result.json"
    if not out_dir.is_dir() or path.exists():
        return          # nothing can be recorded here; the parent charges the bound
    try:
        write_json_exclusive(path, {
            "schema_version": RESULT_SCHEMA, "runner_schema": SCHEMA_VERSION,
            "status": REFUSED_BEFORE_TRAINING,
            "training_began": False,
            "refusal": f"{type(error).__name__}: {error}",
            "stop_reason": REFUSED_BEFORE_TRAINING,
            "published": False, "boundary_is_live": False,
            "censored": False, "censor_reason": None,
            "completed_epoch": None, "completed_epochs": 0,
            "accounting": _zero_accounting(),
            "wall_seconds": float(clock() - started),
        })
    except OSError:
        return          # an unwritable record is simply absent; the bound applies


def run_worker(args, *, build=None, clock=time.perf_counter) -> int:
    """Train one bounded segment in the foreground. Never opens the ledger.

    A refusal that provably precedes training leaves an explicit zero-counter
    record so the parent can charge what it measured; once training has begun the
    real record is written instead, and the cheap path is unreachable.
    """
    started = clock()
    wrote = {"training_began": False}
    try:
        return _worker_segment(args, build=build, clock=clock, started=started, wrote=wrote)
    except RunnerRefusal as error:
        _write_refusal_result(args, error, clock=clock, started=started, wrote=wrote)
        raise


def _worker_segment(args, *, build, clock, started, wrote) -> int:
    plan = resolve_plan(args)
    out_dir = plan.out_dir
    if not out_dir.is_dir():
        raise RunnerRefusal(
            f"the output directory does not exist: {out_dir}; the parent creates it "
            "exclusively before the spawn and the worker never creates its own")
    result_path = out_dir / "result.json"
    if result_path.exists():
        raise RunnerRefusal(
            f"a result record already exists and is never overwritten: {result_path}")

    work_deadline = started + plan.work_deadline_s
    capture_reserve_s = float(getattr(args, "capture_reserve_s", DEFAULT_CAPTURE_RESERVE_S))
    spec = P.stage(plan.stage)
    context = (build or _build_runtime)(args, plan)

    recorder = EpisodeEventRecorder(context.env.num_envs, context.control_dt,
                                    device=getattr(context.env, "device", "cpu"))
    replayed = {"control_transitions": int(getattr(args, "replayed_control_transitions", 0)),
                "physics_steps": int(getattr(args, "replayed_physics_steps", 0))}
    parent_sha256 = None
    resume_report = None

    if plan.parent_segment is not None:
        parent_result = read_json(plan.parent_result)
        lineage = parent_result.get("lineage") or {}
        if not lineage.get("segment_sha256"):
            raise RunnerRefusal(
                f"the parent result {plan.parent_result} published no boundary, so there is "
                "nothing to resume from; re-run the previous segment instead")
        if lineage.get("valid_boundary_epoch") != plan.start_epoch:
            raise RunnerRefusal(
                f"the parent published its boundary at epoch "
                f"{lineage.get('valid_boundary_epoch')!r}, but this segment's sealed round "
                f"starts at epoch {plan.start_epoch}")
        if parent_result.get("segment_index") != plan.segment_index - 1:
            raise RunnerRefusal(
                f"the parent result is segment {parent_result.get('segment_index')!r}, not "
                f"the immediate predecessor of segment {plan.segment_index}")
        resume_report, state = resume_boundary(
            plan.parent_segment, context.algo, context.env,
            epoch=plan.start_epoch, seed=plan.seed,
            segment_index=parent_result["segment_index"],
            parent_segment_sha256=lineage.get("parent_segment_sha256"),
            bindings=context.bindings,
            expected_segment_sha256=lineage["segment_sha256"])
        parent_sha256 = lineage["segment_sha256"]
        recorder.load(state.extra["episode_accumulators"])
    else:
        context.algo._current_obs = context.env.reset()

    evaluation_paths = []
    evaluation_totals = {"attempted": 0, "completed": 0}

    def run_evaluation(epoch):
        record, counter = context.evaluate(int(epoch), deadline=work_deadline)
        worlds = int(counter.get("worlds", 0))
        evaluation_totals["attempted"] += int(counter.get("attempted_calls", 0)) * worlds
        evaluation_totals["completed"] += int(counter.get("completed_calls", 0)) * worlds
        path = out_dir / f"selection_{int(epoch):04d}.json"
        write_json_exclusive(path, record)
        evaluation_paths.append(str(path))
        return record

    published = None
    unexpected = None
    # From here on, work has been done: the refusal record is no longer reachable.
    wrote["training_began"] = True
    try:
        outcome = train_segment(
            context.algo, context.env, start_epoch=plan.start_epoch,
            end_epoch=plan.end_epoch, max_epochs_total=context.max_epochs_total,
            clock=clock, work_deadline=work_deadline, recorder=recorder,
            capture_reserve_s=capture_reserve_s,
            evaluate=run_evaluation if context.evaluate is not None else None,
            evaluation_epochs=P.evaluation_epochs(spec.epoch_cap_per_seed),
            diagnostics=True)
        # The single capture site, reached only at a live completed boundary.
        if outcome.boundary_is_live and outcome.completed_epochs > 0:
            extra = segment_extra(
                stage=plan.stage, recipe=plan.recipe, seed=plan.seed,
                segment_index=plan.segment_index, parent_segment_sha256=parent_sha256,
                bindings=context.bindings, recorder_state=recorder.state(),
                replayed=replayed)
            published = publish_boundary(
                context.algo, context.env, out_dir / "segment.ptc",
                epoch=outcome.completed_epoch, extra=extra,
                label=f"{plan.stage}-{plan.recipe}-s{plan.seed}-g{plan.segment_index}")
    except Exception as error:      # recorded for diagnosis, never retried
        unexpected = f"{type(error).__name__}: {error}"
        outcome = locals().get("outcome") or SegmentOutcome(
            completed_epoch=plan.start_epoch, stop_reason="unexpected_error")
        outcome.boundary_is_live = False
        outcome.censored = True
        outcome.censor_reason = unexpected
        outcome.stop_reason = "unexpected_error"
    finally:
        if context.close is not None:
            context.close()

    worlds = int(context.env.num_envs)
    substeps = int(P.PHYSICS_SUBSTEPS)
    training_attempted = outcome.counters.attempted_control_calls * worlds
    training_boundary = (outcome.counters.boundary_control_calls * worlds
                         if published is not None else 0)
    completed = training_boundary + evaluation_totals["completed"]
    attempted = training_attempted + evaluation_totals["attempted"]
    discarded = attempted - completed
    accounting = {
        "worlds": worlds,
        "substeps": substeps,
        "training_control_transitions": training_attempted,
        "training_boundary_control_transitions": training_boundary,
        "evaluation_control_transitions": evaluation_totals["completed"],
        "attempted_control_transitions": attempted,
        "completed_control_transitions": completed,
        "discarded_control_transitions": discarded,
        "replayed_control_transitions": replayed["control_transitions"],
        "attempted_physics_steps": attempted * substeps,
        "completed_physics_steps": completed * substeps,
        "discarded_physics_steps": discarded * substeps,
        "replayed_physics_steps": replayed["physics_steps"],
        "partial_update_cost_s": float(outcome.counters.partial_update_cost_s),
        "attempted_control_calls": outcome.counters.attempted_control_calls,
        "returned_control_calls": outcome.counters.returned_control_calls,
        "exact_completed_calls": (outcome.counters.attempted_control_calls
                                  == outcome.counters.returned_control_calls),
    }

    valid_boundary = (published["path"] if published is not None
                      else (str(plan.parent_segment) if plan.parent_segment else None))
    result = {
        "schema_version": RESULT_SCHEMA, "runner_schema": SCHEMA_VERSION,
        "stage": plan.stage, "recipe": plan.recipe, "seed": plan.seed,
        "segment_index": plan.segment_index,
        "training_began": True,
        "start_epoch": plan.start_epoch, "end_epoch": plan.end_epoch,
        "epochs_requested": plan.epochs,
        "completed_epoch": outcome.completed_epoch,
        "completed_epochs": outcome.completed_epochs,
        "boundary_is_live": outcome.boundary_is_live,
        "published": published is not None,
        "stop_reason": outcome.stop_reason,
        "censored": outcome.censored,
        "censor_reason": outcome.censor_reason,
        "partial_detail": outcome.partial_detail,
        "unexpected_error": unexpected,
        "lineage": {
            "segment_sha256": None if published is None else published["sha256"],
            "segment_path": None if published is None else published["path"],
            "parent_segment_sha256": parent_sha256,
            "valid_boundary_path": valid_boundary,
            "valid_boundary_epoch": (outcome.completed_epoch if published is not None
                                     else plan.start_epoch),
        },
        "resume": resume_report,
        "accounting": accounting,
        "losses": outcome.losses,
        "epoch_events": outcome.epoch_events,
        "evaluations": evaluation_paths,
        "work_deadline_s": plan.work_deadline_s,
        "capture_reserve_s": capture_reserve_s,
        "wall_seconds": float(clock() - started),
    }
    write_json_exclusive(result_path, result)

    if outcome.censored:
        return EXIT_CENSORED
    if not outcome.boundary_is_live or published is None:
        return EXIT_INTERRUPTED
    return EXIT_OK


def run_select(args) -> int:
    """Select the best checkpoint of ONE run/arm and write the record exclusively."""
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "selection.json"
    if out.exists():
        print(f"REFUSED: a selection record already exists and is never overwritten: {out}",
              file=sys.stderr)
        return EXIT_REFUSED
    candidates, refused = collect_run_records(run_dir)
    try:
        record = select_run(candidates, refused=refused)
    except RunnerFault as error:
        if error.record is not None:
            write_json_exclusive(out, error.record)
        raise
    write_json_exclusive(out, record)
    print(f"selection: {record['status']} "
          f"(compared {record['candidates_compared']} of {record['candidates_considered']}, "
          f"{record['wall_cap_truncations']} wall-cap truncations, "
          f"{len(record['refused_segments'])} refused segments)")
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_identity_arguments(parser) -> None:
    parser.add_argument("--stage", required=True, choices=list(P.STAGE_ORDER))
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--segment-index", required=True, type=int)
    parser.add_argument("--parent-segment", default=None)
    parser.add_argument("--parent-result", default=None)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--work-deadline-s", default=float(P.SEGMENT_WORK_DEADLINE_S), type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-reserve-s", default=DEFAULT_CAPTURE_RESERVE_S, type=float)
    # Work a previous attempt did and discarded, which this segment redoes. It is
    # real cost and is recorded, never netted off. Supplied explicitly by the
    # campaign driver from the previous attempt's ``discarded_*`` counters,
    # because the interrupted attempt published nothing and is therefore not
    # this segment's parent boundary.
    parser.add_argument("--replayed-control-transitions", default=0, type=int)
    parser.add_argument("--replayed-physics-steps", default=0, type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_myoleg26_ppo_v2",
        description="Segmented MyoLeg26 v2 PPO campaign runner (launch/worker/select).")
    modes = parser.add_subparsers(dest="mode", required=True)

    launch = modes.add_parser("launch", help="reserve budget and spawn one segment child")
    _add_identity_arguments(launch)
    launch.add_argument("--ledger", required=True)
    launch.add_argument("--run-dir", required=True)
    launch.add_argument("--shutdown-allowance-s", default=DEFAULT_SHUTDOWN_ALLOWANCE_S, type=float)
    launch.add_argument("--reserved-bound-s", default=None, type=float)
    launch.add_argument("--create-ledger", action="store_true")
    launch.add_argument("--no-contention-probe", action="store_true")

    worker = modes.add_parser("worker", help="train one segment (spawned by launch)")
    _add_identity_arguments(worker)
    worker.add_argument("--out-dir", required=True)

    select = modes.add_parser("select", help="select the best checkpoint of ONE run/arm")
    select.add_argument("--run-dir", required=True)
    select.add_argument("--out", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "launch":
            return run_launch(args)
        if args.mode == "worker":
            return run_worker(args)
        return run_select(args)
    except RunnerRefusal as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except RunnerFault as error:
        print(f"RUNNER FAULT: {error}", file=sys.stderr)
        return EXIT_RUNNER_FAULT
    except CensorRun as error:
        print(f"CENSORED: {error}", file=sys.stderr)
        return EXIT_CENSORED


if __name__ == "__main__":
    raise SystemExit(main())
