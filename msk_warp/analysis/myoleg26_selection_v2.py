"""Revised ("v2") checkpoint-selection key for the official MyoLeg26 walking task.

This is a **revised protocol**, not a correction of the recorded v1 result. The v1
key in ``msk_warp/analysis/myoleg26_baseline.py`` (``behavior_rank``) is untouched
and its recorded selection for the frozen v1 batch is not revisited, recomputed or
relabelled here. The two keys order some candidate pairs differently; that is the
point of a revision, not evidence that either implementation was buggy.

The key, maximised lexicographically::

    (survival_fraction,
     mean_first_episode_duration_s,
     mean_in_band_seconds,
     -mean_episode_velocity_2d_rmse_mps)

Motivation for the second and third components: with every survival fraction at
zero, a survival-then-RMSE key can prefer a *shorter* fall, because a policy that
collapses immediately accumulates less tracking error (``CL-17``). Duration is
therefore consulted before error, and time actually spent inside the speed band is
consulted before error as well, so that standing still is not rewarded over moving.

**This ranking is not claimed to be unexploitable.** It is a preregistered key with
a stated order, nothing more. Behaviour still has to be assessed directly.

**Level note, recorded so the two orders are not mistaken for a contradiction.** This
key ranks **checkpoints within one run** and puts ``survival_fraction`` first. The
campaign's **recipe-level** rank puts duration first. The two operate at different
levels on different objects and the difference is deliberate, not a bug: neither is
derived from the other, and cross-run comparison belongs at the recipe level on plain
floats. Relatedly, no evaluation carries a ``mean_in_band_seconds`` field at all — it
does not exist in ``summary`` and is derived here by :func:`mean_in_band_seconds`
from per-episode counts.

Tie rule, explicit and documented rather than incidental: a strictly greater key
wins; on an **exact** four-component tie the **smaller epoch** wins, keyed on each
candidate's own ``epoch`` field. Selection never depends on iteration order, dict
insertion order, ``max``/``sorted`` stability, or the order in which results arrive
— two candidates carrying the same epoch are refused outright rather than separated
by arrival order.

Rejection discipline, in **two deliberately distinct classes**:

* :class:`InvalidEvaluationError` — the **data** of one checkpoint's evaluation
  cannot be ranked (malformed, incomplete, nonfinite or inconsistent). Nothing is
  coerced, defaulted to zero, replaced by NaN or silently skipped, so a broken
  evaluation can never win or lose a comparison. The future campaign runner
  catches this and records an explicit invalid/incomplete status **for that
  checkpoint**; a rejected checkpoint does not participate in the comparison.
* :class:`InvalidCandidateSetError` — the **caller** handed over a malformed
  candidate set (duplicate or missing ``epoch``, missing ``evaluation``, a
  non-mapping candidate, an empty set). This says nothing about any checkpoint's
  data and **must never be recorded as a checkpoint status**: doing so would file
  a runner bug as an "invalid checkpoint" and quietly lose a real checkpoint from
  the campaign record. It is a caller fault to fix, not a datum to record.

Neither class is catchable as the other; both subclass ``ValueError``, so existing
broad handling still works.

Note on scope, which the runner must respect: epochs are unique *within one run*,
but every seed and every recipe arm has its own epoch 0, so **candidates must be
scoped to a single run/arm**. Pooling across seeds or arms raises
:class:`InvalidCandidateSetError` by design; cross-run comparison is a different
operation that belongs at the recipe level, on plain floats.

Deliberately not re-exported from ``msk_warp.analysis.__init__``, which imports
numpy and mujoco only. ``myoleg26_baseline`` sets that precedent.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import numbers

SELECTION_KEY_VERSION = "myoleg26-selection-v2"

#: The four maximised components, in the order they are compared.
RANK_COMPONENTS = (
    "survival_fraction",
    "mean_first_episode_duration_s",
    "mean_in_band_seconds",
    "negative_mean_episode_velocity_2d_rmse_mps",
)

REQUIRED_EVALUATION_KEYS = ("summary", "episodes", "control_dt", "complete")
REQUIRED_SUMMARY_KEYS = (
    "survival_fraction",
    "mean_first_episode_duration_s",
    "mean_episode_velocity_2d_rmse_mps",
)
REQUIRED_EPISODE_KEYS = ("length", "speed_band_steps")


class InvalidEvaluationError(ValueError):
    """An evaluation that must not be ranked, compared, defaulted or scored.

    ``reason`` is a specific description of the first violation found. It is a
    ``ValueError`` subclass so that existing broad handling still catches it.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class InvalidCandidateSetError(ValueError):
    """A caller/set-level fault in the candidate collection handed to selection.

    Raised for a malformed candidate *set* — a non-mapping candidate, a missing
    ``epoch`` or ``evaluation``, a non-integer or negative ``epoch``, a duplicate
    ``epoch``, or an empty set. These are statements about the **caller**, never
    about any checkpoint's evaluation data, so they are deliberately **not**
    catchable as :class:`InvalidEvaluationError` and must never be recorded as an
    invalid or incomplete checkpoint status.

    Also a ``ValueError`` subclass, so an existing broad handler still catches it.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _describe(value) -> str:
    return f"{type(value).__name__} {value!r}"


def _require_mapping(value, label: str, error=InvalidEvaluationError) -> Mapping:
    if not isinstance(value, Mapping):
        raise error(f"{label} is not a mapping: {_describe(value)}")
    return value


def _finite_real(container: Mapping, key: str, label: str) -> float:
    if key not in container:
        raise InvalidEvaluationError(f"{label} is missing required key {key!r}")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidEvaluationError(
            f"{label} {key!r} is not a real number: {_describe(value)}")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidEvaluationError(f"{label} {key!r} is not finite: {_describe(value)}")
    return number


def _exact_count(container: Mapping, key: str, label: str) -> int:
    """An integral control-step count. ``bool`` and integral-valued floats are refused."""
    if key not in container:
        raise InvalidEvaluationError(f"{label} is missing required key {key!r}")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidEvaluationError(
            f"{label} {key!r} is not an integer: {_describe(value)}")
    return int(value)


def _validated(evaluation):
    """Return ``(metrics, in_band_control_steps, episode_count, control_dt)``.

    Raises :class:`InvalidEvaluationError` on the first violation. Every public
    entry point in this module runs this same complete check, so all of them fail
    closed identically on the same input.
    """
    _require_mapping(evaluation, "evaluation")
    for key in REQUIRED_EVALUATION_KEYS:
        if key not in evaluation:
            raise InvalidEvaluationError(f"evaluation is missing required key {key!r}")

    complete = evaluation["complete"]
    if not isinstance(complete, bool):
        raise InvalidEvaluationError(
            f"evaluation 'complete' flag is not a boolean: {_describe(complete)}")
    if not complete:
        raise InvalidEvaluationError(
            "evaluation is incomplete: 'complete' is False, so a wall cap truncated it")

    control_dt = _finite_real(evaluation, "control_dt", "evaluation")
    if control_dt <= 0.0:
        raise InvalidEvaluationError(
            f"evaluation 'control_dt' is not positive: {control_dt!r}")

    episodes = evaluation["episodes"]
    if not isinstance(episodes, (list, tuple)):
        raise InvalidEvaluationError(
            f"evaluation 'episodes' is not a list or tuple: {_describe(episodes)}")
    if not episodes:
        raise InvalidEvaluationError("evaluation 'episodes' is empty: nothing was scored")

    summary = _require_mapping(evaluation["summary"], "evaluation 'summary'")
    metrics = {key: _finite_real(summary, key, "summary metric")
               for key in REQUIRED_SUMMARY_KEYS}

    in_band_steps = 0
    for index, row in enumerate(episodes):
        label = f"episode {index}"
        _require_mapping(row, label)
        length = _exact_count(row, "length", label)
        if length <= 0:
            raise InvalidEvaluationError(f"{label} 'length' is not positive: {length!r}")
        steps = _exact_count(row, "speed_band_steps", label)
        if steps < 0:
            raise InvalidEvaluationError(
                f"{label} 'speed_band_steps' is negative: {steps!r}")
        if steps > length:
            raise InvalidEvaluationError(
                f"{label} 'speed_band_steps' {steps} exceeds 'length' {length}")
        in_band_steps += steps

    return metrics, in_band_steps, len(episodes), control_dt


def mean_in_band_seconds(evaluation) -> float:
    """Mean per-episode time inside the speed band, in seconds.

    Derived from actual per-episode **counts**::

        mean_over_episodes(row["speed_band_steps"]) * evaluation["control_dt"]

    There is no ``mean_in_band_seconds`` field in the evaluation to read.

    The mean of the per-episode ``forward_speed_band_fraction`` values multiplied by
    ``mean_first_episode_duration_s`` is a **different number** whenever episode
    lengths differ, and is not used. (The *pooled* ``summary`` fraction times the
    mean duration happens to be algebraically identical to the count-based value,
    since both reduce to ``sum(steps) / n_episodes * control_dt``; the per-episode
    fraction mean does not, because it reweights short episodes upward.)

    Validates the complete selection contract, so it refuses exactly what
    :func:`behavior_rank_v2` refuses.
    """
    _, in_band_steps, episode_count, control_dt = _validated(evaluation)
    return float(in_band_steps / episode_count * control_dt)


def behavior_rank_v2(evaluation) -> tuple:
    """The revised maximised selection key, as a tuple of exactly four floats.

    Returns only the valid ranking tuple: never a union type, never an
    "ineligible" record, never a sentinel, never a partially populated tuple, and
    never an ``eligible`` field. An input that cannot be ranked raises
    :class:`InvalidEvaluationError` instead.

    The first, second and fourth components keep their existing official-task
    definitions and are taken from ``evaluation["summary"]``; the third is derived
    by :func:`mean_in_band_seconds`.
    """
    metrics, in_band_steps, episode_count, control_dt = _validated(evaluation)
    return (
        float(metrics["survival_fraction"]),
        float(metrics["mean_first_episode_duration_s"]),
        float(in_band_steps / episode_count * control_dt),
        float(-metrics["mean_episode_velocity_2d_rmse_mps"]),
    )


def select_best_checkpoint(candidates):
    """Return the winning candidate mapping from ``candidates``, unchanged.

    Each candidate is a mapping with an ``epoch`` (a non-negative integer, the
    checkpoint's own identity) and an ``evaluation``. Any other keys it carries,
    such as a checkpoint path, are preserved in the returned object.

    Comparison: a strictly greater :func:`behavior_rank_v2` key wins. On an
    **exact** four-component tie the **smaller epoch** wins. The rule is keyed on
    the candidate's own ``epoch``, so the result is independent of the order in
    which candidates are supplied; two candidates sharing an epoch are refused
    rather than separated by arrival order.

    Raises :class:`InvalidCandidateSetError` for a caller-side fault in the set
    itself — an empty set, a malformed candidate, a missing or non-integer
    ``epoch``, a missing ``evaluation``, or a duplicate epoch — and propagates
    :class:`InvalidEvaluationError` unchanged when one candidate's evaluation data
    is unrankable. One invalid candidate aborts the selection either way: it is
    never scored as zero and never silently dropped.
    """
    if isinstance(candidates, (str, bytes)) or isinstance(candidates, Mapping):
        raise InvalidCandidateSetError(
            f"candidates is not a sequence of candidate mappings: {_describe(candidates)}")

    best = None
    best_rank = None
    best_epoch = None
    seen_epochs = set()

    for position, candidate in enumerate(candidates):
        label = f"candidate {position}"
        _require_mapping(candidate, label, InvalidCandidateSetError)
        if "epoch" not in candidate:
            raise InvalidCandidateSetError(f"{label} is missing required key 'epoch'")
        epoch = candidate["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, numbers.Integral):
            raise InvalidCandidateSetError(
                f"{label} 'epoch' is not an integer: {_describe(epoch)}")
        epoch = int(epoch)
        if epoch < 0:
            raise InvalidCandidateSetError(f"{label} 'epoch' is negative: {epoch}")
        if epoch in seen_epochs:
            raise InvalidCandidateSetError(
                f"duplicate candidate 'epoch' {epoch}: arrival order must not decide a tie; "
                f"candidates must be scoped to a single run/arm, since every seed has an epoch 0")
        seen_epochs.add(epoch)
        if "evaluation" not in candidate:
            raise InvalidCandidateSetError(
                f"{label} is missing required key 'evaluation'")

        rank = behavior_rank_v2(candidate["evaluation"])
        if best is None or rank > best_rank or (rank == best_rank and epoch < best_epoch):
            best, best_rank, best_epoch = candidate, rank, epoch

    if best is None:
        raise InvalidCandidateSetError("no candidate checkpoints were supplied")
    return best
