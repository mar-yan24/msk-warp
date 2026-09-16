"""Read-only PPO observability helpers: stateless, RNG-neutral, no training change.

Every function here is a pure observer of state the caller already holds. Nothing
in this module mutates a parameter, a ``.grad``, an optimizer state, a normalizer,
a module's training mode, a rollout buffer, an environment or any random stream;
nothing samples; nothing steps; nothing writes a checkpoint. The module holds no
state of its own, so the future campaign runner owns every ongoing accumulator.

KL and clip fraction are **diagnostics**. They are not evidence of a correctness
failure and this module adds **no** hard-stop gate: a genuinely nonfinite policy or
state stays the runner's existing hard censor. The guarantee is narrower than "no
exceptions", and is stated precisely: for tensor arguments of the documented shapes,
and for whatever numeric content they carry, an empty, degenerate or nonfinite value
returns an explicit JSON-safe ``status`` with ``None`` metrics — never a fabricated
zero and never an abort of the training loop. (That is the deliberate difference from
``myoleg26_selection_v2``, which *must* raise: a selection key that returned a default
could let a broken evaluation win a comparison.)

**Input-shape preconditions that do raise**, stated rather than papered over. The
three tensor-taking helpers are not type guarded, so a caller violating their input
contract gets an ordinary Python exception instead of a status:

* a non-tensor ``buf_*`` argument raises ``AttributeError`` (it has no ``.reshape``);
* a **zero-width** observation or action tensor — shape ``(N, 0)`` — raises
  ``RuntimeError`` from ``reshape``;
* ``get_logstd()`` returning something that is neither a tensor nor ``None`` raises
  ``AttributeError``; a *missing* ``get_logstd`` is guarded and reported as
  ``logstd_unavailable``.

None of these is reachable from the real PPO seam: the ``buf_*`` objects are always
tensors and the official model has ``obs_dim = 145`` and ``na = 26``. By contrast
``episode_event_summary``, which takes plain data rather than tensors, **is** fully
type guarded and reports ``malformed_event`` for any shape of bad input.

Verified PPO seams this module reads (``msk_warp/algorithms/ppo.py``):

* ``buf_obs`` is **already normalized** by the frozen ``obs_rms_snapshot``.
* ``buf_actions`` holds the **pre-tanh** sample; the executed action is
  ``tanh(buf_actions)``.
* ``buf_log_probs`` is the summed Gaussian log-probability in pre-tanh space, with
  **no tanh Jacobian correction anywhere in production**. This module adds none.
* Shapes are ``(steps_num, num_envs, ...)``; the clip bound is ``algo.clip_range``.

Deliberately **not** read here, because none of them describes the current epoch:
``PPO.episode_length_his`` / ``episode_loss_his`` are *unbounded run-lifetime* lists
appended at every ``done``; ``episode_length_meter`` / ``episode_loss_meter`` are a
*capped* 100-sample rolling mean (``AverageMeter(1, 100)``); and ``episode_loss``
accumulates **negated** reward. Per-epoch episode statistics therefore come from the
explicit completed-episode event list defined by :data:`EPISODE_EVENT_FIELDS`,
supplied by a future runner wrapper.

Deliberately not re-exported from ``msk_warp.analysis.__init__``, which imports
numpy and mujoco only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numbers

import torch

STATUS_OK = "ok"

#: Reported approximate-KL estimators, stated so a log reader cannot mistake which.
KL_ESTIMATORS = {
    "approx_kl_k1": "mean(old_log_prob - new_log_prob)",
    "approx_kl_k3": "mean(exp(d) - 1 - d) with d = new_log_prob - old_log_prob",
    "clip_fraction": "mean(abs(exp(d) - 1) > clip_range)",
}

KL_PHASE = ("post-update actor re-evaluated under no_grad on this epoch's stored "
            "behaviour-policy rollout samples; no resampling, no environment step")

EXCITATION_DEFINITION = ("excitation = 0.5 * (tanh(pre_tanh) + 1) in [0, 1], from the "
                         "stored pre-tanh rollout actions")
EXCITATION_PHASE = ("this epoch's behaviour-policy rollout samples (buf_actions), not "
                    "the environment's post-reset action buffer")
#: ``abs(pre_tanh) > 2`` is tanh saturation. It is not numeric limiting of the action.
SATURATION_DEFINITION = "abs(pre_tanh) > 2"
SATURATION_NOTE = ("tanh saturation of the pre-tanh sample; the executed action is not "
                   "numerically limited at the bound")
#: The two sub-fractions overlap once ``tol`` approaches 0.5, so they are not additive.
SUB_FRACTION_NOTE = ("near_upper_fraction and near_lower_fraction overlap when tol "
                     "approaches 0.5 and must not be summed; near_bound_fraction is the "
                     "union of the two masks and is the only additive-safe figure")

#: Plain-data interface for one completed episode, supplied by a future runner.
#:
#: ``world``           non-negative integer world index
#: ``length_controls`` positive integer count of control transitions
#: ``duration_s``      positive finite float, ``length_controls * control_dt``
#: ``return``          finite float, the episode's **reward** sum (not negated)
#: ``end_reason``      string, e.g. ``horizon`` | ``timeout`` | ``task_failure``
#: ``failure_flags``   mapping of flag name to bool
EPISODE_EVENT_FIELDS = ("world", "length_controls", "duration_s", "return",
                        "end_reason", "failure_flags")

#: Official-task failure flags (``myoleg26_task.failures`` plus the environment's
#: contact flag). Other keys are counted as supplied but reported separately: the
#: legacy ``height`` flag of the non-official branch is **not** an official flag.
OFFICIAL_FAILURE_FLAG_NAMES = ("nonfinite", "low_pelvis", "low_upright",
                               "nonfoot_ground_contact")


def _report(defaults: dict, status: str, **overrides) -> dict:
    payload = dict(defaults)
    payload["status"] = status
    payload.update(overrides)
    return payload


_KL_DEFAULTS = {
    "status": STATUS_OK,
    "samples": None,
    "approx_kl_k1": None,
    "approx_kl_k3": None,
    "clip_fraction": None,
    "clip_range": None,
    "mean_ratio": None,
    "mean_entropy": None,
    "mean_new_log_prob": None,
    "mean_old_log_prob": None,
    "nonfinite_ratio_samples": None,
    "estimators": KL_ESTIMATORS,
    "phase": KL_PHASE,
    "detail": None,
}

_LOGSTD_DEFAULTS = {
    "status": STATUS_OK,
    "dimensions": None,
    "logstd_min": None,
    "logstd_mean": None,
    "logstd_max": None,
    "std_min": None,
    "std_mean": None,
    "std_max": None,
    "detail": None,
}

_EXCITATION_DEFAULTS = {
    "status": STATUS_OK,
    "entries": None,
    "tol": None,
    "near_bound_fraction": None,
    "near_upper_fraction": None,
    "near_lower_fraction": None,
    "mean_excitation": None,
    "tanh_saturation_fraction": None,
    "definition": EXCITATION_DEFINITION,
    "sub_fraction_note": SUB_FRACTION_NOTE,
    "phase": EXCITATION_PHASE,
    "saturation_definition": SATURATION_DEFINITION,
    "saturation_note": SATURATION_NOTE,
    "detail": None,
}

_EVENT_DEFAULTS = {
    "status": STATUS_OK,
    "completed_episodes": None,
    "worlds": None,
    "mean_duration_s": None,
    "min_duration_s": None,
    "max_duration_s": None,
    "mean_return": None,
    "min_return": None,
    "max_return": None,
    "mean_length_controls": None,
    "end_reason_counts": {},
    "failure_flag_counts": {},
    "official_failure_flag_names": [],
    "unofficial_failure_flag_names": [],
    "detail": None,
}


def _is_real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_count(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def approx_kl_clip(actor, buf_obs, buf_actions, buf_log_probs, *, clip_range) -> dict:
    """End-of-update approximate KL, probability ratio and clip fraction.

    Flattens the stored rollout buffers, re-evaluates them with the post-update
    actor under :func:`torch.no_grad`, and compares against the stored behaviour
    log-probabilities. Both reported estimators are named in ``estimators``:
    ``k1 = mean(old - new)`` and ``k3 = mean(exp(d) - 1 - d)`` with
    ``d = new - old``; ``clip_fraction = mean(abs(exp(d) - 1) > clip_range)``, so a
    sample exactly at the bound is **not** counted as clipped.

    ``buf_obs`` must already be normalized (PPO stores it that way) and
    ``buf_actions`` must be the pre-tanh sample. The actor's own ``evaluate_actions``
    is used, so the pre-tanh convention and the absence of a tanh Jacobian term match
    production exactly; no log-probability is recomputed here. The actor's training
    mode is read and left exactly as it is, never toggled.

    ``ActorDeterministicMLP`` has no ``evaluate_actions``: that is reported as the
    explicit status ``actor_has_no_evaluate_actions``, not as zeros.
    """
    if not (_is_real(clip_range) and math.isfinite(float(clip_range))
            and float(clip_range) >= 0.0):
        return _report(_KL_DEFAULTS, "invalid_clip_range",
                       detail=f"clip_range is not a non-negative finite real: {clip_range!r}")
    clip = float(clip_range)

    if not callable(getattr(actor, "evaluate_actions", None)):
        return _report(_KL_DEFAULTS, "actor_has_no_evaluate_actions", clip_range=clip,
                       detail=f"{type(actor).__name__} exposes no evaluate_actions")

    obs = buf_obs.reshape(-1, buf_obs.shape[-1])
    actions = buf_actions.reshape(-1, buf_actions.shape[-1])
    old_log_prob = buf_log_probs.reshape(-1)
    if not obs.shape[0] == actions.shape[0] == old_log_prob.shape[0]:
        return _report(
            _KL_DEFAULTS, "shape_mismatch", clip_range=clip,
            detail=(f"flattened sample counts differ: obs {obs.shape[0]}, "
                    f"actions {actions.shape[0]}, log_probs {old_log_prob.shape[0]}"))

    samples = int(old_log_prob.shape[0])
    if samples == 0:
        return _report(_KL_DEFAULTS, "no_samples", samples=0, clip_range=clip,
                       detail="the flattened rollout holds no samples")
    if not (bool(torch.isfinite(obs).all()) and bool(torch.isfinite(actions).all())
            and bool(torch.isfinite(old_log_prob).all())):
        return _report(_KL_DEFAULTS, "nonfinite_input", samples=samples, clip_range=clip,
                       detail="stored obs, actions or log-probabilities are not all finite")

    with torch.no_grad():
        new_log_prob, entropy = actor.evaluate_actions(obs, actions)
        new_log_prob = new_log_prob.reshape(-1)
        entropy = entropy.reshape(-1)

        status = STATUS_OK
        detail = None
        values = {}

        new_finite = bool(torch.isfinite(new_log_prob).all())
        if new_finite:
            values["approx_kl_k1"] = float((old_log_prob - new_log_prob).mean())
            values["mean_new_log_prob"] = float(new_log_prob.mean())
            values["mean_old_log_prob"] = float(old_log_prob.mean())
        else:
            status = "nonfinite_derived"
            detail = "the actor returned a nonfinite log-probability"

        if bool(torch.isfinite(entropy).all()):
            values["mean_entropy"] = float(entropy.mean())
        elif status == STATUS_OK:
            status = "nonfinite_derived"
            detail = "the actor returned a nonfinite entropy"

        delta = new_log_prob - old_log_prob
        ratio = torch.exp(delta)
        nonfinite_ratios = int((~torch.isfinite(ratio)).sum())
        if new_finite and nonfinite_ratios == 0:
            values["mean_ratio"] = float(ratio.mean())
            values["clip_fraction"] = float(
                ((ratio - 1.0).abs() > clip).to(ratio.dtype).mean())
            k3_terms = ratio - 1.0 - delta
            if bool(torch.isfinite(k3_terms).all()):
                values["approx_kl_k3"] = float(k3_terms.mean())
            elif status == STATUS_OK:
                status = "nonfinite_derived"
                detail = "the k3 estimator overflowed"
        elif status == STATUS_OK:
            status = "nonfinite_derived"
            detail = (f"{nonfinite_ratios} of {samples} probability ratios are not "
                      f"finite (exponential overflow)")

    return _report(_KL_DEFAULTS, status, samples=samples, clip_range=clip,
                   nonfinite_ratio_samples=nonfinite_ratios, detail=detail, **values)


def logstd_summary(actor) -> dict:
    """Per-dimension ``logstd`` bounds and the standard deviations they imply.

    ``ActorDeterministicMLP.get_logstd()`` returns ``None``; that is reported as the
    explicit status ``logstd_unavailable``, never as zeros. The parameter is read
    detached under ``no_grad`` and never modified.
    """
    getter = getattr(actor, "get_logstd", None)
    logstd = getter() if callable(getter) else None
    if logstd is None:
        return _report(_LOGSTD_DEFAULTS, "logstd_unavailable",
                       detail=f"{type(actor).__name__} has no logstd parameter")

    with torch.no_grad():
        values = logstd.detach().reshape(-1)
        dimensions = int(values.numel())
        if dimensions == 0:
            return _report(_LOGSTD_DEFAULTS, "no_samples", dimensions=0,
                           detail="the logstd parameter is empty")
        if not bool(torch.isfinite(values).all()):
            return _report(_LOGSTD_DEFAULTS, "nonfinite_logstd", dimensions=dimensions,
                           detail="the logstd parameter is not all finite")
        std = values.exp()
        summary = {
            "logstd_min": float(values.min()),
            "logstd_mean": float(values.mean()),
            "logstd_max": float(values.max()),
            "std_min": float(std.min()),
            "std_mean": float(std.mean()),
            "std_max": float(std.max()),
        }
    return _report(_LOGSTD_DEFAULTS, STATUS_OK, dimensions=dimensions, **summary)


def near_bound_excitation(buf_actions, *, tol=0.02) -> dict:
    """Fraction of executed muscle excitations sitting within ``tol`` of 0 or 1.

    The executed action is ``tanh(buf_actions)`` and the excitation is
    ``0.5 * (tanh(pre_tanh) + 1)``, so this reads the stored pre-tanh rollout
    actions. It deliberately does **not** read ``env.actions`` after a step: reset
    rows there are overwritten with the passive value ``-1``, which makes it a
    post-reset buffer rather than the executed action.

    The threshold is **inclusive**: an excitation exactly ``tol`` from a bound counts
    as near-bound. ``tol`` must lie in ``[0, 0.5]``.

    ``near_bound_fraction`` is the **union** of the two masks, so it never double
    counts. ``near_upper_fraction`` and ``near_lower_fraction`` are reported as they
    are and are **not** additive: as ``tol`` approaches 0.5 the two bands **overlap**
    (at ``tol = 0.5`` an excitation of exactly 0.5 is within tolerance of both
    bounds), so their sum can exceed 1. They must not be summed, and they are
    deliberately not renormalized — renormalizing would hide the overlap instead of
    reporting it.

    ``tanh_saturation_fraction`` reports ``abs(pre_tanh) > 2`` separately. That is
    tanh saturation of the sample, a different quantity from the excitation band, and
    it is not numeric limiting of the action.
    """
    if not (_is_real(tol) and math.isfinite(float(tol)) and 0.0 <= float(tol) <= 0.5):
        return _report(_EXCITATION_DEFAULTS, "invalid_tol",
                       detail=f"tol is not a real number in [0, 0.5]: {tol!r}")
    tolerance = float(tol)

    pre_tanh = buf_actions.reshape(-1)
    entries = int(pre_tanh.numel())
    if entries == 0:
        return _report(_EXCITATION_DEFAULTS, "no_samples", entries=0, tol=tolerance,
                       detail="the flattened action buffer holds no entries")
    if not bool(torch.isfinite(pre_tanh).all()):
        return _report(_EXCITATION_DEFAULTS, "nonfinite_input", entries=entries,
                       tol=tolerance,
                       detail="the stored pre-tanh actions are not all finite")

    with torch.no_grad():
        excitation = 0.5 * (torch.tanh(pre_tanh) + 1.0)
        near_upper = excitation >= (1.0 - tolerance)
        near_lower = excitation <= tolerance
        dtype = excitation.dtype
        summary = {
            "near_bound_fraction": float((near_upper | near_lower).to(dtype).mean()),
            "near_upper_fraction": float(near_upper.to(dtype).mean()),
            "near_lower_fraction": float(near_lower.to(dtype).mean()),
            "mean_excitation": float(excitation.mean()),
            "tanh_saturation_fraction": float((pre_tanh.abs() > 2.0).to(dtype).mean()),
        }
    return _report(_EXCITATION_DEFAULTS, STATUS_OK, entries=entries, tol=tolerance,
                   **summary)


def _malformed_events(count, detail, status="malformed_event"):
    return _report(_EVENT_DEFAULTS, status, completed_episodes=count, detail=detail)


def episode_event_summary(events) -> dict:
    """Duration, return and failure-type statistics over explicit episode events.

    ``events`` is a concrete sequence of completed-episode records with the fields in
    :data:`EPISODE_EVENT_FIELDS`, supplied by the caller for **this epoch**. This
    helper is stateless: it reads nothing else, and in particular never reads,
    slices, tails or windows ``PPO.episode_length_his`` / ``episode_loss_his`` (which
    are unbounded run-lifetime lists) or ``episode_length_meter`` /
    ``episode_loss_meter`` (which are a capped 100-sample rolling mean). ``return``
    here is the episode reward sum, not PPO's negated ``episode_loss``.

    Zero completed episodes is **not** zero duration or zero return: the report says
    ``completed_episodes: 0`` with status ``no_completed_episodes`` and ``None``
    statistics. A malformed or nonfinite event yields a status and ``None``
    statistics rather than an exception, because an observer must not abort training;
    ``completed_episodes`` still reports how many events were supplied.

    Failure-flag counts are data driven, so an environment variant's own flag names
    survive. Names outside :data:`OFFICIAL_FAILURE_FLAG_NAMES` — for example the
    legacy ``height`` flag — are listed under ``unofficial_failure_flag_names`` and
    are never presented as official-task flags.
    """
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        return _malformed_events(None, f"events is not a sequence: {type(events).__name__}")

    count = len(events)
    if count == 0:
        return _report(_EVENT_DEFAULTS, "no_completed_episodes", completed_episodes=0,
                       worlds=0, end_reason_counts={}, failure_flag_counts={},
                       official_failure_flag_names=[], unofficial_failure_flag_names=[])

    durations, returns, lengths = [], [], []
    worlds = set()
    end_reason_counts: dict = {}
    flag_counts: dict = {}

    for index, event in enumerate(events):
        label = f"event {index}"
        if not isinstance(event, Mapping):
            return _malformed_events(count, f"{label} is not a mapping: "
                                            f"{type(event).__name__}")
        for field in EPISODE_EVENT_FIELDS:
            if field not in event:
                return _malformed_events(count, f"{label} is missing {field!r}")

        world = event["world"]
        if not _is_count(world) or int(world) < 0:
            return _malformed_events(
                count, f"{label} 'world' is not a non-negative integer: {world!r}")
        length = event["length_controls"]
        if not _is_count(length) or int(length) <= 0:
            return _malformed_events(
                count, f"{label} 'length_controls' is not a positive integer: {length!r}")
        end_reason = event["end_reason"]
        if not isinstance(end_reason, str):
            return _malformed_events(
                count, f"{label} 'end_reason' is not a string: {end_reason!r}")
        flags = event["failure_flags"]
        if not isinstance(flags, Mapping):
            return _malformed_events(
                count, f"{label} 'failure_flags' is not a mapping: "
                       f"{type(flags).__name__}")
        for name, flag in flags.items():
            if not isinstance(name, str) or not isinstance(flag, bool):
                return _malformed_events(
                    count, f"{label} 'failure_flags' entry {name!r} is not a "
                           f"name-to-bool pair")

        for field in ("duration_s", "return"):
            if not _is_real(event[field]):
                return _malformed_events(
                    count, f"{label} {field!r} is not a real number: {event[field]!r}")
            if not math.isfinite(float(event[field])):
                return _malformed_events(
                    count, f"{label} {field!r} is not finite: {event[field]!r}",
                    status="nonfinite_event_field")
        duration = float(event["duration_s"])
        if duration <= 0.0:
            return _malformed_events(
                count, f"{label} 'duration_s' is not positive: {duration!r}")

        durations.append(duration)
        returns.append(float(event["return"]))
        lengths.append(int(length))
        worlds.add(int(world))
        end_reason_counts[end_reason] = end_reason_counts.get(end_reason, 0) + 1
        for name, flag in flags.items():
            flag_counts[name] = flag_counts.get(name, 0) + int(bool(flag))

    official = sorted(name for name in flag_counts if name in OFFICIAL_FAILURE_FLAG_NAMES)
    unofficial = sorted(name for name in flag_counts
                        if name not in OFFICIAL_FAILURE_FLAG_NAMES)
    return _report(
        _EVENT_DEFAULTS, STATUS_OK,
        completed_episodes=count,
        worlds=len(worlds),
        mean_duration_s=float(sum(durations) / count),
        min_duration_s=float(min(durations)),
        max_duration_s=float(max(durations)),
        mean_return=float(sum(returns) / count),
        min_return=float(min(returns)),
        max_return=float(max(returns)),
        mean_length_controls=float(sum(lengths) / count),
        end_reason_counts=dict(sorted(end_reason_counts.items())),
        failure_flag_counts=dict(sorted(flag_counts.items())),
        official_failure_flag_names=official,
        unofficial_failure_flag_names=unofficial,
    )
