"""Aggregate a v2 PPO campaign run tree into per-seed and per-recipe summaries.

Read-only. No simulator, no GPU, no re-evaluation, no training and no ledger
write. Outputs are opened exclusively, so an existing report is never
overwritten.

What this module is careful about
---------------------------------
* **Selection is recomputed**, from the hashed evaluation records, through
  :func:`msk_warp.analysis.myoleg26_selection_v2.behavior_rank_v2` by way of the
  runner's own :func:`select_run`. Nothing is copied from a runner verdict, and
  the in-band component always comes from per-episode **counts**.
* **Five run statuses stay disjoint**: ``selected``, ``incomplete``,
  ``censored``, ``runner_fault`` and ``refused_only``. A censored or incomplete
  run contributes **no** behaviour numbers at all -- not a zero, not a default,
  not a NaN placeholder -- and never reaches a headline figure.
* **Exclusions are visible.** A wall-cap truncation excludes exactly one
  checkpoint, listed with its epoch and reason, and is counted at the campaign
  level. Any other invalid evaluation censors its run instead.
* **Refusals carry no identity by design.** A pre-training refusal has no
  stage/recipe/seed in its result record, because plan resolution itself may be
  what refused. Arm attribution is read from ``launch.json`` **only**; it is
  never inferred from directory position or from a sibling record.
* **Costs may be totalled; behaviour never is.** Control transitions, physics
  steps and wall seconds are reported per seed and summed only as costs.
  Behaviour is per seed, always.
* The crossed-cluster bootstrap is **specified here and implemented elsewhere**
  (:data:`BOOTSTRAP_CONTRACT`). 160 episodes are never 160 independent runs.

Nothing here qualifies a physics result, a gradient, a behaviour claim or an
acceleration claim. A loss improvement is not walking; sample efficiency is not
acceleration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msk_warp.analysis import myoleg26_selection_v2 as S
from msk_warp.analysis import ppo_v2_protocol as P
from scripts import run_myoleg26_ppo_v2 as R

#: Bumped to v2 when the report gained ``foreign_cost_totals``,
#: ``review_stop_reasons`` and the ``foreign_recipe_runs`` campaign category.
SCHEMA_VERSION = "myoleg26-ppo-v2-summary-v2"

FOREIGN_RECIPE_NOTE = (
    "A run whose recipe is not one of the sealed protocol's arms is FOREIGN to "
    "this summary. It is named rather than dropped, its cost is totalled "
    "separately rather than added to the campaign's, it never enters a recipe "
    "table or a promotion decision, and its presence forces a review stop. A "
    "protocol amendment's run -- probe E, for instance -- is such a run: it has "
    "its own protocol digest and its own ledger, and a stage summary is not "
    "where it may be read. Nothing here is hidden: both cost accounts are "
    "reported side by side."
)

#: The five disjoint per-run statuses.
STATUS_SELECTED = "selected"
STATUS_INCOMPLETE = "incomplete"
STATUS_CENSORED = "censored"
STATUS_RUNNER_FAULT = "runner_fault"
STATUS_REFUSED_ONLY = "refused_only"

#: Cost counters that may legitimately be summed across runs.
COST_KEYS = ("training_control_transitions", "evaluation_control_transitions",
             "attempted_control_transitions", "completed_control_transitions",
             "discarded_control_transitions", "replayed_control_transitions",
             "attempted_physics_steps", "completed_physics_steps",
             "discarded_physics_steps", "replayed_physics_steps")

#: Every accounting key summed per run (costs plus the two call counters).
_ACCOUNTING_KEYS = COST_KEYS + ("training_boundary_control_transitions",
                                "partial_update_cost_s",
                                "attempted_control_calls", "returned_control_calls")

#: The host-side timing phases, taken from the runner that writes them so the
#: two cannot drift apart.
TIMING_PHASES = R.TIMING_PHASES

TIMING_ATTRIBUTION = R.TIMING_ATTRIBUTION

TIMING_NOTE = (
    "Host-side wall between phase boundaries, read from an unsynchronized clock. "
    "CUDA work is asynchronous, so GPU time may be attributed across phase "
    "boundaries and a cheap-looking phase may reflect attribution rather than "
    "cost. NOT comparable to the v1 synchronized breakdown, whose timers called "
    "torch.cuda.synchronize around every phase. The parts are not forced to sum: "
    "the unattributed residual is reported so a reader can see what is "
    "unaccounted instead of assuming the split is exhaustive."
)

WILSON_SCOPE = (
    "Wilson score interval over the selection episodes of ONE training seed, "
    "reported only where the episode starts are independent within that seed "
    "(one distinct reset id per episode). It is never an interval across "
    "training seeds, and the episodes of different policies share reset ids, so "
    "pooling them would be a crossed-cluster error."
)

BOOTSTRAP_CONTRACT = (
    "Aggregate uncertainty is seed- and reset-cluster-aware and is NOT computed "
    "here. Specification for the later statistical unit: one bootstrap replicate "
    "resamples training seeds and reset IDs, reusing the same sampled reset-ID "
    "weights across policies rather than resampling shared reset IDs "
    "independently per policy; reset ids shared across policies are crossed "
    "clusters, which is why episode_seed is preserved on every emitted episode "
    "row. Per-seed Wilson intervals are used only where episode starts are "
    "independent within that seed. 160 episodes are never 160 independent runs, "
    "and a directional projection is not a full-gradient cosine."
)

GATE_NOTE = (
    "The walking gate is unchanged and is reported per seed: selection >= 15/16 "
    "survive 4 s; confirmation >= 29/32 with mean vx in [0.8, 1.2] m/s and mean "
    "episode 2D RMSE <= 0.3 m/s. Confirmation is absent until checkpoint "
    "selection is locked, and an absent confirmation is reported as absent, "
    "never as a failure or a zero."
)


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson_interval(successes, trials) -> list:
    """The 95 % Wilson score interval for one seed's independent starts.

    Implemented here rather than imported from the v1 summarizer: that file is
    another unit's driver and is not pinned by the v2 freeze, so importing it
    would put an unpinned executed input into the campaign's provenance.
    """
    successes, trials = int(successes), int(trials)
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(f"invalid binomial counts: {successes} of {trials}")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def _read(path, hashes):
    """Read a JSON artifact and record its hash under its resolved path."""
    path = Path(path)
    hashes[str(path.resolve())] = digest(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _refusal_attribution(result_path, hashes) -> dict:
    """Arm attribution for a refusal, from its ``launch.json`` and nowhere else.

    A pre-training refusal declares no identity triple on purpose, so there is
    nothing to read in the result record and a wrong identity would be worse than
    none. Directory position and sibling records are never consulted.
    """
    launch = Path(result_path).parent / "launch.json"
    entry = {"result_path": str(Path(result_path).resolve()),
             "reason": None, "source": "unattributed", "attribution": None}
    payload = None
    try:
        payload = _read(Path(result_path), hashes)
    except (OSError, ValueError):
        payload = None
    entry["reason"] = (payload or {}).get("refusal")
    if not launch.is_file():
        entry["reason"] = (
            f"{entry['reason']}; no launch.json accompanies this refusal, so its arm is "
            "unattributed and is never inferred from directory position or a sibling record")
        return entry
    record = _read(launch, hashes)
    entry["source"] = "launch.json"
    entry["attribution"] = {"stage": record.get("stage"), "recipe": record.get("recipe"),
                            "seed": record.get("seed"),
                            "segment_index": record.get("segment_index")}
    return entry


def _segment_summary(segment_dir, result, hashes) -> dict:
    """One trained segment's identity, progress, cost and timing."""
    child_path = Path(segment_dir) / "child.json"
    child = _read(child_path, hashes) if child_path.is_file() else None
    launch_path = Path(segment_dir) / "launch.json"
    launch = _read(launch_path, hashes) if launch_path.is_file() else None
    timing = result.get("timing_seconds")
    return {
        "segment_index": result.get("segment_index"),
        "segment_dir": str(Path(segment_dir).resolve()),
        "start_epoch": result.get("start_epoch"),
        "completed_epoch": result.get("completed_epoch"),
        "completed_epochs": result.get("completed_epochs"),
        "published": result.get("published"),
        "stop_reason": result.get("stop_reason"),
        "censored": bool(result.get("censored")),
        "censor_reason": result.get("censor_reason"),
        "worker_wall_seconds": result.get("wall_seconds"),
        "measured_process_wall_seconds": (None if child is None
                                          else child.get("measured_process_wall_s")),
        "returncode": None if child is None else child.get("returncode"),
        "timed_out": None if child is None else child.get("timed_out"),
        "settlement_shape": None if child is None else child.get("settlement_shape"),
        # The ledger's reserve row carries the sealed 460 s constant, not the
        # value this call used, so launch.json is authoritative for the actual
        # work deadline. A reader must not treat the sealed number as measured.
        "launch_work_deadline_s": None if launch is None else launch.get("work_deadline_s"),
        "launch_freeze_sha256": None if launch is None else launch.get("freeze_sha256"),
        "timing_seconds": timing,
        "accounting": result.get("accounting") or {},
    }


def _sum_accounting(segments) -> dict:
    totals = {key: 0 for key in _ACCOUNTING_KEYS}
    totals["partial_update_cost_s"] = 0.0
    worlds, substeps, exact = set(), set(), True
    for segment in segments:
        accounting = segment["accounting"]
        for key in _ACCOUNTING_KEYS:
            value = accounting.get(key)
            if value is None:
                continue
            totals[key] += float(value) if key == "partial_update_cost_s" else int(value)
        if accounting.get("worlds") is not None:
            worlds.add(int(accounting["worlds"]))
        if accounting.get("substeps") is not None:
            substeps.add(int(accounting["substeps"]))
        exact = exact and bool(accounting.get("exact_completed_calls", False))
    totals["worlds"] = sorted(worlds)
    totals["substeps"] = sorted(substeps)
    totals["exact_completed_calls"] = exact
    totals["segments"] = len(segments)
    return totals


def _sum_timing(segments) -> dict:
    """Sum the host-side phase split, carrying its label and its residual.

    A segment without a ``timing_seconds`` record contributes nothing and is
    counted: an unrecorded phase is missing, not zero.
    """
    parts = {phase: 0.0 for phase in TIMING_PHASES}
    worker_wall = 0.0
    process_wall = 0.0
    recorded, missing, process_missing = 0, 0, 0
    for segment in segments:
        timing = segment.get("timing_seconds")
        if isinstance(timing, dict) and timing.get("available"):
            recorded += 1
            for phase in TIMING_PHASES:
                value = timing.get(phase)
                if value is not None:
                    parts[phase] += float(value)
        else:
            missing += 1
        if segment.get("worker_wall_seconds") is not None:
            worker_wall += float(segment["worker_wall_seconds"])
        if segment.get("measured_process_wall_seconds") is None:
            process_missing += 1
        else:
            process_wall += float(segment["measured_process_wall_seconds"])
    attributed = sum(parts.values())
    return {
        "attribution": TIMING_ATTRIBUTION,
        "available": recorded > 0 and missing == 0,
        "segments_with_timing": recorded,
        "segments_without_timing": missing,
        **parts,
        "attributed_sum_s": attributed,
        "worker_wall_seconds": worker_wall,
        "measured_process_wall_seconds": (None if process_missing else process_wall),
        "segments_without_process_wall": process_missing,
        # A computed subtraction, never clamped and never assumed zero.
        "unattributed_residual_s": worker_wall - attributed,
        "attribution_note": TIMING_NOTE,
    }


def _behaviour(candidate) -> dict:
    """The selected checkpoint's behaviour, recomputed from its own record."""
    evaluation = candidate["evaluation"]
    summary = evaluation["summary"]
    episodes = list(evaluation["episodes"])
    survivors = sum(1 for row in episodes if row.get("survived"))
    starts = {row.get("episode_seed") for row in episodes}
    independent = len(starts) == len(episodes) and None not in starts
    rows = [{key: row[key] for key in P.EMITTED_EPISODE_KEYS if key in row}
            for row in episodes]
    return {
        "selected_epoch": int(candidate["epoch"]),
        "checkpoint": candidate.get("checkpoint"),
        "duration_s": float(summary["mean_first_episode_duration_s"]),
        "survival_fraction": float(summary["survival_fraction"]),
        # Per-episode counts times the control timestep. The mean of the
        # per-episode band fractions times the mean duration is a different
        # number whenever episode lengths differ, and is not used.
        "in_band_seconds": S.mean_in_band_seconds(evaluation),
        "rmse_mps": float(summary["mean_episode_velocity_2d_rmse_mps"]),
        "mean_forward_velocity_mps": summary.get("mean_forward_velocity_mps"),
        "survivors": survivors,
        "episodes": len(episodes),
        "independent_starts": len(starts),
        "selection_gate_pass": P.selection_gate_pass(survivors),
        "wilson95": wilson_interval(survivors, len(episodes)) if independent else None,
        "wilson_scope": WILSON_SCOPE if independent else (
            "withheld: the episode starts of this seed are not one per distinct reset id, "
            "so a binomial interval over them would understate the clustering"),
        "episode_rows": rows,
    }


def summarize_run(run_dir, *, freeze_sha256=None) -> dict:
    """Summarize ONE ``(stage, recipe, seed)`` run directory.

    Never pools runs: every seed and every arm has its own epoch 0, so the
    upstream collector refuses a directory that mixes identities.
    """
    run_dir = Path(run_dir)
    hashes = {}
    record = {"run_dir": str(run_dir.resolve()), "stage": None, "recipe": None,
              "seed": None, "status": None, "complete": False,
              "completed_epoch": None, "segments": [], "refusals": [],
              "behaviour": None, "censor_reason": None, "fault": None,
              "selection": None, "incomplete_reason": None}

    try:
        candidates, refused = R.collect_run_records(run_dir)
    except R.RunnerFault as error:
        record.update(status=STATUS_RUNNER_FAULT,
                      fault={"kind": "collect_run_records", "reason": str(error)},
                      selection={"status": STATUS_RUNNER_FAULT, "selected_epoch": None,
                                 "selected_checkpoint": None, "rank": None,
                                 "excluded": [], "wall_cap_truncations": 0,
                                 "checkpoint_status": [], "refused_segments": []})
        record["source_hashes"] = hashes
        return record

    record["refusals"] = [_refusal_attribution(path, hashes) for path in refused]
    refused_set = {str(Path(path).resolve()) for path in refused}

    identities, segments = set(), []
    for segment_dir in sorted(run_dir.glob("segment_*")):
        result_path = segment_dir / "result.json"
        if not result_path.is_file():
            continue
        if str(result_path.resolve()) in refused_set:
            continue                    # a refusal is not a trained segment
        result = _read(result_path, hashes)
        identities.add((result.get("stage"), result.get("recipe"), result.get("seed")))
        segments.append(_segment_summary(segment_dir, result, hashes))
    for path in sorted(run_dir.glob("segment_*/selection_*.json")):
        hashes[str(path.resolve())] = digest(path)

    if len(identities) == 1:
        stage, recipe, seed = next(iter(identities))
        record.update(stage=stage, recipe=recipe, seed=seed)
    record["segments"] = segments
    epochs = [segment["completed_epoch"] for segment in segments
              if segment["completed_epoch"] is not None]
    record["completed_epoch"] = max(epochs) if epochs else None
    record["accounting"] = _sum_accounting(segments)
    record["timing_seconds"] = _sum_timing(segments)

    bindings = {segment["launch_freeze_sha256"] for segment in segments
                if segment["launch_freeze_sha256"] is not None}
    record["freeze_binding"] = {
        "launch_freeze_sha256": (sorted(bindings)[0] if len(bindings) == 1
                                 else sorted(bindings) or None),
        "distinct_launch_bindings": len(bindings),
        "matches_supplied_manifest": (None if freeze_sha256 is None or not bindings
                                      else bindings == {freeze_sha256})}

    censored = [segment for segment in segments if segment["censored"]]
    try:
        selection = R.select_run(candidates, refused=refused)
    except R.CensorRun as error:
        record.update(status=STATUS_CENSORED, censor_reason=str(error),
                      selection={"status": STATUS_CENSORED, "selected_epoch": None,
                                 "selected_checkpoint": None, "rank": None,
                                 "excluded": [], "wall_cap_truncations": 0,
                                 "checkpoint_status": [], "refused_segments": refused})
        record["source_hashes"] = hashes
        return record
    except R.RunnerFault as error:
        record.update(status=STATUS_RUNNER_FAULT,
                      fault={"kind": "select_run", "reason": str(error),
                             "record": error.record},
                      selection={"status": STATUS_RUNNER_FAULT, "selected_epoch": None,
                                 "selected_checkpoint": None, "rank": None,
                                 "excluded": [], "wall_cap_truncations": 0,
                                 "checkpoint_status": [], "refused_segments": refused})
        record["source_hashes"] = hashes
        return record

    selected = selection.get("selected")
    record["selection"] = {
        "status": selection["status"],
        "selected_epoch": None if selected is None else int(selected["epoch"]),
        "selected_checkpoint": None if selected is None else selected.get("checkpoint"),
        "rank": None if selected is None else list(S.behavior_rank_v2(selected["evaluation"])),
        "candidates_considered": selection["candidates_considered"],
        "candidates_compared": selection["candidates_compared"],
        "excluded": selection["excluded"],
        "wall_cap_truncations": selection["wall_cap_truncations"],
        "checkpoint_status": selection["checkpoint_status"],
        "refused_segments": selection["refused_segments"],
    }

    cap = None
    if record["stage"] is not None:
        try:
            cap = P.stage(record["stage"]).epoch_cap_per_seed
        except P.ProtocolError:
            cap = None
    reached = (cap is not None and record["completed_epoch"] is not None
               and record["completed_epoch"] >= cap)

    if censored:
        record.update(status=STATUS_CENSORED,
                      censor_reason="; ".join(str(segment["censor_reason"])
                                              for segment in censored))
    elif not segments and record["refusals"]:
        record.update(status=STATUS_REFUSED_ONLY,
                      incomplete_reason="every segment refused before training began, so "
                                        "nothing was ever produced to select from")
    elif selected is None or not reached:
        record.update(status=STATUS_INCOMPLETE, incomplete_reason=(
            f"selection status {selection['status']!r}; completed epoch "
            f"{record['completed_epoch']!r} against the stage cap {cap!r}. A censored or "
            "unfinished run is incomplete, never zero-valued"))
    else:
        record.update(status=STATUS_SELECTED, complete=True,
                      behaviour=_behaviour(selected))
    record["source_hashes"] = hashes
    return record


def _is_foreign(run) -> bool:
    """Is this run's arm outside the sealed protocol's recipe set?

    A refusal-only run carries no arm at all (attribution is deliberately absent
    from a pre-training refusal record), so it is never foreign: there is nothing
    to be foreign about. See :data:`FOREIGN_RECIPE_NOTE`.
    """
    recipe = run.get("recipe")
    return recipe is not None and recipe not in P.RECIPES


def _seed_outcome(run) -> P.SeedOutcome:
    behaviour = run["behaviour"]
    return P.SeedOutcome(seed=int(run["seed"]), complete=True,
                         duration_s=behaviour["duration_s"],
                         survival_fraction=behaviour["survival_fraction"],
                         in_band_seconds=behaviour["in_band_seconds"],
                         rmse_mps=behaviour["rmse_mps"])


def _recipe_report(stage_key, recipe, runs) -> tuple:
    """``(report, RecipeOutcome | None)`` for one arm at one stage.

    A censored or missing seed is **omitted** from the outcome's seeds rather
    than entered with placeholder numbers: the omission already makes the recipe
    incomplete against its schedule, and there is no code path on which an
    invented zero could be ranked.
    """
    spec = P.stage(stage_key)
    by_seed = {run["seed"]: run for run in runs}
    usable = [by_seed[seed] for seed in spec.seeds
              if seed in by_seed and by_seed[seed]["status"] == STATUS_SELECTED]
    outcome = P.RecipeOutcome(recipe=recipe, stage=spec.key,
                              seeds=tuple(_seed_outcome(run) for run in usable))
    report = {
        "recipe": recipe,
        "stage": spec.key,
        "scheduled_seeds": list(spec.seeds),
        "reported_seeds": [run["seed"] for run in usable],
        "censored_seeds": sorted(seed for seed, run in by_seed.items()
                                 if run["status"] == STATUS_CENSORED),
        "incomplete_seeds": sorted(seed for seed, run in by_seed.items()
                                   if run["status"] in (STATUS_INCOMPLETE,
                                                        STATUS_RUNNER_FAULT)),
        "missing_seeds": [seed for seed in spec.seeds if seed not in by_seed],
        "complete": bool(outcome.complete),
        "medians": None, "eligible": None, "rank": None,
        "per_seed": {run["seed"]: {"duration_s": run["behaviour"]["duration_s"],
                                   "survival_fraction": run["behaviour"]["survival_fraction"],
                                   "in_band_seconds": run["behaviour"]["in_band_seconds"],
                                   "rmse_mps": run["behaviour"]["rmse_mps"]}
                      for run in usable},
    }
    if outcome.complete:
        report["medians"] = P.recipe_medians(outcome)
        report["rank"] = list(P.recipe_rank(outcome))
        if spec.max_promoted > 0:
            report["eligible"] = bool(P.eligible(spec.key, outcome))
    return report, outcome


def summarize(run_root, frozen_manifest, *, ledger=None) -> dict:
    """Aggregate every run under ``run_root`` into one campaign report."""
    run_root = Path(run_root)
    frozen_manifest = Path(frozen_manifest)
    manifest_sha256 = digest(frozen_manifest)
    frozen = json.loads(frozen_manifest.read_text(encoding="utf-8"))

    run_dirs = sorted({segment.parent for segment in run_root.rglob("segment_*")
                       if segment.is_dir()})
    runs = [summarize_run(run_dir, freeze_sha256=manifest_sha256)
            for run_dir in run_dirs]
    runs.sort(key=lambda run: (str(run["stage"] or ""), str(run["recipe"] or ""),
                               run["seed"] if run["seed"] is not None else -1,
                               run["run_dir"]))

    hashes = {str(frozen_manifest.resolve()): manifest_sha256}
    for run in runs:
        hashes.update(run.pop("source_hashes", {}))

    campaign = {
        "run_directories": len(runs),
        "wall_cap_truncations": sum((run["selection"] or {}).get("wall_cap_truncations", 0)
                                    for run in runs),
        "censored_runs": [run["run_dir"] for run in runs
                          if run["status"] == STATUS_CENSORED],
        "incomplete_runs": [run["run_dir"] for run in runs
                            if run["status"] == STATUS_INCOMPLETE],
        "runner_faults": [run["run_dir"] for run in runs
                          if run["status"] == STATUS_RUNNER_FAULT],
        "refused_only_runs": [run["run_dir"] for run in runs
                              if run["status"] == STATUS_REFUSED_ONLY],
        "refused_segments": [entry["result_path"] for run in runs
                             for entry in run["refusals"]],
        "unattributed_refusals": [entry["result_path"] for run in runs
                                  for entry in run["refusals"]
                                  if entry["attribution"] is None],
        "freeze_binding_mismatches": [run["run_dir"] for run in runs
                                      if (run.get("freeze_binding") or {}).get(
                                          "matches_supplied_manifest") is False],
        "segments_without_timing": sum((run.get("timing_seconds") or {}).get(
            "segments_without_timing", 0) for run in runs),
        "foreign_recipe_runs": [
            {"run_dir": run["run_dir"], "stage": run["stage"],
             "recipe": run["recipe"], "seed": run["seed"]}
            for run in runs if _is_foreign(run)],
    }

    # Cost is split, not hidden: a foreign run's counters are reported on their
    # own account so they can never be read as this campaign's spend.
    totals = {key: 0 for key in COST_KEYS}
    foreign_totals = {key: 0 for key in COST_KEYS}
    for run in runs:
        target = foreign_totals if _is_foreign(run) else totals
        for key in COST_KEYS:
            target[key] += int((run.get("accounting") or {}).get(key, 0) or 0)
    totals["note"] = ("Cost totals only. Behaviour is never pooled across seeds or "
                      "policies, and an excluded, censored or incomplete run "
                      "contributes no behaviour number anywhere in this report.")
    foreign_totals["note"] = FOREIGN_RECIPE_NOTE

    recipes, promotion, outcomes = {}, {}, {}
    for stage_key in P.STAGE_ORDER:
        staged = [run for run in runs if run["stage"] == stage_key
                  and run["recipe"] is not None and run["seed"] is not None]
        if not staged:
            continue
        recipes[stage_key], outcomes[stage_key] = {}, []
        for recipe in P.CANONICAL_RECIPE_ORDER:
            arm = [run for run in staged if run["recipe"] == recipe]
            if not arm:
                continue
            report, outcome = _recipe_report(stage_key, recipe, arm)
            recipes[stage_key][recipe] = report
            outcomes[stage_key].append(outcome)
        if P.stage(stage_key).max_promoted <= 0:
            promotion[stage_key] = {
                "stage": stage_key, "decision": "no_promotion_rule",
                "promoted": [], "ranked": [], "ineligible": [], "incomplete": [],
                "reason": "the final stage promotes nothing; its winner is already chosen",
                "stop_for_review": False}
            continue
        try:
            promotion[stage_key] = P.promote(stage_key, outcomes[stage_key]).as_dict()
        except P.ProtocolError as error:
            promotion[stage_key] = {
                "stage": stage_key, "decision": "protocol_error", "promoted": [],
                "ranked": [], "ineligible": [], "incomplete": [],
                "reason": f"the sealed promotion rule refused these outcomes: {error}",
                "stop_for_review": True}

    review = [stage_key for stage_key, decision in promotion.items()
              if decision.get("stop_for_review")]
    review_reasons = [f"promotion:{stage_key}" for stage_key in sorted(review)]
    if campaign["foreign_recipe_runs"]:
        review_reasons.append("foreign_recipe_runs")

    confirmations = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("segment_*/*.json")):
            if path.name.startswith("selection_"):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("kind") not in (None, "selection"):
                    confirmations.append(str(path.resolve()))

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root.resolve()),
        "frozen_manifest": {"path": str(frozen_manifest.resolve()),
                            "sha256": manifest_sha256,
                            "schema_version": frozen.get("schema_version")},
        "protocol_digest": P.protocol_digest(),
        "selection_key_version": S.SELECTION_KEY_VERSION,
        "runs": runs,
        "recipes": recipes,
        "promotion": promotion,
        "review_stop": bool(review_reasons),
        "review_stop_stages": review,
        "review_stop_reasons": review_reasons,
        "campaign": campaign,
        "campaign_cost_totals": totals,
        "foreign_cost_totals": foreign_totals,
        "foreign_recipe_note": FOREIGN_RECIPE_NOTE,
        "confirmation": confirmations or None,
        "walking_gate": dict(P.WALKING_GATE),
        "gate_note": GATE_NOTE,
        "bootstrap_contract": BOOTSTRAP_CONTRACT,
        "bootstrap_implemented_here": False,
        "cluster_note": P.CLUSTER_NOTE,
        "timing_note": TIMING_NOTE,
        "ledger": None,
        "source_hashes": hashes,
        "summarizer_sha256": digest(_HERE),
    }
    if ledger is not None:
        report["ledger"] = _ledger_summary(ledger, hashes)
    return report


def _ledger_summary(path, hashes) -> dict:
    """The settled charge, read from the ledger file and never recomputed."""
    from msk_warp.analysis import ppo_v2_budget as B

    path = Path(path)
    hashes[str(path.resolve())] = digest(path)
    # Read-only across protocols on purpose: this summarizer may be pointed at a
    # ledger of a protocol it knows nothing about, and inspection grants nothing.
    ledger = B.BudgetLedger.open(path, inspect=True)
    charged = ledger.charged()
    return {"path": str(path.resolve()), "protocol_digest": ledger.protocol_digest,
            "charged_s": charged["global_s"], "stage_s": charged["stage_s"],
            "run_s": charged["run_s"], "settled_rows": charged["settled_rows"],
            "total_wall_cap_s": P.TOTAL_WALL_CAP_S,
            "remaining_global_s": P.TOTAL_WALL_CAP_S - charged["global_s"],
            "tail_note": B.LEDGER_TAIL_NOTE}


_CSV_FIELDS = ("run_dir", "stage", "recipe", "seed", "status", "complete",
               "completed_epoch", "selected_epoch", "selected_checkpoint",
               "survivors", "episodes", "independent_starts", "duration_s",
               "survival_fraction", "in_band_seconds", "rmse_mps",
               "mean_forward_velocity_mps", "selection_gate_pass",
               "wall_cap_truncations", "censor_reason", "incomplete_reason",
               "refusals", "worker_wall_seconds", "measured_process_wall_seconds",
               "attributed_sum_s", "unattributed_residual_s") + COST_KEYS


def _csv_row(run) -> dict:
    behaviour = run["behaviour"] or {}
    selection = run["selection"] or {}
    timing = run.get("timing_seconds") or {}
    accounting = run.get("accounting") or {}
    row = {"run_dir": run["run_dir"], "stage": run["stage"], "recipe": run["recipe"],
           "seed": run["seed"], "status": run["status"], "complete": run["complete"],
           "completed_epoch": run["completed_epoch"],
           "selected_epoch": selection.get("selected_epoch"),
           "selected_checkpoint": selection.get("selected_checkpoint"),
           "wall_cap_truncations": selection.get("wall_cap_truncations"),
           "censor_reason": run["censor_reason"],
           "incomplete_reason": run["incomplete_reason"],
           "refusals": len(run["refusals"])}
    for key in ("survivors", "episodes", "independent_starts", "duration_s",
                "survival_fraction", "in_band_seconds", "rmse_mps",
                "mean_forward_velocity_mps", "selection_gate_pass"):
        row[key] = behaviour.get(key)
    for key in ("worker_wall_seconds", "measured_process_wall_seconds",
                "attributed_sum_s", "unattributed_residual_s"):
        row[key] = timing.get(key)
    for key in COST_KEYS:
        row[key] = accounting.get(key)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="summarize_myoleg26_ppo_v2",
        description="Aggregate a v2 PPO campaign run tree. Read-only; no GPU, no training.")
    parser.add_argument("--run-root", required=True, type=Path,
                        help="campaign root; every directory holding segment_* is one run")
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ledger", default=None, type=Path,
                        help="optional budget ledger, read for the settled charge")
    args = parser.parse_args(argv)

    report = summarize(args.run_root, args.frozen_manifest, ledger=args.ledger)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (args.out_dir / "seeds.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(_csv_row(run) for run in report["runs"])
    print(json.dumps({
        "runs": report["campaign"]["run_directories"],
        "review_stop": report["review_stop"],
        "censored_runs": len(report["campaign"]["censored_runs"]),
        "incomplete_runs": len(report["campaign"]["incomplete_runs"]),
        "refused_segments": len(report["campaign"]["refused_segments"]),
        "wall_cap_truncations": report["campaign"]["wall_cap_truncations"],
        "campaign_cost_totals": {key: report["campaign_cost_totals"][key]
                                 for key in ("completed_control_transitions",
                                             "completed_physics_steps")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
