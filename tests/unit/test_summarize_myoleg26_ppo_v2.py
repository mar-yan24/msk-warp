"""Unit tests for the v2 PPO campaign summarizer.

CPU only. Synthetic run trees in temp dirs, hand-computed expectations, no
simulator, no CUDA, no Warp, no subprocess and no training. Nothing here spends
the 28,800 s training budget or the 1,800 s diagnostic budget; test-suite time is
separate accounting.

The summarizer is reporting machinery: none of these tests qualifies a physics
result, a gradient, a behaviour claim or an acceleration claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msk_warp.analysis import myoleg26_selection_v2 as S
from msk_warp.analysis import ppo_v2_protocol as P
from scripts import run_myoleg26_ppo_v2 as R
from scripts import summarize_myoleg26_ppo_v2 as Z

CONTROL_DT = 0.008
STAGE = "screen"
CAP = P.STAGES[STAGE].epoch_cap_per_seed          # 128
SEEDS = P.STAGES[STAGE].seeds                     # (1001, 1002)
BLOCK = P.selection_reset_block(STAGE)            # 11000..11015


# ==========================================================================
# Synthetic artifact builders
# ==========================================================================

def _episode(episode_seed, length, band, *, survived=False, vx=1.0, rmse=0.5,
             end_reason=None):
    return {"episode_seed": int(episode_seed), "world": int(episode_seed) - BLOCK[0],
            "length": int(length), "speed_band_steps": int(band),
            "survived": bool(survived),
            "mean_forward_velocity_mps": float(vx),
            "velocity_2d_rmse_mps": float(rmse),
            "forward_speed_band_fraction": band / length,
            "end_reason": end_reason or ("horizon" if survived else "task_failure")}


def _episodes(*, survivors=0, length=100, band=10, count=16):
    rows = []
    for index in range(count):
        survived = index < survivors
        rows.append(_episode(BLOCK[index], length, band, survived=survived))
    return rows


def _evaluation(episodes, *, control_dt=CONTROL_DT, complete=True, rmse=0.5,
                vx=1.0, duration_s=None, survival=None):
    lengths = [row["length"] for row in episodes]
    mean_duration = (sum(lengths) / len(lengths) * control_dt) if lengths else 0.0
    summary = {
        "survival_fraction": (float(survival) if survival is not None else
                              sum(bool(row["survived"]) for row in episodes) / len(episodes)),
        "mean_first_episode_duration_s": (mean_duration if duration_s is None
                                          else float(duration_s)),
        "mean_episode_velocity_2d_rmse_mps": rmse,
        "mean_forward_velocity_mps": vx,
        "mean_displacement_m": 1.0,
        "forward_speed_band_fraction": (sum(row["speed_band_steps"] for row in episodes)
                                        / max(sum(lengths), 1)),
    }
    return {"summary": summary, "episodes": episodes, "control_dt": control_dt,
            "complete": complete, "policy_mode": "deterministic_mean",
            "horizon_control_steps": 500,
            "accounting": {"simulated_control_transitions": 1600,
                           "scored_first_episode_transitions": sum(lengths),
                           "ignored_restarted_world_transitions": 0,
                           "simulated_physics_steps": 6400}}


def _accounting(*, worlds=64, training=8192, evaluation=1600, replayed=0):
    attempted = training + evaluation
    substeps = int(P.PHYSICS_SUBSTEPS)
    return {"worlds": worlds, "substeps": substeps,
            "training_control_transitions": training,
            "training_boundary_control_transitions": training,
            "evaluation_control_transitions": evaluation,
            "attempted_control_transitions": attempted,
            "completed_control_transitions": attempted,
            "discarded_control_transitions": 0,
            "replayed_control_transitions": replayed,
            "attempted_physics_steps": attempted * substeps,
            "completed_physics_steps": attempted * substeps,
            "discarded_physics_steps": 0,
            "replayed_physics_steps": replayed * substeps,
            "partial_update_cost_s": 0.0,
            "attempted_control_calls": attempted // worlds,
            "returned_control_calls": attempted // worlds,
            "exact_completed_calls": True}


def _timing(*, build=10.0, rollout=100.0, update=50.0, evaluation=20.0,
            diagnostics=2.0, capture=8.0, total=200.0):
    parts = {"build_s": build, "rollout_s": rollout, "update_s": update,
             "evaluation_s": evaluation, "diagnostics_s": diagnostics,
             "capture_publish_s": capture}
    attributed = sum(parts.values())
    return {"attribution": "unsynchronized_host_side", "available": True,
            **parts, "attributed_sum_s": attributed,
            "total_wall_seconds": total,
            "unattributed_residual_s": total - attributed,
            "attribution_note": "host-side; async GPU work may cross boundaries",
            "residual_note": "meaningful only when both clocks are perf_counter",
            "v1_comparability": "not comparable to the v1 synchronized breakdown"}


def _write_permissive(path, value):
    """A writer that accepts nonfinite values, which the runner's own never does.

    ``write_json_exclusive`` uses ``allow_nan=False``, so a NaN cannot reach disk
    through the runner. A hand-edited, foreign or truncated artifact still can,
    and the summarizer must refuse it rather than rank it.
    """
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")


def _segment(run_dir, index, *, stage=STAGE, recipe="g990_e010", seed=SEEDS[0],
             start_epoch=0, end_epoch=64, completed_epoch=None, evaluations=(),
             censored=False, censor_reason=None, refusal=False, launch=True,
             accounting=None, timing=None, wall=200.0, process_wall=205.0,
             published=True, child=True, freeze_sha256="f" * 64, allow_nan=False):
    """Write one synthetic segment directory. ``evaluations`` is (epoch, record)."""
    segment = Path(run_dir) / f"segment_{index:04d}"
    segment.mkdir(parents=True)
    write_selection = _write_permissive if allow_nan else R.write_json_exclusive
    if launch:
        R.write_json_exclusive(segment / "launch.json", {
            "schema_version": R.LAUNCH_SCHEMA, "runner_schema": R.SCHEMA_VERSION,
            "stage": stage, "recipe": recipe, "seed": int(seed),
            "segment_index": int(index), "start_epoch": int(start_epoch),
            "end_epoch": int(end_epoch), "epochs": int(end_epoch - start_epoch),
            "work_deadline_s": 300.0, "reserved_bound_s": 360.0,
            "shutdown_allowance_s": 60.0,
            "remaining_before": {"global_s": 28800.0, "stage_s": 7200.0,
                                 "seed_s": 1200.0, "binding": "seed"},
            "freeze": "frozen.json", "freeze_sha256": freeze_sha256,
            "parent_segment": None, "parent_result": None, "device": "cuda:0"})
    for epoch, record in evaluations:
        write_selection(segment / f"selection_{int(epoch):04d}.json", record)
    if refusal:
        R.write_json_exclusive(segment / "result.json", {
            "schema_version": R.RESULT_SCHEMA, "runner_schema": R.SCHEMA_VERSION,
            "status": R.REFUSED_BEFORE_TRAINING, "training_began": False,
            "refusal": "RunnerRefusal: synthetic refusal before training",
            "stop_reason": R.REFUSED_BEFORE_TRAINING, "published": False,
            "boundary_is_live": False, "censored": False, "censor_reason": None,
            "completed_epoch": None, "completed_epochs": 0,
            "accounting": R._zero_accounting(), "wall_seconds": 1.25})
        return segment
    finished = end_epoch if completed_epoch is None else int(completed_epoch)
    R.write_json_exclusive(segment / "result.json", {
        "schema_version": R.RESULT_SCHEMA, "runner_schema": R.SCHEMA_VERSION,
        "stage": stage, "recipe": recipe, "seed": int(seed),
        "segment_index": int(index), "training_began": True,
        "start_epoch": int(start_epoch), "end_epoch": int(end_epoch),
        "epochs_requested": int(end_epoch - start_epoch),
        "completed_epoch": finished,
        "completed_epochs": finished - int(start_epoch),
        "boundary_is_live": bool(published), "published": bool(published),
        "stop_reason": "censored_nonfinite" if censored else "epoch_budget",
        "censored": bool(censored), "censor_reason": censor_reason,
        "partial_detail": None, "unexpected_error": None,
        "lineage": {"segment_sha256": "a" * 64 if published else None,
                    "segment_path": str(segment / "segment.ptc") if published else None,
                    "parent_segment_sha256": None,
                    "valid_boundary_path": str(segment / "segment.ptc"),
                    "valid_boundary_epoch": finished},
        "resume": None,
        "accounting": accounting or _accounting(),
        "losses": [], "epoch_events": [],
        "evaluations": [str(segment / f"selection_{int(epoch):04d}.json")
                        for epoch, _ in evaluations],
        "work_deadline_s": 300.0, "capture_reserve_s": 30.0,
        "timing_seconds": timing or _timing(),
        "wall_seconds": float(wall)})
    if child:
        R.write_json_exclusive(segment / "child.json", {
            "schema_version": R.CHILD_SCHEMA, "argv": ["python", "worker"],
            "argv_sha256": "b" * 64, "returncode": 0, "timed_out": False,
            "measured_process_wall_s": float(process_wall),
            "settlement_shape": "trained", "reserved_bound_s": 360.0,
            "reservation_id": f"{stage}-{recipe}-s{seed}-g{index}-r1",
            "settled_counters": None})
    return segment


def _run(root, *, recipe="g990_e010", seed=SEEDS[0], survivors=0, length=100,
         band=10, rmse=0.5, complete=True, **kwargs):
    """A two-segment run reaching the stage cap, evaluated at 0, 64 and 128."""
    run_dir = Path(root) / f"{recipe}_seed{seed}"
    first = _evaluation(_episodes(survivors=0, length=max(length // 2, 1), band=1))
    best = _evaluation(_episodes(survivors=survivors, length=length, band=band),
                       rmse=rmse)
    _segment(run_dir, 0, recipe=recipe, seed=seed, start_epoch=0, end_epoch=64,
             evaluations=((0, _selection(0, first, recipe=recipe, seed=seed)),
                          (64, _selection(64, best, recipe=recipe, seed=seed))),
             **kwargs)
    _segment(run_dir, 1, recipe=recipe, seed=seed, start_epoch=64, end_epoch=128,
             completed_epoch=128 if complete else 96,
             evaluations=((128, _selection(128, first, recipe=recipe, seed=seed)),),
             **kwargs)
    return run_dir


def _selection(epoch, evaluation, *, recipe="g990_e010", seed=SEEDS[0], stage=STAGE):
    return {"epoch": int(epoch), "kind": "selection", "stage": stage,
            "recipe": recipe, "seed": int(seed),
            "checkpoint": f"epoch_{int(epoch):04d}.pt",
            "checkpoint_sha256": "c" * 64, "reset_block": list(BLOCK),
            "evaluation": evaluation}


def _manifest(tmp_path, *, files=None):
    path = Path(tmp_path) / "frozen.json"
    R.write_json_exclusive(path, {"schema_version": "myoleg26-ppo-v2-freeze-v1",
                                  "files": files or {}, "protocol": {"stages": {}}})
    return path


# ==========================================================================
# Unit 1 -- per-run selection recomputation
# ==========================================================================

def test_a_run_is_summarized_with_its_recomputed_selection(tmp_path):
    root = tmp_path / "campaign"
    run_dir = _run(root, survivors=3, length=200, band=40)
    manifest = _manifest(tmp_path)
    report = Z.summarize(root, manifest)

    assert len(report["runs"]) == 1
    run = report["runs"][0]
    assert (run["stage"], run["recipe"], run["seed"]) == (STAGE, "g990_e010", SEEDS[0])
    assert run["status"] == Z.STATUS_SELECTED
    assert run["complete"] is True
    assert run["selection"]["selected_epoch"] == 64
    assert run["behaviour"]["survivors"] == 3
    assert run["behaviour"]["episodes"] == 16
    # Recomputed from the hashed record, not copied from any runner output.
    best = _evaluation(_episodes(survivors=3, length=200, band=40))
    assert tuple(run["selection"]["rank"]) == S.behavior_rank_v2(best)


def test_in_band_seconds_uses_counts_not_the_per_episode_fraction_shortcut(tmp_path):
    """Binding. Unequal episode lengths make the two forms different numbers.

    The *pooled* summary fraction times the mean duration is algebraically the
    same quantity, so it is not the trap; the mean of the per-episode
    ``forward_speed_band_fraction`` values is, because it reweights short
    episodes upward.
    """
    root = tmp_path / "campaign"
    run_dir = root / "uneven"
    episodes = [_episode(BLOCK[0], 100, 50), _episode(BLOCK[1], 300, 30)]
    evaluation = _evaluation(episodes)
    _segment(run_dir, 0, end_epoch=128, completed_epoch=128,
             evaluations=((0, _selection(0, evaluation)),))
    report = Z.summarize(root, _manifest(tmp_path))
    run = report["runs"][0]

    shortcut = (sum(row["forward_speed_band_fraction"] for row in episodes) / 2
                * evaluation["summary"]["mean_first_episode_duration_s"])
    assert shortcut == 0.48
    assert run["behaviour"]["in_band_seconds"] == 0.32
    assert run["behaviour"]["in_band_seconds"] != shortcut
    assert run["behaviour"]["in_band_seconds"] == S.mean_in_band_seconds(evaluation)


def test_the_earliest_epoch_wins_an_exact_tie_in_both_arrival_orders(tmp_path):
    """Binding. Keyed on epoch, never on the order the records are read."""
    tied = _evaluation(_episodes(survivors=2, length=150, band=20))
    selected = []
    for name, layout in (("natural", (0, 64)), ("reversed", (64, 0))):
        root = tmp_path / name
        run_dir = root / "tie"
        for index, epoch in enumerate(layout):
            _segment(run_dir, index, start_epoch=epoch,
                     end_epoch=epoch + 64, completed_epoch=128,
                     evaluations=((epoch, _selection(epoch, tied)),))
        report = Z.summarize(root, _manifest(root))
        selected.append(report["runs"][0]["selection"]["selected_epoch"])
    assert selected == [0, 0]


# ==========================================================================
# Unit 2 -- the exclusion / censor split
# ==========================================================================

def test_a_wall_cap_truncation_is_excluded_with_its_epoch_and_reason(tmp_path):
    root = tmp_path / "campaign"
    run_dir = root / "truncated"
    good = _evaluation(_episodes(survivors=1, length=120, band=30))
    rows = _episodes(length=60, band=5)
    rows[0]["end_reason"] = "wall_cap"
    cut = _evaluation(rows, complete=False)
    _segment(run_dir, 0, end_epoch=128, completed_epoch=128,
             evaluations=((0, _selection(0, good)), (64, _selection(64, cut))))
    report = Z.summarize(root, _manifest(tmp_path))
    run = report["runs"][0]

    assert run["status"] == Z.STATUS_SELECTED
    assert run["selection"]["selected_epoch"] == 0
    excluded = run["selection"]["excluded"]
    assert [entry["epoch"] for entry in excluded] == [64]
    assert excluded[0]["status"] == R.EVAL_TRUNCATED
    assert "complete" in (excluded[0]["reason"] or "")
    assert run["selection"]["wall_cap_truncations"] == 1
    assert report["campaign"]["wall_cap_truncations"] == 1
    # The excluded checkpoint never reaches a headline number.
    assert run["behaviour"]["in_band_seconds"] == S.mean_in_band_seconds(good)


@pytest.mark.parametrize("mutate,marker", [
    (lambda evaluation: evaluation.update(episodes=[]), "empty"),
    (lambda evaluation: evaluation["summary"].update(
        mean_episode_velocity_2d_rmse_mps=float("nan")), "finite"),
    (lambda evaluation: evaluation.pop("control_dt"), "control_dt"),
    # ``complete`` false with no wall-cap episode is NOT a truncation: the
    # scheduling artifact has to be the sole defect and to be evidenced by the
    # record, so this censors instead of excluding one checkpoint.
    (lambda evaluation: evaluation.update(complete=False), "incomplete"),
])
def test_a_malformed_or_nonfinite_evaluation_censors_the_run(tmp_path, mutate, marker):
    """Binding. Not a truncation: the run is censored and never ranked, and the
    reason distinguishes the cause."""
    root = tmp_path / "campaign"
    run_dir = root / "broken"
    bad = _evaluation(_episodes(length=80, band=8))
    mutate(bad)
    _segment(run_dir, 0, end_epoch=128, completed_epoch=128, allow_nan=True,
             evaluations=((0, _selection(0, bad)),))
    report = Z.summarize(root, _manifest(tmp_path))
    run = report["runs"][0]

    assert run["status"] == Z.STATUS_CENSORED
    assert run["behaviour"] is None
    assert marker in run["censor_reason"]
    assert run["run_dir"] in report["campaign"]["censored_runs"]
    assert run["selection"]["selected_epoch"] is None


def test_a_censored_seed_is_incomplete_and_never_averaged(tmp_path):
    root = tmp_path / "campaign"
    healthy = _run(root, recipe="g990_e010", seed=SEEDS[0], survivors=4,
                   length=250, band=60)
    hurt = _run(root, recipe="g990_e010", seed=SEEDS[1], survivors=0, length=40,
                band=1, censored=True, censor_reason="nonfinite update loss")
    report = Z.summarize(root, _manifest(tmp_path))

    statuses = {run["seed"]: run["status"] for run in report["runs"]}
    assert statuses[SEEDS[1]] == Z.STATUS_CENSORED
    arm = report["recipes"][STAGE]["g990_e010"]
    assert arm["complete"] is False
    assert arm["medians"] is None
    assert arm["eligible"] is None
    assert "g990_e010" in report["promotion"][STAGE]["incomplete"]
    assert report["promotion"][STAGE]["decision"] == P.REVIEW_STOP


# ==========================================================================
# Unit 3 -- refusals carry no identity of their own
# ==========================================================================

def test_a_refusal_is_attributed_only_from_its_launch_record(tmp_path):
    root = tmp_path / "campaign"
    run_dir = root / "refused"
    _segment(run_dir, 0, refusal=True, recipe="g998_e010", seed=SEEDS[1])
    report = Z.summarize(root, _manifest(tmp_path))
    run = report["runs"][0]

    assert run["status"] == Z.STATUS_REFUSED_ONLY
    assert run["behaviour"] is None
    assert run["stage"] is None and run["recipe"] is None and run["seed"] is None
    refusal = run["refusals"][0]
    assert refusal["source"] == "launch.json"
    assert refusal["attribution"] == {"stage": STAGE, "recipe": "g998_e010",
                                      "seed": SEEDS[1], "segment_index": 0}
    assert len(report["campaign"]["refused_segments"]) == 1
    # A refusal is not a trained segment and is never counted as one.
    assert run["segments"] == []


def test_a_refusal_without_a_launch_record_is_never_inferred(tmp_path):
    root = tmp_path / "campaign"
    run_dir = root / "orphan"
    _segment(run_dir, 0, refusal=True, launch=False)
    _segment(run_dir, 1, start_epoch=0, end_epoch=128, completed_epoch=128,
             recipe="g990_e000", seed=SEEDS[0],
             evaluations=((0, _selection(0, _evaluation(_episodes(survivors=1)),
                                         recipe="g990_e000")),))
    report = Z.summarize(root, _manifest(tmp_path))
    refusal = report["runs"][0]["refusals"][0]

    assert refusal["attribution"] is None
    assert refusal["source"] == "unattributed"
    assert "launch" in refusal["reason"]
    # The trained sibling's identity is NOT borrowed for the refusal.
    assert report["runs"][0]["recipe"] == "g990_e000"


# ==========================================================================
# Unit 4 -- eligibility, ranking and promotion
# ==========================================================================

def _complete_arm(root, recipe, *, survivors, length, band, rmse):
    for seed in SEEDS:
        _run(root, recipe=recipe, seed=seed, survivors=survivors, length=length,
             band=band, rmse=rmse)


def test_screen_eligibility_and_promotion_match_the_sealed_rule(tmp_path):
    """Hand-computed. 250 controls * 0.008 s = 2.0 s median duration, which
    clears the 1.25 s screen threshold; 40 controls = 0.32 s does not, and its
    zero survival does not reach 1/16 either."""
    root = tmp_path / "campaign"
    _complete_arm(root, "g990_e010", survivors=4, length=250, band=60, rmse=0.4)
    _complete_arm(root, "g998_e010", survivors=2, length=200, band=50, rmse=0.6)
    _complete_arm(root, "g990_e000", survivors=0, length=40, band=0, rmse=0.9)
    report = Z.summarize(root, _manifest(tmp_path))
    arms = report["recipes"][STAGE]

    assert arms["g990_e010"]["medians"]["median_duration_s"] == 2.0
    assert arms["g990_e010"]["eligible"] is True
    assert arms["g998_e010"]["medians"]["median_duration_s"] == 1.6
    assert arms["g998_e010"]["eligible"] is True
    assert arms["g990_e000"]["medians"]["median_duration_s"] == 0.32
    assert arms["g990_e000"]["medians"]["max_survival_fraction"] == 0.0
    assert arms["g990_e000"]["eligible"] is False
    decision = report["promotion"][STAGE]
    assert decision["decision"] == "promote"
    assert decision["promoted"] == ["g990_e010", "g998_e010"]
    assert decision["ineligible"] == ["g990_e000"]
    assert report["review_stop"] is False


def test_no_eligible_recipe_emits_a_review_stop(tmp_path):
    root = tmp_path / "campaign"
    _complete_arm(root, "g990_e010", survivors=0, length=40, band=0, rmse=0.9)
    _complete_arm(root, "g998_e010", survivors=0, length=50, band=0, rmse=0.9)
    report = Z.summarize(root, _manifest(tmp_path))

    assert report["review_stop"] is True
    assert report["promotion"][STAGE]["decision"] == P.REVIEW_STOP
    assert report["promotion"][STAGE]["promoted"] == []
    assert "STOPS FOR REVIEW" in report["promotion"][STAGE]["reason"]


# ==========================================================================
# Unit 5 -- accounting, timing and the gate
# ==========================================================================

def test_accounting_and_timing_are_summed_per_run_and_never_pooled(tmp_path):
    root = tmp_path / "campaign"
    _run(root, survivors=1, length=150, band=20)
    report = Z.summarize(root, _manifest(tmp_path))
    run = report["runs"][0]
    counts = run["accounting"]
    substeps = int(P.PHYSICS_SUBSTEPS)

    single = _accounting()
    assert counts["training_control_transitions"] == 2 * single["training_control_transitions"]
    assert counts["evaluation_control_transitions"] == 2 * single["evaluation_control_transitions"]
    assert counts["completed_physics_steps"] == counts["completed_control_transitions"] * substeps
    timing = run["timing_seconds"]
    assert timing["rollout_s"] == 200.0 and timing["update_s"] == 100.0
    assert timing["evaluation_s"] == 40.0 and timing["diagnostics_s"] == 4.0
    assert timing["attribution"] == "unsynchronized_host_side"
    assert timing["worker_wall_seconds"] == 400.0
    assert timing["measured_process_wall_seconds"] == 410.0
    assert timing["unattributed_residual_s"] == 400.0 - timing["attributed_sum_s"]
    # Costs may be totalled; behaviour never is.
    assert report["campaign_cost_totals"]["completed_control_transitions"] == \
        counts["completed_control_transitions"]
    assert "mean" not in report["campaign_cost_totals"]


def test_the_walking_gate_and_wilson_interval_are_reported_per_seed(tmp_path):
    root = tmp_path / "campaign"
    _run(root, recipe="g990_e010", seed=SEEDS[0], survivors=1, length=100, band=10)
    _run(root, recipe="g990_e010", seed=SEEDS[1], survivors=15, length=500,
         band=400)
    report = Z.summarize(root, _manifest(tmp_path))
    by_seed = {run["seed"]: run["behaviour"] for run in report["runs"]}

    assert by_seed[SEEDS[0]]["selection_gate_pass"] is False
    assert by_seed[SEEDS[0]]["wilson95"] == [0.01111934476464252, 0.28328737570298945]
    assert by_seed[SEEDS[1]]["selection_gate_pass"] is True
    assert by_seed[SEEDS[1]]["wilson95"] == [0.7167126242970107, 0.9888806552353576]
    assert by_seed[SEEDS[0]]["independent_starts"] == 16
    assert "within" in by_seed[SEEDS[0]]["wilson_scope"]
    assert report["confirmation"] is None


def test_episode_rows_keep_their_reset_ids_and_the_cluster_contract(tmp_path):
    root = tmp_path / "campaign"
    _run(root, survivors=2, length=180, band=30)
    report = Z.summarize(root, _manifest(tmp_path))
    rows = report["runs"][0]["behaviour"]["episode_rows"]

    assert [row["episode_seed"] for row in rows] == list(BLOCK)
    assert all("survived" in row and "length" in row for row in rows)
    contract = report["bootstrap_contract"]
    assert "resamples training seeds and reset IDs" in contract
    assert "same sampled reset-ID weights across policies" in contract
    assert "160 episodes are never 160 independent runs" in contract
    assert report["bootstrap_implemented_here"] is False


# ==========================================================================
# Unit 6 -- provenance and exclusive output
# ==========================================================================

def test_every_consumed_artifact_is_hashed_and_the_freeze_binding_checked(tmp_path):
    root = tmp_path / "campaign"
    run_dir = _run(root, survivors=1, freeze_sha256="d" * 64)
    manifest = _manifest(tmp_path)
    report = Z.summarize(root, manifest)

    hashes = report["source_hashes"]
    assert hashes[str((run_dir / "segment_0000" / "result.json").resolve())] == \
        Z.digest(run_dir / "segment_0000" / "result.json")
    assert str((run_dir / "segment_0000" / "selection_0000.json").resolve()) in hashes
    assert report["frozen_manifest"]["sha256"] == Z.digest(manifest)
    assert report["summarizer_sha256"] == Z.digest(Z._HERE)
    binding = report["runs"][0]["freeze_binding"]
    assert binding["launch_freeze_sha256"] == "d" * 64
    assert binding["matches_supplied_manifest"] is False
    assert report["campaign"]["freeze_binding_mismatches"] == [report["runs"][0]["run_dir"]]


def test_the_report_refuses_to_overwrite_an_existing_output(tmp_path):
    root = tmp_path / "campaign"
    _run(root, survivors=1)
    manifest = _manifest(tmp_path)
    out = tmp_path / "out"
    argv = ["--run-root", str(root), "--frozen-manifest", str(manifest),
            "--out-dir", str(out)]
    assert Z.main(argv) == 0
    assert (out / "summary.json").is_file() and (out / "seeds.csv").is_file()
    with pytest.raises(FileExistsError):
        Z.main(argv)


def test_the_seed_csv_has_one_row_per_run_including_censored_runs(tmp_path):
    root = tmp_path / "campaign"
    _run(root, recipe="g990_e010", seed=SEEDS[0], survivors=1)
    _run(root, recipe="g990_e010", seed=SEEDS[1], survivors=0, censored=True,
         censor_reason="nonfinite parameter actor.mu.weight")
    out = tmp_path / "out"
    assert Z.main(["--run-root", str(root), "--frozen-manifest",
                   str(_manifest(tmp_path)), "--out-dir", str(out)]) == 0
    lines = (out / "seeds.csv").read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 3
    assert "status" in lines[0] and "censor_reason" in lines[0]
    assert any(Z.STATUS_CENSORED in line for line in lines[1:])


def test_a_mixed_identity_run_directory_is_reported_as_a_runner_fault(tmp_path):
    root = tmp_path / "campaign"
    run_dir = root / "mixed"
    _segment(run_dir, 0, seed=SEEDS[0], end_epoch=128, completed_epoch=128,
             evaluations=((0, _selection(0, _evaluation(_episodes()))),))
    _segment(run_dir, 1, seed=SEEDS[1], start_epoch=64, end_epoch=128,
             completed_epoch=128)
    report = Z.summarize(root, _manifest(tmp_path))
    run = report["runs"][0]

    assert run["status"] == Z.STATUS_RUNNER_FAULT
    assert run["behaviour"] is None
    assert "identity" in run["fault"]["reason"]
    assert run["run_dir"] in report["campaign"]["runner_faults"]


def test_an_unfinished_run_is_incomplete_but_its_costs_are_still_counted(tmp_path):
    """Binding. Not reaching the stage cap is its own category: no behaviour
    number is emitted, yet the wall and transitions it really spent are."""
    root = tmp_path / "campaign"
    _run(root, survivors=2, length=200, band=30, complete=False)
    report = Z.summarize(root, _manifest(tmp_path))
    run = report["runs"][0]

    assert run["status"] == Z.STATUS_INCOMPLETE
    assert run["complete"] is False
    assert run["behaviour"] is None
    assert run["completed_epoch"] == 96
    assert "stage cap" in run["incomplete_reason"]
    assert run["run_dir"] in report["campaign"]["incomplete_runs"]
    assert run["run_dir"] not in report["campaign"]["censored_runs"]
    # A selection was still recomputed and recorded; it just does not promote.
    assert run["selection"]["selected_epoch"] == 64
    # Real spent cost is never discarded because the run did not finish.
    assert report["campaign_cost_totals"]["completed_control_transitions"] ==         2 * _accounting()["completed_control_transitions"]
    assert report["recipes"][STAGE]["g990_e010"]["complete"] is False
    assert report["recipes"][STAGE]["g990_e010"]["incomplete_seeds"] == [SEEDS[0]]


def test_a_censored_run_contributes_cost_but_no_behaviour(tmp_path):
    """Binding. Censoring suppresses the metrics, not the accounting."""
    root = tmp_path / "campaign"
    _run(root, survivors=3, length=300, band=90, censored=True,
         censor_reason="nonfinite update loss 'actor_loss'")
    report = Z.summarize(root, _manifest(tmp_path))
    run = report["runs"][0]

    assert run["status"] == Z.STATUS_CENSORED
    assert run["behaviour"] is None
    assert "nonfinite" in run["censor_reason"]
    assert report["campaign_cost_totals"]["completed_physics_steps"] ==         2 * _accounting()["completed_physics_steps"]
    assert report["recipes"][STAGE]["g990_e010"]["censored_seeds"] == [SEEDS[0]]
    assert report["recipes"][STAGE]["g990_e010"]["per_seed"] == {}
