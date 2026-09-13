"""Summarize a completed frozen baseline with training seeds as the statistical unit.

No simulator imports or GPU work. Retain censored/failed seeds in the report;
missing confirmation is never dropped from the five-seed success denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_interval(values, *, seed=20260912):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n_training_seeds": 0, "mean": None, "ci95": None}
    if not np.isfinite(values).all():
        raise ValueError("Cannot summarize nonfinite seed metrics")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {"n_training_seeds": len(values), "mean": float(values.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist() if len(values) >= 5 else None}


def wilson_interval(successes, trials):
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("Invalid binomial counts")
    z, p = 1.959963984540054, successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials))/denominator
    half = z*np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/denominator
    return [max(0., float(center-half)), min(1., float(center+half))]


def summarize(run_dir, frozen_path):
    run_dir, frozen_path = Path(run_dir), Path(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    if (manifest.get("freeze") != frozen or manifest.get("freeze_sha256") != digest(frozen_path)
            or manifest.get("scope") != "registered_full_budget" or manifest.get("seeds") != list(range(5))):
        raise ValueError("Require the exact frozen five-seed registered run")
    combined = json.loads((run_dir / "results.json").read_text())
    if [result["seed"] for result in combined["seeds"]] != list(range(5)):
        raise ValueError("Combined report must retain all five registered seeds")
    rows = []
    source_hashes = {str(path.resolve()): digest(path) for path in
                     (frozen_path, run_dir / "run_manifest.json", run_dir / "results.json")}
    for result in combined["seeds"]:
        seed = result["seed"]
        seed_dir = run_dir / f"seed_{seed}"
        result_path = seed_dir / "result.json"
        if result != json.loads(result_path.read_text()):
            raise ValueError("Combined and per-seed reports disagree")
        source_hashes[str(result_path.resolve())] = digest(result_path)
        count, timing = result["accounting"], result["timing_seconds"]
        if count["training_control_transitions"] != result["training_completed_control_calls"] * 64:
            raise ValueError("Training call accounting differs")
        for phase in ("training", "evaluation"):
            if count[f"{phase}_physics_steps"] != 4 * count[f"{phase}_control_transitions"]:
                raise ValueError("Physics/control accounting differs")
        if count["total_control_transitions"] != count["training_control_transitions"] + count["evaluation_control_transitions"]:
            raise ValueError("Total controls differ from phase counts")
        if count["total_physics_steps"] != count["total_control_transitions"] * 4:
            raise ValueError("Total physics count differs")
        if not result["censored"] and count["training_control_transitions"] != 1048576:
            raise ValueError("Uncensored seed did not complete the fixed budget")
        selected = result.get("selected_checkpoint")
        if selected and digest(seed_dir / selected["filename"]) != selected["sha256"]:
            raise ValueError("Behavior-selected checkpoint bytes changed")
        confirmations = [e for e in result["evaluations"] if e["kind"] == "confirmation"]
        if len(confirmations) > 1:
            raise ValueError("Multiple held-out confirmation attempts")
        confirmation = confirmations[0] if confirmations else None
        complete = bool(confirmation and confirmation["complete"])
        if complete and (len(confirmation["episodes"]) != 32 or confirmation["checkpoint_sha256"] != selected["sha256"]):
            raise ValueError("Confirmation episodes/checkpoint differ from selection")
        first_success = next((e for e in result["evaluations"] if e["kind"] == "selection" and e["success"]), None)
        summary = confirmation["summary"] if complete else {}
        row = {"seed": seed, "completed_epochs": result["completed_epochs"], "censored": result["censored"],
               "stop_reason": result["stop_reason"], "selected_epoch": selected["epoch"] if selected else None,
               "confirmation_complete": complete, "confirmed_success": bool(complete and confirmation["success"]),
               "survivors_of_32": sum(e["survived"] for e in confirmation["episodes"]) if complete else None,
               "first_selection_success_controls": first_success["training_control_transitions"] if first_success else None,
               "first_selection_success_seed_wall_seconds": first_success["seed_wall_seconds"] if first_success else None,
               "success_not_observed_by_controls": None if first_success else count["training_control_transitions"],
               "seed_wall_seconds": timing["total_seed_wall"],
               "training_loop_wall_seconds": timing["training_rollout"] + timing["training_update"],
               "evaluation_wall_seconds": timing["evaluation"], **count}
        for field in ("mean_forward_velocity_mps", "mean_displacement_m", "mean_first_episode_duration_s",
                      "mean_episode_velocity_2d_rmse_mps", "pooled_velocity_2d_rmse_mps", "mean_excitation_squared"):
            row[field] = summary.get(field)
        rows.append(row)
    successful = sum(row["confirmed_success"] for row in rows)
    metrics = {name: mean_interval([row[name] for row in rows if row[name] is not None]) for name in
               ("seed_wall_seconds", "training_loop_wall_seconds", "mean_forward_velocity_mps",
                "mean_first_episode_duration_s", "mean_displacement_m", "mean_episode_velocity_2d_rmse_mps")}
    totals = {field: sum(row[field] for row in rows) for field in
              ("training_control_transitions", "training_physics_steps", "evaluation_control_transitions",
               "evaluation_physics_steps", "total_control_transitions", "total_physics_steps")}
    return {"schema_version": "myoleg26-baseline-summary-v1", "seeds": rows, "totals": totals,
            "total_run_wall_seconds": combined["total_run_wall_seconds"], "seed_metric_summaries": metrics,
            "confirmed_successes": successful, "registered_seed_count": 5,
            "success_fraction": successful / 5, "success_fraction_wilson_ci95": wilson_interval(successful, 5),
            "missing_confirmation_count": sum(not row["confirmation_complete"] for row in rows),
            "ci_scope": "10000 percentile-bootstrap resamples of training seeds, N<=5; descriptive and imprecise. "
                        "Missing confirmations retained as unsuccessful in the fixed five-seed denominator; continuous metrics label observed N. "
                        "Wilson interval is across training-seed successes, not pooled evaluation episodes.",
            "time_to_success_scope": "First scheduled selection success; time includes preceding setup, training and evaluations. "
                                     "Unobserved success is right-censored at attained control budget; final confirmation is a separate outcome.",
            "source_hashes": source_hashes, "summarizer_sha256": digest(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_dir, args.frozen_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.out_dir / "seeds.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["seeds"][0]))
        writer.writeheader()
        writer.writerows(report["seeds"])
    print(json.dumps({key: report[key] for key in ("confirmed_successes", "totals", "total_run_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
