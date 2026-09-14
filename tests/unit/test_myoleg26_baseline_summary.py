"""Seed-level accounting and censoring must survive missing confirmation data."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import summarize_myoleg26_baseline as summary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(tmp_path):
    run, frozen_path = tmp_path / "run", tmp_path / "freeze.json"
    frozen = {"protocol": {"seeds": list(range(5))}}
    write(frozen_path, frozen)
    write(run / "run_manifest.json", {"freeze": frozen, "freeze_sha256": summary.digest(frozen_path),
          "scope": "registered_full_budget", "seeds": list(range(5))})
    results = []
    for seed in range(5):
        censored = seed == 4
        controls, evaluation = (524288 if censored else 1048576), (0 if censored else 16000)
        seed_dir = run / f"seed_{seed}"
        seed_dir.mkdir()
        checkpoint = seed_dir / "epoch_0128.pt"
        checkpoint.write_bytes(f"seed{seed}".encode())
        selected = {"epoch": 128, "filename": checkpoint.name, "sha256": summary.digest(checkpoint)}
        success = seed < 2
        confirmations = [] if censored else [{"kind": "confirmation", "complete": True, "success": success,
            "checkpoint_sha256": selected["sha256"], "episodes": [{"survived": success}] * 32,
            "summary": {"mean_forward_velocity_mps": 1. if success else -.5,
                        "mean_first_episode_duration_s": 4. if success else .8,
                        "mean_displacement_m": 4. if success else -.4,
                        "mean_episode_velocity_2d_rmse_mps": .1 if success else 1.6}}]
        result = {"seed": seed, "completed_epochs": 64 if censored else 128,
            "censored": censored, "stop_reason": "seed_wall_cap" if censored else "epoch_budget",
            "selected_checkpoint": None if censored else selected, "evaluations": confirmations,
            "training_completed_control_calls": controls // 64,
            "timing_seconds": {"total_seed_wall": 1000., "training_rollout": 800., "training_update": 50., "evaluation": 100.},
            "accounting": {"training_control_transitions": controls, "evaluation_control_transitions": evaluation,
                "training_physics_steps": 4*controls, "evaluation_physics_steps": 4*evaluation,
                "total_control_transitions": controls+evaluation, "total_physics_steps": 4*(controls+evaluation)}}
        write(seed_dir / "result.json", result)
        results.append(result)
    write(run / "results.json", {"seeds": results, "total_run_wall_seconds": 5010.})
    return run, frozen_path


def test_missing_confirmation_retains_five_seed_success_denominator(tmp_path):
    run, frozen = fixture_run(tmp_path)
    report = summary.summarize(run, frozen)
    assert report["registered_seed_count"] == 5 and report["confirmed_successes"] == 2
    assert report["success_fraction"] == .4
    assert report["missing_confirmation_count"] == 1
    assert report["seeds"][4]["mean_forward_velocity_mps"] is None
    assert report["seeds"][4]["success_not_observed_by_controls"] == 524288
    assert report["seed_metric_summaries"]["mean_forward_velocity_mps"]["n_training_seeds"] == 4
    assert report["seed_metric_summaries"]["mean_forward_velocity_mps"]["ci95"] is None
    assert report["totals"]["training_control_transitions"] == 4*1048576+524288
    assert report["totals"]["evaluation_physics_steps"] == 4*4*16000


def test_summary_rejects_changed_checkpoint_and_phase_accounting(tmp_path):
    run, frozen = fixture_run(tmp_path)
    path = run / "seed_0/epoch_0128.pt"
    path.write_bytes(b"other checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        summary.summarize(run, frozen)
    path.write_bytes(b"seed0")
    combined = json.loads((run / "results.json").read_text())
    combined["seeds"][0]["accounting"]["training_physics_steps"] += 1
    write(run / "seed_0/result.json", combined["seeds"][0])
    write(run / "results.json", combined)
    with pytest.raises(ValueError, match="Physics/control"):
        summary.summarize(run, frozen)


def test_intervals_use_training_seeds_and_do_not_invent_precision():
    assert summary.mean_interval([])["mean"] is None
    assert summary.mean_interval([1, 2, 3, 4])["ci95"] is None
    report = summary.mean_interval([2]*5)
    assert report["n_training_seeds"] == 5 and report["ci95"] == [2., 2.]
    low = summary.wilson_interval(0, 5)
    high = summary.wilson_interval(5, 5)
    np.testing.assert_allclose(low, [0., .43448246478317476], atol=1e-12)
    np.testing.assert_allclose(high, [1-low[1], 1.], atol=1e-12)
    with pytest.raises(ValueError):
        summary.wilson_interval(6, 5)
