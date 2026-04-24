"""Tests for Warp-vs-native MuJoCo contact-state parity diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "diag_contact_state_parity.py"
    spec = importlib.util.spec_from_file_location("diag_contact_state_parity", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_distribution_l1_basic():
    mod = _load_module()
    a = {"QUADRATIC": 2, "SATISFIED": 2}
    b = {"QUADRATIC": 4}
    l1 = mod.distribution_l1(a, b)
    assert abs(l1 - 1.0) < 1e-9


def test_native_state_view_prefers_island_state():
    mod = _load_module()

    class DummyData:
        nefc = 4
        efc_state = np.array([0, 0, 0, 0], dtype=np.int32)
        iefc_state = np.array([1, 1, 1, 1], dtype=np.int32)

    state, source = mod._native_efc_state_view(DummyData())
    assert source == "iefc_state"
    assert int(state[0]) == 1


def test_mismatch_detector_flags_quadratic_skew():
    mod = _load_module()
    warp = {
        "constraint_contacts": 25,
        "active_contacts": 25,
        "row_states": {"QUADRATIC": 100},
        "normal_states": {"QUADRATIC": 25},
        "friction_states": {"QUADRATIC": 75},
        "friction_to_normal_force_ratio": 1.0,
    }
    native = {
        "constraint_contacts": 25,
        "active_contacts": 20,
        "row_states": {"QUADRATIC": 35, "SATISFIED": 5, "LINEARNEG": 30, "LINEARPOS": 30},
        "normal_states": {"QUADRATIC": 20, "SATISFIED": 5},
        "friction_states": {"QUADRATIC": 15, "LINEARNEG": 30, "LINEARPOS": 30},
        "friction_to_normal_force_ratio": 0.25,
    }
    verdict = mod.detect_contact_parity_mismatches(
        warp,
        native,
        max_contact_delta=2,
        max_state_l1=0.2,
        max_force_ratio_factor=2.0,
    )
    assert verdict["ok"] is False
    assert verdict["mismatch_count"] >= 2
    assert any("friction_state_l1" in msg for msg in verdict["mismatches"])


def test_one_step_state_metrics_capture_delta_error():
    mod = _load_module()
    qpos0 = np.array([[1.0, 2.0]], dtype=np.float64)
    qvel0 = np.array([[0.5, -0.5]], dtype=np.float64)
    warp_qpos1 = np.array([[1.2, 2.3]], dtype=np.float64)
    warp_qvel1 = np.array([[0.8, -0.4]], dtype=np.float64)
    native_qpos1 = np.array([[1.1, 2.1]], dtype=np.float64)
    native_qvel1 = np.array([[0.7, -0.1]], dtype=np.float64)

    metrics = mod.compute_one_step_state_metrics(
        qpos0, qvel0, warp_qpos1, warp_qvel1, native_qpos1, native_qvel1
    )

    assert metrics["qpos_delta_l1_sum"] == pytest.approx(0.3)
    assert metrics["qvel_delta_l1_sum"] == pytest.approx(0.4)
    assert metrics["state_delta_l1_sum"] == pytest.approx(0.7)
    assert metrics["state_delta_l1_mean"] == pytest.approx(0.175)
    assert metrics["state_delta_linf"] == pytest.approx(0.3)


def test_step_state_mismatch_detector_threshold():
    mod = _load_module()
    verdict = mod.detect_step_state_mismatches(
        {"state_delta_l1_mean": 0.03, "state_delta_linf": 0.04},
        max_step_state_delta_mean=0.02,
    )
    assert verdict["ok"] is False
    assert verdict["mismatch_count"] == 1
    assert "one_step_state_delta_l1_mean" in verdict["mismatches"][0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Warp parity diagnostic")
def test_contact_state_parity_smoke_report():
    mod = _load_module()
    report = mod.run_contact_state_parity(
        cfg_path="msk_warp/configs/experiments/ant_soft_surrogate.yaml",
        device="cuda:0",
        settle_steps=8,
        ctrl_val=0.2,
        rollout_cfg=None,
        policy_path=None,
        rollout_steps=8,
        max_contact_delta=50,
        max_state_l1=1.0,
        max_force_ratio_factor=100.0,
        max_step_state_delta_mean=1.0,
    )

    assert report["state"]["source"] == "settled"
    assert "warp" in report and "native" in report and "parity" in report
    assert "one_step" in report and "overall" in report
    assert report["warp"]["summary"]["constraint_contacts"] >= 0
    assert report["native"]["summary"]["constraint_contacts"] >= 0
    assert isinstance(report["parity"]["mismatches"], list)
    assert isinstance(report["one_step"]["state_delta"]["metrics"]["state_delta_l1_mean"], float)
