"""Unit tests for reward gradient diagnostic helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "diag_reward_gradient_variants.py"
    spec = importlib.util.spec_from_file_location("diag_reward_gradient_variants", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_probe_vector_is_normalized():
    mod = _load_module()
    probe = mod.make_probe_vector(8, device="cpu", dtype=torch.float32)
    assert probe.shape == (8,)
    assert torch.linalg.norm(probe).item() == pytest.approx(1.0, rel=1e-6)


def test_summarize_gradient_match_reports_expected_stats():
    mod = _load_module()
    tape = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    fd = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    report = mod.summarize_gradient_match(tape, fd)

    assert report["tape_norm"] == pytest.approx(np.sqrt(0.5))
    assert report["fd_norm"] == pytest.approx(np.sqrt(2.0))
    assert report["ratio"] == pytest.approx(0.5)
    assert report["cosine"] == pytest.approx(1.0)
    assert len(report["per_env"]) == 2
