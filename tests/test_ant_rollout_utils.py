"""Unit tests for shared Ant rollout snapshot helpers."""

from __future__ import annotations

from msk_warp.utils.ant_rollout import normalize_snapshot_steps


def test_normalize_snapshot_steps_sorts_deduplicates_and_filters_negative():
    steps = normalize_snapshot_steps([32, 0, 16, 16, -4])
    assert steps == [0, 16, 32]


def test_normalize_snapshot_steps_uses_fallback_when_empty():
    steps = normalize_snapshot_steps([], fallback_step=64)
    assert steps == [64]
