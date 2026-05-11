"""Shared Ant training metrics helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
    _HAS_TENSORBOARD = True
except ImportError:
    event_accumulator = None
    _HAS_TENSORBOARD = False


def read_grad_metrics(
    logdir: str | Path,
    grad_norm_target: float | None = None,
) -> dict[str, Any]:
    """Read actor gradient clipping metrics from a TensorBoard log directory."""
    if not _HAS_TENSORBOARD:
        return {}
    event_dir = Path(logdir) / "log"
    if not event_dir.exists():
        return {}

    try:
        ea = event_accumulator.EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        ea.Reload()
    except Exception:
        return {}

    tags = ea.Tags().get("scalars", [])
    if "grad_norm/before_clip" not in tags or "grad_norm/after_clip" not in tags:
        return {}

    before = np.array([s.value for s in ea.Scalars("grad_norm/before_clip")], dtype=np.float64)
    after = np.array([s.value for s in ea.Scalars("grad_norm/after_clip")], dtype=np.float64)
    if "grad_norm/actor_clip_threshold" in tags:
        clip_target = np.array(
            [s.value for s in ea.Scalars("grad_norm/actor_clip_threshold")],
            dtype=np.float64,
        )
        n = int(min(len(before), len(after), len(clip_target)))
        if n <= 0:
            return {}
        before = before[:n]
        after = after[:n]
        clip_target = clip_target[:n]
    elif grad_norm_target is not None:
        clip_target = np.full_like(after, float(grad_norm_target), dtype=np.float64)
    else:
        clip_target = np.full_like(after, np.nan, dtype=np.float64)

    if before.size == 0 or after.size == 0:
        return {}

    clip_hits = np.zeros_like(after, dtype=bool)
    if not np.isnan(clip_target).all():
        tol = np.maximum(1e-3, 1e-3 * np.maximum(1.0, np.abs(clip_target)))
        clip_hits = np.abs(after - clip_target) <= tol

    compression = before / np.maximum(after, 1e-12)
    return {
        "epochs_logged": int(len(after)),
        "grad_before_min": float(np.min(before)),
        "grad_before_median": float(np.median(before)),
        "grad_before_max": float(np.max(before)),
        "grad_after_min": float(np.min(after)),
        "grad_after_median": float(np.median(after)),
        "grad_after_max": float(np.max(after)),
        "clip_target_min": float(np.nanmin(clip_target)),
        "clip_target_median": float(np.nanmedian(clip_target)),
        "clip_target_max": float(np.nanmax(clip_target)),
        "clip_hit_rate": float(np.mean(clip_hits)),
        "compression_median": float(np.median(compression)),
    }
