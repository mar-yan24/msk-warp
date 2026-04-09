"""Unit tests for SHAC gradient control helpers."""

from __future__ import annotations

import torch

from msk_warp.algorithms.shac import apply_state_grad_control
from msk_warp.algorithms.shac import compute_actor_clip_threshold


def test_apply_state_grad_control_decay_only():
    grad = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    out = apply_state_grad_control(grad, decay=0.5, clip=0.0)
    assert torch.allclose(out, torch.tensor([[1.5, 2.0]], dtype=torch.float32))


def test_apply_state_grad_control_clip_only():
    grad = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    out = apply_state_grad_control(grad, decay=0.0, clip=2.0)
    assert torch.allclose(out, torch.tensor([[1.2, 1.6]], dtype=torch.float32), atol=1e-6)


def test_apply_state_grad_control_decay_then_clip():
    grad = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    out = apply_state_grad_control(grad, decay=0.5, clip=2.0)
    # [3,4] -> decay -> [1.5,2.0], norm=2.5; then scale by 0.8
    assert torch.allclose(out, torch.tensor([[1.2, 1.6]], dtype=torch.float32), atol=1e-6)


def test_compute_actor_clip_threshold_warmup():
    assert compute_actor_clip_threshold(
        epoch=0, target_clip=40.0, init_clip=200.0, warmup_epochs=20
    ) == 200.0
    assert compute_actor_clip_threshold(
        epoch=10, target_clip=40.0, init_clip=200.0, warmup_epochs=20
    ) == 120.0
    assert compute_actor_clip_threshold(
        epoch=20, target_clip=40.0, init_clip=200.0, warmup_epochs=20
    ) == 40.0


def test_compute_actor_clip_threshold_no_schedule():
    assert compute_actor_clip_threshold(
        epoch=5, target_clip=40.0, init_clip=None, warmup_epochs=20
    ) == 40.0
    assert compute_actor_clip_threshold(
        epoch=5, target_clip=40.0, init_clip=200.0, warmup_epochs=0
    ) == 40.0
