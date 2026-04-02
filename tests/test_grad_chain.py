"""Diagnose WHERE the gradient chain breaks in mjw.step() backward.

Tests each link in the chain:
  qvel.grad → qacc.grad → qacc_smooth.grad → qfrc_smooth.grad → qfrc_bias.grad → qpos.grad

Run: MJW_DEBUG_ADJOINT=1 python tests/test_grad_chain.py
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ["MJW_DEBUG_ADJOINT"] = "1"

import warp as wp
wp.init()

import torch
import numpy as np
import mujoco
import mujoco_warp as mjw

from msk_warp import resolve_model_path


def make_env():
    model_path = resolve_model_path('assets/ant.xml')
    mjm = mujoco.MjModel.from_xml_path(model_path)
    m = mjw.put_model(mjm)
    d = mjw.make_diff_data(mjm, nworld=1, njmax=512)
    mjw.reset_data(m, d)

    qpos = np.zeros((1, 15), dtype=np.float32)
    qpos[0, :] = [0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    wp.copy(d.qpos, wp.array(qpos, dtype=wp.float32))

    np.random.seed(42)
    ctrl = np.random.uniform(-0.3, 0.3, (1, 8)).astype(np.float32)
    wp.copy(d.ctrl, wp.array(ctrl, dtype=wp.float32))
    wp.synchronize()

    return mjm, m, d


def test_kinematics_only():
    """Test 1: d(xpos)/d(qpos) through kinematics only — no dynamics."""
    print("=" * 70)
    print("Test 1: d(xpos)/d(qpos) via kinematics")
    print("=" * 70)

    _, m, d = make_env()

    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    tape = wp.Tape()
    with tape:
        mjw.kinematics(m, d)
        mjw.com_pos(m, d)
        # Sum all xpos as loss
        wp.launch(
            _sum_array_kernel,
            dim=(d.nworld, m.nbody, 3),
            inputs=[d.xpos, loss],
        )
    tape.backward(loss=loss)

    qpos_grad = d.qpos.grad
    grad_norm = wp.to_torch(qpos_grad).norm().item() if qpos_grad is not None else 0.0
    print(f"  d(xpos)/d(qpos) |grad|: {grad_norm:.8e}")
    print(f"  Status: {'OK' if grad_norm > 1e-8 else 'BROKEN — kinematics adjoint fails'}")
    tape.zero()
    print()
    return grad_norm > 1e-8


def test_forward_only():
    """Test 2: d(qacc_smooth)/d(qpos) through forward() only — no integration."""
    print("=" * 70)
    print("Test 2: d(qacc_smooth)/d(qpos) via forward()")
    print("=" * 70)

    _, m, d = make_env()

    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    tape = wp.Tape()
    with tape:
        mjw.forward(m, d)
        # Sum qacc_smooth as loss
        wp.launch(
            _sum_2d_kernel,
            dim=(d.nworld, m.nv),
            inputs=[d.qacc_smooth, loss],
        )
    tape.backward(loss=loss)

    qpos_grad = d.qpos.grad
    grad_norm = wp.to_torch(qpos_grad).norm().item() if qpos_grad is not None else 0.0
    ctrl_grad_norm = wp.to_torch(d.ctrl.grad).norm().item() if d.ctrl.grad is not None else 0.0
    print(f"  d(qacc_smooth)/d(qpos) |grad|: {grad_norm:.8e}")
    print(f"  d(qacc_smooth)/d(ctrl) |grad|: {ctrl_grad_norm:.8e}")
    print(f"  qpos Status: {'OK' if grad_norm > 1e-8 else 'BROKEN — dynamics chain fails'}")
    print(f"  ctrl Status:  {'OK' if ctrl_grad_norm > 1e-8 else 'BROKEN'}")
    tape.zero()
    print()
    return grad_norm > 1e-8


def test_full_step():
    """Test 3: d(qvel_out)/d(qpos_in) through full step() — the actual training path."""
    print("=" * 70)
    print("Test 3: d(qvel_out)/d(qpos) via full step()")
    print("=" * 70)

    _, m, d = make_env()

    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    tape = wp.Tape()
    with tape:
        mjw.step(m, d)
        # forward_vel = qvel[0, 0]
        wp.launch(
            _sum_2d_kernel,
            dim=(d.nworld, 1),
            inputs=[d.qvel, loss],
        )
    tape.backward(loss=loss)

    qpos_grad = d.qpos.grad
    qvel_grad = d.qvel.grad
    ctrl_grad = d.ctrl.grad
    qpos_norm = wp.to_torch(qpos_grad).norm().item() if qpos_grad is not None else 0.0
    qvel_norm = wp.to_torch(qvel_grad).norm().item() if qvel_grad is not None else 0.0
    ctrl_norm = wp.to_torch(ctrl_grad).norm().item() if ctrl_grad is not None else 0.0
    print(f"  d(qvel_out)/d(qpos) |grad|: {qpos_norm:.8e}")
    print(f"  d(qvel_out)/d(qvel) |grad|: {qvel_norm:.8e}")
    print(f"  d(qvel_out)/d(ctrl) |grad|: {ctrl_norm:.8e}")
    print(f"  qpos Status: {'OK' if qpos_norm > 1e-8 else 'BROKEN — state gradient chain broken'}")
    print(f"  qvel Status: {'OK' if qvel_norm > 1e-8 else 'BROKEN'}")
    print(f"  ctrl Status: {'OK' if ctrl_norm > 1e-8 else 'BROKEN'}")
    tape.zero()
    print()
    return qpos_norm > 1e-8


@wp.kernel
def _sum_array_kernel(
    arr: wp.array2d(dtype=wp.vec3),
    loss: wp.array(dtype=float),
):
    worldid, bodyid = wp.tid()
    v = arr[worldid, bodyid]
    wp.atomic_add(loss, 0, v[0] + v[1] + v[2])


@wp.kernel
def _sum_2d_kernel(
    arr: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
):
    worldid, dofid = wp.tid()
    wp.atomic_add(loss, 0, arr[worldid, dofid])


def test_bridge_pattern():
    """Test 4: Reproduce the EXACT bridge.py pattern to find what breaks it.

    Bridge does: forward (no tape) → restore state → tape substeps → backward.
    Compare against fresh-data pattern that works.
    """
    print("=" * 70)
    print("Test 4: Bridge pattern — forward then restore then tape")
    print("=" * 70)

    _, m, d = make_env()

    # Save initial state (like bridge forward does)
    saved_qpos = wp.clone(d.qpos)
    saved_qvel = wp.clone(d.qvel)
    saved_time = wp.clone(d.time)

    # Run forward pass WITHOUT tape (like bridge forward)
    for _ in range(16):
        mjw.step(m, d)
    wp.synchronize()

    print(f"  After forward: d.qpos.requires_grad = {d.qpos.requires_grad}")
    print(f"  After forward: d.qpos.grad = {d.qpos.grad}")

    # Restore state (like bridge backward does)
    wp.copy(d.qpos, saved_qpos)
    wp.copy(d.qvel, saved_qvel)
    wp.copy(d.time, saved_time)
    wp.synchronize()

    print(f"  After restore: d.qpos.requires_grad = {d.qpos.requires_grad}")
    print(f"  After restore: d.qpos.grad = {d.qpos.grad}")

    # --- Pattern A: Bridge pattern (with grad zeroing removed) ---
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    tape = wp.Tape()
    with tape:
        mjw.step(m, d)
        wp.launch(_sum_2d_kernel, dim=(d.nworld, 1), inputs=[d.qvel, loss])
    tape.backward(loss=loss)

    qpos_norm = wp.to_torch(d.qpos.grad).norm().item() if d.qpos.grad is not None else 0.0
    qvel_norm = wp.to_torch(d.qvel.grad).norm().item() if d.qvel.grad is not None else 0.0
    ctrl_norm = wp.to_torch(d.ctrl.grad).norm().item() if d.ctrl.grad is not None else 0.0

    print(f"\n  Pattern A (bridge: forward→restore→tape):")
    print(f"    d(qvel)/d(qpos) |grad|: {qpos_norm:.8e} {'OK' if qpos_norm > 1e-8 else 'BROKEN'}")
    print(f"    d(qvel)/d(qvel) |grad|: {qvel_norm:.8e} {'OK' if qvel_norm > 1e-8 else 'BROKEN'}")
    print(f"    d(qvel)/d(ctrl) |grad|: {ctrl_norm:.8e} {'OK' if ctrl_norm > 1e-8 else 'BROKEN'}")
    tape.zero()

    # --- Pattern B: Fresh data (what works in test 3) ---
    _, m2, d2 = make_env()

    loss2 = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    tape2 = wp.Tape()
    with tape2:
        mjw.step(m2, d2)
        wp.launch(_sum_2d_kernel, dim=(d2.nworld, 1), inputs=[d2.qvel, loss2])
    tape2.backward(loss=loss2)

    qpos_norm2 = wp.to_torch(d2.qpos.grad).norm().item() if d2.qpos.grad is not None else 0.0

    print(f"\n  Pattern B (fresh data, no forward first):")
    print(f"    d(qvel)/d(qpos) |grad|: {qpos_norm2:.8e} {'OK' if qpos_norm2 > 1e-8 else 'BROKEN'}")
    tape2.zero()

    # --- Pattern C: Recreate data between forward and backward ---
    mjm3, m3, d3 = make_env()

    # Forward (no tape)
    for _ in range(16):
        mjw.step(m3, d3)
    wp.synchronize()

    # Recreate data fresh (instead of restoring via wp.copy)
    d3 = mjw.make_diff_data(mjm3, nworld=1, njmax=512)
    mjw.reset_data(m3, d3)
    qpos_np = np.zeros((1, 15), dtype=np.float32)
    qpos_np[0, :] = [0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    wp.copy(d3.qpos, wp.array(qpos_np, dtype=wp.float32))
    np.random.seed(42)
    ctrl_np = np.random.uniform(-0.3, 0.3, (1, 8)).astype(np.float32)
    wp.copy(d3.ctrl, wp.array(ctrl_np, dtype=wp.float32))
    wp.synchronize()

    loss3 = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    tape3 = wp.Tape()
    with tape3:
        mjw.step(m3, d3)
        wp.launch(_sum_2d_kernel, dim=(d3.nworld, 1), inputs=[d3.qvel, loss3])
    tape3.backward(loss=loss3)

    qpos_norm3 = wp.to_torch(d3.qpos.grad).norm().item() if d3.qpos.grad is not None else 0.0

    print(f"\n  Pattern C (recreate data after forward):")
    print(f"    d(qvel)/d(qpos) |grad|: {qpos_norm3:.8e} {'OK' if qpos_norm3 > 1e-8 else 'BROKEN'}")
    tape3.zero()

    print()
    return qpos_norm > 1e-8


if __name__ == '__main__':
    print()
    k_ok = test_kinematics_only()
    f_ok = test_forward_only()
    s_ok = test_full_step()
    b_ok = test_bridge_pattern()

    print("=" * 70)
    print("CHAIN DIAGNOSIS")
    print("=" * 70)
    if not k_ok:
        print("  BROKEN at: kinematics — Warp auto-adjoint for kinematics kernels fails")
    elif not f_ok:
        print("  BROKEN at: forward() dynamics — chain from qacc_smooth back to qpos fails")
        print("  Likely: rne/com_vel/crb backward or mass matrix adjoint")
    elif not s_ok:
        print("  BROKEN at: step() integration — _advance() clone->overwrite chain fails")
        print("  The dynamics produce gradients but the integrator doesn't chain them to input state")
    else:
        print("  ALL OK — full gradient chain works!")
    print()
