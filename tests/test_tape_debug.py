"""Debug why tape-all backward produces zero gradients.

Checks whether d.qpos.requires_grad is True and whether the adjoint
solver_implicit_adjoint actually fires during tape.backward().
"""

import warnings
warnings.filterwarnings('ignore')

import warp as wp
wp.init()

import torch
import numpy as np
import mujoco
import mujoco_warp as mjw

from msk_warp import resolve_model_path
from msk_warp.bridge import WarpSimStep


def test_requires_grad_state():
    """Check d.qpos.requires_grad at each stage."""
    model_path = resolve_model_path('assets/ant.xml')
    mjm = mujoco.MjModel.from_xml_path(model_path)
    warp_model = mjw.put_model(mjm)

    # Stage 1: make_diff_data
    d = mjw.make_diff_data(mjm, nworld=1, njmax=512)
    mjw.reset_data(warp_model, d)
    print(f"After make_diff_data:  d.qpos.requires_grad = {d.qpos.requires_grad}")

    # Stage 2: copy state (simulating bridge forward)
    qpos_np = np.array([[0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]], dtype=np.float32)
    qvel_np = np.zeros((1, 14), dtype=np.float32)
    wp.copy(d.qpos, wp.array(qpos_np, dtype=wp.float32))
    wp.copy(d.qvel, wp.array(qvel_np, dtype=wp.float32))
    print(f"After wp.copy state:   d.qpos.requires_grad = {d.qpos.requires_grad}")

    # Stage 3: set ctrl
    ctrl_np = np.random.uniform(-0.3, 0.3, (1, 8)).astype(np.float32)
    wp.copy(d.ctrl, wp.array(ctrl_np, dtype=wp.float32))
    print(f"After wp.copy ctrl:    d.qpos.requires_grad = {d.qpos.requires_grad}")

    # Stage 4: run step WITHOUT tape (like bridge forward does)
    for _ in range(16):
        mjw.step(warp_model, d)
    wp.synchronize()
    print(f"After 16 steps (no tape): d.qpos.requires_grad = {d.qpos.requires_grad}")

    # Stage 5: restore state and run WITH tape (like bridge backward does)
    wp.copy(d.qpos, wp.array(qpos_np, dtype=wp.float32))
    wp.copy(d.qvel, wp.array(qvel_np, dtype=wp.float32))
    wp.copy(d.ctrl, wp.array(ctrl_np, dtype=wp.float32))
    print(f"After restore:         d.qpos.requires_grad = {d.qpos.requires_grad}")

    return d, warp_model


def test_adjoint_fires():
    """Check if solver_implicit_adjoint actually runs during tape backward."""
    import mujoco_warp._src.adjoint as adjoint_mod

    # Monkey-patch to detect if adjoint fires
    original_fn = adjoint_mod.solver_implicit_adjoint
    adjoint_called = [False]

    def patched_adjoint(m, d):
        adjoint_called[0] = True
        print("  >>> solver_implicit_adjoint CALLED <<<")
        return original_fn(m, d)

    adjoint_mod.solver_implicit_adjoint = patched_adjoint

    try:
        model_path = resolve_model_path('assets/ant.xml')
        mjm = mujoco.MjModel.from_xml_path(model_path)
        warp_model = mjw.put_model(mjm)
        d = mjw.make_diff_data(mjm, nworld=1, njmax=512)
        mjw.reset_data(warp_model, d)

        qpos_np = np.array([[0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]], dtype=np.float32)
        qvel_np = np.zeros((1, 14), dtype=np.float32)
        ctrl_np = np.random.uniform(-0.3, 0.3, (1, 8)).astype(np.float32)
        wp.copy(d.qpos, wp.array(qpos_np, dtype=wp.float32))
        wp.copy(d.qvel, wp.array(qvel_np, dtype=wp.float32))
        wp.copy(d.ctrl, wp.array(ctrl_np, dtype=wp.float32))
        wp.synchronize()

        print(f"\nd.qpos.requires_grad = {d.qpos.requires_grad}")

        # Zero grad fields
        d.qpos.grad = wp.zeros_like(d.qpos)
        d.qvel.grad = wp.zeros_like(d.qvel)
        d.ctrl.grad = wp.zeros_like(d.ctrl)

        # Tape over 16 substeps (like bridge backward tape-all)
        from msk_warp.bridge import _vjp_state_kernel
        nq, nv = 15, 14
        grad_qpos = wp.array(np.zeros((1, nq), dtype=np.float32), dtype=wp.float32)
        grad_qvel = wp.array(np.zeros((1, nv), dtype=np.float32), dtype=wp.float32)
        # Set forward_vel gradient to 1.0
        grad_qvel_np = np.zeros((1, nv), dtype=np.float32)
        grad_qvel_np[0, 0] = 1.0  # d(loss)/d(forward_vel) = 1
        wp.copy(grad_qvel, wp.array(grad_qvel_np, dtype=wp.float32))

        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        tape = wp.Tape()
        print("Running tape forward (16 substeps)...")
        with tape:
            for _ in range(16):
                mjw.step(warp_model, d)
            wp.launch(
                _vjp_state_kernel,
                dim=(1, max(nq, nv)),
                inputs=[d.qpos, d.qvel, grad_qpos, grad_qvel, loss],
            )

        print(f"Running tape.backward()...")
        tape.backward(loss=loss)
        wp.synchronize()

        ctrl_grad = wp.to_torch(d.ctrl.grad).detach().cpu().numpy().flatten()
        qpos_grad = wp.to_torch(d.qpos.grad).detach().cpu().numpy().flatten()
        qvel_grad = wp.to_torch(d.qvel.grad).detach().cpu().numpy().flatten()

        print(f"\nResults:")
        print(f"  adjoint called:   {adjoint_called[0]}")
        print(f"  |ctrl.grad|:      {np.linalg.norm(ctrl_grad):.8e}")
        print(f"  |qpos.grad|:      {np.linalg.norm(qpos_grad):.8e}")
        print(f"  |qvel.grad|:      {np.linalg.norm(qvel_grad):.8e}")
        print(f"  ctrl.grad:        {ctrl_grad}")

        tape.zero()

    finally:
        adjoint_mod.solver_implicit_adjoint = original_fn


if __name__ == '__main__':
    print("=" * 60)
    print("Test 1: requires_grad state tracking")
    print("=" * 60)
    test_requires_grad_state()

    print("\n" + "=" * 60)
    print("Test 2: Does solver_implicit_adjoint fire?")
    print("=" * 60)
    test_adjoint_fires()
