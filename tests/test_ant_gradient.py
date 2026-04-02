"""Focused gradient diagnostics for ant forward locomotion.

Tests whether d(forward_vel)/d(ctrl) is non-zero and compares tape-all vs FD
Jacobian gradients for the ant at its default standing pose. This isolates the
physics gradient quality from the SHAC training loop.
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


class MinimalAntEnv:
    """Minimal ant environment for gradient testing."""

    def __init__(self, nworld=1, substeps=16, device='cuda:0',
                 use_fd_jacobian=False, tape_per_substep=False):
        self.device = device
        self.substeps = substeps
        self.nworld = nworld
        self._njmax = 512
        self.use_fd_jacobian = use_fd_jacobian
        self.tape_per_substep = tape_per_substep

        model_path = resolve_model_path('assets/ant.xml')
        self.mjm = mujoco.MjModel.from_xml_path(model_path)
        self.warp_model = mjw.put_model(self.mjm)

    def set_state(self, qpos_np, qvel_np):
        self.warp_data = mjw.make_diff_data(
            self.mjm, nworld=self.nworld, njmax=self._njmax,
        )
        mjw.reset_data(self.warp_model, self.warp_data)
        wp.copy(self.warp_data.qpos, wp.array(qpos_np.astype(np.float32), dtype=wp.float32))
        wp.copy(self.warp_data.qvel, wp.array(qvel_np.astype(np.float32), dtype=wp.float32))
        wp.synchronize()


def _ant_standing_state():
    """Return the ant's default standing qpos and qvel."""
    nq, nv = 15, 14
    qpos = np.zeros((1, nq), dtype=np.float64)
    qpos[0, :] = [0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    qvel = np.zeros((1, nv), dtype=np.float64)
    return qpos, qvel


def test_forward_vel_gradient_tape():
    """Test that d(forward_vel)/d(ctrl) is non-zero using tape-all backward.

    forward_vel = qvel_out[:, 0] is the primary locomotion reward signal.
    If its gradient w.r.t. ctrl is zero, the actor cannot learn to move forward.
    """
    print("=" * 70)
    print("Test: d(forward_vel)/d(ctrl) via tape-all backward")
    print("=" * 70)

    env = MinimalAntEnv(nworld=1, use_fd_jacobian=False)
    qpos0, qvel0 = _ant_standing_state()

    np.random.seed(42)
    ctrl_val = np.random.uniform(-0.3, 0.3, (1, 8))

    env.set_state(qpos0, qvel0)
    ctrl_torch = torch.tensor(ctrl_val, dtype=torch.float32, device='cuda:0', requires_grad=True)
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().detach().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().detach().requires_grad_(True)

    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos_in, qvel_in, env)

    # The locomotion-critical gradient: forward velocity
    forward_vel = qvel_out[:, 0]
    forward_vel.sum().backward()

    grad = ctrl_torch.grad.cpu().numpy().flatten()
    grad_qpos = qpos_in.grad.cpu().numpy().flatten() if qpos_in.grad is not None else np.zeros(15)
    grad_qvel = qvel_in.grad.cpu().numpy().flatten() if qvel_in.grad is not None else np.zeros(14)

    print(f"ctrl input:          {ctrl_val.flatten()}")
    print(f"forward_vel output:  {forward_vel.item():.8f}")
    print()
    print(f"d(fwd_vel)/d(ctrl):  {grad}")
    print(f"  |grad|:            {np.linalg.norm(grad):.8e}")
    print(f"  max |grad_i|:      {np.abs(grad).max():.8e}")
    print(f"  any NaN:           {np.any(np.isnan(grad))}")
    print()
    print(f"d(fwd_vel)/d(qpos):  norm={np.linalg.norm(grad_qpos):.8e}")
    print(f"d(fwd_vel)/d(qvel):  norm={np.linalg.norm(grad_qvel):.8e}")

    if np.abs(grad).max() < 1e-10:
        print("\nWARNING: tape-all gradient is effectively ZERO.")
        print("This means the physics backward is not differentiating ctrl -> forward_vel.")
    else:
        print(f"\nGradient is non-zero (max={np.abs(grad).max():.6e}). Tape-all backward works.")

    assert not np.any(np.isnan(grad)), "Gradient contains NaN"
    print("DONE\n")
    return grad


def test_forward_vel_gradient_fd():
    """Same test using FD Jacobian backward for comparison."""
    print("=" * 70)
    print("Test: d(forward_vel)/d(ctrl) via FD Jacobian backward")
    print("=" * 70)

    env = MinimalAntEnv(nworld=1, use_fd_jacobian=True)
    qpos0, qvel0 = _ant_standing_state()

    np.random.seed(42)
    ctrl_val = np.random.uniform(-0.3, 0.3, (1, 8))

    env.set_state(qpos0, qvel0)
    ctrl_torch = torch.tensor(ctrl_val, dtype=torch.float32, device='cuda:0', requires_grad=True)
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().detach().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().detach().requires_grad_(True)

    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos_in, qvel_in, env)

    forward_vel = qvel_out[:, 0]
    forward_vel.sum().backward()

    grad = ctrl_torch.grad.cpu().numpy().flatten()

    print(f"d(fwd_vel)/d(ctrl):  {grad}")
    print(f"  |grad|:            {np.linalg.norm(grad):.8e}")
    print(f"  max |grad_i|:      {np.abs(grad).max():.8e}")
    print(f"  any NaN:           {np.any(np.isnan(grad))}")

    if np.abs(grad).max() < 1e-10:
        print("\nWARNING: FD gradient is also ZERO. Issue is NOT tape-specific.")
    else:
        print(f"\nGradient is non-zero (max={np.abs(grad).max():.6e}). FD backward works.")

    assert not np.any(np.isnan(grad)), "Gradient contains NaN"
    print("DONE\n")
    return grad


def test_forward_vel_gradient_native_fd():
    """Ground truth: float64 central-difference FD using native MuJoCo."""
    print("=" * 70)
    print("Test: d(forward_vel)/d(ctrl) via native MuJoCo float64 FD")
    print("=" * 70)

    model_path = resolve_model_path('assets/ant.xml')
    mjm = mujoco.MjModel.from_xml_path(model_path)
    mjd = mujoco.MjData(mjm)

    qpos0, qvel0 = _ant_standing_state()
    np.random.seed(42)
    ctrl_val = np.random.uniform(-0.3, 0.3, (1, 8))

    eps = 1e-6
    fd_grad = np.zeros(8)
    for j in range(8):
        vals = []
        for sign in [1, -1]:
            mujoco.mj_resetData(mjm, mjd)
            mjd.qpos[:] = qpos0.flatten()
            mjd.qvel[:] = qvel0.flatten()
            mjd.ctrl[:] = ctrl_val.flatten()
            mjd.ctrl[j] += sign * eps
            for _ in range(16):
                mujoco.mj_step(mjm, mjd)
            vals.append(mjd.qvel[0])  # forward velocity
        fd_grad[j] = (vals[0] - vals[1]) / (2 * eps)

    print(f"d(fwd_vel)/d(ctrl):  {fd_grad}")
    print(f"  |grad|:            {np.linalg.norm(fd_grad):.8e}")
    print(f"  max |grad_i|:      {np.abs(fd_grad).max():.8e}")
    print("DONE\n")
    return fd_grad


def test_compare_all_modes():
    """Compare all three gradient modes and identify discrepancies."""
    print("=" * 70)
    print("COMPARISON: tape-all vs FD-bridge vs native-MuJoCo-FD")
    print("=" * 70)

    tape_grad = test_forward_vel_gradient_tape()
    fd_bridge_grad = test_forward_vel_gradient_fd()
    native_fd_grad = test_forward_vel_gradient_native_fd()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<20} {'|grad|':<14} {'max|g_i|':<14} {'Status'}")
    print("-" * 70)

    for name, g in [("tape-all", tape_grad), ("FD-bridge", fd_bridge_grad), ("native-MuJoCo-FD", native_fd_grad)]:
        norm = np.linalg.norm(g)
        maxg = np.abs(g).max()
        status = "OK" if maxg > 1e-6 else "ZERO"
        if np.any(np.isnan(g)):
            status = "NaN"
        print(f"{name:<20} {norm:<14.6e} {maxg:<14.6e} {status}")

    # Compare tape vs native FD (ground truth)
    nonzero = np.abs(native_fd_grad) > 1e-8
    if nonzero.any():
        tape_err = np.abs(tape_grad[nonzero] - native_fd_grad[nonzero]) / (np.abs(native_fd_grad[nonzero]) + 1e-10)
        fd_err = np.abs(fd_bridge_grad[nonzero] - native_fd_grad[nonzero]) / (np.abs(native_fd_grad[nonzero]) + 1e-10)
        print(f"\nRelative error vs native FD:")
        print(f"  tape-all:    max={tape_err.max():.4f}, mean={tape_err.mean():.4f}")
        print(f"  FD-bridge:   max={fd_err.max():.4f}, mean={fd_err.mean():.4f}")

    print()


def test_forward_vel_gradient_tape_per_substep():
    """Test state gradients using tape-per-substep (workaround for tape-all).

    tape-all produces zero d(fwd_vel)/d(qpos) and d(fwd_vel)/d(qvel) because
    Warp's tape doesn't chain the clone->overwrite sequence for d.qpos/d.qvel
    across multiple step() calls. tape-per-substep manually chains state
    gradients and should produce non-zero values.
    """
    print("=" * 70)
    print("Test: d(forward_vel)/d(ctrl,qpos,qvel) via tape-per-substep")
    print("=" * 70)

    env = MinimalAntEnv(nworld=1, use_fd_jacobian=False, tape_per_substep=True)
    qpos0, qvel0 = _ant_standing_state()

    np.random.seed(42)
    ctrl_val = np.random.uniform(-0.3, 0.3, (1, 8))

    env.set_state(qpos0, qvel0)
    ctrl_torch = torch.tensor(ctrl_val, dtype=torch.float32, device='cuda:0', requires_grad=True)
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().detach().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().detach().requires_grad_(True)

    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos_in, qvel_in, env)

    forward_vel = qvel_out[:, 0]
    forward_vel.sum().backward()

    grad = ctrl_torch.grad.cpu().numpy().flatten()
    grad_qpos = qpos_in.grad.cpu().numpy().flatten() if qpos_in.grad is not None else np.zeros(15)
    grad_qvel = qvel_in.grad.cpu().numpy().flatten() if qvel_in.grad is not None else np.zeros(14)

    print(f"d(fwd_vel)/d(ctrl):  {grad}")
    print(f"  |grad|:            {np.linalg.norm(grad):.8e}")
    print(f"  max |grad_i|:      {np.abs(grad).max():.8e}")
    print(f"  any NaN:           {np.any(np.isnan(grad))}")
    print()
    print(f"d(fwd_vel)/d(qpos):  norm={np.linalg.norm(grad_qpos):.8e}")
    print(f"d(fwd_vel)/d(qvel):  norm={np.linalg.norm(grad_qvel):.8e}")

    qpos_ok = np.linalg.norm(grad_qpos) > 1e-8
    qvel_ok = np.linalg.norm(grad_qvel) > 1e-8
    ctrl_ok = np.abs(grad).max() > 1e-8

    print()
    print(f"  ctrl gradient:     {'NON-ZERO' if ctrl_ok else 'ZERO (BROKEN)'}")
    print(f"  qpos gradient:     {'NON-ZERO — BPTT works!' if qpos_ok else 'ZERO — BPTT broken!'}")
    print(f"  qvel gradient:     {'NON-ZERO — BPTT works!' if qvel_ok else 'ZERO — BPTT broken!'}")

    assert not np.any(np.isnan(grad)), "Gradient contains NaN"
    print("DONE\n")
    return grad, grad_qpos, grad_qvel


if __name__ == '__main__':
    test_compare_all_modes()
    print("\n" + "=" * 70)
    print("TAPE-PER-SUBSTEP (workaround for tape-all state gradient issue)")
    print("=" * 70 + "\n")
    test_forward_vel_gradient_tape_per_substep()
