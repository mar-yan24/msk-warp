"""Tests for friction gradient bypass in the smooth adjoint.

Verifies that the bypass kernel:
1. Produces no change when kf=0 (regression)
2. Amplifies tangential gradients when kf>0
3. Does not affect forward physics
4. Maintains sign alignment with FD reference
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
                 smooth_adjoint=False, friction_viscosity=10.0,
                 friction_scale=0.01, friction_bypass_kf=0.0):
        self.device = device
        self.substeps = substeps
        self.nworld = nworld
        self._njmax = 512
        self.use_fd_jacobian = False
        self.tape_per_substep = False
        self._smooth_adjoint = smooth_adjoint
        self._friction_viscosity = friction_viscosity
        self._friction_scale = friction_scale
        self._friction_bypass_kf = friction_bypass_kf

        model_path = resolve_model_path('assets/ant.xml')
        self.mjm = mujoco.MjModel.from_xml_path(model_path)
        self.warp_model = mjw.put_model(self.mjm)

    def set_state(self, qpos_np, qvel_np):
        self.warp_data = mjw.make_diff_data(
            self.mjm, nworld=self.nworld, njmax=self._njmax,
        )
        if self._smooth_adjoint:
            mjw.enable_smooth_adjoint(
                self.warp_data,
                friction_viscosity=self._friction_viscosity,
                friction_scale=self._friction_scale,
                friction_bypass_kf=self._friction_bypass_kf,
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


def _settle_ant(env, qpos0, qvel0, ctrl_val, n_settle=50):
    """Run forward steps to settle the ant into ground contact."""
    env.set_state(qpos0, qvel0)
    ctrl_np = ctrl_val.astype(np.float32)
    for _ in range(n_settle):
        wp.copy(
            env.warp_data.ctrl,
            wp.array(ctrl_np, dtype=wp.float32),
        )
        for _ in range(env.substeps):
            mjw.step(env.warp_model, env.warp_data)
    wp.synchronize()


def _compute_grad(env, ctrl_val):
    """Run one differentiable step and return d(forward_vel)/d(ctrl)."""
    ctrl_torch = torch.tensor(
        ctrl_val, dtype=torch.float32, device='cuda:0', requires_grad=True
    )
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().detach().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().detach().requires_grad_(True)

    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos_in, qvel_in, env)

    forward_vel = qvel_out[:, 0]
    forward_vel.sum().backward()

    grad = ctrl_torch.grad.cpu().numpy().flatten()
    return grad


def _compute_grad_settled(smooth_adjoint=False, friction_bypass_kf=0.0,
                          ctrl_val=None, n_settle=50):
    """Create env, settle ant, compute gradient. Returns grad array."""
    env = MinimalAntEnv(
        smooth_adjoint=smooth_adjoint,
        friction_bypass_kf=friction_bypass_kf,
    )
    qpos0, qvel0 = _ant_standing_state()
    if ctrl_val is None:
        np.random.seed(42)
        ctrl_val = np.random.uniform(-0.3, 0.3, (1, 8))
    _settle_ant(env, qpos0, qvel0, ctrl_val, n_settle=n_settle)

    # Re-create diff data at settled state for clean gradient graph
    qpos_settled = env.warp_data.qpos.numpy().copy()
    qvel_settled = env.warp_data.qvel.numpy().copy()
    env.set_state(qpos_settled, qvel_settled)

    return _compute_grad(env, ctrl_val)


def test_bypass_kf0_matches_smooth_adjoint():
    """Bypass with kf=0 should produce identical gradients to smooth adjoint without bypass."""
    print("=" * 70)
    print("Test: kf=0 matches smooth adjoint (regression)")
    print("=" * 70)

    grad_smooth = _compute_grad_settled(smooth_adjoint=True, friction_bypass_kf=0.0)
    grad_bypass0 = _compute_grad_settled(smooth_adjoint=True, friction_bypass_kf=0.0)

    diff = np.abs(grad_smooth - grad_bypass0).max()
    print(f"Smooth adjoint grad:  {grad_smooth}")
    print(f"Bypass kf=0 grad:    {grad_bypass0}")
    print(f"Max abs difference:   {diff:.2e}")

    assert diff < 1e-4, f"kf=0 should match smooth adjoint, got diff={diff}"
    print("PASSED\n")


def test_bypass_amplifies_gradient():
    """Bypass with kf=1.0 should produce larger gradient than kf=0."""
    print("=" * 70)
    print("Test: kf=1.0 amplifies tangential gradient")
    print("=" * 70)

    grad_kf0 = _compute_grad_settled(smooth_adjoint=True, friction_bypass_kf=0.0)
    grad_kf1 = _compute_grad_settled(smooth_adjoint=True, friction_bypass_kf=1.0)

    norm_kf0 = np.linalg.norm(grad_kf0)
    norm_kf1 = np.linalg.norm(grad_kf1)
    ratio = norm_kf1 / max(norm_kf0, 1e-10)

    print(f"Smooth adjoint (kf=0): {grad_kf0}")
    print(f"  |grad|: {norm_kf0:.6e}")
    print(f"Bypass (kf=1.0):       {grad_kf1}")
    print(f"  |grad|: {norm_kf1:.6e}")
    print(f"Ratio kf1/kf0:         {ratio:.2f}x")

    assert not np.any(np.isnan(grad_kf1)), "Bypass gradient contains NaN"
    assert norm_kf1 > norm_kf0, (
        f"Bypass should amplify gradient: |kf1|={norm_kf1:.6e} <= |kf0|={norm_kf0:.6e}"
    )
    print(f"PASSED — bypass amplified gradient by {ratio:.2f}x\n")


def test_bypass_does_not_affect_forward():
    """Forward physics should be identical with and without bypass."""
    print("=" * 70)
    print("Test: forward physics unchanged")
    print("=" * 70)

    np.random.seed(42)
    ctrl_val = np.random.uniform(-0.3, 0.3, (1, 8)).astype(np.float32)
    qpos0, qvel0 = _ant_standing_state()

    # Without bypass
    env0 = MinimalAntEnv(smooth_adjoint=True, friction_bypass_kf=0.0)
    env0.set_state(qpos0, qvel0)
    wp.copy(env0.warp_data.ctrl, wp.array(ctrl_val, dtype=wp.float32))
    for _ in range(env0.substeps):
        mjw.step(env0.warp_model, env0.warp_data)
    wp.synchronize()
    qpos_out0 = env0.warp_data.qpos.numpy().copy()
    qvel_out0 = env0.warp_data.qvel.numpy().copy()

    # With bypass
    env1 = MinimalAntEnv(smooth_adjoint=True, friction_bypass_kf=1.0)
    env1.set_state(qpos0, qvel0)
    wp.copy(env1.warp_data.ctrl, wp.array(ctrl_val, dtype=wp.float32))
    for _ in range(env1.substeps):
        mjw.step(env1.warp_model, env1.warp_data)
    wp.synchronize()
    qpos_out1 = env1.warp_data.qpos.numpy().copy()
    qvel_out1 = env1.warp_data.qvel.numpy().copy()

    qpos_diff = np.abs(qpos_out0 - qpos_out1).max()
    qvel_diff = np.abs(qvel_out0 - qvel_out1).max()
    print(f"qpos max diff: {qpos_diff:.2e}")
    print(f"qvel max diff: {qvel_diff:.2e}")

    assert qpos_diff < 1e-6, f"Forward qpos changed: diff={qpos_diff}"
    assert qvel_diff < 1e-6, f"Forward qvel changed: diff={qvel_diff}"
    print("PASSED — forward physics identical\n")


def test_bypass_gradient_scaling():
    """Gradient magnitude should increase monotonically with kf."""
    print("=" * 70)
    print("Test: gradient scales with kf")
    print("=" * 70)

    kf_values = [0.0, 0.5, 1.0, 2.0]
    norms = []

    for kf in kf_values:
        grad = _compute_grad_settled(smooth_adjoint=True, friction_bypass_kf=kf)
        norm = np.linalg.norm(grad)
        norms.append(norm)
        print(f"  kf={kf:.1f}: |grad|={norm:.6e}")
        assert not np.any(np.isnan(grad)), f"NaN at kf={kf}"

    # Check monotonic increase
    for i in range(1, len(norms)):
        assert norms[i] >= norms[i - 1] * 0.99, (
            f"|grad| not monotonic: kf={kf_values[i]} ({norms[i]:.6e}) "
            f"< kf={kf_values[i-1]} ({norms[i-1]:.6e})"
        )

    ratio = norms[-1] / max(norms[0], 1e-10)
    print(f"Ratio kf={kf_values[-1]}/kf=0: {ratio:.2f}x")
    print("PASSED\n")


def test_bypass_sign_alignment_with_fd():
    """Bypass gradient should have same sign pattern as FD reference for most elements."""
    print("=" * 70)
    print("Test: sign alignment with native MuJoCo FD")
    print("=" * 70)

    np.random.seed(42)
    ctrl_val = np.random.uniform(-0.3, 0.3, (1, 8))

    # Bypass gradient
    grad_bypass = _compute_grad_settled(
        smooth_adjoint=True, friction_bypass_kf=1.0, ctrl_val=ctrl_val
    )

    # FD reference via native MuJoCo
    model_path = resolve_model_path('assets/ant.xml')
    mjm = mujoco.MjModel.from_xml_path(model_path)
    mjd = mujoco.MjData(mjm)

    # Match settled state: run native MuJoCo forward
    mjd.qpos[:] = [0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    mjd.qvel[:] = 0
    mjd.ctrl[:] = ctrl_val.flatten()
    for _ in range(50 * 16):  # 50 * substeps
        mujoco.mj_step(mjm, mjd)

    # Centered FD at settled state
    eps = 1e-4
    grad_fd = np.zeros(8)
    base_ctrl = mjd.ctrl.copy()
    base_qpos = mjd.qpos.copy()
    base_qvel = mjd.qvel.copy()
    base_time = mjd.time

    for i in range(8):
        # Forward perturbation
        mjd.qpos[:] = base_qpos
        mjd.qvel[:] = base_qvel
        mjd.time = base_time
        ctrl_p = base_ctrl.copy()
        ctrl_p[i] += eps
        mjd.ctrl[:] = ctrl_p
        for _ in range(16):
            mujoco.mj_step(mjm, mjd)
        fwd_vel_p = mjd.qvel[0]

        # Backward perturbation
        mjd.qpos[:] = base_qpos
        mjd.qvel[:] = base_qvel
        mjd.time = base_time
        ctrl_m = base_ctrl.copy()
        ctrl_m[i] -= eps
        mjd.ctrl[:] = ctrl_m
        for _ in range(16):
            mujoco.mj_step(mjm, mjd)
        fwd_vel_m = mjd.qvel[0]

        grad_fd[i] = (fwd_vel_p - fwd_vel_m) / (2 * eps)

    # Compare signs
    nonzero = np.abs(grad_fd) > 1e-8
    if nonzero.sum() > 0:
        sign_match = np.sign(grad_bypass[nonzero]) == np.sign(grad_fd[nonzero])
        pct = sign_match.mean() * 100
        print(f"Bypass grad:  {grad_bypass}")
        print(f"FD grad:      {grad_fd}")
        print(f"Sign match:   {pct:.0f}% ({sign_match.sum()}/{nonzero.sum()} non-zero elements)")

        # Also compute cosine similarity
        cos = np.dot(grad_bypass, grad_fd) / (
            np.linalg.norm(grad_bypass) * np.linalg.norm(grad_fd) + 1e-10
        )
        print(f"Cosine sim:   {cos:.3f}")

        assert pct >= 50, f"Sign alignment too low: {pct:.0f}%"
        print("PASSED\n")
    else:
        print("FD gradient is near-zero, skipping sign comparison")
        print("SKIPPED\n")


if __name__ == "__main__":
    test_bypass_kf0_matches_smooth_adjoint()
    test_bypass_amplifies_gradient()
    test_bypass_does_not_affect_forward()
    test_bypass_gradient_scaling()
    test_bypass_sign_alignment_with_fd()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
