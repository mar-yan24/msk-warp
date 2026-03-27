"""Gradient verification: AD (via WarpSimStep) vs finite differences.

Tests:
1. Single-step (cartpole): loss = sum(qpos_after), verify d(loss)/d(ctrl)
2. Network-in-loop (cartpole): ctrl = linear(obs) -> WarpSimStep -> loss
3. Single-step (ant): same as #1 but with free joints and contacts
4. Tape vs FD comparison (cartpole): compare tape-based and FD-based gradients
5. Tape vs FD comparison (ant): same for contact-rich model
6. Tape-per-substep vs tape-all: verify both tape modes agree
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


class MinimalEnv:
    """Minimal environment wrapper for gradient testing."""

    def __init__(self, model_path='assets/cartpole.xml', nworld=1, substeps=4,
                 njmax=None, device='cuda:0', use_fd_jacobian=False,
                 tape_per_substep=False):
        self.device = device
        self.substeps = substeps
        self.nworld = nworld
        self._njmax = njmax
        self.use_fd_jacobian = use_fd_jacobian
        self.tape_per_substep = tape_per_substep

        model_path = resolve_model_path(model_path)

        self.mjm = mujoco.MjModel.from_xml_path(model_path)
        self.warp_model = mjw.put_model(self.mjm)

    def set_state(self, qpos_np, qvel_np):
        """Set state from numpy arrays of shape (nworld, nq) and (nworld, nv)."""
        kwargs = {}
        if self._njmax is not None:
            kwargs['njmax'] = self._njmax
        self.warp_data = mjw.make_diff_data(self.mjm, nworld=self.nworld, **kwargs)
        mjw.reset_data(self.warp_model, self.warp_data)
        wp.copy(self.warp_data.qpos, wp.array(qpos_np.astype(np.float32), dtype=wp.float32))
        wp.copy(self.warp_data.qvel, wp.array(qvel_np.astype(np.float32), dtype=wp.float32))
        wp.synchronize()


def fd_gradient_mujoco(mjm, qpos0, qvel0, ctrl_val, n_substeps, eps=1e-6):
    """Float64 finite-difference gradient using native MuJoCo."""
    mjd = mujoco.MjData(mjm)
    fd_grad = np.zeros(mjm.nu)
    for j in range(mjm.nu):
        vals = []
        for sign in [1, -1]:
            mujoco.mj_resetData(mjm, mjd)
            mjd.qpos[:] = qpos0.flatten()
            mjd.qvel[:] = qvel0.flatten()
            mjd.ctrl[:] = ctrl_val.flatten()
            mjd.ctrl[j] += sign * eps
            for _ in range(n_substeps):
                mujoco.mj_step(mjm, mjd)
            vals.append(mjd.qpos.sum())
        fd_grad[j] = (vals[0] - vals[1]) / (2 * eps)
    return fd_grad


def _run_ad_gradient(env, qpos0, qvel0, ctrl_val):
    """Run a single WarpSimStep and return AD gradient w.r.t. ctrl."""
    env.set_state(qpos0, qvel0)
    ctrl_torch = torch.tensor(ctrl_val, dtype=torch.float32, device='cuda:0', requires_grad=True)
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().requires_grad_(True)
    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos_in, qvel_in, env)
    loss = qpos_out.sum()
    loss.backward()
    return ctrl_torch.grad.clone().cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Test 1: Single-step CartPole
# ---------------------------------------------------------------------------

def test_single_step():
    """Test 1: Single step (cartpole), loss = sum(qpos_after)."""
    print("=" * 60)
    print("Test 1: Single-step gradient verification (CartPole)")
    print("=" * 60)

    env = MinimalEnv(nworld=1, substeps=4)

    qpos0 = np.array([[0.5, 1.0]])
    qvel0 = np.array([[0.1, 0.2]])
    ctrl_val = np.array([[5.0]])

    ad_grad = _run_ad_gradient(env, qpos0, qvel0, ctrl_val)
    fd_grad = fd_gradient_mujoco(env.mjm, qpos0, qvel0, ctrl_val, n_substeps=4)

    print(f"AD gradient:  {ad_grad}")
    print(f"FD gradient:  {fd_grad}")

    if np.abs(fd_grad).max() > 1e-10:
        rel_error = np.abs(ad_grad - fd_grad) / (np.abs(fd_grad) + 1e-10)
        print(f"Relative error: {rel_error}")
        assert rel_error.max() < 0.1, f"Single-step gradient check failed: rel_error={rel_error}"
    else:
        abs_error = np.abs(ad_grad - fd_grad)
        print(f"Absolute error: {abs_error}")
        assert abs_error.max() < 1e-5, f"Single-step gradient check failed: abs_error={abs_error}"

    print("PASS\n")


# ---------------------------------------------------------------------------
# Test 2: Network-in-loop CartPole
# ---------------------------------------------------------------------------

def test_network_in_loop():
    """Test 2: linear(obs) -> WarpSimStep -> loss, verify network grads."""
    print("=" * 60)
    print("Test 2: Network-in-loop gradient verification (CartPole)")
    print("=" * 60)

    env = MinimalEnv(nworld=1, substeps=4)
    nq = env.mjm.nq
    nv = env.mjm.nv
    nu = env.mjm.nu

    qpos0 = np.array([[0.5, 1.0]])
    qvel0 = np.array([[0.1, 0.2]])

    linear = torch.nn.Linear(nq + nv, nu).to('cuda:0')

    env.set_state(qpos0, qvel0)
    wp.synchronize()
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().requires_grad_(True)
    obs = torch.cat([qpos_in, qvel_in], dim=-1)
    ctrl = linear(obs)
    qpos_out, qvel_out = WarpSimStep.apply(ctrl, qpos_in, qvel_in, env)
    loss = qpos_out.sum()
    loss.backward()

    has_grad = linear.weight.grad is not None and torch.any(linear.weight.grad.abs() > 1e-12)
    print(f"linear.weight.grad exists and nonzero: {has_grad}")
    if linear.weight.grad is not None:
        print(f"linear.weight.grad norm: {linear.weight.grad.norm().item():.8f}")
        print(f"linear.bias.grad norm: {linear.bias.grad.norm().item():.8f}")

    assert has_grad, "Network gradient not flowing through WarpSimStep"
    print("PASS\n")


# ---------------------------------------------------------------------------
# Test 3: Single-step Ant
# ---------------------------------------------------------------------------

def test_ant_single_step():
    """Test 3: Single step (ant), loss = sum(qpos_after).

    Validates the bridge with free joints (quaternion integration Jacobian)
    and contact-rich dynamics. Uses relaxed tolerance due to contact
    discontinuities in the FD Jacobian.
    """
    print("=" * 60)
    print("Test 3: Single-step gradient verification (Ant)")
    print("=" * 60)

    env = MinimalEnv(
        model_path='assets/ant.xml', nworld=1, substeps=16, njmax=512,
    )

    nq = env.mjm.nq  # 15
    nv = env.mjm.nv   # 14
    nu = env.mjm.nu   # 8

    # Standing keyframe pose
    qpos0 = np.zeros((1, nq), dtype=np.float64)
    qpos0[0, :] = [0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    qvel0 = np.zeros((1, nv), dtype=np.float64)
    ctrl_val = np.random.uniform(-0.5, 0.5, (1, nu))

    ad_grad = _run_ad_gradient(env, qpos0, qvel0, ctrl_val)
    fd_grad = fd_gradient_mujoco(env.mjm, qpos0, qvel0, ctrl_val, n_substeps=16)

    print(f"AD gradient:  {ad_grad}")
    print(f"FD gradient:  {fd_grad}")

    # Relaxed tolerance for contact-rich dynamics
    nonzero = np.abs(fd_grad) > 1e-8
    if nonzero.any():
        rel_error = np.abs(ad_grad[nonzero] - fd_grad[nonzero]) / (np.abs(fd_grad[nonzero]) + 1e-10)
        max_rel = rel_error.max()
        print(f"Relative error (nonzero components): max={max_rel:.4f}, mean={rel_error.mean():.4f}")
        assert max_rel < 0.5, f"Ant gradient check failed: max rel_error={max_rel:.4f}"
    else:
        abs_error = np.abs(ad_grad - fd_grad)
        print(f"Absolute error: max={abs_error.max():.6f}")

    assert np.abs(ad_grad).max() > 1e-8, "AD gradient is all zeros -- bridge may not be working"
    print("PASS\n")


# ---------------------------------------------------------------------------
# Test 4: Tape vs FD comparison (CartPole)
# ---------------------------------------------------------------------------

def test_tape_vs_fd_cartpole():
    """Compare tape-based and FD-based backward gradients on CartPole."""
    print("=" * 60)
    print("Test 4: Tape vs FD gradient comparison (CartPole)")
    print("=" * 60)

    qpos0 = np.array([[0.5, 1.0]])
    qvel0 = np.array([[0.1, 0.2]])
    ctrl_val = np.array([[5.0]])

    # Tape-all gradient (default)
    env_tape = MinimalEnv(nworld=1, substeps=4, use_fd_jacobian=False)
    tape_grad = _run_ad_gradient(env_tape, qpos0, qvel0, ctrl_val)

    # FD gradient
    env_fd = MinimalEnv(nworld=1, substeps=4, use_fd_jacobian=True)
    fd_grad = _run_ad_gradient(env_fd, qpos0, qvel0, ctrl_val)

    print(f"Tape gradient: {tape_grad}")
    print(f"FD gradient:   {fd_grad}")

    if np.abs(fd_grad).max() > 1e-10:
        rel_error = np.abs(tape_grad - fd_grad) / (np.abs(fd_grad) + 1e-10)
        print(f"Relative error (tape vs FD): {rel_error}")
        assert rel_error.max() < 0.1, f"Tape vs FD CartPole failed: rel_error={rel_error}"
    else:
        abs_error = np.abs(tape_grad - fd_grad)
        print(f"Absolute error: {abs_error}")
        assert abs_error.max() < 1e-5

    print("PASS\n")


# ---------------------------------------------------------------------------
# Test 5: Tape vs FD comparison (Ant)
# ---------------------------------------------------------------------------

def test_tape_vs_fd_ant():
    """Compare tape-based and FD-based backward gradients on Ant."""
    print("=" * 60)
    print("Test 5: Tape vs FD gradient comparison (Ant)")
    print("=" * 60)

    nq, nv, nu = 15, 14, 8
    qpos0 = np.zeros((1, nq), dtype=np.float64)
    qpos0[0, :] = [0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
    qvel0 = np.zeros((1, nv), dtype=np.float64)
    np.random.seed(42)
    ctrl_val = np.random.uniform(-0.5, 0.5, (1, nu))

    # Tape-all gradient
    env_tape = MinimalEnv(
        model_path='assets/ant.xml', nworld=1, substeps=16, njmax=512,
        use_fd_jacobian=False,
    )
    tape_grad = _run_ad_gradient(env_tape, qpos0, qvel0, ctrl_val)

    # FD gradient
    env_fd = MinimalEnv(
        model_path='assets/ant.xml', nworld=1, substeps=16, njmax=512,
        use_fd_jacobian=True,
    )
    fd_grad = _run_ad_gradient(env_fd, qpos0, qvel0, ctrl_val)

    print(f"Tape gradient: {tape_grad}")
    print(f"FD gradient:   {fd_grad}")

    # Both should be in the same ballpark; they compute different approximations
    # of the same true gradient (tape is exact, FD is approximate)
    nonzero = np.abs(fd_grad) > 1e-8
    if nonzero.any():
        rel_error = np.abs(tape_grad[nonzero] - fd_grad[nonzero]) / (np.abs(fd_grad[nonzero]) + 1e-10)
        max_rel = rel_error.max()
        mean_rel = rel_error.mean()
        print(f"Relative error (tape vs FD): max={max_rel:.4f}, mean={mean_rel:.4f}")
        # Relaxed: tape gives exact gradients, FD is approximate + clamped
        assert max_rel < 1.0, f"Tape vs FD Ant failed: max rel_error={max_rel:.4f}"
    print(f"Tape grad norm: {np.linalg.norm(tape_grad):.6f}")
    print(f"FD grad norm:   {np.linalg.norm(fd_grad):.6f}")

    assert np.abs(tape_grad).max() > 1e-8, "Tape gradient is all zeros"
    print("PASS\n")


# ---------------------------------------------------------------------------
# Test 6: Tape-per-substep matches tape-all
# ---------------------------------------------------------------------------

def test_tape_per_substep_matches_tape_all():
    """Verify tape-all and tape-per-substep produce the same gradients."""
    print("=" * 60)
    print("Test 6: Tape-per-substep vs tape-all (CartPole)")
    print("=" * 60)

    qpos0 = np.array([[0.5, 1.0]])
    qvel0 = np.array([[0.1, 0.2]])
    ctrl_val = np.array([[5.0]])

    # Tape-all
    env_all = MinimalEnv(nworld=1, substeps=4, tape_per_substep=False)
    grad_all = _run_ad_gradient(env_all, qpos0, qvel0, ctrl_val)

    # Tape-per-substep
    env_per = MinimalEnv(nworld=1, substeps=4, tape_per_substep=True)
    grad_per = _run_ad_gradient(env_per, qpos0, qvel0, ctrl_val)

    print(f"Tape-all gradient:          {grad_all}")
    print(f"Tape-per-substep gradient:  {grad_per}")

    if np.abs(grad_all).max() > 1e-10:
        rel_error = np.abs(grad_all - grad_per) / (np.abs(grad_all) + 1e-10)
        print(f"Relative error: {rel_error}")
        # These should be very close (same computation, different taping granularity)
        assert rel_error.max() < 0.01, f"Tape modes disagree: rel_error={rel_error}"
    else:
        abs_error = np.abs(grad_all - grad_per)
        print(f"Absolute error: {abs_error}")
        assert abs_error.max() < 1e-5

    print("PASS\n")


if __name__ == '__main__':
    test_single_step()
    test_network_in_loop()
    test_ant_single_step()
    test_tape_vs_fd_cartpole()
    test_tape_vs_fd_ant()
    test_tape_per_substep_matches_tape_all()
    print("All tests passed.")
