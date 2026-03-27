"""Gradient verification: AD (via WarpSimStep) vs finite differences.

Three tests:
1. Single-step (cartpole): loss = sum(qpos_after), verify d(loss)/d(ctrl)
2. Network-in-loop (cartpole): ctrl = linear(obs) -> WarpSimStep -> loss
3. Single-step (ant): same as #1 but with free joints and contacts
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
                 njmax=None, device='cuda:0'):
        self.device = device
        self.substeps = substeps
        self.nworld = nworld
        self._njmax = njmax

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


def test_single_step():
    """Test 1: Single step (cartpole), loss = sum(qpos_after)."""
    print("=" * 60)
    print("Test 1: Single-step gradient verification (CartPole)")
    print("=" * 60)

    env = MinimalEnv(nworld=1, substeps=4)

    qpos0 = np.array([[0.5, 1.0]])
    qvel0 = np.array([[0.1, 0.2]])
    ctrl_val = np.array([[5.0]])

    # AD gradient
    env.set_state(qpos0, qvel0)
    ctrl_torch = torch.tensor(ctrl_val, dtype=torch.float32, device='cuda:0', requires_grad=True)
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().requires_grad_(True)
    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos_in, qvel_in, env)
    loss = qpos_out.sum()
    loss.backward()
    ad_grad = ctrl_torch.grad.clone().cpu().numpy().flatten()

    # Float64 FD
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

    # AD gradient
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

    # AD gradient
    env.set_state(qpos0, qvel0)
    ctrl_torch = torch.tensor(ctrl_val, dtype=torch.float32, device='cuda:0', requires_grad=True)
    qpos_in = wp.to_torch(env.warp_data.qpos).clone().requires_grad_(True)
    qvel_in = wp.to_torch(env.warp_data.qvel).clone().requires_grad_(True)
    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos_in, qvel_in, env)
    loss = qpos_out.sum()
    loss.backward()
    ad_grad = ctrl_torch.grad.clone().cpu().numpy().flatten()

    # Float64 FD
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

    # Also verify gradients are non-trivial
    assert np.abs(ad_grad).max() > 1e-8, "AD gradient is all zeros -- bridge may not be working"
    print("PASS\n")


if __name__ == '__main__':
    test_single_step()
    test_network_in_loop()
    test_ant_single_step()
    print("All tests passed.")
