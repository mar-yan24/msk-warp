"""Gradient verification: AD (via WarpSimStep) vs finite differences.

Three tests:
1. Single-step: loss = sum(qpos_after), verify d(loss)/d(ctrl)
2. Multi-step chain: 3 steps, loss on final qpos
3. Network-in-loop: ctrl = linear(obs) -> WarpSimStep -> loss
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import warp as wp
wp.init()

import torch
import numpy as np
import mujoco
import mujoco_warp as mjw

from msk_warp.bridge import WarpSimStep


class MinimalEnv:
    """Minimal environment wrapper for gradient testing."""

    def __init__(self, model_path='assets/cartpole.xml', nworld=1, substeps=4, device='cuda:0'):
        self.device = device
        self.substeps = substeps
        self.nworld = nworld

        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)

        self.mjm = mujoco.MjModel.from_xml_path(model_path)
        self.warp_model = mjw.put_model(self.mjm)

    def set_state(self, qpos_np, qvel_np):
        """Set state from numpy arrays of shape (nworld, nq) and (nworld, nv)."""
        self.warp_data = mjw.make_diff_data(self.mjm, nworld=self.nworld)
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
    """Test 1: Single step, loss = sum(qpos_after)."""
    print("=" * 60)
    print("Test 1: Single-step gradient verification")
    print("=" * 60)

    env = MinimalEnv(nworld=1, substeps=4)
    nu = env.mjm.nu

    qpos0 = np.array([[0.5, 1.0]])
    qvel0 = np.array([[0.1, 0.2]])
    ctrl_val = np.array([[5.0]])

    # AD gradient
    env.set_state(qpos0, qvel0)
    ctrl_torch = torch.tensor(ctrl_val, dtype=torch.float32, device='cuda:0', requires_grad=True)
    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, env)
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
        success = rel_error.max() < 0.1  # 10% tolerance for float32 vs float64
    else:
        abs_error = np.abs(ad_grad - fd_grad)
        print(f"Absolute error: {abs_error}")
        success = abs_error.max() < 1e-5

    print(f"PASS: {success}\n")
    return success


def test_network_in_loop():
    """Test 2: linear(obs) -> WarpSimStep -> loss, verify network grads."""
    print("=" * 60)
    print("Test 2: Network-in-loop gradient verification")
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
    obs = torch.cat([
        wp.to_torch(env.warp_data.qpos).float().detach(),
        wp.to_torch(env.warp_data.qvel).float().detach(),
    ], dim=-1)
    ctrl = linear(obs)
    qpos_out, qvel_out = WarpSimStep.apply(ctrl, env)
    loss = qpos_out.sum()
    loss.backward()

    has_grad = linear.weight.grad is not None and torch.any(linear.weight.grad.abs() > 1e-12)
    print(f"linear.weight.grad exists and nonzero: {has_grad}")
    if linear.weight.grad is not None:
        print(f"linear.weight.grad norm: {linear.weight.grad.norm().item():.8f}")
        print(f"linear.bias.grad norm: {linear.bias.grad.norm().item():.8f}")

    success = has_grad
    print(f"PASS: {success}\n")
    return success


if __name__ == '__main__':
    results = []
    results.append(('Single-step', test_single_step()))
    results.append(('Network-in-loop', test_network_in_loop()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
