"""Gradient bridge between MuJoCo Warp (Warp autodiff) and PyTorch autograd.

Strategy: use Warp tape ONLY through fwd_actuation (ctrl -> qfrc_actuator),
which has verified correct gradients. The rest of the backward chain is computed
analytically + finite-difference dynamics Jacobian:
  - Euler backward: d(loss)/d(qacc) from incoming PyTorch gradients
  - Mass matrix solve: d(loss)/d(qfrc) = solve_M(d(loss)/d(qacc))
  - Actuation backward via Warp tape: d(loss)/d(ctrl) from VJP on qfrc_actuator
  - FD dynamics Jacobian: d(qacc)/d(qpos), d(qacc)/d(qvel) for state gradient
"""

import warp as wp
import torch
import mujoco_warp as mjw


def _qpos_grad_to_qvel_grad(g_qpos, qpos, nq, nv, dt):
    """Map d(loss)/d(qpos_new) to d(loss)/d(qvel_new) through the integration Jacobian.

    MuJoCo semi-implicit Euler:
      qvel_new = qvel + qacc * dt
      qpos_new = integrate_pos(qpos, qvel_new, dt)

    For simple joints (nq==nv): qpos_new = qpos + qvel_new * dt,
      so d(qpos_new)/d(qvel_new) = dt * I, and the VJP is g_qpos * dt.

    For free joints: position part is the same (dt * I_3), but quaternion uses
      quat_new = quat + 0.5 * dt * quat_mul(quat, [0, omega]), so
      d(quat_new)/d(omega) = 0.5 * dt * J, and the VJP maps g_quat (4D) -> g_omega (3D).

    Returns: d(loss)/d(qvel_new) contribution from qpos path, shape (nworld, nv).
    """
    if nq == nv:
        # Simple joints only (e.g., cartpole): d(qpos)/d(qvel) = dt * I
        return g_qpos * dt

    # Free joint present: nq = nv + 1 (7 qpos vs 6 qvel for the free joint)
    nworld = g_qpos.shape[0]
    g_qvel_from_qpos = torch.zeros(nworld, nv, device=g_qpos.device, dtype=g_qpos.dtype)

    # Free joint position (qpos[0:3] -> qvel[0:3]): d(pos_new)/d(lin_vel_new) = dt * I_3
    g_qvel_from_qpos[:, 0:3] = g_qpos[:, 0:3] * dt

    # Free joint quaternion (qpos[3:7] -> qvel[3:6]):
    # quat_new ≈ quat + 0.5 * dt * quat_mul(quat, [0, ω])
    # VJP: g_omega = 0.5 * dt * (quat_conjugate(quat) ⊗ g_quat)[1:4]
    quat = qpos[:, 3:7]    # [w, x, y, z]
    g_quat = g_qpos[:, 3:7]
    # Hamilton product of conjugate(quat) with g_quat (treated as quaternion)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    gw, gx, gy, gz = g_quat[:, 0], g_quat[:, 1], g_quat[:, 2], g_quat[:, 3]
    # (q*)⊗g = [w,-x,-y,-z]⊗[gw,gx,gy,gz], take xyz components:
    g_omega_x = -x * gw + w * gx + z * gy - y * gz
    g_omega_y = -y * gw - z * gx + w * gy + x * gz
    g_omega_z = -z * gw + y * gx - x * gy + w * gz
    g_qvel_from_qpos[:, 3] = 0.5 * dt * g_omega_x
    g_qvel_from_qpos[:, 4] = 0.5 * dt * g_omega_y
    g_qvel_from_qpos[:, 5] = 0.5 * dt * g_omega_z

    # Hinge/slide joints (qpos[7:] -> qvel[6:]): d(qpos)/d(qvel) = dt * I
    n_hinge = nq - 7
    g_qvel_from_qpos[:, 6:6 + n_hinge] = g_qpos[:, 7:7 + n_hinge] * dt

    return g_qvel_from_qpos


@wp.kernel
def vjp_qfrc_kernel(
    qfrc_actuator: wp.array2d(dtype=float),
    grad_qfrc: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
):
    """Compute loss = sum(qfrc_actuator * grad_qfrc) for VJP."""
    worldid, dofid = wp.tid()
    nv = qfrc_actuator.shape[1]
    if dofid < nv:
        wp.atomic_add(loss, 0, qfrc_actuator[worldid, dofid] * grad_qfrc[worldid, dofid])


class WarpSimStep(torch.autograd.Function):
    """Differentiable simulation step bridging Warp and PyTorch.

    Accepts (ctrl, qpos_in, qvel_in) as differentiable inputs so that
    gradients flow through both the actuation path (ctrl -> forces) AND
    the dynamics path (state -> next_state) across simulation steps.
    """

    @staticmethod
    def forward(ctx, ctrl_torch, qpos_in_torch, qvel_in_torch, env):
        m = env.warp_model
        d = env.warp_data

        nworld = d.qpos.shape[0]
        nq = d.qpos.shape[1]
        nv = d.qvel.shape[1]

        # Copy input state to Warp data
        wp.copy(d.qpos, wp.from_torch(qpos_in_torch.detach().contiguous()))
        wp.copy(d.qvel, wp.from_torch(qvel_in_torch.detach().contiguous()))

        # Save pre-step state for backward checkpointing
        saved_qpos = wp.clone(d.qpos)
        saved_qvel = wp.clone(d.qvel)
        saved_time = wp.clone(d.time)

        # Set ctrl from PyTorch tensor
        ctrl_wp = wp.from_torch(ctrl_torch.contiguous())
        wp.copy(d.ctrl, ctrl_wp)

        # Run substeps (no tape — forward only)
        for _ in range(env.substeps):
            mjw.step(m, d)

        wp.synchronize()

        # Extract post-step state as PyTorch tensors
        qpos_torch = wp.to_torch(d.qpos).clone()
        qvel_torch = wp.to_torch(d.qvel).clone()

        # Save for backward
        ctx.env = env
        ctx.saved_qpos = saved_qpos
        ctx.saved_qvel = saved_qvel
        ctx.saved_time = saved_time
        ctx.ctrl_torch = ctrl_torch.detach()
        ctx.nworld = nworld
        ctx.nq = nq
        ctx.nv = nv

        return qpos_torch, qvel_torch

    @staticmethod
    def backward(ctx, grad_qpos_torch, grad_qvel_torch):
        env = ctx.env
        m = env.warp_model
        d = env.warp_data
        nworld = ctx.nworld
        nq = ctx.nq
        nv = ctx.nv
        substeps = env.substeps

        dt = wp.to_torch(m.opt.timestep).item()
        fd_eps = 1e-4
        fd_max_dqacc = 1.0  # Clamp raw FD diffs to block contact discontinuity spikes
        substep_grad_max = 1e4  # Prevent per-substep gradient explosion

        # Restore to initial state
        wp.copy(d.qpos, ctx.saved_qpos)
        wp.copy(d.qvel, ctx.saved_qvel)
        wp.copy(d.time, ctx.saved_time)
        ctrl_wp = wp.from_torch(ctx.ctrl_torch.contiguous())
        wp.copy(d.ctrl, ctrl_wp)
        wp.synchronize()

        # Save intermediate states for all substeps
        states = []
        for s in range(substeps):
            states.append((wp.clone(d.qpos), wp.clone(d.qvel), wp.clone(d.time)))
            mjw.step(m, d)
        wp.synchronize()

        # Current gradients w.r.t. post-final-substep state
        g_qpos = grad_qpos_torch.clone()
        g_qvel = grad_qvel_torch.clone()

        # Accumulate ctrl gradient across substeps
        total_grad_ctrl = torch.zeros_like(ctx.ctrl_torch)

        # Backward through substeps in reverse
        for s in reversed(range(substeps)):
            pre_qpos, pre_qvel, pre_time = states[s]

            # Restore state to pre-substep
            wp.copy(d.qpos, pre_qpos)
            wp.copy(d.qvel, pre_qvel)
            wp.copy(d.time, pre_time)
            wp.copy(d.ctrl, ctrl_wp)
            wp.synchronize()

            # 1. Analytical Euler backward:
            #    Semi-implicit: qvel_new = qvel + qacc*dt
            #                   qpos_new = integrate_pos(qpos, qvel_new, dt)
            #    d(loss)/d(qacc) = d(loss)/d(qvel_new)*dt + d(loss)/d(qpos_new)*dqpos/dqvel*dt
            #    For nq==nv this simplifies to g_qpos*dt^2 + g_qvel*dt.
            #    For free joints, the quaternion integration Jacobian is used.
            pre_qpos_torch = wp.to_torch(pre_qpos)
            g_qvel_from_qpos = _qpos_grad_to_qvel_grad(g_qpos, pre_qpos_torch, nq, nv, dt)
            grad_qacc_torch = g_qvel_from_qpos * dt + g_qvel * dt

            # 2. Run forward dynamics to get factored mass matrix + qacc
            mjw.forward(m, d)
            wp.synchronize()
            qacc_orig = wp.to_torch(d.qacc).clone()

            # 3. Solve M_inv * grad_qacc to get grad_qfrc
            grad_qacc_wp = wp.from_torch(grad_qacc_torch.contiguous())
            grad_qfrc_wp = wp.zeros((nworld, nv), dtype=wp.float32)
            mjw.solve_m(m, d, grad_qfrc_wp, grad_qacc_wp)
            wp.synchronize()

            # 4. Finite-difference dynamics Jacobian: ∂qacc/∂qpos and ∂qacc/∂qvel
            #    These capture gravity, Coriolis, mass-matrix dependence on state —
            #    the terms missing from the actuation-only tape backward.
            qpos_view = wp.to_torch(d.qpos)
            qvel_view = wp.to_torch(d.qvel)
            fd_g_qpos = torch.zeros(nworld, nq, device=g_qpos.device)
            fd_g_qvel = torch.zeros(nworld, nv, device=g_qvel.device)

            for j in range(nq):
                qpos_view[:, j] += fd_eps
                mjw.forward(m, d)
                wp.synchronize()
                qacc_plus = wp.to_torch(d.qacc).clone()
                qpos_view[:, j] -= fd_eps  # restore

                dqacc_raw = qacc_plus - qacc_orig
                dqacc_raw = dqacc_raw.clamp(-fd_max_dqacc, fd_max_dqacc)
                dqacc = torch.nan_to_num(dqacc_raw / fd_eps, 0.0, 0.0, 0.0)
                # VJP: grad_qacc^T @ (∂qacc/∂qpos_j)
                fd_g_qpos[:, j] = (grad_qacc_torch * dqacc).sum(dim=-1)

            for j in range(nv):
                qvel_view[:, j] += fd_eps
                mjw.forward(m, d)
                wp.synchronize()
                qacc_plus = wp.to_torch(d.qacc).clone()
                qvel_view[:, j] -= fd_eps  # restore

                dqacc_raw = qacc_plus - qacc_orig
                dqacc_raw = dqacc_raw.clamp(-fd_max_dqacc, fd_max_dqacc)
                dqacc = torch.nan_to_num(dqacc_raw / fd_eps, 0.0, 0.0, 0.0)
                fd_g_qvel[:, j] = (grad_qacc_torch * dqacc).sum(dim=-1)

            # 5. Use Warp tape through fwd_actuation only to get d(loss)/d(ctrl)
            #    Restore state again for clean tape
            wp.copy(d.qpos, pre_qpos)
            wp.copy(d.qvel, pre_qvel)
            wp.copy(d.time, pre_time)
            wp.copy(d.ctrl, ctrl_wp)
            wp.synchronize()

            # Run kinematics + velocity OUTSIDE the tape. These contain
            # constraint kernels with enable_backwards=False that produce NaN
            # gradients. Safe for direct-drive actuators (gear * ctrl) where
            # fwd_actuation does not depend on position/velocity outputs.
            mjw.fwd_position(m, d)
            mjw.fwd_velocity(m, d)
            wp.synchronize()

            d.ctrl.grad = wp.zeros_like(d.ctrl)

            loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            tape = wp.Tape()
            with tape:
                mjw.fwd_actuation(m, d)
                wp.launch(
                    vjp_qfrc_kernel,
                    dim=(nworld, nv),
                    inputs=[d.qfrc_actuator, grad_qfrc_wp, loss],
                )
            tape.backward(loss=loss)
            wp.synchronize()

            total_grad_ctrl += wp.to_torch(d.ctrl.grad).clone()

            # 6. Propagate gradients: Euler backward + FD dynamics Jacobian
            #    g_qpos_prev = g_qpos + grad_qacc @ ∂qacc/∂qpos
            #    g_qvel_prev = (g_qvel + dqpos/dqvel^T @ g_qpos) + grad_qacc @ ∂qacc/∂qvel
            g_qpos_prev = g_qpos.clone() + fd_g_qpos
            g_qvel_prev = g_qvel + g_qvel_from_qpos + fd_g_qvel
            g_qpos = g_qpos_prev.clamp(-substep_grad_max, substep_grad_max)
            g_qvel = g_qvel_prev.clamp(-substep_grad_max, substep_grad_max)

            tape.zero()

        # g_qpos, g_qvel now hold the gradient w.r.t. the INPUT state
        total_grad_ctrl = torch.nan_to_num(total_grad_ctrl, 0.0, 0.0, 0.0)
        grad_qpos_in = torch.nan_to_num(g_qpos, 0.0, 0.0, 0.0)
        grad_qvel_in = torch.nan_to_num(g_qvel, 0.0, 0.0, 0.0)

        # Restore to post-step state
        wp.copy(d.qpos, ctx.saved_qpos)
        wp.copy(d.qvel, ctx.saved_qvel)
        wp.copy(d.time, ctx.saved_time)
        wp.copy(d.ctrl, ctrl_wp)
        for _ in range(substeps):
            mjw.step(m, d)
        wp.synchronize()

        return total_grad_ctrl, grad_qpos_in, grad_qvel_in, None
