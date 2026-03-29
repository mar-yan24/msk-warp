"""Gradient bridge between MuJoCo Warp (Warp autodiff) and PyTorch autograd.

Three backward modes:
  1. Tape-all (default): single wp.Tape() over all substeps — fastest, ~2-3x forward cost
  2. Tape-per-substep: tape each substep individually, chain gradients — lower memory
  3. FD Jacobian (fallback): finite-difference dynamics Jacobian — slow but battle-tested

Mode selection via env flags:
  env.use_fd_jacobian = True   → mode 3
  env.tape_per_substep = True  → mode 2
  else                         → mode 1
"""

import os

import warp as wp
import torch
import mujoco_warp as mjw

# Set MSK_GRAD_DIAG=1 to print per-backward gradient diagnostics
_GRAD_DIAG = bool(int(os.environ.get("MSK_GRAD_DIAG", "0")))
_grad_diag_count = 0


def _log_grad_diag(mode, incoming_qpos, incoming_qvel, raw_ctrl, raw_qpos, raw_qvel):
    """Print gradient magnitudes and NaN counts for debugging."""
    global _grad_diag_count
    _grad_diag_count += 1
    # Only log every 32 calls (once per SHAC rollout) to avoid flooding
    if _grad_diag_count % 32 != 1:
        return
    step_label = f"step {(_grad_diag_count - 1) % 32}"
    nan_ctrl = raw_ctrl.isnan().sum().item()
    nan_qpos = raw_qpos.isnan().sum().item()
    nan_qvel = raw_qvel.isnan().sum().item()
    total_elems = raw_ctrl.numel() + raw_qpos.numel() + raw_qvel.numel()
    total_nan = nan_ctrl + nan_qpos + nan_qvel
    print(
        f"  [GRAD DIAG {mode} {step_label}] "
        f"incoming |g_qpos|={incoming_qpos.norm():.4e} |g_qvel|={incoming_qvel.norm():.4e} | "
        f"tape raw |ctrl|={raw_ctrl.norm():.4e} |qpos|={raw_qpos.norm():.4e} |qvel|={raw_qvel.norm():.4e} | "
        f"NaN {total_nan}/{total_elems} (ctrl={nan_ctrl} qpos={nan_qpos} qvel={nan_qvel})"
    )


# ---------------------------------------------------------------------------
# VJP kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _vjp_state_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    grad_qpos: wp.array2d(dtype=float),
    grad_qvel: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
):
    """Seed tape backward: loss = sum(qpos * grad_qpos + qvel * grad_qvel)."""
    worldid, idx = wp.tid()
    nq = qpos.shape[1]
    nv = qvel.shape[1]
    if idx < nq:
        wp.atomic_add(loss, 0, qpos[worldid, idx] * grad_qpos[worldid, idx])
    if idx < nv:
        wp.atomic_add(loss, 0, qvel[worldid, idx] * grad_qvel[worldid, idx])


# ---------------------------------------------------------------------------
# FD backward helpers (used only by _backward_fd)
# ---------------------------------------------------------------------------

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
        return g_qpos * dt

    nworld = g_qpos.shape[0]
    g_qvel_from_qpos = torch.zeros(nworld, nv, device=g_qpos.device, dtype=g_qpos.dtype)

    g_qvel_from_qpos[:, 0:3] = g_qpos[:, 0:3] * dt

    quat = qpos[:, 3:7]
    g_quat = g_qpos[:, 3:7]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    gw, gx, gy, gz = g_quat[:, 0], g_quat[:, 1], g_quat[:, 2], g_quat[:, 3]
    g_omega_x = -x * gw + w * gx + z * gy - y * gz
    g_omega_y = -y * gw - z * gx + w * gy + x * gz
    g_omega_z = -z * gw + y * gx - x * gy + w * gz
    g_qvel_from_qpos[:, 3] = 0.5 * dt * g_omega_x
    g_qvel_from_qpos[:, 4] = 0.5 * dt * g_omega_y
    g_qvel_from_qpos[:, 5] = 0.5 * dt * g_omega_z

    n_hinge = nq - 7
    g_qvel_from_qpos[:, 6:6 + n_hinge] = g_qpos[:, 7:7 + n_hinge] * dt

    return g_qvel_from_qpos


@wp.kernel
def _vjp_qfrc_kernel(
    qfrc_actuator: wp.array2d(dtype=float),
    grad_qfrc: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
):
    """Compute loss = sum(qfrc_actuator * grad_qfrc) for ctrl VJP (FD path only)."""
    worldid, dofid = wp.tid()
    nv = qfrc_actuator.shape[1]
    if dofid < nv:
        wp.atomic_add(loss, 0, qfrc_actuator[worldid, dofid] * grad_qfrc[worldid, dofid])


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _restore_and_rerun(m, d, saved_qpos, saved_qvel, saved_time, saved_act, ctrl_wp, substeps):
    """Restore pre-step state and re-run substeps to reach post-step state."""
    wp.copy(d.qpos, saved_qpos)
    wp.copy(d.qvel, saved_qvel)
    wp.copy(d.time, saved_time)
    if saved_act is not None:
        wp.copy(d.act, saved_act)
    wp.copy(d.ctrl, ctrl_wp)
    for _ in range(substeps):
        mjw.step(m, d)
    wp.synchronize()


def _sanitize_and_clamp(grad_ctrl, grad_qpos, grad_qvel, max_grad=1e4):
    """NaN-to-zero and clamp returned gradients."""
    grad_ctrl = torch.nan_to_num(grad_ctrl, 0.0, 0.0, 0.0).clamp(-max_grad, max_grad)
    grad_qpos = torch.nan_to_num(grad_qpos, 0.0, 0.0, 0.0).clamp(-max_grad, max_grad)
    grad_qvel = torch.nan_to_num(grad_qvel, 0.0, 0.0, 0.0).clamp(-max_grad, max_grad)
    return grad_ctrl, grad_qpos, grad_qvel


# ---------------------------------------------------------------------------
# WarpSimStep: differentiable simulation step
# ---------------------------------------------------------------------------

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
        saved_act = wp.clone(d.act) if d.act.shape[1] > 0 else None

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
        ctx.saved_act = saved_act
        ctx.ctrl_torch = ctrl_torch.detach()
        ctx.nworld = nworld
        ctx.nq = nq
        ctx.nv = nv

        return qpos_torch, qvel_torch

    @staticmethod
    def backward(ctx, grad_qpos_torch, grad_qvel_torch):
        env = ctx.env

        if getattr(env, 'use_fd_jacobian', False):
            return WarpSimStep._backward_fd(ctx, grad_qpos_torch, grad_qvel_torch)
        elif getattr(env, 'tape_per_substep', False):
            return WarpSimStep._backward_tape_per_substep(ctx, grad_qpos_torch, grad_qvel_torch)
        else:
            return WarpSimStep._backward_tape(ctx, grad_qpos_torch, grad_qvel_torch)

    # ------------------------------------------------------------------
    # Mode 1: Tape over ALL substeps (default, fastest)
    # ------------------------------------------------------------------

    @staticmethod
    def _backward_tape(ctx, grad_qpos_torch, grad_qvel_torch):
        env = ctx.env
        m, d = env.warp_model, env.warp_data
        substeps = env.substeps

        # 1. Restore to pre-step state
        wp.copy(d.qpos, ctx.saved_qpos)
        wp.copy(d.qvel, ctx.saved_qvel)
        wp.copy(d.time, ctx.saved_time)
        if ctx.saved_act is not None:
            wp.copy(d.act, ctx.saved_act)
        ctrl_wp = wp.from_torch(ctx.ctrl_torch.contiguous())
        wp.copy(d.ctrl, ctrl_wp)
        wp.synchronize()

        # 2. Convert incoming PyTorch grads to Warp arrays
        grad_qpos_wp = wp.from_torch(grad_qpos_torch.contiguous())
        grad_qvel_wp = wp.from_torch(grad_qvel_torch.contiguous())

        # 3. Zero existing .grad fields
        d.qpos.grad = wp.zeros_like(d.qpos)
        d.qvel.grad = wp.zeros_like(d.qvel)
        d.ctrl.grad = wp.zeros_like(d.ctrl)
        if ctx.saved_act is not None:
            d.act.grad = wp.zeros_like(d.act)

        # 4. Tape through all substeps + VJP kernel
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        tape = wp.Tape()
        with tape:
            for _ in range(substeps):
                mjw.step(m, d)
            wp.launch(
                _vjp_state_kernel,
                dim=(ctx.nworld, max(ctx.nq, ctx.nv)),
                inputs=[d.qpos, d.qvel, grad_qpos_wp, grad_qvel_wp, loss],
            )

        # 5. Backward through tape
        tape.backward(loss=loss)
        wp.synchronize()

        # 6. Extract gradients
        grad_ctrl = wp.to_torch(d.ctrl.grad).clone()
        grad_qpos_in = wp.to_torch(d.qpos.grad).clone()
        grad_qvel_in = wp.to_torch(d.qvel.grad).clone()

        # Diagnostic: log raw gradient magnitudes before sanitization
        if _GRAD_DIAG:
            _log_grad_diag(
                "tape-all", grad_qpos_torch, grad_qvel_torch,
                grad_ctrl, grad_qpos_in, grad_qvel_in,
            )

        # 7. Sanitize and clamp
        grad_ctrl, grad_qpos_in, grad_qvel_in = _sanitize_and_clamp(
            grad_ctrl, grad_qpos_in, grad_qvel_in
        )

        # 8. Clean up tape
        tape.zero()

        # 9. Restore to post-step state
        _restore_and_rerun(
            m, d, ctx.saved_qpos, ctx.saved_qvel, ctx.saved_time,
            ctx.saved_act, ctrl_wp, substeps,
        )

        return grad_ctrl, grad_qpos_in, grad_qvel_in, None

    # ------------------------------------------------------------------
    # Mode 2: Tape per substep (lower memory)
    # ------------------------------------------------------------------

    @staticmethod
    def _backward_tape_per_substep(ctx, grad_qpos_torch, grad_qvel_torch):
        env = ctx.env
        m, d = env.warp_model, env.warp_data
        substeps = env.substeps

        # 1. Restore to pre-step state and capture intermediate states
        wp.copy(d.qpos, ctx.saved_qpos)
        wp.copy(d.qvel, ctx.saved_qvel)
        wp.copy(d.time, ctx.saved_time)
        if ctx.saved_act is not None:
            wp.copy(d.act, ctx.saved_act)
        ctrl_wp = wp.from_torch(ctx.ctrl_torch.contiguous())
        wp.copy(d.ctrl, ctrl_wp)
        wp.synchronize()

        # 2. Save intermediate states for all substeps
        has_act = ctx.saved_act is not None
        states = []
        for s in range(substeps):
            act_snap = wp.clone(d.act) if has_act else None
            states.append((wp.clone(d.qpos), wp.clone(d.qvel), wp.clone(d.time), act_snap))
            mjw.step(m, d)
        wp.synchronize()

        # 3. Current gradients w.r.t. post-final-substep state
        g_qpos = grad_qpos_torch.clone()
        g_qvel = grad_qvel_torch.clone()
        total_grad_ctrl = torch.zeros_like(ctx.ctrl_torch)

        # 4. Backward through substeps in reverse
        for s in reversed(range(substeps)):
            pre_qpos, pre_qvel, pre_time, pre_act = states[s]

            # Restore pre-substep state
            wp.copy(d.qpos, pre_qpos)
            wp.copy(d.qvel, pre_qvel)
            wp.copy(d.time, pre_time)
            if pre_act is not None:
                wp.copy(d.act, pre_act)
            wp.copy(d.ctrl, ctrl_wp)
            wp.synchronize()

            # Convert current grads to Warp
            grad_qpos_wp = wp.from_torch(g_qpos.contiguous())
            grad_qvel_wp = wp.from_torch(g_qvel.contiguous())

            # Zero .grad fields
            d.qpos.grad = wp.zeros_like(d.qpos)
            d.qvel.grad = wp.zeros_like(d.qvel)
            d.ctrl.grad = wp.zeros_like(d.ctrl)
            if has_act:
                d.act.grad = wp.zeros_like(d.act)

            # Tape one substep + VJP
            loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            tape = wp.Tape()
            with tape:
                mjw.step(m, d)
                wp.launch(
                    _vjp_state_kernel,
                    dim=(ctx.nworld, max(ctx.nq, ctx.nv)),
                    inputs=[d.qpos, d.qvel, grad_qpos_wp, grad_qvel_wp, loss],
                )
            tape.backward(loss=loss)
            wp.synchronize()

            # Accumulate ctrl grad and chain state grads
            total_grad_ctrl += wp.to_torch(d.ctrl.grad).clone()
            g_qpos = wp.to_torch(d.qpos.grad).clone()
            g_qvel = wp.to_torch(d.qvel.grad).clone()

            tape.zero()

        # 5. Sanitize and clamp
        total_grad_ctrl, g_qpos, g_qvel = _sanitize_and_clamp(
            total_grad_ctrl, g_qpos, g_qvel
        )

        # 6. Restore to post-step state
        _restore_and_rerun(
            m, d, ctx.saved_qpos, ctx.saved_qvel, ctx.saved_time,
            ctx.saved_act, ctrl_wp, substeps,
        )

        return total_grad_ctrl, g_qpos, g_qvel, None

    # ------------------------------------------------------------------
    # Mode 3: FD Jacobian (fallback for debugging / comparison)
    # ------------------------------------------------------------------

    @staticmethod
    def _backward_fd(ctx, grad_qpos_torch, grad_qvel_torch):
        env = ctx.env
        m = env.warp_model
        d = env.warp_data
        nworld = ctx.nworld
        nq = ctx.nq
        nv = ctx.nv
        substeps = env.substeps

        dt = wp.to_torch(m.opt.timestep).item()
        fd_eps = 1e-4
        fd_max_dqacc = 1.0
        # Per-substep gradient clamp. The FD Jacobian can have entries up to
        # fd_max_dqacc/fd_eps = 1e4. Over 16 substeps the amplification factor
        # is ~(1e4 * nv * dt) per substep ≈ 140x, causing exponential blowup
        # unless clamped tightly. A value of 1.0 keeps state gradients bounded
        # while preserving gradient direction (the correct single-step gradient
        # magnitude is O(0.01-0.1)).
        substep_grad_max = 1.0

        # Restore to initial state
        wp.copy(d.qpos, ctx.saved_qpos)
        wp.copy(d.qvel, ctx.saved_qvel)
        wp.copy(d.time, ctx.saved_time)
        if ctx.saved_act is not None:
            wp.copy(d.act, ctx.saved_act)
        ctrl_wp = wp.from_torch(ctx.ctrl_torch.contiguous())
        wp.copy(d.ctrl, ctrl_wp)
        wp.synchronize()

        # Save intermediate states for all substeps (including act for muscles)
        has_act = ctx.saved_act is not None
        states = []
        for s in range(substeps):
            act_snap = wp.clone(d.act) if has_act else None
            states.append((wp.clone(d.qpos), wp.clone(d.qvel), wp.clone(d.time), act_snap))
            mjw.step(m, d)
        wp.synchronize()

        # Current gradients w.r.t. post-final-substep state
        g_qpos = grad_qpos_torch.clone()
        g_qvel = grad_qvel_torch.clone()

        # Accumulate ctrl gradient across substeps
        total_grad_ctrl = torch.zeros_like(ctx.ctrl_torch)

        # Backward through substeps in reverse
        for s in reversed(range(substeps)):
            pre_qpos, pre_qvel, pre_time, pre_act = states[s]

            # Restore state to pre-substep (including muscle activation)
            wp.copy(d.qpos, pre_qpos)
            wp.copy(d.qvel, pre_qvel)
            wp.copy(d.time, pre_time)
            if pre_act is not None:
                wp.copy(d.act, pre_act)
            wp.copy(d.ctrl, ctrl_wp)
            wp.synchronize()

            # 1. Analytical Euler backward
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

            # 4. Finite-difference dynamics Jacobian
            qpos_view = wp.to_torch(d.qpos)
            qvel_view = wp.to_torch(d.qvel)
            fd_g_qpos = torch.zeros(nworld, nq, device=g_qpos.device)
            fd_g_qvel = torch.zeros(nworld, nv, device=g_qvel.device)

            for j in range(nq):
                qpos_view[:, j] += fd_eps
                mjw.forward(m, d)
                wp.synchronize()
                qacc_plus = wp.to_torch(d.qacc).clone()
                qpos_view[:, j] -= fd_eps

                dqacc_raw = qacc_plus - qacc_orig
                dqacc_raw = dqacc_raw.clamp(-fd_max_dqacc, fd_max_dqacc)
                dqacc = torch.nan_to_num(dqacc_raw / fd_eps, 0.0, 0.0, 0.0)
                fd_g_qpos[:, j] = (grad_qacc_torch * dqacc).sum(dim=-1)

            for j in range(nv):
                qvel_view[:, j] += fd_eps
                mjw.forward(m, d)
                wp.synchronize()
                qacc_plus = wp.to_torch(d.qacc).clone()
                qvel_view[:, j] -= fd_eps

                dqacc_raw = qacc_plus - qacc_orig
                dqacc_raw = dqacc_raw.clamp(-fd_max_dqacc, fd_max_dqacc)
                dqacc = torch.nan_to_num(dqacc_raw / fd_eps, 0.0, 0.0, 0.0)
                fd_g_qvel[:, j] = (grad_qacc_torch * dqacc).sum(dim=-1)

            # 5. Use Warp tape through fwd_actuation only
            wp.copy(d.qpos, pre_qpos)
            wp.copy(d.qvel, pre_qvel)
            wp.copy(d.time, pre_time)
            if pre_act is not None:
                wp.copy(d.act, pre_act)
            wp.copy(d.ctrl, ctrl_wp)
            wp.synchronize()

            mjw.fwd_position(m, d)
            mjw.fwd_velocity(m, d)
            wp.synchronize()

            d.ctrl.grad = wp.zeros_like(d.ctrl)

            loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            tape = wp.Tape()
            with tape:
                mjw.fwd_actuation(m, d)
                wp.launch(
                    _vjp_qfrc_kernel,
                    dim=(nworld, nv),
                    inputs=[d.qfrc_actuator, grad_qfrc_wp, loss],
                )
            tape.backward(loss=loss)
            wp.synchronize()

            total_grad_ctrl += wp.to_torch(d.ctrl.grad).clone()

            # 6. Propagate gradients
            g_qpos_prev = g_qpos.clone() + fd_g_qpos
            g_qvel_prev = g_qvel + g_qvel_from_qpos + fd_g_qvel
            g_qpos = g_qpos_prev.clamp(-substep_grad_max, substep_grad_max)
            g_qvel = g_qvel_prev.clamp(-substep_grad_max, substep_grad_max)

            tape.zero()

        # Sanitize and clamp (consistent with tape-all mode)
        total_grad_ctrl, grad_qpos_in, grad_qvel_in = _sanitize_and_clamp(
            total_grad_ctrl, g_qpos, g_qvel
        )

        # Restore to post-step state
        _restore_and_rerun(
            m, d, ctx.saved_qpos, ctx.saved_qvel, ctx.saved_time,
            ctx.saved_act, wp.from_torch(ctx.ctrl_torch.contiguous()), substeps,
        )

        return total_grad_ctrl, grad_qpos_in, grad_qvel_in, None
