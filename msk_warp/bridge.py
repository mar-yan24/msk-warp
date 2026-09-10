"""Gradient bridge between MuJoCo Warp and PyTorch autograd.

``WarpSimStep`` is a ``torch.autograd.Function`` that advances the simulation by
``env.substeps`` physics steps. Its differentiable inputs are ``ctrl`` and the state
``(qpos, qvel, act)``; its outputs are the next state, so gradients flow both through the
actuation path (policy -> ctrl -> dynamics) and through the dynamics path (state -> next
state, BPTT). Muscle activation ``act`` is part of the state because for Hill-type actuators
``ctrl`` only reaches the dynamics through ``act_dot -> act`` on the following step.

Backward modes (``env.backward_mode``):

``tape_per_substep``  one Warp tape per physics step, output adjoint seeded through a VJP
                      kernel and chained back (production mode; cheapest on the current
                      backend because per-step intermediates are released each step)
``tape``              one Warp tape over all substeps (validated identical to the chained
                      mode on the PR #1423 backend; keeps more intermediates alive)
``fd``                central finite differences of the one-step map, applied as a
                      vector-Jacobian product per substep (the control: slow, backend-agnostic)

All modes restore ``Data`` to the pre-step checkpoint, replay, and (unless
``env.rerun_after_backward`` is False) leave ``Data`` at the post-step state afterwards.
"""

from __future__ import annotations

import mujoco_warp as mjw
import torch
import warp as wp

from msk_warp.backend import quiet_nograd_kernels

STATE_FIELDS = ("qpos", "qvel", "act")
MODES = ("tape_per_substep", "tape", "fd")

# Gradients returned by the bridge are NaN-cleaned and clamped; the count of cleaned entries
# since process start is exposed for tests (a healthy backend produces zero).
sanitized_nan_count = 0
GRAD_CLAMP = 1.0e4


@wp.kernel
def _vjp_state_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    act: wp.array2d(dtype=float),
    g_qpos: wp.array2d(dtype=float),
    g_qvel: wp.array2d(dtype=float),
    g_act: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
):
    """loss = sum(qpos * g_qpos + qvel * g_qvel + act * g_act): seeds the tape with the incoming adjoint."""
    w, j = wp.tid()
    if j < qpos.shape[1]:
        wp.atomic_add(loss, 0, qpos[w, j] * g_qpos[w, j])
    if j < qvel.shape[1]:
        wp.atomic_add(loss, 0, qvel[w, j] * g_qvel[w, j])
    if j < act.shape[1]:
        wp.atomic_add(loss, 0, act[w, j] * g_act[w, j])


def _wp_from(t: torch.Tensor) -> wp.array:
    return wp.from_torch(t.detach().contiguous())


def _write_state(d, qpos, qvel, act, ctrl=None) -> None:
    """Copy torch tensors into the *current* Data arrays (handles are re-read every call)."""
    wp.copy(d.qpos, _wp_from(qpos))
    wp.copy(d.qvel, _wp_from(qvel))
    if d.act.size > 0:
        wp.copy(d.act, _wp_from(act))
    if ctrl is not None and d.ctrl.size > 0:
        wp.copy(d.ctrl, _wp_from(ctrl))


def _snapshot(d) -> dict:
    return {
        "qpos": wp.clone(d.qpos),
        "qvel": wp.clone(d.qvel),
        "act": wp.clone(d.act) if d.act.size > 0 else None,
        "time": wp.clone(d.time),
    }


def _restore(d, snap: dict, ctrl_wp=None) -> None:
    wp.copy(d.qpos, snap["qpos"])
    wp.copy(d.qvel, snap["qvel"])
    if snap["act"] is not None:
        wp.copy(d.act, snap["act"])
    wp.copy(d.time, snap["time"])
    if ctrl_wp is not None and d.ctrl.size > 0:
        wp.copy(d.ctrl, ctrl_wp)


def _state_tensors(d) -> tuple:
    qpos = wp.to_torch(d.qpos).clone()
    qvel = wp.to_torch(d.qvel).clone()
    act = wp.to_torch(d.act).clone() if d.act.size > 0 else torch.zeros((d.qpos.shape[0], 0), device=qpos.device, dtype=qpos.dtype)
    return qpos, qvel, act


def _grad_tensor(arr: wp.array, like: torch.Tensor) -> torch.Tensor:
    if arr is None or arr.size == 0 or arr.grad is None:
        return torch.zeros_like(like)
    return wp.to_torch(arr.grad).clone()


def _sanitize(*tensors):
    global sanitized_nan_count
    out = []
    for t in tensors:
        bad = ~torch.isfinite(t)
        if bad.any():
            sanitized_nan_count += int(bad.sum().item())
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        out.append(t.clamp(-GRAD_CLAMP, GRAD_CLAMP))
    return tuple(out)


def _zeros_act(nworld, d, device):
    return torch.zeros((nworld, d.act.shape[1] if d.act.size > 0 else 0), device=device, dtype=torch.float32)


class WarpSimStep(torch.autograd.Function):
    """Differentiable ``env.substeps`` x ``mjw.step`` with state ``(qpos, qvel, act)``."""

    @staticmethod
    def forward(ctx, ctrl, qpos_in, qvel_in, act_in, env):
        m, d = env.warp_model, env.warp_data
        _write_state(d, qpos_in, qvel_in, act_in)
        ctx.snap = _snapshot(d)  # pre-step checkpoint, before ctrl is applied
        ctx.ctrl = ctrl.detach().clone()
        if d.ctrl.size > 0:
            wp.copy(d.ctrl, _wp_from(ctx.ctrl))
        for _ in range(env.substeps):
            mjw.step(m, d)
        wp.synchronize()
        ctx.env = env
        qpos_out, qvel_out, act_out = _state_tensors(d)
        return qpos_out, qvel_out, act_out

    @staticmethod
    def backward(ctx, g_qpos, g_qvel, g_act):
        env = ctx.env
        mode = getattr(env, "backward_mode", "tape_per_substep")
        if mode not in MODES:
            raise ValueError(f"unknown backward_mode {mode!r}; expected one of {MODES}")
        m, d = env.warp_model, env.warp_data
        ctrl_wp = _wp_from(ctx.ctrl) if d.ctrl.size > 0 else None
        nworld = g_qpos.shape[0]
        g_act = g_act if g_act is not None else _zeros_act(nworld, d, g_qpos.device)
        g_qpos, g_qvel, g_act = g_qpos.contiguous(), g_qvel.contiguous(), g_act.contiguous()

        if mode == "tape":
            grads = _backward_tape(m, d, env.substeps, ctx.snap, ctrl_wp, ctx.ctrl, g_qpos, g_qvel, g_act)
        elif mode == "tape_per_substep":
            grads = _backward_tape_per_substep(m, d, env.substeps, ctx.snap, ctrl_wp, ctx.ctrl, g_qpos, g_qvel, g_act)
        else:
            grads = _backward_fd(m, d, env.substeps, ctx.snap, ctrl_wp, ctx.ctrl, g_qpos, g_qvel, g_act, eps=getattr(env, "fd_eps", 1e-3))

        grad_ctrl, grad_qpos, grad_qvel, grad_act = _sanitize(*grads)
        if getattr(env, "rerun_after_backward", True):
            _restore(d, ctx.snap, ctrl_wp)
            for _ in range(env.substeps):
                mjw.step(m, d)
            wp.synchronize()
        return grad_ctrl, grad_qpos, grad_qvel, grad_act, None


# --------------------------------------------------------------------------- modes


def _taped_backward(m, d, nsteps, g_qpos, g_qvel, g_act):
    """Record ``nsteps`` steps on one tape from the current state, seed with the adjoint, backprop.

    Returns gradients w.r.t. the state/ctrl arrays that were current when called (captured before
    the tape because the backend rebinds them).
    """
    refs = {f: getattr(d, f) for f in ("qpos", "qvel", "act", "ctrl")}
    nworld = g_qpos.shape[0]
    n = max(d.qpos.shape[1], d.qvel.shape[1], d.act.shape[1] if d.act.size > 0 else 0)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    try:
        with tape:
            for _ in range(nsteps):
                mjw.step(m, d)
            act_arr = d.act if d.act.size > 0 else wp.zeros((nworld, 0), dtype=float)
            g_act_arr = _wp_from(g_act) if d.act.size > 0 else wp.zeros((nworld, 0), dtype=float)
            wp.launch(_vjp_state_kernel, dim=(nworld, n), inputs=[d.qpos, d.qvel, act_arr, _wp_from(g_qpos), _wp_from(g_qvel), g_act_arr, loss])
        with quiet_nograd_kernels():
            tape.backward(loss=loss)
        wp.synchronize()
        out = (
            _grad_tensor(refs["ctrl"], torch.zeros((nworld, d.ctrl.shape[1]), device=g_qpos.device)),
            _grad_tensor(refs["qpos"], g_qpos),
            _grad_tensor(refs["qvel"], g_qvel),
            _grad_tensor(refs["act"], g_act),
        )
    finally:
        tape.zero()
        del tape
    return out


def _backward_tape(m, d, substeps, snap, ctrl_wp, ctrl_t, g_qpos, g_qvel, g_act):
    _restore(d, snap, ctrl_wp)
    wp.synchronize()
    return _taped_backward(m, d, substeps, g_qpos, g_qvel, g_act)


def _backward_tape_per_substep(m, d, substeps, snap, ctrl_wp, ctrl_t, g_qpos, g_qvel, g_act):
    # replay once to record every substep's input state
    _restore(d, snap, ctrl_wp)
    states = []
    for _ in range(substeps):
        states.append(_snapshot(d))
        mjw.step(m, d)
    wp.synchronize()
    grad_ctrl = torch.zeros_like(ctrl_t)
    for s in reversed(range(substeps)):
        _restore(d, states[s], ctrl_wp)
        wp.synchronize()
        gc, g_qpos, g_qvel, g_act = _taped_backward(m, d, 1, g_qpos, g_qvel, g_act)
        grad_ctrl += gc
    return grad_ctrl, g_qpos, g_qvel, g_act


def _backward_fd(m, d, substeps, snap, ctrl_wp, ctrl_t, g_qpos, g_qvel, g_act, eps=1e-3):
    """Finite-difference vector-Jacobian products of the one-step map, chained over substeps."""
    _restore(d, snap, ctrl_wp)
    states = []
    for _ in range(substeps):
        states.append(_snapshot(d))
        mjw.step(m, d)
    wp.synchronize()
    device = g_qpos.device
    nworld = g_qpos.shape[0]
    grad_ctrl = torch.zeros_like(ctrl_t)

    def step_from(state_t: dict, ctrl_val: torch.Tensor):
        _restore(d, state_t["snap"])
        _write_state(d, state_t["qpos"], state_t["qvel"], state_t["act"], ctrl_val)
        mjw.step(m, d)
        wp.synchronize()
        qpos_o, qvel_o, act_o = _state_tensors(d)
        return qpos_o, qvel_o, act_o

    for s in reversed(range(substeps)):
        snap_s = states[s]
        base = {
            "snap": snap_s,
            "qpos": wp.to_torch(snap_s["qpos"]).clone(),
            "qvel": wp.to_torch(snap_s["qvel"]).clone(),
            "act": wp.to_torch(snap_s["act"]).clone() if snap_s["act"] is not None else torch.zeros((nworld, 0), device=device),
        }
        ctrl_base = ctrl_t.clone()

        def contract(qpos_o, qvel_o, act_o):
            v = (qpos_o * g_qpos).sum(dim=1) + (qvel_o * g_qvel).sum(dim=1)
            if act_o.shape[1] > 0:
                v = v + (act_o * g_act).sum(dim=1)
            return v

        new = {}
        for f in ("qpos", "qvel", "act"):
            n = base[f].shape[1]
            g = torch.zeros((nworld, n), device=device)
            for j in range(n):
                plus = dict(base)
                plus[f] = base[f].clone()
                plus[f][:, j] += eps
                lp = contract(*step_from(plus, ctrl_base))
                minus = dict(base)
                minus[f] = base[f].clone()
                minus[f][:, j] -= eps
                lm = contract(*step_from(minus, ctrl_base))
                g[:, j] = (lp - lm) / (2.0 * eps)
            new[f] = g
        gcs = torch.zeros_like(ctrl_t)
        for j in range(ctrl_t.shape[1]):
            cp = ctrl_base.clone()
            cp[:, j] += eps
            lp = contract(*step_from(base, cp))
            cm = ctrl_base.clone()
            cm[:, j] -= eps
            lm = contract(*step_from(base, cm))
            gcs[:, j] = (lp - lm) / (2.0 * eps)
        grad_ctrl += gcs
        g_qpos, g_qvel, g_act = new["qpos"], new["qvel"], new["act"]
    return grad_ctrl, g_qpos, g_qvel, g_act
