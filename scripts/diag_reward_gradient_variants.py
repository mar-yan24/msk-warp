"""Compare ant reward-gradient quality across smooth-adjoint variants.

Measures the actual SHAC training signal on settled states or policy-visited
rollout snapshots and compares Warp AD against centered finite differences.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import warp as wp

wp._src.utils.warn = lambda *a, **k: None

import mujoco_warp as mjw

from msk_warp.bridge import WarpSimStep
from msk_warp.envs.ant import AntEnv
from msk_warp.utils.ant_rollout import capture_rollout_snapshots
from msk_warp.utils.ant_rollout import normalize_snapshot_steps


DEVICE = "cuda:0"
DEFAULT_NUM_ENVS = 4
DEFAULT_FD_EPS = 1e-4
DEFAULT_SETTLE_STEPS = 200
DEFAULT_CTRL_SCALE = 0.3
OBJECTIVE_NAMES = ("reward", "forward_vel", "state_probe")


BRANCH_SPECS = {
    "smooth": {
        "smooth_adjoint": True,
    },
    "bypass": {
        "smooth_adjoint": True,
        "friction_bypass_kf": 1.0,
    },
    "free_body": {
        "smooth_adjoint": True,
        "free_body_adjoint": True,
    },
    "penalty": {
        "smooth_adjoint": True,
        "penalty_damping_alpha": 0.1,
    },
    "surrogate": {
        "smooth_adjoint": True,
        "friction_surrogate_adjoint": True,
        "friction_surrogate_alpha": 0.9,
    },
}


MODEL_SPECS = {
    "ant16": {"model_path": "assets/ant.xml", "substeps": 16},
    "soft4": {"model_path": "assets/ant_soft.xml", "substeps": 4},
    "soft16": {"model_path": "assets/ant_soft.xml", "substeps": 16},
    "substeps4": {"model_path": "assets/ant_substeps4.xml", "substeps": 4},
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def make_probe_vector(
    length: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a deterministic normalized probe vector."""
    idx = torch.arange(length, device=device, dtype=dtype)
    vec = torch.sin(0.37 + 0.73 * idx) + 0.5 * torch.cos(1.13 + 0.29 * idx)
    return vec / vec.norm().clamp(min=torch.finfo(dtype).eps)


def compute_state_probe(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    qpos_probe = make_probe_vector(qpos.shape[1], device=qpos.device, dtype=qpos.dtype)
    qvel_probe = make_probe_vector(qvel.shape[1], device=qvel.device, dtype=qvel.dtype)
    return qpos @ qpos_probe + qvel @ qvel_probe


def settle_env(env: AntEnv, steps: int, ctrl_val: float) -> None:
    with torch.no_grad():
        ctrl = torch.full((env.num_envs, env.num_actions), ctrl_val, device=DEVICE)
        ctrl_wp = wp.from_torch(ctrl.contiguous())
        for _ in range(steps):
            wp.copy(env.warp_data.ctrl, ctrl_wp)
            wp.synchronize()
            for _ in range(env.substeps):
                mjw.step(env.warp_model, env.warp_data)
        wp.synchronize()


def reward_from_state(
    env: AntEnv,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs = AntEnv._compute_obs(
        qpos,
        qvel,
        actions,
        env.targets,
        env.up_vec,
        env.heading_vec,
        env.joint_vel_obs_scaling,
    )
    rew = AntEnv._compute_reward(
        obs,
        actions,
        env.action_penalty,
        env.forward_vel_weight,
        env.heading_weight,
        env.up_weight,
        env.height_weight,
        env.joint_vel_penalty,
        env.push_reward_weight,
    )
    return obs, rew


def objectives_from_state(
    env: AntEnv,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    actions = torch.zeros(env.num_envs, env.num_actions, device=qpos.device)
    obs, rew = reward_from_state(env, qpos, qvel, actions)
    return {
        "reward": rew,
        "forward_vel": obs[:, 5],
        "state_probe": compute_state_probe(qpos, qvel),
    }, obs


def step_objectives(
    env: AntEnv,
    ctrl_torch: torch.Tensor,
    qpos0: torch.Tensor,
    qvel0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos0, qvel0, env)
    qpos_out = qpos_out.clamp(-100.0, 100.0)
    qvel_out = qvel_out.clamp(-100.0, 100.0)
    objectives, obs = objectives_from_state(env, qpos_out, qvel_out)
    return qpos_out, qvel_out, objectives, obs


def _constraint_state_name(state: int) -> str:
    return {
        0: "SATISFIED",
        1: "QUADRATIC",
        2: "LINEARNEG",
        3: "LINEARPOS",
        4: "CONE",
    }.get(int(state), f"STATE_{int(state)}")


def _contact_row_count(condim: int, cone_type: int) -> int:
    if condim <= 1:
        return 1
    if cone_type == 0:
        return 2 * (condim - 1)
    return condim


def _row_kind(condim: int, cone_type: int, row_local_idx: int) -> str:
    if condim <= 1:
        return "normal"
    if cone_type == 0:
        return "pyramid"
    return "normal" if row_local_idx == 0 else "friction"


def summarize_contact_state(env: AntEnv) -> dict[str, Any]:
    data = env.warp_data
    nacon = int(data.nacon.numpy()[0])
    if nacon == 0:
        return {
            "constraint_contacts": 0,
            "active_contacts": 0,
            "normal_states": {},
            "friction_states": {},
            "pyramid_states": {},
            "mean_abs_normal_force": 0.0,
            "mean_abs_friction_force": 0.0,
            "mean_abs_pyramid_force": 0.0,
        }

    cone_type = int(env.mjm.opt.cone)
    contact_dim = data.contact.dim.numpy()
    contact_type = data.contact.type.numpy()
    contact_worldid = data.contact.worldid.numpy()
    contact_efc_address = data.contact.efc_address.numpy()
    efc_state = data.efc.state.numpy()
    efc_force = data.efc.force.numpy()

    normal_states: dict[str, int] = {}
    friction_states: dict[str, int] = {}
    pyramid_states: dict[str, int] = {}
    normal_forces: list[float] = []
    friction_forces: list[float] = []
    pyramid_forces: list[float] = []
    constraint_contacts = 0
    active_contacts = 0

    for conid in range(nacon):
        if not (int(contact_type[conid]) & 1):
            continue

        condim = int(contact_dim[conid])
        row_count = _contact_row_count(condim, cone_type)
        worldid = int(contact_worldid[conid])
        row0 = int(contact_efc_address[conid, 0])
        if row0 < 0:
            continue

        constraint_contacts += 1
        contact_active = False
        for j in range(row_count):
            row = int(contact_efc_address[conid, j]) if j < contact_efc_address.shape[1] else -1
            if row < 0:
                continue
            state = _constraint_state_name(efc_state[worldid, row])
            force_abs = abs(float(efc_force[worldid, row]))
            kind = _row_kind(condim, cone_type, j)

            if state != "SATISFIED":
                contact_active = True

            if kind == "normal":
                normal_states[state] = normal_states.get(state, 0) + 1
                normal_forces.append(force_abs)
            elif kind == "friction":
                friction_states[state] = friction_states.get(state, 0) + 1
                friction_forces.append(force_abs)
            else:
                pyramid_states[state] = pyramid_states.get(state, 0) + 1
                pyramid_forces.append(force_abs)

        if contact_active:
            active_contacts += 1

    return {
        "constraint_contacts": constraint_contacts,
        "active_contacts": active_contacts,
        "normal_states": normal_states,
        "friction_states": friction_states,
        "pyramid_states": pyramid_states,
        "mean_abs_normal_force": float(np.mean(normal_forces)) if normal_forces else 0.0,
        "mean_abs_friction_force": float(np.mean(friction_forces)) if friction_forces else 0.0,
        "mean_abs_pyramid_force": float(np.mean(pyramid_forces)) if pyramid_forces else 0.0,
    }


def print_contact_state(summary: dict[str, Any], label: str) -> None:
    print(f"\ncontact state summary ({label}):")
    print(
        f"  constraint_contacts={summary['constraint_contacts']} "
        f"active_contacts={summary['active_contacts']}"
    )
    print(
        f"  mean|normal_force|={summary['mean_abs_normal_force']:.6e} "
        f"mean|friction_force|={summary['mean_abs_friction_force']:.6e}"
    )
    print(f"  normal states:   {summary['normal_states']}")
    print(f"  friction states: {summary['friction_states']}")


def summarize_gradient_match(
    tape_grad: np.ndarray,
    fd_grad: np.ndarray,
) -> dict[str, Any]:
    tape_mean = tape_grad.mean(axis=0)
    fd_mean = fd_grad.mean(axis=0)
    tape_norm = float(np.linalg.norm(tape_mean))
    fd_norm = float(np.linalg.norm(fd_mean))
    ratio = tape_norm / (fd_norm + 1e-10)
    return {
        "tape_norm": tape_norm,
        "fd_norm": fd_norm,
        "ratio": float(ratio),
        "cosine": cosine_sim(tape_mean, fd_mean),
        "tape_mean": tape_mean.tolist(),
        "fd_mean": fd_mean.tolist(),
        "per_env": [
            {
                "env": int(env_idx),
                "tape_norm": float(np.linalg.norm(tape_grad[env_idx])),
                "fd_norm": float(np.linalg.norm(fd_grad[env_idx])),
                "cosine": cosine_sim(tape_grad[env_idx], fd_grad[env_idx]),
            }
            for env_idx in range(tape_grad.shape[0])
        ],
    }


def _finite_difference_gradients(
    env: AntEnv,
    ctrl_base: torch.Tensor,
    qpos0: torch.Tensor,
    qvel0: torch.Tensor,
    fd_eps: float,
) -> dict[str, np.ndarray]:
    fd_grads = {
        name: np.zeros((env.num_envs, env.num_actions), dtype=np.float32)
        for name in OBJECTIVE_NAMES
    }

    for act_idx in range(env.num_actions):
        plus_values: dict[str, np.ndarray] = {}
        minus_values: dict[str, np.ndarray] = {}
        for sign in (1.0, -1.0):
            env.clear_grad()
            wp.copy(env.warp_data.qpos, wp.from_torch(qpos0.contiguous()))
            wp.copy(env.warp_data.qvel, wp.from_torch(qvel0.contiguous()))
            wp.synchronize()

            ctrl_shifted = ctrl_base.clone()
            ctrl_shifted[:, act_idx] += sign * fd_eps
            wp.copy(env.warp_data.ctrl, wp.from_torch(ctrl_shifted.contiguous()))
            wp.synchronize()
            for _ in range(env.substeps):
                mjw.step(env.warp_model, env.warp_data)
            wp.synchronize()

            qp = wp.to_torch(env.warp_data.qpos).clone()
            qv = wp.to_torch(env.warp_data.qvel).clone()
            objective_values, _ = objectives_from_state(env, qp, qv)
            stash = plus_values if sign > 0 else minus_values
            for name, values in objective_values.items():
                stash[name] = values.detach().cpu().numpy()

        for name in OBJECTIVE_NAMES:
            fd_grads[name][:, act_idx] = (plus_values[name] - minus_values[name]) / (2.0 * fd_eps)

    return fd_grads


def _snapshot_state_label(snapshot: dict[str, Any] | None) -> str:
    if snapshot is None:
        return "settled"
    return f"{snapshot['source']}_step_{int(snapshot['step'])}"


def compare_branch(
    branch_name: str,
    model_name: str,
    branch_specs: dict[str, dict[str, Any]],
    num_envs: int,
    fd_eps: float,
    settle_steps: int,
    ctrl_scale: float,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    branch_kwargs = dict(branch_specs[branch_name])
    model_spec = MODEL_SPECS[model_name]
    state_label = _snapshot_state_label(snapshot)

    print(f"\n{'=' * 78}")
    print(
        f"{model_name} | branch={branch_name} | model={model_spec['model_path']} "
        f"| substeps={model_spec['substeps']} | state={state_label}"
    )
    print(f"{'=' * 78}")

    torch.manual_seed(42)
    env = AntEnv(
        num_envs=num_envs,
        device=DEVICE,
        substeps=model_spec["substeps"],
        tape_per_substep=True,
        use_fd_jacobian=False,
        stochastic_init=False,
        model_path=model_spec["model_path"],
        **branch_kwargs,
    )
    env.reset()

    if snapshot is None:
        settle_env(env, steps=settle_steps, ctrl_val=ctrl_scale)
        state_info = {"source": "settled", "settle_steps": int(settle_steps)}
    else:
        wp.copy(env.warp_data.qpos, wp.from_torch(snapshot["qpos"].contiguous()))
        wp.copy(env.warp_data.qvel, wp.from_torch(snapshot["qvel"].contiguous()))
        mjw.forward(env.warp_model, env.warp_data)
        wp.synchronize()
        state_info = {
            "source": snapshot["source"],
            "step": int(snapshot["step"]),
        }

    contact_summary = summarize_contact_state(env)
    print_contact_state(contact_summary, state_label)

    env.clear_grad()
    qpos0 = wp.to_torch(env.warp_data.qpos).clone()
    qvel0 = wp.to_torch(env.warp_data.qvel).clone()
    ctrl_base = torch.randn(num_envs, env.num_actions, device=DEVICE) * ctrl_scale

    ctrl = ctrl_base.clone().requires_grad_(True)
    _, _, objectives, _ = step_objectives(env, ctrl, qpos0, qvel0)
    tape_grads: dict[str, np.ndarray] = {}
    objective_names = list(OBJECTIVE_NAMES)
    for idx, name in enumerate(objective_names):
        if ctrl.grad is not None:
            ctrl.grad.zero_()
        objectives[name].sum().backward(retain_graph=idx < len(objective_names) - 1)
        tape_grads[name] = ctrl.grad.detach().cpu().numpy().copy()

    fd_grads = _finite_difference_gradients(env, ctrl_base, qpos0, qvel0, fd_eps)

    objective_reports: dict[str, Any] = {}
    for name in OBJECTIVE_NAMES:
        report = summarize_gradient_match(tape_grads[name], fd_grads[name])
        values = objectives[name].detach()
        report["value_mean"] = float(values.mean().item())
        report["value_std"] = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
        objective_reports[name] = report
        print(
            f"{name:>12}: tape={report['tape_norm']:.6e} fd={report['fd_norm']:.6e} "
            f"ratio={report['ratio']:.4f} cosine={report['cosine']:.4f} "
            f"value_mean={report['value_mean']:.6e}"
        )

    return {
        "model": model_name,
        "branch": branch_name,
        "state_label": state_label,
        "state": state_info,
        "contact_summary": contact_summary,
        "objectives": objective_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ant reward gradients across adjoint variants")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["soft4", "substeps4"],
        choices=sorted(MODEL_SPECS.keys()),
        help="Model/substep setups to evaluate",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["smooth", "bypass", "free_body", "penalty", "surrogate"],
        choices=sorted(BRANCH_SPECS.keys()),
        help="Smooth-adjoint variants to compare",
    )
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--fd-eps", type=float, default=DEFAULT_FD_EPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--ctrl-scale", type=float, default=DEFAULT_CTRL_SCALE)
    parser.add_argument(
        "--rollout-cfg",
        type=str,
        default=None,
        help="Optional training config used to capture a real policy rollout state",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Checkpoint to use with --rollout-cfg for snapshot generation",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=64,
        help="Number of deterministic policy rollout steps before capturing state",
    )
    parser.add_argument(
        "--rollout-step-list",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of rollout snapshot steps to evaluate",
    )
    parser.add_argument(
        "--surrogate-alpha",
        type=float,
        default=None,
        help="Optional override for surrogate branch friction_surrogate_alpha",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write machine-readable summary output",
    )
    return parser.parse_args()


def main() -> None:
    wp.init()
    args = parse_args()
    branch_specs = {name: dict(spec) for name, spec in BRANCH_SPECS.items()}
    if args.surrogate_alpha is not None:
        if args.surrogate_alpha < 0.0 or args.surrogate_alpha > 1.0:
            raise ValueError("--surrogate-alpha must be in [0, 1]")
        branch_specs["surrogate"]["friction_surrogate_alpha"] = args.surrogate_alpha

    print("Reward Gradient Variant Diagnostic")
    print("=================================")
    print(f"models={args.models}")
    print(f"branches={args.branches}")
    print(
        f"num_envs={args.num_envs} fd_eps={args.fd_eps} "
        f"settle_steps={args.settle_steps} ctrl_scale={args.ctrl_scale}"
    )
    if (args.rollout_cfg is None) != (args.policy is None):
        raise ValueError("--rollout-cfg and --policy must be provided together")
    if args.surrogate_alpha is not None:
        print(f"surrogate_alpha_override={args.surrogate_alpha}")

    snapshots: list[dict[str, Any] | None] = [None]
    if args.rollout_cfg is not None:
        snapshot_steps = normalize_snapshot_steps(
            args.rollout_step_list,
            fallback_step=args.rollout_steps,
        )
        snapshots = capture_rollout_snapshots(
            args.rollout_cfg,
            args.policy,
            device=DEVICE,
            snapshot_steps=snapshot_steps,
            num_envs=args.num_envs,
        )
        print(
            f"rollout snapshots captured from cfg={args.rollout_cfg} "
            f"policy={args.policy} steps={snapshot_steps}"
        )

    results = []
    for snapshot in snapshots:
        for model_name in args.models:
            for branch_name in args.branches:
                results.append(
                    compare_branch(
                        branch_name=branch_name,
                        model_name=model_name,
                        branch_specs=branch_specs,
                        num_envs=args.num_envs,
                        fd_eps=args.fd_eps,
                        settle_steps=args.settle_steps,
                        ctrl_scale=args.ctrl_scale,
                        snapshot=snapshot,
                    )
                )

    print(f"\n{'=' * 108}")
    print("Summary")
    print(f"{'=' * 108}")
    print(
        f"{'state':<18} {'model':<12} {'branch':<12} "
        f"{'reward_cos':>10} {'fwd_cos':>10} {'probe_cos':>10} "
        f"{'reward_ratio':>12} {'probe_ratio':>12}"
    )
    for row in results:
        reward = row["objectives"]["reward"]
        forward = row["objectives"]["forward_vel"]
        probe = row["objectives"]["state_probe"]
        print(
            f"{row['state_label']:<18} {row['model']:<12} {row['branch']:<12} "
            f"{reward['cosine']:>10.4f} {forward['cosine']:>10.4f} {probe['cosine']:>10.4f} "
            f"{reward['ratio']:>12.4f} {probe['ratio']:>12.4f}"
        )

    if args.output_json is not None:
        out_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "models": args.models,
            "branches": args.branches,
            "num_envs": args.num_envs,
            "fd_eps": args.fd_eps,
            "settle_steps": args.settle_steps,
            "ctrl_scale": args.ctrl_scale,
            "rollout_cfg": args.rollout_cfg,
            "policy": args.policy,
            "rollout_steps": args.rollout_steps,
            "rollout_step_list": args.rollout_step_list,
            "surrogate_alpha_override": args.surrogate_alpha,
            "branch_specs": branch_specs,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON summary to: {out_path}")


if __name__ == "__main__":
    main()
