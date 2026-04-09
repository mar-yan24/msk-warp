"""Compare ant reward-gradient quality across smooth-adjoint variants.

Measures the actual SHAC training signal d(reward)/d(ctrl) on a settled ant
contact state and compares Warp AD against centered finite differences.
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import warp as wp
import yaml

wp._src.utils.warn = lambda *a, **k: None

import mujoco_warp as mjw

from msk_warp import PACKAGE_ROOT
from msk_warp.bridge import WarpSimStep
from msk_warp.envs.ant import AntEnv


DEVICE = "cuda:0"
DEFAULT_NUM_ENVS = 4
DEFAULT_FD_EPS = 1e-4
DEFAULT_SETTLE_STEPS = 200
DEFAULT_CTRL_SCALE = 0.3


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


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def settle_env(env, steps, ctrl_val):
    with torch.no_grad():
        ctrl = torch.full((env.num_envs, env.num_actions), ctrl_val, device=DEVICE)
        ctrl_wp = wp.from_torch(ctrl.contiguous())
        for _ in range(steps):
            wp.copy(env.warp_data.ctrl, ctrl_wp)
            wp.synchronize()
            mjw.step(env.warp_model, env.warp_data)
        wp.synchronize()


def reward_from_state(env, qpos, qvel, actions):
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


def reward_from_step(env, ctrl_torch, qpos0, qvel0):
    qpos_out, qvel_out = WarpSimStep.apply(ctrl_torch, qpos0, qvel0, env)
    qpos_out = qpos_out.clamp(-100.0, 100.0)
    qvel_out = qvel_out.clamp(-100.0, 100.0)
    actions = torch.zeros(env.num_envs, env.num_actions, device=DEVICE)
    obs, rew = reward_from_state(env, qpos_out, qvel_out, actions)
    return obs, rew


def _constraint_state_name(state):
    return {
        0: "SATISFIED",
        1: "QUADRATIC",
        2: "LINEARNEG",
        3: "LINEARPOS",
        4: "CONE",
    }.get(int(state), f"STATE_{int(state)}")


def summarize_contact_state(env):
    data = env.warp_data
    nacon = int(data.nacon.numpy()[0])
    if nacon == 0:
        return {
            "constraint_contacts": 0,
            "active_contacts": 0,
            "normal_states": {},
            "friction_states": {},
            "mean_abs_normal_force": 0.0,
            "mean_abs_friction_force": 0.0,
        }

    contact_dim = data.contact.dim.numpy()
    contact_type = data.contact.type.numpy()
    contact_worldid = data.contact.worldid.numpy()
    contact_efc_address = data.contact.efc_address.numpy()
    efc_state = data.efc.state.numpy()
    efc_force = data.efc.force.numpy()

    normal_states = {}
    friction_states = {}
    normal_forces = []
    friction_forces = []
    constraint_contacts = 0
    active_contacts = 0

    for conid in range(nacon):
        if not (int(contact_type[conid]) & 1):
            continue

        constraint_contacts += 1
        condim = int(contact_dim[conid])
        worldid = int(contact_worldid[conid])
        row0 = int(contact_efc_address[conid, 0])
        if row0 >= 0:
            normal_state = _constraint_state_name(efc_state[worldid, row0])
            normal_states[normal_state] = normal_states.get(normal_state, 0) + 1
            normal_forces.append(abs(float(efc_force[worldid, row0])))
            if normal_state != "SATISFIED":
                active_contacts += 1

        if condim == 1:
            continue

        for dimid in range(1, 2 * (condim - 1)):
            row = int(contact_efc_address[conid, dimid])
            if row < 0:
                continue
            friction_state = _constraint_state_name(efc_state[worldid, row])
            friction_states[friction_state] = friction_states.get(friction_state, 0) + 1
            friction_forces.append(abs(float(efc_force[worldid, row])))

    return {
        "constraint_contacts": constraint_contacts,
        "active_contacts": active_contacts,
        "normal_states": normal_states,
        "friction_states": friction_states,
        "mean_abs_normal_force": float(np.mean(normal_forces)) if normal_forces else 0.0,
        "mean_abs_friction_force": float(np.mean(friction_forces)) if friction_forces else 0.0,
    }


def print_contact_state(summary, label):
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


def _resolve_cfg_path(cfg_path):
    if cfg_path is None:
        return None
    if os.path.isabs(cfg_path):
        return cfg_path
    pkg_path = PACKAGE_ROOT / cfg_path
    if pkg_path.exists():
        return str(pkg_path)
    return cfg_path


def rollout_snapshot(cfg_path, policy_path, num_envs, rollout_steps):
    cfg_path = _resolve_cfg_path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["params"]["general"].get("seed", 42))

    env_kwargs = dict(cfg["params"]["env"])
    env_kwargs.pop("name", None)
    env_kwargs.pop("num_actors", None)
    env_kwargs["num_envs"] = num_envs
    env_kwargs["device"] = DEVICE
    env_kwargs["no_grad"] = False

    env = AntEnv(**env_kwargs)
    checkpoint = torch.load(policy_path, weights_only=False)
    actor = checkpoint[0].to(DEVICE)
    obs_rms = checkpoint[3]
    if obs_rms is not None:
        obs_rms = obs_rms.to(DEVICE)

    obs = env.reset()
    with torch.no_grad():
        for _ in range(rollout_steps):
            obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
            actions = torch.tanh(actor(obs_in, deterministic=True))
            obs, _, _, _, _, _ = env.step(actions)

    return (
        wp.to_torch(env.warp_data.qpos).clone(),
        wp.to_torch(env.warp_data.qvel).clone(),
    )


def compare_branch(
    branch_name,
    model_name,
    branch_specs,
    num_envs,
    fd_eps,
    settle_steps,
    ctrl_scale,
    snapshot_qpos=None,
    snapshot_qvel=None,
):
    branch_kwargs = dict(branch_specs[branch_name])
    model_spec = MODEL_SPECS[model_name]

    print(f"\n{'=' * 78}")
    print(
        f"{model_name} | branch={branch_name} | model={model_spec['model_path']} "
        f"| substeps={model_spec['substeps']}"
    )
    print(f"{'=' * 78}")

    torch.manual_seed(42)
    ctrl_base = torch.randn(num_envs, 8, device=DEVICE) * ctrl_scale

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
    if snapshot_qpos is None or snapshot_qvel is None:
        settle_env(env, steps=settle_steps, ctrl_val=ctrl_scale)
        state_label = "settled"
    else:
        wp.copy(env.warp_data.qpos, wp.from_torch(snapshot_qpos.contiguous()))
        wp.copy(env.warp_data.qvel, wp.from_torch(snapshot_qvel.contiguous()))
        mjw.forward(env.warp_model, env.warp_data)
        wp.synchronize()
        state_label = "rollout"

    contact_summary = summarize_contact_state(env)
    print_contact_state(contact_summary, state_label)

    env.clear_grad()
    qpos0 = wp.to_torch(env.warp_data.qpos).clone()
    qvel0 = wp.to_torch(env.warp_data.qvel).clone()

    ctrl = ctrl_base.clone().requires_grad_(True)
    obs, rew = reward_from_step(env, ctrl, qpos0, qvel0)
    rew.mean().backward()
    tape_grad = ctrl.grad.detach().cpu().numpy()

    fd_grad = np.zeros((num_envs, env.num_actions), dtype=np.float32)
    for act_idx in range(env.num_actions):
        for sign in (1.0, -1.0):
            env.clear_grad()
            wp.copy(env.warp_data.qpos, wp.from_torch(qpos0.contiguous()))
            wp.copy(env.warp_data.qvel, wp.from_torch(qvel0.contiguous()))
            wp.synchronize()

            ctrl_shifted = ctrl_base.clone()
            ctrl_shifted[:, act_idx] += sign * fd_eps
            ctrl_wp = wp.from_torch(ctrl_shifted.contiguous())
            wp.copy(env.warp_data.ctrl, ctrl_wp)
            wp.synchronize()
            for _ in range(model_spec["substeps"]):
                mjw.step(env.warp_model, env.warp_data)
            wp.synchronize()

            qp = wp.to_torch(env.warp_data.qpos).clone()
            qv = wp.to_torch(env.warp_data.qvel).clone()
            actions = torch.zeros(num_envs, env.num_actions, device=DEVICE)
            _, rew_fd = reward_from_state(env, qp, qv, actions)
            rew_np = rew_fd.detach().cpu().numpy()
            if sign > 0:
                rew_plus = rew_np
            else:
                rew_minus = rew_np

        fd_grad[:, act_idx] = (rew_plus - rew_minus) / (2.0 * fd_eps)

    tape_mean = tape_grad.mean(axis=0)
    fd_mean = fd_grad.mean(axis=0)
    cosine = cosine_sim(tape_mean, fd_mean)
    tape_norm = float(np.linalg.norm(tape_mean))
    fd_norm = float(np.linalg.norm(fd_mean))
    ratio = tape_norm / (fd_norm + 1e-10)

    print(
        f"mean grad norms: tape={tape_norm:.6e} fd={fd_norm:.6e} "
        f"ratio={ratio:.4f} cosine={cosine:.4f}"
    )

    act_names = ["hip4", "ank4", "hip1", "ank1", "hip2", "ank2", "hip3", "ank3"]
    print("\nper-actuator mean gradient:")
    for act_name, tape_val, fd_val in zip(act_names, tape_mean, fd_mean):
        local_ratio = float(tape_val / (fd_val + 1e-10))
        print(
            f"  {act_name:>5}: tape={tape_val:+.6e} fd={fd_val:+.6e} "
            f"ratio={local_ratio:+.3f}"
        )

    print("\nper-env gradient norms:")
    for env_idx in range(num_envs):
        env_tape = float(np.linalg.norm(tape_grad[env_idx]))
        env_fd = float(np.linalg.norm(fd_grad[env_idx]))
        env_cosine = cosine_sim(tape_grad[env_idx], fd_grad[env_idx])
        print(
            f"  env {env_idx}: tape={env_tape:.6e} fd={env_fd:.6e} cosine={env_cosine:.4f}"
        )

    print("\nreward component gradients (env 0):")
    env.clear_grad()
    wp.copy(env.warp_data.qpos, wp.from_torch(qpos0.contiguous()))
    wp.copy(env.warp_data.qvel, wp.from_torch(qvel0.contiguous()))
    wp.synchronize()

    ctrl_components = ctrl_base.clone().requires_grad_(True)
    obs_components, _ = reward_from_step(env, ctrl_components, qpos0, qvel0)
    components = {
        "forward_vel": obs_components[:, 5],
        "up_reward": env.up_weight * obs_components[:, 27],
        "heading": env.heading_weight * obs_components[:, 28],
        "height": env.height_weight * (obs_components[:, 0] - 0.27),
    }
    component_norms = {}
    component_grads = {}
    for name, component in components.items():
        if ctrl_components.grad is not None:
            ctrl_components.grad.zero_()
        component[0].backward(retain_graph=True)
        grad = ctrl_components.grad[0].detach().cpu().numpy()
        component_norms[name] = float(np.linalg.norm(grad))
        component_grads[name] = grad.tolist()
        print(f"  {name:>11}: norm={np.linalg.norm(grad):.6e} grad={grad}")

    return {
        "model": model_name,
        "branch": branch_name,
        "state_label": state_label,
        "tape_norm": tape_norm,
        "fd_norm": fd_norm,
        "ratio": ratio,
        "cosine": cosine,
        "contact_summary": contact_summary,
        "reward_component_norms_env0": component_norms,
        "reward_component_grads_env0": component_grads,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ant reward gradients across adjoint variants")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ant16", "soft16"],
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


def main():
    args = parse_args()
    branch_specs = {
        name: dict(spec)
        for name, spec in BRANCH_SPECS.items()
    }
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

    snapshot_qpos = None
    snapshot_qvel = None
    if args.rollout_cfg is not None:
        snapshot_qpos, snapshot_qvel = rollout_snapshot(
            args.rollout_cfg,
            args.policy,
            args.num_envs,
            args.rollout_steps,
        )
        print(
            f"rollout snapshot captured from cfg={args.rollout_cfg} "
            f"policy={args.policy} steps={args.rollout_steps}"
        )

    results = []
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
                    snapshot_qpos=snapshot_qpos,
                    snapshot_qvel=snapshot_qvel,
                )
            )

    print(f"\n{'=' * 78}")
    print("Summary")
    print(f"{'=' * 78}")
    print(f"{'model':<12} {'branch':<12} {'tape_norm':>12} {'fd_norm':>12} {'ratio':>10} {'cosine':>10}")
    for row in results:
        print(
            f"{row['model']:<12} {row['branch']:<12} "
            f"{row['tape_norm']:>12.6e} {row['fd_norm']:>12.6e} "
            f"{row['ratio']:>10.4f} {row['cosine']:>10.4f}"
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
            "surrogate_alpha_override": args.surrogate_alpha,
            "branch_specs": branch_specs,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON summary to: {out_path}")


if __name__ == "__main__":
    main()
