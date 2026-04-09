"""Compare Warp vs native MuJoCo contact/constraint state on a matched ant state.

This diagnostic exists to isolate friction/contact representation drift that can
poison SHAC reward gradients under contact-heavy locomotion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
import yaml

import mujoco_warp as mjw

from msk_warp import PACKAGE_ROOT, resolve_model_path
from msk_warp.envs.ant import AntEnv


STATE_NAMES = {
    int(mujoco.mjtConstraintState.mjCNSTRSTATE_SATISFIED): "SATISFIED",
    int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC): "QUADRATIC",
    int(mujoco.mjtConstraintState.mjCNSTRSTATE_LINEARNEG): "LINEARNEG",
    int(mujoco.mjtConstraintState.mjCNSTRSTATE_LINEARPOS): "LINEARPOS",
    int(mujoco.mjtConstraintState.mjCNSTRSTATE_CONE): "CONE",
}


def _state_name(value: int) -> str:
    return STATE_NAMES.get(int(value), f"STATE_{int(value)}")


def _cone_friction_rows(condim: int, cone_type: int) -> int:
    if condim <= 1:
        return 0
    if cone_type == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        return 2 * (condim - 1)
    return condim - 1


def _contact_row_count(condim: int, cone_type: int) -> int:
    if condim <= 1:
        return 1
    if cone_type == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        return 2 * (condim - 1)
    return condim


def _row_kind(condim: int, cone_type: int, row_local_idx: int) -> str:
    if condim <= 1:
        return "normal"
    if cone_type == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        # Pyramidal rows are opposing cone-edge constraints, not explicit
        # normal/friction rows.
        return "pyramid"
    return "normal" if row_local_idx == 0 else "friction"


def _resolve_cfg_path(cfg_path: str) -> str:
    p = Path(cfg_path)
    if p.is_absolute():
        return str(p)
    pkg_path = PACKAGE_ROOT / cfg_path
    if pkg_path.exists():
        return str(pkg_path)
    return cfg_path


def _load_cfg(cfg_path: str) -> dict[str, Any]:
    with open(_resolve_cfg_path(cfg_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _env_from_cfg(cfg: dict[str, Any], device: str, no_grad: bool) -> AntEnv:
    env_kwargs = dict(cfg["params"]["env"])
    env_kwargs.pop("name", None)
    env_kwargs.pop("num_actors", None)
    env_kwargs["num_envs"] = 1
    env_kwargs["device"] = device
    env_kwargs["no_grad"] = no_grad
    return AntEnv(**env_kwargs)


def _snapshot_from_rollout(
    cfg_path: str,
    policy_path: str,
    device: str,
    rollout_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    cfg = _load_cfg(cfg_path)
    env = _env_from_cfg(cfg, device=device, no_grad=False)

    checkpoint = torch.load(policy_path, weights_only=False)
    actor = checkpoint[0].to(device)
    obs_rms = checkpoint[3]
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    actor.eval()

    obs = env.reset()
    actions = torch.zeros((1, env.num_actions), dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(rollout_steps):
            obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
            actions = torch.tanh(actor(obs_in, deterministic=True))
            obs, _, _, _, _, _ = env.step(actions)

    return (
        wp.to_torch(env.warp_data.qpos).detach().clone(),
        wp.to_torch(env.warp_data.qvel).detach().clone(),
        actions.detach().clone(),
        {
            "source": "rollout",
            "rollout_cfg": cfg_path,
            "policy": policy_path,
            "rollout_steps": int(rollout_steps),
        },
    )


def _snapshot_from_settle(
    cfg_path: str,
    device: str,
    settle_steps: int,
    ctrl_val: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    cfg = _load_cfg(cfg_path)
    env = _env_from_cfg(cfg, device=device, no_grad=True)

    actions = torch.full(
        (1, env.num_actions), float(ctrl_val), dtype=torch.float32, device=device
    )
    ctrl = actions * float(env.action_strength)
    ctrl_wp = wp.from_torch(ctrl.contiguous())

    with torch.no_grad():
        env.reset()
        for _ in range(settle_steps):
            wp.copy(env.warp_data.ctrl, ctrl_wp)
            wp.synchronize()
            for _ in range(env.substeps):
                mjw.step(env.warp_model, env.warp_data)
        wp.synchronize()

    return (
        wp.to_torch(env.warp_data.qpos).detach().clone(),
        wp.to_torch(env.warp_data.qvel).detach().clone(),
        actions.detach().clone(),
        {
            "source": "settled",
            "settle_steps": int(settle_steps),
            "ctrl_val": float(ctrl_val),
        },
    )


def _count_dict_add(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _native_efc_state_view(mjd: mujoco.MjData) -> tuple[np.ndarray, str]:
    nefc = int(mjd.nefc)
    # MuJoCo islanded solves keep authoritative per-constraint states in iefc_state.
    if hasattr(mjd, "iefc_state"):
        island_state = np.asarray(mjd.iefc_state)
        if island_state.shape[0] >= nefc:
            return island_state, "iefc_state"
    return np.asarray(mjd.efc_state), "efc_state"


def summarize_warp_contact_state(
    env: AntEnv,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = env.warp_data
    nacon = int(data.nacon.numpy()[0])
    cone_type = int(env.mjm.opt.cone)

    row_state_counts: dict[str, int] = {}
    normal_states: dict[str, int] = {}
    friction_states: dict[str, int] = {}
    pyramid_states: dict[str, int] = {}
    row_forces: list[float] = []
    normal_forces: list[float] = []
    friction_forces: list[float] = []
    pyramid_forces: list[float] = []
    rows_total = 0
    rows_normal = 0
    rows_friction = 0
    rows_pyramid = 0
    rows_active = 0
    constraint_contacts = 0
    active_contacts = 0
    contacts: list[dict[str, Any]] = []

    contact_dim = data.contact.dim.numpy()
    contact_type = data.contact.type.numpy()
    contact_geom = data.contact.geom.numpy()
    contact_worldid = data.contact.worldid.numpy()
    contact_efc_address = data.contact.efc_address.numpy()
    efc_state = data.efc.state.numpy()
    efc_force = data.efc.force.numpy()
    try:
        ct_constraint = int(getattr(getattr(mjw, "types"), "ContactType").CONSTRAINT)
    except Exception:
        ct_constraint = 1

    for conid in range(nacon):
        if not (int(contact_type[conid]) & ct_constraint):
            continue

        condim = int(contact_dim[conid])
        row_count = _contact_row_count(condim, cone_type)
        worldid = int(contact_worldid[conid])
        geom1 = int(contact_geom[conid][0])
        geom2 = int(contact_geom[conid][1])

        row0 = int(contact_efc_address[conid, 0])
        if row0 < 0:
            continue

        constraint_contacts += 1
        contact_rows: list[dict[str, Any]] = []

        contact_has_active_row = False
        for j in range(row_count):
            row = int(contact_efc_address[conid, j]) if j < contact_efc_address.shape[1] else -1
            if row < 0:
                continue
            state = _state_name(int(efc_state[worldid, row]))
            force_abs = abs(float(efc_force[worldid, row]))
            kind = _row_kind(condim, cone_type, j)

            _count_dict_add(row_state_counts, state)
            row_forces.append(force_abs)
            rows_total += 1
            if state != "SATISFIED":
                rows_active += 1
                contact_has_active_row = True

            if kind == "normal":
                rows_normal += 1
                _count_dict_add(normal_states, state)
                normal_forces.append(force_abs)
            elif kind == "friction":
                rows_friction += 1
                _count_dict_add(friction_states, state)
                friction_forces.append(force_abs)
            else:
                rows_pyramid += 1
                _count_dict_add(pyramid_states, state)
                pyramid_forces.append(force_abs)

            contact_rows.append(
                {
                    "row": int(row),
                    "kind": kind,
                    "state": state,
                    "force_abs": force_abs,
                }
            )

        if contact_has_active_row:
            active_contacts += 1

        contacts.append(
            {
                "conid": int(conid),
                "worldid": int(worldid),
                "geom_pair": [geom1, geom2],
                "geom_pair_names": [
                    mujoco.mj_id2name(env.mjm, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                    mujoco.mj_id2name(env.mjm, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                ],
                "condim": int(condim),
                "efc_row0": int(row0),
                "row_count_expected": int(row_count),
                "rows": contact_rows,
            }
        )

    summary = {
        "backend": "warp",
        "ncon_total": nacon,
        "constraint_contacts": int(constraint_contacts),
        "active_contacts": int(active_contacts),
        "rows_total": int(rows_total),
        "rows_normal": int(rows_normal),
        "rows_friction": int(rows_friction),
        "rows_pyramid": int(rows_pyramid),
        "rows_active": int(rows_active),
        "row_states": row_state_counts,
        "normal_states": normal_states,
        "friction_states": friction_states,
        "pyramid_states": pyramid_states,
        "mean_abs_row_force": float(np.mean(row_forces)) if row_forces else 0.0,
        "mean_abs_normal_force": float(np.mean(normal_forces)) if normal_forces else 0.0,
        "mean_abs_friction_force": float(np.mean(friction_forces)) if friction_forces else 0.0,
        "mean_abs_pyramid_force": float(np.mean(pyramid_forces)) if pyramid_forces else 0.0,
        "friction_to_normal_force_ratio": (
            float(np.mean(friction_forces) / max(np.mean(normal_forces), 1e-12))
            if normal_forces and friction_forces
            else 0.0
        ),
    }
    return summary, contacts


def summarize_native_contact_state(
    mjm: mujoco.MjModel,
    mjd: mujoco.MjData,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cone_type = int(mjm.opt.cone)
    ncon = int(mjd.ncon)
    nefc = int(mjd.nefc)

    row_state_counts: dict[str, int] = {}
    normal_states: dict[str, int] = {}
    friction_states: dict[str, int] = {}
    pyramid_states: dict[str, int] = {}
    row_forces: list[float] = []
    normal_forces: list[float] = []
    friction_forces: list[float] = []
    pyramid_forces: list[float] = []
    rows_total = 0
    rows_normal = 0
    rows_friction = 0
    rows_pyramid = 0
    rows_active = 0
    constraint_contacts = 0
    active_contacts = 0
    contacts: list[dict[str, Any]] = []
    efc_state, state_source = _native_efc_state_view(mjd)

    for conid in range(ncon):
        c = mjd.contact[conid]
        row0 = int(c.efc_address)
        if row0 < 0 or row0 >= nefc:
            continue

        condim = int(c.dim)
        row_count = _contact_row_count(condim, cone_type)
        geom1 = int(c.geom1)
        geom2 = int(c.geom2)

        constraint_contacts += 1
        contact_rows: list[dict[str, Any]] = []

        contact_has_active_row = False
        for j in range(row_count):
            row = row0 + j
            if row >= nefc:
                break
            state = _state_name(int(efc_state[row]))
            force_abs = abs(float(mjd.efc_force[row]))
            kind = _row_kind(condim, cone_type, j)

            _count_dict_add(row_state_counts, state)
            row_forces.append(force_abs)
            rows_total += 1
            if state != "SATISFIED":
                rows_active += 1
                contact_has_active_row = True

            if kind == "normal":
                rows_normal += 1
                _count_dict_add(normal_states, state)
                normal_forces.append(force_abs)
            elif kind == "friction":
                rows_friction += 1
                _count_dict_add(friction_states, state)
                friction_forces.append(force_abs)
            else:
                rows_pyramid += 1
                _count_dict_add(pyramid_states, state)
                pyramid_forces.append(force_abs)

            contact_rows.append(
                {
                    "row": int(row),
                    "kind": kind,
                    "state": state,
                    "force_abs": force_abs,
                }
            )

        if contact_has_active_row:
            active_contacts += 1

        contacts.append(
            {
                "conid": int(conid),
                "geom_pair": [geom1, geom2],
                "geom_pair_names": [
                    mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                    mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                ],
                "condim": int(condim),
                "efc_row0": int(row0),
                "row_count_expected": int(row_count),
                "rows": contact_rows,
            }
        )

    summary = {
        "backend": "native",
        "ncon_total": int(ncon),
        "constraint_contacts": int(constraint_contacts),
        "active_contacts": int(active_contacts),
        "rows_total": int(rows_total),
        "rows_normal": int(rows_normal),
        "rows_friction": int(rows_friction),
        "rows_pyramid": int(rows_pyramid),
        "rows_active": int(rows_active),
        "state_source": state_source,
        "row_states": row_state_counts,
        "normal_states": normal_states,
        "friction_states": friction_states,
        "pyramid_states": pyramid_states,
        "mean_abs_row_force": float(np.mean(row_forces)) if row_forces else 0.0,
        "mean_abs_normal_force": float(np.mean(normal_forces)) if normal_forces else 0.0,
        "mean_abs_friction_force": float(np.mean(friction_forces)) if friction_forces else 0.0,
        "mean_abs_pyramid_force": float(np.mean(pyramid_forces)) if pyramid_forces else 0.0,
        "friction_to_normal_force_ratio": (
            float(np.mean(friction_forces) / max(np.mean(normal_forces), 1e-12))
            if normal_forces and friction_forces
            else 0.0
        ),
    }
    return summary, contacts


def _norm_counts(counts: dict[str, int]) -> dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return {}
    return {k: float(v / total) for k, v in counts.items()}


def distribution_l1(a: dict[str, int], b: dict[str, int]) -> float:
    pa = _norm_counts(a)
    pb = _norm_counts(b)
    keys = set(pa) | set(pb)
    return float(sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys))


def detect_contact_parity_mismatches(
    warp_summary: dict[str, Any],
    native_summary: dict[str, Any],
    *,
    max_contact_delta: int,
    max_state_l1: float,
    max_force_ratio_factor: float,
) -> dict[str, Any]:
    mismatches: list[str] = []

    contact_delta = int(warp_summary["constraint_contacts"]) - int(
        native_summary["constraint_contacts"]
    )
    active_delta = int(warp_summary["active_contacts"]) - int(
        native_summary["active_contacts"]
    )
    if abs(contact_delta) > int(max_contact_delta):
        mismatches.append(
            f"constraint_contact_count_delta={contact_delta} exceeds {max_contact_delta}"
        )
    if abs(active_delta) > int(max_contact_delta):
        mismatches.append(f"active_contact_count_delta={active_delta} exceeds {max_contact_delta}")

    warp_normal_states = dict(warp_summary.get("normal_states", {}))
    native_normal_states = dict(native_summary.get("normal_states", {}))
    warp_friction_states_dict = dict(warp_summary.get("friction_states", {}))
    native_friction_states_dict = dict(native_summary.get("friction_states", {}))
    warp_row_states = dict(warp_summary.get("row_states", {}))
    native_row_states = dict(native_summary.get("row_states", {}))

    normal_l1 = distribution_l1(warp_normal_states, native_normal_states)
    friction_l1 = distribution_l1(warp_friction_states_dict, native_friction_states_dict)
    row_l1 = distribution_l1(warp_row_states, native_row_states)
    if normal_l1 > float(max_state_l1):
        mismatches.append(f"normal_state_l1={normal_l1:.4f} exceeds {max_state_l1}")
    if friction_l1 > float(max_state_l1):
        mismatches.append(f"friction_state_l1={friction_l1:.4f} exceeds {max_state_l1}")
    if row_l1 > float(max_state_l1):
        mismatches.append(f"row_state_l1={row_l1:.4f} exceeds {max_state_l1}")

    warp_ratio = float(warp_summary.get("friction_to_normal_force_ratio", 0.0))
    native_ratio = float(native_summary.get("friction_to_normal_force_ratio", 0.0))
    if warp_ratio > 0.0 and native_ratio > 0.0:
        ratio_factor = math.exp(abs(math.log(warp_ratio) - math.log(native_ratio)))
    elif warp_ratio == native_ratio:
        ratio_factor = 1.0
    else:
        ratio_factor = float("inf")
    if ratio_factor > float(max_force_ratio_factor):
        mismatches.append(
            "friction_to_normal_force_ratio_factor="
            f"{ratio_factor:.4f} exceeds {max_force_ratio_factor} "
            f"(warp={warp_ratio:.4f}, native={native_ratio:.4f})"
        )

    warp_friction_states = set(warp_friction_states_dict.keys())
    native_friction_states = set(native_friction_states_dict.keys())
    if warp_friction_states == {"QUADRATIC"} and native_friction_states and native_friction_states != {"QUADRATIC"}:
        mismatches.append(
            "warp friction states are all QUADRATIC while native shows mixed states"
        )

    return {
        "ok": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "metrics": {
            "constraint_contact_delta": int(contact_delta),
            "active_contact_delta": int(active_delta),
            "normal_state_l1": float(normal_l1),
            "friction_state_l1": float(friction_l1),
            "row_state_l1": float(row_l1),
            "force_ratio_factor": float(ratio_factor),
            "warp_force_ratio": float(warp_ratio),
            "native_force_ratio": float(native_ratio),
        },
    }


def run_contact_state_parity(
    *,
    cfg_path: str,
    device: str,
    settle_steps: int,
    ctrl_val: float,
    rollout_cfg: str | None,
    policy_path: str | None,
    rollout_steps: int,
    max_contact_delta: int,
    max_state_l1: float,
    max_force_ratio_factor: float,
) -> dict[str, Any]:
    cfg = _load_cfg(cfg_path)
    model_path = resolve_model_path(cfg["params"]["env"]["model_path"])

    if (rollout_cfg is None) != (policy_path is None):
        raise ValueError("--rollout-cfg and --policy must be provided together")

    if rollout_cfg is not None and policy_path is not None:
        qpos, qvel, actions, state_info = _snapshot_from_rollout(
            cfg_path=rollout_cfg,
            policy_path=policy_path,
            device=device,
            rollout_steps=rollout_steps,
        )
    else:
        qpos, qvel, actions, state_info = _snapshot_from_settle(
            cfg_path=cfg_path,
            device=device,
            settle_steps=settle_steps,
            ctrl_val=ctrl_val,
        )

    # Warp summary at matched state
    warp_env = _env_from_cfg(cfg, device=device, no_grad=True)
    ctrl = actions * float(warp_env.action_strength)
    with torch.no_grad():
        wp.copy(warp_env.warp_data.qpos, wp.from_torch(qpos.contiguous()))
        wp.copy(warp_env.warp_data.qvel, wp.from_torch(qvel.contiguous()))
        wp.copy(warp_env.warp_data.ctrl, wp.from_torch(ctrl.contiguous()))
        wp.synchronize()
        mjw.forward(warp_env.warp_model, warp_env.warp_data)
        wp.synchronize()
    warp_summary, warp_contacts = summarize_warp_contact_state(warp_env)

    # Native summary at same qpos/qvel/ctrl
    mjm = mujoco.MjModel.from_xml_path(model_path)
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos[0].detach().cpu().numpy().astype(np.float64)
    mjd.qvel[:] = qvel[0].detach().cpu().numpy().astype(np.float64)
    ctrl_np = ctrl[0].detach().cpu().numpy().astype(np.float64)
    mjd.ctrl[: ctrl_np.shape[0]] = ctrl_np
    mujoco.mj_forward(mjm, mjd)
    native_summary, native_contacts = summarize_native_contact_state(mjm, mjd)

    parity = detect_contact_parity_mismatches(
        warp_summary,
        native_summary,
        max_contact_delta=max_contact_delta,
        max_state_l1=max_state_l1,
        max_force_ratio_factor=max_force_ratio_factor,
    )

    return {
        "cfg_path": cfg_path,
        "model_path": model_path,
        "device": device,
        "state": state_info,
        "thresholds": {
            "max_contact_delta": int(max_contact_delta),
            "max_state_l1": float(max_state_l1),
            "max_force_ratio_factor": float(max_force_ratio_factor),
        },
        "warp": {
            "summary": warp_summary,
            "contacts": warp_contacts,
        },
        "native": {
            "summary": native_summary,
            "contacts": native_contacts,
        },
        "parity": parity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Warp and native MuJoCo contact state on matched ant state"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="msk_warp/configs/experiments/ant_soft_surrogate.yaml",
        help="Base env config to define model/env settings",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--ctrl-val", type=float, default=0.3)
    parser.add_argument("--rollout-cfg", type=str, default=None)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--max-contact-delta", type=int, default=2)
    parser.add_argument("--max-state-l1", type=float, default=0.35)
    parser.add_argument("--max-force-ratio-factor", type=float, default=3.0)
    parser.add_argument("--assert-on-mismatch", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def print_summary(report: dict[str, Any]) -> None:
    warp = report["warp"]["summary"]
    native = report["native"]["summary"]
    parity = report["parity"]
    metrics = parity["metrics"]

    print("Contact State Parity Diagnostic")
    print("================================")
    print(f"model: {report['model_path']}")
    print(f"state source: {report['state']['source']}")
    print()
    print(
        "warp:   contacts={:d} active={:d} normal_states={} friction_states={}".format(
            warp["constraint_contacts"],
            warp["active_contacts"],
            warp["normal_states"],
            warp["friction_states"],
        )
    )
    print(
        "native: contacts={:d} active={:d} normal_states={} friction_states={}".format(
            native["constraint_contacts"],
            native["active_contacts"],
            native["normal_states"],
            native["friction_states"],
        )
    )
    if "state_source" in native:
        print(f"native state source: {native['state_source']}")
    print()
    print(
        "force ratios (friction/normal): warp={:.4f}, native={:.4f}, factor={:.4f}".format(
            metrics["warp_force_ratio"],
            metrics["native_force_ratio"],
            metrics["force_ratio_factor"],
        )
    )
    print(
        "state L1: normal={:.4f}, friction={:.4f}, row={:.4f}".format(
            metrics["normal_state_l1"],
            metrics["friction_state_l1"],
            metrics["row_state_l1"],
        )
    )
    print(
        "contact deltas: constraint={:d}, active={:d}".format(
            metrics["constraint_contact_delta"],
            metrics["active_contact_delta"],
        )
    )
    print()
    print(f"parity_ok: {parity['ok']} (mismatches={parity['mismatch_count']})")
    for msg in parity["mismatches"]:
        print(f"  - {msg}")


def main() -> None:
    args = parse_args()

    report = run_contact_state_parity(
        cfg_path=args.cfg,
        device=args.device,
        settle_steps=args.settle_steps,
        ctrl_val=args.ctrl_val,
        rollout_cfg=args.rollout_cfg,
        policy_path=args.policy,
        rollout_steps=args.rollout_steps,
        max_contact_delta=args.max_contact_delta,
        max_state_l1=args.max_state_l1,
        max_force_ratio_factor=args.max_force_ratio_factor,
    )
    print_summary(report)

    if args.output_json is not None:
        out_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote JSON report to: {out_path}")

    if args.assert_on_mismatch and not report["parity"]["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    wp.init()
    main()
