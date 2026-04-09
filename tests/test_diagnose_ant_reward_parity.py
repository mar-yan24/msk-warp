"""Reward parity checks for diagnose_ant against AntEnv reward math."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from msk_warp.envs.ant import AntEnv


def _load_diag_module():
    repo_root = Path(__file__).resolve().parents[1]
    diag_path = repo_root / "scripts" / "diagnose_ant.py"
    spec = importlib.util.spec_from_file_location("diagnose_ant_module", diag_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diagnose_reward_matches_antenv_formula():
    diag = _load_diag_module()
    env_cfg = {
        "action_strength": 1.0,
        "substeps": 4,
        "episode_length": 1000,
        "early_termination": True,
        "forward_vel_weight": 1.7,
        "heading_weight": 0.4,
        "up_weight": 0.2,
        "height_weight": 0.8,
        "joint_vel_penalty": 0.03,
        "push_reward_weight": 0.25,
        "action_penalty": -0.005,
    }
    adapter = diag.AntDiagAdapter(env_cfg, device="cpu")

    qpos = np.array(
        [0.0, 0.0, 0.73, 0.98, 0.0, 0.0, 0.2, 0.1, 0.9, -0.2, -0.8, 0.3, -0.7, 0.1, 0.85],
        dtype=np.float32,
    )
    qvel = np.array(
        [0.2, -0.1, 0.0, 0.05, -0.02, 0.01, 0.4, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1, -0.05],
        dtype=np.float32,
    )
    actions_np = np.array([0.3, -0.2, 0.1, 0.05, -0.1, 0.2, -0.15, 0.25], dtype=np.float32)
    adapter.last_actions = torch.tensor(actions_np, dtype=torch.float32).unsqueeze(0)

    obs = adapter.compute_obs(qpos, qvel)
    components = adapter.decompose_reward(obs, actions_np)

    actions_t = torch.tensor(actions_np, dtype=torch.float32).unsqueeze(0)
    reward_ref = AntEnv._compute_reward(
        obs,
        actions_t,
        adapter.action_penalty,
        adapter.forward_vel_weight,
        adapter.heading_weight,
        adapter.up_weight,
        adapter.height_weight,
        adapter.joint_vel_penalty,
        adapter.push_reward_weight,
    )[0].item()

    summed = (
        components["forward_vel"]
        + components["up_reward"]
        + components["heading_reward"]
        + components["height_reward"]
        + components["action_cost"]
        + components["joint_vel_cost"]
        + components["push_reward"]
    )

    assert abs(components["total"] - reward_ref) < 1e-6
    assert abs(summed - reward_ref) < 1e-6
