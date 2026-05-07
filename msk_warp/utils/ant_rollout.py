"""Shared helpers for policy-visited Ant rollout snapshots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from typing import Iterable

import torch
import warp as wp
import yaml

from msk_warp import PACKAGE_ROOT
from msk_warp.envs.ant import AntEnv


def resolve_cfg_path(cfg_path: str | None) -> str | None:
    """Resolve a config path relative to the package root when needed."""
    if cfg_path is None:
        return None
    if os.path.isabs(cfg_path):
        return cfg_path
    pkg_path = PACKAGE_ROOT / cfg_path
    if pkg_path.exists():
        return str(pkg_path)
    return cfg_path


def load_cfg(cfg_path: str) -> dict[str, Any]:
    """Load a YAML config with package-relative path resolution."""
    with open(resolve_cfg_path(cfg_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env_from_cfg(
    cfg: dict[str, Any],
    *,
    device: str,
    no_grad: bool,
    num_envs: int = 1,
) -> AntEnv:
    """Construct an AntEnv from a config payload."""
    env_kwargs = dict(cfg["params"]["env"])
    env_kwargs.pop("name", None)
    env_kwargs.pop("num_actors", None)
    env_kwargs["num_envs"] = int(num_envs)
    env_kwargs["device"] = device
    env_kwargs["no_grad"] = no_grad
    return AntEnv(**env_kwargs)


def normalize_snapshot_steps(
    steps: Iterable[int] | None,
    *,
    fallback_step: int | None = None,
) -> list[int]:
    """Return a sorted unique list of non-negative rollout snapshot steps."""
    normalized = sorted({int(step) for step in steps or [] if int(step) >= 0})
    if normalized:
        return normalized
    if fallback_step is None:
        return []
    return [int(fallback_step)]


def capture_rollout_snapshots(
    cfg_path: str,
    policy_path: str,
    *,
    device: str,
    snapshot_steps: Iterable[int],
    num_envs: int = 1,
) -> list[dict[str, Any]]:
    """Capture deterministic policy rollout snapshots at selected steps.

    Each snapshot stores the post-step state at the requested rollout step plus
    the deterministic policy action to apply from that state on the next step.
    """
    steps = normalize_snapshot_steps(snapshot_steps)
    if not steps:
        raise ValueError("snapshot_steps must contain at least one non-negative step")

    cfg = load_cfg(cfg_path)
    torch.manual_seed(cfg["params"]["general"].get("seed", 42))

    env = env_from_cfg(cfg, device=device, no_grad=False, num_envs=num_envs)
    checkpoint = torch.load(policy_path, weights_only=False)
    actor = checkpoint[0].to(device)
    obs_rms = checkpoint[3]
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    actor.eval()

    def _policy_actions(obs: torch.Tensor) -> torch.Tensor:
        obs_in = obs_rms.normalize(obs) if obs_rms is not None else obs
        return torch.tanh(actor(obs_in, deterministic=True))

    target_steps = set(steps)
    max_step = max(steps)
    obs = env.reset()
    snapshots: list[dict[str, Any]] = []

    with torch.no_grad():
        if 0 in target_steps:
            snapshots.append(
                {
                    "step": 0,
                    "qpos": wp.to_torch(env.warp_data.qpos).detach().clone(),
                    "qvel": wp.to_torch(env.warp_data.qvel).detach().clone(),
                    "actions": _policy_actions(obs).detach().clone(),
                    "source": "rollout",
                }
            )

        for step in range(1, max_step + 1):
            actions = _policy_actions(obs)
            obs, _, _, _, _, _ = env.step(actions)
            if step in target_steps:
                snapshots.append(
                    {
                        "step": int(step),
                        "qpos": wp.to_torch(env.warp_data.qpos).detach().clone(),
                        "qvel": wp.to_torch(env.warp_data.qvel).detach().clone(),
                        "actions": _policy_actions(obs).detach().clone(),
                        "source": "rollout",
                    }
                )

    snapshots.sort(key=lambda item: item["step"])
    return snapshots
