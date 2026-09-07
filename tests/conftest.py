"""Test layout: tests/unit (CPU, every commit), tests/gpu (needs CUDA), tests/slow (training regressions).

Markers are applied from the directory name, so a test file only needs to live
in the right folder. `gpu` tests skip without CUDA (or with MSK_SKIP_GPU=1);
`slow` tests run only with `--run-slow` and also need CUDA.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_LAYOUT_MARKERS = ("unit", "gpu", "slow")


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests under tests/slow (training regressions; minutes to hours)",
    )


def _cuda_available() -> bool:
    if os.environ.get("MSK_SKIP_GPU") == "1":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    cuda = _cuda_available()
    skip_gpu = pytest.mark.skip(reason="needs CUDA (or MSK_SKIP_GPU=1 is set)")
    skip_slow = pytest.mark.skip(reason="slow test: pass --run-slow to run it")
    for item in items:
        parts = Path(str(item.fspath)).parts
        for name in _LAYOUT_MARKERS:
            if name in parts:
                item.add_marker(getattr(pytest.mark, name))
        if item.get_closest_marker("slow"):
            if not run_slow:
                item.add_marker(skip_slow)
                continue
            if not cuda:
                item.add_marker(skip_gpu)
        elif item.get_closest_marker("gpu") and not cuda:
            item.add_marker(skip_gpu)


@pytest.fixture
def ant_ppo_ckpt() -> str:
    """Path to the walking PPO ant checkpoint (42.8 m mean displacement), or skip."""
    path = os.environ.get("MSK_ANT_PPO_CKPT") or os.path.expanduser(
        "~/checkpoints/ant_ppo_bootstrap_iter500_reward5828.841.pt"
    )
    if not os.path.exists(path):
        pytest.skip(f"PPO ant checkpoint not found at {path} (set MSK_ANT_PPO_CKPT)")
    return path
