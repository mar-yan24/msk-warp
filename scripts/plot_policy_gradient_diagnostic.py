"""Plot Phase 1 diagnostic results.

Inputs: tests/data/policy_gradient_diagnostic_<name>.pkl (one per measurement).
Outputs: tests/data/plots/<name>__{histogram,scatter}.png plus a combined
plot when multiple results sweep eps or H.

Usage:
  .venv/Scripts/python.exe scripts/plot_policy_gradient_diagnostic.py
  .venv/Scripts/python.exe scripts/plot_policy_gradient_diagnostic.py --names ant_random_init ant_from_ppo
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Make `tests.test_ant_policy_gradient` importable so pickle can resolve FdAdResult.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DATA_DIR = _PROJECT_ROOT / "tests" / "data"
_PLOT_DIR = _DATA_DIR / "plots"


def _load(name: str):
    path = _DATA_DIR / f"policy_gradient_diagnostic_{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_scatter(result, ax=None, *, title: str | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ad, fd = result.ad_proj, result.fd_proj
    ax.scatter(ad, fd, s=18, alpha=0.7)
    lim = max(np.max(np.abs(ad)), np.max(np.abs(fd))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="y=x (perfect AD/FD)")
    ax.axhline(0, color="grey", linewidth=0.3)
    ax.axvline(0, color="grey", linewidth=0.3)
    ax.set_xlabel(r"AD projection: $g_{AD} \cdot d_i$")
    ax.set_ylabel(r"FD estimate: $(L_+ - L_-) / 2\varepsilon$")
    if title is None:
        title = (
            f"{result.config}\n"
            f"cos={result.cosine:+.4f}  "
            f"CI95=[{result.bootstrap_ci_low:+.3f}, {result.bootstrap_ci_high:+.3f}]  "
            f"N={result.n_directions}  eps={result.eps}"
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_histogram(result, ax=None, *, title: str | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ad, fd = result.ad_proj, result.fd_proj
    edges = np.linspace(
        min(ad.min(), fd.min()), max(ad.max(), fd.max()), 24
    )
    ax.hist(ad, bins=edges, alpha=0.5, label="AD", color="C0")
    ax.hist(fd, bins=edges, alpha=0.5, label="FD", color="C1")
    ax.set_xlabel("projection value")
    ax.set_ylabel("count")
    if title is None:
        title = f"{result.config}: AD vs FD distribution"
    ax.set_title(title, fontsize=9)
    ax.legend()
    return ax


def render_one(result, name: str):
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    plot_scatter(result, ax=ax)
    fig.tight_layout()
    fig.savefig(_PLOT_DIR / f"{name}__scatter.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    plot_histogram(result, ax=ax)
    fig.tight_layout()
    fig.savefig(_PLOT_DIR / f"{name}__histogram.png", dpi=140)
    plt.close(fig)


def render_panel(names: list[str], results: list, out_name: str = "phase1_panel"):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, name, r in zip(axes, names, results):
        plot_scatter(r, ax=ax, title=f"{name}\ncos={r.cosine:+.4f}  N={r.n_directions}  eps={r.eps}")
    fig.tight_layout()
    fig.savefig(_PLOT_DIR / f"{out_name}.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Specific result names to plot (default: all .pkl files in tests/data)",
    )
    args = parser.parse_args()

    if args.names is None:
        names = sorted(
            p.stem.replace("policy_gradient_diagnostic_", "")
            for p in _DATA_DIR.glob("policy_gradient_diagnostic_*.pkl")
        )
    else:
        names = list(args.names)

    if not names:
        print(f"No .pkl files found in {_DATA_DIR}")
        return

    print(f"Plotting {len(names)} result(s) to {_PLOT_DIR}/")
    results = []
    for name in names:
        try:
            result = _load(name)
        except FileNotFoundError:
            print(f"  [skip] {name}: file not found")
            continue
        print(
            f"  {name:30s}  cos={result.cosine:+.4f}  "
            f"N={result.n_directions}  eps={result.eps}  "
            f"CI95=[{result.bootstrap_ci_low:+.3f}, {result.bootstrap_ci_high:+.3f}]"
        )
        render_one(result, name)
        results.append(result)

    if len(results) > 1:
        render_panel(names, results)
        print(f"  panel: {_PLOT_DIR / 'phase1_panel.png'}")


if __name__ == "__main__":
    main()
