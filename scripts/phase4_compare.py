"""Phase 4 cross-engine comparison & analysis.

Loads npz outputs from phase4_harness_{mskwarp,diffrl}.py and compares:
- fall rate
- mean fall step
- height trajectory (mean, std)
- height-distribution KS-like distance over time

Usage:
    python scripts/phase4_compare.py \
        --mskwarp tests/data/phase4_p1_mskwarp_std03.npz \
        --diffrl  tests/data/phase4_p1_diffrl_std03.npz \
        --label   "Phase 1 gaussian std=0.3" \
        --output  tests/data/phase4_p1_compare.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(npz_path):
    data = dict(np.load(npz_path, allow_pickle=True))
    cfg = json.loads(str(data["config_json"]))
    return data, cfg


def summarize_engine(data, name):
    heights = data["heights"]  # (T, N)
    fell_step = data["fell_step"]  # (N,)
    T, N = heights.shape
    fallen_mask = fell_step >= 0
    fall_rate = float(fallen_mask.mean())
    mean_fall_step = float(fell_step[fallen_mask].mean()) if fallen_mask.any() else -1.0
    median_fall_step = float(np.median(fell_step[fallen_mask])) if fallen_mask.any() else -1.0
    # Per-step stats
    mean_h_over_time = heights.mean(axis=1).tolist()  # (T,)
    std_h_over_time = heights.std(axis=1).tolist()
    # Fall rate over time (cumulative)
    cumulative_fall_rate = []
    for t in range(T):
        cumulative_fall_rate.append(float(((fell_step >= 0) & (fell_step <= t)).mean()))
    summary = {
        "engine": name,
        "num_envs": int(N), "num_steps": int(T),
        "fall_rate_total": fall_rate,
        "mean_fall_step": mean_fall_step,
        "median_fall_step": median_fall_step,
        "initial_mean_h": float(heights[0].mean()),
        "final_mean_h": float(heights[-1].mean()),
        "min_mean_h": float(np.min(heights.mean(axis=1))),
        "rewards_mean_per_step": float(data["rewards"].mean()),
        "rewards_total_mean_per_env": float(data["rewards"].sum(axis=0).mean()),
    }
    return summary, mean_h_over_time, std_h_over_time, cumulative_fall_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mskwarp", type=str, required=True)
    parser.add_argument("--diffrl", type=str, required=True)
    parser.add_argument("--label", type=str, default="Phase 1")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    msk_data, msk_cfg = load(args.mskwarp)
    diff_data, diff_cfg = load(args.diffrl)

    msk_summary, msk_h, msk_h_std, msk_cum = summarize_engine(msk_data, "mskwarp")
    diff_summary, diff_h, diff_h_std, diff_cum = summarize_engine(diff_data, "diffrl")

    out = {
        "label": args.label,
        "mskwarp_config": msk_cfg,
        "diffrl_config": diff_cfg,
        "mskwarp": msk_summary,
        "diffrl": diff_summary,
        "ratio_fall_rate_diffrl_over_mskwarp": (
            diff_summary["fall_rate_total"] / max(msk_summary["fall_rate_total"], 1e-6)
        ),
        "time_series": {
            "mean_h_mskwarp": msk_h,
            "mean_h_diffrl": diff_h,
            "std_h_mskwarp": msk_h_std,
            "std_h_diffrl": diff_h_std,
            "cumulative_fall_rate_mskwarp": msk_cum,
            "cumulative_fall_rate_diffrl": diff_cum,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    # Console summary
    print(f"\n========== {args.label} ==========")
    print(f"  config: action_mode={msk_cfg.get('action_mode')} std={msk_cfg.get('action_std')} "
          f"N={msk_cfg.get('num_envs')} T={msk_cfg.get('num_steps')}")
    print(f"\n  {'':>20} {'msk-warp':>12} {'DiffRL':>12}  ratio")
    print(f"  {'fall_rate':>20} {msk_summary['fall_rate_total']:>12.3f} {diff_summary['fall_rate_total']:>12.3f}")
    print(f"  {'mean_fall_step':>20} {msk_summary['mean_fall_step']:>12.1f} {diff_summary['mean_fall_step']:>12.1f}")
    print(f"  {'median_fall_step':>20} {msk_summary['median_fall_step']:>12.1f} {diff_summary['median_fall_step']:>12.1f}")
    print(f"  {'initial_mean_h':>20} {msk_summary['initial_mean_h']:>12.3f} {diff_summary['initial_mean_h']:>12.3f}")
    print(f"  {'final_mean_h':>20} {msk_summary['final_mean_h']:>12.3f} {diff_summary['final_mean_h']:>12.3f}")
    print(f"  {'min_mean_h':>20} {msk_summary['min_mean_h']:>12.3f} {diff_summary['min_mean_h']:>12.3f}")
    print(f"  {'rewards_per_step':>20} {msk_summary['rewards_mean_per_step']:>12.3f} {diff_summary['rewards_mean_per_step']:>12.3f}")
    print(f"\n  saved: {args.output}\n")


if __name__ == "__main__":
    main()
