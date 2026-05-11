"""Phase 2 Exp 3: Gradient-norm trajectory over 50 SHAC epochs.

Tests whether discretization noise amplifies through training. Runs SHAC
from PPO init with state_grad_decay=0 (raw BPTT) and state_grad_clip=0
(no clipping), captures per-epoch grad_norm/before_clip and clip thresholds
via a tap on the TB writer.

Pass: grad_norm/before_clip mean increases >=2x over the run AND
clip-saturation rate (grad_norm > actor_clip_threshold) exceeds 50%
by epoch 50.
Fail: stable or declining grad_norm -> discretization isn't amplifying
through training; H2 weakens.

Output:
  tests/data/phase2_exp3_gradnorm_trajectory.csv  — per-epoch metrics
  tests/data/policy_gradient_diagnostic_summary.json — appended summary
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import yaml

from msk_warp.algorithms.shac import SHAC


_DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--cfg",
        default="msk_warp/configs/experiments/ant_shac_from_ppo.yaml",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        default="~/checkpoints/ant_ppo_bootstrap_iter500_reward5828.841.pt",
    )
    parser.add_argument("--logdir", default=None)
    parser.add_argument(
        "--output",
        default=str(_DATA_DIR / "phase2_exp3_gradnorm_trajectory.csv"),
    )
    args = parser.parse_args()

    logdir = args.logdir or f"logs/_phase2_exp3_seed{args.seed}"

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["params"]["general"]["logdir"] = logdir
    cfg["params"]["general"]["seed"] = args.seed
    cfg["params"]["config"]["max_epochs"] = args.epochs
    cfg["params"]["config"]["save_interval"] = max(args.epochs, 1)

    shac = SHAC(cfg)

    # Load PPO checkpoint
    ckpt_path = os.path.expanduser(args.ppo_checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"PPO checkpoint not found: {ckpt_path}")
    shac.load(ckpt_path, reset_optimizers=True)

    # Override regularization: raw BPTT, no clipping
    shac.state_grad_decay = 0.0
    shac.state_grad_clip = 0.0

    # Intercept writer.add_scalar to record metrics in-memory
    captured: dict[int, dict[str, float]] = {}
    original_add_scalar = shac.writer.add_scalar

    def patched_add_scalar(tag, value, step, *a, **kw):
        try:
            v = float(value)
        except Exception:
            v = float("nan")
        if step not in captured:
            captured[step] = {"step": int(step)}
        captured[step][tag] = v
        return original_add_scalar(tag, value, step, *a, **kw)

    shac.writer.add_scalar = patched_add_scalar

    print(
        f"[Exp 3] Training SHAC for {args.epochs} epochs with "
        f"decay=0, clip=0, seed={args.seed}"
    )
    print(f"[Exp 3] logdir: {logdir}")
    print(f"[Exp 3] PPO checkpoint: {ckpt_path}")
    shac.train()

    # Sort by step
    steps = sorted(captured.keys())
    rows = [captured[s] for s in steps]
    print(f"[Exp 3] Captured {len(rows)} TB-logged steps.")

    # Write CSV with all tags
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    all_keys_sorted = sorted(all_keys)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys_sorted)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Exp 3] Wrote per-step metrics to {output_path}")

    # Analyze grad_norm/before_clip trajectory + clip-saturation
    tag_before = "grad_norm/before_clip"
    tag_thr = "grad_norm/actor_clip_threshold"

    series_before = [r.get(tag_before, float("nan")) for r in rows]
    series_thr = [r.get(tag_thr, float("nan")) for r in rows]

    finite_before = [v for v in series_before if v == v and v > 0]
    if len(finite_before) < 10:
        print(
            f"[Exp 3] WARNING: only {len(finite_before)} finite "
            f"{tag_before} samples found; insufficient for trajectory analysis"
        )
        ratio = float("nan")
        clip_sat = float("nan")
    else:
        first_5 = finite_before[:5]
        last_5 = finite_before[-5:]
        mean_first = sum(first_5) / max(len(first_5), 1)
        mean_last = sum(last_5) / max(len(last_5), 1)
        ratio = mean_last / max(mean_first, 1e-9)

        # Clip saturation: fraction of steps where grad_norm_before > actor_clip_threshold
        n_total = 0
        n_sat = 0
        for r in rows:
            b = r.get(tag_before, float("nan"))
            t = r.get(tag_thr, float("nan"))
            if b == b and t == t and b > 0 and t > 0:
                n_total += 1
                if b > t:
                    n_sat += 1
        clip_sat = (n_sat / n_total) if n_total > 0 else float("nan")

        print(
            f"\n[Exp 3] grad_norm/before_clip mean: "
            f"first5={mean_first:.3e}, last5={mean_last:.3e}, ratio={ratio:.2f}x"
        )
        print(
            f"[Exp 3] clip-saturation: {n_sat}/{n_total} = "
            f"{(clip_sat * 100):.1f}%"
        )

    # Pass criteria
    grew_2x = ratio >= 2.0
    saturated = clip_sat >= 0.5
    if grew_2x and saturated:
        verdict = "PASS — grad norm amplifies + clip saturates"
    elif grew_2x or saturated:
        verdict = "PARTIAL — only one of two criteria met"
    else:
        verdict = "FAIL — grad norm stable or declining; H2 weaker"
    print(f"[Exp 3 verdict] {verdict}")

    # Append summary JSON entry
    summary = {
        "name": "exp3_gradnorm_trajectory",
        "config": (
            f"Phase 2 Exp 3: 50 SHAC epochs from PPO init, decay=0, clip=0, "
            f"seed={args.seed}"
        ),
        "epochs": args.epochs,
        "seed": args.seed,
        "n_logged_steps": len(rows),
        "first5_mean_grad_norm_before_clip": (
            float(sum(finite_before[:5]) / max(len(finite_before[:5]), 1))
            if finite_before
            else float("nan")
        ),
        "last5_mean_grad_norm_before_clip": (
            float(sum(finite_before[-5:]) / max(len(finite_before[-5:]), 1))
            if finite_before
            else float("nan")
        ),
        "grad_norm_ratio_last5_to_first5": ratio,
        "clip_saturation_rate": clip_sat,
        "verdict": verdict,
        "csv_path": str(output_path),
    }
    summary_path = _DATA_DIR / "policy_gradient_diagnostic_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            current = json.load(f)
    else:
        current = []
    current = [e for e in current if e.get("name") != summary["name"]]
    current.append(summary)
    with open(summary_path, "w") as f:
        json.dump(current, f, indent=2)
    print(f"[Exp 3] Appended summary to {summary_path}")


if __name__ == "__main__":
    main()
