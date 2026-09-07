"""Minimal benchmark harness for SHAC speed-up experiments.

Subprocesses scripts/train.py with prescribed config + seed + iter cap, parses
its stdout for per-iter FPS and ep_loss/ep_len, prints a summary row, and
appends to docs/shac_speedup_results.md so variants can be compared.

The variant tag is what identifies a row in the results table. The harness
supports inline YAML key overrides for sweeping num_actors / substeps without
multiplying YAML files.

Examples:
  python scripts/bench_shac_speed.py --cfg msk_warp/configs/ant_shac_soft.yaml \\
      --seed 42 --max-epochs 50 --tag ant_soft_baseline
  python scripts/bench_shac_speed.py --cfg msk_warp/configs/ant_shac_soft.yaml \\
      --seed 42 --max-epochs 50 --tag ant_soft_n256 \\
      --override env.num_actors=256 --override config.num_actors=256
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_MD = REPO_ROOT / "docs" / "shac_speedup_results.md"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"

ITER_RE = re.compile(
    r"^iter\s+(\d+):\s+ep loss\s+(\S+?),\s+ep discounted loss\s+\S+?,\s+"
    r"ep len\s+(\S+?),\s+fps total\s+([0-9.]+),"
)


def _parse_float(s: str) -> float:
    """Tolerate 'inf' / 'nan' from numpy formatting."""
    try:
        return float(s)
    except ValueError:
        return float("nan")


def apply_override(cfg: dict, key: str, val: str) -> None:
    """Apply `a.b.c=value` to a nested dict (yaml-typed)."""
    parts = key.split(".")
    parsed = yaml.safe_load(val)
    node = cfg["params"]
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = parsed


def write_tmp_cfg(cfg: dict) -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="bench_cfg_")
    os.close(fd)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return Path(path)


def run_training(cfg_path: Path, logdir: Path, seed: int, max_epochs: int):
    """Subprocess train.py; tee stdout to a logfile + parse per-iter."""
    logdir.mkdir(parents=True, exist_ok=True)
    stdout_path = logdir / "bench_stdout.log"
    cmd = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "--cfg", str(cfg_path),
        "--logdir", str(logdir),
        "--seed", str(seed),
        "--max-epochs", str(max_epochs),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    iters = []  # list of (iter, ep_loss, ep_len, fps)
    t0 = time.time()
    with open(stdout_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT), env=env, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            m = ITER_RE.match(line.strip())
            if m:
                it = int(m.group(1))
                ep_loss = _parse_float(m.group(2))
                ep_len = _parse_float(m.group(3))
                fps = float(m.group(4))
                iters.append((it, ep_loss, ep_len, fps))
        proc.wait()
    wall = time.time() - t0
    return iters, wall, proc.returncode, stdout_path


def summarize(iters):
    """Median FPS over iters 5..50 (skip JIT warmup), final ep_loss/ep_len."""
    fps_window = [f for (i, _l, _e, f) in iters if 5 <= i <= 50]
    if not fps_window:
        fps_window = [f for (_i, _l, _e, f) in iters]
    median_fps = statistics.median(fps_window) if fps_window else float("nan")
    if iters:
        last = iters[-1]
        return {
            "median_fps_5_50": median_fps,
            "final_iter": last[0],
            "final_ep_loss": last[1],
            "final_ep_len": last[2],
            "n_iters": len(iters),
        }
    return {"median_fps_5_50": float("nan"), "final_iter": 0,
            "final_ep_loss": float("nan"), "final_ep_len": float("nan"),
            "n_iters": 0}


def append_results_row(results_md: Path, tag: str, cfg_path: str, seed: int,
                       max_epochs: int, summary: dict, wall_s: float,
                       returncode: int) -> None:
    results_md.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results_md.exists()
    if returncode != 0:
        note = "crash (rc!=0)"
    elif summary["n_iters"] == 0:
        note = "no iters"
    elif summary["final_ep_loss"] != summary["final_ep_loss"] or summary["final_ep_loss"] == float("inf"):
        note = "bootstrap"
    else:
        note = "ok"
    with open(results_md, "a") as f:
        if new_file:
            f.write("# SHAC speed-up results\n\n")
            f.write("Each row: one bench_shac_speed.py run. `median_fps` is over iters 5..50.\n\n")
            f.write("| tag | cfg | seed | iters | median_fps | final_ep_loss | final_ep_len | wall_s | note |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
        f.write(
            f"| {tag} | {cfg_path} | {seed} | {summary['n_iters']} | "
            f"{summary['median_fps_5_50']:.2f} | {summary['final_ep_loss']:.2f} | "
            f"{summary['final_ep_len']:.1f} | {wall_s:.1f} | {note} |\n"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="YAML config path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--tag", required=True, help="Unique row label in results.md")
    p.add_argument("--override", action="append", default=[],
                   help="YAML key override, e.g. env.num_actors=256")
    p.add_argument("--results-md", default=str(DEFAULT_RESULTS_MD))
    p.add_argument("--logdir", default=None,
                   help="Optional override for run logdir (default: logs/bench_<tag>)")
    args = p.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    for ov in args.override:
        if "=" not in ov:
            raise SystemExit(f"--override must be key=value, got: {ov}")
        k, v = ov.split("=", 1)
        apply_override(cfg, k.strip(), v.strip())

    tmp_cfg = write_tmp_cfg(cfg)
    logdir = Path(args.logdir) if args.logdir else (REPO_ROOT / "logs" / f"bench_{args.tag}")

    print(f"[bench] tag={args.tag} cfg={args.cfg} seed={args.seed} "
          f"max_epochs={args.max_epochs} logdir={logdir}")
    if args.override:
        print(f"[bench] overrides: {args.override}")

    iters, wall, rc, stdout_path = run_training(tmp_cfg, logdir, args.seed, args.max_epochs)
    try:
        os.unlink(tmp_cfg)
    except OSError:
        pass

    summary = summarize(iters)
    print(f"[bench] rc={rc} wall={wall:.1f}s n_iters={summary['n_iters']} "
          f"median_fps[5..50]={summary['median_fps_5_50']:.1f} "
          f"final_ep_loss={summary['final_ep_loss']:.2f} "
          f"final_ep_len={summary['final_ep_len']:.1f}")
    print(f"[bench] stdout: {stdout_path}")

    append_results_row(Path(args.results_md), args.tag, args.cfg, args.seed,
                       args.max_epochs, summary, wall, rc)
    print(f"[bench] row appended to {args.results_md}")

    if rc != 0:
        raise SystemExit(f"train.py exited with code {rc}")


if __name__ == "__main__":
    main()
