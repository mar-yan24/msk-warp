"""Run a staged dual-track ant convergence suite.

This harness automates:
1) Branch diagnostics (smooth/free_body/surrogate-alpha variants)
2) Short training screens (5 epochs)
3) Mid-stage profile sweeps (20 epochs)
4) Acceptance runs (60 epochs)

Outputs machine-readable CSV/JSON artifacts under outputs/ant_suite/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tensorboard.backend.event_processing import event_accumulator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
DEFAULT_DIFFRL_ROOT = "C:/Projects/DiffRL"

TRACK_BASE_CFG = {
    "hard": "msk_warp/configs/experiments/ant_hard_surrogate.yaml",
    "soft": "msk_warp/configs/experiments/ant_soft_surrogate.yaml",
}

TRACK_MODEL = {
    "hard": "substeps4",
    "soft": "soft4",
}

SURROGATE_ALPHAS = [0.80, 0.90, 0.95, 0.99]

MID_PROFILES = {
    "A": {
        "actor_learning_rate": 2e-3,
        "grad_norm": 20.0,
        "state_grad_decay": 0.8,
        "state_grad_clip": 0.0,
    },
    "B": {
        "actor_learning_rate": 1e-3,
        "grad_norm": 40.0,
        "state_grad_decay": 0.9,
        "state_grad_clip": 0.0,
    },
}


@dataclass
class Candidate:
    track: str
    branch: str
    surrogate_alpha: float | None = None
    profile: str | None = None

    @property
    def key(self) -> str:
        parts = [self.track, self.branch]
        if self.surrogate_alpha is not None:
            parts.append(f"a{self.surrogate_alpha:.2f}".replace(".", "p"))
        if self.profile is not None:
            parts.append(f"p{self.profile}")
        return "_".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-track ant convergence suite")
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--diffrl-root", type=str, default=DEFAULT_DIFFRL_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracks", nargs="+", choices=["hard", "soft"], default=["hard", "soft"])
    parser.add_argument("--output-root", type=str, default="outputs/ant_suite")
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--fast-epochs", type=int, default=5)
    parser.add_argument("--mid-epochs", type=int, default=20)
    parser.add_argument("--accept-epochs", type=int, default=60)
    parser.add_argument("--quick-eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--promote-per-track", type=int, default=2)
    parser.add_argument("--accept-forward-vel", type=float, default=0.15)
    parser.add_argument("--accept-x-disp", type=float, default=1.0)
    parser.add_argument("--accept-fall-rate", type=float, default=0.20)
    parser.add_argument("--skip-parity-check", action="store_true")
    parser.add_argument("--skip-contact-parity", action="store_true")
    parser.add_argument("--contact-parity-settle-steps", type=int, default=80)
    parser.add_argument("--contact-parity-ctrl", type=float, default=0.3)
    parser.add_argument("--contact-parity-assert", action="store_true")
    parser.add_argument("--contact-parity-max-contact-delta", type=int, default=2)
    parser.add_argument("--contact-parity-max-state-l1", type=float, default=0.35)
    parser.add_argument("--contact-parity-max-force-ratio-factor", type=float, default=3.0)
    parser.add_argument(
        "--contact-parity-hard-gate",
        action="store_true",
        help="Stop the suite when contact parity reports mismatches",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], dry_run: bool = False) -> bool:
    print(">", " ".join(cmd))
    if dry_run:
        return True
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}")
        return False


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f)


def apply_branch(env_cfg: dict[str, Any], branch: str, surrogate_alpha: float | None) -> None:
    env_cfg["smooth_adjoint"] = True
    env_cfg.pop("friction_bypass_kf", None)
    env_cfg.pop("free_body_adjoint", None)
    env_cfg.pop("penalty_damping_alpha", None)
    env_cfg.pop("friction_surrogate_adjoint", None)
    env_cfg.pop("friction_surrogate_alpha", None)

    if branch == "free_body":
        env_cfg["free_body_adjoint"] = True
    elif branch == "surrogate":
        env_cfg["friction_surrogate_adjoint"] = True
        env_cfg["friction_surrogate_alpha"] = float(surrogate_alpha if surrogate_alpha is not None else 0.9)


def apply_profile(cfg: dict[str, Any], profile_name: str | None) -> None:
    if profile_name is None:
        return
    profile = MID_PROFILES[profile_name]
    conf = cfg["params"]["config"]
    conf["actor_learning_rate"] = float(profile["actor_learning_rate"])
    conf["grad_norm"] = float(profile["grad_norm"])
    conf["state_grad_decay"] = float(profile["state_grad_decay"])
    conf["state_grad_clip"] = float(profile["state_grad_clip"])


def find_policy(logdir: Path) -> Path | None:
    best = logdir / "best_policy.pt"
    final = logdir / "final_policy.pt"
    if best.exists():
        return best
    if final.exists():
        return final
    return None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_grad_metrics(logdir: Path, grad_norm_target: float) -> dict[str, Any]:
    event_dir = logdir / "log"
    if not event_dir.exists():
        return {}
    try:
        ea = event_accumulator.EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        ea.Reload()
    except Exception:
        return {}

    tags = ea.Tags().get("scalars", [])
    if "grad_norm/before_clip" not in tags or "grad_norm/after_clip" not in tags:
        return {}

    before = np.array([s.value for s in ea.Scalars("grad_norm/before_clip")], dtype=np.float64)
    after = np.array([s.value for s in ea.Scalars("grad_norm/after_clip")], dtype=np.float64)
    if "grad_norm/actor_clip_threshold" in tags:
        clip_target = np.array(
            [s.value for s in ea.Scalars("grad_norm/actor_clip_threshold")],
            dtype=np.float64,
        )
        n = int(min(len(before), len(after), len(clip_target)))
        if n <= 0:
            return {}
        before = before[:n]
        after = after[:n]
        clip_target = clip_target[:n]
    else:
        clip_target = np.full_like(after, float(grad_norm_target), dtype=np.float64)

    tol = np.maximum(1e-3, 1e-3 * np.maximum(1.0, np.abs(clip_target)))
    clip_hits = np.abs(after - clip_target) <= tol
    compression = before / np.maximum(after, 1e-12)
    return {
        "epochs_logged": int(len(after)),
        "grad_before_min": float(np.min(before)),
        "grad_before_median": float(np.median(before)),
        "grad_before_max": float(np.max(before)),
        "grad_after_min": float(np.min(after)),
        "grad_after_median": float(np.median(after)),
        "grad_after_max": float(np.max(after)),
        "clip_target_min": float(np.min(clip_target)),
        "clip_target_median": float(np.median(clip_target)),
        "clip_target_max": float(np.max(clip_target)),
        "clip_hit_rate": float(np.mean(clip_hits)),
        "compression_median": float(np.median(compression)),
    }


def score_run(rollout: dict[str, Any] | None, grad_metrics: dict[str, Any]) -> float:
    if rollout is None:
        return -1e9
    fwd = float(rollout.get("forward_vel_mean", 0.0))
    x_disp = float(rollout.get("final_x_disp_mean", 0.0))
    fall = float(rollout.get("fall_rate", 1.0))
    clip = float(grad_metrics.get("clip_hit_rate", 1.0))
    issue_count = float(rollout.get("issue_count", 0))
    return 3.0 * fwd + 0.35 * x_disp - 2.0 * fall - 0.15 * clip - 0.05 * issue_count


def make_candidates(tracks: list[str]) -> list[Candidate]:
    out: list[Candidate] = []
    for track in tracks:
        out.append(Candidate(track=track, branch="smooth"))
        out.append(Candidate(track=track, branch="free_body"))
        for alpha in SURROGATE_ALPHAS:
            out.append(Candidate(track=track, branch="surrogate", surrogate_alpha=alpha))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten(prefix: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[f"{prefix}{key}"] = value
    return out


def build_cfg(
    candidate: Candidate,
    epoch_count: int,
    cfg_out: Path,
    logdir: Path,
    device: str,
    seed: int,
) -> dict[str, Any]:
    base_cfg = load_yaml(REPO_ROOT / TRACK_BASE_CFG[candidate.track])
    base_cfg["params"]["general"]["logdir"] = str(logdir)
    base_cfg["params"]["general"]["device"] = device
    base_cfg["params"]["general"]["seed"] = int(seed)
    base_cfg["params"]["config"]["max_epochs"] = int(epoch_count)
    base_cfg["params"]["config"]["lr_schedule"] = "constant"

    apply_branch(
        base_cfg["params"]["env"],
        branch=candidate.branch,
        surrogate_alpha=candidate.surrogate_alpha,
    )
    apply_profile(base_cfg, candidate.profile)
    save_yaml(cfg_out, base_cfg)
    return base_cfg


def run_diag_for_candidate(
    args: argparse.Namespace,
    candidate: Candidate,
    output_dir: Path,
) -> dict[str, Any] | None:
    diag_json = output_dir / f"diag_{candidate.key}.json"
    cmd = [
        args.python,
        "scripts/diag_reward_gradient_variants.py",
        "--models",
        TRACK_MODEL[candidate.track],
        "--branches",
        candidate.branch,
        "--num-envs",
        "1",
        "--settle-steps",
        "80",
        "--output-json",
        str(diag_json),
    ]
    if candidate.branch == "surrogate" and candidate.surrogate_alpha is not None:
        cmd.extend(["--surrogate-alpha", str(candidate.surrogate_alpha)])

    ok = run_cmd(cmd, dry_run=args.dry_run)
    if not ok:
        return None
    payload = read_json(diag_json)
    if payload is None:
        return None
    results = payload.get("results", [])
    if not results:
        return None
    return results[0]


def run_train_and_eval(
    args: argparse.Namespace,
    candidate: Candidate,
    stage_name: str,
    epoch_count: int,
    eval_episodes: int,
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = output_dir / "runs" / f"{stage_name}_{candidate.key}"
    cfg_path = run_dir / "cfg.yaml"
    logdir = run_dir / "logdir"
    diag_json = run_dir / "rollout_diag.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(
        candidate=candidate,
        epoch_count=epoch_count,
        cfg_out=cfg_path,
        logdir=logdir,
        device=args.device,
        seed=args.seed,
    )

    train_ok = run_cmd(
        [
            args.python,
            "scripts/train.py",
            "--cfg",
            str(cfg_path),
            "--max-epochs",
            str(epoch_count),
        ],
        dry_run=args.dry_run,
    )
    if not train_ok:
        return {
            "ok": False,
            "stage": stage_name,
            "candidate": candidate.key,
            "track": candidate.track,
            "branch": candidate.branch,
            "surrogate_alpha": candidate.surrogate_alpha,
            "profile": candidate.profile,
            "epochs": epoch_count,
            "logdir": str(logdir),
            "score": -1e9,
        }

    policy_path = find_policy(logdir)
    rollout = None
    if policy_path is not None:
        diag_ok = run_cmd(
            [
                args.python,
                "scripts/diagnose_ant.py",
                "--cfg",
                str(logdir / "cfg.yaml"),
                "--policy",
                str(policy_path),
                "--episodes",
                str(eval_episodes),
                "--json-out",
                str(diag_json),
            ],
            dry_run=args.dry_run,
        )
        if diag_ok:
            diag_payload = read_json(diag_json)
            if diag_payload is not None:
                rollout = diag_payload.get("rollout")

    grad_metrics = read_grad_metrics(logdir, float(cfg["params"]["config"]["grad_norm"]))
    score = score_run(rollout, grad_metrics)

    row = {
        "ok": True,
        "stage": stage_name,
        "candidate": candidate.key,
        "track": candidate.track,
        "branch": candidate.branch,
        "surrogate_alpha": candidate.surrogate_alpha,
        "profile": candidate.profile,
        "epochs": epoch_count,
        "logdir": str(logdir),
        "policy_path": str(policy_path) if policy_path is not None else None,
        "score": float(score),
    }
    row.update(flatten("rollout_", rollout))
    row.update(flatten("grad_", grad_metrics))
    return row


def check_acceptance(row: dict[str, Any], args: argparse.Namespace) -> bool:
    fwd = float(row.get("rollout_forward_vel_mean", -1e9))
    x_disp = float(row.get("rollout_final_x_disp_mean", -1e9))
    fall = float(row.get("rollout_fall_rate", 1.0))
    return (
        fwd >= args.accept_forward_vel
        and x_disp >= args.accept_x_disp
        and fall <= args.accept_fall_rate
    )


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.python):
        raise FileNotFoundError(f"Python interpreter not found: {args.python}")

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (REPO_ROOT / args.output_root / run_tag).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    parity_summary: dict[str, Any] | None = None
    contact_parity_rows: list[dict[str, Any]] = []

    print(f"Output directory: {output_dir}")
    print(f"Tracks: {args.tracks}")

    # Stage -1: DiffRL parity preflight
    if args.skip_parity_check:
        print("\n[Stage -1] DiffRL parity preflight: skipped")
    else:
        print("\n[Stage -1] DiffRL parity preflight")
        parity_json = output_dir / "diffrl_parity.json"
        parity_ok = run_cmd(
            [
                args.python,
                "scripts/compare_diffrl_pipeline.py",
                "--diffrl-root",
                args.diffrl_root,
                "--output-json",
                str(parity_json),
            ],
            dry_run=args.dry_run,
        )
        if parity_ok and not args.dry_run:
            payload = read_json(parity_json)
            if payload is not None:
                parity_summary = payload.get("summary")
                if parity_summary is not None:
                    print(
                        "parity status: "
                        f"{parity_summary.get('status')} "
                        f"(critical_drifts={parity_summary.get('critical_drifts')})"
                    )

    # Stage -0.5: contact-state parity preflight (Warp vs native MuJoCo)
    if args.skip_contact_parity:
        print("\n[Stage -0.5] Contact-state parity preflight: skipped")
    else:
        print("\n[Stage -0.5] Contact-state parity preflight")
        for track in args.tracks:
            out_json = output_dir / f"contact_parity_{track}.json"
            cmd = [
                args.python,
                "scripts/diag_contact_state_parity.py",
                "--cfg",
                TRACK_BASE_CFG[track],
                "--device",
                args.device,
                "--settle-steps",
                str(args.contact_parity_settle_steps),
                "--ctrl-val",
                str(args.contact_parity_ctrl),
                "--max-contact-delta",
                str(args.contact_parity_max_contact_delta),
                "--max-state-l1",
                str(args.contact_parity_max_state_l1),
                "--max-force-ratio-factor",
                str(args.contact_parity_max_force_ratio_factor),
                "--output-json",
                str(out_json),
            ]
            if args.contact_parity_assert:
                cmd.append("--assert-on-mismatch")
            ok = run_cmd(cmd, dry_run=args.dry_run)

            row: dict[str, Any] = {"track": track, "ok": bool(ok)}
            if not args.dry_run:
                payload = read_json(out_json)
                if payload is not None:
                    parity = payload.get("parity", {})
                    row.update(flatten("parity_", parity))
                    row.update(flatten("metric_", parity.get("metrics", {})))
                    row["mismatch_count"] = int(parity.get("mismatch_count", 0))
                    row["state_source"] = payload.get("state", {}).get("source")
                    print(
                        f"track={track}: contact_parity_ok={parity.get('ok')} "
                        f"mismatches={parity.get('mismatch_count')}"
                    )
            contact_parity_rows.append(row)

        if args.contact_parity_hard_gate and not args.dry_run:
            failed_tracks = [
                r["track"]
                for r in contact_parity_rows
                if r.get("parity_ok") is False or int(r.get("mismatch_count", 0)) > 0
            ]
            if failed_tracks:
                raise SystemExit(
                    "Contact parity hard gate failed for tracks: "
                    + ", ".join(sorted(set(failed_tracks)))
                )

    # Stage 0: diagnostics for each candidate
    print("\n[Stage 0] Gradient diagnostics")
    candidates = make_candidates(args.tracks)
    for candidate in candidates:
        diag = run_diag_for_candidate(args, candidate, output_dir / "diagnostics")
        row = {
            "track": candidate.track,
            "candidate": candidate.key,
            "branch": candidate.branch,
            "surrogate_alpha": candidate.surrogate_alpha,
        }
        if diag is not None:
            row.update(flatten("diag_", diag))
        diag_rows.append(row)

    # Stage 1: fast screen
    print("\n[Stage 1] Fast screen")
    stage1_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = run_train_and_eval(
            args=args,
            candidate=candidate,
            stage_name="s1",
            epoch_count=args.fast_epochs,
            eval_episodes=args.quick_eval_episodes,
            output_dir=output_dir,
        )
        stage1_rows.append(row)
    all_rows.extend(stage1_rows)

    # Promote top-K per track
    promoted: list[Candidate] = []
    for track in args.tracks:
        track_rows = [
            r for r in stage1_rows
            if r.get("ok") and r.get("track") == track
        ]
        track_rows.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)
        top = track_rows[: args.promote_per_track]
        for row in top:
            cand = next(c for c in candidates if c.key == row["candidate"])
            promoted.append(cand)
        print(f"Track={track}: promoted {[r['candidate'] for r in top]}")

    # Stage 2: profile sweeps for promoted candidates
    print("\n[Stage 2] Mid-stage profile sweeps")
    stage2_rows: list[dict[str, Any]] = []
    for base in promoted:
        for profile_name in MID_PROFILES:
            cand = Candidate(
                track=base.track,
                branch=base.branch,
                surrogate_alpha=base.surrogate_alpha,
                profile=profile_name,
            )
            row = run_train_and_eval(
                args=args,
                candidate=cand,
                stage_name="s2",
                epoch_count=args.mid_epochs,
                eval_episodes=args.quick_eval_episodes,
                output_dir=output_dir,
            )
            stage2_rows.append(row)
    all_rows.extend(stage2_rows)

    # Choose best per track
    final_candidates: list[Candidate] = []
    for track in args.tracks:
        track_rows = [
            r for r in stage2_rows
            if r.get("ok") and r.get("track") == track
        ]
        track_rows.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)
        if not track_rows:
            continue
        best_row = track_rows[0]
        final_candidates.append(
            Candidate(
                track=track,
                branch=best_row["branch"],
                surrogate_alpha=best_row.get("surrogate_alpha"),
                profile=best_row.get("profile"),
            )
        )
        print(f"Track={track}: final candidate {best_row['candidate']}")

    # Stage 3: acceptance runs
    print("\n[Stage 3] Acceptance runs")
    stage3_rows: list[dict[str, Any]] = []
    for cand in final_candidates:
        row = run_train_and_eval(
            args=args,
            candidate=cand,
            stage_name="s3",
            epoch_count=args.accept_epochs,
            eval_episodes=args.final_eval_episodes,
            output_dir=output_dir,
        )
        row["accept_pass"] = bool(check_acceptance(row, args))
        stage3_rows.append(row)
    all_rows.extend(stage3_rows)

    summary = {
        "run_tag": run_tag,
        "tracks": args.tracks,
        "parity": parity_summary,
        "contact_parity": contact_parity_rows,
        "thresholds": {
            "forward_vel": args.accept_forward_vel,
            "x_disp": args.accept_x_disp,
            "fall_rate": args.accept_fall_rate,
        },
        "stage3": stage3_rows,
        "accepted_tracks": [
            r["track"] for r in stage3_rows if r.get("accept_pass")
        ],
    }

    write_csv(output_dir / "diagnostics.csv", diag_rows)
    write_csv(output_dir / "contact_parity.csv", contact_parity_rows)
    write_csv(output_dir / "suite_runs.csv", all_rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    for row in stage3_rows:
        print(
            f"  {row['track']}: pass={row.get('accept_pass')} "
            f"forward_vel={row.get('rollout_forward_vel_mean')} "
            f"x_disp={row.get('rollout_final_x_disp_mean')} "
            f"fall={row.get('rollout_fall_rate')}"
        )
    print(f"\nWrote artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
