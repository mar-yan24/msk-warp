"""Run the staged PPO -> SHAC Ant transfer pipeline.

This runner promotes checkpoints by canonical Ant diagnostics rather than raw
training reward, which prevents unstable movers from being discarded too early
and catches SHAC regressions into the standing basin.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

PPO_CFG = REPO_ROOT / "msk_warp" / "configs" / "experiments" / "ant_ppo_bootstrap.yaml"
STABILIZE_CFG = REPO_ROOT / "msk_warp" / "configs" / "experiments" / "ant_shac_transfer_stabilize.yaml"
CONSOLIDATE_CFG = REPO_ROOT / "msk_warp" / "configs" / "experiments" / "ant_shac_transfer_consolidate.yaml"

ITER_RE = re.compile(r"_iter(\d+)_")

STAGE_GATES = {
    "bootstrap": {
        "forward_vel_mean": 0.50,
        "final_x_disp_mean": 3.0,
    },
    "stabilize": {
        "forward_vel_mean": 0.15,
        "final_x_disp_mean": 2.0,
        "episode_length_mean": 850.0,
        "fall_rate": 0.20,
        "forbidden_issue_markers": ("STANDING STILL", "LOCAL OPTIMUM"),
    },
    "final": {
        "forward_vel_mean": 0.15,
        "final_x_disp_mean": 2.0,
        "fall_rate": 0.20,
        "episode_length_mean": 850.0,
        "forbidden_issue_markers": ("STANDING STILL", "LOCAL OPTIMUM"),
    },
}


@dataclass
class EvalResult:
    policy: Path
    report_path: Path
    rollout: dict[str, Any]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged Ant PPO -> SHAC pipeline")
    parser.add_argument(
        "--python",
        type=str,
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
        help="Python interpreter used to launch training/diagnostic subprocesses",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-root", type=str, default="logs/ant_ppo_shac_pipeline")
    parser.add_argument(
        "--bootstrap-policy",
        type=str,
        default=None,
        help="Optional existing PPO checkpoint. When set, stage 0 training is skipped.",
    )
    parser.add_argument("--stage-eval-episodes", type=int, default=8)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print(">", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def rollout_score(rollout: dict[str, Any]) -> float:
    return (
        3.0 * float(rollout.get("forward_vel_mean", 0.0))
        + 0.35 * float(rollout.get("final_x_disp_mean", 0.0))
        - 2.0 * float(rollout.get("fall_rate", 1.0))
        - 0.05 * float(rollout.get("issue_count", 0))
    )


def gate_passes(rollout: dict[str, Any], gate: dict[str, Any]) -> bool:
    if float(rollout.get("forward_vel_mean", 0.0)) < float(gate.get("forward_vel_mean", -1e9)):
        return False
    if float(rollout.get("final_x_disp_mean", 0.0)) < float(gate.get("final_x_disp_mean", -1e9)):
        return False
    if float(rollout.get("episode_length_mean", 0.0)) < float(gate.get("episode_length_mean", -1e9)):
        return False
    if float(rollout.get("fall_rate", 0.0)) > float(gate.get("fall_rate", 1e9)):
        return False

    issue_markers = gate.get("forbidden_issue_markers", ())
    issues = tuple(rollout.get("issues", []))
    for marker in issue_markers:
        if any(marker in issue for issue in issues):
            return False
    return True


def iter_sort_key(path: Path) -> tuple[int, str]:
    match = ITER_RE.search(path.name)
    if match is not None:
        return (int(match.group(1)), path.name)
    if path.name == "best_policy.pt":
        return (10**9, path.name)
    if path.name == "final_policy.pt":
        return (10**9 + 1, path.name)
    return (10**9 + 2, path.name)


def list_policy_candidates(logdir: Path) -> list[Path]:
    if not logdir.exists():
        return []
    candidates = []
    for path in logdir.glob("*.pt"):
        if path.name == "init_policy.pt":
            continue
        candidates.append(path)
    return sorted({path.resolve() for path in candidates}, key=iter_sort_key)


def evaluate_policy(
    *,
    python: str,
    eval_cfg: Path,
    policy: Path,
    episodes: int,
    report_path: Path,
    dry_run: bool,
) -> EvalResult | None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "scripts/diagnose_ant.py",
        "--cfg",
        str(eval_cfg),
        "--policy",
        str(policy),
        "--episodes",
        str(episodes),
        "--device",
        "cpu",
        "--json-out",
        str(report_path),
    ]
    run_cmd(cmd, dry_run=dry_run)
    if dry_run:
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rollout = payload["rollout"]
    return EvalResult(
        policy=policy,
        report_path=report_path,
        rollout=rollout,
        score=rollout_score(rollout),
    )


def evaluate_candidates(
    *,
    python: str,
    eval_cfg: Path,
    policies: list[Path],
    episodes: int,
    reports_dir: Path,
    dry_run: bool,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for policy in policies:
        report_path = reports_dir / f"{policy.stem}.json"
        result = evaluate_policy(
            python=python,
            eval_cfg=eval_cfg,
            policy=policy,
            episodes=episodes,
            report_path=report_path,
            dry_run=dry_run,
        )
        if result is not None:
            results.append(result)
    return results


def promote_stage(
    *,
    name: str,
    results: list[EvalResult],
    gate: dict[str, Any],
) -> EvalResult:
    passing = [result for result in results if gate_passes(result.rollout, gate)]
    if not passing:
        raise RuntimeError(f"No {name} checkpoint passed the promotion gate")
    passing.sort(key=lambda item: item.score, reverse=True)
    best = passing[0]
    print(
        f"[{name}] promoted {best.policy.name}: "
        f"score={best.score:.3f} "
        f"fwd={best.rollout['forward_vel_mean']:.3f} "
        f"x={best.rollout['final_x_disp_mean']:.3f} "
        f"fall={best.rollout['fall_rate']:.3f} "
        f"len={best.rollout['episode_length_mean']:.1f}"
    )
    return best


def run_training_stage(
    *,
    python: str,
    cfg_path: Path,
    logdir: Path,
    device: str,
    seed: int,
    init_policy: Path | None,
    dry_run: bool,
) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "scripts/train.py",
        "--cfg",
        str(cfg_path),
        "--logdir",
        str(logdir),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if init_policy is not None:
        cmd.extend(["--init-policy", str(init_policy)])
    run_cmd(cmd, dry_run=dry_run)


def ensure_policy(path_str: str) -> Path:
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Policy not found: {path}")
    return path


def write_summary(path: Path, payload: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()

    python = str(Path(args.python))
    log_root = Path(args.log_root).resolve()
    eval_cfg = CONSOLIDATE_CFG

    summary: dict[str, Any] = {
        "python": python,
        "device": args.device,
        "seed": args.seed,
        "log_root": str(log_root),
        "stages": {},
    }

    if args.bootstrap_policy is None:
        stage0_logdir = log_root / "ppo_bootstrap"
        run_training_stage(
            python=python,
            cfg_path=PPO_CFG,
            logdir=stage0_logdir,
            device=args.device,
            seed=args.seed,
            init_policy=None,
            dry_run=args.dry_run,
        )
        stage0_policies = list_policy_candidates(stage0_logdir)
        stage0_results = evaluate_candidates(
            python=python,
            eval_cfg=eval_cfg,
            policies=stage0_policies,
            episodes=args.stage_eval_episodes,
            reports_dir=stage0_logdir / "pipeline_diagnostics",
            dry_run=args.dry_run,
        )
        stage0_best = None if args.dry_run else promote_stage(
            name="bootstrap",
            results=stage0_results,
            gate=STAGE_GATES["bootstrap"],
        )
        summary["stages"]["bootstrap"] = {
            "logdir": str(stage0_logdir),
            "promoted_policy": None if stage0_best is None else str(stage0_best.policy),
        }
        bootstrap_policy = None if stage0_best is None else stage0_best.policy
    else:
        bootstrap_policy = ensure_policy(args.bootstrap_policy)
        bootstrap_reports = log_root / "bootstrap_policy_eval"
        bootstrap_result = evaluate_policy(
            python=python,
            eval_cfg=eval_cfg,
            policy=bootstrap_policy,
            episodes=args.stage_eval_episodes,
            report_path=bootstrap_reports / f"{bootstrap_policy.stem}.json",
            dry_run=args.dry_run,
        )
        if bootstrap_result is not None and not gate_passes(bootstrap_result.rollout, STAGE_GATES["bootstrap"]):
            raise RuntimeError("Provided bootstrap policy failed the promotion gate")
        summary["stages"]["bootstrap"] = {
            "logdir": None,
            "promoted_policy": None if bootstrap_result is None else str(bootstrap_result.policy),
        }

    stage1_logdir = log_root / "shac_stabilize"
    run_training_stage(
        python=python,
        cfg_path=STABILIZE_CFG,
        logdir=stage1_logdir,
        device=args.device,
        seed=args.seed,
        init_policy=bootstrap_policy,
        dry_run=args.dry_run,
    )
    stage1_policies = list_policy_candidates(stage1_logdir)
    stage1_results = evaluate_candidates(
        python=python,
        eval_cfg=eval_cfg,
        policies=stage1_policies,
        episodes=args.stage_eval_episodes,
        reports_dir=stage1_logdir / "pipeline_diagnostics",
        dry_run=args.dry_run,
    )
    stage1_best = None if args.dry_run else promote_stage(
        name="stabilize",
        results=stage1_results,
        gate=STAGE_GATES["stabilize"],
    )
    accepted_baseline = None
    if not args.dry_run and stage1_best is not None:
        accepted_baseline = evaluate_policy(
            python=python,
            eval_cfg=eval_cfg,
            policy=stage1_best.policy,
            episodes=args.final_eval_episodes,
            report_path=stage1_logdir / "pipeline_diagnostics_final" / f"{stage1_best.policy.stem}.json",
            dry_run=False,
        )
        if accepted_baseline is None or not gate_passes(accepted_baseline.rollout, STAGE_GATES["final"]):
            raise RuntimeError("Promoted stabilize checkpoint failed the final acceptance gate")
    summary["stages"]["stabilize"] = {
        "logdir": str(stage1_logdir),
        "promoted_policy": None if stage1_best is None else str(stage1_best.policy),
        "accepted_report": None if accepted_baseline is None else str(accepted_baseline.report_path),
    }

    stage2_logdir = log_root / "shac_consolidate"
    run_training_stage(
        python=python,
        cfg_path=CONSOLIDATE_CFG,
        logdir=stage2_logdir,
        device=args.device,
        seed=args.seed,
        init_policy=None if stage1_best is None else stage1_best.policy,
        dry_run=args.dry_run,
    )
    stage2_policies = list_policy_candidates(stage2_logdir)
    stage2_results = evaluate_candidates(
        python=python,
        eval_cfg=eval_cfg,
        policies=stage2_policies,
        episodes=args.stage_eval_episodes,
        reports_dir=stage2_logdir / "pipeline_diagnostics_short",
        dry_run=args.dry_run,
    )

    final_best = accepted_baseline
    if not args.dry_run:
        short_list = sorted(stage2_results, key=lambda item: item.score, reverse=True)
        for result in short_list:
            final_result = evaluate_policy(
                python=python,
                eval_cfg=eval_cfg,
                policy=result.policy,
                episodes=args.final_eval_episodes,
                report_path=stage2_logdir / "pipeline_diagnostics_final" / f"{result.policy.stem}.json",
                dry_run=False,
            )
            if (
                final_result is not None
                and gate_passes(final_result.rollout, STAGE_GATES["final"])
                and (final_best is None or final_result.score > final_best.score)
            ):
                final_best = final_result
        if final_best is None:
            raise RuntimeError("No stabilize or consolidate checkpoint passed the final acceptance gate")
        print(
            f"[final] accepted {final_best.policy.name}: "
            f"score={final_best.score:.3f} "
            f"fwd={final_best.rollout['forward_vel_mean']:.3f} "
            f"x={final_best.rollout['final_x_disp_mean']:.3f} "
            f"fall={final_best.rollout['fall_rate']:.3f} "
            f"len={final_best.rollout['episode_length_mean']:.1f}"
        )

    summary["stages"]["consolidate"] = {
        "logdir": str(stage2_logdir),
        "accepted_policy": None if final_best is None else str(final_best.policy),
        "accepted_report": None if final_best is None else str(final_best.report_path),
    }
    write_summary(log_root / "pipeline_summary.json", summary, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
