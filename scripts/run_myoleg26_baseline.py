"""Frozen, bounded MyoLeg26 PPO baseline with independent behavior evaluation.

First --write-freeze a manifest and commit it. Training then requires that exact
committed manifest and unchanged source/assets/packages. Output directories are
exclusive, checkpoints follow a fixed schedule, and selection uses behavior.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np
import torch
import yaml

from msk_warp import resolve_model_path
from msk_warp.analysis.myoleg26_baseline import behavior_rank, evaluate_policy, isolated_rng
from msk_warp.envs.myoleg26_task import MyoLegTaskContract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "msk_warp/configs/experiments/myoleg26_ppo.yaml"
PROTOCOL = {
    "version": "myoleg26-baseline-v1", "seeds": [0, 1, 2, 3, 4], "epochs": 128,
    "num_actors": 64, "steps_num": 128, "train_control_transitions_per_seed": 1048576,
    "evaluation_epochs": [0, 32, 64, 96, 128], "horizon": 500, "substeps": 4,
    "selection_episode_seeds": list(range(10000, 10016)),
    "confirmation_episode_seeds": list(range(20000, 20032)),
    "audit_episode_seeds": list(range(30000, 30016)),
    "selection": ["maximum survival", "minimum mean episode 2D velocity RMSE sqrt(mean((vx-1)^2+vy^2))",
                  "maximum speed-band fraction", "earliest checkpoint"],
    "success": {"selection_survivors": 15, "confirmation_survivors": 29,
                "mean_episode_vx_min": .8, "mean_episode_vx_max": 1.2,
                "mean_episode_velocity_2d_rmse_max": .3},
    "reset": "per-episode NumPy default_rng: dx U(-.05,.05), dz U(0,.02), root dv U(-.05,.05), yaw U(-.05,.05)",
    "audit_noise": "per-episode Torch generator(seed+1000000000), independent normal noise before tanh",
    "wall_cap": "inclusive seed wall; checked at whole-epoch boundaries and evaluation steps; setup/epoch/step may overshoot",
    "default_max_seed_wall_seconds": 1200,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args, directory=ROOT):
    return subprocess.run(["git", "-c", f"safe.directory={Path(directory).as_posix()}",
                           "-C", str(directory), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def freeze_record(*, smoke=False):
    """CPU-only provenance snapshot, including every packaged Python dependency."""
    import mujoco_warp

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(PROTOCOL)
    if smoke:
        protocol.update(version="myoleg26-baseline-smoke-v1", seeds=[0], epochs=2,
                        num_actors=2, steps_num=8, train_control_transitions_per_seed=32,
                        evaluation_epochs=[0, 2], horizon=8,
                        selection_episode_seeds=[10000, 10001],
                        confirmation_episode_seeds=[20000, 20001], audit_episode_seeds=[30000, 30001])
        cfg["params"]["config"].update(ppo_epochs=1, num_minibatches=2)
        cfg["params"]["network"]["actor_mlp"]["units"] = [16, 16]
        cfg["params"]["network"]["critic_mlp"]["units"] = [16, 16]
    model_path = Path(resolve_model_path(cfg["params"]["env"]["model_path"])).resolve()
    manifest_path = model_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {path.relative_to(ROOT).as_posix(): sha256(path)
             for path in (ROOT / "msk_warp").rglob("*.py")}
    files[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha256(__file__)
    files[CONFIG.relative_to(ROOT).as_posix()] = sha256(CONFIG)
    files[manifest_path.relative_to(ROOT).as_posix()] = sha256(manifest_path)
    for relative, expected in manifest["files"].items():
        path = (model_path.parent / relative).resolve()
        if not path.is_relative_to(model_path.parent) or sha256(path) != expected:
            raise ValueError(f"Asset hash/path mismatch: {relative}")
        files[path.relative_to(ROOT).as_posix()] = expected
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv, model.na, model.nu) != (47, 46, 26, 26):
        raise ValueError("Frozen official model dimensions changed")
    if model.opt.timestep != .002 or not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_WARMSTART):
        raise ValueError("Expected dt=.002 and warmstart disabled")
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    backend_path = Path(mujoco_warp.__file__).resolve().parent
    backend_root = Path(git("rev-parse", "--show-toplevel", directory=backend_path))
    if git("status", "--porcelain", "--untracked-files=no", directory=backend_root):
        raise ValueError("Backend has tracked modifications")
    return {"schema_version": "myoleg26-baseline-freeze-v1", "protocol": protocol,
            "config": cfg, "task_contract": MyoLegTaskContract().as_dict(), "files": files,
            "model_path": model_path.relative_to(ROOT).as_posix(),
            "model_sha256": sha256(model_path),
            "compiled_model_sha256": hashlib.sha256(buffer.tobytes()).hexdigest(),
            "backend": {"head": git("rev-parse", "HEAD", directory=backend_root)},
            "packages": {name: importlib.metadata.version(name) for name in
                         ("mujoco", "mujoco-warp", "warp-lang", "torch", "numpy", "PyYAML")},
            "python": platform.python_version()}


def validate_freeze(path, *, smoke=False):
    path = Path(path).resolve()
    if not smoke:
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                                   check=True, capture_output=True).stdout
        if committed != path.read_bytes():
            raise ValueError("Freeze manifest must exactly match its committed bytes")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    current = freeze_record(smoke=smoke)
    # Later standalone checker modules may be added. Every existing input stays
    # pinned; changing an import in an existing module still changes its hash.
    current["files"] = {name: current["files"].get(name) for name in frozen["files"]}
    if current != frozen:
        changed = [name for name in current if current[name] != frozen.get(name)]
        raise ValueError(f"Frozen experiment mismatch: {changed}")
    for relative in frozen["files"]:
        # Content equality alone is insufficient: every input must be committed.
        if git("status", "--porcelain", "--", relative):
            raise ValueError(f"Frozen input has uncommitted changes: {relative}")
        git("ls-files", "--error-unmatch", "--", relative)
    return frozen


def success(evaluation, survivors):
    summary = evaluation["summary"]
    return bool(evaluation["complete"] and sum(row["survived"] for row in evaluation["episodes"]) >= survivors
                and .8 <= summary["mean_forward_velocity_mps"] <= 1.2
                and summary["mean_episode_velocity_2d_rmse_mps"] <= .3)


def save_traces(path, traces, frozen, *, seed, checkpoint, checkpoint_epoch, evaluation):
    metadata = {"schema_version": "myoleg26-raw-traces-v1", "training_seed": seed,
                "checkpoint_epoch": checkpoint_epoch, "checkpoint_sha256": sha256(checkpoint),
                "model_path": str(ROOT / frozen["model_path"]), "model_sha256": frozen["model_sha256"],
                "compiled_model_sha256": frozen["compiled_model_sha256"],
                "task_contract": frozen["task_contract"], "substeps": frozen["protocol"]["substeps"],
                "control_dt": evaluation["control_dt"], "protocol": frozen["protocol"],
                "policy_mode": evaluation["policy_mode"], "complete": evaluation["complete"],
                "contact_classification": "unclassified; use fresh native MuJoCo at captured pre-step state",
                "state_timing": "before action; done and terminated are from the following transition",
                "warmstart_disabled": True, "frozen_files": frozen["files"]}
    with Path(path).open("xb") as output:
        np.savez_compressed(output, **traces, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))


def run_seed(frozen, seed, epochs, output_dir, max_wall):
    from msk_warp.algorithms.ppo import PPO
    from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv

    protocol = frozen["protocol"]
    smoke = protocol["version"] == "myoleg26-baseline-smoke-v1"
    output_dir.mkdir()
    started = time.perf_counter()
    deadline = started + max_wall
    timers = {key: 0. for key in ("setup", "training_rollout", "training_update", "evaluation", "checkpoint_and_trace_io")}
    report = {"seed": seed, "requested_epochs": epochs, "completed_epochs": 0,
              "stop_reason": "epoch_budget", "censored": False, "evaluations": [], "losses": [],
              "training_completed_control_calls": 0, "training_attempted_control_calls": 0,
              "evaluation_counters": []}
    progress_path = output_dir / "progress.jsonl"
    progress_path.touch(exist_ok=False)
    cfg = copy.deepcopy(frozen["config"])
    cfg["params"]["general"].update(seed=seed, logdir=str(output_dir), device="cuda:0")
    cfg["params"]["env"]["num_actors"] = protocol["num_actors"]
    cfg["params"]["config"].update(max_epochs=epochs, steps_num=protocol["steps_num"], save_interval=0)
    write_json(output_dir / "effective_config.json", cfg)
    algo = None
    best = None

    def timed(kind, function):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        try:
            return function()
        finally:
            torch.cuda.synchronize()
            timers[kind] += time.perf_counter() - begin

    def evaluate(epoch, env, seeds, kind, checkpoint, *, capture=False, deterministic=True):
        counter = {"kind": kind, "epoch": epoch, "worlds": env.num_envs,
                   "attempted_calls": 0, "completed_calls": 0}
        report["evaluation_counters"].append(counter)
        with isolated_rng():
            result, traces = timed("evaluation", lambda: evaluate_policy(
                env, algo.actor, algo.obs_rms, seeds, horizon=protocol["horizon"], capture=capture,
                deterministic=deterministic, deadline=deadline, transition_counter=counter))
        result.update(epoch=epoch, kind=kind, training_control_transitions=report["training_completed_control_calls"] * algo.num_envs,
                      seed_wall_seconds=time.perf_counter() - started, checkpoint_sha256=sha256(checkpoint))
        result["success"] = success(result, 29 if kind == "confirmation" else 15) if kind != "stochastic_audit" and not smoke else None
        report["evaluations"].append(result)
        timed("checkpoint_and_trace_io", lambda: write_json(output_dir / f"{kind}_{epoch:04d}.json", result))
        if capture:
            timed("checkpoint_and_trace_io", lambda: save_traces(
                output_dir / f"{kind}_traces.npz", traces, frozen, seed=seed, checkpoint=checkpoint,
                checkpoint_epoch=epoch, evaluation=result))
        return result

    try:
        setup_start = time.perf_counter()
        algo = PPO(cfg)
        report["hardware"] = {"gpu": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
                              "platform": platform.platform(), "torch_cpu_threads": torch.get_num_threads()}
        with isolated_rng():
            kwargs = {key: value for key, value in cfg["params"]["env"].items() if key not in ("name", "num_actors")}
            kwargs.update(no_grad=True, stochastic_init=False, device=algo.device, episode_length=protocol["horizon"])
            selection_env = MyoLeg26WalkEnv(num_envs=len(protocol["selection_episode_seeds"]), **kwargs)
            confirmation_env = MyoLeg26WalkEnv(num_envs=len(protocol["confirmation_episode_seeds"]), **kwargs)
        torch.cuda.synchronize()
        timers["setup"] = time.perf_counter() - setup_start
        if algo.env.task_contract.as_dict() != frozen["task_contract"] or algo.num_obs != 145:
            raise ValueError("Runtime task/observation contract differs from frozen experiment")
        original_step = algo.env.step

        def counted_step(*args, **kwargs):
            report["training_attempted_control_calls"] += 1
            result = original_step(*args, **kwargs)
            report["training_completed_control_calls"] += 1
            return result

        algo.env.step = counted_step
        algo.env.begin_epoch(epoch=0, max_epochs=epochs)
        algo._current_obs = algo.env.reset()
        for epoch in range(epochs + 1):
            if time.perf_counter() >= deadline:
                report.update(stop_reason="seed_wall_cap", censored=True)
                break
            if epoch:
                algo.env.begin_epoch(epoch=epoch - 1, max_epochs=epochs)
                timed("training_rollout", algo.collect_rollout)
                losses = timed("training_update", algo.update)
                if not all(np.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite PPO update loss")
                if not all(torch.isfinite(parameter).all() for network in (algo.actor, algo.critic)
                           for parameter in network.parameters()):
                    raise FloatingPointError("Nonfinite PPO parameters")
                algo.iter_count += 1
                report["completed_epochs"] = epoch
                report["losses"].append({"epoch": epoch, **losses})
                with progress_path.open("a", encoding="utf-8", newline="\n") as progress:
                    progress.write(json.dumps({"epoch": epoch, "losses": losses,
                        "training_control_transitions": report["training_completed_control_calls"] * algo.num_envs,
                        "seed_wall_seconds": time.perf_counter() - started,
                        "training_rollout_seconds": timers["training_rollout"],
                        "training_update_seconds": timers["training_update"]}, allow_nan=False) + "\n")
            if epoch in protocol["evaluation_epochs"]:
                filename = f"epoch_{epoch:04d}"
                timed("checkpoint_and_trace_io", lambda: algo.save(filename))
                checkpoint = output_dir / f"{filename}.pt"
                result = evaluate(epoch, selection_env, protocol["selection_episode_seeds"], "selection", checkpoint)
                if result["complete"] and (best is None or behavior_rank(result) > behavior_rank(best[2])):
                    best = (epoch, checkpoint, result)
            # Successful behavior never stops the fixed training budget early.
            if time.perf_counter() >= deadline:
                report.update(stop_reason="seed_wall_cap", censored=True)
                break
        if report["completed_epochs"] not in protocol["evaluation_epochs"]:
            timed("checkpoint_and_trace_io", lambda: algo.save("censored_last_training_state"))
        if best is not None:
            report["selected_checkpoint"] = {"epoch": best[0], "filename": best[1].name,
                                             "sha256": sha256(best[1])}
            if time.perf_counter() < deadline:
                timed("checkpoint_and_trace_io", lambda: algo.load(str(best[1])))
                evaluate(best[0], confirmation_env, protocol["confirmation_episode_seeds"], "confirmation", best[1], capture=True)
            if time.perf_counter() < deadline:
                evaluate(best[0], selection_env, protocol["audit_episode_seeds"], "stochastic_audit", best[1],
                         capture=True, deterministic=False)
        if time.perf_counter() >= deadline:
            report.update(stop_reason="seed_wall_cap", censored=True)
    except Exception as error:
        report.update(stop_reason="error", censored=True, error=f"{type(error).__name__}: {error}")
    finally:
        if algo is not None:
            algo.close()
        elapsed = time.perf_counter() - started
        timers["other_overhead"] = max(0., elapsed - sum(timers.values()))
        timers["total_seed_wall"] = elapsed
        report["timing_seconds"] = timers
        report["wall_cap_seconds"] = max_wall
        report["wall_cap_overshoot_seconds"] = max(0., elapsed - max_wall)
        training = report["training_completed_control_calls"] * protocol["num_actors"]
        evaluation = sum(row["completed_calls"] * row["worlds"] for row in report["evaluation_counters"])
        exact = (report["training_attempted_control_calls"] == report["training_completed_control_calls"]
                 and all(row["attempted_calls"] == row["completed_calls"] for row in report["evaluation_counters"]))
        report["accounting"] = {"training_control_transitions": training, "evaluation_control_transitions": evaluation,
                                "training_physics_steps": training * protocol["substeps"],
                                "evaluation_physics_steps": evaluation * protocol["substeps"],
                                "total_control_transitions": training + evaluation,
                                "total_physics_steps": (training + evaluation) * protocol["substeps"],
                                "exact_completed_calls": exact,
                                "failed_call_physics_count_unknown": not exact}
        write_json(output_dir / "result.json", report)
    return report


def main():
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-freeze", type=Path)
    mode.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true", help="Separate 2-actor/2-epoch/8-step protocol; never a behavior result")
    parser.add_argument("--max-seed-wall-seconds", type=float, default=1200.)
    args = parser.parse_args()
    if args.write_freeze:
        write_json(args.write_freeze, freeze_record(smoke=args.smoke))
        print(f"Freeze written: {args.write_freeze}; commit before training.")
        return 0
    args.seeds = args.seeds if args.seeds is not None else [0] if args.smoke else PROTOCOL["seeds"]
    args.epochs = args.epochs if args.epochs is not None else 2 if args.smoke else PROTOCOL["epochs"]
    if (not args.outdir or not 1 <= args.epochs <= (2 if args.smoke else PROTOCOL["epochs"])
            or not np.isfinite(args.max_seed_wall_seconds) or args.max_seed_wall_seconds <= 0
            or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set([0] if args.smoke else PROTOCOL["seeds"])):
        parser.error("Require new --outdir, unique registered seeds, bounded epochs, and finite positive wall cap")
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    frozen = validate_freeze(args.frozen_manifest, smoke=args.smoke)
    args.outdir.mkdir(parents=True)
    write_json(args.outdir / "run_manifest.json", {"freeze": frozen, "freeze_sha256": sha256(args.frozen_manifest),
               "repo_head": git("rev-parse", "HEAD"), "seeds": args.seeds, "epochs": args.epochs,
               "max_seed_wall_seconds": args.max_seed_wall_seconds,
               "pre_seed_setup_seconds": time.perf_counter() - run_started,
               "scope": "smoke_only" if args.smoke else "registered_full_budget" if args.epochs == 128 and args.seeds == PROTOCOL["seeds"] else "reduced_budget_or_seed_subset"})
    results = []
    for seed in args.seeds:
        # Collect environment/step-closure cycles before constructing another
        # seed's simulator. Cleanup is included in total run wall time.
        gc.collect()
        torch.cuda.empty_cache()
        result = run_seed(frozen, seed, args.epochs, args.outdir / f"seed_{seed}", args.max_seed_wall_seconds)
        results.append(result)
        print(f"Seed {seed}: {result['stop_reason']}, epochs={result['completed_epochs']}, wall={result['timing_seconds']['total_seed_wall']:.1f}s", flush=True)
    write_json(args.outdir / "results.json", {"seeds": results, "total_run_wall_seconds": time.perf_counter() - run_started,
               "all_full_budget": all(
        row["completed_epochs"] == 128 and not row["censored"] for row in results)})
    return int(any(row["stop_reason"] == "error" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
