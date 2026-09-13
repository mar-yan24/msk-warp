"""Select independent policy-visited states without consulting derivative results.

Input: five selected-checkpoint stochastic audit traces from the frozen PPO run.
Output: an exclusive NPZ archive and an adjacent coverage/provenance JSON report.
The fixed quotas can fail; an infeasible selection reports blocked, never relaxes.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np

from msk_warp.analysis.myoleg26_baseline import contiguous_windows
from msk_warp.models.myoleg26 import COLLISION_PREFIX, FOOT_BODIES, GROUND_NAME

SELECTION_SEED = 20260912
SUPPORT = ("left_support", "right_support", "double_support")
STRATA = (*SUPPORT, "airborne", "nonfoot_contact")
_runner_spec = importlib.util.spec_from_file_location(
    "selection_baseline_runner", Path(__file__).with_name("run_myoleg26_baseline.py"))
baseline_runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(baseline_runner)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compiled_hash(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


class Flow:
    """Integer residual flow for a small deterministic lower-bound assignment."""

    def __init__(self):
        self.edges = {}

    def add(self, source, target, capacity):
        forward = [target, int(capacity), None]
        reverse = [source, 0, forward]
        forward[2] = reverse
        self.edges.setdefault(source, []).append(forward)
        self.edges.setdefault(target, []).append(reverse)
        return forward

    def maximum(self, source, sink):
        total = 0
        while True:
            parents = {source: None}
            pending = deque([source])
            while pending and sink not in parents:
                node = pending.popleft()
                for edge in self.edges[node]:
                    if edge[1] > 0 and edge[0] not in parents:
                        parents[edge[0]] = (node, edge)
                        pending.append(edge[0])
            if sink not in parents:
                return total
            amount, node = 10**9, sink
            while node != source:
                previous, edge = parents[node]
                amount = min(amount, edge[1])
                node = previous
            node = sink
            while node != source:
                previous, edge = parents[node]
                edge[1] -= amount
                edge[2][1] += amount
                node = previous
            total += amount


def assign_episodes(episodes, primary_strata, *, per_seed=10, minimum=10, selection_seed=SELECTION_SEED):
    """Exact seed quotas, independent episodes and stratum lower bounds.

    episodes maps unique episode tuples (training seed, checkpoint SHA, episode
    seed) to available strata. Return None if no feasible assignment exists.
    """
    if not set(primary_strata) <= set(SUPPORT):
        raise ValueError("Primary strata must be foot-support modes")
    seeds = sorted({int(key[0]) for key in episodes})
    if seeds != list(range(5)):
        raise ValueError("All five registered training seeds are required")
    rng = np.random.default_rng(selection_seed)
    episode_keys = sorted(episodes)
    rng.shuffle(episode_keys)
    flow, balance, assigned = Flow(), Counter(), {}

    def bounded(source, target, lower, upper):
        edge = flow.add(source, target, upper - lower)
        balance[source] -= lower
        balance[target] += lower
        return edge, upper - lower

    total = len(seeds) * per_seed
    for seed in seeds:
        bounded("source", ("seed", seed), per_seed, per_seed)
    for key in episode_keys:
        node = ("episode", key)
        bounded(("seed", key[0]), node, 0, 1)
        strata = sorted(set(episodes[key]) - {"nonfoot_contact"})
        rng.shuffle(strata)
        for stratum in strata:
            if stratum not in STRATA:
                raise ValueError(f"Unknown stratum: {stratum}")
            assigned[key, stratum] = bounded(node, ("stratum", stratum), 0, 1)
    for stratum in (*SUPPORT, "airborne"):
        bounded(("stratum", stratum), "contact" if stratum in SUPPORT else "sink",
                minimum if stratum in primary_strata else 0, total)
    bounded("contact", "sink", minimum, total)
    bounded("sink", "source", 0, total)
    required = 0
    for node, value in list(balance.items()):
        if value > 0:
            flow.add("super_source", node, value)
            required += value
        elif value < 0:
            flow.add(node, "super_sink", -value)
    if flow.maximum("super_source", "super_sink") != required:
        return None
    result = {key: stratum for (key, stratum), (edge, capacity) in assigned.items() if capacity - edge[1]}
    if len(result) != total or any(sum(key[0] == seed for key in result) != per_seed for seed in seeds):
        raise AssertionError("Flow assignment violated seed/episode quotas")
    return result


def contact_stratum(model, data):
    """Classify the fresh native contact list using the frozen task margin rule."""
    ground = model.geom(GROUND_NAME).id
    left = right = nonfoot = False
    for contact in data.contact[:data.ncon]:
        if not contact.dist < contact.includemargin:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if ground not in (g1, g2):
            continue
        other = g2 if g1 == ground else g1
        name = model.geom(other).name
        if not name.startswith(COLLISION_PREFIX):
            raise ValueError("Unexpected active ground collision outside frozen proxies")
        body = model.body(int(model.geom_bodyid[other])).name
        if body not in FOOT_BODIES:
            nonfoot = True
        else:
            left |= body.endswith("_l")
            right |= body.endswith("_r")
    return ("nonfoot_contact" if nonfoot else "double_support" if left and right
            else "left_support" if left else "right_support" if right else "airborne")


def load_trace(path, *, frozen, freeze_sha256):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    if (metadata.get("schema_version") != "myoleg26-raw-traces-v1"
            or metadata.get("policy_mode") != "stochastic_normal" or not metadata.get("complete")
            or not metadata.get("warmstart_disabled")):
        raise ValueError("Require complete frozen stochastic held-out audit traces")
    expected = {"protocol": frozen["protocol"], "frozen_files": frozen["files"],
                "task_contract": frozen["task_contract"], "substeps": frozen["protocol"]["substeps"],
                "model_sha256": frozen["model_sha256"],
                "compiled_model_sha256": frozen["compiled_model_sha256"]}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Trace differs from committed freeze: {key}")
    if Path(metadata["model_path"]).resolve() != (baseline_runner.ROOT / frozen["model_path"]).resolve():
        raise ValueError("Trace model path differs from committed freeze")
    seed = metadata.get("training_seed")
    if type(seed) is not int or seed not in frozen["protocol"]["seeds"]:
        raise ValueError("Trace training seed is not registered")
    seed_result = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((path.parent.parent / "run_manifest.json").read_text(encoding="utf-8"))
    if (run_manifest.get("freeze_sha256") != freeze_sha256 or run_manifest.get("freeze") != frozen
            or seed not in run_manifest.get("seeds", [])
            or run_manifest.get("epochs") != frozen["protocol"]["epochs"]):
        raise ValueError("Parent run manifest differs from committed freeze or registered budget")
    selected = seed_result.get("selected_checkpoint", {})
    epoch = metadata.get("checkpoint_epoch")
    if (seed_result.get("seed") != seed or selected.get("epoch") != epoch
            or selected.get("sha256") != metadata["checkpoint_sha256"]
            or type(epoch) is not int or epoch not in frozen["protocol"]["evaluation_epochs"]
            or selected.get("filename") != f"epoch_{epoch:04d}.pt"
            or epoch > seed_result.get("completed_epochs", -1)):
        raise ValueError("Trace is not its seed result's behavior-selected checkpoint")
    count = len(arrays["step"])
    for key, width in (("qpos", 47), ("qvel", 46), ("act", 26), ("previous_action", 26), ("action", 26)):
        if arrays[key].shape != (count, width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid trace array {key}")
        if arrays[key].dtype != np.float32:
            raise ValueError(f"Expected captured float32 array {key}")
    for key in ("world", "episode_seed", "step", "done", "terminated"):
        if arrays[key].shape != (count,):
            raise ValueError(f"Invalid trace index array {key}")
    if not set(arrays["episode_seed"].tolist()) <= set(range(30000, 30016)):
        raise ValueError("Trace is not from the registered held-out stochastic audit")
    if (np.abs(arrays["action"]) > 1).any() or (np.abs(arrays["previous_action"]) > 1).any():
        raise ValueError("Signed policy action outside [-1,1]")
    checkpoint = path.parent / selected["filename"]
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash mismatch")
    return arrays, metadata


def select(paths, output, *, frozen_manifest):
    paths = sorted(map(Path, paths))
    output = Path(output)
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Selection output or report already exists")
    frozen = baseline_runner.validate_freeze(frozen_manifest)
    freeze_sha = sha256(frozen_manifest)
    traces = [load_trace(path, frozen=frozen, freeze_sha256=freeze_sha) for path in paths]
    seeds = [meta["training_seed"] for _, meta in traces]
    if sorted(seeds) != list(range(5)):
        raise ValueError("Require exactly one trace for each training seed 0..4")
    reference = traces[0][1]
    for _, metadata in traces:
        for key in ("compiled_model_sha256", "model_sha256", "task_contract", "substeps", "protocol", "frozen_files"):
            if metadata[key] != reference[key]:
                raise ValueError(f"Input traces disagree on frozen {key}")
    model_path = Path(reference["model_path"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if sha256(model_path) != reference["model_sha256"] or compiled_hash(model) != reference["compiled_model_sha256"]:
        raise ValueError("Current model does not match captured model")
    if not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_WARMSTART):
        raise ValueError("Hidden warmstart state would invalidate archive completeness")
    episodes, candidates, counts = {}, {}, Counter()
    for trace_index, (arrays, metadata) in enumerate(traces):
        windows = contiguous_windows(arrays, length=4)
        windows = windows[arrays["step"][windows[:, 0]] >= 4]
        for window in windows:
            row = int(window[0])
            # Fresh data is deliberate: no previous contact/solver state is reused.
            data = mujoco.MjData(model)
            for key in ("qpos", "qvel", "act"):
                getattr(data, key)[:] = arrays[key][row]
            data.ctrl[:] = .5 * (arrays["action"][row].astype(np.float64) + 1)
            mujoco.mj_forward(model, data)
            if np.any(data.warning.number):
                raise RuntimeError("Native warning while classifying captured state")
            stratum = contact_stratum(model, data)
            key = (int(metadata["training_seed"]), metadata["checkpoint_sha256"], int(arrays["episode_seed"][row]))
            episodes.setdefault(key, set()).add(stratum)
            candidates.setdefault((key, stratum), []).append((trace_index, window))
            counts[stratum] += 1
    eligible = {stratum: sum(stratum in available for available in episodes.values()) for stratum in STRATA}
    primary = [stratum for stratum in SUPPORT if eligible[stratum] >= 10]
    eligible_seeds = sorted({key[0] for key in episodes})
    if eligible_seeds != list(range(5)):
        assignments, selection_status = None, "blocked_missing_eligible_seed"
    elif not primary:
        assignments, selection_status = None, "blocked_no_primary_support_stratum"
    else:
        assignments = assign_episodes(episodes, primary)
        selection_status = "pass" if assignments is not None else "blocked_infeasible_coverage"
    metadata = {"schema_version": "myoleg26-visited-v1", "model_path": str(model_path.resolve()),
                "model_sha256": reference["model_sha256"], "compiled_model_sha256": reference["compiled_model_sha256"],
                "substeps": reference["substeps"], "task_contract": reference["task_contract"],
                "primary_strata": primary, "held_out": True, "policy_mode": "stochastic_normal",
                "step_semantics": "pre_action_episode_progress", "selection_seed": SELECTION_SEED,
                "selection_rule": "50 independent episodes, 10 per training seed, >=10 per primary support stratum; step>=4; four nonterminal controls",
                "frozen_manifest_path": str(Path(frozen_manifest).resolve()), "frozen_manifest_sha256": freeze_sha,
                "source_seed_results": {str((path.parent / "result.json").resolve()): sha256(path.parent / "result.json") for path in paths},
                "source_run_manifests": {str((path.parent.parent / "run_manifest.json").resolve()): sha256(path.parent.parent / "run_manifest.json") for path in paths},
                "selector_sha256": sha256(__file__), "raw_traces": {str(path.resolve()): sha256(path) for path in paths},
                "candidate_windows_by_stratum": dict(counts), "eligible_independent_episodes_by_stratum": eligible,
                "uncovered_strata": [stratum for stratum in STRATA if stratum not in primary],
                "eligible_training_seeds": eligible_seeds, "selection_status": selection_status}
    if assignments is not None:
        chosen = {key: [] for key in ("qpos", "qvel", "act", "previous_action", "action")}
        samples = []
        rng = np.random.default_rng(SELECTION_SEED)
        for key, stratum in sorted(assignments.items()):
            options = candidates[key, stratum]
            trace_index, window = options[int(rng.integers(len(options)))]
            arrays, source = traces[trace_index]
            row = int(window[0])
            for field in chosen:
                chosen[field].append(arrays[field][window] if field == "action" else arrays[field][row])
            samples.append({"sample_id": f"seed{key[0]}_episode{key[2]}_step{int(arrays['step'][row])}",
                            "seed": key[0], "checkpoint_sha256": key[1], "episode_id": key[2],
                            "step": int(arrays["step"][row]), "progress": int(arrays["step"][row]),
                            "contact_stratum": stratum, "trace_sha256": sha256(paths[trace_index])})
        metadata["samples"] = samples
        metadata["selected_by_stratum"] = dict(Counter(assignments.values()))
        with output.open("xb") as stream:
            np.savez_compressed(stream, **{key: np.stack(value) for key, value in chosen.items()},
                                metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs=5, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = select(args.traces, args.out, frozen_manifest=args.frozen_manifest)
    print(json.dumps({key: report[key] for key in ("selection_status", "primary_strata", "eligible_independent_episodes_by_stratum")}, indent=2))
    return int(report["selection_status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
