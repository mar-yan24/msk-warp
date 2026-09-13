"""Local action/state derivative gate on frozen, held-out MyoLeg26 policy states.

Native float64 finite differences and Warp AD start from the same float32 inputs.
Each open-loop action window is replayed without reset, timeout or policy feedback.
The qpos tests include all 46 simulation tangent coordinates, including the 40
equality-coupled internal coordinates: these are state inputs to actual BPTT.
Passing this local diagnostic does not authorize AD training or establish a
complete policy gradient. Direction bootstrap intervals are within one state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/check_trajopt_gradients.py"
_spec = importlib.util.spec_from_file_location("myoleg26_fd_helpers", HELPER_PATH)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
DEFAULT_EPS = _helpers.DEFAULT_EPS
TERMS = ("total_task_reward", "locomotion_only", "state_projection_0", "state_projection_1")
BLOCKS = ("action", "qpos_root_tangent", "qpos_internal_tangent", "qvel", "act")
CONTACT_STRATA = ("left_support", "right_support", "double_support")
TASK_CONTRACT = {
    "version": "myoleg26-walk-v1", "target_speed": 1.0, "velocity_variance": 0.25,
    "effort_weight": 0.01, "termination_height": 0.55, "termination_upright": 0.5,
}
STATE_SIZE = 124  # root xyz, root rotation matrix, 40 joint positions, 46 velocities, 26 activations


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compiled_model_sha256(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def validate_model(model):
    if (model.nq, model.nv, model.na, model.nu) != (47, 46, 26, 26):
        raise ValueError("expected official MyoLeg26 nq47/nv46/na26/nu26")
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or np.any(
        ~np.isin(model.jnt_type[1:], [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE])
    ):
        raise ValueError("expected one first free root followed by scalar joints")
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_WARMSTART:
        raise ValueError("fresh-state replay requires the frozen task's disabled warmstart")


def load_dataset(path, horizons=(1, 4)):
    """Strict, pickle-free frozen artifact loader; quantize once before any perturbation."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {k: np.asarray(archive[k], dtype=np.float32).astype(np.float64)
                  for k in ("qpos", "qvel", "act", "previous_action", "action")}
    if metadata.get("schema_version") != "myoleg26-visited-v1" or metadata.get("held_out") is not True:
        raise ValueError("expected held-out myoleg26-visited-v1 metadata")
    if metadata.get("substeps") != 4 or metadata.get("task_contract") != TASK_CONTRACT:
        raise ValueError("dataset differs from frozen four-substep task contract")
    n = len(arrays["qpos"])
    for name, width in (("qpos", 47), ("qvel", 46), ("act", 26), ("previous_action", 26)):
        if arrays[name].shape != (n, width):
            raise ValueError(f"{name} must have shape ({n}, {width})")
    actions = arrays["action"]
    if actions.ndim != 3 or actions.shape[0] != n or actions.shape[2] != 26 or actions.shape[1] < max(horizons):
        raise ValueError("action must be [N, Hmax, 26], with Hmax covering every requested horizon")
    if not n or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("dataset must contain nonempty finite states/actions")
    if any(np.any(np.abs(arrays[k]) > 1) for k in ("action", "previous_action")):
        raise ValueError("frozen signed actions must be in [-1, 1]")
    if np.any(np.abs(np.linalg.norm(arrays["qpos"][:, 3:7], axis=1) - 1) > 1e-5):
        raise ValueError("free-root quaternions must be normalized before dataset capture")
    samples = metadata.get("samples", [])
    if len(samples) != n or len({str(s["sample_id"]) for s in samples}) != n:
        raise ValueError("samples must provide one unique sample_id per row")
    for sample in samples:
        for key in ("seed", "checkpoint_sha256", "episode_id", "contact_stratum"):
            if key not in sample:
                raise ValueError(f"sample metadata missing {key}")
    primary = metadata.get("primary_strata", [])
    if not primary or len(set(primary)) != len(primary) or not set(primary).issubset(CONTACT_STRATA):
        raise ValueError("declare unique primary_strata from left/right/double_support explicitly before measurement")
    model_path = Path(metadata["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = mujoco.MjModel.from_xml_path(str(model_path))
    validate_model(model)
    if compiled_model_sha256(model) != metadata["compiled_model_sha256"]:
        raise ValueError("compiled model SHA256 differs from frozen dataset")
    return arrays, metadata, model, model_path


def coverage(metadata, *, minimum=10, required_seeds=(0, 1, 2, 3, 4)):
    samples = metadata["samples"]
    strata = sorted(set(metadata["primary_strata"]) | {s["contact_stratum"] for s in samples})
    counts, independent = {}, {}
    for stratum in strata:
        rows = [s for s in samples if s["contact_stratum"] == stratum]
        counts[stratum] = len(rows)
        independent[stratum] = len({(s["seed"], s["checkpoint_sha256"], str(s["episode_id"])) for s in rows})
    seeds = sorted({s["seed"] for s in samples})
    episode_count = len({(s["seed"], s["checkpoint_sha256"], str(s["episode_id"])) for s in samples})
    contact_count = sum(counts.get(s, 0) for s in CONTACT_STRATA)
    passed = (set(required_seeds).issubset(seeds) and episode_count == len(samples) and contact_count >= minimum and
              all(independent[s] >= minimum for s in metadata["primary_strata"]))
    return {"passed": passed, "primary_strata": metadata["primary_strata"], "counts": counts,
            "independent_episode_counts": independent, "minimum_per_stratum": minimum,
            "required_seeds": list(required_seeds), "observed_seeds": seeds,
            "contact_state_count": contact_count, "unique_episode_count": episode_count,
            "one_state_per_episode": episode_count == len(samples),
            "episode_counting_unit": "unique (policy seed, checkpoint, episode_id); not IID across policies sharing episode_id",
            "uncovered_strata": [s for s in strata if independent[s] < minimum]}


def qpos_tangent_lift(qpos):
    """d(mj_integratePos(qpos, delta, 1))/d(delta) at zero, in MuJoCo's body frame.

    Free-joint angular velocity is local: q_new = normalize(q * exp(delta/2)).
    This maps raw bridge qpos adjoints onto six root and forty internal directions.
    Dataset quaternion norm error is <=1e-5; its normalization is included here.
    """
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape != (47,) or np.linalg.norm(q[3:7]) == 0:
        raise ValueError("expected a valid official qpos")
    w, x, y, z = q[3:7] / np.linalg.norm(q[3:7])
    result = np.zeros((47, 46))
    result[:3, :3] = np.eye(3)
    result[3:7, 3:6] = .5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])
    result[7:, 6:] = np.eye(40)
    return result


def perturb_state(model, sample, block, delta):
    """Perturb the full simulator state; do not project equality coordinates away."""
    result = {k: np.array(v, dtype=np.float64, copy=True) for k, v in sample.items()}
    delta = np.asarray(delta, dtype=np.float64)
    if block in ("qpos_root_tangent", "qpos_internal_tangent"):
        tangent = np.zeros(model.nv)
        tangent[:6] = delta if block == "qpos_root_tangent" else 0
        tangent[6:] = delta if block == "qpos_internal_tangent" else 0
        mujoco.mj_integratePos(model, result["qpos"], tangent, 1.0)
    elif block in ("action", "qvel", "act"):
        result[block] += delta.reshape(result[block].shape)
    else:
        raise ValueError(f"unknown derivative block {block}")
    return result


def native_state_features(qpos, qvel, act):
    quat = np.array(qpos[3:7], dtype=np.float64, copy=True)
    quat /= np.linalg.norm(quat)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    return np.r_[qpos[:3], rotation, qpos[7:], qvel, act]


def torch_state_features(qpos, qvel, act):
    import torch
    quat = qpos[:, 3:7]
    w, x, y, z = (quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)).unbind(-1)
    rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), dim=-1)
    return torch.cat((qpos[:, :3], rotation, qpos[:, 7:], qvel, act), dim=-1)


def native_reward_components(model, data, excitation, contract, substeps):
    """Independent native pose/Jacobian transcription of the exact post-step reward."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    linear = np.empty((3, model.nv))
    angular = np.empty_like(linear)
    mujoco.mj_jacBody(model, data, linear, angular, pelvis)
    velocity = linear @ data.qvel
    rotation = data.xmat[pelvis].reshape(3, 3)
    tracking = np.exp(-((velocity[0] - contract["target_speed"])**2 + velocity[1]**2) /
                      contract["velocity_variance"])
    locomotion = tracking * np.clip(rotation[2, 1], 0, 1) * np.clip(rotation[0, 0], 0, 1)
    dt = substeps * model.opt.timestep
    locomotion *= dt
    effort = dt * contract["effort_weight"] * np.mean(excitation**2)
    return np.array([locomotion - effort, locomotion])


def native_rollout(model, sample, projections, *, substeps=4, contract=TASK_CONTRACT):
    """One fresh float64 MjData, all four scalar terms per frozen action rollout."""
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:] = sample["qpos"], sample["qvel"], sample["act"]
    accumulated = np.zeros(2)
    history = []
    for action in sample["action"]:
        excitation = .5 * (np.clip(action, -1, 1) + 1)
        data.ctrl[:] = excitation
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        components = native_reward_components(model, data, excitation, contract, substeps)
        accumulated += components
        state = native_state_features(data.qpos, data.qvel, data.act)
        history.append(np.r_[state, components])
    values = np.r_[accumulated, projections @ state]
    history = np.array(history)
    if any(int(w.number) for w in data.warning) or not np.isfinite(history).all():
        values[:] = np.nan
        history[:] = np.nan
    return {"terms": values, "history": history}


def sample_directions(seed, horizon, count=10):
    """Independent, reproducible standard-normal directions, normalized within each block."""
    rng = np.random.default_rng(seed)
    result = {}
    for block, size in zip(BLOCKS, (horizon * 26, 6, 40, 46, 26)):
        vectors = rng.normal(size=(count, size))
        result[block] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return result


def native_block_sweeps(model, sample, projections, directions, epsilons):
    """Reuse each perturbed native rollout for all scalar output terms."""
    result = {}
    for block, vectors in directions.items():
        def function(delta):
            perturbed = perturb_state(model, sample, block, delta)
            return native_rollout(model, perturbed, projections)["terms"]
        sweeps = [_helpers.epsilon_sweep(function, np.zeros(vectors.shape[1]), vector, epsilons)
                  for vector in vectors]
        result[block] = {"directions": vectors.tolist(), "epsilon_sweeps": sweeps, "terms": {}}
    return result


def apply_derivative_gates(blocks, gradients, settings):
    passed = True
    for block, result in blocks.items():
        directions = np.asarray(result["directions"])
        for index, term in enumerate(TERMS):
            gradient = np.asarray(gradients[term][block])
            projected = directions @ gradient
            windows = [s["windows"][index] for s in result["epsilon_sweeps"]]
            if any(w is None for w in windows):
                gate = {"passed": False, "reason": "no_epsilon_window",
                        "missing_directions": [i for i, w in enumerate(windows) if w is None]}
            else:
                gate = _helpers.compare_derivatives(
                    projected, [w["derivative"] for w in windows],
                    cosine_min=settings["cosine_min"], relative_max=settings["gradient_rtol"],
                    atol=settings["gradient_atol"], seed=settings["direction_seed"])
            gate["ad_gradient"] = gradient.tolist()
            gate["ad_directional"] = projected.tolist()
            result["terms"][term] = gate
            passed &= gate["passed"]
    return bool(passed)


def aggregate_state_gates(samples, horizons, *, seed=1701, resamples=2000):
    """Bootstrap shared reset/noise episode-ID clusters, without overriding any gate.

    Policies reuse audit reset/noise seeds: their selected rows are not IID.
    Every resample includes all selected rows sharing an episode_id together.
    The estimand is the row-weighted mean among resolved states; unresolved values
    contribute no value or weight. Policy seeds themselves are never resampled.
    """
    rng = np.random.default_rng(seed)
    cluster_names = sorted({str(s["metadata"]["episode_id"]) for s in samples})
    cluster_index = {name: i for i, name in enumerate(cluster_names)}
    row_clusters = np.array([cluster_index[str(s["metadata"]["episode_id"])] for s in samples], dtype=int)
    policy_seeds = sorted({s["metadata"]["seed"] for s in samples})
    draws = rng.integers(len(cluster_names), size=(resamples, len(cluster_names))) if cluster_names else None
    result = {"unit": "shared episode_id cluster (same reset/noise seed across fixed policies)",
              "scope": "Means across states with resolved finite statistics; missing/no-signal cosine excluded. "
                       "All selected rows sharing an episode_id are resampled together. "
                       "Conditional on frozen policies/selection, not an interval over new training seeds.",
              "selected_state_count": len(samples), "unique_episode_id_count": len(cluster_names),
              "episode_ids": cluster_names, "policy_seeds": policy_seeds, "policy_seed_count": len(policy_seeds),
              "small_cluster_limitation": "Only 16 reset/noise seeds were registered; cluster bootstrap intervals "
                                          "have limited precision and do not make the 50 policy-state rows IID. "
                                          "No interval is reported with fewer than 10 resolved clusters.",
              "bootstrap_resamples": resamples, "confidence": .95, "pass_override": False, "horizons": {}}
    for horizon in horizons:
        hresult = result["horizons"][str(horizon)] = {}
        for block in BLOCKS:
            bresult = hresult[block] = {}
            for term in TERMS:
                rows = [s.get("horizons", {}).get(str(horizon), {}).get("blocks", {}).get(block, {}).get("terms", {}).get(term, {})
                        for s in samples]
                summary = {"selected_state_count": len(rows), "passed_state_count": sum(r.get("passed") is True for r in rows),
                           "indeterminate_state_count": sum(r.get("reason") == "no_epsilon_window" for r in rows),
                           "no_signal_state_count": sum(r.get("signal") == "no_signal" for r in rows)}
                for statistic in ("projected_cosine", "relative_l2"):
                    valid = np.array([r.get(statistic) is not None and np.isfinite(r[statistic]) for r in rows], dtype=bool)
                    values = np.array([r[statistic] for r, keep in zip(rows, valid) if keep])
                    clusters = row_clusters[valid]
                    cluster_count = len(np.unique(clusters))
                    info = {"state_count": len(values), "resolved_episode_id_count": cluster_count,
                            "mean": None, "mean_ci95": None}
                    if len(values):
                        info["mean"] = float(values.mean())
                    if cluster_count >= 10:
                        totals = np.bincount(clusters, weights=values, minlength=len(cluster_names))
                        weights = np.bincount(clusters, minlength=len(cluster_names))
                        denominators = weights[draws].sum(axis=1)
                        usable = denominators > 0
                        means = totals[draws].sum(axis=1)[usable] / denominators[usable]
                        info["mean_ci95"] = np.quantile(means, [.025, .975]).tolist()
                    summary[statistic] = info
                bresult[term] = summary
    return result


def runtime_provenance(model):
    spec = importlib.util.find_spec("mujoco_warp")  # locate without importing or initializing Warp
    versions = {}
    for name in ("numpy", "torch", "mujoco", "mujoco-warp", "warp-lang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"repo": _helpers.git_provenance(ROOT),
            "backend": _helpers.git_provenance(Path(spec.origin).parent) if spec else None,
            "backend_path": spec.origin if spec else None, "packages": versions,
            "python": platform.python_version(), "compiled_model_sha256": compiled_model_sha256(model),
            "model_options": {k: float(getattr(model.opt, k)) for k in
                              ("timestep", "solver", "integrator", "jacobian", "iterations", "ls_iterations",
                               "tolerance", "disableflags", "enableflags")},
            "sources": {str(p.relative_to(ROOT)): sha256(p) for p in
                        (Path(__file__), HELPER_PATH, ROOT / "msk_warp/bridge.py", ROOT / "msk_warp/backend.py",
                         ROOT / "msk_warp/envs/myoleg26_walk.py", ROOT / "msk_warp/envs/myoleg26_task.py")}}


def settings_from_args(args):
    return {k: getattr(args, k) for k in
            ("horizons", "directions", "epsilons", "direction_seed", "forward_rtol", "forward_atol",
             "cosine_min", "gradient_rtol", "gradient_atol", "backward_mode")}


def build_report(args, metadata, model):
    rng = np.random.default_rng(args.direction_seed)
    projections = rng.normal(size=(2, STATE_SIZE))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    return {"schema_version": "myoleg26-derivatives-v1", "passed": False, "gate_status": "measurement_incomplete",
            "scope": "Local full-simulator-state open-loop derivatives, no reset, timeout, discount, critic or policy feedback. "
                     "Equality-dependent coordinates are included; no complete-policy or training authorization.",
            "input_precision": "Captured state/actions rounded once to float32, then promoted to float64 for native FD; "
                               "no quantization of finite-difference perturbations. Quaternion normalization uses mj_integratePos.",
            "ci_scope": "2000 paired bootstrap resamples of directions within a state; not a population/seed or full-gradient CI",
            "fd_window_criteria": {"consecutive_epsilons": 3, "selection": "CPU only; finest stable triple and all finer evidence",
                                   "central_rtol": 1e-3, "central_atol": 1e-7,
                                   "one_sided_symmetry_rtol": 1e-2, "one_sided_symmetry_atol": 1e-6},
            "provenance": runtime_provenance(model),
            "dataset": {"path": str(Path(args.states).resolve()), "sha256": sha256(args.states), "metadata": metadata},
            "coverage": coverage(metadata), "settings": settings_from_args(args),
            "state_projections": projections.tolist(), "samples": []}


def validate_native_cache(cached, report):
    """A cache is evidence only for the exact data, settings, source and runtime identity."""
    if cached.get("schema_version") != report["schema_version"] or cached.get("gate_status") != "native_complete_awaiting_ad":
        raise ValueError("native report is incomplete or has an incompatible schema")
    for key in ("dataset", "settings", "provenance", "state_projections"):
        if cached.get(key) != report[key]:
            raise ValueError(f"native report {key} differs from this measurement")
    if len(cached.get("samples", [])) != len(report["dataset"]["metadata"]["samples"]):
        raise ValueError("native report does not cover every selected state")
    for row, meta in zip(cached["samples"], report["dataset"]["metadata"]["samples"]):
        if row.get("sample_id") != meta["sample_id"] or row.get("metadata") != meta:
            raise ValueError("native report sample order/identity differs from dataset")
        for horizon in report["settings"]["horizons"]:
            item = row.get("horizons", {}).get(str(horizon), {})
            if np.asarray(item.get("native", {}).get("history", [])).shape != (horizon, STATE_SIZE+2):
                raise ValueError("native report missing complete trajectory history")
            for block, size in zip(BLOCKS, (horizon*26, 6, 40, 46, 26)):
                evidence = item.get("blocks", {}).get(block, {})
                count = report["settings"]["directions"]
                directions = np.asarray(evidence.get("directions", []))
                if directions.shape != (count, size) or not np.isfinite(directions).all():
                    raise ValueError("native report missing derivative directions")
                sweeps = evidence.get("epsilon_sweeps", [])
                if len(sweeps) != count or any(s.get("epsilons") != report["settings"]["epsilons"] or
                                               len(s.get("windows", [])) != len(TERMS) for s in sweeps):
                    raise ValueError("native report missing complete epsilon sweeps")


def build_warp_environment(model_path, args):
    """Explicit diagnostic opt-in only; the environment's default training guard stays intact."""
    from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv
    return MyoLeg26WalkEnv(num_envs=1, device=args.device, model_path=str(model_path),
                          model_contract="official", substeps=4, stochastic_init=False,
                          no_grad=False, allow_unvalidated_gradients=True, grad_contract="off",
                          backward_mode=args.backward_mode, njmax=1000)


def warp_rollout(env, sample, projections, *, gradients=False):
    """One fresh forward graph; four retained VJPs, with all input blocks per VJP.

    The bridge restores each saved state before replay and zeroes its Warp tapes.
    No Data survives across independent rollout measurements. Warmstart is disabled.
    """
    import torch
    import mujoco_warp as mjw
    from msk_warp import backend, bridge
    env.warp_data = backend.make_data(env.mjm, env.warp_model, 1, env._njmax, grad=True)
    mjw.reset_data(env.warp_model, env.warp_data)
    inputs = {k: torch.tensor(sample[k], dtype=torch.float32, device=env.device, requires_grad=gradients)
              for k in ("action", "qpos", "qvel", "act")}
    qpos, qvel, act = (inputs[k][None] for k in ("qpos", "qvel", "act"))
    accumulated = torch.zeros(2, device=env.device)
    history = []
    for action in inputs["action"]:
        excitation = .5 * (action.clamp(-1, 1) + 1)[None]
        qpos, qvel, act = bridge.WarpSimStep.apply(excitation, qpos, qvel, act, env)
        obs = env._obs_from_state(qpos, qvel, act, action[None])
        reward = env.task_contract.reward_terms(obs, excitation, env.control_dt)
        components = torch.stack((reward["locomotion_reward"] - reward["effort_cost"],
                                  reward["locomotion_reward"]), dim=-1)[0]
        accumulated = accumulated + components
        state = torch_state_features(qpos, qvel, act)[0]
        history.append(torch.cat((state, components)))
    values = torch.cat((accumulated, torch.tensor(projections, dtype=torch.float32, device=env.device) @ state))
    result = {"terms": values.detach().cpu().numpy(), "history": torch.stack(history).detach().cpu().numpy()}
    if gradients:
        result["gradients"], result["sanitization"] = {}, {}
        lift = qpos_tangent_lift(sample["qpos"])
        for index, term in enumerate(TERMS):
            with _helpers.count_sanitization(bridge) as counts:
                grads = torch.autograd.grad(values[index], tuple(inputs.values()), retain_graph=index < len(TERMS)-1)
            grads = {k: g.detach().cpu().numpy().ravel() for k, g in zip(inputs, grads)}
            tangent = grads.pop("qpos") @ lift
            grads["qpos_root_tangent"], grads["qpos_internal_tangent"] = tangent[:6], tangent[6:]
            result["gradients"][term] = grads
            result["sanitization"][term] = counts
    return result


def forward_gate(native, warp, settings):
    results = {}
    for field in ("history", "terms"):
        cpu, gpu = np.asarray(native[field]), np.asarray(warp[field])
        finite = np.isfinite(cpu).all() and np.isfinite(gpu).all()
        results[field] = {"passed": bool(finite and np.isclose(cpu, gpu, rtol=settings["forward_rtol"],
                                                              atol=settings["forward_atol"]).all()),
                          "max_absolute_error": float(np.max(np.abs(cpu-gpu))),
                          "native": cpu.tolist(), "warp": gpu.tolist()}
    return {"passed": all(r["passed"] for r in results.values()), **results}


def run(args):
    validate_settings(args)
    arrays, metadata, model, model_path = load_dataset(args.states, args.horizons)
    report = build_report(args, metadata, model)
    if args.native_report:
        cached = json.loads(Path(args.native_report).read_text(encoding="utf-8"))
        validate_native_cache(cached, report)
        report["samples"] = copy.deepcopy(cached["samples"])
        report["native_report"] = {"path": str(Path(args.native_report).resolve()), "sha256": sha256(args.native_report)}
    projections = np.asarray(report["state_projections"])
    samples = [{k: a[i] for k, a in arrays.items()} for i in range(len(arrays["qpos"]))]
    env = None if args.native_only else build_warp_environment(model_path, args)
    # Check every selected forward trajectory before any backward pass. Cached native
    # values still face a fresh Warp parity measurement on this invocation.
    forward_failures = []
    for index, (sample, meta) in enumerate(zip(samples, metadata["samples"])):
        if not args.native_report:
            report["samples"].append({"sample_id": meta["sample_id"], "metadata": meta, "horizons": {}})
        row = report["samples"][index]
        for horizon in args.horizons:
            short = {**sample, "action": sample["action"][:horizon]}
            if not args.native_report:
                native = native_rollout(model, short, projections)
                row["horizons"][str(horizon)] = {"native": {k: v.tolist() for k, v in native.items()}}
            item = row["horizons"][str(horizon)]
            if env is not None:
                warp = warp_rollout(env, short, projections)
                item["forward"] = forward_gate(item["native"], warp, report["settings"])
                if not item["forward"]["passed"]:
                    forward_failures.append({"sample_id": meta["sample_id"], "horizon": horizon,
                                             "max_absolute_error": max(item["forward"][f]["max_absolute_error"]
                                                                       for f in ("history", "terms"))})
    if env is not None:
        report["forward_summary"] = {"passed": not forward_failures, "tested": len(samples)*len(args.horizons),
                                     "failed": len(forward_failures), "failures": forward_failures}
    if forward_failures:
        report["gate_status"] = "forward_mismatch"
        return report
    report["required_horizons_measured"] = {1, 4}.issubset(args.horizons)
    passed = report["coverage"]["passed"] and report["required_horizons_measured"]
    for index, sample in enumerate(samples):
        row = report["samples"][index]
        for horizon in args.horizons:
            short = {**sample, "action": sample["action"][:horizon]}
            item = row["horizons"][str(horizon)]
            if not args.native_report:
                directions = sample_directions(args.direction_seed + index * 1009 + horizon, horizon, args.directions)
                item["blocks"] = native_block_sweeps(model, short, projections, directions, args.epsilons)
            if env is not None:
                ad = warp_rollout(env, short, projections, gradients=True)
                item["ad_replay_forward"] = forward_gate(item["native"], ad, report["settings"])
                item["sanitization"] = ad["sanitization"]
                item["unmodified_ad"] = all(c["calls"] == horizon and not c["nonfinite_entries"] and
                                             not c["clamped_finite_entries"] for c in ad["sanitization"].values())
                item["passed"] = (apply_derivative_gates(item["blocks"], ad["gradients"], report["settings"]) and
                                  item["unmodified_ad"] and item["ad_replay_forward"]["passed"])
                passed &= item["passed"]
        print(f"State {index+1}/{len(samples)} ({row['sample_id']}) complete", flush=True)
    report["passed"] = bool(passed and not args.native_only)
    if not args.native_only:
        report["aggregate"] = aggregate_state_gates(report["samples"], args.horizons, seed=args.direction_seed)
    report["gate_status"] = ("native_complete_awaiting_ad" if args.native_only else
                             "passed" if passed else "derivative_or_coverage_failure")
    return report


def validate_settings(args):
    """Diagnostics can tighten preregistered thresholds, never weaken qualification."""
    if args.directions < 10 or not args.horizons or any(h not in (1, 4) for h in args.horizons) or len(set(args.horizons)) != len(args.horizons):
        raise ValueError("require >=10 directions and distinct horizons chosen from 1 and 4")
    if args.native_report and args.native_only:
        raise ValueError("--native-report is for the AD stage, not --native-only")
    eps = np.asarray(args.epsilons)
    if len(eps) < 3 or not np.isfinite(eps).all() or np.any(eps <= 0) or np.any(np.diff(eps) >= 0):
        raise ValueError("epsilons must contain >=3 strictly decreasing positive finite values")
    for key, maximum in (("gradient_rtol", .1), ("gradient_atol", 1e-5),
                         ("forward_rtol", 1e-3), ("forward_atol", 1e-4)):
        if not np.isfinite(getattr(args, key)) or not 0 <= getattr(args, key) <= maximum:
            raise ValueError(f"{key} must be finite and in [0, {maximum}]")
    if not np.isfinite(args.cosine_min) or not .99 <= args.cosine_min <= 1:
        raise ValueError("cosine_min must be finite and in [.99, 1]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--native-only", action="store_true", help="compute reusable float64 FD evidence without importing Warp")
    parser.add_argument("--native-report", help="reuse only an exact data/source/runtime-matched native report")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--directions", type=int, default=10)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPS))
    parser.add_argument("--direction-seed", type=int, default=1701)
    parser.add_argument("--forward-rtol", type=float, default=1e-3)
    parser.add_argument("--forward-atol", type=float, default=1e-4)
    parser.add_argument("--cosine-min", type=float, default=.99)
    parser.add_argument("--gradient-rtol", type=float, default=.1)
    parser.add_argument("--gradient-atol", type=float, default=1e-5)
    parser.add_argument("--backward-mode", choices=("tape_per_substep", "tape"), default="tape_per_substep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        validate_settings(args)
    except ValueError as error:
        parser.error(str(error))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = path.open("w" if args.overwrite else "x", encoding="utf-8")
    except FileExistsError:
        parser.error("--out already exists; use a new artifact path or --overwrite explicitly")
    with output:
        json.dump({"passed": False, "gate_status": "measurement_incomplete", "arguments": vars(args)}, output)
        output.flush()
        started = time.monotonic()
        report = run(args)
        report["seconds"] = time.monotonic() - started
        output.seek(0)
        output.write(json.dumps(_helpers.json_safe(report), indent=2, allow_nan=False) + "\n")
        output.truncate()
    print(f"{report['gate_status']}: {path}", flush=True)
    return 0 if report["passed"] or report["gate_status"] == "native_complete_awaiting_ad" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
