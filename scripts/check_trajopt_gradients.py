"""Bounded AD check of the *actual* trajopt rollout against native float64 MuJoCo.

No optimisation is performed. Random directional derivatives are checked separately for
u_logits and z, for velocity and closure projections (and optionally the augmented objective).
Direction statistics describe the explicitly sampled states only, not a population of states
or a full-gradient cosine. At least ten states are still needed to discharge VALIDITY BE-07.
The CPU reference deliberately does not use the activation-fixed-point return map: trajopt
warms up the entire mechanical trajectory from neutral activation and differentiates it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)


def load_trajopt():
    spec = importlib.util.spec_from_file_location("trajopt_gradient_target", ROOT / "scripts/trajopt_hopper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_initial_state(z, na):
    """Independent float64 transcription; a CPU test pins parity with initial_state."""
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (11,):
        raise ValueError("z must contain the 11 mechanical parameters")
    qpos = np.r_[0.0, 0.15 * np.tanh(z[0]), 0.30 * np.tanh(z[1]),
                 -(0.05 + 1.10 * (0.5 + 0.5 * np.tanh(z[2:4] / 2))),
                 0.70 * np.tanh(z[4])]
    qvel = np.array([2., 2., 3., 5., 5., 5.]) * np.tanh(z[5:11])
    return qpos, qvel, np.full(na, 0.5)


def sample_parameters(seed, cycle, nu):
    """Draw mechanics first so a seed identifies the same z across actuator counts."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=11).astype(np.float32)
    u = rng.normal(size=(cycle, nu)).astype(np.float32)
    return u, z


def native_rollout(model, u, z, scales, *, substeps=4, warmup=1, muscle=False, strength=1.0):
    """Fresh MjData on EVERY call, unchanged model options, warm-up and post-step clamps.

    Output order is velocity, 11 scaled closure components, mean height violation.
    Hidden solver state persists within a trajectory just as in the actual forward rollout.
    """
    if (model.nq, model.nv) != (6, 6):
        raise ValueError("this diagnostic only supports the scalar-joint hopper layout")
    u, scales = np.asarray(u, dtype=np.float64), np.asarray(scales, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != model.nu or len(u) < 1:
        raise ValueError("u must have shape (positive cycle length, model.nu)")
    if scales.shape != (11,) or not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("scales must contain 11 positive finite entries")
    if substeps < 1 or warmup < 0:
        raise ValueError("substeps must be positive and warmup nonnegative")
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:], d.act[:] = native_initial_state(z, model.na)
    action = np.tanh(u)
    controls = (0.5 * (action + 1) if muscle else action) * strength
    violation = 0.0
    for step in range((warmup + 1) * len(u)):
        if step == warmup * len(u):
            start_x = float(d.qpos[0])
            start_shape = np.r_[d.qpos[1:], d.qvel]
        d.ctrl[:] = controls[step % len(u)]
        for _ in range(substeps):
            mujoco.mj_step(model, d)
        d.qpos[:] = np.clip(d.qpos, -100, 100)
        d.qvel[:] = np.clip(d.qvel, -100, 100)
        violation += max(0.0, -0.45 - d.qpos[1])
    velocity = (d.qpos[0] - start_x) / (len(u) * substeps * model.opt.timestep)
    error = (np.r_[d.qpos[1:], d.qvel] - start_shape) / scales
    values = np.r_[velocity, error, violation / ((warmup + 1) * len(u))]
    # MuJoCo can reset an unstable state to finite values. That is not a valid FD sample.
    if any(int(w.number) for w in d.warning):
        return np.full_like(values, np.nan)
    return values


def project_terms(values, projections, *, auglag=False, rho=4.0, multipliers=None, mu=20.0):
    error = values[1:12]
    result = np.r_[values[0], projections @ error]
    if auglag:
        result = np.r_[result, values[0] - multipliers @ error - 0.5 * rho * (error @ error) - mu * values[-1]]
    return result


def epsilon_sweep(function, x, direction, epsilons=DEFAULT_EPS, *, stability_rtol=1e-3,
                  stability_atol=1e-7, symmetry_rtol=1e-2, symmetry_atol=1e-6):
    """Certify a fine three-epsilon window using CPU values alone, BEFORE consulting AD.

    Consecutive central differences must agree and both one-sided slopes must approach
    the central slope. The latter rejects a kink whose central difference is deceptively stable.
    Never discard contradictory finer evidence: every sampled epsilon below a proposed window
    must also be finite and satisfy both checks. Unresolved small scales or noise fail closed.
    A window is empirical local evidence, not a proof of smoothness or contact-mode invariance.
    """
    eps = np.asarray(epsilons, dtype=float)
    if len(eps) < 3 or np.any(eps <= 0) or np.any(np.diff(eps) >= 0):
        raise ValueError("provide at least three strictly decreasing positive epsilons")
    base = np.atleast_1d(function(x))
    plus = np.array([np.atleast_1d(function(x + e * direction)) for e in eps])
    minus = np.array([np.atleast_1d(function(x - e * direction)) for e in eps])
    central = (plus - minus) / (2 * eps[:, None])
    forward, backward = (plus - base) / eps[:, None], (base - minus) / eps[:, None]
    windows = []
    for column in range(len(base)):
        chosen = None
        for start in reversed(range(len(eps) - 2)):
            indices = slice(start, None)
            local = central[indices, column]
            stable = np.all(np.abs(local - local[-1]) <= stability_atol + stability_rtol * np.abs(local[-1]))
            symmetric = np.all(np.abs(forward[indices, column] - backward[indices, column])
                               <= symmetry_atol + symmetry_rtol * np.abs(local))
            if np.all(np.isfinite(local)) and stable and symmetric:
                chosen = {"epsilons": eps[start:start + 3].tolist(),
                          "validated_finer_epsilons": eps[start + 3:].tolist(),
                          "derivative_epsilon": float(eps[-1]), "derivative": float(local[-1])}
                break  # finest certified window; AD is never used in the selection
        windows.append(chosen)
    return {"epsilons": eps.tolist(), "central": central.tolist(),
            "forward": forward.tolist(), "backward": backward.tolist(), "windows": windows}


def compare_derivatives(ad, fd, *, cosine_min=0.99, relative_max=0.1, atol=1e-5, seed=0):
    """Projected derivative-vector metrics and paired direction-bootstrap 95% intervals."""
    ad, fd = np.asarray(ad), np.asarray(fd)
    finite = bool(np.isfinite(ad).all() and np.isfinite(fd).all())
    if not finite:
        return {"passed": False, "reason": "nonfinite_derivative"}

    def metrics(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(np.clip(a @ b / denom, -1, 1)) if denom > 1e-30 else None
        relative = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), atol))
        return cosine, relative

    cosine, relative = metrics(ad, fd)
    close = np.abs(ad - fd) <= atol + relative_max * np.abs(fd)
    zero_reference = bool(np.linalg.norm(fd) <= atol)
    passed = bool(close.all() and (zero_reference or (cosine is not None and cosine >= cosine_min))
                  and (zero_reference or relative <= relative_max))
    rng = np.random.default_rng(seed)
    draws = [metrics(ad[i], fd[i]) for i in rng.integers(0, len(ad), (2000, len(ad)))]
    cosines = [c for c, _ in draws if c is not None]
    return {"passed": passed, "direction_count": len(ad), "projected_cosine": cosine,
            "projected_cosine_ci95": np.quantile(cosines, [.025, .975]).tolist() if cosines else None,
            "relative_l2": relative, "relative_l2_ci95": np.quantile([r for _, r in draws], [.025, .975]).tolist(),
            "max_absolute_error": float(np.max(np.abs(ad - fd))), "directions_within_tolerance": int(close.sum()),
            "zero_reference": zero_reference, "signal": "no_signal" if zero_reference else "resolved",
            "ad": ad.tolist(), "fd": fd.tolist()}


@contextlib.contextmanager
def count_sanitization(bridge):
    """Observe our bridge boundary; never alter its numerical result or Warp internals."""
    import torch
    original = bridge._sanitize
    counts = {"calls": 0, "nonfinite_entries": 0, "clamped_finite_entries": 0}

    def observed(*tensors):
        counts["calls"] += 1
        for tensor in tensors:
            finite = torch.isfinite(tensor)
            counts["nonfinite_entries"] += int((~finite).sum().item())
            counts["clamped_finite_entries"] += int((finite & (tensor.abs() > bridge.GRAD_CLAMP)).sum().item())
        return original(*tensors)

    bridge._sanitize = observed
    try:
        yield counts
    finally:
        bridge._sanitize = original


def git_provenance(directory):
    directory = Path(directory).resolve()
    repository = next((p for p in (directory, *directory.parents) if (p / ".git").exists()), directory)
    def git(*args):
        # Read-only, command-scoped ownership exception for the user's editable worktree.
        # Never change their global safe.directory configuration from a measurement script.
        result = subprocess.run(["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(directory), *args],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "status": git("status", "--porcelain"), "tracked_diff_sha256": hashlib.sha256((git("diff", "HEAD") or "").encode()).hexdigest()}


def provenance(env, cfg_path, scales_path):
    import mujoco_warp
    # A compiled-model digest covers included assets and runtime options, unlike XML alone.
    buffer = np.empty(mujoco.mj_sizeModel(env.mjm), dtype=np.uint8)
    mujoco.mj_saveModel(env.mjm, buffer=buffer)
    versions = {}
    for name in ("numpy", "torch", "mujoco", "mujoco-warp", "warp-lang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"repo": git_provenance(ROOT), "backend": git_provenance(Path(mujoco_warp.__file__).parent),
            "backend_path": mujoco_warp.__file__, "packages": versions, "python": platform.python_version(),
            "compiled_model_sha256": hashlib.sha256(buffer.tobytes()).hexdigest(),
            "inputs": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (cfg_path, scales_path, ROOT / "scripts/trajopt_hopper.py", Path(__file__))},
            "model_options": {name: float(getattr(env.mjm.opt, name)) for name in
                              ("timestep", "solver", "integrator", "jacobian", "iterations", "ls_iterations", "tolerance", "disableflags", "enableflags")},
            "backward_mode": env.backward_mode, "substeps": env.substeps}


def run(args):
    import torch
    import mujoco_warp as mjw
    from msk_warp import bridge

    trajopt = load_trajopt()
    cfg, cfg_path = trajopt.load_cfg(args.cfg)
    name = cfg["params"]["env"]["name"]
    if name not in ("HopperMotor", "HopperMuscle"):
        raise ValueError("--cfg must use HopperMotor or HopperMuscle")
    scales = np.asarray(json.loads(Path(args.scales).read_text())["per_world"][0]["scales"], dtype=np.float32).astype(float)
    seeds = [args.seed + i for i in range(args.samples)]
    nu = 6 if name == "HopperMuscle" else 3
    parameters = [sample_parameters(seed, args.cycle, nu) for seed in seeds]
    rng = np.random.default_rng(args.direction_seed)
    projections = rng.normal(size=(args.closure_projections, 11))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    multipliers = np.full(11, args.multiplier)
    names = ["velocity"] + [f"closure_projection_{i}" for i in range(len(projections))]
    if args.auglag:
        names.append("auglag_total")
    env = trajopt.build_env(cfg, args.samples, args.device, (args.warmup + 1) * args.cycle + 1)
    report = {"passed": False, "scope": "Directional derivatives at explicit sampled states; not full-gradient cosines or population evidence.",
              "arguments": vars(args), "provenance": provenance(env, cfg_path, args.scales), "state_count": args.samples,
              "fd_window_criteria": {"consecutive_epsilons": 3,
                                     "selection": "finest stable triple; all finer evidence must agree; CPU only",
                                     "central_rtol": 1e-3, "central_atol": 1e-7,
                                     "one_sided_symmetry_rtol": 1e-2, "one_sided_symmetry_atol": 1e-6},
              "ci_scope": "2000 paired bootstrap resamples of directions within one state, 95% percentile interval",
              "closure_projections": projections.tolist(),
              "samples": [{"seed": seed, "u_logits": u.tolist(), "z": z.tolist(), "blocks": {}}
                          for seed, (u, z) in zip(seeds, parameters)], "ad_terms": {}}
    cpu_kwargs = dict(substeps=env.substeps, warmup=args.warmup, muscle=name == "HopperMuscle", strength=env.action_strength)
    term_kwargs = dict(auglag=args.auglag, rho=args.rho, multipliers=multipliers, mu=args.mu)
    cpu = np.array([native_rollout(env.mjm, u, z, scales, **cpu_kwargs) for u, z in parameters])
    gradient_terms = []
    for index, term in enumerate(names):
        print(f"AD term {index + 1}/{len(names)}: {term}", flush=True)
        env.clear_grad()
        mjw.reset_data(env.warp_model, env.warp_data)
        u = torch.tensor(np.stack([p[0] for p in parameters]), device=args.device, requires_grad=True)
        z = torch.tensor(np.stack([p[1] for p in parameters]), device=args.device, requires_grad=True)
        v, error, residual, violation = trajopt.rollout(env, u, z, torch.tensor(scales, dtype=torch.float32, device=args.device),
                                                       args.cycle, env.mjm.na, args.device, warmup=args.warmup)
        raw = torch.cat([v[:, None], error, violation[:, None]], dim=1).detach().cpu().numpy()
        parity = np.isclose(raw, cpu, rtol=args.forward_rtol, atol=args.forward_atol)
        report["forward"] = {"passed": bool(parity.all()), "cpu": cpu.tolist(), "warp": raw.tolist(),
                             "max_absolute_error": float(np.max(np.abs(raw - cpu)))}
        if not parity.all():
            report["failure"] = "forward_mismatch"
            return report
        if term == "velocity":
            objective = v
        elif term == "auglag_total":
            objective = v - args.multiplier * error.sum(dim=-1) - 0.5 * args.rho * error.square().sum(dim=-1) - args.mu * violation
        else:
            objective = (error * torch.tensor(projections[index - 1], dtype=torch.float32, device=args.device)).sum(dim=-1)
        with count_sanitization(bridge) as counts:
            gu, gz = torch.autograd.grad(objective.sum(), (u, z))
        gradient_terms.append((gu.cpu().numpy(), gz.cpu().numpy()))
        report["ad_terms"][term] = {"sanitization": counts, "unmodified_ad": not (counts["nonfinite_entries"] or counts["clamped_finite_entries"])}
        del v, error, residual, violation, objective, gu, gz, u, z
    passed = all(t["unmodified_ad"] for t in report["ad_terms"].values())
    for sample, ((u, z), seed) in enumerate(zip(parameters, seeds)):
        row = report["samples"][sample]
        for block_index, block in enumerate(("u_logits", "z")):
            base = (u if block_index == 0 else z).astype(float)
            directions = rng.normal(size=(args.directions,) + base.shape)
            directions /= np.linalg.norm(directions.reshape(args.directions, -1), axis=1).reshape((-1,) + (1,) * base.ndim)
            def function(x):
                values = native_rollout(env.mjm, x if block_index == 0 else u, z if block_index == 0 else x, scales, **cpu_kwargs)
                return project_terms(values, projections, **term_kwargs)
            sweeps = [epsilon_sweep(function, base, d, args.epsilons) for d in directions]
            block_result = {"directions": directions.tolist(), "epsilon_sweeps": sweeps, "terms": {}}
            for term_index, term in enumerate(names):
                windows = [s["windows"][term_index] for s in sweeps]
                if any(w is None for w in windows):
                    result = {"passed": False, "reason": "no_epsilon_window", "missing_directions": [i for i, w in enumerate(windows) if w is None]}
                else:
                    gradient = gradient_terms[term_index][block_index][sample]
                    ad = directions.reshape(args.directions, -1) @ gradient.ravel()
                    result = compare_derivatives(ad, [w["derivative"] for w in windows], cosine_min=args.cosine_min,
                                                 relative_max=args.gradient_rtol, atol=args.gradient_atol, seed=args.direction_seed)
                block_result["terms"][term] = result
                passed &= result["passed"]
            row["blocks"][block] = block_result
        print(f"CPU comparisons complete for state seed {seed}", flush=True)
    report["passed"] = bool(passed)
    return report


def json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--scales", required=True)
    parser.add_argument("--cycle", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1, help="independent explicit random states; directions are NOT states")
    parser.add_argument("--directions", type=int, default=10)
    parser.add_argument("--closure-projections", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPS))
    parser.add_argument("--forward-rtol", type=float, default=1e-3)
    parser.add_argument("--forward-atol", type=float, default=1e-4)
    parser.add_argument("--cosine-min", type=float, default=0.99)
    parser.add_argument("--gradient-rtol", type=float, default=0.1)
    parser.add_argument("--gradient-atol", type=float, default=1e-5)
    parser.add_argument("--auglag", action="store_true")
    parser.add_argument("--rho", type=float, default=4.0)
    parser.add_argument("--multiplier", type=float, default=0.0, help="fixed multiplier in every closure coordinate")
    parser.add_argument("--mu", type=float, default=20.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing output artifact")
    args = parser.parse_args()
    if min(args.cycle, args.samples, args.closure_projections) < 1 or args.directions < 10 or args.warmup < 0:
        parser.error("positive cycle/samples/projections, >=10 directions and nonnegative warmup required")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = path.open("w" if args.overwrite else "x", encoding="utf-8")
    except FileExistsError:
        parser.error("--out already exists; choose a new path or explicitly pass --overwrite")
    with output:
        # Reserve before importing/initialising the GPU; interrupted runs remain identifiable.
        json.dump({"passed": False, "failure": "measurement_incomplete", "arguments": vars(args)}, output)
        output.flush()
        started = time.monotonic()
        report = run(args)
        report["seconds"] = time.monotonic() - started
        output.seek(0)
        output.write(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
        output.truncate()
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
