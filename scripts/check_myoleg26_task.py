"""Audit the pinned MyoLeg26 forward physics and untrained task behavior.

Run after building assets. Results are exclusive-created JSON, with full matched
state comparisons and first-failure episode data. This never validates autodiff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from msk_warp import backend, resolve_model_path


TOLERANCES = {"qpos": (5e-5, 5e-4), "qvel": (1e-3, 1e-3), "act": (1e-6, 1e-5)}


def comparison(actual, expected, field):
    atol, rtol = TOLERANCES[field]
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        error = np.abs(actual - expected)
        scaled = error / (atol + rtol * np.abs(expected))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    finite_error = bool(np.isfinite(error).all() and np.isfinite(scaled).all())
    passed = finite and finite_error and bool((scaled <= 1).all())
    bad = np.flatnonzero(~np.isfinite(scaled))
    return {"pass": passed, "finite": finite, "error_finite": finite_error,
            "failure_reason": None if passed else ('nonfinite_state' if not finite else
                               'nonfinite_error' if not finite_error else 'tolerance_exceeded'),
            "nonfinite_actual_coordinates": np.flatnonzero(~np.isfinite(actual)).tolist(),
            "nonfinite_expected_coordinates": np.flatnonzero(~np.isfinite(expected)).tolist(),
            "max_absolute_error": float(error.max()) if np.isfinite(error).all() else None,
            "max_scaled_error": float(scaled.max()) if np.isfinite(scaled).all() else None,
            "worst_coordinate": int(bad[0]) if len(bad) else int(scaled.argmax())}


def write_report(output, result):
    """Retain nonfinite failures as explicit JSON nulls with their exact paths."""
    nonfinite_paths = []

    def clean(value, path):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, dict):
            return {key: clean(item, f'{path}.{key}') for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item, f'{path}[{index}]') for index, item in enumerate(value)]
        if isinstance(value, float) and not np.isfinite(value):
            nonfinite_paths.append(path)
            return None
        return value

    safe = clean(result, '$')
    safe['report_has_nonfinite_values'] = bool(nonfinite_paths)
    safe['nonfinite_json_paths'] = nonfinite_paths
    json.dump(safe, output, indent=2, allow_nan=False)
    output.write('\n')


def initial_states(model, seeds):
    """Matched, float32-representable states; never perturb dependent joints."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
    if key < 0:
        raise ValueError("Task asset needs its declared nominal stand keyframe")
    cases = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        qpos = model.key_qpos[key].copy()
        qvel = np.zeros(model.nv)
        qvel[:6] = rng.uniform(-.05, .05, 6)
        act = rng.uniform(.05, .4, model.na)
        ctrl = rng.uniform(.05, .7, model.nu)
        for condition, dz in (("contact", -.002), ("airborne", .4)):
            pose = qpos.copy()
            pose[2] += dz
            cases.append({"seed": seed, "condition": condition,
                          "state": {k: np.asarray(v, np.float32) for k, v in
                                    dict(qpos=pose, qvel=qvel, act=act, ctrl=ctrl).items()}})
    return cases


def check_forward(model_path, seeds):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    cases = initial_states(model, seeds)
    horizons = (1, 4, 16)
    native = [[None] * len(horizons) for _ in cases]
    for i, case in enumerate(cases):
        data = mujoco.MjData(model)
        for name, value in case['state'].items():
            getattr(data, name)[:] = value
        for step in range(1, max(horizons) + 1):
            mujoco.mj_step(model, data)
            if np.any(data.warning.number):
                raise RuntimeError(f"Native warning in seed {case['seed']} {case['condition']}")
            if step in horizons:
                native[i][horizons.index(step)] = {
                    name: getattr(data, name).copy() for name in TOLERANCES}

    started = time.perf_counter()
    model_wp = backend.put_model(model)
    data_wp = backend.make_data(model, model_wp, len(cases), njmax=1000, grad=False)
    mjw.reset_data(model_wp, data_wp)
    for name in ('qpos', 'qvel', 'act', 'ctrl'):
        values = np.stack([case['state'][name] for case in cases])
        wp.copy(getattr(data_wp, name), wp.array(values, dtype=wp.float32))
    wp.synchronize()
    setup_s = time.perf_counter() - started
    rows = []
    started = time.perf_counter()
    for step in range(1, max(horizons) + 1):
        mjw.step(model_wp, data_wp)
        if step in horizons:
            wp.synchronize()
            actual = {name: getattr(data_wp, name).numpy().astype(np.float64) for name in TOLERANCES}
            for i, case in enumerate(cases):
                fields = {name: comparison(actual[name][i], native[i][horizons.index(step)][name], name)
                          for name in TOLERANCES}
                rows.append({"seed": case['seed'], "condition": case['condition'],
                             "physics_steps": step, "fields": fields,
                             "pass": all(row['pass'] for row in fields.values())})
    wp.synchronize()
    return {"pass": all(row['pass'] for row in rows), "tolerances": TOLERANCES,
            "setup_s": setup_s, "stepping_and_readback_s": time.perf_counter() - started,
            "sampled_world_physics_steps": len(cases) * max(horizons), "checks": rows}


def measure_behavior(model_path, episodes, seconds, seed):
    from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv

    started = time.perf_counter()
    env = MyoLeg26WalkEnv(model_path=str(model_path), num_envs=episodes, no_grad=True,
                         stochastic_init=True, episode_length=100000)
    wp.synchronize()
    setup_s = time.perf_counter() - started
    dt = float(env.mjm.opt.timestep) * env.substeps
    steps = int(np.ceil(seconds / dt))
    conditions = []
    for condition in ('passive', 'neutral', 'random'):
        torch.manual_seed(seed)
        env.reset()
        qp, _, _ = env.state_tensors()
        start_pos = env._compute_pelvis_position(qp).cpu().numpy()
        generator = torch.Generator(device=env.device).manual_seed(seed + 10000)
        alive = np.ones(episodes, dtype=bool)
        lengths = np.zeros(episodes, dtype=np.int64)
        returns = np.zeros(episodes)
        effort = np.zeros(episodes)
        final_pos = start_pos.copy()
        reason = ['horizon'] * episodes
        end_failure_flags = [None] * episodes
        nonfinite_fields = [[] for _ in range(episodes)]
        started = time.perf_counter()
        actual_steps = 0
        for _ in range(steps):
            shape = (episodes, env.num_actions)
            if condition == 'passive':
                action = torch.full(shape, -1., device=env.device)
            elif condition == 'neutral':
                action = torch.zeros(shape, device=env.device)
            else:
                action = 2 * torch.rand(shape, generator=generator, device=env.device) - 1
            _, reward, done, extras, *_ = env.step(action)
            obs = extras['obs_before_reset']
            terminal_pos = extras['pelvis_position_before_reset'].detach().cpu().numpy()
            reward_np = reward.detach().cpu().numpy()
            fields_finite = {
                'observation': torch.isfinite(obs).all(dim=-1).cpu().numpy(),
                'pelvis_position': np.isfinite(terminal_pos).all(axis=-1),
                'reward': np.isfinite(reward_np),
            }
            finite = np.logical_and.reduce(list(fields_finite.values()))
            terminated = extras['terminated'].cpu().numpy().astype(bool)
            ended = done.detach().cpu().numpy().astype(bool)
            new = alive & (ended | ~finite)
            returns[alive] += reward_np[alive]
            effort[alive] += (.5 * (action + 1)).square().mean(dim=-1).cpu().numpy()[alive]
            final_pos[alive] = terminal_pos[alive]
            lengths[alive] += 1
            for i in np.flatnonzero(new):
                reason[i] = 'nonfinite' if not finite[i] else 'task_failure' if terminated[i] else 'timeout'
                nonfinite_fields[i] = [name for name, flags in fields_finite.items() if not flags[i]]
                if 'failure_flags' in extras:
                    end_failure_flags[i] = {name: bool(flags[i].item())
                                            for name, flags in extras['failure_flags'].items()}
            alive[new] = False
            actual_steps += 1
            if not alive.any():
                break
        wp.synchronize()
        elapsed = time.perf_counter() - started
        durations = lengths * dt
        displacement = final_pos[:, 0] - start_pos[:, 0]
        conditions.append({
            "condition": condition, "survival_fraction": float(alive.mean()),
            "first_episode_duration_s": durations.tolist(), "end_reason": reason,
            "end_failure_flags": end_failure_flags, "nonfinite_fields_at_end": nonfinite_fields,
            "forward_displacement_m": displacement.tolist(),
            "mean_forward_velocity_mps": (displacement / durations).tolist(),
            "return_until_end": returns.tolist(), "mean_excitation_squared": (effort / lengths).tolist(),
            "warm_rollout_with_diagnostics_s": elapsed,
            "simulated_control_transitions": actual_steps * episodes,
            "scored_first_episode_transitions": int(lengths.sum()),
            "simulated_physics_steps": actual_steps * episodes * env.substeps,
        })
    return {"contract": env.task_contract.as_dict(), "setup_s": setup_s,
            "episodes_per_condition": episodes, "seed": seed, "horizon_s": seconds,
            "control_dt": dt, "conditions": conditions}


def _revision(path):
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def application_state(root):
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'],
                            capture_output=True, text=True)
    # Hash the implementation used even when it has not yet been committed.
    sources = ['scripts/check_myoleg26_task.py', 'msk_warp/backend.py', 'msk_warp/__init__.py',
               'msk_warp/envs/myoleg26_walk.py', 'msk_warp/envs/myoleg26_task.py',
               'msk_warp/envs/base_env.py', 'msk_warp/bridge.py', 'msk_warp/models/myoleg26.py']
    return {'application_commit': _revision(root),
            'application_dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
            'application_status': status.stdout.splitlines() if status.returncode == 0 else None,
            'implementation_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                      for name in sources if (root / name).is_file()}}


def model_manifest_status(model_path):
    """Report all declared asset hashes; provenance checks do not change the parity gate."""
    manifest_path = model_path.parent / 'manifest.json'
    record = {'path': str(manifest_path), 'status': 'missing'}
    if not manifest_path.is_file():
        return record
    record['sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        declared = manifest['files']
        if not isinstance(declared, dict) or not declared:
            raise ValueError('manifest files must be a nonempty mapping')
        record['source'] = manifest.get('source')
        files = {}
        base = manifest_path.parent.resolve()
        for relative, expected in declared.items():
            path = (base / relative).resolve()
            row = {'expected_sha256': expected, 'actual_sha256': None}
            if not path.is_relative_to(base):
                row['status'] = 'unsafe_path'
            elif not path.is_file():
                row['status'] = 'missing'
            else:
                row['actual_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                row['status'] = 'match' if row['actual_sha256'] == expected else 'mismatch'
            files[relative] = row
        record['files'] = files
        relative_model = model_path.resolve().relative_to(base).as_posix()
        record['model_listed'] = relative_model in files
        record['status'] = ('verified' if all(row['status'] == 'match' for row in files.values())
                            and record['model_listed'] else 'mismatch')
    except Exception as exc:
        record.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='assets/myoleg26/flat_boxes.xml')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.episodes < 1 or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('seeds, episodes and seconds must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before initializing GPU or starting potentially expensive work.
    with args.out.open('x', encoding='utf-8') as output:
        root = Path(__file__).resolve().parents[1]
        result = {"requested_model": args.model, "autodiff_validated": False, "status": "started"}
        try:
            result.update(application_state(root))
            model_path = Path(resolve_model_path(args.model)).resolve()
            result['model'] = str(model_path)
            result['model_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
            result['model_manifest'] = model_manifest_status(model_path)
            result.update(mujoco=mujoco.__version__, warp=wp.__version__,
                          backend_commit=_revision(Path(mjw.__file__).resolve().parents[1]),
                          backend_path=mjw.__file__)
            result['gpu'] = torch.cuda.get_device_name()
            result['forward'] = check_forward(model_path, range(args.seed, args.seed + args.seeds))
            if not args.parity_only:
                result['behavior'] = measure_behavior(model_path, args.episodes, args.seconds, args.seed)
            result['status'] = 'completed'
        except Exception as exc:
            result['status'] = 'error'
            result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            write_report(output, result)
    print(json.dumps({'forward_pass': result['forward']['pass'], 'output': str(args.out)}, indent=2))
    raise SystemExit(0 if result['forward']['pass'] else 1)


if __name__ == '__main__':
    main()
