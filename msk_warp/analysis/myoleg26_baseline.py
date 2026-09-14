"""First-episode policy evaluation and accounting; no simulator adjoints."""

from __future__ import annotations

from contextlib import contextmanager
import random
import time

import numpy as np
import torch


@contextmanager
def isolated_rng():
    """Evaluation must not advance the training reset/action random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def episode_initial_states(qpos0, qvel0, na, seeds):
    """Independent per-episode streams with the official root-noise distribution.

    Draw order is x translation, height, six root velocities, world yaw. Internal
    coordinates stay at the nominal stand key; activation starts at zero.
    """
    poses, velocities = [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        qpos, qvel = np.asarray(qpos0).copy(), np.asarray(qvel0).copy()
        qpos[0] += rng.uniform(-0.05, 0.05)
        qpos[2] += rng.uniform(0, 0.02)
        qvel[:6] += rng.uniform(-0.05, 0.05, 6)
        half_yaw = rng.uniform(-0.05, 0.05) / 2
        c, s = np.cos(half_yaw), np.sin(half_yaw)
        w, x, y, z = qpos[3:7]
        quat = np.array([c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w])
        qpos[3:7] = quat / np.linalg.norm(quat)
        poses.append(qpos)
        velocities.append(qvel)
    return (np.asarray(poses, dtype=np.float32), np.asarray(velocities, dtype=np.float32),
            np.zeros((len(seeds), na), dtype=np.float32))


def behavior_rank(evaluation):
    """Prespecified lexicographic selection; exact ties keep earlier candidates."""
    summary = evaluation["summary"]
    return (summary["survival_fraction"], -summary["mean_episode_velocity_2d_rmse_mps"],
            summary["forward_speed_band_fraction"])


def write_initial_state(env, initial):
    """Install in live Warp storage: state_tensors() returns detached copies."""
    import warp as wp

    for name, source in zip(("qpos", "qvel", "act"), initial):
        target = wp.to_torch(getattr(env.warp_data, name))
        target.copy_(torch.as_tensor(source, device=target.device))


def aggregate_episodes(episodes, horizon, control_dt, simulated_steps, worlds, substeps):
    lengths = np.array([row["length"] for row in episodes])
    total = int(lengths.sum())
    if total <= 0:
        raise ValueError("evaluation contains no scored transitions")
    summary = {
        "survival_fraction": float(np.mean([row["survived"] for row in episodes])),
        "mean_forward_velocity_mps": float(np.mean([row["mean_forward_velocity_mps"] for row in episodes])),
        "mean_displacement_m": float(np.mean([row["displacement_m"] for row in episodes])),
        "mean_first_episode_duration_s": float(lengths.mean() * control_dt),
        "forward_time_weighted_absolute_speed_error_mps": float(sum(row["forward_absolute_error_sum"] for row in episodes) / total),
        "pooled_velocity_2d_rmse_mps": float(np.sqrt(sum(row["velocity_2d_squared_error_sum"] for row in episodes) / total)),
        "mean_episode_velocity_2d_rmse_mps": float(np.mean([
            np.sqrt(row["velocity_2d_squared_error_sum"] / row["length"]) for row in episodes])),
        "forward_speed_band_fraction": float(sum(row["speed_band_steps"] for row in episodes) / total),
        "mean_excitation": float(sum(row["excitation_sum"] for row in episodes) / total),
        "mean_excitation_squared": float(sum(row["excitation_squared_sum"] for row in episodes) / total),
    }
    accounting = {
        "simulated_control_transitions": int(simulated_steps * worlds),
        "scored_first_episode_transitions": total,
        "ignored_restarted_world_transitions": int(simulated_steps * worlds - total),
        "simulated_physics_steps": int(simulated_steps * worlds * substeps),
    }
    return {"summary": summary, "accounting": accounting, "episodes": episodes,
            "horizon_control_steps": horizon, "control_dt": control_dt}


@torch.no_grad()
def evaluate_policy(env, actor, obs_rms, seeds, *, horizon=500, capture=False,
                    deterministic=True, deadline=None, transition_counter=None):
    """Score only each world's first episode; count every actually stepped world.

    The env is a dedicated evaluation instance. Reset samples depend only on the
    episode seed, not batch order or the training random stream. Captured arrays
    are pre-step inputs, with terminal flags from that same transition.
    """
    if len(seeds) != env.num_envs or horizon < 1:
        raise ValueError("episode seeds must match evaluation worlds; horizon must be positive")
    with isolated_rng():
        env.reset()
        initial = episode_initial_states(
            env.start_qpos[0].cpu().numpy(), env.start_qvel[0].cpu().numpy(), env.mjm.na, seeds,
        )
        write_initial_state(env, initial)
        env.calculateObservations()
        obs = env.obs_buf
        start_position = env._compute_pelvis_position(env.state_tensors()[0]).cpu().numpy()
        episodes = [{"episode_seed": int(seed), "world": i, "length": 0, "survived": False,
                     "displacement_m": 0., "forward_absolute_error_sum": 0., "velocity_2d_squared_error_sum": 0.,
                     "speed_band_steps": 0, "excitation_sum": 0., "excitation_squared_sum": 0.,
                     "return": 0., "end_reason": "horizon", "failure_flags": {}}
                    for i, seed in enumerate(seeds)]
        alive = np.ones(env.num_envs, dtype=bool)
        generators = [torch.Generator(device=env.device).manual_seed(int(seed) + 1000000000)
                      for seed in seeds] if not deterministic else None
        chunks = {key: [] for key in ("qpos", "qvel", "act", "previous_action", "action",
                                      "world", "episode_seed", "step", "done", "terminated")}
        for step in range(horizon):
            normalized = obs_rms.normalize(obs) if obs_rms is not None else obs
            if deterministic:
                action = actor(normalized, deterministic=True).tanh()
            else:
                _, mean, std = actor.forward_with_dist(normalized, deterministic=True)
                noise = torch.stack([torch.randn(env.num_actions, device=env.device, generator=generator)
                                     for generator in generators])
                action = (mean + std * noise).tanh()
            if not torch.isfinite(action).all():
                raise FloatingPointError("nonfinite evaluation action")
            scored_worlds = np.flatnonzero(alive)
            if capture:
                for key, state in zip(("qpos", "qvel", "act"), env.state_tensors()):
                    chunks[key].append(state[scored_worlds].cpu().numpy().copy())
                chunks["previous_action"].append(env.actions[scored_worlds].cpu().numpy().copy())
                chunks["action"].append(action[scored_worlds].cpu().numpy().copy())
                chunks["world"].append(scored_worlds)
                chunks["episode_seed"].append(np.asarray(seeds)[scored_worlds])
                chunks["step"].append(np.full(len(scored_worlds), step, dtype=np.int64))
            if transition_counter is not None:
                transition_counter["attempted_calls"] += 1
            obs, reward, done, extras, *_ = env.step(action)
            if transition_counter is not None:
                transition_counter["completed_calls"] += 1
            terminal_obs = extras["obs_before_reset"].cpu().numpy()
            position = extras["pelvis_position_before_reset"].cpu().numpy()
            reward_np = reward.cpu().numpy()
            if not all(np.isfinite(value).all() for value in (terminal_obs, position, reward_np)):
                raise FloatingPointError("nonfinite evaluation transition")
            ended = done.cpu().numpy().astype(bool)
            terminated = extras["terminated"].cpu().numpy().astype(bool)
            excitation = .5 * (action.cpu().numpy() + 1.)
            if capture:
                chunks["done"].append(ended[scored_worlds])
                chunks["terminated"].append(terminated[scored_worlds])
            for i in scored_worlds:
                row = episodes[i]
                error = float(terminal_obs[i, 5]) - env.task_contract.target_speed
                row["length"] += 1
                row["displacement_m"] = float(position[i, 0] - start_position[i, 0])
                row["forward_absolute_error_sum"] += abs(error)
                row["velocity_2d_squared_error_sum"] += error * error + float(terminal_obs[i, 6]) ** 2
                row["speed_band_steps"] += int(.8 <= terminal_obs[i, 5] <= 1.2)
                row["excitation_sum"] += float(excitation[i].mean())
                row["excitation_squared_sum"] += float(np.square(excitation[i]).mean())
                row["return"] += float(reward_np[i])
                if ended[i] or step + 1 == horizon:
                    row["survived"] = bool(not terminated[i] and step + 1 == horizon)
                    row["end_reason"] = "task_failure" if terminated[i] else "timeout" if ended[i] else "horizon"
                    row["failure_flags"] = {name: bool(flags[i]) for name, flags in extras["failure_flags"].items()}
                    alive[i] = False
            if not alive.any():
                break
            if deadline is not None and time.perf_counter() >= deadline:
                for i in np.flatnonzero(alive):
                    episodes[i]["end_reason"] = "wall_cap"
                break
        for row in episodes:
            row["mean_forward_velocity_mps"] = row["displacement_m"] / (row["length"] * env.control_dt)
            row["velocity_2d_rmse_mps"] = float(np.sqrt(row["velocity_2d_squared_error_sum"] / row["length"]))
            row["forward_absolute_speed_error_mps"] = row["forward_absolute_error_sum"] / row["length"]
            row["forward_speed_band_fraction"] = row["speed_band_steps"] / row["length"]
        evaluation = aggregate_episodes(episodes, horizon, env.control_dt, step + 1, env.num_envs, env.substeps)
        evaluation["complete"] = not any(row["end_reason"] == "wall_cap" for row in episodes)
        evaluation["policy_mode"] = "deterministic_mean" if deterministic else "stochastic_normal"
        traces = {key: np.concatenate(value) for key, value in chunks.items()} if capture else None
        return evaluation, traces


def contiguous_windows(traces, length=5):
    """Row indices for fully nonterminal consecutive actions in one first episode."""
    if length < 1:
        raise ValueError("window length must be positive")
    windows = []
    for world in np.unique(traces["world"]):
        indices = np.flatnonzero(traces["world"] == world)
        for start in range(len(indices) - length + 1):
            window = indices[start:start + length]
            if (not traces["done"][window].any()
                    and np.all(np.diff(traces["step"][window]) == 1)
                    and len(np.unique(traces["episode_seed"][window])) == 1):
                windows.append(window)
    return np.asarray(windows, dtype=np.int64).reshape(-1, length)
