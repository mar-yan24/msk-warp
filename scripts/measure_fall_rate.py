"""Measure how often an environment's model falls over under untrained actions.

The ant's failure in March-May 2026 was not a gradient defect. MuJoCo constraint contacts gave it
a statically stable standing equilibrium worth 99.5% of the reward, so under random actions it
never fell and SHAC's deterministic gradient never saw a fall to learn from. That property is
cheap to measure and expensive to discover after a training run, so it is measured first, against
a prediction written down beforehand.

A world counts as fallen at the first control step where ``obs[0]``, the height the environment
terminates on, drops below ``env.termination_height``. Early termination is switched off so that
each world is scored on physics rather than on the reset rule, and worlds keep running after they
fall so the final state is observable.

Action distributions:

  passive   ``ctrl = 0`` exactly: zero torque, or zero muscle activation. The gravity drop.
  zero      policy action ``a = 0``. Same as passive for a motor model; 50% activation on every
            muscle for a muscle model, which is what an untrained stochastic actor emits.
  sigma=s   ``a ~ N(0, s)`` resampled each control step and clipped to [-1, 1].

The reward split reports what fraction of the accumulated return comes from the posture terms
(height and angle) rather than from forward progress. A posture fraction near 1 is the ant's
pathology: standing still collects almost the whole return.

Usage::

    .venv/Scripts/python.exe scripts/measure_fall_rate.py --env HopperMotor
    .venv/Scripts/python.exe scripts/measure_fall_rate.py --env HopperMuscle --out results.json
"""

import argparse
import json
import time

import numpy as np
import torch


def _fall_stats(fell_at, steps, horizon):
    """Fall rate within ``horizon`` steps and the median step at which falls happen."""
    within = fell_at[(fell_at >= 0) & (fell_at < horizon)]
    rate = float(within.size) / float(fell_at.size)
    median = float(np.median(within)) if within.size else float("nan")
    return rate, median


def run_condition(env, kind, sigma, steps, seed):
    """Roll out every world for ``steps`` control steps and score it."""
    torch.manual_seed(seed)
    env.reset()
    obs = env.obs_buf

    n = env.num_envs
    device = obs.device
    fell_at = np.full(n, -1, dtype=np.int64)
    posture_sum = torch.zeros(n, device=device)
    progress_sum = torch.zeros(n, device=device)
    return_sum = torch.zeros(n, device=device)
    height_trace = np.zeros((steps, n), dtype=np.float32)

    passive = env.passive_action() if kind == "passive" else None

    for t in range(steps):
        if kind == "passive":
            actions = passive
        elif kind == "zero":
            actions = torch.zeros(n, env.num_acts, device=device)
        else:
            actions = torch.clamp(torch.randn(n, env.num_acts, device=device) * sigma, -1.0, 1.0)

        obs, rew, done, _extras, _qp, _qv, _a = env.step(actions)

        with torch.no_grad():
            height = obs[:, 0]
            height_trace[t] = height.detach().cpu().numpy()
            newly = (height < env.termination_height).detach().cpu().numpy() & (fell_at < 0)
            fell_at[newly] = t

            # Reward split, using the environment's own reward terms.
            hd = height - (env.termination_height + env.termination_height_tolerance)
            hr = torch.clip(hd, -1.0, 0.3)
            hr = torch.where(hr < 0.0, -200.0 * hr * hr, hr)
            hr = torch.where(hr > 0.0, env.height_rew_scale * hr, hr)
            ar = 1.0 * (-obs[:, 1] ** 2 / (env.termination_angle ** 2) + 1.0)
            posture_sum += hr + ar
            progress_sum += obs[:, 5]
            return_sum += rew

    rate300, median300 = _fall_stats(fell_at, steps, min(300, steps))
    rate_all, _ = _fall_stats(fell_at, steps, steps)
    posture = float(posture_sum.abs().sum())
    progress = float(progress_sum.abs().sum())

    return {
        "condition": kind if kind in ("passive", "zero") else f"sigma={sigma}",
        "fall_rate_300": rate300,
        "fall_rate_full": rate_all,
        "median_fall_step": median300,
        "final_height_mean": float(height_trace[-1].mean()),
        "final_height_std": float(height_trace[-1].std()),
        "min_height_mean": float(height_trace.min(axis=0).mean()),
        "mean_return": float(return_sum.mean()),
        "posture_fraction": posture / (posture + progress) if posture + progress > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HopperMotor", help="key in msk_warp.envs.ENV_MAP")
    ap.add_argument("--model-path", default=None, help="override the environment's default asset")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.1, 0.3, 0.6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="write the results as JSON to this path")
    args = ap.parse_args()

    from msk_warp.envs import ENV_MAP

    kwargs = dict(
        num_envs=args.num_envs,
        device=args.device,
        no_grad=True,
        substeps=args.substeps,
        stochastic_init=True,
        early_termination=False,      # score on physics, not on the reset rule
        episode_length=args.steps + 1,
    )
    if args.model_path:
        kwargs["model_path"] = args.model_path
    env = ENV_MAP[args.env](**kwargs)

    conditions = [("passive", 0.0), ("zero", 0.0)] + [("gauss", s) for s in args.sigmas]
    rows = []
    started = time.time()
    for kind, sigma in conditions:
        row = run_condition(env, kind, sigma, args.steps, args.seed)
        rows.append(row)
        print(
            f"{row['condition']:>12s}  fall@300 {row['fall_rate_300']:6.1%}  "
            f"fall@{args.steps} {row['fall_rate_full']:6.1%}  "
            f"median step {row['median_fall_step']:6.1f}  "
            f"final h {row['final_height_mean']:+7.3f}  "
            f"return {row['mean_return']:9.1f}  posture {row['posture_fraction']:5.1%}"
        )

    result = {
        "env": args.env,
        "model_path": args.model_path or getattr(env, "model_path", None),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "substeps": args.substeps,
        "seed": args.seed,
        "termination_height": float(env.termination_height),
        "elapsed_s": round(time.time() - started, 1),
        "conditions": rows,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
