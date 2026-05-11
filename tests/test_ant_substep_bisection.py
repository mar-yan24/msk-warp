"""Phase 2 Exp 2: Substep bisection — AD gradient stability across substep counts.

Hypothesis: if discretization noise (H2) is the dominant gradient-quality
problem, finer substeps (more mjw.step calls per env step at the fixed XML
dt) should make AD gradients more consistent. We measure pairwise cosine
between AD policy gradients at substeps in {8, 16, 32, 64} starting from
8 state-space snapshots collected via PPO-warmed random-action rollout.

Pass: mean pairwise-consecutive cosine for the finest pair (32 -> 64) > 0.9
and the pairwise cosine sequence trends upward as substeps refine.
Fail: stays below 0.5 -> discretization isn't the dominant noise source.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import warp as wp

from tests.test_ant_policy_gradient import (
    _DATA_DIR,
    _append_summary_json,
    _ant_shac,
    _compute_loss_and_flat_grad,
    _cosine,
    _flatten_mu_params,
    _initialize_for_rollout,
    _ppo_checkpoint_path,
    _restore_state,
    _set_mu_params_from_vector,
    _snapshot_state,
)


def _collect_warmup_snapshots(shac, n_snapshots: int = 8, snapshot_interval: int = 4):
    """Run PPO-loaded actor for n_snapshots*snapshot_interval env steps,
    capture (qpos, qvel) at each snapshot_interval.

    Uses the loaded actor (deterministic mode) to generate a realistic
    on-policy trajectory rather than random actions — gives more
    representative state samples for the substep bisection.
    """
    snapshots: list[tuple] = []
    n_steps = n_snapshots * snapshot_interval

    _initialize_for_rollout(shac)

    with torch.no_grad():
        for step in range(n_steps):
            obs = shac.env.obs_buf
            if shac.obs_rms is not None:
                obs_normalized = shac.obs_rms.normalize(obs)
            else:
                obs_normalized = obs
            actions = shac.actor.forward(obs_normalized, deterministic=True)
            shac.env.step(actions)

            if step % snapshot_interval == (snapshot_interval - 1):
                snap = (
                    wp.clone(shac.env.warp_data.qpos),
                    wp.clone(shac.env.warp_data.qvel),
                    wp.clone(shac.env.warp_data.time),
                    wp.clone(shac.env.warp_data.act) if shac.env.warp_data.act.shape[1] > 0 else None,
                )
                snapshots.append(snap)

    return snapshots


def _restore_warp_to_snapshot(shac, snap):
    qpos_arr, qvel_arr, time_arr, act_arr = snap
    wp.copy(shac.env.warp_data.qpos, qpos_arr)
    wp.copy(shac.env.warp_data.qvel, qvel_arr)
    wp.copy(shac.env.warp_data.time, time_arr)
    if act_arr is not None and shac.env.warp_data.act.shape[1] > 0:
        wp.copy(shac.env.warp_data.act, act_arr)
    wp.synchronize()


def test_phase2_exp2_substep_bisection():
    """Measure pairwise AD-gradient cosine across substep refinements."""
    n_snapshots = 8
    substeps_list = [8, 16, 32, 64]
    H = 1  # short horizon — focus on per-step gradient stability vs substeps
    n_envs = 8

    ckpt = _ppo_checkpoint_path()

    # ---- Step 1: warmup + collect snapshots ----
    print(f"\n[Exp 2] Collecting {n_snapshots} state snapshots via PPO-warmed rollout...")
    warmup_shac = _ant_shac(
        init_policy=ckpt,
        steps_num=H,
        num_envs=n_envs,
        logdir="logs/_phase2_exp2_warmup",
        extra_env={"substeps": 8},
    )
    snapshots = _collect_warmup_snapshots(
        warmup_shac, n_snapshots=n_snapshots, snapshot_interval=4
    )
    print(f"[Exp 2] Collected {len(snapshots)} snapshots.")

    # Free warmup shac to release memory
    del warmup_shac
    torch.cuda.empty_cache()

    # ---- Step 2: compute AD gradient at each (snapshot, substeps) ----
    # Dict: snap_idx -> {substeps -> flattened grad vector (np.ndarray)}
    grads: dict[int, dict[int, np.ndarray]] = {i: {} for i in range(len(snapshots))}
    g_norms: dict[int, dict[int, float]] = {i: {} for i in range(len(snapshots))}

    for substeps in substeps_list:
        print(f"\n[Exp 2] substeps={substeps}...")
        shac = _ant_shac(
            init_policy=ckpt,
            steps_num=H,
            num_envs=n_envs,
            logdir=f"logs/_phase2_exp2_substeps{substeps}",
            extra_env={"substeps": substeps},
        )
        shac.state_grad_decay = 0.0
        shac.state_grad_clip = 0.0

        for snap_idx, snap in enumerate(snapshots):
            _initialize_for_rollout(shac)
            _restore_warp_to_snapshot(shac, snap)

            # Take fresh state snapshot for determinism freeze
            state_snap = _snapshot_state(shac)
            _restore_state(shac, state_snap)
            _restore_warp_to_snapshot(shac, snap)

            theta0 = _flatten_mu_params(shac.actor)
            _set_mu_params_from_vector(shac.actor, theta0)

            _, g = _compute_loss_and_flat_grad(shac)
            g_np = g.cpu().numpy()
            grads[snap_idx][substeps] = g_np
            g_norms[snap_idx][substeps] = float(np.linalg.norm(g_np))

            print(f"  snap {snap_idx}: ||g||={g_norms[snap_idx][substeps]:.3e}")

        del shac
        torch.cuda.empty_cache()

    # ---- Step 3: pairwise-consecutive cosines ----
    pairs = list(zip(substeps_list[:-1], substeps_list[1:]))
    pairwise_cosines: dict[tuple, list[float]] = {p: [] for p in pairs}

    for snap_idx in range(len(snapshots)):
        for s1, s2 in pairs:
            g1 = grads[snap_idx][s1]
            g2 = grads[snap_idx][s2]
            pairwise_cosines[(s1, s2)].append(_cosine(g1, g2))

    print(f"\n[Exp 2] Pairwise-consecutive cosines across {n_snapshots} snapshots:")
    summary_per_pair: dict[str, dict] = {}
    means_in_order: list[float] = []
    for pair, cos_list in pairwise_cosines.items():
        mean = float(np.mean(cos_list))
        std = float(np.std(cos_list))
        means_in_order.append(mean)
        print(f"  cos({pair[0]:3d} -> {pair[1]:3d}): "
              f"{mean:+.4f} +/- {std:.4f}  (n={len(cos_list)})")
        summary_per_pair[f"cos_{pair[0]}_to_{pair[1]}"] = {
            "mean": mean,
            "std": std,
            "values": [float(c) for c in cos_list],
        }

    # ---- Pass criteria ----
    final_pair_mean = means_in_order[-1]
    monotonic_up = all(
        means_in_order[i + 1] >= means_in_order[i]
        for i in range(len(means_in_order) - 1)
    )

    if final_pair_mean > 0.9:
        verdict = (
            f"PASS — cos(32->64)={final_pair_mean:.4f} > 0.9; "
            f"discretization noise vanishes under refinement"
        )
    elif final_pair_mean > 0.5:
        verdict = (
            f"PARTIAL — cos(32->64)={final_pair_mean:.4f}; some convergence"
        )
    else:
        verdict = (
            f"FAIL — cos(32->64)={final_pair_mean:.4f} < 0.5; "
            f"discretization isn't dominant"
        )
    print(f"\n[Exp 2 verdict] {verdict}")
    print(f"[Exp 2] pairwise-cosine means increase monotonically: {monotonic_up}")

    # ---- Save summary ----
    entry = {
        "name": "exp2_substep_bisection",
        "config": (
            f"Phase 2 Exp 2: 8 snapshots x substeps in {substeps_list}, "
            f"H={H}, decay=0, PPO init"
        ),
        "n_snapshots": n_snapshots,
        "substeps_list": substeps_list,
        "H": H,
        "pairwise_cosines": summary_per_pair,
        "final_pair_mean": final_pair_mean,
        "monotonic_up": bool(monotonic_up),
        "g_norms_per_snapshot": {
            i: {str(s): g_norms[i][s] for s in substeps_list}
            for i in range(len(snapshots))
        },
    }
    _append_summary_json(
        entry,
        _DATA_DIR / "policy_gradient_diagnostic_summary.json",
    )

    assert np.isfinite(final_pair_mean)


if __name__ == "__main__":
    test_phase2_exp2_substep_bisection()
