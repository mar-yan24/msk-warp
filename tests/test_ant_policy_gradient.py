"""Phase 1 diagnostic: FD vs AD policy-gradient cosine on Ant.

Localizes SHAC's failure mode by comparing the autodiff gradient g_AD against
central-difference estimates (L(theta+eps*d) - L(theta-eps*d))/(2*eps) along
N random unit directions d. The determinism freeze ensures three rollouts
(AD, +eps, -eps) are bit-identical except for theta.

Decision tree (per plan):
  cos >= 0.9 on ant from PPO init  -> H3 reward landscape
  0 < cos < 0.9                    -> H2 discretization / BPTT
  cos <= 0                         -> H1 autodiff bug

This file currently provides:
  - Determinism-freeze helpers (snapshot/restore)
  - Hard-precondition test: two frozen rollouts at same theta give |L1-L2| < 1e-5
"""

from __future__ import annotations

import copy
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

from msk_warp.algorithms.shac import SHAC


# ---------- Config / construction helpers ----------


def _shac_from_cfg(
    cfg_path: str,
    *,
    logdir: str,
    seed: int = 42,
    num_envs: int | None = None,
    episode_length: int | None = None,
    steps_num: int | None = None,
    stochastic_init: bool = False,
    early_termination: bool = False,
    model_path: str | None = None,
    extra_env: dict | None = None,
    extra_config: dict | None = None,
) -> SHAC:
    """Instantiate SHAC from a YAML, applying determinism-freeze overrides."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["params"]["general"]["logdir"] = logdir
    cfg["params"]["general"]["seed"] = seed
    if num_envs is not None:
        cfg["params"]["env"]["num_actors"] = num_envs
    if episode_length is not None:
        cfg["params"]["env"]["episode_length"] = episode_length
    if steps_num is not None:
        cfg["params"]["config"]["steps_num"] = steps_num
    if model_path is not None:
        cfg["params"]["env"]["model_path"] = model_path
    cfg["params"]["env"]["stochastic_init"] = stochastic_init
    cfg["params"]["env"]["early_termination"] = early_termination
    if extra_env:
        cfg["params"]["env"].update(extra_env)
    if extra_config:
        cfg["params"]["config"].update(extra_config)
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["save_interval"] = 0
    return SHAC(cfg)


# ---------- Parameter / gradient flatteners ----------


def _flatten_mu_params(actor) -> torch.Tensor:
    """Flatten only mu_net params. logstd is excluded — its gradient is zero
    under deterministic rollouts, so it inflates flat-param dimensionality
    without contributing to the cosine measurement."""
    return torch.nn.utils.parameters_to_vector(
        actor.mu_net.parameters()
    ).detach().clone()


def _set_mu_params_from_vector(actor, theta_vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(theta_vec, actor.mu_net.parameters())


def _flatten_mu_grads(actor) -> torch.Tensor:
    return torch.cat(
        [p.grad.flatten() for p in actor.mu_net.parameters()]
    ).detach().clone()


# ---------- State snapshot / restore ----------


def _snapshot_state(shac: SHAC) -> dict:
    """Capture all non-theta state needed to make a rollout reproducible.

    Sources of nondeterminism (per plan Phase 1 table):
      - warp_data (qpos/qvel/time/act) mutated by env.step + backward
      - torch + CUDA RNG (used by stochastic_init / dist.rsample if enabled)
      - obs_rms / ret_rms running statistics (mutated by .update())
      - episode meters and history lists (mutated when episodes end)
      - env progress / reset / termination buffers
    """
    snap: dict = {}

    # warp_data — clone arrays (mirrors actor_closure pattern shac.py:570-592)
    snap["qpos"] = wp.clone(shac.env.warp_data.qpos)
    snap["qvel"] = wp.clone(shac.env.warp_data.qvel)
    snap["time"] = wp.clone(shac.env.warp_data.time)
    snap["act"] = (
        wp.clone(shac.env.warp_data.act)
        if shac.env.warp_data.act.shape[1] > 0
        else None
    )

    # PyTorch + CUDA RNG
    snap["torch_rng"] = torch.get_rng_state()
    snap["cuda_rng"] = torch.cuda.get_rng_state_all()

    # obs_rms / ret_rms
    if shac.obs_rms is not None:
        snap["obs_rms_mean"] = shac.obs_rms.mean.clone()
        snap["obs_rms_var"] = shac.obs_rms.var.clone()
        snap["obs_rms_count"] = shac.obs_rms.count
    if shac.ret_rms is not None:
        snap["ret_rms_mean"] = shac.ret_rms.mean.clone()
        snap["ret_rms_var"] = shac.ret_rms.var.clone()
        snap["ret_rms_count"] = shac.ret_rms.count

    # Env progress
    snap["progress_buf"] = shac.env.progress_buf.clone()
    snap["reset_buf"] = shac.env.reset_buf.clone()
    snap["termination_buf"] = shac.env.termination_buf.clone()
    snap["env_actions"] = shac.env.actions.clone()

    # SHAC episode-tracking state
    snap["ret"] = shac.ret.clone()
    snap["episode_loss"] = shac.episode_loss.clone()
    snap["episode_discounted_loss"] = shac.episode_discounted_loss.clone()
    snap["episode_gamma"] = shac.episode_gamma.clone()
    snap["episode_length"] = shac.episode_length.clone()
    snap["episode_loss_meter"] = copy.deepcopy(shac.episode_loss_meter)
    snap["episode_discounted_loss_meter"] = copy.deepcopy(
        shac.episode_discounted_loss_meter
    )
    snap["episode_length_meter"] = copy.deepcopy(shac.episode_length_meter)
    snap["episode_loss_his"] = list(shac.episode_loss_his)
    snap["episode_discounted_loss_his"] = list(shac.episode_discounted_loss_his)
    snap["episode_length_his"] = list(shac.episode_length_his)
    snap["step_count"] = shac.step_count

    return snap


def _restore_state(shac: SHAC, snap: dict) -> None:
    wp.copy(shac.env.warp_data.qpos, snap["qpos"])
    wp.copy(shac.env.warp_data.qvel, snap["qvel"])
    wp.copy(shac.env.warp_data.time, snap["time"])
    if snap["act"] is not None:
        wp.copy(shac.env.warp_data.act, snap["act"])
    wp.synchronize()

    torch.set_rng_state(snap["torch_rng"])
    torch.cuda.set_rng_state_all(snap["cuda_rng"])

    if shac.obs_rms is not None:
        shac.obs_rms.mean = snap["obs_rms_mean"].clone()
        shac.obs_rms.var = snap["obs_rms_var"].clone()
        shac.obs_rms.count = snap["obs_rms_count"]
    if shac.ret_rms is not None:
        shac.ret_rms.mean = snap["ret_rms_mean"].clone()
        shac.ret_rms.var = snap["ret_rms_var"].clone()
        shac.ret_rms.count = snap["ret_rms_count"]

    shac.env.progress_buf.copy_(snap["progress_buf"])
    shac.env.reset_buf.copy_(snap["reset_buf"])
    shac.env.termination_buf.copy_(snap["termination_buf"])
    shac.env.actions.copy_(snap["env_actions"])

    shac.ret.copy_(snap["ret"])
    shac.episode_loss.copy_(snap["episode_loss"])
    shac.episode_discounted_loss.copy_(snap["episode_discounted_loss"])
    shac.episode_gamma.copy_(snap["episode_gamma"])
    shac.episode_length.copy_(snap["episode_length"])
    shac.episode_loss_meter = copy.deepcopy(snap["episode_loss_meter"])
    shac.episode_discounted_loss_meter = copy.deepcopy(
        snap["episode_discounted_loss_meter"]
    )
    shac.episode_length_meter = copy.deepcopy(snap["episode_length_meter"])
    shac.episode_loss_his = list(snap["episode_loss_his"])
    shac.episode_discounted_loss_his = list(snap["episode_discounted_loss_his"])
    shac.episode_length_his = list(snap["episode_length_his"])
    shac.step_count = snap["step_count"]


# ---------- Frozen rollout primitives ----------


def _initialize_for_rollout(shac: SHAC) -> None:
    """One-time setup before snapshotting. Fixes curriculum/reward weights and
    zeroes per-env episode trackers."""
    shac.env.begin_epoch(epoch=0, max_epochs=1)
    shac.initialize_env()
    shac.episode_loss.zero_()
    shac.episode_discounted_loss.zero_()
    shac.episode_length.zero_()
    shac.episode_gamma.fill_(1.0)


def _compute_loss_only(shac: SHAC) -> torch.Tensor:
    """Forward pass, no backward. Returns scalar loss tensor (detached)."""
    with torch.no_grad():
        loss = shac.compute_actor_loss(deterministic=True)
    return loss.detach().clone()


def _compute_loss_and_flat_grad(
    shac: SHAC,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward + backward. Returns (loss, flat_grad_over_mu_net)."""
    shac.actor_optimizer.zero_grad()
    loss = shac.compute_actor_loss(deterministic=True)

    # Mirror actor_closure (shac.py:570-592): warp_data backward is destructive,
    # so snapshot+restore the post-rollout warp_data state across .backward().
    with torch.no_grad():
        _saved_qpos = wp.clone(shac.env.warp_data.qpos)
        _saved_qvel = wp.clone(shac.env.warp_data.qvel)
        _saved_time = wp.clone(shac.env.warp_data.time)
        _saved_act = (
            wp.clone(shac.env.warp_data.act)
            if shac.env.warp_data.act.shape[1] > 0
            else None
        )
    loss.backward()
    wp.copy(shac.env.warp_data.qpos, _saved_qpos)
    wp.copy(shac.env.warp_data.qvel, _saved_qvel)
    wp.copy(shac.env.warp_data.time, _saved_time)
    if _saved_act is not None:
        wp.copy(shac.env.warp_data.act, _saved_act)
    wp.synchronize()

    return loss.detach().clone(), _flatten_mu_grads(shac.actor)


# ---------- Hard-precondition: determinism check ----------


def _run_determinism_check(
    cfg_path: str,
    *,
    logdir: str,
    num_envs: int,
    episode_length: int,
    steps_num: int,
    tol: float = 1e-5,
) -> tuple[float, float, float]:
    """Run two frozen rollouts at the same theta; assert |L_1 - L_2| < tol."""
    shac = _shac_from_cfg(
        cfg_path,
        logdir=logdir,
        seed=42,
        num_envs=num_envs,
        episode_length=episode_length,
        steps_num=steps_num,
    )
    _initialize_for_rollout(shac)
    snap = _snapshot_state(shac)

    L1 = _compute_loss_only(shac).item()
    _restore_state(shac, snap)
    L2 = _compute_loss_only(shac).item()

    diff = abs(L1 - L2)
    print(
        f"\n[determinism-check {Path(cfg_path).stem}] "
        f"L1={L1:.10f} L2={L2:.10f} |L1-L2|={diff:.3e} (tol={tol:.0e})"
    )
    return L1, L2, diff


def test_determinism_freeze_ant_from_ppo():
    """Two consecutive rollouts at same theta must agree to <1e-5.

    Uses ant_shac_from_ppo settings (ant_soft.xml, H=8 for speed) with the
    determinism freeze active. If this fails, the FD/AD comparison cannot
    be trusted — abort downstream measurements until the fixture is fixed.
    """
    cfg_path = "msk_warp/configs/experiments/ant_shac_from_ppo.yaml"
    L1, L2, diff = _run_determinism_check(
        cfg_path,
        logdir="logs/_phase1_determinism_check",
        num_envs=4,
        episode_length=64,
        steps_num=8,
    )
    assert diff < 1e-5, (
        f"Determinism freeze broken on ant_shac_from_ppo: "
        f"|L1-L2|={diff:.3e}, L1={L1:.6f}, L2={L2:.6f}"
    )


# ---------- FD-vs-AD core measurement ----------


@dataclass
class FdAdResult:
    config: str
    n_directions: int
    eps: float
    n_params: int
    theta_norm: float
    g_ad_norm: float
    ad_proj: np.ndarray  # (N,) ad_proj[i] = g_AD . d_i
    fd_proj: np.ndarray  # (N,) fd_proj[i] = (L(+eps*d_i) - L(-eps*d_i))/(2eps)
    losses_center: np.ndarray  # (N,) L0 at each step (sanity, should be constant)
    losses_plus: np.ndarray
    losses_minus: np.ndarray
    cosine: float
    bootstrap_ci_low: float = float("nan")
    bootstrap_ci_high: float = float("nan")
    permutation_p_positive: float = float("nan")
    extras: dict = field(default_factory=dict)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _bootstrap_cosine_ci(
    ad: np.ndarray, fd: np.ndarray, *, n_resamples: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    rng = np.random.default_rng(seed=0)
    n = len(ad)
    samples = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[k] = _cosine(ad[idx], fd[idx])
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1.0 - alpha / 2))
    return lo, hi


def _permutation_p_positive(
    ad: np.ndarray, fd: np.ndarray, *, n_shuffles: int = 1000
) -> float:
    """One-sided p-value for cosine > 0 under random fd-shuffling (null: no
    relationship between AD and FD projections)."""
    rng = np.random.default_rng(seed=0)
    observed = _cosine(ad, fd)
    n_geq = 0
    fd_perm = fd.copy()
    for _ in range(n_shuffles):
        rng.shuffle(fd_perm)
        if _cosine(ad, fd_perm) >= observed:
            n_geq += 1
    return (n_geq + 1) / (n_shuffles + 1)


def _compute_fd_ad_projections(
    shac: SHAC,
    *,
    n_directions: int,
    eps: float,
    direction_seed: int = 0,
    verbose: bool = False,
) -> FdAdResult:
    """Run the full FD-vs-AD sweep at the current actor.

    Strategy:
      1. Snapshot fixture state (warp_data + RNG + obs_rms + meters).
      2. Compute AD gradient ONCE at theta_0 (after restore). All directions
         project against the same g_AD.
      3. For each random unit direction d_i:
         - restore state, set theta = theta_0 + eps*d_i, compute L_plus
         - restore state, set theta = theta_0 - eps*d_i, compute L_minus
         - fd_proj[i] = (L_plus - L_minus) / (2*eps)
         - record L_0 at restored theta_0 too (sanity — should be constant)
      4. Restore theta = theta_0 at exit.
    """
    actor = shac.actor
    theta0 = _flatten_mu_params(actor)
    n_params = theta0.numel()
    device = theta0.device

    _initialize_for_rollout(shac)
    snap = _snapshot_state(shac)

    # AD gradient at theta_0 (computed once, reused for all directions)
    _restore_state(shac, snap)
    _set_mu_params_from_vector(actor, theta0)
    L_ad_center, g_ad = _compute_loss_and_flat_grad(shac)
    g_ad_np = g_ad.cpu().numpy()
    g_ad_norm = float(np.linalg.norm(g_ad_np))
    if verbose:
        print(
            f"  AD gradient computed: ||g_AD|| = {g_ad_norm:.4e}, "
            f"L_center = {L_ad_center.item():.6f}"
        )

    # Random unit directions (deterministic per direction_seed)
    gen = torch.Generator(device=device).manual_seed(direction_seed)

    ad_proj = np.empty(n_directions, dtype=np.float64)
    fd_proj = np.empty(n_directions, dtype=np.float64)
    losses_center = np.full(n_directions, L_ad_center.item(), dtype=np.float64)
    losses_plus = np.empty(n_directions, dtype=np.float64)
    losses_minus = np.empty(n_directions, dtype=np.float64)

    for i in range(n_directions):
        d = torch.randn(n_params, generator=gen, device=device, dtype=theta0.dtype)
        d = d / d.norm().clamp(min=1e-12)

        ad_proj[i] = float((g_ad * d).sum().item())

        # L_plus
        _restore_state(shac, snap)
        _set_mu_params_from_vector(actor, theta0 + eps * d)
        L_plus = _compute_loss_only(shac).item()

        # L_minus
        _restore_state(shac, snap)
        _set_mu_params_from_vector(actor, theta0 - eps * d)
        L_minus = _compute_loss_only(shac).item()

        fd_proj[i] = (L_plus - L_minus) / (2.0 * eps)
        losses_plus[i] = L_plus
        losses_minus[i] = L_minus

        if verbose:
            print(
                f"  dir {i:3d}: AD·d={ad_proj[i]:+.4e}  "
                f"FD={fd_proj[i]:+.4e}  L+={L_plus:+.6f}  L-={L_minus:+.6f}"
            )

    # Restore theta to the start state
    _restore_state(shac, snap)
    _set_mu_params_from_vector(actor, theta0)

    cos = _cosine(ad_proj, fd_proj)
    ci_lo, ci_hi = _bootstrap_cosine_ci(ad_proj, fd_proj)
    p_pos = _permutation_p_positive(ad_proj, fd_proj)

    return FdAdResult(
        config="<override-in-caller>",
        n_directions=n_directions,
        eps=eps,
        n_params=n_params,
        theta_norm=float(theta0.norm().item()),
        g_ad_norm=g_ad_norm,
        ad_proj=ad_proj,
        fd_proj=fd_proj,
        losses_center=losses_center,
        losses_plus=losses_plus,
        losses_minus=losses_minus,
        cosine=cos,
        bootstrap_ci_low=ci_lo,
        bootstrap_ci_high=ci_hi,
        permutation_p_positive=p_pos,
        extras={"L_center": L_ad_center.item()},
    )


# ---------- Persistence ----------


_DATA_DIR = Path(__file__).resolve().parent / "data"


def _save_pickle(result: FdAdResult, name: str) -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _DATA_DIR / f"policy_gradient_diagnostic_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return path


def _summarize_for_json(result: FdAdResult, name: str) -> dict:
    return {
        "name": name,
        "config": result.config,
        "n_directions": result.n_directions,
        "eps": result.eps,
        "n_params": result.n_params,
        "theta_norm": result.theta_norm,
        "g_ad_norm": result.g_ad_norm,
        "cosine": result.cosine,
        "cosine_ci_95": [result.bootstrap_ci_low, result.bootstrap_ci_high],
        "permutation_p_cos_positive": result.permutation_p_positive,
        "ad_proj_mean": float(np.mean(result.ad_proj)),
        "ad_proj_std": float(np.std(result.ad_proj)),
        "fd_proj_mean": float(np.mean(result.fd_proj)),
        "fd_proj_std": float(np.std(result.fd_proj)),
        "losses_center_mean": float(np.mean(result.losses_center)),
        "losses_center_std": float(np.std(result.losses_center)),
    }


def _append_summary_json(entry: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path) as f:
            current = json.load(f)
    else:
        current = []
    # Replace prior entry with same name
    current = [e for e in current if e.get("name") != entry["name"]]
    current.append(entry)
    with open(path, "w") as f:
        json.dump(current, f, indent=2)


# ---------- Sanity test: CartPole ----------


def test_cartpole_sanity_random_init():
    """Plan tripwire: cos >= 0.99 expected on CartPole (uses FD-Jacobian
    backward, so 'AD' is FD-Jacobian — comparing it against central FD
    should give near-perfect agreement). If this fails, the test fixture
    itself is broken; do not interpret ant results."""
    shac = _shac_from_cfg(
        "msk_warp/configs/cartpole_shac.yaml",
        logdir="logs/_phase1_cartpole_sanity",
        seed=42,
        num_envs=8,
        episode_length=64,
        steps_num=16,
    )
    result = _compute_fd_ad_projections(
        shac, n_directions=20, eps=1e-3, direction_seed=0, verbose=False
    )
    result.config = "cartpole_shac"
    print(
        f"\n[cartpole sanity] cos={result.cosine:+.4f}  "
        f"CI95=[{result.bootstrap_ci_low:+.4f}, {result.bootstrap_ci_high:+.4f}]  "
        f"p(cos>0)={result.permutation_p_positive:.4f}"
    )
    print(
        f"  ||theta||={result.theta_norm:.4e}  ||g_AD||={result.g_ad_norm:.4e}  "
        f"L_center={result.extras['L_center']:.6f}"
    )

    _save_pickle(result, "cartpole_sanity")
    _append_summary_json(
        _summarize_for_json(result, "cartpole_sanity"),
        _DATA_DIR / "policy_gradient_diagnostic_summary.json",
    )

    assert result.cosine >= 0.99, (
        f"CartPole sanity tripwire failed: cos={result.cosine:.4f} < 0.99 — "
        "fixture is suspect, do not run ant measurements."
    )


# ---------- Phase 2 Gate 0.2: RK4-with-tape cartpole smoke test ----------


def _set_integrator(shac: SHAC, integrator_name: str) -> None:
    """Override the warp_model integrator post-construction.

    Available: 'euler' (default), 'rk4'. The mjw.put_model() warmup
    happens at SHAC env init under whatever integrator the XML defines —
    that's fine, it's just kinematics init. Subsequent rollouts use the
    swapped value.
    """
    from mujoco_warp._src.types import IntegratorType

    name = integrator_name.lower()
    if name == "euler":
        shac.env.warp_model.opt.integrator = IntegratorType.EULER
    elif name == "rk4":
        shac.env.warp_model.opt.integrator = IntegratorType.RK4
    else:
        raise ValueError(f"unknown integrator: {integrator_name}")


def test_gate02_cartpole_rk4_tape_smoke():
    """Phase 2 Gate 0.2: validates rungekutta4()'s four-stage autodiff chain
    on cartpole BEFORE any ant RK4 work.

    Cartpole is smooth (no stiff contacts), so tape-based AD with either
    integrator should agree with central FD at cos >= 0.95. If RK4 mode
    fails this bar, the four-stage intermediate routing inside
    rungekutta4() is broken — H2(b) RK4 path requires debugging in
    mujoco_warp before any ant work.

    Two checks run in sequence:
      1. Liveness: RK4 g_AD norm is finite, non-zero, non-NaN
      2. Correctness: cartpole tripwire cos >= 0.95 for both EULER and RK4
         (EULER serves as the baseline sanity that the tape AD path works
         at all on cartpole; RK4 is the load-bearing check)
    """
    # Cartpole's default config uses use_fd_jacobian=True, tape_per_substep=False
    # (FD-Jacobian backward — bypasses the actual integrator autodiff chain).
    # Gate 0.2 needs tape-based AD to exercise rungekutta4's four-stage chain.
    tape_ad_overrides = {"use_fd_jacobian": False, "tape_per_substep": True}

    # ------ EULER baseline (tape AD on cartpole) ------
    shac_euler = _shac_from_cfg(
        "msk_warp/configs/cartpole_shac.yaml",
        logdir="logs/_gate02_cartpole_euler",
        seed=42,
        num_envs=8,
        episode_length=64,
        steps_num=16,
        extra_env=tape_ad_overrides,
    )
    _set_integrator(shac_euler, "euler")
    result_euler = _compute_fd_ad_projections(
        shac_euler, n_directions=20, eps=1e-3, direction_seed=0, verbose=False
    )
    result_euler.config = "cartpole_shac (EULER, tape AD)"
    _save_pickle(result_euler, "gate02_cartpole_euler")
    _append_summary_json(
        _summarize_for_json(result_euler, "gate02_cartpole_euler"),
        _DATA_DIR / "policy_gradient_diagnostic_summary.json",
    )
    print(
        f"\n[Gate 0.2 euler] cos={result_euler.cosine:+.4f}  "
        f"CI95=[{result_euler.bootstrap_ci_low:+.4f}, "
        f"{result_euler.bootstrap_ci_high:+.4f}]"
    )
    print(
        f"  ||g_AD||={result_euler.g_ad_norm:.4e}  "
        f"L_center={result_euler.extras['L_center']:.6f}"
    )

    # ------ RK4 (tape AD on cartpole) ------
    shac_rk4 = _shac_from_cfg(
        "msk_warp/configs/cartpole_shac.yaml",
        logdir="logs/_gate02_cartpole_rk4",
        seed=42,
        num_envs=8,
        episode_length=64,
        steps_num=16,
        extra_env=tape_ad_overrides,
    )
    _set_integrator(shac_rk4, "rk4")
    result_rk4 = _compute_fd_ad_projections(
        shac_rk4, n_directions=20, eps=1e-3, direction_seed=0, verbose=False
    )
    result_rk4.config = "cartpole_shac (RK4, tape AD)"
    _save_pickle(result_rk4, "gate02_cartpole_rk4")
    _append_summary_json(
        _summarize_for_json(result_rk4, "gate02_cartpole_rk4"),
        _DATA_DIR / "policy_gradient_diagnostic_summary.json",
    )
    print(
        f"\n[Gate 0.2 rk4  ] cos={result_rk4.cosine:+.4f}  "
        f"CI95=[{result_rk4.bootstrap_ci_low:+.4f}, "
        f"{result_rk4.bootstrap_ci_high:+.4f}]"
    )
    print(
        f"  ||g_AD||={result_rk4.g_ad_norm:.4e}  "
        f"L_center={result_rk4.extras['L_center']:.6f}"
    )

    # Liveness — RK4 must produce a real gradient
    assert np.isfinite(result_rk4.g_ad_norm), (
        f"RK4 tape AD produced non-finite g_AD norm: {result_rk4.g_ad_norm}"
    )
    assert result_rk4.g_ad_norm > 0, (
        f"RK4 tape AD produced zero g_AD: ||g_AD||={result_rk4.g_ad_norm}"
    )

    # Sanity — EULER baseline must pass tripwire (else fixture broken)
    assert result_euler.cosine >= 0.95, (
        f"Gate 0.2 EULER baseline failed: cos={result_euler.cosine:.4f} < 0.95. "
        "Tape-based AD on cartpole-EULER is broken; debug fixture before "
        "interpreting RK4 result."
    )

    # Correctness — the load-bearing check for H2(b) viability
    assert result_rk4.cosine >= 0.95, (
        f"RK4 tape AD failed cartpole tripwire: cos={result_rk4.cosine:.4f} < 0.95. "
        f"Four-stage autodiff chain in rungekutta4() likely broken; "
        f"H2(b) RK4 path requires debugging in mujoco_warp before any ant work."
    )

    print(
        f"\n[Gate 0.2 OK] euler cos={result_euler.cosine:.4f}, "
        f"rk4 cos={result_rk4.cosine:.4f} — proceed to H2(b) RK4 swap"
    )


# ---------- Ant tests ----------


# Phase 1 defaults: H=32, N=30 random unit directions, eps=1e-3.
# Override via env vars for quick smoke runs:
#   MSK_PHASE1_N (default 30)
#   MSK_PHASE1_H (default 32)
#   MSK_PHASE1_NENVS (default 64)
def _phase1_defaults() -> dict:
    return {
        "n_directions": int(os.environ.get("MSK_PHASE1_N", 30)),
        "steps_num": int(os.environ.get("MSK_PHASE1_H", 32)),
        "num_envs": int(os.environ.get("MSK_PHASE1_NENVS", 64)),
        "eps": float(os.environ.get("MSK_PHASE1_EPS", 1e-3)),
    }


def _ant_shac(
    *,
    init_policy: str | None,
    steps_num: int,
    num_envs: int,
    logdir: str,
    extra_env: dict | None = None,
) -> SHAC:
    shac = _shac_from_cfg(
        "msk_warp/configs/experiments/ant_shac_from_ppo.yaml",
        logdir=logdir,
        seed=42,
        num_envs=num_envs,
        steps_num=steps_num,
        extra_env=extra_env,
    )
    if init_policy is not None:
        shac.load(init_policy, reset_optimizers=True)
    return shac


def _ppo_checkpoint_path() -> str:
    return os.path.expanduser(
        "~/checkpoints/ant_ppo_bootstrap_iter500_reward5828.841.pt"
    )


def _run_and_record(
    name: str,
    shac: SHAC,
    cfg_label: str,
    *,
    n_directions: int,
    eps: float,
    direction_seed: int = 0,
) -> FdAdResult:
    print(
        f"\n=== Phase 1 measurement: {name} "
        f"(N={n_directions}, eps={eps}, H={shac.steps_num}, "
        f"num_envs={shac.num_envs}) ==="
    )
    result = _compute_fd_ad_projections(
        shac,
        n_directions=n_directions,
        eps=eps,
        direction_seed=direction_seed,
        verbose=False,
    )
    result.config = cfg_label
    _save_pickle(result, name)
    _append_summary_json(
        _summarize_for_json(result, name),
        _DATA_DIR / "policy_gradient_diagnostic_summary.json",
    )
    print(
        f"[{name}] cos={result.cosine:+.4f}  "
        f"CI95=[{result.bootstrap_ci_low:+.4f}, {result.bootstrap_ci_high:+.4f}]  "
        f"p(cos>0)={result.permutation_p_positive:.4f}"
    )
    print(
        f"  ||theta||={result.theta_norm:.4e}  ||g_AD||={result.g_ad_norm:.4e}  "
        f"L_center={result.extras['L_center']:.6f}"
    )
    return result


def test_ant_random_init():
    """'Global' gradient quality: ant from random actor init at H=32.
    Compared against ant-from-PPO to localize whether failure is theta-specific."""
    d = _phase1_defaults()
    shac = _ant_shac(
        init_policy=None,
        steps_num=d["steps_num"],
        num_envs=d["num_envs"],
        logdir="logs/_phase1_ant_random",
    )
    result = _run_and_record(
        "ant_random_init",
        shac,
        cfg_label="ant_shac_from_ppo (random actor init)",
        n_directions=d["n_directions"],
        eps=d["eps"],
        direction_seed=0,
    )
    # No hard threshold here — recorded for the writeup.
    assert np.isfinite(result.cosine)


def test_ant_from_ppo_init():
    """The decisive Phase 1 measurement.

    Decision tree (per plan):
      cos >= 0.9 -> H3 reward landscape
      0 < cos < 0.9 -> H2 discretization/BPTT
      cos <= 0 -> H1 autodiff bug
    """
    d = _phase1_defaults()
    shac = _ant_shac(
        init_policy=_ppo_checkpoint_path(),
        steps_num=d["steps_num"],
        num_envs=d["num_envs"],
        logdir="logs/_phase1_ant_ppo",
    )
    result = _run_and_record(
        "ant_from_ppo",
        shac,
        cfg_label="ant_shac_from_ppo (loaded PPO walking checkpoint)",
        n_directions=d["n_directions"],
        eps=d["eps"],
        direction_seed=0,
    )
    if result.cosine >= 0.9:
        verdict = "H3 (reward landscape)"
    elif result.cosine > 0:
        verdict = "H2 (discretization / BPTT)"
    else:
        verdict = "H1 (autodiff bug)"
    print(f"[ant_from_ppo] verdict: {verdict}")
    assert np.isfinite(result.cosine)


# ---------- Ablations (Phase 1 §5–6) ----------


def _sweep_at_ppo_init(
    *,
    horizons: list[int] | None = None,
    epsilons: list[float] | None = None,
    state_grad_clips: list[float] | None = None,
    n_directions: int = 30,
    num_envs: int = 16,
) -> None:
    """Run the requested ablations at PPO walking init.

    Each call recreates SHAC fresh + reloads the PPO checkpoint to ensure
    no cross-test state leakage."""
    ckpt = _ppo_checkpoint_path()

    if horizons is not None:
        for H in horizons:
            shac = _ant_shac(
                init_policy=ckpt,
                steps_num=H,
                num_envs=num_envs,
                logdir=f"logs/_phase1_horizon_H{H}",
            )
            _run_and_record(
                f"ant_from_ppo_H{H}",
                shac,
                cfg_label=f"ant_shac_from_ppo (PPO init, H={H})",
                n_directions=n_directions,
                eps=1e-3,
                direction_seed=0,
            )

    if epsilons is not None:
        for eps in epsilons:
            shac = _ant_shac(
                init_policy=ckpt,
                steps_num=32,
                num_envs=num_envs,
                logdir=f"logs/_phase1_eps_{eps:.0e}",
            )
            tag = f"eps{eps:.0e}".replace("-", "m").replace("+", "p")
            _run_and_record(
                f"ant_from_ppo_{tag}",
                shac,
                cfg_label=f"ant_shac_from_ppo (PPO init, eps={eps})",
                n_directions=n_directions,
                eps=eps,
                direction_seed=0,
            )

    if state_grad_clips is not None:
        for clip in state_grad_clips:
            shac = _ant_shac(
                init_policy=ckpt,
                steps_num=32,
                num_envs=num_envs,
                logdir=f"logs/_phase1_clip_{clip}",
            )
            # state_grad_clip is read at SHAC __init__ from cfg, so override after.
            shac.state_grad_clip = clip
            _run_and_record(
                f"ant_from_ppo_clip{clip}",
                shac,
                cfg_label=f"ant_shac_from_ppo (PPO init, state_grad_clip={clip})",
                n_directions=n_directions,
                eps=1e-3,
                direction_seed=0,
            )


def test_horizon_ablation():
    """Plan §5: H ∈ {8, 32, 64} at PPO init. Monotonic cos decay = H2 signature."""
    _sweep_at_ppo_init(horizons=[8, 16, 32, 64])


def test_epsilon_sweep():
    """Plan §4: eps ∈ {1e-2, 1e-3, 1e-4, 1e-5, 1e-6} at H=32, PPO init.
    Optimal central-FD eps ≈ 5e-3 for float32. Eps too small -> noise from
    L+/L- truncation; too large -> Taylor truncation. The cos vs eps curve
    should peak in this range."""
    _sweep_at_ppo_init(epsilons=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])


def test_state_grad_clip_ablation():
    """Plan §6: state_grad_clip 5.0 vs 0.0 at PPO init."""
    _sweep_at_ppo_init(state_grad_clips=[0.0, 5.0])


# ---------- Phase 2 confirmation experiments (Exps 1, 4, 5) ----------


def test_phase2_exp1_multi_seed_decay0_eps1em2():
    """Phase 2 Exp 1: Multi-seed stability of decay=0, eps=1e-2 cos=+0.21.

    Phase 1's single-seed reading at (decay=0, eps=1e-2, H=32, PPO init,
    seed=0) was cos=+0.207 with CI95 [-0.132, +0.555] — CI crosses zero,
    p_pos=0.13. Exp 1 tests whether the *positive sign* reproduces across
    5 direction seeds.

    Pass: cos > 0 on >=4/5 seeds AND pooled cos across 150 directions
    has CI95 excluding zero.
    Fail (<=2/5 positive or pooled CI crosses zero): seed=0 artifact;
    H2 weakens before any RK4 work.
    """
    ckpt = _ppo_checkpoint_path()
    seeds = [0, 1, 2, 3, 4]
    per_seed_results: list[FdAdResult] = []
    pooled_ad_list: list[np.ndarray] = []
    pooled_fd_list: list[np.ndarray] = []
    for seed in seeds:
        shac = _ant_shac(
            init_policy=ckpt,
            steps_num=32,
            num_envs=16,
            logdir=f"logs/_phase2_exp1_seed{seed}",
        )
        shac.state_grad_decay = 0.0
        shac.state_grad_clip = 0.0
        result = _run_and_record(
            f"exp1_multi_seed_s{seed}",
            shac,
            cfg_label=(
                f"Phase 2 Exp 1: PPO init, H=32, decay=0, eps=1e-2, "
                f"direction_seed={seed}"
            ),
            n_directions=30,
            eps=1e-2,
            direction_seed=seed,
        )
        per_seed_results.append(result)
        pooled_ad_list.append(result.ad_proj)
        pooled_fd_list.append(result.fd_proj)

    # Pooled measurement across all 150 directions
    pooled_ad = np.concatenate(pooled_ad_list)
    pooled_fd = np.concatenate(pooled_fd_list)
    pooled_cos = _cosine(pooled_ad, pooled_fd)
    pooled_ci_lo, pooled_ci_hi = _bootstrap_cosine_ci(pooled_ad, pooled_fd)
    pooled_p_pos = _permutation_p_positive(pooled_ad, pooled_fd)
    n_positive = sum(1 for r in per_seed_results if r.cosine > 0)
    per_seed_cosines = [r.cosine for r in per_seed_results]

    print(f"\n[Exp 1 pooled] cos={pooled_cos:+.4f}  "
          f"CI95=[{pooled_ci_lo:+.4f}, {pooled_ci_hi:+.4f}]  "
          f"p(cos>0)={pooled_p_pos:.4f}")
    print(f"  per-seed positive count: {n_positive}/{len(seeds)}")
    print(f"  per-seed cosines: "
          f"{['{:+.3f}'.format(c) for c in per_seed_cosines]}")

    # Save pooled summary as its own JSON entry
    pooled_entry = {
        "name": "exp1_pooled_decay0_eps1em2",
        "config": "Phase 2 Exp 1 pooled across 5 direction seeds",
        "n_directions": int(len(pooled_ad)),
        "eps": 1e-2,
        "n_seeds": len(seeds),
        "n_positive_seeds": int(n_positive),
        "per_seed_cosines": per_seed_cosines,
        "cosine": pooled_cos,
        "cosine_ci_95": [pooled_ci_lo, pooled_ci_hi],
        "permutation_p_cos_positive": pooled_p_pos,
        "g_ad_norm": float(np.mean([r.g_ad_norm for r in per_seed_results])),
    }
    _append_summary_json(
        pooled_entry,
        _DATA_DIR / "policy_gradient_diagnostic_summary.json",
    )

    # Verdict per plan
    pooled_ci_excludes_zero = pooled_ci_lo > 0 or pooled_ci_hi < 0
    if n_positive >= 4 and pooled_ci_excludes_zero:
        verdict = "PASS — sign reproduces and pooled CI excludes zero"
    elif n_positive >= 4:
        verdict = "PARTIAL — sign reproduces but pooled CI crosses zero"
    elif n_positive <= 2:
        verdict = "FAIL — sign does not reproduce; seed=0 was an artifact"
    else:
        verdict = "AMBIGUOUS — sign reproduces on 3/5 seeds"
    print(f"[Exp 1 verdict] {verdict}")


def test_phase2_exp4_h1_ppo_theta():
    """Phase 2 Exp 4: One-step AD vs FD at PPO walking theta (H=1).

    Isolates BPTT chaining from single-step kernel correctness. The
    docs/ant_local_minima_analysis.md standing-policy reading was cos
    ~0.94 at H=1; if PPO walking theta shows similar, the noise is
    BPTT-chained (H2), not single-step.

    Pass: cos >= 0.85 (confirms BPTT chaining is the noise source).
    Fail (cos < 0.5): single-step gradient bad; points at H1b
    ant-specific (Newton implicit-diff).
    """
    ckpt = _ppo_checkpoint_path()
    shac = _ant_shac(
        init_policy=ckpt,
        steps_num=1,
        num_envs=16,
        logdir="logs/_phase2_exp4_h1",
    )
    shac.state_grad_decay = 0.0
    shac.state_grad_clip = 0.0
    result = _run_and_record(
        "exp4_h1_ppo_theta",
        shac,
        cfg_label="Phase 2 Exp 4: PPO init, H=1, decay=0, eps=1e-3",
        n_directions=30,
        eps=1e-3,
        direction_seed=0,
    )

    if result.cosine >= 0.85:
        verdict = "PASS — single-step gradient correct; noise is BPTT-chained (H2)"
    elif result.cosine < 0.5:
        verdict = "FAIL — single-step gradient bad; reorder toward H1b"
    else:
        verdict = f"AMBIGUOUS — cos={result.cosine:.4f} in [0.5, 0.85)"
    print(f"[Exp 4 verdict] {verdict}")


def test_phase2_exp5_clean_h_sweep():
    """Phase 2 Exp 5: Clean H sweep at decay=0, eps=1e-2.

    Phase 1's H sweep was confounded by decay=0.85 attenuating ||g_AD|| at
    every H. This rerun gives the textbook H2 picture: cos should
    monotonically decay with H if fixed-step Euler's discretization noise
    amplifies through BPTT chaining.

    Pass: cos monotonically decays with H in {8,16,32,64}.
    Fail (no monotonic decay + Exp 4 also passes): points at H3.
    """
    ckpt = _ppo_checkpoint_path()
    horizon_cosines: list[tuple[int, float]] = []
    for H in [8, 16, 32, 64]:
        shac = _ant_shac(
            init_policy=ckpt,
            steps_num=H,
            num_envs=16,
            logdir=f"logs/_phase2_exp5_h{H}",
        )
        shac.state_grad_decay = 0.0
        shac.state_grad_clip = 0.0
        result = _run_and_record(
            f"exp5_h_sweep_H{H}",
            shac,
            cfg_label=f"Phase 2 Exp 5: PPO init, H={H}, decay=0, eps=1e-2",
            n_directions=30,
            eps=1e-2,
            direction_seed=0,
        )
        horizon_cosines.append((H, result.cosine))

    # Check monotonic decay
    cosines = [c for _, c in horizon_cosines]
    diffs = [cosines[i + 1] - cosines[i] for i in range(len(cosines) - 1)]
    monotonic_decay = all(d <= 0 for d in diffs)
    print(f"\n[Exp 5 H sweep] {horizon_cosines}")
    print(f"  pairwise diffs: {diffs}")
    if monotonic_decay:
        verdict = "PASS — monotonic decay; textbook H2 signature"
    else:
        verdict = "FAIL — non-monotonic; H2 less likely"
    print(f"[Exp 5 verdict] {verdict}")


def test_state_grad_decay_ablation():
    """Follow-up: ant_shac_from_ppo.yaml ships with state_grad_decay=0.85
    which over H=32 attenuates the BPTT signal by 0.85^32 ≈ 0.005. Compare
    cos at decay=0.0 (raw BPTT) vs decay=0.85 to test whether the decay
    regularizer is what's making cos near zero at H=32."""
    ckpt = _ppo_checkpoint_path()
    for decay in (0.0, 0.5, 0.85):
        shac = _ant_shac(
            init_policy=ckpt,
            steps_num=32,
            num_envs=16,
            logdir=f"logs/_phase1_decay_{decay}",
        )
        shac.state_grad_decay = decay
        shac.state_grad_clip = 0.0  # disable clip to isolate decay effect
        _run_and_record(
            f"ant_from_ppo_decay{decay}",
            shac,
            cfg_label=f"ant_shac_from_ppo (PPO init, state_grad_decay={decay}, clip=0)",
            n_directions=30,
            eps=1e-3,
            direction_seed=0,
        )


if __name__ == "__main__":
    test_determinism_freeze_ant_from_ppo()
    test_cartpole_sanity_random_init()
    test_gate02_cartpole_rk4_tape_smoke()
    test_ant_random_init()
    test_ant_from_ppo_init()
    test_horizon_ablation()
    test_epsilon_sweep()
    test_state_grad_clip_ablation()
