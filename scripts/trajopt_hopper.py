"""Periodic-orbit trajectory optimisation through the differentiable simulator.

Phase 3 left one thing unsettled: whether *any* muscle gait exists for ``hopper_muscle.xml``.
Seven optimiser, reward and actuator combinations all landed in one of two attractors, and PPO
landed where SHAC did, so the question is about the model rather than the optimiser. This driver
asks it without a policy at all.

It optimises an open-loop control sequence over one cycle, plus the initial state, directly through
the adjoint, and asks for a **periodic orbit that advances**. That formulation does the
discriminating work, and stage 0 measured that it does
(``docs/research/phase4-capability/protocol.md``)::

    real motor gait   residual 0.234   advance +1.803 m/cycle
    standing          residual 0.482   advance -0.006 m/cycle   -> excluded by the objective
    diving            residual 1.007   advance +0.128 m/cycle   -> excluded by the periodicity term

Both Phase 3 attractors are therefore outside the feasible set by construction, which is what no
reward setting in Phase 3 achieved.

Every world is an independent random restart, optimised simultaneously in one backward pass, so
512 restarts cost one rollout's worth of wall clock rather than 512.

Gates, thresholds and the response to each outcome are pre-registered in the protocol. Run the
motor positive control first: **if it fails, the muscle result is void.**

Usage::

    # stage 1a, cost probe
    .venv/Scripts/python.exe scripts/trajopt_hopper.py --cfg configs/hopper_motor_shac.yaml \\
        --scales docs/research/phase4-capability/gait_motor_seed2.json \\
        --cycle 32 --worlds 256 --iters 20 --probe

    # a real cell
    .venv/Scripts/python.exe scripts/trajopt_hopper.py --cfg configs/hopper_muscle_shac.yaml \\
        --scales docs/research/phase4-capability/gait_motor_seed2.json \\
        --cycle 27 --worlds 512 --iters 500 --out logs/phase4/muscle_T27.json
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import yaml

from msk_warp import PACKAGE_ROOT
from msk_warp.bridge import WarpSimStep
from msk_warp.envs import ENV_MAP
from msk_warp.utils.gait import periodicity_residual, shape_vector

# Bounds on the free initial state, from the protocol. Height and pitch are symmetric about the
# reset pose; thigh and leg are mapped into the interior of their (-2.618, 0) range so a restart
# never starts exactly on a joint limit, where the constraint derivative is undefined (G3.2).
HEIGHT_BOUND = 0.15       # m, about the reset height
PITCH_BOUND = 0.30        # rad, against a termination angle of pi/6
KNEE_HIP_SPAN = 1.20      # rad, span available to thigh and leg below the rest pose
FOOT_BOUND = 0.70         # rad, inside the (-0.785, 0.785) range
# Inset every limited joint by the same 0.05 rad G3.2 uses to move a test pose off a limit.
# Without it a saturated sigmoid underflows to exactly 0.0 in float32 and a restart begins on the
# thigh or leg upper limit, where the constraint switches on and the derivative is undefined.
JOINT_MARGIN = 0.05
QVEL_BOUND = (2.0, 2.0, 3.0, 5.0, 5.0, 5.0)

TERMINATION_HEIGHT = -0.45

#: Cycles run before the scored one, so activation reaches the limit cycle its own control
#: implies. One suffices: the measured end-of-cycle activation gap falls 1.3e-06 after one
#: cycle and 8.4e-12 after two. See `rollout` and docs/VALIDITY.md CL-02.
WARMUP_CYCLES = 1


def load_cfg(cfg_path):
    if not os.path.isabs(cfg_path) and (PACKAGE_ROOT / cfg_path).exists():
        cfg_path = str(PACKAGE_ROOT / cfg_path)
    with open(cfg_path) as fh:
        return yaml.safe_load(fh), cfg_path


def build_env(cfg, worlds, device, episode_length):
    """Differentiable environment with the episode machinery out of the way.

    Early termination and stochastic init are off and the episode is longer than any rollout here,
    so no world is ever reset mid-cycle. The environment is used for its model, its substep count
    and its action map; the rollout is driven through ``WarpSimStep`` directly rather than through
    ``env.step``, which would apply reward, termination and reset logic this objective does not want.
    """
    env_cfg = dict(cfg["params"]["env"])
    name = env_cfg.pop("name")
    env_cfg.pop("num_actors", None)
    env_cfg.pop("episode_length", None)
    env_cfg.pop("stochastic_init", None)
    env_cfg.pop("early_termination", None)
    return ENV_MAP[name](
        num_envs=worlds,
        device=device,
        no_grad=False,
        episode_length=episode_length,
        stochastic_init=False,
        early_termination=False,
        **env_cfg,
    )


def _span():
    """Interior span for thigh and leg, both margins removed."""
    return KNEE_HIP_SPAN - 2.0 * JOINT_MARGIN


def initial_state(z, n_act_state, device):
    """Map unconstrained parameters onto a bounded, physical initial ``(qpos, qvel, act)``.

    ``qpos[0]`` is always 0: displacement is measured from wherever the cycle starts, so the
    absolute track position is not a degree of freedom.
    """
    qvel_bound = torch.tensor(QVEL_BOUND, device=device)
    zero = torch.zeros(z.shape[0], 1, device=device)
    qpos = torch.cat([
        zero,                                                   # x
        HEIGHT_BOUND * torch.tanh(z[:, 0:1]),                   # height, relative to the reset pose
        PITCH_BOUND * torch.tanh(z[:, 1:2]),                    # torso pitch
        -(JOINT_MARGIN + _span() * torch.sigmoid(z[:, 2:3])),   # thigh, strictly inside its range
        -(JOINT_MARGIN + _span() * torch.sigmoid(z[:, 3:4])),   # leg
        FOOT_BOUND * torch.tanh(z[:, 4:5]),                     # foot
    ], dim=1)
    qvel = qvel_bound * torch.tanh(z[:, 5:11])
    # Activation is NOT a decision variable. Under a T_c-periodic control the activation subsystem
    # is autonomous -- MuJoCo's dyntype=muscle computes act_dot from (ctrl, act) alone -- so it has
    # a unique globally attracting periodic solution act*, determined by the control (VALIDITY
    # CL-02). The six sigmoid(z[:, 11:17]) parameters this used to carry were therefore redundant
    # *and* inconsistent with periodicity, and the optimiser spent them: the celebrated per-muscle
    # "activation closing errors" [0.166, 0.427, 0.858, 0.886, 0.369, 0.212] of the T_c 16 candidate
    # are exactly |act_0 - act*|, a mis-specified initial condition rather than a trajectory that
    # fails to close. Start neutral and let the warm-up cycles in `rollout` carry activation onto
    # act* by themselves, which is exact because it uses the engine rather than a reimplementation.
    act = torch.full((z.shape[0], n_act_state), 0.5, device=device) if n_act_state \
        else torch.zeros(z.shape[0], 0, device=device)
    return qpos, qvel, act


def _cycle(env, u_logits, qpos, qvel, act, cycle, violation):
    """One replay of the periodic control. Shared by the warm-up and the scored cycle."""
    for t in range(cycle):
        ctrl = env._to_ctrl(torch.tanh(u_logits[:, t]))
        qpos, qvel, act = WarpSimStep.apply(ctrl, qpos, qvel, act, env)
        # Same guard SHAC's rollout uses: a diverging restart must not poison the tape.
        qpos = qpos.clamp(-100.0, 100.0)
        qvel = qvel.clamp(-100.0, 100.0)
        violation = violation + torch.relu(TERMINATION_HEIGHT - qpos[:, 1])
    return qpos, qvel, act, violation


def rollout(env, u_logits, z, scales, cycle, n_act_state, device, warmup=WARMUP_CYCLES):
    """Warm-up cycles, then one scored cycle. Returns the objective's three terms.

    Kept differentiable end to end: the adjoint runs from the objective back through every substep
    of every control step to both the control sequence and the initial state.

    **The warm-up is what makes the residual mean what it says.** Activation is determined by the
    control, not chosen (VALIDITY CL-02), and it converges onto its limit cycle geometrically -- the
    measured end-of-cycle gap falls 1.3e-06, 8.4e-12, 4.2e-17 over three cycles. So after one
    warm-up cycle the scored cycle *starts* on the activation manifold and therefore *ends* there
    too, and activation closure is automatic rather than something the residual has to police.

    That retires protocol amendment 1 instead of patching it. Amendment 1 excluded activation from
    the residual and leaned on open-loop verification to catch non-closure; verification was later
    cut from 10 cycles to 3 and the gap was never checked (VALIDITY IN-05). Putting activation back
    in is not the fix either, because no activation scale is principled and the choice flips the
    verdict by a factor of 26 (CL-01). Making closure structural removes the question.
    """
    qpos, qvel, act = initial_state(z, n_act_state, device)
    violation = torch.zeros(qpos.shape[0], device=device)
    for _ in range(warmup):
        qpos, qvel, act, violation = _cycle(env, u_logits, qpos, qvel, act, cycle, violation)

    start_x = qpos[:, 0]
    # Mechanical state only -- and after the warm-up that IS the full state, because activation is
    # on its own periodic solution and closes by construction.
    start_shape = shape_vector(qpos, qvel)
    qpos, qvel, act, violation = _cycle(env, u_logits, qpos, qvel, act, cycle, violation)

    dt = env.substeps * float(env.mjm.opt.timestep)
    velocity = (qpos[:, 0] - start_x) / (cycle * dt)
    # The closing error as a VECTOR, not just its norm. The augmented Lagrangian needs the vector:
    # its penalty is ||e||^2, which is smooth at e = 0, whereas the scalar residual R = ||e|| has an
    # infinite-derivative kink exactly there -- at the solution. Constraining R instead of e would
    # put a non-differentiable point where convergence has to happen.
    error = (shape_vector(qpos, qvel) - start_shape) / scales
    residual = error.pow(2).mean(dim=-1).sqrt()
    return velocity, error, residual, violation / ((warmup + 1) * cycle)


@torch.no_grad()
def verify(env, u_logits, z, scales, cycle, n_act_state, device, cycles):
    """Re-simulate open loop for ``cycles`` consecutive cycles. Survival and speed, no gradients.

    An optimum is not yet a gait. This is the check the Phase 3 open-loop sweep lacked.
    """
    qpos, qvel, act = initial_state(z, n_act_state, device)
    # Warm up exactly as `rollout` does, so verification starts on the same trajectory the residual
    # was scored on. Without this the two would disagree about which cycle is the candidate.
    violation = torch.zeros(qpos.shape[0], device=device)
    for _ in range(WARMUP_CYCLES):
        qpos, qvel, act, violation = _cycle(env, u_logits, qpos, qvel, act, cycle, violation)
    start_x = qpos[:, 0].clone()
    alive = torch.ones(qpos.shape[0], dtype=torch.bool, device=device)
    survived = torch.zeros(qpos.shape[0], device=device)
    for _ in range(cycles):
        for t in range(cycle):
            ctrl = env._to_ctrl(torch.tanh(u_logits[:, t]))
            qpos, qvel, act = WarpSimStep.apply(ctrl, qpos, qvel, act, env)
            qpos = qpos.clamp(-100.0, 100.0)
            qvel = qvel.clamp(-100.0, 100.0)
            alive &= qpos[:, 1] >= TERMINATION_HEIGHT
            survived += alive.float()
    dt = env.substeps * float(env.mjm.opt.timestep)
    velocity = (qpos[:, 0] - start_x) / (cycles * cycle * dt)
    return alive, velocity, survived / (cycles * cycle)


def save_params(out, u_logits, z, optimiser=None, order=None, done=0, extra=None):
    """Persist parameters, Adam state and progress beside the result JSON.

    Written every logging interval, not only at the end. This machine runs at about 24.8 GB
    committed of a 31.3 GB limit before any Python starts, and three attempts at the motor control
    were OOM-killed mid-run, one of them after it had already converged and cleared its gate. The
    Adam moments are saved with the parameters so a staged run -- several short processes, each
    exiting and releasing everything -- is continuous rather than restarting momentum each stage.
    """
    if out is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    payload = {"u_logits": u_logits.detach().cpu().numpy(), "z": z.detach().cpu().numpy(),
               "iterations_done": np.array(done)}
    if optimiser is not None:
        for name, tensor in zip(("u", "z"), (u_logits, z)):
            state = optimiser.state.get(tensor, {})
            if "exp_avg" in state:
                payload[f"exp_avg_{name}"] = state["exp_avg"].cpu().numpy()
                payload[f"exp_avg_sq_{name}"] = state["exp_avg_sq"].cpu().numpy()
                payload[f"step_{name}"] = np.array(float(state["step"]))
    if order is not None:
        payload["order"] = order.cpu().numpy()
    # Augmented-Lagrangian multipliers and penalty weights, so a staged run resumes mid-solve
    # rather than restarting the outer loop with y = 0 -- which would throw away exactly the
    # information the method accumulates.
    for key, tensor in (extra or {}).items():
        payload[key] = tensor.detach().cpu().numpy()
    np.savez_compressed(os.path.splitext(out)[0] + "_params.npz", **payload)


def restore_optimiser(optimiser, saved, u_logits, z, device):
    """Put the saved Adam moments back, so a staged run does not re-warm its momentum."""
    for name, tensor in zip(("u", "z"), (u_logits, z)):
        if f"exp_avg_{name}" not in saved:
            continue
        optimiser.state[tensor] = {
            "step": torch.tensor(float(saved[f"step_{name}"])),
            "exp_avg": torch.tensor(saved[f"exp_avg_{name}"], device=device),
            "exp_avg_sq": torch.tensor(saved[f"exp_avg_sq_{name}"], device=device),
        }


#: Constraint bars the Pareto summary reports the fastest world under. A periodic orbit has R = 0,
#: so these are "how fast can a nearly-closed cycle go", which is the quantity the closure-versus-
#: speed trade is about (docs/VALIDITY.md CL-14).
PARETO_BARS = (0.05, 0.10, 0.20, 0.30)


def summarise(tag, velocity, residual, violation, objective, auglag=False):
    v, r, h, j = (t.detach().cpu().numpy() for t in (velocity, residual, violation, objective))
    # Ranking by the objective is meaningless under an augmented Lagrangian, because each world
    # carries its own multipliers: max_e (-y.e - rho/2 |e|^2) = |y|^2 / 2 rho, so argmax(J) selects
    # the world with the largest multipliers rather than the best gait. Observed directly -- a run
    # reported "J +80.8, R 1.436" while the tightest world in the same population sat at R 0.058.
    # With a constraint the meaningful ranking is by constraint violation.
    best = int(np.argmin(r)) if auglag else int(np.argmax(j))
    pareto = {}
    for bar in PARETO_BARS:
        under = r < bar
        pareto[f"fastest_velocity_under_R{bar}"] = float(v[under].max()) if under.any() else None
        pareto[f"worlds_under_R{bar}"] = int(under.sum())
    return dict(pareto, **{
        "tag": tag,
        "best_world": best,
        "best_objective": float(j[best]),
        "best_velocity": float(v[best]),
        "best_residual": float(r[best]),
        "best_violation": float(h[best]),
        "velocity_median": float(np.median(v)),
        "velocity_max": float(v.max()),
        "residual_median": float(np.median(r)),
        "residual_min": float(r.min()),
        "violation_median": float(np.median(h)),
        "objective_median": float(np.median(j)),
        "ranked_by": "residual" if auglag else "objective",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--scales", required=True,
                    help="stage-0 JSON; its component scales are the single ruler both models use")
    ap.add_argument("--cycle", type=int, required=True, help="cycle length T_c in control steps")
    ap.add_argument("--worlds", type=int, default=512, help="independent random restarts")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.02)
    # lambda = 4.0 is derived from stage 0's three reference points on the shared ruler, not tuned:
    # real gait (v 4.01, R 0.234), standing (v -0.01, R 0.088), diving (v 1.28, R 0.485). It is the
    # smallest round value for which J(gait) 3.07 > J(standing) -0.36 > J(diving) -0.66. Below 3.25
    # the dive outranks standing, which would point the optimiser at the worse of the two attractors.
    ap.add_argument("--lam", type=float, default=4.0, help="periodicity weight; see the protocol")
    ap.add_argument("--mu", type=float, default=20.0, help="height-violation weight")
    ap.add_argument("--seed", type=int, default=0)
    # 3, not 10. The trained motor policy's own gait, cycled open loop, falls after 3.7 cycles
    # (scripts/replay_openloop.py), so 10 is unachievable by any real gait. 3 sits below that and
    # above the diving attractor's 2.4 cycles, so it separates them. See results.md amendment 2.
    ap.add_argument("--verify-cycles", type=int, default=3)
    ap.add_argument("--velocity-bar", type=float, default=0.5,
                    help="G4.2 uses 0.5, the same bar G3.5 used; the motor control uses 1.0")
    ap.add_argument("--residual-bar", type=float, default=0.4684,
                    help="2 x R_ref, with R_ref = 0.2342 measured in stage 0")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--resume", default=None, help="continue from a saved _params.npz")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip optimisation; load --resume parameters and run verification only")
    ap.add_argument("--skip-verify", action="store_true",
                    help="optimise and checkpoint only; for intermediate stages of a staged run")
    ap.add_argument("--auglag", action="store_true",
                    help="enforce periodicity as a CONSTRAINT with multipliers instead of the "
                         "lambda*R penalty. The penalty provably trades closure against speed in "
                         "both models (docs/VALIDITY.md CL-14); an augmented Lagrangian does not.")
    ap.add_argument("--al-inner", type=int, default=20,
                    help="Adam iterations between multiplier updates")
    ap.add_argument("--al-rho0", type=float, default=4.0,
                    help="initial penalty weight; 4.0 matches the lambda the penalty runs used, so "
                         "the first inner solve starts where the penalty method left off")
    ap.add_argument("--al-rho-max", type=float, default=1.0e4)
    ap.add_argument("--al-eta", type=float, default=0.5,
                    help="a world's rho is raised only if its constraint norm failed to fall to "
                         "this fraction of its value at the previous outer update")
    ap.add_argument("--probe", action="store_true",
                    help="report term magnitudes and seconds per iteration, then stop (stage 1a)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg, cfg_path = load_cfg(args.cfg)
    env = build_env(cfg, args.worlds, args.device, args.cycle * (args.verify_cycles + 2))

    with open(args.scales) as fh:
        stage0 = json.load(fh)
    scales_list = stage0["per_world"][0]["scales"]
    scales = torch.tensor(scales_list, device=args.device)

    n_act_state = env.warp_data.act.shape[1] if env.has_act else 0
    expected = (env.mjm.nq - 1) + env.mjm.nv
    if len(scales_list) != expected:
        raise SystemExit(
            f"--scales has {len(scales_list)} components but this model's shape vector has "
            f"{expected}. The stage-0 ruler must come from a model with the same mechanical layout."
        )

    u_logits = torch.randn(args.worlds, args.cycle, env.num_actions, device=args.device, requires_grad=True)
    # 11, not 11 + n_act_state: the initial activation is determined by the control, so it is
    # no longer a decision variable (VALIDITY CL-02).
    z = torch.randn(args.worlds, 11, device=args.device, requires_grad=True)
    saved, already_done = None, 0
    if args.resume:
        saved = np.load(args.resume)
        with torch.no_grad():
            u_logits.copy_(torch.tensor(saved["u_logits"], device=args.device))
            z.copy_(torch.tensor(saved["z"], device=args.device))
        already_done = int(saved["iterations_done"]) if "iterations_done" in saved else 0
        print(f"resumed from {args.resume} at iteration {already_done}", flush=True)
    elif args.verify_only:
        raise SystemExit("--verify-only needs --resume to say which parameters to verify")

    optimiser = torch.optim.Adam([u_logits, z], lr=args.lr)
    if saved is not None:
        restore_optimiser(optimiser, saved, u_logits, z, args.device)

    # ---- augmented-Lagrangian state, one multiplier vector and one penalty weight per world ----
    #
    # A soft penalty trades periodicity for speed; it never enforces it. Measured at T_c 16, same
    # budget and seed, only lambda changed (docs/VALIDITY.md CL-14):
    #
    #     muscle  lambda  4 -> v +1.4, R 0.25      lambda 40 -> v +0.01, R 0.036
    #     motor   lambda  4 -> v +2.9, R 0.55      lambda 40 -> v -0.08, R 0.058
    #
    # Ten times the weight buys a seven times tighter cycle and costs essentially all the speed, in
    # BOTH models, so no lambda delivers R -> 0 with v > 0.5. The augmented Lagrangian removes the
    # trade: the objective stays v, the multipliers carry the constraint, and rho only has to be
    # large enough locally rather than large enough to dominate.
    n_con = 11
    al_y = torch.zeros(args.worlds, n_con, device=args.device)
    al_rho = torch.full((args.worlds,), args.al_rho0, device=args.device)
    al_prev = torch.full((args.worlds,), float("inf"), device=args.device)
    if saved is not None and "al_y" in saved.files:
        al_y.copy_(torch.tensor(saved["al_y"], device=args.device))
        al_rho.copy_(torch.tensor(saved["al_rho"], device=args.device))
        al_prev.copy_(torch.tensor(saved["al_prev"], device=args.device))
        print(f"resumed multipliers: |y| median {al_y.norm(dim=-1).median():.3f}, "
              f"rho median {al_rho.median():.1f}", flush=True)
    al_state = {"al_y": al_y, "al_rho": al_rho, "al_prev": al_prev} if args.auglag else None

    def objective():
        velocity, error, residual, violation = rollout(
            env, u_logits, z, scales, args.cycle, n_act_state, args.device,
        )
        if args.auglag:
            obj = (velocity
                   - (al_y * error).sum(dim=-1)
                   - 0.5 * al_rho * error.pow(2).sum(dim=-1)
                   - args.mu * violation)
        else:
            obj = velocity - args.lam * residual - args.mu * violation
        return velocity, residual, violation, obj, error

    history = []
    grad_norms = []
    started = time.time()

    for iteration in range(args.iters if args.verify_only else 0, args.iters):
        env.clear_grad()
        optimiser.zero_grad(set_to_none=True)
        velocity, residual, violation, obj, error = objective()
        (-obj.sum()).backward()
        grad_norms.append(float(torch.cat([u_logits.grad.flatten(), z.grad.flatten()]).norm()))
        optimiser.step()

        if args.auglag and (iteration + 1) % args.al_inner == 0:
            # Outer update, per world and in place so the closure keeps its references.
            with torch.no_grad():
                err = error.detach()
                norm = err.norm(dim=-1)
                al_y.add_(al_rho.unsqueeze(-1) * err)
                # Raise rho only where the constraint is not shrinking fast enough. Worlds that are
                # converging keep a small rho, which is the whole point of the method: the penalty
                # does not have to dominate the objective everywhere.
                stalled = norm > args.al_eta * al_prev
                al_rho.copy_(torch.where(stalled, (al_rho * 10.0).clamp(max=args.al_rho_max), al_rho))
                al_prev.copy_(norm)

        if args.probe or iteration % args.log_every == 0 or iteration == args.iters - 1:
            row = summarise(f"iter{iteration}", velocity, residual, violation, obj, args.auglag)
            row["iteration"] = iteration
            row["grad_norm"] = grad_norms[-1]
            row["seconds"] = time.time() - started
            row["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024 ** 2
            row["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024 ** 2
            history.append(row)
            save_params(args.out, u_logits, z, optimiser, done=already_done + iteration + 1,
                        extra=al_state)
            gc.collect()
            if args.auglag:
                fast = row["fastest_velocity_under_R0.1"]
                print(f"iter {iteration:4d}  Rmin {row['residual_min']:.4f} "
                      f"(v {row['best_velocity']:+5.2f})  |  under R<0.1: "
                      f"{row['worlds_under_R0.1']:3d} worlds, fastest "
                      f"{fast if fast is None else round(fast, 3)}  |  median R "
                      f"{row['residual_median']:.3f}  |g| {row['grad_norm']:9.2e}  "
                      f"{row['seconds']:6.1f}s", flush=True)
            else:
                print(f"iter {iteration:4d}  J {row['best_objective']:+7.3f}  "
                      f"v {row['best_velocity']:+6.2f}  R {row['best_residual']:.3f}  "
                      f"H {row['best_violation']:.3f}  |  median v {row['velocity_median']:+6.2f} "
                      f"R {row['residual_median']:.3f}  |g| {row['grad_norm']:9.2e}  "
                      f"{row['seconds']:6.1f}s  cuda {row['cuda_reserved_mb']:6.0f}MB", flush=True)

        if args.probe and iteration == min(args.iters, 20) - 1:
            elapsed = time.time() - started
            print(f"\nPROBE: {elapsed / (iteration + 1):.2f} s per iteration at "
                  f"worlds={args.worlds} cycle={args.cycle}")
            print(f"PROBE: term magnitudes at iteration 0 -> "
                  f"|v| {abs(history[0]['velocity_median']):.3f}   "
                  f"lambda*R {args.lam * history[0]['residual_median']:.3f}   "
                  f"mu*H {args.mu * history[0]['violation_median']:.3f}")
            return

    if args.skip_verify:
        save_params(args.out, u_logits, z, optimiser, done=already_done + args.iters,
                    extra=al_state)
        print(f"stage complete at iteration {already_done + args.iters}; verification skipped",
              flush=True)
        return

    velocity, residual, violation, obj, _error = objective()
    alive, verified_velocity, alive_fraction = verify(
        env, u_logits.detach(), z.detach(), scales, args.cycle, n_act_state, args.device,
        args.verify_cycles,
    )

    # A candidate counts only if it survives every verification cycle and its open-loop speed is
    # within 25% of what the optimiser reported (protocol, "Verification").
    consistent = (verified_velocity - velocity).abs() <= 0.25 * velocity.abs().clamp(min=1e-6)
    # Surviving the verification is necessary but not sufficient: a trajectory that never closes can
    # still stay upright for three cycles. The gate is the conjunction of all four conditions, so it
    # is computed as one rather than read off a survival count.
    residual_ok = residual <= args.residual_bar
    velocity_ok = velocity >= args.velocity_bar
    qualifies = alive & consistent & residual_ok & velocity_ok
    order = torch.argsort(torch.where(qualifies, verified_velocity, torch.full_like(verified_velocity, -1e9)),
                          descending=True)

    result = {
        "cfg": cfg_path,
        "scales_source": args.scales,
        "cycle": args.cycle,
        "worlds": args.worlds,
        "iters": args.iters,
        "lr": args.lr,
        "lambda": args.lam,
        "mu": args.mu,
        "seed": args.seed,
        "verify_cycles": args.verify_cycles,
        "seconds_total": time.time() - started,
        "seconds_per_iter": (time.time() - started) / args.iters if args.iters else None,
        "grad_norm_median": float(np.median(grad_norms)) if grad_norms else None,
        "grad_norm_p90": float(np.percentile(grad_norms, 90)) if grad_norms else None,
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else None,
        "final": summarise("final", velocity, residual, violation, obj, args.auglag),
        "gate": {
            "velocity_bar": args.velocity_bar,
            "residual_bar": args.residual_bar,
            "passed": bool(qualifies.any()),
        },
        "verified": {
            "survived_all_cycles": int(alive.sum()),
            "speed_consistent": int(consistent.sum()),
            "residual_within_gate": int(residual_ok.sum()),
            "velocity_within_gate": int(velocity_ok.sum()),
            "survived_and_residual_ok": int((alive & residual_ok).sum()),
            "residual_and_velocity_ok": int((residual_ok & velocity_ok).sum()),
            "qualifying": int(qualifies.sum()),
            "best_verified_velocity": float(verified_velocity[order[0]]) if int(qualifies.sum()) else None,
            "best_verified_residual": float(residual[order[0]]) if int(qualifies.sum()) else None,
            "top": [
                {
                    "world": int(w),
                    "optimiser_velocity": float(velocity[w]),
                    "verified_velocity": float(verified_velocity[w]),
                    "residual": float(residual[w]),
                    "alive_fraction": float(alive_fraction[w]),
                    "alive": bool(alive[w]),
                    "consistent": bool(consistent[w]),
                    "qualifies": bool(qualifies[w]),
                }
                for w in order[:10].tolist()
            ],
        },
        "history": history,
        # Standing versus diving in trajectory space rather than policy space.
        "restarts_near_zero_velocity": int((velocity.abs() < 0.1).sum()),
        "restarts_above_gate_residual": int((residual > 0.4684).sum()),
    }

    print(f"\nverified: {result['verified']['qualifying']} of {args.worlds} restarts qualify "
          f"(survived all {args.verify_cycles} cycles and speed within 25%)")
    if result["verified"]["best_verified_velocity"] is not None:
        print(f"best verified: {result['verified']['best_verified_velocity']:+.3f} m/s at "
              f"residual {result['verified']['best_verified_residual']:.4f}")
    else:
        print("best verified: none -- no restart survived verification")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        save_params(args.out, u_logits, z, optimiser, order, done=already_done + args.iters,
                extra=al_state)
        print(f"wrote {args.out} and its _params.npz")


if __name__ == "__main__":
    main()
