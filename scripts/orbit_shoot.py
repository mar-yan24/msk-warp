"""Multistart periodic-orbit shooting on a Phase 4 trajectory-optimisation candidate.

Phase 4 judged a candidate by a scaled RMS periodicity residual against a bar of twice a real
gait's. That instrument cannot answer the existence question: the residual excludes muscle
activation, and no activation scale is principled -- the same candidate passes at 0.4216 with unit
scales and fails at 12.1 with the standing trace's (``docs/VALIDITY.md`` CL-01).

This asks the question exactly instead. A ``T_c``-periodic control admits a periodic orbit **iff**
its ``T_c``-step return map has a fixed point. So reconstruct the candidate's control, build the
map, and shoot. No ruler, no bar, activation included by construction, and the Floquet spectrum at
any root comes free.

Runs in float64 CPU MuJoCo: roughly 5000x cheaper per candidate than the GPU trajectory
optimisation, and independent of the Warp adjoint and its two open defects. Measured cost about
40 s for 96 starts, so the whole eleven-cell Phase 4 sweep is under ten minutes on one core.

Pre-registration: ``docs/research/phase5-orbits/protocol.md``.

Usage::

    .venv/Scripts/python.exe scripts/orbit_shoot.py \\
        --params logs/phase4/muscle_T16_params.npz \\
        --result logs/phase4/muscle_T16.json \\
        --asset assets/hopper_muscle.xml \\
        --scales docs/research/phase4-capability/gait_motor_seed2.json \\
        --starts 96 --out docs/research/phase5-orbits/shoot_muscle_T16.json
"""

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from msk_warp import resolve_model_path
from msk_warp.analysis import orbit as orb
from msk_warp.analysis import stability as st

#: Perturbation radii, in units of component scale. Four radii rather than one so a negative is
#: scoped by how far the search actually looked.
SIGMAS = (0.05, 0.15, 0.30, 0.60)


def _load_trajopt():
    """Import the trajectory optimiser for its initial-state parameterisation.

    Imported rather than reimplemented, at a cost of about 3 s, so the seed states are *exactly*
    the ones the optimiser used. Duplicating ``HEIGHT_BOUND``, ``PITCH_BOUND``, ``KNEE_HIP_SPAN``,
    ``FOOT_BOUND``, ``JOINT_MARGIN`` and ``QVEL_BOUND`` here would silently drift.
    """
    path = Path(__file__).resolve().parent / "trajopt_hopper.py"
    spec = importlib.util.spec_from_file_location("trajopt_hopper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_world(result_path, override=None):
    """The candidate world: the first qualifying entry of ``verified.top``, else ``best_world``."""
    if override is not None:
        return int(override), "explicit"
    if result_path is None:
        return 0, "default"
    with open(result_path) as fh:
        result = json.load(fh)
    for entry in result.get("verified", {}).get("top", []):
        if entry.get("qualifies"):
            return int(entry["world"]), "qualifying"
    return int(result["final"]["best_world"]), "best_objective"


def to_ctrl(mjm, actions, action_strength=1.0):
    """Reproduce the environment's action map, chosen by whether the model has activation state.

    ``HopperMuscleEnv._to_ctrl`` is ``0.5 * (a + 1) * strength`` because muscle control is
    positive-only; ``HopperMotorEnv._to_ctrl`` is ``a * strength``. Getting this wrong would analyse
    a different trajectory from the one the optimiser found, so it is derived from the model rather
    than passed in.
    """
    if mjm.na > 0:
        return 0.5 * (actions + 1.0) * action_strength
    return actions * action_strength


def describe(rmap, x, scales):
    """Everything needed to classify a converged fixed point as gait, stand or slide."""
    out = rmap.roll(x)
    jac = st.jacobian_fd(rmap, x, scales=scales, eps_rel=1e-4)
    sweep = st.eps_sweep(rmap, x, scales=scales)
    spec = st.spectrum(jac.J)
    return {
        "advance_per_cycle_m": out.advance,
        "velocity_mps": out.advance / (rmap.cycle * rmap.control_dt),
        # Within ONE cycle. Over several, divergence dominates and the figure is meaningless --
        # the defect that had 0.508 m reported for a 0.0714 m bounce (docs/VALIDITY.md IN-01).
        "height_range_m": out.height_range,
        "height_min_m": out.height_min,
        "spectral_radius": spec.spectral_radius,
        "unstable_count": spec.unstable_count,
        "period_doubling_margin": spec.period_doubling_margin,
        "eigenvalue_moduli": [float(v) for v in np.abs(spec.eigenvalues)],
        "eps_verdict": sweep.verdict,
        "eps_window_decades": sweep.window_decades,
        "eps_slope": sweep.slope,
        "stance_steps": int(sum(1 for n in out.contacts if n > 0)),
        # Flight, not height range, is what separates a hop from a squat. Measured: the muscle
        # T_c 32 orbit oscillates 0.1335 m vertically -- 72% of a real gait's amplitude -- while
        # airborne for 1 of 32 steps. It is a bob, not a hop. The trained motor gait's orbit is
        # airborne for 22 of 27.
        "flight_steps": int(sum(1 for n in out.contacts if n == 0)),
        "act_excursions": out.act_excursions,
    }


def classify(desc, cycle, standing_advance=st.STANDING_ADVANCE, sliding_range=0.05):
    """Gait / hop-in-place / bob / slide / static, from flight phase and advance."""
    hopping = desc["flight_steps"] >= 2
    moving = abs(desc["advance_per_cycle_m"]) >= standing_advance
    if hopping and moving:
        return "gait" if desc["advance_per_cycle_m"] > 0 else "gait_backwards"
    if hopping:
        return "hop_in_place"
    if desc["height_range_m"] >= sliding_range:
        return "bob"
    return "slide" if moving else "static"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True, help="a trajopt _params.npz")
    ap.add_argument("--asset", required=True, help="e.g. assets/hopper_muscle.xml")
    ap.add_argument("--scales", required=True, help="stage-0 ruler JSON (11 components)")
    ap.add_argument("--result", default=None, help="the matching trajopt JSON, to pick the world")
    ap.add_argument("--world", type=int, default=None)
    ap.add_argument("--starts", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--action-strength", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=0,
                    help="warm-up cycles the candidate was optimised with (trajopt_hopper's "
                         "WARMUP_CYCLES). The scored cycle starts AFTER these, so the orbit to test "
                         "is the state at their end, not the parameterised initial state. Phase 4 "
                         "candidates predate the warm-up and need 0.")
    ap.add_argument("--check-reference", default=None,
                    help="a reference .npz whose ctrl must match the reconstruction")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trajopt = _load_trajopt()
    import torch  # only for the parameterisation; every measurement below is numpy

    params = np.load(args.params)
    u_logits, z = params["u_logits"], params["z"]
    world, how = pick_world(args.result, args.world)
    cycle = int(u_logits.shape[1])

    mjm = mujoco.MjModel.from_xml_path(resolve_model_path(args.asset))
    orb.assert_state_layout(mjm)
    n_act = mjm.na

    actions = np.tanh(u_logits[world].astype(np.float64))
    ctrl = to_ctrl(mjm, actions, args.action_strength)

    if args.check_reference:
        ref = np.load(resolve_model_path(args.check_reference))
        err = float(np.abs(ref["ctrl"].astype(np.float64) - ctrl).max())
        print(f"reconstruction check against {args.check_reference}: max |dctrl| = {err:.3e}")
        if err > 1e-6:
            raise SystemExit(
                f"reconstructed control differs from the exported reference by {err:.3e}; "
                "the world or the action map is wrong, and the analysis would describe a "
                "different trajectory from the one Phase 4 found"
            )

    zw = torch.tensor(z[world: world + 1].astype(np.float64))
    if z.shape[1] == 11 + n_act:          # a pre-warm-up candidate, activation still in z
        qpos0, qvel0, _ = trajopt.initial_state(zw, n_act, "cpu")
    else:
        qpos0, qvel0, _ = trajopt.initial_state(zw, n_act, "cpu")
    x0 = orb.shape_state(qpos0.numpy()[0], qvel0.numpy()[0])

    rmap_seed = orb.ReturnMap(mjm, ctrl)
    for _ in range(args.warmup):
        # Advance to the start of the cycle the optimiser actually scored. Activation is already
        # handled -- ReturnMap pins it to act* -- so only the mechanical state has to be carried.
        out = rmap_seed.roll(x0)
        if out.terminated:
            raise SystemExit("the candidate does not survive its own warm-up cycle")
        x0 = out.state
    if args.warmup:
        print(f"  advanced {args.warmup} warm-up cycle(s) to the scored cycle's start")

    with open(args.scales) as fh:
        scales = np.array(json.load(fh)["per_world"][0]["scales"], dtype=np.float64)
    if len(scales) != orb.state_dim(mjm):
        raise SystemExit(f"--scales has {len(scales)} components, model needs {orb.state_dim(mjm)}")

    rmap = orb.ReturnMap(mjm, ctrl)
    print(f"{Path(args.params).name}: world {world} ({how}), T_c {cycle}, "
          f"nu {mjm.nu} na {n_act}, act* in {rmap.activation_cycles} cycles")
    print(f"  candidate start: scaled residual {rmap.scaled_residual(x0, scales):.5f}")

    checks = st.structural_checks(rmap, x0)
    print(f"  structural: translation {checks.translation_column_error:.2e}, "
          f"act->mech {checks.activation_to_mechanics}, ok={checks.ok()}")

    rng = np.random.default_rng(args.seed)
    starts = [x0]
    for i in range(args.starts - 1):
        sigma = SIGMAS[i % len(SIGMAS)]
        u = rng.normal(size=rmap.dim)
        u /= np.linalg.norm(u / scales)
        starts.append(x0 + sigma * u * scales)

    t0 = time.perf_counter()
    results = st.multistart(rmap, starts, scales=scales, tol=args.tol,
                            max_iter=args.max_iter, classify_terminal=True)
    seconds = time.perf_counter() - t0

    histogram = Counter(r.outcome.value for r in results)
    converged = [r for r in results if r.residual <= args.tol]
    print(f"  {len(results)} starts in {seconds:.1f} s: {dict(histogram)}")
    print(f"  best residual {min(r.residual for r in results):.4e}; "
          f"{len(converged)} converged at <= {args.tol:g}")

    orbits = []
    for r in converged:
        d = describe(rmap, r.x, scales)
        d["residual"] = r.residual
        d["iterations"] = r.iterations
        d["active_bounds"] = list(r.active_bounds)
        d["x"] = [float(v) for v in r.x]  # full float64; rounding can land on an event surface
        d["classification"] = classify(d, cycle)
        orbits.append(d)
        print(f"    residual {r.residual:.2e}  advance {d['advance_per_cycle_m']:+.4f} m/cyc "
              f"({d['velocity_mps']:+.3f} m/s)  h-range {d['height_range_m']:.4f}  "
              f"flight {d['flight_steps']}/{cycle}  rho {d['spectral_radius']:.4f} "
              f"unstable {d['unstable_count']}  eps {d['eps_verdict']}  -> {d['classification']}")
    # The best *non*-converged terminal point, kept because a near-miss is worth re-examining and
    # was not recoverable from the first version of this script: the muscle T_c 24 cell reached
    # 4.62e-05 on one start of 96, five orders better than any other cell, and the state was lost.
    best = min(results, key=lambda r: r.residual)
    payload_best = {
        "residual": best.residual,
        "outcome": best.outcome.value,
        "iterations": best.iterations,
        "x": [float(v) for v in best.x],
        "eps_verdict": best.terminal_sweep.verdict if best.terminal_sweep else None,
        "eps_slope": best.terminal_sweep.slope if best.terminal_sweep else None,
    }
    if not orbits:
        print("    no fixed point found. Scope: "
              f"{args.starts} starts, sigma {min(SIGMAS)} to {max(SIGMAS)}, seed {args.seed}."
              f" Best terminal point: residual {best.residual:.4e}, "
              f"eps {payload_best['eps_verdict']}.")

    payload = {
        "params": args.params,
        "asset": args.asset,
        "scales_source": args.scales,
        "world": world,
        "world_selection": how,
        "cycle": cycle,
        "starts": args.starts,
        "sigmas": list(SIGMAS),
        "seed": args.seed,
        "tol": args.tol,
        "seconds": seconds,
        "activation_limit_cycle": [float(v) for v in rmap.activation],
        "activation_cycles": rmap.activation_cycles,
        "candidate_start_residual": rmap.scaled_residual(x0, scales),
        "structural_checks": {
            "translation_column_error": checks.translation_column_error,
            "activation_to_mechanics": checks.activation_to_mechanics,
            "activation_radius": checks.activation_radius,
            "ok": checks.ok(),
        },
        "outcomes": dict(histogram),
        "best_residual": min(r.residual for r in results),
        "converged": len(converged),
        "orbits": orbits,
        "best_terminal": payload_best,
        "terminal_eps_verdicts": dict(Counter(
            r.terminal_sweep.verdict for r in results if r.terminal_sweep is not None)),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
