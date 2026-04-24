# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

msk-warp is a differentiable RL framework for musculoskeletal control. It bridges MuJoCo Warp (GPU-accelerated differentiable physics via NVIDIA Warp) with PyTorch, enabling SHAC (Short Horizon Actor Critic) to backpropagate gradients through the physics simulation.

## Custom MuJoCo Warp Build

This project depends on a custom MuJoCo Warp build at `C:\Projects\mujoco_warp` (branch `mark/autodifferentiation3`), installed as an editable package. It extends upstream MuJoCo Warp with three autodiff capabilities:

1. **Smooth contact autodifferentiation** (`collision_smooth.py`) — when a Warp tape is active and `d.qpos.requires_grad == True`, discrete collision results are overwritten with smooth, differentiable distance/position/frame values for supported geometry pairs
2. **Newton solver implicit differentiation** (`adjoint.py`) — propagates gradients through the Newton constraint solver via the implicit function theorem. Only supports Newton solver (not CG or PGS)
3. **Differentiable smooth dynamics** (`smooth.py`) — differentiable kinematics, velocity, and acceleration computation through Warp autodiff

Autodiff activates automatically when both conditions are met: a Warp tape is active (`wp.Tape()`) AND `d.qpos.requires_grad == True` (true when using `mjw.make_diff_data()`). No explicit opt-in flag is needed.

**GPU requirement:** CUDA compute capability >= SM 7.0 (Volta/V100, Turing/RTX 20xx, Ampere/RTX 30xx, or newer) for the tile-based Cholesky solver in the adjoint.

### Supported Contact Geometry Pairs

| Geometry Pair     | Status        | Notes                                      |
| ----------------- | ------------- | ------------------------------------------ |
| Plane-Sphere      | Supported     | Smooth signed distance                     |
| Plane-Capsule     | Supported     | 2 contacts at endpoints                    |
| Sphere-Sphere     | Supported     | Smooth normalization at coincident centers  |
| Sphere-Capsule    | Supported     | Clamped closest-point projection           |
| Capsule-Capsule   | Supported     | Blend between parallel/non-parallel paths  |
| Box, Mesh, Convex, Ellipsoid, SDF | Not supported | Zero gradients, no runtime warning |

The ant model uses only sphere, capsule, and plane geometries (all supported). New environments with box or mesh collision geometries will get zero contact gradients through the tape-mode backward path.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Train
python scripts/train.py --cfg configs/cartpole_shac.yaml --logdir logs/cartpole --seed 42 --device cuda:0

# Visualize a trained policy
python scripts/visualize.py --policy logs/cartpole/best_policy.pt --episodes 5

# Run tests
pytest tests/

# Run a single test
pytest tests/test_gradient.py::test_single_step -v
```

Config files live in `msk_warp/configs/` (cartpole_shac.yaml, ant_shac.yaml, myoleg_shac.yaml).

## Architecture

### Gradient Bridge (`bridge.py`)
The core innovation. `WarpSimStep` is a custom `torch.autograd.Function` with a 4-argument `apply(ctrl, qpos_in, qvel_in, env)` signature:
- **Forward**: copies ctrl to Warp, runs substeps, extracts (qpos, qvel) as PyTorch tensors
- **Backward**: three modes selected by env flags:
  1. **Tape-all** (default): single `wp.Tape()` over all substeps with `_vjp_state_kernel` to seed backward. ~2-3x forward cost per substep. Provides full contact dynamics gradients for supported geometry types via the custom MuJoCo Warp build's smooth contact autodiff and Newton solver implicit differentiation
  2. **Tape-per-substep** (`env.tape_per_substep=True`): tapes each substep individually and chains gradients. Lower peak GPU memory, same accuracy as tape-all
  3. **FD Jacobian** (`env.use_fd_jacobian=True`): finite-difference dynamics Jacobian + analytical Euler backward + mass-matrix solve. Costs `nq+nv+2` forward calls per substep. Now optional for supported geometry types (tape-all provides the same gradients). Useful for: debugging/A/B comparison, unsupported geometry types (box/mesh), or situations where Newton solver cannot be used
- **Critical**: backward modifies `warp_data` in-place (restores saved states, re-runs substeps). The caller must save/restore `warp_data.qpos/qvel/time` around `backward()` — see the state save/restore in `shac.py`'s `actor_closure()`. Without this, the next epoch starts from post-step-0 state instead of post-step-31, reducing effective rollout to 1 step per epoch

### Training Loop (`algorithms/shac.py`)
`SHAC.train()` runs epochs of:
1. **Actor rollout** (`compute_actor_loss`): unrolls H=32 differentiable simulation steps, accumulates discounted rewards with critic bootstrap, returns scalar loss. State tensors (qpos, qvel) are explicitly tracked across steps for gradient flow
2. **Backprop**: gradients flow through WarpSimStep → actor network. Two BPTT control mechanisms:
   - `state_bptt` (config, default true): when false, detaches state tensors each step — only single-step ctrl gradients flow, no state-to-state BPTT. Required false for cartpole (BPTT causes gradient norms of 1e6+ through the reward accumulation path)
   - `state_grad_clip` (config, default 0): clips gradient norms on qpos/qvel at step boundaries. Unlike `obs_grad_clip` (which only clips the observation→actor path), this clips ALL BPTT paths including the reward accumulation path (rew_acc → rew → state chain)
   State save/restore wraps `actor_loss.backward()` to prevent warp_data corruption (see bridge section)
3. **Critic training**: 16 mini-batch iterations of MSE on TD-lambda targets, with EMA target critic

Environment construction is generalized: all `cfg.params.env` keys pass through as kwargs to the env constructor. `ENV_MAP` lives in `envs/__init__.py` and maps string names to env classes.

### Environments (`envs/`)
All inherit `MjWarpEnv` which loads MJCF via native MuJoCo, wraps in MuJoCo Warp for GPU batching (nworld = num_envs). Each env implements:
- `step(actions, qpos_in, qvel_in)` — accepts optional state tensors for gradient flow
- `compute_obs(qpos, qvel)` — differentiable observation from state
- `calculateReward()` — reward from current buffers
- Gradient-safe reset masking: detaches state for reset envs to prevent stale graph references

Base env accepts `njmax` parameter for contact constraint buffer sizing (per world). Contact-rich models (ant, humanoid) need this to prevent constraint truncation.

Current envs:
- **CartPole** (validation, 2 DOF) — balance/swing-up, no contacts
- **Ant** (locomotion, 15 DOF) — 4-legged walking with ground contact, 37D obs, 8D action (hip+ankle motors, gear=200), early termination on fall, stochastic init. Config: 512 envs, 1000 epochs
- **MyoLeg** (musculoskeletal, scaffolding) — auto-discovers myosuite model, dynamic obs/action dims from model introspection, muscle activation remapping (tanh → [0,1])
- **MyoLeg26** (musculoskeletal, 14 DOF) — bilateral human gait with 26 Hill-type muscle actuators, pelvis uses slide+hinge joints (nq==nv), terrain config support. Note: `myoLeg26_BASELINE.xml` currently lacks `solver="Newton" jacobian="dense"` — contact gradients will be zero in tape mode until this is added

### Quaternion Utilities (`utils/torch_utils.py`)
MuJoCo [w,x,y,z] quaternion functions (ported from DiffRL's [x,y,z,w] convention): `normalize()`, `quat_mul()`, `quat_conjugate()`, `quat_rotate()`, `quat_rotate_inverse()`, `quat_from_angle_axis()`. All `@torch.jit.script` for GPU performance. Also includes `grad_norm()` for gradient monitoring.

### Networks (`networks/`)
Simple MLPs with orthogonal init (gain=sqrt(2)) + LayerNorm + ELU. Both deterministic and stochastic actor variants use orthogonal initialization. Sizes configured per-env in YAML.

## Session Docs

The `docs/` folder contains markdown summaries generated at the end of each coding session. These document what was done, key decisions made, and open questions. This folder is gitignored — it's a local reference, not part of the codebase. When the user asks for a session summary or end-of-session doc, write it to `docs/`.

## Key Conventions

- **MuJoCo XML for contact gradients**: any MJCF model with contacts must specify `solver="Newton" jacobian="dense"` in the `<option>` element for tape-mode contact gradients to flow. The Newton solver is the only one supporting implicit differentiation; CG emits a runtime warning, PGS may fail silently. Non-contact models (e.g., cartpole with `<flag contact="disable">`) do not need these settings
- **Coordinate system**: z-up (MuJoCo native) throughout, unlike DiffRL which uses y-up
- **Quaternion convention**: `[w, x, y, z]` (MuJoCo), unlike DiffRL's `[x, y, z, w]`
- **Torque scaling**: `action_strength` (config) multiplies tanh'd action before setting ctrl; MJCF `gear` provides final scaling to torque. E.g., ant: action_strength=1.0, gear=200 → max 200 N·m
- All Warp kernel interactions go through `bridge.py` — environments don't call Warp directly for stepping
- Multi-world state tensors have shape `(num_envs, dim)` on the configured device
- Assets (MJCF XMLs) are resolved via `msk_warp.get_asset_path()` or `resolve_model_path()`
- Actions stored in obs are detached (`actions.detach()`) to prevent cross-epoch computational graph references
- Gradient tests compare AD gradients against float64 finite differences from native MuJoCo; tolerance is <10% relative error (relaxed to 50% for contact-rich models like ant)
