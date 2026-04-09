# msk-warp

SHAC (Short Horizon Actor Critic) for musculoskeletal control, powered by MuJoCo Warp for GPU-accelerated differentiable simulation.

This project replaces the dFlex physics backend from [DiffRL](https://github.com/NVlabs/DiffRL) with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), enabling differentiable RL training where gradients backpropagate directly through the physics simulation.

## Current Status

Three environments are implemented, progressing from simple validation to musculoskeletal control:

- **CartPole** (2 DOF, no contacts) -- swing-up balance task. Gradient verification passes: AD vs float64 finite differences within 0.02% relative error. ~900 env FPS on RTX 4060 Laptop GPU. Converges reliably in ~160 epochs.
- **Ant** (15 DOF, contact-rich) -- 4-legged locomotion with ground contact. 37D obs, 8D action (hip+ankle motors), 64 parallel worlds. Full contact dynamics gradients flow via smooth autodiff through the custom MuJoCo Warp build. Currently under active development -- see [Known Issues](#known-issues).
- **MyoLeg26** (14 DOF, musculoskeletal) -- bilateral human gait with 26 Hill-type muscle actuators, pelvis uses slide+hinge joints, terrain config support. Early development.

Active scope is intentionally narrow:
- Core training and visualization entrypoints (`scripts/train.py`, `scripts/visualize.py`)
- Active envs: `CartPoleSwingUp`, `Ant`, `MyoLeg26Walk`
- Lean Ant research surface (branch-core configs, parity diagnostics, convergence harness)

## Architecture

### Gradient Bridge

The core challenge is bridging Warp's tape-based autodiff with PyTorch's autograd. Each simulation step is wrapped in a `torch.autograd.Function` called `WarpSimStep`.

**Forward pass:** Save pre-step state, set ctrl from PyTorch tensor, run `mjw.step()` for N substeps, return (qpos, qvel) as PyTorch tensors.

**Backward pass:** Three modes, selected by env flags:

1. **Tape-all** (default) -- records all substeps under a single `wp.Tape()`, then backpropagates through the full physics pipeline. The custom MuJoCo Warp build provides smooth contact autodifferentiation (differentiable distance/position/frame for supported geometry pairs) and Newton solver implicit differentiation, so gradients flow through contact dynamics. ~2-3x forward cost per substep. Currently broken for multi-substep models due to shared-array gradient accumulation -- see [Known Issues](#known-issues).
2. **Tape-per-substep** (`tape_per_substep: true`) -- tapes each substep individually and chains gradients via norm-based clipping. Lower peak GPU memory, correct gradients. This is the current default for contact-rich environments.
3. **FD Jacobian** (`use_fd_jacobian: true`) -- finite-difference dynamics Jacobian + analytical Euler backward + mass-matrix solve. Costs `nq+nv+2` forward calls per substep. Produces 50-100x larger gradient norms than tape mode. Useful for debugging, A/B comparison, or unsupported geometry types (box/mesh).

`WarpSimStep` accepts `(ctrl, qpos_in, qvel_in)` as differentiable inputs and returns gradients for all three. This provides two gradient paths across simulation steps: through the **actor network** (obs -> policy -> ctrl) and through the **physics dynamics** (state -> next state).

### Observation and Reward

Each environment computes observations and rewards as PyTorch operations on the (qpos, qvel) tensors returned by `WarpSimStep`. This keeps them naturally in the autograd graph with no special handling needed. Observation dimensions are environment-specific (e.g., CartPole: 5D, Ant: 37D including height, quaternion, velocities, joint angles, and heading alignment).

The Ant environment supports configurable reward weights (`forward_vel_weight`, `heading_weight`, `up_weight`, `height_weight`) for reward shaping experiments.

### Multi-World Batching

`num_actors` maps directly to `nworld` in MuJoCo Warp. All simulation arrays are shaped `(nworld, ...)` and the entire batch runs on GPU in a single kernel launch.

### BPTT and Gradient Control

SHAC uses 32-step backpropagation through time (BPTT) to compute policy gradients. Several mechanisms control gradient flow:

- **`state_bptt`** (config, default true) -- when false, detaches state tensors each step so only single-step ctrl gradients flow. Required false for cartpole (BPTT causes gradient norms of 1e6+).
- **`state_grad_clip`** (config, default 0) -- clips gradient norms on qpos/qvel at step boundaries. Unlike `obs_grad_clip`, this clips all BPTT paths including the reward accumulation path.
- **Substep gradient clipping** (bridge.py) -- norm-based per-environment clipping within the FD backward pass to prevent exponential blowup across substeps while preserving gradient direction.

## Project Structure

```
msk-warp/
  msk_warp/
    bridge.py                   # WarpSimStep gradient bridge (3 backward modes)
    assets/
      cartpole.xml              # CartPole MJCF (no contacts)
      ant.xml                   # Ant MJCF (solver=Newton, jacobian=dense)
      ant_soft.xml              # Ant with softer contacts (solref=[0.05,1])
      myoleg/                   # MyoLeg26 model, meshes, terrain configs
    envs/
      base_env.py               # MjWarpEnv base class
      cartpole_swing_up.py      # CartPole swing-up (2 DOF)
      ant.py                    # Ant locomotion (15 DOF, contacts)
      myoleg26_walk.py          # MyoLeg26 bilateral gait (26 muscles, 14 DOF)
    algorithms/
      shac.py                   # SHAC training algorithm
    networks/
      actor.py                  # Stochastic/Deterministic actor MLPs
      critic.py                 # Critic MLP
      model_utils.py            # Network initialization helpers
    configs/
      cartpole_shac.yaml        # CartPole training config
      ant_shac.yaml             # Ant training config (tape-per-substep)
      ant_shac_fd.yaml          # Ant with FD Jacobian backward
      ant_shac_nobptt.yaml      # Ant without state BPTT
      myoleg26_shac.yaml        # MyoLeg26 training config
      experiments/              # Active Ant branch-core research configs
    utils/
      torch_utils.py            # Quaternion ops, grad_norm (@torch.jit.script)
      running_mean_std.py       # Observation normalization
      dataset.py                # Replay buffer
      average_meter.py          # Metric tracking
      time_report.py            # Training timing
  scripts/
    train.py                    # Training entry point
    visualize.py                # Policy visualization in MuJoCo viewer
    visualize_progression.py    # Training progression grid
  tests/
    test_gradient.py            # Gradient verification suite (6 test variants)
    test_ant_gradient.py        # Ant-specific gradient tests
    test_grad_chain.py          # Backprop chain validation
  docs/                         # Canonical tracked docs (scope, playbook, test matrix)
  archive/                      # Local-only archive for legacy/iteration artifacts
```

## Setup

**Requirements:** Python >= 3.11, CUDA 12+, NVIDIA GPU with compute capability >= SM 7.0 (Volta/RTX 20xx or newer) for contact gradient kernels

```bash
# Create virtual environment
py -3.12 -m venv .venv
source .venv/Scripts/activate   # Windows/Git Bash
# source .venv/bin/activate     # Linux/Mac

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install warp-lang mujoco tensorboardX pyyaml numpy pillow

# Install custom mujoco_warp build (branch mark/autodifferentiation3)
# This is a custom build with smooth contact autodiff, not upstream google-deepmind/mujoco_warp
pip install -e /path/to/mujoco_warp

# Install this package
pip install -e .
```

## Usage

### Train

```bash
# CartPole (validation, ~160 epochs to converge)
python scripts/train.py --cfg configs/cartpole_shac.yaml --logdir logs/cartpole

# Ant (locomotion, currently requires ~2000+ epochs)
python scripts/train.py --cfg configs/ant_shac.yaml --logdir logs/ant

# Ant with FD Jacobian backward (larger gradients, slower per step)
python scripts/train.py --cfg configs/ant_shac_fd.yaml --logdir logs/ant_fd

# MyoLeg26 (musculoskeletal gait)
python scripts/train.py --cfg configs/myoleg26_shac.yaml --logdir logs/myoleg26
```

Optional arguments:
- `--seed 42` to set random seed
- `--device cuda:0` to select GPU

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/ant/log
```

### Visualize

```bash
# Interactive MuJoCo viewer
python scripts/visualize.py --cfg logs/cartpole/cfg.yaml --policy logs/cartpole/best_policy.pt

# Headless with stats only
python scripts/visualize.py --cfg logs/cartpole/cfg.yaml --policy logs/cartpole/best_policy.pt --no-render --episodes 10

# Save rendered frames to disk
python scripts/visualize.py --cfg logs/cartpole/cfg.yaml --policy logs/cartpole/best_policy.pt --save-frames outputs/frames
```

### Verify Gradients

```bash
pytest tests/test_gradient.py -v
```

Runs 6 test variants covering CartPole and Ant: single-step AD vs float64 FD, network-in-loop gradient checks, tape-vs-FD comparison, and tape-per-substep vs tape-all consistency. Tolerance is <10% relative error (relaxed to 50% for contact-rich models).

## Training Configuration

Key hyperparameters (see `msk_warp/configs/` for full YAML files):

**CartPole** (`cartpole_shac.yaml`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_actors | 64 | Parallel simulation worlds |
| episode_length | 240 | Steps per episode (4 seconds at 60 Hz) |
| steps_num | 32 | SHAC horizon (rollout length) |
| max_epochs | 1000 | Training iterations |
| actor_learning_rate | 1e-2 | With linear decay |
| action_strength | 20.0 | Scales tanh output to motor force (N) |
| substeps | 4 | Physics substeps per environment step |

**Ant** (`ant_shac.yaml`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_actors | 64 | Parallel simulation worlds |
| episode_length | 1000 | Steps per episode |
| steps_num | 32 | SHAC horizon (rollout length) |
| max_epochs | 2000 | Training iterations |
| actor_learning_rate | 2e-3 | With linear decay |
| action_strength | 1.0 | MJCF gear=200 provides final scaling |
| substeps | 16 | More substeps for contact stability |
| tape_per_substep | true | Per-substep taping with gradient chaining |
| state_bptt | true | BPTT through state across steps |
| state_grad_clip | 5.0 | Norm clip on state gradients at step boundaries |

## How SHAC Works

SHAC is a differentiable RL algorithm that backpropagates through the physics simulation to compute policy gradients directly, rather than using score-function estimators like PPO.

Each training iteration:
1. Roll out the actor for H=32 steps, accumulating discounted rewards
2. Bootstrap terminal values with a target critic
3. Backpropagate `actor_loss` through the reward/obs computation, through the simulation gradient bridge, all the way to the actor network parameters
4. Update the actor with one Adam step (with gradient clipping)
5. Train the critic on TD-lambda targets for 16 iterations

The critic is trained separately using standard supervised regression (no simulation gradients needed). A target critic with EMA updates provides stable bootstrap values.

## Known Issues

### Ant training plateau (hard-contact gradient attenuation)

Ant SHAC training plateaus at approximately -1300 episode loss. The ant survives full 1000-step episodes but stands still, collecting only the baseline standing reward without developing forward locomotion. DiffRL's reference implementation trains the same task to 3000-5000 reward within 2000 epochs.

**Root cause:** MuJoCo's Newton constraint solver produces mathematically exact but very small contact gradients. The implicit differentiation backward pass divides gradients by the contact stiffness (compliance matrix), yielding d(reward)/d(ctrl) of roughly 0.004 per environment -- near the float32 noise floor. In contrast, DiffRL's dflex backend uses penalty-based soft contacts (ke=40,000) that amplify gradients by the spring constant, producing per-step gradients 100x larger with well-conditioned Jacobians.

BPTT across 32 steps amplifies gradients ~100,000x (from ~0.001 to ~30-300 actor grad norm), which is enough to update the policy but too noisy through hard-contact dynamics to consistently push toward locomotion.

**Experiments run:**

| Variant | Result |
|---------|--------|
| Baseline (tape-per-substep, BPTT) | Slow improvement, -1388 at epoch 125 |
| state_bptt=false | Actor grads ~0.001, no learning |
| 5x forward_vel reward weight | Plateaued at -1346 (standing reward unaffected) |
| Pure velocity reward (other weights=0) | Correct gradient direction, but 3mm/s after 50 epochs |
| High LR (1e-2, grad_norm=5) | Faster start, earlier plateau at -1200 |
| FD Jacobian backward | 50-100x larger gradients, slightly better early progress |
| Softer contacts (solref=[0.05,1]) | Smaller gradients than hard contacts (opposite of dflex) |

**Potential directions:**
- Curriculum learning: start with short horizon, increase as policy improves
- Hybrid approach: tape for state BPTT + FD for ctrl gradients within each step
- Test gradient quality at dynamic states (mid-gait) rather than static standing
- Damped training schedule (already partially implemented)

### Tape-all mode broken for multi-substep models

When multiple substeps are recorded under a single `wp.Tape()`, all substeps write to the same MuJoCo Warp `d.*` arrays. During `tape.backward()`, adjoint kernels for substep k read `.grad` arrays that contain accumulated contributions from later substeps, causing exponential gradient amplification (250,000x for ant with 16 substeps). The workaround is `tape_per_substep: true`, which tapes each substep individually. Fixing tape-all requires cloning intermediate arrays (qacc, qfrc_smooth, qfrc_actuator, etc.) per substep in the MuJoCo Warp source.

### MyoLeg26 missing Newton solver config

`myoLeg26_BASELINE.xml` currently lacks `solver="Newton" jacobian="dense"` in its `<option>` element. Contact gradients will be zero in tape mode until this is added. The Newton solver with dense Jacobian is required for implicit differentiation through the constraint solver.

### Unsupported contact geometry types

The custom MuJoCo Warp build only supports smooth contact autodifferentiation for a subset of geometry pairs:

| Geometry Pair | Status |
|---------------|--------|
| Plane-Sphere, Plane-Capsule | Supported |
| Sphere-Sphere, Sphere-Capsule | Supported |
| Capsule-Capsule | Supported |
| Box, Mesh, Convex, Ellipsoid, SDF | Not supported (zero gradients) |

The ant model uses only supported geometry types. New environments with box or mesh collisions will get zero contact gradients through the tape-mode backward path. The FD Jacobian backward mode can be used as a fallback for these cases.

## Changes from Reference DiffRL

Key modifications from the [original DiffRL implementation](https://github.com/NVlabs/DiffRL) to work with MuJoCo Warp:

**Gradient bridge (`bridge.py`):**
- The custom MuJoCo Warp build provides full dynamics gradients through smooth contact autodifferentiation and Newton solver implicit differentiation, analogous to what dFlex provided natively in the original DiffRL. Three backward modes are available (tape-all, tape-per-substep, FD Jacobian), selectable via config flags.
- State tensors (`qpos_in`, `qvel_in`) are now differentiable inputs to `WarpSimStep`, enabling gradient flow through the dynamics path across simulation steps.
- FD backward uses norm-based gradient clipping (per environment) to preserve gradient direction while bounding magnitude across substeps, replacing the original per-element clamping.

**State gradient threading (`shac.py`, `cartpole_swing_up.py`):**
- `compute_actor_loss` tracks `qpos`/`qvel` as PyTorch tensors across the rollout, passing them through each `WarpSimStep` call. Resets are handled with gradient-safe masking (multiply by 0 for reset envs, detached reset state added back).
- `env.step` accepts and returns state tensors. Obs is recomputed from the tracked state in the training loop rather than from Warp arrays, preserving the autograd graph for non-reset environments.

**Hyperparameters (CartPole, `configs/cartpole_shac.yaml`):**
- `ret_rms: false` (reference: False) -- return normalization causes an amplification feedback loop with SHAC's differentiable rollout
- `cart_position_penalty: 0.05` (reference: 0.05) -- was 0.5 (10x too high), conflicts with swing-up
- `cart_action_penalty: 0.0` (reference: 0.0) -- swing-up needs aggressive actions
- `actor_learning_rate: 1e-2` (reference: 1e-2) -- was 2e-3 (too slow for SHAC with grad_norm=1.0)
- Network units: `[64, 64]` (reference: [64, 64]) -- was [128, 64], overparameterized for cartpole

## References

- [DiffRL / SHAC paper](https://arxiv.org/abs/2204.07137): Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation," ICLR 2022
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): GPU-accelerated differentiable MuJoCo via NVIDIA Warp
- Custom MuJoCo Warp build (branch `mark/autodifferentiation3`): adds smooth contact autodiff, Newton solver implicit differentiation, and differentiable smooth dynamics kernels
- [MuJoCo](https://mujoco.org/): Multi-Joint dynamics with Contact

## License

Apache 2.0
