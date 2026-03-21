# msk-warp

SHAC (Short Horizon Actor Critic) for musculoskeletal control, powered by MuJoCo Warp for GPU-accelerated differentiable simulation.

This project replaces the dFlex physics backend from [DiffRL](https://github.com/NVlabs/DiffRL) with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), enabling differentiable RL training where gradients backpropagate directly through the physics simulation.

## Current Status: CartPole Swing-Up

CartPole is the first validation target. It has 2 DOFs (cart slider + pole hinge), 1 actuator, and no contacts, making it ideal for verifying the gradient pipeline before scaling to complex musculoskeletal models.

**Training results (160 epochs, 64 parallel worlds):**
- Episode reward improves from -1252 to -388 (69% reduction in loss)
- ~900 environment FPS on an RTX 4060 Laptop GPU
- Gradient verification passes: AD vs float64 finite differences within 0.02% relative error

## Architecture

### Gradient Bridge

The core challenge is bridging Warp's tape-based autodiff with PyTorch's autograd. Each simulation step is wrapped in a `torch.autograd.Function` called `WarpSimStep`.

**Forward pass:** Save pre-step state, set ctrl from PyTorch tensor, run `mjw.step()` for N substeps, return (qpos, qvel) as PyTorch tensors.

**Backward pass:** Uses a hybrid analytical + tape + finite-difference approach:
1. Compute `d(loss)/d(qacc)` analytically from the semi-implicit Euler integration equations
2. Solve `M * grad_qfrc = grad_qacc` using MuJoCo Warp's forward mass matrix solve (`mjw.solve_m`)
3. Compute the **dynamics Jacobian** `∂qacc/∂qpos` and `∂qacc/∂qvel` via finite differences through `mjw.forward` — this captures gravity, Coriolis, and mass-matrix state-dependence
4. Use a Warp tape through `fwd_actuation()` only to get `d(loss)/d(ctrl)`
5. Propagate state gradients backward through substeps using both the Euler chain and the FD dynamics Jacobian

`WarpSimStep` accepts `(ctrl, qpos_in, qvel_in)` as differentiable inputs and returns gradients for all three. This provides two gradient paths across simulation steps: through the **actor network** (obs → policy → ctrl) and through the **physics dynamics** (state → next state). The dynamics path has bounded eigenvalues (≈1), preventing the exponential gradient amplification that occurs when only the actor path is available.

### Observation and Reward

Observations `[x, xdot, sin(theta), cos(theta), theta_dot]` and rewards are computed as PyTorch operations on the tensors returned by `WarpSimStep`. This keeps them naturally in the autograd graph with no special handling needed.

### Multi-World Batching

`num_actors=64` maps directly to `nworld=64` in MuJoCo Warp. All simulation arrays are shaped `(nworld, ...)` and the entire batch runs on GPU in a single kernel launch.

## Project Structure

```
msk-warp/
  assets/
    cartpole.xml              # MJCF model (MuJoCo XML)
  configs/
    cartpole_shac.yaml        # Training hyperparameters
  msk_warp/
    bridge.py                 # WarpSimStep gradient bridge
    envs/
      base_env.py             # Base MuJoCo Warp environment
      cartpole_swing_up.py    # CartPole environment
    algorithms/
      shac.py                 # SHAC algorithm
    networks/
      actor.py                # Stochastic/Deterministic actor MLPs
      critic.py               # Critic MLP
      model_utils.py          # Network initialization helpers
    utils/                    # Running mean/std, dataset, timers, etc.
  scripts/
    train.py                  # Training entry point
    test_gradient.py          # Gradient verification
    visualize.py              # Policy visualization in MuJoCo viewer
```

## Setup

**Requirements:** Python 3.12, CUDA 12+, an NVIDIA GPU

```bash
# Create virtual environment
py -3.12 -m venv .venv
source .venv/Scripts/activate   # Windows/Git Bash
# source .venv/bin/activate     # Linux/Mac

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install warp-lang mujoco tensorboardX pyyaml numpy pillow

# Install mujoco_warp from local source
pip install -e /path/to/mujoco_warp

# Install this package
pip install -e .
```

## Usage

### Train

```bash
python scripts/train.py --cfg configs/cartpole_shac.yaml --logdir logs/cartpole
```

Optional arguments:
- `--seed 42` to set random seed
- `--device cuda:0` to select GPU

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/cartpole/log
```

### Visualize

```bash
# Interactive MuJoCo viewer
python scripts/visualize.py --policy logs/cartpole/best_policy.pt

# Headless with stats only
python scripts/visualize.py --policy logs/cartpole/best_policy.pt --no-render --episodes 10

# Save rendered frames to disk
python scripts/visualize.py --policy logs/cartpole/best_policy.pt --save-frames outputs/frames
```

### Verify Gradients

```bash
python scripts/test_gradient.py
```

Runs two tests:
1. **Single-step:** `loss = sum(qpos_after)`, compares AD gradient vs float64 finite differences
2. **Network-in-loop:** `ctrl = linear(obs)` through WarpSimStep, verifies network parameter gradients are nonzero and directionally correct

## Training Configuration

Key hyperparameters in `configs/cartpole_shac.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_actors | 64 | Parallel simulation worlds |
| episode_length | 240 | Steps per episode (4 seconds at 60 Hz) |
| steps_num | 32 | SHAC horizon (rollout length) |
| max_epochs | 1000 | Training iterations |
| actor_learning_rate | 1e-2 | With linear decay to 1e-5 |
| critic_learning_rate | 1e-3 | With linear decay to 1e-5 |
| gamma | 0.99 | Discount factor |
| action_strength | 20.0 | Scales tanh output to motor force (N) |
| substeps | 4 | Physics substeps per environment step |

## How SHAC Works

SHAC is a differentiable RL algorithm that backpropagates through the physics simulation to compute policy gradients directly, rather than using score-function estimators like PPO.

Each training iteration:
1. Roll out the actor for H=32 steps, accumulating discounted rewards
2. Bootstrap terminal values with a target critic
3. Backpropagate `actor_loss` through the reward/obs computation, through the simulation gradient bridge, all the way to the actor network parameters
4. Update the actor with one Adam step (with gradient clipping)
5. Train the critic on TD-lambda targets for 16 iterations

The critic is trained separately using standard supervised regression (no simulation gradients needed). A target critic with EMA updates provides stable bootstrap values.

## Changes from Reference DiffRL

Key modifications from the [original DiffRL implementation](https://github.com/NVlabs/DiffRL) to work with MuJoCo Warp:

**Gradient bridge (`bridge.py`):**
- Finite-difference dynamics Jacobian (`∂qacc/∂qpos`, `∂qacc/∂qvel`) added to the backward pass. The original DiffRL uses dFlex which provides full dynamics gradients natively; MuJoCo Warp's constraint kernels don't support backward, so FD fills this gap.
- State tensors (`qpos_in`, `qvel_in`) are now differentiable inputs to `WarpSimStep`, enabling gradient flow through the dynamics path across simulation steps.

**State gradient threading (`shac.py`, `cartpole_swing_up.py`):**
- `compute_actor_loss` tracks `qpos`/`qvel` as PyTorch tensors across the rollout, passing them through each `WarpSimStep` call. Resets are handled with gradient-safe masking (multiply by 0 for reset envs, detached reset state added back).
- `env.step` accepts and returns state tensors. Obs is recomputed from the tracked state in the training loop rather than from Warp arrays, preserving the autograd graph for non-reset environments.

**Hyperparameters (`configs/cartpole_shac.yaml`):**
- `ret_rms: false` (reference: False) — return normalization causes an amplification feedback loop with SHAC's differentiable rollout
- `cart_position_penalty: 0.05` (reference: 0.05) — was 0.5 (10x too high), conflicts with swing-up
- `cart_action_penalty: 0.0` (reference: 0.0) — swing-up needs aggressive actions
- `actor_learning_rate: 1e-2` (reference: 1e-2) — was 2e-3 (too slow for SHAC with grad_norm=1.0)
- Network units: `[64, 64]` (reference: [64, 64]) — was [128, 64], overparameterized for cartpole

## References

- [DiffRL / SHAC paper](https://arxiv.org/abs/2204.07137): Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation," ICLR 2022
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): GPU-accelerated differentiable MuJoCo via NVIDIA Warp
- [MuJoCo](https://mujoco.org/): Multi-Joint dynamics with Contact
