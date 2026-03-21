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

**Backward pass:** Uses a hybrid analytical + tape approach:
1. Compute `d(loss)/d(qacc)` analytically from the semi-implicit Euler integration equations
2. Solve `M * grad_qfrc = grad_qacc` using MuJoCo Warp's forward mass matrix solve (`mjw.solve_m`)
3. Use a Warp tape through `fwd_actuation()` only to get `d(loss)/d(ctrl)`
4. Propagate state gradients backward through substeps for multi-step chains

This design avoids a gradient accumulation issue in Warp's tape backward through the Euler integrator and mass matrix factorization, while still providing correct gradients verified against finite differences.

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
| max_epochs | 500 | Training iterations |
| actor_learning_rate | 1e-2 | With linear decay to 1e-5 |
| critic_learning_rate | 1e-3 | With linear decay to 1e-5 |
| gamma | 0.99 | Discount factor |
| action_strength | 200.0 | Scales tanh output to motor force (N) |
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

## References

- [DiffRL / SHAC paper](https://arxiv.org/abs/2204.07137): Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation," ICLR 2022
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): GPU-accelerated differentiable MuJoCo via NVIDIA Warp
- [MuJoCo](https://mujoco.org/): Multi-Joint dynamics with Contact
