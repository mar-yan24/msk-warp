# Changes

## 2026-03-24: Ant Environment + MyoLeg Scaffolding

### New Environments

**Ant Locomotion (`msk_warp/envs/ant.py`)**
- 4-legged ant locomotion with ground contact, ported from DiffRL
- 37D observation: torso height, quaternion, linear/angular velocity, 8 joint angles, 8 joint velocities (scaled), up-vector z-component, heading alignment, last actions
- 8D action space: hip and ankle motors for 4 legs (gear=200 in MJCF)
- Reward: forward velocity + 0.1 * upright bonus + heading alignment + (height - 0.27)
- Early termination when torso height drops below 0.27
- Stochastic initialization with random position/orientation/joint perturbation
- Config: `configs/ant_shac.yaml` (64 envs, 2000 epochs, [128,64,32] actor)

**MyoLeg Walking Scaffolding (`msk_warp/envs/myoleg_walk.py`)**
- Initial scaffolding for musculoskeletal locomotion using MyoSuite leg models
- Auto-discovers `myolegs.xml` from myosuite installation
- Dynamic observation/action dimensions from model introspection
- Muscle activation remapping: tanh [-1,1] -> activation [0,1]
- Reward: forward velocity + upright bonus + alive bonus - energy cost
- Config: `configs/myoleg_shac.yaml` (16 envs, 1000 epochs, [256,128,64] actor)

### New Assets

- `assets/ant.xml`: MuJoCo ant MJCF with ground plane, contact parameters, actuators with gear=200, keyframe for standing pose, njmax/nconmax sizing for contact buffers

### Quaternion Utilities (`msk_warp/utils/torch_utils.py`)

Added MuJoCo [w,x,y,z] quaternion functions (ported from DiffRL's [x,y,z,w] convention):
- `normalize()`, `quat_mul()`, `quat_conjugate()`, `quat_rotate()`, `quat_rotate_inverse()`, `quat_from_angle_axis()`
- All `@torch.jit.script` for GPU performance

### Gradient Bridge Updates (`msk_warp/bridge.py`)

- Added `_qpos_grad_to_qvel_grad()` to handle free joints where nq != nv (quaternion has 4 components in qpos but 3 angular velocity components in qvel)
- The backward pass now correctly maps qpos gradients through the quaternion integration Jacobian before computing `grad_qacc`
- This enables gradient flow for any model with free joints (ant, humanoid, myoleg, etc.)

### SHAC Algorithm Updates (`msk_warp/algorithms/shac.py`)

- Generalized environment construction: all `cfg.params.env` keys are passed through as kwargs to the environment constructor (no more hardcoded kwargs)
- Updated `ENV_MAP` with `Ant` and `MyoLegWalk` entries
- Changed `env._compute_obs()` call to `env.compute_obs()` for uniform interface across environments

### Base Environment Updates (`msk_warp/envs/base_env.py`)

- Added `njmax` parameter to control MuJoCo Warp per-world constraint buffer size (passed to `make_diff_data`)
- This prevents contact constraint overflow for models with many contact points (ant, humanoid, etc.)

### CartPole Compatibility (`msk_warp/envs/cartpole_swing_up.py`)

- Added `compute_obs(self, qpos, qvel)` instance method wrapper for SHAC interface compatibility

### Key Design Decisions

- **Coordinate system**: z-up (MuJoCo native) throughout, unlike DiffRL which uses y-up
- **Quaternion convention**: [w,x,y,z] (MuJoCo) throughout, unlike DiffRL's [x,y,z,w]
- **Torque scaling**: `action_strength` in config multiplies the tanh'd action before setting MuJoCo ctrl; the MJCF `gear` attribute provides the final scaling to torque. For ant: action_strength=1.0, gear=200 -> max 200 N*m (matching DiffRL)
- **Contact buffers**: `njmax=512` per world (default for ant) prevents constraint truncation that causes simulation instability
- **Actions in obs**: Stored detached (`self.actions = actions.detach()`) to prevent cross-epoch computational graph references
