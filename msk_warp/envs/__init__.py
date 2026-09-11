from msk_warp.envs.cartpole_swing_up import CartPoleSwingUpEnv
from msk_warp.envs.ant import AntEnv
from msk_warp.envs.hopper import HopperMotorEnv, HopperMuscleEnv, HopperMuscleTrackEnv
from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv

ENV_MAP = {
    'CartPoleSwingUp': CartPoleSwingUpEnv,
    'Ant': AntEnv,
    'HopperMotor': HopperMotorEnv,
    'HopperMuscle': HopperMuscleEnv,
    'HopperMuscleTrack': HopperMuscleTrackEnv,
    'MyoLeg26Walk': MyoLeg26WalkEnv,
}
