from msk_warp.envs.cartpole_swing_up import CartPoleSwingUpEnv
from msk_warp.envs.ant import AntEnv
from msk_warp.envs.myoleg_walk import MyoLegWalkEnv

ENV_MAP = {
    'CartPoleSwingUp': CartPoleSwingUpEnv,
    'Ant': AntEnv,
    'MyoLegWalk': MyoLegWalkEnv,
}
