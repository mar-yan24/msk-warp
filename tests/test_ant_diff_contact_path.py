"""Regression tests for the differentiable ant contact path."""

import warnings

warnings.filterwarnings("ignore")

import mujoco
import torch
import warp as wp
import mujoco_warp as mjw

wp.init()

from msk_warp.bridge import WarpSimStep
from msk_warp.envs.ant import AntEnv


def _make_env():
    return AntEnv(
        num_envs=1,
        device="cuda:0",
        stochastic_init=False,
        substeps=4,
        model_path="assets/ant_soft.xml",
        smooth_adjoint=True,
    )


def _assert_floor_identity_and_no_contacts(env):
    floor_id = mujoco.mj_name2id(env.mjm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_xmat = env.warp_data.geom_xmat.numpy()[0, floor_id]
    expected = torch.eye(3, device="cpu", dtype=torch.float32).numpy()
    assert abs(floor_xmat - expected).max() < 1e-6
    assert int(env.warp_data.nacon.numpy()[0]) == 0


def test_ant_env_diff_path_preserves_static_floor_geom():
    env = _make_env()

    # Reset writes qpos/qvel directly, so refresh derived fields before checking.
    mjw.forward(env.warp_model, env.warp_data)
    wp.synchronize()
    _assert_floor_identity_and_no_contacts(env)

    ctrl = torch.zeros((env.num_envs, env.num_actions), device="cuda:0", dtype=torch.float32)
    qpos0 = wp.to_torch(env.warp_data.qpos).detach().clone()
    qvel0 = wp.to_torch(env.warp_data.qvel).detach().clone()

    with torch.no_grad():
        WarpSimStep.apply(ctrl, qpos0, qvel0, env)

    _assert_floor_identity_and_no_contacts(env)
