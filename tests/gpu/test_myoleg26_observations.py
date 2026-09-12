"""Real forward-only MyoLeg steps must expose the integrated pelvis pose."""

import mujoco
import numpy as np
import pytest
import torch

from msk_warp import backend
from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv


def test_baseline_model_reports_multiccd_margin_incompatibility():
    # Pin the actual current import blocker. This is separate from unsupported
    # mesh contact derivatives and affects even a forward-only environment.
    with pytest.raises(NotImplementedError, match='non-zero margin.*MULTICCD'):
        MyoLeg26WalkEnv(num_envs=1, no_grad=True, stochastic_init=False)


def test_post_step_orientation_matches_native_and_recomputed_observation(monkeypatch):
    # Mesh contacts remain unsupported for AD; exercise the production forward
    # path on a deliberately modified test model with MULTICCD disabled. This
    # validates observations, not baseline contact fidelity or trainability.
    put_model = backend.put_model

    def without_multiccd(model):
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        return put_model(model)

    monkeypatch.setattr(backend, 'put_model', without_multiccd)
    env = MyoLeg26WalkEnv(
        num_envs=2, device='cuda:0', no_grad=True,
        stochastic_init=False, early_termination=False,
    )
    actions = torch.tensor([[0.1] * 26, [-0.1] * 26], device='cuda:0')
    for _ in range(3):
        obs, reward, done, *_ = env.step(actions)
        qpos, qvel, act = env.state_tensors()
        torch.testing.assert_close(obs, env.compute_obs(qpos, qvel, act))
        torch.testing.assert_close(reward, env._compute_reward(obs, actions))
        assert not done.any()
        for world in range(env.num_envs):
            native = mujoco.MjData(env.mjm)
            native.qpos[:] = qpos[world].double().cpu().numpy()
            mujoco.mj_kinematics(env.mjm, native)
            rotation = native.xmat[env.pelvis_body_id].reshape(3, 3)
            np.testing.assert_allclose(
                obs[world, 1:5].cpu().numpy(), native.xquat[env.pelvis_body_id], atol=3e-7,
            )
            np.testing.assert_allclose(
                obs[world, -(env.nu + 2):-(env.nu)].cpu().numpy(),
                [rotation[2, 1], rotation[0, 0]], atol=3e-7,
            )
