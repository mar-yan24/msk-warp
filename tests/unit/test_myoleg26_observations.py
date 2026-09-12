"""MyoLeg26 observation kinematics against native MuJoCo, without Warp or a GPU."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from msk_warp import resolve_model_path
from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv, WarpSimStep


@pytest.fixture
def env():
    # Only observation setup is needed. In particular no Warp model/Data exists,
    # so compute_obs cannot accidentally obtain or mutate simulator state.
    result = MyoLeg26WalkEnv.__new__(MyoLeg26WalkEnv)
    result.mjm = mujoco.MjModel.from_xml_path(
        resolve_model_path('assets/myoleg/myoLeg26_BASELINE.xml'),
    )
    result.device = 'cpu'
    result.num_environments = 1
    result.n_joint_q = result.mjm.nq - 6
    result.n_joint_v = result.mjm.nv - 6
    result.actions = torch.zeros((1, result.mjm.nu), dtype=torch.float64)
    result._init_pelvis_kinematics()
    result.up_vec = result.up_vec.double()
    result.heading_vec = result.heading_vec.double()
    return result


def _native_orientation(env, qpos):
    data = mujoco.MjData(env.mjm)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(env.mjm, data)
    bid = env.pelvis_body_id
    rotation = data.xmat[bid].reshape(3, 3)
    return data.xquat[bid].copy(), rotation[2, 1], rotation[0, 0]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('nonzero_reference', [False, True])
def test_pelvis_orientation_matches_native_kinematics(env, dtype, nonzero_reference):
    if nonzero_reference:
        env.mjm.qpos0[3:6] = [0.13, -0.21, 0.34]
        env._init_pelvis_kinematics()
    rng = np.random.default_rng(29)
    poses = np.repeat(env.mjm.key_qpos[0][None], 17, axis=0)
    poses[1:, :6] += rng.uniform(-0.7, 0.7, (16, 6))
    qpos = torch.tensor(poses, dtype=dtype)
    qvel = torch.zeros((len(poses), env.mjm.nv), dtype=dtype)
    env.actions = env.actions.to(dtype).expand(len(poses), -1)
    env.up_vec = env.up_vec.to(dtype).expand(len(poses), -1)
    env.heading_vec = env.heading_vec.to(dtype).expand(len(poses), -1)
    obs = env.compute_obs(qpos, qvel)
    expected = [_native_orientation(env, q) for q in qpos.numpy()]
    atol = 3e-7 if dtype == torch.float32 else 2e-14
    np.testing.assert_allclose(obs[:, 1:5], np.stack([x[0] for x in expected]), atol=atol, rtol=0)
    np.testing.assert_allclose(obs[:, -(env.mjm.nu + 2)], [x[1] for x in expected], atol=atol, rtol=0)
    np.testing.assert_allclose(obs[:, -(env.mjm.nu + 1)], [x[2] for x in expected], atol=atol, rtol=0)
    assert obs.shape == (len(poses), 147)


def test_standing_anatomical_up_and_heading_are_one(env):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    obs = env.compute_obs(qpos, torch.zeros_like(qpos))
    assert obs[0, -(env.mjm.nu + 2)].item() == pytest.approx(1.0, abs=3e-8)
    assert obs[0, -(env.mjm.nu + 1)].item() == pytest.approx(1.0, abs=1e-14)


def test_orientation_reward_gradient_matches_native_fd(env):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    qpos[0, 3:6] = torch.tensor([0.23, -0.31, 0.47], dtype=torch.float64)
    qpos.requires_grad_(True)
    qvel = torch.zeros_like(qpos)

    def reward(q):
        return env._compute_reward(env.compute_obs(q, qvel), env.actions)

    assert torch.autograd.gradcheck(reward, (qpos,), eps=1e-6, atol=1e-7, rtol=1e-6)
    ad = torch.autograd.grad(reward(qpos).sum(), qpos)[0][0, 3:6].numpy()
    fd = []
    for j in range(3, 6):
        plus, minus = qpos.detach().numpy()[0].copy(), qpos.detach().numpy()[0].copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        _, up_p, heading_p = _native_orientation(env, plus)
        _, up_m, heading_m = _native_orientation(env, minus)
        fd.append((0.1 * (up_p - up_m) + heading_p - heading_m) / 2e-6)
    np.testing.assert_allclose(ad, fd, atol=2e-10, rtol=1e-7)
    assert np.all(np.abs(ad) > 1e-3)


def test_compute_obs_uses_supplied_state_without_mutating_it(env):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    qpos[0, 3:6] = torch.tensor([0.2, -0.1, 0.4])
    qvel = torch.zeros_like(qpos)
    before_qpos, before_qvel = qpos.clone(), qvel.clone()
    assert not hasattr(env, 'warp_data')
    env.compute_obs(qpos, qvel)
    assert torch.equal(qpos, before_qpos)
    assert torch.equal(qvel, before_qvel)


def test_step_observations_and_reward_use_post_step_orientation(env, monkeypatch):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    qpos[0, 3:6] = torch.tensor([0.2, -0.1, 0.4])
    qvel = torch.zeros_like(qpos)
    act = torch.zeros((1, env.mjm.na), dtype=torch.float64)
    env.num_environments = 1
    env.num_actions = env.mjm.nu
    env.no_grad = False
    env.action_strength = 1.0
    env.early_termination = True
    env.termination_height = 0.5
    env.episode_length = 1000
    env.reset_buf = torch.zeros(1, dtype=torch.long)
    env.termination_buf = torch.zeros_like(env.reset_buf)
    env.progress_buf = torch.zeros_like(env.reset_buf)
    monkeypatch.setattr(WarpSimStep, 'apply', lambda *_: (qpos, qvel, act))
    obs, reward, *_ = env.step(env.actions, qpos, qvel, act)
    expected = env.compute_obs(qpos, qvel, act)
    torch.testing.assert_close(obs, expected)
    torch.testing.assert_close(reward, env._compute_reward(expected, env.actions))
    torch.testing.assert_close(env.extras['obs_before_reset'], expected)
