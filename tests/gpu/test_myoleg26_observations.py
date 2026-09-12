"""Real forward-only MyoLeg steps must expose the integrated pelvis pose."""

import mujoco
import numpy as np
import pytest
import torch
import warp as wp
import mujoco_warp as mjw

from msk_warp import backend
from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv
from msk_warp.models.myoleg26 import FOOT_BODIES, GROUND_NAME


def test_baseline_model_reports_multiccd_margin_incompatibility():
    # Pin the actual current import blocker. This is separate from unsupported
    # mesh contact derivatives and affects even a forward-only environment.
    with pytest.raises(NotImplementedError, match='non-zero margin.*MULTICCD'):
        MyoLeg26WalkEnv(num_envs=1, no_grad=True, stochastic_init=False,
                        model_path='assets/myoleg/myoLeg26_BASELINE.xml', model_contract='legacy')


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
        model_path='assets/myoleg/myoLeg26_BASELINE.xml', model_contract='legacy',
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


def test_official_forward_observations_and_autoreset_match_native():
    env = MyoLeg26WalkEnv(num_envs=2, no_grad=True, stochastic_init=False,
                         early_termination=False, episode_length=2)
    assert env.num_obs == 145
    assert backend.unsupported_pairs(env.warp_model) == []
    action = torch.full((2, 26), -.4, device=env.device)
    for step in range(3):
        obs, reward, done, extras, *_ = env.step(action)
        qpos, qvel, act = env.state_tensors()
        torch.testing.assert_close(obs, env.compute_obs(qpos, qvel, act))
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        if step == 1:
            assert done.all() and extras['truncated'].all()
            assert not extras['terminated'].any()
            assert (obs[:, -26:] == -1).all()
            assert (obs[:, -54:-28] == 0).all()
            assert (extras['obs_before_reset'][:, -54:-28] > 0).all()
            assert not torch.equal(obs, extras['obs_before_reset'])
        else:
            assert not done.any()
        for world in range(env.num_envs):
            native = mujoco.MjData(env.mjm)
            native.qpos[:] = qpos[world].double().cpu().numpy()
            native.qvel[:] = qvel[world].double().cpu().numpy()
            mujoco.mj_forward(env.mjm, native)
            jacp, jacr = np.zeros((3, env.nv)), np.zeros((3, env.nv))
            mujoco.mj_jacBody(env.mjm, native, jacp, jacr, env.pelvis_body_id)
            np.testing.assert_allclose(obs[world, 0].cpu(), native.xpos[env.pelvis_body_id, 2], atol=3e-7)
            np.testing.assert_allclose(obs[world, 1:5].cpu(), native.xquat[env.pelvis_body_id], atol=3e-7)
            np.testing.assert_allclose(obs[world, 5:8].cpu(), jacp @ native.qvel, atol=3e-6)
            np.testing.assert_allclose(obs[world, 8:11].cpu(), jacr @ native.qvel, atol=3e-6)


def test_official_nonfoot_contact_detection_matches_native_and_clears_stale_slots():
    env = MyoLeg26WalkEnv(num_envs=2, no_grad=True, stochastic_init=False)
    for fallen in (True, False):
        env.reset()
        if fallen:
            wp.to_torch(env.warp_data.qpos)[1, 2] = .2
        mjw.forward(env.warp_model, env.warp_data)
        wp.synchronize()
        actual = env._nonfoot_ground_contacts().cpu().numpy()
        expected = []
        qpos, qvel, act = env.state_tensors()
        for world in range(env.num_envs):
            native = mujoco.MjData(env.mjm)
            native.qpos[:] = qpos[world].cpu().numpy()
            native.qvel[:] = qvel[world].cpu().numpy()
            native.act[:] = act[world].cpu().numpy()
            mujoco.mj_forward(env.mjm, native)
            ground = env.mjm.geom(GROUND_NAME).id
            failed = False
            for contact in native.contact:
                if contact.dist >= contact.includemargin:
                    continue
                pair = (contact.geom1, contact.geom2)
                if ground not in pair:
                    continue
                body = env.mjm.body(env.mjm.geom_bodyid[pair[1] if pair[0] == ground else pair[0]])
                failed |= body.name not in FOOT_BODIES
            expected.append(failed)
        np.testing.assert_array_equal(actual, expected)
        assert actual.tolist() == [False, fallen]
