"""MyoLeg26 observation kinematics against native MuJoCo, without Warp or a GPU."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from msk_warp import resolve_model_path
from msk_warp.envs import myoleg26_walk
from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv, WarpSimStep


@pytest.fixture
def env():
    # Only observation setup is needed. In particular no Warp model/Data exists,
    # so compute_obs cannot accidentally obtain or mutate simulator state.
    model = mujoco.MjModel.from_xml_path(
        resolve_model_path('assets/myoleg/myoLeg26_BASELINE.xml'),
    )
    return _observation_env(model, 'legacy')


def _observation_env(model, contract, num_envs=1):
    result = MyoLeg26WalkEnv.__new__(MyoLeg26WalkEnv)
    result.mjm = model
    result.model_contract = contract
    result.device = 'cpu'
    result.num_environments = num_envs
    result.nq, result.nv, result.na = result.mjm.nq, result.mjm.nv, result.mjm.na
    result.actions = torch.zeros((num_envs, result.mjm.nu), dtype=torch.float64)
    result._init_pelvis_kinematics()
    result.up_vec = result.up_vec.double()
    result.heading_vec = result.heading_vec.double()
    return result


def _zero_act(env, qpos):
    return qpos.new_zeros((qpos.shape[0], env.mjm.na))


def _act_slice(env):
    start = 11 + env.n_joint_q + env.n_joint_v
    return slice(start, start + env.mjm.na)


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
    obs = env.compute_obs(qpos, qvel, _zero_act(env, qpos))
    expected = [_native_orientation(env, q) for q in qpos.numpy()]
    atol = 3e-7 if dtype == torch.float32 else 2e-14
    np.testing.assert_allclose(obs[:, 1:5], np.stack([x[0] for x in expected]), atol=atol, rtol=0)
    np.testing.assert_allclose(obs[:, -(env.mjm.nu + 2)], [x[1] for x in expected], atol=atol, rtol=0)
    np.testing.assert_allclose(obs[:, -(env.mjm.nu + 1)], [x[2] for x in expected], atol=atol, rtol=0)
    assert obs.shape == (len(poses), 13 + env.n_joint_q + env.n_joint_v + env.mjm.na + env.mjm.nu)


def test_standing_anatomical_up_and_heading_are_one(env):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    obs = env.compute_obs(qpos, torch.zeros_like(qpos), _zero_act(env, qpos))
    assert obs[0, -(env.mjm.nu + 2)].item() == pytest.approx(1.0, abs=3e-8)
    assert obs[0, -(env.mjm.nu + 1)].item() == pytest.approx(1.0, abs=1e-14)


def test_orientation_reward_gradient_matches_native_fd(env):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    qpos[0, 3:6] = torch.tensor([0.23, -0.31, 0.47], dtype=torch.float64)
    qpos.requires_grad_(True)
    qvel = torch.zeros_like(qpos)

    def reward(q):
        return env._compute_reward(env.compute_obs(q, qvel, _zero_act(env, q)), env.actions)

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
    act = _zero_act(env, qpos)
    before_qpos, before_qvel = qpos.clone(), qvel.clone()
    before_act = act.clone()
    assert not hasattr(env, 'warp_data')
    env.compute_obs(qpos, qvel, act)
    assert torch.equal(qpos, before_qpos)
    assert torch.equal(qvel, before_qvel)
    assert torch.equal(act, before_act)


def test_step_observations_and_reward_use_post_step_orientation(env, monkeypatch):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    qpos[0, 3:6] = torch.tensor([0.2, -0.1, 0.4])
    qvel = torch.zeros_like(qpos)
    act_in = _zero_act(env, qpos)
    act = torch.full_like(act_in, 0.37, requires_grad=True)
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
    obs, reward, *_ = env.step(env.actions, qpos, qvel, act_in)
    expected = env.compute_obs(qpos, qvel, act)
    torch.testing.assert_close(obs, expected)
    torch.testing.assert_close(reward, env._compute_reward(expected, env.actions))
    torch.testing.assert_close(env.extras['obs_before_reset'], expected)
    torch.testing.assert_close(obs[:, _act_slice(env)], act)
    grad = torch.autograd.grad(obs[:, _act_slice(env)].sum(), act)[0]
    torch.testing.assert_close(grad, torch.ones_like(act))


def test_activation_distinguishes_otherwise_identical_observations(env):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    qvel = torch.zeros_like(qpos)
    low = _zero_act(env, qpos)
    high = torch.linspace(0.1, 0.9, env.mjm.na, dtype=qpos.dtype)[None].requires_grad_(True)
    obs_low = env.compute_obs(qpos, qvel, low)
    obs_high = env.compute_obs(qpos, qvel, high)
    block = _act_slice(env)
    torch.testing.assert_close(obs_high[:, block] - obs_low[:, block], high)
    torch.testing.assert_close(obs_high[:, :block.start], obs_low[:, :block.start])
    torch.testing.assert_close(obs_high[:, block.stop:], obs_low[:, block.stop:])
    weights = torch.arange(1, env.mjm.na + 1, dtype=qpos.dtype)[None]
    grad = torch.autograd.grad((obs_high[:, block] * weights).sum(), high)[0]
    torch.testing.assert_close(grad, weights)
    # Observation-only activation does not change the currently declared reward.
    torch.testing.assert_close(env._compute_reward(obs_high, env.actions), env._compute_reward(obs_low, env.actions))


def test_observation_requires_complete_correctly_shaped_activation_state(env):
    qpos = torch.tensor(env.mjm.key_qpos[0][None], dtype=torch.float64)
    with pytest.raises(ValueError, match='explicit muscle activation state'):
        env.compute_obs(qpos, torch.zeros_like(qpos))
    with pytest.raises(ValueError, match='Expected act shape'):
        env.compute_obs(qpos, torch.zeros_like(qpos), qpos.new_zeros((1, env.mjm.na - 1)))


@pytest.mark.parametrize('nq,nv,na,nu', [(60, 60, 26, 26), (47, 46, 26, 26), (8, 8, 2, 3)])
def test_observation_block_sizes_follow_model_dimensions(nq, nv, na, nu):
    qpos = torch.zeros((2, nq), dtype=torch.float64)
    qvel = torch.zeros((2, nv), dtype=torch.float64)
    act = torch.full((2, na), 0.41, dtype=torch.float64)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).expand(2, -1)
    actions = torch.full((2, nu), 0.17, dtype=torch.float64)
    up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64).expand(2, -1)
    heading = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(2, -1)
    root_nq = 7 if nq != nv else 6
    pos = torch.zeros((2, 3), dtype=torch.float64)
    vel = torch.zeros_like(pos)
    obs = MyoLeg26WalkEnv._compute_obs(qpos, qvel, act, pos, quat, vel, vel, actions, up, heading, root_nq)
    assert obs.shape == (2, 13 + nq - root_nq + nv - 6 + na + nu)
    torch.testing.assert_close(obs[:, -(na + nu + 2):-(nu + 2)], act)
    torch.testing.assert_close(obs[:, -nu:], actions)


def _mock_warp_state(env, monkeypatch):
    """Exercise state plumbing with CPU tensors, without creating a Warp context."""
    n = env.num_envs
    start = env.mjm.key_qpos[0] if env.mjm.nkey else env.mjm.qpos0
    env.warp_data = SimpleNamespace(
        qpos=torch.tensor(start[None], dtype=torch.float64).repeat(n, 1),
        qvel=torch.zeros((n, env.mjm.nv), dtype=torch.float64),
        act=torch.full((n, env.mjm.na), 0.63, dtype=torch.float64),
        ctrl=torch.zeros((n, env.mjm.nu), dtype=torch.float64),
        qacc_warmstart=torch.ones((n, env.mjm.nv), dtype=torch.float64),
        time=torch.ones(n, dtype=torch.float64),
    )
    monkeypatch.setattr(myoleg26_walk.wp, 'to_torch', lambda array: array)
    monkeypatch.setattr(myoleg26_walk.wp, 'from_torch', lambda tensor: tensor)
    monkeypatch.setattr(myoleg26_walk.wp, 'copy', lambda dest, src: dest.copy_(src))
    monkeypatch.setattr(myoleg26_walk.wp, 'synchronize', lambda: None)
    env.num_actions = env.mjm.nu
    env.no_grad = True
    env.action_strength = 1.0
    env.early_termination = True
    env.termination_height = 0.5
    env.episode_length = 1000
    env.substeps = 1
    env.reset_buf = torch.zeros(n, dtype=torch.long)
    env.termination_buf = torch.zeros_like(env.reset_buf)
    env.progress_buf = torch.zeros_like(env.reset_buf)
    env.extras = {}


def test_no_grad_step_observes_current_simulator_activation(env, monkeypatch):
    _mock_warp_state(env, monkeypatch)
    env.warp_model = object()
    monkeypatch.setattr(myoleg26_walk.mjw, 'step', lambda model, data: data.act.fill_(0.78))
    obs = env.step(env.actions)[0]
    torch.testing.assert_close(obs[:, _act_slice(env)], torch.full_like(env.warp_data.act, 0.78))


def test_reset_and_trajectory_initialization_observe_current_activation(env, monkeypatch):
    _mock_warp_state(env, monkeypatch)
    env.stochastic_init = False
    env.start_qpos = env.warp_data.qpos.clone()
    env.start_qvel = env.warp_data.qvel.clone()
    obs = env.reset()
    torch.testing.assert_close(obs[:, _act_slice(env)], torch.zeros_like(env.warp_data.act))
    assert torch.all(env.warp_data.ctrl == 0)
    assert torch.all(env.warp_data.qacc_warmstart == 0)
    assert torch.all(env.warp_data.time == 0)
    assert torch.all(env.actions == -1)
    env.warp_data.act.fill_(0.42)
    monkeypatch.setattr(env, 'clear_grad', lambda: None)
    obs = env.initialize_trajectory()
    torch.testing.assert_close(obs[:, _act_slice(env)], torch.full_like(env.warp_data.act, 0.42))


FREE_ROOT_XML = '''
<mujoco>
  <worldbody>
    <body name="Full Body" pos="0 0 1">
      <freejoint name="root"/><geom type="sphere" size="0.1"/>
      <body name="attachment" pos="0.1 -0.2 0.3" quat="0.988771 0 0.149438 0">
        <body name="pelvis" pos="-0.05 0.04 0.02" quat="0.5 0.5 -0.5 -0.5">
          <geom type="sphere" size="0.1"/>
          <body pos="0 0 -0.2"><joint name="leg" axis="0 1 0"/>
            <geom type="capsule" size="0.02 0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
'''


@pytest.fixture
def free_env():
    return _observation_env(mujoco.MjModel.from_xml_string(FREE_ROOT_XML), 'official')


@pytest.fixture
def official_env():
    model = mujoco.MjModel.from_xml_path(resolve_model_path('assets/myoleg26/flat_boxes.xml'))
    return _observation_env(model, 'official')


def _assert_pelvis_state_matches_native(env, count=16):
    rng = np.random.default_rng(44)
    poses = np.repeat(env.mjm.qpos0[None], count, axis=0)
    poses[:, :3] += rng.uniform(-0.4, 0.4, (count, 3))
    poses[:, 3:7] = rng.normal(size=(count, 4))
    velocities = rng.uniform(-2.0, 2.0, (count, env.mjm.nv))
    qpos = torch.tensor(poses, dtype=torch.float64)
    qvel = torch.tensor(velocities, dtype=torch.float64)
    position, quat, linear, angular = env._compute_pelvis_state(qpos, qvel)
    for i in range(count):
        data = mujoco.MjData(env.mjm)
        data.qpos[:] = poses[i]
        data.qvel[:] = velocities[i]
        mujoco.mj_normalizeQuat(env.mjm, data.qpos)
        mujoco.mj_kinematics(env.mjm, data)
        mujoco.mj_comPos(env.mjm, data)
        jp, jr = np.zeros((3, env.mjm.nv)), np.zeros((3, env.mjm.nv))
        mujoco.mj_jacBody(env.mjm, data, jp, jr, env.pelvis_body_id)
        np.testing.assert_allclose(position[i], data.xpos[env.pelvis_body_id], atol=2e-14)
        np.testing.assert_allclose(quat[i], data.xquat[env.pelvis_body_id], atol=2e-14)
        np.testing.assert_allclose(linear[i], jp @ velocities[i], atol=2e-14)
        np.testing.assert_allclose(angular[i], jr @ velocities[i], atol=2e-14)
    torch.testing.assert_close(quat.norm(dim=-1), torch.ones(count, dtype=qpos.dtype))


def test_free_root_pose_and_world_velocities_match_native_with_fixed_offsets(free_env):
    _assert_pelvis_state_matches_native(free_env)


def test_official_model_pose_velocity_and_observation_dimensions(official_env):
    env = official_env
    _assert_pelvis_state_matches_native(env)
    assert (env.mjm.nq, env.mjm.nv, env.mjm.na, env.mjm.nu) == (47, 46, 26, 26)
    k = mujoco.mj_name2id(env.mjm, mujoco.mjtObj.mjOBJ_KEY, 'stand')
    qpos = torch.tensor(env.mjm.key_qpos[k][None], dtype=torch.float64)
    qvel = qpos.new_zeros((1, env.mjm.nv))
    obs = env.compute_obs(qpos, qvel, _zero_act(env, qpos))
    assert obs.shape == (1, 145)
    torch.testing.assert_close(obs[:, -28:-26], torch.ones((1, 2), dtype=qpos.dtype))
    assert obs[0, 0].item() == pytest.approx(qpos[0, 2].item(), abs=1e-14)


def test_official_stochastic_reset_preserves_equality_consistency(official_env, monkeypatch):
    env = _observation_env(official_env.mjm, 'official', num_envs=8)
    _mock_warp_state(env, monkeypatch)
    env._save_start_state()
    env.stochastic_init = True
    torch.manual_seed(91)
    env.reset()
    for state in env.warp_data.qpos:
        data = mujoco.MjData(env.mjm)
        data.qpos[:] = state.numpy()
        mujoco.mj_forward(env.mjm, data)
        equalities = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        assert np.max(np.abs(data.efc_pos[equalities])) < 1e-10
    torch.testing.assert_close(env.warp_data.qpos[:, 7:], env.start_qpos[:, 7:].double())


def test_free_root_observation_derivatives_and_quaternion_normalization(free_env):
    qpos = torch.tensor(free_env.mjm.qpos0[None], dtype=torch.float64)
    qpos[:, 3:7] = torch.tensor([[0.7, 0.2, -0.4, 0.3]], dtype=qpos.dtype)
    qpos.requires_grad_(True)
    qvel = torch.linspace(-0.5, 0.6, free_env.mjm.nv, dtype=qpos.dtype)[None].requires_grad_(True)
    act = _zero_act(free_env, qpos)
    assert torch.autograd.gradcheck(lambda q, v: free_env.compute_obs(q, v, act), (qpos, qvel))
    scaled = qpos.detach().clone()
    scaled[:, 3:7] *= 3.0
    torch.testing.assert_close(free_env.compute_obs(scaled, qvel, act), free_env.compute_obs(qpos, qvel, act))
    zero = qpos.detach().clone()
    zero[:, 3:7] = 0
    identity = zero.clone()
    identity[:, 3] = 1
    torch.testing.assert_close(free_env.compute_obs(zero, qvel, act), free_env.compute_obs(identity, qvel, act))


@pytest.mark.parametrize('grad_contract', ['strict', 'off'])
def test_official_gradient_training_is_blocked_before_backend_initialization(grad_contract):
    with pytest.raises(myoleg26_walk.GradContractError, match='free-root and tendon gradients are unvalidated'):
        MyoLeg26WalkEnv(device='cpu', grad_contract=grad_contract)


def test_official_requires_unscaled_excitation():
    with pytest.raises(ValueError, match='action_strength=1.0'):
        MyoLeg26WalkEnv(no_grad=True, action_strength=0.5)


def test_official_explicit_diagnostic_override_reaches_backend(monkeypatch):
    class BackendReached(Exception):
        pass

    def stop_at_backend(*args, **kwargs):
        assert kwargs['num_obs'] == 145
        raise BackendReached

    monkeypatch.setattr(myoleg26_walk.MjWarpEnv, '__init__', stop_at_backend)
    with pytest.raises(BackendReached):
        MyoLeg26WalkEnv(allow_unvalidated_gradients=True, grad_contract='off')


def test_official_root_topology_is_not_silently_assumed_for_legacy(env):
    env.model_contract = 'official'
    with pytest.raises(ValueError, match='Full Body'):
        env._init_pelvis_kinematics()


def test_free_root_reset_keeps_internal_reference_and_clears_integration_state(free_env, monkeypatch):
    env = _observation_env(free_env.mjm, 'official', num_envs=8)
    _mock_warp_state(env, monkeypatch)
    env._save_start_state()  # no keyframes: compiled qpos0 is valid
    env.stochastic_init = True
    env.warp_data.qpos[:, 7:] = 1.23
    env.warp_data.qvel.fill_(2.0)
    env.warp_data.ctrl.fill_(0.77)
    torch.manual_seed(91)
    obs = env.reset()
    torch.testing.assert_close(env.warp_data.qpos[:, 7:], env.start_qpos[:, 7:].double())
    torch.testing.assert_close(env.warp_data.qvel[:, 6:], env.start_qvel[:, 6:].double())
    torch.testing.assert_close(env.warp_data.qpos[:, 3:7].norm(dim=-1), torch.ones(8, dtype=torch.float64))
    assert not torch.equal(env.warp_data.qpos[:, :3], env.start_qpos[:, :3].double())
    assert torch.all(env.warp_data.ctrl == 0) and torch.all(env.warp_data.qacc_warmstart == 0)
    assert torch.all(env.warp_data.time == 0) and torch.all(env.actions == -1)
    torch.testing.assert_close(obs, env.compute_obs(env.warp_data.qpos, env.warp_data.qvel, env.warp_data.act))


@pytest.mark.parametrize('no_grad', [True, False])
@pytest.mark.parametrize('cause', ['terminated', 'truncated', 'both'])
def test_autoreset_returns_reset_obs_and_preserves_terminal_state(env, monkeypatch, no_grad, cause):
    _mock_warp_state(env, monkeypatch)
    env.no_grad = no_grad
    env.stochastic_init = False
    env.start_qpos = env.warp_data.qpos.clone()
    env.start_qvel = env.warp_data.qvel.clone()
    env.warp_model = object()
    if cause in ('truncated', 'both'):
        env.progress_buf[:] = env.episode_length - 1
    post_qpos = env.warp_data.qpos.clone()
    if cause in ('terminated', 'both'):
        post_qpos[:, 1] = 0.2
    post_act = torch.full_like(env.warp_data.act, 0.37)
    post_qvel = env.warp_data.qvel.clone()
    before_qpos = env.warp_data.qpos.clone()
    before_act = env.warp_data.act.clone()

    def advance(*_):
        env.warp_data.qpos.copy_(post_qpos)
        env.warp_data.act.copy_(post_act)
        return post_qpos, post_qvel, post_act

    if no_grad:
        monkeypatch.setattr(myoleg26_walk.mjw, 'step', advance)
    else:
        monkeypatch.setattr(WarpSimStep, 'apply', advance)
    actions = torch.full_like(env.actions, 0.2)
    obs, reward, done, extras, qpos_out, _, act_out = env.step(actions, before_qpos, post_qvel, before_act)
    assert done.item() == 1
    assert extras['terminated'].item() == (cause != 'truncated')
    assert extras['truncated'].item() == (cause == 'truncated')
    torch.testing.assert_close(extras['obs_before_reset'][:, _act_slice(env)], post_act)
    torch.testing.assert_close(extras['pelvis_position_before_reset'], env._compute_pelvis_position(post_qpos))
    torch.testing.assert_close(reward, env._compute_reward(extras['obs_before_reset'], actions))
    torch.testing.assert_close(obs, env.compute_obs(env.warp_data.qpos, env.warp_data.qvel, env.warp_data.act))
    assert torch.all(obs[:, -env.mjm.nu:] == -1)
    assert torch.all(env.progress_buf == 0)
    if not no_grad:
        torch.testing.assert_close(qpos_out, post_qpos)
        torch.testing.assert_close(act_out, post_act)


def _mock_official_task(env, monkeypatch, no_grad=True):
    _mock_warp_state(env, monkeypatch)
    env.no_grad = no_grad
    env.control_dt = env.substeps * float(env.mjm.opt.timestep)
    env._init_task_contract()
    env.stochastic_init = False
    env.start_qpos = env.warp_data.qpos.clone()
    env.start_qvel = env.warp_data.qvel.clone()
    env.warp_model = object()
    env.warp_data.nacon = torch.zeros(1, dtype=torch.long)
    env.warp_data.contact = SimpleNamespace(
        geom=torch.zeros((8, 2), dtype=torch.long),
        worldid=torch.zeros(8, dtype=torch.long),
        dist=torch.ones(8, dtype=torch.float64),
        includemargin=torch.zeros(8, dtype=torch.float64),
        efc_address=torch.full((8, 4), -1, dtype=torch.long),
    )
    monkeypatch.setattr(myoleg26_walk.mjw, 'step', lambda *_: None)
    monkeypatch.setattr(WarpSimStep, 'apply', lambda *_: (
        env.warp_data.qpos.clone(), env.warp_data.qvel.clone(), env.warp_data.act.clone(),
    ))


@pytest.mark.parametrize('no_grad', [True, False])
@pytest.mark.parametrize('action,effort', [(-1.0, 0.0), (0.0, 0.25), (1.0, 1.0)])
def test_official_step_uses_task_reward_and_exact_command_effort(official_env, monkeypatch, no_grad, action, effort):
    env = official_env
    _mock_official_task(env, monkeypatch, no_grad)
    env.warp_data.qvel[:, 0] = 1.0
    actions = torch.full_like(env.actions, action)
    obs, reward, done, extras, *_ = env.step(
        actions, env.warp_data.qpos, env.warp_data.qvel, env.warp_data.act,
    )
    assert env.task_contract.version == 'myoleg26-walk-v1'
    assert env.task_contract.target_speed == 1.0
    expected = env.control_dt * (1.0 - 0.01 * effort)
    assert reward.item() == pytest.approx(expected, abs=1e-12)
    assert extras['reward_terms']['excitation_effort'].item() == pytest.approx(effort)
    torch.testing.assert_close(reward, env.task_contract.reward(
        obs, myoleg26_walk.excitation_from_action(actions), env.control_dt,
    ))
    if no_grad:
        torch.testing.assert_close(env.warp_data.ctrl, myoleg26_walk.excitation_from_action(actions))
    assert not done.any()


def test_official_step_retains_action_effort_gradient(official_env, monkeypatch):
    env = official_env
    _mock_official_task(env, monkeypatch, no_grad=False)
    env.warp_data.qvel[:, 0] = 1.0
    actions = torch.zeros_like(env.actions, requires_grad=True)
    reward = env.step(actions, env.warp_data.qpos, env.warp_data.qvel, env.warp_data.act)[1]
    grad = torch.autograd.grad(reward.sum(), actions)[0]
    torch.testing.assert_close(grad, torch.full_like(actions, -env.control_dt * 0.01 * 0.5 / env.mjm.nu))


def test_nonfoot_contacts_filter_valid_prefix_world_ids_and_include_margin(official_env, monkeypatch):
    env = _observation_env(official_env.mjm, 'official', num_envs=7)
    _mock_official_task(env, monkeypatch)
    ground = env._ground_geom
    foot = env.mjm.geom(myoleg26_walk.COLLISION_PREFIX + 'calcn_r').id
    pelvis = env.mjm.geom(myoleg26_walk.COLLISION_PREFIX + 'pelvis').id
    contact = env.warp_data.contact
    contact.geom[:] = torch.tensor([
        [ground, foot], [pelvis, ground], [ground, pelvis], [pelvis, ground],
        [pelvis, foot], [-1, ground], [ground, pelvis], [ground, pelvis],
    ])
    contact.worldid[:] = torch.tensor([0, 1, 2, 3, 4, 5, 99, 6])
    contact.dist[:] = -0.01
    contact.dist[2:4] = 0.001
    contact.includemargin[3] = 0.002
    contact.efc_address[:] = 5  # Cannot make the separated slot 2 active.
    env.warp_data.nacon[:] = 7  # Slot 7 is stale despite a plausible contact.
    assert env._nonfoot_ground_contacts().tolist() == [False, True, False, True, False, False, False]
    contact.dist[3] = contact.includemargin[3]
    assert not env._nonfoot_ground_contacts()[3]
    env.warp_data.nacon[:] = 0
    assert not env._nonfoot_ground_contacts().any()
    env.warp_data.nacon[:] = 9
    with pytest.raises(RuntimeError, match='contact buffer overflow'):
        env._nonfoot_ground_contacts()


@pytest.mark.parametrize('height_drop', [0.0, 0.5])
def test_nonfoot_contact_flag_matches_native_contacts(official_env, monkeypatch, height_drop):
    env = official_env
    _mock_official_task(env, monkeypatch)
    data = mujoco.MjData(env.mjm)
    data.qpos[:] = env.start_qpos[0].numpy()
    data.qpos[2] -= height_drop
    mujoco.mj_forward(env.mjm, data)
    contacts = list(data.contact[:data.ncon])
    env.warp_data.nacon[:] = data.ncon
    env.warp_data.contact = SimpleNamespace(
        geom=torch.tensor(np.array([c.geom for c in contacts], dtype=np.int64).reshape(-1, 2)),
        worldid=torch.zeros(data.ncon, dtype=torch.long),
        dist=torch.tensor([c.dist for c in contacts], dtype=torch.float64),
        includemargin=torch.tensor([c.includemargin for c in contacts], dtype=torch.float64),
    )
    expected = False
    for contact in contacts:
        if contact.efc_address < 0 or env._ground_geom not in contact.geom:
            continue
        body = int(env.mjm.geom_bodyid[contact.geom[1] if contact.geom[0] == env._ground_geom else contact.geom[0]])
        expected |= mujoco.mj_id2name(env.mjm, mujoco.mjtObj.mjOBJ_BODY, body) not in myoleg26_walk.FOOT_BODIES
    assert env._nonfoot_ground_contacts().item() == expected
    assert expected == (height_drop > 0)


@pytest.mark.parametrize('cause', ['low_pelvis', 'low_upright', 'nonfoot_ground_contact', 'timeout'])
@pytest.mark.parametrize('early_termination', [True, False])
def test_official_named_failures_and_timeout_semantics(official_env, monkeypatch, cause, early_termination):
    env = official_env
    _mock_official_task(env, monkeypatch)
    env.early_termination = early_termination
    if cause == 'low_pelvis':
        env.warp_data.qpos[:, 2] = 0.53
    elif cause == 'low_upright':
        tilt = torch.tensor([[2**-0.5, 2**-0.5, 0.0, 0.0]], dtype=torch.float64)
        env.warp_data.qpos[:, 3:7] = myoleg26_walk.tu.quat_mul(tilt, env.warp_data.qpos[:, 3:7])
    elif cause == 'nonfoot_ground_contact':
        env.warp_data.nacon[:] = 1
        env.warp_data.contact.geom[0] = torch.tensor([
            env._ground_geom, env.mjm.geom(myoleg26_walk.COLLISION_PREFIX + 'pelvis').id,
        ])
        env.warp_data.contact.dist[0] = -0.001
    else:
        env.progress_buf[:] = env.episode_length - 1
    _, _, done, extras, *_ = env.step(torch.full_like(env.actions, -1.0))
    assert set(extras['failure_flags']) == {'nonfinite', 'low_pelvis', 'low_upright', 'nonfoot_ground_contact'}
    assert not extras['failure_flags']['nonfinite'].any()
    if cause != 'timeout':
        assert extras['failure_flags'][cause].item()
    assert extras['terminated'].item() == (early_termination and cause != 'timeout')
    assert extras['truncated'].item() == (cause == 'timeout')
    assert done.item() == (early_termination or cause == 'timeout')


@pytest.mark.parametrize('no_grad', [True, False])
def test_official_nonfinite_inputs_and_state_raise_without_sanitization(official_env, monkeypatch, no_grad):
    env = official_env
    _mock_official_task(env, monkeypatch, no_grad)
    with pytest.raises(FloatingPointError, match='nonfinite actions'):
        env.step(torch.full_like(env.actions, float('inf')))
    env.warp_data.qvel[:, 0] = float('inf')
    with pytest.raises(FloatingPointError, match='nonfinite qvel'):
        env.step(env.actions, env.warp_data.qpos, env.warp_data.qvel, env.warp_data.act)


def test_official_diagnostic_does_not_clip_large_finite_forward_state(official_env, monkeypatch):
    env = official_env
    _mock_official_task(env, monkeypatch, no_grad=False)
    env.warp_data.qpos[:, 0] = 500.0
    env.warp_data.qvel[:, 0] = 250.0
    obs, reward, _, _, qpos, qvel, _ = env.step(
        env.actions, env.warp_data.qpos, env.warp_data.qvel, env.warp_data.act,
    )
    assert obs[0, 5] == 250.0 and qpos[0, 0] == 500.0 and qvel[0, 0] == 250.0
    assert torch.isfinite(reward).all()
