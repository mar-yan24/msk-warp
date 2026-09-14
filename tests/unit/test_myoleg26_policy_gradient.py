"""CPU negative controls for conditional complete-policy gradient qualification."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('check_myoleg26_policy_gradient', ROOT / 'scripts/check_myoleg26_policy_gradient.py')
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def _passing_local_gate():
    horizon = {'passed': True, 'unmodified_ad': True, 'ad_replay_forward': {'passed': True},
               'forward': {'passed': True},
               'blocks': {block: {'terms': {term: {'passed': True} for term in check.TERMS}}
                          for block in check.BLOCKS},
               'sanitization': {'reward': {'calls': 1, 'nonfinite_entries': 0, 'clamped_finite_entries': 0}}}
    longer = copy.deepcopy(horizon)
    longer['sanitization']['reward']['calls'] = 4
    return {'schema_version': 'myoleg26-derivatives-v1', 'passed': True,
            'coverage': {'passed': True}, 'settings': {'horizons': [1, 4], 'directions': 10,
                'cosine_min': .99, 'gradient_rtol': .1, 'gradient_atol': 1e-5,
                'forward_rtol': 1e-3, 'forward_atol': 1e-4, 'backward_mode': 'tape_per_substep'},
            'samples': [{'sample_id': i, 'horizons': {'1': copy.deepcopy(horizon), '4': copy.deepcopy(longer)}}
                        for i in range(30)]}


def test_local_gate_rejects_contradictory_block_pass_and_modified_gradients():
    gate = _passing_local_gate()
    assert check.local_gate_reasons(gate) == []
    gate['samples'][0]['horizons']['4']['blocks']['act']['terms']['total_task_reward']['passed'] = False
    assert any('local_block_failed:0:4:act' in reason for reason in check.local_gate_reasons(gate))
    gate = _passing_local_gate()
    gate['samples'][0]['horizons']['1']['sanitization']['reward']['clamped_finite_entries'] = 1
    assert any('local_sanitization_modified_gradient' in reason for reason in check.local_gate_reasons(gate))


@pytest.mark.parametrize('field,value', [('cosine_min', .9), ('gradient_atol', 1),
                                        ('gradient_rtol', .5), ('forward_rtol', float('nan')),
                                        ('forward_atol', .1), ('backward_mode', 'fd')])
def test_policy_gate_rejects_weakened_local_evidence(field, value):
    gate = _passing_local_gate()
    gate['settings'][field] = value
    assert check.local_gate_reasons(gate)


def test_sanitization_gate_requires_observed_calls_but_does_not_treat_calls_as_modifications():
    counts = {'calls': 16, 'nonfinite_entries': 0, 'clamped_finite_entries': 0}
    assert check.unmodified_ad(counts, 16)
    assert not check.unmodified_ad(counts, 4)
    assert not check.unmodified_ad(counts, 16, env_nonfinite=1)
    assert not check.unmodified_ad({**counts, 'clamped_finite_entries': 1}, 16)


@pytest.mark.parametrize('present', [False, True])
def test_blocked_local_gate_never_loads_checkpoint_or_initializes_cuda(tmp_path, monkeypatch, present):
    local, output = tmp_path / 'local.json', tmp_path / 'policy.json'
    if present:
        local.write_text(json.dumps({'schema_version': 'myoleg26-derivatives-v1', 'passed': False,
                                    'gate_status': 'failed_action_gradient', 'provenance': {'backend': 'known failure'}}))
    args = ['policy', '--cfg', str(tmp_path / 'cfg.yaml'), '--checkpoint', str(tmp_path / 'actor.pt'),
            '--states', str(tmp_path / 'states.npz'), '--local-gate', str(local), '--out', str(output)]
    monkeypatch.setattr(sys, 'argv', args)
    def forbidden(*args, **kwargs):
        pytest.fail('blocked qualification attempted a policy load or GPU rollout')
    monkeypatch.setattr(check.torch, 'load', forbidden)
    monkeypatch.setattr(check, 'WarpPolicyRollout', forbidden)
    monkeypatch.setattr(check.torch.cuda, 'get_device_name', forbidden)
    with pytest.raises(SystemExit) as error:
        check.main()
    assert error.value.code == 1
    report = json.loads(output.read_text())
    assert report['passed'] is False and report['gate_status'] == 'blocked'
    expected = 'local_action_state_gates_not_passed' if present else 'missing_local_gate_report'
    assert expected in report['blocked_reasons']
    assert 'samples' not in report
    if present:
        assert report['local_gate']['provenance']['backend'] == 'known failure'
        assert report['inputs']['local_gate']['sha256'] == check.digest(local)


def test_failed_local_report_identifies_a_different_requested_dataset(tmp_path):
    local, states = tmp_path / 'local.json', tmp_path / 'states.npz'
    states.write_bytes(b'current states; no archive load should occur')
    local.write_text(json.dumps({'schema_version': 'myoleg26-derivatives-v1', 'passed': False,
                                'dataset': {'sha256': 'some-other-archive'}}))
    args = SimpleNamespace(local_gate=local, states=states, cfg=tmp_path / 'c', checkpoint=tmp_path / 'p')
    report = {}
    assert check.preflight(args, report) is None
    assert report['gate_status'] == 'blocked'
    assert 'local_gate_dataset_hash_mismatch' in report['blocked_reasons']
    assert report['local_gate']['dataset_matches_requested_states'] is False
    assert report['local_gate']['dataset_sha256'] == 'some-other-archive'
    assert report['inputs']['states']['sha256'] == check.digest(states)


def test_independence_counts_episodes_not_multiple_timesteps():
    samples = [{'seed': 0, 'checkpoint_sha256': 'frozen', 'episode_id': episode, 'progress': step}
               for episode in range(30) for step in (4, 8)]
    metadata = {'schema_version': 'myoleg26-visited-v1', 'held_out': True, 'samples': samples}
    assert check.independent_indices(metadata) == list(range(0, 60, 2))
    with pytest.raises(ValueError, match='found 29'):
        check.independent_indices({**metadata, 'samples': samples[:-2]})
    with pytest.raises(ValueError, match='held-out'):
        check.independent_indices({**metadata, 'held_out': False})
    del samples[0]['progress']
    with pytest.raises(ValueError, match='progress'):
        check.independent_indices(metadata)


def test_shared_reset_ids_across_policies_do_not_satisfy_30_independent_starts():
    samples = [{'seed': seed, 'checkpoint_sha256': f'actor{seed}', 'episode_id': 30000 + episode,
                'progress': 4} for seed in range(5) for episode in range(10)]
    metadata = {'schema_version': 'myoleg26-visited-v1', 'held_out': True, 'samples': samples}
    with pytest.raises(check.QualificationBlocked) as error:
        check.independent_indices(metadata)
    assert error.value.reason == 'additional-independent-starts-required'
    coverage = error.value.details['independence']
    assert coverage['policy_episode_pairs'] == 50
    assert coverage['source_episode_clusters'] == 10
    assert all(count == 5 for count in coverage['policy_episode_pairs_per_reset_cluster'].values())


def test_insufficient_cluster_report_is_blocked_not_an_AD_result(tmp_path, monkeypatch):
    output = tmp_path / 'policy.json'
    monkeypatch.setattr(sys, 'argv', ['policy', '--cfg', 'c', '--checkpoint', 'p', '--states', 's',
                                    '--local-gate', 'g', '--out', str(output)])
    def blocked(args, report):
        raise check.QualificationBlocked('additional-independent-starts-required', 'found 16 distinct reset IDs',
                                        independence={'source_episode_clusters': 16, 'policy_episode_pairs': 50})
    monkeypatch.setattr(check, 'run', blocked)
    with pytest.raises(SystemExit) as error:
        check.main()
    assert error.value.code == 1
    report = json.loads(output.read_text())
    assert report['gate_status'] == 'blocked' and report['passed'] is False
    assert report['blocked_reasons'] == ['additional-independent-starts-required']
    assert report['independence']['source_episode_clusters'] == 16
    assert 'samples' not in report


def test_segment_objective_closes_timeouts_and_terminal_coincidence_correctly():
    # Rewards: 1 + .9*2, timeout value .9^2*10, then 3 + .9*4;
    # a terminal coinciding with the final timeout must not use value 100.
    loss = check.segment_objective([1., 2., 3., 4.], [False, False, False, True],
                                   [False, True, False, True], [5., 10., 20., 100.], .9)
    assert loss == pytest.approx(-(1 + .9 * 2 + .9 ** 2 * 10 + 3 + .9 * 4) / 4)


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_net = torch.nn.Linear(2, 1)
        self.logstd = torch.nn.Parameter(torch.tensor([-.7]))
        with torch.no_grad():
            self.mu_net.weight[:] = torch.tensor([[.4, .8]])
            self.mu_net.bias.fill_(.1)


def _critic():
    critic = torch.nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight[:] = torch.tensor([[.3, .2]])
        critic.bias.fill_(.05)
    return critic.requires_grad_(False)


class ToyEnv:
    """Differentiable scalar simulator with the production previous-action choice."""
    device, num_envs, num_obs, episode_length = 'cpu', 1, 2, 100

    def __init__(self, detach_action=True, endings=None):
        self.x = torch.tensor([[.3]])
        self.actions = torch.tensor([[-.2]])
        self.detach_action = detach_action
        self.endings = endings or [None] * 4
        self.index = 0

    def initialize_trajectory(self):
        return self.compute_obs(self.x, None, None)

    def state_tensors(self):
        return self.x.detach().clone(), torch.zeros((1, 1)), torch.zeros((1, 0))

    def compute_obs(self, qpos, qvel, act):
        return torch.cat((qpos, self.actions), dim=1)

    def step(self, action, qpos, qvel, act):
        new = qpos + .2 * action
        final_obs = torch.cat((new, action), dim=1)
        reward = .2 * new[:, 0] - .03 * action[:, 0].square()
        ending = self.endings[self.index]
        terminated, truncated = ending in ('terminal', 'both'), ending in ('timeout', 'both')
        self.actions = action.detach().clone() if self.detach_action else action
        self.x = new.detach().clone()
        if terminated or truncated:
            self.x = torch.tensor([[.25]])
            self.actions = torch.tensor([[-1.]])
        self.index += 1
        return final_obs, reward, torch.tensor([terminated or truncated]), {
            'obs_before_reset': final_obs, 'terminated': torch.tensor([terminated]),
            'truncated': torch.tensor([truncated]),
        }, new, qvel, act


def toy_native(actor, critic, noise, endings):
    x, previous, rewards, values, terminated, truncated = .3, -.2, [], [], [], []
    with torch.no_grad():
        for draw, ending in zip(noise, endings):
            obs = torch.tensor([[x, previous]], dtype=torch.float64)
            action = float(torch.tanh(actor.mu_net(obs) + actor.logstd.exp() * draw).item())
            x += .2 * action
            rewards.append(.2 * x - .03 * action ** 2)
            values.append(float(critic(torch.tensor([[x, action]], dtype=torch.float64)).item()))
            terminated.append(ending in ('terminal', 'both'))
            truncated.append(ending in ('timeout', 'both'))
            previous = action
            if terminated[-1] or truncated[-1]:
                x, previous = .25, -1.
    return check.segment_objective(rewards, terminated, truncated, values, .9)


def test_actual_shac_forward_matches_independent_stochastic_reset_objective():
    actor, critic = TinyActor(), _critic()
    endings = [None, 'timeout', None, 'both']
    noise = torch.tensor([.4, -.2, .7, -.1]).reshape(4, 1, 1)
    wrapped = check.FixedNoiseActor(actor, noise)
    shac = check.make_shac(ToyEnv(endings=endings), wrapped, critic, None, 4, .9, 0)
    actual = float(shac.compute_actor_loss().detach())
    expected = toy_native(copy.deepcopy(actor).double(), copy.deepcopy(critic).double(), noise.double(), endings)
    assert actual == pytest.approx(expected, abs=2e-7)


@pytest.mark.parametrize('detach_action', [False, True])
def test_complete_policy_negative_control_exposes_previous_action_detach(detach_action):
    actor, critic = TinyActor(), _critic()
    noise = torch.tensor([.4, -.2, .7, -.1]).reshape(4, 1, 1)
    wrapped = check.FixedNoiseActor(actor, noise)
    shac = check.make_shac(ToyEnv(detach_action=detach_action), wrapped, critic, None, 4, .9, 0)
    loss = shac.compute_actor_loss()
    loss.backward()
    ad = torch.nn.utils.parameters_to_vector([p.grad for p in actor.parameters()]).double().numpy()
    reference = copy.deepcopy(actor).double()
    theta = torch.nn.utils.parameters_to_vector(reference.parameters()).detach().numpy().copy()
    critic = copy.deepcopy(critic).double()
    fd = []
    for direction in np.eye(len(theta)):
        torch.nn.utils.vector_to_parameters(torch.tensor(theta + 1e-5 * direction), reference.parameters())
        plus = toy_native(reference, critic, noise.double(), [None] * 4)
        torch.nn.utils.vector_to_parameters(torch.tensor(theta - 1e-5 * direction), reference.parameters())
        minus = toy_native(reference, critic, noise.double(), [None] * 4)
        fd.append((plus - minus) / 2e-5)
    # logstd is the first registered parameter and participates under fixed stochastic draws.
    assert abs(fd[0]) > 1e-4
    if detach_action:
        assert np.linalg.norm(ad - fd) > .01
    else:
        np.testing.assert_allclose(ad, fd, atol=1e-7, rtol=2e-5)


def test_fixed_policy_noise_repeats_and_directions_cover_all_parameters():
    actor = TinyActor()
    noise = torch.tensor([[[.7]], [[-.3]]])
    obs = torch.tensor([[.2, -.1]])
    first, second = check.FixedNoiseActor(actor, noise), check.FixedNoiseActor(actor, noise)
    torch.testing.assert_close(first(obs), second(obs))
    torch.testing.assert_close(first(obs), second(obs))
    with pytest.raises(ValueError, match='exhausted'):
        first(obs)
    directions = check.parameter_directions(sum(p.numel() for p in actor.parameters()), 10, 1)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.)
    assert np.any(directions[:, 0] != 0)
    common = check.policy_noise(16, 26, 7)
    torch.testing.assert_close(common, common.float().double(), rtol=0, atol=0)
    torch.testing.assert_close(common[:4], check.policy_noise(4, 26, 7), rtol=0, atol=0)


def test_native_observation_and_reward_match_cpu_task_contract():
    from msk_warp.envs.myoleg26_task import MyoLegTaskContract
    from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'msk_warp/assets/myoleg26/flat_boxes.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('stand').id)
    rng = np.random.default_rng(8)
    data.qvel[:] = rng.uniform(-.2, .2, model.nv)
    data.act[:] = rng.uniform(0, 1, model.na)
    action = rng.uniform(-.8, .8, model.nu)
    obs = check.native_observation(model, data, action)
    env = object.__new__(MyoLeg26WalkEnv)
    env.mjm, env.device, env.model_contract, env.num_environments = model, 'cpu', 'official', 1
    env._init_pelvis_kinematics()
    env.up_vec, env.heading_vec = env.up_vec.double(), env.heading_vec.double()
    env.actions = torch.tensor(action)[None]
    expected = env.compute_obs(torch.tensor(data.qpos)[None], torch.tensor(data.qvel)[None], torch.tensor(data.act)[None])
    np.testing.assert_allclose(obs, expected.numpy()[0], atol=2e-14, rtol=0)
    task = MyoLegTaskContract()
    reward, flags = check.native_reward_and_failure(obs, action, False, task.as_dict(), .008)
    exact = task.reward(expected, .5 * (torch.tensor(action)[None] + 1), .008).item()
    assert reward == pytest.approx(exact, abs=1e-15)
    assert flags == {name: bool(value.item()) for name, value in task.failures(expected, model.nu).items()} | {'nonfoot_ground_contact': False}


def test_frozen_normalization_uses_checkpoint_statistics_without_updates():
    from msk_warp.utils.running_mean_std import RunningMeanStd
    rms = RunningMeanStd(shape=(2,), device='cpu')
    rms.mean, rms.var = torch.tensor([2., 3.]), torch.tensor([4., 9.])
    obs = torch.tensor([[8., 12.]], dtype=torch.float64)
    before = copy.deepcopy(rms)
    actual = check.frozen_normalize(obs, rms)
    torch.testing.assert_close(actual, (obs - before.mean) / torch.sqrt(before.var.double() + 1e-5))
    torch.testing.assert_close(rms.mean, before.mean)
    assert rms.count == before.count


def test_native_rollout_uses_frozen_reset_draws_and_timeout_final_values():
    from msk_warp.envs.myoleg26_task import MyoLegTaskContract
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'msk_warp/assets/myoleg26/flat_boxes.xml'))
    actor = SimpleNamespace(mu_net=torch.nn.Linear(145, 26).double(), logstd=torch.full((26,), -.7, dtype=torch.float64))
    critic = torch.nn.Linear(145, 1).double()
    with torch.no_grad():
        actor.mu_net.weight.zero_()
        actor.mu_net.bias.fill_(-.1)
        critic.weight.zero_()
        critic.bias.fill_(1.)
    state = {'qpos': model.key_qpos[model.key('stand').id].copy(), 'qvel': np.zeros(model.nv),
             'act': np.zeros(model.na), 'previous_action': np.full(model.nu, -.2), 'progress': 0}
    settings = {'task_contract': MyoLegTaskContract().as_dict(), 'substeps': 1, 'control_dt': .002,
                'episode_length': 1, 'early_termination': True, 'gamma': .9, 'reward_scale': 2.}
    resets = check.reset_schedule(model, 2, 4, True)
    repeat_resets = check.reset_schedule(model, 2, 4, True)
    for first, second in zip(resets, repeat_resets):
        for field in first:
            np.testing.assert_array_equal(first[field], second[field])
    noise = torch.zeros((2, 1, 26), dtype=torch.float64)
    record = check.native_rollout(model, actor, critic, None, state, noise, resets, settings)
    assert [row['truncated'] for row in record['trace']] == [True, True]
    assert [row['terminated'] for row in record['trace']] == [False, False]
    np.testing.assert_array_equal(record['trace'][0]['policy_obs'][-26:], -.2)
    np.testing.assert_array_equal(record['trace'][1]['policy_obs'][-26:], -1.)
    assert record['loss'] == pytest.approx(-sum(row['reward'] + .9 for row in record['trace']) / 2)


@pytest.mark.parametrize('mode', [None, 'tape'])
def test_warp_runner_explicitly_binds_validated_mode_without_simulator(monkeypatch, mode):
    import msk_warp.envs.myoleg26_walk as env_module
    captured = {}
    class FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.backward_mode = kwargs['backward_mode']
        def step(self, *args):
            raise AssertionError('no simulation in this CPU constructor test')
    monkeypatch.setattr(env_module, 'MyoLeg26WalkEnv', FakeEnv)
    env_cfg = {'name': 'MyoLeg26Walk'}
    if mode:
        env_cfg['backward_mode'] = mode
    runner = check.WarpPolicyRollout({'params': {'env': env_cfg}},
                                    [TinyActor(), None, _critic(), None, None], 'unused.xml', 'cpu')
    assert captured['backward_mode'] == (mode or 'tape_per_substep')
    assert runner.env.backward_mode == captured['backward_mode']


def test_policy_aggregate_resamples_reset_id_clusters_not_policy_pairs():
    metadata = [{'episode_id': episode, 'seed': policy} for episode in range(30) for policy in range(2)]
    samples = [{'horizons': {'4': {'AD_projections': [1., 2.], 'FD_projections': [1., 2.]}}} for _ in metadata]
    aggregate = check.aggregate_clusters(samples, metadata, 4, 7)
    assert aggregate['complete']
    assert aggregate['reset_noise_clusters'] == 30
    assert aggregate['policy_episode_pairs'] == 60
    assert aggregate['projected_cosine'] == pytest.approx(1.)
    np.testing.assert_allclose(aggregate['relative_l2_ci95'], [0., 0.])
    assert not check.aggregate_clusters(samples[:-2], metadata[:-2], 4, 7)['complete']
