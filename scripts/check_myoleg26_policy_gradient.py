"""Conditional projected check of the complete stochastic SHAC actor objective.

This does not enable gradient training. Failed or missing local action/state gates
produce a blocked report before loading a policy or initializing CUDA. A qualified
run needs >=30 distinct held-out reset/noise episode IDs, ten actor-parameter directions and
H=4/16 control steps. It compares actual SHAC.compute_actor_loss with an independent
native float64 rollout, using common reparameterization/reset draws, the checkpoint
target critic and frozen observation RMS. All actor parameters, including logstd,
participate. Previous-action features vary naturally under finite differences.

Usage: python scripts/check_myoleg26_policy_gradient.py --cfg CFG --checkpoint PT
       --states VISITED.npz --local-gate LOCAL.json --out POLICY_GATE.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import MethodType
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (4, 16)
BLOCKS = ('action', 'qpos_root_tangent', 'qpos_internal_tangent', 'qvel', 'act')
TERMS = ('total_task_reward', 'locomotion_only', 'state_projection_0', 'state_projection_1')
TRACE_TOLERANCES = {'qpos': (5e-5, 5e-4), 'qvel': (1e-3, 1e-3), 'act': (1e-6, 1e-5),
                    'action': (1e-5, 1e-3), 'obs': (1e-3, 1e-3),
                    'policy_obs': (1e-3, 1e-3), 'reward': (1e-7, 1e-3),
                    'bootstrap_value': (1e-5, 1e-3)}


class QualificationBlocked(ValueError):
    def __init__(self, reason, message, **details):
        super().__init__(message)
        self.reason, self.details = reason, details


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unmodified_ad(counts, horizon, *, env_nonfinite=0):
    return (counts.get('calls') == horizon and counts.get('nonfinite_entries') == 0
            and counts.get('clamped_finite_entries') == 0 and env_nonfinite == 0)


def local_gate_reasons(gate):
    """Do not promote a top-level pass that contradicts individual prerequisites."""
    reasons = []
    if gate.get('schema_version') != 'myoleg26-derivatives-v1':
        reasons.append('unsupported_local_gate_schema')
    if gate.get('passed') is not True:
        reasons.append('local_action_state_gates_not_passed')
        return reasons
    if gate.get('coverage', {}).get('passed') is not True:
        reasons.append('local_coverage_not_passed')
    settings = gate.get('settings', {})
    if not {1, 4}.issubset(settings.get('horizons', [])) or settings.get('directions', 0) < 10:
        reasons.append('insufficient_local_horizons_or_directions')
    for field, bound, lower_bound in (('cosine_min', .99, True), ('gradient_rtol', .1, False),
                                      ('gradient_atol', 1e-5, False), ('forward_rtol', 1e-3, False),
                                      ('forward_atol', 1e-4, False)):
        value = settings.get(field)
        if (not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                or (value < bound if lower_bound else value > bound)):
            reasons.append('missing_or_weakened_local_threshold:' + field)
    if settings.get('backward_mode') not in ('tape', 'tape_per_substep'):
        reasons.append('local_gate_does_not_qualify_an_autodiff_bridge')
    samples = gate.get('samples', [])
    if len(samples) < 30:
        reasons.append('fewer_than_30_local_samples')
    for sample in samples:
        for horizon in ('1', '4'):
            row = sample.get('horizons', {}).get(horizon, {})
            if row.get('forward', {}).get('passed') is not True:
                reasons.append(f"local_forward_failed:{sample.get('sample_id')}:{horizon}")
            if (row.get('passed') is not True or row.get('unmodified_ad') is not True
                    or row.get('ad_replay_forward', {}).get('passed') is not True):
                reasons.append(f"local_AD_replay_not_passed:{sample.get('sample_id')}:{horizon}")
            for block in BLOCKS:
                terms = row.get('blocks', {}).get(block, {}).get('terms', {})
                if any(terms.get(term, {}).get('passed') is not True for term in TERMS):
                    reasons.append(f"local_block_failed:{sample.get('sample_id')}:{horizon}:{block}")
            sanitation = row.get('sanitization')
            if not isinstance(sanitation, dict) or not sanitation:
                reasons.append(f"local_sanitization_unreported:{sample.get('sample_id')}:{horizon}")
            elif any(not unmodified_ad(value, int(horizon)) for value in sanitation.values()):
                reasons.append(f"local_sanitization_modified_gradient:{sample.get('sample_id')}:{horizon}")
    return reasons


def episode_coverage(metadata):
    pairs, clusters = set(), {}
    for sample in metadata['samples']:
        key = (sample['seed'], sample['checkpoint_sha256'], str(sample['episode_id']))
        if key not in pairs:
            pairs.add(key)
            cluster = str(sample['episode_id'])
            clusters[cluster] = clusters.get(cluster, 0) + 1
    return {'policy_episode_pairs': len(pairs), 'source_episode_clusters': len(clusters),
            'policy_episode_pairs_per_reset_cluster': clusters,
            'unit': 'source episode/reset-noise ID shared across frozen policies; policy-episode pairs are correlated'}


def independent_indices(metadata, minimum=30):
    if metadata.get('schema_version') != 'myoleg26-visited-v1' or metadata.get('held_out') is not True:
        raise ValueError('policy qualification requires an explicitly held-out visited-v1 archive')
    seen, selected = set(), []
    for index, sample in enumerate(metadata['samples']):
        key = (sample['seed'], sample['checkpoint_sha256'], str(sample['episode_id']))
        if key not in seen:
            if 'progress' not in sample and metadata.get('step_semantics') != 'pre_action_episode_progress':
                raise ValueError('missing explicit pre-action episode progress; timeout replay would be ambiguous')
            seen.add(key)
            selected.append(index)
    coverage = episode_coverage(metadata)
    if coverage['source_episode_clusters'] < minimum:
        raise QualificationBlocked('additional-independent-starts-required',
            f"need {minimum} distinct held-out reset/noise IDs; found {coverage['source_episode_clusters']}; "
            'reusing reset/noise draws across policies does not increase independent-start coverage', independence=coverage)
    return selected


def parameter_directions(size, count, seed):
    if size < 1 or count < 10:
        raise ValueError('at least ten actor-parameter directions are required')
    directions = np.random.default_rng(seed).normal(size=(count, size))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def policy_noise(horizon, actions, seed):
    # Draws must be identical numbers in both runtimes, not double draws that
    # are rounded only on the Warp side of the comparison.
    values = np.random.default_rng(seed).normal(size=(horizon, 1, actions)).astype(np.float32)
    return torch.tensor(values.astype(np.float64))


class FixedNoiseActor(torch.nn.Module):
    """The ordinary stochastic MLP reparameterization, with external fixed draws."""
    def __init__(self, actor, noise):
        super().__init__()
        if not hasattr(actor, 'mu_net') or not hasattr(actor, 'logstd'):
            raise ValueError('qualification currently requires ActorStochasticMLP')
        self.actor = actor
        self.noise = noise
        self.index = 0

    def forward(self, obs, deterministic=False):
        if deterministic:
            raise ValueError('complete stochastic policy qualification must include logstd')
        if self.index >= len(self.noise):
            raise ValueError('fixed policy noise exhausted')
        result = self.actor.mu_net(obs) + self.actor.logstd.exp() * self.noise[self.index].to(obs)
        self.index += 1
        return result


def make_shac(env, actor, target_critic, rms, horizon, gamma, progress, reward_scale=1.0):
    """Construct rollout bookkeeping around the actual production SHAC objective."""
    from msk_warp.algorithms.shac import SHAC
    from msk_warp.utils.average_meter import AverageMeter
    shac = object.__new__(SHAC)
    shac.env, shac.actor, shac.target_critic = env, actor, target_critic
    shac.device, shac.num_envs, shac.num_obs = env.device, env.num_envs, env.num_obs
    shac.steps_num, shac.gamma, shac.rew_scale = horizon, gamma, reward_scale
    shac.max_episode_length = env.episode_length
    shac.obs_rms, shac.ret_rms = copy.deepcopy(rms), None
    shac.state_bptt, shac.state_grad_clip, shac.state_grad_decay = True, 0., 0.
    shac.obs_buf = torch.zeros((horizon, env.num_envs, env.num_obs), device=env.device)
    for name in ('rew_buf', 'done_mask', 'next_values'):
        setattr(shac, name, torch.zeros((horizon, env.num_envs), device=env.device))
    for name in ('episode_loss', 'episode_discounted_loss'):
        setattr(shac, name, torch.zeros(env.num_envs, device=env.device))
    shac.episode_gamma = torch.ones(env.num_envs, device=env.device)
    shac.episode_length = torch.full((env.num_envs,), progress, dtype=torch.long, device=env.device)
    for name in ('episode_loss', 'episode_discounted_loss', 'episode_length'):
        setattr(shac, name + '_his', [])
        setattr(shac, name + '_meter', AverageMeter(1, 100).to(env.device))
    shac.step_count = 0
    return shac


def segment_objective(rewards, terminated, truncated, values, gamma):
    """Independent SHAC forward objective: discounted episode fragments / horizon.

    Each reset closes a fragment. True terminals have no value, while timeouts
    and a surviving horizon use the frozen critic on the final pre-reset state.
    The next fragment restarts its discount at one.
    """
    total, discount = 0., 1.
    for step, reward in enumerate(rewards):
        total -= discount * reward
        ended = bool(terminated[step] or truncated[step])
        if ended or step == len(rewards) - 1:
            if not terminated[step]:
                total -= gamma * discount * values[step]
        discount = 1. if ended else discount * gamma
    return total / len(rewards)


def frozen_normalize(obs, rms):
    if rms is None:
        return obs
    return (obs - rms.mean.to(obs)) / torch.sqrt(rms.var.to(obs) + 1e-5)


def native_observation(model, data, previous_action):
    """Independent native body Jacobians, without the environment's Torch helpers."""
    observer = mujoco.MjData(model)
    observer.qpos[:], observer.qvel[:] = data.qpos, data.qvel
    mujoco.mj_normalizeQuat(model, observer.qpos)
    mujoco.mj_kinematics(model, observer)
    mujoco.mj_comPos(model, observer)
    pelvis = model.body('pelvis').id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, observer, jp, jr, pelvis)
    rotation = observer.xmat[pelvis].reshape(3, 3)
    return np.r_[observer.xpos[pelvis, 2], observer.xquat[pelvis], jp @ data.qvel, jr @ data.qvel,
                 data.qpos[7:], .1 * data.qvel[6:], data.act, rotation[2, 1], rotation[0, 0], previous_action]


def native_nonfoot_contact(model, data):
    from msk_warp.models.myoleg26 import FOOT_BODIES, GROUND_NAME
    ground = model.geom(GROUND_NAME).id
    for contact in data.contact[:data.ncon]:
        if contact.dist >= contact.includemargin or ground not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == ground else contact.geom1
        body = int(model.geom_bodyid[other])
        if body and model.body(body).name not in FOOT_BODIES:
            return True
    return False


def native_reward_and_failure(obs, action, contact, task, control_dt):
    """Float64 transcription checked against the declared task by CPU regressions."""
    nu = len(action)
    excitation = .5 * (np.clip(action, -1., 1.) + 1.)
    tracking = np.exp(-((obs[5] - task['target_speed']) ** 2 + obs[6] ** 2) / task['velocity_variance'])
    reward = control_dt * (tracking * np.clip(obs[-nu - 2], 0, 1) * np.clip(obs[-nu - 1], 0, 1)
                           - task['effort_weight'] * np.mean(excitation ** 2))
    flags = {'nonfinite': not bool(np.isfinite(obs).all()), 'low_pelvis': bool(obs[0] < task['termination_height']),
             'low_upright': bool(obs[-nu - 2] < task['termination_upright']), 'nonfoot_ground_contact': bool(contact)}
    return float(reward), flags


def reset_schedule(model, horizon, seed, stochastic):
    """Common exogenous reset draws, rounded once to the production state precision."""
    from msk_warp.utils import torch_utils as tu
    generator = torch.Generator(device='cpu').manual_seed(seed)
    stand = model.key('stand').id
    result = []
    for _ in range(horizon):
        qpos = torch.tensor(model.key_qpos[stand].copy(), dtype=torch.float32)
        qvel = torch.tensor(model.key_qvel[stand].copy(), dtype=torch.float32)
        if stochastic:
            qpos[0] += .1 * (torch.rand((), generator=generator) - .5)
            qpos[2] += .02 * torch.rand((), generator=generator)
            qvel[:6] += .1 * (torch.rand(6, generator=generator) - .5)
            yaw = .1 * (torch.rand((1,), generator=generator) - .5)
            rotation = tu.quat_from_angle_axis(yaw, torch.tensor([[0., 0., 1.]]))
            qpos[3:7] = tu.quat_mul(rotation, qpos[3:7][None])[0]
        qpos[3:7] /= qpos[3:7].norm()
        result.append({'qpos': qpos.numpy().astype(float), 'qvel': qvel.numpy().astype(float),
                       'act': np.zeros(model.na)})
    return result


def native_rollout(model, actor, critic, rms, state, noise, resets, settings):
    """Fresh float64 native dynamics; no Warp bridge or production SHAC step loop."""
    data = mujoco.MjData(model)
    for name in ('qpos', 'qvel', 'act'):
        getattr(data, name)[:] = state[name]
    previous_action = state['previous_action'].copy()
    progress = int(state['progress'])
    data.time = progress * settings['control_dt']
    rows = []
    with torch.no_grad():
        for step in range(len(noise)):
            policy_obs = frozen_normalize(torch.tensor(native_observation(model, data, previous_action))[None], rms)
            action = torch.tanh(actor.mu_net(policy_obs) + actor.logstd.exp() * noise[step]).numpy()[0]
            data.ctrl[:] = .5 * (action + 1.)
            for _ in range(settings['substeps']):
                mujoco.mj_step(model, data)
                if np.any(data.warning.number):
                    raise FloatingPointError('native physics emitted a warning; invalid reference trajectory')
            obs = native_observation(model, data, action)
            reward, flags = native_reward_and_failure(obs, action, native_nonfoot_contact(model, data),
                                                       settings['task_contract'], settings['control_dt'])
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise FloatingPointError('native reference state/reward became nonfinite')
            progress += 1
            terminated = settings['early_termination'] and any(flags.values())
            truncated = progress >= settings['episode_length'] and not terminated
            value = 0. if terminated else float(critic(frozen_normalize(torch.tensor(obs)[None], rms)).item())
            rows.append({**{name: getattr(data, name).copy() for name in ('qpos', 'qvel', 'act')},
                         'obs': obs, 'policy_obs': policy_obs.numpy()[0], 'action': action,
                         'reward': reward * settings['reward_scale'], 'bootstrap_value': value,
                         'terminated': bool(terminated), 'truncated': bool(truncated), 'failure_flags': flags})
            previous_action = action
            if terminated or truncated:
                mujoco.mj_resetData(model, data)
                for name, values in resets[step].items():
                    getattr(data, name)[:] = values
                previous_action = -np.ones(model.nu)
                progress = 0
    loss = segment_objective(*([row[name] for row in rows] for name in
                               ('reward', 'terminated', 'truncated', 'bootstrap_value')), settings['gamma'])
    return {'loss': loss, 'trace': rows}


def end_signature(trace):
    return [(row['terminated'], row['truncated'], tuple(sorted(row['failure_flags'].items()))) for row in trace]


def identical_rollouts(actual, expected):
    """Repeated frozen rollouts must agree exactly, not merely meet physics tolerances."""
    return (actual['loss'] == expected['loss'] and end_signature(actual['trace']) == end_signature(expected['trace'])
            and all(np.array_equal(a[field], b[field]) for a, b in zip(actual['trace'], expected['trace'])
                    for field in TRACE_TOLERANCES))


def compare_rollouts(actual, expected):
    checks = []
    if len(actual['trace']) != len(expected['trace']):
        return {'passed': False, 'reason': 'trace_length_mismatch'}
    flags_equal = end_signature(actual['trace']) == end_signature(expected['trace'])
    for step, (a, b) in enumerate(zip(actual['trace'], expected['trace'])):
        for field, (atol, rtol) in TRACE_TOLERANCES.items():
            aa, bb = np.asarray(a[field]), np.asarray(b[field])
            if aa.shape != bb.shape:
                checks.append({'step': step, 'field': field, 'passed': False, 'reason': 'shape_mismatch'})
                continue
            finite = np.isfinite(aa).all() and np.isfinite(bb).all()
            scaled = np.abs(aa - bb) / (atol + rtol * np.abs(bb))
            checks.append({'step': step, 'field': field, 'passed': bool(finite and np.all(scaled <= 1)),
                           'max_scaled_error': float(scaled.max()) if np.isfinite(scaled).all() else None})
    loss_equal = bool(np.isfinite(actual['loss']) and np.isclose(actual['loss'], expected['loss'], atol=1e-5, rtol=1e-3))
    return {'passed': flags_equal and loss_equal and all(row['passed'] for row in checks),
            'end_flags_equal': flags_equal, 'loss_equal': loss_equal,
            'actual_loss': actual['loss'], 'native_loss': expected['loss'], 'checks': checks}


class WarpPolicyRollout:
    """Use production SHAC/env code; only supply exogenous state/noise/reset draws."""
    def __init__(self, cfg, checkpoint, model_path, device):
        import warp as wp
        from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv
        self.wp = wp
        env_cfg = dict(cfg['params']['env'])
        for key in ('name', 'num_actors', 'num_envs'):
            env_cfg.pop(key, None)
        env_cfg.update(model_path=str(model_path), num_envs=1, device=device, no_grad=False,
                       stochastic_init=False, allow_unvalidated_gradients=True, grad_contract='off',
                       backward_mode=env_cfg.get('backward_mode', 'tape_per_substep'))
        self.env = MyoLeg26WalkEnv(**env_cfg)
        if self.env.backward_mode != env_cfg['backward_mode']:
            raise ValueError('constructed policy environment did not retain the validated backward mode')
        self.actor = copy.deepcopy(checkpoint[0]).to(device=device, dtype=torch.float32).eval()
        self.critic = copy.deepcopy(checkpoint[2]).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.rms = checkpoint[3].to(device) if checkpoint[3] is not None else None
        self.original_step = self.env.step
        self.env._reset_warp_state = MethodType(self._reset, self.env)
        self.env.step = self._step
        self.rows, self.resets = [], []
        self.index = 0

    def _write(self, state):
        for name in ('qpos', 'qvel', 'act'):
            tensor = torch.tensor(state[name], dtype=torch.float32, device=self.env.device)[None]
            self.wp.copy(getattr(self.env.warp_data, name), self.wp.from_torch(tensor))

    def _reset(self, env, env_ids):
        if env_ids.tolist() != [0]:
            raise ValueError('qualification replays one independent state at a time')
        self._write(self.resets[self.index])
        for name in ('ctrl', 'qacc_warmstart', 'time'):
            self.wp.to_torch(getattr(env.warp_data, name)).zero_()
        env.actions = torch.full_like(env.actions, -1.)
        env.progress_buf.zero_()

    def _step(self, actions, *state):
        result = self.original_step(actions, *state)
        _, reward, _, extras, qpos, qvel, act = result
        self.rows.append({name: value.detach().cpu().numpy()[0] for name, value in
                          (('qpos', qpos), ('qvel', qvel), ('act', act),
                           ('obs', extras['obs_before_reset']), ('action', actions))})
        self.rows[-1].update(reward=float(reward.item()), terminated=bool(extras['terminated'].item()),
                             truncated=bool(extras['truncated'].item()),
                             failure_flags={name: bool(value.item()) for name, value in extras['failure_flags'].items()})
        self.index += 1
        return result

    def rollout(self, state, noise, resets, settings, *, gradient=False):
        import mujoco_warp as mjw
        env = self.env
        env.clear_grad()
        mjw.reset_data(env.warp_model, env.warp_data)
        self._write(state)
        self.wp.to_torch(env.warp_data.time).fill_(state['progress'] * settings['control_dt'])
        env.actions = torch.tensor(state['previous_action'], dtype=torch.float32, device=env.device)[None]
        env.progress_buf.fill_(state['progress'])
        env.reset_buf.zero_()
        env.termination_buf.zero_()
        self.rows, self.resets, self.index = [], resets, 0
        actor = FixedNoiseActor(self.actor, noise.to(device=env.device, dtype=torch.float32))
        shac = make_shac(env, actor, self.critic, self.rms, len(noise), settings['gamma'], state['progress'], settings['reward_scale'])
        actor.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(gradient):
            loss = shac.compute_actor_loss(deterministic=False)
        for step, row in enumerate(self.rows):
            row.update(policy_obs=shac.obs_buf[step, 0].detach().cpu().numpy().copy(),
                       bootstrap_value=float(shac.next_values[step, 0]))
            row['reward'] *= settings['reward_scale']
        report = {'loss': float(loss.detach()), 'trace': copy.deepcopy(self.rows)}
        return report, loss, actor


def preflight(args, report):
    """Cheap blocked path comes first: no checkpoint deserialization or CUDA here."""
    report['implementation_sha256'] = {name: digest(ROOT / name) for name in
        ('scripts/check_myoleg26_policy_gradient.py', 'scripts/check_trajopt_gradients.py',
         'msk_warp/algorithms/shac.py', 'msk_warp/envs/myoleg26_walk.py', 'msk_warp/envs/myoleg26_task.py',
         'msk_warp/bridge.py', 'msk_warp/backend.py')}
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    report['application_commit'] = revision.stdout.strip() if revision.returncode == 0 else None
    report['inputs'] = {name: {'path': str(getattr(args, name)), 'sha256': digest(getattr(args, name))
                              if Path(getattr(args, name)).is_file() else None}
                        for name in ('cfg', 'checkpoint', 'states', 'local_gate')}
    if not Path(args.local_gate).is_file():
        report.update(gate_status='blocked', blocked_reasons=['missing_local_gate_report'])
        return None
    gate = json.loads(Path(args.local_gate).read_text())
    report['local_gate'] = {'passed': gate.get('passed'), 'gate_status': gate.get('gate_status'),
                          'provenance': gate.get('provenance'), 'coverage': gate.get('coverage')}
    reasons = local_gate_reasons(gate)
    gate_dataset_hash = gate.get('dataset', {}).get('sha256')
    requested_dataset_hash = report['inputs']['states']['sha256']
    report['local_gate']['dataset_sha256'] = gate_dataset_hash
    if not gate_dataset_hash or not requested_dataset_hash:
        report['local_gate']['dataset_matches_requested_states'] = None
        reasons.append('local_gate_dataset_identity_unavailable')
    else:
        report['local_gate']['dataset_matches_requested_states'] = gate_dataset_hash == requested_dataset_hash
        if gate_dataset_hash != requested_dataset_hash:
            reasons.append('local_gate_dataset_hash_mismatch')
    gate_metadata = gate.get('dataset', {}).get('metadata', {})
    if gate_metadata.get('samples'):
        report['independence'] = episode_coverage(gate_metadata)
        if report['independence']['source_episode_clusters'] < 30:
            reasons.append('additional-independent-starts-required')
    if reasons:
        report.update(gate_status='blocked', blocked_reasons=reasons)
        return None
    return gate


def run(args, report):
    gate = preflight(args, report)
    if gate is None:
        return
    helpers = load_helper('check_trajopt_gradients')
    diagnostic = load_helper('check_myoleg26_task')
    from msk_warp import resolve_model_path
    from msk_warp.envs.myoleg26_task import MyoLegTaskContract
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    config, env_cfg = cfg['params']['config'], cfg['params']['env']
    if env_cfg.get('backward_mode', 'tape_per_substep') != gate.get('settings', {}).get('backward_mode'):
        raise ValueError('policy backward mode differs from the passed local gate')
    if config.get('ret_rms', False) or not config.get('state_bptt', True) or any(
            config.get(key, 0) != 0 for key in ('state_grad_clip', 'state_grad_decay')):
        raise ValueError('qualification requires ret_rms=False and full unmodified state BPTT')
    if env_cfg.get('model_contract') != 'official' or env_cfg.get('action_strength', 1.) != 1.:
        raise ValueError('qualification requires the official unscaled-excitation task')
    model_path = Path(resolve_model_path(env_cfg['model_path'])).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    model_hash = hashlib.sha256(buffer.tobytes()).hexdigest()
    if gate.get('provenance', {}).get('compiled_model_sha256') != model_hash:
        raise ValueError('local gate compiled model hash differs from the policy task')
    if gate.get('dataset', {}).get('sha256') != digest(args.states):
        raise ValueError('policy states differ from the local gate dataset')
    import mujoco_warp as mjw
    current_backend = helpers.git_provenance(Path(mjw.__file__).parent)
    for field in ('head', 'tracked_diff_sha256'):
        if not current_backend.get(field) or current_backend.get(field) != gate['provenance'].get('backend', {}).get(field):
            raise ValueError(f'backend {field} changed since local qualification')
        current_repo = helpers.git_provenance(ROOT)
        if not current_repo.get(field) or current_repo.get(field) != gate['provenance'].get('repo', {}).get(field):
            raise ValueError(f'application {field} changed since local qualification')
    import importlib.metadata
    packages = gate['provenance'].get('packages', {})
    if not {'numpy', 'torch', 'mujoco', 'mujoco-warp', 'warp-lang'}.issubset(packages):
        raise ValueError('local gate package provenance is incomplete')
    for package, version in packages.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'package {package} changed since local qualification')
    sources = gate['provenance'].get('sources', {})
    required_sources = {'msk_warp/bridge.py', 'msk_warp/backend.py', 'msk_warp/envs/myoleg26_walk.py',
                        'msk_warp/envs/myoleg26_task.py'}
    normalized_sources = {str(name).replace('\\', '/'): value for name, value in sources.items()}
    if not required_sources.issubset(normalized_sources):
        raise ValueError('local gate implementation source hashes are incomplete')
    for name, value in normalized_sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != value:
            raise ValueError(f'implementation source changed since local qualification: {name}')
    with np.load(args.states, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json'].item()))
        report['independence'] = episode_coverage(metadata)
        selected = independent_indices(metadata)
        arrays = {name: archive[name].astype(float) for name in ('qpos', 'qvel', 'act', 'previous_action')}
    for name, width in (('qpos', model.nq), ('qvel', model.nv), ('act', model.na), ('previous_action', model.nu)):
        value = arrays[name]
        if value.shape != (len(metadata['samples']), width) or not np.isfinite(value).all():
            raise ValueError(f'invalid visited-state array: {name}')
        if not np.array_equal(value, value.astype(np.float32).astype(float)):
            raise ValueError(f'visited state {name} is not exactly representable by the production float32 state')
    if metadata.get('compiled_model_sha256') != model_hash:
        raise ValueError('visited-state compiled model hash mismatch')
    task = MyoLegTaskContract(target_speed=env_cfg.get('target_speed', 1.)).as_dict()
    substeps = env_cfg.get('substeps', 4)
    if metadata.get('substeps') != substeps or metadata.get('task_contract') != task:
        raise ValueError('visited-state task/control contract differs from qualification')
    if args.samples is not None:
        if args.samples != len(selected):
            raise ValueError('qualification must use every preregistered independent selected state, not a post hoc subset')
    if len(selected) < 30:
        raise ValueError('cannot qualify fewer than 30 independent episodes')
    if not any(sample['seed'] == 0 and sample['checkpoint_sha256'] == digest(args.checkpoint)
               for sample in metadata['samples']):
        raise ValueError('target actor must be the predetermined seed-0 checkpoint represented in the archive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if len(checkpoint) < 5 or checkpoint[4] is not None:
        raise ValueError('checkpoint must contain actor/critic/target/RMS and no return normalization')
    actor = copy.deepcopy(checkpoint[0]).to(dtype=torch.float64).eval()
    critic = copy.deepcopy(checkpoint[2]).to(dtype=torch.float64).eval().requires_grad_(False)
    rms = copy.deepcopy(checkpoint[3])
    theta = torch.nn.utils.parameters_to_vector(actor.parameters()).detach().numpy().copy()
    directions = parameter_directions(len(theta), args.directions, args.seed)
    settings = {'gamma': config.get('gamma', .99), 'reward_scale': config.get('rew_scale', 1.),
                'episode_length': env_cfg['episode_length'], 'early_termination': env_cfg.get('early_termination', True),
                'substeps': substeps, 'control_dt': substeps * model.opt.timestep, 'task_contract': task}
    report.update(settings=settings, horizons=list(HORIZONS), parameter_count=len(theta),
                  parameter_names=[name for name, _ in actor.named_parameters()], directions=directions.tolist(),
                  independent_episode_count=report['independence']['source_episode_clusters'],
                  policy_episode_pair_count=len(selected), source_samples=[metadata['samples'][i] for i in selected],
                  initial_distribution='pooled held-out policy-episode states, clustered by shared reset/noise IDs; off-policy for the fixed seed-0 actor',
                  provenance={'application': diagnostic.application_state(ROOT), 'backend': current_backend,
                              'compiled_model_sha256': model_hash, 'model_manifest': diagnostic.model_manifest_status(model_path)},
                  samples=[], gate_status='running')
    runner = WarpPolicyRollout(cfg, checkpoint, model_path, args.device)
    from msk_warp import bridge
    for index in selected:
        sample = metadata['samples'][index]
        progress = int(sample['progress'] if 'progress' in sample else sample['step'])
        if not 0 <= progress < settings['episode_length']:
            raise ValueError('visited sample progress is outside a live episode')
        state = {name: values[index] for name, values in arrays.items()}
        state['progress'] = progress
        output = {'sample_id': sample['sample_id'], 'horizons': {}}
        report['samples'].append(output)
        for horizon in HORIZONS:
            print(f"policy gate sample {sample['sample_id']} H{horizon}", flush=True)
            noise = policy_noise(horizon, model.nu, args.seed + index)
            resets = reset_schedule(model, horizon, args.seed + 100000 + index, env_cfg.get('stochastic_init', True))
            torch.nn.utils.vector_to_parameters(torch.tensor(theta), actor.parameters())
            native = native_rollout(model, actor, critic, rms, state, noise, resets, settings)
            actual, _, _ = runner.rollout(state, noise, resets, settings)
            repeated, _, _ = runner.rollout(state, noise, resets, settings)
            row = {'passed': False, 'forward': compare_rollouts(actual, native),
                   'determinism': {'passed': identical_rollouts(repeated, actual)}, 'noise': noise.numpy().tolist(),
                   'resets': [{name: values.tolist() for name, values in reset.items()} for reset in resets]}
            output['horizons'][str(horizon)] = row
            if not row['forward']['passed'] or not row['determinism']['passed']:
                row['reason'] = 'forward_or_replay_mismatch_before_AD'
                continue
            event_changes, fd_errors = [], []
            def objective(vector):
                torch.nn.utils.vector_to_parameters(torch.tensor(vector), actor.parameters())
                try:
                    result = native_rollout(model, actor, critic, rms, state, noise, resets, settings)
                except FloatingPointError as exc:
                    fd_errors.append(str(exc))
                    return np.nan
                if end_signature(result['trace']) != end_signature(native['trace']):
                    event_changes.append(True)
                    return np.nan
                return result['loss']
            sweeps = [helpers.epsilon_sweep(objective, theta, direction) for direction in directions]
            row['epsilon_sweeps'] = sweeps
            row['event_signature_changes'] = len(event_changes)
            row['native_FD_errors'] = fd_errors
            if event_changes or fd_errors or any(sweep['windows'][0] is None for sweep in sweeps):
                row['reason'] = 'uncertified_native_epsilon_window_or_changed_failure_flags'
                continue
            ad_report, loss, wrapped_actor = runner.rollout(state, noise, resets, settings, gradient=True)
            row['AD_forward'] = compare_rollouts(ad_report, native)
            if not row['AD_forward']['passed']:
                row['reason'] = 'AD_forward_mismatch'
                continue
            env_nan_entries = [0]
            original_nan_to_num = torch.nan_to_num
            def observe_nan(tensor, *a, **kw):
                env_nan_entries[0] += int((~torch.isfinite(tensor)).sum().item())
                return original_nan_to_num(tensor, *a, **kw)
            with helpers.count_sanitization(bridge) as sanitation, patch.object(torch, 'nan_to_num', observe_nan):
                loss.backward()
            gradient = np.concatenate([p.grad.detach().cpu().double().numpy().ravel()
                                       if p.grad is not None else np.zeros(p.numel()) for p in wrapped_actor.parameters()])
            row['sanitization'] = {**sanitation, 'all_nan_to_num_nonfinite_entries': env_nan_entries[0]}
            ad, fd = directions @ gradient, np.array([sweep['windows'][0]['derivative'] for sweep in sweeps])
            row['AD_projections'], row['FD_projections'] = ad.tolist(), fd.tolist()
            row['metrics'] = helpers.compare_derivatives(ad, fd, atol=1e-7, seed=args.seed + index)
            row['passed'] = (row['metrics']['passed'] and unmodified_ad(sanitation, horizon, env_nonfinite=env_nan_entries[0])
                             and np.linalg.norm(fd) > 1e-7)
            if not row['passed']:
                row['reason'] = 'projected_derivative_mismatch_modified_gradient_or_uninformative_reference'
    report['aggregate'] = {str(h): aggregate_clusters(report['samples'], report['source_samples'], h, args.seed)
                           for h in HORIZONS}
    report['passed'] = all(sample['horizons'][str(h)]['passed'] for sample in report['samples'] for h in HORIZONS)
    report['gate_status'] = 'qualified_on_sampled_directions' if report['passed'] else 'not_qualified'


def aggregate_clusters(samples, metadata, horizon, seed):
    """Descriptive paired cluster bootstrap, conditional on the frozen policies."""
    rows = [sample['horizons'][str(horizon)] for sample in samples]
    if any('AD_projections' not in row or 'FD_projections' not in row for row in rows):
        return {'complete': False, 'reason': 'not_all_policy_episode_pairs_have_projected_derivatives'}
    ad, fd = np.array([row['AD_projections'] for row in rows]), np.array([row['FD_projections'] for row in rows])
    clusters = {}
    for index, meta in enumerate(metadata):
        clusters.setdefault(str(meta['episode_id']), []).append(index)
    if len(clusters) < 30:
        return {'complete': False, 'reason': 'additional-independent-starts-required', 'clusters': len(clusters)}
    def metrics(indices):
        a, b = ad[indices].ravel(), fd[indices].ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return [float(np.clip(a @ b / denom, -1, 1)) if denom > 1e-30 else np.nan,
                float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-7))]
    groups = list(clusters.values())
    rng = np.random.default_rng(seed + horizon)
    draws = np.array([metrics(np.concatenate([groups[i] for i in ids]))
                      for ids in rng.integers(0, len(groups), size=(2000, len(groups)))])
    point = metrics(np.arange(len(rows)))
    return {'complete': True, 'reset_noise_clusters': len(groups), 'policy_episode_pairs': len(rows),
            'scope': '2000 paired resamples of source episode/reset-noise ID clusters, conditional on fixed policies and directions',
            'projected_cosine': point[0], 'relative_l2': point[1],
            'projected_cosine_ci95': np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
            'relative_l2_ci95': np.percentile(draws[:, 1], [2.5, 97.5]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cfg', 'checkpoint', 'states', 'local-gate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--directions', type=int, default=10)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--seed', type=int, default=62000)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.directions < 10 or (args.samples is not None and args.samples < 30):
        parser.error('qualification requires >=10 directions and >=30 distinct reset/noise episode IDs')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 'myoleg26-policy-derivatives-v1', 'passed': False, 'gate_status': 'started',
              'objective': 'actual stochastic SHAC actor loss, fixed reparameterization/reset draws, frozen target critic/RMS, full state BPTT',
              'scope': 'projected complete-objective derivatives on sampled policy-episode pairs clustered by reset/noise ID; not full-gradient cosine or training success',
              'ci_scope': 'paired direction-bootstrap intervals within each independent starting state, not a population CI',
              'gradient_criteria': {'cosine_min': .99, 'relative_l2_max': .1, 'direction_rtol': .1,
                                    'direction_atol': 1e-7, 'minimum_FD_projection_norm': 1e-7},
              'FD_window_criteria': {'selection': 'native only; finest stable triple and all finer evidence',
                                     'central_rtol': 1e-3, 'central_atol': 1e-7,
                                     'one_sided_symmetry_rtol': 1e-2, 'one_sided_symmetry_atol': 1e-6},
              'known_unresolved_issue': 'production SHAC previous-action observation is detached; native policy FD retains its dependence'}
    with args.out.open('x', encoding='utf-8') as output:
        try:
            run(args, report)
        except QualificationBlocked as exc:
            report.update(gate_status='blocked', blocked_reasons=[exc.reason], explanation=str(exc), **exc.details)
        except Exception as exc:
            report.update(gate_status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            load_helper('check_myoleg26_task').write_report(output, report)
    print(json.dumps({'passed': report['passed'], 'gate_status': report['gate_status'], 'out': str(args.out)}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
