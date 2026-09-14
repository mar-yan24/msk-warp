"""CPU reporting checks: diagnostic failures must remain readable evidence."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('check_myoleg26_task', ROOT / 'scripts/check_myoleg26_task.py')
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_comparison_fails_with_serializable_coordinates(bad):
    row = check.comparison(np.array([0., bad]), np.zeros(2), 'qpos')
    assert row['pass'] is False
    assert row['failure_reason'] == 'nonfinite_state'
    assert row['nonfinite_actual_coordinates'] == [1]
    assert row['nonfinite_expected_coordinates'] == []
    assert row['max_absolute_error'] is None
    assert row['max_scaled_error'] is None
    assert row['worst_coordinate'] == 1
    json.dumps(row, allow_nan=False)


def test_finite_comparison_keeps_original_tolerance_gate():
    assert check.comparison(np.array([4e-5]), np.zeros(1), 'qpos')['pass']
    failed = check.comparison(np.array([6e-5]), np.zeros(1), 'qpos')
    assert not failed['pass']
    assert failed['failure_reason'] == 'tolerance_exceeded'
    assert failed['max_scaled_error'] == pytest.approx(1.2)
    assert failed['nonfinite_actual_coordinates'] == []


def test_nested_nonfinite_behavior_values_become_null_with_exact_paths():
    output = io.StringIO()
    check.write_report(output, {'forward': {'pass': False}, 'behavior': {
        'returns': [np.float64('nan'), 2.], 'position': np.array([1., np.inf]),
    }})
    record = json.loads(output.getvalue())
    assert record['forward']['pass'] is False
    assert record['behavior']['returns'] == [None, 2.]
    assert record['behavior']['position'] == [1., None]
    assert record['report_has_nonfinite_values'] is True
    assert record['nonfinite_json_paths'] == ['$.behavior.returns[0]', '$.behavior.position[1]']


def test_manifest_reports_model_and_mesh_tampering_and_missing_files(tmp_path):
    model = tmp_path / 'flat_boxes.xml'
    mesh = tmp_path / 'bone.stl'
    model.write_bytes(b'model')
    mesh.write_bytes(b'mesh')
    manifest = {'source': {'commit': 'pin'}, 'files': {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (model, mesh)
    }}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert check.model_manifest_status(model)['status'] == 'verified'
    mesh.write_bytes(b'changed geometry')
    changed = check.model_manifest_status(model)
    assert changed['status'] == 'mismatch'
    assert changed['files']['flat_boxes.xml']['status'] == 'match'
    assert changed['files']['bone.stl']['status'] == 'mismatch'
    mesh.unlink()
    assert check.model_manifest_status(model)['files']['bone.stl']['status'] == 'missing'
    model.write_bytes(b'changed model')
    assert check.model_manifest_status(model)['files']['flat_boxes.xml']['status'] == 'mismatch'


def test_missing_invalid_and_unlisted_manifests_are_not_verified(tmp_path):
    model = tmp_path / 'flat_boxes.xml'
    model.write_bytes(b'model')
    assert check.model_manifest_status(model)['status'] == 'missing'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{bad json')
    assert check.model_manifest_status(model)['status'] == 'invalid'
    other = tmp_path / 'other.xml'
    other.write_bytes(b'other')
    manifest.write_text(json.dumps({'files': {'other.xml': hashlib.sha256(b'other').hexdigest()}}))
    record = check.model_manifest_status(model)
    assert record['status'] == 'mismatch'
    assert record['model_listed'] is False


def test_application_record_captures_dirty_state_and_working_source_hash(tmp_path, monkeypatch):
    script = tmp_path / 'scripts/check_myoleg26_task.py'
    script.parent.mkdir()
    script.write_bytes(b'working implementation')
    def git(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=(' M scripts/check_myoleg26_task.py\n'
                               if 'status' in args else 'commit\n'))
    monkeypatch.setattr(check.subprocess, 'run', git)
    record = check.application_state(tmp_path)
    assert record['application_commit'] == 'commit'
    assert record['application_dirty'] is True
    assert record['application_status'] == [' M scripts/check_myoleg26_task.py']
    assert record['implementation_sha256']['scripts/check_myoleg26_task.py'] == hashlib.sha256(script.read_bytes()).hexdigest()


def _main_args(monkeypatch, model, output):
    monkeypatch.setattr(sys, 'argv', ['check_myoleg26_task', '--model', str(model),
                                    '--parity-only', '--out', str(output)])
    monkeypatch.setattr(check, 'resolve_model_path', lambda path: path)
    monkeypatch.setattr(check, 'application_state', lambda root: {'application_dirty': True})
    monkeypatch.setattr(check, '_revision', lambda root: 'commit')


@pytest.mark.parametrize('failure', ['missing_model', 'gpu_metadata'])
def test_metadata_errors_still_leave_valid_reserved_report(tmp_path, monkeypatch, failure):
    model, output = tmp_path / 'model.xml', tmp_path / 'report.json'
    if failure == 'gpu_metadata':
        model.write_bytes(b'model')
    _main_args(monkeypatch, model, output)
    def no_gpu():
        raise RuntimeError('GPU unavailable')
    monkeypatch.setattr(check.torch.cuda, 'get_device_name', no_gpu)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        check.main()
    record = json.loads(output.read_text())
    assert record['status'] == 'error'
    assert record['application_dirty'] is True
    assert ('FileNotFoundError' if failure == 'missing_model' else 'GPU unavailable') in record['error']


def test_main_reports_failed_all_horizon_checks_and_exits_nonzero(tmp_path, monkeypatch):
    model, output = tmp_path / 'model.xml', tmp_path / 'report.json'
    model.write_bytes(b'model')
    _main_args(monkeypatch, model, output)
    monkeypatch.setattr(check.torch.cuda, 'get_device_name', lambda: 'mock GPU, not initialized')
    row = check.comparison(np.array([np.nan]), np.zeros(1), 'act')
    monkeypatch.setattr(check, 'check_forward', lambda path, seeds: {
        'pass': False, 'checks': [{'physics_steps': horizon, 'fields': {'act': row}}
                                for horizon in (1, 4, 16)],
    })
    with pytest.raises(SystemExit) as error:
        check.main()
    assert error.value.code == 1
    record = json.loads(output.read_text())
    assert record['status'] == 'completed'
    assert record['forward']['pass'] is False
    assert [row['physics_steps'] for row in record['forward']['checks']] == [1, 4, 16]
    assert record['forward']['checks'][0]['fields']['act']['max_absolute_error'] is None


@pytest.mark.parametrize('ending', ['failure', 'timeout', 'nonfinite_reward'])
def test_behavior_stops_first_episode_and_captures_named_failure_flags(monkeypatch, ending):
    import msk_warp.envs.myoleg26_walk as env_module

    class FakeEnv:
        device, num_actions, substeps = 'cpu', 2, 1
        mjm = SimpleNamespace(opt=SimpleNamespace(timestep=.01))
        task_contract = SimpleNamespace(as_dict=lambda: {})

        def __init__(self, **kwargs):
            pass

        def reset(self):
            pass

        def state_tensors(self):
            return torch.tensor([[0., 0., 1.]]), None, None

        def _compute_pelvis_position(self, qpos):
            return qpos

        def step(self, action):
            obs = torch.ones((1, 3))
            reward = torch.tensor([float('nan') if ending == 'nonfinite_reward' else 2.])
            extras = {'obs_before_reset': obs,
                      'pelvis_position_before_reset': torch.tensor([[.1, 0., 1.]]),
                      'terminated': torch.tensor([ending != 'timeout']),
                      'truncated': torch.tensor([ending == 'timeout']),
                      'failure_flags': {'low_pelvis': torch.tensor([ending == 'failure'])}}
            return obs, reward, torch.tensor([True]), extras

    monkeypatch.setattr(env_module, 'MyoLeg26WalkEnv', FakeEnv)
    monkeypatch.setattr(check.wp, 'synchronize', lambda: None)
    record = check.measure_behavior('unused.xml', episodes=1, seconds=.1, seed=0)
    for condition in record['conditions']:
        assert condition['first_episode_duration_s'] == [.01]
        assert condition['scored_first_episode_transitions'] == 1
        assert condition['end_failure_flags'] == [{'low_pelvis': ending == 'failure'}]
        expected_reason = {'failure': 'task_failure', 'timeout': 'timeout', 'nonfinite_reward': 'nonfinite'}[ending]
        assert condition['end_reason'] == [expected_reason]
        assert condition['nonfinite_fields_at_end'] == ([['reward']] if ending == 'nonfinite_reward' else [[]])
    output = io.StringIO()
    check.write_report(output, record)
    serialized = json.loads(output.getvalue())
    assert serialized['report_has_nonfinite_values'] == (ending == 'nonfinite_reward')
