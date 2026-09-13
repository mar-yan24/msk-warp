"""CPU controls for evaluation metrics, reset writes, RNG isolation and provenance."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from msk_warp.analysis.myoleg26_baseline import (
    behavior_rank, contiguous_windows, episode_initial_states, evaluate_policy, isolated_rng,
)
from msk_warp.envs.myoleg26_task import MyoLegTaskContract
from scripts.run_myoleg26_baseline import PROTOCOL, save_traces, success, write_json
import scripts.run_myoleg26_baseline as runner


class FakeEnv:
    """Writable simulator arrays with a deliberately copy-returning state API."""

    def __init__(self):
        self.num_envs, self.num_actions, self.device = 2, 26, "cpu"
        self.mjm, self.substeps, self.control_dt = SimpleNamespace(na=26), 4, .008
        self.task_contract = MyoLegTaskContract()
        self.start_qpos = torch.zeros((2, 47))
        self.start_qpos[:, 2:4] = 1
        self.start_qvel = torch.zeros((2, 46))
        self.warp_data = SimpleNamespace(qpos=self.start_qpos.clone(), qvel=self.start_qvel.clone(),
                                         act=torch.zeros((2, 26)))
        self.actions = torch.full((2, 26), -1.)
        self.progress = torch.zeros(2, dtype=torch.long)

    def state_tensors(self):
        return tuple(getattr(self.warp_data, key).clone() for key in ("qpos", "qvel", "act"))

    def reset(self):
        # Evaluation must restore even random draws in setup/reset code.
        random.random(), np.random.random(), torch.rand(1)
        self.warp_data.qpos.copy_(self.start_qpos)
        self.warp_data.qvel.zero_()
        self.warp_data.act.zero_()
        self.actions.fill_(-1)
        self.progress.zero_()
        self.calculateObservations()
        return self.obs_buf

    def calculateObservations(self):
        self.obs_buf = torch.zeros((2, 145))
        self.obs_buf[:, 0] = self.warp_data.qpos[:, 2]
        self.obs_buf[:, 5:8] = self.warp_data.qvel[:, :3]
        self.obs_buf[:, -26:] = self.actions
        self.last_observed_qpos = self.warp_data.qpos.clone()

    def _compute_pelvis_position(self, qpos):
        return qpos[:, :3]

    def step(self, action):
        self.progress += 1
        self.actions.copy_(action)
        self.warp_data.qvel[:, 0] = torch.tensor([1., .5])
        self.warp_data.qvel[:, 1] = torch.tensor([.3, 0.])
        self.warp_data.qpos[:, :3] += self.warp_data.qvel[:, :3] * self.control_dt
        self.warp_data.act.add_(.01)
        self.calculateObservations()
        terminal_obs = self.obs_buf.clone()
        position = self.warp_data.qpos[:, :3].clone()
        terminated = torch.tensor([self.progress[0] == 2, False])
        done = terminated | (self.progress >= 5)
        for i in done.nonzero().flatten():
            self.warp_data.qpos[i] = self.start_qpos[i]
            self.warp_data.qvel[i] = 0
            self.warp_data.act[i] = 0
            self.actions[i] = -1
            self.progress[i] = 0
        self.calculateObservations()
        return self.obs_buf, torch.ones(2), done, {
            "obs_before_reset": terminal_obs, "pelvis_position_before_reset": position,
            "terminated": terminated, "failure_flags": {"test_failure": terminated},
        }


class FakeActor:
    def __call__(self, obs, deterministic=False):
        return torch.zeros((obs.shape[0], 26))

    def forward_with_dist(self, obs, deterministic=False):
        mean = self(obs)
        return mean, mean, torch.full((26,), .3)


@pytest.fixture
def fake_warp(monkeypatch):
    import warp

    monkeypatch.setattr(warp, "to_torch", lambda tensor: tensor)


def test_episode_seed_sampling_is_batch_order_independent_and_root_only():
    env = FakeEnv()
    first = episode_initial_states(env.start_qpos[0], env.start_qvel[0], 26, [10000, 10001])
    reordered = episode_initial_states(env.start_qpos[0], env.start_qvel[0], 26, [10001, 10000])
    for a, b in zip(first, reordered):
        np.testing.assert_array_equal(a, b[::-1])
    assert not np.array_equal(first[0][0], first[0][1])
    np.testing.assert_allclose(np.linalg.norm(first[0][:, 3:7], axis=1), 1, atol=1e-7)
    np.testing.assert_array_equal(first[0][:, 7:], 0)
    np.testing.assert_array_equal(first[1][:, 6:], 0)
    np.testing.assert_array_equal(first[2], 0)


def test_rng_isolation_restores_all_cpu_streams_after_exception():
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    with isolated_rng():
        expected = random.random(), np.random.random(), torch.rand(4)
    with pytest.raises(RuntimeError), isolated_rng():
        random.random(), np.random.random(), torch.rand(100)
        raise RuntimeError("test failure")
    actual = random.random(), np.random.random(), torch.rand(4)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_first_episode_metrics_account_for_ignored_worlds_and_live_reset(fake_warp):
    env = FakeEnv()
    seeds = [10000, 10001]
    expected = episode_initial_states(env.start_qpos[0], env.start_qvel[0], 26, seeds)
    counter = {"attempted_calls": 0, "completed_calls": 0}
    result, traces = evaluate_policy(env, FakeActor(), None, seeds, horizon=5, capture=True,
                                     transition_counter=counter)
    # Would fail if reset assignment used state_tensors(), whose values are copies.
    np.testing.assert_array_equal(traces["qpos"][:2], expected[0])
    np.testing.assert_array_equal(traces["qvel"][:2], expected[1])
    np.testing.assert_array_equal(traces["previous_action"][:2], -1)
    assert counter == {"attempted_calls": 5, "completed_calls": 5}
    assert result["accounting"] == {
        "simulated_control_transitions": 10, "scored_first_episode_transitions": 7,
        "ignored_restarted_world_transitions": 3, "simulated_physics_steps": 40,
    }
    assert [row["length"] for row in result["episodes"]] == [2, 5]
    assert [row["end_reason"] for row in result["episodes"]] == ["task_failure", "timeout"]
    assert result["summary"]["survival_fraction"] == .5
    assert result["summary"]["mean_forward_velocity_mps"] == pytest.approx(.75, abs=1e-6)
    assert result["summary"]["mean_episode_velocity_2d_rmse_mps"] == pytest.approx(.4)
    assert result["summary"]["pooled_velocity_2d_rmse_mps"] == pytest.approx(np.sqrt((2*.09+5*.25)/7))
    assert result["summary"]["forward_time_weighted_absolute_speed_error_mps"] == pytest.approx(2.5/7)
    assert result["summary"]["forward_speed_band_fraction"] == pytest.approx(2/7)
    assert result["summary"]["mean_excitation"] == .5
    assert result["summary"]["mean_excitation_squared"] == .25
    np.testing.assert_array_equal(traces["step"], [0, 0, 1, 1, 2, 3, 4])
    np.testing.assert_array_equal(contiguous_windows(traces, 3), [[1, 3, 4], [3, 4, 5]])


def test_stochastic_audit_noise_is_repeatable_and_does_not_advance_training_rng(fake_warp):
    torch.manual_seed(29)
    with isolated_rng():
        expected_after = torch.rand(8)
    _, first = evaluate_policy(FakeEnv(), FakeActor(), None, [30000, 30001], horizon=5,
                               capture=True, deterministic=False)
    torch.testing.assert_close(torch.rand(8), expected_after, rtol=0, atol=0)
    _, second = evaluate_policy(FakeEnv(), FakeActor(), None, [30000, 30001], horizon=5,
                                capture=True, deterministic=False)
    np.testing.assert_array_equal(first["action"], second["action"])
    assert not np.array_equal(first["action"][0], first["action"][1])


def test_wall_cap_marks_partial_evaluation_censored(fake_warp):
    result, _ = evaluate_policy(FakeEnv(), FakeActor(), None, [10000, 10001], horizon=5, deadline=0)
    assert not result["complete"]
    assert [row["end_reason"] for row in result["episodes"]] == ["wall_cap", "wall_cap"]
    assert result["accounting"]["simulated_control_transitions"] == 2
    assert not success(result, 0)


def test_behavior_selection_and_success_use_episode_2d_rmse():
    def evaluation(survived, rmse, band=1.):
        return {"complete": True, "episodes": [{"survived": i < survived} for i in range(16)],
                "summary": {"survival_fraction": survived/16, "mean_forward_velocity_mps": 1.,
                            "mean_episode_velocity_2d_rmse_mps": rmse, "forward_speed_band_fraction": band}}
    assert behavior_rank(evaluation(15, .2)) > behavior_rank(evaluation(14, .01))
    assert behavior_rank(evaluation(15, .2)) > behavior_rank(evaluation(15, .3))
    assert behavior_rank(evaluation(15, .2, .9)) > behavior_rank(evaluation(15, .2, .8))
    assert success(evaluation(15, .3), 15)
    assert not success(evaluation(15, .301), 15)
    assert not success(evaluation(14, 0), 15)
    partial = evaluation(16, .1)
    partial["complete"] = False
    assert not success(partial, 15)
    assert PROTOCOL["epochs"] * PROTOCOL["steps_num"] * PROTOCOL["num_actors"] == 1048576
    assert PROTOCOL["seeds"] == [0, 1, 2, 3, 4]


def test_artifacts_are_exclusive_and_trace_metadata_preserves_hashes(tmp_path):
    output = tmp_path / "exclusive.json"
    write_json(output, {"test": True})
    with pytest.raises(FileExistsError):
        write_json(output, {"test": False})
    assert json.loads(output.read_text()) == {"test": True}
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    frozen = {"protocol": PROTOCOL, "model_path": "msk_warp/assets/myoleg26/flat_boxes.xml", "model_sha256": "xmlhash",
              "compiled_model_sha256": "compiledhash", "task_contract": {"version": "v1"}, "files": {"a": "hash"}}
    archive = tmp_path / "trace.npz"
    save_traces(archive, {"qpos": np.ones((1, 47))}, frozen, seed=2, checkpoint=checkpoint,
                checkpoint_epoch=32, evaluation={"control_dt": .008, "policy_mode": "stochastic_normal", "complete": True})
    with np.load(archive, allow_pickle=False) as data:
        metadata = json.loads(data["metadata_json"].item())
        assert metadata["compiled_model_sha256"] == "compiledhash"
        assert metadata["frozen_files"] == frozen["files"]
        assert metadata["training_seed"] == 2 and metadata["checkpoint_epoch"] == 32
        assert len(metadata["checkpoint_sha256"]) == 64
    with pytest.raises(FileExistsError):
        save_traces(archive, {}, frozen, seed=2, checkpoint=checkpoint, checkpoint_epoch=32,
                    evaluation={"control_dt": .008, "policy_mode": "stochastic_normal", "complete": True})


def test_freeze_rejects_changed_inputs_and_uncommitted_manifest(monkeypatch, tmp_path):
    path = tmp_path / "frozen.json"
    frozen = {"files": {"used.py": "hash"}, "protocol": {"version": "v1"}}
    write_json(path, frozen)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    committed = path.read_bytes()
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=committed))
    monkeypatch.setattr(runner, "git", lambda *a, **k: "")
    monkeypatch.setattr(runner, "freeze_record", lambda **k: {**frozen, "files": {"used.py": "hash", "new_checker.py": "new"}})
    assert runner.validate_freeze(path) == frozen  # New independent checker does not alter the baseline.
    monkeypatch.setattr(runner, "freeze_record", lambda **k: {**frozen, "files": {"used.py": "changed"}})
    with pytest.raises(ValueError, match="Frozen experiment mismatch"):
        runner.validate_freeze(path)
    path.write_text(json.dumps({**frozen, "extra": True}))
    with pytest.raises(ValueError, match="committed bytes"):
        runner.validate_freeze(path)
