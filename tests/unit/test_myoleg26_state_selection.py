"""Selection gates must enforce independent episodes without inspecting AD values."""
import importlib.util
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("selector", Path(__file__).resolve().parents[2] / "scripts/select_myoleg26_states.py")
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def test_assignment_balances_seeds_and_primary_strata_deterministically():
    episodes = {(seed, f"hash{seed}", episode): set(selector.SUPPORT)
                for seed in range(5) for episode in range(16)}
    actual = selector.assign_episodes(episodes, list(selector.SUPPORT))
    assert actual == selector.assign_episodes(episodes, list(selector.SUPPORT))
    assert len(actual) == 50
    for seed in range(5):
        assert sum(key[0] == seed for key in actual) == 10
    for stratum in selector.SUPPORT:
        assert list(actual.values()).count(stratum) >= 10


def test_assignment_uses_residual_rerouting_for_overlapping_episode_modes():
    episodes = {(seed, f"hash{seed}", episode): {"airborne"}
                for seed in range(5) for episode in range(10)}
    for seed in range(5):
        for episode in range(4):
            episodes[seed, f"hash{seed}", episode] = {"left_support", "right_support"}
    actual = selector.assign_episodes(episodes, ["left_support", "right_support"])
    assert actual is not None
    assert list(actual.values()).count("left_support") == 10
    assert list(actual.values()).count("right_support") == 10


def test_overlapping_strata_do_not_fake_independent_coverage():
    episodes = {(seed, f"hash{seed}", episode): {"airborne"}
                for seed in range(5) for episode in range(10)}
    for seed in range(5):
        for episode in range(2):
            episodes[seed, f"hash{seed}", episode] = set(selector.SUPPORT)
    assert selector.assign_episodes(episodes, list(selector.SUPPORT)) is None


def test_contact_minimum_and_seed_quota_cannot_be_relaxed():
    episodes = {(seed, f"hash{seed}", episode): {"airborne"}
                for seed in range(5) for episode in range(10)}
    assert selector.assign_episodes(episodes, []) is None
    episodes = {key: {"double_support"} for key in episodes if key != (0, "hash0", 0)}
    assert selector.assign_episodes(episodes, ["double_support"]) is None


def test_missing_seed_rejected():
    with pytest.raises(ValueError, match="five"):
        selector.assign_episodes({(0, "hash", 30000): {"double_support"}}, [])


def test_contact_classifier_uses_margin_and_nonfoot_precedence():
    xml = '''<mujoco><worldbody><geom name="task_ground" type="plane" size="1 1 .1"/>
      <body name="calcn_l"><geom name="task_collision_calcn_l" type="sphere" size=".1"/></body>
      <body name="toes_r"><geom name="task_collision_toes_r" type="sphere" size=".1"/></body>
      <body name="pelvis"><geom name="task_collision_pelvis" type="sphere" size=".1"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    def contact(geom, dist=-.01):
        return SimpleNamespace(geom1=0, geom2=model.geom(geom).id, dist=dist, includemargin=.001)
    left = contact("task_collision_calcn_l")
    right = contact("task_collision_toes_r")
    torso = contact("task_collision_pelvis")
    def classify(contacts):
        return selector.contact_stratum(model, SimpleNamespace(contact=contacts, ncon=len(contacts)))
    assert classify([]) == "airborne"
    assert classify([left]) == "left_support"
    assert classify([right]) == "right_support"
    assert classify([left, right]) == "double_support"
    assert classify([left, right, torso]) == "nonfoot_contact"
    assert classify([contact("task_collision_calcn_l", .001)]) == "airborne"


def test_compiled_model_hash_detects_physics_change():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="free"/><geom size=".1"/></body></worldbody></mujoco>')
    before = selector.compiled_hash(model)
    assert selector.compiled_hash(model) == before
    model.body_mass[1] *= 2
    assert selector.compiled_hash(model) != before


def make_trace_run(tmp_path, monkeypatch, lengths=None):
    """Real native model and pickle-free traces; mock only the already-tested Git freeze gate."""
    model_relative = "msk_warp/assets/myoleg26/flat_boxes.xml"
    model_path = selector.baseline_runner.ROOT / model_relative
    model = mujoco.MjModel.from_xml_path(str(model_path))
    frozen = {"protocol": copy.deepcopy(selector.baseline_runner.PROTOCOL), "files": {"source.py": "sourcehash"},
              "task_contract": selector.baseline_runner.MyoLegTaskContract().as_dict(),
              "model_path": model_relative, "model_sha256": selector.sha256(model_path),
              "compiled_model_sha256": selector.compiled_hash(model)}
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    freeze_sha = selector.sha256(freeze_path)
    monkeypatch.setattr(selector.baseline_runner, "validate_freeze", lambda path: frozen)
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "run_manifest.json").write_text(json.dumps({
        "freeze": frozen, "freeze_sha256": freeze_sha, "seeds": list(range(5)), "epochs": 128}), encoding="utf-8")
    paths = []
    for seed, length in enumerate(lengths or [9] * 5):
        directory = run_path / f"seed_{seed}"
        directory.mkdir()
        checkpoint = directory / "epoch_0000.pt"
        checkpoint.write_bytes(f"test checkpoint {seed}".encode())
        checkpoint_sha = selector.sha256(checkpoint)
        (directory / "result.json").write_text(json.dumps({"seed": seed, "completed_epochs": 128,
            "selected_checkpoint": {"epoch": 0, "sha256": checkpoint_sha, "filename": checkpoint.name}}), encoding="utf-8")
        worlds = np.tile(np.arange(16), length)
        steps = np.repeat(np.arange(length), 16)
        qpos = np.repeat(model.key("stand").qpos[None].astype(np.float32), len(worlds), axis=0)
        qpos[:, 0] = seed * 16 + worlds  # Native classifier tests can identify independent candidates.
        previous_action = np.zeros((len(worlds), 26), np.float32)
        previous_action[steps == 0] = -1
        arrays = {"qpos": qpos, "qvel": np.zeros((len(worlds), 46), np.float32),
                  "act": np.zeros((len(worlds), 26), np.float32), "previous_action": previous_action,
                  "action": np.zeros((len(worlds), 26), np.float32), "world": worlds,
                  "episode_seed": worlds + 30000, "step": steps,
                  "done": steps == length - 1, "terminated": steps == length - 1}
        metadata = {"schema_version": "myoleg26-raw-traces-v1", "training_seed": seed,
                    "policy_mode": "stochastic_normal", "complete": True, "warmstart_disabled": True,
                    "protocol": frozen["protocol"], "frozen_files": frozen["files"],
                    "task_contract": frozen["task_contract"], "substeps": 4,
                    "model_path": str(model_path), "model_sha256": frozen["model_sha256"],
                    "compiled_model_sha256": frozen["compiled_model_sha256"],
                    "checkpoint_epoch": 0, "checkpoint_sha256": checkpoint_sha}
        path = directory / "stochastic_audit_traces.npz"
        np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(metadata)))
        paths.append(path)
    return paths, freeze_path, frozen


def rewrite_metadata(path, key, value):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
        metadata = json.loads(archive["metadata_json"].item())
    metadata[key] = value
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(metadata)))


@pytest.mark.parametrize("field,value", [
    ("protocol", {"version": "stale"}), ("frozen_files", {"source.py": "changed"}),
    ("task_contract", {"version": "different"}), ("substeps", 2),
    ("model_sha256", "wrong"), ("compiled_model_sha256", "wrong"),
])
def test_trace_must_match_committed_freeze(tmp_path, monkeypatch, field, value):
    paths, freeze_path, frozen = make_trace_run(tmp_path, monkeypatch)
    rewrite_metadata(paths[0], field, value)
    with pytest.raises(ValueError, match="committed freeze"):
        selector.load_trace(paths[0], frozen=frozen, freeze_sha256=selector.sha256(freeze_path))


@pytest.mark.parametrize("field", ["seed", "epoch", "sha256", "filename"])
def test_trace_must_use_its_seed_selected_checkpoint(tmp_path, monkeypatch, field):
    paths, freeze_path, frozen = make_trace_run(tmp_path, monkeypatch)
    result_path = paths[0].parent / "result.json"
    result = json.loads(result_path.read_text())
    if field == "seed":
        result["seed"] = 4
    else:
        result["selected_checkpoint"][field] = 32 if field == "epoch" else "different"
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="behavior-selected checkpoint"):
        selector.load_trace(paths[0], frozen=frozen, freeze_sha256=selector.sha256(freeze_path))


def test_trace_parent_manifest_and_checkpoint_bytes_are_verified(tmp_path, monkeypatch):
    paths, freeze_path, frozen = make_trace_run(tmp_path, monkeypatch)
    manifest_path = paths[0].parent.parent / "run_manifest.json"
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest["freeze_sha256"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Parent run manifest"):
        selector.load_trace(paths[0], frozen=frozen, freeze_sha256=selector.sha256(freeze_path))
    manifest_path.write_text(original)
    (paths[0].parent / "epoch_0000.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        selector.load_trace(paths[0], frozen=frozen, freeze_sha256=selector.sha256(freeze_path))


def test_selection_requires_freeze_validation_before_classification(tmp_path, monkeypatch):
    def reject(path):
        raise ValueError("uncommitted canonical freeze")
    monkeypatch.setattr(selector.baseline_runner, "validate_freeze", reject)
    with pytest.raises(ValueError, match="uncommitted canonical freeze"):
        selector.select([], tmp_path / "selected.npz", frozen_manifest=tmp_path / "freeze.json")
    assert not (tmp_path / "selected.npz").exists()


def test_missing_eligible_seed_writes_blocked_report(tmp_path, monkeypatch):
    paths, freeze_path, _ = make_trace_run(tmp_path, monkeypatch, [9, 9, 9, 9, 4])
    monkeypatch.setattr(selector, "contact_stratum", lambda model, data: "double_support")
    output = tmp_path / "selected.npz"
    report = selector.select(paths, output, frozen_manifest=freeze_path)
    assert report["selection_status"] == "blocked_missing_eligible_seed"
    assert report["eligible_training_seeds"] == [0, 1, 2, 3]
    assert json.loads(output.with_suffix(".json").read_text())["selection_status"] == report["selection_status"]
    assert not output.exists()


def test_empty_primary_strata_writes_blocked_report(tmp_path, monkeypatch):
    paths, freeze_path, _ = make_trace_run(tmp_path, monkeypatch)
    def classify(model, data):
        candidate = int(round(data.qpos[0]))
        return selector.SUPPORT[candidate // 9] if candidate < 27 else "airborne"
    monkeypatch.setattr(selector, "contact_stratum", classify)
    output = tmp_path / "selected.npz"
    report = selector.select(paths, output, frozen_manifest=freeze_path)
    assert report["selection_status"] == "blocked_no_primary_support_stratum"
    assert report["primary_strata"] == []
    assert all(report["eligible_independent_episodes_by_stratum"][stratum] == 9 for stratum in selector.SUPPORT)
    assert output.with_suffix(".json").is_file() and not output.exists()


def test_selected_archive_preserves_provenance_and_independent_episode_quotas(tmp_path, monkeypatch):
    paths, freeze_path, _ = make_trace_run(tmp_path, monkeypatch)
    monkeypatch.setattr(selector, "contact_stratum", lambda model, data: "double_support")
    output = tmp_path / "selected.npz"
    report = selector.select(paths, output, frozen_manifest=freeze_path)
    assert report["selection_status"] == "pass"
    assert report["frozen_manifest_sha256"] == selector.sha256(freeze_path)
    assert len(report["source_seed_results"]) == 5 and len(report["source_run_manifests"]) == 1
    with np.load(output, allow_pickle=False) as archive:
        assert archive["qpos"].shape == (50, 47) and archive["action"].shape == (50, 4, 26)
        metadata = json.loads(archive["metadata_json"].item())
        assert len({(s["seed"], s["checkpoint_sha256"], s["episode_id"]) for s in metadata["samples"]}) == 50
        assert all(sum(s["seed"] == seed for s in metadata["samples"]) == 10 for seed in range(5))
    with pytest.raises(FileExistsError):
        selector.select(paths, output, frozen_manifest=freeze_path)
