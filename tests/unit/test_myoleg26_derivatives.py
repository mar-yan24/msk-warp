"""CPU controls for the policy-visited local derivative checker; no GPU launches."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("myoleg26_derivative_check", ROOT / "scripts/check_myoleg26_derivatives.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(ROOT / "msk_warp/assets/myoleg26/flat_boxes.xml"))


@pytest.fixture
def sample(model):
    return {"qpos": model.key_qpos[0].astype(np.float32).astype(float),
            "qvel": np.linspace(-.02, .02, model.nv).astype(np.float32).astype(float),
            "act": np.full(model.na, .2, dtype=np.float32).astype(float),
            "previous_action": np.full(model.nu, -.5),
            "action": np.linspace(-.9, -.1, 4*model.nu).reshape(4, model.nu).astype(np.float32).astype(float)}


def metadata(model, count=50):
    return {"schema_version": "myoleg26-visited-v1", "held_out": True,
            "model_path": "msk_warp/assets/myoleg26/flat_boxes.xml", "substeps": 4,
            "compiled_model_sha256": check.compiled_model_sha256(model),
            "task_contract": check.TASK_CONTRACT.copy(), "primary_strata": ["double_support"],
            "samples": [{"sample_id": str(i), "seed": i % 5, "episode_id": i,
                         "checkpoint_sha256": f"seed{i % 5}", "contact_stratum": "double_support"}
                        for i in range(count)]}


def write_dataset(path, sample, meta):
    n = len(meta["samples"])
    np.savez(path, **{k: np.repeat(v[None], n, axis=0) for k, v in sample.items()}, metadata_json=json.dumps(meta))


def arguments(path):
    return SimpleNamespace(states=str(path), horizons=[1, 4], directions=10, epsilons=list(check.DEFAULT_EPS),
                           direction_seed=1701, forward_rtol=1e-3, forward_atol=1e-4, cosine_min=.99,
                           gradient_rtol=.1, gradient_atol=1e-5, backward_mode="tape_per_substep",
                           native_only=True, native_report=None, device="cuda:0")


def test_compiled_hash_is_deterministic_and_detects_runtime_option_changes(model):
    before = check.compiled_model_sha256(model)
    same = mujoco.MjModel.from_xml_path(str(ROOT / "msk_warp/assets/myoleg26/flat_boxes.xml"))
    assert check.compiled_model_sha256(same) == before
    same.opt.timestep *= 2
    assert check.compiled_model_sha256(same) != before


def test_dataset_quantizes_once_and_preserves_action_window(model, sample, tmp_path):
    sample["qvel"][0] = .123456789123
    path = tmp_path / "states.npz"
    write_dataset(path, sample, metadata(model, 2))
    arrays, meta, loaded, _ = check.load_dataset(path)
    assert arrays["qvel"][0, 0] == float(np.float32(.123456789123))
    assert arrays["action"].shape == (2, 4, 26)
    assert all(a.dtype == np.float64 for a in arrays.values())
    assert check.compiled_model_sha256(loaded) == meta["compiled_model_sha256"]


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m.update(held_out=False), "held-out"),
    (lambda m: m.update(compiled_model_sha256="wrong"), "SHA256"),
    (lambda m: m.update(substeps=1), "four-substep"),
    (lambda m: m.update(primary_strata=[]), "primary_strata"),
    (lambda m: m["samples"][1].update(sample_id="0"), "unique"),
])
def test_dataset_rejects_wrong_contract_and_metadata(model, sample, tmp_path, mutation, match):
    meta = metadata(model, 2)
    mutation(meta)
    path = tmp_path / "states.npz"
    write_dataset(path, sample, meta)
    with pytest.raises(ValueError, match=match):
        check.load_dataset(path)


def test_dataset_rejects_short_windows_nonfinite_and_invalid_quaternions(model, sample, tmp_path):
    path = tmp_path / "states.npz"
    for field, value, match in (("action", sample["action"][:1], "Hmax"),
                               ("act", np.full(26, np.nan), "finite"),
                               ("qpos", np.zeros(47), "normalized")):
        write_dataset(path, {**sample, field: value}, metadata(model, 1))
        with pytest.raises(ValueError, match=match):
            check.load_dataset(path)


def test_coverage_counts_independent_episodes_and_five_seeds(model):
    meta = metadata(model)
    assert check.coverage(meta)["passed"]
    for row in meta["samples"]:
        row["episode_id"] = 0
    assert not check.coverage(meta)["passed"]


def test_coverage_requires_ten_contact_states_even_with_all_seeds_present(model):
    meta = metadata(model, 5)
    result = check.coverage(meta)
    assert result["contact_state_count"] == 5
    assert not result["passed"]
    assert check.coverage(meta)["independent_episode_counts"]["double_support"] == 5
    meta = metadata(model)
    meta["primary_strata"].append("left_support")
    assert check.coverage(meta)["counts"]["left_support"] == 0
    assert "left_support" in check.coverage(meta)["uncovered_strata"]
    assert not check.coverage(meta)["passed"]
    meta = metadata(model)
    meta["samples"] = [r for r in meta["samples"] if r["seed"] != 4]
    assert not check.coverage(meta)["passed"]


def test_qpos_tangent_lift_matches_exact_mujoco_integration_on_random_poses(model):
    rng = np.random.default_rng(53)
    for _ in range(12):
        qpos = model.qpos0.copy()
        quat = rng.normal(size=4)
        qpos[3:7] = (quat / np.linalg.norm(quat)).astype(np.float32)
        direction = rng.normal(size=model.nv)
        plus, minus = qpos.copy(), qpos.copy()
        mujoco.mj_integratePos(model, plus, direction, 1e-6)
        mujoco.mj_integratePos(model, minus, direction, -1e-6)
        np.testing.assert_allclose(check.qpos_tangent_lift(qpos) @ direction, (plus-minus)/2e-6,
                                   atol=2e-10, rtol=2e-8)


def test_internal_tangents_include_every_equality_coordinate_without_projection(model, sample):
    delta = np.linspace(-.01, .01, 40)
    changed = check.perturb_state(model, sample, "qpos_internal_tangent", delta)
    np.testing.assert_allclose(changed["qpos"][7:] - sample["qpos"][7:], delta, atol=1e-15)
    np.testing.assert_array_equal(changed["qvel"], sample["qvel"])
    assert not np.shares_memory(changed["qpos"], sample["qpos"])
    tiny = check.perturb_state(model, sample, "act", np.full(26, 1e-9))
    assert np.all(tiny["act"] != sample["act"]), "FD perturbations must not be rounded back to float32"


def test_full_state_projection_native_torch_parity_and_orientation_gradient(model, sample):
    rng = np.random.default_rng(55)
    qpos = sample["qpos"].copy()
    qpos[3:7] = rng.normal(size=4)
    qpos[3:7] /= np.linalg.norm(qpos[3:7])
    args = [torch.tensor(a[None], dtype=torch.float64, requires_grad=True)
            for a in (qpos, sample["qvel"], sample["act"])]
    state = check.torch_state_features(*args)
    np.testing.assert_allclose(state.detach().numpy()[0], check.native_state_features(qpos, sample["qvel"], sample["act"]), atol=3e-16)
    assert state.shape == (1, 124)
    assert torch.autograd.gradcheck(check.torch_state_features, tuple(args), atol=1e-8)


def test_native_reward_independently_matches_actual_task_on_random_poses(model, sample):
    from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv
    from msk_warp.envs.myoleg26_task import MyoLegTaskContract
    env = MyoLeg26WalkEnv.__new__(MyoLeg26WalkEnv)
    env.mjm, env.model_contract, env.device, env.num_environments = model, "official", "cpu", 1
    env._init_pelvis_kinematics()
    env.up_vec, env.heading_vec = env.up_vec.double(), env.heading_vec.double()
    rng = np.random.default_rng(7)
    for _ in range(8):
        data = mujoco.MjData(model)
        data.qpos[:] = sample["qpos"]
        tangent = rng.normal(size=46) * .15
        mujoco.mj_integratePos(model, data.qpos, tangent, 1)
        data.qvel[:] = rng.normal(size=46)
        excitation = rng.uniform(size=26)
        obs = env._obs_from_state(*(torch.tensor(a[None], dtype=torch.float64) for a in
                                    (data.qpos, data.qvel, sample["act"], 2*excitation-1)))
        terms = MyoLegTaskContract().reward_terms(obs, torch.tensor(excitation[None]), 4*model.opt.timestep)
        expected = [float(terms["locomotion_reward"] - terms["effort_cost"]), float(terms["locomotion_reward"])]
        np.testing.assert_allclose(check.native_reward_components(model, data, excitation, check.TASK_CONTRACT, 4), expected,
                                   atol=1e-15, rtol=2e-12)


def test_native_rollout_is_fresh_and_matches_manual_frozen_actions(model, sample):
    projections = np.random.default_rng(3).normal(size=(2, 124))
    actual = check.native_rollout(model, sample, projections)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:] = sample["qpos"], sample["qvel"], sample["act"]
    for action in sample["action"]:
        data.ctrl[:] = .5 * (action+1)
        for _ in range(4):
            mujoco.mj_step(model, data)
    np.testing.assert_array_equal(actual["history"][-1, :124], check.native_state_features(data.qpos, data.qvel, data.act))
    check.native_rollout(model, {**sample, "action": -sample["action"]}, projections)
    repeat = check.native_rollout(model, sample, projections)
    np.testing.assert_array_equal(actual["terms"], repeat["terms"])
    np.testing.assert_array_equal(actual["history"], repeat["history"])


def test_native_warnings_fail_closed_even_when_mujoco_returns_finite_state(model, sample, monkeypatch):
    original = check.mujoco.mj_step
    def warned(m, data):
        original(m, data)
        data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
    monkeypatch.setattr(check.mujoco, "mj_step", warned)
    result = check.native_rollout(model, sample, np.ones((2, 124)))
    assert np.isnan(result["terms"]).all()
    assert np.isnan(result["history"]).all()


def test_all_four_fd_terms_share_each_evaluation_and_negative_control_is_rejected(model, sample, monkeypatch):
    count = 0
    def analytical(m, state, p):
        nonlocal count
        count += 1
        x = state["act"].sum()
        return {"terms": np.array([x, 2*x, 3*x, 4*x])}
    monkeypatch.setattr(check, "native_rollout", analytical)
    directions = check.sample_directions(1701, 4)["act"]
    blocks = check.native_block_sweeps(model, sample, np.ones((2, 124)), {"act": directions}, (1e-3, 1e-4, 1e-5))
    assert count == 10 * (1+2*3), "evaluate four terms together, not four separate sweeps"
    gradients = {term: {"act": np.full(26, i+1)} for i, term in enumerate(check.TERMS)}
    settings = vars(arguments("unused"))
    assert check.apply_derivative_gates(blocks, gradients, settings)
    gradients["total_task_reward"]["act"] *= -1
    assert not check.apply_derivative_gates(blocks, gradients, settings)
    assert blocks["act"]["terms"]["total_task_reward"]["projected_cosine"] < -.99


def test_direction_dimensions_are_entire_action_window_and_reproducible():
    first = check.sample_directions(2, 4)
    assert first["action"].shape == (10, 104)
    assert first["qpos_root_tangent"].shape == (10, 6)
    assert first["qpos_internal_tangent"].shape == (10, 40)
    for key, values in first.items():
        np.testing.assert_array_equal(values, check.sample_directions(2, 4)[key])
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1)


def test_forward_gate_checks_all_state_fields_all_control_steps_and_nonfinite():
    native = {"terms": np.zeros(4), "history": np.zeros((4, 126))}
    settings = vars(arguments("unused"))
    assert check.forward_gate(native, native, settings)["passed"]
    warp = copy.deepcopy(native)
    warp["history"][0, 123] = .5  # activation at first control step, unchanged terminal projection
    assert not check.forward_gate(native, warp, settings)["passed"]
    warp = copy.deepcopy(native)
    warp["terms"][0] = np.nan
    assert not check.forward_gate(native, warp, settings)["passed"]


def test_native_cache_rejects_hash_settings_and_source_changes(model, sample, tmp_path):
    path = tmp_path / "states.npz"
    meta = metadata(model, 1)
    write_dataset(path, sample, meta)
    report = check.build_report(arguments(path), meta, model)
    cached = copy.deepcopy(report)
    cached["gate_status"] = "native_complete_awaiting_ad"
    row = {"sample_id": "0", "metadata": meta["samples"][0], "horizons": {}}
    for horizon in (1, 4):
        row["horizons"][str(horizon)] = {
            "native": {"history": np.zeros((horizon, 126)).tolist()},
            "blocks": {block: {"directions": vectors.tolist(), "epsilon_sweeps":
                               [{"epsilons": list(check.DEFAULT_EPS), "windows": [None]*4} for _ in range(10)]}
                       for block, vectors in check.sample_directions(1, horizon).items()}}
    cached["samples"] = [row]
    check.validate_native_cache(cached, report)
    for key, subkey in (("dataset", "sha256"), ("settings", "directions"), ("provenance", "compiled_model_sha256")):
        changed = copy.deepcopy(cached)
        changed[key][subkey] = "wrong"
        with pytest.raises(ValueError, match=key):
            check.validate_native_cache(changed, report)
    changed = copy.deepcopy(cached)
    changed["samples"][0]["sample_id"] = "incorrect-row"
    with pytest.raises(ValueError, match="order/identity"):
        check.validate_native_cache(changed, report)


@pytest.mark.parametrize("key,value", [("cosine_min", .5), ("gradient_rtol", .2), ("gradient_atol", 1e-4),
                                       ("forward_rtol", .01), ("forward_atol", 1e-3), ("cosine_min", np.nan)])
def test_thresholds_cannot_be_weakened(key, value):
    args = arguments("unused")
    setattr(args, key, value)
    with pytest.raises(ValueError, match=key):
        check.validate_settings(args)


def test_aggregate_bootstrap_uses_episode_clusters_and_reports_unresolved_denominators():
    samples = []
    for i in range(13):
        term = ({"passed": True, "projected_cosine": .99-i*.001, "relative_l2": i*.001}
                if i < 11 else {"passed": False, "reason": "no_epsilon_window"})
        samples.append({"metadata": {"episode_id": 30000+i, "seed": 0},
                        "horizons": {"1": {"blocks": {"act": {"terms": {"locomotion_only": term}}}}}})
    result = check.aggregate_state_gates(samples, [1], seed=42)
    summary = result["horizons"]["1"]["act"]["locomotion_only"]
    assert summary["selected_state_count"] == 13
    assert summary["indeterminate_state_count"] == 2
    assert summary["projected_cosine"]["state_count"] == 11
    assert summary["projected_cosine"]["resolved_episode_id_count"] == 11
    assert summary["projected_cosine"]["mean"] == pytest.approx(.985)
    low, high = summary["projected_cosine"]["mean_ci95"]
    assert low < .985 < high
    assert not result["pass_override"]
    assert result == check.aggregate_state_gates(samples, [1], seed=42)
    short = check.aggregate_state_gates(samples[:4], [1])["horizons"]["1"]["act"]["locomotion_only"]
    assert short["projected_cosine"]["mean_ci95"] is None


def test_reusing_reset_seeds_across_policies_does_not_artificially_narrow_cluster_interval():
    samples = [{"metadata": {"episode_id": 30000+i, "seed": 0},
                "horizons": {"1": {"blocks": {"act": {"terms": {"locomotion_only": {
                    "passed": True, "projected_cosine": .7+i*.02, "relative_l2": .2-i*.01,
                }}}}}}} for i in range(10)]
    original = check.aggregate_state_gates(samples, [1], seed=42)
    repeated = [copy.deepcopy(s) for policy_seed in range(5) for s in samples]
    for i, row in enumerate(repeated):
        row["metadata"]["seed"] = i//10
    pooled = check.aggregate_state_gates(repeated, [1], seed=42)
    assert pooled["selected_state_count"] == 50
    assert pooled["unique_episode_id_count"] == 10
    assert pooled["policy_seeds"] == list(range(5))
    first = original["horizons"]["1"]["act"]["locomotion_only"]["projected_cosine"]
    second = pooled["horizons"]["1"]["act"]["locomotion_only"]["projected_cosine"]
    np.testing.assert_allclose(first["mean_ci95"], second["mean_ci95"], atol=1e-15)
    assert second["state_count"] == 50 and second["resolved_episode_id_count"] == 10


def test_native_only_never_builds_warp_and_reports_awaiting_ad(model, sample, tmp_path, monkeypatch):
    path = tmp_path / "states.npz"
    write_dataset(path, sample, metadata(model, 1))
    def forbidden(*args, **kwargs):
        pytest.fail("native-only must never import/initialize a Warp environment")
    monkeypatch.setattr(check, "build_warp_environment", forbidden)
    monkeypatch.setattr(check, "native_block_sweeps", lambda *args: {})
    report = check.run(arguments(path))
    assert not report["passed"]
    assert report["gate_status"] == "native_complete_awaiting_ad"
    assert set(report["samples"][0]["horizons"]) == {"1", "4"}


def test_forward_failure_prevents_any_backward_or_fd_work(model, sample, tmp_path, monkeypatch):
    path = tmp_path / "states.npz"
    write_dataset(path, sample, metadata(model, 1))
    args = arguments(path)
    args.native_only = False
    monkeypatch.setattr(check, "build_warp_environment", lambda *args: object())
    calls = []
    def wrong_forward(env, sample, projections, *, gradients=False):
        calls.append(gradients)
        native = check.native_rollout(model, sample, projections)
        native["history"][0, 5] += 5
        return native
    monkeypatch.setattr(check, "warp_rollout", wrong_forward)
    monkeypatch.setattr(check, "native_block_sweeps", lambda *args: pytest.fail("FD should not follow parity failure"))
    report = check.run(args)
    assert report["gate_status"] == "forward_mismatch"
    assert calls == [False, False]
    assert report["forward_summary"]["tested"] == 2
    assert report["forward_summary"]["failed"] == 2


@pytest.mark.parametrize("failure", ["finite_clipping", "unobserved_backward"])
def test_modified_or_unobserved_ad_cannot_pass_even_when_returned_gradients_match_fd(
    model, sample, tmp_path, monkeypatch, failure
):
    path = tmp_path / "states.npz"
    write_dataset(path, sample, metadata(model, 10))
    args = arguments(path)
    args.native_only = False
    monkeypatch.setattr(check, "build_warp_environment", lambda *args: object())
    def reference(m, sample, projections):
        return {"history": np.zeros((len(sample["action"]), 126)), "terms": np.zeros(4)}
    monkeypatch.setattr(check, "native_rollout", reference)
    def fake_warp(env, sample, projections, *, gradients=False):
        result = reference(model, sample, projections)
        if gradients:
            h = len(sample["action"])
            result["gradients"] = {term: {b: np.zeros(n) for b, n in zip(check.BLOCKS, (h*26, 6, 40, 46, 26))}
                                   for term in check.TERMS}
            result["sanitization"] = {term: {"calls": h, "nonfinite_entries": 0, "clamped_finite_entries": 0}
                                      for term in check.TERMS}
            result["sanitization"]["total_task_reward"].update(
                {"clamped_finite_entries": 1} if failure == "finite_clipping" else {"calls": 0})
        return result
    monkeypatch.setattr(check, "warp_rollout", fake_warp)
    def sweeps(m, sample, projections, directions, epsilons):
        return {block: {"directions": vectors.tolist(), "epsilon_sweeps":
                        [{"windows": [{"derivative": 0.0} for _ in check.TERMS]} for _ in vectors], "terms": {}}
                for block, vectors in directions.items()}
    monkeypatch.setattr(check, "native_block_sweeps", sweeps)
    # The numerical comparison has separate positive/negative controls above.
    # Isolate this regression's decision: even a passed oracle cannot excuse clipping.
    monkeypatch.setattr(check._helpers, "compare_derivatives", lambda *args, **kwargs: {"passed": True, "signal": "no_signal"})
    report = check.run(args)
    assert report["coverage"]["passed"] and report["forward_summary"]["passed"]
    assert not report["passed"]
    for h in report["samples"][0]["horizons"].values():
        assert not h["unmodified_ad"]
        assert all(t["passed"] for b in h["blocks"].values() for t in b["terms"].values())


def test_existing_output_is_preserved_before_measurement(tmp_path, monkeypatch):
    path = tmp_path / "existing.json"
    path.write_text("original research evidence")
    monkeypatch.setattr("sys.argv", ["checker", "--states", "unused", "--out", str(path)])
    monkeypatch.setattr(check, "run", lambda args: pytest.fail("must not run"))
    with pytest.raises(SystemExit) as result:
        check.main()
    assert result.value.code == 2
    assert path.read_text() == "original research evidence"
