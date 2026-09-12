"""Independent CPU checks of the trajopt diagnostic and its rejection criteria."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_trajopt_gradients", ROOT / "scripts/check_trajopt_gradients.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_parameter_seeds_match_mechanics_across_actuator_counts_and_cycles():
    motor_u, motor_z = check.sample_parameters(17, 16, 3)
    muscle_u, muscle_z = check.sample_parameters(17, 16, 6)
    _, longer_z = check.sample_parameters(17, 27, 6)
    assert motor_u.shape == (16, 3) and muscle_u.shape == (16, 6)
    np.testing.assert_array_equal(motor_z, muscle_z)
    np.testing.assert_array_equal(motor_z, longer_z)
    repeat_u, repeat_z = check.sample_parameters(17, 16, 3)
    np.testing.assert_array_equal(motor_u, repeat_u)
    np.testing.assert_array_equal(motor_z, repeat_z)
    other_u, other_z = check.sample_parameters(18, 16, 3)
    assert not np.array_equal(motor_z, other_z)
    assert not np.array_equal(motor_u, other_u)


def test_existing_output_is_preserved_before_any_measurement_unless_explicitly_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "prior.json"
    path.write_text("original research artifact")
    arguments = ["check_trajopt_gradients.py", "--cfg", "unused", "--scales", "unused", "--out", str(path)]
    monkeypatch.setattr("sys.argv", arguments)
    calls = []
    monkeypatch.setattr(check, "run", lambda args: calls.append(args) or {"passed": True})
    with pytest.raises(SystemExit) as error:
        check.main()
    assert error.value.code == 2
    assert not calls
    assert path.read_text() == "original research artifact"
    monkeypatch.setattr("sys.argv", arguments + ["--overwrite"])
    assert check.main() == 0
    assert len(calls) == 1 and json.loads(path.read_text())["passed"]


@pytest.mark.parametrize("muscle", [False, True])
def test_native_reference_matches_manual_mujoco_steps_and_is_fresh(muscle):
    name = "muscle" if muscle else "motor"
    model = mujoco.MjModel.from_xml_path(str(ROOT / f"msk_warp/assets/hopper_{name}.xml"))
    u = np.linspace(-0.4, 0.7, 3 * model.nu).reshape(3, model.nu)
    z, scales, strength = np.zeros(11), np.linspace(0.2, 2.0, 11), 0.7
    data = mujoco.MjData(model)
    # Hand-written neutral parameters; do not call the reference parameterisation.
    data.qpos[:] = [0.0, 0.0, 0.0, -0.6, -0.6, 0.0]
    data.qvel[:] = 0
    data.act[:] = 0.5
    violation = 0.0
    for step in range(6):
        if step == 3:
            start_x = data.qpos[0]
            start = np.r_[data.qpos[1:], data.qvel]
        action = np.tanh(u[step % 3])
        data.ctrl[:] = (0.5 * (action + 1) if muscle else action) * strength
        mujoco.mj_step(model, data)
        mujoco.mj_step(model, data)
        data.qpos[:] = np.clip(data.qpos, -100, 100)
        data.qvel[:] = np.clip(data.qvel, -100, 100)
        violation += max(0, -0.45 - data.qpos[1])
    expected = np.r_[(data.qpos[0] - start_x) / (6 * model.opt.timestep),
                     (np.r_[data.qpos[1:], data.qvel] - start) / scales, violation / 6]
    kwargs = dict(substeps=2, warmup=1, muscle=muscle, strength=strength)
    actual = check.native_rollout(model, u, z, scales, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
    check.native_rollout(model, -u, np.ones(11), scales, **kwargs)
    np.testing.assert_array_equal(actual, check.native_rollout(model, u, z, scales, **kwargs))


def test_native_parameterisation_and_action_maps_match_actual_driver():
    trajopt = check.load_trajopt()
    from msk_warp.envs.hopper import HopperMotorEnv, HopperMuscleEnv
    rng = np.random.default_rng(311)
    for _ in range(10):
        z = rng.normal(size=11)
        expected = trajopt.initial_state(torch.tensor(z[None], dtype=torch.float64), 6, "cpu")
        actual = check.native_initial_state(z, 6)
        for cpu, target in zip(actual, expected):
            np.testing.assert_allclose(cpu, target[0].numpy(), atol=1e-14, rtol=1e-14)
    logits = torch.linspace(-8, 8, 30, dtype=torch.float64).reshape(5, 6)
    fake = SimpleNamespace(action_strength=0.71)
    for cls, muscle in ((HopperMotorEnv, False), (HopperMuscleEnv, True)):
        action = np.tanh(logits.numpy())
        native = (0.5 * (action + 1) if muscle else action) * fake.action_strength
        np.testing.assert_allclose(cls._to_ctrl(fake, logits.tanh()).numpy(), native, atol=1e-14)


def test_native_finite_differences_converge_in_a_short_smooth_rollout():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "msk_warp/assets/hopper_motor.xml"))
    rng = np.random.default_rng(81)
    z, u = np.zeros(11), rng.normal(scale=0.1, size=(2, 3))
    direction = rng.normal(size=11)
    direction /= np.linalg.norm(direction)
    function = lambda x: check.native_rollout(model, u, x, np.ones(11), substeps=1, warmup=0)
    sweep = check.epsilon_sweep(function, z, direction)
    derivative = np.asarray(sweep["central"])
    coarse = np.linalg.norm(derivative[0] - derivative[-1])
    refined = np.linalg.norm(derivative[6] - derivative[-1])
    assert coarse > 1e-7
    assert refined < coarse / 100
    assert all(window is not None for window in sweep["windows"])


def test_epsilon_window_checks_one_sided_derivatives_to_reject_a_kink():
    sweep = check.epsilon_sweep(lambda x: np.array([abs(x[0])]), np.zeros(1), np.ones(1))
    assert np.all(np.asarray(sweep["central"]) == 0)
    assert sweep["windows"] == [None]


def test_epsilon_window_rejects_coarse_plateau_with_contradictory_finer_evidence():
    # True derivative is 2. A seemingly stable coarse plateau near 1 must not certify AD=1.
    width = 1e-7
    function = lambda x: np.array([x[0] + width * np.tanh(x[0] / width)])
    unresolved = check.epsilon_sweep(function, np.zeros(1), np.ones(1))
    assert unresolved["central"][0][0] == pytest.approx(1.00001)
    assert unresolved["central"][-1][0] == pytest.approx(1 + np.tanh(1))
    assert unresolved["windows"] == [None]
    # With adequate resolution the same smooth transition is certifiable at derivative 2.
    epsilons = (*check.DEFAULT_EPS, 3e-8, 1e-8, 3e-9, 1e-9, 3e-10, 1e-10)
    resolved = check.epsilon_sweep(function, np.zeros(1), np.ones(1), epsilons)
    window = resolved["windows"][0]
    assert window is not None and window["epsilons"] == list(epsilons[-3:])
    assert window["derivative"] == pytest.approx(2, abs=1e-6)
    assert not check.compare_derivatives(np.ones(10), np.full(10, window["derivative"]))["passed"]


def test_directional_gate_accepts_correct_and_rejects_deliberately_wrong_derivative():
    rng = np.random.default_rng(23)
    x, gradient = np.array([0.4, -0.7]), np.array([0.8, -2.1])
    directions = rng.normal(size=(10, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    function = lambda y: np.array([y[0] ** 2 + 1.5 * y[1] ** 2])
    fd = [check.epsilon_sweep(function, x, d)["windows"][0]["derivative"] for d in directions]
    true = directions @ gradient
    assert check.compare_derivatives(true, fd)["passed"]
    assert not check.compare_derivatives(-true, fd)["passed"]
    assert not check.compare_derivatives(1.3 * true, fd)["passed"]
    assert not check.compare_derivatives(np.zeros(10), fd)["passed"]


def test_zero_sensitivity_is_no_signal_not_a_cosine_claim():
    result = check.compare_derivatives(np.zeros(10), np.zeros(10))
    assert result["passed"] and result["signal"] == "no_signal"
    assert result["projected_cosine"] is None
    assert not check.compare_derivatives(np.ones(10), np.zeros(10))["passed"]


def test_sanitization_observer_preserves_results_and_restores_after_exception():
    def original(*tensors):
        return tuple(torch.nan_to_num(t, nan=0, posinf=0, neginf=0).clamp(-10, 10) for t in tensors)
    bridge = SimpleNamespace(_sanitize=original, GRAD_CLAMP=10)
    tensor = torch.tensor([float("nan"), float("inf"), -11, 12, 3])
    with pytest.raises(RuntimeError, match="test exception"):
        with check.count_sanitization(bridge) as counts:
            torch.testing.assert_close(bridge._sanitize(tensor)[0], original(tensor)[0])
            raise RuntimeError("test exception")
    assert bridge._sanitize is original
    assert counts == {"calls": 1, "nonfinite_entries": 2, "clamped_finite_entries": 2}


def test_augmented_objective_uses_vector_squared_norm_and_height_penalty():
    values, projections = np.arange(13, dtype=float) / 5, np.eye(11)[:2]
    y, rho, mu = np.linspace(-1, 1, 11), 7, 20
    terms = check.project_terms(values, projections, auglag=True, multipliers=y, rho=rho, mu=mu)
    expected = values[0] - y @ values[1:12] - rho / 2 * np.sum(values[1:12] ** 2) - mu * values[-1]
    np.testing.assert_allclose(terms, [values[0], values[1], values[2], expected])
