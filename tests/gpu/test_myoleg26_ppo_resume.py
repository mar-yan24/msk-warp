"""MyoLeg26 continuation segments on the real simulator (Unit 3).

The CPU suite (``tests/unit/test_ppo_resume_state.py``) binds the schema, the
publication mechanism and the omission controls against an analytic environment.
It cannot speak for the simulator. These tests are the ones that decide whether
a restored segment is the same training state on the pinned Warp backend:

* the round trip is compared over the **whole** captured state — actor and
  critic, both optimizers, the normaliser including its count, both meters
  including ``current_size``, every counter, history list and rollout buffer,
  every env torch buffer including the lazily created ``obs_buf_before_reset``,
  all six integration inputs and the ``eq_active`` invariant — into a target that
  has only been reset, never stepped (the shape that exposed review defect D1);
* the RNG streams, including CUDA, are restored, and the **next sampled**
  pre-tanh action is bitwise identical;
* identity gates (recipe, compiled model, lineage via ``strict_expect``) refuse
  before the target is mutated;
* the CUDA RNG omission control lives here, not on CPU;
* the derived Data scratch fields this schema omits are shown not to change the
  next step — the only empirical enforcement of that omission.

Nothing here trains: two tiny epochs at ``num_actors=2`` are plumbing, charged
to test accounting, not to the training or diagnostic budgets.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest
import torch
import warp as wp
import yaml

from msk_warp.algorithms.ppo import PPO
from msk_warp.analysis import ppo_resume

CONFIG = Path(__file__).resolve().parents[2] / "msk_warp/configs/experiments/myoleg26_ppo.yaml"
EPOCHS = 2


def _algo(logdir, seed):
    """A real PPO on the frozen official task, sized down to plumbing scale."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["params"]["general"].update(seed=seed, device="cuda:0", logdir=str(logdir))
    cfg["params"]["env"].update(num_actors=2, episode_length=4, stochastic_init=True)
    cfg["params"]["config"].update(
        steps_num=4, max_epochs=EPOCHS, ppo_epochs=1, num_minibatches=2, save_interval=0)
    cfg["params"]["network"]["actor_mlp"]["units"] = [16, 16]
    cfg["params"]["network"]["critic_mlp"]["units"] = [16, 16]
    algo = PPO(cfg)
    algo.env.begin_epoch(epoch=0, max_epochs=EPOCHS)
    algo._current_obs = algo.env.reset()
    return algo


def _train(algo, epochs):
    """The v1 external-driver shape (``run_myoleg26_baseline.py:246-253``)."""
    for epoch in range(epochs):
        algo.env.begin_epoch(epoch=epoch, max_epochs=EPOCHS)
        algo.collect_rollout()
        algo.update()
        algo.iter_count += 1


def _next_sampled_action(algo):
    """The next pre-tanh sample: actor parameters, normaliser and CUDA RNG combined."""
    with torch.no_grad():
        obs = algo._current_obs
        obs_norm = algo.obs_rms.normalize(obs) if algo.obs_rms is not None else obs
        pre_tanh, mu, std = algo.actor.forward_with_dist(obs_norm)
    return pre_tanh.clone(), mu.clone(), std.clone()


def _warp_fields(env):
    wp.synchronize()
    return {name: wp.to_torch(getattr(env.warp_data, name)).clone()
            for name in ppo_resume.WARP_STATE_FIELDS}


def _full_state(algo, env):
    """Every restored quantity, not a subset: acceptance 3 says "all state tensors".

    Actor *and* critic, both optimizers in full, the normaliser including its
    count, both meters including ``current_size``, all counters and history,
    every rollout buffer, every env torch buffer (incl. the lazily created
    ``obs_buf_before_reset``), all six captured integration inputs, the
    ``eq_active`` invariant, and both RNG streams.
    """
    out = {}
    for label, module in (("actor", algo.actor), ("critic", algo.critic)):
        for key, value in module.state_dict().items():
            out[f"{label}.{key}"] = value.detach().clone()
    for label, optimizer in (("actor_opt", algo.actor_optimizer),
                             ("critic_opt", algo.critic_optimizer)):
        state = optimizer.state_dict()
        out[f"{label}.param_groups"] = repr(state["param_groups"])
        for index, entry in sorted(state["state"].items()):
            for key, value in sorted(entry.items()):
                out[f"{label}.state.{index}.{key}"] = (
                    value.detach().clone() if torch.is_tensor(value) else value)
    for label in ("obs_rms", "ret_rms"):
        rms = getattr(algo, label)
        if rms is None:
            out[label] = None
        else:
            out[f"{label}.mean"] = rms.mean.clone()
            out[f"{label}.var"] = rms.var.clone()
            out[f"{label}.count"] = rms.count
    for label in ("episode_loss_meter", "episode_length_meter"):
        meter = getattr(algo, label)
        out[f"{label}.mean"] = meter.mean.clone()
        out[f"{label}.current_size"] = meter.current_size
    for name in ppo_resume.ALGO_BUFFERS:
        out[name] = getattr(algo, name).detach().clone()
    out["iter_count"] = algo.iter_count
    out["step_count"] = algo.step_count
    out["episode_loss"] = algo.episode_loss.clone()
    out["episode_length"] = algo.episode_length.clone()
    out["episode_loss_his"] = list(algo.episode_loss_his)
    out["episode_length_his"] = list(algo.episode_length_his)
    out["best_policy_loss"] = float(algo.best_policy_loss)
    out["_current_obs"] = algo._current_obs.detach().clone()
    for name in ppo_resume.ENV_TORCH_BUFFERS + ppo_resume.ENV_LAZY_BUFFERS:
        value = getattr(env, name, None)
        out[f"env.{name}"] = None if value is None else value.detach().clone()
    for name, tensor in _warp_fields(env).items():
        out[f"warp.{name}"] = tensor
    for name in ppo_resume.WARP_INVARIANT_FIELDS:
        array = getattr(env.warp_data, name, None)
        if array is not None:
            out[f"warp.{name}"] = wp.to_torch(array).clone()
    out["rng.torch_cpu"] = torch.get_rng_state().clone()
    for index, state in enumerate(torch.cuda.get_rng_state_all()):
        out[f"rng.cuda{index}"] = state.clone()
    return out


def _differences(left, right):
    assert set(left) == set(right)
    bad = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if torch.is_tensor(a) or torch.is_tensor(b):
            if not (torch.is_tensor(a) and torch.is_tensor(b)):
                bad.append(key)
            elif a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                bad.append(key)
        elif a != b:
            bad.append(key)
    return bad


def _extra(**overrides):
    base = {"seed": 0, "segment_index": 1, "parent_segment_sha256": None,
            "unit": "task-3-gpu-plumbing"}
    base.update(overrides)
    return base


@pytest.fixture
def origin(tmp_path):
    algo = _algo(tmp_path / "origin", seed=0)
    try:
        yield algo
    finally:
        algo.close()


def test_captured_dtypes_are_the_actual_backing_array_dtypes(origin):
    """Addendum 1: record what the arrays actually declare; never cast to a claim."""
    state = ppo_resume.capture_state(origin, origin.env, _extra(), epoch=0)
    fields = state.env["warp"]["fields"]
    observed = {}
    for name in ppo_resume.WARP_STATE_FIELDS:
        view = wp.to_torch(getattr(origin.env.warp_data, name))
        observed[name] = str(view.dtype)
        assert fields[name]["dtype"] == str(view.dtype)
        assert tuple(fields[name]["shape"]) == tuple(view.shape)
        assert fields[name]["values"].dtype == view.dtype
    # The addendum's declaration is float32 for all six. A different runtime
    # dtype is a finding to report, not something to cast away.
    assert set(observed.values()) == {"torch.float32"}, observed
    assert state.env["warp"]["model"]["warmstart_disabled"] is True


def test_capture_refuses_when_solver_warmstart_is_enabled(origin):
    """The derived-field omissions rest on this flag, so it is asserted, not assumed."""
    mjm = origin.env.mjm
    original = int(mjm.opt.disableflags)
    mjm.opt.disableflags = original & ~int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    try:
        with pytest.raises(ppo_resume.ResumeValidationError, match="warm-start"):
            ppo_resume.capture_state(origin, origin.env, _extra(), epoch=0)
    finally:
        mjm.opt.disableflags = original


def test_segment_round_trip_restores_state_rng_and_next_sampled_action(origin, tmp_path):
    _train(origin, EPOCHS)
    state = ppo_resume.capture_state(origin, origin.env, _extra(), epoch=EPOCHS)
    path = tmp_path / "myoleg26_segment_0002.ptc"
    published = ppo_resume.write_segment(state, path)
    assert published["bytes"] > 0

    expected = _full_state(origin, origin.env)
    # Drawn after the capture, so the restored stream must reproduce it exactly.
    expected_action = _next_sampled_action(origin)

    fresh = _algo(tmp_path / "fresh", seed=7)
    try:
        # The fresh target has only been reset, never stepped, so it does not yet
        # have the lazily created obs_buf_before_reset (review defect D1).
        assert not hasattr(fresh.env, "obs_buf_before_reset")
        loaded = ppo_resume.read_segment(path, expect={"epoch": EPOCHS})
        report = ppo_resume.restore_state(loaded, fresh, fresh.env)
        assert report["cuda_rng_restored"] is True

        assert _differences(expected, _full_state(fresh, fresh.env)) == []

        restored_action = _next_sampled_action(fresh)
        for restored, want in zip(restored_action, expected_action):
            assert torch.equal(restored, want)
    finally:
        fresh.close()


def test_restore_fails_closed_on_mismatched_recipe_and_lineage(origin, tmp_path):
    """Identity gates refuse before the target is mutated, on the real env too."""
    _train(origin, 1)
    state = ppo_resume.capture_state(origin, origin.env, _extra(), epoch=1)
    path = tmp_path / "identity.ptc"
    ppo_resume.write_segment(state, path)

    # Lineage/epoch are bound only when the caller asks for them (strict_expect).
    with pytest.raises(ppo_resume.ResumeValidationError, match="epoch"):
        ppo_resume.read_segment(path, expect=ppo_resume.strict_expect(
            epoch=99, seed=0, segment_index=1, parent_segment_sha256=None))

    target = _algo(tmp_path / "identity_target", seed=5)
    try:
        before = _full_state(target, target.env)
        target.num_minibatches = target.num_minibatches + 1
        with pytest.raises(ppo_resume.ResumeValidationError, match="recipe mismatch"):
            ppo_resume.restore_state(ppo_resume.read_segment(path), target, target.env)
        target.num_minibatches = target.num_minibatches - 1
        assert _differences(before, _full_state(target, target.env)) == []
    finally:
        target.close()


def test_restore_fails_closed_on_a_different_compiled_model(origin, tmp_path):
    """The compiled-model hash is enforced by restore itself, not just recorded."""
    _train(origin, 1)
    state = ppo_resume.capture_state(origin, origin.env, _extra(), epoch=1)
    state.env["warp"]["model"] = dict(state.env["warp"]["model"],
                                      compiled_model_sha256="0" * 64)
    before = _full_state(origin, origin.env)
    with pytest.raises(ppo_resume.ResumeValidationError, match="compiled_model_sha256"):
        ppo_resume.restore_state(state, origin, origin.env)
    assert _differences(before, _full_state(origin, origin.env)) == []


def test_omitting_cuda_rng_changes_the_next_sampled_action(origin, tmp_path):
    """The CUDA-RNG omission control: on GPU only, never on the CPU path."""
    _train(origin, 1)
    state = ppo_resume.capture_state(origin, origin.env, _extra(), epoch=1)
    assert state.algo["rng"]["cuda_initialized"] is True
    expected_action = _next_sampled_action(origin)[0]

    fresh = _algo(tmp_path / "no_cuda_rng", seed=11)
    try:
        # What a segment that never recorded the CUDA stream would leave behind.
        state.algo["rng"]["cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
        ppo_resume.restore_state(state, fresh, fresh.env)
        assert not torch.equal(_next_sampled_action(fresh)[0], expected_action)
    finally:
        fresh.close()


def test_omitted_derived_scratch_fields_do_not_change_the_next_step(origin):
    """Empirical enforcement of the derived-field omission.

    Only safe numeric fields are perturbed — ``qacc`` (overwritten at solve init,
    ``solver.py:1568``/``:1570``) and ``act_dot`` (assigned per actuator each
    substep, ``forward.py:1030``). Contact/index counters are never touched:
    a forged count would index out of bounds on the device.
    """
    _train(origin, 1)
    state = ppo_resume.capture_state(origin, origin.env, _extra(), epoch=1)
    action = torch.full((origin.num_envs, origin.num_actions), 0.25, device=origin.device)

    origin.env.step(action)
    reference = _warp_fields(origin.env)

    ppo_resume.restore_state(state, origin, origin.env)
    with torch.no_grad():
        wp.to_torch(origin.env.warp_data.qacc).add_(1234.5)
        if origin.env.warp_data.act.size > 0:
            wp.to_torch(origin.env.warp_data.act_dot).add_(7.5)
    wp.synchronize()

    origin.env.step(action)
    for name, tensor in _warp_fields(origin.env).items():
        assert torch.equal(tensor, reference[name]), name


def test_continued_rollout_deviation_is_reported_not_tolerated(origin, tmp_path, capsys):
    """One tiny continued rollout, reported without any derived tolerance.

    BE-10/BE-11 leave forward repeat variability on this backend unresolved, and
    no existing test contract justifies a tolerance for a multi-step continued
    trajectory. This records the deviation and asserts only finiteness; it makes
    no continuous-trajectory or continuous GPU numerical-identity claim, and no
    tolerance is derived from repeats.
    """
    _train(origin, 1)
    state = ppo_resume.capture_state(origin, origin.env, _extra(), epoch=1)
    path = tmp_path / "continued.ptc"
    ppo_resume.write_segment(state, path)
    _train(origin, 1)
    reference = _warp_fields(origin.env)

    fresh = _algo(tmp_path / "continued_fresh", seed=23)
    try:
        ppo_resume.restore_state(ppo_resume.read_segment(path), fresh, fresh.env)
        _train(fresh, 1)
        deviations = {}
        for name, tensor in _warp_fields(fresh.env).items():
            assert torch.isfinite(tensor).all(), name
            deviations[name] = float((tensor - reference[name]).abs().max().item())
        print("continued-rollout max |deviation| after one epoch:", deviations)
        assert set(deviations) == set(ppo_resume.WARP_STATE_FIELDS)
    finally:
        fresh.close()
