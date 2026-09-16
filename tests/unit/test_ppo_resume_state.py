"""CPU contract for bounded PPO continuation state (Unit 3).

The existing ``PPO.save``/``PPO.load`` pair is an *inference* checkpoint: it
pickles live objects and rebuilds fresh optimizers. Continuing a training run
from it silently drops Adam moments, normalisation counts, episode counters,
rollout buffers, the environment's physics state and every RNG stream. These
tests pin the additive helper that captures and restores the complete boundary
instead.

Everything here runs on CPU against a deterministic analytic environment and the
**real** ``PPO.collect_rollout`` / ``PPO.update`` code paths, with real
``ActorStochasticMLP``, ``CriticMLP`` and ``torch.optim.Adam`` objects. The
analytic environment is a verification stand-in, not the simulator: passing
these tests is not full-dynamics qualification. It does reuse the real
``MyoLeg26WalkEnv._cut_action_history`` so the Task 2 previous-action boundary
semantics are exercised rather than re-implemented.

Streams: ``collect_rollout``/``update`` draw from the **torch CPU generator**
only (action sampling, ``torch.randperm``, and this environment's reset noise).
Python ``random`` and the NumPy global generator are seeded by
``msk_warp.utils.common.seeding`` but are never drawn from by those two methods,
so they are captured for completeness and **no coverage is claimed** for them.
CUDA RNG is deliberately untested here (see ``tests/gpu/test_myoleg26_ppo_resume.py``).
"""

from __future__ import annotations

import copy
import hashlib
import io
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

from msk_warp.algorithms.ppo import PPO
from msk_warp.analysis import ppo_resume
from msk_warp.analysis.ppo_resume import ROOT
from msk_warp.envs.myoleg26_walk import MyoLeg26WalkEnv
from msk_warp.networks.actor import ActorStochasticMLP
from msk_warp.networks.critic import CriticMLP
from msk_warp.utils.average_meter import AverageMeter
from msk_warp.utils.running_mean_std import RunningMeanStd


# ----------------------------------------------------------------------
# Deterministic analytic CPU environment
# ----------------------------------------------------------------------


class _AnalyticEnv:
    """Closed-form CPU dynamics with the MyoLeg26 reset/history boundary shape.

    State that must survive a segment boundary: ``sim_qpos``/``sim_qvel``/
    ``sim_act`` (the integration inputs), the previous-action history, and the
    episode progress counter. Reset noise is drawn from the torch CPU generator
    so an unrestored RNG stream is observable.

    ``qacc``/``act_dot`` are declared in a **separate** section
    (``RESUME_PLAIN_DERIVED_FIELDS``) because they are not env-written inputs:
    the real environment's ``_reset_warp_state`` (``myoleg26_walk.py:553-579``)
    writes only the six integration inputs, and the backend writes these two.
    Like the real fields they are recomputed by every ``step()`` and never read
    back as an input here, so this section binds **restore fidelity** and
    deliberately carries no continued-trajectory claim.
    """

    RESUME_PLAIN_STATE_FIELDS = ("sim_qpos", "sim_qvel", "sim_act")
    RESUME_PLAIN_DERIVED_FIELDS = ("qacc", "act_dot")

    def __init__(self, num_envs=4, num_act=3, episode_length=3, device="cpu"):
        self.device = device
        self.num_environments = num_envs
        self.num_actions = num_act
        self.episode_length = episode_length
        self.no_grad = True
        self.sim_qpos = torch.zeros((num_envs, 2), dtype=torch.float32)
        self.sim_qvel = torch.zeros((num_envs, 2), dtype=torch.float32)
        self.sim_act = torch.zeros((num_envs, num_act), dtype=torch.float32)
        self.actions = torch.zeros((num_envs, num_act), dtype=torch.float32)
        self.progress_buf = torch.zeros(num_envs, dtype=torch.long)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.long)
        self.termination_buf = torch.zeros(num_envs, dtype=torch.long)
        self.obs_buf = torch.zeros((num_envs, self.num_obs), dtype=torch.float32)
        self.rew_buf = torch.zeros(num_envs, dtype=torch.float32)
        # Derived scratch, written only by ``step()`` — never by ``_reset`` and
        # never read back, mirroring the real backend's ownership of these two.
        self.qacc = torch.zeros((num_envs, 2), dtype=torch.float32)
        self.act_dot = torch.zeros((num_envs, num_act), dtype=torch.float32)
        self.extras = {}
        # NOTE: obs_buf_before_reset is deliberately NOT created here. The real
        # env creates it only in step() (myoleg26_walk.py:474); base_env.py:80-86
        # does not. Pre-creating it in a constructor is exactly the mock
        # divergence that hid D1 from the first round of this suite.

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_obs(self):
        return 4 + 2 * self.num_actions

    def _observe(self):
        return torch.cat([self.sim_qpos, self.sim_qvel, self.sim_act, self.actions], dim=-1)

    def _reset(self, env_ids):
        n = len(env_ids)
        self.sim_qpos[env_ids] = 0.1 * (torch.rand((n, 2)) - 0.5)
        self.sim_qvel[env_ids] = 0.0
        self.sim_act[env_ids] = 0.0
        # The real Task 2 boundary hook, bound to this adapter.
        MyoLeg26WalkEnv._cut_action_history(self, env_ids)
        self.progress_buf[env_ids] = 0

    def reset(self):
        self._reset(torch.arange(self.num_envs, dtype=torch.long))
        self.obs_buf = self._observe()
        return self.obs_buf

    def step(self, action):
        action = action.view(self.num_envs, self.num_actions).clamp(-1.0, 1.0)
        previous = self.actions
        previous_act, previous_qvel = self.sim_act, self.sim_qvel
        self.sim_act = 0.8 * self.sim_act + 0.2 * (action + 0.5 * previous)
        drive = torch.stack([self.sim_act.sum(dim=-1), self.sim_act.mean(dim=-1)], dim=-1)
        self.sim_qvel = 0.95 * self.sim_qvel + 0.1 * drive
        self.sim_qpos = self.sim_qpos + 0.05 * self.sim_qvel
        self.actions = action.clone()
        # Rewritten every step from the inputs above, exactly like the fields
        # they stand in for; nothing downstream reads them.
        self.qacc = (self.sim_qvel - previous_qvel) / 0.05
        self.act_dot = (self.sim_act - previous_act) / 0.05

        self.obs_buf = self._observe()
        self.rew_buf = self.sim_qvel[:, 0] - 0.01 * (action ** 2).sum(dim=-1)
        self.progress_buf += 1
        terminated = self.sim_qpos[:, 0].abs() > 0.4
        truncated = (self.progress_buf >= self.episode_length) & ~terminated
        self.termination_buf = terminated.long()
        self.reset_buf = (terminated | truncated).long()
        self.obs_buf_before_reset = self.obs_buf.clone()
        self.extras = {
            "obs_before_reset": self.obs_buf_before_reset,
            "terminated": terminated,
            "truncated": truncated,
        }
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset(env_ids)
            self.obs_buf = torch.where(self.reset_buf[:, None].bool(), self._observe(), self.obs_buf)
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


# The adapter's declared derived section, held as a literal so the negative
# controls can narrow the class attribute without moving the assertions with it.
_DERIVED_FIELDS = ("qacc", "act_dot")

_NETWORK_CFG = {
    "actor_mlp": {"units": [8, 8], "activation": "elu"},
    "critic_mlp": {"units": [8, 8], "activation": "elu"},
    "actor_logstd_init": -1.0,
}


def _seed_cpu(seed):
    """Seed the streams without touching ``torch.cuda`` at all.

    ``msk_warp.utils.common.seeding`` also calls ``torch.cuda.manual_seed*``;
    those are lazy, but this test must not depend on that to keep the
    "no CUDA initialisation on the CPU path" assertion meaningful.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build(seed=7, steps_num=4, num_envs=4, num_act=3):
    """Real PPO object graph without SummaryWriter/ENV_MAP/logdir."""
    _seed_cpu(seed)
    env = _AnalyticEnv(num_envs=num_envs, num_act=num_act)
    algo = object.__new__(PPO)
    algo.device = "cpu"
    algo.env = env
    algo.num_envs = env.num_envs
    algo.num_obs = env.num_obs
    algo.num_actions = env.num_actions
    algo.max_episode_length = env.episode_length
    algo.steps_num = steps_num
    algo.max_epochs = 16
    algo.gamma = 0.99
    algo.gae_lambda = 0.95
    algo.actor_lr = 3e-4
    algo.critic_lr = 1e-3
    algo.lr_schedule = "constant"
    algo.betas = (0.9, 0.999)
    algo.clip_range = 0.2
    algo.entropy_coef = 0.01
    algo.value_coef = 0.5
    algo.ppo_epochs = 2
    algo.num_minibatches = 2
    algo.normalize_advantages = True
    algo.max_grad_norm = 0.5
    algo.truncate_grad = True
    algo.name = "ppo_resume_cpu_adapter"

    algo.actor = ActorStochasticMLP(algo.num_obs, algo.num_actions, _NETWORK_CFG, device="cpu")
    algo.critic = CriticMLP(algo.num_obs, _NETWORK_CFG, device="cpu")
    algo._build_optimizers()

    shape = (steps_num, algo.num_envs)
    algo.buf_obs = torch.zeros((*shape, algo.num_obs))
    algo.buf_actions = torch.zeros((*shape, algo.num_actions))
    for name in ("log_probs", "rewards", "dones", "values", "timeout_values",
                 "advantages", "returns"):
        setattr(algo, "buf_" + name, torch.zeros(shape))

    algo.obs_rms = RunningMeanStd(shape=(algo.num_obs,), device="cpu")
    algo.ret_rms = None
    algo.episode_loss = torch.zeros(algo.num_envs)
    algo.episode_length = torch.zeros(algo.num_envs, dtype=torch.long)
    algo.episode_loss_his = []
    algo.episode_length_his = []
    algo.best_policy_loss = np.inf
    algo.episode_loss_meter = AverageMeter(1, 100)
    algo.episode_length_meter = AverageMeter(1, 100)
    algo._current_obs = None
    algo.iter_count = 0
    algo.step_count = 0
    return algo, env


def _start(algo, env):
    algo._current_obs = env.reset()


def _train(algo, epochs):
    """The v1 external-driver shape: no ``train()``, caller bumps iter_count."""
    for _ in range(epochs):
        algo.collect_rollout()
        algo.update()
        algo.iter_count += 1


def _fingerprint(algo, env):
    """Everything a faithful continuation must reproduce, as comparable values."""
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
    for name in ppo_resume.ALGO_BUFFERS:
        out[name] = getattr(algo, name).detach().clone()
    out["obs_rms.mean"] = algo.obs_rms.mean.clone()
    out["obs_rms.var"] = algo.obs_rms.var.clone()
    out["obs_rms.count"] = algo.obs_rms.count
    out["iter_count"] = algo.iter_count
    out["step_count"] = algo.step_count
    out["episode_loss"] = algo.episode_loss.clone()
    out["episode_length"] = algo.episode_length.clone()
    out["episode_loss_his"] = list(algo.episode_loss_his)
    out["episode_length_his"] = list(algo.episode_length_his)
    out["loss_meter.mean"] = algo.episode_loss_meter.mean.clone()
    out["loss_meter.size"] = algo.episode_loss_meter.current_size
    out["length_meter.mean"] = algo.episode_length_meter.mean.clone()
    out["length_meter.size"] = algo.episode_length_meter.current_size
    current = algo._current_obs
    out["_current_obs"] = None if current is None else current.detach().clone()
    for name in _AnalyticEnv.RESUME_PLAIN_STATE_FIELDS:
        out[f"env.{name}"] = getattr(env, name).detach().clone()
    # Both derived fields are part of the complete state comparison. After
    # continued training they are recomputed by ``step()``, so their agreement
    # *here* is not evidence about restore fidelity — the dedicated tests in
    # section 2d bind that.
    for name in _DERIVED_FIELDS:
        out[f"env.{name}"] = getattr(env, name).detach().clone()
    for name in ("actions", "progress_buf", "reset_buf", "termination_buf", "obs_buf", "rew_buf"):
        out[f"env.{name}"] = getattr(env, name).detach().clone()
    out["rng.torch_cpu"] = torch.get_rng_state().clone()
    return out


def _derived_values(env):
    """The adapter's live derived scratch, cloned out of the way of a restore."""
    return {name: getattr(env, name).detach().clone() for name in _DERIVED_FIELDS}


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


def _metadata(**overrides):
    base = {"seed": 7, "segment_index": 1, "parent_segment_sha256": None}
    base.update(overrides)
    return base


def _capture(algo, env, epoch=3, *, include_git=False, **kwargs):
    return ppo_resume.capture_state(algo, env, extra=_metadata(**kwargs), epoch=epoch,
                                    include_git_identity=include_git)


# ----------------------------------------------------------------------
# 1. Core acceptance: 6 uninterrupted epochs == 3 + segment + fresh + 3
# ----------------------------------------------------------------------


def test_six_epochs_equal_three_plus_segment_round_trip_plus_three(tmp_path):
    reference_algo, reference_env = _build()
    _start(reference_algo, reference_env)
    _train(reference_algo, 6)
    expected = _fingerprint(reference_algo, reference_env)

    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 3)
    state = _capture(origin_algo, origin_env)
    path = tmp_path / "segment_0003.ptc"
    published = ppo_resume.write_segment(state, path)
    assert published["path"] == str(path)
    assert published["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    # Unit 4 shape: the origin keeps training after the boundary snapshot.
    _train(origin_algo, 3)

    fresh_algo, fresh_env = _build(seed=999)  # different construction draw
    loaded = ppo_resume.read_segment(path, expect={"schema_version": ppo_resume.SCHEMA_VERSION,
                                                   "epoch": 3})
    ppo_resume.restore_state(loaded, fresh_algo, fresh_env)
    _train(fresh_algo, 3)

    assert _differences(expected, _fingerprint(fresh_algo, fresh_env)) == []
    # The origin that continued independently is also still correct.
    assert _differences(expected, _fingerprint(origin_algo, origin_env)) == []


def test_boundary_snapshot_survives_a_partial_epoch_and_publishes_late(tmp_path):
    """Unit 4 shape: hold the last completed boundary in RAM, censor mid-epoch.

    The origin starts another rollout after the snapshot and is abandoned without
    an update. Publishing afterwards must still yield the completed-epoch state,
    not the partial one.
    """
    reference_algo, reference_env = _build()
    _start(reference_algo, reference_env)
    _train(reference_algo, 6)
    expected = _fingerprint(reference_algo, reference_env)

    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 3)
    held = _capture(origin_algo, origin_env)     # kept in RAM, nothing on disk yet
    origin_algo.collect_rollout()                # partial epoch, no update: censored

    path = tmp_path / "segment_after_censor.ptc"
    ppo_resume.write_segment(held, path)         # published only at segment end

    fresh_algo, fresh_env = _build(seed=321)
    ppo_resume.restore_state(ppo_resume.read_segment(path), fresh_algo, fresh_env)
    _train(fresh_algo, 3)
    assert _differences(expected, _fingerprint(fresh_algo, fresh_env)) == []


def test_capture_is_immutable_under_origin_mutation(tmp_path):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 2)
    state = _capture(algo, env, epoch=2)
    before = copy.deepcopy(state)

    with torch.no_grad():
        next(iter(algo.actor.parameters())).add_(1.0)
        algo.critic.critic[0].weight.mul_(-3.0)
        for entry in algo.actor_optimizer.state.values():
            entry["exp_avg"].add_(5.0)
        algo.obs_rms.mean.add_(2.0)
        algo.obs_rms.var.mul_(7.0)
        algo.obs_rms.count += 100.0
        algo.buf_obs.add_(1.0)
        algo.episode_loss.add_(1.0)
        algo.episode_loss_meter.mean.add_(1.0)
        algo.episode_loss_meter.current_size = 42
        algo._current_obs.add_(1.0)
        env.sim_qpos.add_(1.0)
        env.actions.add_(1.0)
        env.progress_buf.add_(1)
        # The derived section must be deep-copied too, in place and by rebinding.
        env.qacc.add_(1234.5)
        env.act_dot = env.act_dot + 7.5
    algo.episode_loss_his.append(-999.0)
    algo.iter_count += 50
    derived_before_mutation = {name: field["values"].clone()
                               for name, field in before.env["derived"].items()}

    assert _differences(_flatten(before), _flatten(state)) == []

    # And the snapshot still restores the pre-mutation values.
    target_algo, target_env = _build(seed=123)
    path = tmp_path / "immutable.ptc"
    ppo_resume.write_segment(state, path)
    ppo_resume.restore_state(ppo_resume.read_segment(path), target_algo, target_env)
    assert target_algo.iter_count == 2
    assert target_algo.episode_loss_meter.current_size != 42
    assert not torch.equal(target_env.sim_qpos, env.sim_qpos)
    for name, tensor in derived_before_mutation.items():
        assert torch.equal(getattr(target_env, name), tensor), name
        assert not torch.equal(getattr(target_env, name), getattr(env, name)), name


def _flatten(state):
    """Comparable leaf view of a ResumeState (tensors cloned, plain values kept)."""
    out = {}

    def walk(prefix, value):
        if isinstance(value, dict):
            for key in sorted(value, key=repr):
                walk(f"{prefix}.{key!r}", value[key])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(f"{prefix}[{index}]", item)
        elif torch.is_tensor(value):
            out[prefix] = value.detach().clone()
        else:
            out[prefix] = value

    walk("metadata", dict(state.metadata))
    walk("algo", dict(state.algo))
    walk("env", dict(state.env))
    walk("extra", dict(state.extra))
    return out


_CUDA_TRAP_PROBE = """
import json, sys, torch
sys.path.insert(0, {tests_dir!r})
import test_ppo_resume_state as t
from msk_warp.analysis import ppo_resume

algo, env = t._build()
t._start(algo, env)
before = torch.cuda.is_initialized()

# Portable trap: record any call to a CUDA entry point that would initialise a
# device. Works with or without a CUDA build, because it never calls them.
trapped = []
originals = {{}}
for name in ("init", "_lazy_init", "synchronize", "get_rng_state_all", "set_rng_state_all"):
    original = getattr(torch.cuda, name, None)
    if original is not None:
        originals[name] = original
        setattr(torch.cuda, name, (lambda n, f: (lambda *a, **k: (trapped.append(n), f(*a, **k))[1]))(name, original))
try:
    state = ppo_resume.capture_state(algo, env, epoch=0)
finally:
    for name, original in originals.items():
        setattr(torch.cuda, name, original)

print(json.dumps({{
    "available": torch.cuda.is_available(),
    "before": before,
    "after": torch.cuda.is_initialized(),
    "trapped": trapped,
    "cuda_is_none": state.algo["rng"]["cuda"] is None,
    "flag": state.algo["rng"]["cuda_initialized"],
}}))
"""


def test_capture_does_not_initialize_cuda_in_a_clean_process():
    """Capture must never initialise a device just to record an unused stream.

    Run in a clean subprocess because the property is order-sensitive in-process:
    on this pinned environment ``PPO.update()`` itself initialises CUDA even with
    ``device='cpu'`` — ``torch/optim/adam.py:222`` ->
    ``Optimizer._accelerator_graph_capture_health_check``
    (``torch/optim/optimizer.py:485``) -> ``torch.accelerator.current_stream``
    -> ``torch.cuda._lazy_init``. That is existing third-party behaviour in
    Torch 2.14.0+cu130; nothing here modifies PPO or Torch. The epoch-0 boundary
    used below is a legitimate capture point and performs no update.

    The proof is portable: it traps the CUDA entry points rather than requiring a
    CUDA build, so a CPU-only installation still exercises it.
    """
    import json
    import subprocess

    code = _CUDA_TRAP_PROBE.format(tests_dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(ROOT), timeout=300)
    except subprocess.TimeoutExpired:  # pragma: no cover - bounded guard
        pytest.fail("clean-process CUDA guard probe exceeded its 300 s timeout")
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["trapped"] == [], f"capture touched CUDA: {result['trapped']}"
    assert result["before"] == result["after"]
    if result["available"] and not result["before"]:
        assert result["cuda_is_none"] is True
        assert result["flag"] is False
    else:  # pragma: no cover - CPU-only or already-initialised installations
        pytest.skip(f"CUDA-sensitive branch not exercised here: {result}")


def test_capture_never_changes_cuda_initialisation_state():
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    before = torch.cuda.is_initialized()
    state = _capture(algo, env, epoch=1)
    assert torch.cuda.is_initialized() is before
    assert state.algo["rng"]["cuda_initialized"] is before
    assert (state.algo["rng"]["cuda"] is None) is (not before)


# ----------------------------------------------------------------------
# 2. Omission negative controls — each must break the round trip
# ----------------------------------------------------------------------


def _round_trip_matches(mutate=None, tmp_path=None):
    reference_algo, reference_env = _build()
    _start(reference_algo, reference_env)
    _train(reference_algo, 6)
    expected = _fingerprint(reference_algo, reference_env)

    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 3)
    state = _capture(origin_algo, origin_env)
    path = tmp_path / "control.ptc"
    ppo_resume.write_segment(state, path)
    loaded = ppo_resume.read_segment(path)

    # The fresh target is built *before* the omission is applied, so an omitted
    # field can be emulated with exactly what the target already carries — what
    # a segment that never recorded that field would leave in place.
    fresh_algo, fresh_env = _build(seed=999)
    if mutate is not None:
        loaded = mutate(loaded)
    ppo_resume.restore_state(loaded, fresh_algo, fresh_env)
    _train(fresh_algo, 3)
    return _differences(expected, _fingerprint(fresh_algo, fresh_env))


def test_positive_control_round_trip_matches(tmp_path):
    assert _round_trip_matches(tmp_path=tmp_path) == []


def test_omitting_adam_state_breaks_continuation(tmp_path):
    def drop(state):
        state.algo["actor_optimizer"]["state"] = {}
        state.algo["critic_optimizer"]["state"] = {}
        return state

    assert _round_trip_matches(drop, tmp_path) != []


def test_omitting_normalization_count_breaks_continuation(tmp_path):
    def drop(state):
        state.algo["obs_rms"]["count"] = 1e-4  # the constructor's epsilon
        return state

    assert _round_trip_matches(drop, tmp_path) != []


def test_omitting_exercised_cpu_rng_stream_breaks_continuation(tmp_path):
    """The CPU generator is the stream collect/update actually draw from."""

    def drop(state):
        # What the freshly constructed target already holds: no restore at all.
        state.algo["rng"]["torch_cpu"] = torch.get_rng_state().clone()
        return state

    assert _round_trip_matches(drop, tmp_path) != []


def test_omitting_live_env_state_breaks_continuation(tmp_path):
    def drop(state):
        for name in _AnalyticEnv.RESUME_PLAIN_STATE_FIELDS:
            state.env["plain"][name] = torch.zeros_like(state.env["plain"][name])
        return state

    assert _round_trip_matches(drop, tmp_path) != []


def test_omitting_action_history_breaks_continuation(tmp_path):
    def drop(state):
        state.env["buffers"]["actions"] = torch.zeros_like(state.env["buffers"]["actions"])
        return state

    assert _round_trip_matches(drop, tmp_path) != []


def test_omitting_meter_current_size_breaks_meter_state(tmp_path):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 3)
    assert algo.episode_loss_meter.current_size > 0
    state = _capture(algo, env)
    state.algo["episode_loss_meter"]["current_size"] = 0
    fresh_algo, fresh_env = _build(seed=5)
    ppo_resume.restore_state(state, fresh_algo, fresh_env)
    values = torch.full((2, 1), 3.0)
    fresh_algo.episode_loss_meter.update(values)
    algo.episode_loss_meter.update(values)
    assert not torch.equal(fresh_algo.episode_loss_meter.mean, algo.episode_loss_meter.mean)


# ----------------------------------------------------------------------
# 2d. Preserved derived solver scratch — restore fidelity (2026-09-15 ruling)
# ----------------------------------------------------------------------
#
# History, stated exactly. The first schema OMITTED ``qacc`` and ``act_dot`` and
# justified that omission with a GPU test which required two INDEPENDENT physics
# runs to agree bitwise. That test **failed** in the full GPU suite on
# ``7bd388da…`` (``AssertionError: qvel``) after passing in isolation, and the
# preregistered 24-trial diagnostic then showed its premise was unsound: six
# UNPERTURBED control repeats alone split across two post-step ``qvel`` values,
# one entry apart by one float32 ULP. The failing result and both receipts stand
# as historical evidence and are **not** relabelled a pass, not xfailed and not
# given a tolerance. What is withdrawn is the *claim* that omitting these two
# fields is qualified.
#
# So the fields are now captured and restored, and the regression below asserts
# exact **restore fidelity** — a property of this helper alone, independent of
# whether the simulator reproduces a step. It is NOT evidence about continued
# trajectories, which stay UNVALIDATED.
#
# The withdrawn test was also cited as the empirical enforcement for EVERY other
# omitted derived field; that global claim is withdrawn as well. It only ever
# perturbed those two fields, so no untested contact, solver, cache or counter
# array inherits validation from it. Those remain omitted with neither a
# per-field argument nor any empirical control.


def _prepare_fresh(algo, env):
    """A target that has only been reset, never stepped."""
    _start(algo, env)


def _prepare_mutated(algo, env):
    """A target whose derived scratch is already live, and different."""
    _start(algo, env)
    _train(algo, 1)
    with torch.no_grad():
        env.qacc.add_(1234.5)
        env.act_dot.add_(7.5)


def test_derived_scratch_is_captured_at_its_own_dtype_and_shape():
    algo, env = _build()
    _start(algo, env)
    _train(algo, 2)
    live = _derived_values(env)
    for name, tensor in live.items():
        assert float(tensor.abs().sum()) > 0.0, f"{name} is trivial; the test would not bind"

    state = _capture(algo, env, epoch=2)
    derived = state.env["derived"]
    assert set(derived) == set(_DERIVED_FIELDS)
    for name, tensor in live.items():
        assert derived[name]["dtype"] == str(tensor.dtype)
        assert tuple(derived[name]["shape"]) == tuple(tensor.shape)
        assert derived[name]["values"].dtype == tensor.dtype
        assert torch.equal(derived[name]["values"], tensor), name
    # Recorded in its own section, never folded into the env-written inputs.
    assert set(state.env["plain"]) == set(_AnalyticEnv.RESUME_PLAIN_STATE_FIELDS)
    assert not set(state.env["plain"]) & set(_DERIVED_FIELDS)


@pytest.mark.parametrize("prepare", [_prepare_fresh, _prepare_mutated],
                         ids=["fresh_target", "already_mutated_target"])
def test_restore_reproduces_derived_scratch_bytes_exactly(tmp_path, prepare):
    """The replacement regression: exact capture -> write -> read -> restore."""
    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 2)
    captured = _derived_values(origin_env)
    state = _capture(origin_algo, origin_env, epoch=2)
    path = tmp_path / f"derived_{prepare.__name__}.ptc"
    ppo_resume.write_segment(state, path)

    target_algo, target_env = _build(seed=997)
    prepare(target_algo, target_env)
    for name, tensor in captured.items():
        assert not torch.equal(getattr(target_env, name), tensor), (
            f"{name} already matches before the restore; the test would be vacuous")

    ppo_resume.restore_state(ppo_resume.read_segment(path), target_algo, target_env)
    for name, tensor in captured.items():
        restored = getattr(target_env, name)
        assert restored.dtype == tensor.dtype, name
        assert tuple(restored.shape) == tuple(tensor.shape), name
        assert torch.equal(restored, tensor), name
    # The source is untouched by the restore of a copy taken from it.
    assert _differences({f"src.{k}": v for k, v in captured.items()},
                        {f"src.{k}": v for k, v in _derived_values(origin_env).items()}) == []


@pytest.mark.parametrize("dropped", _DERIVED_FIELDS)
def test_dropping_one_derived_field_from_the_schema_loses_it(dropped, monkeypatch):
    """Negative control, per field: without restoration the value is demonstrably lost.

    The sibling field is restored in the same call, so this separates "the field
    needs restoring" from "the machinery does not work at all".
    """
    kept = tuple(name for name in _DERIVED_FIELDS if name != dropped)
    monkeypatch.setattr(_AnalyticEnv, "RESUME_PLAIN_DERIVED_FIELDS", kept)

    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 2)
    captured = _derived_values(origin_env)
    state = _capture(origin_algo, origin_env, epoch=2)
    assert set(state.env["derived"]) == set(kept)

    target_algo, target_env = _build(seed=995)
    _start(target_algo, target_env)
    ppo_resume.restore_state(state, target_algo, target_env)
    assert not torch.equal(getattr(target_env, dropped), captured[dropped]), (
        f"{dropped} survived without being restored; the control does not bind")
    for name in kept:
        assert torch.equal(getattr(target_env, name), captured[name]), name


@pytest.mark.parametrize("break_it,pattern", [
    ("missing_section", "derived"),
    ("missing_field", "qacc"),
    ("extra_field", "qfrc_bias"),
    ("record_dtype", "dtype"),
    ("payload_dtype", "dtype"),
    ("wrong_shape", "shape"),
    ("nonfinite", "nonfinite"),
])
def test_bad_derived_payload_is_refused_before_the_target_is_mutated(break_it, pattern):
    """No zero-fill, no default, no partial write: refuse, leaving the target intact."""
    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 2)
    state = _capture(origin_algo, origin_env, epoch=2)
    derived = state.env["derived"]
    if break_it == "missing_section":
        del state.env["derived"]
    elif break_it == "missing_field":
        del derived["qacc"]
    elif break_it == "extra_field":
        derived["qfrc_bias"] = {key: value for key, value in derived["qacc"].items()}
    elif break_it == "record_dtype":
        derived["qacc"] = dict(derived["qacc"], dtype="torch.float64")
    elif break_it == "payload_dtype":
        derived["qacc"] = dict(derived["qacc"],
                               values=derived["qacc"]["values"].to(torch.float64))
    elif break_it == "wrong_shape":
        derived["qacc"] = dict(derived["qacc"],
                               values=torch.zeros((1, 1), dtype=torch.float32),
                               shape=(1, 1))
    else:
        broken = derived["qacc"]["values"].clone()
        broken[0, 0] = float("nan")
        derived["qacc"] = dict(derived["qacc"], values=broken)

    target_algo, target_env = _build(seed=993)
    _start(target_algo, target_env)
    _train(target_algo, 1)
    before = _fingerprint(target_algo, target_env)
    with pytest.raises(ppo_resume.ResumeValidationError, match=pattern):
        ppo_resume.restore_state(state, target_algo, target_env)
    assert _differences(before, _fingerprint(target_algo, target_env)) == []


def test_nonfinite_derived_scratch_is_refused_at_capture():
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    with torch.no_grad():
        env.act_dot[0, 0] = float("inf")
    with pytest.raises(ppo_resume.ResumeValidationError, match="nonfinite"):
        _capture(algo, env, epoch=1)


def test_schema_version_is_bumped_for_the_newly_required_derived_state():
    """The required state set grew, so the version moves; nothing is migrated.

    No segment of the previous version exists outside these tests' temporary
    directories, and no science artifact uses this schema, so a bump plus an
    outright refusal is the honest encoding: an older payload genuinely lacks
    required state and must never be completed with defaults.
    """
    assert ppo_resume.SCHEMA_VERSION == "myoleg26-ppo-resume-v2"


def test_a_previous_schema_segment_file_is_refused_rather_than_migrated(tmp_path, monkeypatch):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    monkeypatch.setattr(ppo_resume, "SCHEMA_VERSION", "myoleg26-ppo-resume-v1")
    stale = _capture(algo, env, epoch=1)
    path = tmp_path / "stale_v1.ptc"
    ppo_resume.write_segment(stale, path)
    monkeypatch.undo()

    with pytest.raises(ppo_resume.ResumeValidationError, match="myoleg26-ppo-resume-v1"):
        ppo_resume.read_segment(path)

    target_algo, target_env = _build(seed=991)
    _start(target_algo, target_env)
    before = _fingerprint(target_algo, target_env)
    with pytest.raises(ppo_resume.ResumeValidationError, match="myoleg26-ppo-resume-v1"):
        ppo_resume.restore_state(stale, target_algo, target_env)
    assert _differences(before, _fingerprint(target_algo, target_env)) == []


# ----------------------------------------------------------------------
# 2b. Lazily created environment buffers (review defect D1)
# ----------------------------------------------------------------------


def test_fresh_never_stepped_target_accepts_a_stepped_segment(tmp_path):
    """D1: the real restore target has never stepped, so it lacks the lazy buffer.

    ``MyoLeg26WalkEnv`` creates ``obs_buf_before_reset`` only inside ``step()``
    (``myoleg26_walk.py:474``); ``MjWarpEnv.__init__`` (``base_env.py:80-86``)
    and ``reset()`` do not. A target built by ``PPO(cfg)`` + ``env.reset()`` is
    therefore missing the attribute that every trained segment carries.
    """
    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 2)
    assert hasattr(origin_env, "obs_buf_before_reset")
    state = _capture(origin_algo, origin_env, epoch=2)
    assert "obs_buf_before_reset" in state.env["buffers"]

    fresh_algo, fresh_env = _build(seed=71)
    _start(fresh_algo, fresh_env)  # reset() only: the production surface
    assert not hasattr(fresh_env, "obs_buf_before_reset")

    ppo_resume.restore_state(state, fresh_algo, fresh_env)
    assert torch.equal(fresh_env.obs_buf_before_reset, origin_env.obs_buf_before_reset)
    assert fresh_env.obs_buf_before_reset.dtype == origin_env.obs_buf.dtype
    assert tuple(fresh_env.obs_buf_before_reset.shape) == (fresh_env.num_envs, fresh_env.num_obs)


def test_epoch_zero_segment_records_the_absent_lazy_buffer_and_restores_that_surface(tmp_path):
    """D1: absence at epoch 0 is legitimate state, not a field to skip silently."""
    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    state = _capture(origin_algo, origin_env, epoch=0)
    assert "obs_buf_before_reset" not in state.env["buffers"]
    assert "obs_buf_before_reset" in state.env["absent_lazy_buffers"]

    path = tmp_path / "epoch0_lazy.ptc"
    ppo_resume.write_segment(state, path)

    target_algo, target_env = _build(seed=73)
    _start(target_algo, target_env)
    _train(target_algo, 1)  # the target HAS stepped, so it holds a stale buffer
    assert hasattr(target_env, "obs_buf_before_reset")

    ppo_resume.restore_state(ppo_resume.read_segment(path), target_algo, target_env)
    assert not hasattr(target_env, "obs_buf_before_reset"), (
        "restoring a never-stepped boundary must not leave another run's buffer behind")


def test_lazy_buffer_is_validated_against_the_env_obs_contract_before_creation(tmp_path):
    """D1: a wrong-shaped lazy buffer must be refused, not created on the target."""
    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 1)
    state = _capture(origin_algo, origin_env, epoch=1)
    state.env["buffers"]["obs_buf_before_reset"] = torch.zeros((2, 3), dtype=torch.float32)

    fresh_algo, fresh_env = _build(seed=75)
    _start(fresh_algo, fresh_env)
    before = _fingerprint(fresh_algo, fresh_env)
    with pytest.raises(ppo_resume.ResumeValidationError, match="obs_buf_before_reset"):
        ppo_resume.restore_state(state, fresh_algo, fresh_env)
    assert not hasattr(fresh_env, "obs_buf_before_reset")
    assert _differences(before, _fingerprint(fresh_algo, fresh_env)) == []


# ----------------------------------------------------------------------
# 2c. RNG validation happens before any mutation (review defect D2)
# ----------------------------------------------------------------------


def _global_rng_fingerprint():
    """Every process-global stream a restore could touch, CUDA included.

    The whole NumPy ``get_state`` tuple is kept, not just the key: the position
    and the cached-Gaussian flag are state too. CUDA default generators are read
    only when CUDA is **already** initialised, so fingerprinting an unused case
    never initialises a device.
    """
    numpy_state = np.random.get_state(legacy=True)
    initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tobytes(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda_initialized": initialized,
        "cuda": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                 if initialized else None),
    }


def _assert_global_rng_unchanged(before):
    after = _global_rng_fingerprint()
    assert after["python"] == before["python"], "Python global stream changed"
    assert after["numpy"] == before["numpy"], "NumPy global stream changed"
    assert torch.equal(after["torch_cpu"], before["torch_cpu"]), "torch CPU stream changed"
    assert after["cuda_initialized"] == before["cuda_initialized"], "CUDA init state changed"
    if before["cuda"] is None:
        assert after["cuda"] is None
    else:
        assert len(after["cuda"]) == len(before["cuda"])
        for index, (left, right) in enumerate(zip(after["cuda"], before["cuda"])):
            assert torch.equal(left, right), f"CUDA default stream {index} changed"


def _live_cuda_state(index=0):
    """This runtime's own valid CUDA state, or None when CUDA is not initialised."""
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return None
    return torch.cuda.get_rng_state(index).clone()


def _cuda_state_with_offset(offset):
    """A correctly sized but content-invalid state: Philox needs offset % 4 == 0.

    Built from the runtime's own state so nothing about the layout is assumed
    beyond the trailing 8-byte offset word the review's probe identified; the
    *validator* never hard-codes the format — it asks a private generator.
    """
    live = _live_cuda_state()
    if live is None:
        return None
    candidate = live.clone()
    candidate[8:16] = torch.tensor([offset, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)
    return candidate


def _live_cuda_state_length():
    """The target runtime's own CUDA RNG state length — never a hard-coded 16.

    Only read when CUDA is *already* initialised, so nothing here lazily
    initialises a device.
    """
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return None
    return int(torch.cuda.get_rng_state(0).numel())


@pytest.mark.parametrize("break_it,pattern", [
    ("cuda_unavailable", "CUDA"),
    ("cuda_length_short", "CUDA device 0"),
    ("cuda_length_zero", "CUDA device 0"),
    ("cuda_length_long", "CUDA device 0"),
    ("cuda_content_offset_1", "CUDA device 0 state is malformed"),
    ("cuda_content_offset_2", "CUDA device 0 state is malformed"),
    ("cuda_inventory_subset", "device"),
    ("cuda_inventory_extra", "device"),
    ("numpy_key", "numpy"),
    ("torch_cpu_dtype", "torch_cpu"),
    ("python_state", "python"),
    ("missing_stream", "rng"),
])
def test_rng_problems_are_refused_before_any_target_or_global_mutation(
        tmp_path, monkeypatch, break_it, pattern):
    """D2: RNG is restored last, so its checks must be hoisted into validation.

    Restoring a CUDA-carrying segment on a CPU-only host is the realistic case:
    it must not clobber parameters, Adam state, the env, the counters, or the
    process-global Python/NumPy/Torch streams on its way to failing.
    """
    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 2)
    state = _capture(origin_algo, origin_env, epoch=2)

    rng = state.algo["rng"]
    if break_it == "cuda_unavailable":
        rng["cuda"] = [torch.zeros(4, dtype=torch.uint8)]
        rng["cuda_initialized"] = True
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elif break_it.startswith("cuda_length"):
        # A wrong-length CUDA state survives both checksums (they only prove the
        # bytes were not altered after writing) and used to reach
        # torch.cuda.set_rng_state_all, which raises a bare RuntimeError
        # ("RNG state is wrong size") after the whole target was overwritten.
        live = _live_cuda_state_length()
        if live is None:
            pytest.skip("CUDA is not already initialised here; the length branch is unreachable "
                        "without lazily initialising a device, which capture must never do")
        size = {"cuda_length_short": max(live - 13, 1), "cuda_length_zero": 0,
                "cuda_length_long": live + 8}[break_it]
        rng["cuda"] = [torch.zeros(size, dtype=torch.uint8)]
        rng["cuda_initialized"] = True
    elif break_it.startswith("cuda_content_offset"):
        # Correctly sized, valid checksums, still invalid: the review's probe
        # showed torch rejects a Philox offset that is not a multiple of 4, and
        # that the rejection used to land inside _restore_rng after full mutation.
        candidate = _cuda_state_with_offset(int(break_it.rsplit("_", 1)[1]))
        if candidate is None:
            pytest.skip("CUDA is not already initialised here; the content branch is "
                        "unreachable without lazily initialising a device")
        rng["cuda"] = [candidate]
        rng["cuda_initialized"] = True
    elif break_it.startswith("cuda_inventory"):
        # capture_state always records one entry per device; a subset or an extra
        # entry would let set_rng_state_all silently restore part of the host.
        live_state = _live_cuda_state()
        if live_state is None:
            pytest.skip("CUDA is not already initialised here")
        rng["cuda"] = ([] if break_it.endswith("subset")
                       else [live_state, live_state.clone()])
        rng["cuda_initialized"] = True
    elif break_it == "numpy_key":
        rng["numpy"] = dict(rng["numpy"], key=[1, 2, 3])
    elif break_it == "torch_cpu_dtype":
        rng["torch_cpu"] = rng["torch_cpu"].to(torch.int32)
    elif break_it == "python_state":
        rng["python"] = (3, (1, 2, 3), None)
    else:
        del rng["torch_cpu"]

    target_algo, target_env = _build(seed=81)
    _start(target_algo, target_env)
    before_target = _fingerprint(target_algo, target_env)
    before_global = _global_rng_fingerprint()

    with pytest.raises(ppo_resume.ResumeValidationError, match=pattern):
        ppo_resume.restore_state(state, target_algo, target_env)

    assert _differences(before_target, _fingerprint(target_algo, target_env)) == []
    _assert_global_rng_unchanged(before_global)


def test_malformed_cuda_rng_survives_publication_and_is_refused_before_mutation(tmp_path):
    """The reviewer's reopened D2 case: malformed *before* publication.

    Both checksums only prove the bytes were not altered after writing, so a
    record that was already malformed passes ``read_segment`` cleanly. The
    refusal therefore has to come from validation, before anything is written.
    """
    live = _live_cuda_state_length()
    if live is None:
        pytest.skip("CUDA is not already initialised here; see the isolated reproducer")
    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 2)
    state = _capture(origin_algo, origin_env, epoch=2)
    state.algo["rng"]["cuda"] = [torch.zeros(3, dtype=torch.uint8)]
    state.algo["rng"]["cuda_initialized"] = True

    path = tmp_path / "malformed_cuda.ptc"
    published = ppo_resume.write_segment(state, path)
    loaded = ppo_resume.read_segment(path)  # valid checksums: must load
    assert published["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.algo["rng"]["cuda"][0].numel() == 3 != live

    target_algo, target_env = _build(seed=93)
    _start(target_algo, target_env)
    before_target = _fingerprint(target_algo, target_env)
    before_global = _global_rng_fingerprint()
    with pytest.raises(ppo_resume.ResumeValidationError, match="CUDA device 0"):
        ppo_resume.restore_state(loaded, target_algo, target_env)
    assert _differences(before_target, _fingerprint(target_algo, target_env)) == []
    _assert_global_rng_unchanged(before_global)


def test_content_invalid_cuda_rng_survives_publication_and_is_refused_before_mutation(tmp_path):
    """Valid length, valid checksums, invalid Philox content — the round-3 case."""
    candidate = _cuda_state_with_offset(1)
    if candidate is None:
        pytest.skip("CUDA is not already initialised here; see the isolated reproducer")
    origin_algo, origin_env = _build()
    _start(origin_algo, origin_env)
    _train(origin_algo, 2)
    state = _capture(origin_algo, origin_env, epoch=2)
    live = _live_cuda_state()
    state.algo["rng"]["cuda"] = [candidate]
    assert candidate.numel() == live.numel()  # correctly sized: length gate cannot catch it

    path = tmp_path / "content_invalid_cuda.ptc"
    published = ppo_resume.write_segment(state, path)
    loaded = ppo_resume.read_segment(path)
    assert published["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert torch.equal(loaded.algo["rng"]["cuda"][0], candidate)

    target_algo, target_env = _build(seed=101)
    _start(target_algo, target_env)
    before_target = _fingerprint(target_algo, target_env)
    before_global = _global_rng_fingerprint()
    with pytest.raises(ppo_resume.ResumeValidationError, match="CUDA device 0 state is malformed"):
        ppo_resume.restore_state(loaded, target_algo, target_env)
    assert _differences(before_target, _fingerprint(target_algo, target_env)) == []
    _assert_global_rng_unchanged(before_global)


def test_in_memory_content_invalid_cuda_rng_is_refused_before_mutation():
    """The same refusal without any disk round trip at all."""
    candidate = _cuda_state_with_offset(2)
    if candidate is None:
        pytest.skip("CUDA is not already initialised here")
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    state.algo["rng"]["cuda"] = [candidate]

    target_algo, target_env = _build(seed=103)
    _start(target_algo, target_env)
    before_target = _fingerprint(target_algo, target_env)
    before_global = _global_rng_fingerprint()
    with pytest.raises(ppo_resume.ResumeValidationError, match="malformed"):
        ppo_resume.restore_state(state, target_algo, target_env)
    assert _differences(before_target, _fingerprint(target_algo, target_env)) == []
    _assert_global_rng_unchanged(before_global)


def test_rng_validation_itself_touches_no_global_stream():
    """The private-generator oracle must be neutral, including the CUDA default."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    before = _global_rng_fingerprint()

    problems = []
    ppo_resume._validate_rng(state.algo["rng"], problems)
    assert problems == []
    _assert_global_rng_unchanged(before)

    # And on a rejected record, where the oracle actually raises.
    candidate = _cuda_state_with_offset(1)
    if candidate is not None:
        rejected = dict(state.algo["rng"])
        rejected["cuda"] = [candidate]
        before = _global_rng_fingerprint()
        problems = []
        ppo_resume._validate_rng(rejected, problems)
        assert problems and "malformed" in problems[0]
        _assert_global_rng_unchanged(before)


def test_valid_length_cuda_rng_record_is_accepted(tmp_path):
    """The length guard must not reject the runtime's own valid state."""
    live = _live_cuda_state_length()
    if live is None:
        pytest.skip("CUDA is not already initialised here")
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    assert state.algo["rng"]["cuda"] is not None
    assert all(item.numel() == live for item in state.algo["rng"]["cuda"])
    target_algo, target_env = _build(seed=95)
    _start(target_algo, target_env)
    report = ppo_resume.restore_state(state, target_algo, target_env)
    assert report["cuda_rng_restored"] is True


_CUDA_UNINITIALISED_PROBE = """
import json, sys, torch
sys.path.insert(0, {tests_dir!r})
import test_ppo_resume_state as t
from msk_warp.analysis import ppo_resume

algo, env = t._build()
t._start(algo, env)
state = ppo_resume.capture_state(algo, env, epoch=0)
assert state.algo["rng"]["cuda"] is None
# A CUDA-carrying segment reaching a host where CUDA was never initialised:
# torch.cuda.set_rng_state_all() would DEFER through _lazy_call and surface the
# failure at an arbitrary later point, so restore must refuse up front.
state.algo["rng"]["cuda"] = [torch.zeros(3, dtype=torch.uint8)]
state.algo["rng"]["cuda_initialized"] = True
# Self-consistent inventory, so the refusal must come from the
# not-initialised gate rather than the inventory gate.
state.algo["rng"]["cuda_device_count"] = 1

target_algo, target_env = t._build(seed=97)
t._start(target_algo, target_env)
before = target_algo.iter_count, t._fingerprint(target_algo, target_env)
raised = None
try:
    ppo_resume.restore_state(state, target_algo, target_env)
except ppo_resume.ResumeValidationError as error:
    raised = str(error)
print(json.dumps({{
    "available": torch.cuda.is_available(),
    "initialized_after": torch.cuda.is_initialized(),
    "raised": raised,
    "target_unchanged": t._differences(before[1], t._fingerprint(target_algo, target_env)) == [],
}}))
"""


def test_cuda_carrying_segment_on_an_uninitialised_host_refuses_without_initialising():
    """Compatibility requirement made explicit, and the no-lazy-init rule kept."""
    import json
    import subprocess

    code = _CUDA_UNINITIALISED_PROBE.format(
        tests_dir=os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(ROOT), timeout=300)
    except subprocess.TimeoutExpired:  # pragma: no cover - bounded guard
        pytest.fail("clean-process CUDA-uninitialised probe exceeded its 300 s timeout")
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["raised"] is not None and "not initialised" in result["raised"]
    assert result["target_unchanged"] is True
    assert result["initialized_after"] is False, "validation must not initialise a device"


def test_restore_report_states_whether_the_cuda_stream_was_restored(tmp_path):
    """D2 (reverse direction): a CPU segment must not silently imply CUDA coverage.

    A segment captured before CUDA was ever initialised carries no CUDA stream,
    so the target's own CUDA generator is left untouched. The report must say so
    instead of letting the caller assume the restore was total.
    """
    algo, env = _build()
    _start(algo, env)
    state = _capture(algo, env, epoch=0)
    # Force the CPU-captured shape regardless of what earlier tests initialised.
    state.algo["rng"]["cuda"] = None
    state.algo["rng"]["cuda_initialized"] = False

    target_algo, target_env = _build(seed=83)
    _start(target_algo, target_env)
    report = ppo_resume.restore_state(state, target_algo, target_env)
    assert report["cuda_rng_restored"] is False
    assert report["rng_streams_restored"] == ("python", "numpy", "torch_cpu")
    assert report["lazy_buffers_cleared"] == ("obs_buf_before_reset",)

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_state = _capture(algo, env, epoch=0)
        assert cuda_state.algo["rng"]["cuda"] is not None
        cuda_report = ppo_resume.restore_state(cuda_state, target_algo, target_env)
        assert cuda_report["cuda_rng_restored"] is True
        assert cuda_report["rng_streams_restored"][-1] == "cuda"


def test_unsupported_best_metric_record_is_refused_before_mutation(tmp_path):
    """Any recognised validation error must fail closed, not mid-restore."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    state.algo["best_policy_loss"] = {"kind": "complex", "dtype": "", "value": 1.0}
    target_algo, target_env = _build(seed=85)
    _start(target_algo, target_env)
    before = _fingerprint(target_algo, target_env)
    with pytest.raises(ppo_resume.ResumeValidationError, match="best_policy_loss"):
        ppo_resume.restore_state(state, target_algo, target_env)
    assert _differences(before, _fingerprint(target_algo, target_env)) == []


def test_optimizer_state_index_outside_param_groups_is_refused_before_mutation(tmp_path):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    entry = next(iter(state.algo["actor_optimizer"]["state"].values()))
    state.algo["actor_optimizer"]["state"][9999] = entry
    target_algo, target_env = _build(seed=87)
    _start(target_algo, target_env)
    before = _fingerprint(target_algo, target_env)
    with pytest.raises(ppo_resume.ResumeValidationError, match="9999"):
        ppo_resume.restore_state(state, target_algo, target_env)
    assert _differences(before, _fingerprint(target_algo, target_env)) == []


# ----------------------------------------------------------------------
# 3. Schema, sentinel and finiteness handling
# ----------------------------------------------------------------------


def test_zero_epoch_capture_supports_empty_adam_state(tmp_path):
    algo, env = _build()
    _start(algo, env)
    assert algo.actor_optimizer.state_dict()["state"] == {}
    state = _capture(algo, env, epoch=0)
    path = tmp_path / "epoch0.ptc"
    ppo_resume.write_segment(state, path)
    fresh_algo, fresh_env = _build(seed=3)
    _start(fresh_algo, fresh_env)
    _train(fresh_algo, 1)  # give the target non-empty Adam state to overwrite
    ppo_resume.restore_state(ppo_resume.read_segment(path), fresh_algo, fresh_env)
    assert fresh_algo.actor_optimizer.state_dict()["state"] == {}
    assert fresh_algo.critic_optimizer.state_dict()["state"] == {}
    assert fresh_algo.iter_count == 0


@pytest.mark.parametrize("sentinel", [np.inf, -np.inf, 12.5])
def test_best_metric_sentinel_round_trips_without_json(tmp_path, sentinel):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    algo.best_policy_loss = sentinel
    state = _capture(algo, env, epoch=1)
    path = tmp_path / f"sentinel_{str(sentinel).replace('-', 'm').replace('.', '_')}.ptc"
    ppo_resume.write_segment(state, path)
    fresh_algo, fresh_env = _build(seed=11)
    ppo_resume.restore_state(ppo_resume.read_segment(path), fresh_algo, fresh_env)
    assert fresh_algo.best_policy_loss == sentinel
    assert isinstance(fresh_algo.best_policy_loss, float)


def test_best_metric_numpy_scalar_from_meter_round_trips(tmp_path):
    """``AverageMeter.get_mean()`` returns a 0-d NumPy array; ``train()`` stores it."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 2)
    algo.best_policy_loss = algo.episode_loss_meter.get_mean()
    assert isinstance(algo.best_policy_loss, np.ndarray)
    state = _capture(algo, env, epoch=2)
    path = tmp_path / "numpy_sentinel.ptc"
    ppo_resume.write_segment(state, path)
    fresh_algo, fresh_env = _build(seed=13)
    ppo_resume.restore_state(ppo_resume.read_segment(path), fresh_algo, fresh_env)
    assert isinstance(fresh_algo.best_policy_loss, np.ndarray)
    assert fresh_algo.best_policy_loss.dtype == algo.best_policy_loss.dtype
    np.testing.assert_array_equal(fresh_algo.best_policy_loss, algo.best_policy_loss)


@pytest.mark.parametrize("corrupt", ["param", "rms", "env"])
def test_nonfinite_learned_or_physical_state_is_refused_at_capture(corrupt):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    with torch.no_grad():
        if corrupt == "param":
            next(iter(algo.actor.parameters()))[0] = float("nan")
        elif corrupt == "rms":
            algo.obs_rms.var[0] = float("inf")
        else:
            env.sim_qvel[0, 0] = float("nan")
    with pytest.raises(ppo_resume.ResumeValidationError, match="nonfinite"):
        _capture(algo, env, epoch=1)


def test_unknown_buffer_attribute_is_refused(tmp_path):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    algo.buf_experimental = torch.zeros(2)
    with pytest.raises(ppo_resume.ResumeValidationError, match="buf_experimental"):
        _capture(algo, env, epoch=1)


def test_segment_file_loads_with_weights_only(tmp_path):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    path = tmp_path / "weights_only.ptc"
    ppo_resume.write_segment(_capture(algo, env, epoch=1), path)
    container = torch.load(path, weights_only=True)
    assert container["schema_version"] == ppo_resume.SCHEMA_VERSION
    assert isinstance(container["payload"], (bytes, bytearray))
    inner = torch.load(io.BytesIO(container["payload"]), weights_only=True)
    assert set(inner) == {"metadata", "algo", "env", "extra", "metadata_sha256"}


def test_analysis_package_does_not_import_ppo_resume(monkeypatch):
    """``msk_warp.analysis`` must stay torch/warp-free (``test_orbit.py:304``).

    The purge is undone on teardown via ``monkeypatch.delitem``. An unrestored
    purge leaked into every later test in the process: a test that lazily imports
    an ``msk_warp.analysis`` module and patches it would then patch a **stale**
    object that the code under test never touches, and pass while asserting
    nothing. The assertion below is unchanged; only the cleanup was added.
    """
    for name in list(sys.modules):
        if name.startswith("msk_warp.analysis"):
            monkeypatch.delitem(sys.modules, name)
    import msk_warp.analysis  # noqa: F401

    assert "msk_warp.analysis.ppo_resume" not in sys.modules


# ----------------------------------------------------------------------
# 4. Publication, checksum and validation failure paths
# ----------------------------------------------------------------------


def _segment(tmp_path, name="seg.ptc"):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    path = tmp_path / name
    ppo_resume.write_segment(_capture(algo, env, epoch=1), path)
    return path


def test_publication_never_overwrites_an_existing_file(tmp_path):
    path = _segment(tmp_path)
    original = path.read_bytes()
    algo, env = _build(seed=77)
    _start(algo, env)
    _train(algo, 1)
    with pytest.raises(FileExistsError):
        ppo_resume.write_segment(_capture(algo, env, epoch=1), path)
    assert path.read_bytes() == original
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


def test_partial_disk_write_leaves_neither_final_nor_temporary(tmp_path, monkeypatch):
    """D3: fail *after* the temp is open and partly written, not in ``_container``.

    The previous version of this test patched ``torch.save`` globally, so it blew
    up inside ``_container()`` — before ``open(temp, 'xb')`` was ever reached —
    and the temp-cleanup path it claimed to cover never ran. The injection is now
    gated on the destination being a real file handle, and the test asserts that
    the injection actually fired there.
    """
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    path = tmp_path / "partial.ptc"
    fired = {}

    real_save = torch.save

    def exploding_save(obj, target, *args, **kwargs):
        # BytesIO also exposes .fileno(), so the in-memory payload must be
        # excluded explicitly — that aliasing is what made the old test vacuous.
        if hasattr(target, "fileno") and not isinstance(target, io.BytesIO):
            target.write(b"partial segment bytes")
            target.flush()
            fired["after_open"] = True
            fired["bytes_on_disk"] = os.fstat(target.fileno()).st_size
            raise OSError("simulated disk failure mid-write")
        return real_save(obj, target, *args, **kwargs)

    monkeypatch.setattr(ppo_resume.torch, "save", exploding_save)
    with pytest.raises(OSError, match="mid-write"):
        ppo_resume.write_segment(state, path)
    monkeypatch.undo()

    assert fired.get("after_open") is True, "injection never reached the open temp file"
    assert fired["bytes_on_disk"] > 0, "nothing was written, so cleanup was not exercised"
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_fsync_failure_leaves_neither_final_nor_temporary(tmp_path, monkeypatch):
    """D3: the durability step failing must also publish nothing."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    path = tmp_path / "fsync.ptc"
    monkeypatch.setattr(ppo_resume.os, "fsync",
                        lambda fd: (_ for _ in ()).throw(OSError("simulated fsync failure")))
    with pytest.raises(OSError, match="fsync"):
        ppo_resume.write_segment(state, path)
    monkeypatch.undo()
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_publication_failure_cleans_up_and_publishes_nothing(tmp_path, monkeypatch):
    """D3: both the atomic link and the no-replace fallback failing."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    path = tmp_path / "publish.ptc"

    def refuse(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(ppo_resume.os, "link", refuse)
    monkeypatch.setattr(ppo_resume.os, "rename", refuse)
    with pytest.raises(OSError, match="publication failure"):
        ppo_resume.write_segment(state, path)
    monkeypatch.undo()
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_losing_a_publication_race_never_overwrites_the_winner(tmp_path, monkeypatch):
    """D3: the non-overwrite guarantee is the atomic link, not the exists() pre-check."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    path = tmp_path / "raced.ptc"
    winner = b"another writer's complete segment"

    def losing_link(source, destination):
        # Simulate a competitor publishing between the pre-check and the link.
        Path(destination).write_bytes(winner)
        raise FileExistsError(destination)

    monkeypatch.setattr(ppo_resume.os, "link", losing_link)
    with pytest.raises(FileExistsError):
        ppo_resume.write_segment(state, path)
    monkeypatch.undo()
    assert path.read_bytes() == winner
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


def test_a_partially_written_file_can_never_be_restored(tmp_path):
    """D3: publication failure means no reader can reach a restore target."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    truncated = tmp_path / "truncated.ptc"
    ppo_resume.write_segment(_capture(algo, env, epoch=1), truncated)
    data = truncated.read_bytes()
    truncated.unlink()
    truncated.write_bytes(data[: len(data) // 3])

    target_algo, target_env = _build(seed=91)
    before = _fingerprint(target_algo, target_env)
    with pytest.raises(ppo_resume.ResumeIntegrityError):
        ppo_resume.restore_state(ppo_resume.read_segment(truncated), target_algo, target_env)
    assert _differences(before, _fingerprint(target_algo, target_env)) == []


# ----------------------------------------------------------------------
# 4b. Git identity hook (review defect D4)
# ----------------------------------------------------------------------


def _git(*args, cwd):
    import subprocess

    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture
def throwaway_repo(tmp_path):
    """A disposable repository: the project repo, its config and HEAD are untouched."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    _git("-c", "init.defaultBranch=main", "init", "-q", ".", cwd=root)
    return root


def test_git_identity_reports_head_blob_and_uncommitted_status(throwaway_repo):
    """D4: ``git ls-files -s`` reports the INDEX, which is the empty blob under -N."""
    relative = "pkg/module.py"
    target = throwaway_repo / relative
    target.write_bytes(b"print('one')\n")
    _git("add", "-N", "--", relative, cwd=throwaway_repo)

    record = ppo_resume._git_identity(throwaway_repo, (relative,))
    content_id = _git("hash-object", "--", relative, cwd=throwaway_repo)
    assert record["working_tree_blob_sha1"][relative] == content_id
    assert record["working_tree_blob_sha1"][relative] != "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert record["head_blob_sha1"][relative] is None
    assert record["head_status"][relative] == "not_in_head"


def test_git_identity_separates_head_blob_from_a_modified_working_tree(throwaway_repo):
    """D4: staged-vs-HEAD must not be collapsed into one ambiguous blob claim."""
    relative = "pkg/module.py"
    target = throwaway_repo / relative
    target.write_bytes(b"print('one')\n")
    _git("add", "--", relative, cwd=throwaway_repo)
    _git("-c", "user.email=t@example.invalid", "-c", "user.name=T", "commit", "-q",
         "-m", "add", cwd=throwaway_repo)
    committed = _git("rev-parse", "HEAD:" + relative, cwd=throwaway_repo)

    target.write_bytes(b"print('two')\n")  # working tree now differs from HEAD
    record = ppo_resume._git_identity(throwaway_repo, (relative,))
    assert record["head_blob_sha1"][relative] == committed
    assert record["head_status"][relative] == "in_head"
    assert record["working_tree_blob_sha1"][relative] == _git(
        "hash-object", "--", relative, cwd=throwaway_repo)
    assert record["working_tree_blob_sha1"][relative] != committed


def test_include_git_identity_records_both_identities_distinctly(tmp_path):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1, include_git=True)
    sources = state.metadata["sources"]
    git_record = state.metadata["sources_git"]
    assert sources["kind"] == "raw_working_tree_bytes"
    assert git_record["kind"] == "git_object_identity"
    assert set(git_record) >= {"kind", "working_tree_blob_sha1", "head_blob_sha1", "head_status"}
    # Raw execution bytes and git object identity are different records entirely.
    assert set(sources["sha256"]) == set(git_record["working_tree_blob_sha1"])
    for name, digest in sources["sha256"].items():
        assert git_record["working_tree_blob_sha1"][name] != digest


# ----------------------------------------------------------------------
# 4c. Strict identity binding for a real resume
# ----------------------------------------------------------------------


def test_strict_expect_binds_every_required_identity_key(tmp_path):
    """What Task 4 must pass; ``expect=None`` is inspection mode, not a resume gate."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    path = tmp_path / "strict.ptc"
    published = ppo_resume.write_segment(
        _capture(algo, env, epoch=1, segment_index=4, parent_segment_sha256="abc123"), path)
    assert published["sha256"]

    expect = ppo_resume.strict_expect(epoch=1, seed=7, segment_index=4,
                                      parent_segment_sha256="abc123")
    assert set(expect) == set(ppo_resume.REQUIRED_IDENTITY_KEYS)
    loaded = ppo_resume.read_segment(path, expect=expect)
    assert loaded.metadata["epoch"] == 1

    for key, wrong in (("epoch", 2), ("seed", 8), ("segment_index", 5),
                       ("parent_segment_sha256", "def456")):
        broken = dict(expect)
        broken[key] = wrong
        with pytest.raises(ppo_resume.ResumeValidationError, match=key):
            ppo_resume.read_segment(path, expect=broken)


def test_default_read_is_inspection_only_and_does_not_check_lineage(tmp_path):
    """Documented boundary: the helper does not bind lineage by itself."""
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    path = tmp_path / "inspect.ptc"
    ppo_resume.write_segment(
        _capture(algo, env, epoch=1, segment_index=9, parent_segment_sha256="whatever"), path)
    loaded = ppo_resume.read_segment(path)  # no expect: checksums and schema only
    assert loaded.metadata["extra"]["parent_segment_sha256"] == "whatever"
    assert loaded.metadata["epoch"] == 1


def test_reader_rejects_truncated_file_before_touching_target(tmp_path):
    path = _segment(tmp_path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    algo, env = _build(seed=31)
    before = _fingerprint(algo, env)
    with pytest.raises(ppo_resume.ResumeIntegrityError):
        ppo_resume.restore_state(ppo_resume.read_segment(path), algo, env)
    assert _differences(before, _fingerprint(algo, env)) == []


def test_reader_rejects_corrupted_payload(tmp_path):
    path = _segment(tmp_path)
    container = torch.load(path, weights_only=True)
    payload = bytearray(container["payload"])
    payload[len(payload) // 2] ^= 0xFF
    container["payload"] = bytes(payload)
    path.unlink()
    torch.save(container, path)
    with pytest.raises(ppo_resume.ResumeIntegrityError, match="checksum"):
        ppo_resume.read_segment(path)


def test_reader_rejects_tampered_metadata(tmp_path):
    """The payload commits to the metadata, so the outer copy cannot be edited."""
    path = _segment(tmp_path)
    container = torch.load(path, weights_only=True)
    container["metadata"] = dict(container["metadata"])
    container["metadata"]["epoch"] = 99
    path.unlink()
    torch.save(container, path)
    with pytest.raises(ppo_resume.ResumeIntegrityError, match="metadata"):
        ppo_resume.read_segment(path)


def test_reader_rejects_missing_file(tmp_path):
    with pytest.raises(ppo_resume.ResumeValidationError):
        ppo_resume.read_segment(tmp_path / "absent.ptc")


def test_reader_rejects_wrong_identity_and_lineage(tmp_path):
    path = _segment(tmp_path)
    with pytest.raises(ppo_resume.ResumeValidationError, match="epoch"):
        ppo_resume.read_segment(path, expect={"epoch": 99})
    with pytest.raises(ppo_resume.ResumeValidationError, match="parent_segment_sha256"):
        ppo_resume.read_segment(path, expect={"parent_segment_sha256": "deadbeef"})
    with pytest.raises(ppo_resume.ResumeValidationError, match="schema_version"):
        ppo_resume.read_segment(path, expect={"schema_version": "myoleg26-ppo-resume-v0"})


def test_restore_rejects_shape_mismatch_before_mutating_target(tmp_path):
    path = _segment(tmp_path)
    algo, env = _build(seed=41, num_envs=8)  # different world count
    before = _fingerprint(algo, env)
    with pytest.raises(ppo_resume.ResumeValidationError, match="shape"):
        ppo_resume.restore_state(ppo_resume.read_segment(path), algo, env)
    assert _differences(before, _fingerprint(algo, env)) == []


def test_restore_rejects_dtype_mismatch_before_mutating_target(tmp_path):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    state.env["plain"]["sim_qvel"] = state.env["plain"]["sim_qvel"].double()
    target_algo, target_env = _build(seed=43)
    before = _fingerprint(target_algo, target_env)
    with pytest.raises(ppo_resume.ResumeValidationError, match="dtype"):
        ppo_resume.restore_state(state, target_algo, target_env)
    assert _differences(before, _fingerprint(target_algo, target_env)) == []


def test_restore_rejects_a_changed_recipe_before_mutating_target(tmp_path):
    """Construction inputs are caller-owned, so they are checked, not assumed."""
    path = _segment(tmp_path)
    algo, env = _build(seed=61)
    algo.num_minibatches = 4
    before = _fingerprint(algo, env)
    with pytest.raises(ppo_resume.ResumeValidationError, match="recipe mismatch on num_minibatches"):
        ppo_resume.restore_state(ppo_resume.read_segment(path), algo, env)
    assert _differences(before, _fingerprint(algo, env)) == []


def test_restore_rejects_unknown_env_state_protocol(tmp_path):
    path = _segment(tmp_path)

    class _Opaque:
        num_envs = 4
        num_actions = 3

    algo = _build(seed=51)[0]
    opaque = _Opaque()
    # Bind the algorithm to the object under test: the subject here is the
    # unknown state protocol, not the algo/env binding gap 6 guards separately.
    algo.env = opaque
    with pytest.raises(ppo_resume.ResumeValidationError, match="state protocol"):
        ppo_resume.restore_state(ppo_resume.read_segment(path), algo, opaque)


def test_metadata_records_recipe_and_source_identity(tmp_path):
    algo, env = _build()
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1)
    metadata = state.metadata
    assert metadata["schema_version"] == ppo_resume.SCHEMA_VERSION
    assert metadata["epoch"] == 1
    assert metadata["iter_count"] == 1
    recipe = metadata["recipe"]
    assert recipe["steps_num"] == algo.steps_num
    assert recipe["num_minibatches"] == algo.num_minibatches
    assert recipe["obs_rms"] is True and recipe["ret_rms"] is False
    # Raw working-tree bytes and git blob identity stay distinct (IN-25).
    sources = metadata["sources"]
    assert sources["kind"] == "raw_working_tree_bytes"
    assert "msk_warp/analysis/ppo_resume.py" in sources["sha256"]
    assert "git_blob_sha1" not in sources
    assert metadata["extra"]["seed"] == 7


# ----------------------------------------------------------------------
# 4d. The algorithm must be bound to the supplied environment (gap 6)
# ----------------------------------------------------------------------


def test_capture_refuses_an_algorithm_bound_to_a_different_environment():
    """Gap 6: a same-shape wrong env would be combined with the algorithm state.

    Every dtype/shape/recipe gate passes for two adapters built with identical
    arguments, so without an identity check the segment would pair one run's
    parameters, Adam moments, normaliser and RNG with another run's physics and
    call it a faithful boundary.
    """
    algo_a, env_a = _build(seed=7)
    _start(algo_a, env_a)
    _train(algo_a, 2)
    algo_b, env_b = _build(seed=23)      # identical shapes, different state
    _start(algo_b, env_b)
    _train(algo_b, 1)
    # The confound this guard removes: the two envs really do differ.
    assert not torch.equal(env_a.sim_qpos, env_b.sim_qpos)

    before_a = _fingerprint(algo_a, env_a)
    before_b = _fingerprint(algo_b, env_b)
    before_global = _global_rng_fingerprint()

    with pytest.raises(ppo_resume.ResumeValidationError, match="algo.env"):
        ppo_resume.capture_state(algo_a, env_b, extra=_metadata(), epoch=2)

    assert _differences(before_a, _fingerprint(algo_a, env_a)) == []
    assert _differences(before_b, _fingerprint(algo_b, env_b)) == []
    _assert_global_rng_unchanged(before_global)
    # The refusal must not silently rebind the algorithm to the argument.
    assert algo_a.env is env_a and algo_b.env is env_b


def test_restore_refuses_an_algorithm_bound_to_a_different_environment(tmp_path):
    """Gap 6: restoring physics into one env while the algorithm steps another."""
    path = _segment(tmp_path)
    loaded = ppo_resume.read_segment(path)

    algo_b, env_b = _build(seed=71)      # the algorithm's own environment
    _start(algo_b, env_b)
    algo_c, env_c = _build(seed=72)      # same shape, a different object
    _start(algo_c, env_c)

    before_b = _fingerprint(algo_b, env_b)
    before_c = _fingerprint(algo_c, env_c)
    before_global = _global_rng_fingerprint()

    with pytest.raises(ppo_resume.ResumeValidationError, match="algo.env"):
        ppo_resume.restore_state(loaded, algo_b, env_c)

    assert _differences(before_b, _fingerprint(algo_b, env_b)) == []
    assert _differences(before_c, _fingerprint(algo_c, env_c)) == []
    _assert_global_rng_unchanged(before_global)
    assert algo_b.env is env_b and algo_c.env is env_c


def test_capture_and_restore_refuse_an_algorithm_with_no_environment_binding(tmp_path):
    """Absence of ``algo.env`` cannot be read as agreement; it is a refusal."""
    path = _segment(tmp_path)
    algo, env = _build(seed=73)
    _start(algo, env)
    del algo.env
    before = _fingerprint(algo, env)
    before_global = _global_rng_fingerprint()

    with pytest.raises(ppo_resume.ResumeValidationError, match="algo.env"):
        ppo_resume.capture_state(algo, env, extra=_metadata(), epoch=0)
    with pytest.raises(ppo_resume.ResumeValidationError, match="algo.env"):
        ppo_resume.restore_state(ppo_resume.read_segment(path), algo, env)

    assert _differences(before, _fingerprint(algo, env)) == []
    _assert_global_rng_unchanged(before_global)
    assert not hasattr(algo, "env")      # never repaired behind the caller's back


def test_a_bound_pair_still_captures_and_restores():
    """Control: the guard is identity-based, so a legitimate pair is unaffected."""
    algo, env = _build(seed=74)
    _start(algo, env)
    _train(algo, 1)
    state = ppo_resume.capture_state(algo, env, extra=_metadata(), epoch=1)
    target_algo, target_env = _build(seed=75)
    _start(target_algo, target_env)
    report = ppo_resume.restore_state(state, target_algo, target_env)
    assert report["iter_count"] == 1


# ----------------------------------------------------------------------
# 4e. Required source provenance is complete or refused (gap 5)
# ----------------------------------------------------------------------


def test_source_hashes_requires_every_declared_file(tmp_path):
    """Gap 5: a missing required source must fault, not drop a key.

    Controlled paths under ``tmp_path``; no project file is renamed or deleted.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "present.py").write_bytes(b"x = 1\n")
    (tmp_path / "pkg" / "a_directory.py").mkdir()

    complete = ppo_resume._source_hashes(root=tmp_path, files=("pkg/present.py",))
    assert set(complete) == {"pkg/present.py"}
    assert complete["pkg/present.py"] == hashlib.sha256(b"x = 1\n").hexdigest()

    with pytest.raises(ppo_resume.ResumeValidationError, match="pkg/absent.py"):
        ppo_resume._source_hashes(root=tmp_path,
                                  files=("pkg/present.py", "pkg/absent.py"))
    # A directory at a declared source path is not a readable source either.
    with pytest.raises(ppo_resume.ResumeValidationError, match="pkg/a_directory.py"):
        ppo_resume._source_hashes(root=tmp_path, files=("pkg/a_directory.py",))


def test_capture_refuses_a_missing_required_source_and_never_records_a_partial_map(monkeypatch):
    """Gap 5 at the capture seam, via a declared path that does not exist.

    ``SOURCE_FILES`` is monkeypatched rather than deleting a real source, so the
    project tree, the backend and the v1 pins are untouched.
    """
    algo, env = _build(seed=76)
    _start(algo, env)
    _train(algo, 1)
    absent = "msk_warp/analysis/ppo_resume_absent_source.py"
    assert not (ROOT / absent).exists()
    monkeypatch.setattr(ppo_resume, "SOURCE_FILES", ppo_resume.SOURCE_FILES + (absent,))

    with pytest.raises(ppo_resume.ResumeValidationError, match="ppo_resume_absent_source"):
        ppo_resume.capture_state(algo, env, extra=_metadata(), epoch=1)

    monkeypatch.undo()
    state = ppo_resume.capture_state(algo, env, extra=_metadata(), epoch=1)
    recorded = state.metadata["sources"]["sha256"]
    # Complete by construction: every declared source, no quietly omitted key.
    assert set(recorded) == set(ppo_resume.SOURCE_FILES)
    assert all(len(digest) == 64 for digest in recorded.values())


def test_git_identity_covers_the_same_declared_file_set_as_the_raw_bytes(monkeypatch):
    """The two provenance records must describe one file set, resolved together."""
    subset = ("msk_warp/utils/average_meter.py", "msk_warp/utils/running_mean_std.py")
    monkeypatch.setattr(ppo_resume, "SOURCE_FILES", subset)
    algo, env = _build(seed=77)
    _start(algo, env)
    _train(algo, 1)
    state = _capture(algo, env, epoch=1, include_git=True)
    git_record = state.metadata["sources_git"]
    assert set(state.metadata["sources"]["sha256"]) == set(subset)
    assert set(git_record["working_tree_blob_sha1"]) == set(subset)
    assert set(git_record["head_blob_sha1"]) == set(subset)
    assert set(git_record["head_status"]) == set(subset)


def test_git_identity_labels_a_missing_working_tree_file_distinctly(throwaway_repo):
    """A file that is absent is not "git unavailable": the status must say which."""
    relative = "pkg/absent.py"
    record = ppo_resume._git_identity(throwaway_repo, (relative,))
    assert record["working_tree_blob_sha1"][relative] is None
    assert record["head_blob_sha1"][relative] is None
    assert record["head_status"][relative] == "missing_in_working_tree"
