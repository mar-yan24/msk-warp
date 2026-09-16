"""CPU tests for the read-only PPO observability helpers.

No GPU, no training, no sampling. Every expected number is either a
hand-computed closed-form Gaussian literal or recomputed in the test from an
independent formula; nothing is oracled by calling the function under test.
The state-neutrality test proves the helpers touch no parameter, gradient,
optimizer state, normalizer, module mode, rollout buffer or random stream.
"""

from __future__ import annotations

import copy
import json
import math
import random

import numpy as np
import pytest
import torch

from msk_warp.analysis import ppo_diagnostics
from msk_warp.analysis.myoleg26_baseline import isolated_rng
from msk_warp.analysis.ppo_diagnostics import (
    EPISODE_EVENT_FIELDS,
    OFFICIAL_FAILURE_FLAG_NAMES,
    approx_kl_clip,
    episode_event_summary,
    logstd_summary,
    near_bound_excitation,
)
from msk_warp.networks.actor import ActorDeterministicMLP, ActorStochasticMLP
from msk_warp.networks.critic import CriticMLP
from msk_warp.utils.running_mean_std import RunningMeanStd

CFG_NETWORK = {
    "actor_mlp": {"units": [8, 8], "activation": "elu"},
    "critic_mlp": {"units": [8, 8], "activation": "elu"},
    "actor_logstd_init": -1.0,
}

LOG_2PI = math.log(2.0 * math.pi)  # 1.8378770664093453


@pytest.fixture(autouse=True)
def _restore_global_random_streams():
    """This module builds networks, so it must not advance the suite's streams."""
    with isolated_rng():
        yield


class _CannedActor:
    """Returns exactly the supplied log-probs, isolating the ratio arithmetic."""

    def __init__(self, log_probs, entropy):
        self._log_probs = torch.as_tensor(log_probs, dtype=torch.float64)
        self._entropy = torch.as_tensor(entropy, dtype=torch.float64)
        self.training = True

    def get_logstd(self):
        return None

    def evaluate_actions(self, obs, actions):
        return self._log_probs, self._entropy


class _GaussianActor:
    """Fixed diagonal Gaussian in pre-tanh space; no weights and no sampling."""

    def __init__(self, mu, logstd, action_dim):
        self._mu = float(mu)
        self._logstd = torch.nn.Parameter(
            torch.full((action_dim,), float(logstd), dtype=torch.float64))
        self.training = True

    def get_logstd(self):
        return self._logstd

    def evaluate_actions(self, obs, actions):
        std = self._logstd.exp()
        dist = torch.distributions.Normal(torch.full_like(actions, self._mu), std)
        return dist.log_prob(actions).sum(dim=-1), dist.entropy().sum(dim=-1)


class _RecordingActor:
    """Captures exactly what the helper passed in."""

    def __init__(self, samples):
        self.seen_obs = None
        self.seen_actions = None
        self.grad_enabled_during_call = None
        self._samples = samples
        self.training = True

    def get_logstd(self):
        return None

    def evaluate_actions(self, obs, actions):
        self.seen_obs = obs
        self.seen_actions = actions
        self.grad_enabled_during_call = torch.is_grad_enabled()
        return (torch.zeros(self._samples, dtype=obs.dtype),
                torch.zeros(self._samples, dtype=obs.dtype))


# --------------------------------------------------------------------------- #
# approx KL / clip fraction
# --------------------------------------------------------------------------- #

def _closed_form_case():
    """Behaviour policy mu=0 std=1; post-update policy mu=0.5 std=1; two dims.

    actions [[0, 0], [1, 1]]
      old log-probs : -log(2pi) = -1.8378770664093453
                      -1 - log(2pi) = -2.8378770664093453
      new log-probs : -0.25 - log(2pi) = -2.0878770664093453  (both rows)
      d = new - old : [-0.25, +0.75]
    """
    actions = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float64)
    obs = torch.zeros((1, 2, 3), dtype=torch.float64)
    old = torch.tensor([[-LOG_2PI, -1.0 - LOG_2PI]], dtype=torch.float64)
    return obs, actions, old


def test_approx_kl_and_entropy_match_hand_computed_closed_form_gaussians():
    obs, actions, old = _closed_form_case()
    actor = _GaussianActor(mu=0.5, logstd=0.0, action_dim=2)

    report = approx_kl_clip(actor, obs, actions, old, clip_range=0.2)

    assert report["status"] == "ok"
    assert report["samples"] == 2
    # k1 = mean(old - new) = mean(0.25, -0.75) = -0.25
    assert report["approx_kl_k1"] == pytest.approx(-0.25, abs=1e-12)
    # k3 = mean(exp(d) - 1 - d) with d = [-0.25, 0.75]
    expected_k3 = ((math.exp(-0.25) - 1 + 0.25) + (math.exp(0.75) - 1 - 0.75)) / 2
    assert expected_k3 == pytest.approx(0.19790039984203992, abs=1e-15)
    assert report["approx_kl_k3"] == pytest.approx(expected_k3, abs=1e-12)
    # Both |ratio - 1| exceed 0.2 -> every sample is clipped.
    assert report["clip_fraction"] == pytest.approx(1.0, abs=0.0)
    # Pre-tanh convention: no tanh Jacobian term anywhere.
    assert report["mean_new_log_prob"] == pytest.approx(-2.0878770664093453, abs=1e-12)
    assert report["mean_old_log_prob"] == pytest.approx(-2.3378770664093453, abs=1e-12)
    # Entropy of a unit-variance 2-D diagonal Gaussian is log(2*pi*e).
    assert report["mean_entropy"] == pytest.approx(1.0 + LOG_2PI, abs=1e-12)
    assert report["mean_ratio"] == pytest.approx(
        (math.exp(-0.25) + math.exp(0.75)) / 2, abs=1e-12)
    assert json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("clip_range,expected", [(0.2, 1.0), (0.3, 0.5), (1.2, 0.0)])
def test_clip_fraction_counts_only_samples_beyond_the_bound(clip_range, expected):
    """|ratio - 1| is 0.2211992169285951 and 1.117000016612675."""
    obs, actions, old = _closed_form_case()
    actor = _GaussianActor(mu=0.5, logstd=0.0, action_dim=2)

    report = approx_kl_clip(actor, obs, actions, old, clip_range=clip_range)

    assert report["clip_fraction"] == pytest.approx(expected, abs=0.0)
    assert report["clip_range"] == pytest.approx(clip_range, abs=0.0)


def test_a_sample_exactly_at_the_bound_is_not_counted_as_clipped():
    """exp(0) is exactly 1.0, so |ratio - 1| == 0.0 with no rounding slack."""
    obs = torch.zeros((1, 2, 3), dtype=torch.float64)
    actions = torch.zeros((1, 2, 2), dtype=torch.float64)
    old = torch.tensor([[-0.5, 0.25]], dtype=torch.float64)
    unchanged = _CannedActor(log_probs=[-0.5, 0.25], entropy=[1.0, 1.0])

    at_bound = approx_kl_clip(unchanged, obs, actions, old, clip_range=0.0)
    assert at_bound["clip_fraction"] == pytest.approx(0.0, abs=0.0)
    assert at_bound["approx_kl_k1"] == pytest.approx(0.0, abs=0.0)
    assert at_bound["approx_kl_k3"] == pytest.approx(0.0, abs=0.0)

    nudged = _CannedActor(log_probs=[-0.5 + 1e-12, 0.25], entropy=[1.0, 1.0])
    just_outside = approx_kl_clip(nudged, obs, actions, old, clip_range=0.0)
    assert just_outside["clip_fraction"] == pytest.approx(0.5, abs=0.0)


def test_clip_boundary_just_inside_and_just_outside_a_realistic_bound():
    obs, actions, old = _closed_form_case()
    actor = _GaussianActor(mu=0.5, logstd=0.0, action_dim=2)
    deviation = abs(math.exp(0.75) - 1.0)  # the larger of the two

    inside = approx_kl_clip(actor, obs, actions, old,
                            clip_range=deviation * (1.0 + 1e-9))
    outside = approx_kl_clip(actor, obs, actions, old,
                             clip_range=deviation * (1.0 - 1e-9))

    assert inside["clip_fraction"] == pytest.approx(0.0, abs=0.0)
    assert outside["clip_fraction"] == pytest.approx(0.5, abs=0.0)


def test_the_helper_evaluates_the_stored_normalized_obs_and_pre_tanh_actions():
    obs = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3)
    actions = torch.linspace(-2.0, 2.0, 16, dtype=torch.float64).reshape(4, 2, 2)
    old = torch.zeros((4, 2), dtype=torch.float64)
    actor = _RecordingActor(samples=8)

    report = approx_kl_clip(actor, obs, actions, old, clip_range=0.2)

    assert report["samples"] == 8
    assert torch.equal(actor.seen_obs, obs.reshape(8, 3))
    assert torch.equal(actor.seen_actions, actions.reshape(8, 2))
    assert actor.grad_enabled_during_call is False
    assert not actor.seen_obs.requires_grad
    assert not actor.seen_actions.requires_grad


def test_a_deterministic_actor_is_an_explicit_unavailable_status():
    obs, actions, old = _closed_form_case()
    actor = ActorDeterministicMLP(3, 2, CFG_NETWORK, device="cpu")

    report = approx_kl_clip(actor, obs.float(), actions.float(), old.float(), clip_range=0.2)

    assert report["status"] == "actor_has_no_evaluate_actions"
    assert report["approx_kl_k1"] is None
    assert report["approx_kl_k3"] is None
    assert report["clip_fraction"] is None
    assert json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mutate,status", [
    (lambda obs, act, old: (obs[:, :0], act[:, :0], old[:, :0]), "no_samples"),
    (lambda obs, act, old: (obs, act, old * float("nan")), "nonfinite_input"),
    (lambda obs, act, old: (obs * float("inf"), act, old), "nonfinite_input"),
    (lambda obs, act, old: (obs, act[:, :1], old), "shape_mismatch"),
])
def test_degenerate_inputs_report_a_status_instead_of_fabricating_zeros(mutate, status):
    obs, actions, old = mutate(*_closed_form_case())
    actor = _GaussianActor(mu=0.5, logstd=0.0, action_dim=2)

    report = approx_kl_clip(actor, obs, actions, old, clip_range=0.2)

    assert report["status"] == status
    for key in ("approx_kl_k1", "approx_kl_k3", "clip_fraction", "mean_entropy"):
        assert report[key] is None, key
    assert json.dumps(report, allow_nan=False)


def test_an_overflowing_ratio_is_reported_as_null_with_a_status_not_a_quiet_zero():
    obs = torch.zeros((1, 2, 3), dtype=torch.float64)
    actions = torch.zeros((1, 2, 2), dtype=torch.float64)
    old = torch.tensor([[-1e4, 0.0]], dtype=torch.float64)
    actor = _CannedActor(log_probs=[1e4, 0.0], entropy=[1.0, 1.0])

    report = approx_kl_clip(actor, obs, actions, old, clip_range=0.2)

    assert report["status"] == "nonfinite_derived"
    assert report["nonfinite_ratio_samples"] == 1
    assert report["approx_kl_k3"] is None
    assert report["mean_ratio"] is None
    assert report["clip_fraction"] is None
    # k1 needs no exponential and stays available.
    assert report["approx_kl_k1"] == pytest.approx(-1e4, abs=1e-6)
    assert json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("clip_range", [-0.1, float("nan"), float("inf")])
def test_an_invalid_clip_range_is_a_status_not_an_exception(clip_range):
    obs, actions, old = _closed_form_case()
    actor = _GaussianActor(mu=0.5, logstd=0.0, action_dim=2)

    report = approx_kl_clip(actor, obs, actions, old, clip_range=clip_range)

    assert report["status"] == "invalid_clip_range"
    assert report["clip_fraction"] is None
    assert json.dumps(report, allow_nan=False)


# --------------------------------------------------------------------------- #
# logstd summary
# --------------------------------------------------------------------------- #

def test_logstd_summary_reports_per_dimension_bounds_and_the_std_it_implies():
    actor = ActorStochasticMLP(3, 4, CFG_NETWORK, device="cpu")

    report = logstd_summary(actor)

    assert report["status"] == "ok"
    assert report["dimensions"] == 4
    for key in ("logstd_min", "logstd_mean", "logstd_max"):
        assert report[key] == pytest.approx(-1.0, abs=1e-7)
    for key in ("std_min", "std_mean", "std_max"):
        assert report[key] == pytest.approx(0.36787944117144233, abs=1e-7)
    assert json.dumps(report, allow_nan=False)


def test_logstd_summary_separates_the_dimensions_it_is_given():
    actor = _GaussianActor(mu=0.0, logstd=0.0, action_dim=3)
    with torch.no_grad():
        actor.get_logstd().copy_(torch.tensor([-2.0, -1.0, 0.0], dtype=torch.float64))

    report = logstd_summary(actor)

    assert report["logstd_min"] == pytest.approx(-2.0, abs=1e-12)
    assert report["logstd_mean"] == pytest.approx(-1.0, abs=1e-12)
    assert report["logstd_max"] == pytest.approx(0.0, abs=1e-12)
    assert report["std_min"] == pytest.approx(math.exp(-2.0), abs=1e-12)
    assert report["std_max"] == pytest.approx(1.0, abs=1e-12)


def test_logstd_summary_on_a_deterministic_actor_is_unavailable_not_zero():
    report = logstd_summary(ActorDeterministicMLP(3, 2, CFG_NETWORK, device="cpu"))

    assert report["status"] == "logstd_unavailable"
    assert report["dimensions"] is None
    assert report["logstd_mean"] is None and report["std_mean"] is None
    assert json.dumps(report, allow_nan=False)


def test_logstd_summary_refuses_to_average_a_nonfinite_logstd():
    actor = _GaussianActor(mu=0.0, logstd=0.0, action_dim=2)
    with torch.no_grad():
        actor.get_logstd().copy_(torch.tensor([float("nan"), 0.0], dtype=torch.float64))

    report = logstd_summary(actor)

    assert report["status"] == "nonfinite_logstd"
    assert report["logstd_mean"] is None
    assert json.dumps(report, allow_nan=False)


# --------------------------------------------------------------------------- #
# executed-excitation near-bound fraction
# --------------------------------------------------------------------------- #

def test_near_bound_excitation_uses_tanh_of_the_stored_pre_tanh_actions():
    """excitation = 0.5 * (tanh(p) + 1):
        p =  2.5 -> 0.9933071490757152  near the upper bound
        p =  1.0 -> 0.8807970779778824  interior
        p = -3.0 -> 0.0024726231566347  near the lower bound
        p =  0.0 -> 0.5                 interior
    """
    pre_tanh = torch.tensor([[[2.5, 1.0], [-3.0, 0.0]]], dtype=torch.float64)

    report = near_bound_excitation(pre_tanh, tol=0.02)

    assert report["status"] == "ok"
    assert report["entries"] == 4
    assert report["near_bound_fraction"] == pytest.approx(0.5, abs=0.0)
    assert report["near_upper_fraction"] == pytest.approx(0.25, abs=0.0)
    assert report["near_lower_fraction"] == pytest.approx(0.25, abs=0.0)
    assert report["tol"] == pytest.approx(0.02, abs=0.0)
    assert report["mean_excitation"] == pytest.approx(
        (0.9933071490757152 + 0.8807970779778824 + 0.0024726231566347 + 0.5) / 4, abs=1e-12)
    assert "tanh" in report["definition"]
    assert "rollout" in report["phase"]
    assert json.dumps(report, allow_nan=False)


def test_tanh_saturation_is_reported_separately_and_never_called_clamping():
    """p = 2.0 gives excitation 0.9820137900379085, inside the 0.02 band but |p| is not > 2."""
    pre_tanh = torch.tensor([[2.0, 2.5, 0.0, -3.0]], dtype=torch.float64)

    report = near_bound_excitation(pre_tanh, tol=0.02)

    assert report["near_bound_fraction"] == pytest.approx(0.75, abs=0.0)
    assert report["tanh_saturation_fraction"] == pytest.approx(0.5, abs=0.0)
    assert report["saturation_definition"] == "abs(pre_tanh) > 2"
    joined = json.dumps(report)
    assert "clamp" not in joined.lower()
    assert "saturation" in joined


@pytest.mark.parametrize("tol,expected", [(0.0, 0.0), (0.0179, 0.0), (0.0180, 1.0)])
def test_near_bound_threshold_boundary_just_inside_and_just_outside(tol, expected):
    """tanh(2) = 0.9640275800758169 -> excitation 0.98201379003790845.

    Its distance to the upper bound is 0.01798620996209155, which lies strictly
    between the two tolerances below, so neither comparison is decided by
    floating-point rounding.
    """
    pre_tanh = torch.tensor([[2.0]], dtype=torch.float64)

    report = near_bound_excitation(pre_tanh, tol=tol)

    assert report["near_bound_fraction"] == pytest.approx(expected, abs=0.0)


def test_the_sub_fractions_overlap_at_large_tol_and_must_not_be_summed():
    """At tol = 0.5 an excitation of exactly 0.5 lies within tol of BOTH bounds.

    pre_tanh [0.0, 2.5] -> excitation [0.5, 0.9933071490757152]
      near_upper: 0.5 >= 0.5 True, 0.9933 >= 0.5 True  -> 1.0
      near_lower: 0.5 <= 0.5 True, 0.9933 <= 0.5 False -> 0.5
    The headline near_bound_fraction is the OR of the two masks, so it is 1.0 and
    not double counted; the sub-fractions are reported unrenormalized.
    """
    pre_tanh = torch.tensor([[0.0, 2.5]], dtype=torch.float64)

    report = near_bound_excitation(pre_tanh, tol=0.5)

    assert report["near_upper_fraction"] == pytest.approx(1.0, abs=0.0)
    assert report["near_lower_fraction"] == pytest.approx(0.5, abs=0.0)
    assert report["near_upper_fraction"] + report["near_lower_fraction"] > 1.0
    assert report["near_bound_fraction"] == pytest.approx(1.0, abs=0.0)
    note = report["sub_fraction_note"].lower()
    assert "overlap" in note and "not be summed" in note
    assert "overlap" in near_bound_excitation.__doc__.lower()
    assert json.dumps(report, allow_nan=False)


def test_the_threshold_is_inclusive_exactly_at_the_tolerance():
    """pre_tanh 0 gives excitation exactly 0.5, so the comparison is exact."""
    pre_tanh = torch.zeros((1, 1), dtype=torch.float64)

    assert near_bound_excitation(pre_tanh, tol=0.5)["near_bound_fraction"] == 1.0
    assert near_bound_excitation(pre_tanh, tol=0.49)["near_bound_fraction"] == 0.0


@pytest.mark.parametrize("actions,status", [
    (torch.zeros((0, 4), dtype=torch.float64), "no_samples"),
    (torch.tensor([[float("nan"), 0.0]], dtype=torch.float64), "nonfinite_input"),
    (torch.tensor([[float("inf"), 0.0]], dtype=torch.float64), "nonfinite_input"),
])
def test_near_bound_excitation_degenerate_inputs_report_status(actions, status):
    report = near_bound_excitation(actions, tol=0.02)

    assert report["status"] == status
    assert report["near_bound_fraction"] is None
    assert report["mean_excitation"] is None
    assert json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("tol", [-0.01, 0.6, float("nan")])
def test_near_bound_excitation_refuses_a_meaningless_tolerance(tol):
    report = near_bound_excitation(torch.zeros((2, 2)), tol=tol)

    assert report["status"] == "invalid_tol"
    assert report["near_bound_fraction"] is None


# --------------------------------------------------------------------------- #
# completed-episode events
# --------------------------------------------------------------------------- #

def _events():
    return [
        {"world": 0, "length_controls": 500, "duration_s": 4.0, "return": 3.0,
         "end_reason": "horizon",
         "failure_flags": {"nonfinite": False, "low_pelvis": False, "low_upright": False,
                           "nonfoot_ground_contact": False}},
        {"world": 1, "length_controls": 250, "duration_s": 2.0, "return": -1.5,
         "end_reason": "task_failure",
         "failure_flags": {"nonfinite": False, "low_pelvis": True, "low_upright": False,
                           "nonfoot_ground_contact": False}},
        {"world": 2, "length_controls": 125, "duration_s": 1.0, "return": 0.75,
         "end_reason": "task_failure",
         "failure_flags": {"nonfinite": False, "low_pelvis": False, "low_upright": False,
                           "nonfoot_ground_contact": True}},
    ]


def test_event_summary_over_unequal_length_episodes():
    """durations 4.0/2.0/1.0 -> mean 7/3; returns 3.0/-1.5/0.75 -> mean 0.75."""
    report = episode_event_summary(_events())

    assert report["status"] == "ok"
    assert report["completed_episodes"] == 3
    assert report["mean_duration_s"] == pytest.approx(7.0 / 3.0, abs=1e-12)
    assert report["min_duration_s"] == pytest.approx(1.0, abs=0.0)
    assert report["max_duration_s"] == pytest.approx(4.0, abs=0.0)
    assert report["mean_return"] == pytest.approx(0.75, abs=1e-12)
    assert report["min_return"] == pytest.approx(-1.5, abs=0.0)
    assert report["max_return"] == pytest.approx(3.0, abs=0.0)
    assert report["mean_length_controls"] == pytest.approx(875.0 / 3.0, abs=1e-12)
    assert report["end_reason_counts"] == {"horizon": 1, "task_failure": 2}
    assert report["failure_flag_counts"] == {
        "nonfinite": 0, "low_pelvis": 1, "low_upright": 0, "nonfoot_ground_contact": 1}
    assert report["worlds"] == 3
    assert json.dumps(report, allow_nan=False)


def test_zero_completed_episodes_is_not_zero_duration_or_return():
    report = episode_event_summary([])

    assert report["completed_episodes"] == 0
    assert report["status"] == "no_completed_episodes"
    for key in ("mean_duration_s", "min_duration_s", "max_duration_s", "mean_return",
                "min_return", "max_return", "mean_length_controls"):
        assert report[key] is None, key
    assert report["end_reason_counts"] == {}
    assert report["failure_flag_counts"] == {}
    assert json.dumps(report, allow_nan=False)


def test_event_summary_does_not_read_any_run_lifetime_history_or_capped_meter():
    """The helper is stateless: only the events it is handed can reach the output."""
    events = _events()
    first = episode_event_summary(events)
    second = episode_event_summary(events[:1])

    assert first["completed_episodes"] == 3
    assert second["completed_episodes"] == 1
    assert second["mean_duration_s"] == pytest.approx(4.0, abs=0.0)
    assert episode_event_summary(events) == first
    assert set(EPISODE_EVENT_FIELDS) == {
        "world", "length_controls", "duration_s", "return", "end_reason", "failure_flags"}


def test_legacy_failure_flag_keys_are_counted_but_not_labelled_official():
    events = _events()
    events[1]["failure_flags"] = {"height": True}

    report = episode_event_summary(events)

    assert report["failure_flag_counts"]["height"] == 1
    assert report["unofficial_failure_flag_names"] == ["height"]
    assert "height" not in OFFICIAL_FAILURE_FLAG_NAMES
    assert "height" not in report["official_failure_flag_names"]
    assert set(report["official_failure_flag_names"]).issubset(OFFICIAL_FAILURE_FLAG_NAMES)


@pytest.mark.parametrize("mutation,status", [
    ({"duration_s": float("nan")}, "nonfinite_event_field"),
    ({"return": float("inf")}, "nonfinite_event_field"),
    ({"duration_s": "4.0"}, "malformed_event"),
    ({"length_controls": 0}, "malformed_event"),
    ({"length_controls": 1.5}, "malformed_event"),
    ({"end_reason": 7}, "malformed_event"),
    ({"failure_flags": ["low_pelvis"]}, "malformed_event"),
])
def test_malformed_events_report_status_without_raising_into_the_training_loop(mutation, status):
    events = _events()
    events[1].update(mutation)

    report = episode_event_summary(events)

    assert report["status"] == status
    assert report["mean_duration_s"] is None
    assert report["completed_episodes"] == 3
    assert json.dumps(report, allow_nan=False)


def test_a_missing_event_field_is_reported_not_defaulted():
    events = _events()
    del events[2]["return"]

    report = episode_event_summary(events)

    assert report["status"] == "malformed_event"
    assert "return" in report["detail"]
    assert report["mean_return"] is None


def test_events_must_be_a_sequence_of_mappings():
    report = episode_event_summary([_events()[0], 7])

    assert report["status"] == "malformed_event"
    assert report["mean_duration_s"] is None


# --------------------------------------------------------------------------- #
# state neutrality
# --------------------------------------------------------------------------- #

def _same(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and bool(torch.equal(left, right)))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and set(left) == set(right)
                and all(_same(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _snapshot(actor, critic, actor_opt, critic_opt, obs_rms, ret_rms, buffers):
    return {
        "actor_params": {name: p.detach().clone()
                         for name, p in actor.named_parameters()},
        "critic_params": {name: p.detach().clone()
                          for name, p in critic.named_parameters()},
        "actor_grads": {name: (None if p.grad is None else p.grad.detach().clone())
                        for name, p in actor.named_parameters()},
        "critic_grads": {name: (None if p.grad is None else p.grad.detach().clone())
                         for name, p in critic.named_parameters()},
        "requires_grad": {name: bool(p.requires_grad)
                          for name, p in list(actor.named_parameters())
                          + list(critic.named_parameters())},
        "actor_opt": copy.deepcopy(actor_opt.state_dict()),
        "critic_opt": copy.deepcopy(critic_opt.state_dict()),
        "obs_rms": (obs_rms.mean.clone(), obs_rms.var.clone(), float(obs_rms.count)),
        "ret_rms": (ret_rms.mean.clone(), ret_rms.var.clone(), float(ret_rms.count)),
        "modes": (bool(actor.training), bool(critic.training)),
        "buffers": {name: tensor.detach().clone() for name, tensor in buffers.items()},
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
        # Never initialize CUDA merely to inspect an unused stream.
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_initialized() else None),
    }


def test_the_observer_helpers_change_no_parameter_gradient_optimizer_rms_mode_or_rng():
    torch.manual_seed(1234)
    actor = ActorStochasticMLP(6, 3, CFG_NETWORK, device="cpu")
    critic = CriticMLP(6, CFG_NETWORK, device="cpu")
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4, betas=(0.7, 0.95))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    obs_rms = RunningMeanStd(shape=(6,), device="cpu")
    ret_rms = RunningMeanStd(shape=(), device="cpu")
    obs_rms.update(torch.randn(16, 6))
    ret_rms.update(torch.randn(16))

    # Populate a realistic mixed gradient state: some params have .grad, some None.
    (actor.logstd.sum() + actor.mu_net[0].weight.sum()).backward()
    critic.critic[0].weight.sum().backward()
    actor_opt.step()
    critic_opt.step()

    steps_num, num_envs = 4, 2
    buffers = {
        "buf_obs": torch.randn(steps_num, num_envs, 6),
        "buf_actions": torch.randn(steps_num, num_envs, 3),
        "buf_log_probs": torch.randn(steps_num, num_envs),
    }
    actor.train()
    critic.train()

    before = _snapshot(actor, critic, actor_opt, critic_opt, obs_rms, ret_rms, buffers)

    # Anti-vacuity: the snapshot really does carry a gradient, a None gradient and
    # non-empty optimizer state, and the comparison detects a single perturbation.
    assert any(value is not None for value in before["actor_grads"].values())
    assert any(value is None for value in before["actor_grads"].values())
    assert before["actor_opt"]["state"]
    assert not _same(before["actor_params"], {**before["actor_params"], "logstd": torch.zeros(3)})

    reports = [
        approx_kl_clip(actor, buffers["buf_obs"], buffers["buf_actions"],
                       buffers["buf_log_probs"], clip_range=0.2),
        logstd_summary(actor),
        near_bound_excitation(buffers["buf_actions"], tol=0.02),
        episode_event_summary(_events()),
    ]

    after = _snapshot(actor, critic, actor_opt, critic_opt, obs_rms, ret_rms, buffers)

    for key in before:
        assert _same(before[key], after[key]), key
    assert all(report["status"] == "ok" for report in reports)
    assert all(json.dumps(report, allow_nan=False) for report in reports)
    assert actor.logstd.grad is not None  # untouched, not zeroed


def test_repeated_calls_are_pure_and_do_not_advance_any_random_stream():
    torch.manual_seed(99)
    actor = ActorStochasticMLP(6, 3, CFG_NETWORK, device="cpu")
    obs = torch.randn(3, 2, 6)
    actions = torch.randn(3, 2, 3)
    old = torch.randn(3, 2)

    first = approx_kl_clip(actor, obs, actions, old, clip_range=0.2)
    torch_state = torch.get_rng_state().clone()
    second = approx_kl_clip(actor, obs, actions, old, clip_range=0.2)

    assert first == second
    assert torch.equal(torch_state, torch.get_rng_state())
    assert near_bound_excitation(actions, tol=0.02) == near_bound_excitation(actions, tol=0.02)


def test_the_new_helpers_are_not_re_exported_from_the_numpy_only_package_init():
    import msk_warp.analysis

    for name in ("approx_kl_clip", "logstd_summary", "near_bound_excitation",
                 "episode_event_summary"):
        assert name not in msk_warp.analysis.__all__
        assert not hasattr(msk_warp.analysis, name)


# --------------------------------------------------------------------------- #
# documented input-shape preconditions (F2)
# --------------------------------------------------------------------------- #

def test_the_module_states_its_preconditions_instead_of_an_absolute_no_raise_claim():
    """The tensor-taking helpers are not type guarded; the docstring must say so."""
    doc = ppo_diagnostics.__doc__
    lowered = doc.lower()
    assert "never raise on their data" not in lowered
    assert "never an exception" not in lowered
    assert "precondition" in lowered
    assert "zero-width" in lowered
    assert "episode_event_summary" in doc


class _BadLogstdActor:
    """get_logstd() must return a tensor or None; this returns neither."""

    def get_logstd(self):
        return [0.0, 0.0]


@pytest.mark.parametrize("call,expected", [
    (lambda: near_bound_excitation([0.1, 0.2], tol=0.02), AttributeError),
    (lambda: logstd_summary(_BadLogstdActor()), AttributeError),
    (lambda: approx_kl_clip(_GaussianActor(mu=0.0, logstd=0.0, action_dim=2),
                            torch.zeros(2, 0), torch.zeros(2, 0), torch.zeros(2),
                            clip_range=0.2), RuntimeError),
])
def test_a_violated_precondition_raises_rather_than_returning_a_misleading_status(call, expected):
    """Not reachable from the real PPO seam, and deliberately not papered over."""
    with pytest.raises(expected):
        call()


def test_a_missing_get_logstd_attribute_is_guarded_rather_than_a_precondition():
    """An object without the method at all is the ActorDeterministicMLP case."""
    report = logstd_summary(object())

    assert report["status"] == "logstd_unavailable"
    assert report["logstd_mean"] is None


def test_the_plain_data_helper_is_fully_type_guarded_by_contrast():
    for supplied in (7, None, "events", {"world": 0}):
        report = episode_event_summary(supplied)
        assert report["status"] == "malformed_event"
        assert report["mean_duration_s"] is None
        assert json.dumps(report, allow_nan=False)
