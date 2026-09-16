"""CPU tests for the revised v2 checkpoint-selection key.

No simulator, no GPU, no training. Every expected ranking tuple below is
hand-computed in the test body from the stated episode counts; nothing is taken
from the function under test. The v1 key and its recorded result are untouched:
where a test contrasts the two protocols it records that they *order* a pair
differently, which is a property of the revised protocol, not a v1 bug report.
"""

from __future__ import annotations

import json
import math

import pytest

import msk_warp.analysis
from msk_warp.analysis.myoleg26_baseline import behavior_rank
from msk_warp.analysis.myoleg26_selection_v2 import (
    InvalidCandidateSetError,
    InvalidEvaluationError,
    behavior_rank_v2,
    mean_in_band_seconds,
    select_best_checkpoint,
)

# Official myoleg26-walk-v1 control period: 8 ms. Not a tunable here.
CONTROL_DT = 0.008


def _evaluation(*, survival, duration_s, rmse, lengths, band_steps,
                control_dt=CONTROL_DT, complete=True):
    """An evaluation shaped exactly like aggregate_episodes + evaluate_policy output."""
    assert len(lengths) == len(band_steps)
    episodes = [
        {
            "episode_seed": 20000 + index,
            "world": index,
            "length": length,
            "survived": bool(survival >= 1.0),
            "speed_band_steps": steps,
            "end_reason": "horizon",
            "failure_flags": {},
            "velocity_2d_squared_error_sum": rmse * rmse * length,
            "return": 0.0,
            "velocity_2d_rmse_mps": rmse,
            "forward_speed_band_fraction": steps / length,
        }
        for index, (length, steps) in enumerate(zip(lengths, band_steps))
    ]
    return {
        "summary": {
            "survival_fraction": survival,
            "mean_first_episode_duration_s": duration_s,
            "mean_episode_velocity_2d_rmse_mps": rmse,
            "forward_speed_band_fraction": sum(band_steps) / sum(lengths),
            "mean_forward_velocity_mps": 0.0,
            "mean_displacement_m": 0.0,
        },
        "accounting": {"scored_first_episode_transitions": sum(lengths)},
        "episodes": episodes,
        "horizon_control_steps": 500,
        "control_dt": control_dt,
        "complete": complete,
        "policy_mode": "deterministic_mean",
    }


def _candidate(epoch, evaluation):
    return {"epoch": epoch, "evaluation": evaluation, "path": f"epoch_{epoch}.pt"}


# --------------------------------------------------------------------------- #
# the approved key itself
# --------------------------------------------------------------------------- #

def test_rank_is_exactly_the_four_approved_components_in_order():
    """Hand-computed: mean(400, 50, 0) = 150 controls; 150 * 0.008 = 1.2 s in band."""
    evaluation = _evaluation(survival=0.25, duration_s=2.2, rmse=0.75,
                             lengths=[500, 250, 100], band_steps=[400, 50, 0])

    rank = behavior_rank_v2(evaluation)

    assert isinstance(rank, tuple) and len(rank) == 4
    assert all(type(component) is float for component in rank)
    assert rank[0] == pytest.approx(0.25, abs=1e-12)
    assert rank[1] == pytest.approx(2.2, abs=1e-12)
    assert rank[2] == pytest.approx(1.2, abs=1e-12)
    assert rank[3] == pytest.approx(-0.75, abs=1e-12)
    # No union type, no sentinel, no eligibility record smuggled alongside.
    assert json.dumps(list(rank))


def test_in_band_seconds_uses_episode_counts_not_the_fraction_shortcut():
    """Unequal lengths separate the two formulas.

    lengths 500 and 100, in-band counts 250 and 100.
      correct  : mean(250, 100) = 175 controls -> 175 * 0.008 = 1.4 s
      shortcut : mean(0.5, 1.0) = 0.75; mean duration = 300 * 0.008 = 2.4 s
                 -> 0.75 * 2.4 = 1.8 s
    """
    evaluation = _evaluation(survival=0.5, duration_s=300 * CONTROL_DT, rmse=1.0,
                             lengths=[500, 100], band_steps=[250, 100])

    value = mean_in_band_seconds(evaluation)

    assert value == pytest.approx(1.4, abs=1e-12)
    assert value != pytest.approx(1.8, abs=1e-9)
    assert behavior_rank_v2(evaluation)[2] == pytest.approx(1.4, abs=1e-12)


def test_duration_precedes_rmse_when_neither_policy_survives():
    """A 2.4 s fall with RMSE 1.4 beats a 0.6 s fall with RMSE 0.05."""
    early_fall = _evaluation(survival=0.0, duration_s=0.6, rmse=0.05,
                             lengths=[75], band_steps=[0])
    long_fall = _evaluation(survival=0.0, duration_s=2.4, rmse=1.4,
                            lengths=[300], band_steps=[0])

    assert behavior_rank_v2(early_fall) == pytest.approx((0.0, 0.6, 0.0, -0.05))
    assert behavior_rank_v2(long_fall) == pytest.approx((0.0, 2.4, 0.0, -1.4))
    assert behavior_rank_v2(long_fall) > behavior_rank_v2(early_fall)

    winner = select_best_checkpoint([_candidate(0, early_fall), _candidate(96, long_fall)])
    assert winner["epoch"] == 96

    # Contrast only: the two protocols order this pair differently. The v1 key is
    # byte-unchanged and its recorded selection result is not revisited here.
    assert behavior_rank(early_fall) > behavior_rank(long_fall)


def test_in_band_seconds_breaks_the_tie_in_favour_of_the_mover():
    """Same 4 s survival and identical duration: standing has speed_band_steps == 0."""
    standing = _evaluation(survival=1.0, duration_s=4.0, rmse=1.0,
                           lengths=[500], band_steps=[0])
    moving = _evaluation(survival=1.0, duration_s=4.0, rmse=1.0,
                         lengths=[500], band_steps=[500])

    assert behavior_rank_v2(standing) == pytest.approx((1.0, 4.0, 0.0, -1.0))
    assert behavior_rank_v2(moving) == pytest.approx((1.0, 4.0, 4.0, -1.0))
    assert behavior_rank_v2(moving) > behavior_rank_v2(standing)
    assert select_best_checkpoint([_candidate(4, standing), _candidate(8, moving)])["epoch"] == 8


# --------------------------------------------------------------------------- #
# the tie rule is explicit, not incidental
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("descending", [False, True])
def test_exact_four_way_tie_selects_the_earliest_epoch_in_either_arrival_order(descending):
    tied = dict(survival=0.5, duration_s=2.0, rmse=0.9, lengths=[250], band_steps=[100])
    candidates = [_candidate(32, _evaluation(**tied)), _candidate(96, _evaluation(**tied))]
    if descending:
        candidates.reverse()

    winner = select_best_checkpoint(candidates)

    assert behavior_rank_v2(candidates[0]["evaluation"]) == behavior_rank_v2(
        candidates[1]["evaluation"])
    assert winner["epoch"] == 32
    assert winner["path"] == "epoch_32.pt"


def test_a_later_epoch_still_wins_on_a_strictly_better_key():
    """The tie rule must not degrade into an unconditional earliest-epoch bias."""
    worse = _evaluation(survival=0.0, duration_s=1.0, rmse=1.0, lengths=[125], band_steps=[0])
    better = _evaluation(survival=0.0, duration_s=3.0, rmse=1.0, lengths=[375], band_steps=[10])

    for order in ([_candidate(0, worse), _candidate(96, better)],
                  [_candidate(96, better), _candidate(0, worse)]):
        assert select_best_checkpoint(order)["epoch"] == 96


def test_duplicate_epochs_are_refused_rather_than_resolved_by_arrival_order():
    tied = dict(survival=0.5, duration_s=2.0, rmse=0.9, lengths=[250], band_steps=[100])
    with pytest.raises(InvalidCandidateSetError) as excinfo:
        select_best_checkpoint([_candidate(32, _evaluation(**tied)),
                                _candidate(32, _evaluation(**tied))])
    assert "duplicate" in excinfo.value.reason


def test_empty_candidate_list_is_refused_and_never_returns_a_default_winner():
    with pytest.raises(InvalidCandidateSetError) as excinfo:
        select_best_checkpoint([])
    assert "no candidate" in excinfo.value.reason


@pytest.mark.parametrize("epoch", [-1, 1.0, "12", True, None])
def test_candidate_epoch_must_be_a_non_negative_integer(epoch):
    tied = dict(survival=0.5, duration_s=2.0, rmse=0.9, lengths=[250], band_steps=[100])
    with pytest.raises(InvalidCandidateSetError) as excinfo:
        select_best_checkpoint([_candidate(epoch, _evaluation(**tied))])
    assert "epoch" in excinfo.value.reason


def test_an_invalid_candidate_aborts_selection_instead_of_scoring_zero():
    good = _evaluation(survival=1.0, duration_s=4.0, rmse=0.2, lengths=[500], band_steps=[500])
    broken = _evaluation(survival=0.0, duration_s=0.5, rmse=1.0, lengths=[62], band_steps=[0],
                         complete=False)
    with pytest.raises(InvalidEvaluationError):
        select_best_checkpoint([_candidate(0, good), _candidate(8, broken)])


# --------------------------------------------------------------------------- #
# rejection, not coercion
# --------------------------------------------------------------------------- #

def _valid():
    return _evaluation(survival=0.5, duration_s=2.0, rmse=0.9,
                       lengths=[500, 250], band_steps=[300, 25])


def _drop_evaluation_key(key):
    def mutate(evaluation):
        del evaluation[key]
    return mutate


def _drop_summary_key(key):
    def mutate(evaluation):
        del evaluation["summary"][key]
    return mutate


def _set_summary(key, value):
    def mutate(evaluation):
        evaluation["summary"][key] = value
    return mutate


def _set_episode(index, key, value):
    def mutate(evaluation):
        evaluation["episodes"][index][key] = value
    return mutate


def _drop_episode_key(index, key):
    def mutate(evaluation):
        del evaluation["episodes"][index][key]
    return mutate


REJECTIONS = {
    "not_a_mapping": (lambda evaluation: None, ["summary", "episodes"]),
    "missing_summary": (_drop_evaluation_key("summary"), None),
    "missing_episodes": (_drop_evaluation_key("episodes"), None),
    "missing_control_dt": (_drop_evaluation_key("control_dt"), None),
    "missing_complete": (_drop_evaluation_key("complete"), None),
    "incomplete": (lambda evaluation: evaluation.update(complete=False), None),
    "complete_not_boolean": (lambda evaluation: evaluation.update(complete="yes"), None),
    "empty_episodes": (lambda evaluation: evaluation.update(episodes=[]), None),
    "episodes_not_a_sequence": (lambda evaluation: evaluation.update(episodes={}), None),
    "missing_survival": (_drop_summary_key("survival_fraction"), None),
    "missing_duration": (_drop_summary_key("mean_first_episode_duration_s"), None),
    "missing_rmse": (_drop_summary_key("mean_episode_velocity_2d_rmse_mps"), None),
    "nan_survival": (_set_summary("survival_fraction", float("nan")), None),
    "inf_duration": (_set_summary("mean_first_episode_duration_s", float("inf")), None),
    "nonnumeric_rmse": (_set_summary("mean_episode_velocity_2d_rmse_mps", "0.5"), None),
    "boolean_survival": (_set_summary("survival_fraction", True), None),
    "none_duration": (_set_summary("mean_first_episode_duration_s", None), None),
    "control_dt_zero": (lambda evaluation: evaluation.update(control_dt=0.0), None),
    "control_dt_negative": (lambda evaluation: evaluation.update(control_dt=-0.008), None),
    "control_dt_nan": (lambda evaluation: evaluation.update(control_dt=float("nan")), None),
    "control_dt_nonnumeric": (lambda evaluation: evaluation.update(control_dt="8ms"), None),
    "episode_missing_length": (_drop_episode_key(1, "length"), None),
    "episode_missing_band_steps": (_drop_episode_key(1, "speed_band_steps"), None),
    "episode_not_a_mapping": (lambda evaluation: evaluation["episodes"].__setitem__(1, [250, 25]),
                              None),
    "zero_length": (_set_episode(1, "length", 0), None),
    "negative_length": (_set_episode(1, "length", -5), None),
    "float_length": (_set_episode(1, "length", 250.0), None),
    "negative_band_steps": (_set_episode(1, "speed_band_steps", -1), None),
    "float_band_steps": (_set_episode(1, "speed_band_steps", 25.0), None),
    "nan_band_steps": (_set_episode(1, "speed_band_steps", float("nan")), None),
    "band_steps_exceed_length": (_set_episode(1, "speed_band_steps", 251), None),
}


@pytest.mark.parametrize("case", sorted(REJECTIONS))
def test_every_malformed_evaluation_is_refused_with_a_reason(case):
    mutate, replacement = REJECTIONS[case]
    evaluation = _valid()
    if replacement is not None:
        evaluation = replacement
    else:
        mutate(evaluation)

    with pytest.raises(InvalidEvaluationError) as excinfo:
        behavior_rank_v2(evaluation)

    error = excinfo.value
    assert isinstance(error, ValueError)
    assert isinstance(error.reason, str) and error.reason
    assert error.reason in str(error)
    # Same refusal from the selection entry point, and from the derived metric.
    with pytest.raises(InvalidEvaluationError):
        select_best_checkpoint([_candidate(0, evaluation)])
    with pytest.raises(InvalidEvaluationError):
        mean_in_band_seconds(evaluation)


def test_rejection_reasons_distinguish_the_failure_modes():
    reasons = {}
    for case, (mutate, replacement) in REJECTIONS.items():
        evaluation = replacement if replacement is not None else _valid()
        if replacement is None:
            mutate(evaluation)
        with pytest.raises(InvalidEvaluationError) as excinfo:
            behavior_rank_v2(evaluation)
        reasons[case] = excinfo.value.reason

    assert len(set(reasons.values())) == len(reasons), reasons


def test_a_rejected_evaluation_never_produces_a_comparable_score():
    """No NaN placeholder, no zero default, no sentinel that could win a max()."""
    broken = _valid()
    broken["summary"]["survival_fraction"] = float("nan")
    try:
        behavior_rank_v2(broken)
    except InvalidEvaluationError as error:
        assert not hasattr(error, "rank")
        assert not hasattr(error, "score")
    else:  # pragma: no cover - the call above must raise
        pytest.fail("a nonfinite metric was ranked instead of refused")


# --------------------------------------------------------------------------- #
# packaging invariants
# --------------------------------------------------------------------------- #

def test_the_new_helpers_are_not_re_exported_from_the_numpy_only_package_init():
    for name in ("behavior_rank_v2", "select_best_checkpoint", "mean_in_band_seconds",
                 "InvalidEvaluationError"):
        assert name not in msk_warp.analysis.__all__
        assert not hasattr(msk_warp.analysis, name)


def test_v1_behavior_rank_still_returns_its_own_three_component_key():
    """Guards against any accidental edit of the frozen v1 ranking function."""
    evaluation = _valid()
    v1 = behavior_rank(evaluation)
    assert len(v1) == 3
    assert v1[0] == pytest.approx(0.5)
    assert v1[1] == pytest.approx(-0.9)
    assert v1[2] == pytest.approx(325 / 750)
    assert not math.isnan(v1[2])


# --------------------------------------------------------------------------- #
# set-level (caller) refusal is a different diagnosis from data-level refusal
# --------------------------------------------------------------------------- #

def test_set_level_and_data_level_refusals_are_not_catchable_as_each_other():
    """A runner bug must never be recordable as an invalid checkpoint evaluation."""
    assert InvalidCandidateSetError is not InvalidEvaluationError
    assert issubclass(InvalidCandidateSetError, ValueError)
    assert issubclass(InvalidEvaluationError, ValueError)
    assert not issubclass(InvalidCandidateSetError, InvalidEvaluationError)
    assert not issubclass(InvalidEvaluationError, InvalidCandidateSetError)


@pytest.mark.parametrize("candidates,fragment", [
    ("duplicate", "duplicate"),
    ("empty", "no candidate"),
    ("not_a_mapping", "not a mapping"),
    ("missing_epoch", "'epoch'"),
    ("missing_evaluation", "'evaluation'"),
    ("bad_epoch", "'epoch'"),
])
def test_caller_side_candidate_faults_raise_the_set_level_class(candidates, fragment):
    tied = dict(survival=0.5, duration_s=2.0, rmse=0.9, lengths=[250], band_steps=[100])
    cases = {
        "duplicate": [_candidate(32, _evaluation(**tied)), _candidate(32, _evaluation(**tied))],
        "empty": [],
        "not_a_mapping": [[32, _evaluation(**tied)]],
        "missing_epoch": [{"evaluation": _evaluation(**tied)}],
        "missing_evaluation": [{"epoch": 32}],
        "bad_epoch": [_candidate(-1, _evaluation(**tied))],
    }
    try:
        select_best_checkpoint(cases[candidates])
    except InvalidEvaluationError as error:  # checked first: must NOT match
        pytest.fail(f"a candidate-set fault was raised as a data-level error: {error}")
    except InvalidCandidateSetError as error:
        assert isinstance(error.reason, str) and fragment in error.reason
    else:
        pytest.fail("the malformed candidate set was not refused")


def test_an_unrankable_evaluation_still_raises_the_data_level_class():
    broken = _valid()
    broken["summary"]["survival_fraction"] = float("nan")
    try:
        select_best_checkpoint([_candidate(0, broken)])
    except InvalidCandidateSetError as error:  # checked first: must NOT match
        pytest.fail(f"a data-level rejection was raised as a candidate-set fault: {error}")
    except InvalidEvaluationError as error:
        assert "not finite" in error.reason
    else:
        pytest.fail("the nonfinite metric was not refused")


def test_both_refusal_classes_stay_catchable_by_an_existing_value_error_handler():
    tied = dict(survival=0.5, duration_s=2.0, rmse=0.9, lengths=[250], band_steps=[100])
    for call in (lambda: select_best_checkpoint([]),
                 lambda: behavior_rank_v2({"summary": {}})):
        with pytest.raises(ValueError):
            call()
