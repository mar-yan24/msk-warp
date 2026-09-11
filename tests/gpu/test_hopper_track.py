"""Gates for the reference-tracking muscle hopper.

Phase 4's tracking step rests on the reference being *exactly* reproducible by the environment it
was extracted from. If it is not, a tracking reward is unlearnable for reasons that have nothing to
do with the policy, and a failure would be misread as a statement about muscle control. So the
first test drives the environment with the reference's own controls and demands a perfect score,
and the second checks the cycle-closing error against the number the optimiser independently
reported.

The remaining tests pin the things a tracking reward must do to be worth using: reject standing,
carry the phase, and pass gradient to the actions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msk_warp import resolve_model_path
from msk_warp.envs import ENV_MAP

REFERENCE = 'assets/references/hopper_muscle_T16_gait.npz'

#: Residual the trajectory optimiser reported for this candidate (logs/phase4/muscle_T16.json,
#: world 177). The environment must reproduce it at the cycle wrap.
RECORDED_RESIDUAL = 0.3174


def _env(worlds=4, no_grad=True, **kwargs):
    return ENV_MAP['HopperMuscleTrack'](
        num_envs=worlds, device='cuda:0', no_grad=no_grad, episode_length=1000,
        stochastic_init=False, early_termination=False, substeps=4, njmax=128,
        reference_path=REFERENCE, **kwargs,
    )


def _reference_actions(device='cuda:0'):
    """Actions that map onto the reference controls: ``ctrl = 0.5 * (a + 1)``."""
    ctrl = np.load(resolve_model_path(REFERENCE))['ctrl']
    return 2.0 * torch.tensor(ctrl, dtype=torch.float32, device=device) - 1.0


def test_observation_is_the_state_plus_a_phase():
    env = _env()
    # qpos[1:] (5) + qvel (6) + act (6) + (sin, cos) = 19
    assert env.num_obs == 19
    obs = env.reset()
    assert obs.shape == (4, 19)


def test_the_reference_is_exactly_trackable_for_one_cycle():
    """Driving the environment with the reference's own controls must score perfectly.

    This is the precondition for the whole tracking step: the reference came out of this same
    model, timestep and substep count, so reproducing it is a consistency check on the export, the
    phase alignment and the reward, all at once.
    """
    env = _env()
    actions = _reference_actions()
    env.reset()
    errors = []
    for phase in range(env.cycle - 1):
        obs = env.step(actions[phase].expand(env.num_envs, -1))[0]
        errors.append(float(env.tracking_error(obs).max()))
    assert max(errors) < 1e-4, f"reference not reproduced: max error {max(errors):.2e}"


def test_the_cycle_closing_error_matches_the_recorded_residual():
    """At the wrap the comparison is against phase 0, which is the optimiser's residual."""
    env = _env()
    actions = _reference_actions()
    env.reset()
    for phase in range(env.cycle):
        obs = env.step(actions[phase].expand(env.num_envs, -1))[0]
    assert float(env.tracking_error(obs).max()) == pytest.approx(RECORDED_RESIDUAL, abs=2e-3)


def test_the_reward_is_one_on_the_reference_and_collapses_off_it():
    env = _env()
    actions = _reference_actions()
    env.reset()
    on_reference = float(env.step(actions[0].expand(env.num_envs, -1))[1].max())
    assert on_reference == pytest.approx(1.0, abs=1e-4)

    # Zero activation: G3.1 measured that this collapses, so it must score near nothing.
    env.reset()
    for _ in range(env.cycle * 2):
        obs, rew, *_ = env.step(env.passive_action())
    assert float(rew.max()) < 0.2, f"a collapsing hopper scored {float(rew.max()):.3f}"


def test_standing_still_scores_far_below_the_reference():
    """The standing basin took 2 of 3 SHAC seeds in Phase 3. A tracking reward must reject it."""
    env = _env()
    env.reset()
    # Uniform 50% activation is the zero-action corner and the posture G3.1 found statically stable.
    standing = torch.zeros(env.num_envs, env.num_actions, device='cuda:0')
    for _ in range(env.cycle * 3):
        obs, rew, *_ = env.step(standing)
    assert float(rew.max()) < 0.2, f"standing scored {float(rew.max()):.3f}"


def test_the_phase_advances_and_wraps():
    env = _env()
    env.reset()
    seen = []
    for _ in range(env.cycle + 2):
        env.step(env.passive_action())
        seen.append(int(env.phase_buf[0]))
    assert seen[:env.cycle] == list(range(1, env.cycle)) + [0]
    assert seen[env.cycle] == 1, "phase must wrap back into the cycle, not run past it"


def test_the_tracking_reward_passes_gradient_to_the_actions():
    env = _env(no_grad=False)
    env.reset()
    qpos, qvel, act = env.state_tensors()
    actions = torch.zeros(env.num_envs, env.num_actions, device='cuda:0', requires_grad=True)
    _, rew, _, _, qpos, qvel, act = env.step(actions, qpos, qvel, act)
    rew.sum().backward()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert actions.grad.abs().max() > 0, "tracking reward gave the actions no gradient"


def test_reset_starts_on_the_reference_orbit():
    env = _env(worlds=32)
    env.stochastic_init = True
    env.reset()
    qpos, qvel, _ = env.state_tensors()
    target = env.ref_qpos[env.phase_buf]
    # x is zeroed at reset; every other coordinate matches the reference within the reset noise.
    assert torch.allclose(qpos[:, 1:], target[:, 1:], atol=env.reset_state_noise + 1e-5)
    assert torch.equal(qpos[:, 0], torch.zeros_like(qpos[:, 0]))
    assert len(set(env.phase_buf.tolist())) > 1, "reference-state init must spread over the cycle"
