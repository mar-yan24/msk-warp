"""Unit tests for :mod:`msk_warp.analysis.gaitsearch`.

The load-bearing test here is ``test_sparse_jacobian_matches_dense_finite_differences``. The
multiple-shooting Jacobian is assembled by hand from local segment rolls plus an analytic continuity
block, and a mistake in that assembly would produce a solver that silently descends the wrong
surface. It is checked against a brute-force dense difference of the same residual, and agrees to
4e-13.
"""

import json

import numpy as np
import mujoco
import pytest

from msk_warp import resolve_model_path
from msk_warp.analysis import gaitsearch as gs
from msk_warp.analysis import orbit as orb

REFERENCE = "assets/references/hopper_muscle_T16_gait.npz"
RULER = "docs/research/phase4-capability/gait_motor_seed2.json"


@pytest.fixture(scope="module")
def scales():
    try:
        with open(RULER) as fh:
            return np.array(json.load(fh)["per_world"][0]["scales"], dtype=np.float64)
    except FileNotFoundError:  # the ruler lives under the gitignored docs/ tree
        return np.load(resolve_model_path(REFERENCE))["shape_scales"].astype(np.float64)


@pytest.fixture(scope="module")
def candidate(scales):
    d = np.load(resolve_model_path(REFERENCE))
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path("assets/hopper_muscle.xml"))
    ctrl = d["ctrl"].astype(np.float64)
    x0 = orb.shape_state(d["qpos"][0], d["qvel"][0])
    return orb.ReturnMap(mjm, ctrl), x0, ctrl, scales


# ------------------------------------------------------------------ the segment primitive

def test_segments_chain_to_a_full_cycle(candidate):
    """Four four-step segments must reproduce one sixteen-step roll, state and advance."""
    rmap, x0, _, _ = candidate
    full = rmap.roll(x0)
    traj = rmap.activation_trajectory()
    x, act, advance = x0, traj[0], 0.0
    for s in range(4):
        x, act, adv, _, terminated = rmap.segment(x, act, 4 * s, 4)
        assert not terminated
        advance += adv
    assert np.abs(x - full.state).max() <= 1e-12
    assert advance == pytest.approx(full.advance, abs=1e-9)
    assert np.abs(act - traj[0]).max() <= 1e-12, "activation must close over the cycle"


def test_activation_trajectory_starts_at_the_limit_cycle(candidate):
    rmap, _, _, _ = candidate
    traj = rmap.activation_trajectory()
    assert traj.shape == (rmap.cycle, rmap.mjm.na)
    np.testing.assert_allclose(traj[0], rmap.activation, atol=1e-12)


def test_set_control_recomputes_the_activation_limit_cycle(candidate):
    rmap, _, ctrl, _ = candidate
    before = rmap.activation.copy()
    rmap.set_control(np.full_like(ctrl, 0.5))
    assert np.abs(rmap.activation - before).max() > 0.1
    rmap.set_control(ctrl, warm_start=False)
    np.testing.assert_allclose(rmap.activation, before, atol=1e-10)


def test_set_control_rejects_a_reshaped_control(candidate):
    rmap, _, ctrl, _ = candidate
    with pytest.raises(ValueError, match="keep its shape"):
        rmap.set_control(ctrl[:, :3])


# ------------------------------------------------------------------ the joint problem

def test_seed_puts_all_the_error_in_the_wraparound(candidate):
    """Seeding by simulation makes every interior continuity residual exactly zero by construction.

    So the whole closing error sits in one block, which is what makes the problem hard: the gap is
    1.36 RMS scale-units, not a small correction.
    """
    rmap, x0, ctrl, sc = candidate
    problem = gs.MultipleShootingProblem(rmap, sc, segments=4)
    theta = problem.seed(x0, ctrl)
    residual, _ = problem.evaluate(theta, 0.0)
    blocks = [np.linalg.norm(residual[s * problem.n_node:(s + 1) * problem.n_node])
              for s in range(problem.segments)]
    assert max(blocks[:-1]) <= 1e-12
    assert blocks[-1] > 1.0


def test_sparse_jacobian_matches_dense_finite_differences(candidate):
    """The assembly check. Hand-built locality plus an analytic block, against brute force."""
    rmap, x0, ctrl, sc = candidate
    problem = gs.MultipleShootingProblem(rmap, sc, segments=4)
    theta = problem.seed(x0, ctrl)
    sparse = problem.jacobian(theta, 0.0)

    steps = np.concatenate([
        np.tile(np.concatenate([gs.STATE_EPS_REL * sc, np.full(problem.na, gs.STATE_EPS_REL)]),
                problem.segments),
        np.full(problem.n_ctrl, gs.CTRL_EPS),
    ])
    dense = np.zeros_like(sparse)
    for j in range(problem.dim):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += steps[j]
        minus[j] -= steps[j]
        fp, _ = problem.evaluate(plus, 0.0)
        fm, _ = problem.evaluate(minus, 0.0)
        dense[:, j] = (fp - fm) / (2.0 * steps[j])

    assert np.linalg.norm(sparse - dense) / np.linalg.norm(dense) <= 1e-9
    assert (sparse == 0).sum() / sparse.size > 0.7, "the Jacobian should be mostly zero"


def test_more_segments_cost_less_not_more(candidate):
    """Locality means the Jacobian gets cheaper as the cycle is cut finer, not dearer."""
    rmap, x0, ctrl, sc = candidate
    sizes = {}
    for segments in (2, 16):
        problem = gs.MultipleShootingProblem(rmap, sc, segments=segments)
        theta = problem.seed(x0, ctrl)
        jac = problem.jacobian(theta, 0.0)
        sizes[segments] = (jac == 0).sum() / jac.size
    assert sizes[16] > sizes[2] > 0.5


def test_control_projection_respects_the_model_range(candidate):
    """``ctrlrange`` is enforced by the engine, so unlike ``jnt_range`` it must be projected."""
    rmap, x0, ctrl, sc = candidate
    problem = gs.MultipleShootingProblem(rmap, sc, segments=4)
    assert (problem.ctrl_lo, problem.ctrl_hi) == (0.0, 1.0)
    theta = problem.seed(x0, ctrl)
    theta[problem.segments * problem.n_node:] = 5.0
    projected = problem.project(theta)
    _, got = problem.unpack(projected)
    assert got.max() <= 1.0 - problem.ctrl_margin + 1e-12
    assert problem.saturated(projected) == problem.n_ctrl


def test_homotopy_offset_is_exact_at_alpha_zero(candidate):
    """At ``alpha = 0`` the relaxed residual vanishes; ``periodicity`` still reports the truth."""
    rmap, x0, ctrl, sc = candidate
    problem = gs.MultipleShootingProblem(rmap, sc, segments=4)
    theta = problem.seed(x0, ctrl)
    problem.set_homotopy(theta, 0.0)
    relaxed, info = problem.evaluate(theta, 0.0)
    assert np.abs(relaxed).max() <= 1e-12
    assert info["periodicity"] > 0.1, "the true closing error must not be relaxed away in reporting"
    assert info["relaxed_periodicity"] <= 1e-12


# ------------------------------------------------------------------ the manifold diagnostic

def test_velocity_is_unreachable_on_a_standing_orbit():
    """A standing orbit cannot be accelerated at all while staying periodic, and that is the floor.

    The measured ladder across four orbits spanning both actuators, in a physical metric::

        motor trained gait T27   flight 22/27   dv_per_unit 7.80e-01
        motor backwards hop T16  flight  6/16   dv_per_unit 6.11e-01
        muscle bob T32           flight  1/32   dv_per_unit 2.52e-02
        muscle standing T16      flight  0/16   dv_per_unit 7.97e-10

    Velocity reach tracks flight fraction, not actuator type: in flight the body is ballistic and
    horizontal velocity is a free constant of the motion, while in stance the foot is anchored. This
    pins the zero-flight end, which is the one with an analytic expectation.
    """
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path("assets/hopper_muscle.xml"))
    sc = np.load(resolve_model_path(REFERENCE))["shape_scales"].astype(np.float64)
    ctrl = np.full((16, mjm.nu), 0.5)
    rmap = orb.ReturnMap(mjm, ctrl)
    x = np.zeros(11)
    for _ in range(200):
        x = rmap.roll(x).state
    problem = gs.GaitProblem(rmap, sc, velocity_weight=1.0)
    reach = gs.velocity_reach(problem, problem.pack(x, ctrl))
    assert reach["dv_per_unit"] <= 1e-6
    assert reach["retained_fraction"] <= 1e-5
