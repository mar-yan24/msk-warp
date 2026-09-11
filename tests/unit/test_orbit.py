"""Unit tests for :mod:`msk_warp.analysis.orbit`.

Every threshold here is a value measured on 2026-09-11 with at least an order of magnitude of
headroom, so a regression shows up as a failure rather than as a surprise in a results table. The
measurements themselves are recorded in ``docs/VALIDITY.md`` (BE-08, CL-01, CL-02, IN-01, IN-02).

No GPU, no Warp, no torch in the module under test -- only ``test_shape_state_matches_gait``
imports torch, to cross-check the two implementations of the one ruler this project measures with.
"""

import numpy as np
import mujoco
import pytest

from msk_warp import resolve_model_path
from msk_warp.analysis import orbit as orb

REFERENCE = "assets/references/hopper_muscle_T16_gait.npz"

# Measured 2026-09-11 from the committed reference asset. See docs/VALIDITY.md CL-02.
ACT_STAR = np.array([0.941422, 0.168312, 0.063191, 0.985747, 0.037268, 0.977356])

FREE_JOINT_XML = """
<mujoco>
  <worldbody>
    <body name="b"><freejoint/><geom type="sphere" size="0.1"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def reference():
    d = np.load(resolve_model_path(REFERENCE))
    return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def muscle_model():
    return mujoco.MjModel.from_xml_path(resolve_model_path("assets/hopper_muscle.xml"))


@pytest.fixture(scope="module")
def motor_model():
    return mujoco.MjModel.from_xml_path(resolve_model_path("assets/hopper_motor.xml"))


# --------------------------------------------------------------------------------------- state

def test_shape_state_drops_the_translating_coordinate():
    x = orb.shape_state(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert x.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_shape_state_appends_activation_when_present():
    x = orb.shape_state(np.array([1.0, 2.0]), np.array([3.0]), np.array([0.5, 0.6]))
    assert x.tolist() == [2.0, 3.0, 0.5, 0.6]


def test_shape_state_ignores_empty_activation():
    x = orb.shape_state(np.array([1.0, 2.0]), np.array([3.0]), np.zeros(0))
    assert x.tolist() == [2.0, 3.0]


def test_shape_state_matches_gait_shape_vector():
    """The numpy and torch rulers must agree, or two phases stop being comparable.

    This is the only test in the file that imports torch, and it exists because
    ``msk_warp.analysis`` deliberately reimplements ``gait.shape_vector`` rather than importing it
    (``msk_warp/utils/__init__.py`` pulls in torch eagerly).
    """
    torch = pytest.importorskip("torch")
    from msk_warp.utils.gait import shape_vector

    rng = np.random.default_rng(0)
    for na in (0, 6):
        qpos, qvel, act = rng.normal(size=6), rng.normal(size=6), rng.normal(size=na)
        got = orb.shape_state(qpos, qvel, act)
        want = shape_vector(
            torch.tensor(qpos), torch.tensor(qvel), torch.tensor(act)
        ).numpy()
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_state_dim_matches_each_hopper_asset(muscle_model, motor_model):
    assert orb.state_dim(muscle_model) == 11
    assert orb.state_dim(muscle_model, include_activation=True) == 17
    assert orb.state_dim(motor_model) == 11
    assert orb.state_dim(motor_model, include_activation=True) == 11  # na == 0


def test_assert_state_layout_accepts_both_hoppers(muscle_model, motor_model):
    orb.assert_state_layout(muscle_model)
    orb.assert_state_layout(motor_model)


def test_assert_state_layout_rejects_a_free_joint_model():
    """The MyoLeg26 tripwire: ``nq != nv`` must fail loudly rather than mis-index."""
    mjm = mujoco.MjModel.from_xml_string(FREE_JOINT_XML)
    assert mjm.nq == 7 and mjm.nv == 6
    with pytest.raises(ValueError, match="tangent"):
        orb.assert_state_layout(mjm)


# --------------------------------------------------------------- the activation limit cycle

def test_activation_limit_cycle_is_independent_of_its_start(muscle_model, reference):
    """``act*`` is determined by the control alone -- the claim CL-02 rests on.

    Measured difference between starts: exactly 0.0. Gated at 1e-12 for headroom.
    """
    ctrl = reference["ctrl"].astype(np.float64)
    starts = [None, np.zeros(6), np.ones(6), reference["act"][0].astype(np.float64),
              np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])]
    solutions = [orb.activation_limit_cycle(muscle_model, ctrl, start=s)[0] for s in starts]
    for other in solutions[1:]:
        assert np.abs(other - solutions[0]).max() <= 1e-12


def test_activation_limit_cycle_converges_within_four_cycles(muscle_model, reference):
    act_star, cycles, delta = orb.activation_limit_cycle(
        muscle_model, reference["ctrl"].astype(np.float64)
    )
    assert cycles <= 4, (cycles, delta)
    assert delta <= 1e-14
    np.testing.assert_allclose(act_star, ACT_STAR, atol=1e-5)


def test_activation_limit_cycle_matches_a_full_rollout(muscle_model, reference):
    """Cross-check the ODE-only route against reading ``act`` off a full simulation.

    Two independent routes to the same object. The ODE route is used in production because it is
    immune to a diverging rollout; this test is what licenses that substitution.
    """
    ctrl = reference["ctrl"].astype(np.float64)
    act_star, _, _ = orb.activation_limit_cycle(muscle_model, ctrl)

    rmap = orb.ReturnMap(muscle_model, ctrl, activation=reference["act"][0].astype(np.float64))
    x0 = orb.shape_state(reference["qpos"][0], reference["qvel"][0])

    # Three cycles, because this trajectory terminates one step into cycle 4. Asking for more
    # returns activation read at whatever phase the break happened on, which is not a cycle
    # boundary and therefore not act* -- the same class of mistake evaluate_hopper.py's docstring
    # records (reading a final state off a rollout that already reset). Assert survival so the
    # comparison cannot silently be made against a truncated rollout.
    out = rmap.roll(x0, cycles=3)
    assert not out.terminated, "the cross-check must run on a surviving rollout"
    assert np.abs(out.activation - act_star).max() <= 1e-12  # measured 4.2e-17


def test_activation_limit_cycle_explains_the_reference_closing_error(muscle_model, reference):
    """The "activation does not close" finding **is** ``|act_0 - act*|``, to 6 decimals.

    This is the measurement that reinterprets Phase 4's amendment-1 refutation: the closing error
    was a mis-specified initial condition, not a trajectory that cannot close.
    """
    ctrl = reference["ctrl"].astype(np.float64)
    act_star, _, _ = orb.activation_limit_cycle(muscle_model, ctrl)

    # act* is where the control drives activation, so it is also the recorded closing activation.
    np.testing.assert_allclose(act_star, reference["closing_act"].astype(np.float64), atol=2e-6)

    recorded_error = np.abs(
        reference["closing_act"].astype(np.float64) - reference["act"][0].astype(np.float64)
    )
    np.testing.assert_allclose(
        recorded_error, np.abs(act_star - reference["act"][0]), atol=2e-6
    )
    assert recorded_error.max() > 0.88  # the knee antagonist, the headline number


def test_activation_limit_cycle_rejects_a_model_without_muscles(motor_model, reference):
    with pytest.raises(ValueError, match="no activation state"):
        orb.activation_limit_cycle(motor_model, np.zeros((16, motor_model.nu)))


# ----------------------------------------------------------------------------- the return map

def test_return_map_reproduces_the_recorded_warp_cycle(muscle_model, reference):
    """Engine parity, offline and GPU-free (``docs/VALIDITY.md`` BE-08).

    Measured 2026-09-11: qpos 8.4e-07, qvel 3.8e-06, act 4.4e-08. Gated 12x to 26x looser. If this
    fails, the whole float64-CPU route stops describing the Warp candidate and nothing downstream
    means anything.
    """
    ctrl = reference["ctrl"].astype(np.float64)
    rmap = orb.ReturnMap(muscle_model, ctrl, activation=reference["act"][0].astype(np.float64))
    out = rmap.roll(orb.shape_state(reference["qpos"][0], reference["qvel"][0]))

    want = orb.shape_state(reference["closing_qpos"], reference["closing_qvel"])
    nq = muscle_model.nq
    assert np.abs(out.state[: nq - 1] - want[: nq - 1]).max() <= 1e-5, "qpos parity"
    assert np.abs(out.state[nq - 1:] - want[nq - 1:]).max() <= 1e-4, "qvel parity"
    assert np.abs(out.activation - reference["closing_act"]).max() <= 1e-6, "act parity"


def test_advance_spans_exactly_one_cycle(muscle_model, reference):
    """Guards the off-by-one that makes the first cycle one step short (IN-02).

    One ``roll`` of one cycle must advance exactly ``advance_per_cycle``, not ``T_c - 1`` steps of it.
    """
    ctrl = reference["ctrl"].astype(np.float64)
    rmap = orb.ReturnMap(muscle_model, ctrl, activation=reference["act"][0].astype(np.float64))
    out = rmap.roll(orb.shape_state(reference["qpos"][0], reference["qvel"][0]))
    assert abs(out.advance - float(reference["advance_per_cycle"])) <= 1e-5


def test_within_cycle_height_range_is_a_shallow_bounce(muscle_model, reference):
    """Guards IN-01: the retracted 0.508 m was a three-cycle range dominated by divergence.

    The within-cycle range is 0.0714 m, about 38% of the real motor gait's per-cycle 0.1857 m.
    """
    ctrl = reference["ctrl"].astype(np.float64)
    rmap = orb.ReturnMap(muscle_model, ctrl, activation=reference["act"][0].astype(np.float64))
    out = rmap.roll(orb.shape_state(reference["qpos"][0], reference["qvel"][0]))

    # 0.073186 here against 0.071374 read off the reference table's 16 pre-step phases. The
    # difference is the sample set, not the physics: roll() sees the pre-step height at phase 0 and
    # then the post-step height at phases 1..16, so it includes the closing state that the table
    # omits. Both are recorded in docs/VALIDITY.md IN-01. The load-bearing assertion is the second
    # one -- the retracted figure was 0.508 m, seven times either of these.
    assert out.height_range == pytest.approx(0.0732, abs=1e-3)
    assert out.height_range < 0.1, "a multi-cycle range would be ~0.5 m; see IN-01"


def test_reference_cycle_has_five_stance_steps(muscle_model, reference):
    """Stance duty, which bounds what any feedback correction can do (MF-10).

    Five of sixteen control steps in contact, at phases 0, 1, 13, 14, 15 -- against a measured
    activation release constant of 1.2 to 4.3 control steps.
    """
    ctrl = reference["ctrl"].astype(np.float64)
    rmap = orb.ReturnMap(muscle_model, ctrl, activation=reference["act"][0].astype(np.float64))
    out = rmap.roll(orb.shape_state(reference["qpos"][0], reference["qvel"][0]))
    stance = [i for i, n in enumerate(out.contacts) if n > 0]
    assert stance == [0, 1, 13, 14, 15]
    assert all(len(p) <= 1 for p in out.contact_pairs), "only the floor/foot pair should ever touch"


def test_scaled_residual_reproduces_the_recorded_phase4_number(muscle_model, reference):
    """The arithmetic check: this module must reproduce 0.31744 on the recorded start.

    Pinning activation to its limit cycle instead **worsens** it to 0.410, which is the expected
    direction -- the candidate was optimised with the off-manifold start as a crutch (CL-02).
    """
    ctrl = reference["ctrl"].astype(np.float64)
    scales = reference["shape_scales"].astype(np.float64)
    x0 = orb.shape_state(reference["qpos"][0], reference["qvel"][0])

    as_recorded = orb.ReturnMap(muscle_model, ctrl, activation=reference["act"][0].astype(np.float64))
    assert as_recorded.scaled_residual(x0, scales) == pytest.approx(0.31744, abs=1e-4)

    on_manifold = orb.ReturnMap(muscle_model, ctrl)  # activation defaults to act*
    np.testing.assert_allclose(on_manifold.activation, ACT_STAR, atol=1e-5)
    assert on_manifold.scaled_residual(x0, scales) == pytest.approx(0.410, abs=2e-3)


def test_return_map_raises_when_the_rollout_terminates(muscle_model, reference):
    """The candidate survives 3 cycles and fails at 4, measured. ``P`` must say so, not guess."""
    ctrl = reference["ctrl"].astype(np.float64)
    rmap = orb.ReturnMap(muscle_model, ctrl, activation=reference["act"][0].astype(np.float64))
    x0 = orb.shape_state(reference["qpos"][0], reference["qvel"][0])

    assert not rmap.roll(x0, cycles=3).terminated
    four = rmap.roll(x0, cycles=4)
    assert four.terminated and four.terminated_at is not None
    with pytest.raises(orb.OrbitTerminated, match="terminated at control step"):
        rmap(x0, cycles=4)


def test_include_activation_carries_act_as_a_coordinate(muscle_model, reference):
    """The 17-dim formulation must agree with the pinned 11-dim one on the same trajectory."""
    ctrl = reference["ctrl"].astype(np.float64)
    a0 = reference["act"][0].astype(np.float64)
    full = orb.ReturnMap(muscle_model, ctrl, include_activation=True)
    pinned = orb.ReturnMap(muscle_model, ctrl, activation=a0)
    assert full.dim == 17 and pinned.dim == 11

    x_full = orb.shape_state(reference["qpos"][0], reference["qvel"][0], a0)
    x_pin = orb.shape_state(reference["qpos"][0], reference["qvel"][0])
    np.testing.assert_allclose(full(x_full)[:11], pinned(x_pin), atol=1e-12)


def test_return_map_rejects_a_mis_shaped_state_or_control(muscle_model, reference):
    ctrl = reference["ctrl"].astype(np.float64)
    rmap = orb.ReturnMap(muscle_model, ctrl, activation=reference["act"][0].astype(np.float64))
    with pytest.raises(ValueError, match=r"state must be"):
        rmap(np.zeros(7))
    with pytest.raises(ValueError, match=r"ctrl_seq must be"):
        orb.ReturnMap(muscle_model, np.zeros((16, 3)))


def test_motor_hopper_return_map_is_eleven_dimensional(motor_model):
    """The positive control shares every code path; ``na == 0`` must not be a special case."""
    rmap = orb.ReturnMap(motor_model, np.zeros((27, motor_model.nu)))
    assert rmap.dim == 11 and rmap.cycle == 27
    out = rmap.roll(np.zeros(11))
    assert out.state.shape == (11,)
    assert out.activation.shape == (0,)
    assert out.act_excursions == 0


def test_analysis_package_imports_without_torch_or_warp():
    """The no-torch, no-warp claim in the package docstring, enforced.

    It is load-bearing rather than cosmetic: it is what makes the instrument independent of the
    Warp adjoint, and therefore of backend defects BE-01, BE-02 and BE-06. A stray import would
    silently reattach the dependency the whole design exists to cut.
    """
    import subprocess
    import sys

    code = (
        "import sys, msk_warp.analysis as a;"
        "bad=[m for m in ('torch','warp','mujoco_warp') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(__import__('msk_warp').PACKAGE_ROOT.parent))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"msk_warp.analysis pulled in: {out.stdout.strip()}"
