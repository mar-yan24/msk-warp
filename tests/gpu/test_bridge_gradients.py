"""WarpSimStep gradients against finite differences, through the same Function the envs use.

fp32 central differences are batched across worlds (world 0 is the base point, worlds
2k+1 / 2k+2 perturb coordinate k), so one forward call gives the whole FD gradient. AD is the
backward of the world-0 loss through ``WarpSimStep`` (single env step) or a chain of them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from msk_warp import backend, bridge, resolve_model_path
from msk_warp.bridge import WarpSimStep

import mujoco  # noqa: E402

DEVICE = "cuda:0"

# myoLeg26 muscle class, two antagonist spatial-tendon muscles on one hinge (no contact)
MUSCLE_PENDULUM_XML = """
<mujoco model="muscle_pendulum">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" solver="Newton" jacobian="dense"><flag contact="disable"/></option>
  <default><default class="muscle"><general biasprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2 0" biastype="muscle"
     ctrllimited="true" ctrlrange="0 1" dynprm="0.01 0.04 0 0 0 0 0 0 0 0" dyntype="muscle"
     gainprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2 0" gaintype="muscle"/></default></default>
  <worldbody>
    <site name="anchor_f" pos="0.15 0 0.05"/><site name="anchor_b" pos="-0.15 0 0.05"/>
    <body name="link"><joint name="j0" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.1"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.4" mass="1"/>
      <site name="ins_f" pos="0.05 0 -0.15"/><site name="ins_b" pos="-0.05 0 -0.15"/></body>
  </worldbody>
  <tendon><spatial name="t_f" width="0.005"><site site="anchor_f"/><site site="ins_f"/></spatial>
          <spatial name="t_b" width="0.005"><site site="anchor_b"/><site site="ins_b"/></spatial></tendon>
  <actuator><general class="muscle" name="m_f" tendon="t_f"/><general class="muscle" name="m_b" tendon="t_b"/></actuator>
</mujoco>
"""


class _Env:
    """Minimal duck-typed env for the bridge: model, data, substeps, backward mode."""

    def __init__(self, mjm, nworld, substeps, backward_mode, njmax=None):
        self.mjm = mjm
        self.warp_model = backend.put_model(mjm)
        self.warp_data = backend.make_data(mjm, self.warp_model, nworld, njmax)
        import mujoco_warp as mjw

        mjw.reset_data(self.warp_model, self.warp_data)
        self.substeps = substeps
        self.backward_mode = backward_mode
        self.rerun_after_backward = True
        self.fd_eps = 1e-3


def _nominal(mjm, kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "cartpole":
        qpos = np.array([0.1, 2.9]); qvel = np.array([0.05, -0.1])
    elif kind.startswith("hopper"):
        d = mujoco.MjData(mjm)
        if mjm.na:
            d.act[:] = 0.5
            d.ctrl[:] = 0.5
        for _ in range(200):
            mujoco.mj_step(mjm, d)  # settle onto the foot, contacts active
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        # Standing settles the knee onto its upper limit (0.0004 rad past it), where the limit
        # constraint switches on and the derivative is not defined: the fp32 central difference
        # there swings from -7 to +52 to +24 as eps goes 1e-2, 1e-3, 1e-4. The bake-off rubric
        # excludes states inside an active-set switch, so every limited joint is moved a margin
        # inside its range before the comparison.
        for j in range(mjm.njnt):
            if mjm.jnt_limited[j]:
                a = mjm.jnt_qposadr[j]
                lo, hi = mjm.jnt_range[j]
                qpos[a] = min(max(qpos[a], lo + 0.05), hi - 0.05)
    elif kind == "ant":
        d = mujoco.MjData(mjm)
        d.qpos[:] = [0, 0, 0.75, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1]
        for _ in range(120):
            mujoco.mj_step(mjm, d)  # settle onto the floor (CPU MuJoCo)
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
    else:
        qpos = np.array([0.3]); qvel = np.array([0.2])
    qpos = qpos + rng.normal(0, 0.005, mjm.nq)
    for j in range(mjm.njnt):
        if mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = mjm.jnt_qposadr[j] + 3
            qpos[a:a + 4] /= np.linalg.norm(qpos[a:a + 4])
    qvel = qvel + rng.normal(0, 0.01, mjm.nv)
    act = rng.uniform(0.2, 0.6, mjm.na)
    lo = np.where(mjm.actuator_ctrllimited.astype(bool), mjm.actuator_ctrlrange[:, 0], -1.0)
    hi = np.where(mjm.actuator_ctrllimited.astype(bool), mjm.actuator_ctrlrange[:, 1], 1.0)
    ctrl = np.clip(0.5 * (lo + hi) + 0.3 * (hi - lo) * rng.uniform(-0.5, 0.5, mjm.nu), lo, hi)
    return {k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in dict(qpos=qpos, qvel=qvel, act=act, ctrl=ctrl).items()}


def _run(mjm, kind, backward_mode, horizon, substeps, njmax=None, seed=0):
    """Return {field: (ad, fd)} for L = a . qpos_H + b . qvel_H + c . act_H with random weights."""
    st = _nominal(mjm, kind, seed)
    fields = [f for f in ("qpos", "qvel", "act", "ctrl") if st[f].numel() > 0]
    coords = [(f, i) for f in fields for i in range(st[f].numel())]
    nworld = 1 + 2 * len(coords)
    env = _Env(mjm, nworld, substeps, backward_mode, njmax)
    rng = np.random.default_rng(seed + 100)
    w = {f: torch.tensor(rng.normal(size=st[f].numel()), dtype=torch.float32, device=DEVICE) for f in ("qpos", "qvel", "act")}
    eps = 1e-3
    inputs = {f: st[f][None].repeat(nworld, 1).clone() for f in ("qpos", "qvel", "act", "ctrl")}
    for k, (f, i) in enumerate(coords):
        inputs[f][1 + 2 * k, i] += eps
        inputs[f][2 + 2 * k, i] -= eps
    for f in ("qpos", "qvel", "act", "ctrl"):
        inputs[f].requires_grad_(True)
    qpos, qvel, act = inputs["qpos"], inputs["qvel"], inputs["act"]
    for _ in range(horizon):
        qpos, qvel, act = WarpSimStep.apply(inputs["ctrl"], qpos, qvel, act, env)
    per_world = (qpos * w["qpos"]).sum(1) + (qvel * w["qvel"]).sum(1) + ((act * w["act"]).sum(1) if act.shape[1] else 0.0)
    L = per_world.detach().double().cpu().numpy()
    per_world[0].backward()
    out = {}
    for f in fields:
        ad = inputs[f].grad[0].double().cpu().numpy()
        fd = np.zeros(st[f].numel())
        for k, (ff, i) in enumerate(coords):
            if ff == f:
                fd[i] = (L[1 + 2 * k] - L[2 + 2 * k]) / (2 * eps)
        out[f] = (ad, fd)
    return out


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _cartpole():
    xml = open(resolve_model_path("assets/cartpole.xml"), encoding="utf-8").read().replace('jacobian="sparse"', 'jacobian="dense"')
    return mujoco.MjModel.from_xml_string(xml)


@pytest.mark.parametrize("mode", ["tape_per_substep", "tape", "fd"])
@pytest.mark.parametrize("horizon,substeps", [(1, 4), (4, 2)])
def test_cartpole_all_fields_match_fd(mode, horizon, substeps):
    res = _run(_cartpole(), "cartpole", mode, horizon, substeps)
    for f, (ad, fd) in res.items():
        assert np.isfinite(ad).all(), f
        assert _cos(ad, fd) > 0.999, (f, ad, fd)
        assert _rel(ad, fd) < 2e-2, (f, ad, fd)


@pytest.mark.parametrize("mode", ["tape_per_substep", "tape"])
def test_ant_ctrl_and_qvel_match_fd(mode):
    """One physics step at the settled contact pose: ctrl and qvel gradients are exact (bake-off R2)."""
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path("assets/ant_soft_v2.xml"))
    res = _run(mjm, "ant", mode, horizon=1, substeps=1, njmax=512)
    for f in ("ctrl", "qvel"):
        ad, fd = res[f]
        assert np.isfinite(ad).all(), f
        assert _cos(ad, fd) > 0.99 and _rel(ad, fd) < 2e-2, (f, ad, fd)
    assert np.isfinite(res["qpos"][0]).all()


def test_ant_four_substeps_ctrl_direction_holds():
    """Over an env step of 4 substeps the root-qpos defect leaks into ctrl by a few percent; the
    direction stays intact (this is the training-relevant quantity until defect 2 is fixed)."""
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path("assets/ant_soft_v2.xml"))
    res = _run(mjm, "ant", "tape_per_substep", horizon=1, substeps=4, njmax=512)
    ad, fd = res["ctrl"]
    assert _cos(ad, fd) > 0.99, (ad, fd)
    assert _rel(ad, fd) < 0.1, (ad, fd)


@pytest.mark.xfail(reason="pr1423 defect 2: free-joint root qpos gradient through contact is 1.3-15x too large (findings.md)", strict=False)
def test_ant_qpos_matches_fd():
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path("assets/ant_soft_v2.xml"))
    res = _run(mjm, "ant", "tape_per_substep", horizon=1, substeps=4, njmax=512)
    ad, fd = res["qpos"]
    assert _cos(ad, fd) > 0.99 and _rel(ad, fd) < 5e-2, (ad, fd)


@pytest.mark.parametrize("mode", ["tape_per_substep", "tape"])
def test_muscle_act_qvel_and_ctrl_match_fd(mode):
    """Two physics steps (ctrl needs at least two to reach the state through act_dot)."""
    mjm = mujoco.MjModel.from_xml_string(MUSCLE_PENDULUM_XML)
    res = _run(mjm, "muscle", mode, horizon=2, substeps=1)
    for f in ("act", "qvel", "ctrl"):
        ad, fd = res[f]
        assert np.isfinite(ad).all(), f
        assert _cos(ad, fd) > 0.99 and _rel(ad, fd) < 2e-2, (f, ad, fd)


def test_muscle_multistep_matches_fd():
    """Eight physics steps of a tendon-driven muscle model.

    The direction has to be exact; the magnitude tolerance is looser than the short-horizon tests
    because the fp32 finite-difference reference itself degrades as the rollout lengthens (the same
    reason the bake-off judged its 60-DOF rows against float64 CPU differences).
    """
    mjm = mujoco.MjModel.from_xml_string(MUSCLE_PENDULUM_XML)
    res = _run(mjm, "muscle", "tape_per_substep", horizon=4, substeps=2)
    for f in ("qpos", "act", "qvel", "ctrl"):
        ad, fd = res[f]
        assert np.isfinite(ad).all(), f
        assert _cos(ad, fd) > 0.999, (f, ad, fd)
        assert _rel(ad, fd) < 0.1, (f, ad, fd)


def test_fd_mode_matches_tape_on_muscle():
    """The finite-difference control mode agrees with the tape on the fields the tape gets right."""
    mjm = mujoco.MjModel.from_xml_string(MUSCLE_PENDULUM_XML)
    a = _run(mjm, "muscle", "fd", horizon=2, substeps=1)
    b = _run(mjm, "muscle", "tape_per_substep", horizon=2, substeps=1)
    for f in ("act", "qvel", "ctrl"):
        assert _cos(a[f][0], b[f][0]) > 0.99, f
        assert _rel(a[f][0], b[f][0]) < 2e-2, f


def _hopper(kind, smooth=False):
    """Hopper model; ``smooth=True`` disables contacts and joint limits.

    The smooth variant exists because the finite-difference reference is only meaningful away from
    active-set switches. With contacts and limits on, a perturbation at some states crosses a
    constraint boundary and the difference quotient jumps by one to two orders of magnitude, which
    swamps any adjoint comparison. Turning both off leaves purely smooth dynamics, where the motor
    hopper is exact to 1.00000 at every seed and the muscle hopper is not.
    """
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path(f"assets/hopper_{kind}.xml"))
    if smooth:
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mjm.jnt_limited[:] = 0
    return mjm


def _hopper_qpos_cosines(kind, seeds=range(4), smooth=True):
    out = []
    for seed in seeds:
        res = _run(_hopper(kind, smooth), f"hopper_{kind}", "tape_per_substep",
                   horizon=2, substeps=2, njmax=128, seed=seed)
        ad, fd = res["qpos"]
        assert np.isfinite(ad).all(), (kind, seed)
        assert np.abs(ad).max() > 0, f"{kind} seed {seed}: qpos gradient is identically zero"
        out.append(_cos(ad, fd))
    return np.array(out)


@pytest.mark.parametrize("mode", ["tape_per_substep", "tape"])
def test_hopper_motor_matches_fd(mode):
    """Motor hopper at a settled contact pose (Phase 3 gate G3.2)."""
    res = _run(_hopper("motor"), "hopper_motor", mode, horizon=1, substeps=4, njmax=128)
    for f, floor in (("ctrl", 0.99), ("qvel", 0.95), ("qpos", 0.95)):
        ad, fd = res[f]
        assert np.isfinite(ad).all(), f
        assert np.abs(ad).max() > 0, f"{f} gradient is identically zero"
        assert _cos(ad, fd) > floor, (f, _cos(ad, fd), ad, fd)


def test_hopper_motor_eight_steps_ctrl_direction_holds():
    res = _run(_hopper("motor"), "hopper_motor", "tape_per_substep", horizon=8, substeps=1, njmax=128)
    ad, fd = res["ctrl"]
    assert np.isfinite(ad).all()
    assert _cos(ad, fd) > 0.90, (_cos(ad, fd), ad, fd)


def test_hopper_motor_smooth_qpos_is_exact():
    """Control for the muscle case: the same skeleton with torque motors is exact at every seed."""
    cos = _hopper_qpos_cosines("motor")
    assert cos.min() > 0.999, cos


@pytest.mark.parametrize("mode", ["tape_per_substep", "tape"])
def test_hopper_muscle_matches_fd(mode):
    """Muscle hopper at a settled contact pose: contact and tendon gradients together (G3.2).

    Two control steps, because ctrl reaches the dynamics only through ``act_dot -> act`` and so
    has no effect on the state after a single step.
    """
    res = _run(_hopper("muscle"), "hopper_muscle", mode, horizon=2, substeps=2, njmax=128)
    for f, floor in (("act", 0.99), ("ctrl", 0.95), ("qvel", 0.95)):
        ad, fd = res[f]
        assert np.isfinite(ad).all(), f
        assert np.abs(ad).max() > 0, f"{f} gradient is identically zero"
        assert _cos(ad, fd) > floor, (f, _cos(ad, fd), ad, fd)


def test_hopper_muscle_eight_steps_ctrl_direction_holds():
    res = _run(_hopper("muscle"), "hopper_muscle", "tape_per_substep", horizon=8, substeps=1, njmax=128)
    ad, fd = res["ctrl"]
    assert np.isfinite(ad).all()
    assert _cos(ad, fd) > 0.85, (_cos(ad, fd), ad, fd)


@pytest.mark.xfail(
    reason="pr1423 defect 3: the tendon path's dL/dqpos is wrong. On smooth dynamics over 8 seeds "
           "the muscle hopper gives cosine 0.934 (min 0.865) and relative error 0.29-0.60, while "
           "the same skeleton with torque motors gives 1.00000 and 4e-4. Present in a single "
           "physics step, identical in both tape modes, and unchanged by the force-velocity curve, "
           "the force-length curve or welding the root. See docs/research/phase3-hopper/results.md",
    strict=False)
def test_hopper_muscle_smooth_qpos_matches_fd():
    cos = _hopper_qpos_cosines("muscle")
    assert cos.min() > 0.999, cos


def test_hopper_muscle_smooth_qpos_direction_is_usable():
    """The defect-3 gap is bounded: the direction stays mostly right and no field goes to zero."""
    cos = _hopper_qpos_cosines("muscle")
    assert cos.min() > 0.85, cos


# Minimal reproduction of pr1423 defect 3, found by bisecting down from the hopper. Two hinges in
# one chain, one two-site tendon spanning each, driven hard. Every smaller model is exact: one
# hinge with one tendon at any force, two hinges with only ONE tendon at any force, and a single
# hinge with an antagonist pair. The error needs two tendons on two distinct joints and grows
# monotonically with tendon force, which is the signature of a term proportional to force being
# lost, i.e. the (d ten_J / dq)^T F half of d(qfrc)/dq.
TWO_TENDON_CHAIN_XML = """
<mujoco>
  <option timestep="0.004" gravity="0 0 -9.81" solver="Newton" jacobian="dense"
          iterations="20" ls_iterations="20"><flag contact="disable"/></option>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <geom type="capsule" size="0.05 0.2" mass="2"/>
      <site name="s0" pos="0.09 0 -0.14"/>
      <body name="b1" pos="0 0 -0.2">
        <joint name="j1" type="hinge" axis="0 -1 0" damping="1" armature="1"/>
        <geom type="capsule" size="0.05 0.2" pos="0 0 -0.2" mass="2"/>
        <site name="s1" pos="0.07 0 -0.11"/>
        <body name="b2" pos="0 0 -0.4">
          <joint name="j2" type="hinge" axis="0 -1 0" damping="1" armature="1"/>
          <geom type="capsule" size="0.05 0.2" pos="0 0 -0.2" mass="2"/>
          <site name="s2" pos="0.07 0 -0.11"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t0"><site site="s0"/><site site="s1"/></spatial>
    <spatial name="t1"><site site="s1"/><site site="s2"/></spatial>
  </tendon>
  <actuator>
    <motor tendon="t0" gear="{gear}" ctrllimited="true" ctrlrange="0 1"/>
    <motor tendon="t1" gear="{gear}" ctrllimited="true" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""

ONE_TENDON_CHAIN_XML = TWO_TENDON_CHAIN_XML.replace(
    '<spatial name="t1"><site site="s1"/><site site="s2"/></spatial>', "").replace(
    '<motor tendon="t1" gear="{gear}" ctrllimited="true" ctrlrange="0 1"/>', "")


def _chain_qpos_cosines(xml, gear, seeds=range(4)):
    mjm = mujoco.MjModel.from_xml_string(xml.format(gear=gear))
    out = []
    for seed in seeds:
        res = _run(mjm, "hopper_motor", "tape_per_substep", horizon=2, substeps=2, seed=seed)
        ad, fd = res["qpos"]
        assert np.isfinite(ad).all(), seed
        out.append(_cos(ad, fd))
    return np.array(out)


def test_single_tendon_chain_qpos_is_exact():
    """The control for the defect-3 reproduction: one tendon in the same chain is exact."""
    for gear in (3000, 12000):
        cos = _chain_qpos_cosines(ONE_TENDON_CHAIN_XML, gear)
        assert cos.min() > 0.999, (gear, cos)


@pytest.mark.xfail(
    reason="pr1423 defect 3, minimal reproduction: two tendons on two joints of one chain. "
           "Adding the second tendon breaks dL/dqpos and the error grows with tendon force "
           "(cosine 0.98 at gear 800, 0.59 at 3000, 0.12 at 12000), while the same chain with one "
           "tendon is exact at every force. See docs/research/phase3-hopper/results.md",
    strict=False)
def test_two_tendon_chain_qpos_matches_fd():
    cos = _chain_qpos_cosines(TWO_TENDON_CHAIN_XML, 3000)
    assert cos.min() > 0.99, cos


def test_two_tendon_chain_error_grows_with_force():
    """Pin the dose-response, so a backend fix or regression is visible in the shape, not just one
    number. Low force stays usable; high force degrades."""
    low = _chain_qpos_cosines(TWO_TENDON_CHAIN_XML, 800).mean()
    high = _chain_qpos_cosines(TWO_TENDON_CHAIN_XML, 12000).mean()
    assert low > high, (low, high)
    assert low > 0.9, low


def test_no_gradients_were_sanitized():
    assert bridge.sanitized_nan_count == 0
