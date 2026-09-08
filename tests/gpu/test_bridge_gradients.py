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


@pytest.mark.xfail(reason="pr1423 defect 1: spatial-tendon length/moment arm not differentiated w.r.t. qpos (findings.md); the qpos error compounds into act/qvel over a muscle-dominated rollout", strict=False)
def test_muscle_multistep_matches_fd():
    mjm = mujoco.MjModel.from_xml_string(MUSCLE_PENDULUM_XML)
    res = _run(mjm, "muscle", "tape_per_substep", horizon=4, substeps=2)
    for f in ("qpos", "act", "qvel", "ctrl"):
        ad, fd = res[f]
        assert _rel(ad, fd) < 5e-2, (f, ad, fd)


def test_fd_mode_matches_tape_on_muscle():
    """The finite-difference control mode agrees with the tape on the fields the tape gets right."""
    mjm = mujoco.MjModel.from_xml_string(MUSCLE_PENDULUM_XML)
    a = _run(mjm, "muscle", "fd", horizon=2, substeps=1)
    b = _run(mjm, "muscle", "tape_per_substep", horizon=2, substeps=1)
    for f in ("act", "qvel", "ctrl"):
        assert _cos(a[f][0], b[f][0]) > 0.99, f
        assert _rel(a[f][0], b[f][0]) < 2e-2, f


def test_no_gradients_were_sanitized():
    assert bridge.sanitized_nan_count == 0
