"""The gradient contract refuses models whose gradients would be silently wrong and passes the anchors."""

import mujoco
import pytest

from msk_warp import backend, resolve_model_path


@pytest.mark.parametrize("model_path, njmax", [("assets/cartpole.xml", None), ("assets/ant_soft_v2.xml", 512)])
def test_anchor_models_pass_contract(model_path, njmax):
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path(model_path))
    m = backend.put_model(mjm)
    d = backend.make_data(mjm, m, 4, njmax)
    rep = backend.assert_grad_contract(mjm, m, d, njmax=njmax)
    assert rep.ok, rep.problems
    assert rep.tripwire["field"] == "ctrl"
    assert rep.tripwire["nan"] == 0
    assert rep.tripwire["cos"] > 0.99, rep.tripwire


MESH_ON_PLANE_XML = """
<mujoco>
  <option solver="Newton" jacobian="dense"/>
  <asset><mesh name="tet" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.05"><freejoint/><geom type="mesh" mesh="tet" mass="0.1"/></body>
  </worldbody>
</mujoco>
"""


def test_mesh_contact_pair_is_rejected():
    mjm = mujoco.MjModel.from_xml_string(MESH_ON_PLANE_XML)
    m = backend.put_model(mjm)
    d = backend.make_data(mjm, m, 1, 64)
    with pytest.raises(backend.GradContractError) as excinfo:
        backend.assert_grad_contract(mjm, m, d, njmax=64, tripwire=False)
    assert "mesh" in str(excinfo.value)


def test_myoleg26_baseline_cannot_be_used_as_is():
    """The all-mesh baseline is refused before any gradient runs: upstream rejects mesh pairs with a
    non-zero margin under MULTICCD, and the contract rejects mesh pairs outright (Phase 4 replaces the
    feet with sphere/capsule proxies and zero-margin, non-colliding meshes)."""
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path("assets/myoleg/myoLeg26_BASELINE.xml"))
    with pytest.raises((backend.GradContractError, NotImplementedError)) as excinfo:
        m = backend.put_model(mjm)
        d = backend.make_data(mjm, m, 1, 1000)
        backend.assert_grad_contract(mjm, m, d, njmax=1000, tripwire=False)
    assert "mesh" in str(excinfo.value).lower() or "MULTICCD" in str(excinfo.value)


def test_non_newton_solver_is_rejected():
    xml = """
    <mujoco><option solver="CG"/>
      <worldbody><body><joint type="hinge" axis="0 1 0"/><geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3"/></body></worldbody>
      <actuator><motor joint="" gear="1"/></actuator>
    </mujoco>""".replace('joint=""', 'joint="j"').replace('<joint type="hinge"', '<joint name="j" type="hinge"')
    mjm = mujoco.MjModel.from_xml_string(xml)
    m = backend.put_model(mjm)
    d = backend.make_data(mjm, m, 1)
    with pytest.raises(backend.GradContractError) as excinfo:
        backend.assert_grad_contract(mjm, m, d, tripwire=False)
    assert "Newton" in str(excinfo.value)


def test_no_grad_data_fails_data_check():
    mjm = mujoco.MjModel.from_xml_path(resolve_model_path("assets/cartpole.xml"))
    m = backend.put_model(mjm)
    d = backend.make_data(mjm, m, 2, grad=False)
    problems = backend.check_data(mjm, d)
    assert any("qpos" in p for p in problems)
