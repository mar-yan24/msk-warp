"""Pinned reference provenance and the explicit collision approximation's invariants."""

import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from msk_warp.models.myoleg26 import (
    COLLISION_PREFIX, FOOT_BODIES, GROUND_NAME, assert_biology_unchanged,
    collision_body_bounds, geom_body_bounds,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "msk_warp/assets/myoleg26"
PIN = "eb327acbae0fad12279495040607f5235d962328"


@pytest.fixture(scope="module")
def models():
    return tuple(mujoco.MjModel.from_xml_path(str(ASSETS / name))
                 for name in ("reference.xml", "flat_boxes.xml"))


def test_official_reference_signature_and_vendored_hashes(models):
    reference, _ = models
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    assert manifest["source"]["commit"] == PIN and manifest["source"]["clean"]
    assert (reference.nq, reference.nv, reference.nu, reference.na, reference.neq) == (47, 46, 26, 26, 28)
    assert reference.jnt_type[reference.joint("root").id] == mujoco.mjtJoint.mjJNT_FREE
    assert reference.body_mass.sum() == pytest.approx(78.09046847904195, abs=1e-12)
    for relative, digest in manifest["files"].items():
        path = (ASSETS / relative).resolve()
        assert path.is_relative_to(ASSETS.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative


def test_task_preserves_mass_kinematics_tendons_and_muscles(models):
    reference, task = models
    assert_biology_unchanged(reference, task)
    np.testing.assert_array_equal(reference.body_mass, task.body_mass)
    np.testing.assert_array_equal(reference.body_inertia, task.body_inertia)
    np.testing.assert_array_equal(reference.qpos0, task.qpos0)
    assert reference.nkey == 0 and task.nkey == 1


def test_task_collision_graph_is_only_ground_against_measured_body_boxes(models):
    reference, task = models
    expected = collision_body_bounds(reference)
    assert len(expected) == 14
    assert task.npair == 0
    ground = task.geom(GROUND_NAME).id
    assert task.geom_type[ground] == mujoco.mjtGeom.mjGEOM_PLANE
    np.testing.assert_array_equal(task.geom_pos[ground], [0, 0, 0])
    assert (task.geom_contype[ground], task.geom_conaffinity[ground]) == (1, 2)
    allowed = {ground}
    for body, bounds in expected.items():
        geom = task.geom(COLLISION_PREFIX + body).id
        allowed.add(geom)
        assert task.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
        assert task.geom_bodyid[geom] == task.body(body).id
        assert (task.geom_contype[geom], task.geom_conaffinity[geom]) == (2, 1)
        # Reference serialization is checked at 1e-10 during export. These hull coordinates
        # also inherit float32 mesh vertices; tolerate their representational precision.
        np.testing.assert_allclose(task.geom_pos[geom] - task.geom_size[geom], bounds["lower"], atol=2e-7, rtol=0)
        np.testing.assert_allclose(task.geom_pos[geom] + task.geom_size[geom], bounds["upper"], atol=2e-7, rtol=0)
    for geom in set(range(task.ngeom)) - allowed:
        assert task.geom_contype[geom] == task.geom_conaffinity[geom] == 0
    assert len({task.geom(COLLISION_PREFIX + body).bodyid[0] for body in FOOT_BODIES}) == 4


def test_nominal_reset_has_world_heading_and_measured_surface_clearance(models):
    _, task = models
    data = mujoco.MjData(task)
    mujoco.mj_resetDataKeyframe(task, data, task.key("stand").id)
    mujoco.mj_forward(task, data)
    pelvis_forward = data.xmat[task.body("pelvis").id].reshape(3, 3)[:, 0]
    np.testing.assert_allclose(pelvis_forward, [1, 0, 0], atol=1e-10, rtol=0)
    bottom = []
    for body in FOOT_BODIES:
        geom = task.geom(COLLISION_PREFIX + body).id
        bottom.append(data.geom_xpos[geom, 2] - abs(data.geom_xmat[geom].reshape(3, 3)[2]) @ task.geom_size[geom])
    assert min(bottom) == pytest.approx(0.001, abs=1e-9)
    # Surface clearance is not a promise of zero speculative contacts: margin is also 1 mm.
    assert task.opt.timestep == 0.002
    assert task.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert task.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert task.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE
    assert task.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_WARMSTART


def test_mesh_bounds_undo_mujoco_recentring_and_geometry_rotation():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <compiler angle="radian"/>
      <asset><mesh name="cube" vertex="1 2 3  3 2 3  1 4 3  3 4 3  1 2 5  3 2 5  1 4 5  3 4 5"/></asset>
      <worldbody><body name="test"><geom name="cube" type="mesh" mesh="cube"
        pos="0.2 -0.3 0.4" euler="0 0 1.5707963267948966"/></body></worldbody>
      </mujoco>''')
    lower, upper = geom_body_bounds(model, model.geom("cube").id)
    np.testing.assert_allclose(lower, [-3.8, 0.7, 3.4], atol=1e-12)
    np.testing.assert_allclose(upper, [-1.8, 2.7, 5.4], atol=1e-12)


def test_export_gate_rejects_geometry_and_option_changes(models):
    spec = importlib.util.spec_from_file_location("build_myoleg26_assets", ROOT / "scripts/build_myoleg26_assets.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    _, task = models
    changed = mujoco.MjModel.from_xml_path(str(ASSETS / "flat_boxes.xml"))
    changed.geom_pos[changed.geom(COLLISION_PREFIX + "calcn_r").id, 0] += 0.01
    with pytest.raises(ValueError, match="geom_pos"):
        exporter.assert_export_equivalent(task, changed)
    changed = mujoco.MjModel.from_xml_path(str(ASSETS / "flat_boxes.xml"))
    changed.opt.timestep *= 2
    with pytest.raises(ValueError, match="timestep"):
        exporter.assert_export_equivalent(task, changed)
