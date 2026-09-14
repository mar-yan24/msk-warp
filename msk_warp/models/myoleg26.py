"""Measured flat-ground collision task derived from the official MyoLeg26 reference.

Boxes are conservative body-frame bounds of the original collision surfaces, not an
anatomically validated skin model. All original geometry remains for its visual/inertial
contribution. Only the collision graph, simulation options and a nominal reset key change.
The official free root remains; this module does not assert that its adjoint is validated.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

FOOT_BODIES = ("calcn_r", "toes_r", "calcn_l", "toes_l")
COLLISION_PREFIX = "task_collision_"
GROUND_NAME = "task_ground"
BIOLOGY_FIELDS = (
    "body_parentid", "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
    "qpos0", "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range",
    "jnt_limited", "jnt_stiffness", "jnt_margin", "dof_armature", "dof_damping", "dof_frictionloss",
    "eq_type", "eq_obj1id", "eq_obj2id", "eq_data", "eq_solref", "eq_solimp", "eq_active0",
    "site_pos", "site_quat", "site_bodyid", "tendon_adr", "tendon_num", "tendon_range",
    "tendon_limited", "tendon_stiffness", "tendon_damping", "tendon_frictionloss", "tendon_lengthspring",
    "tendon_solref_lim", "tendon_solimp_lim", "wrap_type", "wrap_objid", "wrap_prm",
    "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_dyntype", "actuator_gaintype",
    "actuator_biastype", "actuator_dynprm", "actuator_gainprm", "actuator_biasprm",
    "actuator_lengthrange", "actuator_ctrlrange", "actuator_ctrllimited", "actuator_actrange",
    "actuator_actlimited", "actuator_actadr", "actuator_actnum",
)


def _rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def geom_body_bounds(model, geom_id):
    """Exact body-frame AABB of a compiled mesh or analytic primitive surface.

    Compiled mesh vertices have been recentered/rotated by MuJoCo. geom_pos/quat undo
    that transformation; treating mesh vertices as original body coordinates is wrong.
    """
    kind = int(model.geom_type[geom_id])
    center, rotation = model.geom_pos[geom_id], _rotation(model.geom_quat[geom_id])
    size = model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom_id])
        first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[first:first + count].astype(np.float64) @ rotation.T + center
        return vertices.min(axis=0), vertices.max(axis=0)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.full(3, size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = np.abs(rotation[:, 2]) * size[1] + size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.sqrt(np.sum((rotation * size) ** 2, axis=1))
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(rotation) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[0] * np.sqrt(np.sum(rotation[:, :2] ** 2, axis=1)) + size[1] * np.abs(rotation[:, 2])
    else:
        raise ValueError(f"unsupported body collision surface {mujoco.mjtGeom(kind).name}")
    return center - extent, center + extent


def collision_body_bounds(model):
    """Union original collision surfaces per body, including explicit-pair membership."""
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    active[model.pair_geom1] = True
    active[model.pair_geom2] = True
    result = {}
    for geom_id in np.flatnonzero(active):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        body = model.body(body_id).name
        lower, upper = geom_body_bounds(model, int(geom_id))
        if body not in result:
            result[body] = {"lower": lower, "upper": upper, "source_geoms": []}
        else:
            result[body]["lower"] = np.minimum(result[body]["lower"], lower)
            result[body]["upper"] = np.maximum(result[body]["upper"], upper)
        result[body]["source_geoms"].append({"id": int(geom_id), "name": model.geom(int(geom_id)).name,
                                             "type": mujoco.mjtGeom(int(model.geom_type[geom_id])).name})
    return result


def assert_biology_unchanged(reference, task, *, atol=1e-10):
    """Fail on changes to compiled mass, kinematics, routing or actuator parameters."""
    for name in ("nq", "nv", "nu", "na", "nbody", "njnt", "neq", "nsite", "ntendon", "nwrap"):
        if getattr(reference, name) != getattr(task, name):
            raise ValueError(f"biological dimension changed: {name}")
    for name in BIOLOGY_FIELDS:
        a, b = getattr(reference, name), getattr(task, name)
        # These must be bit-identical: a geometry rewrite must never recompute human mass.
        equal = np.array_equal(a, b) if name in ("body_mass", "body_inertia") or a.dtype.kind in "biu" else np.allclose(a, b, rtol=0, atol=atol)
        if not equal:
            raise ValueError(f"biological model field changed: {name}")


def _nominal_stand(model, clearance):
    root = model.joint("root").id
    if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("official MyoLeg26 task must retain its free root")
    adr = int(model.jnt_qposadr[root])
    qpos = model.qpos0.copy()
    yaw = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    rotated = np.empty(4)
    mujoco.mju_mulQuat(rotated, yaw, qpos[adr + 3:adr + 7])
    qpos[adr + 3:adr + 7] = rotated
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    lowest = min(data.geom_xpos[model.geom(COLLISION_PREFIX + body).id, 2]
                 - np.abs(data.geom_xmat[model.geom(COLLISION_PREFIX + body).id].reshape(3, 3)[2])
                 @ model.geom_size[model.geom(COLLISION_PREFIX + body).id] for body in FOOT_BODIES)
    qpos[adr + 2] += clearance - lowest
    return qpos, {"root_yaw_radians": float(np.pi / 2), "root_vertical_translation_m": float(clearance - lowest),
                  "minimum_foot_clearance_m": clearance, "meaning": "nominal reference pose, not a demonstrated equilibrium"}


def build_flat_ground_task(reference_spec, *, clearance=0.001):
    """Return a copied MjSpec plus an explicit record of this collision/task approximation."""
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("clearance must be finite and nonnegative")
    reference = reference_spec.compile()
    bounds = collision_body_bounds(reference)
    if not set(FOOT_BODIES).issubset(bounds):
        raise ValueError("reference must have separately collidable calcaneus and toe bodies")
    task = reference_spec.copy()
    removed_world_geoms = [g.name for g in task.worldbody.geoms]
    for pair in list(task.pairs):
        task.delete(pair)
    for geom in list(task.worldbody.geoms):
        task.delete(geom)
    for geom in task.geoms:
        geom.contype, geom.conaffinity = 0, 0
    proxies = []
    for body, measurements in bounds.items():
        lower, upper = measurements["lower"], measurements["upper"]
        center, size = (lower + upper) / 2, (upper - lower) / 2
        if np.any(size <= 0):
            raise ValueError(f"degenerate collision bounds for {body}")
        task.body(body).add_geom(name=COLLISION_PREFIX + body, type=mujoco.mjtGeom.mjGEOM_BOX,
                                 pos=center, size=size, mass=0, contype=2, conaffinity=1, condim=3,
                                 friction=[1.0, 0.005, 0.0001], solref=[0.02, 1.0], margin=0.001,
                                 group=4, rgba=[0.2, 0.6, 0.9, 0.25])
        proxies.append({"body": body, "geom": COLLISION_PREFIX + body, "position": center.tolist(),
                        "half_size": size.tolist(), "source_geoms": measurements["source_geoms"],
                        "foot": body in FOOT_BODIES})
    task.worldbody.add_geom(name=GROUND_NAME, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[7, 7, 0.1],
                            pos=[0, 0, 0], mass=0, contype=1, conaffinity=2, condim=3,
                            friction=[1.0, 0.005, 0.0001], solref=[0.02, 1.0], margin=0.001,
                            group=1, rgba=[0.8, 0.85, 0.9, 1])
    task.option.timestep = 0.002
    task.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    task.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    task.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    task.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    task_model = task.compile()
    assert_biology_unchanged(reference, task_model)
    qpos, reset = _nominal_stand(task_model, clearance)
    if task.key("stand") is not None:
        raise ValueError("reference already has a stand key; refusing to replace it")
    task.add_key(name="stand", qpos=qpos, qvel=np.zeros(task_model.nv), act=np.zeros(task_model.na), ctrl=np.zeros(task_model.nu))
    return task, {"name": "flat_ground_body_boxes_no_self_contact", "collision_fit": "body-frame AABB union of originally collidable source surfaces",
                  "approximation": "conservative boxes; not an anatomically validated collision envelope",
                  "removed_world_geoms": removed_world_geoms, "removed_explicit_pairs": reference.npair,
                  "proxies": proxies, "foot_bodies": list(FOOT_BODIES), "nominal_reset": reset,
                  "adjoint_status": "unvalidated; free-root and tendon gradient gates remain required"}


def preserve_site_precision(xml, spec):
    """Avoid MjSpec's six-digit position/orientation rounding on tendon/sensor sites."""
    root = ET.fromstring(xml)
    sites = list(root.find("worldbody").iter("site"))
    model = spec.compile()
    if len(sites) != len(spec.sites):
        raise ValueError("site order/count mismatch during precision-preserving export")
    for index, (node, site) in enumerate(zip(sites, spec.sites)):
        if node.get("name", "") != site.name:
            raise ValueError("site names out of order during export")
        if not np.allclose(site.pos, model.site_pos[index], rtol=0, atol=1e-12) or "fromto" in node.attrib:
            raise ValueError("site uses a transformed frame/fromto; precision export needs explicit handling")
        node.set("pos", " ".join(format(float(v), ".17g") for v in site.pos))
        for orientation in ("axisangle", "euler", "xyaxes", "zaxis"):
            node.attrib.pop(orientation, None)
        node.set("quat", " ".join(format(float(v), ".17g") for v in model.site_quat[index]))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"
