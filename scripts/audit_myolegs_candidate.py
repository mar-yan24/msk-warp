"""Offline source/provenance/inventory audit of the separate 80-muscle `myolegs` candidate.

Compiles the candidate natively on CPU from the pinned upstream checkout and records a
provenance + topology + parameter inventory beside the frozen MyoLeg26 reference. Nothing is
exported or vendored, no environment/registry/task is touched, and one JSON is written to a new
directory that must live under the repository's ignored ``logs/``.

Scope, stated so a later reader cannot mistake it: a successful compile means the XML assembles
into an ``MjModel``. It is **not** validated forward dynamics, not a Warp/gradient result and not
evidence of easier or faster learning. The standing objective (official MyoLeg26, 1.0 m/s walking,
no imitation) is unchanged; this is inventory only. The recorded geom and contact-pair parameters
are static model data: they say nothing about effective contact behaviour, which is UNVALIDATED.

Four deliberate restrictions:

* The candidate is ``load_spec("myolegs")`` -- the composed passive torso scaffold plus legs with a
  free root. ``myo_sim.FRAGMENT_SPEC_BUILDERS["myolegs"]`` is a *different* model (bare legs, no
  scaffold, no free root) and is never used here.
* The source pin is fixed. There is no flag and no keyword argument that accepts another commit.
* ``audit()`` must run in a fresh interpreter that has imported no warp, torch, mujoco_warp or
  ``msk_warp.envs``. The ban is enforced on entry and again before anything is written, so a
  contaminated process can never leave a complete-looking audit directory behind.
* Nothing is ever written to the pinned upstream checkout, to ``msk_warp/`` or outside ``logs/``.

The reused helpers come from ``scripts/build_myoleg26_assets.py``, which only pulls in mujoco and
numpy. No ``MjData``, no ``mj_forward``, no ``mj_step*``, no rollout, no training, no budget spend.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import-only reuse. `checked_source(source, expected_commit)` takes two positional arguments and
# compares the commit by exact string equality, so it is always given the full 40-character pin.
from scripts.build_myoleg26_assets import checked_source, sha256  # noqa: E402

#: Default root of the *write policy* -- the tree whose `logs/`, `msk_warp/` and pinned checkout
#: bound where `--out` may land. Separate from ROOT only so a test can police a throwaway tree; it
#: relaxes no gate, because every rule is expressed relative to whichever root is in force.
POLICY_ROOT = ROOT
ASSETS = ROOT / "msk_warp/assets/myoleg26"
PIN = "eb327acbae0fad12279495040607f5235d962328"
#: The canonical pinned checkout, relative to the repository root. `--out` is refused inside it
#: whatever `--source` names, so another clone can never be used to write into this tree.
PINNED_CHECKOUT = "logs/upstream_myo_sim_20260912"
CANDIDATE_NAME = "myolegs"
#: What the reused helper hard-codes for the *reduced* lineage; normalised away, never recorded.
LEGACY_LABEL = "myolegs26"
OUTPUT_NAME = "myolegs_candidate_audit.json"

SIGNATURE_FIELDS = ("nq", "nv", "nu", "na", "neq")
#: PREDICTIONS NOT MEASURED -- read off the raw upstream XML before any compile. A mismatch is
#: BLOCKED pending source/composition review; the audit never rewrites its own expectation.
CANDIDATE_SIGNATURE = (35, 34, 80, 80, 14)
CANDIDATE_NJNT = 29
BASELINE_SIGNATURE = (47, 46, 26, 26, 28)

BANNED_MODULES = ("warp", "torch", "mujoco_warp", "msk_warp.envs")
#: Only these recorded arrays are checked for finiteness. A legal infinity in an unrelated
#: optional field (forcerange, actrange, ...) must not trip the gate.
FINITE_FIELDS = ("body_mass", "body_pos", "body_quat", "qpos0", "jnt_pos", "jnt_axis", "jnt_range",
                 "actuator_ctrlrange", "actuator_lengthrange", "actuator_gainprm",
                 "actuator_biasprm", "actuator_dynprm", "eq_data", "tendon_range", "wrap_prm")

COUNT_FIELDS = ("nq", "nv", "nu", "na", "neq", "njnt", "nbody", "ngeom", "nsite", "ntendon",
                "nwrap", "npair", "nsensor", "nmesh", "nkey")
OPTION_FIELDS = ("timestep", "integrator", "solver", "jacobian", "cone", "impratio", "iterations",
                 "ls_iterations", "tolerance", "ls_tolerance", "gravity", "wind", "density",
                 "viscosity", "disableflags", "enableflags")
OPTION_ENUMS = {"integrator": mujoco.mjtIntegrator, "solver": mujoco.mjtSolver,
                "jacobian": mujoco.mjtJacobian, "cone": mujoco.mjtCone}
COMPILER_FIELDS = ("degree", "autolimits", "balanceinertia", "boundmass", "boundinertia",
                   "inertiafromgeom", "fusestatic", "discardvisual", "settotalmass", "alignfree",
                   "fitaabb", "saveinertial", "inertiagrouprange")

NAMED_BODIES = ("patella_l", "patella_r", "calcn_l", "calcn_r", "toes_l", "toes_r")
NAMED_JOINTS = ("subtalar_angle_l", "subtalar_angle_r")

# MuJoCo leaves `eq_objtype` at mjOBJ_UNKNOWN for joint, tendon and flex equalities, so the object
# table is resolved from `eq_type` instead of guessed. `eq_objtype` is preferred when it is set,
# because connect/weld may name either bodies or sites.
_EQUALITY_OBJECTS = {mujoco.mjtEq.mjEQ_JOINT: mujoco.mjtObj.mjOBJ_JOINT,
                     mujoco.mjtEq.mjEQ_TENDON: mujoco.mjtObj.mjOBJ_TENDON,
                     mujoco.mjtEq.mjEQ_CONNECT: mujoco.mjtObj.mjOBJ_BODY,
                     mujoco.mjtEq.mjEQ_WELD: mujoco.mjtObj.mjOBJ_BODY,
                     mujoco.mjtEq.mjEQ_FLEX: mujoco.mjtObj.mjOBJ_FLEX,
                     mujoco.mjtEq.mjEQ_FLEXVERT: mujoco.mjtObj.mjOBJ_FLEX,
                     mujoco.mjtEq.mjEQ_FLEXSTRAIN: mujoco.mjtObj.mjOBJ_FLEX,
                     mujoco.mjtEq.mjEQ_DISTANCE: mujoco.mjtObj.mjOBJ_GEOM}
_WRAP_OBJECTS = {mujoco.mjtWrap.mjWRAP_JOINT: mujoco.mjtObj.mjOBJ_JOINT,
                 mujoco.mjtWrap.mjWRAP_SITE: mujoco.mjtObj.mjOBJ_SITE,
                 mujoco.mjtWrap.mjWRAP_SPHERE: mujoco.mjtObj.mjOBJ_GEOM,
                 mujoco.mjtWrap.mjWRAP_CYLINDER: mujoco.mjtObj.mjOBJ_GEOM}
_TRANSMISSION_OBJECTS = {mujoco.mjtTrn.mjTRN_JOINT: mujoco.mjtObj.mjOBJ_JOINT,
                         mujoco.mjtTrn.mjTRN_JOINTINPARENT: mujoco.mjtObj.mjOBJ_JOINT,
                         mujoco.mjtTrn.mjTRN_TENDON: mujoco.mjtObj.mjOBJ_TENDON,
                         mujoco.mjtTrn.mjTRN_SITE: mujoco.mjtObj.mjOBJ_SITE,
                         mujoco.mjtTrn.mjTRN_SLIDERCRANK: mujoco.mjtObj.mjOBJ_SITE,
                         mujoco.mjtTrn.mjTRN_BODY: mujoco.mjtObj.mjOBJ_BODY}


# --- gates ---------------------------------------------------------------------------------------

def validated_out(out, source, root=None):
    """Resolve and police the destination. Runs before any directory or file is created.

    `root` is the repository root whose `logs/`, `msk_warp/` and pinned checkout define the write
    policy; tests point it at a throwaway tree. The pinned checkout is refused whatever `--source`
    names, so pointing `--source` at a second clone cannot unlock a write into it.
    """
    root = POLICY_ROOT if root is None else Path(root)
    out, root, source = Path(out).resolve(), Path(root).resolve(), Path(source).resolve()
    if out.is_relative_to(root / "msk_warp"):
        raise ValueError(f"--out must stay outside the msk_warp package tree: {out}")
    for forbidden in (source, (root / PINNED_CHECKOUT).resolve()):
        if out.is_relative_to(forbidden):
            raise ValueError(f"--out must stay outside the pinned upstream checkout {forbidden}: {out}")
    if not out.is_relative_to(root / "logs"):
        raise ValueError(f"--out must live under the ignored logs/ tree {root / 'logs'}: {out}")
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; use a new destination")
    return out


def candidate_source_record(source):
    """Verified pin metadata, with the helper's reduced-lineage label normalised to the candidate.

    The pin is not a parameter: a checkout at any other commit is refused, never accepted.
    """
    record = dict(checked_source(Path(source).resolve(), PIN))
    if record.get("registry_name") != LEGACY_LABEL:
        raise ValueError(f"unexpected helper registry label {record.get('registry_name')!r}; "
                         f"expected {LEGACY_LABEL!r} to normalise to {CANDIDATE_NAME!r}")
    record["registry_name"] = CANDIDATE_NAME
    record["composition_api"] = f"load_spec({CANDIDATE_NAME!r})"
    record["path"] = Path(source).resolve().as_posix()
    return record


def assert_candidate_source(record):
    """The candidate's source metadata must carry no surviving reduced-lineage label."""
    if record.get("registry_name") != CANDIDATE_NAME:
        raise ValueError(f"candidate registry_name is {record.get('registry_name')!r}, "
                         f"expected {CANDIDATE_NAME!r}")
    expected_api = f"load_spec({CANDIDATE_NAME!r})"
    if record.get("composition_api") != expected_api:
        raise ValueError(f"candidate composition_api is {record.get('composition_api')!r}, "
                         f"expected {expected_api!r}")
    leaked = sorted(str(key) for key, value in record.items()
                    if LEGACY_LABEL in str(key) or LEGACY_LABEL in str(value))
    if leaked:
        raise ValueError(f"{LEGACY_LABEL!r} survives normalisation in candidate source metadata: {leaked}")


def checked_mujoco_version(actual, expected):
    if actual != expected:
        raise ValueError(f"BLOCKED: MuJoCo {actual} differs from the pinned manifest version {expected}")
    return actual


def checked_signature(label, model, expected):
    actual = tuple(int(getattr(model, name)) for name in SIGNATURE_FIELDS)
    if actual != tuple(expected):
        raise ValueError(f"BLOCKED: {label} signature {dict(zip(SIGNATURE_FIELDS, actual))} differs "
                         f"from the preregistered {dict(zip(SIGNATURE_FIELDS, tuple(expected)))}; "
                         "source/composition review required")
    return list(actual)


def assert_recorded_values_valid(label, model):
    for name in FINITE_FIELDS:
        values = np.asarray(getattr(model, name), dtype=float)
        if values.size and not np.all(np.isfinite(values)):
            raise ValueError(f"BLOCKED: {label} {name} holds non-finite values")
    if model.nu:
        low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        bad = sorted(int(i) for i in np.nonzero(~(low <= high))[0])
        if bad:
            raise ValueError(f"BLOCKED: {label} actuator_ctrlrange is unordered at actuators {bad}")


def banned_modules_present():
    return sorted(name for name in BANNED_MODULES if name in sys.modules)


def assert_no_banned_imports():
    present = banned_modules_present()
    if present:
        raise RuntimeError(f"the audit process imported banned modules {present}; "
                           "run it in a fresh interpreter with no warp/torch/env import")


def import_pinned_myo_sim(source):
    """Import `myo_sim` from the verified checkout, leaving no trace behind on failure."""
    source = Path(source).resolve()
    if "myo_sim" in sys.modules:
        raise RuntimeError("myo_sim already imported; run the audit in a fresh Python process")
    sys.path.insert(0, str(source))
    try:
        upstream = importlib.import_module("myo_sim")
        origin = getattr(upstream, "__file__", None)
        if origin is None:
            raise RuntimeError("myo_sim resolved as a namespace package, not the verified checkout")
        resolved = Path(origin).resolve()
        if not resolved.is_relative_to(source):
            raise RuntimeError(f"myo_sim import resolved outside the verified checkout: {resolved}")
    except BaseException:
        for name in [n for n in sys.modules if n == "myo_sim" or n.startswith("myo_sim.")]:
            del sys.modules[name]
        raise
    finally:
        if sys.path and sys.path[0] == str(source):
            sys.path.pop(0)
    return upstream


# --- inventory -----------------------------------------------------------------------------------

def _plain(value):
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _name(model, objtype, objid):
    if int(objid) < 0:
        return None
    return mujoco.mj_id2name(model, int(objtype), int(objid))


def _joints(model):
    return [{"name": model.joint(i).name,
             "type": mujoco.mjtJoint(int(model.jnt_type[i])).name,
             "body": model.body(int(model.jnt_bodyid[i])).name,
             "axis": _plain(model.jnt_axis[i]),
             "pos": _plain(model.jnt_pos[i]),
             "range": _plain(model.jnt_range[i]),
             "limited": bool(model.jnt_limited[i]),
             "qposadr": int(model.jnt_qposadr[i]),
             "dofadr": int(model.jnt_dofadr[i])} for i in range(model.njnt)]


def _equality_objtype(model, i):
    declared = mujoco.mjtObj(int(model.eq_objtype[i]))
    if declared != mujoco.mjtObj.mjOBJ_UNKNOWN:
        return declared
    kind = mujoco.mjtEq(int(model.eq_type[i]))
    if kind not in _EQUALITY_OBJECTS:
        raise ValueError(f"BLOCKED: equality type {kind.name} has no recorded object table")
    return _EQUALITY_OBJECTS[kind]


def _equality(model):
    entries = []
    for i in range(model.neq):
        objtype = _equality_objtype(model, i)
        # Raw ids are always recorded: an endpoint may be unnamed, and -1 is the "absent" sentinel.
        entries.append({"name": model.equality(i).name,
                        "type": mujoco.mjtEq(int(model.eq_type[i])).name,
                        "objtype": objtype.name,
                        "obj1_id": int(model.eq_obj1id[i]),
                        "obj1": _name(model, objtype, model.eq_obj1id[i]),
                        "obj2_id": int(model.eq_obj2id[i]),
                        "obj2": _name(model, objtype, model.eq_obj2id[i]),
                        "active0": bool(model.eq_active0[i]),
                        "data": _plain(model.eq_data[i])})
    return entries


def _tendons(model):
    tendons = []
    for i in range(model.ntendon):
        start, count = int(model.tendon_adr[i]), int(model.tendon_num[i])
        wraps = []
        for j in range(start, start + count):
            kind = mujoco.mjtWrap(int(model.wrap_type[j]))
            wraps.append({"type": kind.name,
                          "obj": _name(model, _WRAP_OBJECTS[kind], model.wrap_objid[j])
                                 if kind in _WRAP_OBJECTS else None,
                          "prm": float(model.wrap_prm[j])})
        # A fixed tendon is the only kind that wraps joints; everything else is spatial.
        fixed = any(w["type"] == "mjWRAP_JOINT" for w in wraps)
        tendons.append({"name": model.tendon(i).name,
                        "kind": "fixed" if fixed else "spatial",
                        "limited": bool(model.tendon_limited[i]),
                        "range": _plain(model.tendon_range[i]),
                        "width": float(model.tendon_width[i]),
                        "wraps": wraps})
    return tendons


def _geoms(model):
    """Per-geom identity and static collision parameters, so pairs can be interpreted later.

    These are model inputs, not behaviour: nothing here establishes effective contact response.
    """
    return [{"id": i,
             "name": model.geom(i).name,
             "type": mujoco.mjtGeom(int(model.geom_type[i])).name,
             "body": model.body(int(model.geom_bodyid[i])).name,
             "contype": int(model.geom_contype[i]),
             "conaffinity": int(model.geom_conaffinity[i]),
             "condim": int(model.geom_condim[i]),
             "friction": _plain(model.geom_friction[i])} for i in range(model.ngeom)]


def _contact_pairs(model):
    pairs = []
    for i in range(model.npair):
        first, second = int(model.pair_geom1[i]), int(model.pair_geom2[i])
        pairs.append({"geom1_id": first, "geom1": model.geom(first).name,
                      "body1": model.body(int(model.geom_bodyid[first])).name,
                      "geom2_id": second, "geom2": model.geom(second).name,
                      "body2": model.body(int(model.geom_bodyid[second])).name,
                      "condim": int(model.pair_dim[i])})
    return pairs


def _actuators(model):
    actuators = []
    for i in range(model.nu):
        transmission = mujoco.mjtTrn(int(model.actuator_trntype[i]))
        actuators.append({"name": model.actuator(i).name,
                          "trntype": transmission.name,
                          "target": _name(model, _TRANSMISSION_OBJECTS[transmission],
                                          model.actuator_trnid[i, 0])
                                    if transmission in _TRANSMISSION_OBJECTS else None,
                          "dyntype": mujoco.mjtDyn(int(model.actuator_dyntype[i])).name,
                          "gaintype": mujoco.mjtGain(int(model.actuator_gaintype[i])).name,
                          "biastype": mujoco.mjtBias(int(model.actuator_biastype[i])).name,
                          "ctrllimited": bool(model.actuator_ctrllimited[i]),
                          "ctrlrange": _plain(model.actuator_ctrlrange[i]),
                          "lengthrange": _plain(model.actuator_lengthrange[i]),
                          "gainprm": _plain(model.actuator_gainprm[i]),
                          "biasprm": _plain(model.actuator_biasprm[i]),
                          "dynprm": _plain(model.actuator_dynprm[i])})
    return actuators


def _sensors(model):
    return [{"name": model.sensor(i).name,
             "type": mujoco.mjtSensor(int(model.sensor_type[i])).name,
             "objtype": mujoco.mjtObj(int(model.sensor_objtype[i])).name,
             "obj": _name(model, model.sensor_objtype[i], model.sensor_objid[i]),
             "dim": int(model.sensor_dim[i])} for i in range(model.nsensor)]


def _option(model):
    option = {name: _plain(getattr(model.opt, name)) for name in OPTION_FIELDS}
    for name, enum in OPTION_ENUMS.items():
        option[name + "_name"] = enum(int(getattr(model.opt, name))).name
    return option


def compiler_settings(spec):
    """Compiler directives of an *uncompiled* spec. Both models are read at the same stage."""
    return {name: _plain(getattr(spec.compiler, name)) for name in COMPILER_FIELDS}


def inventory(model, label, compiler=None):
    """Descriptive, measured inventory of one compiled model. No guessed or defaulted values."""
    bodies = [model.body(i).name for i in range(model.nbody)]
    joints = _joints(model)
    free = [j for j in joints if j["type"] == "mjJNT_FREE"]
    pelvis = None
    if "pelvis" in bodies:
        body = model.body("pelvis")
        pelvis = {"id": int(body.id), "pos": _plain(body.pos), "quat": _plain(body.quat),
                  "mass": float(body.mass[0]), "parent": bodies[int(body.parentid[0])]}
    joint_names = {j["name"] for j in joints}
    return {"label": label,
            "counts": {name: int(getattr(model, name)) for name in COUNT_FIELDS},
            "signature": [int(getattr(model, name)) for name in SIGNATURE_FIELDS],
            "total_body_mass_kg": float(model.body_mass.sum()),
            "bodies": bodies,
            "joints": joints,
            "free_joints": [j["name"] for j in free],
            "free_root": {"joint": free[0]["name"], "body": free[0]["body"]} if len(free) == 1 else None,
            "pelvis": pelvis,
            "qpos0": _plain(model.qpos0),
            "equality": _equality(model),
            "tendons": _tendons(model),
            "geoms": _geoms(model),
            "contact_pairs": _contact_pairs(model),
            "actuators": _actuators(model),
            "sensors": _sensors(model),
            "option": _option(model),
            "compiler": compiler,
            "named_features": {**{name: name in set(bodies) for name in NAMED_BODIES},
                               **{name: name in joint_names for name in NAMED_JOINTS}}}


# --- audit ---------------------------------------------------------------------------------------

def audit(source, out, root=None):
    """Run the whole audit in a fresh interpreter and write exactly one JSON, last.

    Every gate precedes the single write, including both banned-import checks, so a failed run
    never leaves a directory or a complete-looking artifact behind.
    """
    assert_no_banned_imports()
    source = Path(source).resolve()
    out = validated_out(out, source, root=root)

    source_record = candidate_source_record(source)
    assert_candidate_source(source_record)

    upstream = import_pinned_myo_sim(source)
    source_record["myo_sim_file"] = Path(upstream.__file__).resolve().as_posix()
    assert_candidate_source(source_record)

    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    expected_version = checked_mujoco_version(mujoco.__version__, manifest["mujoco_version"])

    candidate_spec = upstream.load_spec(CANDIDATE_NAME)
    # Read both compiler blocks from uncompiled specs so the two are directly comparable.
    candidate_compiler = compiler_settings(candidate_spec)
    candidate = candidate_spec.compile()
    reference_xml = ASSETS / "reference.xml"
    baseline_compiler = compiler_settings(mujoco.MjSpec.from_file(str(reference_xml)))
    baseline = mujoco.MjModel.from_xml_path(str(reference_xml))

    checked_signature("candidate", candidate, CANDIDATE_SIGNATURE)
    if int(candidate.njnt) != CANDIDATE_NJNT:
        raise ValueError(f"BLOCKED: candidate njnt {int(candidate.njnt)} differs from the "
                         f"preregistered {CANDIDATE_NJNT}; source/composition review required")
    free = [i for i in range(candidate.njnt) if candidate.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError(f"BLOCKED: candidate has {len(free)} free joints, expected exactly one root")
    checked_signature("baseline", baseline, BASELINE_SIGNATURE)
    assert_recorded_values_valid("candidate", candidate)
    assert_recorded_values_valid("baseline", baseline)

    record = {
        "schema_version": 2,
        "audit": "myolegs-candidate-native-cpu-inventory",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "scope": ("Native CPU source/compile inventory only. A successful compile means the XML "
                  "assembles into an MjModel; it is not validated forward dynamics, not a Warp or "
                  "gradient result and not evidence of easier or faster learning. Geom and "
                  "contact-pair parameters are static model inputs, not contact behaviour."),
        "provenance": {
            "script": _HERE.relative_to(ROOT).as_posix(),
            "script_sha256": sha256(_HERE),
            "source": source_record,
            "baseline_reference_xml": {"path": reference_xml.relative_to(ROOT).as_posix(),
                                       "sha256": sha256(reference_xml)},
            "versions": {"python": sys.version.split()[0], "mujoco": mujoco.__version__,
                         "numpy": np.__version__},
            "expected_mujoco_version": expected_version,
            "banned_modules": list(BANNED_MODULES),
            "banned_modules_present": banned_modules_present(),
            "checked_finite_fields": list(FINITE_FIELDS),
        },
        "expectations": {
            "status": "PREDICTIONS NOT MEASURED",
            "note": ("Read off the raw upstream XML before any compile and compared at runtime. "
                     "A mismatch is BLOCKED pending source/composition review; the audit never "
                     "rewrites this expectation."),
            "signature_fields": list(SIGNATURE_FIELDS),
            "candidate": {"signature": list(CANDIDATE_SIGNATURE), "njnt": CANDIDATE_NJNT,
                          "free_root": True},
            "baseline": {"signature": list(BASELINE_SIGNATURE)},
        },
        "measured": {"candidate": inventory(candidate, CANDIDATE_NAME, candidate_compiler),
                     "baseline": inventory(baseline, "myoleg26_reference", baseline_compiler)},
        "status": {"dynamics": "UNVALIDATED", "warp_support": "UNVALIDATED",
                   "gradients": "UNVALIDATED", "learnability": "UNVALIDATED",
                   "contact_behaviour": "UNVALIDATED", "task_version": None, "vendored": False,
                   "remote_delta": "not verified",
                   "license_notice": "unresolved - blocks later redistribution only",
                   "successor_work": "not authorised"},
    }
    assert_no_banned_imports()
    out.mkdir(parents=True, exist_ok=False)
    (out / OUTPUT_NAME).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8", newline="\n")
    return record


def _reason(error):
    """One readable line, keeping a subprocess's own stderr (git's, typically) when there is one."""
    text = f"{type(error).__name__}: {error}"
    captured = getattr(error, "stderr", None)
    if isinstance(captured, str) and captured.strip():
        text += "\n" + captured.strip()
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        record = audit(args.source, args.out)
    except Exception as error:
        raise SystemExit(_reason(error))
    candidate = record["measured"]["candidate"]["counts"]
    print(f"Audited {CANDIDATE_NAME} candidate "
          f"(nq={candidate['nq']} nv={candidate['nv']} nu={candidate['nu']} "
          f"na={candidate['na']} neq={candidate['neq']} njnt={candidate['njnt']}) into {args.out}")


if __name__ == "__main__":
    main()
