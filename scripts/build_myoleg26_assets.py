"""Export the pinned official MyoLeg26 reference and explicit flat-ground task.

Uses upstream's canonical composition and XML sanitizer. No upstream installation or
source changes are needed. Output directories must be new; existing artifacts are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from msk_warp.models.myoleg26 import (
    BIOLOGY_FIELDS, COLLISION_PREFIX, assert_biology_unchanged,
    build_flat_ground_task, preserve_site_precision,
)

PIN = "eb327acbae0fad12279495040607f5235d962328"
EXPORT_FIELDS = (
    "geom_type", "geom_bodyid", "geom_dataid", "geom_pos", "geom_quat", "geom_size", "geom_friction",
    "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "geom_condim", "geom_priority",
    "geom_solmix", "geom_contype", "geom_conaffinity", "pair_geom1", "pair_geom2", "pair_dim",
    "pair_friction", "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap",
    "key_time", "key_qpos", "key_qvel", "key_act", "key_ctrl",
)
OPTION_FIELDS = ("timestep", "solver", "integrator", "jacobian", "iterations", "ls_iterations",
                 "tolerance", "ls_tolerance", "gravity", "disableflags", "enableflags", "cone", "impratio")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_source(source, expected_commit):
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], text=True, capture_output=True, check=True).stdout.strip()
    head, status, remote = git("rev-parse", "HEAD"), git("status", "--porcelain"), git("remote", "get-url", "origin")
    if head != expected_commit:
        raise ValueError(f"source HEAD {head} differs from required {expected_commit}")
    if status:
        raise ValueError("source checkout is dirty; export requires a clean pinned upstream tree")
    if remote.removesuffix(".git").lower() != "https://github.com/myohub/myo_sim":
        raise ValueError(f"unexpected source origin: {remote}")
    return {"repository": remote, "commit": head, "clean": True, "registry_name": "myolegs26"}


def _numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def _orientation(node, item):
    names = ("quat", "axisangle", "xyaxes", "zaxis", "euler")
    for name in names:
        node.attrib.pop(name, None)
    name = names[int(item.alt.type)]
    node.set(name, _numbers(item.quat if name == "quat" else getattr(item.alt, name)))


def _preserve_source_transforms(root, spec):
    """Keep source-local coordinates, including alternative orientation encodings."""
    bodies = list(root.find("worldbody").iter("body"))
    geoms = list(root.find("worldbody").iter("geom"))
    if len(bodies) != len(spec.bodies) - 1 or len(geoms) != len(spec.geoms):
        raise ValueError("body/geom order mismatch during export")
    for node, body in zip(bodies, list(spec.bodies)[1:]):
        if node.get("name", "") != body.name:
            raise ValueError("body names out of order during export")
        node.set("pos", _numbers(body.pos))
        _orientation(node, body)
    for node, geom in zip(geoms, spec.geoms):
        if node.get("name", "") != geom.name:
            raise ValueError("geom names out of order during export")
        if np.all(np.isfinite(geom.fromto)):
            for attr in ("pos", "quat", "axisangle", "xyaxes", "zaxis", "euler"):
                node.attrib.pop(attr, None)
            node.set("fromto", _numbers(geom.fromto))
        else:
            node.set("pos", _numbers(geom.pos))
            _orientation(node, geom)
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            node.set("size", _numbers(geom.size))
        for name in ("friction", "solref", "solimp"):
            node.set(name, _numbers(getattr(geom, name)))
        node.set("margin", format(float(geom.margin), ".17g"))


def assert_export_equivalent(expected, reloaded):
    """Standalone XML must preserve reference/task contacts, geometry, options and reset."""
    assert_biology_unchanged(expected, reloaded)
    for name in EXPORT_FIELDS:
        a, b = getattr(expected, name), getattr(reloaded, name)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-10):
            raise ValueError(f"export changed {name}")
    for name in OPTION_FIELDS:
        if not np.allclose(getattr(expected.opt, name), getattr(reloaded.opt, name), rtol=0, atol=1e-14):
            raise ValueError(f"export changed option {name}")


def standalone_xml(spec, sanitize, source, output, assets):
    model = spec.compile()
    root = ET.fromstring(preserve_site_precision(sanitize(spec.to_xml(), model=model), spec))
    _preserve_source_transforms(root, spec)
    compiler = root.find("compiler")
    for node in root.find("asset"):
        for attr in list(node.attrib):
            if not attr.startswith("file"):
                continue
            base = compiler.get("texturedir" if node.tag == "texture" else "meshdir", ".")
            original = (Path(base) / node.get(attr)).resolve()
            if not original.is_relative_to(source):
                raise ValueError(f"asset escapes pinned source: {original}")
            folder = "textures" if node.tag == "texture" else "meshes"
            destination = f"{folder}/{original.name}"
            digest = sha256(original)
            if destination in assets and assets[destination]["sha256"] != digest:
                destination = f"{folder}/{digest[:12]}_{original.name}"
            if destination not in assets:
                path = output / destination
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, path)
                assets[destination] = {"source": original.relative_to(source).as_posix(), "sha256": digest}
            node.set(attr, destination)
    compiler.set("meshdir", ".")
    compiler.set("texturedir", ".")
    # Keep the measured boxes and nominal clearance at their computed precision too.
    for node in root.find("worldbody").iter("geom"):
        name = node.get("name", "")
        if name.startswith(COLLISION_PREFIX):
            geom = spec.geom(name)
            node.set("pos", " ".join(format(float(v), ".17g") for v in geom.pos))
            node.set("size", " ".join(format(float(v), ".17g") for v in geom.size))
    keys = root.find("keyframe")
    if keys is not None:
        for node in keys:
            key = spec.key(node.get("name"))
            for attr in ("qpos", "qvel", "act", "ctrl"):
                values = getattr(key, attr)
                if len(values):
                    node.set(attr, " ".join(format(float(v), ".17g") for v in values))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def export_assets(source, output, expected_commit=PIN):
    source, output = Path(source).resolve(), Path(output).resolve()
    source_record = checked_source(source, expected_commit)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}; use a new destination")
    if "myo_sim" in sys.modules:
        raise RuntimeError("myo_sim already imported; run the exporter in a fresh Python process")
    sys.path.insert(0, str(source))
    upstream = importlib.import_module("myo_sim")
    if not Path(upstream.__file__).resolve().is_relative_to(source):
        raise RuntimeError("canonical myo_sim import did not resolve to the verified checkout")
    sanitize = importlib.import_module("myo_sim.build.compose").sanitize_spec_xml
    reference_spec = upstream.load_spec("myolegs26")
    reference = reference_spec.compile()
    if (reference.nq, reference.nv, reference.nu, reference.na, reference.neq) != (47, 46, 26, 26, 28):
        raise ValueError("pinned canonical model signature changed")
    task_spec, task_record = build_flat_ground_task(reference_spec)
    task = task_spec.compile()
    output.mkdir(parents=True, exist_ok=False)
    assets = {}
    for name, spec, model in (("reference.xml", reference_spec, reference), ("flat_boxes.xml", task_spec, task)):
        (output / name).write_text(standalone_xml(spec, sanitize, source, output, assets), encoding="utf-8", newline="\n")
        reloaded = mujoco.MjModel.from_xml_path(str(output / name))
        assert_biology_unchanged(reference, reloaded)
        assert_export_equivalent(model, reloaded)
    (output / "LICENSE.upstream").write_text((source / "LICENSE").read_text(encoding="utf-8"),
                                             encoding="utf-8", newline="\n")
    # Canonical serialization removes XML comments. Preserve their original credit notices.
    notice_files = [
        "leg/assets/myolegs26_assets.xml", "leg/assets/myolegs26_chain.xml",
        "leg/assets/myolegs26_muscle.xml", "leg/assets/myolegs26_tendon.xml",
        "torso/assets/myotorso_assets.xml", "torso/assets/myotorso_chain.xml",
        "head/assets/myohead_simple_assets.xml", "head/assets/myohead_rigid_chain.xml",
        "scene/myosuite_scene.xml",
    ]
    notices = [f"Official MyoHub/myo_sim model, commit {expected_commit}.\n"
               "Original source notices follow verbatim. Reduced leg lineage is CC-BY 3.0; "
               "upstream package/component notices also specify Apache 2.0.\n"
               "flat_boxes.xml is a derived task collision approximation; reference.xml preserves the upstream assembly.\n"]
    source_inputs = {}
    for relative in notice_files:
        path = source / "myo_sim/models" / relative
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.fromstring(path.read_text(encoding="utf-8"), parser=parser)
        comments = [node.text or "" for node in tree.iter() if node.tag is ET.Comment]
        notices.append(relative + "\n" + "\n".join(comments) + "\n")
        source_inputs[path.relative_to(source).as_posix()] = sha256(path)
    for relative in ("myo_sim/__init__.py", "myo_sim/build/compose.py", "myo_sim/build/utils.py",
                     "myo_sim/models/contacts/myolegs26_contacts.xml", "myo_sim/models/sensors/myolegs26_sensors.xml"):
        source_inputs[relative] = sha256(source / relative)
    (output / "SOURCE_NOTICES.txt").write_text("\n".join(notices), encoding="utf-8", newline="\n")
    files = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = {"schema_version": 1, "source": source_record, "source_inputs": source_inputs,
                "mujoco_version": mujoco.__version__, "assets": assets, "files": files, "task": task_record,
                "dimensions": {n: int(getattr(reference, n)) for n in ("nq", "nv", "nu", "na", "neq", "nbody")},
                "reference_total_mass_kg": float(reference.body_mass.sum()), "invariant_fields": list(BIOLOGY_FIELDS),
                "model_options": {label: {name: np.asarray(getattr(model.opt, name)).tolist() for name in OPTION_FIELDS}
                                  for label, model in (("reference", reference), ("task", task))},
                "export_equivalence_fields": list(EXPORT_FIELDS),
                "export_adjustments": ["upstream sanitize_spec_xml for nested defaults and collision-mask preservation",
                                       "full-precision body/geom/site positions/orientations, task boxes and keyframes to avoid serializer rounding"],
                "builder_sha256": {"scripts/build_myoleg26_assets.py": sha256(__file__),
                                   "msk_warp/models/myoleg26.py": sha256(Path(__file__).parents[1] / "msk_warp/models/myoleg26.py")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-commit", default=PIN)
    args = parser.parse_args()
    manifest = export_assets(args.source, args.out, args.expected_commit)
    print(f"Exported reference and {len(manifest['task']['proxies'])}-box task to {args.out}")


if __name__ == "__main__":
    main()
