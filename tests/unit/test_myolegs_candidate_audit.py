"""Gates of the offline 80-muscle `myolegs` candidate source/compile inventory.

Fully hermetic. Every test uses a tiny synthetic MJCF, a controlled stub source package, a
throwaway git repository or the committed `msk_warp/assets/myoleg26` baseline. **No test depends
on the gitignored upstream checkout**, so a fresh clone runs this suite unchanged; a guard test
enforces that. Real-upstream coverage lives outside the unit suite, in the recorded fresh-child
audit runs under `logs/`.

`audit()` refuses to run in a process that has imported warp, torch, mujoco_warp or
`msk_warp.envs`, and this pytest process has all four. Every end-to-end audit test therefore runs
in a fresh child, which also lets the child arm its own `MjData`/`mj_kinematics`/`mj_step*`/
`mj_forward` tripwires before the source loader is reached. Pure helpers are exercised in process.
"""

import inspect
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from scripts import audit_myolegs_candidate as A

ROOT = Path(__file__).resolve().parents[2]
CHILD_TIMEOUT = 60
#: Armed in every child before the source loader runs. `MjData` and `mj_kinematics` turn the
#: "no state is ever constructed" claim into an enforced one.
TRIPWIRE_APIS = ("MjData", "mj_kinematics", "mj_forward", "mj_step", "mj_step1", "mj_step2")
#: Split so that the portability guard below does not match its own source text.
IGNORED_CHECKOUT = "upstream_" + "myo_sim_20260912"
_MARK = "<<<JSON"
_MARK_END = "JSON>>>"

# A free root, a limited hinge, a second hinge, a floor plane, two masked limb geoms, a spatial
# tendon, a named single-endpoint equality, an unnamed two-endpoint equality, an explicit
# self-contact pair and a sensor: every inventory branch, none of the real anatomy.
PROBE_XML = """<mujoco model="audit_probe">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="1"
          friction="1 0.005 0.0001"/>
    <body name="pelvis">
      <freejoint name="root"/>
      <geom name="pelvis_geom" type="sphere" size="0.1" contype="2" conaffinity="2"/>
      <site name="origin"/>
      <body name="calcn_l" pos="0 0 -0.4">
        <joint name="subtalar_angle_l" type="hinge" axis="0 1 0" range="-0.349 0.349"/>
        <geom name="calcn_geom" type="box" size="0.05 0.03 0.02" contype="2" conaffinity="2"/>
        <site name="insertion"/>
        <body name="toes_l" pos="0 0 -0.05">
          <joint name="mtp_angle_l" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
          <geom name="toes_geom" type="box" size="0.02 0.02 0.01" contype="2" conaffinity="2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_probe"><site site="origin"/><site site="insertion"/></spatial>
  </tendon>
  <equality>
    <joint name="eq_probe" joint1="subtalar_angle_l" polycoef="0 1 0 0 0"/>
    <joint joint1="subtalar_angle_l" joint2="mtp_angle_l" polycoef="0 1 0 0 0"/>
  </equality>
  <actuator>
    <general name="a_probe" tendon="t_probe" ctrlrange="0 1" ctrllimited="true"/>
  </actuator>
  <sensor><jointpos name="s_probe" joint="subtalar_angle_l"/></sensor>
  <contact><pair name="p_probe" geom1="pelvis_geom" geom2="calcn_geom" condim="3"/></contact>
</mujoco>"""

# Two free joints: the only runtime discriminator between the composed `load_spec("myolegs")`
# model and the bare fragment builder is that the composed one has exactly one.
TWO_ROOT_XML = """<mujoco model="two_roots">
  <worldbody>
    <body name="a"><freejoint name="root"/><geom name="ga" size="0.1"/></body>
    <body name="b"><freejoint name="second_root"/><geom name="gb" size="0.1"/></body>
  </worldbody>
</mujoco>"""

STUB_INIT = '''"""Controlled stand-in for the pinned upstream package, used by the unit suite."""
import sys
import types

import mujoco

CANDIDATE_XML = {xml!r}
CONTAMINATE = {contaminate!r}
POKE = {poke!r}


class _PokedSpec:
    """Hands back a model the upstream compile would have produced, with one value spoiled."""

    def __init__(self, spec):
        self._spec = spec
        self.compiler = spec.compiler

    def compile(self):
        model = self._spec.compile()
        if POKE == "nonfinite":
            model.body_mass[1] = float("nan")
        elif POKE == "ctrlrange":
            model.actuator_ctrlrange[0] = [1.0, 0.0]
        return model


def load_spec(name):
    if name != "myolegs":
        raise ValueError("unexpected model name: " + name)
    if CONTAMINATE:
        sys.modules.setdefault("torch", types.ModuleType("torch"))
    spec = mujoco.MjSpec.from_string(CANDIDATE_XML)
    return _PokedSpec(spec) if POKE else spec
'''

DRIVER = '''import json
import sys
import types
from pathlib import Path

sys.path.insert(0, {root!r})
import mujoco
from scripts import audit_myolegs_candidate as A


def _tripwire(name):
    def boom(*args, **kwargs):
        raise AssertionError("the audit reached " + name)
    return boom


for _api in {tripwires!r}:
    setattr(mujoco, _api, _tripwire(_api))

if {stub_checked_source!r}:
    A.checked_source = lambda source, expected: dict(
        repository="https://github.com/MyoHub/myo_sim.git", commit=A.PIN, clean=True,
        registry_name="myolegs26")
A.CANDIDATE_SIGNATURE = tuple({signature!r})
A.CANDIDATE_NJNT = {njnt!r}

{perturbation}

out = Path({out!r})
verdict = dict()
try:
    {entry}
    verdict["ok"] = True
except BaseException as error:
    verdict["ok"] = False
    verdict["error"] = type(error).__name__ + ": " + str(error)
verdict["out_exists"] = out.exists()
verdict["files"] = sorted(p.name for p in out.iterdir()) if out.is_dir() else []
verdict["banned_present"] = A.banned_modules_present()
sys.stdout.write({mark!r} + json.dumps(verdict) + {mark_end!r})
'''

AUDIT_ENTRY = 'verdict["record"] = A.audit({source!r}, {out!r}, root={sandbox!r})'
MAIN_ENTRY = ('A.POLICY_ROOT = Path({sandbox!r})\n'
              '    sys.argv = ["audit_myolegs_candidate.py", "--source", {source!r},'
              ' "--out", {out!r}]\n'
              '    A.main()')


# --- fixtures and helpers -----------------------------------------------------------------------

@pytest.fixture
def probe():
    return mujoco.MjModel.from_xml_string(PROBE_XML)


@pytest.fixture
def sandbox(tmp_path):
    """A throwaway repository root with an ignored logs/ tree and a package tree."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/other_clone").mkdir()
    (tmp_path / "msk_warp").mkdir()
    return tmp_path


def _signature_of(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    return tuple(int(getattr(model, name)) for name in A.SIGNATURE_FIELDS), int(model.njnt)


def _stub_source(tmp_path, xml=PROBE_XML, contaminate=False, poke=""):
    package = tmp_path / "stub_source/myo_sim"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        STUB_INIT.format(xml=xml, contaminate=contaminate, poke=poke), encoding="utf-8")
    return tmp_path / "stub_source"


def _git_repo(path):
    """A real git repository whose HEAD is, necessarily, not the frozen pin."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "README").write_text("stand-in for a differently pinned clone\n", encoding="utf-8")

    def git(*args):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True,
                       text=True, timeout=CHILD_TIMEOUT)

    git("init", "-q")
    git("config", "user.email", "audit-test@example.invalid")
    git("config", "user.name", "Audit Test")
    git("add", "README")
    git("commit", "-q", "-m", "stand-in commit")
    git("remote", "add", "origin", "https://github.com/MyoHub/myo_sim.git")
    return path


def run_audit_child(tmp_path, source=None, out=None, xml=PROBE_XML, perturbation="pass",
                    stub_checked_source=True, contaminate=False, poke="", entry=AUDIT_ENTRY):
    """Run one whole audit in a fresh interpreter and return (completed child, verdict, out)."""
    source = _stub_source(tmp_path, xml=xml, contaminate=contaminate, poke=poke) if source is None \
        else source
    (tmp_path / "logs").mkdir(exist_ok=True)
    out = tmp_path / "logs/audit_01" if out is None else Path(out)
    signature, njnt = _signature_of(xml)
    script = tmp_path / "child.py"
    script.write_text(DRIVER.format(
        root=str(ROOT), tripwires=TRIPWIRE_APIS, stub_checked_source=stub_checked_source,
        signature=signature, njnt=njnt, perturbation=perturbation, out=str(out),
        entry=entry.format(source=str(source), out=str(out), sandbox=str(tmp_path)),
        mark=_MARK, mark_end=_MARK_END), encoding="utf-8")
    done = subprocess.run([sys.executable, "-B", str(script)], cwd=str(ROOT), text=True,
                          capture_output=True, timeout=CHILD_TIMEOUT)
    assert _MARK in done.stdout, f"child produced no verdict\nSTDOUT{done.stdout}\nSTDERR{done.stderr}"
    payload = done.stdout.split(_MARK, 1)[1].split(_MARK_END, 1)[0]
    return done, json.loads(payload), out


# --- the reused source helper and the frozen pin --------------------------------------------------

def test_the_reused_helper_still_returns_the_legacy_label():
    from scripts import build_myoleg26_assets as builder
    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert '"registry_name": "myolegs26"' in source
    assert A.LEGACY_LABEL == "myolegs26"


def test_helper_label_normalises_to_the_candidate_registry_name(monkeypatch):
    monkeypatch.setattr(A, "checked_source", lambda source, expected: dict(
        repository="https://github.com/MyoHub/myo_sim.git", commit=A.PIN, clean=True,
        registry_name="myolegs26"))
    record = A.candidate_source_record(ROOT)
    assert record["registry_name"] == "myolegs"
    assert record["composition_api"] == "load_spec('myolegs')"
    assert record["commit"] == A.PIN and record["clean"] is True
    A.assert_candidate_source(record)


def test_legacy_label_reintroduced_after_normalisation_is_rejected(monkeypatch):
    monkeypatch.setattr(A, "checked_source", lambda source, expected: dict(
        repository="r", commit=A.PIN, clean=True, registry_name="myolegs26"))
    record = A.candidate_source_record(ROOT)
    record["fragment_hint"] = A.LEGACY_LABEL
    with pytest.raises(ValueError, match="survives normalisation"):
        A.assert_candidate_source(record)


def test_unnormalised_registry_name_is_rejected(monkeypatch):
    monkeypatch.setattr(A, "checked_source", lambda source, expected: dict(
        repository="r", commit=A.PIN, clean=True, registry_name="myolegs26"))
    record = A.candidate_source_record(ROOT)
    record["registry_name"] = A.LEGACY_LABEL
    with pytest.raises(ValueError, match="registry_name"):
        A.assert_candidate_source(record)


def test_a_helper_label_other_than_the_legacy_one_is_rejected(monkeypatch):
    monkeypatch.setattr(A, "checked_source", lambda source, expected: dict(
        repository="r", commit=A.PIN, clean=True, registry_name="something_else"))
    with pytest.raises(ValueError, match="unexpected helper registry label"):
        A.candidate_source_record(ROOT)


def test_a_source_at_another_revision_is_refused(tmp_path):
    repo = _git_repo(tmp_path / "other_clone")
    with pytest.raises(ValueError, match="differs from required"):
        A.candidate_source_record(repo)


def test_the_frozen_pin_cannot_be_overridden_through_the_api():
    assert A.PIN == "eb327acbae0fad12279495040607f5235d962328"
    assert list(inspect.signature(A.candidate_source_record).parameters) == ["source"]
    assert list(inspect.signature(A.audit).parameters) == ["source", "out", "root"]


def test_the_cli_has_no_expected_commit_flag(tmp_path):
    done = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/audit_myolegs_candidate.py"),
         "--source", str(tmp_path), "--out", str(tmp_path / "logs/x"),
         "--expected-commit", "0" * 40],
        cwd=str(ROOT), text=True, capture_output=True, timeout=CHILD_TIMEOUT)
    assert done.returncode == 2
    assert "unrecognized arguments" in done.stderr


# --- hard gates, as pure helpers ------------------------------------------------------------------

def test_preregistered_expectations_are_unchanged():
    assert A.CANDIDATE_SIGNATURE == (35, 34, 80, 80, 14)
    assert A.CANDIDATE_NJNT == 29
    assert A.BASELINE_SIGNATURE == (47, 46, 26, 26, 28)
    assert A.SIGNATURE_FIELDS == ("nq", "nv", "nu", "na", "neq")


def test_signature_gate_blocks_a_mismatch(probe):
    with pytest.raises(ValueError, match="BLOCKED"):
        A.checked_signature("candidate", probe, A.CANDIDATE_SIGNATURE)


def test_signature_gate_accepts_the_matching_tuple(probe):
    signature = tuple(int(getattr(probe, name)) for name in A.SIGNATURE_FIELDS)
    assert A.checked_signature("probe", probe, signature) == list(signature)


def test_mujoco_version_gate_blocks_a_mismatch():
    with pytest.raises(ValueError, match="BLOCKED"):
        A.checked_mujoco_version("3.9.0", "3.10.0")
    assert A.checked_mujoco_version("3.10.0", "3.10.0") == "3.10.0"


def test_manifest_pins_the_mujoco_version_the_audit_checks():
    manifest = json.loads((A.ASSETS / "manifest.json").read_text())
    assert manifest["mujoco_version"] == "3.10.0"


def test_non_finite_named_array_is_rejected(probe):
    A.assert_recorded_values_valid("probe", probe)
    probe.body_mass[1] = np.nan
    with pytest.raises(ValueError, match="body_mass"):
        A.assert_recorded_values_valid("probe", probe)


def test_unordered_ctrlrange_is_rejected(probe):
    probe.actuator_ctrlrange[0] = [1.0, 0.0]
    with pytest.raises(ValueError, match="unordered"):
        A.assert_recorded_values_valid("probe", probe)


def test_non_finite_ctrlrange_is_rejected(probe):
    probe.actuator_ctrlrange[0] = [0.0, np.inf]
    with pytest.raises(ValueError, match="actuator_ctrlrange"):
        A.assert_recorded_values_valid("probe", probe)


def test_a_legal_infinity_outside_the_named_arrays_does_not_trip_the_gate(probe):
    probe.actuator_forcerange[0] = [-np.inf, np.inf]
    A.assert_recorded_values_valid("probe", probe)


# --- write-path policy ----------------------------------------------------------------------------

def test_valid_out_is_returned_without_being_created(sandbox):
    out = A.validated_out(sandbox / "logs/audit_01", sandbox / "logs/other_clone", root=sandbox)
    assert out == (sandbox / "logs/audit_01").resolve()
    assert not out.exists()


def test_existing_out_is_rejected(sandbox):
    out = sandbox / "logs/audit_01"
    out.mkdir()
    with pytest.raises(FileExistsError):
        A.validated_out(out, sandbox / "logs/other_clone", root=sandbox)
    assert list(out.iterdir()) == []


def test_out_outside_logs_is_rejected(sandbox):
    out = sandbox / "elsewhere/audit_01"
    with pytest.raises(ValueError, match="must live under the ignored logs/ tree"):
        A.validated_out(out, sandbox / "logs/other_clone", root=sandbox)
    assert not out.exists() and not out.parent.exists()


def test_out_inside_the_given_source_is_rejected(sandbox):
    source = sandbox / "logs/other_clone"
    out = source / "audit_01"
    with pytest.raises(ValueError, match="pinned upstream checkout"):
        A.validated_out(out, source, root=sandbox)
    assert not out.exists()
    assert list(source.iterdir()) == []


def test_out_inside_the_canonical_checkout_is_rejected_even_for_another_source(sandbox):
    """The asymmetric case: --source names a second clone, --out targets the pinned tree."""
    canonical = sandbox / A.PINNED_CHECKOUT
    canonical.mkdir(parents=True)
    out = canonical / "audit_01"
    with pytest.raises(ValueError, match="pinned upstream checkout"):
        A.validated_out(out, sandbox / "logs/other_clone", root=sandbox)
    assert not out.exists()
    assert list(canonical.iterdir()) == []


def test_out_under_the_package_tree_is_rejected(sandbox):
    out = sandbox / "msk_warp/audit_01"
    with pytest.raises(ValueError, match="msk_warp"):
        A.validated_out(out, sandbox / "logs/other_clone", root=sandbox)
    assert not out.exists()
    assert list((sandbox / "msk_warp").iterdir()) == []


def test_cli_rejects_a_bad_out_without_creating_anything(tmp_path):
    out = tmp_path / "elsewhere/audit_01"
    done = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/audit_myolegs_candidate.py"),
         "--source", str(tmp_path / "some_source"), "--out", str(out)],
        cwd=str(ROOT), text=True, capture_output=True, timeout=CHILD_TIMEOUT)
    assert done.returncode != 0
    assert "must live under the ignored logs/ tree" in done.stderr
    assert not out.exists() and not out.parent.exists()


# --- import ban and loader hygiene ----------------------------------------------------------------

def test_banned_import_gate_rejects_a_present_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", object())
    with pytest.raises(RuntimeError, match="warp"):
        A.assert_no_banned_imports()


def test_banned_import_gate_passes_when_all_are_absent(monkeypatch):
    for name in A.BANNED_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert A.banned_modules_present() == []
    A.assert_no_banned_imports()


def test_a_failed_identity_check_leaves_no_sys_path_or_module_residue(tmp_path, monkeypatch):
    """`myo_sim` resolvable from elsewhere must be refused and fully cleaned up."""
    elsewhere = _stub_source(tmp_path)
    empty = tmp_path / "empty_source"
    empty.mkdir()
    sentinel = object()
    monkeypatch.setitem(sys.modules, "unrelated_sentinel_module", sentinel)
    monkeypatch.delitem(sys.modules, "myo_sim", raising=False)
    monkeypatch.syspath_prepend(str(elsewhere))
    saved = list(sys.path)

    with pytest.raises(RuntimeError, match="resolved outside the verified checkout"):
        A.import_pinned_myo_sim(empty)

    assert "myo_sim" not in sys.modules
    assert sys.path == saved, "the rejected source must not stay on sys.path"
    assert sys.modules["unrelated_sentinel_module"] is sentinel
    for name in [n for n in sys.modules if n == "myo_sim" or n.startswith("myo_sim.")]:
        del sys.modules[name]


def test_the_unit_suite_does_not_depend_on_the_ignored_upstream_checkout():
    text = Path(__file__).read_text(encoding="utf-8")
    assert IGNORED_CHECKOUT not in text, "unit tests must stay hermetic on a fresh clone"
    assert A.PINNED_CHECKOUT == "logs/" + IGNORED_CHECKOUT


# --- inventory --------------------------------------------------------------------------------------

def test_inventory_reports_the_probe_topology_and_named_features(probe):
    record = A.inventory(probe, "probe")
    assert record["counts"] == {"njnt": 3, "nbody": 4, "ngeom": 4, "nsite": 2, "ntendon": 1,
                                "nwrap": 2, "npair": 1, "nsensor": 1, "nq": 9, "nv": 8,
                                "nu": 1, "na": 0, "neq": 2, "nkey": 0, "nmesh": 0}
    assert record["signature"] == [9, 8, 1, 0, 2]
    assert record["free_joints"] == ["root"]
    assert record["free_root"] == {"joint": "root", "body": "pelvis"}
    assert [j["name"] for j in record["joints"]] == ["root", "subtalar_angle_l", "mtp_angle_l"]
    hinge = record["joints"][1]
    assert hinge["type"] == "mjJNT_HINGE" and hinge["axis"] == [0.0, 1.0, 0.0]
    assert hinge["range"] == pytest.approx([-0.349, 0.349])
    assert record["named_features"] == {"patella_l": False, "patella_r": False,
                                        "calcn_l": True, "calcn_r": False,
                                        "toes_l": True, "toes_r": False,
                                        "subtalar_angle_l": True, "subtalar_angle_r": False}
    assert record["tendons"][0]["kind"] == "spatial"
    assert [w["type"] for w in record["tendons"][0]["wraps"]] == ["mjWRAP_SITE", "mjWRAP_SITE"]
    assert record["actuators"][0]["ctrlrange"] == [0.0, 1.0]
    assert record["sensors"][0] == {"name": "s_probe", "type": "mjSENS_JOINTPOS",
                                    "objtype": "mjOBJ_JOINT", "obj": "subtalar_angle_l", "dim": 1}
    assert record["option"]["timestep"] == pytest.approx(0.002)
    assert record["total_body_mass_kg"] == pytest.approx(float(probe.body_mass.sum()))
    assert record["qpos0"] == pytest.approx(list(map(float, probe.qpos0)))
    assert record["compiler"] is None


def test_inventory_recovers_equality_endpoints_including_unnamed_and_sentinel_rows(probe):
    equality = A.inventory(probe, "probe")["equality"]
    assert equality[0] == {"name": "eq_probe", "type": "mjEQ_JOINT", "objtype": "mjOBJ_JOINT",
                           "obj1_id": probe.joint("subtalar_angle_l").id, "obj1": "subtalar_angle_l",
                           "obj2_id": -1, "obj2": None, "active0": True,
                           "data": pytest.approx(list(map(float, probe.eq_data[0])))}
    assert equality[1]["name"] == "", "the second equality is deliberately unnamed"
    assert equality[1]["obj1"] == "subtalar_angle_l" and equality[1]["obj2"] == "mtp_angle_l"
    assert equality[1]["obj1_id"] == probe.joint("subtalar_angle_l").id
    assert equality[1]["obj2_id"] == probe.joint("mtp_angle_l").id
    # MuJoCo leaves eq_objtype unset for joint equalities; resolution must not depend on it.
    assert int(probe.eq_objtype[0]) == int(mujoco.mjtObj.mjOBJ_UNKNOWN)


def test_inventory_records_per_geom_identity_and_a_self_contact_pair(probe):
    record = A.inventory(probe, "probe")
    geoms = {g["name"]: g for g in record["geoms"]}
    assert [g["id"] for g in record["geoms"]] == [0, 1, 2, 3]
    assert geoms["floor"]["type"] == "mjGEOM_PLANE" and geoms["floor"]["body"] == "world"
    assert (geoms["floor"]["contype"], geoms["floor"]["conaffinity"]) == (1, 1)
    assert geoms["floor"]["friction"] == pytest.approx([1.0, 0.005, 0.0001])
    assert geoms["pelvis_geom"]["type"] == "mjGEOM_SPHERE"
    assert (geoms["pelvis_geom"]["contype"], geoms["pelvis_geom"]["conaffinity"]) == (2, 2)
    assert record["contact_pairs"] == [
        {"geom1_id": geoms["pelvis_geom"]["id"], "geom1": "pelvis_geom", "body1": "pelvis",
         "geom2_id": geoms["calcn_geom"]["id"], "geom2": "calcn_geom", "body2": "calcn_l",
         "condim": 3}]
    # The explicit pair is body-against-body inside the model; no ground geom takes part in it.
    assert "floor" not in {p["geom1"] for p in record["contact_pairs"]}
    assert "floor" not in {p["geom2"] for p in record["contact_pairs"]}


def test_inventory_records_the_compiler_block_when_a_spec_is_supplied():
    spec = mujoco.MjSpec.from_string(PROBE_XML)
    record = A.inventory(spec.compile(), "probe", A.compiler_settings(spec))
    assert set(record["compiler"]) == set(A.COMPILER_FIELDS)
    assert record["compiler"]["degree"] == 0


def test_inventory_records_no_pelvis_frame_when_the_body_is_absent():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="torso"><geom size="0.1"/></body></worldbody></mujoco>')
    record = A.inventory(model, "probe")
    assert record["pelvis"] is None and record["free_root"] is None


# --- whole-audit integration, one fresh child per case ----------------------------------------------

def test_a_fresh_child_audit_succeeds_and_imports_no_banned_module(tmp_path):
    done, verdict, out = run_audit_child(tmp_path)
    assert verdict["ok"], verdict.get("error")
    assert verdict["banned_present"] == []
    assert verdict["files"] == ["myolegs_candidate_audit.json"]
    record = verdict["record"]
    assert json.loads((out / "myolegs_candidate_audit.json").read_text()) == record
    assert record["provenance"]["banned_modules_present"] == []
    assert record["provenance"]["banned_modules"] == list(A.BANNED_MODULES)
    assert record["provenance"]["source"]["registry_name"] == "myolegs"
    assert record["provenance"]["source"]["composition_api"] == "load_spec('myolegs')"
    assert record["provenance"]["source"]["commit"] == A.PIN
    assert record["provenance"]["script_sha256"] == A.sha256(A.__file__)
    assert record["provenance"]["baseline_reference_xml"]["sha256"] == A.sha256(A.ASSETS / "reference.xml")
    assert record["expectations"]["status"] == "PREDICTIONS NOT MEASURED"
    # The real committed baseline is compiled and gated on every one of these runs.
    assert record["measured"]["baseline"]["signature"] == list(A.BASELINE_SIGNATURE)
    assert record["measured"]["baseline"]["compiler"] is not None
    assert record["measured"]["candidate"]["compiler"] is not None
    assert record["status"] == {"dynamics": "UNVALIDATED", "warp_support": "UNVALIDATED",
                                "gradients": "UNVALIDATED", "learnability": "UNVALIDATED",
                                "contact_behaviour": "UNVALIDATED",
                                "task_version": None, "vendored": False,
                                "remote_delta": "not verified",
                                "license_notice": "unresolved - blocks later redistribution only",
                                "successor_work": "not authorised"}
    assert done.returncode == 0


def test_the_cli_entry_point_succeeds_and_prints_the_measured_counts(tmp_path):
    done, verdict, out = run_audit_child(tmp_path, entry=MAIN_ENTRY)
    assert verdict["ok"], verdict.get("error")
    assert verdict["files"] == ["myolegs_candidate_audit.json"]
    assert "Audited myolegs candidate (nq=9 nv=8 nu=1 na=0 neq=2 njnt=3)" in done.stdout


@pytest.mark.parametrize("case, perturbation, expected, kwargs", [
    ("mujoco_version", 'mujoco.__version__ = "0.0.0"', "BLOCKED: MuJoCo", {}),
    ("candidate_signature", "A.CANDIDATE_SIGNATURE = (1, 2, 3, 4, 5)",
     "BLOCKED: candidate signature", {}),
    ("candidate_njnt", "A.CANDIDATE_NJNT = 999", "BLOCKED: candidate njnt", {}),
    ("baseline_signature", "A.BASELINE_SIGNATURE = (1, 2, 3, 4, 5)",
     "BLOCKED: baseline signature", {}),
    ("free_root", "pass", "BLOCKED: candidate has 2 free joints", {"xml": TWO_ROOT_XML}),
    ("non_finite", "pass", "BLOCKED: candidate body_mass", {"poke": "nonfinite"}),
    ("ctrlrange", "pass", "BLOCKED: candidate actuator_ctrlrange", {"poke": "ctrlrange"}),
    ("contaminated_entry",
     'sys.modules.setdefault("torch", types.ModuleType("torch"))',
     "imported banned modules ['torch']", {}),
    ("contaminated_transitively", "pass", "imported banned modules ['torch']",
     {"contaminate": True}),
    ("myo_sim_preimported", 'sys.modules["myo_sim"] = types.ModuleType("myo_sim")',
     "fresh Python process", {}),
])
def test_every_hard_gate_blocks_the_whole_audit_and_writes_nothing(
        tmp_path, case, perturbation, expected, kwargs):
    _, verdict, out = run_audit_child(tmp_path, perturbation=perturbation, **kwargs)
    assert not verdict["ok"], f"{case} should have been blocked"
    assert expected in verdict["error"], verdict["error"]
    assert verdict["out_exists"] is False and not out.exists()


def test_contamination_at_entry_is_refused_before_any_source_work(tmp_path):
    """A contaminated process must be turned away first, not after the source has been read."""
    perturbation = (
        'sys.modules.setdefault("torch", types.ModuleType("torch"))\n'
        'def _forbidden(*args, **kwargs):\n'
        '    raise AssertionError("source work ran in a contaminated process")\n'
        'A.checked_source = _forbidden\n'
        'A.import_pinned_myo_sim = _forbidden')
    _, verdict, out = run_audit_child(tmp_path, perturbation=perturbation)
    assert not verdict["ok"]
    assert "imported banned modules ['torch']" in verdict["error"]
    assert "source work ran" not in verdict["error"]
    assert verdict["out_exists"] is False and not out.exists()


def test_the_audit_refuses_an_out_inside_the_canonical_checkout(tmp_path):
    """--source names the stub clone; --out targets the canonical pinned tree. Must refuse."""
    canonical = tmp_path / A.PINNED_CHECKOUT
    canonical.mkdir(parents=True)
    _, verdict, out = run_audit_child(tmp_path, out=canonical / "audit_01")
    assert not verdict["ok"]
    assert "pinned upstream checkout" in verdict["error"]
    assert verdict["out_exists"] is False and not out.exists()
    assert list(canonical.iterdir()) == []


def test_the_audit_refuses_a_source_at_another_revision(tmp_path):
    repo = _git_repo(tmp_path / "other_clone")
    _, verdict, out = run_audit_child(tmp_path, source=repo, stub_checked_source=False)
    assert not verdict["ok"]
    assert "differs from required" in verdict["error"]
    assert verdict["out_exists"] is False and not out.exists()
