"""Unit tests for the v2 experiment freeze and the Task 4c timing instrumentation.

CPU only. Temp manifests, tiny real temp git repositories, injected ``git``
runners, fake clocks and the Task 3 analytic CPU adapter. **No simulator, no
CUDA, no Warp, no GPU and no training.** Nothing here spends the 28,800 s
training budget or the 1,800 s diagnostic budget; test-suite time is separate
accounting.

The committed v1 manifest is never touched, never re-hashed and never
regenerated: its post-Task-2 refusal is a documented boundary, not something that
has to pass. ``.gitattributes`` is never modified and no file is normalised.

Nothing here qualifies a physics result, a gradient, a behaviour claim or an
acceleration claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from msk_warp.analysis import ppo_v2_protocol as P
from scripts import run_myoleg26_ppo_v2 as R

from tests.unit.test_run_myoleg26_ppo_v2 import (  # noqa: E402
    TickingClock, _build_seam, _parse, _run, _worker_argv,
)
from tests.unit.test_ppo_resume_state import _build, _start  # noqa: E402

V1_MANIFEST = Path("msk_warp/configs/experiments/myoleg26_baseline_v1.json")


# ==========================================================================
# Helpers
# ==========================================================================

def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout


def _repo(tmp_path, *, autocrlf="true") -> Path:
    """A tiny real git repository, so the three identity forms are real objects."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "core.autocrlf", autocrlf)
    (root / ".gitattributes").write_bytes(b"* text=auto\n")
    return root


def _record(**overrides) -> dict:
    """A minimal freeze record with the shape ``validate_freeze_v2`` compares."""
    record = {
        "schema_version": R.FREEZE_SCHEMA,
        "protocol": {"digest": "0" * 64},
        "config": {"params": {}},
        "files": {"pinned.py": "a" * 64},
        "files_git": {"working_tree_blob_sha1": {"pinned.py": "b" * 40},
                      "head_blob_sha1": {"pinned.py": "b" * 40},
                      "head_status": {"pinned.py": "in_head"}},
        "files_eol": {"pinned.py": "lf"},
        "model_sha256": "c" * 64,
        "compiled_model_sha256": "d" * 64,
        "backend": {"head": "e" * 40},
        "packages": {"mujoco": "3.10.0"},
        "python": "3.12.0",
    }
    record.update(overrides)
    return record


def _clean_git(*args, **kwargs):
    """A ``git`` stand-in that reports a clean, fully committed tree."""
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _manifest(tmp_path, record, *, name="myoleg26_ppo_v2.json") -> Path:
    path = tmp_path / name
    R.write_json_exclusive(path, record)
    return path


def _repo_with_pinned(tmp_path) -> Path:
    """A repository whose one declared input really is committed and clean, so
    the porcelain and ls-files checks run for real instead of against a fake."""
    root = _repo(tmp_path)
    (root / "pinned.py").write_bytes(b"x = 1\n")
    _git(root, "add", "pinned.py")
    _git(root, "commit", "--quiet", "-m", "pinned")
    return root


# ==========================================================================
# Unit 1 -- the pin set: what the campaign executes
# ==========================================================================

def test_the_pin_set_names_every_driver_the_campaign_executes():
    """Binding. The ``msk_warp/**/*.py`` glob covers no top-level script, so each
    driver is pinned by name or it is not pinned at all."""
    assert "scripts/run_myoleg26_ppo_v2.py" in R.PINNED_PATHS
    assert "scripts/summarize_myoleg26_ppo_v2.py" in R.PINNED_PATHS
    assert "msk_warp/configs/experiments/myoleg26_ppo_v2.yaml" in R.PINNED_PATHS
    # Every excluded path carries a stated ground, so it cannot read as an
    # oversight.
    assert "scripts/diag_myoleg26_forward_data_mode.py" in R.EXCLUDED_PATHS
    for path, reason in R.EXCLUDED_PATHS.items():
        assert isinstance(reason, str) and len(reason) > 40, path
    assert not set(R.PINNED_PATHS) & set(R.EXCLUDED_PATHS)


def test_the_diagnostic_and_the_docs_resident_runner_are_excluded_on_stated_grounds():
    grounds = R.EXCLUDED_PATHS
    assert "R05" in grounds["scripts/diag_myoleg26_forward_data_mode.py"]
    key = next(name for name in grounds if name.endswith("byte_exact_runner.py"))
    assert "docs/" in grounds[key] and "ls-files" in grounds[key]


def test_every_pinned_path_exists_and_is_inside_the_repository():
    """Behavioural. A pin that names a missing file would fail at freeze time;
    that must be visible here, not during the campaign.

    Whether each pin is *committed* is deliberately not asserted here: it is
    enforced by ``validate_freeze_v2``'s porcelain and ls-files checks, and a
    driver is legitimately untracked in the working tree of the very unit that
    introduces it.
    """
    for relative in R.PINNED_PATHS:
        path = R.ROOT / relative
        assert path.is_file(), relative
        assert path.resolve().is_relative_to(R.ROOT), relative
    assert R.freeze_identity.__module__ == R.__name__


# ==========================================================================
# Unit 2 -- the three identity forms stay distinct (IN-25)
# ==========================================================================

def test_raw_bytes_filtered_blob_and_head_blob_are_three_distinct_fields(tmp_path):
    """Binding, on real git objects. ``core.autocrlf`` materialises CRLF on
    checkout, so a file's raw sha256 legitimately differs from its blob content
    hash. Both are recorded; nothing is normalised."""
    root = _repo(tmp_path)
    (root / "crlf.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (root / "lf.py").write_bytes(b"x = 1\ny = 2\n")
    _git(root, "add", "crlf.py", "lf.py")
    _git(root, "commit", "--quiet", "-m", "pinned")
    files = {"crlf.py": R.sha256_file(root / "crlf.py"),
             "lf.py": R.sha256_file(root / "lf.py")}

    identity = R.freeze_identity(files, root=root)
    blob = identity["working_tree_blob_sha1"]
    head = identity["head_blob_sha1"]

    # The CRLF file's raw bytes are NOT its blob content, and both LF and CRLF
    # files share the same blob because the filter normalised them.
    assert files["crlf.py"] != files["lf.py"]
    assert blob["crlf.py"] == blob["lf.py"] == head["lf.py"] == head["crlf.py"]
    assert identity["head_status"] == {"crlf.py": "in_head", "lf.py": "in_head"}
    assert identity["eol"] == {"crlf.py": "crlf", "lf.py": "lf"}
    assert "autocrlf" in identity["note"]


def test_the_eol_note_states_the_host_refusal_consequence_not_only_the_mechanism():
    """Binding (M2). The manifest is the artifact a future operator reads first,
    so it must say what they will actually hit: on a host whose core.autocrlf
    differs, raw hashes refuse, and that refusal is correct behaviour rather than
    source drift."""
    note = R.FREEZE_EOL_NOTE
    assert "HOST DIFFERENCE, not source drift" in note
    assert "correct behaviour" in note
    assert "core.autocrlf setting differs" in note
    assert "Frozen experiment mismatch: ['files']" in note
    # How to tell the two apart, named explicitly.
    assert "comparing files_git first" in note
    assert "host-independent" in note
    # And that this is strictly stronger than v1's files-only manifest.
    assert "stronger than the v1 freeze" in note
    assert "no git identity classes and no EOL census" in note
    # The note travels with the manifest, not only with the source.
    frozen = json.loads((R.ROOT / V1_MANIFEST).parent.joinpath(
        "myoleg26_ppo_v2.json").read_text(encoding="utf-8"))
    assert frozen["eol_note"] == note


def test_an_uncommitted_pinned_input_is_recorded_as_not_in_head(tmp_path):
    root = _repo(tmp_path)
    (root / "fresh.py").write_bytes(b"x = 1\n")
    files = {"fresh.py": R.sha256_file(root / "fresh.py")}

    identity = R.freeze_identity(files, root=root)

    assert identity["head_blob_sha1"] == {"fresh.py": None}
    assert identity["head_status"] == {"fresh.py": "not_in_head"}
    # The working-tree blob still exists: that is the point of keeping the two
    # identities apart.
    assert identity["working_tree_blob_sha1"]["fresh.py"]


def test_the_eol_form_names_binary_rather_than_claiming_mixed_endings(tmp_path):
    """Behavioural. A mesh or texture has no line endings; labelling it 'mixed'
    because its payload contains CR LF bytes would be a meaningless claim."""
    root = _repo(tmp_path)
    (root / "mesh.bin").write_bytes(bytes([0, 1, 13, 10, 10, 0]))
    (root / "mixed.py").write_bytes(bytes([97, 32, 61, 32, 49, 13, 10, 98, 32, 61, 32, 50, 10]))
    (root / "empty.py").write_bytes(b"")
    files = {name: R.sha256_file(root / name)
             for name in ("mesh.bin", "mixed.py", "empty.py")}

    identity = R.freeze_identity(files, root=root)

    assert identity["eol"] == {"mesh.bin": "binary", "mixed.py": "mixed",
                               "empty.py": "none"}


def test_a_raw_hash_that_disagrees_with_the_supplied_map_is_refused(tmp_path):
    root = _repo(tmp_path)
    (root / "pinned.py").write_bytes(b"x = 1\n")
    with pytest.raises(ValueError, match="raw"):
        R.freeze_identity({"pinned.py": "0" * 64}, root=root)


# ==========================================================================
# Unit 3 -- validate_freeze_v2
# ==========================================================================

def test_a_changed_pinned_input_fails_validation(tmp_path, monkeypatch):
    frozen = _record()
    path = _manifest(tmp_path, frozen)
    monkeypatch.setattr(R, "_git", _clean_git)
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: _record())
    assert R.validate_freeze_v2(path, smoke=True) == frozen

    monkeypatch.setattr(R, "freeze_record_v2",
                        lambda **kwargs: _record(files={"pinned.py": "f" * 64}))
    with pytest.raises(ValueError, match="Frozen experiment mismatch"):
        R.validate_freeze_v2(path, smoke=True)


def test_each_identity_class_is_compared_separately(tmp_path, monkeypatch):
    """Binding. Agreement on the raw bytes must not excuse a blob or HEAD
    mismatch, and the failure names the class that moved."""
    frozen = _record()
    path = _manifest(tmp_path, frozen)
    monkeypatch.setattr(R, "_git", _clean_git)
    for key in ("working_tree_blob_sha1", "head_blob_sha1", "head_status"):
        moved = _record()
        moved["files_git"][key] = {"pinned.py": "moved"}
        monkeypatch.setattr(R, "freeze_record_v2", lambda m=moved, **kwargs: m)
        with pytest.raises(ValueError, match="files_git"):
            R.validate_freeze_v2(path, smoke=True)
    # The raw map alone moving is reported as ``files``, not as ``files_git``.
    monkeypatch.setattr(R, "freeze_record_v2",
                        lambda **kwargs: _record(files={"pinned.py": "9" * 64}))
    with pytest.raises(ValueError, match=r"\['files'\]"):
        R.validate_freeze_v2(path, smoke=True)


def test_a_later_independent_module_does_not_invalidate_the_freeze(tmp_path, monkeypatch):
    """Behavioural, mirroring v1: a new standalone checker may appear; every
    existing pin stays pinned."""
    frozen = _record()
    path = _manifest(tmp_path, frozen)
    monkeypatch.setattr(R, "_git", _clean_git)
    extended = _record(files={"pinned.py": "a" * 64, "new_checker.py": "1" * 64})
    extended["files_git"]["working_tree_blob_sha1"]["new_checker.py"] = "2" * 40
    extended["files_git"]["head_blob_sha1"]["new_checker.py"] = "2" * 40
    extended["files_git"]["head_status"]["new_checker.py"] = "in_head"
    extended["files_eol"]["new_checker.py"] = "lf"
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: extended)
    assert R.validate_freeze_v2(path, smoke=True) == frozen


def test_an_uncommitted_pinned_input_fails_the_porcelain_check(tmp_path, monkeypatch):
    path = _manifest(tmp_path, _record())
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: _record())

    def dirty(root, *args, **kwargs):
        if args and args[0] == "status":
            return SimpleNamespace(returncode=0, stdout=" M pinned.py\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(R, "_git", dirty)
    with pytest.raises(ValueError, match="uncommitted"):
        R.validate_freeze_v2(path, smoke=True)


def test_an_untracked_pinned_input_fails_the_ls_files_check(tmp_path, monkeypatch):
    path = _manifest(tmp_path, _record())
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: _record())

    def untracked(root, *args, **kwargs):
        if args and args[0] == "ls-files":
            return SimpleNamespace(returncode=1, stdout="", stderr="no such path")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(R, "_git", untracked)
    with pytest.raises(ValueError, match="not committed"):
        R.validate_freeze_v2(path, smoke=True)


def test_a_manifest_whose_committed_bytes_differ_from_disk_fails(tmp_path, monkeypatch):
    """Binding, on a real repository. This is the self-reference-free binding:
    the manifest cannot contain its own hash, so it is bound by comparing its
    committed blob identity against its filtered working-tree blob identity."""
    root = _repo_with_pinned(tmp_path)
    record = _record()
    path = _manifest(root, record)
    _git(root, "add", path.name)
    _git(root, "commit", "--quiet", "-m", "freeze")
    monkeypatch.setattr(R, "ROOT", root)
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: _record())
    assert R.validate_freeze_v2(path) == record

    path.write_text(json.dumps(dict(record, extra=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="committed bytes"):
        R.validate_freeze_v2(path)


def test_a_checkout_materialised_crlf_manifest_still_validates(tmp_path, monkeypatch):
    """Binding, and the reason the comparison is blob-to-blob. This manifest is
    not pinned 'text eol=lf' -- pinning it would mean changing the repository's
    git attribute rules, which this unit must not do -- so a checkout gives it
    CRLF while its committed blob holds LF. Its content is unchanged, and a raw
    byte comparison would refuse it for a line ending."""
    root = _repo_with_pinned(tmp_path)
    record = _record()
    path = _manifest(root, record)
    _git(root, "add", path.name)
    _git(root, "commit", "--quiet", "-m", "freeze")
    monkeypatch.setattr(R, "ROOT", root)
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: _record())
    lf_bytes = path.read_bytes()

    path.unlink()
    _git(root, "checkout", "--", path.name)
    crlf_bytes = path.read_bytes()

    assert R._eol(crlf_bytes) == "crlf" and R._eol(lf_bytes) == "lf"
    assert R.sha256_file(path) != __import__("hashlib").sha256(lf_bytes).hexdigest()
    # Scoped to the manifest: git reports no change to it at all, even though
    # its bytes on disk are now different.
    assert not _git(root, "status", "--porcelain", "--", path.name).strip()
    assert R.validate_freeze_v2(path) == record


def test_the_manifest_never_pins_itself(tmp_path, monkeypatch):
    """Binding. A manifest that hashed itself could never validate, and a pin of
    its own HEAD identity would be created by its own commit."""
    relative = "msk_warp/configs/experiments/myoleg26_ppo_v2.json"
    record = _record(files={"pinned.py": "a" * 64, relative: "7" * 64})
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    R.write_json_exclusive(path, record)
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "_git", _clean_git)
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: record)
    with pytest.raises(ValueError, match="must not pin itself"):
        R.validate_freeze_v2(path, smoke=True)


def test_the_v1_manifest_is_never_written_rehashed_or_pinned_by_this_unit():
    """Binding. v1 stays byte-intact and out of the v2 pin set.

    It is *named* exactly once in the runner, as an :data:`EXCLUDED_PATHS`
    ground, which is the documentation the exclusion requires. The protocol
    module does read it, to cross-check that the v2 reset blocks stay disjoint
    from v1's -- a read of unchanged bytes, not a regeneration. What must never
    happen is a write, a re-hash into this freeze, or a pin.
    """
    before = R.sha256_file(R.ROOT / V1_MANIFEST)
    assert before == "9e6d19e586d5ec1560e6dc2a95cadfd3295966d871315c67f6ab99f341ce04e1"
    source = (R.ROOT / "scripts/run_myoleg26_ppo_v2.py").read_text(encoding="utf-8")
    assert source.count("myoleg26_baseline_v1") == 1
    assert V1_MANIFEST.as_posix() in R.EXCLUDED_PATHS
    assert V1_MANIFEST.as_posix() not in R.PINNED_PATHS
    P.assert_reset_blocks_disjoint()
    assert R.sha256_file(R.ROOT / V1_MANIFEST) == before


def test_the_freeze_cli_writes_then_validates_and_refuses_an_overwrite(tmp_path, monkeypatch):
    record = _record()
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: record)
    monkeypatch.setattr(R, "_git", _clean_git)
    out = tmp_path / "v2.json"

    assert R.main(["freeze", "--write-freeze", str(out)]) == R.EXIT_OK
    assert json.loads(out.read_text(encoding="utf-8")) == record
    assert R.main(["freeze", "--validate-freeze", str(out), "--smoke"]) == R.EXIT_OK

    monkeypatch.setattr(R, "freeze_record_v2",
                        lambda **kwargs: _record(files={"pinned.py": "0" * 64}))
    assert R.main(["freeze", "--validate-freeze", str(out), "--smoke"]) == R.EXIT_REFUSED


def test_a_generation_time_freeze_failure_is_a_refusal_not_a_traceback(tmp_path, monkeypatch,
                                                                      capsys):
    """Binding (M1). An operator setting up the campaign is exactly the person most
    likely to mistype a path or to have an input out of place, so both failure
    classes on the generation branch must arrive as a named refusal on stderr and
    a non-zero exit, not as an unhandled traceback.

    The library functions still raise: only the CLI boundary converts.
    """
    out = tmp_path / "v2.json"
    monkeypatch.setattr(R, "freeze_record_v2", lambda **kwargs: _record())
    monkeypatch.setattr(R, "_git", _clean_git)
    assert R.main(["freeze", "--write-freeze", str(out)]) == R.EXIT_OK
    capsys.readouterr()

    # (a) the output path already exists: the manifest is never overwritten.
    assert R.main(["freeze", "--write-freeze", str(out)]) == R.EXIT_REFUSED
    message = capsys.readouterr().err
    assert "REFUSED" in message and str(out) in message

    # (b) the record itself cannot be built -- a missing pinned input, an asset
    # mismatch, changed model dimensions, a dirty backend, a non-disjoint block.
    fresh = tmp_path / "fresh.json"

    def refusing(**kwargs):
        raise R.FreezeError("pinned input is missing: scripts/absent.py")

    monkeypatch.setattr(R, "freeze_record_v2", refusing)
    assert R.main(["freeze", "--write-freeze", str(fresh)]) == R.EXIT_REFUSED
    message = capsys.readouterr().err
    assert "FREEZE INVALID" in message and "scripts/absent.py" in message
    assert not fresh.exists()

    # The library contract is unchanged: it raises, and only main() converts.
    with pytest.raises(R.FreezeError):
        R.freeze_record_v2()
    with pytest.raises(FileExistsError):
        R.write_json_exclusive(out, {"a": 1})


@pytest.mark.parametrize("mode", ["launch", "worker", "select", "freeze"])
def test_every_mode_including_freeze_has_help(mode, capsys):
    with pytest.raises(SystemExit) as raised:
        R.build_parser().parse_args([mode, "--help"])
    assert raised.value.code == 0
    assert mode in capsys.readouterr().out


def test_the_freeze_mode_requires_exactly_one_action():
    with pytest.raises(SystemExit):
        R.build_parser().parse_args(["freeze"])
    with pytest.raises(SystemExit):
        R.build_parser().parse_args(["freeze", "--write-freeze", "a",
                                     "--validate-freeze", "b"])


# ==========================================================================
# Unit 4 -- the timing instrumentation (authorized Task 4c follow-up)
# ==========================================================================

def test_the_timing_accumulators_add_no_control_clock_read(tmp_path):
    """Binding. The control clock's read sequence decides every deadline, so the
    instrumentation must be provably invisible to it: identical outcomes AND an
    identical read count with the timers off and on."""
    outcomes, reads, rng = [], [], []
    for timing_clock in (None, TickingClock(start=1000.0, tick=0.5)):
        algo, env = _build(seed=11, steps_num=4, num_envs=4)
        _start(algo, env)
        before = torch.get_rng_state().clone()
        clock = TickingClock(tick=1.0)
        outcome = _run(algo, env, epochs=3, clock=clock, deadline=40.0,
                       capture_reserve_s=5.0, timing_clock=timing_clock)
        outcomes.append(outcome)
        reads.append(clock.reads)
        rng.append(torch.equal(torch.get_rng_state(), before))

    first, second = outcomes
    assert reads[0] == reads[1]
    assert first.stop_reason == second.stop_reason
    assert first.completed_epoch == second.completed_epoch
    assert first.completed_epochs == second.completed_epochs
    assert first.boundary_is_live == second.boundary_is_live
    assert first.censored == second.censored
    assert first.counters == second.counters
    assert first.losses == second.losses
    assert len(first.epoch_events) == len(second.epoch_events)
    # Timers off means zero, not a fabricated value; on means real attribution.
    assert all(value == 0.0 for value in first.timers.values())
    assert second.timers["rollout_s"] > 0.0 and second.timers["update_s"] > 0.0
    assert rng == [rng[0], rng[0]]


def test_a_worker_that_never_mentions_timing_gets_timings_anyway(tmp_path):
    """Binding, and the whole point of Ruling 2a's polarity: the default is ON.

    A campaign launcher that simply never passes ``timing_clock`` must still
    produce a populated split, or the measurement gap would only surface after
    the eight hours were spent.
    """
    out = tmp_path / "run" / "segment_0000"
    out.mkdir(parents=True)
    args = _parse(_worker_argv(tmp_path, out))
    assert R.run_worker(args, build=_build_seam(), clock=TickingClock(tick=0.0)) == R.EXIT_OK
    timing = R.read_json(out / "result.json")["timing_seconds"]

    assert timing["available"] is True
    assert timing["attribution"] == R.TIMING_ATTRIBUTION
    assert set(R.TIMING_PHASES) <= set(timing)
    assert all(timing[phase] >= 0.0 for phase in R.TIMING_PHASES)
    assert timing["rollout_s"] > 0.0 and timing["update_s"] > 0.0
    assert timing["build_s"] > 0.0 and timing["capture_publish_s"] > 0.0
    assert timing["attributed_sum_s"] > 0.0


def test_the_worker_default_timing_values_and_residual_are_exact(tmp_path, monkeypatch):
    """Binding. Still no ``timing_clock`` argument; the production default is
    made deterministic by patching the module's clock source, which reaches only
    the timing path because the control clock is injected separately."""
    ticks = TickingClock(start=0.0, tick=2.0)
    monkeypatch.setattr(R, "time", SimpleNamespace(perf_counter=ticks))
    out = tmp_path / "run" / "segment_0000"
    out.mkdir(parents=True)
    args = _parse(_worker_argv(tmp_path, out, **{"--epochs": "1"}))
    control = TickingClock(start=0.0, tick=3.0)
    assert R.run_worker(args, build=_build_seam(), clock=control) == R.EXIT_OK
    result = R.read_json(out / "result.json")
    timing = result["timing_seconds"]

    # Every phase is measured by exactly two reads of a 2.0 s ticking clock.
    for phase in R.TIMING_PHASES:
        assert timing[phase] == 2.0, phase
    assert timing["attributed_sum_s"] == 2.0 * len(R.TIMING_PHASES)
    assert timing["total_wall_seconds"] == result["wall_seconds"]
    assert timing["unattributed_residual_s"] == \
        result["wall_seconds"] - timing["attributed_sum_s"]
    assert timing["unattributed_residual_s"] != 0.0


def test_the_timing_record_labels_its_attribution_and_its_residual(tmp_path):
    out = tmp_path / "run" / "segment_0000"
    out.mkdir(parents=True)
    args = _parse(_worker_argv(tmp_path, out, **{"--epochs": "1"}))
    assert R.run_worker(args, build=_build_seam(), clock=TickingClock(tick=0.0)) == R.EXIT_OK
    timing = R.read_json(out / "result.json")["timing_seconds"]

    assert "unsynchronized" in timing["attribution"]
    assert "asynchronous" in timing["attribution_note"]
    assert "attribution rather than" in timing["attribution_note"]
    assert "perf_counter" in timing["residual_note"]
    assert "not comparable" in timing["v1_comparability"].lower()
    assert "synchronize" in timing["v1_comparability"]


def test_no_timing_value_is_ever_a_deadline_input():
    """Binding, by source inspection. Every deadline comparison in
    ``train_segment`` reads the control clock and the two stop-policy maxima, and
    none of them mentions a timer."""
    source = (R.ROOT / "scripts/run_myoleg26_ppo_v2.py").read_text(encoding="utf-8")
    body = source.split("def train_segment(")[1].split("\ndef ")[0]
    for line in body.splitlines():
        if "work_deadline" in line and ">" in line:
            assert "timers" not in line and "timing" not in line, line
    assert "EPOCH_SAFETY_FACTOR * epoch_cost" in body
    assert "EPOCH_SAFETY_FACTOR * update_cost" in body
    # The timers are written, never read back into a decision.
    assert "if timers" not in body and "timers[" in body


def test_the_partial_update_cost_stays_its_own_field(tmp_path):
    """Behavioural. An update that started and did not finish is recorded as
    partial update cost, not folded into the completed update phase."""
    algo, env = _build(seed=5, steps_num=2, num_envs=4)
    _start(algo, env)
    original = algo.update
    calls = {"n": 0}

    def failing_update():
        calls["n"] += 1
        if calls["n"] == 2:
            raise R.WorkDeadlineExceeded("deadline reached inside the update")
        return original()

    algo.update = failing_update
    outcome = _run(algo, env, epochs=4, clock=TickingClock(tick=1.0),
                   timing_clock=TickingClock(start=0.0, tick=1.0))

    assert outcome.counters.partial_update_cost_s > 0.0
    assert outcome.timers["update_s"] > 0.0
    assert outcome.stop_reason == "work_deadline_partial_epoch"
