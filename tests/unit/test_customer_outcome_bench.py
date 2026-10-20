"""Pure controls: no model, native CAD, listener or real worker is launched."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from scripts import customer_outcome_bench as bench
from scripts.customer_outcome_bench import (
    DEFAULT_CORPUS,
    MODEL,
    CustomerLocalAdapter,
    FixtureAdapter,
    Journal,
    analytic_spec,
    assess,
    current_capture_receipt,
    encoded,
    load_corpus,
    measure_step,
    read_events,
    requested_project_files,
    run_benchmark,
    sha,
    source_inventory,
    strict_json,
    verified_archive,
)


def observation(**changes):
    return {
        "actual_model": MODEL,
        "identity_verified": True,
        "data_origin": "actual_customer_pipeline",
        "state": "complete",
        "files_verified": True,
        "guide_current": True,
        "literal_guide": True,
        "revision_current": True,
        "unresolved": 0,
        "required_failed": 0,
        "nominal": {"status": "pass"},
        "total_seconds": 10.0,
        "first_cad_seconds": 4.0,
        "owner_wait_seconds": 0.0,
        **changes,
    }


@pytest.fixture
def cases():
    return load_corpus(DEFAULT_CORPUS)[0]


def test_frozen_corpus_has_separate_splits_references_and_explicit_gaps(cases):
    assert len(cases) == 30
    assert sum(case["split"] == "development" for case in cases) == 20
    assert sum(case["split"] == "heldout" for case in cases) == 10
    assert len({case["family"] for case in cases}) >= 12
    assert sum(case["oracle"]["kind"] == "unassessed_v1" for case in cases) == 3
    assert sum(bool(case.get("reference")) for case in cases) == 4
    assert any(case.get("edit") for case in cases)
    assert any(case.get("recovery") for case in cases)


def test_synthetic_cannot_earn_actual_pass(tmp_path, cases):
    case = cases[0]
    adapter = FixtureAdapter({case["id"]: observation()})
    row = run_benchmark(
        tmp_path / "run",
        DEFAULT_CORPUS,
        [case["id"]],
        adapter,
        inventory=lambda: {"source": "fixed"},
    )[0]
    assert row["assessment"]["nominal_status"] == "pass"
    assert row["assessment"]["nominal_digital_pass"] is False
    assert row["observation"]["data_origin"] == "synthetic_fixture"


@pytest.mark.parametrize(
    "changes",
    [
        {"actual_model": "other-model"},
        {"identity_verified": False},
        {"guide_current": False},
        {"literal_guide": False},
        {"revision_current": False},
        {"files_verified": False},
        {"unresolved": None},
        {"unresolved": 1},
        {"required_failed": 1},
        {"nominal": {}},
    ],
)
def test_missing_or_stale_authority_never_passes(cases, changes):
    assert assess(cases[0], observation(**changes))["nominal_digital_pass"] is False


def test_physical_unknown_does_not_erase_digital_delivery(cases):
    assessed = assess(cases[0], observation(unresolved=2))
    assert assessed["digital_delivery"] is True
    assert assessed["full_task_success"] is None
    assert assessed["physical_readiness"] == "unverified"


def test_unregistered_oracle_cannot_fall_through(cases):
    with pytest.raises(ValueError, match="oracle"):
        assess({**cases[0], "oracle": {"kind": "unknown"}}, observation())
    complex_case = next(case for case in cases if case["oracle"]["kind"] == "unassessed_v1")
    assert assess(complex_case, observation())["nominal_status"] == "unassessed"


def test_clarification_requires_question_and_no_cad(cases):
    case = next(case for case in cases if case["oracle"]["kind"] == "clarification_v1")
    assert assess(case, observation(state="needs_input", questions=1, revisions=0))[
        "clarification_pass"
    ]
    assert not assess(case, observation(state="needs_input", questions=1, revisions=1))[
        "clarification_pass"
    ]
    assert not assess(case, observation(state="needs_input", questions=0, revisions=0))[
        "clarification_pass"
    ]


def test_resume_skips_completed_and_refuses_drift(tmp_path, cases):
    output = tmp_path / "run"
    key = cases[0]["id"]
    adapter = FixtureAdapter({key: observation()})
    kwargs = {"inventory": lambda: {"source": "original"}}
    run_benchmark(output, DEFAULT_CORPUS, [key], adapter, **kwargs)
    before = (output / "events.jsonl").read_bytes()
    run_benchmark(output, DEFAULT_CORPUS, [key], adapter, **kwargs)
    assert (output / "events.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="drift"):
        run_benchmark(
            output, DEFAULT_CORPUS, [key], adapter, inventory=lambda: {"source": "changed"}
        )
    assert (output / "events.jsonl").read_bytes() == before


def test_failures_are_retained_and_not_retried_on_resume(tmp_path, cases):
    class Failure:
        name = "pure_failure"

        def run(self, case, directory, emit):
            emit("failure_detail", reason="retained")
            raise ValueError("Controlled failed adapter")

    output = tmp_path / "run"
    kwargs = {"inventory": lambda: {}}
    rows = run_benchmark(output, DEFAULT_CORPUS, [cases[0]["id"]], Failure(), **kwargs)
    assert len(rows) == 1
    assert "Controlled failed adapter" in rows[0]["error"]
    assert rows[0]["assessment"]["nominal_digital_pass"] is False
    assert len(run_benchmark(output, DEFAULT_CORPUS, [cases[0]["id"]], Failure(), **kwargs)) == 1


def test_no_concurrent_writer_or_implicit_heldout(tmp_path, cases):
    output = tmp_path / "run"
    output.mkdir()
    (output / ".runner.lock").write_text("123")
    adapter = FixtureAdapter({})
    with pytest.raises(FileExistsError):
        run_benchmark(output, DEFAULT_CORPUS, [cases[0]["id"]], adapter)
    assert (output / ".runner.lock").read_text() == "123"
    with pytest.raises(ValueError, match="Split mismatch"):
        run_benchmark(tmp_path / "other", DEFAULT_CORPUS, [cases[-1]["id"]], adapter)
    with pytest.raises(ValueError, match="bounded"):
        run_benchmark(tmp_path / "other", DEFAULT_CORPUS, [], adapter)


def test_interrupted_attempt_is_retained_then_abandoned(tmp_path, cases):
    class Interrupted:
        name = "pure_interrupt"

        def run(self, case, directory, emit):
            raise KeyboardInterrupt

    output = tmp_path / "run"
    kwargs = {"inventory": lambda: {}}
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(output, DEFAULT_CORPUS, [cases[0]["id"]], Interrupted(), **kwargs)
    original = (output / "events.jsonl").read_bytes()
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(output, DEFAULT_CORPUS, [cases[0]["id"]], Interrupted(), **kwargs)
    assert (output / "events.jsonl").read_bytes().startswith(original)
    rows = read_events(output / "events.jsonl")
    assert sum(row["kind"] == "attempt_abandoned" for row in rows) == 1
    assert sum(row["kind"] == "attempt_started" for row in rows) == 2


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":1e999}'])
def test_ambiguous_or_nonfinite_json_refused(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_journal_tampering_and_truncation_refused(tmp_path):
    path = tmp_path / "journal.jsonl"
    Journal(path).append("run", identity={})
    before = path.read_bytes()
    path.write_bytes(before.replace(b'"run"', b'"bad"'))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_events(path)
    path.write_bytes(before[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        read_events(path)


def archive_bytes(name="make/part.step", listed=True, contents=b"exact CAD"):
    data = io.BytesIO()
    record = {name: {"sha256": sha(contents), "size_bytes": len(contents)}} if listed else {}
    with ZipFile(data, "w") as archive:
        archive.writestr(name, contents)
        archive.writestr("manifest.json", encoded({"files": record}))
    raw = data.getvalue()
    return raw, {"sha256": sha(raw), "bytes": len(raw)}


def test_exact_download_and_every_member_identity_required():
    raw, receipt = archive_bytes()
    assert verified_archive(raw, receipt)["make/part.step"] == b"exact CAD"
    with pytest.raises(ValueError, match="receipt"):
        verified_archive(raw + b"changed", receipt)
    for name, listed in [("../outside.step", True), ("make/part.step", False)]:
        raw, receipt = archive_bytes(name, listed)
        with pytest.raises(ValueError):
            verified_archive(raw, receipt)


def test_analytic_oracles_have_independent_formulas_and_hole_occupancy():
    size, volume, inside = analytic_spec(
        {
            "shape": "plate_holes",
            "size_mm": [10, 10, 2],
            "hole_diameter_mm": 2,
            "hole_centers_mm": [[5, 5]],
        }
    )
    assert size == [10, 10, 2]
    assert volume == pytest.approx(200 - 2 * 3.141592653589793)
    assert inside(1, 1, 1)
    assert not inside(5, 5, 1)
    with pytest.raises(ValueError, match="Unsupported"):
        analytic_spec({"shape": "not_implemented"})


def test_frozen_payload_change_is_refused(tmp_path):
    (tmp_path / "cases.json").write_bytes(b"changed")
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"files": {"cases.json": "0" * 64, "ATTRIBUTION.json": "0" * 64}})
    )
    with pytest.raises(ValueError, match="Frozen corpus mismatch"):
        load_corpus(tmp_path)


def test_journal_serializes_worker_and_observer_events(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = Journal(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: journal.append("observed", index=index), range(40)))
    assert len(read_events(path)) == 40


def test_midrun_source_change_retains_failure_and_stops_next_case(tmp_path, cases):
    version = {"source": "before"}

    class Drift:
        name = "pure_drift"

        def run(self, case, directory, emit):
            version["source"] = "after"
            return observation()

    output = tmp_path / "run"
    with pytest.raises(ValueError, match="changed during trial"):
        run_benchmark(
            output,
            DEFAULT_CORPUS,
            [cases[0]["id"], cases[1]["id"]],
            Drift(),
            max_cases=2,
            inventory=lambda: dict(version),
        )
    rows = read_events(output / "events.jsonl")
    finished = [row for row in rows if row["kind"] == "attempt_finished"]
    assert len(finished) == 1
    assert finished[0]["source_unchanged"] is False
    assert not finished[0]["assessment"]["nominal_digital_pass"]


def test_separate_engine_source_and_harness_are_pinned(tmp_path):
    engine = tmp_path / "src/forge_mind"
    engine.mkdir(parents=True)
    (engine / "__init__.py").write_bytes(b"frozen engine")
    inventory = source_inventory(tmp_path)
    assert inventory["engine/src/forge_mind/__init__.py"] == sha(b"frozen engine")
    assert "harness/scripts/customer_outcome_bench.py" in inventory
    assert "harness/scripts/customer_outcome_report.py" in inventory


def source_archive(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    for name in (
        "src/forge_mind/__init__.py",
        "pyproject.toml",
        "uv.lock",
        "Dockerfile",
        "Dockerfile.workspace",
        "scripts/customer_outcome_bench.py",
        "scripts/customer_outcome_report.py",
    ):
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# exact archive fixture\n", encoding="utf-8")
    monkeypatch.setattr(bench, "ROOT", archive)
    for key in tuple(bench.os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    return archive


def test_source_archive_runs_with_exact_hashes_and_no_invented_git(tmp_path, monkeypatch, cases):
    archive = source_archive(tmp_path, monkeypatch)
    called = []

    class ArchiveFixture:
        name = "customer_local"
        source_root = archive

        def run(self, case, directory, emit):
            called.append(case["id"])
            return observation(data_origin="synthetic_fixture", identity_verified=False)

    def inventory():
        return source_inventory(archive)

    rows = run_benchmark(
        tmp_path / "run", DEFAULT_CORPUS, [cases[0]["id"]], ArchiveFixture(), inventory=inventory
    )
    assert called == [cases[0]["id"]]
    saved = Journal(tmp_path / "run/events.jsonl").rows[0]
    assert saved["identity"]["source_files"] == inventory()
    assert saved["git_head"] is None
    assert saved["git_metadata"]["status"] == "unavailable_source_archive"
    assert saved["git_metadata"]["worktree_cleanliness"] == "not_available"
    assert not rows[0]["assessment"]["nominal_digital_pass"]


def test_archive_resume_and_midrun_drift_keep_hash_authority(tmp_path, monkeypatch, cases):
    archive = source_archive(tmp_path, monkeypatch)
    adapter = FixtureAdapter({cases[0]["id"]: observation()})
    output = tmp_path / "run"

    def inventory():
        return source_inventory(archive)

    run_benchmark(output, DEFAULT_CORPUS, [cases[0]["id"]], adapter, inventory=inventory)
    original = (output / "events.jsonl").read_bytes()
    run_benchmark(output, DEFAULT_CORPUS, [cases[0]["id"]], adapter, inventory=inventory)
    assert (output / "events.jsonl").read_bytes() == original
    (archive / "src/forge_mind/__init__.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        run_benchmark(output, DEFAULT_CORPUS, [cases[0]["id"]], adapter, inventory=inventory)
    assert (output / "events.jsonl").read_bytes() == original

    class Drift:
        name = "pure_drift"

        def run(self, case, directory, emit):
            (archive / "src/forge_mind/__init__.py").write_text("changed again", encoding="utf-8")
            return observation()

    with pytest.raises(ValueError, match="changed during trial"):
        run_benchmark(
            tmp_path / "drift", DEFAULT_CORPUS, [cases[0]["id"]], Drift(), inventory=inventory
        )
    retained = Journal(tmp_path / "drift/events.jsonl").rows[-1]
    assert retained["source_unchanged"] is False
    assert not retained["assessment"]["nominal_digital_pass"]


@pytest.mark.parametrize("invented", [False, True])
def test_actual_archive_without_exact_inventory_refuses_before_adapter(
    tmp_path, monkeypatch, cases, invented
):
    archive = source_archive(tmp_path, monkeypatch)

    class NeverRun:
        name = "customer_local"
        source_root = archive

        def run(self, *args):
            pytest.fail("Incomplete source identity must refuse before adapter entry")

    captured = source_inventory(archive) if invented else {}
    if invented:
        captured["engine/src/forge_mind/__init__.py"] = "f" * 64
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="complete exact file-hash inventory"):
        run_benchmark(
            output, DEFAULT_CORPUS, [cases[0]["id"]], NeverRun(), inventory=lambda: captured
        )
    assert not (output / "events.jsonl").exists()
    assert not (output / ".runner.lock").exists()


@pytest.mark.parametrize("marker", ["directory", "worktree_file", "ancestor"])
def test_existing_broken_git_metadata_never_becomes_archive(tmp_path, monkeypatch, marker):
    archive = source_archive(tmp_path, monkeypatch)
    if marker == "worktree_file":
        (archive / ".git").write_text("gitdir: missing-directory\n", encoding="utf-8")
    else:
        ((tmp_path if marker == "ancestor" else archive) / ".git").mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        bench.source_git_metadata(archive)


@pytest.mark.parametrize("name", ["GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"])
def test_git_redirection_never_gains_archive_or_foreign_head(tmp_path, monkeypatch, name):
    archive = source_archive(tmp_path, monkeypatch)
    monkeypatch.setenv(name, str(tmp_path / "foreign"))
    with pytest.raises(ValueError, match="redirection"):
        bench.source_git_metadata(archive)


def test_existing_git_marker_without_executable_still_refuses(tmp_path, monkeypatch):
    archive = source_archive(tmp_path, monkeypatch)
    (archive / ".git").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "no-executables"))
    with pytest.raises(FileNotFoundError):
        bench.source_git_metadata(archive)


def test_real_repo_records_head_without_claiming_cleanliness_and_unborn_refuses(
    tmp_path, monkeypatch
):
    archive = source_archive(tmp_path, monkeypatch)
    empty_template = tmp_path / "empty-template"
    empty_template.mkdir()
    environment = {
        key: value for key, value in bench.os.environ.items() if not key.startswith("GIT_")
    }

    def git(*arguments):
        return subprocess.run(
            [
                "git",
                "-c",
                f"core.hooksPath={empty_template}",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Archive fixture",
                "-c",
                "user.email=fixture@example.invalid",
                *arguments,
            ],
            cwd=archive,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    git("init", "--initial-branch=main", f"--template={empty_template}")
    with pytest.raises(subprocess.CalledProcessError):
        bench.source_git_metadata(archive)
    (archive / "probe.txt").write_text("temporary fixture only\n", encoding="utf-8")
    git("add", "--", "probe.txt")
    git("commit", "--no-gpg-sign", "-m", "Temporary source identity fixture")
    actual = git("rev-parse", "HEAD")
    metadata = bench.source_git_metadata(archive)
    assert metadata["git_head"] == actual
    assert metadata["git_metadata"]["status"] == "available"
    assert metadata["git_metadata"]["worktree_cleanliness"] == "not_checked"


def test_print_archive_uses_its_explicit_member_schema():
    data = io.BytesIO()
    manifest = encoded({"members": {"print/part.stl": {"sha256": sha(b"mesh"), "bytes": 4}}})
    with ZipFile(data, "w") as archive:
        archive.writestr("print/part.stl", b"mesh")
        archive.writestr("manifest.json", manifest)
    raw = data.getvalue()
    receipt = {"sha256": sha(raw), "bytes": len(raw), "manifest_sha256": sha(manifest)}
    assert verified_archive(raw, receipt, kind="print")["print/part.stl"] == b"mesh"
    with pytest.raises(ValueError, match="manifest"):
        verified_archive(raw, {**receipt, "manifest_sha256": "0" * 64}, kind="print")
    with pytest.raises(ValueError, match="Unknown archive"):
        verified_archive(raw, receipt, kind="unsupported")


def test_actual_generation_policy_preflight_refuses_wrong_flag_before_imports(monkeypatch):
    adapter = CustomerLocalAdapter(
        DEFAULT_CORPUS,
        executor="docker",
        seconds=600,
        actions=16,
        condition="cold",
        generation_policy="staged",
    )
    monkeypatch.delenv("FORGE_NATIVE_STAGED_PLANNING", raising=False)
    with pytest.raises(ValueError, match="Generation policy differs"):
        adapter.preflight()
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    adapter.preflight()
    legacy = CustomerLocalAdapter(
        DEFAULT_CORPUS,
        executor="docker",
        seconds=600,
        actions=16,
        condition="cold",
        generation_policy="legacy",
    )
    with pytest.raises(ValueError, match="Generation policy differs"):
        legacy.preflight()


def test_cad_only_delivery_does_not_require_unrequested_meshes():
    members = {"document.json": b"source", "assembly/part.step": b"saved STEP"}
    assert requested_project_files(members, {"include_stl": False}, printable=False)
    assert not requested_project_files(members, {"include_stl": False}, printable=True)
    assert not requested_project_files({"part.step": b"partial"}, {}, printable=False)
    receipt = {
        "include_stl": True,
        "made_parts": [
            {"files": ["make/a.step", "make/a.stl"]},
            {"files": ["make/b.step", "make/b.stl"]},
        ],
    }
    made = {**members, "make/a.step": b"a", "make/a.stl": b"mesh", "make/b.step": b"b"}
    assert not requested_project_files(made, receipt, printable=True)
    assert requested_project_files({**made, "make/b.stl": b"mesh"}, receipt, printable=True)


def test_only_current_automatic_delivery_establishes_handoff():
    binding = {"revision_id": "b" * 32, "source_fingerprint": "e" * 64}
    receipt = {"project_id": "a" * 32, "revision_id": "b" * 32}
    session = {
        "id": "a" * 32,
        "job_id": "c" * 32,
        "selected_revision": "b" * 32,
        "automatic_delivery": {
            "status": "delivered",
            "readback_verified": True,
            "job_id": "c" * 32,
            "receipt": receipt,
            **binding,
        },
    }
    assert current_capture_receipt(session, "b" * 32, binding) == receipt
    for changed in (
        {"status": "captured"},
        {"readback_verified": False},
        {"job_id": "d" * 32},
        {"source_fingerprint": "f" * 64},
    ):
        assert (
            current_capture_receipt(
                {**session, "automatic_delivery": {**session["automatic_delivery"], **changed}},
                "b" * 32,
                binding,
            )
            is None
        )
    assert current_capture_receipt({**session, "automatic_delivery": {}}, "b" * 32, binding) is None


def test_independent_reader_obeys_installed_shape_validity_property(tmp_path, monkeypatch):
    class SyntheticShape:
        is_valid = True
        volume = 240.0

        def solids(self):
            return [self]

        def bounding_box(self):
            return SimpleNamespace(
                min=SimpleNamespace(X=0, Y=0, Z=0), size=SimpleNamespace(X=10, Y=8, Z=3)
            )

        def is_inside(self, point):
            return all(0 < actual < limit for actual, limit in zip(point, (10, 8, 3), strict=True))

    monkeypatch.setitem(
        sys.modules, "build123d", SimpleNamespace(import_step=lambda _: SyntheticShape())
    )
    step = tmp_path / "synthetic.step"
    step.write_bytes(b"Synthetic protocol fixture, not actual STEP geometry")
    observed = measure_step(step, {"shape": "block", "size_mm": [10, 8, 3]})
    assert observed["status"] == "pass"
    assert observed["sample_count"] == 729
