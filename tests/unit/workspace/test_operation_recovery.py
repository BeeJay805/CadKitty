"""Explicit reruns cannot reinterpret operations, change inputs or duplicate dispatch."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import psutil
import pytest

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace import operation_recovery as recovery
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "engine_fingerprint", lambda: "engine")
    monkeypatch.setattr(
        "forge_mind.workspace.engineering_service.engine_fingerprint", lambda: "engine"
    )
    monkeypatch.setattr("forge_mind.workspace.motion_service.engine_fingerprint", lambda: "engine")
    store = Store(tmp_path)
    sid = store.create("owner", "Manual project", {"requirements": []})["id"]
    rid = "a" * 32
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    for name in ("part.step", "topology.json"):
        (directory / name).write_text("owned artifact")
    revision = {
        "id": rid,
        "contract": {"requirements": []},
        "measurement": {
            "geometry_valid": True,
            "engine_fingerprint": "engine",
            "artifact_hashes": {p.name: file_sha256(p) for p in directory.iterdir()},
            "facts": {"motion_rig": {"engine_fingerprint": "engine"}},
        },
    }
    store.add_revision(sid, revision)
    store.update(sid, accepted_revision=rid, selected_revision=rid)
    return store, sid, rid


def interrupted(workspace, operation="review", **extra):
    store, sid, rid = workspace
    payload = {"engineering_operation": operation, "revision_id": rid, **extra}
    jid = store.enqueue(sid, "owner", payload)
    assert store.claim()["id"] == jid
    with store.connect() as con:
        con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (jid,))
    assert store.claim() is None
    return jid


@pytest.mark.parametrize("operation", ["review", "motion_check"])
def test_rerun_preserves_original_operation_payload_and_design(workspace, operation):
    store, sid, rid = workspace
    extra = {"max_seconds": 77}
    if operation == "motion_check":
        extra["settings"] = {"samples": 9}
    original = interrupted(workspace, operation, **extra)
    before = store.get(sid, "owner")
    descriptor = before["operation_recovery"]
    assert descriptor["available"] is True and descriptor["operation"] == operation
    with store.connect() as con:
        saved = con.execute("SELECT payload FROM jobs WHERE id=?", (original,)).fetchone()[0]
    retry = store.enqueue(sid, "owner", {"retry_operation": original})
    assert retry != original
    job = store.claim()
    assert job["id"] == retry and job["recovery_of"] == original
    assert job["engineering_operation"] == operation and job["revision_id"] == rid
    assert job["max_seconds"] == 77
    assert "resume" not in job and "preview_only" not in job
    if operation == "motion_check":
        assert job["settings"] == extra["settings"]
    after = store.get(sid, "owner")
    for key in ("selected_revision", "accepted_revision", "contract", "proposal"):
        assert before.get(key) == after.get(key)
    with store.connect() as con:
        assert (
            con.execute("SELECT payload FROM jobs WHERE id=?", (original,)).fetchone()[0] == saved
        )


def test_concurrent_repeated_retry_creates_one_linked_attempt(workspace):
    store, sid, _ = workspace
    original = interrupted(workspace)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = list(
            pool.map(lambda _: store.enqueue(sid, "owner", {"retry_operation": original}), range(3))
        )
    assert len(set(jobs)) == 1
    with store.connect() as con:
        assert (
            con.execute("SELECT count(*) FROM jobs WHERE recovery_of=?", (original,)).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "change",
    [
        "head",
        "accepted",
        "input_version",
        "input_contents",
        "sources",
        "engine",
        "contract",
        "revision",
        "artifact",
    ],
)
def test_changed_admission_inputs_block_rerun(workspace, monkeypatch, change):
    store, sid, rid = workspace
    original = interrupted(workspace)
    if change == "head":
        store.update(sid, selected_revision="b" * 32)
    elif change == "accepted":
        store.update(sid, accepted_revision="b" * 32)
    elif change == "contract":
        store.update(sid, contract={"requirements": ["changed"]})
    elif change.startswith("input_"):
        inputs = store.engineering(sid)["inputs"]
        with store.connect() as con:
            con.execute(
                "INSERT INTO engineering_inputs VALUES (?,?,?)",
                (
                    sid,
                    1 if change == "input_version" else 0,
                    json.dumps(inputs if change == "input_version" else {"changed": True}),
                ),
            )
    elif change == "sources":
        monkeypatch.setattr(recovery, "source_fingerprint", lambda *args: "changed")
    elif change == "engine":
        monkeypatch.setattr(recovery, "engine_fingerprint", lambda: "changed")
    elif change == "revision":
        with store.connect() as con:
            revision = json.loads(
                con.execute("SELECT document FROM revisions WHERE id=?", (rid,)).fetchone()[0]
            )
            revision["contract"] = {"changed": True}
            con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(revision), rid))
    else:
        (store.directory(sid, rid) / "part.step").write_text("tampered geometry")
    with pytest.raises(WorkspaceError, match="changed"):
        store.enqueue(sid, "owner", {"retry_operation": original})
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


@pytest.mark.parametrize(
    "operation", ["product_tool", "structural_study", "design_study", "unknown", "document"]
)
def test_unsupported_interruption_cannot_fall_through_to_saved_author_resume(workspace, operation):
    store, sid, rid = workspace
    extra = {"check_id": "b" * 32, "study_id": "c" * 32}
    if operation == "document":
        payload = {
            "document": {"name": "original typed document"},
            "revision_id": rid,
            "preview_only": True,
        }
        jid = store.enqueue(sid, "owner", payload)
        assert store.claim()
        with store.connect() as con:
            con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (jid,))
        assert store.claim() is None
    else:
        jid = interrupted(workspace, operation, **extra)
    assert store.get(sid, "owner")["operation_recovery"]["available"] is False
    store.update(sid, resume_candidate={"revision_id": rid, "based_on": rid})
    with pytest.raises(WorkspaceError, match="cannot resume as authoring"):
        store.enqueue(sid, "owner", {"resume": True, "preview_only": True})
    with pytest.raises(WorkspaceError, match="supported frozen"):
        store.enqueue(sid, "owner", {"retry_operation": jid})


@pytest.mark.parametrize(
    "cause", ["wrong_owner", "superseded", "live", "unknown", "cancelled", "replacement"]
)
def test_retry_admission_guards(workspace, cause):
    store, sid, _ = workspace
    original = interrupted(workspace)
    owner = "intruder" if cause == "wrong_owner" else "owner"
    request = {"retry_operation": original}
    if cause == "superseded":
        store.enqueue(sid, "owner", {})
    elif cause == "replacement":
        request["engineering_operation"] = "review"
    elif cause in {"live", "unknown", "cancelled"}:
        with store.connect() as con:
            if cause == "live":
                con.execute(
                    "UPDATE jobs SET worker_created=? WHERE id=?",
                    (psutil.Process().create_time(), original),
                )
            elif cause == "unknown":
                con.execute("UPDATE jobs SET worker_host='another-host' WHERE id=?", (original,))
            else:
                con.execute("UPDATE jobs SET cancel=1 WHERE id=?", (original,))
    with pytest.raises(WorkspaceError):
        store.enqueue(sid, owner, request)


@pytest.mark.parametrize("unknown", [False, True])
def test_surviving_or_unknown_native_worker_blocks_replacement(workspace, unknown):
    store, sid, _ = workspace
    original = interrupted(workspace)
    directory = store.root / "reports" / "owned"
    directory.mkdir(parents=True)
    with store.connect() as con:
        identity = {
            "worker_pid": psutil.Process().pid,
            "worker_created": psutil.Process().create_time(),
            "worker_host": "unknown-host" if unknown else recovery.socket.gethostname(),
        }
        con.execute(
            "UPDATE jobs SET native_workers=? WHERE id=?", (json.dumps([identity]), original)
        )
    with pytest.raises(WorkspaceError, match="native worker"):
        store.enqueue(sid, "owner", {"retry_operation": original})


def test_payload_tampering_is_not_rebound_on_recovery(workspace):
    store, sid, _ = workspace
    original = interrupted(workspace)
    with store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (original,)).fetchone()[0]
        )
        payload["max_seconds"] = 999
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), original))
    with pytest.raises(WorkspaceError, match="payload changed"):
        store.enqueue(sid, "owner", {"retry_operation": original})


def test_worker_rechecks_inputs_after_retry_admission(workspace):
    store, sid, rid = workspace
    original = interrupted(workspace)
    store.enqueue(sid, "owner", {"retry_operation": original})
    job = store.claim()
    store.update(sid, accepted_revision="b" * 32)
    with pytest.raises(WorkspaceError, match="changed"):
        recovery.guard_job(SimpleNamespace(store=store), job)
    assert store.get(sid, "owner")["selected_revision"] == rid


def test_report_publication_rechecks_frozen_inputs(workspace, monkeypatch):
    from forge_mind.workspace.engineering_service import run_report_job

    store, sid, _ = workspace
    original = interrupted(workspace)
    store.enqueue(sid, "owner", {"retry_operation": original})
    job = store.claim()
    service = SimpleNamespace(
        store=store, executor=SimpleNamespace(mode="trusted_local"), _load_revision=store.revision
    )

    def process(*args, **kwargs):
        store.update(sid, contract={"changed_during_computation": True})
        return {}

    monkeypatch.setattr("forge_mind.workspace.engineering_service.process_revision", process)
    with pytest.raises(WorkspaceError, match="changed"):
        run_report_job(service, job, lambda: False)
    assert not store.get(sid, "owner").get("reports")


def launch(workspace):
    store, sid, rid = workspace
    jid = store.enqueue(sid, "owner", {"engineering_operation": "review", "revision_id": rid})
    job = store.claim()
    directory = store.root / "reports" / "owned"
    context = recovery.worker_arguments(SimpleNamespace(store=store), job, directory)[
        "operation_job"
    ]
    with store.connect() as con:
        assert con.execute("SELECT operation_directory FROM jobs WHERE id=?", (jid,)).fetchone()[
            0
        ] == str(directory.resolve())
    return job, directory, context


def test_worker_directory_precedes_launch_and_child_registers_before_work(workspace, monkeypatch):
    store, _, _ = workspace
    job, directory, context = launch(workspace)
    parent = psutil.Process()
    created = parent.create_time() + 1
    child = SimpleNamespace(pid=987654, parents=lambda: [parent], create_time=lambda: created)
    monkeypatch.setattr(recovery, "psutil", SimpleNamespace(Process=lambda: child))
    recovery.register_child(directory, context)
    with store.connect() as con:
        assert json.loads(
            con.execute("SELECT native_workers FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
        ) == [
            {
                "worker_pid": child.pid,
                "worker_created": created,
                "worker_host": recovery.socket.gethostname(),
            }
        ]
    with pytest.raises(WorkspaceError, match="already registered"):
        recovery.register_child(directory, context)


@pytest.mark.parametrize(
    "failure",
    ["late", "wrong_parent", "wrong_directory", "changed_payload", "cancelled", "launch_token"],
)
def test_child_cannot_start_for_interrupted_or_mismatched_launch(workspace, monkeypatch, failure):
    store, sid, _ = workspace
    job, directory, context = launch(workspace)
    if failure == "late":
        with store.connect() as con:
            con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (job["id"],))
        assert store.claim() is None
        # The retry can be admitted safely while the original child is still importing.
        store.enqueue(sid, "owner", {"retry_operation": job["id"]})
    elif failure == "cancelled":
        store.cancel(sid, "owner")
    elif failure == "wrong_directory":
        directory = store.root / "unrelated"
    elif failure == "changed_payload":
        context["payload_hash"] = "changed"
    elif failure == "launch_token":
        context["launch_token"] = "another-launch"
    else:
        parent = psutil.Process()
        child = SimpleNamespace(parents=lambda: [], create_time=lambda: parent.create_time() + 1)
        monkeypatch.setattr(recovery, "psutil", SimpleNamespace(Process=lambda: child))
    with pytest.raises(WorkspaceError):
        recovery.register_child(directory, context)
    with store.connect() as con:
        assert (
            con.execute("SELECT native_workers FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            is None
        )


@pytest.mark.parametrize("change", ["settings", "remove_binding", "revision"])
def test_dispatched_payload_must_match_stored_frozen_payload(workspace, change):
    store, sid, rid = workspace
    store.enqueue(
        sid,
        "owner",
        {"engineering_operation": "motion_check", "revision_id": rid, "settings": {"samples": 10}},
    )
    job = store.claim()
    if change == "settings":
        job["settings"]["samples"] = 99
    elif change == "remove_binding":
        job.pop("operation_binding")
    else:
        job["revision_id"] = "b" * 32
    with pytest.raises(WorkspaceError, match="Dispatched operation differs"):
        recovery.guard_job(SimpleNamespace(store=store), job)


def test_publication_holds_write_transaction_through_freshness_check(workspace, monkeypatch):
    store, sid, rid = workspace
    store.enqueue(sid, "owner", {"engineering_operation": "review", "revision_id": rid})
    job = store.claim()
    original_guard = recovery._guard
    observed = []

    def checked(con, *args):
        observed.append(con.in_transaction)
        return original_guard(con, *args)

    monkeypatch.setattr(recovery, "_guard", checked)
    recovery.publish(SimpleNamespace(store=store), job, reports=[{"id": "new-report"}])
    assert observed == [True]
    assert store.get(sid, "owner")["reports"] == [{"id": "new-report"}]


def test_unbound_legacy_worker_arguments_remain_compatible(workspace):
    store, sid, rid = workspace
    service = SimpleNamespace(store=store)
    job = {"session": sid, "owner": "owner", "revision_id": rid}
    assert recovery.worker_arguments(service, job, store.root / "legacy") == {}
    recovery.guard_job(service, job)


@pytest.mark.parametrize("rejected", [False, True])
def test_processing_registers_bound_child_before_dispatch(tmp_path, monkeypatch, rejected):
    from forge_mind.workspace import processing

    monkeypatch.chdir(tmp_path)
    request = tmp_path / "processing.json"
    request.write_text(
        json.dumps({"operation": "engineering", "_operation_job": {"job_id": "owned"}})
    )
    events = []

    def register(directory, context):
        assert directory == tmp_path and context == {"job_id": "owned"}
        events.append("registered")
        if rejected:
            raise WorkspaceError("Original operation ended")

    def compute(directory, payload):
        assert "_operation_job" not in payload
        events.append("computed")
        return {}

    monkeypatch.setattr(recovery, "register_child", register)
    monkeypatch.setattr("forge_mind.workspace.engineering.run_engineering", compute)
    if rejected:
        with pytest.raises(WorkspaceError, match="ended"):
            processing.main(request)
        assert events == ["registered"]
        assert not (tmp_path / "processed.json").exists()
    else:
        processing.main(request)
        assert events == ["registered", "computed"]


def test_processing_parent_passes_registration_metadata_without_native_launch(
    tmp_path, monkeypatch
):
    from forge_mind.workspace import processing

    def bounded(command, directory, timeout, cancelled):
        payload = json.loads((directory / "processing.json").read_text())
        assert payload["_operation_job"] == {"job_id": "owned"}
        (directory / "processed.json").write_text("{}")

    monkeypatch.setattr(processing, "bounded_process", bounded)
    assert (
        processing.process_revision(
            tmp_path,
            {"operation": "engineering"},
            None,
            lambda: False,
            operation_job={"job_id": "owned"},
        )
        == {}
    )


def test_native_queued_review_rerun_registers_real_windows_child(tmp_path, monkeypatch):
    from tests.unit.workspace.test_engineering_workspace import create_revision

    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Contract

    store = Store(tmp_path)
    service = WorkspaceService(store, Executor("trusted_local"))
    monkeypatch.setattr(
        service, "_run_agent_job", lambda *args: pytest.fail("Rerun reached model authoring")
    )
    sid = store.create("owner", "Native review recovery", Contract().model_dump())["id"]
    revision = create_revision(service, sid)
    assert revision["measurement"]["geometry_valid"], revision["measurement"]["errors"]
    rid = revision["id"]
    store.update(sid, accepted_revision=rid, selected_revision=rid)
    original = interrupted((store, sid, rid))
    retry = store.enqueue(sid, "owner", {"retry_operation": original})
    job = store.claim()
    service.run_job(job)
    session = store.get(sid, "owner")
    assert session["state"] == "needs_review", session["events"][-3:]
    assert len(session["reports"]) == 1
    assert session["accepted_revision"] == session["selected_revision"] == rid
    with store.connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (retry,)).fetchone()
    assert row["recovery_of"] == original
    assert json.loads(row["native_workers"])[0]["worker_pid"] != psutil.Process().pid
