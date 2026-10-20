from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from forge_mind.workspace.hosting import HostingError, HostLock
from forge_mind.workspace.hosting_backup import fenced_recovery_plan, restore, snapshot, verify


def test_snapshot_tolerates_disappearing_sqlite_sidecar_only(tmp_path, monkeypatch):
    root = durable(tmp_path)
    sidecar = root / "workspace.sqlite3-shm"
    original_rglob, original_lstat = Path.rglob, Path.lstat

    def enumerated(directory, pattern):
        paths = list(original_rglob(directory, pattern))
        # SQLite can remove an already-enumerated SHM when its last connection closes.
        return iter([*paths, sidecar] if directory == root else paths)

    def vanished(path, *args, **kwargs):
        if path == sidecar:
            raise FileNotFoundError("SQLite removed its transient SHM")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", enumerated)
    monkeypatch.setattr(Path, "lstat", vanished)
    captured = tmp_path / "backup"
    snapshot(root, captured)
    manifest = verify(captured)
    assert "workspace.sqlite3" in manifest["files"]
    assert not {"workspace.sqlite3-wal", "workspace.sqlite3-shm"} & manifest["files"].keys()
    assert (captured / "data/part.step").read_bytes() == b"exact immutable CAD"


def test_snapshot_reads_committed_wal_through_sqlite(tmp_path):
    root = durable(tmp_path)
    # Hold a real connection open so committed data remains in WAL during capture.
    connection = sqlite3.connect(root / "workspace.sqlite3")
    try:
        connection.execute("INSERT INTO jobs VALUES ('wal-job','running','old-host')")
        connection.commit()
        assert (root / "workspace.sqlite3-wal").stat().st_size > 0
        captured = tmp_path / "backup"
        manifest = snapshot(root, captured)
        assert {row["id"] for row in manifest["running_jobs"]} == {"job1", "wal-job"}
        assert verify(captured) == manifest
    finally:
        connection.close()


@pytest.mark.parametrize("name", ["workspace.sqlite3-wal", "workspace.sqlite3-shm"])
def test_snapshot_refuses_unsafe_sqlite_sidecar(tmp_path, name):
    import gc
    import os

    root = durable(tmp_path)
    gc.collect()
    outside = tmp_path / "outside"
    outside.write_bytes(b"must not read as SQLite state")
    sidecar = root / name
    assert not sidecar.exists()
    os.link(outside, sidecar)
    with pytest.raises(HostingError, match="linked or special"):
        snapshot(root, tmp_path / "backup")
    assert not (tmp_path / "backup/manifest.json").exists()


def test_snapshot_does_not_ignore_disappearing_artifact(tmp_path, monkeypatch):
    root = durable(tmp_path)
    original_lstat = Path.lstat

    def vanished(path, *args, **kwargs):
        if path == root / "part.step":
            raise FileNotFoundError("artifact disappeared")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", vanished)
    with pytest.raises(FileNotFoundError, match="artifact disappeared"):
        snapshot(root, tmp_path / "backup")
    assert not (tmp_path / "backup/manifest.json").exists()


def test_snapshot_refuses_artifact_changes_during_copy(tmp_path, monkeypatch):
    from forge_mind.workspace import hosting_backup

    root = durable(tmp_path)
    original_copy = hosting_backup.shutil.copyfile

    def changed(source, destination):
        result = original_copy(source, destination)
        if source == root / "part.step":
            source.write_bytes(b"changed immutable CAD")
        return result

    monkeypatch.setattr(hosting_backup.shutil, "copyfile", changed)
    with pytest.raises(HostingError, match="changed during offline snapshot"):
        snapshot(root, tmp_path / "backup")
    assert not (tmp_path / "backup/manifest.json").exists()


@pytest.mark.parametrize("fault", ["corrupted", "truncated", "missing"])
def test_snapshot_refuses_copied_artifacts_different_from_source(tmp_path, monkeypatch, fault):
    from forge_mind.workspace import hosting_backup

    root = durable(tmp_path)
    expected = (root / "part.step").read_bytes()
    original_copy = hosting_backup.shutil.copyfile

    def damaged_copy(source, destination):
        result = original_copy(source, destination)
        if source == root / "part.step":
            if fault == "missing":
                destination.unlink()
            else:
                destination.write_bytes(
                    b"x" * len(expected) if fault == "corrupted" else expected[:4]
                )
        return result

    monkeypatch.setattr(hosting_backup.shutil, "copyfile", damaged_copy)
    captured = tmp_path / "backup"
    with pytest.raises(HostingError, match="do not match the source inventory"):
        snapshot(root, captured)
    assert (root / "part.step").read_bytes() == expected
    assert not (captured / "manifest.json").exists()
    with pytest.raises(HostingError, match="manifest is missing"):
        verify(captured)


def test_snapshot_refuses_directory_in_sqlite_sidecar_location(tmp_path):
    import gc

    root = durable(tmp_path)
    gc.collect()
    (root / "workspace.sqlite3-shm").mkdir()
    with pytest.raises(HostingError, match="linked or special"):
        snapshot(root, tmp_path / "backup")
    assert not (tmp_path / "backup/manifest.json").exists()


def durable(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    with sqlite3.connect(root / "workspace.sqlite3") as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE jobs (id TEXT, state TEXT, worker_host TEXT)")
        con.execute("INSERT INTO jobs VALUES ('job1','running','old-host')")
    (root / "part.step").write_bytes(b"exact immutable CAD")
    return root


def test_snapshot_restore_preserve_exact_artifacts_and_foreign_host_authority(tmp_path):
    root = durable(tmp_path)
    captured = tmp_path / "backup"
    snapshot(root, captured)
    assert verify(captured)["running_jobs"] == [{"id": "job1", "worker_host": "old-host"}]
    restored = tmp_path / "restored"
    receipt = restore(captured, restored)
    assert receipt["requires_fenced_recovery"]
    assert (restored / "part.step").read_bytes() == (root / "part.step").read_bytes()
    with sqlite3.connect(restored / "workspace.sqlite3") as con:
        assert con.execute("SELECT state FROM jobs").fetchone()[0] == "running"
    with pytest.raises(HostingError, match="must not exist"):
        restore(captured, restored)


def test_live_host_or_tampered_snapshot_cannot_be_restored(tmp_path):
    root = durable(tmp_path)
    with HostLock(root), pytest.raises(HostingError, match="already serving"):
        snapshot(root, tmp_path / "blocked")
    captured = tmp_path / "backup"
    snapshot(root, captured)
    (captured / "data/part.step").write_bytes(b"tampered")
    with pytest.raises(HostingError, match="manifest"):
        restore(captured, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_fencing_attestation_requires_exact_host_and_all_job_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "old-host")
    root = durable(tmp_path)
    captured = tmp_path / "backup"
    snapshot(root, captured)
    restored = tmp_path / "restored"
    restore(captured, restored)
    evidence = tmp_path / "fence.json"
    evidence.write_text(
        json.dumps(
            {
                "old_host": "old-host",
                "fenced": True,
                "operator": "operator",
                "evidence": "VM destroyed and original volume detached",
            }
        )
    )
    with pytest.raises(HostingError, match="every exact"):
        fenced_recovery_plan(restored, old_host="old-host", job_ids=[], fencing_evidence=evidence)
    result = fenced_recovery_plan(
        restored, old_host="old-host", job_ids=["job1"], fencing_evidence=evidence
    )
    assert result["action"] == "interrupt_using_original_store_checkpoint_transition"
    evidence.write_text(json.dumps({"old_host": "old-host", "fenced": False}))
    with pytest.raises(HostingError, match="fencing evidence"):
        fenced_recovery_plan(
            restored, old_host="old-host", job_ids=["job1"], fencing_evidence=evidence
        )
