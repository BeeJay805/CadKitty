"""Offline complete-store snapshots and non-destructive verified restoration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sqlite3
import stat
import uuid
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Any

from forge_mind.workspace.hosting import HostingError, HostLock, write_json

_DATABASE_FILES = frozenset({"workspace.sqlite3", "workspace.sqlite3-wal", "workspace.sqlite3-shm"})


def _validate_database_files(root: Path) -> None:
    """SQLite owns sidecar lifetime; optional absence never permits unsafe entries."""
    for name in _DATABASE_FILES:
        try:
            info = (root / name).lstat()
        except FileNotFoundError:
            if name == "workspace.sqlite3":
                raise
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise HostingError("Store inventory contains a linked or special database file")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _replace_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def current_restore(root: Path, expected_id: str | None = None) -> dict[str, Any]:
    """Read the current immutable restore identity; copied history grants no authority."""
    pointer = json.loads((root / ".restore-current.json").read_text())
    restore_id = pointer.get("restore_id", "")
    if (
        not isinstance(restore_id, str)
        or len(restore_id) != 32
        or any(c not in "0123456789abcdef" for c in restore_id)
        or (expected_id is not None and expected_id != restore_id)
    ):
        raise HostingError("Recovery does not target the current restore identity")
    identity = json.loads((root / ".restore-history" / restore_id / "identity.json").read_text())
    if identity.get("restore_id") != restore_id or _json_digest(identity) != pointer.get(
        "identity_sha256"
    ):
        raise HostingError("Restore identity history does not match the current pointer")
    marker = root / ".restore-required.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise HostingError("Current restore marker differs from its immutable identity")
    result: dict[str, Any] = identity
    return result


def _sync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.name != ".host.lock":
            with path.open("r+b") as source:
                os.fsync(source.fileno())
    if os.name != "nt":
        for directory in [*sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True), root]:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def relative_file(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or ":" in name
        or "\\" in name
        or path.as_posix() != name
    ):
        raise HostingError("Invalid relative file name")
    return path


def inventory(
    root: Path, *, maximum: int = 2**63 - 1, exclude: frozenset[str] = frozenset()
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if path.relative_to(root).as_posix() in exclude:
            continue
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise HostingError("Store inventory contains a linked or special file")
        name = path.relative_to(root).as_posix()
        relative_file(name)
        total += info.st_size
        if total > maximum or len(result) >= 16384:
            raise HostingError("File inventory exceeds its resource limit")
        checksum = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                checksum.update(block)
        digest = checksum.hexdigest()
        after = path.stat()
        if (info.st_size, info.st_mtime_ns, info.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise HostingError("File changed while reading its inventory")
        result[name] = {"bytes": info.st_size, "sha256": digest}
    return result


def snapshot(root: Path, destination: Path) -> dict[str, Any]:
    root, destination = root.resolve(), destination.absolute()
    if root == destination or root in destination.parents or destination in root.parents:
        raise HostingError("Snapshot must be outside the durable workspace")
    if destination.exists():
        raise HostingError("Snapshot destination already exists")
    with HostLock(root):
        # Only SQLite backs up database bytes. Its WAL/SHM may disappear when an
        # unrelated connection closes, even while our application lock is held.
        # Validate their layout without hashing their transient contents.
        excluded = _DATABASE_FILES | {".host.lock"}
        _validate_database_files(root)
        original = inventory(root, exclude=excluded)
        destination.mkdir(parents=True, mode=0o700)
        try:
            data = destination / "data"
            data.mkdir(mode=0o700)
            for name in original:
                source, target = root / name, data / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            with (
                closing(
                    sqlite3.connect(f"file:{root / 'workspace.sqlite3'}?mode=ro", uri=True)
                ) as source_db,
                closing(sqlite3.connect(data / "workspace.sqlite3")) as target_db,
            ):
                source_db.backup(target_db)
                # A portable snapshot is one complete database file. Retaining
                # WAL mode would let read-only verification create new sidecars.
                if target_db.execute("PRAGMA journal_mode=DELETE").fetchone() != ("delete",):
                    raise HostingError("Snapshot database could not become self-contained")
                if target_db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise HostingError("Snapshot database failed integrity check")
                running = [
                    dict(zip(("id", "worker_host"), row, strict=True))
                    for row in target_db.execute(
                        "SELECT id,worker_host FROM jobs WHERE state='running'"
                    )
                ]
                hold_rows: list[dict[str, Any]] = []
                if target_db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='host_runner_holds'"
                ).fetchone():
                    has_evidence = (
                        target_db.execute(
                            "SELECT 1 FROM sqlite_master WHERE name='host_runner_hold_evidence'"
                        ).fetchone()
                        is not None
                    )
                    for hold_id, reason in target_db.execute(
                        "SELECT hold_id,reason FROM host_runner_holds"
                    ):
                        evidence = (
                            target_db.execute(
                                "SELECT document FROM host_runner_hold_evidence WHERE hold_id=?",
                                (hold_id,),
                            ).fetchone()
                            if has_evidence
                            else None
                        )
                        hold_rows.append(
                            {
                                "hold_id": hold_id,
                                "reason": reason,
                                "evidence": json.loads(evidence[0]) if evidence else None,
                            }
                        )
            _validate_database_files(root)
            after = inventory(root, exclude=excluded)
            if after != original:
                raise HostingError("Workspace changed during offline snapshot")
            copied = inventory(data)
            if set(copied) & _DATABASE_FILES != {"workspace.sqlite3"}:
                raise HostingError("Snapshot requires one standalone database without sidecars")
            if {
                name: value for name, value in copied.items() if name != "workspace.sqlite3"
            } != original:
                raise HostingError("Copied snapshot artifacts do not match the source inventory")
            manifest = {
                "version": 1,
                "host": socket.gethostname(),
                "files": copied,
                "running_jobs": running,
                "runner_holds": hold_rows,
            }
            _sync_tree(data)
            write_json(destination / "manifest.json", manifest)
            return manifest
        except BaseException:
            # Preserve failed captures for diagnosis; an absent manifest never verifies.
            raise


def verify(snapshot_root: Path) -> dict[str, Any]:
    if (snapshot_root / "data").is_symlink():
        raise HostingError("Snapshot data directory must not be linked")
    manifest_path = snapshot_root / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > 16 * 1024 * 1024
    ):
        raise HostingError("Snapshot manifest is missing or linked")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise HostingError("Unsupported snapshot manifest")
    if inventory(snapshot_root / "data") != manifest["files"]:
        raise HostingError("Snapshot contents do not match their manifest")
    if "workspace.sqlite3" not in manifest["files"]:
        raise HostingError("Snapshot database is missing")
    with closing(
        sqlite3.connect(f"file:{snapshot_root / 'data/workspace.sqlite3'}?mode=ro", uri=True)
    ) as con:
        if con.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise HostingError("Snapshot database failed integrity check")
    return manifest


def restore(snapshot_root: Path, destination: Path) -> dict[str, Any]:
    """Restore to a new root only; serving remains blocked pending fenced recovery."""
    manifest = verify(snapshot_root)
    if destination.exists() or destination.is_symlink():
        raise HostingError("Restore destination must not exist")
    destination.mkdir(parents=True, mode=0o700)
    with HostLock(destination):
        incomplete = destination / ".restore-incomplete.json"
        write_json(incomplete, {"version": 1, "state": "incomplete_restore"})
        for name in manifest["files"]:
            relative_file(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot_root / "data" / name, target)
        actual = inventory(
            destination, exclude=frozenset({".host.lock", ".restore-incomplete.json"})
        )
        actual.pop(".host.lock", None)
        if actual != manifest["files"]:
            raise HostingError("Restored files do not match the snapshot")
        receipt = {
            "version": 1,
            "restore_id": uuid.uuid4().hex,
            "source_host": manifest["host"],
            "restored_host": socket.gethostname(),
            "running_jobs": manifest["running_jobs"],
            "runner_holds": manifest.get("runner_holds", []),
            "requires_fenced_recovery": True,
            "manifest_sha256": hashlib.sha256(
                (snapshot_root / "manifest.json").read_bytes()
            ).hexdigest(),
        }
        history = destination / ".restore-history" / receipt["restore_id"]
        history.mkdir(parents=True)
        write_json(history / "identity.json", receipt)
        _replace_json(
            destination / ".restore-current.json",
            {"restore_id": receipt["restore_id"], "identity_sha256": _json_digest(receipt)},
        )
        _sync_tree(destination)
        write_json(destination / ".restore-required.json", receipt)
        incomplete.unlink()
    return receipt


def _fenced_plan_locked(
    root: Path, *, old_host: str, job_ids: list[str], fencing_evidence: Path
) -> dict[str, Any]:
    receipt = current_restore(root)
    if receipt["source_host"] != old_host:
        raise HostingError("Recovery host differs from the restored authority")
    if (
        fencing_evidence.is_symlink()
        or not fencing_evidence.is_file()
        or fencing_evidence.stat().st_size > 65536
    ):
        raise HostingError("Fencing evidence must be an operator-owned regular file")
    evidence = json.loads(fencing_evidence.read_text(encoding="utf-8"))
    if (
        evidence.get("old_host") != old_host
        or evidence.get("fenced") is not True
        or not evidence.get("operator")
        or not evidence.get("evidence")
    ):
        raise HostingError("Explicit old-host fencing evidence is required")
    expected = receipt.get("running_jobs", [])
    expected_ids = {row["id"] for row in expected}
    if (
        len(job_ids) != len(set(job_ids))
        or set(job_ids) != expected_ids
        or any(row["worker_host"] != old_host for row in expected)
    ):
        raise HostingError("Recovery must bind every exact original-host running job")
    with sqlite3.connect(f"file:{root / 'workspace.sqlite3'}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT id,state,worker_host FROM jobs").fetchall()
    selected = [row for row in rows if row[0] in expected_ids]
    if (
        {row[0] for row in selected} != expected_ids
        or any(row[1] not in {"running", "interrupted"} or row[2] != old_host for row in selected)
        or any(row[1] == "running" and row[0] not in expected_ids for row in rows)
    ):
        raise HostingError("Recovery must bind every exact original-host running job")
    return {
        "restore_id": receipt["restore_id"],
        "old_host": old_host,
        "job_ids": sorted(job_ids),
        "fencing_evidence_sha256": hashlib.sha256(fencing_evidence.read_bytes()).hexdigest(),
        "restore_manifest_sha256": receipt["manifest_sha256"],
        "action": "interrupt_using_original_store_checkpoint_transition",
    }


def fenced_recovery_plan(
    root: Path, *, old_host: str, job_ids: list[str], fencing_evidence: Path
) -> dict[str, Any]:
    """Bind exact abandoned work to an operator fencing attestation, not hardware proof."""
    with HostLock(root):
        return _fenced_plan_locked(
            root, old_host=old_host, job_ids=job_ids, fencing_evidence=fencing_evidence
        )
