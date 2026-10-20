"""Cheap change detection for study polling, never an evidence-validity receipt."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import stat
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.design_study_service import (
    _candidates,
    _decode,
    _study,
    candidate_directory,
)
from forge_mind.workspace.document import digest
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.product_sources import source_fingerprint
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def file_identity(path: Path) -> dict[str, Any]:
    """Metadata detects ordinary changes without reading multi-gigabyte solver artifacts."""
    try:
        info = path.lstat()
        return {
            "exists": True,
            "resolved": str(path.resolve()),
            "regular": stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink == 1,
            "mode": info.st_mode,
            "links": info.st_nlink,
            "device": info.st_dev,
            "inode": info.st_ino,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }
    except OSError as exc:
        return {"exists": False, "error": exc.errno}


def _artifact_identity(directory: Path, name: str) -> dict[str, Any]:
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise WorkspaceError("Design evaluation artifact path escapes its directory")
    return file_identity(path)


def _instruments(study: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for evaluator in study["spec"]["evaluators"]:
        kind = evaluator["kind"]
        packages = (
            ("gmsh",)
            if kind == "flow"
            else ("scikit-fem", "scipy", "gmsh")
            if kind == "structure"
            else ()
        )
        for package in packages:
            try:
                result[package] = version(package)
            except PackageNotFoundError:
                result[package] = "unavailable"
        if kind == "flow":
            from forge_mind.workspace.flow_solver import solver_executable

            try:
                result["flow_solver"] = file_identity(solver_executable())
            except (ValueError, OSError):
                result["flow_solver"] = {"exists": False}
    return result


def _diagnostic_file(path: Path, root: Path) -> bool:
    if not path.resolve().is_relative_to(root.resolve()):
        return False
    info = path.lstat()
    return stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink == 1


def _iteration(stage_directory: Any, root: Path) -> int | None:
    """Read only a bounded header/tail; solver iteration is diagnostic, never convergence."""
    if not isinstance(stage_directory, str):
        return None
    path = root / stage_directory / "history.csv"
    if not _diagnostic_file(path, root):
        return None
    with path.open("rb") as stream:
        header = stream.readline(4096)
        if not header.endswith(b"\n"):
            return None
        columns = next(csv.reader([header.decode("utf-8")], skipinitialspace=True))
        index = [column.strip() for column in columns].index("Inner_Iter")
        size = stream.seek(0, 2)
        offset = max(len(header), size - 65536)
        stream.seek(offset)
        tail = stream.read(65536)
        lines = tail.decode("utf-8").splitlines()
        if tail and not tail.endswith((b"\n", b"\r")):
            lines = lines[:-1]  # solver may currently be appending this row
        if offset > len(header):
            lines = lines[1:]  # the first tail row may be partial
        for line in reversed(lines):
            row = next(csv.reader([line], skipinitialspace=True))
            if len(row) != len(columns):
                continue
            try:
                value = float(row[index])
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0 and value.is_integer():
                return int(value)
    return None


def _flow_progress(
    service: WorkspaceService, sid: str, study_id: str, candidate: dict[str, Any]
) -> dict[str, Any] | None:
    root = candidate_directory(service, sid, study_id, candidate["id"])
    path = root / "flow-cache" / "progress.json"
    try:
        if not root.resolve().is_relative_to(service.store.root.resolve()) or not _diagnostic_file(
            path, root
        ):
            return None
        with path.open("rb") as stream:
            raw = stream.read(16385)
            modified_at = os.fstat(stream.fileno()).st_mtime
        if len(raw) > 16384:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            return None
        result: dict[str, Any] = {
            "candidate_id": candidate["id"],
            "label": candidate.get("label", "Candidate"),
            "snapshot_age_seconds": max(0.0, time.time() - modified_at),
        }
        for key in ("stage", "phase", "stop_reason"):
            text = value.get(key)
            # Never expose an internal stage path or filesystem-bearing exception.
            if isinstance(text, str) and re.fullmatch(r"[\w .(),;=+%\-]{1,240}", text):
                result[key] = text
        for key in ("completed_stages", "reused_stages", "mesh_level", "domain_level"):
            number = value.get(key)
            if type(number) is int and number >= 0:
                result[key] = number
        elapsed = value.get("elapsed_seconds")
        if (
            isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool)
            and math.isfinite(elapsed)
            and elapsed >= 0
        ):
            result["elapsed_seconds"] = elapsed
        sensitivity = value.get("sensitivity")
        if isinstance(sensitivity, dict):
            observed: dict[str, Any] = {}
            for metric in ("drag_n", "drag_coefficient"):
                measure = sensitivity.get(metric)
                change = measure.get("relative_change") if isinstance(measure, dict) else None
                if (
                    isinstance(change, (int, float))
                    and not isinstance(change, bool)
                    and math.isfinite(change)
                    and change >= 0
                ):
                    observed[metric] = {"relative_change": change}
            result["sensitivity"] = observed
        try:
            iteration = _iteration(value.get("stage_directory"), root)
        except (OSError, ValueError, UnicodeError, csv.Error):
            iteration = None
        if iteration is not None:
            result["iteration"] = iteration
        return result if "phase" in result else None
    except (OSError, ValueError, UnicodeError, OverflowError):
        return None


def study_progress(
    service: WorkspaceService, sid: str, owner: str, study_id: str
) -> dict[str, Any]:
    """Return an owner-scoped change token; only read_study verifies evidence bytes."""
    study = _study(service, sid, owner, study_id)
    session = service.store.session_record(sid, owner)
    candidates = _candidates(service, study_id)
    engineering = service.store.engineering(sid)
    with service.store.connect() as con:
        sources = source_fingerprint(con, sid)
        row = con.execute(
            "SELECT document,sha256 FROM design_study_runs WHERE study=? ORDER BY rowid DESC LIMIT 1",
            (study_id,),
        ).fetchone()
        run = _decode(row) if row else None
        job = (
            con.execute("SELECT state FROM jobs WHERE id=?", (run["job_id"],)).fetchone()
            if run
            else None
        )
    status = run["state"] if run else "draft"
    if status in {"queued", "running"} and (job is None or job[0] not in {"queued", "running"}):
        status = "cancelled" if job and job[0] == "cancelled" else "interrupted"
    revision_ids = {study["baseline_revision_id"]}
    head = (session.get("proposal") or {}).get("revision_id") or session.get("selected_revision")
    if head:
        revision_ids.add(head)
    artifacts = []
    for candidate in candidates:
        if candidate.get("revision_id"):
            revision_ids.add(candidate["revision_id"])
        receipts = [
            *candidate.get("evaluations", []),
            *(
                r
                for attempt in candidate.get("attempts", [])
                for r in attempt.get("previous_evaluations", [])
            ),
        ]
        for receipt in receipts:
            if receipt.get("revision_id"):
                revision_ids.add(receipt["revision_id"])
            directory = (
                candidate_directory(service, sid, study_id, candidate["id"])
                / receipt["attempt"]
                / receipt["kind"]
            )
            for name in sorted(receipt.get("hashes", {})):
                artifacts.append(
                    [
                        candidate["id"],
                        receipt["attempt"],
                        receipt["kind"],
                        name,
                        _artifact_identity(directory, name),
                    ]
                )
    revisions = {}
    for rid in sorted(revision_ids):
        revision = service.store.revision(sid, rid)
        directory = service.store.directory(sid, rid)
        revisions[rid] = {
            "record": digest(revision),
            "files": {
                name: _artifact_identity(directory, name)
                for name in sorted(revision.get("measurement", {}).get("artifact_hashes", {}))
            },
        }
    engine = engine_fingerprint()
    token = digest(
        {
            "study": study,
            "candidates": candidates,
            "run": run,
            "status": status,
            "head": head,
            "engineering": engineering,
            "sources": sources,
            "engine": engine,
            "instruments": _instruments(study),
            "revisions": revisions,
            "artifacts": artifacts,
        }
    )
    return {
        "id": study_id,
        "token": token,
        "status": status,
        "input_version": engineering["version"],
        "source_fingerprint": sources,
        "engine_fingerprint": engine,
        "current_revision": head,
        "flow_progress": [
            diagnostic
            for candidate in candidates
            if (diagnostic := _flow_progress(service, sid, study_id, candidate)) is not None
        ],
        "scope": "Change detection only. Metadata equality does not verify artifact contents, numerical validity or study freshness. Read the full study before publishing new evidence.",
    }
