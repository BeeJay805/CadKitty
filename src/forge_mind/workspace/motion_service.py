"""Private owner-scoped motion checks using the existing bounded job queue."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.document import digest
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.motion_check import MotionCheckSettings
from forge_mind.workspace.processing import process_revision, regular_file
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def check_path(service: WorkspaceService, sid: str, owner: str, check_id: str) -> Path:
    session = service.store.get(sid, owner)
    receipt = next((r for r in session.get("motion_checks", []) if r["id"] == check_id), None)
    if not receipt:
        raise WorkspaceError("Motion check not found")
    path = service.store.root / "motion-checks" / sid / check_id / "check.json"
    if not regular_file(path) or file_sha256(path) != receipt["sha256"]:
        raise WorkspaceError("Motion check evidence changed")
    return path


def run_motion_job(
    service: WorkspaceService, job: dict[str, Any], cancelled: Callable[[], bool]
) -> str:
    from forge_mind.workspace.operation_recovery import guard_job, publish, worker_arguments

    guard_job(service, job)
    sid, owner = job["session"], job["owner"]
    session = service.store.get(sid, owner)
    revision = service._load_revision(sid, job["revision_id"])
    motion = revision["measurement"]["facts"].get("motion_rig")
    if not revision["measurement"]["geometry_valid"]:
        raise WorkspaceError("Motion checking requires valid exported geometry")
    if service.executor.mode != "trusted_local" or not motion:
        raise WorkspaceError("Motion checking requires a private-local revision with a motion rig")
    if motion["engine_fingerprint"] != engine_fingerprint():
        raise WorkspaceError("Motion instruments changed; rebuild this revision")
    settings = MotionCheckSettings.model_validate(job["settings"]).model_dump()
    key = digest({"motion": motion, "settings": settings})
    for receipt in reversed(session.get("motion_checks", [])):
        if receipt["key"] == key and receipt["coverage"] == "discrete_only":
            check_path(service, sid, owner, receipt["id"])
            if cancelled():
                raise WorkspaceError("Motion check cancelled")
            publish(service, job, motion_progress={"state": "cached", "check_id": receipt["id"]})
            service.store.event(
                sid, "motion", "Reused identical completed motion check", check_id=receipt["id"]
            )
            return str(receipt["id"])
    check_id = uuid.uuid4().hex
    directory = service.store.root / "motion-checks" / sid / check_id
    directory.mkdir(parents=True)
    service.store.update(
        sid,
        motion_progress={"state": "running", "revision_id": revision["id"], "completed_pairs": 0},
    )
    last_poll = 0.0

    def monitor() -> bool:
        nonlocal last_poll
        if time.monotonic() - last_poll > 0.5:
            last_poll = time.monotonic()
            path = directory / "progress.json"
            if regular_file(path, 4096):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    service.store.update(
                        sid,
                        motion_progress={
                            **value,
                            "state": "running",
                            "revision_id": revision["id"],
                        },
                    )
                except (OSError, ValueError):
                    pass  # A progress read cannot make absent evidence complete.
        return cancelled()

    try:
        result = process_revision(
            directory,
            {
                "operation": "motion_check",
                "source": str(service.store.directory(sid, revision["id"])),
                "motion": motion,
                "settings": settings,
                "max_seconds": job.get("max_seconds"),
            },
            job.get("max_seconds"),
            monitor,
            **worker_arguments(service, job, directory),
        )
        service._load_revision(sid, revision["id"])
        if cancelled() or engine_fingerprint() != motion["engine_fingerprint"]:
            raise WorkspaceError("Motion check cancelled or instruments changed during checking")
        path = directory / "check.json"
        path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        receipt = {
            "id": check_id,
            "key": key,
            "revision_id": revision["id"],
            "engine_fingerprint": motion["engine_fingerprint"],
            "sha256": file_sha256(path),
            "settings": settings,
            "status": result["status"],
            "coverage": result["coverage"],
        }
        publish(
            service,
            job,
            motion_checks=[*session.get("motion_checks", []), receipt][-20:],
            motion_progress={"state": "complete", "check_id": check_id},
        )
        service.store.event(
            sid,
            "motion",
            "Sampled motion check finished",
            check_id=check_id,
            coverage=result["coverage"],
        )
        return check_id
    except Exception:
        service.store.update(
            sid,
            motion_progress={
                "state": "cancelled" if cancelled() else "incomplete",
                "revision_id": revision["id"],
            },
        )
        raise
