"""Owner-scoped report snapshots and consistent release packages."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZIP_DEFLATED, ZipFile

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.processing import process_revision, regular_file
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService

REPORT_FILES = {"report.json", "drawing.svg", "inspection.csv", "assembly.step", "section.png"}


def validate_engineering_worker(
    service: WorkspaceService,
    reason: str = "Engineering parser/solver worker is private-local only until container verification",
) -> None:
    """Apply the existing private-engineering execution boundary without starting work."""
    if service.executor.mode != "trusted_local":
        raise WorkspaceError(reason)


def run_report_job(
    service: WorkspaceService, job: dict[str, Any], cancelled: Callable[[], bool]
) -> None:
    from forge_mind.workspace.operation_recovery import guard_job, publish, worker_arguments

    guard_job(service, job)
    validate_engineering_worker(service)
    sid, owner = job["session"], job["owner"]
    session = service.store.get(sid, owner)
    revision = service._load_revision(sid, job["revision_id"])
    if not revision["measurement"]["geometry_valid"]:
        raise WorkspaceError("Engineering tools require valid geometry")
    if revision["measurement"]["engine_fingerprint"] != engine_fingerprint():
        raise WorkspaceError(
            "Revision instruments changed; preview a restore to rebuild before engineering review"
        )
    if "topology.json" not in revision["measurement"]["artifact_hashes"]:
        raise WorkspaceError("Rebuild this historical revision before running engineering tools")
    report_id = uuid.uuid4().hex
    from forge_mind.workspace.inspection_plan import resolve_inspection_plan

    inspection = resolve_inspection_plan(service, sid, owner, revision)
    directory = service.store.root / "reports" / sid / report_id
    directory.mkdir(parents=True)
    report = process_revision(
        directory,
        {
            "operation": "engineering",
            "source": str(service.store.directory(sid, revision["id"])),
            "revision": revision,
            "inputs": session["engineering"]["inputs"],
            "inspection_plan": inspection,
        },
        job.get("max_seconds"),
        cancelled,
        **worker_arguments(service, job, directory),
    )
    service._load_revision(
        sid, revision["id"]
    )  # do not publish a report if the source changed mid-run
    if resolve_inspection_plan(service, sid, owner, revision) != inspection:
        raise WorkspaceError("Inspection plan or bound probe evidence changed during review")
    report.update(id=report_id, input_version=session["engineering"]["version"], at=time.time())
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    hashes = {
        name: file_sha256(directory / name) for name in REPORT_FILES if (directory / name).is_file()
    }
    receipt = {
        "id": report_id,
        "revision_id": revision["id"],
        "input_version": session["engineering"]["version"],
        "engine_fingerprint": engine_fingerprint(),
        "hashes": hashes,
        "at": report["at"],
        "inspection_plan_hash": inspection["plan_hash"],
    }
    if cancelled():
        raise WorkspaceError("Report cancelled")
    publish(service, job, reports=[*session.get("reports", []), receipt][-100:])
    service.store.event(
        sid,
        "engineering",
        "Engineering review and drawing package computed",
        report_id=report_id,
        revision_id=revision["id"],
    )


def report_path(service: WorkspaceService, sid: str, owner: str, report_id: str, name: str) -> Path:
    session = service.store.get(sid, owner)
    receipt = next((r for r in session.get("reports", []) if r["id"] == report_id), None)
    if receipt is None or name not in REPORT_FILES or name not in receipt["hashes"]:
        raise WorkspaceError("Report artifact not found")
    path = service.store.root / "reports" / sid / report_id / name
    if not regular_file(path) or file_sha256(path) != receipt["hashes"][name]:
        raise WorkspaceError("Report artifact changed; evidence is stale")
    return path


def inspection_handoff(
    service: WorkspaceService,
    sid: str,
    owner: str,
    revision: dict[str, Any],
    *,
    max_artifact_bytes: int | None = None,
) -> dict[str, Any]:
    """Optional current report bytes; missing evidence must not block a geometry handoff."""
    from forge_mind.workspace.inspection_plan import resolve_inspection_plan

    if max_artifact_bytes is not None and (
        type(max_artifact_bytes) is not int or max_artifact_bytes < 0
    ):
        raise ValueError("Inspection artifact budget must be a nonnegative integer")
    session = service.store.get(sid, owner)
    bundle: dict[str, Any] = {
        "plan": session.get("design_plan"),
        "status": {
            "status": "unavailable",
            "reason": "No current engineering review",
            "physical_readiness": "unverified",
        },
        "artifacts": {},
    }
    receipt = next(
        (
            r
            for r in reversed(session.get("reports", []))
            if (
                r["revision_id"] == revision["id"]
                and r["input_version"] == session["engineering"]["version"]
                and r["engine_fingerprint"] == engine_fingerprint()
            )
        ),
        None,
    )
    if receipt is None:
        return bundle
    try:
        artifacts = {}
        transferred = 0
        for name in ("report.json", "drawing.svg", "inspection.csv"):
            if max_artifact_bytes is None:
                data = report_path(service, sid, owner, receipt["id"], name).read_bytes()
            else:
                # The captured receipt is owner-checked above. Do not use report_path's
                # unbounded hash pass before enforcing this aggregate transfer limit.
                path = service.store.root / "reports" / sid / receipt["id"] / name
                remaining = max_artifact_bytes - transferred
                if name not in receipt["hashes"] or not regular_file(path, remaining):
                    raise WorkspaceError(
                        "Review artifact is missing, linked or exceeds the inspection byte budget"
                    )
                with path.open("rb") as stream:
                    data = stream.read(remaining + 1)
                if len(data) > remaining:
                    raise WorkspaceError("Review artifact grew beyond the inspection byte budget")
                transferred += len(data)
            if hashlib.sha256(data).hexdigest() != receipt["hashes"][name]:
                raise WorkspaceError("Review artifact changed during handoff")
            artifacts[name] = data
        report = json.loads(artifacts["report.json"])
        current = resolve_inspection_plan(service, sid, owner, revision)
        if report.get("inspection_plan") != current:
            raise WorkspaceError(
                "Review inspection plan or probe evidence is stale; run review again"
            )
        bundle.update(
            artifacts=artifacts,
            status={
                "status": "included",
                "report_id": receipt["id"],
                "reason": "Current report and exact scoped receipts; author intent is not acceptance",
                "physical_readiness": "unverified",
            },
        )
    except (WorkspaceError, OSError, ValueError, KeyError) as exc:
        bundle["status"]["reason"] = str(exc)
    return bundle


def release_package(service: WorkspaceService, sid: str, owner: str, rid: str) -> Path:
    session = service.store.get(sid, owner)
    revision = service._load_revision(sid, rid)
    if not revision["measurement"]["geometry_valid"]:
        raise WorkspaceError("Cannot package failed geometry as a CAD release")
    inputs = session["engineering"]
    current_report = next(
        (
            r
            for r in reversed(session.get("reports", []))
            if r["revision_id"] == rid
            and r["input_version"] == inputs["version"]
            and r["engine_fingerprint"] == engine_fingerprint()
        ),
        None,
    )
    if current_report is None:
        raise WorkspaceError(
            "Run engineering review for this revision and current inputs before packaging"
        )
    from forge_mind.workspace.completion import _hash

    if current_report.get("inspection_plan_hash") is not None and current_report[
        "inspection_plan_hash"
    ] != _hash(session.get("design_plan")):
        raise WorkspaceError("Inspection plan changed; run engineering review before packaging")
    if current_report.get("inspection_plan_hash") is not None:
        from forge_mind.workspace.inspection_plan import resolve_inspection_plan

        saved = json.loads(
            report_path(service, sid, owner, current_report["id"], "report.json").read_text(
                encoding="utf-8"
            )
        )
        if saved.get("inspection_plan") != resolve_inspection_plan(service, sid, owner, revision):
            raise WorkspaceError(
                "Inspection evidence changed; run engineering review before packaging"
            )
    directory = service.store.directory(sid, rid)
    output = service.store.root / "packages" / sid
    output.mkdir(parents=True, exist_ok=True)
    target = output / (uuid.uuid4().hex + ".zip")
    manifest: dict[str, Any] = {
        "revision_id": rid,
        "input_version": inputs["version"],
        "engine_fingerprint": engine_fingerprint(),
        "report_id": current_report["id"],
        "files": {},
        "disposition": "Review package; unresolved checks remain in report.json. Packaging does not confer engineering approval.",
    }
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        files = {name: directory / name for name in revision["measurement"]["artifact_hashes"]}
        files.update(
            {
                "evidence.json": directory / "evidence.json",
                "contract.json": directory / "contract.json",
            }
        )
        for name in current_report["hashes"]:
            files["engineering/" + name] = report_path(
                service, sid, owner, current_report["id"], name
            )
        from forge_mind.workspace.parameter_study import study_path

        study_hashes: dict[str, str] = {}
        from forge_mind.workspace.motion_service import check_path

        for check in session.get("motion_checks", []):
            if check["revision_id"] == rid and check["engine_fingerprint"] == engine_fingerprint():
                name = "motion-checks/" + check["id"] + ".json"
                files[name] = check_path(service, sid, owner, check["id"])
                study_hashes[name] = check["sha256"]
        for study in session.get("parameter_studies", []):
            if (
                study["source_revision"] == rid
                and study.get("input_version") == inputs["version"]
                and study.get("engine_fingerprint") == engine_fingerprint()
            ):
                study_name = "studies/" + study["id"] + ".json"
                files[study_name] = study_path(service, sid, owner, study["id"])
                study_hashes[study_name] = study["sha256"]
                for sample in json.loads(files[study_name].read_text(encoding="utf-8"))["samples"]:
                    if sample.get("revision_id"):
                        sample_revision = service._load_revision(sid, sample["revision_id"])
                        sample_directory = service.store.directory(sid, sample["revision_id"])
                        for member, digest in sample_revision["measurement"][
                            "artifact_hashes"
                        ].items():
                            name = "study-revisions/" + sample["revision_id"] + "/" + member
                            files[name] = sample_directory / member
                            study_hashes[name] = digest
                    if sample.get("report"):
                        receipt = sample["report"]
                        for member, digest in receipt["hashes"].items():
                            name = "study-reports/" + receipt["id"] + "/" + member
                            files[name] = report_path(service, sid, owner, receipt["id"], member)
                            study_hashes[name] = digest
        for name, path in files.items():
            if not regular_file(path):
                raise WorkspaceError("Release member is missing or linked")
            data = path.read_bytes()
            expected = (
                study_hashes.get(name)
                or revision["measurement"]["artifact_hashes"].get(name)
                or current_report["hashes"].get(name.removeprefix("engineering/"))
            )
            actual = hashlib.sha256(data).hexdigest()
            if expected and expected != actual:
                raise WorkspaceError("Release member changed during packaging")
            if name in {"evidence.json", "contract.json"}:
                expected_json = revision["measurement" if name == "evidence.json" else "contract"]
                if json.loads(data) != expected_json:
                    raise WorkspaceError("Release evidence no longer matches immutable revision")
            archive.writestr(name, data)
            manifest["files"][name] = actual
        input_data = json.dumps(inputs, indent=2, allow_nan=False).encode()
        archive.writestr("engineering-inputs.json", input_data)
        manifest["files"]["engineering-inputs.json"] = hashlib.sha256(input_data).hexdigest()
        from forge_mind.workspace.structural_service import list_studies, read_study
        from forge_mind.workspace.structural_service import study_path as structural_study_path

        for summary in list_studies(service, sid, owner, rid):
            if summary["stale"]:
                continue
            study = read_study(service, sid, owner, summary["id"])
            if study["stale"]:
                continue
            prefix = "structural-studies/" + study["id"] + "/"
            receipt_data = json.dumps(study, sort_keys=True, allow_nan=False).encode()
            archive.writestr(prefix + "receipt.json", receipt_data)
            manifest["files"][prefix + "receipt.json"] = hashlib.sha256(receipt_data).hexdigest()
            for name, expected in study["hashes"].items():
                artifact_data = (
                    structural_study_path(service, sid, study["id"]) / name
                ).read_bytes()
                artifact_hash = hashlib.sha256(artifact_data).hexdigest()
                if artifact_hash != expected:
                    raise WorkspaceError("Structural field or mesh changed during packaging")
                archive.writestr(prefix + name, artifact_data)
                manifest["files"][prefix + name] = artifact_hash
        if session.get("design_plan"):
            plan_data = json.dumps(
                {
                    "scope": "Current author intent, not validation evidence",
                    "plan": session["design_plan"],
                },
                indent=2,
            ).encode()
            archive.writestr("author-plan.json", plan_data)
            manifest["files"]["author-plan.json"] = hashlib.sha256(plan_data).hexdigest()
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise WorkspaceError("Release ZIP integrity verification failed")
    return target
