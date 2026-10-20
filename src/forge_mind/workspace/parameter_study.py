"""Bounded parameter experiments on immutable source revisions, with exported witnesses."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.budget import expired, time_left
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def summarize_report(report: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    names = {r["body_index"]: r["path"] for r in facts.get("assembly", {}).get("occurrences", [])}
    assembly = report.get("assembly", {})
    contexts = {(p["a"], p["b"]): p["context"] for p in assembly.get("overlap_witnesses", [])}
    return {
        "status": assembly.get("pose_status", assembly.get("status", "unverified")),
        "pair_count": len(assembly.get("pairs", [])),
        "collisions": [
            {
                **p,
                "a": names.get(p["a"], str(p["a"])),
                "b": names.get(p["b"], str(p["b"])),
                "context": contexts.get((p["a"], p["b"]), "unknown"),
            }
            for p in assembly.get("pairs", [])
            if p["status"] == "fail"
        ],
        "scope": assembly.get("scope"),
        "reason": assembly.get("reason"),
        "interpretation_scope": assembly.get("interpretation_scope"),
    }


def parameter_study(
    service: WorkspaceService,
    sid: str,
    owner: str,
    source_id: str,
    contract: Contract,
    variable: str,
    values: list[float],
    deadline: float | None,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    from forge_mind.workspace.document_service import build_document
    from forge_mind.workspace.engineering_service import report_path, run_report_job

    session = service.store.get(sid, owner)
    input_version = session["engineering"]["version"]
    source = service._load_revision(sid, source_id)
    from forge_mind.workspace.measure import engine_fingerprint

    if source["measurement"]["engine_fingerprint"] != engine_fingerprint():
        raise WorkspaceError("Rebuild the source with current instruments before a study")
    document = DesignDocument.model_validate(source.get("document"))
    if not document.assembly or variable not in document.variables:
        raise WorkspaceError("Study needs a reusable assembly and an existing input variable")
    if (
        len(values) < 2
        or len(set(values)) != len(values)
        or any(not math.isfinite(v) for v in values)
    ):
        raise WorkspaceError("Study needs at least two distinct finite input values")
    baseline = {r["path"]: r for r in source["measurement"]["facts"]["assembly"]["occurrences"]}
    source_keys = {h["id"]: h.get("key") for h in source["measurement"]["facts"]["feature_history"]}
    result: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "source_revision": source_id,
        "source_step_sha256": source["measurement"]["artifact_hashes"]["part.step"],
        "engine_fingerprint": source["measurement"]["engine_fingerprint"],
        "contract_hash": contract.fingerprint(),
        "input_version": input_version,
        "engineering_inputs": session["engineering"]["inputs"],
        "variable": variable,
        "source_value": document.variables[variable],
        "requested_values": values,
        "samples": [],
        "status": "unverified",
        "coverage": "discrete_only",
        "scope": "Rebuilt static configurations. Displacements are exported centers of mass, not proof of a guide, screw coupling, retention, continuous clearance or strength.",
    }
    for value in values:
        sample: dict[str, Any] = {"value": value}
        try:
            if cancelled() or expired(deadline):
                raise WorkspaceError("Study cancelled or budget exhausted")
            if service.store.get(sid, owner)["engineering"]["version"] != input_version:
                raise WorkspaceError("Engineering inputs changed during study")
            candidate = document.model_copy(deep=True)
            candidate.variables[variable] = value
            revision = build_document(
                service,
                sid,
                owner,
                candidate,
                contract,
                source_id,
                f"Study {variable} = {value}",
                time_left(deadline),
                cancelled,
            )
            measured = revision["measurement"]
            sample.update(
                revision_id=revision["id"],
                geometry_valid=measured["geometry_valid"],
                step_sha256=measured["artifact_hashes"].get("part.step"),
                findings=measured["findings"],
                errors=measured["errors"],
            )
            if not measured["geometry_valid"]:
                raise WorkspaceError("Study configuration failed geometry validation")
            facts = measured["facts"]
            keys = {h["id"]: h.get("key") for h in facts["feature_history"]}
            sample["occurrences"] = [
                {
                    "path": r["path"],
                    "definition": r["definition"],
                    "center_mm": r["center_mm"],
                    "delta_mm": [
                        a - b
                        for a, b in zip(
                            r["center_mm"], baseline[r["path"]]["center_mm"], strict=True
                        )
                    ],
                    "definition_unchanged": bool(keys.get(r["definition"]))
                    and keys.get(r["definition"]) == source_keys.get(r["definition"]),
                }
                for r in facts["assembly"]["occurrences"]
            ]
            sample["extents_mm"] = facts["extents_mm"]
            sample["interfaces"] = facts["assembly"]["mates"]
            run_report_job(
                service,
                {
                    "session": sid,
                    "owner": owner,
                    "revision_id": revision["id"],
                    "max_seconds": time_left(deadline),
                },
                cancelled,
            )
            receipt = next(
                (
                    r
                    for r in reversed(service.store.get(sid, owner).get("reports", []))
                    if r["revision_id"] == revision["id"]
                    and r["engine_fingerprint"] == measured["engine_fingerprint"]
                    and r["input_version"] == input_version
                ),
                None,
            )
            if receipt is None:
                raise WorkspaceError("No report belongs to this study configuration")
            report = json.loads(
                report_path(service, sid, owner, receipt["id"], "report.json").read_text(
                    encoding="utf-8"
                )
            )
            sample["report"] = receipt
            sample["interference"] = summarize_report(report, facts)
            if (
                sample["interference"]["status"] == "fail"
                or any(f["status"] == "fail" for f in measured["findings"])
                or facts["assembly"]["status"] == "fail"
            ):
                result["status"] = "fail"
            if sample["interference"]["status"] not in {"pass", "fail"}:
                result["coverage"] = "incomplete"
        except (ValueError, OSError) as exc:
            sample["error"] = str(exc)[:1000]
            result["coverage"] = "incomplete"
            result["status"] = "fail" if sample.get("geometry_valid") is False else result["status"]
        result["samples"].append(sample)
    directory = service.store.root / "studies" / sid
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (result["id"] + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    session = service.store.get(sid, owner)
    service.store.update(
        sid,
        parameter_studies=[
            *session.get("parameter_studies", []),
            {
                "id": result["id"],
                "source_revision": source_id,
                "sha256": file_sha256(path),
                "variable": variable,
                "values": values,
                "status": result["status"],
                "coverage": result["coverage"],
                "engine_fingerprint": source["measurement"]["engine_fingerprint"],
                "input_version": input_version,
            },
        ][-20:],
    )
    service.store.event(
        sid,
        "study",
        f"Measured {variable} at {len(values)} configurations",
        study_id=result["id"],
        coverage=result["coverage"],
        status=result["status"],
    )
    return result


def study_path(service: WorkspaceService, sid: str, owner: str, study_id: str) -> Path:
    from forge_mind.workspace.engineering_service import report_path
    from forge_mind.workspace.processing import regular_file

    session = service.store.get(sid, owner)
    receipt = next((r for r in session.get("parameter_studies", []) if r["id"] == study_id), None)
    if receipt is None:
        raise WorkspaceError("Study not found")
    path = service.store.root / "studies" / sid / (study_id + ".json")
    if not regular_file(path) or file_sha256(path) != receipt["sha256"]:
        raise WorkspaceError("Study artifact changed")
    study = json.loads(path.read_text(encoding="utf-8"))
    source = service._load_revision(sid, study["source_revision"])
    if source["measurement"]["artifact_hashes"]["part.step"] != study["source_step_sha256"]:
        raise WorkspaceError("Study source geometry changed")
    for sample in study["samples"]:
        if sample.get("revision_id"):
            revision = service._load_revision(sid, sample["revision_id"])
            if revision["measurement"]["artifact_hashes"].get("part.step") != sample.get(
                "step_sha256"
            ):
                raise WorkspaceError("Study sample geometry changed")
        if sample.get("report"):
            report_path(service, sid, owner, sample["report"]["id"], "report.json")
    return path
