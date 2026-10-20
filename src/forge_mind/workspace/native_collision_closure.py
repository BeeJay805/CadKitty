"""Current-rig reference-pose closure, without engineering approval."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.document import digest
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import Contract
from forge_mind.workspace.motion_service import check_path
from forge_mind.workspace.store import WorkspaceError

# Existing motion instrument threshold. Do not replace it with an author limit.
OVERLAP_THRESHOLD_MM3 = 1e-5


def _retained_review(service: Any, sid: str, revision: dict[str, Any], document: Any) -> None:
    """Do not discard an admitted rigid review or reclassify its made bodies."""
    parent = revision.get("parent_id")
    contract_hash = Contract.model_validate(revision["contract"]).fingerprint()
    seen = {revision["id"]}
    while parent:
        if parent in seen:
            raise WorkspaceError("Revision review ancestry is cyclic")
        seen.add(parent)
        previous = service.store.revision(sid, parent)
        if Contract.model_validate(previous["contract"]).fingerprint() != contract_hash:
            break
        measured = previous["measurement"]
        raw = previous.get("document") or {}
        if measured.get("geometry_valid") and raw.get("motion_rig"):
            source = service.store.directory(sid, parent) / "document.json"
            if file_sha256(source) != measured["artifact_hashes"]["document.json"]:
                raise WorkspaceError("Admitted motion review source changed")
            if document.motion_rig is None:
                raise WorkspaceError("Retain the admitted motion rig within these requirements")
            current = (
                {p.id: p.production.role for p in document.assembly.parts}
                if document.assembly
                else {}
            )
            for part in raw["assembly"]["parts"]:
                if (part.get("production") or {}).get("role") == "make" and current.get(
                    part["id"]
                ) != "make":
                    raise WorkspaceError(
                        "Retain admitted made definitions in this rigid review: " + part["id"]
                    )
            from forge_mind.workspace.assembly import expand

            old_counts = Counter(
                row["definition"]
                for row in measured["facts"]["assembly"]["occurrences"]
                if current.get(row["definition"]) == "make"
            )
            new_counts = Counter(row["definition"] for row in expand(document))
            if any(new_counts[key] < count for key, count in old_counts.items()):
                raise WorkspaceError("Retain admitted made occurrences in this rigid review")
        parent = previous.get("parent_id")


def _evaluate_reference(
    report: dict[str, Any], motion: dict[str, Any], roles: dict[str, str]
) -> dict[str, Any]:
    reference = {driver["id"]: driver["reference"] for driver in motion["rig"]["drivers"]}
    result: dict[str, Any] = {
        "status": "unverified",
        "reference_values": reference,
        "collisions": [],
        "overlap_threshold_mm3": OVERLAP_THRESHOLD_MM3,
        "scope": "Separate made bodies at the current rigid reference pose only; no dynamic, fit, strength or physical approval.",
    }
    if (
        report.get("reference_values") != reference
        or report.get("reference_completed") is not True
        or reference not in report.get("completed_configurations", [])
        or not isinstance(report.get("reference_collisions"), list)
    ):
        return {
            **result,
            "reason": "Current reference pose lacks complete, untruncated collision evidence",
        }
    rows = report["reference_collisions"]
    failures = []
    for row in rows:
        if not isinstance(row, dict) or row.get("values") != reference:
            return {**result, "reason": "Reference collision witness has invalid pose identity"}
        a, b, volume = row.get("a"), row.get("b"), row.get("overlap_mm3")
        if (
            not isinstance(a, str)
            or not isinstance(b, str)
            or a == b
            or a not in roles
            or b not in roles
            or not isinstance(volume, (int, float))
            or isinstance(volume, bool)
            or not math.isfinite(volume)
            or volume <= OVERLAP_THRESHOLD_MM3
        ):
            return {
                **result,
                "reason": "Reference collision witness has invalid body or volume data",
            }
        if roles[a] == roles[b] == "make":
            failures.append(row)
    result["collisions"] = failures
    if failures:
        return {
            **result,
            "status": "needs_work",
            "reason": "Repair overlapping separate made bodies at the reference pose",
        }
    if any(role not in {"make", "buy", "reference"} for role in roles.values()):
        return {**result, "reason": "Current occurrence production roles are incomplete"}
    return {
        **result,
        "status": "checked",
        "reason": "No made-body overlap exceeds the current instrument threshold at the reference pose",
    }


def reference_collision_context(
    service: Any, sid: str, owner: str, revision: dict[str, Any], document: Any
) -> dict[str, Any]:
    unavailable = {
        "status": "unverified",
        "collisions": [],
        "scope": "Current-rig reference pose only",
    }
    try:
        if revision:
            _retained_review(service, sid, revision, document)
        if document.motion_rig is None:
            return {"status": "not_applicable", "scope": "No current prescribed rigid motion rig"}
        measurement = revision["measurement"]
        motion = measurement["facts"]["motion_rig"]
        expected_engine = engine_fingerprint()
        expected = {
            "revision_id": revision["id"],
            "engine_fingerprint": expected_engine,
            "step_sha256": measurement["artifact_hashes"]["part.step"],
            "document_sha256": measurement["artifact_hashes"]["document.json"],
        }
        if any(motion.get(key) != value for key, value in expected.items()):
            raise WorkspaceError("Current rig is not bound to the revision and engine")
        directory = service.store.directory(sid, revision["id"])
        for name, key in (("part.step", "step_sha256"), ("document.json", "document_sha256")):
            if file_sha256(directory / name) != expected[key]:
                raise WorkspaceError("Current motion source artifact changed")
        if not document.assembly:
            raise WorkspaceError("Current rig has no assembly definitions")
        definitions = {part.id: part.production.role for part in document.assembly.parts}
        from forge_mind.workspace.assembly import expand

        expanded = {row["path"]: row["definition"] for row in expand(document)}
        occurrences = measurement["facts"]["assembly"]["occurrences"]
        binding_rows = motion["bindings"]
        if (
            len({row["path"] for row in occurrences}) != len(occurrences)
            or len({row["path"] for row in binding_rows}) != len(binding_rows)
            or sorted(row["body_index"] for row in occurrences) != list(range(len(occurrences)))
            or {(row["path"], row["body_index"]) for row in occurrences}
            != {(row["path"], row["body_index"]) for row in binding_rows}
            or any(row["definition"] not in definitions for row in occurrences)
            or {row["path"]: row["definition"] for row in occurrences} != expanded
        ):
            raise WorkspaceError(
                "Motion bodies do not match complete exported occurrence definitions"
            )
        roles = {row["path"]: definitions[row["definition"]] for row in occurrences}
        session = service.store.get(sid, owner)
        for receipt in reversed(session.get("motion_checks", [])):
            if receipt.get("revision_id") != revision["id"]:
                continue
            settings = receipt.get("settings")
            if (
                receipt.get("engine_fingerprint") != expected_engine
                or not isinstance(settings, dict)
                or receipt.get("key") != digest({"motion": motion, "settings": settings})
            ):
                continue
            report = json.loads(
                check_path(service, sid, owner, receipt["id"]).read_text(encoding="utf-8")
            )
            if (
                any(report.get(key) != value for key, value in expected.items())
                or report.get("settings") != settings
            ):
                raise WorkspaceError("Motion report identity differs from its current receipt")
            return {**_evaluate_reference(report, motion, roles), "check_id": receipt["id"]}
        raise WorkspaceError("Run a current hash-bound reference-pose check before final review")
    except (WorkspaceError, ValueError, OSError, KeyError, TypeError) as exc:
        return {**unavailable, "reason": str(exc)}
