"""Host-controlled photo reconstruction stages over immutable observations.

The model supplies correspondences, never a stage, receipt, verdict or source fact.
All automatic stages consume their enclosing author's ordinary action and time.
"""

from __future__ import annotations

import io
import json
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import Field, model_validator

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.models import Record
from forge_mind.workspace.photo_profile_models import PhotoProfileRequest, ProfileAnchor
from forge_mind.workspace.photo_profile_service import (
    _digest,
    _source,
    analyze_reference,
    compare_reference,
    read_profile,
)
from forge_mind.workspace.store import WorkspaceError

POLICY = "photo_generation_v1"
GROUNDING_INSTRUCTIONS = """Return PhotoGrounding, not a document, tool action or verdict.
Use the automatically detected contour feature IDs in photo_generation.inventory.
Select one complete foreground component and exactly four ordered feature IDs:
upper-left, upper-right, lower-right, lower-left of the proposed source rectangle.
The host maps these to (0,height),(width,height),(width,0),(0,0), with +Z thickness.
Use the given literal source fact IDs and one designed subject. Corners, planarity,
lens assumptions and this design frame are proposed interpretations, not owner facts
or independent checks. Never invent a dimension, checkpoint or source quote.
If the available source cannot support this planar rectangle with explicit depth
and through openings, return blocked_reason with the precise missing input or
unsupported geometry. Use the measured editable proposal in subsequent construction;
do not manually re-estimate measured hole centers or diameters from the image.
"""


class PhotoGrounding(Record):
    inventory_observation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    component_id: int | None = Field(default=None, strict=True, ge=1, le=1024)
    subject_id: str | None = Field(default=None, min_length=1)
    part_id: str | None = Field(default=None, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    corner_feature_ids: list[str] | None = Field(default=None, min_length=4, max_length=4)
    size_fact_ids: list[str] | None = Field(default=None, min_length=3, max_length=3)
    hole_count_fact_id: str | None = Field(default=None, min_length=1)
    planar_hypothesis: bool | None = None
    negligible_distortion_hypothesis: bool | None = None
    blocked_reason: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def supported_or_blocked(self) -> PhotoGrounding:
        values = [
            getattr(self, key)
            for key in type(self).model_fields
            if key not in {"inventory_observation_id", "blocked_reason", "hole_count_fact_id"}
        ]
        if self.blocked_reason is not None:
            if any(value is not None for value in values) or self.hole_count_fact_id is not None:
                raise ValueError("Blocked grounding takes only the inventory ID and precise reason")
        elif any(value is None for value in values):
            raise ValueError("Supported grounding requires every correspondence and source binding")
        return self


def applicable(session: dict[str, Any]) -> bool:
    return session.get("authoring_mode") == "native_operations" and bool(
        session.get("customer_reference")
    )


def _table(store: Any) -> None:
    with store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS photo_generation (session TEXT PRIMARY KEY, document TEXT NOT NULL, digest TEXT NOT NULL, FOREIGN KEY(session) REFERENCES sessions(id))"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS photo_generation_history (session TEXT NOT NULL, digest TEXT NOT NULL, document TEXT NOT NULL, PRIMARY KEY(session,digest))"
        )


def _owner_change(record: dict[str, Any], session: dict[str, Any]) -> bool:
    before = record.get("owner_instructions", [])
    after = session.get("instructions", [])
    return bool(
        record.get("job_id") != session.get("job_id")
        and session.get("prompt") == record.get("owner_prompt")
        and len(after) > len(before)
        and after[: len(before)] == before
    )


def _source_binding(service: Any, sid: str, owner: str) -> dict[str, Any]:
    from forge_mind.workspace.intent_attribution_service import request_text

    session = service.store.get(sid, owner)
    _, source = _source(service.store, sid, owner)
    return {
        "source": source,
        "source_text_sha256": _digest(request_text(session)),
        "source_fingerprint": session.get("source_fingerprint"),
    }


def _read(service: Any, sid: str, owner: str) -> dict[str, Any] | None:
    session = service.store.get(sid, owner)
    if not applicable(session):
        return None
    _table(service.store)
    with service.store.connect() as con:
        row = con.execute(
            "SELECT document,digest FROM photo_generation WHERE session=?", (sid,)
        ).fetchone()
    if row is None:
        return None
    record: dict[str, Any] = json.loads(row[0])
    if _digest(record) != row[1] or record.get("policy") != POLICY:
        raise WorkspaceError("Photo generation state changed")
    if record["binding"] != _source_binding(service, sid, owner):
        if _owner_change(record, session):
            return None
        raise WorkspaceError("Photo source changed; a new explicitly admitted target is required")
    target = record.get("target")
    if target:
        if target["target_sha256"] != _digest(
            {key: value for key, value in target.items() if key != "target_sha256"}
        ):
            raise WorkspaceError("Frozen photo target changed")
        observation = read_profile(service, sid, owner, target["observation_id"])
        if _digest(observation) != target["observation_digest"]:
            raise WorkspaceError("Frozen photo observation changed")
    return record


def _write(
    service: Any,
    sid: str,
    owner: str,
    jid: str,
    record: dict[str, Any],
    *,
    deadline: float | None = None,
    cancelled: Any = None,
) -> None:
    _table(service.store)
    if record["binding"] != _source_binding(service, sid, owner):
        raise WorkspaceError("Photo generation source changed before publication")
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        job = con.execute(
            "SELECT state,cancel,deadline FROM jobs WHERE id=? AND session=?", (jid, sid)
        ).fetchone()
        if (
            row is None
            or json.loads(row[0]).get("job_id") != jid
            or job is None
            or job[0] != "running"
            or job[1]
            or (job[2] is not None and time.time() >= job[2])
        ):
            raise WorkspaceError("Photo generation requires this owner's active job")
        if record["binding"] != _source_binding(service, sid, owner):
            raise WorkspaceError("Photo source changed during publication")
        previous = con.execute(
            "SELECT document FROM photo_generation WHERE session=?", (sid,)
        ).fetchone()
        if previous:
            old = json.loads(previous[0])
            changed_by_owner = old["binding"] != record["binding"] and _owner_change(
                old, json.loads(row[0])
            )
            if old.get("target") and old["target"] != record.get("target") and not changed_by_owner:
                raise WorkspaceError("A frozen photo target cannot be replaced during correction")
            if changed_by_owner:
                con.execute(
                    "INSERT OR IGNORE INTO photo_generation_history VALUES (?,?,?)",
                    (sid, _digest(old), previous[0]),
                )
        if (cancelled is not None and cancelled()) or (
            deadline is not None and time.monotonic() >= deadline
        ):
            raise WorkspaceError("Photo stage cancelled or deadline exhausted before publication")
        con.execute(
            "INSERT OR REPLACE INTO photo_generation VALUES (?,?,?)",
            (sid, json.dumps(record, allow_nan=False), _digest(record)),
        )


def _constraints(contract: Any) -> list[dict[str, Any]]:
    from forge_mind.workspace.photo_nominal import photo_source_constraints

    values = []
    for requirement in contract.requirements:
        criterion = requirement.verification
        if getattr(criterion, "kind", None) == "nominal_photo_solid":
            value = photo_source_constraints(criterion)
            if value not in values:
                values.append(value)
    return values


def generation_context(
    service: Any, sid: str, owner: str, rid: str | None, contract: Any
) -> dict[str, Any] | None:
    session = service.store.get(sid, owner)
    if not applicable(session):
        return None
    record = _read(service, sid, owner)
    if record is None:
        return {"stage": "inventory", "policy": POLICY}
    from forge_mind.workspace.photo_nominal import photo_reference_role

    role = photo_reference_role(session)
    if role == "context":
        return None
    if role != "copy":
        return {
            "stage": "unsupported",
            "reason": "Photo reference role is unresolved; retain the pixel inventory and clarify copying versus contextual reference use",
        }
    result: dict[str, Any] = {
        "policy": POLICY,
        "stage": "grounding",
        "source_constraints": _constraints(contract),
    }
    if record.get("blocked_reason"):
        result.update(stage="unsupported", reason=record["blocked_reason"])
    elif not record.get("target"):
        from forge_mind.workspace.photo_profile_context import photo_profile_context

        result["inventory"] = [
            photo_profile_context(read_profile(service, sid, owner, value), sid)
            for value in record["inventory_ids"]
        ]
    else:
        target = record["target"]
        result.update(target=deepcopy(target), stage="construct")
        if rid:
            closure = photo_closure(service, sid, owner, rid)
            result.update(closure=closure)
            result["stage"] = (
                "compare"
                if closure["status"] == "comparison_needed"
                else ("checked" if closure["status"] == "checked" else "correct")
            )
    return result


def construction_example(context: dict[str, Any]) -> dict[str, Any] | None:
    """Provide exact existing transaction syntax, never an admitted plan write."""
    target = (context.get("photo_generation") or {}).get("target")
    plan = context.get("design_plan") or {}
    basis = context.get("edit_basis")
    if not target or not basis or basis.get("status") == "unavailable":
        return None
    steps = [
        step
        for step in (plan.get("operation_plan") or {}).get("steps", [])
        if step["part_id"] == target["part_id"]
    ]
    if not steps:
        return {
            "next_step": "Save provisional PlanIntent with first_part_id and make part id equal to the frozen target part_id"
        }
    document = target["proposal_document"]
    return {
        "scope": "Example proposal syntax only; the author must submit it and existing source/plan/approval validation still runs",
        "document_basis": "Use the frozen measured features/variables/outputs unchanged in the made part; preserve XY origin and positive-Z thickness. For requested additive printing use an explicit one-part assembly and identity root placement.",
        "action": {
            "kind": "document",
            "document_json": "<complete DesignDocument JSON>",
            "plan_patch_json": json.dumps(
                {
                    "basis": basis,
                    "construction_assignments": [
                        {
                            "step_id": steps[0]["id"],
                            "feature_ids": [feature["id"] for feature in document["features"]],
                        }
                    ],
                    "verification_bindings": [
                        {
                            "subject_id": target["source_constraints"]["subject_id"],
                            "part_id": target["part_id"],
                            "frame": "definition_local",
                        }
                    ],
                }
            ),
        },
        "dimension_bindings": {"thickness_mm": target["source_constraints"]["size_mm"][2]},
        "dimension_rule": "Only thickness_mm is a named binding in this measured document. Width/height are exact rectangle sketch coordinates. Do not invent bound width/height variables or owner facts. If adding authored DimensionFact entries, use PlanPatch.dimensions with literal source provenance and actual document binding names.",
    }


def observe_inventory(
    service: Any, sid: str, owner: str, jid: str, *, timeout: float, cancelled: Any
) -> dict[str, Any]:
    started = time.monotonic()
    image, source = _source(service.store, sid, owner)
    with Image.open(io.BytesIO(image)) as decoded:
        histogram = decoded.convert("L").histogram()
    # Otsu's between-class variance, computed from bounded integer histogram data.
    total = sum(histogram)
    intensity = sum(i * count for i, count in enumerate(histogram))
    weight = moment = 0
    best = -1.0
    threshold = 127
    for i, count in enumerate(histogram[:-1]):
        weight += count
        moment += i * count
        if weight and total > weight:
            score = (moment * total - intensity * weight) ** 2 / (weight * (total - weight))
            if score > best:
                threshold, best = i, score
    ids = []
    # Both polarities are preserved. A model may select an observed component,
    # but cannot supply a new crop, threshold or mask to erase an inconvenient loop.
    for foreground, cutoff in (("dark", threshold), ("light", threshold + 1)):
        left = timeout - (time.monotonic() - started)
        if cancelled() or left <= 0:
            raise WorkspaceError("Photo inventory cancelled or deadline exhausted")
        record = analyze_reference(
            service,
            sid,
            owner,
            {
                "vision_image_sha256": source["vision_sha256"],
                "foreground": foreground,
                "threshold": min(255, cutoff),
                "fit_tolerance_mm": 0.05,
                "assumptions": {
                    "source_origin": "Automatic complete pixel inventory; no metric calibration"
                },
            },
            timeout=min(60, left),
            cancelled=cancelled,
        )
        ids.append(record["id"])
    record = {
        "policy": POLICY,
        "binding": _source_binding(service, sid, owner),
        "inventory_ids": ids,
        "job_id": jid,
        "target": None,
        "comparison": None,
        "owner_prompt": service.store.session_record(sid, owner)["prompt"],
        "owner_instructions": service.store.session_record(sid, owner).get("instructions", []),
    }
    if cancelled() or time.monotonic() - started >= timeout:
        raise WorkspaceError("Photo inventory cancelled or deadline exhausted")
    _write(service, sid, owner, jid, record, deadline=started + timeout, cancelled=cancelled)
    return {"inventory_ids": ids, "stage": "grounding"}


def ground_target(
    service: Any,
    sid: str,
    owner: str,
    jid: str,
    grounding: Any,
    contract: Any,
    *,
    timeout: float,
    cancelled: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    grounding = PhotoGrounding.model_validate(grounding)
    record = _read(service, sid, owner)
    if not record or record.get("target"):
        raise WorkspaceError("Photo grounding requires a pending pixel inventory")
    if grounding.inventory_observation_id not in record["inventory_ids"]:
        raise WorkspaceError("Grounding must use the automatic complete pixel inventory")
    if grounding.blocked_reason:
        record["blocked_reason"] = grounding.blocked_reason
        _write(service, sid, owner, jid, record, deadline=started + timeout, cancelled=cancelled)
        return {"stage": "unsupported", "reason": grounding.blocked_reason}
    assert grounding.corner_feature_ids is not None
    session = service.store.session_record(sid, owner)
    from forge_mind.workspace.intent_attribution_service import _record, request_text
    from forge_mind.workspace.photo_nominal import photo_reference_role

    attribution = _record(session)
    authoritative = attribution["proposed_contract"] if attribution else session.get("contract")
    if (
        photo_reference_role(session) != "copy"
        or contract.model_dump(mode="json") != authoritative
        or (attribution and attribution["source_text"] != request_text(session))
    ):
        raise WorkspaceError(
            "Grounding contract must be the current authenticated source interpretation"
        )
    constraints = [
        value
        for value in _constraints(contract)
        if (
            value["subject_id"] == grounding.subject_id
            and value["size_fact_ids"] == grounding.size_fact_ids
            and value["hole_count_fact_id"] == grounding.hole_count_fact_id
        )
    ]
    if len(constraints) != 1:
        raise WorkspaceError(
            "Photo grounding needs one admitted source size/count contract; never invent measurements"
        )
    constraint = deepcopy(constraints[0])
    if not grounding.planar_hypothesis or not grounding.negligible_distortion_hypothesis:
        raise WorkspaceError(
            "Metric photo construction requires explicit conditional plane/lens hypotheses"
        )
    inventory = read_profile(service, sid, owner, grounding.inventory_observation_id)
    components = [c for c in inventory["result"]["components"] if c["id"] == grounding.component_id]
    if len(components) != 1 or components[0]["touches_region_boundary"]:
        raise WorkspaceError("Photo target requires one complete observed component")
    component = components[0]
    if constraint["hole_count"] is None and constraint["hole_count_fact_id"] is None:
        constraint["hole_count"] = component["inner_loop_count"]
    if component["inner_loop_count"] != constraint["hole_count"]:
        raise WorkspaceError(
            "Complete visible opening inventory differs from the stated source count"
        )
    features = {v["id"]: v["pixel"] for v in component["loops"][0]["detected_features"]["vertices"]}
    ids = grounding.corner_feature_ids
    if len(set(ids)) != 4 or any(value not in features for value in ids):
        raise WorkspaceError(
            "Calibration must reference four distinct detected outer contour features"
        )
    width, height, thickness = constraint["size_mm"]
    plane = [(0, height), (width, height), (width, 0), (0, 0)]
    seed = component["seed_pixel"]
    request = PhotoProfileRequest.model_validate(
        {
            **inventory["request"],
            "seed_pixel": seed,
            "calibration_anchors": [
                ProfileAnchor(pixel=features[key], plane_mm=p).model_dump(mode="json")
                for key, p in zip(ids, plane, strict=True)
            ],
            "assumptions": {
                "planar": True,
                "distortion": "negligible_declared",
                "source_origin": "Conditional model-proposed correspondence to admitted source facts; not independent physical calibration",
            },
            "thickness_mm": thickness,
            "through_cuts": True,
            "allow_circle_fits": True,
            "outer_rectangle_mm": [width, height],
        }
    )
    left = timeout - (time.monotonic() - started)
    if cancelled() or left <= 0:
        raise WorkspaceError("Photo grounding cancelled or deadline exhausted")
    observation = analyze_reference(service, sid, owner, request, timeout=left, cancelled=cancelled)
    result = observation["result"]
    if not result["calibration"] or not result["proposal_document"]:
        raise WorkspaceError(
            "Photo grounding could not produce a complete consistent proposal: "
            + "; ".join(result["warnings"])
        )
    chosen = next(c for c in result["components"] if c["id"] == result["selected_component"])
    if (
        chosen["id"] != grounding.component_id
        or chosen["inner_loop_count"] != constraint["hole_count"]
    ):
        raise WorkspaceError(
            "Grounded profile component/topology differs from the frozen inventory"
        )
    outer_fit = chosen["loops"][0]["fit"]
    target = {
        "observation_id": observation["id"],
        "observation_digest": _digest(observation),
        "observation_sha256": result["observation_sha256"],
        "vision_image_sha256": result["vision_image_sha256"],
        "proposal_document": result["proposal_document"],
        "part_id": grounding.part_id,
        "source_constraints": constraint,
        "grounding": grounding.model_dump(mode="json"),
        "frame": "definition_local_xy_z0_positive_thickness",
        "policy": {
            "id": POLICY,
            "max_residual_mm": outer_fit["support_budget_mm"] + request.fit_tolerance_mm + 0.02,
            "complete_loop_coverage": True,
            "camera_refit": False,
        },
    }
    target["target_sha256"] = _digest(target)
    record.update(target=target, comparison=None)
    if cancelled() or time.monotonic() - started >= timeout:
        raise WorkspaceError("Photo grounding cancelled before target publication")
    _write(service, sid, owner, jid, record, deadline=started + timeout, cancelled=cancelled)
    return deepcopy(target)


def require_grounded(service: Any, sid: str, owner: str) -> None:
    session = service.store.get(sid, owner)
    if not applicable(session):
        return
    from forge_mind.workspace.photo_nominal import photo_reference_role

    if photo_reference_role(session) == "context":
        return
    record = _read(service, sid, owner)
    if not record or not record.get("target") or record.get("blocked_reason"):
        raise WorkspaceError("Photo construction requires the host's frozen measured target")


def record_failure(
    service: Any, sid: str, owner: str, jid: str, stage: str, rid: str | None, error: str
) -> None:
    record = _read(service, sid, owner) or {
        "policy": POLICY,
        "binding": _source_binding(service, sid, owner),
        "inventory_ids": [],
        "job_id": jid,
        "target": None,
        "comparison": None,
        "owner_prompt": service.store.session_record(sid, owner)["prompt"],
        "owner_instructions": service.store.session_record(sid, owner).get("instructions", []),
    }
    if stage == "inventory":
        record["blocked_reason"] = "Automatic pixel inventory failed: " + error
    else:
        assert rid is not None and record.get("target")
        receipt = {
            "basis": _basis(service, sid, owner, rid, record["target"]["part_id"]),
            "target_sha256": record["target"]["target_sha256"],
            "status": "needs_correction",
            "measurement": {},
            "reason": "Saved comparison failed; repair this candidate before another comparison: "
            + error,
        }
        receipt["receipt_sha256"] = _digest(receipt)
        record["comparison"] = receipt
    _write(service, sid, owner, jid, record)


def _basis(service: Any, sid: str, owner: str, rid: str, part_id: str) -> dict[str, Any]:
    from forge_mind.workspace.check_amendment import request_hash
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.verification_admission import _part_artifact

    session = service.store.get(sid, owner)
    revision = service.store.revision(sid, rid)
    document = DesignDocument.model_validate(revision["document"])
    _one_copy_scope(document, part_id)
    artifact = _part_artifact(document, part_id)
    expected = revision["measurement"]["artifact_hashes"].get(artifact)
    step = service.store.directory(sid, rid) / artifact
    if session.get("selected_revision") != rid or not expected or file_sha256(step) != expected:
        raise WorkspaceError("Photo target requires the exact current saved STEP")
    candidate = revision["measurement"]["artifact_hashes"].get("part.step")
    if not candidate or file_sha256(service.store.directory(sid, rid) / "part.step") != candidate:
        raise WorkspaceError("Whole saved candidate STEP changed")
    plan_hash = (revision.get("operation_record") or {}).get("plan_hash")
    if (
        not plan_hash
        or plan_hash != session.get("plan_hash")
        or plan_hash != request_hash(session.get("design_plan"))
    ):
        raise WorkspaceError("Photo target requires the current captured operation plan")
    captured = (revision.get("operation_record") or {}).get("plan_snapshot") or {}
    if captured.get("plan") != session.get("design_plan") or captured.get(
        "contract_hash"
    ) != revision["measurement"].get("contract_hash"):
        raise WorkspaceError("Photo target requires an exact current plan/contract snapshot")
    return {
        "revision_id": rid,
        "step_sha256": expected,
        "step_artifact": artifact,
        "candidate_step_sha256": candidate,
        "plan_hash": plan_hash,
        "job_id": session.get("job_id"),
        "contract_hash": revision["measurement"].get("contract_hash"),
        "engine_fingerprint": revision["measurement"].get("engine_fingerprint"),
    }


def _one_copy_scope(document: Any, part_id: str) -> None:
    """Count actual installed definitions without native placement operations."""
    if document.assembly is None:
        if part_id != "part" or document.production is None or document.production.role != "make":
            raise WorkspaceError("Photo copy needs one explicitly made subject")
        return
    definitions = {part.id: part for part in document.assembly.parts}
    groups = {group.id: group for group in document.assembly.subassemblies}
    pending: list[tuple[Any, frozenset[str]]] = [
        (item, frozenset()) for item in document.assembly.roots
    ]
    made = 0
    while pending:
        item, ancestors = pending.pop()
        if item.definition in definitions:
            definition = definitions[item.definition]
            if definition.production.role != "make" or definition.id != part_id:
                raise WorkspaceError(
                    "Additional installed geometry is outside this one-copy proof; retain its scope as unsupported"
                )
            made += 1
            if made > 1:
                raise WorkspaceError(
                    "One-copy photo proof does not cover multiple installed made occurrences"
                )
        elif item.definition in groups and item.definition not in ancestors:
            pending.extend(
                (child, ancestors | {item.definition}) for child in groups[item.definition].children
            )
        else:
            raise WorkspaceError("Photo copy assembly has an unresolved or cyclic occurrence")
    if made != 1:
        raise WorkspaceError("One-copy photo proof needs exactly one installed made occurrence")


def read_photo_target(
    service: Any, sid: str, owner: str, rid: str, subject_id: str
) -> dict[str, Any]:
    require_grounded(service, sid, owner)
    record = _read(service, sid, owner)
    if record is None or record.get("target") is None:
        raise WorkspaceError("No frozen photo target")
    target = record["target"]
    basis = _basis(service, sid, owner, rid, target["part_id"])
    if target["source_constraints"]["subject_id"] != subject_id:
        raise WorkspaceError("Photo target belongs to a different source subject")
    plan = service.store.get(sid, owner).get("design_plan") or {}
    parts = (plan.get("operation_plan") or {}).get("part_ids", [])
    if target["part_id"] not in parts:
        raise WorkspaceError("Frozen photo part is absent from the current operation plan")
    return {**deepcopy(target), "current_basis": basis}


def photo_closure(service: Any, sid: str, owner: str, rid: str | None) -> dict[str, Any]:
    session = service.store.get(sid, owner)
    if not applicable(session):
        return {"status": "not_applicable"}
    from forge_mind.workspace.photo_nominal import photo_reference_role

    if photo_reference_role(session) == "context":
        return {"status": "not_applicable", "reference_role": "context"}
    record = _read(service, sid, owner)
    if not record or not record.get("target") or not rid:
        return {
            "status": "comparison_needed",
            "reason": "Complete source observation and grounding before photo construction/handoff",
        }
    try:
        basis = _basis(service, sid, owner, rid, record["target"]["part_id"])
    except (WorkspaceError, OSError) as exc:
        return {
            "status": "needs_correction",
            "reason": str(exc),
            "target_sha256": record["target"]["target_sha256"],
        }
    comparison = record.get("comparison")
    if not comparison or comparison.get("basis") != basis:
        return {
            "status": "comparison_needed",
            "reason": "Automatically compare the exact current saved candidate",
            "target_sha256": record["target"]["target_sha256"],
        }
    if comparison["receipt_sha256"] != _digest(
        {k: v for k, v in comparison.items() if k != "receipt_sha256"}
    ):
        raise WorkspaceError("Photo comparison receipt changed")
    return {
        "status": comparison["status"],
        "target_sha256": record["target"]["target_sha256"],
        "receipt_sha256": comparison["receipt_sha256"],
        "basis": basis,
        "measurement": deepcopy(comparison["measurement"]),
        "whole_body": deepcopy(comparison.get("whole_body")),
        "whole_body_sha256": comparison.get("whole_body_sha256"),
        "reason": comparison["reason"],
    }


def compare_candidate(
    service: Any, sid: str, owner: str, jid: str, rid: str, *, timeout: float, cancelled: Any
) -> dict[str, Any]:
    started = time.monotonic()
    record = _read(service, sid, owner)
    if not record or not record.get("target"):
        raise WorkspaceError("Photo comparison requires a frozen target")
    target = read_photo_target(
        service, sid, owner, rid, record["target"]["source_constraints"]["subject_id"]
    )
    basis = target["current_basis"]
    left = timeout - (time.monotonic() - started)
    if cancelled() or left <= 0:
        raise WorkspaceError("Photo comparison cancelled or deadline exhausted")
    comparison = compare_reference(
        service,
        sid,
        owner,
        target["observation_id"],
        rid,
        {"body_index": 0, "z_mm": target["source_constraints"]["size_mm"][2]},
        timeout=left,
        cancelled=cancelled,
        part_id=target["part_id"],
    )
    result = comparison["result"]
    from forge_mind.workspace.photo_profile_service import _run
    from forge_mind.workspace.processing import copy_artifact, regular_file

    proof_identity = _digest({"target_sha256": target["target_sha256"], "basis": basis})
    request = {
        "protocol": "nominal_photo_solid_v1",
        "criterion_hash": proof_identity,
        "subject_id": target["source_constraints"]["subject_id"],
        "part_id": target["part_id"],
        "frame": "definition_local",
        "target_sha256": target["target_sha256"],
        "expected_document": target["proposal_document"],
        "source_constraints": target["source_constraints"],
    }
    with tempfile.TemporaryDirectory(prefix="forge-photo-solid-") as temporary:
        directory = Path(temporary)
        source = service.store.directory(sid, rid) / basis["step_artifact"]
        if not regular_file(source) or file_sha256(source) != basis["step_sha256"]:
            raise WorkspaceError("Photo whole-body STEP changed")
        copy_artifact(source, directory / "part.step")
        left = timeout - (time.monotonic() - started)
        if cancelled() or left <= 0:
            raise WorkspaceError("Photo whole-body comparison cancelled or deadline exhausted")
        whole = _run(
            service,
            directory,
            {"operation": "nominal_photo_solid", "request": request},
            left,
            cancelled,
        )
        if (
            file_sha256(directory / "part.step") != basis["step_sha256"]
            or file_sha256(source) != basis["step_sha256"]
        ):
            raise WorkspaceError("Photo whole-body staged STEP changed")
    facts = whole.get("facts") or {}
    if (
        whole.get("method") != "nominal_photo_solid_v1"
        or whole.get("criterion_hash") != proof_identity
        or whole.get("status") not in {"pass", "fail", "unverified"}
        or facts.get("target_sha256") != target["target_sha256"]
        or facts.get("expected_document_sha256") != _digest(target["proposal_document"])
        or facts.get("source_constraints") != target["source_constraints"]
        or (
            whole.get("status") != "unverified"
            and (
                facts.get("step_sha256") != basis["step_sha256"]
                or whole.get("evidence_refs")
                != [{"artifact": "part.step", "sha256": basis["step_sha256"]}]
            )
        )
    ):
        raise WorkspaceError(
            "Photo whole-body result does not match its frozen target and staged STEP"
        )
    bound = result.get("symmetric_residual_interval_mm")
    coverage = result.get("coverage") or {}
    topology = (
        result.get("saved_inner_loop_count")
        == result.get("observed_inner_loop_count")
        == target["source_constraints"]["hole_count"]
    )
    checked = bool(
        bound
        and bound[1] <= target["policy"]["max_residual_mm"]
        and topology
        and result.get("body_count") == 1
        and coverage.get("all_selected_face_wires")
        and coverage.get("all_wire_edges_supported")
        and coverage.get("camera_refitted") is False
        and whole["status"] == "pass"
    )
    from forge_mind.workspace.photo_profile_context import photo_profile_context

    receipt = {
        "basis": basis,
        "target_sha256": target["target_sha256"],
        "status": "checked" if checked else "needs_correction",
        "reason": "Fixed-frame visible planar agreement checked; physical readiness remains unverified"
        if checked
        else "Correct saved geometry against the frozen target; do not change calibration, masks, frame or tolerance. Whole-body observation: "
        + str(whole.get("reason", "unavailable")),
        "measurement": photo_profile_context(comparison, sid),
        "whole_body": whole,
        "whole_body_sha256": _digest(whole),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    if (
        cancelled()
        or time.monotonic() - started >= timeout
        or _basis(service, sid, owner, rid, target["part_id"]) != basis
    ):
        raise WorkspaceError("Photo comparison basis changed or cancelled before publication")
    record["comparison"] = receipt
    _write(service, sid, owner, jid, record, deadline=started + timeout, cancelled=cancelled)
    return photo_closure(service, sid, owner, rid)
