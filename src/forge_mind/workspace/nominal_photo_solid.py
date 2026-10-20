"""Whole saved material proof against an authenticated ordinary profile extrusion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.verification_models import Name, VerificationRecord

PROTOCOL = "nominal_photo_solid_v1"
COORDINATE_ERROR_MM = 1e-5


class PhotoSolidRequest(VerificationRecord):
    protocol: Literal["nominal_photo_solid_v1"]
    criterion_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    subject_id: Name
    part_id: Name
    frame: Literal["definition_local"]
    target_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_document: dict[str, Any]
    source_constraints: dict[str, Any]


def expected_profile_document(request: PhotoSolidRequest) -> DesignDocument:
    """Accept only the complete sketch/extrude/sequential-cut grammar, never Python."""
    document = DesignDocument.model_validate(request.expected_document)
    c = request.source_constraints
    if set(c) != {
        "subject_id",
        "frame",
        "axis",
        "size_mm",
        "hole_count",
        "size_fact_ids",
        "hole_count_fact_id",
    }:
        raise ValueError("Incomplete source constraints")
    if c["subject_id"] != request.subject_id or c["frame"] != request.frame or c["axis"] != "z":
        raise ValueError("Expected profile source frame differs")
    size = c["size_mm"]
    count = c["hole_count"]
    if (
        len(size) != 3
        or not all(
            type(value) in {int, float} and math.isfinite(value) and value > 0 for value in size
        )
        or type(count) is not int
        or count < 0
    ):
        raise ValueError("Source dimensions/count must be finite positive values")
    if document.assembly or document.mechanism or document.motion_rig or document.relations:
        raise ValueError("Expected target must be one ordinary planar extrusion")
    if document.variables != {"thickness_mm": size[2]} or len(document.features) != 2 + 3 * count:
        raise ValueError("Expected target has an extra or missing extrusion/cut")
    output = ""
    cursor = 0
    for index in range(count + 1):
        sketch, extrude = document.features[cursor : cursor + 2]
        cursor += 2
        if (
            sketch.kind != "sketch"
            or sketch.sketch is None
            or sketch.inputs
            or sketch.params
            or sketch.bindings
            or sketch.suppressed
            or sketch.sketch.plane != "XY"
            or sketch.sketch.offset != 0
            or sketch.sketch.constraints
            or extrude.kind != "extrude"
            or extrude.inputs != [sketch.id]
            or extrude.params
            or extrude.bindings != {"amount": "thickness_mm"}
            or extrude.suppressed
        ):
            raise ValueError("Expected target is not a uniform positive XY extrusion")
        entities = sketch.sketch.entities
        if not entities or any(entity.kind not in {"line", "circle"} for entity in entities):
            raise ValueError("Expected target has unsupported profile curves")
        if index == 0:
            points = {point.id: (point.x, point.y) for point in sketch.sketch.points}
            corners = {(0.0, 0.0), (size[0], 0.0), (size[0], size[1]), (0.0, size[1])}
            edges = {
                frozenset((a, b))
                for a, b in [
                    ((0.0, 0.0), (size[0], 0.0)),
                    ((size[0], 0.0), (size[0], size[1])),
                    ((size[0], size[1]), (0.0, size[1])),
                    ((0.0, size[1]), (0.0, 0.0)),
                ]
            }
            actual = {frozenset(points[key] for key in entity.points) for entity in entities}
            if (
                len(points) != 4
                or set(points.values()) != corners
                or len(entities) != 4
                or actual != edges
                or any(entity.kind != "line" for entity in entities)
            ):
                raise ValueError("Complete outer profile is not the source rectangle")
            output = extrude.id
        else:
            cut = document.features[cursor]
            cursor += 1
            if (
                cut.kind != "cut"
                or cut.inputs != [output, extrude.id]
                or cut.params
                or cut.bindings
                or cut.suppressed
            ):
                raise ValueError("Every inner loop must make one full-depth sequential cut")
            output = cut.id
        for feature in (sketch, extrude):
            if feature.code or feature.asset_id or feature.catalog_id or feature.selection:
                raise ValueError("Expected target contains an unrelated execution input")
    if document.outputs != [output]:
        raise ValueError("Expected target must publish exactly its complete final extrusion")
    return document


def evaluate_nominal_photo_solid(step_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    """No positive residual or omitted material can earn a whole-body PASS."""
    req = PhotoSolidRequest.model_validate(request)
    document = expected_profile_document(req)
    import build123d as b

    from forge_mind.workspace.kernel import compile_document
    from forge_mind.workspace.nominal_boolean import checked_boolean, material_result
    from forge_mind.workspace.photo_profile_models import PhotoProfileSelector
    from forge_mind.workspace.photo_profile_native import _saved_face
    from forge_mind.workspace.processing import regular_file

    result: dict[str, Any] = {
        "status": "unverified",
        "method": PROTOCOL,
        "criterion_hash": req.criterion_hash,
        "reason": "Whole saved photo solid not evaluated",
        "evidence_refs": [],
        "facts": {
            "target_sha256": req.target_sha256,
            "expected_document_sha256": digest(req.expected_document),
            "source_constraints": req.source_constraints,
            "scope": "Recorded source nominal and conditional image-derived CAD conformance; physical readiness unverified",
        },
    }
    if not regular_file(step_path):
        return result
    before = file_sha256(step_path)
    try:
        saved = b.import_step(step_path)
        expected, _ = compile_document(document, {})
        if (
            not saved.is_valid
            or len(saved.solids()) != 1
            or saved.volume <= 0
            or not expected.is_valid
            or len(expected.solids()) != 1
        ):
            raise ValueError("Saved/expected solid is invalid or not uniquely selected")
        size = req.source_constraints["size_mm"]
        _, loops, solids = _saved_face(
            expected, PhotoProfileSelector(body_index=0, z_mm=float(size[2]))
        )
        if solids != 1 or len(loops) - 1 != req.source_constraints["hole_count"]:
            raise ValueError(
                "Expected full-depth cuts overlap, vanish or leave incomplete coverage"
            )
        bounds = expected.bounding_box(optimal=True)
        errors = [
            abs(a - z) for a, z in zip([*bounds.min, *bounds.max], [0, 0, 0, *size], strict=True)
        ]
        if any(error > COORDINATE_ERROR_MM for error in errors):
            raise ValueError("Expected extrusion differs from source envelope/depth")
        excess = material_result(checked_boolean(saved, expected, "cut"))
        missing = material_result(checked_boolean(expected, saved, "cut"))
        states = [excess["status"], missing["status"]]
        status = "fail" if "fail" in states else "unverified" if "unverified" in states else "pass"
        result.update(
            status=status,
            reason="Two-way whole saved material comparison with source-constrained ordinary extrusion",
        )
        result["facts"].update(
            excess_material=excess,
            missing_material=missing,
            expected_inner_loop_count=len(loops) - 1,
            source_nominal={
                "size_mm": size,
                "hole_count": len(loops) - 1
                if req.source_constraints["hole_count_fact_id"] is not None
                else None,
                "straight_through": True,
                "rectangular_outline": True,
            },
            derived_profile={
                "expected_document_sha256": digest(req.expected_document),
                "target_sha256": req.target_sha256,
                "hole_count": len(loops) - 1,
            },
        )
        if not regular_file(step_path) or file_sha256(step_path) != before:
            raise ValueError("Saved STEP changed during whole-body inspection")
        result["facts"]["step_sha256"] = before
        result["evidence_refs"] = [{"artifact": step_path.name, "sha256": before}]
    except Exception as exc:
        result.update(
            status="unverified",
            reason=f"Whole saved photo inspection unavailable: {type(exc).__name__}: {str(exc)[:300]}",
        )
    return result
