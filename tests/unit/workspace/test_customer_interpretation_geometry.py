"""Ordinary target admission never substitutes operation parameters for STEP proof."""

from __future__ import annotations

import build123d as b
import pytest
from tests.unit.workspace.test_customer_interpretation import SOURCE, wire

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.nominal_perforated_box import evaluate_nominal_perforated_box
from forge_mind.workspace.source_delivery_contract_draft import lower_source_delivery_draft
from forge_mind.workspace.verification_admission import criterion_hash


@pytest.mark.parametrize(
    "defect", [None, "extra_bore", "diameter", "blind_bore", "width", "rotated"]
)
def test_interpreted_body_checks_actual_complete_saved_geometry(tmp_path, defect):
    lowered = lower_source_delivery_draft(wire(), SOURCE)
    criterion = lowered.contract.requirements[-1].verification
    checksum = criterion_hash(criterion.model_dump(mode="json"))
    request = {
        key: value
        for key, value in criterion.model_dump(mode="json").items()
        if key not in {"version", "kind", "scope", "source_quote", "subject_role"}
    }
    request.update(protocol="nominal_perforated_box_v1", criterion_hash=checksum, part_id="part")
    body = b.Box(61 if defect == "width" else 60, 40, 5)
    for index, hole in enumerate(criterion.holes):
        radius = 1.8 if defect == "diameter" and index == 0 else hole.diameter_mm / 2
        blind = defect == "blind_bore" and index == 0
        body = body - b.Solid.make_cylinder(
            radius,
            3 if blind else 7,
            b.Plane(origin=(*hole.center_mm, 0 if blind else -3.5)),
        )
    if defect == "extra_bore":
        body = body - b.Solid.make_cylinder(0.5, 7, b.Plane(origin=(0, 0, -3.5)))
    if defect == "rotated":
        body = body.rotate(b.Axis.Z, 90)
    path = tmp_path / "part.step"
    b.export_step(body, path)
    result = evaluate_nominal_perforated_box(path, request)
    assert result["criterion_hash"] == checksum
    assert result["facts"]["step_sha256"] == file_sha256(path)
    assert result["evidence_refs"] == [{"artifact": "part.step", "sha256": file_sha256(path)}]
    if defect is None:
        assert result["status"] == "pass", result
        assert len(result["facts"]["pattern"]["observed_holes"]) == 4
        assert result["facts"]["qualification_status"] == "unverified"
    else:
        assert result["status"] != "pass", result
