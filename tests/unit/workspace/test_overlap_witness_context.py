import json

import build123d as b
import pytest
from pydantic import ValidationError

from forge_mind.workspace.agent_memory import remember
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.engineering import assembly_review, describe_overlap_witnesses
from forge_mind.workspace.engineering_models import EngineeringInputs, JointInput, ServicePath


def test_reference_and_claimed_intent_never_waive_a_measured_overlap():
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": name,
                        "name": name,
                        "production": {"role": role},
                        "source": "Author says intentional contact; this is not permission",
                        "features": [{"id": "box", "name": "Box", "kind": "box"}],
                        "output": "box",
                    }
                    for name, role in [("reservation", "reference"), ("support", "make")]
                ],
                "roots": [
                    {"id": "space", "definition": "reservation"},
                    {"id": "mount", "definition": "support"},
                ],
            }
        }
    )
    report, _ = assembly_review([b.Box(2, 2, 2), b.Box(1, 1, 1)], EngineeringInputs(), "r")
    facts = {
        "occurrences": [
            {"body_index": 0, "path": "space", "definition": "reservation"},
            {"body_index": 1, "path": "mount", "definition": "support"},
        ]
    }
    original_pairs = json.dumps(report["pairs"])
    describe_overlap_witnesses(report, document, facts)
    witness = report["overlap_witnesses"][0]
    assert witness["context"] == "reference_involved"
    assert witness["a_path"] == "space" and witness["b_path"] == "mount"
    assert witness["overlap_mm3"] == pytest.approx(1)
    assert witness["status"] == report["pose_status"] == report["status"] == "fail"
    assert json.dumps(report["pairs"]) == original_pairs
    assert report["witness_bodies"]["0"]["role"] == "reference"
    assert "not verified" in report["interpretation_scope"]
    from forge_mind.workspace.parameter_study import summarize_report

    summary = summarize_report({"assembly": report}, {"assembly": facts})
    assert summary["collisions"][0]["context"] == "reference_involved"
    assert summary["status"] == "fail"
    assert summary["interpretation_scope"] == report["interpretation_scope"]


def test_historical_report_without_definitions_keeps_numeric_witness():
    report = {"pairs": [{"a": 0, "b": 1, "overlap_mm3": 2, "status": "fail"}]}
    describe_overlap_witnesses(report, None, None)
    assert report["overlap_witnesses"][0]["a"] == 0
    assert report["overlap_witnesses"][0]["a_path"] is None
    assert report["witness_bodies"] == {}


def test_large_review_preserves_actionable_sibling_and_receipt_across_resume():
    collisions = {
        "status": "fail",
        "collisions": [
            {"a": "frame/post", "b": f"fasteners/screw_{i}", "overlap_mm3": 9.7} for i in range(48)
        ],
    }
    rows = remember(
        [
            {
                "engineering_report_receipt": {"id": "report-id", "revision_id": "revision-id"},
                "static_interference": collisions,
                "engineering_review": {"assembly_definitions": "x" * 90000},
            }
        ]
    )
    assert rows[0]["static_interference"] == collisions
    assert rows[0]["engineering_report_receipt"]["id"] == "report-id"
    assert rows[0]["omitted_keys"] == ["engineering_review"]
    assert len(json.dumps(rows)) <= 16000
    assert remember(rows)[0]["static_interference"] == collisions


@pytest.mark.parametrize("index", [64, 113, 127, 128, 10000])
def test_engineering_inputs_address_entire_supported_assembly(index):
    path = ServicePath(id="remove", body=index, travel=(0, 0, 10), task="Remove panel")
    joint = JointInput(id="mount", kind="fixed", parent=0, child=index)
    inputs = EngineeringInputs(assembly_revision="a" * 32, service_paths=[path], joints=[joint])
    assert inputs.service_paths[0].body == index
    with pytest.raises(ValueError, match="missing body"):
        assembly_review([b.Box(1, 1, 1)], inputs, "a" * 32)


def test_engineering_inputs_reject_negative_indices():
    with pytest.raises(ValidationError):
        ServicePath(id="remove", body=-1, travel=(0, 0, 10), task="Remove panel")
    with pytest.raises(ValidationError):
        JointInput(id="mount", kind="fixed", parent=-1, child=0)
