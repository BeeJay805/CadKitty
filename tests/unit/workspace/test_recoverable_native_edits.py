"""Scripted native recovery, focused edits and exact delivery; no model quality claim."""

from __future__ import annotations

import hashlib
import json
import math
from io import BytesIO
from zipfile import ZipFile

import build123d as b
import pytest
from tests.unit.workspace.test_operation_authoring_flow import contract, document, plan

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.measure import verify_artifacts
from forge_mind.workspace.models import DesignPlan, Measurement
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.print_setup_service import print_setup_context, read_print_export
from forge_mind.workspace.project_delivery import read_project_export
from forge_mind.workspace.store import Store


@pytest.mark.slow
def test_native_job_recovers_reader_and_applies_pinned_feature_edit_before_exact_delivery(tmp_path):
    stock = document(include_bore=False).model_dump()
    features, output = stock.pop("features"), stock.pop("outputs")[0]
    stock["assembly"] = {
        "parts": [
            {
                "id": "mount",
                "name": "Plate",
                "features": features,
                "output": output,
                "production": {
                    "role": "make",
                    "deliver_stl": True,
                    "material": {"name": "PLA"},
                    "process": "additive",
                },
            }
        ],
        "roots": [{"id": "plate_occ", "definition": "mount"}],
    }
    design = DesignDocument.model_validate(stock)
    complete_plan = plan().model_dump()
    bore_step = complete_plan["operation_plan"]["steps"].pop()
    initial_plan = DesignPlan.model_validate(complete_plan)
    appended = [feature.model_dump() for feature in document().features[1:]]
    contexts = []
    initial_revision = None

    class ScriptedAuthor:
        model = "scripted-recoverable-native-edits"

        def generate_structured(self, **kwargs):
            nonlocal initial_revision
            assert kwargs["output_model"] is NativeAction
            context = json.loads(kwargs["user"])
            contexts.append(context)
            index = len(contexts)
            if index == 1:
                action = NativeAction(
                    kind="plan",
                    explanation="Save the initial stock construction plan",
                    document_json=initial_plan.model_dump_json(),
                )
            elif index == 2:
                action = NativeAction(
                    kind="document",
                    explanation="Construct the valid stock checkpoint",
                    document_json=design.model_dump_json(),
                )
            elif index == 3:
                initial_revision = context["selected"]["id"]
                action = NativeAction(
                    kind="read_design",
                    explanation="Exercise recoverable invalid cursor input",
                    query="features",
                    revision_id=initial_revision,
                    document_json='{"scope":"mount","cursor":0,"limit":100}',
                )
            elif index == 4:
                assert context["selected"]["id"] == initial_revision
                rejected = next(
                    row["action_result"]
                    for row in context["history"]
                    if row.get("action_result", {}).get("field") == "read_design.cursor"
                )
                assert rejected["applied"] is False and rejected["retryable"] is True
                action = NativeAction(
                    kind="read_design",
                    explanation="Retry the same revision with a valid first-page request",
                    query="features",
                    revision_id=initial_revision,
                    document_json='{"scope":"mount","limit":100}',
                )
            elif index == 5:
                page = next(
                    row["read_design"] for row in context["history"] if "read_design" in row
                )
                assert [feature["id"] for feature in page["items"]] == ["base"]
                action = NativeAction(
                    kind="plan",
                    query="patch",
                    explanation="Add the bore construction step without applying CAD",
                    document_json=json.dumps(
                        {"basis": context["edit_basis"], "operation_steps": [bore_step]}
                    ),
                )
            elif index == 6:
                assert context["selected"]["id"] == initial_revision
                assert (
                    context["edit_basis"]["plan_version"]
                    == contexts[-2]["edit_basis"]["plan_version"] + 1
                )
                plan_result = next(
                    row["plan_patch"] for row in context["history"] if "plan_patch" in row
                )
                assert plan_result["plan_applied"] is True and plan_result["cad_applied"] is False
                action = NativeAction(
                    kind="feature_edit",
                    explanation="Append the bore cutter and cut, then select the new output",
                    document_json=json.dumps(
                        {
                            "basis": context["edit_basis"],
                            "changes": [
                                {"part_id": "mount", "append": appended, "output": "bored"}
                            ],
                        }
                    ),
                )
            elif index == 7:
                assert context["selected"]["id"] != initial_revision
                action = NativeAction(
                    kind="print_setup",
                    query="prepare",
                    revision_id=context["selected"]["id"],
                    explanation="Prepare current edited made-part files before instruction review",
                )
            elif index == 8:
                guide = context["delivery_guidance"]
                assert guide["basis"]["revision_id"] == context["selected"]["id"]
                assert guide["facts"]["print_setup"]["status"] == "current"
                action = NativeAction(
                    kind="guidance",
                    query="finish",
                    explanation="Bind current instructions and request exact saved delivery",
                    document_json=json.dumps(
                        {
                            "revision_id": guide["basis"]["revision_id"],
                            "basis_hash": guide["basis_hash"],
                            "reviewed_current_facts": True,
                            "guidance": {
                                "print": "Select a printer profile and test physical dimensions before use.",
                                "installation": "The plate is nominally 40 mm wide and 6 mm thick, with a 4 mm bore. Select suitable hardware.",
                                "use_and_removal": "Keep the bore and removal path clear. Check fit after manufacture.",
                                "unresolved": "Printed fit, material strength and physical use remain unverified.",
                            },
                        }
                    ),
                )
            else:
                pytest.fail(
                    "Recovery and bound instruction finish required an unexpected author action"
                )
            return action, LLMUsage(self.model, 0, 0)

        generate_structured_with_image = generate_structured

    service = WorkspaceService(
        Store(tmp_path), Executor("trusted_local"), lambda **_: ScriptedAuthor()
    )
    sid = service.store.create(
        "owner",
        "40 mm width, 4 mm bore and 6 mm plate; deliver the current STEP and STL files",
        contract().model_dump(),
        authoring_mode="native_operations",
    )["id"]
    service.store.enqueue(
        sid,
        "owner",
        {"max_actions": 9, "max_seconds": 240, "delivery_requested": True},
    )
    job = service.store.claim()
    assert job is not None
    service.run_job(job)
    current = service.store.get(sid, "owner")
    assert len(contexts) == 8, json.dumps(current["events"])
    assert current["state"] == "contract_satisfied", json.dumps(current["events"])
    assert current.get("accepted_revision") is None
    assert current["native_review"]["status"] == "ready_for_validation"
    assert current["last_edit_result"]["applied"] is True
    assert current["last_edit_result"]["job_id"] == job["id"]
    assert len(service.store.revisions(sid)) == 2
    revision = service.store.revision(sid, current["selected_revision"])
    assert revision["parent_id"] == initial_revision
    assert revision["operation_record"]["graph_complete"] is True
    assert revision["operation_record"]["plan_hash"] == current["plan_hash"]
    assert all(not step["effect_verified"] for step in revision["operation_record"]["steps"])
    assert all(finding["status"] == "pass" for finding in revision["measurement"]["findings"])
    assert [
        feature["id"] for feature in revision["document"]["assembly"]["parts"][0]["features"]
    ] == ["base", "tool", "bored"]
    directory = service.store.directory(sid, revision["id"])
    assert verify_artifacts(directory, Measurement.model_validate(revision["measurement"]))
    setup = print_setup_context(service, sid, "owner", revision["id"])
    assert setup["status"] == "current"
    print_record = setup["record"]
    assert print_record["result"]["status"] == "prepared"
    assert {row["part_id"]: row["quantity"] for row in print_record["files"].values()} == {
        "mount": 1
    }
    print_package = print_record["package"]
    print_bytes, print_receipt = read_print_export(
        service, sid, "owner", print_package["package_id"], print_package["sha256"]
    )
    assert print_receipt == print_package
    assert len(print_bytes) == print_package["bytes"]
    assert hashlib.sha256(print_bytes).hexdigest() == print_package["sha256"]
    with ZipFile(BytesIO(print_bytes)) as prepared_files:
        for name, metadata in print_record["files"].items():
            mesh = prepared_files.read(name)
            assert len(mesh) == metadata["bytes"]
            assert hashlib.sha256(mesh).hexdigest() == metadata["sha256"]
    delivery = current["automatic_delivery"]
    assert delivery["status"] == "delivered" and delivery["guidance_status"] == "current"
    receipt = delivery["receipt"]
    content, readback = read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert readback == receipt
    assert hashlib.sha256(content).hexdigest() == receipt["sha256"]
    assert len(content) == receipt["bytes"]
    with ZipFile(BytesIO(content)) as archive:
        for member, saved in (
            ("assembly/part.step", "part.step"),
            ("make/definition_mount.step", "definition_mount.step"),
            ("make/definition_mount.stl", "definition_mount.stl"),
            ("document.json", "document.json"),
        ):
            assert archive.read(member) == (directory / saved).read_bytes()
        production = json.loads(archive.read("production.json"))
        assert [(part["part_id"], part["quantity"]) for part in production["make"]] == [
            ("mount", 1)
        ]
        assert "physical" in archive.read("README.md").decode().lower()
    shape = b.import_step(directory / "definition_mount.step")
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(40 * 24 * 6 - math.pi * 2**2 * 6)
    assert current.get("accepted_revision") is None
