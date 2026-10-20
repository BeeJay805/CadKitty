"""Real native construction, fresh plan capture, final review and exact export."""

import json
from io import BytesIO
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_native_process_completion import baseline, guide_payload

from forge_mind.workspace.construction_transaction import build_construction, prepare_construction
from forge_mind.workspace.delivery_guidance import GuidanceUpdate, guidance_context, update_guidance
from forge_mind.workspace.native_finish import finish_context
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.project_delivery import create_project_export, read_project_export


def test_real_construction_build_captures_successor_and_delivers_current_plan(tmp_path):
    service, sid, original, baseline_revision = baseline(tmp_path)
    service.store.enqueue(sid, "owner", {"preview_only": False, "delivery_requested": True})
    job = service.store.claim()
    assert job is not None
    edited = original.model_copy(update={"variables": {"width": 44}})
    patch = PlanPatch.model_validate(
        {
            "basis": service.store.edit_basis(sid, "owner").model_dump(),
            "dimensions": [
                {
                    "id": "width_edit",
                    "value": 44,
                    "source": "customer",
                    "source_quote": "44 mm width",
                    "part_id": "mount",
                    "binding_name": "width",
                    "supersedes_fact_id": "width_source",
                }
            ],
            "verification": ["Read the current saved width and through bore"],
        }
    )
    prepared = prepare_construction(
        service,
        sid,
        "owner",
        job["id"],
        edited,
        patch,
        author_intent="Change to 44 mm width; provide files",
    )
    outcome = build_construction(
        service,
        sid,
        "owner",
        job["id"],
        prepared,
        explanation="Apply explicit width and intent together",
        timeout=90,
        cancelled=lambda: False,
    )
    assert outcome["applied"] and outcome["plan_applied"] and outcome["cad_applied"]
    rid = outcome["revision_id"]
    saved = service.store.revision(sid, rid)
    assert saved["parent_id"] == baseline_revision["id"]
    assert saved["measurement"]["geometry_valid"]
    assert saved["measurement"]["facts"]["extents_mm"][0] == pytest.approx(44)
    assert saved["operation_record"]["plan_snapshot"] == prepared.successor_snapshot
    directory = service.store.directory(sid, rid)
    assert (directory / "part.step").read_bytes().startswith(b"ISO-10303-21;")
    assert (
        json.loads((directory / "operation-plan.json").read_text()) == prepared.successor_snapshot
    )
    update_guidance(
        service,
        sid,
        "owner",
        GuidanceUpdate.model_validate_json(
            guide_payload(
                {
                    "delivery_guidance": guidance_context(service, sid, "owner", rid),
                }
            )
        ),
    )
    review = finish_context(service, sid, "owner", rid)
    assert review["status"] == "ready_for_validation", review["reasons"]
    receipt = create_project_export(service, sid, "owner", rid, include_stl=True)
    package, verified = read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert verified == receipt
    with ZipFile(BytesIO(package)) as archive:
        assert archive.read("document.json") == (directory / "document.json").read_bytes()
        plan_files = [name for name in archive.namelist() if name.endswith("operation-plan.json")]
        assert plan_files
        assert json.loads(archive.read(plan_files[0])) == prepared.successor_snapshot
    assert service.store.get(sid, "owner").get("accepted_revision") is None
