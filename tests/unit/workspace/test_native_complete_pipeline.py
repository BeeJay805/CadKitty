"""Real native author dispatch, saved-file delivery and scoped numeric edit."""

import hashlib
import json
from io import BytesIO
from zipfile import ZipFile

from tests.unit.workspace.test_operation_authoring_flow import contract, document, plan

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import DesignPlan
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.print_setup_service import print_setup_context, read_print_export
from forge_mind.workspace.project_delivery import read_project_export
from forge_mind.workspace.store import Store


def test_native_author_and_numeric_edit_deliver_exact_packages_without_accepting(tmp_path):
    source = document().model_dump()
    source["variables"]["divisor"] = 1
    source["relations"] = [{"name": "inverse", "operation": "divide", "a": 1, "b": "divisor"}]
    features, output = source.pop("features"), source.pop("outputs")[0]
    source["assembly"] = {
        "parts": [
            {
                "id": "mount",
                "name": "Mount",
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
        "roots": [{"id": "mount_occ", "definition": "mount"}],
    }
    design = DesignDocument.model_validate(source)
    intent = plan().model_dump()
    intent["operation_plan"]["dimensions"] = [
        {
            "id": "width_fact",
            "value": 40,
            "source": "customer",
            "source_quote": "40 mm width",
            "binding_name": "width",
            "part_id": "mount",
        }
    ]
    intent["operation_plan"]["steps"].append(
        {
            "id": "inspect_mount",
            "part_id": "mount",
            "kind": "inspection",
            "referenced_feature_ids": ["bored"],
            "depends_on": ["bore"],
            "expected_effect": "Inspect the produced mounting bore",
        }
    )
    intent = DesignPlan.model_validate(intent)
    actions = iter(
        [
            NativeAction(
                kind="plan",
                explanation="Retain source facts",
                document_json=intent.model_dump_json(),
            ),
            NativeAction(
                kind="document",
                explanation="Construct mount",
                document_json=design.model_dump_json(),
            ),
            NativeAction(
                kind="print_setup",
                query="prepare",
                explanation="Prepare current made-part print files before instruction review",
            ),
            NativeAction(kind="guidance", explanation="Review current mount instructions"),
            NativeAction(kind="finish", explanation="Deliver checked CAD; physical use is unknown"),
        ]
    )
    calls = []

    class Author:
        model = "scripted-native-interface"

        def generate_structured(self, **kwargs):
            assert kwargs["output_model"] is NativeAction
            context = json.loads(kwargs["user"])
            assert "native_authoring" in context
            calls.append(context)
            action = next(actions)
            if action.kind == "guidance":
                guide = context["delivery_guidance"]
                action.document_json = json.dumps(
                    {
                        "revision_id": guide["basis"]["revision_id"],
                        "basis_hash": guide["basis_hash"],
                        "reviewed_current_facts": True,
                        "guidance": {
                            "print": "Printing settings and material require physical tests.",
                            "installation": "Mount is 40 mm wide with a 4 mm bore. Select suitable hardware.",
                            "use_and_removal": "Check removal access before installation.",
                            "unresolved": "Physical strength and fit remain unverified.",
                        },
                    }
                )
            return action, LLMUsage(self.model, 0, 0)

        generate_structured_with_image = generate_structured

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"), lambda **kw: Author())
    sid = service.store.create(
        "owner",
        "40 mm width, 4 mm bore and 6 mm plate; deliver STL",
        contract().model_dump(),
        authoring_mode="native_operations",
    )["id"]
    service.store.enqueue(sid, "owner", {"max_actions": 5, "max_seconds": 120})
    job = service.store.claim()
    assert job
    service.run_job(job)
    initial = service.store.get(sid, "owner")
    assert initial["state"] == "contract_satisfied", initial["events"]
    assert initial["automatic_delivery"]["status"] == "delivered"
    assert initial["automatic_delivery"]["guidance_status"] == "current"
    assert initial.get("accepted_revision") is None
    first = initial["selected_revision"]
    setup = print_setup_context(service, sid, "owner", first)
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
    with ZipFile(BytesIO(print_bytes)) as prepared_files:
        for name, metadata in print_record["files"].items():
            mesh = prepared_files.read(name)
            assert len(mesh) == metadata["bytes"]
            assert hashlib.sha256(mesh).hexdigest() == metadata["sha256"]
    first_record = service.store.revision(sid, first)["operation_record"]
    assert (
        first_record["graph_complete"]
        and first_record["dimension_bindings"][0]["status"] == "unchanged"
    )
    assert "plan_snapshot" not in calls[-1]["selected"]["operation_record"]
    saved_fact = first_record["plan_snapshot"]["plan"]["operation_plan"]["dimensions"][0]
    receipt = initial["automatic_delivery"]["receipt"]
    content, returned = read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert returned == receipt and len(content) == receipt["bytes"]
    service.store.enqueue(sid, "owner", {"parameters": {"width": 44}, "max_seconds": 120})
    edit = service.store.claim()
    assert edit
    service.run_job(edit)
    current = service.store.get(sid, "owner")
    assert len(calls) == 5  # deterministic numeric edits do not ask a model
    assert current["automatic_delivery"]["status"] == "delivered", current["automatic_delivery"]
    assert current["automatic_delivery"]["revision_id"] != first
    assert current["automatic_delivery"]["guidance_status"] == "stale"
    record = service.store.revision(sid, current["selected_revision"])["operation_record"]
    assert record["dimension_bindings"][0]["status"] == "changed"
    assert record["dimension_bindings"][0]["current_value"] == 44
    assert record["plan_snapshot"]["plan"]["operation_plan"]["dimensions"][0] == saved_fact
    assert current.get("accepted_revision") is None
    service.store.enqueue(sid, "owner", {"parameters": {"divisor": 0}, "max_seconds": 120})
    invalid = service.store.claim()
    assert invalid
    service.run_job(invalid)
    refused = service.store.get(sid, "owner")
    assert len(calls) == 5
    assert refused["selected_revision"] == current["selected_revision"]
    assert refused["automatic_delivery"]["status"] == "incomplete"
    assert refused["automatic_delivery"]["receipt"] is None


def test_native_flat_finish_keeps_current_candidate_and_its_guidance(tmp_path):
    first = document().model_dump()
    second = document(44).model_dump()
    second["features"][0]["params"]["z"] = 8
    for source in (first, second):
        source["production"] = {"role": "make"}
    intent = {
        "objective": "Keep the latest edited mount",
        "operation_plan": {
            "part_ids": ["part"],
            "steps": [
                {
                    "id": "make",
                    "part_id": "part",
                    "feature_ids": ["base", "tool", "bored"],
                    "requirement_ids": ["opening", "thickness"],
                    "expected_effect": "Bored plate",
                }
            ],
        },
    }
    actions = iter(
        [
            NativeAction(
                kind="plan",
                explanation="Keep explicit native plan",
                document_json=json.dumps(intent),
            ),
            NativeAction(
                kind="document",
                explanation="First mount meets the scoped contract",
                document_json=json.dumps(first),
            ),
            NativeAction(
                kind="document",
                explanation="Latest mount retains an honest thickness failure",
                document_json=json.dumps(second),
            ),
            NativeAction(kind="guidance", explanation="Review instructions for the latest mount"),
        ]
    )
    reviewed = []

    class Author:
        model = "scripted-current-candidate"

        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            action = next(
                actions,
                NativeAction(
                    kind="finish", explanation="Save latest CAD and its open thickness failure"
                ),
            )
            if action.kind == "guidance":
                guide = context["delivery_guidance"]
                reviewed.append(guide["basis"]["revision_id"])
                action.document_json = json.dumps(
                    {
                        "revision_id": guide["basis"]["revision_id"],
                        "basis_hash": guide["basis_hash"],
                        "reviewed_current_facts": True,
                        "guidance": {
                            "print": "Printing is unqualified.",
                            "installation": "Latest plate is 44 mm wide and 8 mm thick.",
                            "use_and_removal": "Test removal access.",
                            "unresolved": "Thickness differs from the frozen 6 mm requirement. Physical use is unverified.",
                        },
                    }
                )
            return action, LLMUsage(self.model, 0, 0)

        generate_structured_with_image = generate_structured

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"), lambda **kw: Author())
    sid = service.store.create(
        "owner",
        "Keep latest mount; export saved CAD files",
        contract().model_dump(),
        authoring_mode="native_operations",
    )["id"]
    service.store.enqueue(
        sid, "owner", {"max_actions": 6, "max_seconds": 120, "delivery_requested": True}
    )
    job = service.store.claim()
    assert job
    service.run_job(job)
    session = service.store.get(sid, "owner")
    assert session["selected_revision"] == reviewed[0]
    assert session["state"] == "needs_review"
    assert session["automatic_delivery"]["guidance_status"] == "current"
    selected = service.store.revision(sid, reviewed[0])
    assert selected["measurement"]["facts"]["extents_mm"] == [44, 24, 8]
    assert any(
        f["requirement_id"] == "thickness" and f["status"] == "fail"
        for f in selected["measurement"]["findings"]
    )


def test_active_plan_uses_its_captured_contract_before_new_geometry(tmp_path, monkeypatch):
    from forge_mind.workspace.completion import completion_context

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("owner", "Mount", {}, authoring_mode="native_operations")["id"]
    target = contract()
    intent = plan().model_dump()
    intent["subsystems"] = [
        {"id": "bore_work", "name": "Bore work", "requirement_ids": ["opening"]}
    ]
    service.store.set_design_plan(
        sid, "owner", DesignPlan.model_validate(intent), expected_plan_version=0, contract=target
    )
    baseline = {
        "id": "a" * 32,
        "contract": {"requirements": [target.requirements[1].model_dump()]},
        "document": document(include_bore=False).model_dump(),
        "measurement": {"facts": {}, "artifact_hashes": {}},
    }
    monkeypatch.setattr(service, "_load_revision", lambda *args: baseline)
    context = completion_context(service, sid, "owner", baseline["id"])
    assert context.get("operation_plan_error") is None
    assert context["next_work_category"] == "construction"
    assert context["next_target"] == "bore"
    assert next(
        row for row in context["requirement_dispositions"] if row["requirement_id"] == "opening"
    )


def test_direct_native_document_job_clears_stale_delivery_and_captures_new_files(
    tmp_path, monkeypatch
):
    from forge_mind.workspace import document_service, native_delivery

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("owner", "Mount", {}, authoring_mode="native_operations")["id"]
    service.store.set_design_plan(
        sid, "owner", plan(), expected_plan_version=0, contract=contract()
    )
    service.store.update(sid, automatic_delivery={"status": "delivered", "revision_id": "old"})
    service.store.enqueue(sid, "owner", {"document": document().model_dump(), "max_seconds": 120})
    job = service.store.claim()
    assert job
    candidate = "a" * 32
    monkeypatch.setattr(document_service, "build_document", lambda *args, **kw: {"id": candidate})
    captures = []

    def capture(service, job, deadline, cancelled, state):
        session = service.store.get(sid, "owner")
        assert session["automatic_delivery"] is None
        assert session["selected_revision"] == candidate
        assert state == "needs_review" and deadline is not None
        captures.append(candidate)
        service.store.update(
            sid, automatic_delivery={"status": "captured", "revision_id": candidate}
        )

    monkeypatch.setattr(native_delivery, "capture_native_delivery", capture)
    service.run_job(job)
    assert captures == [candidate]
    assert service.store.get(sid, "owner")["state"] == "needs_review"
