"""An interrupted author resumes saved CAD through exact delivery without acceptance."""

import json
from io import BytesIO
from zipfile import ZipFile

from tests.unit.workspace.test_native_process_completion import baseline, guide_payload

from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.project_delivery import read_project_export
from forge_mind.workspace.store import Store


def test_reopened_native_author_finishes_same_saved_candidate_and_requested_files(tmp_path):
    service, sid, _, revision = baseline(tmp_path)
    original_step = (service.store.directory(sid, revision["id"]) / "part.step").read_bytes()
    source_plan = service.store.get(sid, "owner")["operation_plan_snapshot"]

    class InterruptedAuthor:
        model = "scripted-interruption"

        def generate_structured(self, **kwargs):
            raise LLMError("Simulated interrupted provider")

        generate_structured_with_image = generate_structured

    service.client_factory = lambda **kw: InterruptedAuthor()
    first_id = service.store.enqueue(
        sid, "owner", {"max_actions": 2, "max_seconds": 120, "delivery_requested": True}
    )
    service.run_job(service.store.claim())
    interrupted = service.store.get(sid, "owner")
    assert interrupted["state"] == "error"
    assert interrupted["job_id"] == first_id
    assert interrupted["resume_candidate"]["revision_id"] == revision["id"]
    assert interrupted["automatic_delivery"]["status"] == "incomplete"
    assert interrupted.get("accepted_revision") is None
    calls = []

    class FinishAuthor:
        model = "scripted-resume"

        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            calls.append(context)
            assert context["selected"]["id"] == revision["id"]
            assert context["design_plan"] == source_plan["plan"]
            payload = json.loads(guide_payload(context))
            payload["guidance"]["installation"] = (
                "Check the 40 mm mount width against the installation space."
            )
            return NativeAction(
                kind="guidance",
                query="finish",
                document_json=json.dumps(payload),
                explanation="Finish current instructions and deliver the saved candidate",
            ), LLMUsage(self.model, 0, 0)

        generate_structured_with_image = generate_structured

    reopened = WorkspaceService(
        Store(tmp_path), Executor("trusted_local"), lambda **kw: FinishAuthor()
    )
    resumed_id = reopened.store.enqueue(
        sid, "owner", {"resume": True, "preview_only": True, "max_actions": 2, "max_seconds": 120}
    )
    job = reopened.store.claim()
    assert job["id"] == resumed_id and job["delivery_requested"] is True
    assert job["preview_only"] is False
    reopened.run_job(job)
    result = reopened.store.get(sid, "owner")
    assert len(calls) == 1
    assert result["selected_revision"] == revision["id"]
    assert len(result["revisions"]) == 1
    assert result["state"] == "contract_satisfied"
    assert result.get("accepted_revision") is None
    assert result["resume_candidate"] is None
    assert result["native_review"]["status"] == "ready_for_validation"
    assert result["automatic_delivery"]["status"] == "delivered"
    assert result["automatic_delivery"]["guidance_status"] == "current"
    assert result["operation_plan_snapshot"] == source_plan
    assert (
        reopened.store.directory(sid, revision["id"]) / "part.step"
    ).read_bytes() == original_step
    receipt = result["automatic_delivery"]["receipt"]
    content, readback = read_project_export(
        reopened, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert receipt == readback
    with ZipFile(BytesIO(content)) as archive:
        readme = archive.read("README.md").decode()
        guidance = json.loads(archive.read("customer-guidance.json"))
        assert guidance["status"] == "current"
        assert guidance["record"]["guidance"]["installation"] == (
            "Check the 40 mm mount width against the installation space."
        )
        assert guidance["instruction_binding"]["status"] == "legacy_unverified"
        assert "Instruction binding status: legacy_unverified." in readme
        assert "[customer-guidance.json](customer-guidance.json)" in readme
        assert archive.read("assembly/part.step") == original_step
        assert archive.read("make/part.step") == original_step
