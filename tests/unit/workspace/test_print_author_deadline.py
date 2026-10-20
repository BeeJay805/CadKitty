"""Author dispatch must stop print publication after its absolute deadline."""

from __future__ import annotations

import json
import time

import pytest
from tests.unit.workspace.test_project_delivery import RID, saved_project

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import agent, completion_inline, print_setup_service
from forge_mind.workspace.models import Action, Contract
from forge_mind.workspace.store import WorkspaceError


@pytest.mark.parametrize("operation", ["inspect", "prepare"])
def test_print_action_checks_absolute_deadline_after_staging(tmp_path, monkeypatch, operation):
    service, sid, directory = saved_project(tmp_path, monkeypatch)
    service.store.update(sid, contract=Contract().model_dump())
    original_files = {path.name: path.read_bytes() for path in directory.iterdir()}
    deadline_expired = False
    callback_observations = []
    publication_attempts = []
    calls = []

    # This test exercises dispatch and cancellation, not native geometry qualification.
    monkeypatch.setattr(agent, "expired", lambda deadline: deadline_expired)
    monkeypatch.setattr(
        completion_inline,
        "author_checkpoint",
        lambda *args: {"status": "unverified", "reason": "Saved byte fixture; no native checks"},
    )

    def forbid_native(*args, **kwargs):
        pytest.fail("Deadline dispatch must not start a native worker")

    monkeypatch.setattr(service.executor, "build", forbid_native)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbid_native)

    def staged_print(service_arg, project, owner, revision, *args, timeout, cancelled):
        nonlocal deadline_expired
        assert service_arg is service and (project, owner, revision) == (sid, "owner", RID)
        assert 0 < timeout <= 30
        assert args == (("base",) if operation == "inspect" else ())
        callback_observations.append(cancelled())
        # Source staging can use the time that remained when dispatch began.
        deadline_expired = True
        stopped = cancelled()
        callback_observations.append(stopped)
        if not stopped:
            publication_attempts.append(operation)
        raise WorkspaceError("Print setup stopped after staging; no output was published")

    target = "inspect_print_setup" if operation == "inspect" else "prepare_print_files"
    monkeypatch.setattr(print_setup_service, target, staged_print)

    class ScriptedAuthor:
        model = "scripted-print-author-deadline"

        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            calls.append(context)
            assert len(calls) == 1
            assert kwargs["output_model"] is Action
            return (
                Action(
                    kind="print_setup",
                    query=operation,
                    part_id="base",
                    revision_id=RID,
                    explanation="Inspect the saved print geometry within this author's budget",
                ),
                LLMUsage(self.model, 0, 0),
            )

        generate_structured_with_image = generate_structured

    service.client_factory = lambda **kwargs: ScriptedAuthor()
    service.store.enqueue(sid, "owner", {"max_actions": 3, "max_seconds": 120})
    job = service.store.claim()
    assert job is not None
    job["author_deadline"] = time.monotonic() + 30
    state = service._run_agent_job(job)

    assert state == "budget_exhausted"
    assert callback_observations == [False, True]
    assert publication_attempts == []
    assert len(calls) == 1
    saved = service.store.session_record(sid, "owner")
    assert not saved.get("print_setup_records")
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original_files
