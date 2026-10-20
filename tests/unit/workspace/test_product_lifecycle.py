from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.workspace.agent_memory import remember
from forge_mind.workspace.document import digest
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.input_application import FIELD
from forge_mind.workspace.measure import ground_contract
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.product_sources import _append, read_sources
from forge_mind.workspace.store import Store, WorkspaceError


def checkpoint(tmp_path: Path) -> tuple[Store, str]:
    store = Store(tmp_path)
    sid = store.create("owner", "Original job", Contract().model_dump())["id"]
    jid = store.enqueue(sid, "owner", {"preview_only": True, "instruction": "Preserved work"})
    job = store.claim()
    assert job and job["id"] == jid
    # Candidate history is consumed after preview admission, independently of its baseline.
    store.apply_input_contract(sid, "owner", jid, job["input_application_id"], Contract())
    store.update(
        sid,
        selected_revision="a" * 32,
        tool_memory=[{"catalog": "screw"}],
    )
    store.finish(jid, sid, "budget_exhausted")
    store.complete_preview(sid, jid, None, {"eligible": True})
    return store, sid


def test_budget_candidate_resumes_without_accepting_or_losing_intent(tmp_path: Path) -> None:
    store, sid = checkpoint(tmp_path)
    state = store.get(sid, "owner")
    assert state["selected_revision"] is None and state["proposal"] is None
    assert state["resume_candidate"]["revision_id"] == "a" * 32
    assert state["instructions"] == []
    assert state[FIELD]["applied"]["source_count"] == 0
    assert state[FIELD]["applied"]["source_hash"] == digest("Original job")
    candidate_application = state["resume_candidate"][FIELD]
    assert candidate_application["pending"] is None
    assert candidate_application["applied"]["source_count"] == 1
    assert candidate_application["applied"]["source_hash"] == digest(
        "Original job\nUser revision instruction: Preserved work"
    )
    assert candidate_application["applied"]["contract_hash"] == Contract().fingerprint()
    store.enqueue(sid, "owner", {"resume": True, "preview_only": True})
    job = store.claim()
    assert job and job["revision_id"] == "a" * 32
    assert job["resume_instructions"] == ["Preserved work"]
    assert job["resume_contract_application"] == candidate_application
    assert store.get(sid, "owner")["tool_memory"] == [{"catalog": "screw"}]


@pytest.mark.parametrize("change", ["input", "source", "baseline"])
def test_resume_refuses_changed_baseline(tmp_path: Path, change: str) -> None:
    store, sid = checkpoint(tmp_path)
    if change == "input":
        store.save_engineering(sid, "owner", 0, EngineeringInputs().model_dump())
    elif change == "baseline":
        store.update(sid, accepted_revision="b" * 32)
    else:
        _append(store, sid, "owner", {"kind": "artifact", "data": {"id": "fixture"}})
    with pytest.raises(WorkspaceError, match="changed"):
        store.enqueue(sid, "owner", {"resume": True, "preview_only": True})


def test_sources_changed_after_proposal_prevent_acceptance(tmp_path: Path) -> None:
    store = Store(tmp_path)
    sid = store.create("owner", "Job", None)["id"]
    fingerprint = read_sources(store, sid, "owner")["fingerprint"]
    store.update(
        sid,
        proposal={
            "revision_id": "a" * 32,
            "based_on": None,
            "input_version": 0,
            "eligible": True,
            "source_fingerprint": fingerprint,
        },
    )
    _append(store, sid, "owner", {"kind": "artifact", "data": {"id": "fixture"}})
    with pytest.raises(WorkspaceError, match="Sources changed"):
        store.accept_proposal(sid, "owner", "a" * 32, None, 0)
    with pytest.raises(WorkspaceError, match="not found"):
        store.enqueue(sid, "intruder", {"resume": True})


def test_tool_memory_retains_failed_search_but_bounds_payload() -> None:
    rows = remember(
        [
            {"catalog": {"query": "fan", "results": []}},
            {"measurement": "x" * 90000},
            {"research": "x" * 90000},
        ]
    )
    assert rows[0]["catalog"]["results"] == []
    assert len(str(rows)) < 64000


def test_within_bound_does_not_confuse_nominal_tolerance() -> None:
    for quote, expected in [
        ("Keep the X extent within 320 mm.", "extent_x"),
        ("Keep the X extent within 1 mm of 320 mm.", "unmeasured"),
    ]:
        req = Requirement(
            id="extent",
            text=quote,
            source_quote=quote,
            metric="extent_x",
            comparator="le",
            value=320,
        )
        assert (
            ground_contract(Contract(requirements=[req]), quote).requirements[0].metric == expected
        )


@pytest.mark.parametrize(
    ("elapsed", "cancel", "expected"),
    [(61, False, "budget_exhausted"), (10, False, "error"), (61, True, "cancelled")],
)
def test_model_failure_at_deadline_preserves_error_and_checkpoint(
    tmp_path: Path, monkeypatch, elapsed: int, cancel: bool, expected: str
) -> None:
    from forge_mind.llm.client import LLMError
    from forge_mind.workspace import agent
    from forge_mind.workspace.executor import Executor

    clock = [0.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock[0])
    store = Store(tmp_path)
    sid = store.create("owner", "Build a bracket", Contract().model_dump())["id"]
    failure = "Codex call cancelled or exceeded its wall-clock limit"

    class Client:
        def generate_structured(self, **kwargs):
            clock[0] = elapsed
            if cancel:
                service.stop.set()
            raise LLMError(failure)

    service = agent.WorkspaceService(
        store, Executor("trusted_local"), client_factory=lambda **kwargs: Client()
    )
    store.enqueue(sid, "owner", {"preview_only": True, "max_seconds": 60, "max_actions": 1})
    job = store.claim()
    assert job is not None
    service.run_job(job)
    state = store.get(sid, "owner")
    assert state["state"] == expected
    assert state["proposal"] is None
    assert state["resume_candidate"] is not None
    assert any(failure in e["message"] for e in state["events"] if e["kind"] == "error")
