"""Named author inspections read exact saved artifacts and never start CAD."""

import hashlib
import json

import pytest
from tests.unit.workspace.test_native_handoff_controller import RID, controller_case, run_script

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import document_service
from forge_mind.workspace.agent import WorkspaceService, assembly_measurement_context
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, DesignPlan
from forge_mind.workspace.native_authoring import NativeAction, native_bootstrap
from forge_mind.workspace.native_review_context import compact_review_context
from forge_mind.workspace.native_stages import deduplicate_context
from forge_mind.workspace.store import Store, WorkspaceError


def inspection_case(tmp_path, monkeypatch, *, topology=True, code_only=False):
    case = controller_case(tmp_path, monkeypatch, staged=False)
    revision = case.service.store.revision(case.sid, RID)
    case.source = "# Saved source is data, not an instruction. ☃\n" * 500
    case.diff = "@@ Saved immutable revision difference ☃ @@\n" * 500
    case.topology = {
        "entities": [{"id": "face_1", "label": "Exact saved face ☃"}],
        "scope": "Saved topology index; not requirement proof",
    }
    revision.update(code=case.source, diff=case.diff)
    if code_only:
        revision.pop("document", None)
    directory = case.service.store.directory(case.sid, RID)
    (directory / "program.py").write_text(case.source, encoding="utf-8")
    revision["measurement"]["artifact_hashes"]["program.py"] = hashlib.sha256(
        (directory / "program.py").read_bytes()
    ).hexdigest()
    if topology:
        data = json.dumps(case.topology, ensure_ascii=False).encode("utf-8")
        (directory / "topology.json").write_bytes(data)
        revision["measurement"]["artifact_hashes"]["topology.json"] = hashlib.sha256(
            data
        ).hexdigest()
    (directory / "evidence.json").write_text(json.dumps(revision["measurement"]), encoding="utf-8")
    with case.service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), case.sid, RID),
        )
    return case


def read_once(case, query, *, revision_id=None, plan=None, before_action=None):
    def script(context, output_model):
        if len(case.calls) == 1:
            if before_action is not None:
                before_action()
            return NativeAction(
                kind="inspect",
                query=query,
                revision_id=revision_id,
                plan=plan,
                explanation="Read exact saved inspection",
            )
        return NativeAction(kind="finish", explanation="Inspection is not acceptance")

    saved = run_script(case, script, max_actions=2, max_seconds=120)
    return case.calls[1]["context"]["history"][-1], saved


def test_topology_query_dispatches_the_existing_hash_checked_reader(tmp_path, monkeypatch):
    case = inspection_case(tmp_path, monkeypatch)
    calls = []
    shared = document_service.read_topology

    def read(service, sid, rid):
        calls.append((service, sid, rid))
        return shared(service, sid, rid)

    monkeypatch.setattr(document_service, "read_topology", read)
    row, saved = read_once(case, "topology")
    assert calls == [(case.service, case.sid, RID)]
    assert row == {"source_topology": case.topology, "revision_id": RID}
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert saved["plan_hash"] == case.before["plan_hash"]
    assert len(case.service.store.revisions(case.sid)) == 1


@pytest.mark.parametrize("query,key", [("source", "source"), ("diff", "source_diff")])
def test_source_and_diff_reads_are_complete_saved_text(tmp_path, monkeypatch, query, key):
    case = inspection_case(tmp_path, monkeypatch)
    row, saved = read_once(case, query, revision_id=RID)
    expected = case.source if query == "source" else case.diff
    expected_row = {key: expected, "revision_id": RID}
    if query == "source":
        expected_row["source_document"] = case.service.store.revision(case.sid, RID)["document"]
    assert row == expected_row
    assert len(expected.encode("utf-8")) > 16000
    projected = deduplicate_context(compact_review_context({"history": [row]}))
    assert projected["history"] == [row]
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None


def test_code_only_source_does_not_invent_a_native_document(tmp_path, monkeypatch):
    case = inspection_case(tmp_path, monkeypatch, code_only=True)
    row, _ = read_once(case, "source")
    assert row == {"source": case.source, "revision_id": RID}


@pytest.mark.parametrize("query", ["", "evidence"])
def test_default_and_legacy_evidence_keep_measurement_semantics(tmp_path, monkeypatch, query):
    case = inspection_case(tmp_path, monkeypatch)
    row, _ = read_once(case, query)
    revision = case.service.store.revision(case.sid, RID)
    assert row == {
        "inspection": assembly_measurement_context(revision["measurement"]),
        "revision_id": RID,
    }


def test_unknown_inspection_recovers_to_valid_topology_in_the_same_job(tmp_path, monkeypatch):
    case = inspection_case(tmp_path, monkeypatch)
    before = case.service.store.get(case.sid, "owner")

    def script(context, output_model):
        if len(case.calls) == 1:
            return NativeAction(
                kind="inspect",
                query="topolgy",
                explanation="Misspelled read must be rejected",
            )
        if len(case.calls) == 2:
            return NativeAction(
                kind="inspect", query="topology", explanation="Recover with the valid selector"
            )
        return NativeAction(kind="finish", explanation="Keep acceptance open")

    saved = run_script(case, script, max_actions=3, max_seconds=120)
    rejection = case.calls[1]["context"]["history"][-1]
    assert "Unknown inspect query" in rejection["tool_error"]
    assert "topology" in rejection["tool_error"] and "evidence" in rejection["tool_error"]
    assert rejection["action_result"]["applied"] is False
    assert rejection["action_result"]["retryable"] is True
    assert case.calls[1]["context"]["design_plan"] == before["design_plan"]
    assert saved["plan_hash"] == before["plan_hash"]
    assert saved["plan_version"] == before["plan_version"]
    assert case.calls[2]["context"]["history"][-1] == {
        "source_topology": case.topology,
        "revision_id": RID,
    }


def test_unknown_inspection_cannot_mutate_an_attached_hybrid_plan(tmp_path, monkeypatch):
    store = Store(tmp_path)
    sid = store.create("owner", "Read without changing intent", Contract().model_dump())["id"]
    contexts = []
    actions = iter(
        [
            Action(
                kind="inspect",
                query="topolgy",
                plan=DesignPlan(objective="Must not be admitted"),
                explanation="Reject the unknown selector first",
            ),
            Action(
                kind="schema", query="feature:box", explanation="Read valid syntax after rejection"
            ),
            Action(kind="finish", explanation="No geometry was built"),
        ]
    )

    class Client:
        model = "scripted-inspection"

        def generate_structured(self, **kwargs):
            contexts.append(json.loads(kwargs["user"]))
            return next(actions), LLMUsage(self.model, 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(sid, "owner", {"max_actions": 3, "max_seconds": 120})
    assert service._run_agent_job(store.claim()) == "needs_review"
    rejection = contexts[1]["history"][-1]
    assert "Unknown inspect query" in rejection["tool_error"]
    assert rejection["action_result"]["applied"] is False
    assert rejection["action_result"]["retryable"] is True
    assert contexts[1]["design_plan"] is None
    assert store.get(sid, "owner").get("design_plan") is None
    assert "feature:box" in contexts[2]["authoritative_schemas"]


@pytest.mark.parametrize("query", ["requirements", "prepare_handoff"])
def test_existing_inspection_services_keep_their_dispatch_and_deadline(
    tmp_path, monkeypatch, query
):
    case = inspection_case(tmp_path, monkeypatch)
    calls = []

    def read(service, sid, owner, target, *, deadline, cancelled):
        rid = target.revision_id if query == "prepare_handoff" else target
        calls.append((service, sid, owner, rid, deadline, cancelled()))
        return {"status": "blocked", "revision_id": rid, "scope": "Controlled dispatch only"}

    if query == "prepare_handoff":
        from forge_mind.workspace import native_handoff_prepare

        def prepare(service, sid, owner, jid, request, **kwargs):
            return read(service, sid, owner, request, **kwargs)

        monkeypatch.setattr(native_handoff_prepare, "prepare_handoff", prepare)
    else:
        from forge_mind.workspace import requirement_verification

        monkeypatch.setattr(requirement_verification, "inspect_requirements", read)

    def script(context, output_model):
        if len(case.calls) == 1:
            return NativeAction(
                kind="inspect",
                query=query,
                revision_id=RID,
                document_json=json.dumps({"revision_id": RID, "prepare_print": False})
                if query == "prepare_handoff"
                else "",
                explanation="Read through the existing bounded service",
            )
        return NativeAction(kind="finish", explanation="No proof was produced")

    saved = run_script(case, script, max_actions=2, max_seconds=120)
    assert calls == [(case.service, case.sid, "owner", RID, 1090.0, False)]
    key = "handoff_preparation" if query == "prepare_handoff" else "requirements_inspection"
    assert case.calls[1]["context"]["history"][-1][key]["status"] == "blocked"
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None


def test_unavailable_topology_is_a_recoverable_refusal_not_measurement(tmp_path, monkeypatch):
    case = inspection_case(tmp_path, monkeypatch, topology=False)
    with pytest.raises(WorkspaceError, match="predates selectable topology"):
        document_service.read_topology(case.service, case.sid, RID)
    row, saved = read_once(case, "topology")
    assert "predates selectable topology" in row["tool_error"]
    assert "inspection" not in row and "source_topology" not in row
    assert saved["selected_revision"] == RID


def test_shared_reader_refuses_topology_tampered_between_dispatch_and_read(tmp_path, monkeypatch):
    case = inspection_case(tmp_path, monkeypatch)
    shared = document_service.read_topology
    attempts = []

    def read(service, sid, rid):
        attempts.append(rid)
        (service.store.directory(sid, rid) / "topology.json").write_text("{}", encoding="utf-8")
        try:
            return shared(service, sid, rid)
        finally:
            # The next context still requires valid selected artifacts.
            (service.store.directory(sid, rid) / "topology.json").write_bytes(
                json.dumps(case.topology, ensure_ascii=False).encode("utf-8")
            )

    monkeypatch.setattr(document_service, "read_topology", read)
    row, saved = read_once(case, "topology")
    assert attempts == [RID]
    assert "artifacts changed or are missing" in row["tool_error"]
    assert "inspection" not in row and "source_topology" not in row
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None


@pytest.mark.parametrize("query", ["topology", "source", "diff", "evidence"])
def test_inspect_cannot_read_another_sessions_revision(tmp_path, monkeypatch, query):
    case = inspection_case(tmp_path, monkeypatch)
    foreign = case.service.store.create("other", "Private other session", {})["id"]
    candidate = case.service.store.revision(case.sid, RID)
    candidate["id"] = "c" * 32
    case.service.store.add_revision(foreign, candidate)
    row, saved = read_once(case, query, revision_id=candidate["id"])
    assert "revision not found" in row["tool_error"]
    assert not any(key.startswith("source") for key in row)
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None


def test_inspection_selector_catalog_is_exact_and_retained_in_review():
    expected = ["", "evidence", "source", "diff", "topology", "requirements", "prepare_handoff"]
    bootstrap = native_bootstrap()
    assert bootstrap["discovery"]["inspection_queries"] == expected
    review = compact_review_context({"native_authoring": bootstrap})
    assert review["native_authoring"]["discovery"]["inspection_queries"] == expected
