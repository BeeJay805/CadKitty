"""Automatic author context is bounded without truncating the saved design."""

import copy
import hashlib
import json

import pytest
from tests.unit.workspace.test_assembly_generation import simple

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument, Occurrence, digest
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.inspection import author_revision_context, query_document
from forge_mind.workspace.models import Action, Contract, Measurement
from forge_mind.workspace.store import Store


def flat_document(count=1, source=None):
    features = [
        {"id": f"body{i}", "name": f"Body {i}", "kind": "box", "params": {"x": 10, "y": 2, "z": 3}}
        for i in range(count)
    ]
    if source is not None:
        features.append({"id": "program", "name": "Source", "kind": "python", "code": source})
    return DesignDocument.model_validate(
        {
            "variables": {f"v{i}": float(i) for i in range(count)},
            "features": features,
            "outputs": [features[-1]["id"]],
        }
    )


def revision(document):
    return {
        "id": "a" * 32,
        "document": document.model_dump(),
        "parameters": document.variables,
        "code": "redundant generated program",
        "diff": "redundant generated diff",
        "measurement": {
            "geometry_valid": True,
            "findings": [{"status": "fail", "id": "important"}],
            "facts": {"feature_history": [{"id": f"row{i}"} for i in range(10000)]},
        },
    }


def test_small_flat_context_remains_complete_and_omits_redundant_program():
    saved = revision(flat_document())
    selected = author_revision_context(saved)
    assert selected["document_is_summary"] is False
    assert selected["document"] == saved["document"]
    assert "code" not in selected and "diff" not in selected
    assert "code" in saved


def test_large_flat_context_is_explicit_summary_with_exact_revision_pages():
    document = flat_document(1000)
    saved = revision(document)
    before = copy.deepcopy(saved)
    selected = author_revision_context(saved)
    assert len(json.dumps(selected)) < 16000
    assert selected["document_is_summary"] is True
    assert selected["document"]["counts"]["variables"] == 1000
    assert selected["document"]["counts"]["features"] == 1000
    assert "features" not in selected["document"]
    assert "parameters" not in selected
    assert selected["measurement"]["findings"] == saved["measurement"]["findings"]
    assert selected["measurement"]["facts"]["feature_history"]["is_summary"] is True
    assert saved == before
    values, cursor = {}, None
    while True:
        page = query_document(document, saved["id"], "variables", limit=137, cursor=cursor)
        assert page["document_hash"] == selected["document_hash"]
        values.update({r["name"]: r["value"] for r in page["items"]})
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert values == document.variables


def test_retained_source_is_hashed_and_requires_explicit_feature_lookup():
    source = "# retained editable source\n" * 5000 + "result=b.Box(1,2,3)"
    document = flat_document(source=source)
    selected = author_revision_context(revision(document))
    assert len(json.dumps(selected)) < 16000
    assert source not in json.dumps(selected)
    feature = next(f for f in selected["document"]["preview"]["features"] if f["id"] == "program")
    assert feature["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert feature["source_characters"] == len(source)
    exact = query_document(document, "a" * 32, "features", feature_id="program", limit=1)
    assert exact["items"][0]["code"] == source
    with pytest.raises(ValueError, match="Feature ID"):
        query_document(document, "a" * 32, "variables", feature_id="program")


def test_assembly_context_keeps_small_index_and_bounds_large_occurrences():
    document = simple()
    selected = author_revision_context(revision(document))
    assert selected["document_is_summary"] is True
    assert selected["document"]["assembly"]["parts"][0]["id"] == "block"
    assert "features" not in selected["document"]["assembly"]["parts"][0]
    document.assembly.roots.extend(
        Occurrence(id=f"root{i}", definition="block") for i in range(1000)
    )
    selected = author_revision_context(revision(document))
    assert len(json.dumps(selected)) < 16000
    assert selected["document"]["counts"]["roots"] == 1002
    assert selected["document"]["preview"]["features"][0]["part_id"] == "block"
    roots = query_document(document, "a" * 32, "roots", limit=1002)
    assert len(roots["items"]) == 1002
    assert query_document(document, "a" * 32, "subassemblies")["total"] == 1
    assert query_document(document, "a" * 32, "metadata")["items"]
    feature = query_document(document, "a" * 32, "features", scope="block", feature_id="body")
    assert feature["items"][0] == document.assembly.parts[0].features[0].model_dump()


def test_feature_cursor_binds_lookup_and_preserves_full_source():
    document = flat_document(3)
    first = query_document(document, "a" * 32, "features", limit=1)
    with pytest.raises(ValueError, match="Cursor"):
        query_document(
            document, "a" * 32, "features", limit=1, cursor=first["next_cursor"], feature_id="body2"
        )


def test_author_restore_keeps_chosen_flat_candidate_without_approving_it(tmp_path, monkeypatch):
    from forge_mind.workspace import agent, completion

    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Choose the measured design candidate", contract.model_dump())["id"]
    document = flat_document()
    baseline, chosen = "a" * 32, "b" * 32
    for rid, satisfied in [(baseline, True), (chosen, False)]:
        measurement = Measurement(
            revision_id=rid,
            contract_hash=contract.fingerprint(),
            engine_fingerprint="frozen",
            artifact_hashes={},
            geometry_valid=True,
            requirements_satisfied=satisfied,
        )
        store.add_revision(
            sid,
            {
                "id": rid,
                "code": "unused",
                "document": document.model_dump(),
                "contract": contract.model_dump(),
                "measurement": measurement.model_dump(),
            },
        )
    store.update(sid, selected_revision=baseline, best_revision=baseline)
    actions = iter(
        [
            Action(kind="restore", explanation="Choose evaluated candidate", revision_id=chosen),
            Action(
                kind="finish", explanation="Candidate selected; physical validation remains open"
            ),
        ]
    )
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            contexts.append(json.loads(kwargs["user"]))
            return next(actions), LLMUsage("test", 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "_load_revision", lambda sid, rid, **kw: store.revision(sid, rid))
    monkeypatch.setattr(agent, "engine_fingerprint", lambda: "frozen")
    monkeypatch.setattr(completion, "completion_context", lambda *a: {"premature_finish": False})
    store.enqueue(sid, "owner", {"max_actions": 2})
    state = service._run_agent_job(store.claim())
    assert state == "needs_review"
    assert store.get(sid, "owner")["selected_revision"] == chosen
    assert store.get(sid, "owner")["best_revision"] == baseline
    assert contexts[1]["selected"]["id"] == chosen
    assert contexts[1]["selected"]["document_is_summary"] is False
    assert "code" not in contexts[1]["selected"]
    assert digest(store.revision(sid, chosen)["document"]) == document.fingerprint()
