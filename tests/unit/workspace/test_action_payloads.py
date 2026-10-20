"""Recoverable payload rejection and revision-bound reads without native execution."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forge_mind.workspace.action_payloads import (
    InputValidationError,
    ReadDesignArguments,
    ReadDesignQuery,
    decode_json_object,
)
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.design_study_agent import execute_study_action
from forge_mind.workspace.design_study_api import design_study_router, study_schema
from forge_mind.workspace.design_study_mcp import register_design_study_tools
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.inspection import query_document
from forge_mind.workspace.lab_agent import run_lab_action
from forge_mind.workspace.models import Action

RID = "a" * 32


def document() -> DesignDocument:
    return DesignDocument.model_validate(
        {
            "variables": {"width": 3, "height": 4},
            "features": [
                {
                    "id": "first",
                    "name": "First",
                    "kind": "box",
                    "params": {"x": 1, "y": 2, "z": 3},
                },
                {
                    "id": "second",
                    "name": "Second",
                    "kind": "box",
                    "params": {"x": 2, "y": 3, "z": 4},
                },
            ],
            "outputs": ["second"],
        }
    )


def reader_service(doc: DesignDocument, calls: list[str]) -> WorkspaceService:
    def load(_sid: str, _rid: str) -> dict[str, Any]:
        calls.append("load")
        return {"document": doc.model_dump()}

    return cast(
        WorkspaceService,
        SimpleNamespace(
            _load_revision=load,
            store=SimpleNamespace(session_record=lambda *_: calls.append("session")),
        ),
    )


def dispatch(service: WorkspaceService, payload: str, query: str = "features") -> dict[str, Any]:
    return execute_study_action(
        service,
        "project",
        "owner",
        {},
        Action(
            kind="read_design",
            explanation="Read current saved features",
            revision_id=RID,
            query=query,
            document_json=payload,
        ),
        RID,
        lambda: False,
    )


@pytest.mark.parametrize("payload", ["[]", "null", '"text"', "0", "true"])
def test_json_root_must_be_an_object(payload):
    with pytest.raises(InputValidationError) as rejected:
        decode_json_object(payload)
    assert rejected.value.error_result() == {
        "code": "invalid_json_object",
        "field": "document_json",
        "message": "Supply a JSON object",
        "retryable": True,
        "applied": False,
    }


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}'])
def test_duplicate_json_keys_are_rejected_at_any_depth(payload):
    with pytest.raises(InputValidationError) as rejected:
        decode_json_object(payload, "action")
    assert rejected.value.code == "duplicate_json_key"
    assert rejected.value.field == "action.x"


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_json_numbers_are_rejected_at_any_depth(number):
    with pytest.raises(InputValidationError) as rejected:
        decode_json_object('{"nested":[' + number + "]}")
    assert rejected.value.code == "nonfinite_json_number"


def test_json_decoder_preserves_finite_values_and_reports_malformed_input():
    expected = {"nested": [None, True, 1, -2.5, {"word": "value"}]}
    assert decode_json_object(json.dumps(expected)) == expected
    with pytest.raises(InputValidationError, match="Malformed JSON at line"):
        decode_json_object("{")


@pytest.mark.parametrize("payload", [None, 0, {}, b"{}"])
def test_decoder_rejects_nonstring_json_payloads(payload):
    with pytest.raises(InputValidationError) as rejected:
        decode_json_object(payload)
    assert rejected.value.code == "invalid_json"


@pytest.mark.parametrize(
    "arguments",
    [
        {"cursor": 0},
        {"cursor": True},
        {"cursor": []},
        {"cursor": {}},
        {"limit": False},
        {"limit": 0},
        {"limit": -1},
        {"limit": 1.5},
        {"limit": "2"},
        {"scope": 4},
        {"feature_id": []},
    ],
)
def test_direct_and_author_readers_reject_the_same_bad_nested_arguments(arguments):
    doc = document()
    before = doc.model_dump()
    calls: list[str] = []
    service = reader_service(doc, calls)
    with pytest.raises(InputValidationError) as direct:
        query_document(doc, RID, "features", **arguments)
    with pytest.raises(InputValidationError) as author:
        dispatch(service, json.dumps(arguments))
    assert direct.value.error_result() == author.value.error_result()
    assert author.value.error_result()["applied"] is False
    assert calls == []
    assert doc.model_dump() == before


def test_unknown_reader_keys_are_rejected_before_revision_access():
    calls: list[str] = []
    with pytest.raises(InputValidationError) as rejected:
        dispatch(reader_service(document(), calls), '{"offset":0}')
    assert rejected.value.field == "read_design.offset"
    assert calls == []


@pytest.mark.parametrize("query", ["unknown", "", "Features"])
def test_unknown_direct_collections_are_rejected(query):
    with pytest.raises(InputValidationError) as rejected:
        query_document(document(), RID, query)
    assert rejected.value.field == "read_design.collection"


def test_invalid_author_collection_is_rejected_before_revision_access():
    calls: list[str] = []
    with pytest.raises(InputValidationError) as rejected:
        dispatch(reader_service(document(), calls), "{}", "unknown")
    assert rejected.value.field == "read_design.collection"
    assert calls == []


@pytest.mark.parametrize("payload", ["null", "[]", "true", '{"limit":1,"limit":2}'])
def test_dispatcher_rejects_ambiguous_or_nonobject_root_without_access(payload):
    calls: list[str] = []
    with pytest.raises(InputValidationError):
        dispatch(reader_service(document(), calls), payload)
    assert calls == []


@pytest.mark.parametrize("payload", ['{"retry_failed":"yes"}', '{"retry_failed":1}', '{"x":1}'])
def test_study_run_arguments_are_validated_before_study_or_job_access(payload):
    service = cast(WorkspaceService, SimpleNamespace())
    action = Action(
        kind="design_study",
        explanation="Retry a saved study",
        query="run:" + "b" * 32,
        document_json=payload,
    )
    with pytest.raises(InputValidationError):
        execute_study_action(service, "project", "owner", {}, action, RID, lambda: False)


def test_author_can_retry_invalid_cursor_and_follow_real_continuation():
    doc = document()
    calls: list[str] = []
    service = reader_service(doc, calls)
    before = doc.model_dump()
    with pytest.raises(InputValidationError, match="omit cursor for the first page"):
        dispatch(service, '{"cursor":0}')
    first = dispatch(service, '{"cursor":null,"limit":1}')
    assert first["items"][0]["id"] == "first"
    second = dispatch(service, json.dumps({"cursor": first["next_cursor"], "limit": 1}))
    assert second["items"][0]["id"] == "second"
    assert second["next_cursor"] is None
    assert calls == ["load", "load"]
    assert doc.model_dump() == before


@pytest.mark.parametrize("cursor", ["!", "0", "bm90anNvbg==", "W10="])
def test_malformed_cursors_are_recoverable(cursor):
    with pytest.raises(InputValidationError) as rejected:
        query_document(document(), RID, "variables", cursor=cursor)
    assert rejected.value.code == "invalid_cursor"


def test_cursor_binding_rejects_other_revision_query_scope_and_changed_document():
    doc = document()
    cursor = query_document(doc, RID, "variables", limit=1)["next_cursor"]
    for revision_id, collection, scope in [
        ("b" * 32, "variables", None),
        (RID, "relations", None),
        (RID, "variables", "other"),
    ]:
        with pytest.raises(InputValidationError) as rejected:
            query_document(doc, revision_id, collection, scope=scope, cursor=cursor)
        assert rejected.value.code == "invalid_cursor"
    changed = doc.model_copy(update={"variables": {"width": 4, "height": 4}})
    with pytest.raises(InputValidationError):
        query_document(changed, RID, "variables", cursor=cursor)
    assert query_document(doc, RID, "variables", cursor=cursor)["items"] == [
        {"name": "height", "value": 4.0, "editable": True}
    ]


def test_reader_preserves_large_direct_page_sizes_and_exact_feature_queries():
    assert query_document(document(), RID, "features", limit=1002)["returned"] == 2
    page = query_document(document(), RID, "features", feature_id="second", limit=1)
    assert page["items"][0]["id"] == "second"


def test_discovery_exposes_the_runtime_models(monkeypatch):
    monkeypatch.setattr("forge_mind.workspace.design_study_api.evaluator_discovery", lambda: [])
    schema = study_schema()
    assert schema["read_design"] == ReadDesignArguments.model_json_schema()
    assert schema["read_design_query"] == ReadDesignQuery.model_json_schema()
    assert schema["read_design"]["additionalProperties"] is False


def test_http_reader_uses_the_closed_shared_arguments_and_keeps_transport_budget():
    calls: list[str] = []
    app = FastAPI()
    app.include_router(design_study_router(reader_service(document(), calls), lambda: "owner"))
    client = TestClient(app)
    url = f"/workspace/sessions/project/design/{RID}/page"
    rejected = client.get(url, params={"collection": "features", "offset": 0})
    assert rejected.status_code == 409
    assert "read_design.offset" in rejected.json()["detail"]
    assert calls == []
    assert client.get(url, params={"collection": "features", "limit": 1002}).status_code == 422
    assert calls == []
    first = client.get(url, params={"collection": "features", "limit": 1})
    assert first.status_code == 200
    next_page = client.get(
        url, params={"collection": "features", "cursor": first.json()["next_cursor"], "limit": 1}
    )
    assert next_page.status_code == 200
    assert next_page.json()["items"][0]["id"] == "second"


def test_mcp_reader_validates_before_forwarding_and_preserves_flat_tool_fields():
    tools: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []

    def tool(**_kwargs):
        def register(function):
            tools[function.__name__] = function
            return function

        return register

    async def request(_method, _path, **kwargs):
        requests.append(kwargs)
        return {"items": []}

    register_design_study_tools(SimpleNamespace(tool=tool), SimpleNamespace(json=request))
    read = tools["read_design_page"]
    with pytest.raises(InputValidationError):
        asyncio.run(read("project", RID, "features", cursor=0))
    assert requests == []
    assert asyncio.run(read("project", RID, "features", limit=2)) == {"items": []}
    assert requests == [{"params": {"collection": "features", "limit": "2"}}]


@pytest.mark.parametrize(
    "body",
    [
        {"body_index": True},
        {"body_index": -1},
        {"body_index": "0"},
        {"part_id": []},
        {"feature_id": 1},
        {"unknown_selector": "body"},
    ],
)
def test_lab_explain_rejects_invalid_selectors_before_service_access(body, monkeypatch):
    monkeypatch.setattr(
        "forge_mind.workspace.lab_agent.explain_design",
        lambda *_args, **_kwargs: pytest.fail("Invalid selectors reached the explanation service"),
    )
    with pytest.raises(InputValidationError):
        run_lab_action(
            cast(WorkspaceService, SimpleNamespace()),
            "project",
            "owner",
            RID,
            "explain",
            json.dumps(body),
        )


def test_lab_explain_preserves_valid_combined_selectors(monkeypatch):
    def explain(_service, _sid, _owner, _rid, **selectors):
        return selectors

    monkeypatch.setattr("forge_mind.workspace.lab_agent.explain_design", explain)
    assert run_lab_action(
        cast(WorkspaceService, SimpleNamespace()),
        "project",
        "owner",
        RID,
        "explain",
        '{"body_index":0,"part_id":"member"}',
    ) == {"body_index": 0, "part_id": "member", "feature_id": None}


def test_unexpected_service_fault_is_not_reclassified_as_bad_input():
    def failed_load(_sid, _rid):
        raise AttributeError("Internal revision-reader defect")

    service = cast(WorkspaceService, SimpleNamespace(_load_revision=failed_load))
    with pytest.raises(AttributeError, match="Internal revision-reader defect"):
        dispatch(service, "{}")
