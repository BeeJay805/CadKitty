"""Explicit schema reads remain usable during review and instruction repair."""

import json
from copy import deepcopy

import pytest

from forge_mind.workspace.agent_memory import cache_schema, recent_context
from forge_mind.workspace.authoring import authoring_schema
from forge_mind.workspace.document import digest
from forge_mind.workspace.instruction_context import customer_instruction_context
from forge_mind.workspace.native_review_context import compact_review_context
from forge_mind.workspace.native_stages import deduplicate_context


@pytest.mark.parametrize("projection", [compact_review_context, customer_instruction_context])
def test_explicit_assembly_schema_survives_next_review_turn_and_serialization(projection):
    exact = authoring_schema("assembly_edit")
    cache = cache_schema({}, "assembly_edit", exact)
    context = {
        "authoritative_schemas": cache,
        "schemas_already_read": ["assembly_edit"],
        "schema_catalog": [{"name": "assembly_edit", "sha256": "stale"}],
        "history": recent_context(
            [{"schema_read": "assembly_edit", "authoring_schema": exact}], cache
        ),
        "handoff_correction": {"scope": "CAD repair remains available", "applied": False},
    }
    before = deepcopy(context)
    result = projection(context)
    wire = json.loads(json.dumps(deduplicate_context(result)))
    assert wire["authoritative_schemas"]["assembly_edit"] == exact
    assert result["history"] == [{"schema_read": "assembly_edit"}]
    assert result["schema_catalog"] == [
        {
            "name": "assembly_edit",
            "query": "assembly_edit",
            "sha256": digest(exact),
            "retrieve": "schema query=assembly_edit",
        }
    ]
    assert result["handoff_correction"] == context["handoff_correction"]
    assert context == before


@pytest.mark.parametrize("projection", [compact_review_context, customer_instruction_context])
@pytest.mark.parametrize("channel", ["history", "persistent_tool_observations"])
def test_only_same_query_and_exact_schema_duplicates_are_removed(projection, channel):
    cached = {"type": "object", "additionalProperties": True}
    rows = [
        {"schema_read": "edit", "authoring_schema": deepcopy(cached), "tool_error": "Keep failure"},
        {"schema_read": "edit", "authoring_schema": {"type": "object", "additionalProperties": 1}},
        {"schema_read": "evicted", "authoring_schema": deepcopy(cached)},
        {"schema_read": "edit", "authoring_schema": {"older": "complete syntax"}},
        {"authoring_schema": deepcopy(cached)},
    ]
    result = projection({"authoritative_schemas": {"edit": cached}, channel: rows})
    assert result[channel] == [{"schema_read": "edit", "tool_error": "Keep failure"}, *rows[1:]]
    assert rows[0]["authoring_schema"] == cached


@pytest.mark.parametrize("projection", [compact_review_context, customer_instruction_context])
def test_uncached_oversized_explicit_schema_remains_complete(projection):
    schema = {"source": "exact explicit syntax " * 6000}
    cache = cache_schema({}, "large", schema)
    assert cache == {}
    rows = [{"schema_read": "large", "authoring_schema": schema}]
    result = projection({"authoritative_schemas": cache, "history": rows})
    assert result["history"] == rows


@pytest.mark.parametrize("projection", [compact_review_context, customer_instruction_context])
def test_projection_honors_effective_bounded_cache_without_resurrecting_evicted_schema(projection):
    cache = {}
    for index in range(8):
        cache = cache_schema(cache, str(index), {"definition": str(index) * 40000})
    before = deepcopy(cache)
    result = projection({"authoritative_schemas": cache})
    assert result["authoritative_schemas"] == before
    assert len(result["authoritative_schemas"]) <= 6
    assert len(json.dumps(result["authoritative_schemas"])) <= 160000
    assert {row["name"] for row in result["schema_catalog"]} == set(before)
