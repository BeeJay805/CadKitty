"""Automatic handoff duplicates can shrink while failed facts remain complete."""

import copy
import hashlib
import json

import pytest

from forge_mind.workspace.native_stages import deduplicate_context


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def expand(context):
    def walk(value, chain=()):
        if isinstance(value, dict) and set(value) == {"context_reference"}:
            reference = value["context_reference"]
            pointer = reference["json_pointer"]
            assert pointer not in chain, "Context reference cycle"
            target = context
            for token in pointer.lstrip("/").split("/"):
                token = token.replace("~1", "/").replace("~0", "~")
                target = target[int(token)] if isinstance(target, list) else target[token]
            assert hashlib.sha256(encoded(target).encode()).hexdigest() == reference["sha256"]
            return walk(target, (*chain, pointer))
        if isinstance(value, dict):
            return {key: walk(item, chain) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item, chain) for item in value]
        return value

    return walk(context)


def handoff_context():
    catalog = {
        "steps": [{"fact_id": f"width_{index}", "text": "Exact width ☃"} for index in range(80)]
    }
    binding = {"facts": [{"id": f"width_{index}", "value": index} for index in range(80)]}
    schema = {"description": "Exact allowed instruction syntax " * 100}
    guidance = {
        "basis_hash": "current-basis",
        "basis": {"revision_id": "current-revision", "artifact_hashes": {"part.step": "current"}},
        "instruction_catalog": catalog,
        "instruction_binding": binding,
    }
    historical = {
        **copy.deepcopy(guidance),
        "basis_hash": "historical-basis",
        "basis": {"revision_id": "historical-revision", "artifact_hashes": {"part.step": "old"}},
        "facts": {"status": "needs_review", "findings": [{"status": "fail", "measured": -0.2}]},
    }
    return {
        "authoritative_schemas": {"guidance": schema},
        "delivery_guidance": guidance,
        "history": [
            {
                "origin": "host",
                "job_id": "owned-job",
                "revision_id": "historical-revision",
                "handoff_preparation": {
                    "status": "prepared_for_instruction_review",
                    "basis_hash": "historical-preparation-basis",
                    "delivery_guidance": historical,
                    "guidance_schema": copy.deepcopy(schema),
                    "requirements_inspection": {"status": "fail", "errors": ["Exact collision"]},
                },
            }
        ],
    }


@pytest.mark.parametrize("channel", ["history", "persistent_tool_observations"])
def test_failed_automatic_handoff_keeps_exact_failures_and_references_safe_duplicates(channel):
    context = handoff_context()
    context[channel] = context.pop("history")
    original = copy.deepcopy(context)
    result = deduplicate_context(context)
    preparation = result[channel][0]["handoff_preparation"]
    before = original[channel][0]["handoff_preparation"]
    for field in ("basis", "basis_hash", "facts"):
        assert preparation["delivery_guidance"][field] == before["delivery_guidance"][field]
    assert preparation["requirements_inspection"] == before["requirements_inspection"]
    assert preparation["basis_hash"] == "historical-preparation-basis"
    for field in ("instruction_catalog", "instruction_binding"):
        assert "context_reference" in preparation["delivery_guidance"][field]
    assert "context_reference" in preparation["guidance_schema"]
    assert expand(result) == expand(original)
    assert context == original
    assert len(json.dumps(result)) < len(json.dumps(original)) * 0.7


@pytest.mark.parametrize("field", ["instruction_catalog", "instruction_binding"])
def test_failure_bearing_instruction_sibling_remains_exact(field):
    context = handoff_context()
    original = context["history"][0]["handoff_preparation"]["delivery_guidance"]
    original[field] = {
        "status": "fail",
        "errors": ["Keep this exact failed field"],
        **original[field],
    }
    result = deduplicate_context(context)
    observed = result["history"][0]["handoff_preparation"]["delivery_guidance"]
    assert observed[field] == original[field]
    assert expand(result) == expand(context)


@pytest.mark.parametrize("origin", [None, "model", "user"])
def test_explicit_or_unattributed_handoff_row_is_not_partially_projected(origin):
    context = handoff_context()
    row = context["history"][0]
    row["origin"] = origin
    result = deduplicate_context(context)
    assert result["history"] == context["history"]


@pytest.mark.parametrize(
    "name", ["source_text", "read_design", "part", "tool_error", "failed_submission"]
)
def test_explicit_read_in_host_row_stays_exact(name):
    context = handoff_context()
    context["history"][0][name] = {"text": "Exact explicit evidence " * 100}
    result = deduplicate_context(context)
    assert result["history"] == context["history"]
