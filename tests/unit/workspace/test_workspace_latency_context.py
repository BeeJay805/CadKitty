"""Exact current-request references reduce repeated input without losing authority."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from forge_mind.workspace.native_authoring import native_bootstrap
from forge_mind.workspace.native_stages import deduplicate_context
from forge_mind.workspace.plan_patch import PlanPatch


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _expand(projected: dict[str, Any]) -> dict[str, Any]:
    def visit(value: Any, seen: tuple[str, ...] = ()) -> Any:
        if isinstance(value, dict):
            if set(value) == {"context_reference"}:
                reference = value["context_reference"]
                path = reference["json_pointer"]
                assert path not in seen, "Reference cycle"
                target: Any = projected
                for token in path.split("/")[1:]:
                    key = token.replace("~1", "/").replace("~0", "~")
                    target = target[int(key)] if isinstance(target, list) else target[key]
                assert not (isinstance(target, dict) and "context_reference" in target)
                assert hashlib.sha256(_encoded(target)).hexdigest() == reference["sha256"]
                return visit(target, (*seen, path))
            return {key: visit(item, seen) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, seen) for item in value]
        return value

    return visit(projected)


def _context() -> dict[str, Any]:
    facts = {"holes": [{"center_mm": [i, -i], "diameter_mm": 5.2} for i in range(80)]}
    facts["physical_limits"] = ["Unverified strength and physical fit"]
    contract = {"requirements": [{"id": "body", "critical": True, "quote": "exact source"}]}
    return {
        "frozen_contract": contract,
        "design_plan": {"unresolved": ["Unknown retained load"], "scope": "Authored intent"},
        "edit_basis": {"revision_id": "a" * 32, "input_version": 7},
        "selected": {
            "id": "a" * 32,
            "contract": copy.deepcopy(contract),
            "measurement": {"facts": facts, "findings": [{"status": "unverified"}]},
        },
        "delivery_guidance": {"facts": copy.deepcopy(facts), "basis_hash": "b" * 64},
        "history": [{"product_review": {"facts": copy.deepcopy(facts)}} for _ in range(6)],
        "persistent_tool_observations": [{"print_setup": {"observations": copy.deepcopy(facts)}}],
        "new_unknown_field": {"facts": copy.deepcopy(facts), "unknown": [False, 0, None]},
    }


def test_lossless_references_reduce_material_repetition_and_are_detached():
    context = _context()
    original = copy.deepcopy(context)
    projected = deduplicate_context(context)
    assert _expand(projected) == original
    assert context == original
    assert len(_encoded(projected)) < len(_encoded(context)) * 0.6
    assert projected["new_unknown_field"] == original["new_unknown_field"]
    assert projected["selected"]["measurement"] == original["selected"]["measurement"]
    assert projected["frozen_contract"] == original["frozen_contract"]
    projected["design_plan"]["unresolved"].append("changed result only")
    assert context == original


@pytest.mark.parametrize("name", ["part", "read_part", "read_design", "source_text", "source_page"])
def test_explicit_read_rows_remain_full_even_when_identical(name):
    context = _context()
    row = {name: copy.deepcopy(context["selected"]["measurement"]["facts"])}
    context["history"] = [row]
    context["persistent_tool_observations"] = [copy.deepcopy(row)]
    projected = deduplicate_context(context)
    assert projected["history"] == context["history"]
    assert projected["persistent_tool_observations"] == context["persistent_tool_observations"]


def test_explicit_guidance_read_and_rejected_source_remain_complete():
    context = _context()
    context["history"] = [
        {"guidance": copy.deepcopy(context["delivery_guidance"]), "explicit_guidance_read": True},
        {"tool_error": "Wrong owner", "action": {"document_json": "exact rejected source"}},
    ]
    context["failed_submission"] = copy.deepcopy(context["history"][1])
    context["failed_attempt"] = {"errors": ["Invalid solid"], "source": "complete source"}
    projected = deduplicate_context(context)
    for key in ("history", "failed_submission", "failed_attempt"):
        assert projected[key] == context[key]


@pytest.mark.parametrize(
    "status",
    ["fail", "failed", "invalid", "validation_failed", "checked_fail", "needs_review", "error"],
)
def test_failure_and_review_rows_remain_complete(status):
    context = _context()
    context["history"] = [{"result": {"status": status, **context["delivery_guidance"]}}]
    assert deduplicate_context(context)["history"] == context["history"]


def test_nested_errors_and_unknown_status_shape_are_not_lost():
    context = _context()
    context["history"] = [
        {"receipt": {"errors": ["Clearance failure"], "facts": context["delivery_guidance"]}},
        {"unknown": {"status": {"future": True}, "facts": context["delivery_guidance"]}},
    ]
    projected = deduplicate_context(context)
    assert projected["history"][0] == context["history"][0]
    assert _expand(projected) == context


def test_json_types_and_changed_targets_do_not_alias():
    value = {"target": 1, "notes": "retained exact target " * 80}
    context = {
        "frozen_contract": value,
        "history": [
            {"inspection": {**value, "target": True}},
            {"inspection": {**value, "target": 1.0}},
            {"inspection": {**value, "target": 2}},
        ],
    }
    projected = deduplicate_context(context)
    assert _encoded(_expand(projected)) == _encoded(context)
    assert projected["history"][0]["inspection"]["target"] is True
    assert type(projected["history"][1]["inspection"]["target"]) is float
    assert projected["history"][2]["inspection"]["target"] == 2


def test_pointer_escaping_and_previous_row_anchors_resolve_without_cycles():
    exact = {"description": "Unknown physical property " * 80}
    context = {
        "selected": {"measurement": {"facts": {"a/b~c": exact}}},
        "history": [{"observation": exact}, {"novel": [exact]}, {"novel": [exact]}],
    }
    projected = deduplicate_context(context)
    assert _expand(projected) == context
    pointer = projected["history"][0]["observation"]["context_reference"]["json_pointer"]
    assert pointer.endswith("/a~1b~0c")


def test_small_duplicate_values_are_not_expanded_into_long_references():
    context = {"edit_basis": {"revision_id": "a" * 32}, "history": [{"revision_id": "a" * 32}]}
    assert deduplicate_context(context) == context


def test_bootstrap_construction_subset_uses_canonical_models_before_first_cad():
    bootstrap = native_bootstrap()
    subset = bootstrap["construction_patch_schema"]
    canonical = PlanPatch.model_json_schema()
    assert set(subset["properties"]) == {
        "basis",
        "construction_assignments",
        "verification_bindings",
    }
    assert subset["required"] == ["basis"]
    for name, value in subset["properties"].items():
        assert value == canonical["properties"][name]
    for name, value in subset["$defs"].items():
        assert value == canonical["$defs"][name]
    assert (
        "retained" in bootstrap["discovery"]["construction_assignments"]
        or "retain" in bootstrap["discovery"]["construction_assignments"]
    )
    assert "construction_transaction" in bootstrap["discovery"]
