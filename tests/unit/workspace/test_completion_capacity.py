from __future__ import annotations

import copy
import json

import pytest

from forge_mind.workspace.capacity import completion_policy
from forge_mind.workspace.completion import validate_plan
from forge_mind.workspace.completion_models import Subsystem
from forge_mind.workspace.processing import resource_policy

CONTRACT = {"requirements": [{"id": "fit"}]}
REQUEST = {
    "tool": "tolerance_stack",
    "units": "mm",
    "terms": [{"low": 1, "high": 2}],
    "allowable": {"low": 0, "high": 3},
}
POLICIES = (
    "COMPLETION_SUBSYSTEM_LIMIT",
    "COMPLETION_CHECK_LIMIT",
    "COMPLETION_PLAN_BYTES",
)


@pytest.fixture(autouse=True)
def isolated_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in POLICIES:
        monkeypatch.delenv("FORGE_WORKSPACE_" + name, raising=False)


def plan(count: int, checks: int = 1) -> dict:
    return {
        "subsystems": [
            {
                "id": f"s_{i}",
                "name": f"Subsystem {i}",
                "requirement_ids": ["fit"],
                "dependencies": [f"s_{i - 1}"] if i else [],
                "checks": [
                    {"id": f"c_{i}_{j}", "request": copy.deepcopy(REQUEST)} for j in range(checks)
                ],
            }
            for i in range(count)
        ]
    }


def test_large_reverse_chain_and_json_exceed_old_ceilings() -> None:
    value = plan(2048)
    value["subsystems"].reverse()
    assert len(json.dumps(value)) > 96000
    validate_plan(value, CONTRACT)
    assert all(item["value"] is None for item in completion_policy().values())


@pytest.mark.parametrize(
    ("name", "count", "checks", "limit"),
    [
        ("COMPLETION_SUBSYSTEM_LIMIT", 33, 0, 33),
        ("COMPLETION_CHECK_LIMIT", 1, 65, 65),
    ],
)
def test_explicit_count_boundary(
    monkeypatch: pytest.MonkeyPatch, name: str, count: int, checks: int, limit: int
) -> None:
    value = plan(count, checks)
    monkeypatch.setenv("FORGE_WORKSPACE_" + name, str(limit))
    validate_plan(value, CONTRACT)
    monkeypatch.setenv("FORGE_WORKSPACE_" + name, str(limit - 1))
    with pytest.raises(ValueError, match="Operator resource policy exceeded"):
        validate_plan(value, CONTRACT)


def test_utf8_byte_boundary_and_unchanged_input(monkeypatch: pytest.MonkeyPatch) -> None:
    value = plan(1)
    value["subsystems"][0]["name"] = "齿轮 assembly"
    original = copy.deepcopy(value)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert len(encoded) > len(encoded.decode("utf-8"))
    monkeypatch.setenv("FORGE_WORKSPACE_COMPLETION_PLAN_BYTES", str(len(encoded)))
    validate_plan(value, CONTRACT)
    monkeypatch.setenv("FORGE_WORKSPACE_COMPLETION_PLAN_BYTES", str(len(encoded) - 1))
    with pytest.raises(ValueError, match="completion plan JSON bytes"):
        validate_plan(value, CONTRACT)
    assert value == original


def test_check_policy_precedes_nested_model_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("Nested models must not be constructed after admission fails")

    monkeypatch.setenv("FORGE_WORKSPACE_COMPLETION_CHECK_LIMIT", "1")
    monkeypatch.setattr(Subsystem, "model_validate", unexpected)
    with pytest.raises(ValueError, match="prepared completion checks requires 2, limit is 1"):
        validate_plan(plan(1, 2), CONTRACT)


@pytest.mark.parametrize(
    "value",
    ["not an object", {"subsystems": None}, {"subsystems": [1]}, {"subsystems": [{"checks": 1}]}],
)
def test_invalid_raw_shapes_fail_explicitly(value) -> None:
    with pytest.raises(ValueError, match="Completion"):
        validate_plan(value, CONTRACT)


@pytest.mark.parametrize("name", POLICIES)
@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "garbage"])
def test_invalid_operator_policy_is_explicit(
    monkeypatch: pytest.MonkeyPatch, name: str, raw: str
) -> None:
    monkeypatch.setenv("FORGE_WORKSPACE_" + name, raw)
    with pytest.raises(ValueError, match=name + " must be a positive integer or unset"):
        validate_plan(plan(1), CONTRACT)


@pytest.mark.parametrize("name", POLICIES)
def test_blank_policy_and_discovery(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv("FORGE_WORKSPACE_" + name, " ")
    validate_plan(plan(40), CONTRACT)
    item = resource_policy()["completion"][name.lower()]
    assert item == {
        "environment": "FORGE_WORKSPACE_" + name,
        "value": None,
        "unit": "bytes" if name.endswith("_BYTES") else "count",
    }
    monkeypatch.setenv("FORGE_WORKSPACE_" + name, "80")
    assert resource_policy()["completion"][name.lower()]["value"] == 80


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_unlimited_does_not_admit_nonfinite_data(number: float) -> None:
    value = plan(1)
    value["subsystems"][0]["checks"][0]["request"]["terms"][0]["low"] = number
    with pytest.raises(ValueError):
        validate_plan(value, CONTRACT)


@pytest.mark.parametrize("fault", ["back_edge", "disconnected", "duplicate", "unknown", "self"])
def test_large_graph_still_rejects_invalid_dependencies(fault: str) -> None:
    value = plan(256, 0)
    rows = value["subsystems"]
    if fault == "back_edge":
        rows[-2]["dependencies"].append("s_255")
    elif fault == "disconnected":
        rows[-2]["dependencies"] = ["s_255"]
    elif fault == "duplicate":
        rows[-1]["dependencies"].append("s_254")
    elif fault == "unknown":
        rows[-1]["dependencies"].append("missing")
    else:
        rows[-1]["dependencies"].append("s_255")
    with pytest.raises(ValueError):
        validate_plan(value, CONTRACT)


@pytest.mark.parametrize("removed", ["subsystem", "check", "requirement"])
def test_expanded_plan_cannot_drop_previous_work(removed: str) -> None:
    original = plan(65)
    changed = copy.deepcopy(original)
    if removed == "subsystem":
        changed["subsystems"].pop()
    elif removed == "check":
        changed["subsystems"][-1]["checks"] = []
    else:
        changed["subsystems"][-1]["requirement_ids"] = []
    with pytest.raises(ValueError, match="cannot be removed"):
        validate_plan(changed, CONTRACT, original)
