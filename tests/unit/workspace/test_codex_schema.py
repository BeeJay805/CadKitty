from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from typing import Any

import pytest

from forge_mind.llm.client import LLMError
from forge_mind.llm.codex_client import CodexClient, _restore_maps, _strict_schema
from forge_mind.workspace.models import Action, Contract


@pytest.mark.parametrize("model", [Contract, Action])
def test_cli_schema_has_strict_objects_and_homogeneous_arrays(model: Any) -> None:
    def visit(node: Any) -> None:
        if isinstance(node, dict):
            assert "default" not in node
            assert "prefixItems" not in node
            assert not {"minLength", "maxLength", "minItems", "maxItems"} & node.keys()
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
                assert not isinstance(node.get(key), (int, float))
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            if node.get("type") == "array":
                assert "items" in node
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(_strict_schema(model.model_json_schema()))


def test_wire_length_guidance_does_not_weaken_local_validation() -> None:
    from pydantic import BaseModel, Field, ValidationError

    class Bounded(BaseModel):
        text: str = Field(min_length=1, max_length=4)
        points: tuple[float, float]

    original = Bounded.model_json_schema()
    wire = _strict_schema(original)
    assert '"maxLength": 4' in wire["properties"]["text"]["description"]
    assert '"maxItems": 2' in wire["properties"]["points"]["description"]
    for invalid in ({"text": "oversized", "points": [1, 2]}, {"text": "ok", "points": [1]}):
        with pytest.raises(ValidationError):
            Bounded.model_validate(_restore_maps(invalid, original, original))


def test_parameter_maps_roundtrip_through_strict_wire_schema() -> None:
    schema = Action.model_json_schema()
    output = {
        "kind": "parameters",
        "explanation": "Change width",
        "parameters": [{"key": "width", "value": 20.5}],
    }
    action = Action.model_validate(_restore_maps(output, schema, schema))
    assert action.parameters == {"width": 20.5}
    output["parameters"] = [{"key": "width", "value": 20.5}, {"key": "width", "value": 30.0}]
    with pytest.raises(ValueError, match="duplicate"):
        _restore_maps(output, schema, schema)


@pytest.mark.parametrize("unlimited", [False, True])
def test_cli_that_never_reads_input_cannot_block_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    unlimited: bool,
) -> None:
    monkeypatch.setattr(
        "forge_mind.llm.codex_client._command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    start = time.monotonic()
    with pytest.raises(LLMError, match="wall-clock"):
        CodexClient(
            timeout_s=None if unlimited else 0.3,
            cancelled=lambda: unlimited and time.monotonic() - start >= 0.3,
        ).generate_structured(
            system="", user="x" * 300_000, output_model=Action, model="gpt-6-astra", max_tokens=10
        )
    assert time.monotonic() - start < 5


def test_numeric_wire_guidance_preserves_named_fields_and_original_schema():
    from forge_mind.workspace.motion_rig import MotionRig

    original = MotionRig.model_json_schema()
    snapshot = deepcopy(original)
    wire = _strict_schema(original)
    driver = wire["$defs"]["MotionDriver"]
    assert {"minimum", "maximum"} <= driver["properties"].keys()
    assert {"minimum", "maximum"} <= set(driver["required"])
    assert driver["properties"]["minimum"]["type"] == "number"
    assert original == snapshot
    # Keyword-looking user fields must not be treated as schema keywords.
    names = {
        k: {"type": "number", "minimum": 0} for k in ("minimum", "maximum", "default", "minItems")
    }
    result = _strict_schema({"type": "object", "properties": names})
    assert set(result["properties"]) == set(names)
    assert '"minimum": 0' in result["properties"]["minimum"]["description"]


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"minimum": -3},
        {"maximum": 6},
        {"exclusive": 0},
        {"exclusive": 10},
        {"step": 0.3},
        {"count": 1.5},
    ],
)
def test_provider_response_still_enforces_original_numeric_contract(monkeypatch, change):
    from pydantic import BaseModel, Field

    class Numeric(BaseModel):
        minimum: float = Field(ge=-2)
        maximum: float = Field(le=5)
        exclusive: float = Field(gt=0, lt=10)
        step: float = Field(multiple_of=0.5)
        count: int = Field(ge=1)

    value = {"minimum": -2, "maximum": 5, "exclusive": 1, "step": 0.5, "count": 1}
    value.update(change or {})
    code = (
        "import pathlib,sys; pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text("
        + repr(json.dumps(value))
        + ")"
    )
    monkeypatch.setattr(
        "forge_mind.llm.codex_client._command", lambda: [sys.executable, "-c", code]
    )

    def call():
        return CodexClient().generate_structured(
            system="", user="", output_model=Numeric, model="gpt-6-astra", max_tokens=100
        )

    if change:
        with pytest.raises(LLMError):
            call()
    else:
        result, _ = call()
        assert result.model_dump() == value
