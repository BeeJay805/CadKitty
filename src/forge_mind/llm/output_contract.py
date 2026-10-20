"""Bounded output-rejection metadata and immutable correction arguments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from forge_mind.llm.client import LLMError

_RETRYABLE = {"json_decode", "pydantic_validation"}
_STAGES = {*_RETRYABLE, "map_restore", "utf8_decode", "missing_response", "output_overflow"}
_CODES = {
    "json_invalid",
    "utf8_invalid",
    "map_restore_invalid",
    "missing",
    "extra_forbidden",
    "bool_parsing",
    "bool_type",
    "int_parsing",
    "int_type",
    "int_from_float",
    "float_parsing",
    "float_type",
    "string_type",
    "string_too_short",
    "string_too_long",
    "string_pattern_mismatch",
    "list_type",
    "dict_type",
    "too_short",
    "too_long",
    "literal_error",
    "enum",
    "greater_than",
    "greater_than_equal",
    "less_than",
    "less_than_equal",
    "multiple_of",
    "finite_number",
    "union_tag_invalid",
    "union_tag_not_found",
    "value_error",
    "assertion_error",
}


def _nodes(
    schema: dict[str, Any], root: dict[str, Any], seen: tuple[str, ...] = ()
) -> Iterator[dict[str, Any]]:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if not reference.startswith("#/") or reference in seen:
            return
        target: Any = root
        for key in reference[2:].split("/"):
            target = target.get(key.replace("~1", "/").replace("~0", "~"), {})
            if not isinstance(target, dict):
                return
        yield from _nodes(target, root, (*seen, reference))
    elif "anyOf" in schema or "oneOf" in schema:
        for candidate in schema.get("anyOf", schema.get("oneOf", [])):
            yield from _nodes(candidate, root, seen)
    else:
        yield schema


def _safe_path(location: Any, schema: dict[str, Any]) -> list[str]:
    """Retain declared fields; never retain dynamic map keys or variant labels."""
    path: list[str] = []
    candidates = [schema]
    for part in location[:8] if isinstance(location, (tuple, list)) else ():
        nodes = [node for candidate in candidates for node in _nodes(candidate, schema)]
        fields = [
            node["properties"][part]
            for node in nodes
            if isinstance(part, str) and part in node.get("properties", {})
        ]
        variants = [node for node in nodes if isinstance(part, str) and node.get("title") == part]
        mappings = [
            node["additionalProperties"]
            for node in nodes
            if isinstance(node.get("additionalProperties"), dict)
        ]
        mappings += [
            child for node in nodes for child in node.get("patternProperties", {}).values()
        ]
        if fields:
            path.append(part[:64])
            candidates = fields
        elif variants:
            path.append("<variant>")
            candidates = variants
        elif type(part) is int:
            path.append("<item>")
            candidates = [
                node.get("prefixItems", [])[part]
                if 0 <= part < len(node.get("prefixItems", []))
                else node.get("items", {})
                for node in nodes
            ]
        elif mappings:
            path.append("<key>")
            candidates = mappings
        else:
            path.append("<unknown>")
            break
    return path


class OutputContractError(LLMError):
    """Only sanitized immutable metadata leaves the original live exception chain."""

    def __init__(
        self,
        *,
        stage: str,
        schema: dict[str, Any],
        violations: Sequence[Mapping[str, Any]] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__("Codex response did not satisfy the output contract")
        self._stage = stage if stage in _STAGES else "unknown"
        self._retryable = bool(retryable and self._stage in _RETRYABLE)
        violations = violations or []
        errors = [
            {
                "code": item.get("type") if item.get("type") in _CODES else "invalid_value",
                "path": _safe_path(item.get("loc", ()), schema),
            }
            for item in violations[:12]
        ]
        feedback = {
            "version": 1,
            "stage": self._stage,
            "schema_sha256": hashlib.sha256(
                json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "error_count": len(violations),
            "errors": errors,
        }
        while len(json.dumps(feedback, ensure_ascii=True)) > 2048 and errors:
            errors.pop()
        self._encoded = json.dumps(feedback, ensure_ascii=True, sort_keys=True)

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def retryable(self) -> bool:
        return self._retryable

    @property
    def feedback(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._encoded))

    def _with_retryable(self, retryable: bool) -> OutputContractError:
        error = OutputContractError(stage=self.stage, schema={}, retryable=retryable)
        error._encoded = self._encoded
        return error


def response_correction(arguments: dict[str, Any], error: OutputContractError) -> dict[str, Any]:
    """Append bounded host feedback without replacing owner input or call options."""
    if not error.retryable or not isinstance(arguments.get("system"), str):
        raise ValueError("Output correction requires eligible feedback and an original system")
    corrected = dict(arguments)
    corrected["system"] += (
        "\n\nHost output correction: The previous response failed local validation. "
        "Return a corrected complete response under the same original instructions and schema. "
        "The following bounded diagnostic metadata is not owner input or evidence:\n"
        + json.dumps(error.feedback, ensure_ascii=True, sort_keys=True)
    )
    return corrected
