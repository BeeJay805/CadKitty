"""Strict author payloads shared by direct services and transport adapters."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class InputValidationError(ValueError):
    """A documented input rejection, without concealing internal runtime faults."""

    def __init__(self, code: str, field: str, message: str, *, retryable: bool = True) -> None:
        self.code = code
        self.field = field
        self.message = message
        self.retryable = retryable
        super().__init__(f"{field}: {message}")

    def error_result(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "retryable": self.retryable,
            "applied": False,
        }


def decode_json_object(value: str, label: str = "document_json") -> dict[str, Any]:
    """Decode one object; reject ambiguous keys and nonfinite numbers at any depth."""
    if not isinstance(value, str):
        raise InputValidationError(
            "invalid_json", label, "Supply a JSON object encoded as a string"
        )

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise InputValidationError(
                    "duplicate_json_key", f"{label}.{key}", "JSON object keys must be unique"
                )
            result[key] = item
        return result

    def constant(_value: str) -> Any:
        raise InputValidationError("nonfinite_json_number", label, "JSON numbers must be finite")

    def floating(number: str) -> float:
        parsed = float(number)
        if not math.isfinite(parsed):
            raise InputValidationError(
                "nonfinite_json_number", label, "JSON numbers must be finite"
            )
        return parsed

    try:
        result = json.loads(
            value, object_pairs_hook=object_pairs, parse_constant=constant, parse_float=floating
        )
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            "invalid_json", label, f"Malformed JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(result, dict):
        raise InputValidationError("invalid_json_object", label, "Supply a JSON object")
    return result


PayloadModel = TypeVar("PayloadModel", bound=BaseModel)


def validate_action_payload(
    model: type[PayloadModel],
    value: dict[str, Any],
    label: str = "document_json",
    *,
    strict: bool | None = None,
) -> PayloadModel:
    """Translate only model validation failures into recoverable field errors."""
    try:
        return model.model_validate(value, strict=strict)
    except ValidationError as exc:
        error = exc.errors(include_url=False, include_input=False)[0]
        location = ".".join(str(part) for part in error["loc"])
        field = f"{label}.{location}" if location else label
        message = str(error["msg"])
        if model in {ReadDesignArguments, ReadDesignQuery} and location == "cursor":
            message = "Use null or a returned cursor string; omit cursor for the first page"
        raise InputValidationError("invalid_action_argument", field, message) from exc


DesignCollection = Literal[
    "variables",
    "relations",
    "parts",
    "features",
    "roots",
    "subassemblies",
    "mates",
    "metadata",
    "occurrences",
    "nodes",
    "edges",
]


class ReadDesignArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scope: str | None = None
    cursor: str | None = None
    limit: Annotated[int, Field(gt=0)] = 100
    feature_id: str | None = None


class ReadDesignQuery(ReadDesignArguments):
    collection: DesignCollection


def read_design_arguments(collection: Any, value: dict[str, Any]) -> ReadDesignQuery:
    arguments = validate_action_payload(ReadDesignArguments, value, "read_design")
    return validate_action_payload(
        ReadDesignQuery, {"collection": collection, **arguments.model_dump()}, "read_design"
    )


class EmptyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExplainDesignArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    body_index: Annotated[int, Field(ge=0)] | None = None
    part_id: str | None = None
    feature_id: str | None = None
