"""Pure stage projections; the caller owns phase selection and admission."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

_STARTUP_FIELDS = (
    "request",
    "user_revision_history",
    "edit_instruction",
    "frozen_contract",
    "proposed_contract",
    "request_attribution",
    "engineering_inputs",
    "owned_assets",
    "source_claims",
    "source_artifacts",
    "edit_basis",
    "seconds_left",
    "actions_left",
    "per_call_seconds",
    "image_observation",
    "failed_submission",
    "photo_generation",
)


def startup_context(context: dict[str, Any]) -> dict[str, Any]:
    """Supply original intent and references without construction syntax or verdicts."""
    result = {key: deepcopy(context[key]) for key in _STARTUP_FIELDS if key in context}
    result["phase"] = "startup_intent"
    result["scope"] = "Pending authored intent; not geometry or completion evidence"
    return result


def _schema_record(name: str, value: Any) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "name": name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "query": name,
        "retrieve": f"schema query={name}",
    }


def construction_context(context: dict[str, Any]) -> dict[str, Any]:
    """Retain exact cached syntax and every other saved or requested fact.

    The caller already bounds the schema cache. Keep its exact contents across
    construction turns and deduplicate only matching historical schema copies.
    No history, geometry, evidence, error, source, plan, or explicit read is bounded.
    """
    result = deepcopy(context)
    bootstrap = result.get("native_authoring")
    if isinstance(bootstrap, dict):
        bootstrap.pop("plan_schema", None)

    schemas = result.get("authoritative_schemas")
    if not isinstance(schemas, dict):
        return result

    history = result.get("history")
    result["schema_catalog"] = [_schema_record(name, value) for name, value in schemas.items()]
    if isinstance(history, list):
        for row in history:
            if not isinstance(row, dict):
                continue
            name = row.get("schema_read")
            if (
                isinstance(name, str)
                and name in schemas
                and row.get("authoring_schema") == schemas[name]
            ):
                row.pop("authoring_schema", None)
    return result


def deduplicate_context(context: dict[str, Any]) -> dict[str, Any]:
    """Replace exact automatic copies with pointers to full values in this request.

    Run after all stage-specific projections, immediately before serialization.
    Explicit reads, failure witnesses and unknown top-level fields stay full.
    Host handoff receipts can reference identical nonfailure instruction siblings.
    Automatic binding facts and provenance copies can reuse full guidance anchors.
    No record, source, diagnostic, schema or evidence is truncated or discarded.
    """
    result = deepcopy(context)
    anchors: dict[str, tuple[str, Any]] = {}

    def encoded(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def pointer(path: str, name: str | int) -> str:
        return path + "/" + str(name).replace("~", "~0").replace("/", "~1")

    def index(value: Any, path: str) -> None:
        if isinstance(value, dict) and set(value) == {"context_reference"}:
            return
        if isinstance(value, (dict, list, str)):
            anchors.setdefault(encoded(value), (path, value))
        if isinstance(value, dict):
            for name, child in value.items():
                index(child, pointer(path, name))
        elif isinstance(value, list):
            for number, child in enumerate(value):
                index(child, pointer(path, number))

    def reference(path: str, text: str) -> dict[str, Any]:
        return {
            "context_reference": {
                "json_pointer": path,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "retrieve": "Complete exact value at this JSON pointer in this request; not a verdict or replacement CAD.",
            }
        }

    def compact(value: Any) -> Any:
        if not isinstance(value, (dict, list, str)):
            return value
        text = encoded(value)
        match = anchors.get(text)
        if match is not None:
            path, original = match
            # JSON equality is exact here: Python equality alone conflates 1/True.
            if text == encoded(original):
                replacement = reference(path, text)
                if len(encoded(replacement)) < len(text):
                    return replacement
        if isinstance(value, dict):
            return {name: compact(child) for name, child in value.items()}
        if isinstance(value, list):
            return [compact(child) for child in value]
        return value

    def protected(value: Any, *, row: bool = False) -> bool:
        if isinstance(value, dict):
            if row and (
                any(
                    name
                    in {
                        "part",
                        "read_part",
                        "read_design",
                        "inspection",
                        "tool_error",
                        "failed_submission",
                    }
                    or name.startswith("source")
                    for name in value
                )
                or value.get("explicit_guidance_read") is True
            ):
                return True
            if value.get("status") in (
                "fail",
                "failed",
                "invalid",
                "validation_failed",
                "checked_fail",
                "needs_review",
                "error",
            ):
                return True
            if any(value.get(name) for name in ("errors", "error", "validation_error")):
                return True
            return any(protected(child) for child in value.values())
        if isinstance(value, list):
            return any(protected(child) for child in value)
        return False

    def has_reference(value: Any) -> bool:
        if isinstance(value, dict):
            return "context_reference" in value or any(has_reference(v) for v in value.values())
        return isinstance(value, list) and any(has_reference(v) for v in value)

    def instruction_copies(guidance: Any, path: str) -> None:
        """Reuse full automatic guidance anchors, without changing explicit tool reads."""
        if not isinstance(guidance, dict) or guidance.get("explicit_guidance_read") is True:
            return
        catalog = guidance.get("instruction_catalog")
        binding = guidance.get("instruction_binding")
        facts = catalog.get("facts") if isinstance(catalog, dict) else None
        bound = binding.get("facts") if isinstance(binding, dict) else None
        if isinstance(facts, dict) and isinstance(bound, list):
            exact = {
                encoded(value): pointer(path + "/instruction_catalog/facts", name)
                for name, value in facts.items()
                if isinstance(value, dict) and not protected(value) and not has_reference(value)
            }
            for number, value in enumerate(bound):
                text = encoded(value)
                anchor = exact.get(text)
                if anchor is not None:
                    replacement = reference(anchor, text)
                    if len(encoded(replacement)) < len(text):
                        bound[number] = replacement
        record = guidance.get("record")
        provenance = guidance.get("record_provenance")
        history = provenance.get("history") if isinstance(provenance, dict) else None
        if (
            isinstance(record, dict)
            and isinstance(history, list)
            and not protected(record)
            and not has_reference(record)
        ):
            text = encoded(record)
            replacement = reference(path + "/record", text)
            if len(encoded(replacement)) < len(text):
                for row in history:
                    if (
                        isinstance(row, dict)
                        and not protected(row, row=True)
                        and "record" in row
                        and encoded(row["record"]) == text
                    ):
                        row["record"] = deepcopy(replacement)

    # Full current authority remains first. Only explicitly named automatic
    # duplicates below can change; an unknown root is not a compaction candidate.
    for name in (
        "frozen_contract",
        "design_plan",
        "edit_basis",
        "source_claims",
        "source_artifacts",
        "engineering_inputs",
        "failed_submission",
        "failed_attempt",
        "early_checks",
        "completion",
        "native_authoring",
        "authoritative_schemas",
    ):
        if name in result:
            index(result[name], pointer("", name))
    for name, field in (("request_attribution", "proposed_contract"), ("selected", "contract")):
        value = result.get(name)
        if isinstance(value, dict):
            if field in value:
                value[field] = compact(value[field])
            index(value, pointer("", name))
    for name in ("delivery_guidance", "customer_delivery"):
        if name in result:
            result[name] = compact(result[name])
            if name == "delivery_guidance":
                instruction_copies(result[name], pointer("", name))
            index(result[name], pointer("", name))
    for name in ("history", "persistent_tool_observations"):
        rows = result.get(name)
        if isinstance(rows, list):
            for number, row in enumerate(rows):
                if not protected(row, row=True):
                    rows[number] = compact(row)
                elif (
                    isinstance(row, dict)
                    and row.get("origin") == "host"
                    and isinstance(row.get("handoff_preparation"), dict)
                    and row.get("explicit_guidance_read") is not True
                    and not any(
                        name
                        in {
                            "part",
                            "read_part",
                            "read_design",
                            "inspection",
                            "tool_error",
                            "failed_submission",
                        }
                        or name.startswith("source")
                        for name in row
                    )
                ):
                    # An automatic receipt can contain failed facts beside exact
                    # copies of current instruction syntax. Keep its historical
                    # basis, verdicts and all facts complete. Only these safe
                    # siblings may reference identical values already in context.
                    preparation = row["handoff_preparation"]
                    schema = preparation.get("guidance_schema")
                    if schema is not None and not protected(schema):
                        preparation["guidance_schema"] = compact(schema)
                    guidance = preparation.get("delivery_guidance")
                    if isinstance(guidance, dict):
                        for field in ("instruction_catalog", "instruction_binding"):
                            if field in guidance and not protected(guidance[field]):
                                guidance[field] = compact(guidance[field])
                index(rows[number], pointer(pointer("", name), number))
    return result
