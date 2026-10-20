"""Bounded durable tool observations, separate from model-authored design intent."""

from __future__ import annotations

import json
from typing import Any

MAX_FAILED_SUBMISSION_BYTES = 32000


def rejected_submission(
    action: dict[str, Any], revision_id: str | None, contract_hash: str, error: str
) -> dict[str, Any]:
    """Retain exact bounded author input, without claiming construction or evidence."""
    encoded = json.dumps(action, ensure_ascii=False).encode("utf-8")
    failure: dict[str, Any] = {
        "status": "validation_failed",
        "based_on_revision": revision_id,
        "contract_hash": contract_hash,
        "action_kind": action["kind"],
        "validation_error": error[:4000],
        "geometry_produced": False,
        "evidence_produced": False,
        "revision_created": False,
    }
    if len(encoded) <= MAX_FAILED_SUBMISSION_BYTES:
        failure["submitted_action"] = action
    else:
        failure["unavailable"] = (
            f"Exact submitted action exceeds the {MAX_FAILED_SUBMISSION_BYTES}-byte "
            "context retention limit; source was not retained. Read the active document "
            "and use a scoped edit, or submit a smaller action. This is not a CAD size limit."
        )
        failure["submitted_action_bytes"] = len(encoded)
    return failure


def active_submission(
    failure: dict[str, Any] | None, revision_id: str | None, contract_hash: str
) -> dict[str, Any] | None:
    """Only expose rejected input from the active contract and source revision."""
    if (
        failure
        and failure.get("based_on_revision") == revision_id
        and failure.get("contract_hash") == contract_hash
    ):
        return failure
    return None


def cache_schema(cache: dict[str, Any], name: str, schema: Any) -> dict[str, Any]:
    """Retain usable schema contents with a strict context/storage ceiling."""
    updated = {k: v for k, v in cache.items() if k != name}
    if len(json.dumps(schema)) <= 96000:
        updated[name] = schema
    while updated and (len(updated) > 6 or len(json.dumps(updated)) > 160000):
        del updated[next(iter(updated))]
    return updated


def _transaction_measurement(row: dict[str, Any]) -> dict[str, Any]:
    """Project only successful automatic transaction inventories; keep receipts exact."""
    if set(row) != {"construction_transaction"}:
        return row
    receipt = row["construction_transaction"]
    if not isinstance(receipt, dict) or receipt.get("action") != "construction":
        return row
    if any(receipt.get(key) is not True for key in ("applied", "plan_applied", "cad_applied")):
        return row
    if (
        "status" in receipt
        and receipt["status"] not in ("pass", "passed", "valid", "ok", "success", "applied")
    ) or any(
        receipt.get(key) for key in ("errors", "error", "caveats", "unresolved", "validation_error")
    ):
        return row
    measurement = receipt.get("measurement")
    revision_id = receipt.get("revision_id")
    if (
        not isinstance(measurement, dict)
        or measurement.get("geometry_valid") is not True
        or not isinstance(revision_id, str)
        or len(revision_id) != 32
        or any(c not in "0123456789abcdef" for c in revision_id)
        or receipt.get("candidate_revision_id") != revision_id
        or measurement.get("revision_id") != revision_id
    ):
        return row
    facts = measurement.get("facts")
    if not isinstance(facts, dict):
        return row
    # Reuse the established inventory projection only for the exact successful
    # assembly-history and cylinder shapes emitted by the existing producers.
    # Unknown extensions may contain diagnostics, so leave the entire receipt full.
    if any(key in facts for key in ("bodies", "faces", "edges", "surfaces", "topology")):
        return row
    history = facts.get("feature_history", [])
    if not isinstance(history, list) or any(
        not isinstance(item, dict)
        or set(item) != {"id", "state", "key"}
        or not isinstance(item["id"], str)
        or item["state"] not in ("built", "reused")
        or not isinstance(item["key"], str)
        or len(item["key"]) != 64
        or any(c not in "0123456789abcdef" for c in item["key"])
        for item in history
    ):
        return row
    assembly = facts.get("assembly")
    if assembly is not None:
        if not isinstance(assembly, dict):
            return row
        occurrences = assembly.get("occurrences", [])
        if not isinstance(occurrences, list) or any(
            not isinstance(item, dict) for item in occurrences
        ):
            return row
        from forge_mind.workspace.occurrence_facts import _surface

        fields = {
            "surface_id",
            "radius_mm",
            "diameter_mm",
            "axis_point_mm",
            "axis_direction",
            "concave",
            "axial_extent_mm",
        }
        for occurrence in occurrences:
            surfaces = occurrence.get("cylindrical_surfaces", [])
            if not isinstance(surfaces, list):
                return row
            for surface in surfaces:
                if (
                    not isinstance(surface, dict)
                    or set(surface) != fields
                    or type(surface["concave"]) is not bool
                    or surface["axial_extent_mm"] is None
                ):
                    return row
                try:
                    _surface(surface)
                except ValueError:
                    return row
    from forge_mind.workspace.native_review_context import _measurement

    return {
        "construction_transaction": {
            **receipt,
            "measurement": _measurement(measurement, revision_id),
        }
    }


def recent_context(history: list[dict[str, Any]], schemas: dict[str, Any]) -> list[dict[str, Any]]:
    """Send recent observations once, retaining failed construction diagnostics.

    Cached schemas are already supplied separately. Full measurements and source
    remain in immutable revisions; recent rows retain the exact diagnostic scope.
    Explicit part/source reads must stay complete so the next action can use them.
    """
    rows = []
    for item in history[-6:]:
        row = dict(item)
        name = row.get("schema_read")
        if name in schemas and row.get("authoring_schema") == schemas[name]:
            del row["authoring_schema"]
        measurement = row.get("measurement")
        if isinstance(measurement, dict):
            row["measurement"] = {
                key: measurement[key]
                for key in (
                    "revision_id",
                    "geometry_valid",
                    "requirements_satisfied",
                    "findings",
                    "errors",
                    "caveats",
                )
                if key in measurement
            }
            row["measurement"]["full_measurement_in_revision"] = True
        rows.append(_transaction_measurement(row))
    return rows


def remember(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest durable observations within the existing row/byte limits."""
    retained = []
    for item in reversed(history):
        # Large geometry results already live in immutable revision artifacts.
        value = {
            k: v
            for k, v in item.items()
            if k not in {"measurement", "inspection", "part", "authoring_schema"}
        }
        if not value:
            continue
        encoded = json.dumps(value)
        if len(encoded) > 16000:
            # A large assembly report must not erase its small sibling collision
            # summary or receipt. Preserve independently bounded observations.
            omitted = []
            bounded: dict[str, Any] = {}
            for key, result in value.items():
                if len(json.dumps({**bounded, key: result})) <= 15000:
                    bounded[key] = result
                else:
                    omitted.append(key)
            value = {
                **bounded,
                "observation": "Large tool fields retained in their revision/receipt; request them again by ID",
                "omitted_keys": omitted,
            }
        retained.append(value)
        if len(retained) == 32:
            break
    retained.reverse()
    while len(json.dumps(retained)) > 64000:
        retained.pop(0)
    return retained
