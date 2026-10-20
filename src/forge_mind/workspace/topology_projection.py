"""Pure, paged views of saved topology; no CAD reconstruction or new measurements."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any

KINDS = ("body", "face", "edge")
FACT_KEYS = frozenset(
    (
        "id",
        "kind",
        "signature",
        "body_index",
        "body_signature",
        "center",
        "min",
        "max",
        "measure",
        "unit",
        "surface",
        "normal",
        "curve",
    )
)
SCOPE = (
    "Saved revision-local topology facts; no adjacency, full surface coverage, "
    "continuity, physical fit or new geometric measurement is established."
)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate topology JSON field")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError("Topology JSON contains a nonfinite number: " + value)


def _vector(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Topology vector must contain three finite numbers")
    for item in value:
        if type(item) not in (int, float):
            raise ValueError("Topology vector must contain three finite numbers")
        try:
            finite = math.isfinite(item)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("Topology vector must contain three finite numbers")


def topology_projection(
    content: bytes,
    revision_id: str,
    *,
    offset: int = 0,
    limit: int = 40,
    body_index: int | None = None,
    kind: str | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    """Caller supplies owned, hash-verified artifact bytes and enforces response limits.

    The legacy response preserves complete entity records and its four original fields.
    Filters precede pagination. Compact facts omit meshes and edge display polylines.
    """
    if (
        type(offset) is not int
        or offset < 0
        or type(limit) is not int
        or not 1 <= limit <= 100
        or (body_index is not None and (type(body_index) is not int or body_index < 0))
        or (kind is not None and kind not in KINDS)
        or type(compact) is not bool
    ):
        raise ValueError("Invalid topology page or filter")
    try:
        topology = json.loads(content, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Revision has no valid measured topology") from exc
    if not isinstance(topology, dict) or not isinstance(topology.get("entities"), list):
        raise ValueError("Revision has no valid measured topology")
    entities = topology["entities"]
    # Preserve old unfiltered, noncompact response semantics and complete records.
    if body_index is None and kind is None and not compact:
        next_offset = offset + limit
        return {
            "revision_id": revision_id,
            "total": len(entities),
            "entities": deepcopy(entities[offset:next_offset]),
            "next_offset": next_offset if next_offset < len(entities) else None,
        }
    identities: set[str] = set()
    for row in entities:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in identities
            or row.get("kind") not in KINDS
            or type(row.get("body_index")) is not int
            or row["body_index"] < 0
        ):
            raise ValueError("Malformed or duplicate topology entity")
        identities.add(row["id"])
        if compact:
            for key in ("signature", "body_signature", "unit"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError("Missing topology fact identity or unit")
            for key in ("center", "min", "max"):
                _vector(row.get(key))
            if any(a > b for a, b in zip(row["min"], row["max"], strict=True)):
                raise ValueError("Reversed topology bounds")
            measure = row.get("measure")
            if not isinstance(measure, (int, float)) or isinstance(measure, bool):
                raise ValueError("Invalid topology measure")
            try:
                valid_measure = math.isfinite(measure) and measure >= 0
            except OverflowError:
                valid_measure = False
            if not valid_measure:
                raise ValueError("Invalid topology measure")
            if row.get("normal") is not None:
                _vector(row["normal"])
    scoped = [row for row in entities if body_index is None or row["body_index"] == body_index]
    filtered = [row for row in scoped if kind is None or row["kind"] == kind]
    next_offset = offset + limit
    page = filtered[offset:next_offset]
    result = {
        "revision_id": revision_id,
        "total": len(filtered),
        "entities": deepcopy(page),
        "next_offset": next_offset if next_offset < len(filtered) else None,
    }
    if not compact:
        return result
    if topology.get("frame") != "cad_z_up_mm":
        raise ValueError("Compact topology requires a saved CAD Z-up millimeter frame")
    native_counts = topology.get("topology_counts")
    if native_counts is not None and (
        not isinstance(native_counts, dict)
        or any(
            type(native_counts.get(key)) is not int or native_counts[key] < 0
            for key in ("bodies", "faces", "edges")
        )
    ):
        raise ValueError("Malformed native topology counts")
    policy = topology.get("detail_policy")
    if policy is not None and not isinstance(policy, dict):
        raise ValueError("Malformed topology detail policy")
    counts = {key: sum(row["kind"] == key for row in scoped) for key in KINDS}
    result.update(
        topology_sha256=hashlib.sha256(content).hexdigest(),
        frame=topology["frame"],
        returned=len(page),
        unfiltered_total=len(entities),
        filters={"body_index": body_index, "kind": kind},
        indexed_counts=counts,
        topology_counts=deepcopy(native_counts),
        detail_policy=deepcopy(policy),
        inventory={
            "scope": "all indexed bodies" if body_index is None else "requested body_index",
            "body_present": counts["body"] > 0,
            "faces": "indexed" if counts["face"] else "unavailable",
            "edges": "indexed" if counts["edge"] else "unavailable",
            "missing_detail": "No indexed faces or edges does not establish absent geometry.",
            "native_counts_scope": "whole saved model; not selected-body counts",
        },
        scope=SCOPE,
        entities=[
            {key: deepcopy(value) for key, value in row.items() if key in FACT_KEYS} for row in page
        ],
    )
    return result
