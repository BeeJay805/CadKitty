"""Exact selected native adjacency; CAD imports occur only inside evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge_mind.workspace.geometry_neighbors_contract import ResolvedNeighborSelector


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Native neighborhood returned invalid/nonfinite geometry")
    return float(value)


def _face_edge_uses(face: Any) -> list[Any]:
    """Traverse edge uses, preserving repeated seam uses that face.edges() deduplicates."""
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    explorer = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
    result = []
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result


def inspect_selected_neighbors(
    step_path: Path,
    topology: dict[str, Any],
    selector: ResolvedNeighborSelector,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    """Read a full native one-ring before paging; missing identity is never an empty ring.

    The service owns source hashes and access. Incidence uses IsSame native identity,
    not proximity, signatures alone or author-declared functional relationships.
    """
    import build123d as cad
    from OCP.BRepCheck import BRepCheck_Analyzer

    from forge_mind.workspace.assembly import named_solids, occurrence_label
    from forge_mind.workspace.geometry_neighbors_contract import (
        SCOPE,
        EntityFact,
        NeighborQuery,
        ResolvedNeighborSelector,
    )
    from forge_mind.workspace.topology import same_exported_solid, signature

    if not isinstance(selector, ResolvedNeighborSelector):
        raise ValueError("Neighbors require a resolved saved selector")
    query = NeighborQuery(selector=selector.requested, offset=offset, limit=limit)
    if not isinstance(topology, dict) or topology.get("frame") != "cad_z_up_mm":
        raise ValueError("Neighbors topology has no supported saved coordinate frame")
    entities = topology.get("entities")
    if not isinstance(entities, list) or any(not isinstance(row, dict) for row in entities):
        raise ValueError("Neighbors topology has an invalid entity inventory")
    shape = cad.import_step(step_path)
    if (
        shape is None
        or shape.wrapped is None
        or shape.wrapped.IsNull()
        or not BRepCheck_Analyzer(shape.wrapped).IsValid()
    ):
        raise ValueError("Saved neighbors STEP is invalid or empty")
    solids = list(shape.solids())
    if not solids:
        raise ValueError("Saved neighbors STEP contains no solids")
    body_rows = [row for row in entities if row.get("kind") == "body"]
    indices = [row.get("body_index") for row in body_rows]
    if (
        any(type(index) is not int for index in indices)
        or len(body_rows) != len(solids)
        or set(indices) != set(range(len(solids)))
    ):
        raise ValueError("Saved topology body inventory is missing or ambiguous")
    counts = topology.get("topology_counts")
    if isinstance(counts, dict) and (
        type(counts.get("bodies")) is not int or counts["bodies"] != len(solids)
    ):
        raise ValueError("Saved topology native body count disagrees with STEP")
    body_signatures = [signature(solid) for solid in solids]
    for row in body_rows:
        index = row["body_index"]
        if (
            row.get("signature") != body_signatures[index]
            or row.get("body_signature") != body_signatures[index]
        ):
            raise ValueError("Saved topology body signature disagrees with actual STEP placement")
    index = selector.body_index
    if index >= len(solids) or body_signatures[index] != selector.body_signature:
        raise ValueError("Selected body identity disagrees with actual saved STEP")
    if body_signatures.count(selector.body_signature) != 1:
        raise ValueError("Selected saved body correspondence is ambiguous")
    body = solids[index]
    if not BRepCheck_Analyzer(body.wrapped).IsValid():
        raise ValueError("Selected saved native body is invalid")
    if selector.requested.occurrence_path is not None:
        occurrence = named_solids(shape).get(occurrence_label(selector.requested.occurrence_path))
        matches = (
            []
            if occurrence is None
            else [i for i, solid in enumerate(solids) if same_exported_solid(occurrence, solid)]
        )
        if matches != [index]:
            raise ValueError("Selected occurrence label disagrees with indexed saved body")

    native = {"face": list(body.faces()), "edge": list(body.edges())}
    signatures = {kind: [signature(item) for item in items] for kind, items in native.items()}
    saved = {
        kind: [
            (ordinal, row)
            for ordinal, row in enumerate(entities)
            if row.get("kind") == kind
            and type(row.get("body_index")) is int
            and row["body_index"] == index
        ]
        for kind in native
    }

    def paired(kind: str, item: Any) -> tuple[int, dict[str, Any]]:
        if (
            item.wrapped is None
            or item.wrapped.IsNull()
            or not BRepCheck_Analyzer(item.wrapped).IsValid()
        ):
            raise ValueError("Selected neighborhood contains invalid native topology")
        digest = signature(item)
        native_matches = [
            candidate
            for candidate, value in zip(native[kind], signatures[kind], strict=True)
            if value == digest
        ]
        rows = [(ordinal, row) for ordinal, row in saved[kind] if row.get("signature") == digest]
        matched_shape = native_matches[0].wrapped if len(native_matches) == 1 else None
        if (
            matched_shape is None
            or not matched_shape.IsSame(item.wrapped)
            or len(rows) != 1
            or rows[0][1].get("body_signature") != selector.body_signature
        ):
            raise ValueError("Neighborhood native/indexed entity is missing or ambiguous")
        ordinal, row = rows[0]
        bounds, center = item.bounding_box(), item.center()
        values: dict[str, Any] = {
            "id": row.get("id"),
            "kind": kind,
            "signature": digest,
            "body_index": index,
            "body_signature": selector.body_signature,
            "center": [_number(value) for value in center],
            "min": [_number(value) for value in bounds.min],
            "max": [_number(value) for value in bounds.max],
            "measure": _number(item.area if kind == "face" else item.length),
            "unit": "mm^2" if kind == "face" else "mm",
        }
        if kind == "face":
            values["surface"] = str(item.geom_type)
            try:
                normal = item.normal_at()
            except (ValueError, RuntimeError):
                normal = None
            values["normal"] = None if normal is None else [_number(value) for value in normal]
        else:
            values["curve"] = str(item.geom_type)
        return ordinal, EntityFact.model_validate(values).model_dump(mode="json", exclude_none=True)

    kind = selector.requested.kind
    entity_matches = [
        item
        for item, value in zip(native[kind], signatures[kind], strict=True)
        if value == selector.requested.signature
    ]
    if len(entity_matches) != 1:
        raise ValueError("Selected actual saved entity is missing or ambiguous")
    focal_shape = entity_matches[0]
    _, focal = paired(kind, focal_shape)
    neighbors: list[tuple[int, dict[str, Any]]] = []
    if kind == "face":
        grouped: list[tuple[Any, int]] = []
        for use in _face_edge_uses(focal_shape):
            positions = [i for i, (edge, _) in enumerate(grouped) if edge.IsSame(use)]
            if positions:
                position = positions[0]
                edge, count = grouped[position]
                grouped[position] = (edge, count + 1)
            else:
                grouped.append((use, 1))
        for raw_edge, count in grouped:
            edge_matches = [
                edge
                for edge in native["edge"]
                if (wrapped := edge.wrapped) is not None and wrapped.IsSame(raw_edge)
            ]
            if len(edge_matches) != 1:
                raise ValueError("Native face boundary edge correspondence is missing or ambiguous")
            ordinal, fact = paired("edge", edge_matches[0])
            neighbors.append((ordinal, {"entity": fact, "incidence_count": count}))
        relation = "face_boundary_edges"
    else:
        for face in native["face"]:
            count = sum(use.IsSame(focal_shape.wrapped) for use in _face_edge_uses(face))
            if count:
                ordinal, fact = paired("face", face)
                neighbors.append((ordinal, {"entity": fact, "incidence_count": count}))
        relation = "edge_incident_faces"
    # Pair the COMPLETE ring before slicing; missing records cannot disappear off-page.
    neighbors.sort(key=lambda value: value[0])
    items = [value for _, value in neighbors]
    ids = [row["entity"]["id"] for row in items]
    digests = [row["entity"]["signature"] for row in items]
    if len(set(ids)) != len(ids) or len(set(digests)) != len(digests) or focal["id"] in ids:
        raise ValueError("Neighborhood saved identities are duplicate or ambiguous")
    page = items[query.offset : query.offset + query.limit]
    end = query.offset + len(page)
    return {
        "focal": focal,
        "relation": relation,
        "neighbors": page,
        "total_neighbors": len(items),
        "total_incidences": sum(row["incidence_count"] for row in items),
        "next_offset": end if end < len(items) else None,
        "native_completed": True,
        "coverage": "complete_selected_neighborhood",
        "scope": SCOPE,
    }
