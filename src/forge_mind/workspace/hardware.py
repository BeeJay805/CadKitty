"""Reuse the sourced catalog; only dependency-reachable inserted hardware contributes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from forge_mind.agentic.catalog import registry
from forge_mind.workspace.document import DesignDocument


def search(
    query: str = "",
    *,
    family: str | None = None,
    subtype: str | None = None,
    thread_d: float | None = None,
    length: float | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Filter existing sourced entries before limiting; dimensions are exact millimeters."""
    if not 1 <= limit <= 500:
        raise ValueError("Catalog limit must be between 1 and 500")
    candidates = registry.search_catalog(
        query, family=family, limit=registry.catalog_counts()["total"]
    )
    return [
        asdict(part)
        for part in candidates
        if (subtype is None or part.subtype == subtype)
        and (thread_d is None or part.dims_mm.get("thread_d") == thread_d)
        and (length is None or part.dims_mm.get("length") == length)
    ][:limit]


def hardware_shape(catalog_id: str, envelope: bool = False) -> Any:
    part = registry.get_catalog_part(catalog_id)
    if envelope:
        import build123d as b

        d = part.dims_mm
        if part.family == "bearing":
            outer, bore, width = d["outer_d"], d["bore_d"], d["width"]
            return b.Pos(0, 0, width / 2) * (
                b.Cylinder(outer / 2, width) - b.Cylinder(bore / 2, width + 2)
            )
        if part.subtype in {"socket_head_cap_screw", "button_head_screw", "hex_head_bolt"}:
            # bd_warehouse simple threads expose the minor diameter. An explicit
            # nominal envelope includes the thread crests for collision/port checks.
            shaft = b.Pos(0, 0, -d["length"] / 2) * b.Cylinder(d["thread_d"] / 2, d["length"])
            return registry.catalog_geometry(part).fuse(shaft)
        raise ValueError(
            "Catalog envelope mode supports bearings and non-countersunk headed screws"
        )
    return registry.catalog_geometry(part)


def bill_of_materials(
    document: DesignDocument | None, assets: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if document is None:
        return {
            "lines": [],
            "unmapped": ["Imported/Python bases have no declared hardware inventory"],
        }
    if document.assembly:
        from forge_mind.workspace.production import production_report

        production = production_report(document)
        purchased: list[dict[str, Any]] = []
        for row in production["buy"]:
            definition = next(p for p in document.assembly.parts if p.id == row["part_id"])
            child = bill_of_materials(
                DesignDocument(
                    variables=document.variables,
                    relations=document.relations,
                    features=definition.features,
                    outputs=[definition.output],
                ),
                assets,
            )
            if child["lines"]:
                purchased.extend(
                    {
                        **line,
                        "quantity": line["quantity"] * row["quantity"],
                        "role": "buy",
                        "part_id": row["part_id"],
                    }
                    for line in child["lines"]
                )
            else:
                purchased.append(row)
        return {
            "lines": purchased,
            "part_definitions": production["lines"],
            "make": production["make"],
            "buy": production["buy"],
            "reference": production["reference"],
            "unresolved": production["unresolved"],
            "unmapped": production["gaps"],
            "scope": production["scope"],
        }
    features = {f.id: f for f in document.features}
    counts: dict[str, int] = {}
    unmapped = []
    inventory = {a["id"]: a for a in assets or []}
    imports: dict[str, dict[int, int]] = {}
    captured_lines = []
    modified: set[str] = set()

    weights = dict.fromkeys(features, 0)
    for output in document.outputs or ([document.features[-1].id] if document.features else []):
        weights[output] += 1
    # Reverse topological propagation is linear in the document graph, including shared inputs.
    for f in reversed(document.features):
        multiplier = weights[f.id]
        if f.suppressed or not multiplier:
            continue
        if f.id in modified or f.kind in {"union", "cut", "intersect"}:
            modified.update(f.inputs)
        if f.id in modified and f.kind in {"hardware", "import"}:
            unmapped.append(
                f"{f.name}: boolean-modified hardware/base needs inventory reassignment"
            )
            continue
        if f.kind == "hardware":
            counts[f.catalog_id or ""] = counts.get(f.catalog_id or "", 0) + multiplier
        if f.kind == "import":
            asset = inventory.get(f.asset_id or "")
            if not asset or not asset.get("inventory"):
                unmapped.append(f"{f.name}: imported inventory not declared")
            else:
                selected = imports.setdefault(asset["id"], {})
                index = document.values().get(
                    f.bindings.get("solid_index", ""), f.params.get("solid_index")
                )
                indices = range(asset["solid_count"]) if index is None else [int(index)]
                for i in indices:
                    selected[i] = selected.get(i, 0) + multiplier
        if f.kind == "pattern":
            multiplier *= int(
                document.values().get(f.bindings.get("count", ""), f.params.get("count", 1))
            )
        # Subtractive tools are not installed parts.
        for source in f.inputs[:1] if f.kind in {"cut", "intersect"} else f.inputs:
            weights[source] += multiplier
    for asset_id, selected in imports.items():
        asset = inventory[asset_id]
        if set(selected) != set(range(asset["solid_count"])) or len(set(selected.values())) != 1:
            unmapped.append(
                f"{asset['name']}: partial body selection requires inventory reassignment"
            )
            continue
        copies = next(iter(selected.values()))
        for line in asset["inventory"]["lines"]:
            captured_lines.append({**line, "quantity": copies * line["quantity"]})
        unmapped.extend(asset["inventory"].get("unmapped", []))
    lines = []
    for key, quantity in counts.items():
        part = registry.get_catalog_part(key)
        lines.append(
            {
                "catalog_id": key,
                "geometry_detail": "dimensional envelope; internal operation unverified"
                if any(
                    f.catalog_id == key and f.params.get("envelope") == 1 for f in features.values()
                )
                else "catalog geometry; thread detail and ratings require review",
                "name": part.name,
                "quantity": quantity,
                "source": part.source,
                "provisional": part.provisional,
                "dimensions_mm": part.dims_mm,
            }
        )
    from forge_mind.workspace.production import production_report

    production = production_report(document)
    if document.production and document.production.role in {"make", "reference"}:
        lines, captured_lines = [], []
    elif (
        document.production
        and document.production.role == "buy"
        and not lines
        and not captured_lines
    ):
        lines = production["buy"]
    return {
        "lines": [*lines, *captured_lines],
        "make": production["make"],
        "buy": production["buy"],
        "reference": production["reference"],
        "unresolved": production["unresolved"],
        "unmapped": sorted(set(unmapped)),
        "scope": "Declared inserted hardware in reachable output history; captured catalog snapshots remain separate lines. Boolean-modified hardware needs reassignment. No hardware inferred from holes.",
    }
