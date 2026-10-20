"""Revision-bound vector drawings and an explicit inspection plan."""

from __future__ import annotations

import csv
import html
import math
import textwrap
from pathlib import Path
from typing import Any

from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.topology import resolve


def write_drawings(
    directory: Path,
    topology: dict[str, Any],
    revision: dict[str, Any],
    inputs: EngineeringInputs,
    *,
    inspection_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edges = [e for e in topology["entities"] if e["kind"] == "edge"]
    facts = revision["measurement"]["facts"]
    lo, hi = facts["bbox_min"], facts["bbox_max"]
    definitions = {
        part["id"]: part
        for part in ((revision.get("document") or {}).get("assembly") or {}).get("parts", [])
    }
    reference_indices = {
        occurrence["body_index"]
        for occurrence in (facts.get("assembly") or {}).get("occurrences", [])
        if (definitions.get(occurrence["definition"], {}).get("production") or {}).get("role")
        == "reference"
    }
    document = revision.get("document") or {}
    if (
        not document.get("assembly")
        and (document.get("production") or {}).get("role") == "reference"
    ):
        reference_indices = {
            entity["body_index"] for entity in topology["entities"] if entity["kind"] == "body"
        }
    model_bodies = [
        entity
        for entity in topology["entities"]
        if entity["kind"] == "body" and entity.get("body_index") not in reference_indices
    ]
    model_bounds = (
        {
            "min_mm": [min(body["min"][i] for body in model_bodies) for i in range(3)],
            "max_mm": [max(body["max"][i] for body in model_bodies) for i in range(3)],
        }
        if reference_indices and model_bodies
        else None
    )
    # Context volumes stay visible but cannot masquerade as manufactured outlines.
    edges.sort(key=lambda edge: edge.get("body_index") not in reference_indices)
    schedule = (inspection_plan or {}).get("rows", [])
    callouts: list[dict[str, Any]] = []
    omitted_callouts: list[dict[str, Any]] = []
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="620" viewBox="0 0 1200 620">',
        '<rect width="1200" height="620" fill="white"/>',
        '<g font-family="sans-serif" fill="#172025">',
        '<defs><marker id="dimension-arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#9b4620"/></marker></defs>',
        f'<text x="30" y="35" font-size="22">Engineering drawing · revision {revision["id"][:8]}</text>',
        '<text x="30" y="60" font-size="14">Millimeters. Orthographic edge projections; curves tessellated for display. Hidden edges are not classified.</text>',
    ]
    if reference_indices:
        svg.append(
            '<text x="30" y="82" font-size="14">Dashed gray: reference workpiece, tool or service volumes. Scene dimensions include these references.</text>'
        )
    for view, (a, z), column in (
        ("Top (XY)", (0, 1), 0),
        ("Front (XZ)", (0, 2), 1),
        ("Right (YZ)", (1, 2), 2),
    ):
        width, height = hi[a] - lo[a], hi[z] - lo[z]
        scale = min(300 / max(width, 1e-6), 330 / max(height, 1e-6))
        x0, y0 = 50 + 400 * column, 440
        svg.append(f'<text x="{x0}" y="105" font-size="18">{view}</text>')
        for edge in edges:
            points = " ".join(
                f"{x0 + (p[a] - lo[a]) * scale:.3f},{y0 - (p[z] - lo[z]) * scale:.3f}"
                for p in edge["polyline"]
            )
            style = (
                'stroke="#8b95a0" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.5"'
                if edge.get("body_index") in reference_indices
                else 'stroke="#26333a" stroke-width="1"'
            )
            svg.append(f'<polyline points="{points}" fill="none" {style}/>')
        view_count = 0
        for number, row in enumerate(schedule, 1):
            cad = row["cad"]
            endpoints = cad.get("dimension_points_mm")
            if cad["status"] != "measured" or not endpoints:
                continue
            xy = [(x0 + (p[a] - lo[a]) * scale, y0 - (p[z] - lo[z]) * scale) for p in endpoints]
            projection = math.dist(*xy)
            reason = (
                "Projection is coincident in this view"
                if projection < 1e-9
                else "Eight annotation display slots filled; see complete schedule"
                if view_count >= 8
                else "Datum-axis endpoints lie outside this view"
                if not all(x0 - 10 <= x <= x0 + 310 and 100 <= y <= 450 for x, y in xy)
                else None
            )
            if reason:
                omitted_callouts.append({"id": row["intent"]["id"], "view": view, "reason": reason})
                continue
            (x1, y1), (x2, y2) = xy
            label = f"#{number}: {cad['value_mm']:.4g} mm"
            label_y = 125 + 17 * view_count
            view_count += 1
            if projection < 1:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                svg.append(
                    f'<circle data-inspection="{html.escape(row["intent"]["id"])}" cx="{mx:.3f}" cy="{my:.3f}" r="3" fill="none" stroke="#9b4620"/>'
                )
                svg.append(
                    f'<path d="M{mx:.3f} {my:.3f}L{x0 + 160} {label_y - 4}" fill="none" stroke="#9b4620"/>'
                )
                label += " (small-gap location)"
            else:
                svg.append(
                    f'<path data-inspection="{html.escape(row["intent"]["id"])}" d="M{x1:.3f} {y1:.3f}L{x2:.3f} {y2:.3f}" fill="none" stroke="#9b4620" stroke-width="1.5" marker-start="url(#dimension-arrow)" marker-end="url(#dimension-arrow)"/>'
                )
            svg.append(f'<text x="{x0}" y="{label_y}" font-size="12" fill="#9b4620">{label}</text>')
            callouts.append(
                {
                    "id": row["intent"]["id"],
                    "view": view,
                    "number": number,
                    "display": "small-gap leader" if projection < 1 else "dimension arrows",
                    "dimension_points_mm": endpoints,
                    "scope": cad["dimension_scope"],
                }
            )
        svg.append(f'<path d="M{x0} 470h{width * scale:.3f}" stroke="#476e80"/>')
        envelope_label = "Scene" if reference_indices else "Overall"
        svg.append(
            f'<text x="{x0}" y="495" font-size="16">{envelope_label}: {width:.4g} x {height:.4g} mm</text>'
        )
        if model_bounds:
            model_width = model_bounds["max_mm"][a] - model_bounds["min_mm"][a]
            model_height = model_bounds["max_mm"][z] - model_bounds["min_mm"][z]
            svg.append(
                f'<text x="{x0}" y="516" font-size="14">Non-reference: {model_width:.4g} x {model_height:.4g} mm</text>'
            )
    dimension_results = []
    for dimension in inputs.dimensions:
        first = resolve(dimension.a.model_dump(), topology)
        second = resolve(dimension.b.model_dump(), topology)
        value = (
            math.dist(first["entity"]["center"], second["entity"]["center"])
            if first["entity"] and second["entity"]
            else None
        )
        dimension_results.append(
            {
                "id": dimension.id,
                "label": dimension.label,
                "value_mm": value,
                "status": "measured" if value is not None else "unresolved",
                "method": "Distance between kernel entity centers, not arbitrary clicked screen points",
            }
        )
    labels = [
        f"{d['label']}: {d['value_mm']:.4g} mm"
        if d["value_mm"] is not None
        else f"{d['label']}: unresolved"
        for d in dimension_results
    ]
    dimension_text = html.escape("; ".join(labels))
    schedule_lines = []
    for number, row in enumerate(schedule, 1):
        intent, cad = row["intent"], row["cad"]
        limits = (
            f"-{intent['minus_mm']:g}/+{intent['plus_mm']:g} mm"
            if intent["minus_mm"] is not None
            else "allowances unspecified"
        )
        nominal = (
            f"{intent['nominal_mm']:g} mm" if intent["nominal_mm"] is not None else "unspecified"
        )
        measurement_text = (
            f"{cad['value_mm']:g} mm" if cad["value_mm"] is not None else "no numeric measurement"
        )
        paragraphs = [
            f"#{number} {intent['id']} | {intent['label']} | Proposed nominal {nominal}, {limits}",
            f"Characteristic: {intent['characteristic']} | Occurrences: {', '.join(intent['occurrence_paths']) or 'unspecified'}",
            f"Datum: {intent['datum']} | Basis (author proposal): {intent['basis']}",
            f"CAD: {measurement_text}; {cad['status']}; probe {cad.get('probe_status', 'unavailable')}; receipt {cad['receipt_id'] or 'none'}",
            f"Numeric CAD references: occurrences {', '.join(cad.get('measured_occurrence_paths', [])) or 'unavailable'}; exported body indices {', '.join(str(i) for i in cad.get('measured_body_indices', [])) or 'unavailable'}",
            f"Scope: {cad.get('algorithm_scope') or cad['reason']}",
            f"Drawing: {cad.get('dimension_scope', 'No instrument-derived dimension endpoints')}",
            f"Proposed physical inspection: {intent['physical_method']} | Physical result: unverified",
            f"Missing evidence: {'; '.join(row['missing_evidence']) or 'None declared; this is not completeness verification'}",
        ]
        for paragraph in paragraphs:
            schedule_lines.extend(textwrap.wrap(paragraph, width=130) or [""])
        schedule_lines.append("")
    if schedule_lines:
        height = 665 + 21 * len(schedule_lines)
        svg[0] = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">'
        )
        svg[1] = f'<rect width="1200" height="{height}" fill="white"/>'
    svg += [
        f'<text x="30" y="540" font-size="14">{dimension_text}</text>',
        '<text x="30" y="580" font-size="14">Review datums, tolerances and critical features before manufacture. This sheet is not a standards-compliant GD&amp;T certification.</text>',
    ]
    if schedule_lines:
        svg.append(
            '<text x="30" y="625" font-size="18">Authored inspection schedule: CAD evidence and physical acceptance are separate</text>'
        )
        svg.extend(
            f'<text x="30" y="{655 + index * 21}" font-size="14">{html.escape(line)}</text>'
            for index, line in enumerate(schedule_lines)
        )
    svg.append("</g></svg>")
    (directory / "drawing.svg").write_text("\n".join(svg), encoding="utf-8")
    with (directory / "inspection.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Revision",
                "Requirement",
                "Target",
                "Tolerance",
                "Unit / method",
                "CAD status",
                "Physical measured",
                "Instrument",
                "Accepted by",
                "Row kind",
                "Characteristic ID",
                "Requirement IDs",
                "Occurrence paths",
                "Datum",
                "Proposed basis",
                "CAD measured mm",
                "Probe status",
                "Receipt ID",
                "Algorithm scope",
                "Missing evidence",
                "Physical uncertainty mm",
                "Proposed physical method",
                "Proposed lower mm",
                "Proposed upper mm",
                "Measured occurrence paths",
                "Measured body indices",
            ]
        )
        findings = {f["requirement_id"]: f for f in revision["measurement"]["findings"]}
        for req in revision["contract"]["requirements"]:
            f = findings.get(req["id"], {})
            text = str(req["text"])
            # Spreadsheet formula injection must not travel in a shareable CSV.
            if text.lstrip()[:1] in ("=", "+", "-", "@"):
                text = "'" + text
            writer.writerow(
                [
                    revision["id"],
                    text,
                    req["value"],
                    req["tolerance"],
                    f.get("method", "Unverified"),
                    f.get("status", "unverified"),
                    "",
                    "",
                    "",
                    "Frozen requirement",
                    "",
                    req["id"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
        for row in schedule:
            intent, cad = row["intent"], row["cad"]
            cells = [
                revision["id"],
                intent["characteristic"],
                intent["nominal_mm"],
                f"-{intent['minus_mm']}/+{intent['plus_mm']}"
                if intent["minus_mm"] is not None
                else "Unspecified",
                "mm",
                cad["status"],
                "",
                "",
                "",
                "Authored inspection proposal",
                intent["id"],
                "; ".join(intent["requirement_ids"]),
                "; ".join(intent["occurrence_paths"]),
                intent["datum"],
                intent["basis"],
                cad["value_mm"],
                cad.get("probe_status", ""),
                cad["receipt_id"],
                cad["algorithm_scope"],
                "; ".join(row["missing_evidence"]),
                "",
                intent["physical_method"],
                intent["nominal_mm"] - intent["minus_mm"] if intent["minus_mm"] is not None else "",
                intent["nominal_mm"] + intent["plus_mm"] if intent["plus_mm"] is not None else "",
                "; ".join(cad.get("measured_occurrence_paths", [])),
                "; ".join(str(i) for i in cad.get("measured_body_indices", [])),
            ]
            writer.writerow(
                [
                    "'" + cell
                    if isinstance(cell, str) and cell.lstrip()[:1] in ("=", "+", "-", "@")
                    else cell
                    for cell in cells
                ]
            )
    return {
        "revision_id": revision["id"],
        "sheet": "drawing.svg",
        "inspection_plan": "inspection.csv",
        "dimensions": dimension_results,
        "inspection_characteristics": len(schedule),
        "reference_body_count": len(reference_indices),
        "non_reference_bounds": model_bounds,
        "inspection_callouts": callouts,
        "omitted_inspection_callouts": omitted_callouts,
        "status": "review_required",
    }
