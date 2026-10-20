"""Build-preview metadata derived from a PartProgram.

Additive and presentation-only: turns the (already JSON-serializable) operation list into a
frontend-friendly ``build`` block:

  - ``program``:      a trimmed parametric op list the viewer reconstructs in three.js
  - ``build_steps``:  an ordered, plain-English timeline (one entry per operation)
  - ``feature_tags``: maps an operation to the dimensions it controls + how to preview it

The frontend uses this to power the Watch-build reveal and the live dimension previews. Nothing here
touches geometry or the deterministic compiler, so it can never change the produced part.
"""

from __future__ import annotations

from typing import Any

from forge_mind.dsl.primitives import Box, ComputedSolid, Cylinder, OpMode
from forge_mind.dsl.program import PartProgram


def _nice(op_id: str) -> str:
    return op_id.replace("_", " ").strip() or "feature"


def _r(v: float) -> float:
    return round(float(v), 2)


def _step(op: Any) -> tuple[str, str, list[str]]:
    """(label, kind, affected_dimensions) for one operation."""
    verb = "Cut" if op.mode == OpMode.SUBTRACT else "Add"
    name = _nice(op.op_id)
    if isinstance(op, Box):
        return (
            f"{verb} {name} ({_r(op.width)} x {_r(op.depth)} x {_r(op.height)} mm)",
            "box",
            [f"{op.op_id}_size_mm"],
        )
    if isinstance(op, Cylinder):
        dia = _r(op.radius * 2)
        if op.mode == OpMode.SUBTRACT:
            return f"{verb} {name} (dia {dia} mm)", "cylinder", [f"{op.op_id}_diameter_mm"]
        return (
            f"{verb} {name} (dia {dia} x {_r(op.height)} mm)",
            "cylinder",
            [f"{op.op_id}_diameter_mm"],
        )
    if isinstance(op, ComputedSolid):
        return (
            f"{verb} {name} ({op.section} strut, {_r(op.length)} mm)",
            "computed_solid",
            [f"{op.op_id}_length_mm"],
        )
    return f"{verb} {name}", "operation", []


def _preview_mode(op: Any) -> str:
    if isinstance(op, Box):
        return "extrusion_depth"
    if isinstance(op, Cylinder):
        return "hole_diameter" if op.mode == OpMode.SUBTRACT else "radius"
    if isinstance(op, ComputedSolid):
        return "length"
    return "none"


def build_meta(program: PartProgram | None) -> dict[str, Any] | None:
    """The ``build`` block for a generate response, or None when there is no program."""
    if program is None or not program.operations:
        return None

    program_json: dict[str, Any] = {
        "version": program.version,
        "units": program.metadata.units,
        "name": program.metadata.name,
        "operations": [op.model_dump(mode="json") for op in program.operations],
    }

    build_steps: list[dict[str, Any]] = []
    feature_tags: list[dict[str, Any]] = []
    for i, op in enumerate(program.operations):
        label, kind, dims = _step(op)
        build_steps.append(
            {
                "index": i,
                "op_id": op.op_id,
                "label": label,
                "kind": kind,
                "mode": op.mode.value,
                "affected_dimensions": dims,
            }
        )
        # Tag operations that are meaningful highlight targets: every ADD feature, plus holes/bores.
        if op.mode == OpMode.ADD or isinstance(op, Cylinder):
            feature_tags.append(
                {
                    "op_id": op.op_id,
                    "feature_id": op.op_id,
                    "human_label": _nice(op.op_id).title(),
                    "affected_dimensions": dims,
                    "preview_mode": _preview_mode(op),
                }
            )

    return {"program": program_json, "build_steps": build_steps, "feature_tags": feature_tags}


__all__ = ["build_meta"]
