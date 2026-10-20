"""Deterministic parametric implicit archetypes (ADR-0026, Phase 7).

Each generator turns a few real parameters (size, cell, wall/strut, in mm) into a watertight mesh via
the SDF field math in ``field.py``. Code derives every coordinate (ADR-0009) -- no LLM authors the
field. These are the long-tail shapes B-rep can't express: TPMS infills and strut lattices.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge_mind.implicit import field as sdf

# Clamp user/parsed parameters into sane, buildable ranges (mm).
_MIN_SIZE, _MAX_SIZE = 8.0, 200.0
_MIN_CELL, _MAX_CELL = 3.0, 60.0
_MIN_FEATURE, _MAX_FEATURE = 0.6, 12.0


@dataclass(frozen=True)
class ImplicitPart:
    """A built implicit part: the mesh plus the honest metadata the result layer surfaces."""

    name: str
    archetype: str
    mesh: object  # trimesh.Trimesh
    step_mm: float
    watertight: bool
    params: dict[str, float]
    note: str


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _choose_step(span: float, feature: float) -> float:
    """Pick a sampling step: fine enough to resolve the thinnest feature (~4 samples across), but
    never so fine the grid blows past the field-module guard."""
    by_feature = feature / 4.0
    by_grid = span / (sdf._MAX_SAMPLES_PER_AXIS - 6)
    return max(by_feature, by_grid, 0.12)


def _build(
    field: sdf.Field, half: tuple[float, float, float], step: float
) -> tuple[object, float, bool]:
    """Mesh ``field`` over a box slightly larger than the part; if not watertight, retry once finer.
    Returns (mesh, step_used, watertight)."""
    margin = 2.0 * step
    lo = (-half[0] - margin, -half[1] - margin, -half[2] - margin)
    hi = (half[0] + margin, half[1] + margin, half[2] + margin)
    mesh = sdf.sdf_to_mesh(field, lo, hi, step)
    if bool(mesh.is_watertight) and float(mesh.volume) > 0.0:
        return mesh, step, True
    finer = step / 1.5
    if sdf.estimate_axis_samples(lo, hi, finer) <= sdf._MAX_SAMPLES_PER_AXIS:
        m2 = sdf.sdf_to_mesh(field, lo, hi, finer)
        if bool(m2.is_watertight) and float(m2.volume) > 0.0:
            return m2, finer, True
        # Fall-through paths must keep the SAME watertight-AND-positive-volume invariant as the
        # passing paths above: a watertight-but-zero/negative-volume mesh is degenerate and must NOT
        # report watertight=True (audit H2 -- the implicit lane's only print-validity gate reads it).
        return m2, finer, bool(m2.is_watertight) and float(m2.volume) > 0.0
    return mesh, step, bool(mesh.is_watertight) and float(mesh.volume) > 0.0


def gyroid_block(
    size_mm: float = 40.0, cell_mm: float = 10.0, wall_mm: float = 1.6
) -> ImplicitPart:
    """A gyroid (TPMS) lattice block: a periodic minimal-surface wall clipped to a cube."""
    size = _clamp(size_mm, _MIN_SIZE, _MAX_SIZE)
    cell = _clamp(cell_mm, _MIN_CELL, _MAX_CELL)
    wall = _clamp(wall_mm, _MIN_FEATURE, _MAX_FEATURE)
    half = (size / 2, size / 2, size / 2)
    fld = sdf.op_intersect(sdf.gyroid(cell, wall), sdf.box(*half))
    step = _choose_step(size, wall)
    mesh, used, wt = _build(fld, half, step)
    return ImplicitPart(
        name=f"Gyroid lattice {size:g}mm (cell {cell:g}, wall {wall:g})",
        archetype="gyroid",
        mesh=mesh,
        step_mm=used,
        watertight=wt,
        params={"size_mm": size, "cell_mm": cell, "wall_mm": wall},
        note="A gyroid TPMS lattice -- an implicit/field shape with no B-rep equivalent. Delivered as "
        "a watertight mesh (STL) for printing; an editable STEP is not available for this geometry.",
    )


def schwarz_block(
    size_mm: float = 40.0, cell_mm: float = 10.0, wall_mm: float = 1.6
) -> ImplicitPart:
    """A Schwarz-P (TPMS) lattice block -- a second minimal-surface family."""
    size = _clamp(size_mm, _MIN_SIZE, _MAX_SIZE)
    cell = _clamp(cell_mm, _MIN_CELL, _MAX_CELL)
    wall = _clamp(wall_mm, _MIN_FEATURE, _MAX_FEATURE)
    half = (size / 2, size / 2, size / 2)
    fld = sdf.op_intersect(sdf.schwarz_p(cell, wall), sdf.box(*half))
    step = _choose_step(size, wall)
    mesh, used, wt = _build(fld, half, step)
    return ImplicitPart(
        name=f"Schwarz-P lattice {size:g}mm (cell {cell:g}, wall {wall:g})",
        archetype="schwarz_p",
        mesh=mesh,
        step_mm=used,
        watertight=wt,
        params={"size_mm": size, "cell_mm": cell, "wall_mm": wall},
        note="A Schwarz-P TPMS lattice -- an implicit/field shape with no B-rep equivalent. Delivered "
        "as a watertight mesh (STL) for printing; an editable STEP is not available for this geometry.",
    )


def strut_lattice_block(
    size_mm: float = 40.0, cell_mm: float = 10.0, strut_mm: float = 2.4
) -> ImplicitPart:
    """A cubic strut lattice block: axis-aligned struts meeting at nodes, clipped to a cube."""
    size = _clamp(size_mm, _MIN_SIZE, _MAX_SIZE)
    cell = _clamp(cell_mm, _MIN_CELL, _MAX_CELL)
    strut = _clamp(strut_mm, _MIN_FEATURE, _MAX_FEATURE)
    half = (size / 2, size / 2, size / 2)
    fld = sdf.op_intersect(sdf.strut_lattice(cell, strut), sdf.box(*half))
    step = _choose_step(size, strut)
    mesh, used, wt = _build(fld, half, step)
    return ImplicitPart(
        name=f"Strut lattice {size:g}mm (cell {cell:g}, strut {strut:g})",
        archetype="strut_lattice",
        mesh=mesh,
        step_mm=used,
        watertight=wt,
        params={"size_mm": size, "cell_mm": cell, "strut_mm": strut},
        note="A cubic strut lattice -- an implicit/field shape. Delivered as a watertight mesh (STL) "
        "for printing; an editable STEP is not available for this geometry type.",
    )


__all__ = [
    "ImplicitPart",
    "gyroid_block",
    "schwarz_block",
    "strut_lattice_block",
]
