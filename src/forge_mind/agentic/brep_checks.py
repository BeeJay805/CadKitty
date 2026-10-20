"""B-rep geometry checks on the OCP solid (idioms adapted from text-to-cad's cadpy, MIT).

These run on the editable STEP B-rep the validator already loads, and are more authoritative
than tessellated-mesh heuristics:

  - ``brep_volume``      exact mass-properties volume (BRepGProp), tessellation-independent --
                         a truer signal for the degeneracy gate than trimesh's mesh volume.
  - ``edge_adjacency``   counts NAKED edges (exactly one adjacent face -> an open boundary, the
                         B-rep is not a sealed solid) and NON-MANIFOLD edges (>2 adjacent faces),
                         and localizes them -- catches leaky/unsewn solids a watertight mesh hides.
  - ``is_valid``         the canonical OCC validity check (BRepCheck_Analyzer): catches
                         self-intersections and invalid faces (e.g. from an over-large fillet).

Seam edges (on cylinders / revolves / lofts) are excluded so round closed solids are not
falsely flagged as open. All checks are best-effort: an unreadable part yields a neutral
report (no false FAIL), so the mesh-based gates still apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BBox6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class BRepReport:
    """Outcome of the B-rep checks. Neutral (valid, 0 defects, None volume) when uncheckable."""

    valid: bool
    invalid_reason: str | None
    volume_mm3: float | None
    open_edges: int
    nonmanifold_edges: int
    defect_bbox: BBox6 | None  # combined bbox of naked/non-manifold edges (localization)

    @property
    def open_or_nonmanifold(self) -> bool:
        return self.open_edges > 0 or self.nonmanifold_edges > 0


def brep_volume(part: Any) -> float | None:
    """Exact mass-properties volume (mm^3, absolute) of the B-rep solid. None on failure."""
    if part is None:
        return None
    try:
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps

        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(part.wrapped, props)
        return abs(float(props.Mass()))
    except Exception:
        return None


def is_valid(part: Any) -> tuple[bool, str | None]:
    """OCC BRepCheck validity: a geometrically valid solid (no self-intersection / bad faces).

    Returns (True, None) when uncheckable -- we never invent a FAIL we could not verify.
    """
    if part is None:
        return True, None
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer

        ok = bool(BRepCheck_Analyzer(part.wrapped).IsValid())
        return ok, (None if ok else "OCC BRepCheck reports an invalid solid")
    except Exception:
        return True, None


def _edge_bbox(edge: Any) -> BBox6 | None:
    try:
        from build123d import Edge

        bb = Edge(edge).bounding_box()
        return (
            float(bb.min.X),
            float(bb.min.Y),
            float(bb.min.Z),
            float(bb.max.X),
            float(bb.max.Y),
            float(bb.max.Z),
        )
    except Exception:
        return None


def _union_bbox(a: BBox6 | None, b: BBox6 | None) -> BBox6 | None:
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
        max(a[4], b[4]),
        max(a[5], b[5]),
    )


def edge_adjacency(part: Any) -> tuple[int, int, BBox6 | None]:
    """Count naked (1 face) + non-manifold (>2 faces) edges and localize them.

    Seam edges are skipped: a closed cylinder/revolve/loft has seam edges that would otherwise
    look naked. DEGENERATE edges (sphere poles, cone/revolve apexes -- parameter-space edges
    with zero 3D length) are skipped for the same reason: they have exactly one adjacent face
    BY DEFINITION and cannot be a physical opening, but counting them made ANY part containing
    a Sphere or an on-axis revolve apex permanently "not sealed" (measured: Sphere(5) alone
    reported 2 naked edges and hard-gated `passed` -- the hook_coat_double false-FAIL, engine
    deep review 2026-07-04). A real opening always has a boundary curve of nonzero length, and
    the mesh watertight check remains the backstop. Returns (open, nonmanifold, defect_bbox).
    """
    if part is None:
        return 0, 0, None
    try:
        from OCP.BRep import BRep_Tool
        from OCP.ShapeAnalysis import ShapeAnalysis_Edge
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCP.TopExp import TopExp
        from OCP.TopoDS import TopoDS
        from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    except Exception:
        return 0, 0, None

    try:
        edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(part.wrapped, TopAbs_EDGE, TopAbs_FACE, edge_faces)
        sae = ShapeAnalysis_Edge()
        open_e = nonman = 0
        defect_bbox: BBox6 | None = None
        for i in range(1, edge_faces.Extent() + 1):
            edge = edge_faces.FindKey(i)
            faces = edge_faces.FindFromIndex(i)
            n = faces.Extent()
            if n == 2:  # the manifold-solid case (the vast majority of edges) -- fast path
                continue
            try:
                if bool(BRep_Tool.Degenerated_s(TopoDS.Edge_s(edge))):
                    continue  # a pole/apex artifact, not an open boundary
            except Exception:
                pass  # cannot tell -> keep the edge in the count (fail toward flagging)
            is_seam = False
            try:
                is_seam = bool(sae.IsSeam(edge, faces.First()))
            except Exception:
                is_seam = False
            if is_seam:
                continue
            if n == 1:
                open_e += 1
                defect_bbox = _union_bbox(defect_bbox, _edge_bbox(edge))
            elif n > 2:
                nonman += 1
                defect_bbox = _union_bbox(defect_bbox, _edge_bbox(edge))
        return open_e, nonman, defect_bbox
    except Exception:
        return 0, 0, None


def brep_report(part: Any) -> BRepReport:
    """Run every B-rep check once. Best-effort: a None / unreadable part gives a neutral report."""
    valid, reason = is_valid(part)
    open_e, nonman, defect_bbox = edge_adjacency(part)
    return BRepReport(
        valid=valid,
        invalid_reason=reason,
        volume_mm3=brep_volume(part),
        open_edges=open_e,
        nonmanifold_edges=nonman,
        defect_bbox=defect_bbox,
    )


__all__ = ["BRepReport", "brep_report", "brep_volume", "edge_adjacency", "is_valid"]
