"""B-rep feature extraction — the bridge from build123d topology to measurable
features the validator reasons about.

We measure the FINAL compiled body, not the topology_index of intended ops.
That way a hole that got filled, merged, or shifted by a later boolean is seen
as it actually is, not as it was meant to be.

Kept backend-aware (imports build123d) but exposes plain dataclasses so the
tiers and interface library never touch build123d types directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from build123d import GeomType

Vec3T = tuple[float, float, float]


@dataclass(frozen=True)
class CylinderFeature:
    """A cylindrical face of the body — almost always a hole wall.

    ``axis_point`` is a point on the cylinder's axis (the true hole center line),
    ``axis_dir`` is the unit axis direction. ``radius`` is the cylinder radius.

    ``concave`` (ADR-0035 B1) is the measured material side: True = the outward face normal
    points TOWARD the axis, so material surrounds the cylinder — a HOLE wall; False = a BOSS/pin;
    None = could not be measured (never guessed — an unmeasurable face must not read as either).
    ``axial_extent`` is the (min, max) projection of the face's vertices onto the axis — the
    measured depth span — or None when unmeasurable.
    """

    radius: float
    axis_point: Vec3T
    axis_dir: Vec3T
    concave: bool | None = None
    axial_extent: tuple[float, float] | None = None

    def axis_is_parallel_to(self, direction: Vec3T, tol_deg: float = 5.0) -> bool:
        """True if this cylinder's axis is (anti)parallel to ``direction``."""
        return _angle_between(self.axis_dir, direction) <= tol_deg or (
            _angle_between(self.axis_dir, _negate(direction)) <= tol_deg
        )


@dataclass(frozen=True)
class CylinderStack:
    """Coaxial same-radius cylindrical faces merged into ONE physical feature (ADR-0035 B1).

    Booleans routinely split a single hole wall into several faces (seam splits, a cross-cut);
    counting FACES double-counts holes and a counterbore's two diameters must stay distinct.
    A stack groups faces that share an axis line, a radius, and a concavity verdict, and merges
    their axial extents. One stack ~= one hole wall (or one boss).
    """

    radius: float
    axis_point: Vec3T
    axis_dir: Vec3T
    concave: bool | None
    axial_extent: tuple[float, float] | None
    n_faces: int
    # The number of axially-SEPARATED contiguous spans in the stack (default 1). Two clevis
    # EARS drilled on one pivot axis are ONE coaxial stack but TWO physical wall penetrations
    # -- counting stacks alone under-reports them (measured live: a turret's two ear bores
    # audited as 1/2 and the author chased a phantom missing hole). Computed from the member
    # faces' axial extents; unmeasurable extents conservatively collapse to 1 (the old
    # behavior -- never fabricate a separation that was not measured).
    segments: int = 1

    @property
    def depth_mm(self) -> float | None:
        if self.axial_extent is None:
            return None
        return self.axial_extent[1] - self.axial_extent[0]


@dataclass(frozen=True)
class PlaneFeature:
    """A planar face of the body."""

    normal: Vec3T  # unit outward normal
    area: float
    centroid: Vec3T


def _negate(v: Vec3T) -> Vec3T:
    return (-v[0], -v[1], -v[2])


def _normalize(v: Vec3T) -> Vec3T:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n == 0.0:
        return v
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a: Vec3T, b: Vec3T) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _angle_between(a: Vec3T, b: Vec3T) -> float:
    """Angle in degrees between two vectors (each assumed roughly unit)."""
    a, b = _normalize(a), _normalize(b)
    d = max(-1.0, min(1.0, _dot(a, b)))
    return math.degrees(math.acos(d))


def _face_concavity(face: Any, axis: Any) -> bool | None:
    """Measured material side of a cylindrical face: True = hole wall, False = boss, None =
    unmeasurable. The outward normal at a point ON the surface points toward the axis for a hole
    (material outside the cylinder) and away for a boss. Proven on booleaned bodies (a subtracted
    hole reads dot=-r, an added boss dot=+r); any API surprise returns None, never a guess."""
    try:
        p = face.position_at(0.5, 0.5)
        n = face.normal_at(p)
        ap = p - axis.position
        d = axis.direction
        radial = ap - d * (ap.dot(d))
        dot = float(n.dot(radial))
        if abs(dot) < 1e-9:  # degenerate (point on axis / zero normal): unmeasurable
            return None
        return dot < 0
    except Exception:
        return None


def _face_axial_extent(face: Any, axis: Any) -> tuple[float, float] | None:
    """The (min, max) projection of the face's vertices onto the axis direction — the measured
    depth span of this face. None when the face has no vertices to project (unmeasurable)."""
    try:
        d = axis.direction
        projs = [
            float(c.X * d.X + c.Y * d.Y + c.Z * d.Z) for c in (v.center() for v in face.vertices())
        ]
        if not projs:
            return None
        return (min(projs), max(projs))
    except Exception:
        return None


def extract_cylinders(part: Any) -> list[CylinderFeature]:
    """Return every cylindrical face of the body as a CylinderFeature (with the measured
    concavity + axial extent where measurable — ADR-0035 B1; both default None on failure so
    existing callers see exactly the prior fields)."""
    out: list[CylinderFeature] = []
    for face in part.faces():
        if face.geom_type != GeomType.CYLINDER:
            continue
        axis = face.axis_of_rotation
        if axis is None:
            # Some trimmed/seam cylindrical faces have no defined rotation axis
            # (an OCP quirk). They carry no measurable hole axis, so skip them.
            continue
        p, d = axis.position, axis.direction
        out.append(
            CylinderFeature(
                radius=float(face.radius),
                axis_point=(float(p.X), float(p.Y), float(p.Z)),
                axis_dir=_normalize((float(d.X), float(d.Y), float(d.Z))),
                concave=_face_concavity(face, axis),
                axial_extent=_face_axial_extent(face, axis),
            )
        )
    return out


def main_bore_axis(part: Any) -> tuple[Vec3T, Vec3T] | None:
    """The rotation axis of a BORED rotor -- its shaft bore: the concave (hole) cylinder
    stack whose axis lies nearest the part's bounding-box centre. This axis is INDEPENDENT of
    the mass distribution (unlike the inertia principal axis, which trivially "balances" any
    part), so balance measured about it is meaningful. Returns ``(axis_point, axis_dir)``, or
    ``None`` when there is no concave bore -- e.g. a journal-shaft rotor like a crankshaft,
    which spins on a CONVEX journal, so its axis must instead come from intent. Fails CLOSED:
    ``None`` means the caller must not certify balance about a guessed axis.
    """
    bores = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    if not bores:
        return None
    bb = part.bounding_box()
    centre = (
        (float(bb.min.X) + float(bb.max.X)) / 2.0,
        (float(bb.min.Y) + float(bb.max.Y)) / 2.0,
        (float(bb.min.Z) + float(bb.max.Z)) / 2.0,
    )

    def offset_from_centre(stack: CylinderStack) -> float:
        # Perpendicular distance from the bbox centre to the stack's axis LINE.
        p, d = stack.axis_point, _normalize(stack.axis_dir)
        w = (centre[0] - p[0], centre[1] - p[1], centre[2] - p[2])
        along = w[0] * d[0] + w[1] * d[1] + w[2] * d[2]
        perp = (w[0] - along * d[0], w[1] - along * d[1], w[2] - along * d[2])
        return float((perp[0] ** 2 + perp[1] ** 2 + perp[2] ** 2) ** 0.5)

    shaft = min(bores, key=offset_from_centre)
    return shaft.axis_point, _normalize(shaft.axis_dir)


def cluster_coaxial(
    cylinders: list[CylinderFeature],
    *,
    radius_tol: float = 0.05,
    axis_tol_deg: float = 1.0,
    coaxial_tol_mm: float = 0.1,
) -> list[CylinderStack]:
    """Merge coaxial same-radius, same-concavity faces into stacks (one stack ~= one physical
    hole wall or boss). Greedy single-pass: each face joins the first compatible stack —
    parallel axis (within ``axis_tol_deg``), axis lines coincident within ``coaxial_tol_mm``
    (perpendicular distance), radius within ``radius_tol``, same concavity verdict. A
    counterbore's two diameters stay distinct stacks; a seam-split hole wall merges into one."""
    groups: list[list[CylinderFeature]] = []
    for cyl in cylinders:
        placed = False
        for group in groups:
            head = group[0]
            if abs(cyl.radius - head.radius) > radius_tol:
                continue
            if not cyl.axis_is_parallel_to(head.axis_dir, tol_deg=axis_tol_deg):
                continue
            if planar_distance(cyl.axis_point, head.axis_point, head.axis_dir) > coaxial_tol_mm:
                continue
            if cyl.concave != head.concave:
                continue
            group.append(cyl)
            placed = True
            break
        if not placed:
            groups.append([cyl])

    stacks: list[CylinderStack] = []
    for group in groups:
        head = group[0]
        # Canonicalize every member's extent onto the HEAD's axis direction before merging.
        # Grouping accepts ANTI-parallel axes (correct: one hole wall can tessellate into
        # opposed-normal faces), but each face's extent is a projection onto its OWN axis --
        # so a 180-degree pair of radial set-screw holes read as the SAME interval and
        # collapsed to one segment, making the demanded "2 holes" unmeasurable-by-construction
        # (mech_star, engine deep review 2026-07-04). Along one canonical direction the pair
        # becomes two disjoint spans = two penetrations.
        hd = _normalize(head.axis_dir)
        extents: list[tuple[float, float] | None] = []
        for c in group:
            e = c.axial_extent
            if e is None:
                extents.append(None)
            elif _dot(_normalize(c.axis_dir), hd) < 0.0:
                extents.append((-e[1], -e[0]))
            else:
                extents.append(e)
        merged: tuple[float, float] | None = None
        segments = 1
        if all(e is not None for e in extents):
            known = sorted(e for e in extents if e is not None)
            merged = (min(e[0] for e in known), max(e[1] for e in known))
            # Axially-separated spans = distinct wall penetrations on one axis (clevis
            # ears). Merge overlapping/near-touching spans (a seam split has ~0 gap; 1mm
            # absorbs tessellation noise); every REAL air gap starts a new segment.
            segments = 0
            span_end: float | None = None
            for lo, hi in known:
                if span_end is None or lo > span_end + 1.0:
                    segments += 1
                    span_end = hi
                else:
                    span_end = max(span_end, hi)
        stacks.append(
            CylinderStack(
                radius=head.radius,
                axis_point=head.axis_point,
                axis_dir=head.axis_dir,
                concave=head.concave,
                axial_extent=merged,
                n_faces=len(group),
                segments=max(1, segments),
            )
        )
    return stacks


def stack_pierces(part: Any, stack: CylinderStack, probe_mm: float = 0.2) -> bool | None:
    """Does this bore go THROUGH the body? (ADR-0082, closing the ADR-0054 through-ness escape.)

    Probes the two points on the stack's centerline just BEYOND its measured axial ends: a
    through bore is open at both ends (both probes land in air), a blind pocket has floor
    material beyond one end (a probe lands INSIDE the solid). Local by construction -- no
    body-span estimate, so stepped bodies and multi-thickness parts cannot false-FAIL.

    Returns True (open both ends), False (a floor was measured), or None (unmeasurable --
    missing extent, degenerate axis, or classifier failure). Callers must treat None as
    "no claim", never as blind: the measured-join stays fail-open on doubt."""
    if stack.axial_extent is None:
        return None
    try:
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_State

        d = _normalize(stack.axis_dir)
        if d == (0.0, 0.0, 0.0):
            return None
        ap = stack.axis_point
        t0 = _dot(ap, d)  # the axis point's own parameter along the axis direction
        lo, hi = stack.axial_extent
        shape = getattr(part, "wrapped", part)

        def _inside(t: float) -> bool | None:
            p = (
                ap[0] + (t - t0) * d[0],
                ap[1] + (t - t0) * d[1],
                ap[2] + (t - t0) * d[2],
            )
            c = BRepClass3d_SolidClassifier(shape, gp_Pnt(*p), 1e-6)
            state = c.State()
            if state == TopAbs_State.TopAbs_IN:
                return True
            if state == TopAbs_State.TopAbs_OUT:
                return False
            return None  # ON / UNKNOWN: a face graze -- no claim

        below = _inside(lo - probe_mm)
        above = _inside(hi + probe_mm)
        if below is None or above is None:
            return None
        # material beyond an end is a floor -- the bore pierces only when BOTH probes are in air
        return not (below or above)
    except Exception:
        return None


def extract_planes(part: Any) -> list[PlaneFeature]:
    """Return every planar face of the body as a PlaneFeature."""
    out: list[PlaneFeature] = []
    for face in part.faces():
        if face.geom_type != GeomType.PLANE:
            continue
        n = face.normal_at()
        c = face.center()
        out.append(
            PlaneFeature(
                normal=_normalize((float(n.X), float(n.Y), float(n.Z))),
                area=float(face.area),
                centroid=(float(c.X), float(c.Y), float(c.Z)),
            )
        )
    return out


def project_point_onto_plane(point: Vec3T, axis_dir: Vec3T) -> Vec3T:
    """Drop the component of ``point`` along ``axis_dir`` (for comparing hole
    centers within the plane of a face, ignoring depth along the axis).
    """
    d = _normalize(axis_dir)
    along = _dot(point, d)
    return (point[0] - along * d[0], point[1] - along * d[1], point[2] - along * d[2])


def planar_distance(a: Vec3T, b: Vec3T, axis_dir: Vec3T) -> float:
    """Euclidean distance between two points after removing the ``axis_dir``
    component from both — i.e. distance within the plane perpendicular to the
    axis. Used to compare bolt-pattern spacing regardless of hole depth.
    """
    pa = project_point_onto_plane(a, axis_dir)
    pb = project_point_onto_plane(b, axis_dir)
    return math.dist(pa, pb)


__all__ = [
    "CylinderFeature",
    "CylinderStack",
    "PlaneFeature",
    "cluster_coaxial",
    "extract_cylinders",
    "extract_planes",
    "planar_distance",
]
