"""Tier 3: printability. Oriented per metadata.print_orientation.

Mesh-based overhang (ADR-0018): we read the STL per-triangle, not the B-rep
planar faces. Every triangle whose outward normal points downward beyond the
overhang limit (and which sits above the first layer) is flagged; flagged
triangles are clustered into connected regions and a region counts as an
overhang unless it is a small, two-sided, bridgeable hole.

This is geometry-agnostic: it catches curved / lofted / swept overhangs the old
planar check was blind to (the spike's 145 mm^2 swept-tube underside false-passed
before). It stays byte-identical on prismatic parts -- a flat ceiling tessellates
into downward triangles that cluster into one large region (flagged), and small
horizontal hole ceilings (bench-001's M5 bores) are exempted as bridgeable.

THE BRIDGE EXEMPTION IS DELIBERATELY NARROW (the no-false-pass rule, ADR-0018): a
region is exempted ONLY if it is (1) clearly CONCAVE (a channel/hole, not a flat
ceiling or a convex boss underside), (2) NARROW -- major horizontal span <= the
FDM bridge limit, and (3) a genuine TWO-SIDED bridge -- solid material anchors
both ends of the span. A flat, convex, wide, or one-sided (cantilever) region is
never exempted. Bore-ceiling area scales with wall thickness (16 mm^2 at a 4 mm
wall, 24 mm^2 at 6 mm), so an area gate alone cannot separate bores from real
overhangs; the bridge test keys on span + shape, which do not.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from trimesh import graph

from forge_mind.dsl.program import PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.report import Check, CheckStatus, Fix, Locator, Tier

# A downward region smaller than this (mm^2) is ignored as printability noise
# (mesh slivers, tiny chamfer undersides). The NEMA17 ceiling failure is ~1000 mm^2.
MIN_OVERHANG_AREA_MM2 = 20.0

# Triangles whose centroid sits within this distance of the bed (lowest geometry
# along the build direction) rest on the plate and need no support.
BED_CONTACT_TOL_MM = 0.5

# Bridge exemption (all three required; conservative -- over-flagging is a safe
# false-FAIL, a missed overhang is the forbidden false-PASS).
BRIDGE_LIMIT_MM = 10.0  # max horizontal span FDM bridges without support
CONCAVITY_MIN_MM = 0.3  # region must be clearly concave (flat reads ~0, holes >> 1)
PROBE_MARGIN_MM = 0.8  # step beyond the region edge to land in flanking material
PROBE_DEPTH_MM = 0.5  # drop below the ceiling so the anchor probe is inside material


def _unit(v: Vec3) -> np.ndarray:
    a = np.array([v.x, v.y, v.z], dtype=float)
    n = float(np.linalg.norm(a))
    return np.array([0.0, 0.0, 1.0]) if n == 0.0 else a / n


def _horizontal_axes(centers: np.ndarray, up: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Principal horizontal axes of the region with the extent (max-min) along each."""
    horiz = centers - np.outer(centers @ up, up)
    hc = horiz - horiz.mean(axis=0)
    if len(hc) < 2:
        return []
    _u, _s, vh = np.linalg.svd(hc, full_matrices=False)
    axes: list[tuple[np.ndarray, float]] = []
    for k in range(min(2, len(vh))):
        e = vh[k]
        proj = hc @ e
        axes.append((e, float(proj.max() - proj.min())))
    return axes


def _concavity_mm(centers: np.ndarray, normals: np.ndarray) -> float:
    """How far ahead (along each face normal) the furthest region point lies.
    > 0 means the region curves toward its normals = concave (a channel/hole);
    ~ 0 for a flat face; < 0 for a convex boss underside."""
    return float(
        np.median([np.max((centers - centers[i]) @ normals[i]) for i in range(len(centers))])
    )


def _is_two_sided_bridge(
    mesh: trimesh.Trimesh, centers: np.ndarray, up: np.ndarray, axes: list[tuple[np.ndarray, float]]
) -> bool:
    """True iff some horizontal axis has SOLID material anchoring BOTH ends of the
    span -- a genuine bridge. A one-sided cantilever (open on a side) anchors no
    axis. Needs a watertight mesh for containment; otherwise conservatively False."""
    if not mesh.is_watertight or not axes:
        return False
    ceiling_z = float((centers @ up).max())
    centroid = centers.mean(axis=0)
    base = centroid - (centroid @ up) * up + (ceiling_z - PROBE_DEPTH_MM) * up
    for e, ext in axes:
        half = ext / 2.0 + PROBE_MARGIN_MM
        probes = np.array([base + half * e, base - half * e])
        if bool(np.asarray(mesh.contains(probes)).all()):
            return True
    return False


def _overhang_regions(
    mesh: trimesh.Trimesh, up: np.ndarray, max_overhang_deg: float
) -> tuple[list[tuple[float, float]], float, tuple[Vec3, Vec3] | None]:
    """Return (offender_regions, worst_angle_deg, worst_region_bbox).

    offender_regions is a list of (area_mm2, worst_angle_deg) for each connected
    cluster of overhanging triangles whose area >= MIN_OVERHANG_AREA_MM2 and which
    is NOT a small two-sided bridgeable hole. worst_angle_deg is the steepest
    downward angle anywhere (for the PASS note). worst_region_bbox is the axis-aligned
    bounding box (in the mesh's Z-up mm frame) of the LARGEST-area offender region, so the
    viewer can box exactly where the worst overhang is -- None when there are no offenders."""
    face_normals = np.asarray(mesh.face_normals)
    if face_normals.size == 0:
        return [], 0.0, None
    centers = np.asarray(mesh.triangles_center)
    down = -(face_normals @ up)  # >0 means the triangle faces downward
    proj = centers @ up
    bed = float(proj.min())
    worst_angle = math.degrees(math.asin(min(1.0, max(0.0, float(down.max())))))

    sin_max = math.sin(math.radians(max_overhang_deg))
    flagged = (down > sin_max + 1e-9) & ((proj - bed) > BED_CONTACT_TOL_MM)
    idx = np.nonzero(flagged)[0]
    if idx.size == 0:
        return [], worst_angle, None

    # Cluster flagged triangles into connected regions via mesh face adjacency.
    adj = np.asarray(mesh.face_adjacency)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[idx] = True
    if adj.size:
        both = mask[adj[:, 0]] & mask[adj[:, 1]]
        edges = adj[both]
    else:
        edges = np.empty((0, 2), dtype=int)
    comps = graph.connected_components(edges, nodes=idx)

    areas = np.asarray(mesh.area_faces)
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)
    offenders: list[tuple[float, float]] = []
    worst_area = 0.0
    worst_bbox: tuple[Vec3, Vec3] | None = None
    for comp in comps:
        comp = np.asarray(comp)
        region_area = float(areas[comp].sum())
        if region_area < MIN_OVERHANG_AREA_MM2:
            continue
        c_centers = centers[comp]
        region_angle = math.degrees(math.asin(min(1.0, float(down[comp].max()))))
        if _is_bridgeable_hole(mesh, c_centers, face_normals[comp], up):
            continue
        offenders.append((region_area, region_angle))
        if region_area > worst_area:
            # The enclosing box of the region's actual triangle vertices (tighter than centers).
            verts = vertices[faces[comp].reshape(-1)]
            lo = verts.min(axis=0)
            hi = verts.max(axis=0)
            worst_area = region_area
            worst_bbox = (
                Vec3(x=float(lo[0]), y=float(lo[1]), z=float(lo[2])),
                Vec3(x=float(hi[0]), y=float(hi[1]), z=float(hi[2])),
            )
    return offenders, worst_angle, worst_bbox


def _is_bridgeable_hole(
    mesh: trimesh.Trimesh, centers: np.ndarray, normals: np.ndarray, up: np.ndarray
) -> bool:
    """A region is an exempt bridgeable hole ONLY if all three hold (ADR-0018):
    concave, narrow (major span <= bridge limit), and a two-sided bridge."""
    if _concavity_mm(centers, normals) <= CONCAVITY_MIN_MM:
        return False  # flat ceiling or convex boss underside -> real overhang
    axes = _horizontal_axes(centers, up)
    major_span = max((ext for _e, ext in axes), default=0.0)
    if major_span > BRIDGE_LIMIT_MM:
        return False  # too wide to bridge (e.g. the arched swept tube)
    return _is_two_sided_bridge(mesh, centers, up, axes)


def run_printability_checks(stl_path: Path, program: PartProgram) -> tuple[list[Check], list[Fix]]:
    up = _unit(program.metadata.print_orientation)
    max_overhang = program.constraints.max_overhang_deg

    mesh = trimesh.load(str(stl_path), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)

    offenders, worst_angle, worst_bbox = _overhang_regions(mesh, up, max_overhang)

    if not offenders:
        checks = [
            Check(
                id="printability.overhang_max_angle",
                tier=Tier.PRINTABILITY,
                status=CheckStatus.PASS,
                message="No unsupported overhangs beyond the limit.",
                expected=f"<= {max_overhang:.0f} deg",
                actual=f"worst {worst_angle:.0f} deg (supported, bridgeable, or below threshold)",
            )
        ]
        return checks, []

    total_area = sum(a for a, _ in offenders)
    worst = max(offenders, key=lambda o: o[0])
    checks = [
        Check(
            id="printability.overhang_max_angle",
            tier=Tier.PRINTABILITY,
            status=CheckStatus.FAIL,
            message=(
                f"{len(offenders)} unsupported overhang region(s), total {total_area:.0f} mm^2. "
                f"Largest: {worst[0]:.0f} mm^2 at {worst[1]:.0f} deg."
            ),
            expected=f"<= {max_overhang:.0f} deg",
            actual=f"{worst[1]:.0f} deg",
            # Box the largest offending region so the viewer's overhang lens can fly there (Z-up STL).
            locator=(
                Locator(bbox_min=worst_bbox[0], bbox_max=worst_bbox[1], lens="overhang")
                if worst_bbox is not None
                else None
            ),
        )
    ]
    fixes = [
        Fix(
            op_id=None,
            field="metadata.print_orientation",
            reason=(
                f"A {worst[0]:.0f} mm^2 region overhangs at {worst[1]:.0f} deg in the current "
                "print orientation. Reorient the part or add support-free geometry."
            ),
            suggested_change={
                "hint": "choose an orientation that puts large flat faces on the bed"
            },
        )
    ]
    return checks, fixes


__all__ = ["run_printability_checks"]
