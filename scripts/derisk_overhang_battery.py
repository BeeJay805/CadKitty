"""ADR-0018 Step-1 SAFETY BATTERY de-risk (READ-ONLY, $0): verify each battery solid
trips the right guard BEFORE committing them as tests. Uses the REAL printability
internals so the test solids are known-good.

Required (zero false-PASS):
  swept tube                 -> FLAG (curved/wide)
  flat cantilever ledge      -> FLAG (concavity guard: flat)
  convex boss underside      -> FLAG (concavity guard: convex)
  wide concave bore          -> FLAG (span guard: major > 10mm)
  one-sided concave lip      -> FLAG (two-sided guard: no anchored axis)
  small through-hole ceiling -> EXEMPT/PASS (bridgeable: concave + narrow + two-sided)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import Box, Circle, Cylinder, Plane, Pos, Rot, Spline, export_stl, sweep

from forge_mind.validator import printability as pr

WORK = Path(tempfile.mkdtemp(prefix="battery_"))
UP = np.array([0.0, 0.0, 1.0])


def _report(name: str, part: Any, expect: str) -> None:
    stl = WORK / f"{name}.stl"
    export_stl(part, str(stl))
    mesh = trimesh.load(str(stl), force="mesh")
    offenders, _worst = pr._overhang_regions(mesh, UP, 45.0)
    status = "FLAG" if offenders else "PASS"
    ok = "OK" if status == expect else "*** WRONG ***"
    print(f"\n[{name}] -> {status} (want {expect}) {ok}  watertight={mesh.is_watertight}")
    # per-region introspection
    fn = np.asarray(mesh.face_normals)
    down = -(fn @ UP)
    cen = np.asarray(mesh.triangles_center)
    proj = cen @ UP
    bed = float(proj.min())
    flagged = (down > np.sin(np.radians(45.0)) + 1e-9) & ((proj - bed) > pr.BED_CONTACT_TOL_MM)
    idx = np.nonzero(flagged)[0]
    if idx.size:
        adj = np.asarray(mesh.face_adjacency)
        m = np.zeros(len(mesh.faces), bool)
        m[idx] = True
        edges = adj[m[adj[:, 0]] & m[adj[:, 1]]] if adj.size else np.empty((0, 2), int)
        areas = np.asarray(mesh.area_faces)
        from trimesh import graph

        for comp in graph.connected_components(edges, nodes=idx):
            c = np.asarray(comp)
            a = float(areas[c].sum())
            if a < pr.MIN_OVERHANG_AREA_MM2:
                continue
            cc, nn = cen[c], fn[c]
            conc = pr._concavity_mm(cc, nn)
            axes = pr._horizontal_axes(cc, UP)
            major = max((e for _a, e in axes), default=0.0)
            two = pr._is_two_sided_bridge(mesh, cc, UP, axes)
            exempt = pr._is_bridgeable_hole(mesh, cc, nn, UP)
            print(
                f"   region area={a:6.1f} concavity={conc:+.2f} major={major:5.1f} "
                f"two_sided={two} -> {'EXEMPT' if exempt else 'OFFENDER'}"
            )


def main() -> None:
    # 1. swept tube (curved overhang).
    path = Spline((0, 0, 0), (0, 0, 15), (15, 0, 25), (30, 0, 30), tangents=((0, 0, 1), (1, 0, 0)))
    _report("swept_tube", sweep(Plane.XY * Circle(4), path=path), "FLAG")

    # 2. flat cantilever ledge (wide flat top on a narrow stem).
    ledge = (Pos(0, 0, 0) * Box(8, 8, 12)) + (Pos(0, 0, 8) * Box(30, 30, 4))
    _report("flat_cantilever", ledge, "FLAG")

    # 3. convex boss underside: a solid horizontal half-pin lying above the bed on legs.
    legs = (Pos(-12, 0, 0) * Box(4, 8, 12)) + (Pos(12, 0, 0) * Box(4, 8, 12))
    pin = Pos(0, 0, 14) * Rot(0, 90, 0) * Cylinder(radius=5, height=20)  # axis X, convex underside
    _report("convex_pin", legs + pin, "FLAG")

    # 4. wide concave bore fully enclosed (r=15 -> ceiling span ~30mm > 10mm).
    wide = Box(60, 20, 60) - (Pos(0, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=15, height=20))
    _report("wide_bore", wide, "FLAG")

    # 5. one-sided concave scoop: a transverse groove under a cantilevered bar's free end.
    # Open on the free (+X) end and both Y ends -> NO horizontal axis is anchored both sides.
    wall = Pos(0, 0, 0) * Box(4, 10, 30)
    bar = Pos(9, 0, 20) * Box(18, 10, 4)  # cantilever +X at the top
    onesided = (wall + bar) - (Pos(15, 0, 18.5) * Rot(90, 0, 0) * Cylinder(radius=2.5, height=12))
    _report("one_sided_lip", onesided, "FLAG")

    # 6. small through-hole ceiling (bench-001 essence): r=2.6 hole through a 6mm wall.
    wall2 = Box(40, 6, 30)
    hole = wall2 - (Pos(0, 0, 16) * Rot(90, 0, 0) * Cylinder(radius=2.6, height=8))
    _report("small_through_hole", hole, "PASS")

    print(f"\n(artifacts in {WORK})")


if __name__ == "__main__":
    main()
