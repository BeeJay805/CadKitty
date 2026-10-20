"""ADR-0018 Step-1 fork de-risk (READ-ONLY, $0): the area gate alone does not reconcile.

bench-001 at min_wall=6 false-FAILs: the horizontal M5 bore CEILING arc scales with wall
length (16 mm^2 at 4 mm -> ~24 mm^2 at 6 mm), crossing the 20 mm^2 area gate, even though a
5 mm horizontal hole bridges fine. This probe characterizes every flagged region by area,
horizontal span, and CONCAVITY, to test whether "concave AND minor horizontal span <= bridge
limit -> unflag" cleanly separates bridgeable hole-ceilings from real overhangs WITHOUT ever
unflagging a real (flat/convex) overhang (the FORBIDDEN false-pass).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import Box as BBox3
from build123d import Circle, Plane, Pos, Spline, export_stl, sweep
from trimesh import graph

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.layout.spec import LBracketMotorMountSpec  # type: ignore

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures" / "handwritten"
LSPEC = REPO / "tests" / "fixtures" / "layout" / "bench001_l_bracket.layoutspec.json"
WORK = Path(tempfile.mkdtemp(prefix="spanrule_"))
UP = np.array([0.0, 0.0, 1.0])
AREA_GATE = 20.0
BED_TOL = 0.5
BRIDGE_LIMIT = 10.0


def _regions(
    mesh: trimesh.Trimesh, up: np.ndarray = UP, max_deg: float = 45.0
) -> list[dict[str, Any]]:
    fn = np.asarray(mesh.face_normals)
    down = -(fn @ up)
    cen = np.asarray(mesh.triangles_center)
    proj = cen @ up
    bed = float(proj.min())
    flagged = (down > math.sin(math.radians(max_deg)) + 1e-9) & ((proj - bed) > BED_TOL)
    idx = np.nonzero(flagged)[0]
    if idx.size == 0:
        return []
    adj = np.asarray(mesh.face_adjacency)
    m = np.zeros(len(mesh.faces), bool)
    m[idx] = True
    edges = adj[m[adj[:, 0]] & m[adj[:, 1]]] if adj.size else np.empty((0, 2), int)
    areas = np.asarray(mesh.area_faces)
    out = []
    for comp in graph.connected_components(edges, nodes=idx):
        c = np.asarray(comp)
        area = float(areas[c].sum())
        if area < AREA_GATE:
            continue
        p = cen[c]
        n = fn[c]
        # horizontal span: PCA on the in-plane projection of triangle centers.
        ph = p - np.outer(p @ up, up)
        ph = ph - ph.mean(axis=0)
        if len(ph) >= 2:
            _, s, _ = np.linalg.svd(ph, full_matrices=False)
            # span ~ 2*sqrt(3)*std along each principal axis (uniform-ish strip).
            spans = sorted(2.0 * math.sqrt(3.0) * (sv / math.sqrt(len(ph))) for sv in s[:2])
            minor, major = (spans[0], spans[-1]) if len(spans) >= 2 else (spans[0], spans[0])
        else:
            minor = major = 0.0
        # concavity: for each face, how far ahead (along its normal) the furthest region
        # point lies. >0 region = concave (channel/hole); <=0 = flat/convex (real overhang).
        conc = float(np.median([np.max((p - p[i]) @ n[i]) for i in range(len(c))]))
        out.append({"area": area, "minor": minor, "major": major, "concavity": conc})
    return out


def _show(label: str, mesh: trimesh.Trimesh, up: np.ndarray = UP) -> None:
    rs = _regions(mesh, up)
    print(f"\n[{label}]  {len(rs)} region(s) >= {AREA_GATE} mm^2")
    for r in rs:
        concave = r["concavity"] > 0.05
        bridgeable = (
            concave and r["major"] <= BRIDGE_LIMIT
        )  # MAJOR span (minor falsely passed the tube)
        verdict = "UNFLAG (bridgeable hole)" if bridgeable else "FLAG (real overhang)"
        print(
            f"   area={r['area']:6.1f}  minorSpan={r['minor']:5.1f}  majorSpan={r['major']:6.1f}"
            f"  concavity={r['concavity']:+.2f}  -> {verdict}"
        )


def main() -> None:
    base = LBracketMotorMountSpec.model_validate(json.loads(LSPEC.read_text()))
    for mw in (4.0, 6.0):
        spec = base.model_copy(update={"base": None, "upright": None, "min_wall_mm": mw})
        res = run_layout_pipeline(spec, WORK / f"lb{mw}")
        assert res.model is not None, res.error
        _show(
            f"bench-001 L-bracket min_wall={mw}",
            trimesh.load(str(res.model.stl_path), force="mesh"),
        )

    prog = PartProgram.model_validate(
        json.loads((FIX / "nema17_mount_broken_overhang.partdsl.json").read_text())
    )
    model = compile_program(prog, WORK / "broken_overhang")
    o = prog.metadata.print_orientation
    up = np.array([o.x, o.y, o.z], float)
    up = up / (np.linalg.norm(up) or 1.0)
    _show(
        "nema17_mount_broken_overhang (real overhang, must FLAG)",
        trimesh.load(str(model.stl_path), force="mesh"),
        up,
    )

    # A flat 90-deg ledge overhang (wide top on a narrow stem): planar, must FLAG.
    ledge = (Pos(0, 0, 0) * BBox3(8, 8, 10)) + (Pos(0, 0, 6) * BBox3(30, 30, 4))
    stl_l = WORK / "ledge.stl"
    export_stl(ledge, str(stl_l))
    _show(
        "flat ledge cantilever (planar 90deg overhang, must FLAG)",
        trimesh.load(str(stl_l), force="mesh"),
    )

    path = Spline((0, 0, 0), (0, 0, 15), (15, 0, 25), (30, 0, 30), tangents=((0, 0, 1), (1, 0, 0)))
    tube = sweep(Plane.XY * Circle(4), path=path)
    stl = WORK / "tube.stl"
    export_stl(tube, str(stl))
    _show(
        "spike spline_sweep_tube (real curved overhang, must FLAG)",
        trimesh.load(str(stl), force="mesh"),
    )


if __name__ == "__main__":
    main()
