"""ADR-0018 Step-0 RECONCILIATION PROBE (deterministic, $0, READ-ONLY).

Before touching the validator, prove the proposed mesh per-triangle overhang check can
REPLACE the planar B-rep check byte-identically on prismatic parts AND catch the curved
overhang the planar check false-passes. Runs a CANDIDATE mesh-overhang helper (the code
that will move into printability.py) alongside the live planar check on:

  * every hand-authored prismatic fixture (incl. bench-001's horizontal M5 bores),
  * every feasible catalog family (the 102-family motor/extrusion multiplier),
  * the spike's computed solids (the swept tube is the 145 mm^2 false-pass).

GATE A: mesh status == planar status on EVERY prismatic part (no divergence).
GATE B: the swept tube flips PASS -> FAIL (the false-pass is gone).
If a bore region >= the area gate, or a real overhang < the gate, the area gate cannot
reconcile -> STOP (or switch to the span rule).

Run: uv run python scripts/derisk_mesh_overhang.py
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import (
    Circle,
    Plane,
    Polygon,
    Pos,
    Rectangle,
    Spline,
    export_step,
    export_stl,
    extrude,
    loft,
    sweep,
)
from trimesh import graph

from forge_mind.compiler.compile import CADModel, compile_program
from forge_mind.compiler.topology_tags import TopologyIndex
from forge_mind.dsl.primitives import Box as DslBox
from forge_mind.dsl.primitives import OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.scaling.enumerate import build_spec, feasible_shapes
from forge_mind.types import BBox, Vec3
from forge_mind.validator.run import validate

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures" / "handwritten"
WORK = Path(tempfile.mkdtemp(prefix="derisk_overhang_"))

# Candidate gate constants (mirror printability.py).
AREA_GATE_MM2 = 20.0
BED_TOL_MM = 0.5


def _mesh_overhang_regions(
    mesh: trimesh.Trimesh, up: tuple[float, float, float], max_overhang_deg: float
) -> tuple[list[float], float]:
    """Candidate mesh overhang: per-triangle downward test above the bed, connected
    flagged triangles clustered, region areas returned (mm^2). Also returns worst angle."""
    u = np.asarray(up, dtype=float)
    u = u / (np.linalg.norm(u) or 1.0)
    fn = np.asarray(mesh.face_normals)
    down = -(fn @ u)  # >0 downward-facing
    proj = np.asarray(mesh.triangles_center) @ u
    bed = float(proj.min())
    worst = math.degrees(math.asin(min(1.0, max(0.0, float(down.max()))))) if down.size else 0.0

    sin_max = math.sin(math.radians(max_overhang_deg))
    flagged = (down > sin_max + 1e-9) & ((proj - bed) > BED_TOL_MM)
    idx = np.nonzero(flagged)[0]
    if idx.size == 0:
        return [], worst

    adj = np.asarray(mesh.face_adjacency)
    m = np.zeros(len(mesh.faces), dtype=bool)
    m[idx] = True
    both = m[adj[:, 0]] & m[adj[:, 1]] if adj.size else np.zeros(0, dtype=bool)
    edges = adj[both] if adj.size else np.empty((0, 2), dtype=int)
    comps = graph.connected_components(edges, nodes=idx) if idx.size else []
    areas = [float(np.asarray(mesh.area_faces)[np.asarray(c)].sum()) for c in comps]
    return sorted(areas, reverse=True), worst


def _planar_status(report: Any) -> str:
    for c in report.checks:
        if c.id == "printability.overhang_max_angle":
            return c.status.value
    return "-"


def _minwall_status(report: Any) -> str:
    for c in report.checks:
        if c.id == "features.min_wall_thickness":
            return c.status.value
    return "-"


def _up(program: PartProgram) -> tuple[float, float, float]:
    o = program.metadata.print_orientation
    return (o.x, o.y, o.z)


def _evaluate(
    name: str, model: CADModel, program: PartProgram, rows: list, divergences: list
) -> None:
    report = validate(model, program)
    planar = _planar_status(report)
    mesh = trimesh.load(str(model.stl_path), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    regions, worst = _mesh_overhang_regions(
        mesh, _up(program), program.constraints.max_overhang_deg
    )
    offenders = [a for a in regions if a >= AREA_GATE_MM2]
    mesh_status = "fail" if offenders else "pass"
    biggest = f"{regions[0]:.1f}" if regions else "-"
    rows.append((name, planar, mesh_status, f"{worst:.0f}", biggest, len(offenders)))
    if planar != mesh_status:
        divergences.append((name, planar, mesh_status, regions))


# --- spike computed solids (for GATE B) -------------------------------------
_DUMMY = DslBox(
    op_id="placeholder", mode=OpMode.ADD, position=Vec3(x=0, y=0, z=0), width=1, depth=1, height=1
)


def _spike_program() -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="spike", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[_DUMMY],
        interfaces=[],
        constraints=Constraints(
            min_wall_thickness=1.0, max_overhang_deg=45.0, max_bbox=Vec3(x=300.0, y=300.0, z=300.0)
        ),
    )


def _spike_model(part: Any, name: str) -> CADModel:
    stl, step = WORK / f"{name}.stl", WORK / f"{name}.step"
    export_stl(part, str(stl))
    export_step(part, str(step))
    idx = TopologyIndex()
    try:
        for i, f in enumerate(part.faces()):
            idx.add(name, f"face[{i}]", f)
    except Exception:
        pass
    bb = part.bounding_box()
    bbox = BBox(
        min=Vec3(x=bb.min.X, y=bb.min.Y, z=bb.min.Z), max=Vec3(x=bb.max.X, y=bb.max.Y, z=bb.max.Z)
    )
    return CADModel(part=part, stl_path=stl, step_path=step, topology_index=idx, bounding_box=bbox)


def main() -> None:
    rows: list = []
    divergences: list = []

    # 1) Hand-authored prismatic fixtures (incl. bench-001 horizontal M5 bores).
    for fp in sorted(FIX.glob("*.partdsl.json")):
        program = PartProgram.model_validate(json.loads(fp.read_text()))
        model = compile_program(program, WORK / fp.stem)
        _evaluate(f"fixture:{fp.stem}", model, program, rows, divergences)

    # 2) Every feasible catalog family (the 102-family multiplier).
    feas = feasible_shapes()
    n_fam = 0
    for i, shape in enumerate(feas):
        spec = build_spec(shape, name=f"fam{i:03d}", envelope_mm=Vec3(x=400.0, y=400.0, z=40.0))
        res = run_layout_pipeline(spec, WORK / f"fam{i:03d}")
        if res.model is None or res.program is None:
            divergences.append((f"family:{shape.key}", "pipeline-error", res.error or "?", []))
            continue
        n_fam += 1
        _evaluate(f"family:{shape.key}", res.model, res.program, rows, divergences)

    # 3) Spike computed solids -> GATE B (the swept tube must flip to FAIL).
    spike_rows: list = []
    trap = extrude(
        Plane.XZ * Polygon((-10, 0), (10, 0), (35, 15), (-35, 15), align=None), amount=15
    )
    loft_s = loft([Pos(0, 0, 0) * Circle(8), Pos(0, 0, 30) * Rectangle(40, 40)])
    path = Spline((0, 0, 0), (0, 0, 15), (15, 0, 25), (30, 0, 30), tangents=((0, 0, 1), (1, 0, 0)))
    tube = sweep(Plane.XY * Circle(4), path=path)
    for nm, part in (
        ("trapezoid_flared", trap),
        ("loft_round_to_square", loft_s),
        ("spline_sweep_tube", tube),
    ):
        model = _spike_model(part, nm)
        prog = _spike_program()
        report = validate(model, prog)
        planar = _planar_status(report)
        mesh = trimesh.load(str(model.stl_path), force="mesh")
        regions, worst = _mesh_overhang_regions(mesh, (0.0, 0.0, 1.0), 45.0)
        offenders = [a for a in regions if a >= AREA_GATE_MM2]
        spike_rows.append(
            (
                nm,
                planar,
                "fail" if offenders else "pass",
                f"{worst:.0f}",
                f"{regions[0]:.1f}" if regions else "-",
                len(offenders),
            )
        )

    # --- report ---
    print(f"\nPRISMATIC PARTS ({len(rows)} total: fixtures + {n_fam} catalog families)")
    print(f"{'name':<52} {'report':>7} {'cand':>6} {'worst':>6} {'bigRgn':>7} {'#off':>4}")
    for name, planar, mesh_status, worst, biggest, noff in rows:
        flag = "  <-- DIVERGES" if planar != mesh_status else ""
        # only print the bench-001 bore case + any divergence to keep output tight
        if name.startswith("fixture:nema17_mount_good") or planar != mesh_status:
            print(
                f"{name:<52} {planar:>7} {mesh_status:>6} {worst:>6} {biggest:>7} {noff:>4}{flag}"
            )

    print("\nSPIKE COMPUTED SOLIDS (GATE B: tube must flip planar=pass -> mesh=fail)")
    print(f"{'name':<52} {'report':>7} {'cand':>6} {'worst':>6} {'bigRgn':>7} {'#off':>4}")
    for name, planar, mesh_status, worst, biggest, noff in spike_rows:
        print(f"{name:<52} {planar:>7} {mesh_status:>6} {worst:>6} {biggest:>7} {noff:>4}")

    # bench-001 bore region detail (the reconciliation risk).
    good = next((r for r in rows if r[0].startswith("fixture:nema17_mount_good")), None)

    print("\n=== VERDICT ===")
    print(
        "(Note: post-Step-1 the live report IS the mesh check, so the 'report' column and the"
        " independent 'cand' helper are both mesh-based; agreement = self-consistency.)"
    )
    n_div = len(divergences)
    print(
        f"GATE A (prismatic byte-identical): {'PASS' if n_div == 0 else 'FAIL'} ({n_div} divergence(s))"
    )
    for name, planar, mesh_status, regions in divergences[:20]:
        rstr = (
            ", ".join(f"{a:.1f}" for a in regions[:6])
            if isinstance(regions, list)
            else str(regions)
        )
        print(f"   DIVERGE {name}: report={planar} cand={mesh_status} regions=[{rstr}]")
    tube_row = next((r for r in spike_rows if r[0] == "spline_sweep_tube"), None)
    gate_b = tube_row is not None and tube_row[2] == "fail"
    print(
        f"GATE B (swept tube flagged, false-pass fixed): {'PASS' if gate_b else 'FAIL'}"
        + (f"  (tube mesh={tube_row[2]}, region={tube_row[4]}mm^2)" if tube_row else "")
    )
    print(
        f"\nReconciliation window: bench-001 biggest bore region = "
        f"{good[4] if good else '?'} mm^2  (must be < {AREA_GATE_MM2}); "
        f"tube overhang region = {tube_row[4] if tube_row else '?'} mm^2 (must be >= {AREA_GATE_MM2})"
    )
    print(f"\n(artifacts in {WORK})")


if __name__ == "__main__":
    main()
