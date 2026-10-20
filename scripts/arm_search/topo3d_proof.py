"""The committed 3-D generative proof (T5 / gap c): design a bracket FOR a corner point load,
then MEASURE it under that same corner load -- not a face-spread.

The reproducibility fix Fable flagged: the 3-D bracket work had no committed driver, and the
one number it did produce measured a CORNER-designed part under a FACE-SPREAD load, which
averages the torsion away and reports an over-stiff (misaligned) case -- the same failure the
row-76 2-D winner's absurd 33x margin exposed (the autotuner bisecting a case the part was not
designed for). With the ``load_patch`` case (ADR added in T5) the measured case IS the design
case.

Flow: a z-major Domain3D (base fixed at z=0, a transverse point load at a TOP CORNER, offset
from the centroid so the design carries torsion) -> optimize3d (3-D SIMP) -> density3d_to_mesh
(watertight B-rep) -> run_load_case measured BOTH ways (face-spread vs the corner patch = the
DESIGN case), same total force. The part is built base-down/Z-up so no reorientation is needed
and the patch point is exactly the loaded corner.

Usage:
  FORGE_FEA_TIMEOUT_S=600 uv run --no-sync python scripts/arm_search/topo3d_proof.py \
    --out C:/Users/brenn/work/arm_m1_runs/topo3d_proof
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from forge_mind.structures import mesh3d, topopt3d
from forge_mind.structures.loadcase import LoadCase, run_load_case

# The bracket benchmark (declared, printed on the report). A slender vertical member so the
# corner load produces real bending + torsion; modest grid so the proof runs in minutes.
NELX, NELY, NELZ = 10, 10, 30
CELL_MM = (2.4, 2.4, 2.4)  # part ~ 24 x 24 x 72 mm
PAYLOAD_N = 4.905  # 0.5 kg
VOLFRAC = 0.3


def corner_bracket(volfrac: float) -> topopt3d.Domain3D:
    """z-major cantilever: the z=0 face fully fixed, a -Y transverse point load at the TOP
    (iz=nelz) corner offset in +X from the centroid -- so the stiffest structure must carry
    the torsion, not just bending."""
    dom = topopt3d.Domain3D(nelx=NELX, nely=NELY, nelz=NELZ, volfrac=volfrac, rmin=1.4)
    dom.fixed_dofs = [
        dom.dof(ix, iy, 0, c) for ix in range(NELX + 1) for iy in range(NELY + 1) for c in (0, 1, 2)
    ]
    dom.load_dofs = [dom.dof(NELX, NELY, NELZ, 1)]  # top (x-max, y-max) corner, -Y
    dom.load_vals = [-1.0]
    return dom


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Committed 3-D corner-load generative proof.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--patch-r", type=float, default=6.0, help="corner patch radius (mm)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "benchmark": {
            "grid": [NELX, NELY, NELZ],
            "cell_mm": list(CELL_MM),
            "payload_n": PAYLOAD_N,
            "design_case": "transverse -Y point load at the top (x-max, y-max) corner",
        }
    }

    dom = corner_bracket(VOLFRAC)
    res = topopt3d.optimize3d(dom, max_iter=args.max_iter)
    report["simp"] = {
        "compliance_first": round(float(res["compliance"][0]), 4),
        "compliance_last": round(float(res["compliance"][-1]), 4),
        "iters": len(res["compliance"]),
    }
    meshed = mesh3d.density3d_to_mesh(res["density"], out_dir=out_dir, cell_mm=CELL_MM)
    if meshed is None:
        report["verdict"] = "mesh_failed_no_surface"
        (out_dir / "topo3d_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    import trimesh

    mesh: Any = trimesh.load(str(meshed.stl), force="mesh")
    xmax = float(mesh.bounds[1][0])
    ymax = float(mesh.bounds[1][1])
    report["mesh"] = {
        "stl": str(meshed.stl),
        "watertight": bool(meshed.watertight),
        "volume_mm3": round(float(mesh.volume), 1),
        "bounds_mm": [list(map(float, mesh.bounds[0])), list(map(float, mesh.bounds[1]))],
    }

    # The measured cases, same total force, base (z-min) fixed, transverse -Y load:
    #   design = the corner PATCH (matches the SIMP load); mismatch = the whole top face.
    face_case = LoadCase(name="bracket_face_spread", force_n=PAYLOAD_N, load_dir=(0.0, 1.0, 0.0))
    patch_case = LoadCase(
        name="bracket_corner_patch_DESIGN",
        force_n=PAYLOAD_N,
        load_dir=(0.0, 1.0, 0.0),
        load_patch=((xmax, ymax), args.patch_r),  # the top-face corner the part was designed for
    )
    face = run_load_case(mesh, face_case)
    patch = run_load_case(mesh, patch_case)
    report["measured"] = {
        "face_spread_mm": None if face is None else face.max_deflection_mm,
        "corner_patch_DESIGN_mm": None if patch is None else patch.max_deflection_mm,
    }
    if face is not None and patch is not None and face.max_deflection_mm > 0:
        ratio = patch.max_deflection_mm / face.max_deflection_mm
        report["measured"]["patch_over_face_ratio"] = round(ratio, 3)
        # Honest reading (no manufactured gap): the point of load_patch is that the measured
        # case CAN be the design case. Whether a given part's patch and face-spread numbers
        # diverge depends on its geometry -- a slender bending-dominated member barely twists,
        # so the two coincide here; a stubbier / off-axis member (and the unit-test cantilever,
        # patch 1.02x softer) shows the torsion the face-spread averages away.
        report["measured"]["note"] = (
            "measured under the DESIGN (corner-patch) case, not only a face-spread. For this "
            f"slender bracket the two coincide (ratio {ratio:.3f}, bending-dominated); the "
            "capability's pronounced effect is in the unit-test transverse-end cantilever."
            if abs(ratio - 1.0) < 0.05
            else f"the design-aligned corner patch is {ratio:.2f}x the face-spread deflection -- "
            "measuring the corner-designed part under a face-spread would report an over-stiff case."
        )
    report["verdict"] = (
        "measured_the_design_case" if patch is not None else "fea_unavailable_or_patch_empty"
    )
    (out_dir / "topo3d_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if patch is not None else 1


if __name__ == "__main__":
    sys.exit(main())
