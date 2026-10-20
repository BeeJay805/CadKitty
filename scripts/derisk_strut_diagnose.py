"""ADR-0019 Step-0 diagnosis: separate the TAPER artifact from the ORIENTATION artifact.

(1) Find a flat-bottom hex orientation where overhang PASSES (a straight hex prism is a
    candidate validator-ready first op if so).
(2) Confirm the TAPERED strut's min_wall sliver is an ACUTE-EDGE ARTIFACT (true thin-end
    cross-section is ~8.7mm) by locating where the thin hits land, with a SEEDED sampler
    so the number is stable.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import Plane, RegularPolygon, export_stl, loft

from forge_mind.compiler.compile import CADModel
from forge_mind.compiler.topology_tags import TopologyIndex
from forge_mind.dsl.primitives import Box as DslBox
from forge_mind.dsl.primitives import OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import BBox, Vec3
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="strutdiag_"))
_DUMMY = DslBox(
    op_id="placeholder", mode=OpMode.ADD, position=Vec3(x=0, y=0, z=0), width=1, depth=1, height=1
)


def _prog(mw: float) -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="s", print_orientation=Vec3(x=0, y=0, z=1)),
        operations=[_DUMMY],
        interfaces=[],
        constraints=Constraints(
            min_wall_thickness=mw, max_overhang_deg=45.0, max_bbox=Vec3(x=400, y=400, z=400)
        ),
    )


def _model(part: Any, name: str) -> CADModel:
    stl = WORK / f"{name}.stl"
    export_stl(part, str(stl))
    bb = part.bounding_box()
    return CADModel(
        part=part,
        stl_path=stl,
        step_path=stl,
        topology_index=TopologyIndex(),
        bounding_box=BBox(
            min=Vec3(x=bb.min.X, y=bb.min.Y, z=bb.min.Z),
            max=Vec3(x=bb.max.X, y=bb.max.Y, z=bb.max.Z),
        ),
    )


def _hex(r: float, x: float, rot: float):  # type: ignore[no-untyped-def]
    return Plane(origin=(x, 0, 0), z_dir=(1, 0, 0)) * RegularPolygon(
        radius=r, side_count=6, rotation=rot
    )


def overhang_of(part: Any, name: str) -> str:
    rep = validate(_model(part, name), _prog(3.0))
    oh = next((c for c in rep.checks if c.id == "printability.overhang_max_angle"), None)
    return f"{oh.status.value} ({oh.actual})" if oh else "-"


def seeded_minwall(part: Any, name: str) -> tuple[float, float, float]:
    """Seeded thinnest reading + where it lands: returns (thinnest_mm, dist_to_nearest_edge_proxy, bulk_median)."""
    stl = WORK / f"{name}.stl"
    export_stl(part, str(stl))
    mesh = trimesh.load(str(stl), force="mesh")
    s, fi = trimesh.sample.sample_surface(mesh, 4000, seed=0)
    n = mesh.face_normals[fi]
    o = s - n * 0.05
    loc, ri, _ = mesh.ray.intersects_location(ray_origins=o, ray_directions=-n, multiple_hits=False)
    d = np.linalg.norm(loc - o[ri], axis=1)
    d = d[d > 0.3] + 0.05
    thin = float(d.min())
    # how far is the thinnest sample's X from the two end-cap planes (x=0, x=80)? near a cap = end artifact.
    xs = (
        s[ri][d.argmin() if False else np.argmin(np.linalg.norm(loc - o[ri], axis=1))][0]
        if False
        else 0.0
    )
    return thin, float(np.median(d)), xs


def main() -> None:
    print("=== (1) straight hex overhang vs rotation (find flat-bottom, want overhang PASS) ===")
    for rot in (0.0, 15.0, 30.0, 45.0):
        part = loft([_hex(6, 0, rot), _hex(6, 80, rot)])
        print(f"  rotation={rot:>5}  overhang={overhang_of(part, f'hexstr_{rot}')}")

    print("\n=== (2) taper artifact: true thin-end across-flats vs measured sliver ===")
    for r1 in (6.0, 5.0, 4.0):
        af = r1 * math.sqrt(3)  # hex across-flats = true min cross-section thickness
        part = loft([_hex(6, 0, 30), _hex(r1, 80, 30)])
        thin, med, _ = seeded_minwall(part, f"taper_{r1}")
        tag = "TAPERED" if r1 != 6.0 else "straight"
        print(
            f"  r0=6 r1={r1} ({tag:<8}) true across-flats~{af:5.2f}mm  seeded thinnest={thin:5.2f}mm  median_hit={med:5.2f}mm"
        )
    print(f"\n(artifacts in {WORK})")


if __name__ == "__main__":
    main()
