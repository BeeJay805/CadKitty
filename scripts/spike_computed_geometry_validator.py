"""SPIKE (throwaway, deterministic, $0) — surface-area map: does the prismatic 4-tier
validator cope with COMPUTED geometry (angled/faceted, lattice, ruled, curved)?

Hand-authors a few watertight engineering solids DIRECTLY in build123d (bypassing the
box/cylinder DSL), runs EACH through the real validator, and introspects each tier to
record PASS / BREAKS / GIVES-NONSENSE. Does NOT modify the validator (read-only probe).

Run: uv run python scripts/spike_computed_geometry_validator.py
"""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from build123d import (
    Box as B3Box,
)
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

from forge_mind.compiler.compile import CADModel
from forge_mind.compiler.topology_tags import TopologyIndex
from forge_mind.dsl.primitives import Box as DslBox
from forge_mind.dsl.primitives import OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import BBox, Vec3
from forge_mind.validator.feature_checks import _measure_min_wall
from forge_mind.validator.geometry import extract_cylinders, extract_planes
from forge_mind.validator.run import validate
from forge_mind.viz.render import render_part_to_png

WORK = Path(tempfile.mkdtemp(prefix="spike_computed_"))
# Portable: resolves to the author's Desktop folder on their machine, and to any user's
# ~/Desktop elsewhere (no hard-coded personal path -- Codex).
DESK = Path.home() / "Desktop" / "forge-mind-adr0017-renders"
_DUMMY = DslBox(
    op_id="placeholder", mode=OpMode.ADD, position=Vec3(x=0, y=0, z=0), width=1, depth=1, height=1
)
_UP = np.array([0.0, 0.0, 1.0])


def _program(min_wall: float = 1.0) -> PartProgram:
    # The validator reads metadata/constraints/interfaces, NOT operations (one dummy op just
    # satisfies the schema's min_length). Empty interfaces -> Tier 4 is a no-op here.
    return PartProgram(
        metadata=Metadata(name="spike", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[_DUMMY],
        interfaces=[],
        constraints=Constraints(
            min_wall_thickness=min_wall,
            max_overhang_deg=45.0,
            max_bbox=Vec3(x=300.0, y=300.0, z=300.0),
        ),
    )


def _model(part: Any, name: str) -> tuple[CADModel, str | None]:
    stl, step = WORK / f"{name}.stl", WORK / f"{name}.step"
    export_stl(part, str(stl))
    export_step(part, str(step))
    idx, tag_err = TopologyIndex(), None
    try:  # mimic the compiler's best-effort _tag_faces on a hand-authored solid
        for i, f in enumerate(part.faces()):
            idx.add(name, f"face[{i}]", f)
    except Exception as e:
        tag_err = repr(e)
    bb = part.bounding_box()
    bbox = BBox(
        min=Vec3(x=bb.min.X, y=bb.min.Y, z=bb.min.Z), max=Vec3(x=bb.max.X, y=bb.max.Y, z=bb.max.Z)
    )
    return CADModel(
        part=part, stl_path=stl, step_path=step, topology_index=idx, bounding_box=bbox
    ), tag_err


def _true_overhang_area(stl: Path, max_deg: float = 45.0) -> tuple[float, float]:
    """Ground truth from the MESH: total area of faces facing downward beyond max_deg and
    above the bed. This is what a correct overhang check would catch on ANY surface."""
    m = trimesh.load(str(stl), force="mesh")
    assert isinstance(m, trimesh.Trimesh)
    down = -(m.face_normals @ _UP)  # >0 = downward-facing
    proj = m.triangles_center @ _UP
    bed = proj.min()
    sin_max = math.sin(math.radians(max_deg))
    mask = (down > sin_max) & ((proj - bed) > 0.5)
    return float(m.area_faces[mask].sum()), float(m.area)


def _geoms(part: Any) -> dict[str, int]:
    return dict(Counter(str(f.geom_type) for f in part.faces()))


def probe(name: str, category: str, part: Any, min_wall_constraint: float = 1.0) -> None:
    model, tag_err = _model(part, name)
    rep = validate(model, _program(min_wall_constraint))
    by_id = {c.id: c for c in rep.checks}

    n_faces = len(part.faces())
    geoms = _geoms(part)
    n_plane = len(extract_planes(part))
    n_cyl = len(extract_cylinders(part))
    raw_minwall = _measure_min_wall(trimesh.load(str(model.stl_path), force="mesh"))
    oh_true, total_area = _true_overhang_area(model.stl_path)
    oh_check = by_id.get("printability.overhang_max_angle")

    print(f"\n=== [{category}] {name} ===")
    mesh_ids = [
        "mesh.watertight",
        "mesh.is_volume",
        "mesh.winding_consistent",
        "mesh.single_body",
        "mesh.positive_volume",
        "mesh.bbox_within_max",
    ]
    mstat = {i: (by_id[i].status.value if i in by_id else "-") for i in mesh_ids}
    print("  T1 mesh:        ", ", ".join(f"{i.split('.')[1]}={s}" for i, s in mstat.items()))
    mw = by_id.get("features.min_wall_thickness")
    raw_str = f"{raw_minwall:.2f}mm" if raw_minwall is not None else "None"
    print(
        f"  T2 min_wall:     status={mw.status.value if mw else '-'}  measured={mw.actual if mw else '-'}"
        f"  (raw ray-cast={raw_str})"
    )
    print(
        f"  T3 overhang:     status={oh_check.status.value if oh_check else '-'}  reports={oh_check.actual if oh_check else '-'}"
    )
    print(
        f"     GROUND TRUTH: {oh_true:.0f} mm^2 of mesh faces overhang >45deg (of {total_area:.0f} mm^2 total)"
    )
    print(f"  faces: total={n_faces}  geom_types={geoms}")
    print(
        f"  introspection: extract_planes={n_plane}  extract_cylinders={n_cyl}  topology_tag_error={tag_err}"
    )
    DESK.mkdir(parents=True, exist_ok=True)
    render_part_to_png(model.stl_path, DESK / f"spike_{name}.png", title=f"[{category}] {name}")


def main() -> None:
    # 1. ANGLED / FACETED PLANAR: a flared trapezoidal prism. The two slanted sides are PLANAR
    #    but non-axis-aligned and face downward-out (~59deg overhang). Tests angled flat faces.
    trap = extrude(
        Plane.XZ * Polygon((-10, 0), (10, 0), (35, 15), (-35, 15), align=None), amount=15
    )
    probe("trapezoid_flared", "ANGLED-PLANAR", trap, min_wall_constraint=10.0)

    # 2. LATTICE / TRUSS: a grid of thin 2mm struts unioned (thin walls + many intersections).
    bars: Any = None
    for y in (-15.0, 0.0, 15.0):
        member = Pos(0, y, 1) * B3Box(40, 2, 2)
        bars = member if bars is None else bars + member
    for x in (-15.0, 0.0, 15.0):
        bars = bars + (Pos(x, 0, 1) * B3Box(2, 40, 2))
    probe("lattice_grid", "LATTICE", bars, min_wall_constraint=2.0)

    # 3. RULED / CURVED: a loft from a small circle up to a large square -> flares OUT, so the
    #    curved transition has a downward-facing (overhanging) underside.
    transition = loft([Pos(0, 0, 0) * Circle(8), Pos(0, 0, 30) * Rectangle(40, 40)])
    probe("loft_round_to_square", "CURVED-LOFT", transition, min_wall_constraint=1.0)

    # 4. SPLINE SWEEP: a circular profile swept along a curved spline path -> all-BSpline tube.
    path = Spline((0, 0, 0), (0, 0, 15), (15, 0, 25), (30, 0, 30), tangents=((0, 0, 1), (1, 0, 0)))
    tube = sweep(Plane.XY * Circle(4), path=path)
    probe("spline_sweep_tube", "CURVED-SWEEP", tube, min_wall_constraint=1.0)

    print(f"\n(artifacts in {WORK})")


if __name__ == "__main__":
    main()
