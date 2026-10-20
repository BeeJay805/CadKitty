"""ADR-0019 Step-0 PREMISE DE-RISK (READ-ONLY, $0): is a SOLID obtuse-dihedral tapered
strut validator-ready TODAY? Hand-authors rect/hex tapered struts in build123d and runs
each through the LIVE validator. The premise holds iff min_wall reads the TRUE thin-end
cross-section (NOT an acute-edge sliver like the trapezoid's 0.63mm) and overhang behaves.
If min_wall over-rejects a solid obtuse strut, the "validator-ready first op" premise FAILS
-> STOP (reconsider the first op or accelerate the SDF min_wall).

Run: uv run python scripts/derisk_tapered_strut.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import trimesh
from build123d import Plane, Rectangle, RegularPolygon, export_step, export_stl, loft

from forge_mind.compiler.compile import CADModel
from forge_mind.compiler.topology_tags import TopologyIndex
from forge_mind.dsl.primitives import Box as DslBox
from forge_mind.dsl.primitives import OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import BBox, Vec3
from forge_mind.validator.feature_checks import _measure_min_wall
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="strut_"))
_DUMMY = DslBox(
    op_id="placeholder", mode=OpMode.ADD, position=Vec3(x=0, y=0, z=0), width=1, depth=1, height=1
)


def _program(min_wall: float) -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="strut", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[_DUMMY],
        interfaces=[],
        constraints=Constraints(
            min_wall_thickness=min_wall,
            max_overhang_deg=45.0,
            max_bbox=Vec3(x=400.0, y=400.0, z=400.0),
        ),
    )


def _model(part: Any, name: str) -> CADModel:
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


def _section(kind: str, r: float, x: float):  # type: ignore[no-untyped-def]
    """A cross-section polygon in the YZ plane at axis position x (strut axis = X)."""
    plane = Plane(origin=(x, 0, 0), z_dir=(1, 0, 0))
    if kind == "hex":
        return plane * RegularPolygon(radius=r, side_count=6)  # 120-deg dihedrals
    return plane * Rectangle(2 * r, 1.4 * r)  # 90-deg dihedrals


def _strut(kind: str, r0: float, r1: float, length: float) -> Any:
    return loft([_section(kind, r0, 0.0), _section(kind, r1, length)])


def probe(name: str, kind: str, r0: float, r1: float, length: float, min_wall: float) -> None:
    part = _strut(kind, r0, r1, length)
    model = _model(part, name)
    rep = validate(model, _program(min_wall))
    by = {c.id: c for c in rep.checks}
    mesh = trimesh.load(str(model.stl_path), force="mesh")
    raw = _measure_min_wall(mesh)
    mw = by.get("features.min_wall_thickness")
    oh = by.get("printability.overhang_max_angle")
    wt = by.get("mesh.watertight")
    raw_s = f"{raw:.2f}" if raw is not None else "None"
    print(
        f"[{name:<22}] min_wall={mw.status.value if mw else '-':<4} measured={mw.actual if mw else '-':<9}"
        f" (raw={raw_s})  overhang={oh.status.value if oh else '-':<4} {oh.actual if oh else '':<8}"
        f"  watertight={wt.status.value if wt else '-'}"
    )


def main() -> None:
    print("min_wall constraint = 3.0 mm (chunky structural member); strut lies axis-along-X\n")
    # mild taper (r 6->5) and moderate (r 6->4) over an 80mm member, hex + rect.
    probe("hex_mild", "hex", 6.0, 5.0, 80.0, 3.0)
    probe("hex_moderate", "hex", 6.0, 4.0, 80.0, 3.0)
    probe("rect_mild", "rect", 6.0, 5.0, 80.0, 3.0)
    probe("rect_moderate", "rect", 6.0, 4.0, 80.0, 3.0)
    # a straight (no-taper) member for reference.
    probe("hex_straight", "hex", 6.0, 6.0, 80.0, 3.0)
    probe("rect_straight", "rect", 6.0, 6.0, 80.0, 3.0)
    print(
        f"\nPREMISE: holds iff every min_wall=PASS with measured ~thin-end cross-section "
        f"(not a <1mm sliver).\n(artifacts in {WORK})"
    )


if __name__ == "__main__":
    main()
