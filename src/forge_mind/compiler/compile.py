"""Compiler: PartProgram -> build123d Part -> STL + topology index.

Walks the program's operations sequentially. The first op (which must be ADD)
seeds the running body; subsequent ops are unioned (ADD) or differenced
(SUBTRACT) onto that body. Each op's resulting faces are registered in the
topology index keyed by op_id, so the validator can later ask
"what's the geometry of op_id=motor_bolt_0?".

Deterministic: a fixed PartProgram produces a byte-identical STL across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build123d import export_step, export_stl

from forge_mind.compiler.build_ops.box import build_box
from forge_mind.compiler.build_ops.computed import build_computed_solid
from forge_mind.compiler.build_ops.cylinder import build_cylinder
from forge_mind.compiler.build_ops.gear import build_gear_solid
from forge_mind.compiler.topology_tags import TopologyIndex
from forge_mind.dsl.primitives import Box, ComputedSolid, Cylinder, GearSolid, Operation, OpMode
from forge_mind.dsl.program import PartProgram
from forge_mind.types import BBox, Vec3


@dataclass
class CADModel:
    """Compiled output passed to the validator.

    Holds the build123d body, the STL + STEP paths on disk, the topology index,
    and a cached bounding box. Future stages should not touch ``part`` directly —
    they read ``topology_index`` and ``bounding_box`` for typed introspection.
    """

    part: Any  # build123d.Part
    stl_path: Path
    step_path: Path  # editable B-rep alongside the STL (additive; STL unaffected)
    topology_index: TopologyIndex
    bounding_box: BBox


class CompileError(RuntimeError):
    """Raised when a PartProgram cannot be turned into a solid body."""


def _build_one(op: Operation) -> Any:
    if isinstance(op, Box):
        return build_box(op)
    if isinstance(op, Cylinder):
        return build_cylinder(op)
    if isinstance(op, ComputedSolid):
        return build_computed_solid(op)
    if isinstance(op, GearSolid):
        return build_gear_solid(op)
    raise CompileError(f"unsupported operation type: {type(op).__name__}")


def _tag_faces(index: TopologyIndex, op: Operation, body_fragment: Any) -> None:
    """Record the faces produced by this op into the topology index.

    Tagging is best-effort: we record every face of the fragment under the
    op_id with role ``"face[i]"``. Later phases will refine roles
    (e.g. ``"top"``, ``"side"``, ``"instance[0]"``) per primitive.
    """
    try:
        faces = list(body_fragment.faces())
    except AttributeError:
        return
    for i, face in enumerate(faces):
        index.add(op.op_id, f"face[{i}]", face)


def _bbox_of(body: Any) -> BBox:
    bb = body.bounding_box()
    return BBox(
        min=Vec3(x=bb.min.X, y=bb.min.Y, z=bb.min.Z),
        max=Vec3(x=bb.max.X, y=bb.max.Y, z=bb.max.Z),
    )


def compile_program(program: PartProgram, output_dir: Path) -> CADModel:
    """Compile a PartProgram into a CADModel + STL file.

    Side effects: writes ``<output_dir>/<name>.stl`` and, additively, ``<name>.step``
    (an editable B-rep of the same body; the STL is written first and is unchanged).

    Raises ``CompileError`` if the program is empty or produces no body.
    """
    if not program.operations:
        raise CompileError("PartProgram has no operations")

    index = TopologyIndex()
    body: Any = None

    for i, op in enumerate(program.operations):
        fragment = _build_one(op)

        if op.mode == OpMode.ADD:
            body = fragment if body is None else body + fragment
            # Tag against the fragment so we capture this op's own faces
            # (not the entire merged body's faces).
            _tag_faces(index, op, fragment)
        elif op.mode == OpMode.SUBTRACT:
            if body is None:
                raise CompileError(
                    f"operation {op.op_id!r} (index {i}) has mode=subtract but no body to subtract from"
                )
            # Tag BEFORE subtraction: the fragment carries the cylindrical /
            # box faces we want to query later, even though the subtracted
            # region's faces don't persist in the final body.
            _tag_faces(index, op, fragment)
            body = body - fragment
        else:  # pragma: no cover - exhaustiveness; OpMode is closed
            raise CompileError(f"unknown OpMode: {op.mode}")

    if body is None:
        raise CompileError("compile produced no body")
    # An engulfing subtract (or a fully-cancelled boolean) yields a non-None but EMPTY Compound with
    # volume 0 -- not None, so the check above misses it. compile_program's contract is "raise on no
    # body"; a zero-volume body is no printable body, and silently returning it wrote a phantom part
    # (STL never written) that broke the grader's CompileError contract downstream (audit L-COMP/M6).
    if float(body.volume) <= 0.0:
        raise CompileError(
            f"compile produced an empty/zero-volume body (volume={float(body.volume):.4g} mm^3) -- "
            "a subtract likely engulfed the solid, so there is nothing to print"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / f"{program.metadata.name}.stl"
    export_stl(body, str(stl_path))
    # Additive: emit an editable B-rep (STEP) of the SAME body. The STL above is
    # written first and is byte-unchanged by this; export_step is a separate file.
    # The label names the PRODUCT in the STEP (ADR-0024/0037 fidelity): an engineer's
    # CAD tree shows the part's name, not OCCT's default 'SOLID'.
    step_path = output_dir / f"{program.metadata.name}.step"
    body.label = program.metadata.name
    export_step(body, str(step_path))

    return CADModel(
        part=body,
        stl_path=stl_path,
        step_path=step_path,
        topology_index=index,
        bounding_box=_bbox_of(body),
    )


__all__ = ["CADModel", "CompileError", "compile_program"]
