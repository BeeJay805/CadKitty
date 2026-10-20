"""Multi-body compiler — compile each part with the EXISTING single-part compiler,
then carry its world transform + meshes for relational validation.

Deliberately additive: ``compile_program``'s STL output is unchanged (it now also
emits an editable STEP per body, additively), so single-part benchmarks are
unaffected. Each body is written to its own STL + STEP (ready to slice / edit); the
relational validator applies the world transforms when it needs placed geometry.

A combined multi-body STEP is intentionally NOT emitted: placing each B-rep body at
its solved pose needs a 4x4 -> build123d Location conversion (fiddly and error-prone),
and the per-body STEP files open directly in any CAD tool. Non-printed hardware nodes
get no STEP (they get no STL either).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray

from forge_mind.assembly.joints import Joint, Port
from forge_mind.assembly.program import AssemblyProgram
from forge_mind.compiler.compile import CADModel, compile_program
from forge_mind.dsl.program import PartProgram


@dataclass
class PlacedBody:
    """One compiled body plus the world transform that places it in the assembly."""

    id: str
    program: PartProgram  # the single-part program (for per-body mesh checks)
    cad: CADModel  # holds the build123d part (B-rep) + the per-body STL path
    mesh: trimesh.Trimesh  # the per-body mesh in LOCAL coords (apply transform to place)
    transform: NDArray[np.float64]  # 4x4 world transform
    ports: list[Port]


@dataclass
class PlacedHardware:
    """A placed NON-PRINTED hardware part (ADR-0012): frames + transform (+ envelope).

    A hardware node (motor, servo, bearing) is mated and placed like any node but is
    never a deliverable print. ``mesh``/``stl_path`` carry its compiled collision/display
    ENVELOPE when the hardware kind provides one -- that is what lets the swept-collision
    gate include hardware-anchored joints instead of skipping them (the ADR-0012
    deferral). Both stay None for frames-only hardware (legacy shape); the motion gate
    then reports the sweep as skipped (WARN), exactly as before."""

    id: str
    transform: NDArray[np.float64]  # 4x4 world transform
    ports: list[Port]
    mesh: trimesh.Trimesh | None = None  # envelope in LOCAL coords (apply transform)
    stl_path: Path | None = None  # the compiled envelope STL (for the viewer export)
    kind: str | None = None  # registry hardware kind, e.g. "mg996r_servo" (cited patterns)


@dataclass
class AssemblyModel:
    """The compiled assembly handed to ``validate_assembly``.

    ``bodies`` are the PRINTED parts (each has a mesh). ``hardware`` are the non-printed
    placed parts (frames only). Joint endpoints may reference either kind.
    """

    bodies: list[PlacedBody]
    joints: list[Joint]
    hardware: list[PlacedHardware] = field(default_factory=list)

    def body(self, part_id: str) -> PlacedBody:
        """The PRINTED body for ``part_id`` (raises if it is hardware / unknown)."""
        return next(b for b in self.bodies if b.id == part_id)

    def find_hardware(self, part_id: str) -> PlacedHardware | None:
        return next((h for h in self.hardware if h.id == part_id), None)

    def placed_ports(self, part_id: str) -> tuple[NDArray[np.float64], list[Port]]:
        """(transform, ports) for any endpoint -- printed body OR hardware node. Lets a
        frame check (coaxial anchor) read placement without assuming a mesh."""
        for b in self.bodies:
            if b.id == part_id:
                return b.transform, b.ports
        for h in self.hardware:
            if h.id == part_id:
                return h.transform, h.ports
        raise KeyError(f"no placed part {part_id!r}")


def compile_assembly(assembly: AssemblyProgram, output_dir: Path) -> AssemblyModel:
    """Compile every part to its own body + STL. Never raises on a single part's
    failure being silently swallowed -- a compile error propagates (the spike wants
    to see it)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bodies: list[PlacedBody] = []
    hardware: list[PlacedHardware] = []
    for part in assembly.parts:
        # Non-printed hardware node (ADR-0012): place its frames; compile its collision
        # ENVELOPE when the hardware kind provides one (program present, printed=False)
        # so the motion gate can sweep hardware-anchored joints. Best-effort: an
        # envelope that fails to compile degrades to the frames-only shape (WARN'd
        # sweep), never a crash -- the envelope is evidence, not the deliverable.
        if not part.printed or part.program is None:
            hw_mesh: trimesh.Trimesh | None = None
            hw_stl: Path | None = None
            if part.program is not None:
                try:
                    hw_cad = compile_program(part.program, output_dir / part.id)
                    loaded = trimesh.load(str(hw_cad.stl_path), force="mesh")
                    if isinstance(loaded, trimesh.Trimesh):
                        hw_mesh, hw_stl = loaded, Path(hw_cad.stl_path)
                except Exception:  # pragma: no cover - envelope compile is best-effort
                    hw_mesh, hw_stl = None, None
            hardware.append(
                PlacedHardware(
                    id=part.id,
                    transform=part.transform.as_matrix(),
                    ports=part.ports,
                    mesh=hw_mesh,
                    stl_path=hw_stl,
                    kind=part.hardware_kind,
                )
            )
            continue
        # compile_program writes <part.id>.stl AND, additively, <part.id>.step.
        cad = compile_program(part.program, output_dir / part.id)
        mesh = trimesh.load(str(cad.stl_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):  # pragma: no cover - defensive
            raise TypeError(f"body {part.id} did not load as a single mesh: {type(mesh).__name__}")
        bodies.append(
            PlacedBody(
                id=part.id,
                program=part.program,
                cad=cad,
                mesh=mesh,
                transform=part.transform.as_matrix(),
                ports=part.ports,
            )
        )
    return AssemblyModel(bodies=bodies, joints=list(assembly.joints), hardware=hardware)


__all__ = ["AssemblyModel", "PlacedBody", "PlacedHardware", "compile_assembly"]
