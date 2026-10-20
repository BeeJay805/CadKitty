"""AssemblyProgram — the multi-body compile/validate target (ADR-0010 spike).

A single part is one ``PartProgram`` (unchanged). An assembly is N parts, each with
a world ``Transform`` the solver computed, plus the joints (so the validator knows
the motion semantics). A single part is just a 1-node assembly with identity
transform, which is how the existing single-part benches stay untouched.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from forge_mind.assembly.joints import Joint, Port
from forge_mind.dsl.program import PartProgram


class Transform(BaseModel):
    """A 4x4 rigid world transform, JSON-serializable (rows of floats)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: list[list[float]] = Field(
        ..., description="4x4 homogeneous matrix as 4 rows of 4 floats."
    )

    @classmethod
    def identity(cls) -> Transform:
        return cls(
            rows=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    @classmethod
    def from_matrix(cls, m: NDArray[np.float64]) -> Transform:
        arr = np.asarray(m, dtype=float)
        return cls(rows=[[float(v) for v in row] for row in arr])

    def as_matrix(self) -> NDArray[np.float64]:
        return np.array(self.rows, dtype=float)


class AssemblyPart(BaseModel):
    """One placed body: its single-part program, world transform, and local ports.

    ``program`` is None for a non-printed HARDWARE part (ADR-0012): it has placed
    frames (ports) + a transform but no geometry to compile. (A hardware part MAY
    carry a program -- its collision/display ENVELOPE -- while staying printed=False;
    the compiler routes on the flag, never the program.) ``printed`` defaults True so
    every existing printed part stays byte-identical. ``hardware_kind`` names the
    registry entry a hardware part cites (e.g. "mg996r_servo") so the mount-pattern
    gate knows which CITED hole pattern the mated printed part must carry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    program: PartProgram | None
    transform: Transform
    ports: list[Port]
    printed: bool = True
    hardware_kind: str | None = None


class AssemblyProgram(BaseModel):
    """A complete multi-body assembly ready for compilation + relational validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parts: list[AssemblyPart] = Field(..., min_length=1)
    joints: list[Joint] = Field(default_factory=list)

    def part(self, part_id: str) -> AssemblyPart:
        return next(p for p in self.parts if p.id == part_id)


__all__ = ["AssemblyPart", "AssemblyProgram", "Transform"]
