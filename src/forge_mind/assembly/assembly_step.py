"""Positioned multi-body STEP export for solved assemblies (ADR-0044 increment).

The servo/gripper lane's per-part STEPs are authored in PART-LOCAL frames and placed
by solver transforms; this module re-imports each PRINTED part's STEP, applies its
solved world pose as a real B-rep Location, labels it, and exports ONE labeled
Compound -- the file a real engineer opens in Fusion/SolidWorks with the parts where
the solver put them.

HONESTY SCOPE: printed parts ONLY. Hardware (servos, dowels) ships as BOM lines +
GLB collision envelopes -- a box-and-cylinder "servo" in a manufacturing-exchange
format invites dimensioning against fake geometry (council 3-2, founder item). The
manifest note says exactly that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from forge_mind.compiler.compile_assembly import AssemblyModel

ASSEMBLY_STEP_NOTE = (
    "Printed parts only, placed at the solved home pose. Bought hardware (servos, "
    "dowel pins) is NOT modeled here -- see the BOM for the cited parts and the GLB "
    "for their collision envelopes."
)


def _location_from_matrix(m: NDArray[np.float64]) -> Any:
    """A build123d Location from a rigid 4x4 (rotation rows + translation)."""
    from build123d import Location
    from OCP.gp import gp_Trsf

    t = gp_Trsf()
    t.SetValues(
        float(m[0, 0]),
        float(m[0, 1]),
        float(m[0, 2]),
        float(m[0, 3]),
        float(m[1, 0]),
        float(m[1, 1]),
        float(m[1, 2]),
        float(m[1, 3]),
        float(m[2, 0]),
        float(m[2, 1]),
        float(m[2, 2]),
        float(m[2, 3]),
    )
    return Location(t)


def export_assembly_step(model: AssemblyModel, step_path: Path, *, name: str) -> Path | None:
    """Write the PLACED printed parts as one labeled STEP Compound. Best-effort:
    returns None when nothing exports (the per-part STEPs remain the fallback --
    artifact generation must never take down a shipped result)."""
    try:
        from build123d import Compound, export_step, import_step
    except Exception:  # pragma: no cover - kernel import is environment-level
        return None
    children: list[Any] = []
    for body in model.bodies:
        step_src = Path(body.cad.step_path)
        if not step_src.exists():
            continue
        try:
            solid = import_step(str(step_src))
            solid.label = body.id
            m = np.asarray(body.transform, dtype=float)
            if not np.allclose(m, np.eye(4), atol=1e-12):
                solid.move(_location_from_matrix(m))
            children.append(solid)
        except Exception:  # pragma: no cover - a single bad part must not kill the rest
            continue
    if not children:
        return None
    try:
        asm = Compound(children=children)
        asm.label = name
        export_step(asm, str(step_path))
        return step_path
    except Exception:  # pragma: no cover - best-effort artifact
        return None


__all__ = ["ASSEMBLY_STEP_NOTE", "export_assembly_step"]
