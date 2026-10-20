"""Volume-to-bound auto-tune (Fable move #2): min mass at a stiffness bound.

SIMP answers "stiffest structure at a FIXED mass budget"; the M1/M2 objective is the inverse --
"the LIGHTEST structure whose validated deflection meets the bound". This module closes that gap
with an outer bisection on ``volfrac`` against the VALIDATED deflection of the extruded B-rep
candidate (the real tet FEA, not the SIMP-internal 2-D model), targeting ``bound - 3x scatter``
exactly like the section author.

Monotonicity assumption (verified by SIMP's construction): more material budget -> stiffer
optimum. Fail-closed: if no evaluated volume fraction meets the bound, or the FEA cannot measure
a candidate (None), the tuner returns the honest outcome -- ``meets_bound`` False / no claim; the
best-so-far artifact is still reported for diagnosis.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trimesh

from forge_mind.materials import mass_grams, resolve_material
from forge_mind.structures.extrude import ExtrudeResult, extrude_density
from forge_mind.structures.loadcase import run_load_case
from forge_mind.structures.objectives import LoadCaseLike
from forge_mind.structures.topopt import Domain, optimize

_FEA_SCATTER_MM = 0.001  # same convergence-band margin as sections.py


@dataclass(frozen=True)
class TuneOutcome:
    """The tuner's honest result. ``meets_bound`` True only for a MEASURED pass with margin."""

    meets_bound: bool
    volfrac: float | None
    mass_g: float | None
    deflection_mm: float | None
    stl: Path | None
    step: Path | None
    history: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "meets_bound": self.meets_bound,
            "volfrac": self.volfrac,
            "mass_g": self.mass_g,
            "deflection_mm": self.deflection_mm,
            "stl": None if self.stl is None else str(self.stl),
            "step": None if self.step is None else str(self.step),
            "history": self.history,
        }


def _measure(
    res: ExtrudeResult, load_case: LoadCaseLike, material_id: str
) -> tuple[float | None, float | None]:
    """(mass_g, deflection_mm) of an extruded candidate; None components are fail-closed."""
    mesh: Any = trimesh.load(str(res.stl), force="mesh")
    if not bool(mesh.is_watertight) or float(mesh.volume) <= 0.0:
        return None, None
    mass = mass_grams(float(mesh.volume), resolve_material(material_id))
    case = load_case(mesh) if callable(load_case) else load_case
    lr = run_load_case(mesh, case)
    return round(mass, 2), (None if lr is None else lr.max_deflection_mm)


def tune_volfrac_to_bound(
    domain_builder: Callable[[float], Domain],
    *,
    load_case: LoadCaseLike,
    bound_mm: float,
    out_dir: Path,
    scale_xy: tuple[float, float],
    height_mm: float,
    material_id: str = "pla",
    lo: float = 0.15,
    hi: float = 0.85,
    max_fea_evals: int = 6,
    simp_max_iter: int = 60,
    extrude_thresh: float = 0.4,
    open_radius_cells: int | None = None,
) -> TuneOutcome:
    """Bisect ``volfrac`` to the lightest extruded candidate whose VALIDATED deflection clears
    ``bound_mm`` with margin. Each eval = one SIMP solve + one B-rep extrude + one real FEA
    (minutes at arm scale -- ``max_fea_evals`` keeps the budget honest)."""
    target = bound_mm - 3 * _FEA_SCATTER_MM
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None  # lightest candidate that MEASURED within target

    for i in range(max_fea_evals):
        vf = 0.5 * (lo + hi)
        dom = domain_builder(vf)
        dom.volfrac = vf
        res = optimize(dom, max_iter=simp_max_iter)
        ext = extrude_density(
            res["density"],
            out_dir=out_dir,
            scale_xy=scale_xy,
            height_mm=height_mm,
            thresh=extrude_thresh,
            open_radius_cells=open_radius_cells,
            name=f"cand_vf{vf:.3f}",
        )
        if ext is None:
            # Vectorization failed at this budget (e.g. fragmented field) -> treat as too light.
            history.append({"volfrac": round(vf, 4), "extrude": "failed"})
            lo = vf
            continue
        mass, defl = _measure(ext, load_case, material_id)
        history.append(
            {"volfrac": round(vf, 4), "mass_g": mass, "deflection_mm": defl, "stl": str(ext.stl)}
        )
        if defl is None:
            # Unmeasurable candidate: no verdict possible at this budget; do NOT claim. Try
            # heavier (a denser field is likelier to mesh), but never record a pass.
            lo = vf
            continue
        if defl <= target:
            best = {
                "volfrac": vf,
                "mass_g": mass,
                "deflection_mm": defl,
                "stl": ext.stl,
                "step": ext.step,
            }
            hi = vf  # meets the bound -> try lighter
        else:
            lo = vf  # too floppy -> needs more material
        _ = i

    (out_dir / "tune_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if best is None:
        return TuneOutcome(False, None, None, None, None, None, history)
    return TuneOutcome(
        True,
        round(float(best["volfrac"]), 4),
        best["mass_g"],
        best["deflection_mm"],
        best["stl"],
        best["step"],
        history,
    )


__all__ = ["TuneOutcome", "tune_volfrac_to_bound"]
