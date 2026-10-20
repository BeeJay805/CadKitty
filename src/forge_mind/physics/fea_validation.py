"""FEA validation harness — the GATE that lets the linear-elastic solver be called decision-grade.

A stress/deflection number is only trustworthy if the solver reproduces cases with a KNOWN closed-form
answer. This harness drives ``fea.solve_elasticity`` on analytic load cases and reports the relative
error against the exact solution:

  - AXIAL tension (along X and along Z) — GATING. Uniaxial stress sigma = traction, elongation
    delta = traction*L/E. Reproduced essentially exactly.
  - CANTILEVER end-load bending — GATING. Euler-Bernoulli tip deflection delta = F L^3 / (3 E I). With
    QUADRATIC (P2) tets this matches to ~2% (P1 linear tets shear-locked at ~47% and could only be
    informational); P2 makes bending deflection decision-grade, so it is now a trust gate too.

The solver uses P2 here (``element="P2"``) — the order the decision-grade load-case path uses. The
illustrative heatmap (``estimate_stress``) stays P1 (fast, relative-only). ``fea_is_validated()`` is True
only when every GATING case passes. If scikit-fem is absent it returns False (we cannot prove correctness
-> not validated; fail closed, never an unearned "decision-grade"). Unit-tested so a solver regression
flips the gate red in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from forge_mind.physics.fea import solve_elasticity

# Quadratic (P2) tets are used for the decision-grade path, so a modest density converges well:
# axial is exact and the cantilever lands ~2% (P1 at the same DOF count locks at ~47%).
_REFINE = 2
_ELEMENT = "P2"
_E_MPA = 3500.0
_NU = 0.3
_TRACTION_MPA = 5.0
# Axial (uniaxial) is essentially exact; 5% absorbs clamp-edge + meshing noise.
_AXIAL_TOL = 0.05
# Bending with P2 tets converges to a few % here, so it is GATING (decision-grade) — not informational.
# (With P1 it was ~47% off and could only be informational; P2 removes the shear-locking.)
_BENDING_TOL = 0.06


@dataclass(frozen=True)
class ValidationOutcome:
    case: str
    quantity: str  # "stress" | "deflection"
    fem: float
    analytical: float
    rel_error: float
    tolerance: float
    gating: bool
    passed: bool


def _bar(dims: tuple[float, float, float]) -> Any | None:
    """A refined MeshTet box scaled to ``dims`` (mm), or None if scikit-fem is unavailable."""
    try:
        from skfem import MeshTet
    except Exception:
        return None
    return MeshTet().refined(_REFINE).scaled(dims)


def _axial_case(axis: int, length: float) -> list[ValidationOutcome]:
    """Prismatic bar long along ``axis``: fix the axis=0 face, pull with a +axis traction on the far face.
    Analytic uniaxial: von Mises == traction; end elongation == traction*length/E."""
    dims = [2.0, 2.0, 2.0]
    dims[axis] = length
    mesh = _bar((dims[0], dims[1], dims[2]))
    if mesh is None:
        return []
    tol = max(length * 0.02, 1e-6)
    load_dir = [0.0, 0.0, 0.0]
    load_dir[axis] = _TRACTION_MPA
    res = solve_elasticity(
        mesh,
        _E_MPA,
        _NU,
        fixed_pred=lambda x: x[axis] < 1e-6,
        load_facet_pred=lambda x: x[axis] > length - tol,
        load_dir=(load_dir[0], load_dir[1], load_dir[2]),
        element=_ELEMENT,
    )
    if res is None:
        return []
    disp, vm_node = res
    pts = np.asarray(mesh.p)  # (3, n_nodes)
    tip = pts[axis] > length - tol
    fem_disp = float(np.mean(disp[tip, axis]))
    ana_disp = _TRACTION_MPA * length / _E_MPA
    fem_stress = float(np.median(vm_node))  # interior median dodges the clamp stress riser
    ana_stress = _TRACTION_MPA
    name = f"axial_{'xyz'[axis]}"
    return [
        ValidationOutcome(
            case=name,
            quantity="stress",
            fem=round(fem_stress, 4),
            analytical=round(ana_stress, 4),
            rel_error=round(abs(fem_stress - ana_stress) / ana_stress, 4),
            tolerance=_AXIAL_TOL,
            gating=True,
            passed=abs(fem_stress - ana_stress) / ana_stress <= _AXIAL_TOL,
        ),
        ValidationOutcome(
            case=name,
            quantity="deflection",
            fem=round(fem_disp, 6),
            analytical=round(ana_disp, 6),
            rel_error=round(abs(fem_disp - ana_disp) / ana_disp, 4),
            tolerance=_AXIAL_TOL,
            gating=True,
            passed=abs(fem_disp - ana_disp) / ana_disp <= _AXIAL_TOL,
        ),
    ]


def _cantilever_case() -> list[ValidationOutcome]:
    """Cantilever along X (L x W x H), fixed x=0, transverse -Z traction on the x=L end face.
    Euler-Bernoulli tip deflection: delta = F L^3 / (3 E I), F = traction*W*H, I = W*H^3/12. INFORMATIONAL
    (P1 tets are over-stiff in bending) — measured so we never over-claim bending accuracy."""
    length, w, h = 20.0, 3.0, 3.0
    mesh = _bar((length, w, h))
    if mesh is None:
        return []
    tol = max(length * 0.02, 1e-6)
    res = solve_elasticity(
        mesh,
        _E_MPA,
        _NU,
        fixed_pred=lambda x: x[0] < 1e-6,
        load_facet_pred=lambda x: x[0] > length - tol,
        load_dir=(0.0, 0.0, -_TRACTION_MPA),
        element=_ELEMENT,
    )
    if res is None:
        return []
    disp, _vm = res
    pts = np.asarray(mesh.p)
    tip = pts[0] > length - tol
    fem_defl = float(abs(np.mean(disp[tip, 2])))
    force = _TRACTION_MPA * w * h
    inertia = w * h**3 / 12.0
    ana_defl = force * length**3 / (3.0 * _E_MPA * inertia)
    return [
        ValidationOutcome(
            case="cantilever_bending",
            quantity="deflection",
            fem=round(fem_defl, 5),
            analytical=round(ana_defl, 5),
            rel_error=round(abs(fem_defl - ana_defl) / ana_defl, 4),
            tolerance=_BENDING_TOL,
            gating=True,  # P2 removes the shear-locking -> bending deflection is now decision-grade
            passed=abs(fem_defl - ana_defl) / ana_defl <= _BENDING_TOL,
        )
    ]


def run_validation() -> list[ValidationOutcome]:
    """Run every analytic case. Empty list if scikit-fem is unavailable (cannot validate)."""
    outcomes: list[ValidationOutcome] = []
    outcomes += _axial_case(0, 10.0)
    outcomes += _axial_case(2, 10.0)
    outcomes += _cantilever_case()
    return outcomes


@lru_cache(maxsize=1)
def fea_is_validated() -> bool:
    """True iff every GATING case reproduces its analytic answer within tolerance. False when scikit-fem
    is absent (no proof = not validated). The single switch a caller checks before trusting FEA numbers.
    Cached: the harness is deterministic, so it runs once per process (every decision-grade FEA call gates
    on it)."""
    try:
        outcomes = run_validation()
    except Exception:
        # Any failure to RUN the harness (scikit-fem absent OR a solver raise) means the solver is
        # NOT proven correct -- fail closed: not validated. This bool is the single switch callers
        # trust before reporting a decision-grade FEA number; it must never propagate (audit).
        return False
    gating = [o for o in outcomes if o.gating]
    return bool(gating) and all(o.passed for o in gating)


__all__ = ["ValidationOutcome", "fea_is_validated", "run_validation"]
