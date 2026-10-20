"""Validated-FEA load cases for structural parts (promoted from scripts/arm_search/fea_probe.py).

A ``LoadCase`` pins force + direction + material + orientation as DATA so the reference and every
candidate get the IDENTICAL case (the ADR-0096 fair-comparison requirement). It composes
``physics.fea.estimate_load_response`` (crash-isolated, gated on ``fea_is_validated``) and never
emits a number the solver did not earn: unavailable/failed FEA -> ``None``, fail-closed.

Contract note -- why ``orient`` exists: ``estimate_load_response`` FIXES the z-min face and spreads
the load over the z-max face. A ``LoadCase`` realizes "this end fixed, that end loaded" by carrying
a 4x4 transform that puts the fixed port at z-min; identity means the mesh already arrives in that
convention (base-down, Z-up).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from forge_mind.physics.fea import (
    DEFAULT_E_MPA,
    DEFAULT_NU,
    LoadResponse,
    estimate_load_response,
)
from forge_mind.physics.fea_validation import fea_is_validated, run_validation

G_M_S2 = 9.80665  # standard gravity, for payload_kg -> newtons

# 4x4 rotations mapping each AABB long axis onto +Z (base translated to z=0 separately).
_ROT_LONG_TO_Z: dict[int, np.ndarray] = {
    0: np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float),  # +X -> +Z
    1: np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float),  # +Y -> +Z
    2: np.eye(4),  # already along Z
}

Orient = tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class LoadCase:
    """A named, decision-grade load case for one part (the stiffness objective's input)."""

    name: str
    force_n: float
    load_dir: tuple[float, float, float] = (0.0, 0.0, -1.0)
    e_mpa: float = DEFAULT_E_MPA
    nu: float = DEFAULT_NU
    edge_length_fac: float = 0.05
    orient: Orient | None = None
    # A PATCH / POINT load: ((px, py), r) in the loaded-face plane (post-orient z-max frame). None
    # spreads the force over the WHOLE top face (today's behaviour -- byte-identical). A patch
    # concentrates the total force on the facets within radius r of (px, py): an off-centre patch
    # produces the torsion+bending a face-spread averages away -- the gap-c fix (a part DESIGNED for
    # a corner point load must be MEASURED under one, or the autotuner bisects a misaligned case).
    load_patch: tuple[tuple[float, float], float] | None = None


# Illustrative cantilever case for smokes/probes. NOT a frozen claim -- benchmarks (plan §6) pin
# the real payload/direction/material when a north star is locked.
SMOKE_CANTILEVER_5N = LoadCase(name="smoke_cantilever_5N", force_n=5.0, load_dir=(1.0, 0.0, 0.0))


def run_load_case(mesh: Any, case: LoadCase) -> LoadResponse | None:
    """Apply ``case`` to a Z-up trimesh; ``None`` is the fail-closed outcome (solver not validated
    on this box, or the crash-isolated solve produced no result). ``orient`` applies to a copy."""
    if case.orient is not None:
        mesh = mesh.copy()
        mesh.apply_transform(np.asarray(case.orient, dtype=float))
    return estimate_load_response(
        mesh,
        case.force_n,
        load_dir=case.load_dir,
        e_mpa=case.e_mpa,
        nu=case.nu,
        edge_length_fac=case.edge_length_fac,
        load_patch=case.load_patch,
    )


def canonical_link_orient(mesh: Any) -> np.ndarray:
    """Deterministic 4x4 standing a link up for the cantilever case: longest AABB axis -> +Z,
    base (min-Z) translated to z=0. Same pose for reference and candidate; geometry-derived, so
    reproducible without per-part frame annotations."""
    long_axis = int(np.argmax(mesh.extents))
    rot = _ROT_LONG_TO_Z[long_axis]
    m2 = mesh.copy()
    m2.apply_transform(rot)
    trans = np.eye(4)
    trans[2, 3] = -float(m2.bounds[0][2])  # drop the base to z=0
    result: np.ndarray = trans @ rot
    return result


def canonical_link_case(
    mesh: Any, *, payload_kg: float = 0.5, name: str = "arm_link_v0"
) -> LoadCase:
    """The PROPOSED canonical arm-link stiffness case (plan §3; v0, pending the §6 human lock).

    Longest axis vertical, base fixed at z-min; ``payload_kg`` transverse at the far end along the
    THINNER cross-section (weak-axis bending -- the design-discriminating direction), PLA modulus.
    Deterministic from the mesh, so reference and candidate get the identical case."""
    orient = canonical_link_orient(mesh)
    m2 = mesh.copy()
    m2.apply_transform(orient)
    ex, ey = float(m2.extents[0]), float(m2.extents[1])
    load_dir = (1.0, 0.0, 0.0) if ex <= ey else (0.0, 1.0, 0.0)
    return LoadCase(
        name=name,
        force_n=payload_kg * G_M_S2,
        load_dir=load_dir,
        orient=cast(Orient, tuple(tuple(float(v) for v in row) for row in orient)),
    )


def describe_load_case(case: LoadCase) -> str:
    """The case in HUMAN units -- printed on every report (Fable's mitigation for the load-case
    single point of failure: a human can spot '50 kg' where '0.5 kg' was meant; a JSON blob of
    newtons hides it)."""
    kg = case.force_n / G_M_S2
    axis = {(1.0, 0.0, 0.0): "X", (0.0, 1.0, 0.0): "Y", (0.0, 0.0, -1.0): "-Z"}.get(
        case.load_dir, str(case.load_dir)
    )
    oriented = (
        "part stood on its long axis, base fixed"
        if case.orient is not None
        else "as-modeled orientation, z-min face fixed"
    )
    where = (
        "over the whole loaded face"
        if case.load_patch is None
        else f"on a patch (r={case.load_patch[1]:.1f} mm) at {case.load_patch[0]}"
    )
    return (
        f"{case.name}: {kg:.2f} kg ({case.force_n:.2f} N) applied at the free end along {axis} "
        f"{where}; {oriented}; E={case.e_mpa:.0f} MPa."
    )


def probe_env() -> dict[str, Any]:
    """Is the validated-FEA gate up on this box? Surfaces per-case rel-errors so a solver
    regression is visible, not just a red bool."""
    return {
        "fea_is_validated": fea_is_validated(),
        "validation_cases": [
            {
                "case": o.case,
                "quantity": o.quantity,
                "rel_error": o.rel_error,
                "tolerance": o.tolerance,
                "passed": o.passed,
            }
            for o in run_validation()
        ],
    }


__all__ = [
    "G_M_S2",
    "SMOKE_CANTILEVER_5N",
    "LoadCase",
    "LoadResponse",
    "Orient",
    "canonical_link_case",
    "canonical_link_orient",
    "probe_env",
    "run_load_case",
]
