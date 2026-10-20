"""The closed-form section author (promoted from scripts/arm_search/analytic_link.py; ADR-0097).

For a CLASSICAL structure -- a prismatic cantilever section -- minimum mass at a deflection bound
is beam theory, not a search: the LLM idiom search plateaued at ~130 mm^4/g while a load-aligned
I-section delivers ~690. This author sizes the section from Euler-Bernoulli, builds it WITH the
real bolt interface, FEA-corrects (E-B is ~36% optimistic here: shear deformation + shear lag --
the verifier catches what theory cannot see), records fail-closed buckling + yield margins,
measures BOTH bending axes (anisotropy honesty), and runs the PRINT GATES (min-wall + slicer) so
its emitted verdict is the reproducible record of the claim (Fable ruling: rerunning the author
must reproduce what the ADR recorded -- no off-script manual gates).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.optimize import brentq

from forge_mind.materials import mass_grams, resolve_material
from forge_mind.slicing.slicer import find_slicer_exe, slice_part
from forge_mind.structures.loadcase import (
    G_M_S2,
    LoadCase,
    canonical_link_case,
    run_load_case,
)
from forge_mind.validator.feature_checks import _measure_min_wall

_PLA_YIELD_MPA = 50.0  # conservative PLA tensile yield
_BUCKLE_MIN_MARGIN = 2.0  # fail-closed floor on plate-buckling / yield margins
_FEA_SCATTER_MM = 0.001  # row-73 mesh-convergence band; clear the bound by MORE than this


def i_required(force_n: float, length_mm: float, e_mpa: float, delta_mm: float) -> float:
    """Euler-Bernoulli cantilever: the I that holds tip deflection to ``delta_mm``."""
    return force_n * length_mm**3 / (3.0 * e_mpa * delta_mm)


def i_isection(tf: float, hy: float, wx: float, tw: float) -> float:
    """Second moment of a symmetric I-section about the bending axis (flanges at +/- hy/2)."""
    yb = hy / 2 - tf / 2
    return 2 * (wx * tf * yb**2 + wx * tf**3 / 12) + tw * (hy - 2 * tf) ** 3 / 12


def build_link(
    tf: float,
    *,
    length: float,
    hy: float,
    wx: float,
    tw: float,
    pad: float,
    hole_d: float,
    hole_sp: float,
) -> Any:
    """I-section link (flanges on the loaded/thin axis) + solid end pads carrying 2 bolt holes
    each. Single external shell (body_count 1). Long axis X; thin/load axis Y; wide axis Z."""
    from build123d import Box, Location, Pos

    bl = length - 2 * pad
    part = (
        (Pos(0, +(hy / 2 - tf / 2), 0) * Box(bl, tf, wx))
        + (Pos(0, -(hy / 2 - tf / 2), 0) * Box(bl, tf, wx))
        + Box(bl, hy - 2 * tf, tw)
    )
    for sx in (-1, 1):
        part += Pos(sx * (bl / 2 + pad / 2), 0, 0) * Box(pad, hy, wx)
        for sz in (-1, 1):
            part -= Location((sx * (bl / 2 + pad / 2), 0, sz * hole_sp / 2)) * Box(
                pad + 2, hole_d, hole_d
            )
    return part


def buckling_margins(
    tf: float,
    *,
    hy: float,
    wx: float,
    tw: float,
    force_n: float,
    length: float,
    e_mpa: float,
    nu: float,
) -> dict[str, float]:
    """Closed-form peak bending stress vs PLA yield, and the compression-flange plate-buckling
    critical stress (one-edge-free outstand, conservative k=0.425)."""
    ii = i_isection(tf, hy, wx, tw)
    sigma = force_n * length * (hy / 2) / ii  # M*c/I at the fixed base
    b = (wx - tw) / 2  # flange outstand
    sig_cr = 0.425 * float(np.pi) ** 2 * e_mpa / (12 * (1 - nu**2)) * (tf / b) ** 2
    return {
        "peak_stress_mpa": round(float(sigma), 4),
        "yield_margin": round(float(_PLA_YIELD_MPA / sigma), 1),
        "buckling_sig_cr_mpa": round(float(sig_cr), 2),
        "buckling_margin": round(float(sig_cr / sigma), 1),
    }


def print_gates(stl: Path, *, min_wall_required_mm: float) -> dict[str, Any]:
    """The deterministic print gates on a finished STL: measured min-wall (always available) and a
    real slice (tri-state: ``slice_ok`` None when no slicer is installed -- same polarity as the
    FEA gate: search fails open, a CLAIM requires True)."""
    mesh: Any = trimesh.load(str(stl), force="mesh")
    mw = _measure_min_wall(mesh)
    min_wall = None if mw is None else round(float(mw[0]), 3)
    gates: dict[str, Any] = {
        "min_wall_mm": min_wall,
        "min_wall_required_mm": min_wall_required_mm,
        "min_wall_pass": (min_wall is not None and min_wall >= min_wall_required_mm - 1e-6),
    }
    if find_slicer_exe() is None:
        gates["slice_ok"] = None  # no slicer on this box -> unproven, never claimed
        gates["slice_error"] = "slicer unavailable"
        return gates
    with tempfile.TemporaryDirectory(prefix="structures_slice_") as td:
        outcome = slice_part(stl, Path(td))
    gates["slice_ok"] = bool(outcome.ok)
    gates["slice_error"] = outcome.error
    return gates


def _strong_axis_case(mesh: Any) -> LoadCase:
    """The orthogonal (strong-axis) case, reported UNCLAIMED (open sections are anisotropic)."""
    c = canonical_link_case(mesh)
    strong_dir = (0.0, 1.0, 0.0) if c.load_dir == (1.0, 0.0, 0.0) else (1.0, 0.0, 0.0)
    return LoadCase(
        name="arm_link_v0_strong", force_n=c.force_n, load_dir=strong_dir, orient=c.orient
    )


def author_isection_link(
    *,
    ref_mass_g: float,
    ref_deflection_mm: float,
    length: float = 142.0,
    hy: float = 24.5,
    wx: float = 67.3,
    tw: float = 1.2,
    pad: float = 6.0,
    hole_d: float = 3.4,
    hole_sp: float = 20.0,
    payload_kg: float = 0.5,
    material_id: str = "pla",
    out_dir: Path,
    max_iters: int = 6,
) -> dict[str, Any]:
    """Author one section-optimized link: analytic size -> build -> FEA-correct with margin above
    the mesh scatter -> both axes -> buckling -> PRINT GATES. Emits the selection candidate schema
    (``selection.beats_reference`` consumes it) + full provenance."""
    from build123d import export_step, export_stl

    mat = resolve_material(material_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = ref_deflection_mm - 3 * _FEA_SCATTER_MM
    f_n, e_mpa, nu = payload_kg * G_M_S2, 3500.0, 0.35

    tf = float(
        brentq(
            lambda t: i_isection(t, hy, wx, tw) - i_required(f_n, length, e_mpa, target),
            0.3,
            hy / 2 - 0.5,
        )
    )
    history: list[dict[str, Any]] = []
    stl = out_dir / "analytic_link.stl"
    step = out_dir / "analytic_link.step"
    weak: float | None = None
    mesh: Any = None
    for _ in range(max_iters):
        part = build_link(
            tf, length=length, hy=hy, wx=wx, tw=tw, pad=pad, hole_d=hole_d, hole_sp=hole_sp
        )
        export_stl(part, str(stl))
        export_step(part, str(step))
        mesh = trimesh.load(str(stl), force="mesh")
        lr = run_load_case(mesh, canonical_link_case(mesh, payload_kg=payload_kg))
        weak = None if lr is None else lr.max_deflection_mm
        history.append({"tf": round(tf, 3), "weak_deflection_mm": weak})
        if weak is None or weak <= target:
            break
        tf *= weak / target  # delta ~ 1/I ~ 1/tf (flange-dominated)

    mesh = trimesh.load(str(stl), force="mesh")
    mass = mass_grams(float(mesh.volume), mat) if mesh.is_watertight else None
    lr_s = run_load_case(mesh, _strong_axis_case(mesh))
    strong = None if lr_s is None else lr_s.max_deflection_mm
    buck = buckling_margins(tf, hy=hy, wx=wx, tw=tw, force_n=f_n, length=length, e_mpa=e_mpa, nu=nu)
    buckling_ok = (
        buck["buckling_margin"] >= _BUCKLE_MIN_MARGIN and buck["yield_margin"] >= _BUCKLE_MIN_MARGIN
    )
    gates = print_gates(stl, min_wall_required_mm=mat.min_wall_mm)

    within = None if weak is None else bool(weak <= ref_deflection_mm)
    lighter = None if mass is None else bool(mass < ref_mass_g)
    record: dict[str, Any] = {
        "author": "analytic_isection",
        "built": bool(mesh.is_watertight and mass is not None),
        # geometry_passed = the deterministic gates that ALWAYS run (watertight, buckling,
        # min-wall). slice_ok rides separately (tri-state; a claim requires True).
        "geometry_passed": bool(mesh.is_watertight and buckling_ok and gates["min_wall_pass"]),
        "watertight": bool(mesh.is_watertight),
        "mass_g": None if mass is None else round(mass, 2),
        "deflection_mm": weak,
        "within_deflection_bound": within,
        "lighter_than_reference": lighter,
        "mass_vs_reference_pct": None if mass is None else round(100 * (mass / ref_mass_g - 1), 1),
        "strong_axis_deflection_mm": strong,  # reported UNCLAIMED
        "buckling": buck,
        "buckling_ok": buckling_ok,
        "print_gates": gates,
        "tf_mm": round(tf, 3),
        "target_with_margin_mm": round(target, 4),
        "reference": {"mass_g": ref_mass_g, "deflection_mm": ref_deflection_mm},
        "stl": str(stl),
        "step": str(step),
        "history": history,
    }
    (out_dir / "analytic_link_report.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


__all__ = [
    "author_isection_link",
    "buckling_margins",
    "build_link",
    "i_isection",
    "i_required",
    "print_gates",
]
