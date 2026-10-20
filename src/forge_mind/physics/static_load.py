"""Requirement-driven static-load estimate for the L-bracket — SPIKE (ADR-0039).

Council-vetted v0 (2026-06-10, three-seat council: mechanical-engineer /
reliability-wedge / FEA-feasibility; majority verdict): a deterministic
TWO-SEGMENT small-deflection model of the bench L-bracket under a face-mounted
mass, plus the sizing inversion ("you stated the load; code sized the wall").
The council's unanimous corrections to the existing ``stiffness.py`` shape:

- upright-only Euler-Bernoulli FLATTERS the design — both segments (base-plate
  bending AND upright bending + root rotation) plus shear correction are
  modeled;
- a point value is false precision — the estimate is a RANGE over
  printed-coupon E values, two significant figures, assumptions inline;
- refuse-not-guess — unknown material, non-bracket geometry, or an
  out-of-envelope model emits REFUSALS, never a number (the no-false-PASS
  doctrine applied to physics);
- STRENGTH IS NEVER REPORTED. FDM layer adhesion is orientation- and
  slicer-dependent; a stress/FoS claim would be wrong-but-confident.

REPORTED-ONLY AND UNWIRED: this module is called by nothing in the product
(no validate(), no API path) — the strongest possible flag. The payload is
machine-honest (``estimate=True, gates=False``) so a future consumer cannot
wire it into a gate by accident. The FEM second method (netgen LGPL +
scikit-fem BSD, bracketed BCs) is SPEC'D in ADR-0039, deliberately not built
tonight (the FEA seat's dissent is recorded there).

Geometry convention (the bench L-bracket, ``layout/interpreter.py``):
``motor_flange`` = the HORIZONTAL base plate (the NEMA17 face: 4x M3 + shaft
bore), ``ext_flange`` = the VERTICAL upright at the rear carrying the 2x M5
extrusion bolts. The motor + payload weight enters at the shaft-bore center
and the plate cantilevers from the upright root; the upright bends between
the M5 bolt line and the root, and its root rotation re-projects to the face.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.primitives import Box
from forge_mind.dsl.program import PartProgram

G_M_S2 = 9.81

# Printed-coupon Young's modulus RANGES (MPa) — deliberately not the bulk datasheet values
# (injection-molded, optimistic) and deliberately a range, not a point: FDM prints vary with
# slicer, infill, and humidity. Council seat A's values (2026-06-10). Nylon is the DRY range;
# moisture roughly halves it (a named assumption when used).
PRINTED_E_RANGE_MPA: dict[str, tuple[float, float]] = {
    "PLA": (2000.0, 3500.0),
    "PETG": (1800.0, 2100.0),
    "ABS": (1800.0, 2400.0),
    "NYLON": (1200.0, 1700.0),
}

# Typical 40mm-body NEMA17 mass (kg). The component registry carries no mass yet (flagged in
# ADR-0039 as the v1 registry addition); this is a cited data-table constant, listed in the
# assumptions of every estimate that uses it.
NEMA17_MASS_KG = 0.28

# Manufacturable wall increment the sizing inversion snaps UP to.
_SIZE_INCREMENT_MM = 0.5
_SIZE_MAX_MM = 20.0

# Timoshenko shear correction for a rectangular section; G ~ E / 2.6 (nu ~ 0.3).
_SHEAR_K = 5.0 / 6.0
_G_OVER_E = 1.0 / 2.6

_METHOD_ID = "two_segment_bracket_v0"

_BASE_ASSUMPTIONS = [
    "Static dead load only: F = (motor + payload) * g applied at the shaft-bore center, "
    "normal to the base plate (worst static case for the flange).",
    "Two-segment small-deflection model: base-plate bending + upright bending + upright "
    "root rotation, each with Timoshenko shear correction (G = E/2.6).",
    "The M5 bolt line is treated as built-in (washer clamping not modeled; a bolted joint "
    "is somewhat softer -- real deflection can exceed this estimate).",
    "Solid, homogeneous, isotropic section at 23 C -- infill, fillets, holes, and stress "
    "concentrations are not modeled.",
    "The corner gusset (when present) is IGNORED -- the estimate is conservative (stiffer "
    "real part).",
    "E is a printed-coupon range, not a datasheet bulk value.",
]

_DISCLAIMER = (
    "Estimate for design guidance only; not a substitute for engineering analysis; "
    "verify before load-bearing use."
)


class StaticLoadCase(BaseModel):
    """The user-stated requirement: what hangs on the bracket (the LLM picks none of this)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    payload_kg: float = Field(..., ge=0.0, description="User-stated payload mass (kg).")
    motor_kg: float = Field(
        default=NEMA17_MASS_KG,
        gt=0.0,
        description="Motor mass (kg); default = typical 40mm NEMA17 (cited constant).",
    )


class StaticLoadEstimate(BaseModel):
    """The machine-honest payload (council framing, unanimous): an ESTIMATE that can never be
    read -- by a person or a program -- as a validation verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: Literal[True] = True
    gates: Literal[False] = False  # structurally never a gate
    method: str = _METHOD_ID
    load_n: float
    # Deflection at the motor face, RANGE over the printed-coupon E range (mm, 2 sig figs).
    deflection_mm_lo: float  # stiff end (E high)
    deflection_mm_hi: float  # soft end (E low)
    # Per-segment breakdown at NOMINAL E (mm) -- the shown work that makes it auditable.
    e_nominal_mpa: float
    plate_bending_mm: float
    plate_shear_mm: float
    upright_bending_mm: float
    upright_shear_mm: float
    root_rotation_mm: float
    assumptions: list[str]
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = _DISCLAIMER
    # Set by the sizing inversion only.
    sized_field: str | None = None
    sized_value_mm: float | None = None
    sizing_target_mm: float | None = None


class StaticLoadRefusal(BaseModel):
    """An honest 'cannot estimate this' -- the no-false-PASS doctrine applied to physics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: Literal[False] = False
    gates: Literal[False] = False
    method: str = _METHOD_ID
    reasons: list[str]


class _BracketDims(BaseModel):
    """Measured off the INTERPRETED program ops (the real geometry, not the spec estimate)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_width_mm: float  # X: bending section width of the base plate
    base_depth_mm: float  # Y: cantilever run direction
    upright_width_mm: float  # X: bending section width of the upright
    upright_height_mm: float  # Z
    wall_mm: float  # both plate thicknesses (the spec's min_wall)
    bolt_line_z_mm: float  # M5 row height (upright fixity line)
    has_gusset: bool


def _find_box(program: PartProgram, op_id: str) -> Box | None:
    for op in program.operations:
        if op.op_id == op_id and isinstance(op, Box):
            return op
    return None


def extract_bracket_dims(program: PartProgram) -> _BracketDims | None:
    """The L-bracket dims, measured from the interpreted ops. None if this is not the
    bench L-bracket shape (motor_flange + ext_flange) -- the caller refuses."""
    base = _find_box(program, "motor_flange")
    upright = _find_box(program, "ext_flange")
    if base is None or upright is None:
        return None
    bolt_z: float | None = None
    for op in program.operations:
        if op.op_id in ("ext_bolt_left", "ext_bolt_right"):
            bolt_z = float(op.position.z)
            break
    if bolt_z is None:
        return None
    return _BracketDims(
        base_width_mm=base.width,
        base_depth_mm=base.depth,
        upright_width_mm=upright.width,
        upright_height_mm=upright.height,
        wall_mm=base.height,  # the base plate thickness == spec min_wall
        bolt_line_z_mm=bolt_z,
        has_gusset=any(op.op_id == "corner_gusset" for op in program.operations),
    )


def _deflection_mm(dims: _BracketDims, f_n: float, e_mpa: float) -> dict[str, float]:
    """Two-segment closed form: deflection components (mm) at the motor-face center.

    Segment 1 (base plate): cantilever of run L_b from the upright root to the shaft-bore
    center, section (base_width x wall). Segment 2 (upright): between the M5 bolt line and
    the base root, section (upright_width x wall), loaded by the shear F and the root moment
    M = F * L_b; its tip deflection and rotation re-project to the face (delta + theta*L_b).
    All MPa/mm/N (1 MPa = 1 N/mm^2).
    """
    w = dims.wall_mm
    g_mpa = e_mpa * _G_OVER_E

    # Base plate: root at the upright front face (y = rear - wall), load at y = 0.
    l_b = dims.base_depth_mm / 2.0 - w
    i_b = dims.base_width_mm * w**3 / 12.0
    a_b = dims.base_width_mm * w
    plate_bending = f_n * l_b**3 / (3.0 * e_mpa * i_b)
    plate_shear = f_n * l_b / (_SHEAR_K * g_mpa * a_b)

    # Upright: fixity at the M5 bolt line, root at the base mid-plane (z = wall/2).
    l_u = max(dims.bolt_line_z_mm - w / 2.0, w)  # never below one wall (stubby floor)
    i_u = dims.upright_width_mm * w**3 / 12.0
    a_u = dims.upright_width_mm * w
    m_root = f_n * l_b
    upright_bending = f_n * l_u**3 / (3.0 * e_mpa * i_u) + m_root * l_u**2 / (2.0 * e_mpa * i_u)
    upright_shear = f_n * l_u / (_SHEAR_K * g_mpa * a_u)
    theta = f_n * l_u**2 / (2.0 * e_mpa * i_u) + m_root * l_u / (e_mpa * i_u)
    root_rotation = theta * l_b

    return {
        "plate_bending": plate_bending,
        "plate_shear": plate_shear,
        "upright_bending": upright_bending,
        "upright_shear": upright_shear,
        "root_rotation": root_rotation,
        "total": plate_bending + plate_shear + upright_bending + upright_shear + root_rotation,
    }


def _round2(x: float) -> float:
    """Two significant figures -- the council's false-precision guard."""
    if x == 0.0:
        return 0.0
    return float(f"{x:.2g}")


def _material_range(material: str) -> tuple[float, float] | None:
    return PRINTED_E_RANGE_MPA.get(material.strip().upper().split()[0] if material else "")


def estimate_static_load(
    program: PartProgram,
    load: StaticLoadCase,
    material: str,
) -> StaticLoadEstimate | StaticLoadRefusal:
    """Forward estimate: deflection range at the motor face for the stated load.

    Refuses (never guesses) on: a non-bracket program, an unknown material, or degenerate
    dims. Strength is structurally absent from the payload.
    """
    reasons: list[str] = []
    dims = extract_bracket_dims(program)
    if dims is None:
        reasons.append(
            "not the bench L-bracket shape (motor_flange + ext_flange + M5 bolts); the "
            "two-segment model only claims that geometry."
        )
    e_range = _material_range(material)
    if e_range is None:
        reasons.append(
            f"no printed-coupon stiffness data for material {material!r} "
            f"(known: {sorted(PRINTED_E_RANGE_MPA)}); refusing rather than guessing."
        )
    if dims is not None and (dims.base_depth_mm / 2.0 - dims.wall_mm) <= dims.wall_mm:
        reasons.append("base plate run is shorter than one wall -- outside the beam model.")
    if reasons:
        return StaticLoadRefusal(reasons=reasons)
    assert dims is not None and e_range is not None

    f_n = (load.payload_kg + load.motor_kg) * G_M_S2
    e_lo, e_hi = e_range
    e_nom = (e_lo + e_hi) / 2.0
    parts_nom = _deflection_mm(dims, f_n, e_nom)
    d_lo = _deflection_mm(dims, f_n, e_hi)["total"]  # stiff end
    d_hi = _deflection_mm(dims, f_n, e_lo)["total"]  # soft end

    warnings: list[str] = []
    mat = material.strip().upper()
    if mat.startswith("PLA"):
        warnings.append(
            "PLA softens at 50-80 C; a working stepper face can reach that range. For a "
            "motor mount under sustained load prefer PETG/ABS (named council refusal-case)."
        )
    if mat.startswith("NYLON"):
        warnings.append("Nylon E is the DRY range; moisture saturation roughly halves it.")
    if dims.has_gusset:
        warnings.append("Corner gusset present and ignored: the real part is stiffer.")
    warnings.append(
        "No strength/failure claim is made or derivable: FDM layer adhesion is orientation- "
        "and slicer-dependent."
    )

    assumptions = [
        f"Load: motor {load.motor_kg:.3f} kg "
        + (
            "(typical 40mm NEMA17, registry carries no mass yet)"
            if load.motor_kg == NEMA17_MASS_KG
            else "(stated)"
        )
        + f" + payload {load.payload_kg:.3f} kg -> F = {f_n:.2f} N.",
        f"E({mat}) = {e_lo:.0f}-{e_hi:.0f} MPa (printed coupon range; nominal {e_nom:.0f}).",
        *_BASE_ASSUMPTIONS,
    ]

    return StaticLoadEstimate(
        load_n=round(f_n, 2),
        deflection_mm_lo=_round2(d_lo),
        deflection_mm_hi=_round2(d_hi),
        e_nominal_mpa=e_nom,
        plate_bending_mm=_round2(parts_nom["plate_bending"]),
        plate_shear_mm=_round2(parts_nom["plate_shear"]),
        upright_bending_mm=_round2(parts_nom["upright_bending"]),
        upright_shear_mm=_round2(parts_nom["upright_shear"]),
        root_rotation_mm=_round2(parts_nom["root_rotation"]),
        assumptions=assumptions,
        warnings=warnings,
    )


def size_wall_for_deflection(
    program: PartProgram,
    load: StaticLoadCase,
    material: str,
    max_deflection_mm: float,
    interpret: Any,
    spec: Any,
) -> StaticLoadEstimate | StaticLoadRefusal:
    """Sizing inversion (v1 of the council ladder): find the smallest manufacturable
    ``min_wall_mm`` whose RE-INTERPRETED bracket meets the deflection target at nominal E.

    The wedge-seat rule: size on NOMINAL E (sizing from the worst edge of the band makes
    every wall comically thick); report the full band for the sized part. The sized value is
    applied through the SAME parametric model (the spec's ``min_wall_mm``) and the part is
    re-derived by the REAL interpreter, then forward-verified -- never an interpolation.
    ``interpret``/``spec`` are passed in so this module stays import-light (no layout dep).
    """
    if max_deflection_mm <= 0:
        return StaticLoadRefusal(reasons=["deflection target must be positive."])
    base = estimate_static_load(program, load, material)
    if isinstance(base, StaticLoadRefusal):
        return base

    def _nominal_total(p: PartProgram) -> float | None:
        d = extract_bracket_dims(p)
        if d is None:
            return None
        e_range = _material_range(material)
        assert e_range is not None
        f_n = (load.payload_kg + load.motor_kg) * G_M_S2
        return _deflection_mm(d, f_n, (e_range[0] + e_range[1]) / 2.0)["total"]

    current = extract_bracket_dims(program)
    assert current is not None
    wall = current.wall_mm
    sized_program: PartProgram | None = None
    sized_wall: float | None = None
    while wall <= _SIZE_MAX_MM:
        # Null the explicit plate dims so the interpreter re-derives the tightest LEGAL
        # dims for each candidate wall (a thicker wall needs more hole-to-edge land; pinned
        # dims would make growth infeasible). This is the parametric model working as
        # designed -- the dependent re-derivation is reported on the sized estimate.
        candidate_spec = spec.model_copy(
            update={"min_wall_mm": wall, "base": None, "upright": None}
        )
        try:
            candidate = interpret(candidate_spec)
        except Exception as exc:
            return StaticLoadRefusal(
                reasons=[f"sizing made the spec infeasible at wall={wall}mm: {exc}"]
            )
        total = _nominal_total(candidate)
        if total is None:
            return StaticLoadRefusal(reasons=["re-derived part lost the bracket shape."])
        if total <= max_deflection_mm:
            sized_program, sized_wall = candidate, wall
            break
        wall = round(wall + _SIZE_INCREMENT_MM, 3)
    if sized_program is None or sized_wall is None:
        return StaticLoadRefusal(
            reasons=[
                f"no wall up to {_SIZE_MAX_MM}mm meets {max_deflection_mm}mm at nominal E -- "
                "the requirement needs a different design, not a thicker wall."
            ]
        )

    verified = estimate_static_load(sized_program, load, material)
    if isinstance(verified, StaticLoadRefusal):
        return verified
    return verified.model_copy(
        update={
            "sized_field": "min_wall_mm",
            "sized_value_mm": sized_wall,
            "sizing_target_mm": max_deflection_mm,
            "warnings": [
                *verified.warnings,
                "Sizing re-derived the dependent plate dims (hole-to-edge lands grow with "
                "the wall); the sized part is the re-interpreted geometry, not the original "
                "with a thicker wall.",
            ],
        }
    )


__all__ = [
    "NEMA17_MASS_KG",
    "PRINTED_E_RANGE_MPA",
    "StaticLoadCase",
    "StaticLoadEstimate",
    "StaticLoadRefusal",
    "estimate_static_load",
    "extract_bracket_dims",
    "size_wall_for_deflection",
]
