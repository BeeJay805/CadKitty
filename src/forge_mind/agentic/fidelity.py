"""Fidelity check: is a part built with REAL construction, or is it primitives piled to fake a
part? (founder-directed 2026-07-14 -- the failure mode the project exists to kill.)

The deterministic gate never covered this: watertight + right-envelope + right-hole-count accepts a
box+cylinder pile. This check measures CONSTRUCTION richness from the B-rep (faces + curved surfaces,
via the STEP the sandbox already writes) plus the authored code's operations, and gates it RELATIVE
to the part's expected complexity -- a washer legitimately IS a ring, but a nozzle or gear built from
a bare cylinder is the failure. "A few primitives is fine": the gate fires only on a BARE pile (too
few faces, no curves, no shaping ops) for a part that should be complex; a detailed part (many faces
from patterns/fillets/teeth) passes even if every face is analytic. A continuous ``score`` is kept
for training-data ranking, but the hard gate is conservative (no false FAIL on simple/detailed parts).

Fail-closed: an unreadable B-rep yields verdict ``not_run`` (never a false FAIL). Confidence tier:
computed geometry (exact face census) + an estimated complexity heuristic; ships advisory, ratchets.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# --- expected-complexity lexicon (deterministic, matched on WHOLE words, not substrings) --------
# LOW: parts legitimately built from primitives -- fidelity is N/A, never failed.
_LOW = frozenset(
    [
        "washer",
        "spacer",
        "gasket",
        "shim",
        "ring",
        "disc",
        "disk",
        "plate",
        "slab",
        "block",
        "cube",
        "bar",
        "rod",
        "standoff",
        "bushing",
        "sleeve",
        "tube",
        "pipe",
        "pin",
        "peg",
        "dowel",
        "nut",
        "stopper",
        "plug",
        "puck",
        "coupon",
        "shackle",
        "bracket",
        "clip",
        "mount",
        "panel",
    ]
)
# HIGH: parts that DEMAND real (freeform / revolved / lofted / patterned / filleted) construction.
_HIGH = frozenset(
    [
        "nozzle",
        "impeller",
        "turbine",
        "blade",
        "propeller",
        "rotor",
        "airfoil",
        "vane",
        "volute",
        "turbopump",
        "manifold",
        "cowl",
        "fairing",
        "diffuser",
        "venturi",
        "organic",
        "ergonomic",
        "aerodynamic",
        "streamlined",
        "engine",
        "pump",
        "auger",
        "helix",
        "gyroid",
        "vase",
        "handle",
        "grip",
        "horn",
        "cam",
        "sprocket",
        "spinner",
        "scroll",
        "compressor",
        "gear",
        "worm",
        "cyclone",
        "propellor",
        "lattice",
        "sculpted",
        "contoured",
    ]
)

# authored-code operations that signal real construction (build123d idioms)
_ADV_OPS = re.compile(
    r"\b(loft|sweep|revolve|make_loft|make_sweep|fillet|chamfer|shell|offset|Bezier|Spline|"
    r"Ellipse|Helix|make_face|thicken|SlotArc|RadiusArc|TangentArc|section|involute)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z]+")

# tier -> the face-count floor below which, with no curves/ops, a part is a BARE primitive pile.
# low is never gated. Conservative: only a truly bare part for its complexity trips it.
_PILE_FLOOR = {"med": 6, "high": 16}


@dataclass(frozen=True)
class FidelityReport:
    verdict: str  # "pass" | "fail" | "n/a" | "not_run"
    score: float | None  # 0..1 construction richness (for training ranking), None if not_run
    expected: str  # "low" | "med" | "high"
    face_count: int
    primitive_frac: float | None  # analytic (plane/cyl/cone) share -- transparency only
    freeform_frac: float | None  # bspline/bezier/revolved/extruded share (real curves)
    blend_faces: int  # torus/sphere faces (fillet corners, rounds)
    advanced_code: bool  # code uses loft/sweep/revolve/fillet/...
    primitive_pile: bool  # the failure: bare, few faces, no curves, no shaping ops
    reason: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ANALYTIC = {"Plane", "Cylinder", "Cone"}
_ROUND = {"Sphere", "Torus"}
_FREEFORM = {"BSpline", "Bezier", "Revolution", "Extrusion", "Offset"}


def _face_census(step_path: Path) -> dict[str, int] | None:
    """Count B-rep faces by surface type from the STEP. None if unreadable (fail-closed)."""
    try:
        from build123d import import_step

        part = import_step(str(step_path))
        shape = getattr(part, "wrapped", None)
        if shape is None:
            return None
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import (
            GeomAbs_BezierSurface,
            GeomAbs_BSplineSurface,
            GeomAbs_Cone,
            GeomAbs_Cylinder,
            GeomAbs_OffsetSurface,
            GeomAbs_Plane,
            GeomAbs_Sphere,
            GeomAbs_SurfaceOfExtrusion,
            GeomAbs_SurfaceOfRevolution,
            GeomAbs_Torus,
        )
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS

        name = {
            GeomAbs_Plane: "Plane",
            GeomAbs_Cylinder: "Cylinder",
            GeomAbs_Cone: "Cone",
            GeomAbs_Sphere: "Sphere",
            GeomAbs_Torus: "Torus",
            GeomAbs_BezierSurface: "Bezier",
            GeomAbs_BSplineSurface: "BSpline",
            GeomAbs_SurfaceOfRevolution: "Revolution",
            GeomAbs_SurfaceOfExtrusion: "Extrusion",
            GeomAbs_OffsetSurface: "Offset",
        }
        census: dict[str, int] = {}
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            try:
                key = name.get(BRepAdaptor_Surface(face).GetType(), "Other")
            except Exception:
                key = "Other"
            census[key] = census.get(key, 0) + 1
            exp.Next()
        return census or None
    except Exception:
        return None


def expected_tier(text: str) -> str:
    """Expected construction complexity from the prompt / object name. Whole-word match so
    'camera' does NOT match 'cam' and 'proper' does NOT match 'prop'."""
    words = set(_WORD.findall((text or "").lower()))
    if words & _HIGH:
        return "high"
    if words & _LOW:
        return "low"
    return "med"


def _score(freeform_frac: float, round_frac: float, adv_code: bool, face_count: int) -> float:
    """0..1 construction richness (for training ranking, NOT the gate). Real curves dominate;
    fillets/rounds, shaping ops, and feature-detail (faces beyond a bare box) each add."""
    s = 0.40 * min(1.0, freeform_frac / 0.20)
    s += 0.20 * min(1.0, round_frac / 0.08)
    s += 0.20 * (1.0 if adv_code else 0.0)
    s += 0.20 * min(1.0, max(0, face_count - 6) / 34.0)
    return round(min(1.0, s), 3)


def fidelity_report(step_path: Path, code: str | None, prompt_or_name: str) -> FidelityReport:
    """Score a finished part's construction fidelity and gate it against its expected complexity."""
    expected = expected_tier(prompt_or_name)
    census = _face_census(step_path)
    adv_code = bool(code and _ADV_OPS.search(code))
    if not census:
        return FidelityReport(
            verdict="not_run",
            score=None,
            expected=expected,
            face_count=0,
            primitive_frac=None,
            freeform_frac=None,
            blend_faces=0,
            advanced_code=adv_code,
            primitive_pile=False,
            reason="no readable B-rep (fail-closed: not certified either way)",
            evidence={"census": census},
        )
    total = sum(census.values())
    prim = sum(census.get(k, 0) for k in _ANALYTIC)
    rnd = sum(census.get(k, 0) for k in _ROUND)
    free = sum(census.get(k, 0) for k in _FREEFORM)
    primitive_frac = round(prim / total, 3)
    round_frac = round(rnd / total, 3)
    freeform_frac = round(free / total, 3)
    score = _score(freeform_frac, round_frac, adv_code, total)

    # BARE PILE = no real curves, no fillets/rounds, no shaping ops, and too few faces for the tier.
    floor = _PILE_FLOOR.get(expected, 0)
    primitive_pile = freeform_frac < 0.02 and rnd == 0 and not adv_code and total <= floor

    if expected == "low":
        verdict = "n/a"
        reason = "simple part -- primitive construction is legitimate here"
    elif primitive_pile:
        verdict = "fail"
        reason = (
            f"{expected}-complexity part is a BARE primitive pile: {total} analytic faces, no "
            f"curves/fillets/shaping ops (floor {floor}) -- a primitive faking a real part"
        )
    else:
        verdict = "pass"
        reason = (
            f"{expected}-complexity part shows real construction "
            f"(faces {total}, freeform {freeform_frac}, rounds {rnd}, advanced_code {adv_code})"
        )
    return FidelityReport(
        verdict=verdict,
        score=score,
        expected=expected,
        face_count=total,
        primitive_frac=primitive_frac,
        freeform_frac=freeform_frac,
        blend_faces=rnd,
        advanced_code=adv_code,
        primitive_pile=primitive_pile,
        reason=reason,
        evidence={"census": census, "pile_floor": floor},
    )


__all__ = ["FidelityReport", "expected_tier", "fidelity_report"]
