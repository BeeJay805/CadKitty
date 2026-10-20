"""FitProof: evaluate a part's detected holes against the hardware the user says they mate with.

Deterministic, decision-grade, no LLM. A mate spec (``{kind, spec}`` from intent) is parsed to a target
diameter + fit rule; the nearest detected hole is checked against it with cited standards:
  - bolt/screw  -> ISO 273 clearance (the bolt must pass freely, not sloppily)
  - shaft       -> ISO 286 fit (clearance/interference) via the resolve_fit engine
  - bearing     -> the part hole is the bearing SEAT (housing bore == bearing OD), light-press target

Fail-closed: an unparseable spec, or no hole near the target, returns ``None`` (the projector emits
not_run) — never a fabricated pass. The part hole is MEASURED; the mate is the STATED nominal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from forge_mind.dsl.standards.fasteners import FASTENERS, clearance_hole
from forge_mind.physics.fits import FitError, parse_designation, resolve_fit
from forge_mind.physics.report.holes import DetectedHole
from forge_mind.physics.report.schema import Comparator, PhysicsStatus

# Common deep-groove ball bearings: model -> (bore, outer diameter, width) mm. The part hole that
# "mates with a 608" is the housing seat for the bearing's OUTER diameter.
_BEARINGS: dict[str, tuple[float, float, float]] = {
    "623": (3.0, 10.0, 4.0),
    "624": (4.0, 13.0, 5.0),
    "625": (5.0, 16.0, 5.0),
    "626": (6.0, 19.0, 6.0),
    "627": (7.0, 22.0, 7.0),
    "608": (8.0, 22.0, 7.0),
    "629": (9.0, 26.0, 8.0),
    "6000": (10.0, 26.0, 8.0),
    "6001": (12.0, 28.0, 8.0),
    "6002": (15.0, 32.0, 9.0),
}

_FASTENER_SIZES = ("M2", "M2.5", "M3", "M4", "M5", "M6", "M8")
# A detected hole must be within this of the target diameter to be considered "the" mating hole.
_NEAR_TOL_MM = 1.5


@dataclass(frozen=True)
class FitEval:
    """The result of checking one mate against the nearest detected hole — enough to build a Finding."""

    label: str
    status: PhysicsStatus
    measured_mm: float
    limit_mm: float
    comparator: Comparator
    source: str  # cited standard
    hole: DetectedHole  # the located hole the verdict is about
    explanation: str
    fix_text: str | None
    method: str


def _nearest_hole(holes: list[DetectedHole], target_mm: float) -> DetectedHole | None:
    near = [h for h in holes if abs(h.diameter_mm - target_mm) <= _NEAR_TOL_MM]
    if not near:
        return None
    return min(near, key=lambda h: abs(h.diameter_mm - target_mm))


def _eval_bolt(spec: str, holes: list[DetectedHole]) -> FitEval | None:
    """A clearance hole for size ``spec`` (e.g. 'M5'): the bolt must pass (>= ISO 273 clearance) and not
    be sloppily oversized (> 2x thread diameter)."""
    size = spec.strip().upper().replace(" ", "")
    if size not in _FASTENER_SIZES:
        return None
    f = FASTENERS.get(size)
    clearance = clearance_hole(size)  # ISO 273 normal
    target = clearance
    hole = _nearest_hole(holes, target)
    if hole is None:
        return None
    d = hole.diameter_mm
    if d < f.clearance_close_mm - 0.05:
        status: PhysicsStatus = "fail"
        explanation = (
            f"The nearest hole is {d} mm — too small for an {size} bolt to pass "
            f"(needs >= {clearance} mm clearance)."
        )
        fix: str | None = f"Open this hole to {clearance} mm so the {size} bolt clears."
    elif d > 2.0 * f.nominal_diameter_mm:
        status = "warn"
        explanation = (
            f"The hole is {d} mm — far larger than an {size} clearance hole ({clearance} mm); the head "
            "may pull through and the bolt won't locate."
        )
        fix = f"Reduce this hole toward {clearance} mm, or use a washer / larger bolt."
    else:
        status = "pass"
        explanation = f"An {size} bolt passes through the {d} mm hole with a clean clearance fit."
        fix = None
    return FitEval(
        label=f"{size} clearance",
        status=status,
        measured_mm=d,
        limit_mm=clearance,
        comparator=">=",
        source=f"ISO 273 {size} clearance",
        hole=hole,
        explanation=explanation,
        fix_text=fix,
        method="bore diameter vs ISO 273 clearance",
    )


def _eval_shaft(spec: str, holes: list[DetectedHole]) -> FitEval | None:
    """A shaft fit. ``spec`` is an ISO designation ('8H7/h6'); the hole (bore) is measured, the shaft is
    the stated half. We report the ACTUAL clearance of the measured bore over the shaft's max diameter."""
    try:
        nominal, hole_code, shaft_code = parse_designation(spec)
    except FitError:
        return None
    try:
        fit = resolve_fit(nominal, hole_code, shaft_code)
    except Exception:
        return None
    hole = _nearest_hole(holes, nominal)
    if hole is None:
        return None
    shaft_max_mm = fit.shaft_limits_mm[1]
    actual_clear_mm = round(hole.diameter_mm - shaft_max_mm, 3)
    if actual_clear_mm < 0:
        status: PhysicsStatus = "fail"
        explanation = (
            f"The measured bore {hole.diameter_mm} mm is smaller than the {fit.designation} shaft's max "
            f"{round(shaft_max_mm, 3)} mm — interference, it will not slide on."
        )
        fix: str | None = (
            f"Open the bore to at least {round(shaft_max_mm, 3)} mm for a {fit.designation} fit."
        )
    elif actual_clear_mm > 0.2 + nominal * 0.02:
        status = "warn"
        explanation = (
            f"The bore {hole.diameter_mm} mm gives {actual_clear_mm} mm clearance over the "
            f"{fit.designation} shaft — looser than the intended fit; it may rattle."
        )
        fix = f"Tighten the bore toward {round(shaft_max_mm, 3)} mm for the {fit.designation} fit."
    else:
        status = "pass"
        explanation = (
            f"The bore {hole.diameter_mm} mm clears the {fit.designation} shaft by {actual_clear_mm} mm — "
            "a sound running fit."
        )
        fix = None
    return FitEval(
        label=f"{fit.designation} shaft fit",
        status=status,
        measured_mm=hole.diameter_mm,
        limit_mm=round(shaft_max_mm, 3),
        comparator=">=",
        source="ISO 286 (computed, validated vs published tables)",
        hole=hole,
        explanation=explanation,
        fix_text=fix,
        method="measured bore vs ISO 286 shaft limit",
    )


def _eval_bearing(spec: str, holes: list[DetectedHole]) -> FitEval | None:
    """The part hole is the bearing SEAT (housing bore). spec is a bearing model ('608') or 'BORExODxW'.
    A good seat holds the bearing's OUTER diameter with near-zero clearance (light press / H7)."""
    s = spec.strip().lower().replace(" ", "")
    od: float | None = None
    m = re.match(r"^\d+(?:\.\d+)?x(\d+(?:\.\d+)?)x\d+(?:\.\d+)?$", s)
    if m:
        od = float(m.group(1))
    else:
        model = re.sub(r"[^0-9]", "", s)
        if model in _BEARINGS:
            od = _BEARINGS[model][1]
    if od is None:
        return None
    hole = _nearest_hole(holes, od)
    if hole is None:
        return None
    gap = round(hole.diameter_mm - od, 3)
    if gap < -0.05:
        status: PhysicsStatus = "warn"  # tight press is workable but heavy
        explanation = (
            f"The seat bore {hole.diameter_mm} mm is {abs(gap)} mm under the bearing OD {od} mm — a heavy "
            "press; it may distort the bearing."
        )
        fix: str | None = f"Open the seat toward {od} mm for a light-press bearing fit."
    elif gap > 0.1:
        status = "fail"
        explanation = (
            f"The seat bore {hole.diameter_mm} mm is {gap} mm over the bearing OD {od} mm — the bearing "
            "is loose and will spin in the housing."
        )
        fix = f"Reduce the seat to {od} mm (light press) so the bearing holds."
    else:
        status = "pass"
        explanation = (
            f"The {hole.diameter_mm} mm seat holds the {od} mm bearing OD with a sound fit."
        )
        fix = None
    return FitEval(
        label="bearing seat",
        status=status,
        measured_mm=hole.diameter_mm,
        limit_mm=od,
        comparator="==",
        source="bearing OD seat (light press)",
        hole=hole,
        explanation=explanation,
        fix_text=fix,
        method="seat bore vs bearing OD",
    )


def evaluate_mate(kind: str, spec: str, holes: list[DetectedHole]) -> FitEval | None:
    """Dispatch a mate to its fit check. None = unparseable spec or no matching hole (-> not_run)."""
    k = kind.strip().lower()
    if k in {"bolt", "screw", "fastener"}:
        return _eval_bolt(spec, holes)
    if k in {"shaft", "pin", "rod", "dowel"}:
        return _eval_shaft(spec, holes)
    if k in {"bearing"}:
        return _eval_bearing(spec, holes)
    return None


__all__ = ["FitEval", "evaluate_mate"]
