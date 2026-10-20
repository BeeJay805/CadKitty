"""The honest confidence + fitness label — COMPUTED, never composed (ADR-0041).

Every result carries a verdict the user (and any API consumer) can trust: overall
confidence, basis, manufacturability, the dimension-tier mix (ADR-0040), which
dimensions to verify, and what the part is NOT fit for. The council's laws:

- **Computed, never composed** (skeptic seat): every field is a closed enum or a
  table-selected canned string. The LLM picks NOTHING here — the label is derived
  from evidence (the judge's structured verdict, the geometry gates, the earned
  dimension tiers, the part-class table). Free text cannot inflate or soften it.
- **Weakest link wins** (methodology seat): overall confidence is the MIN over
  independent axes (dimension floor, judge cap, gates cap) — total by construction,
  monotone by construction: degrading any axis can never raise the label.
- **Functional-critical caps at conceptual**: geometric fidelity is not functional
  fidelity. A part whose identity demands balance / fatigue / pressure rating /
  certification (a crankshaft, a pressure manifold, a carabiner) stays `conceptual`
  until something functional is verified — however good it looks.
- **The middle is actionable** (skeptic seat): an approximate part names WHICH
  dimensions to verify (`verify_dimensions` -> the frontend's /patch deep-links),
  not a disclaimer to tune out.
- **No quantified performance claims, ever**: there is no field in this schema that
  can carry "rated for X kg" — structurally absent, not policy-absent.

This replaces refusal as the trust layer: the engine attempts the part; THIS block
says exactly how solid the result is.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic.provenance import DimensionProvenance, Tier, TierCounts, tier_counts

Overall = Literal["high", "medium", "low"]
Basis = Literal["validated", "approximate", "conceptual"]
Manufacturability = Literal[
    "prints_as_is",  # print_3d-validated geometry, gates pass
    "makeable",  # general-mode geometry, gates pass (faithful CAD model)
    "printable_as_model",  # printable as a VISUAL model; the real part needs another process
    "not_makeable_as_shown",  # the gates failed: not physically makeable as shown
]

_RANK_OVERALL: dict[str, int] = {"high": 2, "medium": 1, "low": 0}


# --- the part-class table (closed; reviewed in git, never LLM-selected) --------------------------
#
# Classes whose FUNCTION requires more than geometry. Keyword tables over the brief's object
# name/summary -- deliberately narrow: a bracket or a gear is NOT here (a printed bracket is a
# real bracket; over-capping flattens the label until it means nothing -- skeptic seat).

_ROTATING_POWER = (
    "crankshaft",
    "camshaft",
    "connecting rod",
    "conrod",
    "piston",
    "turbine",
    "impeller",
    "flywheel",
    "drive shaft",
    "driveshaft",
    "propeller",
)
_PRESSURE = (
    "pressure vessel",
    "boiler",
    "hydraulic cylinder",
    "pneumatic cylinder",
    "intake manifold",
    "exhaust manifold",
    "pressure tank",
)
_LIFE_SAFETY = (
    "carabiner",
    "climbing",
    "harness",
    "lifting hook",
    "lifting eye",
    "shackle",
    "sling",
    "implant",
    "prosthe",  # prosthesis / prosthetic
    "brake caliper",
    "brake rotor",
    "steering",
)

PartClass = Literal["none", "rotating_power", "pressure_bearing", "life_safety"]

# Canned not-fit-for text per trigger (table-selected; prose may never be generated here).
_NOT_FIT_FOR: dict[str, str] = {
    "life_safety": (
        "any use where failure endangers people -- this is not a certified, rated component"
    ),
    "rotating_power": (
        "rotating service in a real engine or machine -- not balanced, not fatigue-rated"
    ),
    "pressure_bearing": "pressurized service -- not pressure-rated",
    "estimated_dims": "dimension-critical fits without first verifying the estimated dimensions",
}
_PROCESS_NOTE: dict[str, str] = {
    "rotating_power": "a working part of this class is forged/machined metal, then balanced",
    "pressure_bearing": "a working part of this class is welded/machined metal, then tested",
    "life_safety": "a working part of this class requires certified manufacture and rating",
}


def classify_part(object_name: str | None, summary: str | None = None) -> PartClass:
    """Closed keyword classification of functional-critical part classes. Deterministic."""
    text = f"{object_name or ''} {summary or ''}".lower()
    if any(k in text for k in _LIFE_SAFETY):
        return "life_safety"
    if any(k in text for k in _ROTATING_POWER):
        return "rotating_power"
    if any(k in text for k in _PRESSURE):
        return "pressure_bearing"
    return "none"


class ConfidenceLabel(BaseModel):
    """The machine-honest verdict: closed enums + table text only. ``basis`` is None exactly
    when the gates hard-failed (no basis is claimed for a part that is not a coherent solid;
    the manufacture stamp already marks it not-for-manufacture)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: Overall
    basis: Basis | None
    manufacturability: Manufacturability
    process_note: str | None = None
    dimension_tiers: TierCounts
    verify_dimensions: list[str] = Field(default_factory=list)
    not_fit_for: list[str] = Field(default_factory=list)
    part_class: PartClass = "none"
    summary: str  # assembled deterministically from the fields above


def _dims_floor(records: list[DimensionProvenance]) -> Overall:
    if not records:
        # No researched dimensions at all (brief-less build): nothing is grounded.
        return "low"
    methods = [
        (r.estimate_basis.method if r.estimate_basis is not None else "default_guess")
        for r in records
        if r.tier is Tier.ESTIMATED
    ]
    if not methods:
        return "high"  # every dimension verified / sourced / stated
    if all(m != "default_guess" for m in methods):
        return "medium"  # estimates, but each carries engineering reasoning
    return "low"  # at least one 'it looked right'


def _judge_cap(judge_accepted: bool | None, judge_confidence: str | None) -> Overall:
    if judge_accepted is None:
        return "high"  # no judge ran (deterministic lanes / judge unavailable): no cap here
    if not judge_accepted:
        return "low"
    if judge_confidence == "high":
        return "high"
    if judge_confidence == "medium":
        return "medium"
    return "low"


def _gates_cap(gates: Literal["pass", "warn", "fail"]) -> Overall:
    return {"pass": "high", "warn": "medium", "fail": "low"}[gates]  # type: ignore[return-value]


def derive_confidence(
    *,
    records: list[DimensionProvenance],
    gates: Literal["pass", "warn", "fail"],
    judge_accepted: bool | None,
    judge_confidence: str | None,
    object_name: str | None,
    summary_text: str | None = None,
    fabrication_mode: str = "general",
) -> ConfidenceLabel:
    """Derive the label from evidence. Pure, total, monotone (the lattice test proves it)."""
    counts = tier_counts(records)
    part_class = classify_part(object_name, summary_text)

    floors = [
        _dims_floor(records),
        _judge_cap(judge_accepted, judge_confidence),
        _gates_cap(gates),
    ]
    overall: Overall = min(floors, key=lambda v: _RANK_OVERALL[v])

    has_default_guess = any(
        r.tier is Tier.ESTIMATED
        and (r.estimate_basis is None or r.estimate_basis.method == "default_guess")
        for r in records
    )
    basis: Basis | None
    if gates == "fail":
        basis = None  # no basis claimed for a part that is not a coherent solid
    elif (
        has_default_guess
        or not records
        or judge_accepted is False
        or part_class != "none"  # functional-critical: geometry is not function
    ):
        basis = "conceptual"
    elif counts.estimated > 0:
        basis = "approximate"
    else:
        basis = "validated"

    if gates == "fail":
        manufacturability: Manufacturability = "not_makeable_as_shown"
        process_note = None
    elif part_class != "none":
        manufacturability = "printable_as_model"
        process_note = _PROCESS_NOTE[part_class]
    elif fabrication_mode == "print_3d":
        manufacturability = "prints_as_is"
        process_note = None
    else:
        manufacturability = "makeable"
        process_note = None

    not_fit_for: list[str] = []
    if part_class != "none":
        not_fit_for.append(_NOT_FIT_FOR[part_class])
    if counts.estimated > 0:
        not_fit_for.append(_NOT_FIT_FOR["estimated_dims"])
    not_fit_for = not_fit_for[:3]

    verify = [r.feature for r in records if r.tier is Tier.ESTIMATED][:6]

    summary = (
        f"{overall} confidence - "
        + (basis if basis is not None else "failed validation")
        + f" - dimensions: {counts.verified} verified / {counts.sourced} sourced / "
        f"{counts.stated} stated / {counts.estimated} estimated"
    )
    return ConfidenceLabel(
        overall=overall,
        basis=basis,
        manufacturability=manufacturability,
        process_note=process_note,
        dimension_tiers=counts,
        verify_dimensions=verify,
        not_fit_for=not_fit_for,
        part_class=part_class,
        summary=summary,
    )


def standard_lane_label(
    interface_count: int, *, fabrication_mode: str = "print_3d"
) -> ConfidenceLabel:
    """The deterministic lane's label: registry-derived interfaces + strict gates ARE the
    evidence (no judge runs there; the gates are the verdict). Per-dimension records are NOT
    enumerated -- only the interface count is claimed verified (honest scope, ADR-0040)."""
    counts = TierCounts(verified=max(interface_count, 0))
    return ConfidenceLabel(
        overall="high",
        basis="validated",
        manufacturability="prints_as_is" if fabrication_mode == "print_3d" else "makeable",
        dimension_tiers=counts,
        part_class="none",
        summary=(
            f"high confidence - validated - dimensions: {counts.verified} verified / 0 sourced "
            "/ 0 stated / 0 estimated"
        ),
    )


def assembly_lane_label(
    sourced_count: int, *, swept: bool, fabrication_mode: str = "general"
) -> ConfidenceLabel:
    """The deterministic ASSEMBLY lane's label. Dimensions anchor to PROVISIONAL
    registry rows (e.g. the MG996R datasheet entry), so they are claimed "sourced",
    never "verified" (ADR-0040 anchor invariant). ``basis`` is "validated" ONLY when
    the full assembly claim gate passed INCLUDING the earned swept-collision level;
    an un-swept assembly is conceptual no matter how clean its static checks look."""
    counts = TierCounts(sourced=max(sourced_count, 0))
    basis: Literal["validated", "conceptual"] = "validated" if swept else "conceptual"
    overall: Overall = "high" if swept else "low"
    return ConfidenceLabel(
        overall=overall,
        basis=basis,
        manufacturability="prints_as_is" if fabrication_mode == "print_3d" else "makeable",
        process_note=(
            "Articulation swept collision-free through the full declared range "
            "(chain vs rest, hardware envelopes included)."
            if swept
            else "Motion was NOT swept; do not treat the articulation as verified."
        ),
        dimension_tiers=counts,
        part_class="none",
        summary=(
            f"{overall} confidence - {basis} - dimensions: 0 verified / "
            f"{counts.sourced} sourced / 0 stated / 0 estimated (registry datasheet, "
            "provisional)"
        ),
    )


__all__ = [
    "ConfidenceLabel",
    "PartClass",
    "assembly_lane_label",
    "classify_part",
    "derive_confidence",
    "standard_lane_label",
]
