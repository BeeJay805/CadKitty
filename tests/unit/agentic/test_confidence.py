"""The confidence + fitness label (ADR-0041): computed, total, and MONOTONE.

The council's no-inflation proof: enumerate the evidence lattice and assert that
degrading any single axis never raises any output field. Plus the two end-to-end
fixtures the council pinned: the crankshaft (medium/conceptual, never validated)
and the NEMA bracket (high/validated -- the label must DISCRIMINATE). Money-free.
"""

from __future__ import annotations

from itertools import product

from forge_mind.agentic.confidence import (
    ConfidenceLabel,
    classify_part,
    derive_confidence,
    standard_lane_label,
)
from forge_mind.agentic.provenance import (
    DimensionProvenance,
    EstimateBasis,
    Tier,
)


def _rec(feature: str, tier: Tier, method: str | None = None) -> DimensionProvenance:
    mechanism = {
        Tier.VERIFIED: "registry_resolution",
        Tier.SOURCED: "intake_sourced",
        Tier.STATED: "intake_stated",
        Tier.ESTIMATED: "default_estimated",
    }[tier]
    return DimensionProvenance(
        feature=feature,
        dims_text="x",
        tier=tier,
        mechanism=mechanism,  # type: ignore[arg-type]
        citation="cite" if tier is not Tier.ESTIMATED else None,
        estimate_basis=(
            EstimateBasis(method=method, anchor="a" if method != "default_guess" else None)  # type: ignore[arg-type]
            if tier is Tier.ESTIMATED and method
            else None
        ),
    )


# The evidence lattice, each axis ordered BEST -> WORST.
_DIMS_AXIS: list[list[DimensionProvenance]] = [
    [_rec("holes", Tier.VERIFIED), _rec("plate", Tier.STATED)],  # fully anchored
    [_rec("holes", Tier.VERIFIED), _rec("web", Tier.ESTIMATED, "anchor_scaled")],  # eng. estimate
    [_rec("web", Tier.ESTIMATED, "default_guess")],  # a guess
    [],  # nothing researched
]
_JUDGE_AXIS: list[tuple[bool, str]] = [
    (True, "high"),
    (True, "medium"),
    (True, "low"),
    (False, "high"),  # rejected
]
_GATES_AXIS: list[str] = ["pass", "warn", "fail"]
_CLASS_AXIS: list[str | None] = ["bracket", "crankshaft"]  # none -> rotating_power

_OVERALL_RANK = {"high": 2, "medium": 1, "low": 0}
_BASIS_RANK = {"validated": 3, "approximate": 2, "conceptual": 1, None: 0}


def _derive(d: int, j: int, g: int, c: int) -> ConfidenceLabel:
    accepted, conf = _JUDGE_AXIS[j]
    return derive_confidence(
        records=_DIMS_AXIS[d],
        gates=_GATES_AXIS[g],  # type: ignore[arg-type]
        judge_accepted=accepted,
        judge_confidence=conf,
        object_name=_CLASS_AXIS[c],
        fabrication_mode="print_3d",
    )


def test_lattice_is_total_and_monotone() -> None:
    """For EVERY evidence combination: a valid label; and degrading any ONE axis by one
    step never raises overall or basis. Monotonicity IS 'nothing claims what it didn't
    earn' (methodology seat)."""
    axes = (len(_DIMS_AXIS), len(_JUDGE_AXIS), len(_GATES_AXIS), len(_CLASS_AXIS))
    for d, j, g, c in product(*(range(n) for n in axes)):
        label = _derive(d, j, g, c)  # totality: never raises, enums validated by pydantic
        for axis, idx in enumerate((d, j, g, c)):
            if idx + 1 >= axes[axis]:
                continue
            degraded = [d, j, g, c]
            degraded[axis] += 1
            worse = _derive(*degraded)
            assert _OVERALL_RANK[worse.overall] <= _OVERALL_RANK[label.overall], (
                f"degrading axis {axis} RAISED overall at {(d, j, g, c)}"
            )
            assert _BASIS_RANK[worse.basis] <= _BASIS_RANK[label.basis], (
                f"degrading axis {axis} RAISED basis at {(d, j, g, c)}"
            )


def test_estimated_dims_never_label_validated() -> None:
    """The WS2 headline: an estimate-based part is NEVER 'validated', whatever the judge
    and gates say."""
    for j, g in product(range(len(_JUDGE_AXIS)), range(len(_GATES_AXIS))):
        for d in (1, 2, 3):  # every dims state containing an estimate (or nothing)
            assert _derive(d, j, g, 0).basis != "validated"


def test_crankshaft_earns_conceptual_never_for_your_car() -> None:
    """The crankshaft-class case: builds, looks right, judge accepts -- and the label says
    geometric concept, estimated dims, not a working part."""
    records = [
        _rec("main journals", Tier.ESTIMATED, "anchor_scaled"),
        _rec("rod journals", Tier.ESTIMATED, "anchor_scaled"),
        _rec("counterweights", Tier.ESTIMATED, "proportion_rule"),
    ]
    label = derive_confidence(
        records=records,
        gates="pass",
        judge_accepted=True,
        judge_confidence="high",
        object_name="automotive 4-cylinder crankshaft",
        fabrication_mode="print_3d",
    )
    assert label.overall == "medium"  # engineering estimates floor it; nothing raised it
    assert label.basis == "conceptual"  # rotating_power: geometry is not function
    assert label.manufacturability == "printable_as_model"
    assert label.process_note is not None and "machined" in label.process_note
    assert any("rotating service" in n for n in label.not_fit_for)
    assert label.verify_dimensions  # the middle is actionable
    # The forbidden claims are structurally absent:
    assert label.basis != "validated" and label.manufacturability != "prints_as_is"


def test_grounded_bracket_still_earns_the_top_label() -> None:
    """Discrimination (methodology seat): clean evidence hits no caps -- the top label is
    reachable, or the vocabulary stops meaning anything."""
    records = [
        _rec("mount holes", Tier.VERIFIED),
        _rec("extrusion bolts", Tier.VERIFIED),
        _rec("plate", Tier.STATED),
    ]
    label = derive_confidence(
        records=records,
        gates="pass",
        judge_accepted=True,
        judge_confidence="high",
        object_name="NEMA17 motor bracket",
        fabrication_mode="print_3d",
    )
    assert label.overall == "high"
    assert label.basis == "validated"
    assert label.manufacturability == "prints_as_is"
    assert label.not_fit_for == []
    assert label.verify_dimensions == []


def test_failed_gates_claim_no_basis() -> None:
    label = _derive(0, 0, 2, 0)  # best dims + judge, gates FAIL
    assert label.basis is None
    assert label.manufacturability == "not_makeable_as_shown"
    assert label.overall == "low"


def test_no_judge_does_not_cap_but_other_axes_hold() -> None:
    """judge_accepted=None (no judge ran) is not a degradation axis -- the deterministic
    lanes' gates are their verdict. The dims floor still binds."""
    label = derive_confidence(
        records=_DIMS_AXIS[1],
        gates="pass",
        judge_accepted=None,
        judge_confidence=None,
        object_name="bracket",
        fabrication_mode="print_3d",
    )
    assert label.overall == "medium"  # the engineering-estimate floor, not the judge


def test_part_class_table() -> None:
    assert classify_part("automotive crankshaft") == "rotating_power"
    assert classify_part("a climbing carabiner") == "life_safety"
    assert classify_part("intake manifold for a 1.6L engine") == "pressure_bearing"
    assert classify_part("NEMA17 motor bracket") == "none"
    assert classify_part("a 24-tooth spur gear") == "none"  # deliberate: ADR-0041 records this


def test_life_safety_label() -> None:
    label = derive_confidence(
        records=[_rec("gate", Tier.ESTIMATED, "anchor_scaled")],
        gates="pass",
        judge_accepted=True,
        judge_confidence="high",
        object_name="a climbing carabiner",
        fabrication_mode="print_3d",
    )
    assert label.basis == "conceptual"
    assert any("endangers people" in n for n in label.not_fit_for)
    assert label.manufacturability == "printable_as_model"


def test_standard_lane_label_is_validated_by_construction() -> None:
    label = standard_lane_label(6)
    assert label.overall == "high" and label.basis == "validated"
    assert label.manufacturability == "prints_as_is"
    assert label.dimension_tiers.verified == 6
    assert label.not_fit_for == []


def test_summary_is_assembled_not_composed() -> None:
    label = _derive(1, 0, 0, 0)
    assert "medium confidence" in label.summary
    assert "1 estimated" in label.summary
