"""The requirements fact layer (ADR-0087 S1): provenance round-trips at the honest tier, a
researched fact can NEVER surface as STATED (P7), unit normalization is deterministic (P4), and a
conflicting pair is surfaced not averaged (P10)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from forge_mind.agentic.provenance import Tier, parse_stated_dimensions
from forge_mind.invention.requirements import Fact, FactLedger, normalize_measure

# --- unit normalization (P4): deterministic, code-only, unknown units pass through -----------------


def test_normalize_length_and_mass() -> None:
    assert normalize_measure(1.0, "in") == (25.4, "mm")
    assert normalize_measure(2.5, "inch") == (63.5, "mm")
    assert normalize_measure(4.0, '"') == (101.6, "mm")
    assert normalize_measure(1.0, "cm") == (10.0, "mm")
    assert normalize_measure(0.1, "m") == (100.0, "mm")
    assert normalize_measure(69.85, "mm") == (69.85, "mm")
    assert normalize_measure(1.0, "kg") == (1000.0, "g")
    v, u = normalize_measure(1.0, "lb")
    assert u == "g" and math.isclose(v, 453.59237, rel_tol=1e-6)


def test_unknown_unit_passes_through_never_guessed() -> None:
    # a non-length/mass unit is NEVER converted -- we don't invent a factor
    assert normalize_measure(30.0, "deg") == (30.0, "deg")
    assert normalize_measure(4.0, "count") == (4.0, "count")


# --- Fact: the fail-closed citation invariant -----------------------------------------------------


def test_non_user_fact_must_cite() -> None:
    # a researched/registry/derived number with no anchor cannot exist (the vendor-gate discipline)
    for src in ("registry", "researched", "derived"):
        with pytest.raises(ValidationError):
            Fact(name="x", value=1.0, unit="mm", source=src)  # type: ignore[arg-type]
    # a user fact needs no citation -- the user IS the source
    Fact(name="max thickness", value=9.5, unit="mm", source="user")


# --- round-trip: ledger -> block -> parse_stated_dimensions, at the honest tier per source --------


def _ssd_ledger() -> FactLedger:
    return FactLedger(
        facts=(
            Fact(name="ssd_width", value=69.85, unit="mm", source="registry", citation="SFF-8201"),
            Fact(
                name="bay_depth",
                value=4.0,
                unit="in",  # 4 in -> 101.6 mm on the way out
                source="researched",
                citation="https://example.com/nr200-manual",
                quote="drive bay depth 4 inches",
            ),
            Fact(name="max_thickness", value=9.5, unit="mm", source="user"),
            Fact(
                name="bracket_inner_width",
                value=95.6,
                unit="mm",
                source="derived",
                citation="= bay_width 101.6 - 2 x wall 3.0",
            ),
        )
    )


def test_round_trip_tiers() -> None:
    block = _ssd_ledger().to_stated_facts_block()
    dims = {
        d.line.split(" =")[0].replace("dimension: ", ""): d for d in parse_stated_dimensions(block)
    }

    assert dims["ssd_width"].tier is Tier.SOURCED  # registry -> standard -> SOURCED
    assert dims["bay_depth"].tier is Tier.SOURCED  # researched -> SOURCED
    assert dims["bay_depth"].value == 101.6  # normalized 4 in -> 101.6 mm in the block
    assert dims["max_thickness"].tier is Tier.STATED  # user -> STATED
    # derived facts are NOT anchors: they never enter the stated-facts block
    assert "bracket_inner_width" not in dims


def test_p7_researched_never_stated() -> None:
    # THE pin (ADR-0087 P7 / the anchor invariant): a researched dimension round-trips SOURCED and
    # can never be laundered into STATED ("you told us"). Sweep it alone so nothing else masks it.
    led = FactLedger(
        facts=(
            Fact(
                name="case_clearance",
                value=80.0,
                unit="mm",
                source="researched",
                citation="https://example.com/spec",
                quote="80 mm clearance",
            ),
        )
    )
    (dim,) = parse_stated_dimensions(led.to_stated_facts_block())
    assert dim.tier is Tier.SOURCED
    assert dim.tier is not Tier.STATED
    assert dim.mechanism == "intake_researched"


# --- conflict detection (P10): surfaced, compared on the normalized value, never averaged ---------


def test_conflict_detected_on_normalized_value() -> None:
    led = FactLedger(
        facts=(
            Fact(name="bay_depth", value=4.0, unit="in", source="researched", citation="a"),
            Fact(name="bay_depth", value=100.0, unit="mm", source="researched", citation="b"),
        )
    )
    conflicts = led.conflicts()
    assert len(conflicts) == 1 and conflicts[0].name == "bay_depth"  # 101.6 mm vs 100.0 mm


def test_same_measure_different_unit_is_not_a_conflict() -> None:
    # 4 in and 101.6 mm for one name are the SAME fact, not a conflict (normalized comparison)
    led = FactLedger(
        facts=(
            Fact(name="bay_depth", value=4.0, unit="in", source="researched", citation="a"),
            Fact(name="bay_depth", value=101.6, unit="mm", source="researched", citation="b"),
        )
    )
    assert led.conflicts() == []


def test_anchors_exclude_derived() -> None:
    led = _ssd_ledger()
    assert {f.name for f in led.anchors()} == {"ssd_width", "bay_depth", "max_thickness"}
    assert {f.name for f in led.derived()} == {"bracket_inner_width"}
