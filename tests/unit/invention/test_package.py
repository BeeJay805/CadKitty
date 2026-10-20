"""The InventionPackage assembler (ADR-0047): composes real engine outputs, REPORTED, and
enforces the two moat-hole gates STRUCTURALLY — the physical-validation banner is always
present and feasibility can never read as a passed gate."""

from __future__ import annotations

from forge_mind.agentic.confidence import ConfidenceLabel, standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import (
    FeasibilityVerdict,
    InventionPackage,
    PhysicalValidationStatus,
    build_invention_package,
)
from forge_mind.kit import build_kit
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _program(features: list[HardwareFeature]) -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="invention-test-plate", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=features,
        constraints=Constraints(min_wall_thickness=4.0),
    )


def _hw(category: str, key: str) -> HardwareFeature:
    return HardwareFeature(category=category, key=key, nominal_mm=0.0, fit=None, where="t")


def _package(
    program: PartProgram,
    *,
    disposition: str = "shipped",
    confidence: ConfidenceLabel | None = None,
    problem_statement: str | None = None,
) -> InventionPackage:
    return build_invention_package(
        part_name=program.metadata.name,
        disposition=disposition,
        kit=build_kit(program),
        confidence=confidence,
        problem_statement=problem_statement,
    )


# --- law 1: composed from closed fields only (no generated fitness prose) ---------------------


def test_schema_has_no_free_text_fitness_field() -> None:
    """The envelope's only verbatim string is the user's own problem_statement; every other field
    is a typed sub-object, a closed enum, or a table-selected string. This guards the moat the same
    way confidence.py's 'no quantified performance claims, ever' does — structurally, not by policy."""
    assert set(InventionPackage.model_fields) == {
        "part_name",
        "problem_statement",
        "disposition",
        "artifacts",
        "kit",
        "bom_summary",
        "confidence",
        "concept",
        "feasibility",
        "physical_validation",
    }


def test_problem_statement_is_the_users_words_verbatim() -> None:
    pkg = _package(_program([]), problem_statement="route two USB-C cables on my desk edge")
    assert pkg.problem_statement == "route two USB-C cables on my desk edge"


# --- law 2: the physical-validation banner is structural and always present -------------------


def test_physical_banner_always_present() -> None:
    pkg = _package(_program([]))
    assert pkg.physical_validation.status is PhysicalValidationStatus.NOT_VALIDATED
    assert "NOT PHYSICALLY VALIDATED" in pkg.physical_validation.banner
    assert "CAD-valid is not mission success" in pkg.physical_validation.banner


def test_no_false_physical_maturity_is_reachable() -> None:
    """The only status the digital engine can claim is 'not validated' — higher physical maturity
    tiers are absent until real test evidence can exist (no false maturity)."""
    assert [s.value for s in PhysicalValidationStatus] == ["not_physically_validated"]


def test_package_cannot_be_built_without_a_banner() -> None:
    """A package is structurally incapable of rendering without the banner: the field is required."""
    assert InventionPackage.model_fields["physical_validation"].is_required()


# --- law 3: feasibility is advisory, never a passed gate --------------------------------------


def test_feasibility_defaults_to_advisory_not_assessed() -> None:
    pkg = _package(_program([]))
    assert pkg.feasibility.verdict is FeasibilityVerdict.NOT_ASSESSED
    assert "not a guarantee" in pkg.feasibility.disclaimer
    assert "never as a passed validation gate" in pkg.feasibility.disclaimer


def test_feasibility_has_no_pass_verdict() -> None:
    """No feasibility value is a certifying PASS — it can never be mistaken for the validator's
    verdict. The vocabulary is hypotheses, not gates."""
    values = {v.value for v in FeasibilityVerdict}
    assert values == {
        "not_assessed",
        "likely_feasible",
        "feasible_with_changes",
        "likely_infeasible",
    }
    assert not any(tok in v for v in values for tok in ("pass", "validated", "certified"))


# --- composition: the empty BOM is the correct answer, not a gap ------------------------------


def test_empty_bom_surfaces_honestly() -> None:
    """A screwless part has no fasteners; build_bom honestly returns 'no registry-backed hardware'
    and the package surfaces that verbatim — the empty BOM is the correct v0 answer."""
    pkg = _package(_program([]))
    assert pkg.bom_summary == "no registry-backed hardware"
    assert pkg.kit.bom.lines == []


def test_populated_bom_composes_through() -> None:
    """A part that declares a clearance hole produces a real, cited BOM line — proving the populated
    path composes too (the council's v0.1 gate certifies both empty and non-empty)."""
    pkg = _package(_program([_hw("clearance_hole", "M3")]))
    assert pkg.bom_summary != "no registry-backed hardware"
    assert any("M3" in line.component for line in pkg.kit.bom.lines)


# --- REPORTED, composed faithfully ------------------------------------------------------------


def test_disposition_is_copied_verbatim() -> None:
    """The package never softens the engine's honest disposition."""
    pkg = _package(_program([]), disposition="shipped_with_caveats")
    assert pkg.disposition == "shipped_with_caveats"


def test_confidence_may_be_absent_for_a_no_build() -> None:
    pkg = _package(_program([]), disposition="no_build", confidence=None)
    assert pkg.confidence is None
    # The banner and feasibility gates still hold for a part that never earned a label.
    assert pkg.physical_validation.status is PhysicalValidationStatus.NOT_VALIDATED
    assert pkg.feasibility.verdict is FeasibilityVerdict.NOT_ASSESSED


def test_confidence_composes_when_present() -> None:
    pkg = _package(_program([]), confidence=standard_lane_label(2))
    assert isinstance(pkg.confidence, ConfidenceLabel)
    assert pkg.confidence.overall == "high"


def test_package_is_frozen() -> None:
    pkg = _package(_program([]))
    try:
        pkg.disposition = "rejected"  # type: ignore[misc]
    except (TypeError, ValueError):
        return
    raise AssertionError("InventionPackage must be immutable (frozen)")
