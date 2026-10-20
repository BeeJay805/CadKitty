"""The additive invention_package API view (ADR-0048, P2): the projection carries ONLY typed/closed
data (maturity + evidence labels, advisory feasibility, the NOT-PHYSICALLY-VALIDATED banner, the
chosen concept), the field is optional (byte-identical when off), and the envelope is None unless
FORGE_INVENTION is on. No model calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.agentic.loop import LoopResult
from forge_mind.agentic.result import PartResult
from forge_mind.api.core import _invention_package_view, _invention_view_for_result
from forge_mind.api.schema import GenerateOk, InventionPackageView
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.fabrication import FabricationMode
from forge_mind.invention import (
    ConceptBrief,
    EvidenceTier,
    MaturityLabel,
    Mechanism,
    PhysicalValidationStatus,
    build_invention_package,
    derive_maturity,
    derive_package_evidence,
)
from forge_mind.invention.package import InventionPackage
from forge_mind.kit import build_kit
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _kit() -> Any:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_kit(program)


def _concept() -> ConceptBrief:
    return ConceptBrief(
        concept_id="c1",
        name="C-clip",
        primary_mechanism=Mechanism.SNAP_FIT,
        approach="flex over the edge",
        obvious_baseline="bracket",
        difference_from_obvious="no screws",
    )


def _package(*, concept: ConceptBrief | None = None) -> InventionPackage:
    return build_invention_package(
        part_name="cable-clip",
        disposition="shipped",
        kit=_kit(),
        confidence=standard_lane_label(0),
        problem_statement="route two cables",
        concept=concept,
    )


def _result(tmp_path: Path) -> PartResult:
    brief = DesignBrief(
        object_name="cable clip",
        summary="s",
        visual_description="v",
        overall_envelope_mm="40 x 20 x 15",
        features=[BriefFeature(name="jaw", purpose="grip", approx_dims_mm="40", method="box")],
        min_wall_mm=2.0,
        print_orientation_note="flat",
        target="cosmetic",
    )
    loop = LoopResult(
        prompt="x",
        brief=brief,
        final_code=None,
        final_stl=None,
        final_step=None,
        final_render=None,
        validation=None,
        critique=None,
        geometry_passed=True,
        critic_approved=True,
        stopped_reason="test",
        iterations=[],
        total_cost_usd=0.0,
        total_duration_s=0.0,
    )
    return PartResult(
        out_dir=tmp_path,
        loop=loop,
        verdict=None,
        disposition="shipped",
        total_cost_usd=0.0,
        artifacts={},
        provenance=[],
    )


# --- labels: deterministic maturity + evidence (never auto-upgraded) --------------------------


def test_derive_maturity_caps_below_physical() -> None:
    nv = PhysicalValidationStatus.NOT_VALIDATED
    assert derive_maturity("validated", nv) is MaturityLabel.DIGITALLY_VALIDATED
    assert derive_maturity("approximate", nv) is MaturityLabel.PROTOTYPE_CANDIDATE
    assert derive_maturity("conceptual", nv) is MaturityLabel.CONCEPT
    assert derive_maturity(None, nv) is MaturityLabel.CONCEPT
    # PHYSICALLY_TESTED is unreachable while the engine has no physical-test loop.
    for basis in ("validated", "approximate", "conceptual", None):
        assert derive_maturity(basis, nv) is not MaturityLabel.PHYSICALLY_TESTED


def test_derive_package_evidence_never_claims_simulation_or_physical() -> None:
    assert derive_package_evidence(None) is EvidenceTier.AI_HYPOTHESIS
    assert derive_package_evidence("validated") is EvidenceTier.DETERMINISTIC_CHECK
    assert derive_package_evidence("conceptual") is EvidenceTier.DETERMINISTIC_CHECK
    for basis in (None, "validated", "approximate", "conceptual"):
        assert derive_package_evidence(basis) not in (
            EvidenceTier.SIMPLIFIED_SIMULATION,
            EvidenceTier.HIGH_FIDELITY_SIMULATION,
            EvidenceTier.PHYSICAL_TEST,
        )


# --- the projection: typed/closed only, banner + advisory feasibility always present ----------


def test_view_carries_the_banner_and_advisory_feasibility() -> None:
    view = _invention_package_view(_package())
    assert "NOT PHYSICALLY VALIDATED" in view.physical_validation.banner
    assert "not a guarantee" in view.feasibility.disclaimer
    assert view.maturity in {m.value for m in MaturityLabel}
    assert view.evidence_tier in {e.value for e in EvidenceTier}
    assert view.concept is None  # /generate has no front-door concept
    assert view.bom_summary == "no registry-backed hardware"


def test_view_projects_the_concept_when_present() -> None:
    view = _invention_package_view(_package(concept=_concept()))
    assert view.concept is not None
    assert view.concept.concept_id == "c1"
    assert view.concept.primary_mechanism == "snap_fit"


def test_view_has_no_free_text_fitness_field() -> None:
    # The set is pinned so a NEW field can never sneak a generated fitness claim into the envelope.
    # support_kb is a structured pass-through (the deterministic SupportCopilotKB dump), not free text.
    assert set(InventionPackageView.model_fields) == {
        "part_name",
        "problem_statement",
        "disposition",
        "maturity",
        "evidence_tier",
        "bom_summary",
        "feasibility",
        "physical_validation",
        "concept",
        "support_kb",
    }


def test_view_carries_the_support_kb_that_drives_the_support_endpoint() -> None:
    """The envelope embeds the grounded support KB (ADR-0050) so the frontend can POST it straight to
    /invention/support: the dump must round-trip back into a SupportCopilotKB (the endpoint's body)."""
    from forge_mind.invention.support import SupportCopilotKB

    view = _invention_package_view(_package())
    assert isinstance(view.support_kb, dict)
    assert "NOT PHYSICALLY VALIDATED" in view.support_kb["validation_status"]["banner"]
    # the dump is exactly the SupportBody.kb contract -- re-validates with no loss
    kb = SupportCopilotKB.model_validate(view.support_kb)
    assert kb.product_purpose == "route two cables"
    assert any(p.kind == "printed" for p in kb.parts)


# --- additive on GenerateOk; the envelope is flag-gated ---------------------------------------


def test_generate_ok_field_is_optional_and_defaults_none() -> None:
    """Existing clients are unaffected: the field is not required and defaults to None."""
    field = GenerateOk.model_fields["invention_package"]
    assert field.is_required() is False
    assert field.default is None


def test_envelope_is_none_when_flag_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_INVENTION", raising=False)
    view = _invention_view_for_result(
        _result(tmp_path), "route cables", "PLA", FabricationMode.GENERAL
    )
    assert view is None


def test_envelope_present_when_flag_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_INVENTION", "1")
    view = _invention_view_for_result(
        _result(tmp_path), "route cables", "PLA", FabricationMode.GENERAL
    )
    assert isinstance(view, InventionPackageView)
    assert view.part_name == "cable clip"
    assert view.problem_statement == "route cables"
    assert "NOT PHYSICALLY VALIDATED" in view.physical_validation.banner
    assert view.concept is None  # /generate path: no front-door concept
    assert (
        view.support_kb is not None
    )  # the grounded KB rides the envelope (drives /invention/support)
