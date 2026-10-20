"""The real produce adapter (ADR-0048, P1): produce_for_invention turns a produce_part PartResult
into the lifecycle's GeneratedPart, REUSING the /generate kit + confidence derivation. Tested with
a canned PartResult injected via produce_fn -- no real model call."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.loop import LoopResult
from forge_mind.agentic.result import PartResult
from forge_mind.api.core import produce_for_invention
from forge_mind.invention import (
    ConceptBrief,
    ConceptPlan,
    FeasibilityVerdict,
    GeneratedPart,
    Mechanism,
    run_lifecycle,
)
from forge_mind.kit import Kit
from forge_mind.llm.cost import LLMUsage


def _brief(*, standard: str | None = None) -> DesignBrief:
    return DesignBrief(
        object_name="cable clip",
        summary="a desk-edge cable clip",
        visual_description="a C-shaped clip",
        overall_envelope_mm="40 x 20 x 15",
        features=[
            BriefFeature(
                name="jaw", purpose="grip", approx_dims_mm="40", method="box", standard=standard
            )
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat",
        target="cosmetic",
    )


def _result(
    tmp_path: Path, *, brief: DesignBrief | None, disposition: str = "shipped"
) -> PartResult:
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
        disposition=disposition,
        total_cost_usd=0.0,
        artifacts={"stl": "part.stl", "glb": "part.glb"},
        provenance=[],
    )


def _stub_produce(result: PartResult) -> Any:
    def produce(prompt: str, out_dir: Path, **_: Any) -> PartResult:
        return result

    return produce


def test_adapter_composes_a_generated_part(tmp_path: Path) -> None:
    part = produce_for_invention(
        "clip", tmp_path, produce_fn=_stub_produce(_result(tmp_path, brief=_brief()))
    )
    assert isinstance(part, GeneratedPart)
    assert part.part_name == "cable clip"
    assert part.disposition == "shipped"
    assert isinstance(
        part.kit, Kit
    )  # the DOMAIN kit the package embeds (projected from the API kit)
    assert part.confidence is not None
    assert part.artifacts == {"stl": "part.stl", "glb": "part.glb"}


def test_adapter_empty_bom_is_honest(tmp_path: Path) -> None:
    part = produce_for_invention(
        "clip", tmp_path, produce_fn=_stub_produce(_result(tmp_path, brief=_brief()))
    )
    assert part.kit.bom.summary() == "no registry-backed hardware"


def test_adapter_standard_feature_is_a_provisional_line(tmp_path: Path) -> None:
    """A brief-declared standard becomes a PROVISIONAL bom line (never a fake registry citation)."""
    part = produce_for_invention(
        "clip", tmp_path, produce_fn=_stub_produce(_result(tmp_path, brief=_brief(standard="M3")))
    )
    assert len(part.kit.bom.lines) == 1
    line = part.kit.bom.lines[0]
    assert line.provisional is True
    assert "M3" in line.component


def test_adapter_copies_disposition_verbatim(tmp_path: Path) -> None:
    part = produce_for_invention(
        "clip",
        tmp_path,
        produce_fn=_stub_produce(_result(tmp_path, brief=_brief(), disposition="rejected")),
    )
    assert part.disposition == "rejected"


def test_adapter_missing_brief_uses_a_safe_name(tmp_path: Path) -> None:
    part = produce_for_invention(
        "clip", tmp_path, produce_fn=_stub_produce(_result(tmp_path, brief=None))
    )
    assert part.part_name == "Generated part"
    assert part.kit.bom.summary() == "no registry-backed hardware"


class _FrontDoorStub:
    def __init__(self, plan: ConceptPlan) -> None:
        self.plan = plan

    def generate_structured(self, **_: Any) -> tuple[Any, LLMUsage]:
        return self.plan, LLMUsage(model="claude-fable-5", input_tokens=1, output_tokens=1)

    def generate_structured_with_image(self, **_: Any) -> tuple[Any, LLMUsage]:
        raise AssertionError("no image")


def test_adapter_plugs_into_run_lifecycle(tmp_path: Path) -> None:
    """The integration: the real adapter as run_lifecycle's `produce`, the front door mocked -> a
    full InventionPackage carrying the concept + the engine-derived kit/confidence, banner intact."""
    plan = ConceptPlan(
        problem_restated="route flexible conductors along an edge",
        feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
        feasibility_note="routine FDM print",
        concepts=[
            ConceptBrief(
                concept_id="c1",
                name="C-clip",
                primary_mechanism=Mechanism.SNAP_FIT,
                approach="flex over the edge",
                obvious_baseline="bracket",
                difference_from_obvious="no screws",
            )
        ],
        recommended_concept_id="c1",
        recommendation_reason="simplest",
    )
    result = _result(tmp_path, brief=_brief())
    package, got_plan = run_lifecycle(
        "cables fall off my desk edge",
        produce=lambda p: produce_for_invention(p, tmp_path, produce_fn=_stub_produce(result)),
        client=_FrontDoorStub(plan),
    )
    assert package.concept is not None
    assert package.concept.concept_id == "c1"
    assert package.part_name == "cable clip"
    assert isinstance(package.kit, Kit)
    assert package.problem_statement == "cables fall off my desk edge"
    assert "NOT PHYSICALLY VALIDATED" in package.physical_validation.banner
    assert got_plan.recommended_concept_id == "c1"
