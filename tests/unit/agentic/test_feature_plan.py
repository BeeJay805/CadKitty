"""Lever f (FORGE_FEATURE_PLAN=1): the deterministic feature checklist, the plan-to-code
mapping audit, and the loop's PRE-BUILD refusal of a draft whose mapping misses a declared
feature. Also pins the deployed default: with the flag unset, the author prompt suffix is
empty and the structured-output schema is the plain AuthoredCode (byte-identical)."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic import author as author_mod
from forge_mind.agentic import feature_plan
from forge_mind.agentic import loop as loop_mod
from forge_mind.agentic.author import (
    AuthoredCode,
    AuthoredCodeWithPlan,
    FeatureMapEntry,
    _output_model,
    _plan_user_suffix,
)
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.llm.cost import LLMUsage


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="test bracket",
        summary="a bracket",
        visual_description="an L-shaped bracket",
        overall_envelope_mm="40 x 40 x 10",
        features=[
            BriefFeature(
                name="bolt holes",
                purpose="mounting",
                approx_dims_mm="M4",
                method="bolt_holes",
                count=4,
                standard="M4",
            ),
            BriefFeature(
                name="stiffening rib", purpose="rigidity", approx_dims_mm="3mm", method="box"
            ),
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat side down",
    )


# ---------------------------------------------------------------- checklist + audit


def test_checklist_transcribes_every_feature() -> None:
    items = feature_plan.feature_checklist(_brief())
    assert [(i.name, i.count) for i in items] == [("bolt holes", 4), ("stiffening rib", 1)]
    assert items[0].standard == "M4"
    assert items[0].dims == "M4"


def test_checklist_block_empty_without_features() -> None:
    assert feature_plan.checklist_block([]) == ""


def test_audit_passes_a_complete_mapping() -> None:
    items = feature_plan.feature_checklist(_brief())
    code = "holes = bolt_holes(pcd=30, hole_d=4.5, count=4, depth=12)\nrib = Box(3, 30, 8)"
    mapping = [
        FeatureMapEntry(feature="bolt holes", where="bolt_holes(pcd=30"),
        FeatureMapEntry(feature="Stiffening Rib", where="Box(3, 30, 8)"),  # case-insensitive
    ]
    assert feature_plan.audit_feature_map(items, mapping, code) == []


def test_audit_names_a_missing_feature() -> None:
    items = feature_plan.feature_checklist(_brief())
    code = "holes = bolt_holes(pcd=30, hole_d=4.5, count=4, depth=12)"
    mapping = [FeatureMapEntry(feature="bolt holes", where="bolt_holes(pcd=30")]
    gaps = feature_plan.audit_feature_map(items, mapping, code)
    assert len(gaps) == 1
    assert "stiffening rib" in gaps[0]


def test_audit_rejects_a_where_not_in_the_code() -> None:
    items = feature_plan.feature_checklist(_brief())
    code = "rib = Box(3, 30, 8)"
    mapping = [
        FeatureMapEntry(feature="bolt holes", where="bolt_holes(pcd=30"),  # not in code
        FeatureMapEntry(feature="stiffening rib", where="Box(3, 30, 8)"),
    ]
    gaps = feature_plan.audit_feature_map(items, mapping, code)
    assert len(gaps) == 1
    assert "does not appear" in gaps[0]


# ---------------------------------------------------------------- deployed default pinned


def test_flag_off_keeps_prompt_and_schema_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORGE_FEATURE_PLAN", raising=False)
    assert feature_plan.enabled() is False
    assert _plan_user_suffix(_brief()) == ""
    assert _output_model(_brief()) is AuthoredCode


def test_flag_on_appends_checklist_and_switches_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_FEATURE_PLAN", "1")
    suffix = _plan_user_suffix(_brief())
    assert "FEATURE CHECKLIST" in suffix
    assert "bolt holes x4" in suffix
    assert _output_model(_brief()) is AuthoredCodeWithPlan


# ---------------------------------------------------------------- the loop refusal gate


def test_loop_refuses_prebuild_then_builds_when_mapping_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Draft 1 maps only 1 of 2 features -> the loop must NOT sandbox it; the refusal
    feedback names the gap; the corrected draft 2 (complete mapping) IS sandboxed."""
    monkeypatch.setenv("FORGE_FEATURE_PLAN", "1")
    code_incomplete = "holes = bolt_holes(pcd=30, hole_d=4.5, count=4, depth=12)\nresult = holes"
    code_complete = code_incomplete + "\nrib = Box(3, 30, 8)\nresult = holes + rib"

    def _initial(*_a: object, **_k: object) -> tuple[AuthoredCodeWithPlan, LLMUsage]:
        return (
            AuthoredCodeWithPlan(
                approach="block it out",
                code=code_incomplete,
                feature_map=[FeatureMapEntry(feature="bolt holes", where="bolt_holes(pcd=30")],
            ),
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        )

    revision_feedback: list[str] = []

    def _revision(
        _brief_arg: object, _base: object, feedback: str, **_k: object
    ) -> tuple[AuthoredCodeWithPlan, LLMUsage]:
        revision_feedback.append(feedback)
        return (
            AuthoredCodeWithPlan(
                approach="add the rib",
                code=code_complete,
                feature_map=[
                    FeatureMapEntry(feature="bolt holes", where="bolt_holes(pcd=30"),
                    FeatureMapEntry(feature="stiffening rib", where="Box(3, 30, 8)"),
                ],
            ),
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        )

    sandboxed: list[str] = []

    def _sandbox(code: str, *_a: object, **_k: object) -> object:
        sandboxed.append(code)
        from forge_mind.agentic.sandbox import SandboxResult

        return SandboxResult(ok=False, error_type="exec", error_message="stub: not built")

    monkeypatch.setattr(loop_mod, "author_initial", _initial)
    # Revisions flow through revise_dispatch (ADR-0114), which imports author_revision from the
    # author module at call time -- that module is the patch seam, not loop.
    monkeypatch.setattr(author_mod, "author_revision", _revision)
    monkeypatch.setattr(loop_mod, "run_build123d_code", _sandbox)

    result = loop_mod.run_agentic_pipeline(
        "a bracket",
        tmp_path,
        max_iterations=2,
        brief=_brief(),
        client=object(),
    )
    # The incomplete draft was never sandboxed; the corrected one was.
    assert sandboxed == [code_complete]
    assert revision_feedback and "FEATURE-PLAN REFUSAL" in revision_feedback[0]
    assert "stiffening rib" in revision_feedback[0]
    refusals = [
        it for it in result.iterations if (it.sandbox_error or "").startswith("feature_plan")
    ]
    assert len(refusals) == 1 and refusals[0].built is False


def test_refusals_are_bounded_then_the_sandbox_judges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An author that NEVER completes the mapping is refused only refusal_cap() times;
    after that the draft proceeds to the sandbox -- the lever cannot deadlock a part."""
    monkeypatch.setenv("FORGE_FEATURE_PLAN", "1")
    code_incomplete = "holes = bolt_holes(pcd=30, hole_d=4.5, count=4, depth=12)\nresult = holes"

    def _stubborn(*_a: object, **_k: object) -> tuple[AuthoredCodeWithPlan, LLMUsage]:
        return (
            AuthoredCodeWithPlan(
                approach="ignore the rib",
                code=code_incomplete,
                feature_map=[FeatureMapEntry(feature="bolt holes", where="bolt_holes(pcd=30")],
            ),
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        )

    def _stubborn_rev(*_a: object, **_k: object) -> tuple[AuthoredCodeWithPlan, LLMUsage]:
        return _stubborn()

    sandbox_calls: list[int] = []

    def _sandbox(code: str, *_a: object, **_k: object) -> object:
        sandbox_calls.append(1)
        from forge_mind.agentic.sandbox import SandboxResult

        return SandboxResult(ok=False, error_type="exec", error_message="stub: not built")

    monkeypatch.setattr(loop_mod, "author_initial", _stubborn)
    monkeypatch.setattr(author_mod, "author_revision", _stubborn_rev)
    monkeypatch.setattr(loop_mod, "run_build123d_code", _sandbox)

    result = loop_mod.run_agentic_pipeline(
        "a bracket",
        tmp_path,
        max_iterations=6,
        brief=_brief(),
        client=object(),
    )
    refusals = [
        it for it in result.iterations if (it.sandbox_error or "").startswith("feature_plan")
    ]
    assert len(refusals) == feature_plan.refusal_cap()
    assert len(sandbox_calls) >= 1  # the cap released the draft to the measured gates


def test_flag_off_never_audits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deployed default: no checklist, no refusal -- the first draft goes straight to the
    sandbox even with no mapping at all."""
    monkeypatch.delenv("FORGE_FEATURE_PLAN", raising=False)

    def _initial(*_a: object, **_k: object) -> tuple[AuthoredCode, LLMUsage]:
        return AuthoredCode(approach="plain", code="result = Box(10, 10, 10)"), LLMUsage(
            model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0
        )

    sandbox_calls: list[str] = []

    def _sandbox(code: str, *_a: object, **_k: object) -> object:
        sandbox_calls.append(code)
        from forge_mind.agentic.sandbox import SandboxResult

        return SandboxResult(ok=False, error_type="exec", error_message="stub")

    monkeypatch.setattr(loop_mod, "author_initial", _initial)
    monkeypatch.setattr(author_mod, "author_revision", _initial)
    monkeypatch.setattr(loop_mod, "run_build123d_code", _sandbox)

    result = loop_mod.run_agentic_pipeline(
        "a box", tmp_path, max_iterations=1, brief=_brief(), client=object()
    )
    assert sandbox_calls  # sandboxed immediately, no refusal records
    assert not [
        it for it in result.iterations if (it.sandbox_error or "").startswith("feature_plan")
    ]
