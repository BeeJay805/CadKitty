"""Guidance selects relevant detail without hiding tools or evidence boundaries."""

import copy
from pathlib import Path

import pytest

from forge_mind.workspace.author_guidance import (
    ASSEMBLY,
    CORE,
    LAB,
    MOTION,
    NOMINAL_INSPECTION_DETAIL,
    PRINT_PREPARATION_DETAIL,
    PRODUCT,
    SOURCES,
    STUDIES,
    build_author_guidance,
    guidance_topics,
    repair_focus,
)
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.parametric_guidance import PARAMETRIC_DESIGN_GUIDANCE


@pytest.mark.parametrize("executor_mode", ["trusted_local", "docker"])
def test_shared_parameter_advice_is_present_once_without_changing_contract(executor_mode):
    context = {"request": "Make an editable part", "frozen_contract": {"requirements": []}}
    before = copy.deepcopy(context)
    system = build_author_guidance(context, executor_mode)
    assert system.count(PARAMETRIC_DESIGN_GUIDANCE) == 1
    assert "never change its targets" in system
    assert context == before


def test_simple_part_uses_compact_construction_context_but_all_tools_stay_discoverable():
    context = {"request": "Make a 20 mm by 10 mm by 5 mm rectangular block"}
    system = build_author_guidance(context, "trusted_local")
    assert len(system) < 7000
    assert guidance_topics(context) == []
    for tool in (
        "construction:<recipe_id>",
        "feature:<kind>",
        "assembly_edit",
        "engineering_tool",
        "source_page",
        "motion_check",
        "design_study",
        "lab",
        "read_design",
        "python",
        "guidance",
    ):
        assert tool in system
    for module in (ASSEMBLY, PRODUCT, MOTION, SOURCES, STUDIES, LAB):
        assert module not in system
    assert "frozen contract" in system
    assert "failed_attempt" in system


def test_native_binding_and_instruction_detail_is_scoped_to_native_authoring():
    ordinary = build_author_guidance({"request": "Make a simple block"}, "trusted_local")
    native = build_author_guidance(
        {"request": "Make a simple block", "authoring_mode": "native_operations"}, "trusted_local"
    )
    for detail in ("promote_relations", "derive_variables", "reviewed_current_facts=true"):
        assert detail not in ordinary
        assert detail in native


@pytest.mark.parametrize("kind", ["nominal_perforated_box", "nominal_through_hole_pattern"])
def test_ordinary_frozen_nominal_context_receives_only_its_full_inspection_detail(kind):
    context = {
        "request": "Continue the saved part",
        "frozen_contract": {"requirements": [{"verification": {"kind": kind}}]},
    }
    before = copy.deepcopy(context)
    system = build_author_guidance(context, "trusted_local")
    assert system.count(NOMINAL_INSPECTION_DETAIL) == 1
    assert PRINT_PREPARATION_DETAIL not in system
    assert "print_setup query=prepare" in system
    assert "no printer profile, G-code, supports or physical success is inferred" in system
    assert context == before


def test_ordinary_prepared_print_duty_receives_only_shared_preparation_detail():
    context = {
        "request": "Continue the saved part",
        "frozen_contract": {
            "requirements": [
                {
                    "verification": {
                        "kind": "digital_delivery",
                        "deliverable": "prepared_print_package",
                    }
                }
            ]
        },
    }
    before = copy.deepcopy(context)
    system = build_author_guidance(context, "trusted_local")
    assert system.count(PRINT_PREPARATION_DETAIL) == 1
    assert NOMINAL_INSPECTION_DETAIL not in system
    assert context == before


@pytest.mark.parametrize("collection", ["history", "persistent_tool_observations"])
def test_actual_print_operation_context_loads_detail_without_a_request_keyword(collection):
    context = {"request": "Continue", collection: [{"print_setup": {"status": "current"}}]}
    before = copy.deepcopy(context)
    system = build_author_guidance(context, "trusted_local")
    assert system.count(PRINT_PREPARATION_DETAIL) == 1
    assert NOMINAL_INSPECTION_DETAIL not in system
    assert context == before


def test_untyped_request_words_and_other_digital_duties_do_not_load_full_detail():
    context = {
        "request": "Inspect nominal geometry and prepare a print package",
        "frozen_contract": {
            "requirements": [
                {
                    "verification": {
                        "kind": "digital_delivery",
                        "deliverable": "project_step",
                    }
                }
            ]
        },
    }
    system = build_author_guidance(context, "trusted_local")
    assert len(system) < 7000
    assert PRINT_PREPARATION_DETAIL not in system
    assert NOMINAL_INSPECTION_DETAIL not in system
    assert "print_setup query=prepare" in system
    assert "inspect reads evidence/source/diff" in system


def test_native_detail_text_positions_and_output_do_not_depend_on_optional_context():
    context = {"request": "Make a simple block", "authoring_mode": "native_operations"}
    native = build_author_guidance(context, "trusted_local")
    with_details = build_author_guidance(
        {
            **context,
            "frozen_contract": {
                "requirements": [
                    {"verification": {"kind": "nominal_perforated_box"}},
                    {
                        "verification": {
                            "kind": "digital_delivery",
                            "deliverable": "prepared_print_package",
                        }
                    },
                ]
            },
            "history": [{"print_setup": {"status": "current"}}],
        },
        "trusted_local",
    )
    assert native == with_details
    assert native.count(PRINT_PREPARATION_DETAIL) == 1
    assert native.count(NOMINAL_INSPECTION_DETAIL) == 1
    assert native.index(PRINT_PREPARATION_DETAIL) < native.index("schema query=construction")
    assert native.index(NOMINAL_INSPECTION_DETAIL) < native.index(
        "inspect reads evidence/source/diff"
    )
    assert native.count(PARAMETRIC_DESIGN_GUIDANCE) == 1


def test_implicit_existing_assembly_activates_guidance_without_keyword_match():
    context = {
        "request": "Make it a little narrower",
        "selected": {"document": {"assembly": {"parts": [{"id": "body"}]}}},
    }
    assert guidance_topics(context) == ["assembly"]
    assert ASSEMBLY in build_author_guidance(context, "trusted_local")


def test_summary_failed_assembly_and_existing_plan_are_authoritative_state_hints():
    context = {
        "request": "Repair it",
        "failed_attempt": {"document": {"counts": {"parts": 24}}},
        "design_plan": {"inspection": [{"id": "bore"}]},
    }
    assert guidance_topics(context) == ["assembly", "product"]


def test_current_schema_and_tool_observations_activate_implicit_specialized_tools():
    context = {
        "request": "Continue",
        "authoritative_schemas": {"lab:prediction": {}, "design_study": {}, "motion_rig": {}},
        "persistent_tool_observations": [{"source_text": {}}, {"engineering_tool": {}}],
    }
    assert guidance_topics(context) == ["product", "motion", "sources", "studies", "lab"]


def test_word_boundaries_do_not_turn_workflow_into_flow_optimization():
    assert guidance_topics({"request": "Improve the workflow for this block"}) == []


def test_default_empty_engineering_wrapper_does_not_load_product_guidance():
    context = {
        "request": "Make a plain block",
        "engineering_inputs": {
            "version": 12,
            "inputs": {"material": None, "process": None, "fits": [], "loads": []},
        },
    }
    assert guidance_topics(context) == []
    context["engineering_inputs"]["inputs"]["process"] = "CNC"
    assert guidance_topics(context) == ["product"]


def test_complex_fixture_keeps_product_source_and_motion_detail():
    context = {
        "request": "Make a rotating fixture with sourced bearing, CNC tolerance and cable access"
    }
    system = build_author_guidance(context, "trusted_local")
    for module in (ASSEMBLY, PRODUCT, MOTION, SOURCES):
        assert module in system
    assert STUDIES not in system and LAB not in system
    assert "at least 17 integer teeth" in system
    assert "no fixed upper count" in system


def test_hosted_guidance_reports_unavailable_native_execution_and_preserves_assembly():
    system = build_author_guidance({}, "docker")
    assert "Typed document actions are unavailable" in system
    assert "instead of flattening an existing assembly" in system


def test_qualified_container_guidance_keeps_native_authoring_available():
    system = build_author_guidance({}, "docker", typed_container_verified=True)
    assert CORE in system
    assert "Typed document actions are unavailable" not in system
    assert "Use general build actions" not in system
    assert "Preserve an existing assembly" in system


def test_repair_focus_retains_actual_failed_ids_without_creating_a_pass():
    context = {
        "selected": {
            "id": "valid",
            "measurement": {
                "findings": [
                    {"requirement_id": "gap", "status": "fail", "measured": 79, "expected": 80},
                    {"requirement_id": "height", "status": "pass", "measured": 20},
                ]
            },
        },
        "failed_attempt": {"id": "broken", "measurement": {"errors": ["Bad loft section"]}},
        "early_checks": {
            "revision_id": "valid",
            "subsystems": [
                {
                    "id": "jaws",
                    "checks": [
                        {
                            "id": "spacing",
                            "status": "fail",
                            "receipt_id": "receipt",
                            "reason": "Too close",
                        },
                    ],
                }
            ],
        },
        "completion": {"next_target": "jaws"},
        "history": [{"revision": "valid", "regressions": ["gap"]}],
    }
    before = copy.deepcopy(context)
    focus = repair_focus(context)
    assert focus["failed_requirements"] == [context["selected"]["measurement"]["findings"][0]]
    assert focus["failed_checks"][0]["receipt_id"] == "receipt"
    assert focus["construction_errors"] == ["Bad loft section"]
    assert focus["next_subsystem"] == "jaws"
    assert focus["failed_attempt_revision_id"] == "broken"
    assert focus["regressions"] == [{"revision_id": "valid", "requirement_ids": ["gap"]}]
    assert context == before


def test_stale_checkpoint_never_becomes_a_current_repair_failure():
    context = {
        "early_checks": {
            "stale": True,
            "subsystems": [
                {"id": "body", "checks": [{"id": "gap", "status": "fail"}]},
            ],
        }
    }
    assert repair_focus(context)["failed_checks"] == []


def test_unsupported_requirement_remains_open_without_becoming_a_repair_failure():
    context = {
        "selected": {
            "id": "plate",
            "measurement": {
                "geometry_valid": True,
                "findings": [
                    {
                        "requirement_id": "corners",
                        "status": "unverified",
                        "method": "no applicable instrument",
                        "reason": "R4 independent measurement unavailable",
                    },
                ],
            },
        },
    }
    focus = repair_focus(context)
    assert focus["failed_requirements"] == []
    assert focus["unverified_requirements"][0]["requirement_id"] == "corners"
    assert focus["unverified_requirement_count"] == 1
    assert "finish a useful draft" in focus["unverified_scope"]


def test_failure_priority_list_is_bounded_and_reports_remaining_count():
    context = {
        "selected": {
            "measurement": {
                "findings": [
                    {"requirement_id": str(index), "status": "fail"} for index in range(100)
                ]
            }
        }
    }
    focus = repair_focus(context)
    assert len(focus["failed_requirements"]) == 16
    assert focus["failed_requirement_count"] == 100


def numeric_context(
    measured=90.00000460121548, *, metric="extent_x", target=90, tolerance=0, comparator="eq"
):
    requirement = Requirement(
        id="dimension",
        text="Requested dimension",
        source_quote="Requested dimension",
        metric=metric,
        value=target,
        tolerance=tolerance,
        comparator=comparator,
    )
    contract = Contract(requirements=[requirement])
    return {
        "frozen_contract": contract.model_dump(),
        "selected": {
            "id": "candidate",
            "measurement": {
                "contract_hash": contract.fingerprint(),
                "requirements_satisfied": False,
                "findings": [
                    {
                        "requirement_id": requirement.id,
                        "status": "fail",
                        "measured": measured,
                        "expected": target,
                        "method": "STEP bounds",
                    }
                ],
            },
        },
    }


@pytest.mark.parametrize(
    "comparator,measured,target,tolerance,outside",
    [("eq", 9, 10, 0.25, 0.75), ("ge", 9, 10, 0.25, 0.75), ("le", 11, 10, 0.25, 0.75)],
)
def test_numeric_advice_describes_comparator_direction_without_changing_verdict(
    comparator, measured, target, tolerance, outside
):
    context = numeric_context(measured, target=target, tolerance=tolerance, comparator=comparator)
    before = copy.deepcopy(context)
    focus = repair_focus(context)
    comparison = focus["numeric_comparisons"][0]
    assert comparison["comparator"] == comparator
    assert comparison["delta"] == measured - target
    assert comparison["outside_contract_by"] == outside
    assert comparison["tolerance"] == tolerance
    assert comparison["status"] == "fail"
    assert focus["failed_requirements"] == before["selected"]["measurement"]["findings"]
    assert context == before


def test_small_actual_excess_stays_visible_without_a_repair_cutoff_or_uncertainty_claim():
    context = numeric_context()
    focus = repair_focus(context)
    comparison = focus["numeric_comparisons"][0]
    assert comparison["outside_contract_by"] == pytest.approx(0.00000460121548)
    assert comparison["unit"] == "mm"
    assert comparison["status"] == "fail"
    assert (
        "not evaluator slack, uncertainty bounds or a repair cutoff"
        in focus["numeric_comparison_scope"]
    )
    assert "needs_review, never a pass" in focus["numeric_comparison_scope"]
    assert not context["selected"]["measurement"]["requirements_satisfied"]
    assert "Explicit tight precision remains actionable" in CORE
    assert "Prioritize missing requested geometry" in CORE


@pytest.mark.parametrize(
    "metric,measured,target,unit",
    [("solid_count", 2, 1, "count"), ("hole_count", 1, 2, "count"), ("volume", 101, 100, "mm^3")],
)
def test_counts_and_volume_keep_their_quantity_and_failed_obligation(
    metric, measured, target, unit
):
    focus = repair_focus(numeric_context(measured, metric=metric, target=target))
    comparison = focus["numeric_comparisons"][0]
    assert comparison["unit"] == unit
    assert comparison["outside_contract_by"] == 1
    assert comparison["status"] == "fail"
    assert "Counts are discrete; volume is in cubic mm" in focus["numeric_comparison_scope"]


@pytest.mark.parametrize("measured", [None, float("nan"), float("inf"), -float("inf"), True, "90"])
def test_invalid_measured_fields_cannot_create_numeric_advice(measured):
    focus = repair_focus(numeric_context(measured))
    assert "numeric_comparisons" not in focus
    assert focus["failed_requirement_count"] == 1


@pytest.mark.parametrize("field", ["measured", "expected"])
def test_missing_finding_fields_retain_failure_without_inventing_discrepancy(field):
    context = numeric_context()
    del context["selected"]["measurement"]["findings"][0][field]
    focus = repair_focus(context)
    assert "numeric_comparisons" not in focus
    assert focus["failed_requirement_count"] == 1


@pytest.mark.parametrize("field", ["value", "tolerance"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_contract_fields_cannot_create_numeric_advice(field, value):
    context = numeric_context()
    context["frozen_contract"]["requirements"][0][field] = value
    assert "numeric_comparisons" not in repair_focus(context)


@pytest.mark.parametrize("expected", [None, float("nan"), float("inf")])
def test_invalid_expected_fields_do_not_match_a_contract_target(expected):
    context = numeric_context()
    context["selected"]["measurement"]["findings"][0]["expected"] = expected
    assert "numeric_comparisons" not in repair_focus(context)


@pytest.mark.parametrize("comparator,measured", [("ge", 11), ("le", 9)])
def test_advice_does_not_overwrite_a_failed_finding_even_when_its_delta_is_inside(
    comparator, measured
):
    context = numeric_context(measured, target=10, comparator=comparator)
    focus = repair_focus(context)
    assert focus["numeric_comparisons"][0]["outside_contract_by"] == 0
    assert focus["numeric_comparisons"][0]["status"] == "fail"
    assert focus["failed_requirement_count"] == 1


def test_scoped_numeric_requirement_keeps_local_part_identity():
    context = numeric_context()
    context["frozen_contract"]["requirements"][0]["part_id"] = "blade"
    context["selected"]["measurement"]["contract_hash"] = Contract.model_validate(
        context["frozen_contract"]
    ).fingerprint()
    assert repair_focus(context)["numeric_comparisons"][0]["part_id"] == "blade"


def test_numeric_comparison_context_is_bounded_without_omitting_failure_count():
    context = numeric_context()
    first = context["frozen_contract"]["requirements"][0]
    finding = context["selected"]["measurement"]["findings"][0]
    context["frozen_contract"]["requirements"] = [
        {**first, "id": f"dimension_{index}"} for index in range(100)
    ]
    context["selected"]["measurement"]["findings"] = [
        {**finding, "requirement_id": f"dimension_{index}"} for index in range(100)
    ]
    context["selected"]["measurement"]["contract_hash"] = Contract.model_validate(
        context["frozen_contract"]
    ).fingerprint()
    focus = repair_focus(context)
    assert len(focus["numeric_comparisons"]) == 16
    assert focus["failed_requirement_count"] == 100


@pytest.mark.parametrize(
    "change", ["missing_contract", "changed_contract", "wrong_target", "unknown_id"]
)
def test_numeric_advice_requires_current_contract_and_matching_finding(change):
    context = numeric_context()
    if change == "missing_contract":
        del context["frozen_contract"]
    elif change == "changed_contract":
        context["frozen_contract"]["requirements"][0]["tolerance"] = 0.01
    elif change == "wrong_target":
        context["selected"]["measurement"]["findings"][0]["expected"] = 89
    else:
        context["selected"]["measurement"]["findings"][0]["requirement_id"] = "historical"
    focus = repair_focus(context)
    assert "numeric_comparisons" not in focus
    assert focus["failed_requirement_count"] == 1


def test_fractional_count_evidence_is_not_treated_as_a_length_residual():
    assert "numeric_comparisons" not in repair_focus(
        numeric_context(1.000005, metric="solid_count", target=1)
    )


def test_unverified_numeric_evidence_never_becomes_a_failed_numeric_comparison():
    context = numeric_context()
    context["selected"]["measurement"]["findings"][0]["status"] = "unverified"
    context["selected"]["measurement"]["findings"][0]["reason"] = "Instrument unavailable"
    focus = repair_focus(context)
    assert "numeric_comparisons" not in focus
    assert focus["unverified_requirement_count"] == 1


def test_native_box_excess_and_explicit_tolerance_keep_existing_measurement_verdicts(
    tmp_path: Path,
):
    from build123d import Box, export_step, export_stl

    from forge_mind.workspace.measure import measure_revision

    part = Box(90.000005, 90, 24)
    export_step(part, tmp_path / "part.step")
    export_stl(part, tmp_path / "part.stl")
    (tmp_path / "program.py").write_text("# Native independent control", encoding="utf-8")
    for tolerance, expected_status in [(0, "fail"), (0.000002, "fail"), (0.000005, "pass")]:
        context = numeric_context(tolerance=tolerance)
        contract = Contract.model_validate(context["frozen_contract"])
        report = measure_revision(tmp_path, "candidate", contract)
        assert report.geometry_valid, report.errors
        assert report.findings[0].status == expected_status
        context["selected"]["measurement"] = report.model_dump()
        before = copy.deepcopy(context)
        focus = repair_focus(context)
        assert context == before
        assert report.requirements_satisfied == (expected_status == "pass")
        if expected_status == "fail":
            assert focus["numeric_comparisons"][0]["status"] == "fail"
            assert focus["numeric_comparisons"][0]["outside_contract_by"] == pytest.approx(
                0.000005 - tolerance
            )
        else:
            assert "numeric_comparisons" not in focus
