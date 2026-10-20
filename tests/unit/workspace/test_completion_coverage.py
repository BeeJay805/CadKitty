"""Unmeasured intent cannot disappear behind scalar checks or representation choices."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.unit.workspace.test_completion import FakeStore
from tests.unit.workspace.test_inventory_verification_admission import (
    INVENTORY_QUOTE,
    SCOPE_QUOTE,
)
from tests.unit.workspace.test_inventory_verification_admission import (
    plan as inventory_plan,
)
from tests.unit.workspace.test_inventory_verification_admission import (
    target as inventory_target,
)
from tests.unit.workspace.test_requirement_verification import digital, geometry_target

from forge_mind.workspace.completion import completion_context, finalize_completion
from forge_mind.workspace.completion_coverage import functional_coverage
from forge_mind.workspace.completion_models import (
    FunctionalInterface,
    InterfaceDuty,
    RequirementDisposition,
)
from forge_mind.workspace.document import digest
from forge_mind.workspace.models import Contract, DesignPlan, Finding, Measurement, Requirement
from forge_mind.workspace.native_finish import finish_context
from forge_mind.workspace.operation_plan import snapshot
from forge_mind.workspace.product_tools import _SCOPES


def requirement(identifier="function", metric="unmeasured"):
    return {"id": identifier, "metric": metric}


def interface(identifier="function", **duty_changes):
    values = {"id": "hold", "kind": "holding", "proof_scope": "physical_function"}
    values.update(duty_changes)
    return FunctionalInterface(
        id="connection",
        requirement_ids=[identifier],
        participant_paths=["member", "pin"],
        duties=[InterfaceDuty.model_validate(values)],
    )


def test_absent_unmeasured_mapping_is_actionable():
    result = functional_coverage({}, {"requirements": [requirement()]})
    assert result["status"] == "needs_review"
    assert result["missing_requirement_ids"] == ["function"]
    assert result["next_target"] == "functional_coverage"
    assert result["next_work_category"] == "review"
    assert "function" in result["next_work_reason"]


@pytest.mark.parametrize("field", ["missing_inputs", "unsupported_checks", "physical_validation"])
@pytest.mark.parametrize("texts", [[], [""], [" "], ["\t", "\n"]])
def test_empty_and_whitespace_external_notes_do_not_cover_a_requirement(field, texts):
    plan = {"requirement_dispositions": [{"requirement_id": "function", field: texts}]}
    result = functional_coverage(plan, {"requirements": [requirement()]})
    assert result["status"] == "needs_review"
    assert result["missing_requirement_ids"] == ["function"]
    assert result["external_requirements"] == []


def test_scalar_check_ids_alone_do_not_supply_functional_coverage():
    plan = {
        "subsystems": [{"id": "support", "checks": [{"id": "scalar", "status": "pass"}]}],
        "requirement_dispositions": [{"requirement_id": "function", "check_ids": ["scalar"]}],
    }
    result = functional_coverage(plan, {"requirements": [requirement()]})
    assert result["status"] == "needs_review"
    assert result["missing_requirement_ids"] == ["function"]
    assert result["interface_requirement_ids"] == []


def test_aesthetic_external_mapping_stays_unknown_without_an_interface_taxonomy():
    text = "No deterministic evaluator establishes the requested appearance"
    result = functional_coverage(
        {
            "requirement_dispositions": [
                {"requirement_id": "appearance", "unsupported_checks": [text]}
            ]
        },
        {"requirements": [requirement("appearance")]},
    )
    assert result["status"] == "unverified"
    assert result["missing_requirement_ids"] == []
    assert result["interface_requirement_ids"] == []
    assert result["external_requirements"] == [
        {
            "requirement_id": "appearance",
            "missing_inputs": [],
            "unsupported_checks": [text],
            "physical_validation": [],
        }
    ]
    assert "next_target" not in result


def test_valid_interface_mapping_does_not_prove_whole_functional_prose():
    result = functional_coverage(
        {"functional_interfaces": [interface()]}, {"requirements": [requirement()]}
    )
    assert result["status"] == "unverified"
    assert result["interface_requirement_ids"] == ["function"]
    assert result["missing_requirement_ids"] == []
    assert "next_target" not in result
    assert "do not prove" in result["scope"]


def test_external_duty_text_is_preserved_exactly_from_both_mapping_sources():
    disposition = RequirementDisposition(
        requirement_id="function", missing_inputs=["  Measure the applied load in N  "]
    )
    mapped = interface(
        unsupported_checks=["Spatial friction is unsupported"],
        physical_validation=["Test the exact fabricated assembly"],
    )
    result = functional_coverage(
        {"requirement_dispositions": [disposition], "functional_interfaces": [mapped]},
        {"requirements": [requirement()]},
    )
    assert result["status"] == "unverified"
    assert result["external_requirements"] == [
        {
            "requirement_id": "function",
            "missing_inputs": ["  Measure the applied load in N  "],
            "unsupported_checks": ["Spatial friction is unsupported"],
            "physical_validation": ["Test the exact fabricated assembly"],
        }
    ]


@pytest.mark.parametrize("name", ["engine_holding", "visual_style", "alpha", "arbitrary_name"])
def test_requirement_names_do_not_change_the_coverage_rule(name):
    contract = {"requirements": [requirement(name)]}
    assert functional_coverage({}, contract)["status"] == "needs_review"
    assert functional_coverage({"functional_interfaces": [interface(name)]}, contract)[
        "status"
    ] == ("unverified")


def test_known_metrics_need_no_new_functional_mapping():
    contract = {
        "requirements": [requirement("size", "extent_x"), requirement("holes", "hole_count")]
    }
    result = functional_coverage({}, contract)
    assert result["status"] == "not_applicable"
    assert result["missing_requirement_ids"] == []
    assert result["external_requirements"] == []
    assert "next_target" not in result


def typed_plan():
    return DesignPlan(
        objective="Frozen nominal and digital obligations",
        verification_bindings=[
            {"subject_id": "plate", "part_id": "plate_body", "frame": "definition_local"}
        ],
    )


def test_valid_typed_only_plan_uses_subject_and_delivery_mapping_without_external_invention():
    frozen = Contract(requirements=[geometry_target("nominal_perforated_box"), digital()])
    plan = typed_plan()
    before = (plan.model_dump_json(), frozen.model_dump_json())
    result = functional_coverage(plan, frozen)
    assert result["status"] == "unverified"
    assert result["missing_requirement_ids"] == [] and result["external_requirements"] == []
    assert [row["mapping_status"] for row in result["typed_requirements"]] == [
        "subject_binding_declared",
        "delivery_obligation",
    ]
    assert all(row["status"] == "unverified" for row in result["typed_requirements"])
    assert "next_target" not in result
    assert (plan.model_dump_json(), frozen.model_dump_json()) == before


def test_typed_geometry_missing_binding_stays_actionable_without_invented_physical_review():
    frozen = Contract(requirements=[geometry_target("nominal_through_hole_pattern")])
    result = functional_coverage(DesignPlan(objective="Missing explicit binding"), frozen)
    assert result["status"] == "needs_review"
    assert result["next_target"] == "verification_bindings"
    assert result["typed_requirements"][0]["mapping_status"] == "missing_subject_binding"
    assert result["typed_requirements"][0]["status"] == "unverified"


def test_typed_predicates_do_not_hide_a_physical_conjunct_or_grant_it_a_pass():
    frozen = Contract(
        requirements=[
            geometry_target("nominal_through_hole_pattern"),
            digital(),
            Requirement(id="holding", text="Hold the load", source_quote="Hold the load"),
        ]
    )
    plan = typed_plan()
    missing = functional_coverage(plan, frozen)
    assert missing["missing_requirement_ids"] == ["holding"]
    assert missing["next_target"] == "functional_coverage"
    plan.requirement_dispositions = [
        RequirementDisposition(
            requirement_id="holding",
            physical_validation=["Test the actual fabricated holding assembly"],
        )
    ]
    mapped = functional_coverage(plan, frozen)
    assert mapped["status"] == "unverified" and mapped["missing_requirement_ids"] == []
    assert [row["requirement_id"] for row in mapped["external_requirements"]] == ["holding"]
    assert all(row["status"] == "unverified" for row in mapped["typed_requirements"])


@pytest.mark.parametrize("fault", ["kind", "extra", "assumption", "source"])
def test_malformed_or_unsupported_typed_criteria_keep_strict_old_mapping_duties(fault):
    raw = digital().model_dump(mode="json")
    if fault == "kind":
        raw["verification"]["kind"] = "unsupported"
    elif fault == "extra":
        raw["verification"]["status"] = "pass"
    elif fault == "assumption":
        raw["source"] = "assumption"
    else:
        raw["verification"]["source_quote"] = "A different source"
    result = functional_coverage(DesignPlan(objective="Strict fallback"), {"requirements": [raw]})
    assert result["missing_requirement_ids"] == ["files"]
    assert result["next_target"] == "functional_coverage"
    assert not result.get("typed_requirements")


def test_mixed_metrics_require_only_unmeasured_ids_and_preserve_contract_order():
    contract = {
        "requirements": [
            requirement("known", "extent_x"),
            requirement("first"),
            requirement("second"),
            requirement("third"),
        ]
    }
    plan = {
        "functional_interfaces": [interface("first")],
        "requirement_dispositions": [
            {"requirement_id": "second", "physical_validation": ["Test the stated use"]},
            {"requirement_id": "known", "physical_validation": ["Unrelated measured duty"]},
        ],
    }
    result = functional_coverage(plan, contract)
    assert result["status"] == "needs_review"
    assert result["missing_requirement_ids"] == ["third"]
    assert result["interface_requirement_ids"] == ["first"]
    assert [row["requirement_id"] for row in result["external_requirements"]] == ["second"]


def test_model_inputs_and_outputs_share_no_mutable_lists():
    frozen = Contract(
        requirements=[
            Requirement(id="function", text="Hold the load", source_quote="Hold the load")
        ]
    )
    plan = DesignPlan(
        objective="Prototype",
        requirement_dispositions=[
            RequirementDisposition(
                requirement_id="function", physical_validation=["Test after making"]
            )
        ],
    )
    before = (plan.model_dump_json(), frozen.model_dump_json())
    result = functional_coverage(plan, frozen)
    assert (plan.model_dump_json(), frozen.model_dump_json()) == before
    result["external_requirements"][0]["physical_validation"].append("Result-only note")
    assert plan.requirement_dispositions[0].physical_validation == ["Test after making"]


def test_dictionary_inputs_remain_exact_and_legacy_absence_is_not_rewritten():
    plan = {"objective": "Legacy design"}
    contract = {"requirements": [{"id": "legacy_function"}]}
    before = deepcopy((plan, contract))
    result = functional_coverage(plan, contract)
    assert result["missing_requirement_ids"] == ["legacy_function"]
    assert (plan, contract) == before
    assert "functional_interfaces" not in plan


def test_interface_and_external_notes_for_other_ids_cannot_hide_missing_intent():
    result = functional_coverage(
        {
            "functional_interfaces": [interface("other")],
            "requirement_dispositions": [
                {"requirement_id": "other", "unsupported_checks": ["No supported evaluator"]}
            ],
        },
        {"requirements": [requirement()]},
    )
    assert result["missing_requirement_ids"] == ["function"]
    assert result["interface_requirement_ids"] == []
    assert result["external_requirements"] == []


def test_mapping_without_real_interface_structure_is_rejected():
    with pytest.raises(ValueError):
        functional_coverage(
            {"functional_interfaces": [{"id": "fake", "requirement_ids": ["function"]}]},
            {"requirements": [requirement()]},
        )


@pytest.fixture
def coverage_pipeline(monkeypatch):
    """Use actual completion control flow with a saved document and stub receipts."""
    stores = []
    monkeypatch.setattr(
        "forge_mind.workspace.completion.read_sources", lambda *args: {"fingerprint": "source"}
    )
    monkeypatch.setattr("forge_mind.workspace.completion.engine_fingerprint", lambda: "engine")
    monkeypatch.setattr(
        "forge_mind.workspace.product_interfaces.validate_interfaces", lambda *args: []
    )
    monkeypatch.setattr(
        "forge_mind.workspace.engineering_service.run_report_job", lambda *args: None
    )
    monkeypatch.setattr(
        "forge_mind.workspace.native_finish.reference_collision_context",
        lambda *args: {"status": "not_applicable"},
    )

    def create(*, assembly=False, metric="unmeasured", external=False):
        frozen = Contract(
            requirements=[
                Requirement(
                    id="requested_behavior",
                    text="Hold the stated load" if metric == "unmeasured" else "Use 10 mm width",
                    source_quote="Hold the stated load"
                    if metric == "unmeasured"
                    else "Use 10 mm width",
                    metric=metric,
                    value=None if metric == "unmeasured" else 10,
                )
            ]
        ).model_dump()
        body = {"id": "body", "name": "Support", "kind": "box", "params": {"x": 10, "y": 8, "z": 4}}
        document = (
            {
                "assembly": {
                    "parts": [
                        {"id": "part", "name": "Support", "features": [body], "output": "body"}
                    ],
                    "roots": [{"id": "support", "definition": "part"}],
                }
            }
            if assembly
            else {"features": [body], "outputs": ["body"]}
        )
        request = {
            "tool": "tolerance_stack",
            "units": "mm",
            "terms": [{"low": 1.0, "high": 2.0}],
            "allowable": {"low": 0.0, "high": 3.0},
        }
        plan = DesignPlan.model_validate(
            {
                "objective": "Saved support prototype",
                "operation_plan": {
                    "part_ids": ["part"],
                    "steps": [
                        {
                            "id": "build",
                            "part_id": "part",
                            "feature_ids": ["body"],
                            "requirement_ids": ["requested_behavior"],
                            "expected_effect": "Build support",
                        }
                    ],
                },
                "subsystems": [
                    {
                        "id": "support",
                        "name": "Support",
                        "part_ids": ["part"],
                        "requirement_ids": ["requested_behavior"],
                        "checks": [{"id": "scalar", "request": request}],
                    }
                ],
                "requirement_dispositions": [
                    {
                        "requirement_id": "requested_behavior",
                        "check_ids": ["scalar"],
                        "unsupported_checks": [
                            "Whole holding behavior needs independent qualification"
                        ]
                        if external
                        else [],
                    }
                ],
            }
        ).model_dump()
        binding = snapshot(plan, 1, frozen)
        session = {
            "prompt": "Use 10 mm width" if metric != "unmeasured" else "Hold the stated load",
            "authoring_mode": "native_operations",
            "selected_revision": "a" * 32,
            "contract": frozen,
            "engineering": {"version": 0},
            "design_plan": plan,
            "plan_version": 1,
            "plan_hash": binding["plan_hash"],
            "plan_contract_hash": Contract.model_validate(frozen).fingerprint(),
            "operation_plan_snapshot": binding,
        }
        facts = {"solid_count": 1}
        if assembly:
            facts["assembly"] = {
                "occurrences": [{"path": "support", "definition": "part", "body_index": 0}]
            }
        revision = {
            "id": "a" * 32,
            "contract": frozen,
            "document": document,
            "operation_record": {"plan_snapshot": binding},
            "measurement": {
                "geometry_valid": True,
                "artifact_hashes": {"part.step": "saved-step", "document.json": "saved-document"},
                "findings": [
                    {
                        "requirement_id": "requested_behavior",
                        "status": "unverified" if metric == "unmeasured" else "pass",
                    }
                ],
                "facts": facts,
            },
        }
        if metric != "unmeasured":
            # A scoped tolerance-stack receipt is not a frozen dimension finding.
            # Supply the independent, complete saved-measurement fixture that this
            # positive scalar test has always intended. No geometry worker runs.
            revision["measurement"] = Measurement(
                revision_id=revision["id"],
                contract_hash=Contract.model_validate(frozen).fingerprint(),
                engine_fingerprint="engine",
                geometry_valid=True,
                artifact_hashes=revision["measurement"]["artifact_hashes"],
                facts=facts,
                findings=[
                    Finding(
                        requirement_id="requested_behavior",
                        status="pass",
                        measured=10.0,
                        expected=10.0,
                        method="Controlled saved STEP extent fixture",
                        reason="Complete scalar authority fixture; no native worker",
                    )
                ],
            ).model_dump()
        store = FakeStore(session)
        stores.append(store)
        service = SimpleNamespace(store=store, _load_revision=lambda sid, rid: deepcopy(revision))
        if metric != "unmeasured":
            monkeypatch.setattr(
                "forge_mind.workspace.requirement_verification._frozen_inputs",
                lambda *args: (
                    deepcopy(store.session),
                    deepcopy(revision),
                    {
                        "contract_hash": Contract.model_validate(frozen).fingerprint(),
                        "revision_contract_hash": Contract.model_validate(frozen).fingerprint(),
                        "engine_fingerprint": "engine",
                        "measured_engine_fingerprint": "engine",
                        "request_hash": digest(session["prompt"]),
                    },
                ),
            )
        calls = []

        def scalar_receipt(*args):
            calls.append("scalar")
            receipt = {
                "id": "scalar-receipt",
                "revision_id": revision["id"],
                "source_fingerprint": "source",
                "engine_fingerprint": "engine",
                "artifact_hashes": revision["measurement"]["artifact_hashes"],
                "contract": frozen,
                "request": args[4],
                "result": {
                    "status": "pass",
                    "tool": "tolerance_stack",
                    "algorithm_scope": _SCOPES["tolerance_stack"],
                },
            }
            encoded = json.dumps(receipt)
            store.connection.execute(
                "INSERT INTO product_checks VALUES (?,?,?,?)",
                (receipt["id"], "s", encoded, hashlib.sha256(encoded.encode()).hexdigest()),
            )
            return receipt

        monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", scalar_receipt)
        contract_hash = Contract.model_validate(frozen).fingerprint()
        monkeypatch.setattr(
            "forge_mind.workspace.native_finish.guidance_context",
            lambda *args: {
                "status": "current",
                "basis_hash": "current-guide",
                "basis": {"contract_hash": contract_hash, "session_contract_hash": contract_hash},
            },
        )
        return SimpleNamespace(
            service=service, revision=revision, calls=calls, job={"session": "s", "owner": "o"}
        )

    yield create
    for store in stores:
        store.connection.close()


@pytest.mark.parametrize("assembly", [False, True])
def test_native_scalar_only_coverage_blocks_actual_completion_and_finish(
    coverage_pipeline, assembly
):
    harness = coverage_pipeline(assembly=assembly)
    report = finalize_completion(harness.service, harness.job, None, lambda: False, "needs_review")
    assert harness.calls == ["scalar"]
    assert report["subsystems"][0]["status"] == "checked_pass"
    assert report["status"] == "unverified"
    assert report["functional_coverage"]["missing_requirement_ids"] == ["requested_behavior"]
    context = completion_context(harness.service, "s", "o", harness.revision["id"])
    assert not context["stale"]
    assert context["next_target"] == "functional_coverage"
    assert context["next_work_category"] == "review"
    assert context["status"] == "unverified"
    review = finish_context(harness.service, "s", "o", harness.revision["id"])
    assert review["status"] == "needs_work"
    assert review["functional_coverage"]["status"] == "needs_review"
    assert len(review["reasons"]) == 1
    assert harness.service.store.session["contract"] == harness.revision["contract"]


@pytest.mark.parametrize("assembly", [False, True])
def test_native_external_mapping_allows_draft_review_without_aggregate_pass(
    coverage_pipeline, assembly
):
    harness = coverage_pipeline(assembly=assembly, external=True)
    report = finalize_completion(harness.service, harness.job, None, lambda: False, "needs_review")
    assert harness.calls == ["scalar"]
    assert report["subsystems"][0]["status"] == "checked_pass"
    assert report["status"] == "unverified"
    assert report["functional_coverage"]["status"] == "unverified"
    context = completion_context(harness.service, "s", "o", harness.revision["id"])
    assert context["next_target"] is None
    assert context["status"] == "unverified"
    assert context["functional_coverage"]["external_requirements"][0]["unsupported_checks"] == (
        ["Whole holding behavior needs independent qualification"]
    )
    review = finish_context(harness.service, "s", "o", harness.revision["id"])
    assert review["status"] == "ready_for_validation"
    assert review["functional_coverage"]["status"] == "unverified"
    assert review["physical_readiness"] == "unverified"


@pytest.mark.parametrize("assembly", [False, True])
def test_native_known_metric_keeps_existing_aggregate_behavior(coverage_pipeline, assembly):
    harness = coverage_pipeline(assembly=assembly, metric="extent_x")
    report = finalize_completion(harness.service, harness.job, None, lambda: False, "needs_review")
    assert report["status"] == "checked_pass"
    assert report["functional_coverage"]["status"] == "not_applicable"
    context = completion_context(harness.service, "s", "o", harness.revision["id"])
    assert context["status"] == "checked_pass"
    assert context["next_target"] is None
    assert finish_context(harness.service, "s", "o", harness.revision["id"])["status"] == (
        "ready_for_validation"
    )


@pytest.mark.parametrize("external", [False, True])
def test_cached_scalar_green_cannot_override_current_unmeasured_coverage(
    coverage_pipeline, external
):
    harness = coverage_pipeline(external=external)
    finalize_completion(harness.service, harness.job, None, lambda: False, "needs_review")
    # A mutable report is guidance. Matching scalar receipts cannot grant broader proof.
    harness.service.store.session["completion_report"]["status"] = "checked_pass"
    context = completion_context(harness.service, "s", "o", harness.revision["id"])
    assert context["status"] == "unverified"
    assert context["functional_coverage"]["status"] == (
        "unverified" if external else "needs_review"
    )


def test_inventory_and_policy_are_only_unverified_mapping_obligations():
    result = functional_coverage(
        inventory_plan(), inventory_target(), source_text=INVENTORY_QUOTE + " " + SCOPE_QUOTE
    )
    assert result["status"] == "unverified"
    assert result["missing_requirement_ids"] == []
    rows = result["typed_requirements"]
    assert [row["mapping_status"] for row in rows] == [
        "subject_bindings_declared",
        "policy_obligation",
    ]
    assert all(row["status"] == "unverified" for row in rows)
    assert "next_target" not in result


def test_any_missing_inventory_subject_binding_is_actionable():
    plan = inventory_plan().model_dump(mode="json")
    plan["verification_bindings"].pop()
    result = functional_coverage(
        plan, inventory_target(), source_text=INVENTORY_QUOTE + " " + SCOPE_QUOTE
    )
    assert result["status"] == "needs_review"
    assert result["next_target"] == "verification_bindings"
    inventory = result["typed_requirements"][0]
    assert inventory["missing_subject_ids"] == ["spacers"]
    assert inventory["status"] == "unverified"


def test_unsupported_inventory_and_scope_retain_generic_strict_mapping():
    contract = inventory_target().model_dump(mode="json")
    contract["requirements"][0]["text"] += " and support a physical load"
    result = functional_coverage(
        inventory_plan(), contract, source_text=INVENTORY_QUOTE + " " + SCOPE_QUOTE
    )
    assert result["status"] == "needs_review"
    assert result["missing_requirement_ids"] == ["inventory", "scope"]
    assert "typed_requirements" not in result
