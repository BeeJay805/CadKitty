"""Sourced intent remains separate from saved CAD values and physical acceptance."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge_mind.workspace.document import (
    AssemblySpec,
    DesignDocument,
    Feature,
    Occurrence,
    PartDefinition,
    VariableRelation,
)
from forge_mind.workspace.models import (
    Contract,
    DesignPlan,
    DimensionFact,
    OperationPlan,
    OperationStep,
    Requirement,
)
from forge_mind.workspace.plan_reconciliation import (
    PlanReconciliation,
    alignment,
    reconcile_plan,
)


def fact(identifier="bore_original", value=20.4, **changes):
    values = {
        "id": identifier,
        "value": value,
        "source": "customer",
        "source_quote": f"Use a {value} mm bore",
        "binding_name": "bore",
        "part_id": "sleeve",
        "requirement_ids": ["bore_intent"],
    }
    values.update(changes)
    return DimensionFact.model_validate(values)


def plan():
    return DesignPlan(
        objective="Editable sleeve",
        unresolved=["Physical fit is unverified"],
        operation_plan=OperationPlan(
            part_ids=["sleeve"],
            dimensions=[fact()],
            steps=[
                OperationStep(
                    id="body_step",
                    part_id="sleeve",
                    feature_ids=["body"],
                    binding_names=["bore"],
                    requirement_ids=["bore_intent"],
                    expected_effect="Construct the sleeve",
                )
            ],
        ),
    )


def contract():
    return Contract(
        requirements=[
            Requirement(
                id="bore_intent",
                text="The sleeve must fit the supplied reference",
                source_quote="Fit the supplied reference",
                part_id="sleeve",
            )
        ]
    )


def document(value=20.4):
    return DesignDocument(
        variables={"bore": value},
        features=[
            Feature(
                id="body",
                name="Sleeve body",
                kind="cylinder",
                params={"height": 30},
                bindings={"radius": "bore"},
            )
        ],
        outputs=["body"],
    )


def update(*facts):
    return PlanReconciliation(
        expected_plan_version=3,
        revision_id="a" * 32,
        updates=list(facts),
    )


def successor(identifier="bore_updated", value=20.8, predecessor="bore_original"):
    return fact(identifier, value, supersedes_fact_id=predecessor)


def test_reconciliation_preserves_original_plan_contract_and_document():
    original, frozen, doc = plan(), contract(), document()
    saved = (original.model_dump(), frozen.model_dump(), doc.model_dump())
    result = reconcile_plan(original, update(successor()), doc, frozen)
    assert (original.model_dump(), frozen.model_dump(), doc.model_dump()) == saved
    assert (
        result.operation_plan.dimensions[0].model_dump()
        == saved[0]["operation_plan"]["dimensions"][0]
    )
    assert result.unresolved == ["Physical fit is unverified"]
    assert result.operation_plan.dimensions[-1].source_quote == "Use a 20.8 mm bore"
    assert alignment(result, doc)["status"] == "mismatch"
    assert alignment(result, document(20.8))["status"] == "aligned"


def test_chained_updates_and_identical_retries_are_harmless():
    first = successor()
    second = successor("bore_final", 21.0, "bore_updated")
    result = reconcile_plan(plan(), update(first, first, second), document(), contract())
    repeated = reconcile_plan(result, update(first, second), document(), contract())
    assert repeated.model_dump() == result.model_dump()
    report = alignment(result, document(21.0))
    assert report["status"] == "aligned"
    assert [row["fact_id"] for row in report["active_bindings"]] == ["bore_final"]


def test_conflicting_update_id_does_not_mutate_original():
    original = plan()
    before = original.model_dump()
    with pytest.raises(ValueError, match="ID conflicts"):
        reconcile_plan(original, update(successor(), successor(value=21)), document(), contract())
    assert original.model_dump() == before


def test_competing_successors_are_ambiguous_and_rejected():
    first, second = successor(), successor("other_update", 21.0)
    with pytest.raises(ValueError, match="conflicting active endpoints"):
        reconcile_plan(plan(), update(first, second), document(), contract())
    ambiguous = plan()
    ambiguous.operation_plan.dimensions.extend([first, second])
    report = alignment(ambiguous, document(20.8))
    assert report["status"] == "ambiguous"
    assert len(report["active_bindings"]) == 2
    assert all(row["status"] == "ambiguous" for row in report["active_bindings"])


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"supersedes_fact_id": None}, "explicit superseding"),
        ({"supersedes_fact_id": "unknown"}, "unknown fact"),
        ({"supersedes_fact_id": "bore_updated"}, "supersession cycle"),
        ({"units": "inch"}, "retain binding"),
        ({"part_id": "another"}, "retain binding"),
        ({"binding_name": "other"}, "retain binding"),
        ({"requirement_ids": ["unknown"]}, "unknown frozen requirement"),
    ],
)
def test_invalid_successor_is_rejected(changes, message):
    candidate = successor().model_copy(update=changes)
    with pytest.raises(ValueError, match=message):
        reconcile_plan(plan(), update(candidate), document(), contract())


def test_missing_binding_rejects_update_even_without_present_construction():
    original = plan()
    original.operation_plan.steps[0].feature_ids = ["future"]
    doc = document()
    doc.variables = {}
    doc.features[0].bindings = {}
    with pytest.raises(ValueError, match="unknown binding"):
        reconcile_plan(original, update(successor()), doc, contract())


def test_missing_document_part_scope_cannot_read_a_global_value_as_that_part():
    original = plan()
    body = document().features[0]
    doc = DesignDocument(
        variables={"bore": 20.8},
        assembly=AssemblySpec(
            parts=[PartDefinition(id="other", name="Other", features=[body], output="body")],
            roots=[Occurrence(id="one", definition="other")],
        ),
    )
    report = alignment(original, doc)
    assert report["status"] == "mismatch"
    assert report["active_bindings"][0]["current_value"] is None
    assert not report["active_bindings"][0]["part_scope_present"]
    with pytest.raises(ValueError, match="missing document part scope"):
        reconcile_plan(original, update(successor()), doc, contract())


def test_unbound_facts_are_retained_without_blocking_alignment():
    original = plan()
    unbound = fact("source_only").model_copy(update={"binding_name": None})
    original.operation_plan.dimensions.append(unbound)
    result = reconcile_plan(original, update(successor()), document(), contract())
    report = alignment(result, document(20.8))
    assert report["status"] == "aligned"
    assert report["unbound_fact_ids"] == ["source_only"]
    assert len(report["active_bindings"]) == 1
    assert result.operation_plan.dimensions[1].model_dump() == unbound.model_dump()


def test_alignment_reports_all_active_bindings_and_evaluated_relations():
    original = plan()
    original.operation_plan.dimensions.append(
        fact("half_bore", 10.2).model_copy(update={"binding_name": "radius"})
    )
    doc = document()
    doc.relations = [VariableRelation(name="radius", operation="divide", a="bore", b=2)]
    report = alignment(original, doc)
    assert report["status"] == "aligned"
    assert len(report["active_bindings"]) == 2
    assert report["active_bindings"][1]["current_value"] == 10.2
    assert "not geometry or physical fit" in report["scope"]


def test_existing_unbuilt_binding_remains_explicit_without_rejecting_other_updates():
    original = plan()
    original.operation_plan.dimensions.append(
        fact("future_dimension").model_copy(update={"binding_name": "future_input"})
    )
    result = reconcile_plan(original, update(successor()), document(), contract())
    report = alignment(result, document(20.8))
    assert report["status"] == "mismatch"
    rows = {row["fact_id"]: row for row in report["active_bindings"]}
    assert rows["future_dimension"]["status"] == "missing"
    assert rows["bore_updated"]["status"] == "aligned"


def test_alignment_has_numeric_tolerance_without_hiding_material_drift():
    assert alignment(plan(), document(20.4 + 1e-10))["status"] == "aligned"
    assert alignment(plan(), document(20.401))["status"] == "mismatch"


def test_cyclic_intent_cannot_be_reported_aligned_with_no_active_endpoint():
    original = plan()
    original.operation_plan.dimensions[0].supersedes_fact_id = "bore_updated"
    original.operation_plan.dimensions.append(successor())
    assert alignment(original, document())["status"] == "ambiguous"


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_fact_values_are_rejected(value):
    bad = successor().model_copy(update={"value": value})
    with pytest.raises(ValidationError):
        reconcile_plan(plan(), update(bad), document(), contract())


@pytest.mark.parametrize("version", [True, False, -1, 1.0, "1"])
def test_request_version_is_a_strict_nonnegative_integer(version):
    with pytest.raises(ValidationError):
        PlanReconciliation(
            expected_plan_version=version, revision_id="a" * 32, updates=[successor()]
        )


def test_request_requires_revision_and_sourced_nonempty_updates():
    with pytest.raises(ValidationError):
        PlanReconciliation(expected_plan_version=0, revision_id="bad", updates=[successor()])
    with pytest.raises(ValidationError):
        update()
    with pytest.raises(ValidationError):
        reconcile_plan(
            plan(),
            update(successor().model_copy(update={"source_quote": ""})),
            document(),
            contract(),
        )
    with pytest.raises(ValidationError):
        reconcile_plan(
            plan(),
            update(successor().model_copy(update={"source": "invented"})),
            document(),
            contract(),
        )
