"""Construction ownership and source parameter observations never prove an effect."""

from __future__ import annotations

import hashlib
import json

import pytest

from forge_mind.workspace.document import (
    AssemblySpec,
    DesignDocument,
    Feature,
    Occurrence,
    PartDefinition,
)
from forge_mind.workspace.models import (
    Contract,
    DesignPlan,
    DimensionFact,
    OperationPlan,
    OperationStep,
)
from forge_mind.workspace.operation_plan import coverage, snapshot, validate_plan


def fact(identifier="spacing40", value=40, **changes):
    return DimensionFact(
        id=identifier,
        value=value,
        source="customer",
        source_quote=f"Set spacing to {value} mm",
        part_id="main",
        binding_name="spacing",
        **changes,
    )


def plan(*, observations=True):
    steps = [
        OperationStep(
            id="base",
            part_id="main",
            feature_ids=["body"],
            binding_names=["spacing"],
            expected_effect="Build the plate",
        )
    ]
    if observations:
        steps.extend(
            [
                OperationStep(
                    id="inspect",
                    part_id="main",
                    kind="inspection",
                    depends_on=["base"],
                    referenced_feature_ids=["body"],
                    expected_effect="Inspect spacing",
                ),
                OperationStep(
                    id="deliver",
                    part_id="main",
                    kind="delivery",
                    depends_on=["inspect"],
                    referenced_feature_ids=["body"],
                    expected_effect="Capture the manufacturing package",
                ),
            ]
        )
    return DesignPlan(
        objective="A plate",
        operation_plan=OperationPlan(part_ids=["main"], dimensions=[fact()], steps=steps),
    )


def document(value=40):
    return DesignDocument(
        variables={"spacing": value},
        features=[
            Feature(
                id="body",
                name="Plate",
                kind="box",
                params={"y": 10, "z": 3},
                bindings={"x": "spacing"},
            )
        ],
        outputs=["body"],
    )


def report(value=None, doc=None):
    return coverage(snapshot(value or plan(), 1, Contract()), doc or document())


def test_legacy_default_serialization_and_snapshot_hash_are_unchanged():
    expected_fact = {
        "id": "spacing40",
        "value": 40.0,
        "source": "customer",
        "source_quote": "Set spacing to 40 mm",
        "units": "mm",
        "part_id": "main",
        "frame_id": None,
        "binding_name": "spacing",
        "requirement_ids": [],
    }
    expected_step = {
        "id": "base",
        "part_id": "main",
        "depends_on": [],
        "feature_ids": ["body"],
        "requirement_ids": [],
        "binding_names": ["spacing"],
        "frame_id": None,
        "expected_effect": "Build the plate",
        "inspection_targets": [],
    }
    legacy = {
        "objective": "A plate",
        "parts": [],
        "interfaces": [],
        "assembly_sequence": [],
        "verification": [],
        "unresolved": [],
        "subsystems": [],
        "requirement_dispositions": [],
        "inspection": [],
        "operation_plan": {
            "version": 1,
            "units": "mm",
            "part_ids": ["main"],
            "frames": [],
            "dimensions": [expected_fact],
            "steps": [expected_step],
            "checkpoint": "",
            "unresolved": [],
        },
    }
    value = plan(observations=False)
    assert value.model_dump(mode="json") == legacy
    frozen = snapshot(value, 1, Contract())
    assert (
        frozen["plan_hash"]
        == hashlib.sha256(json.dumps(legacy, sort_keys=True, allow_nan=False).encode()).hexdigest()
    )
    assert coverage(frozen, document())["graph_complete"]


def test_inspection_and_delivery_reference_geometry_without_claiming_completion():
    result = report()
    assert result["graph_complete"]
    assert result["references_complete"]
    for row in result["steps"][1:]:
        assert row["referenced_feature_ids_present"] == ["body"]
        assert row["feature_ids_present"] == []
        assert not row["graph_present"]
        assert row["references_present"]
        assert not row["effect_verified"]
        assert row["status"] == "pending"


def test_no_construction_cannot_be_graph_complete():
    value = plan()
    value.operation_plan.steps = value.operation_plan.steps[1:]
    value.operation_plan.steps[0].depends_on = []
    assert not report(value)["graph_complete"]


def test_legacy_unowned_step_can_gain_an_explicit_observation_role():
    previous = plan()
    previous.operation_plan.steps[1].kind = "construction"
    previous.operation_plan.steps[1].referenced_feature_ids = []
    value = previous.model_copy(deep=True)
    value.operation_plan.steps[1].kind = "inspection"
    value.operation_plan.steps[1].referenced_feature_ids = ["body"]
    validate_plan(value, Contract(), previous)
    assert report(value)["graph_complete"]
    assert not report(previous)["graph_complete"]


def test_existing_construction_cannot_be_hidden_by_changing_its_role():
    previous = plan()
    value = previous.model_copy(deep=True)
    value.operation_plan.steps[0].kind = "inspection"
    value.operation_plan.steps[0].referenced_feature_ids = ["body"]
    value.operation_plan.steps[0].feature_ids = []
    with pytest.raises(ValueError, match="owned Features cannot change kind"):
        validate_plan(value, Contract(), previous)


def test_missing_reference_is_separate_from_construction_completeness():
    value = plan()
    value.operation_plan.steps[1].referenced_feature_ids = ["future"]
    result = report(value)
    assert result["graph_complete"]
    assert not result["references_complete"]
    assert result["steps"][1]["missing_referenced_feature_ids"] == ["future"]
    assert result["steps"][2]["missing_dependency_steps"] == ["inspect"]


def two_part_reference_plan(*, repeated_feature_id=False):
    value = plan()
    value.operation_plan.part_ids.append("other")
    value.operation_plan.steps.append(
        OperationStep(
            id="other_base",
            part_id="other",
            feature_ids=["body" if repeated_feature_id else "other_body"],
            expected_effect="Build a separately scoped second part",
        )
    )
    return value


def test_declared_cross_part_reference_warns_without_claiming_missing_local_geometry():
    from forge_mind.workspace.operation_plan import reference_scope_warnings

    value = two_part_reference_plan()
    value.operation_plan.steps[1].referenced_feature_ids = ["body", "other_body"]
    before = value.model_dump()
    validate_plan(value, Contract())
    warnings = reference_scope_warnings(value)
    assert len(warnings) == 1
    assert warnings[0]["step_id"] == "inspect"
    assert warnings[0]["part_id"] == "main"
    assert warnings[0]["referenced_feature_id"] == "other_body"
    assert warnings[0]["declared_part_ids"] == ["other"]
    assert warnings[0]["status"] == "possible_scope_mismatch"
    assert "unowned local Feature may exist" in warnings[0]["message"]
    assert value.model_dump() == before


def test_duplicate_feature_ids_in_distinct_part_scopes_remain_valid():
    value = two_part_reference_plan(repeated_feature_id=True)
    validate_plan(value, Contract())
    value.operation_plan.steps.append(
        OperationStep(
            id="other_inspect",
            part_id="other",
            kind="inspection",
            referenced_feature_ids=["body"],
            expected_effect="Inspect the second part's body",
        )
    )
    validate_plan(value, Contract())
    from forge_mind.workspace.operation_plan import reference_scope_warnings

    assert reference_scope_warnings(value) == []


def test_unowned_local_reference_remains_valid_despite_another_planned_owner():
    value = two_part_reference_plan()
    value.operation_plan.steps[1].referenced_feature_ids = ["other_body"]
    doc = document()
    shared = Feature(
        id="other_body", name="Local feature", kind="box", params={"x": 2, "y": 2, "z": 2}
    )
    assembly = DesignDocument(
        variables=doc.variables,
        assembly=AssemblySpec(
            parts=[
                PartDefinition(
                    id="main", name="Main", features=[*doc.features, shared], output="body"
                ),
                PartDefinition(id="other", name="Other", features=[shared], output="other_body"),
            ],
            roots=[
                Occurrence(id="main_occ", definition="main"),
                Occurrence(id="other_occ", definition="other"),
            ],
        ),
    )
    validate_plan(value, Contract(), document=assembly)
    assert coverage(snapshot(value, 1, Contract()), assembly)["references_complete"]


def test_undeclared_future_reference_remains_incomplete_without_a_scope_guess():
    value = two_part_reference_plan()
    value.operation_plan.steps[1].referenced_feature_ids = ["future"]
    validate_plan(value, Contract())
    result = report(value)
    assert not result["references_complete"]
    assert result["steps"][1]["missing_referenced_feature_ids"] == ["future"]


def test_scope_warning_does_not_block_valid_plan_save_or_create_geometry(tmp_path):
    from forge_mind.workspace.models import Requirement
    from forge_mind.workspace.store import Store

    contract = Contract(
        requirements=[Requirement(id="mount", text="A mount", source_quote="A mount")]
    )
    store = Store(tmp_path)
    sid = store.create("owner", "A mount", contract.model_dump(), "native_operations")["id"]
    initial = store.set_design_plan(sid, "owner", plan(), 0)
    value = two_part_reference_plan()
    value.operation_plan.steps[1].referenced_feature_ids = ["other_body"]
    saved = store.set_design_plan(sid, "owner", value, initial["plan_version"])
    assert saved["plan_version"] == initial["plan_version"] + 1
    assert saved["plan"] == value.model_dump()
    assert store.session_record(sid, "owner")["design_plan"] == value.model_dump()
    assert store.revisions(sid) == []


def test_plan_scope_warnings_survive_durable_author_context():
    from forge_mind.workspace.agent_memory import recent_context, remember
    from forge_mind.workspace.operation_plan import reference_scope_warnings

    value = two_part_reference_plan()
    value.operation_plan.steps[1].referenced_feature_ids = ["other_body"]
    history = [{"plan_saved": True, "reference_scope_warnings": reference_scope_warnings(value)}]
    assert recent_context(remember(history), {}) == history


def test_plan_scope_warning_reaches_next_author_call_before_any_cad_submission(tmp_path):
    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Action, Requirement
    from forge_mind.workspace.store import Store

    value = two_part_reference_plan()
    value.operation_plan.steps[1].referenced_feature_ids = ["other_body"]
    seen = []

    class Author:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            seen.append(context)
            if len(seen) == 1:
                action = Action(
                    kind="plan",
                    explanation="Save scoped intent",
                    document_json=value.model_dump_json(),
                )
            else:
                warning = next(row for row in context["history"] if row.get("plan_saved"))[
                    "reference_scope_warnings"
                ][0]
                assert warning["status"] == "possible_scope_mismatch"
                assert warning["part_id"] == "main"
                assert warning["referenced_feature_id"] == "other_body"
                action = Action(
                    kind="schema", explanation="Review existing syntax", query="feature:box"
                )
            return action, LLMUsage("test", 0, 0)

        generate_structured_with_image = generate_structured

    store = Store(tmp_path)
    contract = Contract(
        requirements=[Requirement(id="mount", text="A mount", source_quote="A mount")]
    )
    sid = store.create("owner", "A mount", contract.model_dump(), "native_operations")["id"]
    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Author())
    store.enqueue(sid, "owner", {"max_actions": 2, "max_seconds": 60})
    job = store.claim()
    assert job
    service.run_job(job)
    assert len(seen) == 2
    assert store.revisions(sid) == []
    assert store.session_record(sid, "owner")["design_plan"] == value.model_dump()


def test_undeclared_reference_in_another_saved_part_keeps_document_guard():
    value = plan()
    value.operation_plan.part_ids.append("other")
    value.operation_plan.steps[1].referenced_feature_ids = ["other_body"]
    validate_plan(value, Contract())
    doc = document()
    main = PartDefinition(id="main", name="Main", features=doc.features, output="body")
    other = PartDefinition(
        id="other",
        name="Other",
        features=[
            Feature(id="other_body", name="Other body", kind="box", params={"x": 2, "y": 2, "z": 2})
        ],
        output="other_body",
    )
    assembly = DesignDocument(
        variables=doc.variables,
        assembly=AssemblySpec(
            parts=[main, other],
            roots=[
                Occurrence(id="main_occ", definition="main"),
                Occurrence(id="other_occ", definition="other"),
            ],
        ),
    )
    with pytest.raises(ValueError, match=r"inspect.*different part.*other_body"):
        coverage(snapshot(value, 1, Contract()), assembly)


def test_missing_construction_propagates_through_observation_dependencies():
    value = plan()
    value.operation_plan.steps[0].feature_ids = ["future"]
    result = report(value)
    assert not result["graph_complete"]
    assert result["steps"][1]["missing_dependency_steps"] == ["base"]
    assert result["steps"][2]["missing_dependency_steps"] == ["inspect"]


@pytest.mark.parametrize("kind", ["inspection", "delivery"])
def test_observation_steps_cannot_own_features(kind):
    value = plan()
    value.operation_plan.steps[1].kind = kind
    value.operation_plan.steps[1].feature_ids = ["body"]
    with pytest.raises(ValueError, match="Only construction steps"):
        validate_plan(value, Contract())


def test_duplicate_construction_owners_are_still_rejected():
    value = plan()
    value.operation_plan.steps.append(
        OperationStep(id="duplicate", part_id="main", feature_ids=["body"], expected_effect="Copy")
    )
    with pytest.raises(ValueError, match="more than one"):
        validate_plan(value, Contract())


def test_reference_cannot_select_a_feature_from_another_part():
    value = plan()
    value.operation_plan.part_ids.append("other")
    value.operation_plan.steps[1].referenced_feature_ids = ["other_body"]
    base = document().features[0]
    other = base.model_copy(update={"id": "other_body"})
    doc = DesignDocument(
        variables={"spacing": 40},
        assembly=AssemblySpec(
            parts=[
                PartDefinition(id="main", name="Main", features=[base], output="body"),
                PartDefinition(id="other", name="Other", features=[other], output="other_body"),
            ],
            roots=[
                Occurrence(id="one", definition="main"),
                Occurrence(id="two", definition="other"),
            ],
        ),
    )
    with pytest.raises(ValueError, match="different part"):
        report(value, doc)


def test_observation_dependency_cycle_is_rejected():
    value = plan()
    value.operation_plan.steps[0].depends_on = ["deliver"]
    with pytest.raises(ValueError, match="dependency cycle"):
        validate_plan(value, Contract())


def test_binding_edit_preserves_source_fact_and_original_snapshot():
    frozen = snapshot(plan(), 1, Contract())
    saved = json.dumps(frozen, sort_keys=True)
    original = coverage(frozen, document())
    edited = coverage(frozen, document(44))
    restored = coverage(frozen, document())
    assert original["dimension_bindings"][0]["status"] == "unchanged"
    row = edited["dimension_bindings"][0]
    assert (row["declared_value"], row["current_value"], row["delta"]) == (40, 44, 4)
    assert row["status"] == "changed"
    assert row["units"] == "mm"
    assert not row["superseded"]
    assert restored == original
    assert json.dumps(frozen, sort_keys=True) == saved
    assert "not geometry measurements or customer approval" in edited["dimension_binding_scope"]


def test_unbound_and_missing_facts_remain_explicit():
    value = plan()
    unbound = fact("unbound").model_copy(update={"binding_name": None})
    missing = fact("missing").model_copy(update={"binding_name": "not_yet_built"})
    value.operation_plan.dimensions.extend([unbound, missing])
    rows = {row["fact_id"]: row for row in report(value)["dimension_bindings"]}
    for identifier, state in [("unbound", "unbound"), ("missing", "missing")]:
        assert rows[identifier]["status"] == state
        assert rows[identifier]["current_value"] is None
        assert rows[identifier]["delta"] is None


def test_explicit_supersession_preserves_both_facts_and_marks_history():
    previous = plan()
    value = previous.model_copy(deep=True)
    value.operation_plan.dimensions.append(fact("spacing44", 44, supersedes_fact_id="spacing40"))
    validate_plan(value, Contract(), previous)
    result = report(value, document(44))
    old, new = result["dimension_bindings"]
    assert old["status"] == "changed"
    assert old["superseded"]
    assert old["superseded_by_fact_ids"] == ["spacing44"]
    assert new["status"] == "unchanged"
    assert not new["superseded"]
    assert previous.operation_plan.dimensions[0].value == 40
    assert len(previous.operation_plan.dimensions) == 1


@pytest.mark.parametrize("change", ["value", "source_quote", "remove"])
def test_same_contract_updates_must_retain_original_source_facts(change):
    previous = plan()
    value = previous.model_copy(deep=True)
    if change == "remove":
        value.operation_plan.dimensions.clear()
    elif change == "value":
        value.operation_plan.dimensions[0].value = 44
    else:
        value.operation_plan.dimensions[0].source_quote = "A different source"
    with pytest.raises(ValueError, match="Retain original dimension facts"):
        validate_plan(value, Contract(), previous)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"supersedes_fact_id": "unknown"}, "unknown fact"),
        ({"binding_name": "other"}, "retain binding"),
        ({"part_id": None}, "retain binding"),
        ({"units": "degrees"}, "retain binding"),
        ({"supersedes_fact_id": "spacing44"}, "supersession cycle"),
    ],
)
def test_invalid_supersession_is_rejected(changes, message):
    value = plan()
    newer = fact("spacing44", 44, supersedes_fact_id="spacing40").model_copy(update=changes)
    value.operation_plan.dimensions.append(newer)
    with pytest.raises(ValueError, match=message):
        validate_plan(value, Contract())


def test_multifact_supersession_cycle_is_rejected():
    value = plan()
    value.operation_plan.dimensions[0].supersedes_fact_id = "spacing44"
    value.operation_plan.dimensions.append(fact("spacing44", 44, supersedes_fact_id="spacing40"))
    with pytest.raises(ValueError, match="supersession cycle"):
        validate_plan(value, Contract())


def test_large_mixed_role_plan_has_no_list_cap():
    value = plan(observations=False)
    for index in range(600):
        value.operation_plan.steps.append(
            OperationStep(
                id=f"inspect{index}",
                part_id="main",
                kind="inspection" if index % 2 == 0 else "delivery",
                depends_on=["base"],
                referenced_feature_ids=["body"],
                expected_effect="Independent review obligation",
            )
        )
    result = report(value)
    assert len(result["steps"]) == 601
    assert result["graph_complete"]
    assert result["references_complete"]
    assert all(not row["effect_verified"] for row in result["steps"])


def test_parameter_prerequisites_are_source_structure_not_measured_effects():
    intent = plan(observations=False)
    intent.operation_plan.steps.insert(
        0,
        OperationStep(
            id="inputs",
            part_id="main",
            kind="parameters",
            binding_names=["spacing"],
            expected_effect="Define spacing before geometry",
        ),
    )
    intent.operation_plan.steps[1].depends_on = ["inputs"]
    source = DesignDocument(
        variables={"spacing": 40},
        features=[Feature(id="body", name="Plate", kind="box", params={"x": 10, "y": 8, "z": 4})],
        outputs=["body"],
    )
    record = coverage(snapshot(intent, 1, Contract()), source)
    assert record["graph_complete"]
    assert record["steps"][0]["parameters_present"]
    assert not any(row["effect_verified"] for row in record["steps"])
    source.variables.clear()
    intent.operation_plan.steps[1].binding_names = []
    record = coverage(snapshot(intent, 1, Contract()), source)
    assert not record["graph_complete"]
    assert record["steps"][1]["missing_dependency_steps"] == ["inputs"]


def test_parameter_steps_require_names_and_cannot_own_features():
    intent = plan(observations=False)
    intent.operation_plan.steps[0].kind = "parameters"
    with pytest.raises(ValueError, match="own"):
        validate_plan(intent, Contract())
    intent.operation_plan.steps[0].feature_ids = []
    intent.operation_plan.steps[0].binding_names = []
    with pytest.raises(ValueError, match="named bindings"):
        validate_plan(intent, Contract())


def test_partial_assembly_can_reuse_feature_ids_in_an_unbuilt_part():
    intent = DesignPlan(
        objective="Two plates",
        operation_plan=OperationPlan(
            part_ids=["first", "second"],
            steps=[
                OperationStep(
                    id=part, part_id=part, feature_ids=["base"], expected_effect="Build plate"
                )
                for part in ("first", "second")
            ],
        ),
    )
    source = DesignDocument(
        assembly=AssemblySpec(
            parts=[
                PartDefinition(
                    id="first",
                    name="First plate",
                    features=[
                        Feature(
                            id="base", name="Plate", kind="box", params={"x": 10, "y": 8, "z": 4}
                        )
                    ],
                    output="base",
                )
            ],
            roots=[Occurrence(id="first_occ", definition="first")],
        )
    )
    record = coverage(snapshot(intent, 1, Contract()), source)
    assert record["steps"][0]["graph_present"]
    assert not record["steps"][1]["graph_present"]
    assert record["steps"][1]["missing_feature_ids"] == ["base"]
