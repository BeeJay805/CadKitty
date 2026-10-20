"""Plan writes and admission bind intent without executing CAD or another model."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

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
    OperationFrame,
    OperationPlan,
    OperationStep,
    Requirement,
    WorkspaceRequest,
)
from forge_mind.workspace.operation_plan import coverage, snapshot, validate_plan
from forge_mind.workspace.store import Store, WorkspaceError


def contract(value=20):
    return Contract(
        requirements=[
            Requirement(
                id="width",
                text=f"Width {value} mm",
                source_quote=f"Width {value} mm",
                metric="extent_x",
                value=value,
            )
        ]
    )


def plan(*, future=False):
    steps = [
        OperationStep(
            id="base",
            part_id="main",
            feature_ids=["body"],
            requirement_ids=["width"],
            expected_effect="A plate with the requested width",
        )
    ]
    if future:
        steps.append(
            OperationStep(
                id="finish",
                part_id="main",
                depends_on=["base"],
                feature_ids=["rounded"],
                expected_effect="Rounded edges",
            )
        )
    return DesignPlan(
        objective="A plate",
        operation_plan=OperationPlan(
            part_ids=["main"],
            steps=steps,
        ),
    )


def document():
    return DesignDocument(
        features=[
            Feature(
                id="body",
                name="Plate",
                kind="box",
                params={"x": 20, "y": 10, "z": 3},
            )
        ],
        outputs=["body"],
    )


def native(tmp_path):
    store = Store(tmp_path)
    session = store.create("owner", "Width 20 mm", None, "native_operations")
    return store, session["id"]


def initialized(tmp_path):
    store, sid = native(tmp_path)
    binding = store.set_design_plan(sid, "owner", plan(), 0, contract=contract())
    return store, sid, binding


def payload(store, jid):
    with store.connect() as con:
        return json.loads(con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0])


def test_legacy_json_shape_is_exact_when_sections_absent(tmp_path):
    old = {
        "objective": "Plate",
        "parts": [],
        "interfaces": [],
        "assembly_sequence": [],
        "verification": [],
        "unresolved": [],
        "subsystems": [],
        "requirement_dispositions": [],
        "inspection": [],
    }
    assert DesignPlan.model_validate(old).model_dump() == old
    assert json.loads(DesignPlan.model_validate(old).model_dump_json()) == old
    request = WorkspaceRequest(prompt="Plate")
    assert request.model_dump() == {
        "prompt": "Plate",
        "contract": None,
        "max_actions": None,
        "max_seconds": None,
        "preview_only": True,
    }
    store = Store(tmp_path)
    session = store.create("owner", "Plate", None)
    assert set(session) == {
        "id",
        "prompt",
        "contract",
        "state",
        "selected_revision",
        "best_revision",
        "created_at",
        "events",
        "usage",
        "job_id",
    }
    jid = store.enqueue(session["id"], "owner", {"max_seconds": 30})
    application = store.session_record(session["id"], "owner")["contract_application"]
    assert application["applied"] is None
    assert application["pending"]["mode"] == "full"
    assert application["pending"]["source_count"] == 0
    # Public JSON stays exact; private admission records also bind unconsumed input.
    assert payload(store, jid) == {
        "max_seconds": 30,
        "based_on": None,
        "input_application_id": application["pending"]["input_id"],
        "input_instructions": [],
        "input_origin": "existing",
    }


def test_mode_is_persisted_only_when_explicit_and_reopen_preserves_it(tmp_path):
    store, sid = native(tmp_path)
    assert Store(tmp_path).authoring_mode(sid) == "native_operations"
    hybrid = store.create("owner", "Plate", None, "hybrid")
    assert hybrid["authoring_mode"] == "hybrid"
    with pytest.raises(WorkspaceError, match="authoring_mode"):
        store.create("owner", "Plate", None, "python")
    with pytest.raises(ValidationError):
        WorkspaceRequest(prompt="Plate", authoring_mode="python")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.operation_plan.steps.append(p.operation_plan.steps[0]), "unique"),
        (lambda p: p.operation_plan.steps[0].depends_on.append("unknown"), "dependency"),
        (lambda p: p.operation_plan.steps[0].depends_on.append("base"), "cycle"),
        (lambda p: p.operation_plan.steps[0].requirement_ids.append("unknown"), "requirement"),
        (lambda p: setattr(p.operation_plan.steps[0], "part_id", "unknown"), "part"),
    ],
)
def test_invalid_operation_graph_is_rejected(mutation, match):
    value = plan()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        validate_plan(value, contract())


def test_dependency_cycle_and_unique_feature_mapping_are_rejected():
    value = plan(future=True)
    value.operation_plan.steps[0].depends_on.append("finish")
    with pytest.raises(ValueError, match="cycle"):
        validate_plan(value, contract())
    value.operation_plan.steps[0].depends_on.clear()
    value.operation_plan.steps[1].feature_ids = ["body"]
    with pytest.raises(ValueError, match="more than one"):
        validate_plan(value, contract())


def test_future_steps_remain_pending_and_presence_never_proves_effects():
    value = plan(future=True)
    validate_plan(value, contract(), document=document())
    result = coverage(snapshot(value, 1, contract()), document())
    assert result["graph_complete"] is False
    assert result["steps"][0]["graph_present"] is True
    assert result["steps"][0]["effect_verified"] is False
    assert result["steps"][1]["missing_feature_ids"] == ["rounded"]


def test_snapshot_is_detached_and_tampering_fails():
    value = plan()
    target = contract()
    binding = snapshot(value, 4, target)
    value.objective = "Changed"
    target.requirements[0].value = 30
    assert binding["plan"]["objective"] == "A plate"
    assert binding["contract"]["requirements"][0]["value"] == 20
    coverage(binding, document())
    binding["plan"]["objective"] = "Changed"
    with pytest.raises(ValueError, match="hash"):
        coverage(binding, document())


def test_initialize_contract_plus_plan_is_atomic_and_owner_scoped(tmp_path):
    store, sid = native(tmp_path)
    with pytest.raises(WorkspaceError, match="not found"):
        store.set_design_plan(sid, "other", plan(), 0, contract())
    bad = plan()
    bad.operation_plan.steps[0].requirement_ids.append("missing")
    with pytest.raises(ValueError, match="requirement"):
        store.set_design_plan(sid, "owner", bad, 0, contract())
    assert store.session_record(sid, "owner")["contract"] is None
    assert store.plan_snapshot(sid, "owner") is None
    binding = store.set_design_plan(sid, "owner", plan(), 0, contract())
    current = store.session_record(sid, "owner")
    assert current["contract"] == binding["contract"]
    assert current["plan_version"] == 1
    assert store.plan_snapshot(sid, "owner") == binding
    with pytest.raises(WorkspaceError, match="not found"):
        store.plan_snapshot(sid, "other")


@pytest.mark.parametrize("value", [None, Contract()])
def test_native_plan_requires_meaningful_contract(tmp_path, value):
    store, sid = native(tmp_path)
    with pytest.raises(WorkspaceError, match="initialized contract"):
        store.set_design_plan(sid, "owner", plan(), 0, value)


def test_native_plan_requires_operations_and_no_build_before_initialization(tmp_path):
    store, sid = native(tmp_path)
    with pytest.raises(WorkspaceError, match="operation plan"):
        store.set_design_plan(sid, "owner", DesignPlan(objective="Plate"), 0, contract())
    with pytest.raises(WorkspaceError, match="before construction"):
        store.enqueue(sid, "owner", {"document": document().model_dump()})
    assert not store.session_record(sid, "owner")["job_id"]


def test_initial_builtin_author_can_own_first_plan_write(tmp_path):
    store, sid = native(tmp_path)
    jid = store.enqueue(sid, "owner", {})
    assert store.claim()["id"] == jid
    binding = store.set_design_plan_for_job(sid, "owner", jid, plan(), 0, contract())
    assert binding["plan_version"] == 1
    assert store.session_record(sid, "owner")["state"] == "queued"


def test_external_plan_write_cannot_enter_active_job_or_use_wrong_job(tmp_path):
    store, sid, _ = initialized(tmp_path)
    jid = store.enqueue(sid, "owner", {})
    with pytest.raises(WorkspaceError, match="active operation"):
        store.set_design_plan(sid, "owner", plan(), 1)
    with pytest.raises(WorkspaceError, match="owned running"):
        store.set_design_plan_for_job(sid, "owner", jid, plan(), 1)
    assert store.claim()["id"] == jid
    with pytest.raises(WorkspaceError, match="owned running"):
        store.set_design_plan_for_job(sid, "owner", "b" * 32, plan(), 1)
    store.cancel(sid, "owner")
    with pytest.raises(WorkspaceError, match="owned running"):
        store.set_design_plan_for_job(sid, "owner", jid, plan(), 1)


def test_plan_version_compare_and_set_across_connections(tmp_path):
    store, sid, _ = initialized(tmp_path)

    def save(_):
        try:
            return Store(tmp_path).set_design_plan(sid, "owner", plan(), 1)["plan_version"]
        except WorkspaceError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, range(2)))
    assert sorted(x for x in results if x is not None) == [2]
    assert results.count(None) == 1
    assert len(store.session_record(sid, "owner")["design_plan_history"]) == 1


def test_current_revision_and_engineering_inputs_are_checked(tmp_path):
    store, sid, _ = initialized(tmp_path)
    store.update(sid, selected_revision="a" * 32)
    with pytest.raises(WorkspaceError, match="Design changed"):
        store.set_design_plan(sid, "owner", plan(), 1, expected_current_revision=None)
    with pytest.raises(WorkspaceError, match="Design changed"):
        store.set_design_plan(sid, "owner", plan(), 1, expected_current_revision="b" * 32)
    store.save_engineering(sid, "owner", 0, {})
    with pytest.raises(WorkspaceError, match="inputs changed"):
        store.set_design_plan(sid, "owner", plan(), 1, expected_input_version=0)
    store.set_design_plan(
        sid, "owner", plan(), 1, expected_current_revision="a" * 32, expected_input_version=1
    )


def test_authoring_mode_cannot_be_relaxed_or_retroactively_changed(tmp_path):
    store, sid = native(tmp_path)
    with pytest.raises(WorkspaceError, match="immutable"):
        store.update(sid, authoring_mode="hybrid")
    store.update(sid, authoring_mode="native_operations")
    assert store.authoring_mode(sid) == "native_operations"
    legacy = store.create("owner", "Plate", None)["id"]
    with pytest.raises(WorkspaceError, match="immutable"):
        store.update(legacy, authoring_mode="native_operations")
    store.update(legacy, authoring_mode="hybrid")
    assert "authoring_mode" not in store.session_record(legacy, "owner")


@pytest.mark.parametrize("change", ["value", "omit", "tolerance", "critical"])
def test_plan_repairs_cannot_weaken_frozen_contract(tmp_path, change):
    store, sid, _ = initialized(tmp_path)
    target = contract()
    if change == "value":
        target.requirements[0].value = 1
    elif change == "omit":
        target.requirements = []
    elif change == "tolerance":
        target.requirements[0].tolerance = 100
    else:
        target.requirements[0].critical = False
    with pytest.raises(WorkspaceError, match=r"frozen|initialized contract"):
        store.set_design_plan(sid, "owner", plan(), 1, target)
    assert store.plan_snapshot(sid, "owner")["contract"] == contract().model_dump(mode="json")


def test_queued_document_captures_exact_plan_and_contract(tmp_path):
    store, sid, original = initialized(tmp_path)
    jid = store.enqueue(sid, "owner", {"document": document().model_dump(), "preview_only": True})
    captured = payload(store, jid)["plan_snapshot"]
    assert captured == original
    assert payload(store, jid)["contract"] == original["contract"]
    with pytest.raises(WorkspaceError, match="active operation"):
        store.set_design_plan(sid, "owner", plan(future=True), 1)
    # Simulate a later server record mutation; queued JSON remains independent.
    store.update(sid, design_plan={"objective": "Changed"})
    assert payload(store, jid)["plan_snapshot"] == captured


def test_native_enqueue_rejects_override_and_snapshot_injection(tmp_path):
    store, sid, binding = initialized(tmp_path)
    with pytest.raises(WorkspaceError, match="frozen"):
        store.enqueue(
            sid,
            "owner",
            {"document": document().model_dump(), "contract": contract(1).model_dump()},
        )
    with pytest.raises(WorkspaceError, match="server-owned"):
        store.enqueue(sid, "owner", {"document": document().model_dump(), "plan_snapshot": binding})
    with pytest.raises(WorkspaceError, match="authored Python"):
        store.enqueue(sid, "owner", {"code": "pass"})
    assert store.session_record(sid, "owner")["job_id"] is None


def test_operation_count_has_no_artificial_cap():
    value = plan()
    value.operation_plan.steps = [
        OperationStep(
            id=f"step{i}",
            part_id="main",
            feature_ids=[f"feature{i}"],
            expected_effect="Future geometry",
        )
        for i in range(600)
    ]
    validate_plan(value, contract())
    assert len(coverage(snapshot(value, 1, contract()), document())["steps"]) == 600


def test_future_bindings_need_not_exist_before_their_construction():
    value = plan(future=True)
    value.operation_plan.steps[1].binding_names = ["future_radius"]
    validate_plan(value, contract(), document=document())
    value.operation_plan.steps[0].binding_names = ["missing_width"]
    with pytest.raises(ValueError, match="unknown bindings"):
        validate_plan(value, contract(), document=document())


def test_transitive_missing_dependency_is_not_complete():
    value = plan(future=True)
    value.operation_plan.steps[0].feature_ids = ["not_built"]
    value.operation_plan.steps[1].feature_ids = ["body"]
    value.operation_plan.steps.append(
        OperationStep(
            id="third",
            part_id="main",
            feature_ids=["body2"],
            depends_on=["finish"],
            expected_effect="Later step",
        )
    )
    doc = document()
    doc.features.append(
        Feature(id="body2", name="Body 2", kind="box", params={"x": 2, "y": 2, "z": 2})
    )
    results = coverage(snapshot(value, 1, contract()), doc)["steps"]
    assert results[1]["missing_dependency_steps"] == ["base"]
    assert results[2]["missing_dependency_steps"] == ["finish"]
    assert all(not r["graph_present"] for r in results)


def test_generic_author_cannot_override_contract_without_customer_instruction(tmp_path):
    store, sid, _ = initialized(tmp_path)
    with pytest.raises(WorkspaceError, match="frozen"):
        store.enqueue(sid, "owner", {"contract": contract(21).model_dump()})
    with pytest.raises(WorkspaceError, match="frozen"):
        store.enqueue(
            sid,
            "owner",
            {"contract": contract(21).model_dump(), "instruction": "Show a cross section"},
        )


def test_explicit_customer_edit_reuses_source_grounding_and_records_identity(tmp_path):
    store, sid, binding = initialized(tmp_path)
    target = contract(21)
    target.requirements[0].source_quote = "Change Width to 21 mm"
    target.requirements[0].position_tolerance = 0.001
    jid = store.enqueue(
        sid,
        "owner",
        {
            "contract": target.model_dump(),
            "instruction": "Change Width to 21 mm",
        },
    )
    job = payload(store, jid)
    assert job["contract_change"] == {
        "previous_contract_hash": binding["contract_hash"],
        "contract_hash": target.fingerprint(),
        "source_instruction": "Change Width to 21 mm",
    }
    assert store.claim()["id"] == jid
    # Source-checked prepare_job publishes the target and consumed input together.
    store.apply_input_contract(sid, "owner", jid, job["input_application_id"], target)
    application = store.session_record(sid, "owner")["contract_application"]
    assert application["pending"] is None
    assert application["applied"]["input_id"] == job["input_application_id"]
    changed = store.set_design_plan_for_job(sid, "owner", jid, plan(), 1, target)
    assert changed["contract_hash"] == target.fingerprint()
    assert (
        store.session_record(sid, "owner")["design_plan_history"][0]["contract_hash"]
        == binding["contract_hash"]
    )


def test_single_part_plan_can_use_named_part_without_extra_scope_declaration():
    value = plan()
    value.operation_plan.part_ids = []
    value.operation_plan.steps[0].part_id = "plate"
    assert coverage(snapshot(value, 1, contract()), document())["graph_complete"]


def test_known_assembly_part_cannot_claim_a_feature_in_another_part():
    value = plan()
    value.operation_plan.part_ids = ["main", "other"]
    base = document().features[0]
    other = base.model_copy(update={"id": "other_body"})
    doc = DesignDocument(
        assembly=AssemblySpec(
            parts=[
                PartDefinition(id="main", name="Main", features=[base], output="body"),
                PartDefinition(id="other", name="Other", features=[other], output="other_body"),
            ],
            roots=[
                Occurrence(id="one", definition="main"),
                Occurrence(id="two", definition="other"),
            ],
        )
    )
    value.operation_plan.steps[0].feature_ids = ["other_body"]
    with pytest.raises(ValueError, match="different part"):
        validate_plan(value, contract(), document=doc)


def test_frames_and_dimension_facts_keep_sources_and_scopes():
    value = plan()
    value.operation_plan.frames = [
        OperationFrame(
            id="datum",
            part_id="main",
            description="Plate lower corner",
            origin=(1, 2, 3),
        )
    ]
    value.operation_plan.dimensions = [
        DimensionFact(
            id="width_fact",
            value=20,
            source="customer",
            source_quote="Width 20 mm",
            part_id="main",
            frame_id="datum",
            binding_name="width",
            requirement_ids=["width"],
        )
    ]
    value.operation_plan.steps[0].frame_id = "datum"
    value.operation_plan.steps[0].binding_names = ["width"]
    doc = document()
    doc.variables["width"] = 20
    doc.features[0].params.pop("x")
    doc.features[0].bindings["x"] = "width"
    validate_plan(value, contract(), document=doc)
    binding = snapshot(value, 1, contract())
    assert binding["plan"]["operation_plan"]["dimensions"][0]["source"] == "customer"
    value.operation_plan.frames[0].part_id = "other"
    value.operation_plan.part_ids.append("other")
    with pytest.raises(ValueError, match="frame belongs"):
        validate_plan(value, contract())


def test_native_restore_checks_retained_source_policy_without_execution(tmp_path):
    store, sid, _ = initialized(tmp_path)
    rid = "a" * 32
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    source = DesignDocument(
        features=[
            Feature(
                id="body",
                name="Python",
                kind="python",
                code="result = b.Box(20, 10, 3)",
            )
        ],
        outputs=["body"],
    )
    (directory / "document.json").write_text(source.model_dump_json(), encoding="utf-8")
    store.add_revision(sid, {"id": rid})
    with pytest.raises(WorkspaceError, match="body"):
        store.enqueue(sid, "owner", {"restore_only": True, "revision_id": rid})
    assert store.session_record(sid, "owner")["job_id"] is None


def test_preview_customer_contract_then_typed_edit_then_accept_keeps_targets(tmp_path):
    store, sid, baseline = initialized(tmp_path)
    accepted = "a" * 32
    store.add_revision(sid, {"id": accepted, "operation_record": {"plan_snapshot": baseline}})
    store.update(sid, selected_revision=accepted, accepted_revision=accepted)
    baseline_application = store.session_record(sid, "owner")["contract_application"]
    target = contract(21)
    target.requirements[0].source_quote = "Change Width to 21 mm"
    target.requirements[0].position_tolerance = 0.001
    jid = store.enqueue(
        sid,
        "owner",
        {
            "contract": target.model_dump(),
            "instruction": "Change Width to 21 mm",
            "preview_only": True,
        },
    )
    assert store.claim()["id"] == jid
    job = payload(store, jid)
    store.apply_input_contract(sid, "owner", jid, job["input_application_id"], target)
    candidate_binding = store.set_design_plan_for_job(sid, "owner", jid, plan(), 1, target)
    candidate_state = store.session_record(sid, "owner")
    candidate_application = candidate_state["contract_application"]
    candidate_instructions = candidate_state["instructions"]
    assert candidate_instructions == ["Change Width to 21 mm"]
    assert candidate_application["pending"] is None
    assert candidate_application["applied"]["input_id"] == job["input_application_id"]
    assert candidate_application["applied"]["source_count"] == 1
    candidate = "b" * 32
    store.add_revision(
        sid, {"id": candidate, "operation_record": {"plan_snapshot": candidate_binding}}
    )
    store.update(sid, selected_revision=candidate)
    store.finish(jid, sid, "needs_review")
    store.complete_preview(
        sid,
        jid,
        accepted,
        {
            "revision_id": candidate,
            "based_on": accepted,
            "input_version": 0,
            "eligible": True,
            "contract": target.model_dump(),
            "instructions": candidate_instructions,
        },
    )
    session = store.session_record(sid, "owner")
    assert session["contract"] == baseline["contract"]
    assert session["instructions"] == []
    assert session["contract_application"] == baseline_application
    assert store.plan_snapshot(sid, "owner") == candidate_binding
    assert session["proposal"]["operation_plan_snapshot"] == candidate_binding
    assert session["proposal"]["instructions"] == candidate_instructions
    assert session["proposal"]["contract_application"] == candidate_application
    next_job = store.enqueue(
        sid,
        "owner",
        {
            "document": document().model_dump(),
            "revision_id": candidate,
            "preview_only": True,
            "expected_current_revision": candidate,
        },
    )
    assert payload(store, next_job)["plan_snapshot"] == candidate_binding
    assert payload(store, next_job)["contract"] == candidate_binding["contract"]
    assert payload(store, next_job)["baseline_plan_snapshot"] == baseline
    typed_payload = payload(store, next_job)
    assert typed_payload["target_instructions"] == candidate_instructions
    assert typed_payload["target_contract_application"] == candidate_application
    assert store.claim()["id"] == next_job
    # The typed worker promotes its admitted target history before publishing geometry.
    store.update(sid, contract=target.model_dump())
    typed_candidate = store.session_record(sid, "owner")
    assert typed_candidate["instructions"] == candidate_instructions
    assert typed_candidate["contract_application"] == candidate_application
    final = "c" * 32
    store.add_revision(sid, {"id": final, "operation_record": {"plan_snapshot": candidate_binding}})
    store.update(sid, selected_revision=final)
    store.finish(next_job, sid, "needs_review")
    store.complete_preview(
        sid,
        next_job,
        accepted,
        {
            "revision_id": final,
            "based_on": accepted,
            "input_version": 0,
            "eligible": True,
            "contract": target.model_dump(),
            "instructions": typed_candidate["instructions"],
        },
    )
    preview = store.session_record(sid, "owner")
    assert preview["contract"] == baseline["contract"]
    assert preview["instructions"] == []
    assert preview["contract_application"] == baseline_application
    assert preview["proposal"]["contract_application"] == candidate_application
    store.accept_proposal(sid, "owner", final, accepted, 0)
    session = store.session_record(sid, "owner")
    assert session["selected_revision"] == final
    assert session["contract"] == candidate_binding["contract"]
    assert session["accepted_operation_plan_snapshot"] == candidate_binding
    assert session["instructions"] == candidate_instructions
    assert session["contract_application"] == candidate_application


def test_abandoned_preview_restores_baseline_plan_without_rebinding_it(tmp_path):
    store, sid, baseline = initialized(tmp_path)
    jid = store.enqueue(sid, "owner", {"preview_only": True})
    assert store.claim()["id"] == jid
    changed = plan(future=True)
    store.set_design_plan_for_job(sid, "owner", jid, changed, 1)
    store.finish(jid, sid, "error")
    store.complete_preview(sid, jid, None, None)
    assert store.plan_snapshot(sid, "owner") == baseline
    assert store.session_record(sid, "owner")["plan_version"] == 2


def test_plan_snapshot_cannot_rebind_intent_after_session_contract_changes(tmp_path):
    store, sid, baseline = initialized(tmp_path)
    store.update(sid, contract=contract(30).model_dump())
    assert store.plan_snapshot(sid, "owner") == baseline
    with pytest.raises(WorkspaceError, match="before construction"):
        store.enqueue(sid, "owner", {"document": document().model_dump()})


def test_same_contract_failed_preview_restores_exact_accepted_plan_then_accepts_candidate(tmp_path):
    store, sid, baseline = initialized(tmp_path)
    accepted = "a" * 32
    store.add_revision(sid, {"id": accepted, "operation_record": {"plan_snapshot": baseline}})
    store.update(sid, accepted_revision=accepted, selected_revision=accepted)
    acceptance_keys = (
        "accepted_revision",
        "accepted_operation_plan_snapshot",
        "accepted_operation_plan_revision",
    )
    original = store.session_record(sid, "owner")
    acceptance_before = {key: original[key] for key in acceptance_keys if key in original}
    repair = plan(future=True)
    repair.objective = "Repair finishing construction"
    candidate_binding = store.set_design_plan(sid, "owner", repair, 1)
    assert baseline["contract_hash"] == candidate_binding["contract_hash"]
    assert baseline["plan_hash"] != candidate_binding["plan_hash"]
    failed = store.enqueue(
        sid, "owner", {"document": document().model_dump(), "preview_only": True}
    )
    assert payload(store, failed)["plan_snapshot"] == candidate_binding
    assert payload(store, failed)["baseline_plan_snapshot"] == baseline
    assert store.claim()["id"] == failed
    store.finish(failed, sid, "error")
    store.complete_preview(sid, failed, accepted, None)
    session = store.session_record(sid, "owner")
    assert session["accepted_revision"] == accepted
    assert session["selected_revision"] == accepted
    assert {key: session[key] for key in acceptance_keys if key in session} == acceptance_before
    assert store.plan_snapshot(sid, "owner") == baseline
    candidate_binding = store.set_design_plan(sid, "owner", repair, 2)
    successful = store.enqueue(
        sid, "owner", {"document": document().model_dump(), "preview_only": True}
    )
    assert payload(store, successful)["baseline_plan_snapshot"] == baseline
    assert store.claim()["id"] == successful
    candidate = "b" * 32
    store.add_revision(
        sid, {"id": candidate, "operation_record": {"plan_snapshot": candidate_binding}}
    )
    store.update(sid, selected_revision=candidate)
    store.finish(successful, sid, "needs_review")
    store.complete_preview(
        sid,
        successful,
        accepted,
        {
            "revision_id": candidate,
            "based_on": accepted,
            "input_version": 0,
            "eligible": True,
            "contract": contract().model_dump(),
        },
    )
    proposed = store.session_record(sid, "owner")
    assert {key: proposed[key] for key in acceptance_keys if key in proposed} == acceptance_before
    store.accept_proposal(sid, "owner", candidate, accepted, 0)
    session = store.session_record(sid, "owner")
    assert session["accepted_operation_plan_revision"] == candidate
    assert session["accepted_operation_plan_snapshot"] == candidate_binding
    assert store.plan_snapshot(sid, "owner") == candidate_binding


def test_accepted_plan_cache_requires_exact_revision_correspondence(tmp_path):
    store, sid, baseline = initialized(tmp_path)
    accepted = "a" * 32
    store.add_revision(sid, {"id": accepted})
    store.update(
        sid,
        accepted_revision=accepted,
        selected_revision=accepted,
        accepted_operation_plan_snapshot=baseline,
        accepted_operation_plan_revision="b" * 32,
    )
    with pytest.raises(WorkspaceError, match="Accepted native revision"):
        store.enqueue(sid, "owner", {"preview_only": True})
    store.update(sid, accepted_operation_plan_revision=accepted)
    jid = store.enqueue(sid, "owner", {"preview_only": True})
    assert payload(store, jid)["baseline_plan_snapshot"] == baseline
