"""Ordinary wording has bound CAD subchecks, never a physical success shortcut."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError
from tests.unit.workspace.source_delivery_fixtures import source_delivery_wire_fixture

from forge_mind.workspace.customer_interpretation import (
    attach_customer_interpretation,
    customer_nominal_conformance,
    validate_customer_interpretation,
    validate_server_interpretation_proposal,
)
from forge_mind.workspace.customer_interpretation_models import CustomerNominalTarget
from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import ContractDraft, IntentAttribution, IntentIssue
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.measure import ground_contract
from forge_mind.workspace.models import Requirement
from forge_mind.workspace.requirement_verification import _intent_authority, aggregate
from forge_mind.workspace.source_contract_draft import compile_source_contract_draft
from forge_mind.workspace.source_delivery_contract_draft import (
    SourceDeliveryContractDraft,
    lower_source_delivery_draft,
    source_delivery_proposal_record,
)
from forge_mind.workspace.verification_admission import validate_verification_contract

SOURCE = (
    "Make a 60 mm by 40 mm by 5 mm PLA plate with four 4 mm through holes, "
    "each 8 mm from its adjacent edges. It must hold 2 kg."
)
DIMENSIONS = SOURCE.split(". It")[0] + "."


def ordinary_draft() -> ContractDraft:
    return ContractDraft.model_validate(
        {
            "contract": {
                "requirements": [
                    {"id": "body_duty", "text": DIMENSIONS, "source_quote": DIMENSIONS},
                    {
                        "id": "load_duty",
                        "text": "It must hold 2 kg.",
                        "source_quote": "It must hold 2 kg.",
                    },
                ]
            },
            "attribution": {
                "subjects": [
                    {"id": "plate", "label": "plate", "role": "designed", "source_quote": "plate"}
                ],
                "facts": [
                    {
                        "id": key,
                        "subject_id": "plate",
                        "property": key,
                        "value": value,
                        "unit": unit,
                        "source_quote": quote,
                        "requirement_ids": ["body_duty"],
                    }
                    for key, value, unit, quote in [
                        ("length", 60.0, "mm", "60 mm"),
                        ("width", 40.0, "mm", "40 mm"),
                        ("thickness", 5.0, "mm", "5 mm"),
                        ("bore", 4.0, "mm", "4 mm"),
                        ("edge", 8.0, "mm", "8 mm"),
                        ("count", 4.0, "holes", "four 4 mm through holes"),
                    ]
                ],
                "requirement_subjects": {"body_duty": ["plate"], "load_duty": ["plate"]},
            },
        }
    )


def target() -> CustomerNominalTarget:
    fields = [
        {"field": f"size_mm.{i}", "terms": [{"fact_id": key}]}
        for i, key in enumerate(["length", "width", "thickness"])
    ] + [{"field": f"center_mm.{i}", "basis": "design_origin"} for i in range(3)]
    holes = []
    for index, (x, y) in enumerate([(-22.0, -12.0), (-22.0, 12.0), (22.0, -12.0), (22.0, 12.0)]):
        holes.append(
            {
                "center_mm": [x, y],
                "diameter_mm": 4.0,
                "diameter_tolerance_mm": 0.0,
                "position_tolerance_mm": 0.0,
            }
        )
        fields.append({"field": f"holes.{index}.diameter_mm", "terms": [{"fact_id": "bore"}]})
        for axis, (key, value) in enumerate([("length", x), ("width", y)]):
            fields.append(
                {
                    "field": f"holes.{index}.center_mm.{axis}",
                    "terms": [
                        {"fact_id": key, "operation": "negative_half" if value < 0 else "half"},
                        {"fact_id": "edge", "operation": "value" if value < 0 else "negative"},
                    ],
                }
            )
    return CustomerNominalTarget.model_validate(
        {
            "requirement_id": "cad_body",
            "source_requirement_ids": ["body_duty"],
            "frame_policy": "centered_body",
            "hole_count_fact_id": "count",
            "field_bindings": fields,
            "criterion": {
                "kind": "nominal_perforated_box",
                "scope": "nominal_cad",
                "subject_id": "plate",
                "subject_role": "designed",
                "frame": "definition_local",
                "axis": "z",
                "count_policy": "exact",
                "size_mm": [60.0, 40.0, 5.0],
                "center_mm": [0.0, 0.0, 0.0],
                "holes": holes,
                "source_quote": DIMENSIONS,
            },
        }
    )


def wire():
    value = source_delivery_wire_fixture(ordinary_draft())
    value["nominal_targets"] = [target().model_dump(mode="json")]
    return SourceDeliveryContractDraft.model_validate(value)


def retained_proposal(proposal=None, *, job="job"):
    proposal = proposal or wire()
    lowered = lower_source_delivery_draft(proposal, SOURCE)
    return lowered, {
        "intent_compilation": {
            "job_id": job,
            "source_hash": digest(SOURCE),
            "status": "ready",
            "source_proposal": source_delivery_proposal_record(proposal, SOURCE, lowered),
        }
    }


def test_actual_initial_source_pipeline_admits_ordinary_wording_and_preserves_source():
    original = ordinary_draft()
    lowered, retained = retained_proposal()
    assert lowered.contract.requirements[:-1] == original.contract.requirements
    assert original.attribution.customer_nominal_interpretation is None
    assert lowered.contract.requirements[-1].critical is False
    assert compile_intent_draft(lowered, SOURCE)["status"] == "ready"
    assert (
        compile_source_contract_draft(lowered, SOURCE, source_only_requirement_ids=())["status"]
        == "ready"
    )
    assert validate_customer_interpretation(lowered.contract, SOURCE, lowered.attribution) == {
        "cad_body"
    }
    validate_server_interpretation_proposal(
        retained, "job", SOURCE, lowered.contract, lowered.attribution
    )
    assert (
        ground_contract(lowered.contract, SOURCE, attribution=lowered.attribution)
        == lowered.contract
    )
    # The old explicit source path still rejects this ordinary wording.
    with pytest.raises(ValueError):
        validate_verification_contract(lowered.contract, SOURCE, original.attribution)


def test_source_fact_units_convert_before_coordinate_arithmetic():
    draft = ordinary_draft()
    source = SOURCE.replace("60 mm", "6 cm")
    draft.contract.requirements[0].source_quote = DIMENSIONS.replace("60 mm", "6 cm")
    draft.contract.requirements[0].text = draft.contract.requirements[0].source_quote
    draft.attribution.facts[0].value = 6.0
    draft.attribution.facts[0].unit = "cm"
    draft.attribution.facts[0].source_quote = "6 cm"
    proposal = target()
    proposal.criterion.source_quote = draft.contract.requirements[0].source_quote
    lowered = attach_customer_interpretation(draft, source, [proposal])
    assert lowered.contract.requirements[-1].verification.size_mm == (60.0, 40.0, 5.0)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_field",
        "wrong_value",
        "unit",
        "subject",
        "unknown_fact",
        "tolerance",
        "count",
        "unknown_parent",
        "not_critical",
        "scalar_parent",
        "origin",
        "frame",
        "issue",
        "source",
    ],
)
def test_unbound_or_ambiguous_target_cannot_be_admitted(mutation):
    draft, proposal = ordinary_draft(), target()
    source = SOURCE
    if mutation == "missing_field":
        proposal.field_bindings.pop()
    if mutation == "wrong_value":
        proposal.criterion.size_mm = (61.0, 40.0, 5.0)
    if mutation == "unit":
        draft.attribution.facts[0].unit = "plate"
    if mutation == "subject":
        draft.attribution.facts[0].subject_id = None
    if mutation == "unknown_fact":
        proposal.field_bindings[0].terms[0].fact_id = "unknown"
    if mutation == "tolerance":
        proposal.criterion.holes[0].diameter_tolerance_mm = 0.1
    if mutation == "count":
        proposal.criterion.holes.pop()
    if mutation == "unknown_parent":
        proposal.source_requirement_ids = ["absent"]
    if mutation == "not_critical":
        draft.contract.requirements[0].critical = False
    if mutation == "scalar_parent":
        draft.contract.requirements[0].metric = "extent_x"
        draft.contract.requirements[0].value = 60.0
    if mutation == "origin":
        proposal.field_bindings[0].basis, proposal.field_bindings[0].terms = "design_origin", []
    if mutation == "frame":
        proposal.frame_policy = "source_defined_frame"
    if mutation == "issue":
        draft.attribution.issues = [
            IntentIssue.model_validate(
                {
                    "id": "conflict",
                    "question": "Which length?",
                    "reason": "Conflicting source",
                    "fact_ids": ["length"],
                }
            )
        ]
        draft = ContractDraft.model_validate(draft.model_dump())
    if mutation == "source":
        source = SOURCE.replace("60 mm", "61 mm")
    with pytest.raises(ValueError):
        attach_customer_interpretation(draft, source, [proposal])


@pytest.mark.parametrize(
    "mutation", ["source", "contract", "attribution", "drop_parent", "critical", "target"]
)
def test_frozen_identity_invalidates_changed_source_or_targets(mutation):
    lowered = lower_source_delivery_draft(wire(), SOURCE)
    source = SOURCE
    if mutation == "source":
        source += " Keep it red."
    if mutation == "contract":
        lowered.contract.coverage_notes.append("Changed")
    if mutation == "attribution":
        lowered.attribution.facts[0].property = "different interpretation"
    if mutation == "drop_parent":
        lowered.contract.requirements.pop(0)
    if mutation == "critical":
        lowered.contract.requirements[-1].critical = True
    if mutation == "target":
        lowered.contract.requirements[-1].verification.holes[0].diameter_mm = 5.0
    with pytest.raises(ValueError):
        validate_customer_interpretation(lowered.contract, source, lowered.attribution)
    with pytest.raises(ValueError):
        ground_contract(lowered.contract, source, attribution=lowered.attribution)


@pytest.mark.parametrize(
    "mutation", ["missing", "job", "status", "raw", "raw_hash", "lowered_hash"]
)
def test_author_computed_marker_needs_current_internal_job_proposal(mutation):
    lowered, document = retained_proposal()
    compilation = document["intent_compilation"]
    if mutation == "missing":
        document.clear()
    if mutation == "job":
        compilation["job_id"] = "other"
    if mutation == "status":
        compilation["status"] = "invalid_interpretation"
    if mutation == "raw":
        compilation["source_proposal"]["raw_response"]["nominal_targets"] = []
    if mutation == "raw_hash":
        compilation["source_proposal"]["raw_response_hash"] = "a" * 64
    if mutation == "lowered_hash":
        compilation["source_proposal"]["lowered_draft_hash"] = "a" * 64
    with pytest.raises(ValueError):
        validate_server_interpretation_proposal(
            document, "job", SOURCE, lowered.contract, lowered.attribution
        )


def test_nominal_pass_stays_separate_from_critical_physical_unknowns():
    lowered = lower_source_delivery_draft(wire(), SOURCE)
    results = [
        {"requirement_id": row.id, "status": "pass" if row.verification else "unverified"}
        for row in lowered.contract.requirements
    ]
    assert aggregate(lowered.contract, results) == ("unverified", False)
    display = customer_nominal_conformance(lowered.attribution, results)
    assert display["status"] == "pass"
    assert display["physical_readiness"] == "unverified"
    assert display["interpretation_status"] == "recorded_model_interpretation"
    assert customer_nominal_conformance(lowered.attribution, [])["status"] == "unverified"


def test_hidden_policy_is_not_an_authored_schema_field_and_legacy_serialization_is_unchanged():
    assert (
        "customer_nominal_interpretation" not in IntentAttribution.model_json_schema()["properties"]
    )
    assert "customer_nominal_interpretation" not in ordinary_draft().attribution.model_dump()
    raw = target().model_dump()
    raw["criterion"]["holes"][0]["diameter_mm"] = float("nan")
    with pytest.raises(ValidationError):
        CustomerNominalTarget.model_validate(raw)


def test_saved_proof_authority_revalidates_the_interpretation_record():
    lowered = lower_source_delivery_draft(wire(), SOURCE)
    record = {
        "status": "interpreted",
        "source_text": SOURCE,
        "source_hash": digest(SOURCE),
        "proposed_contract": lowered.contract.model_dump(),
        "contract_hash": lowered.contract.fingerprint(),
        "attribution": lowered.attribution.model_dump(),
    }
    record["draft_hash"] = digest(record)
    session = {"prompt": SOURCE, "intent_attribution": record}
    _intent_authority(session, lowered.contract)
    changed = deepcopy(session)
    changed["intent_attribution"]["attribution"]["facts"][0]["value"] = 61.0
    changed["intent_attribution"].pop("draft_hash")
    changed["intent_attribution"]["draft_hash"] = digest(changed["intent_attribution"])
    with pytest.raises(ValueError):
        _intent_authority(changed, lowered.contract)


def test_digital_source_guard_and_grounding_keep_independent_file_duty():
    draft = ordinary_draft()
    quote = "Deliver STEP for everything made."
    source = SOURCE + " " + quote
    draft.contract.requirements.append(
        Requirement.model_validate(
            {
                "id": "step_file",
                "text": quote,
                "source_quote": quote,
                "verification": {
                    "kind": "digital_delivery",
                    "scope": "digital_artifacts",
                    "deliverable": "project_step",
                    "coverage": "all_made",
                    "source_quote": quote,
                },
            }
        )
    )
    raw = source_delivery_wire_fixture(draft, digital_requirement_ids=["step_file"])
    raw["nominal_targets"] = [target().model_dump(mode="json")]
    lowered = lower_source_delivery_draft(SourceDeliveryContractDraft.model_validate(raw), source)
    assert (
        compile_source_contract_draft(lowered, source, source_only_requirement_ids=["step_file"])[
            "status"
        ]
        == "ready"
    )
    assert (
        ground_contract(lowered.contract, source, attribution=lowered.attribution)
        == lowered.contract
    )


def test_atomic_store_admission_replays_host_source_then_current_proof_revalidates(tmp_path):
    from forge_mind.workspace.intent_attribution_service import record_proposal, request_text
    from forge_mind.workspace.store import Store, WorkspaceError

    lowered, document = retained_proposal()
    store = Store(tmp_path)
    sid = store.create("owner", SOURCE, None, authoring_mode="native_operations")["id"]
    jid = store.enqueue(sid, "owner", {})
    assert store.claim()["id"] == jid
    basis = store.edit_basis(sid, "owner")
    with pytest.raises(WorkspaceError, match="retained source proposal"):
        record_proposal(
            store, sid, "owner", jid, lowered.attribution, lowered.contract, SOURCE, basis
        )
    compilation = document["intent_compilation"]
    compilation["job_id"] = jid
    store.update(sid, intent_compilation=compilation)
    record = record_proposal(
        store, sid, "owner", jid, lowered.attribution, lowered.contract, SOURCE, basis
    )
    assert record["status"] == "interpreted"
    session = store.session_record(sid, "owner")
    assert request_text(session) == SOURCE
    _intent_authority(session, lowered.contract)
    assert (
        record["attribution"]["customer_nominal_interpretation"]["contract_sha256"]
        == lowered.contract.fingerprint()
    )


@pytest.mark.parametrize("linked", [False, True])
def test_open_source_question_is_preserved_and_only_blocks_dependent_nominal_target(
    tmp_path, linked
):
    from forge_mind.workspace.intent_attribution_service import record_proposal
    from forge_mind.workspace.store import Store, WorkspaceError

    draft = ordinary_draft()
    issue = IntentIssue(
        id="load_direction",
        question="Which load direction is required?",
        reason="The source does not specify the load direction.",
        requirement_ids=["body_duty" if linked else "load_duty"],
    )
    draft.attribution.issues.append(issue)
    raw = source_delivery_wire_fixture(draft)
    raw["nominal_targets"] = [target().model_dump(mode="json")]
    proposal = SourceDeliveryContractDraft.model_validate(raw)
    if linked:
        with pytest.raises(ValueError, match="unresolved source issue"):
            lower_source_delivery_draft(proposal, SOURCE)
        return
    lowered = lower_source_delivery_draft(proposal, SOURCE)
    compilation = compile_source_contract_draft(lowered, SOURCE, source_only_requirement_ids=())
    assert compilation["status"] == "needs_customer_input"
    assert compilation["blocking_issue_ids"] == [issue.id]
    assert lowered.contract.requirements[:-1] == draft.contract.requirements
    assert lowered.attribution.issues == [issue]
    store = Store(tmp_path)
    sid = store.create("owner", SOURCE, None, authoring_mode="native_operations")["id"]
    jid = store.enqueue(sid, "owner", {})
    assert store.claim()["id"] == jid
    store.update(
        sid,
        intent_compilation={
            "job_id": jid,
            "source_hash": digest(SOURCE),
            "status": compilation["status"],
            "source_proposal": source_delivery_proposal_record(proposal, SOURCE, lowered),
        },
    )
    saved = record_proposal(
        store,
        sid,
        "owner",
        jid,
        lowered.attribution,
        lowered.contract,
        SOURCE,
        store.edit_basis(sid, "owner"),
    )
    assert saved["status"] == "needs_input"
    assert saved["attribution"]["issues"] == [issue.model_dump()]
    store.finish(jid, sid, "needs_input")
    with pytest.raises(WorkspaceError, match="clarification"):
        store.enqueue(sid, "owner", {"instruction": "Proceed without answering"})


def test_replay_status_must_match_actual_retained_source_questions():
    lowered, document = retained_proposal()
    document["intent_compilation"]["status"] = "needs_customer_input"
    with pytest.raises(ValueError, match="server-lowered"):
        validate_server_interpretation_proposal(
            document, "job", SOURCE, lowered.contract, lowered.attribution
        )
