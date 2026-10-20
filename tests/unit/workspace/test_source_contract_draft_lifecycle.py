"""Real intake hook with scripted initial wire and canonical bounded correction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.unit.workspace.compact_source_fixtures import compact_wire_fixture
from tests.unit.workspace.source_delivery_fixtures import (
    legacy_compact_profile,
    source_delivery_wire_fixture,
)
from tests.unit.workspace.test_source_contract_draft import deliveries

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.compact_source_contract_draft import CompactSourceContractDraft
from forge_mind.workspace.document import digest
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.measure import ground_contract
from forge_mind.workspace.models import Requirement
from forge_mind.workspace.source_delivery_contract_draft import SourceDeliveryContractDraft
from forge_mind.workspace.store import Store


def source_wire(draft):
    return source_delivery_wire_fixture(
        draft,
        digital_requirement_ids=["native", "step", "stl", "printing"],
    )


def run_source_intake(tmp_path, monkeypatch, source, responses, *, legacy_v1=False):
    if legacy_v1:
        with legacy_compact_profile(monkeypatch):
            return _run_source_intake(tmp_path, monkeypatch, source, responses)
    return _run_source_intake(tmp_path, monkeypatch, source, responses)


def _run_source_intake(tmp_path, monkeypatch, source, responses):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("owner", source, None, authoring_mode="native_operations")["id"]
    service.store.enqueue(sid, "owner", {})
    job = service.store.claim()
    assert job is not None
    calls = []

    class Scripted:
        model = "source-only-intake"
        timeout_s = None

        def generate_structured(self, **kwargs):
            calls.append(kwargs)
            index = len(calls) - 1
            assert index < len(responses), "No extra extraction or correction is authorized"
            response = responses[index]
            self.last_call_diagnostics = {"call_id": f"source-{index}", "outcome": "success"}
            return kwargs["output_model"].model_validate(response), LLMUsage(self.model, 13, 8)

        def generate_structured_with_image(self, **kwargs):
            pytest.fail("Intake must not request image inference")

    service.client_factory = lambda **kwargs: Scripted()

    def forbid(*args, **kwargs):
        pytest.fail("Intake cannot start geometry or native workers")

    monkeypatch.setattr(service, "build", forbid)
    monkeypatch.setattr(service.executor, "build", forbid)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbid)
    state = service._run_agent_job({**job, "max_actions": 0, "max_seconds": 100})
    return SimpleNamespace(state=state, saved=service.store.get(sid, "owner"), calls=calls)


def test_actual_source_output_model_and_canonical_initial_history_without_repair(
    tmp_path, monkeypatch
):
    draft, source = deliveries()
    raw = source_wire(draft)
    case = run_source_intake(tmp_path, monkeypatch, source, [raw])
    assert len(case.calls) == 1 and case.calls[0]["output_model"] is SourceDeliveryContractDraft
    record = case.saved["intent_compilation"]
    assert record["status"] == "ready"
    assert len(record["attempts"]) == 1
    attempt = record["attempts"][0]
    assert attempt["raw_draft"] == draft.model_dump(mode="json")
    wire = SourceDeliveryContractDraft.model_validate(raw).model_dump(mode="json")
    assert (
        attempt["source_proposal"]
        == record["source_proposal"]
        == {
            "protocol": "source_delivery_contract_draft_v1",
            "source_hash": digest(source),
            "raw_response": wire,
            "raw_response_hash": digest(wire),
            "source_only_requirement_ids": ["native", "step", "stl", "printing"],
            "lowered_draft_hash": digest(draft.model_dump(mode="json")),
        }
    )
    assert "repair_request" not in record
    assert case.saved["intent_attribution"]["status"] == "interpreted"
    assert case.saved["intent_attribution"]["proposed_contract"] == draft.contract.model_dump(
        mode="json"
    )
    assert len(case.saved["usage"]) == 1
    assert not case.saved["revisions"] and case.saved["contract"] is None
    assert not case.saved.get("accepted_revision")


def test_malformed_no_text_scalar_plus_typed_wire_retains_raw_before_lowering(
    tmp_path, monkeypatch
):
    draft, source = deliveries()
    # V1 can express this malformed scalar+digital combination; the new digital
    # schema forbids scalar fields. Keep the old raw-retention guard unchanged.
    raw = compact_wire_fixture(
        draft, source_only_requirement_ids=["native", "step", "stl", "printing"]
    )
    row = draft.contract.requirements[0].model_dump(mode="json")
    raw["requirements"][0]["scalar"] = {
        field: row[field]
        for field in (
            "metric",
            "comparator",
            "value",
            "tolerance",
            "part_id",
            "axis",
            "center",
            "position_tolerance",
        )
    }
    raw["requirements"][0]["scalar"].update(metric="extent_x", value=80.0)
    case = run_source_intake(tmp_path, monkeypatch, source, [raw], legacy_v1=True)
    assert len(case.calls) == 1 and case.calls[0]["output_model"] is CompactSourceContractDraft
    assert case.state == "error"
    record = case.saved["intent_compilation"]
    assert record["status"] == "lowering_rejected"
    assert record["source_proposal"]["raw_response"] == CompactSourceContractDraft.model_validate(
        raw
    ).model_dump(mode="json")
    assert "lowered_draft_hash" not in record["source_proposal"]
    assert record["attempts"] == []
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_scope_refusal_is_recorded_then_full_canonical_correction_retains_all_fields(
    tmp_path, monkeypatch
):
    initial, source = deliveries()
    source = source[:-1] + " for my Ender 3. Its size is 80 mm in local X."
    initial.contract.requirements.append(
        Requirement(
            id="width",
            text="Keep the stated width",
            source_quote="80 mm in local X",
            metric="extent_x",
            value=80,
        )
    )
    corrected = initial.model_copy(deep=True)
    corrected.contract.requirements.append(
        Requirement(id="machine", text="Retain the Ender 3 duty", source_quote="for my Ender 3")
    )
    case = run_source_intake(
        tmp_path, monkeypatch, source, [source_wire(initial), corrected.model_dump(mode="json")]
    )
    assert [call["output_model"] for call in case.calls] == [
        SourceDeliveryContractDraft,
        ContractDraft,
    ]
    record = case.saved["intent_compilation"]
    assert record["status"] == "ready" and len(record["attempts"]) == 2
    first, second = record["attempts"]
    assert first["raw_draft"] == initial.model_dump(mode="json")
    assert first["source_proposal"] == record["source_proposal"]
    assert any(row["code"] == "incomplete_digital_source_scope" for row in first["diagnostics"])
    assert "source_proposal" not in second and "verification_repair" not in second
    assert second["raw_draft"] == corrected.model_dump(mode="json")
    assert (
        second["raw_draft"]["contract"]["requirements"][:-1]
        == first["raw_draft"]["contract"]["requirements"]
    )
    grounded = ground_contract(corrected.contract, source, attribution=corrected.attribution)
    # Existing numeric grounding tightens only the association setting, not the target.
    assert (
        next(row for row in grounded.requirements if row.id == "width").position_tolerance == 0.001
    )
    assert case.saved["intent_attribution"]["proposed_contract"] == grounded.model_dump(mode="json")
    assert (
        corrected.contract.requirements[-1].critical
        and corrected.contract.requirements[-1].verification is None
    )
    assert len(case.saved["usage"]) == 2
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_full_correction_cannot_change_a_retained_scalar_or_remove_a_source_peer(
    tmp_path, monkeypatch
):
    initial, source = deliveries()
    source = source[:-1] + " for my Ender 3. Its size is 80 mm in local X."
    initial.contract.requirements.append(
        Requirement(
            id="width",
            text="Keep the stated width",
            source_quote="80 mm in local X",
            metric="extent_x",
            value=80,
        )
    )
    corrected = initial.model_copy(deep=True)
    corrected.contract.requirements[-1].value = 81
    corrected.contract.requirements = [
        row for row in corrected.contract.requirements if row.id != "stl"
    ]
    case = run_source_intake(
        tmp_path, monkeypatch, source, [source_wire(initial), corrected.model_dump(mode="json")]
    )
    assert [call["output_model"] for call in case.calls] == [
        SourceDeliveryContractDraft,
        ContractDraft,
    ]
    assert case.state == "needs_review"
    diagnostics = case.saved["intent_compilation"]["attempts"][1]["diagnostics"]
    assert any(row["code"] == "dropped_requirement" for row in diagnostics)
    assert any(row["code"] == "changed_source_duty" for row in diagnostics)
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_full_canonical_correction_cannot_escape_original_source_only_scope(tmp_path, monkeypatch):
    initial, source = deliveries()
    source = source[:-1] + " for my Ender 3."
    # Canonical FULL response has text on every digital row. That does not switch
    # the original no-text branch or release its saved source-only IDs.
    corrected = initial.model_copy(deep=True)
    case = run_source_intake(
        tmp_path, monkeypatch, source, [source_wire(initial), corrected.model_dump(mode="json")]
    )
    assert [call["output_model"] for call in case.calls] == [
        SourceDeliveryContractDraft,
        ContractDraft,
    ]
    assert case.state == "needs_review" and len(case.calls) == 2
    record = case.saved["intent_compilation"]
    assert record["source_proposal"]["source_only_requirement_ids"] == [
        "native",
        "step",
        "stl",
        "printing",
    ]
    assert all(attempt["status"] == "invalid_interpretation" for attempt in record["attempts"])
    assert any(
        row["code"] == "incomplete_digital_source_scope"
        for row in record["attempts"][1]["diagnostics"]
    )
    assert record["attempts"][1]["raw_draft"] == corrected.model_dump(mode="json")
    assert case.saved["contract"] is None and not case.saved["revisions"]
