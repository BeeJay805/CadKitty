"""Target history and carried author provenance use saved bytes, never native CAD."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import shutil
from copy import deepcopy
from zipfile import ZipFile

import pytest

from forge_mind.workspace import delivery_guidance as guidance
from forge_mind.workspace import project_delivery
from forge_mind.workspace.document import digest
from forge_mind.workspace.guidance_history import verify_guidance_history_context
from forge_mind.workspace.handoff import (
    _verify_guidance,
    handoff_context,
    verify_manufacturing_handoff,
)
from forge_mind.workspace.input_application import FIELD, baseline
from forge_mind.workspace.instruction_plan import resolve_instruction_plan
from forge_mind.workspace.native_review_context import compact_review_context
from forge_mind.workspace.project_delivery import build_manufacturing_handoff
from forge_mind.workspace.store import WorkspaceError
from tests.unit.workspace.test_delivery_guidance_sdk import connection, delivered_zip, snapshot
from tests.unit.workspace.test_flat_project_delivery_transport import flat_saved_project
from tests.unit.workspace.test_guidance_transport import instructions
from tests.unit.workspace.test_project_delivery import RID

NEXT = "d" * 32


def request(context, *, text="Literal author note. 中文\nPhysical fit remains unverified."):
    value = instructions(context)
    value["guidance"] = dict.fromkeys(value["guidance"], text)
    value["instruction_plan"] = {
        "version": 1,
        "fact_refs": context["instruction_catalog"]["required_refs"],
        "steps": [{"action": "review_physical_limits", "target": "physical:readiness"}],
    }
    return guidance.GuidanceUpdate.model_validate(value)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(guidance, "engine_fingerprint", lambda: "fixture-measurement-engine")
    monkeypatch.setattr(
        project_delivery, "engine_fingerprint", lambda: "fixture-measurement-engine"
    )
    context = guidance.guidance_context(service, sid, "owner", RID)
    current = guidance.update_guidance(service, sid, "owner", request(context))
    assert current["instruction_binding"]["status"] == "bound"
    return service, sid, current


def successor(service, sid, *, numeric=False):
    """Seed a distinct saved revision identity; fixture bytes are not CAD proof."""
    old = service.store.revision(sid, RID)
    revision = deepcopy(old)
    revision.update(id=NEXT, based_on=RID)
    revision["measurement"]["revision_id"] = NEXT
    source = service.store.directory(sid, NEXT)
    shutil.copytree(service.store.directory(sid, RID), source)
    if numeric:
        revision["document"]["variables"]["width"] = 44
        raw = json.dumps(revision["document"], ensure_ascii=False).encode()
        (source / "document.json").write_bytes(raw)
        revision["measurement"]["artifact_hashes"]["document.json"] = hashlib.sha256(
            raw
        ).hexdigest()
    service.store.add_revision(sid, revision)
    service.store.update(sid, selected_revision=NEXT)
    return guidance.guidance_context(service, sid, "owner", NEXT)


def verify(service, sid, context):
    revision = service.store.revision(sid, context["basis"]["revision_id"])
    captured = handoff_context(
        service.store.get(sid, "owner"), revision, "fixture-measurement-engine"
    )
    captured.update(
        guidance_basis=context["basis"],
        guidance_basis_hash=context["basis_hash"],
        guidance_context_hash=digest(context),
    )
    return _verify_guidance(context, captured, {}, revision["document"])


def test_current_record_references_primary_literal_history_without_duplicating_it(saved):
    service, sid, current = saved
    provenance = current["record_provenance"]
    assert provenance == {
        "source": "target",
        "revision_id": RID,
        "record_hash": digest(current["record"]),
        "history": {"ref": "history"},
        "history_status": current["history_status"],
    }
    assert verify_guidance_history_context(current) == current["history_status"]
    assert verify(service, sid, current) == "current"


@pytest.mark.parametrize("numeric", [False, True])
def test_successor_carries_exact_old_chain_without_rebinding_or_writes(saved, numeric):
    service, sid, current = saved
    carried = successor(service, sid, numeric=numeric)
    before = snapshot(service, sid)
    repeated = guidance.guidance_context(service, sid, "owner", NEXT)
    assert repeated == carried
    assert carried["status"] == "stale"
    assert carried["record"] == current["record"]
    assert carried["history"] == []
    assert carried["history_status"] == {"status": "unavailable", "head_hash": None, "errors": []}
    provenance = carried["record_provenance"]
    assert provenance["source"] == "selected_fallback"
    assert provenance["revision_id"] == RID
    assert provenance["record_hash"] == digest(current["record"])
    assert provenance["history"] == current["history"]
    assert provenance["history_status"] == current["history_status"]
    assert carried["input_authority"]["identity"]["revision_id"] == NEXT
    assert carried["instruction_binding"]["status"] == "stale"
    assert carried["instruction_binding"]["steps"] == []
    assert verify(service, sid, carried) == "stale"
    assert snapshot(service, sid) == before
    assert service.store.session_record(sid, "owner").get("accepted_revision") is None


def test_explicit_revision_queries_and_new_target_update_keep_both_chains_exact(saved):
    service, sid, current = saved
    carried = successor(service, sid)
    historical = guidance.guidance_context(service, sid, "owner", RID)
    assert historical["target_kind"] == "historical"
    assert historical["history"] == current["history"]
    assert historical["record_provenance"]["history"] == {"ref": "history"}
    assert historical["record_provenance"]["revision_id"] == RID
    updated = guidance.update_guidance(
        service, sid, "owner", request(carried, text="New literal guide.")
    )
    assert updated["status"] == "current"
    assert updated["record_provenance"]["revision_id"] == NEXT
    assert updated["record_provenance"]["history"] == {"ref": "history"}
    assert len(updated["history"]) == 1 and updated["history"][0]["revision_id"] == NEXT
    stored = service.store.session_record(sid, "owner")["delivery_guidance_history"]
    assert stored[RID] == current["history"]
    assert stored[NEXT] == updated["history"]
    assert verify(service, sid, updated) == "current"


@pytest.mark.parametrize("fault", ["missing", "hash", "head", "revision", "predecessor", "record"])
def test_current_and_carried_tracked_history_corruption_still_refuses(saved, fault):
    service, sid, current = saved
    successor(service, sid)
    history = deepcopy(service.store.session_record(sid, "owner")["delivery_guidance_history"])
    if fault == "missing":
        history.pop(RID)
    else:
        entry = history[RID][-1]
        if fault == "hash":
            entry["record_hash"] = "0" * 64
        elif fault == "head":
            entry["record"]["guidance"]["print"] = "Substituted literal."
            entry["record_hash"] = digest(entry["record"])
        elif fault == "revision":
            entry["revision_id"] = NEXT
        elif fault == "predecessor":
            entry["predecessor_hash"] = "0" * 64
        else:
            entry["record"]["guidance"]["print"] = "Unhashed substitution."
    service.store.update(sid, delivery_guidance_history=history)
    before = snapshot(service, sid)
    for rid in (RID, NEXT):
        context = guidance.guidance_context(service, sid, "owner", rid)
        with pytest.raises(ValueError, match="guidance"):
            verify(service, sid, context)
    assert snapshot(service, sid) == before
    assert current["record"]["guidance"]["print"].startswith("Literal author note.")


@pytest.mark.parametrize(
    "fault", ["origin", "record_hash", "head_hash", "source", "reference", "status", "session"]
)
def test_offline_recomputed_provenance_refuses_outer_hash_resealed_tampering(saved, fault):
    service, sid, _ = saved
    carried = successor(service, sid)
    altered = deepcopy(carried)
    provenance = altered["record_provenance"]
    if fault == "origin":
        provenance["revision_id"] = NEXT
    elif fault == "record_hash":
        provenance["record_hash"] = "0" * 64
    elif fault == "head_hash":
        provenance["history_status"]["head_hash"] = "0" * 64
    elif fault == "source":
        provenance["source"] = "target"
    elif fault == "reference":
        provenance["history"] = {"ref": "history"}
    elif fault == "status":
        altered["status"] = "current"
    else:
        record = altered["record"]
        record["basis"]["session_id"] = "e" * 32
        record["basis_hash"] = digest(record["basis"])
        provenance["history"][-1]["record"] = deepcopy(record)
        provenance["history"][-1]["record_hash"] = digest(record)
        provenance["record_hash"] = digest(record)
        provenance["history_status"]["head_hash"] = digest(provenance["history"][-1])
    with pytest.raises(ValueError, match="guidance"):
        verify(service, sid, altered)


def test_orphan_target_chain_cannot_stand_in_for_a_missing_target_record(saved):
    service, sid, current = saved
    successor(service, sid)
    history = deepcopy(service.store.session_record(sid, "owner")["delivery_guidance_history"])
    orphan = deepcopy(current["history"])
    orphan[0]["revision_id"] = NEXT
    history[NEXT] = orphan
    service.store.update(sid, delivery_guidance_history=history)
    context = guidance.guidance_context(service, sid, "owner", NEXT)
    assert context["history_status"]["status"] == "unverified"
    assert context["record_provenance"]["history_status"]["status"] == "verified"
    with pytest.raises(ValueError, match="guidance"):
        verify(service, sid, context)


def test_wrong_map_key_is_not_reclassified_as_valid_carried_provenance(saved):
    service, sid, current = saved
    successor(service, sid)
    records = deepcopy(service.store.session_record(sid, "owner")["delivery_guidance_records"])
    records[NEXT] = current["record"]
    service.store.update(sid, delivery_guidance_records=records)
    context = guidance.guidance_context(service, sid, "owner", NEXT)
    assert context["record_provenance"]["source"] == "target"
    assert context["record_provenance"]["history_status"]["status"] == "unverified"
    with pytest.raises(ValueError, match="guidance"):
        verify(service, sid, context)


def test_clear_origin_provenance_cannot_override_current_pending_owner_input(saved):
    service, sid, _ = saved
    successor(service, sid)
    session = service.store.session_record(sid, "owner")
    application = baseline(session)
    application["pending"] = {"input_id": "a" * 32, "source_hash": "b" * 64}
    service.store.update(sid, **{FIELD: application})
    before = snapshot(service, sid)
    context = guidance.guidance_context(service, sid, "owner", NEXT)
    assert context["record_provenance"]["history_status"]["status"] == "verified"
    assert context["input_authority"]["status"] == "blocked"
    assert context["status"] == "stale"
    assert context["instruction_binding"]["status"] == "blocked_input"
    assert verify(service, sid, context) == "stale"
    with pytest.raises(WorkspaceError, match="Owner input"):
        guidance.update_guidance(service, sid, "owner", request(context))
    assert snapshot(service, sid) == before


def test_old_current_archive_without_origin_facet_keeps_exact_target_history_compatibility(saved):
    service, sid, current = saved
    old = deepcopy(current)
    old.pop("record_provenance")
    assert verify(service, sid, old) == "current"
    old["history"] = []
    old["history_status"] = {"status": "legacy_unverified", "head_hash": None, "errors": []}
    with pytest.raises(ValueError, match="authored history"):
        verify_guidance_history_context(old)


@pytest.mark.parametrize("unknown_origin", [False, True])
def test_genuine_legacy_absence_preserves_literal_notes_without_inventing_origin(
    saved, unknown_origin
):
    service, sid, current = saved
    record = deepcopy(current["record"])
    for key in ("instruction_plan", "instruction_plan_hash", "input_authority_hash"):
        record.pop(key)
    if unknown_origin:
        record.pop("basis")
        record.pop("basis_hash")
    service.store.update(
        sid,
        delivery_guidance_record=record,
        delivery_guidance_records={} if unknown_origin else {RID: record},
        delivery_guidance_history={},
    )
    context = successor(service, sid)
    before = snapshot(service, sid)
    provenance = context["record_provenance"]
    assert provenance["revision_id"] == (None if unknown_origin else RID)
    assert provenance["history"] == []
    assert provenance["history_status"]["status"] == "legacy_unverified"
    assert context["record"] == record
    assert context["status"] == "stale"
    assert context["instruction_binding"]["status"] == "legacy_unverified"
    assert verify(service, sid, context) == "stale"
    assert snapshot(service, sid) == before


@pytest.mark.parametrize("include_stl", [False, True])
def test_partial_and_complete_packages_keep_literal_carried_provenance(
    saved, tmp_path, include_stl
):
    service, sid, current = saved
    context = successor(service, sid)
    before = snapshot(service, sid)
    target = tmp_path / "carried.zip"
    build_manufacturing_handoff(
        service, sid, "owner", NEXT, target, include_stl=include_stl, include_assembly_step=True
    )
    verified = verify_manufacturing_handoff(target)
    assert verified["integrity"] == "verified"
    assert verified["guidance_status"] == "stale"
    with ZipFile(target) as archive:
        saved_context = json.loads(archive.read("customer-guidance.json"))
        assert saved_context == context
        assert saved_context["record"]["guidance"] == current["record"]["guidance"]
        assert saved_context["history"] == []
        assert saved_context["record_provenance"]["history"] == current["history"]
        assert ("make/part.stl" in archive.namelist()) is include_stl
        assert saved_context["instruction_binding"]["steps"] == []
    assert snapshot(service, sid) == before


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_actual_sdk_read_and_export_preserve_target_and_origin_coordinates(
    saved, tmp_path, surface
):
    service, sid, current = saved
    context = successor(service, sid)
    before = snapshot(service, sid)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            response = await sdk.call_tool("get_guidance", {"project_id": sid, "revision_id": NEXT})
            assert not response.is_error
            assert response.structured_content == context
            response = await sdk.call_tool(
                "export_project",
                {"project_id": sid, "revision_id": NEXT, "include_stl": False, "preview": "none"},
            )
            receipt, content = await delivered_zip(sdk, response, surface)
            assert receipt["guidance_status"] == "stale"
            with ZipFile(io.BytesIO(content)) as archive:
                sidecar = json.loads(archive.read("customer-guidance.json"))
                assert sidecar["history"] == []
                assert sidecar["record_provenance"]["history"] == current["history"]
                assert sidecar["record"] == current["record"]

    asyncio.run(scenario())
    assert snapshot(service, sid) == before


@pytest.mark.parametrize("carried", [False, True])
def test_automatic_context_compacts_both_histories_but_explicit_reads_stay_literal(saved, carried):
    service, sid, current = saved
    context = successor(service, sid) if carried else current
    before = deepcopy(context)
    automatic = {"guidance": deepcopy(context)}
    explicit = {"guidance": deepcopy(context), "explicit_guidance_read": True}
    result = compact_review_context(
        {
            "delivery_guidance": context,
            "history": [automatic, explicit],
            "persistent_tool_observations": [explicit],
        }
    )
    for value in (result["delivery_guidance"], result["history"][0]["guidance"]):
        assert value["history"]["is_summary"] is True
        assert value["history"]["entry_count"] == len(context["history"])
        assert value["history_status"] == context["history_status"]
        provenance = value["record_provenance"]
        assert provenance["source"] == context["record_provenance"]["source"]
        assert provenance["record_hash"] == context["record_provenance"]["record_hash"]
        assert provenance["history_status"] == context["record_provenance"]["history_status"]
        if carried:
            assert provenance["history"]["entry_count"] == len(current["history"])
            assert provenance["history"]["sha256"] == digest(current["history"])
            assert all("record" not in entry for entry in provenance["history"]["entries"])
        else:
            assert provenance["history"] == {"ref": "history"}
        assert value["facts"] == context["facts"]
        assert value["record"]["guidance"] == context["record"]["guidance"]
    assert result["history"][1] == explicit
    assert result["persistent_tool_observations"] == [explicit]
    assert context == before


@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("substitution", ["bool_int", "int_float"])
def test_resealed_history_head_cannot_alias_a_different_json_scalar(saved, carried, substitution):
    service, sid, current = saved
    context = deepcopy(successor(service, sid) if carried else current)
    chain = context["record_provenance"]["history"] if carried else context["history"]
    if substitution == "int_float":
        context["record"]["literal_number"] = 1
        chain[-1]["record"] = deepcopy(context["record"])
        chain[-1]["record_hash"] = digest(chain[-1]["record"])
        context["record_provenance"]["record_hash"] = digest(context["record"])
        context["record_provenance"]["history_status"]["head_hash"] = digest(chain[-1])
        if not carried:
            context["history_status"]["head_hash"] = digest(chain[-1])
        assert verify(service, sid, context) == ("stale" if carried else "current")
        chain[-1]["record"]["literal_number"] = 1.0
    else:
        assert context["record"]["reviewed_current_facts"] is True
        chain[-1]["record"]["reviewed_current_facts"] = 1
    chain[-1]["record_hash"] = digest(chain[-1]["record"])
    context["record_provenance"]["history_status"]["head_hash"] = digest(chain[-1])
    if not carried:
        context["history_status"]["head_hash"] = digest(chain[-1])
    with pytest.raises(ValueError, match="guidance"):
        verify(service, sid, context)


@pytest.mark.parametrize("field", ["revision_id", "session_id"])
def test_facet_absence_cannot_relabel_tracked_origin_with_a_resealed_target_chain(saved, field):
    service, sid, current = saved
    context = deepcopy(current)
    context.pop("record_provenance")
    # deepcopy preserves aliases inside the graph; detach only the malicious record.
    context["record"] = deepcopy(context["record"])
    context["record"]["basis"][field] = NEXT if field == "revision_id" else "e" * 32
    context["record"]["basis_hash"] = digest(context["record"]["basis"])
    entry = context["history"][-1]
    entry["record"] = deepcopy(context["record"])
    entry["record_hash"] = digest(entry["record"])
    context["history_status"]["head_hash"] = digest(entry)
    context["status"] = "stale"
    context["instruction_binding"] = resolve_instruction_plan(context)
    assert context["basis"] == current["basis"]
    assert context["basis"]["revision_id"] == RID
    assert context["basis"]["session_id"] == sid
    assert context["basis_hash"] == digest(context["basis"])
    assert context["history"][-1]["revision_id"] == RID
    assert context["history"][-1]["record"] == context["record"]
    with pytest.raises(ValueError, match="requires provenance"):
        verify_guidance_history_context(context)
    with pytest.raises(ValueError, match="requires provenance"):
        verify(service, sid, context)


def test_stale_alias_and_valid_truncated_chain_cannot_hide_newer_source_map_head(saved):
    service, sid, first = saved
    second = guidance.update_guidance(
        service, sid, "owner", request(first, text="Second literal source-map head.")
    )
    assert len(second["history"]) == 2
    successor(service, sid)
    service.store.update(
        sid,
        delivery_guidance_record=deepcopy(first["record"]),
        delivery_guidance_history={RID: deepcopy(first["history"])},
    )
    before = snapshot(service, sid)
    context = guidance.guidance_context(service, sid, "owner", NEXT)
    assert context["record"] == first["record"]
    assert context["history"] == []
    assert context["record_provenance"]["history"] == first["history"]
    status = context["record_provenance"]["history_status"]
    assert status["status"] == "unverified"
    assert any("source-map head" in reason for reason in status["errors"])
    assert context["instruction_binding"]["status"] == "invalid"
    with pytest.raises(ValueError, match="guidance"):
        verify(service, sid, context)
    source = guidance.guidance_context(service, sid, "owner", RID)
    assert source["record"] == second["record"]
    assert source["history"] == first["history"]
    assert source["history_status"]["status"] == "unverified"
    assert snapshot(service, sid) == before


def test_invalid_json_in_supplied_head_record_is_reported_unverified(saved):
    _, _, current = saved
    invalid = deepcopy(current["record"])
    invalid["literal_number"] = float("nan")
    status = guidance.guidance_history_status(current["history"], RID, invalid)
    assert status["status"] == "unverified"
    assert status["head_hash"] is None
