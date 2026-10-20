"""Guidance freshness checks use saved fixture bytes, never CAD or model calls."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from forge_mind.workspace import completion
from forge_mind.workspace import delivery_guidance as guidance
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.product_sources import _digest, read_sources
from forge_mind.workspace.store import Store, WorkspaceError

RID = "a" * 32
OTHER = "b" * 32
TEXT = {
    "print": "Use prototype settings; printability is untested.",
    "installation": "Check physical fit before installation.",
    "use_and_removal": "Check the actual removal path.",
    "unresolved": "Strength, fit and printing remain unverified.",
}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    store = Store(tmp_path / "store")
    contract = Contract().model_dump()
    sid = store.create("owner", "Guidance fixture", contract)["id"]
    document = DesignDocument.model_validate(
        {
            "name": "Saved fixture",
            "variables": {"width": 40, "multiplier": 2},
            "relations": [
                {"name": "span", "operation": "multiply", "a": "width", "b": "multiplier"}
            ],
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "kind": "box",
                    "params": {"x": 40, "y": 10, "z": 6},
                    "bindings": {"x": "width"},
                }
            ],
            "outputs": ["body"],
            "delivery_guidance": TEXT,
        }
    ).model_dump()
    directory = store.directory(sid, RID)
    directory.mkdir(parents=True)
    files = {
        "program.py": b"# Saved fixture; never execute.\n",
        "part.step": b"Saved STEP fixture, not parsed",
        "part.stl": b"Saved STL fixture, not parsed",
        "document.json": json.dumps(document).encode(),
    }
    hashes = {}
    for name, content in files.items():
        (directory / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    measurement = Measurement(
        revision_id=RID,
        geometry_valid=True,
        engine_fingerprint="fixture-engine",
        contract_hash=Contract().fingerprint(),
        artifact_hashes=hashes,
        facts={"extents_mm": [40, 10, 6], "solid_count": 1, "volume_mm3": 2400, "holes": []},
    ).model_dump()
    store.add_revision(
        sid, {"id": RID, "document": document, "contract": contract, "measurement": measurement}
    )
    store.update(
        sid, selected_revision=RID, accepted_revision=None, design_plan={"notes": "Existing plan"}
    )
    service = SimpleNamespace(store=store, _load_revision=store.revision)
    monkeypatch.setattr(guidance, "engine_fingerprint", lambda: "fixture-engine")
    monkeypatch.setattr(completion, "engine_fingerprint", lambda: "fixture-engine")
    return SimpleNamespace(service=service, sid=sid, directory=directory)


def context(fixture, rid=None):
    return guidance.guidance_context(fixture.service, fixture.sid, "owner", rid)


def update_request(value, text=None):
    return guidance.GuidanceUpdate(
        revision_id=value["basis"]["revision_id"],
        basis_hash=value["basis_hash"],
        reviewed_current_facts=True,
        guidance=text or TEXT,
    )


def save_revision(fixture, revision):
    with fixture.service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), fixture.sid, RID),
        )


def test_legacy_is_unbound_and_saved_measurements_differ_from_declarations(prepared):
    value = context(prepared)
    assert value["status"] == "unbound" and value["record"] is None
    assert value["legacy_guidance"] == TEXT
    assert value["legacy_guidance_status"] == "unbound"
    assert value["facts"]["measured"]["extents_mm"] == [40, 10, 6]
    assert value["facts"]["declared_variables"]["inputs"]["width"] == 40
    assert value["facts"]["declared_variables"]["evaluated"]["span"] == 80
    assert "not independently measured" in value["facts"]["declared_variables"]["scope"]
    assert value["basis_hash"] == digest(value["basis"])
    assert value["basis"]["facts_hash"] == digest(value["facts"])


def test_update_changes_only_metadata_and_never_claims_correctness(prepared):
    store, sid = prepared.service.store, prepared.sid
    before = store.session_record(sid, "owner")
    revisions = store.revisions(sid)
    files = {path.name: path.read_bytes() for path in prepared.directory.iterdir()}
    wrong = {**TEXT, "installation": "The finished part has no mounting holes."}
    saved = guidance.update_guidance(
        prepared.service, sid, "owner", update_request(context(prepared), wrong)
    )
    assert saved["status"] == context(prepared)["status"] == "current"
    assert saved["record"]["guidance"] == wrong
    assert saved["physical_readiness"] == "unverified"
    assert "does not establish textual correctness" in saved["scope"]
    after = store.session_record(sid, "owner")
    after.pop("delivery_guidance_record")
    after.pop("delivery_guidance_records")
    history = after.pop("delivery_guidance_history")
    assert history == {
        RID: [
            {
                "version": 1,
                "predecessor_hash": None,
                "revision_id": RID,
                "record": saved["record"],
                "record_hash": digest(saved["record"]),
            }
        ]
    }
    assert saved["history"] == history[RID]
    assert saved["history_status"] == {
        "status": "verified",
        "head_hash": digest(history[RID][0]),
        "errors": [],
    }
    assert after == before
    assert store.revisions(sid) == revisions
    assert {path.name: path.read_bytes() for path in prepared.directory.iterdir()} == files
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("field", list(TEXT))
@pytest.mark.parametrize("replacement", [None, "", "  "])
def test_all_four_fields_must_be_nonempty(field, replacement):
    with pytest.raises(ValidationError):
        guidance.GuidanceUpdate(
            revision_id=RID,
            basis_hash="a" * 64,
            reviewed_current_facts=True,
            guidance={**TEXT, field: replacement},
        )


@pytest.mark.parametrize("reviewed", [False, 1, "true", None])
def test_review_requires_explicit_boolean(reviewed):
    with pytest.raises(ValidationError):
        guidance.GuidanceUpdate(
            revision_id=RID, basis_hash="a" * 64, reviewed_current_facts=reviewed, guidance=TEXT
        )


def test_schema_requires_all_fields():
    schema = guidance.GuidanceUpdate.model_json_schema()
    assert set(schema["$defs"]["ReviewedDeliveryGuidance"]["required"]) == set(TEXT)


def test_guidance_exposes_package_paths_separate_from_saved_artifact_provenance(prepared):
    value = context(prepared)
    inventory = value["facts"]["package_files"]
    assert "assembly/part.step" in inventory["geometry"]
    assert "part.stl" in inventory["excluded_saved_artifacts"]
    assert "part.stl" in value["basis"]["artifact_hashes"]
    assert "part.stl" not in inventory["geometry"]
    assert value["basis"]["facts_hash"] == digest(value["facts"])


def test_guidance_keeps_explicit_excluded_file_warning_and_exact_package_path(prepared):
    text = {
        **TEXT,
        "print": "Do not print part.stl; it is not included in this project package.",
        "installation": "Use assembly/part.step for the saved overall CAD.",
    }
    saved = guidance.update_guidance(
        prepared.service, prepared.sid, "owner", update_request(context(prepared), text)
    )
    assert saved["status"] == "current"
    assert saved["record"]["guidance"] == text
    assert saved["physical_readiness"] == "unverified"


@pytest.mark.parametrize(
    "mutation", ["revision", "inputs", "plan", "requirements", "prose", "source", "facts", "engine"]
)
def test_changed_basis_refuses_stale_update(prepared, mutation, monkeypatch):
    store, sid = prepared.service.store, prepared.sid
    request = update_request(context(prepared))
    if mutation == "revision":
        store.update(sid, selected_revision=OTHER)
    elif mutation == "inputs":
        # Same version, different content must invalidate the captured basis too.
        with store.connect() as con:
            con.execute(
                "INSERT INTO engineering_inputs VALUES (?,?,?)",
                (sid, 0, json.dumps({"annotations": [{"fixture": "changed"}]})),
            )
    elif mutation == "plan":
        store.update(sid, design_plan={"notes": "Changed plan"})
    elif mutation == "requirements":
        store.update(sid, contract={"requirements": [], "explanation": "Changed requirements"})
    elif mutation == "source":
        read_sources(store, sid, "owner")
        event = {"kind": "artifact", "data": {"id": "c" * 32, "kind": "fixture"}}
        with store.connect() as con:
            con.execute(
                "INSERT INTO product_source_events VALUES (?,?,?,?)",
                (sid, 1, json.dumps(event), _digest(event)),
            )
    elif mutation == "engine":
        monkeypatch.setattr(guidance, "engine_fingerprint", lambda: "different-engine")
    else:
        revision = store.revision(sid, RID)
        if mutation == "prose":
            revision["document"]["delivery_guidance"]["print"] = "Changed legacy prose"
            content = json.dumps(revision["document"]).encode()
            (prepared.directory / "document.json").write_bytes(content)
            revision["measurement"]["artifact_hashes"]["document.json"] = hashlib.sha256(
                content
            ).hexdigest()
        else:
            revision["measurement"]["facts"]["extents_mm"][0] = 44
        save_revision(prepared, revision)
    with pytest.raises(WorkspaceError, match=r"changed|different engine"):
        guidance.update_guidance(prepared.service, sid, "owner", request)
    assert "delivery_guidance_record" not in store.session_record(sid, "owner")


def test_atomic_recheck_detects_change_after_preflight(prepared, monkeypatch):
    request = update_request(context(prepared))
    original = guidance._snapshot

    def changed(*args):
        result = original(*args)
        prepared.service.store.update(prepared.sid, design_plan={"notes": "Concurrent change"})
        return result

    monkeypatch.setattr(guidance, "_snapshot", changed)
    with pytest.raises(WorkspaceError, match="Guidance inputs changed"):
        guidance.update_guidance(prepared.service, prepared.sid, "owner", request)


def test_tampered_prose_stays_stale_and_cannot_replace_literal_history(prepared):
    store, sid = prepared.service.store, prepared.sid
    saved = guidance.update_guidance(
        prepared.service, sid, "owner", update_request(context(prepared))
    )
    original = copy.deepcopy(saved["record"])
    history = copy.deepcopy(store.session_record(sid, "owner")["delivery_guidance_history"])
    record = copy.deepcopy(original)
    record["guidance"]["print"] = "Tampered saved prose"
    store.update(sid, delivery_guidance_record=record, delivery_guidance_records={RID: record})
    before = store.session_record(sid, "owner")
    stale = context(prepared)
    assert stale["status"] == "stale"
    assert stale["history_status"]["status"] == "unverified"
    assert stale["history"] == history[RID]
    assert stale["history"][0]["record"] == original
    with pytest.raises(WorkspaceError, match="Guidance history integrity changed"):
        guidance.update_guidance(prepared.service, sid, "owner", update_request(stale))
    assert store.session_record(sid, "owner") == before
    assert before["delivery_guidance_record"] == record
    assert before["delivery_guidance_records"][RID] == record
    assert before["delivery_guidance_history"] == history
    assert context(prepared)["status"] == "stale"


def test_old_revision_context_is_stale_without_modifying_guidance_history(prepared):
    store, sid = prepared.service.store, prepared.sid
    saved = guidance.update_guidance(
        prepared.service, sid, "owner", update_request(context(prepared))
    )
    history = copy.deepcopy(store.session_record(sid, "owner")["delivery_guidance_history"])
    store.update(sid, selected_revision=OTHER)
    assert context(prepared, RID)["status"] == "stale"
    session = store.session_record(sid, "owner")
    assert session["delivery_guidance_records"][RID] == saved["record"]
    assert session["delivery_guidance_history"] == history


@pytest.mark.parametrize("mutation", ["artifacts", "invalid", "foreign"])
def test_invalid_or_unowned_revision_cannot_receive_guidance(prepared, mutation):
    request = update_request(context(prepared))
    if mutation == "artifacts":
        (prepared.directory / "part.step").write_bytes(b"Tampered bytes")
    elif mutation == "invalid":
        revision = prepared.service.store.revision(prepared.sid, RID)
        revision["measurement"]["geometry_valid"] = False
        save_revision(prepared, revision)
    owner = "foreign" if mutation == "foreign" else "owner"
    with pytest.raises(WorkspaceError):
        guidance.update_guidance(prepared.service, prepared.sid, owner, request)
    with pytest.raises(WorkspaceError):
        guidance.guidance_context(prepared.service, prepared.sid, owner)


def test_display_summary_has_limits_without_limiting_design(prepared):
    revision = copy.deepcopy(prepared.service.store.revision(prepared.sid, RID))
    revision["document"]["variables"].update({f"value_{i}": float(i) for i in range(80)})
    facts = guidance._facts(revision)
    declared = facts["declared_variables"]
    assert declared["inputs_total"] == 82 and declared["evaluated_total"] == 83
    assert len(declared["inputs"]) == len(declared["evaluated"]) == 64
    assert declared["inputs_truncated"] and declared["evaluated_truncated"]


def test_missing_hole_observation_is_not_reported_as_zero_holes(prepared):
    revision = copy.deepcopy(prepared.service.store.revision(prepared.sid, RID))
    revision["measurement"]["facts"].pop("holes")
    assert "hole_count" not in guidance._facts(revision)["measured"]


def test_invalid_measurement_summary_values_are_not_presented_as_dimensions(prepared):
    revision = copy.deepcopy(prepared.service.store.revision(prepared.sid, RID))
    revision["measurement"]["facts"].update(
        extents_mm=[40, float("nan"), 6],
        bbox_min=[False, 0, 0],
        bbox_max=[40, 10],
        volume_mm3=float("inf"),
        solid_count=True,
        step_valid="true",
    )
    assert guidance._facts(revision)["measured"] == {"hole_count": 0}


def pending_revision(fixture, rid=OTHER, *, selected=RID, native_binding=False):
    """Save an equivalent immutable fixture revision; never construct geometry."""
    store = fixture.service.store
    revision = copy.deepcopy(store.revision(fixture.sid, RID))
    revision.update(id=rid, parent_id=RID)
    revision["measurement"]["revision_id"] = rid
    directory = store.directory(fixture.sid, rid)
    directory.mkdir(parents=True)
    for path in fixture.directory.iterdir():
        (directory / path.name).write_bytes(path.read_bytes())
    proposal = {
        "revision_id": rid,
        "based_on": selected,
        "eligible": True,
        "input_version": 0,
        "contract": revision["contract"],
    }
    if native_binding:
        binding = {
            "plan": {"objective": "Pending candidate intent"},
            "plan_hash": "e" * 64,
            "contract_hash": Contract().fingerprint(),
            "plan_version": 2,
        }
        revision["operation_record"] = {"plan_snapshot": binding}
        proposal["operation_plan_snapshot"] = binding
    store.add_revision(fixture.sid, revision)
    store.update(
        fixture.sid,
        selected_revision=selected,
        accepted_revision=selected,
        proposal=proposal,
    )
    return revision, proposal


def test_first_pending_candidate_receives_guidance_without_acceptance(prepared):
    store, sid = prepared.service.store, prepared.sid
    revision, _ = pending_revision(prepared, selected=None)
    before = store.session_record(sid, "owner")
    with pytest.raises(WorkspaceError, match="saved structured revision"):
        context(prepared)
    pending = context(prepared, OTHER)
    assert pending["target_kind"] == "pending_candidate"
    assert pending["selected_revision"] is pending["accepted_revision"] is None
    assert pending["design_revision"] == OTHER
    saved = guidance.update_guidance(prepared.service, sid, "owner", update_request(pending))
    assert saved["status"] == context(prepared, OTHER)["status"] == "current"
    after = store.session_record(sid, "owner")
    assert "delivery_guidance_record" not in after
    assert after["delivery_guidance_records"][OTHER] == saved["record"]
    after.pop("delivery_guidance_records")
    history = after.pop("delivery_guidance_history")
    assert history == {
        OTHER: [
            {
                "version": 1,
                "predecessor_hash": None,
                "revision_id": OTHER,
                "record": saved["record"],
                "record_hash": digest(saved["record"]),
            }
        ]
    }
    assert saved["history"] == history[OTHER]
    assert saved["history_status"] == {
        "status": "verified",
        "head_hash": digest(history[OTHER][0]),
        "errors": [],
    }
    assert after == before
    assert store.revision(sid, OTHER) == revision
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_pending_guidance_preserves_selected_record_and_survives_rejection(prepared):
    store, sid = prepared.service.store, prepared.sid
    original = guidance.update_guidance(
        prepared.service, sid, "owner", update_request(context(prepared))
    )["record"]
    # A pre-existing single-record session migrates only during a successful write.
    store.update(sid, delivery_guidance_records={})
    baseline = store.revision(sid, RID)
    files = {path.name: path.read_bytes() for path in prepared.directory.iterdir()}
    pending_revision(prepared)
    pending = context(prepared, OTHER)
    text = {**TEXT, "installation": "Review the new candidate before fitting it."}
    saved = guidance.update_guidance(prepared.service, sid, "owner", update_request(pending, text))
    session = store.session_record(sid, "owner")
    assert session["selected_revision"] == session["accepted_revision"] == RID
    assert session["delivery_guidance_record"] == original
    assert session["delivery_guidance_records"][RID] == original
    assert session["delivery_guidance_records"][OTHER] == saved["record"]
    assert context(prepared)["status"] == "current"
    assert context(prepared)["record"] == original
    store.update(sid, proposal=None)
    reopened = SimpleNamespace(
        service=SimpleNamespace(store=Store(store.root), _load_revision=store.revision), sid=sid
    )
    assert context(reopened)["record"] == original
    assert context(reopened)["status"] == "current"
    historical = context(reopened, OTHER)
    assert historical["target_kind"] == "historical"
    assert historical["status"] == "stale"
    assert historical["record"]["guidance"] == text
    with pytest.raises(WorkspaceError, match="Current design revision changed"):
        guidance.update_guidance(prepared.service, sid, "owner", update_request(historical))
    assert store.revision(sid, RID) == baseline
    assert {path.name: path.read_bytes() for path in prepared.directory.iterdir()} == files


def test_superseded_pending_write_is_refused_and_prior_instructions_remain(prepared):
    sid = prepared.sid
    pending_revision(prepared)
    saved = guidance.update_guidance(
        prepared.service, sid, "owner", update_request(context(prepared, OTHER))
    )["record"]
    request = update_request(context(prepared, OTHER))
    third = "c" * 32
    pending_revision(prepared, third)
    with pytest.raises(WorkspaceError, match="Current design revision changed"):
        guidance.update_guidance(prepared.service, sid, "owner", request)
    assert context(prepared, OTHER)["record"] == saved
    assert context(prepared, OTHER)["target_kind"] == "historical"
    assert context(prepared, third)["status"] == "unbound"


def test_pending_write_rechecks_head_after_preflight(prepared, monkeypatch):
    store, sid = prepared.service.store, prepared.sid
    pending_revision(prepared)
    request = update_request(context(prepared, OTHER))
    original = guidance._snapshot

    def superseded(*args):
        result = original(*args)
        store.update(sid, proposal=None)
        return result

    monkeypatch.setattr(guidance, "_snapshot", superseded)
    with pytest.raises(WorkspaceError, match="Current design revision changed"):
        guidance.update_guidance(prepared.service, sid, "owner", request)
    assert "delivery_guidance_records" not in store.session_record(sid, "owner")


@pytest.mark.parametrize("defect", ["stale_basis", "foreign", "engine", "artifacts", "contract"])
def test_pending_candidate_refuses_unowned_or_stale_instructions(prepared, monkeypatch, defect):
    store, sid = prepared.service.store, prepared.sid
    _, proposal = pending_revision(prepared)
    request = update_request(context(prepared, OTHER))
    owner = "foreign" if defect == "foreign" else "owner"
    if defect == "stale_basis":
        request = request.model_copy(update={"basis_hash": "f" * 64})
    elif defect == "engine":
        monkeypatch.setattr(guidance, "engine_fingerprint", lambda: "changed-engine")
        request = update_request(context(prepared, OTHER))
    elif defect == "artifacts":
        (store.directory(sid, OTHER) / "part.step").write_bytes(b"Changed")
    elif defect == "contract":
        proposal["contract"] = Contract.model_validate(
            {
                "requirements": [
                    {"id": "release", "text": "Add release", "source_quote": "Add release"}
                ]
            }
        ).model_dump()
        store.update(sid, proposal=proposal)
        request = update_request(context(prepared, OTHER))
    with pytest.raises(WorkspaceError):
        guidance.update_guidance(prepared.service, sid, owner, request)
    assert "delivery_guidance_records" not in store.session_record(sid, "owner")


def test_pending_native_plan_binds_to_candidate_instead_of_baseline(prepared):
    store, sid = prepared.service.store, prepared.sid
    revision, proposal = pending_revision(prepared, native_binding=True)
    pending = context(prepared, OTHER)
    assert pending["basis"]["plan_hash"] == completion._hash(
        proposal["operation_plan_snapshot"]["plan"]
    )
    assert (
        pending["captured_plan_hash"] == revision["operation_record"]["plan_snapshot"]["plan_hash"]
    )
    saved = guidance.update_guidance(prepared.service, sid, "owner", update_request(pending))
    store.update(
        sid,
        selected_revision=OTHER,
        accepted_revision=OTHER,
        proposal=None,
        design_plan=proposal["operation_plan_snapshot"]["plan"],
    )
    assert context(prepared)["status"] == "current"
    assert context(prepared)["record"] == saved["record"]


def test_changed_pending_plan_is_refused_without_rebinding_saved_cad(prepared):
    store, sid = prepared.service.store, prepared.sid
    _, proposal = pending_revision(prepared, native_binding=True)
    request = update_request(context(prepared, OTHER))
    proposal = copy.deepcopy(proposal)
    proposal["operation_plan_snapshot"]["plan"]["objective"] = "Changed intent"
    store.update(sid, proposal=proposal)
    with pytest.raises(WorkspaceError, match="Pending candidate plan differs"):
        guidance.update_guidance(prepared.service, sid, "owner", request)


def test_pending_native_candidate_requires_saved_plan_binding(prepared):
    pending_revision(prepared)
    # Corrupt historical storage, rather than asking the API to change immutable mode.
    store = prepared.service.store
    saved = store.session_record(prepared.sid, "owner")
    saved["authoring_mode"] = "native_operations"
    with store.connect() as con:
        con.execute(
            "UPDATE sessions SET document=? WHERE id=?",
            (json.dumps(saved), prepared.sid),
        )
    with pytest.raises(WorkspaceError, match="captured operation plan"):
        context(prepared, OTHER)


def test_legacy_record_is_only_read_for_its_exact_revision(prepared):
    store, sid = prepared.service.store, prepared.sid
    saved = guidance.update_guidance(
        prepared.service, sid, "owner", update_request(context(prepared))
    )["record"]
    store.update(sid, delivery_guidance_records={})
    pending_revision(prepared)
    assert context(prepared)["record"] == saved
    assert context(prepared, OTHER)["record"] is None


@pytest.mark.parametrize("legacy_only", [False, True])
def test_selected_edit_retains_stale_predecessor_without_binding_new_target(prepared, legacy_only):
    store, sid = prepared.service.store, prepared.sid
    saved = guidance.update_guidance(
        prepared.service, sid, "owner", update_request(context(prepared))
    )["record"]
    if legacy_only:
        store.update(sid, delivery_guidance_records={})
    pending_revision(prepared)
    store.update(sid, selected_revision=OTHER, proposal=None)
    before = store.session_record(sid, "owner")
    for target in (None, OTHER):
        edited = context(prepared, target)
        assert edited["basis"]["revision_id"] == OTHER
        assert edited["status"] == "stale"
        assert edited["record"] == saved
        assert edited["record"]["basis"]["revision_id"] == RID
    assert store.session_record(sid, "owner") == before
    assert OTHER not in before["delivery_guidance_records"]

    third = "c" * 32
    pending_revision(prepared, third)
    before_pending_read = store.session_record(sid, "owner")
    pending = context(prepared, third)
    assert pending["target_kind"] == "pending_candidate"
    assert pending["status"] == "unbound" and pending["record"] is None
    assert store.session_record(sid, "owner") == before_pending_read
