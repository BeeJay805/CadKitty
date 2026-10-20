"""Pending final work never supplies a verdict or hides authored probe duties."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from forge_mind.workspace import requirement_verification as verification
from forge_mind.workspace.completion import _pending_final_requirements
from forge_mind.workspace.completion_coverage import functional_coverage
from forge_mind.workspace.completion_work import remaining_work
from forge_mind.workspace.document import digest
from forge_mind.workspace.input_application import baseline
from forge_mind.workspace.models import Contract, DesignPlan, Finding, Measurement, Requirement
from forge_mind.workspace.operation_plan import snapshot
from forge_mind.workspace.store import WorkspaceError
from tests.unit.workspace.test_requirement_verification import digital, geometry_target

RID = "a" * 32


def subsystem(**updates):
    row = {
        "id": "body",
        "requirement_ids": ["nominal", "files", "solid"],
        "status": "ready",
        "missing_part_ids": [],
        "missing_occurrence_paths": [],
        "geometry_declared": True,
        "checks": [],
        "dependencies": [],
        "unresolved": [],
    }
    row.update(updates)
    return row


def bind_saved_intent(case):
    """Capture explicit current identities; the reader never repairs stale fixtures."""
    contract = case.contract.model_dump(mode="json")
    binding = snapshot(case.plan, case.session.get("plan_version", 0) + 1, case.contract)
    case.plan.clear()
    case.plan.update(deepcopy(binding["plan"]))
    case.session.update(
        contract=contract,
        design_plan=case.plan,
        plan_hash=binding["plan_hash"],
        plan_version=binding["plan_version"],
        operation_plan_snapshot=deepcopy(binding),
    )
    case.session["contract_application"] = baseline(case.session)
    case.revision.update(contract=contract, operation_record={"plan_snapshot": deepcopy(binding)})
    case.revision["measurement"]["contract_hash"] = case.contract.fingerprint()
    case.frozen.update(
        contract_hash=case.contract.fingerprint(),
        revision_contract_hash=case.contract.fingerprint(),
        request_hash=digest(case.session["prompt"]),
    )


def bind_saved_service(service, sid, owner, revision, plan):
    """Persist complete saved-byte fixture identities without running geometry."""
    import json

    session = service.store.get(sid, owner)
    case = SimpleNamespace(
        contract=Contract.model_validate(session["contract"]),
        plan=plan,
        session=session,
        revision=revision,
        frozen={},
    )
    bind_saved_intent(case)
    service.store.update(
        sid,
        design_plan=plan,
        plan_hash=session["plan_hash"],
        plan_version=session["plan_version"],
        operation_plan_snapshot=session["operation_plan_snapshot"],
        plan_contract_hash=case.contract.fingerprint(),
        contract_application=session["contract_application"],
    )
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), sid, revision["id"]),
        )


@pytest.fixture
def scheduling(monkeypatch):
    contract = Contract(
        requirements=[
            geometry_target("nominal_perforated_box"),
            digital(),
            Requirement(
                id="solid",
                text="One solid",
                source_quote="One solid",
                source="explicit",
                metric="solid_count",
                value=1,
            ),
        ]
    )
    subject = contract.requirements[0].verification.subject_id
    plan = DesignPlan(
        objective="Nominal digital work",
        verification_bindings=[
            {
                "subject_id": subject,
                "part_id": "base",
                "frame": "definition_local",
            }
        ],
    ).model_dump(mode="json")
    document = {
        "assembly": {
            "parts": [
                {
                    "id": "base",
                    "name": "Plate",
                    "production": {"role": "make"},
                    "features": [
                        {
                            "id": "body",
                            "name": "Nominal body",
                            "kind": "box",
                            "params": {"x": 10, "y": 10, "z": 10},
                        }
                    ],
                    "output": "body",
                }
            ],
            "roots": [{"id": "installed", "definition": "base"}],
        }
    }
    session = {
        "selected_revision": RID,
        "contract": contract.model_dump(mode="json"),
        "design_plan": plan,
        "prompt": "\n".join(r.source_quote for r in contract.requirements),
    }
    measured = Measurement(
        revision_id=RID,
        contract_hash=contract.fingerprint(),
        artifact_hashes={},
        engine_fingerprint="fixture",
        geometry_valid=True,
        findings=[
            Finding(
                requirement_id=r.id,
                status="pass" if r.id == "solid" else "unverified",
                measured=1 if r.id == "solid" else None,
                expected=r.value,
                method="STEP topology" if r.id == "solid" else "no applicable instrument",
                reason="Synthetic reader result; no native execution",
            )
            for r in contract.requirements
        ],
    )
    revision = {"document": document, "measurement": measured.model_dump(mode="json")}
    frozen = {
        "contract_hash": contract.fingerprint(),
        "revision_contract_hash": contract.fingerprint(),
        "engine_fingerprint": "fixture",
        "measured_engine_fingerprint": "fixture",
    }
    reads = []

    def reader(store, sid, owner):
        reads.append((store, sid, owner))
        return deepcopy(session), deepcopy(revision), deepcopy(frozen)

    monkeypatch.setattr(verification, "_frozen_inputs", reader)
    case = SimpleNamespace(
        service=SimpleNamespace(store=object()),
        contract=contract,
        plan=plan,
        session=session,
        revision=revision,
        frozen=frozen,
        reads=reads,
    )
    bind_saved_intent(case)
    return case


def pending(case):
    mapping = functional_coverage(
        case.plan, case.contract.model_dump(mode="json"), source_text=case.session["prompt"]
    )
    return _pending_final_requirements(
        case.service, "s", "o", RID, case.plan, case.contract.model_dump(mode="json"), mapping
    )


def test_exact_host_read_routes_typed_and_existing_scalar_without_pass(scheduling):
    before = deepcopy(scheduling.session)
    ids = pending(scheduling)
    assert ids == {"nominal", "files", "solid"}
    result = remaining_work([subsystem()], [], pending_final_requirement_ids=ids)
    assert result["next_target"] is None
    assert scheduling.reads == [(scheduling.service.store, "s", "o")]
    assert scheduling.session == before
    assert scheduling.revision["measurement"]["findings"][0]["status"] == "unverified"


@pytest.mark.parametrize("status", ["fail", "unverified"])
def test_scalar_fail_or_unknown_keeps_no_check_work_actionable(scheduling, status):
    scheduling.revision["measurement"]["findings"][2]["status"] = status
    ids = pending(scheduling)
    assert "solid" not in ids
    assert (
        remaining_work([subsystem()], [], pending_final_requirement_ids=ids)["next_target"]
        == "body"
    )


def test_scalar_expected_mismatch_is_not_trusted(scheduling):
    scheduling.revision["measurement"]["findings"][2]["expected"] = 2
    assert "solid" not in pending(scheduling)


@pytest.mark.parametrize("change", ["engine", "revision", "contract", "plan"])
def test_stale_or_invalid_authority_refuses_pending_route(scheduling, change):
    if change == "engine":
        scheduling.frozen["measured_engine_fingerprint"] = "old"
    elif change == "revision":
        scheduling.session["selected_revision"] = "b" * 32
    elif change == "contract":
        scheduling.frozen["revision_contract_hash"] = "old"
    elif change == "plan":
        scheduling.session["design_plan"] = {}

    assert pending(scheduling) == set()


def test_missing_geometry_binding_keeps_entire_row_actionable(scheduling):
    scheduling.plan["verification_bindings"] = []
    bind_saved_intent(scheduling)
    ids = pending(scheduling)
    assert ids == {"files", "solid"}
    assert (
        remaining_work([subsystem()], [], pending_final_requirement_ids=ids)["next_target"]
        == "body"
    )


def test_changed_interpreted_source_cannot_use_original_pending_authority(scheduling):
    from forge_mind.workspace.document import digest
    from forge_mind.workspace.intent_attribution import IntentAttribution, validate_attribution

    scheduling.contract.requirements[0].source = "request"
    bind_saved_intent(scheduling)
    contract = scheduling.contract.model_dump(mode="json")
    scheduling.session["contract"] = contract
    checksum = scheduling.contract.fingerprint()
    scheduling.revision["measurement"]["contract_hash"] = checksum
    scheduling.frozen.update(contract_hash=checksum, revision_contract_hash=checksum)
    source = scheduling.session["prompt"]
    attribution = IntentAttribution.model_validate(
        {
            "subjects": [
                {"id": "plate", "label": "Plate", "role": "designed", "source_quote": "plate"}
            ],
            "requirement_subjects": {"nominal": ["plate"]},
        }
    )
    validate_attribution(attribution, scheduling.contract, source)
    record = {
        "version": 1,
        "source_hash": digest(source),
        "source_text": source,
        "proposed_contract": contract,
        "contract_hash": checksum,
        "attribution": attribution.model_dump(mode="json"),
        "basis": {},
        "status": "interpreted",
        "resolutions": [],
    }
    record["draft_hash"] = digest(record)
    scheduling.session["intent_attribution"] = record
    mapping = functional_coverage(scheduling.plan, contract, source_text=source)
    assert _pending_final_requirements(
        scheduling.service, "s", "o", RID, scheduling.plan, contract, mapping
    ) == {"nominal", "files", "solid"}
    scheduling.session["prompt"] = "Different owner source"
    # Keep the earlier mapping to test the fresh host/source authority guard itself.
    assert (
        _pending_final_requirements(
            scheduling.service, "s", "o", RID, scheduling.plan, contract, mapping
        )
        == set()
    )


def test_artifact_reader_refusal_does_not_fall_back_to_cached_status(scheduling, monkeypatch):
    def refuse(*args):
        raise WorkspaceError("Saved artifact changed")

    monkeypatch.setattr(verification, "_frozen_inputs", refuse)
    assert pending(scheduling) == set()


def test_scalar_only_contract_requires_current_reader_instead_of_cached_status(scheduling):
    scheduling.contract = Contract(requirements=[scheduling.contract.requirements[2]])
    scheduling.plan["verification_bindings"] = []
    scheduling.revision["measurement"]["findings"] = [
        scheduling.revision["measurement"]["findings"][2]
    ]
    bind_saved_intent(scheduling)
    assert pending(scheduling) == {"solid"}
    assert scheduling.reads == [(scheduling.service.store, "s", "o")]
    scheduling.revision["measurement"]["findings"][0]["measured"] = None
    assert pending(scheduling) == set()


def test_unmeasured_only_legacy_contract_needs_no_pending_proof_reader(scheduling):
    scheduling.contract = Contract(
        requirements=[
            Requirement(
                id="physical_fit",
                text="Fit the actual object",
                source_quote="Fit the actual object",
            )
        ]
    )
    assert pending(scheduling) == set()
    assert scheduling.reads == []


@pytest.mark.parametrize("missing", ["snapshot", "request_hash", "applied_source"])
def test_current_scalar_reader_cannot_use_missing_saved_identity(scheduling, missing):
    if missing == "snapshot":
        scheduling.revision.pop("operation_record")
    elif missing == "request_hash":
        scheduling.frozen.pop("request_hash")
    else:
        scheduling.session["contract_application"]["applied"].pop("source_hash")
    assert pending(scheduling) == set()


@pytest.mark.parametrize("status", ["fail", "unverified"])
def test_pending_final_route_cannot_close_actual_failed_or_unknown_geometry(scheduling, status):
    assert pending(scheduling) == {"nominal", "files", "solid"}
    results = [
        {"requirement_id": r.id, "status": status if r.id == "nominal" else "pass"}
        for r in scheduling.contract.requirements
    ]
    assert verification.aggregate(scheduling.contract, results) == (status, False)


def test_actual_host_reader_binds_scalar_and_refuses_changed_saved_bytes(tmp_path, monkeypatch):
    import json

    from tests.unit.workspace.test_requirement_verification import prepared

    target = Contract(
        requirements=[
            geometry_target("nominal_perforated_box"),
            digital(),
            Requirement(
                id="solid",
                text="One solid",
                source_quote="One solid",
                source="explicit",
                metric="solid_count",
                value=1,
            ),
        ]
    )
    service, sid, _, _, _, _, directory = prepared(tmp_path, monkeypatch, target=target)
    session = service.store.get(sid, "owner")
    rid = session["selected_revision"]
    revision = service.store.revision(sid, rid)
    revision["measurement"]["findings"][2].update(
        status="pass", expected=1, measured=1, method="STEP topology"
    )
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), sid, rid),
        )
    plan = session["design_plan"]
    bind_saved_service(service, sid, "owner", revision, plan)
    mapping = functional_coverage(
        plan, target.model_dump(mode="json"), source_text=session["prompt"]
    )
    ids = _pending_final_requirements(
        service, sid, "owner", rid, plan, target.model_dump(mode="json"), mapping
    )
    assert ids == {"nominal", "files", "solid"}
    (directory / "part.step").write_bytes(b"Changed saved STEP")
    assert (
        _pending_final_requirements(
            service, sid, "owner", rid, plan, target.model_dump(mode="json"), mapping
        )
        == set()
    )


@pytest.mark.parametrize(
    "updates,category",
    [
        ({"unresolved": ["Unsupported physical strength remains open"]}, "review"),
        ({"requirement_ids": ["nominal", "strength"]}, "review"),
        ({"requirement_ids": []}, "review"),
        ({"geometry_declared": False}, "review"),
        ({"missing_part_ids": ["other"]}, "construction"),
        ({"missing_occurrence_paths": ["other"]}, "construction"),
        ({"status": "checked_fail"}, "repair"),
        ({"checks": [{"status": "fail"}]}, "repair"),
        ({"checks": [{"status": "unverified", "reason": "Not run"}]}, "check"),
    ],
)
def test_pending_cannot_hide_declared_missing_or_mixed_duties(updates, category):
    result = remaining_work(
        [subsystem(**updates)], [], pending_final_requirement_ids={"nominal", "files", "solid"}
    )
    assert result["next_target"] == "body"
    assert result["next_work_category"] == category


def test_parameters_and_interfaces_remain_actionable():
    ids = {"nominal", "files", "solid"}
    step = {"step_id": "inputs", "kind": "parameters", "parameters_present": False}
    result = remaining_work([subsystem()], [], [step], pending_final_requirement_ids=ids)
    assert result["next_target"] == "inputs"
    interface = {
        "next_target": "connection",
        "next_work_category": "review",
        "next_work_reason": "Physical contact unknown",
    }
    result = remaining_work(
        [subsystem()], [], interface_work=interface, pending_final_requirement_ids=ids
    )
    assert result["next_target"] == "connection"


def test_legacy_row_without_unresolved_field_keeps_original_route():
    row = subsystem()
    row.pop("unresolved")
    assert remaining_work([row], [])["next_target"] == "body"
    assert (
        remaining_work([row], [], pending_final_requirement_ids={"nominal", "files", "solid"})[
            "next_target"
        ]
        == "body"
    )


@pytest.mark.parametrize(
    "mode", ["supported", "unresolved", "mixed", "declared", "partial_external"]
)
def test_actual_completion_context_routes_only_complete_pending_work(tmp_path, monkeypatch, mode):
    import json

    from forge_mind.workspace import delivery_guidance
    from forge_mind.workspace.completion import completion_context
    from forge_mind.workspace.document import digest

    actual_guidance_context = delivery_guidance.guidance_context
    from tests.unit.workspace.test_requirement_verification import prepared

    requirements = [geometry_target("nominal_perforated_box"), digital()]
    if mode == "mixed":
        requirements.append(
            Requirement(id="strength", text="Hold one kilogram", source_quote="Hold one kilogram")
        )
    target = Contract(requirements=requirements)
    from forge_mind.workspace.store import Store

    create = Store.create

    def create_native(store, owner, prompt, contract, authoring_mode=None):
        assert authoring_mode is None
        return create(store, owner, prompt, contract, authoring_mode="native_operations")

    # Reuse saved-byte fixtures, but choose immutable native mode at creation.
    monkeypatch.setattr(Store, "create", create_native)
    service, sid, jid, _, _, _, _ = prepared(tmp_path, monkeypatch, target=target)
    assert service.store.authoring_mode(sid) == "native_operations"
    session = service.store.get(sid, "owner")
    rid = session["selected_revision"]
    revision = service.store.revision(sid, rid)
    revision["measurement"].setdefault("facts", {})["assembly"] = {
        "occurrences": [{"path": "installed", "definition": "base", "body_index": 0}]
    }
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), sid, rid),
        )
    row = {
        "id": "body",
        "name": "Saved nominal body",
        "requirement_ids": [r.id for r in requirements],
        "part_ids": ["base"],
        "unresolved": ["A physical qualification duty remains open"]
        if mode == "unresolved"
        else [],
        "checks": [],
    }
    if mode == "declared":
        row["checks"] = [
            {
                "id": "probe",
                "request": {
                    "tool": "tolerance_stack",
                    "units": "mm",
                    "terms": [{"low": 1.0, "high": 2.0}],
                    "allowable": {"low": 0.0, "high": 3.0},
                },
            }
        ]
    plan_value = {**session["design_plan"], "subsystems": [row]}
    if mode == "partial_external":
        plan_value["requirement_dispositions"] = [
            {"requirement_id": "nominal", "physical_validation": ["Retained physical validation"]}
        ]
    plan = DesignPlan.model_validate(plan_value).model_dump(mode="json")
    service.store.update(
        sid,
        design_plan=plan,
        plan_hash=digest(plan),
        plan_version=session["plan_version"] + 1,
    )
    bind_saved_service(service, sid, "owner", revision, plan)
    before = deepcopy(service.store.revision(sid, rid))
    context = completion_context(service, sid, "owner", rid)
    assert context["status"] == "unverified"
    assert context["pending_final_requirement_ids"] == ["files", "nominal"]
    assert context["pending_final_verification_status"] == "unverified"
    assert context["uncovered_requirement_ids"] == []
    if mode == "supported":
        assert context["next_target"] is None
    else:
        assert context["next_target"] == "body"
        assert context["next_work_category"] == ("check" if mode == "declared" else "review")
    assert service.store.revision(sid, rid) == before
    assert all(f["status"] == "unverified" for f in before["measurement"]["findings"])
    if mode == "supported":
        import hashlib
        import time

        from forge_mind.workspace.native_delivery import current_delivery_binding
        from forge_mind.workspace.project_delivery import create_project_export, read_project_export

        # This injects a measured failure at the worker boundary. It is not native proof.
        invoked = []

        def failed_geometry_worker(directory, payload, timeout, cancelled):
            request = payload["request"]
            checksum = hashlib.sha256((directory / "part.step").read_bytes()).hexdigest()
            invoked.append(request["criterion_hash"])
            return {
                "status": "fail",
                "reason": "Injected actual final worker failure",
                "method": request["protocol"],
                "criterion_hash": request["criterion_hash"],
                "facts": {"step_sha256": checksum},
                "evidence_refs": [{"artifact": "part.step", "sha256": checksum}],
            }

        monkeypatch.setattr(service.executor, "mode", "trusted_local")
        monkeypatch.setattr(verification, "process_revision", failed_geometry_worker)
        # The reused prepared fixture substitutes a small postcapture guidance projection.
        # Regeneration needs the real complete current context and its export basis.
        monkeypatch.setattr(delivery_guidance, "guidance_context", actual_guidance_context)
        receipt = create_project_export(service, sid, "owner", rid, include_stl=True)
        _, readback = read_project_export(
            service, sid, "owner", receipt["package_id"], receipt["sha256"]
        )
        assert readback == receipt
        service.store.update(
            sid,
            automatic_delivery={
                **current_delivery_binding(service, sid, "owner"),
                "job_id": jid,
                "status": "delivered",
                "readback_verified": True,
                "receipt": receipt,
            },
        )
        proof_receipt = verification.verify_contract(
            service, sid, "owner", jid, deadline=time.monotonic() + 30, cancelled=lambda: False
        )
        report = json.loads(
            verification._report_path(service.store, sid, proof_receipt["report_id"]).read_bytes()
        )
        assert len(invoked) == 1
        assert report["results"][0]["requirement_id"] == "nominal"
        assert report["results"][0]["status"] == "fail"
        assert report["status"] == "fail" and report["requirements_satisfied"] is False
        service.store.finish(jid, sid, "contract_satisfied", verification_report=proof_receipt)
        assert service.store.get(sid, "owner")["state"] != "contract_satisfied"


@pytest.mark.parametrize("key", ["missing_inputs", "unsupported_checks", "physical_validation"])
@pytest.mark.parametrize("explicit_context", [False, True])
def test_partial_external_disposition_keeps_existing_review_target(key, explicit_context):
    disposition = {"requirement_id": "nominal", "declared": True, key: ["Retained external duty"]}
    result = remaining_work(
        [subsystem()],
        [disposition],
        pending_final_requirement_ids={"nominal", "files", "solid"},
        **(
            {"executable_requirement_ids": {"nominal", "files", "solid"}}
            if explicit_context
            else {}
        ),
    )
    assert result["next_target"] == "body"
    assert result["next_work_category"] == "review"
    assert result["external_obligations"] == [
        {
            "subsystem_id": "body",
            "obligations": [{"requirement_id": "nominal", key: ["Retained external duty"]}],
        }
    ]


@pytest.mark.parametrize("key", ["physical_validation", "unsupported_checks"])
@pytest.mark.parametrize("check_status", ["pass", "fail", "Not run", "stale"])
def test_executable_obligation_remains_after_unrelated_prepared_check(key, check_status):
    row = subsystem(unresolved=["Extra obligation remains open"])
    row["checks"] = [
        {
            "id": "unrelated",
            "status": "unverified" if check_status in {"Not run", "stale"} else check_status,
            "reason": check_status,
        }
    ]
    row["status"] = "checked_pass" if check_status == "pass" else "unverified"
    before = deepcopy(row)
    result = remaining_work(
        [row],
        [{"requirement_id": "nominal", "declared": True, key: ["Extra source duty"]}],
        pending_final_requirement_ids={"nominal", "files", "solid"},
        executable_requirement_ids={"nominal", "files", "solid"},
    )
    assert result["next_target"] == "body"
    assert (
        result["next_work_category"]
        == {"pass": "review", "fail": "repair", "Not run": "check", "stale": "review"}[check_status]
    )
    assert result["external_obligations"][0]["obligations"] == [
        {"requirement_id": "nominal", key: ["Extra source duty"]}
    ]
    assert row == before
