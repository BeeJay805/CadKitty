"""Scheduling coverage combines current evidence without changing acceptance."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.unit.workspace.test_completion import FakeStore

from forge_mind.workspace.completion import (
    _current_attribution,
    _pending_final_requirements,
    completion_context,
)
from forge_mind.workspace.completion_coverage import functional_coverage
from forge_mind.workspace.completion_work import remaining_work
from forge_mind.workspace.document import digest
from forge_mind.workspace.models import Contract, DesignPlan, Finding, Measurement, Requirement
from forge_mind.workspace.operation_plan import snapshot
from forge_mind.workspace.requirement_verification import _frozen_inputs as real_frozen_inputs

RID = "a" * 32
ENGINE = "b" * 64
SOURCE = "Make a 30 mm by 20 mm by 8 mm part. Physical fit needs a bench test."


@pytest.fixture
def current_case(monkeypatch):
    contract = Contract(
        requirements=[
            Requirement(
                id=f"extent_{axis}",
                text=f"{value} mm size",
                source_quote=f"{value} mm",
                metric=f"extent_{axis}",
                value=float(value),
            )
            for axis, value in zip("xyz", [30, 20, 8], strict=True)
        ]
        + [
            Requirement(
                id="fit",
                text="Physical fit needs a bench test.",
                source_quote="Physical fit needs a bench test.",
            )
        ]
    )
    plan = DesignPlan.model_validate(
        {
            "objective": "Preserve the source dimensions and unknown physical fit",
            "operation_plan": {
                "part_ids": ["part"],
                "steps": [
                    {
                        "id": "build",
                        "part_id": "part",
                        "feature_ids": ["body"],
                        "requirement_ids": [row.id for row in contract.requirements],
                        "expected_effect": "Build requested material",
                    }
                ],
            },
            "subsystems": [
                {
                    "id": "body",
                    "name": "Requested part",
                    "part_ids": ["part"],
                    "requirement_ids": [row.id for row in contract.requirements],
                    "unresolved": ["Physical fit still needs independent bench observations."],
                }
            ],
            "requirement_dispositions": [
                {"requirement_id": "fit", "physical_validation": ["Test the actual mating object"]}
            ],
        }
    ).model_dump(mode="json")
    binding = snapshot(plan, 1, contract)
    session = {
        "authoring_mode": "native_operations",
        "selected_revision": RID,
        "prompt": SOURCE,
        "contract": contract.model_dump(),
        "design_plan": plan,
        "plan_hash": binding["plan_hash"],
        "plan_version": 1,
        "plan_contract_hash": contract.fingerprint(),
        "operation_plan_snapshot": binding,
        "engineering": {"version": 0},
        "job_id": "job",
    }
    measured = Measurement(
        revision_id=RID,
        contract_hash=contract.fingerprint(),
        engine_fingerprint=ENGINE,
        artifact_hashes={"part.step": "c" * 64},
        geometry_valid=True,
        findings=[
            Finding(
                requirement_id=row.id,
                status="pass" if row.value is not None else "unverified",
                expected=row.value,
                measured=row.value,
                method="STEP bounding box" if row.value is not None else "unsupported",
                reason="Controlled saved measurement; no native worker in this test",
            )
            for row in contract.requirements
        ],
    )
    revision = {
        "id": RID,
        "contract": contract.model_dump(),
        "operation_record": {"plan_snapshot": binding},
        "document": {
            "features": [
                {"id": "body", "name": "Body", "kind": "box", "params": {"x": 30, "y": 20, "z": 8}}
            ],
            "outputs": ["body"],
        },
        "measurement": measured.model_dump(),
    }
    frozen = {
        "contract_hash": contract.fingerprint(),
        "revision_contract_hash": contract.fingerprint(),
        "engine_fingerprint": ENGINE,
        "measured_engine_fingerprint": ENGINE,
        "request_hash": digest(SOURCE),
        "revision_id": RID,
        "plan_hash": binding["plan_hash"],
        "plan_version": 1,
        "job_id": "job",
    }
    store = FakeStore(session)
    service = SimpleNamespace(store=store, _load_revision=lambda *args: deepcopy(revision))
    monkeypatch.setattr(
        "forge_mind.workspace.requirement_verification._frozen_inputs",
        lambda *args: (deepcopy(store.session), deepcopy(revision), deepcopy(frozen)),
    )
    monkeypatch.setattr(
        "forge_mind.workspace.completion.read_sources", lambda *args: {"fingerprint": "source"}
    )
    monkeypatch.setattr("forge_mind.workspace.completion.engine_fingerprint", lambda: ENGINE)
    case = SimpleNamespace(
        service=service,
        contract=contract,
        revision=revision,
        session=store.session,
        plan=plan,
        frozen=frozen,
    )
    yield case
    store.connection.close()


def pending(case):
    return _pending_final_requirements(
        case.service,
        "s",
        "o",
        RID,
        case.plan,
        case.contract.model_dump(),
        functional_coverage(case.plan, case.contract, source_text=SOURCE),
    )


def test_scalar_only_current_measurements_are_collected_without_typed_predicates(current_case):
    assert pending(current_case) == {"extent_x", "extent_y", "extent_z"}


def test_mixed_scalar_passes_and_retained_external_unknowns_need_no_duplicate_probe(current_case):
    original = deepcopy((current_case.session, current_case.revision))
    result = completion_context(current_case.service, "s", "o", RID)
    assert result["next_target"] is None
    assert result["status"] == "unverified"
    assert result["subsystems"][0]["checks"] == []
    assert result["external_obligations"][0]["obligations"][0]["requirement_id"] == "fit"
    assert (current_case.session, current_case.revision) == original


def work_row():
    return {
        "id": "body",
        "requirement_ids": ["extent_x", "fit"],
        "dependencies": [],
        "status": "ready",
        "checks": [],
        "geometry_declared": True,
        "missing_part_ids": [],
        "missing_occurrence_paths": [],
        "unresolved": ["Physical qualification remains unknown"],
    }


@pytest.mark.parametrize("with_passing_check", [False, True])
def test_pure_projector_combines_scheduling_categories_without_rewriting_rows(with_passing_check):
    row = work_row()
    if with_passing_check:
        row["checks"] = [{"id": "current_probe", "status": "pass"}]
    result = remaining_work(
        [row],
        [{"requirement_id": "fit", "declared": True, "physical_validation": ["Bench test"]}],
        pending_final_requirement_ids={"extent_x"},
        executable_requirement_ids={"extent_x"},
    )
    assert result["next_target"] is None
    assert row["status"] == "ready" and row["unresolved"]


def rebind(case):
    binding = snapshot(case.plan, case.session["plan_version"], case.contract)
    case.session.update(
        contract=case.contract.model_dump(),
        design_plan=binding["plan"],
        plan_hash=binding["plan_hash"],
        plan_contract_hash=case.contract.fingerprint(),
        operation_plan_snapshot=binding,
    )
    case.plan = case.session["design_plan"]
    case.revision.update(
        contract=case.contract.model_dump(), operation_record={"plan_snapshot": deepcopy(binding)}
    )
    case.revision["measurement"]["contract_hash"] = case.contract.fingerprint()
    case.frozen.update(
        contract_hash=case.contract.fingerprint(),
        revision_contract_hash=case.contract.fingerprint(),
        plan_hash=binding["plan_hash"],
        plan_version=case.session["plan_version"],
    )


@pytest.mark.parametrize(
    "finding_change", ["fail", "missing", "wrong_target", "missing_value", "contradictory_value"]
)
@pytest.mark.parametrize("unrelated_check_pass", [False, True])
def test_external_disposition_cannot_mask_a_failed_or_missing_executable_scalar(
    current_case, finding_change, unrelated_check_pass
):
    case = current_case
    case.plan["requirement_dispositions"].append(
        {
            "requirement_id": "extent_x",
            "unsupported_checks": [
                "An authored unsupported label cannot waive the saved instrument"
            ],
        }
    )
    rebind(case)
    finding = case.revision["measurement"]["findings"][0]
    if finding_change == "fail":
        finding["status"] = "fail"
    if finding_change == "missing":
        case.revision["measurement"]["findings"].pop(0)
    if finding_change == "wrong_target":
        finding["expected"] = 31.0
    if finding_change == "missing_value":
        finding["measured"] = None
    if finding_change == "contradictory_value":
        finding["measured"] = 31.0
    assert "extent_x" not in pending(case)
    row = work_row()
    if unrelated_check_pass:
        row.update(status="checked_pass", checks=[{"id": "other", "status": "pass"}])
    result = remaining_work(
        [row],
        [
            {"requirement_id": "fit", "declared": True, "physical_validation": ["Bench test"]},
            {
                "requirement_id": "extent_x",
                "declared": True,
                "unsupported_checks": ["Do not check"],
            },
        ],
        pending_final_requirement_ids=pending(case),
        executable_requirement_ids={"extent_x"},
    )
    assert result["next_target"] == "body" and result["next_work_category"] == "review"


@pytest.mark.parametrize(
    "fault",
    [
        "revision",
        "contract",
        "engine",
        "source",
        "plan",
        "captured_plan",
        "captured_contract",
        "captured_version",
    ],
)
def test_current_scalar_authority_cannot_be_reused_after_bound_identity_changes(
    current_case, fault
):
    case = current_case
    if fault == "revision":
        case.revision["measurement"]["revision_id"] = "d" * 32
    if fault == "contract":
        case.revision["measurement"]["contract_hash"] = "d" * 64
    if fault == "engine":
        case.revision["measurement"]["engine_fingerprint"] = "d" * 64
    if fault == "source":
        case.session["prompt"] += " Change the requested width."
    if fault == "plan":
        case.session["plan_hash"] = "d" * 64
    if fault == "captured_plan":
        case.revision["operation_record"]["plan_snapshot"]["plan"] = {}
    if fault == "captured_contract":
        case.revision["operation_record"]["plan_snapshot"]["contract_hash"] = "d" * 64
    if fault == "captured_version":
        case.revision["operation_record"]["plan_snapshot"]["plan_version"] += 1
    assert pending(case) == set()
    assert completion_context(case.service, "s", "o", RID)["next_target"] is not None


@pytest.mark.parametrize("fault", ["source", "contract", "pending"])
def test_fresh_snapshot_cannot_reuse_scalar_coverage_for_unapplied_owner_input(current_case, fault):
    from forge_mind.workspace.input_application import baseline

    case = current_case
    case.session["contract_application"] = baseline(case.session)
    assert pending(case) == {"extent_x", "extent_y", "extent_z"}
    if fault == "source":
        case.session["prompt"] += " Change to 31 mm."
        # A fresh current snapshot alone cannot bind an old measurement to new intent.
        case.frozen["request_hash"] = digest(case.session["prompt"])
    elif fault == "contract":
        case.session["contract_application"]["applied"]["contract_hash"] = "d" * 64
    else:
        case.session["contract_application"]["pending"] = {"input_id": "unapplied"}
    assert pending(case) == set()


@pytest.mark.parametrize("unrelated_check_pass", [False, True])
def test_unsupported_label_cannot_cover_an_executable_typed_duty(unrelated_check_pass):
    row = work_row()
    row["requirement_ids"] = ["typed_target", "fit"]
    if unrelated_check_pass:
        row.update(status="checked_pass", checks=[{"id": "other", "status": "pass"}])
    result = remaining_work(
        [row],
        [
            {"requirement_id": "typed_target", "declared": True, "unsupported_checks": ["Skip"]},
            {"requirement_id": "fit", "declared": True, "physical_validation": ["Bench test"]},
        ],
        executable_requirement_ids={"typed_target"},
    )
    assert result["next_target"] == "body" and result["next_work_category"] == "review"


@pytest.mark.parametrize(
    "check",
    [
        {"status": "unverified", "reason": "Not run"},
        {"status": "fail", "reason": "Measured failure"},
        {"status": "unverified", "reason": "Receipt is stale"},
        {"status": "unverified", "reason": "Invalid request"},
    ],
)
def test_declared_checks_remain_actionable_after_scheduling_union(check):
    row = work_row()
    row["checks"] = [check]
    result = remaining_work(
        [row],
        [{"requirement_id": "fit", "declared": True, "physical_validation": ["Bench test"]}],
        pending_final_requirement_ids={"extent_x"},
        executable_requirement_ids={"extent_x"},
    )
    assert result["next_target"] == "body"
    assert result["next_work_category"] == (
        "repair"
        if check["status"] == "fail"
        else "check"
        if check["reason"] == "Not run"
        else "review"
    )


@pytest.mark.parametrize(
    "fault",
    ["part", "occurrence", "operation", "functional_interface", "uncovered", "source_input"],
)
def test_other_actionable_work_is_not_erased_by_valid_scalar_and_external_categories(
    current_case, fault
):
    case = current_case
    if fault == "part":
        case.plan["subsystems"][0]["part_ids"] = ["missing_part"]
    if fault == "occurrence":
        case.plan["subsystems"][0]["occurrence_paths"] = ["missing_occurrence"]
    if fault == "operation":
        case.plan["operation_plan"]["steps"][0]["feature_ids"] = ["missing_feature"]
    if fault == "functional_interface":
        case.plan["functional_interfaces"] = [
            {
                "id": "mating",
                "requirement_ids": ["fit"],
                "participant_paths": ["part", "absent"],
                "duties": [{"id": "fit_duty", "kind": "fit", "proof_scope": "nominal_fit"}],
            }
        ]
    if fault == "uncovered":
        case.plan["subsystems"][0]["requirement_ids"].remove("extent_x")
    if fault == "source_input":
        case.plan["requirement_dispositions"][0]["missing_inputs"] = [
            "Owner must specify the mating object's size"
        ]
    rebind(case)
    if fault == "uncovered":
        case.revision["measurement"]["findings"][0]["status"] = "unverified"
    result = completion_context(case.service, "s", "o", RID)
    assert result["next_target"] is not None
    assert result["status"] != "checked_pass"


def test_valid_multipart_coverage_does_not_require_an_artificial_probe(current_case):
    case = current_case
    features = case.revision["document"]["features"]
    case.revision["document"] = {
        "assembly": {
            "parts": [
                {"id": "part", "name": "Support", "features": features, "output": "body"},
                {"id": "cover", "name": "Cover", "features": deepcopy(features), "output": "body"},
            ],
            "roots": [
                {"id": "support", "definition": "part"},
                {"id": "cover", "definition": "cover"},
            ],
        }
    }
    case.revision["measurement"]["facts"]["assembly"] = {
        "occurrences": [
            {"path": "support", "definition": "part", "body_index": 0},
            {"path": "cover", "definition": "cover", "body_index": 1},
        ]
    }
    case.plan["subsystems"].append(
        {
            "id": "cover",
            "name": "Cover",
            "dependencies": ["body"],
            "part_ids": ["cover"],
            "requirement_ids": ["extent_x", "fit"],
        }
    )
    case.plan["operation_plan"]["part_ids"].append("cover")
    case.plan["operation_plan"]["steps"].append(
        {
            "id": "cover_build",
            "part_id": "cover",
            "feature_ids": ["body"],
            "requirement_ids": ["fit"],
            "expected_effect": "Construct cover",
        }
    )
    rebind(case)
    result = completion_context(case.service, "s", "o", RID)
    assert result["next_target"] is None
    case.session["plan_version"] += 1
    assert completion_context(case.service, "s", "o", RID)["next_target"] is not None


def test_supported_typed_pending_final_and_external_unknown_union_retains_unknown_verdict(
    current_case,
    monkeypatch,
):
    import json

    from tests.unit.workspace.test_requirement_verification import digital

    from forge_mind.workspace.requirement_verification import inspect_requirements

    case = current_case
    case.contract.requirements.append(digital("project_step"))
    case.plan["subsystems"][0]["requirement_ids"].append("files")
    case.revision["measurement"]["findings"].append(
        Finding(
            requirement_id="files",
            status="unverified",
            method="digital_delivery",
            reason="Exact capture has not run",
        ).model_dump()
    )
    rebind(case)
    assert "files" in pending(case)
    assert completion_context(case.service, "s", "o", RID)["next_target"] is None
    case.service.store.connection.execute(
        "CREATE TABLE sessions(id TEXT, owner TEXT, document TEXT)"
    )
    case.service.store.connection.execute(
        "INSERT INTO sessions VALUES(?,?,?)", ("s", "o", json.dumps(case.session))
    )
    case.service.store.connection.commit()
    monkeypatch.setattr(
        "forge_mind.workspace.requirement_verification.current_basis",
        lambda *args: deepcopy(case.frozen),
    )
    result = inspect_requirements(
        case.service, "s", "o", RID, deadline=None, cancelled=lambda: False
    )
    assert {row["requirement_id"]: row["status"] for row in result["results"]} == {
        "extent_x": "pass",
        "extent_y": "pass",
        "extent_z": "pass",
        "fit": "unverified",
        "files": "unverified",
    }


@pytest.mark.parametrize("fault", [None, "source", "contract", "marker", "binding"])
def test_common_nominal_mapping_requires_current_admitted_attribution_and_binding(fault):
    from tests.unit.workspace.test_customer_interpretation import SOURCE as NOMINAL_SOURCE
    from tests.unit.workspace.test_customer_interpretation import wire

    from forge_mind.workspace.source_delivery_contract_draft import lower_source_delivery_draft

    draft = lower_source_delivery_draft(wire(), NOMINAL_SOURCE)
    plan = DesignPlan(
        objective="Exact recorded interpretation",
        verification_bindings=[
            {"subject_id": "plate", "part_id": "part", "frame": "definition_local"}
        ],
    ).model_dump()
    record = {
        "status": "interpreted",
        "source_hash": digest(NOMINAL_SOURCE),
        "source_text": NOMINAL_SOURCE,
        "proposed_contract": draft.contract.model_dump(),
        "contract_hash": draft.contract.fingerprint(),
        "attribution": draft.attribution.model_dump(),
    }
    record["draft_hash"] = digest(record)
    session = {"prompt": NOMINAL_SOURCE, "intent_attribution": record}
    if fault == "source":
        session["prompt"] += " Change it."
    if fault == "contract":
        draft.contract.coverage_notes.append("Changed source duty")
    if fault == "marker":
        record["attribution"]["customer_nominal_interpretation"]["source_sha256"] = "d" * 64
        record.pop("draft_hash")
        record["draft_hash"] = digest(record)
    if fault == "binding":
        plan["verification_bindings"] = []
    attribution = _current_attribution(session, draft.contract.model_dump())
    result = functional_coverage(
        plan, draft.contract, source_text=session["prompt"], attribution=attribution
    )
    if fault is None:
        assert [row["requirement_id"] for row in result["typed_requirements"]] == ["cad_body"]
        assert "cad_body" not in result["missing_requirement_ids"]
        assert all(
            row.critical and row.metric == "unmeasured" for row in draft.contract.requirements[:-1]
        )
    elif fault == "binding":
        assert result["typed_requirements"][0]["mapping_status"] == "missing_subject_binding"
    else:
        assert "cad_body" in result["missing_requirement_ids"]


def test_pure_native_handoff_capture_and_final_readback_preserve_physical_unknown(
    current_case, tmp_path, monkeypatch
):
    """Exercise service boundaries with saved bytes; no geometry or model executes."""
    import hashlib
    import io
    from zipfile import ZipFile

    from tests.unit.workspace.test_requirement_verification import digital

    from forge_mind.workspace import completion, delivery_guidance, measure, project_delivery
    from forge_mind.workspace import requirement_verification as verification
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.input_application import baseline
    from forge_mind.workspace.instruction_context import ProtectedGuidanceUpdate
    from forge_mind.workspace.native_delivery import capture_native_delivery
    from forge_mind.workspace.native_finish import finish_context
    from forge_mind.workspace.native_handoff_prepare import (
        PrepareHandoffRequest,
        prepare_handoff,
    )
    from forge_mind.workspace.operation_plan import coverage
    from forge_mind.workspace.store import Store

    monkeypatch.setattr(verification, "_frozen_inputs", real_frozen_inputs)
    monkeypatch.setattr(
        "forge_mind.workspace.native_handoff_prepare._frozen_inputs", real_frozen_inputs
    )
    for module in (completion, measure, delivery_guidance, project_delivery):
        monkeypatch.setattr(module, "engine_fingerprint", lambda: ENGINE)
    service = WorkspaceService(Store(tmp_path / "handoff"), Executor("trusted_local"))

    def forbidden(*args, **kwargs):
        pytest.fail("Pure handoff test cannot execute a model, CAD or worker")

    monkeypatch.setattr(service, "start", forbidden)
    monkeypatch.setattr(service, "build", forbidden)
    monkeypatch.setattr(service.executor, "build", forbidden)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbidden)
    case = current_case
    contract = case.contract.model_copy(deep=True)
    contract.requirements.append(digital("project_step"))
    plan = deepcopy(case.plan)
    plan["subsystems"][0]["requirement_ids"].append("files")
    sid = service.store.create(
        "owner",
        SOURCE + " Deliver STEP.",
        contract.model_dump(),
        authoring_mode="native_operations",
    )["id"]
    service.store.set_design_plan(
        sid, "owner", DesignPlan.model_validate(plan), expected_plan_version=0, contract=contract
    )
    session = service.store.session_record(sid, "owner")
    service.store.update(sid, contract_application=baseline(session))
    binding = service.store.plan_snapshot(sid, "owner")
    document = DesignDocument.model_validate(
        {
            **case.revision["document"],
            "production": {"role": "make", "process": "cnc", "deliver_stl": False},
        }
    )
    directory = service.store.directory(sid, RID)
    directory.mkdir(parents=True)
    files = {
        "program.py": b"# Saved fixture; never execute.\n",
        "part.step": b"Saved-byte fixture, not independent geometric evidence",
        "part.stl": b"Saved mesh fixture; never render",
        "document.json": document.model_dump_json().encode(),
    }
    for name, content in files.items():
        (directory / name).write_bytes(content)
    measured = Measurement.model_validate(
        {
            **case.revision["measurement"],
            "contract_hash": contract.fingerprint(),
            "artifact_hashes": {
                name: hashlib.sha256(value).hexdigest() for name, value in files.items()
            },
            "findings": case.revision["measurement"]["findings"]
            + [
                Finding(
                    requirement_id="files",
                    status="unverified",
                    method="digital_delivery",
                    reason="Final exact capture has not run",
                ).model_dump()
            ],
        }
    )
    service.store.add_revision(
        sid,
        {
            "id": RID,
            "document": document.model_dump(),
            "contract": contract.model_dump(),
            "measurement": measured.model_dump(),
            "operation_record": {**coverage(binding, document), "plan_snapshot": binding},
        },
    )
    service.store.update(sid, selected_revision=RID)
    jid = service.store.enqueue(sid, "owner", {"max_seconds": 120, "delivery_requested": True})
    job = service.store.claim()
    assert job["id"] == jid
    prepared = prepare_handoff(
        service,
        sid,
        "owner",
        jid,
        PrepareHandoffRequest(revision_id=RID, prepare_print=False),
        deadline=None,
        cancelled=lambda: False,
    )
    assert prepared["status"] == "prepared_for_instruction_review", prepared
    assert {
        r["requirement_id"]: r["status"] for r in prepared["requirements_inspection"]["results"]
    }["fit"] == "unverified"
    guide = prepared["delivery_guidance"]
    delivery_guidance.update_guidance(
        service,
        sid,
        "owner",
        ProtectedGuidanceUpdate.model_validate(
            {
                "revision_id": RID,
                "basis_hash": guide["basis_hash"],
                "reviewed_current_facts": True,
                "guidance": {
                    "print": "No printer setup is qualified for this CNC file fixture.",
                    "installation": "Check the actual mating object before installation.",
                    "use_and_removal": "Keep access clear during use and removal.",
                    "unresolved": "Physical fit and strength remain unverified.",
                },
                "instruction_plan": {
                    "version": 1,
                    "fact_refs": guide["instruction_catalog"]["required_refs"],
                    "steps": [{"action": "review_physical_limits", "target": "physical:readiness"}],
                },
            }
        ),
    )
    review = finish_context(service, sid, "owner", RID, require_bound_instructions=True)
    assert review["status"] == "ready_for_validation", review
    assert review["physical_readiness"] == "unverified"
    service.store.update(sid, native_review={**review, "job_id": jid})
    capture = capture_native_delivery(service, job, None, lambda: False, "needs_review")
    assert capture["status"] == "delivered", capture
    content, receipt = project_delivery.read_project_export(
        service, sid, "owner", capture["receipt"]["package_id"], capture["receipt"]["sha256"]
    )
    assert receipt == capture["receipt"]
    with ZipFile(io.BytesIO(content)) as archive:
        assert archive.read("assembly/part.step") == files["part.step"]
        assert "Physical fit and strength remain unverified." in archive.read("README.md").decode()
    proof = verification.verify_contract(
        service, sid, "owner", jid, deadline=None, cancelled=lambda: False
    )
    service.store.finish(jid, sid, "contract_satisfied", verification_report=proof)
    result = service.store.get(sid, "owner")
    assert result["state"] == "needs_review"
    assert result.get("accepted_revision") is None
    report = verification.read_verification(service.store, sid, "owner")
    assert report["requirements_satisfied"] is False
    assert report["author_closure_ready"] is True
    assert report["physical_readiness"] == "unverified"
    rows = {row["requirement_id"]: row for row in report["results"]}
    assert rows["files"]["status"] == "pass"
    assert rows["fit"]["status"] == "unverified" and rows["fit"]["critical"] is True
    assert all(rows[f"extent_{axis}"]["status"] == "pass" for axis in "xyz")
    assert len(service.store.revisions(sid)) == 1
