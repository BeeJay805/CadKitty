"""Customer file policy dispatch and negative controls; mocked readers prove no geometry."""

from copy import deepcopy

import pytest
from pydantic import ValidationError
from tests.unit.workspace.test_mesh_handoff import document_data

from forge_mind.workspace import delivery_guidance, native_delivery, print_setup_service
from forge_mind.workspace.author_guidance import build_author_guidance
from forge_mind.workspace.customer_delivery_policy import (
    customer_delivery_context,
    customer_print_document_gaps,
    customer_print_requested,
)
from forge_mind.workspace.customer_projects import CustomerProjectRequest
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.store import WorkspaceError


def customer(printable=True):
    return {
        "project_kind": "workspace",
        "creation_request_id": "a" * 32,
        "authoring_mode": "native_operations",
        "customer_request_version": 2,
        "customer_printable": printable,
    }


def test_print_choice_is_strict_and_defaults_only_at_new_request_admission():
    request = {"request_id": "a" * 32, "prompt": "A plate"}
    assert CustomerProjectRequest.model_validate(request).printable is True
    assert CustomerProjectRequest.model_validate({**request, "printable": False}).printable is False
    for value in ("false", 0, 1, None):
        with pytest.raises(ValidationError):
            CustomerProjectRequest.model_validate({**request, "printable": value})
    assert customer_print_requested(customer())
    assert not customer_print_requested(customer(False))
    assert not customer_print_requested({"customer_printable": True})
    legacy = {
        key: value
        for key, value in customer().items()
        if key not in {"customer_printable", "customer_request_version"}
    }
    assert not customer_print_requested(legacy)
    assert customer_delivery_context(legacy) is None


def test_context_keeps_software_choice_separate_from_physical_proof_and_source_quotes():
    policy = customer_delivery_context(customer())
    assert policy["printable"] is True
    assert set(policy["required_deliverables"]) == {
        "project_step",
        "editable_native",
        "project_stl",
        "prepared_print_package",
        "bound_instruction_plan",
    }
    cad = customer_delivery_context(customer(False))
    assert cad["printable"] is False
    assert "prepared_print_package" not in cad["required_deliverables"]
    assert "physical fit" in cad["scope"]
    assert "Explicit owner print requirements" in cad["scope"]
    with pytest.raises(WorkspaceError, match="strict printable"):
        customer_delivery_context(customer("false"))
    guidance = build_author_guidance(
        {"authoring_mode": "native_operations", "delivery_policy": policy}, "trusted_local"
    )
    assert "production.deliver_stl=true" in guidance
    assert "without inventing owner source quotes" in guidance
    assert "Do not invent a material" in guidance


def test_print_policy_cannot_be_disabled_by_missing_mesh_roles_or_bought_only_inventory():
    assert customer_print_document_gaps(DesignDocument.model_validate(document_data())) == []
    missing_mesh = document_data(selected=False)
    assert any(
        "deliver_stl=true" in reason
        for reason in customer_print_document_gaps(DesignDocument.model_validate(missing_mesh))
    )
    unknown = document_data()
    unknown["assembly"]["parts"][0]["production"] = {"role": "unspecified"}
    assert any(
        "Declare production roles" in reason
        for reason in customer_print_document_gaps(DesignDocument.model_validate(unknown))
    )
    bought_only = document_data(selected=False)
    for part in bought_only["assembly"]["parts"]:
        part["production"]["role"] = "buy"
    assert any(
        "at least one" in reason
        for reason in customer_print_document_gaps(DesignDocument.model_validate(bought_only))
    )
    flat = DesignDocument.model_validate({"production": {"role": "make", "process": "cnc"}})
    assert any("explicit assembly" in reason for reason in customer_print_document_gaps(flat))


@pytest.mark.parametrize(
    "printable,process,required",
    [(True, "cnc", True), (False, "cnc", False), (False, "additive", True)],
)
def test_final_review_reuses_preparation_and_keeps_additive_prerequisites(
    monkeypatch, printable, process, required
):
    from tests.unit.workspace.test_bound_instruction_context import closure_case

    module, service, sid, rid = closure_case(monkeypatch, binding="bound")
    get, revision = service.store.get, service._load_revision
    raw = document_data(selected=printable)
    for part in raw["assembly"]["parts"]:
        if part["production"]["role"] == "make":
            part["production"]["process"] = process
    service.store.get = lambda *args: {**get(*args), **customer(printable)}
    service._load_revision = lambda *args: {**revision(*args), "document": raw}
    review = module.finish_context(service, sid, "owner", rid)
    assert review["print_preparation"]["required"] is required
    assert review["status"] == ("needs_work" if required else "ready_for_validation")
    assert review["physical_readiness"] == "unverified"


def policy_capture(monkeypatch, printable=True):
    from tests.unit.workspace.test_native_delivery import RID, SID, prepared

    fixture = prepared(monkeypatch)
    fixture.session.update(customer(printable))
    fixture.revision["document"] = DesignDocument.model_validate(document_data()).model_dump()
    fixture.receipt.update(
        include_stl=True,
        made_parts=[
            {
                "part_id": part_id,
                "quantity": quantity,
                "files": [f"make/definition_{part_id}.step", f"make/definition_{part_id}.stl"],
            }
            for part_id, quantity in (("base", 1), ("cap", 2))
        ],
    )
    fixture.guide = {
        "status": "current",
        "target_kind": "selected_candidate",
        "selected_revision": RID,
        "basis_hash": "d" * 64,
        "record": {
            "guidance": dict.fromkeys(
                ("print", "installation", "use_and_removal", "unresolved"), "Saved design note"
            )
        },
        "instruction_binding": {"status": "bound"},
    }
    fixture.setup = {
        "status": "current",
        "record_hash": "e" * 64,
        "record": {
            "package": {"revision_id": RID, "package_id": "f" * 32, "sha256": "a" * 64},
            "files": {
                "print/base.stl": {"part_id": "base", "quantity": 1},
                "print/cap.stl": {"part_id": "cap", "quantity": 2},
            },
        },
    }
    fixture.setup_reads = []

    def guide(*args):
        assert args[1:] == (SID, "owner", RID)
        return deepcopy(fixture.guide)

    def setup(*args):
        assert args[1:] == (SID, "owner", RID)
        fixture.setup_reads.append(args)
        return deepcopy(fixture.setup)

    monkeypatch.setattr(delivery_guidance, "guidance_context", guide)
    monkeypatch.setattr(print_setup_service, "print_setup_context", setup)
    return fixture


def test_customer_policy_routes_all_made_print_delivery_through_existing_readers(monkeypatch):
    fixture = policy_capture(monkeypatch)
    report = native_delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "delivered"
    assert report["readback_verified"] is True
    assert fixture.captures[0][1]["include_stl"] is True
    assert len(fixture.setup_reads) == 2


@pytest.mark.parametrize(
    "change", ["missing_mesh", "stale_print", "wrong_quantity", "missing_part", "unbound_guide"]
)
def test_missing_customer_print_prerequisites_cannot_deliver(monkeypatch, change):
    fixture = policy_capture(monkeypatch)
    if change == "missing_mesh":
        fixture.revision["document"]["assembly"]["parts"][0]["production"]["deliver_stl"] = False
    elif change == "stale_print":
        fixture.setup["status"] = "stale"
    elif change == "wrong_quantity":
        fixture.setup["record"]["files"]["print/cap.stl"]["quantity"] = 1
    elif change == "missing_part":
        fixture.setup["record"]["files"].pop("print/cap.stl")
    else:
        fixture.guide["instruction_binding"]["status"] = "legacy_unverified"
    report = native_delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert "readback_verified" not in report
    assert fixture.captures == []


def test_print_reader_corruption_cannot_become_delivery(monkeypatch):
    fixture = policy_capture(monkeypatch)

    def corrupt(*args):
        raise WorkspaceError("Print package member differs from its receipt")

    monkeypatch.setattr(print_setup_service, "print_setup_context", corrupt)
    report = native_delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert "differs from its receipt" in report["reason"]
    assert fixture.captures == []


@pytest.mark.parametrize(
    "fault", ["stl_disabled", "missing_part", "wrong_quantity", "missing_mesh"]
)
def test_a_captured_package_without_complete_stl_coverage_is_not_delivery(monkeypatch, fault):
    fixture = policy_capture(monkeypatch)
    if fault == "stl_disabled":
        fixture.receipt["include_stl"] = False
    elif fault == "missing_part":
        fixture.receipt["made_parts"].pop()
    elif fault == "wrong_quantity":
        fixture.receipt["made_parts"][1]["quantity"] = 1
    else:
        fixture.receipt["made_parts"][0]["files"].pop()
    report = native_delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert report["receipt"] is not None
    assert "readback_verified" not in report


def test_changed_prepared_package_retains_capture_without_current_delivery(monkeypatch):
    fixture = policy_capture(monkeypatch)
    read = native_delivery.read_project_export

    def changed(*args):
        result = read(*args)
        fixture.setup["record_hash"] = "b" * 64
        return result

    monkeypatch.setattr(native_delivery, "read_project_export", changed)
    report = native_delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert report["receipt"] is not None
    assert "readback_verified" not in report


def test_cad_only_does_not_require_print_but_retains_current_bound_guide(monkeypatch):
    fixture = policy_capture(monkeypatch, printable=False)
    fixture.revision["document"] = {
        "name": "CAD only",
        "features": [],
        "production": {"role": "make", "process": "cnc"},
    }
    report = native_delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "delivered"
    assert fixture.captures[0][1]["include_stl"] is False
    assert fixture.setup_reads == []


def test_cad_only_customer_keeps_explicit_declared_mesh_in_automatic_capture(monkeypatch):
    fixture = policy_capture(monkeypatch, printable=False)
    report = native_delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "delivered"
    assert fixture.captures[0][1]["include_stl"] is True
    assert fixture.setup_reads == []
