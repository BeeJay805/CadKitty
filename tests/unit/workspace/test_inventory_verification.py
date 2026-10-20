"""Saved-byte inventory and finite scope proof, without native/model execution."""

from __future__ import annotations

import hashlib
import io
import json
import time
from copy import deepcopy
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_mesh_handoff import document_data
from tests.unit.workspace.test_project_delivery import RID, saved_project

from forge_mind.workspace import completion, delivery_guidance, measure, native_finish
from forge_mind.workspace import requirement_verification as verification
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.models import Contract, DesignPlan, Finding, Requirement
from forge_mind.workspace.native_delivery import current_delivery_binding
from forge_mind.workspace.nominal_delivery_scope import evaluate_delivery_scope
from forge_mind.workspace.nominal_inventory import evaluate_inventory
from forge_mind.workspace.project_delivery import create_project_export, read_project_export
from forge_mind.workspace.store import WorkspaceError


def requirement(kind, **fields):
    criterion = {"version": 1, "kind": kind, **fields}
    quote = json.dumps(criterion)
    return Requirement(
        id="inventory" if kind == "nominal_subject_inventory" else "scope",
        text="Nominal CAD inventory" if kind == "nominal_subject_inventory" else "Digital scope",
        source="explicit",
        source_quote=quote,
        verification={**criterion, "source_quote": quote},
    )


def inventory(cap_quantity=2):
    return requirement(
        "nominal_subject_inventory",
        scope="nominal_cad",
        coverage="all_made",
        items=[
            {"subject_id": name, "subject_role": "designed", "required_name": name, "quantity": n}
            for name, n in (("base", 1), ("cap", cap_quantity))
        ],
    )


def scope():
    return requirement(
        "nominal_delivery_scope", scope="software_policy", policy="nominal_cad_and_digital_only"
    )


def plan():
    return DesignPlan(
        objective="Exact source inventory",
        verification_bindings=[
            {"subject_id": name, "part_id": name, "frame": "definition_local"}
            for name in ("base", "cap")
        ],
    )


def prepared_inventory(tmp_path, monkeypatch, *, physical=False):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    target = Contract(requirements=[inventory(), scope()])
    if physical:
        target.requirements.append(
            Requirement(id="physical", text="Hold the motor", source_quote="Hold the motor")
        )
    revision = service.store.revision(sid, RID)
    revision["contract"] = target.model_dump(mode="json")
    revision["measurement"].update(
        contract_hash=target.fingerprint(),
        engine_fingerprint="fixture-export-engine",
        findings=[
            Finding(
                requirement_id=r.id,
                status="unverified",
                method="unsupported",
                reason="Build-time unknown",
            ).model_dump()
            for r in target.requirements
        ],
    )
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), sid, RID),
        )
    frozen_plan = plan().model_dump(mode="json")
    service.store.update(
        sid,
        contract=target.model_dump(mode="json"),
        design_plan=frozen_plan,
        plan_hash=digest(frozen_plan),
        plan_version=1,
    )
    monkeypatch.setattr(measure, "engine_fingerprint", lambda: "fixture-export-engine")
    monkeypatch.setattr(completion, "engine_fingerprint", lambda: "fixture-export-engine")
    jid = service.store.enqueue(sid, "owner", {"max_seconds": 120})
    assert service.store.claim()["id"] == jid
    review = {
        "status": "ready_for_validation",
        "job_id": jid,
        "revision_id": RID,
        "plan_hash": digest(frozen_plan),
        "basis_hash": "b" * 64,
        "reasons": [],
    }
    service.store.update(sid, native_review=review)
    receipt = create_project_export(service, sid, "owner", RID, include_stl=True)
    content, readback = read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert readback == receipt
    delivery = {
        **current_delivery_binding(service, sid, "owner"),
        "job_id": jid,
        "status": "delivered",
        "readback_verified": True,
        "receipt": receipt,
    }
    service.store.update(sid, automatic_delivery=delivery)
    guide = {"status": "current", "basis_hash": review["basis_hash"], "record": {}}
    monkeypatch.setattr(delivery_guidance, "guidance_context", lambda *a, **k: deepcopy(guide))
    monkeypatch.setattr(native_finish, "finish_context", lambda *a, **k: deepcopy(review))
    return (
        service,
        sid,
        jid,
        target,
        delivery,
        content,
        DesignDocument.model_validate(revision["document"]),
    )


def rewrite(content, delivery, change):
    with ZipFile(io.BytesIO(content)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    change(files)
    manifest = json.loads(files["manifest.json"])
    manifest["files"] = {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        for name, data in files.items()
        if name != "manifest.json"
    }
    files["manifest.json"] = json.dumps(manifest).encode()
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    content = output.getvalue()
    delivery = deepcopy(delivery)
    delivery["receipt"].update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    return content, delivery


def test_current_inventory_waits_for_capture_and_does_not_use_solid_count():
    document = DesignDocument.model_validate(document_data())
    result = evaluate_inventory(inventory(), plan=plan(), document=document)
    assert result["status"] == "unverified"
    assert result["facts"]["current_design_status"] == "pass"
    assert result["facts"]["actual_made_quantities"] == {"base": 1, "cap": 2}
    assert result["facts"]["stage"] == "current_design_before_capture"


def test_reused_subassembly_inventory_counts_every_installed_copy():
    raw = document_data()
    raw["assembly"]["subassemblies"] = [
        {
            "id": "pair",
            "name": "Pair",
            "children": [{"id": "left", "definition": "cap"}, {"id": "right", "definition": "cap"}],
        }
    ]
    raw["assembly"]["roots"] = [
        {"id": "base", "definition": "base"},
        {"id": "pair1", "definition": "pair"},
        {"id": "pair2", "definition": "pair"},
    ]
    result = evaluate_inventory(
        inventory(cap_quantity=4), plan=plan(), document=DesignDocument.model_validate(raw)
    )
    assert result["facts"]["actual_made_quantities"] == {"base": 1, "cap": 4}
    assert result["facts"]["current_design_status"] == "pass"


@pytest.mark.parametrize("change", ["wrong_name", "extra_copy", "unused_make"])
def test_wrong_name_quantity_and_extra_unused_make_cannot_pass(change):
    raw = document_data()
    if change == "wrong_name":
        raw["assembly"]["parts"][0]["name"] = "another name"
    elif change == "extra_copy":
        raw["assembly"]["roots"].append({"id": "extra_copy", "definition": "base"})
    else:
        extra = deepcopy(raw["assembly"]["parts"][0])
        extra.update(id="unused", name="unused")
        raw["assembly"]["parts"].append(extra)
    result = evaluate_inventory(
        inventory(), plan=plan(), document=DesignDocument.model_validate(raw)
    )
    assert result["status"] == "fail"


@pytest.mark.parametrize("change", ["bought", "reference", "imported", "unused", "no_binding"])
def test_unproved_or_wrong_role_binding_cannot_substitute(change):
    raw, binding = document_data(), plan()
    if change in {"bought", "reference"}:
        raw["assembly"]["parts"][0]["production"] = {
            "role": "buy" if change == "bought" else "reference"
        }
    elif change == "imported":
        raw["assembly"]["parts"][0]["features"] = [
            {"id": "body", "name": "Imported", "kind": "import", "asset_id": "a" * 32}
        ]
    elif change == "unused":
        raw["assembly"]["roots"] = raw["assembly"]["roots"][1:]
    else:
        binding.verification_bindings = []
    with pytest.raises(ValueError):
        evaluate_inventory(inventory(), plan=binding, document=DesignDocument.model_validate(raw))


def test_final_capture_and_actual_package_inventory_required(tmp_path, monkeypatch):
    _, _, _, _, delivery, content, document = prepared_inventory(tmp_path, monkeypatch)
    result = evaluate_inventory(
        inventory(),
        plan=plan(),
        document=document,
        captured_delivery=delivery,
        package_content=content,
        final=True,
    )
    assert result["status"] == "pass"
    assert result["facts"]["stage"] == "final_captured_inventory"
    with pytest.raises(ValueError, match="readback"):
        evaluate_inventory(inventory(), plan=plan(), document=document, final=True)

    def wrong_quantities(files):
        production = json.loads(files["production.json"])
        production["make"][0]["quantity"] = 5
        files["production.json"] = json.dumps(production).encode()

    wrong_content, wrong_delivery = rewrite(content, delivery, wrong_quantities)
    assert (
        evaluate_inventory(
            inventory(),
            plan=plan(),
            document=document,
            captured_delivery=wrong_delivery,
            package_content=wrong_content,
            final=True,
        )["status"]
        == "fail"
    )
    wrong_delivery = deepcopy(delivery)
    wrong_delivery["receipt"]["made_parts"][0]["quantity"] = 5
    assert (
        evaluate_inventory(
            inventory(),
            plan=plan(),
            document=document,
            captured_delivery=wrong_delivery,
            package_content=content,
            final=True,
        )["status"]
        == "fail"
    )


def test_scope_uses_captured_protocol_not_requested_policy_alone(tmp_path, monkeypatch):
    _, _, _, _, delivery, content, document = prepared_inventory(tmp_path, monkeypatch)
    assert evaluate_delivery_scope(scope(), document=document)["status"] == "unverified"
    result = evaluate_delivery_scope(
        scope(), document=document, captured_delivery=delivery, package_content=content, final=True
    )
    assert result["status"] == "pass"
    assert "arbitrary authored prose" in result["facts"]["scope"]
    for field, value in (("manufacturing_ready", True), ("physical_readiness", "verified")):
        wrong = deepcopy(delivery)
        wrong["receipt"][field] = value
        assert (
            evaluate_delivery_scope(
                scope(),
                document=document,
                captured_delivery=wrong,
                package_content=content,
                final=True,
            )["status"]
            == "unverified"
        )

    def unknown_protocol(files):
        manifest = json.loads(files["manifest.json"])
        manifest["bundle_kind"] = "unknown_machine_protocol"
        files["manifest.json"] = json.dumps(manifest).encode()

    unknown_content, unknown_delivery = rewrite(content, delivery, unknown_protocol)
    assert (
        evaluate_delivery_scope(
            scope(),
            document=document,
            captured_delivery=unknown_delivery,
            package_content=unknown_content,
            final=True,
        )["status"]
        == "unverified"
    )


def test_optional_print_scope_requires_exact_known_unsliced_protocol(tmp_path, monkeypatch):
    _, _, _, _, delivery, content, document = prepared_inventory(tmp_path, monkeypatch)
    # Controlled protocol observations exercise the finite policy predicate.
    # Existing print reader tests establish actual prepared ZIP byte authority.
    context = {
        "status": "current",
        "readback_verified": True,
        "record": {
            "package": {
                "schema_version": 1,
                "status": "prepared",
                "revision_id": RID,
                "manufacturing_ready": False,
                "physical_readiness": "unverified",
                "basis": {"algorithm": "measured-planar-mesh-placement-v1"},
            },
            "result": {
                "algorithm": "measured-planar-mesh-placement-v1",
                "operation": "prepare",
                "manufacturing_ready": False,
                "physical_readiness": "unverified",
            },
        },
    }
    assert (
        evaluate_delivery_scope(
            scope(),
            document=document,
            captured_delivery=delivery,
            package_content=content,
            print_context=context,
            final=True,
        )["status"]
        == "pass"
    )
    for field, value in (("algorithm", "unknown_protocol"), ("physical_readiness", "verified")):
        wrong = deepcopy(context)
        wrong["record"]["result"][field] = value
        assert (
            evaluate_delivery_scope(
                scope(),
                document=document,
                captured_delivery=delivery,
                package_content=content,
                print_context=wrong,
                final=True,
            )["status"]
            == "unverified"
        )


@pytest.mark.parametrize("physical", [False, True])
def test_same_author_final_dispatch_and_atomic_critical_aggregation(
    tmp_path, monkeypatch, physical
):
    service, sid, jid, _, _, _, _ = prepared_inventory(tmp_path, monkeypatch, physical=physical)
    old_revision = deepcopy(service.store.revision(sid, RID))
    before = deepcopy(service.store.session_record(sid, "owner"))
    author = verification.inspect_requirements(
        service, sid, "owner", RID, deadline=time.monotonic() + 30, cancelled=lambda: False
    )
    assert [r["status"] for r in author["results"]] == ["unverified"] * (3 if physical else 2)
    assert service.store.session_record(sid, "owner") == before
    receipt = verification.verify_contract(
        service, sid, "owner", jid, deadline=time.monotonic() + 30, cancelled=lambda: False
    )
    service.store.finish(jid, sid, "needs_review", verification_report=receipt)
    final = verification.read_contract_verification(service, sid, "owner")
    assert [r["method"] for r in final["results"][:2]] == [
        "nominal_subject_inventory_v1",
        "nominal_delivery_scope_v1",
    ]
    assert [r["status"] for r in final["results"][:2]] == ["pass", "pass"]
    assert final["requirements_satisfied"] is not physical
    assert service.store.get(sid, "owner")["state"] == (
        "needs_review" if physical else "contract_satisfied"
    )
    assert service.store.revision(sid, RID) == old_revision
    assert service.store.get(sid, "owner")["accepted_revision"] is None


def test_new_policy_proof_keeps_cancellation_and_stale_publication_guards(tmp_path, monkeypatch):
    service, sid, jid, _, _, _, _ = prepared_inventory(tmp_path, monkeypatch)
    with pytest.raises(WorkspaceError, match="cancelled"):
        verification.inspect_requirements(
            service, sid, "owner", RID, deadline=time.monotonic() + 30, cancelled=lambda: True
        )
    receipt = verification.verify_contract(
        service, sid, "owner", jid, deadline=time.monotonic() + 30, cancelled=lambda: False
    )
    service.store.update(sid, plan_version=2)
    service.store.finish(jid, sid, "needs_review", verification_report=receipt)
    assert service.store.get(sid, "owner")["state"] == "needs_review"
    assert service.store.get(sid, "owner").get("contract_verification_report") is None
