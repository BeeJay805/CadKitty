"""Pending intent and package disclosure use saved bytes, never CAD or model jobs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_delivery_guidance import saved_assembly
from tests.unit.workspace.test_guidance_transport import instructions
from tests.unit.workspace.test_mesh_handoff import RID

from forge_mind.workspace import delivery_guidance, project_delivery
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.delivery_guidance import GuidanceUpdate, guidance_context, update_guidance
from forge_mind.workspace.edit_service import patch_plan
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.project_delivery import build_manufacturing_handoff
from forge_mind.workspace.store import Store, WorkspaceError

MISSING_INPUT = "Applied load and center of mass are missing."
UNSUPPORTED = "Holding capacity needs a compatible analysis."
PHYSICAL = "Measure slip under the stated load."


@pytest.fixture
def pending(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path / "store"), Executor("trusted_local"))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("No worker may start"))
    monkeypatch.setattr(delivery_guidance, "engine_fingerprint", lambda: "f" * 64)
    monkeypatch.setattr(project_delivery, "engine_fingerprint", lambda: "f" * 64)
    contract = Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "support",
                    "text": "The assembly supports the item.",
                    "source_quote": "supports the item",
                }
            ]
        }
    )
    sid = service.store.create(
        "owner",
        "The assembly supports the item.",
        contract.model_dump(),
        authoring_mode="native_operations",
    )["id"]
    source = service.store.directory(sid, RID)
    document, hashes = saved_assembly(source)
    for name, content in {
        "program.py": b"# Saved artifact fixture; never executed.\n",
        "part.stl": b"Saved assembly mesh fixture; never rendered.\n",
    }.items():
        (source / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    plan = {
        "objective": "Prepare a disclosed support prototype",
        "requirement_dispositions": [
            {"requirement_id": "support", "physical_validation": ["Verify physical support."]}
        ],
        "operation_plan": {
            "part_ids": [part.id for part in document.assembly.parts],
            "steps": [
                {
                    "id": part.id + "_body",
                    "part_id": part.id,
                    "feature_ids": ["body"],
                    "requirement_ids": ["support"],
                    "expected_effect": "Retain this saved part",
                }
                for part in document.assembly.parts
            ],
        },
    }
    binding = service.store.set_design_plan(sid, "owner", plan, expected_plan_version=0)
    facts = {
        "assembly": {
            "occurrences": [
                {
                    "path": occurrence.id,
                    "definition": occurrence.definition,
                    "body_index": index,
                    "volume_mm3": 1.0,
                }
                for index, occurrence in enumerate(document.assembly.roots)
            ],
            "ports": {},
            "mates": [],
        }
    }
    measurement = Measurement(
        revision_id=RID,
        contract_hash=contract.fingerprint(),
        engine_fingerprint="f" * 64,
        geometry_valid=True,
        artifact_hashes=hashes,
        facts=facts,
    )
    service.store.add_revision(
        sid,
        {
            "id": RID,
            "code": "# Saved artifact fixture; never executed.",
            "contract": contract.model_dump(),
            "document": document.model_dump(mode="json"),
            "measurement": measurement.model_dump(),
            "operation_record": {"plan_snapshot": binding, "plan_hash": binding["plan_hash"]},
        },
    )
    service.store.update(
        sid,
        selected_revision=None,
        accepted_revision=None,
        proposal={
            "revision_id": RID,
            "based_on": None,
            "eligible": False,
            "input_version": 0,
            "contract": contract.model_dump(),
            "operation_plan_snapshot": binding,
        },
    )
    context = guidance_context(service, sid, "owner", RID)
    old_update = GuidanceUpdate.model_validate(instructions(context))
    current = update_guidance(service, sid, "owner", old_update)
    assert current["status"] == "current"
    return service, sid, document, hashes, facts, binding, old_update


def add_duty(service, sid):
    interface = {
        "id": "support_joint",
        "requirement_ids": ["support"],
        "participant_paths": ["base", "missing_pin"],
        "installed_occurrence_paths": ["base", "missing_pin"],
        "required_features": [{"occurrence_path": "base", "feature_id": "shoulder"}],
        "port_ids": ["base.axis"],
        "mate_ids": ["pin_capture"],
        "duties": [
            {
                "id": "hold",
                "kind": "holding",
                "proof_scope": "physical_function",
                "missing_inputs": [MISSING_INPUT],
                "unsupported_checks": [UNSUPPORTED],
                "physical_validation": [PHYSICAL],
            }
        ],
    }
    patch = PlanPatch.model_validate(
        {
            "basis": service.store.edit_basis(sid, "owner").model_dump(),
            "functional_interfaces": [interface],
        }
    )
    return patch_plan(service, sid, "owner", patch)


def test_pending_plan_patch_stales_prior_guide_and_exposes_actual_missing_duties(pending):
    service, sid, _, _, _, binding, old_update = pending
    original = deepcopy(service.store.revision(sid, RID))
    result = add_duty(service, sid)
    assert result["plan_applied"] is True and result["cad_applied"] is False
    context = guidance_context(service, sid, "owner", RID)
    assert context["status"] == "stale"
    assert context["basis_hash"] != old_update.basis_hash
    assert context["captured_plan_hash"] == binding["plan_hash"]
    assert service.store.revision(sid, RID) == original
    projection = context["facts"]["functional_interfaces"]
    row = projection["rows"][0]
    assert row["status"] == "missing_construction"
    assert row["missing_participant_paths"] == ["missing_pin"]
    assert row["missing_installed_occurrence_paths"] == ["missing_pin"]
    assert row["missing_features"] == [{"occurrence_path": "base", "feature_id": "shoulder"}]
    assert row["missing_port_ids"] == ["base.axis"]
    assert row["missing_mate_ids"] == ["pin_capture"]
    external = projection["external_obligations"][0]
    assert external["missing_inputs"] == [MISSING_INPUT]
    assert external["unsupported_checks"] == [UNSUPPORTED]
    assert external["physical_validation"] == [PHYSICAL]
    assert context["physical_readiness"] == "unverified"


def test_old_guidance_acknowledgment_cannot_erase_new_pending_intent(pending):
    service, sid, _, _, _, _, old_update = pending
    add_duty(service, sid)
    before = deepcopy(service.store.session_record(sid, "owner"))
    with pytest.raises(WorkspaceError):
        update_guidance(service, sid, "owner", old_update)
    assert service.store.session_record(sid, "owner") == before


def test_package_readme_lists_missing_construction_and_explicit_external_information(
    pending, tmp_path
):
    service, sid, _, _, _, _, _ = pending
    add_duty(service, sid)
    context = guidance_context(service, sid, "owner", RID)
    context = update_guidance(
        service, sid, "owner", GuidanceUpdate.model_validate(instructions(context))
    )
    assert context["status"] == "current"
    archive = tmp_path / "disclosed-prototype.zip"
    build_manufacturing_handoff(
        service,
        sid,
        "owner",
        RID,
        archive,
        include_assembly_step=True,
    )
    with ZipFile(archive) as package:
        readme = package.read("README.md").decode()
        sidecar = json.loads(package.read("customer-guidance.json"))
        assert sidecar["facts"] == context["facts"]
        assert sidecar["basis"] == context["basis"]
        assert sidecar["basis_hash"] == context["basis_hash"]
        assert sidecar["record"]["guidance"] == context["record"]["guidance"]
        assert sidecar["instruction_binding"]["status"] == "legacy_unverified"
        assert sidecar["instruction_binding"]["steps"] == []
        assert (
            package.read("assembly/part.step")
            == (service.store.directory(sid, RID) / "part.step").read_bytes()
        )
        for value in (
            "support_joint",
            "missing_construction",
            "missing_pin",
            "shoulder",
            "base.axis",
            "pin_capture",
            MISSING_INPUT,
        ):
            assert value in readme
        assert "Missing installed parts or hardware" in readme
        assert "Physical fit, printing and load capacity are not verified." in readme
        assert PHYSICAL not in readme
        assert "Required physical tests:" not in readme
        assert UNSUPPORTED not in readme
        external = sidecar["facts"]["functional_interfaces"]["external_obligations"][0]
        assert external["missing_inputs"] == [MISSING_INPUT]
        assert external["unsupported_checks"] == [UNSUPPORTED]
        assert external["physical_validation"] == [PHYSICAL]
        assert "Instruction binding status: legacy_unverified." in readme
        assert "No bound customer steps are supplied." in readme
        assert "not part of the bound customer steps" in readme
        assert "[customer-guidance.json](customer-guidance.json)" in readme
    assert service.store.session_record(sid, "owner")["accepted_revision"] is None
    assert len(service.store.revisions(sid)) == 1
    with service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs WHERE session=?", (sid,)).fetchone()[0] == 0
