"""Captured guidance consistency controls; fixture bytes are not CAD qualification."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from zipfile import ZipFile

import pytest

from forge_mind.workspace import delivery_guidance as guidance
from forge_mind.workspace import project_delivery as delivery
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.handoff import json_bytes, verify_manufacturing_handoff
from forge_mind.workspace.production import manufacture_bundle
from tests.unit.workspace.test_project_delivery import RID, saved_project


def project(tmp_path, monkeypatch, *, current=False):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(guidance, "engine_fingerprint", lambda: "fixture-export-engine")
    # Existing export fixtures deliberately report a different measured engine. A
    # reviewed fixture uses a matched pair; this does not establish actual geometry.
    if current:
        with service.store.connect() as con:
            row = con.execute(
                "SELECT document FROM revisions WHERE session=? AND id=?", (sid, RID)
            ).fetchone()
            revision = json.loads(row[0])
            revision["measurement"]["engine_fingerprint"] = "fixture-export-engine"
            con.execute(
                "UPDATE revisions SET document=? WHERE session=? AND id=?",
                (json.dumps(revision), sid, RID),
            )
        context = guidance.guidance_context(service, sid, "owner", RID)
        guidance.update_guidance(
            service,
            sid,
            "owner",
            guidance.GuidanceUpdate.model_validate(
                {
                    "revision_id": RID,
                    "basis_hash": context["basis_hash"],
                    "reviewed_current_facts": True,
                    "guidance": {
                        "print": "Load the separate made-part STL files. Test print settings first.",
                        "installation": "Check the actual mounting surface and hardware before installation.",
                        "use_and_removal": "Check physical fit before use. Remove without excessive force.",
                        "unresolved": "Physical fit, anchors, load capacity and printer accuracy remain unverified.",
                    },
                }
            ),
        )
    return service, sid


def export(service, sid, tmp_path):
    receipt = delivery.create_project_export(service, sid, "owner", RID)
    data, _ = delivery.read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    path = tmp_path / "received.zip"
    path.write_bytes(data)
    return path, receipt


@pytest.mark.parametrize("current", [False, True])
def test_owned_export_records_guidance_reference_state(tmp_path, monkeypatch, current):
    service, sid = project(tmp_path, monkeypatch, current=current)
    reviewed = guidance.guidance_context(service, sid, "owner", RID)
    path, receipt = export(service, sid, tmp_path)
    expected = "current" if current else "unbound"
    assert receipt["guidance_status"] == expected
    assert verify_manufacturing_handoff(path)["guidance_status"] == expected
    with ZipFile(path) as archive:
        sidecar = json.loads(archive.read("customer-guidance.json"))
        readme = archive.read("README.md").decode()
        assert sidecar["status"] == expected
        assert sidecar["basis"]["revision_id"] == RID
        assert sidecar["facts"] == reviewed["facts"]
        assert sidecar["basis"] == reviewed["basis"]
        assert sidecar["basis_hash"] == reviewed["basis_hash"]
        assert "[customer-guidance.json](customer-guidance.json)" in readme
        if current:
            # Fresh prose remains literal unverified history. The sidecar keeps
            # its exact content; only compiled reference facts become current steps.
            assert "Instruction reference state: current." in readme
            assert "## Print, install and use" in readme
            assert "Instruction binding status: legacy_unverified." in readme
            assert sidecar["record"]["guidance"] == reviewed["record"]["guidance"]
            for instructions in reviewed["record"]["guidance"].values():
                assert instructions not in readme
            assert "## Unverified author notes and history" in readme
            assert "Physical fit, printing and load capacity are not verified." in readme
        else:
            assert "Current saved CAD facts" in readme
            assert "not measured dimensions" in readme
    assert receipt["physical_readiness"] == "unverified"


@pytest.mark.parametrize("explicit_empty", [False, True])
def test_missing_or_null_acceptance_is_not_inferred_in_guidance_and_export(
    tmp_path, monkeypatch, explicit_empty
):
    service, sid = project(tmp_path, monkeypatch, current=True)
    if not explicit_empty:
        session = service.store.session_record(sid, "owner")
        session.pop("accepted_revision", None)
        with service.store.connect() as con:
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(session), sid))
    path, receipt = export(service, sid, tmp_path)
    assert receipt["guidance_status"] == "current"
    assert receipt["context"]["accepted_revision"] is None
    assert receipt["context"]["matches_accepted_revision"] is False
    assert receipt["context"]["matches_current_design"] is True
    assert receipt["context"]["selected_revision"] == RID
    with ZipFile(path) as archive:
        sidecar = json.loads(archive.read("customer-guidance.json"))
        assert sidecar["accepted_revision"] is None
    assert verify_manufacturing_handoff(path)["guidance_status"] == "current"


def test_export_of_previous_revision_marks_guidance_stale(tmp_path, monkeypatch):
    service, sid = project(tmp_path, monkeypatch, current=True)
    service.store.update(sid, selected_revision="c" * 32)
    path, receipt = export(service, sid, tmp_path)
    assert receipt["guidance_status"] == "stale"
    with ZipFile(path) as archive:
        readme = archive.read("README.md").decode()
        assert "Instruction reference state: stale." in readme
        assert "## Unverified author notes and history" in readme
        assert "Guidance acknowledged against the current saved facts" not in readme


def test_bound_generated_metadata_steps_match_the_final_archive_members(tmp_path, monkeypatch):
    service, sid = project(tmp_path, monkeypatch, current=True)
    reviewed = guidance.guidance_context(service, sid, "owner", RID)
    refs = [
        *reviewed["instruction_catalog"]["required_refs"],
        "project:README.md",
        "project:manifest.json",
    ]
    guidance.update_guidance(
        service,
        sid,
        "owner",
        guidance.GuidanceUpdate.model_validate(
            {
                "revision_id": RID,
                "basis_hash": reviewed["basis_hash"],
                "reviewed_current_facts": True,
                "guidance": reviewed["record"]["guidance"],
                "instruction_plan": {
                    "version": 1,
                    "fact_refs": refs,
                    "steps": [
                        {"action": "open_project_file", "target": "project:README.md"},
                        {"action": "open_project_file", "target": "project:manifest.json"},
                    ],
                },
            }
        ),
    )
    path, receipt = export(service, sid, tmp_path)
    assert receipt["guidance_status"] == "current"
    assert verify_manufacturing_handoff(path)["guidance_status"] == "current"
    with ZipFile(path) as archive:
        readme = archive.read("README.md").decode()
        for name in ("README.md", "manifest.json"):
            assert name in archive.namelist()
            assert f"Project file {name}: included in this download" in readme
            assert f"Open {name} from this project download." in readme
            assert f"Do not use {name}: it is omitted" not in readme


def rewrite(path, mutate):
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    mutate(members, manifest)
    manifest["files"] = {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        for name, data in members.items()
        if name != "manifest.json"
    }
    members["manifest.json"] = json_bytes(manifest)
    with ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


@pytest.mark.parametrize(
    "change",
    ["prose", "facts", "revision", "document", "step", "missing", "legacy", "false_current"],
)
def test_rehashed_guidance_substitutions_are_rejected(tmp_path, monkeypatch, change):
    service, sid = project(tmp_path, monkeypatch, current=change != "false_current")
    path, _ = export(service, sid, tmp_path)

    def mutate(members, manifest):
        sidecar = json.loads(members["customer-guidance.json"])
        if change == "missing":
            members.pop("customer-guidance.json")
            return
        if change == "prose":
            sidecar["record"]["guidance"]["print"] = "A substituted guide."
            sidecar["record"]["guidance_hash"] = digest(sidecar["record"]["guidance"])
        elif change == "facts":
            sidecar["facts"]["measured"]["solid_count"] = 999
            sidecar["basis"]["facts_hash"] = digest(sidecar["facts"])
        elif change == "revision":
            sidecar["basis"]["revision_id"] = "d" * 32
        elif change == "document":
            sidecar["basis"]["document_hash"] = "0" * 64
        elif change == "step":
            sidecar["basis"]["artifact_hashes"]["part.step"] = "0" * 64
        elif change == "legacy":
            sidecar["legacy_guidance_status"] = "current"
        else:
            sidecar["status"] = "current"
        sidecar["basis_hash"] = digest(sidecar["basis"])
        members["customer-guidance.json"] = json_bytes(sidecar)
        # Deliberately repair member hashes only. Captured context remains authoritative.

    rewrite(path, mutate)
    with pytest.raises(ValueError, match="guidance"):
        verify_manufacturing_handoff(path)


@pytest.mark.parametrize("change", ["revision", "step", "document", "false_current", "old_current"])
def test_self_consistent_sidecar_cannot_contradict_saved_context(tmp_path, monkeypatch, change):
    service, sid = project(tmp_path, monkeypatch, current=change != "false_current")
    path, _ = export(service, sid, tmp_path)

    def mutate(members, manifest):
        sidecar = json.loads(members["customer-guidance.json"])
        if change == "revision":
            sidecar["basis"]["revision_id"] = "d" * 32
        elif change == "step":
            sidecar["basis"]["artifact_hashes"]["part.step"] = "0" * 64
        elif change == "document":
            sidecar["basis"]["document_hash"] = "0" * 64
        elif change == "old_current":
            manifest["context"]["selected_revision"] = "c" * 32
            manifest["context"]["matches_current_design"] = False
        else:
            sidecar["status"] = "current"
        sidecar["basis_hash"] = digest(sidecar["basis"])
        if sidecar["record"]:
            sidecar["record"]["basis"] = deepcopy(sidecar["basis"])
            sidecar["record"]["basis_hash"] = sidecar["basis_hash"]
        manifest["context"]["guidance_basis"] = deepcopy(sidecar["basis"])
        manifest["context"]["guidance_basis_hash"] = sidecar["basis_hash"]
        manifest["context"]["guidance_context_hash"] = digest(sidecar)
        members["customer-guidance.json"] = json_bytes(sidecar)

    rewrite(path, mutate)
    with pytest.raises(ValueError, match="guidance"):
        verify_manufacturing_handoff(path)


def test_direct_legacy_bundle_labels_prose_unbound_without_new_sidecar(tmp_path):
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "production": {"role": "make"},
            "delivery_guidance": {"print": "An earlier printing declaration."},
        }
    )
    step = b"Saved STEP fixture; no geometry qualification."
    (tmp_path / "part.step").write_bytes(step)
    path = manufacture_bundle(
        document,
        tmp_path,
        tmp_path / "legacy.zip",
        {"part.step": hashlib.sha256(step).hexdigest()},
        include_assembly_step=True,
    )
    with ZipFile(path) as archive:
        assert "customer-guidance.json" not in archive.namelist()
        assert (
            "Legacy customer guidance: unbound reference state"
            in archive.read("README.md").decode()
        )
    assert "guidance_status" not in verify_manufacturing_handoff(path)
