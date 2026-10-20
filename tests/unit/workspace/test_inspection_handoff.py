"""Optional inspection handoff remains useful and honest when evidence is unavailable."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_handoff import package as base_package
from tests.unit.workspace.test_handoff import rewrite

from forge_mind.workspace.completion import _hash
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.engineering_service import inspection_handoff
from forge_mind.workspace.handoff import verify_manufacturing_handoff
from forge_mind.workspace.production import manufacture_bundle


def package(tmp_path, included=True):
    path = base_package.__wrapped__(tmp_path)
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        document = DesignDocument.model_validate_json(archive.read("document.json"))
        sources = json.loads(archive.read("sources.json"))
        evidence = json.loads(archive.read("engineering-evidence.json"))
        inputs = json.loads(archive.read("engineering-inputs.json"))
    plan = {"objective": "Inspection proposal", "inspection": [], "subsystems": []}
    context = manifest["context"]
    context["inspection_plan_hash"] = digest(plan)
    status = {
        "status": "included" if included else "unavailable",
        "reason": "Controlled fixture",
        "physical_readiness": "unverified",
    }
    artifacts = {}
    if included:
        status["report_id"] = "c" * 32
        report = {
            "id": status["report_id"],
            "inspection_plan": {
                "plan_hash": _hash(plan),
                "revision_id": context["revision_id"],
                "engine_fingerprint": context["export_engine_fingerprint"],
                "source_fingerprint": context["source_fingerprint"],
                "input_version": inputs["version"],
                "rows": [],
                "receipts": {},
            },
        }
        artifacts = {
            "report.json": json.dumps(report).encode(),
            "drawing.svg": b"<svg/>",
            "inspection.csv": b"CAD measured,Physical measured\n,\n",
        }
    return manufacture_bundle(
        document,
        tmp_path,
        path,
        {"part.step": hashlib.sha256((tmp_path / "part.step").read_bytes()).hexdigest()},
        sources=sources,
        engineering_evidence=evidence,
        revision_context=context,
        engineering_inputs=inputs,
        inspection_bundle={"plan": plan, "status": status, "artifacts": artifacts},
    )


@pytest.mark.parametrize("included", [True, False])
def test_optional_inspection_bundle_roundtrips_with_full_intent_and_explicit_status(
    tmp_path, included
):
    path = package(tmp_path, included)
    result = verify_manufacturing_handoff(path)
    assert result["manufacturing_ready"] is False
    with ZipFile(path) as archive:
        assert "inspection-intent.json" in archive.namelist()
        assert ("engineering/drawing.svg" in archive.namelist()) is included
        assert json.loads(archive.read("inspection-status.json"))["status"] == (
            "included" if included else "unavailable"
        )


@pytest.mark.parametrize(
    "member",
    [
        "inspection-intent.json",
        "inspection-status.json",
        "engineering/report.json",
        "engineering/drawing.svg",
        "engineering/inspection.csv",
    ],
)
def test_each_added_member_is_manifest_bound(tmp_path, member):
    path = package(tmp_path)
    rewrite(path, lambda members: members.__setitem__(member, b"changed"))
    with pytest.raises(ValueError, match="member changed"):
        verify_manufacturing_handoff(path)


@pytest.mark.parametrize(
    "change",
    [
        "intent",
        "report_binding",
        "false_status",
        "missing_drawing",
        "missing_status",
        "missing_both",
    ],
)
def test_rehashed_contradictory_inspection_bundle_rejected(tmp_path, change):
    path = package(tmp_path)

    def mutate(members):
        if change == "intent":
            plan = json.loads(members["inspection-intent.json"])
            plan["objective"] = "Changed"
            members["inspection-intent.json"] = json.dumps(plan).encode()
        elif change == "report_binding":
            report = json.loads(members["engineering/report.json"])
            report["inspection_plan"]["revision_id"] = "d" * 32
            members["engineering/report.json"] = json.dumps(report).encode()
        elif change == "false_status":
            status = json.loads(members["inspection-status.json"])
            status["status"] = "unavailable"
            members["inspection-status.json"] = json.dumps(status).encode()
        elif change == "missing_drawing":
            members.pop("engineering/drawing.svg")
        else:
            members.pop("inspection-status.json")
            if change == "missing_both":
                members.pop("inspection-intent.json")
        manifest = json.loads(members["manifest.json"])
        manifest["files"] = {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for name, data in members.items()
            if name != "manifest.json"
        }
        members["manifest.json"] = json.dumps(manifest).encode()

    rewrite(path, mutate)
    with pytest.raises(ValueError, match=r"Inspection|inspection"):
        verify_manufacturing_handoff(path)


def test_missing_review_returns_explicit_unavailable_without_blocking_geometry():
    service = SimpleNamespace(
        store=SimpleNamespace(
            get=lambda *a: {
                "engineering": {"version": 0},
                "design_plan": {"objective": "Retain proposal"},
            }
        )
    )
    result = inspection_handoff(service, "s", "o", {"id": "r"})
    assert result["artifacts"] == {}
    assert result["status"]["status"] == "unavailable"
    assert result["plan"]["objective"] == "Retain proposal"


def test_route_rechecks_inspection_bundle_even_when_project_context_is_unchanged(
    tmp_path, monkeypatch
):
    from tests.unit.workspace.test_handoff import (
        test_route_preserves_historical_export_and_rejects_mixed_snapshot as run_route_control,
    )

    calls = 0

    def inspection(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "plan": None,
            "artifacts": {},
            "status": {
                "status": "unavailable",
                "reason": "No report" if calls == 1 else "Report evidence changed",
                "physical_readiness": "unverified",
            },
        }

    monkeypatch.setattr("forge_mind.workspace.project_delivery.inspection_handoff", inspection)
    run_route_control(tmp_path, monkeypatch, "inspection")
    assert calls == 2


@pytest.mark.parametrize("tamper", [False, True])
def test_stale_or_changed_report_is_unavailable_with_no_stale_files(tmp_path, monkeypatch, tamper):
    import forge_mind.workspace.engineering_service as module

    paths = {}
    for name, data in {
        "report.json": b'{"inspection_plan":{"old":true}}',
        "drawing.svg": b"svg",
        "inspection.csv": b"csv",
    }.items():
        paths[name] = tmp_path / name
        paths[name].write_bytes(data)
    receipt = {
        "id": "report",
        "revision_id": "r",
        "input_version": 0,
        "engine_fingerprint": "engine",
        "hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
    }
    if tamper:
        paths["drawing.svg"].write_bytes(b"changed")
    monkeypatch.setattr(module, "engine_fingerprint", lambda: "engine")
    monkeypatch.setattr(module, "report_path", lambda *a: paths[a[-1]])
    monkeypatch.setattr(
        "forge_mind.workspace.inspection_plan.resolve_inspection_plan", lambda *a: {"new": True}
    )
    session = {"engineering": {"version": 0}, "reports": [receipt]}
    service = SimpleNamespace(store=SimpleNamespace(get=lambda *a: session))
    result = inspection_handoff(service, "s", "o", {"id": "r"})
    assert result["artifacts"] == {}
    assert result["status"]["status"] == "unavailable"
    assert "changed" in result["status"]["reason"] or "stale" in result["status"]["reason"]
