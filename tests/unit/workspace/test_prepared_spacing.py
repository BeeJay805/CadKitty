"""Existing spacing probes remain exploratory when scheduled for completion."""

import copy
import hashlib
import json
import time
from types import SimpleNamespace

import build123d as b
import pytest

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.assembly import named_solids, occurrence_label
from forge_mind.workspace.completion import _snapshot, completion_context, finalize_completion
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, Requirement
from forge_mind.workspace.product_service import read_tool_receipt
from forge_mind.workspace.store import Store
from forge_mind.workspace.topology import signature

RID = "a" * 32


@pytest.fixture
def jaws(tmp_path, monkeypatch):
    store = Store(tmp_path)
    contract = Contract(
        requirements=[
            Requirement(
                id="jaw_spacing", text="Adjust jaw spacing", source_quote="Adjust jaw spacing"
            )
        ]
    ).model_dump()
    sid = store.create("owner", "Adjust jaw spacing", contract)["id"]
    directory = store.directory(sid, RID)
    directory.mkdir(parents=True)
    bodies = []
    for name, x in (("fixed_jaw", -50), ("moving_jaw", 50)):
        shape = b.Box(20, 12, 8).translate((x, 0, 0))
        shape.label = occurrence_label(name)
        bodies.append(shape)
    b.export_step(b.Compound(children=bodies), directory / "part.step")
    named = named_solids(b.import_step(directory / "part.step"))
    faces = []
    for name, normal in (("fixed_jaw", 1), ("moving_jaw", -1)):
        face = next(
            face
            for face in named[occurrence_label(name)].faces()
            if abs(face.normal_at().X - normal) < 1e-12
        )
        faces.append({"body": {"occurrence_path": name}, "face_signature": signature(face)})
    revision = {
        "id": RID,
        "contract": contract,
        "document": {"assembly": {"parts": [{"id": "fixed_jaw"}, {"id": "moving_jaw"}]}},
        "measurement": {
            "geometry_valid": True,
            "requirements_satisfied": False,
            "artifact_hashes": {"part.step": file_sha256(directory / "part.step")},
            "facts": {
                "assembly": {
                    "occurrences": [
                        {"path": name, "definition": name} for name in ("fixed_jaw", "moving_jaw")
                    ]
                }
            },
            "findings": [
                {
                    "requirement_id": "jaw_spacing",
                    "status": "unverified",
                    "method": "no applicable instrument",
                    "reason": "Frozen acceptance is separate",
                }
            ],
        },
    }
    store.add_revision(sid, revision)
    store.update(sid, selected_revision=RID)
    service = SimpleNamespace(
        store=store, _load_revision=store.revision, executor=Executor("trusted_local")
    )
    # Keep this integration bounded to native product probes and their real receipts.
    monkeypatch.setattr(
        "forge_mind.workspace.product_interfaces.validate_interfaces", lambda *args: []
    )
    monkeypatch.setattr(
        "forge_mind.workspace.engineering_service.run_report_job", lambda *args: None
    )
    return service, sid, faces


def request_for(tool, faces):
    request = {
        "tool": tool,
        "units": "mm",
        "datum": [0, 0, 0],
        "expected_separation_mm": 80,
        "expected_midpoint_mm": 0,
        "tolerance_mm": 1e-5,
    }
    if tool == "projected_pair":
        request.update(
            a={"occurrence_path": "fixed_jaw"}, b={"occurrence_path": "moving_jaw"}, axis=[1, 0, 0]
        )
    else:
        request.update(a=faces[0], b=faces[1])
    return request


def save_plan(service, sid, request, reference=None):
    # Roundtrip the public author action, not just the private tool adapter.
    action = Action.model_validate(
        {
            "kind": "plan",
            "explanation": "Prepare exploratory jaw measurement",
            "plan": {
                "objective": "Inspect jaw spacing",
                "subsystems": [
                    {
                        "id": "jaws",
                        "name": "Jaws",
                        "requirement_ids": ["jaw_spacing"],
                        "part_ids": ["fixed_jaw", "moving_jaw"],
                        "checks": [
                            {
                                "id": "spacing",
                                "request": request,
                                "reference_revision_id": reference,
                            }
                        ],
                    }
                ],
            },
        }
    )
    restored = Action.model_validate_json(action.model_dump_json())
    service.store.update(sid, design_plan=restored.plan.model_dump())
    return service.store.get(sid, "owner")["design_plan"]["subsystems"][0]["checks"][0]["request"]


@pytest.mark.parametrize("tool", ["projected_pair", "planar_face_pair"])
def test_native_prepared_spacing_receipt_preserves_frozen_acceptance(jaws, tool):
    service, sid, faces = jaws
    before = copy.deepcopy(service.store.revision(sid, RID))
    request = save_plan(service, sid, request_for(tool, faces))
    report = finalize_completion(
        service,
        {"session": sid, "owner": "owner"},
        time.monotonic() + 90,
        lambda: False,
        "budget_exhausted",
    )
    check = report["subsystems"][0]["checks"][0]
    snapshot = _snapshot(service, sid, "owner", RID)
    assert check["status"] == "pass", {
        "check": check,
        "changed": {
            key: (report.get(key), value)
            for key, value in snapshot.items()
            if report.get(key) != value
        },
    }
    receipt = read_tool_receipt(service, sid, "owner", check["receipt_id"])
    assert receipt["result"]["results"]["ordered_separation_mm"] == pytest.approx(80)
    assert receipt["result"]["qualification_status"] == "unverified"
    assert receipt["revision_id"] == RID
    assert receipt["request"] == request
    assert receipt["artifact_hashes"] == before["measurement"]["artifact_hashes"]
    assert (
        receipt["result"]["artifact_hashes"]["part.step"]
        == before["measurement"]["artifact_hashes"]["part.step"]
    )
    assert receipt["source_fingerprint"] == report["source_fingerprint"]
    assert receipt["engine_fingerprint"] == report["engine_fingerprint"]
    assert receipt["contract"] == before["contract"]
    assert not receipt["stale"]
    assert service.store.revision(sid, RID) == before
    assert service.store.get(sid, "owner")["contract"] == before["contract"]
    assert report["physical_readiness"] == "unverified"
    assert completion_context(service, sid, "owner", RID)["status"] == "checked_pass"
    # A receipt cannot remain current after its stored content is changed.
    with service.store.connect() as con:
        row = con.execute(
            "SELECT document,digest FROM product_checks WHERE id=?", (check["receipt_id"],)
        ).fetchone()
        assert hashlib.sha256(row[0].encode()).hexdigest() == row[1]
        con.execute("UPDATE product_checks SET document='{}' WHERE id=?", (check["receipt_id"],))
    assert completion_context(service, sid, "owner", RID)["stale"]


@pytest.mark.parametrize("tool", ["projected_pair", "planar_face_pair"])
def test_nested_numeric_spacing_selectors_require_final_revision(jaws, monkeypatch, tool):
    service, sid, faces = jaws
    request = request_for(tool, faces)
    if tool == "projected_pair":
        request["a"] = {"body_index": 0}
    else:
        request["a"] = {"body": {"body_index": 0}, "face_signature": faces[0]["face_signature"]}
    save_plan(service, sid, request, reference="b" * 32)
    monkeypatch.setattr(
        "forge_mind.workspace.product_service.run_tool",
        lambda *args: pytest.fail("stale numeric selection must not launch"),
    )
    report = finalize_completion(
        service,
        {"session": sid, "owner": "owner"},
        time.monotonic() + 30,
        lambda: False,
        "budget_exhausted",
    )
    check = report["subsystems"][0]["checks"][0]
    assert check["status"] == "unverified"
    assert "exact final reference revision" in check["reason"]
    assert report["receipt_ids"] == []


def test_spacing_prepared_wire_schema_remains_untagged_and_closed():
    from forge_mind.llm.codex_client import _strict_schema
    from forge_mind.workspace.completion_models import PreparedCheck

    schema = _strict_schema(PreparedCheck.model_json_schema())

    def check(node):
        if isinstance(node, dict):
            assert "oneOf" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)
    text = json.dumps(schema)
    assert "projected_pair" in text and "planar_face_pair" in text
