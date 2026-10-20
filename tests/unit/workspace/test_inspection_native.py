"""Real exported geometry, immutable probe, drawing and offline package readback."""

from __future__ import annotations

import copy
import hashlib
import json
from zipfile import ZipFile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.completion import _snapshot
from forge_mind.workspace.completion_inline import execute_prepared
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.engineering_service import release_package, run_report_job
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.handoff import verify_manufacturing_handoff
from forge_mind.workspace.models import Contract, DesignPlan, Requirement
from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.store import Store, WorkspaceError


def test_native_pair_receipt_populates_inspection_and_survives_package_verification(tmp_path):
    store = Store(tmp_path)
    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: None)
    contract = Contract(
        requirements=[
            Requirement(
                id="gap",
                text="80 mm gap and physical fit",
                source_quote="80 mm gap and physical fit",
            )
        ]
    )
    sid = store.create("owner", "Inspection control", contract.model_dump())["id"]
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "jaw",
                        "name": "Jaw",
                        "production": {"role": "make"},
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"x": 10, "y": 8, "z": 6},
                            }
                        ],
                        "output": "body",
                    }
                ],
                "roots": [
                    {"id": "left", "definition": "jaw", "position": [-5, 0, 0]},
                    {"id": "right", "definition": "jaw", "position": [85, 0, 0]},
                ],
            }
        }
    )
    revision = build_document(
        service, sid, "owner", document, contract, None, "Controlled inspection", 120, lambda: False
    )
    assert revision["measurement"]["geometry_valid"], revision["measurement"]["errors"]
    definition = {
        "id": "gap_check",
        "request": {
            "tool": "projected_pair",
            "units": "mm",
            "a": {"occurrence_path": "left"},
            "b": {"occurrence_path": "right"},
            "axis": [1, 0, 0],
            "datum": [0, 0, 0],
            "expected_separation_mm": 80,
            "tolerance_mm": 0.00001,
        },
    }
    plan = DesignPlan.model_validate(
        {
            "objective": "Propose inspection with fresh CAD measurements",
            "subsystems": [
                {
                    "id": "jaws",
                    "name": "Jaws",
                    "part_ids": ["jaw"],
                    "requirement_ids": ["gap"],
                    "checks": [definition],
                }
            ],
            "inspection": [
                {
                    "id": "jaw_gap",
                    "label": "Jaw opening",
                    "requirement_ids": ["gap"],
                    "occurrence_paths": ["left", "right"],
                    "characteristic": "Whole-body X support separation",
                    "datum": "Global X datum-axis support planes",
                    "nominal_mm": 80,
                    "minus_mm": 0.02,
                    "plus_mm": 0.04,
                    "basis": "Proposed machining allowance, not approved",
                    "physical_method": "Calibrated length gauge between locating surfaces",
                    "missing_evidence": ["Physical repeatability and process capability"],
                    "probe": {"check_id": "gap_check", "result": "ordered_separation_mm"},
                }
            ],
        }
    ).model_dump()
    store.update(sid, selected_revision=revision["id"], accepted_revision=None, design_plan=plan)
    frozen = copy.deepcopy(store.get(sid, "owner")["engineering"])
    receipt, reused = execute_prepared(
        service,
        sid,
        "owner",
        _snapshot(service, sid, "owner", revision["id"]),
        plan["subsystems"][0]["checks"][0],
        120,
        lambda: False,
    )
    assert not reused
    assert receipt["result"]["results"]["ordered_separation_mm"] == pytest.approx(80)
    run_report_job(
        service,
        {"session": sid, "owner": "owner", "revision_id": revision["id"], "max_seconds": 120},
        lambda: False,
    )
    report_id = store.get(sid, "owner")["reports"][-1]["id"]
    with ZipFile(release_package(service, sid, "owner", revision["id"])) as archive:
        report = json.loads(archive.read("engineering/report.json"))
        row = report["inspection_plan"]["rows"][0]
        assert row["cad"]["value_mm"] == pytest.approx(80)
        assert row["cad"]["receipt_id"] == receipt["id"]
        assert row["physical"]["status"] == "unverified"
        assert report["drawings"]["inspection_callouts"]
        assert report["inspection_plan"]["receipts"][receipt["id"]] == json.loads(
            json.dumps(receipt)
        )
        manifest = json.loads(archive.read("manifest.json"))
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    assert store.get(sid, "owner")["engineering"] == frozen
    assert store.get(sid, "owner")["accepted_revision"] is None
    assert store.revision(sid, revision["id"])["contract"] == contract.model_dump()
    assert all(f["status"] == "unverified" for f in revision["measurement"]["findings"])
    app = FastAPI()
    app.include_router(product_router(store, service, lambda: "owner"))
    client = TestClient(app)
    response = client.get(f"/workspace/sessions/{sid}/manufacturing/{revision['id']}")
    assert response.status_code == 200, response.text
    downloaded = tmp_path / "inspection-handoff.zip"
    downloaded.write_bytes(response.content)
    assert verify_manufacturing_handoff(downloaded)["manufacturing_ready"] is False
    with ZipFile(downloaded) as archive:
        assert {
            "engineering/report.json",
            "engineering/drawing.svg",
            "engineering/inspection.csv",
            "inspection-intent.json",
            "make/definition_jaw.step",
        } <= set(archive.namelist())
        assert json.loads(archive.read("inspection-status.json"))["status"] == "included"
        assert json.loads(archive.read("inspection-intent.json")) == json.loads(json.dumps(plan))
    changed = copy.deepcopy(plan)
    changed["inspection"][0]["physical_method"] = "Changed proposed method"
    store.update(sid, design_plan=changed)
    with pytest.raises(WorkspaceError, match="Inspection plan changed"):
        release_package(service, sid, "owner", revision["id"])
    response = client.get(f"/workspace/sessions/{sid}/manufacturing/{revision['id']}")
    assert response.status_code == 200, response.text
    downloaded.write_bytes(response.content)
    verify_manufacturing_handoff(downloaded)
    with ZipFile(downloaded) as archive:
        assert json.loads(archive.read("inspection-status.json"))["status"] == "unavailable"
        assert "engineering/drawing.svg" not in archive.namelist()
        assert "make/definition_jaw.step" in archive.namelist()
    store.update(sid, design_plan=plan)
    with store.connect() as con:
        con.execute("UPDATE product_checks SET document='{}' WHERE id=?", (receipt["id"],))
    with pytest.raises(WorkspaceError, match="Inspection evidence changed"):
        release_package(service, sid, "owner", revision["id"])
    assert store.get(sid, "owner")["reports"][-1]["id"] == report_id
