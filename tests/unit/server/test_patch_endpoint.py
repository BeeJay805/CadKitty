"""POST /patch (ADR-0037): contract projection, billing, and the generations record.

patch_part itself is covered in tests/unit/api/test_patch_part.py; here it is stubbed so
the route's projection (files rewrite, charge-only-on-ship, jsonb spec recoverability)
is proven money-free against the ACTUAL FastAPI app.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from forge_mind.api.core import GenerateResult, StandardLaneOutcome
from forge_mind.api.schema import (
    BillOfMaterials,
    Files,
    GenerateOk,
    GenerateRefused,
    Kit,
    MakeInstructions,
    PatchView,
    ValidationView,
    Vec3,
)
from forge_mind.server import app as server_app

_SPEC = {"pattern": "pi4_case", "name": "tray"}  # opaque to the stubbed route


def _kit() -> Kit:
    return Kit(
        part_name="p",
        bom=BillOfMaterials(lines=[], unmapped=[]),
        instructions=MakeInstructions(
            print_orientation=Vec3(x=0.0, y=0.0, z=1.0),
            orientation_note="n",
            material="PLA",
            recommended_min_wall_mm=None,
            layer_height_mm=0.2,
            wall_note="w",
            infill_note="i",
            insert_install_note=None,
            steps=["s"],
            advisory=True,
            caveat="c",
        ),
    )


def _ok_result(spec: dict[str, Any]) -> GenerateResult:
    return GenerateResult(
        status="ok",
        ok=GenerateOk(
            prompt="raise the standoffs to 6mm",
            gen_path="standard:patch",
            cost_usd=0.002,
            validation=ValidationView(passed=True, overall="pass", tiers=[], failed_checks=[]),
            kit=_kit(),
            files=Files(glb="part.glb", stl="part.stl"),
            fabrication_mode="print_3d",
            spec=spec,
            patch=PatchView(
                field_path="standoffs.height_mm", op="set", old_value=4.0, new_value=6.0
            ),
        ),
        lane=StandardLaneOutcome(
            attempted=True, outcome="shipped", gen_path="standard:patch", cost_usd=0.002
        ),
    )


@pytest.fixture()
def harness(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"records": [], "deductions": []}
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))
    monkeypatch.setattr(server_app, "_resolve_credit_gate", lambda tok, q: (False, 1))
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    monkeypatch.setattr(
        server_app, "_store_generation", lambda tok, rec: state["records"].append(rec)
    )

    def _deduct(tok: str, amount: int, reason: str, job: str) -> int:
        state["deductions"].append((amount, reason))
        return 7

    monkeypatch.setattr(server_app, "_deduct_credits", _deduct)
    monkeypatch.setattr(server_app, "_upload_artifact", lambda *a, **k: None)
    return state


def test_patch_ok_projects_contract_and_records(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    new_spec = {"pattern": "pi4_case", "name": "tray", "standoffs": {"height_mm": 6.0}}

    def _fake_patch(spec: dict[str, Any], edit: str, out: Any, **kw: Any) -> GenerateResult:
        (out / "part.glb").write_bytes(b"")
        (out / "part.stl").write_bytes(b"")
        (out / "part.step").write_bytes(b"")
        return _ok_result(new_spec)

    monkeypatch.setattr("forge_mind.api.patch.patch_part", _fake_patch)
    client = TestClient(server_app.app)
    r = client.post("/patch", json={"spec": _SPEC, "edit": "raise the standoffs to 6mm"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["gen_path"] == "standard:patch"
    # Files rewritten onto the served job path, step included.
    assert payload["files"]["glb"].startswith("files/") and payload["files"]["glb"].endswith(
        "part.glb"
    )
    assert payload["files"]["step"].endswith("part.step")
    # The NEW parametric model + the one-field diff ride the payload (the next edit's input).
    assert payload["spec"] == new_spec
    assert payload["patch"]["field_path"] == "standoffs.height_mm"
    assert payload["not_for_manufacture"] is False
    assert payload["credits_charged"] == 1
    assert payload["credits_remaining"] == 7
    assert harness["deductions"] == [(1, "patch_standard_layoutspec")]
    (rec,) = harness["records"]
    assert rec.engine == "layoutspec"
    assert rec.status == "ok"
    assert rec.spec["gen_path"] == "standard:patch"
    assert rec.spec["layout_spec"] == new_spec  # replayable from the row alone
    assert rec.spec["patch"]["new_value"] == 6.0


def test_patch_refusal_charges_nothing(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = GenerateResult(
        status="refused",
        refused=GenerateRefused(reason="Can't apply that edit: exceeds the 95mm hard max."),
        lane=StandardLaneOutcome(attempted=True, outcome="refused", gen_path="standard:patch"),
    )
    monkeypatch.setattr("forge_mind.api.patch.patch_part", lambda *a, **k: refused)
    client = TestClient(server_app.app)
    r = client.post("/patch", json={"spec": _SPEC, "edit": "add 30mm of clearance"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "refused"
    assert "hard max" in payload["reason"]
    assert harness["deductions"] == []  # a refusal is never charged
    (rec,) = harness["records"]
    assert rec.status == "rejected"
    assert rec.error_type == "refused"


def test_patch_requires_edit_and_spec(harness: dict[str, Any]) -> None:
    client = TestClient(server_app.app)
    assert client.post("/patch", json={"spec": _SPEC, "edit": "  "}).status_code == 400
    assert client.post("/patch", json={"spec": {}, "edit": "wider"}).status_code == 400
