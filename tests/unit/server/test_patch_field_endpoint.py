"""POST /patch-field (ADR-0060 slice B): the deterministic structured-edit route. The route's
projection + the FREE billing (no LLM, no credit) are proven money-free against the real FastAPI app
with patch_part_structured stubbed (its geometry is covered in tests/unit/api/test_patch_field.py)."""

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


def _ok_result(spec: dict[str, Any]) -> GenerateResult:
    kit = Kit(
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
    return GenerateResult(
        status="ok",
        ok=GenerateOk(
            prompt="standoffs.height_mm set 6.0",
            gen_path="standard:patch",
            cost_usd=0.0,  # structured edit is free (no LLM)
            validation=ValidationView(passed=True, overall="pass", tiers=[], failed_checks=[]),
            kit=kit,
            files=Files(glb="part.glb", stl="part.stl"),
            fabrication_mode="print_3d",
            spec=spec,
            patch=PatchView(
                field_path="standoffs.height_mm", op="set", old_value=4.0, new_value=6.0
            ),
        ),
        lane=StandardLaneOutcome(
            attempted=True, outcome="shipped", gen_path="standard:patch", cost_usd=0.0
        ),
    )


@pytest.fixture()
def harness(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"records": [], "deductions": []}
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    monkeypatch.setattr(
        server_app, "_store_generation", lambda tok, rec: state["records"].append(rec)
    )
    monkeypatch.setattr(
        server_app,
        "_deduct_credits",
        lambda tok, amt, reason, job: state["deductions"].append((amt, reason)) or 7,
    )
    monkeypatch.setattr(server_app, "_upload_artifact", lambda *a, **k: None)
    return state


def test_patch_field_ok_is_free(harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    new_spec = {"pattern": "pi4_case", "name": "tray", "standoffs": {"height_mm": 6.0}}

    def _fake(spec: dict[str, Any], field: str, op: str, value: Any, out: Any) -> GenerateResult:
        (out / "part.glb").write_bytes(b"")
        (out / "part.stl").write_bytes(b"")
        return _ok_result(new_spec)

    monkeypatch.setattr("forge_mind.api.patch.patch_part_structured", _fake)
    client = TestClient(server_app.app)
    r = client.post(
        "/patch-field",
        json={"spec": _SPEC, "field_path": "standoffs.height_mm", "op": "set", "value": 6.0},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["spec"] == new_spec
    assert payload["patch"]["field_path"] == "standoffs.height_mm"
    # Free: no credit charged, no deduction made.
    assert payload["credits_charged"] == 0
    assert harness["deductions"] == []
    # Still recorded for the learning loop.
    (rec,) = harness["records"]
    assert rec.engine == "layoutspec" and rec.status == "ok"


def test_patch_field_refusal(harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    refused = GenerateResult(
        status="refused",
        refused=GenerateRefused(reason="Can't apply that edit: out of range."),
        lane=StandardLaneOutcome(attempted=True, outcome="refused", gen_path="standard:patch"),
    )
    monkeypatch.setattr("forge_mind.api.patch.patch_part_structured", lambda *a, **k: refused)
    client = TestClient(server_app.app)
    r = client.post(
        "/patch-field",
        json={"spec": _SPEC, "field_path": "standoffs.height_mm", "op": "set", "value": -5.0},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "refused"
    assert harness["deductions"] == []


def test_patch_field_requires_spec_and_field(harness: dict[str, Any]) -> None:
    client = TestClient(server_app.app)
    assert (
        client.post("/patch-field", json={"spec": {}, "field_path": "x", "value": 1.0}).status_code
        == 400
    )
    assert (
        client.post(
            "/patch-field", json={"spec": _SPEC, "field_path": "", "value": 1.0}
        ).status_code
        == 400
    )
