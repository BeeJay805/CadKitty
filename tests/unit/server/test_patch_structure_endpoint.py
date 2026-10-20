"""POST /patch-structure (ADR-0060 slice C): the deterministic add/remove route. The route's
projection + the FREE billing (no LLM, no credit) are proven money-free against the real FastAPI app
with patch_part_structural stubbed (its geometry is covered in tests/unit/api/test_patch_structure.py)."""

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

_SPEC = {"pattern": "composable_plate", "name": "adapter"}  # opaque to the stubbed route


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
            prompt="add interface clearance_hole",
            gen_path="standard:patch",
            cost_usd=0.0,  # structural edit is free (no LLM)
            validation=ValidationView(passed=True, overall="pass", tiers=[], failed_checks=[]),
            kit=kit,
            files=Files(glb="part.glb", stl="part.stl"),
            fabrication_mode="print_3d",
            spec=spec,
            patch=PatchView(
                field_path="interfaces.2",
                op="add",
                old_value=None,
                new_value="clearance_hole@plate_bottom",
            ),
        ),
        lane=StandardLaneOutcome(
            attempted=True, outcome="shipped", gen_path="standard:patch", cost_usd=0.0
        ),
    )


@pytest.fixture()
def harness(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"records": [], "deductions": [], "calls": []}
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


def test_patch_structure_add_is_free(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    new_spec = {"pattern": "composable_plate", "name": "adapter", "interfaces": [1, 2, 3]}

    def _fake(spec: dict[str, Any], op: str, out: Any, **kw: Any) -> GenerateResult:
        harness["calls"].append((op, kw))
        (out / "part.glb").write_bytes(b"")
        (out / "part.stl").write_bytes(b"")
        return _ok_result(new_spec)

    monkeypatch.setattr("forge_mind.api.patch.patch_part_structural", _fake)
    client = TestClient(server_app.app)
    r = client.post(
        "/patch-structure",
        json={
            "spec": _SPEC,
            "op": "add",
            "ref": "clearance_hole",
            "face": "plate_bottom",
            "component_id": "M3",
        },
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["spec"] == new_spec
    assert payload["patch"]["op"] == "add"
    # Free: no credit charged, no deduction made.
    assert payload["credits_charged"] == 0
    assert harness["deductions"] == []
    # the route forwarded op="add" + the assembled placement dict
    (op, kw) = harness["calls"][0]
    assert op == "add"
    assert kw["placement"]["ref"] == "clearance_hole"
    assert kw["placement"]["component_id"] == "M3"
    (rec,) = harness["records"]
    assert rec.engine == "layoutspec" and rec.status == "ok"


def test_patch_structure_remove_forwards_index(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake(spec: dict[str, Any], op: str, out: Any, **kw: Any) -> GenerateResult:
        harness["calls"].append((op, kw))
        (out / "part.glb").write_bytes(b"")
        (out / "part.stl").write_bytes(b"")
        return _ok_result({"pattern": "composable_plate", "name": "adapter"})

    monkeypatch.setattr("forge_mind.api.patch.patch_part_structural", _fake)
    client = TestClient(server_app.app)
    r = client.post("/patch-structure", json={"spec": _SPEC, "op": "remove", "index": 1})
    assert r.status_code == 200
    (op, kw) = harness["calls"][0]
    assert op == "remove" and kw["index"] == 1 and kw["placement"] is None


def test_patch_structure_refusal(harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    refused = GenerateResult(
        status="refused",
        refused=GenerateRefused(reason="Can't apply that edit: out of range."),
        lane=StandardLaneOutcome(attempted=True, outcome="refused", gen_path="standard:patch"),
    )
    monkeypatch.setattr("forge_mind.api.patch.patch_part_structural", lambda *a, **k: refused)
    client = TestClient(server_app.app)
    r = client.post("/patch-structure", json={"spec": _SPEC, "op": "remove", "index": 0})
    assert r.status_code == 200
    assert r.json()["status"] == "refused"
    assert harness["deductions"] == []


def test_patch_structure_validates_body(harness: dict[str, Any]) -> None:
    client = TestClient(server_app.app)
    # missing spec
    assert (
        client.post(
            "/patch-structure",
            json={"spec": {}, "op": "add", "ref": "clearance_hole", "face": "plate_top"},
        ).status_code
        == 400
    )
    # add without ref/face
    assert client.post("/patch-structure", json={"spec": _SPEC, "op": "add"}).status_code == 400
    # remove without index
    assert client.post("/patch-structure", json={"spec": _SPEC, "op": "remove"}).status_code == 400
