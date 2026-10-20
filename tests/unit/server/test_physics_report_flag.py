"""The FORGE_PHYSICS_REPORT flag on /validate-upload-demo (ADR-0063), proven money-free against the app.

Invariants: default OFF -> the response is byte-identical (no physics_report key, nothing else changed);
ON -> the block is purely ADDITIVE (removing it yields the same response); and the physics block never
leaks into any manufacture/verdict key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import trimesh
from fastapi.testclient import TestClient

from forge_mind.server import app as server_app


def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    monkeypatch.setattr(server_app, "estimate_stress", lambda *a, **k: None)  # fast + deterministic
    monkeypatch.setattr(server_app, "_DEMO_HITS", {})
    return TestClient(server_app.app)


def _stl_bytes() -> bytes:
    data = trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(file_type="stl")
    return data if isinstance(data, bytes) else bytes(data, "utf-8")


def _post(client: TestClient) -> dict[str, Any]:
    resp = client.post(
        "/validate-upload-demo", files={"file": ("part.stl", _stl_bytes(), "model/stl")}
    )
    assert resp.status_code == 200
    return resp.json()


def _strip_volatile(body: dict[str, Any]) -> dict[str, Any]:
    # `files` carries a per-request random job uuid; `physics_report` is the additive key under test.
    return {k: v for k, v in body.items() if k not in ("files", "physics_report")}


def test_flag_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FORGE_PHYSICS_REPORT", raising=False)
    body = _post(_client(monkeypatch, tmp_path))
    assert "physics_report" not in body


def test_flag_on_is_purely_additive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FORGE_PHYSICS_REPORT", raising=False)
    off = _post(_client(monkeypatch, tmp_path))

    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    on = _post(_client(monkeypatch, tmp_path))

    assert "physics_report" in on
    # everything else is byte-identical: the change is additive only.
    assert _strip_volatile(on) == _strip_volatile(off)


def test_physics_report_never_leaks_into_manufacture_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The upload payload has no manufacture stamp; turning the flag on must not introduce one (the
    physics block is orthogonal to any verdict/credit/manufacture decision)."""
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    on = _post(_client(monkeypatch, tmp_path))
    assert "validation" not in on
    assert "not_for_manufacture" not in on
    assert "manufacture_warning" not in on
    # the honest pre-print report verdict is untouched and never says "safe"
    assert on["report"]["fit_function_verified"] is False
    assert "safe" not in on["report"]["verdict"].lower()


def test_flag_on_emits_the_frozen_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    on = _post(_client(monkeypatch, tmp_path))
    pr = on["physics_report"]
    assert pr["schema_version"] == 2  # v2: the optional ADR-0086 motion block
    assert pr["scope"] == "single_part"
    assert {s["id"] for s in pr["suites"]} == {
        "printproof",
        "fitproof",
        "boltbrain",
        "loadproof",
        "motion",
    }
    assert pr["overall"]["verdict"] in {"pass", "warn", "fail"}
    assert pr["vitals"]["volume_mm3"] is not None


# ── slice 2: STEP ingest ──────────────────────────────────────────────────────────────────────────


def _step_bytes(tmp_path: Path) -> bytes:
    from build123d import Box, BuildPart, export_step

    out = tmp_path / "fixture.step"
    with BuildPart() as bp:
        Box(20, 20, 20)
    export_step(bp.part, str(out))
    return out.read_bytes()


def _post_step(client: TestClient, content: bytes) -> Any:
    return client.post(
        "/validate-upload-demo", files={"file": ("part.step", content, "application/step")}
    )


def test_step_rejected_when_flag_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With the flag OFF, STEP ingest does not exist — a .step is a 400, exactly as before."""
    monkeypatch.delenv("FORGE_PHYSICS_REPORT", raising=False)
    content = _step_bytes(tmp_path)
    resp = _post_step(_client(monkeypatch, tmp_path), content)
    assert resp.status_code == 400


def test_step_accepted_when_flag_on_with_brep_finding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    content = _step_bytes(tmp_path)
    resp = _post_step(_client(monkeypatch, tmp_path), content)
    assert resp.status_code == 200
    body = resp.json()
    # the SAME intrinsic pipeline ran on the tessellated STL
    assert body["report"]["printable"] in {"pass", "warn", "fail"}
    pr = body["physics_report"]
    findings = [f for s in pr["suites"] if s["id"] == "printproof" for f in s["findings"]]
    step_int = next((f for f in findings if f["id"] == "printproof.step_integrity"), None)
    assert step_int is not None
    assert step_int["status"] == "pass"  # a real build123d B-rep box
    assert step_int["label"] == "Editable CAD (B-rep)"
    assert step_int["confidence"]["tier"] == "verified"


def test_garbage_step_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unreadable STEP is a 400 (fail closed) — never a silent pass on geometry we can't read."""
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    resp = _post_step(_client(monkeypatch, tmp_path), b"ISO-10303-21; not really a step file")
    assert resp.status_code == 400


def test_stl_when_flag_on_has_no_step_finding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    body = _post(_client(monkeypatch, tmp_path))
    findings = [f for s in body["physics_report"]["suites"] for f in s["findings"]]
    assert not any(f["id"] == "printproof.step_integrity" for f in findings)


# ── slice 3: hole detection + BoltBrain on a real STEP ─────────────────────────────────────────────


def _step_with_hole_bytes(tmp_path: Path) -> bytes:
    """A 20x20x10 box with a Ø3.4 mm through hole = an M3 clearance hole."""
    from build123d import Box, BuildPart, Cylinder, Mode, export_step

    out = tmp_path / "holed.step"
    with BuildPart() as bp:
        Box(20, 20, 10)
        Cylinder(radius=1.7, height=20, mode=Mode.SUBTRACT)
    export_step(bp.part, str(out))
    return out.read_bytes()


def test_step_with_hole_is_classified_by_boltbrain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    content = _step_with_hole_bytes(tmp_path)
    resp = _post_step(_client(monkeypatch, tmp_path), content)
    assert resp.status_code == 200
    pr = resp.json()["physics_report"]
    bolt = [f for s in pr["suites"] if s["id"] == "boltbrain" for f in s["findings"]]
    m3 = next((f for f in bolt if "m3" in f["id"]), None)
    assert m3 is not None, f"expected an M3 classification, got {[f['id'] for f in bolt]}"
    assert m3["status"] == "info"
    assert m3["confidence"]["tier"] == "verified"
    assert m3["locator"] is not None and m3["locator"]["point"] is not None  # located on the hole


# ── slice 4: edge distance via radial ray-cast (the mutation gate) ─────────────────────────────────


def _plate_with_hole_bytes(tmp_path: Path, hole_x: float) -> bytes:
    """A 20x20x4 plate with a Ø4 mm through hole offset to x=hole_x (edges at +/-10)."""
    from build123d import Box, BuildPart, Cylinder, Locations, Mode, export_step

    out = tmp_path / f"plate_{hole_x}.step"
    with BuildPart() as bp:
        Box(20, 20, 4)
        with Locations((hole_x, 0, 0)):
            Cylinder(radius=2, height=10, mode=Mode.SUBTRACT)
    export_step(bp.part, str(out))
    return out.read_bytes()


def _edge_finding(body: dict[str, Any]) -> dict[str, Any] | None:
    bolt = [
        f for s in body["physics_report"]["suites"] if s["id"] == "boltbrain" for f in s["findings"]
    ]
    return next((f for f in bolt if f["id"] == "boltbrain.edge_distance"), None)


def test_hole_near_edge_warns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """MUTATION GATE: a hole 7 mm from center (~1 mm wall to the 10 mm edge) MUST warn on edge distance."""
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    resp = _post_step(_client(monkeypatch, tmp_path), _plate_with_hole_bytes(tmp_path, 7.0))
    assert resp.status_code == 200
    ed = _edge_finding(resp.json())
    assert ed is not None and ed["status"] == "warn"
    assert ed["locator"] is not None and ed["locator"]["point"] is not None


def test_centered_hole_edge_distance_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    resp = _post_step(_client(monkeypatch, tmp_path), _plate_with_hole_bytes(tmp_path, 0.0))
    assert resp.status_code == 200
    ed = _edge_finding(resp.json())
    assert ed is not None and ed["status"] == "pass"


# ── slice 5: intent unlocks FitProof (end-to-end through the multipart endpoint) ───────────────────


def _m5_plate_step(tmp_path: Path) -> bytes:
    """A 30x30x4 plate with a centered Ø5.5 mm hole = an M5 clearance hole."""
    from build123d import Box, BuildPart, Cylinder, Mode, export_step

    out = tmp_path / "m5plate.step"
    with BuildPart() as bp:
        Box(30, 30, 4)
        Cylinder(radius=2.75, height=10, mode=Mode.SUBTRACT)
    export_step(bp.part, str(out))
    return out.read_bytes()


def test_intent_unlocks_fitproof(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    client = _client(monkeypatch, tmp_path)
    intent = json.dumps({"mates_with": [{"kind": "bolt", "spec": "M5"}]})
    resp = client.post(
        "/validate-upload-demo",
        files={"file": ("m5plate.step", _m5_plate_step(tmp_path), "application/step")},
        data={"intent": intent},
    )
    assert resp.status_code == 200
    fit = [
        f
        for s in resp.json()["physics_report"]["suites"]
        if s["id"] == "fitproof"
        for f in s["findings"]
    ]
    passed = [f for f in fit if f["status"] == "pass"]
    assert passed, f"expected a passing M5 fit, got {[(f['id'], f['status']) for f in fit]}"
    assert passed[0]["confidence"]["tier"] == "verified"
    assert passed[0]["locator"] is not None and passed[0]["locator"]["point"] is not None


def test_no_intent_keeps_fitproof_not_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    resp = _post_step(_client(monkeypatch, tmp_path), _m5_plate_step(tmp_path))  # no intent field
    fit = [
        f
        for s in resp.json()["physics_report"]["suites"]
        if s["id"] == "fitproof"
        for f in s["findings"]
    ]
    assert all(f["status"] == "not_run" for f in fit)


# ── slice 8: intent.load unlocks the decision-grade LoadProof (mesh FEA) ───────────────────────────


def test_intent_load_unlocks_loadproof(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    from forge_mind.physics.fea import LoadResponse

    monkeypatch.setenv("FORGE_PHYSICS_REPORT", "1")
    # The FEA solve is exercised in test_fea_load_response.py; mock it here for a fast, deterministic
    # end-to-end wiring check (same pattern as estimate_stress).
    fake = LoadResponse(
        max_deflection_mm=2.1,
        deflection_at_mm=(1.0, 2.0, 3.0),
        max_von_mises_mpa=12.0,
        stress_at_mm=(4.0, 5.0, 6.0),
        load_n=49.0,
        load_case="49 N over the top face, base fixed",
        material_e_mpa=3500.0,
        note="Isotropic FEA; FDM is weaker across layers — not a safety guarantee.",
    )
    monkeypatch.setattr(server_app, "estimate_load_response", lambda *a, **k: fake)
    client = _client(monkeypatch, tmp_path)
    intent = json.dumps(
        {"load": {"magnitude": 5, "unit": "kg", "direction": {"x": 0, "y": 0, "z": -1}}}
    )
    resp = client.post(
        "/validate-upload-demo",
        files={"file": ("box.stl", _stl_bytes(), "model/stl")},
        data={"intent": intent},
    )
    assert resp.status_code == 200
    lp = [
        f
        for s in resp.json()["physics_report"]["suites"]
        if s["id"] == "loadproof"
        for f in s["findings"]
    ]
    defl = next((f for f in lp if f["id"] == "loadproof.deflection"), None)
    assert defl is not None and defl["measured"]["value"] == 2.1
    assert defl["confidence"]["tier"] == "computed"
    assert defl["locator"] is not None and defl["locator"]["point"] is not None
