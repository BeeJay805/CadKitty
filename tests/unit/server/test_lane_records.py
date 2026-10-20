"""Lane telemetry persistence (ADR-0035): the standard-lane attempt outcome rides the
``generations.spec`` jsonb column additively, and the ``engine`` column is honest about which
engine BUILT the part. All money-free (generate_part is stubbed; _store_generation captured)."""

from __future__ import annotations

from typing import Any

import pytest

from forge_mind.api.core import (
    Files,
    GenerateError,
    GenerateOk,
    GenerateResult,
    Kit,
    StandardLaneOutcome,
)
from forge_mind.api.schema import (
    BillOfMaterials,
    MakeInstructions,
    ValidationView,
    Vec3,
)
from forge_mind.server import app as server_app


def test_engine_for_maps_provenance() -> None:
    assert server_app._engine_for("standard:layoutspec") == "layoutspec"
    assert server_app._engine_for("implicit:gyroid") == "implicit"
    assert server_app._engine_for("cosmetic") == "agentic"
    assert server_app._engine_for("functional") == "agentic"


def test_lane_spec_fragment() -> None:
    lane = StandardLaneOutcome(
        attempted=True, outcome="fallback_blocked", gen_path="direct-dsl", fallback_reason="why"
    )
    ag = GenerateResult(status="error", lane=lane)
    frag = server_app._lane_spec(ag)
    assert frag is not None
    assert frag["standard_lane"]["outcome"] == "fallback_blocked"
    assert frag["standard_lane"]["gen_path"] == "direct-dsl"
    assert server_app._lane_spec(GenerateResult(status="error")) is None


def _minimal_ok(gen_path: str, lane: StandardLaneOutcome | None) -> GenerateResult:
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
    ok = GenerateOk(
        prompt="p",
        gen_path=gen_path,
        cost_usd=0.02,
        validation=ValidationView(passed=True, overall="pass", tiers=[], failed_checks=[]),
        kit=kit,
        files=Files(glb="part.glb", stl="part.stl"),
        fabrication_mode="print_3d",
    )
    return GenerateResult(status="ok", ok=ok, lane=lane)


def test_run_agentic_records_lane_and_engine(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standard-lane ship records engine='layoutspec' with the lane fragment in spec."""
    import forge_mind.api.core as core_mod

    lane = StandardLaneOutcome(
        attempted=True, outcome="shipped", gen_path="standard:layoutspec", cost_usd=0.02
    )
    result = _minimal_ok("standard:layoutspec", lane)

    def _fake_generate(prompt: str, out: Any, **kw: Any) -> GenerateResult:
        (out / "part.glb").write_bytes(b"")
        (out / "part.stl").write_bytes(b"")
        return result

    monkeypatch.setattr(core_mod, "generate_part", _fake_generate)
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    records: list[Any] = []
    monkeypatch.setattr(server_app, "_store_generation", lambda tok, rec: records.append(rec))
    monkeypatch.setattr(server_app, "_deduct_credits", lambda *a, **k: 4)
    monkeypatch.setattr(server_app, "_upload_artifact", lambda *a, **k: None)

    payload = server_app._run_agentic("a nema plate", "standard", "tok", 1, "user-1")

    assert payload["gen_path"] == "standard:layoutspec"
    assert len(records) == 1
    rec = records[0]
    assert rec.engine == "layoutspec"
    assert rec.spec is not None
    assert rec.spec["gen_path"] == "standard:layoutspec"
    assert rec.spec["standard_lane"]["outcome"] == "shipped"


def test_run_agentic_records_lane_on_error(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane miss followed by an agentic failure still persists the lane attempt."""
    import forge_mind.api.core as core_mod

    lane = StandardLaneOutcome(
        attempted=True, outcome="fallback_blocked", gen_path="direct-dsl", fallback_reason="r"
    )
    result = GenerateResult(
        status="error", error=GenerateError(stage="build", message="m"), lane=lane
    )
    monkeypatch.setattr(core_mod, "generate_part", lambda prompt, out, **kw: result)
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    records: list[Any] = []
    monkeypatch.setattr(server_app, "_store_generation", lambda tok, rec: records.append(rec))

    payload = server_app._run_agentic("a nema plate", "standard", "tok", 1, "user-1")

    assert payload["status"] == "error"
    assert len(records) == 1
    rec = records[0]
    assert rec.engine == "agentic"
    assert rec.spec is not None
    assert rec.spec["standard_lane"]["outcome"] == "fallback_blocked"
