"""LayoutSpec discriminated union + interpret_layout dispatch (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.generator import GeneratedLayout
from forge_mind.layout.layoutspec import LayoutSpec
from forge_mind.layout.pi4_interpreter import interpret_electronics_tray, interpret_pi4
from forge_mind.layout.pi4_spec import ElectronicsTraySpec, Pi4CaseSpec
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.layout.spec import LBracketMotorMountSpec, Pattern

REPO = Path(__file__).resolve().parents[3]
L_FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench001_l_bracket.layoutspec.json"
PI4_FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench002_pi4_case.layoutspec.json"
ADAPTER_FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench003_adapter_plate.layoutspec.json"

_ADAPTER: TypeAdapter[LayoutSpec] = TypeAdapter(LayoutSpec)


def _load(path: Path) -> LayoutSpec:
    return _ADAPTER.validate_python(json.loads(path.read_text()))


def test_union_discriminates_on_pattern() -> None:
    lb = _load(L_FIXTURE)
    pi4 = _load(PI4_FIXTURE)
    adapter = _load(ADAPTER_FIXTURE)
    assert isinstance(lb, LBracketMotorMountSpec) and lb.pattern is Pattern.L_BRACKET_MOTOR_MOUNT
    assert isinstance(pi4, ElectronicsTraySpec) and pi4.pattern is Pattern.ELECTRONICS_TRAY
    assert isinstance(adapter, AdapterPlateSpec) and adapter.pattern is Pattern.ADAPTER_PLATE


def test_dispatch_routes_bench001_to_l_bracket(tmp_path: Path) -> None:
    result = run_layout_pipeline(_load(L_FIXTURE), tmp_path)
    assert result.passed
    assert result.program is not None
    assert any(o.op_id == "motor_flange" for o in result.program.operations)


def test_dispatch_routes_bench002_to_pi4_case(tmp_path: Path) -> None:
    result = run_layout_pipeline(_load(PI4_FIXTURE), tmp_path)
    assert result.passed
    assert result.program is not None
    assert any(o.op_id.startswith("standoff") for o in result.program.operations)


def test_dispatch_routes_bench003_to_adapter_plate(tmp_path: Path) -> None:
    result = run_layout_pipeline(_load(ADAPTER_FIXTURE), tmp_path)
    assert result.passed
    assert result.program is not None
    op_ids = {o.op_id for o in result.program.operations}
    assert "plate" in op_ids
    assert {"ext_bolt_back", "ext_bolt_front"} <= op_ids


def test_backward_compat_aliases() -> None:
    # ADR-0007 Stage E: the old Pi4 names alias the board-generic ones (same objects),
    # and the serialized pattern value stays "pi4_case" so old fixtures/records load.
    assert Pi4CaseSpec is ElectronicsTraySpec
    assert interpret_pi4 is interpret_electronics_tray
    # PI4_CASE is an alias of ELECTRONICS_TRAY (same enum member). Widen to Pattern so
    # the runtime identity check isn't statically reduced to a literal comparison.
    pi4_alias: Pattern = Pattern.PI4_CASE
    assert pi4_alias is Pattern.ELECTRONICS_TRAY
    assert Pattern.ELECTRONICS_TRAY.value == "pi4_case"


def test_generated_layout_schema_supports_plate_and_tray() -> None:
    # ADR-0014: the LLM wrapper collapsed to {plate (composable), pi4_case}.
    plate_cfg = {
        "name": "t",
        "envelope_mm": {"x": 80.0, "y": 60.0, "z": 8.0},
        "material": "PLA",
        "min_wall_mm": 4.0,
        "base": "flat_plate",
        "interfaces": [
            {"ref": "nema17_face", "face": "plate_top"},
            {"ref": "extrusion_2020", "face": "plate_bottom"},
        ],
    }
    only_plate = GeneratedLayout.model_validate({"plate": plate_cfg})
    assert only_plate.plate is not None and only_plate.pi4_case is None

    only_pi4 = GeneratedLayout.model_validate({"pi4_case": json.loads(PI4_FIXTURE.read_text())})
    assert only_pi4.pi4_case is not None and only_pi4.plate is None
