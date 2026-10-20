"""ADR-0019 Step-3: the mass + center-of-mass physics metric (reported, not gated)."""

from __future__ import annotations

import math
from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.computed_ops import FacetedStrutOp
from forge_mind.dsl.primitives import Box
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.layout.computed_interpreter import ComputedBodySpec, interpret_computed_body
from forge_mind.types import Vec3
from forge_mind.validator.metrics import compute_physics, density_for
from forge_mind.validator.run import validate


def test_density_registry_lookup() -> None:
    assert density_for("PLA") == 1.24
    assert density_for("petg") == 1.27  # case-insensitive
    assert density_for("unobtanium") == 1.24  # unknown -> PLA fallback


def test_box_mass_and_center_of_mass(tmp_path: Path) -> None:
    program = PartProgram(
        metadata=Metadata(name="box", units="mm"),
        operations=[Box(op_id="b", width=10.0, depth=20.0, height=4.0)],
    )
    model = compile_program(program, tmp_path)
    m = compute_physics(model.part, "PLA")
    assert math.isclose(m.volume_mm3, 800.0, rel_tol=1e-3)
    assert math.isclose(m.mass_g, 0.8 * 1.24, rel_tol=1e-3)  # 800 mm^3 = 0.8 cm^3 * 1.24
    # Box is centered at origin -> COM ~ origin.
    assert abs(m.center_of_mass.x) < 1e-6
    assert abs(m.center_of_mass.y) < 1e-6
    assert abs(m.center_of_mass.z) < 1e-6


def test_material_scales_mass(tmp_path: Path) -> None:
    program = PartProgram(
        metadata=Metadata(name="box", units="mm"),
        operations=[Box(op_id="b", width=10.0, depth=10.0, height=10.0)],
    )
    model = compile_program(program, tmp_path)
    pla = compute_physics(model.part, "PLA")
    petg = compute_physics(model.part, "PETG")
    assert math.isclose(pla.volume_mm3, petg.volume_mm3)  # same geometry
    assert petg.mass_g > pla.mass_g  # PETG is denser


def test_validate_attaches_metrics_to_report(tmp_path: Path) -> None:
    spec = ComputedBodySpec(
        name="strut",
        envelope_mm=Vec3(x=100.0, y=20.0, z=20.0),
        material="PLA",
        min_wall_mm=3.0,
        op=FacetedStrutOp(cross_section="hexagonal"),
    )
    program = interpret_computed_body(spec)
    model = compile_program(program, tmp_path)
    report = validate(model, program, material="PLA")
    assert report.metrics is not None
    assert report.metrics.mass_g > 0.0
    assert report.metrics.volume_mm3 > 0.0
    assert report.metrics.material == "PLA"


def test_metrics_are_reported_not_gated(tmp_path: Path) -> None:
    # The metric never changes pass/fail: a passing prismatic box stays passing WITH metrics.
    program = PartProgram(
        metadata=Metadata(name="box", units="mm"),
        operations=[Box(op_id="b", width=10.0, depth=10.0, height=4.0)],
    )
    model = compile_program(program, tmp_path)
    report = validate(model, program)
    assert report.metrics is not None
    # metrics presence does not introduce any check (it is not a Check, so no new ids).
    assert "metrics" not in report.failed_check_ids()
