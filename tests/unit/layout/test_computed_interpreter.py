"""ADR-0019 Step-2: the faceted_strut computed-body interpreter.

Proves the framework end-to-end on ONE op: a CATEGORICAL ComputedOp -> derived resolved
PartProgram -> compiles -> validates through the existing 4 tiers. The LLM emits no numbers
(the op is categorical only); dims derive from coarse intent; an infeasible request is
rejected up front.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.computed_ops import FacetedStrutOp, LatticeOp
from forge_mind.dsl.primitives import Box, ComputedSolid
from forge_mind.layout.computed_interpreter import ComputedBodySpec, interpret_computed_body
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.types import Vec3
from forge_mind.validator.metrics import density_for
from forge_mind.validator.run import validate


def _spec(
    cross_section: str, envelope: Vec3, min_wall: float = 3.0, name: str = "strut"
) -> ComputedBodySpec:
    return ComputedBodySpec(
        name=name,
        envelope_mm=envelope,
        min_wall_mm=min_wall,
        op=FacetedStrutOp(cross_section=cross_section),
    )


def _checks(spec: ComputedBodySpec, tmp_path: Path):  # type: ignore[no-untyped-def]
    program = interpret_computed_body(spec)
    model = compile_program(program, tmp_path / spec.name)
    report = validate(model, program)
    return program, report, {c.id: c.status.value for c in report.checks}


def test_faceted_strut_hex_validates_all_tiers(tmp_path: Path) -> None:
    _prog, report, ck = _checks(_spec("hexagonal", Vec3(x=100.0, y=20.0, z=20.0)), tmp_path)
    assert report.passed, report.failed_check_ids()
    assert ck["mesh.watertight"] == "pass"
    assert ck["features.min_wall_thickness"] == "pass"  # true across-flats, not an acute sliver
    assert ck["printability.overhang_max_angle"] == "pass"  # flat-bottom hex (30 deg faces)


def test_faceted_strut_rect_validates_all_tiers(tmp_path: Path) -> None:
    _prog, report, ck = _checks(_spec("rectangular", Vec3(x=100.0, y=20.0, z=20.0)), tmp_path)
    assert report.passed, report.failed_check_ids()
    assert ck["features.min_wall_thickness"] == "pass"
    assert ck["printability.overhang_max_angle"] == "pass"


def test_op_is_categorical_only_no_numbers() -> None:
    # The LLM-facing op carries ONLY the discriminant + a categorical (ADR-0019 Q1/Q2).
    op = FacetedStrutOp(cross_section="hexagonal")
    assert set(op.model_dump()) == {"kind", "cross_section"}


def test_dims_derive_from_envelope(tmp_path: Path) -> None:
    program = interpret_computed_body(_spec("hexagonal", Vec3(x=100.0, y=20.0, z=20.0)))
    op = program.operations[0]
    assert isinstance(op, ComputedSolid)
    assert op.length == 100.0  # = envelope.x (member axis)
    assert op.size > 0
    assert op.size * math.sqrt(3.0) >= 3.0 - 1e-9  # across-flats clears min_wall
    assert 2.0 * op.size <= 20.0 + 1e-9  # across-corners fits the envelope minor


def test_undersized_cross_section_grows_not_rejects(tmp_path: Path) -> None:
    # min_wall too large for the LLM-estimated minor -> GROW to the minimum buildable cross-section
    # (grow-don't-reject), instead of the old up-front LayoutError -> silent direct-DSL fallback.
    program, report, _ck = _checks(
        _spec("hexagonal", Vec3(x=100.0, y=4.0, z=4.0), min_wall=10.0), tmp_path
    )
    assert report.passed, report.failed_check_ids()
    assert program.constraints.max_bbox is not None
    assert program.constraints.max_bbox.y > 4.0  # the minor grew to host the 10mm-wall section
    assert program.constraints.max_bbox.x == 100.0  # the functional span (X) is unchanged


def test_undersized_strut_with_hard_max_refuses() -> None:
    # A user HARD max that cannot host the min_wall cross-section -> honest refusal, NOT a grow and
    # NOT a silent fallback: the interpreter never grows past a user-stated limit.
    spec = ComputedBodySpec(
        name="strut",
        envelope_mm=Vec3(x=100.0, y=4.0, z=4.0),
        max_envelope_mm=Vec3(x=100.0, y=4.0, z=4.0),
        min_wall_mm=10.0,
        op=FacetedStrutOp(cross_section="hexagonal"),
    )
    with pytest.raises(LayoutError, match="cannot fit within the requested max envelope"):
        interpret_computed_body(spec)


# --- LLM selection path wiring (ADR-0019 Step 4) ----------------------------


def test_interpret_layout_routes_computed_body(tmp_path: Path) -> None:
    # The LayoutSpec union dispatch routes a ComputedBodySpec to interpret_computed_body. This
    # is the DETERMINISTIC path (a directly-constructed spec); the LLM front-door is blocked by
    # the Anthropic compiled-grammar ceiling (ADR-0019 Step 4/6 — a 3rd wrapper member is over
    # budget) and awaits a structural fix, so GeneratedLayout stays 2-member.
    spec = _spec("hexagonal", Vec3(x=100.0, y=20.0, z=20.0))
    program = interpret_layout(spec)
    assert isinstance(program.operations[0], ComputedSolid)
    model = compile_program(program, tmp_path)
    assert validate(model, program).passed


# --- LATTICE op (the straight-strut weight-saver) ---------------------------


def _lattice_spec(
    density: str, envelope: Vec3, min_wall: float = 4.0, name: str = "lat"
) -> ComputedBodySpec:
    return ComputedBodySpec(
        name=name,
        envelope_mm=envelope,
        min_wall_mm=min_wall,
        op=LatticeOp(cell="grid", density=density),
    )


def _solid_mass_g(envelope: Vec3, material: str = "PLA") -> float:
    return (envelope.x * envelope.y * envelope.z / 1000.0) * density_for(material)


@pytest.mark.parametrize("density", ["light", "medium", "dense"])
def test_lattice_validates_all_tiers(density: str, tmp_path: Path) -> None:
    # Every member is a vertical wall (no overhang) at min_wall (no acute sliver); the union is a
    # single watertight waffle. The conservative checks pass without an SDF min_wall.
    _prog, report, ck = _checks(_lattice_spec(density, Vec3(x=60.0, y=60.0, z=20.0)), tmp_path)
    assert report.passed, report.failed_check_ids()
    assert ck["mesh.single_body"] == "pass"  # union integrity
    assert ck["mesh.watertight"] == "pass"
    assert ck["features.min_wall_thickness"] == "pass"  # walls derived >= min_wall
    assert ck["printability.overhang_max_angle"] == "pass"  # all walls vertical


@pytest.mark.parametrize("density", ["light", "medium", "dense"])
def test_lattice_is_lighter_than_solid(density: str, tmp_path: Path) -> None:
    env = Vec3(x=60.0, y=60.0, z=20.0)
    _prog, report, _ck = _checks(_lattice_spec(density, env), tmp_path)
    assert report.metrics is not None
    assert report.metrics.mass_g < _solid_mass_g(env)  # the lattice payoff: lighter than solid


def test_lattice_op_is_categorical_only_no_numbers() -> None:
    op = LatticeOp(density="light")
    assert set(op.model_dump()) == {"kind", "cell", "density"}


def test_lattice_derives_straight_wall_members_at_min_wall() -> None:
    env = Vec3(x=60.0, y=60.0, z=20.0)
    program = interpret_computed_body(_lattice_spec("medium", env, min_wall=4.0))
    assert len(program.operations) >= 4  # a grid of wall members, unioned by the compiler
    for op in program.operations:
        assert isinstance(op, Box)
        assert op.height == env.z  # full height along the print Z -> vertical -> no overhang
        assert min(op.width, op.depth) == 4.0  # the wall thickness == min_wall (the strut dim)


def test_lattice_density_monotonic_in_member_count() -> None:
    env = Vec3(x=80.0, y=80.0, z=20.0)
    counts = [
        len(interpret_computed_body(_lattice_spec(d, env)).operations)
        for d in ("light", "medium", "dense")
    ]
    assert counts[0] <= counts[1] <= counts[2]  # denser -> more members
    assert counts[0] < counts[2]


def test_undersized_lattice_grows_not_rejects(tmp_path: Path) -> None:
    # Too small for a cell at min_wall -> GROW the lateral footprint to the 1-cell minimum instead
    # of rejecting; still a real lattice that validates every tier (never a false-pass solid).
    program = interpret_computed_body(
        _lattice_spec("light", Vec3(x=10.0, y=10.0, z=10.0), min_wall=4.0)
    )
    assert program.constraints.max_bbox is not None
    assert program.constraints.max_bbox.x >= 12.0 - 1e-9  # grew to >= 3*min_wall (one real cell)
    assert program.constraints.max_bbox.y >= 12.0 - 1e-9
    model = compile_program(program, tmp_path / "lat")
    assert validate(model, program).passed


def test_undersized_lattice_with_hard_max_refuses() -> None:
    spec = ComputedBodySpec(
        name="lat",
        envelope_mm=Vec3(x=10.0, y=10.0, z=10.0),
        max_envelope_mm=Vec3(x=10.0, y=10.0, z=10.0),
        min_wall_mm=4.0,
        op=LatticeOp(cell="grid", density="light"),
    )
    with pytest.raises(LayoutError, match="cannot fit within the requested max envelope"):
        interpret_computed_body(spec)


def test_interpret_layout_routes_lattice(tmp_path: Path) -> None:
    program = interpret_layout(_lattice_spec("medium", Vec3(x=60.0, y=60.0, z=20.0)))
    assert all(isinstance(op, Box) for op in program.operations)
    model = compile_program(program, tmp_path)
    assert validate(model, program).passed


def test_envelope_derived_from_size_class_when_null(tmp_path: Path) -> None:
    # Step 2: the LLM no longer sizes the member -> envelope null, derived from the coarse size_class
    # (the router nulls it). compact -> 80mm span; the derived body validates every tier.
    spec = ComputedBodySpec(
        name="cb",
        envelope_mm=None,
        size_class="compact",
        min_wall_mm=4.0,
        op=FacetedStrutOp(cross_section="hexagonal"),
    )
    program = interpret_computed_body(spec)
    assert program.constraints.max_bbox is not None
    assert (
        program.constraints.max_bbox.x == 80.0
    )  # compact span (SIZE_CLASS_ENVELOPE), not LLM-picked
    model = compile_program(program, tmp_path / "cb")
    assert validate(model, program).passed
