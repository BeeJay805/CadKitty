"""Measured feature verification, the B1 slice (ADR-0035): concavity + coaxial stacks on the
B-rep, the declared-vs-MEASURED conformance join, the scale lower bound, and the INFO advisory
mechanics that make all of it structurally non-gating this pass."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature, check_measured_holes
from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders
from forge_mind.validator.report import (
    Check,
    CheckStatus,
    Tier,
    aggregate_status,
    build_report,
)
from forge_mind.validator.run import validate

# ------------------------------------------------------------------ fixtures ---


def _plate(
    *,
    hole_mode: OpMode = OpMode.SUBTRACT,
    hole_r: float = 1.7,
    n_holes: int = 4,
    declared_mm: float = 3.4,
    pin_height: float = 20.0,
) -> PartProgram:
    """A 60x40x4 plate with ``n_holes`` M3-position cylinders (SUBTRACT = real holes,
    ADD = solid pins) and the matching clearance_hole declarations."""
    ops: list[Any] = [
        Box(
            op_id="plate", mode=OpMode.ADD, width=60.0, depth=40.0, height=4.0, position=Vec3(z=2.0)
        )
    ]
    corners = [(-20.0, -12.0), (-20.0, 12.0), (20.0, -12.0), (20.0, 12.0)]
    for i, (x, y) in enumerate(corners[:n_holes]):
        ops.append(
            Cylinder(
                op_id=f"hole_{i}",
                mode=hole_mode,
                position=Vec3(x=x, y=y, z=2.0),
                radius=hole_r,
                height=pin_height,
            )
        )
    return PartProgram(
        metadata=Metadata(name="measured_truth_plate"),
        operations=ops,
        constraints=Constraints(min_wall_thickness=2.0),
        hardware_features=[
            HardwareFeature(category="clearance_hole", key="M3", nominal_mm=declared_mm, where="t")
            for _ in range(4)
        ],
    )


def _compiled(program: PartProgram, work: Path) -> Any:
    return compile_program(program, work)


# ------------------------------------------------ concavity + stacks (geometry) ---


def test_concavity_separates_holes_from_bosses(tmp_path: Path) -> None:
    """A subtracted bore reads CONCAVE; an added pin reads convex -- measured, never guessed."""
    program = PartProgram(
        metadata=Metadata(name="hole_and_boss"),
        operations=[
            Box(
                op_id="p", mode=OpMode.ADD, width=60.0, depth=40.0, height=6.0, position=Vec3(z=3.0)
            ),
            Cylinder(
                op_id="hole",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=-15.0, z=3.0),
                radius=2.0,
                height=20.0,
            ),
            Cylinder(
                op_id="boss", mode=OpMode.ADD, position=Vec3(x=15.0, z=9.0), radius=3.0, height=8.0
            ),
        ],
    )
    model = _compiled(program, tmp_path)
    cyls = extract_cylinders(model.part)
    holes = [c for c in cyls if abs(c.radius - 2.0) < 0.01]
    bosses = [c for c in cyls if abs(c.radius - 3.0) < 0.01]
    assert holes and all(c.concave is True for c in holes)
    assert bosses and all(c.concave is False for c in bosses)
    # Extents measured: the hole wall spans the plate thickness.
    assert holes[0].axial_extent is not None
    lo, hi = holes[0].axial_extent
    assert abs((hi - lo) - 6.0) < 0.1


def test_cluster_coaxial_merges_seams_and_keeps_counterbores_distinct(tmp_path: Path) -> None:
    """A counterbore's two diameters stay TWO stacks; same-radius coaxial segments merge."""
    program = PartProgram(
        metadata=Metadata(name="counterbore"),
        operations=[
            Box(
                op_id="p",
                mode=OpMode.ADD,
                width=40.0,
                depth=40.0,
                height=10.0,
                position=Vec3(z=5.0),
            ),
            # through hole r=1.7 + a coaxial counterbore r=3.0 in the top 4mm
            Cylinder(
                op_id="thru", mode=OpMode.SUBTRACT, position=Vec3(z=5.0), radius=1.7, height=30.0
            ),
            Cylinder(
                op_id="cb", mode=OpMode.SUBTRACT, position=Vec3(z=8.0), radius=3.0, height=4.0
            ),
        ],
    )
    model = _compiled(program, tmp_path)
    stacks = [s for s in cluster_coaxial(extract_cylinders(model.part)) if s.concave is True]
    radii = sorted(round(s.radius, 2) for s in stacks)
    assert radii == [1.7, 3.0]  # two distinct features, not one merged, not face-split duplicates


# ------------------------------------------------ declared-vs-measured join ---


def test_measured_true_when_geometry_matches(tmp_path: Path) -> None:
    program = _plate()
    model = _compiled(program, tmp_path)
    checks = check_measured_holes(model.part, program.hardware_features)
    assert len(checks) == 1
    c = checks[0]
    assert c.id == "conformance.measured.clearance_hole.M3"
    assert c.status == CheckStatus.INFO
    assert "MEASURED-TRUE" in c.message


def test_measured_missing_when_hole_cut_undersize(tmp_path: Path) -> None:
    """THE flagship false-PASS: declared 3.4mm, cut 3.0mm. The declaration conforms to the
    registry (existing check passes) but the measured join sees no 3.4mm concave bore."""
    program = _plate(hole_r=1.5)  # cut 3.0mm; declared 3.4mm
    model = _compiled(program, tmp_path)
    checks = check_measured_holes(model.part, program.hardware_features)
    assert len(checks) == 1
    assert "MEASURED-MISSING" in checks[0].message
    # ... while the full report still PASSES this pass (advisory-first; promotion per ADR-0035).
    report = validate(model, program)
    assert report.passed
    assert "conformance.measured.clearance_hole.M3" in report.info_check_ids()


def test_measured_missing_when_holes_are_solid_pins(tmp_path: Path) -> None:
    """Convex pins where holes were declared: concavity excludes them -- 0 measured."""
    program = _plate(hole_mode=OpMode.ADD, pin_height=8.0)
    model = _compiled(program, tmp_path)
    checks = check_measured_holes(model.part, program.hardware_features)
    assert len(checks) == 1
    assert "MEASURED-MISSING" in checks[0].message
    assert "0 measured" in (checks[0].actual or "")


def test_measured_unmeasurable_without_brep() -> None:
    feats = [HardwareFeature(category="clearance_hole", key="M3", nominal_mm=3.4)]
    checks = check_measured_holes(None, feats)
    assert len(checks) == 1
    assert "UNMEASURABLE" in checks[0].message
    assert checks[0].status == CheckStatus.INFO


def test_no_round_declarations_emit_no_measured_checks() -> None:
    feats = [HardwareFeature(category="captive_nut_pocket", key="M5", nominal_mm=8.2)]
    assert check_measured_holes(None, feats) == []


# ------------------------------------------------------------ scale lower bound ---


def test_bbox_scale_flags_a_shrunken_part(tmp_path: Path) -> None:
    """A 0.1x part passes the upper bbox bound; the INFO scale check names it SHRUNKEN."""
    program = PartProgram(
        metadata=Metadata(name="tiny"),
        operations=[
            Box(op_id="p", mode=OpMode.ADD, width=6.0, depth=4.0, height=0.4, position=Vec3(z=0.2))
        ],
        constraints=Constraints(max_bbox=Vec3(x=60.0, y=40.0, z=4.0)),
    )
    model = _compiled(program, tmp_path)
    report = validate(model, program)
    scale = report.check("mesh.bbox_scale")
    assert scale is not None and scale.status == CheckStatus.INFO
    assert "SHRUNKEN" in (scale.actual or "")
    bbox = report.check("mesh.bbox_within_max")
    assert bbox is not None and bbox.status == CheckStatus.PASS  # the documented blind spot


def test_bbox_scale_quiet_on_a_proportionate_part(tmp_path: Path) -> None:
    program = _plate()
    grown = program.model_copy(
        update={
            "constraints": Constraints(min_wall_thickness=2.0, max_bbox=Vec3(x=60.0, y=40.0, z=4.0))
        }
    )
    model = _compiled(grown, tmp_path)
    report = validate(model, grown)
    scale = report.check("mesh.bbox_scale")
    assert scale is not None and "SHRUNKEN" not in (scale.actual or "")


# ------------------------------------------------------------ INFO mechanics ---


def _info_check() -> Check:
    return Check(
        id="conformance.measured.test", tier=Tier.INTERFACES, status=CheckStatus.INFO, message="m"
    )


def test_info_never_moves_the_aggregate() -> None:
    passing = Check(id="a", tier=Tier.MESH, status=CheckStatus.PASS, message="m")
    assert aggregate_status([passing, _info_check()]) == CheckStatus.PASS
    warning = Check(id="b", tier=Tier.MESH, status=CheckStatus.WARN, message="m")
    assert aggregate_status([warning, _info_check()]) == CheckStatus.WARN


def test_info_does_not_block_passes_criteria() -> None:
    passing = Check(id="a", tier=Tier.MESH, status=CheckStatus.PASS, message="m")
    report = build_report([passing, _info_check()], [])
    assert report.passed
    assert report.passes_criteria(required_checks=["a"])
    assert report.info_check_ids() == ["conformance.measured.test"]


# ------------------------------------------------------- agentic measured audit ---


def test_measured_hole_feedback_names_existing_bores(tmp_path: Path) -> None:
    """The agentic audit tells the author WHICH bores exist when the count is short."""
    from forge_mind.agentic.spatial import measured_hole_feedback

    program = _plate(n_holes=2)  # built 2 of the brief's 4
    model = _compiled(program, tmp_path)
    brief = SimpleNamespace(
        features=[SimpleNamespace(name="bolt holes", approx_dims_mm="3.4 mm", count=4)]
    )
    lines = measured_hole_feedback(model.part, brief)
    assert len(lines) == 1
    assert "2 concave bore(s)" in lines[0]
    assert "missing 2" in lines[0]


def test_measured_hole_feedback_quiet_when_complete(tmp_path: Path) -> None:
    from forge_mind.agentic.spatial import measured_hole_feedback

    program = _plate(n_holes=4)
    model = _compiled(program, tmp_path)
    brief = SimpleNamespace(
        features=[SimpleNamespace(name="bolt holes", approx_dims_mm="3.4 mm", count=4)]
    )
    assert measured_hole_feedback(model.part, brief) == []
