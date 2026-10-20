"""CadQuery exec bridge (ADR-0112): the $0, model-free de-risk. Proves arbitrary CadQuery source
runs in the sandbox, bridges to a build123d solid on the shared OCP kernel, exports STL + STEP, and
feeds the EXISTING validator -- so wiring CAD-Coder in later is transport, not new geometry risk.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.agentic.cadquery_bridge import load_validation_model, run_cadquery_code
from forge_mind.agentic.sandbox import SandboxLimits
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate

_LIMITS = SandboxLimits(timeout_s=180)


def test_cadquery_box_bridges_executes_and_validates(tmp_path: Path) -> None:
    """A plain CadQuery box: executes, exports STL+STEP, re-imports into a CADModel, and runs the
    full 4-tier validator without raising. This is the end-to-end bridge contract."""
    code = "import cadquery as cq\nresult = cq.Workplane('XY').box(40, 20, 10)\n"
    res = run_cadquery_code(code, tmp_path, part_name="block", limits=_LIMITS)

    assert res.ok, f"expected ok, got {res.error_type}: {res.error_message}"
    assert res.stl_path is not None and res.stl_path.exists()
    assert res.step_path is not None and res.step_path.exists()
    # bbox came back with the box extents (proves the bridged solid is real geometry).
    assert res.extents is not None
    assert sorted(round(e) for e in res.extents) == [10, 20, 40]

    model = load_validation_model(res.step_path, res.stl_path)
    report = validate(model, _validation_program())
    assert isinstance(report, ValidationReport)
    # A clean welded box is watertight -> the mesh tier does not fail on it.
    assert "mesh" not in {str(c.tier) for c in report.checks if c.status.name == "FAIL"}


def test_workplane_result_is_wrapped_not_just_a_shape(tmp_path: Path) -> None:
    """CAD-Coder binds a Workplane to `result`; the bridge must call `.val()` and wrap it. A part
    with a through-hole exercises the .val() path on a non-trivial solid."""
    code = (
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').circle(15).extrude(8).faces('>Z').workplane()"
        ".hole(10)\n"
    )
    res = run_cadquery_code(code, tmp_path, part_name="bored_disc", limits=_LIMITS)
    assert res.ok, f"{res.error_type}: {res.error_message}"
    assert res.step_path is not None and res.step_path.exists()


def test_broken_cadquery_is_a_clean_exec_failure(tmp_path: Path) -> None:
    """A raised exception in the authored code comes back as ok=False/error_type=exec -- never a
    crash of the parent (one bad part must not abort a bench run)."""
    res = run_cadquery_code("import cadquery as cq\nresult = cq.oops()\n", tmp_path, limits=_LIMITS)
    assert not res.ok
    assert res.error_type == "exec"


def test_non_shape_result_is_no_result(tmp_path: Path) -> None:
    """Binding a non-CAD object to `result` must fail closed as no_result, not export a bogus part."""
    res = run_cadquery_code("import cadquery as cq\nresult = 42\n", tmp_path, limits=_LIMITS)
    assert not res.ok
    assert res.error_type == "no_result"


def _validation_program():  # type: ignore[no-untyped-def]
    from forge_mind.dsl.primitives import Box, OpMode
    from forge_mind.dsl.program import Constraints, Metadata, PartProgram
    from forge_mind.types import Vec3

    return PartProgram(
        metadata=Metadata(
            name="bridge_test", units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)
        ),
        operations=[Box(op_id="massing", mode=OpMode.ADD, width=1.0, depth=1.0, height=1.0)],
        constraints=Constraints(min_wall_thickness=1.0),
    )
