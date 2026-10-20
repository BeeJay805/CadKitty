"""OCCT failure -> targeted fallback hint (lever b's guard): a kernel raise names the
failing op and its ordered fallback; a plain Python error gets NO hint (the traceback is
already self-explanatory); resource blow-ups get the simplify hint."""

from __future__ import annotations

from forge_mind.agentic.kernel_hints import kernel_failure_hint
from forge_mind.agentic.loop import _sandbox_feedback

_FILLET_TB = (
    "Traceback (most recent call last):\n"
    '  File "<authored_part>", line 9, in <module>\n'
    "    fillet(bp.edges().filter_by(Axis.Z), radius=12)\n"
    "OCP.StdFail.StdFail_NotDone: BRep_API: command not done"
)


def test_fillet_raise_gets_the_backoff_ladder() -> None:
    hint = kernel_failure_hint("exec", "StdFail_NotDone: BRep_API: command not done", _FILLET_TB)
    assert hint is not None
    assert "HALVE the radius" in hint
    assert "NEVER delete the underlying feature" in hint


def test_boolean_raise_names_the_overlap_fix() -> None:
    hint = kernel_failure_hint(
        "exec", "Standard_ConstructionError", "BOPAlgo_BuilderSolid failed in BRepAlgoAPI_Cut"
    )
    assert hint is not None and "OVERLAP the tool DEEPER" in hint


def test_plain_python_error_gets_no_hint() -> None:
    assert (
        kernel_failure_hint(
            "exec",
            "NameError: name 'boxx' is not defined",
            "Traceback ...\nNameError: name 'boxx' is not defined",
        )
        is None
    )


def test_timeout_gets_the_simplify_hint() -> None:
    hint = kernel_failure_hint("timeout", "exceeded time limit of 90s", None)
    assert hint is not None and "Simplify" in hint


def test_native_kernel_crash_gets_the_actionable_hint() -> None:
    """A native OCC crash (no Python traceback, empty stderr) must still steer the author:
    the fix is 'remove the global edge-break', not a shrug (measured 2026-07-13 forearm segfault)."""
    hint = kernel_failure_hint(
        "kernel_crash",
        "the geometry kernel crashed natively (exit 3221225477) with no Python error",
        None,  # native crash -> no traceback at all
    )
    assert hint is not None
    assert "chamfer" in hint.lower() and "edge" in hint.lower()


def test_selector_null_gets_reselect_hint() -> None:
    hint = kernel_failure_hint(
        "exec", "Standard_NullObject", "OCP.Standard.Standard_NullObject: TopoDS_Shape is null"
    )
    assert hint is not None and "Re-select" in hint


def test_sandbox_feedback_carries_the_diagnosis() -> None:
    fb = _sandbox_feedback("exec", "StdFail_NotDone: BRep_API: command not done", _FILLET_TB)
    assert "KERNEL DIAGNOSIS:" in fb
    assert "HALVE the radius" in fb
    # and the base contract is intact
    assert "Your script FAILED to build" in fb


def test_sandbox_feedback_without_signature_is_unchanged() -> None:
    fb = _sandbox_feedback("exec", "NameError: name 'x' is not defined", "NameError: ...")
    assert "KERNEL DIAGNOSIS" not in fb


# --- API-misuse hints (measured 2026-07-18 e2e A/B hallucination signatures) -----------------

from forge_mind.agentic.kernel_hints import api_misuse_hint  # noqa: E402


def test_revolve_angle_hallucination_names_revolution_arc() -> None:
    hint = api_misuse_hint("TypeError: revolve() got an unexpected keyword argument 'angle'", None)
    assert hint is not None and "revolution_arc" in hint


def test_fillet_kwarg_hallucinations_name_the_signature() -> None:
    for msg in (
        "TypeError: fillet() got an unexpected keyword argument 'fillet_radius'",
        "TypeError: fillet() missing 1 required positional argument: 'radius'",
        "ValueError: 3D fillet operation takes only Edges",
    ):
        hint = api_misuse_hint(msg, None)
        assert hint is not None and "radius" in hint


def test_arc_hallucinations_name_the_real_classes() -> None:
    for msg in (
        "NameError: name 'Arc' is not defined",
        "NameError: name 'ArcByCenter' is not defined",
    ):
        hint = api_misuse_hint(msg, None)
        assert hint is not None and "CenterArc" in hint


def test_forge_mind_import_hallucination_requires_both_substrings() -> None:
    hint = api_misuse_hint(
        "ImportError: cannot import name 'clearance_hole_d' from 'forge_mind.agentic.helpers'",
        None,
    )
    assert hint is not None and "DELETE the import line" in hint
    # a generic cannot-import (not forge_mind) must NOT fire this rule
    assert api_misuse_hint("ImportError: cannot import name 'foo' from 'bar'", None) is None


def test_polygon_axis_buildpart_matmul_signatures() -> None:
    assert "closes itself" in (
        api_misuse_hint(
            "TypeError: Polygon.__init__() got an unexpected keyword argument 'close'", None
        )
        or ""
    )
    assert "CURVE is required" in (
        api_misuse_hint("AttributeError: 'Axis' object has no attribute 'edges'", None) or ""
    )
    assert "extrude" in (
        api_misuse_hint("ValueError: BuildPart doesn't accept {<Shell object>}", None) or ""
    )
    assert "position-at" in (
        api_misuse_hint("TypeError: unsupported operand type(s) for @: 'BuildLine' and 'int'", None)
        or ""
    )


def test_unknown_python_error_still_gets_no_hint() -> None:
    assert api_misuse_hint("ZeroDivisionError: division by zero", None) is None


def test_sandbox_feedback_carries_the_api_correction() -> None:
    fb = _sandbox_feedback(
        "exec", "TypeError: revolve() got an unexpected keyword argument 'angle'", None
    )
    assert "API CORRECTION:" in fb and "revolution_arc" in fb
    assert "KERNEL DIAGNOSIS" not in fb  # kernel hint keeps precedence only when kernelish


def test_null_topods_names_the_revolve_plane_rule() -> None:
    # Verified live: Plane.XY sketch + revolve(Axis.Z) -> zero-volume part -> Null TopoDS later.
    hint = api_misuse_hint("ValueError: Null TopoDS_Shape object", None)
    assert hint is not None and "CONTAINING the axis" in hint
    # and the kernel path does not swallow it (TopoDS is kernel-shaped but matches no kernel op)
    fb = _sandbox_feedback("exec", "ValueError: Null TopoDS_Shape object", None)
    assert "API CORRECTION:" in fb and "Plane.XZ" in fb


def test_offset_alternative_kind_gets_the_stop_shelling_hint() -> None:
    """The measured cable-clip fixed point (breadth sweep 2026-07-18): 'offset Error, an
    alternative kind may resolve this error' survived 6 identical iterations under the generic
    shell fallback AND survived kind=Kind.INTERSECTION on the recorded code (verified) -- the
    only working escape is to not shell at all, so the specific hint must outrank the generic."""
    # the REAL chained traceback (reproduced 2026-07-18): the OCCT cause rides along, so the
    # kernelish gate passes exactly as it does in production
    tb = (
        "    offset_occt_solid = offset_builder.Shape()\n"
        "OCP.OCP.StdFail.StdFail_NotDone: BRep_API: command not done\n"
        "The above exception was the direct cause of the following exception:\n"
        '  File "authored", line 38, in <module>\n'
        "    offset(amount=-1.5, openings=clip.faces().sort_by(Axis.Z)[-1])\n"
        "RuntimeError: offset Error, an alternative kind may resolve this error"
    )
    hint = kernel_failure_hint(
        "exec", "RuntimeError: offset Error, an alternative kind may resolve this error", tb
    )
    assert hint is not None
    assert "STOP shelling" in hint
    assert "Mode.SUBTRACT" in hint  # the verified direct-wall idiom rides along
    # the generic shell fallback must NOT win for this signature
    assert "opening face" not in hint


def test_unknown_catalog_part_gets_the_verbatim_id_hint() -> None:
    """Measured (breadth 2026-07-18/19): spur_gear invented catalog size variants three times
    (terminal once); the error's own Did-you-mean list carries the real ids."""
    from forge_mind.agentic.kernel_hints import api_misuse_hint

    hint = api_misuse_hint(
        "CatalogError: unknown catalog part 'spur_m2.5_z24_w6_b5'. "
        "Did you mean: spur_m1p5_z24_w8_b5, spur_m2_z24_w8_b5?",
        None,
    )
    assert hint is not None
    assert "Did you mean" in hint and "verbatim" in hint
    assert "FIXED enumeration" in hint


def test_l70_sweep_path_signature_gets_the_ordering_hint() -> None:
    """The exact generation-3 crash: 4/6 manifold iterations died on these."""
    from forge_mind.agentic.kernel_hints import api_misuse_hint

    hint = api_misuse_hint("ValueError: path must be provided", None)
    assert hint is not None and "BuildLine" in hint and "sweep(section, path=path_edge)" in hint


def test_l70_sweep_sections_signature_gets_the_ordering_hint() -> None:
    from forge_mind.agentic.kernel_hints import api_misuse_hint

    hint = api_misuse_hint("ValueError: No sections provided", None)
    assert hint is not None and "BuildSketch" in hint and "loft" in hint


def test_offset_objects_signature_routes_to_the_flow_forms_helper() -> None:
    """Measured 2026-07-25 (impeller smoke S1, iter 1): a hand-rolled curved blade died on
    offset() inside BuildSketch; the repair hand-rolled it AGAIN. The hint must both fix the
    offset misuse AND name the verified helper so the revision stops re-rolling the class."""
    from forge_mind.agentic.kernel_hints import api_misuse_hint

    # The VERBATIM production traceback from S1 iter_01 (result.json), not a hand-made one.
    # Load-bearing detail a review caught: the sandbox compiles "<authored_part>" without
    # linecache, so the authored SOURCE LINE ("offset(...)") never appears -- only the library
    # frame name "in offset" does. A test with an invented source line passed while production
    # returned None.
    tb = (
        "Traceback (most recent call last):\n"
        '  File "...\\_sandbox_runner.py", line 440, in _process_job\n'
        '    exec(compile(exec_code, "<authored_part>", "exec"), ns)\n'
        '  File "<authored_part>", line 27, in <module>\n'
        '  File "...\\build123d\\operations_generic.py", line 593, in offset\n'
        '    raise ValueError("objects must be provided")\n'
        "ValueError: objects must be provided"
    )
    hint = api_misuse_hint("ValueError: objects must be provided", tb)
    assert hint is not None
    assert "backward_curved_impeller" in hint and "flow_forms" in hint
    assert "offset()" in hint


def test_other_ops_sharing_the_error_text_do_not_get_the_offset_rewrite() -> None:
    """FIVE build123d ops raise the same "objects must be provided" (bounding_box, mirror,
    offset, scale, split -- verified in the installed kernel). Only the offset FRAME may get
    the impeller rewrite; a mirror() failure routed there would waste a bounded revision on an
    unrelated part."""
    from forge_mind.agentic.kernel_hints import api_misuse_hint

    tb = (
        '  File "<authored_part>", line 12, in <module>\n'
        '  File "...\\build123d\\operations_generic.py", line 527, in mirror\n'
        '    raise ValueError("objects must be provided")'
    )
    hint = api_misuse_hint("ValueError: objects must be provided", tb)
    assert hint is None or "backward_curved_impeller" not in hint


def test_a_user_helper_named_offset_something_is_not_the_offset_frame() -> None:
    """The discriminator is the frame TERMINATOR "in offset\\n", not the substring "in offset":
    authored code that wraps mirror() in a helper called offset_blade puts "in offset_blade" in
    the traceback, and a bare-substring match would have routed that unrelated failure into the
    impeller rewrite."""
    from forge_mind.agentic.kernel_hints import api_misuse_hint

    tb = (
        '  File "<authored_part>", line 8, in <module>\n'
        '  File "<authored_part>", line 3, in offset_blade\n'
        '  File "...\\build123d\\operations_generic.py", line 527, in mirror\n'
        '    raise ValueError("objects must be provided")'
    )
    hint = api_misuse_hint("ValueError: objects must be provided", tb)
    assert hint is None or "backward_curved_impeller" not in hint


def test_sections_signature_now_also_names_the_helper() -> None:
    from forge_mind.agentic.kernel_hints import api_misuse_hint

    hint = api_misuse_hint("ValueError: No sections provided", None)
    assert hint is not None and "backward_curved_impeller" in hint
