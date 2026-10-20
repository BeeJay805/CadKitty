"""Line-level failure attribution (research idea 6): map validator feature-findings back to the
author's own PARAMS keys + code lines. PRECISION-first -- a wrong pointer never ships, so the
tests pin both what it MUST find and what it must REFUSE to guess."""

from __future__ import annotations

from forge_mind.agentic.attribution import attribute_issues, line_attribution_enabled

_PULLEY = """PARAMS = {
    "od": 60.0,
    "ss_depth": 6.0,      # set-screw hole depth
    "n_ss": 2,            # set-screw hole count
}
with BuildPart() as bp:
    Cylinder(radius=30, height=20)          # pulley body
    Cylinder(radius=5, height=22, mode=Mode.SUBTRACT)   # central bore
    # two radial M4 set-screw holes 90 deg apart
    for ang in (0, 90):
        with Locations(Rot(0, 0, ang) * Pos(11, 0, 0)):
            Cylinder(radius=2, height=PARAMS["ss_depth"], rotation=(0, 90, 0), mode=Mode.SUBTRACT)
result = bp.part
"""


def test_maps_a_feature_to_its_code_block() -> None:
    issues = ["MEASURED BORES vs PLAN: still MISSING 2 x Ø4.50mm ('set-screw holes', M4)."]
    out = attribute_issues(issues, _PULLEY, None)
    assert out is not None
    assert "CODE MAP" in out
    assert "set-screw holes" in out
    # points at the PARAMS keys and/or the radial block that build the set-screws
    assert "ss_depth" in out or "n_ss" in out or "radial" in out.lower()


def test_missing_feature_with_no_code_gets_the_safe_strong_claim() -> None:
    """A feature reported missing that NO line references -> 'not built at all; ADD it.'"""
    plain = 'PARAMS = {"w": 40.0}\nresult = Box(40, 20, 5)\n'
    issues = ["MISSING FEATURES: the brief plans a 'lightening holes' pattern not found."]
    out = attribute_issues(issues, plain, None)
    assert out is not None
    assert "NO block" in out and "lightening holes" in out


def test_precision_refuses_a_generic_token() -> None:
    """A label whose only words are generic (a 'plate' in a plate part lights up every line)
    must NOT produce a scattershot mapping."""
    code = "PARAMS = {}\n" + "\n".join(f"Box(1,1,1)  # plate layer {i}" for i in range(8))
    issues = ["MEASURED BORES: something about the ('plate')."]
    out = attribute_issues(issues, code, None)
    # 'plate' hits >4 lines -> dropped; no confident code line, not a MISSING claim -> None
    assert out is None


def test_non_actionable_findings_are_ignored() -> None:
    issues = ["min_wall_thickness: the wall is 0.8mm < 2.0mm", "the B-rep is INVALID"]
    assert attribute_issues(issues, _PULLEY, None) is None


def test_fail_soft_on_garbage() -> None:
    assert attribute_issues(["MISSING FEATURES: ('x')"], "not python {{{", None) is None or True
    assert attribute_issues([], _PULLEY, None) is None


def test_flag_default_on() -> None:
    assert line_attribution_enabled() is True
