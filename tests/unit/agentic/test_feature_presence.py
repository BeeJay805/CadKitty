"""Feature-presence gate v1 (ADR-0117): the three measured omission mechanisms, with the
2026-07-19 offline-calibration facts pinned -- zero false-FAILs on the archived shipped runs
(a single-additive-feature plate authored as one primitive + cuts must NOT fire), the
wall_hook carve-pattern and fan_shroud dropped-count mechanisms MUST fire, and everything
unparseable is UNKNOWN (fail-open, contributes nothing)."""

from __future__ import annotations

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.feature_presence import (
    check_additive_as_subtract,
    check_count_pattern,
    check_volume_excess,
    feature_presence_report,
)


def _brief(features: list[BriefFeature]) -> DesignBrief:
    return DesignBrief(
        object_name="part",
        summary="s",
        visual_description="v",
        overall_envelope_mm="50 x 40 x 40",
        features=features,
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def _hook_brief() -> DesignBrief:
    # the wall_hook calibration exemplar: three parseable additive features
    return _brief(
        [
            BriefFeature(
                name="rail_clip", purpose="clip", approx_dims_mm="12 x 8 x 15", method="loft"
            ),
            BriefFeature(
                name="hook_curve", purpose="hold", approx_dims_mm="40 x 15 x 8", method="loft"
            ),
            BriefFeature(
                name="back_plate", purpose="mount", approx_dims_mm="30 x 12 x 2", method="box"
            ),
            BriefFeature(
                name="screw_hole", purpose="fix", approx_dims_mm="3.4 x 20", method="pocket"
            ),
        ]
    )


_CARVE_CODE = """
with BuildPart() as p:
    Box(120, 40, 25)
    with BuildSketch(p.faces()[0]):
        Rectangle(40, 15)
    extrude(amount=-15, mode=Mode.SUBTRACT)
result = p.part
"""

_ADDITIVE_CODE = """
with BuildPart() as p:
    Box(30, 12, 2)
    with BuildSketch(Plane.XZ):
        Rectangle(8, 15)
    extrude(amount=12)
    with BuildSketch(Plane.XY):
        Circle(1.7)
    extrude(amount=-2, mode=Mode.SUBTRACT)
result = p.part
"""


def test_carve_only_with_multiple_additive_features_gates() -> None:
    rep = check_additive_as_subtract(_hook_brief(), _CARVE_CODE)
    assert [f.check for f in rep.gating] == ["additive_as_subtract"]
    assert "ADDED solid" in rep.findings[0].message


def test_real_additive_construction_passes() -> None:
    assert check_additive_as_subtract(_hook_brief(), _ADDITIVE_CODE).findings == []


def test_single_additive_feature_part_never_fires_g2() -> None:
    """Calibration: the shipped nema17_plate (plate body = its only additive feature,
    one Box + subtractive holes) false-FAILed the first draft of this check."""
    plate = _brief(
        [
            BriefFeature(
                name="plate", purpose="mount", approx_dims_mm="100 x 100 x 5", method="box"
            ),
            BriefFeature(name="bore", purpose="shaft", approx_dims_mm="d12", method="pocket"),
        ]
    )
    assert check_additive_as_subtract(plate, _CARVE_CODE).findings == []


def test_declared_count_without_pattern_gates() -> None:
    shroud = _brief(
        [
            BriefFeature(
                name="mounting_bosses",
                purpose="receive the mounting screws",  # purpose text mentioning screws must NOT mark it negative
                approx_dims_mm="8.0 mm diameter x 2.0 mm height",
                method="revolve",
                count=4,
            ),
        ]
    )
    rep = check_count_pattern(shroud, _CARVE_CODE)
    assert [f.check for f in rep.gating] == ["count_pattern"]
    rep2 = check_count_pattern(shroud, _CARVE_CODE + "\nwith PolarLocations(20, 4):\n    pass")
    assert rep2.findings == []


def test_volume_excess_gates_only_when_fully_parsed() -> None:
    b = _hook_brief()  # all three additive features parse -> expected ~6,960 mm^3
    over = check_volume_excess(b, measured_volume_mm3=60_000.0)
    assert [f.check for f in over.gating] == ["volume_excess"]
    ok = check_volume_excess(b, measured_volume_mm3=9_000.0)
    assert ok.findings == []


def test_partial_parse_demotes_to_advisory() -> None:
    """The slab class: one unparseable feature -> the estimate can ADVISE, never gate."""
    b = _brief(
        [
            BriefFeature(
                name="legs", purpose="join", approx_dims_mm="50x40x5 each", method="box", count=2
            ),
            BriefFeature(
                name="rib",
                purpose="stiffen",
                approx_dims_mm="3 thick, 30x30 triangle",
                method="box",
            ),
        ]
    )
    rep = check_volume_excess(b, measured_volume_mm3=74_960.0)
    assert rep.gating == []  # never gates on a partial parse
    assert any(f.check == "volume_excess_advisory" for f in rep.findings)


def test_wordy_diameter_dims_parse() -> None:
    b = _brief(
        [
            BriefFeature(
                name="boss_a",
                purpose="locate",
                approx_dims_mm="8.0 mm diameter x 2.0 mm height",
                method="revolve",
            ),
            BriefFeature(
                name="boss_b",
                purpose="locate",
                approx_dims_mm="8.0 mm diameter x 2.0 mm height",
                method="revolve",
            ),
        ]
    )
    # both parse (~100.5 mm^3 each) -> a 10,000 mm^3 measured solid is a 50x excess -> gates
    rep = check_volume_excess(b, measured_volume_mm3=10_000.0)
    assert [f.check for f in rep.gating] == ["volume_excess"]


def test_report_merges_and_no_code_is_fail_open() -> None:
    rep = feature_presence_report(_hook_brief(), None, None)
    assert rep.findings == []


# ---- v2 (ADR-0118): the enclosure_lid class -- name-positive beats method-negative ----

_LID_SLAB_CODE = """
with BuildPart() as p:
    Box(70, 50, 3)
    with GridLocations(60, 40, 2, 2):
        Cylinder(1.25, 3, mode=Mode.SUBTRACT)
result = p.part
"""


def _lid_brief() -> DesignBrief:
    """The measured lid class: the planner declares each corner standoff with
    method="pocket" and the dims of the bore it RECEIVES (verbatim field shapes from the
    frozen breadth brief)."""
    standoffs = [
        BriefFeature(
            name=f"standoff_{i}",
            purpose="Mounting hole for corner standoff",
            approx_dims_mm="3 mm diameter threaded bore",
            method="pocket",
        )
        for i in range(1, 5)
    ]
    return _brief(
        [
            BriefFeature(
                name="lid_base", purpose="main surface", approx_dims_mm="70 x 50 x 3", method="box"
            ),
            *standoffs,
        ]
    )


def test_name_positive_standoffs_stay_additive_and_slab_gates() -> None:
    """v2 calibration: three archived lid acceptances were bare slabs with carved holes;
    with standoffs additive-by-name, G2 must gate that code."""
    rep = check_additive_as_subtract(_lid_brief(), _LID_SLAB_CODE)
    assert [f.check for f in rep.gating] == ["additive_as_subtract"]


def test_lid_with_real_standoff_material_passes() -> None:
    """The FINAL s0 boundary case: any genuine added-material attempt (a second additive
    op line) must NOT gate -- proportion quality stays the critic's job."""
    code = _LID_SLAB_CODE + "\nwith BuildPart() as s:\n    Cylinder(3, 8)\n"
    assert check_additive_as_subtract(_lid_brief(), code).findings == []


def test_name_positive_with_negative_word_stays_negative() -> None:
    """'boss_bore' carries a negative word -> the override never rescues it (fail-open):
    the additive set stays at one instance and G2 stays silent."""
    b = _brief(
        [
            BriefFeature(name="plate", purpose="m", approx_dims_mm="50 x 40 x 3", method="box"),
            BriefFeature(name="boss_bore", purpose="m", approx_dims_mm="d5", method="pocket"),
        ]
    )
    assert check_additive_as_subtract(b, _CARVE_CODE).findings == []
