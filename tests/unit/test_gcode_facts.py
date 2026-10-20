"""Deterministic gcode-fact parsing (ADR-0012 SS3 slicer gate, fact extractor).

Runs on a committed hand-authored fixture with KNOWN geometry -- no slicer binary needed -- plus
inline gcode for the dropped-feature and unparseable paths. The no-false-PASS rule is pinned: every
metric is None (UNVERIFIABLE), never a fabricated 0, when it cannot be measured.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.slicing.gcode_facts import parse_gcode_facts

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "slicer" / "tiny.gcode"


def _fixture_mesh() -> trimesh.Trimesh:
    # matches the fixture's two 10x10 layers (z 0.2 and 0.4)
    mesh = trimesh.creation.box((10.0, 10.0, 0.6))
    mesh.apply_translation([5.0, 5.0, 0.3])
    return mesh


def test_fixture_facts_are_predictable() -> None:
    facts = parse_gcode_facts(
        _FIXTURE.read_text(), warnings=["slicer said hi"], mesh=_fixture_mesh()
    )
    assert facts.layer_count == 2
    # support block: total dE 0.60 * filament area (pi*0.875^2 = 2.405) ~= 1.44 mm^3
    assert facts.support_volume_mm3 is not None
    assert 1.3 < facts.support_volume_mm3 < 1.6
    # first-layer contact = layer-1 deposited volume / first layer height, > 0 and finite
    assert facts.first_layer_contact_mm2 is not None
    assert facts.first_layer_contact_mm2 > 0
    # both layers print the full 10x10 silhouette -> ratio ~1, nothing dropped
    assert facts.dropped_layers == []
    assert facts.min_area_ratio is not None
    assert 0.9 < facts.min_area_ratio < 1.1
    # warnings are threaded through verbatim
    assert facts.warnings == ["slicer said hi"]


def test_support_free_gcode_reports_zero_not_none() -> None:
    # a gcode we DID parse but that has no support block reports 0.0 (measured), not None.
    gcode = "M82\n;TYPE:External perimeter\nG1 X10 Y0 E0.5\nG1 X10 Y10 E1.0\n"
    facts = parse_gcode_facts(gcode)
    assert facts.support_volume_mm3 == 0.0


def test_dropped_thin_feature_is_flagged() -> None:
    # layer 2's printed silhouette collapses to a 2x2 square (area 4) while the mesh cross-section
    # is 10x10 (area 100): ratio 0.04 < 0.60 -> the slicer silently dropped the feature.
    gcode = "\n".join(
        [
            "M82",
            ";LAYER_CHANGE",
            ";Z:0.2",
            ";TYPE:External perimeter",
            "G0 X0 Y0",
            "G1 X10 Y0 E1.0",
            "G1 X10 Y10 E2.0",
            "G1 X0 Y10 E3.0",
            "G1 X0 Y0 E4.0",
            ";LAYER_CHANGE",
            ";Z:0.4",
            ";TYPE:External perimeter",
            "G0 X0 Y0",
            "G1 X2 Y0 E4.5",
            "G1 X2 Y2 E5.0",
            "G1 X0 Y2 E5.5",
            "G1 X0 Y0 E6.0",
        ]
    )
    mesh = trimesh.creation.box((10.0, 10.0, 0.6))
    mesh.apply_translation([5.0, 5.0, 0.3])
    facts = parse_gcode_facts(gcode, mesh=mesh)
    assert facts.dropped_layers == [2]
    assert facts.min_area_ratio is not None and facts.min_area_ratio < 0.1


def test_per_layer_diff_is_none_without_a_mesh() -> None:
    facts = parse_gcode_facts(_FIXTURE.read_text())
    assert facts.dropped_layers is None
    assert facts.min_area_ratio is None


def test_empty_gcode_is_all_none() -> None:
    facts = parse_gcode_facts("   \n  \n")
    assert facts.support_volume_mm3 is None
    assert facts.first_layer_contact_mm2 is None
    assert facts.layer_count is None
    assert facts.dropped_layers is None


def test_relative_extrusion_mode_is_handled() -> None:
    # M83 relative E: each E value is a delta, not a cumulative position.
    gcode = "M83\n;TYPE:Support material\nG1 X1 Y1 E0.5\nG1 X2 Y2 E0.5\n"
    facts = parse_gcode_facts(gcode)
    assert facts.support_volume_mm3 is not None
    assert 2.2 < facts.support_volume_mm3 < 2.6  # dE 1.0 * 2.405
