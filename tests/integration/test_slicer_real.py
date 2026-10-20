"""Slow integration: the REAL slicer binary end-to-end (ADR-0012 SS3 step 6).

Skipped cleanly when no slicer is installed, so CI without a binary stays green. When a binary IS
present, it proves the wrapper -> real slice -> fact-parse chain on a trivial box.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from forge_mind.slicing.gcode_facts import parse_gcode_facts
from forge_mind.slicing.slicer import slice_part, slicer_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not slicer_available(), reason="no slicer binary installed"),
]


def test_real_slice_of_a_box(tmp_path: Path) -> None:
    mesh = trimesh.creation.box((20.0, 20.0, 10.0))
    mesh.apply_translation([0.0, 0.0, 5.0])
    stl = tmp_path / "box.stl"
    mesh.export(stl)

    outcome = slice_part(stl, tmp_path / "out")
    assert outcome.ok, f"real slice failed: {outcome.error}"
    assert outcome.gcode_path is not None and outcome.gcode_path.exists()

    facts = parse_gcode_facts(
        outcome.gcode_path.read_text(encoding="utf-8", errors="replace"),
        warnings=outcome.warnings,
        mesh=mesh,
    )
    assert facts.layer_count is not None and facts.layer_count > 10  # 10 mm / 0.2 mm ~ 50
    assert facts.support_volume_mm3 == 0.0  # a box is self-supporting
    assert facts.first_layer_contact_mm2 is not None and facts.first_layer_contact_mm2 > 0
    # the box prints close to its full 400 mm^2 cross-section on every layer
    assert facts.min_area_ratio is not None and facts.min_area_ratio > 0.7
    assert facts.dropped_layers == []
