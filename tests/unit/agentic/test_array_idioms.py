"""Verified repeated-unit idioms (ADR-0071): radial_pattern / linear_pattern array a PRE-BUILT
solid around/along an axis and fuse -- the 'build ONE unit, array it' primitive for blades, teeth,
throws, fins, so the author never hand-rolls the error-prone rotate+fuse loop.
"""

from __future__ import annotations

import pytest
from build123d import Box, Pos

from forge_mind.agentic.helpers import linear_pattern, radial_pattern


def test_radial_pattern_fuses_evenly_spaced_copies() -> None:
    unit = Pos(10, 0, 0) * Box(2, 2, 2)  # a solid offset from the Z axis
    out = radial_pattern(unit, 4)  # 90 deg apart, fused into one solid
    assert abs(out.volume - 4 * 8) < 1e-3  # four non-overlapping 2x2x2 boxes -> exactly 4x


def test_radial_pattern_returns_list_when_not_fused() -> None:
    copies = radial_pattern(Pos(10, 0, 0) * Box(2, 2, 2), 3, fuse=False)
    assert isinstance(copies, list) and len(copies) == 3


def test_linear_pattern_spaces_along_axis_without_overlap() -> None:
    out = linear_pattern(Box(2, 2, 2), 3, 10.0)  # along X at x=0,10,20 -> no overlap
    assert abs(out.volume - 3 * 8) < 1e-3


def test_pattern_validates_count_and_spacing() -> None:
    with pytest.raises(ValueError):
        radial_pattern(Box(1, 1, 1), 0)  # count must be >= 1
    with pytest.raises(ValueError):
        linear_pattern(Box(1, 1, 1), 2, 0.0)  # spacing must be > 0


def test_helpers_are_exported_for_sandbox_injection() -> None:
    # the sandbox does `from forge_mind.agentic.helpers import *`, so they must be in __all__
    from forge_mind.agentic import helpers

    assert "radial_pattern" in helpers.__all__ and "linear_pattern" in helpers.__all__
