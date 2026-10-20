"""toothed_rim (ADR-0078): the external tooth band that fuses onto a disc rim -- the idiom the
hard-5 flywheel was missing when it dropped its 120-tooth starter ring."""

from __future__ import annotations

import pytest
from build123d import Cylinder

from forge_mind.agentic.gears import toothed_rim


def test_toothed_rim_fuses_onto_a_disc() -> None:
    band = toothed_rim(100.0, 24, 8.0)
    tip = max(band.bounding_box().size.X, band.bounding_box().size.Y)
    assert 112.0 < tip < 128.0  # module = 100/21.5 -> tip diameter ~120.9
    disc = Cylinder(radius=50, height=8)
    fused = disc + band
    assert len(fused.solids()) == 1  # the band FUSES -- one solid, never a floating ring
    assert fused.volume > disc.volume  # the teeth stand proud of the rim


def test_toothed_rim_validates() -> None:
    with pytest.raises(ValueError):
        toothed_rim(100.0, 6, 8.0)  # too few teeth
    with pytest.raises(ValueError):
        toothed_rim(0.0, 24, 8.0)
