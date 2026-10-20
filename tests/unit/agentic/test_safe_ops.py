"""Graceful-failure edge ops (lever-b): ``safe_fillet`` / ``safe_chamfer`` must
auto-back-off an over-large radius/length and NEVER raise -- a cosmetic round must not
kill an otherwise-valid build (the author loop's #1 documented kernel failure)."""

from __future__ import annotations

import pytest
from build123d import Axis, Box

from forge_mind.agentic.helpers import bottom_face, safe_chamfer, safe_fillet, safe_shell


def test_safe_fillet_rounds_when_the_radius_fits() -> None:
    box = Box(40, 40, 20)
    rounded = safe_fillet(box, box.edges().filter_by(Axis.Z), radius=4)
    # A fillet on convex vertical edges shaves material, so volume drops but the part
    # stays a valid positive-volume solid that actually changed.
    assert rounded.volume > 0
    assert rounded.volume < box.volume


def test_safe_fillet_backs_off_an_impossible_radius_without_raising() -> None:
    # 4 mm square cross-section: a 10 mm corner radius cannot fit (raw fillet would raise).
    thin = Box(4, 4, 40)
    out = safe_fillet(thin, thin.edges().filter_by(Axis.Z), radius=10)
    assert out.volume > 0  # got SOMETHING valid back, no exception escaped


def test_safe_fillet_returns_the_original_when_nothing_can_be_filleted() -> None:
    box = Box(30, 30, 10)
    # No edges to fillet -> every attempt fails -> the original part comes back unchanged.
    out = safe_fillet(box, [], radius=3)
    assert out is box


def test_safe_chamfer_bevels_when_it_fits_and_degrades_when_it_does_not() -> None:
    box = Box(40, 40, 20)
    beveled = safe_chamfer(box, box.edges().filter_by(Axis.Z), length=3)
    assert 0 < beveled.volume < box.volume
    # An absurd chamfer length backs off rather than raising.
    thin = Box(4, 4, 40)
    assert safe_chamfer(thin, thin.edges().filter_by(Axis.Z), length=9).volume > 0
    # Nothing to chamfer -> original returned.
    assert safe_chamfer(box, [], length=2) is box


def test_safe_shell_hollows_to_walls_with_a_default_top_opening() -> None:
    box = Box(40, 30, 25)
    hollow = safe_shell(box, thickness=2)
    assert len(hollow.solids()) == 1
    assert hollow.is_manifold
    assert hollow.volume < box.volume * 0.6  # genuinely hollowed, not a token cut


def test_safe_shell_opens_a_chosen_face() -> None:
    box = Box(40, 30, 25)
    hollow = safe_shell(box, thickness=2, openings=bottom_face(box))
    assert 0 < hollow.volume < box.volume


def test_safe_shell_raises_rather_than_returning_a_solid_when_the_wall_is_too_thick() -> None:
    box = Box(40, 30, 25)
    # A 20 mm wall cannot fit a 30 mm-wide body -> must raise, NEVER silently return solid.
    with pytest.raises(ValueError, match="safe_shell"):
        safe_shell(box, thickness=20)
