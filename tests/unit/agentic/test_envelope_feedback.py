"""Envelope-overshoot advisory (bench rows 17-18, the bbox x3 plateau class): the mismatch
message must name the offending AXIS, the target, the overage, and the direction -- 'rebuild
at the intended scale' gave the author nothing to edit and three parts stalled on it."""

from __future__ import annotations

from types import SimpleNamespace

from forge_mind.agentic.validate import _envelope_mismatch
from forge_mind.types import BBox, Vec3


def _bbox(x: float, y: float, z: float) -> BBox:
    return BBox(min=Vec3(x=0, y=0, z=0), max=Vec3(x=x, y=y, z=z))


BRIEF = SimpleNamespace(overall_envelope_mm="60 x 36 x 12 mm")


def test_names_axis_direction_and_amount() -> None:
    mismatch, issue = _envelope_mismatch(_bbox(83, 43, 12), BRIEF)
    assert mismatch and issue is not None
    assert "X is 83.0 mm" in issue and "~60 mm" in issue
    assert "SHRINK" in issue and "+38%" in issue
    assert "Y is 43.0 mm" in issue  # +19% now exceeds the 15% derived-envelope bound


def test_undersize_says_grow() -> None:
    # sorted pairing: 12<->12, 30<->36 (-17%, inside tol), 36<->60 (-40%) -> the Y=36 extent
    # is the one matched against the brief's 60 slot and must GROW
    mismatch, issue = _envelope_mismatch(_bbox(30, 36, 12), BRIEF)
    assert mismatch and issue is not None
    assert "GROW" in issue and "Y is 36.0 mm" in issue and "~60 mm" in issue


def test_clean_and_unparseable_stay_silent() -> None:
    assert _envelope_mismatch(_bbox(60, 36, 12), BRIEF) == (False, None)
    noisy = SimpleNamespace(overall_envelope_mm="roughly palm sized")
    assert _envelope_mismatch(_bbox(83, 43, 12), noisy) == (False, None)  # fail-open


def test_parses_per_number_units_and_prefers_the_primary_statement() -> None:
    # Row 50 golden_ssd_bracket k4: the brief's PRIMARY envelope ("110 mm L x 80 mm W x
    # 17.5 mm H") failed to parse, so the scanner latched onto the compact "110x80x4"
    # inside the trailing rationale and condemned a correct part. The primary must win.
    from forge_mind.agentic.brief import parse_envelope_mm

    k4 = (
        "110 mm L x 80 mm W x 17.5 mm H -- base is 110x80x4 mm per spec; rail height is "
        "drive thickness 9.5 mm + 4 mm base + ~4 mm rail-top lip = ~17.5 mm total envelope"
    )
    assert parse_envelope_mm(k4) == (110.0, 80.0, 17.5)
    # the plain per-number-unit form, no axis letters
    assert parse_envelope_mm("110 mm x 80 mm x 17.5 mm") == (110.0, 80.0, 17.5)
    # historical forms stay byte-identical
    assert parse_envelope_mm("60 x 36 x 12 mm") == (60.0, 36.0, 12.0)
    assert parse_envelope_mm("120 L x 40 W x 85 H mm") == (120.0, 40.0, 85.0)
    assert parse_envelope_mm("120 by 30 by 15 (handle + base)") == (120.0, 30.0, 15.0)
    assert parse_envelope_mm("no numbers here") is None
