"""Conformance check: the no-false-PASS gate. Pure data -- fast suite.

The central assertion is that a deliberately-WRONG hardware feature is caught (FAIL) and an
unrecognized one is FLAGGED (WARN), never silently passed."""

from __future__ import annotations

from forge_mind.validator.conformance import HardwareFeature, check_conformance
from forge_mind.validator.report import CheckStatus


def test_correct_feature_passes() -> None:
    c = check_conformance(
        HardwareFeature(category="clearance_hole", key="M3", fit="normal", nominal_mm=3.4)
    )
    assert c.status == CheckStatus.PASS


def test_deliberately_wrong_feature_fails() -> None:
    # an M3 clearance built at 5.0 mm (registry says 3.4) MUST be caught.
    c = check_conformance(
        HardwareFeature(category="clearance_hole", key="M3", fit="normal", nominal_mm=5.0)
    )
    assert c.status == CheckStatus.FAIL
    assert "NON-CONFORMANT" in c.message


def test_unknown_category_is_flagged_not_passed() -> None:
    c = check_conformance(HardwareFeature(category="unicorn_slot", key="X", nominal_mm=1.0))
    assert c.status == CheckStatus.WARN  # flagged, never PASS


def test_unknown_key_is_flagged_not_passed() -> None:
    c = check_conformance(HardwareFeature(category="clearance_hole", key="M99", nominal_mm=9.0))
    assert c.status == CheckStatus.WARN


def test_provisional_match_passes_with_note() -> None:
    c = check_conformance(HardwareFeature(category="heat_set_insert", key="M3", nominal_mm=4.0))
    assert c.status == CheckStatus.PASS
    assert "PROVISIONAL" in c.message


def test_bearing_seat_press_fit_conforms() -> None:
    c = check_conformance(
        HardwareFeature(category="bearing_seat", key="608", fit="press", nominal_mm=21.95)
    )
    assert c.status == CheckStatus.PASS


def test_never_false_pass() -> None:
    # Anything not an exact registry match is FAIL or WARN -- never PASS.
    bad = [
        HardwareFeature(
            category="clearance_hole", key="M3", fit="normal", nominal_mm=3.0
        ),  # off by 0.4
        HardwareFeature(category="clearance_hole", key="M77", nominal_mm=3.4),  # unknown key
        HardwareFeature(category="not_a_thing", key="M3", nominal_mm=3.4),  # unknown category
        HardwareFeature(
            category="bearing_seat", key="608", fit="press", nominal_mm=25.0
        ),  # wrong dim
    ]
    for f in bad:
        assert check_conformance(f).status != CheckStatus.PASS
