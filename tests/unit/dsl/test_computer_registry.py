"""Computer-build hardware registry (ADR-0087 S2): every drive/fan fact is cited, provisional, and
resolvable, and the exact values match the SFF-8201/8301 + de-facto fan sources they were entered
from (a golden test per entry — the registry-typo poison-pill guard, ADR-0087 P9)."""

from __future__ import annotations

import pytest

from forge_mind.dsl.standards import ALL_REGISTRIES, case_fan, drive_form
from forge_mind.dsl.standards.computer import CASE_FANS, DRIVE_FORMS


def test_2_5_inch_drive_matches_sff_8201() -> None:
    d = drive_form("2.5in_drive")
    assert (d.length_mm, d.width_mm, d.max_width_mm) == (100.20, 69.85, 70.10)
    assert d.heights_mm == (7.0, 9.5, 15.0)
    assert d.mount_thread == "M3"
    assert d.bottom_holes == 4 and d.side_holes == 4
    # the exact hole coordinates are NOT in the public SFF-8201 excerpts -> never invented
    assert d.bottom_hole_pitch_mm is None
    assert "SFF-8201" in d.source


def test_3_5_inch_drive_matches_sff_8301() -> None:
    d = drive_form("3.5in_drive")
    assert (d.width_mm, d.length_mm) == (101.60, 147.00)
    assert d.heights_mm == (17.8, 26.1, 42.0)
    assert d.mount_thread == "6-32 UNC"
    assert d.bottom_hole_pitch_mm == 76.20  # SFF-8301 A13, cited
    assert "SFF-8301" in d.source


@pytest.mark.parametrize(
    ("key", "frame", "spacing"),
    [
        ("fan_80", 80.0, 71.5),
        ("fan_92", 92.0, 82.5),
        ("fan_120", 120.0, 105.0),
        ("fan_140", 140.0, 124.5),
    ],
)
def test_case_fan_hole_spacing(key: str, frame: float, spacing: float) -> None:
    f = case_fan(key)
    assert f.frame_mm == frame
    assert f.hole_spacing_mm == spacing  # the cited de-facto center-to-center corner spacing
    assert f.thickness_mm == 25.0


def test_every_entry_is_cited_and_provisional() -> None:
    # the honesty invariant of the whole library: nothing here is a published-standard-verified fact
    # (drive hole coords unsourced; fans have no SFF; nothing physically measured) -> all provisional
    for spec in (*DRIVE_FORMS.all(), *CASE_FANS.all()):
        assert spec.source.strip(), f"{spec.key} has no citation"
        assert spec.provisional is True, f"{spec.key} must be provisional until verified"


def test_registered_in_all_registries() -> None:
    assert ALL_REGISTRIES["drive_form"].has("2.5in_drive")
    assert ALL_REGISTRIES["case_fan"].has("fan_120")


def test_unknown_key_raises_with_the_known_list() -> None:
    with pytest.raises(ValueError, match="unknown drive_form"):
        drive_form("5.25in_drive")
