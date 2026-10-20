"""Blast-radius advisory: pinned to the measured wall_hook silent-rebuild specimen.

The incident, verbatim from arm A1 (2026-07-22, `wall_hook_s0`): iteration 3's working
part measured 4,311 mm^3; iteration 4's "revision" returned 119,891 mm^3 -- a 27.8x
rebuild (+2,681%) answering a fix list that asked for neither bulk nor envelope changes.
Legitimate refinement in the same three arms measured p50 0.3% / p90 19.9% (n=119
consecutive pairs); the threshold (40%) sits between that band and the incident tail.
"""

from __future__ import annotations

import pytest

from forge_mind.agentic.repair_diff import repair_diff_advisory

# The A1 wall_hook specimen, exact measured volumes.
_WALL_HOOK_I3_MM3 = 4311.0
_WALL_HOOK_I4_MM3 = 119891.0


@pytest.fixture(autouse=True)
def _flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGE_REPAIR_DIFF", "1")


def test_the_wall_hook_rebuild_is_named() -> None:
    note = repair_diff_advisory(_WALL_HOOK_I3_MM3, _WALL_HOOK_I4_MM3)
    assert note is not None
    assert "2681%" in note or "2,681" in note.replace("2681", "2,681") or "2681" in note
    assert "grew" in note


def test_shrink_direction_is_named() -> None:
    note = repair_diff_advisory(_WALL_HOOK_I4_MM3, _WALL_HOOK_I3_MM3)
    assert note is not None
    assert "shrank" in note


def test_legitimate_refinement_stays_silent() -> None:
    # p50-class refinement (0.3%) and even p90-class (19.9%) stay under the threshold.
    assert repair_diff_advisory(10000.0, 10030.0) is None
    assert repair_diff_advisory(10000.0, 11990.0) is None


def test_threshold_boundary() -> None:
    assert repair_diff_advisory(10000.0, 13900.0) is None  # 39% -- under
    assert repair_diff_advisory(10000.0, 14100.0) is not None  # 41% -- over


def test_unmeasurable_volumes_stay_silent() -> None:
    assert repair_diff_advisory(None, 5000.0) is None
    assert repair_diff_advisory(5000.0, None) is None
    assert repair_diff_advisory(0.0, 5000.0) is None


def test_flag_off_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_REPAIR_DIFF", raising=False)
    assert repair_diff_advisory(_WALL_HOOK_I3_MM3, _WALL_HOOK_I4_MM3) is None
