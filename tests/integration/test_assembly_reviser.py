"""The Path-4 closing loop: validator-guided deterministic repair of a failing assembly.

Demonstrates the reviser repairs the defects it has an allow-listed patch for
(too-tight fit -> loosen; slide escapes -> shorten travel), leaves a good assembly
untouched, and stops honestly on a defect it cannot repair (a welded joint = a
missing degree of freedom).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.assembly.reviser import revise_assembly
from tests.fixtures.assembly.slide_graphs import (
    good_slide_graph,
    interference_slide_graph,
    over_travel_slide_graph,
    welded_slide_graph,
)

pytestmark = pytest.mark.slow


def test_reviser_repairs_a_too_tight_fit(tmp_path: Path) -> None:
    result = revise_assembly(interference_slide_graph(), tmp_path)
    assert result.passed, result.report.failed_check_ids()
    assert any("loosen fit" in a for a in result.applied)


def test_reviser_repairs_a_slide_that_escapes(tmp_path: Path) -> None:
    result = revise_assembly(over_travel_slide_graph(), tmp_path)
    assert result.passed, result.report.failed_check_ids()
    assert any("shorten travel" in a for a in result.applied)


def test_reviser_leaves_a_good_assembly_untouched(tmp_path: Path) -> None:
    result = revise_assembly(good_slide_graph(), tmp_path)
    assert result.passed
    assert result.iterations == 0


def test_reviser_stops_honestly_on_an_unrepairable_defect(tmp_path: Path) -> None:
    # A welded joint has no degree of freedom; there is no allow-listed patch for it.
    result = revise_assembly(welded_slide_graph(), tmp_path)
    assert not result.passed
    assert result.iterations == 0
    assert "assembly.range_of_motion" in result.report.failed_check_ids()
