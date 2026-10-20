"""Smoke test for the fidelity (anti-primitive) gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic.fidelity import expected_tier, fidelity_report


def test_expected_tier_whole_word_not_substring() -> None:
    # 'camera' must NOT match 'cam'; 'proper' must NOT match 'prop'
    assert expected_tier("Raspberry Pi Camera Mounting Plate") == "low"
    assert expected_tier("a proper mounting bracket") == "low"
    # genuine high-complexity keywords
    assert expected_tier("a rocket engine nozzle") == "high"
    assert expected_tier("a centrifugal impeller") == "high"
    # default
    assert expected_tier("a widget thing") == "med"


def test_missing_step_is_not_run_never_false_fail(tmp_path: Path) -> None:
    r = fidelity_report(tmp_path / "nope.step", "code", "a turbine blade")
    assert r.verdict == "not_run"
    assert r.score is None


def test_bare_box_gated_by_expected_complexity(tmp_path: Path) -> None:
    build123d = pytest.importorskip("build123d")
    box = build123d.Box(20, 20, 20)
    step = tmp_path / "box.step"
    build123d.export_step(box, str(step))
    code = "from build123d import Box\nresult = Box(20, 20, 20)\n"

    # a bare box called a "block" (low tier) -> N/A, never failed (a block IS a primitive)
    low = fidelity_report(step, code, "a 20mm block")
    assert low.verdict == "n/a"

    # the SAME bare box called a "turbine blade" (high tier) -> FAIL (primitive faking a real part)
    high = fidelity_report(step, code, "a turbine blade")
    assert high.verdict == "fail"
    assert high.primitive_pile is True
