"""The result-plumbing advisory (ADR-0117): the sandbox names the shipped-the-wrong-solid
class (measured 2026-07-18: bottle_cap built a genuine cap and bound ``result`` to its
0.5 mm base plate). Advisory only -- ``ok`` stays True and nothing gates on it."""

from __future__ import annotations

from pathlib import Path

from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code

_WRONG_SOLID = """
from build123d import *
with BuildPart() as cap:
    Cylinder(radius=15, height=12)
with BuildPart() as plate:
    Box(10, 10, 0.5)
result = plate.part
"""

_RIGHT_SOLID = """
from build123d import *
with BuildPart() as cap:
    Cylinder(radius=15, height=12)
result = cap.part
"""


def test_wrong_solid_gets_the_plumbing_note(tmp_path: Path) -> None:
    res = run_build123d_code(
        _WRONG_SOLID, tmp_path, part_name="p", limits=SandboxLimits(timeout_s=240)
    )
    assert res.ok  # advisory, never a failure
    assert res.plumbing_note is not None
    assert "RESULT PLUMBING" in res.plumbing_note
    assert "cap" in res.plumbing_note  # names the larger solid the script built


def test_right_solid_gets_no_note(tmp_path: Path) -> None:
    res = run_build123d_code(
        _RIGHT_SOLID, tmp_path, part_name="p", limits=SandboxLimits(timeout_s=240)
    )
    assert res.ok
    assert res.plumbing_note is None
