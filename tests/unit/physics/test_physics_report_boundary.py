"""Boundary guard (ADR-0056/0063): the validator must NEVER import the physics-report projection.

The projection is a LEAF that depends on the validator, not the reverse. If the validator ever imported
``physics.report``, the projection could be wired into a gate and become a second verdict authority. This
test fails the moment that dependency edge appears.
"""

from __future__ import annotations

from pathlib import Path

import forge_mind


def test_validator_never_imports_physics_report() -> None:
    validator_dir = Path(forge_mind.__file__).parent / "validator"
    offenders: list[str] = []
    for py in validator_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "physics.report" in text or "physics import report" in text:
            offenders.append(str(py))
    assert not offenders, (
        f"validator modules import physics.report (verdict-authority leak): {offenders}"
    )


def test_compose_takes_a_finished_report_not_geometry() -> None:
    """Structural check: the composer's first parameter is a finished IntrinsicReport, so it is incapable
    of loading a mesh or running a check."""
    import inspect

    from forge_mind.physics.report.compose import compose_physics_report
    from forge_mind.validator.intrinsic import IntrinsicReport

    sig = inspect.signature(compose_physics_report)
    first = next(iter(sig.parameters.values()))
    assert first.annotation in (IntrinsicReport, "IntrinsicReport")
