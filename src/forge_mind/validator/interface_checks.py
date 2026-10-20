"""Tier 4: resolve each declared interface against the checker library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge_mind.dsl.program import PartProgram
from forge_mind.validator.geometry import extract_cylinders
from forge_mind.validator.interfaces.base import InterfaceContext
from forge_mind.validator.interfaces.registry import get_interface
from forge_mind.validator.report import Check, CheckStatus, Fix, Tier


def run_interface_checks(
    part: Any, program: PartProgram, stl_path: Path | None = None
) -> tuple[list[Check], list[Fix]]:
    if not program.interfaces:
        return [], []

    cylinders = extract_cylinders(part)
    ctx = InterfaceContext(
        part=part,
        stl_path=stl_path,
        min_wall_mm=program.constraints.min_wall_thickness,
    )
    checks: list[Check] = []
    fixes: list[Fix] = []

    for decl in program.interfaces:
        checker = get_interface(decl.interface_type.value)
        if checker is None:
            checks.append(
                Check(
                    id=f"interfaces.{decl.name}.unknown_type",
                    tier=Tier.INTERFACES,
                    status=CheckStatus.FAIL,
                    message=f"No checker registered for interface type {decl.interface_type.value!r}.",
                    expected="known interface type",
                    actual=decl.interface_type.value,
                )
            )
            continue
        c, f = checker.check(decl, cylinders, ctx)
        checks.extend(c)
        fixes.extend(f)

    return checks, fixes


__all__ = ["run_interface_checks"]
