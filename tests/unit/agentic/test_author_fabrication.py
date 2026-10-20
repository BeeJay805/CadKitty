"""The author receives FDM design guidance ONLY when the part is being 3D-printed (ADR-0031): the
"take it into account when building" half of the printable toggle. In GENERAL (the default) the
author models the real object faithfully, unconstrained by printer physics."""

from __future__ import annotations

from forge_mind.agentic.author import _print_guidance
from forge_mind.fabrication import FabricationMode


def test_print_3d_adds_fdm_design_guidance() -> None:
    block = _print_guidance(FabricationMode.PRINT_3D)
    assert block  # non-empty
    lowered = block.lower()
    assert "overhang" in lowered
    assert "orientation" in lowered


def test_general_adds_no_guidance() -> None:
    assert _print_guidance(FabricationMode.GENERAL) == ""
