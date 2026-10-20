"""FORGE_DOCTRINE (ADR-0122 candidate): the proactive construction-doctrine block.

Null hypothesis on record: the 2026-07-18 static rules block was net-zero and regressed
the pulley class (arms J-N) -- so byte-identical-off is the contract, and promotion needs
the arm.
"""

from __future__ import annotations

import pytest

from forge_mind.agentic.author import _doctrine_block
from forge_mind.agentic.cheatsheet import CONSTRUCTION_DOCTRINE


def test_flag_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_DOCTRINE", raising=False)
    assert _doctrine_block() == ""


def test_flag_on_carries_the_measured_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_DOCTRINE", "1")
    block = _doctrine_block()
    assert CONSTRUCTION_DOCTRINE in block
    for rule in (
        "fillets and chamfers LAST",
        "ONE pass",
        "path FIRST",
        "NAMED variable",
        "ShapeList",
    ):
        assert rule in block
