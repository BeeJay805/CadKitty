"""The m1_search CLI's script-level pieces: the idiom prompts and the selection re-export.

Selection logic itself lives in forge_mind.structures.selection (tested in
tests/unit/structures/test_selection.py); this file guards the CLI wiring only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "scripts" / "arm_search" / "m1_search.py"
_spec = importlib.util.spec_from_file_location("arm_m1_search", _SRC)
assert _spec and _spec.loader
m1s = importlib.util.module_from_spec(_spec)
sys.modules["arm_m1_search"] = m1s
_spec.loader.exec_module(m1s)


def test_build_prompt_states_interface_and_single_shell() -> None:
    p = m1s.build_prompt("an I-beam section", span=142.0, width=24.0, height=60.0)
    assert "142 mm" in p
    assert "3.4 mm bolt holes" in p
    assert "NO enclosed internal cavities" in p
    assert "I-beam section" in p


def test_idioms_are_distinct() -> None:
    names = [n for n, _ in m1s.IDIOMS]
    assert len(names) == len(set(names)) >= 3


def test_selection_reexported_from_structures() -> None:
    from forge_mind.structures.selection import select_winner

    assert m1s.select_m1_winner is select_winner  # the CLI delegates, no forked logic
