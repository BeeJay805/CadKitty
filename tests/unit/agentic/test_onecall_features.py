"""ADR-0133 fix 2: FORGE_ONECALL_FEATURES -- one-call hole vocabulary.

R3 measured Hole/CounterBoreHole/CounterSinkHole at 0% usage in 131 generated files
while the hand-rolled stack (the density signal, 78 stacks in 46 files) fails on
through-depth arithmetic and stale-face re-selection. The one-call idioms eliminate
both BY CONSTRUCTION. The snippets themselves are execution-verified by the
parametrized sandbox test in test_cheatsheet_idioms.py (it runs the whole
VERIFIED_IDIOMS dict, dark entries included). mirror_solid was priced OUT.
"""

from __future__ import annotations

import pytest

from forge_mind.agentic.cheatsheet import (
    _ONECALL_ORDER,
    _ORDER,
    VERIFIED_IDIOMS,
    author_skills_block,
    onecall_enabled,
)


def test_flag_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_ONECALL_FEATURES", raising=False)
    assert not onecall_enabled()
    off = author_skills_block()
    assert off == author_skills_block(one_call=False)
    assert "CounterBoreHole(" not in off  # the old sketch-stack teaching stands
    assert "extrude(amount=-24, mode=Mode.SUBTRACT)" in off  # old counterbore stack present


def test_flag_on_swaps_to_one_call_teaching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_ONECALL_FEATURES", "1")
    assert onecall_enabled()
    on = author_skills_block()
    assert on == author_skills_block(one_call=True)
    assert "Hole(radius=" in on and "CounterBoreHole(" in on and "CounterSinkHole(" in on
    # the swapped-out stacks are GONE (no mixed messaging on the same feature)
    assert "extrude(amount=-24, mode=Mode.SUBTRACT)" not in on
    assert "one call" in on.lower()


def test_order_is_derived_never_hand_copied() -> None:
    """Law 2: the ON-order derives from _ORDER via the swap map -- same length, same
    sequence, exactly two entries swapped, every key present in VERIFIED_IDIOMS."""
    assert len(_ONECALL_ORDER) == len(_ORDER)
    diffs = [(a[0], b[0]) for a, b in zip(_ORDER, _ONECALL_ORDER, strict=True) if a[0] != b[0]]
    assert diffs == [
        ("through_holes", "one_call_holes"),
        ("counterbore_holes", "one_call_counterbore"),
    ]
    for key, _intro in _ONECALL_ORDER:
        assert key in VERIFIED_IDIOMS


def test_author_prompts_swap_only_the_skills_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge_mind.agentic import author as au

    # prebuilt variants differ exactly by the skills swap
    assert au._SYSTEM_INITIAL_ONECALL != au._SYSTEM_INITIAL
    assert au._SYSTEM_INITIAL_ONECALL.replace(au._SKILLS_ONECALL, au._SKILLS) == au._SYSTEM_INITIAL
    assert au._SYSTEM_REVISE_ONECALL.replace(au._SKILLS_ONECALL, au._SKILLS) == au._SYSTEM_REVISE
    # call-time selection honors the env
    monkeypatch.delenv("FORGE_ONECALL_FEATURES", raising=False)
    assert au._system_initial() == au._SYSTEM_INITIAL
    assert au._system_revise() == au._SYSTEM_REVISE
    monkeypatch.setenv("FORGE_ONECALL_FEATURES", "1")
    assert au._system_initial() == au._SYSTEM_INITIAL_ONECALL
    assert au._system_revise() == au._SYSTEM_REVISE_ONECALL
