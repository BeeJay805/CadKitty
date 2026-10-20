from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge_mind.agentic._cadquery_runner import _find_result


def test_explicit_result_binding_wins_over_a_later_cutting_tool() -> None:
    part = SimpleNamespace(wrapped=object())
    tool = SimpleNamespace(wrapped=object())
    assert _find_result({"result": part, "cutting_tool": tool}) is part


def test_unnamed_last_shape_is_not_exported_as_the_part() -> None:
    with pytest.raises(NameError, match="named result"):
        _find_result({"body_candidate": SimpleNamespace(wrapped=object())})
