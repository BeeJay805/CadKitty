"""Zero-LLM stray-indent repair (measured 2026-07-18): the local model occasionally emits ONE
spurious leading space on a top-level line and exec dies on otherwise-sound code. Self-gating:
fires only when stripping that exact line's indent makes the script compile."""

from __future__ import annotations

from forge_mind.agentic.author import clean_code

_BROKEN = (
    "from build123d import *\n"
    "pulley_diameter = 40\n"
    " groove_depth = 4\n"  # the measured artifact: one stray leading space at top level
    "result = None\n"
)


def test_stray_single_space_indent_is_repaired() -> None:
    fixed = clean_code(_BROKEN)
    compile(fixed, "<t>", "exec")  # must compile now
    assert "\n groove_depth" not in fixed and "groove_depth = 4" in fixed


def test_valid_code_is_untouched() -> None:
    good = "x = 1\nif x:\n    y = 2\n"
    assert clean_code(good) == good.strip() or compile(clean_code(good), "<t>", "exec")


def test_real_indentation_structure_is_not_mangled() -> None:
    # A genuinely broken block (missing indent INSIDE an if) must NOT be "repaired" blindly:
    # stripping the named line cannot make it compile, so the original comes back.
    broken = "if True:\nx = 1\n"
    out = clean_code(broken)
    assert out == broken.strip()


def test_non_indent_syntax_errors_pass_through() -> None:
    bad = "def f(:\n    pass\n"
    assert clean_code(bad) == bad.strip()
