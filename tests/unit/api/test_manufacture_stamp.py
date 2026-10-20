"""The refusal-honesty stamp: never present a part as manufacturable when the gates said no.

This test exists because the logic was previously UNTESTABLE. It lived inside `server/app.py`
behind a module-level `import fastapi`, which the local venv does not have, so `tests/unit/server`
cannot collect and the decision shipped verified only by reading -- on a code path that decides
whether a user is warned not to print a part.

The `not_assemblable` case is the one that motivated the extraction: this stamp is a DENY-LIST, so
a new disposition string fails OPEN here (no stamp, no warning) while the allow-list consumers
elsewhere fail CLOSED on the same unknown string. Both shapes exist in this repo.
"""

from __future__ import annotations

from forge_mind.api.manufacture_stamp import manufacture_stamp


def test_a_good_part_is_not_stamped() -> None:
    stamped, warning = manufacture_stamp({"disposition": "shipped", "validation": {"passed": True}})
    assert stamped is False and warning is None


def test_caveats_still_ship_unstamped() -> None:
    # A caveat means real-but-imperfect; it must not be branded unmanufacturable.
    stamped, warning = manufacture_stamp(
        {"disposition": "shipped_with_caveats", "validation": {"passed": True}}
    )
    assert stamped is False and warning is None


def test_an_unassemblable_assembly_is_stamped_and_says_why() -> None:
    # THE REGRESSION THIS FILE EXISTS FOR (ADR-0126). These parts pass their per-part gates and
    # are NOT judge-rejected, so both pre-existing flags miss them -- without an explicit branch
    # the deny-list would return not_for_manufacture=False for an assembly that cannot be built.
    stamped, warning = manufacture_stamp(
        {"disposition": "not_assemblable", "validation": {"passed": True}}
    )
    assert stamped is True
    assert warning is not None and "NOT for manufacture" in warning
    # It must NOT blame the acceptance review: the judge accepted H3; the physics refused it.
    assert "acceptance review" not in warning
    assert "cannot be built as designed" in warning


def test_a_judge_rejection_keeps_its_own_distinct_reason() -> None:
    stamped, warning = manufacture_stamp(
        {"disposition": "rejected", "validation": {"passed": True}}
    )
    assert stamped is True
    assert warning is not None and "acceptance review" in warning
    assert "cannot be built as designed" not in warning


def test_failed_validation_outranks_the_disposition() -> None:
    for disposition in ("shipped", "rejected", "not_assemblable"):
        stamped, warning = manufacture_stamp(
            {"disposition": disposition, "validation": {"passed": False}}
        )
        assert stamped is True, disposition
        assert warning is not None and "did not pass validation" in warning, disposition


def test_a_missing_validation_block_does_not_invent_a_failure() -> None:
    # `passed` defaults True when absent: this stamp reports the OTHER gates' verdicts, it does
    # not manufacture one of its own.
    assert manufacture_stamp({"disposition": "shipped"}) == (False, None)
    assert manufacture_stamp({"disposition": "shipped", "validation": None}) == (False, None)


def test_an_unknown_disposition_is_not_stamped_and_that_is_the_trap() -> None:
    # DELIBERATE, and the reason this module carries a warning in its docstring: a deny-list
    # cannot recognise a string it was never taught. Any FUTURE non-shipping disposition must be
    # added here explicitly -- it will NOT inherit the stamp the way the allow-list consumers
    # elsewhere would fail closed on it.
    assert manufacture_stamp({"disposition": "some_future_refusal"}) == (False, None)
