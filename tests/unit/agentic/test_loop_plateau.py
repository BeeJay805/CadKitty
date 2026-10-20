"""The loop's dynamic early-stop must give a CONCRETE missing feature a longer leash than a vague
quality plateau (ADR-0030): with the live iteration cap at 20+, a flat _PLATEAU_PATIENCE=2 was
abandoning a hole-short bracket after ~2 tries. ``_plateau_stop`` encodes that patience choice."""

from __future__ import annotations

from forge_mind.agentic.loop import (
    _MISSING_FEATURE_PATIENCE,
    _PLATEAU_PATIENCE,
    _plateau_stop,
)
from forge_mind.agentic.validate import AgenticValidation


def _val(*, features_missing: bool) -> AgenticValidation:
    """A coherent built candidate (one watertight solid) that is or isn't short a feature."""
    return AgenticValidation(
        passed=not features_missing,
        hard_ok=True,
        report=None,
        body_count=1,
        watertight=True,
        volume_mm3=1000.0,
        features_missing=features_missing,
    )


def test_missing_feature_earns_more_patience() -> None:
    assert _MISSING_FEATURE_PATIENCE > _PLATEAU_PATIENCE


def test_plain_plateau_stops_at_base_patience() -> None:
    val = _val(features_missing=False)
    assert _plateau_stop(_PLATEAU_PATIENCE - 1, val) is False
    assert _plateau_stop(_PLATEAU_PATIENCE, val) is True


def test_missing_feature_keeps_going_past_base_patience() -> None:
    val = _val(features_missing=True)
    # must NOT give up at the short window while a requested hole is still absent ...
    assert _plateau_stop(_PLATEAU_PATIENCE, val) is False
    assert _plateau_stop(_MISSING_FEATURE_PATIENCE - 1, val) is False
    # ... but it is still bounded, so a part the author cannot finish eventually stops.
    assert _plateau_stop(_MISSING_FEATURE_PATIENCE, val) is True


def test_attempt_note_makes_every_revision_prompt_unique() -> None:
    """Measured (arm G): an unchanged anchor composed an identical revision prompt and the greedy
    local roster returned a byte-identical part. The attempt note must differ per attempt."""
    from forge_mind.agentic.loop import _attempt_note

    notes = [_attempt_note(a, 1) for a in (2, 3, 4)]
    assert len(set(notes)) == 3  # unique per attempt -> the composed prompt can never repeat
    assert all(f"ATTEMPT {a}" in n for a, n in zip((2, 3, 4), notes, strict=True))
    assert "different" in notes[0].lower()  # demands a changed approach, not just uniqueness
