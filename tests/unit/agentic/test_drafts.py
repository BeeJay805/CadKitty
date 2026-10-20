"""Best-of-N draft selection weights FEATURE-COMPLETENESS above mere coherence (feature-reliability
work): a coherent draft that DROPPED a requested through-hole must not outrank a coherent draft that
built all its features, so the loop never starts from a holeless massing when a better draft exists."""

from __future__ import annotations

from forge_mind.agentic.drafts import _draft_score


def test_fully_valid_is_top() -> None:
    assert _draft_score(hard_ok=True, passed=True, features_missing=False) == 4


def test_features_present_outranks_features_missing() -> None:
    complete = _draft_score(hard_ok=True, passed=False, features_missing=False)
    dropped = _draft_score(hard_ok=True, passed=False, features_missing=True)
    assert complete == 3
    assert dropped == 2
    assert complete > dropped  # the whole point: a holes-complete draft is the better base


def test_incoherent_built_is_lowest_built_rank() -> None:
    assert _draft_score(hard_ok=False, passed=False, features_missing=True) == 1
    # features_missing is irrelevant once the draft isn't even one watertight solid
    assert _draft_score(hard_ok=False, passed=False, features_missing=False) == 1
