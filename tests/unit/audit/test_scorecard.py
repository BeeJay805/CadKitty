"""The standing validator scorecard render (audit.scorecard), money-free."""

from __future__ import annotations

from forge_mind.audit.scorecard import scorecard_markdown

_RATES = {
    "n_valid_mutants": 86,
    "n_invalid_mutants": 63,
    "n_known_goods": 33,
    "false_pass_rate": 0.593,
    "false_pass_wilson95": [0.487, 0.691],
    "print_false_pass_rate": 0.593,
    "surfaced_rate": 0.395,
    "false_fail_rate": 0.03,
    "false_fail_wilson95": [0.005, 0.153],
    "per_class": {
        "off_brief_stated_dim": {
            "n": 43,
            "surfaced_rate": 0.07,
            "fp_rate": 0.91,
            "print_fp_rate": 0.91,
            "wilson95": [0.78, 0.97],
        },
        "blind_where_through": {
            "n": 6,
            "surfaced_rate": 1.0,
            "fp_rate": 0.0,
            "print_fp_rate": 0.0,
            "wilson95": [0.0, 0.39],
        },
    },
    "escapes_pinned": [{"label": "x"}],
    "false_fails_pinned": [],
}


def test_scorecard_contains_the_load_bearing_numbers() -> None:
    md = scorecard_markdown(_RATES, stamp="20260707T000000Z", corpus_glob="arbitrary_*")
    assert "surfaced (reward-readiness): 40%" in md
    assert "false-FAIL on known-goods: 3% [0, 15]%" in md
    # per-class rows, largest class first
    stated = md.index("off_brief_stated_dim")
    blind = md.index("blind_where_through")
    assert stated < blind
    assert "| off_brief_stated_dim | 43 | 7% | 91% | 91% | [78, 97]% |" in md
    # the honesty legend rides every render
    assert "Owning checks now exist" in md
    assert "escapes pinned this run: 1" in md
    assert "20260707T000000Z" in md


def test_scorecard_survives_missing_fields() -> None:
    md = scorecard_markdown({}, stamp="s", corpus_glob="g")
    assert "n/a" in md  # renders, never raises, on an empty rates block
