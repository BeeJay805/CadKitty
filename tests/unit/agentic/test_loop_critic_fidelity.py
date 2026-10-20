"""The critic-fidelity keep path (measured arm R, n=8, 2026-07-18): a validator-passing part
the critic blocks scores 3, and under strict-improve every revision that implemented one NAMED
fix but did not flip looks_right in a single leap ALSO scored 3 and was discarded -- the anchor
never advanced, so multi-step fidelity progress was impossible (L-bracket 8/8 plateau, 23/40
passing iterations, 0 ships). ``_fidelity_improved`` is the equal-score tie-break that lets
that progress accumulate, and ``_combine_feedback`` must state the true blocker when every
measured gate is green."""

from __future__ import annotations

from pathlib import Path

from forge_mind.agentic.critic import DesignCritique, FeatureVerdict
from forge_mind.agentic.loop import (
    _Candidate,
    _combine_feedback,
    _fidelity_improved,
    _fidelity_rank,
    _score,
)
from forge_mind.agentic.validate import AgenticValidation


def _val(*, passed: bool, issues: list[str] | None = None) -> AgenticValidation:
    return AgenticValidation(
        passed=passed,
        hard_ok=True,
        report=None,
        body_count=1,
        watertight=True,
        volume_mm3=1000.0,
        issues=issues or [],
    )


def _crit(
    *, looks_right: bool, resemblance: str, present: int, absent: int, fixes: list[str]
) -> DesignCritique:
    verdicts = [FeatureVerdict(feature=f"f{i}", present=True) for i in range(present)]
    verdicts += [
        FeatureVerdict(feature=f"g{i}", present=False, note="missing") for i in range(absent)
    ]
    return DesignCritique(
        looks_right=looks_right,
        resemblance=resemblance,
        feature_verdicts=verdicts,
        prioritized_fixes=fixes,
        summary="s",
    )


def _cand(validation: AgenticValidation, critique: DesignCritique | None) -> _Candidate:
    return _Candidate(
        score=_score(validation, critique),
        code="c",
        stl=Path("a.stl"),
        step=Path("a.step"),
        render=None,
        validation=validation,
        critique=critique,
    )


def test_fidelity_rank_orders_features_then_resemblance() -> None:
    weak2 = _crit(looks_right=False, resemblance="weak", present=2, absent=2, fixes=["x"])
    strong1 = _crit(looks_right=False, resemblance="strong", present=1, absent=3, fixes=["x"])
    partial2 = _crit(looks_right=False, resemblance="partial", present=2, absent=2, fixes=["x"])
    assert _fidelity_rank(None) < _fidelity_rank(strong1)
    # feature presence dominates (a named fix implemented = one more feature present) ...
    assert _fidelity_rank(weak2) > _fidelity_rank(strong1)
    # ... resemblance breaks feature ties.
    assert _fidelity_rank(partial2) > _fidelity_rank(weak2)


def test_named_fix_progress_is_kept_at_equal_score() -> None:
    """The arm-R failure: hub added (one more feature present), critic still blocks -> the
    revision must REPLACE the anchor so the next fix builds on it."""
    best = _cand(
        _val(passed=True),
        _crit(
            looks_right=False,
            resemblance="weak",
            present=1,
            absent=2,
            fixes=["add the hub", "add the V-groove"],
        ),
    )
    val = _val(passed=True)
    crit = _crit(
        looks_right=False, resemblance="partial", present=2, absent=1, fixes=["add the V-groove"]
    )
    assert _score(val, crit) == best.score == 3
    assert _fidelity_improved(best, 3, val, crit) is True


def test_equal_critique_is_not_churn() -> None:
    best = _cand(
        _val(passed=True),
        _crit(looks_right=False, resemblance="partial", present=2, absent=1, fixes=["x"]),
    )
    val = _val(passed=True)
    same = _crit(looks_right=False, resemblance="partial", present=2, absent=1, fixes=["x"])
    assert _fidelity_improved(best, 3, val, same) is False


def test_fidelity_never_overrules_a_measured_gate() -> None:
    """No-Goodhart order: the tie-break fires ONLY between validator-PASSING candidates."""
    best = _cand(
        _val(passed=False),
        _crit(looks_right=False, resemblance="weak", present=0, absent=3, fixes=["x"]),
    )
    val = _val(passed=False)  # red gate: a better critique must not earn a keep
    crit = _crit(looks_right=False, resemblance="strong", present=3, absent=0, fixes=["x"])
    assert _fidelity_improved(best, best.score, val, crit) is False


def test_green_feedback_leads_with_the_fixes_and_downgrades_advisories() -> None:
    val = _val(passed=True, issues=["OVERHANG advisory: 400 mm^2 unsupported"])
    crit = _crit(
        looks_right=False,
        resemblance="partial",
        present=1,
        absent=1,
        fixes=["add the V-groove around the rim"],
    )
    fb = _combine_feedback(val, crit)
    assert fb.startswith("#1 EVERY MEASURED GATE IS GREEN")
    assert "add the V-groove around the rim" in fb
    assert "ADVISORY" in fb and "must fix" not in fb
    # the fixes come BEFORE the advisory block
    assert fb.index("V-groove") < fb.index("OVERHANG")


def test_fidelity_gap_earns_missing_feature_patience() -> None:
    """A critique naming ABSENT features on a validator-green part is a concrete gap (the arm-S
    block attractor) -- the plateau leash must extend so the restart rung gets room to run."""
    from forge_mind.agentic.loop import _MISSING_FEATURE_PATIENCE, _PLATEAU_PATIENCE, _plateau_stop

    val = _val(passed=True)
    blocked = _crit(looks_right=False, resemblance="weak", present=0, absent=3, fixes=["x"])
    clean = _crit(looks_right=False, resemblance="weak", present=3, absent=0, fixes=["x"])
    assert _plateau_stop(_PLATEAU_PATIENCE, val, blocked) is False
    assert _plateau_stop(_MISSING_FEATURE_PATIENCE, val, blocked) is True
    # no absent features -> the short leash stands (no free iterations for a vague plateau)
    assert _plateau_stop(_PLATEAU_PATIENCE, val, clean) is True
    # a RED part gets no fidelity extension (its leash is the validator's business)
    assert _plateau_stop(_PLATEAU_PATIENCE, _val(passed=False), blocked) is True


def test_fidelity_stuck_requires_green_absent_and_exhausted_revisions() -> None:
    from forge_mind.agentic.loop import _FIDELITY_RESTART_AFTER, _fidelity_stuck

    blocked = _crit(looks_right=False, resemblance="weak", present=0, absent=2, fixes=["x"])
    green_absent = _cand(_val(passed=True), blocked)
    assert _fidelity_stuck(green_absent, _FIDELITY_RESTART_AFTER) is True
    # not before both anchored-revision paths (plain + salted) have failed
    assert _fidelity_stuck(green_absent, _FIDELITY_RESTART_AFTER - 1) is False
    # a red part never restarts on fidelity (the validator owns that frontier)
    red = _cand(_val(passed=False), blocked)
    assert _fidelity_stuck(red, _FIDELITY_RESTART_AFTER) is False
    # all features present -> nothing to restart for
    clean = _cand(
        _val(passed=True),
        _crit(looks_right=False, resemblance="weak", present=2, absent=0, fixes=["x"]),
    )
    assert _fidelity_stuck(clean, _FIDELITY_RESTART_AFTER) is False
    assert _fidelity_stuck(None, _FIDELITY_RESTART_AFTER) is False


def test_fidelity_restart_note_is_mechanical_and_unique_per_attempt() -> None:
    from forge_mind.agentic.loop import _fidelity_restart_note

    crit = _crit(
        looks_right=False, resemblance="weak", present=0, absent=2, fixes=["build the legs"]
    )
    n5, n6 = _fidelity_restart_note(crit, 5), _fidelity_restart_note(crit, 6)
    assert n5 != n6  # unique per attempt -> greedy decoding cannot fixed-point across restarts
    assert "g0" in n5 and "g1" in n5  # the absent features are named verbatim
    assert "build the legs" in n5  # the critique's own fixes ride along


def test_red_feedback_is_unchanged() -> None:
    val = _val(passed=False, issues=["wall below minimum: 0.4 mm"])
    crit = _crit(
        looks_right=False, resemblance="weak", present=0, absent=2, fixes=["thicken the base"]
    )
    fb = _combine_feedback(val, crit)
    assert "VALIDATOR (must fix -- these block the part):" in fb
    assert "REVIEWER (make it look like the real object):" in fb
    assert "EVERY MEASURED GATE IS GREEN" not in fb


def test_widened_ordinal_orders_all_six_levels() -> None:
    """FORGE_CRITIC_ORDINAL (ADR-0117): the 6-point scale is a superset -- old 3-point values
    keep their relative order, and every adjacent pair of the widened scale is ordered."""
    from forge_mind.agentic.loop import _RESEMBLANCE_ORD

    levels = ["none", "weak", "fair", "partial", "good", "strong"]
    ords = [_RESEMBLANCE_ORD[level] for level in levels]
    assert ords == sorted(ords) and len(set(ords)) == 6
    # the legacy 3-point subset stays ordered exactly as before
    assert _RESEMBLANCE_ORD["weak"] < _RESEMBLANCE_ORD["partial"] < _RESEMBLANCE_ORD["strong"]


def test_ordinal_addendum_is_flag_gated(monkeypatch) -> None:
    import forge_mind.agentic.critic as critic_mod

    monkeypatch.delenv("FORGE_CRITIC_ORDINAL", raising=False)
    assert critic_mod.critic_ordinal_enabled() is False
    monkeypatch.setenv("FORGE_CRITIC_ORDINAL", "1")
    assert critic_mod.critic_ordinal_enabled() is True
    assert (
        "fair" in critic_mod._ORDINAL_ADDENDUM
        and "register progress" in critic_mod._ORDINAL_ADDENDUM
    )
