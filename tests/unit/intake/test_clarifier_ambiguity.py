"""Ambiguity regression corpus (ProCAD's "clarify before you draw", borrowed not vendored). Hand-
authored prompts in three classes assert the deterministic ``HeuristicPlannerBrain`` does the right
thing WITHOUT an LLM:

  * under-specified  -> ASK (questions, no refusal) -- never silently guess;
  * direct-conflict  -> REFUSE (two mutually-exclusive standards of one family);
  * impossible       -> REFUSE (a hole stated larger than the body it sits in).

This is the first measurable proof that cadkitty asks-instead-of-guesses; it does not depend on the
-79.9% figure from the paper (oracle-answered, not transferable)."""

from __future__ import annotations

from forge_mind.intake.agent import HeuristicPlannerBrain
from forge_mind.intake.session import DesignSession

# Under-specified: a real request that is missing fit-critical facts -> ask, do not refuse, do not
# silently mark ready.
UNDERSPECIFIED = [
    "make a motor mount",
    "a bracket to hold something",
    "a fan mount",
    "an enclosure for a board",
    "a bearing block",
]

# Direct conflict: two members of one mutually-exclusive standard family -> refuse.
CONFLICTS = [
    "a mount for both a NEMA-17 and a NEMA-23 motor",
    "an adapter for NEMA17 and NEMA23 steppers",
]

# Geometrically impossible (both dimensions explicit) -> refuse.
IMPOSSIBLE = [
    "a 40 mm square plate with an 80 mm bore",
    "a 20 mm disc with a 30 mm hole",
    "an 8 mm hole in a 6 mm wide bar",
]

# Looks similar to the impossible set but is perfectly buildable -> must NOT refuse.
FEASIBLE = [
    "a 40 mm plate with a 5 mm hole",
    "an 80 mm diameter disc with a 5 mm bore",
]


def _analyze(text: str):  # type: ignore[no-untyped-def]
    return HeuristicPlannerBrain().analyze(DesignSession(original_text_prompt=text))


def test_underspecified_requests_ask_not_guess() -> None:
    for prompt in UNDERSPECIFIED:
        outcome = _analyze(prompt)
        assert outcome.refusal is None, f"should not refuse an under-specified prompt: {prompt!r}"
        assert outcome.ready is False, f"should not silently generate from: {prompt!r}"
        assert outcome.questions, f"should ask at least one question for: {prompt!r}"


def test_direct_conflicts_are_refused() -> None:
    for prompt in CONFLICTS:
        outcome = _analyze(prompt)
        assert outcome.refusal is not None, f"should refuse a same-family conflict: {prompt!r}"
        assert outcome.questions == [], f"a refusal asks nothing: {prompt!r}"
        assert outcome.ready is False


def test_impossible_dimensions_are_refused() -> None:
    for prompt in IMPOSSIBLE:
        outcome = _analyze(prompt)
        assert outcome.refusal is not None, f"should refuse an impossible dimension: {prompt!r}"
        assert "fit" in outcome.refusal.lower() or "cannot" in outcome.refusal.lower()


def test_feasible_lookalikes_are_not_refused() -> None:
    for prompt in FEASIBLE:
        outcome = _analyze(prompt)
        assert outcome.refusal is None, f"must not false-refuse a buildable part: {prompt!r}"


def test_nema_conflict_message_names_both_sizes() -> None:
    # Backwards-compat with the original hardcoded case: the message still names both frames.
    outcome = _analyze("a mount for both a NEMA-17 and a NEMA-23 motor")
    assert outcome.refusal is not None
    assert "NEMA-17" in outcome.refusal and "NEMA-23" in outcome.refusal
