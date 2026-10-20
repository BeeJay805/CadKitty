from __future__ import annotations

from types import SimpleNamespace

from forge_mind.agentic.result import _disposition


def _loop(*, critic_approved: bool) -> SimpleNamespace:
    return SimpleNamespace(
        final_stl="part.stl",
        validation=SimpleNamespace(hard_ok=True),
        geometry_passed=True,
        critic_approved=critic_approved,
    )


def test_acceptance_judge_cannot_bypass_an_unresolved_loop_critic() -> None:
    verdict = SimpleNamespace(is_the_object=True)
    assert _disposition(_loop(critic_approved=False), verdict, judge_failed=False) == "rejected"


def test_critic_approved_green_part_still_ships() -> None:
    verdict = SimpleNamespace(is_the_object=True)
    assert _disposition(_loop(critic_approved=True), verdict, judge_failed=False) == "shipped"
