"""The benchmark pass bar must keep its MEANING under a --trials override (audit H7).

min_passing_trials/trials is a pass RATE; a custom trial count scales the bar to that rate. Without
this, `--trials 100` against a fixed min_passing_trials=9 would PASS at a 9% pass rate, and
`--trials 5` could never pass.
"""

from __future__ import annotations

from forge_mind.bench.criteria import PassCriteria
from forge_mind.bench.runner import _required_passes


def test_default_bar_is_unchanged() -> None:
    crit = PassCriteria(trials=10, min_passing_trials=9)
    assert _required_passes(crit, 10, overridden=False) == 9


def test_override_scales_the_bar_to_the_configured_rate() -> None:
    crit = PassCriteria(trials=10, min_passing_trials=9)  # 90% required
    assert _required_passes(crit, 100, overridden=True) == 90  # not the absolute 9
    assert _required_passes(crit, 5, overridden=True) == 5  # ceil(9/10 * 5) = ceil(4.5) = 5
    assert _required_passes(crit, 20, overridden=True) == 18


def test_override_with_zero_configured_trials_is_safe() -> None:
    crit = PassCriteria(trials=1, min_passing_trials=0)
    assert _required_passes(crit, 50, overridden=True) == 0
