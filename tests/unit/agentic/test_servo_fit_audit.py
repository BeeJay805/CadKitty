"""Servo-fit audit pattern matchers: pinned to the CITED + FOUNDER-VERIFIED dims.

The horn bolt circle is 14.0 mm by physical caliper (founder, 2026-07-22) -- the
registry's prior 17.0 was sourced-and-wrong, which is the whole reason this audit
measures geometry instead of trusting any layer above it. Synthesized specimens per
ADR-0124 (the archive holds zero true servo-interface positives)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from forge_mind.agentic.servo_fit_audit import (
    Hole,
    match_horn_pattern,
    match_tab_pattern,
    servo_fit_findings,
)

_Z = (0.0, 0.0, 1.0)


def _hole(x: float, y: float, d: float, axis=_Z) -> Hole:
    return Hole(point=(x, y, 0.0), axis=axis, diameter=d)


def _horn_good() -> list[Hole]:
    # The known-good hub: center bore d8.1 + 4x d2.2 on the VERIFIED d14.0 circle.
    r = 14.0 / 2.0
    holes = [_hole(0.0, 0.0, 8.1)]
    for k in range(4):
        a = math.radians(90 * k + 45)
        holes.append(_hole(r * math.cos(a), r * math.sin(a), 2.2))
    return holes


def _tab_good() -> list[Hole]:
    # The cited 49.5 x 10.0 rectangle at d4.5.
    return [_hole(sx * 49.5 / 2.0, sy * 10.0 / 2.0, 4.5) for sx in (-1, 1) for sy in (-1, 1)]


def test_known_good_horn_hub_passes() -> None:
    assert match_horn_pattern(_horn_good()) is None


def test_solid_disc_gates() -> None:
    # The known-bad specimen: no holes at all (the desk_grommet class).
    assert match_horn_pattern([]) is not None


def test_prior_wrong_17mm_circle_gates() -> None:
    # Screws on the OLD sourced-and-wrong d17.0 circle must NOT match the verified 14.0.
    r = 17.0 / 2.0
    holes = [_hole(0.0, 0.0, 8.1)] + [
        _hole(r * math.cos(math.radians(90 * k)), r * math.sin(math.radians(90 * k)), 2.2)
        for k in range(4)
    ]
    assert match_horn_pattern(holes) is not None


def test_horn_pattern_is_frame_invariant() -> None:
    # Same hub, arbitrary axis direction and offset along it: axis-perpendicular
    # distances must carry the match.
    ax = (0.6, 0.0, 0.8)
    r = 14.0 / 2.0
    holes = [Hole(point=(3.0 * ax[0], 3.0 * ax[1], 3.0 * ax[2]), axis=ax, diameter=8.1)]
    # Build the ring in a plane perpendicular to ax.
    u = (0.8, 0.0, -0.6)
    v = (0.0, 1.0, 0.0)
    for k in range(4):
        a = math.radians(90 * k)
        p = tuple(r * (math.cos(a) * u[i] + math.sin(a) * v[i]) for i in range(3))
        holes.append(Hole(point=p, axis=ax, diameter=2.2))
    assert match_horn_pattern(holes) is None


def test_known_good_tab_rectangle_passes() -> None:
    assert match_tab_pattern(_tab_good()) is None


def test_wrong_span_tab_gates() -> None:
    # A 40 x 10 rectangle is NOT the cited 49.5 x 10.0.
    holes = [_hole(sx * 20.0, sy * 5.0, 4.5) for sx in (-1, 1) for sy in (-1, 1)]
    assert match_tab_pattern(holes) is not None


def test_three_holes_are_not_a_pattern() -> None:
    assert match_tab_pattern(_tab_good()[:3]) is not None


def test_undeclared_briefs_are_out_of_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_SERVO_FIT_AUDIT", "1")
    brief = SimpleNamespace(features=[SimpleNamespace(name="arm_link", standard=None)])
    assert servo_fit_findings(brief, part=None) == []


def test_flag_off_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_SERVO_FIT_AUDIT", raising=False)
    brief = SimpleNamespace(features=[SimpleNamespace(name="horn_hub", standard="MG996R horn")])
    assert servo_fit_findings(brief, part=None) == []
