"""grow_envelope: grow-don't-reject, honour a user hard max (pure, no geometry)."""

from __future__ import annotations

import pytest

from forge_mind.layout.envelope import grow_envelope
from forge_mind.layout.interpreter import LayoutError
from forge_mind.types import Vec3


def test_grows_only_undersized_axes() -> None:
    out = grow_envelope(Vec3(x=80.0, y=80.0, z=40.0), Vec3(x=104.8, y=60.0, z=4.0))
    assert out.x == pytest.approx(104.8)  # grew X to the derived footprint
    assert out.y == 80.0 and out.z == 40.0  # already big enough -> unchanged


def test_feasible_envelope_unchanged() -> None:
    # required fits within requested -> no growth at all (this is why existing parts are byte-equ).
    out = grow_envelope(Vec3(x=80.0, y=80.0, z=40.0), Vec3(x=50.0, y=50.0, z=4.0))
    assert (out.x, out.y, out.z) == (80.0, 80.0, 40.0)


def test_hard_max_refuses_when_required_exceeds_it() -> None:
    with pytest.raises(LayoutError, match="cannot fit within the requested max envelope"):
        grow_envelope(
            Vec3(x=80.0, y=80.0, z=40.0),
            Vec3(x=104.8, y=60.0, z=4.0),
            hard_max=Vec3(x=100.0, y=100.0, z=100.0),
        )


def test_hard_max_clamps_an_overpicked_envelope() -> None:
    # required fits the max, but the LLM over-picked the working envelope -> clamp to the user max.
    out = grow_envelope(
        Vec3(x=200.0, y=50.0, z=50.0),
        Vec3(x=60.0, y=40.0, z=4.0),
        hard_max=Vec3(x=120.0, y=120.0, z=120.0),
    )
    assert out.x == 120.0  # clamped to the hard max, not 200
    assert out.y == 50.0 and out.z == 50.0
