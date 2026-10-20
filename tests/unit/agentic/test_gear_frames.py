"""The gear idioms' Z frame is a CONTRACT (the measured timing-gear incident): the
author-facing generators are Z-CENTERED like every build123d primitive -- the cheatsheet
taught "centered at origin" while the code built 0..thickness, and the author's correctly
reasoned hub boss sank 4 mm into the web with every gate green. The DSL compiler op keeps
its own documented [0, thickness] frame via an explicit lift.
"""

from __future__ import annotations

import pytest
from build123d import Cylinder

from forge_mind.agentic.gears import ring_gear, spur_gear, toothed_rim
from forge_mind.compiler.build_ops.gear import build_gear_solid
from forge_mind.dsl.primitives import GearSolid


def _z_span(part: object) -> tuple[float, float]:
    bb = part.bounding_box()  # type: ignore[attr-defined]
    return float(bb.min.Z), float(bb.max.Z)


def test_spur_gear_is_z_centered() -> None:
    lo, hi = _z_span(spur_gear(2.0, 24, 8.0, bore=6.0))
    assert lo == pytest.approx(-4.0, abs=1e-6)
    assert hi == pytest.approx(4.0, abs=1e-6)


def test_ring_gear_is_z_centered() -> None:
    lo, hi = _z_span(ring_gear(2.0, 48, 10.0))
    assert lo == pytest.approx(-5.0, abs=1e-6)
    assert hi == pytest.approx(5.0, abs=1e-6)


def test_toothed_rim_fully_engages_a_centered_disc() -> None:
    """The documented flywheel idiom `disc + toothed_rim(...)`: before the fix the band
    (0..width) sat HALF OFF a z-centered disc; now the union's z span equals the disc's."""
    disc = Cylinder(radius=50, height=8)
    fused = disc + toothed_rim(100.0, 24, 8.0)
    lo, hi = _z_span(fused)
    assert lo == pytest.approx(-4.0, abs=1e-3)
    assert hi == pytest.approx(4.0, abs=1e-3)
    assert len(fused.solids()) == 1


def test_dsl_gear_op_keeps_its_zero_based_contract() -> None:
    """DSL programs place gears assuming z in [0, thickness] (the op's documented frame);
    the compiler op must restore it regardless of the generator's author-facing frame."""
    op = GearSolid(op_id="g1", module=2.0, teeth=24, thickness=8.0, bore_d=6.0)
    lo, hi = _z_span(build_gear_solid(op))
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert hi == pytest.approx(8.0, abs=1e-6)


def test_bd_backend_shares_the_native_frame() -> None:
    """Backend swap must never move the part: bd_warehouse and native gears share the
    centered frame (they are documented as interchangeable)."""
    pytest.importorskip("bd_warehouse.gear")
    lo, hi = _z_span(spur_gear(2.0, 24, 8.0, bore=6.0, backend="bd_warehouse"))
    assert lo == pytest.approx(-4.0, abs=1e-3)
    assert hi == pytest.approx(4.0, abs=1e-3)
