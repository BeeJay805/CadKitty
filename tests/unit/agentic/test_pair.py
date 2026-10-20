"""Fit-pair certification (complex-parts arc P4): measured, three-state, never silent.

Deterministic: parts are built with build123d directly (no LLM), exported to STEP, and
certified against the SHIPPED ISO 286 engine. Expected bands come from the engine itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic.pair import (
    MatingContract,
    certify_fit,
    derive_mating_dim,
    measure_cylinder_diameter,
)
from forge_mind.physics.fits import parse_designation, resolve_fit

pytestmark = pytest.mark.slow


def _export(part, path: Path) -> Path:
    from build123d import export_step

    export_step(part, str(path))
    return path


def _bore_part(tmp: Path, bore_d: float) -> Path:
    from build123d import Box, BuildPart, Hole, Locations

    with BuildPart() as p:
        Box(60, 60, 20)
        with Locations((0, 0, 0)):
            Hole(radius=bore_d / 2, depth=25)
    return _export(p.part, tmp / "bore.step")


def _shaft_part(tmp: Path, shaft_d: float) -> Path:
    from build123d import BuildPart, Cylinder

    with BuildPart() as p:
        Cylinder(radius=shaft_d / 2, height=40)
    return _export(p.part, tmp / "shaft.step")


CONTRACT = MatingContract(
    designation="25 H7/g6", bore_part="a", bore_feature="bearing bore", shaft_feature="pivot shaft"
)


def test_derive_mating_dim_applies_the_shaft_mid_band() -> None:
    nominal, _hole, (letter, grade) = parse_designation("25 H7/g6")
    fit = resolve_fit(nominal, ("H", 7), (letter, grade))
    lo, hi = fit.shaft_dev_um
    expected = 25.1 + ((lo + hi) / 2.0) / 1000.0
    assert derive_mating_dim(25.1, "25 H7/g6") == pytest.approx(expected, abs=1e-4)


def test_certified_pair_within_band(tmp_path: Path) -> None:
    bore = _bore_part(tmp_path, 25.000)
    shaft = _shaft_part(tmp_path, derive_mating_dim(25.000, "25 H7/g6"))
    v = certify_fit(CONTRACT, bore, shaft)
    assert v.status == "verified", v.detail
    assert v.band_clearance_um is not None
    lo, hi = v.band_clearance_um
    assert lo <= v.measured_clearance_um <= hi


def test_line_to_line_pair_is_violated_for_a_clearance_fit(tmp_path: Path) -> None:
    bore = _bore_part(tmp_path, 25.000)
    shaft = _shaft_part(tmp_path, 25.000)  # zero clearance < the g6 minimum
    v = certify_fit(CONTRACT, bore, shaft)
    assert v.status == "violated"
    assert "OUTSIDE" in v.detail
    assert v.gate is False  # report-tier by default (FORGE_FIT_GATE unset)


def test_unmeasurable_is_unverifiable_never_a_pass(tmp_path: Path) -> None:
    from build123d import Box, BuildPart

    with BuildPart() as p:
        Box(30, 30, 30)  # no cylinders at all
    holeless = _export(p.part, tmp_path / "holeless.step")
    shaft = _shaft_part(tmp_path, 24.99)
    v = certify_fit(CONTRACT, holeless, shaft)
    assert v.status == "unverifiable"
    assert "could not be measured" in v.detail


def test_measure_rejects_a_wildly_off_feature(tmp_path: Path) -> None:
    bore = _bore_part(tmp_path, 8.0)  # nearest cylinder is 8mm; contract nominal is 25mm
    assert measure_cylinder_diameter(bore, near_mm=25.0, concave=True) is None
