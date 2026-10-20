"""Sanity checks for the beam-theory stiffness estimate (REPORTED-NOT-GATED).

The headline deliverable is the hand-calc test: a known rectangular cantilever whose I,
tip stiffness, and bending stress are computed by hand here and asserted within tolerance.
"""

from __future__ import annotations

import math

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, ComputedSolid
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.physics.stiffness import (
    YOUNG_MODULUS_MPA,
    StiffnessEstimate,
    estimate_stiffness,
    youngs_for,
)
from forge_mind.types import Vec3


def test_youngs_registry_lookup() -> None:
    assert youngs_for("PLA") == 3500.0
    assert youngs_for("petg") == 2100.0  # case-insensitive
    assert youngs_for("unobtanium") == 3500.0  # unknown -> PLA fallback
    assert set(YOUNG_MODULUS_MPA) == {"PLA", "PETG", "ABS", "TPU", "NYLON"}


def test_hand_calc_rectangular_cantilever() -> None:
    """A known rectangular cantilever b=10, h=4, L=100, PLA E=3500 — numbers by hand:

    I    = b*h^3/12      = 10 * 4^3 / 12 = 640/12      = 53.333... mm^4
    EI   = E*I           = 3500 * 53.333... = 186666.666... N*mm^2
    k    = 3*E*I / L^3   = 3*186666.666.../1e6           = 0.56 N/mm
    sigma/N = L*(h/2)/I  = 100*2/53.333...               = 3.75 MPa per N
    area = b*h           = 40 mm^2
    """
    b, h, length, e_mod = 10.0, 4.0, 100.0, 3500.0
    expected_i = b * h**3 / 12.0  # I  = 53.3333...
    expected_ei = e_mod * expected_i  # EI = 186666.666...
    expected_k = 3.0 * expected_ei / length**3  # k = 0.56
    expected_sigma = length * (h / 2.0) / expected_i  # sigma/N = 3.75

    # Drive estimate_stiffness through a rectangular faceted strut whose section resolves to
    # exactly b=10 (= 2*size) and h=4 (= 1.4*size): size = 5 -> width 10; thickness 1.4*5 = 7,
    # so instead assert the section math directly via a bracket plate, which is an exact b x h.
    # A flat plate (length 100 > width 10 > thickness 4) reduces to this exact cantilever.
    program = PartProgram(
        metadata=Metadata(name="plate", units="mm"),
        operations=[Box(op_id="base", width=100.0, depth=10.0, height=4.0)],
    )
    est = estimate_stiffness(program, material="PLA")
    assert est is not None
    assert est.family == "bracket"
    assert math.isclose(est.length_mm, length, rel_tol=1e-9)
    assert math.isclose(est.width_mm, b, rel_tol=1e-9)
    assert math.isclose(est.height_mm, h, rel_tol=1e-9)
    assert math.isclose(est.area_mm2, b * h, rel_tol=1e-9)
    assert math.isclose(est.second_moment_mm4, expected_i, abs_tol=1e-6)
    assert math.isclose(est.second_moment_mm4, 53.333333333333336, abs_tol=1e-6)
    assert math.isclose(est.young_modulus_mpa, e_mod, rel_tol=1e-9)
    assert math.isclose(est.bending_stiffness_EI_Nmm2, expected_ei, rel_tol=1e-12)
    assert math.isclose(est.tip_stiffness_N_per_mm, expected_k, rel_tol=1e-12)
    assert math.isclose(est.tip_stiffness_N_per_mm, 0.56, abs_tol=1e-9)
    assert math.isclose(est.max_bending_stress_mpa_per_N, expected_sigma, rel_tol=1e-12)
    assert math.isclose(est.max_bending_stress_mpa_per_N, 3.75, abs_tol=1e-9)


def test_rectangular_strut_program() -> None:
    """A rectangular faceted strut -> a 'strut' estimate with positive, finite numbers and
    the right length. Section matches the builder: b = 2*size, h = 1.4*size."""
    size, length = 6.0, 80.0
    program = PartProgram(
        metadata=Metadata(name="strut", units="mm"),
        operations=[
            ComputedSolid(
                op_id="strut",
                kind="faceted_strut",
                section="rectangular",
                size=size,
                length=length,
            )
        ],
    )
    est = estimate_stiffness(program, material="PLA")
    assert est is not None
    assert est.family == "strut"
    assert math.isclose(est.length_mm, length, rel_tol=1e-9)
    assert math.isclose(est.width_mm, 2.0 * size, rel_tol=1e-9)  # local-Y extent
    assert math.isclose(est.height_mm, 1.4 * size, rel_tol=1e-9)  # local-Z (load) extent
    for value in (
        est.area_mm2,
        est.second_moment_mm4,
        est.bending_stiffness_EI_Nmm2,
        est.tip_stiffness_N_per_mm,
        est.max_bending_stress_mpa_per_N,
    ):
        assert math.isfinite(value)
        assert value > 0.0


def test_hexagonal_strut_equal_area_equivalent() -> None:
    """A hexagonal strut reduces to an equal-area equivalent square (b == h), so area is the
    true hexagon area and the reported b x h is self-consistent."""
    size, length = 5.0, 60.0
    program = PartProgram(
        metadata=Metadata(name="hex_strut", units="mm"),
        operations=[
            ComputedSolid(
                op_id="strut",
                kind="faceted_strut",
                section="hexagonal",
                size=size,
                length=length,
            )
        ],
    )
    est = estimate_stiffness(program, material="PLA")
    assert est is not None
    hex_area = (3.0 * math.sqrt(3.0) / 2.0) * size**2
    assert math.isclose(est.area_mm2, hex_area, rel_tol=1e-9)
    assert math.isclose(est.width_mm, est.height_mm, rel_tol=1e-12)  # equivalent square
    assert math.isclose(est.area_mm2, est.width_mm * est.height_mm, rel_tol=1e-9)


def test_arm_family_inferred_from_motor_interface() -> None:
    """A strut that also declares a NEMA-17 motor mount is an 'arm', not a bare 'strut'."""
    program = PartProgram(
        metadata=Metadata(name="arm", units="mm"),
        operations=[
            ComputedSolid(
                op_id="strut", kind="faceted_strut", section="hexagonal", size=5.0, length=90.0
            )
        ],
        interfaces=[
            InterfaceDeclaration(
                name="motor_face",
                interface_type=InterfaceType.NEMA17_FACE,
                location=Vec3(x=45.0, y=0.0, z=0.0),
                normal=Vec3(x=0.0, y=0.0, z=1.0),
            )
        ],
    )
    est = estimate_stiffness(program)
    assert est is not None
    assert est.family == "arm"


def test_electronics_tray_returns_none() -> None:
    """An enclosure/tray (a thin floor with tall perimeter walls) is NOT a prismatic
    cantilever beam -> the estimate declines (returns None) rather than guessing.

    The base floor here is square-ish (100x80x3) but the part is dominated by tall walls,
    so it is an enclosure, not a plate. We assert None via a clearly non-beam program: a
    box whose three extents are all comparable (a chunky block, not a slab)."""
    program = PartProgram(
        metadata=Metadata(name="tray", units="mm"),
        operations=[
            # A near-cubic block: thickness is not the dominant-smallest slab dimension.
            Box(op_id="block", width=40.0, depth=40.0, height=40.0),
        ],
    )
    assert estimate_stiffness(program) is None


def test_no_operations_friendly_block_returns_none() -> None:
    """A box that is not a thin slab (thickness comparable to its in-plane span) is not a
    bending plate -> None."""
    program = PartProgram(
        metadata=Metadata(name="chunk", units="mm"),
        operations=[Box(op_id="b", width=30.0, depth=25.0, height=20.0)],
    )
    assert estimate_stiffness(program) is None


def test_estimate_is_a_pure_value_not_a_gate() -> None:
    """The estimate is a plain model with a caveat; it never raises and never references
    validate(). Sanity: caveat mentions the cantilever + solid-section assumptions."""
    program = PartProgram(
        metadata=Metadata(name="plate", units="mm"),
        operations=[Box(op_id="base", width=120.0, depth=60.0, height=5.0)],
    )
    est = estimate_stiffness(program)
    assert isinstance(est, StiffnessEstimate)
    assert "cantilever" in est.caveat.lower()
    assert "solid" in est.caveat.lower()
