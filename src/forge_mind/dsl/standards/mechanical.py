"""Mechanical components: bearings, dowel pins, o-rings, magnets, retaining rings, standoffs.

Standard dims (bearings, dowel-pin diameters, retaining-ring grooves) cite their governing
standard; FIT clearances/interferences for FDM and product-specific items (magnets, heat-set-style
o-ring grooves, hex standoffs) are PROVISIONAL rules-of-thumb, flagged for verification.

Sources:
  - Deep-groove ball bearings (ISO 15 boundary dims), web-verified this session (RHD chart).
  - Dowel pins: ISO 2338 nominal diameters (m6 plain cylindrical pin).
  - Retaining rings: DIN 471 external; d1=4 web-verified this session, others PROVISIONAL.
  - O-ring groove: static-seal ~20% squeeze rule (Parker handbook / ISO 3601) -- PROVISIONAL.
  - Magnets / hex standoffs: common product sizes, no single standard -- PROVISIONAL.
"""

from __future__ import annotations

from typing import Literal

from forge_mind.dsl.standards.framework import Registry, StandardSpec

# FDM fit offsets (mm) over the standard nominal -- a documented convention, not a standard.
_PRESS_OFFSET_MM = -0.05  # interference: bore modeled slightly under nominal -> light press
_SLIP_OFFSET_MM = 0.10  # clearance: slips on/over by hand


class Bearing(StandardSpec):
    """Deep-groove ball bearing boundary dims (ISO 15)."""

    bore_id_mm: float
    outer_od_mm: float
    width_mm: float


class DowelPin(StandardSpec):
    diameter_mm: float  # ISO 2338 nominal


class ORing(StandardSpec):
    cross_section_mm: float
    groove_width_mm: float  # PROVISIONAL static-seal rule
    groove_depth_mm: float


class Magnet(StandardSpec):
    diameter_mm: float
    thickness_mm: float


class RetainingRing(StandardSpec):
    shaft_diameter_mm: float
    groove_diameter_mm: float
    groove_width_mm: float


class Standoff(StandardSpec):
    thread: str
    hex_across_flats_mm: float


_BRG = "deep-groove ball bearing (ISO 15 boundary dims), web-verified (RHD chart)"
BEARINGS: Registry[Bearing] = Registry(
    "bearing",
    (
        Bearing(key="623", source=_BRG, bore_id_mm=3.0, outer_od_mm=10.0, width_mm=4.0),
        Bearing(key="625", source=_BRG, bore_id_mm=5.0, outer_od_mm=16.0, width_mm=5.0),
        Bearing(key="626", source=_BRG, bore_id_mm=6.0, outer_od_mm=19.0, width_mm=6.0),
        Bearing(key="608", source=_BRG, bore_id_mm=8.0, outer_od_mm=22.0, width_mm=7.0),
        Bearing(key="688", source=_BRG, bore_id_mm=8.0, outer_od_mm=16.0, width_mm=4.0),
        Bearing(key="6800", source=_BRG, bore_id_mm=10.0, outer_od_mm=19.0, width_mm=5.0),
        Bearing(key="6900", source=_BRG, bore_id_mm=10.0, outer_od_mm=22.0, width_mm=6.0),
        Bearing(key="6000", source=_BRG, bore_id_mm=10.0, outer_od_mm=26.0, width_mm=8.0),
    ),
)

DOWEL_PINS: Registry[DowelPin] = Registry(
    "dowel_pin",
    tuple(
        DowelPin(
            key=str(d),
            source="ISO 2338 (m6 plain cylindrical pin) nominal diameter",
            diameter_mm=float(d),
        )
        for d in (2, 3, 4, 5, 6, 8)
    ),
)

_ORING_SRC = (
    "static-seal ~20% squeeze rule (Parker O-ring handbook / ISO 3601) -- PROVISIONAL groove"
)
O_RINGS: Registry[ORing] = Registry(
    "o_ring",
    tuple(
        ORing(
            key=f"cs{cs}",
            source=_ORING_SRC,
            provisional=True,
            cross_section_mm=cs,
            groove_width_mm=round(cs * 1.4, 2),
            groove_depth_mm=round(cs * 0.8, 2),
        )
        for cs in (1.0, 1.5, 2.0, 2.5, 3.0)
    ),
)

_MAG_SRC = "common neodymium disc magnet size -- PROVISIONAL (no single standard)"
MAGNETS: Registry[Magnet] = Registry(
    "magnet",
    (
        Magnet(key="5x2", source=_MAG_SRC, provisional=True, diameter_mm=5.0, thickness_mm=2.0),
        Magnet(key="6x3", source=_MAG_SRC, provisional=True, diameter_mm=6.0, thickness_mm=3.0),
        Magnet(key="8x3", source=_MAG_SRC, provisional=True, diameter_mm=8.0, thickness_mm=3.0),
        Magnet(key="10x3", source=_MAG_SRC, provisional=True, diameter_mm=10.0, thickness_mm=3.0),
    ),
)

_RR_VERIFIED = "DIN 471 external retaining ring, web-verified this session"
_RR_PROVISIONAL = "DIN 471 external retaining ring -- PROVISIONAL (not web-verified this session)"
RETAINING_RINGS: Registry[RetainingRing] = Registry(
    "retaining_ring",
    (
        RetainingRing(
            key="3",
            source=_RR_PROVISIONAL,
            provisional=True,
            shaft_diameter_mm=3.0,
            groove_diameter_mm=2.8,
            groove_width_mm=0.6,
        ),
        RetainingRing(
            key="4",
            source=_RR_VERIFIED,
            shaft_diameter_mm=4.0,
            groove_diameter_mm=3.8,
            groove_width_mm=0.5,
        ),
        RetainingRing(
            key="5",
            source=_RR_PROVISIONAL,
            provisional=True,
            shaft_diameter_mm=5.0,
            groove_diameter_mm=4.8,
            groove_width_mm=0.7,
        ),
        RetainingRing(
            key="6",
            source=_RR_PROVISIONAL,
            provisional=True,
            shaft_diameter_mm=6.0,
            groove_diameter_mm=5.7,
            groove_width_mm=0.8,
        ),
        RetainingRing(
            key="8",
            source=_RR_PROVISIONAL,
            provisional=True,
            shaft_diameter_mm=8.0,
            groove_diameter_mm=7.6,
            groove_width_mm=0.9,
        ),
        RetainingRing(
            key="10",
            source=_RR_PROVISIONAL,
            provisional=True,
            shaft_diameter_mm=10.0,
            groove_diameter_mm=9.6,
            groove_width_mm=1.15,
        ),
    ),
)

_STANDOFF_SRC = "common metal hex standoff (across-flats) -- PROVISIONAL (product convention, no single standard)"
STANDOFFS: Registry[Standoff] = Registry(
    "standoff",
    (
        Standoff(
            key="M3", source=_STANDOFF_SRC, provisional=True, thread="M3", hex_across_flats_mm=5.5
        ),
        Standoff(
            key="M4", source=_STANDOFF_SRC, provisional=True, thread="M4", hex_across_flats_mm=7.0
        ),
        Standoff(
            key="M5", source=_STANDOFF_SRC, provisional=True, thread="M5", hex_across_flats_mm=8.0
        ),
    ),
)


def bearing(designation: str) -> Bearing:
    return BEARINGS.get(designation)


def bearing_housing_bore(designation: str, fit: Literal["press", "slip"] = "press") -> float:
    """Housing-bore DIAMETER (mm) to seat a bearing's outer race. The FIT offset is an FDM
    convention (PROVISIONAL); the bearing OD is standard."""
    od = BEARINGS.get(designation).outer_od_mm
    return round(od + (_PRESS_OFFSET_MM if fit == "press" else _SLIP_OFFSET_MM), 3)


def dowel_hole(size: str, fit: Literal["press", "slip"] = "press") -> float:
    """Hole DIAMETER (mm) for a dowel pin. The FIT offset is an FDM convention (PROVISIONAL)."""
    d = DOWEL_PINS.get(size).diameter_mm
    return round(d + (_PRESS_OFFSET_MM if fit == "press" else _SLIP_OFFSET_MM), 3)


def magnet_pocket(size: str, clearance_mm: float = 0.1) -> tuple[float, float]:
    """(diameter, depth) mm for a slip-fit magnet pocket (PROVISIONAL fit)."""
    m = MAGNETS.get(size)
    return round(m.diameter_mm + clearance_mm, 3), m.thickness_mm


__all__ = [
    "BEARINGS",
    "DOWEL_PINS",
    "MAGNETS",
    "O_RINGS",
    "RETAINING_RINGS",
    "STANDOFFS",
    "Bearing",
    "DowelPin",
    "Magnet",
    "ORing",
    "RetainingRing",
    "Standoff",
    "bearing",
    "bearing_housing_bore",
    "dowel_hole",
    "magnet_pocket",
]
