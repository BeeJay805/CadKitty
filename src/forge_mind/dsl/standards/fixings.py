"""Fixings: hex nuts + captive-nut pockets (ISO 4032), plain washers (ISO 7089), and heat-set
threaded inserts (manufacturer guidance -> PROVISIONAL).

Sources:
  - ISO 4032 hexagon nuts, style 1, grade A/B: width across flats (s, max), thickness (m, max),
    width across corners (e). (web-verified this session)
  - ISO 7089 plain washers, 200 HV, form A: outer dia / inner dia / thickness.
  - Heat-set inserts: no ISO standard -- recommended pilot-hole dia + length are per-manufacturer
    (accu-components / Spirol) for FDM PLA/PETG. Marked PROVISIONAL: verify against the insert
    datasheet + a test print. https://accu-components.com/us/p/488-threaded-insert-hole-size-charts
"""

from __future__ import annotations

from forge_mind.dsl.standards.framework import Registry, StandardSpec

_NUT_SRC = "ISO 4032 (hex nut style 1, grade A/B): s/m/e"
_WASHER_SRC = "ISO 7089 (plain washer, 200 HV, form A): od/id/thickness"
_INSERT_SRC = "manufacturer (accu-components / Spirol), FDM PLA/PETG -- PROVISIONAL: verify per datasheet + test print"


class HexNut(StandardSpec):
    width_across_flats_mm: float  # s (max)
    width_across_corners_mm: float  # e
    thickness_mm: float  # m (max)


class Washer(StandardSpec):
    outer_diameter_mm: float
    inner_diameter_mm: float
    thickness_mm: float


class HeatSetInsert(StandardSpec):
    hole_diameter_mm: float  # the printed pilot hole the insert melts into
    length_mm: float
    min_boss_wall_mm: float  # min material around the insert (else the boss can crack)


HEX_NUTS: Registry[HexNut] = Registry(
    "hex_nut",
    (
        HexNut(
            key="M2",
            source=_NUT_SRC,
            width_across_flats_mm=4.0,
            width_across_corners_mm=4.32,
            thickness_mm=1.6,
        ),
        HexNut(
            key="M2.5",
            source=_NUT_SRC,
            width_across_flats_mm=5.0,
            width_across_corners_mm=5.45,
            thickness_mm=2.0,
        ),
        HexNut(
            key="M3",
            source=_NUT_SRC,
            width_across_flats_mm=5.5,
            width_across_corners_mm=6.01,
            thickness_mm=2.4,
        ),
        HexNut(
            key="M4",
            source=_NUT_SRC,
            width_across_flats_mm=7.0,
            width_across_corners_mm=7.66,
            thickness_mm=3.2,
        ),
        HexNut(
            key="M5",
            source=_NUT_SRC,
            width_across_flats_mm=8.0,
            width_across_corners_mm=8.79,
            thickness_mm=4.7,
        ),
        HexNut(
            key="M6",
            source=_NUT_SRC,
            width_across_flats_mm=10.0,
            width_across_corners_mm=11.05,
            thickness_mm=5.2,
        ),
        HexNut(
            key="M8",
            source=_NUT_SRC,
            width_across_flats_mm=13.0,
            width_across_corners_mm=14.38,
            thickness_mm=6.8,
        ),
    ),
)

WASHERS: Registry[Washer] = Registry(
    "washer",
    (
        Washer(
            key="M2",
            source=_WASHER_SRC,
            outer_diameter_mm=5.0,
            inner_diameter_mm=2.2,
            thickness_mm=0.3,
        ),
        Washer(
            key="M2.5",
            source=_WASHER_SRC,
            outer_diameter_mm=6.0,
            inner_diameter_mm=2.7,
            thickness_mm=0.5,
        ),
        Washer(
            key="M3",
            source=_WASHER_SRC,
            outer_diameter_mm=7.0,
            inner_diameter_mm=3.2,
            thickness_mm=0.5,
        ),
        Washer(
            key="M4",
            source=_WASHER_SRC,
            outer_diameter_mm=9.0,
            inner_diameter_mm=4.3,
            thickness_mm=0.8,
        ),
        Washer(
            key="M5",
            source=_WASHER_SRC,
            outer_diameter_mm=10.0,
            inner_diameter_mm=5.3,
            thickness_mm=1.0,
        ),
        Washer(
            key="M6",
            source=_WASHER_SRC,
            outer_diameter_mm=12.0,
            inner_diameter_mm=6.4,
            thickness_mm=1.6,
        ),
        Washer(
            key="M8",
            source=_WASHER_SRC,
            outer_diameter_mm=16.0,
            inner_diameter_mm=8.4,
            thickness_mm=1.6,
        ),
    ),
)

HEAT_SET_INSERTS: Registry[HeatSetInsert] = Registry(
    "heat_set_insert",
    (
        HeatSetInsert(
            key="M2",
            source=_INSERT_SRC,
            provisional=True,
            hole_diameter_mm=3.2,
            length_mm=3.0,
            min_boss_wall_mm=2.0,
        ),
        HeatSetInsert(
            key="M2.5",
            source=_INSERT_SRC,
            provisional=True,
            hole_diameter_mm=3.5,
            length_mm=4.0,
            min_boss_wall_mm=2.0,
        ),
        HeatSetInsert(
            key="M3",
            source=_INSERT_SRC,
            provisional=True,
            hole_diameter_mm=4.0,
            length_mm=4.0,
            min_boss_wall_mm=2.0,
        ),
        HeatSetInsert(
            key="M4",
            source=_INSERT_SRC,
            provisional=True,
            hole_diameter_mm=5.25,
            length_mm=5.7,
            min_boss_wall_mm=2.0,
        ),
        HeatSetInsert(
            key="M5",
            source=_INSERT_SRC,
            provisional=True,
            hole_diameter_mm=6.4,
            length_mm=5.8,
            min_boss_wall_mm=2.0,
        ),
    ),
)


def hex_nut(size: str) -> HexNut:
    return HEX_NUTS.get(size)


def captive_nut_pocket(size: str, clearance_mm: float = 0.2) -> tuple[float, float, float]:
    """(across_flats, across_corners, depth) mm for a pocket that captures a hex nut from the side
    (a hex/slot recess of WAF + clearance, depth = nut thickness + clearance)."""
    n = HEX_NUTS.get(size)
    return (
        n.width_across_flats_mm + clearance_mm,
        n.width_across_corners_mm + clearance_mm,
        n.thickness_mm + clearance_mm,
    )


def washer(size: str) -> Washer:
    return WASHERS.get(size)


def heat_set_insert(size: str) -> HeatSetInsert:
    return HEAT_SET_INSERTS.get(size)


__all__ = [
    "HEAT_SET_INSERTS",
    "HEX_NUTS",
    "WASHERS",
    "HeatSetInsert",
    "HexNut",
    "Washer",
    "captive_nut_pocket",
    "heat_set_insert",
    "hex_nut",
]
