"""Standards-backed hardware fact library: the single source of truth for every hardware
dimension + the placeable-component vocabulary the composition engine will compose.

Typed API (the LLM picks component + size + fit; the registry supplies the geometry):
    clearance_hole("M3", fit="normal") -> 3.4      # ISO 273
    tap_drill("M3") -> 2.5                          # cut an M3 thread
    captive_nut_pocket("M3") -> (across_flats, across_corners, depth)
    bearing("608") -> Bearing(id=8, od=22, w=7)     # ISO 15
    heat_set_insert("M3").hole_diameter_mm -> 4.0   # PROVISIONAL (manufacturer)

Every spec carries ``source`` (cited) + ``provisional``. ``ALL_REGISTRIES`` enumerates every
category for tests, the provisional report, and the conformance check.
"""

from forge_mind.dsl.standards.computer import (
    CASE_FANS,
    DRIVE_FORMS,
    CaseFan,
    DriveForm,
    case_fan,
    drive_form,
)
from forge_mind.dsl.standards.fasteners import (
    FASTENERS,
    MetricFastener,
    clearance_hole,
    counterbore,
    tap_drill,
    thread_nominal,
)
from forge_mind.dsl.standards.fixings import (
    HEAT_SET_INSERTS,
    HEX_NUTS,
    WASHERS,
    HeatSetInsert,
    HexNut,
    Washer,
    captive_nut_pocket,
    heat_set_insert,
    hex_nut,
    washer,
)
from forge_mind.dsl.standards.framework import Fit, Registry, StandardSpec
from forge_mind.dsl.standards.materials import FILAMENTS, FilamentSpec, filament
from forge_mind.dsl.standards.mechanical import (
    BEARINGS,
    DOWEL_PINS,
    MAGNETS,
    O_RINGS,
    RETAINING_RINGS,
    STANDOFFS,
    Bearing,
    DowelPin,
    Magnet,
    ORing,
    RetainingRing,
    Standoff,
    bearing,
    bearing_housing_bore,
    dowel_hole,
    magnet_pocket,
)
from forge_mind.dsl.standards.payloads import (
    PAYLOAD_OBJECTS,
    PayloadObject,
    payload_object,
)

# Every category, keyed by its registry name -- the single place to enumerate the whole library.
ALL_REGISTRIES: dict[str, Registry[StandardSpec]] = {
    r.name: r  # type: ignore[misc]
    for r in (
        FASTENERS,
        HEX_NUTS,
        WASHERS,
        HEAT_SET_INSERTS,
        BEARINGS,
        DOWEL_PINS,
        O_RINGS,
        MAGNETS,
        RETAINING_RINGS,
        STANDOFFS,
        PAYLOAD_OBJECTS,
        DRIVE_FORMS,
        CASE_FANS,
        FILAMENTS,
    )
}

__all__ = [
    "ALL_REGISTRIES",
    "BEARINGS",
    "CASE_FANS",
    "DOWEL_PINS",
    "DRIVE_FORMS",
    "FASTENERS",
    "FILAMENTS",
    "HEAT_SET_INSERTS",
    "HEX_NUTS",
    "MAGNETS",
    "O_RINGS",
    "PAYLOAD_OBJECTS",
    "RETAINING_RINGS",
    "STANDOFFS",
    "WASHERS",
    "Bearing",
    "CaseFan",
    "DowelPin",
    "DriveForm",
    "FilamentSpec",
    "Fit",
    "HeatSetInsert",
    "HexNut",
    "Magnet",
    "MetricFastener",
    "ORing",
    "PayloadObject",
    "Registry",
    "RetainingRing",
    "StandardSpec",
    "Standoff",
    "Washer",
    "bearing",
    "bearing_housing_bore",
    "captive_nut_pocket",
    "case_fan",
    "clearance_hole",
    "counterbore",
    "dowel_hole",
    "drive_form",
    "filament",
    "heat_set_insert",
    "hex_nut",
    "magnet_pocket",
    "payload_object",
    "tap_drill",
    "thread_nominal",
    "washer",
]
