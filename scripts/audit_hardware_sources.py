"""A0 audit ($0, offline): where do hardware dimensions come from today, and where is the
single-source-of-truth missing or inconsistent?

Inventories every existing hardware constant by source module and checks the fastener clearances
against ISO 273 (the authoritative table A1 will own). Findings:
  - clearances are scattered as bare module constants (m5.py, m6.py, nema17.py, pi4.py) -- no
    cross-category registry, no fit-class label, no cited source.
  - fit classes are INCONSISTENT: M2.5 + M3 are ISO 273 fine (close), M6 is medium (normal).
  - M5 (5.2 mm) matches NO ISO 273 class (tighter than fine 5.3) -> a non-standard value to FLAG.
  - whole categories the composition engine will need (nuts, washers, bearings, heat-set inserts,
    o-rings, dowels, magnets, retaining rings, standoffs) have NO source of truth at all.

Run: uv run python scripts/audit_hardware_sources.py
"""

from __future__ import annotations

from forge_mind.dsl.components.registry import NEMA_14, NEMA_23
from forge_mind.dsl.interfaces.fan import FAN_40MM_BOLT_SQUARE_MM, FAN_60MM_BOLT_SQUARE_MM
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.m6 import M6_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.nema17 import (
    BOLT_PITCH_MM,
    M3_CLEARANCE_RADIUS_MM,
    SHAFT_BORE_RADIUS_MM,
)
from forge_mind.dsl.interfaces.pi4 import M25_CLEARANCE_RADIUS_MM

# ISO 273:1979 metric clearance hole DIAMETERS (mm): fine / medium / coarse.
_ISO273 = {
    "M2.5": (2.7, 2.9, 3.1),
    "M3": (3.2, 3.4, 3.6),
    "M5": (5.3, 5.5, 5.8),
    "M6": (6.4, 6.6, 7.0),
}
# (size, existing clearance RADIUS, source module)
_EXISTING_CLEARANCE = [
    ("M2.5", M25_CLEARANCE_RADIUS_MM, "dsl/interfaces/pi4.py"),
    ("M3", M3_CLEARANCE_RADIUS_MM, "dsl/interfaces/nema17.py"),
    ("M5", M5_CLEARANCE_RADIUS_MM, "dsl/interfaces/m5.py"),
    ("M6", M6_CLEARANCE_RADIUS_MM, "dsl/interfaces/m6.py"),
]
_FIT_NAMES = ("close (ISO fine)", "normal (ISO medium)", "coarse")
_MISSING = [
    "hex nuts + captive pockets",
    "washers",
    "heat-set inserts",
    "bearings (seat fit)",
    "o-ring grooves",
    "dowel/locating pins (press/slip)",
    "standoffs / threaded rod",
    "magnet pockets",
    "retaining rings (groove)",
]


def _classify(size: str, dia: float) -> str:
    triple = _ISO273.get(size)
    if triple is None:
        return "no ISO 273 reference"
    for name, std in zip(_FIT_NAMES, triple, strict=True):
        if abs(dia - std) < 1e-6:
            return f"MATCH {name} ({std} mm)"
    return f"*** NO ISO 273 MATCH (nearest fine {triple[0]} mm) -> FLAG"


def main() -> None:
    print("=== A0 hardware-dimension audit ===\n")
    print("Fastener clearances (scattered across modules; no fit label; no cited source):")
    for size, radius, src in _EXISTING_CLEARANCE:
        dia = 2.0 * radius
        print(f"  {size:5} r={radius:<5} dia={dia:<5} [{src}]  -> {_classify(size, dia)}")

    print("\nOther hardware facts (also bare constants, no unified registry):")
    print(
        f"  NEMA bolt squares: 14={NEMA_14.mount.bolt_pitch_mm} 17={BOLT_PITCH_MM} "
        f"23={NEMA_23.mount.bolt_pitch_mm}; register bore r={SHAFT_BORE_RADIUS_MM} "
        f"[dsl/components/registry.py, dsl/interfaces/nema17.py]"
    )
    print(
        f"  fan bolt squares: 40mm={FAN_40MM_BOLT_SQUARE_MM} 60mm={FAN_60MM_BOLT_SQUARE_MM} "
        f"[dsl/interfaces/fan.py]"
    )
    print(
        f"  motor clearances: M3(14/17)={NEMA_14.mount.bolt_clearance_radius_mm} "
        f"M5(23)={NEMA_23.mount.bolt_clearance_radius_mm} [registry wraps the bare constants]"
    )

    print("\nMISSING entirely (no source of truth) -- the composition-engine vocabulary gap:")
    for m in _MISSING:
        print(f"  - {m}")

    print(
        "\nVERDICT: no single source of truth across categories; clearances inconsistent "
        "(M2.5/M3 close, M6 normal); M5 non-standard (FLAG). A1 = one typed, cited registry; "
        "A3 routes the matches + flags the rest; A4 adds the conformance check."
    )


if __name__ == "__main__":
    main()
