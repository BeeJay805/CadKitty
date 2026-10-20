"""A small catalog of well-known hardware standards the planner CITES (never invents) so it does not
ask the user for dimensions a standard already provides.

Values are public facts (NEMA stepper face/hole patterns, T-slot extrusion profiles, common fan
sizes, Raspberry Pi mount) plus bearing facts pulled straight from the engine's own registry
(``forge_mind.dsl.standards``) so the brief can never drift from the geometry the compiler builds.
Each entry carries a ``source`` citation. ``detect_standards`` scans a request for mentions.
"""

from __future__ import annotations

import re

from forge_mind.dsl.standards import bearing
from forge_mind.intake.schema import KnownStandard, ScalarValue

_NEMA_SRC = "NEMA ICS 16-2001 stepper frame sizes (face + bolt-circle + shaft)"
_EXTRUSION_SRC = "20-series T-slot aluminium extrusion (Misumi / OpenBuilds / 8020)"
_FAN_SRC = "Standard DC axial fan frame sizes (square frame + diagonal bolt pattern)"
_PI_SRC = "Raspberry Pi 4 Model B mechanical specification"


def _std(name: str, source: str, values: dict[str, ScalarValue]) -> KnownStandard:
    return KnownStandard(name=name, source=source, values=values)


def _bearing_std(key: str) -> KnownStandard | None:
    """Build a bearing standard from the engine registry (so the facts match the geometry)."""
    try:
        b = bearing(key)
    except Exception:
        return None
    return KnownStandard(
        name=f"{key} bearing",
        source=b.source,
        values={
            "bore_id_mm": b.bore_id_mm,
            "outer_od_mm": b.outer_od_mm,
            "width_mm": b.width_mm,
        },
    )


# The static catalog (non-bearing). Bearings are added dynamically from the registry below so they
# stay byte-consistent with the compiler.
_STATIC: dict[str, KnownStandard] = {
    "NEMA-17": _std(
        "NEMA-17",
        _NEMA_SRC,
        {
            "face_size_mm": 42.3,
            "mounting_hole_pattern_mm": 31.0,
            "motor_screws": "M3",
            "shaft_diameter_mm": 5.0,
            "shaft_clearance_mm": 22.0,
        },
    ),
    "NEMA-23": _std(
        "NEMA-23",
        _NEMA_SRC,
        {
            "face_size_mm": 56.4,
            "mounting_hole_pattern_mm": 47.14,
            "motor_screws": "M5",
            "shaft_diameter_mm": 6.35,
            "shaft_clearance_mm": 38.1,
        },
    ),
    "2020 aluminum extrusion": _std(
        "2020 aluminum extrusion",
        _EXTRUSION_SRC,
        {
            "profile_mm": "20x20",
            "slot_type": "T-slot",
            "slot_width_mm": 6.0,
            "recommended_fasteners": "M5",
        },
    ),
    "2040 aluminum extrusion": _std(
        "2040 aluminum extrusion",
        _EXTRUSION_SRC,
        {
            "profile_mm": "20x40",
            "slot_type": "T-slot",
            "slot_width_mm": 6.0,
            "recommended_fasteners": "M5",
        },
    ),
    "40mm fan": _std(
        "40mm fan",
        _FAN_SRC,
        {"frame_size_mm": 40.0, "mounting_hole_pattern_mm": 32.0, "screws": "M3"},
    ),
    "60mm fan": _std(
        "60mm fan",
        _FAN_SRC,
        {"frame_size_mm": 60.0, "mounting_hole_pattern_mm": 50.0, "screws": "M4"},
    ),
    "Raspberry Pi 4": _std(
        "Raspberry Pi 4",
        _PI_SRC,
        {"mount_pattern_mm": "49 x 58", "screws": "M2.5"},
    ),
}


def known_standards() -> dict[str, KnownStandard]:
    """The full catalog (static facts + registry-backed bearings), keyed by canonical name."""
    catalog = dict(_STATIC)
    for key in ("623", "625", "608", "6900"):
        std = _bearing_std(key)
        if std is not None:
            catalog[std.name] = std
    return catalog


# Mentions -> canonical catalog name. The key is matched against the request with spaces/hyphens
# removed (so "nema 17", "nema-17", "NEMA17" all hit). Order matters: longer/more-specific first.
_PATTERNS: list[tuple[str, str]] = [
    (r"nema17", "NEMA-17"),
    (r"nema23", "NEMA-23"),
    (r"2020", "2020 aluminum extrusion"),
    (r"2040", "2040 aluminum extrusion"),
    (r"40mmfan", "40mm fan"),
    (r"60mmfan", "60mm fan"),
    (r"raspberrypi4", "Raspberry Pi 4"),
    (r"raspberrypi", "Raspberry Pi 4"),
    (r"\bpi4\b", "Raspberry Pi 4"),
    (r"608", "608 bearing"),
    (r"623", "623 bearing"),
    (r"625", "625 bearing"),
]


def detect_standards(text: str | None) -> list[KnownStandard]:
    """Return the catalog standards explicitly mentioned in ``text`` (deduped, order-stable).

    Never guesses: only returns a standard whose token appears. A bare "fan" with no size, or a
    bare "bearing" with no number, returns nothing -- the planner asks instead.
    """
    if not text:
        return []
    squished = re.sub(r"[\s\-_]+", "", text.lower())
    catalog = known_standards()
    found: list[KnownStandard] = []
    seen: set[str] = set()
    for pattern, name in _PATTERNS:
        # \b anchors are meaningless on the squished string; apply those against the raw lowercase.
        target = text.lower() if pattern.startswith("\\b") else squished
        if re.search(pattern, target) and name not in seen and name in catalog:
            seen.add(name)
            found.append(catalog[name])
    return found


__all__ = ["detect_standards", "known_standards"]
