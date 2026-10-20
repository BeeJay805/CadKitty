"""Standards registry: pinned cited values, provenance, and the A3 byte-equivalence of routed
constants. Pure data -- no geometry, so this is part of the fast suite."""

from __future__ import annotations

import pytest

from forge_mind.dsl import standards as s
from forge_mind.dsl.standards.framework import Registry, StandardSpec

# ISO 273 clearance hole DIAMETERS (close/normal/coarse), web-verified this session.
_ISO273 = {
    "M2": (2.2, 2.4, 2.6),
    "M2.5": (2.7, 2.9, 3.1),
    "M3": (3.2, 3.4, 3.6),
    "M4": (4.3, 4.5, 4.8),
    "M5": (5.3, 5.5, 5.8),
    "M6": (6.4, 6.6, 7.0),
    "M8": (8.4, 9.0, 10.0),
}


@pytest.mark.parametrize(("size", "triple"), list(_ISO273.items()))
def test_clearance_holes_match_iso273(size: str, triple: tuple[float, float, float]) -> None:
    assert (
        s.clearance_hole(size, "close"),
        s.clearance_hole(size, "normal"),
        s.clearance_hole(size, "coarse"),
    ) == triple


def test_hex_nuts_match_iso4032() -> None:
    assert (s.hex_nut("M3").width_across_flats_mm, s.hex_nut("M3").thickness_mm) == (5.5, 2.4)
    assert (s.hex_nut("M5").width_across_flats_mm, s.hex_nut("M5").thickness_mm) == (8.0, 4.7)
    assert (s.hex_nut("M8").width_across_flats_mm, s.hex_nut("M8").thickness_mm) == (13.0, 6.8)


def test_washers_match_iso7089() -> None:
    assert (s.washer("M3").outer_diameter_mm, s.washer("M3").inner_diameter_mm) == (7.0, 3.2)
    assert (s.washer("M5").outer_diameter_mm, s.washer("M5").inner_diameter_mm) == (10.0, 5.3)


def test_bearings_match_chart() -> None:
    assert (
        s.bearing("608").bore_id_mm,
        s.bearing("608").outer_od_mm,
        s.bearing("608").width_mm,
    ) == (8.0, 22.0, 7.0)
    assert (
        s.bearing("625").bore_id_mm,
        s.bearing("625").outer_od_mm,
        s.bearing("625").width_mm,
    ) == (5.0, 16.0, 5.0)
    assert (
        s.bearing("688").bore_id_mm,
        s.bearing("688").outer_od_mm,
        s.bearing("688").width_mm,
    ) == (8.0, 16.0, 4.0)


def test_typed_api_derivations() -> None:
    assert s.tap_drill("M3") == 2.5
    assert s.counterbore("M3") == (6.0, 3.0)
    assert s.captive_nut_pocket("M3", 0.2)[0] == pytest.approx(5.7)  # across-flats + clearance
    assert s.bearing_housing_bore("608", "press") == pytest.approx(21.95)
    assert s.dowel_hole("6", "slip") == pytest.approx(6.10)


# Every category, and which ones are PROVISIONAL (not from a published standard).
# payload_object: nominal industry figures (can-size designations + derived masses),
# provisional until a physical sample is measured (ADR-0044 grounded-ingest tiering).
# drive_form: entered from public SFF-8201/8301 EXCERPTS (the 2.5" hole coordinates are
# not publicly excerpted and deliberately absent) — provisional until the full SFF PDFs
# are reviewed (ADR-0087 S2). case_fan: de-facto sizes; no governing standard exists.
# filament: datasheet/guide-cited planning values (Tg/HDT bands disagree ±5-15 °C across
# formulations; no governing standard) — provisional by construction (ADR-0087 N1).
_PROVISIONAL = {
    "heat_set_insert",
    "o_ring",
    "magnet",
    "standoff",
    "payload_object",
    "drive_form",
    "case_fan",
    "filament",
}


def test_every_spec_is_cited() -> None:
    for reg in s.ALL_REGISTRIES.values():
        for spec in reg.all():
            assert spec.source.strip(), f"{reg.name}/{spec.key} has no cited source"


@pytest.mark.parametrize("name", sorted(s.ALL_REGISTRIES))
def test_provisional_flags(name: str) -> None:
    reg = s.ALL_REGISTRIES[name]
    if name in _PROVISIONAL:
        assert all(spec.provisional for spec in reg.all()), f"{name} should be all provisional"
    elif name == "retaining_ring":
        # only the web-verified d1=4 is non-provisional; the rest are flagged.
        assert reg.get("4").provisional is False
        assert all(spec.provisional for spec in reg.all() if spec.key != "4")
    else:
        assert not any(spec.provisional for spec in reg.all()), (
            f"{name} (a real standard) should not be provisional"
        )


def test_registry_unknown_key_lists_known() -> None:
    with pytest.raises(ValueError, match="unknown metric_fastener 'M99'; known:"):
        s.FASTENERS.get("M99")


def test_registry_rejects_duplicate_keys() -> None:
    class _Tmp(StandardSpec):
        v: float

    with pytest.raises(ValueError, match="duplicate"):
        Registry("tmp", (_Tmp(key="a", source="x", v=1.0), _Tmp(key="a", source="y", v=2.0)))


def test_a3_routed_constants_are_byte_identical() -> None:
    # A3 routed these through the registry; they MUST still equal the legacy literals (byte-equ).
    from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
    from forge_mind.dsl.interfaces.m6 import M6_CLEARANCE_RADIUS_MM
    from forge_mind.dsl.interfaces.nema17 import M3_CLEARANCE_RADIUS_MM
    from forge_mind.dsl.interfaces.pi4 import M25_CLEARANCE_DIAMETER_MM, M25_CLEARANCE_RADIUS_MM

    assert M3_CLEARANCE_RADIUS_MM == 1.6  # ISO 273 close 3.2 / 2
    assert M6_CLEARANCE_RADIUS_MM == 3.3  # ISO 273 normal 6.6 / 2
    assert M25_CLEARANCE_RADIUS_MM == 1.35 and M25_CLEARANCE_DIAMETER_MM == 2.7
    assert M5_CLEARANCE_RADIUS_MM == 2.6  # FLAGGED non-standard, left unchanged


# ---------------------------------------------------------------- key coercion
# (caught live on the robotic-arm run: the agentic author wrote dowel_hole_d(6) and got
# "unknown dowel_pin 6; known: [... '6' ...]" -- int vs string-key. Coercion resolves to
# the SAME cited row; an unknown size still raises with the known list.)


def test_get_coerces_int_and_integral_float_keys() -> None:
    assert s.DOWEL_PINS.get(6) is s.DOWEL_PINS.get("6")
    assert s.DOWEL_PINS.get(6.0) is s.DOWEL_PINS.get("6")
    assert s.BEARINGS.get(608) is s.BEARINGS.get("608")


def test_get_case_insensitive_only_when_unambiguous() -> None:
    assert s.FASTENERS.get("m3") is s.FASTENERS.get("M3")


def test_has_coerces_like_get() -> None:
    assert s.DOWEL_PINS.has(6) is True
    assert s.BEARINGS.has(608) is True
    assert s.DOWEL_PINS.has(7) is False


def test_unknown_key_still_raises_with_known_list() -> None:
    with pytest.raises(ValueError, match="unknown dowel_pin"):
        s.DOWEL_PINS.get(99)


def test_duplicate_detection_unchanged() -> None:
    class _Spec(StandardSpec):
        pass

    with pytest.raises(ValueError, match="duplicate"):
        Registry(
            "t",
            (
                _Spec(key="a", source="x"),
                _Spec(key="a", source="x"),
            ),
        )
