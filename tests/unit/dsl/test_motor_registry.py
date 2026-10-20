"""Motor component registry (ADR-0017) — the third component class.

Three hand-authored motors load with verified standard facts; the nema17 entry is
single-sourced from the shared ``dsl/interfaces/nema17.py`` constants (the
byte-equivalence anchor -- a nema17 mount must read the exact numbers the checker falls
back to); the registry round-trips. Parallel to ``test_fan_registry.py``.
"""

from __future__ import annotations

import pytest

from forge_mind.dsl.components.motor import MotorComponent
from forge_mind.dsl.components.registry import get_motor_component, known_motor_component_ids
from forge_mind.dsl.interfaces import nema17 as nema17_facts
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM


def test_known_ids() -> None:
    assert set(known_motor_component_ids()) == {"nema14", "nema17", "nema23"}


def test_nema14_facts() -> None:
    c = get_motor_component("nema14")
    assert c.body_mm == 35.2
    assert c.mount.bolt_pitch_mm == 26.0
    assert c.mount.bolt_clearance_radius_mm == 1.6  # M3
    assert c.mount.pilot_bore_radius_mm == 11.0  # 22 mm register
    assert c.source


def test_nema17_facts_single_sourced_from_interface_facts() -> None:
    # The byte-equivalence anchor: nema17 must equal the constants the checker falls back to.
    c = get_motor_component("nema17")
    assert c.mount.bolt_pitch_mm == nema17_facts.BOLT_PITCH_MM  # 31
    assert c.mount.bolt_clearance_radius_mm == nema17_facts.M3_CLEARANCE_RADIUS_MM  # 1.6
    assert c.mount.pilot_bore_radius_mm == nema17_facts.SHAFT_BORE_RADIUS_MM  # 11
    assert c.body_mm == 42.3


def test_nema23_facts() -> None:
    c = get_motor_component("nema23")
    assert c.body_mm == 56.4
    assert c.mount.bolt_pitch_mm == 47.14  # verified NEMA 23 bolt square
    assert c.mount.bolt_clearance_radius_mm == M5_CLEARANCE_RADIUS_MM  # M5, 2.6
    assert c.mount.pilot_bore_radius_mm == 19.05  # 38.1 mm register
    assert c.shaft_diameter_mm == 6.35
    assert c.source


def test_round_trip() -> None:
    c = get_motor_component("nema23")
    assert MotorComponent.model_validate(c.model_dump()) == c


def test_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown motor component"):
        get_motor_component("nope")
