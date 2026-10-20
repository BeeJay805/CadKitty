"""Extrusion component registry (ADR-0017) — the fourth component class.

Three extrusions load with verified T-slot facts; ext_2020 is single-sourced from the
shared M5 fact + the legacy 20 mm spacing (byte-equivalence anchor); ext_3030 carries
the new M6 / 8 mm-slot facts. Parallel to ``test_fan_registry.py``.
"""

from __future__ import annotations

import pytest

from forge_mind.dsl.components.extrusion import ExtrusionComponent
from forge_mind.dsl.components.registry import (
    get_extrusion_component,
    known_extrusion_component_ids,
)
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.m6 import M6_CLEARANCE_RADIUS_MM


def test_known_ids() -> None:
    assert set(known_extrusion_component_ids()) == {"ext_2020", "ext_2040", "ext_3030"}


def test_ext_2020_facts_single_sourced() -> None:
    # Byte-equivalence anchor: ext_2020 must equal today's M5 + 20 mm constants.
    c = get_extrusion_component("ext_2020")
    assert c.slot.slot_width_mm == 6.0
    assert c.slot.bolt_clearance_radius_mm == M5_CLEARANCE_RADIUS_MM  # 2.6
    assert c.slot.default_tnut_spacing_mm == 20.0
    assert c.profile_mm == 20.0


def test_ext_2040_is_m5_like_2020() -> None:
    c = get_extrusion_component("ext_2040")
    assert c.slot.slot_width_mm == 6.0
    assert c.slot.bolt_clearance_radius_mm == M5_CLEARANCE_RADIUS_MM  # same drilled pair as 2020


def test_ext_3030_facts() -> None:
    c = get_extrusion_component("ext_3030")
    assert c.slot.slot_width_mm == 8.0
    assert c.slot.bolt_clearance_radius_mm == M6_CLEARANCE_RADIUS_MM  # M6, 3.3
    assert c.slot.default_tnut_spacing_mm == 30.0
    assert c.profile_mm == 30.0
    assert c.source


def test_round_trip() -> None:
    c = get_extrusion_component("ext_3030")
    assert ExtrusionComponent.model_validate(c.model_dump()) == c


def test_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown extrusion component"):
        get_extrusion_component("nope")
