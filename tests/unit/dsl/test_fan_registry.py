"""Fan component registry (bench-006, ADR-0011) — the second component class.

Two hand-authored fans load with their verified facts, the mount facts are
single-sourced from the shared ``dsl/interfaces/fan.py`` module + the M3 fact (no
drift with the fan_face checker), and the registry round-trips. Parallel to
``test_board_registry.py`` -- proving boards and fans share one mechanism.
"""

from __future__ import annotations

import pytest

from forge_mind.dsl.components.fan import FanComponent
from forge_mind.dsl.components.registry import get_fan_component, known_fan_component_ids
from forge_mind.dsl.interfaces import fan as fan_facts


def test_known_ids() -> None:
    assert set(known_fan_component_ids()) == {"fan_40mm", "fan_60mm"}


def test_fan_40mm_facts() -> None:
    c = get_fan_component("fan_40mm")
    assert c.body_mm == 40.0
    assert c.mount.bolt_square_mm == 32.0
    assert c.mount.screw_clearance_radius_mm == 1.6  # M3
    assert c.mount.airflow_bore_radius_mm == 19.0
    assert c.source  # provenance present


def test_fan_60mm_facts() -> None:
    c = get_fan_component("fan_60mm")
    assert c.body_mm == 60.0
    assert c.mount.bolt_square_mm == 50.0
    assert c.mount.screw_clearance_radius_mm == 1.6  # M3
    assert c.mount.airflow_bore_radius_mm == 28.0
    assert c.source


def test_mount_facts_single_sourced_from_interface_facts() -> None:
    c40 = get_fan_component("fan_40mm")
    assert c40.mount.bolt_square_mm == fan_facts.FAN_40MM_BOLT_SQUARE_MM
    assert c40.mount.airflow_bore_radius_mm == fan_facts.FAN_40MM_AIRFLOW_BORE_RADIUS_MM
    assert c40.mount.screw_clearance_radius_mm == fan_facts.FAN_BOLT_CLEARANCE_RADIUS_MM
    c60 = get_fan_component("fan_60mm")
    assert c60.mount.bolt_square_mm == fan_facts.FAN_60MM_BOLT_SQUARE_MM
    assert c60.mount.airflow_bore_radius_mm == fan_facts.FAN_60MM_AIRFLOW_BORE_RADIUS_MM


def test_round_trip() -> None:
    c = get_fan_component("fan_60mm")
    assert FanComponent.model_validate(c.model_dump()) == c


def test_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown fan component"):
        get_fan_component("nope")
