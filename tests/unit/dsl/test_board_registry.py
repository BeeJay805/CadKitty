"""Board component registry (ADR-0007 Stage D).

The two hand-authored boards load with their facts, the Pi 4 has ports and the
generic PCB has none, and the mount facts are single-sourced from the shared
interface fact modules (no drift with the validator checker).
"""

from __future__ import annotations

import pytest

from forge_mind.dsl.components.registry import get_board_component, known_board_component_ids
from forge_mind.dsl.interfaces import pi4 as pi4_facts
from forge_mind.dsl.interfaces.nema17 import M3_CLEARANCE_RADIUS_MM


def test_known_ids() -> None:
    assert set(known_board_component_ids()) == {"raspberry_pi_4", "generic_rect_pcb"}


def test_raspberry_pi_4_has_mount_and_ports() -> None:
    c = get_board_component("raspberry_pi_4")
    assert (c.length_mm, c.width_mm, c.thickness_mm) == (85.0, 56.0, 1.5)
    assert (c.mount.hole_dx_mm, c.mount.hole_dy_mm) == (58.0, 49.0)
    assert c.mount.screw_clearance_radius_mm == 1.35
    assert c.defaults.standoff_outer_diameter_mm == 7.0
    assert {p.name for p in c.ports} == {"usb_ethernet", "usbc_hdmi"}
    assert c.source  # provenance present


def test_generic_rect_pcb_has_mount_and_no_ports() -> None:
    c = get_board_component("generic_rect_pcb")
    assert (c.length_mm, c.width_mm, c.thickness_mm) == (70.0, 50.0, 1.6)
    assert (c.mount.hole_dx_mm, c.mount.hole_dy_mm) == (60.0, 40.0)
    assert c.mount.screw_clearance_radius_mm == 1.6  # M3
    assert c.defaults.standoff_outer_diameter_mm == 8.0
    assert c.ports == ()
    assert c.source


def test_mount_facts_single_sourced_from_interface_facts() -> None:
    pi4 = get_board_component("raspberry_pi_4")
    assert pi4.mount.hole_dx_mm == pi4_facts.HOLE_RECT_DX_MM
    assert pi4.mount.hole_dy_mm == pi4_facts.HOLE_RECT_DY_MM
    assert pi4.mount.screw_clearance_radius_mm == pi4_facts.M25_CLEARANCE_RADIUS_MM
    generic = get_board_component("generic_rect_pcb")
    assert generic.mount.screw_clearance_radius_mm == M3_CLEARANCE_RADIUS_MM


def test_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown board component"):
        get_board_component("nope")
