"""Component registry — boards (ADR-0007 Stage D) + fans (bench-006, ADR-0011).

``get_board_component(id)`` / ``get_fan_component(id)`` resolve an id to its
component; the ``known_*_ids()`` helpers list what exists. Component facts WRAP the
existing single-source interface facts (``dsl/interfaces/{pi4,nema17,fan}.py``) so
mounting dimensions + screw radii are defined once and the checker validates against
the same numbers.

Two component CLASSES now share this one mechanism: boards (hosted by the electronics
tray) and fans (hosted by the adapter_plate fan path). Both are flat ``frozen``
records resolved by id -- NOT a universal ``ComponentDefinition`` union (ADR-0007
deferred that; two classes is the earned state). Motors / bearings / Arduino /
external CAD stay deferred.
"""

from __future__ import annotations

from forge_mind.dsl.components.board import (
    BoardComponent,
    BoardDefaults,
    BoardMountFacts,
    BoardPortFact,
    PortEdge,
)
from forge_mind.dsl.components.extrusion import ExtrusionComponent, ExtrusionSlotFacts
from forge_mind.dsl.components.fan import FanComponent, FanMountFacts
from forge_mind.dsl.components.motor import MotorComponent, MotorMountFacts
from forge_mind.dsl.interfaces.fan import (
    FAN_40MM_AIRFLOW_BORE_RADIUS_MM,
    FAN_40MM_BOLT_SQUARE_MM,
    FAN_60MM_AIRFLOW_BORE_RADIUS_MM,
    FAN_60MM_BOLT_SQUARE_MM,
    FAN_BOLT_CLEARANCE_RADIUS_MM,
)
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.m6 import M6_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.nema17 import (
    BOLT_PITCH_MM,
    M3_CLEARANCE_RADIUS_MM,
    SHAFT_BORE_RADIUS_MM,
)
from forge_mind.dsl.interfaces.pi4 import (
    HOLE_RECT_DX_MM,
    HOLE_RECT_DY_MM,
    M25_CLEARANCE_RADIUS_MM,
)

RASPBERRY_PI_4 = BoardComponent(
    component_id="raspberry_pi_4",
    display_name="Raspberry Pi 4 Model B",
    length_mm=85.0,
    width_mm=56.0,
    thickness_mm=1.5,
    mount=BoardMountFacts(
        hole_dx_mm=HOLE_RECT_DX_MM,  # 58 (shared fact)
        hole_dy_mm=HOLE_RECT_DY_MM,  # 49 (shared fact)
        screw_clearance_radius_mm=M25_CLEARANCE_RADIUS_MM,  # M2.5, 1.35 (shared fact)
        edge_inset_mm=3.5,
    ),
    defaults=BoardDefaults(standoff_height_mm=4.0, standoff_outer_diameter_mm=7.0),
    ports=(
        BoardPortFact(name="usb_ethernet", edge=PortEdge.LONG_PLUS_Y, width_mm=50.0),
        BoardPortFact(name="usbc_hdmi", edge=PortEdge.SHORT_MINUS_X, width_mm=44.0),
    ),
    source="Raspberry Pi 4 Model B mechanical drawing (raspberrypi.com).",
)

GENERIC_RECT_PCB = BoardComponent(
    component_id="generic_rect_pcb",
    display_name="Generic rectangular PCB",
    length_mm=70.0,
    width_mm=50.0,
    thickness_mm=1.6,
    mount=BoardMountFacts(
        hole_dx_mm=60.0,
        hole_dy_mm=40.0,
        screw_clearance_radius_mm=M3_CLEARANCE_RADIUS_MM,  # M3, 1.6 (shared fact)
        edge_inset_mm=3.5,
    ),
    defaults=BoardDefaults(standoff_height_mm=4.0, standoff_outer_diameter_mm=8.0),
    ports=(),  # a plain PCB has no fixed connector banks -> closed tray
    source="Synthetic reference (ADR-0007 Stage C/D): a plain rectangular PCB, M3 corner holes.",
)

_REGISTRY: dict[str, BoardComponent] = {
    RASPBERRY_PI_4.component_id: RASPBERRY_PI_4,
    GENERIC_RECT_PCB.component_id: GENERIC_RECT_PCB,
}


def get_board_component(component_id: str) -> BoardComponent:
    """Return the board component for ``component_id``; raise ValueError if unknown."""
    try:
        return _REGISTRY[component_id]
    except KeyError:
        raise ValueError(
            f"unknown board component {component_id!r}; known: {sorted(_REGISTRY)}"
        ) from None


def known_board_component_ids() -> list[str]:
    """The ids the registry can resolve."""
    return list(_REGISTRY)


# --- Fans: the second component class, same mechanism (bench-006, ADR-0011). The
# mount facts wrap the shared dsl/interfaces/fan.py numbers (verified axial-fan
# standards) + the M3 clearance fact, so nothing is duplicated.
FAN_40MM = FanComponent(
    component_id="fan_40mm",
    display_name="40 mm axial fan",
    body_mm=40.0,
    thickness_mm=10.0,
    mount=FanMountFacts(
        bolt_square_mm=FAN_40MM_BOLT_SQUARE_MM,  # 32 (shared fact)
        screw_clearance_radius_mm=FAN_BOLT_CLEARANCE_RADIUS_MM,  # M3, 1.6 (shared fact)
        airflow_bore_radius_mm=FAN_40MM_AIRFLOW_BORE_RADIUS_MM,  # 19 (~38mm opening)
    ),
    source="Axial-fan industry standard: 40 mm frame, 32 mm bolt spacing, M3 corners.",
)

FAN_60MM = FanComponent(
    component_id="fan_60mm",
    display_name="60 mm axial fan",
    body_mm=60.0,
    thickness_mm=25.0,
    mount=FanMountFacts(
        bolt_square_mm=FAN_60MM_BOLT_SQUARE_MM,  # 50 (shared fact)
        screw_clearance_radius_mm=FAN_BOLT_CLEARANCE_RADIUS_MM,  # M3, 1.6 (shared fact)
        airflow_bore_radius_mm=FAN_60MM_AIRFLOW_BORE_RADIUS_MM,  # 28 (~56mm opening)
    ),
    source="Axial-fan industry standard: 60 mm frame, 50 mm bolt spacing, M3 corners.",
)

_FAN_REGISTRY: dict[str, FanComponent] = {
    FAN_40MM.component_id: FAN_40MM,
    FAN_60MM.component_id: FAN_60MM,
}


def get_fan_component(component_id: str) -> FanComponent:
    """Return the fan component for ``component_id``; raise ValueError if unknown."""
    try:
        return _FAN_REGISTRY[component_id]
    except KeyError:
        raise ValueError(
            f"unknown fan component {component_id!r}; known: {sorted(_FAN_REGISTRY)}"
        ) from None


def known_fan_component_ids() -> list[str]:
    """The fan ids the registry can resolve."""
    return list(_FAN_REGISTRY)


# --- Motors: the third component class (ADR-0017). Multi-entry day one (NEMA 14/17/23).
# The nema17 entry imports today's nema17 facts verbatim (BOLT_PITCH/M3/SHAFT_BORE), so a
# nema17 mount stays byte-identical; nema14/nema23 carry their own standard facts. The
# "pilot bore" is the central register/boss clearance (what the legacy 22 mm "shaft bore"
# actually clears), not the shaft. Facts: NEMA faceplate standard (ICS 16) + datasheets.
NEMA_14 = MotorComponent(
    component_id="nema14",
    display_name="NEMA 14 stepper motor",
    body_mm=35.2,
    shaft_diameter_mm=5.0,
    mount=MotorMountFacts(
        bolt_pitch_mm=26.0,  # NEMA 14: 26 mm bolt square
        bolt_clearance_radius_mm=M3_CLEARANCE_RADIUS_MM,  # M3, 1.6 (shared fact)
        pilot_bore_radius_mm=11.0,  # 22 mm register boss
    ),
    source="NEMA 14 standard: 35 mm frame, 26 mm M3 bolt square, 22 mm register, 5 mm shaft.",
)

NEMA_17 = MotorComponent(
    component_id="nema17",
    display_name="NEMA 17 stepper motor",
    body_mm=42.3,
    shaft_diameter_mm=5.0,
    mount=MotorMountFacts(
        bolt_pitch_mm=BOLT_PITCH_MM,  # 31 (shared fact -- byte-identical w/ the checker)
        bolt_clearance_radius_mm=M3_CLEARANCE_RADIUS_MM,  # M3, 1.6 (shared fact)
        pilot_bore_radius_mm=SHAFT_BORE_RADIUS_MM,  # 11 (shared fact; the 22 mm register)
    ),
    source="NEMA 17 standard (dsl/interfaces/nema17.py): 31 mm M3 bolt square, 22 mm register.",
)

NEMA_23 = MotorComponent(
    component_id="nema23",
    display_name="NEMA 23 stepper motor",
    body_mm=56.4,
    shaft_diameter_mm=6.35,  # 1/4 in (8 mm variants exist on higher-torque models)
    mount=MotorMountFacts(
        bolt_pitch_mm=47.14,  # NEMA 23: 47.14 mm (1.856 in) bolt square -- verified
        bolt_clearance_radius_mm=M5_CLEARANCE_RADIUS_MM,  # M5, 2.6 (shared fact; #10-24 variant)
        pilot_bore_radius_mm=19.05,  # 38.1 mm (1.5 in) register boss
    ),
    source="NEMA 23 standard: 56.4 mm frame, 47.14 mm M5 bolt square, 38.1 mm register, 6.35 mm shaft.",
)

_MOTOR_REGISTRY: dict[str, MotorComponent] = {
    NEMA_14.component_id: NEMA_14,
    NEMA_17.component_id: NEMA_17,
    NEMA_23.component_id: NEMA_23,
}


def get_motor_component(component_id: str) -> MotorComponent:
    """Return the motor component for ``component_id``; raise ValueError if unknown."""
    try:
        return _MOTOR_REGISTRY[component_id]
    except KeyError:
        raise ValueError(
            f"unknown motor component {component_id!r}; known: {sorted(_MOTOR_REGISTRY)}"
        ) from None


def known_motor_component_ids() -> list[str]:
    """The motor ids the registry can resolve."""
    return list(_MOTOR_REGISTRY)


# --- Extrusions: the fourth component class (ADR-0017). Multi-entry day one
# (2020/2040/3030). ext_2020 imports today's M5 + 20 mm constants verbatim (byte-identical);
# 2040 is M5 like 2020 (a T-nut mount drills the same pair); 3030 is the M6/8 mm slot axis.
EXT_2020 = ExtrusionComponent(
    component_id="ext_2020",
    display_name="2020 T-slot extrusion (20x20 mm)",
    profile_mm=20.0,
    slot=ExtrusionSlotFacts(
        slot_width_mm=6.0,
        bolt_clearance_radius_mm=M5_CLEARANCE_RADIUS_MM,  # M5, 2.6 (shared fact)
        default_tnut_spacing_mm=20.0,  # the legacy DEFAULT_SPACING
    ),
    source="20-series T-slot standard: 6 mm slot, M5 T-nuts.",
)

EXT_2040 = ExtrusionComponent(
    component_id="ext_2040",
    display_name="2040 T-slot extrusion (20x40 mm)",
    profile_mm=20.0,  # same 20-series module / slot face as 2020 (the 40 is the long side)
    slot=ExtrusionSlotFacts(
        slot_width_mm=6.0,
        bolt_clearance_radius_mm=M5_CLEARANCE_RADIUS_MM,  # M5, 2.6 (shared fact)
        default_tnut_spacing_mm=20.0,
    ),
    source="20-series T-slot standard (20x40 profile): 6 mm slot, M5 T-nuts.",
)

EXT_3030 = ExtrusionComponent(
    component_id="ext_3030",
    display_name="3030 T-slot extrusion (30x30 mm)",
    profile_mm=30.0,
    slot=ExtrusionSlotFacts(
        slot_width_mm=8.0,
        bolt_clearance_radius_mm=M6_CLEARANCE_RADIUS_MM,  # M6, 3.3 (shared fact)
        default_tnut_spacing_mm=30.0,
    ),
    source="30-series T-slot standard: 8 mm slot, M6 T-nuts.",
)

_EXTRUSION_REGISTRY: dict[str, ExtrusionComponent] = {
    EXT_2020.component_id: EXT_2020,
    EXT_2040.component_id: EXT_2040,
    EXT_3030.component_id: EXT_3030,
}


def get_extrusion_component(component_id: str) -> ExtrusionComponent:
    """Return the extrusion component for ``component_id``; raise ValueError if unknown."""
    try:
        return _EXTRUSION_REGISTRY[component_id]
    except KeyError:
        raise ValueError(
            f"unknown extrusion component {component_id!r}; known: {sorted(_EXTRUSION_REGISTRY)}"
        ) from None


def known_extrusion_component_ids() -> list[str]:
    """The extrusion ids the registry can resolve."""
    return list(_EXTRUSION_REGISTRY)


__all__ = [
    "get_board_component",
    "get_extrusion_component",
    "get_fan_component",
    "get_motor_component",
    "known_board_component_ids",
    "known_extrusion_component_ids",
    "known_fan_component_ids",
    "known_motor_component_ids",
]
