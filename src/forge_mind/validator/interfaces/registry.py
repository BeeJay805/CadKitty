"""Interface registry — maps an InterfaceType to its checker.

Adding a new interface = add one checker file + one registry entry.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceType
from forge_mind.validator.interfaces.base import BaseInterface
from forge_mind.validator.interfaces.extrusion_2020_slot import Extrusion2020SlotInterface
from forge_mind.validator.interfaces.fan_face import FanFaceInterface
from forge_mind.validator.interfaces.m5_tnut import M5TnutInterface
from forge_mind.validator.interfaces.nema17_face import Nema17FaceInterface
from forge_mind.validator.interfaces.pi4_mount import Pi4MountInterface
from forge_mind.validator.interfaces.pi4_ports import Pi4PortsInterface

_REGISTRY: dict[str, BaseInterface] = {
    InterfaceType.NEMA17_FACE.value: Nema17FaceInterface(),
    InterfaceType.EXTRUSION_2020_SLOT.value: Extrusion2020SlotInterface(),
    InterfaceType.M5_TNUT.value: M5TnutInterface(),
    InterfaceType.PI4_MOUNT.value: Pi4MountInterface(),
    InterfaceType.PI4_PORTS.value: Pi4PortsInterface(),
    InterfaceType.FAN_FACE.value: FanFaceInterface(),
}


def get_interface(interface_type: str) -> BaseInterface | None:
    """Return the checker for an interface type, or None if unknown."""
    return _REGISTRY.get(interface_type)


def known_interface_types() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = ["get_interface", "known_interface_types"]
