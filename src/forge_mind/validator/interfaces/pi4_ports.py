"""Raspberry Pi 4 port-opening interface.

Ports are rectangular openings in the case walls, not cylinders, so this checker
works on the mesh (via ``ctx.stl_path``) rather than the cylinder features. It is
deliberately practical and not cosmetic: it confirms that the long port wall (USB
/ Ethernet group) and the short port wall (USB-C power / micro-HDMI group) are
each breached by an opening wide enough for connectors, by scanning the wall at
connector height and measuring the largest open span. It does NOT check exact
connector shapes or positions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.validator.geometry import CylinderFeature
from forge_mind.validator.interfaces.base import BaseInterface, InterfaceContext
from forge_mind.validator.report import Check, CheckStatus, Fix, Tier

LONG_OPENING_MIN_MM = 15.0  # USB + Ethernet bank
SHORT_OPENING_MIN_MM = 8.0  # USB-C / micro-HDMI
_SURFACE_INSET_MM = 0.5  # sample just inside the outer wall surface
_N_SAMPLES = 80


def _max_open_span(mesh: Any, start: np.ndarray, end: np.ndarray) -> float:
    """Largest contiguous run of not-inside-solid sample points along start->end,
    returned as a length in mm.
    """
    pts = np.linspace(start, end, _N_SAMPLES)
    inside = mesh.contains(pts)  # True = solid material
    best = cur = 0
    for solid in inside:
        if not solid:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    seg_len = float(np.linalg.norm(end - start)) / (_N_SAMPLES - 1)
    return best * seg_len


class Pi4PortsInterface(BaseInterface):
    type_name = InterfaceType.PI4_PORTS.value

    def check(
        self,
        declaration: InterfaceDeclaration,
        cylinders: list[CylinderFeature],
        ctx: InterfaceContext | None = None,
    ) -> tuple[list[Check], list[Fix]]:
        prefix = f"interfaces.{declaration.name}"
        if ctx is None or ctx.stl_path is None:
            return (
                [
                    Check(
                        id=f"{prefix}.long_edge_opening",
                        tier=Tier.INTERFACES,
                        status=CheckStatus.WARN,
                        message="Cannot check port openings without the mesh.",
                    )
                ],
                [],
            )

        import trimesh

        mesh = trimesh.load(str(ctx.stl_path), force="mesh")
        if not bool(getattr(mesh, "is_watertight", False)):
            # The open-span scan relies on mesh.contains (inside/outside), which is unreliable on a
            # non-watertight mesh: it can read solid wall as "open" and falsely PASS the opening
            # check. Fail closed -- WARN rather than certify an opening we cannot verify (audit).
            return (
                [
                    Check(
                        id=f"{prefix}.{edge}_edge_opening",
                        tier=Tier.INTERFACES,
                        status=CheckStatus.WARN,
                        message=(
                            "Cannot verify port openings: the mesh is not a closed solid, so the "
                            "inside/outside sampling is unreliable."
                        ),
                    )
                    for edge in ("long", "short")
                ],
                [],
            )
        xmin, ymin, zmin = mesh.bounds[0]
        xmax, ymax, zmax = mesh.bounds[1]
        z = zmin + (zmax - zmin) * 0.55  # connector height, above the floor
        inset = _SURFACE_INSET_MM

        def scan_x(y: float) -> float:
            return _max_open_span(mesh, np.array([xmin + 2, y, z]), np.array([xmax - 2, y, z]))

        def scan_y(x: float) -> float:
            return _max_open_span(mesh, np.array([x, ymin + 2, z]), np.array([x, ymax - 2, z]))

        # Long edges run along X (the longer footprint axis) -> the +/-Y walls.
        long_span = max(scan_x(ymax - inset), scan_x(ymin + inset))
        # Short edges -> the +/-X walls.
        short_span = max(scan_y(xmin + inset), scan_y(xmax - inset))

        checks: list[Check] = []
        fixes: list[Fix] = []

        long_ok = long_span >= LONG_OPENING_MIN_MM
        checks.append(
            Check(
                id=f"{prefix}.long_edge_opening",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if long_ok else CheckStatus.FAIL,
                message=(
                    "Long-edge (USB/Ethernet) opening present."
                    if long_ok
                    else f"No long-edge opening: largest span {long_span:.1f} mm."
                ),
                expected=f">= {LONG_OPENING_MIN_MM:.0f} mm open",
                actual=f"{long_span:.1f} mm",
            )
        )
        if not long_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason="Cut a USB/Ethernet opening in a long wall at connector height.",
                    suggested_change={"min_opening_mm": LONG_OPENING_MIN_MM},
                )
            )

        short_ok = short_span >= SHORT_OPENING_MIN_MM
        checks.append(
            Check(
                id=f"{prefix}.short_edge_opening",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if short_ok else CheckStatus.FAIL,
                message=(
                    "Short-edge (USB-C / micro-HDMI) opening present."
                    if short_ok
                    else f"No short-edge opening: largest span {short_span:.1f} mm."
                ),
                expected=f">= {SHORT_OPENING_MIN_MM:.0f} mm open",
                actual=f"{short_span:.1f} mm",
            )
        )
        if not short_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason="Cut a USB-C / micro-HDMI opening in a short wall at connector height.",
                    suggested_change={"min_opening_mm": SHORT_OPENING_MIN_MM},
                )
            )

        return checks, fixes


__all__ = ["Pi4PortsInterface"]
