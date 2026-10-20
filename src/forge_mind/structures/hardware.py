"""Hardware-as-data (Fable move #3): a declared component becomes keep-void + keep-solid regions.

The founder's rule made mechanical: the engine never invents a motor -- a ``ComponentPlacement``
resolves to the component's DATASHEET footprint (single-sourced from the assembly hardware
registry, the same numbers the mount validators check) and enters the topology domain as a
KEEP-VOID pocket (body + clearance: the cutout the real part drops into) plus KEEP-SOLID mount
bands (the flange tabs that bolt down). SIMP then grows structure around them. This is also the
M3 servo-bridge contract brought forward: the same declaration will later emit PlacedHardware for
the assembly validators.

Fail-closed: an unknown ``kind`` raises -- a guessed footprint would be the false-precision
vector; only registry-known hardware may shape a domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from forge_mind.assembly.servo import (
    MG996R_BODY_L_MM,
    MG996R_BODY_W_MM,
    MG996R_FLANGE_SPAN_MM,
)
from forge_mind.structures.spec import ComponentPlacement
from forge_mind.structures.topopt import Domain


@dataclass(frozen=True)
class HardwareFootprint:
    """Plan-view footprint (mm): the body pocket and the full mount span across the tabs."""

    kind: str
    body_mm: tuple[float, float]  # (x, y) body outline -> keep-void (+ clearance)
    mount_span_mm: float  # overall x-span across both mounting tabs -> keep-solid bands


_REGISTRY: dict[str, HardwareFootprint] = {
    "mg996r_servo": HardwareFootprint(
        kind="mg996r_servo",
        body_mm=(MG996R_BODY_L_MM, MG996R_BODY_W_MM),
        mount_span_mm=MG996R_FLANGE_SPAN_MM,
    ),
}


def footprint_for(kind: str) -> HardwareFootprint:
    """Registry lookup, fail-closed: unknown hardware may not shape a domain."""
    try:
        return _REGISTRY[kind]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown hardware kind {kind!r}; known: {known}") from None


def _box_elements(
    dom: Domain, *, x0: float, y0: float, x1: float, y1: float, cell_mm: tuple[float, float]
) -> list[int]:
    """Elements covering the mm-box [x0,x1] x [y0,y1] (cell (ix,iy) spans [ix*cx,(ix+1)*cx))."""
    cx, cy = cell_mm
    return dom.elements_in_box(
        math.floor(x0 / cx),
        math.floor(y0 / cy),
        math.ceil(x1 / cx),
        math.ceil(y1 / cy),
    )


def place_component(
    dom: Domain, comp: ComponentPlacement, *, cell_mm: tuple[float, float]
) -> dict[str, int]:
    """Mutate ``dom`` with the component's regions. ``comp.at_mm`` is the body CENTER in domain
    mm. Body + clearance -> keep_void; the two flange tabs beyond the body (x direction) ->
    keep_solid. Returns the region sizes for the caller's record."""
    fp = footprint_for(comp.kind)
    cxm, cym = comp.at_mm
    bx, by = fp.body_mm
    c = comp.clearance_mm

    pocket = _box_elements(
        dom,
        x0=cxm - bx / 2 - c,
        y0=cym - by / 2 - c,
        x1=cxm + bx / 2 + c,
        y1=cym + by / 2 + c,
        cell_mm=cell_mm,
    )
    tab = (fp.mount_span_mm - bx) / 2  # flange overhang beyond the body, each side
    mounts: list[int] = []
    for side in (-1, 1):
        inner = cxm + side * (bx / 2 + c)
        outer = cxm + side * (bx / 2 + c + tab)
        x0, x1 = min(inner, outer), max(inner, outer)
        mounts += _box_elements(
            dom, x0=x0, y0=cym - by / 2, x1=x1, y1=cym + by / 2, cell_mm=cell_mm
        )
    pocket_set = set(pocket)
    mounts = [e for e in mounts if e not in pocket_set]  # pocket wins where they touch
    dom.keep_void = sorted(set(dom.keep_void) | pocket_set)
    dom.keep_solid = sorted((set(dom.keep_solid) | set(mounts)) - pocket_set)
    return {"pocket_elements": len(pocket_set), "mount_elements": len(set(mounts))}


__all__ = ["HardwareFootprint", "footprint_for", "place_component"]
