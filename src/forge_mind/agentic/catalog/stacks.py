"""Author-facing catalog access + deterministic hardware-stack helpers.

Two layers on top of the registry:

  PART WRAPPERS -- ``catalog_bearing("608")`` / ``catalog_gear(id)`` / ``catalog_screw(id)`` / ...
    resolve a friendly designation (short or full id) to its verified geometry. The author never
    invents a screw/bearing/gear; it names one and gets the exact solid.

  STACK HELPERS -- ``screw_stack`` / ``bearing_mount`` / ``gear_pair`` / ``planetary_set`` /
    ``screw_counterbore`` compute the assembly metadata (hole diameter, stack height, centre
    distance, clearances) and validate compatibility, reusing each family's deterministic rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge_mind.agentic.catalog.bearings import validate_bearing_mount
from forge_mind.agentic.catalog.fasteners import screw_counterbore_ok
from forge_mind.agentic.catalog.gears import catalog_center_distance, validate_gear_pair_catalog
from forge_mind.agentic.catalog.nuts import nut_pocket_dims, validate_nut_fit
from forge_mind.agentic.catalog.registry import (
    CatalogError,
    CatalogIssue,
    CatalogPart,
    catalog_geometry,
    get_catalog_part,
)
from forge_mind.agentic.catalog.washers import validate_washer_fit, washer_stack_height


def resolve_part(designation: str, family: str | None = None) -> CatalogPart:
    """Resolve a short or full catalog id (``"608"`` or ``"bearing_608"``). Raises with suggestions."""
    for candidate in [designation, f"{family}_{designation}"] if family else [designation]:
        try:
            return get_catalog_part(candidate)
        except CatalogError:
            continue
    return get_catalog_part(designation)  # raise the helpful CatalogError


def _geom(designation: str, family: str | None = None, **overrides: Any) -> Any:
    return catalog_geometry(resolve_part(designation, family), **overrides)


def catalog_bearing(designation: str) -> Any:
    """Verified bearing solid by designation (``"608"`` or ``"bearing_608"``)."""
    return _geom(designation, "bearing")


def catalog_gear(part_id: str, **overrides: Any) -> Any:
    """Verified spur-gear solid by catalog id (e.g. ``"spur_m1_z24_w4_b6"``). ``clock_deg`` etc. override."""
    return _geom(part_id, "gear", **overrides)


def catalog_screw(part_id: str, **overrides: Any) -> Any:
    """Verified screw solid by catalog id (e.g. ``"shcs_m5x20"``)."""
    return _geom(part_id, "screw", **overrides)


def catalog_washer(part_id: str) -> Any:
    """Verified washer solid by catalog id (e.g. ``"washer_flat_m5_normal"``)."""
    return _geom(part_id, "washer")


def catalog_nut(part_id: str) -> Any:
    """Verified nut solid by catalog id (e.g. ``"hexnut_m5"``)."""
    return _geom(part_id, "nut")


# ----------------------------------------------------------------------------
# Hardware stacks
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareSpec:
    """The deterministic result of composing standard parts: the parts, computed dims, and issues."""

    parts: list[CatalogPart]
    dims_mm: dict[str, float]
    issues: list[CatalogIssue] = field(default_factory=list)
    notes: str = ""

    @property
    def ok(self) -> bool:
        return not any(i.is_error for i in self.issues)


def screw_stack(*, screw: str, washer: str | None = None, nut: str | None = None) -> HardwareSpec:
    """A screw + optional washer + optional nut: the clearance hole, the stack height, and a check
    that the washer and nut fit the screw thread."""
    s = resolve_part(screw, "screw")
    parts: list[CatalogPart] = [s]
    issues: list[CatalogIssue] = []
    thread_d = float(s.dims_mm["thread_d"])
    height = float(s.dims_mm.get("head_h", 0.0))
    washers: list[CatalogPart] = []
    if washer is not None:
        w = resolve_part(washer, "washer")
        parts.append(w)
        washers.append(w)
        issues.extend(validate_washer_fit(w, thread_d))
    if nut is not None:
        n = resolve_part(nut, "nut")
        parts.append(n)
        issues.extend(validate_nut_fit(n, thread_d, float(s.dims_mm["pitch"])))
        height += float(n.dims_mm["thickness"])
    height += washer_stack_height(washers)
    dims = {
        "clearance_hole_d": float(s.dims_mm["clearance_hole_d"]),
        "stack_height": round(height, 3),
        "screw_length": float(s.dims_mm["length"]),
    }
    return HardwareSpec(parts=parts, dims_mm=dims, issues=issues)


def screw_counterbore(
    screw: str, *, bore_d: float | None = None, depth: float | None = None
) -> HardwareSpec:
    """A counterbored screw seat: the recommended counterbore (from the screw's head) + a check of a
    proposed bore if given."""
    s = resolve_part(screw, "screw")
    issues: list[CatalogIssue] = []
    dims: dict[str, float] = {"clearance_hole_d": float(s.dims_mm["clearance_hole_d"])}
    cb_d = s.dims_mm.get("counterbore_d")
    if cb_d is None:  # only socket-head cap screws take a cylindrical counterbore
        issues.append(CatalogIssue("error", f"{s.id} does not take a cylindrical counterbore"))
    else:
        dims["counterbore_d"] = cb_d
        dims["counterbore_depth"] = float(s.dims_mm["counterbore_depth"])
        if bore_d is not None and depth is not None:
            issues.extend(screw_counterbore_ok(s, bore_d, depth))
    return HardwareSpec(parts=[s], dims_mm=dims, issues=issues)


def bearing_mount(*, bearing: str, shaft_d: float, housing_bore_d: float) -> HardwareSpec:
    """A bearing seated on a shaft in a housing bore: validates both fits against the bearing."""
    b = resolve_part(bearing, "bearing")
    issues = validate_bearing_mount(b, shaft_d, housing_bore_d)
    return HardwareSpec(
        parts=[b],
        dims_mm={
            "bore_d": float(b.dims_mm["bore_d"]),
            "outer_d": float(b.dims_mm["outer_d"]),
            "width": float(b.dims_mm["width"]),
            "shaft_d": shaft_d,
            "housing_bore_d": housing_bore_d,
        },
        issues=issues,
    )


def gear_pair(*, driver: str, driven: str) -> HardwareSpec:
    """Two spur gears: validates they mesh (equal module/pressure angle) and gives the centre distance."""
    a = resolve_part(driver, "gear")
    b = resolve_part(driven, "gear")
    issues = validate_gear_pair_catalog(a, b)
    cd = catalog_center_distance(a, b) if a.dims_mm["module"] == b.dims_mm["module"] else 0.0
    ratio = float(b.dims_mm["teeth"]) / float(a.dims_mm["teeth"])
    return HardwareSpec(
        parts=[a, b],
        dims_mm={"center_distance": round(cd, 3), "ratio": round(ratio, 4)},
        issues=issues,
    )


def gear_pair_geometry(*, driver: str, driven: str) -> Any:
    """Build the MESHING geometry of a gear pair: the driver + the driven clocked + placed at the
    centre distance (a labelled Compound). Reuses the Phase-2 clocking."""
    from build123d import Compound, Pos

    from forge_mind.agentic.gears import mesh_clock_deg

    a = resolve_part(driver, "gear")
    b = resolve_part(driven, "gear")
    spec = gear_pair(driver=driver, driven=driven)
    if not spec.ok:
        raise CatalogError("; ".join(i.message for i in spec.issues if i.is_error))
    cd = spec.dims_mm["center_distance"]
    g1 = catalog_geometry(a)
    g1.label = a.id
    g2 = Pos(cd, 0, 0) * catalog_geometry(b, clock_deg=mesh_clock_deg(int(b.dims_mm["teeth"])))
    g2.label = b.id
    return Compound(children=[g1, g2])


def planetary_set(*, sun: str, planet: str, n_planets: int) -> HardwareSpec:
    """A planetary train from catalog gears: validates the tooth relation + spacing (ring teeth are
    derived)."""
    from forge_mind.agentic.catalog.gears import validate_planetary_catalog

    s = resolve_part(sun, "gear")
    p = resolve_part(planet, "gear")
    ring_teeth = int(s.dims_mm["teeth"]) + 2 * int(p.dims_mm["teeth"])
    issues = validate_planetary_catalog(s.id, p.id, ring_teeth, n_planets)
    return HardwareSpec(
        parts=[s, p],
        dims_mm={
            "ring_teeth": float(ring_teeth),
            "reduction": round(1.0 + ring_teeth / s.dims_mm["teeth"], 3),
        },
        issues=issues,
    )


__all__ = [
    "HardwareSpec",
    "bearing_mount",
    "catalog_bearing",
    "catalog_gear",
    "catalog_nut",
    "catalog_screw",
    "catalog_washer",
    "gear_pair",
    "gear_pair_geometry",
    "nut_pocket_dims",
    "planetary_set",
    "resolve_part",
    "screw_counterbore",
    "screw_stack",
]
