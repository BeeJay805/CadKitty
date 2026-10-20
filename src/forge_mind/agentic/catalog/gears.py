"""Gear catalog: a deterministic grid of standard involute spur gears.

Every entry's geometry comes from OUR own validated involute generator
(:mod:`forge_mind.agentic.gears`) -- never hand-typed. The catalog is a controlled GENERATED grid
over the standard meshing parameters (module x tooth count at a 20-degree pressure angle), with a
representative face width + shaft bore per module. Each gear's pitch / outer / root diameters are
read straight off a :class:`~forge_mind.agentic.gears.GearSpec`, so a search like "spur module 1"
returns parts whose teeth provably mesh. Geometry is built on demand by the ``native_gear`` backend.
~132 entries (a bored and a solid variant of each grid point).
"""

from __future__ import annotations

from forge_mind.agentic.catalog.registry import CatalogIssue, CatalogPart, register_builder

# The generated grid: standard modules x tooth counts at the modern 20-degree pressure angle.
_MODULES: tuple[float, ...] = (0.5, 0.8, 1.0, 1.25, 1.5, 2.0)
_TOOTH_COUNTS: tuple[int, ...] = (12, 16, 18, 20, 24, 28, 32, 36, 40, 48, 60)
_PRESSURE_ANGLE = 20.0

# Common shaft-bore diameters (mm); we pick the largest that stays comfortably inside the root.
_COMMON_BORES: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12)


def _module_token(module: float) -> str:
    """Compact, filesystem-safe module token: 1.0 -> 'm1', 0.5 -> 'm0p5', 1.25 -> 'm1p25'."""
    return "m" + format(module, "g").replace(".", "p")


def _face_width(module: float) -> int:
    """Representative face width (mm) for a module -- scales with size, never below 4mm."""
    return max(4, round(4.0 * module))


def _shaft_bore(module: float, teeth: int) -> int:
    """A sensible shaft bore (mm): the largest common size at most half the pitch radius and kept
    strictly inside the root circle (so the bore never erases the gear body). 0 if none fits."""
    from forge_mind.agentic.gears import GearSpec

    spec = GearSpec(module, teeth, _PRESSURE_ANGLE)
    target = spec.pitch_radius * 0.5
    root_d = 2.0 * spec.root_radius
    fit = [b for b in _COMMON_BORES if b <= target and b < root_d]
    return max(fit) if fit else 0


def _entry(module: float, teeth: int, width: int, bore: int) -> CatalogPart:
    """Build one catalog gear from a grid point. All diameters come from the GearSpec (our math)."""
    from forge_mind.agentic.gears import GearSpec

    spec = GearSpec(module, teeth, _PRESSURE_ANGLE, float(width), float(bore))
    mtok = _module_token(module)
    pid = f"spur_{mtok}_z{teeth}_w{width}_b{bore}"
    tags = [
        "gear",
        "spur",
        "involute",
        mtok,
        f"z{teeth}",
        f"module_{format(module, 'g')}",
        f"teeth_{teeth}",
    ]
    return CatalogPart(
        id=pid,
        name=f"spur gear module {format(module, 'g')} x {teeth} teeth"
        + (f", {bore}mm bore" if bore else ", solid"),
        family="gear",
        subtype="spur",
        dims_mm={
            "module": module,
            "teeth": float(teeth),
            "pressure_angle": _PRESSURE_ANGLE,
            "pitch_d": round(2.0 * spec.pitch_radius, 6),
            "outer_d": round(2.0 * spec.outer_radius, 6),
            "root_d": round(2.0 * spec.root_radius, 6),
            "width": float(width),
            "bore_d": float(bore),
        },
        source="Forge-Mind involute generator (deterministic)",
        backend="native_gear",
        geo_params={
            "kind": "spur",
            "module": module,
            "teeth": teeth,
            "thickness": float(width),
            "bore": float(bore),
        },
        provisional=False,
        tags=tuple(dict.fromkeys(tags)),  # de-dup, keep order
        # same-module gears mesh: tag the module so search finds the meshing set (no O(n^2) listing).
        compatible={"meshes_module": [mtok]},
        notes=(
            f"meshes any module-{format(module, 'g')} 20-deg gear; "
            f"centre distance to a like gear = (this pitch_d + other pitch_d)/2."
        ),
    )


@register_builder
def build_gears() -> list[CatalogPart]:
    """The full generated spur-gear grid: a bored and a solid variant of each module x tooth point."""
    parts: list[CatalogPart] = []
    for module in _MODULES:
        width = _face_width(module)
        for teeth in _TOOTH_COUNTS:
            bore = _shaft_bore(module, teeth)
            if bore > 0:
                parts.append(_entry(module, teeth, width, bore))
            parts.append(_entry(module, teeth, width, 0))  # solid variant
    return parts


# ----------------------------------------------------------------------------
# Catalog-level gear validation (reuses the pure gear math)
# ----------------------------------------------------------------------------


def _spec_of(part: CatalogPart):  # type: ignore[no-untyped-def]
    """Reconstruct a GearSpec from a catalog gear's geo_params (the authoritative build params)."""
    from forge_mind.agentic.gears import GearSpec

    gp = part.geo_params
    return GearSpec(
        module=float(gp["module"]),
        teeth=int(gp["teeth"]),
        pressure_angle=_PRESSURE_ANGLE,
        thickness=float(gp["thickness"]),
        bore=float(gp["bore"]),
    )


def validate_gear_pair_catalog(a: CatalogPart, b: CatalogPart) -> list[CatalogIssue]:
    """Will these two catalog gears mesh? Delegates to the pure ``validate_gear_pair`` rules
    (equal module + equal pressure angle) and maps each GearIssue to a CatalogIssue."""
    from forge_mind.agentic.gears import validate_gear_pair

    issues: list[CatalogIssue] = []
    for part in (a, b):
        if part.family != "gear" or part.subtype != "spur":
            issues.append(CatalogIssue("error", f"{part.id} is not a spur gear"))
    if issues:
        return issues
    return [
        CatalogIssue(i.severity, i.message) for i in validate_gear_pair(_spec_of(a), _spec_of(b))
    ]


def catalog_center_distance(a: CatalogPart, b: CatalogPart) -> float:
    """Theoretical external-mesh centre distance (mm) for two catalog gears -- reuses
    ``center_distance`` on the reconstructed GearSpecs."""
    from forge_mind.agentic.gears import center_distance

    return center_distance(_spec_of(a), _spec_of(b))


def validate_planetary_catalog(
    sun_id: str, planet_id: str, ring_teeth: int, n_planets: int
) -> list[CatalogIssue]:
    """Validate a planetary set built from two catalog gear ids (sun + planet) plus a ring tooth
    count and planet count. Delegates to the pure ``validate_planetary`` (ring = sun + 2*planet,
    equal spacing, neighbour clearance) and maps each GearIssue to a CatalogIssue."""
    from forge_mind.agentic.catalog.registry import get_catalog_part
    from forge_mind.agentic.gears import validate_planetary

    sun = get_catalog_part(sun_id)
    planet = get_catalog_part(planet_id)
    issues: list[CatalogIssue] = []
    for part in (sun, planet):
        if part.family != "gear" or part.subtype != "spur":
            issues.append(CatalogIssue("error", f"{part.id} is not a spur gear"))
    if issues:
        # A non-spur part has no module/teeth geo_params; return the clean issue rather than
        # KeyError-ing on the module read below (audit: catalog KeyError before the guard returned).
        return issues
    if abs(float(sun.geo_params["module"]) - float(planet.geo_params["module"])) > 1e-9:
        issues.append(
            CatalogIssue(
                "error", "sun and planet have different modules -- a planetary set cannot mesh"
            )
        )
    if issues:
        return issues
    sun_teeth = int(sun.geo_params["teeth"])
    planet_teeth = int(planet.geo_params["teeth"])
    return [
        CatalogIssue(i.severity, i.message)
        for i in validate_planetary(sun_teeth, planet_teeth, ring_teeth, n_planets)
    ]


__all__ = [
    "build_gears",
    "catalog_center_distance",
    "validate_gear_pair_catalog",
    "validate_planetary_catalog",
]
