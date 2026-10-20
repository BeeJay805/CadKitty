"""The canonical material library: one source of truth for every material fact.

A ``Material`` carries the physical, manufacturing, and appearance properties Forge-Mind
needs to (a) estimate mass + cost from the exact B-rep volume, (b) gate geometry against
what the material can actually hold (min wall / min feature / fit clearance), and (c) render
the part in its real appearance instead of a generic steel-blue.

This module is PURE (no build123d, no I/O) so it imports cheaply everywhere -- the validator,
the agentic path, the renderer, and the CLI all read the SAME registry. ``validator.metrics``
delegates its density table here, so there is exactly one place a material is defined.

HONEST CAVEATS:
  - ``density_g_cm3`` x volume assumes 100% SOLID material -- a comparative estimate, not a
    scale-accurate FDM-infill mass (the slicer's filament estimate is the real print mass).
  - ``cost_per_kg_usd`` is a rough MATERIAL-stock basis (no machining / electricity / labor).
  - ``min_wall_mm`` / ``clearance_mm`` are sane process defaults, NOT calibrated against real
    prints yet (see ``docs/calibration_protocol.md`` -- the loop that will tighten these).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

RGB = tuple[float, float, float]


class ProcessCategory(StrEnum):
    """The fabrication family a material is typically made by. Used for spec grouping + hints."""

    FDM = "fdm"  # fused-filament 3D printing
    RESIN = "resin"  # SLA / DLP / MSLA vat photopolymer
    CNC = "cnc"  # subtractive machining (sheet / stock)
    METAL = "metal"  # machined / cast / sintered metal
    GENERAL = "general"


@dataclass(frozen=True)
class Appearance:
    """How a material LOOKS, for material-accurate rendering (a small PBR-ish surface model)."""

    base_color: RGB  # linear RGB 0..1
    roughness: float  # 0 = mirror, 1 = fully matte
    metallic: float  # 0 = dielectric (plastic/wood), 1 = metal
    opacity: float = 1.0  # 1 = opaque; < 1 for translucent (PETG, resin)


@dataclass(frozen=True)
class Mechanical:
    """Elastic + strength properties for fit / stress math (ISO 286 press-fit, hub hoop stress).

    Populated for metals (the common press-fit materials); ``None`` on a Material whose
    mechanical properties are print-/process-dependent (FDM plastics) -- there a confident
    number would be wrong-but-confident, so the fit engine REFUSES instead of guessing.
    ``basis`` names the alloy / temper the handbook values assume (stated in every result).
    """

    youngs_modulus_gpa: float  # E
    poisson_ratio: float  # nu
    yield_strength_mpa: float  # tensile yield (for hub-burst safety factor)
    basis: str


@dataclass(frozen=True)
class Material:
    """Everything Forge-Mind knows about one material. Immutable; looked up by id or alias."""

    id: str  # canonical, lower-snake, e.g. "pla"
    name: str  # display name, e.g. "PLA"
    process: ProcessCategory
    density_g_cm3: float
    cost_per_kg_usd: float  # rough material-stock basis (estimate)
    min_wall_mm: float  # thinnest reliably-producible wall / rib
    min_feature_mm: float  # smallest reliably-producible detail (pin, slot, embossed line)
    clearance_mm: tuple[float, float]  # recommended slip-fit gap PER SIDE (min, max)
    shrinkage_comp: float  # multiply a nominal dim by this to hit target after shrink (1.0 = none)
    appearance: Appearance
    notes: str
    mechanical: Mechanical | None = (
        None  # elastic + yield for fit/stress math; None if print-dependent
    )
    # ADR-0107 stress-safety screen: a conservative strength allowable + its CITED basis. Plastics
    # carry an explicit interlayer screen; metals leave this None and derive from ``mechanical``
    # yield. None (no defensible basis) -> the stress check reports "n/a", never a guessed PASS.
    strength_allowable_mpa: float | None = None
    strength_basis: str | None = None

    @property
    def recommended_clearance_mm(self) -> float:
        """The midpoint of the recommended slip-fit clearance band (a single design value)."""
        lo, hi = self.clearance_mm
        return (lo + hi) / 2.0

    def compensated_mm(self, nominal_mm: float) -> float:
        """Nominal dimension scaled for shrink so the as-made part lands on target."""
        return nominal_mm * self.shrinkage_comp


# ----------------------------------------------------------------------------
# The registry. Densities for PLA/PETG/ABS/TPU/NYLON are pinned to the historical
# validator.metrics values so mass numbers stay byte-stable; the rest are added.
# ----------------------------------------------------------------------------
_MATERIALS_LIST: list[Material] = [
    Material(
        id="pla",
        name="PLA",
        process=ProcessCategory.FDM,
        density_g_cm3=1.24,
        cost_per_kg_usd=22.0,
        min_wall_mm=1.0,
        min_feature_mm=0.8,
        clearance_mm=(0.2, 0.4),
        shrinkage_comp=1.002,
        appearance=Appearance(base_color=(0.82, 0.82, 0.85), roughness=0.7, metallic=0.0),
        notes="Stiff, easy FDM default. Low warp, low temp resistance, somewhat brittle.",
        strength_allowable_mpa=20.0,
        strength_basis="PLA interlayer screen 20 MPa (~0.5x50 MPa XY UTS layer-adhesion knockdown; ADR-0039 screen, not a certified strength)",
    ),
    Material(
        id="petg",
        name="PETG",
        process=ProcessCategory.FDM,
        density_g_cm3=1.27,
        cost_per_kg_usd=25.0,
        min_wall_mm=1.0,
        min_feature_mm=0.8,
        clearance_mm=(0.25, 0.45),
        shrinkage_comp=1.003,
        appearance=Appearance(
            base_color=(0.74, 0.83, 0.84), roughness=0.4, metallic=0.0, opacity=0.92
        ),
        notes="Tough, slightly flexible, chemical/water resistant. Stringy; wants more clearance.",
        strength_allowable_mpa=20.0,
        strength_basis="PETG interlayer screen 20 MPa (~0.5x50 MPa XY UTS knockdown; screen, not a certified strength)",
    ),
    Material(
        id="abs",
        name="ABS",
        process=ProcessCategory.FDM,
        density_g_cm3=1.04,
        cost_per_kg_usd=24.0,
        min_wall_mm=1.2,
        min_feature_mm=1.0,
        clearance_mm=(0.2, 0.4),
        shrinkage_comp=1.006,
        appearance=Appearance(base_color=(0.55, 0.56, 0.58), roughness=0.8, metallic=0.0),
        notes="Heat/impact resistant, machinable. Warps and shrinks; needs an enclosure.",
        strength_allowable_mpa=15.0,
        strength_basis="ABS interlayer screen 15 MPa (~0.5x35 MPa XY UTS knockdown; screen, not a certified strength)",
    ),
    Material(
        id="tpu",
        name="TPU",
        process=ProcessCategory.FDM,
        density_g_cm3=1.21,
        cost_per_kg_usd=35.0,
        min_wall_mm=1.4,
        min_feature_mm=1.2,
        clearance_mm=(0.3, 0.6),
        shrinkage_comp=1.002,
        appearance=Appearance(base_color=(0.25, 0.25, 0.28), roughness=0.9, metallic=0.0),
        notes="Flexible elastomer. Needs thicker walls + more clearance; prints slowly.",
    ),
    Material(
        id="nylon",
        name="Nylon",
        process=ProcessCategory.FDM,
        density_g_cm3=1.14,
        cost_per_kg_usd=45.0,
        min_wall_mm=1.2,
        min_feature_mm=1.0,
        clearance_mm=(0.25, 0.5),
        shrinkage_comp=1.015,
        appearance=Appearance(base_color=(0.90, 0.89, 0.85), roughness=0.75, metallic=0.0),
        notes="Strong, wear/chemical resistant, low friction. Hygroscopic; shrinks notably.",
    ),
    Material(
        id="resin",
        name="Resin",
        process=ProcessCategory.RESIN,
        density_g_cm3=1.18,
        cost_per_kg_usd=50.0,
        min_wall_mm=0.6,
        min_feature_mm=0.3,
        clearance_mm=(0.1, 0.25),
        shrinkage_comp=1.0,
        appearance=Appearance(base_color=(0.66, 0.66, 0.70), roughness=0.35, metallic=0.0),
        notes="SLA/DLP photopolymer. Very fine detail, tight tolerances; brittle, UV-sensitive.",
    ),
    Material(
        id="aluminum",
        name="Aluminum",
        process=ProcessCategory.METAL,
        density_g_cm3=2.70,
        cost_per_kg_usd=6.0,
        min_wall_mm=0.8,
        min_feature_mm=0.5,
        clearance_mm=(0.05, 0.15),
        shrinkage_comp=1.0,
        appearance=Appearance(base_color=(0.80, 0.81, 0.83), roughness=0.35, metallic=1.0),
        notes="Machined 6061-class stock. Light, strong, conductive. Tight CNC tolerances.",
        mechanical=Mechanical(
            youngs_modulus_gpa=68.9, poisson_ratio=0.33, yield_strength_mpa=276.0, basis="6061-T6"
        ),
    ),
    Material(
        id="steel",
        name="Steel",
        process=ProcessCategory.METAL,
        density_g_cm3=7.85,
        cost_per_kg_usd=2.5,
        min_wall_mm=1.0,
        min_feature_mm=0.5,
        clearance_mm=(0.05, 0.15),
        shrinkage_comp=1.0,
        appearance=Appearance(base_color=(0.62, 0.63, 0.66), roughness=0.4, metallic=1.0),
        notes="Mild/stainless machined stock. Very strong/heavy; tight CNC tolerances.",
        mechanical=Mechanical(
            youngs_modulus_gpa=200.0,
            poisson_ratio=0.30,
            yield_strength_mpa=250.0,
            basis="mild steel (A36/1018)",
        ),
    ),
    Material(
        id="brass",
        name="Brass",
        process=ProcessCategory.METAL,
        density_g_cm3=8.50,
        cost_per_kg_usd=12.0,
        min_wall_mm=0.8,
        min_feature_mm=0.5,
        clearance_mm=(0.05, 0.15),
        shrinkage_comp=1.0,
        appearance=Appearance(base_color=(0.80, 0.64, 0.30), roughness=0.35, metallic=1.0),
        notes="Machined C360-class stock. Corrosion resistant, low friction; decorative.",
        mechanical=Mechanical(
            youngs_modulus_gpa=105.0,
            poisson_ratio=0.34,
            yield_strength_mpa=124.0,
            basis="C360 free-cutting brass (soft)",
        ),
    ),
    Material(
        id="wood",
        name="Wood",
        process=ProcessCategory.CNC,
        density_g_cm3=0.68,
        cost_per_kg_usd=3.0,
        min_wall_mm=2.5,
        min_feature_mm=1.5,
        clearance_mm=(0.3, 0.6),
        shrinkage_comp=1.0,
        appearance=Appearance(base_color=(0.66, 0.47, 0.28), roughness=0.85, metallic=0.0),
        notes="Plywood/MDF sheet stock, CNC-routed. Brittle in thin sections; grain-dependent.",
    ),
]

MATERIALS: dict[str, Material] = {m.id: m for m in _MATERIALS_LIST}
DEFAULT_MATERIAL_ID = "pla"

# Common aliases -> canonical id. Lets a user (or a brief) say "aluminium", "PA12", "SLA".
_ALIASES: dict[str, str] = {
    "pla+": "pla",
    "pla pro": "pla",
    "co-petg": "petg",
    "pet-g": "petg",
    "pet": "petg",
    "asa": "abs",  # close enough for spec purposes
    "tpu95a": "tpu",
    "tpe": "tpu",
    "pa": "nylon",
    "pa6": "nylon",
    "pa12": "nylon",
    "polyamide": "nylon",
    "sla": "resin",
    "dlp": "resin",
    "msla": "resin",
    "photopolymer": "resin",
    "aluminium": "aluminum",
    "al": "aluminum",
    "alu": "aluminum",
    "6061": "aluminum",
    "7075": "aluminum",
    "mild steel": "steel",
    "stainless": "steel",
    "stainless steel": "steel",
    "ss": "steel",
    "304": "steel",
    "316": "steel",
    "c360": "brass",
    "plywood": "wood",
    "mdf": "wood",
    "ply": "wood",
}


class UnknownMaterialError(ValueError):
    """Raised by :func:`get_material` for a name that is neither a known id nor an alias."""


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def get_material(name: str) -> Material:
    """Resolve ``name`` (id or alias) to a :class:`Material`, STRICTLY.

    Raises :class:`UnknownMaterialError` with the list of known ids for a typo. Use this on
    user-facing input (the CLI) so a misspelled material fails loudly instead of silently
    becoming PLA.
    """
    key = _normalize(name)
    if key in MATERIALS:
        return MATERIALS[key]
    if key in _ALIASES:
        return MATERIALS[_ALIASES[key]]
    known = ", ".join(sorted(MATERIALS))
    raise UnknownMaterialError(f"unknown material {name!r}; known materials: {known}")


def resolve_material(name: str | None, *, default: str = DEFAULT_MATERIAL_ID) -> Material:
    """Resolve ``name`` LENIENTLY, falling back to ``default`` for None/unknown.

    Use this on INTERNAL paths (a brief's free-text ``material`` string, legacy callers) where a
    crash on an odd value would be worse than a safe default. The CLI uses the strict variant.
    """
    if name is None:
        return MATERIALS[default]
    try:
        return get_material(name)
    except UnknownMaterialError:
        return MATERIALS[default]


def strength_allowable(material: Material) -> tuple[float | None, str | None]:
    """The stress allowable (MPa) + its CITED basis for the yield-safety screen (ADR-0107), or
    ``(None, None)`` when no defensible basis exists -- the stress check then reports "n/a", never a
    guessed PASS. Plastics carry an explicit conservative interlayer SCREEN; metals derive from the
    handbook yield in ``mechanical`` (certifiable, named by alloy/temper)."""
    if material.strength_allowable_mpa is not None:
        return material.strength_allowable_mpa, material.strength_basis
    if material.mechanical is not None:
        return (
            material.mechanical.yield_strength_mpa,
            f"{material.mechanical.basis} yield (handbook)",
        )
    return None, None


def mass_grams(volume_mm3: float, material: Material) -> float:
    """Mass (g) of a SOLID body of this volume in this material. mm^3 -> cm^3 x density."""
    return (volume_mm3 / 1000.0) * material.density_g_cm3


def material_cost_usd(mass_g: float, material: Material) -> float:
    """Rough material-stock cost (USD) for this mass. Material basis only -- no process cost."""
    return (mass_g / 1000.0) * material.cost_per_kg_usd


def clearance_fit(material: Material, measured_gap_mm: float) -> tuple[bool, str]:
    """Classify a measured mating gap against the material's recommended slip-fit band.

    Returns ``(ok, note)``. ``ok`` is True when the gap sits within (or close to) the band -- an
    interference (negative/zero gap) or an over-wide gap is flagged. Pure + deterministic.
    """
    lo, hi = material.clearance_mm
    if measured_gap_mm < lo - 0.05:
        return False, (
            f"gap {measured_gap_mm:.2f}mm is below {material.name}'s {lo:.2f}-{hi:.2f}mm slip-fit "
            "band -- the parts will bind / not assemble; loosen the mating dimension."
        )
    if measured_gap_mm > hi + 0.2:
        return False, (
            f"gap {measured_gap_mm:.2f}mm exceeds {material.name}'s {lo:.2f}-{hi:.2f}mm slip-fit "
            "band -- the joint will be loose / rattly; tighten the mating dimension."
        )
    return True, (
        f"gap {measured_gap_mm:.2f}mm is a good {material.name} slip fit "
        f"({lo:.2f}-{hi:.2f}mm recommended)."
    )


def material_spec(material: Material, volume_mm3: float) -> dict[str, object]:
    """A concise, JSON-serializable spec sheet for a part of ``volume_mm3`` in ``material``.

    The summary the output report carries: identity, the estimated mass + cost from the EXACT
    volume, and the manufacturing limits the part was (or should be) checked against.
    """
    m = mass_grams(volume_mm3, material)
    lo, hi = material.clearance_mm
    return {
        "id": material.id,
        "name": material.name,
        "process": material.process.value,
        "density_g_cm3": material.density_g_cm3,
        "volume_mm3": round(volume_mm3, 2),
        "mass_g": round(m, 2),
        "est_cost_usd": round(material_cost_usd(m, material), 3),
        "min_wall_mm": material.min_wall_mm,
        "min_feature_mm": material.min_feature_mm,
        "clearance_mm": [lo, hi],
        "shrinkage_comp": material.shrinkage_comp,
        "notes": material.notes,
        "caveat": "mass/cost assume 100% solid material; a comparative estimate, not print mass.",
    }


__all__ = [
    "DEFAULT_MATERIAL_ID",
    "MATERIALS",
    "Appearance",
    "Material",
    "Mechanical",
    "ProcessCategory",
    "UnknownMaterialError",
    "clearance_fit",
    "get_material",
    "mass_grams",
    "material_cost_usd",
    "material_spec",
    "resolve_material",
    "strength_allowable",
]
