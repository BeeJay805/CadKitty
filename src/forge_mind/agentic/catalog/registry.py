"""The deterministic standard-hardware catalog: schema, registry, and geometry dispatch.

The McMaster-style brain of Forge-Mind. The LLM never invents standard mechanical geometry -- it
SELECTS a catalog id (a bearing designation, a screw size, a gear preset) and deterministic code
generates the verified geometry + validation rules. Every entry's dimensions come from an
AUTHORITATIVE source (bd_warehouse's ISO/DIN data tables, ``forge_mind.dsl.standards``, or our own
involute gear math) -- never hand-typed guesses -- and carries its provenance + a ``provisional``
flag.

The registry is built LAZILY on first access: ``import catalog`` only registers each family's
builder (cheap); the first lookup runs the builders, which read the live data sources (importing
bd_warehouse on demand, never eagerly in the sandbox). Geometry is built only when requested.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# A builder reads a live data source and returns that family's entries. Families register theirs
# at import; the registry runs them once on first access (so bd_warehouse stays lazy).
_BUILDERS: list[Callable[[], list[CatalogPart]]] = []
_BY_ID: dict[str, CatalogPart] = {}
_BUILT = False


class CatalogError(ValueError):
    """Raised for an unknown catalog id; carries useful alternatives in its message."""


@dataclass(frozen=True)
class CatalogIssue:
    """One finding from a catalog validation rule. ``severity`` is 'error' or 'warn'."""

    severity: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@dataclass(frozen=True)
class CatalogPart:
    """One verified standard part. Pure data + provenance; geometry is generated on demand."""

    id: str  # stable, e.g. "bearing_608", "shcs_m5x20", "spur_m1_z24_w8_b5"
    name: str  # human display name
    family: str  # "bearing" | "gear" | "screw" | "washer" | "nut" | "spacer"
    subtype: str  # e.g. "deep_groove_ball" | "socket_head_cap_screw" | "flat_washer_normal"
    dims_mm: dict[str, float]  # family-specific measured dimensions (mm)
    source: str  # cited provenance (standard / library / generator)
    backend: str  # geometry maker: "bd_bearing" | "bd_fastener" | "native_gear" | "envelope"
    geo_params: dict[str, Any]  # params handed to the maker to build the solid
    provisional: bool = False  # True = NOT fully source-backed; verify before production
    tags: tuple[str, ...] = ()  # search terms
    compatible: dict[str, list[str]] = field(default_factory=dict)  # relation -> ids/sizes
    notes: str = ""

    @property
    def haystack(self) -> str:
        return " ".join((self.id, self.name, self.family, self.subtype, *self.tags)).lower()


def register_builder(fn: Callable[[], list[CatalogPart]]) -> Callable[[], list[CatalogPart]]:
    """Register a family's entry-builder. Run once, lazily, on first registry access."""
    _BUILDERS.append(fn)
    return fn


def _ensure_built() -> None:
    global _BUILT
    if _BUILT:
        return
    for build in _BUILDERS:
        for part in build():
            if part.id in _BY_ID:
                raise ValueError(f"duplicate catalog id {part.id!r} (from {part.family})")
            _BY_ID[part.id] = part
    _BUILT = True


# ----------------------------------------------------------------------------
# Lookup / search / introspection
# ----------------------------------------------------------------------------


def get_catalog_part(part_id: str) -> CatalogPart:
    """Resolve a catalog id, STRICTLY. An unknown id raises :class:`CatalogError` with the closest
    real alternatives -- a bad designation is a failure, never an invitation to improvise."""
    _ensure_built()
    key = part_id.strip()
    if key in _BY_ID:
        return _BY_ID[key]
    alts = suggest_catalog_ids(key)
    hint = f" Did you mean: {', '.join(alts)}?" if alts else ""
    raise CatalogError(f"unknown catalog part {part_id!r}.{hint}")


def search_catalog(query: str, *, family: str | None = None, limit: int = 20) -> list[CatalogPart]:
    """Substring/token search over id + name + tags, optionally within one family."""
    _ensure_built()
    terms = query.lower().split()
    out: list[CatalogPart] = []
    for part in _BY_ID.values():
        if family is not None and part.family != family:
            continue
        hay = part.haystack
        if all(t in hay for t in terms):
            out.append(part)
    out.sort(key=lambda p: (p.family, p.id))
    return out[:limit]


def list_catalog_family(family: str) -> list[CatalogPart]:
    """Every part in a family, sorted by id. Empty list for an unknown family name."""
    _ensure_built()
    return sorted((p for p in _BY_ID.values() if p.family == family), key=lambda p: p.id)


def suggest_catalog_ids(bad_id: str, n: int = 5) -> list[str]:
    """The closest real catalog ids to a bad one (fuzzy), so an error can guide the caller."""
    _ensure_built()
    key = bad_id.strip().lower()
    ids = list(_BY_ID)
    close = difflib.get_close_matches(key, [i.lower() for i in ids], n=n, cutoff=0.45)
    lower_to_id = {i.lower(): i for i in ids}
    out = [lower_to_id[c] for c in close]
    if not out:  # fall back to substring matches (e.g. "608" -> "bearing_608")
        out = [i for i in ids if key in i.lower()][:n]
    return out


def catalog_counts() -> dict[str, int]:
    """Entry counts per family plus the total -- the catalog's size at a glance."""
    _ensure_built()
    counts: dict[str, int] = {}
    for part in _BY_ID.values():
        counts[part.family] = counts.get(part.family, 0) + 1
    counts["total"] = len(_BY_ID)
    return counts


# ----------------------------------------------------------------------------
# Geometry dispatch (lazy: imports bd_warehouse / our gears only when building)
# ----------------------------------------------------------------------------


def catalog_geometry(part: CatalogPart | str, **overrides: Any) -> Any:
    """Build the verified build123d solid for a catalog part (by part or id). Lazy imports.

    ``overrides`` patch the geometry params (e.g. a screw ``length`` or a gear ``clock_deg``).
    """
    if isinstance(part, str):
        part = get_catalog_part(part)
    params: dict[str, Any] = {**part.geo_params, **overrides}
    backend = part.backend

    if backend == "bd_bearing":
        from bd_warehouse import bearing as _b

        cls = getattr(_b, params.pop("cls"))
        return cls(**params)
    if backend == "bd_fastener":
        from bd_warehouse import fastener as _f

        cls = getattr(_f, params.pop("cls"))
        return cls(**params)
    if backend == "native_gear":
        from forge_mind.agentic.gears import ring_gear, spur_gear

        kind = params.pop("kind", "spur")
        return (ring_gear if kind == "ring" else spur_gear)(**params)
    if backend == "envelope":
        return _envelope_solid(params)
    raise CatalogError(f"part {part.id!r} has unknown geometry backend {backend!r}")


def _envelope_solid(params: dict[str, Any]) -> Any:
    """A documented PLACEHOLDER solid (provisional geometry) from boundary dimensions -- a bounding
    ring (bearings) or cylinder, used only when no verified generator covers a part."""
    from build123d import Cylinder, Pos

    shape = params.get("shape", "ring")
    height = float(params["height"])
    if shape == "ring":
        outer = Cylinder(radius=float(params["outer_d"]) / 2.0, height=height)
        bore = Cylinder(radius=float(params["bore_d"]) / 2.0, height=height + 2.0)
        solid = outer - bore
    else:
        solid = Cylinder(radius=float(params["outer_d"]) / 2.0, height=height)
    return Pos(0.0, 0.0, height / 2.0) * solid


__all__ = [
    "CatalogError",
    "CatalogIssue",
    "CatalogPart",
    "catalog_counts",
    "catalog_geometry",
    "get_catalog_part",
    "list_catalog_family",
    "register_builder",
    "search_catalog",
    "suggest_catalog_ids",
]
