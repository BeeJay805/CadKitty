"""Bearing catalog: every bd_warehouse SKT bearing across five families.

Boundary dimensions (bore / outer diameter / width) come straight from each bd_warehouse SKT
designation (e.g. ``M8-22-7`` = bore 8, OD 22, width 7), which are ISO 15 boundary dims -- never
hand-typed. Common deep-groove designations (608, 625, ...) are mapped from the canonical
``dsl/standards`` table where they match, so a trade lookup like "608" works. Geometry is built by
the bd_warehouse class on demand. ~100 entries.
"""

from __future__ import annotations

from forge_mind.agentic.catalog.registry import CatalogIssue, CatalogPart, register_builder

# bd_warehouse bearing class -> (subtype, short id token, search words)
_FAMILIES: list[tuple[str, str, str, str]] = [
    ("SingleRowDeepGrooveBallBearing", "deep_groove_ball", "dgbb", "deep groove ball radial"),
    (
        "SingleRowCappedDeepGrooveBallBearing",
        "capped_deep_groove_ball",
        "cdgbb",
        "sealed shielded capped",
    ),
    (
        "SingleRowAngularContactBallBearing",
        "angular_contact_ball",
        "acbb",
        "angular contact thrust",
    ),
    ("SingleRowCylindricalRollerBearing", "cylindrical_roller", "crb", "cylindrical roller heavy"),
    ("SingleRowTaperedRollerBearing", "tapered_roller", "trb", "tapered roller cone cup"),
]

# FDM/print fit conventions (provisional): how close a shaft / housing bore must be to seat.
_SHAFT_TOL_MM = 0.15
_HOUSING_TOL_MM = 0.15


def _parse(designation: str) -> tuple[float, float, float] | None:
    """('M8-22-7') -> (bore, od, width) mm. None if it does not parse."""
    try:
        bore, od, width = designation.lstrip("Mm").split("-")
        return float(bore), float(od), float(width)
    except (ValueError, AttributeError):
        return None


def _trade_names() -> dict[tuple[float, float, float], str]:
    """Canonical deep-groove trade designations (608, 625, ...) from the in-repo standards table."""
    from forge_mind.dsl.standards import BEARINGS as _STD

    return {(b.bore_id_mm, b.outer_od_mm, b.width_mm): b.key for b in _STD.all()}


@register_builder
def build_bearings() -> list[CatalogPart]:
    from bd_warehouse import bearing as _b

    trade = _trade_names()
    parts: list[CatalogPart] = []
    for cls_name, subtype, short, words in _FAMILIES:
        cls = getattr(_b, cls_name, None)
        if cls is None:  # pragma: no cover - library shape changed
            continue
        for size in sorted(cls.sizes("SKT")):
            dims = _parse(size)
            if dims is None:
                continue
            bore, od, width = dims
            metric = f"{bore:g}x{od:g}x{width:g}"
            # a canonical trade name only for the deep-groove family (608 etc. are deep-groove)
            trade_key = trade.get(dims) if subtype == "deep_groove_ball" else None
            pid = f"bearing_{trade_key}" if trade_key else f"bearing_{short}_{metric}"
            tags = ["bearing", subtype, short, *words.split(), metric]
            if trade_key:
                tags.append(trade_key)
            parts.append(
                CatalogPart(
                    id=pid,
                    name=f"{trade_key or metric} {subtype.replace('_', ' ')} bearing",
                    family="bearing",
                    subtype=subtype,
                    dims_mm={"bore_d": bore, "outer_d": od, "width": width},
                    source=(
                        "dsl/standards (ISO 15) + bd_warehouse SKT"
                        if trade_key
                        else "bd_warehouse SKT (ISO 15 boundary dims)"
                    ),
                    backend="bd_bearing",
                    geo_params={"cls": cls_name, "size": size, "bearing_type": "SKT"},
                    tags=tuple(dict.fromkeys(tags)),  # de-dup, keep order
                    compatible={"shaft_d": [f"{bore:g}"], "housing_bore_d": [f"{od:g}"]},
                    notes=f"shaft fits the {bore:g}mm bore; press into a {od:g}mm housing bore.",
                )
            )
    return parts


def validate_bearing_mount(
    bearing: CatalogPart, shaft_d: float, housing_bore_d: float
) -> list[CatalogIssue]:
    """Check a shaft + housing bore against a bearing's bore + OD (FDM-ish fit tolerances)."""
    issues: list[CatalogIssue] = []
    if bearing.family != "bearing":
        return [CatalogIssue("error", f"{bearing.id} is not a bearing")]
    bore = bearing.dims_mm["bore_d"]
    od = bearing.dims_mm["outer_d"]
    if shaft_d <= 0 or housing_bore_d <= 0:
        issues.append(CatalogIssue("error", "shaft and housing bore diameters must be positive"))
    if abs(shaft_d - bore) > _SHAFT_TOL_MM:
        issues.append(
            CatalogIssue(
                "error",
                f"shaft {shaft_d:g}mm does not fit the {bore:g}mm bore "
                f"(allow +/-{_SHAFT_TOL_MM}mm); pick a {bore:g}mm shaft or a different bearing.",
            )
        )
    if abs(housing_bore_d - od) > _HOUSING_TOL_MM:
        issues.append(
            CatalogIssue(
                "error",
                f"housing bore {housing_bore_d:g}mm does not seat the {od:g}mm OD "
                f"(allow +/-{_HOUSING_TOL_MM}mm); bore the housing to {od:g}mm.",
            )
        )
    return issues


__all__ = ["build_bearings", "validate_bearing_mount"]
