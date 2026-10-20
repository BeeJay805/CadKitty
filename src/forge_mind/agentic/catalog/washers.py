"""Washer catalog: every common-metric plain + chamfered washer from bd_warehouse's ISO tables.

Inner diameter (ID), outer diameter (OD), and thickness come straight from each bd_warehouse
washer's ISO data row -- never hand-typed. Four plain forms (ISO 7089 normal, 7091 narrow, 7093
wide, 7094 extra-wide) plus the ISO 7090 chamfered form, across the standard metric size range each
form defines (M1.6 .. M36). The normal flat washer keeps the canonical
``washer_flat_m<size>_normal`` id (the fasteners module references that convention).
``forge_mind.dsl.standards.washer`` carries the same ISO 7089 dims, so the normal form cross-checks.
Geometry is built on demand by the ``bd_fastener`` backend. ~80 entries.
"""

from __future__ import annotations

from forge_mind.agentic.catalog.registry import CatalogIssue, CatalogPart, register_builder

# Standard metric sizes we list (each form contributes only the ones its ISO table defines).
_SIZES: tuple[str, ...] = (
    "M1.6",
    "M2",
    "M2.5",
    "M3",
    "M3.5",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M10",
    "M12",
    "M14",
    "M16",
    "M18",
    "M20",
    "M22",
    "M24",
    "M27",
    "M30",
    "M33",
    "M36",
)

# Each form: bd_warehouse class, fastener_type (ISO column prefix), id suffix, subtype, search words.
_FORMS: tuple[tuple[str, str, str | None, str, str], ...] = (
    ("PlainWasher", "iso7089", "normal", "flat_washer_normal", "plain flat normal regular"),
    ("PlainWasher", "iso7091", "narrow", "flat_washer_narrow", "plain flat narrow small"),
    ("PlainWasher", "iso7093", "wide", "flat_washer_wide", "plain flat wide large fender"),
    ("PlainWasher", "iso7094", "xwide", "flat_washer_extra_wide", "plain flat extra wide penny"),
    ("ChamferedWasher", "iso7090", None, "flat_washer_chamfered", "chamfered bevelled hardened"),
)

# Fit convention (provisional): how much clearance over the screw thread counts as "loose".
_LOOSE_CLEARANCE_MM = 1.5


def _size_token(size: str) -> str:
    """Compact, filesystem-safe size token: 'M5' -> 'm5', 'M2.5' -> 'm2p5', 'M1.6' -> 'm1p6'."""
    return "m" + size.lstrip("Mm").replace(".", "p")


def _row_dims(row: dict[str, str], ftype: str) -> tuple[float, float, float] | None:
    """Parse (inner_d, outer_d, thickness) mm from a fastener_data row for one ISO type.

    A column may be '' (that size is not defined for this type) -> return None to skip.
    """
    raw_id = row.get(f"{ftype}:d1", "")
    raw_od = row.get(f"{ftype}:d2", "")
    raw_h = row.get(f"{ftype}:h", "")
    if raw_id == "" or raw_od == "" or raw_h == "":
        return None
    return float(raw_id), float(raw_od), float(raw_h)


@register_builder
def build_washers() -> list[CatalogPart]:
    from bd_warehouse.fastener import ChamferedWasher, PlainWasher

    classes = {"PlainWasher": PlainWasher, "ChamferedWasher": ChamferedWasher}
    parts: list[CatalogPart] = []
    for cls_name, ftype, suffix, subtype, words in _FORMS:
        cls = classes[cls_name]
        for size in _SIZES:
            row = cls.fastener_data.get(size)
            if row is None:
                continue  # this size is not in the table at all
            dims = _row_dims(row, ftype)
            if dims is None:
                continue  # this form does not define this size
            inner_d, outer_d, thickness = dims
            tok = _size_token(size)
            pid = f"washer_{tok}_chamfered" if suffix is None else f"washer_flat_{tok}_{suffix}"
            label = (
                "chamfered" if suffix is None else f"flat {suffix.replace('xwide', 'extra wide')}"
            )
            screw_size = size.lower()
            tags = [
                "washer",
                subtype,
                tok,
                ftype,
                screw_size,
                *words.split(),
            ]
            parts.append(
                CatalogPart(
                    id=pid,
                    name=f"{size} {label} washer ({ftype.upper().replace('ISO', 'ISO ')})",
                    family="washer",
                    subtype=subtype,
                    dims_mm={
                        "inner_d": inner_d,
                        "outer_d": outer_d,
                        "thickness": thickness,
                    },
                    source=f"bd_warehouse {ftype} (ISO washer data: d1/d2/h)",
                    backend="bd_fastener",
                    geo_params={
                        "cls": cls_name,
                        "size": size,
                        "fastener_type": ftype,
                    },
                    tags=tuple(dict.fromkeys(tags)),  # de-dup, keep order
                    compatible={"screw_size": [screw_size]},
                    notes=(
                        f"passes a {size} screw (bore {inner_d:g}mm); "
                        f"{outer_d:g}mm OD spreads load, {thickness:g}mm thick."
                    ),
                )
            )
    return parts


def validate_washer_fit(washer: CatalogPart, screw_thread_d: float) -> list[CatalogIssue]:
    """Check a washer's bore against a screw's thread diameter (the screw must pass through it)."""
    issues: list[CatalogIssue] = []
    if washer.family != "washer":
        return [CatalogIssue("error", f"{washer.id} is not a washer")]
    if screw_thread_d <= 0:
        issues.append(CatalogIssue("error", "screw thread diameter must be positive"))
        return issues
    inner_d = washer.dims_mm["inner_d"]
    if inner_d < screw_thread_d:
        issues.append(
            CatalogIssue(
                "error",
                f"washer bore {inner_d:g}mm is smaller than the {screw_thread_d:g}mm screw thread "
                f"-- the screw will not pass through; pick a larger washer.",
            )
        )
    elif inner_d > screw_thread_d + _LOOSE_CLEARANCE_MM:
        issues.append(
            CatalogIssue(
                "warn",
                f"washer bore {inner_d:g}mm is loose on a {screw_thread_d:g}mm screw "
                f"(>{_LOOSE_CLEARANCE_MM}mm clearance); a narrower washer would seat better.",
            )
        )
    return issues


def washer_stack_height(washers: list[CatalogPart]) -> float:
    """Total stacked thickness (mm) of a list of washers."""
    return sum(float(w.dims_mm["thickness"]) for w in washers)


__all__ = ["build_washers", "validate_washer_fit", "washer_stack_height"]
