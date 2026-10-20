"""Nut catalog: hex, thin/jam, flanged, square, and domed-cap nuts from bd_warehouse ISO/DIN data.

Every dimension comes straight from a bd_warehouse fastener data table -- never hand-typed. The
thread diameter and pitch are parsed from the size designation (``M5-0.8`` -> thread 5 mm, pitch
0.8 mm); the width across flats (``<ftype>:s``) and thickness (``<ftype>:m``) come from the
standard's column for that nut. Five styles are covered:

* hex nut, ISO 4032 (the standard style-1 nut; ``hexnut_m5`` -- referenced by the fasteners module)
* thin / jam hex nut, ISO 4035 (``hexnut_thin_m5``)
* hex flange nut, DIN 1665 (``flangenut_m5``; fine-pitch variants get a pitch suffix)
* square nut, DIN 557 (``squarenut_m5``)
* domed cap (acorn) nut, DIN 1587 (``capnut_m5``)

bd_warehouse is imported lazily inside the builder. Geometry is built on demand by the bd_warehouse
class via the ``bd_fastener`` backend. ~89 entries.
"""

from __future__ import annotations

import math

from forge_mind.agentic.catalog.registry import CatalogIssue, CatalogPart, register_builder

# (bd_warehouse class, fastener_type, subtype, id token, source standard label, search words).
# Order matters only for readability; ids are made unique below.
_STYLES: list[tuple[str, str, str, str, str, str]] = [
    ("HexNut", "iso4032", "hex_nut", "hexnut", "ISO 4032", "hex nut style1 standard"),
    ("HexNut", "iso4035", "hex_thin_nut", "hexnut_thin", "ISO 4035", "thin jam lock hex nut"),
    (
        "HexNutWithFlange",
        "din1665",
        "hex_flange_nut",
        "flangenut",
        "DIN 1665",
        "flange flanged hex nut",
    ),
    ("SquareNut", "din557", "square_nut", "squarenut", "DIN 557", "square four sided nut"),
    ("DomedCapNut", "din1587", "domed_cap_nut", "capnut", "DIN 1587", "domed cap acorn dome nut"),
]

# A small radial clearance for a captive-nut pocket where dsl/standards has no canonical pocket.
_POCKET_CLEARANCE_MM = 0.2
# A nut fits only its own thread: tighter than a press fit, this is a thread-match check.
_THREAD_TOL_MM = 0.01


def _parse_size(designation: str) -> tuple[float, float] | None:
    """('M5-0.8') -> (thread_d, pitch) mm. None if it does not parse."""
    try:
        thread, pitch = designation.split("-")
        return float(thread.lstrip("Mm")), float(pitch)
    except (ValueError, AttributeError):
        return None


def _size_token(thread_d: float) -> str:
    """5.0 -> 'm5', 2.5 -> 'm2.5' -- the metric token used in ids and compatible tables."""
    return f"m{thread_d:g}"


def _coarse_pitches() -> dict[float, float]:
    """Thread diameter -> ISO coarse pitch, read from the (all-coarse) ISO 4032 designations.

    Used to give the standard coarse nut the clean id and suffix only fine-pitch variants.
    """
    from bd_warehouse import fastener as _f

    out: dict[float, float] = {}
    for size in _f.HexNut.sizes("iso4032"):
        parsed = _parse_size(size)
        if parsed is not None:
            out[parsed[0]] = parsed[1]
    return out


@register_builder
def build_nuts() -> list[CatalogPart]:
    from bd_warehouse import fastener as _f

    coarse = _coarse_pitches()
    parts: list[CatalogPart] = []
    for cls_name, ftype, subtype, token, std, words in _STYLES:
        cls = getattr(_f, cls_name, None)
        if cls is None:  # pragma: no cover - library shape changed
            continue
        data = cls.fastener_data
        for size in sorted(cls.sizes(ftype)):
            parsed = _parse_size(size)
            if parsed is None:
                continue
            thread_d, pitch = parsed
            row = data.get(size, {})
            waf_raw = row.get(f"{ftype}:s", "")
            thick_raw = row.get(f"{ftype}:m", "")
            if waf_raw == "" or thick_raw == "":  # a blank column = not standardised; skip
                continue
            waf = float(waf_raw)
            thickness = float(thick_raw)

            metric = _size_token(thread_d)
            # The coarse-pitch nut gets the clean id; a fine-pitch variant is disambiguated.
            is_coarse = coarse.get(thread_d, pitch) == pitch
            pid = f"{token}_{metric}" if is_coarse else f"{token}_{metric}x{pitch:g}"

            tags = ["nut", subtype, token, *words.split(), metric]
            if not is_coarse:
                tags.append("fine")
            parts.append(
                CatalogPart(
                    id=pid,
                    name=f"{metric.upper()} {subtype.replace('_', ' ')} ({std})",
                    family="nut",
                    subtype=subtype,
                    dims_mm={
                        "thread_d": thread_d,
                        "pitch": pitch,
                        "width_across_flats": waf,
                        "thickness": thickness,
                    },
                    source=f"bd_warehouse {ftype} ({std}): s/m columns",
                    backend="bd_fastener",
                    geo_params={
                        "cls": cls_name,
                        "size": size,
                        "fastener_type": ftype,
                        "simple": True,
                    },
                    tags=tuple(dict.fromkeys(tags)),  # de-dup, keep order
                    compatible={"screw_size": [metric]},
                    notes=(
                        f"threads onto an {metric.upper()} screw (pitch {pitch:g}mm); "
                        f"{waf:g}mm across flats, {thickness:g}mm thick."
                    ),
                )
            )
    return parts


def validate_nut_fit(
    nut: CatalogPart, screw_thread_d: float, screw_pitch: float | None = None
) -> list[CatalogIssue]:
    """A nut fits ONLY its own thread: error unless the screw's major diameter AND pitch match the
    nut's. ``screw_pitch`` (when given) catches a coarse/fine mismatch (an M8x1.0 nut will NOT thread
    onto an M8x1.25 screw); omit it to check the major diameter alone."""
    if nut.family != "nut":
        return [CatalogIssue("error", f"{nut.id} is not a nut")]
    issues: list[CatalogIssue] = []
    if screw_thread_d <= 0:
        issues.append(CatalogIssue("error", "screw thread diameter must be positive"))
    thread_d = nut.dims_mm["thread_d"]
    if abs(thread_d - screw_thread_d) > _THREAD_TOL_MM:
        issues.append(
            CatalogIssue(
                "error",
                f"{nut.id} threads M{thread_d:g} but the screw is M{screw_thread_d:g}; "
                f"a nut only fits its own thread -- pick an M{thread_d:g} screw.",
            )
        )
    nut_pitch = nut.dims_mm.get("pitch")
    if screw_pitch is not None and nut_pitch is not None and abs(nut_pitch - screw_pitch) > 0.001:
        issues.append(
            CatalogIssue(
                "error",
                f"{nut.id} is pitch {nut_pitch:g}mm but the screw is pitch {screw_pitch:g}mm; "
                "coarse and fine threads do not mate -- match the pitch.",
            )
        )
    return issues


def nut_pocket_dims(nut: CatalogPart) -> tuple[float, float, float]:
    """(across_flats, across_corners, depth) mm for a captive pocket that traps this nut from the
    side. Uses the canonical ``dsl/standards`` pocket where the size exists, else the nut's own WAF
    plus a small clearance (across-corners from the polygon geometry: hex /cos30, square *sqrt2)."""
    waf = nut.dims_mm["width_across_flats"]
    thickness = nut.dims_mm["thickness"]
    size_label = f"M{nut.dims_mm['thread_d']:g}"
    if nut.subtype != "square_nut":  # canonical hex pocket table only covers hex nuts
        from forge_mind.dsl.standards import captive_nut_pocket

        try:
            return captive_nut_pocket(size_label)
        except (KeyError, ValueError):
            pass
    corner_factor = (
        math.sqrt(2.0) if nut.subtype == "square_nut" else 1.0 / math.cos(math.radians(30.0))
    )
    return (
        waf + _POCKET_CLEARANCE_MM,
        waf * corner_factor + _POCKET_CLEARANCE_MM,
        thickness + _POCKET_CLEARANCE_MM,
    )


__all__ = ["build_nuts", "nut_pocket_dims", "validate_nut_fit"]
