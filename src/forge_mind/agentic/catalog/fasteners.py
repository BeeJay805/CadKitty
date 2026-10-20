"""Fastener catalog: ISO metric screws/bolts across five head types.

Every dimension is read live from bd_warehouse's ISO data tables (thread + head dims) and from
``forge_mind.dsl.standards`` (clearance hole / tap drill / counterbore) -- never hand-typed. Five
head families are covered: socket-head cap (ISO 4762), button-head (ISO 7380-1), countersunk
(ISO 10642), hex-head (ISO 4017), and set screw (ISO 4026). Entries are restricted to the metric
sizes M2..M8 (the sizes ``dsl/standards`` carries hole data for) and the standard length grid,
clamped per size to each designation's catalogued ISO length range [short, long], so every entry is
a real off-the-shelf screw. Geometry is built by the bd_warehouse fastener class on demand. ~255
entries.
"""

from __future__ import annotations

from forge_mind.agentic.catalog.registry import CatalogIssue, CatalogPart, register_builder

# bd_warehouse fastener class -> (ISO fastener_type, family, subtype, short id token, search words).
_FAMILIES: list[tuple[str, str, str, str, str, str]] = [
    (
        "SocketHeadCapScrew",
        "iso4762",
        "screw",
        "socket_head_cap_screw",
        "shcs",
        "socket head cap allen",
    ),
    (
        "ButtonHeadScrew",
        "iso7380_1",
        "screw",
        "button_head_screw",
        "button",
        "button head dome rounded",
    ),
    (
        "CounterSunkScrew",
        "iso10642",
        "screw",
        "countersunk_screw",
        "csk",
        "countersunk flat flush 90",
    ),
    ("HexHeadScrew", "iso4017", "screw", "hex_head_bolt", "hexbolt", "hex head bolt wrench"),
    ("SetScrew", "iso4026", "screw", "set_screw", "setscrew", "set grub headless threaded"),
]

# Full ISO metric size set we carry (dsl/standards has hole data only for these). designation -> key.
_SIZE_TO_KEY: dict[str, str] = {
    "M2-0.4": "M2",
    "M2.5-0.45": "M2.5",
    "M3-0.5": "M3",
    "M4-0.7": "M4",
    "M5-0.8": "M5",
    "M6-1": "M6",
    "M8-1.25": "M8",
}

# Standard length grid (mm). Clamped per size to that designation's catalogued [short, long] range.
_LENGTH_GRID_MM: tuple[int, ...] = (4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40)


def _f(value: str | None) -> float | None:
    """Parse a bd_warehouse dim cell to float; '' / None / non-numeric -> None (column absent)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:  # pragma: no cover - library data is numeric
        return None


def _pitch(designation: str) -> float:
    """'M5-0.8' -> 0.8 (the thread pitch encoded in every ISO designation)."""
    return float(designation.split("-", 1)[1])


def _thread_d(key: str) -> float:
    """'M5' -> 5.0 (nominal thread/major diameter from the simple size token)."""
    return float(key[1:])


@register_builder
def build_fasteners() -> list[CatalogPart]:
    from bd_warehouse import fastener as _f_mod

    from forge_mind.dsl.standards import clearance_hole, counterbore, tap_drill

    parts: list[CatalogPart] = []
    for cls_name, ftype, family, subtype, short, words in _FAMILIES:
        cls = getattr(_f_mod, cls_name, None)
        if cls is None:  # pragma: no cover - library shape changed
            continue
        available = set(cls.sizes(ftype))
        for designation, key in _SIZE_TO_KEY.items():
            if designation not in available:
                continue
            row = {k.strip(): v for k, v in cls.fastener_data[designation].items()}
            thread_d = _thread_d(key)
            pitch = _pitch(designation)
            head_h = _f(row.get(f"{ftype}:k"))  # head height; None for set screws (headless)
            # head dia only for headed screws: dk where present (cap/button/csk); the hex-head bolt
            # has no dk, so its head is sized by the across-flats s. A set screw (no k) gets no head
            # (its s column is the internal drive socket, not a head).
            head_d = None
            if head_h is not None:
                head_d = _f(row.get(f"{ftype}:dk")) or _f(row.get(f"{ftype}:s"))
            short_len = _f(row.get(f"{ftype}:short"))
            long_len = _f(row.get(f"{ftype}:long"))
            clear_d = clearance_hole(key, "normal")
            drill_d = tap_drill(key)
            # The dsl/standards counterbore() table is ISO-4762 -- a CYLINDRICAL recess sized to a
            # SOCKET-HEAD-CAP-SCREW head. It is wrong for every other family: a countersunk head needs
            # a CONICAL countersink, a button/hex head a different recess, and a set screw has no head.
            # So only SHCS carries a counterbore; others get none (validated as such).
            is_shcs = subtype == "socket_head_cap_screw"
            cbore_d, cbore_depth = counterbore(key) if is_shcs else (0.0, 0.0)
            for length in _LENGTH_GRID_MM:
                if short_len is not None and length < short_len:
                    continue
                if long_len is not None and length > long_len:
                    continue
                dims: dict[str, float] = {
                    "thread_d": thread_d,
                    "pitch": pitch,
                    "length": float(length),
                    "clearance_hole_d": clear_d,
                    "tap_drill_d": drill_d,
                }
                if is_shcs:
                    dims["counterbore_d"] = cbore_d
                    dims["counterbore_depth"] = cbore_depth
                if head_d is not None:
                    dims["head_d"] = head_d
                if head_h is not None:
                    dims["head_h"] = head_h
                size_tok = key.lower().replace(".", "_")
                pid = f"{short}_{size_tok}x{length:g}"
                tags = [
                    "screw",
                    "fastener",
                    subtype,
                    short,
                    key.lower(),
                    f"{key.lower()}x{length:g}",
                    *words.split(),
                ]
                parts.append(
                    CatalogPart(
                        id=pid,
                        name=f"{key}x{length:g} {subtype.replace('_', ' ')}",
                        family=family,
                        subtype=subtype,
                        dims_mm=dims,
                        source=(
                            f"bd_warehouse {ftype} (thread + head dims) + "
                            "dsl/standards (ISO 273/coarse tap/ISO 4762 counterbore)"
                        ),
                        backend="bd_fastener",
                        geo_params={
                            "cls": cls_name,
                            "size": designation,
                            "length": int(length),
                            "fastener_type": ftype,
                            "simple": True,
                        },
                        tags=tuple(dict.fromkeys(tags)),  # de-dup, keep order
                        compatible={
                            "washer": [f"washer_flat_{key.lower()}_normal"],
                            "nut": [f"hexnut_{key.lower()}"],
                            "clearance_hole_d": [f"{clear_d:g}"],
                        },
                        notes=(
                            f"{key} thread, {length:g}mm long; pass through a {clear_d:g}mm "
                            f"clearance hole or tap a {drill_d:g}mm hole."
                        ),
                    )
                )
    return parts


def validate_screw_fit(screw: CatalogPart, hole_d: float) -> list[CatalogIssue]:
    """Check a clearance hole against a screw: it must pass freely yet not be sloppily oversized.

    A correct clearance hole is ``>= screw.clearance_hole_d`` (the screw passes) and not absurdly
    large (``> 2x thread_d`` warns: the head may pull through / poor location). Deterministic + pure.
    """
    issues: list[CatalogIssue] = []
    if screw.family != "screw":
        return [CatalogIssue("error", f"{screw.id} is not a screw")]
    length = screw.dims_mm.get("length", 0.0)
    if length <= 0:
        issues.append(CatalogIssue("error", f"{screw.id} has non-positive length {length:g}mm"))
    thread_d = screw.dims_mm["thread_d"]
    clearance_d = screw.dims_mm["clearance_hole_d"]
    if hole_d <= 0:
        issues.append(CatalogIssue("error", "hole diameter must be positive"))
        return issues
    if hole_d < clearance_d:
        issues.append(
            CatalogIssue(
                "error",
                f"hole {hole_d:g}mm is too small for an {thread_d:g}mm screw "
                f"(needs >= {clearance_d:g}mm clearance hole to pass).",
            )
        )
    elif hole_d > 2.0 * thread_d:
        issues.append(
            CatalogIssue(
                "warn",
                f"hole {hole_d:g}mm is sloppy for an {thread_d:g}mm screw "
                f"(> 2x thread dia); the screw will not locate well.",
            )
        )
    return issues


def screw_counterbore_ok(screw: CatalogPart, bore_d: float, depth: float) -> list[CatalogIssue]:
    """Check a CYLINDRICAL counterbore recess seats the screw head flush. Only socket-head cap screws
    take a counterbore here -- any other head type (countersunk needs a conical countersink, etc.) is
    REFUSED rather than falsely certified. ``bore_d >= counterbore_d`` (and >= the real head) and
    ``depth >= counterbore_depth``.
    """
    issues: list[CatalogIssue] = []
    if screw.family != "screw":
        return [CatalogIssue("error", f"{screw.id} is not a screw")]
    cbore_d = screw.dims_mm.get("counterbore_d")
    if cbore_d is None:  # non-SHCS / headless -> no cylindrical counterbore is defined
        return [
            CatalogIssue(
                "error",
                f"{screw.id} does not take a cylindrical counterbore (only socket-head cap screws "
                "do; a countersunk head needs a conical countersink).",
            )
        ]
    cbore_depth = screw.dims_mm["counterbore_depth"]
    head_d = screw.dims_mm.get("head_d")
    min_bore = max(cbore_d, head_d) if head_d is not None else cbore_d
    if bore_d < min_bore:
        issues.append(
            CatalogIssue(
                "error",
                f"counterbore {bore_d:g}mm is too narrow for the {screw.id} head "
                f"(needs >= {min_bore:g}mm).",
            )
        )
    if depth < cbore_depth:
        issues.append(
            CatalogIssue(
                "error",
                f"counterbore depth {depth:g}mm is too shallow to seat the {screw.id} head flush "
                f"(needs >= {cbore_depth:g}mm).",
            )
        )
    return issues


__all__ = ["build_fasteners", "screw_counterbore_ok", "validate_screw_fit"]
