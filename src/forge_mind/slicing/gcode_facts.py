"""Deterministic gcode-fact extraction for the print-lane slicer gate (ADR-0012 §3).

Parses a sliced gcode into the physical facts the mesh-level validator cannot see:

  (a) slicer WARNINGS (captured from the slicer's stdio by the wrapper, threaded through here);
  (b) SUPPORT material volume -- the parts are meant to be self-supporting (ADR-0012 §4), so any
      support at all is a signal. Summed from the filament extruded inside support feature blocks
      (``;TYPE:Support material``) times the filament cross-section: deposited volume is conserved,
      so no line-width guess is needed;
  (c) FIRST-LAYER contact area -- the bed-adhesion footprint, = first-layer deposited volume / first
      layer height (path-length x line-width, derived volumetrically);
  (d) per-LAYER printed silhouette area vs the MESH cross-section at that height -- a layer whose
      printed silhouette collapses below the mesh outline is a SILENTLY DROPPED thin feature (the
      slicer omitted a wall/tab too thin to path). The silhouette is the area enclosed by the
      EXTERNAL-PERIMETER loops (infill-independent); the mesh side is ``mesh.section`` shoelace.

No-false-PASS rule (ADR-0065): every metric returns ``None`` -- never 0.0 -- when it cannot be
parsed, so a silent no-parse renders as UNVERIFIABLE upstream rather than a spurious pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_MOVE = {"G0", "G1"}
# The dropped-feature threshold (ADR-0012 §3 fact d): a printed layer silhouette under this
# fraction of the mesh cross-section means the slicer omitted geometry the mesh has.
_DROP_FRAC = 0.60
# Ignore layers whose mesh cross-section is tiny (mm^2): a near-tangent top/bottom slice measures
# a sliver where a small toolpath rounding reads as a huge relative miss -- pure noise, not a drop.
_MIN_MESH_AREA_MM2 = 4.0


@dataclass(frozen=True)
class GcodeFacts:
    """The gate-relevant facts parsed from one sliced gcode. Each numeric field is ``None`` when it
    could not be parsed (UNVERIFIABLE), never a fabricated 0."""

    warnings: list[str] = field(default_factory=list)
    support_volume_mm3: float | None = None
    first_layer_contact_mm2: float | None = None
    layer_count: int | None = None
    dropped_layers: list[int] | None = None  # 1-based layer indices printed < _DROP_FRAC of mesh
    min_area_ratio: float | None = None  # smallest printed/mesh silhouette ratio across layers


def _tokens(code: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in code.split()[1:]:
        if len(tok) >= 2 and tok[0].isalpha():
            try:
                out[tok[0].upper()] = float(tok[1:])
            except ValueError:
                continue
    return out


def _signed_area(loop: list[tuple[float, float]]) -> float:
    """Shoelace signed area of a closed XY loop (sign encodes winding)."""
    n = len(loop)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _mesh_section_area(mesh: Any, z: float) -> float | None:
    """Net solid cross-section area of ``mesh`` at height ``z`` (mm^2), shapely-free.

    ``mesh.section`` returns ordered discrete loops (outer CCW, holes CW); ``abs`` of the signed-area
    sum is outer-minus-holes regardless of global winding. ``None`` on any failure -- never guess."""
    try:
        section = mesh.section(plane_origin=(0.0, 0.0, z), plane_normal=(0.0, 0.0, 1.0))
        if section is None:
            return None
        loops = section.discrete  # list of (n,3) ordered vertex arrays
        if not len(loops):
            return None
        total = 0.0
        for loop in loops:
            pts = [(float(p[0]), float(p[1])) for p in loop]
            total += _signed_area(pts)
        return abs(total)
    except Exception:
        return None


def parse_gcode_facts(
    gcode_text: str,
    *,
    warnings: list[str] | None = None,
    mesh: Any | None = None,
    filament_diameter_mm: float = 1.75,
) -> GcodeFacts:
    """Extract the four gate facts from a sliced gcode. ``warnings`` are the slicer's stdio warnings
    (captured by the wrapper). ``mesh`` (the source trimesh) enables the per-layer silhouette diff;
    without it that fact stays ``None`` (UNVERIFIABLE, never a false clear)."""
    warns = list(warnings or [])
    if not gcode_text.strip():
        return GcodeFacts(warnings=warns)

    fil_area = math.pi * (filament_diameter_mm / 2.0) ** 2

    absolute_e = True  # M82 absolute (PrusaSlicer default); flipped by M83
    e_prev = 0.0
    x = y = z = 0.0
    feature = ""  # current ;TYPE:
    layer_z: list[float] = []  # z of each layer in encounter order
    # per-layer accumulators, keyed by layer index (0-based in encounter order)
    support_e = 0.0
    first_layer_e = 0.0
    first_layer_z: float | None = None
    # external-perimeter loop capture for the silhouette diff
    ext_loops_by_layer: dict[int, list[list[tuple[float, float]]]] = {}
    cur_loop: list[tuple[float, float]] = []
    cur_layer = -1
    parsed_any_e = False

    def _flush_loop() -> None:
        nonlocal cur_loop
        if cur_layer >= 0 and len(cur_loop) >= 3:
            ext_loops_by_layer.setdefault(cur_layer, []).append(cur_loop)
        cur_loop = []

    for raw in gcode_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(";"):
            comment = line[1:].strip()
            low = comment.lower()
            if low.startswith("type:"):
                _flush_loop()
                feature = comment[5:].strip()
            elif comment.upper().startswith("LAYER_CHANGE") or low.startswith("layer:"):
                _flush_loop()
                cur_layer = len(layer_z)
                # z is set by the following ;Z: or the first move; record a placeholder now
                layer_z.append(z)
            elif low.startswith("z:"):
                try:
                    zval = float(comment[2:].strip())
                    if layer_z:
                        layer_z[-1] = zval
                    z = zval
                except ValueError:
                    pass
            continue
        code = line.split(";", 1)[0].strip()
        if not code:
            continue
        cmd = code.split()[0].upper()
        if cmd == "M82":
            absolute_e = True
            continue
        if cmd == "M83":
            absolute_e = False
            continue
        if cmd == "G92":
            tok = _tokens(code)
            if "E" in tok:
                e_prev = tok["E"]
            continue
        if cmd not in _MOVE:
            continue
        tok = _tokens(code)
        new_x = tok.get("X", x)
        new_y = tok.get("Y", y)
        if "Z" in tok:
            z = tok["Z"]
            if layer_z and cur_layer == len(layer_z) - 1:
                layer_z[-1] = z
        de = 0.0
        if "E" in tok:
            de = (tok["E"] - e_prev) if absolute_e else tok["E"]
            e_prev = tok["E"] if absolute_e else e_prev + tok["E"]
            parsed_any_e = True
        extruding = de > 1e-6
        # A gcode with no explicit layer markers still yields facts: bucket by first-seen layer 0.
        if extruding and cur_layer < 0:
            cur_layer = 0
            if not layer_z:
                layer_z.append(z)
        if extruding:
            if first_layer_z is None:
                first_layer_z = z
            if abs(z - (first_layer_z or z)) < 1e-6:
                first_layer_e += de
            if "support" in feature.lower():
                support_e += de
            if feature.lower().startswith("external perimeter") or feature.lower() == "wall-outer":
                if not cur_loop:
                    cur_loop.append((x, y))  # anchor at the loop's entry point
                cur_loop.append((new_x, new_y))
        else:
            _flush_loop()  # a travel move ends the current perimeter loop
        x, y = new_x, new_y
    _flush_loop()

    n_layers = len(layer_z) if layer_z else (1 if parsed_any_e else 0)

    support_vol = support_e * fil_area if support_e > 0 else (0.0 if parsed_any_e else None)
    first_contact: float | None = None
    if parsed_any_e and first_layer_z and first_layer_z > 0 and first_layer_e > 0:
        first_contact = first_layer_e * fil_area / first_layer_z

    dropped, min_ratio = _dropped_layers(ext_loops_by_layer, layer_z, mesh)

    return GcodeFacts(
        warnings=warns,
        support_volume_mm3=support_vol,
        first_layer_contact_mm2=first_contact,
        layer_count=n_layers or None,
        dropped_layers=dropped,
        min_area_ratio=min_ratio,
    )


def _dropped_layers(
    ext_loops_by_layer: dict[int, list[list[tuple[float, float]]]],
    layer_z: list[float],
    mesh: Any | None,
) -> tuple[list[int] | None, float | None]:
    """Compare each layer's printed silhouette (external-perimeter loops) to the mesh cross-section.
    Returns (dropped 1-based layer indices, smallest printed/mesh ratio). Both ``None`` when the mesh
    or any section is unavailable -- the fact is UNVERIFIABLE, never a silent clear."""
    if mesh is None or not ext_loops_by_layer or not layer_z:
        return None, None
    # The slicer drops the part so its lowest point prints at bed z=0; the mesh keeps its authored
    # z. Section the mesh at (its own min-z + the gcode layer height) so the two align regardless of
    # where the part was modelled -- without this, a part not modelled on the XY plane false-drops.
    try:
        z_off = float(mesh.bounds[0][2])
    except Exception:
        return None, None
    dropped: list[int] = []
    min_ratio: float | None = None
    measured_any = False
    for idx, loops in sorted(ext_loops_by_layer.items()):
        if idx >= len(layer_z):
            continue
        z = layer_z[idx]
        mesh_area = _mesh_section_area(mesh, z_off + z)
        if mesh_area is None or mesh_area < _MIN_MESH_AREA_MM2:
            continue
        printed = abs(sum(_signed_area(loop) for loop in loops))
        ratio = printed / mesh_area
        measured_any = True
        if min_ratio is None or ratio < min_ratio:
            min_ratio = ratio
        if ratio < _DROP_FRAC:
            dropped.append(idx + 1)  # 1-based for humans
    if not measured_any:
        return None, None
    return dropped, min_ratio


__all__ = ["GcodeFacts", "parse_gcode_facts"]
