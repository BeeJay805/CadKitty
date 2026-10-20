"""Deterministic missing-hole REPAIR: the loop's guarded last resort (ADR-0033 Stage 2).

When the author finishes a coherent part that is still SHORT its requested through-holes
(``features_missing`` -- the measured ~50% bracket/shroud drop rate), this cuts the missing holes
deterministically with the same verified toolkit the author should have used. It is a BACKSTOP,
not an authority:

  - It fires only where placement is DEFENSIBLE: a stated bolt circle ("31 mm bolt circle") or
    square pattern ("50 mm square pattern"), else the canonical corner/line/center layout on the
    part's dominant axis-aligned planar face. A brief whose holes span multiple regions ("two per
    leg", "each side") is VETOED -- the genus gate cannot verify POSITION, so guessing would turn
    a visible miss into an invisible wrong-part (the council's no-false-PASS warning).
  - The repaired part must RE-PASS the full validator with the deficit closed and NO new failures;
    otherwise the original is returned untouched.
  - It is recorded transparently (``LoopResult.repaired_features``) so the bench can distinguish
    "the author built it" from "the backstop saved it" -- a rising repair rate is an author
    regression, never a hidden one.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_mind.agentic.brief import DesignBrief
from forge_mind.agentic.spatial import _expected_through_holes
from forge_mind.agentic.validate import AgenticValidation, validate_authored_part
from forge_mind.fabrication import FabricationMode
from forge_mind.materials import Material

# A hole spec naming MULTIPLE regions cannot be placed from a single-face heuristic; refuse.
_MULTI_REGION_WORDS = ("per ", "each ", "both ", "side", "leg")

_HOLE_WORDS = ("hole", "bore", "bolt", "drill")  # mirrors spatial's through-hole vocabulary

_PCD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*(?:bolt\s*circle|pcd|bcd)", re.IGNORECASE)
_SQUARE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*square", re.IGNORECASE)
_METRIC_RE = re.compile(r"\bM(\d+(?:\.\d+)?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RepairOutcome:
    """A successful, re-validated repair. The caller swaps these in for the originals."""

    stl: Path
    step: Path
    validation: AgenticValidation
    holes_added: int
    note: str


def _hole_features(brief: DesignBrief) -> list[Any]:
    out = []
    for f in brief.features:
        name = f.name.lower()
        if any(w in name for w in _HOLE_WORDS) and "blind" not in name:
            out.append(f)
    return out


def _feature_text(feature: Any) -> str:
    return f"{feature.name} {feature.purpose} {feature.approx_dims_mm}".lower()


def _hole_diameter_mm(feature: Any) -> float | None:
    """Hole DIAMETER from the feature: its ISO standard first, else an explicit M-size in the text.
    None when nothing defensible is stated (the repair then refuses -- never guess a dimension)."""
    if getattr(feature, "standard", None):
        try:
            from forge_mind.agentic.standards_helpers import clearance_hole_d

            return float(clearance_hole_d(str(feature.standard)))
        except Exception:
            return None
    m = _METRIC_RE.search(_feature_text(feature))
    if m:
        try:
            from forge_mind.agentic.standards_helpers import clearance_hole_d

            return float(clearance_hole_d(f"M{m.group(1)}"))
        except Exception:
            return None
    return None


def _largest_axis_planar_face(part: Any) -> tuple[Any, str] | None:
    """The dominant AXIS-ALIGNED planar face and its normal axis name ('x'|'y'|'z'), or None.
    Only an axis-aligned face gives an unambiguous in-plane frame for pattern placement."""
    from build123d import GeomType

    best: tuple[Any, str] | None = None
    best_area = 0.0
    for f in part.faces().filter_by(GeomType.PLANE):
        try:
            n = f.normal_at()
            comps = {"x": abs(n.X), "y": abs(n.Y), "z": abs(n.Z)}
            axis = max(comps, key=lambda k: comps[k])
            if comps[axis] < 0.999:  # tilted plane: no unambiguous frame
                continue
            if f.area > best_area:
                best, best_area = (f, axis), float(f.area)
        except Exception:
            continue
    return best


def _pattern_centers(
    feature: Any, count: int, face: Any, axis: str, hole_d: float
) -> list[tuple[float, float]] | None:
    """In-plane (u, v) hole centres for the requested COUNT, or None when not defensible.
    Order of preference: a STATED bolt circle, a STATED square pattern, then the canonical
    corner/line/center layout inset from the face bounds."""
    from forge_mind.agentic.helpers import bolt_circle_points

    bb = face.bounding_box()
    spans = {
        "x": (bb.min.X, bb.max.X),
        "y": (bb.min.Y, bb.max.Y),
        "z": (bb.min.Z, bb.max.Z),
    }
    u_ax, v_ax = [a for a in ("x", "y", "z") if a != axis]
    (u0, u1), (v0, v1) = spans[u_ax], spans[v_ax]
    cu, cv = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    half_u, half_v = (u1 - u0) / 2.0, (v1 - v0) / 2.0
    inset = hole_d * 1.5 + 1.0  # land >= ~1 diameter of material around each hole

    def _fits(pts: list[tuple[float, float]]) -> bool:
        margin = hole_d / 2.0 + 0.5
        return all(
            u0 + margin <= u <= u1 - margin and v0 + margin <= v <= v1 - margin for u, v in pts
        )

    text = _feature_text(feature)
    m = _PCD_RE.search(text)
    if m:
        pts = [(cu + du, cv + dv) for du, dv in bolt_circle_points(float(m.group(1)), count)]
        return pts if _fits(pts) else None
    m = _SQUARE_RE.search(text)
    if m and count == 4:
        s = float(m.group(1)) / 2.0
        pts = [(cu - s, cv - s), (cu + s, cv - s), (cu + s, cv + s), (cu - s, cv + s)]
        return pts if _fits(pts) else None
    if count == 1:
        return [(cu, cv)]
    if count == 2:
        # along the longer in-plane axis
        if half_u >= half_v:
            pts = [(u0 + inset, cv), (u1 - inset, cv)]
        else:
            pts = [(cu, v0 + inset), (cu, v1 - inset)]
        return pts if _fits(pts) else None
    if count == 4:
        pts = [
            (u0 + inset, v0 + inset),
            (u1 - inset, v0 + inset),
            (u1 - inset, v1 - inset),
            (u0 + inset, v1 - inset),
        ]
        return pts if _fits(pts) else None
    return None  # other counts: no canonical defensible layout


def attempt_hole_repair(
    stl_path: Path,
    step_path: Path,
    brief: DesignBrief,
    work_dir: Path,
    *,
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
    material: Material | None = None,
) -> RepairOutcome | None:
    """Cut the missing through-holes into the finished part, GUARDED. Returns None (leaving the
    original untouched) whenever any precondition, the cut, or the re-validation is not clean."""
    try:
        import trimesh
        from build123d import Cylinder, Pos, Rot, export_step, export_stl, import_step

        hole_feats = _hole_features(brief)
        if len(hole_feats) != 1:
            return None  # multiple hole specs: placement across them is not defensible
        feature = hole_feats[0]
        if any(w in _feature_text(feature) for w in _MULTI_REGION_WORDS):
            return None  # "two per leg" / "each side": genus cannot verify position -- refuse

        expected = _expected_through_holes(brief)
        mesh: Any = trimesh.load(str(stl_path), force="mesh")
        genus = max(0, round(1 - mesh.euler_number / 2))
        deficit = expected - genus
        if deficit <= 0 or deficit != max(1, getattr(feature, "count", 1)):
            return None  # only the clean ALL-holes-of-the-one-feature-missing case is defensible

        hole_d = _hole_diameter_mm(feature)
        if hole_d is None:
            return None

        part = import_step(str(step_path))
        picked = _largest_axis_planar_face(part)
        if picked is None:
            return None
        face, axis = picked
        centers = _pattern_centers(feature, deficit, face, axis, hole_d)
        if centers is None:
            return None

        # Through-cutters: centred on the part's mid along the face axis, taller than the body.
        pb = part.bounding_box()
        spans = {
            "x": (pb.min.X, pb.max.X),
            "y": (pb.min.Y, pb.max.Y),
            "z": (pb.min.Z, pb.max.Z),
        }
        lo, hi = spans[axis]
        height = (hi - lo) + 4.0
        mid = (lo + hi) / 2.0
        rot = {"z": Rot(0, 0, 0), "x": Rot(0, 90, 0), "y": Rot(90, 0, 0)}[axis]
        u_ax, v_ax = [a for a in ("x", "y", "z") if a != axis]
        tool: Any = None
        for u, v in centers:
            coords = {axis: mid, u_ax: u, v_ax: v}
            cyl = (
                Pos(coords["x"], coords["y"], coords["z"])
                * rot
                * Cylinder(radius=hole_d / 2.0, height=height)
            )
            tool = cyl if tool is None else tool + cyl
        repaired = part - tool

        out_dir = work_dir / "repair"
        out_dir.mkdir(parents=True, exist_ok=True)
        new_stl = out_dir / "repaired.stl"
        new_step = out_dir / "repaired.step"
        # Keep the STEP product named after repair (the bench's named_step check; an
        # unlabeled boolean result exports as PRODUCT('SOLID')).
        with contextlib.suppress(Exception):
            repaired.label = getattr(part, "label", "") or "repaired_part"
        export_stl(repaired, str(new_stl))
        export_step(repaired, str(new_step))

        # THE GUARD: full re-validation; the deficit must close with no new hard failures.
        val = validate_authored_part(
            new_stl, new_step, brief, fabrication_mode=fabrication_mode, material=material
        )
        if not val.hard_ok or val.features_missing or val.body_count != 1:
            return None
        return RepairOutcome(
            stl=new_stl,
            step=new_step,
            validation=val,
            holes_added=deficit,
            note=(
                f"deterministic repair: cut {deficit} x {hole_d:.1f} mm through-hole(s) "
                f"({feature.name}) on the dominant {axis.upper()}-normal face; re-validated"
            ),
        )
    except Exception:
        return None  # any surprise leaves the original part untouched


__all__ = ["RepairOutcome", "attempt_hole_repair"]
