"""Interface + stated-tube audits (ADR-0120): the measured answer to the L68 fidelity holes.

Two audits born from one measured part (the first validator-green manifold):

  A. STATED TUBE/BOSS OD -- the brief said 24 mm OD runners; the build's convex geometry
     said otherwise and nothing measured it. When a feature states an outer diameter and
     the B-rep exposes convex cylinders, the nearest measured OD must sit within the
     stated band. When a feature states an OD and NOTHING convex-cylindrical matches at
     all, that absence is said out loud (advisory -- the author may have built tapered/
     swept geometry the extractor cannot certify; the critic still owns looks).

  B. BORE-TO-BORE LIGAMENT -- the flange bolt holes were placed 0.05 mm from the port
     walls (center distance 12.85 vs summed radii 12.80): a bolt can never seat, and no
     gate looked. A NEAR-MISS ligament (0 < web < the gate floor) between same-axis
     neighboring bores is never intentional and GATES. Fully intersecting bores
     (ligament <= 0) stay ADVISORY: cross-drilled passages are a legitimate design.

Fail-open discipline throughout: anything unparseable or unmeasurable contributes
nothing. Calibration contract (same as ADR-0117/0118): the archived accepted corpus must
replay with ZERO new gating findings before either audit wires in as gating.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


def interface_audit_enabled() -> bool:
    """OPT-IN via FORGE_INTERFACE_AUDIT=1 -- its OWN flag (ADR-0119 attributability
    contract: every lever bisects and reverts alone). Flag off = validate byte-identical."""
    return os.environ.get("FORGE_INTERFACE_AUDIT") == "1"


if TYPE_CHECKING:
    from forge_mind.agentic.brief import DesignBrief

# The envelope/stated-dims family band: generous on purpose (authors round; meshes sample).
_OD_REL_BAND = 0.20
# Ligament thresholds (PLA-scale): below the floor a web cannot print or seat a fastener.
_LIGAMENT_GATE_MM = 1.0
_LIGAMENT_ADVISORY_MM = 2.0
# Bores participate in a ligament pair only when roughly coplanar along their shared axis
# (same slab/flange); far-apart features (a deck hole vs an outlet hole) never pair.
_AXIS_SLAB_MM = 15.0
_AXIS_PARALLEL = 0.996  # ~5 deg, mirrors mapped_dims

_OD_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mm\s*(?:outer[\s-]*diam\w*|od\b)|od\s*[=:]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_WALL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*wall", re.IGNORECASE)


@dataclass(frozen=True)
class InterfaceFinding:
    check: str  # "stated_od" | "stated_od_unmeasured" | "bore_ligament"
    message: str
    gates: bool


def _stated_tubes(brief: DesignBrief) -> list[tuple[str, float, float | None, int]]:
    """(feature name, stated OD, stated wall|None, count) for every feature whose dims
    text states an outer diameter. Purpose text is never read (the fan_shroud lesson)."""
    out: list[tuple[str, float, float | None, int]] = []
    for f in brief.features:
        dims = str(getattr(f, "approx_dims_mm", "") or "")
        m = _OD_RE.search(dims)
        if not m:
            continue
        od = float(next(g for g in m.groups() if g is not None))
        if od <= 0:
            continue
        wm = _WALL_RE.search(dims)
        wall = float(wm.group(1)) if wm else None
        count = getattr(f, "count", None)
        n = int(count) if isinstance(count, int) and count >= 1 else 1
        out.append((f.name, od, wall, n))
    return out


def check_stated_ods(brief: DesignBrief, stacks: list[Any]) -> list[InterfaceFinding]:
    """Audit A. ``stacks`` are cluster_coaxial(extract_cylinders(part)) results; convex
    stacks (concave is not True) are the measurable outer diameters."""
    findings: list[InterfaceFinding] = []
    convex_ds = sorted(
        2.0 * float(s.radius) for s in stacks if getattr(s, "concave", None) is not True
    )
    for name, od, wall, count in _stated_tubes(brief):
        band = _OD_REL_BAND * od
        if not convex_ds:
            findings.append(
                InterfaceFinding(
                    "stated_od_unmeasured",
                    f"STATED OD UNMEASURED (advisory): '{name}' states {od:g} mm outer "
                    "diameter but the built solid exposes NO convex cylindrical surface to "
                    "certify it -- if it was built tapered/swept/faceted, rebuild the "
                    "feature as the stated tube so its diameter is real.",
                    gates=False,
                )
            )
            continue
        matched = [d for d in convex_ds if abs(d - od) <= band]
        nearest = min(convex_ds, key=lambda d: abs(d - od))
        if not matched:
            findings.append(
                InterfaceFinding(
                    "stated_od",
                    f"STATED OD OFF: '{name}' states {od:g} mm outer diameter; the nearest "
                    f"measured convex diameter is {nearest:.1f} mm (allowed +/-{band:.1f}). "
                    "Build the feature AT its stated OD -- a stated diameter is the user's "
                    "spec, not a suggestion.",
                    gates=True,
                )
            )
        elif count >= 2 and len(matched) < count:
            findings.append(
                InterfaceFinding(
                    "stated_od",
                    f"STATED OD COUNT SHORT (advisory): '{name}' declares {count}x "
                    f"{od:g} mm OD but only {len(matched)} measured convex diameter(s) sit "
                    "in band -- instances may share surfaces or hide in blends.",
                    gates=False,
                )
            )
        if wall is not None and matched:
            # Wall stays ADVISORY at v1: it needs a coaxial concave partner and coarse
            # sampling makes it noisy; the wall GATES already own printability.
            for s in stacks:
                if getattr(s, "concave", None) is not True:
                    continue
                inner = 2.0 * float(s.radius)
                got = (matched[0] - inner) / 2.0
                if got > 0 and abs(got - wall) > max(0.75, 0.4 * wall):
                    findings.append(
                        InterfaceFinding(
                            "stated_od",
                            f"STATED WALL DRIFT (advisory): '{name}' states {wall:g} mm "
                            f"walls; a matched tube measures ~{got:.1f} mm.",
                            gates=False,
                        )
                    )
                    break
            # one wall note per feature is plenty
    return findings


def check_bore_ligaments(stacks: list[Any], np: Any) -> list[InterfaceFinding]:
    """Audit B. Near-miss webs between neighboring same-axis bores gate; intersections
    stay advisory (cross-drilling is sometimes the design)."""
    bores = [s for s in stacks if getattr(s, "concave", None) is True]
    findings: list[InterfaceFinding] = []
    seen: set[tuple[int, int]] = set()
    for i, a in enumerate(bores):
        for j in range(i + 1, len(bores)):
            b = bores[j]
            if (i, j) in seen:
                continue
            da = np.asarray(a.axis_dir, dtype=float)
            db = np.asarray(b.axis_dir, dtype=float)
            if abs(float(np.dot(da, db))) < _AXIS_PARALLEL:
                continue
            pa = np.asarray(a.axis_point, dtype=float)
            pb = np.asarray(b.axis_point, dtype=float)
            rel = pb - pa
            along = float(np.dot(rel, da))
            if abs(along) > _AXIS_SLAB_MM:
                continue  # different slabs/flanges -- never a seating conflict
            perp = rel - along * da
            center_d = float(np.linalg.norm(perp))
            lig = center_d - float(a.radius) - float(b.radius)
            if lig >= _LIGAMENT_ADVISORY_MM:
                continue
            seen.add((i, j))
            where = tuple(round(float(c), 1) for c in ((pa + pb) / 2.0))
            if lig <= 0:
                findings.append(
                    InterfaceFinding(
                        "bore_ligament",
                        f"BORES INTERSECT (advisory): two bores (d={2 * a.radius:.1f} and "
                        f"d={2 * b.radius:.1f}) overlap near {where} (ligament "
                        f"{lig:.2f} mm). If this is NOT an intended cross-drilled passage, "
                        "separate the patterns.",
                        gates=False,
                    )
                )
            elif lig < _LIGAMENT_GATE_MM:
                findings.append(
                    InterfaceFinding(
                        "bore_ligament",
                        f"BORE LIGAMENT TOO THIN: {lig:.2f} mm web between bores "
                        f"d={2 * a.radius:.1f} and d={2 * b.radius:.1f} near {where} "
                        f"(floor {_LIGAMENT_GATE_MM:g} mm). A fastener cannot seat and the "
                        "web cannot print -- move the pattern apart.",
                        gates=True,
                    )
                )
            else:
                findings.append(
                    InterfaceFinding(
                        "bore_ligament",
                        f"BORE LIGAMENT TIGHT (advisory): {lig:.2f} mm web between bores "
                        f"d={2 * a.radius:.1f} and d={2 * b.radius:.1f} near {where}.",
                        gates=False,
                    )
                )
    return findings


def interface_findings(brief: DesignBrief, part: Any) -> list[InterfaceFinding]:
    """Both audits over a loaded B-rep part. Every failure path returns [] (fail-open)."""
    try:
        import numpy as np

        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = cluster_coaxial(extract_cylinders(part))
    except Exception:
        return []
    if not stacks:
        return []
    out: list[InterfaceFinding] = []
    with contextlib.suppress(Exception):
        out.extend(check_stated_ods(brief, stacks))
    with contextlib.suppress(Exception):
        out.extend(check_bore_ligaments(stacks, np))
    return out


__all__ = [
    "InterfaceFinding",
    "check_bore_ligaments",
    "check_stated_ods",
    "interface_findings",
]
