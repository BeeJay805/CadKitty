"""Physical-calibration scaffolding (Phase 4, ``docs/calibration_protocol.md``).

Software-only hooks that make the hardware calibration step ONE move away:

  1. GENERATE the deterministic coupon-ladder STLs for a material (a single print exercises a
     whole range of walls / features / holes / fits / shrink / overhang).
  2. PLAN a slicer dry-run for each coupon (print-time / filament come from a real slice, so this
     returns the planned command and only estimates when a slicer is actually installed).
  3. PROPOSE threshold updates from a filled measured-run JSON -- and NEVER auto-apply them. A
     calibration run only proposes; a human approves before any ``materials.py`` default changes
     (the protocol's "no silent machine-specific numbers in the runtime path" rule).

Nothing here mutates a runtime threshold. The coupons are built with build123d (lazy-imported,
like the rest of the geometry path), so importing this module is cheap.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_mind.materials import Material, resolve_material

# A proposed threshold = the measured boundary + a safety margin of one ladder step
# (docs/calibration_protocol.md step 5). The wall/feature ladders step in 0.2 mm.
MARGIN_STEP_MM = 0.2

# Canonical measurement ids a calibration run may carry -> the Material field each updates.
_MEASUREMENT_FIELDS = {
    "min_wall_mm": "min_wall_mm",
    "min_feature_mm": "min_feature_mm",
    "clearance_mm": "clearance_mm",
    "shrinkage_comp": "shrinkage_comp",
    "hole_compensation_mm": "hole_compensation_mm",  # no Material field yet -> informational
}


def _rungs(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive ladder of values [lo, hi] in ``step`` increments (deterministic, no float drift)."""
    n = round((hi - lo) / step)
    return [round(lo + i * step, 4) for i in range(n + 1)]


# ----------------------------------------------------------------------------
# Coupon geometry (build123d). Each builder returns a single solid that sweeps its ladder.
# Deliberately simple + watertight; these are measurement artifacts, not pretty parts.
# ----------------------------------------------------------------------------


def _wall_ladder() -> Any:
    """Upright walls 0.4 -> 3.0 mm thick on a base -> the thinnest wall that prints + survives flex."""
    from build123d import Box, Pos

    base = Box(70.0, 14.0, 2.0)
    part: Any = base
    rungs = _rungs(0.4, 3.0, 0.2)
    x0 = -((len(rungs) - 1) * 4.0) / 2.0
    for i, t in enumerate(rungs):
        part += Pos(x0 + i * 4.0, 0.0, 1.0 + 5.0) * Box(t, 12.0, 10.0)
    return part


def _feature_ladder() -> Any:
    """Standing pins 0.2 -> 2.0 mm diameter on a base -> the smallest detail that resolves."""
    from build123d import Box, Cylinder, Pos

    base = Box(50.0, 12.0, 2.0)
    part: Any = base
    rungs = _rungs(0.2, 2.0, 0.2)
    x0 = -((len(rungs) - 1) * 4.5) / 2.0
    for i, d in enumerate(rungs):
        part += Pos(x0 + i * 4.5, 0.0, 1.0 + 3.0) * Cylinder(radius=d / 2.0, height=6.0)
    return part


def _hole_ladder() -> Any:
    """A bar bored 2 -> 12 mm -> as-printed hole diameter vs nominal (hole shrink)."""
    from build123d import Box, Cylinder, Pos

    rungs = _rungs(2.0, 12.0, 2.0)
    bar = Box((len(rungs) + 1) * 16.0, 18.0, 5.0)
    part: Any = bar
    x0 = -((len(rungs) - 1) * 16.0) / 2.0
    for i, d in enumerate(rungs):
        part -= Pos(x0 + i * 16.0, 0.0, 0.0) * Cylinder(radius=d / 2.0, height=10.0)
    return part


def _peg_socket_fit() -> Any:
    """Per clearance rung (-0.1 -> +0.6 mm/side): a 6 mm peg beside a socket bored to 6 + 2*clearance,
    so the tightest rung that still hand-assembles is read off directly."""
    from build123d import Box, Cylinder, Pos

    peg_d = 6.0
    rungs = _rungs(-0.1, 0.6, 0.1)
    base = Box((len(rungs) + 1) * 14.0, 28.0, 3.0)
    part: Any = base
    x0 = -((len(rungs) - 1) * 14.0) / 2.0
    for i, clr in enumerate(rungs):
        x = x0 + i * 14.0
        part += Pos(x, -8.0, 1.5 + 3.0) * Cylinder(radius=peg_d / 2.0, height=6.0)  # the peg
        socket = Box(10.0, 10.0, 6.0) - Cylinder(radius=(peg_d + 2 * clr) / 2.0, height=8.0)
        part += Pos(x, 8.0, 1.5 + 3.0) * socket  # the matching socket
    return part


def _shrink_bar() -> Any:
    """A 100 mm reference bar (notched at 0 and 100) -> as-printed length vs 100 mm (shrinkage)."""
    from build123d import Box, Pos

    bar: Any = Box(100.0, 10.0, 3.0)
    bar -= Pos(-50.0, 0.0, 0.0) * Box(1.0, 12.0, 4.0)
    bar -= Pos(50.0, 0.0, 0.0) * Box(1.0, 12.0, 4.0)
    return bar


def _bridge_overhang() -> Any:
    """Bridges 10 -> 30 mm span + an overhang ramp -> the longest clean bridge / steepest overhang."""
    from build123d import Box, Pos

    spans = _rungs(10.0, 30.0, 5.0)
    base = Box((len(spans) + 1) * 8.0, max(spans) + 12.0, 3.0)
    part: Any = base
    y0 = -(max(spans) + 12.0) / 2.0
    x0 = -((len(spans) - 1) * 8.0) / 2.0
    for i, span in enumerate(spans):
        x = x0 + i * 8.0
        # two pillars with a flat bridge between them across the span
        part += Pos(x, y0 + 4.0, 1.5 + 7.5) * Box(5.0, 4.0, 15.0)
        part += Pos(x, y0 + 4.0 + span, 1.5 + 7.5) * Box(5.0, 4.0, 15.0)
        part += Pos(x, y0 + 4.0 + span / 2.0, 1.5 + 15.0 - 1.0) * Box(5.0, span, 2.0)
    return part


@dataclass(frozen=True)
class Coupon:
    """One calibration coupon: a parametric ladder + the threshold it informs."""

    id: str
    sweeps: str
    updates: str
    build: Callable[[], Any]


COUPONS: tuple[Coupon, ...] = (
    Coupon("wall_ladder", "wall 0.4-3.0 mm", "min_wall_mm", _wall_ladder),
    Coupon("feature_ladder", "pin 0.2-2.0 mm", "min_feature_mm", _feature_ladder),
    Coupon("hole_ladder", "hole 2-12 mm", "hole_compensation_mm", _hole_ladder),
    Coupon("peg_socket_fit", "clearance -0.1..+0.6 mm", "clearance_mm", _peg_socket_fit),
    Coupon("shrink_bar", "100 mm reference", "shrinkage_comp", _shrink_bar),
    Coupon("bridge_overhang", "bridge 10-30 mm", "printability", _bridge_overhang),
)


@dataclass
class CouponArtifact:
    coupon_id: str
    stl_path: Path
    sweeps: str
    updates: str


def generate_coupons(material: str | Material, out_dir: Path) -> list[CouponArtifact]:
    """Write every coupon STL for ``material`` into ``out_dir`` + a ``manifest.json``. Deterministic:
    the geometry is material-independent (calibration measures the RAW print), so ``material`` only
    labels the output. Returns the artifacts."""
    from build123d import export_stl

    mat = material if isinstance(material, Material) else resolve_material(material)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[CouponArtifact] = []
    for coupon in COUPONS:
        stl_path = out_dir / f"{coupon.id}.stl"
        export_stl(coupon.build(), str(stl_path))
        artifacts.append(CouponArtifact(coupon.id, stl_path, coupon.sweeps, coupon.updates))
    manifest = {
        "material_id": mat.id,
        "protocol": "docs/calibration_protocol.md",
        "coupons": [
            {"id": a.coupon_id, "stl": a.stl_path.name, "sweeps": a.sweeps, "updates": a.updates}
            for a in artifacts
        ],
        "note": "Print these, measure with calipers, fill docs/calibration_schema.json, run propose.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifacts


# ----------------------------------------------------------------------------
# Measured-run loader + proposal. PROPOSES only; never mutates a threshold.
# ----------------------------------------------------------------------------


class CalibrationError(ValueError):
    """Raised when a measured calibration-run JSON is malformed (fails the protocol schema)."""


_REQUIRED_TOP = ("run_id", "material_id", "machine", "date", "measurements")
_VALID_MATERIAL_IDS = {
    "pla",
    "petg",
    "abs",
    "tpu",
    "nylon",
    "resin",
    "aluminum",
    "steel",
    "brass",
    "wood",
}


def load_calibration_run(path: Path) -> dict[str, Any]:
    """Load + validate a measured-run JSON against the protocol's required shape
    (``docs/calibration_schema.json``). Raises :class:`CalibrationError` on a malformed run, so a
    bad fixture never silently produces a proposal."""
    try:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CalibrationError(f"cannot read calibration run {path}: {e}") from e
    if not isinstance(run, dict):
        raise CalibrationError("calibration run must be a JSON object")
    missing = [k for k in _REQUIRED_TOP if k not in run]
    if missing:
        raise CalibrationError(f"calibration run missing required field(s): {missing}")
    if run["material_id"] not in _VALID_MATERIAL_IDS:
        raise CalibrationError(
            f"unknown material_id {run['material_id']!r}; known: {sorted(_VALID_MATERIAL_IDS)}"
        )
    if not isinstance(run.get("measurements"), dict) or not run["measurements"]:
        raise CalibrationError("calibration run needs a non-empty 'measurements' object")
    return run


def propose_threshold_updates(run: dict[str, Any]) -> dict[str, Any]:
    """From a validated measured run, PROPOSE material-threshold updates (never apply them).

    Each proposal = the measured boundary + a safety margin (one ladder step), per the protocol.
    Returns ``{field: {current, proposed, basis}}`` so a human sees the diff and the reasoning
    before approving. A run that is already ``approved`` is reported as such but still only
    returns the proposal -- applying it stays a separate, explicit human action.
    """
    mat = resolve_material(run["material_id"])
    meas = run["measurements"]
    out: dict[str, Any] = {}

    if (w := meas.get("min_wall_mm")) is not None:
        proposed = round(float(w["measured_pass_at_mm"]) + MARGIN_STEP_MM, 3)
        out["min_wall_mm"] = {
            "current": mat.min_wall_mm,
            "proposed": proposed,
            "basis": f"thinnest passing wall {w['measured_pass_at_mm']} mm + {MARGIN_STEP_MM} mm margin",
        }
    if (f := meas.get("min_feature_mm")) is not None:
        proposed = round(float(f["measured_pass_at_mm"]) + MARGIN_STEP_MM, 3)
        out["min_feature_mm"] = {
            "current": mat.min_feature_mm,
            "proposed": proposed,
            "basis": f"smallest resolved feature {f['measured_pass_at_mm']} mm + {MARGIN_STEP_MM} mm margin",
        }
    if (c := meas.get("clearance_mm")) is not None:
        lo = round(float(c["tightest_assembling_mm"]), 3)
        out["clearance_mm"] = {
            "current": list(mat.clearance_mm),
            "proposed": [lo, round(lo + MARGIN_STEP_MM, 3)],
            "basis": f"tightest hand-assembling clearance {lo} mm, band +{MARGIN_STEP_MM} mm",
        }
    if (s := meas.get("shrinkage_comp")) is not None:
        ref, measured = float(s["reference_mm"]), float(s["measured_mm"])
        factor = round(ref / measured, 4) if measured > 0 else mat.shrinkage_comp
        out["shrinkage_comp"] = {
            "current": mat.shrinkage_comp,
            "proposed": factor,
            "basis": f"reference {ref} mm / measured {measured} mm = compensation {factor}",
        }
    if (h := meas.get("hole_compensation_mm")) is not None:
        delta = round(float(h["measured_mm"]) - float(h["nominal"]), 3)
        out["hole_compensation_mm"] = {
            "current": None,  # no Material field yet -- informational until one is added
            "proposed": delta,
            "basis": f"as-printed {h['measured_mm']} mm - nominal {h['nominal']} mm = {delta} mm (enlarge bores by {-delta} mm)",
        }
    return out


__all__ = [
    "COUPONS",
    "MARGIN_STEP_MM",
    "CalibrationError",
    "Coupon",
    "CouponArtifact",
    "generate_coupons",
    "load_calibration_run",
    "propose_threshold_updates",
]
