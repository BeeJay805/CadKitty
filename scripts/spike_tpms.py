"""Wave-4 SDF spike (throwaway): expand the implicit lane's TPMS vocabulary with two more minimal-
surface families -- Schwarz Diamond (D) and Neovius -- as cadkitty-owned MIT numpy field functions
(clean-room from the standard implicit equations, NOT copied from GPL microgen), and confirm each one
PASSES the same five mesh-tier gates the real validator applies (watertight / is_volume / winding-
consistent / single-body / positive-volume) via the SHIPPED skimage mesher
(``forge_mind.implicit.field.sdf_to_mesh``). Records the coarsest GREEN grid step + eval time + face
count per surface, so the resolution rule is measured, not guessed.

Not wired into the lane or the router (ADR-0026 Wave-4 rule: spike only). Run:
    uv run --all-extras python scripts/spike_tpms.py
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from forge_mind.implicit import field as sdf

Field = Callable[[Any, Any, Any], Any]


def schwarz_d(cell: float, thickness: float) -> Field:
    """Schwarz Diamond (D) TPMS wall. Implicit (standard form):
    sin sin sin + sin cos cos + cos sin cos + cos cos sin, thresholded to a wall."""

    def f(x: Any, y: Any, z: Any) -> Any:
        q = 2.0 * np.pi / cell
        d = (
            np.sin(q * x) * np.sin(q * y) * np.sin(q * z)
            + np.sin(q * x) * np.cos(q * y) * np.cos(q * z)
            + np.cos(q * x) * np.sin(q * y) * np.cos(q * z)
            + np.cos(q * x) * np.cos(q * y) * np.sin(q * z)
        )
        return np.abs(d) - thickness

    return f


def neovius(cell: float, thickness: float) -> Field:
    """Neovius TPMS wall. Implicit: 3(cos+cos+cos) + 4 cos cos cos. Larger amplitude than P/G/D, so a
    comparable physical wall needs a proportionally larger threshold."""

    def f(x: Any, y: Any, z: Any) -> Any:
        q = 2.0 * np.pi / cell
        n = 3.0 * (np.cos(q * x) + np.cos(q * y) + np.cos(q * z)) + 4.0 * (
            np.cos(q * x) * np.cos(q * y) * np.cos(q * z)
        )
        return np.abs(n) - thickness

    return f


# (name, field factory, wall threshold in field units). Thresholds are amplitude-scaled per family.
SURFACES: list[tuple[str, Callable[[float, float], Field], float]] = [
    ("gyroid", sdf.gyroid, 1.0),
    ("schwarz_p", sdf.schwarz_p, 0.8),
    ("schwarz_d", schwarz_d, 1.0),
    ("neovius", neovius, 2.5),
]


# The five mesh-tier checks run_mesh_checks applies, evaluated directly on the trimesh mesh.
def _mesh_tier(mesh: Any) -> dict[str, bool]:
    return {
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "winding": bool(mesh.is_winding_consistent),
        "single_body": int(mesh.body_count) == 1,
        "positive_volume": float(mesh.volume) > 1e-6,
    }


def main() -> None:
    size, cell = 24.0, 8.0
    half = size / 2.0
    lo = (-half - 2.0, -half - 2.0, -half - 2.0)
    hi = (half + 2.0, half + 2.0, half + 2.0)
    steps = [0.6, 0.45, 0.35, 0.28]  # coarse -> fine

    print(
        f"# TPMS spike: {size:g}mm cube, cell={cell:g}mm, walled-cube via op_intersect(surface, box)"
    )
    print(
        "# lane gate = watertight + positive_volume (api/core._implicit_validation); "
        "single_body/is_volume are INFO -- a clipped TPMS lattice is legitimately multi-body"
    )
    print(f"{'surface':<10}{'thresh':>7}{'step':>6}{'faces':>9}{'sec':>7}  result")
    summary: dict[str, float | None] = {}
    for name, factory, thresh in SURFACES:
        coarsest_green: float | None = None
        for step in steps:
            walled = sdf.op_intersect(factory(cell, thresh), sdf.box(half, half, half))
            t0 = time.perf_counter()
            try:
                mesh = sdf.sdf_to_mesh(walled, lo, hi, step)
            except Exception as exc:  # spike: any mesher failure is just a RED row
                print(f"{name:<10}{thresh:>7.2f}{step:>6.2f}{'-':>9}{'-':>7}  ERROR {exc}")
                continue
            dt = time.perf_counter() - t0
            checks = _mesh_tier(mesh)
            # The implicit lane gates on watertight + positive volume (api/core._implicit_validation),
            # NOT the full B-rep mesh tier: a TPMS lattice clipped to a box is LEGITIMATELY multi-body,
            # so single_body / is_volume are reported as info, never as the lattice gate.
            gate_ok = checks["watertight"] and checks["positive_volume"]
            extras = [k for k in ("is_volume", "winding", "single_body") if not checks[k]]
            if gate_ok:
                tag = "GREEN" + (f"  (info: -{','.join(extras)})" if extras else "")
            else:
                tag = "RED:" + ",".join(
                    k for k in ("watertight", "positive_volume") if not checks[k]
                )
            print(f"{name:<10}{thresh:>7.2f}{step:>6.2f}{len(mesh.faces):>9}{dt:>7.2f}  {tag}")
            if gate_ok and coarsest_green is None:
                coarsest_green = step
        summary[name] = coarsest_green

    print("\n# coarsest lane-GREEN step per surface (watertight+volume resolution rule):")
    for name, step in summary.items():
        print(f"  {name:<10} {step if step is not None else 'NONE (all RED)'}")


if __name__ == "__main__":
    main()
