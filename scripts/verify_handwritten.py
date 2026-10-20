"""Phase 0 verification: load the hand-coded NEMA 17 STL and run sanity checks.

This is a stand-in for the real Validator (Phase 2). It checks the things the
Validator will eventually check, against the spec the part claims to satisfy.

Exit code 0 = pass; 1 = fail (with first failure printed).

Run with:
    uv run python scripts/verify_handwritten.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import trimesh

# Spec (must stay in sync with examples/handwritten_nema17.py).
SPEC = {
    "stl_path": Path(__file__).parent.parent / "examples" / "handwritten_nema17.stl",
    "max_bbox_xyz": (60.0, 60.0, 40.0),  # bench-001 hard cap
    "expected_bbox_xyz": (43.0, 50.0, 30.0),  # what this design actually produces
    "bbox_tol": 0.2,  # mm; mesh tessellation tolerance
    "min_volume_mm3": 9000.0,  # below this means a hole leaked or a flange vanished
    "max_volume_mm3": 14000.0,  # above this means a hole is missing
    "expected_volume_mm3": 11252.89,  # from examples/handwritten_nema17.py
    "volume_tol_pct": 1.0,  # %; mesh tessellation
}


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print(f"Loading {SPEC['stl_path']}")
    if not SPEC["stl_path"].exists():
        print(
            "STL missing. Run examples/handwritten_nema17.py first.",
            file=sys.stderr,
        )
        return 1

    mesh = trimesh.load(str(SPEC["stl_path"]), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh), f"Expected Trimesh, got {type(mesh)}"

    all_passed = True

    # ------------------------------------------------------------------------
    # Tier 1 surrogate: mesh checks
    # ------------------------------------------------------------------------
    print("\nTier 1 — mesh:")
    all_passed &= check("is_watertight", mesh.is_watertight)
    all_passed &= check("is_volume", mesh.is_volume)
    all_passed &= check("is_winding_consistent", mesh.is_winding_consistent)
    all_passed &= check(
        "single_body",
        mesh.body_count == 1,
        f"body_count={mesh.body_count}",
    )

    # ------------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------------
    print("\nBounding box:")
    bbox = mesh.bounding_box.extents  # (X, Y, Z) sizes
    print(f"  actual: X={bbox[0]:.3f}  Y={bbox[1]:.3f}  Z={bbox[2]:.3f}")

    for axis, actual, max_cap, expected in zip(
        ("X", "Y", "Z"),
        bbox,
        SPEC["max_bbox_xyz"],
        SPEC["expected_bbox_xyz"],
        strict=True,
    ):
        all_passed &= check(
            f"{axis} <= bench-001 cap ({max_cap}mm)",
            actual <= max_cap + SPEC["bbox_tol"],
            f"{actual:.3f}",
        )
        all_passed &= check(
            f"{axis} matches design ({expected}mm)",
            math.isclose(actual, expected, abs_tol=SPEC["bbox_tol"]),
            f"{actual:.3f}",
        )

    # ------------------------------------------------------------------------
    # Volume sanity
    # ------------------------------------------------------------------------
    print("\nVolume:")
    vol = float(mesh.volume)
    print(f"  actual: {vol:.2f} mm^3")
    all_passed &= check(
        f"within [{SPEC['min_volume_mm3']}, {SPEC['max_volume_mm3']}] mm^3",
        SPEC["min_volume_mm3"] <= vol <= SPEC["max_volume_mm3"],
        f"{vol:.2f}",
    )
    vol_err_pct = abs(vol - SPEC["expected_volume_mm3"]) / SPEC["expected_volume_mm3"] * 100
    all_passed &= check(
        f"within {SPEC['volume_tol_pct']}% of expected ({SPEC['expected_volume_mm3']:.0f})",
        vol_err_pct <= SPEC["volume_tol_pct"],
        f"err={vol_err_pct:.3f}%",
    )

    # ------------------------------------------------------------------------
    # Tier 2 surrogate: count cylindrical holes via face-clustering heuristics.
    # We're not building the real B-rep validator here — just sanity-checking
    # the mesh has *roughly* the right hole count by looking at the number of
    # connected face groups whose normals point opposite a coordinate axis.
    # ------------------------------------------------------------------------
    # Actual feature checks happen in Phase 2 against the build123d B-rep, not the mesh.
    # This surrogate just verifies the mesh isn't catastrophically wrong (no holes at all,
    # or way too many holes).
    print("\nFeature surrogate (rough):")

    # Through-Z holes (motor flange): faces with normals having |z| < 0.1 inside the
    # motor-flange Z range (0..4) form the cylinder walls.
    z_lo, z_hi = 0.0, 4.0
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    in_motor_flange = (face_centers[:, 2] >= z_lo - 0.01) & (face_centers[:, 2] <= z_hi + 0.01)
    near_z_normal = np.abs(face_normals[:, 2]) < 0.1
    motor_hole_walls = in_motor_flange & near_z_normal
    n_motor_wall_faces = int(motor_hole_walls.sum())
    all_passed &= check(
        "motor-flange cylindrical walls present (heuristic)",
        n_motor_wall_faces > 50,
        f"face count={n_motor_wall_faces} (expect >50 for 4 M3 + 1 shaft holes)",
    )

    # Through-Y holes (extrusion flange): faces with |normal.y| < 0.1 at Y >= 21.
    in_ext_flange = face_centers[:, 1] >= 21.0 - 0.01
    near_y_normal = np.abs(face_normals[:, 1]) < 0.1
    ext_hole_walls = in_ext_flange & near_y_normal
    n_ext_wall_faces = int(ext_hole_walls.sum())
    all_passed &= check(
        "extrusion-flange cylindrical walls present (heuristic)",
        n_ext_wall_faces > 20,
        f"face count={n_ext_wall_faces} (expect >20 for 2 M5 holes)",
    )

    # ------------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------------
    print()
    if all_passed:
        print("ALL CHECKS PASSED")
        return 0
    print("ONE OR MORE CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
