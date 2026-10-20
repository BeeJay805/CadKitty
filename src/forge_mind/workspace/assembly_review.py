"""All-body interference witnesses across solved assembly poses, without sampled PASS."""

from __future__ import annotations

import itertools
from typing import Any

from forge_mind.workspace.geometry_results import boolean_volume
from forge_mind.workspace.mechanism import pose_delta_matrix
from forge_mind.workspace.mechanism_geometry import matrix_location
from forge_mind.workspace.mechanism_review import contact_poses
from forge_mind.workspace.motion_check import bounds_overlap, shape_bounds


def intersects_bounds(a: Any, other: Any) -> bool:
    left, right = a.bounding_box(), other.bounding_box()
    return all(
        min(tuple(left.max)[i], tuple(right.max)[i]) - max(tuple(left.min)[i], tuple(right.min)[i])
        > 1e-7
        for i in range(3)
    )


def assembly_motion_review(
    shape: Any, motion: dict[str, Any], assembly: dict[str, Any]
) -> dict[str, Any]:
    solids = list(shape.solids())
    names = {r["body_index"]: r["path"] for r in assembly["occurrences"]}
    bindings = {r["body_index"]: r["component"] for r in motion["bindings"]}
    if set(bindings) != set(range(len(solids))):
        raise ValueError("Motion review requires complete exported occurrence bindings")
    samples = contact_poses(motion)
    failures: list[dict[str, Any]] = []
    errors: list[str] = []
    checked = culled = 0
    fixed_pairs: set[tuple[int, int]] = set()
    grounded = {c["id"] for c in motion["graph"]["components"] if c["grounded"]}
    for sample in [motion["current"], *samples]:
        if sample["status"] != "solved":
            errors.append(f"Unsolved input angle {sample['angle_deg']}")
            continue
        posed = []
        for index, solid in enumerate(solids):
            cid = bindings[index]
            delta = pose_delta_matrix(
                sample["poses"][cid],
                motion["current"]["poses"][cid],
                motion["graph"].get("frame", "planar_xy_mm_deg"),
            )
            posed.append(matrix_location(delta) * solid)
        boxes = [shape_bounds(s) for s in posed]
        for i, j in itertools.combinations(range(len(solids)), 2):
            if bindings[i] == bindings[j] or {bindings[i], bindings[j]} <= grounded:
                if (i, j) in fixed_pairs:
                    continue
                fixed_pairs.add((i, j))
            if not bounds_overlap(boxes[i], boxes[j]):
                culled += 1
                continue
            try:
                common = posed[i] & posed[j]
                volume = boolean_volume(common)
                checked += 1
                if volume > 1e-5:
                    failures.append(
                        {
                            "a": names[i],
                            "b": names[j],
                            "angle_deg": sample["angle_deg"],
                            "overlap_mm3": volume,
                        }
                    )
            except Exception as exc:
                errors.append(f"{names[i]} / {names[j]}: {type(exc).__name__}")
    return {
        "status": "fail" if failures else "unverified",
        "collisions": failures,
        "errors": errors,
        "exact_intersections": checked,
        "disjoint_bounds": culled,
        "requested_angles": [s["angle_deg"] for s in samples],
        "coverage": "incomplete" if errors else "discrete_only",
        "scope": "All physical pairs at design pose and seven nonuniform solved inputs; same rigid group tested once. No continuous clearance, branch, contact-force or strength certification.",
    }
