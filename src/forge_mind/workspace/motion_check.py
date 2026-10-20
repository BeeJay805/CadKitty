"""Bounded sampled collision witnesses; playback never invokes the CAD kernel."""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from forge_mind.workspace.capacity import enforce_limit, optional_limit
from forge_mind.workspace.geometry_results import boolean_overlap_components, boolean_volume
from forge_mind.workspace.models import Record


class MotionCheckSettings(Record):
    samples_per_driver: int = Field(default=7, ge=3)
    include_corners: bool = True
    max_configurations: int | None = Field(default=None, ge=1)


def configurations(rig: dict[str, Any], settings: MotionCheckSettings) -> list[dict[str, float]]:
    rest = {d["id"]: d["reference"] for d in rig["drivers"]}
    samples: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}

    def add(sample: dict[str, float]) -> None:
        samples[tuple(sorted(sample.items()))] = sample
        enforce_limit(len(samples), settings.max_configurations, "motion configurations")

    add(rest)
    # Independent nonuniform sweeps avoid repeated coupled tooth/screw phases.
    for driver in rig["drivers"]:
        fractions = itertools.chain(
            (0.0, 1.0),
            ((i * 0.6180339887498949) % 1 for i in range(1, settings.samples_per_driver - 1)),
        )
        for fraction in fractions:
            # Convex interpolation avoids overflow from subtracting opposite large limits.
            value = (1 - fraction) * driver["minimum"] + fraction * driver["maximum"]
            add({**rest, driver["id"]: float(value)})
    if settings.include_corners:
        # Linear extreme cover, explicitly not the full Cartesian product of corners.
        for endpoint, opposite in (("minimum", "maximum"), ("maximum", "minimum")):
            extreme = {d["id"]: d[endpoint] for d in rig["drivers"]}
            add(extreme)
            for driver in rig["drivers"]:
                add({**extreme, driver["id"]: driver[opposite]})
    return list(samples.values())


def broadphase_pairs(boxes: list[Any]) -> Iterator[tuple[int, int]]:
    """Sweep along X, conservatively testing the remaining axes for active boxes."""
    active: list[int] = []
    for i in sorted(range(len(boxes)), key=lambda index: boxes[index][0, 0]):
        low = boxes[i][0, 0] - 1e-6
        active = [j for j in active if boxes[j][1, 0] >= low]
        for j in active:
            if bounds_overlap(boxes[i], boxes[j]):
                yield min(i, j), max(i, j)
        active.append(i)


def shape_bounds(shape: Any) -> Any:
    box = shape.bounding_box()
    return np.asarray([tuple(box.min), tuple(box.max)])


def bounds_overlap(a: Any, other: Any) -> bool:
    # A small expansion prevents roundoff from culling thin real intersections.
    return bool(np.all(np.minimum(a[1], other[1]) >= np.maximum(a[0], other[0]) - 1e-6))


def review_poses(
    solids: list[Any],
    names: list[str],
    groups: list[str | None],
    poses: list[tuple[dict[str, float], list[Any]]],
    *,
    optimized: bool = True,
    seconds: float | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    reference_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    from forge_mind.workspace.engineering import transform_shape

    if not (len(solids) == len(names) == len(groups)):
        raise ValueError("Motion review requires complete body bindings and poses")
    if reference_values is not None and (not poses or poses[0][0] != reference_values):
        raise ValueError("Reference pose must be checked first before rigid-pair reuse")
    enforce_limit(len(solids), optional_limit("BODY_LIMIT"), "motion bodies")
    start, cpu = time.perf_counter(), time.process_time()
    corners = [np.array([(*p, 1) for p in itertools.product(*shape_bounds(s).T)]) for s in solids]
    failures: list[dict[str, Any]] = []
    errors: list[str] = []
    invariant: set[tuple[int, int]] = set()
    exact = culled = reused = completed = collision_count = 0
    pairs_per_pose = len(solids) * (len(solids) - 1) // 2
    total = len(poses) * pairs_per_pose
    checked_poses: list[dict[str, float]] = []
    reference_collisions: list[dict[str, Any]] = []
    reference_completed = False
    stopped = False
    for values, transforms in poses:
        previous_errors = len(errors)
        if len(transforms) != len(solids):
            raise ValueError("Missing motion transforms")
        if any(
            np.shape(matrix) != (4, 4) or not np.isfinite(matrix).all() for matrix in transforms
        ):
            raise ValueError("Motion transforms must be finite 4 by 4 matrices")
        points = [c @ m.T for c, m in zip(corners, transforms, strict=True)]
        if any(not np.isfinite(point).all() for point in points):
            raise ValueError("Transformed motion bounds must remain finite")
        boxes = [np.array([p[:, :3].min(axis=0), p[:, :3].max(axis=0)]) for p in points]
        posed: dict[int, Any] = {}
        if seconds is not None and time.perf_counter() - start > seconds:
            errors.append("Motion check budget exhausted; remaining pairs were not checked")
            stopped = True
            break
        candidates = 0
        pairs = (
            broadphase_pairs(boxes) if optimized else itertools.combinations(range(len(solids)), 2)
        )
        for i, j in pairs:
            candidates += 1
            if progress and completed % 32 == 0:
                progress(
                    {
                        "completed_pairs": completed,
                        "total_pairs": total,
                        "completed_poses": len(checked_poses),
                        "total_poses": len(poses),
                    }
                )
            if seconds is not None and time.perf_counter() - start > seconds:
                errors.append("Motion check budget exhausted; remaining pairs were not checked")
                stopped = True
                break
            if optimized and groups[i] == groups[j] and (i, j) in invariant:
                reused += 1
                completed += 1
                continue
            if optimized and not bounds_overlap(boxes[i], boxes[j]):
                culled += 1
            else:
                try:
                    for k in (i, j):
                        if k not in posed:
                            posed[k] = transform_shape(solids[k], transforms[k])
                    common = posed[i] & posed[j]
                    volume = boolean_volume(common)
                    exact += 1
                    if volume > 1e-5:
                        collision_count += 1
                        witness = {
                            "a": names[i],
                            "b": names[j],
                            "values": values,
                            "overlap_mm3": volume,
                        }
                        if len(failures) < 200:
                            failures.append(witness)
                        if reference_values is not None and values == reference_values:
                            reference_collisions.append(witness)
                        witness["overlap_components"] = boolean_overlap_components(common)
                except Exception as exc:
                    errors.append(f"{names[i]} / {names[j]}: {type(exc).__name__}")
                    continue
            completed += 1
            if groups[i] == groups[j]:
                invariant.add((i, j))
        if not stopped and optimized:
            # The completed sweep proves all omitted pairs have disjoint bounds.
            disjoint = pairs_per_pose - candidates
            culled += disjoint
            completed += disjoint
        if not stopped and len(errors) == previous_errors:
            checked_poses.append(values)
            if reference_values is not None and values == reference_values:
                reference_completed = True
        if progress:
            progress(
                {
                    "completed_pairs": completed,
                    "total_pairs": total,
                    "completed_poses": len(checked_poses),
                    "total_poses": len(poses),
                }
            )
        if stopped:
            break
    return {
        "status": "fail" if failures else "unverified",
        "coverage": "incomplete" if errors else "discrete_only",
        "collisions": failures,
        "reference_values": reference_values,
        "reference_completed": reference_completed,
        "reference_collisions": reference_collisions,
        "volume_guard_mm3": 1e-5,
        "collision_count": collision_count,
        "witness_limit": 200,
        "errors": errors,
        "requested_configurations": [v for v, _ in poses],
        "completed_configurations": checked_poses,
        "completed_pairs": completed,
        "total_pairs": total,
        "exact_intersections": exact,
        "disjoint_bounds": culled,
        "rigid_pair_reuses": reused,
        "wall_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu,
        "scope": "All body pairs at listed configurations only. Same rigid group pairs reused; no excluded contact pairs. No continuous clearance, joint feasibility, thread/contact physics or strength certificate. Independent-driver sweeps and a linear sample of corners do not cover all combinations.",
    }


def run_motion_check(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    import build123d as b
    import psutil

    from forge_mind.validation_provenance import file_sha256
    from forge_mind.workspace.motion_rig import group_matrices

    source = Path(payload["source"])
    motion = payload["motion"]
    if (
        file_sha256(source / "part.step") != motion["step_sha256"]
        or file_sha256(source / "document.json") != motion["document_sha256"]
    ):
        raise ValueError("Motion source hashes changed")
    settings = MotionCheckSettings.model_validate(payload["settings"])
    shape = b.import_step(str(source / "part.step"))
    solids = list(shape.solids())
    bindings = sorted(motion["bindings"], key=lambda r: r["body_index"])
    if [r["body_index"] for r in bindings] != list(range(len(solids))):
        raise ValueError("Motion check requires complete exported bindings")
    poses = []
    for values in configurations(motion["rig"], settings):
        matrices = group_matrices(motion["rig"], values)
        poses.append((values, [matrices.get(r["group"], np.eye(4)) for r in bindings]))

    def progress(value: dict[str, Any]) -> None:
        path = directory / "progress.tmp"
        try:
            path.write_text(json.dumps(value), encoding="utf-8")
            path.replace(directory / "progress.json")
        except OSError:
            # Windows readers can briefly deny replacement. Telemetry is
            # best-effort; final evidence is written separately and must succeed.
            pass

    result = review_poses(
        solids,
        [r["path"] for r in bindings],
        [r["group"] for r in bindings],
        poses,
        seconds=payload.get("max_seconds"),
        progress=progress,
        reference_values={d["id"]: d["reference"] for d in motion["rig"]["drivers"]},
    )
    process = psutil.Process()
    return {
        **result,
        "revision_id": motion["revision_id"],
        "engine_fingerprint": motion["engine_fingerprint"],
        "step_sha256": motion["step_sha256"],
        "document_sha256": motion["document_sha256"],
        "settings": settings.model_dump(),
        "sampling": {
            "method": "independent_nonuniform_sweeps_and_linear_extreme_cover",
            "driver_count": len(motion["rig"]["drivers"]),
            "corner_strategy": "global_extremes_and_single_driver_reversals"
            if settings.include_corners
            else "none",
            "all_corners_sampled": settings.include_corners and len(motion["rig"]["drivers"]) <= 3,
            "continuous_coverage": False,
            "configuration_count": len(poses),
        },
        "worker_cpu_affinity": process.cpu_affinity() if hasattr(process, "cpu_affinity") else None,
    }
