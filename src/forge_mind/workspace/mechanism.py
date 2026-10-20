"""Bounded planar loop closure, independently evaluated residuals and local mobility."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from forge_mind.workspace.mechanism_models import Mechanism

POSITION_TOL = 1e-5  # Numerical closure instrument, not a manufacturing tolerance.
ANGLE_SCALE = 10.0  # mm/rad, fixed scaling for Jacobian conditioning.


def frame_matrix(frame: str = "planar_xy_mm_deg") -> NDArray[np.float64]:
    """Map mechanism-local x/y/depth into a right-handed principal CAD frame."""
    rotations = {
        "planar_xy_mm_deg": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "planar_yz_mm_deg": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        "planar_zx_mm_deg": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    }
    if not isinstance(frame, str) or frame not in rotations:
        raise ValueError("Unsupported planar mechanism frame")
    result = np.eye(4)
    result[:3, :3] = rotations[frame]
    return result


def pose_matrix(pose: Sequence[float], frame: str = "planar_xy_mm_deg") -> NDArray[np.float64]:
    """Convert the unchanged local [x, y, angle_degrees, depth] pose to world CAD."""
    try:
        values = list(pose)
        valid = len(values) == 4 and all(
            not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
            for value in values
        )
    except (TypeError, OverflowError) as exc:
        raise ValueError("Mechanism pose requires four finite numbers") from exc
    if not valid:
        raise ValueError("Mechanism pose requires four finite numbers")
    x, y, angle, depth = values
    angle = math.radians(angle)
    cosine, sine = math.cos(angle), math.sin(angle)
    local: NDArray[np.float64] = np.array(
        [[cosine, -sine, 0, x], [sine, cosine, 0, y], [0, 0, 1, depth], [0, 0, 0, 1]],
        dtype=float,
    )
    return frame_matrix(frame) @ local


def pose_delta_matrix(
    sample_pose: Sequence[float],
    rest_pose: Sequence[float],
    frame: str = "planar_xy_mm_deg",
) -> NDArray[np.float64]:
    """Move already-exported rest-world geometry by sample @ inverse(rest)."""
    sample, rest = pose_matrix(sample_pose, frame), pose_matrix(rest_pose, frame)
    inverse = np.eye(4)
    inverse[:3, :3] = rest[:3, :3].T
    with np.errstate(over="ignore", invalid="ignore"):
        inverse[:3, 3] = -inverse[:3, :3] @ rest[:3, 3]
        result = sample @ inverse
    if not np.isfinite(result).all():
        raise ValueError("Mechanism pose delta must remain finite")
    return result


def preserved_intent(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    old = baseline.get("mechanism")
    if not old:
        return True, "No prior mechanism"
    new = candidate.get("mechanism")
    if not new:
        return False, "Mechanism design intent was removed"
    missing: list[str] = []
    for field in ("components", "joints", "gears"):
        removed = {v["id"] for v in old[field]} - {v["id"] for v in new[field]}
        missing.extend(f"{field}.{v}" for v in sorted(removed))
    ownership = {c["id"]: c["feature"] for c in new["components"]}
    missing.extend(
        f"ownership.{c['id']}" for c in old["components"] if ownership.get(c["id"]) != c["feature"]
    )
    if missing:
        return False, "Existing mechanism identities were removed: " + ", ".join(missing)
    return (
        True,
        "Existing component, joint and coupling identities are retained; changed relationships require preview review",
    )


def resolve_mechanism(model: Mechanism, variables: dict[str, float]) -> dict[str, Any]:
    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: resolve(v)
                if k in {"x", "y", "z", "angle", "axis", "ratio", "phase", "pa", "pb"}
                or isinstance(v, (dict, list))
                else v
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [resolve(v) for v in value]
        if isinstance(value, str):
            if value not in variables:
                raise ValueError(f"Mechanism references missing variable {value}")
            return variables[value]
        if isinstance(value, (float, int)) and (not math.isfinite(value) or abs(value) > 100000):
            raise ValueError("Mechanism value exceeds finite model bounds")
        return value

    data: dict[str, Any] = resolve(model.model_dump())
    for gear in data["gears"]:
        if abs(gear["ratio"]) < 1e-6 or abs(gear["ratio"]) > 100:
            raise ValueError("Gear ratio magnitude must be between 0.000001 and 100")
    return data


def solve_mechanism(
    data: dict[str, Any], angle: float, seed: list[float] | None = None
) -> dict[str, Any]:
    components = {c["id"]: c for c in data["components"]}
    moving = [c for c in data["components"] if not c["grounded"]]
    indexes = {c["id"]: i * 3 for i, c in enumerate(moving)}
    initial = np.array([v for c in moving for v in (c["x"], c["y"], math.radians(c["angle"]))])
    x0 = np.array(seed) if seed is not None else initial

    def pose(cid: str, x: Any) -> Any:
        if cid in indexes:
            i = indexes[cid]
            return x[i : i + 3]
        c = components[cid]
        return np.array([c["x"], c["y"], math.radians(c["angle"])])

    def point(q: Any, p: list[float]) -> Any:
        c, s = math.cos(q[2]), math.sin(q[2])
        return q[:2] + np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])

    def residual(x: Any, driven: bool = True) -> Any:
        values = []
        for j in data["joints"]:
            a, b = pose(j["a"], x), pose(j["b"], x)
            delta = point(b, j["pb"]) - point(a, j["pa"])
            if j["kind"] == "slider":
                axis = a[2] + math.radians(j["axis"])
                values.append(float(np.dot(delta, [-math.sin(axis), math.cos(axis)])))
            else:
                values.extend(delta)
            if j["kind"] in {"fixed", "slider"}:
                values.append(ANGLE_SCALE * (b[2] - a[2] - math.radians(j["angle"])))
        for g in data["gears"]:
            values.append(
                ANGLE_SCALE
                * (pose(g["a"], x)[2] + g["ratio"] * pose(g["b"], x)[2] - math.radians(g["phase"]))
            )
        if driven:
            values.append(
                ANGLE_SCALE * (pose(data["driver"]["component"], x)[2] - math.radians(angle))
            )
        return np.array(values, dtype=float)

    result = least_squares(residual, x0, max_nfev=160, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    # Re-evaluate rather than trusting the optimizer success flag.
    errors = residual(result.x)
    maximum = float(np.max(np.abs(errors))) if len(errors) else 0.0
    constraints = []
    offset = 0
    rows = [(j["id"], 3 if j["kind"] == "fixed" else 2) for j in data["joints"]]
    rows += [(g["id"], 1) for g in data["gears"]] + [("driver", 1)]
    for cid, count in rows:
        error = float(np.max(np.abs(errors[offset : offset + count])))
        constraints.append({"id": cid, "residual_mm_equivalent": error})
        offset += count

    def rank(driven: bool) -> int:
        base = residual(result.x, driven)
        jac = np.empty((len(base), len(result.x)))
        for i in range(len(result.x)):
            probe = result.x.copy()
            probe[i] += 1e-6
            jac[:, i] = (residual(probe, driven) - base) / 1e-6
        return int(np.linalg.matrix_rank(jac, tol=1e-5)) if jac.size else 0

    dof = len(result.x) - rank(False)
    remaining = len(result.x) - rank(True)
    satisfied = maximum <= POSITION_TOL
    status = (
        "solved"
        if satisfied and remaining == 0
        else "singular_or_free"
        if satisfied
        else "unresolved"
    )
    poses = {
        cid: [float(q[0]), float(q[1]), float(math.degrees(q[2])), float(components[cid]["z"])]
        for cid in components
        for q in [pose(cid, result.x)]
    }
    return {
        "angle_deg": angle,
        "status": status,
        "residual_mm_equivalent": maximum,
        "undriven_local_dof": dof,
        "driven_local_dof": remaining,
        "redundant_equations": max(0, len(errors) - rank(True)),
        "poses": poses,
        "seed": result.x.tolist(),
        "evaluations": result.nfev,
        "constraints": constraints,
        "suspect_constraints": [
            c["id"] for c in constraints if c["residual_mm_equivalent"] > POSITION_TOL
        ],
    }


def study_mechanism(data: dict[str, Any]) -> dict[str, Any]:
    driver = data["driver"]
    current = solve_mechanism(data, driver["angle"])
    samples = []
    seed = None
    previous = None
    for angle in np.linspace(driver["start"], driver["end"], driver["samples"]):
        sample = solve_mechanism(data, float(angle), seed)
        if previous is not None:
            jump = max(
                abs(sample["poses"][c][2] - previous["poses"][c][2]) for c in sample["poses"]
            )
            if jump > 90:
                sample["status"] = "discontinuous"
        samples.append(sample)
        if sample["status"] != "solved":
            break
        seed, previous = sample["seed"], sample
    complete = len(samples) == driver["samples"] and all(s["status"] == "solved" for s in samples)
    for item in [current, *samples]:
        item.pop("seed", None)
    return {
        "status": "sampled" if complete else "incomplete",
        "current": current,
        "samples": samples,
        "requested_samples": driver["samples"],
        "range_deg": [driver["start"], driver["end"]],
        "closure_tolerance_mm_equivalent": POSITION_TOL,
        "angular_scale_mm_per_rad": ANGLE_SCALE,
        "method": "Planar rigid-body least-squares closure with independently recomputed residuals and local Jacobian rank; deterministic continuation.",
        "limitations": [
            "Local rank is not global mobility proof. Numerical failure does not prove impossibility.",
            "Sampled constraints do not certify continuous motion, tooth contact, clearance, strength or physical operation.",
        ],
    }
