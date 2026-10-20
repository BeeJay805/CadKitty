"""Actual exported gear interference and independently declared shaft-load calculations."""

from __future__ import annotations

import math
from typing import Any

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.geometry_results import boolean_volume
from forge_mind.workspace.mechanism import pose_delta_matrix
from forge_mind.workspace.mechanism_geometry import matrix_location


def contact_poses(motion: dict[str, Any]) -> list[dict[str, Any]]:
    """Nonuniform exact input angles avoid repeating common gear-tooth phases."""
    from forge_mind.workspace.mechanism import solve_mechanism

    start, end = motion["range_deg"]
    golden = (math.sqrt(5) - 1) / 2
    fractions = sorted([0.0, 1.0, *((k * golden) % 1 for k in range(1, 6))])
    solved = [s for s in [motion["current"], *motion["samples"]] if s["status"] == "solved"]
    if not solved:
        return []
    result = []
    for fraction in fractions:
        angle = start + (end - start) * fraction
        nearest = min(solved, key=lambda s: abs(s["angle_deg"] - angle))
        seed = []
        for component in motion["graph"]["components"]:
            if not component["grounded"]:
                x, y, rotation, _ = nearest["poses"][component["id"]]
                seed.extend([x, y, math.radians(rotation)])
        result.append(solve_mechanism(motion["graph"], angle, seed))
    return result


def mechanism_review(
    shape: Any,
    document: DesignDocument | None,
    motion: dict[str, Any] | None,
    inputs: EngineeringInputs,
) -> dict[str, Any]:
    if not document or not document.mechanism or not motion:
        return {"status": "unverified", "reason": "No geometry-bound mechanism document"}
    solids = list(shape.solids())
    bindings = {c["component"]: c for c in motion["bindings"]}
    features = {f.id: f for f in document.features}
    rest = motion["current"]["poses"]
    contact = []
    samples = contact_poses(motion) if motion["graph"]["gears"] else []

    def posed(cid: str, sample: dict[str, Any]) -> Any:
        delta = pose_delta_matrix(
            sample["poses"][cid], rest[cid], motion["graph"].get("frame", "planar_xy_mm_deg")
        )
        return matrix_location(delta) * solids[bindings[cid]["body_index"]]

    for pair in motion["graph"]["gears"]:
        if document.assembly:
            contact.append(
                {
                    "id": pair["id"],
                    "status": "unverified",
                    "reason": "Reusable assembly gear checking requires a definition-level tooth binding; all-body interference is reviewed separately",
                }
            )
            continue
        measurements = []
        failure = None if samples else "No solved poses available for contact checking"
        for sample in samples:
            if sample["status"] != "solved":
                failure = "A requested pose was not solved"
                break
            try:
                common = posed(pair["a"], sample) & posed(pair["b"], sample)
                volume = boolean_volume(common)
                measurements.append(
                    {
                        "angle_deg": sample["angle_deg"],
                        "closure_residual_mm_equivalent": sample["residual_mm_equivalent"],
                        "overlap_mm3": volume,
                    }
                )
            except Exception as exc:
                failure = f"Kernel intersection failed: {type(exc).__name__}"
                break
        collision = any(m["overlap_mm3"] > 1e-5 for m in measurements)
        contact.append(
            {
                "id": pair["id"],
                "status": "fail" if collision else "unverified",
                "samples": measurements,
                "requested_samples": 7,
                "coverage": "incomplete" if failure else "discrete_only",
                "sampling": "Range endpoints and five golden-ratio interior fractions, solved at exact angles",
                "reason": failure
                or "Seven nonuniform discrete tooth-overlap checks; no continuous clearance or branch-continuity guarantee",
                "method": "B-rep Boolean intersection of the exported paired gears under solved rigid poses",
                "numerical_volume_floor_mm3": 1e-5,
            }
        )
    loads = []
    for load in inputs.mechanism_loads:
        result: dict[str, Any] = {
            "id": load.id,
            "component": load.component,
            "status": "unverified",
            "setup": load.model_dump(),
        }
        binding = bindings.get(load.component)
        f = features.get(binding["feature"]) if binding else None
        if f is None or f.kind != "shaft":
            result["reason"] = "Nominal torsion requires an unchanged native shaft output"
        else:
            p = {**f.params, **{k: document.values()[v] for k, v in f.bindings.items()}}
            outer, inner = p["diameter"], p.get("bore", 0)
            stress = 16 * load.torque_n_mm * outer / (math.pi * (outer**4 - inner**4))
            margin = load.allowable_shear_mpa / (stress * load.safety_factor)
            result.update(
                nominal_shear_mpa=stress,
                declared_limit_mpa=load.allowable_shear_mpa / load.safety_factor,
                nominal_margin=margin,
                method="tau=16*T*D/(pi*(D^4-d^4)); static circular shaft torsion under owner-declared torque and allowable shear",
                status="fail" if margin < 1 else "unverified",
                reason="Nominal screening only. Keyway stress concentration, bending, fatigue, bearing life and material provenance remain unresolved.",
            )
        loads.append(result)
    return {
        "status": "fail"
        if any(c["status"] == "fail" for c in contact + loads + motion["gear_layout"])
        else "unverified",
        "kinematics": {
            k: motion[k]
            for k in ("status", "requested_samples", "range_deg", "method", "limitations")
        },
        "gear_layout": motion["gear_layout"],
        "tooth_interference": contact,
        "shaft_loads": loads,
        "assembly_notes": [
            "Bearing features are dimensional envelopes, not sourced or rated rolling bearings.",
            "Keys model positive drive geometry; key capacity, retention and fastener sizing still require engineering review.",
            "Planar loops describe rigid constraints; friction, compliance, dynamics, heat and combustion were not simulated.",
        ],
    }
