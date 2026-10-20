"""Real engineering tools run only inside the bounded CAD worker."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import build123d as b
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.engineering_models import EngineeringInputs, JointInput
from forge_mind.workspace.geometry_results import boolean_volume
from forge_mind.workspace.hardware import bill_of_materials
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.topology import resolve


def datum(
    origin: tuple[float, float, float],
    axis: tuple[float, float, float],
    x_hint: tuple[float, float, float],
) -> Any:
    z = np.array(axis, dtype=float)
    if np.linalg.norm(z) < 1e-8:
        raise ValueError("Datum axis is zero")
    z /= np.linalg.norm(z)
    x = np.array(x_hint) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-8:
        raise ValueError("Datum X direction is parallel to its axis")
    x /= np.linalg.norm(x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    matrix[:3, 3] = origin
    return matrix


def assembly_transforms(count: int, joints: list[JointInput]) -> list[Any]:
    transforms = [np.eye(4) for _ in range(count)]
    incoming = {}
    for joint in joints:
        if max(joint.parent, joint.child) >= count or joint.child in incoming:
            raise ValueError("Joint references missing body or gives a child more than one parent")
        incoming[joint.child] = joint
    visiting: set[int] = set()
    done: set[int] = set()

    def place(body: int) -> Any:
        if body in done:
            return transforms[body]
        if body in visiting:
            raise ValueError(
                "Closed-loop assemblies require a loop-closure solver; this tree cannot be solved"
            )
        visiting.add(body)
        joint = incoming.get(body)
        if joint:
            motion = np.eye(4)
            if joint.kind == "revolute":
                motion[:3, :3] = Rotation.from_euler("z", joint.value, degrees=True).as_matrix()
            elif joint.kind == "slider":
                motion[2, 3] = joint.value
            transforms[body] = (
                place(joint.parent)
                @ datum(joint.parent_origin, joint.parent_axis, joint.parent_x)
                @ motion
                @ np.linalg.inv(datum(joint.child_origin, joint.child_axis, joint.child_x))
            )
        visiting.remove(body)
        done.add(body)
        return transforms[body]

    for body in range(count):
        place(body)
    return transforms


def transform_shape(shape: Any, matrix: Any) -> Any:
    # Rigid transform only, preserving analytic B-rep geometry.
    angles = Rotation.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=True)
    return b.Pos(*matrix[:3, 3]) * b.Rot(*angles) * shape


def overlap(a: Any, other: Any) -> float:
    from forge_mind.workspace.motion_check import bounds_overlap, shape_bounds

    if not bounds_overlap(shape_bounds(a), shape_bounds(other)):
        return 0.0
    common = a & other
    return boolean_volume(common)


def assembly_review(
    solids: list[Any], inputs: EngineeringInputs, revision_id: str
) -> tuple[dict[str, Any], list[Any]]:
    if inputs.assembly_revision and inputs.assembly_revision != revision_id:
        return {
            "status": "unverified",
            "reason": "Assembly body references belong to another revision; rebind explicitly",
        }, solids
    transforms = assembly_transforms(len(solids), inputs.joints)
    posed = [
        transform_shape(solid, matrix) for solid, matrix in zip(solids, transforms, strict=True)
    ]
    pairs: list[dict[str, Any]] = []
    from forge_mind.workspace.motion_check import bounds_overlap, shape_bounds

    boxes = [shape_bounds(s) for s in posed]
    separated = 0
    for i, a in enumerate(posed):
        for j in range(i + 1, len(posed)):
            nearby = bounds_overlap(boxes[i], boxes[j])
            volume = overlap(a, posed[j]) if nearby else 0.0
            distance = float(a.distance_to(posed[j])) if nearby else None
            # Expanded boxes already prove no intersection. Exact distance is
            # informational for these pose pairs, unlike service-path clearance.
            # Never label their conservative separation bound as an exact distance.
            gaps = np.maximum(boxes[i][0] - boxes[j][1], boxes[j][0] - boxes[i][1])
            lower_bound = float(np.linalg.norm(np.maximum(gaps - 1e-6, 0)))
            separated += not nearby
            pairs.append(
                {
                    "a": i,
                    "b": j,
                    "overlap_mm3": volume,
                    "distance_mm": distance,
                    "distance_lower_bound_mm": lower_bound if not nearby else None,
                    "distance_method": "Exact B-rep distance"
                    if nearby
                    else "Conservative expanded bounding-box separation; exact distance not evaluated",
                    "status": "fail" if volume > 1e-6 else "pass",
                }
            )
    paths: list[dict[str, Any]] = []
    for path in inputs.service_paths:
        if path.body >= len(posed):
            raise ValueError("Service path references missing body")
        witness = None
        for step in range(path.samples):
            fraction = step / (path.samples - 1)
            translation = [v * fraction for v in path.travel]
            moving = b.Pos(*translation) * posed[path.body]
            for i, stationary in enumerate(posed):
                if i == path.body:
                    continue
                volume = overlap(moving, stationary)
                clearance = float(moving.distance_to(stationary))
                if volume > 1e-6 or (path.tool_radius_mm > 0 and clearance < path.tool_radius_mm):
                    witness = {
                        "fraction": fraction,
                        "other_body": i,
                        "overlap_mm3": volume,
                        "distance_mm": clearance,
                    }
                    break
            if witness:
                break
        paths.append(
            {
                "id": path.id,
                "task": path.task,
                "status": "fail" if witness else "unverified",
                "witness": witness,
                "method": "Exact B-rep intersection at declared sampled translated poses; tool radius is an isotropic clearance envelope. No continuous-path clearance certificate.",
            }
        )
    return {
        "status": "fail"
        if any(p["status"] == "fail" for p in [*pairs, *paths])
        else "unverified"
        if paths
        else "pass",
        "pose_status": "fail" if any(p["status"] == "fail" for p in pairs) else "pass",
        "scope": "Declared pose only; zero overlap is not a mating, retention or motion certificate",
        "pairs": pairs,
        "pair_evaluation": {
            "total": len(pairs),
            "separated_by_bounds": separated,
            "exact_distance_queries": len(pairs) - separated,
            "scope": "All pose pairs checked for intersection; bounding-box separation is a lower bound, not an exact clearance measurement",
        },
        "transforms": [m.tolist() for m in transforms],
        "feature_transforms": [
            dict(
                zip(
                    ("x", "y", "z", "rx", "ry", "rz"),
                    [
                        *m[:3, 3].tolist(),
                        *Rotation.from_matrix(m[:3, :3]).as_euler("XYZ", degrees=True).tolist(),
                    ],
                    strict=True,
                )
            )
            for m in transforms
        ],
        "service_paths": paths,
        "degrees_of_freedom": sum(j.kind != "fixed" and j.lower != j.upper for j in inputs.joints),
        "grounded_roots": sorted(set(range(len(solids))) - {j.child for j in inputs.joints}),
    }, posed


def describe_overlap_witnesses(
    assembly: dict[str, Any], document: DesignDocument | None, facts: dict[str, Any] | None
) -> None:
    """Attach exported identities and declared context without waiving intersections."""
    occurrences = {o["body_index"]: o for o in (facts or {}).get("occurrences", [])}
    definitions = (
        {p.id: p for p in document.assembly.parts} if document and document.assembly else {}
    )
    bodies = {}
    for index, occurrence in occurrences.items():
        part = definitions.get(occurrence["definition"])
        bodies[str(index)] = {
            "path": occurrence["path"],
            "definition": occurrence["definition"],
            "name": part.name if part else occurrence.get("name"),
            "role": part.production.role if part else "unknown",
            "declared_context": part.source if part else None,
        }
    witnesses = []
    for pair in assembly.get("pairs", []):
        if pair["status"] != "fail":
            continue
        a, other = (bodies.get(str(pair[key]), {}) for key in ("a", "b"))
        witnesses.append(
            {
                "a": pair["a"],
                "b": pair["b"],
                "a_path": a.get("path"),
                "b_path": other.get("path"),
                "overlap_mm3": pair["overlap_mm3"],
                "context": "reference_involved"
                if "reference" in (a.get("role"), other.get("role"))
                else "modeled_body_intersection",
                "status": "fail",
            }
        )
    assembly["witness_bodies"] = {
        str(index): bodies[str(index)]
        for index in sorted({p[key] for p in witnesses for key in ("a", "b")})
        if str(index) in bodies
    }
    assembly["overlap_witnesses"] = witnesses
    assembly["interpretation_scope"] = (
        "All positive-volume intersections remain failures. Reference involvement is a declared "
        "role, not an exemption. Part descriptions are author assertions, not verified contact "
        "permissions. Thread envelopes, unloaded elastomers and reserved volumes require "
        "interface-specific evidence before physical interference or fit can be decided."
    )


def tolerance_review(inputs: EngineeringInputs) -> list[dict[str, Any]]:
    from forge_mind.validator.tolerance_stack import (
        Contributor,
        FitRequirement,
        cylindrical_fit_stack,
    )

    results = []
    for fit in inputs.fits:
        result = cylindrical_fit_stack(
            hole=Contributor(**fit.hole.model_dump()),
            shaft=Contributor(**fit.shaft.model_dump()),
            extras=[Contributor(**c.model_dump()) for c in fit.extras],
            requirement=FitRequirement(
                fit.min_clearance_mm, fit.max_clearance_mm, fit.fit_class, fit.source
            ),
        )
        results.append(
            {
                "id": fit.id,
                **asdict(result),
                "scope": "Worst-case stack of user-declared dimensions and allowances; not a measurement of current CAD or process calibration",
            }
        )
    return results


def manufacturing_review(mesh: Any, shape: Any, inputs: EngineeringInputs) -> list[dict[str, Any]]:
    process = inputs.process
    if process is None:
        return [
            {
                "id": "process",
                "status": "unverified",
                "reason": "Choose a manufacturing process and explicit limits",
            }
        ]
    extents = list(shape.bounding_box().size)
    checks: list[dict[str, Any]] = [
        {
            "id": "build_volume",
            "status": "pass"
            if all(a <= limit for a, limit in zip(extents, process.build_volume_mm, strict=True))
            else "fail",
            "measured_mm": extents,
            "limits_mm": process.build_volume_mm,
            "method": "World-axis dimensions at current CAD Z-up orientation",
            "source": process.source,
        }
    ]
    if process.min_wall_mm:
        from forge_mind.validator.feature_checks import measure_wall_fraction

        wall = measure_wall_fraction(mesh, process.min_wall_mm)
        minimum = wall.thinnest_mm if wall else None
        checks.append(
            {
                "id": "wall",
                "status": "fail"
                if minimum is not None and minimum < process.min_wall_mm
                else "unverified",
                "sampled_min_mm": minimum,
                "limit_mm": process.min_wall_mm,
                "method": "Sampled rays; thin-wall witness only, no full-surface certificate",
                "source": process.source,
            }
        )
    if process.max_unsupported_angle_deg is not None:
        down = mesh.face_normals[:, 2] < -1e-8
        off_bed = mesh.triangles_center[:, 2] > mesh.bounds[0, 2] + 0.05
        indices = np.flatnonzero(down & off_bed)
        angles = np.degrees(np.arcsin(np.clip(-mesh.face_normals[indices, 2], 0, 1)))
        maximum = float(np.max(angles)) if len(angles) else 0
        checks.append(
            {
                "id": "overhang",
                "status": "unverified",
                "max_downward_slope_deg": maximum,
                "limit_deg": process.max_unsupported_angle_deg,
                "exceeds_limit": maximum > process.max_unsupported_angle_deg,
                "method": "Tessellated downward slope above build plane; does not determine support beneath faces or bridgeability",
                "source": process.source,
            }
        )
    if process.process == "cnc":
        checks.append(
            {
                "id": "tool_access",
                "status": "unverified",
                "tool_diameter_mm": process.tool_diameter_mm,
                "reason": "No cutter, fixture and machine kinematic access proof; review cavity reach and setup manually",
            }
        )
    return checks


def simulation_review(mesh: Any, shape: Any, inputs: EngineeringInputs) -> list[dict[str, Any]]:
    results = []
    material = inputs.material
    for load in inputs.loads:
        result: dict[str, Any] = {"id": load.id, "status": "unverified", "setup": load.model_dump()}
        if material is None or len(shape.solids()) != 1:
            result["reason"] = (
                "Simulation requires explicit material and one connected solid; contacts are unsupported"
            )
            results.append(result)
            continue
        if load.method == "rectangular_cantilever":
            dims = list(shape.bounding_box().size)
            width, depth, length = dims
            box_volume = width * depth * length
            if (
                abs(shape.volume - box_volume) > max(1e-6, box_volume * 1e-8)
                or abs(load.direction[1]) > 1e-8
                or abs(load.direction[2]) > 1e-8
                or length < 5 * max(width, depth)
            ):
                result["reason"] = (
                    "Analytic tool requires a solid axis-aligned rectangular beam, Z length at least five times width/depth, and transverse X load"
                )
            else:
                inertia = depth * width**3 / 12
                deflection = load.force_n * length**3 / (3 * material.elastic_modulus_mpa * inertia)
                stress = load.force_n * length * width / (2 * inertia)
                result.update(
                    max_deflection_mm=deflection,
                    max_von_mises_mpa=stress,
                    convergence={
                        "method": "Closed-form Euler-Bernoulli cantilever; mesh convergence not applicable"
                    },
                    method="F L^3 / (3 E I), sigma=F L c/I; I=depth*width^3/12. Fully fixed Z base, end transverse force, linear isotropic slender beam.",
                )
        else:
            from forge_mind.physics.fea import estimate_load_response

            runs = [
                estimate_load_response(
                    mesh,
                    load.force_n,
                    load_dir=load.direction,
                    e_mpa=material.elastic_modulus_mpa,
                    nu=material.poisson_ratio,
                    edge_length_fac=factor,
                    planar_faces=True,
                )
                for factor in load.mesh_factors
            ]
            if any(run is None for run in runs):
                result["reason"] = (
                    "Validated FEA solver or mesh failed/timed out; no substituted result"
                )
            else:
                coarse, fine = runs
                assert coarse is not None and fine is not None
                delta = abs(fine.max_deflection_mm - coarse.max_deflection_mm) / max(
                    abs(fine.max_deflection_mm), 1e-12
                )
                stress_delta = abs(fine.max_von_mises_mpa - coarse.max_von_mises_mpa) / max(
                    abs(fine.max_von_mises_mpa), 1e-12
                )
                result.update(
                    asdict(fine),
                    convergence={
                        "deflection_relative_change": delta,
                        "peak_stress_relative_change": stress_delta,
                        "tolerance": load.convergence_tolerance,
                    },
                    method="Validated quadratic linear-elastic tetrahedral solver, full lowest-Z face fixed and full highest-Z face traction, coarse/fine mesh comparison",
                )
                if max(delta, stress_delta) > load.convergence_tolerance:
                    result["reason"] = (
                        "Mesh refinement has not converged within the declared tolerance"
                    )
        if "max_deflection_mm" in result:
            stress = result["max_von_mises_mpa"]
            safety = (
                material.yield_strength_mpa / stress
                if material.yield_strength_mpa and stress > 0
                else None
            )
            result["safety_factor_under_declared_material"] = safety
            exceeds = result["max_deflection_mm"] > load.max_deflection_mm or (
                safety is not None and safety < load.min_safety_factor
            )
            result["status"] = "fail" if exceeds else "unverified"
            result["within_declared_limits"] = (
                not exceeds and "reason" not in result and safety is not None
            )
            result["limitations"] = (
                "Computed response to declared idealized setup, not product safety certification. No nonlinear contact, fatigue, buckling, plasticity or print anisotropy model. Convergence comparison is not an error bound."
            )
        results.append(result)
    return results


def run_engineering(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    source = Path(payload["source"])
    inputs = EngineeringInputs.model_validate(payload["inputs"])
    revision = payload["revision"]
    shape = b.import_step(str(source / "part.step"))
    if len(shape.solids()) > (
        128 if (revision.get("document") or {}).get("assembly") else 64
    ) or len(shape.faces()) > (
        4500
        if (revision.get("document") or {}).get("mechanism")
        or (revision.get("document") or {}).get("assembly")
        else 1200
    ):
        raise ValueError("Report geometry exceeds topology limits")
    mesh = trimesh.load(str(source / "part.stl"), force="mesh")
    topology = json.loads((source / "topology.json").read_text(encoding="utf-8"))
    document = (
        DesignDocument.model_validate(revision["document"]) if revision.get("document") else None
    )
    references = [
        {"id": a.id, **resolve(a.reference.model_dump(), topology)} for a in inputs.annotations
    ]
    for item in references:
        if item.get("entity"):
            item["entity"] = {k: v for k, v in item["entity"].items() if k != "polyline"}
    assembly, posed = assembly_review(list(shape.solids()), inputs, revision["id"])
    describe_overlap_witnesses(assembly, document, revision["measurement"]["facts"].get("assembly"))
    b.export_step(b.Compound(children=posed), str(directory / "assembly.step"))
    checks = (
        [
            {
                "id": "per_part_process",
                "status": "unverified",
                "reason": "Assembly manufacturing requires per-make-part process limits and setup validation; a global process does not apply to purchased or reference geometry.",
            }
        ]
        if document and document.assembly
        else manufacturing_review(mesh, shape, inputs)
    )
    from forge_mind.workspace.mechanism_review import mechanism_review

    mechanism = mechanism_review(
        shape, document, revision["measurement"]["facts"].get("mechanism"), inputs
    )
    from forge_mind.workspace.production import production_report

    assembly_facts = revision["measurement"]["facts"].get("assembly")
    production_facts = assembly_facts or {
        "occurrences": [{"path": "part", "volume_mm3": sum(s.volume for s in shape.solids())}]
    }
    production = production_report(document, production_facts)
    report = {
        "production": production,
        "assembly_definitions": revision["measurement"]["facts"].get("assembly"),
        "revision_id": revision["id"],
        "geometry_hash": file_sha256(source / "part.step"),
        "input_hash": inputs.fingerprint(),
        "engine_fingerprint": engine_fingerprint(),
        "frame": "cad_z_up_mm",
        "inputs": inputs.model_dump(),
        "geometry": revision["measurement"],
        "references": references,
        "assembly": assembly,
        "tolerances": tolerance_review(inputs),
        "manufacturing": checks,
        "simulation": simulation_review(mesh, shape, inputs),
        "mechanism": mechanism,
        "bom": bill_of_materials(document, revision.get("assets", [])),
        "mass_kg": production["mass_kg"],
        "physical_feedback": [
            {
                **f.model_dump(),
                "proposed_parameter_correction_mm": (f.nominal_mm - f.measured_mm)
                * f.parameter_gain,
                "automatic_application": False,
            }
            for f in inputs.feedback
        ],
        "open_items": [
            "Full functional intent, manufacturing success and physical performance require independent acceptance."
        ]
        + [a.text for a in inputs.assumptions if not a.confirmed],
    }
    if document and document.assembly:
        assembly_facts = revision["measurement"]["facts"]["assembly"]
        if document.mechanism:
            from forge_mind.workspace.assembly_review import assembly_motion_review

            report["assembly_motion"] = assembly_motion_review(
                shape, revision["measurement"]["facts"]["mechanism"], assembly_facts
            )
    from forge_mind.workspace.drawings import write_drawings

    report["inspection_plan"] = payload.get("inspection_plan")
    report["drawings"] = write_drawings(
        directory, topology, revision, inputs, inspection_plan=payload.get("inspection_plan")
    )
    from forge_mind.workspace.measure import section_revision

    report["section"] = section_revision(
        source, "z", float(shape.bounding_box().center().Z), directory / "section.png"
    )
    report["reviews"] = [
        {
            "category": "geometry",
            "status": "pass" if revision["measurement"]["geometry_valid"] else "fail",
            "method": "Independent exported STEP/STL checks",
        },
        {
            "category": "requirements",
            "status": "pass" if revision["measurement"]["requirements_satisfied"] else "unverified",
            "method": "Frozen contract coverage; see individual findings",
        },
        {
            "category": "references",
            "status": "unverified"
            if any(r["status"] != "resolved" for r in references)
            else "pass",
            "method": "Unique complete supported geometry correspondence; annotation anchors only",
        },
        {
            "category": "manufacturing",
            "status": "fail" if any(c["status"] == "fail" for c in checks) else "unverified",
            "method": "Process-specific checks with explicit coverage limits",
        },
        {
            "category": "assembly",
            "status": assembly["status"],
            "method": assembly.get("scope", assembly.get("reason", "Declared pose")),
        },
    ]
    if document and document.mechanism:
        report["reviews"].append(
            {
                "category": "mechanism",
                "status": mechanism["status"],
                "method": "Sampled closure, exported gear intersections and declared shaft loads; continuous clearance and strength remain unverified",
            }
        )
    if document and document.assembly:
        report["reviews"].append(
            {
                "category": "assembly_interfaces",
                "status": report["assembly_definitions"]["status"],
                "method": "Measured exported occurrences and cylindrical interfaces",
            }
        )
        if "assembly_motion" in report:
            report["reviews"].append(
                {
                    "category": "assembly_motion",
                    "status": report["assembly_motion"]["status"],
                    "method": report["assembly_motion"]["scope"],
                }
            )
    return report
