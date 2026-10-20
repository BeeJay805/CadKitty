"""Measure exported artifacts, never model self-reports or author-picked limits."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.validation_provenance import file_sha256
from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders
from forge_mind.workspace.engine_identity import engine_fingerprint as engine_fingerprint
from forge_mind.workspace.models import Contract, ContractEdit, Finding, Measurement, Requirement
from forge_mind.workspace.source_numbers import count_word_values
from forge_mind.workspace.tessellation import (
    export_verified_tessellation,
    read_tessellation_settings,
)

ARTIFACTS = ("program.py", "part.step", "part.stl", "part.glb", "render.png", "tessellation.json")


def ground_contract(
    contract: Contract,
    prompt: str,
    *,
    allow_explicit: bool = False,
    attribution: Any = None,
) -> Contract:
    """Unsupported or ungrounded interpretations remain visible, never earn a pass."""
    fixed = []
    interpreted_ids: frozenset[str] = frozenset()
    if attribution is not None and attribution.customer_nominal_interpretation is not None:
        from forge_mind.workspace.customer_interpretation import validate_customer_interpretation

        interpreted_ids = validate_customer_interpretation(contract, prompt, attribution)
    verification_errors: dict[str, str] = {}
    if any(req.verification is not None for req in contract.requirements):
        from forge_mind.workspace.verification_admission import verification_diagnostics

        # Facts may legitimately link sibling duties. Validate their references
        # against the complete contract, then retain row-specific eligibility.
        requirement_ids = {req.id for req in contract.requirements}
        for diagnostic in verification_diagnostics(
            contract, prompt, attribution, allow_explicit=allow_explicit
        ):
            identifier = diagnostic["requirement_id"]
            if identifier is None or identifier not in requirement_ids:
                # A cross-row conflict cannot be assigned to an arbitrary row
                # or disappear when a row is independently grounded.
                raise ValueError(diagnostic["message"])
            verification_errors.setdefault(identifier, diagnostic["message"])
    for req in contract.requirements:
        if req.id in interpreted_ids:
            fixed.append(req)
            continue
        if req.verification is not None:
            if req.id in verification_errors:
                # Unsupported interpretation retains the entire critical claim.
                # Removing typed eligibility cannot create an acceptance path.
                req = req.model_copy(
                    update={
                        "verification": None,
                        "metric": "unmeasured",
                        "critical": True,
                        "text": f"Typed verification unavailable: {verification_errors[req.id]}. {req.text}",
                    }
                )
            if not (req.source == "explicit" and allow_explicit):
                req = req.model_copy(
                    update={"critical": True, "source": "request"}
                    if req.source == "explicit"
                    else {"critical": True}
                )
            fixed.append(req)
            continue
        if req.source == "explicit" and allow_explicit:
            fixed.append(req)
            continue
        if req.source == "explicit":
            req = req.model_copy(update={"source": "request"})
        req = req.model_copy(update={"critical": True})
        reason = ""
        if req.source == "assumption":
            reason = "Assumption requires explicit confirmation. "
        elif req.source_quote not in prompt:
            reason = "Source quote does not occur verbatim in the request. "
        elif req.value is not None:
            from forge_mind.workspace.source_envelope import ground_source_envelope

            envelope = ground_source_envelope(req, prompt, attribution)
            if envelope.state != "not_applicable":
                req = req.model_copy(update={"position_tolerance": 0.001})
                fixed.append(
                    req.model_copy(
                        update={
                            "metric": "unmeasured",
                            "text": f"Envelope source unavailable ({envelope.code}). {req.text}",
                        }
                    )
                    if envelope.state == "unsupported"
                    else req
                )
                continue
            assert req.value is not None
            values = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", req.source_quote)]
            if req.metric in {"solid_count", "hole_count"}:
                values.extend(count_word_values(req.source_quote))
            for value in re.findall(r"(?:±|\+\s*/\s*-)\s*(\d+(?:\.\d+)?)", req.source_quote):
                values.extend([float(value), -float(value)])
            factors = [1.0]
            if req.metric not in {"solid_count", "hole_count"}:
                if re.search(r"\b(?:inch|inches|in)\b|\"", req.source_quote, re.I):
                    factors.append(25.4)
                if re.search(r"\bcm\b", req.source_quote, re.I):
                    factors.append(10.0)
            if not any(
                math.isclose(req.value, v * f, abs_tol=1e-6) for v in values for f in factors
            ):
                reason = "Numeric target cannot be traced to the quoted request. "
            tolerance_values = []
            for nominal, allowance in re.findall(
                r"(\d+(?:\.\d+)?)\s*(?:mm|cm|inches|inch|in)?\s*(?:±|\+\s*/\s*-)\s*(\d+(?:\.\d+)?)",
                req.source_quote,
                re.I,
            ):
                if any(math.isclose(req.value, float(nominal) * f, abs_tol=1e-6) for f in factors):
                    tolerance_values.append(float(allowance))
            # Worded allowances are accepted only with an unambiguous nominal target.
            explicit_allowances = [
                float(x)
                for x in re.findall(
                    r"\b(?:tolerance\s*(?:of|is|:|=)?|within)\s*(\d+(?:\.\d+)?)",
                    req.source_quote,
                    re.I,
                )
            ]
            if len(set(values) - set(explicit_allowances)) == 1:
                tolerance_values.extend(explicit_allowances)
            if req.tolerance and not any(
                math.isclose(req.tolerance, v * f, abs_tol=1e-6)
                for v in tolerance_values
                for f in factors
            ):
                reason = "Tolerance cannot be traced to the quoted request. "
            if req.center is not None and not all(
                any(math.isclose(c, v * f, abs_tol=1e-6) for v in values for f in factors)
                for c in req.center
            ):
                reason = "Feature location cannot be traced to the quoted request. "
            bound_words = {
                "ge": r"\b(?:minimum|at least|no less than)\b|>=|≥",
                "le": r"\b(?:maximum|at most|no more than)\b|<=|≤|\bwithin\s+\d+(?:\.\d+)?\s*(?:mm|cm|inches|inch|in)?\s*[.!;]?$",
            }
            if req.comparator != "eq" and not re.search(
                bound_words[req.comparator], req.source_quote, re.I
            ):
                reason = "Bound direction cannot be traced to the quoted request. "
            # Association is an instrument setting, never an author-selected acceptance allowance.
            req = req.model_copy(update={"position_tolerance": 0.001})
        if req.part_id and req.part_id.lower() not in prompt.lower():
            reason = "Part selector cannot be traced to the request. "
        fixed.append(
            req.model_copy(update={"metric": "unmeasured", "text": reason + req.text})
            if reason
            else req
        )
    return contract.model_copy(update={"requirements": fixed})


def merge_contract_edit(previous: Contract, proposed: Contract, instruction: str) -> Contract:
    """An edit changes only requirements grounded in the new instruction.

    Omissions and rephrasings cannot erase previously explicit user targets. Numeric
    targets edited in the UI retain their authority even if the original quote is old.
    """
    by_id = {r.id: r for r in previous.requirements}
    if isinstance(proposed, ContractEdit):
        for removal in proposed.removals:
            quote = removal.source_quote
            if (
                quote in instruction
                and re.search(r"\b(?:remove|delete|omit|without|no longer)\b", quote, re.I)
                and not re.search(r"\b(?:do not|don't|never)\b", quote, re.I)
            ):
                by_id.pop(removal.requirement_id, None)
    for raw in proposed.requirements:
        if raw.source_quote not in instruction:
            continue
        grounded = ground_contract(Contract(requirements=[raw]), instruction).requirements[0]
        if raw.verification is not None and grounded.verification is None:
            # An unsupported proposal cannot replace a frozen typed target.
            continue
        if raw.metric != "unmeasured" and grounded.metric == "unmeasured":
            # Keep the prior target and surface the unsupported proposed change separately.
            continue
        if (
            grounded.id not in by_id
            and grounded.metric != "unmeasured"
            and re.search(
                r"\b(?:change|make|set|increase|decrease|resize|replace|adjust|update|reduce|enlarge|shrink)\b",
                instruction,
                re.I,
            )
        ):
            same_instrument = [
                r
                for r in by_id.values()
                if (r.metric, r.axis, r.center, r.comparator, r.part_id)
                == (
                    grounded.metric,
                    grounded.axis,
                    grounded.center,
                    grounded.comparator,
                    grounded.part_id,
                )
            ]
            if len(same_instrument) == 1:
                grounded = grounded.model_copy(update={"id": same_instrument[0].id})
        by_id[grounded.id] = grounded
    if list(by_id.values()) == previous.requirements:
        return previous
    notes = list(dict.fromkeys([*previous.coverage_notes, *proposed.coverage_notes]))[:100]
    return Contract(requirements=list(by_id.values()), coverage_notes=notes)


def _finding(
    req: Requirement,
    value: float | None,
    method: str,
    reason: str,
    locator: list[float] | None = None,
) -> Finding:
    if value is None or req.value is None or not math.isfinite(value):
        return Finding(
            requirement_id=req.id,
            status="unverified",
            expected=req.value,
            method=method,
            reason=reason,
            locator=locator,
        )
    slack = 1e-6  # numerical comparison epsilon, not a manufacturing allowance
    limit = req.value
    passed = {
        "eq": abs(value - limit) <= req.tolerance + slack,
        "ge": value >= limit - req.tolerance - slack,
        "le": value <= limit + req.tolerance + slack,
    }[req.comparator]
    return Finding(
        requirement_id=req.id,
        status="pass" if passed else "fail",
        measured=value,
        expected=limit,
        method=method,
        reason=reason,
        locator=locator,
    )


def _check(req: Requirement, facts: dict[str, Any], mesh: Any) -> Finding:
    if req.metric.startswith("extent_"):
        return _finding(
            req,
            facts["extents_mm"]["xyz".index(req.metric[-1])],
            "STEP bounding box",
            "World-axis extent of final exported B-rep; not an internal feature dimension.",
        )
    if req.metric in {"volume", "solid_count"}:
        return _finding(
            req,
            facts["volume_mm3" if req.metric == "volume" else "solid_count"],
            "STEP topology",
            "Measured on reimported final STEP.",
        )
    if req.metric.startswith("hole_"):
        axis = "xyz".index(req.axis)
        others = [i for i in range(3) if i != axis]
        holes = [h for h in facts["holes"] if abs(h["axis"][axis]) > 0.99999]
        ambiguous = [
            h
            for h in facts.get("ambiguous_cylindrical_walls", [])
            if abs(h["axis"][axis]) > 0.99999
        ]
        if req.center is not None:
            holes = [
                h
                for h in holes
                if math.dist([h["point"][i] for i in others], req.center) <= req.position_tolerance
            ]
            ambiguous = [
                h
                for h in ambiguous
                if math.dist([h["point"][i] for i in others], req.center) <= req.position_tolerance
            ]
        if ambiguous:
            return _finding(
                req,
                None,
                "B-rep cylindrical wall coverage",
                "Partial, split or unclassified cylindrical walls make this bore association incomplete; a rounded slot end is not a complete bore.",
            )
        if req.metric == "hole_count":
            return _finding(
                req,
                float(len(holes)),
                "B-rep concave coaxial cylindrical walls",
                "Counts cylindrical bores on this axis; does not prove throughness or noncircular cutouts.",
            )
        if len(holes) != 1:
            return _finding(
                req,
                None,
                "B-rep feature association",
                f"Expected one uniquely located bore; found {len(holes)}. Specify its center.",
            )
        hole = holes[0]
        if req.metric == "hole_depth" and hole.get("segments", 1) != 1:
            return _finding(
                req,
                None,
                "B-rep axial continuity",
                "Separated coaxial wall spans cannot establish one bore depth.",
            )
        key = "diameter_mm" if req.metric == "hole_diameter" else "depth_mm"
        return _finding(
            req,
            hole[key],
            "B-rep concave cylindrical wall",
            "Axial wall span is not proof of a through-hole.",
            hole["point"],
        )
    if req.metric == "min_wall":
        from forge_mind.validator.feature_checks import measure_wall_fraction

        wall = measure_wall_fraction(mesh, req.value or 0.0)
        if (
            wall is not None
            and wall.thinnest_mm is not None
            and req.comparator == "ge"
            and wall.thinnest_mm < (req.value or 0) - req.tolerance
        ):
            return _finding(
                req,
                wall.thinnest_mm,
                "sampled wall rays",
                "Observed thin-wall witness; sampling cannot certify all surfaces.",
            )
        return _finding(
            req,
            None,
            "sampled wall rays",
            "No certified whole-surface coverage bound; positive samples cannot prove a minimum wall.",
        )
    return _finding(
        req,
        None,
        "no applicable instrument",
        "Requirement retained, but no reliable measurement is implemented for this claim.",
    )


def _mesh_error(reason: str) -> str:
    if len(reason) <= 2000:
        return reason
    suffix = (
        f" [message truncated from {len(reason)} characters; full detail in definition evidence]"
    )
    return reason[: 2000 - len(suffix)] + suffix


def definition_measurements(
    directory: Path,
    rid: str,
    part_id: str,
    requirements: list[Requirement],
    *,
    retain_stl: bool = False,
) -> tuple[dict[str, Any], list[Finding]]:
    """Reimport exactly the named manufactured-part export in its local frame."""
    import tempfile

    from forge_mind.workspace.processing import copy_artifact, regular_file

    source = directory / f"definition_{part_id}.step"
    if not regular_file(source):
        unavailable: dict[str, Any] = {"frame": "definition_local", "status": "unverified"}
        if retain_stl:
            unavailable["mesh_delivery"] = {
                "status": "failed",
                "step_file": source.name,
                "step_sha256": None,
                "error": f"Named definition {part_id} STEP is unavailable or nonregular",
            }
        return unavailable, [
            _finding(
                r,
                None,
                "Named definition unavailable",
                f"No exported definition {part_id}; no whole-model fallback",
            )
            for r in requirements
        ]
    delivery: dict[str, Any] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="ck-def-") as temp:
            work = Path(temp)
            copy_artifact(source, work / "part.step")
            _linear, _angular, collapsed = export_verified_tessellation(work)
            measured = measure_revision(
                work,
                rid,
                Contract(
                    requirements=[r.model_copy(update={"part_id": None}) for r in requirements]
                ),
            )
            if retain_stl:
                target = directory / f"definition_{part_id}.stl"
                consistency = measured.facts.get("export_consistency", {})
                if not isinstance(consistency, dict):
                    consistency = {}
                reference = consistency.get("reference_tessellation")
                tessellation = measured.facts.get("tessellation")
                delivery = {
                    "status": "failed",
                    "step_file": source.name,
                    "step_sha256": file_sha256(source),
                    "error": f"Definition {part_id} mesh did not pass its measured geometry/export checks",
                }
                if (
                    measured.geometry_valid
                    and isinstance(tessellation, dict)
                    and consistency.get("volume_within_relative_0_001") is True
                    and (
                        consistency.get("bounds_within_0_01_mm") is True
                        or (
                            isinstance(reference, dict)
                            and reference.get("triangle_inventory_matches") is True
                        )
                    )
                ):
                    target_existed = target.exists()
                    try:
                        copy_artifact(work / "part.stl", target)
                    except TimeoutError:
                        raise
                    except (ValueError, OSError, RuntimeError) as exc:
                        # A requested export can fail without changing successful
                        # native dimensions or ordinary requirement findings.
                        if not target_existed and target.exists() and regular_file(target):
                            target.unlink()
                        delivery["error"] = f"Definition {part_id} STL retention failed: {exc}"
                    else:
                        delivery = {
                            "status": "available",
                            "step_file": source.name,
                            "step_sha256": file_sha256(source),
                            "stl_file": target.name,
                            "stl_sha256": file_sha256(target),
                            "stl_bytes": target.stat().st_size,
                            "body_count": measured.facts["solid_count"],
                            "bounds_mm": [measured.facts["bbox_min"], measured.facts["bbox_max"]],
                            "tessellation": {
                                **tessellation,
                                "collapsed_triangles_removed": collapsed,
                            },
                            "export_consistency": consistency,
                        }
    except TimeoutError:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        if not retain_stl:
            raise
        reason = f"Definition {part_id} mesh measurement failed: {exc}"
        return {
            "frame": "definition_local",
            "status": "unverified",
            "step_sha256": file_sha256(source),
            "mesh_delivery": {
                "status": "failed",
                "step_file": source.name,
                "step_sha256": file_sha256(source),
                "error": _mesh_error(reason),
            },
            "mesh_error_detail": reason,
        }, [_finding(r, None, "Definition mesh unavailable", reason) for r in requirements]
    evidence = {
        "frame": "definition_local",
        "step_sha256": file_sha256(source),
        "geometry_valid": measured.geometry_valid,
        "facts": measured.facts,
    }
    if delivery is not None:
        if delivery["status"] == "failed":
            evidence["mesh_error_detail"] = delivery["error"]
            delivery["error"] = _mesh_error(delivery["error"])
        evidence["mesh_delivery"] = delivery
    findings = [
        f.model_copy(
            update={
                "method": f"Definition {part_id}, local frame: {f.method}",
                "reason": f.reason
                + " Local part evidence does not establish assembled placement or function.",
                **(
                    {"status": "unverified", "measured": None}
                    if not measured.geometry_valid
                    else {}
                ),
            }
        )
        for f in measured.findings
    ]
    return evidence, findings


def measure_revision(
    directory: Path,
    rid: str,
    contract: Contract,
    *,
    structured_assembly: bool = False,
    definition_mesh_ids: tuple[str, ...] = (),
) -> Measurement:
    hashes = {
        n: file_sha256(directory / n)
        for n in ARTIFACTS
        if n != "tessellation.json" and (directory / n).is_file()
    }
    result = Measurement(
        revision_id=rid,
        contract_hash=contract.fingerprint(),
        artifact_hashes=hashes,
        engine_fingerprint=engine_fingerprint(),
        geometry_valid=False,
        caveats=[
            "Dimensional evidence is conditional on the displayed contract and coordinate frame.",
            "No load capacity, process success, or physical fit is certified without a dedicated check.",
        ],
    )
    try:
        from build123d import import_step

        settings = read_tessellation_settings(directory)
        if (directory / "tessellation.json").exists():
            result.artifact_hashes["tessellation.json"] = file_sha256(
                directory / "tessellation.json"
            )
        part = import_step(str(directory / "part.step"))
        mesh: Any = trimesh.load(str(directory / "part.stl"), force="mesh")
        bbox = part.bounding_box()
        solid_count = len(part.solids())
        # Compound.volume in build123d omits deeper subassemblies. Measure every
        # exported physical solid, matching the tessellated assembly inventory.
        volume = sum(float(solid.volume) for solid in part.solids())
        is_valid = bool(part.is_valid)
        mesh_volume = abs(float(mesh.volume))
        step_bounds = np.array(
            [[bbox.min.X, bbox.min.Y, bbox.min.Z], [bbox.max.X, bbox.max.Y, bbox.max.Z]]
        )
        volume_agrees = math.isclose(mesh_volume, volume, rel_tol=0.001, abs_tol=0.001)
        bounds_agree = bool(np.allclose(mesh.bounds, step_bounds, rtol=0, atol=0.01))
        export_agrees = volume_agrees and bounds_agree
        reference_evidence = None
        body_meshes = None
        mesh_valid = bool(mesh.is_watertight)
        if structured_assembly or (directory / "tessellation.json").exists():
            from forge_mind.workspace.mesh_evidence import assembly_mesh_evidence

            raw_mesh: Any = trimesh.load(str(directory / "part.stl"), force="mesh", process=False)
            body_meshes = assembly_mesh_evidence(part, raw_mesh, settings)
            mesh_valid = bool(body_meshes["valid"]) and (structured_assembly or mesh_valid)
            if not mesh.is_watertight:
                result.caveats.append(
                    "Combined STL is not watertight after contact vertices are welded. Separate body tessellations are checked against its exact oriented triangle inventory. Use STEP or separate part meshes for manufacturing; assembly validity is not combined-STL printability."
                )
        if (
            not bounds_agree
            and volume_agrees
            and is_valid
            and solid_count > 0
            and volume > 0
            and mesh_valid
            and mesh.is_winding_consistent
        ):
            # Relative OCCT deflection samples curved extrema. Never relax the
            # dimensional comparison: require exact independent STEP tessellation identity.
            if body_meshes is not None:
                reference_evidence = {
                    "triangle_inventory_matches": body_meshes["triangle_inventory_matches"],
                    "scope": body_meshes["scope"],
                }
            else:
                from forge_mind.workspace.mesh_evidence import reference_tessellation_matches

                reference_evidence = reference_tessellation_matches(part, mesh, settings)
            export_agrees = bool(reference_evidence["triangle_inventory_matches"])
        # Export agreement is numerical tessellation consistency, not a user tolerance.
        result.geometry_valid = (
            is_valid
            and solid_count > 0
            and volume > 0
            and mesh_valid
            and bool(mesh.is_winding_consistent)
            and export_agrees
        )
        from build123d import GeomType
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        complete_walls = []
        ambiguous_walls = []
        for face in part.faces():
            if face.geom_type != GeomType.CYLINDER:
                continue
            cylinders = extract_cylinders(face)
            for cylinder in cylinders:
                if cylinder.concave is False:
                    continue
                surface = BRepAdaptor_Surface(face.wrapped)
                angular_span = abs(surface.LastUParameter() - surface.FirstUParameter())
                axial_span = abs(surface.LastVParameter() - surface.FirstVParameter())
                full = (
                    cylinder.concave is True
                    and math.isclose(angular_span, 2 * math.pi, abs_tol=1e-6)
                    and math.isclose(
                        float(face.area),
                        2 * math.pi * cylinder.radius * axial_span,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                )
                if full:
                    complete_walls.append(cylinder)
                else:
                    ambiguous_walls.append(
                        {"axis": list(cylinder.axis_dir), "point": list(cylinder.axis_point)}
                    )
        holes = []
        for h in cluster_coaxial(complete_walls):
            if h.concave is True:
                holes.append(
                    {
                        "diameter_mm": 2 * h.radius,
                        "axis": list(h.axis_dir),
                        "point": list(h.axis_point),
                        "depth_mm": h.depth_mm,
                        "segments": h.segments,
                    }
                )
        facts: dict[str, Any] = {
            "extents_mm": [float(bbox.size.X), float(bbox.size.Y), float(bbox.size.Z)],
            "bbox_min": [float(bbox.min.X), float(bbox.min.Y), float(bbox.min.Z)],
            "bbox_max": [float(bbox.max.X), float(bbox.max.Y), float(bbox.max.Z)],
            "volume_mm3": volume,
            "mesh_volume_mm3": mesh_volume,
            "solid_count": solid_count,
            "watertight": bool(mesh.is_watertight),
            "body_mesh_evidence": body_meshes,
            "step_valid": is_valid,
            "export_agrees": export_agrees,
            "tessellation": {
                "linear_deflection": settings[0],
                "linear_deflection_is_relative": True,
                "angular_deflection": settings[1],
            },
            "export_consistency": {
                "volume_within_relative_0_001": volume_agrees,
                "bounds_within_0_01_mm": bounds_agree,
                "maximum_sampled_bounds_difference_mm": float(
                    np.max(np.abs(mesh.bounds - step_bounds))
                ),
                "reference_tessellation": reference_evidence,
                "scope": "Numerical export consistency only. Exact dimensions and requirement checks use STEP; sampled tessellation bounds do not define manufacturing precision.",
            },
            "holes": holes,
            "ambiguous_cylindrical_walls": ambiguous_walls,
        }
        result.facts = facts
        scoped: dict[str, Finding] = {}
        for part_id in sorted(
            {r.part_id for r in contract.requirements if r.part_id} | set(definition_mesh_ids)
        ):
            requirements = [r for r in contract.requirements if r.part_id == part_id]
            evidence, findings = definition_measurements(
                directory,
                rid,
                part_id,
                requirements,
                retain_stl=part_id in definition_mesh_ids,
            )
            facts.setdefault("definitions", {})[part_id] = evidence
            scoped.update({f.requirement_id: f for f in findings})
        for req in contract.requirements:
            try:
                result.findings.append(scoped[req.id] if req.part_id else _check(req, facts, mesh))
            except Exception as exc:
                result.findings.append(
                    _finding(
                        req, None, "measurement failed", f"Instrument failed: {type(exc).__name__}"
                    )
                )
        if not result.geometry_valid:
            result.errors.append(
                "Exported geometry failed validity, watertightness, positive volume, or STEP/STL consistency."
            )
    except TimeoutError:
        raise
    except Exception as exc:
        result.errors.append(
            f"Unable to measure exported geometry: {type(exc).__name__}: {str(exc)[:200]}"
        )
        result.findings = [
            _finding(r, None, "artifact unavailable", "No valid exported geometry to measure.")
            for r in contract.requirements
        ]
    critical = {r.id for r in contract.requirements if r.critical}
    result.requirements_satisfied = (
        result.geometry_valid
        and bool(critical)
        and all(f.status == "pass" for f in result.findings if f.requirement_id in critical)
        and len([f for f in result.findings if f.requirement_id in critical]) == len(critical)
    )
    return result


def verify_artifacts(directory: Path, measurement: Measurement) -> bool:
    return all(
        (directory / name).is_file() and file_sha256(directory / name) == digest
        for name, digest in measurement.artifact_hashes.items()
    ) and all(
        name in measurement.artifact_hashes
        for name in (
            ("program.py", "part.step", "part.stl")
            if measurement.geometry_valid
            else ("program.py",)
        )
    )


def section_revision(directory: Path, axis: str, offset: float, target: Path) -> dict[str, Any]:
    """Read-only slice of a revision; output is separate from immutable artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh: Any = trimesh.load(str(directory / "part.stl"), force="mesh")
    normal = np.zeros(3)
    normal["xyz".index(axis)] = 1
    origin = normal * offset
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return {
            "axis": axis,
            "offset_mm": offset,
            "loops": 0,
            "message": "Plane does not intersect geometry",
        }
    fig, ax = plt.subplots(figsize=(7, 7))
    indices = [i for i in range(3) if i != "xyz".index(axis)]
    try:
        for points in section.discrete:
            ax.plot(points[:, indices[0]], points[:, indices[1]], color="#2255aa")
        ax.set_aspect("equal")
        ax.set_xlabel(f"{'xyz'[indices[0]]} (mm)")
        ax.set_ylabel(f"{'xyz'[indices[1]]} (mm)")
        ax.set_title(f"Section {axis} = {offset:g} mm")
        fig.savefig(target, dpi=130)
    finally:
        plt.close(fig)
    return {"axis": axis, "offset_mm": offset, "loops": len(section.discrete), "image": target.name}
