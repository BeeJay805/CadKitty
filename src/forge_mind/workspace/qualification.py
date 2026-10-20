"""A derived evidence view, never an additional approval authority."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.engineering_service import report_path
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.product_service import product_review
from forge_mind.workspace.structural_service import list_studies, read_study

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def analysis_geometry(
    service: WorkspaceService, sid: str, owner: str, rid: str, body_index: int
) -> dict[str, Any]:
    from forge_mind.workspace.document_service import read_topology
    from forge_mind.workspace.store import WorkspaceError

    service.store.session_record(sid, owner)
    topology = read_topology(service, sid, rid)
    bodies = [
        e for e in topology["entities"] if e["kind"] == "body" and e["body_index"] == body_index
    ]
    if len(bodies) != 1:
        raise WorkspaceError("Analysis body is missing or ambiguous")
    faces = [
        e for e in topology["entities"] if e["kind"] == "face" and e["body_index"] == body_index
    ]
    return {
        "revision_id": rid,
        "frame": "cad_z_up_mm",
        "body": bodies[0],
        "faces": [{k: v for k, v in e.items() if k != "polyline"} for e in faces],
        "face_inventory_available": bool(faces),
        "scope": "Exact revision-local exported identity. Missing face inventory requires a less detailed component export; never guess signatures.",
    }


def qualification_review(
    service: WorkspaceService, sid: str, owner: str, rid: str
) -> dict[str, Any]:
    session = service.store.get(sid, owner)
    revision = service._load_revision(sid, rid)
    evidence = revision["measurement"]
    fingerprint = engine_fingerprint()
    fresh = evidence["engine_fingerprint"] == fingerprint
    product = product_review(service, sid, owner, rid)
    studies = list_studies(service, sid, owner, rid)
    # A compact list checks receipt identity only. Coverage must verify the actual
    # field bytes before presenting a calculation as available evidence.
    studies = [read_study(service, sid, owner, s["id"]) if not s["stale"] else s for s in studies]
    current_studies = [s for s in studies if not s["stale"]]
    checks = [c for c in product["checks"] if not c["stale"]]
    receipt = next(
        (
            r
            for r in reversed(session.get("reports", []))
            if r["revision_id"] == rid
            and r["input_version"] == session["engineering"]["version"]
            and r["engine_fingerprint"] == fingerprint
        ),
        None,
    )
    report = (
        json.loads(
            report_path(service, sid, owner, receipt["id"], "report.json").read_text(
                encoding="utf-8"
            )
        )
        if receipt
        else None
    )
    production = product.get("production") or {
        "lines": [],
        "gaps": ["Production roles and materials are not declared"],
    }
    fields = (report or {}).get("simulation", [])
    structural_comparisons = [c for s in current_studies for c in s.get("comparisons", [])]
    fits = (report or {}).get("tolerances", [])
    manufacturing = (report or {}).get("manufacturing", [])
    fitting = [
        c
        for c in checks
        if c["request"]["tool"]
        in {"fastener_stack", "tolerance_stack", "hole_pattern", "mount_grid"}
    ]
    categories = []

    def add(
        identifier: str,
        title: str,
        status: str,
        scope: str,
        count: int,
        missing: list[str],
        action: str,
    ) -> None:
        categories.append(
            {
                "id": identifier,
                "title": title,
                "status": status,
                "scope": scope,
                "evidence_count": count,
                "missing": missing,
                "next_action": action,
            }
        )

    add(
        "geometry",
        "Geometry",
        "pass"
        if fresh and evidence["geometry_valid"]
        else "fail"
        if not evidence["geometry_valid"]
        else "unverified",
        "Exported STEP/STL identity and geometry validity",
        1,
        [] if fresh else ["Current-engine rebuild"],
        "Inspect geometry and rebuild stale evidence",
    )
    requirements = product["obligations"]
    requirement_status = (
        "fail"
        if any(r["status"] == "fail" for r in requirements)
        else "pass"
        if requirements and all(r["status"] == "pass" for r in requirements) and fresh
        else "unverified"
    )
    add(
        "requirements",
        "Design requirements",
        requirement_status,
        "The original frozen findings and explicit interface evidence",
        len(requirements),
        [r["text"] for r in requirements if r["status"] != "pass"],
        "Inspect each requirement and its existing measured interface; exploratory studies cannot approve requirements",
    )
    structural_fail = any(c["status"] == "fail" for c in structural_comparisons) or any(
        f.get("status") == "fail" for f in fields
    )
    add(
        "structure",
        "Stress and deformation",
        "fail"
        if structural_fail
        else "computed"
        if any(s["status"] == "computed" for s in current_studies)
        else "unverified",
        "Selected component response under a declared linear-static setup",
        len(current_studies) + len(fields),
        []
        if current_studies
        else ["Component, material, force, restraint and numerical comparison"],
        "Set up a component study and inspect both mesh refinements",
    )
    motion = None
    motion_ref = next(
        (
            r
            for r in reversed(session.get("motion_checks", []))
            if r["revision_id"] == rid and r["engine_fingerprint"] == fingerprint
        ),
        None,
    )
    if motion_ref:
        from forge_mind.workspace.motion_service import check_path

        motion = json.loads(
            check_path(service, sid, owner, motion_ref["id"]).read_text(encoding="utf-8")
        )
    add(
        "motion",
        "Motion and interference",
        "fail" if motion and motion.get("collisions") else "computed" if motion else "unverified",
        "Recorded sampled poses only; no continuous clearance or contact-force certificate",
        int(motion is not None),
        ["Continuous clearance and physical load transfer"],
        "Inspect recorded positions and collision witnesses in Motion",
    )
    add(
        "materials",
        "Materials and purchasing",
        "unverified",
        "Per-part declarations and source interpretation, not tested material properties",
        len(production["lines"]),
        [
            *production.get("gaps", []),
            "Material applicability and supplier suitability require independent evidence",
        ],
        "Inspect material grades, source claims and purchase identities for each part",
    )
    add(
        "manufacturing",
        "Manufacturing",
        "fail"
        if any(c["status"] == "fail" for c in manufacturing)
        else "computed"
        if manufacturing and any("measured_mm" in c or "sampled_min_mm" in c for c in manufacturing)
        else "unverified",
        "Declared process limits and supported geometry checks; setup and process capability remain separate",
        len(manufacturing),
        ["Per-make-part process capability, tooling, fixtures and inspection plan"],
        "Assign part processes and retain measured setup/inspection evidence",
    )
    add(
        "fits",
        "Fits and fasteners",
        "fail"
        if any(c["result"]["status"] == "fail" for c in fitting)
        or any(c["status"] == "fail" for c in fits)
        else "computed"
        if fits or fitting
        else "unverified",
        "Declared tolerance intervals and supported geometric fit probes",
        len(fits) + len(fitting),
        ["Measured feature binding, thread engagement, preload and joint strength"],
        "Measure interfaces and provide sourced stack, thread and load limits",
    )
    add(
        "contact",
        "Assembly load transfer",
        "unverified",
        "Component stress does not determine force transfer between parts",
        0,
        ["Contact, friction, preload, bearing reactions and joint stiffness"],
        "Define and validate the actual load path and connection model",
    )
    add(
        "stability",
        "Buckling, fatigue and dynamics",
        "unverified",
        "A linear-static study does not evaluate these failure modes",
        0,
        ["Mode applicability, stability, cyclic load history and dynamic response"],
        "Specify applicable failure modes and retain an appropriate solver or test result",
    )
    add(
        "thermal",
        "Temperature and fluid flow",
        "unverified",
        "No thermal or fluid solver result has been established for this revision",
        0,
        ["Thermal/flow applicability, boundary conditions and validated solver evidence"],
        "Provide the thermal or flow operating case when relevant",
    )
    feedback = session["engineering"]["inputs"].get("feedback", [])
    current_feedback = [f for f in feedback if f.get("revision_id") == rid]
    add(
        "physical",
        "Physical verification",
        "computed" if current_feedback else "unverified",
        "Recorded measurements and uncertainty; no automatic product certification",
        len(current_feedback),
        ["Independent functional, material and manufacturing tests"],
        "Record measured outcomes and uncertainty against this exact revision",
    )

    from forge_mind.workspace.document_service import read_topology

    topology = read_topology(service, sid, rid)
    bodies = [
        {
            k: v
            for k, v in e.items()
            if k in {"id", "body_index", "signature", "center", "min", "max", "measure", "unit"}
        }
        for e in topology["entities"]
        if e["kind"] == "body"
    ]
    assembly_rows = evidence.get("facts", {}).get("assembly", {}).get("occurrences", [])
    for body in bodies:
        occurrence = next(
            (o for o in assembly_rows if o.get("body_index") == body["body_index"]), None
        )
        body["occurrence_path"] = occurrence.get("path") if occurrence else None
        line = next(
            (
                line
                for line in production["lines"]
                if body["occurrence_path"] in line.get("occurrence_paths", [])
            ),
            None,
        )
        if not assembly_rows and len(production["lines"]) == 1:
            line = production["lines"][0]
        body["name"] = line["name"] if line else f"Body {body['body_index'] + 1}"
        body["production"] = line["production"] if line else None
    witnesses = [
        p
        for p in (report or {}).get("assembly", {}).get("pairs", [])
        if p.get("overlap_mm3", 0) > 1e-6
    ]
    dimensions = [
        {
            "check_id": c["id"],
            "tool": c["request"]["tool"],
            "status": c["result"]["status"],
            "results": c["result"]["results"],
            "scope": c["result"]["algorithm_scope"],
        }
        for c in checks
        if c["request"]["tool"] in {"projected_pair", "planar_face_pair"}
    ]
    return {
        "schema_version": 1,
        "revision_id": rid,
        "geometry_fresh": fresh,
        "input_version": session["engineering"]["version"],
        "physical_readiness": "unverified",
        "categories": categories,
        "requirements": requirements,
        "structural_studies": studies,
        "production": production,
        "report_id": receipt["id"] if receipt else None,
        "spatial": {
            "bodies": bodies,
            "witnesses": witnesses[:100],
            "witness_count": len(witnesses),
            "measurements": dimensions,
            "fits": fits,
            "fit_probes": [
                {"id": c["id"], "request": c["request"], "result": c["result"]} for c in fitting
            ],
        },
        "scope": "Derived view of existing evidence. Computed and passing scoped checks do not automatically qualify a physical product or rewrite frozen requirements.",
    }
