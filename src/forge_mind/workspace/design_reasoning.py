"""Declared design reasons and executable, revision-bound engineering challenges."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

from forge_mind.workspace.document import DesignDocument, GeometryRef, Name
from forge_mind.workspace.document_service import read_topology
from forge_mind.workspace.lab_records import capture_binding, list_records, read_record, save_record
from forge_mind.workspace.models import Record
from forge_mind.workspace.product_sources import read_sources
from forge_mind.workspace.project_access import design_relationships
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


class DesignTarget(Record):
    part_id: Name | None = None
    feature_id: Name | None = None
    reference: GeometryRef | None = None

    @model_validator(mode="after")
    def chosen(self) -> DesignTarget:
        if not (self.part_id or self.feature_id or self.reference):
            raise ValueError("Choose a part, feature or current geometry reference")
        return self


class RationaleRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    input_version: int = Field(ge=0)
    target: DesignTarget
    purpose: str = Field(min_length=1, max_length=3000)
    variable_names: list[Name] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    assumption_ids: list[Name] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ChallengeRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    input_version: int = Field(ge=0)
    question: str = Field(min_length=1, max_length=3000)
    variables: dict[Name, float] = Field(min_length=1)
    reference: GeometryRef | None = None
    part_id: Name | None = None


def affected_paths(document: DesignDocument, placements: list[str]) -> set[str]:
    """Expand a changed reusable placement through each parent occurrence to actual leaves."""
    if not document.assembly:
        return set()
    groups = {g.id: g for g in document.assembly.subassemblies}
    result: set[str] = set()

    def visit(children: Any, prefix: str, group: str, inherited: bool) -> None:
        for occurrence in children:
            path = f"{prefix}/{occurrence.id}" if prefix else occurrence.id
            changed = inherited or f"{group}/{occurrence.id}" in placements
            if occurrence.definition in groups:
                visit(groups[occurrence.definition].children, path, occurrence.definition, changed)
            elif changed:
                result.add(path)

    visit(document.assembly.roots, "", "root", False)
    return result


def _part(
    service: WorkspaceService,
    sid: str,
    revision: dict[str, Any],
    document: DesignDocument,
    *,
    body_index: int | None = None,
    part_id: str | None = None,
    feature_id: str | None = None,
    reference: GeometryRef | None = None,
) -> dict[str, Any]:
    if reference:
        if reference.revision_id != revision["id"]:
            raise WorkspaceError("Selection belongs to a different design revision")
        topology = read_topology(service, sid, revision["id"])
        entity = next(
            (
                e
                for e in topology["entities"]
                if (
                    e["id"] == reference.entity_id
                    and e["signature"] == reference.signature
                    and e["body_signature"] == reference.body_signature
                    and e["kind"] == reference.kind
                )
            ),
            None,
        )
        if entity is None:
            raise WorkspaceError("Selected geometry signature no longer resolves")
        body_index = entity["body_index"]
    occurrences = (
        revision["measurement"].get("facts", {}).get("assembly", {}).get("occurrences", [])
    )
    selection_scope = "feature" if feature_id else "part" if part_id else "document"
    if body_index is not None:
        if document.assembly:
            found = [o for o in occurrences if o["body_index"] == body_index]
            if len(found) != 1 or (part_id is not None and part_id != found[0]["definition"]):
                raise WorkspaceError("Selected body does not resolve uniquely to this part")
            part_id = found[0]["definition"]
            selection_scope = "part"
        else:
            count = revision["measurement"].get("facts", {}).get("solid_count")
            if count is None or not 0 <= body_index < count:
                raise WorkspaceError("Selected body is not part of the saved geometry")
            if count == 1 and len(document.outputs) == 1:
                output = document.outputs[0]
                if feature_id is not None and feature_id != output:
                    raise WorkspaceError("Selected body does not establish the requested feature")
                feature_id = output
                selection_scope = "feature"
            else:
                if feature_id is not None:
                    raise WorkspaceError("This saved body has no verified output-feature mapping")
                selection_scope = "unmapped_body"
    definition = (
        next((p for p in document.assembly.parts if p.id == part_id), None)
        if document.assembly
        else None
    )
    if part_id and definition is None:
        raise WorkspaceError("Part definition not found")
    features = definition.features if definition else document.features
    if feature_id and not any(f.id == feature_id for f in features):
        raise WorkspaceError("Feature not found in the selected definition")
    paths = [o for o in occurrences if part_id and o["definition"] == part_id]
    return {
        "id": part_id or feature_id,
        "name": definition.name
        if definition
        else f"Body {body_index + 1}"
        if selection_scope == "unmapped_body" and body_index is not None
        else next((f.name for f in features if f.id == feature_id), document.name),
        "feature_id": feature_id,
        "selection_scope": selection_scope,
        "occurrence_paths": [o["path"] for o in paths],
        "body_indices": [o["body_index"] for o in paths]
        if paths
        else ([] if body_index is None else [body_index]),
        "production": definition.production.model_dump()
        if definition and definition.production
        else document.production.model_dump()
        if document.production
        else None,
        "features": []
        if selection_scope == "unmapped_body"
        else [f.model_dump() for f in features if not feature_id or f.id == feature_id],
    }


def evidence_inventory(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
) -> list[dict[str, Any]]:
    from forge_mind.workspace.measure import engine_fingerprint
    from forge_mind.workspace.structural_service import list_studies

    session = service.store.get(sid, owner)
    version = session["engineering"]["version"]
    engine = engine_fingerprint()
    evidence: list[dict[str, Any]] = []
    for category, key in (
        ("report", "reports"),
        ("motion", "motion_checks"),
        ("parameter_study", "parameter_studies"),
    ):
        for item in session.get(key, []):
            source = item.get("revision_id", item.get("source_revision"))
            evidence.append(
                {
                    "id": f"{category}:{item['id']}",
                    "kind": category,
                    "status": "recorded",
                    "revision_id": source,
                    "stale": source != rid
                    or item.get("engine_fingerprint") != engine
                    or item.get("input_version", version) != version,
                    "scope": "Read the retained artifact for the exact checked scope",
                    "summary": "Recorded engineering evidence",
                    "receipt": item,
                }
            )
    for item in list_studies(service, sid, owner):
        evidence.append(
            {
                "id": f"structural:{item['id']}",
                "kind": "structural",
                "status": item.get("status"),
                "revision_id": item["revision_id"],
                "stale": item["stale"] or item["revision_id"] != rid,
                "scope": item.get("scope", "Declared isolated-component linear elasticity"),
                "summary": item.get("request", {}).get("name", "Structural study"),
                "target": item.get("request", {}).get("target"),
            }
        )
    for item in list_records(service, sid, owner):
        evidence.append(
            {
                "id": "lab:" + item["id"],
                "kind": item["kind"],
                "status": item.get("status", "recorded"),
                "revision_id": item["binding"]["revision_id"],
                "stale": item["stale"],
                "scope": item.get(
                    "scope", "Immutable investigation; scope is defined by its protocol"
                ),
                "summary": item.get("question", item.get("name", item["kind"])),
                "receipt_sha256": item["receipt_sha256"],
            }
        )
    return evidence


def explain_design(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    *,
    body_index: int | None = None,
    part_id: str | None = None,
    feature_id: str | None = None,
) -> dict[str, Any]:
    binding = capture_binding(service, sid, owner, rid, require_current=False)
    revision = service._load_revision(sid, rid)
    document = DesignDocument.model_validate(revision["document"])
    part = _part(
        service,
        sid,
        revision,
        document,
        body_index=body_index,
        part_id=part_id,
        feature_id=feature_id,
    )
    feature_id = part["feature_id"]
    graph = design_relationships(document)
    variables = []
    from forge_mind.workspace.inspection import reachable, variable_impact

    targets = []
    if document.assembly and part["id"]:
        targets.append("part/" + part["id"])
        targets.extend(
            node["id"]
            for node in graph["nodes"]
            if node["kind"] == "placement"
            and affected_paths(document, [node["name"]]).intersection(part["occurrence_paths"])
        )
    elif feature_id:
        targets.append("feature/" + feature_id)
    ancestors = set(reachable(graph, targets, reverse=True))
    for name in document.variables:
        if part["selection_scope"] != "document" and "variable/" + name not in ancestors:
            continue
        impact = variable_impact(graph, [name])
        if part["selection_scope"] == "unmapped_body":
            continue
        if (
            part["id"]
            and document.assembly
            and part["id"] not in impact["parts"]
            and not affected_paths(document, impact["placements"]).intersection(
                part["occurrence_paths"]
            )
        ):
            continue
        if feature_id and not document.assembly and feature_id not in impact["features"]:
            continue
        variables.append({"name": name, "value": document.variables[name], "impact": impact})
    records = list_records(service, sid, owner, "rationale")
    purpose = [
        r
        for r in records
        if r["binding"]["revision_id"] == rid
        and (
            part["selection_scope"] == "document"
            or (part_id is not None and r["target"].get("part_id") == part["id"])
            or (document.assembly is not None and r["target"].get("part_id") == part["id"])
            or (feature_id and r["target"].get("feature_id") == feature_id)
            or (body_index is not None and body_index in r.get("body_indices", []))
        )
    ]
    inputs = service.store.engineering(sid)["inputs"]
    requirements = (revision.get("contract") or {}).get("requirements", [])
    findings = {f["requirement_id"]: f for f in revision["measurement"].get("findings", [])}
    return {
        "binding": binding,
        "part": part,
        "variables": variables,
        "relations": graph["relations"],
        "purpose": purpose,
        "assumptions": inputs.get("assumptions", []),
        "material": inputs.get("material"),
        "requirements": [{**r, "finding": findings.get(r["id"])} for r in requirements],
        "evidence": evidence_inventory(service, sid, owner, rid),
        "affected_body_indices": part["body_indices"],
        "limitations": [
            "Purpose and evidence links are authored explanations, not independently confirmed causality.",
            "Dependencies cover explicit CAD variables; unstated physical connections are unknown.",
            "Existing component studies do not establish assembly contact or complete force transfer.",
            *(
                [
                    "This standalone body has no saved feature mapping; only rationale explicitly linked to its geometry is shown."
                ]
                if part["selection_scope"] == "unmapped_body"
                else []
            ),
        ],
        "suggested_questions": [
            "Why is this dimension needed?",
            "What fails if I reduce its size?",
            "Which experiment would test this assumption?",
        ],
    }


def add_rationale(
    service: WorkspaceService,
    sid: str,
    owner: str,
    request: RationaleRequest,
) -> dict[str, Any]:
    binding = capture_binding(service, sid, owner, request.revision_id, request.input_version)
    revision = service._load_revision(sid, request.revision_id)
    document = DesignDocument.model_validate(revision["document"])
    part = _part(
        service,
        sid,
        revision,
        document,
        **request.target.model_dump(exclude={"reference"}),
        reference=request.target.reference,
    )
    known = {
        "variable_names": set(document.values()),
        "requirement_ids": {
            r["id"] for r in (revision.get("contract") or {}).get("requirements", [])
        },
        "assumption_ids": {
            a["id"] for a in service.store.engineering(sid)["inputs"].get("assumptions", [])
        },
        "source_claim_ids": {c["id"] for c in read_sources(service.store, sid, owner)["claims"]},
    }
    for key, ids in known.items():
        if not set(getattr(request, key)) <= ids:
            raise WorkspaceError(f"Rationale references unknown {key}")
    evidence = evidence_inventory(service, sid, owner, request.revision_id)
    links = [e for e in evidence if e["id"] in request.evidence_ids]
    if len(links) != len(set(request.evidence_ids)) or any(e["stale"] for e in links):
        raise WorkspaceError("Rationale evidence is unavailable or stale")
    if not request.purpose.strip():
        raise WorkspaceError("Describe the intended purpose")
    return save_record(
        service,
        sid,
        owner,
        "rationale",
        binding,
        {
            **request.model_dump(exclude={"revision_id", "input_version"}),
            "target": {
                **request.target.model_dump(),
                "part_id": part["id"] if document.assembly else None,
            },
            "body_indices": part["body_indices"],
            "status": "authored",
            "evidence_links": links,
            "scope": "Author-declared purpose and linked evidence; the relationship itself is not a validated engineering claim",
        },
        parents=[e[4:] for e in request.evidence_ids if e.startswith("lab:")],
    )


def preview_challenge(
    service: WorkspaceService,
    sid: str,
    owner: str,
    request: ChallengeRequest,
) -> dict[str, Any]:
    binding = capture_binding(service, sid, owner, request.revision_id, request.input_version)
    revision = service._load_revision(sid, request.revision_id)
    document = DesignDocument.model_validate(revision["document"])
    if request.reference or request.part_id:
        _part(
            service, sid, revision, document, reference=request.reference, part_id=request.part_id
        )
    if not request.question.strip() or not request.variables.keys() <= document.variables.keys():
        raise WorkspaceError("Describe the challenge and change existing input variables")
    changes = [
        {"name": name, "before": document.variables[name], "after": value}
        for name, value in request.variables.items()
        if document.variables[name] != value
    ]
    if not changes:
        raise WorkspaceError("The challenge does not change a design variable")
    proposed = document.model_dump()
    proposed["variables"].update(request.variables)
    candidate = DesignDocument.model_validate(proposed)
    graph = design_relationships(document)
    from forge_mind.workspace.inspection import variable_impact

    impact = variable_impact(graph, [str(change["name"]) for change in changes])
    occurrences = (
        revision["measurement"].get("facts", {}).get("assembly", {}).get("occurrences", [])
    )
    moved = affected_paths(document, impact["placements"])
    body_indices = [
        o["body_index"]
        for o in occurrences
        if o["definition"] in impact["parts"] or o["path"] in moved or impact["mechanism"]
    ]
    if not document.assembly and impact["features"]:
        body_indices = list(range(revision["measurement"].get("facts", {}).get("solid_count", 0)))
    evidence = evidence_inventory(service, sid, owner, request.revision_id)
    return save_record(
        service,
        sid,
        owner,
        "challenge",
        binding,
        {
            "question": request.question,
            "changes": changes,
            "impact": {**impact, "body_indices": body_indices},
            "invalidated_evidence": [
                {
                    "id": e["id"],
                    "kind": e["kind"],
                    "reason": "Evidence binds the previous design; applicability must be re-established after this edit",
                }
                for e in evidence
                if e["revision_id"] == request.revision_id
            ],
            "required_checks": [
                {"kind": "geometry", "reason": "Rebuild and measure actual exported geometry"},
                {"kind": "requirements", "reason": "Recheck the unchanged original requirements"},
                {
                    "kind": "engineering",
                    "reason": "Recompute applicable interfaces, motion and load studies; physical tests remain separate",
                },
            ],
            "candidate_document": candidate.model_dump(),
            "candidate_document_hash": candidate.fingerprint(),
            "original_contract_hash": binding["contract_hash"],
            "status": "proposal",
            "scope": "Executable parameter proposal with declared dependency impacts; rebuild required before geometry or performance claims",
        },
    )


def apply_challenge(
    service: WorkspaceService, sid: str, owner: str, identifier: str
) -> dict[str, Any]:
    record = read_record(service, sid, owner, identifier, "challenge")
    if record["stale"]:
        raise WorkspaceError("Challenge is stale; preview it against the current design")
    candidate = DesignDocument.model_validate(record["candidate_document"])
    if candidate.fingerprint() != record["candidate_document_hash"]:
        raise WorkspaceError("Challenge candidate changed")
    revision = service._load_revision(sid, record["binding"]["revision_id"])
    job_id = service.store.enqueue(
        sid,
        owner,
        {
            "document": candidate.model_dump(),
            "revision_id": revision["id"],
            "contract": revision["contract"],
            "expected_current_revision": revision["id"],
            "expected_input_version": record["binding"]["input_version"],
            "preview_only": True,
            "explanation": record["question"],
            "lab_challenge_id": identifier,
            "lab_binding": record["binding"],
        },
    )
    service.start()
    return {"id": sid, "job_id": job_id, "state": "queued"}
