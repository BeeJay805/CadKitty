"""Owned, unadmitted planning input with typed corrections and explicit admission."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import Field, StrictStr, model_validator

from forge_mind.workspace.completion_models import (
    FunctionalInterface,
    RequirementDisposition,
    Subsystem,
)
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.inspection_models import InspectionCharacteristic
from forge_mind.workspace.models import (
    Contract,
    DesignPlan,
    DimensionFact,
    OperationFrame,
    OperationStep,
    PlanName,
    Record,
)
from forge_mind.workspace.store import EditConflictError, Store, WorkspaceError
from forge_mind.workspace.verification_models import VerificationSubjectBinding

_PROSE = ("objective", "parts", "interfaces", "assembly_sequence", "verification", "unresolved")
_COLLECTIONS = (
    "subsystems",
    "inspection",
    "requirement_dispositions",
    "functional_interfaces",
    "verification_bindings",
)
_OPERATIONS = {"operation_steps": "steps", "operation_frames": "frames", "dimensions": "dimensions"}
Collection = Literal[
    "metadata",
    "parts",
    "interfaces",
    "assembly_sequence",
    "verification",
    "unresolved",
    "subsystems",
    "inspection",
    "requirement_dispositions",
    "functional_interfaces",
    "verification_bindings",
    "operation_steps",
    "operation_frames",
    "dimensions",
    "part_ids",
    "diagnostics",
]


class PlanDraftChanges(Record):
    """Locally typed records; cross-record validity is checked only at admission."""

    objective: StrictStr | None = Field(default=None, min_length=1, max_length=2000)
    parts: list[StrictStr] | None = None
    interfaces: list[StrictStr] | None = None
    assembly_sequence: list[StrictStr] | None = None
    verification: list[StrictStr] | None = None
    unresolved: list[StrictStr] | None = None
    part_ids: list[PlanName] | None = None
    operation_steps: list[OperationStep] = Field(default_factory=list)
    operation_frames: list[OperationFrame] = Field(default_factory=list)
    dimensions: list[DimensionFact] = Field(default_factory=list)
    subsystems: list[Subsystem] = Field(default_factory=list)
    inspection: list[InspectionCharacteristic] = Field(default_factory=list)
    requirement_dispositions: list[RequirementDisposition] = Field(default_factory=list)
    functional_interfaces: list[FunctionalInterface] = Field(default_factory=list)
    verification_bindings: list[VerificationSubjectBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def explicit_changes(self) -> PlanDraftChanges:
        if not self.model_fields_set:
            raise ValueError("Draft changes must contain at least one explicit field")
        if any(getattr(self, key) is None for key in self.model_fields_set):
            raise ValueError("Omit unchanged draft fields; explicit null is not a correction")
        return self


class PlanDraftPatch(Record):
    basis: EditBasis
    draft_version: int = Field(ge=0, strict=True)
    draft_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    changes: PlanDraftChanges
    replace_collections: list[Collection] = Field(default_factory=list)
    replace_operation_plan: bool = Field(default=False, strict=True)
    remove_unknown_fields: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def identity(self) -> PlanDraftPatch:
        if (self.draft_version == 0) != (self.draft_hash is None):
            raise ValueError(
                "Version zero initializes; later corrections need the exact draft hash"
            )
        if len(set(self.remove_unknown_fields)) != len(self.remove_unknown_fields) or any(
            not key or key in DesignPlan.model_fields for key in self.remove_unknown_fields
        ):
            raise ValueError("Only explicitly named unknown top-level fields can be removed")
        if len(set(self.replace_collections)) != len(self.replace_collections) or any(
            key not in self.changes.model_fields_set for key in self.replace_collections
        ):
            raise ValueError("Replacement collections must be unique explicit changes")
        if self.replace_operation_plan and not any(
            key in self.changes.model_fields_set for key in (*_OPERATIONS, "part_ids")
        ):
            raise ValueError("Operation-plan replacement requires explicit typed operation records")
        return self


class PlanDraftCommit(Record):
    basis: EditBasis
    draft_version: int = Field(ge=1, strict=True)
    draft_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class PlanDraftRead(Record):
    collection: Collection = "metadata"
    cursor: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=20, ge=1, le=50, strict=True)
    record_id: StrictStr | None = Field(default=None, min_length=1, max_length=256)


def _encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def _latest(con: Any, sid: str) -> dict[str, Any] | None:
    row = con.execute(
        "SELECT * FROM plan_drafts WHERE session=? ORDER BY version DESC LIMIT 1", (sid,)
    ).fetchone()
    return dict(row) if row is not None else None


def _owned(con: Any, sid: str, owner: str) -> dict[str, Any]:
    row = con.execute(
        "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
    ).fetchone()
    if row is None:
        raise WorkspaceError("workspace not found")
    return dict(json.loads(row[0]))


def _matches(row: dict[str, Any] | None, version: int, digest: str | None) -> bool:
    return bool(
        row and not row["admitted"] and row["version"] == version and row["content_hash"] == digest
    )


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    content = json.loads(row["content"])
    raw_operations = content.get("operation_plan")
    operations = raw_operations if isinstance(raw_operations, dict) else {}
    diagnostics = json.loads(row["validation"])
    conflicts: list[dict[str, Any]] = []
    for conflict in diagnostics.get("conflicts", []):
        if len(_encoded([*conflicts, conflict]).encode("utf-8")) > 8000:
            continue
        conflicts.append(conflict)
        if len(conflicts) >= 20:
            break
    return {
        "status": "unadmitted_draft",
        "draft_version": row["version"],
        "draft_hash": row["content_hash"],
        "basis": json.loads(row["basis"]),
        "objective": content["objective"][:2000]
        if isinstance(content.get("objective"), str)
        else None,
        "collections": {
            key: len(content.get(key, [])) if isinstance(content.get(key, []), list) else None
            for key in (*_PROSE[1:], *_COLLECTIONS)
        }
        | {
            key: len(operations.get(value, []))
            if isinstance(operations.get(value, []), list)
            else None
            for key, value in {**_OPERATIONS, "part_ids": "part_ids"}.items()
        },
        "validation_error": diagnostics.get("message", "")[:4000],
        "conflicts": conflicts,
        "conflict_count": len(diagnostics.get("conflicts", [])),
        "omitted_conflicts": len(diagnostics.get("conflicts", [])) - len(conflicts),
        "source_bytes": len(row["raw_json"].encode("utf-8")),
        "source_sha256": hashlib.sha256(row["raw_json"].encode("utf-8")).hexdigest(),
        "plan_applied": False,
        "cad_applied": False,
        "scope": "Unadmitted input only. Read affected records, correct them, then explicitly commit.",
    }


def summary(store: Store, sid: str, owner: str) -> dict[str, Any] | None:
    with store.connect() as con:
        doc = _owned(con, sid, owner)
        row = _latest(con, sid)
        if row is None or row["admitted"]:
            return None
        result = _metadata(row)
        if store._edit_basis(con, sid, doc).model_dump() != result["basis"]:
            result["status"] = "stale_draft"
        return result


def read_draft(store: Store, sid: str, owner: str, query: PlanDraftRead) -> dict[str, Any]:
    with store.connect() as con:
        _owned(con, sid, owner)
        row = _latest(con, sid)
        if row is None or row["admitted"]:
            raise WorkspaceError("No unadmitted plan draft is available")
        result = _metadata(row)
        if query.collection == "metadata":
            return result
        content = json.loads(row["content"])
        if query.collection == "diagnostics":
            values = json.loads(row["validation"]).get("conflicts", [])
        elif query.collection in _OPERATIONS or query.collection == "part_ids":
            operations = content.get("operation_plan", {})
            if not isinstance(operations, dict):
                raise WorkspaceError("The rejected operation plan is not an object")
            values = operations.get(_OPERATIONS.get(query.collection, query.collection), [])
        else:
            values = content.get(query.collection, [])
        if not isinstance(values, list):
            raise WorkspaceError(
                "The rejected collection is not a list; replace that typed collection"
            )
        if query.record_id is not None:
            values = [
                value
                for value in values
                if isinstance(value, dict)
                and (
                    value.get("id", value.get("requirement_id", value.get("subject_id")))
                    == query.record_id
                )
            ]
        result.update(
            collection=query.collection,
            records=values[query.cursor : query.cursor + query.limit],
            next_cursor=query.cursor + query.limit
            if len(values) > query.cursor + query.limit
            else None,
            total=len(values),
        )
        return result


def _upsert(previous: Any, changes: list[Any], key: str) -> list[dict[str, Any]]:
    if not isinstance(previous, list) or any(
        not isinstance(value, dict) or key not in value for value in previous
    ):
        raise WorkspaceError("Draft collection needs identified records before a typed upsert")
    identities = [value[key] for value in previous]
    replacements = [getattr(value, key) for value in changes]
    if any(not isinstance(value, str) for value in identities):
        raise WorkspaceError("Draft record identifiers must be strings")
    if len(set(identities)) != len(identities) or len(set(replacements)) != len(replacements):
        raise WorkspaceError(
            "Draft record identifiers must be unique; ambiguous upserts are refused"
        )
    result = {value[key]: deepcopy(value) for value in previous}
    result.update({getattr(value, key): value.model_dump(mode="json") for value in changes})
    return list(result.values())


def _merge(content: dict[str, Any], patch: PlanDraftPatch) -> dict[str, Any]:
    changes = patch.changes
    result = deepcopy(content)
    for key in patch.remove_unknown_fields:
        if key not in result:
            raise WorkspaceError("Unknown-field removal must name an existing draft field")
        del result[key]
    for key in _PROSE:
        if key in changes.model_fields_set:
            result[key] = getattr(changes, key)
    for key in _COLLECTIONS:
        if key in changes.model_fields_set:
            result[key] = (
                [value.model_dump(mode="json") for value in getattr(changes, key)]
                if key in patch.replace_collections
                else _upsert(
                    result.get(key, []),
                    getattr(changes, key),
                    "requirement_id"
                    if key == "requirement_dispositions"
                    else "subject_id"
                    if key == "verification_bindings"
                    else "id",
                )
            )
    if any(key in changes.model_fields_set for key in (*_OPERATIONS, "part_ids")):
        if patch.replace_operation_plan:
            result["operation_plan"] = {}
        operations = result.setdefault("operation_plan", {})
        if not isinstance(operations, dict):
            raise WorkspaceError("Draft operation plan is not an object")
        if "part_ids" in changes.model_fields_set:
            operations["part_ids"] = changes.part_ids
        for key, canonical in _OPERATIONS.items():
            if key in changes.model_fields_set:
                operations[canonical] = (
                    [value.model_dump(mode="json") for value in getattr(changes, key)]
                    if key in patch.replace_collections or patch.replace_operation_plan
                    else _upsert(operations.get(canonical, []), getattr(changes, key), "id")
                )
    return result


def _diagnostics(content: dict[str, Any], error: str = "") -> str:
    from forge_mind.workspace.operation_plan import plan_semantic_conflicts

    return _encoded({"message": error, "conflicts": plan_semantic_conflicts(content)})


def _insert(
    con: Any,
    sid: str,
    basis: EditBasis,
    content: dict[str, Any],
    raw_json: str,
    previous: dict[str, Any] | None,
    error: str = "",
) -> dict[str, Any]:
    version = previous["version"] + 1 if previous else 1
    con.execute(
        "INSERT INTO plan_drafts VALUES (?,?,?,?,?,?,?,?,?)",
        (
            sid,
            version,
            _encoded(basis.model_dump()),
            _encoded(content),
            raw_json,
            _diagnostics(content, error),
            _hash(content),
            0,
            None,
        ),
    )
    row = _latest(con, sid)
    assert row is not None
    return _metadata(row)


def retain_rejected_plan(
    store: Store,
    sid: str,
    owner: str,
    raw_json: str,
    basis: EditBasis,
    jid: str | None = None,
    error: str = "",
) -> dict[str, Any]:
    from forge_mind.workspace.action_payloads import decode_json_object

    content = decode_json_object(raw_json)
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        store._compare_edit_basis(con, sid, doc, basis)
        store._check_plan_writer(con, sid, doc, jid)
        return _insert(con, sid, basis, content, raw_json, _latest(con, sid), error)


def patch_draft(
    store: Store, sid: str, owner: str, patch: PlanDraftPatch, jid: str | None = None
) -> dict[str, Any]:
    patch = PlanDraftPatch.model_validate(patch.model_dump(exclude_unset=True))
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        store._compare_edit_basis(con, sid, doc, patch.basis)
        store._check_plan_writer(con, sid, doc, jid)
        row = _latest(con, sid)
        if patch.draft_version == 0:
            if row and not row["admitted"] and json.loads(row["basis"]) == patch.basis.model_dump():
                raise EditConflictError(
                    "An active draft exists; read its identity before correcting it"
                )
            content: dict[str, Any] = {}
            raw_json = "{}"
        else:
            if not _matches(row, patch.draft_version, patch.draft_hash):
                raise EditConflictError(
                    "Plan draft changed; read the current draft before correcting it"
                )
            assert row is not None
            if json.loads(row["basis"]) != patch.basis.model_dump():
                raise EditConflictError("Plan draft belongs to another editing basis")
            content = json.loads(row["content"])
            raw_json = row["raw_json"]
        return _insert(con, sid, patch.basis, _merge(content, patch), raw_json, row)


def check_draft_for_admission(
    con: Any, sid: str, request: PlanDraftCommit, plan: DesignPlan
) -> None:
    """Called inside the canonical publication transaction, after its basis guard."""
    row = _latest(con, sid)
    if not _matches(row, request.draft_version, request.draft_hash):
        raise EditConflictError("Plan draft changed before admission")
    assert row is not None
    if json.loads(row["basis"]) != request.basis.model_dump():
        raise EditConflictError("Plan draft admission has another editing basis")
    if DesignPlan.model_validate_json(row["content"]).model_dump(mode="json") != plan.model_dump(
        mode="json"
    ):
        raise EditConflictError("Admitted plan differs from the exact current draft")


def commit_draft(
    store: Store,
    sid: str,
    owner: str,
    request: PlanDraftCommit,
    jid: str | None = None,
    author_intent: str | None = None,
    *,
    pending_contract: Contract | None = None,
) -> dict[str, Any]:
    request = PlanDraftCommit.model_validate(request.model_dump())
    with store.connect() as con:
        doc = _owned(con, sid, owner)
        store._compare_edit_basis(con, sid, doc, request.basis)
        row = _latest(con, sid)
        if not _matches(row, request.draft_version, request.draft_hash):
            raise EditConflictError("Plan draft changed; read it before admission")
        assert row is not None
        content = json.loads(row["content"])
    try:
        plan = DesignPlan.model_validate(content)
        if jid is not None and plan.operation_plan is not None:
            captured = store._edit_target_snapshot(doc, request.basis.contract_hash)
            previous_plan = (
                captured["plan"]
                if captured and captured.get("plan_hash") == request.basis.plan_hash
                else doc.get("design_plan")
            )
            admitted = DesignPlan.model_validate(previous_plan) if previous_plan else None
            old_facts = (
                {
                    fact.id: fact.model_dump(mode="json")
                    for fact in admitted.operation_plan.dimensions
                }
                if admitted and admitted.operation_plan
                else {}
            )
            for fact in plan.operation_plan.dimensions:
                if old_facts.get(fact.id) == fact.model_dump(mode="json"):
                    continue
                if fact.source == "measured_reference":
                    raise WorkspaceError("A plan draft cannot assert a measured reference")
                if fact.source == "customer" and fact.source_quote not in (
                    author_intent or doc["prompt"]
                ):
                    raise WorkspaceError(
                        "Customer fact quote must occur in the current request history"
                    )
        binding = store._set_design_plan(
            sid,
            owner,
            plan,
            request.basis.plan_version,
            pending_contract,
            request.basis.revision_id,
            request.basis.input_version,
            jid,
            request.basis,
            expected_draft=request,
        )
    except ValueError as exc:
        if isinstance(exc, EditConflictError):
            raise
        with store.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            live = _owned(con, sid, owner)
            store._compare_edit_basis(con, sid, live, request.basis)
            store._check_plan_writer(con, sid, live, jid)
            latest = _latest(con, sid)
            if not _matches(latest, request.draft_version, request.draft_hash):
                raise EditConflictError("Draft changed during rejected admission") from exc
            con.execute(
                "UPDATE plan_drafts SET validation=? WHERE session=? AND version=?",
                (
                    _diagnostics(content, str(exc)),
                    sid,
                    request.draft_version,
                ),
            )
        raise
    return {
        "action": "plan_draft_commit",
        "applied": True,
        "plan_applied": True,
        "cad_applied": False,
        "draft_version": request.draft_version,
        "plan_hash": binding["plan_hash"],
        "plan_version": binding["plan_version"],
        "edit_basis": store.edit_basis(sid, owner).model_dump(),
        "scope": "Fully validated intent admitted; no geometry or evidence was created",
    }
