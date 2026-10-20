"""Small generic authoring transactions; design intent never constitutes evidence."""

from __future__ import annotations

import heapq
import math
from typing import Any

from pydantic import BaseModel, Field

from forge_mind.workspace.document import (
    AssemblySpec,
    DesignDocument,
    Feature,
    InterfaceMate,
    Name,
    Occurrence,
    PartDefinition,
    Subassembly,
    VariableRelation,
)
from forge_mind.workspace.models import DesignPlan, Record
from forge_mind.workspace.motion_rig import MotionRig


class AssemblyEdit(Record):
    motion_rig: MotionRig | None = None
    name: str = Field(default="New assembly", max_length=120)
    variables: dict[Name, float] = Field(default_factory=dict)
    relations: list[VariableRelation] = Field(default_factory=list)
    promote_relations: dict[Name, float] = Field(default_factory=dict)
    derive_variables: list[VariableRelation] = Field(default_factory=list)
    parts: list[PartDefinition] = Field(default_factory=list)
    subassemblies: list[Subassembly] = Field(default_factory=list)
    roots: list[Occurrence] = Field(default_factory=list)
    mates: list[InterfaceMate] = Field(default_factory=list)


def merge_named(old: list[Any], changes: list[Any], key: str = "id") -> list[Any]:
    if len({getattr(v, key) for v in changes}) != len(changes):
        raise ValueError("Duplicate identifiers in assembly edit")
    merged = {getattr(v, key): v for v in old}
    changes = [
        value.model_copy(update={"production": merged[getattr(value, key)].production})
        if isinstance(value, PartDefinition)
        and "production" not in value.model_fields_set
        and getattr(value, key) in merged
        else value
        for value in changes
    ]
    merged.update({getattr(v, key): v for v in changes})
    return list(merged.values())


def _order_relations(
    relations: list[VariableRelation], variables: dict[str, float]
) -> list[VariableRelation]:
    """Order merged arithmetic dependencies, retaining valid input order exactly."""
    indices = {relation.name: index for index, relation in enumerate(relations)}
    if len(indices) != len(relations):
        raise ValueError("Duplicate relation names in assembly edit")
    dependents: list[list[int]] = [[] for _ in relations]
    pending = [0] * len(relations)
    for index, relation in enumerate(relations):
        if relation.name in variables:
            raise ValueError(f"Relation {relation.name} cannot overwrite a variable")
        dependencies = {value for value in (relation.a, relation.b) if isinstance(value, str)}
        for dependency in sorted(dependencies):
            if dependency in variables:
                continue
            if dependency not in indices:
                raise ValueError(f"Relation {relation.name} references missing value {dependency}")
            pending[index] += 1
            dependents[indices[dependency]].append(index)
    ready = [index for index, count in enumerate(pending) if count == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        index = heapq.heappop(ready)
        ordered.append(relations[index])
        for dependent in dependents[index]:
            pending[dependent] -= 1
            if pending[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(ordered) != len(relations):
        blocked = ", ".join(relations[index].name for index, count in enumerate(pending) if count)
        raise ValueError(f"Cyclic relation dependencies block: {blocked}")
    return ordered


def apply_assembly_edit(document: DesignDocument | None, edit: AssemblyEdit) -> DesignDocument:
    if document is not None and not document.assembly:
        raise ValueError("Assembly edit requires an assembly or a new project")
    promoted = set(edit.promote_relations)
    derived = {relation.name for relation in edit.derive_variables}
    if len(derived) != len(edit.derive_variables):
        raise ValueError("Duplicate derived variable transitions in assembly edit")
    if promoted & derived:
        raise ValueError("A binding cannot be both promoted and derived in one assembly edit")
    ordinary = set(edit.variables) | {relation.name for relation in edit.relations}
    if (promoted | derived) & ordinary:
        raise ValueError("Binding transitions cannot overlap ordinary variable or relation edits")
    original_variables = document.variables if document else {}
    original_relations = document.relations if document else []
    relation_names = {relation.name for relation in original_relations}
    for name, value in edit.promote_relations.items():
        if name not in relation_names or name in original_variables:
            raise ValueError(f"Promoted binding {name} must be an existing derived relation")
        if not math.isfinite(value):
            raise ValueError(f"Promoted binding {name} must have a finite value")
    for name in sorted(derived):
        if name not in original_variables or name in relation_names:
            raise ValueError(f"Derived binding {name} must be an existing input variable")
    previous = document.assembly if document else None
    spec = AssemblySpec(
        **{
            key: merge_named(getattr(previous, key) if previous else [], getattr(edit, key))
            for key in ("parts", "subassemblies", "roots", "mates")
        }
    )
    data: dict[str, Any] = document.model_dump() if document else {"name": edit.name}
    variables = {
        **{name: value for name, value in original_variables.items() if name not in derived},
        **edit.variables,
        **edit.promote_relations,
    }
    relations = merge_named(
        [relation for relation in original_relations if relation.name not in promoted],
        [*edit.relations, *edit.derive_variables],
        "name",
    )
    data.update(
        assembly=spec.model_dump(),
        variables=variables,
        relations=[v.model_dump() for v in _order_relations(relations, variables)],
    )
    if edit.motion_rig is not None:
        data["motion_rig"] = edit.motion_rig.model_dump()
    result = DesignDocument.model_validate(data)
    if document:
        from forge_mind.workspace.assembly import preserved_assembly

        retained, reason = preserved_assembly(document.model_dump(), result.model_dump())
        if not retained:
            raise ValueError(reason)
    return result


def authoring_schema(query: str) -> dict[str, Any]:
    if query == "photo_profile":
        from forge_mind.workspace.photo_profile_api import PhotoProfileAction

        return {
            "name": query,
            "schema": PhotoProfileAction.model_json_schema(),
            "scope": "Derived photo observations only; never owner facts or acceptance.",
        }
    from forge_mind.workspace.action_payloads import ReadDesignArguments
    from forge_mind.workspace.delivery_guidance import GuidanceUpdate
    from forge_mind.workspace.feature_delta import FeatureDelta
    from forge_mind.workspace.intent_attribution import (
        IntentAttribution,
        IntentConflict,
        IntentResolution,
    )
    from forge_mind.workspace.plan_draft import PlanDraftCommit, PlanDraftPatch, PlanDraftRead
    from forge_mind.workspace.plan_intent import PlanIntent
    from forge_mind.workspace.plan_patch import PlanPatch
    from forge_mind.workspace.plan_reconciliation import PlanReconciliation

    if query == "construction" or query.startswith("construction:"):
        from forge_mind.workspace.construction_reference import construction_reference

        return construction_reference(query.partition(":")[2])
    if query.startswith("feature:"):
        from forge_mind.workspace.construction_reference import feature_reference

        return feature_reference(query.partition(":")[2])
    if query == "python":
        from forge_mind.agentic.cheatsheet import author_skills_block

        return {"name": "python", "construction_reference": author_skills_block()}
    models: dict[str, type[BaseModel]] = {
        "document": DesignDocument,
        "assembly_edit": AssemblyEdit,
        "part": PartDefinition,
        "feature": Feature,
        "plan": DesignPlan,
        "motion_rig": MotionRig,
        "guidance": GuidanceUpdate,
        "plan_reconciliation": PlanReconciliation,
        "read_design": ReadDesignArguments,
        "feature_delta": FeatureDelta,
        "plan_patch": PlanPatch,
        "plan_intent": PlanIntent,
        "intent_attribution": IntentAttribution,
        "intent_conflict": IntentConflict,
        "intent_resolution": IntentResolution,
        "plan_draft_read": PlanDraftRead,
        "plan_draft_patch": PlanDraftPatch,
        "plan_draft_commit": PlanDraftCommit,
    }
    if query not in models:
        return {
            "available": [*models, "python", "construction"],
            "usage": "Query a listed schema name, construction:<recipe_id>, or feature:<kind>. Start with construction for a compact capability index.",
        }
    return {
        "name": query,
        "schema": models[query].model_json_schema(),
        **(
            {
                "native_construction": "Optional NativeAction.plan_patch_json carries this complete PlanPatch on document, assembly_edit or feature_edit. Use one current EditBasis for the patch and CAD edit. Both publish together; other actions cannot carry this field.",
                "criterion_amendment": "When changing a saved check request, copy previous_request_hash from completion.subsystems[].checks[].request_hash and supply explicit reason and basis. Old failed predicates remain historical; frozen obligations remain unchanged.",
            }
            if query == "plan_patch"
            else {}
        ),
        "conventions": "Millimeters. Box and cylinder centered at origin. Cylinder axis Z. Occurrence rotation is XYZ Euler degrees, then position. Assembly has parts/subassemblies/roots/mates, no top-level features or outputs. Each part output must be exactly one solid. A detailed hardware catalog shape may contain multiple solids. Hardware envelope:1 supports bearings as a single ring, with local Z from 0 to width (not centered). Hardware envelope:1 also supports socket-head, button-head and hex-head screws with nominal thread crest cylinders, retaining catalog head/origin. This measures nominal clearance, not helical engagement. Without envelope:1 simple catalog screw threads use minor diameter. Inspect dimensions before placing them. Relations support add/subtract/multiply/divide/min/max, sin_deg/cos_deg/tan_deg, atan2_deg and sqrt. Unary relations use a and omitted or zero b. Angle inputs and outputs are degrees. Domain errors and tangent singularities are rejected. Feature bindings reference variables or derived relation names, not expressions. AssemblyEdit retains omitted bindings. To intentionally replace an existing relation with an input, use promote_relations={name: finite_value}; to replace an existing input with a relation, use derive_variables=[VariableRelation]. Both transitions are atomic and keep the binding name. Their names cannot overlap variables, relations or each other. Targets must have the opposite category in the original document. Dependencies are validated after all transitions; feature bindings and omitted records remain intact. No inverse formula is inferred.",
        "scope": "Schema and construction syntax, not mechanical design or validation evidence",
        "construction_query": "construction",
        "feature_parameters_query": "feature:<kind>",
    }
