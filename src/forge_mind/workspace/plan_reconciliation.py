"""Explicit dimension intent updates; alignment never proves geometry or fit."""

from __future__ import annotations

import math
from typing import Any

from pydantic import Field

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.models import Contract, DesignPlan, DimensionFact, Record
from forge_mind.workspace.operation_plan import validate_plan


class PlanReconciliation(Record):
    expected_plan_version: int = Field(ge=0, strict=True)
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    updates: list[DimensionFact] = Field(min_length=1)


def _identity(fact: DimensionFact) -> tuple[str | None, str | None, str]:
    return fact.binding_name, fact.part_id, fact.units


def _active(
    plan: DesignPlan,
) -> tuple[list[DimensionFact], dict[tuple[str | None, str | None, str], list[str]], list[str]]:
    dimensions = plan.operation_plan.dimensions if plan.operation_plan else []
    by_id = {fact.id: fact for fact in dimensions}
    invalid: set[str] = set()
    if len(by_id) != len(dimensions):
        invalid.update(fact.id for fact in dimensions)
    superseded: set[str] = set()
    checked: set[str] = set()
    for fact in dimensions:
        predecessor = fact.supersedes_fact_id
        if predecessor is not None:
            if predecessor not in by_id or _identity(by_id[predecessor]) != _identity(fact):
                invalid.add(fact.id)
            superseded.add(predecessor)
        chain: set[str] = set()
        current: str | None = fact.id
        while current is not None and current in by_id and current not in checked:
            if current in chain:
                invalid.update(chain)
                break
            chain.add(current)
            current = by_id[current].supersedes_fact_id
        checked.update(chain)
    active = [fact for fact in dimensions if fact.id not in superseded]
    groups: dict[tuple[str | None, str | None, str], list[str]] = {}
    for fact in active:
        if fact.binding_name is not None:
            groups.setdefault(_identity(fact), []).append(fact.id)
    return active, groups, sorted(invalid)


def _scope_present(fact: DimensionFact, plan: DesignPlan, document: DesignDocument) -> bool:
    if fact.part_id is None:
        return True
    if document.assembly is not None:
        return any(part.id == fact.part_id for part in document.assembly.parts)
    operations = plan.operation_plan
    parts = (
        operations.part_ids or list(dict.fromkeys(step.part_id for step in operations.steps))
        if operations
        else []
    )
    return fact.part_id == (parts[0] if len(parts) == 1 else "main")


def alignment(plan: DesignPlan, document: DesignDocument) -> dict[str, Any]:
    """Observe every active bound fact, without requiring unbound facts to bind."""
    plan = DesignPlan.model_validate(plan.model_dump())
    document = DesignDocument.model_validate(document.model_dump())
    values = document.values()
    active, groups, invalid = _active(plan)
    ambiguous = bool(invalid) or any(len(ids) > 1 for ids in groups.values())
    observations: list[dict[str, Any]] = []
    for fact in active:
        if fact.binding_name is None:
            continue
        scoped = _scope_present(fact, plan, document)
        value = values.get(fact.binding_name) if scoped else None
        active_ids = groups[_identity(fact)]
        status = (
            "ambiguous"
            if len(active_ids) > 1 or fact.id in invalid
            else "missing"
            if value is None
            else "aligned"
            if math.isfinite(value) and math.isclose(value, fact.value, rel_tol=1e-9, abs_tol=1e-9)
            else "mismatch"
        )
        observations.append(
            {
                "fact_id": fact.id,
                "binding_name": fact.binding_name,
                "part_id": fact.part_id,
                "units": fact.units,
                "declared_value": fact.value,
                "current_value": value,
                "part_scope_present": scoped,
                "status": status,
                "active_fact_ids": active_ids,
            }
        )
    return {
        "status": (
            "ambiguous"
            if ambiguous
            else "mismatch"
            if any(row["status"] != "aligned" for row in observations)
            else "aligned"
        ),
        "active_bindings": observations,
        "unbound_fact_ids": [fact.id for fact in active if fact.binding_name is None],
        "invalid_fact_ids": invalid,
        "scope": "Declared CAD inputs and evaluated relations only; not geometry or physical fit",
    }


def reconcile_plan(
    plan: DesignPlan,
    update: PlanReconciliation,
    document: DesignDocument,
    contract: Contract,
) -> DesignPlan:
    """Append explicit successors; callers own revision and version comparisons."""
    original = DesignPlan.model_validate(plan.model_dump())
    update = PlanReconciliation.model_validate(update.model_dump())
    document = DesignDocument.model_validate(document.model_dump())
    contract = Contract.model_validate(contract.model_dump())
    if original.operation_plan is None:
        raise ValueError("Dimension reconciliation requires an operation plan")
    result = original.model_copy(deep=True)
    assert result.operation_plan is not None
    by_id = {fact.id: fact for fact in result.operation_plan.dimensions}
    for fact in update.updates:
        if fact.supersedes_fact_id is None:
            raise ValueError("Dimension updates require an explicit superseding fact")
        existing = by_id.get(fact.id)
        if existing is not None:
            if existing.model_dump() != fact.model_dump():
                raise ValueError("Dimension update ID conflicts with an existing fact")
            continue
        result.operation_plan.dimensions.append(fact.model_copy(deep=True))
        by_id[fact.id] = fact
    validate_plan(result, contract, previous=original, document=document)
    report = alignment(result, document)
    if report["status"] == "ambiguous":
        raise ValueError("Dimension reconciliation has conflicting active endpoints")
    values = document.values()
    for fact in update.updates:
        if fact.binding_name not in values:
            raise ValueError("Dimension update references an unknown binding")
        if not _scope_present(fact, result, document):
            raise ValueError("Dimension update references a missing document part scope")
    return result
