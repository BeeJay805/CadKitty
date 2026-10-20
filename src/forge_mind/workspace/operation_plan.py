"""Construction intent and immutable submission identity, never geometric evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from typing import Any

from forge_mind.workspace.models import Contract, DesignPlan


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _plan(value: Any) -> DesignPlan:
    return DesignPlan.model_validate(value.model_dump() if isinstance(value, DesignPlan) else value)


def plan_semantic_conflicts(plan: Any) -> list[dict[str, Any]]:
    """Supplement typed errors with safely detectable conflicts in decoded plan JSON.

    Paths are JSON pointers. Invalid field types are left to model validation;
    absence is reported only when the referenced collection can be read safely.
    This inspection neither changes input nor establishes admission or evidence.
    """
    value = plan.model_dump(mode="json") if isinstance(plan, DesignPlan) else plan
    if not isinstance(value, dict):
        return []
    conflicts: list[dict[str, Any]] = []

    def report(path: str, record: dict[str, Any], code: str, ids: list[str]) -> None:
        identifier = record.get("id")
        conflicts.append(
            {
                "field_path": path,
                "record_id": identifier if isinstance(identifier, str) else path,
                "code": code,
                "conflicting_ids": sorted(set(ids)),
            }
        )

    def strings(raw: Any) -> list[str] | None:
        return raw if isinstance(raw, list) and all(isinstance(v, str) for v in raw) else None

    def records(raw: Any) -> list[dict[str, Any]] | None:
        return raw if isinstance(raw, list) and all(isinstance(v, dict) for v in raw) else None

    def duplicates(raw: list[str], path: str, record: dict[str, Any]) -> None:
        seen: set[str] = set()
        repeated: set[str] = set()
        for identifier in raw:
            if identifier in seen:
                repeated.add(identifier)
            seen.add(identifier)
        if repeated:
            report(path, record, "duplicate_reference", list(repeated))

    operations = value.get("operation_plan")
    if isinstance(operations, dict):
        frames = records(operations.get("frames", []))
        if frames is not None and all(isinstance(frame.get("id"), str) for frame in frames):
            frame_ids = [frame["id"] for frame in frames]
            frame_counts = Counter(frame_ids)
            duplicates(frame_ids, "/operation_plan/frames", operations)
            frame_by_id = {frame["id"]: frame for frame in frames}
            for collection in ("steps", "dimensions"):
                for index, record in enumerate(records(operations.get(collection, [])) or []):
                    frame_id = record.get("frame_id")
                    if not isinstance(frame_id, str):
                        continue
                    path = f"/operation_plan/{collection}/{index}/frame_id"
                    frame = frame_by_id.get(frame_id)
                    if frame is None:
                        report(path, record, "unknown_frame", [frame_id])
                        continue
                    if frame_counts[frame_id] != 1:
                        continue  # Duplicate frame IDs make ownership ambiguous.
                    owner, part = frame.get("part_id"), record.get("part_id")
                    if owner is not None and not isinstance(owner, str):
                        continue
                    if part is not None and not isinstance(part, str):
                        continue
                    if collection == "steps" and part is None:
                        continue  # Missing required part scope is a typed error.
                    if owner is not None and owner != part:
                        report(
                            path,
                            record,
                            "frame_owner_mismatch",
                            [frame_id, owner, *([part] if part is not None else [])],
                        )

    interfaces = records(value.get("functional_interfaces", []))
    if interfaces is not None and all(isinstance(row.get("id"), str) for row in interfaces):
        duplicates([row["id"] for row in interfaces], "/functional_interfaces", value)
    for index, interface in enumerate(interfaces or []):
        base = f"/functional_interfaces/{index}"
        participants = strings(interface.get("participant_paths"))
        for field in (
            "requirement_ids",
            "participant_paths",
            "installed_occurrence_paths",
            "port_ids",
            "mate_ids",
        ):
            references = strings(interface.get(field, []))
            if references is not None:
                duplicates(references, f"{base}/{field}", interface)
        installed = strings(interface.get("installed_occurrence_paths", []))
        if participants is not None and installed is not None:
            missing = sorted(set(installed) - set(participants))
            if missing:
                report(
                    f"{base}/installed_occurrence_paths",
                    interface,
                    "installed_occurrence_not_participant",
                    missing,
                )
        features = records(interface.get("required_features", []))
        feature_keys: set[tuple[str, str]] = set()
        for feature_index, feature in enumerate(features or []):
            owner, feature_id = feature.get("occurrence_path"), feature.get("feature_id")
            if not isinstance(owner, str) or not isinstance(feature_id, str):
                continue
            path = f"{base}/required_features/{feature_index}"
            if (owner, feature_id) in feature_keys:
                report(path, interface, "duplicate_reference", [owner, feature_id])
            feature_keys.add((owner, feature_id))
            if participants is not None and owner not in participants:
                report(
                    f"{path}/occurrence_path",
                    interface,
                    "feature_owner_not_participant",
                    [owner, feature_id],
                )
        ports = strings(interface.get("port_ids", []))
        if participants is not None and ports is not None:
            for port_index, port in enumerate(ports):
                owner = port.rpartition(".")[0]
                if owner not in participants:
                    report(
                        f"{base}/port_ids/{port_index}",
                        interface,
                        "port_owner_not_participant",
                        [port, owner],
                    )
        states = records(interface.get("states", []))
        state_ids = (
            [state["id"] for state in states]
            if states is not None and all(isinstance(state.get("id"), str) for state in states)
            else None
        )
        if state_ids is not None:
            duplicates(state_ids, f"{base}/states", interface)
        for state_index, state in enumerate(states or []):
            kind, drivers = state.get("kind", "reference"), state.get("driver_values", {})
            if kind not in ("reference", "motion") or not isinstance(drivers, dict):
                continue
            if (kind == "motion") != bool(drivers):
                driver_ids = [name for name in drivers if isinstance(name, str)]
                report(
                    f"{base}/states/{state_index}/driver_values",
                    state,
                    "state_driver_mismatch",
                    driver_ids,
                )
        duties = records(interface.get("duties", []))
        if duties is not None and all(isinstance(duty.get("id"), str) for duty in duties):
            duplicates([duty["id"] for duty in duties], f"{base}/duties", interface)
        for duty_index, duty in enumerate(duties or []):
            for field in ("check_ids", "state_ids"):
                references = strings(duty.get(field, []))
                if references is None:
                    continue
                path = f"{base}/duties/{duty_index}/{field}"
                duplicates(references, path, duty)
                if field == "state_ids" and state_ids is not None:
                    missing = sorted(set(references) - set(state_ids))
                    if missing:
                        report(path, duty, "unknown_duty_state", missing)
    return conflicts


def validate_plan(plan: Any, contract: Any, previous: Any = None, document: Any = None) -> None:
    """Retain existing completion checks and validate construction references."""
    from forge_mind.workspace.completion import validate_plan as validate_completion_plan

    plan = _plan(plan)
    contract = Contract.model_validate(
        contract.model_dump() if isinstance(contract, Contract) else contract
    )
    validate_completion_plan(plan, contract, previous)
    from forge_mind.workspace.verification_admission import validate_verification_bindings

    validate_verification_bindings(plan, contract, document)
    if previous:
        current_subjects = {binding.subject_id for binding in plan.verification_bindings}
        if any(
            binding.subject_id not in current_subjects
            for binding in _plan(previous).verification_bindings
        ):
            raise ValueError("Retain existing verification subject bindings")
    operations = plan.operation_plan
    if operations is None:
        return
    requirements = {r.id: r for r in contract.requirements}
    steps = {step.id: step for step in operations.steps}
    frames = {frame.id: frame for frame in operations.frames}
    if len(steps) != len(operations.steps) or len(frames) != len(operations.frames):
        raise ValueError("Operation step and frame IDs must be unique")
    if len({fact.id for fact in operations.dimensions}) != len(operations.dimensions):
        raise ValueError("Dimension fact IDs must be unique")
    facts = {fact.id: fact for fact in operations.dimensions}
    for fact in operations.dimensions:
        if fact.supersedes_fact_id is None:
            continue
        prior = facts.get(fact.supersedes_fact_id)
        if prior is None:
            raise ValueError("Dimension fact supersedes an unknown fact")
        if (
            fact.binding_name is None
            or fact.binding_name != prior.binding_name
            or fact.part_id != prior.part_id
            or fact.units != prior.units
        ):
            raise ValueError("Superseding dimension facts must retain binding, part and units")
    checked: set[str] = set()
    for fact in operations.dimensions:
        chain: set[str] = set()
        current: str | None = fact.id
        while current is not None and current not in checked:
            if current in chain:
                raise ValueError("Dimension fact supersession cycle")
            chain.add(current)
            current = facts[current].supersedes_fact_id
        checked.update(chain)
    if len(set(operations.part_ids)) != len(operations.part_ids):
        raise ValueError("Operation part IDs must be unique")
    parts = set(operations.part_ids)
    features: set[tuple[str, str]] = set()
    for item in [*operations.steps, *operations.dimensions, *operations.frames]:
        if parts and item.part_id is not None and item.part_id not in parts:
            raise ValueError("Operation record references an unknown part")
    for item in [*operations.steps, *operations.dimensions]:
        if set(item.requirement_ids) - requirements.keys():
            raise ValueError("Operation record references an unknown frozen requirement")
        if len(item.requirement_ids) != len(set(item.requirement_ids)):
            raise ValueError("Operation requirement references must be unique")
        if item.frame_id is not None:
            if item.frame_id not in frames:
                raise ValueError("Operation record references an unknown frame")
            if frames[item.frame_id].part_id not in (None, item.part_id):
                raise ValueError("Operation frame belongs to a different part")
        for rid in item.requirement_ids:
            if requirements[rid].part_id not in (None, item.part_id):
                raise ValueError("Operation requirement belongs to a different part")
    dependents: dict[str, list[str]] = {sid: [] for sid in steps}
    counts = {sid: len(step.depends_on) for sid, step in steps.items()}
    for step in operations.steps:
        if step.kind == "parameters" and not step.binding_names:
            raise ValueError("Parameter steps require named bindings")
        for values in (
            step.depends_on,
            step.feature_ids,
            step.referenced_feature_ids,
            step.binding_names,
        ):
            if len(values) != len(set(values)):
                raise ValueError("Operation references must be unique")
        if step.kind != "construction" and step.feature_ids:
            raise ValueError("Only construction steps can own Features; use referenced_feature_ids")
        if set(step.depends_on) - steps.keys():
            raise ValueError("Operation step references an unknown dependency")
        for dependency in step.depends_on:
            dependents[dependency].append(step.id)
        for fid in step.feature_ids:
            key = (step.part_id, fid)
            if key in features:
                raise ValueError("A planned Feature belongs to more than one operation step")
            features.add(key)
    ready = deque(sid for sid, count in counts.items() if count == 0)
    visited = 0
    while ready:
        visited += 1
        for dependent in dependents[ready.popleft()]:
            counts[dependent] -= 1
            if counts[dependent] == 0:
                ready.append(dependent)
    if visited != len(steps):
        raise ValueError("Operation dependency cycle")
    if previous:
        old = _plan(previous).operation_plan
        if old:
            if {s.id for s in old.steps} - steps.keys():
                raise ValueError("Retain existing operation step IDs within one frozen contract")
            for step in old.steps:
                if step.part_id != steps[step.id].part_id:
                    raise ValueError("A stable operation step cannot change part scope")
                if step.feature_ids and step.kind != steps[step.id].kind:
                    raise ValueError("An operation step with owned Features cannot change kind")
            for fact in old.dimensions:
                if fact.id not in facts or fact.model_dump() != facts[fact.id].model_dump():
                    raise ValueError(
                        "Retain original dimension facts; append explicit superseding facts"
                    )
    if document is not None:
        coverage(snapshot(plan, 0, contract), document)


def reference_scope_warnings(plan: Any) -> list[dict[str, Any]]:
    """Flag possible scope mistakes without inferring absence of local Features."""
    operations = _plan(plan).operation_plan
    if operations is None:
        return []
    owners: dict[str, set[str]] = {}
    for step in operations.steps:
        for feature_id in step.feature_ids:
            owners.setdefault(feature_id, set()).add(step.part_id)
    return [
        {
            "step_id": step.id,
            "part_id": step.part_id,
            "referenced_feature_id": feature_id,
            "declared_part_ids": sorted(owners[feature_id]),
            "status": "possible_scope_mismatch",
            "message": (
                "References are local to the step's part. This Feature has construction "
                "owners only in other parts. Check the scope or declare its local construction. "
                "An unowned local Feature may exist; saved-document coverage decides validity."
            ),
            "scope": "Plan declarations only; not evidence of absent geometry or a blocking failure",
        }
        for step in operations.steps
        for feature_id in step.referenced_feature_ids
        if feature_id in owners and step.part_id not in owners[feature_id]
    ]


def snapshot(plan: Any, version: int, contract: Any) -> dict[str, Any]:
    """Copy default-expanded JSON so later author edits cannot change its identity."""
    plan_value = _plan(plan).model_dump(mode="json")
    contract_value = Contract.model_validate(
        contract.model_dump() if isinstance(contract, Contract) else contract
    ).model_dump(mode="json")
    return {
        "plan": plan_value,
        "plan_version": version,
        "plan_hash": _hash(plan_value),
        "contract": contract_value,
        "contract_hash": _hash(contract_value),
    }


def coverage(binding: dict[str, Any], document: Any) -> dict[str, Any]:
    """Report scoped graph presence; missing future steps are explicitly incomplete."""
    from forge_mind.workspace.document import DesignDocument, digest

    plan = _plan(binding["plan"])
    if binding["plan_hash"] != _hash(plan.model_dump(mode="json")):
        raise ValueError("Plan snapshot hash does not match its JSON")
    if binding["contract_hash"] != _hash(binding["contract"]):
        raise ValueError("Plan snapshot contract hash does not match its JSON")
    validate_plan(plan, binding["contract"])
    doc = DesignDocument.model_validate(
        document.model_dump() if isinstance(document, DesignDocument) else document
    )
    operations = plan.operation_plan
    planned_parts = (
        operations.part_ids or list(dict.fromkeys(s.part_id for s in operations.steps))
        if operations
        else []
    )
    single_part = planned_parts[0] if len(planned_parts) == 1 else "main"
    scopes = (
        {part.id: {f.id for f in part.features if not f.suppressed} for part in doc.assembly.parts}
        if doc.assembly
        else {single_part: {f.id for f in doc.features if not f.suppressed}}
    )
    values = doc.values()
    results: list[dict[str, Any]] = []
    dimension_bindings: list[dict[str, Any]] = []
    for step in operations.steps if operations else []:
        scoped = scopes.get(step.part_id, set())
        missing = sorted(set(step.feature_ids) - scoped)
        missing_references = sorted(set(step.referenced_feature_ids) - scoped)
        wrong = [
            fid
            for fid in [*missing, *missing_references]
            if step.part_id in scopes and any(fid in ids for ids in scopes.values())
        ]
        if wrong:
            raise ValueError(f"Operation step {step.id} Feature is in a different part: {wrong}")
        unknown_bindings = sorted(set(step.binding_names) - values.keys())
        if unknown_bindings and set(step.feature_ids) & scoped:
            raise ValueError(
                f"Operation step {step.id} references unknown bindings: {unknown_bindings}"
            )
        results.append(
            {
                "step_id": step.id,
                "part_id": step.part_id,
                "kind": step.kind,
                "feature_ids_present": sorted(set(step.feature_ids) & scoped),
                "missing_feature_ids": missing,
                "referenced_feature_ids_present": sorted(set(step.referenced_feature_ids) & scoped),
                "missing_referenced_feature_ids": missing_references,
                "references_present": not missing_references,
                "parameters_present": step.kind == "parameters"
                and bool(step.binding_names)
                and not unknown_bindings,
                "requirement_ids": step.requirement_ids,
                "graph_present": (
                    step.kind == "construction" and bool(step.feature_ids) and not missing
                ),
                "effect_verified": False,
                "status": "pending",
            }
        )
    if operations:
        present_bindings = {
            name
            for step, result in zip(operations.steps, results, strict=True)
            if result["feature_ids_present"]
            for name in step.binding_names
        }
        for fact in operations.dimensions:
            if fact.binding_name in present_bindings and fact.binding_name not in values:
                raise ValueError(f"Dimension fact {fact.id} references an unknown binding")
            current_value = values.get(fact.binding_name) if fact.binding_name is not None else None
            status = (
                "unbound"
                if fact.binding_name is None
                else "missing"
                if current_value is None
                else "unchanged"
                if current_value == fact.value
                else "changed"
            )
            dimension_bindings.append(
                {
                    "fact_id": fact.id,
                    "part_id": fact.part_id,
                    "units": fact.units,
                    "binding_name": fact.binding_name,
                    "declared_value": fact.value,
                    "current_value": current_value,
                    "delta": current_value - fact.value if current_value is not None else None,
                    "status": status,
                    "superseded": False,
                    "superseded_by_fact_ids": [],
                }
            )
        observations = {row["fact_id"]: row for row in dimension_bindings}
        for fact in operations.dimensions:
            if fact.supersedes_fact_id is not None:
                prior_observation = observations[fact.supersedes_fact_id]
                prior_observation["superseded"] = True
                prior_observation["superseded_by_fact_ids"].append(fact.id)
        by_id = {r["step_id"]: r for r in results}
        remaining = {s.id: len(s.depends_on) for s in operations.steps}
        dependents: dict[str, list[str]] = {s.id: [] for s in operations.steps}
        step_by_id = {s.id: s for s in operations.steps}
        for step in operations.steps:
            for dependency in step.depends_on:
                if dependency not in dependents:
                    raise ValueError("Operation step references an unknown dependency")
                dependents[dependency].append(step.id)
        ready = deque(sid for sid, count in remaining.items() if not count)
        visited = 0
        while ready:
            sid = ready.popleft()
            visited += 1
            result = by_id[sid]
            result["missing_dependency_steps"] = [
                dependency
                for dependency in step_by_id[sid].depends_on
                if not (
                    by_id[dependency]["graph_present"]
                    if step_by_id[dependency].kind == "construction"
                    else by_id[dependency]["parameters_present"]
                    if step_by_id[dependency].kind == "parameters"
                    else by_id[dependency]["references_present"]
                )
            ]
            if result["missing_dependency_steps"]:
                result["graph_present"] = False
                result["references_present"] = False
                result["parameters_present"] = False
            for dependent in dependents[sid]:
                remaining[dependent] -= 1
                if not remaining[dependent]:
                    ready.append(dependent)
        if visited != len(results):
            raise ValueError("Operation dependency cycle")
    construction_results = [row for row in results if row["kind"] == "construction"]
    return {
        "plan_hash": binding["plan_hash"],
        "plan_version": binding["plan_version"],
        "contract_hash": binding["contract_hash"],
        "document_hash": digest(doc.model_dump()),
        "steps": results,
        "dimension_bindings": dimension_bindings,
        "dimension_binding_scope": (
            "Source document parameter values only; not geometry measurements or customer approval"
        ),
        "graph_complete": bool(construction_results)
        and all(r["graph_present"] for r in construction_results),
        "references_complete": bool(results) and all(r["references_present"] for r in results),
        "parameters_complete": all(
            r["parameters_present"] for r in results if r["kind"] == "parameters"
        ),
        "scope": "Construction graph presence only; intended effects require independent inspection",
    }
