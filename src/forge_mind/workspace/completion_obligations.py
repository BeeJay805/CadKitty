"""Pure physical-interface work projection; no result grants physical qualification.

The caller supplies service-verified receipts, never authored check verdicts.
This module still verifies exact requests, participants, states and proof scope.
"""

from __future__ import annotations

from typing import Any

from forge_mind.workspace.completion_bindings import PREPARED_CHECK_TOOLS
from forge_mind.workspace.completion_models import (
    FunctionalInterface,
    InterfaceDuty,
    InterfaceState,
)
from forge_mind.workspace.product_tools import _SCOPES


def _binding_diagnostic(source: str, code: str, **details: Any) -> dict[str, Any]:
    return {"binding_source": source, "code": code, **details}


def _motion_diagnostic(
    states: list[InterfaceState],
    participants: list[str],
    bindings: list[dict[str, Any]],
    evidence: dict[str, Any],
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Explain an existing failed motion branch; never supply a verdict."""
    details: dict[str, Any] = {}
    if not states:
        details["empty_state_selection"] = True
    non_motion = sorted(state.id for state in states if state.kind != "motion")
    if non_motion:
        details["non_motion_state_ids"] = non_motion
    uncovered = [
        {"state_id": state.id, "driver_values": state.driver_values}
        for state in sorted(states, key=lambda state: state.id)
        if state.kind == "motion"
        and state.driver_values not in evidence.get("requested_configurations", [])
    ]
    if uncovered:
        details["uncovered_states"] = uncovered
    unbound = sorted(set(participants) - {binding["path"] for binding in bindings})
    if unbound:
        details["unbound_participant_paths"] = unbound
    artifacts = (revision.get("measurement") or {}).get("artifact_hashes") or {}
    expected = {
        "revision_id": revision.get("id"),
        "step_sha256": artifacts.get("part.step"),
        "document_sha256": artifacts.get("document.json"),
        "engine_fingerprint": (revision.get("completion_basis") or {}).get("engine_fingerprint"),
        "coverage": "discrete_only",
        "completed_configurations": evidence.get("requested_configurations"),
    }
    failed = [key for key, value in expected.items() if evidence.get(key) != value]
    if evidence.get("errors"):
        failed.append("errors")
    if not evidence:
        failed.append("verified_evidence")
    if failed:
        details["failed_evidence_fields"] = sorted(failed)
    return _binding_diagnostic("motion_receipt", "sampled_motion_binding_incomplete", **details)


def _reachable(part: dict[str, Any]) -> set[str]:
    features = {feature["id"]: feature for feature in part.get("features", [])}
    outputs = [part["output"]] if "output" in part else part.get("outputs") or list(features)[-1:]
    pending = list(outputs)
    reached: set[str] = set()
    while pending:
        identifier = pending.pop()
        feature = features.get(identifier)
        if identifier in reached or feature is None or feature.get("suppressed"):
            continue
        reached.add(identifier)
        pending.extend(feature.get("inputs", []))
    return reached


def _legacy_bought(part: dict[str, Any]) -> bool:
    features = {feature["id"]: feature for feature in part.get("features", [])}
    outputs = [part["output"]] if "output" in part else part.get("outputs") or list(features)[-1:]
    if len(outputs) != 1:
        return False
    feature = features.get(outputs[0])
    while feature and not feature.get("suppressed") and feature.get("kind") == "transform":
        feature = features.get((feature.get("inputs") or [None])[0])
    return bool(feature and not feature.get("suppressed") and feature.get("kind") == "hardware")


def _selectors(value: Any, indices: dict[int, str]) -> set[str] | None:
    if isinstance(value, dict):
        if "occurrence_path" in value or "body_index" in value:
            path = value.get("occurrence_path")
            if path is None:
                index = value.get("body_index")
                path = indices.get(index) if type(index) is int else None
            return {path} if isinstance(path, str) else None
        result: set[str] = set()
        for child in value.values():
            found = _selectors(child, indices)
            if found is None:
                return None
            result.update(found)
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            found = _selectors(child, indices)
            if found is None:
                return None
            result.update(found)
        return result
    return set()


def _numeric_selector(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("body_index") is not None or any(
            _numeric_selector(child) for child in value.values()
        )
    return isinstance(value, list) and any(_numeric_selector(child) for child in value)


def _current_receipt(
    check: dict[str, Any], definition: dict[str, Any], revision: dict[str, Any]
) -> dict[str, Any] | None:
    receipt = check.get("verified_evidence")
    basis = revision.get("completion_basis") or {}
    if not isinstance(receipt, dict) or not basis:
        return None
    result = receipt.get("result") or {}
    request = definition["request"]
    tool = request["tool"]
    if (
        receipt.get("id") != check.get("receipt_id")
        or receipt.get("revision_id") != revision.get("id")
        or receipt.get("request") != request
        or (
            _numeric_selector(request)
            and definition.get("reference_revision_id") != revision.get("id")
        )
        or receipt.get("contract") != revision.get("contract")
        or receipt.get("stale")
        or any(
            receipt.get(key) != basis.get(key)
            for key in ("source_fingerprint", "engine_fingerprint", "artifact_hashes")
        )
        or not basis.get("artifact_hashes", {}).get("part.step")
        or result.get("status") != check.get("status")
        or result.get("status") not in {"pass", "fail"}
        or result.get("tool") != tool
        or result.get("algorithm_scope") != _SCOPES.get(tool)
    ):
        return None
    return receipt


def _moving_paths(revision: dict[str, Any], assembly: dict[str, Any]) -> set[str]:
    """Join actual moving rig/mechanism bindings to current occurrence paths."""
    document = revision.get("document") or {}
    facts = revision.get("measurement", {}).get("facts") or {}
    paths = {occurrence["path"] for occurrence in assembly.get("occurrences", [])}
    moving: set[str] = set()
    if document.get("motion_rig"):
        moving.update(
            binding["path"]
            for binding in (facts.get("motion_rig") or {}).get("bindings", [])
            if binding.get("group") and binding.get("path") in paths
        )
    mechanism = facts.get("mechanism") or {}
    artifacts = revision.get("measurement", {}).get("artifact_hashes") or {}
    if (
        not document.get("mechanism")
        or not artifacts.get("part.step")
        or not artifacts.get("document.json")
        or mechanism.get("step_sha256") != artifacts["part.step"]
        or mechanism.get("document_sha256") != artifacts["document.json"]
    ):
        return moving
    components = {
        component["id"]
        for component in (mechanism.get("graph") or {}).get("components", [])
        if component.get("grounded") is False
    }
    indices = {
        occurrence["body_index"]: occurrence["path"]
        for occurrence in assembly.get("occurrences", [])
        if type(occurrence.get("body_index")) is int
    }
    for binding in mechanism.get("bindings", []):
        index = binding.get("body_index")
        if binding.get("component") in components and type(index) is int and index in indices:
            moving.add(indices[index])
    return moving


def _connected_mates(participants: list[str], mates: list[dict[str, Any]]) -> bool:
    """One interface needs a connected graph, not just coverage of endpoint names."""
    if len(participants) < 2:
        return False
    adjacent: dict[str, set[str]] = {path: set() for path in participants}
    for mate in mates:
        a = mate.get("a", "").rpartition(".")[0]
        b = mate.get("b", "").rpartition(".")[0]
        if a not in adjacent or b not in adjacent:
            return False
        adjacent[a].add(b)
        adjacent[b].add(a)
    reached = {participants[0]}
    pending = [participants[0]]
    while pending:
        for neighbor in adjacent[pending.pop()] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    return reached == set(participants)


def _duty(
    duty: InterfaceDuty,
    interface: FunctionalInterface,
    revision: dict[str, Any],
    assembly: dict[str, Any],
    definitions: dict[str, dict[str, Any]],
    checks: dict[str, dict[str, Any]],
    motion: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": duty.id, "kind": duty.kind, "proof_scope": duty.proof_scope}
    states = {state.id: state for state in interface.states}
    selected_states = [states[identifier] for identifier in duty.state_ids]
    declared_external = bool(
        duty.missing_inputs or duty.unsupported_checks or duty.physical_validation
    )
    if duty.kind == "travel" and duty.proof_scope == "sampled_clearance":
        evidence = (motion or {}).get("verified_evidence") or {}
        requested = evidence.get("requested_configurations", [])
        bindings = (revision.get("measurement", {}).get("facts", {}).get("motion_rig") or {}).get(
            "bindings", []
        )
        moving = (
            {binding["path"] for binding in bindings if binding.get("group")}
            if revision.get("document", {}).get("motion_rig")
            else set()
        )
        if not moving.intersection(interface.participant_paths):
            return {
                **row,
                "status": "missing_construction",
                "reason": "Required moving participants have no bound motion rig",
                "binding_diagnostics": _binding_diagnostic(
                    "motion_rig",
                    "moving_participant_required",
                    expected_participant_paths=sorted(interface.participant_paths),
                    actual_moving_paths=sorted(moving),
                ),
            }
        if (
            selected_states
            and all(state.kind == "motion" for state in selected_states)
            and not evidence
        ):
            return {
                **row,
                "status": "check_pending",
                "reason": "Run the current bound sampled-motion check",
                "suggested_action": "motion_check",
                "binding_diagnostics": _binding_diagnostic(
                    "motion_receipt", "current_verified_motion_receipt_required"
                ),
            }
        if (
            not selected_states
            or any(
                state.kind != "motion" or state.driver_values not in requested
                for state in selected_states
            )
            or evidence.get("revision_id") != revision.get("id")
            or evidence.get("step_sha256")
            != revision.get("measurement", {}).get("artifact_hashes", {}).get("part.step")
            or evidence.get("document_sha256")
            != revision.get("measurement", {}).get("artifact_hashes", {}).get("document.json")
            or evidence.get("engine_fingerprint")
            != revision.get("completion_basis", {}).get("engine_fingerprint")
            or evidence.get("coverage") != "discrete_only"
            or evidence.get("requested_configurations") != evidence.get("completed_configurations")
            or evidence.get("errors")
            or set(interface.participant_paths) - {binding["path"] for binding in bindings}
            or not evidence
        ):
            return {
                **row,
                "status": "unknown",
                "reason": "No verified sampled evidence covers the declared states",
                "binding_diagnostics": _motion_diagnostic(
                    selected_states, interface.participant_paths, bindings, evidence, revision
                ),
            }
        status = (motion or {}).get("status")
        return {
            **row,
            "status": "check_failed"
            if status == "checked_fail"
            else "unknown"
            if declared_external and status == "checked"
            else "nominal_checked"
            if status == "checked"
            else "check_pending",
            "reason": "Listed-state collision evidence only; joint feasibility and holding remain unverified",
        }
    if duty.kind == "travel" and not _moving_paths(revision, assembly).intersection(
        interface.participant_paths
    ):
        return {
            **row,
            "status": "missing_construction",
            "reason": "Required participants have no current moving rig or mechanism binding",
            "binding_diagnostics": _binding_diagnostic(
                "moving_participant",
                "moving_participant_required",
                expected_participant_paths=sorted(interface.participant_paths),
                actual_moving_paths=sorted(_moving_paths(revision, assembly)),
            ),
        }
    if (
        duty.kind == "connection"
        and duty.proof_scope == "nominal_fit"
        and all(state.kind == "reference" for state in selected_states)
    ):
        mates = {mate["id"]: mate for mate in assembly.get("mates", [])}
        selected_mates = [mates.get(identifier, {}) for identifier in interface.mate_ids]
        if not selected_mates:
            return {
                **row,
                "status": "review_pending",
                "reason": "Bind a supported nominal connection check to current participants",
                "binding_diagnostics": _binding_diagnostic(
                    "assembly_mates",
                    "nominal_mates_required",
                    required_field="functional_interfaces[].mate_ids",
                    expected_participant_paths=sorted(interface.participant_paths),
                    minimum_participants=2,
                ),
            }
        if any(mate.get("status") == "fail" for mate in selected_mates):
            return {
                **row,
                "status": "check_failed",
                "reason": "A required nominal mate failed",
                "binding_diagnostics": _binding_diagnostic(
                    "assembly_mates",
                    "nominal_mate_failed",
                    failed_mate_ids=sorted(
                        identifier
                        for identifier, mate in zip(interface.mate_ids, selected_mates, strict=True)
                        if mate.get("status") == "fail"
                    ),
                ),
            }
        if any(mate.get("status") != "pass" for mate in selected_mates):
            return {
                **row,
                "status": "check_pending",
                "reason": "A required nominal mate has no supported current measurement",
                "binding_diagnostics": _binding_diagnostic(
                    "assembly_mates",
                    "current_measured_mates_required",
                    nonpassing_mate_ids=sorted(
                        identifier
                        for identifier, mate in zip(interface.mate_ids, selected_mates, strict=True)
                        if mate.get("status") != "pass"
                    ),
                ),
            }
        if not _connected_mates(interface.participant_paths, selected_mates):
            return {
                **row,
                "status": "review_pending",
                "reason": "Declared mates do not form one connected participant graph",
                "binding_diagnostics": _binding_diagnostic(
                    "assembly_mates",
                    "connected_mate_graph_required",
                    minimum_participants=2,
                    expected_participant_paths=sorted(interface.participant_paths),
                    mate_participant_pairs=[
                        {
                            "mate_id": identifier,
                            "a": mate.get("a", "").rpartition(".")[0],
                            "b": mate.get("b", "").rpartition(".")[0],
                        }
                        for identifier, mate in sorted(
                            zip(interface.mate_ids, selected_mates, strict=True)
                        )
                    ],
                    outside_participant_paths=sorted(
                        {
                            mate.get(endpoint, "").rpartition(".")[0]
                            for mate in selected_mates
                            for endpoint in ("a", "b")
                        }
                        - set(interface.participant_paths)
                    ),
                ),
            }
        return {
            **row,
            "status": "unknown" if declared_external else "nominal_checked",
            "reason": "Measured cylindrical mates only; retention and capacity remain unverified",
        }
    allowed = PREPARED_CHECK_TOOLS.get((duty.kind, duty.proof_scope), frozenset())
    if not allowed or any(state.kind != "reference" for state in selected_states):
        return {
            **row,
            "status": "unknown",
            "reason": "No current instrument supports this duty, scope and state",
            **(
                {
                    "binding_diagnostics": _binding_diagnostic(
                        "prepared_check",
                        "reference_states_required",
                        non_reference_state_ids=sorted(
                            state.id for state in selected_states if state.kind != "reference"
                        ),
                    )
                }
                if allowed
                else {}
            ),
        }
    if not duty.check_ids:
        return {
            **row,
            "status": "review_pending",
            "reason": "Declare a compatible supported geometry check; external duties remain open",
            "binding_diagnostics": _binding_diagnostic(
                "prepared_check", "prepared_checks_required", allowed_tools=sorted(allowed)
            ),
        }
    indices = {
        occurrence["body_index"]: occurrence["path"]
        for occurrence in assembly.get("occurrences", [])
        if type(occurrence.get("body_index")) is int
    }
    covered: set[str] = set()
    statuses = []
    for identifier in duty.check_ids:
        definition = definitions.get(identifier)
        if definition is None:
            return {
                **row,
                "status": "review_pending",
                "reason": "Declare the referenced prepared check",
                "binding_diagnostics": _binding_diagnostic(
                    "prepared_check", "prepared_check_not_declared", check_id=identifier
                ),
            }
        request = definition["request"]
        selected = _selectors(request, indices)
        if (
            request["tool"] not in allowed
            or not selected
            or selected - set(interface.participant_paths)
        ):
            return {
                **row,
                "status": "review_pending",
                "reason": "Prepared check scope or participants do not match the interface",
                "binding_diagnostics": _binding_diagnostic(
                    "prepared_check",
                    "prepared_tool_not_supported"
                    if request["tool"] not in allowed
                    else "prepared_selector_unresolved"
                    if selected is None
                    else "prepared_check_participants_outside_interface"
                    if selected
                    else "prepared_check_selects_no_participants",
                    check_id=identifier,
                    actual_tool=request["tool"],
                    allowed_tools=sorted(allowed),
                    expected_participant_paths=sorted(interface.participant_paths),
                    selected_participant_paths=sorted(selected) if selected is not None else None,
                    extra_participant_paths=sorted(
                        (selected or set()) - set(interface.participant_paths)
                    ),
                    unresolved_selector=selected is None,
                ),
            }
        covered.update(selected)
        check = checks.get(identifier, {})
        receipt = _current_receipt(check, definition, revision)
        if receipt is None:
            return {
                **row,
                "status": "check_pending",
                "reason": "No current verified receipt matches the prepared request",
                "binding_diagnostics": _binding_diagnostic(
                    "prepared_check", "current_verified_receipt_required", check_id=identifier
                ),
            }
        statuses.append(receipt["result"]["status"])
    if covered != set(interface.participant_paths):
        return {
            **row,
            "status": "review_pending",
            "reason": "Prepared checks do not cover all declared interface participants",
            "binding_diagnostics": _binding_diagnostic(
                "prepared_check",
                "prepared_check_participants_uncovered",
                check_ids=sorted(duty.check_ids),
                missing_participant_paths=sorted(set(interface.participant_paths) - covered),
            ),
        }
    return {
        **row,
        "status": "check_failed"
        if "fail" in statuses
        else "unknown"
        if declared_external
        else "nominal_checked",
        "reason": "Compatible current nominal geometry evidence only; declared external duties remain open",
    }


def interface_obligations(
    plan: dict[str, Any],
    revision: dict[str, Any],
    verified_subsystems: list[dict[str, Any]],
    motion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project current work from validated intent and service-verified evidence."""
    interfaces = [
        FunctionalInterface.model_validate(value) for value in plan.get("functional_interfaces", [])
    ]
    document = revision.get("document") or {}
    assembly = revision.get("measurement", {}).get("facts", {}).get("assembly") or {}
    definitions = {part["id"]: part for part in (document.get("assembly") or {}).get("parts", [])}
    occurrences = {row["path"]: row for row in assembly.get("occurrences", [])}
    if (
        document
        and not document.get("assembly")
        and revision.get("measurement", {}).get("geometry_valid")
    ):
        definitions, occurrences = (
            {"part": document},
            {"part": {"path": "part", "definition": "part"}},
        )
    if not revision.get("measurement", {}).get("geometry_valid"):
        occurrences = {}
    prepared = {
        check["id"]: check
        for subsystem in plan.get("subsystems", [])
        for check in subsystem.get("checks", [])
    }
    checks = {
        check["id"]: check
        for subsystem in verified_subsystems
        for check in subsystem.get("checks", [])
    }
    rows: list[dict[str, Any]] = []
    actions: list[tuple[int, str, str, str]] = []
    external: list[dict[str, Any]] = []
    for interface in interfaces:
        missing = [path for path in interface.participant_paths if path not in occurrences]
        inventory = []
        for path in interface.installed_occurrence_paths:
            part = definitions.get(occurrences.get(path, {}).get("definition"), {})
            role = (part.get("production") or {}).get("role", "unspecified")
            legacy_buy = _legacy_bought(part)
            if path not in occurrences or (
                role not in {"make", "buy"} and not (role == "unspecified" and legacy_buy)
            ):
                inventory.append(path)
        missing_features = [
            feature.model_dump()
            for feature in interface.required_features
            if feature.feature_id
            not in _reachable(
                definitions.get(occurrences.get(feature.occurrence_path, {}).get("definition"), {})
            )
        ]
        ports = assembly.get("ports", {})
        mates = {mate["id"]: mate for mate in assembly.get("mates", [])}
        missing_ports = [identifier for identifier in interface.port_ids if identifier not in ports]
        missing_mates = [identifier for identifier in interface.mate_ids if identifier not in mates]
        failed_interfaces = [
            identifier
            for identifier in interface.port_ids
            if ports.get(identifier, {}).get("status") == "fail"
        ]
        pending_interfaces = [
            identifier
            for identifier in interface.port_ids
            if identifier in ports and ports[identifier].get("status") not in {"pass", "fail"}
        ]
        for identifier in interface.mate_ids:
            mate = mates.get(identifier, {})
            if mate and {
                mate.get("a", "").rpartition(".")[0],
                mate.get("b", "").rpartition(".")[0],
            } - set(interface.participant_paths):
                missing_mates.append(identifier)
            elif mate.get("status") == "fail":
                failed_interfaces.append(identifier)
            elif identifier in mates and mate.get("status") != "pass":
                pending_interfaces.append(identifier)
        duties = [
            _duty(duty, interface, revision, assembly, prepared, checks, motion)
            for duty in interface.duties
        ]
        status = (
            "missing_construction"
            if missing
            or inventory
            or missing_features
            or missing_ports
            or missing_mates
            or any(duty["status"] == "missing_construction" for duty in duties)
            else "check_failed"
            if failed_interfaces or any(duty["status"] == "check_failed" for duty in duties)
            else "check_pending"
            if pending_interfaces or any(duty["status"] == "check_pending" for duty in duties)
            else "review_pending"
            if any(duty["status"] == "review_pending" for duty in duties)
            else "nominal_checked"
            if all(duty["status"] == "nominal_checked" for duty in duties)
            else "unknown"
        )
        row = {
            "id": interface.id,
            "requirement_ids": interface.requirement_ids,
            "status": status,
            "missing_participant_paths": missing,
            "missing_installed_occurrence_paths": inventory,
            "missing_features": missing_features,
            "missing_port_ids": missing_ports,
            "missing_mate_ids": missing_mates,
            "failed_interface_ids": failed_interfaces,
            "pending_interface_ids": pending_interfaces,
            "duties": duties,
            "physical_readiness": "unverified",
        }
        rows.append(row)
        if status in {"missing_construction", "check_failed", "check_pending", "review_pending"}:
            priority, category = {
                "check_failed": (0, "repair"),
                "missing_construction": (1, "construction"),
                "check_pending": (2, "check"),
                "review_pending": (3, "review"),
            }[status]
            actions.append(
                (
                    priority,
                    interface.id,
                    category,
                    "Resolve current physical-interface " + status.replace("_", " "),
                )
            )
        for duty, outcome in zip(interface.duties, duties, strict=True):
            if (
                outcome["status"] == "unknown"
                or duty.missing_inputs
                or duty.unsupported_checks
                or duty.physical_validation
            ):
                external.append(
                    {
                        "interface_id": interface.id,
                        "duty_id": duty.id,
                        "requirement_ids": interface.requirement_ids,
                        "reason": outcome["reason"],
                        "missing_inputs": duty.missing_inputs,
                        "unsupported_checks": duty.unsupported_checks,
                        "physical_validation": duty.physical_validation,
                    }
                )
    selected = min(actions, default=None, key=lambda value: value[0])
    result: dict[str, Any] = {
        "rows": rows,
        "status": "not_declared"
        if not rows
        else "needs_work"
        if selected
        else "nominal_checked"
        if all(row["status"] == "nominal_checked" for row in rows)
        else "unverified",
        "next_target": selected[1] if selected else None,
        "next_work_category": selected[2] if selected else None,
        "next_work_reason": selected[3] if selected else None,
        "external_obligations": external,
        "physical_readiness": "unverified",
        "scope": "Saved construction and compatible nominal evidence only; no frozen acceptance or physical qualification",
    }
    if selected and selected[2] == "check":
        chosen = next(row for row in rows if row["id"] == selected[1])
        if any(duty.get("suggested_action") == "motion_check" for duty in chosen["duties"]):
            result["suggested_action"] = "motion_check"
    return result
