"""Resolve inspection intent against exact existing receipts without new verdicts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.inspection_models import InspectionCharacteristic

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def dimension_points(tool: str, values: dict[str, Any]) -> list[list[float]] | None:
    """Endpoints on the instrument's datum axis, not arbitrary surface anchors."""
    try:
        if tool == "planar_face_pair":
            points = [values[side]["plane_dimension_point_mm"] for side in ("a", "b")]
        else:
            axis, datum = values["axis"], values["datum"]
            if len(axis) != 3 or len(datum) != 3 or abs(sum(v * v for v in axis) - 1) > 1e-7:
                return None
            coordinates = [values["a"]["interval_mm"][1], values["b"]["interval_mm"][0]]
            points = [
                [d + a * coordinate for d, a in zip(datum, axis, strict=True)]
                for coordinate in coordinates
            ]
        if len(points) != 2 or any(len(p) != 3 for p in points):
            return None
        if not all(type(v) in (int, float) and math.isfinite(v) for p in points for v in p):
            return None
        return [[float(v) for v in p] for p in points]
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return None


def measured_pair_identity(
    request: dict[str, Any], values: dict[str, Any], revision: dict[str, Any]
) -> tuple[list[str], list[int]]:
    """Cross-check ordered instrument identities against the current exported assembly."""
    from forge_mind.workspace.product_tools import FaceSelector, Selector

    facts = revision["measurement"].get("facts", {})
    occurrences = (facts.get("assembly") or {}).get("occurrences", [])
    paths: list[str] = []
    indices: list[int] = []
    for side in ("a", "b"):
        result_side = values.get(side)
        if not isinstance(result_side, dict):
            raise ValueError("Measured pair lacks selected-body identity")
        index = result_side.get("body_index")
        if type(index) is not int or index < 0:
            raise ValueError("Measured pair has invalid body index")
        if request["tool"] == "planar_face_pair":
            face = FaceSelector.model_validate(request[side])
            actual_face = FaceSelector.model_validate(result_side.get("selector"))
            if face != actual_face or face.face_signature != result_side.get("face_signature"):
                raise ValueError("Measured face identity disagrees with exact probe request")
            selector = face.body
        else:
            selector = Selector.model_validate(request[side])
            if selector != Selector.model_validate(result_side.get("selector")):
                raise ValueError("Measured body selector disagrees with exact probe request")
        if selector.body_index is not None and selector.body_index != index:
            raise ValueError("Measured body index disagrees with exact probe request")
        matches = [o for o in occurrences if o.get("body_index") == index]
        if occurrences:
            if len(matches) != 1 or not isinstance(matches[0].get("path"), str):
                raise ValueError("Measured body does not map uniquely to the current assembly")
            path = matches[0]["path"]
            if selector.occurrence_path is not None and selector.occurrence_path != path:
                raise ValueError("Measured occurrence disagrees with the current assembly")
            if sum(o.get("path") == path for o in occurrences) != 1:
                raise ValueError("Measured occurrence is ambiguous in the current assembly")
            paths.append(path)
        elif selector.occurrence_path is not None or not (
            type(facts.get("solid_count")) is int and index < facts["solid_count"]
        ):
            raise ValueError("Measured body is unavailable in the current revision")
        indices.append(index)
    if len(set(indices)) != 2:
        raise ValueError("Measured pair must identify two distinct bodies")
    return paths, indices


def validate_inspection_plan(plan: dict[str, Any], contract: dict[str, Any] | None) -> None:
    checks = {c["id"]: c for s in plan.get("subsystems", []) for c in s.get("checks", [])}
    requirements = {r["id"] for r in (contract or {}).get("requirements", [])}
    ids: set[str] = set()
    for raw in plan.get("inspection", []):
        row = InspectionCharacteristic.model_validate(raw)
        if row.id in ids:
            raise ValueError("Duplicate inspection characteristic ID")
        ids.add(row.id)
        if not set(row.requirement_ids) <= requirements:
            raise ValueError("Inspection refers to an unknown frozen requirement")
        if row.probe:
            check = checks.get(row.probe.check_id)
            if check is None:
                raise ValueError("Inspection refers to an unknown prepared check")
            if row.probe.result != "probe" and check["request"]["tool"] not in {
                "projected_pair",
                "planar_face_pair",
            }:
                raise ValueError("Numeric inspection result requires a supported pair measurement")


def resolve_inspection_plan(
    service: WorkspaceService, sid: str, owner: str, revision: dict[str, Any]
) -> dict[str, Any]:
    from forge_mind.workspace.completion import _hash, _numeric_selector, _snapshot
    from forge_mind.workspace.completion_inline import _read_cached
    from forge_mind.workspace.store import WorkspaceError

    session = service.store.get(sid, owner)
    plan = session.get("design_plan") or {}
    validate_inspection_plan(plan, revision["contract"])
    snapshot = _snapshot(service, sid, owner, revision["id"])
    if snapshot["plan_hash"] != _hash(session.get("design_plan")):
        raise WorkspaceError("Inspection plan changed while reading")
    checks = {c["id"]: c for s in plan.get("subsystems", []) for c in s.get("checks", [])}
    cache = session.get("prepared_check_cache") or {}
    obligations = {d["requirement_id"]: d for d in plan.get("requirement_dispositions", [])}
    paths = {
        o["path"]
        for o in (revision["measurement"].get("facts", {}).get("assembly") or {}).get(
            "occurrences", []
        )
    }
    receipts: dict[str, Any] = {}
    rows = []
    for raw in plan.get("inspection", []):
        row = InspectionCharacteristic.model_validate(raw)
        missing = list(row.missing_evidence)
        for requirement_id in row.requirement_ids:
            disposition = obligations.get(requirement_id, {})
            for key in ("missing_inputs", "unsupported_checks", "physical_validation"):
                missing.extend(f"{requirement_id}: {item}" for item in disposition.get(key, []))
        if row.nominal_mm is None:
            missing.append("Proposed nominal is unspecified")
        if row.minus_mm is None:
            missing.append("Manufacturing allowances are unspecified")
        missing_paths = sorted(set(row.occurrence_paths) - paths)
        if missing_paths:
            missing.append("Occurrence unavailable: " + ", ".join(missing_paths))
        cad: dict[str, Any] = {
            "status": "unavailable",
            "value_mm": None,
            "receipt_id": None,
            "reason": "No prepared probe linked",
            "algorithm_scope": None,
            "measured_occurrence_paths": [],
            "measured_body_indices": [],
        }
        if row.probe:
            definition = checks[row.probe.check_id]
            receipt = _read_cached(
                service, sid, snapshot, definition, cache.get(row.probe.check_id, {})
            )
            numeric_stale = _numeric_selector(definition["request"]) and (
                definition.get("reference_revision_id") != revision["id"]
            )
            if (
                receipt is None
                or numeric_stale
                or (missing_paths and row.probe.result == "probe")
                or receipt.get("result", {}).get("tool") != definition["request"]["tool"]
            ):
                cad["reason"] = (
                    "Current exact probe receipt unavailable or geometry reference stale"
                )
            else:
                result = receipt["result"]
                receipts[receipt["id"]] = receipt
                cad.update(
                    receipt_id=receipt["id"],
                    receipt_digest=_hash(receipt),
                    probe_status=result["status"],
                    algorithm_scope=result.get("algorithm_scope"),
                    status="probe_only",
                    reason="Exploratory probe; no numeric measurement selected",
                )
                if row.probe.result != "probe":
                    values = result.get("results")
                    value = values.get(row.probe.result) if isinstance(values, dict) else None
                    if (
                        isinstance(values, dict)
                        and values.get("units") == "mm"
                        and values.get("measurement_status") == "measured"
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                    ):
                        try:
                            actual_paths, actual_indices = measured_pair_identity(
                                definition["request"], values, revision
                            )
                            cad.update(
                                measured_occurrence_paths=actual_paths,
                                measured_body_indices=actual_indices,
                            )
                            if row.occurrence_paths and (
                                len(row.occurrence_paths) != 2
                                or set(row.occurrence_paths) != set(actual_paths)
                            ):
                                raise ValueError(
                                    "Inspection attribution mismatch: authored occurrences do not match the measured pair"
                                )
                        except (ValueError, KeyError, TypeError) as exc:
                            cad.update(status="unavailable", reason=str(exc))
                            missing.append(str(exc))
                        else:
                            cad.update(
                                status="measured",
                                value_mm=value,
                                reason="Scoped CAD measurement; proposed limits are not acceptance",
                                instrument_frame={
                                    "axis": values.get("axis"),
                                    "datum": values.get("datum"),
                                    "a": values.get("a"),
                                    "b": values.get("b"),
                                },
                            )
                            if row.probe.result == "ordered_separation_mm":
                                cad["dimension_points_mm"] = dimension_points(
                                    result["tool"], values
                                )
                                cad["dimension_scope"] = (
                                    "Ordered support/plane separation on the declared datum axis; endpoints need not lie on trimmed faces"
                                )
                    else:
                        cad.update(
                            status="unavailable",
                            reason="Receipt has no supported numeric measurement",
                        )
        rows.append(
            {
                "intent": row.model_dump(),
                "cad": cad,
                "missing_evidence": missing,
                "physical": {
                    "measured_mm": None,
                    "uncertainty_mm": None,
                    "instrument": None,
                    "accepted_by": None,
                    "status": "unverified",
                },
            }
        )
    if _snapshot(service, sid, owner, revision["id"]) != snapshot:
        raise WorkspaceError("Inspection inputs changed while reading receipts")
    return {
        **snapshot,
        "rows": rows,
        "receipts": receipts,
        "requirement_dispositions": plan.get("requirement_dispositions", []),
        "scope": "Authored inspection intent and scoped CAD evidence. Frozen requirements and physical acceptance are unchanged.",
    }
