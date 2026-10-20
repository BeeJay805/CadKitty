"""Read saved completion proof under a caller-owned snapshot; never run checks."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    import sqlite3


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reusable_result(result: dict[str, Any]) -> bool:
    return result.get("status") in {"pass", "fail"} or (
        result.get("status") == "unverified"
        and isinstance(result.get("results"), dict)
        and result["results"].get("measurement_status") == "measured"
    )


def prepared_binding(snapshot: dict[str, Any], definition: dict[str, Any]) -> str:
    # Unrelated plan amendments do not change the measured request.
    return _hash(
        {
            "inputs": {key: value for key, value in snapshot.items() if key != "plan_hash"},
            "definition": definition,
        }
    )


def read_prepared_receipt(
    con: sqlite3.Connection,
    sid: str,
    snapshot: dict[str, Any],
    definition: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Verify the actual immutable receipt, not a cached verdict or receipt ID."""
    if entry.get("binding") != prepared_binding(snapshot, definition) or not entry.get(
        "receipt_id"
    ):
        return None
    try:
        row = con.execute(
            "SELECT document,digest FROM product_checks WHERE session=? AND id=?",
            (sid, entry["receipt_id"]),
        ).fetchone()
        if row is None or hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            return None
        receipt = json.loads(row[0])
        if receipt.get("id") != entry["receipt_id"]:
            return None
        if (
            any(
                receipt.get(key) != snapshot.get(key)
                for key in (
                    "revision_id",
                    "source_fingerprint",
                    "engine_fingerprint",
                    "artifact_hashes",
                )
            )
            or _hash(receipt.get("contract")) != snapshot["contract_hash"]
        ):
            return None
        if _hash(receipt.get("request")) != _hash(definition["request"]):
            return None
        if not reusable_result(receipt.get("result", {})):
            return None
        return dict(receipt)
    except Exception:
        return None


def motion_summary(
    result: dict[str, Any],
    revision: dict[str, Any],
    snapshot: dict[str, Any],
    check_id: str,
) -> dict[str, Any]:
    """Validate saved sampled coverage using the existing completion rules."""
    from forge_mind.workspace.motion_check import MotionCheckSettings, configurations

    settings = MotionCheckSettings()
    expected = {
        "revision_id": snapshot["revision_id"],
        "engine_fingerprint": snapshot["engine_fingerprint"],
        "step_sha256": snapshot["artifact_hashes"].get("part.step"),
        "document_sha256": snapshot["artifact_hashes"].get("document.json"),
        "settings": settings.model_dump(),
    }
    if any(value is None or result.get(key) != value for key, value in expected.items()):
        raise WorkspaceError("Motion receipt does not match the revision, artifacts or settings")
    motion = revision["measurement"]["facts"]["motion_rig"]
    poses = configurations(motion["rig"], settings)
    bodies = len(motion["bindings"])
    expected_pairs = len(poses) * bodies * (bodies - 1) // 2
    counts: dict[str, Any] = {
        key: result.get(key)
        for key in (
            "collision_count",
            "completed_pairs",
            "total_pairs",
            "exact_intersections",
            "disjoint_bounds",
            "rigid_pair_reuses",
        )
    }
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise WorkspaceError("Motion receipt has invalid coverage counts")
    errors, collisions = result.get("errors"), result.get("collisions")
    if not isinstance(errors, list) or not isinstance(collisions, list):
        raise WorkspaceError("Motion receipt is missing error or collision evidence")
    complete = (
        result.get("status") == "unverified"
        and result.get("coverage") == "discrete_only"
        and not errors
        and expected_pairs > 0
        and counts["completed_pairs"] == counts["total_pairs"] == expected_pairs
        and counts["exact_intersections"] + counts["disjoint_bounds"] + counts["rigid_pair_reuses"]
        == expected_pairs
        and result.get("requested_configurations") == poses
        and result.get("completed_configurations") == poses
    )
    collided = bool(counts["collision_count"] or collisions or result.get("status") == "fail")
    return {
        "status": "checked_fail" if collided else "checked" if complete else "unverified",
        "reason": "Sampled body intersections found"
        if collided
        else "All pairs checked at the listed discrete configurations; physical readiness remains unverified"
        if complete
        else "Motion coverage is incomplete, empty or contains errors",
        "check_id": check_id,
        "coverage": result.get("coverage", "incomplete"),
        **counts,
        "completed_configurations": len(result.get("completed_configurations") or []),
        "requested_configurations": len(poses),
        "collisions": collisions[:5],
        "errors": [str(error)[:300] for error in errors[:5]],
        "scope": str(result.get("scope", "Discrete motion evidence only"))[:1000],
        "qualification_status": "unverified",
    }


def project_motion(
    session: dict[str, Any],
    revision: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    motion: dict[str, Any] | None = None,
    root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify the latest current sampled-motion receipt without running a check."""
    verified_motion = deepcopy(motion or {})
    verified_motion.pop("verified_evidence", None)
    motion_proof = None
    registered = session.get("motion_checks", [])
    current_motion = (
        next(
            (
                item
                for item in reversed(registered)
                if item.get("revision_id") == snapshot["revision_id"]
            ),
            None,
        )
        if motion is None
        else None
    )
    if current_motion or verified_motion.get("status") in {"checked", "checked_fail"}:
        check_id = current_motion["id"] if current_motion else verified_motion.get("check_id")
        try:
            receipt = next(item for item in registered if item["id"] == check_id)
            if root is None:
                raise WorkspaceError("Motion receipt location is unavailable")
            path = root / "motion-checks" / session["id"] / str(check_id) / "check.json"
            if not regular_file(path):
                raise WorkspaceError("Motion check evidence changed")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
                raise WorkspaceError("Motion check evidence changed")
            result = json.loads(content)
            summary = motion_summary(result, revision, snapshot, str(check_id))
            if motion is not None and summary != motion:
                raise WorkspaceError("Motion summary differs from its current receipt")
            verified_motion = summary
            verified_motion["verified_evidence"] = result
            motion_proof = {
                key: value
                for key, value in result.items()
                if key not in {"wall_seconds", "cpu_seconds", "worker_cpu_affinity"}
            }
        except (ValueError, OSError, KeyError, StopIteration, WorkspaceError):
            verified_motion.update(status="unverified", reason="Motion receipt is unavailable")
    if (
        verified_motion
        and revision.get("measurement", {}).get("engine_fingerprint")
        != snapshot["engine_fingerprint"]
    ):
        verified_motion.pop("verified_evidence", None)
        verified_motion.update(status="unverified", reason="Motion receipt is unavailable")
        motion_proof = None
    return verified_motion, motion_proof


def project_interfaces(
    con: sqlite3.Connection,
    sid: str,
    session: dict[str, Any],
    plan: dict[str, Any],
    revision: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    rows: list[dict[str, Any]] | None = None,
    motion: dict[str, Any] | None = None,
    root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Project exact saved proof without fetching state outside the caller's transaction."""
    from forge_mind.workspace.completion_obligations import interface_obligations

    verified = (
        deepcopy(rows)
        if rows is not None
        else [
            {
                "id": subsystem["id"],
                "checks": [{"id": check["id"]} for check in subsystem.get("checks", [])],
            }
            for subsystem in plan.get("subsystems", [])
        ]
    )
    definitions = {
        check["id"]: check
        for subsystem in plan.get("subsystems", [])
        for check in subsystem.get("checks", [])
    }
    cache = session.get("prepared_check_cache") or {}
    proof: dict[str, Any] = {}
    for subsystem in verified:
        for check in subsystem.get("checks", []):
            check.pop("verified_evidence", None)
            receipt = read_prepared_receipt(
                con, sid, snapshot, definitions[check["id"]], cache.get(check["id"], {})
            )
            if receipt is not None and (
                rows is None or receipt["result"]["status"] == check.get("status")
            ):
                if rows is None:
                    check.update(status=receipt["result"]["status"], receipt_id=receipt["id"])
                check["verified_evidence"] = receipt
                proof[check["id"]] = {
                    key: receipt.get(key)
                    for key in (
                        "revision_id",
                        "source_fingerprint",
                        "engine_fingerprint",
                        "artifact_hashes",
                        "contract",
                        "request",
                        "result",
                    )
                }
    verified_motion, motion_proof = project_motion(
        {**session, "id": sid}, revision, snapshot, motion=motion, root=root
    )
    if revision.get("measurement", {}).get("engine_fingerprint") != snapshot["engine_fingerprint"]:
        # Old exported mates are declarations until the current engine measures them.
        revision = deepcopy(revision)
        assembly = revision.get("measurement", {}).get("facts", {}).get("assembly", {})
        for mate in assembly.get("mates", []):
            mate["status"] = "unverified"
        verified = []
        verified_motion = {}
        proof = {}
        motion_proof = None
    return (
        interface_obligations(
            plan, {**revision, "completion_basis": snapshot}, verified, verified_motion
        ),
        _hash({"prepared_checks": proof, "motion": motion_proof}),
    )
