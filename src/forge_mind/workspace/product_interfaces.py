"""User-confirmed source-to-tool bindings, separate from frozen acceptance findings."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.budget import end_time, expired, time_left
from forge_mind.workspace.models import Record
from forge_mind.workspace.processing import MAX_JOB_BYTES, regular_file
from forge_mind.workspace.product_sources import _access, _digest, _read, read_sources
from forge_mind.workspace.product_tools import _ADAPTER
from forge_mind.workspace.store import Store, WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


class ClaimBinding(Record):
    claim_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    request_path: str = Field(min_length=1, max_length=160)


class InterfaceProposal(Record):
    name: str = Field(min_length=1, max_length=160)
    reference_revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    requirement_ids: list[str] = Field(min_length=1, max_length=16)
    configuration: str = Field(min_length=1, max_length=1000)
    datum: str = Field(min_length=1, max_length=1000)
    request: dict[str, Any]
    bindings: list[ClaimBinding] = Field(default_factory=list, max_length=8)


def _table(store: Store) -> None:
    with store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS product_interface_events (session TEXT NOT NULL, version INTEGER NOT NULL, document TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(session,version), FOREIGN KEY(session) REFERENCES sessions(id))"
        )


def _state(con: Any, sid: str) -> dict[str, Any]:
    records: dict[str, Any] = {}
    version = 0
    for row in con.execute(
        "SELECT version,document,digest FROM product_interface_events WHERE session=? ORDER BY version",
        (sid,),
    ):
        event = json.loads(row[1])
        version += 1
        if row[0] != version or _digest(event) != row[2]:
            raise WorkspaceError("Interface event integrity check failed")
        if event["kind"] == "proposal":
            records[event["data"]["id"]] = event["data"]
        elif event["kind"] == "confirmation":
            records[event["interface_id"]]["confirmation"] = event["data"]
        elif event["kind"] == "check":
            records[event["interface_id"]]["last_check"] = event["data"]
    return {"version": version, "records": list(records.values())}


def _context(con: Any, sid: str, owner: str, rid: str | None = None) -> tuple[dict[str, Any], str]:
    _access(con, sid, owner)
    session = json.loads(
        con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
    )
    if rid:
        row = con.execute(
            "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
        ).fetchone()
        if not row:
            raise WorkspaceError("Interface reference revision not found")
        contract = json.loads(row[0]).get("contract")
    else:
        contract = session.get("contract")
    if not isinstance(contract, dict):
        return {}, _digest(None)
    return contract, _digest(contract)


def _eligible_hashes(con: Any, sid: str, rid: str) -> set[str]:
    session = json.loads(
        con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
    )
    allowed = {_digest(session["contract"])} if isinstance(session.get("contract"), dict) else set()
    proposal = session.get("proposal") or {}
    current_version = con.execute(
        "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
    ).fetchone()
    if (
        proposal.get("eligible") is True
        and proposal.get("revision_id") == rid
        and isinstance(proposal.get("contract"), dict)
        and proposal.get("based_on")
        == session.get("accepted_revision", session.get("selected_revision"))
        and proposal.get("input_version", 0) == (current_version[0] if current_version else 0)
        and (
            proposal.get("source_fingerprint") is None
            or proposal["source_fingerprint"] == _read(con, sid)["fingerprint"]
        )
    ):
        allowed.add(_digest(proposal["contract"]))
    return allowed


def _check_bindings(
    proposal: InterfaceProposal, sources: dict[str, Any], contract: dict[str, Any]
) -> None:
    _ADAPTER.validate_python(proposal.request)
    known = {r["id"] for r in contract.get("requirements", [])}
    if (
        len(set(proposal.requirement_ids)) != len(proposal.requirement_ids)
        or set(proposal.requirement_ids) - known
    ):
        raise WorkspaceError("Interface references a missing or duplicate frozen requirement")
    artifacts = {a["id"] for a in sources["artifacts"] if a.get("kind") == "public_fetch"}
    claims = {c["id"]: c for c in sources["claims"]}
    prefixes = {
        "fastener_stack": r"/(?:screw_length/(?:low|high)|grip_layers/\d+/(?:low|high)|available_depth/(?:low|high)|required_engagement|minimum_tip_clearance)",
        "tolerance_stack": r"/(?:terms/\d+/(?:low|high)|allowable/(?:low|high))",
        "cable_bend": r"/(?:centerline_radii/\d+|bundle_radius|minimum_inside_radius)",
        "mount_grid": r"/(?:pitch_x|pitch_y|diameter|origin/[012])",
        "hole_pattern": r"/holes/\d+/(?:center/[012]|diameter)",
        "vent_open_fraction": r"/(?:holes/\d+/(?:center/[012]|diameter)|region_origin/[012]|region_width|region_height)",
    }
    seen = set()
    for binding in proposal.bindings:
        if binding.request_path in seen:
            raise WorkspaceError("Duplicate dimensional binding path")
        seen.add(binding.request_path)
        claim = claims.get(binding.claim_id)
        if (
            not claim
            or not claim.get("confirmation")
            or claim.get("source_artifact") not in artifacts
        ):
            raise WorkspaceError(
                "Binding requires an independently confirmed claim and fetched artifact"
            )
        if claim["unit"] not in {"mm", "cm", "m", "in"}:
            raise WorkspaceError("Binding supports length units mm/cm/m/in only")
        if not re.fullmatch(prefixes.get(proposal.request["tool"], r"(?!)"), binding.request_path):
            raise WorkspaceError(
                "Binding pointer must target a supported length field, not an axis or selector"
            )
        value: Any = proposal.request
        try:
            for segment in binding.request_path.split("/")[1:]:
                value = value[int(segment)] if isinstance(value, list) else value[segment]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise WorkspaceError("Binding JSON pointer does not exist") from exc
        expected = (
            claim["quantity"] * {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}[claim["unit"]]
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(expected)
            or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-7)
        ):
            raise WorkspaceError(
                "Bound request value differs from the confirmed source quantity in mm"
            )


def _append(con: Any, sid: str, state: dict[str, Any], event: dict[str, Any]) -> None:
    text = json.dumps(event, sort_keys=True, allow_nan=False)
    if len(text.encode()) > 256000:
        raise WorkspaceError("Interface event exceeds size budget")
    con.execute(
        "INSERT INTO product_interface_events VALUES (?,?,?,?)",
        (sid, state["version"] + 1, text, _digest(event)),
    )


def propose_interface(
    store: Store,
    sid: str,
    owner: str,
    payload: InterfaceProposal | dict[str, Any],
    *,
    agent: bool = False,
) -> dict[str, Any]:
    proposal = InterfaceProposal.model_validate(payload)
    proposal = proposal.model_copy(
        update={"request": _ADAPTER.validate_python(proposal.request).model_dump(mode="json")}
    )
    read_sources(store, sid, owner)  # Initialize before transaction; no nested SQL writes.
    _table(store)
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        _access(con, sid, owner, mutation=True, agent=agent)
        state = _state(con, sid)
        if len(state["records"]) >= 16:
            raise WorkspaceError("Workspace supports at most 16 interface proposals")
        contract, contract_hash = _context(con, sid, owner, proposal.reference_revision_id)
        if contract_hash not in _eligible_hashes(con, sid, proposal.reference_revision_id):
            raise WorkspaceError("Reference revision contract is no longer current")
        sources = _read(con, sid)
        _check_bindings(proposal, sources, contract)
        data = {
            **proposal.model_dump(),
            "id": uuid.uuid4().hex,
            "at": time.time(),
            "contract_hash": contract_hash,
            "source_fingerprint": sources["fingerprint"],
            "binding_scope": "source-bound" if proposal.bindings else "user-declared",
            "confirmation": None,
            "last_check": None,
        }
        _append(con, sid, state, {"kind": "proposal", "data": data})
    return read_interfaces(store, sid, owner, proposal.reference_revision_id)


def confirm_interface(
    store: Store, sid: str, owner: str, interface_id: str, version: int
) -> dict[str, Any]:
    _table(store)
    read_sources(store, sid, owner)
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        _access(con, sid, owner, mutation=True)
        state = _state(con, sid)
        if version != state["version"]:
            raise WorkspaceError("Interface version changed; review it again")
        record = next((r for r in state["records"] if r["id"] == interface_id), None)
        if not record:
            raise WorkspaceError("Interface not found")
        contract, contract_hash = _context(con, sid, owner, record["reference_revision_id"])
        sources = _read(con, sid)
        if (
            record["contract_hash"] != contract_hash
            or record["contract_hash"]
            not in _eligible_hashes(con, sid, record["reference_revision_id"])
            or record["source_fingerprint"] != sources["fingerprint"]
        ):
            raise WorkspaceError("Interface source or frozen contract changed; propose again")
        proposal = InterfaceProposal.model_validate(
            {k: record[k] for k in InterfaceProposal.model_fields}
        )
        _check_bindings(proposal, sources, contract)
        _append(
            con,
            sid,
            state,
            {
                "kind": "confirmation",
                "interface_id": interface_id,
                "data": {
                    "confirmed_by": owner,
                    "at": time.time(),
                    "proposal_hash": _digest(proposal.model_dump()),
                    "meaning": "User accepts this exact configuration, datum, source mapping and request. This is semantic binding confirmation, not source truth or physical validation.",
                },
            },
        )
    return read_interfaces(store, sid, owner)


def _indexed_selector(request: Any) -> bool:
    if isinstance(request, dict):
        return request.get("body_index") is not None or any(
            _indexed_selector(v) for v in request.values()
        )
    if isinstance(request, list):
        return any(_indexed_selector(v) for v in request)
    return False


def _artifacts_current(store: Store, sid: str, rid: str, check: dict[str, Any], tool: str) -> bool:
    hashes = check.get("artifact_hashes", {})
    geometry = tool in {"hole_pattern", "mount_grid", "vent_open_fraction", "clearance_volume"}
    if (
        not isinstance(hashes, dict)
        or len(hashes) > 1024
        or (geometry and "part.step" not in hashes)
    ):
        return False
    directory = store.directory(sid, rid)
    try:
        if (
            sum(
                (directory / name).stat().st_size
                for name in hashes
                if isinstance(name, str)
                and Path(name).name == name
                and regular_file(directory / name)
            )
            > MAX_JOB_BYTES
        ):
            return False
        return all(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name)
            and Path(name).name == name
            and regular_file(directory / name)
            and file_sha256(directory / name) == digest
            for name, digest in hashes.items()
        )
    except (OSError, ValueError):
        return False


def read_interfaces(store: Store, sid: str, owner: str, rid: str | None = None) -> dict[str, Any]:
    _table(store)
    read_sources(store, sid, owner)
    with store.connect() as con:
        _access(con, sid, owner)
        state = _state(con, sid)
        if not state["records"]:
            return state
        _, contract_hash = _context(con, sid, owner, rid)
        allowed = {
            r["id"]: _eligible_hashes(con, sid, rid or r["reference_revision_id"])
            for r in state["records"]
        }
        sources = _read(con, sid)
    from forge_mind.workspace.measure import engine_fingerprint

    current_engine = (
        engine_fingerprint() if any(r.get("last_check") for r in state["records"]) else None
    )
    for record in state["records"]:
        record["stale"] = (
            (rid is not None and record["contract_hash"] != contract_hash)
            or record["contract_hash"] not in allowed[record["id"]]
            or record["source_fingerprint"] != sources["fingerprint"]
        )
        record["stale_reason"] = (
            "Interface source or frozen contract changed" if record["stale"] else None
        )
        if (
            rid is not None
            and rid != record["reference_revision_id"]
            and _indexed_selector(record["request"])
        ):
            record["stale"] = True
            record["stale_reason"] = (
                "Body indices may change between revisions; propose and confirm this interface again or use occurrence paths"
            )
        check = record.get("last_check")
        artifacts_current = bool(
            check and rid and _artifacts_current(store, sid, rid, check, record["request"]["tool"])
        )
        record["check_reason"] = record["stale_reason"] or (
            "Receipt artifacts are missing, changed, linked, or unavailable"
            if check and not artifacts_current
            else None
        )
        record["check_status"] = (
            check["result"]["status"]
            if check
            and check.get("revision_id") == rid
            and not record["stale"]
            and check.get("engine_fingerprint") == current_engine
            and artifacts_current
            else "unverified"
        )
        record["scope"] = "Supplemental interface evidence only; frozen findings remain unchanged"
    return state


def validate_interfaces(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    timeout: float | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> list[dict[str, Any]]:
    from forge_mind.workspace.product_service import run_tool

    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise WorkspaceError("Interface validation requires a positive finite timeout")
    deadline = end_time(timeout)
    state = read_interfaces(service.store, sid, owner, rid)
    results = []
    for record in state["records"]:
        if not record["confirmation"]:
            continue
        receipt: dict[str, Any] = {
            "interface_id": record["id"],
            "revision_id": rid,
            "requirement_ids": record["requirement_ids"],
            "source_fingerprint": record["source_fingerprint"],
            "contract_hash": record["contract_hash"],
            "binding_confirmation": record["confirmation"],
            "result": {"status": "unverified"},
        }
        if record["stale"]:
            receipt["reason"] = record["stale_reason"]
        elif cancelled() or expired(deadline):
            receipt["reason"] = "Interface validation cancelled or time budget exhausted"
        else:
            try:
                receipt.update(
                    run_tool(
                        service,
                        sid,
                        owner,
                        rid,
                        record["request"],
                        time_left(deadline),
                        cancelled,
                    )
                )
            except (WorkspaceError, ValueError, OSError) as exc:
                receipt["reason"] = f"Interface tool unavailable: {type(exc).__name__}"

        with service.store.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            _, contract_hash = _context(con, sid, owner, rid)
            allowed_hashes = _eligible_hashes(con, sid, rid)
            source_state = _read(con, sid)
            if (
                contract_hash != record["contract_hash"]
                or record["contract_hash"] not in allowed_hashes
                or source_state["fingerprint"] != record["source_fingerprint"]
            ):
                receipt["result"] = {"status": "unverified"}
                receipt["reason"] = "Interface inputs changed during validation"
            _append(
                con,
                sid,
                _state(con, sid),
                {"kind": "check", "interface_id": record["id"], "data": receipt},
            )
        results.append(receipt)
    return results


run_confirmed_interfaces = validate_interfaces
