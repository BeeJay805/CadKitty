"""Owner-scoped durable jobs and immutable revisions, with transactional admission."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import socket
import sqlite3
import time
import uuid
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import psutil

_UNSET_REVISION = object()
_PREVIEW_ACCEPTANCE_IDENTITY = "preview_acceptance_identity"
_BASELINE_SOURCE_FINGERPRINT = "baseline_source_fingerprint"
_ACCEPTANCE_FIELDS = (
    "accepted_revision",
    "accepted_operation_plan_snapshot",
    "accepted_operation_plan_revision",
)


class WorkspaceError(ValueError):
    pass


class EditConflictError(WorkspaceError):
    """The source authority changed; current state must not be overwritten."""


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "workspace.sqlite3"
        with self.connect() as con:
            con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, document TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS session_owner ON sessions(owner);
                CREATE TABLE IF NOT EXISTS customer_requests (
                    owner TEXT NOT NULL, request_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    session TEXT NOT NULL, initial_job TEXT NOT NULL,
                    PRIMARY KEY(owner, request_id),
                    FOREIGN KEY(session) REFERENCES sessions(id),
                    FOREIGN KEY(initial_job) REFERENCES jobs(id)
                );
                CREATE TABLE IF NOT EXISTS customer_operations (
                    owner TEXT NOT NULL, request_id TEXT NOT NULL, session TEXT NOT NULL,
                    payload_hash TEXT NOT NULL, job TEXT NOT NULL,
                    PRIMARY KEY(owner, request_id),
                    FOREIGN KEY(session) REFERENCES sessions(id),
                    FOREIGN KEY(job) REFERENCES jobs(id)
                );
                CREATE TABLE IF NOT EXISTS plan_drafts (
                    session TEXT NOT NULL, version INTEGER NOT NULL,
                    basis TEXT NOT NULL, content TEXT NOT NULL, raw_json TEXT NOT NULL,
                    validation TEXT NOT NULL, content_hash TEXT NOT NULL,
                    admitted INTEGER NOT NULL DEFAULT 0, admitted_plan_hash TEXT,
                    PRIMARY KEY(session, version),
                    FOREIGN KEY(session) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, document TEXT NOT NULL,
                    FOREIGN KEY(session) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, state TEXT NOT NULL,
                    payload TEXT NOT NULL, started REAL, deadline REAL, cancel INTEGER DEFAULT 0,
                    FOREIGN KEY(session) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS engineering_inputs (
                    session TEXT PRIMARY KEY, version INTEGER NOT NULL, document TEXT NOT NULL,
                    FOREIGN KEY(session) REFERENCES sessions(id)
                );
                CREATE TABLE IF NOT EXISTS engineering_history (
                    session TEXT NOT NULL, version INTEGER NOT NULL, document TEXT NOT NULL,
                    PRIMARY KEY(session, version)
                );
            """)
            # Serialize additive migration across independently starting web workers.
            con.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in con.execute("PRAGMA table_info(jobs)")}
            for name, kind in (
                ("worker_pid", "INTEGER"),
                ("worker_created", "REAL"),
                ("worker_host", "TEXT"),
                ("recovery_of", "TEXT"),
                ("operation_directory", "TEXT"),
                ("operation_launch_token", "TEXT"),
                ("native_workers", "TEXT"),
            ):
                if name not in columns:
                    con.execute(f"ALTER TABLE jobs ADD COLUMN {name} {kind}")
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS operation_retry ON jobs(recovery_of) WHERE recovery_of IS NOT NULL"
            )

    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db, timeout=20)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    def create(
        self,
        owner: str,
        prompt: str,
        contract: dict[str, Any] | None,
        authoring_mode: str | None = None,
    ) -> dict[str, Any]:
        with self.connect() as con:
            return self._create_with_connection(con, owner, prompt, contract, authoring_mode)

    def _create_with_connection(
        self,
        con: sqlite3.Connection,
        owner: str,
        prompt: str,
        contract: dict[str, Any] | None,
        authoring_mode: str | None = None,
    ) -> dict[str, Any]:
        from forge_mind.workspace.authoring_policy import validate_authoring_mode

        mode = validate_authoring_mode(authoring_mode)
        sid = uuid.uuid4().hex
        doc: dict[str, Any] = {
            "id": sid,
            "prompt": prompt,
            "contract": contract,
            "state": "idle",
            "selected_revision": None,
            "best_revision": None,
            "created_at": time.time(),
            "events": [],
            "usage": [],
            "job_id": None,
        }
        if authoring_mode is not None:
            doc["authoring_mode"] = mode
        con.execute("INSERT INTO sessions VALUES (?,?,?)", (sid, owner, json.dumps(doc)))
        return doc

    def create_customer_project(
        self,
        owner: str,
        body: dict[str, Any],
        *,
        hosted_limits: dict[str, int] | None = None,
        server_metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create exactly one owned session and initial job, or recover its identity.

        The identity and queue admission share one transaction. A full queue or
        failed admission leaves neither an idle project nor a consumed key.
        """
        from forge_mind.workspace.customer_photo_authority import validate_photo_identity
        from forge_mind.workspace.customer_projects import CustomerProjectRequest
        from forge_mind.workspace.customer_references import (
            _rollback_reference,
            decode_reference,
            retain_reference,
        )
        from forge_mind.workspace.product_sources import _table

        request = CustomerProjectRequest.model_validate(body)
        content = decode_reference(request.reference) if request.reference else None
        photo_identity = validate_photo_identity(server_metadata, content[0] if content else None)
        if request.reference:
            _table(self)
        values = request.model_dump(mode="json")
        request_id = values.pop("request_id")
        payload_hash = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        retained: Path | None = None
        try:
            with self.connect() as con:
                con.execute("BEGIN IMMEDIATE")
                prior = con.execute(
                    "SELECT payload_hash,session FROM customer_requests WHERE owner=? AND request_id=?",
                    (owner, request_id),
                ).fetchone()
                if prior is not None:
                    if prior["payload_hash"] != payload_hash:
                        saved = con.execute(
                            "SELECT document FROM sessions WHERE id=? AND owner=?",
                            (prior["session"], owner),
                        ).fetchone()
                        legacy_values = {
                            key: value for key, value in values.items() if key != "printable"
                        }
                        legacy_hash = hashlib.sha256(
                            json.dumps(
                                legacy_values,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ).encode()
                        ).hexdigest()
                        if (
                            "printable" in body
                            or saved is None
                            or "customer_request_version" in json.loads(saved["document"])
                            or prior["payload_hash"] != legacy_hash
                        ):
                            raise WorkspaceError(
                                "Request identity already belongs to a different request"
                            )
                    if photo_identity is not None:
                        saved_photo = con.execute(
                            "SELECT document FROM sessions WHERE id=? AND owner=?",
                            (prior["session"], owner),
                        ).fetchone()
                        if (
                            saved_photo is None
                            or json.loads(saved_photo["document"]).get("customer_photo_identity")
                            != photo_identity
                        ):
                            raise WorkspaceError(
                                "Request identity already belongs to a different photo request"
                            )
                    sid = prior["session"]
                else:
                    doc = self._create_with_connection(
                        con, owner, request.prompt, values["contract"], "native_operations"
                    )
                    sid = doc["id"]
                    doc.update(
                        project_kind="workspace",
                        creation_request_id=request_id,
                        approval_required=request.approval_required,
                        customer_printable=request.printable,
                        customer_request_version=2,
                    )
                    if photo_identity is not None:
                        doc["customer_photo_identity"] = photo_identity
                    if hosted_limits is not None:
                        if set(hosted_limits) != {"max_seconds", "max_actions"} or any(
                            type(value) is not int or value <= 0 for value in hosted_limits.values()
                        ):
                            raise WorkspaceError(
                                "Hosted customer admission requires explicit positive server caps"
                            )
                        doc["customer_hosted_limits"] = dict(hosted_limits)
                    if request.reference and content:
                        metadata, retained = retain_reference(
                            self, con, sid, request.reference, content
                        )
                        doc["customer_reference"] = metadata
                        dimension = request.reference.known_dimension
                        if dimension:
                            doc["instructions"] = [
                                f"Owner reference measurement: {dimension.label} = {dimension.value:g} {dimension.unit}"
                            ]
                    con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
                    payload = {
                        key: value
                        for key, value in values.items()
                        if key not in {"prompt", "approval_required", "reference", "printable"}
                    }
                    jid = self._enqueue_with_connection(
                        con,
                        sid,
                        owner,
                        {**payload, "preview_only": False, "delivery_requested": True},
                    )
                    queued = con.execute(
                        "SELECT document FROM sessions WHERE id=?", (sid,)
                    ).fetchone()
                    doc = json.loads(queued["document"])
                    doc["initial_job_id"] = jid
                    con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
                    con.execute(
                        "INSERT INTO customer_requests VALUES (?,?,?,?,?)",
                        (owner, request_id, payload_hash, sid, jid),
                    )
        except Exception as exc:
            if retained is not None:
                _rollback_reference(retained, exc)
            raise
        return self.get(sid, owner)

    def customer_request(self, owner: str, request_id: str) -> dict[str, Any]:
        if re.fullmatch(r"[a-f0-9]{32}", request_id) is None:
            raise WorkspaceError("Customer request not found")
        with self.connect() as con:
            row = con.execute(
                "SELECT session FROM customer_requests WHERE owner=? AND request_id=?",
                (owner, request_id),
            ).fetchone()
        if row is None:
            raise WorkspaceError("Customer request not found")
        return self.get(row["session"], owner)

    def authoring_mode(self, sid: str) -> str:
        """Read internal policy without loading revisions or owner-facing metadata."""
        from forge_mind.workspace.authoring_policy import validate_authoring_mode

        with self.connect() as con:
            row = con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise WorkspaceError("workspace not found")
        return validate_authoring_mode(json.loads(row[0]).get("authoring_mode"))

    def plan_snapshot(self, sid: str, owner: str) -> dict[str, Any] | None:
        from forge_mind.workspace.operation_plan import snapshot

        doc = self.session_record(sid, owner)
        if doc.get("operation_plan_snapshot") is not None:
            return dict(doc["operation_plan_snapshot"])
        if doc.get("authoring_mode") == "native_operations":
            return None
        if not doc.get("design_plan"):
            return None
        return snapshot(doc["design_plan"], doc.get("plan_version", 0), doc.get("contract") or {})

    def set_design_plan(
        self,
        sid: str,
        owner: str,
        plan: Any,
        expected_plan_version: int,
        contract: Any = None,
        expected_current_revision: str | None | object = _UNSET_REVISION,
        expected_input_version: int | None = None,
        expected_edit_basis: Any = None,
    ) -> dict[str, Any]:
        return self._set_design_plan(
            sid,
            owner,
            plan,
            expected_plan_version,
            contract,
            expected_current_revision,
            expected_input_version,
            expected_edit_basis=expected_edit_basis,
        )

    def set_design_plan_for_job(
        self,
        sid: str,
        owner: str,
        jid: str,
        plan: Any,
        expected_plan_version: int,
        contract: Any = None,
        expected_current_revision: str | None | object = _UNSET_REVISION,
        expected_input_version: int | None = None,
        expected_edit_basis: Any = None,
    ) -> dict[str, Any]:
        """Only the current local worker can write its plan between native batches."""
        return self._set_design_plan(
            sid,
            owner,
            plan,
            expected_plan_version,
            contract,
            expected_current_revision,
            expected_input_version,
            jid,
            expected_edit_basis,
        )

    def _set_design_plan(
        self,
        sid: str,
        owner: str,
        plan: Any,
        expected_plan_version: int,
        contract: Any,
        expected_current_revision: str | None | object,
        expected_input_version: int | None,
        jid: str | None = None,
        expected_edit_basis: Any = None,
        expected_draft: Any = None,
    ) -> dict[str, Any]:
        from forge_mind.workspace.models import Contract, DesignPlan
        from forge_mind.workspace.operation_plan import snapshot, validate_plan

        plan = DesignPlan.model_validate(
            plan.model_dump() if isinstance(plan, DesignPlan) else plan
        )
        if type(expected_plan_version) is not int or expected_plan_version < 0:
            raise WorkspaceError("expected_plan_version must be a nonnegative integer")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            doc = json.loads(row[0])
            if expected_edit_basis is not None:
                self._compare_edit_basis(con, sid, doc, expected_edit_basis)
            self._check_plan_writer(con, sid, doc, jid)
            if expected_draft is not None:
                from forge_mind.workspace.plan_draft import check_draft_for_admission

                check_draft_for_admission(con, sid, expected_draft, plan)
            if doc.get("plan_version", 0) != expected_plan_version:
                raise WorkspaceError("Design plan changed; refresh before saving")
            head = (doc.get("proposal") or {}).get("revision_id") or doc.get("selected_revision")
            if (
                expected_current_revision is not _UNSET_REVISION
                and expected_current_revision != head
            ):
                raise WorkspaceError("Design changed; refresh before saving the design plan")
            current_input = con.execute(
                "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
            ).fetchone()
            if expected_input_version is not None and expected_input_version != (
                current_input[0] if current_input else 0
            ):
                raise WorkspaceError(
                    "Engineering inputs changed; refresh before saving the design plan"
                )
            proposal = doc.get("proposal") or {}
            current_contract = Contract.model_validate(
                (proposal.get("contract") if jid is None else None) or doc.get("contract") or {}
            )
            target = (
                current_contract
                if contract is None
                else Contract.model_validate(
                    contract.model_dump() if isinstance(contract, Contract) else contract
                )
            )
            native = doc.get("authoring_mode") == "native_operations"
            if native and (
                not target.requirements
                or plan.operation_plan is None
                or not plan.operation_plan.steps
            ):
                raise WorkspaceError(
                    "Native operations require an initialized contract and operation plan"
                )
            if target.fingerprint() != current_contract.fingerprint():
                if current_contract.requirements or doc.get("design_plan"):
                    raise WorkspaceError(
                        "Contract is frozen; use an explicit customer requirement edit"
                    )
                if con.execute("SELECT 1 FROM revisions WHERE session=?", (sid,)).fetchone():
                    raise WorkspaceError("Initialize the contract and plan before the first build")
            same_contract = doc.get("plan_contract_hash") == target.fingerprint()
            previous_plan = doc.get("design_plan") if same_contract else None
            if expected_edit_basis is not None:
                captured = self._edit_target_snapshot(doc, target.fingerprint())
                if captured and captured.get("contract_hash") == target.fingerprint():
                    previous_plan = captured["plan"]
            # An explicit empty predecessor marks first admission. Standalone
            # reads omit previous and must keep historical amended plans loadable.
            validate_plan(plan, target, previous_plan if previous_plan is not None else {})
            if expected_draft is not None:
                from forge_mind.workspace.plan_patch import (
                    _retain_obligations,
                    _unambiguous_dimensions,
                )

                if previous_plan:
                    _retain_obligations(DesignPlan.model_validate(previous_plan), plan)
                _unambiguous_dimensions(plan)
            if jid is not None:
                from forge_mind.workspace.customer_construction import (
                    JOB_BINDING,
                    construction_finalization_allowed,
                )

                job = con.execute(
                    "SELECT payload FROM jobs WHERE id=? AND session=?", (jid, sid)
                ).fetchone()
                if job and json.loads(job[0]).get(JOB_BINDING):
                    if not construction_finalization_allowed(self, sid, owner, jid):
                        raise WorkspaceError(
                            "Approved construction must publish before plan review"
                        )
                    approved = doc["customer_proposal"]["command"]["target_snapshot"]
                    if (
                        plan.model_dump(mode="json") != approved["plan"]
                        or target.model_dump(mode="json") != approved["contract"]
                    ):
                        raise WorkspaceError(
                            "This job must retain its exact approved target plan. Use its saved checks and guidance; changed intent requires a new owner edit."
                        )
                    # An identical resubmission is a read of the approved target,
                    # not a new plan version or a new construction authorization.
                    return deepcopy(approved)
            binding = snapshot(plan, expected_plan_version + 1, target)
            from forge_mind.workspace.input_application import FIELD, apply, baseline, pending

            application = pending(doc)
            if application is not None:
                if jid is None or application["mode"] != "full":
                    raise WorkspaceError("Resume the pending owner input before changing its plan")
                apply(self, con, sid, doc, application["input_id"], target)
            history = list(doc.get("design_plan_history", []))
            if doc.get("design_plan"):
                from forge_mind.workspace.plan_history import plan_history_entry

                history.append(plan_history_entry(con, sid, doc, head_revision_id=head))
            doc.update(
                design_plan=binding["plan"],
                plan_version=binding["plan_version"],
                plan_hash=binding["plan_hash"],
                plan_contract_hash=binding["contract_hash"],
                design_plan_history=history,
                operation_plan_snapshot=binding,
            )
            if jid is not None or not proposal or not native:
                doc["contract"] = binding["contract"]
                if FIELD not in doc or (
                    application is None and target.fingerprint() != current_contract.fingerprint()
                ):
                    doc[FIELD] = baseline(doc)
            if target.fingerprint() != current_contract.fingerprint():
                doc["contract_initialization"] = {
                    "contract_hash": binding["contract_hash"],
                    "plan_hash": binding["plan_hash"],
                    "source": "initial_design_plan",
                    "at": time.time(),
                }
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
            if expected_draft is not None:
                con.execute(
                    "UPDATE plan_drafts SET admitted=1, admitted_plan_hash=? "
                    "WHERE session=? AND version=?",
                    (binding["plan_hash"], sid, expected_draft.draft_version),
                )
        return binding

    @staticmethod
    def _check_plan_writer(
        con: sqlite3.Connection,
        sid: str,
        doc: dict[str, Any],
        jid: str | None,
        *,
        allow_intent_record: bool = False,
    ) -> None:
        """Admitted plans and unadmitted drafts share the same worker ownership guard."""
        active = con.execute(
            "SELECT * FROM jobs WHERE session=? AND state IN ('queued','running')", (sid,)
        ).fetchone()
        if jid is None and active is not None:
            raise WorkspaceError("Wait for the active operation before changing the design plan")
        if jid is not None and (
            active is None
            or active["id"] != jid
            or active["state"] != "running"
            or active["cancel"]
            or doc.get("job_id") != jid
            or active["worker_pid"] != os.getpid()
            or active["worker_host"] != socket.gethostname()
            or active["worker_created"] != psutil.Process(os.getpid()).create_time()
        ):
            raise WorkspaceError("Design plan write requires the current owned running job")
        if (
            jid is not None
            and active is not None
            and not Store._preview_acceptance_matches(doc, json.loads(active["payload"]))
        ):
            raise EditConflictError("Acceptance authority changed before the owned write")
        if (
            not allow_intent_record
            and (doc.get("intent_attribution") or {}).get("status") == "needs_input"
        ):
            raise WorkspaceError("Owner clarification is required before changing the design plan")

    @staticmethod
    def _edit_basis(
        con: sqlite3.Connection,
        sid: str,
        doc: dict[str, Any],
        rid: Any = _UNSET_REVISION,
        target: Any = None,
    ) -> Any:
        from forge_mind.workspace.document import DesignDocument
        from forge_mind.workspace.edit_contracts import EditBasis
        from forge_mind.workspace.models import Contract
        from forge_mind.workspace.product_sources import source_fingerprint

        if rid is _UNSET_REVISION:
            rid = (doc.get("proposal") or {}).get("revision_id") or doc.get("selected_revision")
        source_hash = None
        if rid is not None:
            row = con.execute(
                "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
            ).fetchone()
            revision = json.loads(row[0]) if row else {}
            if not revision.get("document"):
                raise WorkspaceError("Focused edits require a saved typed document")
            source_hash = DesignDocument.model_validate(revision["document"]).fingerprint()
        inputs = con.execute(
            "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
        ).fetchone()
        proposal = doc.get("proposal") or {}
        effective = (
            target.get("contract")
            if target
            else proposal.get("contract") or doc.get("contract") or {}
        )
        captured = target or Store._edit_target_snapshot(
            doc, Contract.model_validate(effective).fingerprint()
        )
        plan_hash = (
            captured["plan_hash"]
            if captured
            and captured.get("contract_hash") == Contract.model_validate(effective).fingerprint()
            else doc.get("plan_hash")
        )
        return EditBasis(
            revision_id=rid,
            document_hash=source_hash,
            plan_version=doc.get("plan_version", 0),
            plan_hash=plan_hash,
            contract_hash=Contract.model_validate(effective).fingerprint(),
            input_version=inputs[0] if inputs else 0,
            source_fingerprint=source_fingerprint(con, sid),
        )

    @classmethod
    def _compare_edit_basis(
        cls,
        con: sqlite3.Connection,
        sid: str,
        doc: dict[str, Any],
        basis: Any,
        rid: Any = _UNSET_REVISION,
        target: Any = None,
    ) -> None:
        from forge_mind.workspace.edit_contracts import EditBasis

        expected = EditBasis.model_validate(
            basis.model_dump() if isinstance(basis, EditBasis) else basis
        )
        actual = cls._edit_basis(con, sid, doc, rid, target)
        if expected != actual:
            changed = [
                key
                for key, value in expected.model_dump().items()
                if actual.model_dump()[key] != value
            ]
            raise EditConflictError(f"Edit basis changed; refresh before editing: {changed}")

    def edit_basis(self, sid: str, owner: str) -> Any:
        from pydantic import ValidationError

        with self.connect() as con:
            row = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            try:
                return self._edit_basis(con, sid, json.loads(row[0]))
            except ValidationError as exc:
                raise WorkspaceError(
                    "Saved revision or plan is not a complete typed edit source"
                ) from exc

    @staticmethod
    def _edit_target_snapshot(doc: dict[str, Any], contract_hash: str) -> dict[str, Any] | None:
        candidates = [
            value
            for value in (
                doc.get("operation_plan_snapshot"),
                (doc.get("proposal") or {}).get("operation_plan_snapshot"),
                doc.get("accepted_operation_plan_snapshot"),
            )
            if value and value.get("contract_hash") == contract_hash
        ]
        return (
            max(candidates, key=lambda value: value.get("plan_version", 0)) if candidates else None
        )

    def publish_construction(
        self, sid: str, owner: str, jid: str, prepared: Any, rid: str
    ) -> dict[str, Any]:
        """Publish a staged explicit successor through the existing guarded boundary."""
        return self.publish_feature_edit(
            sid, owner, jid, prepared.basis.model_dump(), rid, _construction=prepared
        )

    def publish_feature_edit(
        self, sid: str, owner: str, jid: str, basis: Any, rid: str, *, _construction: Any = None
    ) -> dict[str, Any]:
        """Publish one measured candidate only while its owned base remains current."""
        from forge_mind.workspace.edit_contracts import EditBasis

        basis = EditBasis.model_validate(basis)
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            doc = json.loads(row[0])
            job = con.execute("SELECT * FROM jobs WHERE id=? AND session=?", (jid, sid)).fetchone()
            if (
                job is None
                or job["state"] != "running"
                or job["cancel"]
                or doc.get("job_id") != jid
                or job["worker_pid"] != os.getpid()
                or job["worker_host"] != socket.gethostname()
                or job["worker_created"] != psutil.Process(os.getpid()).create_time()
            ):
                raise WorkspaceError("Feature publication requires the current owned running job")
            payload = json.loads(job["payload"])
            if not self._preview_acceptance_matches(doc, payload):
                raise EditConflictError("Acceptance authority changed before feature publication")
            successor = None
            if _construction is not None:
                from forge_mind.workspace.construction_transaction import validated_successor

                self._check_plan_writer(con, sid, doc, jid)
                if (
                    self._native_author_binding(doc) != _construction.authority
                    or doc.get("selection_version", 0) != _construction.selection_version
                    or self._preview_acceptance_identity(doc) != _construction.acceptance
                ):
                    raise EditConflictError(
                        "Design authority changed before construction publication"
                    )
                successor = validated_successor(_construction)
                if basis.revision_id is not None:
                    from forge_mind.workspace.measure import verify_artifacts
                    from forge_mind.workspace.models import Measurement

                    source_row = con.execute(
                        "SELECT document FROM revisions WHERE session=? AND id=?",
                        (sid, basis.revision_id),
                    ).fetchone()
                    source = json.loads(source_row[0]) if source_row else {}
                    if not source or not verify_artifacts(
                        self.directory(sid, basis.revision_id),
                        Measurement.model_validate(source.get("measurement")),
                    ):
                        raise WorkspaceError("Construction source artifacts changed or are missing")
            admission = payload.get("feature_edit_admission")
            if admission is not None:
                if admission != self._native_author_binding(doc):
                    raise EditConflictError("Design authority changed before feature publication")
                if payload.get("feature_edit_basis") != basis.model_dump():
                    raise WorkspaceError("Feature publication basis differs from its admission")
                target = payload.get("plan_snapshot") or {
                    "contract": payload.get("contract") or doc.get("contract"),
                    "plan_hash": basis.plan_hash,
                    "contract_hash": basis.contract_hash,
                }
                self._compare_edit_basis(con, sid, doc, basis, basis.revision_id, target)
            else:
                self._compare_edit_basis(con, sid, doc, basis)
            candidate_row = con.execute(
                "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
            ).fetchone()
            candidate = json.loads(candidate_row[0]) if candidate_row else {}
            record = candidate.get("operation_record") or {}
            from forge_mind.workspace.document import DesignDocument
            from forge_mind.workspace.measure import engine_fingerprint, verify_artifacts
            from forge_mind.workspace.models import Measurement

            measured = Measurement.model_validate(candidate.get("measurement") or {})
            if not measured.geometry_valid or measured.revision_id != rid:
                raise WorkspaceError("Feature candidate has no valid measured geometry")
            directory = self.directory(sid, rid)
            if measured.engine_fingerprint != engine_fingerprint() or not verify_artifacts(
                directory, measured
            ):
                raise WorkspaceError("Feature candidate artifacts are missing, changed or stale")
            if (
                "document.json" not in measured.artifact_hashes
                or DesignDocument.model_validate_json(
                    (directory / "document.json").read_text(encoding="utf-8")
                ).model_dump(mode="json")
                != candidate.get("document")
            ):
                raise WorkspaceError("Feature candidate source differs from its measured document")
            if candidate.get("parent_id") != basis.revision_id:
                raise WorkspaceError("Feature candidate belongs to another source revision")
            from forge_mind.workspace.models import Contract

            if (
                Contract.model_validate(candidate.get("contract") or {}).fingerprint()
                != basis.contract_hash
            ):
                raise WorkspaceError("Feature candidate belongs to another contract")
            if measured.contract_hash != basis.contract_hash:
                raise WorkspaceError("Feature candidate measurement belongs to another contract")
            if (
                admission is not None
                and successor is None
                and candidate.get("document") != payload.get("document")
            ):
                raise WorkspaceError("Feature candidate differs from the admitted merged document")
            if doc.get("authoring_mode") == "native_operations" and record.get("plan_hash") != (
                successor["plan_hash"] if successor else basis.plan_hash
            ):
                raise WorkspaceError(
                    "Feature candidate does not capture the current operation plan"
                )
            if successor is not None:
                from forge_mind.workspace.plan_history import plan_history_entry

                if (
                    candidate.get("document") != _construction.document.model_dump(mode="json")
                    or record.get("plan_snapshot") != successor
                    or record.get("plan_version") != successor["plan_version"]
                ):
                    raise WorkspaceError(
                        "Construction candidate differs from its staged source and plan"
                    )
                for name, value in (
                    ("operation-plan.json", successor),
                    ("operation-record.json", record),
                ):
                    if (
                        name not in measured.artifact_hashes
                        or json.loads((directory / name).read_text(encoding="utf-8")) != value
                    ):
                        raise WorkspaceError(
                            "Construction candidate plan artifacts differ from its saved record"
                        )
                history = list(doc.get("design_plan_history", []))
                if doc.get("design_plan"):
                    history.append(
                        plan_history_entry(con, sid, doc, head_revision_id=basis.revision_id)
                    )
                doc.update(
                    design_plan=successor["plan"],
                    plan_version=successor["plan_version"],
                    plan_hash=successor["plan_hash"],
                    plan_contract_hash=successor["contract_hash"],
                    operation_plan_snapshot=successor,
                    design_plan_history=history,
                )
            receipt = {
                "action": "construction" if successor else "feature_edit",
                "job_id": jid,
                "applied": True,
                "base_revision_id": basis.revision_id,
                "revision_id": rid,
                "plan_version": successor["plan_version"] if successor else basis.plan_version,
                "plan_applied": successor is not None,
            }
            if successor is not None:
                receipt.update(
                    cad_applied=True, candidate_revision_id=rid, plan_hash=successor["plan_hash"]
                )
            doc.update(
                selected_revision=rid,
                last_edit_result=receipt,
                failed_submission=None,
                contract=candidate["contract"],
                proposal=None,
                resume_candidate=None,
            )
            from forge_mind.workspace.input_application import FIELD, baseline

            doc["instructions"] = list(
                payload.get("target_instructions", doc.get("instructions", []))
            )
            doc[FIELD] = deepcopy(payload.get("target_contract_application") or baseline(doc))
            # Inferred author jobs can consume a staged delta after enqueue.
            # Terminal candidate recovery must retain the application that this
            # publication actually selected, not its pre-edit baseline.
            payload["target_contract_application"] = deepcopy(doc[FIELD])
            payload["target_instructions"] = list(doc["instructions"])
            if admission is not None and successor is None:
                self._promote_plan(doc, payload.get("plan_snapshot"))
            payload["feature_edit_publication"] = {
                "job_id": jid,
                "session_id": sid,
                "applied": True,
                "revision_id": rid,
                "basis": basis.model_dump(),
                "document_hash": DesignDocument.model_validate(candidate["document"]).fingerprint(),
                "contract_hash": measured.contract_hash,
                "engine_fingerprint": measured.engine_fingerprint,
                "artifact_hashes": measured.artifact_hashes,
            }
            if successor is not None:
                payload["feature_edit_publication"]["construction"] = {
                    "patch": _construction.patch.model_dump(mode="json", exclude_unset=True),
                    "base_snapshot": _construction.base_snapshot,
                    "successor_snapshot": successor,
                    "author_intent": _construction.author_intent,
                    "request_hash": _construction.request_hash,
                    "selection_version": doc.get("selection_version", 0),
                }
            con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), jid))
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
            return receipt

    @staticmethod
    def _feature_publication(
        con: sqlite3.Connection, sid: str, jid: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Read server-owned publication proof, independently of the latest action receipt."""
        from forge_mind.workspace.document import DesignDocument
        from forge_mind.workspace.edit_contracts import EditBasis
        from forge_mind.workspace.models import Contract, Measurement

        publication = payload.get("feature_edit_publication")
        if not isinstance(publication, dict) or (
            publication.get("job_id") != jid
            or publication.get("session_id") != sid
            or publication.get("applied") is not True
        ):
            return None
        row = con.execute(
            "SELECT document FROM revisions WHERE session=? AND id=?",
            (sid, publication.get("revision_id")),
        ).fetchone()
        if row is None:
            return None
        try:
            candidate = json.loads(row[0])
            basis = EditBasis.model_validate(publication.get("basis"))
            measured = Measurement.model_validate(candidate.get("measurement"))
            source_row = con.execute(
                "SELECT document FROM revisions WHERE session=? AND id=?", (sid, basis.revision_id)
            ).fetchone()
            construction = publication.get("construction")
            if source_row is None and not (construction and basis.revision_id is None):
                return None
            source = json.loads(source_row[0]) if source_row else {}
            if construction is not None:
                from forge_mind.workspace.construction_transaction import publication_matches

                if not isinstance(construction, dict) or not publication_matches(
                    construction, candidate, basis
                ):
                    return None
                session = con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()
                current = json.loads(session[0]) if session else {}
                if current.get("selection_version", 0) != construction.get("selection_version"):
                    return None
            if (
                candidate.get("id") != publication["revision_id"]
                or candidate.get("parent_id") != basis.revision_id
                or (
                    source
                    and DesignDocument.model_validate(source.get("document")).fingerprint()
                    != basis.document_hash
                )
                or DesignDocument.model_validate(candidate.get("document")).fingerprint()
                != publication.get("document_hash")
                or Contract.model_validate(candidate.get("contract")).fingerprint()
                != basis.contract_hash
                or not measured.geometry_valid
                or measured.revision_id != publication["revision_id"]
                or measured.contract_hash != basis.contract_hash
                or measured.contract_hash != publication.get("contract_hash")
                or measured.engine_fingerprint != publication.get("engine_fingerprint")
                or measured.artifact_hashes != publication.get("artifact_hashes")
                or (
                    construction is None
                    and candidate.get("operation_record")
                    and candidate["operation_record"].get("plan_hash") != basis.plan_hash
                )
            ):
                return None
        except ValueError:
            return None
        return publication

    def feature_publication(self, sid: str, owner: str, jid: str) -> dict[str, Any] | None:
        """Return only the current owned job's immutable publication identity."""
        with self.connect() as con:
            session = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
            if session is None:
                raise WorkspaceError("workspace not found")
            doc = json.loads(session[0])
            if doc.get("job_id") != jid or doc.get("authority_conflict_job_id") == jid:
                return None
            job = con.execute(
                "SELECT payload FROM jobs WHERE id=? AND session=?", (jid, sid)
            ).fetchone()
            payload = json.loads(job[0]) if job else {}
            return (
                self._feature_publication(con, sid, jid, payload)
                if job and self._preview_acceptance_matches(doc, payload)
                else None
            )

    @staticmethod
    def _publication_proposal(
        con: sqlite3.Connection, sid: str, payload: dict[str, Any], publication: dict[str, Any]
    ) -> dict[str, Any]:
        """Retain a committed candidate for review when no protection review finished."""
        row = con.execute(
            "SELECT document FROM revisions WHERE session=? AND id=?",
            (sid, publication["revision_id"]),
        ).fetchone()
        candidate = json.loads(row[0])
        return {
            "revision_id": publication["revision_id"],
            "based_on": payload.get("based_on"),
            "input_version": publication["basis"]["input_version"],
            "source_fingerprint": publication["basis"]["source_fingerprint"],
            "eligible": False,
            "protection_checks": [],
            "engine_fingerprint": publication["engine_fingerprint"],
            "contract": candidate["contract"],
            "instructions": payload.get(
                "target_instructions", payload.get("baseline_instructions", [])
            ),
            "contract_application": deepcopy(
                payload.get("target_contract_application")
                or payload.get("baseline_contract_application")
            ),
            "operation_plan_snapshot": (candidate.get("operation_record") or {}).get(
                "plan_snapshot"
            ),
            "scope": "Published geometry retained; protection review remains incomplete",
        }

    @staticmethod
    def _promote_plan(doc: dict[str, Any], binding: dict[str, Any] | None) -> None:
        """Select saved intent without decreasing the plan-write CAS counter."""
        if binding is not None:
            doc.update(
                design_plan=binding["plan"],
                plan_hash=binding["plan_hash"],
                plan_contract_hash=binding["contract_hash"],
                operation_plan_snapshot=binding,
            )

    @staticmethod
    def _preview_acceptance_identity(doc: dict[str, Any]) -> dict[str, Any]:
        """Capture exact explicit authority for comparison, never restoration."""
        return {
            key: {"present": key in doc, "value": deepcopy(doc.get(key))}
            for key in _ACCEPTANCE_FIELDS
        }

    @staticmethod
    def _preview_acceptance_matches(doc: dict[str, Any], payload: dict[str, Any]) -> bool:
        if _PREVIEW_ACCEPTANCE_IDENTITY not in payload:
            return True  # Legacy jobs cannot establish their original acceptance identity.
        identity = payload[_PREVIEW_ACCEPTANCE_IDENTITY]
        valid = (
            isinstance(identity, dict)
            and set(identity) == set(_ACCEPTANCE_FIELDS)
            and all(
                isinstance(value, dict)
                and set(value) == {"present", "value"}
                and type(value["present"]) is bool
                for value in identity.values()
            )
        )
        if not valid:
            return False
        try:
            return json.dumps(identity, sort_keys=True, allow_nan=False) == json.dumps(
                Store._preview_acceptance_identity(doc), sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _restore_preview_working(
        doc: dict[str, Any],
        payload: dict[str, Any],
        *,
        selected: str | None | object = _UNSET_REVISION,
        restore_instructions: bool = True,
        restore_plan: bool = True,
    ) -> bool:
        """Restore working state without promoting or replaying acceptance."""
        if not Store._preview_acceptance_matches(doc, payload):
            return False
        doc.update(
            selected_revision=payload.get("based_on") if selected is _UNSET_REVISION else selected,
            contract=deepcopy(payload.get("baseline_contract")),
            contract_application=deepcopy(payload.get("baseline_contract_application")),
        )
        if restore_instructions:
            doc["instructions"] = list(payload.get("baseline_instructions", []))
        if restore_plan and doc.get("authoring_mode") == "native_operations":
            Store._promote_plan(doc, payload.get("baseline_plan_snapshot"))
        return True

    @staticmethod
    def _target_snapshot(doc: dict[str, Any], target_hash: str) -> dict[str, Any] | None:
        for binding in (
            doc.get("operation_plan_snapshot"),
            (doc.get("proposal") or {}).get("operation_plan_snapshot"),
            doc.get("accepted_operation_plan_snapshot"),
        ):
            if binding and binding["contract_hash"] == target_hash:
                return dict(binding)
        return None

    def _baseline_snapshot(
        self, con: sqlite3.Connection, sid: str, doc: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Accepted geometry owns its plan even when a repair uses the same contract."""
        from forge_mind.workspace.input_application import FIELD, source
        from forge_mind.workspace.models import Contract
        from forge_mind.workspace.operation_plan import snapshot
        from forge_mind.workspace.product_sources import source_fingerprint

        rid = doc.get("accepted_revision", doc.get("selected_revision"))
        target_hash = Contract.model_validate(doc.get("contract") or {}).fingerprint()
        if rid is None:
            binding = self._target_snapshot(doc, target_hash)
        else:
            row = con.execute(
                "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
            ).fetchone()
            revision = json.loads(row[0]) if row else {}
            binding = (revision.get("operation_record") or {}).get("plan_snapshot")
            if binding is None and doc.get("accepted_operation_plan_revision") == rid:
                binding = doc.get("accepted_operation_plan_snapshot")
        if binding is not None and binding["contract_hash"] == target_hash:
            return dict(binding)
        # A prior preview can retain candidate intent while its working baseline
        # has no explicit acceptance pointer. Reuse only that exact admission tuple.
        previous = con.execute(
            "SELECT payload FROM jobs WHERE id=? AND session=?", (doc.get("job_id"), sid)
        ).fetchone()
        if previous is not None:
            payload = json.loads(previous[0])
            saved = payload.get("baseline_plan_snapshot")
            try:
                current_input = con.execute(
                    "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
                ).fetchone()
                baseline_instructions = payload.get("baseline_instructions", [])
                valid = (
                    payload.get("preview_only") is True
                    and payload.get("based_on") == rid
                    and self._preview_acceptance_matches(doc, payload)
                    and payload.get(_BASELINE_SOURCE_FINGERPRINT) == source_fingerprint(con, sid)
                    and payload.get("input_version", 0)
                    == (current_input[0] if current_input else 0)
                    and isinstance(baseline_instructions, list)
                    and doc.get("instructions", [])[: len(baseline_instructions)]
                    == baseline_instructions
                    and source(doc, len(baseline_instructions))
                    == source({**doc, "instructions": baseline_instructions})
                    and doc.get(FIELD) == payload.get("baseline_contract_application")
                    and Contract.model_validate(
                        payload.get("baseline_contract") or {}
                    ).fingerprint()
                    == target_hash
                    and isinstance(saved, dict)
                    and type(saved.get("plan_version")) is int
                    and saved.get("contract_hash") == target_hash
                    and json.dumps(
                        snapshot(saved["plan"], saved["plan_version"], saved["contract"]),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    == json.dumps(saved, sort_keys=True, allow_nan=False)
                )
                if valid:
                    return dict(deepcopy(saved))
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        if rid is not None:
            raise WorkspaceError("Accepted native revision is missing its matching frozen plan")
        if (
            doc.get("operation_plan_snapshot") is not None
            and Contract.model_validate(doc.get("contract") or {}).requirements
        ):
            raise WorkspaceError("Working native baseline is missing its matching frozen plan")
        return None

    def session_record(self, sid: str, owner: str) -> dict[str, Any]:
        """Read owner-scoped job metadata without expanding historical revisions."""
        with self.connect() as con:
            row = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
        if row is None:
            raise WorkspaceError("workspace not found")
        doc: dict[str, Any] = json.loads(row[0])
        from forge_mind.workspace.operation_recovery import descriptor

        with self.connect() as con:
            doc["operation_recovery"] = descriptor(self, con, sid, doc)
        return doc

    def get(self, sid: str, owner: str) -> dict[str, Any]:
        from forge_mind.workspace.product_sources import source_fingerprint

        doc = self.session_record(sid, owner)
        doc["revisions"] = self.revisions(sid)
        doc["engineering"] = self.engineering(sid)
        with self.connect() as con:
            doc["source_fingerprint"] = source_fingerprint(con, sid)
        return doc

    def engineering(self, sid: str) -> dict[str, Any]:
        from forge_mind.workspace.engineering_models import EngineeringInputs

        with self.connect() as con:
            row = con.execute(
                "SELECT version,document FROM engineering_inputs WHERE session=?", (sid,)
            ).fetchone()
        return (
            {"version": row[0], "inputs": json.loads(row[1])}
            if row
            else {"version": 0, "inputs": EngineeringInputs().model_dump()}
        )

    def save_engineering(self, sid: str, owner: str, version: int, inputs: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if not con.execute(
                "SELECT 1 FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone():
                raise WorkspaceError("workspace not found")
            if con.execute(
                "SELECT 1 FROM jobs WHERE session=? AND state IN ('queued','running')", (sid,)
            ).fetchone():
                raise WorkspaceError(
                    "Wait for the active operation before changing engineering inputs"
                )
            row = con.execute(
                "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
            ).fetchone()
            if (row[0] if row else 0) != version:
                raise WorkspaceError("Engineering inputs changed; refresh before saving")
            data = json.dumps(inputs, allow_nan=False)
            con.execute(
                "INSERT INTO engineering_inputs VALUES (?,?,?) ON CONFLICT(session) DO UPDATE SET version=excluded.version,document=excluded.document",
                (sid, version + 1, data),
            )
            con.execute("INSERT INTO engineering_history VALUES (?,?,?)", (sid, version + 1, data))

    def accept_proposal(
        self, sid: str, owner: str, rid: str, expected: str | None, version: int
    ) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            doc = json.loads(row[0])
            if con.execute(
                "SELECT 1 FROM jobs WHERE session=? AND state IN ('queued','running')", (sid,)
            ).fetchone():
                raise WorkspaceError("Wait for the active operation before accepting")
            current_version = con.execute(
                "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
            ).fetchone()
            proposal = doc.get("proposal") or {}
            from forge_mind.workspace.product_sources import source_fingerprint

            if proposal.get("source_fingerprint") is not None and proposal[
                "source_fingerprint"
            ] != source_fingerprint(con, sid):
                raise WorkspaceError("Sources changed; rebuild the candidate before accepting")
            if (
                (current_version[0] if current_version else 0) != version
                or proposal.get("input_version") != version
                or proposal.get("revision_id") != rid
                or proposal.get("based_on") != expected
                or doc.get("accepted_revision", doc.get("selected_revision")) != expected
            ):
                raise WorkspaceError(
                    "Candidate or acceptance baseline is stale; rebuild against the current inputs"
                )
            if not proposal.get("eligible"):
                raise WorkspaceError("Candidate failed geometry or protected-intent checks")
            if doc.get("authoring_mode") == "native_operations":
                binding = proposal.get("operation_plan_snapshot")
                if not binding:
                    raise WorkspaceError("Native candidate is missing its frozen operation plan")
                self._promote_plan(doc, binding)
                doc["accepted_operation_plan_snapshot"] = binding
                doc["accepted_operation_plan_revision"] = rid
            doc.update(
                accepted_revision=rid,
                selected_revision=rid,
                contract=proposal.get("contract", doc.get("contract")),
                instructions=proposal.get("instructions", doc.get("instructions", [])),
                contract_application=deepcopy(proposal.get("contract_application")),
                proposal=None,
            )
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

    def list_sessions(self, owner: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT document FROM sessions WHERE owner=? ORDER BY rowid DESC LIMIT 100",
                (owner,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def update(self, sid: str, **fields: Any) -> None:
        if "selection_version" in fields:
            raise WorkspaceError("Selection write version is server-owned")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            doc = json.loads(row[0])
            if "authoring_mode" in fields:
                from forge_mind.workspace.authoring_policy import validate_authoring_mode

                if validate_authoring_mode(fields["authoring_mode"]) != validate_authoring_mode(
                    doc.get("authoring_mode")
                ):
                    raise WorkspaceError("The project authoring_mode is immutable")
                # Preserve historical omission and explicit policy identity.
                fields.pop("authoring_mode")
            if "selected_revision" in fields:
                # An explicit restoration to the same ID is still a selection.
                # Recovery must distinguish it from an unstarted continuation.
                doc["selection_version"] = doc.get("selection_version", 0) + 1
            doc.update(fields)
            if "contract" in fields:
                from forge_mind.workspace.input_application import FIELD, baseline, contract_hash

                active = con.execute(
                    "SELECT id,payload FROM jobs WHERE session=? AND state='running'",
                    (sid,),
                ).fetchone()
                payload = json.loads(active["payload"]) if active else {}
                if (
                    active is not None
                    and payload.get("document") is not None
                    and payload.get("feature_edit_basis") is None
                ):
                    self._check_plan_writer(con, sid, doc, active["id"])
                    application = payload.get("target_contract_application")
                    if application is not None:
                        applied = application.get("applied") or {}
                        if applied.get("contract_hash") != contract_hash(fields["contract"]):
                            raise WorkspaceError("Typed target application binding changed")
                        doc["instructions"] = list(payload.get("target_instructions", []))
                        doc[FIELD] = deepcopy(application)
                    else:
                        doc[FIELD] = baseline(doc)
            doc.pop("revisions", None)
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

    def apply_input_contract(
        self, sid: str, owner: str, jid: str, input_id: str, contract: Any
    ) -> None:
        """Publish one validated target and its consumed input under the worker lease."""
        from forge_mind.workspace.input_application import FIELD, apply, contract_hash, source
        from forge_mind.workspace.models import Contract

        target = Contract.model_validate(
            contract.model_dump() if isinstance(contract, Contract) else contract
        )
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            doc = json.loads(row[0])
            self._check_plan_writer(con, sid, doc, jid)
            job = con.execute(
                "SELECT payload FROM jobs WHERE id=? AND session=?", (jid, sid)
            ).fetchone()
            if job is None or json.loads(job[0]).get("input_application_id") != input_id:
                raise WorkspaceError("Owner input is not bound to this job")
            application = doc.get(FIELD) or {}
            applied = application.get("applied") or {}
            if application.get("pending") is None and applied.get("input_id") == input_id:
                if (
                    applied.get("contract_hash") != target.fingerprint()
                    or contract_hash(doc.get("contract")) != target.fingerprint()
                    or source(doc) != {key: applied[key] for key in ("source_count", "source_hash")}
                ):
                    raise WorkspaceError("Applied owner input has a different target")
                return
            item = application.get("pending") or {}
            if item.get("mode") == "supplied" and (
                item.get("supplied_contract_hash") != target.fingerprint()
            ):
                raise WorkspaceError("Supplied owner target changed")
            apply(self, con, sid, doc, input_id, target)
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

    def abandon_input(
        self,
        sid: str,
        owner: str,
        input_id: str,
        expected_edit_basis: Any,
        expected_source_hash: str,
    ) -> dict[str, Any]:
        """Archive one unapplied input, retaining the accepted target and geometry."""
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            return self._abandon_input_with_connection(
                con, sid, owner, input_id, expected_edit_basis, expected_source_hash
            )

    def _abandon_input_with_connection(
        self,
        con: sqlite3.Connection,
        sid: str,
        owner: str,
        input_id: str,
        expected_edit_basis: Any,
        expected_source_hash: str,
    ) -> dict[str, Any]:
        from forge_mind.workspace.input_application import FIELD, contract_hash, pending, source

        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if row is None:
            raise WorkspaceError("workspace not found")
        if con.execute(
            "SELECT 1 FROM jobs WHERE session=? AND state IN ('queued','running')", (sid,)
        ).fetchone():
            raise WorkspaceError("Wait for the active job before abandoning owner input")
        doc = json.loads(row[0])
        self._compare_edit_basis(con, sid, doc, expected_edit_basis)
        snapshots = (doc, doc.get("proposal") or {}, doc.get("resume_candidate") or {})
        snapshot = next(
            (item for item in snapshots if (pending(item) or {}).get("input_id") == input_id),
            None,
        )
        if snapshot is None:
            raise WorkspaceError("Pending owner input identity changed")
        item = pending(snapshot)
        assert item is not None
        if item["source_hash"] != expected_source_hash:
            raise WorkspaceError("Pending owner input source changed")
        candidate_source = {**doc, **snapshot}
        if source(candidate_source) != {key: item[key] for key in ("source_count", "source_hash")}:
            raise WorkspaceError("Pending owner input source changed")
        if contract_hash(snapshot.get("contract")) != item["base_contract_hash"]:
            raise WorkspaceError("Pending owner input target changed")
        application = deepcopy(doc.get(FIELD) or snapshot.get(FIELD) or {})
        applied = application.get("applied")
        prefix_count = applied["source_count"] if applied is not None else 0
        if contract_hash(doc.get("contract")) != (
            applied["contract_hash"] if applied is not None else None
        ):
            raise WorkspaceError("Accepted owner input baseline is unavailable")
        if (
            applied is not None
            and source(doc, prefix_count)["source_hash"] != applied["source_hash"]
        ):
            raise WorkspaceError("Accepted owner input source changed")
        retained = {
            key: deepcopy(snapshot.get(key))
            for key in (
                "contract",
                "instructions",
                FIELD,
                "revision_id",
                "based_on",
                "input_version",
                "source_fingerprint",
                "plan_snapshot",
                "operation_plan_snapshot",
            )
        }
        current_record = doc.get("intent_attribution")
        doc.setdefault("abandoned_input_applications", []).append(
            {
                "input_id": input_id,
                "status": "abandoned",
                "at": time.time(),
                "snapshot": retained,
                "intent_attribution": deepcopy(current_record),
            }
        )
        for key in ("proposal", "resume_candidate"):
            if (pending(doc.get(key) or {}) or {}).get("input_id") == input_id:
                doc[key] = None
        if current_record is not None and (
            current_record["source_hash"] == item["source_hash"]
            or any(
                current_record["status"] == "answered"
                and answer["source_hash"] == current_record["source_hash"]
                and answer["question_version"] + 1 == current_record["version"]
                for answer in item["owner_answers"]
            )
        ):
            doc.setdefault("intent_attribution_history", []).append(current_record)
            doc.pop("intent_attribution", None)
        doc["instructions"] = list(doc.get("instructions", []))[:prefix_count]
        doc[FIELD] = {"applied": applied, "pending": None}
        doc["state"] = "idle"
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
        return {"input_id": input_id, "status": "abandoned", "retained_history": True}

    def append_asset(self, sid: str, owner: str, asset: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            doc = json.loads(row[0])
            assets = doc.get("assets", [])
            if len(assets) >= 100:
                raise WorkspaceError("Workspace import limit reached (100 immutable assets)")
            if any(a["id"] == asset["id"] for a in assets):
                raise WorkspaceError("Asset already registered")
            doc["assets"] = [*assets, asset]
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

    def event(self, sid: str, kind: str, message: str, **data: Any) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            doc = json.loads(row[0])
            doc["events"].append({"kind": kind, "message": message, "at": time.time(), **data})
            doc["events"] = doc["events"][-200:]
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

    def revisions(self, sid: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT document FROM revisions WHERE session=? ORDER BY rowid", (sid,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def revision(self, sid: str, rid: str) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute(
                "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
            ).fetchone()
        if row is None:
            raise WorkspaceError("revision not found")
        result: dict[str, Any] = json.loads(row[0])
        return result

    def add_revision(self, sid: str, document: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO revisions VALUES (?,?,?)", (document["id"], sid, json.dumps(document))
            )

    def directory(self, sid: str, rid: str) -> Path:
        # IDs are generated internally, but enforce the invariant at the filesystem boundary.
        if not all(len(v) == 32 and all(c in "0123456789abcdef" for c in v) for v in (sid, rid)):
            raise WorkspaceError("invalid artifact ID")
        return self.root / "sessions" / sid / "revisions" / rid

    def enqueue(self, sid: str, owner: str, payload: dict[str, Any]) -> str:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            return self._enqueue_with_connection(con, sid, owner, payload)

    def _enqueue_with_connection(
        self,
        con: sqlite3.Connection,
        sid: str,
        owner: str,
        payload: dict[str, Any],
        *,
        construction_approval: bool = False,
    ) -> str:
        """Use the same admission guards within an already held write transaction."""
        from forge_mind.workspace.input_application import (
            CONTINUATION,
            FIELD,
            SERVER_FIELDS,
            authoring,
            baseline,
            continuation,
            pending,
            stage,
        )
        from forge_mind.workspace.input_application import (
            source as input_source,
        )

        if SERVER_FIELDS.intersection(payload) or any(
            key in payload for key in (_PREVIEW_ACCEPTANCE_IDENTITY, _BASELINE_SOURCE_FINGERPRINT)
        ):
            raise WorkspaceError("Input application metadata is server-owned")
        if payload.get("resume") and (
            payload.get("instruction") or payload.get("contract") is not None
        ):
            raise WorkspaceError("Resume cannot replace the saved owner input or target")
        supplied_target = payload.get("contract") is not None
        payload = dict(payload)
        jid = uuid.uuid4().hex
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if row is None:
            raise WorkspaceError("workspace not found")
        document = json.loads(row[0])
        if "feature_edit_publication" in payload:
            raise WorkspaceError("Feature publication records are server-owned")
        if "reconcile_intent" in payload:
            raise WorkspaceError("Intent reconciliation metadata is server-owned")
        if "retry_operation" in payload:
            from forge_mind.workspace.operation_recovery import retry_payload

            if set(payload) != {"retry_operation"}:
                raise WorkspaceError("Operation rerun cannot replace the original payload")
            retry, existing = retry_payload(self, con, sid, document, payload["retry_operation"])
            if existing is not None:
                return existing
            assert retry is not None
            payload = retry
            if not self._preview_acceptance_matches(document, payload):
                raise EditConflictError("Acceptance authority changed before operation retry")
            payload.pop(_PREVIEW_ACCEPTANCE_IDENTITY, None)
            payload.pop(_BASELINE_SOURCE_FINGERPRINT, None)
        elif any(key in payload for key in ("operation_binding", "recovery_of")):
            raise WorkspaceError("Operation recovery metadata is server-owned")
        active = con.execute(
            "SELECT 1 FROM jobs WHERE session=? AND state IN ('queued','running')", (sid,)
        ).fetchone()
        if active:
            raise WorkspaceError("workspace already has an active job")
        from forge_mind.workspace.intent_attribution_service import apply_resolution

        document, payload = apply_resolution(document, payload, con, sid, owner)
        if payload.get("resume"):
            prior = con.execute(
                "SELECT payload FROM jobs WHERE id=? AND session=?", (document.get("job_id"), sid)
            ).fetchone()
            if prior is not None:
                previous_payload = json.loads(prior[0])
                for key in ("max_seconds", "max_actions"):
                    if payload.get(key) is None and previous_payload.get(key) is not None:
                        payload[key] = previous_payload[key]
        if document.get("customer_hosted_limits"):
            for key, maximum in document["customer_hosted_limits"].items():
                supplied = payload.get(key)
                payload[key] = maximum if supplied is None else min(supplied, maximum)
        answer_binding = payload.get("intent_answer_binding")
        proposal_input = document.get("proposal") or {}
        interrupted_input = document.get("resume_candidate") or {}
        if (
            authoring(payload)
            and not supplied_target
            and not document.get(FIELD)
            and document.get("contract") is not None
            and document.get("state") in {"error", "cancelled", "budget_exhausted"}
        ):
            prior = con.execute(
                "SELECT payload FROM jobs WHERE id=? AND session=?",
                (document.get("job_id"), sid),
            ).fetchone()
            prior_payload = json.loads(prior[0]) if prior else {}
            if authoring(prior_payload) and (
                document.get("instructions")
                or prior_payload.get("instruction")
                or prior_payload.get("resume_instructions")
            ):
                raise WorkspaceError(
                    "Legacy interrupted owner input needs an explicit supplied target"
                )
        if (
            authoring(payload)
            and not supplied_target
            and not interrupted_input.get(FIELD)
            and interrupted_input.get("contract") is not None
        ):
            prior = con.execute(
                "SELECT payload FROM jobs WHERE id=? AND session=?",
                (document.get("job_id"), sid),
            ).fetchone()
            prior_payload = json.loads(prior[0]) if prior else {}
            if (
                interrupted_input.get("instructions")
                or prior_payload.get("instruction")
                or prior_payload.get("resume_instructions")
                or (
                    prior_payload.get("contract") is not None
                    and prior_payload["contract"] != interrupted_input["contract"]
                )
            ):
                raise WorkspaceError(
                    "Legacy interrupted owner input needs an explicit supplied target"
                )
        outstanding = next(
            (
                snapshot
                for snapshot in (document, proposal_input, interrupted_input)
                if pending(snapshot) is not None
            ),
            None,
        )
        if outstanding is not None and not payload.get("resume") and answer_binding is None:
            raise WorkspaceError("Resume the pending owner input before starting another edit")
        input_snapshot = deepcopy(document)
        input_location = None
        input_origin = "supplied" if supplied_target else "existing"
        if answer_binding is not None and interrupted_input:
            from forge_mind.workspace.models import Contract

            checkpoint_source = input_source({**document, **interrupted_input})
            question_basis = answer_binding["basis"]
            if (
                checkpoint_source["source_hash"] != answer_binding["source_hash"]
                or interrupted_input.get("input_version") != question_basis["input_version"]
                or interrupted_input.get("source_fingerprint")
                != question_basis["source_fingerprint"]
            ):
                raise WorkspaceError("Answered checkpoint source or engineering basis changed")
            checkpoint_plan = interrupted_input.get("plan_snapshot") or {}
            checkpoint_contract = Contract.model_validate(interrupted_input.get("contract") or {})
            self._compare_edit_basis(
                con,
                sid,
                document,
                question_basis,
                rid=interrupted_input.get("revision_id"),
                target={
                    "contract": checkpoint_contract.model_dump(),
                    "contract_hash": checkpoint_contract.fingerprint(),
                    "plan_hash": checkpoint_plan.get("plan_hash", document.get("plan_hash")),
                },
            )
            input_snapshot = deepcopy(interrupted_input)
            input_location = "resume_candidate"
            input_origin = "inherited"
            payload["revision_id"] = interrupted_input.get("revision_id")
        elif (
            payload.get("revision_id")
            and payload["revision_id"] == proposal_input.get("revision_id")
            and proposal_input.get("contract") is not None
        ):
            input_snapshot = deepcopy(proposal_input)
            input_location = "proposal"
            input_origin = "supplied" if supplied_target else "inherited"
        if "feature_edit_admission" in payload or "feature_edit_original" in payload:
            raise WorkspaceError("Feature edit admission is server-owned")
        if "feature_edit_basis" in payload:
            self._compare_edit_basis(con, sid, document, payload["feature_edit_basis"])
        design_head = (document.get("proposal") or {}).get("revision_id") or document.get(
            "selected_revision"
        )
        if (
            "expected_current_revision" in payload
            and payload["expected_current_revision"] != design_head
        ):
            raise WorkspaceError(
                "Design changed before admission; read the current revision and retry"
            )
        if payload.get("resume"):
            from forge_mind.workspace.operation_recovery import operation
            from forge_mind.workspace.product_sources import source_fingerprint

            last = con.execute(
                "SELECT payload,state FROM jobs WHERE id=? AND session=?",
                (document.get("job_id"), sid),
            ).fetchone()
            if last and operation(json.loads(last[0])):
                if not construction_approval:
                    raise WorkspaceError(
                        "An interrupted typed or engineering operation cannot resume as authoring; use its explicit operation action"
                    )
                from forge_mind.workspace.customer_construction import approval_binding
                from forge_mind.workspace.customer_projects import customer_project

                record = document.get("customer_proposal") or {}
                if (
                    not customer_project(document)
                    or document.get("approval_required") is not True
                    or record.get("job_id") != document.get("job_id")
                    or record.get("status") != "pending"
                    or document.get("state") != "needs_approval"
                    or approval_binding(self, con, sid, owner, document, record) is None
                ):
                    raise WorkspaceError("Typed approval continuation lost its exact command")
                # Construct only the existing native checkpoint of the unchanged
                # real predecessor. The proposed new CAD does not exist yet.
                self._checkpoint(con, sid, document, payload)

            checkpoint = document.get("resume_candidate")
            if (
                not checkpoint
                and last is not None
                and last[1] == "needs_review"
                and document.get("state") == "needs_review"
                and document.get("contract") is None
                and document.get("selected_revision") is None
                and not document.get("accepted_revision")
                and not document.get("proposal")
                and pending(document) is not None
                and self._native_author_job(document, json.loads(last[0]))
                and con.execute("SELECT 1 FROM revisions WHERE session=?", (sid,)).fetchone()
                is None
            ):
                previous_payload = json.loads(last[0])
                item = pending(document)
                assert item is not None
                if (
                    not item.get("input_id")
                    or previous_payload.get("input_application_id") != item["input_id"]
                    or previous_payload.get("input_instructions")
                    != document.get("instructions", [])
                ):
                    raise WorkspaceError("Pending intake owner input changed before recovery")
                self._compare_edit_basis(con, sid, document, item["edit_basis"], rid=None)
                # Older terminal intake jobs omitted this checkpoint. Reconstruct
                # only their exact pending server-owned input, never a CAD revision.
                self._checkpoint(con, sid, document, previous_payload)
                checkpoint = document.get("resume_candidate")
            if not checkpoint:
                raise WorkspaceError("No interrupted candidate is available to resume")
            if not checkpoint.get(FIELD) and checkpoint.get("contract") is not None:
                previous_payload = json.loads(last[0]) if last else {}
                if previous_payload.get("instruction") or (
                    previous_payload.get("contract") is not None
                    and previous_payload["contract"] != checkpoint["contract"]
                ):
                    raise WorkspaceError(
                        "Legacy interrupted owner input has no application binding; start an explicit edit"
                    )
            native_binding = checkpoint.get("native_author_binding")
            if native_binding is not None:
                current_binding = self._native_author_binding(document)
                if not checkpoint.get(FIELD) and "contract_application_hash" not in native_binding:
                    current_binding.pop("contract_application_hash", None)
                if document.get("authoring_mode") != "native_operations" or (
                    native_binding != current_binding
                ):
                    raise WorkspaceError(
                        "Saved native author selection, acceptance or intent changed; start an explicit edit"
                    )
                if (
                    payload.get("instruction")
                    or payload.get("parameters")
                    or payload.get("restore_only")
                    or payload.get("document") is not None
                    or payload.get("engineering_operation")
                    or payload.get("code")
                ):
                    raise WorkspaceError("Native resume cannot include a new edit or operation")
                # The API's historical preview flag must not adopt an unfinished
                # non-preview candidate as an accepted baseline.
                payload = {
                    **payload,
                    "preview_only": False,
                    "delivery_requested": checkpoint.get("delivery_requested") is True,
                }
            current_input = con.execute(
                "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
            ).fetchone()
            if (
                (
                    native_binding is None
                    and checkpoint["based_on"]
                    != document.get("accepted_revision", document.get("selected_revision"))
                )
                or checkpoint["input_version"] != (current_input[0] if current_input else 0)
                or checkpoint["source_fingerprint"] != source_fingerprint(con, sid)
            ):
                raise WorkspaceError(
                    "Resume baseline or engineering sources changed; start an explicit edit of the saved revision"
                )
            payload = {
                **payload,
                "revision_id": checkpoint["revision_id"],
                "contract": checkpoint["contract"],
                "resume_instructions": checkpoint["instructions"],
                "resume_contract_application": deepcopy(checkpoint.get(FIELD)),
            }
            input_snapshot = deepcopy(checkpoint)
            input_location = "resume_candidate"
            input_origin = "inherited"
        if self._native_author_job(document, payload):
            # Capture admission identity, not the version observed after an
            # interruption. Sources retain the existing checkpoint guard.
            current_input = con.execute(
                "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
            ).fetchone()
            payload = {**payload, "input_version": current_input[0] if current_input else 0}
        if "expected_input_version" in payload:
            current_input = con.execute(
                "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
            ).fetchone()
            if payload["expected_input_version"] != (current_input[0] if current_input else 0):
                raise WorkspaceError(
                    "Engineering inputs changed before admission; refresh and retry"
                )
        if payload.get("preview_only"):
            from forge_mind.workspace.product_sources import source_fingerprint

            payload = dict(payload)
            payload[_PREVIEW_ACCEPTANCE_IDENTITY] = self._preview_acceptance_identity(document)
            payload[_BASELINE_SOURCE_FINGERPRINT] = source_fingerprint(con, sid)
            proposal = document.get("proposal") or {}
            if (
                payload.get("contract") is None
                and payload.get("revision_id")
                and payload["revision_id"] == proposal.get("revision_id")
                and proposal.get("contract") is not None
            ):
                # Preview completion restores the accepted contract on the
                # session. Continue this specific candidate's full contract
                # before clearing its proposal, without adopting old targets
                # from an arbitrary historical editing source.
                payload["contract"] = proposal["contract"]
            payload["based_on"] = document.get(
                "accepted_revision", document.get("selected_revision")
            )
            payload["baseline_contract"] = document.get("contract")
            payload["baseline_instructions"] = document.get("instructions", [])
            payload["baseline_contract_application"] = deepcopy(
                document.get(FIELD) or baseline(document)
            )
            input_row = con.execute(
                "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
            ).fetchone()
            payload["input_version"] = input_row[0] if input_row else 0
            if (
                "expected_input_version" in payload
                and payload["expected_input_version"] != payload["input_version"]
            ):
                raise WorkspaceError(
                    "Engineering inputs changed before admission; recompute the pose"
                )
        if document.get("authoring_mode") == "native_operations":
            from forge_mind.workspace.authoring_policy import enforce_document_policy
            from forge_mind.workspace.document import DesignDocument
            from forge_mind.workspace.models import Contract, DesignPlan
            from forge_mind.workspace.operation_plan import validate_plan

            if payload.get("code"):
                raise WorkspaceError("Native operations reject authored Python code")
            if "plan_snapshot" in payload:
                raise WorkspaceError("Operation plan admission snapshots are server-owned")
            # Read-only engineering probes do not require construction intent.
            geometry_write = payload.get("document") is not None or (
                not payload.get("engineering_operation")
                and any(payload.get(key) for key in ("parameters", "restore_only"))
            )
            proposal = document.get("proposal") or {}
            continuing_candidate = bool(
                payload.get("revision_id") and payload["revision_id"] == proposal.get("revision_id")
            )
            active_contract = Contract.model_validate(
                (
                    (document.get("resume_candidate") or {}).get("contract")
                    if payload.get("resume")
                    else None
                )
                or (proposal.get("contract") if continuing_candidate else None)
                or document.get("contract")
                or {}
            )
            if payload.get("contract") is not None:
                target = Contract.model_validate(payload["contract"])
                if target.fingerprint() != active_contract.fingerprint():
                    from forge_mind.workspace.measure import merge_contract_edit

                    instruction = payload.get("instruction", "")
                    if (
                        geometry_write
                        or not instruction
                        or merge_contract_edit(active_contract, target, instruction).fingerprint()
                        != target.fingerprint()
                    ):
                        raise WorkspaceError(
                            "Contract is frozen; use an explicit customer requirement edit"
                        )
                    payload = {
                        **payload,
                        "contract_change": {
                            "previous_contract_hash": active_contract.fingerprint(),
                            "contract_hash": target.fingerprint(),
                            "source_instruction": instruction,
                        },
                    }
            if geometry_write:
                target = active_contract
                captured = (
                    (document.get("resume_candidate") or {}).get("plan_snapshot")
                    if payload.get("resume")
                    else None
                ) or self._target_snapshot(document, target.fingerprint())
                if payload.get("feature_edit_basis") is not None:
                    captured = self._edit_target_snapshot(document, target.fingerprint())
                    if (
                        not captured
                        or captured.get("plan_hash") != payload["feature_edit_basis"]["plan_hash"]
                    ):
                        raise EditConflictError(
                            "Operation plan changed before focused edit admission"
                        )
                raw_plan = captured["plan"] if captured else None
                plan = DesignPlan.model_validate(raw_plan) if raw_plan else None
                if (
                    plan is None
                    or plan.operation_plan is None
                    or not plan.operation_plan.steps
                    or not target.requirements
                    or captured is None
                    or captured["contract_hash"] != target.fingerprint()
                ):
                    raise WorkspaceError(
                        "Initialize the operation plan and frozen contract before construction"
                    )
                resolved = None
                if payload.get("document") is not None:
                    resolved = DesignDocument.model_validate(payload["document"])
                    enforce_document_policy(resolved, "native_operations")
                elif payload.get("revision_id"):
                    revision = con.execute(
                        "SELECT document FROM revisions WHERE session=? AND id=?",
                        (sid, payload["revision_id"]),
                    ).fetchone()
                    if revision is None:
                        raise WorkspaceError("revision not found")
                    source = json.loads(revision[0])
                    # The selected source is immutable on disk. Restore/build
                    # policy also checks it at the common worker boundary.
                    source_path = self.directory(sid, source["id"]) / "document.json"
                    if not source_path.is_file():
                        raise WorkspaceError("Native operation source document is unavailable")
                    resolved = DesignDocument.model_validate_json(
                        source_path.read_text(encoding="utf-8")
                    )
                    enforce_document_policy(resolved, "native_operations")
                validate_plan(plan, target, document=resolved)
                payload = {
                    **payload,
                    "contract": target.model_dump(),
                    "plan_snapshot": captured,
                }
            if payload.get("preview_only"):
                payload = {
                    **payload,
                    "baseline_plan_snapshot": self._baseline_snapshot(con, sid, document),
                }
        # Admission is shared across web workers, unlike a process-local semaphore.
        if payload.get("document") is not None:
            target_snapshot = {**document, **input_snapshot}
            if payload.get("contract") is not None:
                target_snapshot["contract"] = payload["contract"]
            target_application = target_snapshot.get(FIELD)
            from forge_mind.workspace.input_application import contract_hash

            if target_application is None or (target_application.get("applied") or {}).get(
                "contract_hash"
            ) != contract_hash(target_snapshot.get("contract")):
                target_application = baseline(target_snapshot)
            payload["target_contract_application"] = deepcopy(target_application)
            payload["target_instructions"] = list(target_snapshot.get("instructions", []))
        pending_count = con.execute(
            "SELECT count(*) FROM jobs WHERE state IN ('queued','running')"
        ).fetchone()[0]
        if pending_count >= 16:
            raise WorkspaceError("workspace queue is full")
        admitted = (
            continuation(document, input_snapshot, input_location)
            if authoring(payload) and input_location is not None
            else None
        )
        if payload.get("resume") and not construction_approval:
            from forge_mind.workspace.customer_construction import resume_binding

            prior = con.execute(
                "SELECT payload FROM jobs WHERE id=? AND session=?", (document.get("job_id"), sid)
            ).fetchone()
            if prior is not None:
                resume_binding(con, document, json.loads(prior[0]), payload)
        if admitted is not None:
            payload.setdefault("input_version", input_snapshot.get("input_version", 0))
        if not payload.get("recovery_of"):
            from forge_mind.workspace.operation_recovery import bind_admission

            payload = bind_admission(con, sid, document, payload)
        con.execute(
            "INSERT INTO jobs(id,session,state,payload,recovery_of) VALUES (?,?,'queued',?,?)",
            (jid, sid, json.dumps(payload), payload.get("recovery_of")),
        )
        doc = document
        doc.update(state="queued", job_id=jid)
        if (payload.get("document") or payload.get("engineering_operation")) and payload.get(
            "feature_edit_basis"
        ) is None:
            doc["resume_candidate"] = None
        if payload.get("engineering_operation") == "product_tool":
            doc["last_product_probe"] = {
                "job_id": jid,
                "check_id": payload["check_id"],
                "revision_id": payload.get("revision_id"),
            }
        if payload.get("engineering_operation") == "structural_study":
            doc["last_structural_study"] = {
                "job_id": jid,
                "study_id": payload["study_id"],
                "revision_id": payload["revision_id"],
            }
        if payload.get("preview_only") and payload.get("feature_edit_basis") is None:
            doc["proposal"] = None
        if authoring(payload):
            payload.setdefault(
                "based_on", document.get("accepted_revision", document.get("selected_revision"))
            )
            doc["contract"] = input_snapshot.get("contract")
            doc["instructions"] = list(input_snapshot.get("instructions", []))
            doc[FIELD] = deepcopy(input_snapshot.get(FIELD) or baseline(doc))
            basis_document = None
            if admitted is not None:
                target_plan = input_snapshot.get("plan_snapshot") or input_snapshot.get(
                    "operation_plan_snapshot"
                )
                if doc.get("authoring_mode") == "native_operations" and target_plan:
                    self._promote_plan(doc, target_plan)
                basis_document = deepcopy(doc)
                for name in admitted["predecessors"]:
                    basis_document[name] = None
            stage(
                self,
                con,
                sid,
                jid,
                doc,
                payload,
                origin=input_origin,
                answer_binding=answer_binding,
                basis_document=basis_document,
            )
            if admitted is not None:
                from forge_mind.workspace.product_sources import source_fingerprint

                for name in admitted["predecessors"]:
                    doc[name] = None
                continuation_target = admitted["target"]
                continuation_target.update(
                    contract=deepcopy(doc.get("contract")),
                    instructions=list(doc.get("instructions", [])),
                    contract_application=deepcopy(doc[FIELD]),
                )
                continuation_target.setdefault("input_version", payload.get("input_version", 0))
                continuation_target.setdefault("source_fingerprint", source_fingerprint(con, sid))
                continuation_target["plan_snapshot"] = deepcopy(
                    input_snapshot.get("plan_snapshot")
                    or input_snapshot.get("operation_plan_snapshot")
                )
                payload[CONTINUATION] = admitted
            con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), jid))
        if "feature_edit_basis" in payload:
            payload["feature_edit_admission"] = self._native_author_binding(doc)
            con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), jid))
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
        return jid

    @staticmethod
    def _owner_dead(row: sqlite3.Row) -> bool:
        identity = (row["worker_pid"], row["worker_created"], row["worker_host"])
        if all(value is None for value in identity):
            # Only pre-migration jobs use the historical recovery deadline.
            return row["deadline"] is not None and row["deadline"] < time.time()
        if any(value is None for value in identity) or row["worker_host"] != socket.gethostname():
            return False
        try:
            process = psutil.Process(row["worker_pid"])
            return bool(
                process.create_time() != row["worker_created"]
                or not process.is_running()
                or process.status() == psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            return True
        except (psutil.AccessDenied, OSError, ValueError):
            # Unavailable process evidence cannot authorize taking over healthy work.
            return False

    def _recover_interrupted_jobs(
        self,
        con: sqlite3.Connection,
        *,
        fenced_jobs: frozenset[str] = frozenset(),
        fenced_host: str | None = None,
    ) -> None:
        """Reuse exact checkpoint transitions after process death or offline host fencing."""
        # Work budgets are not ownership leases. Only confirmed owner death
        # interrupts a new job; recovery never silently re-authors saved work.
        expired = con.execute(
            "SELECT * FROM jobs WHERE state='running'",
        ).fetchall()
        for row in expired:
            if fenced_host is None:
                dead = self._owner_dead(row)
            else:
                dead = row["id"] in fenced_jobs and row["worker_host"] == fenced_host
            if not dead:
                continue
            con.execute("UPDATE jobs SET state='interrupted' WHERE id=?", (row["id"],))
            doc = json.loads(
                con.execute(
                    "SELECT document FROM sessions WHERE id=?", (row["session"],)
                ).fetchone()[0]
            )
            if doc.get("job_id") != row["id"]:
                continue
            doc["state"] = "interrupted"
            expired_payload = json.loads(row["payload"])
            focused_edit = expired_payload.get("feature_edit_basis") is not None
            authority_conflict = doc.pop("authority_conflict_job_id", None) == row[
                "id"
            ] or not self._preview_acceptance_matches(doc, expired_payload)
            if authority_conflict:
                doc.pop("pending_finish", None)
            publication = (
                None
                if authority_conflict
                else self._feature_publication(con, row["session"], row["id"], expired_payload)
            )
            if focused_edit and publication is None and not authority_conflict:
                doc["last_edit_result"] = {
                    "action": "feature_edit",
                    "job_id": row["id"],
                    "applied": False,
                    "code": "interrupted",
                    "base_revision_id": expired_payload["feature_edit_basis"]["revision_id"],
                }
            if (
                expired_payload.get("preview_only")
                and not authority_conflict
                and (doc.get("intent_attribution") or {}).get("status") == "needs_input"
            ):
                self._pause_intent_preview(con, row["session"], doc, expired_payload)
                doc.pop("pending_finish", None)
            elif publication is not None and expired_payload.get("preview_only"):
                doc.update(
                    proposal=self._publication_proposal(
                        con, row["session"], expired_payload, publication
                    ),
                    resume_candidate=None,
                )
                self._restore_preview_working(doc, expired_payload, restore_plan=False)
                doc.pop("pending_finish", None)
            elif publication is not None:
                doc["selected_revision"] = publication["revision_id"]
                self._checkpoint(con, row["session"], doc, expired_payload)
            elif (
                expired_payload.get("preview_only") and not focused_edit and not authority_conflict
            ):
                self._checkpoint(con, row["session"], doc, expired_payload)
                doc["proposal"] = None
                self._restore_preview_working(doc, expired_payload)
                doc.pop("pending_finish", None)
            elif (
                self._native_author_job(doc, expired_payload)
                or (doc.get("contract_application") or {}).get("pending") is not None
                or expired_payload.get("input_application_id")
                or expired_payload.get("admitted_input_continuation")
            ) and not authority_conflict:
                self._checkpoint(con, row["session"], doc, expired_payload)
            if (
                (expired_payload.get("document") or expired_payload.get("engineering_operation"))
                and not focused_edit
                and publication is None
                and not authority_conflict
            ):
                doc["resume_candidate"] = None
            con.execute(
                "UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), row["session"])
            )

    def _claim_allowed(self, con: sqlite3.Connection) -> bool:
        """Explicit hosting policies can fence claims under the original transaction."""
        return True

    def claim(self) -> dict[str, Any] | None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if not self._claim_allowed(con):
                return None
            self._recover_interrupted_jobs(con)
            if con.execute("SELECT 1 FROM jobs WHERE state='running'").fetchone():
                return None
            row = con.execute(
                "SELECT * FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            payload = json.loads(row["payload"])
            document_row = con.execute(
                "SELECT document FROM sessions WHERE id=?", (row["session"],)
            ).fetchone()
            document = json.loads(document_row[0])
            if not self._preview_acceptance_matches(document, payload):
                con.execute("UPDATE jobs SET state='needs_review' WHERE id=?", (row["id"],))
                if document.get("job_id") == row["id"]:
                    document.update(state="needs_review")
                    document.pop("pending_finish", None)
                    con.execute(
                        "UPDATE sessions SET document=? WHERE id=?",
                        (json.dumps(document), row["session"]),
                    )
                return None
            started = time.time()
            seconds = payload.get("max_seconds")
            deadline = None if seconds is None else started + seconds
            pid = os.getpid()
            created = psutil.Process(pid).create_time()
            con.execute(
                "UPDATE jobs SET state='running',started=?,deadline=?,"
                "worker_pid=?,worker_created=?,worker_host=? WHERE id=?",
                (started, deadline, pid, created, socket.gethostname(), row["id"]),
            )
            owner = con.execute(
                "SELECT owner FROM sessions WHERE id=?", (row["session"],)
            ).fetchone()[0]
            # Persisted work fields cannot replace the database-owned job identity.
            return {**payload, "id": row["id"], "session": row["session"], "owner": owner}

    def cancelled(self, jid: str) -> bool:
        with self.connect() as con:
            row = con.execute("SELECT cancel,state FROM jobs WHERE id=?", (jid,)).fetchone()
        return row is None or bool(row[0]) or row[1] != "running"

    def cancel(self, sid: str, owner: str) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
            ).fetchone()
            if row is None:
                raise WorkspaceError("workspace not found")
            running = con.execute(
                "SELECT 1 FROM jobs WHERE session=? AND state='running'", (sid,)
            ).fetchone()
            queued = con.execute(
                "SELECT id,payload FROM jobs WHERE session=? AND state='queued'", (sid,)
            ).fetchone()
            con.execute("UPDATE jobs SET cancel=1 WHERE session=? AND state='running'", (sid,))
            con.execute(
                "UPDATE jobs SET state='cancelled' WHERE session=? AND state='queued'", (sid,)
            )
            if running or queued:
                doc = json.loads(row[0])
                doc["state"] = "cancelling" if running else "cancelled"
                if queued:
                    payload = json.loads(queued["payload"])
                    authority_conflict = doc.get("authority_conflict_job_id") == queued[
                        "id"
                    ] or not self._preview_acceptance_matches(doc, payload)
                    if not authority_conflict and (
                        payload.get("input_application_id")
                        or payload.get("admitted_input_continuation")
                    ):
                        self._checkpoint(con, sid, doc, payload)
                        if payload.get("preview_only"):
                            doc["proposal"] = None
                            self._restore_preview_working(doc, payload)
                    if not authority_conflict and payload.get("feature_edit_basis") is not None:
                        doc["last_edit_result"] = {
                            "action": "feature_edit",
                            "job_id": queued["id"],
                            "applied": False,
                            "code": "cancelled",
                            "base_revision_id": payload["feature_edit_basis"]["revision_id"],
                        }
                con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

            if not running and not queued:
                doc = json.loads(row[0])
                if doc.get("state") == "needs_approval":
                    doc["state"] = "cancelled"
                    if doc.get("customer_proposal"):
                        doc["customer_proposal"]["status"] = "stale"
                    con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

    def finish(
        self, jid: str, sid: str, state: str, *, verification_report: dict[str, Any] | None = None
    ) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT state,cancel,payload,deadline FROM jobs WHERE id=? AND session=?",
                (jid, sid),
            ).fetchone()
            if row is None or row[0] != "running":
                return
            state = "cancelled" if row[1] else state
            doc = json.loads(
                con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
            )
            payload = json.loads(row[2])
            from forge_mind.workspace.requirement_verification import (
                checked_report,
                has_verification,
            )

            typed = has_verification(doc.get("contract"))
            authenticated_proof = False
            if verification_report is not None or (typed and state == "contract_satisfied"):
                try:
                    if (
                        verification_report is None
                        or doc.get("job_id") != jid
                        or verification_report.get("job_id") != jid
                        or payload.get("preview_only")
                        or row[1]
                        or (row[3] is not None and time.time() >= row[3])
                    ):
                        raise WorkspaceError("Exact current final proof is required")
                    report = checked_report(self, con, sid, doc, verification_report)
                    if row[3] is not None and time.time() >= row[3]:
                        raise WorkspaceError(
                            "Original job deadline expired during final proof readback"
                        )
                    doc["contract_verification_report"] = verification_report
                    doc.pop("contract_verification_error", None)
                    if state in {"needs_review", "contract_satisfied"}:
                        state = (
                            "contract_satisfied"
                            if report["requirements_satisfied"]
                            and report["author_closure_ready"] is True
                            and doc.get("incomplete_finish_job_id") != jid
                            and doc.get("authority_conflict_job_id") != jid
                            else "needs_review"
                        )
                    # Retain proof-bound failure markers only after every publication
                    # check succeeds; rejected caller metadata keeps legacy cleanup.
                    authenticated_proof = True
                except (ValueError, OSError, KeyError, TypeError, WorkspaceError) as exc:
                    if state == "contract_satisfied":
                        state = "needs_review"
                    if doc.get("job_id") == jid:
                        doc["contract_verification_error"] = str(exc)[:1000]
                        doc["contract_verification_report"] = None
            if state == "needs_approval" and payload.get("preview_only"):
                from forge_mind.workspace.customer_construction import approval_binding
                from forge_mind.workspace.customer_projects import customer_project

                record = doc.get("customer_proposal") or {}
                self._check_plan_writer(con, sid, doc, jid)
                owner = con.execute("SELECT owner FROM sessions WHERE id=?", (sid,)).fetchone()[0]
                if (
                    not customer_project(doc)
                    or doc.get("approval_required") is not True
                    or record.get("job_id") != jid
                    or record.get("status") != "pending"
                    or approval_binding(self, con, sid, owner, doc, record) is None
                ):
                    raise WorkspaceError(
                        "Preview approval pause requires the current exact command"
                    )
                doc.pop("pending_finish", None)
                doc["state"] = state
                con.execute("UPDATE jobs SET state=? WHERE id=?", (state, jid))
                con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
                return
            if payload.get("preview_only"):
                if doc.get("job_id") != jid:
                    con.execute("UPDATE jobs SET state=? WHERE id=?", (state, jid))
                    return
                if not self._preview_acceptance_matches(doc, payload):
                    terminal = (
                        state
                        if state in {"cancelled", "error", "budget_exhausted"}
                        else "needs_review"
                    )
                    doc.update(state=terminal)
                    doc.pop("pending_finish", None)
                    con.execute("UPDATE jobs SET state=? WHERE id=?", (terminal, jid))
                    con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
                    return
                # Keep the running lease until candidate/protection/baseline state is atomic.
                doc["pending_finish"] = state
                con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
                return
            con.execute("UPDATE jobs SET state=? WHERE id=?", (state, jid))
            if doc["job_id"] == jid:
                payload = json.loads(row[2])
                authority_conflict = doc.get("authority_conflict_job_id") == jid
                if authority_conflict and not authenticated_proof:
                    doc.pop("authority_conflict_job_id", None)
                if (
                    self._native_author_job(doc, payload)
                    or (doc.get("contract_application") or {}).get("pending") is not None
                    or payload.get("input_application_id")
                    or payload.get("admitted_input_continuation")
                ) and not authority_conflict:
                    incomplete_finish = doc.get("incomplete_finish_job_id") == jid
                    if incomplete_finish and not authenticated_proof:
                        doc.pop("incomplete_finish_job_id", None)
                    publication = self._feature_publication(con, sid, jid, payload)
                    if publication is not None and state in {
                        "cancelled",
                        "error",
                        "budget_exhausted",
                    }:
                        doc["selected_revision"] = publication["revision_id"]
                    if state in {
                        "cancelled",
                        "error",
                        "budget_exhausted",
                        "needs_input",
                        "needs_approval",
                    } or (
                        state == "needs_review"
                        and (
                            incomplete_finish
                            or (doc.get("contract_application") or {}).get("pending") is not None
                        )
                    ):
                        self._checkpoint(con, sid, doc, payload)
                    else:
                        doc["resume_candidate"] = None
                doc["state"] = state
                con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

    def complete_preview(
        self, sid: str, jid: str, based_on: str | None, proposal: dict[str, Any] | None
    ) -> None:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT state,cancel,payload FROM jobs WHERE id=? AND session=?", (jid, sid)
            ).fetchone()
            if row is None or row[0] != "running":
                return
            doc = json.loads(
                con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
            )
            if doc["job_id"] != jid:
                return
            pending = doc.pop("pending_finish", "needs_review")
            state = "cancelled" if row[1] else pending
            payload = json.loads(row[2])
            acceptance_conflict = not self._preview_acceptance_matches(doc, payload)
            authority_conflict = (
                doc.pop("authority_conflict_job_id", None) == jid or acceptance_conflict
            )
            if acceptance_conflict and state not in {"cancelled", "error", "budget_exhausted"}:
                state = "needs_review"
            publication = (
                None if authority_conflict else self._feature_publication(con, sid, jid, payload)
            )
            if authority_conflict or (
                payload.get("feature_edit_basis") is not None and publication is None
            ):
                # A rejected basis never authorizes terminal baseline restoration.
                doc.update(state=state)
                con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
                con.execute("UPDATE jobs SET state=? WHERE id=?", (state, jid))
                return
            if state == "needs_input" or (
                (doc.get("intent_attribution") or {}).get("status") == "needs_input"
            ):
                self._pause_intent_preview(con, sid, doc, payload)
                doc["state"] = state
                doc.pop("incomplete_finish_job_id", None)
                con.execute("UPDATE jobs SET state=? WHERE id=?", (state, jid))
                con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
                return
            incomplete_finish = doc.pop("incomplete_finish_job_id", None) == jid
            interrupted = state in {"cancelled", "error", "budget_exhausted"}
            if (
                publication is not None
                and (proposal is None or interrupted)
                and (proposal is None or proposal.get("revision_id") != publication["revision_id"])
            ):
                proposal = self._publication_proposal(con, sid, payload, publication)
            if interrupted and publication is None:
                self._checkpoint(con, sid, doc, payload)
                proposal = None
            elif state == "needs_review" and (
                incomplete_finish
                or (doc.get("contract_application") or {}).get("pending") is not None
            ):
                self._checkpoint(con, sid, doc, payload)
            else:
                doc["resume_candidate"] = None
            if proposal is not None:
                from forge_mind.workspace.input_application import copy_to_proposal

                copy_to_proposal(doc, proposal)
            doc.update(
                state=state,
                proposal=proposal,
            )
            payload = json.loads(row[2])
            if doc.get("authoring_mode") == "native_operations":
                from forge_mind.workspace.models import Contract

                if proposal is not None:
                    candidate_row = con.execute(
                        "SELECT document FROM revisions WHERE session=? AND id=?",
                        (sid, proposal["revision_id"]),
                    ).fetchone()
                    candidate = json.loads(candidate_row[0]) if candidate_row else {}
                    candidate_binding = (candidate.get("operation_record") or {}).get(
                        "plan_snapshot"
                    )
                    target_hash = Contract.model_validate(proposal["contract"]).fingerprint()
                    candidate_binding = candidate_binding or payload.get("plan_snapshot")
                    if not candidate_binding or candidate_binding["contract_hash"] != target_hash:
                        raise WorkspaceError("Native candidate is missing its matching frozen plan")
                    proposal = {**proposal, "operation_plan_snapshot": candidate_binding}
                    doc["proposal"] = proposal
            self._restore_preview_working(
                doc, payload, selected=based_on, restore_plan=proposal is None
            )
            con.execute("UPDATE jobs SET state=? WHERE id=?", (state, jid))
            con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))

    @staticmethod
    def _pause_intent_preview(
        con: sqlite3.Connection, sid: str, doc: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        """Retain the questioned draft without accepting it or replacing its source."""
        if not Store._preview_acceptance_matches(doc, payload):
            return
        current_proposal = doc.get("proposal") or {}
        head = current_proposal.get("revision_id") or doc.get("selected_revision")
        target = current_proposal.get("contract") or doc.get("contract")
        Store._checkpoint(con, sid, doc, payload)
        checkpoint = doc.get("resume_candidate") or {}
        pending_proposal = (
            {
                "revision_id": head,
                "based_on": payload.get("based_on"),
                "input_version": payload.get("input_version", 0),
                "eligible": False,
                "contract": target,
                "instructions": doc.get("instructions", []),
                "contract_application": deepcopy(doc.get("contract_application")),
                "source_fingerprint": checkpoint.get("source_fingerprint"),
                "operation_plan_snapshot": doc.get("operation_plan_snapshot"),
                "scope": "Unfinished work awaiting owner clarification; not validated or accepted",
            }
            if head is not None
            else None
        )
        doc.update(
            proposal=pending_proposal,
        )
        Store._restore_preview_working(doc, payload, restore_instructions=False, restore_plan=False)

    def _feature_applied(self, doc: dict[str, Any], jid: str) -> bool:
        """Compatibility predicate backed by durable publication, never a display receipt."""
        if doc.get("job_id") != jid or doc.get("authority_conflict_job_id") == jid:
            return False
        with self.connect() as con:
            row = con.execute(
                "SELECT payload FROM jobs WHERE id=? AND session=?", (jid, doc["id"])
            ).fetchone()
            payload = json.loads(row[0]) if row else {}
            return bool(
                row
                and self._preview_acceptance_matches(doc, payload)
                and self._feature_publication(con, doc["id"], jid, payload)
            )

    @staticmethod
    def _native_author_job(doc: dict[str, Any], payload: dict[str, Any]) -> bool:
        """Only native non-preview author loops use this additional checkpoint."""
        return bool(
            doc.get("authoring_mode") == "native_operations"
            and not payload.get("preview_only")
            and payload.get("document") is None
            and not payload.get("engineering_operation")
            and not payload.get("parameters")
            and not payload.get("restore_only")
        )

    @staticmethod
    def _native_author_binding(doc: dict[str, Any]) -> dict[str, Any]:
        """Keep current candidate authority distinct from accepted-key identity."""
        from forge_mind.workspace.document import digest

        return {
            "selected_revision": doc.get("selected_revision"),
            "accepted_revision_present": "accepted_revision" in doc,
            "accepted_revision": doc.get("accepted_revision"),
            "proposal_hash": digest(doc.get("proposal")),
            "contract_hash": digest(doc.get("contract")),
            "instructions_hash": digest(doc.get("instructions", [])),
            "contract_application_hash": digest(doc.get("contract_application")),
            "plan_version": doc.get("plan_version", 0),
            "plan_hash": doc.get("plan_hash"),
            "plan_contract_hash": doc.get("plan_contract_hash"),
            "design_plan_hash": digest(doc.get("design_plan")),
            "operation_plan_hash": digest(doc.get("operation_plan_snapshot")),
        }

    @staticmethod
    def _checkpoint(
        con: sqlite3.Connection, sid: str, doc: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        from forge_mind.workspace.input_application import CONTINUATION
        from forge_mind.workspace.product_sources import source_fingerprint

        if payload.get("document") or payload.get("engineering_operation"):
            doc["resume_candidate"] = None
            return
        candidate = doc.get("selected_revision")
        admitted = payload.get(CONTINUATION)
        inherited = (
            admitted["target"]
            if admitted is not None
            and candidate == admitted["selected_revision"]
            and doc.get("selection_version", 0) == admitted["selection_version"]
            else None
        )
        if inherited is not None:
            # A queued or just-claimed job has not necessarily selected its
            # inherited candidate. Later worker selection/publication wins.
            candidate = inherited.get("revision_id")
        if (candidate and candidate != payload.get("based_on")) or not payload.get("document"):
            doc["resume_candidate"] = {
                "revision_id": candidate,
                "based_on": payload.get("based_on"),
                "contract": doc.get("contract"),
                "instructions": doc.get("instructions", []),
                "input_version": admitted["target"]["input_version"]
                if admitted is not None
                else payload.get("input_version", 0),
                "source_fingerprint": admitted["target"]["source_fingerprint"]
                if admitted is not None
                else source_fingerprint(con, sid),
                "scope": "Unfinished work, not an accepted or validated design",
            }
            if doc.get("contract_application") is not None:
                doc["resume_candidate"]["contract_application"] = deepcopy(
                    doc["contract_application"]
                )
            if doc.get("operation_plan_snapshot") is not None:
                doc["resume_candidate"]["plan_snapshot"] = (
                    inherited.get("plan_snapshot")
                    if inherited is not None
                    and admitted is not None
                    and doc.get("plan_version", 0) == admitted["plan_version"]
                    else doc.get("operation_plan_snapshot")
                )
            if Store._native_author_job(doc, payload):
                doc["resume_candidate"].update(
                    native_author_binding=Store._native_author_binding(doc),
                    delivery_requested=payload.get("delivery_requested") is True,
                )
