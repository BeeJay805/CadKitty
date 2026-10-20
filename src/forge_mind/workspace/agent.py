"""Bounded CAD tool loop. The author cannot change its acceptance contract."""

from __future__ import annotations

import difflib
import json
import logging
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any

from pydantic import BaseModel, ValidationError

from forge_mind.agentic.params import extract_params, override_params
from forge_mind.llm.codex_client import CodexClient
from forge_mind.llm.output_contract import OutputContractError, response_correction
from forge_mind.llm.provider import LLMClientLike
from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent_memory import (
    active_submission,
    cache_schema,
    recent_context,
    rejected_submission,
)
from forge_mind.workspace.author_guidance import CORE, build_author_guidance, repair_focus
from forge_mind.workspace.budget import action_indices, end_time, expired, time_left
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.generation_policy import (
    author_model,
    native_call_allowance,
    staged_native_planning,
)
from forge_mind.workspace.measure import (
    engine_fingerprint,
    ground_contract,
    merge_contract_edit,
    verify_artifacts,
)
from forge_mind.workspace.models import Action, Contract, ContractEdit, Finding, Measurement
from forge_mind.workspace.processing import copy_artifact, process_revision
from forge_mind.workspace.store import Store, WorkspaceError

CONTRACT_PROMPT = """Extract a CAD acceptance contract from the user's request, independently of any generated geometry.
Keep every requested functional/appearance feature, even when unmeasurable. Never invent a dimension or tolerance.
Separate the subject's designation or product category from the properties, behavior, interfaces,
appearance, procedures and files the customer actually requests. A category or purpose label alone
does not add a claim of unspecified physical operation. Retain that source-backed designation in
the attributed subject label/source_quote and coverage_notes, rather than a separate achieved-result
requirement. Do not invent an operating test, mating object, load or interface from the category.
Keep every actually requested property or behavior as a requirement, including one expressed in a
descriptive label. An explicit duty to withstand a load, mate with an interface, resist water, have
a specified appearance, or follow a procedure remains critical even beside nominal-only language.
Do not hide such a duty in subject context or coverage_notes. Preserve consequential source ambiguity
without inventing either an achieved function or an absent customer constraint.
source=request and source_quote must be an exact substring. Retain assumptions as source=assumption.
Optional part_id selects one reusable definition by exact ID, measured in its local coordinates. Only use if the request explicitly identifies that part ID; never use another part or the whole assembly as a substitute.
Supported measurements: world-axis extent_x/y/z in mm; volume in mm^3; solid_count;
hole_count (cylindrical bore axes, not throughness); hole_diameter and hole_depth (axial wall span)
in mm, selected by world axis and center (the two other coordinates, increasing axis order).
Selection is separate from conformance. Do not guess an axis, coordinate or orientation. Use unmeasured
when ambiguous. min_wall is sampled and can find failures but cannot certify coverage. All other
actually requested properties or behaviors, including fit, mechanisms, cavity shapes, throughness, load capacity, text or style, must remain
unmeasured requirements, except the closed optional verification predicates below. Counts have integer value, zero tolerance. An exact numeric request has tolerance
zero unless the request explicitly allows a tolerance. Do not use whole-object extent for an internal feature.
Optional verification is independent frozen authority, never an author check request. Use nominal_through_hole_pattern only for explicit nominal CAD scope with a designed subject, exact local axis/frame, complete hole centers/diameters/count and source-stated tolerances. Keep one complete exact source_quote for that predicate, including its allowances; the containing requirement has the same quote and metric=unmeasured, value=null, part_id=null. Ambiguous scope, physical fit, intended drilled workpiece results and compound physical/subjective claims remain separate critical unmeasured requirements.
Use nominal_perforated_box only when the source explicitly requests a rectangular nominal CAD body, exact local XYZ size and outer-body center, with its complete embedded uniform through-hole pattern and source-stated hole allowances. size_mm and center_mm define the outer prism, not the mass centroid. This closed predicate permits no other material additions or removals. It preserves admitted hole diameter/position allowances; it adds no outer-body tolerance. Preserve any extra appearance, physical or procedural clause separately. If body shape or origin is ambiguous, keep it unmeasured instead of inferring a box.
Use a separate digital_delivery criterion for each explicitly requested project_stl, project_step, editable_native, prepared_print_package or bound_instruction_plan deliverable, coverage=all_made. Quote only that exact deliverable clause. A print-file request does not imply prepared print placement; simple instructions do not imply verified prose or a bound instruction plan. Preserve every requested clause even when no typed predicate can prove it. Never add assumption tolerances or label unsupported clauses noncritical to obtain a pass.
Use nominal_subject_inventory for explicit designed-part names and occurrence quantities: coverage=all_made, items with attributed subject_id, subject_role=designed, required_name and positive integer quantity. required_name means the exact saved PartDefinition.name, not an ID alias. One solid in a definition does not prove one occurrence. Keep the complete inventory source quote and all designed subjects; plan bindings and actual captured production establish the later result.
Use nominal_delivery_scope only for an explicit nominal CAD/digital-only restriction, scope=software_policy, policy=nominal_cad_and_digital_only. This is a narrow software-delivery policy result, not body geometry, physical manufacture, fit, prose correctness or a waiver. Keep every physical/procedural conjunct separately critical and unmeasured. Unsupported scope language remains unmeasured.
List ambiguities and omissions in coverage_notes. Return the Contract JSON. Do not use tools."""

AUTHOR_PROMPT = CORE  # Compatibility export; production uses state-selected guidance.


def assembly_measurement_context(measurement: dict[str, Any]) -> dict[str, Any]:
    """Keep measured failures and interfaces without replaying every rigid transform."""
    facts = measurement.get("facts", {})
    if not facts.get("assembly"):
        return measurement
    facts = {k: v for k, v in facts.items() if k != "feature_history"}
    # Detailed surfaces are available through occurrence inspection. Keep them
    # out of automatic author context; the full evidence reader stays complete.
    assembly = facts["assembly"]
    facts["assembly"] = {
        **assembly,
        "occurrences": [
            {k: v for k, v in row.items() if k != "cylindrical_surfaces"}
            for row in assembly.get("occurrences", [])
        ],
    }
    motion = facts.get("mechanism")
    if motion:
        samples = motion.get("samples", [])
        facts["mechanism"] = {
            k: v for k, v in motion.items() if k not in {"samples", "bindings"}
        } | {
            "physical_binding_count": len(motion.get("bindings", [])),
            "sample_summary": [
                {k: v for k, v in sample.items() if k != "poses"} for sample in samples
            ],
        }
    return {**measurement, "facts": facts}


def _rank(measurement: Measurement) -> tuple[int, int, int]:
    return (
        int(measurement.geometry_valid),
        sum(f.status == "pass" for f in measurement.findings),
        -sum(f.status == "fail" for f in measurement.findings),
    )


class WorkspaceService:
    def __init__(
        self,
        store: Store,
        executor: Executor,
        client_factory: Callable[..., LLMClientLike] = CodexClient,
    ) -> None:
        self.store, self.executor, self.client_factory = store, executor, client_factory
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.worker = threading.Thread(target=self._worker, name="forge-workspace", daemon=True)
        self.worker.start()

    def _worker(self) -> None:
        while not self.stop.is_set():
            job = self.store.claim()
            if job is None:
                self.stop.wait(1)
                continue
            try:
                self.run_job(job)
            except Exception as exc:
                # A failed job must not silently kill the worker while its process
                # continues to own the queue. Keep the saved candidate recoverable.
                self.store.event(
                    job["session"], "error", f"Job failed: {type(exc).__name__}: {str(exc)[:500]}"
                )
                self.store.finish(job["id"], job["session"], "error")
                if job.get("preview_only"):
                    from forge_mind.workspace.document_service import finish_preview

                    finish_preview(self, job)

    def _load_revision(
        self, sid: str, rid: str, *, allow_optional_display_loss: bool = False
    ) -> dict[str, Any]:
        revision = self.store.revision(sid, rid)
        measurement = Measurement.model_validate(revision["measurement"])
        # Optional display aids are checked separately against the exact bytes
        # delivered to the author. Their loss must retain the complete view.
        required = measurement
        if allow_optional_display_loss:
            required = measurement.model_copy(
                update={
                    "artifact_hashes": {
                        name: digest
                        for name, digest in measurement.artifact_hashes.items()
                        if name not in {"render-inspection.png", "render-inspection.json"}
                    }
                }
            )
        if not verify_artifacts(self.store.directory(sid, rid), required):
            raise WorkspaceError(
                "Revision artifacts changed or are missing; stale evidence cannot be reused"
            )
        return revision

    def build(
        self,
        sid: str,
        code: str,
        contract: Contract,
        parent: str | None,
        explanation: str,
        timeout: float | None,
        cancelled: Callable[[], bool],
    ) -> dict[str, Any]:
        from forge_mind.workspace.authoring_policy import enforce_action_policy

        enforce_action_policy("build", self.store.authoring_mode(sid), code)
        rid = uuid.uuid4().hex
        directory = self.store.directory(sid, rid)
        directory.mkdir(parents=True)
        (directory / "program.py").write_text(code, encoding="utf-8")
        self.store.event(sid, "build", explanation, revision_id=rid)
        started = time.monotonic()
        measured = Measurement(
            revision_id=rid,
            contract_hash=contract.fingerprint(),
            artifact_hashes={"program.py": file_sha256(directory / "program.py")},
            engine_fingerprint=engine_fingerprint(),
            geometry_valid=False,
        )
        # Author execution never mounts or writes the final revision directory.
        with tempfile.TemporaryDirectory(prefix="forge-cad-output-") as scratch:
            from pathlib import Path

            outcome = self.executor.build(code, Path(scratch), timeout, cancelled)
            if outcome.ok:
                try:
                    copy_artifact(Path(scratch) / "part.step", directory / "part.step")
                    measured = Measurement.model_validate(
                        process_revision(
                            directory,
                            {"revision_id": rid, "contract": contract.model_dump()},
                            None
                            if timeout is None
                            else max(0, timeout - (time.monotonic() - started)),
                            cancelled,
                        )
                    )
                except (ValueError, OSError) as exc:
                    measured.errors.append(str(exc))
        if not outcome.ok:
            measured.errors.insert(
                0, outcome.error_message or outcome.error_type or "CAD execution failed"
            )
        if not measured.findings:
            measured.findings = [
                Finding(
                    requirement_id=r.id,
                    status="unverified",
                    expected=r.value,
                    method="No measured final artifact",
                    reason="Build or postprocessing did not complete successfully.",
                )
                for r in contract.requirements
            ]
        previous_code = self.store.revision(sid, parent)["code"] if parent else ""
        diff = "".join(
            difflib.unified_diff(
                previous_code.splitlines(True),
                code.splitlines(True),
                fromfile=parent or "empty",
                tofile=rid,
            )
        )
        params = extract_params(code)
        revision = {
            "id": rid,
            "parent_id": parent,
            "at": time.time(),
            "explanation": explanation,
            "code": code,
            "parameters": params.values if params else {},
            "editable_parameters": params.float_names if params else [],
            "diff": diff,
            "contract": contract.model_dump(),
            "measurement": measured.model_dump(),
            "execution_seconds": outcome.duration_s,
            "execution_mode": self.executor.mode,
        }
        (directory / "evidence.json").write_text(
            measured.model_dump_json(indent=2), encoding="utf-8"
        )
        (directory / "contract.json").write_text(
            contract.model_dump_json(indent=2), encoding="utf-8"
        )
        self.store.add_revision(sid, revision)
        self.store.event(
            sid,
            "measurement",
            "Measured final exported revision",
            revision_id=rid,
            passed=sum(f.status == "pass" for f in measured.findings),
            failed=sum(f.status == "fail" for f in measured.findings),
            unverified=sum(f.status == "unverified" for f in measured.findings),
        )
        return revision

    def run_job(self, job: dict[str, Any]) -> None:
        from forge_mind.workspace.document_service import finish_preview, run_document_job

        native = self.store.authoring_mode(job["session"]) == "native_operations"
        if native:
            if job.get("parameters"):
                job = {
                    **job,
                    "native_delivery_baseline": job.get("revision_id")
                    or self.store.get(job["session"], job["owner"]).get("selected_revision"),
                }
            self.store.update(
                job["session"], automatic_delivery=None, contract_verification_report=None
            )
        if job.get("document") or job.get("engineering_operation"):
            run_document_job(self, job)
            return
        # Only an explicitly supplied work budget creates a deadline. Benchmarks
        # retain their protected validation tail; normal jobs validate to completion.
        automatic = not (job.get("parameters") or job.get("restore_only"))
        total = job.get("max_seconds")
        job_started = time.monotonic()
        overall_deadline = end_time(total)
        reserve = min(300.0, total * 0.25) if automatic and total is not None else 0.0
        protected_handoff = native and automatic and staged_native_planning()
        author_job = {
            **job,
            "max_seconds": None if total is None else total - reserve,
            "author_deadline": None if overall_deadline is None else overall_deadline - reserve,
        }
        if automatic:
            self.store.update(
                job["session"],
                completion_progress={
                    "phase": "authoring",
                    "reserved_seconds": None if total is None else reserve,
                    "author_seconds": None if total is None else total - reserve,
                    "message": "Completing subsystems without an automatic time cutoff; saved checks run after authoring"
                    if total is None
                    else "Completing subsystems; validation time is reserved",
                },
            )
        state = self._run_agent_job(author_job)
        from forge_mind.workspace.customer_construction import construction_finalization_allowed

        construction_stopped = not construction_finalization_allowed(
            self.store, job["session"], job["owner"], job["id"]
        )
        if state == "needs_approval":
            from forge_mind.workspace.customer_approval import preserve_budget

            preserve_budget(
                self.store,
                job["session"],
                job["owner"],
                job["id"],
                time_left(overall_deadline),
                self.store.get(job["session"], job["owner"]).get("customer_approval_actions_left"),
            )
        if (
            state != "cancelled"
            and not construction_stopped
            and not self.stop.is_set()
            and not self.store.cancelled(job["id"])
            and (self.store.get(job["session"], job["owner"]).get("intent_attribution") or {}).get(
                "status"
            )
            == "needs_input"
        ):
            state = "needs_input"
        if (
            automatic
            and not construction_stopped
            and state not in {"needs_input", "needs_approval"}
        ):
            from forge_mind.workspace.completion import finalize_completion

            validation_started = time.monotonic()
            self.store.event(
                job["session"],
                "validation",
                "Authoring stopped; running saved checks"
                if total is None
                else "Authoring stopped; running saved checks within the remaining job budget",
            )

            def cancelled() -> bool:
                return self.stop.is_set() or self.store.cancelled(job["id"])

            try:
                validation_deadline = overall_deadline
                if protected_handoff and not job.get("preview_only"):
                    from forge_mind.workspace.native_handoff_budget import capture_deadline

                    validation_deadline = capture_deadline(overall_deadline, reserve)
                finalize_completion(self, job, validation_deadline, cancelled, state)
            except Exception as exc:
                # A finalizer failure must never strand the running job or erase
                # the author's original failure. No validation is inferred.
                self.store.event(
                    job["session"],
                    "error",
                    f"Automatic validation failed: {type(exc).__name__}: {str(exc)[:500]}",
                )
                self.store.update(
                    job["session"],
                    completion_report={
                        "status": "unverified",
                        "author_state": state,
                        "revision_id": self.store.get(job["session"], job["owner"]).get(
                            "selected_revision"
                        ),
                        "reason": str(exc)[:500],
                        "subsystems": [],
                        "physical_readiness": "unverified",
                    },
                )
            progress = self.store.get(job["session"], job["owner"]).get("completion_progress", {})
            self.store.update(
                job["session"],
                completion_progress={
                    **progress,
                    "phase": "finished",
                    "message": "Automatic validation finished; review remaining subsystem work",
                    "author_elapsed_seconds": validation_started - job_started,
                    "validation_elapsed_seconds": time.monotonic() - validation_started,
                    "remaining_seconds": time_left(overall_deadline),
                },
            )
            if cancelled():
                state = "cancelled"
        if native and not construction_stopped and state not in {"needs_input", "needs_approval"}:
            from forge_mind.workspace.native_delivery import capture_native_delivery

            capture_native_delivery(
                self,
                job,
                overall_deadline,
                lambda: self.stop.is_set() or self.store.cancelled(job["id"]),
                state,
            )
        if state in {"needs_input", "needs_approval"} or construction_stopped:
            progress = self.store.get(job["session"], job["owner"]).get("completion_progress") or {}
            self.store.update(
                job["session"],
                completion_progress={
                    **progress,
                    "phase": "finished",
                    "message": "Approved construction stopped; review the saved failure before retrying"
                    if construction_stopped
                    else "Review the saved design plan before construction"
                    if state == "needs_approval"
                    else "Answer the current intent question before generation continues",
                },
                automatic_delivery={
                    "status": "incomplete",
                    "job_id": job["id"],
                    "author_state": state,
                    "reason": "Approved construction stopped; no new package was captured"
                    if construction_stopped
                    else "Owner plan approval remains; no new package was captured"
                    if state == "needs_approval"
                    else "A material intent question remains; no new package was captured",
                    "receipt": None,
                },
            )
        verification_report = None
        if (
            native
            and not job.get("preview_only")
            and state in {"needs_review", "contract_satisfied"}
        ):
            from forge_mind.workspace.requirement_verification import (
                has_requirement_status,
                verify_contract,
            )

            if has_requirement_status(self.store.get(job["session"], job["owner"]).get("contract")):
                try:
                    verification_report = verify_contract(
                        self,
                        job["session"],
                        job["owner"],
                        job["id"],
                        deadline=overall_deadline,
                        cancelled=lambda: self.stop.is_set() or self.store.cancelled(job["id"]),
                    )
                except Exception as exc:
                    self.store.event(
                        job["session"],
                        "verification",
                        f"Final frozen proof unavailable: {type(exc).__name__}: {str(exc)[:500]}",
                    )
        if self.stop.is_set() or self.store.cancelled(job["id"]):
            state = "cancelled"
        if verification_report is None:
            self.store.finish(job["id"], job["session"], state)
        else:
            self.store.finish(
                job["id"], job["session"], state, verification_report=verification_report
            )
        if job.get("preview_only"):
            finish_preview(self, job)

    def _run_agent_job(self, job: dict[str, Any]) -> str:
        sid, jid, owner = job["session"], job["id"], job["owner"]

        def cancelled() -> bool:
            return self.stop.is_set() or self.store.cancelled(jid)

        author_deadline = job.get("author_deadline", end_time(job.get("max_seconds")))
        deadline = author_deadline
        inference_deadline = author_deadline

        def author_cancelled() -> bool:
            return cancelled() or expired(deadline)

        def inference_cancelled() -> bool:
            return cancelled() or expired(inference_deadline)

        usage: list[dict[str, Any]] = []
        construction_outcome: dict[str, Any] | None = None
        claimed_construction: dict[str, Any] | None = None

        def stop_claimed_construction(reason: str) -> str:
            """An attempted approved command must not enter model repair or replay."""
            nonlocal claimed_construction
            from forge_mind.workspace.customer_construction import complete_construction

            if claimed_construction is not None:
                try:
                    complete_construction(self.store, sid, owner, jid, None, error=reason)
                except (ValueError, OSError) as exc:
                    reason += "; durable outcome remains unresolved: " + str(exc)[:500]
                finally:
                    claimed_construction = None
            self.store.event(sid, "error", "Approved construction stopped: " + reason[:1000])
            return "error"

        try:
            self.store.update(sid, state="running")
            session = self.store.get(sid, owner)
            staged = (
                session.get("authoring_mode") == "native_operations" and staged_native_planning()
            )
            protected_handoff = staged and not (job.get("parameters") or job.get("restore_only"))
            if staged:
                self.store.event(
                    sid,
                    "author_strategy",
                    "Staged native intent; each inference attempt has a 300-second allowance",
                    strategy="staged_intent",
                    per_call_seconds=300,
                    scope="Inference budget only; construction and completion gates are unchanged",
                )
            if session.get("authoring_mode") == "native_operations":
                from forge_mind.workspace.customer_construction import claim_construction

                if cancelled() or expired(author_deadline):
                    return "cancelled" if cancelled() else "budget_exhausted"
                try:
                    outcome = claim_construction(self.store, sid, owner, jid)
                    construction_outcome = dict(outcome) if outcome is not None else None
                    if construction_outcome is not None:
                        if construction_outcome["mode"] == "execute_saved_command":
                            claimed_construction = construction_outcome
                        elif construction_outcome["mode"] != "continue_published":
                            raise WorkspaceError("Unknown approved construction claim outcome")
                except WorkspaceError as exc:
                    return stop_claimed_construction(str(exc))
            if job.get("revision_id"):
                # Ordinary resumes retain their explicit selection behavior.
                # An approved command already binds that selection and counter;
                # selecting even the same ID again would invalidate its basis.
                self._load_revision(sid, job["revision_id"])
                if construction_outcome is None:
                    self.store.update(sid, selected_revision=job["revision_id"])
                elif job["revision_id"] != self.store.get(sid, owner).get("selected_revision"):
                    return stop_claimed_construction("Approved resume selection changed")
            if (
                construction_outcome is not None
                and construction_outcome["mode"] == "continue_published"
                and construction_outcome["revision_id"]
                != self.store.get(sid, owner).get("selected_revision")
            ):
                return stop_claimed_construction("Published construction selection changed")
            prompt = session["prompt"]
            client = self.client_factory(
                timeout_s=time_left(deadline), cancelled=inference_cancelled
            )
            model = author_model(client)
            self.store.update(sid, author_model_requested=model)
            recorded_calls: set[str] = set()

            def observed_call(operation: Callable[[], Any]) -> Any:
                """Keep transport evidence after temporary CLI files disappear."""
                try:
                    return operation()
                finally:
                    try:
                        diagnostics = getattr(client, "last_call_diagnostics", None)
                        if isinstance(diagnostics, dict):
                            call_id = diagnostics.get("call_id")
                            if (
                                isinstance(call_id, str)
                                and call_id
                                and call_id not in recorded_calls
                            ):
                                outcome = diagnostics.get("outcome")
                                message = {
                                    "success": "Assistant response received",
                                    "timeout": "Assistant response timed out",
                                    "cancelled": "Assistant request cancelled",
                                }.get(
                                    outcome if isinstance(outcome, str) else "",
                                    "Assistant request ended",
                                )
                                self.store.event(
                                    sid,
                                    "model_call",
                                    message,
                                    job_id=jid,
                                    diagnostics=diagnostics,
                                )
                                recorded_calls.add(call_id)
                    except Exception:
                        # Telemetry cannot replace the primary call result or retry.
                        # Do not log exception text: storage errors can contain data.
                        logging.getLogger(__name__).warning(
                            "Could not persist assistant call diagnostics"
                        )

            def inference(
                operation: Callable[[], Any],
                *,
                timeout_retry: bool,
                correction_factory: Callable[[OutputContractError], Callable[[], Any] | None],
                workflow: dict[str, Any] | None,
            ) -> Any:
                from sqlite3 import Error as DatabaseError

                from forge_mind.workspace.document import digest
                from forge_mind.workspace.model_retry import run_model_call

                if claimed_construction is not None:
                    raise WorkspaceError(
                        "Dispatch the saved approved construction before any further model call"
                    )

                def configure_timeout(left: float | None) -> None:
                    if hasattr(client, "timeout_s"):
                        client.timeout_s = native_call_allowance(left) if staged else left

                def authority_snapshot() -> dict[str, Any]:
                    doc = self.store.session_record(sid, owner)
                    with self.store.connect() as con:
                        version = con.execute(
                            "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
                        ).fetchone()
                        has_sources = con.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='product_source_events'"
                        ).fetchone()
                        # Source events are append-only. Capture the pinned head,
                        # rather than reading captured documents or image bytes.
                        sources = (
                            con.execute(
                                "SELECT version,digest FROM product_source_events WHERE session=? ORDER BY version DESC LIMIT 1",
                                (sid,),
                            ).fetchone()
                            if has_sources
                            else None
                        )
                    return {
                        "authority": self.store._native_author_binding(doc),
                        "acceptance": digest(self.store._preview_acceptance_identity(doc)),
                        "prompt": digest(doc.get("prompt")),
                        "sources": tuple(sources) if sources else None,
                        "input_version": version[0] if version else 0,
                        "selection_version": doc.get("selection_version", 0),
                        "assets": digest(doc.get("assets", [])),
                        "customer_reference": digest(doc.get("customer_reference")),
                        "customer_proposal": digest(doc.get("customer_proposal")),
                        "intent_status": (doc.get("intent_attribution") or {}).get("status"),
                    }

                try:
                    original_authority = authority_snapshot()
                except (ValueError, OSError, DatabaseError, TypeError, KeyError):
                    # Unavailable optional recovery metadata cannot block the original call.
                    original_authority = None
                original_workflow = digest(workflow)

                def corrected_operation(error: OutputContractError) -> Callable[[], Any] | None:
                    if original_authority is None:
                        return None
                    corrected = correction_factory(error)
                    if corrected is None:
                        return None

                    def dispatch() -> Any:
                        from forge_mind.workspace.customer_construction import (
                            author_workflow_context,
                        )

                        if claimed_construction is not None:
                            raise WorkspaceError(
                                "Approved construction must dispatch before correction"
                            )
                        with self.store.connect() as con:
                            row = con.execute(
                                "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
                            ).fetchone()
                            if row is None:
                                raise WorkspaceError("workspace not found")
                            doc = json.loads(row[0])
                            self.store._check_plan_writer(
                                con, sid, doc, jid, allow_intent_record=True
                            )
                        if (
                            (doc.get("intent_attribution") or {}).get("status") == "needs_input"
                            or authority_snapshot() != original_authority
                            or digest(author_workflow_context(self.store, sid, owner, jid))
                            != original_workflow
                        ):
                            raise WorkspaceError("Owner authority changed before output correction")
                        if cancelled() or expired(inference_deadline):
                            raise LLMError(
                                "Output correction cancelled or its inference deadline expired"
                            )
                        left = time_left(inference_deadline)
                        if left is not None and left < 45:
                            raise LLMError(
                                "Not enough inference time remains for output correction"
                            )
                        return observed_call(corrected)

                    return dispatch

                return run_model_call(
                    lambda: observed_call(operation),
                    native=session.get("authoring_mode") == "native_operations",
                    allow_timeout_retry=timeout_retry,
                    remaining=lambda: time_left(inference_deadline),
                    cancelled=cancelled,
                    configure_timeout=configure_timeout,
                    on_retry=lambda reason, attempt, delay: self.store.event(
                        sid,
                        "model_retry",
                        "The model is temporarily busy; retrying this step"
                        if reason == "capacity"
                        else "The model response could not be used. Retrying this step."
                        if reason == "output_contract"
                        else "Model call timed out; retrying once within the remaining job budget",
                        reason=reason,
                        attempt=attempt,
                        delay_seconds=delay,
                    ),
                    correction_factory=corrected_operation,
                    sleep=time.sleep,
                    clock=time.monotonic,
                )

            def workflow_arguments(
                arguments: dict[str, Any],
            ) -> tuple[dict[str, Any], dict[str, Any] | None]:
                from forge_mind.workspace.author_guidance import AUTHOR_WORKFLOW_AUTHORITY
                from forge_mind.workspace.completion_amendment import completion_amendment
                from forge_mind.workspace.customer_construction import author_workflow_context

                workflow = author_workflow_context(self.store, sid, owner, jid)
                raw_user = arguments.get("user", "")
                try:
                    user = json.loads(raw_user)
                except (TypeError, ValueError):
                    user = None
                if not isinstance(user, dict):
                    user = {"source_context": raw_user}
                amendment = completion_amendment(user, workflow)
                if workflow is None and amendment is None:
                    return arguments, None
                arguments = dict(arguments)
                if amendment is not None:
                    user["completion_amendment"] = amendment
                if workflow is not None:
                    user["author_workflow_context"] = workflow
                    arguments["system"] += "\n" + AUTHOR_WORKFLOW_AUTHORITY
                arguments["user"] = json.dumps(user)
                return arguments, workflow

            def call_model(*, timeout_retry: bool = True, **arguments: Any) -> Any:
                arguments, workflow = workflow_arguments(arguments)
                if "image_png" in arguments:
                    return inference(
                        lambda: client.generate_structured_with_image(**arguments),
                        timeout_retry=timeout_retry,
                        correction_factory=lambda error: partial(
                            client.generate_structured_with_image,
                            **response_correction(arguments, error),
                        ),
                        workflow=workflow,
                    )
                return inference(
                    lambda: client.generate_structured(**arguments),
                    timeout_retry=timeout_retry,
                    correction_factory=lambda error: partial(
                        client.generate_structured, **response_correction(arguments, error)
                    ),
                    workflow=workflow,
                )

            from forge_mind.workspace.input_application import FIELD, pending

            input_id = job.get("input_application_id")
            pending_input = pending(session)
            application = session.get(FIELD) or {}
            if input_id and (
                (pending_input or {}).get("input_id") != input_id
                and (application.get("applied") or {}).get("input_id") != input_id
            ):
                raise WorkspaceError("Owner input application identity changed")
            instructions = list(
                job.get(
                    "input_instructions",
                    job.get("resume_instructions", session.get("instructions", [])),
                )
            )
            if not input_id and job.get("instruction"):
                instructions.append(job["instruction"])
            intent = prompt + "".join("\nUser revision instruction: " + i for i in instructions)
            if staged and not input_id:
                # A question raised during extraction must bind the actual request,
                # including the queued owner answer, rather than the prior source.
                self.store.update(sid, instructions=instructions)
            raw_contract = (
                job["contract"] if job.get("contract") is not None else session["contract"]
            )
            input_mode = pending_input["mode"] if input_id and pending_input else None
            revision_instruction = (
                pending_input["instruction"]
                if input_mode == "delta" and pending_input is not None
                else job.get("instruction", "")
            )
            initial_contract_pending = staged and raw_contract is None
            attribution = None
            attribution_basis = None
            extraction_model: type[BaseModel] = (
                ContractEdit if raw_contract is not None else Contract
            )
            extraction_system = CONTRACT_PROMPT
            from forge_mind.workspace.customer_delivery_policy import customer_delivery_context

            delivery_policy = customer_delivery_context(session)
            policy_instructions = (
                "\nServer delivery policy is separate software-delivery metadata, not owner "
                "source text or a source quote. It requests outputs, not nominal acceptance "
                "authority. Preserve every owner duty and physical unknown; never invent "
                "quotes, scope, frames, targets or proof from this metadata."
                if delivery_policy is not None
                else ""
            )
            if initial_contract_pending:
                from forge_mind.workspace.author_guidance import ATTRIBUTION_EXTRACTION
                from forge_mind.workspace.edit_service import read_edit_basis
                from forge_mind.workspace.intent_attribution import ContractDraft
                from forge_mind.workspace.source_delivery_contract_draft import (
                    SOURCE_DELIVERY_DRAFT_INSTRUCTIONS,
                    SourceDeliveryContractDraft,
                )

                extraction_model = SourceDeliveryContractDraft
                source_instructions = SOURCE_DELIVERY_DRAFT_INSTRUCTIONS
                if session.get("customer_reference"):
                    from forge_mind.workspace.source_delivery_contract_draft import (
                        PHOTO_SOURCE_DRAFT_INSTRUCTIONS_V2,
                        PhotoSourceContractDraftV2,
                    )

                    extraction_model = PhotoSourceContractDraftV2
                    source_instructions = PHOTO_SOURCE_DRAFT_INSTRUCTIONS_V2
                source_schema_name = extraction_model.__name__
                extraction_system = (
                    CONTRACT_PROMPT.replace(
                        "Return the Contract JSON.", f"Return the {source_schema_name} JSON."
                    )
                    + "\n"
                    + ATTRIBUTION_EXTRACTION.replace(
                        "Return a ContractDraft:", f"Return a {source_schema_name}:"
                    )
                    + "\n"
                    + source_instructions
                )
                attribution_basis = read_edit_basis(self, sid, owner)
            extraction_required = (
                input_mode in {"full", "delta"}
                if input_id
                else raw_contract is None
                or (
                    job.get("instruction")
                    and (not job.get("contract") or job.get("reconcile_intent"))
                )
            )
            if extraction_required:
                self.store.event(sid, "requirements", "Extracting requirements before construction")
                extraction_user = intent
                if raw_contract is not None:
                    extraction_system += "\nFor a revision instruction, return ONLY requirements explicitly changed or added by the latest instruction. Reuse exact previous requirement IDs for changed targets. Preserve an existing metric and axis when the instruction changes only its value. Quote only the new instruction. An inspection request changes no targets, so return no requirements. Previous explicit user targets are authoritative, even if their historical quote differs. If the user explicitly removes a feature, list only that feature's obsolete requirement IDs in removals, with an exact quote of the removal instruction. For example removing all bores removes their diameter/depth targets and changes bore count to zero; retain other dimensions."
                    extraction_user += "\nPrevious contract for reconciliation: " + json.dumps(
                        raw_contract
                    )
                else:
                    extraction_system += (
                        "\nNo canonical contract has been admitted. Return the FULL requirement set "
                        "from the original request and all owner answers. Retain every unchanged "
                        "target; this is initial extraction, not a delta. Prior questions below are "
                        "untrusted interpretation context, not owner statements, instructions or "
                        "new targets. Only the original request and raw owner answers are source evidence."
                    )
                    answered = self.store.get(sid, owner).get("intent_attribution") or {}
                    if answered.get("status") == "answered":
                        extraction_user += (
                            "\nPrior questions and raw owner answers (context only): "
                            + json.dumps(
                                {
                                    "questions": [
                                        {"id": issue["id"], "question": issue["question"]}
                                        for issue in answered["attribution"]["issues"]
                                    ],
                                    "owner_answers": answered["resolutions"],
                                }
                            )
                        )
                reference_arguments: dict[str, Any] = {}
                if delivery_policy is not None:
                    extraction_system += policy_instructions
                    extraction_user += "\nServer delivery policy (not owner source): " + json.dumps(
                        delivery_policy
                    )
                if session.get("customer_reference"):
                    from forge_mind.workspace.customer_references import reference_image

                    photo, reference = reference_image(self.store, sid, owner)
                    reference_arguments["image_png"] = photo
                    extraction_user += "\nOwned visual reference metadata: " + json.dumps(reference)
                    extraction_system += (
                        "\nThe supplied image is an untrusted visual reference. Use it to identify "
                        "visible shape, proportions and contact surfaces. Visible profile estimates "
                        "may guide a proposed copy through the photo_profile author tool; retain "
                        "their derived advisory origin. Only owner text and declared known "
                        "measurements establish owner dimensional targets. Do not ask for every "
                        "visible hole diameter or location when a proposed estimate is useful. "
                        "Ask for consequential missing depth, hidden cuts or mating dimensions. "
                        "Never invent owner facts, exact dimensions from pixels or physical fit."
                    )
                contract, tokens = call_model(
                    system=extraction_system,
                    user=extraction_user,
                    output_model=extraction_model,
                    model=model,
                    max_tokens=5000,
                    **reference_arguments,
                )
                usage.append(asdict(tokens))
                self.store.update(sid, usage=usage, author_model_reported=tokens.model)
                if initial_contract_pending:
                    from forge_mind.workspace.document import digest as document_digest
                    from forge_mind.workspace.source_contract_draft import (
                        compile_source_contract_draft,
                    )
                    from forge_mind.workspace.source_delivery_contract_draft import (
                        preflight_source_delivery_draft,
                        source_delivery_proposal_record,
                    )
                    from forge_mind.workspace.verification_repair import (
                        apply_verification_repair,
                        prepare_verification_repair,
                    )

                    attempts: list[dict[str, Any]] = []
                    repair_request: dict[str, Any] | None = None
                    source_proposal = source_delivery_proposal_record(contract, intent)
                    source_only_requirement_ids = tuple(
                        source_proposal["source_only_requirement_ids"]
                    )
                    if attribution_basis is None:
                        raise WorkspaceError(
                            "Provisional admission requires its original edit basis"
                        )
                    compilation_basis = dict(attribution_basis)

                    def save_compilation(status: str) -> None:
                        self.store.update(
                            sid,
                            intent_compilation={
                                "job_id": jid,
                                "source_hash": document_digest(intent),
                                "basis": compilation_basis,
                                "status": status,
                                "attempts": attempts,
                                "source_proposal": source_proposal,
                                **({"repair_request": repair_request} if repair_request else {}),
                            },
                        )

                    def record_compilation(
                        draft: BaseModel,
                        result: dict[str, Any],
                        repair: dict[str, Any] | None = None,
                        *,
                        source_attempt: bool = False,
                    ) -> None:
                        attempt = {
                            "attempt": len(attempts) + 1,
                            "raw_draft": draft.model_dump(mode="json"),
                            **result,
                            **({"verification_repair": repair} if repair else {}),
                            **(
                                {"source_proposal": source_proposal}
                                if not attempts or source_attempt
                                else {}
                            ),
                        }
                        attempts.append(attempt)
                        save_compilation(result["status"])
                        self.store.event(
                            sid,
                            "intent_compilation",
                            "Checked provisional interpretation before canonical admission",
                            job_id=jid,
                            source_hash=document_digest(intent),
                            **attempt,
                        )

                    save_compilation("checking")
                    source_wire = contract
                    try:
                        source_check = preflight_source_delivery_draft(source_wire, intent)
                    except ValueError:
                        save_compilation("lowering_rejected")
                        raise
                    source_proposal = source_check.source_proposal
                    original_draft = source_check.draft
                    contract = original_draft or source_wire
                    compilation = source_check.compilation
                    record_compilation(contract, compilation)
                    if compilation["status"] == "invalid_interpretation" and not author_cancelled():
                        wire_correction = (
                            original_draft is None
                            or bool(source_proposal["raw_response"].get("nominal_targets"))
                            or bool(source_proposal["raw_response"].get("photo_nominal_claims"))
                            or bool(source_proposal["raw_response"].get("photo_requirements"))
                        )
                        prepared_repair = None
                        if not wire_correction:
                            assert original_draft is not None
                            prepared_repair = prepare_verification_repair(
                                original_draft, intent, compilation
                            )
                        correction_model: type[BaseModel] = ContractDraft
                        correction_system = (
                            CONTRACT_PROMPT.replace(
                                "Return the Contract JSON.", "Return the ContractDraft JSON."
                            )
                            + "\n"
                            + ATTRIBUTION_EXTRACTION
                            + policy_instructions
                        )
                        correction_user = (
                            extraction_user
                            + "\nThe provisional interpretation failed shared admission. "
                            "Return one corrected FULL ContractDraft. Retain every prior requirement "
                            "ID, text, source duty, critical flag, scalar target/selector, valid "
                            "predicate, source subject/fact and genuine customer question. "
                            "You may supply missing valid verification or correct invalid typed "
                            "interpretation using the original source. Do not invent answers, "
                            "shorten exact source quotes, drop duties or weaken targets. "
                            "The host will not infer a replacement predicate from prose. "
                            "A question needs a real missing customer input; saved CAD prerequisites "
                            "belong to authoring.\nPrevious provisional draft and diagnostics: "
                            + json.dumps(attempts[0])
                        )
                        if wire_correction:
                            correction_model = type(source_wire)
                            correction_system = extraction_system
                            retained_duty_fields = (
                                "all original duty IDs, exact source quotes, critical flags, "
                                if "photo_requirements" in source_proposal["raw_response"]
                                else "all original duty IDs, text, critical flags, "
                            )
                            correction_user = (
                                extraction_user
                                + "\nThe source proposal failed deterministic preflight. Return one "
                                "corrected FULL proposal in the SAME source wire schema. Retain "
                                + retained_duty_fields
                                + "valid source facts, "
                                "subjects and genuine questions, and every proposed nominal target "
                                "ID and parent duty. Correct the rejected references or target fields "
                                "from the original source. Do not choose new source meaning, invent "
                                "answers, drop targets or provide host policy markers. The host reruns "
                                "the same strict lowering and atomic replay.\nOriginal proposal and "
                                "preflight diagnostics: " + json.dumps(attempts[0])
                            )
                        elif prepared_repair is not None:
                            correction_model = prepared_repair.response_model
                            correction_system += (
                                "\nFor this correction, return only VerificationRepair JSON matching "
                                "the supplied schema. The host retains the full original draft. "
                                "Do not return ContractDraft or attribution fields."
                            )
                            repair_request = prepared_repair.request_record()
                            save_compilation(compilation["status"])
                            correction_user = (
                                extraction_user
                                + "\nRepair only the invalid verification slots in the response schema. "
                                "Each slot fixes its requirement ID and exact parent source quote. "
                                "Populate exactly one named canonical family field and set "
                                "unverified_reason to null. Set every other family field to null. "
                                "Emit that literal quote with your replacement criterion. Optional "
                                "shared source certificates exclude any listed whole kinds; "
                                "an empty exclusion list leaves all canonical kinds available. "
                                "These restrictions do not select "
                                "a replacement kind or supply targets. Use only the original source. "
                                "When no supported predicate applies, set all family fields to null "
                                "and supply a nonblank unverified_reason. This retains the critical "
                                "unverified duty. All other draft fields stay unchanged. "
                                "No owner answers, dropped duties or relaxed targets are allowed.\n"
                                "Original draft and all admission diagnostics: "
                                + json.dumps(attempts[0])
                                + "\nPrepared repair basis: "
                                + json.dumps(prepared_repair.basis)
                            )
                        correction, tokens = call_model(
                            system=correction_system,
                            user=correction_user,
                            output_model=correction_model,
                            model=model,
                            max_tokens=5000,
                            timeout_retry=False,
                            **reference_arguments,
                        )
                        usage.append(asdict(tokens))
                        self.store.update(sid, usage=usage, author_model_reported=tokens.model)
                        repair_response = None
                        if wire_correction:
                            source_proposal = source_delivery_proposal_record(correction, intent)
                            save_compilation("checking")
                            corrected_check = preflight_source_delivery_draft(
                                correction, intent, previous=source_check
                            )
                            source_proposal = corrected_check.source_proposal
                            contract = corrected_check.draft or correction
                            compilation = corrected_check.compilation
                        elif prepared_repair is not None:
                            assert original_draft is not None
                            repair_response = prepared_repair.response_record(correction)
                            # Retain the wire response even if fresh basis validation rejects it.
                            repair_request = repair_response
                            save_compilation(compilation["status"])
                            contract = apply_verification_repair(
                                original_draft, intent, prepared_repair, correction
                            )
                        else:
                            contract = correction
                        if not wire_correction:
                            compilation = compile_source_contract_draft(
                                contract,
                                intent,
                                source_only_requirement_ids=source_only_requirement_ids,
                                previous=original_draft,
                            )
                        record_compilation(
                            contract, compilation, repair_response, source_attempt=wire_correction
                        )
                    if compilation["status"] == "invalid_interpretation":
                        if author_cancelled():
                            return "cancelled" if cancelled() else "budget_exhausted"
                        self.store.event(
                            sid,
                            "requirements",
                            "Provisional interpretation remains invalid; no contract was admitted "
                            "and no customer answer was invented",
                        )
                        return "needs_review"
                if author_cancelled():
                    return "cancelled" if cancelled() else "budget_exhausted"
                if (self.store.get(sid, owner).get("intent_attribution") or {}).get(
                    "status"
                ) == "needs_input":
                    if claimed_construction is not None:
                        return stop_claimed_construction(
                            "Owner input changed before approved dispatch"
                        )
                    return "cancelled" if cancelled() else "needs_input"
                if initial_contract_pending:
                    attribution = contract.attribution
                    contract = contract.contract
                contract = (
                    merge_contract_edit(
                        Contract.model_validate(raw_contract), contract, revision_instruction
                    )
                    if raw_contract is not None
                    else ground_contract(contract, intent, attribution=attribution)
                )
            else:
                contract = Contract.model_validate(raw_contract)
            if not input_id:
                self.store.update(sid, instructions=instructions)
            if initial_contract_pending:
                from forge_mind.workspace.intent_attribution_service import record_proposal

                if attribution is None or attribution_basis is None:
                    raise WorkspaceError(
                        "Initial intent attribution and editing basis are required"
                    )
                proposal = record_proposal(
                    self.store,
                    sid,
                    owner,
                    jid,
                    attribution,
                    contract,
                    intent,
                    attribution_basis,
                )
                if proposal["status"] == "needs_input":
                    self.store.event(
                        sid,
                        "intent_question",
                        "A detail changes the part's size or function; answer before construction",
                        intent_attribution=proposal,
                    )
                    return "needs_input"
            else:
                if input_id and pending_input:
                    self.store.apply_input_contract(sid, owner, jid, input_id, contract)
                elif not input_id:
                    self.store.update(sid, contract=contract.model_dump())
            current = job.get("revision_id") or session.get("selected_revision")
            if current:
                self._load_revision(sid, current, allow_optional_display_loss=True)
            best = current
            best_measurement = (
                Measurement.model_validate(self.store.revision(sid, best)["measurement"])
                if best
                else None
            )
            if best_measurement and (
                best_measurement.contract_hash != contract.fingerprint()
                or best_measurement.engine_fingerprint != engine_fingerprint()
            ):
                best_measurement, best = None, None  # a new contract cannot inherit old approval
            history: list[dict[str, Any]] = (
                list(session.get("tool_memory", [])) if job.get("resume") else []
            )
            schema_engine = engine_fingerprint()
            schemas_seen = (
                set(session.get("schemas_seen", []))
                if session.get("schema_engine") == schema_engine
                else set()
            )
            schema_cache = (
                dict(session.get("schema_cache", {}))
                if session.get("schema_engine") == schema_engine
                else {}
            )
            hashes: set[str] = set()
            inspection_image = None
            inspection_revision_id = None
            inspection_instrument = None
            pending_source_observation = None
            edited_candidate = None
            explicitly_restored = None
            exhausted_actions = False
            incomplete_finish = False
            premature_finishes = 0
            reviewing_guidance = False
            published_revision_id: str | None = None
            failed_submission = session.get("failed_submission")
            action_limit = job.get("max_actions")
            photo_generation: dict[str, Any] | None = None
            phase = "construction"
            schedule = None
            from forge_mind.workspace.agent_memory import remember

            def report_phase(next_phase: str) -> None:
                message = (
                    "Reviewing current customer instructions"
                    if next_phase == "instructions"
                    else "Preparing saved geometry, plan and print files for handoff"
                )
                self.store.event(
                    sid,
                    "handoff_phase",
                    message,
                    phase=next_phase,
                    previous_phase=phase,
                    job_id=jid,
                )
                progress = self.store.get(sid, owner).get("completion_progress") or {}
                self.store.update(
                    sid,
                    completion_progress={**progress, "phase": "authoring", "message": message},
                )

            if protected_handoff:
                from forge_mind.workspace.native_handoff_budget import HandoffBudget

                schedule = HandoffBudget(author_deadline, job.get("max_seconds"), action_limit)

            def automatic_outcome(
                claim: dict[str, Any], started: float, outcome: dict[str, Any]
            ) -> None:
                record = {
                    "origin": "host",
                    "job_id": jid,
                    "action_index": claim["action_index"],
                    "automatic_handoff_attempt": claim,
                    "elapsed_seconds": max(0.0, time.monotonic() - started),
                    "outcome": outcome,
                }
                history.append({"automatic_handoff": record})
                self.store.event(
                    sid, "automatic_handoff_result", "Automatic preparation outcome", **record
                )

            for index in action_indices(action_limit):
                correcting_handoff = False
                inferred_action = False
                inferred_work_deadline: float | None = None
                automatic_claim: dict[str, Any] | None = None
                automatic_started = 0.0
                automatic_recorded = False
                if (self.store.get(sid, owner).get("intent_attribution") or {}).get(
                    "status"
                ) == "needs_input":
                    if claimed_construction is not None:
                        return stop_claimed_construction(
                            "Owner input changed before approved dispatch"
                        )
                    return "cancelled" if cancelled() else "needs_input"
                if schedule is not None:
                    next_phase = schedule.phase(index, time.monotonic(), phase)
                    if next_phase != phase:
                        report_phase(next_phase)
                    phase = next_phase
                    deadline = schedule.deadline(phase)
                inference_deadline = (
                    schedule.inference_deadline(phase) if schedule is not None else deadline
                )
                remaining = time_left(deadline)
                if cancelled() or expired(author_deadline):
                    if claimed_construction is not None:
                        return stop_claimed_construction(
                            "Cancellation or the original deadline stopped approved dispatch"
                        )
                    return "cancelled" if cancelled() else "budget_exhausted"
                if hasattr(client, "timeout_s"):
                    client.timeout_s = remaining
                selected = (
                    self._load_revision(sid, current, allow_optional_display_loss=True)
                    if current
                    else None
                )
                from forge_mind.workspace.photo_generation import generation_context

                photo_generation = generation_context(self, sid, owner, current, contract)
                photo_stage = (photo_generation or {}).get("stage")
                pending_direct_change = (
                    bool(job.get("parameters")) and edited_candidate is None
                ) or (bool(job.get("restore_only")) and explicitly_restored is None)
                if pending_direct_change and photo_stage == "compare":
                    photo_stage = "construct"
                if photo_stage in {"inventory", "compare"}:
                    from forge_mind.workspace.photo_generation import (
                        compare_candidate,
                        observe_inventory,
                    )

                    try:
                        photo_timeout = min(30.0, remaining) if remaining is not None else 30.0
                        if photo_stage == "inventory":
                            photo_result = observe_inventory(
                                self,
                                sid,
                                owner,
                                jid,
                                timeout=photo_timeout,
                                cancelled=author_cancelled,
                            )
                        else:
                            assert current is not None
                            photo_result = compare_candidate(
                                self,
                                sid,
                                owner,
                                jid,
                                current,
                                timeout=photo_timeout,
                                cancelled=author_cancelled,
                            )
                        history.append({"automatic_photo_generation": photo_result})
                        self.store.event(
                            sid,
                            "photo_generation",
                            "Automatic photo stage completed",
                            job_id=jid,
                            action_index=index,
                            origin="host",
                            stage=photo_stage,
                            result=photo_result,
                        )
                    except (ValueError, OSError) as exc:
                        if not author_cancelled():
                            from forge_mind.workspace.photo_generation import record_failure

                            record_failure(
                                self, sid, owner, jid, photo_stage, current, str(exc)[:2000]
                            )
                        history.append({"automatic_photo_error": str(exc)[:2000]})
                        self.store.event(
                            sid,
                            "photo_generation",
                            "Automatic photo stage failed",
                            job_id=jid,
                            action_index=index,
                            origin="host",
                            stage=photo_stage,
                            error=str(exc)[:2000],
                        )
                    self.store.update(sid, tool_memory=remember(history))
                    continue
                if photo_stage == "unsupported":
                    assert photo_generation is not None
                    self.store.event(sid, "review", photo_generation["reason"])
                    return "needs_review"
                if photo_stage in {None, "checked"} and (
                    (job.get("parameters") and edited_candidate is not None)
                    or (job.get("restore_only") and explicitly_restored is not None)
                ):
                    break
                revisions = self.store.revisions(sid)
                context = {
                    "request": prompt,
                    "authoring_mode": session.get("authoring_mode", "hybrid"),
                    "user_revision_history": instructions,
                    "edit_instruction": revision_instruction,
                    "frozen_contract": contract.model_dump(),
                    "actions_left": None if action_limit is None else action_limit - index,
                    "seconds_left": None if remaining is None else round(remaining),
                    "selected": selected,
                    "history": recent_context(history, schema_cache),
                    "available_revisions": [r["id"] for r in revisions],
                    "engineering_inputs": session.get("engineering", {}),
                    "owned_assets": session.get("assets", []),
                    "design_plan": self.store.get(sid, owner).get("design_plan"),
                    "schemas_already_read": sorted(schemas_seen),
                    "authoritative_schemas": schema_cache,
                }
                if delivery_policy is not None:
                    context["delivery_policy"] = delivery_policy
                if photo_generation is not None:
                    context["photo_generation"] = photo_generation
                if staged:
                    context["staged_native_planning"] = True
                    context["per_call_seconds"] = native_call_allowance(
                        time_left(inference_deadline)
                    )
                if protected_handoff:
                    context["handoff_phase"] = phase
                attribution_record = self.store.get(sid, owner).get("intent_attribution")
                if attribution_record:
                    context["request_attribution"] = attribution_record
                if initial_contract_pending:
                    context["proposed_contract"] = context["frozen_contract"]
                    context["frozen_contract"] = None
                failed_submission = active_submission(
                    failed_submission, current, contract.fingerprint()
                )
                if failed_submission:
                    context["failed_submission"] = failed_submission
                # Fresh model calls otherwise see only the last valid candidate
                # and the failed measurement, losing the source they must repair.
                if (
                    revisions
                    and revisions[-1]["id"] != current
                    and revisions[-1].get("parent_id") == current
                    and revisions[-1].get("measurement", {}).get("contract_hash")
                    == contract.fingerprint()
                    and not revisions[-1].get("measurement", {}).get("geometry_valid")
                ):
                    failed_id = revisions[-1]["id"]
                    try:
                        from forge_mind.workspace.inspection import author_revision_context

                        failed = self._load_revision(sid, failed_id)
                        context["failed_attempt"] = author_revision_context(
                            {
                                **{k: v for k, v in failed.items() if k != "diff"},
                                "measurement": assembly_measurement_context(failed["measurement"]),
                            }
                        )
                    except WorkspaceError as exc:
                        context["failed_attempt"] = {
                            "id": failed_id,
                            "unavailable": str(exc),
                            "scope": "Failed source could not be verified; do not reuse its evidence",
                        }
                from forge_mind.workspace.completion import completion_context
                from forge_mind.workspace.completion_inline import author_checkpoint

                try:
                    correction_checks = bool(
                        protected_handoff
                        and phase == "instructions"
                        and schedule is not None
                        and schedule.can_correct(index, time.monotonic())
                    )
                    context["early_checks"] = (
                        {
                            "status": "deferred",
                            "reason": "Saved validation follows instruction review",
                        }
                        if protected_handoff and phase == "instructions" and not correction_checks
                        else author_checkpoint(
                            self,
                            sid,
                            owner,
                            current,
                            jid,
                            schedule.correction_deadline()
                            if correction_checks and schedule is not None
                            else deadline,
                            cancelled,
                        )
                    )
                except Exception as exc:
                    context["early_checks"] = {
                        "status": "unverified",
                        "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                    }
                if cancelled() or expired(deadline):
                    if protected_handoff and not cancelled() and not expired(author_deadline):
                        continue
                    return "cancelled" if cancelled() else "budget_exhausted"
                if hasattr(client, "timeout_s"):
                    client.timeout_s = time_left(deadline)
                remaining_after_checks = time_left(deadline)
                context["seconds_left"] = (
                    None if remaining_after_checks is None else round(remaining_after_checks)
                )
                if protected_handoff:
                    inference_remaining = time_left(inference_deadline)
                    context["phase_seconds_left"] = context["seconds_left"]
                    context["seconds_left"] = (
                        None if inference_remaining is None else round(inference_remaining)
                    )
                    context["per_call_seconds"] = native_call_allowance(inference_remaining)

                completion = completion_context(self, sid, owner, current)
                context["completion"] = completion
                if session.get("authoring_mode") == "native_operations":
                    from forge_mind.workspace.check_amendment import request_hash

                    check_requests = {
                        check["id"]: check["request"]
                        for subsystem in (context.get("design_plan") or {}).get("subsystems", [])
                        for check in subsystem.get("checks", [])
                    }
                    for subsystem in completion.get("subsystems", []):
                        for check in subsystem.get("checks", []):
                            if check["id"] in check_requests:
                                check["request_hash"] = request_hash(check_requests[check["id"]])
                from forge_mind.workspace.agent_memory import remember
                from forge_mind.workspace.product_sources import read_sources

                source_set = read_sources(self.store, sid, owner)
                context["source_claims"] = source_set["claims"]
                context["source_artifacts"] = [
                    {k: v for k, v in a.items() if k not in {"text", "content"}}
                    for a in source_set["artifacts"]
                ]
                native = session.get("authoring_mode") == "native_operations"
                context["persistent_tool_observations"] = remember(history[:-6])
                if selected and selected.get("document"):
                    from forge_mind.workspace.inspection import author_revision_context

                    context["selected"] = author_revision_context(
                        {
                            **selected,
                            "measurement": assembly_measurement_context(selected["measurement"]),
                        }
                    )
                    if native:
                        record = context["selected"].get("operation_record")
                        if (
                            record
                            and (record.get("plan_snapshot") or {}).get("plan")
                            == context["design_plan"]
                        ):
                            context["selected"]["operation_record"] = {
                                key: value
                                for key, value in record.items()
                                if key != "plan_snapshot"
                            }
                            context["selected"]["operation_plan_in_context"] = "design_plan"
                if native:
                    from forge_mind.workspace.native_authoring import native_bootstrap
                    from forge_mind.workspace.plan_draft import summary as plan_draft_summary

                    context["native_authoring"] = native_bootstrap(
                        contract if context.get("frozen_contract") is not None else None
                    )
                    if session.get("customer_reference"):
                        from forge_mind.workspace.photo_profile_service import profile_discovery

                        context["native_authoring"]["discovery"]["photo_profile"] = (
                            profile_discovery(self.store, sid, owner)
                        )
                    else:
                        context["native_authoring"]["discovery"]["all_actions"].remove(
                            "photo_profile"
                        )
                    context["plan_draft"] = plan_draft_summary(self.store, sid, owner)
                    from forge_mind.workspace.edit_service import read_edit_basis

                    try:
                        context["edit_basis"] = read_edit_basis(self, sid, owner)
                    except WorkspaceError as exc:
                        context["edit_basis"] = {"status": "unavailable", "reason": str(exc)}
                    if current and selected and selected.get("document"):
                        from forge_mind.workspace.document import DesignDocument
                        from forge_mind.workspace.models import DesignPlan
                        from forge_mind.workspace.plan_reconciliation import alignment

                        live_plan = self.store.get(sid, owner)
                        try:
                            dimension_alignment = alignment(
                                DesignPlan.model_validate(live_plan.get("design_plan")),
                                DesignDocument.model_validate(selected["document"]),
                            )
                        except ValueError as exc:
                            # Optional author inspection can precede a valid plan or
                            # encounter a historical partial document. Strict patch
                            # writes and final review still require complete inputs.
                            dimension_alignment = {"status": "unavailable", "reason": str(exc)}
                        context["plan_reconciliation"] = {
                            "expected_plan_version": live_plan.get("plan_version", 0),
                            "revision_id": current,
                            "captured_plan_hash": (selected.get("operation_record") or {}).get(
                                "plan_hash"
                            ),
                            "current_plan_hash": live_plan.get("plan_hash"),
                            **dimension_alignment,
                        }
                    if current and selected and selected.get("document"):
                        from forge_mind.workspace.delivery_guidance import guidance_context

                        try:
                            context["delivery_guidance"] = guidance_context(
                                self, sid, owner, current
                            )
                        except WorkspaceError as exc:
                            # Optional display loss must not stop geometry inspection. Strict
                            # guidance writes and exports still require every saved artifact.
                            context["delivery_guidance"] = {
                                "status": "unavailable",
                                "reason": str(exc),
                                "scope": "No usable review basis; repair saved artifacts before binding guidance.",
                            }
                        if protected_handoff:
                            from forge_mind.workspace.native_finish import finish_context
                            from forge_mind.workspace.native_preflight import preflight_context

                            try:
                                context["handoff_preflight"] = preflight_context(
                                    finish_context(
                                        self, sid, owner, current, require_bound_instructions=True
                                    )
                                )
                            except (ValueError, OSError) as exc:
                                context["handoff_preflight"] = {
                                    "work_reasons": [str(exc)[:1000]],
                                    "scope": "Current preflight is unavailable; repair saved inputs before closure.",
                                }
                context["repair_focus"] = repair_focus(context)
                if photo_generation is not None:
                    from forge_mind.workspace.photo_generation import construction_example

                    context["photo_generation"]["construction_transaction"] = construction_example(
                        context
                    )
                if (
                    protected_handoff
                    and pending_source_observation is None
                    and claimed_construction is None
                    and photo_stage in {None, "checked"}
                ):
                    from forge_mind.workspace.native_handoff_auto import (
                        claim_automatic_preparation,
                    )

                    automatic_claim = claim_automatic_preparation(
                        self.store,
                        sid,
                        owner,
                        jid,
                        context=context,
                        published_revision_id=published_revision_id,
                        index=index,
                        schedule=schedule,
                        phase=phase,
                        now=time.monotonic(),
                        enabled=True,
                    )
                replaying_construction = claimed_construction is not None
                action: Action
                if replaying_construction:
                    from forge_mind.workspace.native_authoring import NativeAction

                    assert claimed_construction is not None
                    action = NativeAction.model_validate(claimed_construction["action"])
                elif automatic_claim is not None:
                    from forge_mind.workspace.native_handoff_prepare import PrepareHandoffRequest

                    assert schedule is not None
                    deadline = schedule.deadline("construction")
                    inference_deadline = deadline
                    automatic_started = time.monotonic()
                    action = Action(
                        kind="inspect",
                        query="prepare_handoff",
                        revision_id=automatic_claim["revision_id"],
                        explanation="Prepare current saved inputs before editable visual review",
                        document_json=PrepareHandoffRequest(
                            revision_id=automatic_claim["revision_id"], prepare_print=True
                        ).model_dump_json(),
                    )
                elif job.get("parameters") and edited_candidate is None:
                    action = Action(
                        kind="parameters",
                        explanation="Apply requested parameter edit and revalidate",
                        parameters=job["parameters"],
                    )
                elif job.get("restore_only") and explicitly_restored is None:
                    action = Action(
                        kind="restore",
                        explanation="Restore the requested revision",
                        revision_id=current,
                    )
                else:
                    system = build_author_guidance(
                        context,
                        self.executor.mode,
                        typed_container_verified=getattr(
                            self.executor, "typed_container_verified", False
                        )
                        is True,
                    )
                    if not selected and self.executor.mode != "trusted_local":
                        from forge_mind.agentic.cheatsheet import author_skills_block

                        system += "\n" + author_skills_block()
                    from forge_mind.workspace.author_images import (
                        invoke_author_images,
                        prepare_author_images,
                        with_customer_reference,
                    )

                    if pending_source_observation is not None:
                        from forge_mind.workspace.source_page_agent import (
                            prepare_source_page_images,
                        )

                        images = prepare_source_page_images(
                            self.store, sid, owner, pending_source_observation
                        )
                        pending_source_observation = None
                    else:
                        images = prepare_author_images(
                            self.store.directory(sid, current) if current else None,
                            selected,
                            section=inspection_image,
                            section_revision_id=inspection_revision_id,
                            section_instrument=inspection_instrument,
                            multiple=bool(session.get("customer_reference"))
                            or callable(getattr(client, "generate_structured_with_images", None)),
                        )
                    inspection_image = None
                    inspection_revision_id = None
                    inspection_instrument = None
                    if session.get("customer_reference"):
                        images = with_customer_reference(
                            self.store,
                            sid,
                            owner,
                            images,
                            multiple=callable(
                                getattr(client, "generate_structured_with_images", None)
                            ),
                        )
                    context["image_observation"] = images.context()
                    response_model: type[BaseModel] = Action
                    startup = False
                    final_instruction = False
                    photo_grounding_turn = photo_stage == "grounding"
                    if native:
                        from forge_mind.workspace.native_authoring import NativeAction

                        response_model = NativeAction
                        startup = (
                            staged
                            and not context.get("design_plan")
                            and not context.get("plan_draft")
                            and not (failed_submission or {}).get("startup_intent_rejected")
                        )
                        if photo_grounding_turn:
                            from forge_mind.workspace.photo_generation import (
                                GROUNDING_INSTRUCTIONS,
                                PhotoGrounding,
                            )

                            startup = False
                            response_model = PhotoGrounding
                            system = GROUNDING_INSTRUCTIONS
                            context = {
                                key: context[key]
                                for key in (
                                    "request",
                                    "frozen_contract",
                                    "proposed_contract",
                                    "request_attribution",
                                    "photo_generation",
                                    "history",
                                    "seconds_left",
                                    "actions_left",
                                    "image_observation",
                                )
                                if key in context
                            }
                        elif startup:
                            from forge_mind.workspace.author_guidance import STARTUP_INTENT
                            from forge_mind.workspace.native_stages import startup_context
                            from forge_mind.workspace.plan_intent import PlanIntent

                            response_model = PlanIntent
                            system = STARTUP_INTENT
                            context = startup_context(context)
                        elif staged:
                            from forge_mind.workspace.native_stages import construction_context

                            context = construction_context(context)
                        if reviewing_guidance and not startup and photo_stage in {None, "checked"}:
                            from forge_mind.workspace.author_guidance import INSTRUCTION_REVIEW

                            system = INSTRUCTION_REVIEW
                        guide = context.get("delivery_guidance") or {}
                        correcting_handoff = bool(
                            protected_handoff
                            and not photo_grounding_turn
                            and phase == "instructions"
                            and (context.get("handoff_preflight") or {}).get("work_reasons")
                            and schedule is not None
                            and schedule.can_correct(index, time.monotonic())
                        )
                        if correcting_handoff:
                            assert schedule is not None
                            deadline = schedule.correction_deadline()
                            inference_deadline = deadline
                            if photo_stage in {None, "checked"}:
                                from forge_mind.workspace.author_guidance import INSTRUCTION_REVIEW

                                reviewing_guidance = True
                                system = INSTRUCTION_REVIEW
                            correction_left = time_left(deadline)
                            context["seconds_left"] = (
                                None if correction_left is None else round(correction_left)
                            )
                            context["phase_seconds_left"] = context["seconds_left"]
                            context["per_call_seconds"] = native_call_allowance(correction_left)
                            context["handoff_correction"] = {
                                "final_review_deadline": author_deadline,
                                "scope": "Repair inside the original author budget. Final protected instruction time and one existing action slot remain reserved.",
                            }
                        if (
                            protected_handoff
                            and photo_stage in {None, "checked"}
                            and phase == "instructions"
                            and not correcting_handoff
                            and current
                            and guide.get("basis_hash")
                            and guide.get("status") != "unavailable"
                        ):
                            from forge_mind.workspace.author_guidance import FINAL_INSTRUCTIONS
                            from forge_mind.workspace.instruction_context import (
                                ProtectedGuidanceUpdate,
                            )

                            response_model = ProtectedGuidanceUpdate
                            system = FINAL_INSTRUCTIONS
                            final_instruction = True
                    args: dict[str, Any] = {
                        "system": system,
                        "output_model": response_model,
                        "model": model,
                        "max_tokens": 3000 if startup else 12000,
                    }
                    user_context = context
                    if (
                        native
                        and (reviewing_guidance or final_instruction)
                        and photo_stage in {None, "checked"}
                    ):
                        from forge_mind.workspace.instruction_context import (
                            customer_instruction_context,
                        )

                        user_context = customer_instruction_context(context)
                    if native and staged:
                        from forge_mind.workspace.native_stages import deduplicate_context

                        user_context = deduplicate_context(user_context)
                    if delivery_policy is not None:
                        # Stage and instruction projections must retain the same
                        # server policy on startup and every continuation call.
                        user_context["delivery_policy"] = delivery_policy
                    args["user"] = json.dumps(user_context)
                    from forge_mind.llm.client import LLMError

                    try:
                        if cancelled() or expired(deadline):
                            raise LLMError("Authoring deadline reached before inference")
                        inferred_action = True
                        inferred_work_deadline = inference_deadline
                        if images.images:
                            if hasattr(client, "timeout_s"):
                                client.timeout_s = time_left(inference_deadline)
                            args, workflow = workflow_arguments(args)

                            image_request = partial(
                                invoke_author_images,
                                client,
                                images,
                                observation_root=self.store.root / "observations" / sid / job["id"],
                                event=lambda kind, message, **data: self.store.event(
                                    sid, kind, message, **data
                                ),
                            )

                            def correct_images(
                                error: OutputContractError,
                                request: Callable[..., Any] = image_request,
                                original: dict[str, Any] = args,
                            ) -> Callable[[], Any]:
                                return partial(request, response_correction(original, error))

                            action, tokens = inference(
                                partial(image_request, args),
                                timeout_retry=False,
                                correction_factory=correct_images,
                                workflow=workflow,
                            )
                        else:
                            action, tokens = call_model(**args)
                    except LLMError:
                        if (
                            protected_handoff
                            and (phase != "instructions" or correcting_handoff)
                            and expired(deadline)
                            and not expired(author_deadline)
                            and not cancelled()
                        ):
                            self.store.event(
                                sid,
                                "handoff_boundary",
                                "The construction allowance ended; reserved review remains available",
                                phase=phase,
                                job_id=jid,
                            )
                            continue
                        raise
                    if photo_grounding_turn:
                        action = Action(
                            kind="photo_profile",
                            query="ground",
                            explanation="Ground the detected photo features against source facts",
                            document_json=action.model_dump_json(),
                        )
                    elif startup:
                        action = Action(
                            kind="plan",
                            query="intent",
                            explanation="Save provisional part intent before construction",
                            document_json=action.model_dump_json(),
                        )
                    elif final_instruction:
                        action = Action(
                            kind="guidance",
                            query="finish",
                            explanation="Review the latest saved facts and customer instructions",
                            document_json=action.model_dump_json(),
                        )
                    usage.append(asdict(tokens))
                    self.store.update(sid, usage=usage, author_model_reported=tokens.model)
                if (self.store.get(sid, owner).get("intent_attribution") or {}).get(
                    "status"
                ) == "needs_input":
                    if claimed_construction is not None:
                        return stop_claimed_construction(
                            "Owner input changed before approved dispatch"
                        )
                    if automatic_claim is not None:
                        automatic_outcome(
                            automatic_claim,
                            automatic_started,
                            {"status": "not_executed", "reason": "needs_input"},
                        )
                    return "cancelled" if cancelled() else "needs_input"
                submitted_action = action.model_dump()
                plan_patch_json = submitted_action.pop("plan_patch_json", "")
                action = Action.model_validate(submitted_action)
                if plan_patch_json:
                    submitted_action["plan_patch_json"] = plan_patch_json
                feature_basis = None
                if replaying_construction and claimed_construction is not None:
                    from forge_mind.workspace.edit_contracts import EditBasis

                    saved_feature_basis = claimed_construction.get("feature_basis")
                    if saved_feature_basis is not None:
                        feature_basis = EditBasis.model_validate(saved_feature_basis)
                construction_patch = None
                self.store.event(
                    sid,
                    "action",
                    action.explanation,
                    action=action.kind,
                    query=action.query,
                    part_id=action.part_id,
                    parameters=action.parameters,
                    study_values=action.study_values,
                    **(
                        {
                            "origin": "host",
                            "job_id": jid,
                            "action_index": index,
                            "automatic_handoff_attempt": automatic_claim,
                        }
                        if automatic_claim is not None
                        else {}
                    ),
                )
                # Only inference may cross the construction cue. Refresh the work
                # phase before admission, retaining the request's hard cutoff.
                response_phase = phase
                if schedule is not None:
                    next_phase = schedule.phase(index, time.monotonic(), phase)
                    if next_phase != phase:
                        report_phase(next_phase)
                    phase = next_phase
                    deadline = schedule.deadline(phase)
                    if automatic_claim is not None:
                        # Host work cannot borrow the model's soft-cue allowance.
                        deadline = schedule.deadline("construction")
                        inference_deadline = deadline
                    elif correcting_handoff:
                        deadline = schedule.correction_deadline()
                inferred_cad_action = (
                    inferred_action
                    and native
                    and action.kind in {"document", "assembly_edit", "feature_edit"}
                )
                if inferred_cad_action:
                    # Keep explicit CAD inside its request's hard cutoff on both
                    # sides of the construction cue, including publication checks.
                    cutoffs = [
                        cutoff
                        for cutoff in (inferred_work_deadline, author_deadline)
                        if cutoff is not None
                    ]
                    if correcting_handoff and schedule is not None:
                        correction_cutoff = schedule.correction_deadline()
                        if correction_cutoff is not None:
                            cutoffs.append(correction_cutoff)
                    deadline = min(cutoffs) if cutoffs else None
                if cancelled() or expired(inference_deadline) or expired(deadline):
                    if replaying_construction:
                        return stop_claimed_construction(
                            "Cancellation or the original deadline stopped approved dispatch"
                        )
                    if automatic_claim is not None:
                        automatic_outcome(
                            automatic_claim,
                            automatic_started,
                            {"status": "not_executed", "reason": "cancelled_or_deadline_exhausted"},
                        )
                        automatic_recorded = True
                    if protected_handoff and not cancelled() and not expired(author_deadline):
                        self.store.event(
                            sid,
                            "handoff_boundary",
                            "Automatic preparation discarded at its construction cutoff"
                            if automatic_claim is not None
                            else "Late phase response discarded before applying any action",
                            phase=response_phase,
                            active_phase=phase,
                            job_id=jid,
                            **(
                                {"origin": "host", "action_index": index}
                                if automatic_claim is not None
                                else {}
                            ),
                        )
                        continue
                    return "cancelled" if cancelled() else "budget_exhausted"
                decoded_document = None
                try:
                    from forge_mind.workspace.authoring_policy import enforce_action_policy

                    enforce_action_policy(
                        action.kind, session.get("authoring_mode", "hybrid"), action.code
                    )
                    if action.kind == "inspect":
                        from forge_mind.workspace.native_authoring import INSPECTION_QUERIES

                        if action.query not in INSPECTION_QUERIES:
                            raise WorkspaceError(
                                f"Unknown inspect query {action.query!r}; valid queries: "
                                + ", ".join(repr(query) for query in INSPECTION_QUERIES)
                            )
                    # Resolve read-only schema names before an attached plan can mutate state.
                    if action.kind == "schema":
                        from forge_mind.workspace.authoring import authoring_schema

                        if action.query == "design_study":
                            from forge_mind.workspace.design_study_api import study_schema

                            schema = study_schema()
                        elif action.query == "guidance":
                            from forge_mind.workspace.delivery_guidance import GuidanceUpdate

                            schema = GuidanceUpdate.model_json_schema()
                        elif action.query == "prepare_handoff":
                            from forge_mind.workspace.native_handoff_prepare import (
                                PrepareHandoffRequest,
                            )

                            schema = PrepareHandoffRequest.model_json_schema()
                        elif action.query == "plan_reconciliation":
                            from forge_mind.workspace.plan_reconciliation import PlanReconciliation

                            schema = PlanReconciliation.model_json_schema()
                        elif action.query == "lab" or action.query.startswith("lab:"):
                            from forge_mind.workspace.lab_api import lab_schema

                            available = lab_schema()
                            if action.query == "lab":
                                schema = available
                            else:
                                name = action.query[4:]
                                if name not in available:
                                    raise WorkspaceError(
                                        f"Unknown lab schema {action.query!r}; available names: "
                                        + ", ".join(f"lab:{name}" for name in available)
                                    )
                                schema = available[name]
                        elif action.query == "production":
                            from forge_mind.workspace.product_models import PartProduction

                            schema = PartProduction.model_json_schema()
                        elif action.query == "interface":
                            from forge_mind.workspace.product_interfaces import InterfaceProposal

                            schema = InterfaceProposal.model_json_schema()
                        elif action.query == "source_claim":
                            from forge_mind.workspace.product_sources import DimensionalClaim

                            schema = DimensionalClaim.model_json_schema()
                        elif action.query == "source_page":
                            from forge_mind.workspace.source_page_models import source_page_schema

                            schema = source_page_schema()
                        elif action.query == "product_tools" or action.query.startswith("tool:"):
                            from forge_mind.workspace.product_tools import product_tool_schema

                            schema = product_tool_schema(
                                action.query.removeprefix("tool:")
                                if action.query.startswith("tool:")
                                else None
                            )
                        else:
                            schema = authoring_schema(action.query)
                    if native and action.kind in {
                        "document",
                        "build",
                        "parameters",
                        "restore",
                        "feature_edit",
                        "assembly_edit",
                    }:
                        from forge_mind.workspace.photo_generation import require_grounded

                        require_grounded(self, sid, owner)
                    if plan_patch_json:
                        from forge_mind.workspace.plan_patch import PlanPatch

                        if not native or action.kind not in {
                            "document",
                            "assembly_edit",
                            "feature_edit",
                        }:
                            raise WorkspaceError(
                                "plan_patch_json is allowed only on native document, assembly_edit or feature_edit"
                            )
                        construction_patch = PlanPatch.model_validate_json(plan_patch_json)
                    from forge_mind.workspace.action_payloads import decode_json_object

                    if action.document_json:
                        decoded_document = decode_json_object(action.document_json)
                    if native and action.kind == "plan" and action.query == "clarify":
                        from forge_mind.workspace.intent_attribution import IntentConflict
                        from forge_mind.workspace.intent_attribution_service import record_conflict

                        conflict = IntentConflict.model_validate_json(action.document_json)
                        question = record_conflict(
                            self.store,
                            sid,
                            owner,
                            jid,
                            conflict.issues,
                            contract,
                            intent,
                            context["edit_basis"],
                        )
                        self.store.event(
                            sid,
                            "intent_question",
                            "A design interpretation needs an owner answer before further work",
                            intent_attribution=question,
                        )
                        return "needs_input"
                    if action.kind == "plan" and action.query in {
                        "draft_read",
                        "draft_patch",
                        "draft_commit",
                    }:
                        if not native:
                            raise WorkspaceError("Plan draft author actions require native mode")
                        from forge_mind.workspace.plan_draft import (
                            PlanDraftCommit,
                            PlanDraftPatch,
                            PlanDraftRead,
                            commit_draft,
                            patch_draft,
                            read_draft,
                        )

                        if action.query == "draft_read":
                            draft_result = read_draft(
                                self.store,
                                sid,
                                owner,
                                PlanDraftRead.model_validate_json(action.document_json),
                            )
                        elif action.query == "draft_patch":
                            draft_result = patch_draft(
                                self.store,
                                sid,
                                owner,
                                PlanDraftPatch.model_validate_json(action.document_json),
                                jid,
                            )
                        else:
                            draft_result = commit_draft(
                                self.store,
                                sid,
                                owner,
                                PlanDraftCommit.model_validate_json(action.document_json),
                                jid,
                                author_intent=intent,
                                pending_contract=contract if initial_contract_pending else None,
                            )
                            initial_contract_pending = False
                            if failed_submission and failed_submission["action_kind"] == "plan":
                                failed_submission = None
                                self.store.update(sid, failed_submission=None)
                        history.append({"plan_draft_result": draft_result})
                        self.store.update(sid, tool_memory=remember(history))
                        continue
                    if native and action.kind == "plan" and action.query == "patch":
                        from forge_mind.workspace.edit_service import patch_plan
                        from forge_mind.workspace.plan_patch import PlanPatch

                        patch_result = patch_plan(
                            self,
                            sid,
                            owner,
                            PlanPatch.model_validate_json(action.document_json),
                            jid,
                            author_intent=intent,
                        )
                        self.store.update(sid, last_edit_result=patch_result)
                        history.append({"plan_patch": patch_result})
                        self.store.update(sid, tool_memory=remember(history))
                        continue
                    if action.plan is not None or action.kind == "plan":
                        from forge_mind.workspace.completion import validate_plan
                        from forge_mind.workspace.models import DesignPlan

                        previous = self.store.get(sid, owner)
                        patch = None
                        if native and action.kind == "plan" and action.query == "intent":
                            from forge_mind.workspace.plan_draft import summary as current_draft
                            from forge_mind.workspace.plan_intent import PlanIntent, lower_intent

                            if (
                                previous.get("design_plan")
                                or current
                                or current_draft(self.store, sid, owner)
                            ):
                                raise WorkspaceError(
                                    "Initial intent cannot replace an admitted plan; use plan patch"
                                )
                            startup_intent = PlanIntent.model_validate_json(action.document_json)
                            if startup_intent.attribution_issues:
                                from forge_mind.workspace.intent_attribution_service import (
                                    record_conflict,
                                )

                                question = record_conflict(
                                    self.store,
                                    sid,
                                    owner,
                                    jid,
                                    startup_intent.attribution_issues,
                                    contract,
                                    intent,
                                    context["edit_basis"],
                                )
                                self.store.event(
                                    sid,
                                    "intent_question",
                                    "Clarify a material interpretation before constructing the part",
                                    intent_attribution=question,
                                )
                                return "needs_input"
                            plan = lower_intent(startup_intent, contract)
                        elif native and action.kind == "plan" and action.query == "reconcile":
                            from forge_mind.workspace.document import DesignDocument
                            from forge_mind.workspace.plan_reconciliation import (
                                PlanReconciliation,
                                reconcile_plan,
                            )

                            patch = PlanReconciliation.model_validate_json(action.document_json)
                            if (
                                patch.revision_id != current
                                or previous.get("selected_revision") != current
                            ):
                                raise WorkspaceError("Design changed; refresh plan reconciliation")
                            if patch.expected_plan_version != previous.get("plan_version", 0):
                                raise WorkspaceError(
                                    "Design plan changed; refresh before reconciling"
                                )
                            for fact in patch.updates:
                                if fact.source == "measured_reference":
                                    raise WorkspaceError(
                                        "A plan patch cannot assert a new measured reference"
                                    )
                                if fact.source == "customer" and fact.source_quote not in intent:
                                    raise WorkspaceError(
                                        "Customer fact quote must occur in the current request history"
                                    )
                            plan = reconcile_plan(
                                DesignPlan.model_validate(previous.get("design_plan")),
                                patch,
                                DesignDocument.model_validate(
                                    self._load_revision(sid, current)["document"]
                                ),
                                contract,
                            )
                        else:
                            plan = action.plan or DesignPlan.model_validate_json(
                                action.document_json
                            )
                        validate_plan(
                            plan,
                            contract,
                            previous.get("design_plan") or {}
                            if previous.get("plan_contract_hash") == contract.fingerprint()
                            else {},
                        )
                        if failed_submission and failed_submission["action_kind"] == action.kind:
                            failed_submission = None
                            self.store.update(sid, failed_submission=None)
                        plan_history = list(previous.get("design_plan_history", []))
                        if (
                            previous.get("design_plan")
                            and previous["design_plan"] != plan.model_dump()
                        ):
                            prior_report = previous.get("completion_report") or {}
                            plan_history.append(
                                {
                                    "plan": previous["design_plan"],
                                    "contract_hash": previous.get("plan_contract_hash"),
                                    "revision_id": prior_report.get("revision_id"),
                                    "receipt_ids": prior_report.get("receipt_ids", []),
                                    "status": prior_report.get("status", "unverified"),
                                }
                            )
                        if session.get("authoring_mode") == "native_operations":
                            self.store.set_design_plan_for_job(
                                sid,
                                owner,
                                jid,
                                plan,
                                expected_plan_version=previous.get("plan_version", 0),
                                contract=contract,
                                expected_current_revision=current
                                if patch is not None
                                else (previous.get("proposal") or {}).get("revision_id")
                                or previous.get("selected_revision"),
                                expected_edit_basis=context.get("edit_basis")
                                if action.query == "intent"
                                else None,
                            )
                        else:
                            self.store.update(
                                sid,
                                design_plan=plan.model_dump(),
                                plan_contract_hash=contract.fingerprint(),
                                design_plan_history=plan_history[-8:],
                            )
                        initial_contract_pending = False
                        from forge_mind.workspace.operation_plan import reference_scope_warnings

                        history.append(
                            {
                                "plan_saved": True,
                                "scope": "Author intent, not evidence",
                                "reference_scope_warnings": reference_scope_warnings(plan),
                            }
                        )
                        self.store.update(sid, tool_memory=remember(history))
                        if action.kind == "plan":
                            continue
                except (ValueError, WorkspaceError) as exc:
                    if replaying_construction:
                        return stop_claimed_construction(str(exc))
                    from forge_mind.workspace.action_payloads import InputValidationError

                    failed_submission = rejected_submission(
                        submitted_action, current, contract.fingerprint(), str(exc)
                    )
                    if native and action.kind == "plan" and action.query == "intent":
                        failed_submission["startup_intent_rejected"] = True
                    self.store.update(sid, failed_submission=failed_submission)
                    if native and action.kind == "plan":
                        from forge_mind.workspace.edit_contracts import EditBasis
                        from forge_mind.workspace.plan_draft import (
                            retain_rejected_plan,
                        )
                        from forge_mind.workspace.plan_draft import (
                            summary as plan_draft_summary,
                        )

                        if (
                            action.query
                            not in {
                                "patch",
                                "intent",
                                "reconcile",
                                "draft_read",
                                "draft_patch",
                                "draft_commit",
                            }
                            and decoded_document is not None
                        ):
                            try:
                                failed_submission["plan_draft"] = retain_rejected_plan(
                                    self.store,
                                    sid,
                                    owner,
                                    action.document_json,
                                    EditBasis.model_validate(context["edit_basis"]),
                                    jid,
                                    error=str(exc),
                                )
                                if "unavailable" in failed_submission:
                                    failed_submission["unavailable"] = (
                                        "Exact action is outside automatic context; the exact plan "
                                        "is retained in plan_draft. Read affected records and use "
                                        "draft_patch, then explicitly draft_commit."
                                    )
                            except (ValueError, WorkspaceError) as retention_error:
                                failed_submission["plan_draft_retention_error"] = str(
                                    retention_error
                                )[:1000]
                        else:
                            failed_submission["plan_draft"] = plan_draft_summary(
                                self.store, sid, owner
                            )
                    self.store.update(sid, failed_submission=failed_submission)
                    history.append(
                        {
                            "tool_error": str(exc)[:1000],
                            "action": submitted_action["kind"],
                            "action_result": exc.error_result()
                            if isinstance(exc, InputValidationError)
                            else {
                                "code": "rejected",
                                "message": str(exc)[:1000],
                                "retryable": True,
                                "applied": False,
                            },
                        }
                    )
                    self.store.event(sid, "tool_error", str(exc)[:1000])
                    from forge_mind.workspace.store import EditConflictError

                    if isinstance(exc, EditConflictError):
                        self.store.update(sid, authority_conflict_job_id=jid)
                        return "needs_review"
                    self.store.update(sid, tool_memory=remember(history))
                    continue
                if action.plan is not None:
                    completion = completion_context(self, sid, owner, current)
                if action.kind == "finish":
                    if photo_generation is not None:
                        from forge_mind.workspace.photo_generation import photo_closure

                        closure = photo_closure(self, sid, owner, current)
                        if closure["status"] != "checked":
                            history.append({"photo_correction_required": closure})
                            self.store.update(sid, tool_memory=remember(history))
                            continue
                    from forge_mind.workspace.native_delivery import _declared_mesh

                    if native and current is None and job.get("preview_only"):
                        break
                    if native and current is None and not job.get("preview_only"):
                        premature_finishes += 1
                        incomplete_finish = True
                        self.store.update(sid, incomplete_finish_job_id=jid)
                        review: dict[str, Any] = {
                            "status": "needs_work",
                            "revision_id": None,
                            "reasons": [
                                "No saved CAD revision exists. Submit the complete native construction document before finishing."
                            ],
                            "scope": "Recoverable author feedback within the original action and time budget; no CAD or approval is inferred.",
                        }
                        history.append({"final_review_correction": review})
                        self.store.update(
                            sid,
                            native_review={**review, "job_id": jid},
                            tool_memory=remember(history),
                        )
                        self.store.event(sid, "review", review["reasons"][0])
                        continue
                    guide = context.get("delivery_guidance") or {}
                    if (
                        protected_handoff
                        and current
                        and guide.get("basis_hash")
                        and guide.get("status") != "unavailable"
                        and (
                            guide.get("status") != "current"
                            or (guide.get("instruction_binding") or {}).get("status") != "bound"
                        )
                    ):
                        report_phase("instructions")
                        phase = "instructions"
                        history.append(
                            {
                                "handoff_request": "Review current customer instructions before stopping"
                            }
                        )
                        continue
                    if (
                        native
                        and not job.get("preview_only")
                        and (
                            job.get("delivery_requested")
                            or _declared_mesh((selected or {}).get("document") or {})
                        )
                    ):
                        from forge_mind.workspace.native_finish import finish_context

                        final_review = finish_context(
                            self,
                            sid,
                            owner,
                            current,
                            require_bound_instructions=protected_handoff,
                        )
                        self.store.update(sid, native_review={**final_review, "job_id": jid})
                        if final_review["status"] != "ready_for_validation":
                            premature_finishes += 1
                            incomplete_finish = True
                            self.store.update(sid, incomplete_finish_job_id=jid)
                            history.append({"final_review_correction": final_review})
                            if premature_finishes == 1:
                                continue
                            break
                        incomplete_finish = False
                        self.store.update(sid, incomplete_finish_job_id=None)
                    guide = context.get("delivery_guidance") or {}
                    if native and guide and guide.get("status") != "current":
                        from forge_mind.workspace.native_delivery import _declared_mesh

                        if job.get("delivery_requested") or _declared_mesh(
                            (selected or {}).get("document") or {}
                        ):
                            if premature_finishes == 0:
                                premature_finishes += 1
                                history.append(
                                    {
                                        "guidance_correction": "Refresh all four customer instructions against current facts before finishing requested files.",
                                        "status": guide.get("status"),
                                    }
                                )
                                continue
                            incomplete_finish = True
                            self.store.update(sid, incomplete_finish_job_id=jid)
                    if completion.get("next_target"):
                        premature_finishes += 1
                        if premature_finishes == 1:
                            history.append(
                                {
                                    "completion_correction": completion["next_target"],
                                    "reason": "Finish requested with actionable subsystem work outstanding. Complete it or state the unresolved prerequisite.",
                                }
                            )
                            self.store.update(sid, tool_memory=remember(history))
                            continue
                        incomplete_finish = True
                        self.store.update(sid, incomplete_finish_job_id=jid)
                        self.store.event(
                            sid,
                            "review",
                            "Author stopped with unfinished subsystem work; saved candidate needs review",
                        )
                    break
                try:
                    if action.kind == "print_setup":
                        from forge_mind.workspace.print_setup_service import (
                            WORKER_TIMEOUT,
                            inspect_print_setup,
                            prepare_print_files,
                        )

                        target = action.revision_id or current
                        if target is None:
                            raise WorkspaceError("Print setup requires a saved revision")
                        print_remaining = time_left(deadline)
                        print_timeout = (
                            WORKER_TIMEOUT
                            if print_remaining is None
                            else min(WORKER_TIMEOUT, print_remaining)
                        )
                        if action.query == "inspect":
                            if not action.part_id:
                                raise WorkspaceError("Print inspection requires an exact part_id")
                            response = inspect_print_setup(
                                self,
                                sid,
                                owner,
                                target,
                                action.part_id,
                                timeout=print_timeout,
                                cancelled=author_cancelled,
                            )
                        elif action.query == "prepare":
                            response = prepare_print_files(
                                self,
                                sid,
                                owner,
                                target,
                                timeout=print_timeout,
                                cancelled=author_cancelled,
                            )
                        else:
                            raise WorkspaceError("Print setup query must be inspect or prepare")
                        history.append({"print_setup": response})
                        self.store.event(sid, "tool", "Measured saved print setup")
                        reviewing_guidance = False
                        continue
                    if action.kind == "guidance":
                        from forge_mind.workspace.delivery_guidance import (
                            GuidanceUpdate,
                            guidance_context,
                            update_guidance,
                        )

                        update = (
                            GuidanceUpdate.model_validate_json(action.document_json)
                            if action.document_json
                            else None
                        )
                        if protected_handoff and update and update.instruction_plan is None:
                            raise WorkspaceError(
                                "Protected instructions require instruction_plan with exact "
                                "instruction_catalog references and compatible actions"
                            )
                        response = (
                            update_guidance(
                                self,
                                sid,
                                owner,
                                update,
                            )
                            if update
                            else guidance_context(self, sid, owner, current)
                        )
                        history.append(
                            {"guidance": response, "explicit_guidance_read": update is None}
                        )
                        self.store.event(sid, "tool", "Read or refreshed current customer guidance")
                        reviewing_guidance = True
                        if native and action.query == "finish":
                            from forge_mind.workspace.native_finish import finish_context

                            if not action.document_json:
                                raise WorkspaceError(
                                    "Guidance finish requires an explicit reviewed instruction update"
                                )
                            final_review = finish_context(
                                self,
                                sid,
                                owner,
                                current,
                                require_bound_instructions=protected_handoff,
                            )
                            self.store.update(sid, native_review={**final_review, "job_id": jid})
                            if final_review["status"] == "ready_for_validation":
                                incomplete_finish = False
                                self.store.update(sid, incomplete_finish_job_id=None)
                                break
                            incomplete_finish = True
                            self.store.update(sid, incomplete_finish_job_id=jid)
                            history.append({"final_review_correction": final_review})
                            if (
                                protected_handoff
                                and phase == "instructions"
                                and (
                                    schedule is None
                                    or not schedule.can_correct(index + 1, time.monotonic())
                                )
                            ):
                                break
                        continue
                    if action.kind == "schema":
                        reviewing_guidance = action.query == "guidance"
                        schemas_seen.add(action.query)
                        schema_cache = cache_schema(schema_cache, action.query, schema)
                        self.store.update(
                            sid,
                            schemas_seen=sorted(schemas_seen),
                            schema_engine=schema_engine,
                            schema_cache=schema_cache,
                        )
                        history.append({"authoring_schema": schema, "schema_read": action.query})
                        continue
                    if action.kind in {"design_study", "read_design"}:
                        from forge_mind.workspace.design_study_agent import execute_study_action

                        study_result = execute_study_action(
                            self, sid, owner, job, action, current, author_cancelled
                        )
                        history.append({action.kind: study_result})
                        continue
                    if action.kind == "photo_profile":
                        if action.query == "ground":
                            from forge_mind.workspace.photo_generation import (
                                PhotoGrounding,
                                ground_target,
                            )

                            left = time_left(deadline)
                            grounded = ground_target(
                                self,
                                sid,
                                owner,
                                jid,
                                PhotoGrounding.model_validate_json(action.document_json),
                                contract,
                                timeout=30 if left is None else min(30, left),
                                cancelled=author_cancelled,
                            )
                            history.append({"frozen_photo_target": grounded})
                            continue
                        from forge_mind.workspace.photo_profile_api import (
                            PhotoProfileAction,
                            execute_photo_profile,
                        )

                        try:
                            profile_action = PhotoProfileAction.model_validate_json(
                                action.document_json
                            )
                            if action.query and action.query != profile_action.action:
                                raise ValueError("Photo profile query and action disagree")
                            observed = execute_photo_profile(
                                self,
                                sid,
                                owner,
                                profile_action,
                                timeout=30
                                if deadline is None
                                else min(30, time_left(deadline) or 0),
                                cancelled=author_cancelled,
                            )
                            from forge_mind.workspace.photo_profile_context import (
                                photo_profile_context,
                            )

                            history.append({"photo_profile": photo_profile_context(observed, sid)})
                            self.store.event(
                                sid,
                                "photo_profile",
                                "Derived photo observation; no acceptance authority",
                                action=profile_action.action,
                            )
                        except (WorkspaceError, ValueError, OSError) as exc:
                            history.append({"photo_profile_error": str(exc)})
                        continue
                    if action.kind == "source_page":
                        from forge_mind.workspace.source_observations import (
                            SourceObservationError,
                            observe_source_page,
                        )
                        from forge_mind.workspace.source_page_agent import source_page_context

                        assert action.page_number is not None
                        try:
                            observed = observe_source_page(
                                self,
                                sid,
                                owner,
                                action.query,
                                action.page_number,
                                scale=action.parameters.get("scale", 2.0),
                                timeout=30
                                if deadline is None
                                else min(30, time_left(deadline) or 0),
                                cancelled=author_cancelled,
                            )
                        except SourceObservationError as exc:
                            history.append(
                                {"source_page_error": {"code": exc.code, "message": str(exc)}}
                            )
                            self.store.event(
                                sid, "tool_error", str(exc), code=exc.code, action="source_page"
                            )
                            continue
                        page_context = source_page_context(
                            observed, action.parameters.get("offset", 0)
                        )
                        pending_source_observation = observed["id"]
                        history.append({"source_page": page_context})
                        self.store.event(
                            sid,
                            "source_page",
                            "Captured PDF page observed; claims remain unconfirmed",
                            observation_id=observed["id"],
                            source_artifact=observed["source_artifact"],
                            page_number=observed["page_number"],
                        )
                        continue
                    if action.kind == "read_source":
                        source_set = read_sources(self.store, sid, owner)
                        artifact = next(
                            (a for a in source_set["artifacts"] if a["id"] == action.query), None
                        )
                        if artifact is None:
                            raise WorkspaceError("Unknown owned source artifact")
                        offset = int(action.parameters.get("offset", 0))
                        if offset < 0 or offset > 120000:
                            raise WorkspaceError("Source offset must be 0..120000")
                        history.append(
                            {
                                "source_text": {
                                    "id": artifact["id"],
                                    "offset": offset,
                                    "text": artifact.get("text", "")[offset : offset + 16000],
                                }
                            }
                        )
                        continue
                    if action.kind == "propose_interface":
                        from forge_mind.workspace.product_interfaces import propose_interface

                        proposal = propose_interface(
                            self.store, sid, owner, json.loads(action.document_json), agent=True
                        )
                        history.append({"interface_proposal": proposal})
                        continue
                    if action.kind == "validate_interfaces":
                        from forge_mind.workspace.product_interfaces import validate_interfaces

                        if not current:
                            raise WorkspaceError("Build before checking interfaces")
                        result = validate_interfaces(
                            self,
                            sid,
                            owner,
                            current,
                            time_left(deadline),
                            author_cancelled,
                        )
                        history.append({"interface_checks": result})
                        continue
                    if action.kind in {"research", "fetch_source", "source_claim"}:
                        from forge_mind.workspace.product_sources import (
                            DimensionalClaim,
                            fetch_source,
                            propose_claim,
                            research_sources,
                        )

                        if action.kind == "source_claim":
                            sourced = propose_claim(
                                self.store,
                                sid,
                                owner,
                                DimensionalClaim.model_validate_json(action.document_json),
                                agent=True,
                            )
                        elif action.kind == "fetch_source":
                            sourced = fetch_source(
                                self,
                                sid,
                                owner,
                                action.query,
                                timeout=30
                                if deadline is None
                                else min(30, time_left(deadline) or 0),
                                cancelled=author_cancelled,
                                agent=True,
                            )
                        else:
                            sourced = research_sources(
                                self,
                                sid,
                                owner,
                                action.query,
                                timeout=time_left(deadline),
                                cancelled=author_cancelled,
                                agent=True,
                            )
                        history.append(
                            {
                                action.kind: {
                                    "version": sourced["version"],
                                    "fingerprint": sourced["fingerprint"],
                                    "latest_artifact": (
                                        {
                                            **sourced["artifacts"][-1],
                                            "text": sourced["artifacts"][-1].get("text", "")[
                                                :12000
                                            ],
                                        }
                                        if sourced["artifacts"]
                                        else None
                                    ),
                                    "latest_claim": sourced["claims"][-1]
                                    if sourced["claims"]
                                    else None,
                                }
                            }
                        )
                        continue
                    if action.kind == "lab":
                        from forge_mind.workspace.lab_agent import run_lab_action

                        if not current:
                            raise WorkspaceError(
                                "Build editable geometry before using Engineering Lab"
                            )
                        history.append(
                            {
                                "lab": run_lab_action(
                                    self,
                                    sid,
                                    owner,
                                    current,
                                    action.query,
                                    action.document_json,
                                )
                            }
                        )
                        continue
                    if action.kind in {"engineering_tool", "product_review"}:
                        from forge_mind.workspace.product_service import product_review, run_tool

                        product_result = (
                            product_review(self, sid, owner, current)
                            if action.kind == "product_review"
                            else run_tool(
                                self,
                                sid,
                                owner,
                                current,
                                json.loads(action.document_json),
                                time_left(deadline),
                                author_cancelled,
                            )
                        )
                        history.append({action.kind: product_result})
                        continue
                    if action.kind == "assembly_edit":
                        from forge_mind.workspace.authoring import AssemblyEdit, apply_assembly_edit
                        from forge_mind.workspace.document import DesignDocument

                        try:
                            existing = (
                                DesignDocument.model_validate(selected["document"])
                                if selected and selected.get("document")
                                else None
                            )
                            if selected and existing is None:
                                raise WorkspaceError(
                                    "Cannot replace a script revision with an incremental assembly edit"
                                )
                            edit = AssemblyEdit.model_validate_json(action.document_json)
                            edited = apply_assembly_edit(existing, edit)
                        except (ValueError, WorkspaceError) as exc:
                            failed_submission = rejected_submission(
                                submitted_action, current, contract.fingerprint(), str(exc)
                            )
                            self.store.update(sid, failed_submission=failed_submission)
                            raise
                        if failed_submission and failed_submission["action_kind"] == action.kind:
                            failed_submission = None
                            self.store.update(sid, failed_submission=None)
                        action = action.model_copy(
                            update={"kind": "document", "document_json": edited.model_dump_json()}
                        )
                    if action.kind == "feature_edit":
                        from forge_mind.workspace.document import DesignDocument
                        from forge_mind.workspace.edit_service import check_edit_basis
                        from forge_mind.workspace.feature_delta import (
                            FeatureDelta,
                            apply_feature_delta,
                        )

                        delta = FeatureDelta.model_validate_json(action.document_json)
                        feature_basis = delta.basis
                        if (
                            construction_patch is not None
                            and delta.basis != construction_patch.basis
                        ):
                            raise WorkspaceError(
                                "Feature edit and construction plan patch must use the same EditBasis"
                            )
                        check_edit_basis(self, sid, owner, feature_basis)
                        if current is None or feature_basis.revision_id != current:
                            raise WorkspaceError(
                                "Feature edit must target the current typed revision"
                            )
                        existing = DesignDocument.model_validate(
                            self._load_revision(sid, current)["document"]
                        )
                        edited = apply_feature_delta(existing, delta)
                        action = action.model_copy(
                            update={"kind": "document", "document_json": edited.model_dump_json()}
                        )
                    if action.kind == "study":
                        from forge_mind.workspace.parameter_study import parameter_study

                        if not current:
                            raise WorkspaceError("Build before a parameter study")
                        study = parameter_study(
                            self,
                            sid,
                            owner,
                            current,
                            contract,
                            action.query,
                            action.study_values,
                            deadline,
                            author_cancelled,
                        )
                        history.append({"parameter_study": study})
                        continue
                    if action.kind == "template":
                        if selected and (selected.get("document") or {}).get("assembly"):
                            raise WorkspaceError(
                                "Edit the existing assembly; start another project to replace it"
                            )
                        if action.query == "steering":
                            from forge_mind.workspace.steering_template import (
                                SteeringTemplate,
                                steering_document,
                            )

                            template_doc = steering_document(
                                SteeringTemplate.model_validate(action.parameters)
                            )
                        else:
                            from forge_mind.workspace.mechanism_models import MechanismTemplate
                            from forge_mind.workspace.mechanism_templates import template_document

                            template_doc = template_document(
                                MechanismTemplate.model_validate(
                                    {**action.parameters, "kind": action.query}
                                )
                            )
                        action = action.model_copy(
                            update={
                                "kind": "document",
                                "document_json": template_doc.model_dump_json(),
                            }
                        )
                    if action.kind == "catalog":
                        from forge_mind.workspace.hardware import search

                        results = search(action.query)
                        history.append(
                            {
                                "catalog": {
                                    "query": action.query,
                                    "results": results[:12],
                                    "returned": min(12, len(results)),
                                    "hint": "Search one item/family at a time; all query terms must match. Empty results are not proof of absence.",
                                }
                            }
                        )
                        continue
                    if action.kind == "motion_check":
                        if not current:
                            raise WorkspaceError("Build before checking motion")
                        from forge_mind.workspace.motion_service import check_path, run_motion_job

                        check_id = run_motion_job(
                            self,
                            {
                                **job,
                                "revision_id": current,
                                "settings": {},
                                "max_seconds": time_left(deadline),
                            },
                            author_cancelled,
                        )
                        history.append(
                            {
                                "motion_check": json.loads(
                                    check_path(self, sid, owner, check_id).read_text(
                                        encoding="utf-8"
                                    )
                                )
                            }
                        )
                        continue
                    if action.kind == "review":
                        if not current:
                            raise WorkspaceError("Build before engineering review")
                        from forge_mind.workspace.engineering_service import (
                            report_path,
                            run_report_job,
                        )

                        run_report_job(
                            self,
                            {
                                **job,
                                "revision_id": current,
                                "max_seconds": time_left(deadline),
                            },
                            author_cancelled,
                        )
                        receipt = self.store.get(sid, owner)["reports"][-1]
                        review = json.loads(
                            report_path(self, sid, owner, receipt["id"], "report.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        from forge_mind.workspace.parameter_study import summarize_report

                        history.append(
                            {
                                "engineering_report_receipt": receipt,
                                "static_interference": summarize_report(
                                    review, review["geometry"]["facts"]
                                ),
                                "engineering_review": {
                                    k: review[k]
                                    for k in (
                                        "reviews",
                                        "mechanism",
                                        "assembly_definitions",
                                        "assembly_motion",
                                        "open_items",
                                    )
                                    if k in review
                                },
                            }
                        )
                        continue
                    if action.kind in {"read_part", "edit_part"}:
                        from forge_mind.workspace.document import DesignDocument, PartDefinition

                        try:
                            doc = DesignDocument.model_validate((selected or {}).get("document"))
                            if not doc.assembly:
                                raise WorkspaceError(
                                    "Selected revision has no assembly definitions"
                                )
                            part = next(
                                (p for p in doc.assembly.parts if p.id == action.part_id), None
                            )
                            if part is None:
                                raise WorkspaceError("Unknown assembly part definition")
                            if action.kind == "read_part":
                                history.append({"part": part.model_dump()})
                                continue
                            replacement = PartDefinition.model_validate_json(action.document_json)
                            if "production" not in replacement.model_fields_set:
                                replacement.production = part.production.model_copy(deep=True)
                            if replacement.id != part.id or {p.name for p in part.ports} - {
                                p.name for p in replacement.ports
                            }:
                                raise WorkspaceError(
                                    "Scoped edit must retain part and interface identities"
                                )
                            doc.assembly.parts = [
                                replacement if p.id == part.id else p for p in doc.assembly.parts
                            ]
                            doc = DesignDocument.model_validate(doc.model_dump())
                        except (ValueError, WorkspaceError) as exc:
                            if action.kind == "edit_part":
                                failed_submission = rejected_submission(
                                    submitted_action, current, contract.fingerprint(), str(exc)
                                )
                                self.store.update(sid, failed_submission=failed_submission)
                            raise
                        if failed_submission and failed_submission["action_kind"] == action.kind:
                            failed_submission = None
                            self.store.update(sid, failed_submission=None)
                        action = action.model_copy(
                            update={"kind": "document", "document_json": doc.model_dump_json()}
                        )
                    if action.kind == "parameters" and selected and selected.get("document"):
                        from forge_mind.workspace.document import DesignDocument

                        doc = DesignDocument.model_validate(selected["document"])
                        if not action.parameters or set(action.parameters) - doc.variables.keys():
                            raise WorkspaceError(
                                "Only existing design input variables may be changed"
                            )
                        doc.variables.update(action.parameters)
                        action = action.model_copy(
                            update={"kind": "document", "document_json": doc.model_dump_json()}
                        )
                    if action.kind in {"restore", "inspect", "section"}:
                        target = action.revision_id or current
                        if not target:
                            raise WorkspaceError("No revision exists to inspect or restore")
                        revision = self._load_revision(
                            sid, target, allow_optional_display_loss=action.kind != "restore"
                        )
                        if action.kind == "restore":
                            published_revision_id = None
                            if session.get("authoring_mode") == "native_operations":
                                from forge_mind.workspace.authoring_policy import (
                                    enforce_document_policy,
                                )
                                from forge_mind.workspace.document import DesignDocument

                                if not revision.get("document"):
                                    raise WorkspaceError(
                                        "Native operation mode cannot restore Python source"
                                    )
                                enforce_document_policy(
                                    DesignDocument.model_validate(revision["document"]),
                                    "native_operations",
                                )
                            current = target
                            explicitly_restored = target
                            self.store.update(sid, selected_revision=current)
                            history.append({"restored": target})
                            if job.get("restore_only"):
                                restored_measurement = Measurement.model_validate(
                                    revision["measurement"]
                                )
                                if (
                                    restored_measurement.contract_hash != contract.fingerprint()
                                    or restored_measurement.engine_fingerprint
                                    != engine_fingerprint()
                                ):
                                    if revision.get("document"):
                                        from forge_mind.workspace.document import DesignDocument
                                        from forge_mind.workspace.document_service import (
                                            build_document,
                                        )

                                        revision = build_document(
                                            self,
                                            sid,
                                            owner,
                                            DesignDocument.model_validate(revision["document"]),
                                            contract,
                                            target,
                                            "Revalidate restored feature document",
                                            time_left(deadline),
                                            author_cancelled,
                                        )
                                    else:
                                        revision = self.build(
                                            sid,
                                            revision["code"],
                                            contract,
                                            target,
                                            "Rebuild restored source against the current contract and instruments",
                                            time_left(deadline),
                                            author_cancelled,
                                        )
                                    target = revision["id"]
                                best, best_measurement = (
                                    target,
                                    Measurement.model_validate(revision["measurement"]),
                                )
                                if photo_generation is not None:
                                    current = explicitly_restored = target
                                    self.store.update(sid, selected_revision=current)
                                    continue
                                break
                        elif action.kind == "inspect" and action.query == "prepare_handoff":
                            from forge_mind.workspace.native_handoff_prepare import (
                                PrepareHandoffRequest,
                                prepare_handoff,
                            )

                            request = PrepareHandoffRequest.model_validate_json(
                                action.document_json
                            )
                            if target != request.revision_id:
                                raise WorkspaceError("Handoff request and action revisions differ")
                            response = prepare_handoff(
                                self,
                                sid,
                                owner,
                                jid,
                                request,
                                deadline=deadline,
                                cancelled=author_cancelled,
                            )
                            if automatic_claim is not None:
                                automatic_outcome(
                                    automatic_claim,
                                    automatic_started,
                                    {"status": response["status"]},
                                )
                                automatic_recorded = True
                            preparation_origin = (
                                {
                                    "origin": "host",
                                    "job_id": jid,
                                    "action_index": index,
                                    "automatic_handoff_attempt": automatic_claim,
                                }
                                if automatic_claim is not None
                                else {}
                            )
                            history.append(
                                {
                                    "handoff_preparation": response,
                                    "revision_id": target,
                                    **preparation_origin,
                                }
                            )
                            if response["status"] == "prepared_for_instruction_review":
                                schemas_seen.add("guidance")
                                schema_cache = cache_schema(
                                    schema_cache, "guidance", response["guidance_schema"]
                                )
                                self.store.update(
                                    sid,
                                    schemas_seen=sorted(schemas_seen),
                                    schema_engine=schema_engine,
                                    schema_cache=schema_cache,
                                )
                                reviewing_guidance = True
                            self.store.event(
                                sid,
                                "inspection",
                                "Prepared current handoff inputs",
                                revision_id=target,
                                handoff_preparation=response,
                                **preparation_origin,
                            )
                        elif action.kind == "inspect" and action.query == "requirements":
                            from forge_mind.workspace.requirement_verification import (
                                inspect_requirements,
                            )

                            response = inspect_requirements(
                                self,
                                sid,
                                owner,
                                target,
                                deadline=deadline,
                                cancelled=author_cancelled,
                            )
                            history.append(
                                {"requirements_inspection": response, "revision_id": target}
                            )
                            self.store.event(
                                sid,
                                "inspection",
                                "Inspected frozen geometric requirements",
                                revision_id=target,
                                requirements_inspection=response,
                            )
                        elif action.kind == "inspect" and action.query in {
                            "source",
                            "diff",
                            "topology",
                        }:
                            observation: dict[str, Any] = {"revision_id": target}
                            if action.query == "topology":
                                from forge_mind.workspace.document_service import read_topology

                                observation["source_topology"] = read_topology(self, sid, target)
                            elif action.query == "source":
                                observation["source"] = revision["code"]
                                if revision.get("document") is not None:
                                    observation["source_document"] = revision["document"]
                            else:
                                if "diff" not in revision:
                                    raise WorkspaceError("This revision has no saved source diff")
                                observation["source_diff"] = revision["diff"]
                            history.append(observation)
                        elif action.kind == "section":
                            target_dir = self.store.root / "sections" / sid
                            target_dir.mkdir(parents=True, exist_ok=True)
                            with tempfile.TemporaryDirectory(
                                prefix="forge-cad-section-"
                            ) as scratch:
                                from pathlib import Path

                                process_section = (
                                    process_revision
                                    if self.executor.mode == "trusted_local"
                                    else self.executor.process_observation
                                )
                                section = process_section(
                                    Path(scratch),
                                    {
                                        "operation": "section",
                                        "source": str(self.store.directory(sid, target)),
                                        "axis": action.axis,
                                        "offset_mm": action.offset_mm,
                                    },
                                    time_left(deadline),
                                    author_cancelled,
                                )
                                if section.get("image"):
                                    inspection_image = target_dir / f"{uuid.uuid4().hex}.png"
                                    copy_artifact(Path(scratch) / "section.png", inspection_image)
                                    inspection_revision_id = target
                                    inspection_instrument = {
                                        "axis": action.axis,
                                        "offset_mm": action.offset_mm,
                                    }
                                    section["image"] = inspection_image.name
                            history.append({"section": section, "revision_id": target})
                            self.store.event(
                                sid,
                                "section",
                                "Cross-section computed",
                                revision_id=target,
                                section=section,
                                **{
                                    key: section[key]
                                    for key in ("axis", "offset_mm", "loops", "image")
                                    if key in section
                                },
                            )
                        else:
                            history.append(
                                {
                                    "inspection": assembly_measurement_context(
                                        revision["measurement"]
                                    ),
                                    "revision_id": target,
                                }
                            )
                        continue
                    code = action.code
                    if action.kind == "document":
                        from forge_mind.workspace.document import DesignDocument
                        from forge_mind.workspace.document_service import build_document

                        try:
                            document = DesignDocument.model_validate_json(action.document_json)
                        except ValidationError as exc:
                            failed_submission = rejected_submission(
                                submitted_action, current, contract.fingerprint(), str(exc)
                            )
                            self.store.update(sid, failed_submission=failed_submission)
                            raise
                        failed_submission = None
                        self.store.update(sid, failed_submission=None)
                        prepared_construction = None
                        if construction_patch is not None:
                            from forge_mind.workspace.construction_transaction import (
                                prepare_construction,
                            )

                            prepared_construction = prepare_construction(
                                self,
                                sid,
                                owner,
                                jid,
                                document,
                                construction_patch,
                                author_intent=intent,
                            )
                        if native:
                            from forge_mind.workspace.customer_construction import (
                                stage_construction,
                            )

                            stage_construction(
                                self,
                                sid,
                                owner,
                                jid,
                                document,
                                contract,
                                current,
                                action.explanation,
                                prepared=prepared_construction,
                                feature_basis=feature_basis,
                                remaining_seconds=time_left(author_deadline),
                                remaining_actions=None
                                if action_limit is None
                                else max(0, action_limit - index - 1),
                            )
                        digest = document.fingerprint()
                        if prepared_construction is not None:
                            # An identical document with explicit new ownership is
                            # a distinct build against a new operation snapshot.
                            digest += ":" + prepared_construction.successor_snapshot["plan_hash"]
                        if digest in hashes and (
                            (selected or {}).get("operation_record", {}).get("plan_hash")
                            == self.store.get(sid, owner).get("plan_hash")
                            or not native
                        ):
                            raise WorkspaceError(
                                "Identical feature document already attempted; inspect the failure or change it"
                            )
                        hashes.add(digest)
                        construction_receipt = None
                        if prepared_construction is not None:
                            from forge_mind.workspace.construction_transaction import (
                                build_construction,
                            )

                            construction_receipt = build_construction(
                                self,
                                sid,
                                owner,
                                jid,
                                prepared_construction,
                                explanation=action.explanation,
                                timeout=time_left(deadline),
                                cancelled=author_cancelled,
                            )
                            history.append({"construction_transaction": construction_receipt})
                            if not construction_receipt.get("applied"):
                                failed_submission = rejected_submission(
                                    submitted_action,
                                    current,
                                    contract.fingerprint(),
                                    str(
                                        construction_receipt.get(
                                            "code", "Construction was not applied"
                                        )
                                    ),
                                )
                                self.store.update(sid, failed_submission=failed_submission)
                            revision = self._load_revision(
                                sid, construction_receipt["candidate_revision_id"]
                            )
                        else:
                            revision = build_document(
                                self,
                                sid,
                                owner,
                                document,
                                contract,
                                current,
                                action.explanation,
                                time_left(deadline),
                                author_cancelled,
                            )
                        measured = Measurement.model_validate(revision["measurement"])
                        if job.get("parameters"):
                            edited_candidate = revision["id"]
                        passed = (
                            {
                                f.requirement_id
                                for f in best_measurement.findings
                                if f.status == "pass"
                            }
                            if best_measurement
                            else set()
                        )
                        regressions = [
                            f.requirement_id
                            for f in measured.findings
                            if f.requirement_id in passed and f.status != "pass"
                        ]
                        history.append(
                            {
                                "revision": revision["id"],
                                "measurement": assembly_measurement_context(measured.model_dump()),
                                "regressions": regressions,
                            }
                        )
                        if (
                            inferred_cad_action
                            and construction_receipt is None
                            and author_cancelled()
                        ):
                            # Atomic construction checks its own publication boundary.
                            # Retain this nontransaction candidate without selecting it.
                            self.store.event(
                                sid,
                                "construction_not_published",
                                "Authoring stopped before selecting this candidate",
                                revision_id=revision["id"],
                                reason="cancelled" if cancelled() else "deadline",
                            )
                            continue
                        if measured.geometry_valid:
                            publication = construction_receipt
                            native_publication_applied = (
                                feature_basis is None and construction_receipt is None
                            )
                            if construction_receipt is not None:
                                if not construction_receipt.get("applied"):
                                    if replaying_construction:
                                        return stop_claimed_construction(
                                            "Approved atomic construction did not publish"
                                        )
                                    continue
                                native_publication_applied = True
                            elif feature_basis is not None:
                                receipt = self.store.publish_feature_edit(
                                    sid, owner, jid, feature_basis.model_dump(), revision["id"]
                                )
                                history.append({"feature_edit": receipt})
                                publication = receipt
                                native_publication_applied = receipt.get("applied") is True
                            current = revision["id"]
                            if feature_basis is None and construction_receipt is None:
                                self.store.update(sid, selected_revision=current)
                            if native and native_publication_applied:
                                published_revision_id = current
                            if replaying_construction:
                                if not native_publication_applied:
                                    return stop_claimed_construction(
                                        "Approved candidate did not publish against its saved basis"
                                    )
                                from forge_mind.workspace.customer_construction import (
                                    complete_construction,
                                )

                                complete_construction(
                                    self.store, sid, owner, jid, current, publication=publication
                                )
                                claimed_construction = None
                            if not regressions and (
                                best_measurement is None
                                or _rank(measured) >= _rank(best_measurement)
                            ):
                                best, best_measurement = current, measured
                                self.store.update(sid, best_revision=best)
                        elif feature_basis is not None and construction_receipt is None:
                            receipt = {
                                "action": "feature_edit",
                                "job_id": jid,
                                "applied": False,
                                "base_revision_id": feature_basis.revision_id,
                                "candidate_revision_id": revision["id"],
                                "code": "invalid_geometry",
                            }
                            self.store.update(sid, last_edit_result=receipt)
                            history.append({"feature_edit": receipt})
                        if replaying_construction and claimed_construction is not None:
                            return stop_claimed_construction(
                                "Approved construction produced invalid geometry or no accepted publication"
                            )
                        if job.get("parameters"):
                            # Numeric edits must not ask an author to rewrite the document.
                            if photo_generation is not None:
                                continue
                            break
                        continue
                    if action.kind == "parameters":
                        if selected is None:
                            raise WorkspaceError("Build a model before editing parameters")
                        spec = extract_params(selected["code"])
                        if (
                            spec is None
                            or not action.parameters
                            or set(action.parameters) - set(spec.float_names)
                        ):
                            raise WorkspaceError(
                                "Only existing named float parameters may be changed"
                            )
                        code = override_params(selected["code"], spec, action.parameters)
                    if not code.strip():
                        raise WorkspaceError("Build action requires CAD source")
                    import hashlib

                    digest = hashlib.sha256(code.encode()).hexdigest()
                    if digest in hashes:
                        raise WorkspaceError(
                            "Identical source already attempted; inspect the failure or change construction strategy"
                        )
                    hashes.add(digest)
                    revision = self.build(
                        sid,
                        code,
                        contract,
                        current,
                        action.explanation,
                        time_left(deadline),
                        author_cancelled,
                    )
                    measured = Measurement.model_validate(revision["measurement"])
                    if job.get("parameters"):
                        edited_candidate = revision["id"]
                    regressions = []
                    if best_measurement:
                        formerly_passed = {
                            f.requirement_id
                            for f in best_measurement.findings
                            if f.status == "pass"
                        }
                        regressions = [
                            f.requirement_id
                            for f in measured.findings
                            if f.requirement_id in formerly_passed and f.status != "pass"
                        ]
                    history.append(
                        {
                            "revision": revision["id"],
                            "measurement": measured.model_dump(),
                            "regressions": regressions,
                        }
                    )
                    if measured.geometry_valid:
                        current = revision["id"]
                        self.store.update(sid, selected_revision=current)
                    if (
                        measured.geometry_valid
                        and not regressions
                        and (best_measurement is None or _rank(measured) >= _rank(best_measurement))
                    ):
                        best, best_measurement = revision["id"], measured
                        self.store.update(sid, best_revision=best)
                    if job.get("parameters"):
                        # A user's numeric edit must never trigger an unrequested model rewrite.
                        if photo_generation is not None:
                            continue
                        break
                except (ValueError, OSError) as exc:
                    from forge_mind.workspace.customer_approval import CustomerApprovalRequiredError

                    if replaying_construction:
                        return stop_claimed_construction(str(exc))
                    if isinstance(exc, CustomerApprovalRequiredError):
                        self.store.update(
                            sid,
                            failed_submission=rejected_submission(
                                submitted_action, current, contract.fingerprint(), str(exc)
                            ),
                            customer_approval_actions_left=None
                            if action_limit is None
                            else max(0, action_limit - index - 1),
                        )
                        return "needs_approval"
                    if automatic_claim is not None and not automatic_recorded:
                        automatic_outcome(
                            automatic_claim,
                            automatic_started,
                            {"status": "error", "type": type(exc).__name__, "message": str(exc)},
                        )
                        automatic_recorded = True
                    from forge_mind.workspace.action_payloads import InputValidationError

                    rejection = (
                        exc.error_result()
                        if isinstance(exc, InputValidationError)
                        else {
                            "code": "rejected",
                            "message": str(exc)[:1000],
                            "retryable": True,
                            "applied": False,
                        }
                    )
                    history.append({"tool_error": str(exc)[:1000], "action_result": rejection})
                    if plan_patch_json:
                        failed_submission = rejected_submission(
                            submitted_action, current, contract.fingerprint(), str(exc)
                        )
                        self.store.update(sid, failed_submission=failed_submission)
                    if submitted_action["kind"] == "feature_edit":
                        self.store.update(
                            sid,
                            last_edit_result={"action": "feature_edit", "job_id": jid, **rejection},
                        )
                    self.store.event(sid, "tool_error", str(exc)[:1000])
                    from forge_mind.workspace.store import EditConflictError

                    if isinstance(exc, EditConflictError):
                        self.store.update(sid, authority_conflict_job_id=jid)
                        return "needs_review"
                    if job.get("parameters"):
                        break
                finally:
                    if automatic_claim is not None and not automatic_recorded:
                        import sys

                        error = sys.exc_info()[1]
                        automatic_outcome(
                            automatic_claim,
                            automatic_started,
                            {
                                "status": "error" if error is not None else "not_executed",
                                "type": type(error).__name__ if error is not None else None,
                                "message": str(error)
                                if error is not None
                                else "Action not applied",
                            },
                        )
                    from forge_mind.workspace.agent_memory import remember

                    self.store.update(sid, tool_memory=remember(history))
            else:
                exhausted_actions = True
            if claimed_construction is not None:
                return stop_claimed_construction(
                    "Original action budget ended before approved dispatch"
                )
            final_revision = (
                (edited_candidate or current) if job.get("parameters") else best or current
            )
            if explicitly_restored == current and current and not job.get("restore_only"):
                final_revision = current
            if (
                current
                and not job.get("restore_only")
                and (
                    (self.store.revision(sid, current).get("document") or {}).get("assembly")
                    or (
                        self.store.authoring_mode(sid) == "native_operations"
                        and self.store.revision(sid, current).get("document")
                    )
                )
            ):
                # Show the latest valid native document or assembly, including unresolved requirements,
                # rather than silently returning an earlier incomplete numeric winner.
                final_revision = current
            final_measurement = (
                Measurement.model_validate(self.store.revision(sid, final_revision)["measurement"])
                if final_revision
                else None
            )
            if construction_outcome is not None:
                from forge_mind.workspace.customer_construction import (
                    construction_finalization_allowed,
                )

                if final_revision != self.store.get(sid, owner).get(
                    "selected_revision"
                ) or not construction_finalization_allowed(self.store, sid, owner, jid):
                    return stop_claimed_construction(
                        "Published construction changed before final review; refresh current saved state"
                    )
                # The publication already selected this exact revision. Repeating
                # that write would change the sealed selection counter.
                self.store.update(sid, usage=usage)
            else:
                self.store.update(sid, selected_revision=final_revision, usage=usage)
            state = (
                "needs_review"
                if incomplete_finish
                else "contract_satisfied"
                if final_measurement
                and (not job.get("parameters") or edited_candidate is not None)
                and final_measurement.contract_hash == contract.fingerprint()
                and final_measurement.engine_fingerprint == engine_fingerprint()
                and final_measurement.requirements_satisfied
                else "budget_exhausted"
                if exhausted_actions
                else "needs_review"
            )
            if photo_generation is not None:
                from forge_mind.workspace.photo_generation import photo_closure

                if photo_closure(self, sid, owner, final_revision)["status"] != "checked":
                    state = "needs_review"
            return state
        except Exception as exc:
            if claimed_construction is not None:
                return stop_claimed_construction(f"{type(exc).__name__}: {exc}")
            self.store.event(sid, "error", f"{type(exc).__name__}: {str(exc)[:500]}")
            return (
                "cancelled" if cancelled() else "budget_exhausted" if expired(deadline) else "error"
            )
