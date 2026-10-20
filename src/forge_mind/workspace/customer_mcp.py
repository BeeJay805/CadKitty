"""Versioned ordinary customer workflow on the existing authenticated MCP server.

The tools reuse the customer Store admission, command approval and exact package
readers. They introduce no alternate CAD pipeline or caller-selected identity.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.auth.provider import AccessToken
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, StrictBool, field_validator

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.connector_auth import ScopedMCPServer, scoped_tool
from forge_mind.workspace.connector_downloads import ConnectorDownloads
from forge_mind.workspace.customer_approval import (
    ApprovalRequest,
    ProposalRevisionRequest,
    approve_proposal,
    read_proposal,
    revise_proposal,
)
from forge_mind.workspace.customer_photo import (
    ChatGPTPhotoFile,
    create_from_photo,
    photo_capability,
)
from forge_mind.workspace.customer_projects import (
    CustomerProjectRequest,
    customer_capabilities,
    customer_project,
    hosted_customer_limits,
)
from forge_mind.workspace.customer_references import CustomerReference, KnownDimension
from forge_mind.workspace.delivery_guidance import guidance_context
from forge_mind.workspace.intent_attribution_service import attribution_summary, read_attribution
from forge_mind.workspace.native_delivery import current_delivery_binding
from forge_mind.workspace.photo_profile_api import PhotoProfileAction, execute_photo_profile
from forge_mind.workspace.print_setup_service import print_setup_context, read_print_export
from forge_mind.workspace.project_access import project_overview
from forge_mind.workspace.project_delivery import read_project_export
from forge_mind.workspace.requirement_verification import read_verification
from forge_mind.workspace.store import WorkspaceError

VERSION = "customer_workflow_v1"
Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
MAX_PHOTO_BYTES = 512 * 1024
MAX_PHOTO_BASE64 = ((MAX_PHOTO_BYTES + 2) // 3) * 4
INSTRUCTIONS = (
    "For an ordinary customer request, prefer customer_workflow and start_customer_project. "
    "Use a stable request_id and the owner's short prompt, plus an optional owned photo and known measurement. "
    "Do not ask the owner to complete a CAD schema. Preserve original duties and unknowns. "
    "For a host-supplied ChatGPT photo, use start_customer_project_from_photo with its top-level file object; never invent download URLs or ask the model to encode base64. "
    "Photo import is unavailable until an operator configures approved CDN hosts and remains unqualified for actual ChatGPT files. "
    "If unavailable, continue from an ordinary text request; a website upload is an alternative only on a matching enabled private host. "
    "Recover a lost creation response with recover_customer_request before exact replay. "
    "Use get_customer_progress for current questions, proposal state, CAD conformance and literal guidance. "
    "Use resolve_intent with current version/hash, raw owner answers, and max_seconds=null to retain the original budget; answers never approve CAD. "
    "If approval is requested, show get_customer_proposal and call approve_customer_proposal only after explicit owner approval of that exact proposal. "
    "Use revise_customer_proposal for a changed pending proposal, or edit_customer_project after CAD exists. "
    "Do not retry an uncertain edit automatically; read progress first. Use resume_customer_project for interrupted work and cancel_job to stop it. "
    "Use get_customer_files to read the already captured current project and print packages. It never accepts CAD or starts new exports. "
    "Files, nominal CAD conformance and unknown physical duties have separate meanings. Never infer physical readiness or complete request satisfaction from file delivery. "
    "Advanced create_project/submit_document tools remain available for explicit engineering workflows. "
    "The ordinary workflow version is customer_workflow_v1. This remains a private staging service, not a published connector."
)


class CustomerMCPReference(CustomerReference):
    """The same owned reference contract, bounded to the existing 1 MiB MCP envelope."""

    content_base64: str = Field(min_length=1, max_length=MAX_PHOTO_BASE64)

    @field_validator("content_base64")
    @classmethod
    def bounded_photo(cls, value: str) -> str:
        try:
            content = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Photo must contain valid base64 image bytes") from exc
        if len(content) > MAX_PHOTO_BYTES:
            raise ValueError("Photo exceeds the 512 KiB MCP limit")
        return value


class CustomerWorkflow:
    def __init__(self, service: WorkspaceService):
        self.service = service

    def owned(self, project_id: str, principal: str) -> dict[str, Any]:
        if len(project_id) != 32 or any(c not in "0123456789abcdef" for c in project_id):
            raise WorkspaceError("Invalid project identifier")
        session = self.service.store.session_record(project_id, principal)
        if not customer_project(session):
            raise WorkspaceError("This project is not an ordinary customer workspace")
        return session

    def qualified(self) -> None:
        if not customer_capabilities(self.service.executor)["enabled"]:
            raise WorkspaceError(
                "Customer delivery requires a qualified worker and explicit host budgets"
            )

    @staticmethod
    def queued(session: dict[str, Any]) -> dict[str, Any]:
        return {
            "workflow_version": VERSION,
            **{
                key: session.get(key)
                for key in ("id", "job_id", "state", "selected_revision", "creation_request_id")
            },
        }

    def create(self, principal: str, request: CustomerProjectRequest) -> dict[str, Any]:
        self.qualified()
        result = self.service.store.create_customer_project(
            principal, request.model_dump(mode="json"), hosted_limits=hosted_customer_limits()
        )
        self.service.start()
        return self.queued(result)

    def recover(self, principal: str, request_id: str) -> dict[str, Any]:
        return self.queued(self.service.store.customer_request(principal, request_id))

    def progress(self, principal: str, project_id: str) -> dict[str, Any]:
        session = self.owned(project_id, principal)
        overview = project_overview(self.service.store, project_id, principal)
        record = read_attribution(self.service.store, project_id, principal)["record"]
        proposal = read_proposal(self.service.store, project_id, principal)
        try:
            conformance = read_verification(self.service.store, project_id, principal)
        except (WorkspaceError, ValueError, OSError, KeyError, TypeError) as exc:
            conformance = {"status": "unverified", "results": [], "reason": str(exc)}
        guide: dict[str, Any] = {"status": "unavailable", "literal_procedures": None}
        rid = overview.get("selected_revision")
        if rid:
            try:
                context = guidance_context(self.service, project_id, principal, rid)
                current = (
                    context.get("status") == "current" and context.get("selected_revision") == rid
                )
                guide = {
                    "status": context.get("status", "unavailable"),
                    "revision_id": rid,
                    "basis_hash": context.get("basis_hash"),
                    "instruction_binding": context.get("instruction_binding"),
                    "literal_procedures": (context.get("record") or {}).get("guidance")
                    if current
                    else None,
                    "scope": context.get("scope"),
                }
            except (WorkspaceError, ValueError, OSError) as exc:
                guide = {"status": "unavailable", "literal_procedures": None, "reason": str(exc)}
        latest = self.owned(project_id, principal)
        if latest.get("selected_revision") != rid or latest.get("job_id") != session.get("job_id"):
            guide = {
                "status": "stale",
                "literal_procedures": None,
                "reason": "Project changed during this read; refresh progress",
            }
            conformance = {
                "status": "unverified",
                "results": [],
                "reason": "Project changed during this read",
            }
        return {
            "workflow_version": VERSION,
            **{
                key: overview.get(key)
                for key in (
                    "id",
                    "state",
                    "job_id",
                    "selected_revision",
                    "design_revision",
                    "input_version",
                    "revision_count",
                    "current_completion_status",
                )
            },
            "questions": attribution_summary(record),
            "proposal": {key: proposal.get(key) for key in ("status", "proposal_hash", "basis")},
            "cad_conformance": conformance,
            "guidance": guide,
            "file_capture_status": (latest.get("automatic_delivery") or {}).get(
                "status", "unavailable"
            ),
            "physical_readiness": "unverified",
            "scope": "Current saved lifecycle, checked CAD predicates and literal author guidance. Unknown duties remain unknown. Neither author text nor files establish full function or physical readiness.",
        }

    def proposal(self, principal: str, project_id: str) -> dict[str, Any]:
        self.owned(project_id, principal)
        return {
            "workflow_version": VERSION,
            **read_proposal(self.service.store, project_id, principal),
        }

    def approve(self, principal: str, project_id: str, approval: ApprovalRequest) -> dict[str, Any]:
        self.owned(project_id, principal)
        self.qualified()
        jid = approve_proposal(self.service.store, project_id, principal, approval)
        self.service.start()
        return {
            "workflow_version": VERSION,
            "id": project_id,
            "job_id": jid,
            "state": self.owned(project_id, principal)["state"],
        }

    def revise(
        self, principal: str, project_id: str, request: ProposalRevisionRequest
    ) -> dict[str, Any]:
        self.owned(project_id, principal)
        self.qualified()
        jid = revise_proposal(self.service.store, project_id, principal, request)
        self.service.start()
        return {
            "workflow_version": VERSION,
            "id": project_id,
            "job_id": jid,
            "state": self.owned(project_id, principal)["state"],
        }

    def edit(
        self,
        principal: str,
        project_id: str,
        instruction: str,
        revision_id: str,
        input_version: int,
    ) -> dict[str, Any]:
        session = self.owned(project_id, principal)
        self.qualified()
        if session["state"] == "needs_approval":
            raise WorkspaceError("Use the exact proposal approval or proposal revision action")
        if not instruction.strip():
            raise WorkspaceError("A written customer edit is required")
        self.service.store.revision(project_id, revision_id)
        jid = self.service.store.enqueue(
            project_id,
            principal,
            {
                "instruction": instruction,
                "revision_id": revision_id,
                "expected_current_revision": revision_id,
                "expected_input_version": input_version,
                "preview_only": False,
                "delivery_requested": True,
                **(hosted_customer_limits() or {}),
            },
        )
        self.service.start()
        return {"workflow_version": VERSION, "id": project_id, "job_id": jid, "state": "queued"}

    def resume(self, principal: str, project_id: str) -> dict[str, Any]:
        session = self.owned(project_id, principal)
        self.qualified()
        if session["state"] == "needs_approval":
            raise WorkspaceError("Approve the current saved command before resuming construction")
        jid = self.service.store.enqueue(
            project_id,
            principal,
            {
                "resume": True,
                "preview_only": False,
                "delivery_requested": True,
            },
        )
        self.service.start()
        return {"workflow_version": VERSION, "id": project_id, "job_id": jid, "state": "queued"}

    def files(self, principal: str, project_id: str) -> dict[str, Any]:
        session = self.owned(project_id, principal)
        delivery = session.get("automatic_delivery") or {}
        result: dict[str, Any] = {
            "workflow_version": VERSION,
            "id": project_id,
            "status": "not_ready",
            "project": None,
            "print": None,
            "physical_readiness": "unverified",
            "scope": "Exact saved digital packages only; no acceptance, regeneration or physical approval.",
        }
        if delivery.get("status") != "delivered" or delivery.get("readback_verified") is not True:
            return result
        if not session.get("job_id") or delivery.get("job_id") != session["job_id"]:
            return {
                **result,
                "status": "stale",
                "reason": "Captured package belongs to a different customer job",
            }
        frozen = current_delivery_binding(self.service, project_id, principal)
        if any(delivery.get(key) != value for key, value in frozen.items()):
            return {
                **result,
                "status": "stale",
                "reason": "Captured package does not match current saved inputs",
            }
        receipt = delivery.get("receipt") or {}
        package_id, package_hash = receipt.get("package_id"), receipt.get("sha256")
        if not isinstance(package_id, str) or not isinstance(package_hash, str):
            raise WorkspaceError("Captured customer package has no immutable identity")
        content, checked = read_project_export(
            self.service, project_id, principal, package_id, package_hash
        )
        if (
            checked != receipt
            or receipt.get("revision_id") != frozen["revision_id"]
            or len(content) != receipt.get("bytes")
            or hashlib.sha256(content).hexdigest() != receipt.get("sha256")
        ):
            raise WorkspaceError("Captured customer package differs from exact owned readback")
        result["project"] = {
            **receipt,
            "resource_uri": f"cadkitty://projects/{project_id}/exports/{receipt['package_id']}/{receipt['sha256']}.zip",
        }
        if session.get("customer_printable") is True:
            setup = print_setup_context(self.service, project_id, principal, frozen["revision_id"])
            package = (setup.get("record") or {}).get("package")
            if setup.get("status") != "current" or not package:
                return {
                    **result,
                    "status": "stale",
                    "reason": "Requested print package is not current",
                }
            _, verified = read_print_export(
                self.service, project_id, principal, package["package_id"], package["sha256"]
            )
            if verified != package or package.get("revision_id") != frozen["revision_id"]:
                raise WorkspaceError("Print package differs from the current revision")
            result["print"] = {
                **package,
                "resource_uri": f"cadkitty://projects/{project_id}/print-exports/{package['package_id']}/{package['sha256']}",
            }
        if current_delivery_binding(self.service, project_id, principal) != frozen:
            return {
                **result,
                "status": "stale",
                "project": None,
                "print": None,
                "reason": "Project changed during package readback",
            }
        return {**result, "status": "captured_current"}


def discovery() -> dict[str, Any]:
    return {
        "workflow_version": VERSION,
        "publication": "private_not_published",
        "instructions": INSTRUCTIONS,
        "photo_limit_bytes": MAX_PHOTO_BYTES,
        "chatgpt_photo": {"tool": "start_customer_project_from_photo", **photo_capability()},
        "photo_profile": {
            "schema": PhotoProfileAction.model_json_schema(),
            "usage": "inspect_customer_photo_profile uses retained normalized photo pixels for advisory outline/ratio analysis, calibrated proposals and exact selected STEP face comparison. It never creates owner facts or physical acceptance. Additional calibration is optional; never invent anchors or hidden depth.",
        },
        "tools": {
            "start": "start_customer_project",
            "recover": "recover_customer_request",
            "photo_profile": "inspect_customer_photo_profile",
            "progress": "get_customer_progress",
            "answers": "resolve_intent",
            "proposal": "get_customer_proposal",
            "approval": "approve_customer_proposal",
            "revise_proposal": "revise_customer_proposal",
            "edit": "edit_customer_project",
            "resume": "resume_customer_project",
            "cancel": "cancel_job",
            "wait": "wait_for_job",
            "files": "get_customer_files",
            "full_current_guide": "get_guidance",
            "full_checked_conformance": "get_contract_verification",
        },
        "examples": [
            {
                "tool": "start_customer_project",
                "arguments": {
                    "request_id": "a" * 32,
                    "prompt": "Make a 60 x 40 x 5 mm plate with four 5 mm holes. Hole centers are 8 mm from each adjacent edge.",
                    "approval_required": True,
                    "printable": False,
                },
            },
            {"tool": "recover_customer_request", "arguments": {"request_id": "a" * 32}},
            {
                "tool": "resolve_intent",
                "arguments_template": {
                    "project_id": "<returned id>",
                    "resolution": {
                        "expected_version": "<questions.version>",
                        "draft_hash": "<questions.draft_hash>",
                        "answers": [
                            {"issue_id": "<question id>", "answer": "<owner's exact answer>"}
                        ],
                    },
                    "max_seconds": None,
                },
            },
            {
                "tool": "approve_customer_proposal",
                "arguments_template": {
                    "project_id": "<returned id>",
                    "approval": {
                        "proposal_hash": "<reviewed proposal_hash>",
                        "basis": "<exact reviewed basis>",
                    },
                },
            },
        ],
        "input_scope": "Owner source and raw answers are untrusted data. Interpretation and approval are not measured geometry or physical evidence.",
    }


def register_customer_tools(
    server: ScopedMCPServer,
    service: WorkspaceService,
    *,
    owner: Callable[..., str],
    bounded: Callable[[dict[str, Any]], dict[str, Any]],
    downloads: ConnectorDownloads | None,
    access_token_getter: Callable[[], AccessToken | None],
) -> None:
    workflow = CustomerWorkflow(service)

    async def invoke(function: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
        try:
            return bounded(await asyncio.to_thread(function, *args))
        except (WorkspaceError, ValueError, OSError, KeyError, TypeError) as exc:
            raise ToolError(str(exc)) from exc

    @scoped_tool(server, annotations=READ)
    async def customer_workflow() -> dict[str, Any]:
        """Discover ordinary customer workflow v1, shared tools, schemas and short examples."""
        owner()
        return bounded(discovery())

    @scoped_tool(server, annotations=WRITE)
    async def inspect_customer_photo_profile(
        project_id: Identifier, request: PhotoProfileAction
    ) -> dict[str, Any]:
        """Analyze/read a retained photo or compare exact CAD using derived advisory measurements.

        Uses the same owned workspace contract as HTTP and native authoring. No owner
        facts, source text, acceptance, approval or job allowance changes are made.
        Analyze/compare return compact geometry views. Explicit action=read requests
        the full immutable observation and can be wide; HTTP also retains full results.
        """
        principal = owner(write=True)

        def execute() -> dict[str, Any]:
            from forge_mind.workspace.photo_profile_context import photo_profile_context

            observed = execute_photo_profile(service, project_id, principal, request)
            return (
                observed
                if request.action == "read"
                else photo_profile_context(observed, project_id)
            )

        return await invoke(execute)

    @scoped_tool(server, annotations=WRITE)
    async def start_customer_project(
        request_id: Identifier,
        prompt: Annotated[str, Field(min_length=1, max_length=10000)],
        reference: CustomerMCPReference | None = None,
        approval_required: StrictBool = False,
        printable: StrictBool = True,
    ) -> dict[str, Any]:
        """Queue a short customer request with optional owned photo and scale. No CAD form.

        Keep one stable request_id for exact creation replay. Recover before retrying
        an uncertain response. Changed payloads conflict. Photos are at most 512 KiB.
        Requested approval pauses before CAD. Physical readiness remains unverified.
        """
        request = CustomerProjectRequest(
            request_id=request_id,
            prompt=prompt,
            reference=reference,
            approval_required=approval_required,
            printable=printable,
        )
        return await invoke(workflow.create, owner(write=True), request)

    @scoped_tool(server, annotations=READ)
    async def recover_customer_request(request_id: Identifier) -> dict[str, Any]:
        """Recover only this owner's original request and job without starting work."""
        return await invoke(workflow.recover, owner(), request_id)

    @scoped_tool(server, annotations=WRITE, file_params=("file",))
    async def start_customer_project_from_photo(
        request_id: Identifier,
        prompt: Annotated[str, Field(min_length=1, max_length=10000)],
        file: ChatGPTPhotoFile,
        known_dimension: KnownDimension | None = None,
        approval_required: StrictBool = False,
        printable: StrictBool = True,
    ) -> dict[str, Any]:
        """Start from the host-supplied photo file object and short owner request.

        download_url and file_id are required; mime_type and file_name are optional.
        Do not synthesize URLs or encode the image. Approved HTTPS CDN hosts are
        mandatory; configuration is not actual ChatGPT-file qualification. Signed
        URLs are never retained. Exact replay/recovery uses the original owned image.
        """
        principal = owner(write=True)

        def create() -> dict[str, Any]:
            session = create_from_photo(
                service,
                principal,
                request_id=request_id,
                prompt=prompt,
                file=file,
                known_dimension=known_dimension,
                approval_required=approval_required,
                printable=printable,
            )
            return workflow.queued(session)

        return await invoke(create)

    @scoped_tool(server, annotations=READ)
    async def get_customer_progress(project_id: Identifier) -> dict[str, Any]:
        """Read focused current questions, proposal, literal guide and checked CAD conformance.

        Author text and file status never establish physical readiness. Unknown duties
        remain visible. Use existing wait_for_job for bounded asynchronous waiting.
        """
        return await invoke(workflow.progress, owner(), project_id)

    @scoped_tool(server, annotations=READ)
    async def get_customer_proposal(project_id: Identifier) -> dict[str, Any]:
        """Read exact pending command/plan and basis for owner review before approval."""
        return await invoke(workflow.proposal, owner(), project_id)

    @scoped_tool(server, annotations=WRITE)
    async def approve_customer_proposal(
        project_id: Identifier, approval: ApprovalRequest
    ) -> dict[str, Any]:
        """Approve only the exact proposal explicitly reviewed and approved by the owner.

        Send its unchanged hash and full basis. Source or plan changes refuse approval.
        This reuses frozen command approval and cannot extend the original budget.
        """
        return await invoke(workflow.approve, owner(write=True), project_id, approval)

    @scoped_tool(server, annotations=WRITE)
    async def revise_customer_proposal(
        project_id: Identifier, request: ProposalRevisionRequest
    ) -> dict[str, Any]:
        """Submit an owner's written change to the exact pending proposal; no approval.

        Its stable request_id replays the same revision job. Changed replays conflict.
        """
        return await invoke(workflow.revise, owner(write=True), project_id, request)

    @scoped_tool(server, annotations=WRITE)
    async def edit_customer_project(
        project_id: Identifier,
        instruction: Annotated[str, Field(min_length=1, max_length=10000)],
        revision_id: Identifier,
        input_version: Annotated[int, Field(ge=0, strict=True)],
    ) -> dict[str, Any]:
        """Queue a written edit against current CAD and input versions from progress.

        Pending approvals require revise_customer_proposal. Edits request new complete
        delivery. After an uncertain response, read progress before any retry.
        """
        return await invoke(
            workflow.edit, owner(write=True), project_id, instruction, revision_id, input_version
        )

    @scoped_tool(server, annotations=WRITE)
    async def resume_customer_project(project_id: Identifier) -> dict[str, Any]:
        """Resume saved customer work with its original finite budgets and approval guards."""
        return await invoke(workflow.resume, owner(write=True), project_id)

    @scoped_tool(server, annotations=READ)
    async def get_customer_files(project_id: Identifier) -> dict[str, Any]:
        """Read exact already captured current project/print packages and optional download links.

        Not-ready or stale results are explicit. No new export, geometry, acceptance or
        physical approval occurs. Existing immutable MCP resource readers serve bytes.
        """
        principal = owner()
        result = await invoke(workflow.files, principal, project_id)
        if downloads is not None and result["status"] == "captured_current":
            token = access_token_getter()
            if token is None or token.expires_at is None:
                raise ToolError("Current authorization is required for exact package links")
            for kind in ("project", "print"):
                package = result.get(kind)
                if package:
                    try:
                        links = await asyncio.to_thread(
                            downloads.mint,
                            owner=principal,
                            project_id=project_id,
                            package_id=package["package_id"],
                            package_hash=package["sha256"],
                            kind=kind,
                            individual_files=True,
                            authorization_expires_at=token.expires_at,
                        )
                    except (WorkspaceError, ValueError, OSError) as exc:
                        raise ToolError(str(exc)) from exc
                    package["downloads"] = links
        return bounded(result)
