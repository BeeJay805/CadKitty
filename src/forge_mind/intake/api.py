"""HTTP layer for the design-intake flow (FastAPI).

The frontend contract: the planner RETURNS question definitions and the UI renders by ``question.type``
(no part-specific question types are hardcoded client-side). Every endpoint is a thin wrapper over
``IntakeService``; the engine/validator are untouched.

Two ways to use it:
  - MOUNT into an existing app:  ``app.include_router(create_router(IntakeService()))`` -- this is how
    the deployed backend (``forge_mind.api.app``) serves ``/api/design-sessions/*`` alongside
    ``/generate``.
  - RUN standalone:  ``uv run --extra api uvicorn forge_mind.intake.api:app --reload``.

``fastapi`` is an optional extra (``pip install 'forge-mind[api]'``). Importing this module is safe
without it -- the import happens inside ``create_router`` / ``create_app`` so the rest of
``forge_mind.intake`` stays dependency-light.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.intake.brief import brief_to_prompt
from forge_mind.intake.schema import Answer, AssetType, CaptureContext, ScaleRef, Sketch, Viewpoint
from forge_mind.intake.service import (
    GenerationBlockedError,
    IntakeService,
    PlannerNotReadyError,
)
from forge_mind.intake.session import DesignSession, SessionNotFoundError

_PREFIX = "/api/design-sessions"

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".glb": "model/gltf-binary",
    ".stl": "model/stl",
    ".json": "application/json",
}

# Cap base64 asset uploads at the same 25 MB the STL/STEP upload path enforces
# (forge_mind.server.app._MAX_UPLOAD_BYTES). The intake `assets` endpoint is open (only `generate`
# is auth-gated), and the field previously had no ceiling, so an unauthenticated client could POST an
# unbounded body and exhaust disk/RAM. base64 inflates ~4/3, so cap the encoded string accordingly.
_MAX_ASSET_BYTES = 25 * 1024 * 1024
_MAX_ASSET_B64_CHARS = ((_MAX_ASSET_BYTES + 2) // 3) * 4


class InputBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str | None = None
    sketch: Sketch | None = None
    # Multimodal capture bundle (ADR-0025), flat to mirror the contract. All optional + additive.
    speech_transcript: str | None = None
    notes: str | None = None
    labels: list[str] = Field(default_factory=list)
    scale: ScaleRef | None = None
    viewpoint: Viewpoint | None = None
    # The 3D-printable toggle (ADR-0031), finally reaching the planner (ADR-0036): True ->
    # print_3d, False -> general, None (legacy clients) -> today's FDM-default behavior.
    printable: bool | None = None


class AssetBody(BaseModel):
    """Base64 file upload (keeps the API dependency-light -- no multipart needed)."""

    model_config = ConfigDict(extra="forbid")
    filename: str = Field(..., min_length=1)
    content_type: str = "application/octet-stream"
    data_base64: str = Field(..., min_length=1, max_length=_MAX_ASSET_B64_CHARS)
    asset_type: AssetType = AssetType.REFERENCE_IMAGE


class SketchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sketch: Sketch


class AnswerBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answers: list[Answer] = Field(default_factory=list)


def _outcome_payload(outcome: Any) -> dict[str, Any]:
    ready_brief = outcome.brief if (outcome.ready and outcome.brief is not None) else None
    return {
        "ready": outcome.ready,
        "refusal": outcome.refusal,
        "note": outcome.note,
        "questions": [q.model_dump(mode="json") for q in outcome.questions],
        # Structure-first interpretation (ADR-0025): the feature list + intentional constraints the
        # UI renders as confirmation chips before dimensions are pinned.
        "features": [f.model_dump(mode="json") for f in outcome.features],
        "constraints": [c.model_dump(mode="json") for c in outcome.constraints],
        "brief": outcome.brief.model_dump(mode="json") if outcome.brief is not None else None,
        # The standards-resolved prompt for a READY brief, so a client can hand it straight to the
        # existing /generate (no duplicate generation, no TS-side prompt synthesis that could drift).
        "prompt": brief_to_prompt(ready_brief) if ready_brief is not None else None,
        "planner_state": outcome.state.model_dump(mode="json"),
    }


def _capture_from_body(body: InputBody) -> CaptureContext | None:
    """Assemble the capture bundle from the flat input body; None if nothing multimodal was sent (so a
    plain text/sketch request leaves the session's capture untouched)."""
    if not (body.speech_transcript or body.notes or body.labels or body.scale or body.viewpoint):
        return None
    return CaptureContext(
        speech_transcript=body.speech_transcript,
        notes=body.notes,
        labels=body.labels,
        scale=body.scale,
        viewpoint=body.viewpoint,
    )


def create_router(service: IntakeService, *, auth_dependency: Any = None) -> Any:
    """Build an ``APIRouter`` with the design-session endpoints, to ``include_router`` into any
    FastAPI app (no ``/health`` here -- that is an app-level concern).

    ``auth_dependency`` (a FastAPI dependency callable), when provided, gates the COSTLY ``generate``
    endpoint only -- it runs the paid pipeline, so generation requires it while planning stays open."""
    from fastapi import APIRouter, Depends, HTTPException
    from fastapi.responses import FileResponse

    svc = service
    router = APIRouter()
    _gen_deps = [Depends(auth_dependency)] if auth_dependency is not None else []

    def _session_or_404(session_id: str) -> DesignSession:
        try:
            return svc.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="design session not found") from exc

    def _files_base(session_id: str) -> str:
        return f"{_PREFIX}/{session_id}/files"

    @router.post(_PREFIX)
    def create() -> dict[str, Any]:
        return svc.create_session().model_dump(mode="json")

    @router.get(_PREFIX + "/{session_id}")
    def get(session_id: str) -> dict[str, Any]:
        return _session_or_404(session_id).model_dump(mode="json")

    @router.post(_PREFIX + "/{session_id}/input")
    def add_input(session_id: str, body: InputBody) -> dict[str, Any]:
        _session_or_404(session_id)
        session = svc.add_input(
            session_id,
            text=body.text,
            sketch=body.sketch,
            capture=_capture_from_body(body),
            base_url=_files_base(session_id),
            printable=body.printable,
        )
        return session.model_dump(mode="json")

    @router.post(_PREFIX + "/{session_id}/assets")
    def add_asset(session_id: str, body: AssetBody) -> dict[str, Any]:
        _session_or_404(session_id)
        try:
            data = base64.b64decode(body.data_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=400, detail="data_base64 is not valid base64") from exc
        asset = svc.add_asset(
            session_id,
            filename=body.filename,
            data=data,
            asset_type=body.asset_type,
            content_type=body.content_type,
            base_url=_files_base(session_id),
        )
        return asset.model_dump(mode="json")

    @router.post(_PREFIX + "/{session_id}/sketch")
    def add_sketch(session_id: str, body: SketchBody) -> dict[str, Any]:
        _session_or_404(session_id)
        sketch_asset = svc.add_sketch(
            session_id, sketch=body.sketch, base_url=_files_base(session_id)
        )
        return sketch_asset.model_dump(mode="json")

    @router.get(_PREFIX + "/{session_id}/files/{name}")
    def get_file(session_id: str, name: str) -> Any:
        """Serve a stored upload / sketch preview / SVG for this session (path-traversal guarded)."""
        _session_or_404(session_id)
        path = svc.asset_file_path(session_id, name)
        if path is None:
            raise HTTPException(status_code=404, detail="file not found")
        media = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media, filename=path.name)

    @router.post(_PREFIX + "/{session_id}/planner/run")
    def planner_run(session_id: str) -> dict[str, Any]:
        _session_or_404(session_id)
        return _outcome_payload(svc.run_planner(session_id))

    @router.post(_PREFIX + "/{session_id}/questions/answer")
    def answer(session_id: str, body: AnswerBody) -> dict[str, Any]:
        _session_or_404(session_id)
        return svc.answer_questions(session_id, body.answers).model_dump(mode="json")

    @router.post(_PREFIX + "/{session_id}/planner/finalize")
    def finalize(session_id: str) -> dict[str, Any]:
        _session_or_404(session_id)
        try:
            return svc.finalize(session_id).model_dump(mode="json")
        except PlannerNotReadyError as exc:
            raise HTTPException(
                status_code=409, detail={"error": "not_ready", "reasons": exc.reasons}
            ) from exc

    @router.post(_PREFIX + "/{session_id}/generate", dependencies=_gen_deps)
    def generate(session_id: str) -> dict[str, Any]:
        session = _session_or_404(session_id)
        try:
            svc.start_generation(session_id)
        except GenerationBlockedError as exc:
            raise HTTPException(
                status_code=409, detail={"error": "blocked", "reasons": exc.reasons}
            ) from exc
        fresh = svc.get_session(session_id)
        return {
            "linked_generation_id": fresh.linked_generation_id,
            "status": fresh.status.value,
            "prompt": svc.generation_prompt(session_id) if session.final_design_brief else None,
        }

    return router


def create_app(service: IntakeService | None = None) -> Any:
    """Build a standalone FastAPI app for the intake flow. Raises a clear error if the optional
    ``fastapi`` extra is missing."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The design-intake API needs FastAPI. Install it with: pip install 'forge-mind[api]'"
        ) from exc

    svc = service or IntakeService()
    app = FastAPI(title="Forge-Mind Design Intake", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(create_router(svc))
    return app


def _make_default_app() -> Any:
    """Module-level ``app`` for uvicorn. Returns None if FastAPI isn't installed (import stays safe)."""
    try:
        return create_app()
    except ImportError:  # pragma: no cover
        return None


app = _make_default_app()

__all__ = ["create_app", "create_router"]
