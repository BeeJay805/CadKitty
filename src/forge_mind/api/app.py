"""FastAPI server exposing the agentic CAD engine to the web frontend.

    POST /generate            {prompt, quality?, fabrication_mode?} -> GenerateResponse (200; 402 if out of credits)
    GET  /files/{job}/{name}  serve a job's part.glb / part.stl (local, with durable-storage fallback)
    GET  /health              liveness
    /api/design-sessions/*    guided intake: clarifying questions, image + sketch upload

Production hardening is OPTIONAL + env-gated (see forge_mind.api.supa). With the Supabase env vars set:
  - /generate verifies the Supabase bearer token (401 if missing/invalid) and meters credits via the
    deduct_credits RPC (402 when out; refunded if the generation then fails).
  - intake sessions persist in the design_sessions table (survive restarts, multi-instance safe).
  - generated GLB/STL are mirrored to the storage bucket, so /files survives restarts / other instances.
With those vars UNSET the server runs open / in-memory / local-disk (dev mode) -- nothing breaks.

Run locally:
    uv run --extra api uvicorn forge_mind.api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from forge_mind.agentic.result import produce_part
from forge_mind.api import supa
from forge_mind.api.core import GLB_NAME, STL_NAME, generate_part
from forge_mind.api.schema import GenerateError, GenerateRequest
from forge_mind.api.session_store import SupabaseSessionStore
from forge_mind.intake.api import create_router
from forge_mind.intake.service import IntakeService
from forge_mind.intake.session import InMemorySessionStore, SessionStore

load_dotenv()  # make ANTHROPIC_API_KEY + SUPABASE_* available

# Per-job artifacts (gitignored, regenerable). One subdir per request id. Configurable for deploys.
JOB_ROOT = Path(os.environ.get("FORGE_JOB_ROOT", "benchmarks/results/_api"))
INTAKE_ASSET_ROOT = Path(os.environ.get("FORGE_INTAKE_ASSET_DIR", "benchmarks/results/_intake"))
_MEDIA = {".glb": "model/gltf-binary", ".stl": "model/stl"}
_SERVABLE = {GLB_NAME, STL_NAME}

# Credit cost per generation by quality (mirrors the frontend's credits model).
_CREDIT_COST = {"standard": 1, "studio": 5}

# CORS: comma-separated origins in FORGE_CORS_ORIGINS, or "*" for local dev (default). Tighten to the
# deployed frontend origin before any public deployment.
_ORIGINS = [o.strip() for o in os.environ.get("FORGE_CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Forge-Mind API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_auth(authorization: str | None = Header(default=None)) -> tuple[str, str] | None:
    """Auth dependency. Returns (user_id, token) when auth is enabled + the bearer token is valid;
    raises 401 when enabled + missing/invalid; returns None when auth is disabled (open dev mode)."""
    if not supa.auth_enabled():
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    user_id = supa.verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return (user_id, token)


def _agentic_generate(prompt: str, out_dir: Path) -> object:
    """Generation backend for the intake flow's own /api/design-sessions/{id}/generate route. The web
    frontend doesn't use that route (it hands the planner's resolved prompt to POST /generate), but
    wiring it to the agentic engine keeps intake self-consistent and free of the retired pipeline."""
    return produce_part(prompt, out_dir)


def _build_session_store() -> SessionStore:
    """Supabase-backed store when the service role is configured (durable, multi-instance), else the
    in-memory default (single-process dev)."""
    client = supa.service_client()
    if client is not None:
        return SupabaseSessionStore(client)
    return InMemorySessionStore()


# Mount the design-intake flow (/api/design-sessions/*): clarifying questions, image + sketch upload.
app.include_router(
    create_router(
        IntakeService(
            store=_build_session_store(),
            asset_dir=INTAKE_ASSET_ROOT,
            generate_fn=_agentic_generate,
        ),
        auth_dependency=_require_auth if supa.auth_enabled() else None,
    )
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate(
    req: GenerateRequest, auth: tuple[str, str] | None = Depends(_require_auth)
) -> JSONResponse:
    """Generate a part. Returns the contract shape (ok/refused/error) at HTTP 200, EXCEPT out-of-credits
    which is HTTP 402 (the frontend shows the upgrade modal on 402). Credits are metered upfront and
    refunded if the generation does not yield a part."""
    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    cost = _CREDIT_COST.get(req.quality, 1)

    charged = False
    if auth is not None:
        try:
            supa.charge_credit(auth[1], cost, job_id)
            charged = True
        except supa.InsufficientCreditsError:
            return JSONResponse(
                status_code=402,
                content={"status": "error", "stage": "credits", "message": "Out of credits."},
            )

    try:
        result = generate_part(
            req.prompt, job_dir, quality=req.quality, fabrication_mode=req.fabrication_mode
        )
    except Exception as exc:  # never leak a stack trace; refund the upfront charge
        if charged and auth is not None:
            supa.refund_credit(auth[0], cost)
        return JSONResponse(
            GenerateError(stage="server", message=f"{type(exc).__name__}: {exc}").model_dump()
        )

    if result.status == "ok" and result.ok is not None:
        # Mirror artifacts to durable storage so /files survives restarts / other instances.
        supa.upload_artifacts(job_id, {GLB_NAME: job_dir / GLB_NAME, STL_NAME: job_dir / STL_NAME})
        files = result.ok.files.model_copy(
            update={"glb": f"files/{job_id}/{GLB_NAME}", "stl": f"files/{job_id}/{STL_NAME}"}
        )
        return JSONResponse(result.ok.model_copy(update={"files": files}).model_dump())

    # refused / error: no part delivered, so refund the upfront charge.
    if charged and auth is not None:
        supa.refund_credit(auth[0], cost)
    payload = result.refused or result.error or GenerateError(message="unknown failure")
    return JSONResponse(payload.model_dump())


@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str) -> Response:
    """Serve a job's generated GLB / STL. Tries the local copy first, then durable storage (so a part
    generated by another instance / before a restart still serves). Guards path traversal + unknown
    files (404)."""
    if not job_id.isalnum() or filename not in _SERVABLE:
        raise HTTPException(status_code=404, detail="not found")
    path = (JOB_ROOT / job_id / filename).resolve()
    if JOB_ROOT.resolve() in path.parents and path.is_file():
        return FileResponse(path, media_type=_MEDIA[path.suffix], filename=filename)
    data = supa.download_artifact(job_id, filename)
    if data is None:
        raise HTTPException(status_code=404, detail="not found")
    return Response(content=data, media_type=_MEDIA[Path(filename).suffix])


def main() -> None:  # pragma: no cover - convenience runner
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["app"]
