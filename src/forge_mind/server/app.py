"""FastAPI server: POST /generate (prompt -> validated part), GET /files/<job>/<name>, GET /health.

Maps a ``PipelineResult`` to the frontend ``GenerateResponse`` (web/src/lib/schema.ts). Generation
costs real Anthropic money, so /generate requires a valid Supabase session (verified against the
Supabase Auth API). Files are written per-job under an ephemeral work dir and served back so the
browser can fetch the GLB/STL and upload them to its own storage.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import trimesh
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from forge_mind.api.manufacture_stamp import manufacture_stamp
from forge_mind.config import HAIKU, LAYOUTSPEC_MODEL, OPUS
from forge_mind.intake.agent import HeuristicPlannerBrain, LlmPlannerBrain, PlannerAgent
from forge_mind.intake.api import create_router as _intake_router
from forge_mind.intake.service import IntakeService
from forge_mind.intake.session import JSONFileSessionStore
from forge_mind.invention import invention_enabled
from forge_mind.invention.dashboard import build_dashboard
from forge_mind.invention.lifecycle import GeneratedPart
from forge_mind.invention.mission import MissionLog
from forge_mind.invention.support import KBSnippet, SupportCopilotKB, answer_question
from forge_mind.learning import GenerationRecord, TokenUsage
from forge_mind.learning.mission_records import log_from_rows
from forge_mind.physics.fea import DEFAULT_E_MPA, estimate_load_response, estimate_stress
from forge_mind.physics.report import Intent, compose_physics_report, physics_report_enabled
from forge_mind.physics.report.holes import DetectedHole
from forge_mind.physics.report.motion_project import project_motion
from forge_mind.physics.stability import compute_stability
from forge_mind.server.billing import create_billing_router
from forge_mind.server.lab import create_lab_router
from forge_mind.types import Vec3
from forge_mind.validator.intrinsic import validate_intrinsic
from forge_mind.validator.report import CheckStatus, ValidationReport
from forge_mind.viz.glb import to_gltf_yup
from forge_mind.viz.motion_glb import export_motion_glb

# --- config (env) ----------------------------------------------------------------------------
WORK_DIR = Path(os.environ.get("FORGE_WORK_DIR", "/tmp/forge-jobs"))
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
# Durable artifact storage (Supabase Storage). OFF unless a bucket is named: when off, artifacts are
# served from the ephemeral /files path (lost on redeploy). When on, GLB/STL are uploaded under
# <user_id>/<job>/ and returned as short-lived signed URLs (durable + owner-scoped, not the public
# /files path). VERIFY against live Supabase before relying on it.
STORAGE_BUCKET = os.environ.get("FORGE_STORAGE_BUCKET", "").strip()
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "https://cadkitty.com,https://www.cadkitty.com,http://localhost:3000"
    ).split(",")
    if o.strip()
]

WORK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CadKitty backend", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class GenerateBody(BaseModel):
    prompt: str
    quality: str = "standard"  # "standard" (1 credit) | "studio" (5 credits)
    # 3D-printable toggle (ADR-0031). False (default) -> GENERAL: validate as a faithful CAD model,
    # FDM overhang/min-wall checks OFF, author models the real object. True -> PRINT_3D: run the
    # printability tier + steer the author to design for FDM printing.
    printable: bool = False
    # The guided-intake session that produced this prompt (ADR-0036, additive). When set and the
    # session holds a READY typed brief, its STATED-FACTS block is injected into the agentic brief
    # research so user-confirmed facts bind generation instead of being re-guessed from prose.
    # Unknown/absent/brief-less ids degrade silently to today's prompt-only behavior.
    design_session_id: str | None = None


class PatchBody(BaseModel):
    """One parametric edit to an existing standard-lane part (ADR-0037). STATELESS: the
    client sends back the ``spec`` it received on the /generate (or previous /patch)
    response, plus the edit in plain English. The backend changes ONE spec field,
    re-derives through the deterministic interpreter + strict gates, and returns the
    same response contract with the NEW spec for the next edit."""

    spec: dict[str, Any]
    edit: str
    quality: str = "standard"  # billing tier (same gate as /generate)


class FieldsBody(BaseModel):
    """The stored LayoutSpec to enumerate editable fields for, so the UI can render the
    structured blueprint's direct controls (sliders / dropdowns / toggles)."""

    spec: dict[str, Any]


class DraftBody(BaseModel):
    """The blueprint pregen request: a resolved prompt (usually the planner's) to draft a quick,
    mutable preview from BEFORE the full validated build. No credit is charged."""

    prompt: str


class JobRebuildBody(BaseModel):
    """PARAMS overrides for a job's stored script (ADR-0073 contract): float dimensions only —
    int params are frozen structure, unknown names are rejected (fail-closed, never silently
    ignored)."""

    overrides: dict[str, float]


class PatchFieldBody(BaseModel):
    """One STRUCTURED parametric edit (a direct UI control change) to a standard-lane part: the
    catalog ``field_path``, the ``op``, and the ``value`` the user dialed. Deterministic -- no LLM
    and no credit (the free, instant slider/dropdown path; prose edits go to /patch)."""

    spec: dict[str, Any]
    field_path: str
    op: Literal["set", "increase", "decrease"] = "set"
    value: float | bool | str


class PatchStructureBody(BaseModel):
    """One STRUCTURAL edit (ADR-0060 slice C): ADD or REMOVE an interface/feature on a composable-
    plate part. Deterministic -- no LLM, no credit (the add/remove-feature path of the editable
    blueprint). ADD needs ``ref`` + ``face`` (+ optional placement fields); REMOVE needs ``index``."""

    spec: dict[str, Any]
    op: Literal["add", "remove"]
    ref: str | None = None
    face: str | None = None
    offset_mm: float | None = None
    spacing_mm: float | None = None
    component_id: str | None = None
    fit: str | None = None
    name: str | None = None
    index: int | None = None


# Credit cost per generation by quality tier (standard = cheaper loop, studio = best quality).
CREDIT_COST = {"standard": 1, "studio": 5}

# Cap one in-flight generation per user (per worker process) so a single user can't fire many
# parallel expensive jobs before any credit deduction lands (the check-then-act race). In-memory
# per worker: with N uvicorn workers a user can reach N concurrent; Fly per-machine concurrency
# limits bound the global total. Redis would make this exact across workers/machines.
_GEN_INFLIGHT_LOCK = threading.Lock()
_GEN_INFLIGHT: set[str] = set()


def _acquire_inflight(user_id: str) -> None:
    with _GEN_INFLIGHT_LOCK:
        if user_id in _GEN_INFLIGHT:
            raise HTTPException(
                status_code=429, detail="a generation is already running; wait for it to finish"
            )
        _GEN_INFLIGHT.add(user_id)


def _release_inflight(user_id: str) -> None:
    with _GEN_INFLIGHT_LOCK:
        _GEN_INFLIGHT.discard(user_id)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


def _require_user(authorization: str | None) -> tuple[str, str]:
    """Reject the request unless the caller presents a valid Supabase access token.

    Returns ``(token, user_id)``. Verification is delegated to Supabase's /auth/v1/user endpoint so
    it works regardless of the project's JWT signing scheme (HS256 secret or asymmetric keys) -- no
    local key handling.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="auth not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="sign in to generate")
    token = authorization.split(" ", 1)[1].strip()
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY},
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="could not verify session") from exc
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="invalid or expired session")
    try:
        user_id = str(resp.json()["id"])
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=401, detail="invalid session") from exc
    return token, user_id


class NarrateBody(BaseModel):
    prompt: str


class _Narration(BaseModel):
    """Per-phase narration of the build, grounded in the request. The model phrases each REAL
    pipeline phase for this part; it never claims specific operations or dimensions the request did
    not imply."""

    model_config = ConfigDict(extra="forbid")

    understanding: str = Field(..., description="Present-tense line: understanding the request.")
    planning: str = Field(..., description="Present-tense line: planning the part + standards.")
    building: str = Field(..., description="Present-tense line: building the geometry.")
    checking: str = Field(..., description="Present-tense line: checking fit + printability.")
    preparing: str = Field(..., description="Present-tense line: preparing the model.")


_NARRATE_SYSTEM = (
    "You narrate a CAD generation pipeline for a 3D-printable part. Given the user's request, write "
    "ONE short present-tense line (max ~8 words) per phase describing what is happening for THIS "
    "part, grounded ONLY in the request. Do not invent specific dimensions, counts, or features the "
    "user did not state. Stay concrete and calm: no marketing, no emojis."
)


@app.post("/narrate")
def narrate(
    body: NarrateBody, authorization: str | None = Header(default=None)
) -> dict[str, list[dict[str, str]]]:
    """Best-effort per-phase build narration via one cheap Haiku call. Requires a session (same as
    /generate) but spends NO credits. On any failure returns an empty list, so the frontend falls
    back to its built-in phase labels — we never fabricate narration when the model did not run."""
    _require_user(authorization)
    prompt = (body.prompt or "").strip()
    if not prompt:
        return {"lines": []}
    try:
        from forge_mind.llm.provider import make_client

        client = make_client()
        narration, _ = client.generate_structured(
            system=_NARRATE_SYSTEM,
            user=prompt,
            output_model=_Narration,
            model=HAIKU,
            max_tokens=400,
        )
    except Exception:
        # Best-effort UX: any failure (no key, network, API, parse) degrades to no narration
        # rather than failing the request. The frontend then falls back to its built-in labels.
        return {"lines": []}
    lines = [
        {"key": "intent", "text": narration.understanding},
        {"key": "plan", "text": narration.planning},
        {"key": "cad", "text": narration.building},
        {"key": "validate", "text": narration.checking},
        {"key": "export", "text": narration.preparing},
    ]
    return {"lines": lines}


# --- Invention support copilot (ADR-0050) ----------------------------------------------------
# Read-only Q&A over a product's grounded support KB (invention/support.py). Flag-gated behind
# FORGE_INVENTION (404 when off, so the deployed contract is byte-identical by default) and session-
# gated like /narrate (no credits). The honesty rules live in answer_question and run BEFORE any model
# call: a load/safety question is REFUSED, an ungrounded question is "unknown" (never fabricated), and
# the injected model phrases ONLY from the retrieved grounding snippets. With no model wired (or on any
# model failure) it answers deterministically from the snippets -- never a fabricated fitness claim.

_SUPPORT_SYSTEM = (
    "You are a product support copilot for a 3D-printed part. Answer the user's question using ONLY "
    "the provided grounding snippets. Do NOT add any fact, dimension, or claim that is not in the "
    "grounding. Never give a load, safety, weight, or strength rating. If the grounding does not "
    "answer the question, say you do not have that information. Be concise, plain, and calm."
)


class _SupportPhrasing(BaseModel):
    """The model's phrasing of a grounded answer -- the ONE free-text field, constrained to the
    retrieved snippets by the system prompt. Closed model so nothing else can leak in."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., description="The answer, phrased ONLY from the grounding snippets.")


class _LlmKBAnswerer:
    """A KBAnswerer that phrases a grounded answer via one cheap Haiku call. Best-effort INTERNALLY:
    on any model failure it falls back to the deterministic snippet join, so it never raises and the
    endpoint always returns a valid answer. The refuse/unknown verdicts never reach here -- they are
    decided in answer_question before the model is consulted."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _deterministic(snippets: list[KBSnippet]) -> str:
        return " | ".join(f"[{s.section}] {s.text}" for s in snippets)

    def answer(self, query: str, snippets: list[KBSnippet]) -> str:
        grounding = "\n".join(f"[{s.section}] {s.text}" for s in snippets)
        try:
            phrasing, _ = self._client.generate_structured(
                system=_SUPPORT_SYSTEM,
                user=f"Question: {query}\n\nGrounding (answer ONLY from this):\n{grounding}",
                output_model=_SupportPhrasing,
                model=HAIKU,
                max_tokens=400,
            )
        except Exception:
            return self._deterministic(snippets)
        answer: str = phrasing.answer
        return answer or self._deterministic(snippets)


def _make_kb_answerer() -> _LlmKBAnswerer | None:
    """Build the live answerer best-effort: None (deterministic grounding) when no model client can be
    constructed (no key / spend guard), exactly like /narrate degrades. Never raises."""
    try:
        from forge_mind.llm.provider import make_client

        return _LlmKBAnswerer(make_client())
    except Exception:
        return None


class SupportBody(BaseModel):
    """A read-only support question against an already-assembled, typed support KB. STATELESS like
    /patch: the client sends back the grounded KB (a closed model -- no free text can enter it) plus
    the question. (Rebuilding the KB server-side from a persisted mission_id is a v1 convenience.)"""

    kb: SupportCopilotKB
    question: str


@app.post("/invention/support")
def invention_support(
    body: SupportBody, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Answer a product-support question, GROUNDED and read-only (ADR-0050). Flag-gated behind
    FORGE_INVENTION (404 when off) and session-gated like /narrate (spends no credits). Returns the
    SupportAnswer contract: status answered|unknown|refused, the answer (or null), and citations."""
    if not invention_enabled():
        raise HTTPException(status_code=404, detail="not found")
    _require_user(authorization)
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")
    answer = answer_question(body.kb, question, answerer=_make_kb_answerer())
    return answer.model_dump(mode="json")


# --- Invention lab dashboard (ADR-0048 / ADR-0052) -------------------------------------------
# The read-model behind the frontend /app/lab, served for a PERSISTED mission. Flag-gated behind
# FORGE_INVENTION (404 when off, so the deployed /generate contract is byte-identical by default) and
# session-gated like /invention/support (no credits). The deployed server has no supabase-py client,
# so the mission is read over the raw-httpx REST seam (the same pattern as the entitlement/generation
# calls) and folded with log_from_rows -- NOT via the supabase-py-bound MissionEventStore, whose import
# would fail in the deploy image. Read-only: nothing here gates or generates.


def _load_mission_log(mission_id: str) -> MissionLog | None:
    """Fetch one mission's append-only events from the service-role ``mission_events`` table (ADR-0050)
    and fold them into a :class:`MissionLog`, or ``None`` when storage is unconfigured / the mission has
    no rows. Service-role read (the table's RLS denies anon/authenticated); mirrors ``_store_generation``
    rather than constructing a supabase-py client the deployed server does not carry."""
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        return None
    try:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/mission_events",
            params={"select": "*", "mission_id": f"eq.{mission_id}", "order": "seq"},
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
            },
            timeout=10.0,
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    rows: list[dict[str, Any]] = r.json()
    if not rows:
        return None
    return log_from_rows(rows)


@app.get("/invention/dashboard/{mission_id}")
def invention_dashboard(
    mission_id: str, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """The lab dashboard read-model for a persisted mission (ADR-0048/0052). Flag-gated behind
    FORGE_INVENTION (404 when off) and session-gated like /invention/support (spends no credits).
    Returns the MissionDashboard contract (web/src/lib/dashboard.ts), folded deterministically from the
    mission timeline; 404 when the mission has no persisted events. The package-derived maturity /
    evidence_tier / banner are absent until a run also persists its InventionPackage (the timeline
    alone does not carry it) -- everything that folds from the timeline is present, including the
    always-on NOT-PHYSICALLY-VALIDATED risk."""
    if not invention_enabled():
        raise HTTPException(status_code=404, detail="not found")
    _require_user(authorization)
    log = _load_mission_log(mission_id)
    if log is None:
        raise HTTPException(status_code=404, detail="mission not found")
    dashboard = build_dashboard(log.state(), log.events())
    return dashboard.model_dump(mode="json")


def _credit_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "apikey": SUPABASE_ANON_KEY,
        "content-type": "application/json",
    }


def _get_entitlement(token: str) -> dict[str, Any] | None:
    """The signed-in user's resolved entitlement via the `entitlements` view under RLS -- the SAME
    row the web app renders, so what we ENFORCE here is exactly what the user is SHOWN (plan, credit
    balance, per-quality cost, allowed qualities, and the comp flags). None only on a genuine read
    failure (fail closed); an unprovisioned account resolves to a Free/0-credit default so it yields a
    clean upgrade prompt rather than a 503."""
    try:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/entitlements",
            params={"select": "*"},
            headers=_credit_headers(token),
            timeout=10.0,
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    rows: list[dict[str, Any]] = r.json()
    if rows:
        return rows[0]
    return {
        "plan_id": "free",
        "is_unlimited": False,
        "is_complimentary": False,
        "credits_balance": 0,
        "allowed_qualities": ["standard"],
        "per_quality_cost": {},
        "upgrade_to": "pro",
    }


def _paywall(
    code: str, ent: dict[str, Any], quality: str, cost: int | None = None
) -> dict[str, Any]:
    """A machine-readable 402 body so the frontend renders exactly ONE correct paywall (no guessing):
    the reason code, the user's current plan, the EXACT plan that unblocks the action, the credits
    needed, and a one-line human reason it can show verbatim."""
    reason = (
        f"{quality.capitalize()} quality isn't included in your plan."
        if code == "quality_not_allowed"
        else "You're out of design credits."
    )
    return {
        "error_code": code,
        "current_plan_id": ent.get("plan_id") or "free",
        "upgrade_to_plan_id": ent.get("upgrade_to"),
        "credits_required": cost,
        "reason_human": reason,
    }


def _resolve_credit_gate(token: str, quality: str) -> tuple[bool, int]:
    """Resolve the entitlement and apply the quality + credit gate BEFORE spending the API call.
    Returns (skip_charge, cost): skip_charge=True for a complimentary/unlimited account (metering is
    bypassed). Raises 503 if the entitlement can't be read (fail closed), or a structured 402 when the
    requested quality isn't in the plan or the balance is too low. The atomic deduct_credits RPC stays
    the final gate on a metered account. Cost comes from the entitlement's per-quality table (the
    single source of truth); CREDIT_COST is only a defensive fallback if that table is unreadable."""
    ent = _get_entitlement(token)
    if ent is None:
        raise HTTPException(status_code=503, detail="could not verify credits; try again")
    allowed = ent.get("allowed_qualities") or ["standard"]
    if quality not in allowed:
        raise HTTPException(status_code=402, detail=_paywall("quality_not_allowed", ent, quality))
    per_quality = ent.get("per_quality_cost") or {}
    cost = int(per_quality.get(quality, CREDIT_COST.get(quality, CREDIT_COST["standard"])))
    if ent.get("is_unlimited"):
        return True, cost  # complimentary / unlimited: build without metering
    balance = int(ent.get("credits_balance") or 0)
    if balance < cost:
        raise HTTPException(
            status_code=402, detail=_paywall("insufficient_credits", ent, quality, cost)
        )
    return False, cost


def _deduct_credits(token: str, amount: int, reason: str, generation_id: str) -> int | None:
    """Atomically spend `amount` credits (the DB function enforces the balance). Returns the new
    balance, or None on failure (insufficient / network). The work is already done by call time, so
    a failure here is logged via the ledger gap, never a reason to withhold a built part."""
    try:
        r = httpx.post(
            f"{SUPABASE_URL}/rest/v1/rpc/deduct_credits",
            headers=_credit_headers(token),
            json={"p_amount": amount, "p_reason": reason, "p_generation_id": generation_id},
            timeout=10.0,
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        return int(r.json())
    except (ValueError, TypeError):
        return None


def _store_generation(token: str, record: GenerationRecord) -> None:
    """Best-effort: persist one generations row as the user (RLS) so the learning loop accumulates
    EVERY attempt -- success, refusal, and failure alike (LEARNING_LOOP.md). Never raises and never
    blocks the response; a recording miss is invisible to the user (mirrors the credit deduct)."""
    if not SUPABASE_URL or not token:
        return
    try:
        httpx.post(
            f"{SUPABASE_URL}/rest/v1/generations",
            headers={**_credit_headers(token), "Prefer": "return=minimal"},
            json=record.to_row(),
            timeout=10.0,
        )
    except httpx.HTTPError:
        return


def _validation_view(report: ValidationReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "overall": report.overall.value,
        "tiers": [
            {"tier": tr.tier.value, "status": tr.status.value} for tr in report.tiers.values()
        ],
        "failed_checks": [
            {"id": c.id, "status": c.status.value, "message": c.message}
            for c in report.checks
            # INFO = advisory measured evidence (ADR-0035 B1), kept out of the failed list.
            if c.status not in (CheckStatus.PASS, CheckStatus.INFO)
        ],
    }


# Material densities (g/cm^3) for a SOLID mass estimate. Substring match so "PLA or PETG" resolves.
_DENSITY_G_CM3 = {
    "petg": 1.27,
    "pla": 1.24,
    "asa": 1.07,
    "abs": 1.04,
    "nylon": 1.14,
    "pa12": 1.01,
    "tpu": 1.21,
    "resin": 1.18,
    "aluminum": 2.70,
    "aluminium": 2.70,
    "stainless": 7.90,
    "steel": 7.85,
    "titanium": 4.43,
    "brass": 8.50,
}


def _density_for(material: str | None) -> float | None:
    if not material:
        return None
    m = material.lower()
    for key, d in _DENSITY_G_CM3.items():
        if key in m:
            return d
    return None


def _stability_view(mesh: Any) -> dict[str, Any] | None:
    """Stand-or-tip for a Z-up mesh (the STL frame -- NOT the Y-up GLB: ``compute_stability`` rests on
    the lowest +Z face). ``None`` when the part has no clear flat base -- refuse, don't guess.

    Coordinates stay Z-up here (``com_mm``, ``footprint_mm`` at ``base_z_mm``); the viewer converts to
    its Y-up frame. MEASURED geometry, never a guarantee against a bump (see ``physics/stability.py``)."""
    try:
        result = compute_stability(mesh)
    except Exception:
        return None
    if result is None:
        return None
    return {
        "stable": result.stable,
        "margin_mm": result.margin_mm,
        "tip_angle_deg": result.tip_angle_deg,
        "com_mm": {"x": result.com[0], "y": result.com[1], "z": result.com[2]},
        "com_height_mm": result.com_height_mm,
        "base_z_mm": round(result.com[2] - result.com_height_mm, 2),
        "footprint_mm": [[x, y] for x, y in result.footprint],
    }


def _mesh_metrics(mesh: Any, material: str | None, stl_mesh: Any = None) -> dict[str, Any] | None:
    """Geometric facts computed straight from the exported mesh: bounding box, volume, surface area,
    center of mass, and a SOLID (100%-infill) mass estimate from the material density. Best-effort;
    a field is null rather than guessed. ``stl_mesh`` (the Z-up STL) feeds the stand-or-tip view --
    omitted/``None`` leaves ``stability`` null."""
    try:
        ext = mesh.extents
        bbox = {"x": float(ext[0]), "y": float(ext[1]), "z": float(ext[2])}
    except Exception:
        return None
    try:
        volume_mm3: float | None = abs(float(mesh.volume))
    except Exception:
        volume_mm3 = None
    try:
        area_mm2: float | None = float(mesh.area)
    except Exception:
        area_mm2 = None
    try:
        com = mesh.center_mass
        center_of_mass: dict[str, float] | None = {
            "x": float(com[0]),
            "y": float(com[1]),
            "z": float(com[2]),
        }
    except Exception:
        center_of_mass = None
    try:
        watertight = bool(mesh.is_watertight)
    except Exception:
        watertight = False
    density = _density_for(material)
    mass_g = (
        (volume_mm3 / 1000.0) * density
        if (volume_mm3 is not None and density is not None)
        else None
    )
    return {
        "bbox_mm": bbox,
        "volume_mm3": volume_mm3,
        "surface_area_mm2": area_mm2,
        "center_of_mass_mm": center_of_mass,
        "watertight": watertight,
        "mass_g": mass_g,
        "mass_basis_material": material if mass_g is not None else None,
        "stability": _stability_view(stl_mesh) if stl_mesh is not None else None,
    }


def _storage_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and STORAGE_BUCKET)


def _upload_artifact(path: Path, key: str, mime: str) -> str | None:
    """Upload one artifact to Supabase Storage (service role) and return a 1h signed URL, or None.

    Best-effort + flag-gated: returns None (caller keeps the /files fallback) when storage is off,
    the file is missing, or any call fails. Never raises -- a storage hiccup must not fail a build
    that already succeeded.
    """
    if not _storage_enabled() or not path.is_file():
        return None
    base = f"{SUPABASE_URL}/storage/v1"
    auth = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
    }
    try:
        up = httpx.post(
            f"{base}/object/{STORAGE_BUCKET}/{key}",
            headers={**auth, "content-type": mime, "x-upsert": "true"},
            content=path.read_bytes(),
            timeout=30.0,
        )
        if up.status_code not in (200, 201):
            return None
        signed = httpx.post(
            f"{base}/object/sign/{STORAGE_BUCKET}/{key}",
            headers={**auth, "content-type": "application/json"},
            json={"expiresIn": 3600},
            timeout=10.0,
        )
        if signed.status_code != 200:
            return None
        rel = signed.json().get("signedURL")
        return f"{base}{rel}" if isinstance(rel, str) and rel else None
    except (httpx.HTTPError, ValueError, OSError):
        return None


def _apply_manufacture_stamp(payload: dict[str, Any]) -> None:
    """Stamp the payload in place. The decision lives in api.manufacture_stamp so it is
    testable without the fastapi import this module needs (tests/unit/server cannot collect in the
    local venv, so the logic would otherwise ship verified only by reading)."""
    not_for_manufacture, warning = manufacture_stamp(payload)
    payload["not_for_manufacture"] = not_for_manufacture
    payload["manufacture_warning"] = warning


def _lane_spec(ag: Any) -> dict[str, Any] | None:
    """The standard-lane attempt telemetry (ADR-0035) as a ``generations.spec`` fragment, or None
    when the prompt never matched the lane. Additive: rides the existing jsonb column, no schema
    change."""
    lane = getattr(ag, "lane", None)
    return {"standard_lane": lane.to_record()} if lane is not None else None


def _engine_for(gen_path: str) -> str:
    """Honest ``generations.engine`` value from the result's provenance (ADR-0035): the column's
    documented domain is the engine that BUILT the part, and the standard/implicit lanes are not
    the agentic author."""
    if gen_path.startswith("standard:"):
        return "layoutspec"
    if gen_path.startswith("implicit:"):
        return "implicit"
    return "agentic"


def _brief_context_for(design_session_id: str | None) -> str | None:
    """The typed STATED-FACTS block for a generate request that came from a guided-intake session
    (ADR-0036). Best-effort and silent: an unknown id, a session without a ready brief, or any
    store hiccup degrades to None -- prompt-only behavior, never a failed generation."""
    if not design_session_id:
        return None
    try:
        from forge_mind.intake.brief import brief_to_context

        session = _intake_service.store.get(design_session_id)
        if session is None or session.final_design_brief is None:
            return None
        return brief_to_context(session.final_design_brief)
    except Exception:
        return None


def _run_agentic(
    prompt: str,
    quality: str,
    token: str | None,
    cost: int,
    user_id: str,
    charge: bool = True,
    on_phase: Callable[[str, str], None] | None = None,
    *,
    printable: bool = False,
    brief_context: str | None = None,
    store: bool = True,
) -> dict[str, Any]:
    """Generate via the AGENTIC engine -- the main path. It authors ARBITRARY parts (a hammer, a
    gear train, a bracket) from plain English, not just the bounded legacy LayoutSpec patterns. The
    PartResult is projected into the exact GenerateResponse the frontend expects (via api.core), the
    GLB/STL are served from the job dir, and only a real part is charged. A refusal/error returns the
    contract shape WITHOUT a deduction.

    ``store=False`` (the guest taste-build, /generate-demo): no session token exists, so nothing is
    written to the generations record and no artifact is uploaded to owner-scoped storage -- the
    part is served from the ephemeral /files job dir only."""
    from forge_mind.api.core import generate_part as _agentic_generate

    job = uuid.uuid4().hex
    out = WORK_DIR / job
    out.mkdir(parents=True, exist_ok=True)
    from forge_mind.server.job_ownership import claim

    claim(WORK_DIR, job, user_id)
    ag = _agentic_generate(
        prompt,
        out,
        quality=quality,
        fabrication_mode="print_3d" if printable else "general",
        on_phase=on_phase,
        brief_context=brief_context,
    )

    if ag.status == "refused" and ag.refused is not None:
        if store and token is not None:
            _store_generation(
                token,
                GenerationRecord(
                    user_id=user_id,
                    prompt=prompt,
                    engine="agentic",
                    status="rejected",
                    quality=quality,
                    error_type="refused",
                    error_message=ag.refused.reason,
                    spec=_lane_spec(ag),
                ),
            )
        return {"status": "refused", "reason": ag.refused.reason}
    if ag.status != "ok" or ag.ok is None:
        err = ag.error
        message = (
            err.message if err is not None else "Generation failed before a part was produced."
        )
        stage = err.stage if err is not None else "engine"
        if store and token is not None:
            _store_generation(
                token,
                GenerationRecord(
                    user_id=user_id,
                    prompt=prompt,
                    engine="agentic",
                    status="failed",
                    quality=quality,
                    error_type=stage,
                    error_message=message,
                    spec=_lane_spec(ag),
                ),
            )
        return {"status": "error", "stage": stage, "message": message}

    # api.core.generate_part already wrote out/part.glb + out/part.stl (the viewer + STL download).
    metrics = None
    try:
        mesh = trimesh.load(str(out / "part.glb"), force="mesh")
        # Stability needs the Z-up STL frame (compute_stability rests on the lowest +Z face); the GLB
        # is Y-up. A missing/unreadable STL loses only stability, never the rest of the metrics.
        try:
            stl_mesh: Any = trimesh.load(str(out / "part.stl"), force="mesh")
        except Exception:
            stl_mesh = None
        metrics = _mesh_metrics(mesh, "PLA", stl_mesh=stl_mesh)
    except Exception:
        metrics = None

    # Complimentary / unlimited accounts (charge=False) build without metering -- no deduction.
    # The token None-check is for the demo path (charge is already False there); a charging call
    # always carries a real session token.
    remaining = (
        _deduct_credits(token, cost, f"generation_{quality}_agentic", job)
        if charge and token is not None
        else None
    )
    payload = ag.ok.model_dump(mode="json")
    _apply_manufacture_stamp(payload)
    # STEP is the editable B-rep an engineer opens in SolidWorks/Fusion/Onshape -- a first-class
    # download (ADR-0024), not just the viewer GLB + the STL mesh.
    files_local = {"glb": f"files/{job}/part.glb", "stl": f"files/{job}/part.stl"}
    if (out / "part.step").is_file():
        files_local["step"] = f"files/{job}/part.step"
    if (out / "certificate.html").is_file():  # the physics-route certificate (T3), additive
        files_local["certificate"] = f"files/{job}/certificate.html"
    payload["files"] = files_local
    # Durable storage (if configured): upload + return signed URLs so artifacts survive a redeploy
    # and are owner-scoped, not served from the public /files path. Falls back to /files otherwise.
    # Demo builds (store=False) have no owner — they stay on the ephemeral /files path by design.
    if store:
        glb_url = _upload_artifact(
            out / "part.glb", f"{user_id}/{job}/part.glb", "model/gltf-binary"
        )
        stl_url = _upload_artifact(out / "part.stl", f"{user_id}/{job}/part.stl", "model/stl")
        if glb_url and stl_url:
            files_remote = {"glb": glb_url, "stl": stl_url}
            step_url = _upload_artifact(
                out / "part.step", f"{user_id}/{job}/part.step", "application/step"
            )
            if step_url:
                files_remote["step"] = step_url
            payload["files"] = files_remote
    payload["metrics"] = metrics
    payload["build"] = None  # the agentic engine has no DSL op list; the live-build UI uses phases
    payload["credits_charged"] = cost if charge else 0
    payload["credits_remaining"] = remaining
    engine = _engine_for(ag.ok.gen_path)
    if store and token is not None:
        _store_generation(
            token,
            GenerationRecord(
                user_id=user_id,
                prompt=prompt,
                engine=engine,
                status="ok",
                quality=quality,
                model=LAYOUTSPEC_MODEL if engine == "layoutspec" else OPUS,
                spec={"gen_path": ag.ok.gen_path, **(_lane_spec(ag) or {})},
                validation=payload.get("validation"),
                files=payload.get("files"),
                usage=TokenUsage(cost_usd=float(ag.ok.cost_usd or 0.0)),
            ),
        )
    return payload


def _run_patch(
    spec_data: dict[str, Any],
    edit: str,
    quality: str,
    token: str,
    cost: int,
    user_id: str,
    charge: bool = True,
    structured: tuple[str, Literal["set", "increase", "decrease"], float | bool | str]
    | None = None,
    structural: tuple[Literal["add", "remove"], dict[str, Any] | None, int | None] | None = None,
) -> dict[str, Any]:
    """Apply one parametric or structural edit via the deterministic patch path (ADR-0037/0060) and
    project the result into the same contract /generate returns. Mirrors ``_run_agentic``: only a
    shipped part is charged; refusals/errors return the contract shape with no deduction, and every
    outcome lands a generations row (gen_path=standard:patch -> engine=layoutspec)."""
    job = uuid.uuid4().hex
    out = WORK_DIR / job
    out.mkdir(parents=True, exist_ok=True)
    if structural is not None:
        from forge_mind.api.patch import patch_part_structural

        res = patch_part_structural(
            spec_data, structural[0], out, placement=structural[1], index=structural[2]
        )
    elif structured is not None:
        from forge_mind.api.patch import patch_part_structured

        res = patch_part_structured(spec_data, structured[0], structured[1], structured[2], out)
    else:
        from forge_mind.api.patch import patch_part as _patch_part

        res = _patch_part(spec_data, edit, out)

    if res.status == "refused" and res.refused is not None:
        _store_generation(
            token,
            GenerationRecord(
                user_id=user_id,
                prompt=edit,
                engine="layoutspec",
                status="rejected",
                quality=quality,
                error_type="refused",
                error_message=res.refused.reason,
                spec=_lane_spec(res),
            ),
        )
        return {"status": "refused", "reason": res.refused.reason}
    if res.status != "ok" or res.ok is None:
        err = res.error
        message = err.message if err is not None else "Patch failed before a part was produced."
        stage = err.stage if err is not None else "patch"
        _store_generation(
            token,
            GenerationRecord(
                user_id=user_id,
                prompt=edit,
                engine="layoutspec",
                status="failed",
                quality=quality,
                error_type=stage,
                error_message=message,
                spec=_lane_spec(res),
            ),
        )
        return {"status": "error", "stage": stage, "message": message}

    metrics = None
    try:
        mesh = trimesh.load(str(out / "part.glb"), force="mesh")
        # Stability needs the Z-up STL frame (compute_stability rests on the lowest +Z face); the GLB
        # is Y-up. A missing/unreadable STL loses only stability, never the rest of the metrics.
        try:
            stl_mesh: Any = trimesh.load(str(out / "part.stl"), force="mesh")
        except Exception:
            stl_mesh = None
        metrics = _mesh_metrics(mesh, "PLA", stl_mesh=stl_mesh)
    except Exception:
        metrics = None

    remaining = _deduct_credits(token, cost, f"patch_{quality}_layoutspec", job) if charge else None
    payload = res.ok.model_dump(mode="json")
    _apply_manufacture_stamp(payload)
    files_local = {"glb": f"files/{job}/part.glb", "stl": f"files/{job}/part.stl"}
    if (out / "part.step").is_file():
        files_local["step"] = f"files/{job}/part.step"
    if (out / "certificate.html").is_file():  # the physics-route certificate (T3), additive
        files_local["certificate"] = f"files/{job}/certificate.html"
    payload["files"] = files_local
    glb_url = _upload_artifact(out / "part.glb", f"{user_id}/{job}/part.glb", "model/gltf-binary")
    stl_url = _upload_artifact(out / "part.stl", f"{user_id}/{job}/part.stl", "model/stl")
    if glb_url and stl_url:
        files_remote = {"glb": glb_url, "stl": stl_url}
        step_url = _upload_artifact(
            out / "part.step", f"{user_id}/{job}/part.step", "application/step"
        )
        if step_url:
            files_remote["step"] = step_url
        payload["files"] = files_remote
    payload["metrics"] = metrics
    payload["build"] = None
    payload["credits_charged"] = cost if charge else 0
    payload["credits_remaining"] = remaining
    _store_generation(
        token,
        GenerationRecord(
            user_id=user_id,
            prompt=edit,
            engine="layoutspec",
            status="ok",
            quality=quality,
            model=HAIKU,
            # The full parametric model + the one-field diff ride the jsonb spec column, so a
            # patched version is REPLAYABLE from the row alone (recoverability without volumes).
            spec={
                "gen_path": "standard:patch",
                **(_lane_spec(res) or {}),
                "layout_spec": payload.get("spec"),
                "patch": payload.get("patch"),
            },
            validation=payload.get("validation"),
            files=payload.get("files"),
            usage=TokenUsage(cost_usd=float(res.ok.cost_usd or 0.0)),
        ),
    )
    return payload


@app.post("/patch")
def patch(body: PatchBody, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Parametric edit-survival (ADR-0037): change ONE spec parameter, re-derive the rest.
    Auth + credit gate + in-flight guard are identical to /generate; an infeasible or
    unmappable edit refuses with the reason and charges nothing."""
    token, user_id = _require_user(authorization)
    edit = body.edit.strip()
    if not edit:
        raise HTTPException(status_code=400, detail="empty edit")
    if not body.spec:
        raise HTTPException(status_code=400, detail="missing spec (parametric model)")
    skip_charge, cost = _resolve_credit_gate(token, body.quality)
    _acquire_inflight(user_id)
    try:
        return _run_patch(
            body.spec, edit, body.quality, token, cost, user_id, charge=not skip_charge
        )
    finally:
        _release_inflight(user_id)


@app.post("/patch-field")
def patch_field(
    body: PatchFieldBody, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Deterministic STRUCTURED edit (ADR-0060): a direct UI control change (field + op + value).
    No LLM, no credit -- the instant slider/dropdown path. Same re-derive + strict gates + the same
    no-false-PASS refusals as /patch; an infeasible / out-of-range value refuses and ships nothing."""
    token, user_id = _require_user(authorization)
    if not body.spec:
        raise HTTPException(status_code=400, detail="missing spec (parametric model)")
    if not body.field_path:
        raise HTTPException(status_code=400, detail="missing field_path")
    _acquire_inflight(user_id)
    try:
        return _run_patch(
            body.spec,
            f"{body.field_path} {body.op} {body.value}",
            "standard",
            token,
            0,
            user_id,
            charge=False,
            structured=(body.field_path, body.op, body.value),
        )
    finally:
        _release_inflight(user_id)


@app.post("/patch-structure")
def patch_structure(
    body: PatchStructureBody, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Deterministic STRUCTURAL edit (ADR-0060 slice C): ADD or REMOVE an interface/feature on a
    composable-plate part. No LLM, no credit -- the add/remove-feature path of the editable blueprint.
    Same re-derive + strict gates + no-false-PASS refusals as /patch; an infeasible / out-of-range
    edit refuses and ships nothing."""
    token, user_id = _require_user(authorization)
    if not body.spec:
        raise HTTPException(status_code=400, detail="missing spec (parametric model)")
    placement: dict[str, Any] | None = None
    if body.op == "add":
        if not body.ref or not body.face:
            raise HTTPException(status_code=400, detail="add needs ref + face")
        placement = {
            k: v
            for k, v in {
                "ref": body.ref,
                "face": body.face,
                "offset_mm": body.offset_mm,
                "spacing_mm": body.spacing_mm,
                "component_id": body.component_id,
                "fit": body.fit,
                "name": body.name,
            }.items()
            if v is not None
        }
        edit_label = f"add {body.ref} on {body.face}"
    else:  # remove
        if body.index is None:
            raise HTTPException(status_code=400, detail="remove needs index")
        edit_label = f"remove interface {body.index}"
    _acquire_inflight(user_id)
    try:
        return _run_patch(
            body.spec,
            edit_label,
            "standard",
            token,
            0,
            user_id,
            charge=False,
            structural=(body.op, placement, body.index),
        )
    finally:
        _release_inflight(user_id)


@app.post("/blueprint-fields")
def blueprint_fields_endpoint(
    body: FieldsBody, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """The editable-field catalog for a stored LayoutSpec -- the read side of the editable
    blueprint, so the UI can render direct controls (sliders/dropdowns/toggles). No LLM, no
    credit. The structured EDIT (apply) path is /patch (and the deterministic field path)."""
    _require_user(authorization)
    if not body.spec:
        raise HTTPException(status_code=400, detail="missing spec (parametric model)")
    from forge_mind.api.patch import blueprint_fields
    from forge_mind.layout.patch import SpecPatchError

    try:
        return {"fields": blueprint_fields(body.spec)}
    except SpecPatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --- The blueprint pregen + the job-scoped parametric API (founder direction 2026-07-02) -------
# A DRAFT is a simple-prompt, small-model author pass run in the real sandbox: honest geometry
# before the validated build, no credit. Every draft (and every agentic part, which persists its
# final script as program.py) leaves a parametric job dir behind, so ONE pair of endpoints edits
# both: GET /jobs/{job}/params reads the PARAMS contract, POST /jobs/{job}/rebuild re-executes
# the stored script with overrides. Compute-only (sandbox), no LLM, no credit.


def _job_dir_checked(job: str, user_id: str) -> Path:
    from forge_mind.server.job_ownership import owns

    if not job.isalnum():
        raise HTTPException(status_code=404, detail="not found")
    if not owns(WORK_DIR, job, user_id):
        raise HTTPException(status_code=404, detail="not found")
    return WORK_DIR / job


def _draft_glb_writer(stl: Path, glb: Path) -> bool:
    from forge_mind.api.core import _stl_to_glb

    return _stl_to_glb(stl, glb)


@app.post("/draft")
def draft_endpoint(
    body: DraftBody, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Author + build a quick DRAFT preview (blueprint pregen). Signed-in, NO credit charge —
    the draft model is the cheap tier and the sandbox is compute-only. The response is honest
    about what it is: draft geometry that never saw the validator."""
    _, user_id = _require_user(authorization)
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")

    from forge_mind.api.draft import DraftError, author_draft, build_into_job

    _acquire_inflight(user_id)
    try:
        job = uuid.uuid4().hex
        out = WORK_DIR / job
        out.mkdir(parents=True, exist_ok=True)
        from forge_mind.server.job_ownership import claim

        claim(WORK_DIR, job, user_id)
        code, _spec = author_draft(prompt)
        built = build_into_job(code, out, glb_writer=_draft_glb_writer)
        return {
            "job": job,
            "params": built.params,
            "files": {"glb": f"files/{job}/part.glb", "stl": f"files/{job}/part.stl"},
            "draft": True,
            "note": "Draft preview — massing only, not validated. Generate runs the full engine.",
        }
    except DraftError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        _release_inflight(user_id)


@app.get("/jobs/{job}/params")
def job_params_endpoint(
    job: str, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """The PARAMS contract of a job's stored script (draft or finished agentic part). 404 when
    the job has no parametric script — the UI then simply doesn't offer the panel."""
    from forge_mind.api.draft import job_params

    _, user_id = _require_user(authorization)
    params = job_params(_job_dir_checked(job, user_id))
    if params is None:
        raise HTTPException(status_code=404, detail="no parametric script for this job")
    return {"params": params}


@app.post("/jobs/{job}/rebuild")
def job_rebuild_endpoint(
    job: str, body: JobRebuildBody, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """Re-execute a job's stored script with PARAMS overrides — the parametric edit the viewer's
    click-to-iterate panel POSTs. Sandbox-only (~seconds), no LLM, no credit. The mesh files are
    overwritten in place; `validation` is null because NO validator ran on the rebuilt mesh —
    the panel says so rather than inheriting the old verdict."""
    _, user_id = _require_user(authorization)
    if not body.overrides:
        raise HTTPException(status_code=400, detail="no overrides given")

    from forge_mind.api.draft import DraftError, rebuild_job

    _acquire_inflight(user_id)
    try:
        built = rebuild_job(
            _job_dir_checked(job, user_id), body.overrides, glb_writer=_draft_glb_writer
        )
        return {
            "params": built.params,
            "stl_url": f"files/{job}/part.stl",
            "glb_url": f"files/{job}/part.glb",
            "validation": None,
        }
    except DraftError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        _release_inflight(user_id)


@app.post("/generate")
def generate(
    body: GenerateBody, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    token, user_id = _require_user(authorization)

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")

    # Credits: resolve the entitlement and apply the quality + credit gate before spending the API
    # call. Fail CLOSED on a read failure; only a successful part is charged (a refusal/error below
    # returns early without a charge), and a complimentary/unlimited account skips metering entirely.
    skip_charge, cost = _resolve_credit_gate(token, body.quality)

    # One in-flight generation per user (guards against parallel cost floods).
    _acquire_inflight(user_id)
    try:
        # The agentic engine is the main path: it builds arbitrary parts from plain English.
        return _run_agentic(
            prompt,
            body.quality,
            token,
            cost,
            user_id,
            charge=not skip_charge,
            printable=body.printable,
            brief_context=_brief_context_for(body.design_session_id),
        )
    finally:
        _release_inflight(user_id)


@app.post("/generate/stream")
def generate_stream(
    body: GenerateBody, authorization: str | None = Header(default=None)
) -> StreamingResponse:
    """Server-Sent Events sibling of POST /generate. Emits `phase` events as each stage runs and a
    `program` event the moment the op list exists (before compile + validate), so the viewer reveals
    the part operation-by-operation WHILE the slower steps finish, then a final `result` event with
    the same payload /generate returns. Auth + credit gate are identical; only a real part is charged.
    """
    token, user_id = _require_user(authorization)
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    skip_charge, cost = _resolve_credit_gate(token, body.quality)
    # One in-flight generation per user; released in the worker's finally below.
    _acquire_inflight(user_id)

    events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()

    def worker() -> None:
        # The engine now drives REAL phase transitions via `emit` (plan/cad/export, each fired as the
        # stage actually starts/ends -- see api.core.generate_part / agentic.result.produce_part). A
        # keepalive re-asserts the CURRENT phase every 10s so the SSE channel stays warm through the
        # long authoring step (defeats proxy idle-timeouts on a silent long request) WITHOUT faking a
        # phase the engine has not reached. _run_agentic does the build + projection + credit charge.
        stop = threading.Event()
        state = {"phase": "intent"}  # the phase the engine is genuinely in; read by the keepalive

        def emit(key: str, status: str) -> None:
            if status == "active":
                state["phase"] = key
            events.put(("phase", {"key": key, "status": status}))

        def keepalive() -> None:
            while not stop.wait(10.0):
                # Re-assert the phase actually in progress -- honest, not a scripted heartbeat.
                events.put(("phase", {"key": state["phase"], "status": "active"}))

        emit("intent", "active")
        emit("intent", "done")
        threading.Thread(target=keepalive, daemon=True).start()
        try:
            payload = _run_agentic(
                prompt,
                body.quality,
                token,
                cost,
                user_id,
                charge=not skip_charge,
                on_phase=emit,
                printable=body.printable,
                brief_context=_brief_context_for(body.design_session_id),
            )
            stop.set()
            if payload.get("status") == "ok":
                emit("export", "done")
            events.put(("result", payload))
        except Exception:
            stop.set()
            logging.getLogger("forge_mind.server").exception("generation stream failed")
            events.put(
                (
                    "result",
                    {
                        "status": "error",
                        "stage": "stream",
                        "message": "internal error during generation",
                    },
                )
            )
        finally:
            stop.set()
            _release_inflight(user_id)
            events.put(("__done__", {}))

    threading.Thread(target=worker, daemon=True).start()

    def stream() -> Any:
        yield ": open\n\n"  # flush headers + open the channel immediately
        while True:
            kind, payload = events.get()
            if kind == "__done__":
                break
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # No `Connection` header: it is a hop-by-hop header that is invalid over HTTP/2 (browser <->
        # Fly) and can make the client reject or truncate the stream. Cache-Control + X-Accel-Buffering
        # are enough to keep proxies from buffering.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Flavor A (ADR-0056): upload an STL -> intent-free pre-print check. The check is $0 LOCAL geometry
# compute (no LLM, bounded), so the only abuse surface is disk/CPU -- auth + a size cap cover it
# (guardrail #5's bounded-compute clause). STL-first; STEP ingest is gated on the import-fidelity census.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# Bound the FEA solve's latency: skip it on very high-poly uploads (the stress lens just doesn't appear).
_FEA_MAX_FACES = 80000


def _bake_stress_colors(ymesh: Any, stress: Any) -> None:
    """Bake the ILLUSTRATIVE relative von Mises field (ADR-0058) into the GLB as per-vertex viridis
    colours (COLOR_0). The field is aligned to the INPUT vertex order, which ``to_gltf_yup`` preserves,
    so colour[i] <-> vertex i. The viewer's stress lens renders these; every other lens ignores them."""
    import matplotlib
    import numpy as np

    vm = np.asarray(stress.von_mises, dtype=float)
    scale = stress.von_mises_p95 if stress.von_mises_p95 > 0 else stress.von_mises_max
    norm = np.clip(vm / scale, 0.0, 1.0) if scale > 0 else np.zeros_like(vm)
    rgba = (matplotlib.colormaps["viridis"](norm) * 255.0).astype(np.uint8)
    if len(rgba) == len(ymesh.vertices):
        ymesh.visual.vertex_colors = rgba


def _hole_wall_thickness(
    mesh: Any,
    axis_point: tuple[float, float, float],
    axis_dir: tuple[float, float, float],
    radius: float,
    axial_extent: tuple[float, float] | None,
) -> float | None:
    """Minimum material wall thickness from a hole wall to the nearest part edge, by casting radial rays
    in the plane perpendicular to the hole axis. None if not measurable. CONSERVATIVE on purpose: a
    direction where the material is open to the edge (a clean hole-wall hit but no outer wall) reports 0,
    biasing toward a WARN — never a false PASS (a false 'edge distance OK' is the forbidden failure)."""
    try:
        import numpy as np

        a = np.array(axis_dir, dtype=float)
        n = float(np.linalg.norm(a))
        if n == 0.0 or radius <= 0.0:
            return None
        a = a / n
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u = u / float(np.linalg.norm(u))
        v = np.cross(a, u)
        c = np.array(axis_point, dtype=float)
        # The axis position OCP returns can sit at the face's parametric origin — possibly OUTSIDE the
        # material along the axis (a through-hole's axis point at z beyond the plate). Slide the ray
        # origin to the MIDDLE of the hole's measured axial span so the perpendicular rays cross material.
        if axial_extent is not None:
            mid = 0.5 * (axial_extent[0] + axial_extent[1])
            c = c + a * (mid - float(np.dot(c, a)))
        rays = 48
        angs = np.linspace(0.0, 2.0 * np.pi, rays, endpoint=False)
        dirs = np.outer(np.cos(angs), u) + np.outer(np.sin(angs), v)
        origins = np.repeat(c[None, :], rays, axis=0)
        locs, ray_idx, _tri = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=dirs, multiple_hits=True
        )
        if len(locs) == 0:
            return None
        along = np.einsum("ij,ij->i", locs - c[None, :], dirs[ray_idx])
        walls: list[float] = []
        for i in range(rays):
            ds = np.sort(along[(ray_idx == i) & (along > 1e-6)])
            if ds.size == 0:
                continue
            first = float(ds[0])
            if not (0.5 * radius <= first <= 1.5 * radius):
                continue  # this ray did not cleanly start at the hole wall -> unmeasurable, skip
            beyond = ds[ds > first + 1e-6]
            walls.append(float(beyond[0]) - first if beyond.size else 0.0)
        if len(walls) < rays // 4:  # too few clean measurements -> don't trust it (fail closed)
            return None
        return round(min(walls), 3)
    except Exception:
        return None


def _detect_holes(brep_part: Any, mesh: Any) -> list[DetectedHole]:
    """Concave cylindrical bores from a B-rep, as plain DetectedHole (stl_z_up frame, == the STL the
    pipeline reads). Best-effort: any failure -> [] (no holes -> BoltBrain not_run), never a fabricated
    feature. Only concave faces are holes; bosses/unmeasurable are excluded. ``wall_mm`` (the surrounding
    material thickness) is measured on the tessellated ``mesh`` for the edge-distance check."""
    try:
        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = cluster_coaxial(extract_cylinders(brep_part))
    except Exception:
        return []
    holes: list[DetectedHole] = []
    for s in stacks:
        if s.concave is not True:
            continue
        holes.append(
            DetectedHole(
                diameter_mm=round(2.0 * s.radius, 3),
                center_mm=Vec3(x=s.axis_point[0], y=s.axis_point[1], z=s.axis_point[2]),
                axis=Vec3(x=s.axis_dir[0], y=s.axis_dir[1], z=s.axis_dir[2]),
                depth_mm=s.depth_mm,
                segments=s.segments,
                wall_mm=_hole_wall_thickness(
                    mesh, s.axis_point, s.axis_dir, s.radius, s.axial_extent
                ),
            )
        )
    return holes


_G_M_S2 = 9.80665
_LBF_N = 4.4482216


def _e_for_material(material: str | None) -> float:
    """Printed-coupon Young's modulus (MPa) for the stated material, nominal of the published range; the
    canonical default when unknown/unstated. E only scales deflection — the heatmap/verdict caveats stand."""
    if not material:
        return DEFAULT_E_MPA
    try:
        from forge_mind.physics.static_load import PRINTED_E_RANGE_MPA

        rng = PRINTED_E_RANGE_MPA.get(material.strip().upper().split()[0])
    except Exception:
        rng = None
    return (rng[0] + rng[1]) / 2.0 if rng else DEFAULT_E_MPA


def _fea_load_response(mesh: Any, intent: Intent) -> Any:
    """Validated mesh-FEA response under ``intent.load`` (ADR-0064), or None when no load is stated or the
    solve is refused (gated on ``fea_is_validated`` inside ``estimate_load_response``). Converts the stated
    magnitude to newtons and the material to a coupon modulus."""
    load = intent.load
    if load is None:
        return None
    mag = float(load.magnitude)
    if load.unit == "kg":
        load_n = mag * _G_M_S2
    elif load.unit == "lbf":
        load_n = mag * _LBF_N
    elif load.unit == "N":
        load_n = mag
    else:
        return None
    if load_n <= 0.0:
        return None
    d = load.direction
    load_dir = (d.x, d.y, d.z) if d is not None else (0.0, 0.0, -1.0)
    try:
        return estimate_load_response(
            mesh, load_n, load_dir=load_dir, e_mpa=_e_for_material(intent.material)
        )
    except Exception:
        return None


def _intrinsic_upload(
    file: UploadFile, intent_json: str | None = None
) -> tuple[dict[str, Any], Path, str]:
    """Shared core of the pre-print check: validate the STL, run validate_intrinsic, build the viewer
    GLB + metrics + ILLUSTRATIVE stress, and assemble the payload (LOCAL file paths). No auth and no
    durable storage -- the authed and public-demo endpoints wrap this. Raises HTTPException on a
    bad/oversized/unreadable upload (fail-closed at the boundary). Returns (payload, out_dir, job).

    ``intent_json`` is the optional physics ``Intent`` (JSON) the frontend collects; it unlocks the
    intent-gated suites (FitProof on mates_with). A malformed intent degrades to empty (the dependent
    findings become not_run) -- it never breaks the upload and never fabricates a datum."""
    filename = (file.filename or "").lower()
    # STEP ingest rides the physics-report flag (slice 2): with the flag OFF only .stl is accepted
    # (byte-identical to before). With it ON, a .step/.stp is imported to a B-rep, tessellated to the STL
    # the rest of the pipeline reads, and additionally unlocks the editable-CAD (B-rep) integrity finding.
    allow_step = physics_report_enabled()
    is_step = allow_step and filename.endswith((".step", ".stp"))
    if not (filename.endswith(".stl") or is_step):
        detail = (
            "upload an .stl or .step file"
            if allow_step
            else "upload an .stl file (STL-first; STEP is coming)"
        )
        raise HTTPException(status_code=400, detail=detail)
    data = file.file.read(_MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"file too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
        )

    job = uuid.uuid4().hex
    out = WORK_DIR / job
    out.mkdir(parents=True, exist_ok=True)
    stl_path = out / "part.stl"
    brep_part: Any = None
    step_path: Path | None = None
    if is_step:
        # Import the B-rep, then tessellate to the STL the rest of the pipeline reads. Fail CLOSED: an
        # unreadable STEP is a 400, never a silent pass (and never a verdict on geometry we can't read).
        step_path = out / "part.step"
        step_path.write_bytes(data)
        try:
            from build123d import export_stl, import_step

            brep_part = import_step(str(step_path))
            export_stl(brep_part, str(stl_path))
        except Exception:
            raise HTTPException(
                status_code=400, detail="could not read that STEP file as a solid"
            ) from None
    else:
        stl_path.write_bytes(data)

    # Ingest gate: confirm it reads as a single non-empty mesh before any check is trusted (fail closed).
    try:
        mesh = trimesh.load(str(stl_path), force="mesh")
    except Exception:
        mesh = None
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise HTTPException(status_code=400, detail="could not read that file as an STL mesh")

    report = validate_intrinsic(stl_path, part=brep_part, step_path=step_path)

    # Viewer GLB: rotate Z-up -> Y-up so the existing viewer + the Wave 1 physics lenses render it.
    # `metrics` is the SAME PartMetrics shape /generate returns (Y-up CoM from the GLB + stand-or-tip
    # from the Z-up mesh) so the CoM / balance / overhang lenses fire on the uploaded part unchanged.
    # `stress` (ADR-0058) is the ILLUSTRATIVE relative von Mises summary; its per-vertex field is baked
    # into the GLB as colours for the stress lens. REPORTED-only -- it never touches the verdict.
    files_local: dict[str, str] = {"stl": f"files/{job}/part.stl"}
    metrics: dict[str, Any] | None = None
    stress: dict[str, Any] | None = None
    stress_colors: Any = None  # per-vertex RGBA (Y-up order) — carried into the motion GLB
    try:
        glb_path = out / "part.glb"
        ymesh = to_gltf_yup(
            mesh.copy()
        )  # Y-up; preserves vertex order so the FEA field aligns by index
        sf = estimate_stress(mesh) if len(mesh.faces) <= _FEA_MAX_FACES else None
        if sf is not None:
            _bake_stress_colors(ymesh, sf)
            stress_colors = ymesh.visual.vertex_colors
            stress = {
                "available": True,
                "von_mises_max": sf.von_mises_max,
                "von_mises_p95": sf.von_mises_p95,
                "max_location_mm": {
                    "x": sf.max_location_mm[0],
                    "y": sf.max_location_mm[1],
                    "z": sf.max_location_mm[2],
                },
                "load_case": sf.load_case,
                "note": sf.note,
            }
        ymesh.export(str(glb_path))
        files_local["glb"] = f"files/{job}/part.glb"
        metrics = _mesh_metrics(trimesh.load(str(glb_path), force="mesh"), "PLA", stl_mesh=mesh)
    except Exception:
        metrics = None
        stress = None

    payload: dict[str, Any] = {
        "status": "ok",
        "report": report.model_dump(mode="json"),
        "metrics": metrics,
        "stress": stress,
        "files": files_local,
        "fabrication_mode": "print_3d",
    }

    # Physics-validation report (ADR-0063) — ADDITIVE, behind FORGE_PHYSICS_REPORT (default off → this
    # block is skipped and the payload is byte-identical). A READ-ONLY projection of the SAME finished
    # `report`; it never gates or touches the verdict. Vitals + locators come from the Z-up STL mesh so
    # every coordinate stays in the `stl_z_up` frame the contract promises (NOT the Y-up `metrics` dict).
    if physics_report_enabled():
        try:
            ext = mesh.extents
            bbox = Vec3(x=float(ext[0]), y=float(ext[1]), z=float(ext[2]))
            area: float | None = float(mesh.area)
        except Exception:
            bbox, area = Vec3(), None
        part_name = Path(file.filename or "part.stl").stem or "Uploaded part"
        holes = _detect_holes(brep_part, mesh) if brep_part is not None else []
        parsed_intent = Intent()
        if intent_json:
            try:
                parsed_intent = Intent.model_validate_json(intent_json)
            except Exception:
                parsed_intent = (
                    Intent()
                )  # malformed -> empty (gated checks not_run), never break upload
        # Decision-grade LoadProof (ADR-0064): a validated mesh-FEA solve under the stated load, on the
        # Z-up STL (base fixed / top loaded). None unless intent.load is given AND the solver is validated.
        load_response = _fea_load_response(mesh, parsed_intent)
        # MotionProof (ADR-0086): measured kinematics on the same Z-up mesh — the viewer animates ONLY
        # what these claims state. Advisory-only; project_motion never raises.
        motion_block, motion_findings = project_motion(mesh, brep_part)
        # Slice 2: animatable claims are baked into the SAME frozen ck_joint contract the engine's
        # declared mechanisms use, so the viewer's existing articulation rig (JointPanel + FK) moves
        # exactly what was certified — slider limits ARE the measured arc. Fail-soft: on any failure
        # the flat GLB stays and the part renders rigid (honest degradation).
        if (
            motion_block is not None
            and (out / "part.glb").exists()
            and any(c.status in ("free", "arc_limited") for c in motion_block.claims)
        ):
            try:
                # Same welded body derivation as project_motion, so body_<i> ids agree.
                welded = mesh.copy()
                welded.merge_vertices()
                bodies_zup = [
                    b for b in welded.split(only_watertight=False) if len(b.faces) > 0
                ] or [welded]
                if len(bodies_zup) == 1:
                    # single body: keep the ORIGINAL mesh so the baked stress colors
                    # stay vertex-aligned (welding changes the vertex count)
                    bodies_zup = [mesh]
                export_motion_glb(
                    out / "part.glb",
                    bodies_zup,
                    list(motion_block.claims),
                    stress_colors if len(bodies_zup) == 1 else None,
                )
            except Exception:
                pass  # rigid render is the honest fallback, never a broken viewer
        physics_report = compose_physics_report(
            report,
            intent=parsed_intent,
            bbox_mm=bbox,
            surface_area_mm2=area,
            scope="single_part",
            part_name=part_name,
            holes=holes,
            load_response=load_response,
            motion=motion_block,
            motion_findings=motion_findings,
        )
        payload["physics_report"] = physics_report.model_dump(mode="json", by_alias=True)

    return payload, out, job


@app.post("/validate-upload")
def validate_upload(
    file: Annotated[UploadFile, File()],
    intent: Annotated[str | None, Form()] = None,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Upload an STL -> the honest intent-free pre-print check (validate_intrinsic) + a viewer GLB.
    Authed + size-capped. Returns an IntrinsicReport that NEVER claims fit/function and never says
    'safe'. /generate is untouched. ``intent`` (optional JSON) unlocks the intent-gated physics
    suites (FitProof)."""
    _token, user_id = _require_user(authorization)
    payload, out, job = _intrinsic_upload(file, intent_json=intent)
    # Durable storage (optional), same owner-scoped pattern as /generate; falls back to /files.
    glb_url = _upload_artifact(out / "part.glb", f"{user_id}/{job}/part.glb", "model/gltf-binary")
    stl_url = _upload_artifact(out / "part.stl", f"{user_id}/{job}/part.stl", "model/stl")
    if glb_url and stl_url:
        payload["files"] = {"glb": glb_url, "stl": stl_url}
    return payload


# Public "try it without signing in" demo: the SAME intrinsic check, no auth, no durable storage,
# IP-throttled + size-capped. In-memory window (per worker, resets on redeploy) -- a soft abuse brake
# for a public demo, not a security control. The honest verdict is identical to the authed path.
_DEMO_MAX_PER_HOUR = 20
_DEMO_HITS: dict[str, list[float]] = {}


def _demo_rate_limit(request: Request) -> None:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    recent = [t for t in _DEMO_HITS.get(ip, []) if now - t < 3600.0]
    if len(recent) >= _DEMO_MAX_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"demo limit reached ({_DEMO_MAX_PER_HOUR}/hour); sign in for unlimited checks",
        )
    recent.append(now)
    _DEMO_HITS[ip] = recent


# The guest TASTE-BUILD (founder decision 2026-07-02): one real generation per IP per day, so an
# anonymous visitor experiences the actual engine instead of the canned sample. Much tighter than
# the upload-check cap because every build spends real LLM money. Standard quality only; nothing
# is persisted (store=False) — the part lives on the ephemeral /files path until they sign up.
_TASTE_HITS: dict[str, list[float]] = {}
_TASTE_MAX_PER_DAY = 1
_TASTE_WINDOW_S = 86400.0


def _taste_rate_limit(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    recent = [t for t in _TASTE_HITS.get(ip, []) if now - t < _TASTE_WINDOW_S]
    if len(recent) >= _TASTE_MAX_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail="free build used for today — create a free account for 10 credits",
        )
    recent.append(now)
    _TASTE_HITS[ip] = recent
    return ip


@app.post("/generate-demo")
def generate_demo(request: Request, body: GenerateBody) -> dict[str, Any]:
    """PUBLIC, no-sign-in taste-build: the SAME agentic engine as /generate, standard quality,
    IP-rate-limited to one real build per day. No credits exist to charge, nothing is stored or
    uploaded; the response says demo=True so the UI labels it a guest build honestly."""
    ip = _taste_rate_limit(request)
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    payload = _run_agentic(
        prompt,
        "standard",
        token=None,
        cost=0,
        user_id=f"demo:{ip}",
        charge=False,
        printable=body.printable,
        store=False,
    )
    payload["demo"] = True
    return payload


@app.post("/validate-upload-demo")
def validate_upload_demo(
    request: Request,
    file: Annotated[UploadFile, File()],
    intent: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """PUBLIC, no-sign-in pre-print check -- the SAME validate_intrinsic logic as /validate-upload,
    IP-rate-limited and size-capped, with no durable storage. For the public 'try it' funnel.
    ``intent`` (optional JSON) unlocks the intent-gated physics suites (FitProof)."""
    _demo_rate_limit(request)
    payload, _out, _job = _intrinsic_upload(file, intent_json=intent)
    payload["demo"] = True
    return payload


@app.get("/files/{job}/{name}")
def files(job: str, name: str) -> FileResponse:
    # `job` is a hex uuid and `name` is one of a few fixed filenames — guard against traversal.
    if not job.isalnum() or name not in ("part.glb", "part.stl", "part.step", "certificate.html"):
        raise HTTPException(status_code=404, detail="not found")
    path = WORK_DIR / job / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    if name.endswith(".glb"):
        mime = "model/gltf-binary"
    elif name.endswith(".step"):
        mime = "application/step"
    elif name.endswith(".html"):
        mime = "text/html"  # the physics-route certificate (T3): a static computed-facts page
    else:
        mime = "model/stl"
    return FileResponse(path, media_type=mime)


# --- Planner-first design intake -------------------------------------------------------------
# The guided flow (sessions + uploaded images + Draw it! sketches + clarifying questions -> a
# DesignBrief -> the existing pipeline) is served on /api/design-sessions/* by THIS backend.
# Sessions + assets persist under WORK_DIR (durable across restarts). Planning is open; the costly
# generate endpoint reuses the same Supabase sign-in gate as POST /generate.
def _intake_auth(authorization: str | None = Header(default=None)) -> None:
    _require_user(authorization)


class _ResilientPlannerBrain:
    """Use the LLM brain so the planner actually understands the request (and any uploaded image) and
    asks questions that fit the specific part, instead of the keyword heuristic's canned per-type set.
    Falls back to the deterministic heuristic brain on any LLM error so a transient blip never breaks
    the guided flow."""

    def __init__(self) -> None:
        self._llm = LlmPlannerBrain()
        self._fallback = HeuristicPlannerBrain()

    def analyze(self, session: Any) -> Any:
        try:
            return self._llm.analyze(session)
        except Exception:
            # Visibility: a silently-failing LLM planner is exactly what made the questions look
            # generic before (it fell back here every time). Log the real error so it shows up in
            # `fly logs`, then fall back so the guided flow still works.
            logging.getLogger("forge_mind.intake").exception(
                "LLM planner failed; falling back to heuristic questions"
            )
            return self._fallback.analyze(session)


def _make_planner_agent() -> PlannerAgent:
    # LLM understanding needs a key; without one, the deterministic heuristic brain is the default.
    if os.environ.get("ANTHROPIC_API_KEY"):
        return PlannerAgent(_ResilientPlannerBrain())
    return PlannerAgent()


_intake_service = IntakeService(
    store=JSONFileSessionStore(WORK_DIR / "_intake" / "sessions"),
    asset_dir=WORK_DIR / "_intake" / "assets",
    agent=_make_planner_agent(),
)
app.include_router(_intake_router(_intake_service, auth_dependency=_intake_auth))

# Stripe billing: /api/billing/checkout, /api/billing/portal, /api/stripe/webhook.
# Inert (503) until STRIPE_SECRET_KEY + STRIPE_PRICE_PRO + SUPABASE_SERVICE_ROLE_KEY are set.
app.include_router(create_billing_router())


# --- Problem-mode lab endpoints (ADR-0087 S6b) -------------------------------------------------
# Flag-gated behind FORGE_PROBLEM_MODE (404 when off -> the deployed contract is byte-identical
# by default; the S7 bench gates the flag) and auth-gated like /generate. The produce adapter
# REUSES api.core's GeneratedPart composition (produce_for_invention) — never re-derived here.
# Credit metering on /lab/confirm is S9's wiring, before the flag ever turns on for real users.
def _lab_produce(prompt: str, extra_context: str, material: str = "PLA") -> GeneratedPart:
    from forge_mind.api.core import produce_for_invention
    from forge_mind.fabrication import FabricationMode

    job = uuid.uuid4().hex
    out = WORK_DIR / job
    out.mkdir(parents=True, exist_ok=True)
    # Problem-mode v1 IS the printable lane (the consumer thesis: "print it yourself") — the
    # engine designs for FDM; a cnc intake refuses before this adapter is ever reached.
    return produce_for_invention(
        prompt,
        out,
        extra_context=extra_context or None,
        material=material,
        fabrication_mode=FabricationMode.PRINT_3D,
    )


def _lab_billing_gate(authorization: str | None) -> tuple[str, bool, int]:
    """The confirm paywall (S9): resolve the session + entitlement BEFORE any engine spend —
    the exact /generate gate. Problem-mode builds meter as 'standard' quality (1 credit) in v1."""
    token, _user = _require_user(authorization)
    skip_charge, cost = _resolve_credit_gate(token, "standard")
    return token, skip_charge, cost


def _lab_billing_charge(token: str, cost: int, session_id: str) -> None:
    """Charge the successfully-built confirm (the /generate rule: only a real part is charged;
    a deduct failure is logged as a ledger gap, never a reason to withhold the part)."""
    if _deduct_credits(token, cost, "generation_standard_problem", session_id) is None:
        logging.getLogger("forge_mind.server").warning(
            "lab confirm charge failed (ledger gap): session=%s cost=%s", session_id, cost
        )


app.include_router(
    create_lab_router(
        store=_intake_service.store,
        produce=_lab_produce,
        auth_dependency=_intake_auth,
        billing_gate=_lab_billing_gate,
        billing_charge=_lab_billing_charge,
    )
)
