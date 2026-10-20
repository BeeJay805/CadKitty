"""Problem-mode lab endpoints (ADR-0087 S6b) — flag-gated dark behind ``FORGE_PROBLEM_MODE``.

Four endpoints carry the single-pass loop the 2026-07-06 addendum specifies:

  * ``POST /lab/intake``   — classify; a SPEC prompt routes back to ``/generate`` BYTE-UNTOUCHED
    (P12/P17: no session, no model call); a PROBLEM runs the one gather pass and returns the
    complete typed form (or goes straight to ``spec_ready``).
  * ``POST /lab/answers``  — the one answer round: deterministic (cached tree, zero model calls),
    ending in ``spec_ready``, one conflict re-ask, or a REFUSAL naming the un-gettable fact.
  * ``GET  /lab/spec/{id}``— the confirm card: the tree with per-leaf provenance, the fact ledger,
    assumptions, the open form or the refusal.
  * ``POST /lab/confirm``  — the ONLY spend path (L3): builds the EXACT spec the user confirmed,
    then runs the advisory env checks and returns the assumption ledger.

Every endpoint 404s when the flag is off (the FORGE_INVENTION precedent — the deployed contract
is byte-identical by default) and sits behind the same auth gate as ``/generate``. Credit metering
on confirm is deliberately S9's wiring: the flag stays OFF until the S7 bench passes, and billing
a dark endpoint now would be dead surface — S9 MUST wire it before the flag turns on."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.config import BRAIN_MODEL
from forge_mind.intake.session import DesignSession, SessionStore
from forge_mind.invention.problem_intake import (
    IntakeAnswerError,
    IntakeAnswerItem,
    ProblemIntake,
    ProblemState,
    apply_answers,
    classify_intake,
    confirm_and_build,
    start_problem,
)
from forge_mind.invention.problem_mode import ProduceLike, problem_mode_enabled
from forge_mind.invention.research import ResearchProviderLike, get_research_provider
from forge_mind.llm.provider import LLMClientLike


class IntakeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=1)
    context: str = ""
    # N1: the user's printable-materials set (empty = all known) + the fabrication base fact
    materials: list[str] = Field(default_factory=list)
    process: str = "print"


class AnswersBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1)
    answers: list[IntakeAnswerItem] = Field(default_factory=list)


class ConfirmBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1)


def create_lab_router(
    *,
    store: SessionStore,
    produce: ProduceLike,
    client_factory: Callable[[], LLMClientLike] | None = None,
    model: str | None = None,
    provider_factory: Callable[[], ResearchProviderLike | None] = get_research_provider,
    auth_dependency: Any = None,
    enabled: Callable[[], bool] = problem_mode_enabled,
    billing_gate: Callable[[str | None], tuple[str, bool, int]] | None = None,
    billing_charge: Callable[[str, int, str], None] | None = None,
) -> Any:
    """Build the ``/lab/*`` APIRouter. Every collaborator is injectable (the intake-router DI
    pattern): tests pass canned clients/providers/produce spies; the deployed app passes the real
    adapter and the live provider factory. ``enabled`` is called PER REQUEST so the env flag can
    flip without a restart (and tests can monkeypatch the env).

    Billing (S9): ``billing_gate(authorization) -> (token, skip_charge, cost)`` runs BEFORE the
    confirm build and raises the 402/503 paywall fail-closed (no engine spend on a blocked
    account); ``billing_charge(token, cost, session_id)`` runs only AFTER a successful build —
    the /generate discipline exactly (a charge failure is logged upstream, never a reason to
    withhold a built part). Both None (tests/dev) -> no metering, today's behavior."""
    from fastapi import APIRouter, Depends, Header, HTTPException

    router = APIRouter()
    deps = [Depends(auth_dependency)] if auth_dependency is not None else []

    def _gate() -> None:
        if not enabled():
            raise HTTPException(status_code=404, detail="not found")

    def _client() -> LLMClientLike:
        if client_factory is not None:
            return client_factory()
        from forge_mind.llm.provider import make_client

        return make_client()

    def _load(session_id: str) -> tuple[DesignSession, ProblemState]:
        session = store.get(session_id)
        if session is None or session.problem_state is None:
            raise HTTPException(status_code=404, detail="problem session not found")
        return session, ProblemState.model_validate(session.problem_state)

    def _save(session: DesignSession, state: ProblemState) -> None:
        session.problem_state = state.model_dump(mode="json")
        store.save(session)

    def _view(state: ProblemState, session_id: str) -> dict[str, Any]:
        """One response shape for every endpoint: the stage, the form (when answering), the card
        + ledger + assumptions (the confirm surface), the refusal, and the build record."""
        compiled = state.compiled
        return {
            "session_id": session_id,
            "stage": state.stage,
            "material": state.material,
            "form": state.form.model_dump(mode="json") if state.form is not None else None,
            "card": compiled.card if compiled is not None else "",
            "assumptions": list(compiled.assumptions) if compiled is not None else [],
            "facts": [
                {
                    "name": f.name,
                    "value": f.as_mm_or_g[0],
                    "unit": f.as_mm_or_g[1],
                    "source": f.source,
                    "citation": f.citation,
                }
                for f in (compiled.ledger.facts if compiled is not None else ())
            ],
            "refusal": state.refusal,
            "notes": [*(compiled.notes if compiled is not None else ()), *state.notes],
            "build": state.build.model_dump(mode="json") if state.build is not None else None,
        }

    @router.post("/lab/intake", dependencies=deps)
    def lab_intake(body: IntakeBody) -> dict[str, Any]:
        _gate()
        route = classify_intake(body.goal, body.context)
        if route == "spec":
            # P12/P17: a plain SPEC prompt belongs to today's /generate path, BYTE-UNTOUCHED —
            # no session, no model call, no problem-mode machinery of any kind.
            return {"route": "spec", "prompt": body.goal}
        if not body.context.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "context required: name the product (e.g. 'Cooler Master NR200') "
                    "or give the key dimensions"
                ),
            )
        if body.process not in ("print", "cnc"):
            raise HTTPException(status_code=422, detail="process must be 'print' or 'cnc'")
        session = store.create()
        try:
            state, _usages = start_problem(
                ProblemIntake(
                    goal=body.goal,
                    context=body.context,
                    materials=tuple(body.materials),
                    process=body.process,  # type: ignore[arg-type]
                ),
                client=_client(),
                model=model or BRAIN_MODEL,
                provider=provider_factory(),
            )
        except IntakeAnswerError as exc:  # e.g. an unknown material in the availability set
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _save(session, state)
        return {"route": "problem", **_view(state, session.id)}

    @router.post("/lab/answers", dependencies=deps)
    def lab_answers(body: AnswersBody) -> dict[str, Any]:
        _gate()
        session, state = _load(body.session_id)
        if state.stage != "awaiting_answers":
            raise HTTPException(
                status_code=409, detail=f"session is {state.stage!r}, not awaiting answers"
            )
        try:
            state = apply_answers(state, tuple(body.answers))
        except IntakeAnswerError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _save(session, state)
        return _view(state, session.id)

    @router.get("/lab/spec/{session_id}", dependencies=deps)
    def lab_spec(session_id: str) -> dict[str, Any]:
        _gate()
        _session, state = _load(session_id)
        return _view(state, session_id)

    @router.post("/lab/confirm", dependencies=deps)
    def lab_confirm(
        body: ConfirmBody, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _gate()
        session, state = _load(body.session_id)
        if state.stage != "spec_ready":
            raise HTTPException(
                status_code=409,
                detail=f"session is {state.stage!r}; only a spec_ready session can be confirmed",
            )
        # S9 metering: the paywall gate runs BEFORE any engine spend (402/503 fail closed);
        # problem-mode builds meter as "standard" quality (1 credit) in v1.
        token: str | None = None
        skip_charge, cost = True, 0
        if billing_gate is not None:
            token, skip_charge, cost = billing_gate(authorization)
        state = confirm_and_build(state, produce=produce)  # L3: the ONLY spend path
        _save(session, state)
        if billing_charge is not None and token is not None and not skip_charge:
            # charge only the successfully-built confirm (the /generate rule: a deduct failure
            # is the seam's problem to log — never a reason to withhold the built part)
            billing_charge(token, cost, session.id)
        return _view(state, session.id)

    return router


__all__ = ["AnswersBody", "ConfirmBody", "IntakeBody", "create_lab_router"]
