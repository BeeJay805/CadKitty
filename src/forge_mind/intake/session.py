"""DesignSession -- the persistent object that holds one guided-intake conversation -- plus a
pluggable ``SessionStore`` (in-memory for tests/dev; a JSON-file store for simple persistence).

The store is an interface so production can swap in Supabase/Postgres without touching the planner
or the API: anything implementing ``SessionStore`` works.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.intake.schema import (
    Answer,
    Asset,
    CaptureContext,
    DesignBrief,
    PlannerState,
    Question,
    SessionStatus,
    SketchAsset,
)


def _now() -> datetime:
    return datetime.now(UTC)


def new_session_id() -> str:
    return uuid.uuid4().hex


class DesignSession(BaseModel):
    """One design-intake session. Mutable: the service updates it as input arrives, the planner runs,
    questions are answered, and a brief is finalized. ``updated_at`` is bumped via ``touch``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_session_id)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    status: SessionStatus = SessionStatus.COLLECTING_INPUT

    original_text_prompt: str | None = None
    uploaded_assets: list[Asset] = Field(default_factory=list)
    sketch_assets: list[SketchAsset] = Field(default_factory=list)
    # The multimodal capture bundle (speech transcript, scale, viewpoint, notes, labels) -- ADR-0025.
    # Optional: text-only / image-only sessions leave it None and behave exactly as before.
    capture: CaptureContext | None = None
    # The user's fabrication intent (ADR-0031's toggle, finally reaching the PLANNER -- the
    # board-reported gap where the brief claimed printability with the toggle OFF). "print_3d" |
    # "general"; None = not stated -> legacy FDM-default behavior, so old sessions and old
    # clients are byte-identical until the frontend sends the field.
    fabrication: str | None = None

    planner_state: PlannerState | None = None
    questions: list[Question] = Field(default_factory=list)
    answers: list[Answer] = Field(default_factory=list)
    final_design_brief: DesignBrief | None = None
    linked_generation_id: str | None = Field(
        default=None, description="Set once a generation run has been started from this session."
    )
    # Problem-mode carrier (ADR-0087 S6b): an OPAQUE, invention-owned state blob (the schema is
    # invention.problem_intake.ProblemState). The intake package never interprets it — the guided
    # flow, its states, and its planner are untouched; old sessions default to None.
    problem_state: dict[str, Any] | None = None

    def touch(self) -> None:
        self.updated_at = _now()

    def answer_for(self, question_id: str) -> Answer | None:
        # The latest answer wins (a question may be re-answered before generation).
        return next((a for a in reversed(self.answers) if a.question_id == question_id), None)

    def unanswered_required(self) -> list[Question]:
        """Required questions with no answer yet -- these block generation."""
        answered = {a.question_id for a in self.answers}
        return [q for q in self.questions if q.required and q.id not in answered]

    def has_visual_input(self) -> bool:
        return bool(self.uploaded_assets or self.sketch_assets)


class SessionStore(Protocol):
    """Persistence interface for design sessions. Implementations: in-memory, JSON file, or a DB."""

    def create(self) -> DesignSession: ...

    def get(self, session_id: str) -> DesignSession | None: ...

    def save(self, session: DesignSession) -> None: ...

    def delete(self, session_id: str) -> bool: ...


class SessionNotFoundError(KeyError):
    """Raised by the service when a session id is unknown."""


class InMemorySessionStore:
    """A dict-backed store. The default for tests and a single-process dev server."""

    def __init__(self) -> None:
        self._sessions: dict[str, DesignSession] = {}

    def create(self) -> DesignSession:
        session = DesignSession()
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> DesignSession | None:
        return self._sessions.get(session_id)

    def save(self, session: DesignSession) -> None:
        session.touch()
        self._sessions[session.id] = session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


class JSONFileSessionStore:
    """A directory of ``<id>.json`` files -- simple durable persistence with no DB. Each session is
    serialized via Pydantic, so the on-disk shape always matches the models."""

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def create(self) -> DesignSession:
        session = DesignSession()
        self.save(session)
        return session

    def get(self, session_id: str) -> DesignSession | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        return DesignSession.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, session: DesignSession) -> None:
        session.touch()
        self._path(session.id).write_text(
            json.dumps(session.model_dump(mode="json"), indent=2), encoding="utf-8"
        )

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if not path.is_file():
            return False
        path.unlink()
        return True


__all__ = [
    "DesignSession",
    "InMemorySessionStore",
    "JSONFileSessionStore",
    "SessionNotFoundError",
    "SessionStore",
    "new_session_id",
]
