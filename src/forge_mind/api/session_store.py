"""A Supabase-backed SessionStore so guided-intake sessions survive restarts and work across backend
instances (the in-memory default is single-process only). The intake planner/API are untouched --
this just implements the ``SessionStore`` protocol and is injected into ``IntakeService(store=...)``.

Sessions are stored in the ``design_sessions`` table as a JSON blob (the DesignSession pydantic dump),
managed by the service role (planning is open, so sessions may be anonymous; the table's RLS denies
everyone except the service role).
"""

from __future__ import annotations

from typing import Any

from forge_mind.intake.session import DesignSession

_TABLE = "design_sessions"


class SupabaseSessionStore:
    """SessionStore over the ``design_sessions`` table. ``client`` is a service-role Supabase client."""

    def __init__(self, client: Any) -> None:
        self._sb = client

    def create(self) -> DesignSession:
        session = DesignSession()
        self._upsert(session)
        return session

    def get(self, session_id: str) -> DesignSession | None:
        res = self._sb.table(_TABLE).select("data").eq("id", session_id).limit(1).execute()
        rows = res.data or []
        if not rows:
            return None
        return DesignSession.model_validate(rows[0]["data"])

    def save(self, session: DesignSession) -> None:
        session.touch()
        self._upsert(session)

    def delete(self, session_id: str) -> bool:
        res = self._sb.table(_TABLE).delete().eq("id", session_id).execute()
        return bool(res.data)

    def _upsert(self, session: DesignSession) -> None:
        self._sb.table(_TABLE).upsert(
            {
                "id": session.id,
                "data": session.model_dump(mode="json"),
                "updated_at": session.updated_at.isoformat(),
            }
        ).execute()


__all__ = ["SupabaseSessionStore"]
