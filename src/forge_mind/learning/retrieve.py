"""Retrieve similar prior generations to seed planning (the USE side of the learning loop).

Feature-based for now: a coarse prompt-keyword ILIKE over the user's own ACCEPTED generations via
Supabase REST. Swap to pgvector cosine similarity once the ``embedding`` column is populated
(DATA_MODEL.md) -- the call signature stays the same.

Read-only + best-effort: returns ``[]`` on any failure. RLS scopes every query to the caller, so
this can ONLY ever return the user's OWN parts -- no cross-user retrieval (the privacy rule in
LEARNING_LOOP.md).

NOT yet wired into the author. Feeding retrieved examples into the LLM prompt should land WITH the
benchmark so its effect on the clean-pass rate is MEASURED, not assumed.
"""

from __future__ import annotations

from typing import Any

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "with",
        "for",
        "and",
        "of",
        "that",
        "make",
        "me",
        "please",
        "to",
        "in",
        "my",
        "this",
        "some",
        "part",
        "design",
        "create",
        "build",
        "small",
        "sturdy",
        "simple",
    }
)


def _first_keyword(prompt: str) -> str:
    """A coarse content word for an ILIKE match (placeholder until embeddings land)."""
    for word in prompt.lower().split():
        token = "".join(c for c in word if c.isalnum())
        if len(token) >= 3 and token not in _STOPWORDS:
            return token
    return ""


def similar_accepted(
    base_url: str, token: str, anon_key: str, prompt: str, *, limit: int = 3
) -> list[dict[str, Any]]:
    """Up to ``limit`` of the caller's prior OK generations whose prompt overlaps ``prompt``.

    Best-effort: ``[]`` on missing config, no keyword, or any error. The user's token + RLS mean the
    result is always the caller's own accepted parts (privacy-safe). Each row carries prompt / spec /
    validation / files so a planner can reuse proven dimensions or pass the part as an example.
    """
    keyword = _first_keyword(prompt)
    if not base_url or not token or not keyword:
        return []
    import httpx

    try:
        r = httpx.get(
            f"{base_url}/rest/v1/generations",
            params={
                "select": "id,prompt,spec,validation,files,created_at",
                "status": "eq.ok",
                "prompt": f"ilike.*{keyword}*",
                "order": "created_at.desc",
                "limit": str(limit),
            },
            headers={"Authorization": f"Bearer {token}", "apikey": anon_key},
            timeout=10.0,
        )
    except httpx.HTTPError:
        return []
    if r.status_code != 200:
        return []
    try:
        rows = r.json()
    except ValueError:
        return []
    return rows if isinstance(rows, list) else []


__all__ = ["similar_accepted"]
