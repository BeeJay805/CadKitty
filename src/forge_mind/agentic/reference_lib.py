"""Reference-model retrieval (Step 2 of the reference spine): fetch a real 3D model of the
target part from Sketchfab so the vision critic has ground truth, not the model's own guess.

RELEVANCE-GUARDED + FAIL-CLOSED. Sketchfab is artist-oriented -- a search for "crankshaft"
returns whole engines -- so a naive top-result would feed the critic a WRONG reference, which
is worse than none. We return a model only when its NAME contains the query's head noun, and
return None (text-only fallback) otherwise, on any error, or with no token.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

_API = "https://api.sketchfab.com/v3"

# Reject references heavier than this at retrieval: the matplotlib renderer has no decimation
# backend here (fast_simplification is absent), so a million-triangle artist mesh would hang
# the reference render. A part reference this side of the cap draws in seconds.
_MAX_FACES = 150_000


def sketchfab_token() -> str | None:
    """The API token from the env, falling back to the repo-root ``.env`` (the app convention)."""
    tok = os.environ.get("SKETCHFAB_API_TOKEN")
    if tok:
        return tok.strip()
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("SKETCHFAB_API_TOKEN="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                return value or None
    return None


def _get_json(url: str, token: str, *, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # fixed Sketchfab host
        data = json.load(resp)
    return data if isinstance(data, dict) else {}


def _head_noun(query: str) -> str:
    """The last alphabetic word of the query -- the part's head noun ('spur gear' -> 'gear')."""
    words = [w for w in query.lower().replace("-", " ").split() if w.isalpha()]
    return words[-1] if words else query.lower().strip()


def search_reference(query: str, *, token: str, count: int = 16) -> str | None:
    """UID of the most-liked downloadable model whose NAME contains the query's head noun --
    the relevance guard that refuses a whole-assembly model for an isolated-part query. Searches
    the full query first (specific), then the bare head noun (a verbose brief name like
    'automotive inline-4 crankshaft' returns nothing, but 'crankshaft' does). None if no relevant,
    downloadable, light-enough match (or on any API error)."""
    noun = _head_noun(query)
    queries = [query, noun] if query.strip().lower() != noun else [noun]
    for term in queries:
        uid = _search_hits(term, noun, token=token, count=count)
        if uid:
            return uid
    return None


def _search_hits(term: str, noun: str, *, token: str, count: int) -> str | None:
    """First downloadable, light-enough model for ``term`` whose name contains ``noun``."""
    params = urllib.parse.urlencode(
        {
            "type": "models",
            "q": term,
            "downloadable": "true",
            "count": str(count),
            "sort_by": "-likeCount",
        }
    )
    try:
        data = _get_json(f"{_API}/search?{params}", token)
    except (HTTPError, URLError, ValueError, TimeoutError):
        return None
    for model in data.get("results", []):
        name = str(model.get("name") or "").lower()
        faces = model.get("faceCount") or 0
        if model.get("isDownloadable") and noun in name and 0 < faces <= _MAX_FACES:
            uid = model.get("uid")
            if uid:
                return str(uid)
    return None


def download_glb(uid: str, *, token: str, out_dir: Path) -> Path | None:
    """Download the model's self-contained ``.glb`` (trimesh-loadable) to ``out_dir``; cached by
    uid. None if there is no glb flavour or the download fails."""
    dest = out_dir / f"reference_{uid}.glb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        flavours = _get_json(f"{_API}/models/{uid}/download", token)
    except (HTTPError, URLError, ValueError, TimeoutError):
        return None
    glb = flavours.get("glb")
    url = glb.get("url") if isinstance(glb, dict) else None
    if not url:
        return None  # gltf is a zip (needs extraction); only the single-file glb is used here
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # The url is a short-lived signed CDN link (no auth header).
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (HTTPError, URLError, OSError, TimeoutError):
        dest.unlink(missing_ok=True)
        return None
    return dest


def fetch_reference_mesh(query: str, out_dir: Path, *, token: str | None = None) -> Path | None:
    """A real reference mesh (.glb) for ``query`` to feed the critic, or None (text-only
    fallback). Fail-closed + relevance-guarded end to end: no token, no relevant match, or a
    download failure all yield None -- a wrong reference is worse than none."""
    token = token or sketchfab_token()
    if not token:
        return None
    uid = search_reference(query, token=token)
    if uid is None:
        return None
    return download_glb(uid, token=token, out_dir=out_dir)


__all__ = ["download_glb", "fetch_reference_mesh", "search_reference", "sketchfab_token"]
