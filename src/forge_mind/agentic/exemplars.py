"""Exemplar memory: few-shot reference builds harvested ONLY from validator-PASSING
generations (lever e of the breadth objective).

The author's distribution problem is breadth: on long-tail parts it has no anchor for
what a WORKING build123d script of "a thing like this" looks like. This module gives
it one: a small library of validated builds, retrieved by request similarity and
appended to the author's context as REFERENCE (adapt, never copy).

Honesty + hygiene rules:
  * Harvest ONLY passing builds (the validator's verdict, recorded by the bench/job
    metadata -- never self-claimed here).
  * BENCHMARK CONTAMINATION GUARD: anything matching a frozen-benchmark prompt
    (arbitrary_v1 base prompts or paraphrases) is REFUSED at harvest time -- an
    exemplar library that contains the test set is memorization, not breadth.
  * OPT-IN via ``FORGE_AUTHOR_EXEMPLARS`` = the library DIRECTORY (falsy = off, the
    deployed default -- byte-identical author prompts). Same path semantics as
    FORGE_BUILD_CACHE: a directory, never a boolean string.
  * Retrieval is deterministic (difflib ratio, stable tie-break) and failure here is
    never fatal to authoring (best-effort: no library -> no block).
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_ENV_DIR = "FORGE_AUTHOR_EXEMPLARS"
_MAX_CODE_CHARS = 6000  # an exemplar is a reference, not a payload; oversized scripts skipped
_DEFAULT_K = 2


class ExemplarRecord(BaseModel):
    """One validated build: the request it answered and the script that passed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_name: str
    prompt: str
    code: str
    source: str  # where it was harvested from (job dir / bench run)
    harvested_at: str


def exemplar_dir() -> Path | None:
    """The library directory from ``FORGE_AUTHOR_EXEMPLARS``, or None (disabled)."""
    d = os.environ.get(_ENV_DIR)
    return Path(d) if d else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def benchmark_deny_list(repo_root: Path | None = None) -> set[str]:
    """Every frozen-benchmark prompt variant (normalized) that may NEVER enter the
    library. Reads benchmarks/arbitrary_v1 when present; absence = empty set (a
    harvest outside the repo still works, but in-repo harvests are always guarded)."""
    root = repo_root or Path.cwd()
    bench = root / "benchmarks" / "arbitrary_v1" / "arbitrary_v1.json"
    deny: set[str] = set()
    if not bench.exists():
        return deny
    try:
        data = json.loads(bench.read_text(encoding="utf-8"))
        for p in data.get("prompts", []):
            deny.add(_norm(str(p.get("prompt", ""))))
            for para in p.get("paraphrases", []) or []:
                deny.add(_norm(str(para)))
    except Exception:
        return deny
    deny.discard("")
    return deny


def _is_denied(prompt: str, deny: set[str]) -> bool:
    n = _norm(prompt)
    if not n:
        return True
    if n in deny:
        return True
    # Substring containment either way catches light rewording around a bench prompt.
    return any(d in n or n in d for d in deny)


def store(record: ExemplarRecord, lib_dir: Path) -> Path:
    """Write one exemplar (content-addressed: re-harvesting the same build is a no-op)."""
    lib_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((record.prompt + record.code).encode("utf-8")).hexdigest()[:16]
    path = lib_dir / f"exemplar_{digest}.json"
    path.write_text(json.dumps(record.model_dump(), indent=2), encoding="utf-8")
    return path


def load_all(lib_dir: Path) -> list[ExemplarRecord]:
    out: list[ExemplarRecord] = []
    if not lib_dir.is_dir():
        return out
    for p in sorted(lib_dir.glob("exemplar_*.json")):
        try:
            out.append(ExemplarRecord.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue  # one bad file never breaks retrieval
    return out


def harvest_bench_run(
    run_dir: Path,
    lib_dir: Path,
    *,
    deny: set[str] | None = None,
) -> int:
    """Harvest every PASSING variant of a bench-run directory (results.json + per-
    variant job dirs) into the library. Returns how many exemplars were stored.

    The deny list defaults to the frozen benchmark's own prompts -- harvesting a
    bench run OF the frozen benchmark therefore stores NOTHING (the guard working),
    while seed runs of disjoint prompts harvest normally."""
    deny = benchmark_deny_list() if deny is None else deny
    results = run_dir / "results.json"
    if not results.exists():
        return 0
    try:
        data = json.loads(results.read_text(encoding="utf-8"))
    except Exception:
        return 0
    stored = 0
    for rec in data.get("results", []):
        if not rec.get("passed"):
            continue
        prompt = str(rec.get("prompt", ""))
        if _is_denied(prompt, deny):
            continue
        label = str(rec.get("label", ""))
        variant = rec.get("variant", 0)
        job = run_dir / f"{label}_v{variant}"
        code_path = job / "program.py"
        if not code_path.exists():
            continue
        code = code_path.read_text(encoding="utf-8", errors="replace")
        if not code.strip() or len(code) > _MAX_CODE_CHARS:
            continue
        object_name = label.replace("_", " ")
        brief_path = job / "brief.json"
        if brief_path.exists():
            with contextlib.suppress(Exception):
                object_name = str(
                    json.loads(brief_path.read_text(encoding="utf-8")).get(
                        "object_name", object_name
                    )
                )
        store(
            ExemplarRecord(
                object_name=object_name,
                prompt=prompt,
                code=code,
                source=str(job),
                harvested_at=datetime.now(UTC).isoformat(),
            ),
            lib_dir,
        )
        stored += 1
    return stored


def retrieve(query: str, lib_dir: Path, k: int = _DEFAULT_K) -> list[ExemplarRecord]:
    """The k most similar exemplars to ``query`` (deterministic: difflib ratio over
    normalized object-name + prompt, ties broken by object_name)."""
    nq = _norm(query)
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, nq, _norm(r.object_name + " " + r.prompt)).ratio(), r)
            for r in load_all(lib_dir)
        ),
        key=lambda t: (-t[0], t[1].object_name),
    )
    return [r for score, r in scored[:k] if score > 0.1]


def exemplars_block(query: str) -> str:
    """The author-context block for ``query``, or '' when the library is disabled,
    empty, or anything fails (best-effort -- exemplars never break authoring)."""
    try:
        lib = exemplar_dir()
        if lib is None:
            return ""
        records = retrieve(query, lib)
        if not records:
            return ""
        parts = [
            "\n\nREFERENCE BUILDS -- validated scripts of SIMILAR parts. Adapt the"
            " construction patterns to THIS brief; never copy dimensions or names:"
        ]
        for i, r in enumerate(records, 1):
            parts.append(f"\n--- reference {i}: {r.object_name} ---\n```python\n{r.code}\n```")
        return "\n".join(parts)
    except Exception:
        return ""


__all__ = [
    "ExemplarRecord",
    "benchmark_deny_list",
    "exemplar_dir",
    "exemplars_block",
    "harvest_bench_run",
    "load_all",
    "retrieve",
    "store",
]
