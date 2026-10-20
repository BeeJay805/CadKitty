"""CAD-Coder generation route (ADR-0112): a flag-gated LOCAL first-pass in the live agentic loop.

Default OFF. ``FORGE_GENERATOR=cadcoder`` swaps ONLY the initial construction: instead of the
build123d author writing the first draft, the local CAD-Coder model writes CadQuery, which the exec
bridge runs to the same solid the rest of the loop consumes. If CAD-Coder errors (server down, empty
output) the loop falls through to the build123d author unchanged -- the flag never breaks the pipeline.

Deliberately first-pass only: CAD-Coder is a per-part generator (no repair, no assembly). So every
REVISION iteration stays the build123d author -- the frontier repair lane, exactly as designed. The
single exec site in the loop picks its child harness per candidate via :func:`runner_for`: a CadQuery
first draft builds through ``_cadquery_runner``, a build123d revision through the default runner.

Flag-off is byte-identical: ``cadcoder_enabled()`` is False, so neither helper is ever called and
``runner_for`` is never consulted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from forge_mind.agentic.sandbox import _RUNNER_MODULE  # the default build123d child harness
from forge_mind.llm.cost import LLMUsage

_CADQUERY_RUNNER = "forge_mind.agentic._cadquery_runner"
_GENERATOR_ENV = "FORGE_GENERATOR"


def cadcoder_enabled() -> bool:
    """True when the operator selected the local CAD-Coder first-pass. Default OFF."""
    return os.environ.get(_GENERATOR_ENV, "").strip().lower() == "cadcoder"


def cadcoder_initial(prompt: str) -> str | None:
    """Generate the initial part as CadQuery via the local CODER model (ADR-0114: qwen3-coder, a
    strong general coder grounded by the coordinate/idiom block -- measured 10/10 on a block+hole vs
    the CAD-Coder specialist's 0/10). Set ``FORGE_CODER_MODEL=hf-cad-coder`` style overrides to swap.

    Returns the CadQuery source, or ``None`` on ANY failure (server unreachable, empty output) so the
    caller falls through to the build123d author -- the route must never break the loop.
    """
    try:
        from forge_mind.llm.cadcoder_client import generate_cadquery_coder

        code, _usage = generate_cadquery_coder(prompt)
        return code
    except Exception:
        return None


def is_cadquery(code: str) -> bool:
    """True if this candidate is CadQuery source (vs build123d)."""
    return "import cadquery" in code or "cadquery as cq" in code


def _draft_count() -> int:
    """How many CAD-Coder samples the initial pass draws (local + fast, so default > 1)."""
    try:
        return max(1, int(os.environ.get("FORGE_CADCODER_DRAFTS", "3")))
    except ValueError:
        return 3


def _score_draft(result: Any, sc_spec: dict[str, Any] | None) -> int:
    """Deterministic draft score from MEASURED geometry (the autopsy fix: a draft whose hole
    silently vanished -- the construction-circle trap -- builds fine but scores below one whose
    bores match the brief). built=1; every expected hole group fully matched +2; partially +1;
    envelope within 15% on all axes +1."""
    if not result.ok:
        return 0
    score = 1
    sc = result.selfcheck or {}
    holes = sc.get("holes") or []
    if holes:
        matched = sum(int(h.get("matched") or 0) for h in holes)
        wanted = sum(int(h.get("want") or 0) for h in holes)
        if wanted and matched >= wanted:
            score += 2
        elif matched:
            score += 1
    env = sc.get("envelope") or {}
    pcts = [p for p in (env.get("delta_pct") or []) if p is not None]
    if pcts and all(abs(p) <= 15.0 for p in pcts):
        score += 1
    return score


def cadcoder_best_initial(
    prompt: str,
    work_dir: Path,
    *,
    sc_spec: dict[str, Any] | None,
    sandbox_limits: Any = None,
    n: int | None = None,
) -> str | None:
    """Best-of-N CAD-Coder first pass (ADR-0114): draw N free local samples, BUILD each in the
    sandbox, score against the brief's measured targets, keep the best. Attacks the measured #1
    failure -- a draft that executes but silently lost a requested feature -- with selection
    instead of prompting. Returns the winning CadQuery source, or None if nothing built."""
    from forge_mind.agentic.sandbox import run_build123d_code

    n = n or _draft_count()
    perfect = 4 if sc_spec else 1
    best_code: str | None = None
    best_score = 0
    for i in range(n):
        code = cadcoder_initial(prompt)
        if code is None:
            continue
        result = run_build123d_code(
            code,
            work_dir / f"cadcoder_draft_{i}",
            part_name="draft",
            limits=sandbox_limits,
            selfcheck=sc_spec,
            runner_module=_CADQUERY_RUNNER,
        )
        score = _score_draft(result, sc_spec)
        if score > best_score:
            best_code, best_score = code, score
        if best_score >= perfect:
            break  # a fully-matching draft: no need to keep sampling
    return best_code


def runner_for(code: str) -> str:
    """Pick the sandbox child harness for one candidate. CadQuery source (CAD-Coder's first pass)
    builds through the CadQuery bridge; everything else (the build123d author's drafts and every
    revision) builds through the default runner -- so the build123d path is untouched."""
    return _CADQUERY_RUNNER if is_cadquery(code) else _RUNNER_MODULE


def revise_dispatch(
    brief: Any,
    previous_code: str,
    feedback: str,
    *,
    client: Any = None,
    model: str | None = None,
    max_tokens: int | None = None,
    fabrication_mode: Any = None,
    image_png: bytes | None = None,
    image_has_reference: bool = False,
) -> tuple[Any, LLMUsage]:
    """Revision dispatcher (ADR-0114): repair in the SAME language the part is written in.

    A CadQuery part is repaired by the CadQuery repair model (``repair_cadquery``) -- handing CadQuery
    to the build123d author is the language mismatch that made every local revision fail to build. A
    build123d part still goes to ``author_revision`` unchanged. Signature matches ``author_revision``
    so it drops into the loop's ``_retry_once`` calls; returns ``(AuthoredCode, usage)`` either way."""
    if is_cadquery(previous_code):
        from forge_mind.agentic.author import AuthoredCode
        from forge_mind.llm.cadcoder_client import repair_cadquery

        object_name = getattr(brief, "object_name", "the part")
        code, usage = repair_cadquery(
            previous_code, feedback, object_name, max_tokens=max_tokens or 2048
        )
        return AuthoredCode(approach="cadquery-repair", code=code), usage

    from forge_mind.agentic.author import author_revision
    from forge_mind.fabrication import FabricationMode

    return author_revision(
        brief,
        previous_code,
        feedback,
        client=client,
        model=model,
        max_tokens=max_tokens,
        fabrication_mode=fabrication_mode or FabricationMode.GENERAL,
        image_png=image_png,
        image_has_reference=image_has_reference,
    )


__all__ = ["cadcoder_enabled", "cadcoder_initial", "is_cadquery", "revise_dispatch", "runner_for"]
