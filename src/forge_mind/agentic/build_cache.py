"""Content-hash build cache (roadmap task #13a): skip the LLM when an identical part was already
authored.

A part's output is deterministic in (prompt + material + draft/iteration settings + models), so a
second request with the same inputs can reuse the prior build at ZERO LLM cost -- only the
author/critic/judge calls are skipped. Companion to the prompt caching already in ``llm/client.py``
(which cuts the cost of a MISS); this cuts a repeat to free.

OPT-IN via the ``FORGE_BUILD_CACHE`` env var (a cache directory). OFF by default, so default
behavior and the whole test suite stay byte-identical (no env -> no cache path is ever taken). A hit
restores the geometry VERDICT (passed / hard_ok / body_count / ...), not the full ValidationReport
(``report`` is None on a restored result) -- downstream coherence checks read the booleans, not the
report. Best-effort throughout: a cache read/write failure never fails a build.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge_mind.agentic.loop import LoopResult
    from forge_mind.fabrication import FabricationMode
    from forge_mind.materials import Material

# Bump when author logic / cheatsheet / validator output changes, so stale builds are not reused.
# v2: the key now folds in the cheatsheet text + the helper-module source hash, so an edit to the
# authoring guidance or the deterministic helpers can no longer serve a stale (wrong) part.
# v3 (ADR-0035 B8): the manifest now PERSISTS the brief. A v2 hit restored brief=None, which made
# produce_part SKIP the acceptance judge and _disposition fabricate "rejected" for a previously
# SHIPPED identical part -- a cache hit lied. With the brief restored, the judge re-runs on a hit
# (a few cents; the geometry stays free) and the disposition is real again.
# v4 (2026-07-27): hole-repaired parts used to persist the PRE-repair critique's approval. A v3
# hit returns that stored `critic_approved` before the fresh re-critique can run, so the stale
# approval this fix removes would keep being served (review-caught). Geometry re-authors once.
# v5 (2026-07-27): the material prompt grew a TARGET VOLUME line, and `_author_context_hash`
# does not cover the prompt-building modules -- so a v4 entry authored under the same flag would
# be served without the new line ever running (review-caught). One re-author, then free again.
# v6: cache identity includes the validator source fingerprint and every restored artifact's
# SHA-256. A changed validator or modified file can no longer resurrect a recorded verdict.
_CACHE_VERSION = "6"
_ARTIFACTS = (
    ("final.stl", "stl"),
    ("final.step", "step"),
    ("final.png", "png"),
    ("final.py", "py"),
)

# Sibling helper modules whose source shapes the authored geometry. Read by path (not imported) so
# the build_cache module stays light and never drags in build123d.
_SOURCE_DIR = Path(__file__).resolve().parent
_SOURCE_SIBLINGS = ("helpers.py", "standards_helpers.py")


def _author_context_hash() -> str:
    """Hash of everything that steers authoring besides the prompt: the rendered cheatsheet text
    plus the source of the deterministic helper modules. FAIL-OPEN: any read/import error yields a
    stable sentinel so a missing input never produces a NEW (false-miss-forcing) key per call -- the
    bumped _CACHE_VERSION still guarantees old entries are not reused."""
    h = hashlib.sha256()
    try:
        from forge_mind.agentic.cheatsheet import author_skills_block

        h.update(author_skills_block().encode("utf-8"))
    except Exception:
        h.update(b"<cheatsheet-unavailable>")
    here = _SOURCE_DIR
    for name in _SOURCE_SIBLINGS:
        try:
            h.update(b"\x00")
            h.update(name.encode("utf-8"))
            h.update((here / name).read_bytes())
        except Exception:
            h.update(b"<source-unavailable>")
    return h.hexdigest()


def cache_dir() -> Path | None:
    """The cache directory from ``FORGE_BUILD_CACHE``, or None when caching is disabled (default)."""
    d = os.environ.get("FORGE_BUILD_CACHE")
    return Path(d) if d else None


def enabled() -> bool:
    return cache_dir() is not None


def _finish_mode() -> str:
    """'finish' when the designed-finish pass is armed, else 'clean' -- a cache key input."""
    from forge_mind.agentic.detail import finish_enabled

    return "finish" if finish_enabled() else "clean"


def _spec_dims_mode() -> bool:
    """The spec_dims prompt flag, read through its OWN predicate (see compute_key)."""
    from forge_mind.agentic.spec_dim_prompt import spec_dim_prompt_enabled

    return spec_dim_prompt_enabled()


def _material_prompt_mode() -> bool:
    """The material prompt flag, read through its OWN predicate (see compute_key)."""
    from forge_mind.agentic.spec_dim_prompt import material_prompt_enabled

    return material_prompt_enabled()


def compute_key(
    prompt: str,
    *,
    material: Material | None,
    n_drafts: int,
    max_iterations: int,
    fabrication_mode: FabricationMode,
    author_model: str | None,
    critic_model: str | None,
    brief_model: str | None,
    target_mass_g: float | None = None,
) -> str:
    """A stable sha256 over every input that affects the authored part.

    ``target_mass_g`` is the CALLER-SUPPLIED brief's mass target, when the caller pre-researched
    the brief (``produce_part`` does). Two requests can share a prompt and every setting yet
    carry different masses -- via ``extra_context`` at intake, say -- and the lookup returns
    before the brief is ever rendered, so without this the second would be served the first
    one's geometry (review-caught). None when the loop researches its own brief: the prompt is
    then the only mass evidence and is already in the key."""
    from forge_mind.validation_provenance import validation_fingerprint

    payload = {
        "v": _CACHE_VERSION,
        "prompt": prompt,
        "target_mass_g": target_mass_g,
        "material": material.id if material is not None else "none",
        "n_drafts": n_drafts,
        "max_iterations": max_iterations,
        "fab": str(fabrication_mode),
        "author": author_model or "",
        "critic": critic_model or "",
        "brief": brief_model or "",
        "ctx": _author_context_hash(),
        "validator": validation_fingerprint(),
        # FORGE_FINISH partitions the cache (review-caught): without it, a clean entry kept
        # serving after the flag turned on (silently skipping finish), and a finished entry
        # kept serving after it turned off. Env-read here so the key tracks the live mode.
        "finish": _finish_mode(),
        # ADR-0133 prompt-shaping flags partition the cache too (review-caught: a cache hit
        # returns BEFORE authoring, so a flag flip would serve off-mode artifacts). One-call
        # is already captured implicitly via "ctx" (author_skills_block() honors its flag);
        # these two shape the prompt OUTSIDE the skills block, so they ride explicitly.
        "edge_finish": os.environ.get("FORGE_EDGE_FINISH") == "1",
        "sweep_idiom": os.environ.get("FORGE_SWEEP_IDIOM") == "1",
        # Same rule, same reason, for the 2026-07-27 prompt-shaping flags (review-caught: the
        # spec_dims flag shipped at 50e134c WITHOUT its partition, so an A/B could have been
        # served pre-arm geometry from cache and scored as if the arm ran). These call the
        # flags' OWN predicates rather than re-testing == "1" (second review-catch: the helpers
        # accept true/yes/whitespace, so a hand-typed FORGE_SPEC_DIM_PROMPT=true would arm the
        # prompt while leaving the key unpartitioned -- exactly the bug this line prevents).
        "spec_dims": _spec_dims_mode(),
        "material_prompt": _material_prompt_mode(),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load(key: str, work_dir: Path) -> LoopResult | None:
    """Return a reconstructed LoopResult from the cache (copying artifacts into ``work_dir``), or
    None on a miss / disabled cache / any read error."""
    cdir = cache_dir()
    if cdir is None:
        return None
    entry = cdir / key
    manifest = entry / "manifest.json"
    if not manifest.exists():
        return None
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        from forge_mind.validation_provenance import file_sha256, validation_fingerprint

        if meta.get("version") != _CACHE_VERSION:
            return None
        if meta.get("validation_fingerprint") != validation_fingerprint():
            return None
        hashes = meta.get("artifact_sha256") or {}
        allowed = {fname for fname, _kind in _ARTIFACTS}
        if set(hashes) - allowed:
            return None
        if "final.stl" not in hashes:
            return None
        for fname, expected in hashes.items():
            src = entry / fname
            if not src.exists() or file_sha256(src) != expected:
                return None
        work_dir.mkdir(parents=True, exist_ok=True)
        copied: dict[str, Path] = {}
        for fname, kind in _ARTIFACTS:
            src = entry / fname
            if src.exists():
                dst = work_dir / fname
                shutil.copy2(src, dst)
                copied[kind] = dst

        from forge_mind.agentic.brief import DesignBrief
        from forge_mind.agentic.loop import LoopResult
        from forge_mind.agentic.validate import AgenticValidation

        v = meta.get("validation")
        validation = (
            AgenticValidation(
                passed=bool(v["passed"]),
                hard_ok=bool(v["hard_ok"]),
                report=None,
                body_count=int(v["body_count"]),
                watertight=bool(v["watertight"]),
                volume_mm3=float(v["volume_mm3"]),
            )
            if v
            else None
        )
        # Restore the BRIEF (v3, ADR-0035 B8): without it the acceptance judge is skipped on a hit
        # and the disposition is fabricated as "rejected" for a part that originally SHIPPED.
        b = meta.get("brief")
        brief = DesignBrief.model_validate(b) if b else None
        code_path = copied.get("py")
        return LoopResult(
            prompt=meta["prompt"],
            brief=brief,
            final_code=code_path.read_text(encoding="utf-8") if code_path else None,
            final_stl=copied.get("stl"),
            final_step=copied.get("step"),
            final_render=copied.get("png"),
            validation=validation,
            critique=None,
            geometry_passed=bool(meta["geometry_passed"]),
            critic_approved=bool(meta["critic_approved"]),
            stopped_reason="cached",
            iterations=[],
            total_cost_usd=0.0,
            total_duration_s=0.0,
        )
    except Exception:
        return None


def save(key: str, result: LoopResult) -> None:
    """Store a coherent build's artifacts + verdict under ``key``. No-op when caching is disabled or
    the build produced no STL. Best-effort: a write failure is swallowed."""
    cdir = cache_dir()
    if cdir is None or result.final_stl is None:
        return
    entry = cdir / key
    try:
        from forge_mind.validation_provenance import file_sha256, validation_fingerprint

        entry.mkdir(parents=True, exist_ok=True)
        for fname, _kind in _ARTIFACTS:
            stale = entry / fname
            if stale.exists():
                stale.unlink()
        for src, fname in (
            (result.final_stl, "final.stl"),
            (result.final_step, "final.step"),
            (result.final_render, "final.png"),
        ):
            if src is not None and Path(src).exists():
                shutil.copy2(src, entry / fname)
        if result.final_code:
            (entry / "final.py").write_text(result.final_code, encoding="utf-8")
        artifact_sha256 = {
            fname: file_sha256(entry / fname)
            for fname, _kind in _ARTIFACTS
            if (entry / fname).exists()
        }
        v = result.validation
        meta = {
            "version": _CACHE_VERSION,
            "validation_fingerprint": validation_fingerprint(),
            "artifact_sha256": artifact_sha256,
            "prompt": result.prompt,
            "geometry_passed": result.geometry_passed,
            "critic_approved": result.critic_approved,
            # Persist the brief (v3, ADR-0035 B8) so a cache hit can re-run the acceptance judge
            # against the SAME brief and report an honest disposition.
            "brief": (result.brief.model_dump(mode="json") if result.brief is not None else None),
            "validation": (
                {
                    "passed": v.passed,
                    "hard_ok": v.hard_ok,
                    "body_count": v.body_count,
                    "watertight": v.watertight,
                    "volume_mm3": v.volume_mm3,
                }
                if v is not None
                else None
            ),
        }
        (entry / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        return


__all__ = ["cache_dir", "compute_key", "enabled", "load", "save"]
