"""Metric-steered best-of-N at the REVISION step (ADR-0070, OPT-IN FORGE_REVISION_BEST_OF_N=k).

The single-candidate greedy revision plateaus ~4 iterations. This authors k candidate revisions
from the SAME base+feedback, builds them in parallel, and keeps the best by a MULTI-OBJECTIVE
deterministic rank -- validity tier FIRST (must be a coherent solid: the Goodhart guard, so the
metric can never promote a broken part), then proximity to the reference's shape signature among
equal-tier candidates. No critic runs on the losers (k-1 vision calls saved); the winner flows into
the loop's normal validate -> critique -> keep-best machinery.

k=1 (the default) authors exactly one candidate = byte-identical to the current single revision.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from forge_mind.agentic.author import author_revision
from forge_mind.agentic.brief import DesignBrief
from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code
from forge_mind.agentic.shape_signature import signature_deltas
from forge_mind.agentic.validate import validate_authored_part
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.provider import LLMClientLike
from forge_mind.materials import Material


def revision_best_of_n() -> int:
    """k for best-of-N revision (env FORGE_REVISION_BEST_OF_N). 1 (default) = the current single
    revision, byte-identical. Clamped to >=1."""
    raw = os.environ.get("FORGE_REVISION_BEST_OF_N")
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    return 1


def _validity_tier(hard_ok: bool, passed: bool, features_missing: bool) -> int:
    """Higher = better. Mirrors drafts._draft_score: a coherent solid with all its features and
    every gate green outranks one that dropped a feature, which outranks a non-solid. This is the
    FIRST rank key, so the signature can only ever break a tie between EQUALLY-valid candidates --
    it can never promote a broken part (the no-Goodhart guard)."""
    if passed:
        return 3
    if hard_ok and not features_missing:
        return 2
    if hard_ok:
        return 1
    return 0


# Matches the validator's own emission ("N unsupported overhang region(s), total X mm^2" --
# printability.py); parser and emitter live in this repo, pinned together by a test.
_OVERHANG_TOTAL = re.compile(r"overhang region\(s\), total (\d+(?:\.\d+)?) mm\^2")


def advisory_overhang_mm2(issues: list[str]) -> float:
    """Measured unsupported-overhang area from a validation's issue list (0.0 when clean).

    The attractor-escape probe (2026-07-18) showed salted candidates can reduce this number
    (313 -> 253 mm^2) while every rank the loop had was blind to it -- equal-tier candidates
    tied and the winner was arbitrary. This makes the measured advisory a tiebreak signal.
    """
    for issue in issues:
        m = _OVERHANG_TOTAL.search(issue)
        if m:
            return float(m.group(1))
    return 0.0


def _signature_distance(stl: Path, reference_stl: Path | None) -> float:
    """Sum of the shape-signature deltas vs the reference (lower = closer to the real form). 0.0
    when there is no reference (signature does not discriminate, so tier + order decide); +inf when
    a reference exists but the signature is uncomputable, so it never outranks a measurable one."""
    if reference_stl is None:
        return 0.0
    d = signature_deltas(stl, reference_stl)
    if not d:
        return float("inf")
    return float(sum(d.values()))


def best_revision(
    brief: DesignBrief,
    base: str,
    feedback: str,
    *,
    k: int,
    client: LLMClientLike,
    author_model: str | None,
    fabrication_mode: FabricationMode,
    image_png: bytes | None,
    image_has_reference: bool,
    reference_stl: Path | None,
    material: Material | None,
    sandbox_limits: SandboxLimits | None,
    work_dir: Path,
    part_name: str,
) -> tuple[str, float, int]:
    """Author k revisions, build them in parallel, return ``(best_code, author_cost, n_built)``.
    The winner is the highest ``(validity_tier, -signature_distance)``. Falls back to ``base`` if
    none of the k build (no regression: base was already the working candidate)."""
    coded: list[tuple[int, str]] = []
    total_cost = 0.0
    n = max(k, 1)
    for i in range(n):
        # Candidate diversification (measured 2026-07-18): the local OllamaClient decodes greedily
        # (temperature 0), so re-rolling the IDENTICAL prompt returns near-identical completions --
        # k>1 burned k GPU calls for one distinct candidate. A per-candidate suffix diversifies at
        # the PROMPT level (works at any temperature, cloud and local alike); k=1 stays
        # byte-identical to the single-revision path.
        salted = feedback
        if n > 1:
            salted = (
                f"{feedback}\n\nCANDIDATE {i + 1} of {n}: take a distinctly different valid "
                "repair approach from the other candidates (different construction order, "
                "different primitive choice, or a different way to satisfy the failing check)."
            )
        try:
            authored, usage = author_revision(
                brief,
                base,
                salted,
                client=client,
                model=author_model,
                fabrication_mode=fabrication_mode,
                image_png=image_png,
                image_has_reference=image_has_reference,
            )
        except Exception:
            continue
        total_cost += usage.cost_usd
        coded.append((len(coded), authored.code))
    if not coded:
        return base, total_cost, 0

    limits = sandbox_limits or SandboxLimits(timeout_s=240)

    def _build(item: tuple[int, str]) -> tuple[str, Any]:
        i, code = item
        res = run_build123d_code(
            code, work_dir / f"cand_{i:02d}", part_name=part_name, limits=limits
        )
        return code, res

    with ThreadPoolExecutor(max_workers=min(len(coded), 4)) as pool:
        built = list(pool.map(_build, coded))

    best_code, n_built = base, 0
    best_key: tuple[int, float, float] = (-1, float("-inf"), float("-inf"))
    for code, res in built:  # validate SEQUENTIALLY (OCP import_step is not thread-safe)
        if not res.ok or res.stl_path is None or res.step_path is None:
            continue
        n_built += 1
        val = validate_authored_part(
            res.stl_path,
            res.step_path,
            brief,
            fabrication_mode=fabrication_mode,
            material=material,
            author_code=code,  # feature-presence gate sees the candidate (ADR-0117)
        )
        # Rank: validity tier FIRST (no-Goodhart), then FEWER measured overhang mm^2 among
        # equal tiers (the attractor-escape probe's selection gap), then shape signature.
        key = (
            _validity_tier(val.hard_ok, val.passed, val.features_missing),
            -advisory_overhang_mm2(val.issues),
            -_signature_distance(res.stl_path, reference_stl),
        )
        if key > best_key:
            best_key, best_code = key, code
    return best_code, total_cost, n_built


__all__ = ["advisory_overhang_mm2", "best_revision", "revision_best_of_n"]
