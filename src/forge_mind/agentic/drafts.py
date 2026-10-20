"""Best-of-N first drafts: the variance-killer for hard parts.

The Wave 1 spike + the Wave 2 hammer runs proved the hammer is a coin-flip on the INITIAL
massing draft -- one draft fuses into a clean single body, another draws a degenerate
boolean that never recovers. So author N drafts with DISTINCT massing strategies, build
them in parallel, and hand the loop the structurally-best one to refine. The selection
"judge" is the DETERMINISTIC validity (the failure mode is fusion / watertightness, not
aesthetics) -- which also keeps best-of-N cheap (no per-draft vision call).

Parallelism is on the sandbox BUILDS (isolated subprocesses, safe). Validation runs
SEQUENTIALLY because the in-process OCP B-rep load (import_step) is not thread-safe.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from forge_mind.agentic.author import AuthoredCode, author_initial
from forge_mind.agentic.brief import DesignBrief
from forge_mind.agentic.sandbox import SandboxLimits, SandboxResult, run_build123d_code
from forge_mind.agentic.validate import validate_authored_part
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.provider import LLMClientLike

# Distinct massing strategies, all biased toward the measured failure (FUSION). Index 0 is
# the plain prompt; the rest push different single-body construction approaches. Sampling
# adds further variance on top.
DRAFT_STRATEGIES = [
    "",
    "Build the ENTIRE part inside ONE `with BuildPart()` context so every solid auto-fuses; "
    "make each feature overlap the previous by several mm so nothing is ever left detached.",
    "Mass the part from a few LARGE overlapping primitives unioned with `+` (revolve / box / "
    "cylinder / loft), each overlapping its neighbour by several mm, THEN cut the details. "
    "Prioritize a single watertight body over fine detail in this draft.",
    "Build the main body as one solid first, then add each feature by unioning a solid that "
    "STARTS several mm INSIDE the body (never merely touching its surface) so every union fuses.",
]


@dataclass
class Draft:
    index: int
    code: str
    approach: str
    built: bool
    score: int  # 0 not built, 1 built, 2 coherent (dropped a feature), 3 coherent+features, 4 valid
    volume_mm3: float
    error: str | None


@dataclass
class DraftSelection:
    best_code: str
    best_approach: str
    drafts: list[Draft]
    cost_usd: float

    @property
    def n_built(self) -> int:
        return sum(1 for d in self.drafts if d.built)


def _draft_score(hard_ok: bool, passed: bool, features_missing: bool) -> int:
    """Rank a built draft for best-of-N selection; higher is better. Feature-completeness is weighted
    ABOVE mere coherence: a coherent draft that DROPPED a requested through-hole is a worse base to
    refine from than a coherent draft that built all its features but tripped some other gate -- so
    best-of-N stops handing the loop a holeless massing to polish (the measured "author drops holes"
    variance). The draft validator already computes ``features_missing`` at zero extra cost."""
    if passed:
        return 4  # one watertight solid, every gate green (features present by definition)
    if hard_ok and not features_missing:
        return 3  # coherent AND has its requested features; failed some other (often advisory) gate
    if hard_ok:
        return 2  # coherent but DROPPED a requested feature -- the variance we are fighting
    return 1  # built but not one watertight solid


def select_best_draft(
    brief: DesignBrief,
    n: int,
    work_dir: Path,
    *,
    client: LLMClientLike,
    author_model: str | None = None,
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
    sandbox_limits: SandboxLimits | None = None,
    part_name: str = "draft",
    strategy_suffix: str = "",
) -> DraftSelection:
    """Author n diverse drafts, build them in parallel, return the structurally-best code.

    ``strategy_suffix`` (fidelity-restart rung): extra strategy text appended to EVERY draft's
    massing strategy -- the loop passes the critique-derived restart note so all fresh drafts
    aim at the named absent features while keeping their construction diversity. Empty (the
    default) = byte-identical behavior."""
    n = max(1, n)
    limits = sandbox_limits or SandboxLimits(timeout_s=240)
    strategies = [
        f"{DRAFT_STRATEGIES[i % len(DRAFT_STRATEGIES)]}\n{strategy_suffix}".strip()
        if strategy_suffix
        else DRAFT_STRATEGIES[i % len(DRAFT_STRATEGIES)]
        for i in range(n)
    ]

    # 1. author the n drafts (sequential -- shares the cached system prompt; cheap vs the build).
    authored: list[AuthoredCode] = []
    cost = 0.0
    for strat in strategies:
        try:
            ac, usage = author_initial(
                brief,
                client=client,
                model=author_model,
                strategy=strat,
                fabrication_mode=fabrication_mode,
            )
        except Exception as exc:
            # GUARDED: one malformed authoring must not kill best-of-N -- the failed draft flows
            # through as code="" (the sandbox marks it not-built) and the others still compete.
            ac = AuthoredCode(approach=f"author failed: {str(exc)[:160]}", code="")
        else:
            cost += usage.cost_usd
        authored.append(ac)

    # 2a. BUILD each draft in PARALLEL (isolated subprocesses -- safe to parallelize).
    def _build(item: tuple[int, AuthoredCode]) -> tuple[int, AuthoredCode, SandboxResult]:
        i, ac = item
        res = run_build123d_code(
            ac.code, work_dir / f"draft_{i:02d}", part_name=part_name, limits=limits
        )
        return i, ac, res

    with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
        built = list(pool.map(_build, list(enumerate(authored))))

    # 2b. VALIDATE sequentially (OCP import_step is not thread-safe in-process).
    drafts: list[Draft] = []
    for i, ac, res in built:
        if not res.ok or res.stl_path is None or res.step_path is None:
            drafts.append(
                Draft(
                    i, ac.code, ac.approach, False, 0, 0.0, f"{res.error_type}: {res.error_message}"
                )
            )
            continue
        val = validate_authored_part(
            res.stl_path,
            res.step_path,
            brief,
            fabrication_mode=fabrication_mode,
            author_code=ac.code,  # feature-presence gate sees the draft (ADR-0117)
        )
        drafts.append(
            Draft(
                i,
                ac.code,
                ac.approach,
                True,
                _draft_score(val.hard_ok, val.passed, val.features_missing),
                val.volume_mm3,
                None,
            )
        )

    drafts.sort(key=lambda d: (d.score, d.volume_mm3), reverse=True)
    best = drafts[0]
    return DraftSelection(
        best_code=best.code, best_approach=best.approach, drafts=drafts, cost_usd=cost
    )


__all__ = ["DRAFT_STRATEGIES", "Draft", "DraftSelection", "select_best_draft"]
