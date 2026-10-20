"""The agentic iterate loop: research -> author -> sandbox -> validate -> render ->
vision critique -> revise, until the part is geometrically valid AND reads as the
intended object, or a hard iteration / budget cap is hit (honest "stopped" reasons).

Order per iteration (the deterministic gate runs BEFORE any vision spend):
  1. run the authored code in the sandbox
  2. if it did not build -> repair from the traceback (cheap, no render/vision)
  3. validate the STL (mesh / features / printability + functional gate)
  4. render the part (always -- for the log + the acceptance judge)
  5. if the part is watertight + ONE solid -> ask the vision critic; ACCEPT iff it also
     passes every tier AND the critic signs off. a not-yet-coherent part skips the vision
     spend and repairs from the validator alone.
  6. revise -- ALWAYS from the BEST working candidate so far, never from a regression.

STRICT-IMPROVE (the anti-thrash rule): each built candidate is scored; the loop keeps
the best and always revises FROM it. A revision that makes the part worse (splits it into
separate bodies, breaks the build) is discarded -- the author is sent back to the working
script with an explicit "that made it worse, change less" note. This is why a good early
massing is never lost to an over-eager later edit.

Defaults to GENERAL fabrication: the goal is to MODEL the real object faithfully, not to
gate it on FDM overhang/support rules (a hammer's head overhangs by nature). Pass
``fabrication_mode=PRINT_3D`` for a part meant to print as-authored.

Everything is instrumented: per-iteration phase, cost, latency, validator + critic
verdicts, and the render path land in the returned ``LoopResult`` (the iteration log).
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from forge_mind.agentic import build_cache, feature_plan
from forge_mind.agentic import selfcheck as selfcheck_mod
from forge_mind.agentic.attribution import attribute_issues, line_attribution_enabled
from forge_mind.agentic.author import author_initial, author_vision_enabled
from forge_mind.agentic.brief import DesignBrief, research_brief
from forge_mind.agentic.cadcoder_route import (
    cadcoder_best_initial,
    cadcoder_enabled,
    is_cadquery,
    revise_dispatch,
    runner_for,
)
from forge_mind.agentic.collapse import (
    build_ratio,
    collapse_restart_enabled,
    collapse_restart_note,
    envelope_collapsed,
    lineage_all_collapsed,
)
from forge_mind.agentic.critic import (
    DesignCritique,
    composite_reference,
    critique_part,
    reference_enabled,
)
from forge_mind.agentic.drafts import select_best_draft
from forge_mind.agentic.idioms import idiom_pointer
from forge_mind.agentic.kernel_hints import api_misuse_hint, kernel_failure_hint
from forge_mind.agentic.param_opt import MassTarget, polish_params
from forge_mind.agentic.params import param_opt_polish_enabled
from forge_mind.agentic.reference_lib import fetch_reference_mesh
from forge_mind.agentic.repair import attempt_hole_repair
from forge_mind.agentic.revision import (
    _validity_tier,
    advisory_overhang_mm2,
    best_revision,
    revision_best_of_n,
)
from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code
from forge_mind.agentic.shape_match import shape_similarity
from forge_mind.agentic.shape_signature import conformance_advisory
from forge_mind.agentic.spec_dim_prompt import material_prompt_enabled
from forge_mind.agentic.staged import staged_enabled, staged_initial, staged_min_features
from forge_mind.agentic.stated_dims import stated_dim_audit, stated_numbers
from forge_mind.agentic.validate import AgenticValidation, validate_authored_part
from forge_mind.agentic.validator_loss import validator_target
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.provider import LLMClientLike, make_client
from forge_mind.materials import Material
from forge_mind.viz.render import render_part_to_png


@dataclass
class IterationRecord:
    """One pass through the loop: what happened, what it cost, how long it took."""

    iteration: int
    phase: str  # "initial" | "revise" | "restart" (fidelity restart: a fresh massing)
    approach: str
    built: bool
    sandbox_error: str | None
    duration_s: float
    cost_usd: float
    score: int | None = None
    kept_as_best: bool = False
    validation_passed: bool | None = None
    body_count: int | None = None
    issues: list[str] = field(default_factory=list)
    critic_looks_right: bool | None = None
    critic_resemblance: str | None = None
    critic_fixes: list[str] = field(default_factory=list)
    render_path: str | None = None


@dataclass
class LoopResult:
    """Outcome of an agentic run + the full iteration log (the instrumentation)."""

    prompt: str
    brief: DesignBrief | None
    final_code: str | None
    final_stl: Path | None
    final_step: Path | None
    final_render: Path | None
    validation: AgenticValidation | None
    critique: DesignCritique | None
    geometry_passed: bool  # deterministic: one watertight solid + all tiers (no FAILs)
    critic_approved: bool  # the vision critic signed off on resemblance
    stopped_reason: str  # "accepted"|"plateau"|"iteration_cap"|"budget"|"no_build"|"brief_failed"|"author_failed"|"infeasible_spec"
    iterations: list[IterationRecord] = field(default_factory=list)
    # Why the run was REFUSED before authoring (infeasible_spec): the stated arithmetic
    # conflicts, verbatim -- the "say why" surface (engine deep review 2026-07-04 ruling).
    refusal_reasons: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_duration_s: float = 0.0
    drafts_tried: int = 1
    drafts_built: int = 0
    # True when the deterministic hole-repair backstop closed a missing-feature deficit AFTER the
    # author finished (ADR-0033 Stage 2). Transparent by design: the bench separates "the author
    # built it" from "the backstop saved it", so a rising repair rate reads as an author regression.
    repaired_features: bool = False
    # Checkpoint resume (audit P2): a part reloaded from a prior run's checkpoint carries no
    # in-memory AgenticValidation; this preserves its recorded hard_ok so the coherence gate
    # (functional._part_coherent) honors the resumed verdict instead of degrading the disposition.
    resumed_hard_ok: bool = False


@dataclass
class _Candidate:
    """A built candidate the loop may keep as the working best."""

    score: int
    code: str
    stl: Path
    step: Path
    render: Path | None
    validation: AgenticValidation
    critique: DesignCritique | None
    selfcheck: dict[str, Any] | None = None  # the sandbox's measured deltas (lever: selfcheck)
    # ADR-0133 fix 3: safe_fillet/safe_chamfer outcome log (requested vs realized, or
    # SKIP) -- rides only when FORGE_EDGE_FINISH armed the sandbox job. Never gates.
    edge_finish: list[dict[str, Any]] | None = None


def _edge_finish_degraded(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ADR-0133 fix-3 degradation predicate: a SKIP (realized None) or a back-off to
    <= 50% of the requested radius/length. ``<=`` is load-bearing (review-caught): one
    failed kernel attempt succeeds at EXACTLY half -- the most common degradation."""
    out: list[dict[str, Any]] = []
    for e in events:
        req, rea = e.get("requested"), e.get("realized")
        if rea is None or (
            isinstance(req, (int, float)) and isinstance(rea, (int, float)) and rea <= 0.5 * req
        ):
            out.append(e)
    return out


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in name.strip().lower()]
    return ("".join(keep)[:48] or "part").strip("_") or "part"


def _disambiguate_slugs(names: list[str]) -> list[str]:
    """Slug each name to an id that is UNIQUE within the list, preserving order.

    Distinct names can slug to the same value ("Finger" and "finger" -> "finger";
    "arm A" and "arm-A" -> "arm_a"). Used naively as a per-part output dir, the second
    would silently OVERWRITE the first's geometry. Repeats get a stable per-index suffix
    so the ids stay 1:1 with the parts. Same input -> same output, so callers that
    re-derive ids (the assembly adapter) stay consistent with the output dirs."""
    ids: list[str] = []
    used: set[str] = set()
    for i, name in enumerate(names):
        candidate = _slug(name)
        if candidate in used:
            candidate = f"{candidate}_{i}"
        used.add(candidate)
        ids.append(candidate)
    return ids


def _score(validation: AgenticValidation, critique: DesignCritique | None) -> int:
    """Rank a built candidate. 0 = not a coherent solid (do not keep). Higher is better:
    1 coherent only, 2 coherent + looks right, 3 fully valid, 4 valid AND looks right."""
    if not validation.hard_ok:
        return 0
    looks_right = critique is not None and critique.looks_right
    if validation.passed and looks_right:
        return 4
    if validation.passed:
        return 3
    if looks_right:
        return 2
    return 1


_DEFAULT_MAX_ITERATIONS = 8  # raised 4 -> 8 (ADR-0078): multi-feature parts were dying at the cap
# WHILE STILL IMPROVING (hard-5: carrier + crank arm). The plateau logic is the adaptive early
# exit, so stalled parts still stop early; only improving ones use the extra room. Production's
# explicit 20/32 caps (api/core.py) are unaffected -- this is the default for benches/produce_part.


def default_max_iterations() -> int:
    """The iteration cap when the caller does not pass one: FORGE_MAX_ITERATIONS or 8."""
    raw = os.environ.get("FORGE_MAX_ITERATIONS")
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    return _DEFAULT_MAX_ITERATIONS


def _retry_once(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """One retry for an author-stage call (ADR-0078): the hard-5 flywheel died mid-loop on a
    SINGLE transient author parse failure. A second raise propagates to the existing graceful
    author_failed handling."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return fn(*args, **kwargs)


def _sandbox_feedback(error_type: str | None, message: str | None, traceback: str | None) -> str:
    tb = f"\n\nTraceback:\n{traceback}" if traceback else ""
    base = (
        f"Your script FAILED to build ({error_type}): {message}{tb}\n\n"
        "Fix the error so the script runs and binds `result` to one valid solid. "
        "Do not delete a correct feature to dodge the error; correct the API usage or geometry."
    )
    # Targeted OCCT fallback (lever b): name the FAILING OP and its ordered fallback
    # (smaller radius -> narrower selection -> skip the cosmetic op), so an opaque kernel
    # raise is never answered by deleting a required feature.
    hint = kernel_failure_hint(error_type, message, traceback)
    if hint:
        return f"{base}\n\nKERNEL DIAGNOSIS: {hint}"
    # Measured API hallucinations (2026-07-18 e2e A/B): local builders re-author the same
    # invented API verbatim; the correction must NAME the verified replacement.
    api = api_misuse_hint(message, traceback)
    return f"{base}\n\nAPI CORRECTION: {api}" if api else base


_MISSING_FEATURE_BANNER = (
    "#1 BLOCKING DEFECT -- A REQUIRED FEATURE IS ABSENT. Fix THIS before anything else; "
    "do NOT touch fillets, chamfers, or cosmetics this pass -- they are worthless while the "
    "feature is missing. The hole tools you last wrote produced ZERO holes in the geometry: "
    "STOP hand-rolling `Pos(...) * Cylinder(...)` boolean cuts and instead use the VERIFIED "
    "through_holes idiom -- sketch Circles on the target FACE (re-selected by axis, e.g. "
    "faces().sort_by(Axis.Z)[-1]), replicate the centres with GridLocations from the COUNT, "
    "then `extrude(amount=-<more than the thickness>, mode=Mode.SUBTRACT)` so each hole opens "
    "on BOTH faces. Apply edge fillets LAST, only after the holes are confirmed present."
)


# Anti-gaming escalation: how many CONSECUTIVE iterations a deterministic gate may stay red
# (while the critic score holds or rises) before author feedback is escalated to a HARD
# deterministic-fix instruction. The measured L-bracket failure -- the critic kept rating the
# render highly while the validator stayed red across all iterations -- generalized: a critic
# never authorizes accept while a deterministic gate is red, and a stuck red gate is hammered.
_STUCK_RED_ESCALATE_AFTER = 2

_STUCK_DETERMINISTIC_BANNER = (
    "#1 DETERMINISTIC GATE STILL RED -- a measured, NON-NEGOTIABLE check has failed for "
    "{streak} iterations straight while the look-right review stays positive. The reviewer's "
    "approval is IRRELEVANT until this clears: fix ONLY the validator failures below this pass, "
    "do NOT spend effort on cosmetics or resemblance. The part CANNOT be accepted while any "
    "validator check is failing, no matter how good it looks."
)


_RESEMBLANCE_ORD = {
    # the widened 6-point ordinal (FORGE_CRITIC_ORDINAL) is a superset of the 3-point scale,
    # so one map serves both: old transcripts and flag-off runs order exactly as before.
    "none": 0,
    "weak": 1,
    "fair": 2,
    "partial": 3,
    "good": 4,
    "strong": 5,
}


def _fidelity_rank(critique: DesignCritique | None) -> tuple[int, int]:
    """Orderable fidelity signal from a critique: (features judged present, resemblance
    ordinal). Compared ONLY between equal-score validator-passing candidates -- never to
    overrule a measured gate (score compares first, so the no-Goodhart order holds)."""
    if critique is None:
        return (-1, -1)
    present = sum(1 for v in critique.feature_verdicts if v.present)
    return (present, _RESEMBLANCE_ORD.get(critique.resemblance.strip().lower(), 0))


def _fidelity_improved(
    best: _Candidate | None,
    score: int,
    validation: AgenticValidation,
    critique: DesignCritique | None,
) -> bool:
    """Equal-score keep tie-break for the critic-blocked class (measured arm R, n=8,
    2026-07-18): a validator-passing part the critic blocks scores 3, and a revision that
    implements one NAMED fix (adds the V-groove) but does not flip looks_right in a single
    leap ALSO scores 3 -- under strict-improve it was discarded, the anchor never advanced,
    and multi-step fidelity progress was structurally impossible (L-bracket: 8/8 plateau,
    23/40 passing iterations, 0 ships; STLs show revisions returning to earlier geometry).
    A strictly better critique on a still-passing part is progress, not churn."""
    return (
        best is not None
        and score == best.score
        and validation.passed
        and _fidelity_rank(critique) > _fidelity_rank(best.critique)
    )


def _deterministic_red(validation: AgenticValidation) -> bool:
    """Is a DETERMINISTIC gate failing? (the authoritative no-false-PASS signal).

    ``validation.passed`` is True only when every measured gate holds (hard_ok, no failed checks,
    no features-missing / envelope / wall / material block). So a red deterministic gate is exactly
    ``not passed``. Used to (1) refuse early-accept while red regardless of the critic's score, and
    (2) detect a STUCK red gate to escalate author feedback."""
    return not validation.passed


# The critic-blocked class (measured arm R, n=8, 2026-07-18: the critic blocked 16/32
# validator-passing runs with NAMED fixes). When every measured gate is green, the reviewer's
# fix list IS the acceptance path -- but the old feedback labeled leftover advisory issues
# "must fix -- these block the part" (false on a passing part) and framed the fixes as
# cosmetics ("make it look like...") while other banners hammer "do NOT spend effort on
# cosmetics". This banner states the true blocker.
_CRITIC_BLOCKED_BANNER = (
    "#1 EVERY MEASURED GATE IS GREEN -- the ONLY thing between this part and acceptance is "
    "the review below. Implement EACH fix as a concrete geometric change to the working "
    "script (add the named feature exactly where it says); keep every feature that already "
    "passes. These fixes are the acceptance path, not cosmetic polish:"
)


def _combine_feedback(validation: AgenticValidation, critique: DesignCritique | None) -> str:
    parts: list[str] = []
    # A missing REQUIRED feature dominates: hoist it above every other note so the author
    # drills the holes FIRST and stops refining cosmetics (the measured L-bracket failure --
    # it kept tweaking the inner-corner fillet across all 5 iterations and never drilled).
    missing = [i for i in validation.issues if i.startswith("MISSING FEATURES")]
    if missing:
        parts.append(_MISSING_FEATURE_BANNER)
        parts.extend(f"  - {i}" for i in missing)
    rest = [i for i in validation.issues if not i.startswith("MISSING FEATURES")]
    if validation.passed and critique is not None and critique.prioritized_fixes:
        # Validator green + critic blocked: the fixes lead, and the leftover issues are
        # labeled what they are on a passing part -- advisories, not blockers.
        parts.append(_CRITIC_BLOCKED_BANNER)
        parts.extend(f"  - {f}" for f in critique.prioritized_fixes)
        if rest:
            parts.append(
                "ADVISORY (context only -- nothing below blocks the part; never regress a "
                "fix above to chase these):"
            )
            parts.extend(f"  - {i}" for i in rest)
        return "\n".join(parts)
    if rest:
        parts.append("VALIDATOR (must fix -- these block the part):")
        parts.extend(f"  - {i}" for i in rest)
    if critique is not None and critique.prioritized_fixes:
        parts.append("REVIEWER (make it look like the real object):")
        parts.extend(f"  - {f}" for f in critique.prioritized_fixes)
    return "\n".join(parts) if parts else "Refine the part to better match the brief."


_REGRESSION_NOTE = (
    "IMPORTANT: your last edit made the part WORSE (it split into separate bodies, failed to "
    "build, or stopped being a closed solid). The script above is the WORKING version -- start "
    "from it and make a SMALLER, safer change. The whole part must stay ONE fused, watertight "
    "solid: union features with `+`, never leave them as separate solids. KEEP every feature "
    "(fuse by overlapping); do NOT simplify the shape into a plain block to achieve one body."
)

# Measured 2026-07-18 (e2e A/B, benchmarks/2026-07-18_ctx_untruncation_ab/e2e_results.json): with
# greedy local decoding, a failed repair was re-authored UNCHANGED for up to 4 consecutive
# iterations (the same ImportError four times). A bare traceback re-roll reproduces the same
# completion; only naming the non-progress -- and demanding a structurally different fix --
# breaks the cycle.
_STUCK_REPAIR_NOTE = (
    "REPAIR NON-PROGRESS: your previous repair produced the IDENTICAL error again. Do NOT repeat "
    "the same approach. Remove or completely rewrite the failing construct named in the error "
    "below -- achieve the same feature a different, verified way (plain documented operations: "
    "Box, Cylinder, extrude, cut, fillet on selected edges; no invented functions, keyword "
    "arguments, or imports)."
)


def _improve_feedback(
    validation: AgenticValidation,
    critique: DesignCritique | None,
    regressed: bool,
    *,
    stuck_red_streak: int = 0,
) -> str:
    body = _combine_feedback(validation, critique)
    # Escalate when a deterministic gate has stayed red for too many consecutive iterations: the
    # critic's resemblance verdict is suppressed and the validator failures are hoisted to the top
    # with a HARD deterministic-fix banner, so the author stops polishing and fixes the gate.
    if stuck_red_streak >= _STUCK_RED_ESCALATE_AFTER and _deterministic_red(validation):
        validator_only = _combine_feedback(validation, None)
        body = f"{_STUCK_DETERMINISTIC_BANNER.format(streak=stuck_red_streak)}\n\n{validator_only}"
    return f"{_REGRESSION_NOTE}\n\n{body}" if regressed else body


# Dynamic early-stop: consecutive non-improving built iterations before the loop gives up. Once the
# output plateaus, more rounds on a stuck part only burn time + money without getting better. Tunable.
_PLATEAU_PATIENCE = 2

# A CONCRETE missing feature (a dropped through-hole the validator localizes) is a fixable,
# well-localized gap -- not a vague quality plateau -- so the loop keeps drilling at it for more
# non-improving rounds before giving up. Bounded (not unlimited): a part the author genuinely cannot
# complete still stops well inside the budget rather than grinding to the iteration/budget cap. This
# is the measured "author drops holes" fix (ADR-0030): with the live cap at 20+ iterations, a plain
# _PLATEAU_PATIENCE=2 was abandoning the bracket after ~2 tries while the holes were still absent.
_MISSING_FEATURE_PATIENCE = 5


def _attempt_note(attempt: int, no_improve: int) -> str:
    """Prefix for a revision that follows non-improving revisions.

    Measured (arm G, 2026-07-18): a revision from an UNCHANGED anchor composes an IDENTICAL
    prompt, and the greedy local roster returns an identical part -- L-bracket iterations 2 and 3
    produced byte-identical STLs in every sample, burning a full author+build+validate+critic
    iteration on a duplicate. Naming the attempt makes every revision prompt UNIQUE (breaking the
    deterministic fixed point) and demands a different tack once one revision failed to improve.
    """
    return (
        f"REVISION ATTEMPT {attempt}: the previous {no_improve} revision(s) did NOT improve the "
        "part -- repeating the same edit will produce the same result. Change your approach to "
        "the TOP issue this time: a different geometry idiom, a different order of operations, "
        "or a different way to satisfy the failing check."
    )


def _plateau_stop(
    no_improve: int,
    best_validation: AgenticValidation,
    best_critique: DesignCritique | None = None,
) -> bool:
    """Has the loop plateaued long enough to stop? A vague plateau gets a short leash
    (``_PLATEAU_PATIENCE``); while the best candidate is still SHORT a requested feature the leash is
    longer (``_MISSING_FEATURE_PATIENCE``) -- a dropped hole is concrete and fixable, worth more tries
    than a part that has merely stopped getting prettier. A critique naming ABSENT brief features on a
    validator-green part is the same concrete-gap class (measured arm S: the L-bracket that passed
    every gate as a solid block while the critic correctly saw no legs/rib) -- it earns the same
    longer leash, so the fidelity-restart rung gets room to run before the loop gives up."""
    fidelity_gap = (
        best_validation.passed
        and best_critique is not None
        and any(not v.present for v in best_critique.feature_verdicts)
    )
    patience = (
        _MISSING_FEATURE_PATIENCE
        if (best_validation.features_missing or fidelity_gap)
        else _PLATEAU_PATIENCE
    )
    return no_improve >= patience


# After how many consecutive non-improving revisions a fidelity-stuck run stops REVISING the
# attractor code and re-authors from scratch. 2 = the plain revise (no_improve 0->1) AND the
# salted best-of-3 (no_improve 1->2) both failed to move the critique -- the anchored-revision
# path is exhausted (measured arm S: L-bracket i4 plain and i5 salted both returned the anchor's
# byte-identical STL in 8/8 samples).
_FIDELITY_RESTART_AFTER = 2


def _fidelity_stuck(best: _Candidate | None, no_improve: int) -> bool:
    """Is the loop stuck on a validator-green part whose critique names ABSENT features, with the
    anchored-revision path exhausted? (the arm-S block attractor: every revision of the collapsed
    L-bracket block returned the block; only a fresh massing can escape)."""
    return (
        best is not None
        and no_improve >= _FIDELITY_RESTART_AFTER
        and best.validation.passed
        and best.critique is not None
        and any(not v.present for v in best.critique.feature_verdicts)
    )


_DETAIL_OPS = ("sweep(", "revolve(", "loft(", "PolarLocations", "Helix")


def _survivorship_note(failed_code: str, best_code: str) -> str | None:
    """Name a DETAIL attempt that just died in the sandbox while the working best lacks it
    (ADR-0117; measured mechanism (c): curved-feature attempts crash disproportionately, so
    keep-best survivorship ships blockiness while detail dies silently). Without this note the
    next revision quietly abandons the feature the crash was trying to build."""
    attempted = [op for op in _DETAIL_OPS if op in failed_code and op not in best_code]
    if not attempted:
        return None
    ops = ", ".join(o.rstrip("(") for o in attempted)
    return (
        f"DETAIL ATTEMPT DIED, NOT ABANDONED: the crashed code attempted {ops} that the "
        "working version does NOT have -- the feature it was building is still required. "
        "Fix the failing construct (smaller radius / correct plane / overlapping fuse); do "
        "not fall back to the featureless working version."
    )


def _error_restart_note(err: str, hint: str | None, attempt: int) -> str:
    """The MASSING STRATEGY note for an error-path restart (ADR-0117 rung 4 + the
    iteration-1 hard-error fresh start): mechanical -- the exact recurring error plus the
    triggered verified hint when one matched. Attempt number keeps every restart prompt
    unique under greedy decoding."""
    base = (
        f"FRESH RESTART {attempt}: the previous design kept dying on the SAME error: {err}. "
        "Do NOT reuse the previous construction -- choose a different, documented approach "
        "that cannot trigger that failure."
    )
    return f"{base}\nVERIFIED GUIDANCE: {hint}" if hint else base


def _fidelity_restart_note(critique: DesignCritique, attempt: int) -> str:
    """The MASSING STRATEGY note for a fidelity restart: mechanical, derived verbatim from the
    measured critique (absent features + its prioritized fixes) -- not a tuned prompt. The
    attempt number keeps every restart prompt unique under greedy decoding."""
    absent = ", ".join(v.feature for v in critique.feature_verdicts if not v.present)
    fixes = " ".join(f"- {f}" for f in critique.prioritized_fixes)
    note = (
        f"FRESH RESTART {attempt}: earlier drafts converged to the same design and the reviewer "
        f"still cannot see these brief features in the render: {absent}. Do NOT repeat the "
        "previous massing. Construct each listed feature as its OWN distinct solid with the "
        "stated proportions FIRST, then fuse them -- the silhouette must read as the object. "
        f"{fixes}"
    )
    # Verified idiom pointer (ADR-0117): when the absent features / fixes name a class we
    # hold an execution-verified construction for, ride it along (finding-triggered only).
    idiom = idiom_pointer(f"{absent} {fixes}")
    return f"{note}\n{idiom}" if idiom else note


def _author_vision_image(
    anchor: _Candidate | None,
    reference_stl: Path | None,
    out_dir: Path,
    object_name: str,
) -> tuple[bytes | None, bool]:
    """The render the author revises WITH when FORGE_AUTHOR_VISION=1 (ADR-0066): the working
    best candidate's own six-view render, stacked OVER the real reference render when the
    reference model is also on (the same BUILD-vs-REFERENCE composite the vision critic uses).

    Returns ``(png_bytes | None, image_has_reference)``. FAIL-CLOSED end to end: flag off, no
    anchor render, or ANY render/read failure yields ``(None, False)`` -- which makes
    ``author_revision`` byte-identical to the text-only path. The render is the anchor's, so the
    author sees exactly the code it is editing (``base = anchor.code``)."""
    if not author_vision_enabled() or anchor is None or anchor.render is None:
        return None, False
    try:
        if not anchor.render.exists():
            return None, False
        image = anchor.render
        has_reference = False
        if reference_stl is not None and reference_enabled():
            composite = composite_reference(anchor.render, reference_stl, out_dir, object_name)
            if composite is not None:
                image, has_reference = composite, True
        return image.read_bytes(), has_reference
    except Exception:
        return None, False


def _reference_render_bytes(reference_stl: Path | None, object_name: str) -> bytes | None:
    """The reference render shown to author_initial (ADR-0069) when FORGE_AUTHOR_VISION and
    FORGE_REFERENCE_MODEL are both on and a reference was fetched, so the first massing starts FROM
    the right form. Rendered once, cached next to the reference (the SAME cache the critic's
    composite uses -- no double render). Fail-closed: flag off / no reference / render failure ->
    None (a byte-identical text-only initial pass)."""
    if not author_vision_enabled() or reference_stl is None or not reference_enabled():
        return None
    try:
        ref_png = reference_stl.parent / f"{reference_stl.stem}__ref_render.png"
        if not ref_png.exists():
            render_part_to_png(reference_stl, ref_png, title=f"REFERENCE: {object_name}")
        return ref_png.read_bytes()
    except Exception:
        return None


def _reference_for_polish(reference_stl: Path | None, best_score: int) -> Path | None:
    """GUARD (ADR-0075, from the measured turbo-wheel incident): a CRITIC-ACCEPTED part
    (score 4 = fully valid AND looks right) must never be re-formed toward a retrieved
    reference -- a low-trust artist mesh cannot overrule an accepted form. The polish keeps
    only the validator terms (balance/wall/envelope), which never destroy form."""
    if best_score >= 4:
        return None
    return reference_stl


def _param_polish(
    best: _Candidate,
    reference_stl: Path | None,
    brief: DesignBrief,
    work_dir: Path,
    iterations: list[IterationRecord],
    *,
    fabrication_mode: FabricationMode,
    material: Material | None,
    sandbox_limits: SandboxLimits | None,
    part_name: str,
    prompt: str,
    mass_target: MassTarget | None = None,
) -> _Candidate:
    """Post-loop parameter fit (ADR-0073/0074, OPT-IN FORGE_PARAM_OPT): optimize the final best's
    float PARAMS against the reference form (when one was fetched) AND the validator-derived
    targets (wall/envelope/balance -- no reference needed) -- structure from the model, numbers
    from the optimizer, zero LLM calls. The optimized candidate replaces ``best`` ONLY if the FULL
    validator says its validity tier did not degrade (final acceptance authority stays with the
    validator -- the metric can steer taste, never validity). Fail-soft on every path."""
    t0 = time.monotonic()
    try:
        vtarget = validator_target(
            brief,
            baseline_step=best.step,
            baseline_stl=best.stl,
            fabrication_mode=fabrication_mode,
        )
        stated = stated_numbers(prompt, brief)
        outcome = polish_params(
            best.code,
            _reference_for_polish(reference_stl, best.score),
            work_dir / "param_opt",
            vtarget=vtarget,
            stated=stated,
            part_name=part_name,
            sandbox_limits=sandbox_limits,
            mass_target=mass_target,
        )
    except Exception:
        return best
    if outcome is None:
        return best  # no PARAMS contract in the script (or nothing measurable) -- not applicable
    note = f"param-opt: {outcome.evals} evals, loss {outcome.baseline_loss:.3f}->{outcome.best_loss:.3f}"
    if outcome.pinned:
        held = ", ".join(f"{n} {a:g}->{w:g}" for n, (a, w) in outcome.pinned.items())
        note += f" pinned[{held}]"
    accepted = False
    candidate = best
    if outcome.improved and outcome.best_stl is not None and outcome.best_step is not None:
        try:
            val = validate_authored_part(
                outcome.best_stl,
                outcome.best_step,
                brief,
                fabrication_mode=fabrication_mode,
                material=material,
                prompt=prompt,
            )
        except Exception:
            val = None
        old = best.validation
        tier_ok = (
            val is not None
            and val.hard_ok
            and _validity_tier(val.hard_ok, val.passed, val.features_missing)
            >= _validity_tier(old.hard_ok, old.passed, old.features_missing)
        )
        # ADR-0080: the stated-dim non-degrade audit is a CONJUNCT of acceptance -- the pins
        # make drift rare; this measured check is what makes the protection an honest claim.
        audit_ok = False
        if tier_ok:
            audit = stated_dim_audit(
                brief, best.step, outcome.best_step, best.stl, outcome.best_stl
            )
            audit_ok = audit.ok
            if not audit.ok:
                kind = "STATED-AUDIT-ERROR-REJECT" if audit.error else "STATED-DRIFT-REJECT"
                note += f" {kind}: " + "; ".join(audit.notes[:3])
        if tier_ok and audit_ok and val is not None:
            render_path = best.render
            try:
                target = work_dir / "param_opt" / "render.png"
                render_part_to_png(
                    outcome.best_stl,
                    target,
                    title=f"{brief.object_name} (param-fit)",
                    # model eyes get CLEAN geometry: a long prompt-as-caption occluded the
                    # panels and blinded the critic/judge (row-37 autopsy); the captioned
                    # render is the human artifact (_render_final)
                )
                render_path = target
            except Exception:
                render_path = best.render
            candidate = _Candidate(
                score=_score(val, None),
                code=outcome.best_code,
                stl=outcome.best_stl,
                step=outcome.best_step,
                render=render_path,
                validation=val,
                critique=None,  # the geometry changed; a stale critique must not ride along
            )
            accepted = True
    iterations.append(
        IterationRecord(
            iteration=len(iterations) + 1,
            phase="param_opt",
            approach=note + (" ACCEPTED" if accepted else " (kept original)"),
            built=True,
            sandbox_error=None,
            duration_s=time.monotonic() - t0,
            cost_usd=0.0,  # the whole point: the inner loop never calls the LLM
            score=candidate.score,
            kept_as_best=accepted,
            validation_passed=candidate.validation.passed,
        )
    )
    return candidate


def _maybe_apply_single_part_finish(
    result: LoopResult, work_dir: Path, client: Any, model: str | None
) -> LoopResult:
    """FORGE_FINISH for the SINGLE-PART pipeline (ADR-0129 fix #7): the designed-finish pass
    existed only in `produce_functional_part`, so the flag was silently a no-op on every
    single-part run (measured in the S4 smoke: armed, zero `finish:` lines).

    Same laws as the multi-part pass: the planner is forbidden from mating surfaces, the
    finished geometry must revalidate NOT-WORSE (`_finished_part_not_worse`) before it replaces
    the clean files, and any failure anywhere keeps the clean part -- polish never ruins a
    working part, and this function never raises.
    """
    from forge_mind.agentic.detail import finish_enabled

    if not finish_enabled() or result.final_step is None or result.final_stl is None:
        return result
    if result.stopped_reason == "budget":
        # A blown dollar cap must not fund one more planner call (review-caught).
        return result
    try:
        import json as _json

        import trimesh as _tm

        from forge_mind.agentic.finish_planner import (
            PartFinishContext,
            finish_step_file,
            plan_part_finish,
        )
        from forge_mind.agentic.functional import _finished_part_not_worse

        mesh = _tm.load(str(result.final_stl), force="mesh")
        lo, hi = mesh.bounds
        name = result.brief.object_name if result.brief is not None else "part"
        # CONCRETE mating exclusions (review-caught: "none declared" left edge_treatment free
        # to chamfer a bearing bore). No Port structures exist on this path, but the brief's
        # own feature cards name the mating geometry -- hand every bore/hole/thread/journal
        # feature to the planner verbatim, on top of its categorical prohibition.
        exclusions: list[str] = []
        if result.brief is not None:
            mating = re.compile(
                r"bore|hole|thread|shaft|bearing|bushing|journal|slot|pin\b|fit\b", re.IGNORECASE
            )
            for f in getattr(result.brief, "features", None) or []:
                if mating.search(f.name) or mating.search(f.approx_dims_mm or ""):
                    exclusions.append(
                        f"feature '{f.name}' ({f.approx_dims_mm}) -- a mating surface, keep "
                        "ALL finish clear of it"
                    )
        ctx = PartFinishContext(
            name=name,
            role=(result.brief.summary[:120] if result.brief is not None else name),
            bbox=((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])),
            mating_exclusions=exclusions,
        )
        ops = plan_part_finish(ctx, client=client or make_client(role="planner"), model=model)
        print(
            f"finish: '{name}' -> {len(ops)} op(s)"
            + (f" {[o.op for o in ops]}" if ops else " (restraint: left bare)"),
            flush=True,
        )
        if not ops:
            return result
        fdir = work_dir / "finish"
        fdir.mkdir(parents=True, exist_ok=True)
        out_stl, out_step = fdir / "part.stl", fdir / "part.step"
        if not finish_step_file(result.final_step, ops, out_stl, out_step):
            print("finish: op(s) failed to apply; clean part kept", flush=True)
            return result
        ok, why, fin_v = _finished_part_not_worse(result, out_stl, out_step)
        if not ok:
            print(f"finish: reverted ({why}); clean part kept", flush=True)
            return result
        # RE-RENDER the finished geometry (review-caught): the acceptance judge downstream
        # reads final_render, and judging the clean image against finished files would let a
        # visually wrong finish inherit the clean part's approval. Render failure -> revert;
        # a part we cannot show honestly is not shipped polished.
        out_render = fdir / "render.png"
        try:
            render_part_to_png(out_stl, out_render, title=f"{name} (finished)")
        except Exception:
            print("finish: reverted (finished render failed); clean part kept", flush=True)
            return result
        # Sidecar so the applied ops are auditable. KNOWN LIMITATION (documented, not hidden):
        # final_code still builds the CLEAN body, so a parametric rebuild regenerates the part
        # WITHOUT these ops -- finish is a post-pass on files, not code.
        with contextlib.suppress(Exception):
            (fdir / "finish_ops.json").write_text(
                _json.dumps([{"op": o.op, "params": o.params} for o in ops], indent=1),
                encoding="utf-8",
            )
        print("finish: applied; finished geometry revalidated not-worse", flush=True)
        return replace(
            result,
            final_stl=out_stl,
            final_step=out_step,
            final_render=out_render,
            # the FINISHED validation replaces the clean record (review-caught): reports,
            # mass/cost and training labels must describe the shipped files.
            validation=fin_v if fin_v is not None else result.validation,
            geometry_passed=(fin_v.passed if fin_v is not None else result.geometry_passed),
        )
    except Exception as exc:
        print(f"finish: skipped ({type(exc).__name__}); clean part kept", flush=True)
        return result


def _recritique_repaired(
    outcome: Any,
    brief: DesignBrief,
    repaired_render: Path | None,
    *,
    client: LLMClientLike | None,
    model: str | None,
    work_dir: Path,
    reference_stl: Path | None,
    budget_left_usd: float,
) -> tuple[DesignCritique | None, float]:
    """``(critique of the REPAIRED geometry, cost)`` -- never the pre-repair one.

    The hole-repair backstop CUTS HOLES into the finished part, and the loop used to rebuild its
    kept candidate with the critique of the PRE-repair render. `critic_approved` reads
    ``looks_right`` off that, so a stale approval could ride changed geometry -- a potential
    false-PASS (measured on H5B's foot_r, whose recorded rejection described geometry the repair
    had already replaced).

    Fail-CLOSED on every doubt, so the shipped part reads ``critic_approved=False`` rather than
    inheriting a verdict about geometry that no longer exists:

    * ``repaired_render`` must be a render OF THE REPAIRED STL. The caller's display fallback to
      the pre-repair image must NOT reach here (review-caught): with critic views off,
      ``critique_part`` reuses the given image, so a stale render would recreate the exact false
      approval this exists to remove.
    * an incoherent repaired part (not one watertight solid) is never critiqued;
    * ``budget_left_usd <= 0`` skips the call: a run that already stopped ON the budget ceiling
      must not spend past it for a verdict (review-caught).

    The measured-refutation override (ADR-0121, a promoted lane default) is applied exactly as
    the iteration path applies it, so a validator-green repaired part is not rejected merely for
    having taken the repair path. Never raises."""
    if repaired_render is None or not outcome.validation.hard_ok or budget_left_usd <= 0:
        return None, 0.0
    try:
        from forge_mind.agentic.critic_context import critic_measured_block
        from forge_mind.agentic.critic_refutation import OVERRIDE_MARKER, refutation_override

        critique, usage = critique_part(
            outcome.stl,
            brief,
            client=client,
            model=model,
            work_dir=work_dir,
            render_path=repaired_render,
            reference_stl=reference_stl,
            measured=critic_measured_block(outcome.validation, None, brief),
        )
        if critique is not None and refutation_override(outcome.validation, critique, brief):
            critique = critique.model_copy(
                update={
                    "looks_right": True,
                    "summary": f"{critique.summary} {OVERRIDE_MARKER}",
                }
            )
        return critique, usage.cost_usd
    except Exception as exc:
        print(
            f"repair re-critique failed ({type(exc).__name__}) -- the repaired part ships with "
            "NO critique rather than the pre-repair one",
            flush=True,
        )
        return None, 0.0


def run_agentic_pipeline(
    prompt: str,
    work_dir: Path,
    *,
    apply_finish: bool = True,
    max_iterations: int | None = None,
    budget_usd: float = 2.0,
    n_drafts: int = 1,
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
    material: Material | None = None,
    brief_model: str | None = None,
    author_model: str | None = None,
    critic_model: str | None = None,
    sandbox_limits: SandboxLimits | None = None,
    brief: DesignBrief | None = None,
    client: LLMClientLike | None = None,
    mass_target: MassTarget | None = None,
) -> LoopResult:
    """Run one part end-to-end through the agentic loop. Never raises on a stage failure.

    ``brief`` (optional): a pre-researched brief. When given, the research call is skipped
    (and its cost is the caller's -- e.g. ``produce_part`` researches once to route on
    ``brief.target``); when None, the loop researches the prompt itself.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # Role-based clients (ADR-0113, local-first roster). With no explicitly-injected client each stage
    # uses its role's provider: the CODER is the local CAD-Coder route (below); REPAIR (author_revision)
    # runs on the local qwen3-coder; the VISION critic on the local Qwen2.5-VL; a rare fallback cloud
    # author uses the coder role. An explicit ``client`` (tests) overrides all roles -> byte-identical.
    coder_client = client or make_client(role="coder")
    repair_client = client or make_client(role="repair")
    vision_client = client or make_client(role="vision")
    # The planner client is built ONLY when the research step will run: constructing it eagerly
    # crashed every pre-researched all-local run under FORGE_FORBID_API_SPEND (the planner role
    # defaults to the metered client, which the guard rightly refuses) even though the planner is
    # never called when ``brief`` is given (measured 2026-07-18, arm F).
    planner_client = (client or make_client(role="planner")) if brief is None else client
    client = client or coder_client  # legacy single-client references default to the coder role
    max_iterations = max_iterations if max_iterations is not None else default_max_iterations()
    sandbox_limits = sandbox_limits or SandboxLimits(timeout_s=240)
    t_start = time.monotonic()
    iterations: list[IterationRecord] = []
    total_cost = 0.0

    # Content-hash cache (task #13a, opt-in via FORGE_BUILD_CACHE): a repeat of an identical
    # (prompt + settings) build is reused at ZERO LLM cost. Keyed on prompt+settings (NOT the brief),
    # so it fires on the LIVE path too -- the SAME prompt returns the SAME part (output determinism),
    # whether or not the caller pre-researched the brief (the deployed /generate does). A cached result
    # carries brief=None; every downstream `loop.brief` read is guarded, so it flows through safely.
    cache_key: str | None = None
    if build_cache.enabled():
        cache_key = build_cache.compute_key(
            prompt,
            material=material,
            n_drafts=n_drafts,
            max_iterations=max_iterations,
            fabrication_mode=fabrication_mode,
            author_model=author_model,
            critic_model=critic_model,
            brief_model=brief_model,
            # A caller-supplied brief can carry a mass target the prompt does not state
            # (review-caught): the key must separate those, or the second request is served
            # the first mass's geometry. None here when the loop researches its own brief.
            target_mass_g=getattr(brief, "target_mass_g", None),
        )
        cached = build_cache.load(cache_key, work_dir)
        if cached is not None:
            # The cache loader restores the brief in memory but copies only final.* -- without
            # this, every production repeat (FORGE_BUILD_CACHE is set in fly.toml/Dockerfile)
            # would lack the brief.json promised below (review-caught).
            if cached.brief is not None:
                try:
                    work_dir.mkdir(parents=True, exist_ok=True)
                    (work_dir / "brief.json").write_text(
                        cached.brief.model_dump_json(indent=1), encoding="utf-8"
                    )
                except Exception:
                    pass
            return cached

    # A RETRIED work_dir may hold the PREVIOUS attempt's brief.json; if this attempt's research
    # fails, that stale file would describe a brief the returned result does not carry
    # (review-caught). Clear first, write only on success.
    with contextlib.suppress(Exception):
        (work_dir / "brief.json").unlink(missing_ok=True)

    # --- research (skipped when the caller already researched + passed the brief) ---
    if brief is None:
        try:
            brief, usage = research_brief(prompt, client=planner_client, model=brief_model)
            total_cost += usage.cost_usd
        except Exception as exc:
            # NAME the failure in the run log (north-star iteration 2: four parts died
            # brief_failed with ZERO trace -- the cascade's error class was unrecoverable
            # from the artifacts). Diagnosability only; the graceful result is unchanged.
            print(f"brief_failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            return LoopResult(
                prompt=prompt,
                brief=None,
                final_code=None,
                final_stl=None,
                final_step=None,
                final_render=None,
                validation=None,
                critique=None,
                geometry_passed=False,
                critic_approved=False,
                stopped_reason="brief_failed",
                iterations=[],
                total_cost_usd=total_cost,
                total_duration_s=time.monotonic() - t_start,
            )

    # MATERIAL RECONCILIATION (FORGE_MATERIAL_PROMPT, dark; review-caught): a caller-selected
    # material drives mass reporting while `brief.material` -- the researcher's own guess --
    # drives validation and (with this flag) the author prompt. Left split, the flag could tell
    # the author "PLA 1.24" for a run the caller selected steel for, recreating the very
    # density error it exists to remove. When the caller states a material, it WINS: one value
    # reaches author, validator, and report. Flag-off leaves the split exactly as it was.
    if material is not None and material_prompt_enabled():
        brief = brief.model_copy(update={"material": material.id})

    slug = _slug(brief.object_name)

    # ARCHIVE THE BRIEF (2026-07-25, the S1-vs-S3 lesson): the brief is the run's single most
    # causal input -- the S1/S3 impeller pair isolated researched-vs-curated brief as the
    # no_build-vs-curved-wheel delta, and NEITHER run had archived it, so the comparison had to
    # be reconstructed from drivers. Best-effort: an unwritable dir must never kill a run.
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "brief.json").write_text(brief.model_dump_json(indent=1), encoding="utf-8")
    except Exception:
        pass

    # Reference-model spine (Step 3, OPT-IN FORGE_REFERENCE_MODEL): fetch ONE real 3D model of
    # the target part (Sketchfab, relevance-guarded + fail-closed) so the vision critic judges
    # against ground truth, not the model's own text idea. None (flag off / no match / offline)
    # -> the critic stays text-only, byte-identical. Fetched once here; reused every iteration.
    reference_stl: Path | None = None
    if reference_enabled():
        try:
            reference_stl = fetch_reference_mesh(brief.object_name, work_dir / "reference")
        except Exception:
            reference_stl = None

    # Feature-plan lever (lever f, OPT-IN FORGE_FEATURE_PLAN=1): the deterministic checklist
    # the author must map its code against. Empty when the lever is off (deployed default).
    plan_checklist = feature_plan.feature_checklist(brief) if feature_plan.enabled() else []
    plan_refusals = 0
    feature_map: list[Any] | None = None  # the author's latest plan-to-code mapping

    # Self-check lever (OPT-IN FORGE_AUTHOR_SELFCHECK=1): the brief-derived measurement
    # targets the sandbox child checks each build against. None = no epilogue (deployed
    # default; job.json byte-identical).
    sc_spec = selfcheck_mod.build_spec(brief, fabrication_mode) if selfcheck_mod.enabled() else None

    # --- first (blocked-out) version: best-of-N drafts, or a single draft -----
    drafts_tried, drafts_built = 1, 0
    # Staged construction (FORGE_STAGED_BUILD, default OFF -- flag-off byte-identical):
    # replaces ONLY this initial construction with base+per-feature verified stages
    # (agentic/staged.py); everything downstream is unchanged. A partial staged build
    # (stage refusal) still enters the loop -- the missing-feature machinery finishes it.
    code = None
    # Always bound: the CAD-Coder first-pass (local, $0) sets `code` without going through the
    # staged/drafts/author branches that used to be the only place this was assigned (bugfix).
    last_author_cost = 0.0
    if staged_enabled() and len(brief.features) >= staged_min_features():
        sres_staged = staged_initial(
            brief,
            client=client,
            model=author_model,
            fabrication_mode=fabrication_mode,
            work_dir=work_dir / "staged",
            sandbox_limits=sandbox_limits,
            part_name=slug,
        )
        total_cost += sres_staged.cost_usd
        last_author_cost = sres_staged.cost_usd
        if sres_staged.code is not None:
            code, approach, phase = sres_staged.code, sres_staged.approach, "initial"
            feature_map = None
    # CAD-Coder first-pass (FORGE_GENERATOR=cadcoder, ADR-0112/0114, default OFF -- flag-off
    # byte-identical): BEST-OF-N local samples, each built in the sandbox and scored against the
    # brief's measured targets (envelope + bore counts), keep the winner. Selection kills the
    # measured #1 failure -- a draft that builds but silently lost a requested hole. On any failure
    # it returns None and the build123d author below runs instead.
    if code is None and cadcoder_enabled():
        # The draft scorer needs the brief's measured targets even when the selfcheck epilogue
        # flag is off for the main loop (build_spec is pure -- deterministic from the brief).
        # Wall target stripped: the draft scorer reads envelope + bores only, so leaving it in
        # would pay a 4096-ray wall measurement per draft that nothing reads (review-caught).
        score_spec = selfcheck_mod.scoring_spec(
            sc_spec if sc_spec is not None else selfcheck_mod.build_spec(brief, fabrication_mode)
        )
        cc_code = cadcoder_best_initial(
            prompt, work_dir, sc_spec=score_spec, sandbox_limits=sandbox_limits
        )
        if cc_code is not None:
            code, approach, phase = cc_code, "cadcoder", "initial"
            feature_map = None
    if code is not None:
        pass
    elif n_drafts > 1:
        selection = select_best_draft(
            brief,
            n_drafts,
            work_dir / "drafts",
            client=client,
            author_model=author_model,
            fabrication_mode=fabrication_mode,
            sandbox_limits=sandbox_limits,
            part_name=slug,
        )
        total_cost += selection.cost_usd
        last_author_cost = selection.cost_usd
        code, approach, phase = selection.best_code, selection.best_approach, "initial"
        drafts_tried, drafts_built = len(selection.drafts), selection.n_built
    else:
        # GUARDED: an author-stage failure (e.g. a structured response the SDK cannot parse even
        # after the client-level retry) must end the part GRACEFULLY, never crash the pipeline --
        # the measured fan-shroud bench run died here on a truncated response. Same contract as the
        # brief stage above.
        try:
            authored, usage = _retry_once(
                author_initial,
                brief,
                client=client,
                model=author_model,
                fabrication_mode=fabrication_mode,
                image_png=_reference_render_bytes(reference_stl, brief.object_name),
            )
        except Exception:
            return LoopResult(
                prompt=prompt,
                brief=brief,
                final_code=None,
                final_stl=None,
                final_step=None,
                final_render=None,
                validation=None,
                critique=None,
                geometry_passed=False,
                critic_approved=False,
                stopped_reason="author_failed",
                iterations=[],
                total_cost_usd=total_cost,
                total_duration_s=time.monotonic() - t_start,
            )
        total_cost += usage.cost_usd
        last_author_cost = usage.cost_usd
        code, approach, phase = authored.code, authored.approach, "initial"
        feature_map = getattr(authored, "feature_map", None)

    best: _Candidate | None = None
    regressed = False
    stopped_reason = "iteration_cap"
    # Anti-gaming: consecutive built iterations where a DETERMINISTIC gate stayed red while the
    # critic score held or rose. A stuck streak escalates author feedback to a hard fix and the
    # loop refuses to early-accept while red (the critic can never override a failing measured gate).
    stuck_red_streak = 0
    prev_score = 0
    no_improve = 0  # consecutive built iterations that did not improve the best score (early-stop)
    # Per-iteration built-extent/envelope ratios (FORGE_COLLAPSE_RESTART, dark; L102 class).
    collapse_ratios: list[float | None] = []

    for i in range(1, max_iterations + 1):
        it_dir = work_dir / f"iter_{i:02d}"
        t_it = time.monotonic()

        # --- feature-plan pre-build audit (lever f, FORGE_FEATURE_PLAN=1) ----
        # A draft whose plan-to-code mapping misses a declared feature is refused BEFORE
        # the sandbox (no build / render / vision spend) and sent back with the exact gaps.
        # BOUNDED (refusal_cap): a stubborn mapping eventually proceeds to the measured
        # gates -- the audit is a cheap pre-filter, never a replacement for the validator.
        # `feature_map is None` (lever off, or draft-selected code) skips the audit.
        if (
            plan_checklist
            and feature_map is not None
            and plan_refusals < feature_plan.refusal_cap()
        ):
            gaps = feature_plan.audit_feature_map(plan_checklist, feature_map, code)
            if gaps:
                plan_refusals += 1
                iterations.append(
                    IterationRecord(
                        iteration=i,
                        phase=phase,
                        approach=approach,
                        built=False,
                        sandbox_error=(
                            f"feature_plan: refused pre-build -- {len(gaps)} unmapped/unbuilt "
                            "declared feature(s)"
                        ),
                        duration_s=time.monotonic() - t_it,
                        cost_usd=last_author_cost,
                    )
                )
                if i == max_iterations or total_cost >= budget_usd:
                    stopped_reason = (
                        "budget"
                        if total_cost >= budget_usd
                        else ("iteration_cap" if best else "no_build")
                    )
                    break
                # Revise FROM THE REFUSED DRAFT (the gaps reference ITS mapping), never from
                # best: the refused code was never built/scored, and it is usually one
                # feature short of complete -- the smallest fix is to finish IT.
                try:
                    authored, usage = _retry_once(
                        revise_dispatch,
                        brief,
                        code,
                        feature_plan.refusal_feedback(gaps),
                        client=repair_client,
                        model=author_model,
                        fabrication_mode=fabrication_mode,
                    )
                except Exception:
                    stopped_reason = "author_failed"
                    break
                total_cost += usage.cost_usd
                last_author_cost = usage.cost_usd
                code, approach, phase = authored.code, authored.approach, "revise"
                feature_map = getattr(authored, "feature_map", None)
                continue

        sandbox = run_build123d_code(
            code,
            it_dir,
            part_name=slug,
            limits=sandbox_limits,
            selfcheck=sc_spec,
            runner_module=runner_for(code),
        )
        collapse_ratios.append(
            build_ratio(sandbox.extents, brief.overall_envelope_mm if brief is not None else None)
        )

        # --- did not build: repair from the traceback, from the BEST base -----
        if not sandbox.ok:
            iterations.append(
                IterationRecord(
                    iteration=i,
                    phase=phase,
                    approach=approach,
                    built=False,
                    sandbox_error=f"{sandbox.error_type}: {sandbox.error_message}",
                    duration_s=time.monotonic() - t_it,
                    cost_usd=last_author_cost,
                )
            )
            regressed = best is not None
            if i == max_iterations or total_cost >= budget_usd:
                stopped_reason = (
                    "budget"
                    if total_cost >= budget_usd
                    else ("iteration_cap" if best else "no_build")
                )
                break
            base = best.code if best is not None else code
            fb = _sandbox_feedback(sandbox.error_type, sandbox.error_message, sandbox.traceback)
            # Consecutive trailing iterations that failed with the IDENTICAL error.
            error_streak = 0
            for rec in reversed(iterations):
                if rec.sandbox_error and rec.sandbox_error == iterations[-1].sandbox_error:
                    error_streak += 1
                else:
                    break
            if error_streak >= 2:
                fb = f"{_STUCK_REPAIR_NOTE}\n\n{fb}"
            if best is not None:
                fb = f"{_REGRESSION_NOTE}\n\n{fb}"
                # Survivorship guard (ADR-0117): the crash was ATTEMPTING detail the working
                # best lacks -- keep the attempt alive in the feedback instead of letting
                # keep-best silently ship the blocky version.
                survivorship = _survivorship_note(code, best.code)
                if survivorship:
                    fb = f"{survivorship}\n\n{fb}"
            triggered_hint = api_misuse_hint(
                sandbox.error_message, sandbox.traceback
            ) or kernel_failure_hint(sandbox.error_type, sandbox.error_message, sandbox.traceback)
            # Iteration-1 hard-error fresh start (ADR-0117): when the FIRST draft dies on an
            # error a verified hint recognizes, a patch-repair inherits a structurally doomed
            # construction (measured: revolve(angle=) recurs at iteration 1 across runs and its
            # repair lands in Null TopoDS). Author a FRESH massing with the hint as the strategy
            # note instead -- one bounded, cost-neutral swap (author call replaces repair call).
            if i == 1 and best is None and brief is not None and triggered_hint is not None:
                try:
                    authored, usage = _retry_once(
                        author_initial,
                        brief,
                        client=coder_client,
                        model=author_model,
                        strategy=_error_restart_note(
                            f"{sandbox.error_type}: {sandbox.error_message}", triggered_hint, i + 1
                        ),
                        fabrication_mode=fabrication_mode,
                    )
                except Exception:
                    stopped_reason = "author_failed"
                    break
                total_cost += usage.cost_usd
                last_author_cost = usage.cost_usd
                code, approach, phase = authored.code, "iter-1 hard-error fresh start", "restart"
                feature_map = getattr(authored, "feature_map", None)
                continue
            # Rung 4 (ADR-0117): the salted best-of-3 REPAIR (rung 3, below) already fired at
            # streak 3 and the SAME error came back -- repair cannot leave this basin at all
            # (measured: 6 whole pulley runs burned on identical Null TopoDS; salted revision
            # escapes were 0/8 on the block attractor). Author 3 FRESH massings through the
            # draft selector with the error class as the strategy suffix; only code that BUILT
            # is forwarded when any did.
            if error_streak >= 4 and brief is not None and not is_cadquery(base):
                try:
                    selection = select_best_draft(
                        brief,
                        3,
                        work_dir / f"error_restart_{i:02d}",
                        client=coder_client,
                        author_model=author_model,
                        fabrication_mode=fabrication_mode,
                        sandbox_limits=sandbox_limits,
                        part_name=slug,
                        strategy_suffix=_error_restart_note(
                            f"{sandbox.error_type}: {sandbox.error_message}", triggered_hint, i + 1
                        ),
                    )
                except Exception:
                    stopped_reason = "author_failed"
                    break
                total_cost += selection.cost_usd
                last_author_cost = selection.cost_usd
                code, approach, phase = (
                    selection.best_code,
                    f"error restart (best-of-3, {selection.n_built} built)",
                    "restart",
                )
                feature_map = None
                continue
            # Structural escalation (measured 2026-07-18): the stuck note + a correct API hint
            # still let an identical error run 3-4 iterations (cable clip, both recheck runs) --
            # a single greedy repair keeps landing in the same basin. On the THIRD identical
            # failure, author k=3 SALTED candidates (ADR-0116) and keep the best that builds;
            # build123d path only (a CadQuery part repairs in CadQuery, ADR-0114).
            if error_streak >= 3 and brief is not None and not is_cadquery(base):
                code, esc_cost, _n = best_revision(
                    brief,
                    base,
                    fb,
                    k=3,
                    client=repair_client,
                    author_model=author_model,
                    fabrication_mode=fabrication_mode,
                    image_png=None,
                    image_has_reference=False,
                    reference_stl=None,
                    material=material,
                    sandbox_limits=sandbox_limits,
                    work_dir=it_dir / "stuck_escalation",
                    part_name=slug,
                )
                total_cost += esc_cost
                last_author_cost = esc_cost
                approach, phase = "stuck-escalation best-of-3", "revise"
                feature_map = None
                continue
            try:
                authored, usage = _retry_once(
                    revise_dispatch,
                    brief,
                    base,
                    fb,
                    client=repair_client,
                    model=author_model,
                    fabrication_mode=fabrication_mode,
                )
            except Exception:
                # GUARDED: end the part gracefully, keeping the best so far (never crash mid-loop).
                stopped_reason = "author_failed"
                break
            total_cost += usage.cost_usd
            last_author_cost = usage.cost_usd
            code, approach, phase = authored.code, authored.approach, "revise"
            feature_map = getattr(authored, "feature_map", None)
            continue

        # --- built: validate (deterministic) BEFORE any vision spend ----------
        assert sandbox.stl_path is not None and sandbox.step_path is not None
        validation = validate_authored_part(
            sandbox.stl_path,
            sandbox.step_path,
            brief,
            fabrication_mode=fabrication_mode,
            material=material,
            author_code=code,
            prompt=prompt,
        )
        render_target = it_dir / "render.png"
        render_path: Path | None
        try:
            render_part_to_png(
                sandbox.stl_path,
                render_target,
                title=f"{brief.object_name} (iter {i})",
                # no caption: this render feeds the CRITIC (and ultimately the judge) — model
                # eyes get clean geometry; the prompt reaches them as text (row-37 occlusion fix)
            )
            render_path = render_target
        except Exception:
            render_path = None

        critique: DesignCritique | None = None
        critic_cost = 0.0
        if validation.hard_ok:
            try:
                # ADR-0119 (FORGE_CRITIC_MEASURED=1): hand the critic the instruments'
                # verdict so its fixes anchor on MEASURED deficits. A render hides a thin
                # flange or occluded holes; the B-rep does not -- without this the critic
                # re-demands features the validator measured present (the L66 manifold
                # flange class) and revision budget burns on phantoms. None = byte-identical.
                # Construction extracted to critic_context.critic_measured_block (L71):
                # the brief's STATED dims now ride the block so spec-true slender
                # geometry reads as conformant, not fragile.
                from forge_mind.agentic.critic_context import critic_measured_block

                measured_block = critic_measured_block(validation, sandbox, brief)
                critique, c_usage = critique_part(
                    sandbox.stl_path,
                    brief,
                    client=vision_client,
                    model=critic_model,
                    work_dir=it_dir,
                    render_path=render_path,
                    reference_stl=reference_stl,
                    measured=measured_block,
                )
                total_cost += c_usage.cost_usd
                critic_cost = c_usage.cost_usd
                # Measured-refutation override (L72 design, founder-approved 2026-07-22;
                # FORGE_CRITIC_REFUTATION_OVERRIDE=1, default OFF): a veto whose EVERY
                # named fix contradicts a measured value cannot block a validator-green
                # part. The critique text survives verbatim; the acceptance is marked.
                from forge_mind.agentic.critic_refutation import (
                    OVERRIDE_MARKER,
                    refutation_override,
                )

                if critique is not None and refutation_override(validation, critique, brief):
                    critique = critique.model_copy(
                        update={
                            "looks_right": True,
                            "summary": f"{critique.summary} {OVERRIDE_MARKER}",
                        }
                    )
            except Exception:
                critique = None

        score = _score(validation, critique)
        # Tie-break is DETERMINISTIC: keep the EARLIER best on an equal score (strict ">"), so a
        # later candidate of the same score never churns the working base -- UNLESS the equal-score
        # candidate strictly REDUCED the measured unsupported-overhang area (the attractor-escape
        # probe found real 313->253 mm^2 improvements the 0-4 score is blind to; a measured
        # advisory decrease is progress, not churn, and the no-Goodhart order holds: score first).
        # ``regressed`` stays decoupled from ``keep`` -- only a STRICTLY worse build is a
        # regression (an equal-score rebuild is held back without the misleading "worse" note).
        overhang_improved = (
            best is not None
            and score == best.score
            and advisory_overhang_mm2(validation.issues)
            < advisory_overhang_mm2(best.validation.issues)
        )
        fidelity_improved = _fidelity_improved(best, score, validation, critique)
        keep = score >= 1 and (
            best is None or score > best.score or overhang_improved or fidelity_improved
        )
        regressed = best is not None and score < best.score
        # Anti-gaming streak: a deterministic gate staying red while the critic score holds or rises
        # is exactly the L-bracket failure (the look-right verdict drifting up while the validator
        # never clears). Count those runs; a clear resets it. The critic NEVER unblocks accept.
        if _deterministic_red(validation):
            stuck_red_streak = stuck_red_streak + 1 if score >= prev_score else 1
        else:
            stuck_red_streak = 0
        prev_score = score
        if keep:
            no_improve = 0
            best = _Candidate(
                score=score,
                code=code,
                stl=sandbox.stl_path,
                step=sandbox.step_path,
                render=render_path,
                validation=validation,
                critique=critique,
                selfcheck=sandbox.selfcheck,
                edge_finish=sandbox.edge_finish,
            )
        else:
            no_improve += 1

        iterations.append(
            IterationRecord(
                iteration=i,
                phase=phase,
                approach=approach,
                built=True,
                sandbox_error=None,
                duration_s=time.monotonic() - t_it,
                cost_usd=last_author_cost + critic_cost,
                score=score,
                kept_as_best=keep,
                validation_passed=validation.passed,
                body_count=validation.body_count,
                issues=list(validation.issues),
                critic_looks_right=(critique.looks_right if critique else None),
                critic_resemblance=(critique.resemblance if critique else None),
                critic_fixes=(list(critique.prioritized_fixes) if critique else []),
                render_path=str(render_path) if render_path else None,
            )
        )

        # ACCEPT only when fully valid AND the deterministic gate is GREEN. The second clause is the
        # anti-gaming guard: the critic (folded into score 4) can NEVER authorize an early accept
        # while a measured gate is red -- a red gate keeps `passed` False, so this never early-stops
        # a part the validator rejects, no matter how strong the resemblance verdict.
        if score >= 4 and not _deterministic_red(validation):
            # Accept-side consistency gate (L78 desk_grommet incident; FORGE_ACCEPT_CONSISTENCY=1,
            # default OFF): a score-4 whose own critique demands brief-declared features no
            # instrument measured is a self-contradictory approval -- keep iterating instead;
            # the demands ride the next revision like any other critique fixes.
            from forge_mind.agentic.accept_consistency import accept_consistency_block

            consistency_block = accept_consistency_block(critique, brief)
            if consistency_block is None:
                stopped_reason = "accepted"
                break
            validation.issues.append(consistency_block)
        if best is not None and _plateau_stop(no_improve, best.validation, best.critique):
            # The output has plateaued (no improvement for the patience window). Stop now and keep
            # the best so far -- more iterations on a stuck part only burn time + money. A part still
            # short a requested feature gets a LONGER window (see _plateau_stop): a dropped hole is a
            # concrete, fixable gap worth more tries than a part that has just stopped improving.
            stopped_reason = "plateau"
            break
        if i == max_iterations or total_cost >= budget_usd:
            stopped_reason = "budget" if total_cost >= budget_usd else "iteration_cap"
            break

        # --- revise ALWAYS from the best working candidate --------------------
        anchor = best if best is not None else None
        base = anchor.code if anchor is not None else code
        # Collapse-restart rung (FORGE_COLLAPSE_RESTART, dark; the L102 construction-collapse
        # class): a lineage that BUILDS every iteration but spans a sliver of the stated
        # envelope satisfies neither error_streak (no sandbox error) nor _fidelity_stuck
        # (validation never passes) -- measured: chest_plate reproduced identical collapsed
        # extents across iterations and upper_arm_l's op volume pinned at 48.9mm^3 for all 5.
        # Route it to the SAME fresh best-of-3 draft selector the other restart rungs use.
        if (
            collapse_restart_enabled()
            and envelope_collapsed(collapse_ratios)
            and brief is not None
            and not is_cadquery(base)
        ):
            try:
                selection = select_best_draft(
                    brief,
                    3,
                    work_dir / f"collapse_restart_{i:02d}",
                    client=coder_client,
                    author_model=author_model,
                    fabrication_mode=fabrication_mode,
                    sandbox_limits=sandbox_limits,
                    part_name=slug,
                    strategy_suffix=collapse_restart_note(
                        collapse_ratios[-1], brief.overall_envelope_mm, i + 1
                    ),
                )
            except Exception:
                stopped_reason = "author_failed"
                break
            total_cost += selection.cost_usd
            last_author_cost = selection.cost_usd
            code, approach, phase = (
                selection.best_code,
                f"collapse restart (best-of-3, {selection.n_built} built)",
                "restart",
            )
            feature_map = None
            # Review-probed defect: when EVERY built iteration collapsed, the kept incumbent
            # is itself a sliver -- keeping it (and the accrued plateau patience) makes an
            # envelope-recovered-but-still-red fresh draft lose the keep-best tie and lets
            # plateau ship the sliver. Drop the untrustworthy incumbent and its patience so
            # the fresh lineage competes cleanly; a mixed lineage keeps its genuine best.
            if lineage_all_collapsed(collapse_ratios):
                best = None
                no_improve = 0
            collapse_ratios.clear()  # fresh lineage: the old tail must not re-fire on it
            continue
        # Fidelity-restart rung (measured arm S, 2026-07-18): a validator-green part whose
        # critique names ABSENT brief features, after BOTH the plain revise and the salted
        # best-of-3 failed to move the critique, is anchored on a semantic attractor (the
        # L-bracket block: 8/8 samples returned the anchor's byte-identical STL from both
        # paths). Revising that code cannot escape -- author a FRESH massing instead, with a
        # strategy note derived verbatim from the critique. The fresh draft still runs the
        # full build/validate/critique/keep gauntlet, so it must BEAT the incumbent (via the
        # fidelity tie-break) to become the new anchor: no gate is weakened.
        if _fidelity_stuck(best, no_improve) and best is not None and best.critique is not None:
            # Best-of-3 fresh drafts, not one bare author call: measured (breadth sweep
            # 2026-07-18) single restart drafts died to hallucinated APIs / kernel fails in
            # 3 of 5 critic-blocked runs (NameError Ring/rotate, BRep command-not-done),
            # burning the iteration. The draft selector only forwards code that BUILT (when
            # any did) -- restart iterations stop being coin-flips on first-try build luck.
            try:
                selection = select_best_draft(
                    brief,
                    3,
                    work_dir / f"restart_{i:02d}",
                    client=coder_client,
                    author_model=author_model,
                    fabrication_mode=fabrication_mode,
                    sandbox_limits=sandbox_limits,
                    part_name=slug,
                    strategy_suffix=_fidelity_restart_note(best.critique, i + 1),
                )
            except Exception:
                stopped_reason = "author_failed"
                break
            total_cost += selection.cost_usd
            last_author_cost = selection.cost_usd
            code, approach, phase = (
                selection.best_code,
                f"fidelity restart {i + 1} (best-of-3, {selection.n_built} built)",
                "restart",
            )
            feature_map = None
            continue
        fb = (
            _improve_feedback(
                anchor.validation, anchor.critique, regressed, stuck_red_streak=stuck_red_streak
            )
            if anchor is not None
            else _combine_feedback(validation, critique)
        )
        if no_improve >= 1:
            fb = f"{_attempt_note(i + 1, no_improve)}\n\n{fb}"
        # Line-level failure attribution (research idea 6, bench row-18 plateau class): map the
        # validator's feature findings back to the AUTHOR's own PARAMS keys + code lines, so a
        # revision is a targeted edit instead of a re-diagnosis. Precision-first (silent unless
        # confident); prepended so the CODE MAP sits just above the findings it explains.
        if line_attribution_enabled():
            code_map = attribute_issues(
                (anchor.validation if anchor is not None else validation).issues, base, brief
            )
            if code_map:
                fb = f"{code_map}\n\n{fb}"
        # Reference shape-match advisory (Step 4, FORGE_REFERENCE_MODEL): a coarse pose/scale-
        # invariant proximity of the working best to the real reference model. Advisory context
        # for the author; fail-closed (no reference / unsampleable -> skipped).
        if reference_stl is not None and anchor is not None:
            shape_score = shape_similarity(anchor.stl, reference_stl)
            if shape_score is not None:
                fb = (
                    f"SHAPE MATCH: the overall form is {shape_score:.0%} similar to the real "
                    "reference "
                    "(a coarse proportion/elongation metric). Getting the silhouette and "
                    "proportions closer to the reference raises it.\n\n" + fb
                )
            # Localized shape-signature advisory: the coarse D2 scalar above saturates (~0.9 for
            # everything), so it also names WHICH aspect of the form is off and WHICH way to move
            # it -- measured concavity (too solid vs the reference) and axial asymmetry (missing
            # off-axis mass like counterweights). Report-only, fail-closed (None -> skipped).
            advisory = conformance_advisory(anchor.stl, reference_stl)
            if advisory is not None:
                fb = advisory + "\n\n" + fb
        # Hole-regression callout (the measured dropped-holes pattern): the JUST-TRIED code
        # erased through-holes the working best already had -- a min-wall or cosmetic fix
        # that deleted the bores. Name it explicitly so the next revision keeps the holes
        # instead of rediscovering the gap an iteration later.
        if (
            anchor is not None
            and validation.through_holes is not None
            and anchor.validation.through_holes is not None
            and validation.through_holes < anchor.validation.through_holes
        ):
            lost = anchor.validation.through_holes - validation.through_holes
            fb = (
                f"HOLE REGRESSION: your last attempt REMOVED {lost} through-hole(s) the working "
                f"version has ({anchor.validation.through_holes} -> {validation.through_holes}). "
                "Whatever you change next, re-apply EVERY bore after reshaping -- holes are part "
                "of the requested function, not decoration.\n\n" + fb
            )
        # Blast-radius advisory (FORGE_REPAIR_DIFF=1, default OFF): a revision whose volume
        # swung past the measured legitimate-progression band gets the change named in its
        # feedback -- the wall_hook 27.8x silent-rebuild class. Advisory only; never gates.
        if anchor is not None:
            from forge_mind.agentic.repair_diff import repair_diff_advisory

            diff_note = repair_diff_advisory(anchor.validation.volume_mm3, validation.volume_mm3)
            if diff_note is not None:
                fb = f"{diff_note}\n\n{fb}"
        # Result-plumbing advisory (ADR-0117, the measured shipped-the-base-plate class): the
        # JUST-TRIED build bound `result` to a much smaller solid than the script constructed.
        # Named explicitly so the next revision fixes the binding instead of redesigning.
        if sandbox.plumbing_note:
            fb = f"{sandbox.plumbing_note}\n\n{fb}"
        # Per-op measured consequence (ADR-0119, FORGE_OP_TRACE=1): what each builder op did
        # to volume / solid count / extents, so the revision reacts to the LINE that went
        # wrong instead of only the end state. Feedback only; absent unless the flag rode
        # the build that produced this sandbox result.
        if os.environ.get("FORGE_OP_TRACE") == "1" and sandbox.optrace:
            rows = [
                "L{}: vol={} solids={} ext={}{}".format(
                    e.get("line"),
                    e.get("volume"),
                    e.get("solids"),
                    e.get("extents"),
                    f" [{e['err']}]" if e.get("err") else "",
                )
                for e in sandbox.optrace[:24]
            ]
            fb = "OP TRACE (measured after each builder op):\n" + "\n".join(rows) + f"\n\n{fb}"
        # ADR-0133 fix 3 (FORGE_EDGE_FINISH=1): silent edge-finish degradation made
        # VISIBLE at the failure point. Study 0 measured 5/7 archived fillet-calling
        # shipped parts with zero toroidal faces -- safe_fillet no-oped and nothing saw
        # it. A skip or a >=50% back-off gets named to the reviser with the sizing and
        # scoping rules that make the next attempt succeed. Feedback only; never gates.
        # Events come from the ANCHOR (the kept code the reviser will edit) when one
        # exists -- review-caught twice: a rejected revision resets base to anchor.code,
        # so the DISCARDED attempt's events would advise on absent code; and an anchor
        # with an EMPTY list must still win over the discarded events (existence, not
        # list truthiness, selects the source).
        edge_events = anchor.edge_finish if anchor is not None else sandbox.edge_finish
        if os.environ.get("FORGE_EDGE_FINISH") == "1" and edge_events:
            degraded = _edge_finish_degraded(edge_events)
            if degraded:
                ef_rows = [
                    "- safe_{op}(requested {req}) on {n} edge(s) -> {out}".format(
                        op=e.get("op"),
                        req=e.get("requested"),
                        n=e.get("edges"),
                        out=(
                            "SKIPPED ENTIRELY (part shipped UN-rounded, no error raised)"
                            if e.get("realized") is None
                            else f"backed off to {e['realized']} (visually near no-op)"
                        ),
                    )
                    for e in degraded[:8]
                ]
                fb = (
                    "EDGE FINISH (measured in the sandbox -- these calls degraded SILENTLY):\n"
                    + "\n".join(ef_rows)
                    + "\nThe radius/length exceeds the local wall or the edge set spans "
                    "complex topology. Size cosmetic fillets 0.5-1 mm and keep radius <= 1/3 "
                    "of the smallest adjacent wall; NARROW the selection (vertical_edges(part), "
                    "group_by ring, filter_by radius) instead of part.edges(); keep bore "
                    "lead-ins as chamfer per the chamfered_bore_entry idiom.\n\n" + fb
                )
        # ADR-0120: verified idioms must be reachable from THIN-WALL / LIGAMENT findings,
        # not only count-gate findings and critique text -- L68 measured the junction
        # idiom never firing on the exact failure class it was written for. Scan the
        # assembled feedback PLUS the brief's feature names (the junction words usually
        # live there) and append at most one idiom.
        if "THIN-WALL" in fb or "LIGAMENT" in fb or "min_wall" in fb:
            from forge_mind.agentic.idioms import idiom_pointer

            names = " ".join(f.name for f in (brief.features or []))
            snippet = idiom_pointer(f"{fb} {names}")
            if snippet and snippet not in fb:
                fb = f"{fb}\n\n{snippet}"
        # Self-check measurements (FORGE_AUTHOR_SELFCHECK): the sandbox's measured deltas
        # for the code being revised -- the author's own numbers, appended verbatim.
        sc_block = selfcheck_mod.feedback_block(
            anchor.selfcheck if anchor is not None else sandbox.selfcheck
        )
        if sc_block:
            fb = f"{fb}\n\n{sc_block}"
        # Multimodal author (ADR-0066, FORGE_AUTHOR_VISION): hand the author a render of the build
        # it is revising -- its own six-view, stacked over the real reference when that flag is on
        # too -- so it revises WITH vision, not on lossy text alone. (None, False) when the flag is
        # off = byte-identical to the text-only revise call.
        author_image, image_has_reference = _author_vision_image(
            anchor, reference_stl, it_dir, brief.object_name
        )
        k = revision_best_of_n()
        if no_improve >= 1:
            # Measured (attractor-escape probe, 2026-07-18): SALTED candidates escape the
            # semantic attractor ~1-in-3 where a unique single prompt does not, and t=0.5
            # sampling adds hallucination risk without gain. A non-improving revision is
            # exactly the stuck case -- give it the multi-candidate treatment.
            k = max(k, 3)
        try:
            # fabrication_mode here too (it was missing on THIS site only): the PRINT_3D design
            # guidance must reach the main quality-revision path, not just the initial + repair.
            if k > 1:
                # Metric-steered best-of-N revision (ADR-0070): author k candidates from the same
                # base+feedback, build them in PARALLEL, keep the best by validity-tier-then-shape-
                # signature. The winner flows into the normal build/validate/critique/keep below.
                code, rev_cost, _n = best_revision(
                    brief,
                    base,
                    fb,
                    k=k,
                    client=repair_client,
                    author_model=author_model,
                    fabrication_mode=fabrication_mode,
                    image_png=author_image,
                    image_has_reference=image_has_reference,
                    reference_stl=reference_stl,
                    material=material,
                    sandbox_limits=sandbox_limits,
                    work_dir=it_dir / "candidates",
                    part_name=slug,
                )
                total_cost += rev_cost
                last_author_cost = rev_cost
                approach, phase = f"revise (best-of-{k})", "revise"
                feature_map = None
            else:
                authored, usage = _retry_once(
                    revise_dispatch,
                    brief,
                    base,
                    fb,
                    client=repair_client,
                    model=author_model,
                    fabrication_mode=fabrication_mode,
                    image_png=author_image,
                    image_has_reference=image_has_reference,
                )
                total_cost += usage.cost_usd
                last_author_cost = usage.cost_usd
                code, approach, phase = authored.code, authored.approach, "revise"
                feature_map = getattr(authored, "feature_map", None)
        except Exception:
            # GUARDED: end the part gracefully, keeping the best so far (never crash mid-loop).
            stopped_reason = "author_failed"
            break

    duration = time.monotonic() - t_start
    if best is None:
        return LoopResult(
            prompt=prompt,
            brief=brief,
            final_code=None,
            final_stl=None,
            final_step=None,
            final_render=None,
            validation=None,
            critique=None,
            geometry_passed=False,
            critic_approved=False,
            # honest reason: an author-stage failure is not the same as "never built"
            stopped_reason="author_failed" if stopped_reason == "author_failed" else "no_build",
            iterations=iterations,
            total_cost_usd=total_cost,
            total_duration_s=duration,
            drafts_tried=drafts_tried,
            drafts_built=drafts_built,
        )
    # Parameter polish (ADR-0073/0074, OPT-IN FORGE_PARAM_OPT): fit the final best's float
    # PARAMS against the reference form AND/OR the validator-derived targets (reference optional
    # since ADR-0074). BEFORE the hole-repair backstop on purpose -- polish rebuilds from CODE,
    # repair patches the MESH, so repair must stay last or its patch would be lost.
    if param_opt_polish_enabled(reference_stl is not None):
        best = _param_polish(
            best,
            reference_stl,
            brief,
            work_dir,
            iterations,
            fabrication_mode=fabrication_mode,
            material=material,
            sandbox_limits=sandbox_limits,
            part_name=slug,
            prompt=prompt,
            mass_target=mass_target,
        )
        duration = time.monotonic() - t_start  # include the polish in the honest total
    # GUARDED last resort (ADR-0033 Stage 2): the author finished still short its requested
    # through-holes -- cut them deterministically where placement is defensible, re-validate, and
    # keep the repair ONLY if the deficit closed cleanly. attempt_hole_repair returns None on any
    # doubt (multi-region brief, unstated dimension, failed re-validation), leaving best untouched.
    repaired = False
    if best.validation.features_missing:
        outcome = attempt_hole_repair(
            best.stl,
            best.step,
            brief,
            work_dir,
            fabrication_mode=fabrication_mode,
            material=material,
        )
        if outcome is not None:
            repaired = True
            render_path = None
            # `repaired_render` stays None unless THIS render succeeded: the display fallback
            # below may reuse the pre-repair image, and that image must never reach the critic
            # (review-caught -- it would re-approve geometry the repair replaced).
            repaired_render: Path | None = None
            try:
                render_target = work_dir / "repair" / "render.png"
                render_part_to_png(
                    outcome.stl,
                    render_target,
                    title=f"{brief.object_name} (repaired)",
                    # no caption: model-facing render (row-37 occlusion fix)
                )
                render_path = render_target
                repaired_render = render_target
            except Exception:
                render_path = best.render  # display only; NOT critique input
            fresh_critique, recritique_cost = _recritique_repaired(
                outcome,
                brief,
                repaired_render,
                client=vision_client,
                model=critic_model,
                work_dir=work_dir / "repair",
                reference_stl=reference_stl,
                budget_left_usd=budget_usd - total_cost,
            )
            total_cost += recritique_cost
            best = _Candidate(
                score=best.score,
                code=best.code,
                stl=outcome.stl,
                step=outcome.step,
                render=render_path,
                validation=outcome.validation,
                critique=fresh_critique,
            )
            # The re-critique is a real LLM round-trip: fold its latency into the honest total
            # (review-caught -- `duration` was captured before it, so every repaired part
            # under-reported total_duration_s by the whole critic call).
            duration = time.monotonic() - t_start

    # The accept-side consistency gate guards EVERY door the approval exits through: the
    # accept site above AND this final verdict. Measured escape (arm A1, desk_grommet):
    # the gate refused the early accept, the run capped at iteration_cap with the blocked
    # candidate still kept-best, and critic_approved read its score-4 critique -- the
    # false-PASS shipped anyway. Flag off = byte-identical (the block is None).
    from forge_mind.agentic.accept_consistency import accept_consistency_block as _acb

    result = LoopResult(
        prompt=prompt,
        brief=brief,
        final_code=best.code,
        final_stl=best.stl,
        final_step=best.step,
        final_render=best.render,
        validation=best.validation,
        critique=best.critique,
        geometry_passed=best.validation.passed,
        critic_approved=best.critique is not None
        and best.critique.looks_right
        and _acb(best.critique, brief) is None,
        stopped_reason=stopped_reason,
        iterations=iterations,
        total_cost_usd=total_cost,
        total_duration_s=duration,
        drafts_tried=drafts_tried,
        drafts_built=drafts_built,
        repaired_features=repaired,
    )
    # Designed finish for THIS path (ADR-0129 fix #7), BEFORE the cache save so a repeat
    # serves the finished part. Flag off / no ops / any failure = result unchanged.
    # `apply_finish=False` = a SUBPART call from produce_functional_part, whose own finish
    # stage owns polish with port exclusions + an assembly-fit baseline (review-caught:
    # double-finishing here would poison that baseline and pay twice).
    if apply_finish:
        result = _maybe_apply_single_part_finish(result, work_dir, planner_client, brief_model)
    if cache_key is not None:
        build_cache.save(cache_key, result)
    # 4090-flywheel intake (FORGE_TRAIN_LOG, default on, fail-open): every authored part
    # becomes a validator-labeled (prompt -> code) training row for the local CAD-author
    # model. One file append; can never break the run.
    from forge_mind.train_log import log_part

    log_part(result, material)
    return result


__all__ = ["IterationRecord", "LoopResult", "run_agentic_pipeline"]
