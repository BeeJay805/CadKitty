"""End-to-end pipeline: natural-language prompt -> validated part.

    text -> IntentParser -> Blueprint -> DSLGenerator -> compile -> validate

No reviser yet (Phase 7). Each stage's model is overridable for measurement.
Stage failures are captured, not raised, so a caller always gets a result it can
inspect (and the Phase 7 reviser can act on).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.blueprint.schema import Blueprint
from forge_mind.compiler.compile import CADModel, compile_program
from forge_mind.dsl.build_meta import build_meta
from forge_mind.dsl.generator import DSLGenerator
from forge_mind.dsl.program import PartProgram
from forge_mind.intent.parser import IntentParser
from forge_mind.intent.schema import IntentSpec
from forge_mind.kit import Kit, build_kit
from forge_mind.layout.interpreter import InfeasibleRequestError, LayoutError
from forge_mind.layout.layoutspec import LayoutSpec, interpret_layout
from forge_mind.layout.router import RoutedLayoutGenerator, UnsupportedRequestError
from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.validator.fidelity import check_fidelity
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate


@dataclass
class PipelineResult:
    prompt: str
    intent: IntentSpec | None = None
    blueprint: Blueprint | None = None
    # The LayoutSpec the router emitted, when the layoutspec path ran. This is the part's
    # PARAMETRIC MODEL (ADR-0037): persisted by the API layer and replayed by /patch, so an
    # edit changes one field and re-derives instead of re-cutting wholesale.
    spec: LayoutSpec | None = None
    program: PartProgram | None = None
    model: CADModel | None = None
    report: ValidationReport | None = None
    usages: list[LLMUsage] = field(default_factory=list)
    failed_stage: str | None = None
    error: str | None = None
    # Which Blueprint->program path produced the part. "unknown" until a path
    # actually runs, so a trial that dies at intent/blueprint (before any
    # generation path) is not falsely attributed to one.
    gen_path: str = "unknown"
    # When use_layoutspec is set but the router path declined (LLMError) or the chosen spec was
    # infeasible (LayoutError), the part falls back to direct-DSL. This records WHY (the caught
    # exception type + message) so the fallback is observable instead of a silent pass.
    fallback_reason: str | None = None
    # Set when the request is OUT OF DOMAIN (classifier -> unsupported) or genuinely infeasible: a
    # clean human REFUSAL -- no fallback, no part. Distinct from `error` (an unexpected failure).
    refusal: str | None = None
    # The buildable-kit outputs (BOM + make-instructions) for the generated part: what to buy + how
    # to fabricate it. Attached for any generated part; None for a refusal (no part) or if kit
    # assembly itself failed. Reported, never a gate -- additive, derived from the program.
    kit: Kit | None = None

    @property
    def passed(self) -> bool:
        return self.report is not None and self.report.passed

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usages)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.usages)


def run_pipeline(
    prompt: str,
    output_dir: Path,
    *,
    intent_model: str | None = None,
    blueprint_model: str | None = None,
    dsl_model: str | None = None,
    use_layoutspec: bool = False,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> PipelineResult:
    """Run the full pipeline. Never raises on stage failure; records it.

    When ``use_layoutspec`` is set, the Blueprint->program step first tries the
    deterministic LayoutSpec path (the ADR-0020 generation-path ROUTER: classify ->
    per-path structured-output call -> interpret_layout), which is far more reliable on the
    part families it covers, and falls back to the direct DSLGenerator when the blueprint
    matches no known pattern / a path declines (LLMError) or the chosen spec is infeasible
    (LayoutError). Default off so the analysis harness keeps measuring the direct-DSL path
    unchanged.
    """
    result = PipelineResult(prompt=prompt)

    # Optional progress sink for the streaming endpoint. Swallows callback errors so streaming can
    # never break the pipeline; a no-op when not streaming (every existing caller).
    def _emit(kind: str, payload: dict[str, Any]) -> None:
        if on_event is None:
            return
        with contextlib.suppress(Exception):
            on_event(kind, payload)

    _emit("phase", {"key": "intent", "status": "active"})
    try:
        intent, u = IntentParser(model=intent_model)(prompt)
        result.intent = intent
        result.usages.append(u)
    except Exception as exc:
        result.failed_stage = "intent"
        result.error = str(exc)
        return result

    _emit("phase", {"key": "intent", "status": "done"})
    _emit("phase", {"key": "plan", "status": "active"})
    try:
        blueprint, u = BlueprintGenerator(model=blueprint_model)(intent)
        result.blueprint = blueprint
        result.usages.append(u)
    except Exception as exc:
        result.failed_stage = "blueprint"
        result.error = str(exc)
        return result
    _emit("phase", {"key": "plan", "status": "done"})
    _emit("phase", {"key": "cad", "status": "active"})

    program: PartProgram | None = None
    if use_layoutspec:
        try:
            # ADR-0020: the generation-path ROUTER (classify -> per-path call) replaces the
            # single GeneratedLayout wrapper, removing the grammar ceiling and making
            # computed_body routable. Each call carries only one small spec.
            spec, usages = RoutedLayoutGenerator()(blueprint)
            result.usages.extend(usages)
            result.spec = spec
            program = interpret_layout(spec)
            result.gen_path = "layoutspec"
            # Intent-fidelity gate (no-false-PASS at the INTENT level): the deterministic path can
            # DROP a requested interface (a strut routed to computed_body cannot host inserts) or
            # MISMAP it, yet still produce geometrically-valid output. Refuse rather than ship a
            # silently-wrong part. Only the layoutspec path is gated -- it has the structured spec +
            # the hardware-aware program; the direct-DSL fallback is the legacy escape hatch.
            infidelity = check_fidelity(blueprint, program)
            if infidelity is not None:
                result.program = program  # keep it for diagnostics; passed stays False (no report)
                result.gen_path = "infidelity"
                result.refusal = infidelity
                return result
        except UnsupportedRequestError as exc:
            # The classifier judged the request OUT OF the buildable domain -> a CLEAN refusal: no
            # direct-DSL fallback, no part. Carry the classify cost if present.
            if exc.usage is not None:
                result.usages.append(exc.usage)
            result.gen_path = "unsupported"
            result.refusal = str(exc)
            return result
        except InfeasibleRequestError as exc:
            # The request is genuinely not physically buildable (a fan's bore-vs-bolt land, or it
            # cannot fit a user-stated hard max) -> a CLEAN refusal WITH the reason, NOT a direct-DSL
            # fallback that would build a wrong part. (usages were already recorded above.) Must
            # precede the (LLMError, LayoutError) catch since this is a LayoutError subclass.
            result.gen_path = "infeasible"
            result.refusal = f"Can't build that as specified: {exc}"
            return result
        except (LLMError, LayoutError) as exc:
            # Blueprint matches no known LayoutSpec pattern, or the spec is infeasible -> record
            # WHY, then leave program None so the direct-DSL block below runs (and labels
            # gen_path). Nothing regresses; the fallback is now observable instead of silent.
            result.fallback_reason = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            # Any other failure (e.g. transient API error) is captured, not raised.
            result.failed_stage = "layoutspec"
            result.error = f"{type(exc).__name__}: {exc}"
            return result

    if program is None:
        # Committed to the direct-DSL path; label it now so a DSL-stage failure is
        # attributed to direct-dsl, not left as the "unknown" default.
        result.gen_path = "direct-dsl"
        try:
            program, u = DSLGenerator(model=dsl_model)(blueprint)
            result.usages.append(u)
        except Exception as exc:
            result.failed_stage = "dsl"
            result.error = str(exc)
            return result

    result.program = program

    # Stream the op list the moment it exists (before the slower compile + validate), so the viewer
    # can reveal the part operation-by-operation while the rest of the pipeline finishes.
    _emit("program", build_meta(program) or {})

    # Buildable-kit outputs (BOM + make-it): additive, REPORTED, never a gate. Built now from the
    # program so even a later compile/validate failure still yields the parts list; refreshed with
    # the report after validation so the orientation note can use the printability verdict. A
    # refusal returned earlier (no part) carries no kit, per the contract. Best-effort: a kit
    # failure must never break the pipeline (mirrors the physics metric).
    try:
        result.kit = build_kit(program, material=blueprint.material)
    except Exception:
        result.kit = None

    try:
        model = compile_program(program, output_dir)
        result.model = model
        _emit("phase", {"key": "cad", "status": "done"})
        _emit("phase", {"key": "validate", "status": "active"})
    except Exception as exc:
        # Beyond CompileError: a disconnected (multi-body) result can make
        # build123d's STL export raise (e.g. ShapeList has no .wrapped). The
        # pipeline must record such failures, never crash the caller.
        result.failed_stage = "compile"
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    try:
        result.report = validate(model, program, material=blueprint.material)
    except Exception as exc:
        result.failed_stage = "validate"
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    _emit("phase", {"key": "validate", "status": "done"})
    if not result.report.passed:
        result.failed_stage = "validate"
    # Refresh the kit with the validation report so the make-it orientation note reflects the
    # actual printability verdict (the BOM is unchanged). Best-effort, as above.
    with contextlib.suppress(Exception):
        result.kit = build_kit(program, material=blueprint.material, report=result.report)
    return result


__all__ = ["PipelineResult", "run_pipeline"]
