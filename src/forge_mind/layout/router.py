"""Generation-path ROUTER (ADR-0020) — hierarchical decompose-then-select.

The single ``GeneratedLayout`` wrapper held every spec and sat at Anthropic's compiled-
grammar ceiling (ADR-0019: a 3rd member is rejected). This router removes the ceiling:

  Stage A — ``classify``: a cheap structured-output call maps a Blueprint to a
            ``GenerationPath`` enum (a one-field schema -> a trivial grammar).
  Stage B — ``generate_for_path``: dispatch to the chosen path's OWN structured-output call,
            each emitting ONLY its spec (a small grammar per call). No single call holds all
            specs, so the ceiling is never approached and new paths cost nothing.

The specs are REUSED verbatim (ADR-0014 composable plate, ADR-0019 computed body); only the
EMISSION changes. ``computed_body`` is now a first-class routable path (the ADR-0019 unblock).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.blueprint.schema import Blueprint
from forge_mind.config import BUDGETS, HAIKU, LAYOUTSPEC_MODEL
from forge_mind.layout.arm_interpreter import ArmSpec
from forge_mind.layout.composable_spec import ComposablePlateSpec
from forge_mind.layout.computed_interpreter import ComputedBodySpec
from forge_mind.layout.layoutspec import LayoutSpec
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client
from forge_mind.types import Vec3

_PROMPTS = Path(__file__).parent / "prompts"
_CLASSIFIER_PROMPT = (_PROMPTS / "path_classifier.md").read_text(encoding="utf-8")
_PLATE_TRAY_PROMPT = (_PROMPTS / "layoutspec_v3.md").read_text(encoding="utf-8")
_COMPUTED_PROMPT = (_PROMPTS / "computed_body.md").read_text(encoding="utf-8")
_ARM_PROMPT = (_PROMPTS / "motor_arm.md").read_text(encoding="utf-8")

_T = TypeVar("_T", bound=BaseModel)


class GenerationPath(StrEnum):
    """The generation paths the router selects between (ADR-0020). The plate family
    (l_bracket / adapter / fan) is ONE composable_plate spec (ADR-0014)."""

    COMPOSABLE_PLATE = "composable_plate"
    ELECTRONICS_TRAY = "electronics_tray"
    COMPUTED_BODY = "computed_body"
    MOTOR_ARM = "motor_arm"
    UNSUPPORTED = "unsupported"  # out of the buildable domain -> a clean refusal, not a part


class PathChoice(BaseModel):
    """Stage-A output — a single enum field, so the compiled grammar is trivial."""

    model_config = ConfigDict(extra="forbid")

    path: GenerationPath = Field(
        ..., description="The generation path that best fits the Blueprint."
    )


# A clear, demo-ready refusal for a request outside the buildable domain.
UNSUPPORTED_MESSAGE = (
    "I can't make that yet. I build validated mechanical parts: flat plates / brackets, "
    "enclosures / trays, structural members (struts / lattices), and motor mounts / arms."
)


class UnsupportedRequestError(Exception):
    """The classifier judged the request OUT OF the buildable domain. A clean refusal (the pipeline
    surfaces the message) -- NOT a failure to build and NOT a direct-DSL fallback. Carries the
    classify ``usage`` so cost is still reported. Deliberately not a LayoutError/LLMError, so the
    pipeline's fallback catch does not swallow it into a (wrong) direct-DSL part."""

    def __init__(
        self, message: str = UNSUPPORTED_MESSAGE, *, usage: LLMUsage | None = None
    ) -> None:
        super().__init__(message)
        self.usage = usage


def _hard_max(blueprint: Blueprint) -> Vec3 | None:
    """The user-stated hard-max envelope (intent -> blueprint) as a Vec3, or None if unset."""
    m = blueprint.max_envelope_mm
    return Vec3(x=m.x, y=m.y, z=m.z) if m is not None else None


class RoutedLayoutGenerator:
    """Blueprint -> LayoutSpec via classify-then-generate. Drop-in for LayoutSpecGenerator in
    the pipeline, but each LLM call carries only ONE small spec (never the full wrapper)."""

    def __init__(
        self,
        client: LLMClientLike | None = None,
        classifier_model: str | None = None,
        path_model: str | None = None,
    ) -> None:
        self._client = client or make_client()
        self._classifier_model = classifier_model or HAIKU
        self._path_model = path_model or LAYOUTSPEC_MODEL

    # -- Stage A -------------------------------------------------------------
    def classify(self, blueprint: Blueprint) -> tuple[GenerationPath, LLMUsage]:
        choice, usage = self._client.generate_structured(
            system=_CLASSIFIER_PROMPT,
            user=(
                "Choose the generation path for this Blueprint by its primary role.\n\n"
                f"Blueprint (JSON):\n{blueprint.model_dump_json(indent=2)}\n"
            ),
            output_model=PathChoice,
            model=self._classifier_model,
            max_tokens=BUDGETS["router"].max_tokens,
        )
        return choice.path, usage

    # -- Stage B -------------------------------------------------------------
    def _emit(
        self, spec_model: type[_T], system: str, path: GenerationPath, blueprint: Blueprint
    ) -> tuple[_T, LLMUsage]:
        user = (
            f"This Blueprint was routed to the '{path.value}' path. Emit a "
            f"{spec_model.__name__} directly (the output schema IS that spec; there is no "
            "wrapper). Copy envelope, material, and min_wall from the Blueprint; leave "
            "dims/offsets null so the interpreter derives them.\n\n"
            f"Blueprint (JSON):\n{blueprint.model_dump_json(indent=2)}\n"
        )
        return self._client.generate_structured(
            system=system,
            user=user,
            output_model=spec_model,
            model=self._path_model,
            max_tokens=BUDGETS["layoutspec"].max_tokens,
        )

    def generate_for_path(
        self, path: GenerationPath, blueprint: Blueprint
    ) -> tuple[LayoutSpec, LLMUsage]:
        if path is GenerationPath.UNSUPPORTED:
            raise UnsupportedRequestError()
        # The user hard max (if any) is plumbed from intent -> blueprint -> spec deterministically,
        # so the grow-don't-reject interpreters honour it. None -> grow the envelope freely.
        hard_max = _hard_max(blueprint)
        if path is GenerationPath.COMPOSABLE_PLATE:
            plate, usage = self._emit(ComposablePlateSpec, _PLATE_TRAY_PROMPT, path, blueprint)
            return plate.model_copy(update={"max_envelope_mm": hard_max}), usage
        if path is GenerationPath.ELECTRONICS_TRAY:
            tray, usage = self._emit(ElectronicsTraySpec, _PLATE_TRAY_PROMPT, path, blueprint)
            # Same hard-max plumbing as composable/computed (ADR-0035 A2): the user-stated max
            # is deterministic intent, never LLM-picked; the interpreter's grow-don't-reject
            # honours it. The LLM-emitted value (if any) is overwritten.
            return tray.model_copy(update={"max_envelope_mm": hard_max}), usage
        if path is GenerationPath.MOTOR_ARM:
            arm, usage = self._emit(ArmSpec, _ARM_PROMPT, path, blueprint)
            # Part A (span-sizing): the arm's envelope is DERIVED deterministically from the
            # categorical reach + the motor facts, never LLM-sized. Force it null so interpret_arm
            # always derives it -- independent of whatever envelope the LLM happened to emit.
            return arm.model_copy(update={"envelope_mm": None}), usage
        computed, usage = self._emit(ComputedBodySpec, _COMPUTED_PROMPT, path, blueprint)
        # Null the LLM envelope -> derive it from the categorical size_class (the LLM picks no numbers).
        return computed.model_copy(update={"envelope_mm": None, "max_envelope_mm": hard_max}), usage

    # -- combined -----------------------------------------------------------
    def __call__(self, blueprint: Blueprint) -> tuple[LayoutSpec, list[LLMUsage]]:
        path, u_classify = self.classify(blueprint)
        if path is GenerationPath.UNSUPPORTED:
            # Refuse cleanly, carrying the classify cost; no spec is emitted (no part).
            raise UnsupportedRequestError(usage=u_classify)
        spec, u_generate = self.generate_for_path(path, blueprint)
        return spec, [u_classify, u_generate]


# Authoritative list of EVERY structured-output schema the router emits, with the model it
# runs on — the single source the grammar smoke-check tests. Kept in sync with
# ``generate_for_path`` (test_router asserts every path's spec is covered here).
ROUTER_SCHEMAS: list[tuple[str, str, type[BaseModel]]] = [
    ("classifier:PathChoice", HAIKU, PathChoice),
    ("path:composable_plate", LAYOUTSPEC_MODEL, ComposablePlateSpec),
    ("path:electronics_tray", LAYOUTSPEC_MODEL, ElectronicsTraySpec),
    ("path:computed_body", LAYOUTSPEC_MODEL, ComputedBodySpec),
    ("path:motor_arm", LAYOUTSPEC_MODEL, ArmSpec),
]


def _is_grammar_too_large(err: Exception) -> bool:
    msg = str(err).lower()
    return "grammar" in msg and "large" in msg


def check_grammar(client: LLMClientLike | None = None) -> list[tuple[str, bool, str]]:
    """Real-API grammar-compile smoke-check (ADR-0020, the durable meta-fix).

    The offline char-count proxy was unfaithful and the ``not llm`` suite is blind to
    compiled-grammar errors (they surface only on real API calls). This asks each router
    schema to generate with a tiny ``max_tokens`` and reports whether the compiled grammar
    FITS. A rejected compile costs ~$0. A NON-grammar error (e.g. the dummy prompt can't fill
    required fields) still means the grammar compiled -> the schema FITS. Returns
    (label, fits, detail); ``fits`` is False ONLY on a grammar-too-large rejection."""
    client = client or make_client()
    results: list[tuple[str, bool, str]] = []
    for label, model, schema in ROUTER_SCHEMAS:
        try:
            client.generate_structured(
                system="Return an empty object.",
                user="{}",
                output_model=schema,
                model=model,
                max_tokens=16,
            )
            results.append((label, True, "fits (compiled + generated)"))
        except Exception as exc:  # classify grammar-ceiling vs downstream failure
            if _is_grammar_too_large(exc):
                results.append((label, False, "OVER: compiled grammar too large"))
            else:
                results.append(
                    (label, True, f"fits (grammar compiled; downstream {type(exc).__name__})")
                )
    return results


__all__ = [
    "ROUTER_SCHEMAS",
    "UNSUPPORTED_MESSAGE",
    "GenerationPath",
    "PathChoice",
    "RoutedLayoutGenerator",
    "UnsupportedRequestError",
    "check_grammar",
]
