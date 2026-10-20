"""ADR-0020 router — deterministic dispatch tests (a fake client; no real LLM).

Proves Stage A (classify) runs first on the cheap model, then Stage B emits ONLY the chosen
path's spec (its own small-grammar call). The real-API grammar fit is covered separately by
the opt-in grammar smoke-check (tests/integration/test_grammar_smoke.py).
"""

from __future__ import annotations

from typing import Any

import pytest

from forge_mind.blueprint.schema import (
    Blueprint,
    BlueprintVec3,
    FeatureSpec,
    InterfaceSpec,
    RequestedInterface,
)
from forge_mind.config import HAIKU, LAYOUTSPEC_MODEL
from forge_mind.dsl.computed_ops import FacetedStrutOp
from forge_mind.layout.arm_interpreter import ArmSpec
from forge_mind.layout.composable_spec import ComposablePlateSpec
from forge_mind.layout.computed_interpreter import ComputedBodySpec
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.layout.router import (
    GenerationPath,
    PathChoice,
    RoutedLayoutGenerator,
    UnsupportedRequestError,
)
from forge_mind.llm.cost import LLMUsage
from forge_mind.types import Vec3


def _usage() -> LLMUsage:
    return LLMUsage(
        model="t",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.0,
    )


def _dummy_spec() -> ComputedBodySpec:
    return ComputedBodySpec(
        name="d",
        envelope_mm=Vec3(x=100, y=20, z=20),
        min_wall_mm=3.0,
        op=FacetedStrutOp(cross_section="hexagonal"),
    )


class _FakeClient:
    """Records (output_model, model) per call; returns a canned PathChoice then a dummy spec."""

    def __init__(self, path: GenerationPath) -> None:
        self.path = path
        self.calls: list[tuple[type, str]] = []

    def generate_structured(
        self, *, output_model: type, model: str, **_kw: Any
    ) -> tuple[Any, LLMUsage]:
        self.calls.append((output_model, model))
        if output_model is PathChoice:
            return PathChoice(path=self.path), _usage()
        return _dummy_spec(), _usage()


def _bp() -> Blueprint:
    return Blueprint(
        name="x",
        summary="s",
        envelope_mm=BlueprintVec3(x=100, y=20, z=20),
        print_orientation=BlueprintVec3(x=0, y=0, z=1),
        material="PLA",
        min_wall_mm=3.0,
        max_overhang_deg=45.0,
        features=[
            FeatureSpec(
                name="f", kind="strut", purpose="p", approx_size_mm=BlueprintVec3(x=100, y=14, z=14)
            )
        ],
        required_interfaces=[
            InterfaceSpec(
                name="i", interface_type=RequestedInterface.M5_TNUT, on_feature="f", facing="-X"
            )
        ],
    )


def test_path_enum_has_all_paths() -> None:
    assert set(GenerationPath) == {
        GenerationPath.COMPOSABLE_PLATE,
        GenerationPath.ELECTRONICS_TRAY,
        GenerationPath.COMPUTED_BODY,
        GenerationPath.MOTOR_ARM,
        GenerationPath.UNSUPPORTED,
    }


def test_unsupported_classification_raises_clean_refusal() -> None:
    # classifier -> UNSUPPORTED: the router refuses (no spec emitted) and carries the classify cost,
    # so the pipeline can surface a clean refusal instead of building a wrong part.
    fake = _FakeClient(GenerationPath.UNSUPPORTED)
    gen = RoutedLayoutGenerator(client=fake)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedRequestError) as exc_info:
        gen(_bp())
    assert fake.calls == [(PathChoice, HAIKU)]  # only classify ran; no path spec emitted
    assert exc_info.value.usage is not None  # classify cost carried on the refusal


def test_generate_for_path_emits_only_that_paths_spec() -> None:
    expected = {
        GenerationPath.COMPOSABLE_PLATE: ComposablePlateSpec,
        GenerationPath.ELECTRONICS_TRAY: ElectronicsTraySpec,
        GenerationPath.COMPUTED_BODY: ComputedBodySpec,
        GenerationPath.MOTOR_ARM: ArmSpec,
    }
    for path, spec_model in expected.items():
        fake = _FakeClient(path)
        gen = RoutedLayoutGenerator(client=fake)  # type: ignore[arg-type]
        gen.generate_for_path(path, _bp())
        # exactly one call, with output_model == this path's spec (its own small grammar).
        assert fake.calls == [(spec_model, LAYOUTSPEC_MODEL)]


def test_call_classifies_then_generates() -> None:
    fake = _FakeClient(GenerationPath.COMPUTED_BODY)
    gen = RoutedLayoutGenerator(client=fake)  # type: ignore[arg-type]
    spec, usages = gen(_bp())
    # Stage A (PathChoice on the cheap model) THEN Stage B (the path spec on the path model).
    assert fake.calls[0] == (PathChoice, HAIKU)
    assert fake.calls[1] == (ComputedBodySpec, LAYOUTSPEC_MODEL)
    assert isinstance(spec, ComputedBodySpec)
    assert len(usages) == 2  # classify + generate, both reported
