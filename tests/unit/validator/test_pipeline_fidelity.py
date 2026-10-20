"""The intent-fidelity gate is WIRED into the pipeline (Step 2), deterministically (no LLM).

Monkeypatches the four generation stages so the Blueprint requests a heat-set insert while the
interpreted PartProgram carries none: run_pipeline must REFUSE (gen_path 'infidelity', a refusal
message, no validation report) instead of compiling + shipping the silently-wrong part. A faithful
program must pass the gate through to compile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind import pipeline as pl
from forge_mind.blueprint.schema import (
    Blueprint,
    BlueprintVec3,
    FeatureSpec,
    InterfaceSpec,
    RequestedInterface,
)
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.intent.schema import IntentSpec
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


def _blueprint() -> Blueprint:
    return Blueprint(
        name="p",
        summary="a plate with a heat-set insert",
        envelope_mm=BlueprintVec3(x=60, y=40, z=10),
        print_orientation=BlueprintVec3(x=0, y=0, z=1),
        material="PLA",
        min_wall_mm=4.0,
        max_overhang_deg=45.0,
        features=[
            FeatureSpec(
                name="body",
                kind="plate",
                purpose="host",
                approx_size_mm=BlueprintVec3(x=60, y=40, z=4),
            )
        ],
        required_interfaces=[
            InterfaceSpec(
                name="ins",
                interface_type=RequestedInterface.HEAT_SET_INSERT,
                on_feature="body",
                facing="+Z",
                component_id="M3",
            )
        ],
    )


def _program(*, with_hw: bool) -> PartProgram:
    hw = []
    if with_hw:
        from forge_mind.validator.conformance import HardwareFeature

        hw = [HardwareFeature(category="heat_set_insert", key="M3", nominal_mm=4.0, where="ins")]
    return PartProgram(
        metadata=Metadata(name="p"),
        operations=[
            Box(
                op_id="plate",
                mode=OpMode.ADD,
                position=Vec3(x=0, y=0, z=2),
                width=60,
                depth=40,
                height=4,
            )
        ],
        interfaces=[],
        hardware_features=hw,
        constraints=Constraints(min_wall_thickness=4.0),
    )


def _patch_stages(monkeypatch: pytest.MonkeyPatch, *, program: PartProgram) -> None:
    class _Intent:
        def __init__(self, *_a: object, **_k: object) -> None: ...
        def __call__(self, _prompt: str) -> tuple[IntentSpec, LLMUsage]:
            return IntentSpec(part_family="bracket", summary="s"), _usage()

    class _BP:
        def __init__(self, *_a: object, **_k: object) -> None: ...
        def __call__(self, _intent: IntentSpec) -> tuple[Blueprint, LLMUsage]:
            return _blueprint(), _usage()

    class _Router:
        def __init__(self, *_a: object, **_k: object) -> None: ...
        def __call__(self, _bp: Blueprint) -> tuple[object, list[LLMUsage]]:
            return object(), [_usage()]

    monkeypatch.setattr(pl, "IntentParser", _Intent)
    monkeypatch.setattr(pl, "BlueprintGenerator", _BP)
    monkeypatch.setattr(pl, "RoutedLayoutGenerator", _Router)
    monkeypatch.setattr(pl, "interpret_layout", lambda _spec: program)


def test_pipeline_refuses_a_dropped_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stages(monkeypatch, program=_program(with_hw=False))
    result = pl.run_pipeline("a plate with a heat-set insert", tmp_path, use_layoutspec=True)
    assert result.gen_path == "infidelity"
    assert result.refusal is not None and "heat-set insert" in result.refusal
    assert result.report is None  # never compiled / validated -> not shipped
    assert not result.passed


def test_pipeline_passes_a_faithful_part_through_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_stages(monkeypatch, program=_program(with_hw=True))

    def _boom(_program: PartProgram, _out: Path) -> object:
        raise RuntimeError("reached compile")

    monkeypatch.setattr(pl, "compile_program", _boom)
    result = pl.run_pipeline("a plate with a heat-set insert", tmp_path, use_layoutspec=True)
    # The faithful part is NOT refused: it flows past the gate to compile (our sentinel).
    assert result.refusal is None
    assert result.gen_path == "layoutspec"
    assert result.failed_stage == "compile" and "reached compile" in (result.error or "")
