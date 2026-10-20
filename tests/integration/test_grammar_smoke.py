"""ADR-0020 grammar-compile smoke-check — the durable real-API gate.

The default ``-m "not llm"`` suite is BLIND to compiled-grammar errors (they surface only on
real API calls), which is exactly how ADR-0019 shipped a latent break. The real-API check
below is marked ``llm`` so it is deselected by the default suite but runnable on demand
(``uv run pytest -m llm tests/integration/test_grammar_smoke.py``). The drift guard above is
deterministic and runs always.
"""

from __future__ import annotations

import pytest

from forge_mind.layout.arm_interpreter import ArmSpec
from forge_mind.layout.composable_spec import ComposablePlateSpec
from forge_mind.layout.computed_interpreter import ComputedBodySpec
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.layout.router import ROUTER_SCHEMAS, GenerationPath, check_grammar
from forge_mind.llm.client import api_key_available


def test_router_schemas_cover_every_path() -> None:
    # Drift guard (deterministic): adding a path without covering its spec in ROUTER_SCHEMAS
    # would silently exclude it from the grammar check. Catch that here.
    covered = {schema for _label, _model, schema in ROUTER_SCHEMAS}
    for spec in (ComposablePlateSpec, ElectronicsTraySpec, ComputedBodySpec, ArmSpec):
        assert spec in covered, f"{spec.__name__} not covered by the grammar smoke-check"
    # UNSUPPORTED is a refusal path that emits NO spec, so it has no ROUTER_SCHEMAS entry; every
    # GENERATING path must (a new generating path added without a schema still trips this guard).
    generating_paths = set(GenerationPath) - {GenerationPath.UNSUPPORTED}
    assert len(ROUTER_SCHEMAS) == 1 + len(generating_paths)  # 1 classifier + 1 per generating path


@pytest.mark.llm
@pytest.mark.skipif(not api_key_available(), reason="no ANTHROPIC_API_KEY")
def test_all_router_schemas_compile_against_real_api() -> None:
    results = check_grammar()
    over = [label for label, fits, _detail in results if not fits]
    assert not over, f"schemas over the compiled-grammar ceiling: {over}"
