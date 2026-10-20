"""LLM end-to-end proof (ADR-0017): a NEMA 23 Blueprint dispatches the right motor
component AND derives valid geometry from NULL dims -- the full registry multiplier path.

Proves the LLM picks `component_id=nema23` (no silent NEMA 17 fallback), leaves base_dims +
offsets null, and the interpreter derives + the validator certifies the 47.14 mm pattern --
zero explicit geometry. Marked `llm`; skipped without a key. Run:

    uv run pytest -m llm tests/integration/test_nema23_dispatch_llm.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forge_mind.blueprint.schema import Blueprint
from forge_mind.layout.composable_spec import BaseArchetype, ComposablePlateSpec, InterfaceRef
from forge_mind.layout.generator import LayoutSpecGenerator
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.llm.client import api_key_available

pytestmark = pytest.mark.llm

REPO = Path(__file__).resolve().parents[2]
BLUEPRINT = REPO / "tests" / "fixtures" / "blueprints" / "nema23_adapter.blueprint.json"


@pytest.mark.skipif(not api_key_available(), reason="no ANTHROPIC_API_KEY")
def test_nema23_dispatches_and_derives_with_null_dims() -> None:
    blueprint = Blueprint.model_validate(json.loads(BLUEPRINT.read_text()))

    spec, usage = LayoutSpecGenerator()(blueprint)
    assert isinstance(spec, ComposablePlateSpec)
    assert spec.base is BaseArchetype.FLAT_PLATE  # a NEMA 23 motor uses flat_plate (not l_bracket)

    # Dispatch: the motor face must carry component_id=nema23 -- NO silent NEMA 17 fallback.
    motor = next(p for p in spec.interfaces if p.ref is InterfaceRef.NEMA17_FACE)
    assert motor.component_id == "nema23", f"expected nema23 dispatch, got {motor.component_id!r}"

    # The required 3030 extrusion must be PRESENT and dispatched to ext_3030 (M6) -- not
    # dropped (a 1-interface plate would still validate), and not defaulted to a 2020 (M5,
    # the deferred M5/M5 combo that would fail to interpret). Both halves of the combo.
    ext = next((p for p in spec.interfaces if p.ref is InterfaceRef.EXTRUSION_2020), None)
    assert ext is not None and ext.component_id == "ext_3030", (
        f"expected an ext_3030 extrusion, got {ext.component_id if ext else 'NO extrusion'!r}"
    )

    # The LLM left geometry to the interpreter (null dims/offsets) -- the derive path.
    assert spec.base_dims is None
    assert all(p.offset_mm is None for p in spec.interfaces)

    # Derives valid geometry + certifies BOTH interfaces (47 mm motor pattern + M6 T-nut
    # pair), with zero explicit dims.
    result = run_layout_pipeline(spec, Path(tempfile.mkdtemp()))
    assert result.error is None, result.error
    assert result.report is not None and result.passed, (
        result.report.failed_check_ids() if result.report else "no report"
    )
    cert = {c.id: c.status.value for c in result.report.checks}
    assert any(
        cid.endswith(".bolt_pattern_47mm_square") and st == "pass" for cid, st in cert.items()
    ), f"NEMA 23 47mm pattern not certified: {cert}"
    assert any(cid.endswith(".two_m5_holes") and st == "pass" for cid, st in cert.items()), (
        f"3030 extrusion T-nut pair not certified: {cert}"
    )
    assert not any("bolt_pattern_31mm_square" in cid for cid in cert), (
        "silent NEMA 17 fallback -- the 31mm pattern was certified, not NEMA 23's 47mm"
    )
    assert usage.cost_usd > 0
