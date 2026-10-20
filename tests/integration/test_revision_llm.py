"""Phase 7 LLM smoke: the LLMReviser repairs a known-failing part.

Builds a thin slab that fails only the wall-thickness check, runs the real
LLMReviser through the revise() loop, and asserts it proposes a valid patch that
clears the defect. Marked `llm`; skipped without ANTHROPIC_API_KEY. Run:

    uv run pytest -m llm tests/integration/test_revision_llm.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.llm.client import api_key_available
from forge_mind.revision.llm_reviser import LLMReviser
from forge_mind.revision.reviser import revise
from forge_mind.types import Vec3
from forge_mind.validator.run import validate

pytestmark = pytest.mark.llm

CAP = (60.0, 60.0, 40.0)


@pytest.mark.skipif(not api_key_available(), reason="no ANTHROPIC_API_KEY")
def test_llm_reviser_repairs_thin_wall(tmp_path: Path) -> None:
    program = PartProgram(
        metadata=Metadata(name="thin_slab"),
        operations=[
            Box(op_id="slab", mode=OpMode.ADD, position=Vec3(z=1.0), width=40, depth=40, height=2)
        ],
        constraints=Constraints(min_wall_thickness=4.0, max_bbox=Vec3(x=60, y=60, z=40)),
    )
    model = compile_program(program, tmp_path / "start")
    report = validate(model, program)
    assert "features.min_wall_thickness" in report.failed_check_ids()

    result = revise(
        program,
        report,
        LLMReviser(),
        output_dir=tmp_path / "rev",
        envelope_cap_mm=CAP,
        render=False,
    )

    # The reviser must have produced at least one valid, applied patch.
    assert any(s.applied for s in result.steps)
    # Easy single-defect repair: expect the wall failure to be gone.
    assert "features.min_wall_thickness" not in result.end_failed_check_ids
    assert result.end_passed
