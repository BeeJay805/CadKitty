"""compile_program must REJECT an empty / zero-volume body, not return a phantom part.

Audit L-COMP/M6: an engulfing subtract yields a non-None but EMPTY Compound (volume 0). The old
`body is None` guard missed it, so compile_program returned a part with no STL written, which broke
the grader's `except CompileError` contract downstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.compiler.compile import CompileError, compile_program
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Vec3


def test_engulfing_subtract_raises_compile_error(tmp_path: Path) -> None:
    program = PartProgram(
        metadata=Metadata(name="engulfed"),
        operations=[
            Box(
                op_id="solid",
                mode=OpMode.ADD,
                width=10.0,
                depth=10.0,
                height=10.0,
                position=Vec3(z=5.0),
            ),
            # a far larger subtract fully containing the solid -> empty result (volume 0), not None
            Box(
                op_id="cut",
                mode=OpMode.SUBTRACT,
                width=40.0,
                depth=40.0,
                height=40.0,
                position=Vec3(z=5.0),
            ),
        ],
    )
    with pytest.raises(CompileError):
        compile_program(program, tmp_path)
