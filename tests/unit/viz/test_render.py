"""Test the headless renderer produces a valid PNG from a compiled part."""

from __future__ import annotations

from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.viz.render import render_part_to_png

# PNG magic bytes.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_render_produces_valid_png(tmp_path: Path) -> None:
    program = PartProgram(
        metadata=Metadata(name="plate_with_hole"),
        operations=[
            Box(op_id="plate", width=20.0, depth=20.0, height=4.0, position=Vec3(z=2.0)),
            Cylinder(
                op_id="hole",
                mode=OpMode.SUBTRACT,
                radius=3.0,
                height=4.0,
                position=Vec3(z=2.0),
            ),
        ],
    )
    model = compile_program(program, tmp_path)

    png = render_part_to_png(model.stl_path, tmp_path / "part.png", title="test part")
    assert png.exists()
    data = png.read_bytes()
    assert data[:8] == _PNG_MAGIC, "output is not a PNG"
    assert len(data) > 5000, "PNG suspiciously small (likely blank)"


def _png_height(path: Path) -> int:
    # PNG IHDR: width at bytes 16-20, height at 20-24 (big-endian) — no PIL needed
    return int.from_bytes(path.read_bytes()[20:24], "big")


def test_long_caption_grows_the_figure_instead_of_overlapping(tmp_path: Path) -> None:
    """The row-37 occlusion fix, pinned: a long prompt-as-caption must EXPAND the figure into
    its own band below the panels — never overdraw them. Observable: the PNG gets strictly
    taller as the caption grows (the geometry area is constant)."""
    program = PartProgram(
        metadata=Metadata(name="cap"),
        operations=[Box(op_id="plate", width=20.0, depth=20.0, height=4.0, position=Vec3(z=2.0))],
    )
    model = compile_program(program, tmp_path)
    clean = render_part_to_png(model.stl_path, tmp_path / "clean.png", title="t")
    short = render_part_to_png(
        model.stl_path, tmp_path / "short.png", title="t", caption="a one-line prompt"
    )
    long_caption = "\n".join(
        f"- hard constraint line {i}: some measured fact {i} mm held exactly" for i in range(22)
    )
    long = render_part_to_png(
        model.stl_path, tmp_path / "long.png", title="t", caption=long_caption
    )
    h_clean, h_short, h_long = _png_height(clean), _png_height(short), _png_height(long)
    assert h_short > h_clean, "a caption must add a band below the panels"
    assert h_long > h_short + 100, "a 22-line caption must grow the band, not flood the panels"
