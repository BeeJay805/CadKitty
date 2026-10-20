"""Image asset metadata + Draw it! sketch storage (PNG preview + SVG export)."""

from __future__ import annotations

from pathlib import Path

from forge_mind.intake.assets import (
    render_sketch_png,
    save_image_asset,
    save_sketch_asset,
    sketch_to_svg,
)
from forge_mind.intake.schema import AssetType, Sketch, SketchPoint, SketchStroke


def _sketch() -> Sketch:
    return Sketch(
        canvas_width=1200,
        canvas_height=800,
        strokes=[
            SketchStroke(
                id="stroke_1",
                points=[SketchPoint(x=10, y=20, pressure=0.5), SketchPoint(x=100, y=120)],
                color="#1C1917",
                width=3,
            )
        ],
    )


def test_save_image_asset_records_metadata(tmp_path: Path) -> None:
    data = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"
    asset = save_image_asset(
        tmp_path,
        asset_id="a1",
        filename="drawing.png",
        data=data,
        asset_type=AssetType.TECHNICAL_DRAWING,
        content_type="image/png",
        base_url="/files/sess1",
    )
    assert asset.type == AssetType.TECHNICAL_DRAWING
    assert asset.size_bytes == len(data)
    assert asset.path is not None and Path(asset.path).is_file()
    assert asset.preview_url == "/files/sess1/a1_drawing.png"
    # An uploaded image's scale is NOT assumed exact.
    assert asset.scale_confirmed is False


def test_sketch_stored_as_raw_json_plus_preview(tmp_path: Path) -> None:
    sketch = _sketch()
    sk = save_sketch_asset(tmp_path, asset_id="sk1", sketch=sketch)
    # Raw stroke JSON is preserved verbatim (the source of truth).
    assert sk.sketch == sketch
    assert sk.sketch.strokes[0].points[0].x == 10
    # Preview PNG + SVG written; a sketch is shape-only until a dimension is given.
    assert sk.preview_path is not None and Path(sk.preview_path).is_file()
    assert sk.svg is not None and "<svg" in sk.svg
    assert sk.scale_confirmed is False


def test_render_sketch_png_returns_png_bytes() -> None:
    png = render_sketch_png(_sketch())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_sketch_to_svg_contains_polyline() -> None:
    svg = sketch_to_svg(_sketch())
    assert "<polyline" in svg
    assert 'stroke="#1C1917"' in svg
