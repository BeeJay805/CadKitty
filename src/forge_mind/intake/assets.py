"""Input-asset handling: persist uploaded images and ``Draw it!`` mouse sketches.

An uploaded image gets a metadata record (path, size, content type, preview URL); its scale is NOT
treated as exact until the user supplies a real dimension (``scale_confirmed`` stays False).

A sketch is stored three ways -- the raw stroke JSON (the source of truth), a rendered PNG preview
(so the planner's vision call and the UI have a raster), and an optional SVG export -- but it remains
a shape/intention reference, never converted straight to CAD geometry without a confirmed scale.
"""

from __future__ import annotations

import io
from pathlib import Path

from forge_mind.intake.schema import Asset, AssetType, Sketch, SketchAsset


def _file_url(base_url: str | None, *parts: str) -> str | None:
    if not base_url:
        return None
    return base_url.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)


def save_image_asset(
    directory: Path | str,
    *,
    asset_id: str,
    filename: str,
    data: bytes,
    asset_type: AssetType = AssetType.REFERENCE_IMAGE,
    content_type: str = "application/octet-stream",
    base_url: str | None = None,
) -> Asset:
    """Write the uploaded bytes under ``directory`` and return the Asset metadata record.

    ``base_url`` (e.g. '/files/<session>') makes ``preview_url`` a servable link; without it the
    record carries only the local ``path``. Scale is never assumed exact (``scale_confirmed=False``).
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or f"{asset_id}.bin"
    path = out_dir / f"{asset_id}_{safe_name}"
    path.write_bytes(data)
    return Asset(
        id=asset_id,
        type=asset_type,
        filename=safe_name,
        content_type=content_type,
        size_bytes=len(data),
        path=str(path),
        preview_url=_file_url(base_url, f"{asset_id}_{safe_name}"),
        scale_confirmed=False,
    )


def sketch_to_svg(sketch: Sketch) -> str:
    """Export the strokes as a standalone SVG (canvas coords map 1:1; SVG y-down matches the canvas)."""
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sketch.canvas_width}" '
        f'height="{sketch.canvas_height}" viewBox="0 0 {sketch.canvas_width} {sketch.canvas_height}">',
        f'<rect width="{sketch.canvas_width}" height="{sketch.canvas_height}" fill="#FBFAF7"/>',
    ]
    for stroke in sketch.strokes:
        if not stroke.points:
            continue
        pts = " ".join(f"{p.x:.1f},{p.y:.1f}" for p in stroke.points)
        lines.append(
            f'<polyline points="{pts}" fill="none" stroke="{stroke.color}" '
            f'stroke-width="{stroke.width:g}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    for ann in sketch.annotations:
        text = ann.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(
            f'<text x="{ann.x:.1f}" y="{ann.y:.1f}" font-family="sans-serif" '
            f'font-size="14" fill="#1C1917">{text}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def render_sketch_png(sketch: Sketch) -> bytes:
    """Render the strokes to a PNG (headless matplotlib). A best-effort raster preview for the UI and
    the planner's vision call -- not a measurement source."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    w_in = max(sketch.canvas_width, 1) / 100.0
    h_in = max(sketch.canvas_height, 1) / 100.0
    fig = Figure(figsize=(w_in, h_in), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, sketch.canvas_width)
    ax.set_ylim(sketch.canvas_height, 0)  # canvas y is top-down
    ax.axis("off")
    fig.patch.set_facecolor("#FBFAF7")
    ax.set_facecolor("#FBFAF7")

    for stroke in sketch.strokes:
        if not stroke.points:
            continue
        xs = [p.x for p in stroke.points]
        ys = [p.y for p in stroke.points]
        if len(xs) == 1:
            ax.plot(xs, ys, marker="o", markersize=stroke.width, color=stroke.color)
        else:
            ax.plot(xs, ys, color=stroke.color, linewidth=stroke.width, solid_capstyle="round")
    for ann in sketch.annotations:
        ax.text(ann.x, ann.y, ann.text, fontsize=12, color="#1C1917")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.getvalue()


def save_sketch_asset(
    directory: Path | str,
    *,
    asset_id: str,
    sketch: Sketch,
    write_svg: bool = True,
    base_url: str | None = None,
) -> SketchAsset:
    """Store a sketch: raw JSON is kept on the returned model; the rendered PNG (+ optional SVG) is
    written under ``directory``. The PNG render is best-effort -- a render failure still stores the
    raw strokes (the source of truth)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_path: str | None = None
    try:
        png = render_sketch_png(sketch)
        p = out_dir / f"{asset_id}_preview.png"
        p.write_bytes(png)
        preview_path = str(p)
    except Exception:
        preview_path = None  # raster is optional; the raw strokes are what matter

    svg: str | None = None
    if write_svg:
        svg = sketch_to_svg(sketch)
        (out_dir / f"{asset_id}.svg").write_text(svg, encoding="utf-8")

    return SketchAsset(
        id=asset_id,
        sketch=sketch,
        preview_path=preview_path,
        preview_url=_file_url(base_url, f"{asset_id}_preview.png") if preview_path else None,
        svg=svg,
        scale_confirmed=False,
    )


__all__ = [
    "render_sketch_png",
    "save_image_asset",
    "save_sketch_asset",
    "sketch_to_svg",
]
