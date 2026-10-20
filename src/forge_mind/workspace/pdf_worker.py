"""Standalone PDF observation worker. No project imports; run under python -I.

This is process isolation and resource supervision, not an OS security sandbox.
PDF canvas coordinates and raster pixels are presentation, never part dimensions.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from contextlib import closing
from importlib import import_module
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_DIMENSION = 8192
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_TEXT_CHARS = 120000
MAX_TEXT_BYTES = MAX_TEXT_CHARS * 4


class WorkerError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class BoundedBuffer(io.BytesIO):
    def write(self, data: Any) -> int:
        if self.tell() + memoryview(data).nbytes > MAX_IMAGE_BYTES:
            raise WorkerError("output_limit", "Rendered page exceeds the image byte limit")
        return super().write(data)


def observe(
    root: Path, page_number: int, scale: float, source_limit_bytes: int = MAX_SOURCE_BYTES
) -> dict[str, Any]:
    if type(source_limit_bytes) is not int or source_limit_bytes <= 0:
        raise WorkerError("invalid_request", "Source byte limit must be a positive integer")
    try:
        pdfium = import_module("pypdfium2")
        pillow = import_module("PIL")
        import_module("PIL.Image")
    except ImportError as exc:
        raise WorkerError("backend_unavailable", "PDFium or Pillow is unavailable") from exc

    flags = list(pdfium.PDFIUM_INFO.flags)
    if any("V8" in str(flag).upper() or "XFA" in str(flag).upper() for flag in flags):
        raise WorkerError("backend_unsupported", "V8/XFA PDFium builds are not supported")
    source = root / "source.pdf"
    if not 0 < source.stat().st_size <= source_limit_bytes:
        raise WorkerError("source_limit", "PDF source exceeds the byte limit or is empty")
    raw = source.read_bytes()
    if page_number < 1 or not math.isfinite(scale) or scale <= 0:
        raise WorkerError("invalid_request", "Page number and scale must be positive")
    try:
        document = pdfium.PdfDocument(raw)
    except pdfium.PdfiumError as exc:
        code = "encrypted" if getattr(exc, "err_code", None) == 4 else "malformed"
        raise WorkerError(code, "PDF could not be opened") from exc
    with document:
        page_count = len(document)
        if page_number > page_count:
            raise WorkerError("page_out_of_range", "Requested page is outside this PDF")
        with closing(document[page_number - 1]) as page:
            width, height = page.get_size()
            bounds = list(page.get_bbox())
            if not all(math.isfinite(v) for v in [width, height, *bounds]):
                raise WorkerError("malformed", "PDF page has nonfinite dimensions")
            if (
                width <= 0
                or height <= 0
                or not all(math.isfinite(v * scale) for v in (width, height))
            ):
                raise WorkerError("pixel_limit", "PDF page cannot be rendered at this scale")
            pixels = (math.ceil(width * scale), math.ceil(height * scale))
            if min(pixels) < 1 or max(pixels) > MAX_DIMENSION or pixels[0] * pixels[1] > MAX_PIXELS:
                raise WorkerError(
                    "pixel_limit", "Requested page exceeds the bitmap allocation limit"
                )
            with closing(page.get_textpage()) as textpage:
                count = textpage.count_chars()
                # The bounded range avoids allocating the entire text of an abusive page.
                # PDFium's range API is UCS-2; disclose this extraction limitation.
                text = (
                    textpage.get_text_range(0, min(count, MAX_TEXT_CHARS), errors="strict")
                    if count
                    else ""
                )
            truncated = count > MAX_TEXT_CHARS or len(text) > MAX_TEXT_CHARS
            text = text[:MAX_TEXT_CHARS]
            text_bytes = text.encode("utf-8")
            if len(text_bytes) > MAX_TEXT_BYTES:
                raise WorkerError("output_limit", "Extracted page text exceeds its byte limit")
            with (
                closing(
                    page.render(scale=scale, rotation=0, crop=(0, 0, 0, 0), may_draw_forms=False)
                ) as bitmap,
                bitmap.to_pil() as image,
            ):
                if image.size != pixels:
                    raise WorkerError(
                        "render_failed",
                        "Rendered dimensions differ from allocation bounds",
                    )
                buffer = BoundedBuffer()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
            metadata = {
                "schema_version": 1,
                "mode": "source_page",
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source_bytes": len(raw),
                "source_limit_bytes": source_limit_bytes,
                "page_number": page_number,
                "page_count": page_count,
                "page_bounds": bounds,
                "page_size": [width, height],
                "rotation": page.get_rotation(),
                "crop": [0, 0, 0, 0],
                "bitmap_width": pixels[0],
                "bitmap_height": pixels[1],
                "scale": scale,
                "backend": {
                    "pypdfium2": pdfium.PYPDFIUM_INFO.version,
                    "pdfium": pdfium.PDFIUM_INFO.version,
                    "pillow": pillow.__version__,
                    "flags": flags,
                },
                "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
                "text_bytes": len(text_bytes),
                "text_available": bool(text),
                "text_truncated": truncated,
                "text_method": "pdfium_get_text_range_ucs2_no_ocr",
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_bytes": len(image_bytes),
                "scope": "Untrusted source page observation. PDF canvas units and pixels are presentation only; no physical dimensions or claim confirmation are established. Forms are not activated; text order and visual applicability require review.",
            }
    (root / "text.txt").write_bytes(text_bytes)
    (root / "page.png").write_bytes(image_bytes)
    return metadata


def main() -> int:
    root = Path(sys.argv[1])
    try:
        try:
            if len(sys.argv) != 5:
                raise ValueError("explicit policy required")
            source_limit = int(sys.argv[4])
            if source_limit <= 0:
                raise ValueError("positive policy required")
            page_number, scale = int(sys.argv[2]), float(sys.argv[3])
        except (ValueError, IndexError) as exc:
            raise WorkerError(
                "invalid_request", "Worker requires page, scale and a positive source byte limit"
            ) from exc
        metadata = observe(root, page_number, scale, source_limit)
        (root / "metadata.json").write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
        return 0
    except WorkerError as exc:
        error = {"code": exc.code, "message": str(exc)}
    except Exception:
        error = {"code": "extraction_failed", "message": "PDF observation failed"}
    (root / "error.json").write_text(json.dumps(error), encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
