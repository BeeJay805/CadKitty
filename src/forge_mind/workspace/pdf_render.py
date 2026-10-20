"""Supervise a standalone PDFium worker and verify its bounded observation outputs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from forge_mind.workspace.source_capacity import DEFAULT_SOURCE_BYTES, source_byte_limit

MAX_SOURCE_BYTES = DEFAULT_SOURCE_BYTES  # Legacy name describes the default, not active policy.
MAX_PIXELS = 16_000_000
MAX_DIMENSION = 8192
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_TEXT_CHARS = 120000
MAX_MEMORY_BYTES = 512 * 1024 * 1024
MAX_SECONDS = 30.0
WORKER = Path(__file__).with_name("pdf_worker.py")
_ADMISSION = threading.BoundedSemaphore(1)
ERROR_CODES = {
    "backend_unavailable",
    "backend_unsupported",
    "source_limit",
    "invalid_request",
    "encrypted",
    "malformed",
    "page_out_of_range",
    "pixel_limit",
    "output_limit",
    "render_failed",
    "extraction_failed",
}


class PdfObservationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _read_regular(path: Path, limit: int) -> bytes:
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError("nonregular output")
        if info.st_size > limit:
            raise ValueError("oversized output")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (info.st_dev, info.st_ino, info.st_size) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ):
                raise ValueError("output identity changed")
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size or len(raw) > limit:
            raise ValueError("output size changed")
        return raw
    except (OSError, ValueError) as exc:
        raise PdfObservationError(
            "invalid_output",
            "PDF worker output is missing, changed, linked, or oversized",
        ) from exc


def _environment(root: Path) -> dict[str, str]:
    # Keep only OS loader essentials. No API credentials, Python paths or user config.
    result = {
        key: os.environ[key]
        for key in ("SYSTEMROOT", "WINDIR", "SystemRoot", "COMSPEC", "LANG", "LC_ALL")
        if key in os.environ
    }
    result.update(
        TEMP=str(root),
        TMP=str(root),
        TMPDIR=str(root),
        HOME=str(root),
        USERPROFILE=str(root),
    )
    return result


def _stop(proc: subprocess.Popen[bytes]) -> None:
    children: list[psutil.Process] = []
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
    except psutil.Error:
        pass
    for child in reversed(children):
        with contextlib.suppress(psutil.Error):
            child.kill()
    if proc.poll() is None:
        proc.kill()
    psutil.wait_procs(children, timeout=2)
    proc.wait(timeout=5)


def _verify_png(image: bytes, width: int, height: int) -> None:
    """Validate bounded PNG framing/CRCs without decoding image data in the parent."""
    if len(image) < 33 or image[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG header")
    offset, saw_header, saw_data = 8, False, False
    while offset + 12 <= len(image):
        length = struct.unpack(">I", image[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(image):
            raise ValueError("truncated PNG chunk")
        kind = image[offset + 4 : offset + 8]
        body = image[offset + 8 : end - 4]
        if zlib.crc32(kind + body) != struct.unpack(">I", image[end - 4 : end])[0]:
            raise ValueError("PNG checksum mismatch")
        if not saw_header:
            if kind != b"IHDR" or length != 13 or struct.unpack(">II", body[:8]) != (width, height):
                raise ValueError("PNG dimensions mismatch")
            saw_header = True
        elif kind == b"IHDR":
            raise ValueError("duplicate PNG header")
        if kind == b"IDAT":
            saw_data = True
        if kind == b"IEND":
            if length != 0 or end != len(image) or not saw_data:
                raise ValueError("invalid PNG ending")
            return
        offset = end
    raise ValueError("PNG lacks complete image data")


def _verify(
    root: Path,
    raw: bytes,
    page_number: int,
    scale: float,
    source_limit_bytes: int = DEFAULT_SOURCE_BYTES,
) -> tuple[dict[str, Any], str, bytes]:
    try:
        metadata = json.loads(_read_regular(root / "metadata.json", 16384))
        text_raw = _read_regular(root / "text.txt", MAX_TEXT_CHARS * 4)
        image = _read_regular(root / "page.png", MAX_IMAGE_BYTES)
        text = text_raw.decode("utf-8")
        expected = {
            "schema_version": 1,
            "mode": "source_page",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_bytes": len(raw),
            "source_limit_bytes": source_limit_bytes,
            "page_number": page_number,
            "scale": scale,
            "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
            "text_sha256": hashlib.sha256(text_raw).hexdigest(),
            "text_bytes": len(text_raw),
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "image_bytes": len(image),
            "text_available": bool(text),
            "text_method": "pdfium_get_text_range_ucs2_no_ocr",
            "crop": [0, 0, 0, 0],
        }
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("request or output identity mismatch")
        extra_fields = {
            "page_count",
            "page_bounds",
            "page_size",
            "rotation",
            "bitmap_width",
            "bitmap_height",
            "backend",
            "text_truncated",
            "scope",
        }
        if set(metadata) != set(expected) | extra_fields or type(metadata["scale"]) not in (
            int,
            float,
        ):
            raise ValueError("unexpected metadata fields or scale type")
        if (
            len(text) > MAX_TEXT_CHARS
            or type(metadata.get("text_truncated")) is not bool
            or type(metadata.get("text_available")) is not bool
        ):
            raise ValueError("invalid text metadata")
        for name in (
            "page_count",
            "page_number",
            "source_bytes",
            "source_limit_bytes",
            "text_bytes",
            "image_bytes",
            "bitmap_width",
            "bitmap_height",
            "rotation",
            "schema_version",
        ):
            if type(metadata.get(name)) is not int:
                raise ValueError("noninteger metadata")
        if metadata["page_count"] < page_number or metadata["rotation"] not in (
            0,
            90,
            180,
            270,
        ):
            raise ValueError("invalid page metadata")
        width, height = metadata["bitmap_width"], metadata["bitmap_height"]
        if (
            min(width, height) <= 0
            or max(width, height) > MAX_DIMENSION
            or width * height > MAX_PIXELS
        ):
            raise ValueError("invalid bitmap dimensions")
        for name, size in (("page_bounds", 4), ("page_size", 2)):
            values = metadata.get(name)
            if (
                not isinstance(values, list)
                or len(values) != size
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in values
                )
            ):
                raise ValueError("invalid canvas metadata")
        bounds = metadata["page_bounds"]
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1] or min(metadata["page_size"]) <= 0:
            raise ValueError("unordered page bounds")
        if [math.ceil(value * scale) for value in metadata["page_size"]] != [
            width,
            height,
        ]:
            raise ValueError("bitmap scaling mismatch")
        _verify_png(image, width, height)
        backend = metadata.get("backend")
        if (
            not isinstance(backend, dict)
            or set(backend) != {"pypdfium2", "pdfium", "pillow", "flags"}
            or any(
                not isinstance(backend.get(key), str) or not backend[key]
                for key in ("pypdfium2", "pdfium", "pillow")
            )
            or not isinstance(backend.get("flags"), list)
        ):
            raise ValueError("invalid backend metadata")
        if any(
            not isinstance(flag, str) or "V8" in flag.upper() or "XFA" in flag.upper()
            for flag in backend["flags"]
        ):
            raise ValueError("unsupported backend flags")
        if not isinstance(metadata.get("scope"), str) or not metadata["scope"]:
            raise ValueError("missing source scope")
        return metadata, text, image
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        UnicodeError,
        struct.error,
    ) as exc:
        raise PdfObservationError(
            "invalid_output", "PDF worker result failed identity or format validation"
        ) from exc


def _render_pdf_page(
    raw: bytes,
    page_number: int,
    *,
    scale: float = 2.0,
    timeout: float | None = 30,
    cancelled: Callable[[], bool] | None = None,
    source_limit_bytes: int | None = None,
) -> tuple[dict[str, Any], str, bytes]:
    """Observe one exact-source page with bounded process supervision, not an OS sandbox."""
    limit = source_byte_limit() if source_limit_bytes is None else source_limit_bytes
    if type(limit) is not int or limit <= 0:
        raise PdfObservationError("invalid_request", "Source byte limit must be a positive integer")
    if not isinstance(raw, bytes) or not 0 < len(raw) <= limit:
        raise PdfObservationError("source_limit", f"PDF source must contain 1 to {limit} bytes")
    if (
        type(page_number) is not int
        or page_number < 1
        or type(scale) not in (int, float)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise PdfObservationError(
            "invalid_request", "A positive page number and finite scale are required"
        )
    if timeout is not None and (
        type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0
    ):
        raise PdfObservationError("invalid_request", "Timeout must be finite and nonnegative")
    remaining = MAX_SECONDS if timeout is None else min(timeout, MAX_SECONDS)
    deadline = time.monotonic() + remaining
    check_cancelled = cancelled or (lambda: False)
    if check_cancelled():
        raise PdfObservationError("cancelled", "PDF observation cancelled")
    if remaining <= 0:
        raise PdfObservationError("timeout", "PDF observation exceeded its deadline")
    executable = Path(os.environ.get("FORGE_PDF_PYTHON") or sys.executable)
    if not executable.is_absolute() or not executable.is_file():
        raise PdfObservationError(
            "backend_unavailable",
            "PDF worker interpreter must be an existing absolute executable path",
        )
    with tempfile.TemporaryDirectory(prefix="forge-pdf-page-") as folder:
        root = Path(folder)
        (root / "source.pdf").write_bytes(raw)
        try:
            proc = subprocess.Popen(
                [
                    str(executable),
                    "-I",
                    str(WORKER),
                    str(root),
                    str(page_number),
                    str(float(scale)),
                    str(limit),
                ],
                cwd=root,
                env=_environment(root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise PdfObservationError(
                "backend_unavailable", "PDF worker could not be started"
            ) from exc
        try:
            while proc.poll() is None:
                if check_cancelled():
                    raise PdfObservationError("cancelled", "PDF observation cancelled")
                if time.monotonic() >= deadline:
                    raise PdfObservationError("timeout", "PDF observation exceeded its deadline")
                try:
                    process = psutil.Process(proc.pid)
                    rss = process.memory_info().rss + sum(
                        child.memory_info().rss for child in process.children(recursive=True)
                    )
                    if rss > MAX_MEMORY_BYTES:
                        raise PdfObservationError(
                            "memory_limit", "PDF observation exceeded its memory limit"
                        )
                except psutil.NoSuchProcess:
                    pass
                except psutil.Error as exc:
                    raise PdfObservationError(
                        "worker_failed", "PDF worker memory supervision failed"
                    ) from exc
                time.sleep(0.025)
            if check_cancelled():
                raise PdfObservationError("cancelled", "PDF observation cancelled")
            if time.monotonic() >= deadline:
                raise PdfObservationError("timeout", "PDF observation exceeded its deadline")
            if proc.returncode != 0:
                error_file = root / "error.json"
                if error_file.exists():
                    try:
                        error = json.loads(_read_regular(error_file, 4096))
                        if error.get("code") in ERROR_CODES and isinstance(
                            error.get("message"), str
                        ):
                            raise PdfObservationError(error["code"], error["message"][:500])
                    except (ValueError, AttributeError, TypeError):
                        pass
                raise PdfObservationError(
                    "worker_failed", "PDF worker exited without a valid observation"
                )
            return _verify(root, raw, page_number, float(scale), limit)
        finally:
            if proc.poll() is None:
                _stop(proc)
            else:
                proc.wait(timeout=5)


def render_pdf_page(
    raw: bytes,
    page_number: int,
    *,
    scale: float = 2.0,
    timeout: float | None = 30,
    cancelled: Callable[[], bool] | None = None,
    source_limit_bytes: int | None = None,
) -> tuple[dict[str, Any], str, bytes]:
    """Admit one PDF worker per serving process; additional callers fail immediately.

    This is not host-wide admission. Separate server or MCP processes each have
    their own allowance and require deployment-wide resource orchestration.
    """
    if not _ADMISSION.acquire(blocking=False):
        raise PdfObservationError(
            "backend_busy", "A PDF observation is already running in this process"
        )
    try:
        return _render_pdf_page(
            raw,
            page_number,
            scale=scale,
            timeout=timeout,
            cancelled=cancelled,
            source_limit_bytes=source_limit_bytes,
        )
    finally:
        _ADMISSION.release()
