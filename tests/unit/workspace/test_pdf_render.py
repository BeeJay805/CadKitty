import hashlib
import json
import math
import os
import struct
import sys
import threading
import time
import zlib
from pathlib import Path

import psutil
import pytest

from forge_mind.workspace import pdf_render as renderer
from forge_mind.workspace import pdf_worker as worker


def make_pdf(pages, password=None):
    """Small explicit PDF fixture: Helvetica text, dimensions and rotation per page."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    file_id = bytes.fromhex("00112233445566778899aabbccddeeff")
    padding = bytes.fromhex("28bf4e5e4e758a4164004e56fffa01082e2e00b6d0683e802f0ca9fe6453697a")

    def rc4(key, raw):
        state = list(range(256))
        j = 0
        for i in range(256):
            j = (j + state[i] + key[i % len(key)]) % 256
            state[i], state[j] = state[j], state[i]
        result = bytearray()
        i = j = 0
        for value in raw:
            i = (i + 1) % 256
            j = (j + state[i]) % 256
            state[i], state[j] = state[j], state[i]
            result.append(value ^ state[(state[i] + state[j]) % 256])
        return bytes(result)

    if password:
        user_pad = (password.encode() + padding)[:32]
        owner = rc4(hashlib.md5((b"fixture-owner" + padding)[:32]).digest()[:5], user_pad)
        encryption_key = hashlib.md5(user_pad + owner + struct.pack("<i", -4) + file_id).digest()[
            :5
        ]
        user = rc4(encryption_key, padding)
    ids = []
    for text, width, height, rotation in pages:
        page_id = len(objects) + 1
        ids.append(page_id)
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        font_size = 0.001 if len(text) > 120000 else 14
        content = f"BT /F1 {font_size} Tf 20 100 Td ({escaped}) Tj ET".encode("ascii")
        if len(text) > 120000:
            content = b"\n".join(
                f"BT /F1 0.001 Tf 20 {100 + offset * 0.002} Td ({escaped[offset * 1000 : (offset + 1) * 1000]}) Tj ET".encode()
                for offset in range(math.ceil(len(escaped) / 1000))
            )
        if password:
            object_key = hashlib.md5(
                encryption_key + (page_id + 1).to_bytes(3, "little") + b"\0\0"
            ).digest()[:10]
            content = rc4(object_key, content)
        objects.extend(
            [
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Rotate {rotation} /Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode(),
                f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
            ]
        )
    objects[1] = (
        f"<< /Type /Pages /Count {len(ids)} /Kids [{' '.join(f'{i} 0 R' for i in ids)}] >>".encode()
    )
    if password:
        objects.append(
            f"<< /Filter /Standard /V 1 /R 2 /Length 40 /O <{owner.hex()}> /U <{user.hex()}> /P -4 >>".encode()
        )
    output = bytearray(b"%PDF-1.7\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010} 00000 n \n".encode())
    security = (
        f" /Encrypt {len(objects)} 0 R /ID [<{file_id.hex()}> <{file_id.hex()}>]"
        if password
        else ""
    )
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R{security} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(output)


def fake_png():
    def chunk(name, raw):
        return struct.pack(">I", len(raw)) + name + raw + struct.pack(">I", zlib.crc32(name + raw))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" * 14))
        + chunk(b"IEND", b"")
    )


def outputs(root, raw=b"pdf", page=1, scale=2.0):
    text, image = b"Dimension 25 mm", fake_png()
    metadata = {
        "schema_version": 1,
        "mode": "source_page",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "source_limit_bytes": renderer.MAX_SOURCE_BYTES,
        "page_number": page,
        "page_count": page,
        "scale": scale,
        "page_bounds": [0, 0, 1, 1],
        "page_size": [1, 1],
        "rotation": 0,
        "crop": [0, 0, 0, 0],
        "bitmap_width": 2,
        "bitmap_height": 2,
        "backend": {
            "pypdfium2": "5.13.0",
            "pdfium": "153.0.7999.0",
            "pillow": "12.3.0",
            "flags": [],
        },
        "worker_sha256": hashlib.sha256(renderer.WORKER.read_bytes()).hexdigest(),
        "text_sha256": hashlib.sha256(text).hexdigest(),
        "text_bytes": len(text),
        "text_available": True,
        "text_truncated": False,
        "text_method": "pdfium_get_text_range_ucs2_no_ocr",
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "image_bytes": len(image),
        "scope": "Presentation only",
    }
    (root / "text.txt").write_bytes(text)
    (root / "page.png").write_bytes(image)
    (root / "metadata.json").write_text(json.dumps(metadata))
    return metadata


def test_verify_complete_binding(tmp_path):
    expected = outputs(tmp_path)
    metadata, text, image = renderer._verify(tmp_path, b"pdf", 1, 2.0)
    assert metadata == expected
    assert text == "Dimension 25 mm"
    assert image == fake_png()


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "0" * 64),
        ("source_bytes", 99),
        ("page_number", 2),
        ("page_number", True),
        ("page_count", 0),
        ("scale", 3),
        ("image_sha256", "0" * 64),
        ("text_sha256", "0" * 64),
        ("text_bytes", 0),
        ("image_bytes", 0),
        ("worker_sha256", "0" * 64),
        ("bitmap_width", 3),
        ("bitmap_width", True),
        ("bitmap_height", 999999),
        ("rotation", 45),
        ("page_bounds", [0, 0, float("nan"), 1]),
        ("page_size", [0, 1]),
        ("crop", [1, 0, 0, 0]),
        ("mode", "cad"),
        ("text_available", False),
        ("text_truncated", "false"),
        ("backend", {"pypdfium2": "5.13.0", "pdfium": "153", "flags": ["V8"]}),
    ],
)
def test_reject_mismatched_or_invalid_manifest(tmp_path, field, value):
    metadata = outputs(tmp_path)
    metadata[field] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer._verify(tmp_path, b"pdf", 1, 2.0)
    assert error.value.code == "invalid_output"


@pytest.mark.parametrize("file", ["metadata.json", "text.txt", "page.png"])
def test_reject_missing_outputs(tmp_path, file):
    outputs(tmp_path)
    (tmp_path / file).unlink()
    with pytest.raises(renderer.PdfObservationError, match="output"):
        renderer._verify(tmp_path, b"pdf", 1, 2.0)


def test_reject_hard_link_output(tmp_path):
    outputs(tmp_path)
    os.link(tmp_path / "text.txt", tmp_path / "alias.txt")
    with pytest.raises(renderer.PdfObservationError):
        renderer._verify(tmp_path, b"pdf", 1, 2.0)


def test_bounded_output_read(tmp_path):
    path = tmp_path / "large"
    path.write_bytes(b"1234")
    with pytest.raises(renderer.PdfObservationError):
        renderer._read_regular(path, 3)


@pytest.mark.parametrize("change", ["truncated", "checksum", "trailing"])
def test_invalid_png_rejected_even_with_matching_manifest_hash(tmp_path, change):
    metadata = outputs(tmp_path)
    image = fake_png()
    if change == "truncated":
        image = image[:-12]
    elif change == "checksum":
        image = image[:-1] + bytes([image[-1] ^ 1])
    else:
        image += b"trailing"
    metadata["image_bytes"] = len(image)
    metadata["image_sha256"] = hashlib.sha256(image).hexdigest()
    (tmp_path / "page.png").write_bytes(image)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer._verify(tmp_path, b"pdf", 1, 2.0)
    assert error.value.code == "invalid_output"


@pytest.mark.parametrize(
    "raw,page,scale,timeout",
    [
        (b"", 1, 2, 30),
        (b"x" * (renderer.MAX_SOURCE_BYTES + 1), 1, 2, 30),
        (b"pdf", True, 2, 30),
        (b"pdf", 0, 2, 30),
        (b"pdf", 1, float("nan"), 30),
        (b"pdf", 1, 0, 30),
        (b"pdf", 1, 2, -1),
        (b"pdf", 1, 2, float("inf")),
    ],
    ids=[f"invalid-{i}" for i in range(8)],
)
def test_invalid_request_never_spawns(monkeypatch, raw, page, scale, timeout):
    monkeypatch.setattr(renderer.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(renderer.PdfObservationError):
        renderer.render_pdf_page(raw, page, scale=scale, timeout=timeout)


def test_environment_excludes_credentials_and_python_paths(tmp_path, monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "FORGE_PDF_PYTHON",
        "PYTHONPATH",
        "AWS_SECRET_ACCESS_KEY",
        "PATH",
    ):
        monkeypatch.setenv(name, "secret")
    environment = renderer._environment(tmp_path)
    assert all("secret" not in value for value in environment.values())
    assert environment["HOME"] == str(tmp_path)


def test_mock_success_uses_isolated_worker_and_closed_handles(monkeypatch):
    captured = {}

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            captured.update(args=args, **kwargs)
            outputs(Path(args[3]))

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(renderer.subprocess, "Popen", Process)
    _, text, _ = renderer.render_pdf_page(b"pdf", 1)
    assert text == "Dimension 25 mm"
    assert captured["args"][1] == "-I"
    assert captured["close_fds"] is True
    assert captured["stdin"] == renderer.subprocess.DEVNULL


def test_pre_cancel_never_spawns(monkeypatch):
    monkeypatch.setattr(renderer.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer.render_pdf_page(b"pdf", 1, cancelled=lambda: True)
    assert error.value.code == "cancelled"


def test_worker_rejects_feature_enabled_pdfium_before_parsing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    backend = SimpleNamespace(PDFIUM_INFO=SimpleNamespace(flags=["XFA"]))
    monkeypatch.setitem(sys.modules, "pypdfium2", backend)
    with pytest.raises(worker.WorkerError) as error:
        worker.observe(tmp_path, 1, 2)
    assert error.value.code == "backend_unsupported"


def test_png_buffer_enforces_limit_before_write(monkeypatch):
    monkeypatch.setattr(worker, "MAX_IMAGE_BYTES", 4)
    buffer = worker.BoundedBuffer()
    buffer.write(b"1234")
    with pytest.raises(worker.WorkerError):
        buffer.write(b"5")
    assert buffer.getvalue() == b"1234"


@pytest.mark.parametrize("cancel", [False, True])
def test_supervisor_terminates_and_reaps_actual_sleeping_worker(tmp_path, monkeypatch, cancel):
    script = tmp_path / "sleep_worker.py"
    pidfile = tmp_path / "pid"
    script.write_text(
        f"import os,time,pathlib\npathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    monkeypatch.setattr(renderer, "WORKER", script)
    monkeypatch.setenv("FORGE_PDF_PYTHON", sys.executable)
    started = time.monotonic()

    def callback():
        return cancel and pidfile.exists()

    with pytest.raises(renderer.PdfObservationError) as error:
        renderer.render_pdf_page(b"pdf", 1, timeout=0.7, cancelled=callback)
    assert error.value.code == ("cancelled" if cancel else "timeout")
    assert time.monotonic() - started < 7
    assert pidfile.exists()
    assert not psutil.pid_exists(int(pidfile.read_text()))


@pytest.fixture
def native(monkeypatch):
    configured = os.environ.get("FORGE_PDF_NATIVE_TEST_PYTHON")
    if not configured:
        pytest.skip("Set FORGE_PDF_NATIVE_TEST_PYTHON for the required PDF backend smoke suite")
    monkeypatch.setenv("FORGE_PDF_PYTHON", configured)


def test_native_selected_page_text_rotation_and_exact_source(native):
    raw = make_pdf([("WRONG PAGE 11 mm", 200, 300, 0), ("Drawing width 25 mm", 200, 300, 90)])
    metadata, text, image = renderer.render_pdf_page(raw, 2)
    assert "25 mm" in text and "WRONG PAGE" not in text
    assert metadata["page_count"] == 2
    assert metadata["rotation"] == 90
    assert (metadata["bitmap_width"], metadata["bitmap_height"]) == (600, 400)
    assert metadata["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert metadata["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert metadata["backend"]["pypdfium2"] == "5.13.0"
    assert metadata["backend"]["pillow"] in {"12.2.0", "12.3.0"}
    assert metadata["backend"]["flags"] == []


def test_native_scanned_page_returns_image_and_explicit_no_text(native):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(buffer, format="PDF")
    metadata, text, image = renderer.render_pdf_page(buffer.getvalue(), 1)
    assert text == ""
    assert metadata["text_available"] is False
    assert image.startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "raw,page,code",
    [
        (b"%PDF-1.7 broken", 1, "malformed"),
        (make_pdf([("Hello", 100, 100, 0)]), 2, "page_out_of_range"),
        (make_pdf([("Hello", 1000000, 1000000, 0)]), 1, "pixel_limit"),
    ],
    ids=["corrupt", "page_out_of_range", "huge_page"],
)
def test_native_typed_failures(native, raw, page, code):
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer.render_pdf_page(raw, page)
    assert error.value.code == code


def test_native_text_truncation_is_explicit(native):
    raw = make_pdf([("Q" * 120010, 200, 300, 0)])
    metadata, text, _ = renderer.render_pdf_page(raw, 1)
    assert metadata["text_truncated"] is True
    assert len(text) <= 120000


def test_native_password_protected_pdf_is_explicitly_unavailable(native):
    raw = make_pdf([("Drawing width 25 mm", 200, 300, 0)], password="fixture-password")
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer.render_pdf_page(raw, 1)
    assert error.value.code == "encrypted"


def test_supervisor_memory_limit_kills_worker(monkeypatch):
    from types import SimpleNamespace

    stopped = []

    class Process:
        pid = 1234
        returncode = None

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return None

    monkeypatch.setattr(renderer.subprocess, "Popen", Process)
    monkeypatch.setattr(
        renderer.psutil,
        "Process",
        lambda pid: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=renderer.MAX_MEMORY_BYTES + 1),
            children=lambda **kwargs: [],
        ),
    )
    monkeypatch.setattr(renderer, "_stop", lambda proc: stopped.append(proc.pid))
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer.render_pdf_page(b"pdf", 1)
    assert error.value.code == "memory_limit"
    assert stopped == [1234]


def test_simultaneous_threads_admit_one_worker_then_release_for_next_call(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    launches = []
    results = []

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            launches.append(args)
            if len(launches) == 1:
                entered.set()
                assert release.wait(5)
            outputs(Path(args[3]))

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(renderer.subprocess, "Popen", Process)

    def call():
        try:
            results.append(renderer.render_pdf_page(b"pdf", 1))
        except BaseException as exc:
            results.append(exc)

    first = threading.Thread(target=call)
    second = threading.Thread(target=call)
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        second.join(timeout=2)
        assert not second.is_alive()
        assert len(launches) == 1
        assert len(results) == 1
        assert isinstance(results[0], renderer.PdfObservationError)
        assert results[0].code == "backend_busy"
    finally:
        release.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
    assert not first.is_alive()
    assert isinstance(results[1], tuple)
    assert renderer.render_pdf_page(b"pdf", 1)[1] == "Dimension 25 mm"
    assert len(launches) == 2


@pytest.mark.parametrize(
    "failure", ["cancelled", "invalid_request", "worker_failed", "runtime_error"]
)
def test_admission_released_after_each_failure(monkeypatch, failure):
    calls = []

    def render_once(*args, **kwargs):
        calls.append(failure)
        if len(calls) == 1:
            if failure == "runtime_error":
                raise RuntimeError("unexpected")
            raise renderer.PdfObservationError(failure, "controlled failure")
        return {"mode": "source_page"}, "25 mm", b"png"

    monkeypatch.setattr(renderer, "_render_pdf_page", render_once)
    with pytest.raises(RuntimeError):
        renderer.render_pdf_page(b"pdf", 1)
    assert renderer.render_pdf_page(b"pdf", 1)[1] == "25 mm"
    assert len(calls) == 2


@pytest.mark.parametrize("value", [None, "", 12.3])
def test_pillow_backend_version_required(tmp_path, value):
    metadata = outputs(tmp_path)
    if value is None:
        del metadata["backend"]["pillow"]
    else:
        metadata["backend"]["pillow"] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer._verify(tmp_path, b"pdf", 1, 2.0)
    assert error.value.code == "invalid_output"
