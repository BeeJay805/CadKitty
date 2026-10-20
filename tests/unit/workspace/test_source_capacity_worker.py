"""Source admission policy transport; no change to PDF interpretation guarantees."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.unit.workspace.test_pdf_render import make_pdf, outputs

from forge_mind.workspace import pdf_render as renderer
from forge_mind.workspace import pdf_worker as worker
from forge_mind.workspace.source_capacity import (
    DEFAULT_SOURCE_BYTES,
    source_byte_limit,
    source_policy,
)


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    monkeypatch.delenv("FORGE_WORKSPACE_SOURCE_BYTES", raising=False)


@pytest.mark.parametrize("raw", [None, "", " ", "8388608"])
def test_bounded_default_and_explicit_discovery(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", raw)
    expected = 8388608 if raw == "8388608" else DEFAULT_SOURCE_BYTES
    assert source_byte_limit() == expected
    assert source_policy()["value"] == expected
    assert source_policy()["default"] == 2097152
    assert source_policy()["environment"] == "FORGE_WORKSPACE_SOURCE_BYTES"


@pytest.mark.parametrize("raw", ["0", "-1", "NaN", "inf", "2.5", "unlimited"])
def test_bad_operator_settings_are_explicit(monkeypatch, raw):
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", raw)
    with pytest.raises(ValueError, match="SOURCE_BYTES must be a positive integer"):
        source_byte_limit()


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "100"])
def test_private_limit_must_be_positive_integer_without_spawn(monkeypatch, limit):
    monkeypatch.setattr(
        renderer.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not spawn")
    )
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer.render_pdf_page(b"pdf", 1, source_limit_bytes=limit)
    assert error.value.code == "invalid_request"


def test_default_rejects_larger_input_before_spawn(monkeypatch):
    monkeypatch.setattr(
        renderer.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not spawn")
    )
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer.render_pdf_page(b"x" * (DEFAULT_SOURCE_BYTES + 1), 1)
    assert error.value.code == "source_limit"


def test_resolves_once_passes_private_argument_and_verifies_same_value(monkeypatch):
    calls, launches = [], []

    def policy():
        calls.append(True)
        return 8388608

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            launches.append((args, kwargs))
            assert args[-1] == "8388608"
            root = Path(args[3])
            metadata = outputs(root)
            metadata["source_limit_bytes"] = int(args[-1])
            (root / "metadata.json").write_text(json.dumps(metadata))
            monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", "1")

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(renderer, "source_byte_limit", policy)
    monkeypatch.setattr(renderer.subprocess, "Popen", Process)
    metadata, _, _ = renderer.render_pdf_page(b"pdf", 1)
    assert metadata["source_limit_bytes"] == 8388608
    assert calls == [True]
    assert launches[0][0][1] == "-I"
    assert "FORGE_WORKSPACE_SOURCE_BYTES" not in launches[0][1]["env"]


@pytest.mark.parametrize("claimed", [None, 1, True, 8388608])
def test_metadata_cannot_change_or_omit_invocation_limit(tmp_path, claimed):
    metadata = outputs(tmp_path)
    if claimed is None:
        metadata.pop("source_limit_bytes", None)
    else:
        metadata["source_limit_bytes"] = claimed
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer._verify(tmp_path, b"pdf", 1, 2.0, DEFAULT_SOURCE_BYTES)
    assert error.value.code == "invalid_output"


@pytest.mark.parametrize("policy_args", [[], ["0"], ["-1"], ["2.5"], ["NaN"], ["True"]])
def test_worker_cli_requires_explicit_valid_limit(tmp_path, monkeypatch, policy_args):
    monkeypatch.setattr(sys, "argv", ["pdf_worker", str(tmp_path), "1", "2", *policy_args])
    assert worker.main() == 2
    assert json.loads((tmp_path / "error.json").read_text())["code"] == "invalid_request"


@pytest.fixture
def native(monkeypatch):
    executable = os.environ.get("FORGE_PDF_NATIVE_TEST_PYTHON")
    if not executable:
        pytest.skip("Set FORGE_PDF_NATIVE_TEST_PYTHON for dedicated PDF runtime")
    monkeypatch.setenv("FORGE_PDF_PYTHON", executable)
    return executable


def test_native_larger_pdf_only_after_explicit_policy(native, monkeypatch):
    raw = make_pdf([("Configured catalog width 25 mm", 200, 300, 0)])
    raw += b" " * (DEFAULT_SOURCE_BYTES + 128 - len(raw))
    with pytest.raises(renderer.PdfObservationError) as error:
        renderer.render_pdf_page(raw, 1)
    assert error.value.code == "source_limit"
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", "8388608")
    metadata, text, image = renderer.render_pdf_page(raw, 1)
    assert metadata["source_limit_bytes"] == 8388608
    assert metadata["source_bytes"] == len(raw)
    assert metadata["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "25 mm" in text and image.startswith(b"\x89PNG")
    assert metadata["bitmap_width"] * metadata["bitmap_height"] <= renderer.MAX_PIXELS


def test_native_worker_independently_rejects_oversized_input(native, tmp_path):
    (tmp_path / "source.pdf").write_bytes(make_pdf([("25 mm", 200, 300, 0)]))
    process = subprocess.run(
        [native, "-I", str(renderer.WORKER), str(tmp_path), "1", "2", "2"],
        env=renderer._environment(tmp_path),
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.returncode == 2
    assert json.loads((tmp_path / "error.json").read_text())["code"] == "source_limit"
    assert not (tmp_path / "page.png").exists()
