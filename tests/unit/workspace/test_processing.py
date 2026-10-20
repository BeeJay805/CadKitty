from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from forge_mind.workspace.processing import MAX_FILE_BYTES, bounded_process, copy_artifact


@pytest.mark.parametrize(
    "name", ["part.step", "part.glb", "render.png", "evidence.json", "contract.json"]
)
def test_untrusted_links_cannot_overwrite_host_files(tmp_path: Path, name: str) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    source = tmp_path / name
    try:
        source.symlink_to(sentinel)
    except OSError:
        os.link(sentinel, source)  # Windows without symlink privilege: hard links must also fail.
    with pytest.raises(ValueError, match="linked"):
        copy_artifact(source, tmp_path / "copied")
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "copied").exists()


def test_oversized_artifact_is_rejected_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_WORKSPACE_FILE_BYTES", str(MAX_FILE_BYTES))
    source = tmp_path / "part.step"
    with source.open("wb") as file:
        file.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="oversized"):
        copy_artifact(source, tmp_path / "copied")


@pytest.mark.parametrize("cancel", [True, False])
def test_blocked_postprocessor_obeys_cancel_and_deadline(tmp_path: Path, cancel: bool) -> None:
    start = time.monotonic()
    with pytest.raises(ValueError, match=r"cancelled|wall-clock"):
        bounded_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            0.5 if not cancel else 10,
            lambda: cancel and time.monotonic() - start > 0.3,
        )
    assert time.monotonic() - start < 5
