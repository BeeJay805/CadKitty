"""The slicer wrapper contract (ADR-0012 SS3), tested with a FAKE slicer stub -- no real binary.

The wrapper must NEVER raise (ADR-0065 fail-closed): a missing binary, a bad exe, a failed slice, or
a missing profile are all degraded ``SliceOutcome(ok=False, ...)``. A fake stub exe (a per-platform
launcher into a tiny Python script) exercises the real subprocess path and warning capture.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from forge_mind.slicing import slicer as slicer_mod
from forge_mind.slicing.slicer import SliceOutcome, find_slicer_exe, slice_part


def _make_fake_slicer(
    tmp_path: Path,
    *,
    gcode: str = "; fake\nM82\nG1 X1 Y1 E1\n",
    stderr: str = "",
    exit_code: int = 0,
) -> Path:
    """Write a stub that mimics the slicer CLI: parse --output, emit stderr, write gcode, exit."""
    logic = tmp_path / "stub_logic.py"
    logic.write_text(
        textwrap.dedent(
            f"""
            import sys
            args = sys.argv[1:]
            out = None
            for i, a in enumerate(args):
                if a == "--output" and i + 1 < len(args):
                    out = args[i + 1]
            sys.stderr.write({stderr!r})
            if out and {exit_code} == 0:
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write({gcode!r})
            sys.exit({exit_code})
            """
        ),
        encoding="utf-8",
    )
    if sys.platform == "win32":
        exe = tmp_path / "fake-slicer.cmd"
        exe.write_text(f'@echo off\r\n"{sys.executable}" "{logic}" %*\r\n', encoding="utf-8")
    else:
        exe = tmp_path / "fake-slicer.sh"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{logic}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


@pytest.fixture()
def profile(tmp_path: Path) -> Path:
    p = tmp_path / "prof.ini"
    p.write_text("layer_height = 0.2\n", encoding="utf-8")
    return p


def test_successful_slice_returns_gcode(tmp_path: Path, profile: Path) -> None:
    exe = _make_fake_slicer(tmp_path)
    stl = tmp_path / "part.stl"
    stl.write_text("solid\nendsolid\n", encoding="utf-8")
    out = slice_part(stl, tmp_path / "out", profile=profile, exe=exe)
    assert out.ok
    assert out.gcode_path is not None and out.gcode_path.exists()
    assert out.gcode_path.read_text(encoding="utf-8").startswith("; fake")


def test_warnings_are_captured_from_stderr(tmp_path: Path, profile: Path) -> None:
    exe = _make_fake_slicer(tmp_path, stderr="WARNING: object is not manifold\nSome info line\n")
    stl = tmp_path / "part.stl"
    stl.write_text("solid\n", encoding="utf-8")
    out = slice_part(stl, tmp_path / "out", profile=profile, exe=exe)
    assert out.ok
    assert any("not manifold" in w for w in out.warnings)
    assert "Some info line" not in out.warnings  # non-warning noise dropped


def test_failed_slice_never_raises(tmp_path: Path, profile: Path) -> None:
    exe = _make_fake_slicer(tmp_path, gcode="", exit_code=2)  # writes nothing, exits non-zero
    stl = tmp_path / "part.stl"
    stl.write_text("solid\n", encoding="utf-8")
    out = slice_part(stl, tmp_path / "out", profile=profile, exe=exe)
    assert not out.ok
    assert out.error is not None and "no gcode" in out.error
    assert out.gcode_path is None


def test_missing_binary_is_unavailable_not_exception(
    tmp_path: Path, profile: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slicer_mod, "find_slicer_exe", lambda: None)
    stl = tmp_path / "part.stl"
    stl.write_text("solid\n", encoding="utf-8")
    out = slice_part(stl, tmp_path / "out", profile=profile)
    assert out == SliceOutcome(ok=False, error="slicer unavailable")


def test_missing_profile_is_reported_not_run(tmp_path: Path) -> None:
    exe = _make_fake_slicer(tmp_path)
    stl = tmp_path / "part.stl"
    stl.write_text("solid\n", encoding="utf-8")
    out = slice_part(stl, tmp_path / "out", profile=tmp_path / "nope.ini", exe=exe)
    assert not out.ok
    assert out.error is not None and "profile missing" in out.error


def test_bad_exe_path_degrades_without_raising(tmp_path: Path, profile: Path) -> None:
    stl = tmp_path / "part.stl"
    stl.write_text("solid\n", encoding="utf-8")
    out = slice_part(stl, tmp_path / "out", profile=profile, exe=tmp_path / "does-not-exist.exe")
    assert not out.ok
    assert out.error is not None


def test_env_override_wins_and_stale_override_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = _make_fake_slicer(tmp_path)
    monkeypatch.setenv("FORGE_SLICER_EXE", str(exe))
    assert find_slicer_exe() == exe
    # a stale override must not crash discovery -- it falls through (here to whatever else exists/None)
    monkeypatch.setenv("FORGE_SLICER_EXE", str(tmp_path / "gone.exe"))
    assert find_slicer_exe() != (tmp_path / "gone.exe")
