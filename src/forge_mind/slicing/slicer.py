"""Production slicer wrapper for the print-lane gate (ADR-0012 SS3): run a real FDM slicer CLI on a
part's STL with the committed profile and return a fact-carrying outcome.

Contract (fail-closed, ADR-0065 -- a silent no-run must never masquerade as a pass): this function
NEVER raises. A missing / unusable binary, a bad profile, a timeout, or a slice that produced no
gcode all return ``SliceOutcome(ok=False, error=...)``. The caller (the validate.py tier) turns a
non-ok outcome into an UNVERIFIABLE issue (report lane) or a REFUSAL (cell lane).

Distinct from the sibling ``slicing.slice_mesh`` (the dry-run/plan oracle from the text-to-cad gcode
skill, which *raises* when no binary is found): this is the gate wrapper -- fixed profile, warning
capture, degrade-honestly discovery. It reuses the package's ``discover_slicer`` as a last resort.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SLICER_TIMEOUT_S = 120.0
DEFAULT_PROFILE = Path(__file__).parent / "profiles" / "fdm_04n_pla.ini"

# Known install locations, checked after FORGE_SLICER_EXE and before PATH. PrusaSlicer's console
# binary writes gcode + warnings to real stdio and is the sanctioned Windows path (ADR-0012 delivery
# note: OrcaSlicer's Windows GUI-subsystem binary does not pipe console output, and its arm64 build
# will not launch from an x64 interpreter). Orca kept for Linux/CI where its CLI behaves.
_KNOWN_PATHS: tuple[str, ...] = (
    r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",
    r"C:\Program Files\OrcaSlicer\orca-slicer.exe",
    "/usr/bin/prusa-slicer",
    "/usr/local/bin/prusa-slicer",
    "/usr/bin/orca-slicer",
    "/usr/local/bin/orca-slicer",
)
_PATH_NAMES: tuple[str, ...] = (
    "prusa-slicer-console",
    "prusa-slicer",
    "PrusaSlicer",
    "orca-slicer",
    "OrcaSlicer",
)
# Slicer stdio lines that count as a WARNING for the gate. Curated + the generic token, so a real
# printability warning is caught without every info line reading as noise.
_WARNING_MARKERS: tuple[str, ...] = (
    "warning",
    "not manifold",
    "non-manifold",
    "auto-repaired",
    "repaired",
    "empty layers",
    "cannot be printed",
    "detected print stability",
    "is too tall",
    "objects could not be printed",
)


@dataclass(frozen=True)
class SliceOutcome:
    """The result of one gate slice. ``ok`` means a gcode file was produced; anything else is a
    degraded outcome carrying ``error`` (never an exception)."""

    ok: bool
    gcode_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    returncode: int | None = None
    stderr: str = ""


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def slicer_gate_enabled() -> bool:
    """Is the slicer gate tier switched on? Default OFF -- the hot validation loop does not slice."""
    # Wave D: the cell lane (FORGE_CELL_PROFILE) must never hand the arm an
    # unsliceable part -- the gate defaults ON there; elsewhere stays opt-in.
    return _flag("FORGE_SLICER_GATE") or _flag("FORGE_CELL_PROFILE")


def cell_lane_enabled() -> bool:
    """Is this the fabrication CELL lane (fail-closed slicer gate)? Default OFF (website lane).

    Reads ``FORGE_CELL_PROFILE`` -- the seam W2 (grasp-tab emission) also gates on. Kept here so the
    W1 lane law is testable now; W2 owns wiring the profile into generation."""
    return _flag("FORGE_CELL_PROFILE")


def find_slicer_exe() -> Path | None:
    """Locate a slicer binary: FORGE_SLICER_EXE override -> known install paths -> PATH -> the
    package's discover_slicer. ``None`` when none is found (honest, never a guess)."""
    override = os.environ.get("FORGE_SLICER_EXE")
    if override:
        p = Path(override)
        if p.exists():
            return p
        # a stale override falls through to real discovery rather than crashing the gate
    for cand in _KNOWN_PATHS:
        p = Path(cand)
        if p.exists():
            return p
    for name in _PATH_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    try:
        from forge_mind.slicing import discover_slicer

        backend = discover_slicer()
        return backend.path if backend is not None else None
    except Exception:
        return None


def slicer_available() -> bool:
    """True when a slicer binary is discoverable -- the skip predicate for the integration test."""
    return find_slicer_exe() is not None


def _parse_warnings(stdout: str, stderr: str) -> list[str]:
    seen: list[str] = []
    for stream in (stdout, stderr):
        for line in stream.splitlines():
            text = line.strip()
            if not text:
                continue
            low = text.lower()
            if any(marker in low for marker in _WARNING_MARKERS) and text not in seen:
                seen.append(text)
    return seen


def _build_command(exe: Path, stl: Path, profile: Path, out_gcode: Path) -> list[str]:
    """The slicer argv. PrusaSlicer/OrcaSlicer share this CLI: load config, export gcode, fixed
    output path. The slicer auto-arranges the part onto the bed (a single object is centred, only
    translated) -- deliberately, so a part the author modelled at the origin or off-bed still slices
    instead of failing 'outside print volume'; every gate fact is area/volume-based and so
    translation-invariant, and the mesh cross-section diff offsets by the mesh's own min-z."""
    return [
        str(exe),
        "--load",
        str(profile),
        "--export-gcode",
        "--output",
        str(out_gcode),
        str(stl),
    ]


def slice_part(
    stl_path: Path,
    outdir: Path,
    *,
    profile: Path = DEFAULT_PROFILE,
    exe: Path | None = None,
    timeout_s: float = SLICER_TIMEOUT_S,
) -> SliceOutcome:
    """Slice ``stl_path`` with ``profile`` into ``outdir``. Never raises -- see the module contract.

    ``exe`` overrides discovery (used by tests with a fake stub); otherwise ``find_slicer_exe`` runs.
    """
    exe = exe or find_slicer_exe()
    if exe is None:
        return SliceOutcome(ok=False, error="slicer unavailable")
    if not Path(profile).exists():
        return SliceOutcome(ok=False, error=f"slicer profile missing: {profile}")
    try:
        stl_path = Path(stl_path)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_gcode = outdir / f"{stl_path.stem}.gcode"
        if out_gcode.exists():
            out_gcode.unlink()
    except OSError as exc:
        return SliceOutcome(ok=False, error=f"slicer setup failed: {type(exc).__name__}: {exc}")
    cmd = _build_command(Path(exe), stl_path, Path(profile), out_gcode)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return SliceOutcome(ok=False, error=f"slicer timed out after {timeout_s:.0f}s")
    except OSError as exc:
        # e.g. WinError 216: an incompatible-architecture binary (arm64 exe / x64 interpreter)
        return SliceOutcome(ok=False, error=f"slicer could not run: {type(exc).__name__}: {exc}")
    warnings = _parse_warnings(proc.stdout or "", proc.stderr or "")
    tail = (proc.stderr or "")[-2000:]
    if not out_gcode.exists() or out_gcode.stat().st_size == 0:
        combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
        if "outside of the print volume" in combined or "outside the print volume" in combined:
            error = "part is outside the printer build volume (too large or off-bed)"
        else:
            error = f"slicer produced no gcode (returncode {proc.returncode})"
        return SliceOutcome(
            ok=False,
            error=error,
            warnings=warnings,
            returncode=proc.returncode,
            stderr=tail,
        )
    return SliceOutcome(
        ok=True,
        gcode_path=out_gcode,
        warnings=warnings,
        returncode=proc.returncode,
        stderr=tail,
    )


__all__ = [
    "DEFAULT_PROFILE",
    "SLICER_TIMEOUT_S",
    "SliceOutcome",
    "cell_lane_enabled",
    "find_slicer_exe",
    "slice_part",
    "slicer_available",
    "slicer_gate_enabled",
]
