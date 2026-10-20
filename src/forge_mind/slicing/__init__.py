"""Slicer printability oracle + G-code analysis (approach from text-to-cad's gcode skill, MIT).

Upgrades Forge-Mind's printability claim from geometry heuristics to "a real production slicer
accepted this mesh and produced bounded, valid toolpaths." Two halves:

  - the SLICE: discover an installed slicer CLI (OrcaSlicer / PrusaSlicer / CuraEngine) and run it,
    DRY-RUN by default -- the planned command is returned for inspection and nothing runs until
    ``execute=True`` (the safety pattern from the bambu skill applied where the side effect is).
  - the CHECK: a static G-code validator (movement / extrusion / temperature / in-bounds) AND an
    estimate parser (print time + filament length/mass). The estimates are the predicted physical
    numbers that, diffed against a real print, are the bridge to CALIBRATING the validator oracle
    (the noted scaling bottleneck). text-to-cad ships the validator but NOT the estimates -- the
    parser here is net-new.

The slicer binary is an external dependency: ``discover_slicer`` returns None when none is
installed, so this degrades honestly. The validator + estimate parser need no slicer and run on
any ``.gcode``.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Slicer CLIs, in preference order: (kind, env override, candidate binary names).
_SLICERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("orca", "ORCA_SLICER_BIN", ("orca-slicer", "OrcaSlicer", "orcaslicer")),
    ("prusa", "PRUSA_SLICER_BIN", ("prusa-slicer", "PrusaSlicer", "prusa-slicer-console")),
    ("cura", "CURA_ENGINE_BIN", ("CuraEngine", "curaengine")),
)
_MOVEMENT = {"G0", "G1", "G2", "G3"}
_TEMPERATURE = {"M104", "M109", "M140", "M190"}


@dataclass(frozen=True)
class MachineProfile:
    """The printer envelope a sliced part must fit, plus the slicer's native config path."""

    name: str
    bounds_mm: dict[str, tuple[float, float]]  # {"x": (min, max), "y": ..., "z": ...}
    native_config: Path | None = None  # the slicer's own settings file (required for a real slice)


@dataclass(frozen=True)
class SlicerBackend:
    kind: str  # orca | prusa | cura
    path: Path


@dataclass(frozen=True)
class GCodeEstimates:
    """Slicer-predicted physical numbers parsed from the G-code -- the calibration signal."""

    print_time_s: float | None = None
    filament_mm: float | None = None
    filament_g: float | None = None


@dataclass(frozen=True)
class GCodeValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    estimates: GCodeEstimates = field(default_factory=GCodeEstimates)


def discover_slicer(*, search_path: str | None = None) -> SlicerBackend | None:
    """Find an installed slicer CLI (env override wins, then PATH). None if none is installed."""
    for kind, env, names in _SLICERS:
        override = os.environ.get(env)
        if override and Path(override).exists():
            return SlicerBackend(kind=kind, path=Path(override))
        for name in names:
            found = shutil.which(name, path=search_path)
            if found:
                return SlicerBackend(kind=kind, path=Path(found))
    return None


def _command(line: str) -> str:
    code = line.split(";", 1)[0].strip()
    if not code:
        return ""
    return code.split()[0].upper()


def _numeric_tokens(line: str) -> dict[str, float]:
    code = line.split(";", 1)[0].strip()
    tokens: dict[str, float] = {}
    for tok in code.split()[1:]:
        if len(tok) >= 2 and tok[0].isalpha():
            try:
                tokens[tok[0].upper()] = float(tok[1:])
            except ValueError:
                continue
    return tokens


_TIME_HMS = re.compile(r"estimated printing time.*?=\s*([0-9hdms :]+)", re.IGNORECASE)
_TIME_CURA = re.compile(r";TIME:\s*([0-9.]+)")
_FIL_G = re.compile(r"filament used\s*\[g\]\s*=\s*([0-9.]+)", re.IGNORECASE)
_FIL_MM = re.compile(r"filament used\s*\[mm\]\s*=\s*([0-9.]+)", re.IGNORECASE)
_FIL_CURA_M = re.compile(r";Filament used:\s*([0-9.]+)\s*m", re.IGNORECASE)
_HMS_PART = re.compile(r"(\d+)\s*([dhms])", re.IGNORECASE)


def _parse_hms(text: str) -> float | None:
    mult = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}
    total = 0.0
    found = False
    for value, unit in _HMS_PART.findall(text):
        total += float(value) * mult[unit.lower()]
        found = True
    return total if found else None


def parse_estimates(gcode_text: str) -> GCodeEstimates:
    """Parse the slicer's own print-time + filament estimates from the G-code comments.

    Handles PrusaSlicer / OrcaSlicer (`; estimated printing time ... = 1h 23m`, `filament used [g]`)
    and Cura (`;TIME:`, `;Filament used: ...m`). Missing fields stay None.
    """
    time_s: float | None = None
    m = _TIME_HMS.search(gcode_text)
    if m:
        time_s = _parse_hms(m.group(1))
    if time_s is None:
        c = _TIME_CURA.search(gcode_text)
        if c:
            time_s = float(c.group(1))

    filament_g = float(g.group(1)) if (g := _FIL_G.search(gcode_text)) else None
    filament_mm: float | None = None
    if (mm := _FIL_MM.search(gcode_text)) is not None:
        filament_mm = float(mm.group(1))
    elif (cm := _FIL_CURA_M.search(gcode_text)) is not None:
        filament_mm = float(cm.group(1)) * 1000.0
    return GCodeEstimates(print_time_s=time_s, filament_mm=filament_mm, filament_g=filament_g)


def validate_gcode(source: str | Path, profile: MachineProfile) -> GCodeValidation:
    """Static-validate G-code: movement / extrusion / temperature present + every absolute move
    inside the machine envelope. ``source`` is the G-code text or a path to a ``.gcode`` file.

    Does NOT simulate extrusion physics, firmware state, or acceleration -- ``ok`` means structurally
    sound and in-bounds, not "safe to print".
    """
    text = (
        Path(source).read_text(encoding="utf-8", errors="replace")
        if isinstance(source, Path)
        else source
    )
    lines = text.splitlines()
    errors: list[str] = []
    warnings: list[str] = []
    stats = {"lines": len(lines), "movement": 0, "extrusion": 0, "temperature": 0}
    if not text.strip():
        errors.append("G-code is empty.")

    (x_min, x_max) = profile.bounds_mm.get("x", (float("-inf"), float("inf")))
    (y_min, y_max) = profile.bounds_mm.get("y", (float("-inf"), float("inf")))
    (z_min, z_max) = profile.bounds_mm.get("z", (float("-inf"), float("inf")))
    absolute = True
    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}  # resolved absolute position (origin assumed homed at 0)
    for n, line in enumerate(lines, start=1):
        cmd = _command(line)
        if not cmd:
            continue
        if cmd == "G90":
            absolute = True
        elif cmd == "G91":
            absolute = False
        if cmd in _MOVEMENT:
            stats["movement"] += 1
            tok = _numeric_tokens(line)
            if cmd == "G1" and "E" in tok:
                stats["extrusion"] += 1
            # Resolve each move to an ABSOLUTE position (accumulate deltas in G91 relative mode) and
            # check the envelope in BOTH modes. The old code checked only in absolute mode, so an
            # out-of-bounds relative toolpath passed silently (audit: G91 envelope escape).
            for axis, lo, hi in (("X", x_min, x_max), ("Y", y_min, y_max), ("Z", z_min, z_max)):
                if axis not in tok:
                    continue
                pos[axis] = tok[axis] if absolute else pos[axis] + tok[axis]
                if not lo <= pos[axis] <= hi:
                    errors.append(
                        f"line {n}: {axis}={pos[axis]} outside {axis} range {lo}..{hi} mm."
                    )
        elif cmd in _TEMPERATURE:
            stats["temperature"] += 1

    if stats["movement"] == 0:
        errors.append("no G0/G1/G2/G3 movement commands found.")
    if stats["extrusion"] == 0:
        errors.append("no extrusion moves (G1 with E) found.")
    if stats["temperature"] == 0:
        errors.append("no nozzle/bed temperature commands found.")
    return GCodeValidation(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        stats=stats,
        estimates=parse_estimates(text),
    )


def build_slice_command(
    backend: SlicerBackend, stl_path: Path, profile: MachineProfile, out_gcode: Path
) -> list[str]:
    """The slicer argv (per kind). Best-effort: OrcaSlicer / PrusaSlicer share a CLI shape; Cura's
    CuraEngine differs. The native config is required for a real slice."""
    cfg = str(profile.native_config) if profile.native_config else ""
    if backend.kind in {"orca", "prusa"}:
        cmd = [str(backend.path)]
        if cfg:
            cmd += ["--load-settings", cfg]
        cmd += ["--outputdir", str(out_gcode.parent), "--slice", "0", str(stl_path)]
        return cmd
    # cura
    cmd = [str(backend.path), "slice", "-v"]
    if cfg:
        cmd += ["-j", cfg]
    cmd += ["-l", str(stl_path), "-o", str(out_gcode)]
    return cmd


@dataclass(frozen=True)
class SliceResult:
    executed: bool
    command: list[str]
    gcode_path: Path | None = None
    returncode: int | None = None
    stderr: str = ""


def slice_mesh(
    stl_path: Path,
    profile: MachineProfile,
    out_gcode: Path,
    *,
    backend: SlicerBackend | None = None,
    execute: bool = False,
) -> SliceResult:
    """Plan (and optionally run) a slice. DRY-RUN by default: returns the planned command without
    running anything. Pass ``execute=True`` to actually invoke the slicer (requires it installed).
    """
    backend = backend or discover_slicer()
    if backend is None:
        raise RuntimeError("no slicer CLI found (install OrcaSlicer / PrusaSlicer / CuraEngine)")
    command = build_slice_command(backend, stl_path, profile, out_gcode)
    if not execute:
        return SliceResult(executed=False, command=command)
    import subprocess

    out_gcode.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    return SliceResult(
        executed=True,
        command=command,
        gcode_path=out_gcode if out_gcode.exists() else None,
        returncode=proc.returncode,
        stderr=proc.stderr[-2000:],
    )


__all__ = [
    "GCodeEstimates",
    "GCodeValidation",
    "MachineProfile",
    "SliceResult",
    "SlicerBackend",
    "build_slice_command",
    "discover_slicer",
    "parse_estimates",
    "slice_mesh",
    "validate_gcode",
]
