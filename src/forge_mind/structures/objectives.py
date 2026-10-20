"""Objective-vector measurement (promoted from scripts/arm_search/score_arm.py; plan §3).

Read-only measurement over a part (mesh/STL) or a directory of them. It measures; it never ranks
(that is ``selection``) and never scalarizes. Fail-closed everywhere: a non-watertight mesh earns
no mass number it could win on; an unmeasured axis is ``None``, never a guess.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import trimesh

from forge_mind.materials import DEFAULT_MATERIAL_ID, mass_grams, resolve_material
from forge_mind.structures.loadcase import SMOKE_CANTILEVER_5N, LoadCase, run_load_case

# A load case may be given as a LoadCase or a callable(mesh)->LoadCase factory (per-part orient).
LoadCaseLike = LoadCase | Callable[[Any], LoadCase]

# Only these dispositions may be scored into an archive (plan §4 HARD FILTER).
ALLOWED_DISPOSITIONS = frozenset({"shipped", "shipped_with_caveats"})


@dataclass(frozen=True)
class PartObjectives:
    """Measured objectives for ONE printed part (fail-closed: unmeasurable -> None)."""

    stl: str
    watertight: bool
    volume_mm3: float | None
    mass_g: float | None
    material: str
    deflection_mm: float | None
    von_mises_mpa: float | None
    load_case: str | None


@dataclass(frozen=True)
class ArmObjectives:
    """Aggregate objective vector for a part set (a dir / run of parts)."""

    part_count: int
    total_mass_g: float | None  # None if ANY part is unmeasurable (fail-closed sum)
    max_deflection_mm: float | None  # worst part deflection under the load case
    material: str
    parts: list[PartObjectives]
    disposition: str | None = None  # set for run dirs
    scorable: bool = True  # False when a run's disposition is outside ALLOWED_DISPOSITIONS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_mesh(
    mesh: Any,
    *,
    material: str = DEFAULT_MATERIAL_ID,
    load_case: LoadCaseLike | None = SMOKE_CANTILEVER_5N,
    stl_label: str = "",
) -> PartObjectives:
    """Measure one mesh. ``load_case=None`` skips the (slow) FEA and leaves deflection None."""
    mat = resolve_material(material)
    watertight = bool(mesh.is_watertight)
    volume: float | None = float(mesh.volume) if watertight else None
    if volume is not None and volume <= 0.0:
        volume = None  # inverted / degenerate winding -> not a trustworthy solid volume
    mass = round(mass_grams(volume, mat), 4) if volume is not None else None
    case = load_case(mesh) if callable(load_case) else load_case
    lr = run_load_case(mesh, case) if case is not None else None
    return PartObjectives(
        stl=stl_label,
        watertight=watertight,
        volume_mm3=None if volume is None else round(volume, 4),
        mass_g=mass,
        material=mat.id,
        deflection_mm=None if lr is None else lr.max_deflection_mm,
        von_mises_mpa=None if lr is None else lr.max_von_mises_mpa,
        load_case=None if lr is None or case is None else case.name,
    )


def fea_deflection_within(mesh: Any, load_case: LoadCaseLike, bound_mm: float) -> bool | None:
    """Tri-state stiffness gate: True (measured <= bound), False (measured > bound), None (FEA
    unavailable). Search rejects ONLY on False (fail-open on None: a missing FEA env must not
    reject every candidate); a CLAIM requires True (a None is unproven and never claimed).
    Applied to COMPLETED candidates, not per ES eval (arm-scale FEA is 2-3 min; TEST_LOG row 14)."""
    case = load_case(mesh) if callable(load_case) else load_case
    lr = run_load_case(mesh, case)
    if lr is None:
        return None
    return bool(lr.max_deflection_mm <= bound_mm)


def score_stl(
    path: str | Path,
    *,
    material: str = DEFAULT_MATERIAL_ID,
    load_case: LoadCaseLike | None = SMOKE_CANTILEVER_5N,
) -> PartObjectives:
    mesh = trimesh.load(str(path), force="mesh")
    return score_mesh(mesh, material=material, load_case=load_case, stl_label=str(path))


def _aggregate(
    parts: list[PartObjectives],
    *,
    material: str,
    disposition: str | None = None,
    scorable: bool = True,
) -> ArmObjectives:
    masses = [p.mass_g for p in parts]
    total_mass = (
        None
        if any(m is None for m in masses)
        else round(sum(m for m in masses if m is not None), 3)
    )
    defls = [p.deflection_mm for p in parts if p.deflection_mm is not None]
    return ArmObjectives(
        part_count=len(parts),
        total_mass_g=total_mass,
        max_deflection_mm=max(defls) if defls else None,
        material=resolve_material(material).id,
        parts=parts,
        disposition=disposition,
        scorable=scorable,
    )


def score_stl_dir(
    directory: str | Path,
    *,
    material: str = DEFAULT_MATERIAL_ID,
    load_case: LoadCaseLike | None = SMOKE_CANTILEVER_5N,
) -> ArmObjectives:
    """Score every top-level ``*.stl`` in ``directory`` (the reference-dir mode, plan §5)."""
    stls = sorted(Path(directory).glob("*.stl"))
    parts = [score_stl(p, material=material, load_case=load_case) for p in stls]
    return _aggregate(parts, material=material)


def run_disposition(run_dir: str | Path) -> str | None:
    """The ``disposition`` from a run's functional_report.json, or None if absent."""
    report = Path(run_dir) / "functional_report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    d = data.get("disposition")
    return str(d) if d is not None else None


def run_is_scorable(run_dir: str | Path, allowed: frozenset[str] = ALLOWED_DISPOSITIONS) -> bool:
    """Plan §4 HARD FILTER: a run may be scored only if it shipped (or shipped w/ caveats)."""
    return run_disposition(run_dir) in allowed


def _run_part_count(run_dir: str | Path) -> int:
    for name, key in (("functional_report.json", "parts"), ("plan.json", "parts")):
        p = Path(run_dir) / name
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            parts = data.get(key)
            if isinstance(parts, list):
                return len(parts)
    return 0


def score_run_dir(run_dir: str | Path, *, material: str = DEFAULT_MATERIAL_ID) -> ArmObjectives:
    """Apply the §4 hard filter to a run dir (geometry scoring is the search harness's job --
    the report does not record final STL paths, so this path does not guess)."""
    disposition = run_disposition(run_dir)
    scorable = disposition in ALLOWED_DISPOSITIONS
    return ArmObjectives(
        part_count=_run_part_count(run_dir),
        total_mass_g=None,
        max_deflection_mm=None,
        material=resolve_material(material).id,
        parts=[],
        disposition=disposition,
        scorable=scorable,
    )


__all__ = [
    "ALLOWED_DISPOSITIONS",
    "ArmObjectives",
    "LoadCaseLike",
    "PartObjectives",
    "fea_deflection_within",
    "run_disposition",
    "run_is_scorable",
    "score_mesh",
    "score_run_dir",
    "score_stl",
    "score_stl_dir",
]
