"""Environment-fit checks (ADR-0087 S5) — the compiled CheckRequests, run against built geometry.

The compiler (S4) emits :class:`~forge_mind.invention.compile_requirements.CheckRequest`s from the
closed vocabulary; this module implements them over the finished mesh. Design disciplines:

  * **ADVISORY-FIRST (P13).** Nothing here touches ``validation.passed``. Every result is a
    located, explained finding for the lab report / assumption ledger; gating comes only after a
    seeded-defect audit on real artifacts (the honest-senses lesson, applied in advance).
  * **Size, not pose.** ``envelope_fits`` compares SORTED extents to sorted limits — the
    arbitrary_v1.1 erratum (a prompt-faithful part must not fail over incidental build
    orientation) baked in from day one.
  * **Stated conventions, never silent ones.** ``keepout_clear`` and ``path_cross_section`` are
    margin/area conventions in v1 (no environment mesh exists yet to collide against); each result
    STATES its convention verbatim so a reader knows exactly what was and was not measured.
  * **Fail closed.** Unreadable/degenerate geometry or missing parameters produce a FLAG result
    ("could not verify"), never a silent pass.

Result statuses: ``pass`` / ``flag`` (advisory concern or could-not-verify) — there is no ``fail``
because these do not gate (yet)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict

from forge_mind.invention.compile_requirements import CheckRequest

# interface_pattern: measured hole-pitch tolerance (print shrinkage + census tolerance)
_PITCH_TOL_MM = 0.8
# path_cross_section: the free-flow area must be at least this fraction of the fan frame area
_MIN_AREA_FRACTION = 0.6


class EnvCheckResult(BaseModel):
    """One advisory environment-fit finding: measured, explained, convention stated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    status: Literal["pass", "flag"]
    measured: str
    explanation: str


def _flag(check_id: str, why: str) -> EnvCheckResult:
    return EnvCheckResult(check_id=check_id, status="flag", measured="", explanation=why)


def _extents(mesh: trimesh.Trimesh) -> np.ndarray | None:
    try:
        ext = np.asarray(mesh.extents, dtype=float)
        if ext.shape != (3,) or not np.all(np.isfinite(ext)) or np.any(ext <= 0):
            return None
        return ext
    except Exception:
        return None


def _check_envelope_fits(mesh: trimesh.Trimesh, params: dict[str, float]) -> EnvCheckResult:
    """Sorted extents vs sorted stated availability — size, not pose (the v1.1 erratum)."""
    want = [
        params.get("available_length"),
        params.get("available_width"),
        params.get("available_height"),
    ]
    if any(v is None for v in want):
        return _flag("envelope_fits", "missing availability parameter(s) — could not verify")
    ext = _extents(mesh)
    if ext is None:
        return _flag("envelope_fits", "geometry unreadable — could not verify")
    have = np.sort(ext)
    limit = np.sort(np.asarray([float(v) for v in want if v is not None]))
    ok = bool(np.all(have <= limit + 1e-6))
    measured = f"part {have[0]:.1f} x {have[1]:.1f} x {have[2]:.1f} mm vs available {limit[0]:g} x {limit[1]:g} x {limit[2]:g} mm (sorted)"
    if ok:
        return EnvCheckResult(
            check_id="envelope_fits",
            status="pass",
            measured=measured,
            explanation="the part fits the stated available envelope in some orientation",
        )
    return EnvCheckResult(
        check_id="envelope_fits",
        status="flag",
        measured=measured,
        explanation="the part exceeds the stated available envelope",
    )


def _hole_pitches(part: Any) -> list[float]:
    """Pairwise center distances of measured coaxial hole stacks that share an axis direction
    (the mounting-pattern geometry). Empty when unmeasurable — the caller flags, never guesses."""
    try:
        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    except Exception:
        return []
    pitches: list[float] = []
    for i, a in enumerate(stacks):
        for b in stacks[i + 1 :]:
            da = np.asarray(a.axis_dir, dtype=float)
            db = np.asarray(b.axis_dir, dtype=float)
            if abs(float(np.dot(da, db))) < 0.996:  # ~5 deg: not a parallel pattern pair
                continue
            pa = np.asarray(a.axis_point, dtype=float)
            pb = np.asarray(b.axis_point, dtype=float)
            rel = pb - pa
            perp = rel - float(np.dot(rel, da)) * da  # center distance perpendicular to the axes
            pitches.append(float(np.linalg.norm(perp)))
    return pitches


def _check_interface_pattern(brep_part: Any, params: dict[str, float]) -> EnvCheckResult:
    """The stated mounting pitch must exist between measured parallel hole pairs."""
    pitch = params.get("mount_hole_pitch")
    if pitch is None:
        return _flag("interface_pattern", "missing mount_hole_pitch parameter — could not verify")
    if brep_part is None:
        return _flag("interface_pattern", "no B-rep available — hole pattern could not be measured")
    pitches = _hole_pitches(brep_part)
    if not pitches:
        return _flag("interface_pattern", "no parallel hole pairs measured — pattern not found")
    best = min(pitches, key=lambda p: abs(p - float(pitch)))
    if abs(best - float(pitch)) <= _PITCH_TOL_MM:
        return EnvCheckResult(
            check_id="interface_pattern",
            status="pass",
            measured=f"measured pitch {best:.2f} mm vs stated {pitch:g} mm (tol {_PITCH_TOL_MM} mm)",
            explanation="the stated mounting-hole pitch exists on the part",
        )
    return EnvCheckResult(
        check_id="interface_pattern",
        status="flag",
        measured=f"nearest measured pitch {best:.2f} mm vs stated {pitch:g} mm",
        explanation="no hole pair matches the stated mounting pitch within tolerance",
    )


def _check_keepout_clear(mesh: trimesh.Trimesh, params: dict[str, float]) -> EnvCheckResult:
    """v1 margin convention (STATED): with no environment mesh to collide against, the part's
    smallest extent must leave the stated clearance inside the stated fan frame span — i.e. the
    part cannot consume the airflow gap. An environment-mesh version supersedes this in v2."""
    frame = params.get("fan_frame_size")
    clearance = params.get("airflow_clearance")
    if frame is None or clearance is None:
        return _flag("keepout_clear", "missing fan/clearance parameter(s) — could not verify")
    ext = _extents(mesh)
    if ext is None:
        return _flag("keepout_clear", "geometry unreadable — could not verify")
    thinnest = float(np.min(ext))
    budget = float(frame) - float(clearance)
    convention = (
        f"convention: the part's thinnest extent ({thinnest:.1f} mm) must fit the fan span "
        f"({frame:g} mm) minus the stated clearance ({clearance:g} mm) = {budget:g} mm; "
        "no environment mesh was collided (v1 margin check)"
    )
    if thinnest <= budget + 1e-6:
        return EnvCheckResult(
            check_id="keepout_clear",
            status="pass",
            measured=convention,
            explanation="the stated airflow clearance survives the part's thinnest placement",
        )
    return EnvCheckResult(
        check_id="keepout_clear",
        status="flag",
        measured=convention,
        explanation="the part is too thick to leave the stated airflow clearance",
    )


def _check_path_cross_section(mesh: trimesh.Trimesh, params: dict[str, float]) -> EnvCheckResult:
    """v1 area convention (STATED): the part's largest face-on silhouette (product of its two
    largest extents) may not exceed (1 - {frac}) of the fan frame area — the free cross-section
    keeps at least {frac}. Geometric only; no flow is simulated."""
    frame = params.get("fan_frame_size")
    if frame is None:
        return _flag("path_cross_section", "missing fan_frame_size parameter — could not verify")
    ext = _extents(mesh)
    if ext is None:
        return _flag("path_cross_section", "geometry unreadable — could not verify")
    top = np.sort(ext)[::-1]
    silhouette = float(top[0] * top[1])
    frame_area = float(frame) * float(frame)
    free = max(0.0, 1.0 - silhouette / frame_area)
    convention = (
        f"convention: worst-case silhouette {silhouette:.0f} mm^2 vs fan frame {frame_area:.0f} mm^2 "
        f"-> free fraction {free:.2f} (floor {_MIN_AREA_FRACTION}); geometric only, no flow simulated"
    )
    if free >= _MIN_AREA_FRACTION:
        return EnvCheckResult(
            check_id="path_cross_section",
            status="pass",
            measured=convention,
            explanation="the free flow cross-section stays above the stated floor",
        )
    return EnvCheckResult(
        check_id="path_cross_section",
        status="flag",
        measured=convention,
        explanation="the part could choke the flow path below the stated floor",
    )


def run_env_checks(
    requests: list[CheckRequest] | tuple[CheckRequest, ...],
    mesh: trimesh.Trimesh | None,
    brep_part: Any = None,
) -> list[EnvCheckResult]:
    """Run every compiled check. ADVISORY results only; an unknown check id or missing geometry
    FLAGS (fail closed) — the vocabulary is closed at the compiler, so unknown here means drift."""
    out: list[EnvCheckResult] = []
    for req in requests:
        if mesh is None and req.check_id != "interface_pattern":
            out.append(_flag(req.check_id, "no mesh available — could not verify"))
            continue
        if req.check_id == "envelope_fits":
            out.append(_check_envelope_fits(mesh, req.params))  # type: ignore[arg-type]
        elif req.check_id == "interface_pattern":
            out.append(_check_interface_pattern(brep_part, req.params))
        elif req.check_id == "keepout_clear":
            out.append(_check_keepout_clear(mesh, req.params))  # type: ignore[arg-type]
        elif req.check_id == "path_cross_section":
            out.append(_check_path_cross_section(mesh, req.params))  # type: ignore[arg-type]
        else:
            out.append(
                _flag(req.check_id, "unknown check id — vocabulary drift, refusing to guess")
            )
    return out


__all__ = ["EnvCheckResult", "run_env_checks"]
