"""Validator-derived fit loss (ADR-0074): the optimizer fixes what the VALIDATOR measures.

ADR-0073 proved the division of labor (LLM = structure, optimizer = continuous dimensions) by
fitting against a retrieved reference. But the validator already measures continuous deficits on
EVERY part -- no reference required:

  * ``wall``     -- thinnest measured wall vs the brief's minimum (the same sampled measurement
    the min-wall gate uses), as a normalized deficit;
  * ``envelope`` -- built bbox extents vs the brief's stated envelope (sorted both ways, the same
    orientation-agnostic comparison the SIZE-MISMATCH gate uses), as mean relative error;
  * ``balance``  -- for a ``rotating_power`` part with a real shaft bore: the static + dynamic
    imbalance ratios about the bore axis (the same physics the balance advisory reports). The
    axis is located ONCE from the baseline B-rep and held fixed across the fit (dimension changes
    do not move a bore's centerline in the authored frame; re-deriving it per candidate would
    need a per-eval STEP load, which is neither cheap nor thread-safe).

Each term is normalized to ~[0, 1] and FAILS SOFT to 0 (an unmeasurable signal contributes
nothing -- the optimizer never chases a claim the validator could not make). ONE deliberate
exception: the radial feature-count term (ADR-0079) charges 1.0 when the brief names a tooth
count that cannot be measured -- fail-soft there would pay the optimizer to erase the teeth
(the council's measurement-suppression finding). Combined with the
reference terms in ``param_opt``, this turns the parameter fit into a general "make it valid,
right-sized, and balanced" machine that runs with or without a reference. Acceptance authority is
unchanged: the loop's polish hook still requires the FULL validator's tier not to degrade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.agentic.brief import DesignBrief, parse_envelope_mm
from forge_mind.agentic.confidence import classify_part
from forge_mind.agentic.radial_count import (
    RadialLock,
    lock_from_baseline,
    measure_radial_count,
    measure_radial_count_locked,
    planned_radial_teeth,
    radial_deficit,
)
from forge_mind.agentic.spatial import _planned_hole_specs, mesh_genus
from forge_mind.validator.balance import rotational_balance
from forge_mind.validator.feature_checks import _measure_min_wall
from forge_mind.validator.geometry import main_bore_axis

Vec3T = tuple[float, float, float]


@dataclass(frozen=True)
class ValidatorTarget:
    """The brief-derived targets the loss fits against. Any None field contributes zero."""

    min_wall_mm: float | None
    envelope_sorted: tuple[float, float, float] | None  # brief L/W/H sorted ascending
    balance_axis: tuple[Vec3T, Vec3T] | None  # (point, direction) of the shaft bore
    planned_through_holes: int | None = None  # the brief's through-hole count (ADR-0078)
    radial_teeth: int | None = None  # the brief's named tooth count (ADR-0079)
    radial_lock: RadialLock | None = None  # baseline measurement frame (idea 2); None = full sweep

    @property
    def applicable(self) -> bool:
        return any(
            v is not None
            for v in (
                self.min_wall_mm,
                self.envelope_sorted,
                self.balance_axis,
                self.planned_through_holes,
                self.radial_teeth,
            )
        )


def validator_target(
    brief: DesignBrief,
    *,
    baseline_step: Path | None = None,
    baseline_stl: Path | None = None,
    fabrication_mode: Any | None = None,
) -> ValidatorTarget:
    """Build the targets ONCE per fit (the bore axis needs one B-rep load -- sequential, done
    here, never inside the parallel eval loop). Every field fails soft to None.

    ``fabrication_mode``: when given, the wall target is the mode's EFFECTIVE threshold -- the
    same one the acceptance gate enforces. Measured on the timing-gear run: the researcher
    declared 2.5 mm, the GENERAL gate enforces 1.0, and the loss chasing 2.5 spent four
    brief-stated dimensions (tip 76->92 mm) buying wall the gate never demanded. The loss must
    never want what the gate does not require."""
    min_wall = float(brief.min_wall_mm) if brief.min_wall_mm and brief.min_wall_mm > 0 else None
    if min_wall is not None and fabrication_mode is not None:
        try:
            eff = fabrication_mode.min_wall_override(min_wall)
            if eff is not None:
                min_wall = float(eff)
        except Exception:
            pass  # unknown mode object -> keep the declared wall

    envelope: tuple[float, float, float] | None = None
    parsed = parse_envelope_mm(brief.overall_envelope_mm)
    if parsed is not None and all(v > 0 for v in parsed):
        env = sorted(float(v) for v in parsed)
        envelope = (env[0], env[1], env[2])

    balance_axis: tuple[Vec3T, Vec3T] | None = None
    if baseline_step is not None and classify_part(brief.object_name, brief.summary) == (
        "rotating_power"
    ):
        try:
            from forge_mind.agentic.validate import _load_part

            part = _load_part(baseline_step)
            axis = main_bore_axis(part) if part is not None else None
            if axis is not None:
                point, direction = axis
                balance_axis = (
                    (float(point[0]), float(point[1]), float(point[2])),
                    (float(direction[0]), float(direction[1]), float(direction[2])),
                )
        except Exception:
            balance_axis = None  # no axis -> no balance claim (fail-soft)
    planned: int | None = None
    try:
        n = sum(spec.count for spec in _planned_hole_specs(brief))
        planned = n if n > 0 else None
    except Exception:
        planned = None  # unparseable plan -> no claim (fail-soft)
    teeth: int | None = None
    try:
        teeth = planned_radial_teeth(brief)
    except Exception:
        teeth = None  # unparseable brief -> no claim (fail-soft)
    lock: RadialLock | None = None
    if teeth is not None and baseline_stl is not None:
        # Lock the measurement frame ONCE from the baseline (idea 2, the balance-axis
        # precedent): per-candidate measurement then skips axis detection + the band sweep and
        # cannot band-hop. Fail-soft to None -> candidates get the full unlocked measurement.
        try:
            baseline_mesh: Any = trimesh.load(str(baseline_stl), force="mesh")
            lock = lock_from_baseline(baseline_mesh, teeth)
        except Exception:
            lock = None
    return ValidatorTarget(
        min_wall_mm=min_wall,
        envelope_sorted=envelope,
        balance_axis=balance_axis,
        planned_through_holes=planned,
        radial_teeth=teeth,
        radial_lock=lock,
    )


def _wall_term(mesh: trimesh.Trimesh, required: float) -> float:
    """Normalized thin-wall deficit: 0 when the thinnest measured wall meets the requirement (or
    is unmeasurable -- no claim), rising toward 1 as the wall thins toward zero."""
    try:
        measured = _measure_min_wall(mesh)
    except Exception:
        return 0.0
    if measured is None:
        return 0.0
    thinnest = float(measured[0])
    if thinnest >= required:
        return 0.0
    return min((required - thinnest) / required, 1.0)


def _envelope_term(mesh: trimesh.Trimesh, want_sorted: tuple[float, float, float]) -> float:
    """Mean relative extent error vs the brief envelope (sorted-vs-sorted, orientation-agnostic),
    each axis clipped at 100% error."""
    extents = np.sort(np.asarray(mesh.bounding_box.extents, dtype=np.float64))
    want = np.asarray(want_sorted, dtype=np.float64)
    rel = np.abs(extents - want) / want
    return float(np.mean(np.clip(rel, 0.0, 1.0)))


def _feature_term(mesh: trimesh.Trimesh, planned: int) -> float:
    """Missing-through-hole deficit (ADR-0078, from the hard-5 carrier whose param-opt loss read
    0.000 while its bores were dimples): fraction of the brief's planned through-holes ABSENT from
    the measured mesh genus. A cut that exists in code but does not pierce is a DEPTH float in
    PARAMS -- optimizer-fixable once the loss can see it. Unmeasurable genus -> 0 (no claim)."""
    try:
        g = mesh_genus(mesh)
    except Exception:
        return 0.0
    if g is None:
        return 0.0
    return max(0.0, float(planned - int(g))) / float(planned)


def _radial_term(mesh: trimesh.Trimesh, planned: int, lock: RadialLock | None = None) -> float:
    """Radial feature-count deficit (ADR-0079, from the hard-5 flywheel's dropped 120-tooth
    ring). DELIBERATELY not fail-soft: once the brief NAMES a count, not_run costs the full 1.0
    -- the council's Goodhart finding was that a not_run->0 term pays the optimizer to FLATTEN
    the teeth until the spectral peak dies. A detected-but-wrong count costs < 1, so countable
    features always beat erased ones. With a baseline ``lock`` the candidate is measured in the
    frozen frame (fast, band-hop-free); acceptance always re-measures unlocked."""
    try:
        rc = (
            measure_radial_count_locked(mesh, planned, lock)
            if lock is not None
            else measure_radial_count(mesh, planned)
        )
    except Exception:
        return 1.0  # unmeasurable when a count is named = the not_run price
    return radial_deficit(planned, rc)


def _balance_term(mesh: trimesh.Trimesh, axis: tuple[Vec3T, Vec3T]) -> float:
    """Static + dynamic imbalance about the fixed bore axis, each ratio clipped at 1 (so the term
    is ~[0, 2] before the 0.5 weight below). Unverifiable -> 0 (no claim)."""
    try:
        report = rotational_balance(mesh, axis[0], axis[1])
    except Exception:
        return 0.0
    if report.balanced is None:
        return 0.0
    return min(float(report.static_ratio), 1.0) + min(float(report.dynamic_ratio), 1.0)


def validator_loss(stl_path: Path, target: ValidatorTarget) -> float:
    """The validator-derived loss of one built candidate. 0 when everything measurable is
    satisfied; fails soft to 0 per-term. The caller (param_opt) owns the +inf validity guard."""
    if not target.applicable:
        return 0.0
    try:
        mesh: Any = trimesh.load(str(stl_path), force="mesh")
    except Exception:
        return 0.0  # unreadable is the validity guard's problem, not a fit signal
    loss = 0.0
    if target.min_wall_mm is not None:
        loss += _wall_term(mesh, target.min_wall_mm)
    if target.envelope_sorted is not None:
        loss += _envelope_term(mesh, target.envelope_sorted)
    if target.balance_axis is not None:
        loss += 0.5 * _balance_term(mesh, target.balance_axis)
    if target.planned_through_holes:
        loss += _feature_term(mesh, target.planned_through_holes)
    if target.radial_teeth:
        loss += _radial_term(mesh, target.radial_teeth, target.radial_lock)
    return loss


__all__ = ["ValidatorTarget", "validator_loss", "validator_target"]
