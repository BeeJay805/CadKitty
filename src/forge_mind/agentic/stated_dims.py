"""Stated-dimension protection (ADR-0080): what the user typed is not optimizer currency.

The measured timing-gear incident: the (1+lambda)-ES polish traded four brief-stated
dimensions (bore 12->12.5 -- a shaft interface -- holes 8->6.76, boss 22->18.8, module
2.0->2.43) for wall margin, and every gate stayed green. Two mechanisms close it:

  * ``stated_numbers`` -- the protected-number set. A number qualifies only when it traces
    to the USER (prompt text in dimensional context, intake stated facts) or a cited
    STANDARD (registry-resolved hole targets) -- never to the model's own brief numbers
    (the ADR-0040 anchor invariant, applied to pinning). ``params.default_bounds`` pins
    value-matched float PARAMS to +/-2% (prevention, zero eval cost).
  * ``stated_dim_audit`` -- the acceptance-time claim anchor. One B-rep pass on the
    baseline AND the winner: a measurable stated bore may never get WORSE than the
    baseline beyond tolerance (non-degrade -- pre-existing author error is the revision
    loop's job, optimizer-ADDED drift is rejected), a spec measurable on the baseline but
    not on the winner is a rejection (differential measurability: erasure is never the
    cheap move, the ADR-0079 pattern at acceptance), and the winner's envelope may not
    degrade either (this MEASURES the derived module->tip channel the pins cannot see).
    Any exception inside the audit rejects the polish -- the pre-polish candidate passed
    the same gates minutes earlier, so keeping it is always safe; the cost is one wasted
    $0 fit, never a false PASS.

The drift tolerances are deliberately NOT the audit's 0.6mm identification window
(``spatial._DIA_TOL_MM``): the council measured that 0.6 as a drift tolerance would have
PASSED the incident's 12->12.5 bore. Identification and drift are different questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge_mind.agentic.brief import DesignBrief, parse_envelope_mm
from forge_mind.agentic.provenance import parse_stated_dimensions
from forge_mind.agentic.spatial import _DIA_TOL_MM, _HoleSpec, _planned_hole_specs

# Drift tolerances (ADR-0080): free-text stated dims get max(2%, 0.2mm); registry-resolved
# targets are exact by construction and get a tight absolute band.
_DRIFT_REL = 0.02
_DRIFT_ABS_MM = 0.2
_DRIFT_REGISTRY_MM = 0.1
_ENV_DRIFT_REL = 0.02

# A number in the user's prompt counts as dimensional only in mm / diameter / bore / angle
# context -- bare integers ("38 teeth") never qualify.
_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)
_DIA_CONTEXT_RE = re.compile(
    r"(?:[øØ⌀]\s*|\b(?:dia(?:meter)?|bore)\b\D{0,12}?)(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# A unit-bearing dimension CHAIN ("80 x 40 x 4 mm") states EVERY number in the chain, not just
# the one carrying the unit -- the measured M4 drifts (hook_j's 50 in "30 x 50 x 4 mm" -> 62.70
# under ACCEPTED) rode this gap. A chain without a trailing unit ("3 x 3 grid") stays free.
_DIM_CHAIN_RE = re.compile(
    # the multiplication sign is deliberate: users type "80 x 40 mm" or the unicode variant
    r"\b(\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)+)\s*mm\b",  # noqa: RUF001
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# Angles are stated dims too (hook_coat's "30 degrees upward" drifted to 33.4 under ACCEPTED).
_ANGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|deg(?:ree)?s?\b)", re.IGNORECASE)


def stated_numbers(
    prompt: str, brief: DesignBrief, extra_context: str | None = None
) -> frozenset[float]:
    """The protected-number set: user-prompt dims (mm/dia context) + intake stated facts +
    registry-resolved hole targets. The model's own brief numbers never enter (anchor
    invariant); the wall floor is excluded (a floor, not a dimension -- pinning wall-adjacent
    params would fight the wall repair)."""
    values: set[float] = set()
    for m in _MM_RE.finditer(prompt or ""):
        values.add(float(m.group(1)))
    for m in _DIA_CONTEXT_RE.finditer(prompt or ""):
        values.add(float(m.group(1)))
    for m in _DIM_CHAIN_RE.finditer(prompt or ""):
        values.update(float(n) for n in _NUM_RE.findall(m.group(1)))
    for m in _ANGLE_RE.finditer(prompt or ""):
        values.add(float(m.group(1)))
    if extra_context:
        values.update(d.value for d in parse_stated_dimensions(extra_context))
    try:
        for spec in _planned_hole_specs(brief):
            if spec.target_d_mm is not None and "ISO" in spec.source:
                values.add(float(spec.target_d_mm))
    except Exception:
        pass  # an unparseable plan loses only the registry source, never the prompt's
    wall = float(brief.min_wall_mm) if brief.min_wall_mm else 0.0
    return frozenset(v for v in values if 0.0 < v <= 10_000.0 and abs(v - wall) > 1e-6)


@dataclass(frozen=True)
class StatedDimAudit:
    """The audit verdict: ``ok`` accepts the polish; ``error`` marks a rejection that came
    from audit machinery failing (logged distinctly from measured drift -- flake visibility)."""

    ok: bool
    error: bool = False
    notes: list[str] = field(default_factory=list)


def _spec_tol(spec: _HoleSpec) -> float:
    if "ISO" in spec.source:
        return _DRIFT_REGISTRY_MM
    return max(_DRIFT_REL * float(spec.target_d_mm or 0.0), _DRIFT_ABS_MM)


def _bore_deviations(part: Any, specs: list[_HoleSpec]) -> dict[str, float | None]:
    """Per sized spec: the WORST |measured diameter - target| over the concave stacks inside
    the identification window, or None when no stack matches (unmeasurable for that spec)."""
    devs: dict[str, float | None] = {s.label: None for s in specs}
    if part is None:
        return devs
    from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

    stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    for spec in specs:
        target = float(spec.target_d_mm or 0.0)
        matched = [
            abs(2.0 * s.radius - target)
            for s in stacks
            if abs(2.0 * s.radius - target) <= _DIA_TOL_MM
        ]
        if matched:
            devs[spec.label] = max(matched)
    return devs


def _judge_bores(
    specs: list[_HoleSpec],
    base_devs: dict[str, float | None],
    win_devs: dict[str, float | None],
) -> tuple[bool, list[str]]:
    """The pure non-degrade judgment (unit-tested without any B-rep)."""
    notes: list[str] = []
    ok = True
    for spec in specs:
        target = float(spec.target_d_mm or 0.0)
        tol = _spec_tol(spec)
        b, w = base_devs.get(spec.label), win_devs.get(spec.label)
        if b is None and w is None:
            notes.append(f"'{spec.label}' Ø{target:g}: unverifiable (no matching bore) -- no claim")
            continue
        if w is None:
            ok = False
            notes.append(
                f"'{spec.label}' Ø{target:g}: measurable on the baseline but NOT on the "
                "winner -- the fit erased or moved it out of recognition (rejected)"
            )
            continue
        allowed = max(b if b is not None else 0.0, tol)
        if w > allowed + 1e-9:
            ok = False
            notes.append(
                f"'{spec.label}' Ø{target:g}: winner deviates {w:.2f}mm vs allowed "
                f"{allowed:.2f}mm (rejected)"
            )
        else:
            notes.append(f"'{spec.label}' Ø{target:g}: held ({w:.2f}mm <= {allowed:.2f}mm)")
    return ok, notes


def _sorted_extents(stl_path: Path | None) -> tuple[float, float, float] | None:
    if stl_path is None:
        return None
    import numpy as np
    import trimesh

    mesh: Any = trimesh.load(str(stl_path), force="mesh")
    ext = np.sort(np.asarray(mesh.bounding_box.extents, dtype=np.float64))
    return (float(ext[0]), float(ext[1]), float(ext[2]))


def _judge_envelope(
    env: tuple[float, float, float],
    base_ext: tuple[float, float, float] | None,
    win_ext: tuple[float, float, float],
) -> tuple[bool, list[str]]:
    """Per-axis non-degrade vs the brief envelope (sorted-vs-sorted, orientation-agnostic)."""
    want = sorted(env)
    ok = True
    worst = 0.0
    for i, e in enumerate(want):
        base_dev = abs(base_ext[i] - e) if base_ext is not None else 0.0
        allowed = max(base_dev, _ENV_DRIFT_REL * e)
        dev = abs(win_ext[i] - e)
        worst = max(worst, dev - allowed)
        if dev > allowed + 1e-9:
            ok = False
    if ok:
        return True, ["envelope: held"]
    return False, [f"envelope: winner drifted {worst:.1f}mm beyond the allowed band (rejected)"]


def stated_dim_audit(
    brief: DesignBrief,
    baseline_step: Path | None,
    winner_step: Path,
    baseline_stl: Path | None,
    winner_stl: Path,
) -> StatedDimAudit:
    """The acceptance-time claim anchor (ADR-0080). Run only by the polish accept path;
    vacuous pass (nothing claimed) when the brief states no auditable target."""
    try:
        specs = [s for s in _planned_hole_specs(brief) if s.target_d_mm is not None]
    except Exception:
        specs = []
    env = parse_envelope_mm(brief.overall_envelope_mm)
    if not specs and env is None:
        return StatedDimAudit(ok=True, notes=["stated-dim audit: no stated targets"])
    try:
        notes: list[str] = []
        ok = True
        if specs:
            from forge_mind.agentic.validate import _load_part

            base_part = _load_part(baseline_step) if baseline_step is not None else None
            win_part = _load_part(winner_step)
            b_ok, b_notes = _judge_bores(
                specs, _bore_deviations(base_part, specs), _bore_deviations(win_part, specs)
            )
            ok = ok and b_ok
            notes.extend(b_notes)
        if env is not None:
            win_ext = _sorted_extents(winner_stl)
            if win_ext is None:
                raise ValueError("winner STL unreadable for the envelope check")
            base_ext = None
            try:
                base_ext = _sorted_extents(baseline_stl)
            except Exception:
                base_ext = (
                    None  # a lenient baseline never blocks; the winner side may not fail soft
                )
            e_ok, e_notes = _judge_envelope(env, base_ext, win_ext)
            ok = ok and e_ok
            notes.extend(e_notes)
        return StatedDimAudit(ok=ok, notes=notes)
    except Exception as exc:  # fail CLOSED: an unverifiable winner is never accepted
        return StatedDimAudit(
            ok=False, error=True, notes=[f"stated-dim audit failed to run: {exc!r} (rejected)"]
        )


__all__ = ["StatedDimAudit", "stated_dim_audit", "stated_numbers"]
