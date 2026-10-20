"""Spatial feedback: tell the author WHERE a part is wrong, not just THAT.

The Wave 1 spike proved every failure is "the author works blind": it cannot see WHERE
two bodies fail to fuse (the hammer came back as 2 separate solids every iteration),
where a hole tool missed (the bracket shipped no holes), or where a wall is too thin
(the claw-tip sliver). This module localizes those from the B-rep + mesh and emits
author-readable descriptions the loop feeds into the next revision -- the measured #1
reliability lever.

Wave 2 ships DISCONNECTION localization first (the hammer's actual blocker); thin-region
and feature-count localization follow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def _bbox_str(bb: Any) -> str:
    return (
        f"x[{bb.min.X:.0f},{bb.max.X:.0f}] "
        f"y[{bb.min.Y:.0f},{bb.max.Y:.0f}] "
        f"z[{bb.min.Z:.0f},{bb.max.Z:.0f}]"
    )


def _bbox_gap(a: Any, b: Any) -> float | None:
    """Axis-separation distance between two bounding boxes (0 if they overlap on every axis)."""
    try:
        dx = max(0.0, a.min.X - b.max.X, b.min.X - a.max.X)
        dy = max(0.0, a.min.Y - b.max.Y, b.min.Y - a.max.Y)
        dz = max(0.0, a.min.Z - b.max.Z, b.min.Z - a.max.Z)
        return float((dx * dx + dy * dy + dz * dz) ** 0.5)
    except Exception:
        return None


# Any coordinate beyond this is not a real part dimension (the bed is 250mm; briefs run <300mm):
# it is kernel garbage from an invalid operation.
_SANE_COORD_MM = 1e6


def _is_degenerate(bb: Any, vol: float) -> bool:
    """A body whose bounds or volume cannot belong to a real part (invalid-op garbage)."""
    if vol <= 0.0:
        return True
    try:
        return any(
            abs(float(v)) > _SANE_COORD_MM
            for v in (bb.min.X, bb.min.Y, bb.min.Z, bb.max.X, bb.max.Y, bb.max.Z)
        )
    except Exception:
        return False


def disconnection_feedback(part: Any) -> list[str]:
    """Localize a multi-body B-rep: each separate solid's bbox + volume + the gap + how to fuse.

    Returns [] for a single-body (or unreadable) part. The author gets the COORDINATES of the
    bodies that failed to fuse, not just 'it is disconnected', so it can target the join.
    """
    if part is None:
        return []
    try:
        solids = list(part.solids())
    except Exception:
        return []
    info: list[tuple[Any, float]] = []
    for s in solids:
        try:
            info.append((s.bounding_box(), float(s.volume)))
        except Exception:
            continue
    if len(info) <= 1:
        return []
    info.sort(key=lambda t: t[1], reverse=True)  # largest body first

    lines = [
        f"DISCONNECTED: the part is {len(info)} SEPARATE solids, not one fused body. They must "
        "become ONE connected solid. The separate bodies are:"
    ]
    for i, (bb, vol) in enumerate(info[:6], 1):
        if _is_degenerate(bb, vol):
            # Measured 2026-07-18 (cable clip, 8/8 runs): an invalid sweep produced a body at
            # x ~ -5.9e15 plus a NEGATIVE-volume body, and this feedback printed the 16-digit
            # coordinates verbatim -- unactionable noise the author cannot reason about. Name
            # the state instead and demand a rebuild of that feature.
            lines.append(
                f"  body {i}: DEGENERATE geometry (absurd coordinates or non-positive volume) -- "
                "the operation that made it (usually an invalid sweep profile/path or a broken "
                "boolean) produced garbage. Do not try to move or fuse this body: REBUILD that "
                "feature from scratch with a simpler construction (extrude a closed sketch, or "
                "revolve a valid half-profile)."
            )
        else:
            lines.append(f"  body {i}: {_bbox_str(bb)} volume {vol:.0f} mm^3")
    if any(_is_degenerate(bb, vol) for bb, vol in info[:2]):
        return lines  # a gap distance to garbage coordinates is itself garbage -- omit it
    gap = _bbox_gap(info[0][0], info[1][0])
    if gap is not None:
        nearness = (
            "they OVERLAP in bounds but still didn't fuse" if gap == 0.0 else f"~{gap:.1f}mm apart"
        )
        lines.append(
            f"The two largest bodies are {nearness}. EXTEND or OVERLAP the bodies so they "
            "physically share material BEFORE you union them -- a union of TOUCHING solids fuses "
            "into one body, but even a hairline gap (or a surface-only touch) leaves them separate. "
            "Make each feature start INSIDE the body it joins (e.g. the handle top passes UP INTO "
            "the head; the claw root begins INSIDE the head, not flush against its face). KEEP "
            "every feature -- fuse by OVERLAPPING them; NEVER delete or simplify a feature, and do "
            "NOT collapse the part into a plain block, to achieve one body."
        )
    return lines


def thin_region_feedback(mesh: Any, min_wall_mm: float, samples: int = 2500) -> list[str]:
    """Localize sub-min-wall regions: ray-cast inward from sampled surface points to measure
    local wall thickness, then report the bbox + centroid of the thin cluster so the author
    can thicken THAT feature (the claw tip / blade / chamfer tapering to a knife edge) rather
    than guessing. Best-effort: returns [] on any failure (never blocks validation)."""
    try:
        import numpy as np

        pts, face_idx = mesh.sample(samples, return_index=True)
        normals = mesh.face_normals[face_idx]
        origins = pts - normals * 1e-3  # nudge just inside the surface
        locs, ray_idx, _tri = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
        if len(locs) == 0:
            return []
        thickness = np.linalg.norm(locs - origins[ray_idx], axis=1)
        thin = locs[thickness < min_wall_mm]
        if len(thin) < 3:
            return []
        lo, hi, c = thin.min(0), thin.max(0), thin.mean(0)
        return [
            f"THIN-WALL location: ~{len(thin)} of {samples} sampled points are under {min_wall_mm}mm "
            f"thick, clustered at x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] "
            f"z[{lo[2]:.0f},{hi[2]:.0f}] (centroid {c[0]:.0f},{c[1]:.0f},{c[2]:.0f}). Thicken the "
            f"feature THERE -- a tip/edge/web tapering to a knife edge -- so no wall is under "
            f"{min_wall_mm}mm (give it a small flat or a fuller cross-section); do not thin the rest."
        ]
    except Exception:
        return []


_HOLE_WORDS = ("hole", "bore", "bolt", "drill")

# Feature-name semantics for the hole plan (engine deep review, 2026-07-04). English compound
# names are HEAD-FINAL: the last word says what the feature IS; hole words earlier in the name
# only describe association. Counting by keyword-anywhere manufactured phantom demands the
# authors could never satisfy: a solid "bolt EAR" became two planned through-holes
# (bracket_pipe_clamp latched features_missing at ~6-of-5 forever); a "bore CHAMFER" x2
# demanded genus 3 of a correct genus-1 pulley; and a blind "5 mm deep" set-screw was ordered
# to pierce (the knob was then drilled through and the wrong part certified).
_SOLID_HEADS = frozenset(
    {
        "ear",
        "ears",
        "boss",
        "bosses",
        "tab",
        "tabs",
        "lug",
        "lugs",
        "pin",
        "pins",
        "post",
        "posts",
        "standoff",
        "standoffs",
        "spoke",
        "spokes",
        "rib",
        "ribs",
    }
)
_MODIFIER_HEADS = frozenset({"chamfer", "chamfers", "fillet", "fillets"})
_DEPTH_WORD_RE = re.compile(r"\b(?:deep|depth)\b")


def _head_word(name: str) -> str:
    words = re.findall(r"[a-z]+", name.lower())
    return words[-1] if words else ""


def _plans_through_hole(f: Any) -> bool:
    """Does this brief feature plan NEW through-hole(s)?

    Name-based (not method) for the measured bracket reason: 'bolt holes' + method 'pocket'
    must still count. But the name is read by its HEAD word, and depth semantics are honored:
      * a solid-head compound ('bolt ear') plans no hole;
      * a modifier-head compound ('bore chamfer') decorates an existing hole, adds none;
      * 'blind', or a stated depth ('5 mm deep') without 'through', means NOT through."""
    name = f.name.lower()
    if not any(w in name for w in _HOLE_WORDS):
        return False
    head = _head_word(name)
    if head in _SOLID_HEADS or head in _MODIFIER_HEADS:
        return False
    dims = str(getattr(f, "approx_dims_mm", "") or "").lower()
    text = f"{name} {dims}"
    if "blind" in text:
        return False
    # a stated depth ('5 mm deep') without 'through' anywhere = a blind feature
    return not ("through" not in text and _DEPTH_WORD_RE.search(text))


def _expected_through_holes(brief: Any) -> int:
    """Sum the counts of brief features that plan THROUGH holes (see _plans_through_hole)."""
    total = 0
    for f in brief.features:
        if _plans_through_hole(f):
            total += max(1, getattr(f, "count", 1))
    return total


@dataclass(frozen=True)
class _HoleSpec:
    """One planned hole group from the brief: its label, instance count, and -- when the
    brief states one -- the TARGET diameter (resolved from the standards registry for
    metric designations, or parsed from the dims text). None = size unstated."""

    label: str
    count: int
    target_d_mm: float | None
    source: str  # where the diameter came from, for the feedback text
    # Nominal fastener diameter only for registry-derived clearance holes. This
    # separates loose assignment to a plan item from process-fit conformance.
    fit_nominal_d_mm: float | None = None


_METRIC_RE = re.compile(r"\bM(\d+(?:\.\d+)?)\b", re.IGNORECASE)
# Diameter-in-text parser. Two measured misparse fixes (engine deep review, 2026-07-04):
#   * the keyword->number gap must never cross a DEPTH word -- "set-screw bore, depth 5 mm"
#     once parsed as a STATED Ø5.0, outranked ADR-0081's nominal, and the engine ordered the
#     knob's blind M3 drilled out to Ø5 through-into-the-bore, then certified it;
#   * "5 mm through-bore" must parse as Ø5 (branch 2 accepts the hyphenated form), so the
#     leftmost match is the real bore, not the trailing "bore, 10 mm countersink" fragment
#     that once demanded a Ø10 bore on the coat hook.
_DIA_RE = re.compile(
    r"(?:[øØ⌀]\s*|(?:\bdia(?:meter)?|\bbore)(?:(?!\b(?:depth|deep)\b)\D){0,12})"
    r"(\d+(?:\.\d+)?)\s*mm"
    r"|(\d+(?:\.\d+)?)\s*mm\s*(?:through-)?(?:dia(?:meter)?|bore|hole)",
    re.IGNORECASE,
)
_DIA_TOL_MM = 0.6  # a measured bore within this of the target counts as that spec's hole


def _metric_clearance_d(designation: str) -> float | None:
    """Clearance-hole diameter for an M-designation via the REAL standards registry
    (single-sourced with placement + conformance); None for sizes it doesn't know."""
    try:
        from forge_mind.dsl.standards.fasteners import clearance_hole

        return float(clearance_hole(f"M{designation}"))
    except Exception:
        return None


def _metric_thread_nominal_d(designation: str) -> float | None:
    """Nominal (major) diameter for an M-designation -- the target for a THREADED / tapped /
    set-screw hole (ADR-0081: a threaded hole is identified by its nominal size, not the tap
    drill and not the bolt clearance). None for sizes the registry doesn't know."""
    try:
        from forge_mind.dsl.standards.fasteners import thread_nominal

        return float(thread_nominal(f"M{designation}"))
    except Exception:
        return None


_THREAD_WORDS = ("thread", "tapped", "tap ", "set-screw", "set screw", "setscrew", "grub")


def _is_threaded(name: str, dims: str) -> bool:
    """A hole feature that RECEIVES a thread (tapped bore / set-screw), vs a clearance hole a
    bolt passes through. Threaded holes target the nominal major diameter."""
    text = f"{name} {dims}".lower()
    return any(w in text for w in _THREAD_WORDS)


def _planned_hole_specs(brief: Any) -> list[_HoleSpec]:
    """The brief's through-hole plan with target diameters where stated. Mirrors the
    detection rules of ``_expected_through_holes`` (same features, same blind exclusion)
    so the audit and the genus gate always talk about the same plan."""
    specs: list[_HoleSpec] = []
    for f in brief.features:
        if not _plans_through_hole(f):
            continue
        name = f.name.lower()
        dims = str(getattr(f, "approx_dims_mm", "") or "")
        count = max(1, getattr(f, "count", 1))
        target: float | None = None
        source = ""
        fit_nominal: float | None = None
        std = str(getattr(f, "standard", "") or "")
        # A diameter STATED in the dims is EXACT and outranks the fastener-standard
        # conversion (bench row 17, ratchet pawl: the prompt said 'pivot bore 6 mm
        # diameter', the brief tagged it M6, and the registry silently redrilled it to
        # the 6.6 clearance -- the stated number is the user's dimension, the tag a hint).
        d = _DIA_RE.search(dims)
        if d:
            target = float(d.group(1) or d.group(2))
            source = "stated in the brief"
        if target is None:
            m = _METRIC_RE.search(std) or _METRIC_RE.search(f"{f.name} {dims}")
            if m:
                # A THREADED/tapped/set-screw hole is identified by its NOMINAL major diameter
                # (ADR-0081, bench row 18: pulley set-screw M4 and handle M8 threaded bore --
                # the tap-drill/clearance target was a three-way mismatch with the author and
                # the rubric); a plain clearance hole passes a bolt through -> ISO 273 clearance.
                if _is_threaded(name, dims):
                    target = _metric_thread_nominal_d(m.group(1))
                    if target is not None:
                        source = f"M{m.group(1)} thread nominal"
                else:
                    target = _metric_clearance_d(m.group(1))
                    if target is not None:
                        source = f"M{m.group(1)} clearance, ISO 273"
                        fit_nominal = float(m.group(1))
        specs.append(
            _HoleSpec(
                label=f.name,
                count=count,
                target_d_mm=target,
                source=source,
                fit_nominal_d_mm=fit_nominal,
            )
        )
    return specs


def _stack_segments(s: Any) -> int:
    """A stack counts for its SEGMENTS (axially-separated wall penetrations on one axis):
    two clevis ears on one pivot axis are one coaxial stack but TWO physical bores --
    counting stacks alone audited a drilled pair as 1/2 (measured live on the turret)
    and sent the author chasing a phantom missing hole."""
    return max(1, int(getattr(s, "segments", 1)))


def _reconcile_hole_plan(
    specs: list[_HoleSpec], stacks: list[Any]
) -> tuple[list[str], int, list[Any], list[tuple[_HoleSpec, Any]]]:
    """Assign measured bore stacks to the brief's hole plan; report what stays unmet.

    Assignment is NEAREST-DIAMETER, sized specs first (engine deep review, 2026-07-04): the
    old first-fit greedy let a Ø5.0 detent spec consume Ø4.5 corner holes (both inside the
    0.6 window) and then reported the CORNER holes missing -- the author rewrote a healthy
    block five times chasing the misattribution. And unsized specs ran FIRST, popping
    arbitrary stacks a later sized spec needed; they now consume leftovers only.

    SAME-diameter specs share the pool (measured on the hdd35 golden re-run, 2026-07-07):
    "nearest spec wins" broke ties by list order, so with 'case mount hole' and 'HDD side
    mount hole' both Ø3.4 the FIRST spec was every stack's winner and the second could never
    claim a bore -- a COMPLETE part reported 8 phantom-missing holes and the author chased
    them for six iterations. A stack is now claimable by every spec that TIES its nearest
    distance; earlier specs consume first, later same-size specs take the leftovers (labels
    may cross-assign inside a tie group -- specs carry no positions, aggregate count is the
    honest claim).

    Shared by the advisory feedback AND the completeness gate (``measured_holes_short``) so
    the loop's pressure and the author's revision context always agree. Returns
    (sized-spec deficit descriptions, unmet unsized-spec segment count, unassigned stacks,
    assigned sized-spec/stack pairs). Assignment remains deliberately loose; callers apply
    conformance rules separately after identity is established."""
    sized_specs = [p for p in specs if p.target_d_mm is not None]

    def _best_dist_for(s: Any) -> float | None:
        if not sized_specs:
            return None
        d = min(abs(2 * s.radius - float(p.target_d_mm or 0.0)) for p in sized_specs)
        return d if d <= _DIA_TOL_MM else None

    remaining = list(stacks)
    deficits: list[str] = []
    assignments: list[tuple[_HoleSpec, Any]] = []
    for spec in sized_specs:
        target = float(spec.target_d_mm or 0.0)
        # a stack belongs to this spec when the spec TIES the stack's nearest sized target
        matched = sorted(
            (
                s
                for s in remaining
                if (bd := _best_dist_for(s)) is not None
                and abs(abs(2 * s.radius - target) - bd) < 1e-9
            ),
            key=lambda s: abs(2 * s.radius - target),
        )
        need = spec.count
        for s in matched:
            if need <= 0:
                break
            remaining.remove(s)
            assignments.append((spec, s))
            need -= _stack_segments(s)
        if need > 0:
            deficits.append(
                f"{need} x Ø{target:.2f}mm "
                f"('{spec.label}'{', ' + spec.source if spec.source else ''})"
            )
    unsized_short = 0
    for spec in [p for p in specs if p.target_d_mm is None]:
        # Unsized spec: consume LEFTOVER bores worth `count` segments (nothing size-specific
        # to say; a leftover shortfall still means the plan is not met).
        need = spec.count
        while remaining and need > 0:
            need -= _stack_segments(remaining.pop(0))
        unsized_short += max(0, need)
    return deficits, unsized_short, remaining, assignments


def measured_hole_feedback(part: Any, brief: Any) -> list[str]:
    """Measured-hole audit (ADR-0035 B1, size-aware): plan vs B-rep, by DIAMETER.

    The genus gate proves the through-hole COUNT; this measures the B-rep directly --
    every concave cylinder stack with its diameter and center -- and reconciles it
    against the brief's hole plan with TARGET diameters (the registry's clearance size
    for M-designations). So a revision pass sees "you drilled 2 of 4 Ø4.5 mounting
    holes; the Ø2.0 bores you cut match no planned size -- likely under-sized" instead
    of guessing. Catches the size-blind genus gap: a part with the right COUNT of
    wrong-SIZE holes is named, not waved through silently.
    Advisory: feeds ``issues`` (the author's revision context), never gates ``passed``.
    Best-effort: returns [] on any failure or when there is nothing to say."""
    if part is None:
        return []
    specs = _planned_hole_specs(brief)
    expected = sum(s.count for s in specs)
    if expected <= 0:
        return []
    try:
        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    except Exception:
        return []
    if not stacks:
        return [
            f"MEASURED BORES: the B-rep contains ZERO concave cylindrical bores, but the brief "
            f"plans ~{expected} hole(s). Any cylinders present are solid PINS/bosses, not holes "
            "-- cut the holes with SUBTRACT-mode tools that pass through the material."
        ]

    deficits, _unsized_short, remaining, _assignments = _reconcile_hole_plan(specs, stacks)
    lines: list[str] = []
    if deficits:
        found = (
            "; ".join(
                f"Ø{2 * s.radius:.2f}mm at ({s.axis_point[0]:.0f}, {s.axis_point[1]:.0f}, "
                f"{s.axis_point[2]:.0f})"
                for s in stacks[:8]
            )
            or "none"
        )
        lines.append(
            f"MEASURED BORES vs PLAN: still MISSING {', '.join(deficits)}. The B-rep's "
            f"measured concave bores are: {found}. Add the missing hole(s) at the planned "
            "positions with the planned diameters; KEEP every measured bore in place."
        )
        wrong = [
            s
            for s in remaining
            if all(  # bores that satisfy no planned size
                spec.target_d_mm is None or abs(2 * s.radius - spec.target_d_mm) > _DIA_TOL_MM
                for spec in specs
            )
        ]
        if wrong:
            w = "; ".join(
                f"Ø{2 * s.radius:.2f}mm at ({s.axis_point[0]:.0f}, {s.axis_point[1]:.0f}, "
                f"{s.axis_point[2]:.0f})"
                for s in wrong[:4]
            )
            lines.append(
                f"SIZE MISMATCH: {len(wrong)} measured bore(s) match NO planned diameter -- {w}. "
                "These are likely the planned holes cut at the WRONG size; resize them to the "
                "planned diameters instead of adding extra holes."
            )
    elif sum(_stack_segments(s) for s in stacks) < expected:
        total = sum(_stack_segments(s) for s in stacks)
        listed = "; ".join(
            f"Ø{2 * s.radius:.2f}mm at ({s.axis_point[0]:.0f}, {s.axis_point[1]:.0f}, "
            f"{s.axis_point[2]:.0f})"
            for s in stacks[:8]
        )
        lines.append(
            f"MEASURED BORES: the B-rep contains {total} concave bore(s) -- {listed} -- but "
            f"the brief plans ~{expected}. Add the missing {expected - total} hole(s); keep "
            "the measured ones in place."
        )
    return lines


def measured_holes_short(part: Any, brief: Any) -> bool:
    """True when the B-rep is missing planned holes -- per SPEC and size-aware.

    The completeness signal the genus gate (``feature_count_feedback``) cannot trust: a grip loop,
    ring, or handle inflates the mesh genus, so a dropped REQUIRED hole can still satisfy
    ``genus >= expected``. Counting the actual concave-bore SEGMENTS cannot be fooled that way (audit
    H8, ADR-0065) -- and the count is reconciled per spec by NEAREST DIAMETER (the same assignment
    the advisory feedback shows the author), because a TOTAL-count compare let 16 off-plan Ø7 vent
    bores mask 4 never-cut Ø3.4 rail holes on the shipped hdd35 golden (TEST_LOG row 38). FAIL-OPEN
    -- returns False when nothing is planned or the measurement fails, so it only ADDS a missing
    verdict the genus gate would have waved through, never a false FAIL on unmeasurable geometry."""
    if part is None:
        return False
    specs = _planned_hole_specs(brief)
    if sum(s.count for s in specs) <= 0:
        return False
    try:
        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    except Exception:
        return False
    deficits, unsized_short, _remaining, _assignments = _reconcile_hole_plan(specs, stacks)
    return bool(deficits) or unsized_short > 0


def measured_hole_conformance_issues(part: Any, brief: Any, material: Any) -> list[str]:
    """Hard fit-floor failures among already-assigned standard clearance bores.

    ``_DIA_TOL_MM`` is only an association window. Once a measured bore is assigned
    to an ISO clearance-hole plan, it must also clear the selected material's F2
    minimum diametral running-fit law. Unmeasurable geometry stays owned by the
    existing completeness gate.
    """
    if part is None:
        return []
    specs = _planned_hole_specs(brief)
    if not any(spec.fit_nominal_d_mm is not None for spec in specs):
        return []
    try:
        from forge_mind.assembly.fits import minimum_running_diametral_clearance_mm
        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
        _deficits, _unsized, _remaining, assignments = _reconcile_hole_plan(specs, stacks)
        fit_floor = minimum_running_diametral_clearance_mm(material)
    except Exception:
        return []
    issues: list[str] = []
    for spec, stack in assignments:
        if spec.fit_nominal_d_mm is None:
            continue
        actual = 2.0 * float(stack.radius)
        required = spec.fit_nominal_d_mm + fit_floor
        if actual < required - 1e-6:
            issues.append(
                f"BORE FIT: '{spec.label}' measures {actual:.2f} mm but a "
                f"{spec.fit_nominal_d_mm:.2f} mm fastener needs at least {required:.2f} mm "
                f"in {material.name} ({fit_floor:.2f} mm diametral process clearance). "
                "The bore was associated with the right plan item but is too tight to assemble."
            )
    return issues


def mesh_genus(mesh: Any) -> int | None:
    """The watertight mesh's through-hole count (euler = 2 - 2*genus); None if unmeasurable."""
    try:
        return max(0, round(1 - int(mesh.euler_number) / 2))
    except Exception:
        return None


# ---------------------------------------------------------------------------------
# Measured feedback for ALL features (the lever-c pattern extended beyond holes):
# CONVEX cylindrical features (bosses / pins / standoffs) audited by stated diameter,
# and a wall-thickness CLUSTER map when the part is under the minimum. Both are
# advisory (never gate `passed`) and fail-open ([] on any doubt) -- false feedback
# poisons the author worse than no feedback. Feature kinds with no reliable B-rep
# measurement (ribs, slots, fins -- planar-cluster heuristics misfire on real parts)
# are deliberately NOT counted here; the brief carries no positions either, so
# per-feature region presence is the feature-plan mapping's job (lever f), not a
# geometry guess.
# ---------------------------------------------------------------------------------

_BOSS_WORDS = ("boss", "standoff", "stand-off", "pin", "peg", "stud", "pillar")


def _planned_boss_specs(brief: Any) -> list[_HoleSpec]:
    """Boss/pin-kind features WITH a stated diameter (the only reliably auditable case:
    without a target size, the part's own round body would count as a 'boss'). Features
    whose dims state no diameter are skipped -- never guessed."""
    specs: list[_HoleSpec] = []
    for f in brief.features:
        name = f.name.lower()
        if not any(w in name for w in _BOSS_WORDS):
            continue
        if any(w in name for w in _HOLE_WORDS):
            continue  # 'pin hole' is a hole plan, not a boss plan
        dims = str(getattr(f, "approx_dims_mm", "") or "")
        d = _DIA_RE.search(dims)
        if not d:
            continue
        specs.append(
            _HoleSpec(
                label=f.name,
                count=max(1, getattr(f, "count", 1)),
                target_d_mm=float(d.group(1) or d.group(2)),
                source="stated in the brief",
            )
        )
    return specs


def measured_boss_feedback(part: Any, brief: Any) -> list[str]:
    """Measured-boss audit (lever-c pattern, convex side): the brief's planned bosses /
    pins / standoffs with stated diameters vs the B-rep's measured CONVEX cylinder stacks.
    A deficit is named with the measured stacks' diameters and centers so the author adds
    the missing boss THERE instead of guessing. Advisory; [] on any doubt."""
    if part is None:
        return []
    specs = _planned_boss_specs(brief)
    if not specs:
        return []
    try:
        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is False]
    except Exception:
        return []
    remaining = list(stacks)
    deficits: list[str] = []
    for spec in specs:
        if spec.target_d_mm is None:
            continue
        matched = [s for s in remaining if abs(2 * s.radius - spec.target_d_mm) <= _DIA_TOL_MM]
        used = matched[: spec.count]
        for u in used:
            remaining.remove(u)
        if len(used) < spec.count:
            deficits.append(
                f"{spec.count - len(used)} x Ø{spec.target_d_mm:.2f}mm ('{spec.label}')"
            )
    if not deficits:
        return []
    found = (
        "; ".join(
            f"Ø{2 * s.radius:.2f}mm at ({s.axis_point[0]:.0f}, {s.axis_point[1]:.0f}, "
            f"{s.axis_point[2]:.0f})"
            for s in stacks[:8]
        )
        or "none"
    )
    return [
        f"MEASURED BOSSES vs PLAN: still MISSING {', '.join(deficits)} (convex cylindrical "
        f"bosses/pins the brief plans). The B-rep's measured convex stacks are: {found}. Add "
        "the missing boss(es) at the planned positions with the planned diameters -- a boss "
        "must OVERLAP into the body it stands on so the union fuses."
    ]


def wall_cluster_feedback(mesh: Any, min_wall_mm: float, samples: int = 2500) -> list[str]:
    """A wall-thickness CLUSTER map for a part whose walls dip under the minimum: where
    ``thin_region_feedback`` localizes the worst spot, this gives the author the part's
    measured thickness STRUCTURE ("walls cluster at ~2.0mm (61%), ~6.0mm (24%)"), so the
    fix thickens the right cluster instead of fattening everything. Emits only when the
    sampled 2nd-percentile thickness is under the minimum; [] otherwise / on any failure."""
    try:
        from collections import Counter

        import numpy as np

        pts, face_idx = mesh.sample(samples, return_index=True)
        normals = mesh.face_normals[face_idx]
        origins = pts - normals * 1e-3
        locs, ray_idx, _tri = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
        if len(locs) < 30:
            return []
        thickness = np.linalg.norm(locs - origins[ray_idx], axis=1)
        if len(thickness) < 30:
            return []
        thinnest = float(np.percentile(thickness, 2))
        if thinnest >= min_wall_mm:
            return []
        bins = np.round(thickness * 2.0) / 2.0  # 0.5mm bins
        top = Counter(bins.tolist()).most_common(3)
        total = len(bins)
        clusters = ", ".join(
            f"~{b:.1f}mm ({100 * c / total:.0f}%)" for b, c in sorted(top, key=lambda t: t[0])
        )
        return [
            f"WALL MAP: sampled wall thickness clusters at {clusters}; the thinnest 2% of "
            f"samples are ~{thinnest:.1f}mm vs the {min_wall_mm}mm minimum. Thicken ONLY the "
            "sub-minimum cluster (the thin feature), keep the rest as-is."
        ]
    except Exception:
        return []


def feature_count_feedback(mesh: Any, brief: Any) -> tuple[list[str], bool]:
    """Compare the geometry's THROUGH-hole count (mesh genus) to the brief's expected holes.

    Returns ``(feedback, missing)``. ``missing`` is True when the part is short of its planned
    through-holes -- the bracket's measured gap (planned 4 bolt holes, geometry has 0 because the
    cut tools never passed through). A part with no planned holes is never flagged. genus = the
    through-hole count of a single watertight body (euler = 2 - 2*genus).
    """
    expected = _expected_through_holes(brief)
    if expected <= 0:
        return [], False
    genus = mesh_genus(mesh)
    if genus is None:
        return [], False
    if genus >= expected:
        return [], False
    return (
        [
            f"MISSING FEATURES: the brief plans ~{expected} through-hole(s) (bolt / clearance "
            f"holes) but the geometry has only {genus}. Add the missing {expected - genus} hole(s) "
            "where the brief places them, and ensure each cutting tool passes ALL THE WAY THROUGH "
            "(extend the cut past both faces so the hole opens on both sides)."
        ],
        True,
    )


__all__ = [
    "disconnection_feedback",
    "feature_count_feedback",
    "measured_boss_feedback",
    "measured_hole_conformance_issues",
    "measured_hole_feedback",
    "measured_holes_short",
    "mesh_genus",
    "thin_region_feedback",
    "wall_cluster_feedback",
]
