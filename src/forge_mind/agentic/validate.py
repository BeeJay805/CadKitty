"""Validate an authored part: feed the sandbox's raw STL/STEP through the EXISTING
validator (mesh / features / printability tiers) plus an agentic functional gate.

The bounded-DSL validator expects a ``(CADModel, PartProgram)``; the agentic path has
neither -- only an STL + STEP on disk from sandboxed code. This adapter wraps them: a
constraints-carrier ``PartProgram`` (a 1mm dummy op to satisfy the schema, carrying the
brief's min_wall + print orientation) and a ``CADModel`` (the STEP-loaded B-rep + the
mesh bbox). The geometry tiers MEASURE the STL directly, so the dummy op is never read;
the interfaces + conformance tiers auto-skip on empty declarations.

On top we add the no-false-PASS FUNCTIONAL gate the DSL validator cannot express: the
part must be ONE connected, watertight solid of positive volume. (A hammer that comes
out as three unfused bodies is 'watertight' per-body yet functionally broken.) We gate
on FAILs only -- a WARN is advisory and does not block, but flips no part to PASS.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh

from forge_mind.agentic.brep_checks import brep_report
from forge_mind.agentic.brief import DesignBrief, parse_envelope_mm
from forge_mind.agentic.material_report import MaterialReport, evaluate_material
from forge_mind.agentic.radial_count import radial_count_feedback
from forge_mind.agentic.spatial import (
    disconnection_feedback,
    feature_count_feedback,
    measured_boss_feedback,
    measured_hole_conformance_issues,
    measured_hole_feedback,
    measured_holes_short,
    mesh_genus,
    thin_region_feedback,
    wall_cluster_feedback,
)
from forge_mind.agentic.spin_check import balance_feedback
from forge_mind.agentic.standards_audit import audit_standards
from forge_mind.compiler.compile import CADModel
from forge_mind.compiler.topology_tags import TopologyIndex
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.fabrication import FabricationMode
from forge_mind.materials import Material, resolve_material
from forge_mind.slicing.slicer import cell_lane_enabled, slicer_gate_enabled
from forge_mind.types import BBox, Vec3
from forge_mind.validator.report import CheckStatus, ValidationReport
from forge_mind.validator.run import validate

# Checks the agentic path treats as ADVISORY (surfaced, never gates `passed`): arbitrary user parts
# legitimately have printable overhangs and fillet/gusset knife-edges that the sampled min-wall
# over-flags. The DSL/template path keeps these STRICT (ADR-0029). Mirrored in api/core's view.
ADVISORY_CHECK_IDS = frozenset({"printability.overhang_max_angle", "features.min_wall_thickness"})


@dataclass(frozen=True)
class AgenticValidation:
    """Outcome of validating an authored part. ``passed`` = no FAIL checks AND the
    functional gate holds (one connected watertight solid). ``issues`` is the prioritized
    list of what to fix, fed verbatim to the author's revision pass."""

    passed: bool
    hard_ok: bool  # watertight + ONE solid + positive volume: a coherent, renderable part
    report: ValidationReport | None
    body_count: int
    watertight: bool
    volume_mm3: float
    issues: list[str] = field(default_factory=list)
    material_report: MaterialReport | None = None  # mass/cost/spec + material wall verdict
    standards_cites: list[str] = field(
        default_factory=list
    )  # verified ISO/DIN dims the author used
    features_missing: bool = False  # short of the brief's planned through-holes: a concrete,
    # fixable gap (drives the loop's plateau patience -- see loop._plateau_stop), distinct from a
    # vague quality plateau. Already folded into `passed`; exposed so the loop can read it directly.
    through_holes: int | None = None  # measured mesh genus (None = unmeasurable). The loop
    # compares iterations: a revision whose genus DROPS vs the working best erased holes the
    # part already had -- the author gets told explicitly instead of rediscovering the gap.

    def feedback(self) -> str:
        return "\n".join(f"- {issue}" for issue in self.issues)


def _load_part(step_path: Path) -> Any:
    """Load the editable B-rep from the sandbox's STEP (for physics / interface tiers).
    Best-effort: a load failure just means no mass metric, never a validation failure."""
    try:
        from build123d import import_step

        return import_step(str(step_path))
    except Exception:
        return None


def _count_solids(part: Any) -> int:
    """Number of disjoint solids in the B-rep (1 = a single fused body). The authoritative
    'is it one part' check, from the kernel rather than mesh connectivity. 1 if unknown."""
    if part is None:
        return 1
    try:
        return max(1, len(part.solids()))
    except Exception:
        return 1


def _measured_min_wall(report: ValidationReport | None) -> float | None:
    """The thinnest-wall value the feature tier already measured (mm), parsed from its check.

    Reuses the validator's measurement so the material gate adds no extra ray-casting. None when
    the check is absent or unmeasurable -- never fabricate a wall value (no false material FAIL).
    """
    if report is None:
        return None
    check = report.check("features.min_wall_thickness")
    if check is None or check.actual is None:
        return None
    try:
        return float(check.actual.split()[0])  # "1.23 mm" -> 1.23; "unmeasurable" -> ValueError
    except (ValueError, IndexError):
        return None


def _is_degenerate(watertight: bool, eff_volume: float, bbox_vol: float) -> bool:
    """Is a WATERTIGHT solid a near-zero-volume sliver? (no-false-PASS for the wine-glass collapse).

    Conservative by design: tiny ABSOLUTE volume (< 200 mm^3) AND under 0.1% of the bbox. A
    legitimately thin hollow part is > 1% fill, so it is never flagged; the trip case is a collapsed
    / self-intersecting profile that OCC still reports watertight (the two-rail wine-glass that came
    out 18 mm^3 inside a normal-size envelope).
    """
    return watertight and 0 < eff_volume < 200.0 and bbox_vol > 0 and eff_volume < 0.001 * bbox_vol


def _step_load_failed(step_path: Path, part: Any) -> bool:
    """Did a REAL STEP file fail to yield a usable B-rep? (no-false-PASS for the step downgrade).

    A STEP that did not parse does NOT reliably raise: build123d returns an EMPTY Compound (0
    solids), not None, on a broken file. Either way -- ``part is None`` OR a part with zero solids --
    means the kernel has no solid to check, so ``_count_solids`` defaults body_count to 1 and
    ``brep_report`` runs neutral, and a multi-body / invalid solid that only exported an STL would
    otherwise PASS. We BLOCK that. We fail-open (return False -> skip) whenever the STEP file is
    absent, so a legitimately step-less caller never gets a false FAIL; in the production loop the
    sandbox guarantees the STEP exists on disk before validation, so this fires exactly on a genuine
    downgrade.
    """
    try:
        if not step_path.exists():
            return False
    except Exception:
        return False
    if part is None:
        return True
    try:
        return len(part.solids()) == 0
    except Exception:
        # The STEP loaded into SOMETHING we cannot enumerate -- do not fabricate a downgrade.
        return False


def _wall_unmeasurable(report: ValidationReport | None) -> bool:
    """Is the min-wall check present but UNMEASURABLE (not measured-and-fine)?

    The feature tier emits a WARN with ``actual='unmeasurable'`` when no inward ray hit a wall, so
    the thinnest wall is unknown -- the part is NOT certified thick enough, it is uncertified. We
    distinguish that from a measured PASS (``actual='1.23 mm'``) and a measured FAIL (already gated):
    only the explicit 'unmeasurable' marker blocks. Absent check -> not blocking (fail-open).
    """
    if report is None:
        return False
    check = report.check("features.min_wall_thickness")
    if check is None:
        return False
    return check.status == CheckStatus.WARN and check.actual == "unmeasurable"


def _envelope_mismatch(
    bbox: BBox, brief: DesignBrief, *, tol_frac: float = 0.15
) -> tuple[bool, str | None]:
    """Compare the built bbox extents to the brief's overall envelope (no-false-PASS for scale).

    Sorts both the three built extents and the three brief L/W/H (orientation may differ) and fails
    when any matched pair is off by more than ``tol_frac`` (default +/-15%). Returns
    ``(mismatch, issue)``. Fails OPEN -- ``(False, None)`` -- when the brief envelope is missing or
    unparseable, so a noisy envelope string never produces a false FAIL.
    """
    target = parse_envelope_mm(brief.overall_envelope_mm)
    if target is None:
        return False, None
    dims = {
        "X": bbox.max.x - bbox.min.x,
        "Y": bbox.max.y - bbox.min.y,
        "Z": bbox.max.z - bbox.min.z,
    }
    built_pairs = sorted(dims.items(), key=lambda kv: kv[1])  # (axis, extent) ascending
    want = sorted(target)
    # Name EVERY offending axis with direction + amount (bench rows 17-18: three parts
    # plateaued on this check because "rebuild at the intended scale" gave the author no
    # actionable target -- which axis, which way, by how much).
    offenses: list[str] = []
    for (axis, got), exp in zip(built_pairs, want, strict=True):
        if exp <= 0:
            return False, None  # fail-open on a degenerate brief dimension
        if abs(got - exp) > tol_frac * exp:
            direction = "SHRINK" if got > exp else "GROW"
            offenses.append(
                f"{axis} is {got:.1f} mm but should be ~{exp:.0f} mm "
                f"({'+' if got > exp else '-'}{abs(got - exp) / exp * 100:.0f}%) -- "
                f"{direction} the feature(s) that set the {axis} extent"
            )
    if offenses:
        built = [v for _, v in built_pairs]
        issue = (
            f"ENVELOPE: built {built[0]:.0f} x {built[1]:.0f} x {built[2]:.0f} mm vs the brief's "
            f"{want[0]:.0f} x {want[1]:.0f} x {want[2]:.0f} mm (each dimension must be within "
            f"+/-{tol_frac * 100:.0f}%). Fix EXACTLY these: " + "; ".join(offenses) + ". "
            "Adjust the named axis dimension(s) in PARAMS -- do NOT rescale the whole part when "
            "only one axis is off."
        )
        return True, issue
    return False, None


# The stated-dim OWNING band (ADR-0012 E1-1): drift beyond max(8%, 1.0mm) per sorted axis
# fails; under it passes. Owns the band below the envelope gate's 15% -- the seeded mutants
# drift 12-15% and previously sailed through (91% false-PASS, audit rows 29/40); known-good
# build noise measured sub-1mm / low-% (row 28), so the floor never fires on it. Tuned
# against the audit corpus; re-tune THERE, not by feel.
_STATED_EXT_REL = 0.08
_STATED_EXT_ABS_MM = 1.0


def _stated_extent_mismatch(
    bbox: BBox, brief: DesignBrief, stated: frozenset[float]
) -> tuple[bool, str | None]:
    """The stated-dimension owning check (ADR-0083 v2.2 follow-up, shipped per ADR-0012).

    Compares the SORTED built extents to the brief's envelope triple with a band far tighter
    than the envelope gate's 20% -- allowed drift per axis = max(8%, 1.0 mm) -- but ONLY for
    envelope numbers that trace to the USER (the ``stated`` protected set, ADR-0080's anchor
    invariant). The first audit false-FAIL proved why: clip_tube_saddle_clamp's 46 mm height
    was the MODEL's own stack estimate (prompt states 60/20/22/15...), and the accepted build
    legitimately measured 40.1 -- a model estimate is not protected currency, the coarse
    envelope gate is its only bound. Row-28's noise lesson is honored by the 1.0 mm floor
    (known-good build noise measured sub-1 mm). Fails OPEN when the envelope is missing,
    unparseable, or nothing in it is user-stated."""
    target = parse_envelope_mm(brief.overall_envelope_mm)
    if target is None or not stated:
        return False, None
    dims = {
        "X": bbox.max.x - bbox.min.x,
        "Y": bbox.max.y - bbox.min.y,
        "Z": bbox.max.z - bbox.min.z,
    }
    built_pairs = sorted(dims.items(), key=lambda kv: kv[1])
    want = sorted(target)
    offenses: list[str] = []
    for (axis, got), exp in zip(built_pairs, want, strict=True):
        if exp <= 0:
            return False, None
        if not any(abs(exp - s) <= 0.01 for s in stated):
            continue  # a model-derived envelope number: the 15% envelope gate owns it
        allowed = max(_STATED_EXT_REL * exp, _STATED_EXT_ABS_MM)
        if abs(got - exp) > allowed:
            offenses.append(
                f"{axis} measures {got:.1f} mm vs the stated {exp:g} mm "
                f"({'+' if got > exp else '-'}{abs(got - exp) / exp * 100:.0f}%, allowed "
                f"{allowed:.1f} mm) -- set the {axis}-extent dimension to the stated value"
            )
    if offenses:
        return True, (
            "STATED DIM: the built part drifts off USER-stated dimensions -- "
            + "; ".join(offenses)
            + ". A stated dimension is the user's number; build it exactly, do not trade it "
            "for other constraints."
        )
    return False, None


# The area-significance gate (ADR-0012 E1-2): the sampled min-wall stays ADVISORY because a
# fillet tip or chamfer edge legitimately measures thin at a POINT -- but a WALL is thin over
# an AREA. The seeded sub-floor mutants measured 65-79% of surface samples thin; a knife edge
# measures well under 1%. Gate on the fraction, deterministic (seeded sampling), fail-open.
# The MEASUREMENT and the THRESHOLD both live in validator.feature_checks (shared with the
# author self-check epilogue, 2026-07-27); only the verdict TEXT is here.


def _thin_wall_gate(
    mesh: Any, floor_mm: float | None, min_area_mm2: float | None = None
) -> tuple[bool, str | None]:
    """Fail when an AREA-significant share of true WALL measures under the wall floor.

    Same inward-ray measurement as the advisory check (offset + noise floor), with two
    discriminators the advisory lacks: the verdict reads the thin FRACTION (not the minimum)
    and counts only near-antiparallel face pairs (real walls). A tapering knife edge or a
    chamfer sliver -- the reason the advisory could never gate (ADR-0029) -- stays advisory,
    while a genuinely sub-floor WALL blocks ``passed``. Deterministic via a local RNG seed.
    A required floor fails closed when the measurement cannot be claimed: unknown
    thickness is not certified thickness."""
    if floor_mm is None or floor_mm <= 0:
        return False, None
    from forge_mind.validator.feature_checks import measure_wall_fraction, wall_gate_triggered

    measured = measure_wall_fraction(mesh, floor_mm)
    if measured is None:
        return True, (
            f"WALL GATE: wall thickness could not be measured against the {floor_mm:.2f} mm "
            "floor, so the part is not certified manufacturable."
        )
    min_area = min_area_mm2 if min_area_mm2 is not None else floor_mm**2
    if (
        wall_gate_triggered(
            measured.thin_fraction,
            measured.thin_area_mm2,
            measured.thin_sample_count,
            min_area,
        )
        and measured.thinnest_mm is not None
    ):
        return True, (
            f"WALL GATE: {measured.thin_fraction * 100:.2f}% of the measured surface "
            f"({measured.thin_area_mm2:.2f} mm^2) is a WALL thinner than the "
            f"{floor_mm:.2f} mm floor (thinnest wall {measured.thinnest_mm:.2f} mm) -- "
            "fraction-significant or larger than one material feature, and "
            "parallel-faced rather than a knife edge. Thicken the thin "
            "cluster (the THIN-WALL location above names where); do not thin the rest."
        )
    return False, None


def _slicer_gate(stl_path: Path, mesh: Any, *, cell_lane: bool) -> tuple[bool, list[str]]:
    """The print-lane SLICER gate (ADR-0012 SS3): 'the validator judges the mesh; the printer prints
    the gcode'. Slice the part with the committed profile and read gcode-level facts the mesh tiers
    cannot see -- slicer warnings, added support volume (target zero), and silently dropped thin
    layers. Returns ``(would_gate, issues)``.

    ``would_gate`` is True when a fact that must block a CELL-lane part is present (no/failed slice,
    a warning, support > 0, or a dropped layer). The CALLER feeds it into ``passed`` ONLY on the cell
    lane (fail-closed, ADR-0065); on the website lane the tier is REPORT-ONLY -- issues surfaced,
    ``passed`` unaffected -- until two audit runs prove the false-FAIL delta <= 1 part (ADR-0012 SS3
    promotion path). Never raises: the wrapper degrades a missing binary to a non-ok outcome, which
    becomes an UNVERIFIABLE issue here (a REFUSAL on the cell lane -- no silent no-run pass)."""
    import tempfile

    from forge_mind.slicing.gcode_facts import parse_gcode_facts
    from forge_mind.slicing.slicer import slice_part

    issues: list[str] = []
    with tempfile.TemporaryDirectory(prefix="forge_slice_") as td:
        outcome = slice_part(stl_path, Path(td))
        if not outcome.ok:
            unavailable = "unavailable" in (outcome.error or "")
            reason = (
                "no slicer binary available"
                if unavailable
                else f"the slice failed ({outcome.error})"
            )
            tail = "; REFUSED on the cell lane" if cell_lane else " (report-only on this lane)"
            issues.append(
                f"SLICER GATE: {reason}, so gcode-level printability is UNVERIFIABLE{tail}."
            )
            return cell_lane, issues  # fail-closed on the cell lane, report-only otherwise
        gcode_text = outcome.gcode_path.read_text(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        facts = parse_gcode_facts(gcode_text, warnings=outcome.warnings, mesh=mesh)

    would_gate = False
    for warning in facts.warnings:
        issues.append(f"SLICER GATE: the slicer warned -- {warning}")
        would_gate = True
    if facts.support_volume_mm3 is not None and facts.support_volume_mm3 > 0.0:
        issues.append(
            f"SLICER GATE: the slicer added {facts.support_volume_mm3:.0f} mm^3 of SUPPORT material "
            "-- ForgeMind parts must be self-supporting (ADR-0012 SS4). Chamfer/fillet the overhang "
            "or reorient the feature so it prints without support."
        )
        would_gate = True
    # The per-layer silhouette diff (dropped-feature fact) is COMPUTED (facts.dropped_layers /
    # min_area_ratio, recorded for the physical-join dataset) but NOT surfaced or gated: the E2
    # measurement flagged 14/34 known-goods (41%), far past the <=5% false-FAIL budget -- shoelace
    # over raw gcode perimeters underestimates the printed silhouette (anchor/seam/hole-winding
    # artifacts). Per ADR-0012 ("a check that cannot meet the budget ships report-tier"), it stays
    # OUT of the verdict until a robust silhouette method replaces the shoelace (TEST_LOG row 47).
    return would_gate, issues


# The in-process OCP B-rep load (import_step, inside brep/spatial checks) is NOT
# thread-safe (the drafts.py lesson); parallel part authoring (FORGE_PARALLEL_AUTHORS)
# validates from worker threads. Serialize ALL authored-part validation here -- a
# single-threaded caller pays one uncontended acquire.
_VALIDATE_LOCK = threading.Lock()


def validate_authored_part(
    stl_path: Path,
    step_path: Path,
    brief: DesignBrief,
    *,
    fabrication_mode: FabricationMode = FabricationMode.PRINT_3D,
    material: Material | None = None,
    author_code: str | None = None,
    prompt: str | None = None,
) -> AgenticValidation:
    with _VALIDATE_LOCK:
        return _validate_authored_part_unlocked(
            stl_path,
            step_path,
            brief,
            fabrication_mode=fabrication_mode,
            material=material,
            author_code=author_code,
            prompt=prompt,
        )


def _validate_authored_part_unlocked(
    stl_path: Path,
    step_path: Path,
    brief: DesignBrief,
    *,
    fabrication_mode: FabricationMode = FabricationMode.PRINT_3D,
    material: Material | None = None,
    author_code: str | None = None,
    prompt: str | None = None,
) -> AgenticValidation:
    """Run the existing geometry tiers + the functional gate on a sandboxed STL/STEP.

    ``material`` (optional) drives the material-aware verdict: mass/cost/spec for the report and
    the one HARD material gate -- a wall thinner than the material can hold blocks ``passed``.
    Defaults to the brief's suggested material (lenient resolve, never crashes on an odd string).
    ``prompt`` (optional) enables the stated-dim OWNING check: user-stated envelope numbers get
    the tight drift band (ADR-0012 E1-1); without it the check stays off (fail-open) and the
    coarse envelope gate alone bounds scale.
    """
    mat = material or resolve_material(brief.material)
    try:
        mesh: Any = trimesh.load(str(stl_path), force="mesh")
        watertight = bool(mesh.is_watertight)
        volume = float(abs(mesh.volume)) if watertight else 0.0
        lo, hi = mesh.bounds
        bbox = BBox(
            min=Vec3(x=float(lo[0]), y=float(lo[1]), z=float(lo[2])),
            max=Vec3(x=float(hi[0]), y=float(hi[1]), z=float(hi[2])),
        )
    except Exception as exc:
        return AgenticValidation(
            passed=False,
            hard_ok=False,
            report=None,
            body_count=0,
            watertight=False,
            volume_mm3=0.0,
            issues=[f"could not load the authored mesh: {type(exc).__name__}: {exc}"],
        )

    part = _load_part(step_path)
    body_count = _count_solids(part)
    # A REAL STEP that would not load is a downgrade we must BLOCK: with part=None, body_count
    # defaults to 1 and the B-rep checks go neutral, so a multi-body / invalid solid that only
    # exported an STL could otherwise PASS. Fail-open when there is simply no STEP file.
    step_downgrade = _step_load_failed(step_path, part)
    # B-rep checks on the OCP solid (more authoritative than the tessellated mesh): exact volume,
    # naked / non-manifold edge detection, and OCC validity. Neutral when the STEP did not load.
    brep = brep_report(part)
    eff_volume = brep.volume_mm3 if brep.volume_mm3 is not None else volume
    hard_ok = body_count == 1 and watertight and volume > 0
    # Degeneracy guard (no-false-PASS): a watertight solid can still be a near-zero-volume sliver
    # (the wine-glass two-rail profile collapsed to 18 mm^3). Uses the exact B-rep volume when
    # available; see ``_is_degenerate`` for the conservative band that never flags a legit thin part.
    bbox_vol = (bbox.max.x - bbox.min.x) * (bbox.max.y - bbox.min.y) * (bbox.max.z - bbox.min.z)
    degenerate = _is_degenerate(watertight, eff_volume, bbox_vol)

    # The feature tier validates against the brief's min wall in PRINT_3D; a GENERAL (not-3D-printed)
    # part is held only to the relaxed structural-sanity floor (ADR-0031), so the FDM-strict wall does
    # not over-flag fillet/gusset knife-edges. None -> use the brief wall as-is (PRINT_3D, unchanged).
    eff_min_wall = fabrication_mode.effective_min_wall(brief.min_wall_mm, mat.min_wall_mm)
    program = PartProgram(
        metadata=Metadata(
            name=brief.object_name[:128], units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)
        ),
        operations=[Box(op_id="massing", mode=OpMode.ADD, width=1.0, depth=1.0, height=1.0)],
        constraints=Constraints(min_wall_thickness=eff_min_wall),
    )
    model = CADModel(
        part=part,
        stl_path=stl_path,
        step_path=step_path,
        topology_index=TopologyIndex(),
        bounding_box=bbox,
    )

    try:
        # The mode gates the printability tier (overhang runs only in PRINT_3D -- ADR-0031).
        report = validate(
            model, program, material=brief.material, fabrication_mode=fabrication_mode
        )
    except Exception as exc:
        return AgenticValidation(
            passed=False,
            hard_ok=hard_ok,
            report=None,
            body_count=body_count,
            watertight=watertight,
            volume_mm3=volume,
            issues=[f"validator crashed: {type(exc).__name__}: {exc}"],
        )

    issues: list[str] = []
    if body_count != 1:
        issues.extend(
            disconnection_feedback(part)
            or [f"the part is {body_count} disconnected bodies -- it must be ONE fused solid."]
        )
    if not watertight:
        issues.append(
            "the mesh is not watertight (holes / non-manifold edges) -- the solid must be closed."
        )
    if volume <= 0:
        issues.append("the part has zero or negative volume.")
    if brep.open_or_nonmanifold:
        where = ""
        if brep.defect_bbox is not None:
            d = brep.defect_bbox
            where = (
                f" The gap edges cluster at x[{d[0]:.0f},{d[3]:.0f}] "
                f"y[{d[1]:.0f},{d[4]:.0f}] z[{d[2]:.0f},{d[5]:.0f}]."
            )
        issues.append(
            f"the B-rep is not a sealed solid: {brep.open_edges} naked edge(s) (open boundary) and "
            f"{brep.nonmanifold_edges} non-manifold edge(s).{where} A boolean or an unclosed profile "
            "left a gap or a self-touching face -- fuse overlapping bodies and close every sketch "
            "profile so the part is ONE watertight solid."
        )
    if not brep.valid:
        issues.append(
            "the B-rep is geometrically INVALID (OCC BRepCheck) -- usually a self-intersection or a "
            "bad face from an over-large fillet/chamfer or a degenerate boolean. Back off the radius "
            "or rebuild the offending feature so the solid is valid."
        )
    fails = report.failed_check_ids()
    for cid in fails:
        check = report.check(cid)
        issues.append(f"{cid}: {check.message}" if check is not None else cid)
    if any("min_wall" in cid for cid in fails):
        issues.extend(
            thin_region_feedback(
                mesh, eff_min_wall if eff_min_wall is not None else brief.min_wall_mm
            )
        )
        # The wall-thickness CLUSTER map (lever-c pattern extended): the part's measured
        # thickness structure, so the author thickens the right cluster. Advisory.
        issues.extend(
            wall_cluster_feedback(
                mesh, eff_min_wall if eff_min_wall is not None else brief.min_wall_mm
            )
        )

    # Feature-completeness (the bracket gap): a part short of its planned through-holes is NOT done.
    # Soft no-false-PASS gate -- blocks `passed` so the loop keeps drilling until the holes appear.
    features_missing = False
    bore_fit_issues: list[str] = []
    through_holes: int | None = None
    if hard_ok:
        through_holes = mesh_genus(mesh)
        fc_lines, genus_missing = feature_count_feedback(mesh, brief)
        issues.extend(fc_lines)
        # Measured-hole audit (ADR-0035 B1, size-aware): the B-rep measurement reconciles the
        # brief's hole PLAN (target diameters from the standards registry) against every
        # measured concave bore -- "2 of 4 Ø4.5 drilled; the Ø2.0 bores match no planned
        # size". Runs even when the genus COUNT is satisfied: right-count/wrong-size holes
        # are named instead of waved through. Advisory -- context only, never the verdict.
        issues.extend(measured_hole_feedback(part, brief))
        bore_fit_issues = measured_hole_conformance_issues(part, brief, mat)
        issues.extend(bore_fit_issues)
        # Measured-boss audit (the same pattern, CONVEX side): planned bosses / pins /
        # standoffs with stated diameters vs the measured convex cylinder stacks. Advisory.
        issues.extend(measured_boss_feedback(part, brief))
        # Balance advisory (functional-mechanism spine): for a rotating-power part, MEASURE
        # balance about its shaft bore and tell the author to counterweight it if unbalanced.
        # Gated to that class + fail-closed + advisory (never gates `passed`) -- see spin_check.
        issues.extend(balance_feedback(brief, part, mesh))
        # Radial feature-count advisory (ADR-0079): when the brief NAMES a tooth count, the
        # polar spectrum is measured and a mismatch is fed back -- composition is STRUCTURE,
        # so a dropped tooth band is the AUTHOR's fix (the optimizer only tunes floats).
        # Advisory, never gates `passed`.
        issues.extend(radial_count_feedback(mesh, brief))
        # The completeness GATE counts MEASURED concave bores, not genus alone: a handle/ring inflates
        # the mesh genus so a dropped required hole can pass feature_count_feedback. Bore count can't
        # be fooled (audit H8, ADR-0065) -- and it reconciles per spec by diameter, because a
        # total-count compare let off-plan vent bores mask never-cut planned holes (TEST_LOG row 38).
        # Either signal short -> the part isn't done.
        features_missing = genus_missing or measured_holes_short(part, brief)
    if degenerate:
        issues.append(
            f"the part is nearly empty: {volume:.0f} mm^3 of material in a {bbox_vol:.0f} mm^3 "
            f"envelope (~{100 * volume / bbox_vol:.3f}% fill) -- the solid collapsed or "
            "self-intersected. Build the OUTER solid form and SHELL it (offset openings=...), "
            "do not draw a two-rail inner+outer profile."
        )

    # Envelope conformance (no-false-PASS for scale): a part built at the wrong scale still passes
    # every geometry tier, so compare the built bbox to the brief's overall envelope. Sorted extents
    # (orientation may differ); fail when any dimension is off by >20%. Fail-open when the brief's
    # envelope is missing / unparseable, so a noisy envelope string never produces a false FAIL.
    envelope_mismatch, envelope_issue = _envelope_mismatch(bbox, brief)
    if envelope_issue is not None:
        issues.append(envelope_issue)
    # Stated-dimension OWNING check (ADR-0012 E1-1): the band UNDER the envelope gate's 20%,
    # where the audited 12-15% drifts lived (91% false-PASS). Skipped when the coarse envelope
    # gate already fired -- one defect, one actionable message.
    stated_dim_mismatch = False
    mapped_pos_mismatch = False
    if prompt:
        # ADR-0088 slice 1: one parse of (prompt, brief) into the typed DimTable feeds both E1
        # checks (flag FORGE_DIM_TABLE). Off = each check parses as before (byte-identical); the
        # table's queries reproduce those parsers exactly, so on/off parity is provable on the audit.
        table = None
        positions = None
        try:
            from forge_mind.agentic.dim_table import build_dim_table, dim_table_enabled

            if dim_table_enabled():
                table = build_dim_table(prompt, brief)
        except Exception:
            table = None
        if table is not None:
            stated = table.stated_values()
            positions = table.positions()
        else:
            try:
                from forge_mind.agentic.stated_dims import stated_numbers

                stated = stated_numbers(prompt, brief)
            except Exception:
                stated = frozenset()
        if not envelope_mismatch:
            stated_dim_mismatch, stated_dim_issue = _stated_extent_mismatch(bbox, brief, stated)
            if stated_dim_issue is not None:
                issues.append(stated_dim_issue)
        # Brief->feature POSITION owning check (ADR-0012 E1): a user-stated bore pattern/height
        # that drifted. Fail-open on every unmeasurable path (see mapped_dims). Runs only with a
        # measurable solid (hard_ok) so a broken build's positions are not judged.
        if hard_ok:
            try:
                from forge_mind.agentic.mapped_dims import mapped_position_mismatch

                mapped_pos_mismatch, mapped_pos_issue = mapped_position_mismatch(
                    part, mesh, brief, prompt, stated, positions=positions
                )
            except Exception:
                mapped_pos_mismatch, mapped_pos_issue = False, None
            if mapped_pos_issue is not None:
                issues.append(mapped_pos_issue)
    # Area-significant thin-wall GATE (ADR-0012 E1-2): the advisory min-wall stays advisory
    # (knife edges), but a WALL thin over >=3% of the surface blocks `passed` -- the audited
    # sub-floor mutants escaped precisely because nothing gated here.
    thin_wall_gated = False
    if hard_ok:
        thin_wall_gated, thin_wall_issue = _thin_wall_gate(
            mesh, eff_min_wall, mat.min_feature_mm**2
        )
        if thin_wall_issue is not None:
            issues.append(thin_wall_issue)
            if not any("min_wall" in cid for cid in fails):
                # the advisory point-min missed what the area gate caught: give the author
                # the same localization context it would have had on an advisory FAIL
                issues.extend(
                    thin_region_feedback(
                        mesh, eff_min_wall if eff_min_wall is not None else brief.min_wall_mm
                    )
                )

    # Print-lane SLICER gate (ADR-0012 SS3), flag-gated OFF by default so the hot loop never slices.
    # On the CELL lane it fails closed (a would-gate fact blocks `passed`); on the website lane it is
    # REPORT-ONLY (issues surfaced, `passed` unaffected) until two audit runs prove it low-noise.
    # Only a coherent PRINT_3D solid is worth slicing.
    slicer_gated = False
    if slicer_gate_enabled() and hard_ok and fabrication_mode.enforces_printability:
        would_gate, slicer_issues = _slicer_gate(stl_path, mesh, cell_lane=cell_lane_enabled())
        issues.extend(slicer_issues)
        if cell_lane_enabled():
            slicer_gated = would_gate

    # STEP-downgrade block (no-false-PASS): a real STEP that would not load means the kernel checks
    # ran neutral, so an honest issue is emitted and the part is excluded from PASS.
    if step_downgrade:
        issues.append(
            "the STEP B-rep failed to load, so the kernel checks (solid count, naked-edge, OCC "
            "validity) could not run and cannot certify the part -- it may be a multi-body or "
            "invalid solid that only exported an STL. Re-export a single valid STEP solid."
        )

    # Unmeasurable-wall block (no-false-PASS): an unmeasurable min wall is UNCERTIFIED, not fine.
    wall_unmeasurable = _wall_unmeasurable(report)
    if wall_unmeasurable:
        issues.append(
            "the minimum wall thickness is UNMEASURABLE (no inward ray hit a wall), so the part is "
            "not certified thick enough -- it may be hollow, shell-less, or non-manifold inside. "
            "Make the part a solid (or properly shelled) body so its wall thickness can be measured."
        )

    # Material-aware verdict: mass/cost/spec for the report. The sub-min-wall material flag is now
    # ADVISORY (it reuses the SAME low-confidence sampled wall as features.min_wall_thickness, so it
    # must not gate `passed` either -- ADR-0029); its issue is still surfaced. Exact volume -> mass/cost.
    mat_report = evaluate_material(
        mat, volume_mm3=eff_volume, measured_min_wall_mm=_measured_min_wall(report)
    )
    issues.extend(mat_report.issues)

    # Standards-conformance gate (ADR-0009 / no-false-PASS): if the brief declares a STANDARD part
    # (M3, 608, ...), the author must have fetched its exact ISO/DIN dimension from the registry, not
    # hand-modeled it. CITE every standard it fetched and REFUSE any it declared but bypassed -- a
    # guessed standard dimension is the silent-wrong-part a senior engineer catches with one ruler.
    standards_cites: list[str] = []
    standards_fails: list[str] = []
    if author_code is not None:
        declared_standards = [f.standard for f in brief.features if f.standard]
        if declared_standards:
            standards_cites, standards_fails = audit_standards(author_code, declared_standards)
            issues.extend(standards_fails)

    # Stated-dimension conformance history: the row-28 bolt-on advisory (a direct reuse of the
    # ADR-0080 acceptance machinery) fired on 21% of known-goods and was backed out. The owning
    # check that replaced it (ADR-0012 E1-1, `_stated_extent_mismatch` above) is scoped to what
    # measures cleanly -- sorted extents vs the stated envelope triple, band max(8%, 1.0mm) --
    # while bore diameters stay owned by the measured-hole audit's ID windows. Tuned on the
    # seeded-defect audit corpus (scorecard), never by feel.

    # Printability OVERHANG + the SAMPLED min-wall estimate are ADVISORY on the agentic path (arbitrary
    # user parts): an overhang prints with supports, and the min-of-N-rays wall over-flags fillet/gusset
    # knife-edges (a self-admitted low-confidence estimate). They are surfaced (issues + the report) but
    # do NOT gate `passed` here -- the high-confidence gates below still do; the DSL/template path keeps
    # them STRICT (ADR-0029). Keep ADVISORY_CHECK_IDS in sync with api/core.py::_validation_view.
    gating_fails = [f for f in fails if f not in ADVISORY_CHECK_IDS]

    # Feature-presence gate v1 (ADR-0117): deterministic checks that the authored geometry
    # plausibly contains the brief's declared additive features (the measured omission class:
    # 27/40 breadth + 40/64 e2e non-shipped runs). Calibrated offline 2026-07-19 against every
    # archived shipped run (zero false-FAILs); anything unparseable is UNKNOWN and contributes
    # nothing (fail-open). Gating findings block `passed`; advisory findings only inform.
    presence_gating: list[str] = []
    from forge_mind.agentic.feature_presence import (
        feature_presence_enabled,
        feature_presence_report,
    )

    if feature_presence_enabled():
        presence = feature_presence_report(brief, volume if volume > 0 else None, author_code)
        for finding in presence.findings:
            issues.append(finding.message)
            if finding.gates:
                presence_gating.append(finding.check)

    # Interface audits (ADR-0120, FORGE_INTERFACE_AUDIT=1): stated-OD + bore-ligament.
    # Calibrated 2026-07-21: 23 archived accepted parts replay clean (one true latent
    # defect found + sidecar-quarantined); the L68 bolt-hole-into-port class gates.
    interface_gating: list[str] = []
    from forge_mind.agentic.interface_audit import interface_audit_enabled

    if interface_audit_enabled() and hard_ok and part is not None:
        from forge_mind.agentic.interface_audit import interface_findings

        for ifinding in interface_findings(brief, part):
            issues.append(ifinding.message)
            if ifinding.gates:
                interface_gating.append(ifinding.check)

    # Servo-fit audit (ADR-0124, FORGE_SERVO_FIT_AUDIT=1): a brief-DECLARED MG996R
    # interface must expose the cited hole pattern in the measured geometry (horn BC
    # 14.0 founder-verified). Declared-but-unmeasurable GATES (fail-closed).
    servo_fit_gating: list[str] = []
    from forge_mind.agentic.servo_fit_audit import servo_fit_audit_enabled

    if servo_fit_audit_enabled() and hard_ok and part is not None:
        from forge_mind.agentic.servo_fit_audit import servo_fit_findings

        for sfinding in servo_fit_findings(brief, part):
            issues.append(sfinding.message)
            if sfinding.gates:
                servo_fit_gating.append(sfinding.check)

    # Tendon-routability audit v1 (ADR-0124 R4 gate, FORGE_TENDON_AUDIT=1): a declared
    # tendon channel must measure as a routable through-channel (same fail-closed law).
    tendon_gating: list[str] = []
    from forge_mind.agentic.tendon_audit import tendon_audit_enabled

    if tendon_audit_enabled() and hard_ok and part is not None:
        from forge_mind.agentic.tendon_audit import tendon_findings

        _bb = part.bounding_box()
        _longest = max(float(_bb.size.X), float(_bb.size.Y), float(_bb.size.Z))
        for tfinding in tendon_findings(brief, part, _longest):
            issues.append(tfinding.message)
            if tfinding.gates:
                tendon_gating.append(tfinding.check)

    passed = (
        hard_ok
        and not gating_fails
        and not features_missing
        and not bore_fit_issues
        and not degenerate
        and not brep.open_or_nonmanifold
        and brep.valid
        and not envelope_mismatch
        and not stated_dim_mismatch
        and not mapped_pos_mismatch
        and not thin_wall_gated
        and not slicer_gated
        and not step_downgrade
        and not wall_unmeasurable
        and not standards_fails
        and not presence_gating
        and not servo_fit_gating
        and not tendon_gating
        and not interface_gating
    )
    return AgenticValidation(
        passed=passed,
        hard_ok=hard_ok,
        report=report,
        body_count=body_count,
        watertight=watertight,
        volume_mm3=volume,
        issues=issues,
        material_report=mat_report,
        standards_cites=standards_cites,
        features_missing=features_missing,
        through_holes=through_holes,
    )


__all__ = ["AgenticValidation", "validate_authored_part"]
