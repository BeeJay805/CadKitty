"""Adapt the agentic engine to the /generate contract.

Runs ``produce_part`` (the same path as the CLI), then projects its ``PartResult`` into the
``GenerateResponse`` shape the frontend expects, and guarantees a servable ``part.stl`` + ``part.glb``
in the job directory regardless of which route (cosmetic single-body vs functional assembly) ran.

HONEST NOTES:
  - The agentic engine does NOT produce a hardware Bill of Materials. We synthesize a Kit from the
    design brief: part name / min-wall / material / orientation are REAL (from the brief); any
    hardware lines are marked ``provisional`` with a brief-derived source, never a fake registry
    citation. The BOM is empty when the brief declared no standard parts.
  - Mass / volume / centre-of-mass ARE computed by the validator (report.metrics) but are not part of
    the frontend contract yet, so they are not surfaced here. Adding them is a contract change.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from forge_mind.agentic.confidence import ConfidenceLabel, derive_confidence, standard_lane_label
from forge_mind.agentic.provenance import DimensionProvenance
from forge_mind.agentic.result import PartResult, produce_part
from forge_mind.agentic.validate import ADVISORY_CHECK_IDS
from forge_mind.api.schema import (
    AssemblyJointView,
    AssemblyPartView,
    AssemblyView,
    BillOfMaterials,
    BomLine,
    CertificateView,
    CheckView,
    ConceptView,
    ConfidenceView,
    DimensionTierCounts,
    DimensionView,
    FeasibilityView,
    Files,
    GenerateError,
    GenerateOk,
    GenerateRefused,
    InventionPackageView,
    Kit,
    MakeInstructions,
    PhysicalValidationView,
    StepProofView,
    TierStatus,
    ValidationView,
    Vec3,
)
from forge_mind.fabrication import FabricationMode
from forge_mind.implicit.router import ImplicitRequest, build_implicit, implicit_match
from forge_mind.invention import (
    GeneratedPart,
    InventionPackage,
    build_invention_package,
    derive_maturity,
    derive_package_evidence,
    invention_enabled,
)
from forge_mind.invention.support import build_support_kb
from forge_mind.kit import Kit as DomainKit
from forge_mind.layout.standard_match import standard_match
from forge_mind.llm.provider import LLMClientLike
from forge_mind.validator.report import CheckStatus
from forge_mind.viz.glb import to_gltf_yup

GLB_NAME = "part.glb"
STL_NAME = "part.stl"

_LOG = logging.getLogger(__name__)

# quality -> loop effort. "standard" is the cheap single-draft loop; "studio" tries more drafts and
# allows more iterations + budget for a better result.
_QUALITY = {
    # budget_usd is the HARD per-generation ceiling -- the loop stops the moment cumulative spend
    # reaches it (the real "no infinite money" guard). max_iterations is a high SAFETY backstop so
    # the BUDGET, not a low iteration count, is what bounds a complex part. For a multi-part object
    # the budget is SHARED across sub-parts (produce_functional_part), so part-count can't multiply
    # spend. Tune here to trade thoroughness vs cost/latency.
    "standard": {"n_drafts": 1, "max_iterations": 20, "budget_usd": 3.0},
    "studio": {"n_drafts": 3, "max_iterations": 32, "budget_usd": 8.0},
}

_DEFAULT_MATERIAL = "PLA"

_CAVEAT = (
    "Advisory guidance. Step-by-step fabrication instructions and a registry-verified "
    "bill of materials are not generated for agentic parts yet."
)


@dataclass(frozen=True)
class StandardLaneOutcome:
    """One standard-lane attempt, classified (ADR-0035 decision a).

    The lane used to swallow every failure (``except Exception: return None``), which is how a
    100%-dead part family (the Pi-4 tray) went unnoticed by telemetry. Every attempt now produces
    one of these: attached to ``GenerateResult.lane``, persisted additively into the
    ``generations.spec`` JSON column by the server, and logged. ``outcome`` values:

      shipped            -- the deterministic interpreter built it; the lane's result shipped
      refused            -- the pipeline refused cleanly (unsupported / infeasible / infidelity)
      failed_stage       -- a pipeline stage failed (intent / blueprint / layoutspec / compile)
      validation_failed  -- the part built but did not PASS the strict gates
      fallback_blocked   -- the pipeline fell back to direct-DSL; the lane refuses to ship a
                            stochastic one-shot under the "standard:" label (the honesty policy)
      missing_artifacts  -- the STL/GLB could not be produced for the viewer
      exception          -- an unexpected error (captured + classified, no longer swallowed)
    """

    attempted: bool
    outcome: str
    gen_path: str | None = None
    fallback_reason: str | None = None
    refusal: str | None = None
    error: str | None = None
    duration_s: float | None = None
    cost_usd: float | None = None

    def to_record(self) -> dict[str, Any]:
        """The ``generations.spec`` payload: unset fields dropped, floats rounded."""
        row: dict[str, Any] = {
            "attempted": self.attempted,
            "outcome": self.outcome,
            "gen_path": self.gen_path,
            "fallback_reason": self.fallback_reason,
            "refusal": self.refusal,
            "error": self.error,
            "duration_s": round(self.duration_s, 2) if self.duration_s is not None else None,
            "cost_usd": round(self.cost_usd, 4) if self.cost_usd is not None else None,
        }
        return {k: v for k, v in row.items() if v is not None}


@dataclass(frozen=True)
class GenerateResult:
    """One request's outcome. The file URLs in ``ok.files`` are bare names; the HTTP layer rewrites
    them to ``files/<job>/<name>`` once the job id is known. ``lane`` carries the standard-lane
    attempt telemetry (ADR-0035) whenever the prompt matched the lane -- regardless of which
    engine ultimately produced the result."""

    status: str  # "ok" | "refused" | "error"
    ok: GenerateOk | None = None
    refused: GenerateRefused | None = None
    error: GenerateError | None = None
    lane: StandardLaneOutcome | None = None


def _validation_view(result: PartResult) -> ValidationView:
    """Project final deliverability, not merely per-part geometry, into ValidationView.

    The frontend keys its headline status on ``validation.passed``. A measured
    ``not_assemblable`` or ``motion_blocked`` result must therefore be red here even
    when every individual part's geometry passed.
    """
    shipping = result.disposition in ("shipped", "shipped_with_caveats")
    val = result.loop.validation
    if val is None or val.report is None:
        passed = result.loop.geometry_passed and shipping
        failed = []
        if not shipping:
            failed.append(
                CheckView(
                    id="delivery.disposition",
                    status="fail",
                    message=f"Final disposition is {result.disposition}; this result is not deliverable.",
                )
            )
        return ValidationView(
            passed=passed,
            overall="pass" if passed else "fail",
            tiers=[],
            failed_checks=failed,
        )
    report = val.report
    tiers = [
        TierStatus(tier=tier.value, status=tr.status.value) for tier, tr in report.tiers.items()
    ]
    # Advisory checks (overhang / sampled min-wall) show as "warn", consistent with `passed` -- they
    # do not gate the agentic verdict (ADR-0029), so they must not read as a hard "fail" in the view.
    # INFO checks are measured evidence (ADR-0035 B1): recorded in the report/telemetry, kept out of
    # the user-facing failed list until promoted.
    failed = [
        CheckView(
            id=c.id,
            status="warn" if c.id in ADVISORY_CHECK_IDS else c.status.value,
            message=c.message,
        )
        for c in report.checks
        if c.status not in (CheckStatus.PASS, CheckStatus.INFO)
    ]
    if not shipping:
        failed.append(
            CheckView(
                id="delivery.disposition",
                status="fail",
                message=f"Final disposition is {result.disposition}; this result is not deliverable.",
            )
        )
    passed = val.passed and shipping
    return ValidationView(
        passed=passed,
        overall=("warn" if failed else "pass") if passed else "fail",
        tiers=tiers,
        failed_checks=failed,
        step_proof=_step_view(report.step_proof),
    )


def _step_view(step: Any) -> StepProofView | None:
    """Project the validator's StepProof into the API view (shared by both report shapes).
    Carries the named-B-rep fidelity fields (ADR-0037 rung 2) additively."""
    if step is None:
        return None
    return StepProofView(
        matches=step.matches,
        volume_mm3=step.volume_mm3,
        area_mm2=step.area_mm2,
        centroid=Vec3(x=step.centroid.x, y=step.centroid.y, z=step.centroid.z),
        volume_rel_err=step.volume_rel_err,
        area_rel_err=step.area_rel_err,
        centroid_dist_mm=step.centroid_dist_mm,
        schema_id=step.schema_id,
        is_brep=step.is_brep,
        tessellated=step.tessellated,
        product_name=step.product_name,
        named=step.named,
        solid_count=step.solid_count,
        face_count=step.face_count,
        topology_matches=step.topology_matches,
    )


def _confidence_view(
    label: ConfidenceLabel, records: list[DimensionProvenance] | None = None
) -> ConfidenceView:
    """Project a derived ConfidenceLabel (+ optional per-dimension records) into the API view."""
    return ConfidenceView(
        overall=label.overall,
        basis=label.basis,
        manufacturability=label.manufacturability,
        process_note=label.process_note,
        dimension_tiers=DimensionTierCounts(**label.dimension_tiers.model_dump()),
        dimensions=[
            DimensionView(
                feature=r.feature, tier=r.tier.value, citation=r.citation, dims=r.dims_text
            )
            for r in (records or [])
        ],
        verify_dimensions=label.verify_dimensions,
        not_fit_for=label.not_fit_for,
        part_class=label.part_class,
        summary=label.summary,
    )


def agentic_confidence_label(result: PartResult, fabrication_mode: str) -> ConfidenceLabel:
    """Derive the agentic result's confidence LABEL from its evidence (ADR-0041): the earned
    dimension tiers, the geometry gates, and the judge's structured verdict. Deterministic. The
    /generate view AND the invention adapter both build on this ONE derivation (no duplication)."""
    val = result.loop.validation
    if val is None:
        gates: Any = "fail"
    elif result.loop.geometry_passed:
        gates = "pass"
    elif val.hard_ok:
        gates = "warn"
    else:
        gates = "fail"
    return derive_confidence(
        records=result.provenance,
        gates=gates,
        judge_accepted=result.verdict.is_the_object if result.verdict is not None else None,
        judge_confidence=result.verdict.confidence if result.verdict is not None else None,
        object_name=result.loop.brief.object_name if result.loop.brief else None,
        summary_text=result.loop.brief.summary if result.loop.brief else None,
        fabrication_mode=fabrication_mode,
    )


def _agentic_confidence(result: PartResult, fabrication_mode: str) -> ConfidenceView:
    """Project the agentic confidence label into the API view."""
    return _confidence_view(agentic_confidence_label(result, fabrication_mode), result.provenance)


def _infer_kind(standard: str) -> str:
    """Best-effort hardware kind from a brief standard designation (M3, 608, 6x3, ...). The line is
    flagged provisional regardless, so this only sets a sensible default category."""
    s = standard.strip().lower()
    if s.startswith("m") and s[1:].replace(".", "").isdigit():
        return "fastener"
    if "x" in s:
        return "magnet"
    if s.isdigit():
        return "bearing" if len(s) >= 3 else "dowel"
    return "fastener"


def _synthesize_kit(result: PartResult, material: str) -> Kit:
    """Build an honest Kit from the design brief. Geometry-grounded fields (name, min-wall, material,
    orientation) are real; hardware lines are provisional and brief-derived (never a fake registry)."""
    brief = result.loop.brief
    part_name = brief.object_name if brief and brief.object_name else "Generated part"

    lines: list[BomLine] = []
    if brief is not None:
        for feat in brief.features:
            std = feat.standard
            if not std:
                continue
            lines.append(
                BomLine(
                    component=f"{std} ({feat.name})",
                    kind=_infer_kind(std),
                    key=std,
                    quantity=feat.count,
                    source="Derived from the design brief (not registry-verified).",
                    provisional=True,
                    length_mm=None,
                    length_note="Confirm size and length against your hardware before ordering.",
                )
            )

    instructions = MakeInstructions(
        print_orientation=Vec3(x=0.0, y=0.0, z=1.0),
        orientation_note=(
            brief.print_orientation_note
            if brief and brief.print_orientation_note
            else "Orient the largest flat face down."
        ),
        material=material,
        recommended_min_wall_mm=(brief.min_wall_mm if brief else None),
        layer_height_mm=0.2,
        wall_note="Use enough perimeters to realize the minimum wall thickness.",
        infill_note="Advisory: 20-40% infill suits most functional parts.",
        insert_install_note=None,
        steps=[
            f"Fabricate the part in {material}.",
            "Clean up supports and test-fit any hardware before final assembly.",
        ],
        advisory=True,
        caveat=_CAVEAT,
    )
    return Kit(
        part_name=part_name,
        bom=BillOfMaterials(lines=lines, unmapped=[]),
        instructions=instructions,
    )


def _compose_generated_part(
    result: PartResult, material: str, fabrication_mode: FabricationMode
) -> GeneratedPart:
    """Compose an already-produced PartResult into a GeneratedPart, REUSING the /generate kit +
    confidence derivation (``_synthesize_kit`` + ``agentic_confidence_label``) -- never duplicating
    per-lane generation behavior. Pure (no model call). The API Kit is projected to the domain Kit
    the package embeds (identical field shapes -> total round-trip)."""
    brief = result.loop.brief
    return GeneratedPart(
        part_name=brief.object_name if brief and brief.object_name else "Generated part",
        disposition=result.disposition,
        kit=DomainKit.model_validate(_synthesize_kit(result, material).model_dump()),
        confidence=agentic_confidence_label(result, fabrication_mode.value),
        artifacts=dict(result.artifacts),
    )


def produce_for_invention(
    prompt: str,
    out_dir: Path,
    *,
    client: LLMClientLike | None = None,
    material: str = _DEFAULT_MATERIAL,
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
    produce_fn: Callable[..., PartResult] = produce_part,
    extra_context: str | None = None,
) -> GeneratedPart:
    """The invention lifecycle's engine adapter (ADR-0048): run the proven agentic engine, then
    compose its ``PartResult`` into a :class:`GeneratedPart`. ``produce_fn`` is injectable so a test
    passes a canned ``PartResult`` (no real model call). Only reached on the ``FORGE_INVENTION`` /
    problem-mode paths; ``/generate`` is untouched. ``extra_context`` (ADR-0087 S6b) threads the
    compiled stated-facts block into the brief research — the ADR-0036 seam, unchanged."""
    result = produce_fn(
        prompt,
        out_dir,
        client=client,
        material=material,
        fabrication_mode=fabrication_mode,
        extra_context=extra_context,
    )
    return _compose_generated_part(result, material, fabrication_mode)


def _invention_package_view(pkg: InventionPackage) -> InventionPackageView:
    """Project a domain InventionPackage into the additive API view (ADR-0048): ONLY typed/closed
    data -- the maturity + evidence labels, the advisory feasibility, the NOT-PHYSICALLY-VALIDATED
    banner, the honest BOM headline, and the chosen concept. No generated prose can enter it."""
    basis = pkg.confidence.basis if pkg.confidence is not None else None
    concept_view: ConceptView | None = None
    if pkg.concept is not None:
        concept = pkg.concept
        concept_view = ConceptView(
            concept_id=concept.concept_id,
            name=concept.name,
            primary_mechanism=concept.primary_mechanism.value,
            secondary_mechanisms=[m.value for m in concept.secondary_mechanisms],
            approach=concept.approach,
            obvious_baseline=concept.obvious_baseline,
            difference_from_obvious=concept.difference_from_obvious,
            key_risks=list(concept.key_risks),
        )
    return InventionPackageView(
        part_name=pkg.part_name,
        problem_statement=pkg.problem_statement,
        disposition=pkg.disposition,
        maturity=derive_maturity(basis, pkg.physical_validation.status).value,
        evidence_tier=derive_package_evidence(basis).value,
        bom_summary=pkg.bom_summary,
        feasibility=FeasibilityView(
            verdict=pkg.feasibility.verdict.value,
            note=pkg.feasibility.note,
            disclaimer=pkg.feasibility.disclaimer,
        ),
        physical_validation=PhysicalValidationView(
            status=pkg.physical_validation.status.value,
            banner=pkg.physical_validation.banner,
        ),
        concept=concept_view,
        # The grounded support KB (ADR-0050), composed deterministically from this same package so the
        # frontend can drive POST /invention/support without a second build. No model call.
        support_kb=build_support_kb(pkg).model_dump(mode="json"),
    )


def _invention_view_for_result(
    result: PartResult, prompt: str, material: str, fabrication_mode: FabricationMode
) -> InventionPackageView | None:
    """The additive /generate envelope (ADR-0048): when ``FORGE_INVENTION`` is on, compose the
    deterministic InventionPackage over the produced part (concept None -- the front door is the
    lifecycle entry, not /generate) and project it. None when the flag is off -> the response is
    byte-identical to today's contract."""
    if not invention_enabled():
        return None
    gp = _compose_generated_part(result, material, fabrication_mode)
    pkg = build_invention_package(
        part_name=gp.part_name,
        disposition=gp.disposition,
        kit=gp.kit,
        confidence=gp.confidence,
        artifacts=gp.artifacts,
        problem_statement=prompt,
    )
    return _invention_package_view(pkg)


def _stl_to_glb(stl: Path, glb: Path) -> bool:
    try:
        import trimesh

        mesh: Any = trimesh.load(str(stl), force="mesh")
        to_gltf_yup(mesh)  # Z-up kernel mesh -> glTF Y-up (viewer only)
        mesh.export(str(glb))
        return True
    except Exception:
        return False


def _finalize_files(result: PartResult, job_dir: Path) -> bool:
    """Guarantee ``job_dir/part.stl`` and ``job_dir/part.glb`` exist. The cosmetic route already
    writes both; the functional route writes an assembled STL elsewhere and no GLB, so copy the STL
    in and build the GLB from it. Returns True only when both are present (the viewer needs the GLB)."""
    stl_dst = job_dir / STL_NAME
    glb_dst = job_dir / GLB_NAME

    if not stl_dst.exists():
        stl_src = result.artifacts.get("stl")
        if stl_src and Path(stl_src).exists():
            stl_dst.write_bytes(Path(stl_src).read_bytes())
    if not stl_dst.exists():
        return False

    if not glb_dst.exists():
        glb_src = result.artifacts.get("glb")
        if glb_src and Path(glb_src).exists():
            glb_dst.write_bytes(Path(glb_src).read_bytes())
        else:
            _stl_to_glb(stl_dst, glb_dst)
    return glb_dst.exists()


def _implicit_validation(watertight: bool) -> ValidationView:
    """Honest geometry verdict for an implicit mesh: watertight + positive volume is the bar."""
    if watertight:
        return ValidationView(
            passed=True,
            overall="pass",
            tiers=[TierStatus(tier="geometry", status="pass")],
            failed_checks=[],
        )
    return ValidationView(
        passed=False,
        overall="warn",
        tiers=[TierStatus(tier="geometry", status="warn")],
        failed_checks=[
            CheckView(
                id="watertight",
                status="warn",
                message="The lattice mesh is not fully watertight at this resolution; printability may suffer.",
            )
        ],
    )


def _implicit_kit(part: Any) -> Kit:
    """A minimal honest Kit for a lattice: no hardware, no-supports print note, STL-only caveat."""
    instructions = MakeInstructions(
        print_orientation=Vec3(x=0.0, y=0.0, z=1.0),
        orientation_note="Print on a flat face; TPMS / strut lattices self-support and need no supports.",
        material=_DEFAULT_MATERIAL,
        recommended_min_wall_mm=None,
        layer_height_mm=0.2,
        wall_note="The lattice wall / strut is the printed feature -- use 2+ perimeters to realize it.",
        infill_note="The lattice IS the infill; print perimeters-only (no additional infill).",
        insert_install_note=None,
        steps=[
            f"Print the {part.archetype} lattice in {_DEFAULT_MATERIAL}.",
            "No supports are needed for TPMS / strut lattices.",
        ],
        advisory=True,
        caveat=(
            "Implicit / field geometry: delivered as a watertight STL. An editable parametric STEP is "
            "not available for this geometry type -- a lattice has no faithful B-rep equivalent."
        ),
    )
    return Kit(
        part_name=part.name, bom=BillOfMaterials(lines=[], unmapped=[]), instructions=instructions
    )


def _generate_implicit(req: ImplicitRequest, prompt: str, job_dir: Path) -> GenerateResult:
    """Build a deterministic implicit lattice and project it into the /generate contract. Free (no
    LLM), mesh-only. Never raises -- a build/export failure comes back as an error result."""
    try:
        part = build_implicit(req)
        mesh: Any = part.mesh
        mesh.export(str(job_dir / STL_NAME))  # STL stays Z-up for slicers
        to_gltf_yup(mesh.copy()).export(str(job_dir / GLB_NAME))  # viewer GLB is Y-up
    except Exception as exc:
        return GenerateResult(
            status="error",
            error=GenerateError(stage="implicit", message=f"{type(exc).__name__}: {exc}"),
        )
    if not (job_dir / STL_NAME).exists() or not (job_dir / GLB_NAME).exists():
        return GenerateResult(
            status="error",
            error=GenerateError(
                stage="export", message="The lattice built but could not be exported."
            ),
        )
    return GenerateResult(
        status="ok",
        ok=GenerateOk(
            prompt=prompt,
            gen_path=f"implicit:{req.archetype}",
            cost_usd=0.0,
            validation=_implicit_validation(part.watertight),
            kit=_implicit_kit(part),
            files=Files(glb=GLB_NAME, stl=STL_NAME),
        ),
    )


def _report_view(report: Any) -> ValidationView:
    """Project a DSL-path :class:`ValidationReport` into the frontend's ValidationView. STRICT --
    no advisory filtering: the standard lane's templates are designed to pass the full print gates
    (ADR-0029 keeps the DSL path strict), so every check renders verbatim."""
    tiers = [
        TierStatus(tier=tier.value, status=tr.status.value) for tier, tr in report.tiers.items()
    ]
    # INFO = advisory measured evidence (ADR-0035 B1): never in the user-facing failed list.
    failed = [
        CheckView(id=c.id, status=c.status.value, message=c.message)
        for c in report.checks
        if c.status not in (CheckStatus.PASS, CheckStatus.INFO)
    ]
    return ValidationView(
        passed=report.passed,
        overall=("warn" if failed else "pass") if report.passed else report.overall.value,
        tiers=tiers,
        failed_checks=failed,
        step_proof=_step_view(report.step_proof),
    )


def _bridge_kit(pipeline_kit: Any, part_name: str) -> Kit:
    """Project the pipeline's registry-backed Kit into the API schema (the schema was modeled on
    it, so a dump/validate round-trip is the bridge). Falls back to a minimal honest kit -- never
    a fake registry citation -- if the shapes ever drift."""
    try:
        return Kit.model_validate(pipeline_kit.model_dump(mode="json"))
    except Exception:
        return Kit(
            part_name=part_name,
            bom=BillOfMaterials(lines=[], unmapped=[]),
            instructions=MakeInstructions(
                print_orientation=Vec3(x=0.0, y=0.0, z=1.0),
                orientation_note="Orient the largest flat face down.",
                material=_DEFAULT_MATERIAL,
                recommended_min_wall_mm=None,
                layer_height_mm=0.2,
                wall_note="Use enough perimeters to realize the minimum wall thickness.",
                infill_note="Advisory: 20-40% infill suits most functional parts.",
                insert_install_note=None,
                steps=[f"Fabricate the part in {_DEFAULT_MATERIAL}."],
                advisory=True,
                caveat=_CAVEAT,
            ),
        )


def _assembly_kit(model: Any, part_name: str) -> Kit:
    """The assembly lane's REAL bill of materials, built from the graph's hardware
    nodes + the cited mount interfaces (the old ``_bridge_kit(None, ...)`` fallback
    shipped an EMPTY BOM under the flagship lane -- a blank panel reads as a toy).
    Every line cites the same registry source the geometry was built from."""
    from forge_mind.viz.assembly_glb import hardware_source

    lines: list[BomLine] = []
    servo_count = sum(1 for h in model.hardware if h.kind == "mg996r_servo")
    dowel_count = sum(1 for h in model.hardware if h.kind == "dowel_pin")
    if servo_count:
        src = hardware_source("mg996r_servo") or ""
        lines.append(
            BomLine(
                component="MG996R hobby servo",
                kind="actuator",
                key="mg996r",
                quantity=servo_count,
                source=src,
                provisional=True,  # the registry row is provisional (ADR-0040)
            )
        )
        lines.append(
            BomLine(
                component="Servo tab screw (M4, fits the 4.0-4.8 mm tab holes)",
                kind="fastener",
                key="M4",
                quantity=4 * servo_count,
                source="ISO 4762 socket head cap screw; MG996R tab clearance (TowerPro)",
                length_mm=None,
                length_note="length depends on the cradle plate stack; 10-12 mm typical",
            )
        )
        lines.append(
            BomLine(
                component="Servo horn + horn screws (supplied with the servo)",
                kind="fastener",
                key="mg996r_horn",
                quantity=servo_count,
                source=src,
                provisional=True,
            )
        )
    if dowel_count:
        lines.append(
            BomLine(
                component="Dowel pin 6 x 40 mm (idler-finger pivot)",
                kind="dowel",
                key="6",
                quantity=dowel_count,
                source=hardware_source("dowel_pin") or "ISO 2338",
            )
        )
    return Kit(
        part_name=part_name,
        bom=BillOfMaterials(lines=lines, unmapped=[]),
        instructions=MakeInstructions(
            print_orientation=Vec3(x=0.0, y=0.0, z=1.0),
            orientation_note="Print each part flat as exported (per-part STLs in parts/).",
            material=_DEFAULT_MATERIAL,
            recommended_min_wall_mm=None,
            layer_height_mm=0.2,
            wall_note="Use enough perimeters to realize the minimum wall thickness.",
            infill_note="Advisory: 30-50% infill for the load-bearing links/fingers.",
            insert_install_note=None,
            steps=[
                f"Print every printed part in {_DEFAULT_MATERIAL} (see parts/).",
                "Bolt each servo into its cradle window with the tab screws.",
                "Attach each driven part to its servo horn with the horn screws.",
            ],
            advisory=True,
            caveat=_CAVEAT,
        ),
    )


def _assembly_view(manifest: dict[str, Any], step_note: str | None) -> AssemblyView:
    """Project the exporter's manifest + gate map into the API block VERBATIM --
    the validator/exporter stay the sole issuers of every value here (recomputing
    any of it in the API layer could let two surfaces disagree)."""
    return AssemblyView(
        parts=[
            AssemblyPartView(
                id=p["id"],
                printed=bool(p["printed"]),
                hardware_kind=p.get("hardware_kind"),
                source=p.get("source"),
            )
            for p in manifest.get("parts", [])
        ],
        joints=[
            AssemblyJointView(
                id=j["id"],
                name=j.get("name"),
                type=j["type"],
                limits=[float(x) for x in j["limits"]] if j.get("limits") else None,
                checked=j["checked"],
                mimic_of=(j.get("mimic") or {}).get("of"),
                mimic_ratio=(j.get("mimic") or {}).get("ratio"),
            )
            for j in manifest.get("joints", [])
        ],
        gate=dict(manifest.get("gate", {}).get("checks", {})),
        step_note=step_note,
    )


def _generate_assembly(
    prompt: str, job_dir: Path, on_phase: Callable[[str, str], None]
) -> GenerateResult | None:
    """Articulated-assembly lane (flag-gated via ``FORGE_ASSEMBLY_LANE``, default OFF).

    A cited-servo serial chain (MG996R arm / pan-tilt / turntable) builds DETERMINISTICALLY:
    regex spec extraction (zero LLM), the arm-family generator, the constraint solver, and
    the full assembly claim gate -- static fit, per-joint chain-vs-rest swept collision
    (hardware envelopes included) + corner poses, cited mount patterns, coaxiality. Ships
    ONLY a gate PASS; anything else returns None and the agentic engine runs (strictly-
    no-worse, same contract as the standard lane). The GLB is the CK_JOINT v1 contract
    (parented nodes + extras.ck_joint with the EARNED ``checked`` level), so the viewer
    can articulate it honestly."""
    from forge_mind.assembly.arm_match import arm_match, joint_display_names

    arm_spec = arm_match(prompt)
    gripper_spec = None
    if arm_spec is None and os.environ.get("FORGE_GRIPPER") == "1":
        # The gripper family (ADR-0044) is DOUBLY flag-gated: inside the assembly
        # lane flag AND behind FORGE_GRIPPER, so existing arm prompts hit
        # byte-identical code paths until the gripper is promoted.
        from forge_mind.assembly.gripper_match import gripper_match

        gripper_spec = gripper_match(prompt)
    if arm_spec is None and gripper_spec is None:
        return None
    start = time.monotonic()
    try:
        import trimesh

        from forge_mind.agentic.confidence import assembly_lane_label
        from forge_mind.assembly.solver import solve
        from forge_mind.compiler.compile_assembly import compile_assembly
        from forge_mind.validator.assembly_checks import validate_assembly
        from forge_mind.viz.assembly_glb import export_assembly_glb

        on_phase("plan", "active")
        payload_key: str | None = None
        if arm_spec is not None:
            from forge_mind.assembly.arm_family import build_arm_graph

            spec_name = arm_spec.name
            graph = build_arm_graph(arm_spec)
            names = joint_display_names(arm_spec.dof)
            gen_path = "assembly:servo_arm"
            part_name = (
                "MG996R pan-tilt head"
                if arm_spec.name == "pan_tilt"
                else f"MG996R {arm_spec.dof}-DOF servo arm"
            )
        else:
            from forge_mind.assembly.gripper_family import (
                build_gripper_graph,
                gripper_joint_display_names,
            )

            assert gripper_spec is not None
            spec_name = gripper_spec.name
            graph = build_gripper_graph(gripper_spec)
            names = gripper_joint_display_names()
            gen_path = "assembly:gripper"
            part_name = "MG996R two-finger gripper"
            if gripper_spec.payload_declared:
                payload_key = gripper_spec.payload
        on_phase("plan", "done")
        on_phase("cad", "active")
        model = compile_assembly(solve(graph), job_dir / "parts")
        report = validate_assembly(model, payload_object=payload_key)
        if not report.passed:
            failed = report.failed_check_ids()
            _LOG.warning("assembly lane: claim gate not passed (%s); agentic fall-through", failed)
            return None
        # report.passed means no FAIL and no WARN: every moving joint's sweep ran and
        # passed (a skipped sweep WARNs), so the swept claim is earned here.
        manifest = export_assembly_glb(
            model,
            job_dir / GLB_NAME,
            report=report,
            joint_names=names,
        )
        # The download STL is the PRINTED parts at home pose (the print files are the
        # per-part STLs in parts/ -- servo envelopes are bought, not printed).
        placed = []
        for body in model.bodies:
            m = body.mesh.copy()
            m.apply_transform(body.transform)
            placed.append(m)
        combined = trimesh.util.concatenate(placed)
        combined.export(str(job_dir / STL_NAME))
        manifest["gate"] = {
            "overall": report.overall.value,
            "checks": {c.id: c.status.value for c in report.checks if c.id.startswith("assembly.")},
        }
        (job_dir / "parts.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # The positioned multi-body STEP (printed parts only; hardware ships as BOM
        # lines + GLB envelopes -- see ASSEMBLY_STEP_NOTE). Named part.step so the
        # existing /files route + the frontend's STEP download serve it unchanged.
        # Best-effort like every artifact step: a STEP failure never sinks a PASS.
        from forge_mind.assembly.assembly_step import ASSEMBLY_STEP_NOTE, export_assembly_step

        step_note: str | None = None
        if export_assembly_step(model, job_dir / "part.step", name=spec_name) is not None:
            step_note = ASSEMBLY_STEP_NOTE
        # A merged-mesh preview PNG (printed + envelopes): the same renderer single
        # parts use, on the full placed assembly.
        try:
            from forge_mind.viz.render import render_part_to_png

            full = list(placed)
            for hw in model.hardware:
                if hw.mesh is not None:
                    m = hw.mesh.copy()
                    m.apply_transform(hw.transform)
                    full.append(m)
            preview = trimesh.util.concatenate(full)
            preview_stl = job_dir / "assembly_preview.stl"
            preview.export(str(preview_stl))
            render_part_to_png(preview_stl, job_dir / "part.png", title=spec_name, caption=prompt)
        except Exception:  # pragma: no cover - presentation artifact, never gates
            pass
        on_phase("cad", "done")
        on_phase("export", "active")
        # Sourced dimension groups, COMPUTED from what the gate actually measured
        # (never a per-family hardcode -- a stale count would be a self-claimed
        # provenance figure, the exact ADR-0040 sin): one group per cited
        # mount-pattern check that ran, one per registry-cited hardware envelope,
        # one for the cited payload object when the aperture claim was measured.
        sourced_groups = (
            sum(1 for c in report.checks if c.id.startswith("assembly.mount_pattern."))
            + sum(1 for h in model.hardware if h.kind)
            + sum(1 for c in report.checks if c.id == "assembly.gripper_aperture")
        )
        label = assembly_lane_label(sourced_groups, swept=True, fabrication_mode="general")
        _LOG.info(
            "assembly lane SHIPPED: %s (%d parts, %d joints) in %.1fs, $0 LLM",
            spec_name,
            len(model.bodies) + len(model.hardware),
            len(manifest["joints"]),
            time.monotonic() - start,
        )
        return GenerateResult(
            status="ok",
            ok=GenerateOk(
                prompt=prompt,
                gen_path=gen_path,
                cost_usd=0.0,
                validation=_report_view(report),
                kit=_assembly_kit(model, part_name),
                files=Files(glb=GLB_NAME, stl=STL_NAME),
                fabrication_mode="general",
                confidence=_confidence_view(label),
                assembly=_assembly_view(manifest, step_note),
            ),
        )
    except Exception as exc:  # capture-and-fall-through: the agentic engine still runs
        _LOG.warning("assembly lane raised (%s: %s); agentic fall-through", type(exc).__name__, exc)
        return None


# --- The physics route (T3 / ADR-0102): FORGE_PHYSICS_ROUTE=1, default OFF -> byte-inert ------

# The design volume used when the user stated none -- ALWAYS labeled "defaulted" on the
# certificate (the engine never silently invents dims; it flags them).
_PHYSICS_DEFAULT_ENVELOPE_MM = (150.0, 30.0, 60.0)


def _physics_validation(gate: dict[str, Any]) -> ValidationView:
    """Project the uniform gate's per-check verdicts into the frontend view, verbatim: fail ->
    failed check; unproven -> warn (shown, never hidden); n/a is an explicit non-claim, not a
    failure. ``passed`` mirrors ``claimable`` (the strict conjunction)."""
    checks: dict[str, str] = gate["checks"]
    failed = [
        CheckView(
            id=f"physics.{cid}",
            status="fail" if verdict == "fail" else "warn",
            message=(
                "the uniform gate failed this check"
                if verdict == "fail"
                else "unproven on this run (explicit non-claim, not a pass)"
            ),
        )
        for cid, verdict in checks.items()
        if verdict in ("fail", "unproven")
    ]
    claimable = bool(gate["claimable"])
    return ValidationView(
        passed=claimable,
        overall="pass" if claimable else ("warn" if failed else "pass"),
        tiers=[TierStatus(tier="physics_gate", status="pass" if claimable else "warn")],
        failed_checks=failed,
    )


def _physics_confidence(extraction: dict[str, Any], gate: dict[str, Any]) -> ConfidenceView:
    """The honest confidence block for a physics-authored part: basis is VALIDATED (the claim
    passed the uniform gate incl. real FEA under the spec's own case), with the envelope's
    stated-vs-defaulted provenance carried as the thing to verify when defaulted."""
    defaulted = extraction.get("envelope_provenance") == "defaulted"
    stated_dims = len(extraction.get("placed_components") or []) + (0 if defaulted else 1)
    return ConfidenceView(
        overall="high",
        basis="validated",
        manufacturability="prints_as_is",  # claimable requires min-wall PASS + a real slice PASS
        process_note=None,
        dimension_tiers=DimensionTierCounts(
            verified=1 if gate["checks"].get("deflection_bound") == "pass" else 0,
            sourced=0,
            stated=stated_dims,
            estimated=1 if defaulted else 0,
        ),
        dimensions=[],
        verify_dimensions=(
            ["overall envelope (the engine used a labeled default design volume)"]
            if defaulted
            else []
        ),
        not_fit_for=[],
        part_class="structural",
        summary=(
            "Physics-authored structural part: geometry computed from the declared load case and "
            "verified by real FEA + print gates (the certificate is the claim record)."
        ),
    )


def _physics_kit(extraction: dict[str, Any], material_id: str, gate: dict[str, Any]) -> Kit:
    """A minimal honest Kit for a physics-authored part: no hardware BOM, solid-infill print
    guidance (the FEA verdict assumes solid material), certificate pointer in the caveat."""
    part_name = str(extraction.get("part_name") or "structural part")
    min_wall = gate["measured"].get("min_wall_mm")
    instructions = MakeInstructions(
        print_orientation=Vec3(x=0.0, y=0.0, z=1.0),
        orientation_note="Print on the largest flat face.",
        material=material_id,
        recommended_min_wall_mm=float(min_wall) if min_wall is not None else None,
        layer_height_mm=0.2,
        wall_note=(
            f"Thinnest structural wall measured {min_wall} mm; print with 2+ perimeters."
            if min_wall is not None
            else "Print with 2+ perimeters."
        ),
        infill_note="Structural part: print SOLID (100% infill) -- the FEA verdict assumes solid material.",
        insert_install_note=None,
        steps=[
            f"Print the {part_name} in {material_id} at 100% infill.",
            "Check the certificate (certificate.html) for the measured claim before structural use.",
        ],
        advisory=True,
        caveat=(
            "Physics-authored structural part; the certificate is the claim record (spec echo in "
            "human units, per-check verdicts, provenance, measured numbers)."
        ),
    )
    return Kit(
        part_name=part_name, bom=BillOfMaterials(lines=[], unmapped=[]), instructions=instructions
    )


def _physics_certificate(
    extraction: dict[str, Any],
    gate: dict[str, Any],
    provenance: str,
    ref_mass_g: float | None,
    attempts: list[dict[str, Any]],
) -> CertificateView:
    """Compose THE CERTIFICATE from computed/echoed facts only (never LLM prose): the spec echo
    (human-units load case + envelope provenance), the gate's per-check verdicts + measured
    numbers, the author provenance, honest caveats, and the monotone fallback trail."""
    caveats: list[str] = []
    if ref_mass_g is not None:
        caveats.append("reference optimizes cost/assembly, not mass")
    if extraction.get("envelope_provenance") == "defaulted":
        caveats.append(
            f"envelope was not stated; the engine used a labeled default design volume of "
            f"{extraction.get('envelope_mm')} mm"
        )
    bp = extraction.get("bound_provenance")
    if isinstance(bp, str) and bp.startswith("defaulted"):
        caveats.append(
            f"no stiffness bound was stated; the part was sized to a labeled default "
            f"({bp} = {extraction.get('deflection_bound_mm')} mm) -- state a bound for a "
            f"precision requirement"
        )
    dropped = extraction.get("dropped_components") or []
    if dropped:
        caveats.append(f"unrecognized components were dropped, never guessed: {dropped}")
    spec_echo = {
        key: extraction.get(key)
        for key in (
            "part_name",
            "load_case_echo",
            "load_note_verbatim",
            "envelope_mm",
            "envelope_provenance",
            "deflection_bound_mm",
            "bound_provenance",  # stated | defaulted: span/N ... (B2a -- never a hidden bound)
            "placed_components",
        )
    }
    return CertificateView(
        spec_echo=spec_echo,
        checks=dict(gate["checks"]),
        provenance=provenance,
        measured=dict(gate["measured"]),
        caveats=caveats,
        claimable=bool(gate["claimable"]),
        attempts=[
            {"rung": a["rung"], "provenance": a["provenance"], "claimable": a["gate"]["claimable"]}
            for a in attempts
        ],
    )


def render_certificate_html(cert: CertificateView, prompt: str) -> str:
    """One-page static HTML certificate (the demo prop + the flywheel label): pure computed
    facts, everything user-derived escaped, zero scripts/deps."""
    import html as _html

    def esc(value: Any) -> str:
        return _html.escape(str(value))

    check_rows = "\n".join(
        f"<tr><td>{esc(cid)}</td><td class='v-{esc(v)}'>{esc(v).upper()}</td></tr>"
        for cid, v in cert.checks.items()
    )
    measured_rows = "\n".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in cert.measured.items()
    )
    echo_rows = "\n".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in cert.spec_echo.items()
    )
    caveat_items = "\n".join(f"<li>{esc(c)}</li>" for c in cert.caveats) or "<li>none</li>"
    attempt_rows = "\n".join(
        f"<tr><td>{esc(a.get('rung'))}</td><td>{esc(a.get('provenance'))}</td>"
        f"<td>{'claimable' if a.get('claimable') else 'refused'}</td></tr>"
        for a in cert.attempts
    )
    verdict = "CLAIMABLE" if cert.claimable else "NOT CLAIMABLE"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>CadKitty part certificate</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; color: #111; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.05rem; margin-top: 1.4rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 td {{ border: 1px solid #ccc; padding: 4px 8px; font-size: 0.9rem; }}
 .verdict {{ font-weight: bold; padding: 6px 10px; display: inline-block; border-radius: 4px; }}
 .ok {{ background: #e6f4e6; color: #1a6b1a; }} .no {{ background: #fbeaea; color: #8a1f1f; }}
 .v-pass {{ color: #1a6b1a; }} .v-fail {{ color: #8a1f1f; }} .v-unproven, .v-n\\/a {{ color: #7a6a1f; }}
 .prov {{ color: #444; font-size: 0.9rem; }}
</style></head><body>
<h1>Part certificate</h1>
<p class="prov">request: {esc(prompt)}</p>
<p><span class="verdict {"ok" if cert.claimable else "no"}">{verdict}</span>
 &nbsp; provenance: <b>{esc(cert.provenance)}</b></p>
<h2>Requirement echo (what was asked, in human units)</h2>
<table>{echo_rows}</table>
<h2>Per-check verdicts (the uniform gate)</h2>
<table>{check_rows}</table>
<h2>Measured</h2>
<table>{measured_rows}</table>
<h2>Honest caveats</h2>
<ul>{caveat_items}</ul>
<h2>Author attempts (monotone fallback trail)</h2>
<table>{attempt_rows}</table>
<p class="prov">Every value above is computed or echoed -- never composed by a language model.
Checks: pass / fail / unproven (explicit non-claim) / n&#47;a (not applicable).</p>
</body></html>
"""


def _agentic_certificate(
    result: PartResult, fab: FabricationMode, prompt: str
) -> CertificateView | None:
    """A MANUFACTURABILITY certificate for the DEFAULT (agentic) path -- the trustworthy verdict on
    EVERY part, not only the physics lane (ADR-0109). Maps the full validator's per-check verdicts to
    the SAME legible CertificateView, with caveats that keep the SCOPE honest: this certifies geometry
    + manufacturability (watertight, min-wall, declared features, slice), NOT load capacity -- no load
    was specified. None when there is no per-part report (nothing deterministic to certify)."""
    val = result.loop.validation
    if val is None or val.report is None:
        return None
    verdict = {
        CheckStatus.PASS: "pass",
        CheckStatus.FAIL: "fail",
        CheckStatus.WARN: "unproven",
        CheckStatus.INFO: "n/a",
    }
    checks = {c.id: verdict.get(c.status, "unproven") for c in val.report.checks}
    brief = result.loop.brief
    spec_echo: dict[str, Any] = {
        "object": brief.object_name if brief is not None else prompt[:80],
        "envelope_mm": getattr(brief, "overall_envelope_mm", None) if brief is not None else None,
        "features_declared": len(brief.features) if brief is not None else None,
        "material": (getattr(brief, "material", None) if brief is not None else None)
        or _DEFAULT_MATERIAL,
        "fabrication_mode": fab.value,
    }
    caveats = [
        "certifies MANUFACTURABILITY (watertight, min-wall, declared features, slice) -- NOT load "
        'capacity; no load was specified. State a load ("holds 5 kg") for a strength/stiffness '
        "certificate."
    ]
    if fab == FabricationMode.GENERAL:
        caveats.append(
            "validated for GENERAL fabrication, not claimed 3D-printable (overhang unchecked, "
            "min-wall relaxed)."
        )
    if val.features_missing:
        caveats.append("declared features are MISSING -- see the failing feature check.")
    return CertificateView(
        spec_echo=spec_echo,
        checks=checks,
        provenance="agentic",
        measured={"disposition": result.disposition, "hard_ok": bool(val.hard_ok)},
        caveats=caveats,
        claimable=bool(val.passed),
        attempts=[],
    )


def _generate_physics(
    prompt: str, job_dir: Path, phase: Callable[[str, str], None]
) -> tuple[GenerateResult | None, CertificateView | None]:
    """The physics route (T3, gated on FORGE_PHYSICS_ROUTE=1; unset -> byte-inert).

    Returns ``(result, refusal_cert)``. One cheap structured extraction turns intent into the
    typed RequirementSpec; the deterministic router decides. Routed AGENTIC (no declared load --
    "sturdy" is not a load) -> ``(None, None)``: today's path runs UNCHANGED. Routed
    analytic/topopt -> ``author_part`` (T2) with the LLM floor as a REFUSAL. Claimable ->
    ``(GenerateResult, None)``: servable files + THE CERTIFICATE. NON-claimable (a load WAS
    declared, no author could prove it) -> ``(None, refusal_cert)`` (B3): the claimable:false
    certificate rides the agentic fall-through so the REFUSAL is visible -- attempted, could not
    prove, here is exactly which check failed. Fail-open (None, None) on any lane error."""
    if os.environ.get("FORGE_PHYSICS_ROUTE") != "1":
        return None, None
    try:
        from forge_mind.structures.dispatch import author_part, refusing_author
        from forge_mind.structures.extract import extract_requirements, to_spec
        from forge_mind.structures.router import AGENTIC, route

        phase("plan", "active")
        extracted = extract_requirements(prompt)
        spec, extraction = to_spec(extracted, default_envelope_mm=_PHYSICS_DEFAULT_ENVELOPE_MM)
        if route(spec) == AGENTIC:
            return None, None  # no declared load -> the production agentic path, unchanged
        phase("plan", "done")
        phase("cad", "active")
        outcome = author_part(
            spec,
            job_dir / "physics",
            authors={
                "agentic": refusing_author(
                    "agentic",
                    "physics route: the agentic engine is today's production path, "
                    "not a physics fallback",
                )
            },
            part_name=str(extraction.get("part_name") or "structural part"),
        )
        phase("cad", "done")
        final = outcome["final"]
        gate = outcome["attempts"][-1]["gate"]
        # T6 flywheel label (fail-open, additive): the method-SELECTION signal -- which author a
        # typed spec routed to and whether its uniform gate was claimable -- is what the near-term
        # flywheel learns. Logged whether or not the part shipped (a decline is a selection too).
        try:
            import dataclasses as _dc

            from forge_mind.train_log import log_selection

            log_selection(
                prompt,
                provenance=final.provenance,
                spec=_dc.asdict(spec),
                checks=dict(gate["checks"]),
                claimable=bool(gate["claimable"]),
                material=spec.material_id,
                object_name=str(extraction.get("part_name") or None),
            )
        except Exception:  # a label must never affect the response
            pass
        if not gate["claimable"] or final.stl is None:
            # WARNING with per-rung errors: a production decline must be diagnosable from the
            # server log alone (the smoke's first decline was invisible at INFO -- row-78 lesson).
            _LOG.warning(
                "physics route: no claimable physics author; falling through to agentic. "
                "attempts=%s",
                [
                    {
                        "rung": a["rung"],
                        "claimable": a["gate"]["claimable"],
                        "error": a["self_reported"].get("error"),
                    }
                    for a in outcome["attempts"]
                ],
            )
            # B3 -- the VISIBLE REFUSAL (half the moat): a load was declared and NO author could
            # prove it. Ride the claimable:false certificate on the agentic fall-through so the
            # user sees exactly which check failed, not a silent guess. Fail-open.
            try:
                refusal = _physics_certificate(
                    extraction, gate, final.provenance, spec.ref_mass_g, outcome["attempts"]
                )
            except Exception:
                refusal = None
            return None, refusal
        phase("export", "active")
        import trimesh

        shutil.copyfile(final.stl, job_dir / STL_NAME)
        mesh: Any = trimesh.load(str(final.stl), force="mesh")
        to_gltf_yup(mesh.copy()).export(str(job_dir / GLB_NAME))
        if final.step is not None and Path(final.step).exists():
            shutil.copyfile(final.step, job_dir / "part.step")
        cert = _physics_certificate(
            extraction, gate, final.provenance, spec.ref_mass_g, outcome["attempts"]
        )
        (job_dir / "certificate.html").write_text(
            render_certificate_html(cert, prompt), encoding="utf-8"
        )
        return (
            GenerateResult(
                status="ok",
                ok=GenerateOk(
                    prompt=prompt,
                    gen_path=f"physics:{final.provenance}",
                    # one structured extraction call (subscription-routed); authoring is deterministic
                    cost_usd=0.0,
                    validation=_physics_validation(gate),
                    kit=_physics_kit(extraction, spec.material_id, gate),
                    files=Files(glb=GLB_NAME, stl=STL_NAME),
                    fabrication_mode="print_3d",  # claimable requires min-wall PASS + a real slice PASS
                    disposition="shipped",
                    confidence=_physics_confidence(extraction, gate),
                    certificate=cert,
                ),
            ),
            None,
        )
    except Exception as exc:  # fail-open: the lane never takes down /generate
        _LOG.warning(
            "physics route failed open to the agentic path: %s: %s", type(exc).__name__, exc
        )
        return None, None


def _generate_standard(
    prompt: str, job_dir: Path, on_phase: Callable[[str, str], None]
) -> tuple[GenerateResult | None, StandardLaneOutcome]:
    """Standard-parts lane (ADR-0034): build a CANONICAL part on the measured-~100% deterministic
    LayoutSpec pipeline instead of the ~25-50% stochastic author. Returns ``(result, outcome)``;
    ``result`` is None -- and the caller FALLS THROUGH to the agentic engine -- on refusal, any
    stage failure, a failed validation, a direct-DSL fallback (never shipped under the
    "standard:" label -- ADR-0035 decision a), or missing artifacts, so a misroute costs one
    cheap attempt and never a lost generation. Only a part the DETERMINISTIC interpreter built
    AND that PASSED the strict DSL gates ships from here. The outcome classifies every attempt
    for telemetry -- nothing is swallowed silently anymore."""
    start = time.monotonic()

    def _miss(outcome: str, result: Any = None, **kw: Any) -> tuple[None, StandardLaneOutcome]:
        lane = StandardLaneOutcome(
            attempted=True,
            outcome=outcome,
            gen_path=None if result is None else result.gen_path,
            fallback_reason=None if result is None else result.fallback_reason,
            duration_s=time.monotonic() - start,
            cost_usd=None if result is None else float(result.total_cost_usd),
            **kw,
        )
        _LOG.info(
            "standard lane MISS (%s): gen_path=%s fallback=%s error=%s -- agentic fall-through",
            lane.outcome,
            lane.gen_path,
            lane.fallback_reason,
            lane.error,
        )
        return None, lane

    try:
        from forge_mind.pipeline import run_pipeline

        # run_pipeline emits intent/plan/cad/validate; the /generate stream speaks plan/cad/export.
        keymap = {"intent": "plan", "plan": "plan", "cad": "cad", "validate": "cad"}

        def _on_event(kind: str, payload: dict[str, Any]) -> None:
            if kind != "phase":
                return
            key = keymap.get(str(payload.get("key", "")))
            if key is not None:
                on_phase(key, str(payload.get("status", "active")))

        result = run_pipeline(prompt, job_dir, use_layoutspec=True, on_event=_on_event)
        if result.refusal is not None:
            return _miss("refused", result, refusal=result.refusal)
        if result.failed_stage is not None:
            # The pipeline marks a built-but-not-passing part as failed_stage="validate"; keep
            # that distinct from a stage that died outright (intent/blueprint/compile/...).
            if result.failed_stage == "validate" and result.report is not None:
                return _miss("validation_failed", result, error="strict gates not passed")
            return _miss("failed_stage", result, error=f"{result.failed_stage}: {result.error}")
        if result.gen_path != "layoutspec":
            # ADR-0035 decision (a): the router declined and run_pipeline fell back to the
            # one-shot direct-DSL author (measured 7-23%). That output -- even when it passes
            # validation -- is a stochastic re-imagining, not the canonical deterministic part
            # the "standard:" label promises. Block it; the agentic engine (~25-50%) gets the
            # prompt instead, so this is strictly-no-worse AND the label stays true.
            return _miss("fallback_blocked", result)
        if not result.passed or result.model is None or result.report is None:
            return _miss("validation_failed", result, error="strict gates not passed")

        stl_src = Path(result.model.stl_path)
        if not stl_src.exists():
            return _miss("missing_artifacts", result, error="pipeline STL missing on disk")
        stl_dst = job_dir / STL_NAME
        if stl_src.resolve() != stl_dst.resolve():
            shutil.copy2(stl_src, stl_dst)
        if not _stl_to_glb(stl_dst, job_dir / GLB_NAME):
            # the viewer needs the GLB; without it, let the agentic path try
            return _miss("missing_artifacts", result, error="GLB conversion failed")
        step_src = Path(result.model.step_path) if result.model.step_path else None
        if step_src is not None and step_src.exists():
            shutil.copy2(step_src, job_dir / "part.step")

        part_name = result.program.metadata.name if result.program is not None else "Standard part"
        # The LayoutSpec is the part's PARAMETRIC MODEL (ADR-0037): persist it next to the
        # artifacts and ride it on the response so /patch can re-derive from it statelessly.
        spec_dump = result.spec.model_dump(mode="json") if result.spec is not None else None
        if spec_dump is not None:
            (job_dir / "part.layoutspec.json").write_text(
                json.dumps(spec_dump, indent=2), encoding="utf-8"
            )
        on_phase("export", "active")
        lane = StandardLaneOutcome(
            attempted=True,
            outcome="shipped",
            gen_path=f"standard:{result.gen_path}",
            duration_s=time.monotonic() - start,
            cost_usd=float(result.total_cost_usd),
        )
        _LOG.info(
            "standard lane SHIPPED: gen_path=%s cost=$%.4f in %.1fs",
            lane.gen_path,
            lane.cost_usd or 0.0,
            lane.duration_s or 0.0,
        )
        return (
            GenerateResult(
                status="ok",
                ok=GenerateOk(
                    prompt=prompt,
                    gen_path=f"standard:{result.gen_path}",
                    cost_usd=round(result.total_cost_usd, 4),
                    validation=_report_view(result.report),
                    kit=(
                        _bridge_kit(result.kit, part_name)
                        if result.kit is not None
                        else _bridge_kit(None, part_name)
                    ),
                    files=Files(glb=GLB_NAME, stl=STL_NAME),
                    # The standard lane validates against the FULL strict print gates (the DSL
                    # path), so the result is honestly labeled print-ready regardless of the
                    # user's toggle.
                    fabrication_mode="print_3d",
                    spec=spec_dump,
                    # Registry-derived interfaces + strict gates ARE the evidence here: the
                    # deterministic lane earns the top label by construction (ADR-0041).
                    confidence=_confidence_view(
                        standard_lane_label(
                            len(result.program.interfaces) + len(result.program.hardware_features)
                            if result.program is not None
                            else 0
                        )
                    ),
                ),
                lane=lane,
            ),
            lane,
        )
    except Exception as exc:  # capture-classify-fall-through: the agentic engine still runs
        lane = StandardLaneOutcome(
            attempted=True,
            outcome="exception",
            error=f"{type(exc).__name__}: {exc}",
            duration_s=time.monotonic() - start,
        )
        _LOG.warning("standard lane attempt raised (%s); agentic fall-through", lane.error)
        return None, lane


def generate_part(
    prompt: str,
    job_dir: Path,
    *,
    quality: str = "standard",
    fabrication_mode: str = "general",
    on_phase: Callable[[str, str], None] | None = None,
    brief_context: str | None = None,
    client: Any | None = None,
) -> GenerateResult:
    """Run the agentic engine for ``prompt`` and project it into the API contract. Never raises on a
    generation failure -- a bad/infeasible prompt comes back as refused/error.

    ``on_phase(key, status)`` (optional) is called as each live stage actually starts/ends -- ``plan``
    (brief research), ``cad`` (author + build + validate), ``export`` (finalize) -- so the SSE stream
    reports REAL progress instead of a scripted heartbeat. ``None`` (the non-stream path) is a no-op.

    ``brief_context`` (ADR-0036) is the guided intake's typed STATED-FACTS block, threaded into the
    agentic brief research so user-confirmed facts bind generation instead of being re-guessed from
    prose. None (legacy callers / no intake session) is byte-identical.

    ``client`` (optional) injects the AGENTIC engine's LLM client -- the bench harness passes the
    subscription provider here (``FORGE_LLM_PROVIDER=claude_code``) so paid iteration runs bill
    the plan instead of the API key. None (the server / all production callers) is byte-identical.
    The deterministic lanes' own small spec-pick calls are NOT affected by this parameter."""
    job_dir.mkdir(parents=True, exist_ok=True)

    def _phase(key: str, status: str) -> None:
        if on_phase is not None:
            on_phase(key, status)

    # Implicit / voxel lane (ADR-0026): a lattice / TPMS request is built deterministically (no LLM,
    # no agentic engine). Keyword-gated, so a normal part never matches and the B-rep path is unchanged.
    implicit_req = implicit_match(prompt)
    if implicit_req is not None:
        # No LLM planning for the deterministic lane; still drive the phases so the UI advances.
        _phase("plan", "active")
        _phase("plan", "done")
        _phase("cad", "active")
        res = _generate_implicit(implicit_req, prompt, job_dir)
        _phase("cad", "done")
        if res.status == "ok":
            _phase("export", "active")
        return res

    # Articulated-assembly lane (flag-gated, DEFAULT OFF): a cited-servo serial chain
    # (MG996R arm / pan-tilt) builds deterministically and ships only on a full assembly
    # claim-gate PASS with the earned swept-collision level. Behind FORGE_ASSEMBLY_LANE=1
    # until the frozen breadth benchmark confirms the gain (the promotion rule); unset,
    # this branch is byte-inert.
    if os.environ.get("FORGE_ASSEMBLY_LANE") == "1":
        asm = _generate_assembly(prompt, job_dir, _phase)
        if asm is not None:
            return asm

    # Physics route (T3 / ADR-0102, flag-gated FORGE_PHYSICS_ROUTE=1, DEFAULT OFF): a stated
    # load routes to the deterministic physics authors (analytic/topopt) via extract -> route ->
    # author_part; the response gains THE CERTIFICATE. Everything else -- including "sturdy"
    # with no declared load -- falls through to today's path UNCHANGED. Unset, byte-inert.
    phys, physics_refusal_cert = _generate_physics(prompt, job_dir, _phase)
    if phys is not None:
        return phys
    # B3: physics declined a DECLARED-load request -> its claimable:false certificate rides the
    # agentic result below so the refusal is visible ("attempted, could not prove, here's why").

    # Standard-parts lane (ADR-0034): a CANONICAL standard part (NEMA mount, Pi case, extrusion
    # bracket, ...) routes to the measured-~100% deterministic LayoutSpec pipeline -- the same part
    # every time, ~5x cheaper and faster than the ~25-50% stochastic author. Conservative keyword
    # gate; on refusal/failure/non-pass the lane returns None and the agentic engine runs as today
    # (strictly-no-worse semantics). The attempt's classified outcome (ADR-0035) rides on the
    # final result either way, so lane health is observable downstream.
    lane: StandardLaneOutcome | None = None
    if standard_match(prompt):
        std, lane = _generate_standard(prompt, job_dir, _phase)
        if std is not None:
            return std

    def _with_lane(res: GenerateResult) -> GenerateResult:
        return replace(res, lane=lane) if lane is not None else res

    effort = _QUALITY.get(quality, _QUALITY["standard"])
    fab = FabricationMode.GENERAL if fabrication_mode == "general" else FabricationMode.PRINT_3D

    try:
        result = produce_part(
            prompt,
            job_dir,
            material=_DEFAULT_MATERIAL,
            fabrication_mode=fab,
            n_drafts=int(effort["n_drafts"]),
            max_iterations=int(effort["max_iterations"]),
            budget_usd=float(effort["budget_usd"]),
            on_phase=on_phase,
            extra_context=brief_context,
            client=client,
        )
    except Exception as exc:  # produce_part is meant to be safe, but never leak a stack trace
        return _with_lane(
            GenerateResult(
                status="error",
                error=GenerateError(stage="engine", message=f"{type(exc).__name__}: {exc}"),
            )
        )

    # No buildable geometry produced.
    if result.loop.final_stl is None or result.disposition == "no_build":
        # An INFEASIBLE spec is a first-class refusal WITH the arithmetic (founder ruling,
        # engine deep review 2026-07-04): the user hears exactly which numbers contradict,
        # not a generic build failure. Rides the existing refused contract -- no schema change.
        if result.disposition == "refused_infeasible" and result.loop.refusal_reasons:
            return _with_lane(
                GenerateResult(
                    status="refused",
                    refused=GenerateRefused(reason=" ".join(result.loop.refusal_reasons)),
                )
            )
        if result.loop.brief is None:
            return _with_lane(
                GenerateResult(
                    status="refused",
                    refused=GenerateRefused(
                        reason=(
                            "I couldn't ground this into a concrete, buildable part. Try "
                            "describing a mechanical part with rough dimensions and any "
                            "standard hardware it uses."
                        )
                    ),
                )
            )
        return _with_lane(
            GenerateResult(
                status="error",
                error=GenerateError(
                    stage="build",
                    message=(
                        f"No part was built within the limits (stopped: {result.loop.stopped_reason})."
                    ),
                ),
            )
        )

    # A real part exists. Make the files servable; a missing GLB is a hard error (the viewer needs it).
    _phase("export", "active")
    if not _finalize_files(result, job_dir):
        return _with_lane(
            GenerateResult(
                status="error",
                error=GenerateError(
                    stage="export", message="The part built but could not be exported for viewing."
                ),
            )
        )

    gen_path = (result.loop.brief.target if result.loop.brief else None) or "agentic"
    payload = GenerateOk(
        prompt=prompt,
        gen_path=gen_path,
        cost_usd=round(result.total_cost_usd, 4),
        validation=_validation_view(result),
        kit=_synthesize_kit(result, _DEFAULT_MATERIAL),
        files=Files(glb=GLB_NAME, stl=STL_NAME),  # rewritten to files/<job>/<name> by the caller
        # Surface the mode so the frontend can label a GENERAL part honestly (validated for general
        # fabrication, NOT claimed 3D-printable -- the no-false-PASS rule from fabrication.py).
        fabrication_mode=fab.value,
        # The acceptance judge's verdict, finally PROJECTED into the contract (ADR-0035 B8): a
        # disposition of "rejected" means the Opus judge did not accept the part against the brief.
        # Serve-but-stamp semantics -- the server stamps the payload; the file still ships for
        # inspection, never silently dropped, never silently presented as accepted.
        disposition=result.disposition,
        # The honest confidence + fitness verdict (ADR-0041), computed from the earned
        # dimension tiers + the gates + the judge -- never LLM-composed.
        confidence=_agentic_confidence(result, fab.value),
        # The invention-lab package envelope (ADR-0048), additive + flag-gated: None unless
        # FORGE_INVENTION is on, so the /generate contract is byte-identical by default.
        invention_package=_invention_view_for_result(result, prompt, _DEFAULT_MATERIAL, fab),
        # The certificate rides EVERY agentic result (ADR-0109): the B3 visible refusal when a load
        # was DECLARED but no physics author could prove it (claimable:false + the failing check);
        # ELSE a MANUFACTURABILITY certificate mapped from the full validator's per-check verdicts --
        # the trustworthy verdict on the default path, not only the narrow physics lane.
        certificate=physics_refusal_cert or _agentic_certificate(result, fab, prompt),
    )
    return _with_lane(GenerateResult(status="ok", ok=payload))


__all__ = [
    "GLB_NAME",
    "STL_NAME",
    "GenerateResult",
    "StandardLaneOutcome",
    "generate_part",
]
