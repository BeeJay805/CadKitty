"""Top-level agentic entry: run the loop, run the acceptance gate, write artifacts.

``produce_part(prompt, out_dir)`` is the Wave-1 deliverable: a plain-English request ->
a reproducible build123d program + STL / STEP / GLB + the brief + an iteration log + a
confidence/validation report, with an HONEST disposition that never claims a part is valid
when it is not:

  - shipped              : the object AND geometrically valid (all tiers pass)
  - shipped_with_caveats : the object AND a coherent watertight solid, but a tier flagged it
                           (e.g. a sub-mm edge) -- usable as a model, not certified print-ready
  - rejected             : the acceptance judge did not accept it, or it is not a coherent solid
  - no_build             : nothing built within the caps
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from forge_mind.agentic.brief import DesignBrief, research_brief
from forge_mind.agentic.feasibility import spec_conflicts
from forge_mind.agentic.functional import FunctionalResult, produce_functional_part
from forge_mind.agentic.judge import AcceptanceVerdict, acceptance_judge
from forge_mind.agentic.loop import LoopResult, default_max_iterations, run_agentic_pipeline
from forge_mind.agentic.provenance import DimensionProvenance, derive_provenance
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.provider import LLMClientLike, make_client
from forge_mind.materials import Material, material_spec, resolve_material
from forge_mind.viz.glb import to_gltf_yup
from forge_mind.viz.render import render_part_to_png


@dataclass
class PartResult:
    """The finished product: the loop outcome, the acceptance verdict, and the artifacts."""

    out_dir: Path
    loop: LoopResult
    verdict: AcceptanceVerdict | None
    disposition: str
    total_cost_usd: float
    artifacts: dict[str, str]
    # Per-dimension provenance tiers, EARNED in code from the brief + intake facts
    # (ADR-0040). Empty when no brief was researched.
    provenance: list[DimensionProvenance] = field(default_factory=list)


def _disposition(loop: LoopResult, verdict: AcceptanceVerdict | None, judge_failed: bool) -> str:
    if loop.final_stl is None or loop.validation is None:
        return "no_build"
    # The deterministic/vision loop critic is a shipping gate. A later object-identity
    # judge cannot silently bypass its unresolved veto; a measured refutation updates
    # critic_approved inside the loop before this function is reached.
    if not loop.critic_approved:
        return "rejected"
    if verdict is not None:
        if not verdict.is_the_object:
            return "rejected"
        if loop.geometry_passed:
            return "shipped"
        return "shipped_with_caveats" if loop.validation.hard_ok else "rejected"
    # The judge returned no verdict. If it FAILED (a transient API error, not a "no") on a
    # coherent, critic-approved part, rejecting it is a no-false-FAIL hazard -- the part is good,
    # the final gate just could not run -> caveats (never a clean "shipped" without the judge).
    if judge_failed and loop.validation.hard_ok and loop.critic_approved:
        return "shipped_with_caveats"
    return "rejected"


def _maybe_emit_grasp_tab(
    loop: LoopResult, disposition: str, out_dir: Path, material: Material
) -> str:
    """CELL LANE ONLY (ADR-0012 §6): fuse the standardized grasp tab onto a SHIPPED part before
    export, updating ``loop.final_stl``/``final_step`` in place. Runs only behind
    ``FORGE_CELL_PROFILE`` -- the website paths (/generate, /lab, /app/solve) leave the flag off, so
    their output is BYTE-IDENTICAL (the two-sided test). A part with no valid, non-colliding tab
    placement (or whose tab breaks re-validation / the mass cap) is REFUSED, never tabbed unsafely."""
    from forge_mind.slicing.slicer import cell_lane_enabled

    if not cell_lane_enabled():
        return disposition
    if disposition not in ("shipped", "shipped_with_caveats"):
        return disposition
    if loop.final_step is None or loop.final_stl is None or loop.brief is None:
        return disposition

    from forge_mind.agentic.grasp_tab import emit_grasp_tab

    tab = emit_grasp_tab(
        loop.final_step, loop.final_stl, out_dir / "work", material=material, brief=loop.brief
    )
    if tab.ok and tab.stl_path is not None and tab.step_path is not None:
        loop.final_stl = tab.stl_path
        loop.final_step = tab.step_path
        return disposition
    loop.stopped_reason = "grasp_tab_refused"
    loop.refusal_reasons = [tab.reason or "no valid grasp-tab placement"]
    return "refused_grasp_tab"


def _export_glb(stl_path: Path, glb_path: Path, material: Material | None = None) -> bool:
    """Write a GLB from the STL, carrying the material's colour so a web viewer shows it in its
    real appearance (a PBR material when possible, vertex colour as a robust fallback)."""
    try:
        import trimesh

        mesh: Any = trimesh.load(str(stl_path), force="mesh")
        if material is not None:
            a = material.appearance
            rgba = [round(255 * c) for c in a.base_color] + [round(255 * a.opacity)]
            try:
                from trimesh.visual.material import PBRMaterial

                mesh.visual = trimesh.visual.TextureVisuals(  # type: ignore[no-untyped-call]
                    material=PBRMaterial(  # type: ignore[no-untyped-call]
                        baseColorFactor=rgba,
                        metallicFactor=float(a.metallic),
                        roughnessFactor=float(a.roughness),
                    )
                )
            except Exception:
                mesh.visual.vertex_colors = rgba
        to_gltf_yup(mesh)  # Z-up kernel mesh -> glTF Y-up (viewer only; the STL stays Z-up)
        mesh.export(str(glb_path))
        return True
    except Exception:
        return False


def _export_3mf(step_path: Path, threemf_path: Path) -> bool:
    """Write a 3MF (the modern print-ready exchange format) from the STEP solid. Best-effort."""
    try:
        from build123d import Mesher, import_step

        mesher = Mesher()
        mesher.add_shape(import_step(str(step_path)))
        mesher.write(str(threemf_path))
        return True
    except Exception:
        return False


def _material_section(loop: LoopResult, material: Material) -> dict[str, Any]:
    """The material spec for the report: the validator's measured-volume spec when available
    (carries the wall verdict), else computed from the final mesh volume."""
    if loop.validation is not None and loop.validation.material_report is not None:
        mr = loop.validation.material_report
        return {**mr.spec, "mass_g": round(mr.mass_g, 2), "cost_usd": round(mr.cost_usd, 3)}
    volume = loop.validation.volume_mm3 if loop.validation is not None else 0.0
    return material_spec(material, volume)


def _report_dict(
    loop: LoopResult,
    verdict: AcceptanceVerdict | None,
    disposition: str,
    total_cost: float,
    material: Material,
    judge_error: str | None = None,
    provenance: list[DimensionProvenance] | None = None,
) -> dict[str, Any]:
    validation: dict[str, Any] | None = None
    if loop.validation is not None:
        validation = {
            "passed": loop.validation.passed,
            "hard_ok": loop.validation.hard_ok,
            "body_count": loop.validation.body_count,
            "watertight": loop.validation.watertight,
            "volume_mm3": round(loop.validation.volume_mm3, 2),
            "issues": loop.validation.issues,
            "failed_checks": (
                loop.validation.report.failed_check_ids() if loop.validation.report else []
            ),
        }
    acceptance: dict[str, Any] | None = None
    if verdict is not None:
        acceptance = {
            "is_the_object": verdict.is_the_object,
            "confidence": verdict.confidence,
            "present_features": verdict.present_features,
            "missing_or_wrong": verdict.missing_or_wrong,
            "reason": verdict.reason,
        }
    return {
        "prompt": loop.prompt,
        "object_name": loop.brief.object_name if loop.brief else None,
        "target": loop.brief.target if loop.brief else None,
        "disposition": disposition,
        "geometry_passed": loop.geometry_passed,
        "critic_approved": loop.critic_approved,
        "acceptance": acceptance,
        "judge_error": judge_error,
        # Per-dimension provenance (ADR-0040): tier + mechanism + citation per feature.
        "provenance": [p.model_dump(mode="json") for p in (provenance or [])],
        "material": _material_section(loop, material),
        "validation": validation,
        "stopped_reason": loop.stopped_reason,
        # The SAY-WHY surface for a pre-authoring refusal (infeasible_spec): the stated
        # arithmetic conflicts, verbatim. Empty for every non-refused run.
        "refusal_reasons": loop.refusal_reasons,
        "drafts_tried": loop.drafts_tried,
        "drafts_built": loop.drafts_built,
        "total_cost_usd": round(total_cost, 4),
        "total_duration_s": round(loop.total_duration_s, 1),
        "iterations": [asdict(it) for it in loop.iterations],
    }


def _copy(src: Path | None, dst: Path) -> str | None:
    if src is not None and src.exists():
        dst.write_bytes(src.read_bytes())
        return str(dst)
    return None


def _render_final(out_dir: Path, loop: LoopResult, material: Material) -> str | None:
    """Render the final STL artifact in the chosen material (colour + CAD edges). Falls back to
    the loop's last render if a fresh render fails, so we never lose the preview."""
    title = loop.brief.object_name if loop.brief else None
    try:
        render_part_to_png(
            out_dir / "part.stl",
            out_dir / "part.png",
            title=title,
            caption=loop.prompt,
        )
        return str(out_dir / "part.png")
    except Exception:
        return _copy(loop.final_render, out_dir / "part.png")


def _write_artifacts(
    loop: LoopResult,
    verdict: AcceptanceVerdict | None,
    disposition: str,
    out_dir: Path,
    cost: float,
    material: Material,
    judge_error: str | None = None,
    provenance: list[DimensionProvenance] | None = None,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if loop.final_code:
        (out_dir / "program.py").write_text(loop.final_code, encoding="utf-8")
        artifacts["program"] = str(out_dir / "program.py")
    if (stl := _copy(loop.final_stl, out_dir / "part.stl")) is not None:
        artifacts["stl"] = stl
        if _export_glb(out_dir / "part.stl", out_dir / "part.glb", material=material):
            artifacts["glb"] = str(out_dir / "part.glb")
    if (step := _copy(loop.final_step, out_dir / "part.step")) is not None:
        artifacts["step"] = step
        if _export_3mf(out_dir / "part.step", out_dir / "part.3mf"):
            artifacts["3mf"] = str(out_dir / "part.3mf")
    render = _render_final(out_dir, loop, material) if "stl" in artifacts else None
    if render is None:
        render = _copy(loop.final_render, out_dir / "part.png")
    if render is not None:
        artifacts["render"] = render
    if loop.brief is not None:
        (out_dir / "brief.json").write_text(
            json.dumps(loop.brief.model_dump(), indent=2), encoding="utf-8"
        )
        artifacts["brief"] = str(out_dir / "brief.json")
    (out_dir / "report.json").write_text(
        json.dumps(
            _report_dict(loop, verdict, disposition, cost, material, judge_error, provenance),
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts["report"] = str(out_dir / "report.json")
    return artifacts


def _functional_to_part_result(
    fres: FunctionalResult,
    prompt: str,
    total_cost: float,
    provenance: list[DimensionProvenance] | None = None,
) -> PartResult:
    """Adapt a multi-part :class:`FunctionalResult` to the common :class:`PartResult` surface so
    every caller (the benchmark, artifact writers) reads ONE shape regardless of route. The
    synthesized loop carries the AGGREGATE instrumentation: geometry_passed iff every part passed,
    the concatenated per-part iteration logs, and summed drafts / duration."""
    parts = fres.parts
    iterations = [it for _name, lr in parts for it in lr.iterations]
    geometry_passed = bool(parts) and all(lr.geometry_passed for _name, lr in parts)
    loop = LoopResult(
        prompt=prompt,
        brief=fres.whole_brief,
        final_code=None,
        final_stl=fres.assembled_stl,
        final_step=fres.assembled_step,
        final_render=fres.assembled_render,
        validation=None,
        critique=None,
        geometry_passed=geometry_passed,
        critic_approved=bool(fres.verdict and fres.verdict.is_the_object),
        stopped_reason="functional",
        iterations=iterations,
        total_cost_usd=total_cost,
        total_duration_s=sum(lr.total_duration_s for _name, lr in parts),
        drafts_tried=sum(lr.drafts_tried for _name, lr in parts),
        drafts_built=sum(lr.drafts_built for _name, lr in parts),
    )
    # Artifacts list ONLY files that exist: the decompose-failure path returns before the
    # report is written, and an artifact entry pointing at nothing misleads every reader.
    artifacts: dict[str, str] = {}
    report_path = fres.out_dir / "functional_report.json"
    if report_path.exists():
        artifacts["functional_report"] = str(report_path)
    if fres.assembled_stl is not None and fres.assembled_stl.exists():
        artifacts["stl"] = str(fres.assembled_stl)
    if fres.assembled_step is not None and fres.assembled_step.exists():
        artifacts["step"] = str(fres.assembled_step)
    if fres.assembled_render is not None and fres.assembled_render.exists():
        artifacts["render"] = str(fres.assembled_render)
    # The assembly's own GLB (the ck_joint articulated export when the mechanism assembled,
    # the flat coloured one otherwise). Without this key the API rebuilt a rigid GLB from the
    # concatenated STL and the viewer could never articulate an agentic mechanism.
    if fres.assembled_glb is not None and fres.assembled_glb.exists():
        artifacts["glb"] = str(fres.assembled_glb)
    return PartResult(
        out_dir=fres.out_dir,
        loop=loop,
        verdict=fres.verdict,
        disposition=fres.disposition,
        total_cost_usd=total_cost,
        artifacts=artifacts,
        provenance=provenance or [],
    )


def produce_part(
    prompt: str,
    out_dir: Path,
    *,
    max_iterations: int | None = None,
    budget_usd: float = 2.0,
    n_drafts: int = 1,
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
    material: str = "PLA",
    brief_model: str | None = None,
    author_model: str | None = None,
    critic_model: str | None = None,
    judge_model: str | None = None,
    client: LLMClientLike | None = None,
    on_phase: Callable[[str, str], None] | None = None,
    extra_context: str | None = None,
) -> PartResult:
    """Run a part end-to-end and write all artifacts. Never raises on a stage failure.

    Researches the brief ONCE, then routes on ``brief.target``: a 'functional' object (a tool /
    mechanism that is genuinely several parts -- a hammer is a head + a handle) goes to
    ``produce_functional_part`` (each part its own single body, assembled at a shared origin frame,
    no fusion-variance); a 'cosmetic' monolith goes to the proven single-body loop. Both return a
    :class:`PartResult`.

    ``extra_context`` (ADR-0036) is typed reference material injected into the brief research --
    the guided intake's STATED-FACTS block (``intake.brief.brief_to_context``) or, later,
    request-time researched component facts. The researcher copies these instead of re-guessing
    them from the prose prompt. None (every legacy caller) is byte-identical behavior.
    """
    if max_iterations is None:
        max_iterations = default_max_iterations()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Role-based clients (ADR-0113, local-first roster): with no explicitly-injected client, each role
    # picks its own provider -- planner (brief/decompose) on Fable(API), vision (acceptance judge) on
    # the local VLM, and the loop (coder/repair/critic) self-routes internally. An explicit ``client``
    # (tests, callers) overrides every role, so those paths are byte-identical.
    planner_client = client or make_client(role="planner")
    vision_client = client or make_client(role="vision")
    planner_model = brief_model or os.environ.get("FORGE_PLANNER_MODEL") or None
    mat = resolve_material(material)  # the chosen material drives validation, mass/cost, rendering

    def _phase(key: str, status: str) -> None:
        if on_phase is not None:
            on_phase(key, status)

    # Research once to choose the route. The brief is threaded into the chosen stage so it is not
    # re-researched; we pay for it here, the stage's cost excludes it. "plan" = this research call;
    # "cad" = the author + build + validate work that follows on the chosen route.
    _phase("plan", "active")
    brief: DesignBrief | None = None
    brief_cost = 0.0
    try:
        brief, brief_usage = research_brief(
            prompt, client=planner_client, model=planner_model, extra_context=extra_context
        )
        brief_cost = brief_usage.cost_usd
    except Exception:
        brief = None
    # Dimension provenance is EARNED here, once, from the brief + the intake facts
    # (ADR-0040): registry resolutions -> verified/sourced, intake matches ->
    # sourced/stated, everything else estimated. Deterministic; rides both routes.
    provenance = derive_provenance(brief, extra_context) if brief is not None else []
    _phase("plan", "done")

    # Spec-feasibility pre-check (founder ruling, engine deep review 2026-07-04): a spec whose
    # numbers contradict each other (the measured case: 8 holes on a 90 mm circle through an
    # 80 mm-wide plate -- unfittable at ANY rotation) is REFUSED here with the arithmetic
    # stated, instead of burning the full iteration budget against an impossibility. The
    # check is deterministic and fires only on a rotation-proof contradiction.
    conflicts = spec_conflicts(brief) if brief is not None else []
    if conflicts:
        loop = LoopResult(
            prompt=prompt,
            brief=brief,
            final_code=None,
            final_stl=None,
            final_step=None,
            final_render=None,
            validation=None,
            critique=None,
            geometry_passed=False,
            critic_approved=False,
            stopped_reason="infeasible_spec",
            total_cost_usd=0.0,
            refusal_reasons=conflicts,
        )
        artifacts = _write_artifacts(
            loop, None, "refused_infeasible", out_dir, brief_cost, mat, None, provenance=provenance
        )
        _phase("cad", "done")
        return PartResult(
            out_dir=out_dir,
            loop=loop,
            verdict=None,
            disposition="refused_infeasible",
            total_cost_usd=brief_cost,
            artifacts=artifacts,
            provenance=provenance,
        )

    _phase("cad", "active")

    if brief is not None and brief.target == "functional":
        fres = produce_functional_part(
            prompt,
            out_dir,
            max_iterations=max_iterations,
            budget_usd=budget_usd,
            n_drafts=n_drafts,
            material=mat,
            client=client,
            brief=brief,
            brief_model=brief_model,
            author_model=author_model,
            critic_model=critic_model,
            judge_model=judge_model,
        )
        functional_result = _functional_to_part_result(
            fres, prompt, brief_cost + fres.total_cost_usd, provenance=provenance
        )
        _phase("cad", "done")
        return functional_result

    # Cosmetic monolith: the proven single-body research -> author -> validate -> critique loop.
    loop = run_agentic_pipeline(
        prompt,
        out_dir / "work",
        max_iterations=max_iterations,
        budget_usd=budget_usd,
        n_drafts=n_drafts,
        fabrication_mode=fabrication_mode,
        material=mat,
        brief_model=brief_model,
        author_model=author_model,
        critic_model=critic_model,
        brief=brief,
        client=client,
    )

    verdict: AcceptanceVerdict | None = None
    judge_error: str | None = None
    total_cost = brief_cost + loop.total_cost_usd
    if loop.final_render is not None and loop.brief is not None:
        try:
            verdict, usage = acceptance_judge(
                loop.final_render, loop.brief, client=vision_client, model=judge_model
            )
            total_cost += usage.cost_usd
        except Exception as exc:
            judge_error = f"{type(exc).__name__}: {exc}"
            verdict = None

    disposition = _disposition(loop, verdict, judge_error is not None)
    disposition = _maybe_emit_grasp_tab(loop, disposition, out_dir, mat)
    artifacts = _write_artifacts(
        loop, verdict, disposition, out_dir, total_cost, mat, judge_error, provenance=provenance
    )
    _phase("cad", "done")
    return PartResult(
        out_dir=out_dir,
        loop=loop,
        verdict=verdict,
        disposition=disposition,
        total_cost_usd=total_cost,
        artifacts=artifacts,
        provenance=provenance,
    )


__all__ = ["PartResult", "produce_part"]
