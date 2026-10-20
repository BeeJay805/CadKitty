"""The dispatch loop (T2 / Fable gap a): route -> author -> uniform gate -> monotone fallback.

ADR-0098 SS4 made executable. ``route()`` was called once, as an assert, and no author
implemented the C2 ``Author`` type -- the organs existed with no circulatory system. This module
is that system: three thin adapters make every engine speak ``(RequirementSpec, Path) ->
AuthorResult``, and ``author_part`` runs the routed author, gates its artifact with
``verify_full``, and falls through the monotone cost ladder (ANALYTIC -> TOPOPT -> AGENTIC)
collecting every attempt's per-check gate report. Authors propose; the gate disposes; the
agentic path is the FLOOR -- it never falls through, its honest gate ships with the result.

Load-case honesty (the T5 gap, named not hidden): when the spec's case leaves ``orient`` open
(the extractor's common output), the gate must still measure the case the artifact was DESIGNED
for. That finalization is a deterministic table over provenance (``_gate_spec``): topopt parts
measure the in-plane transverse case (the plane SIMP designed), everything else the canonical
link case (rows 73-77's convention) -- never an LLM judgment call, echoed in human units on the
gate report. A spec that fully pins ``orient`` is measured verbatim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from forge_mind.structures.author import Author, AuthorResult
from forge_mind.structures.autotune import tune_volfrac_to_bound
from forge_mind.structures.hardware import place_component
from forge_mind.structures.loadcase import G_M_S2, LoadCase, canonical_link_case
from forge_mind.structures.router import AGENTIC, ANALYTIC, TOPOPT, route
from forge_mind.structures.sections import author_isection_link
from forge_mind.structures.spec import RequirementSpec
from forge_mind.structures.topopt import Domain
from forge_mind.structures.verify import verify_full

_LADDER = (ANALYTIC, TOPOPT, AGENTIC)  # monotone in cost; fallback only ever moves right


def analytic_author(spec: RequirementSpec, work_dir: Path) -> AuthorResult:
    """Closed-form I-section link author (wraps ``sections.author_isection_link``).

    Envelope (L, W, H) -> (length, hy, wx) verbatim; the spec's bound is the sizing target
    (the section is solved TO the bound with FEA margin); payload comes from the spec's own
    case. Preconditions fail SOFT (stl=None) so ``author_part`` falls through, never crashes."""
    if spec.load is None or spec.deflection_bound_mm is None:
        return AuthorResult(
            provenance="analytic_isection",
            stl=None,
            step=None,
            self_reported={"error": "analytic author needs a load case AND a deflection bound"},
        )
    length, hy, wx = spec.envelope_mm
    record = author_isection_link(
        ref_mass_g=spec.ref_mass_g if spec.ref_mass_g is not None else float("inf"),
        ref_deflection_mm=spec.deflection_bound_mm,
        length=length,
        hy=hy,
        wx=wx,
        payload_kg=spec.load.force_n / G_M_S2,
        material_id=spec.material_id,
        out_dir=work_dir,
    )
    if spec.ref_mass_g is None:
        # no declared reference: the inf sentinel sized nothing, so strip its artifacts from
        # the advisory record rather than report a vacuous "lighter than infinity"
        record = {
            **record,
            "lighter_than_reference": None,
            "mass_vs_reference_pct": None,
            "reference": None,
        }
    stl, step = record.get("stl"), record.get("step")
    return AuthorResult(
        provenance="analytic_isection",
        stl=Path(stl) if stl else None,
        step=Path(step) if step else None,
        self_reported=record,
    )


def topopt_author(
    spec: RequirementSpec,
    work_dir: Path,
    *,
    nelx: int | None = None,
    nely: int | None = None,
    rmin: float = 2.4,
    extrude_thresh: float = 0.45,
    open_radius_cells: int = 1,
    max_fea_evals: int = 6,
    simp_max_iter: int = 60,
) -> AuthorResult:
    """Generative SIMP author (wraps ``autotune.tune_volfrac_to_bound`` around a domain carved
    by ``hardware.place_component``).

    The SIMP plane is envelope L x H, extruded W (the generative_proof convention). Root edge
    (x=0) is fixed; the load enters at the LAST declared component's mount (the proven drone-arm
    rule) or, for a bare span, at the tip -- positional load cases are the T5 gap, so this
    convention is recorded in the self-report rather than silently assumed. Knob defaults are
    the row-76 CLAIMABLE recipe (rmin 2.4, thresh 0.45, opening 1)."""
    if spec.load is None or spec.deflection_bound_mm is None:
        return AuthorResult(
            provenance="topopt2d_extrude",
            stl=None,
            step=None,
            self_reported={"error": "topopt author needs a load case AND a deflection bound"},
        )
    length, depth, height = spec.envelope_mm
    nx = nelx if nelx is not None else max(24, min(160, round(length / 1.2)))
    ny = nely if nely is not None else max(12, min(160, round(height / 1.2)))
    cell = (length / nx, height / ny)
    force_n = spec.load.force_n
    regions: dict[str, Any] = {}

    def build(vf: float) -> Domain:
        dom = Domain(nelx=nx, nely=ny, volfrac=vf, rmin=rmin)
        dom.fixed_dofs = [dom.dof(0, iy, c) for iy in range(dom.nely + 1) for c in (0, 1)]
        load_at: tuple[float, float] | None = None
        for i, comp in enumerate(spec.components):
            sizes = place_component(dom, comp, cell_mm=cell)
            regions[f"{comp.kind}_{i}"] = sizes
            load_at = comp.at_mm  # the load enters at the component mount (proof convention)
        if load_at is None:
            load_at = (length, height / 2.0)  # bare span: tip load, mid-height
        ix = min(dom.nelx, int(load_at[0] / cell[0]))
        iy = min(dom.nely, int(load_at[1] / cell[1]))
        dom.load_dofs = [dom.dof(ix, iy, 1)]
        dom.load_vals = [-force_n]
        return dom

    load = spec.load
    case: LoadCase | Any = load if load.orient is not None else (lambda m: _inplane_case(load, m))
    outcome = tune_volfrac_to_bound(
        build,
        load_case=case,
        bound_mm=spec.deflection_bound_mm,
        out_dir=work_dir,
        scale_xy=cell,
        height_mm=depth,
        material_id=spec.material_id,
        max_fea_evals=max_fea_evals,
        simp_max_iter=simp_max_iter,
        extrude_thresh=extrude_thresh,
        open_radius_cells=open_radius_cells,
    )
    return AuthorResult(
        provenance="topopt2d_extrude",
        stl=outcome.stl,
        step=outcome.step,
        self_reported={
            "tune": outcome.to_dict(),
            "regions": regions,
            "grid": {"nelx": nx, "nely": ny, "rmin": rmin},
            "load_at_convention": "component mount" if spec.components else "tip",
        },
    )


def agentic_author(spec: RequirementSpec, work_dir: Path) -> AuthorResult:
    """The LLM path as an Author (lazy import: the $0 physics flows never load the LLM stack).

    The spec renders to a minimal prompt (envelope, material, human-units load echo, interfaces,
    components); artifacts come from the LoopResult; the loop's own validation rides along as
    the advisory self-report. Acceptance authority stays with the uniform gate."""
    from forge_mind.agentic.loop import run_agentic_pipeline

    lr = run_agentic_pipeline(render_prompt(spec), work_dir, n_drafts=1)
    return AuthorResult(
        provenance="agentic",
        stl=lr.final_stl,
        step=lr.final_step,
        self_reported={
            "stopped_reason": lr.stopped_reason,
            "geometry_passed": bool(lr.geometry_passed),
            "critic_approved": bool(lr.critic_approved),
            "total_cost_usd": lr.total_cost_usd,
        },
    )


def render_prompt(spec: RequirementSpec) -> str:
    """The minimal deterministic spec -> prompt renderer for the agentic floor."""
    from forge_mind.structures.loadcase import describe_load_case

    length, width, height = spec.envelope_mm
    lines = [
        "Design a single 3D-printable structural part.",
        f"Envelope: {length:.0f} x {width:.0f} x {height:.0f} mm. Material: {spec.material_id}.",
    ]
    if spec.load is not None:
        lines.append(f"It must carry this load: {describe_load_case(spec.load)}.")
    if spec.deflection_bound_mm is not None:
        lines.append(f"Maximum allowed deflection under that load: {spec.deflection_bound_mm} mm.")
    for key, value in spec.interfaces.items():
        lines.append(f"Stated interface dimension: {key} = {value} mm.")
    for comp in spec.components:
        lines.append(
            f"Leave an open pocket + mounts for a real {comp.kind} with its body reference at "
            f"({comp.at_mm[0]:.0f}, {comp.at_mm[1]:.0f}) mm, {comp.clearance_mm} mm clearance."
        )
    return " ".join(lines)


def _inplane_case(load: LoadCase, mesh: Any) -> LoadCase:
    """The topopt measurement case when the spec leaves orientation open: transverse IN the
    SIMP plane (the plane the structure was designed in), finalized from the winner's geometry
    -- the generative_proof convention, verbatim."""
    c = canonical_link_case(mesh, payload_kg=load.force_n / G_M_S2)
    strong = (0.0, 1.0, 0.0) if c.load_dir == (1.0, 0.0, 0.0) else (1.0, 0.0, 0.0)
    return LoadCase(
        name=load.name,
        force_n=c.force_n,
        load_dir=strong,
        e_mpa=load.e_mpa,
        nu=load.nu,
        orient=c.orient,
    )


def _gate_spec(spec: RequirementSpec, result: AuthorResult) -> RequirementSpec:
    """The spec the gate measures under. Verbatim when the spec pins ``orient``; otherwise the
    case is finalized deterministically from the ARTIFACT by provenance (topopt -> in-plane
    transverse, its design plane; everything else -> the canonical link case, rows 73-77). The
    gate echoes the finalized case in human units, so the certificate shows what was measured."""
    if spec.load is None or spec.load.orient is not None or result.stl is None:
        return spec
    import trimesh

    try:
        mesh: Any = trimesh.load(str(result.stl), force="mesh")
        if result.provenance.startswith("topopt"):
            case = _inplane_case(spec.load, mesh)
        else:
            c = canonical_link_case(mesh, payload_kg=spec.load.force_n / G_M_S2)
            case = LoadCase(
                name=spec.load.name,
                force_n=c.force_n,
                load_dir=c.load_dir,
                e_mpa=spec.load.e_mpa,
                nu=spec.load.nu,
                orient=c.orient,
            )
        return replace(spec, load=case)
    except Exception:
        return spec  # unreadable artifact: gate under the spec verbatim (verify fails it anyway)


def refusing_author(provenance: str, reason: str) -> Author:
    """An Author that honestly refuses (stl=None). Callers that must not fall through to a
    path (the $0 physics proof must never silently burn LLM calls) inject this as that rung."""

    def refuse(_spec: RequirementSpec, _work_dir: Path) -> AuthorResult:
        return AuthorResult(
            provenance=provenance, stl=None, step=None, self_reported={"refused": reason}
        )

    return refuse


_AUTHORS: dict[str, Author] = {
    ANALYTIC: analytic_author,
    TOPOPT: topopt_author,
    AGENTIC: agentic_author,
}


def author_part(
    spec: RequirementSpec,
    work_dir: Path,
    *,
    authors: Mapping[str, Author] | None = None,
    part_name: str = "structural part",
) -> dict[str, Any]:
    """The dispatch loop: route -> author -> ``verify_full`` -> monotone fallback.

    Runs the routed author in ``work_dir/attempt_<rung>/``, gates the artifact against the
    (deterministically finalized) spec, and while the gate is not claimable AND the rung is not
    the agentic floor, falls through the ladder collecting every attempt's per-check report.
    An author that RAISES degrades to a fail-soft empty result (the error in its self-report)
    and falls through like any other refusal -- one bad engine never takes down the loop.

    Returns ``{"chosen": <routed rung>, "attempts": [{"rung", "provenance", "gate",
    "self_reported"}...], "final": AuthorResult}`` -- ``final`` is the last attempt (the
    claimable one, or the floor's honest best effort); its gate is ``attempts[-1]["gate"]``."""
    table = dict(_AUTHORS)
    if authors:
        table.update(authors)
    chosen = route(spec)
    attempts: list[dict[str, Any]] = []
    final: AuthorResult | None = None
    for rung in _LADDER[_LADDER.index(chosen) :]:
        sub = work_dir / f"attempt_{rung}"
        sub.mkdir(parents=True, exist_ok=True)
        try:
            result = table[rung](spec, sub)
        except Exception as exc:  # fail-soft: a crashing author is a refusal, not a crash
            result = AuthorResult(
                provenance=rung,
                stl=None,
                step=None,
                self_reported={"error": f"{type(exc).__name__}: {exc}"},
            )
        gate = verify_full(result, _gate_spec(spec, result), part_name=part_name)
        # self_reported rides EVERY attempt (row-78 lesson: a failed rung with an invisible
        # error/tune-history is undiagnosable from the report -- structured feedback or bust).
        attempts.append(
            {
                "rung": rung,
                "provenance": result.provenance,
                "gate": gate,
                "self_reported": result.self_reported,
            }
        )
        final = result
        if gate["claimable"] or rung == AGENTIC:
            break
    assert final is not None  # the ladder always contains at least the routed rung
    return {"chosen": chosen, "attempts": attempts, "final": final}


__all__ = [
    "agentic_author",
    "analytic_author",
    "author_part",
    "refusing_author",
    "render_prompt",
    "topopt_author",
]
