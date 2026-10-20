"""CLI entry for forge-mind."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from forge_mind import __version__

app = typer.Typer(
    name="forge-mind",
    help="AI-native CAD pipeline (intent -> blueprint -> PartDSL -> compile -> validate -> revise).",
    no_args_is_help=True,
)


def _echo_kit(kit: object) -> None:
    """Print the buildable kit (BOM + make-it) for `forge-mind run`. Typed loosely to keep the
    heavy `forge_mind.kit` import lazy inside `run`."""
    from forge_mind.kit import Kit

    assert isinstance(kit, Kit)
    typer.echo("\nBILL OF MATERIALS (what to buy):")
    if kit.bom.lines:
        for line in kit.bom.lines:
            typer.echo(f"  - {line.render()}   [{line.source}]")
    else:
        typer.echo("  (no registry-backed hardware)")
    for flagged in kit.bom.unmapped:
        typer.echo(f"  ! unmapped (not in the registry): {flagged}")

    typer.echo("\nMAKE IT (how to fabricate, advisory):")
    typer.echo(
        f"  material: {kit.instructions.material}   layer: {kit.instructions.layer_height_mm:g} mm"
    )
    typer.echo(f"  walls: {kit.instructions.wall_note}")
    for i, step in enumerate(kit.instructions.steps, 1):
        typer.echo(f"  {i}. {step}")
    if kit.instructions.insert_install_note:
        typer.echo(f"  note: {kit.instructions.insert_install_note}")


@app.command()
def version() -> None:
    """Print the installed forge-mind version."""
    typer.echo(f"forge-mind {__version__}")


@app.command()
def status() -> None:
    """Show what's implemented."""
    typer.echo(
        "Phases 0-6 done. Full pipeline runs from a prompt (`forge-mind run`); "
        "reviser (Phase 7) is next. See docs/PLAN.md for the roadmap."
    )


@app.command()
def fit(
    designation: Annotated[str, typer.Argument(help="ISO 286 fit, e.g. '50 H7/p6'")],
    hub_od: Annotated[
        float | None, typer.Option(help="Hub outer diameter (mm) -- enables press-fit")
    ] = None,
    length: Annotated[
        float | None, typer.Option(help="Engagement length (mm) -- enables press-fit")
    ] = None,
    hub_material: Annotated[
        str, typer.Option(help="Hub material (steel/aluminum/brass)")
    ] = "steel",
    shaft_material: Annotated[str, typer.Option(help="Shaft material")] = "steel",
) -> None:
    """Check an ISO 286 fit, and (with --hub-od/--length) the press-fit mechanics. No sign-in needed.

    e.g.  forge-mind fit "50 H7/p6" --hub-od 90 --length 20
    """
    from forge_mind.physics.fits import FitError, evaluate

    try:
        out = evaluate(
            designation,
            hub_od_mm=hub_od,
            engagement_mm=length,
            hub_material=hub_material,
            shaft_material=shaft_material,
        )
    except FitError as exc:
        typer.echo(f"cannot assess: {exc}")
        raise typer.Exit(1) from exc

    f = out.fit
    typer.echo(f"\n{f.summary}")
    typer.echo(
        f"  hole {f.hole_limits_mm[0]:.3f}-{f.hole_limits_mm[1]:.3f} mm "
        f"(+{f.hole_dev_um[0]}/{f.hole_dev_um[1]:+d} um)   "
        f"shaft {f.shaft_limits_mm[0]:.3f}-{f.shaft_limits_mm[1]:.3f} mm "
        f"({f.shaft_dev_um[0]:+d}/{f.shaft_dev_um[1]:+d} um)"
    )
    if out.press is not None:
        p = out.press
        typer.echo(
            f"\nPRESS-FIT (Lame, {p.hub_material} hub on {p.shaft_material} shaft) [{p.verdict.upper()}]:"
        )
        typer.echo(f"  interference   {p.interference_um[0]}-{p.interference_um[1]} um")
        typer.echo(f"  contact press. {p.contact_pressure_mpa[0]}-{p.contact_pressure_mpa[1]} MPa")
        typer.echo(f"  insertion      {p.insertion_force_kn[0]}-{p.insertion_force_kn[1]} kN")
        typer.echo(f"  torque cap.    {p.torque_capacity_nm[0]}-{p.torque_capacity_nm[1]} N.m")
        typer.echo(
            f"  hub hoop {p.hub_hoop_stress_mpa} MPa vs {p.yield_mpa} MPa yield -> SF {p.safety_factor}"
        )
        typer.echo(f"  -> {p.recommendation}")
        typer.echo("  assumptions:")
        for a in p.assumptions:
            typer.echo(f"    - {a}")
    elif out.note:
        typer.echo(f"  {out.note}")


@app.command()
def interference(
    part_a: Annotated[Path, typer.Argument(help="First mesh (STL/OBJ/PLY).")],
    part_b: Annotated[Path, typer.Argument(help="Second mesh, in the SAME coordinate frame.")],
    offset: Annotated[
        str | None, typer.Option(help="Translate B by 'x,y,z' mm before the check")
    ] = None,
) -> None:
    """Check whether two meshes collide (exact overlap volume) or how close they are. No sign-in.

    e.g.  forge-mind interference shaft.stl housing.stl --offset 0,0,12
    """
    import trimesh

    from forge_mind.validator.collision import interference_report

    a = trimesh.load(part_a, force="mesh")
    b = trimesh.load(part_b, force="mesh")
    if not isinstance(a, trimesh.Trimesh) or not isinstance(b, trimesh.Trimesh):
        typer.echo("could not load both inputs as single meshes")
        raise typer.Exit(2)
    b_offset: tuple[float, float, float] | None = None
    if offset is not None:
        parts = [float(x) for x in offset.split(",")]
        if len(parts) != 3:
            typer.echo("offset must be 'x,y,z'")
            raise typer.Exit(2)
        b_offset = (parts[0], parts[1], parts[2])

    r = interference_report(a, b, b_offset_mm=b_offset)
    typer.echo(f"\n{r.summary}")
    if r.penetration_volume_mm3:
        typer.echo(f"  penetration volume: {r.penetration_volume_mm3} mm^3")
    if r.min_clearance_mm is not None:
        typer.echo(f"  min clearance: {r.min_clearance_mm:.3f} mm")
    typer.echo(f"  method: {r.method}")
    for c in r.caveats:
        typer.echo(f"    - {c}")
    if not r.assessed:
        # Could not verify (refused, or overlapping bboxes with no boolean volume): do NOT report a
        # clean pass. Exit 2 so `if forge-mind interference a b; then proceed; fi` treats it as
        # "unknown", not "no interference" (audit M2, ADR-0065).
        raise typer.Exit(2)
    raise typer.Exit(1 if r.collide else 0)


@app.command()
def validate(
    dsl_path: Annotated[Path, typer.Argument(help="Path to a .partdsl.json file.")],
    out_dir: Annotated[Path, typer.Option(help="Where to write the STL.")] = Path(
        "benchmarks/results/_cli"
    ),
) -> None:
    """Compile a PartDSL file and run the full validator, printing the report."""
    # Imported lazily so `forge-mind version` stays fast (build123d import is heavy).
    from forge_mind.compiler.compile import compile_program
    from forge_mind.dsl.program import PartProgram
    from forge_mind.validator.run import validate as run_validate

    program = PartProgram.model_validate(json.loads(dsl_path.read_text()))
    model = compile_program(program, out_dir)
    report = run_validate(model, program)

    typer.echo(f"STL:     {model.stl_path}")
    typer.echo(f"STEP:    {model.step_path}")
    typer.echo(f"OVERALL: {report.overall.value.upper()}")
    for check in report.checks:
        mark = check.status.value.upper()
        line = f"  [{mark}] {check.id}"
        if check.status.value != "pass":
            line += f"  -> {check.message}"
        typer.echo(line)

    raise typer.Exit(code=0 if report.passed else 1)


@app.command()
def layout(
    spec_path: Annotated[Path, typer.Argument(help="Path to a .layoutspec.json file.")],
    out_dir: Annotated[Path, typer.Option(help="Where to write the STL.")] = Path(
        "benchmarks/results/_layout"
    ),
) -> None:
    """Run the deterministic LayoutSpec path: spec -> interpret -> compile -> validate.

    The alternate (ADR-0004) generation path. Proves a high-level LayoutSpec
    deterministically becomes a valid PartDSL part the existing validator accepts.
    """
    from pydantic import TypeAdapter

    from forge_mind.layout.layoutspec import LayoutSpec
    from forge_mind.layout.pipeline import run_layout_pipeline

    spec: LayoutSpec = TypeAdapter(LayoutSpec).validate_python(json.loads(spec_path.read_text()))
    result = run_layout_pipeline(spec, out_dir)

    if result.error is not None:
        typer.echo(f"FAILED at {result.failed_stage}: {result.error}", err=True)
        raise typer.Exit(code=1)

    assert result.model is not None and result.report is not None
    typer.echo(f"STL:     {result.model.stl_path}")
    typer.echo(f"STEP:    {result.model.step_path}")
    typer.echo(f"OVERALL: {result.report.overall.value.upper()}")
    for check in result.report.checks:
        mark = check.status.value.upper()
        line = f"  [{mark}] {check.id}"
        if check.status.value != "pass":
            line += f"  -> {check.message}"
        typer.echo(line)

    raise typer.Exit(code=0 if result.passed else 1)


@app.command()
def assemble(
    graph_path: Annotated[Path, typer.Argument(help="Path to an .assembly.json AssemblyGraph.")],
    out_dir: Annotated[Path, typer.Option(help="Where to write per-body STLs.")] = Path(
        "benchmarks/results/_assembly"
    ),
) -> None:
    """Solve an AssemblyGraph, compile each body to its own STL, and run the
    relational + kinematic validator (ADR-0010 multi-body path).

    Writes one printable STL per body, so a passing assembly can go straight to a
    slicer for the physical-rotation kill-gate.
    """
    from forge_mind.assembly.joints import AssemblyGraph
    from forge_mind.assembly.solver import solve
    from forge_mind.compiler.compile_assembly import compile_assembly
    from forge_mind.validator.assembly_checks import validate_assembly

    graph = AssemblyGraph.model_validate_json(graph_path.read_text())
    model = compile_assembly(solve(graph), out_dir)
    report = validate_assembly(model)

    for body in model.bodies:
        typer.echo(f"STL: {body.cad.stl_path}")
        typer.echo(f"STEP: {body.cad.step_path}")
    typer.echo(f"OVERALL: {report.overall.value.upper()}")
    for check in report.checks:
        mark = check.status.value.upper()
        line = f"  [{mark}] {check.id}"
        if check.status.value != "pass":
            line += f"  -> {check.message}"
        typer.echo(line)

    raise typer.Exit(code=0 if report.passed else 1)


@app.command("assemble-from-text")
def assemble_from_text(
    request: Annotated[str, typer.Argument(help="Plain-English mechanism request.")],
    out_dir: Annotated[Path, typer.Option(help="Where to write per-body STLs.")] = Path(
        "benchmarks/results/_assembly_from_text"
    ),
) -> None:
    """The full Path-4 chain from natural language (ADR-0010).

    request -> FunctionalPlan -> AssemblyGraph -> ConstraintSolver -> PartDSL ->
    compile -> validate -> reviser -> per-body STLs. The LLM picks the functional
    plan + topology; deterministic code derives every coordinate, sizes the fits,
    validates the relations + motion, and repairs a near-miss.
    """
    from forge_mind.assembly.generator import AssemblyGraphGenerator
    from forge_mind.assembly.reviser import revise_assembly
    from forge_mind.assembly.solver import solve
    from forge_mind.compiler.compile_assembly import compile_assembly
    from forge_mind.llm.client import LLMError

    try:
        graph, usage = AssemblyGraphGenerator()(request)
    except LLMError as exc:
        typer.echo(f"could not generate a coherent assembly: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"generated assembly (cost ${usage.cost_usd:.4f}); validating + revising...")

    result = revise_assembly(graph, out_dir / "revision")
    for patch in result.applied:
        typer.echo(f"  revised: {patch}")

    model = compile_assembly(solve(result.graph), out_dir)
    for body in model.bodies:
        typer.echo(f"STL: {body.cad.stl_path}")
        typer.echo(f"STEP: {body.cad.step_path}")
    typer.echo(f"OVERALL: {result.report.overall.value.upper()}")
    for check in result.report.checks:
        if check.status.value != "pass":
            typer.echo(f"  [{check.status.value.upper()}] {check.id} -> {check.message}")
    raise typer.Exit(code=0 if result.passed else 1)


@app.command("assemble-bench")
def assemble_bench(
    bench_dir: Annotated[
        Path, typer.Argument(help="Folder with prompts.json + pass_criteria.yaml.")
    ] = Path("benchmarks/bench-005-assembly-mechanisms"),
    trials: Annotated[int | None, typer.Option(help="Override trials-per-prompt.")] = None,
) -> None:
    """Measure the LLM driving the GENERAL assembly path (hinge + slide), with Wilson CIs.

    Decontaminated: the prompts describe the mechanism, not the answer. A trial passes
    when generate -> solve -> compile -> validate_assembly all pass.
    """
    import json

    import yaml

    from forge_mind.assembly.bench import run_assembly_benchmark, to_detail_dict

    prompts: dict[str, str] = json.loads((bench_dir / "prompts.json").read_text())
    criteria = yaml.safe_load((bench_dir / "pass_criteria.yaml").read_text())
    n = trials or int(criteria["trials_per_prompt"])
    gate = float(criteria["gate_wilson_lb"])
    max_cost = float(criteria["max_cost_usd"])
    # Tier-1 honest scoring (bench-007): required checks by kind + accepted known-deferred
    # warns. Absent for bench-004/005 -> empty -> the plain all-PASS gate, unchanged.
    required_checks = list(criteria.get("required_checks", []))
    accepted_warns = list(criteria.get("accepted_warns", []))

    typer.echo(f"Running {n} trials/prompt over {list(prompts)} (cap ${max_cost:.2f})...")
    result = run_assembly_benchmark(
        prompts,
        n,
        max_cost_usd=max_cost,
        required_checks=required_checks,
        accepted_warns=accepted_warns,
    )

    for p in result.prompts:
        stages = ", ".join(
            f"{s}={sum(1 for t in p.trials if t.stage == s)}"
            for s in ("pass", "validate", "compile", "solve", "emit")
            if any(t.stage == s for t in p.trials)
        )
        typer.echo(
            f"  {p.prompt_id:8} {p.passed}/{p.n} pass  Wilson LB {p.wilson_lb * 100:.0f}%  "
            f"${p.cost_usd:.3f}  [{stages}]"
        )
    typer.echo(
        f"OVERALL: {result.passed}/{result.n}  Wilson LB {result.wilson_lb * 100:.0f}%  ${result.cost_usd:.3f}"
    )
    detail_path = Path("benchmarks/results") / bench_dir.name / "trials.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(to_detail_dict(result), indent=2))
    typer.echo(f"per-trial detail (stage + failed checks): {detail_path}")
    if result.stopped_on_budget:
        typer.echo("(stopped on cost ceiling)")
    ok = result.wilson_lb >= gate and not result.stopped_on_budget
    typer.echo(f"GATE (>= {gate * 100:.0f}% Wilson LB): {'PASS' if ok else 'FAIL'}")
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def analyze(
    bench_id: Annotated[str, typer.Argument(help="Benchmark folder under benchmarks/.")],
    trials: Annotated[int, typer.Option(help="Number of end-to-end trials.")] = 30,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: UTC timestamp).")] = None,
    layoutspec: Annotated[
        bool,
        typer.Option(
            help="Route Blueprint->program via the LayoutSpec path (as `forge-mind run`)."
        ),
    ] = False,
) -> None:
    """Run N end-to-end trials and produce a failure-analysis dataset.

    Measurement only: runs the pipeline unchanged, classifies each failure, and
    writes report.md / report.json / records.jsonl with counts by label, by
    stage, top failed checks, cost, latency, and links to failed artifacts.
    With --layoutspec, measures the same path `forge-mind run` now uses.
    """
    from forge_mind.analysis.run import run_failure_analysis

    bench_dir = Path("benchmarks") / bench_id
    if not bench_dir.is_dir():
        typer.echo(f"No benchmark folder at {bench_dir}", err=True)
        raise typer.Exit(code=2)

    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    output_root = Path("benchmarks") / "results" / "_analysis"

    typer.echo(
        f"Running {trials} end-to-end trials for {bench_id} (path: {'layoutspec' if layoutspec else 'direct-dsl'})..."
    )
    dataset = run_failure_analysis(
        bench_dir, trials, output_root, resolved_run_id, use_layoutspec=layoutspec
    )

    typer.echo("")
    typer.echo(f"PASS RATE: {dataset.n_passed}/{dataset.n_trials} ({dataset.pass_rate * 100:.0f}%)")
    typer.echo(
        f"mean ${dataset.mean_cost_usd:.4f}/trial  {dataset.mean_duration_ms / 1000:.1f}s/trial"
    )
    typer.echo("labels: " + ", ".join(f"{k}={v}" for k, v in dataset.label_counts.items()))
    run_dir = output_root / bench_id / resolved_run_id
    typer.echo(f"report: {run_dir / 'report.md'}")


@app.command()
def candidates(
    bench_id: Annotated[str, typer.Argument(help="Benchmark folder under benchmarks/.")],
    trials: Annotated[int, typer.Option(help="Number of end-to-end trials.")] = 30,
    k: Annotated[int, typer.Option(help="Candidates generated per trial.")] = 5,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: UTC timestamp).")] = None,
    revise: Annotated[
        bool, typer.Option(help="Run the patch-only reviser on a failing winner.")
    ] = False,
) -> None:
    """Best-of-k reliability run: compare first-candidate vs best-of-k pass rate.

    Per trial: one intent + one blueprint, then k independent DSL candidates,
    each compiled, validated, scored deterministically; the best is kept. Saves
    every candidate's artifacts. With --revise, the patch-only reviser repairs a
    still-failing winner and a third pass rate is reported.
    """
    bench_dir = Path("benchmarks") / bench_id
    if not bench_dir.is_dir():
        typer.echo(f"No benchmark folder at {bench_dir}", err=True)
        raise typer.Exit(code=2)

    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    if revise:
        from forge_mind.analysis.revision import run_revision_analysis

        output_root = Path("benchmarks") / "results" / "_revision"
        typer.echo(f"Running {trials} trials x k={k} + revision for {bench_id}...")
        rev = run_revision_analysis(bench_dir, trials, k, output_root, resolved_run_id)
        first = rev.first_candidate_pass_rate * 100
        best = rev.best_of_k_pass_rate * 100
        revised = rev.revised_pass_rate * 100
        typer.echo("")
        typer.echo(
            f"FIRST: {first:.0f}%   BEST-OF-{k}: {best:.0f}%   "
            f"+REVISION: {revised:.0f}%   (+{revised - best:.0f} pts from revision)"
        )
        typer.echo(
            f"mean ${rev.mean_cost_usd:.4f}/trial (revision +${rev.mean_revision_cost_usd:.4f})  "
            f"{rev.mean_duration_ms / 1000:.1f}s/trial"
        )
        typer.echo(
            f"revision: {rev.n_revision_repaired}/{rev.n_revision_attempted} failing winners repaired"
        )
        typer.echo(f"report: {output_root / bench_id / resolved_run_id / 'report.md'}")
        return

    from forge_mind.analysis.bestofk import run_best_of_k_analysis

    output_root = Path("benchmarks") / "results" / "_bestofk"

    typer.echo(f"Running {trials} trials x k={k} candidates for {bench_id}...")
    dataset = run_best_of_k_analysis(bench_dir, trials, k, output_root, resolved_run_id)

    first = dataset.first_candidate_pass_rate * 100
    best = dataset.best_of_k_pass_rate * 100
    typer.echo("")
    typer.echo(
        f"FIRST-CANDIDATE: {first:.0f}%   BEST-OF-{k}: {best:.0f}%   (+{best - first:.0f} pts)"
    )
    typer.echo(
        f"mean ${dataset.mean_cost_usd:.4f}/trial  {dataset.mean_duration_ms / 1000:.1f}s/trial"
    )
    run_dir = output_root / bench_id / resolved_run_id
    typer.echo(f"report: {run_dir / 'report.md'}")


@app.command()
def abtest(
    bench_id: Annotated[str, typer.Argument(help="Benchmark folder under benchmarks/.")],
    generation_path: Annotated[
        str, typer.Option(help="Generation path: 'direct-dsl' or 'layoutspec'.")
    ] = "layoutspec",
    trials: Annotated[int, typer.Option(help="Number of end-to-end trials.")] = 5,
    k: Annotated[int, typer.Option(help="Candidates generated per trial.")] = 1,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: UTC timestamp).")] = None,
) -> None:
    """A/B benchmark: compare the LayoutSpec generation path vs the direct DSL path.

    Both paths share intent + blueprint + selection + compile + validate; only the
    generation step differs. Artifacts are saved per path under separate roots.
    """
    from forge_mind.analysis.bestofk import run_best_of_k_analysis

    if generation_path not in ("direct-dsl", "layoutspec"):
        typer.echo("generation-path must be 'direct-dsl' or 'layoutspec'", err=True)
        raise typer.Exit(code=2)

    bench_dir = Path("benchmarks") / bench_id
    if not bench_dir.is_dir():
        typer.echo(f"No benchmark folder at {bench_dir}", err=True)
        raise typer.Exit(code=2)

    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    typer.echo(f"A/B [{generation_path}]: {trials} trials x k={k} for {bench_id}...")
    if generation_path == "layoutspec":
        from forge_mind.layout.bench import run_layoutspec_best_of_k

        output_root = Path("benchmarks") / "results" / "_layoutspec"
        dataset = run_best_of_k_analysis(
            bench_dir, trials, k, output_root, resolved_run_id, runner=run_layoutspec_best_of_k
        )
    else:
        from forge_mind.candidates import run_best_of_k

        output_root = Path("benchmarks") / "results" / "_bestofk"
        dataset = run_best_of_k_analysis(
            bench_dir, trials, k, output_root, resolved_run_id, runner=run_best_of_k
        )

    single = dataset.first_candidate_pass_rate * 100
    best = dataset.best_of_k_pass_rate * 100
    typer.echo("")
    typer.echo(f"PATH: {generation_path}")
    typer.echo(f"  single-shot pass: {single:.0f}%   best-of-{k}: {best:.0f}%")
    typer.echo(
        f"  mean ${dataset.mean_cost_usd:.4f}/trial  {dataset.mean_duration_ms / 1000:.1f}s/trial"
    )
    if dataset.top_failed_after:
        typer.echo(
            "  top failed checks: "
            + ", ".join(f"{cid}={n}" for cid, n in dataset.top_failed_after[:6])
        )
    typer.echo(f"report: {output_root / bench_id / resolved_run_id / 'report.md'}")


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="Plain-English part request.")],
    out_dir: Annotated[Path, typer.Option(help="Output directory.")] = Path(
        "benchmarks/results/_run"
    ),
) -> None:
    """Run the full pipeline: prompt -> part. Writes DSL + STL + PNG + report.

    Routes the Blueprint->program step through the deterministic LayoutSpec path
    (with automatic fallback to direct DSL when no pattern matches), the reliable
    path the part benchmarks measure.
    """
    from forge_mind.pipeline import run_pipeline
    from forge_mind.viz.render import render_part_to_png

    result = run_pipeline(prompt, out_dir, use_layoutspec=True)

    if result.program is not None:
        (out_dir / "part.partdsl.json").write_text(result.program.model_dump_json(indent=2))
    if result.kit is not None:
        (out_dir / "kit.json").write_text(result.kit.model_dump_json(indent=2))
    if result.model is not None:
        render_part_to_png(result.model.stl_path, out_dir / "part.png", title=result.prompt[:60])

    typer.echo(
        f"path {result.gen_path}  cost ${result.total_cost_usd:.4f}  tokens {result.total_tokens}"
    )
    if result.refusal:
        # Out-of-domain / infeasible: a clean refusal, not a failure and not a (wrong) part.
        typer.echo(f"\n{result.refusal}")
        return
    if result.fallback_reason:
        typer.echo(f"  (router path declined -> direct-dsl: {result.fallback_reason})")
    if result.error:
        typer.echo(f"FAILED at {result.failed_stage}: {result.error}", err=True)
        raise typer.Exit(code=1)

    assert result.report is not None
    verdict = "PASS" if result.passed else "FAIL"
    typer.echo(f"OVERALL: {verdict}")
    for check in result.report.checks:
        if check.status.value != "pass":
            typer.echo(f"  [{check.status.value.upper()}] {check.id} -> {check.message}")
    if result.kit is not None:
        _echo_kit(result.kit)
    if result.model is not None:
        typer.echo(f"\nSTL: {result.model.stl_path}")
        typer.echo(f"STEP: {result.model.step_path}")
        typer.echo(f"PNG: {out_dir / 'part.png'}")
        typer.echo(f"KIT: {out_dir / 'kit.json'}")
    raise typer.Exit(code=0 if result.passed else 1)


@app.command("analyze-next")
def analyze_next(
    directory: Annotated[
        Path,
        typer.Argument(
            help="Folder containing committed *-report.json (e.g. benchmarks/analysis/<bench>) "
            "or an auto-runner output folder (e.g. benchmarks/results/_analysis/<bench>/<run>).",
        ),
    ],
    out_path: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Write next_action.md here instead of inside <directory>.",
        ),
    ] = None,
) -> None:
    """Read existing benchmark/result data and recommend the next scoped task.

    Read-only: never runs trials, never calls an LLM, never edits source. Picks
    the newest report.json under ``directory``, compares to the prior one if
    present, applies a small rule engine, and writes ``next_action.md`` (an
    implementation prompt with constraints + stop conditions). The actual
    engineering is still the human's call.
    """
    from forge_mind.analysis.research import (
        analyze_next as run_analyze_next,
    )
    from forge_mind.analysis.research import (
        diagnose,
        find_reports,
        load_report,
        recommend,
    )

    if not directory.is_dir():
        typer.echo(f"No such directory: {directory}", err=True)
        raise typer.Exit(code=2)

    reports = find_reports(directory)
    if not reports:
        typer.echo(f"No *report.json under {directory}", err=True)
        raise typer.Exit(code=2)

    current = load_report(reports[-1])
    baseline = load_report(reports[-2]) if len(reports) >= 2 else None
    diag = diagnose(current, baseline)
    rec = recommend(current, diag)

    written = run_analyze_next(directory, out_path)

    typer.echo(f"Latest report: {current.path}")
    if baseline is not None:
        typer.echo(f"Baseline:      {baseline.path}")
    typer.echo(f"Pass rate: {current.pass_rate * 100:.0f}% ({current.n_passed}/{current.n_trials})")
    if diag.dominant_label is not None:
        typer.echo(
            f"Dominant label: {diag.dominant_label} "
            f"({diag.dominant_label_share * 100:.0f}% of trials)"
        )
    if diag.dominant_check is not None:
        typer.echo(
            f"Dominant check: {diag.dominant_check[0]} "
            f"({diag.dominant_check[1]}/{current.n_trials})"
        )
    if diag.regression:
        typer.echo("Regression vs baseline.")
    elif diag.improvement:
        typer.echo("Improvement vs baseline.")
    typer.echo(f"Recommendation: [{rec.area}] {rec.title}")
    typer.echo(f"Wrote: {written}")


@app.command()
def render(
    stl_path: Annotated[Path, typer.Argument(help="Path to an STL file.")],
    png_path: Annotated[
        Path | None, typer.Argument(help="Output PNG (default: alongside the STL).")
    ] = None,
) -> None:
    """Render an STL to a shaded 3-view PNG for visual inspection."""
    from forge_mind.viz.render import render_part_to_png

    out = png_path or stl_path.with_suffix(".png")
    written = render_part_to_png(stl_path, out, title=stl_path.stem)
    typer.echo(f"Wrote {written}")


@app.command()
def bench(
    bench_id: Annotated[str, typer.Argument(help="Benchmark folder name under benchmarks/.")],
    use_handwritten: Annotated[
        bool,
        typer.Option("--use-handwritten", help="Bypass the LLM; feed the reference DSL."),
    ] = False,
    trials: Annotated[
        int | None, typer.Option(help="Override the trial count from pass_criteria.yaml.")
    ] = None,
    run_id: Annotated[str | None, typer.Option(help="Run id (default: UTC timestamp).")] = None,
    render: Annotated[
        bool, typer.Option("--render", help="Render each trial's STL to part.png.")
    ] = False,
) -> None:
    """Run a benchmark as N measured trials and print the summary."""
    from forge_mind.bench.criteria import load_criteria
    from forge_mind.bench.runner import run_benchmark
    from forge_mind.bench.runners import HandwrittenRunner

    bench_dir = Path("benchmarks") / bench_id
    if not bench_dir.is_dir():
        typer.echo(f"No benchmark folder at {bench_dir}", err=True)
        raise typer.Exit(code=2)

    criteria = load_criteria(bench_dir / "pass_criteria.yaml")
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    output_root = Path("benchmarks") / "results"

    if not use_handwritten:
        typer.echo(
            "Only --use-handwritten is supported until the LLM pipeline lands (Phase 4+).",
            err=True,
        )
        raise typer.Exit(code=2)
    if criteria.reference_dsl is None:
        typer.echo("pass_criteria.yaml has no reference_dsl for --use-handwritten.", err=True)
        raise typer.Exit(code=2)

    runner = HandwrittenRunner(Path(criteria.reference_dsl))
    summary = run_benchmark(
        bench_dir=bench_dir,
        runner=runner,
        criteria=criteria,
        run_id=resolved_run_id,
        output_root=output_root,
        n_trials=trials,
        render=render,
        reference_dsl=Path(criteria.reference_dsl),
    )

    verdict = "PASS" if summary.benchmark_passed else "FAIL"
    typer.echo(
        f"{verdict}: {summary.n_passed}/{summary.n_trials} trials "
        f"(need {summary.min_passing_trials})  "
        f"mean {summary.mean_duration_ms / 1000:.2f}s  ${summary.mean_cost_usd:.4f}"
    )
    typer.echo(f"Report: {output_root / bench_id / resolved_run_id / 'report.md'}")
    raise typer.Exit(code=0 if summary.benchmark_passed else 1)


@app.command()
def scale(
    validate: Annotated[
        bool,
        typer.Option(
            "--validate", help="Run the small-scale loop validation on known controls (UNGATED)."
        ),
    ] = False,
    cap_usd: Annotated[float, typer.Option(help="Hard budget cap (USD); stop-on-cap.")] = 8.0,
    n_paraphrases: Annotated[int, typer.Option(help="Paraphrases per perturbation point.")] = 15,
) -> None:
    """Automated scaling engine (ADR-0016, SAFE LAYER ONLY).

    ``--validate`` runs the small-scale loop on the feasible catalog at low N (ungated) to
    prove the machinery. The full budget scale-run (without ``--validate``) is BLOCKED until
    the validator (the oracle) is calibrated against physical prints -- the bottleneck is
    oracle TRUST, not budget (see config.SCALING_PRINT_CALIBRATED).
    """
    from forge_mind.scaling.engine import require_print_calibration, run_engine
    from forge_mind.scaling.enumerate import feasible_shapes
    from forge_mind.scaling.trial import LlmTrialRunner

    if not validate:
        # Gated: raises a clear, actionable error unless print-calibrated.
        require_print_calibration()

    shapes = feasible_shapes()
    typer.echo(
        f"{len(shapes)} feasible families; {'small-scale validation' if validate else 'SCALE-RUN'} (cap ${cap_usd:.0f})"
    )
    runner = LlmTrialRunner(out_dir=Path("benchmarks/results/scaling/_work"))
    report = run_engine(
        shapes, runner, cap_usd=cap_usd, n_paraphrases=(3 if validate else n_paraphrases)
    )
    typer.echo(
        f"banked {len(report.banked)} | flagged {len(report.flagged)} | "
        f"rejected {len(report.rejected)} | ${report.total_cost_usd:.2f} | "
        f"stopped_on_cap={report.stopped_on_cap}"
    )


@app.command()
def invention_bench(
    live: Annotated[
        bool,
        typer.Option(
            help="Run the LIVE bench (needs a configured model; fails clearly if unavailable). "
            "Default: offline with deterministic fixtures + mocked outputs, no spend."
        ),
    ] = False,
    out: Annotated[Path | None, typer.Option(help="Write the full run JSON to this path.")] = None,
) -> None:
    """Run the invention lifecycle benchmark (problem -> InventionPackage) (ADR-0048)."""
    from forge_mind.invention.bench import (
        LiveModelUnavailableError,
        assert_live_model_available,
        default_problems,
        offline_bench,
        run_bench,
        run_to_dict,
        save_run,
    )

    if live:
        try:
            assert_live_model_available()
        except LiveModelUnavailableError as exc:
            typer.echo(f"LIVE bench blocked: {exc}")
            raise typer.Exit(code=1) from exc
        from forge_mind.api.core import produce_for_invention
        from forge_mind.config import BRAIN_MODEL
        from forge_mind.llm.provider import make_client

        client = make_client()
        art_dir = (out.parent if out is not None else Path.cwd()) / "invention_bench_artifacts"
        run = run_bench(
            default_problems(),
            client=client,
            produce=lambda prompt: produce_for_invention(prompt, art_dir, client=client),
            mode="live",
            model=BRAIN_MODEL,
        )
    else:
        run = offline_bench()

    typer.echo(json.dumps(run_to_dict(run), indent=2))
    if out is not None:
        save_run(run, out)
        typer.echo(f"wrote {len(run.entries)} entries to {out}")


if __name__ == "__main__":
    app()
