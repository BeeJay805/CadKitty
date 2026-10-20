"""bench-002 prompt-perturbation experiment.

Mirrors bench-001 (`scripts/bench001_perturbation_experiment.py`) and
bench-003 (`scripts/bench003_perturbation_experiment.py`): synthesises
Blueprints that diverge from the canonical bench-002 along one dimension
and asks whether the LLM picks values consistent with the land rules.

The bench-002 fragility surface is narrower than bench-001/bench-003's:
when the LLM uses `board_component_id` the board + standoff facts come
from the registry (already derived; no LLM fragility). The risk is on
the LEGACY `board`+`standoffs` path -- the LLM picks `outer_diameter_mm`
and `clearance_mm` directly. The canonical bench-002 fixture uses the
legacy path, and the canonical standoff_outer_diameter_mm = 7.0 only
keeps a 2.15 mm land for min_wall = 2.0 (the ADR-0005 resolution). Under
thicker-wall pressure (min_wall = 3 or 4), the LLM has to bump the OD;
if it doesn't, the interpreter rejects.

Two perturbations:

1. **THICKER WALL** (95 x 70 x 35 envelope, min_wall = 4). The canonical
   OD = 7 keeps only 2.15 mm land -- below the 4 mm wall. The LLM MUST
   either bump OD to >= 10.7 mm OR switch to the board_component_id path
   (which the registry can't satisfy at min_wall = 4 either, so the
   interpreter still rejects -- the request would be honestly infeasible
   in that case).
2. **TIGHT XY** (90 x 65 x 30 envelope). Canonical case body is 93 x 64;
   shrinking X below 93 forces the LLM to either pick a tighter clearance
   (down to >= min_wall = 2) or fail. Tests whether the LLM understands
   the case = board + 2 * (clearance + min_wall) sum.

Each variant runs N independent LLM trials. Output goes to
``benchmarks/analysis/bench-002-pi4-case-v1/perturbation{-<tag>}-n{N}.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from forge_mind.blueprint.schema import Blueprint
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces.declaration import InterfaceType
from forge_mind.layout.generator import LayoutSpecGenerator
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.llm.client import LLMError, api_key_available
from forge_mind.llm.cost import LLMUsage
from forge_mind.validator.run import validate

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "benchmarks" / "analysis" / "bench-002-pi4-case-v1"

DEFAULT_N_TRIALS = 5


@dataclass(frozen=True)
class Variant:
    tag: str
    description: str
    blueprint: Blueprint


def _make_blueprint(
    *,
    name: str,
    envelope: tuple[float, float, float],
    min_wall: float,
) -> Blueprint:
    """Build a synthetic bench-002-shaped Blueprint with one dim varied."""
    return Blueprint.model_validate(
        {
            "name": name,
            "summary": (
                "Open-top case for a Raspberry Pi 4 Model B SBC: floor + perimeter walls "
                "+ four standoffs + port cutouts. The board is 85x56 mm with four M2.5 "
                "holes on a 58x49 rectangle."
            ),
            "envelope_mm": {"x": envelope[0], "y": envelope[1], "z": envelope[2]},
            "print_orientation": {"x": 0.0, "y": 0.0, "z": 1.0},
            "material": "PLA",
            "min_wall_mm": min_wall,
            "max_overhang_deg": 45.0,
            "features": [
                {
                    "name": "floor",
                    "kind": "plate",
                    "purpose": "Tray floor that carries the board on four standoffs.",
                    "approx_size_mm": {
                        "x": min(envelope[0] - 2.0, 93.0),
                        "y": min(envelope[1] - 2.0, 64.0),
                        "z": min_wall,
                    },
                },
                {
                    "name": "wall",
                    "kind": "wall",
                    "purpose": "Perimeter walls retaining the board with cutouts for ports.",
                    "approx_size_mm": {
                        "x": min_wall,
                        "y": min(envelope[1] - 2.0, 64.0),
                        "z": min(envelope[2] - 5.0, 16.0),
                    },
                },
            ],
            "required_interfaces": [
                {
                    "name": "pi4_mount",
                    "interface_type": InterfaceType.PI4_MOUNT.value,
                    "on_feature": "floor",
                    "facing": "+Z",
                },
                {
                    "name": "pi4_ports",
                    "interface_type": InterfaceType.PI4_PORTS.value,
                    "on_feature": "wall",
                    "facing": "+Y",
                },
            ],
        }
    )


def variants() -> list[Variant]:
    # Both envelopes are sized so the case CAN fit given the larger walls --
    # the perturbation tests whether the LLM picks the right standoff_OD on
    # the legacy path (or correctly notices the registry's OD-7 won't clear
    # the new min_wall and picks the legacy path instead). Pre-pick envelopes
    # that are infeasible-by-construction don't measure LLM fragility, they
    # measure the interpreter's honesty (already proven by bench-003).
    return [
        Variant(
            tag="wall-3",
            description=(
                "110 x 85 x 30 envelope with min_wall = 3 mm. At min_wall=3 the "
                "required standoff OD is >= 2*(1.35+3) = 8.7 mm; the registry's "
                "raspberry_pi_4 OD-7 keeps only a 2.15 mm land (< 3 mm wall) "
                "but env has headroom (case >= ~105x72 fits). Tests whether the "
                "LLM bumps OD or stays stuck on the registry's hardcoded default."
            ),
            blueprint=_make_blueprint(
                name="bench002_wall_3",
                envelope=(110.0, 85.0, 30.0),
                min_wall=3.0,
            ),
        ),
        Variant(
            tag="wall-4",
            description=(
                "120 x 95 x 35 envelope with min_wall = 4 mm. Required standoff OD "
                ">= 10.7 mm; registry's OD-7 fails the land check by ~1.8 mm. "
                "Envelope deliberately roomy (case ~107x72 fits well inside) so "
                "the failure mode is purely about standoff sizing, not about "
                "case envelope sums."
            ),
            blueprint=_make_blueprint(
                name="bench002_wall_4",
                envelope=(120.0, 95.0, 35.0),
                min_wall=4.0,
            ),
        ),
    ]


@dataclass
class TrialOutcome:
    trial_index: int
    variant: str
    pattern_picked: str
    path_used: str  # "component" | "legacy" | "(none)"
    clearance_mm: float | None
    standoff_outer_diameter_mm: float | None
    board_component_id: str | None
    interpreter_accepted: bool
    validator_passed: bool
    failed_check_ids: list[str]
    cost_usd: float
    duration_ms: float
    notes: str


def _summarise(spec: object) -> dict[str, float | str | None]:
    if isinstance(spec, ElectronicsTraySpec):
        path = "component" if spec.board_component_id is not None else "legacy"
        od = spec.standoffs.outer_diameter_mm if spec.board_component_id is None else None
        return {
            "pattern": spec.pattern.value,
            "path": path,
            "clearance_mm": spec.clearance_mm,
            "standoff_outer_diameter_mm": od,
            "board_component_id": spec.board_component_id,
        }
    pattern = getattr(spec, "pattern", None)
    return {
        "pattern": pattern.value if pattern is not None else "(none)",
        "path": "(none)",
    }


def _run_one(variant: Variant, trial_index: int) -> TrialOutcome:
    t0 = time.perf_counter()
    gen = LayoutSpecGenerator()
    cost = 0.0
    notes: list[str] = []
    pattern_picked = "(none)"

    try:
        spec, usage = gen(variant.blueprint)
        assert isinstance(usage, LLMUsage)
        cost += usage.cost_usd
        pattern_picked = spec.pattern.value
    except LLMError as exc:
        return TrialOutcome(
            trial_index=trial_index,
            variant=variant.tag,
            pattern_picked=pattern_picked,
            path_used="(none)",
            clearance_mm=None,
            standoff_outer_diameter_mm=None,
            board_component_id=None,
            interpreter_accepted=False,
            validator_passed=False,
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=f"LLM generator error: {exc}",
        )

    if not isinstance(spec, ElectronicsTraySpec):
        notes.append(f"model picked '{pattern_picked}' instead of pi4_case (electronics_tray)")
        return TrialOutcome(
            trial_index=trial_index,
            variant=variant.tag,
            pattern_picked=pattern_picked,
            path_used="(none)",
            clearance_mm=None,
            standoff_outer_diameter_mm=None,
            board_component_id=None,
            interpreter_accepted=False,
            validator_passed=False,
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes="; ".join(notes),
        )

    summary = _summarise(spec)
    path = str(summary["path"])
    notes.append(
        f"path={path}, clearance={summary['clearance_mm']}, "
        f"standoff_OD={summary['standoff_outer_diameter_mm']}, "
        f"board_id={summary['board_component_id']}"
    )

    interpreter_ok = False
    validator_ok = False
    failed_ids: list[str] = []
    try:
        program = interpret_layout(spec)
        interpreter_ok = True
    except LayoutError as exc:
        notes.append(f"interpreter rejected: {exc}")
        return TrialOutcome(
            trial_index=trial_index,
            variant=variant.tag,
            pattern_picked=pattern_picked,
            path_used=path,
            clearance_mm=spec.clearance_mm,
            standoff_outer_diameter_mm=(
                spec.standoffs.outer_diameter_mm if spec.board_component_id is None else None
            ),
            board_component_id=spec.board_component_id,
            interpreter_accepted=False,
            validator_passed=False,
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes="; ".join(notes),
        )

    with tempfile.TemporaryDirectory() as td:
        model = compile_program(program, Path(td))
        report = validate(model, program)
        validator_ok = report.passed
        failed_ids = report.failed_check_ids()
        if not validator_ok:
            notes.append(f"validator failed: {failed_ids}")

    return TrialOutcome(
        trial_index=trial_index,
        variant=variant.tag,
        pattern_picked=pattern_picked,
        path_used=path,
        clearance_mm=spec.clearance_mm,
        standoff_outer_diameter_mm=(
            spec.standoffs.outer_diameter_mm if spec.board_component_id is None else None
        ),
        board_component_id=spec.board_component_id,
        interpreter_accepted=interpreter_ok,
        validator_passed=validator_ok,
        failed_check_ids=failed_ids,
        cost_usd=cost,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
        notes="; ".join(notes),
    )


def render_markdown(n_trials: int, outcomes: list[TrialOutcome]) -> str:
    by_variant: dict[str, list[TrialOutcome]] = {}
    for o in outcomes:
        by_variant.setdefault(o.variant, []).append(o)

    lines: list[str] = []
    lines.append("# bench-002 perturbation experiment")
    lines.append("")
    lines.append(
        f"{n_trials} LLM trials per variant on synthetic Blueprints that diverge "
        "from the canonical bench-002 along ONE dim. Asks whether the LLM picks "
        "values consistent with the standoff-land + envelope rules."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| variant | trials | component path | interpreter accepted | validator passed | cost |"
    )
    lines.append("|---|---|---|---|---|---|")
    for tag, runs in by_variant.items():
        comp = sum(1 for r in runs if r.path_used == "component")
        accepted = sum(1 for r in runs if r.interpreter_accepted)
        passed = sum(1 for r in runs if r.validator_passed)
        cost = sum(r.cost_usd for r in runs)
        lines.append(
            f"| {tag} | {len(runs)} | {comp}/{len(runs)} | {accepted}/{len(runs)} | "
            f"{passed}/{len(runs)} | ${cost:.4f} |"
        )
    lines.append("")
    lines.append("## Per-trial detail")
    lines.append("")
    for tag, runs in by_variant.items():
        lines.append(f"### {tag}")
        lines.append("")
        lines.append(
            "| trial | path | clearance | standoff OD | board id | interp | valid | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in runs:
            interp = "PASS" if r.interpreter_accepted else "REJECT"
            valid = "PASS" if r.validator_passed else ("FAIL" if r.interpreter_accepted else "n/a")
            lines.append(
                f"| {r.trial_index} | {r.path_used} | {r.clearance_mm} | "
                f"{r.standoff_outer_diameter_mm} | {r.board_component_id} | "
                f"{interp} | {valid} | {r.notes} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not api_key_available():
        raise SystemExit("ANTHROPIC_API_KEY not set; this experiment makes real LLM calls.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS, help="trials per variant")
    parser.add_argument(
        "--tag",
        default="",
        help="Output-filename tag, e.g. 'prefix-fix' -> perturbation-prefix-fix-n5.{json,md}.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = args.trials
    tag = f"-{args.tag}" if args.tag else ""
    all_outcomes: list[TrialOutcome] = []
    for variant in variants():
        print(f"\n=== variant: {variant.tag} ===")
        print(f"    {variant.description}")
        for i in range(n):
            o = _run_one(variant, i)
            all_outcomes.append(o)
            label = (
                "PASS"
                if o.validator_passed
                else ("REJECT" if not o.interpreter_accepted else "FAIL")
            )
            print(
                f"  trial {i}: {label}  ${o.cost_usd:.4f}  {o.duration_ms / 1000:.1f}s -- {o.notes}"
            )

    json_path = OUTPUT_DIR / f"perturbation{tag}-n{n}.json"
    md_path = OUTPUT_DIR / f"perturbation{tag}-n{n}.md"
    json_path.write_text(
        json.dumps(
            {"n_trials_per_variant": n, "outcomes": [asdict(o) for o in all_outcomes]}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(n, all_outcomes), encoding="utf-8")
    print(f"\nwrote {json_path.relative_to(REPO)}")
    print(f"wrote {md_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
