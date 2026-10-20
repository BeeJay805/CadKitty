"""bench-001 prompt-perturbation experiment.

The bench-003 perturbation work (`scripts/bench003_perturbation_experiment.py`)
exposed a real failure mode: when the LLM picks dimensions the structure
should derive, it makes confidently wrong choices under envelope pressure.
The fix was to mark those dimensions optional in the spec and derive them
in the interpreter. That moved the bench-003 tight-X variant from 0/5 to 5/5.

This script asks the same question of bench-001: does the L-bracket pattern
have the same fragility? `LBracketMotorMountSpec` has four LLM-set dims
(``base.width_mm``, ``base.depth_mm``, ``upright.width_mm``,
``upright.height_mm``); on the canonical envelope (60x60x40, min_wall=4) the
prompt's defaults (43 x 50 base, 43 x 30 upright) fit easily, so the model
copies them. Under envelope pressure it might or might not adapt.

Three perturbations, mirroring the bench-003 script's shape:

1. **TIGHTER X** (45 x 60 x 40 envelope, canonical min_wall): the canonical
   base width 43 still fits in 45, but the model has to compute whether
   the lands hold at the tighter X. Mild pressure.
2. **THICKER WALL** (60 x 60 x 40 envelope, min_wall = 6): canonical base
   width 43 now violates the wall rule (needs 2*(15.5+1.6+6) = 46.2). The
   model MUST grow the base, or the interpreter rejects.
3. **SHORT Z** (60 x 60 x 25 envelope, canonical min_wall): canonical
   upright_height 30 does not fit in Z=25 (envelope minus base = 21 max).
   The model MUST shrink the upright.

Each variant runs N independent LLM trials. Output goes to
``benchmarks/analysis/bench-001-nema17-mount-v1/perturbation{-<tag>}-n{N}.{json,md}``.

Run with::

    uv run python scripts/bench001_perturbation_experiment.py            # default N=5 per variant
    uv run python scripts/bench001_perturbation_experiment.py --tag prefix-fix --trials 5
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
from forge_mind.layout.interpreter import LayoutError, derive_l_bracket_dims
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.spec import LBracketMotorMountSpec
from forge_mind.llm.client import LLMError, api_key_available
from forge_mind.llm.cost import LLMUsage
from forge_mind.validator.run import validate

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "benchmarks" / "analysis" / "bench-001-nema17-mount-v1"

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
    """Build a synthetic bench-001-shaped L-bracket Blueprint with one dim varied."""
    return Blueprint.model_validate(
        {
            "name": name,
            "summary": (
                "L-bracket: motor flange carries the NEMA 17; extrusion flange bolts to a "
                "2020 rail."
            ),
            "envelope_mm": {"x": envelope[0], "y": envelope[1], "z": envelope[2]},
            "print_orientation": {"x": 0.0, "y": 0.0, "z": 1.0},
            "material": "PLA",
            "min_wall_mm": min_wall,
            "max_overhang_deg": 45.0,
            "features": [
                {
                    "name": "motor_flange",
                    "kind": "plate",
                    "purpose": "Carries the NEMA 17 motor; holds the bolt pattern and shaft bore.",
                    "approx_size_mm": {
                        "x": min(envelope[0] - 5.0, 43.0),
                        "y": min(envelope[1] - 5.0, 50.0),
                        "z": min_wall,
                    },
                },
                {
                    "name": "extrusion_flange",
                    "kind": "plate",
                    "purpose": "Bolts to the 2020 extrusion via two M5 T-nuts.",
                    "approx_size_mm": {
                        "x": min(envelope[0] - 5.0, 43.0),
                        "y": min_wall,
                        "z": min(envelope[2] - 5.0, 30.0),
                    },
                },
            ],
            "required_interfaces": [
                {
                    "name": "motor_face",
                    "interface_type": InterfaceType.NEMA17_FACE.value,
                    "on_feature": "motor_flange",
                    "facing": "+Z",
                },
                {
                    "name": "extrusion_face",
                    "interface_type": InterfaceType.EXTRUSION_2020_SLOT.value,
                    "on_feature": "extrusion_flange",
                    "facing": "+Y",
                },
            ],
        }
    )


def variants() -> list[Variant]:
    return [
        Variant(
            tag="tight-x",
            description=(
                "45 x 60 x 40 envelope (X tightened from 60). The canonical 43 mm base "
                "width still fits 45 mm, so the model COULD just copy defaults; this "
                "tests whether the model notices the X envelope shrunk and stays inside."
            ),
            blueprint=_make_blueprint(
                name="bench001_tight_x",
                envelope=(45.0, 60.0, 40.0),
                min_wall=4.0,
            ),
        ),
        Variant(
            tag="thicker-wall",
            description=(
                "60 x 60 x 40 envelope with min_wall = 6 mm. The prompt's default base "
                "width 43 violates the wall rule (needs 2*(15.5+1.6+6) = 46.2). If the "
                "model copies the default base size, the interpreter MUST reject."
            ),
            blueprint=_make_blueprint(
                name="bench001_thicker_wall",
                envelope=(60.0, 60.0, 40.0),
                min_wall=6.0,
            ),
        ),
        Variant(
            tag="short-z",
            description=(
                "60 x 60 x 25 envelope, canonical min_wall = 4. The prompt's default "
                "upright_height of 30 mm does NOT fit Z=25 (envelope minus 4 mm base = "
                "21 mm max upright). The model MUST shrink the upright."
            ),
            blueprint=_make_blueprint(
                name="bench001_short_z",
                envelope=(60.0, 60.0, 25.0),
                min_wall=4.0,
            ),
        ),
    ]


@dataclass
class TrialOutcome:
    trial_index: int
    variant: str
    pattern_picked: str
    base_width_mm: float | None
    base_depth_mm: float | None
    upright_width_mm: float | None
    upright_height_mm: float | None
    m5_spacing_mm: float | None
    interpreter_accepted: bool
    validator_passed: bool
    failed_check_ids: list[str]
    cost_usd: float
    duration_ms: float
    notes: str


def _resolve_dims(spec: LBracketMotorMountSpec) -> tuple[float, float, float, float]:
    """Resolve base/upright dims for reporting -- explicit values win, else derive."""
    derived_base, derived_up = derive_l_bracket_dims(spec)
    base_w = spec.base.width_mm if spec.base is not None else derived_base.width_mm
    base_d = spec.base.depth_mm if spec.base is not None else derived_base.depth_mm
    up_w = spec.upright.width_mm if spec.upright is not None else derived_up.width_mm
    up_h = spec.upright.height_mm if spec.upright is not None else derived_up.height_mm
    return (base_w, base_d, up_w, up_h)


def _summarise(spec: object) -> dict[str, float | str | None]:
    if isinstance(spec, LBracketMotorMountSpec):
        base_w, base_d, up_w, up_h = _resolve_dims(spec)
        return {
            "pattern": spec.pattern.value,
            "base_width_mm": base_w,
            "base_depth_mm": base_d,
            "upright_width_mm": up_w,
            "upright_height_mm": up_h,
            "m5_spacing_mm": spec.extrusion_2020.m5_spacing_mm,
        }
    pattern = getattr(spec, "pattern", None)
    return {"pattern": pattern.value if pattern is not None else "(none)"}


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
            base_width_mm=None,
            base_depth_mm=None,
            upright_width_mm=None,
            upright_height_mm=None,
            m5_spacing_mm=None,
            interpreter_accepted=False,
            validator_passed=False,
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=f"LLM generator error: {exc}",
        )

    if not isinstance(spec, LBracketMotorMountSpec):
        notes.append(f"model picked '{pattern_picked}' instead of l_bracket")
        return TrialOutcome(
            trial_index=trial_index,
            variant=variant.tag,
            pattern_picked=pattern_picked,
            base_width_mm=None,
            base_depth_mm=None,
            upright_width_mm=None,
            upright_height_mm=None,
            m5_spacing_mm=None,
            interpreter_accepted=False,
            validator_passed=False,
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes="; ".join(notes),
        )

    summary = _summarise(spec)
    base_explicit = spec.base is not None
    upright_explicit = spec.upright is not None
    if base_explicit or upright_explicit:
        notes.append(
            "emitted explicit dims: "
            + ", ".join(
                f"{k}={summary[k]}"
                for k in (
                    "base_width_mm",
                    "base_depth_mm",
                    "upright_width_mm",
                    "upright_height_mm",
                    "m5_spacing_mm",
                )
            )
        )
    else:
        notes.append(
            "left base/upright null -> interpreter derived "
            f"base {summary['base_width_mm']:.2f}x{summary['base_depth_mm']:.2f} "
            f"upright {summary['upright_width_mm']:.2f}x{summary['upright_height_mm']:.2f} "
            f"(m5_spacing={summary['m5_spacing_mm']})"
        )

    interpreter_ok = False
    validator_ok = False
    failed_ids: list[str] = []
    base_w, base_d, up_w, up_h = _resolve_dims(spec)
    try:
        program = interpret_layout(spec)
        interpreter_ok = True
    except LayoutError as exc:
        notes.append(f"interpreter rejected: {exc}")
        return TrialOutcome(
            trial_index=trial_index,
            variant=variant.tag,
            pattern_picked=pattern_picked,
            base_width_mm=base_w,
            base_depth_mm=base_d,
            upright_width_mm=up_w,
            upright_height_mm=up_h,
            m5_spacing_mm=spec.extrusion_2020.m5_spacing_mm,
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
        base_width_mm=base_w,
        base_depth_mm=base_d,
        upright_width_mm=up_w,
        upright_height_mm=up_h,
        m5_spacing_mm=spec.extrusion_2020.m5_spacing_mm,
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
    lines.append("# bench-001 perturbation experiment")
    lines.append("")
    lines.append(
        f"{n_trials} LLM trials per variant on synthetic Blueprints that diverge "
        "from the canonical bench-001 Blueprint along ONE dimension at a time. "
        "Asks whether the L-bracket pattern has the same 'LLM picks dims that "
        "should be derived' fragility bench-003 had."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| variant | trials | left dims null | interpreter accepted | validator passed | cost |"
    )
    lines.append("|---|---|---|---|---|---|")
    for tag, runs in by_variant.items():
        null_dims = sum(1 for r in runs if "left base/upright null" in r.notes)
        accepted = sum(1 for r in runs if r.interpreter_accepted)
        passed = sum(1 for r in runs if r.validator_passed)
        cost = sum(r.cost_usd for r in runs)
        lines.append(
            f"| {tag} | {len(runs)} | {null_dims}/{len(runs)} | {accepted}/{len(runs)} | "
            f"{passed}/{len(runs)} | ${cost:.4f} |"
        )
    lines.append("")
    lines.append("## Per-trial detail")
    lines.append("")
    for tag, runs in by_variant.items():
        lines.append(f"### {tag}")
        lines.append("")
        lines.append("| trial | base WxD | upright WxH | m5 | interp | valid | notes |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in runs:
            base = f"{r.base_width_mm}x{r.base_depth_mm}" if r.base_width_mm is not None else "-"
            up = (
                f"{r.upright_width_mm}x{r.upright_height_mm}"
                if r.upright_width_mm is not None
                else "-"
            )
            interp = "PASS" if r.interpreter_accepted else "REJECT"
            valid = "PASS" if r.validator_passed else ("FAIL" if r.interpreter_accepted else "n/a")
            lines.append(
                f"| {r.trial_index} | {base} | {up} | {r.m5_spacing_mm} | "
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
        help="Output-filename tag, e.g. 'prefix-fix' -> perturbation-prefix-fix-n5.{json,md}. "
        "Default (empty) writes perturbation-n5.{json,md}.",
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
