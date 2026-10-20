"""bench-003 prompt-perturbation experiment.

The headline bench-003 LayoutSpec A/B was 10/10 on the prompt's recommended
default geometry (W=65, D=45, motor X=+11, T-nut X=-22, m5_spacing=20). The
prompt itself spells those numbers out. So the run measured:

    "the model can pattern-pick and copy defaults"

What it did NOT measure:

    "the model can actually compute valid adapter_plate dimensions for a
     Blueprint that asks for something the prompt's defaults do not satisfy"

This script measures the second thing. It synthesises Blueprints that diverge
from the canonical bench-003 Blueprint along ONE dimension at a time, feeds
each through ``LayoutSpecGenerator`` -> ``interpret_layout`` -> compile ->
validate, and records:

- did the model pick the ``adapter_plate`` pattern at all?
- what dimensions did it emit?
- did the deterministic interpreter accept the spec (= valid lands +
  envelope), or did it raise ``LayoutError``?
- did the resulting part pass the validator?

Two perturbations, deliberately picked so that copying the documented
defaults *cannot* succeed:

1. **TIGHT envelope** (60 x 40 x 8). The prompt's default plate (65 x 45)
   does not fit. The interpreter MUST reject a copy-the-default spec.
2. **THICKER wall** (min_wall = 6.0 inside an 80 x 60 x 10 envelope). The
   prompt's default motor-X offset (+11) gives a 4.4 mm right-edge land,
   which violates a 6 mm min_wall. The interpreter MUST reject a copy-the-
   default spec.

Each variant runs N independent LLM trials. Output goes to
``benchmarks/analysis/bench-003-nema17-2020-adapter-plate-v1/perturbation-
n{N}-{tag}.{json,md}`` so it can be browsed alongside the other committed
snapshots.

Run with::

    uv run python scripts/bench003_perturbation_experiment.py            # default N=5 per variant
    uv run python scripts/bench003_perturbation_experiment.py --trials 3 # cheaper smoke
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from forge_mind.blueprint.schema import Blueprint
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces.declaration import InterfaceType
from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.generator import LayoutSpecGenerator
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.llm.client import LLMError, api_key_available
from forge_mind.llm.cost import LLMUsage
from forge_mind.validator.run import validate

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "benchmarks" / "analysis" / "bench-003-nema17-2020-adapter-plate-v1"

# Default per-variant trial count. Keep small -- this is a measurement, not a
# benchmark; the architectural question is binary (did the model adapt or copy).
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
    extrusion_name: str = "extrusion_face",
) -> Blueprint:
    """Build a synthetic bench-003-shaped Blueprint with one dim varied."""
    return Blueprint.model_validate(
        {
            "name": name,
            "summary": (
                "A flat 3D-printable PLA adapter plate that mounts a NEMA 17 stepper motor "
                "to a 2020 aluminium extrusion via M5 T-nuts. The plate lies flat on the "
                "build plate with the motor face on top and the extrusion interface on "
                "the bottom."
            ),
            "envelope_mm": {"x": envelope[0], "y": envelope[1], "z": envelope[2]},
            "print_orientation": {"x": 0.0, "y": 0.0, "z": 1.0},
            "material": "PLA",
            "min_wall_mm": min_wall,
            "max_overhang_deg": 45.0,
            "features": [
                {
                    "name": "adapter_plate",
                    "kind": "plate",
                    "purpose": (
                        "Flat body bridging the NEMA 17 bolt pattern on the top face and "
                        "the 2020 extrusion M5 T-nut holes on the bottom face."
                    ),
                    "approx_size_mm": {
                        "x": min(envelope[0] - 5.0, 60.0),
                        "y": min(envelope[1] - 5.0, 40.0),
                        "z": min_wall,
                    },
                }
            ],
            "required_interfaces": [
                {
                    "name": "motor_face",
                    "interface_type": InterfaceType.NEMA17_FACE.value,
                    "on_feature": "adapter_plate",
                    "facing": "+Z",
                },
                {
                    "name": extrusion_name,
                    "interface_type": InterfaceType.EXTRUSION_2020_SLOT.value,
                    "on_feature": "adapter_plate",
                    "facing": "-Z",
                },
            ],
        }
    )


def variants() -> list[Variant]:
    return [
        Variant(
            tag="tight-x",
            description=(
                "60 x 50 x 8 envelope (X tightened from canonical 70). The prompt's "
                "default 65 mm plate does NOT fit X but Y has room. A model picking the "
                "canonical motor offset +11 demands plate width 64.2 (over X budget); "
                "the model must pick a smaller motor offset OR (with the structural fix) "
                "let the interpreter derive the plate from a smaller offset choice."
            ),
            blueprint=_make_blueprint(
                name="bench003_tight_x",
                envelope=(60.0, 50.0, 8.0),
                min_wall=4.0,
            ),
        ),
        Variant(
            tag="tight-xy-impossible",
            description=(
                "60 x 40 x 8 envelope. A NEMA17 M3 cluster + 4 mm walls REQUIRES depth "
                ">= 42.2 mm regardless of offsets, so this envelope is geometrically "
                "infeasible. Useful test: does the interpreter reject honestly (vs the "
                "pre-fix behaviour where the model emitted inconsistent dims and the "
                "interpreter rejected on a confusing model-error message)?"
            ),
            blueprint=_make_blueprint(
                name="bench003_tight_xy_impossible",
                envelope=(60.0, 40.0, 8.0),
                min_wall=4.0,
            ),
        ),
        Variant(
            tag="thicker-wall",
            description=(
                "80 x 60 x 10 envelope with min_wall = 6 mm; the prompt's default motor X "
                "(+11) gives a 4.4 mm right-edge land that violates a 6 mm min_wall. "
                "If the model copies the default motor offset, the interpreter MUST reject."
            ),
            blueprint=_make_blueprint(
                name="bench003_thicker_wall",
                envelope=(80.0, 60.0, 10.0),
                min_wall=6.0,
            ),
        ),
    ]


@dataclass
class TrialOutcome:
    trial_index: int
    variant: str
    pattern_picked: str  # the LLM's chosen pattern (or "(none)" if it failed early)
    plate_width_mm: float | None
    plate_depth_mm: float | None
    motor_center_x_mm: float | None
    extrusion_center_x_mm: float | None
    m5_spacing_mm: float | None
    interpreter_accepted: bool
    validator_passed: bool
    failed_check_ids: list[str]
    cost_usd: float
    duration_ms: float
    notes: str  # plain-language "did it copy or adapt"


def _summarise_spec(spec: object) -> dict[str, float | str | None]:
    if isinstance(spec, AdapterPlateSpec):
        return {
            "pattern": spec.pattern.value,
            "plate_width_mm": spec.plate_width_mm,
            "plate_depth_mm": spec.plate_depth_mm,
            "motor_center_x_mm": spec.motor_center_x_mm,
            "extrusion_center_x_mm": spec.extrusion_center_x_mm,
            "m5_spacing_mm": spec.extrusion_2020.m5_spacing_mm,
        }
    pattern = getattr(spec, "pattern", None)
    return {"pattern": pattern.value if pattern is not None else "(none)"}


def _run_one(variant: Variant, trial_index: int) -> TrialOutcome:
    import time

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
            plate_width_mm=None,
            plate_depth_mm=None,
            motor_center_x_mm=None,
            extrusion_center_x_mm=None,
            m5_spacing_mm=None,
            interpreter_accepted=False,
            validator_passed=False,
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes=f"LLM generator error: {exc}",
        )

    summary = _summarise_spec(spec)

    if not isinstance(spec, AdapterPlateSpec):
        notes.append(f"model picked '{pattern_picked}' instead of adapter_plate")
        return TrialOutcome(
            trial_index=trial_index,
            variant=variant.tag,
            pattern_picked=pattern_picked,
            plate_width_mm=None,
            plate_depth_mm=None,
            motor_center_x_mm=None,
            extrusion_center_x_mm=None,
            m5_spacing_mm=None,
            interpreter_accepted=False,
            validator_passed=False,
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            notes="; ".join(notes),
        )

    # Did the model copy the canonical bench-003 defaults verbatim?
    default_dims = (65.0, 45.0, 11.0, -22.0, 20.0)
    emitted = (
        spec.plate_width_mm,
        spec.plate_depth_mm,
        spec.motor_center_x_mm,
        spec.extrusion_center_x_mm,
        spec.extrusion_2020.m5_spacing_mm,
    )
    if emitted == default_dims:
        notes.append("emitted the prompt's default dims verbatim (copy, not adapt)")
    else:
        notes.append(
            "adapted dims from defaults: "
            + ", ".join(
                f"{k}={summary[k]}"
                for k in (
                    "plate_width_mm",
                    "plate_depth_mm",
                    "motor_center_x_mm",
                    "extrusion_center_x_mm",
                    "m5_spacing_mm",
                )
            )
        )

    # Run the deterministic path: interpret + compile + validate.
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
            plate_width_mm=spec.plate_width_mm,
            plate_depth_mm=spec.plate_depth_mm,
            motor_center_x_mm=spec.motor_center_x_mm,
            extrusion_center_x_mm=spec.extrusion_center_x_mm,
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
        plate_width_mm=spec.plate_width_mm,
        plate_depth_mm=spec.plate_depth_mm,
        motor_center_x_mm=spec.motor_center_x_mm,
        extrusion_center_x_mm=spec.extrusion_center_x_mm,
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
    lines.append("# bench-003 perturbation experiment")
    lines.append("")
    lines.append(
        f"{n_trials} LLM trials per variant on synthetic Blueprints that diverge "
        "from the canonical bench-003 Blueprint along ONE dimension at a time. "
        "If the model just copies the prompt's documented defaults, the "
        "interpreter MUST reject the spec; if it adapts, the deterministic path "
        "still gives a passing part."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| variant | trials | adapted (dims != defaults) | interpreter accepted | validator passed | total cost |"
    )
    lines.append("|---|---|---|---|---|---|")
    for tag, runs in by_variant.items():
        adapted = sum(1 for r in runs if "adapted" in r.notes)
        accepted = sum(1 for r in runs if r.interpreter_accepted)
        passed = sum(1 for r in runs if r.validator_passed)
        cost = sum(r.cost_usd for r in runs)
        lines.append(
            f"| {tag} | {len(runs)} | {adapted}/{len(runs)} | {accepted}/{len(runs)} | "
            f"{passed}/{len(runs)} | ${cost:.4f} |"
        )
    lines.append("")
    lines.append("## Per-trial detail")
    lines.append("")
    for tag, runs in by_variant.items():
        lines.append(f"### {tag}")
        lines.append("")
        lines.append("| trial | plate WxD | motor X | T-nut X | m5 | interp | valid | notes |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in runs:
            dims = f"{r.plate_width_mm}x{r.plate_depth_mm}" if r.plate_width_mm is not None else "-"
            interp = "PASS" if r.interpreter_accepted else "REJECT"
            valid = "PASS" if r.validator_passed else ("FAIL" if r.interpreter_accepted else "n/a")
            lines.append(
                f"| {r.trial_index} | {dims} | {r.motor_center_x_mm} | "
                f"{r.extrusion_center_x_mm} | {r.m5_spacing_mm} | {interp} | {valid} | "
                f"{r.notes} |"
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
        help="Output-filename tag, e.g. 'postfix' -> perturbation-postfix-n5.{json,md}. "
        "Default (empty) writes perturbation-n5.{json,md}. Use this to keep old runs "
        "readable when re-running after a code change.",
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
            tag = (
                "PASS"
                if o.validator_passed
                else ("REJECT" if not o.interpreter_accepted else "FAIL")
            )
            print(
                f"  trial {i}: {tag}  ${o.cost_usd:.4f}  {o.duration_ms / 1000:.1f}s  -- {o.notes}"
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
