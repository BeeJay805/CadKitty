"""CAD-Coder smoke bench (ADR-0112): de-risk the local text-to-CadQuery model against a frontier
baseline, scored by the EXISTING deterministic validator -- not by "did it execute".

The bar (founder-confirmed): **validator-verdict pass@1**. For each prompt we generate CadQuery,
run it through the exec bridge to an OCP solid, and run the 4-tier ``validate()``. A row PASSES
only if the validator's overall verdict is PASS; we also record the failing tiers so a
generate-but-garbage part reads as a failure, which is the whole point -- the validator is the moat,
so it is the scorer.

Two arms, same prompts, same scorer:
  * ``cadcoder``  -- ``CadCoderClient`` (local, raw CadQuery). Requires ``ollama serve`` + the model.
  * ``frontier`` -- ``make_client()`` structured, asked for CadQuery (Sonnet by default). Honors the
                    billing env (``FORGE_LLM_PROVIDER=claude_code`` => $0 subscription).

One command:  ``uv run python -m forge_mind.bench.cadcoder_smoke``
Writes:       ``benchmarks/analysis/cadcoder_vs_sonnet.md``  (the committed ablation table)
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic.cadquery_bridge import load_validation_model, run_cadquery_code
from forge_mind.config import SONNET
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.run import validate

# --------------------------------------------------------------------------------------------------
# Stratified prompt set: simple primitive -> multi-feature part -> hardest single part.
# Single-part only -- CAD-Coder's sweet spot and the only thing it was trained on. Kept small so a
# real run is minutes, not hours; each tier is one honest step up in construction difficulty.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    id: str
    tier: str  # simple | multi_feature | hard
    text: str


PROMPTS: tuple[Prompt, ...] = (
    Prompt("simple_block", "simple", "A rectangular block 40 x 20 x 10 mm."),
    Prompt(
        "simple_bored_disc",
        "simple",
        "A cylindrical disc 30 mm in diameter and 8 mm thick with a 10 mm diameter hole through the "
        "center.",
    ),
    Prompt(
        "mf_bracket",
        "multi_feature",
        "An L-shaped mounting bracket, 60 mm long and 40 mm tall, 4 mm wall thickness, with two 5 mm "
        "clearance holes on the vertical face and two on the horizontal face.",
    ),
    Prompt(
        "mf_flanged_boss",
        "multi_feature",
        "A round flange 50 mm in diameter and 5 mm thick with a raised central boss 20 mm in diameter "
        "and 12 mm tall bored 8 mm through, and four 4 mm bolt holes on a 38 mm circle.",
    ),
    Prompt(
        "hard_bearing_mount",
        "hard",
        "A bearing pillow block: a 70 x 30 mm base plate 8 mm thick with two 6 mm mounting slots, "
        "supporting an upright cylindrical bearing housing bored 22 mm to seat a 608 bearing, the "
        "bore axis 25 mm above the base and parallel to the long edge.",
    ),
)


class CadQueryCode(BaseModel):
    """Frontier-arm structured output: a short plan + a complete CadQuery script (bind ``result``)."""

    model_config = ConfigDict(extra="forbid")

    plan: str = Field(description="One or two sentences on the construction approach.")
    code: str = Field(
        description="Complete CadQuery script. `import cadquery as cq`; bind the final object to a "
        "top-level variable named `result`. Millimeters."
    )


_FRONTIER_SYSTEM = (
    "You are an expert CadQuery engineer. Produce a CadQuery script that builds the requested part "
    "and binds the finished object to a top-level variable named `result`. Import cadquery as cq. "
    "All units are millimeters."
)


class _Generator(Protocol):
    def __call__(self, prompt: str) -> str: ...


# --------------------------------------------------------------------------------------------------
# One scored trial: generate -> bridge-exec -> validate. Every failure mode is a recorded row, never
# a raised exception (one bad part must not abort the run).
# --------------------------------------------------------------------------------------------------

_VALIDATION_PROGRAM = PartProgram(
    metadata=Metadata(
        name="cadcoder_smoke", units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)
    ),
    operations=[Box(op_id="massing", mode=OpMode.ADD, width=1.0, depth=1.0, height=1.0)],
    constraints=Constraints(min_wall_thickness=1.0),
)


@dataclass
class TrialResult:
    arm: str
    prompt_id: str
    tier: str
    executed: bool = False
    validated: bool = False  # validator overall == PASS
    error_type: str | None = None  # generation | exec | export | validate | ...
    failing_tiers: list[str] = field(default_factory=list)
    wall_s: float = 0.0


def run_trial(arm: str, prompt: Prompt, generate: _Generator, work_dir: Path) -> TrialResult:
    res = TrialResult(arm=arm, prompt_id=prompt.id, tier=prompt.tier)
    start = time.monotonic()
    try:
        code = generate(prompt.text)
    except Exception as exc:
        res.error_type = f"generation:{type(exc).__name__}"
        res.wall_s = time.monotonic() - start
        return res

    job_dir = work_dir / f"{arm}_{prompt.id}"
    sandbox = run_cadquery_code(code, job_dir, part_name=prompt.id)
    if not sandbox.ok:
        res.error_type = sandbox.error_type or "exec"
        res.wall_s = time.monotonic() - start
        return res
    res.executed = True
    assert sandbox.step_path is not None and sandbox.stl_path is not None  # ok => artifacts on disk

    try:
        model = load_validation_model(sandbox.step_path, sandbox.stl_path)
        report = validate(model, _VALIDATION_PROGRAM)
    except Exception as exc:
        res.error_type = f"validate:{type(exc).__name__}"
        res.wall_s = time.monotonic() - start
        return res

    res.validated = report.passed
    if not report.passed:
        res.failing_tiers = sorted({str(c.tier) for c in report.checks if c.status.name == "FAIL"})
    res.wall_s = time.monotonic() - start
    return res


# --------------------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------------------


def cadcoder_generator() -> _Generator:
    from forge_mind.llm.cadcoder_client import CadCoderClient

    client = CadCoderClient()

    def gen(prompt: str) -> str:
        code, _usage = client.generate_cadquery(prompt)
        return code

    return gen


def frontier_generator(model: str = SONNET) -> _Generator:
    from forge_mind.llm.provider import make_client

    client = make_client()

    def gen(prompt: str) -> str:
        out, _usage = client.generate_structured(
            system=_FRONTIER_SYSTEM,
            user=prompt,
            output_model=CadQueryCode,
            model=model,
            max_tokens=2048,
        )
        return out.code

    return gen


# --------------------------------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------------------------------


def render_table(results: list[TrialResult]) -> str:
    arms = sorted({r.arm for r in results})
    lines = [
        "# CAD-Coder vs frontier — validator-scored smoke bench",
        "",
        "Bar: **validator-verdict pass@1** (generate → CadQuery exec bridge → 4-tier `validate()`; "
        "PASS only if the validator's overall verdict is PASS). Reproduce: "
        "`uv run python -m forge_mind.bench.cadcoder_smoke`.",
        "",
        "| arm | prompt | tier | executed | validator PASS | failing tiers / error | wall s |",
        "| --- | --- | --- | :---: | :---: | --- | ---: |",
    ]
    for r in results:
        detail = ", ".join(r.failing_tiers) if r.failing_tiers else (r.error_type or "")
        lines.append(
            f"| {r.arm} | {r.prompt_id} | {r.tier} | {'Y' if r.executed else 'N'} | "
            f"{'Y' if r.validated else 'N'} | {detail} | {r.wall_s:.1f} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| arm | exec pass@1 | validator pass@1 |")
    lines.append("| --- | ---: | ---: |")
    for arm in arms:
        rows = [r for r in results if r.arm == arm]
        n = len(rows)
        ex = sum(1 for r in rows if r.executed)
        va = sum(1 for r in rows if r.validated)
        lines.append(f"| {arm} | {ex}/{n} | {va}/{n} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------

_ARMS: dict[str, object] = {
    "cadcoder": cadcoder_generator,
    "frontier": frontier_generator,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="CAD-Coder validator-scored smoke bench")
    parser.add_argument(
        "--arms",
        default="cadcoder,frontier",
        help="comma-separated: cadcoder,frontier (default both)",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/analysis/cadcoder_vs_sonnet.md",
        help="markdown table output path",
    )
    args = parser.parse_args()

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    with tempfile.TemporaryDirectory(prefix="cadcoder_smoke_") as tmp:
        work_dir = Path(tmp)
        results: list[TrialResult] = []
        for arm in wanted:
            factory = _ARMS.get(arm)
            if factory is None:
                print(f"skip unknown arm {arm!r}")
                continue
            try:
                generate = factory()  # type: ignore[operator]
            except Exception as exc:
                print(f"skip arm {arm!r}: could not construct generator: {exc}")
                continue
            for prompt in PROMPTS:
                r = run_trial(arm, prompt, generate, work_dir)
                verdict = "PASS" if r.validated else ("exec-only" if r.executed else "FAIL")
                print(f"[{arm}] {prompt.id}: {verdict} ({r.wall_s:.1f}s) {r.error_type or ''}")
                results.append(r)

    if not results:
        print("no arms ran (need `ollama serve` for cadcoder, a provider for frontier)")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_table(results), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
