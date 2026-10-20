"""bench-002 paraphrase experiment — does the 30/30 decontaminated pass
generalise across reworded prompts?

bench-002 was the one of three benches that survived the decontamination
of `layout/prompts/layoutspec_v2.md`: 30/30 on the canonical prompt at
n=30. The honest follow-up: is that 30/30 specific to the canonical
prompt's exact wording, or does the LayoutSpec path still produce a
passing electronics-tray spec when the user asks for the same functional
part in different words?

Three paraphrased prompts (no original-prompt phrases copied), 10 trials
each. Each trial runs the full pipeline: NL prompt -> Intent -> Blueprint
-> LayoutSpec (Sonnet) -> deterministic interpreter -> compiler ->
validator. Records pass/fail + chosen pattern + chosen board id + LLM
spend + duration. Writes outcomes to
`benchmarks/analysis/bench-002-pi4-case-v1/paraphrase-n{N}.{json,md}`.

Cost estimate: ~$0.022/trial x 30 trials = ~$0.66.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from forge_mind.analysis.run import load_expectations
from forge_mind.layout.bench import run_layoutspec_best_of_k
from forge_mind.llm.client import api_key_available

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks" / "bench-002-pi4-case-v1"
OUTPUT_DIR = REPO / "benchmarks" / "analysis" / "bench-002-pi4-case-v1"

DEFAULT_N_TRIALS = 10


# Three paraphrased prompts that ask for the same functional part (a Pi 4
# open-top tray with port openings) without copying the canonical bench-002
# prompt's phrasing. The mechanical facts the LLM cannot infer otherwise
# (board family, envelope, wall thickness, material) are still present;
# only the wording style varies.
PROMPTS = [
    {
        "tag": "hobbyist",
        "prompt": (
            "I need an enclosure for my Pi 4. Open top so I can still poke at it, "
            "vent slots underneath, 2 mm walls in PLA. Must fit in 95 x 70 x 30 mm. "
            "Standard Pi 4 mounting holes."
        ),
    },
    {
        "tag": "bullet-spec",
        "prompt": (
            "Pi 4B tray. Open top. PLA. Floor + 4 corner standoffs for the "
            "M2.5 board screws. Cutouts on the long side for USB / Ethernet, "
            "cutout on the short side for USB-C and HDMI. Slot ventilation. "
            "Max bbox 95 x 70 x 30 mm. 2 mm walls."
        ),
    },
    {
        "tag": "functional",
        "prompt": (
            "Build a printable mount that holds a Raspberry Pi 4 board "
            "(the Model B, with the standard 58 x 49 mm mount rectangle) "
            "inside a 95 x 70 x 30 mm volume. Floor with corner posts the "
            "board screws onto, walls around the sides with openings for the "
            "port banks, slot ventilation. 2 mm wall thickness, PLA, prints flat."
        ),
    },
]


@dataclass
class ParaphraseOutcome:
    variant: str
    trial_index: int
    passed: bool
    failed_check_ids: list[str]
    cost_usd: float
    duration_ms: float
    error: str | None


def _run_one(variant_tag: str, prompt: str, trial_index: int, work: Path) -> ParaphraseOutcome:
    expectations = load_expectations(BENCH_DIR)
    t0 = time.perf_counter()
    try:
        result = run_layoutspec_best_of_k(
            prompt,
            work,
            k=1,
            envelope_cap_mm=expectations.envelope_cap_mm,
            render=False,
        )
    except Exception as exc:
        return ParaphraseOutcome(
            variant=variant_tag,
            trial_index=trial_index,
            passed=False,
            failed_check_ids=[],
            cost_usd=0.0,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )
    winner = (
        result.candidates[result.winner_id]
        if result.winner_id is not None and result.candidates
        else None
    )
    return ParaphraseOutcome(
        variant=variant_tag,
        trial_index=trial_index,
        passed=result.best_passed,
        failed_check_ids=(winner.failed_check_ids if winner else []),
        cost_usd=result.total_cost_usd,
        duration_ms=result.total_duration_ms,
        error=result.error,
    )


def render_markdown(n_trials: int, outcomes: list[ParaphraseOutcome]) -> str:
    by_variant: dict[str, list[ParaphraseOutcome]] = {}
    for o in outcomes:
        by_variant.setdefault(o.variant, []).append(o)

    lines: list[str] = []
    lines.append("# bench-002 paraphrase experiment")
    lines.append("")
    lines.append(
        f"{n_trials} LLM trials per paraphrased prompt. Tests whether the bench-002 "
        "decontaminated 30/30 generalises across reworded prompts, or is specific to "
        "the canonical bench-002 prompt's exact phrasing."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| variant | trials | passed | mean cost | mean latency |")
    lines.append("|---|---|---|---|---|")
    for tag, runs in by_variant.items():
        passed = sum(1 for r in runs if r.passed)
        mean_cost = sum(r.cost_usd for r in runs) / len(runs) if runs else 0.0
        mean_dur = sum(r.duration_ms for r in runs) / len(runs) if runs else 0.0
        lines.append(
            f"| {tag} | {len(runs)} | {passed}/{len(runs)} | ${mean_cost:.4f} | "
            f"{mean_dur / 1000:.1f}s |"
        )
    lines.append("")
    lines.append("## Per-trial detail")
    lines.append("")
    for tag, runs in by_variant.items():
        lines.append(f"### {tag}")
        lines.append("")
        lines.append("| trial | passed | failed_check_ids | cost | latency | error |")
        lines.append("|---|---|---|---|---|---|")
        for r in runs:
            v = "PASS" if r.passed else "FAIL"
            checks = ",".join(r.failed_check_ids) if r.failed_check_ids else "-"
            err = r.error if r.error else "-"
            lines.append(
                f"| {r.trial_index} | {v} | {checks} | ${r.cost_usd:.4f} | "
                f"{r.duration_ms / 1000:.1f}s | {err} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not api_key_available():
        raise SystemExit("ANTHROPIC_API_KEY not set; this experiment makes real LLM calls.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS, help="trials per variant")
    args = parser.parse_args()
    n = args.trials

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_outcomes: list[ParaphraseOutcome] = []
    with tempfile.TemporaryDirectory() as td:
        for p in PROMPTS:
            print(f"\n=== {p['tag']} ===")
            for i in range(n):
                work = Path(td) / p["tag"] / f"trial-{i}"
                work.mkdir(parents=True, exist_ok=True)
                o = _run_one(p["tag"], p["prompt"], i, work)
                all_outcomes.append(o)
                label = "PASS" if o.passed else ("ERR" if o.error else "FAIL")
                print(
                    f"  trial {i}: {label}  ${o.cost_usd:.4f}  "
                    f"{o.duration_ms / 1000:.1f}s" + (f"  -- {o.error}" if o.error else "")
                )

    json_path = OUTPUT_DIR / f"paraphrase-n{n}.json"
    md_path = OUTPUT_DIR / f"paraphrase-n{n}.md"
    json_path.write_text(
        json.dumps(
            {
                "n_trials_per_variant": n,
                "prompts": PROMPTS,
                "outcomes": [asdict(o) for o in all_outcomes],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(n, all_outcomes), encoding="utf-8")
    print(f"\nwrote {json_path.relative_to(REPO)}")
    print(f"wrote {md_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
