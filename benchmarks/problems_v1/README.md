# problems_v1 — the Problem Solver intake bench (ADR-0087 S7)

Measures the INTAKE half of the Problem Solver (ADR-0087, S6b surface): problem → decompose →
gather (registry / cited research / one typed answer round) → confirmable spec | refusal. This
bench is **THE GATE**: `FORGE_PROBLEM_MODE` does not turn on until it passes.

**14 prompts** spanning §6's classes: route-spec (P17), registry-only, needs-research (3 golden,
incl. the frozen SSD acceptance scenario), needs-clarify, ambiguous-variant (P2),
registry-dropdown, must-refuse (L5).

## Scored metrics (deterministic scorer: `forge_mind.invention.bench_problems`)

| Metric | Gate? | Definition |
|---|---|---|
| **invented-fact rate** | **must be 0** | Every ledger fact must trace: `user` → a canned answer; `registry` → a committed cited registry row value; `researched` → quote verbatim in this bench's corpus + value+unit in quote + citation = the doc URL. Anything else is INVENTED. |
| **refusal correctness** | **must be 100%** | Every must-refuse prompt refuses naming the expected gap; NO solvable prompt refuses. |
| route accuracy (P17) | report | `classify_intake` matches `expected_route`; spec prompts route byte-untouched. |
| missing-branch rate (P16) | report | `expected_domains ⊆ tree.touched_domains()` per prompt. |
| fact recall / precision | report | Recall: expected load-bearing facts closed as expected (source class + value for sourced ones). Precision: resolved sourced facts that are expected (noise visible). |
| question discipline (P5) | report | Round-1 form fields ≤ `question_budget`; no NUMBER question for a fact the seeded registry already closes. |
| thermal ceiling (P14) | report | When `thermal_airflow` is touched, the honesty ceiling rides the assumptions verbatim. |
| golden end-to-end fit | report | For the 3 golden prompts with `--build-goldens`: the real engine builds the confirmed spec and the compiled env checks pass. |

## The canned user (answer policy)

One answer round, per the 2026-07-06 addendum. The runner answers every round-1 field: numbers by
fact name from `answers.facts`; dropdowns from `registry_choices`; the variant from
`variant_choice`; free-text asks with `default_text`. **Solvable prompts carry
`fallback_number`** — a real owner can measure their own hardware, so any uncovered number
question gets that measurement (over-asking stays visible via `question_budget`, not via fake
refusals). **Must-refuse prompts have no fallback**: their `unknown_facts` (and anything
uncovered) answer `unknown=true`, so refusal correctness measures refusal LOGIC. (Run-2
shakedown finding: without the fallback, 7 solvable prompts refused purely because the model
asked reasonable measurements my fixed answer map hadn't anticipated — that measured my
authoring, not the pipeline.)

## Corpus rule

Research corpora are **canned fixtures** (`bench://` URLs, `corpora/*.json`) — they measure the
pipeline's honesty *against a known corpus*, not the corpus against the world. Their values are
plausible but fictional; nothing here is a real product claim. Conflict-resolution (P10) is
covered by unit tests (`test_problem_intake.py`) and the S8 edit flow, not by this bench: the
single-pass gather structurally cannot produce a same-name double entry from one query.

## Freeze rule

Frozen as of 2026-07-06 (MANIFEST.sha256 covers the prompt file + every corpus; a unit test
recomputes it). Any change to any prompt, expectation, answer, or corpus is a NEW version
(problems_v2); scores are comparable only within a version and a provider setting.

## Independence rule

Authored by claude-fable-5 (the bench author). The runtime decompose/research brain
(`BRAIN_MODEL`, Opus by default) never sees these files. Registry expectation values are copied
from the committed cited registries (`dsl/standards/computer.py`), never from model memory.

## Running

```bash
# live (subscription: FORGE_LLM_PROVIDER=claude_code; or metered with ANTHROPIC_API_KEY)
uv run python scripts/run_problems_v1.py                 # intake-only, all 14 prompts
uv run python scripts/run_problems_v1.py --build-goldens # + the 3 real golden builds
uv run python scripts/run_problems_v1.py --only nr200_ssd
```

Results land in `benchmarks/results/problems_v1_<UTC-stamp>/` (gitignored); every run gets a
`docs/TEST_LOG.md` row the day it happens.
