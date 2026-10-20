# arbitrary_v1.1 — errata-corrected successor to arbitrary_v1

Same 32 prompts, **byte-identical** to arbitrary_v1 (prompts and paraphrases untouched);
only rubric `expect` blocks and scoring semantics changed, correcting five defects that were
verified against real artifacts in the 2026-07-04 engine deep review
(`docs/research/2026-07-04-engine-deep-review.md` §M5, TEST_LOG rows 32–33, ADR-0085):

1. **mech_detent_wheel_12pos** — bbox Z cap 13 → 26 (arithmetic: plate 8 + hub 8 = 16 mm
   stacked nominal; the v1 cap margined 8 alone).
2. **hook_j_utility_wall** — height cap 85 → 128 (founder ruling: "up-leg 30 mm tall above
   the plate" reads STACKED → 80 mm nominal).
3. **cam_indexing_template_plate** — geometric expect replaced by `expect_refusal`: the spec
   is mathematically infeasible, so the CORRECT engine outcome is a pre-authoring refusal
   naming the arithmetic (ADR-0084). Building anything is the failure.
4. **hole counting: stacks → penetrations** — an opposed 180° pair of real holes is one
   coaxial stack but two penetrations; v1 made `min_count: 2` unsatisfiable-by-construction.
5. **bbox compare: per-world-axis → sorted** — an envelope cap tests SIZE, not pose; v1's
   per-axis compare failed prompt-faithful parts over incidental build orientation.

The full arithmetic and evidence live in `errata_from_v1` inside the JSON.

**Scoring/freeze/independence rules are inherited from arbitrary_v1** (min-across-paraphrases,
frozen by MANIFEST.sha256 — the runner refuses on drift, the CAD author model never sees these
files). Scores compare only WITHIN a version: v1 history stays v1, v1.1 runs are labelled
(`"benchmark": "arbitrary_v1.1"` in results; results dirs are prefixed `arbitrary_v1_1_`).
Select at run time with `FORGE_BENCH_VERSION=v1.1` (default remains v1).

Measured on the same post-#141 artifacts (TEST_LOG row 33): v1 25/32 = 78% → v1.1
**30/32 = 94%**; the two residual failures are genuine engine misses, and a v1-semantics
replay through the same checker reproduces 25/32 exactly (byte-compatibility guard).
