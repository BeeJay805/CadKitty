# frontier_v1 — the print-mode frontier bench (Engine Month E2)

Measures **pass@1 of the agentic author in PRINT_3D mode** on the classes the engine
still FAILS (ADR-0012, `docs/ROADMAP.md` E0/E2): print-mode multi-hole brackets and
trays, thin-wall ducts/combs/trays, radial patterns (flange, indexer), and internal
bore junctions (manifold). 14 prompts:

- 3 golden-derived (nr200_ssd / nr200_hdd35 / meshify_duct briefs, TEST_LOG rows 37–38,
  with the resolved intake facts embedded so the bench drives `produce_part` directly;
  the duct carries the printable-envelope constraint W1 exposed in row 47),
- 2 arbitrary_v1 residual fails verbatim (row 33: `bracket_pipe_clamp_single`,
  `organizer_pen_cup_hex`),
- 3 audit-fixture families verbatim (`scripts/make_audit_fixtures.py`),
- 2 hard-holdout families (flange / indexer — the pass@8 HARD set classes),
- 4 fresh thin-wall / multi-hole-group prompts.

**Score metric:** pass@1, ONE variant per prompt (no paraphrases), mean over prompts.
A prompt passes iff the engine gate holds (status ok, validation.passed, confidence
basis "validated") AND every rubric expectation verifies against the artifacts by
independent measurement (`scripts/bench_arbitrary.py::_evaluate`). Exit gate
(ADR-0012): pass@1 >= 70% print-mode over 2 independent runs.

**Semantics (declared by the frozen content):** `fabrication_mode: print_3d`,
`hole_semantics: penetrations`, `bbox_semantics: sorted` (the ADR-0085 corrected
semantics — an envelope cap tests SIZE, not pose).

**Rubric self-consistency rule:** every expected hole diameter and every envelope cap
traces to a number stated verbatim in its own prompt; hole groups within a prompt are
tolerance-disjoint. The unit test enforces both.

**Freeze rule:** frozen 2026-07-09; `MANIFEST.sha256` pins the content (CRLF-normalized).
Any change to any prompt or expect field is a NEW version (frontier_v2). Scores compare
only within a version and provider setting; subscription runs are iteration-grade,
production decisions need the metered API.

**Independence rule:** the corpus/rubric author (claude-fable-5) is not the run arm —
frontier runs use the opus author arm (the model-economy default; Opus/Fable measured
DEAD EVEN as authors, TEST_LOG row 24), so the CAD author never grades itself.

**Run:**

```powershell
$env:FORGE_LLM_PROVIDER='claude_code'
uv run --no-sync python -u scripts/bench_passk.py benchmarks/frontier_v1/frontier_v1.json 1 opus
```
