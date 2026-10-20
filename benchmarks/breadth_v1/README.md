# breadth_v1 — the frozen breadth benchmark (pre-registered)

The measured definition of "CadKitty got broader." 20 prompts across 4 classes
(4 arm, 6 articulated_assembly, 2 static_assembly, 8 single_complex), each with 2
paraphrases by a fixed rule. Authored 2026-06-10 by claude-sonnet-4-6 acting as the
independent rubric author; the CAD-authoring models (claude-opus-4-8, candidate
claude-fable-5) never see or edit these files — prompts reach the engine one at a
time at generation, exactly as a user's would.

## Protocol (fixed)

* **pass@1**: each variant (base + 2 paraphrases) runs ONCE through the product path
  (`api.core.generate_part`, build cache off). No retries, no best-of.
* A prompt's score is the **MIN across its 3 variants**; the headline is the mean
  over the 20 prompts.
* A variant **passes** iff the engine's own claim holds (status ok, validation
  passed, `confidence.basis == "validated"`) AND every rubric expectation verifies
  against the artifacts, re-measured independently by `scripts/bench_breadth.py`
  (GLB byte-level node/extras census, STEP text census, B-rep hole measurement,
  trimesh mesh checks). Mount expectations check the **cited part's real pattern**
  from `mount_patterns.json`. An articulated-class artifact must carry joints whose
  motion claim is EARNED (`ck_joint.checked == "swept"`).
* **Freeze**: `MANIFEST.sha256` pins both content files; the scorer refuses to run
  on drifted content. Any change = a new versioned benchmark (breadth_v2, ...);
  before/after deltas only count within one version.
* **Provider pairing** (pre-registered before any run): results record `provider`
  ("api" = the production engine path, "claude_code" = the subscription provider).
  A BEFORE/AFTER delta counts ONLY between runs with the SAME provider; subscription
  runs are working measurements (cheap iteration at larger n), and any delta cited
  for promotion or an ADR must be confirmed on an "api" pair.

## Runs

| date | engine | pass@1 | per-class | spend | results |
|---|---|---|---|---|---|
| (BEFORE pending) | main @ 602e1e4 + bench harness | — | — | — | — |
