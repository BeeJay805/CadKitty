# local_breadth_v1 — the widened local-builder corpus (frozen 2026-07-18)

24 briefs for the LOCAL roster (qwen3-coder:30b author/repair + qwen2.5vl:7b critic), one per
part of the 24-part planner-A/B set (scripts/plan_ab.py: bench_agentic.PROMPTS base 12 +
bench_matrix._EXTENDED 12). Each DesignBrief was researched ONCE by the local planner
(qwen3-coder greedy) and frozen here so every later arm runs the SAME briefs — planner variance
is out of the measurement. SHA-pinned in MANIFEST.sha256; a corpus change is a new version dir,
never an edit.

Ship = run_agentic_pipeline stopped_reason "accepted" (validator green + vision critic
approves), max_iterations=6, PRINT_3D, FORBID_API_SPEND=1. Results are gitignored
(benchmarks/2026-07-18_local_breadth/ etc.); the corpus is the only committed artifact.
