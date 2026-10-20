# arbitrary_v1 — general single-part benchmark

Measures **pass@1 of the general agentic single-part author** across the long tail of real
engineer requests: brackets, enclosures, knobs, clips, hooks, spacers, ducts, handles,
pulleys, cams, housings, guides, organizers, and mechanical coupling shapes.

**Score metric:** min-across-paraphrases per prompt, mean over all 32 prompts.  
Each prompt has a base variant plus two paraphrases (P1 reorders clauses, P2 swaps phrasing
and register); only the worst variant counts. A variant passes iff the engine gate holds
(status ok, validation.passed, confidence.basis == "validated") AND every rubric expectation
is verified against the produced artifacts via independent measurement.

**Freeze rule:** this benchmark is frozen as of 2026-06-11. Any change to any prompt, any
paraphrase, or any expect field constitutes a new benchmark version (arbitrary_v2, etc.).
Pass-rate scores are only comparable within a version and within a single provider setting;
cross-provider comparisons require api-provider runs, not subscription runs.

**Independence rule:** the CAD author model never sees these files. This benchmark was
authored by an independent rubric-author model (see `authored_by` in the JSON).
