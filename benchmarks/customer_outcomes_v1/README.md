# Frozen customer outcome corpus v1

This corpus contains author-written development requests, not observed customer demand. Its frozen cases, photo attribution, and reference bytes are inputs to an evaluation harness. Keeping them in a source snapshot does not establish a current generation result.

`MANIFEST.json` pins `cases.json`, `ATTRIBUTION.json`, and the washer photograph. These files remain unchanged. The photo's source, exact revision, and public-domain attribution are in `ATTRIBUTION.json`. Request dimensions come from text; the photograph supplies no measured dimensions.

## Available and unavailable parts

The snapshot includes the corpus, benchmark/report harnesses, and pure regression controls. It does not include the old author's local environments, private trial directories, qualified worker-image digests, credentials, or raw model transcripts. Historical machine-specific launch commands have been omitted from this public guide.

A real trial needs a separately qualified native worker, the selected immutable source and matching engine identity, an explicitly configured model account, and a new output directory. Review time and account usage before running it. Read the current CLI options from the repository root:

```powershell
uv run --locked --all-extras python -m scripts.customer_outcome_bench --help
uv run --locked --all-extras python -m scripts.customer_outcome_report --help
```

These help commands do not constitute a trial or a result. The normal local MCP setup is in the repository README. Do not copy private image identifiers or substitute an unqualified worker to manufacture a benchmark success.

Source archives can run without Git metadata. The journal records a null `git_head` and explicit metadata unavailability. Exact source-file hashes still identify the engine and harness. Actual archive trials reject absent or incomplete inventories; existing repository errors and source drift still refuse. A recorded commit does not prove a clean worktree.

## Result scope

The harness separates file delivery, elapsed time, model calls, unresolved duties, and supported independent STEP observations. Finite geometric sampling does not prove exact shape equivalence, physical fit, strength, safety, or every English requirement. Unsupported independent oracles remain unassessed. The benchmark never fills a missing customer handoff by creating a new export afterward.

Development and held-out groups remain separate. Failures remain in denominators. A held-out case that informs a repair becomes development evidence for that lineage. Synthetic FixtureAdapter controls test harness mechanics; they never count as actual generation outcomes.

Raw run data can contain private input and worker output. The report exporter uses a public allowlist, but an integrator must still inspect every intended publication. Use a new output directory when source, corpus, runtime, or selection changes. No fresh benchmark outcome is claimed by this README.
