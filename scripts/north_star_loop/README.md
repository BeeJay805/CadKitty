# Archived north-star acceptance tooling

This directory retains historical acceptance specifications and their checker. It does not record current generation capability or a reproduced public result.

`north_stars.json` and `north_stars.lock.json` remain byte-identical to the reviewed source. `tests/unit/test_north_star_acceptance.py` requires both files and verifies each acceptance-condition hash. The pinned conditions must not be weakened to turn an unavailable or failed run into a pass.

## Availability

| Entry | Retained source reference | Public snapshot status |
| --- | --- | --- |
| `inline4` | A historical absolute path ending in `run_i4_v2.py` | Runner and its historical run evidence are not included; the `passing-baseline` label is archival metadata |
| `planetary` | A historical absolute path ending in `run_planetary.py` | Runner and its historical run evidence are not included; the label describes an old target |
| `strandbeest` | `TBD (author via discover then freeze)` | No frozen prompt runner is supplied |
| `arm_link_v1` | `scripts/arm_search/analytic_link.py` with specified reference inputs | Script source is included; its native/FEA/slicer prerequisites and result are not qualified by retaining the script |

The two absolute Windows paths are preserved provenance strings, not portable commands. They are not searched for or executed by the acceptance guard tests. The checker evaluates a supplied report; it does not run those prompt references. A matching acceptance hash proves that the conditions did not change. It does not establish the origin or truth of a report.

The old `status`, `note`, TEST_LOG row references, and numerical targets are historical specification data. They do not describe a freshly measured release. In particular, the archived inline-four entry is not evidence that this public snapshot can generate a complete engine.

## Retained and excluded files

The snapshot retains `acceptance.py`, the registry, and its lock for regression tests. It excludes private agent instructions and account-usage tooling. The current public security and contribution documents govern use of this snapshot.

Five historical result reports are excluded from the public candidate: structural proof, engineering lab, MCP workspace, and both design-study reports. Their benchmark input files can remain useful without promoting their old result statements. No replacement current results were invented.

Before reporting a new result, publish an exact source revision, immutable inputs, relevant environment, scoped output evidence, and known limits. Do not run `acceptance.py lock` merely to repair a failing test or unavailable run.

## Optional utilities omitted from the public snapshot

Six unreferenced historical utilities use author-machine output or archive defaults: `compose_from_description.py`, `measure_bases.py`, `measure_fidelity.py`, `measure_trio.py`, `measure_layout.py`, and `mine_repairs_archive.py`. They are excluded from the public candidate only. The private originals remain intact. Text and import searches found no consumers in source, tests, or other scripts.

Other retained scripts can contain dated command examples. Their required output/input arguments must refer to the caller's own files. Those examples do not establish that the referenced archive exists or that a current result was reproduced. The public quickstart remains the repository README.
