# Two incomplete inline-four CAD studies

Both runs delivered saved CAD files and current instructions. Neither run established complete engine function or a full-task PASS. The main task assessment remains **unassessed**, and physical readiness remains **unverified**.

## Matched request

Both runs used this exact prompt:

> Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers. Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z. Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype. Use conventional milling, turning and drilling. No combustion qualification is requested.

Each run used reported model `gpt-6-astra`, Docker CAD execution, a 1,800-second budget, and a 60-action limit. Plan approval and printable-output requests were disabled. Each run interpreted the prompt afresh; the frozen requirement lists were different. These are two observations, one per software condition, not a controlled estimate of reliability or causal speed improvement.

## Recorded results

| Observation | Earlier baseline004 | Later candidate001 |
| --- | ---: | ---: |
| Valid saved solid occurrences | 16 | 31 |
| Made-part definitions | 8 | 9 |
| All part definitions, including purchased parts | 8 | 11 |
| Recorded assembly extents, X × Y × Z mm¹ | 168 × 84 × 104 | 172 × 83 × 95 |
| Passing scoped nominal mate checks | 12 of 12 | 16 of 16 |
| Recorded model calls, all stages | 23 | 18 |
| Summed model-call duration, seconds | 1,290.094 | 1,296.111 |
| Project creation to attempt journal finish, seconds | 1,491.138 | 1,484.563 |
| Current instructions and verified files delivered | Yes | Yes |
| Project state / accepted | needs_review / No | needs_review / No |
| Frozen requirements | 20 | 21 |
| Build-time requirement findings | 20 unverified | 3 envelope pass; 18 unverified |
| Original final proof | 20 unverified results | Unavailable; no final results |
| Full-task success assessment | Unassessed | Unassessed |

¹ Displayed dimensions are rounded. The companion JSON retains the recorded values and assembly STEP hashes. Nominal mate checks do not prove practical assembly, retention, physical fit, or continuous motion.

Fewer calls did not reduce summed model time in this pair. Project-to-attempt-finish elapsed time was slightly lower in the candidate. Neither observation attributes a speed change to a particular repair. This elapsed interval includes final readback after the first observed terminal state. Qualification occurred before project creation. Benchmark, adapter, and project clocks have different boundaries; they must not be mixed to infer exact overhead.

## Remaining engineering work

- Complete matching rod caps and their fasteners.
- Provide and check shaft, wrist-pin, and bushing retention.
- Model a crankshaft-to-camshaft timing connection.
- Bind a coupled mechanism to the saved assembly and obtain the required cycle evidence.
- Resolve lubrication continuity, machining feasibility, tolerance stacks, assembly access, and service removal.

Strength, durability, manufacture, lubrication adequacy, and physical hand-turnability remain unverified. The candidate's original missing report was a mixed-contract reporting defect. A later regression-tested repair retains the 18 unknown duties; it does not retroactively change the original result or complete the engine.

[Machine-readable evidence summary](ENGINE_COMPARISON-public002.json) includes prompt identity, request controls, exact recorded dimensions, artifact hashes, timing definitions, and scope. This comparison uses saved audits and artifact identities. It performs no new geometry measurement or model generation.
