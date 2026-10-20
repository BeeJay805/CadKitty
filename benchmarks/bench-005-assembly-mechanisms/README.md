# bench-005-assembly-mechanisms

The first benchmark of the **general** assembly path (ADR-0010): the LLM composes THREE
mechanism families -- a revolute **hinge**, a prismatic linear **slide**, and a 2-bar
serial **linkage** (a kinematic chain) -- from one generic archetype catalog
(`knuckled_plate`, `rod`, `tube`, `flat_link`) wired by typed joints. It proves the
assembly spine is reliable AND not a single-mechanism template.

## What it measures

Per **decontaminated** prompt (describes the mechanism, never the answer dimensions),
N trials of `request -> AssemblyGraphGenerator (LLM) -> solve -> compile ->
validate_assembly`. A trial passes only if every stage passes. Reported with Wilson
95% lower bounds + cost. A hard cost ceiling stops the run cleanly.

## Run

```powershell
uv run forge-mind assemble-bench
```

## Result (Sonnet 4.6, n=8 x 6 paraphrased prompts, decontaminated)

| prompt | pass | Wilson 95% LB |
|---|---|---|
| hinge_butt | 8/8 | 68% |
| hinge_door | 5/8 | 31% |
| slide_piston | 8/8 | 68% |
| slide_telescope | 8/8 | 68% |
| linkage_arm | 8/8 | 68% |
| linkage_chain | 7/8 | 53% |
| **OVERALL** | **44/48** | **80%** |

Gate (>= 70% overall Wilson LB): **PASS** (~$0.35). Two wordings per family. The weak
spot is `hinge_door`: now that the catalog includes `flat_link`, the phrase "flat leaves"
sometimes lures the LLM to a linkage bar instead of a knuckled hinge leaf -- an honest
cost of a richer catalog (sharpening the prompt recovered it from 0/8 to 5/8).

The LLM picks the topology + archetypes + knuckle counts + coarse sizes; deterministic
code derives every relational metric (interleave positions + notches, pin span, piston
length + travel) and the solver derives the rod radius from the bore. The LLM never
emits a coordinate -- "structure beats scale" on assemblies, across two mechanism types.

## Honest scope

Three mechanism families, six paraphrased prompts. The weak spot (hinge_door) is a
catalog-ambiguity, not a geometry, failure. Not yet stress-tested with adversarial
wording or harder geometries, and physical prints are not yet done -- the one test that
touches reality.
