# bench-006-fan-mount-v1

The first benchmark of a **fan mount** — the first new hardware CLASS added to the
component library (ADR-0011). A flat plate mounts a 60 mm axial cooling fan to a 2020
extrusion: a `fan_face` (4 M3 corner bolts on a 50 mm square + a central airflow bore)
on the +Z face, a 2020 M5 T-nut pair through to the -Z face.

## What it proves

A fan mount reuses the **existing `adapter_plate` pattern** (bench-003) with the +Z
interface swapped for a `fan_face` — **no 4th pattern**. The fan's facts come from a
new `FanComponent` registry record, parallel to `BoardComponent`: two component
classes (boards + fans) now run through one registry mechanism. The LLM picks only the
fan id (`fan_60mm`); the interpreter derives every coordinate, the plate dims, and the
airflow bore.

## Run

```powershell
# untuned A/B (LayoutSpec vs direct DSL)
uv run forge-mind abtest bench-006-fan-mount-v1 --generation-path layoutspec --trials 10

# full prompt -> STL end-to-end (the product number)
uv run forge-mind analyze bench-006-fan-mount-v1 --trials 30 --layoutspec
```

## Scope / known limitation

This bench targets the **60 mm** fan. A flat **40 mm** fan mount is geometrically
infeasible at a 4 mm wall: the ~Ø38 airflow bore leaves only ~2.03 mm to the M3 corner
bolts (below a 4 mm min_wall). The interpreter rejects it up front with the
bore-to-bolt diagnostic; it is feasible at a 2 mm wall. This is a documented
limitation (ADR-0011 / Stage 1), not a benchmark target — a real geometric constraint
of small flat fan mounts, surfaced honestly rather than fudged.
