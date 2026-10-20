# bench-007-actuated-arm-joint-v1

The project's **demo unit cell**: the first ACTUATED joint (ADR-0012, tier 1). A
NEMA 17 stepper drives an output arm that rotates about its shaft axis — a base
bracket holds the motor, the arm grips the 5 mm shaft and revolves about it.

## What it proves

The assembly track composes a **non-printed hardware node** (the motor — frames +
bbox, no geometry; from the hardware-node spike) with a new driven **`output_arm`**
archetype and a new **`shaft_coupling`** interface (a running-fit grip bore + radial
M3 set screw — the *opposite* of the 22 mm face clearance bore) into a statically
valid actuated joint whose revolute is **coaxial with the real shaft** — and the
coaxiality falls out of the existing frame-mate solver for free. No new LayoutSpec
pattern (this is the assembly track); reuses the solver + 5th-tier assembly validator.

## Run

```powershell
# untuned / end-to-end LLM A/B (tier-1 scoring)
uv run forge-mind assemble-bench benchmarks/bench-007-actuated-arm-joint-v1
```

## Tier-1 scope (static)

Validated: per-body shells (bracket + arm), the shaft coupling (grip bore + set
screw), and the revolute coaxial with the motor shaft. The arm is cantilevered off
the shaft (no far-end bearing).

**Deferred (not graded in tier 1):** the powered **sweep** of the arm vs the motor
body. The revolute is anchored to the non-printed motor, which has no collision mesh,
so `assembly.range_of_motion` WARNs — accepted as a known-deferral via the tier-1
scoring (`passes_criteria`), never a phantom failure and never a fake "clear." A
trustworthy powered sweep needs a motor collision mesh and stays for a later tier,
alongside the bearing-supported far end, 2-DOF chaining, the D-bore coupling, and the
physical-print kill-gate.
