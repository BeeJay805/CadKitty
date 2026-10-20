# bench-004-door-hinge-v1

The first **assembly** benchmark (ADR-0010): a 2-leaf butt hinge that folds 0-180.
It is the smallest object that is simultaneously multi-body, a mechanism (1 revolute
DOF), a clearance problem (pin-in-bore running fit), and a spatial-interleave problem
(alternating knuckles) -- yet has an objective, hand-verifiable ground truth: *does
it actually rotate?*

The hinge is **composed from generic archetypes** (`knuckled_plate` x2 + `rod`), not a
hinge template. See `reference.assembly.json` for the known-good `AssemblyGraph`.

## Generate the printable STLs

```powershell
uv run forge-mind assemble benchmarks/bench-004-door-hinge-v1/reference.assembly.json
```

Writes three bodies to `benchmarks/results/_assembly/{leafA,leafB,pin}/*.stl` and runs
the relational validator (expect `OVERALL: PASS`). Those three STLs are the print package.

## Reference geometry (what to expect)

- Two leaves, 40 x 18 x 4 mm, interleaved knuckles (leafA: 2 outer, leafB: 1 centered).
- Bore 2.6 mm radius; pin 2.2 mm radius (solver-derived: bore - 0.4 mm running fit).
- Folded envelope ~ 40 x 30 x 10 mm.

## LLM generation (decontaminated probe)

The layer-4 front (`assembly/generator.py` + `prompts/assembly_v1.md`) lets the LLM
(Sonnet 4.6) compose the hinge from the archetype catalog. Decontaminated: the prompt
gives the catalog + relational *rules*, never the answer dimensions. The LLM picks
topology + knuckle counts + coarse sizes; the converter derives the interleaved
knuckle positions + notches; the solver derives the pin.

Measured trajectory (n=8 each, ~$0.04/run) -- "structure beats scale" on assemblies:

| schema the LLM emits | pass | dominant failure |
|---|---|---|
| raw knuckle coordinates | 0/8 | self-disconnect (LLM used a 0..len frame; the plate is centered) |
| counts, naive interleave | 2/8 | bad 2+3 interleave -> sweep collision |
| counts + clean interleave (current) | **8/8** | none |

The LLM's *topology* was correct from the first probe (interleave concept, notch
concept, coaxial + revolute wiring); reliability came entirely from moving the
relational *metric* (knuckle positions) out of the LLM and into derivation. Known
deterministic edge case: a 2+2 even-symmetric interleave collides in the sweep (the
LLM does not pick it; recorded as a test).

## Physical validation checklist (the real kill-gate)

Print the three STLs, push the pin through the interleaved knuckles, and record:

- [ ] printer / filament / nozzle / layer height
- [ ] orientation (leaves flat, knuckle axis along the bed) / supports (none expected)
- [ ] pin slides into the knuckles and is retained by friction
- [ ] hinge rotates **0 deg (folded) to 180 deg (open flat)** without binding
- [ ] no knuckle clash, no leaf-on-leaf collision through the sweep
- [ ] **photos**: folded, mid-sweep (~90 deg), open flat
- [ ] notes: clearance feels right? too loose / too tight? (running fit is 0.4 mm; tighten later)

A printed hinge that rotates is the first data point in this project that has ever
touched reality. If it binds where the validator said it sweeps free, the
sampled-collision model needs tightening -- record exactly where it catches.
