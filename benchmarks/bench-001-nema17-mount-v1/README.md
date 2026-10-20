# bench-001-nema17-mount-v1

The first benchmark. A 3D-printable NEMA 17 stepper motor mount that bolts to a
2020 aluminium extrusion via two M5 T-nuts.

## Why this part

- **Standardized**: NEMA 17 specs are public and unambiguous, so the answer key
  is objective.
- **Two-sided**: it must present a correct interface on two faces (motor +
  extrusion), exercising the interface library on both axes.
- **Constraint-loaded**: 4mm walls (feature tier), no-support printing
  (printability tier), 60x60x40 envelope (mesh tier).
- **Cheap to ground-truth**: prints in ~30 min for ~$0.50; failures are
  inspectable by hand.

## Files

- `prompt.txt` — the free-text input a future LLM pipeline receives.
- `ground_truth.json` — the answer key: required dimensions, tolerances, and
  interfaces.
- `pass_criteria.yaml` — trial/benchmark pass conditions + the reference DSL for
  the `--use-handwritten` bypass.

## Running

```powershell
# Phase 3: bypass the (not-yet-built) LLM, feed the reference DSL straight to
# compile + validate. Establishes the measurement floor.
uv run forge-mind bench bench-001-nema17-mount-v1 --use-handwritten --trials 10
```

## Target

9/10 trials pass, mean wall time < 60s, mean cost <= $0.30/trial. With the
handwritten bypass the target is 10/10 (the reference DSL is known-good), proving
the harness measures what we expect before any model is involved.
