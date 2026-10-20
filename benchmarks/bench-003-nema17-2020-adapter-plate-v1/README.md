# bench-003-nema17-2020-adapter-plate-v1 (DETERMINISTIC REFERENCE ONLY)

Tests **interface composition on a single flat element**: a NEMA17 motor face and
a 2020 extrusion M5 T-nut interface co-hosted on one plate, both reusing the
bench-001 facts + checkers with zero new validator code. See
[ADR-0008](../../docs/decision_log/0008-bench003-plan.md) for the planning trail.

## Why this benchmark

bench-001 and bench-002 both saturated at 100% on the LayoutSpec path, and the
research-loop automation (`forge-mind analyze-next`) flagged both as needing
coverage, not tuning. bench-003 expands coverage along the architectural seam
ADR-0006 was built around: composing two known interfaces on one element,
without adding to the template graveyard.

## Files

- `prompt.txt` — plain-English part request (LLM input for the future A/B).
- `ground_truth.json` — mechanical facts + required interfaces + hard
  constraints; `load_expectations` reads envelope + min wall + interfaces.
- `pass_criteria.yaml` — required validator check ids + trial budgets. The
  reference DSL fixture lives at
  `tests/fixtures/handwritten/nema17_2020_adapter_good.partdsl.json`.

## Status

- **Deterministic reference DONE** — `examples/handwritten_nema17_2020_adapter.py`
  builds a `PartProgram` that compiles and passes every entry in
  `pass_criteria.yaml: required_checks`. Verified by
  `tests/integration/test_bench003_handwritten.py`.
- **NOT done yet:** `AdapterPlateSpec` + `AdapterPlateInterpreter` + the
  `Pattern.ADAPTER_PLATE` enum + the `place_m5_pair` axis generalisation +
  the LLM A/B run. Each is gated on this reference passing first.

## Geometry summary

Flat plate, origin at center, Z=0 on the build plate:

- Plate body: 65 x 45 x 4 mm.
- Motor face (`nema17_face`, normal +Z) centered at (+11, 0, top): four M3
  clearance holes on the 31 mm square + a 22 mm shaft bore, all drilled along Z.
- Extrusion mount (`extrusion_2020_slot`, normal -Z) centered at (-22, 0, top):
  two M5 T-nut clearance holes spaced 20 mm along Y, drilled along Z.
- Every hole keeps >= 4 mm land to the nearest plate edge and >= 4 mm
  hole-to-hole solid in-plane.
- Prints flat on the bed: only the cylindrical hole walls have any vertical
  surface (the printability checker is plane-only and skips them).
