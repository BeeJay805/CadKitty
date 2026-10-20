# bench-002-pi4-case-v1 (PLANNED)

Generalisation test for the LayoutSpec architecture (ADR-0004 → ADR-0005). Target
part: a Raspberry Pi 4 Model B open-top case. This folder is **spec/design only** —
the `pi4_case` interpreter and the `pi4_mount` / `pi4_ports` interface checkers are
not implemented yet.

## Why this benchmark

bench-001 (NEMA 17 L-bracket) is a flat-plate bracket. A Pi 4 case is an
enclosure: a floored tray with standoffs, perimeter walls, port cutouts, and
ventilation. If the LayoutSpec → deterministic interpreter approach still wins
here without bench-001-specific tuning, the architecture generalises. If it
needs heavy per-pattern special-casing, that is the measured cost of
generalisation — and a finding either way.

## Files

- `prompt.txt` — plain-English request (the LLM front end's input).
- `ground_truth.json` — mechanical facts + required interfaces + hard
  constraints (load_expectations reads envelope, wall, required_interfaces).
- `expected_interfaces.json` — the `pi4_mount` + `pi4_ports` interfaces to build.
- `pass_criteria.yaml` — target thresholds; untuned bar is 6/10.

## Design target

A hand-authored LayoutSpec stands in for the future interpreter:
`tests/fixtures/layout/bench002_pi4_case.layoutspec.json`. The drafted spec model
is `src/forge_mind/layout/pi4_spec.py` (`Pi4CaseSpec`). The full design plan,
generalisation analysis, and feasibility checklist are in
`docs/decision_log/0005-bench002-pi4-layoutspec.md`.

## Foundation built (deterministic validation)

- `pi4_mount` + `pi4_ports` interface checkers in the validator (registered).
- A hand-authored open-tray reference (`examples/handwritten_pi4_case.py` ->
  `tests/fixtures/handwritten/pi4_case_good.partdsl.json`) that compiles and
  passes ALL required checks, including the standoff-land resolution (OD 7).

## Not done yet (LLM stage, post-foundation)

- `Pi4CaseInterpreter` (LayoutSpec → PartProgram).
- Wiring the discriminated-union LayoutSpec + the abtest harness for bench-002.
- The untuned LLM A/B run.
