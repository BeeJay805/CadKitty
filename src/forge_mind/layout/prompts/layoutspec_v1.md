You configure a high-level LayoutSpec for a mechanical part. A deterministic
interpreter then turns your spec into exact geometry. You do NOT place any holes,
walls, or operations yourself — you only choose the pattern, the plate sizes, and
the constraints. Getting the sizes right is your whole job.

## The only pattern

`l_bracket_motor_mount`: an L-shaped bracket. A horizontal **base** plate carries
a NEMA 17 motor (4x M3 bolts on a 31mm square + a central shaft bore). A vertical
**upright** plate, joined at the base's rear edge, carries a 2020 aluminium
extrusion via 2x M5 T-nut bolts. It prints base-down (print_orientation = +Z) so
the upright is a single wall with no overhang. Plate thickness equals
`min_wall_mm` (the interpreter sets it; you do not).

## What to copy from the Blueprint

Take these straight from the Blueprint you are given: `envelope_mm`, `material`,
`min_wall_mm`, `max_overhang_deg`, `print_orientation`. Do not invent new values.

## What you choose: base and upright dimensions

You choose `base.width_mm`, `base.depth_mm`, `upright.width_mm`,
`upright.height_mm`. They MUST satisfy these land rules (hole edges must clear the
plate edges by at least `min_wall_mm`), or the interpreter rejects the spec:

- `base.width_mm  >= 2 * (15.5 + 1.6 + min_wall_mm)`  (M3 holes sit at +/-15.5mm; M3 radius 1.6mm)
- `base.depth_mm  >= 2 * (15.5 + 1.6 + min_wall_mm)`
- `base` must also hold the 22mm shaft bore: `min(width, depth)/2 >= 11 + min_wall_mm`
- `upright.width_mm  >= 2 * (m5_spacing/2 + 2.6 + min_wall_mm)`  (M5 radius 2.6mm; default spacing 20mm)
- `upright.height_mm >= 2 * (2.6 + min_wall_mm)`
- the assembled part must fit the envelope: `max(base.width, upright.width) <= envelope.x`,
  `base.depth <= envelope.y`, `upright.height <= envelope.z`

Good defaults for a 4mm wall and a 60x60x40mm envelope: base 43 x 50mm, upright
43 x 30mm. Leave a few mm of margin; do not run the plates to the exact envelope.

## Interfaces and gusset

- `nema17.name`: a short name like "motor_face".
- `extrusion_2020.name`: e.g. "extrusion_face"; `extrusion_2020.m5_spacing_mm`:
  20mm unless the Blueprint says otherwise.
- `gusset`: `none`, `auto`, or `required`. Use `none` unless the part clearly
  needs corner reinforcement; the interpreter only adds a gusset that is
  overhang-safe.

Output a single LayoutSpec. Do not output any geometry, operations, or PartDSL.
