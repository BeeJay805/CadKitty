You configure a standalone STRUCTURAL MEMBER (a strut / beam / brace) as a ComputedBodySpec
(ADR-0019/0020). A deterministic builder turns your CATEGORICAL choice into exact geometry —
you place NO coordinates and invent NO numbers.

Fill `op` by picking the member TYPE (`op.kind`):

- `faceted_strut` — a SOLID straight prism. Choose when stiffness/strength matters most or the
  member is small, and weight is not the priority. Then set `op.cross_section`:
  - `hexagonal` — a stiff, light hex prism; the general solid member, when no flat mating face
    is required.
  - `rectangular` — a rectangular prism; use when the member must present FLAT faces (to stack,
    or to mate flush against another part along its length).

- `lattice` — a LIGHTWEIGHT member: a cellular grid of internal walls that removes material
  while staying printable. Choose when the request stresses LOW WEIGHT / lightweighting / a
  lattice or open structure and the member is large enough to host cells. Then set:
  - `op.cell`: `grid` — square cells (the available cell shape).
  - `op.density`: how much material to keep — `light` (most weight removed), `medium`, or
    `dense` (stiffest). Pick by how much the request favors lightness over stiffness.

Pick `size_class` — the coarse overall size — by the member's role: `compact` (short / small),
`standard` (the default), or `large` (long / heavy-duty). Copy `material`, `min_wall_mm`, and
`max_overhang_deg` straight from the Blueprint. You do NOT size the member: leave `envelope_mm`
null. The interpreter derives the length, the cross-section size, and (for a lattice) the wall
thickness and cell counts from `size_class` + `min_wall_mm`. Invent no numbers.

The derived size always hosts the chosen member at `min_wall_mm` (the interpreter grows the
cross-section / cell footprint to the minimum that prints if a class is tight), so you never need
to fabricate a size to force a pass.

(Taper and curved members are not yet available; use `faceted_strut` or `lattice`.)
