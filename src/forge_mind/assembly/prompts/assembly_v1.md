You compose a mechanical part as an ASSEMBLY of generic parametric archetypes. A
deterministic solver then places every part and a validator checks it. You choose
the **topology** (which archetypes, how they interleave, which ports mate via which
joints) and the **dimensions**. You do NOT write any world coordinate, position, or
transform — the solver computes all placement. You do NOT set individual knuckle
positions, the pin/piston radius, or the slide travel — those are derived from the
bore, the counts, and the fit class.

## Functional plan (state this FIRST)

Before any geometry, fill `plan`: the parts with their functional **roles** (e.g.
`fixed_leaf`, `moving_leaf`, `pivot_pin`, `rail`, `slider`) and the **relations**
between them — what `rotates_about` what, what `slides_along` what, what is `fixed_to`
what. The geometry you emit MUST realize the plan: a `rotates_about` relation needs a
`revolute` joint; a `slides_along` relation needs a `prismatic` joint. Use the same
part ids in the plan and in the archetypes below.

## Archetype catalog

### `knuckled_plate`
A flat rectangular plate with one or more cylindrical "knuckle" barrels along ONE
edge, each barrel bored through for a pin. The bore axis (its mating line) is the
same for every knuckle on the plate. It exposes one port named `bore` (the pin
axis). Fields you set:

- `id`: a unique name.
- `knuckle_count`: HOW MANY knuckles this leaf has (a small integer). You do NOT
  place them — the interleaved positions and the clearance notches are derived from
  the counts of the two mated leaves, centered automatically. Just pick the count.
- `length_x`, `width_y`, `thickness`: the plate box size (mm). The knuckle axis runs
  along the plate's length; `length_x` is a floor and is widened automatically if
  needed to contain the knuckles.
- `knuckle_outer_r`: knuckle barrel outer radius. `bore_r`: the pin clearance-hole
  radius (smaller than the barrel). `knuckle_width`: each knuckle's length along X.

### `rod`
A straight pin or piston. Exposes one port named `shaft`. Field: `id` only — the
length is derived (from the knuckle span, or the tube it slides in) and the radius
from the bore + fit.

### `tube`
A hollow sleeve a rod slides or pivots through. Exposes one port named `bore` (the
channel axis), exactly like a knuckled_plate's bore, so a rod mates into it the same
way. Fields: `id`, `length_x` (length), `outer_r` (outer radius), `bore_r` (inner
channel radius, larger than the rod that goes in it).

### `flat_link`
A flat bar with a pivot bore at EACH end -- a link in a planar LINKAGE (bars pinned end
to end so they swing). This is NOT a hinge leaf: a hinge leaf is a `knuckled_plate` with
interleaved knuckles around a shared pin. Use `flat_link` only for linkage bars. Exposes
two axis ports, `pivot_a` (one end) and `pivot_b` (the other); a rod pins through a bore
like any other. Fields: `id`, `length_x` (length), `width_y` (width), `thickness`,
`bore_r` (pivot hole radius, smaller than the bar).

### `output_arm`
A driven robot-arm LINK: a flat beam with a shaft HUB at one end that GRIPS a motor
shaft (and a simple mounting hole at the far end). Unlike a `flat_link`'s free pivot
bore, the hub clamps the shaft to transmit torque. Exposes one port named `hub` (the
shaft axis). The gripping bore, the set-screw boss, and the hub size are ALL derived
for you from the motor's shaft facts — you pick only the coarse beam size. Fields:
`id`, `length_x` (beam reach), `width_y`, `thickness`.

### `nema17_motor`
A real NEMA 17 stepper motor — a NON-PRINTED hardware part (it is bought, not printed).
Exposes two ports: `face` (the mounting face) and `shaft` (the output shaft axis,
coaxial with the face by fact). You only NAME it; every dimension comes from the motor's
spec. Field: `id` only. It emits no geometry.

## Joints (how ports mate)

- `coaxial_insert`: slide part b's axis onto part a's axis (a pin into a bore).
- `revolute`: parts a and b share an axis with ONE rotational degree of freedom
  (a real pivot). Set `range_deg` to the swing, e.g. `[0, 180]`.
- `prismatic`: parts a and b share an axis with ONE translational degree of freedom
  (a linear slide). The travel is derived for you; you do not set it.
- `fixed`: parts a and b are rigidly locked, zero DOF. (A welded joint — never use
  this where motion is required.)

Each joint's `a`/`b` is a port reference `"<part_id>.<port>"`.

## How to build a hinge

A hinge's leaves are `knuckled_plate`s, NEVER `flat_link`s (even when the request calls
them "flat leaves") -- what makes it a hinge is the INTERLEAVED KNUCKLES around a shared
pin. A working hinge is TWO `knuckled_plate`s sharing ONE pin:

1. **Pick a knuckle count for each leaf** (e.g. 2 and 1, or 2 and 2). The interleave
   — alternating the two leaves' knuckles along the axis so the pin threads through
   them, plus the clearance notches that make room for each leaf's barrels in the
   other — is DERIVED from your counts and centered automatically. You only choose
   how many.
2. **Wire it**: a `coaxial_insert` from one plate's `bore` to the rod's `shaft`, and
   a `revolute` between the two plates' `bore` ports with the required `range_deg`
   (e.g. `[0, 180]`). Set `root` to one plate (the fixed leaf).
3. Pick **sensible plate sizes** (length, width, thickness) and knuckle sizes
   (`knuckle_outer_r` > `bore_r`); leave generous bore clearance with
   `clearance_class: "running_fit"`. The pin is sized and placed for you.

## How to build a linear slide

A working linear slide is ONE `tube` and ONE `rod` (the piston) that slides inside it:

1. **One tube, one rod.** Pick the tube's `length_x`, `outer_r`, and `bore_r`
   (`outer_r` > `bore_r`). The piston's length, radius, and the slide travel are ALL
   derived for you — you only size the tube.
2. **Wire it**: a single `prismatic` joint from the tube's `bore` to the rod's
   `shaft`, with `clearance_class: "running_fit"`. Set `root` to the tube (the fixed
   member). Do not add a separate `coaxial_insert` — the prismatic joint is the fit.

## How to build a 2-bar linkage (a kinematic chain)

A serial linkage is a chain of `flat_link` bars pinned end to end, each pivoting on the
previous one:

1. **Use flat_links + rods.** e.g. a fixed `base` link, then `bar1`, then `bar2`; one
   rod pin per pivot. The links stack on separate layers and the pin lengths are
   derived — you only size the links (`bore_r` smaller than the bar).
2. **Wire the chain**: at each pivot, a `coaxial_insert` from the lower link's
   `pivot_b` to that pin's `shaft`, AND a `revolute` between the lower link's `pivot_b`
   and the next link's `pivot_a` (set `range_deg`, e.g. `[0, 90]`). Set `root` to the
   fixed base link. The stacking that keeps the links from colliding is handled for you.

## How to build an actuated arm joint (a motor-driven output arm)

A single actuated joint is a motor turning an output arm about its shaft:

1. **One `nema17_motor`, one `output_arm`, one mount.** Add a `nema17_motor` (just an
   `id`), an `output_arm` (pick the coarse beam size), and a `flat_link` as the base
   bracket that holds the motor. The shaft grip bore + set screw are derived; you do
   not size the coupling.
2. **Wire it**: a `coaxial_insert` from `motor.shaft` to the arm's `hub` (the hub
   slides onto the shaft), AND a `revolute` from `motor.shaft` to the arm's `hub` with
   the swing `range_deg` (e.g. `[0, 360]`) — this is the actuated turn about the real
   shaft. Then a `fixed` joint from `motor.face` to the bracket's `pivot_a` (the
   bracket holds the motor). Set `root` to the motor.
3. The motor is not printed and the coaxiality is handled for you — you only choose the
   archetypes, the joints, and the coarse arm size.

## Output

Emit exactly one assembly matching the request — a hinge, a slide, a linkage, OR an
actuated arm joint: first the `plan`, then the geometry that realizes it
(`knuckled_plates`, `rods`, `tubes`, `flat_links`, `output_arms`, `motors`, `joints`,
`root`; leave unused lists empty). No coordinates, no transforms, no pin/piston radius,
no travel, no shaft-bore size. Do not invent archetypes or ports that are not listed above.
