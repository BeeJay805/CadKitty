"""Curated design-rules packs injected as reference context (the "base docs" seam).

Row-54 lesson: the engine probe's failures were KNOWLEDGE failures, not authoring
failures -- no concept of dimension-chain closure (a block sized too small for its
crank), swing clearance, or physical insertability (a one-piece crank cannot enter a
one-piece block). These packs put that engineering knowledge in front of the brief,
the plan, and the chief revision, selected DETERMINISTICALLY by prompt keywords --
no retrieval model, no network, byte-stable for a given prompt.

Each pack is small on purpose: rules the model must APPLY, not an encyclopedia.
"""

from __future__ import annotations

_MECHANISM_RULES = """\
MECHANISM DESIGN RULES (apply to every moving assembly):
- DIMENSION-CHAIN CLOSURE: before writing part cards, close the kinematic chain
  numerically. Slider-crank: top-dead-center wrist height = throw + rod length; the
  bore ceiling must sit ABOVE TDC wrist + the piston height above its pin, and the
  bore floor BELOW bottom-dead-center wrist minus the skirt below the pin. Write the
  arithmetic into the cards.
- SWING CLEARANCE: every point of a rotating part sweeps a circle about its axis;
  the housing must clear the LARGEST swing radius (for a crank: web corner or
  counterweight tip, sqrt(y^2+z^2) from the axis) plus running clearance, at EVERY
  angle -- not just the assembled pose. State the swing radius in the housing card.
- INSERTABILITY: parts must be assemblable BY HAND after printing. A crankshaft whose
  webs exceed its journal diameter cannot pass through a plain bearing bore: give the
  housing a split line, a bolt-on cap, an open-top cradle, or print-in-place
  clearances. Say HOW each moving part gets into its cavity.
- CLEARANCE ENVELOPE: moving pairs need running clearance (0.3-0.45mm radial for
  printed PLA) on every surface that passes another part, including web faces vs
  housing walls AND crank-web faces vs connecting-rod cheeks (give at least 1 mm
  AXIAL gap between a web face and the rod ring beside it -- coincident faces
  measure as collision the moment the mechanism turns), not only pin-in-bore fits.
- VISUAL PROMISES ARE BINDING: every window, cutaway, exposed-mechanism view, or
  see-through feature the design description promises must actually appear in the
  part cards -- acceptance is judged against the description, and a promised view
  that was never built fails the whole object.
- JOINT TYPES ARE NOT ALL THE SAME -- classify every moving connection by how it MOVES,
  never default everything to coaxial_insert (a coaxial_insert is a STATIC pin in a bore
  with ZERO degrees of freedom, so an assembly of only coaxial_insert joints CANNOT move
  and fails the motion check). A crankshaft turning in its case/main-bearing is a REVOLUTE
  joint. A piston sliding in its cylinder is a PRISMATIC joint. A rod pivoting on a crank
  pin or wrist pin is a REVOLUTE joint. Two meshing gears are a GEAR_MESH joint. Only a
  truly non-moving press-fit pin is coaxial_insert. If the object must MOVE (the prompt
  says it turns, cranks, strokes, spins, or articulates), the plan MUST contain at least
  one revolute or prismatic joint -- give the driving shaft a revolute and every slider a
  prismatic, with range_deg / range_mm set.
- ROTATED STATIONS (radial / star / polar layouts): for station k of N about the X
  axis, the angle is A_k = k * 360/N degrees and the station direction is
  d_k = (0, sin A_k, cos A_k) -- write the EVALUATED sin/cos numbers (3 decimals)
  into every card, never the formula alone. Each part at a rotated station gets its
  port origin AND axis in those evaluated world coordinates, and the two ports of a
  joint must state the IDENTICAL axis line (same origin point, same unit axis) --
  a joint whose port axes differ is dropped and the mechanism cannot be certified.
"""

_ENGINE_RULES = """\
PISTON-ENGINE SPECIFICS:
- Multi-cylinder cranks: state each throw's angular phase; counterweights (or web
  paddles) must clear the crankcase at every crank angle; the crankcase interior must
  clear the rod big-end swing (throw + big-end outer radius) along the FULL crank length.
- Piston at TDC must stay below the head deck; at BDC the skirt must not strike the
  crank webs. Check both in numbers before writing the block card.
- One-piece demonstration blocks: prefer an OPEN-TOP or split crankcase so the crank
  drops in; a closed tunnel block requires a built-up (multi-piece) crankshaft.
"""

_FIT_RULES = """\
PRINTED-FIT TABLE (PLA, FDM, as-printed):
- running/slip fit (shafts, pins, sliding pistons): 0.30-0.45mm radial clearance.
- snug locating fit (dowels, alignment): 0.15-0.25mm radial.
- press fit: 0.00-0.05mm interference; never demand press fits on >20mm bores.
- through-holes for M-series bolts: bolt diameter + 0.4mm.
"""

_FINISH_RULES = """\
DESIGNED FINISH -- a real part is DESIGNED, not a bare functional primitive. Every part
must carry the finish its job and its look demand, WITHOUT changing any working surface:
- FUNCTIONAL FINISH (this is safety/usability, not decoration -- REQUIRED where it applies):
  * a walking / foot / grip surface (ladder step, tread, pedal, footplate) MUST have
    anti-slip texture -- raised ribs or a recessed diamond/grid pattern -- AND drainage /
    grip holes; a bare flat step is WRONG, not merely plain.
  * a hand-turned or hand-held surface (knob, handle, grip, wheel) MUST be knurled or
    ribbed for grip, never a smooth cylinder.
  * a bolt / pin entry (bore mouth, pin end) gets a lead-in chamfer so it starts by hand.
- AESTHETIC / STRUCTURAL FINISH (make it read as engineered, not a prototype block):
  * break FREE edges with a small chamfer or fillet (~0.5-1mm) -- corners, outer edges;
  * prefer ribbed / pocketed / lightened form over a solid block where load allows;
  * proportions should look intentional (consistent wall thickness, aligned features).
- THE HARD CONSTRAINT (so finish never breaks the part): apply detail ONLY to FREE and
  CONTACT surfaces, NEVER to a mating / bearing / sealing / sliding surface -- a fillet or
  texture on a bore lip, journal, or piston skirt changes the FIT and is forbidden. Keep
  every working dimension exact; polish the faces that touch nothing but air or a hand/foot.
- PRINTABILITY: every finish feature >= 1mm (a 0.4mm nozzle cannot resolve finer); keep the
  part watertight. If a detail would drop below 1mm or risk the mate, OMIT it -- a clean
  working part beats a broken pretty one.
"""

_PACKS: tuple[tuple[frozenset[str], str], ...] = (
    (
        # Finish applies to almost any real designed object -- broad selection.
        frozenset(
            {
                "step",
                "steps",
                "tread",
                "rung",
                "rungs",
                "ladder",
                "pedal",
                "grip",
                "knob",
                "handle",
                "wheel",
                "lever",
                "engine",
                "pump",
                "mechanism",
                "model",
                "bracket",
                "housing",
                "part",
                "tool",
                "gear",
                "platform",
            }
        ),
        _FINISH_RULES,
    ),
    (
        frozenset(
            {
                "engine",
                "piston",
                "crankshaft",
                "crank",
                "cylinder",
                "conrod",
                "camshaft",
            }
        ),
        _ENGINE_RULES,
    ),
    (
        frozenset(
            {
                "mechanism",
                "linkage",
                "hinge",
                "slider",
                "gear",
                "rotate",
                "rotating",
                "spin",
                "pivot",
                "joint",
                "moving",
                "engine",
                "piston",
                "crankshaft",
                "crank",
            }
        ),
        _MECHANISM_RULES,
    ),
    (
        frozenset(
            {
                "fit",
                "clearance",
                "pin",
                "bore",
                "shaft",
                "assembly",
                "assemble",
                "mechanism",
                "engine",
                "piston",
                "moving",
            }
        ),
        _FIT_RULES,
    ),
)


def select_design_rules(prompt: str) -> str | None:
    """The rule packs whose keywords appear in ``prompt`` (word-boundary-ish, cased down).

    Returns None when nothing matches (a static bracket pays no context tax).
    Selection is deterministic: pack order is fixed, a pack appears at most once.
    """
    words = {w.strip(".,;:()[]") for w in prompt.lower().split()}
    picked = [text for keys, text in _PACKS if keys & words]
    return "\n".join(picked) if picked else None
