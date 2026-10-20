"""Functional multi-part authoring (the Wave 4 centerpiece).

The benchmark proved the hammer's no_build IS the cosmetic-fusion problem: a hammer is
naturally a steel HEAD + a wooden HANDLE, and forcing them into one fused solid is the
hard, variance-bound part. So when ``brief.target == "functional"`` we model each part as
its OWN single body (reliable to author) and ASSEMBLE them, instead of fusing.

The trick that makes assembly trivial: a SHARED MATE FRAME. Each part is authored in a
common coordinate frame (the receiving socket at the origin; the inserting tenon at the
origin extending away), so overlaying the parts at the origin yields the assembled object.
No geometric joint-solving, no fusion between parts -> the fusion-variance is gone.

Reuses the proven loop per part + the acceptance judge on the assembled render.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from forge_mind.agentic.mech_arch import MechArchitecture

from forge_mind.agentic.brief import DesignBrief, research_brief
from forge_mind.agentic.design_rules import select_design_rules
from forge_mind.agentic.judge import AcceptanceVerdict, acceptance_judge
from forge_mind.agentic.loop import (
    LoopResult,
    _disambiguate_slugs,
    _slug,
    run_agentic_pipeline,
)
from forge_mind.assembly.fits import (
    minimum_running_diametral_clearance_mm,
    running_fit_clearance_mm,
)
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.config import BRAIN_MODEL, PLAN_MAX_TOKENS, SONNET
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client
from forge_mind.materials import Material, clearance_fit, material_spec, resolve_material
from forge_mind.viz.glb import to_gltf_yup
from forge_mind.viz.render import render_part_to_png


class PartPlan(BaseModel):
    """One physical part to model on its own, in the shared mate frame."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Short part name, e.g. 'head', 'handle', 'finger'.")
    prompt: str = Field(
        ...,
        description="A SELF-CONTAINED prompt to model THIS part as ONE solid, INCLUDING where its "
        "mating interface sits in the shared origin frame (so parts overlay at the origin correctly).",
    )
    ports: list[Port] = Field(
        default_factory=list,
        # 12 (was 8, was 3): the i4-finish audit found the 3-port cap made a 4-cylinder engine
        # UNDECLARABLE -- a crankshaft honestly needs ~7 ports (mains + 4 crankpins), the
        # block ~7 (main bearing + 4 cylinder slides), so cylinders 3-4 went dangling
        # (zero joints) and the sweep hit them as phantom static obstacles. Caps must
        # scale together: parts/joints went to 16, ports must carry the same mechanism.
        # ADR-0128 (flat-6): the SAME arithmetic one cylinder-pair further. A flat-six
        # crankshaft needs 4 mains + 6 crankpins = 10 ports, plus a hand-crank hub = 11;
        # at 8 the crank could not declare half its own throws. 12 carries it with headroom.
        max_length=20,
        description="For EVERY part that participates in a joint (moving OR receiving): the "
        "connection frame(s) on this part, in the shared origin coords. Each port is a pivot/slide "
        "axis: origin = a point on the axis, axis = the unit axis direction, normal = any unit "
        "direction perpendicular to it. Declare ALL of a part's connections (a 4-cylinder "
        "crankshaft needs a port per crankpin plus its main journal). A part in no joint leaves "
        "this empty.",
    )
    retention: str | None = Field(
        default=None,
        description="PIN-class parts only (TEST_LOG L101): the axial-retention decision this "
        "pin's geometry realizes, copied VERBATIM from the architecture's joint decision when "
        "one is in force (e.g. 'printed shoulder on pin root + push-on cap'). The card must "
        "then MODEL it. Non-pin parts leave this null.",
    )


class FunctionalPlan(BaseModel):
    """A naturally-multi-part object decomposed into separate parts in a shared frame."""

    model_config = ConfigDict(extra="forbid")

    object_name: str = Field(..., description="The whole object, e.g. 'claw hammer'.")
    # RAISED CAPS (2-4 -> 8 -> 10 -> 16/16 -> 32/32, founder direction 2026-07-25), GATED behind
    # the all-pairs seating check: with _assembly_fit scanning EVERY part pair (any interfering
    # pair OR a floating part FAILS ``seated``), a larger decomposition can no longer hide a
    # mis-placed extra part -- so a genuine multi-cylinder mechanism can be authored without a
    # silent no-false-PASS. 16 was sized for an inline-4: honestly 14 parts (block, 4-throw crank,
    # 4 rods, 4 pistons, 4 pins) with 13 joints.
    #
    # 56 (was 40) -- and 56 is not a preference, it is where the SINGLE-CALL ceiling lands.
    # Founder 2026-07-25: "why is there ever a reason to limit amount of parts we can make".
    # Correct in principle, and the honest answer is that the binding constraint is not this
    # number, it is that the whole plan must come back from ONE structured-output call.
    # Measured on the F2 flat-six plan: 1,447 chars/part card and 277 chars/joint, so
    # 56 parts + 72 joints ~= 25.2k tokens against PLAN_MAX_TOKENS=32000 -- which is itself
    # Opus/Fable's OUTPUT LIMIT and therefore cannot simply be raised.
    #
    # So this cap is real but temporary: ~56-64 parts is the most any single-call decomposition
    # can express, and a V12 (~68 parts) is already out of reach. Removing the limit for good
    # needs HIERARCHICAL decomposition (subassemblies decomposed separately), after which no
    # single call holds the whole roster and this number stops mattering.
    #
    # Deleting the cap without that would be worse than keeping it: the failure converts from a
    # clean pydantic rejection into a plan JSON truncated mid-object -> parse failure -> no_build,
    # after the planning call is already paid for.
    #
    # 40 is sized for a FLAT-SIX (ADR-0128), measured off a real architecture the engine produced
    # rather than guessed: split case 2, deck plates 2, barrels 6, pistons 6, conrods 6, rod caps
    # 6, crankshaft 1, hand-crank 1 = 30 -- PLUS the 6 wrist pins, because this system prompt
    # requires "each pin as its OWN part" (a pivot declared on empty bores cannot be certified).
    # That is 36, and it is the count that matters: a first pass at this cap said 32 by reading
    # the architecture's mirrored_pairs, which lists pins as JOINTS and therefore undercounted
    # the roster by exactly the six parts the prompt insists on. 40 leaves headroom.
    #
    # At 16/16 the boxer six was not merely hard, it was INEXPRESSIBLE -- decompose had to
    # silently drop half the engine while the object kept the name "flat six", which is a
    # false-PASS at the plan tier.
    #
    # The all-pairs seating gate is O(n^2), so this takes pair checks from 120 to 780. That cost
    # is the point: it is the mechanism that stops a bigger roster from hiding a bad part.
    parts: list[PartPlan] = Field(..., min_length=2, max_length=56)
    assembly_note: str = Field(..., description="One line on how the parts fit together.")
    joints: list[Joint] = Field(
        default_factory=list,
        # Scales with `parts` (ADR-0128), and sized by COUNTING, not by matching the part cap --
        # a first pass set this to 40 to mirror `parts` and it was one short of the very example
        # the ADR names. The COMPLETE-JOINT-GRAPH rule means every part must appear in >= 1 joint,
        # so a 36-part flat-six needs 41:
        #   30 in the cylinder chains -- per cylinder: crankpin (crank<->conrod), rod cap<->conrod,
        #      wrist pin<->conrod, wrist pin<->piston, piston<->barrel slide, x6
        #   11 structural -- case halves, crank<->case main, 2 decks<->case, 6 barrels<->decks,
        #      hand-crank<->crank
        # 48 carries that plus the variant where all 4 main bearings are declared separately (44).
        # At 16 the graph could not close at all, leaving parts dangling and the sweep treating
        # them as phantom static obstacles.
        max_length=72,
        description="ONLY if parts MOVE relative to each other: one joint per moving connection "
        "(revolute = pivot/hinge, prismatic = linear slide, coaxial_insert = a pin in a bore, "
        "gear_mesh = two meshing gears that turn together, fixed = rigidly joined). a/b name the two "
        "mated ports '<part>.<port>'. Give range_deg for a revolute, range_mm for a prismatic; for "
        "gear_mesh set module + teeth_a + teeth_b, and internal=true when one gear has INTERNAL "
        "teeth (an annulus/ring: its centre distance is the pitch-radius DIFFERENCE, not the sum). "
        "A rigid object (a hammer) leaves this empty.",
    )


_DECOMPOSE_SYSTEM = """You decompose a naturally-multi-part object into its SEPARATE physical PARTS \
to be modeled individually and then assembled (a claw hammer = a steel HEAD + a wooden HANDLE; a \
two-finger gripper = a BASE + two FINGERS).

Output as FEW parts as the object honestly needs (2-4 is typical; up to 56 only for a genuine \
multi-link mechanism, counting each pivot's PIN as its own part -- a flat-six engine is honestly \
36: split case 2, deck plates 2, barrels 6, pistons 6, conrods 6, rod caps 6, wrist pins 6, \
crankshaft, hand-crank). NEVER drop members of a repeated set to fit a budget: an engine that \
silently becomes a flat-four while still called a flat-six is a worse answer than a refusal. \
For EACH part write a \
SELF-CONTAINED prompt to model it as ONE solid. CRITICAL: \
every part is built in a SHARED coordinate frame so the parts ASSEMBLE by overlaying at the ORIGIN. \
State, in each part's prompt, WHERE its mating interface sits relative to the origin so the parts \
meet correctly:
- the RECEIVING part (a hammer head with an eye socket): build it with the SOCKET centered at the \
origin, socket axis vertical (Z), the body extending from there.
- the INSERTING part (the handle): build it with its TOP/tenon at the origin, extending DOWNWARD \
(-Z), sized to fit the socket.
Choose the frame so overlaying all parts at the origin yields the correctly-assembled object. Each \
part prompt should read like a normal single-object request plus the one frame sentence.

If a part is a graspable HANDLE / PULL / GRIP / LOOP (a door pull, drawer pull, bail, lever), say \
explicitly in its prompt that it is a LOOP with a finger APERTURE you wrap your hand THROUGH -- NOT \
a solid spike, knob, or blob -- so the modeler builds an open loop with the opening, then merges its \
leg ends into the base.

MOVING PARTS (mechanisms). If two parts MOVE relative to each other -- a hinge that folds, a gripper \
finger that pivots, a slider that travels -- declare it so the motion gets collision-validated:
- Add a JOINT to `joints`: type `revolute` (a pivot/hinge), `prismatic` (a linear slide), \
`coaxial_insert` (a pin/shaft seated in a bore, no motion), or `fixed` (rigidly joined). Set `a` and \
`b` to the two mated ports as '<part>.<port>'. Give `range_deg` [lo,hi] for a revolute (a hinge \
[0,180], a gripper finger [0,40]) or `range_mm` for a prismatic. `clearance_class` is usually \
'running_fit'.
- On EACH moving part, declare the matching `ports`: a port is the pivot/slide axis frame in the \
SHARED origin coords -- `origin` a point on the axis, `axis` the unit axis direction (e.g. {z:1} for \
a vertical pivot), `normal` any unit direction perpendicular to it. The two mated ports MUST share \
the same axis line (same origin point + same axis direction) so the joint is truly coaxial.
- CRITICAL: author the two moving parts so that, overlaid at the origin, they are ADJACENT and do \
NOT already interpenetrate -- leave the real gap a working mechanism has at its pivot. The motion \
check sweeps one part through its range and FAILS on any collision.
- THE PIN IS A PART: a revolute / pivot / hinge joint physically needs its PIN in the assembly. \
Include each pin as its OWN part (e.g. "shoulder pin: a plain 6 mm diameter x 50 mm cylindrical \
pin, axis along the pivot axis, centered on the joint origin in the shared frame") -- a pivot \
declared on empty bores cannot be certified (the pin-in-bore fit check finds nothing to verify). \
Size every bore for a RUNNING fit over its pin (about 0.2-0.3 mm diametral clearance), never \
line-to-line.
- DO THE CLEARANCE ARITHMETIC YOURSELF and state RESULTING dimensions -- the part modelers never \
see each other's geometry, so a subtraction left to them WILL collide. A tab seating between ears \
30 mm apart is AT MOST 27 mm wide (>= 1.5 mm air per side); say "27 mm wide", not "fits between". \
At the HOME pose (all joints at 0) moving pairs must not TOUCH anywhere except pin-in-bore: \
surfaces that rub at 0 deg read as a COLLISION in the sweep.
- A purely RIGID object (a hammer head fixed on its handle) has NO moving joint: leave `joints` and \
`ports` empty. Do not invent motion.
- DECLARE DIAMETERS ON MATED PORTS: every port on a pin/bore/slide joint carries `diameter_mm` (bore ID on the receiving port, pin OD on the inserting port) with the running clearance already in the numbers -- a deterministic gate checks the pair closes BEFORE any part is modeled.
- COMPLETE THE JOINT GRAPH: once ANY joint is declared, EVERY part must appear in at least one \
joint (fixed/coaxial_insert for rigid mounts) and every joint endpoint's port must be DECLARED on \
that part -- a multi-cylinder mechanism declares the joints and ports of EVERY cylinder (crankpin, \
wrist, slide per cylinder; up to 20 ports per part), not just the first. A part in no joint is \
kinematically dangling and the plan is REJECTED by a deterministic gate.

MESHING GEARS. If two parts are GEARS that mesh and turn together (a gear pair, a reduction), declare \
a `gear_mesh` joint so the mesh is collision-validated:
- Author each gear with the `spur_gear` helper at its real tooth count. Two meshing gears share the \
same `module` (mm of pitch diameter per tooth); pitch diameter = module * teeth.
- PRINTABILITY: make each gear a PLAIN toothed disc -- ONE uniform face width with a through bore and \
NO hub, collar, boss, or step under the teeth. An underside ledge (a narrow hub below a wider disc) \
is an unsupported overhang that FAILS printability. The mounting shaft is a SEPARATE part, never a \
stub grown on the gear; the gear's full flat face lies on the build plate.
- Place the two gear axes EXACTLY module*(teeth_a+teeth_b)/2 mm apart (the sum of the pitch radii) so \
the teeth interleave -- e.g. module 2, 20T + 40T -> 2*(20+40)/2 = 60 mm centre-to-centre.
- An INTERNAL (ring) gear meshes at the pitch-radius DIFFERENCE instead: axes \
module*|teeth_ring-teeth_pinion|/2 mm apart (e.g. module 2, 48T ring + 18T planet -> 2*(48-18)/2 = \
30 mm), and its `gear_mesh` joint MUST set `internal: true`.
- Add the joint: type `gear_mesh`, `a`/`b` = the two gears' axis ports '<part>.<port>', and SET \
`module`, `teeth_a`, `teeth_b` to match the authored gears.
- On EACH gear declare one `port`: `origin` the gear centre, `axis` the rotation axis (e.g. {z:1}), \
`normal` any perpendicular unit. The check FAILS a wrong centre distance, tooth count, or clocking.
- CARRIER TRAINS (planets orbiting on a carrier): every frame-fixed feature (housing walls, webs, stand \
legs, rims) must stay OUTSIDE the ORBIT ANNULUS the planets sweep -- radially r_orbit +/- (planet tip \
radius + 1 mm) across the planets' full axial span, where tip radius = module*(teeth+2)/2 and r_orbit = \
module*(z_sun+z_planet)/2. SHOW this arithmetic on the housing card (measured: three runs put housing \
material 1.9-4.2 mm into the orbit and the motion gate failed them every time).

ENCLOSED PARTS. A part that lives INSIDE another (an impeller in a volute, a gear inside a ring, a ball \
in a valve): author the OUTER part hollow with its cavity sized to the inner part plus running clearance \
on every inner wall; place the inner part at its cavity station in the shared frame; guarantee an \
INSERTION path (an open face, or a separate end-cap part that closes the cavity after the inner part is \
placed). Designed clearance all around is CORRECT here -- the fit tier recognises enclosure; never weld the \
inner part to a wall just to make contact."""


# ---------------------------------------------------------------------------------
# CHIEF-ENGINEER mode (FORGE_CHIEF=1, founder architecture 2026-06-11): ONE highly
# capable model (the BRAIN_MODEL) is the BRAIN -- it plans every aspect of the build as exact
# BUILD CARDS (all arithmetic done, zero design freedom left), and when the assembled
# mechanism measurably fails it revises ITS OWN PLAN -- while cheap drafter models
# (sonnet) write the per-part code from the cards. "A team of engineers, but one
# brain running everything." Evidence base: blind per-part authors caused the v1
# tab-vs-ear collision and the v2d pin stacking (each pin authored at the local
# origin); the holistic watch-train probe showed the brain's relational arithmetic
# is flawless when it holds the whole design. The VALIDATOR remains the sole truth
# gate -- the brain's cohesion is checked, never trusted. OPT-IN; flag off keeps the
# legacy path byte-identical.
# ---------------------------------------------------------------------------------


def _scaled_iterations(part_prompt: str, base: int) -> int:
    """Feature-dense parts get iteration headroom (L86: the torso and shin+foot combos
    died at the SAME cap simple brackets get). v1 proxy for density: the count of numeric
    dimensions in the part prompt (dense parts carry many stated dims); >= 8 numbers
    earns +2 iterations. FORGE_ITER_SCALE=1 (default OFF; off = budgets byte-identical)."""
    import re as _re

    if os.environ.get("FORGE_ITER_SCALE") != "1":
        return base
    numbers = len(_re.findall(r"\d+(?:\.\d+)?", part_prompt))
    return base + 2 if numbers >= 8 else base


# The L86 humanoid probe's two plan-level failure classes, as decompose-time doctrine
# (FORGE_PLAN_DOCTRINE=1): feature-dense combos capped out (torso with every pocket;
# shin+foot as ONE body) and the home pose self-collided (forearm at hip level, 2.08 mm).
_PLAN_DOCTRINE = """

PLAN DOCTRINE (each rule prevents a measured whole-assembly failure):
- SPLIT FEATURE-DENSE PARTS: a part carrying more than ~4 major features (pockets,
  patterns, interfaces) fails authoring far more often than two simple parts joined by
  a fixed joint. Split at natural seams: a shin and its foot are TWO parts (fixed
  joint); a torso with many servo bays becomes a frame plus bolt-on bay plates.
- HOME-POSE CLEARANCE: place every limb so the assembly is collision-free AT THE HOME
  POSE with >= 5 mm air between a moving part and any other body or servo across its
  whole travel corridor -- offset shoulder axles OUTBOARD of the torso and hips, never
  let a hanging arm's station overlap the hip line.
- POCKETED HARDWARE BAYS face OUTWARD or UP so the printed part drops onto the
  hardware, never capturing it on six sides.
"""

# The H2 humanoid's measured FORM gap (L88 render review: every part accepted, yet the
# assembly reads as a lab prototype of flat plates -- box limbs, bare pins, no proportion
# spec), GENERALIZED at founder direction (2026-07-23 mid-session: "everything needs to
# be detailed and crafted, not primitives" -- not a robot-only rule). General crafted-form
# doctrine + a humanoid annex (FORGE_FIDELITY_DOCTRINE=1, default OFF; off = decompose
# byte-identical). Annex ratio derivation: human knee ~0.285H / hip ~0.53H stature
# ratios, head enlarged to 0.13H (consumer humanoids -- the Unitree-G1 class -- read
# slightly large-headed), athletic V-taper shoulders 0.30H -> waist 0.25H.
_FIDELITY_DOCTRINE = """

FORM-FIDELITY DOCTRINE (every object): the finished object must read as the DESIGNED,
CRAFTED product -- the real thing's form language -- never as an assembly of bare CAD
primitives.
- CARDS STATE THE FORM LANGUAGE: every body / segment names its cross-section shape
  (round / elliptical / rounded-rect / airfoil), its taper or crown, and its edge
  treatment (fillet / chamfer radii) as NUMBERS. A card that leaves form unstated
  gets a primitive from the drafter, and that is a DEFECT -- unless the real feature
  is genuinely prismatic (a flat mounting plate is honestly flat; state its edge
  treatment even so).
- PRIMITIVES ARE SCAFFOLDING, NOT FINAL FORM: a bare Box or Cylinder ships only where
  the real object's feature is truly prismatic. Route form through the vocabulary:
  LOFT for tapered / organic volumes (the tapered_limb_shell idiom), SWEEP for curved
  members, REVOLVE for turned bodies, SHELL for thin vessels / housings, crowned or
  gently curved plates where the product's panels are curved, domed sphere-capped
  pods for pivots and end caps (the joint_module idiom).
- SURFACE DETAIL IS PART OF THE PART: every detail the brief names (visor recess,
  panel lines, vents, grilles, badges-as-recesses, grip textures) appears in a card
  as a LOCATED pocket / recess / rib on a free face. A promised detail with no card
  is a DEFECT.
- HUMANOID / FIGURE ANNEX (stated for a 350 mm figure; scale stations linearly, hold
  each segment within ~10%). Z from the sole: knee axis 105 (0.30H), hip axis 180
  (0.51H), shoulder axis 290 (0.83H), torso top 300, head top 350. Segments: head 45
  tall x ~50 wide (a rounded helmet form); torso shell 120 tall, width 105 at the
  shoulders -> 88 at the waist (0.30H -> 0.25H), depth ~55; thigh 75 (hip -> knee);
  shank+foot 105 (knee -> sole, foot pad ~20 tall x ~60 long); upper arm 55; forearm
  ~60 (a hanging arm ends near the hip line). Limb cross-sections (elliptical,
  root -> tip): thigh 34 -> 26, shin 28 -> 22, upper arm 26 -> 20, forearm 24 -> 18.
  RECTANGULAR-BOX LIMBS ARE FORBIDDEN: a limb card is a tapered lofted volume, never
  a plain box. Every shoulder, elbow, hip, and knee is dressed as a cylindrical
  joint module with domed caps (~24-36 mm dia on a 350 mm figure) housing or
  covering its pin / servo; bare pins through bare plates are FORBIDDEN at these
  four joint classes.
- FORM NEVER OVERRIDES FIT: stated dimensions, servo pockets, clearances, print
  orientation, and every card number still win any conflict with styling.
"""

# The H3 render-review gap (founder, 2026-07-23 evening: "detailed but not engineered --
# no real company would put this out"): every interface must carry a NAMED engineering
# decision a deterministic audit can verify. FORGE_ENG_DOCTRINE=1, default OFF; off =
# decompose byte-identical. The audit twin is _plan_engineering_issues (FORGE_ENG_AUDIT).
_ENG_DOCTRINE = """

ENGINEERING DOCTRINE (every assembly -- cards must read like a professional CAD team's
detail drawings, not sculpture):
- EVERY INTERFACE CARRIES A NAMED ENGINEERING DECISION, in the card: the fastening
  method BY NAME (screw boss + self-tapping screw, heat-set insert + machine screw,
  tongue-and-groove seam + screw bosses, dowel + retained pin -- never a bare glue-fit
  tenon for anything that must assemble or be serviced), the clearance CLASS by fit
  (running fit 0.30-0.40 mm, location/insertion fit 0.15-0.20 mm, press fit 0.05 mm --
  chosen per function, never one number everywhere), and a LEAD-IN (chamfer >= 0.5 mm)
  on every inserted feature (tenon, pin, boss, tab).
- EVERY MOVING JOINT IS A BEARING PROBLEM: name the journal and its bore land, state
  the running clearance, and RETAIN the pin axially BY NAME (shoulder + cap screw,
  printed shoulder + push-on cap, e-clip groove, threaded end + nut pocket). A pin
  with nothing stopping axial walk-out is a DEFECT.
- BOUGHT ACTUATORS ARE GEOMETRY: when hardware is declared (e.g. an MG996R servo),
  the housing part carries its POCKET (stated body dims + stated air per side), lug
  SEATS with screw holes on the datasheet pattern (MG996R: 4 holes on 49.5 x 10.0),
  a wire exit, and the driven link couples to the servo HORN (state the horn bore
  circle), never to the case by friction. A declared actuator with no pocket card is
  a DEFECT.
- SHELL-AND-RIB, NOT SOLID: a body whose smallest bbox dimension exceeds ~5x the
  stated wall is SHELLED to the stated wall (ribs/bosses inside as needed); solid
  limbs and housings waste material and read as unengineered.
- MIRRORED PAIRS ARE MIRRORS: _L/_R cards state IDENTICAL dimensions mirrored about
  the sagittal plane; a left/right pair differing in size is a DEFECT.
- SERVICEABILITY: shells that enclose hardware state how they OPEN (seam + named
  fasteners), and wiring gets a named route (channel or grommet hole) between bays.
"""

_CHIEF_OVERLAY = """

YOU ARE THE CHIEF ENGINEER (one brain, many hands). The per-part modelers are DRAFTERS with
ZERO design freedom: they never see each other's geometry or talk to each other, so every
number they need must be IN their card. You own the whole design and all of its arithmetic:
- Write each part prompt as a BUILD CARD: a complete dimension table (every dimension a
  NUMBER, never a relation), every feature's exact position IN THE SHARED FRAME (x, y, z),
  every axis named, every clearance/center-distance/tooth-count already computed BY YOU.
- A card that says "fits between", "matches", "aligned with", or "same as" instead of a
  number is a DEFECT -- do the subtraction and write the result.
- Distinct parts occupy DISTINCT places: state each part's exact position so no two parts
  (especially pins) land on the same spot. Re-check the assembled picture in your head:
  every mating pair adjacent with its stated clearance, every pin ON its own joint axis.
- Sanity-check the WHOLE before answering: walk the assembly part by part and verify the
  numbers compose (stack-ups close, bores over pins, gear pairs at module*(z1+z2)/2)."""


def _chief_enabled() -> bool:
    """Is CHIEF-ENGINEER mode on? (FORGE_CHIEF=1; unset/0 = legacy, byte-identical)."""
    return os.environ.get("FORGE_CHIEF", "").strip().lower() in {"1", "true", "yes"}


def _decompose_system() -> str:
    """Assemble the decompose system prompt from the enabled overlay flags.

    Every overlay defaults OFF; with all flags unset the return is _DECOMPOSE_SYSTEM
    byte-identical (the flag law, pinned by test). PLAN doctrine (L86 classes) and
    FIDELITY doctrine (L88 form classes) are independent flags so a stacked arm stays
    bisectable by env stamps."""
    base = _DECOMPOSE_SYSTEM + _CHIEF_OVERLAY if _chief_enabled() else _DECOMPOSE_SYSTEM
    return base + _enabled_overlays()


def _enabled_overlays() -> str:
    """The flag-gated doctrine overlays, concatenated (all default OFF -> empty). The
    mech-arch stage rides the SAME overlays so the architecture obeys the rules the
    cards are audited against."""
    s = ""
    if os.environ.get("FORGE_PLAN_DOCTRINE") == "1":
        s += _PLAN_DOCTRINE
    if os.environ.get("FORGE_FIDELITY_DOCTRINE") == "1":
        s += _FIDELITY_DOCTRINE
    if os.environ.get("FORGE_ENG_DOCTRINE") == "1":
        s += _ENG_DOCTRINE
    return s


class PlanRevision(BaseModel):
    """The chief's response to measured assembly failures: replacement build cards."""

    model_config = ConfigDict(extra="forbid")

    diagnosis: str = Field(
        ..., description="One short paragraph: the root cause of the measured failures."
    )
    revised_parts: list[PartPlan] = Field(
        ...,
        min_length=1,
        # Scales with `parts` (ADR-0128): one repair round on a flat-six can legitimately need to
        # re-card a whole bank (6 barrels + 6 pistons) plus the crank. Capped below `parts` on
        # purpose -- a revision that replaces EVERY card is a re-plan, not a repair.
        max_length=16,
        description="REPLACEMENT build cards -- `name` must match an existing part EXACTLY; "
        "include ONLY the parts that must change. Full card each (the old one is discarded).",
    )


_CHIEF_REVISE_SYSTEM = (
    "You are the CHIEF ENGINEER who authored this multi-part mechanism plan. The parts were "
    "built exactly to your cards, assembled at the shared origin, and MEASURED. The measured "
    "failures below are facts about geometry, not opinions. Diagnose the root cause across the "
    "WHOLE design (the defect may live in a different part than the symptom), then return "
    "replacement BUILD CARDS for only the parts that must change. Cards follow the same rules "
    "as before: every dimension a number, exact shared-frame positions, all arithmetic done by "
    "you, zero design freedom for the drafters. Keep part names and the joint/port topology "
    "EXACTLY as planned (ports may move only if you move the joint axis they describe). "
    "EVERY revised card MUST re-declare the part's FULL ports list -- copy the original "
    "ports VERBATIM unless the revision deliberately moves an axis. The joints reference "
    "those ports: a revised card that omits or drops a referenced port is REFUSED "
    "deterministically and the repair round is wasted (measured: two live runs lost their "
    "only revision to omitted ports). "
    "When an assembled render is attached, read it as evidence: protrusions, buried parts, "
    "and misplaced features visible in the image are measurements too."
)


def _part_extent_lines(part_results: list[tuple[str, LoopResult]]) -> list[str]:
    """AS-BUILT sizes for the chief: per part, bbox extents + world center + Z span (the
    revision diagnosis was BLIND to built geometry -- row 54's chief fixed the symptom part
    because it never saw the sizes). Guarded per part; a missing/broken STL is skipped."""
    import trimesh

    lines: list[str] = []
    for name, lr in part_results:
        if lr.final_stl is None or not lr.final_stl.exists():
            continue
        try:
            m = trimesh.load(str(lr.final_stl), force="mesh")
            e = m.extents
            c = m.bounds.mean(axis=0)
            lines.append(
                f"{name}: {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm, "
                f"center ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), "
                f"Z {m.bounds[0][2]:.1f}..{m.bounds[1][2]:.1f}"
            )
        except Exception:
            continue
    return lines


def _chief_revise_plan(
    plan: FunctionalPlan,
    failures: list[str],
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    render_png: Path | None = None,
    extent_lines: list[str] | None = None,
    notes: list[str] | None = None,
) -> PlanRevision | None:
    """Ask the chief to revise its own plan against the measured failures. None on any
    failure (the caller keeps the honest FAIL -- a broken revision never masks one).

    ``render_png`` (the assembled render) and ``extent_lines`` (as-built part sizes) un-blind
    the diagnosis: the chief sees WHAT was built, not only which pairs failed. ``notes``
    (optional sink) receives WHY a None was returned -- row 56's repair no-fire was
    undiagnosable because this except was silent."""
    client = client or make_client()
    cards = "\n\n".join(f"## {p.name}\n{p.prompt}" for p in plan.parts)
    user = (
        f"YOUR PLAN ({plan.object_name}):\n{cards}\n\n"
        # 24000 (was 2000): a 23-joint flat-six serializes to ~6.7k chars and the old cap cut the
        # dump mid-object after ~6 joints, so the revision was asked to preserve structured ports
        # it had never been shown and `_revision_port_guard` then refused the result -- a
        # bank-wide repair silently could not work. Sized for the ADR-0128 cap, not for today's
        # engine: a FULL 40-joint plan serializes to ~12.9k, so 24000 clears it with headroom
        # rather than moving the same cliff a little further out.
        f"JOINTS: {json.dumps([j.model_dump(mode='json') for j in plan.joints], default=str)[:24000]}\n\n"
        "MEASURED FAILURES (from the assembled geometry):\n"
        + "\n".join(f"  - {f}" for f in failures)
    )
    if extent_lines:
        user += "\n\nAS-BUILT PART SIZES (measured from the meshes):\n" + "\n".join(
            f"  - {ln}" for ln in extent_lines
        )
    rules = select_design_rules(plan.object_name + " " + cards)
    if rules:
        user += (
            f"\n\nREFERENCE DESIGN RULES (apply; show the arithmetic in revised cards):\n{rules}"
        )
    user += "\n\nDiagnose and return replacement build cards for the parts that must change."
    try:
        png = render_png.read_bytes() if render_png is not None and render_png.exists() else None
        if png is not None:
            revision, _usage = client.generate_structured_with_image(
                system=_CHIEF_REVISE_SYSTEM,
                user=user,
                image_png=png,
                output_model=PlanRevision,
                model=model or BRAIN_MODEL,
                max_tokens=PLAN_MAX_TOKENS,
            )
        else:
            revision, _usage = client.generate_structured(
                system=_CHIEF_REVISE_SYSTEM,
                user=user,
                output_model=PlanRevision,
                model=model or BRAIN_MODEL,
                max_tokens=PLAN_MAX_TOKENS,
            )
        return revision
    except Exception as exc:
        if notes is not None:
            notes.append(f"chief revision call failed: {type(exc).__name__}: {exc}")
        return None


def _revision_port_guard(plan: FunctionalPlan, revised: PartPlan, canonical: str) -> str | None:
    """Refusal reason if the revised card no longer declares every port a plan joint
    references on this part; None accepts the card as-is. Enforces the contract the
    revision prompt only STATES ('keep the joint/port topology EXACTLY as planned'):
    a card that lost a referenced port silently unresolves those joints at re-validation
    (the north-star smoke: 13/13 -> 9/13 after a chief revision of 5 cards).

    Attribution + separator normalization are the RESOLVER'S OWN predicates
    (assembly_adapter.joint_referenced_ports / port_norm) so 'guarded' is definitionally
    'resolvable' -- the adversarial review broke a re-implementation twice (dotted part
    names slipped through; 'main-axis' vs 'main axis' over-refused). NO port carry-forward
    on an empty card: a stale frame on relocated geometry could sweep-pass about the wrong
    axis (a false PASS); refusing keeps the ORIGINAL card -- topology intact, geometry
    unrevised, the honest FAIL stands (fail-closed)."""
    from forge_mind.agentic.assembly_adapter import joint_referenced_ports, port_norm

    referenced = joint_referenced_ports([p.name for p in plan.parts], plan.joints).get(
        canonical, set()
    )
    if not referenced:
        return None
    declared = {port_norm(p.name) for p in revised.ports}
    missing = sorted(referenced - declared)
    if missing:
        return (
            f"revision for '{canonical}' dropped joint-referenced port(s) {missing} -- "
            "card refused, original kept"
        )
    return None


def _assembly_failure_lines(
    fit: dict[str, Any] | None, motion: dict[str, Any] | None, part_names: list[str]
) -> list[str]:
    """The measured assembly failures as plain lines for the chief: buried/floating pairs
    BY PART NAME (the fit check reports indices) + the sweep verdict + per-joint checks."""
    lines: list[str] = []
    if fit is not None and not fit.get("seated", True):
        for p in fit.get("pairs", []):
            if p.get("touches") and not p.get("seated"):
                i, j = p.get("pair", [0, 0])
                a = part_names[i] if i < len(part_names) else f"part{i}"
                b = part_names[j] if j < len(part_names) else f"part{j}"
                lines.append(
                    f"{a} <-> {b}: {p.get('note', 'interference')} "
                    f"(overlap {p.get('interference_mm3', '?')} mm^3, "
                    f"{round(100 * float(p.get('overlap_ratio', 0)))}% of the smaller part)"
                )
        for idx in fit.get("floating_parts", []):
            name = part_names[idx] if idx < len(part_names) else f"part{idx}"
            lines.append(f"{name}: FLOATING -- touches no other body in the assembly")
    if motion is not None and not motion.get("articulates", True):
        detail = motion.get("motion_detail")
        if detail:
            lines.append(f"motion sweep: {detail}")
        for note in motion.get("joint_drop_notes") or []:
            lines.append(f"joint declaration: {note}")
    if motion is not None:
        # failing checks reach the chief even when the sweep PASSED (a too-far gear pair
        # never touches; an off-axis pin can sweep clean). Prefer the COMPLETE report
        # fail list; fall back to the whitelisted dict for older verdicts.
        fails = motion.get("assembly_fails")
        if fails:
            for f in fails:
                if isinstance(f, dict):
                    lines.append(f"{f.get('id', 'assembly')}: {f.get('message', '')}")
        else:
            for cid, c in (motion.get("checks") or {}).items():
                if isinstance(c, dict) and c.get("status") == "fail":
                    lines.append(f"{cid}: {c.get('message', '')}")
    if motion is not None and not motion.get("articulates", True):
        # DETERMINISTIC joint-TYPE backstop (row 58/radial): a plan of only static joints
        # cannot move. When the motion check fails for want of a moving joint, name the fix
        # explicitly so the repair re-TYPES the joints (chief flakily typed everything
        # coaxial_insert) instead of guessing.
        detail = str(motion.get("motion_detail", ""))
        if "no revolute or prismatic joint present" in detail:
            lines.append(
                "JOINT-TYPE ERROR: every declared joint is static (coaxial_insert/fixed), so "
                "the mechanism has zero degrees of freedom and cannot move. Re-declare the "
                "driving shaft's joint as REVOLUTE (with range_deg) and every sliding part's "
                "joint as PRISMATIC (with range_mm); keep only genuine press-fits as coaxial_insert."
            )
    return lines


def _canonicalize_plan_endpoints(plan: FunctionalPlan) -> FunctionalPlan:
    """Deterministic endpoint-name repair (the pump lesson, TEST_LOG row 61): the chief
    writes joint endpoints in shorthand ('housing.bearing_bore') while declaring the part
    as 'volute_housing'. When the shorthand matches exactly ONE declared part
    (normalized containment / suffix), rewrite the endpoint to the canonical name --
    the same sanitize-don't-crash philosophy as the adapter's slug-insensitive port
    resolution. Ambiguous or unmatched shorthand is left verbatim for the topology gate
    to report. Pure function: returns a new plan (or the same object when no rewrite)."""

    def _norm(s: str) -> str:
        return _slug(s).replace("-", "_")

    names = sorted((p.name for p in plan.parts), key=len, reverse=True)
    norm_names = {n: _norm(n) for n in names}

    def _resolve(ref: str) -> str:
        part = next((n for n in names if ref == n or ref.startswith(n + ".")), None)
        if part is not None:
            return ref  # already canonical
        head, sep, port = ref.partition(".")
        if not sep:
            return ref
        h = _norm(head)
        hits = [n for n in names if h == norm_names[n] or norm_names[n].endswith("_" + h)]
        if len(hits) != 1:
            hits = [n for n in names if h in norm_names[n]]
        return f"{hits[0]}.{port}" if len(hits) == 1 else ref

    rewritten = []
    changed = False
    for j in plan.joints:
        a, b = _resolve(j.a), _resolve(j.b)
        if a != j.a or b != j.b:
            changed = True
            rewritten.append(j.model_copy(update={"a": a, "b": b}))
        else:
            rewritten.append(j)
    return plan.model_copy(update={"joints": rewritten}) if changed else plan


_PLAN_FIT_MAX_MM = 1.5  # ...farther than this = sloppy beyond any declared running fit


_TRANSIENT_MARKS = (
    "401",
    "unable to connect",
    "connectionrefused",
    "rate limit",
    "overloaded",
    "529",
)


def _call_with_transient_retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Stage-level transient retry (the pump re-run died in 0.1 min to a token-refresh
    401 at RESEARCH -- the per-part author retry never got a chance). One retry after a
    short backoff, only for error text that reads transient; real failures re-raise."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if not any(m in msg for m in _TRANSIENT_MARKS):
            raise
        time.sleep(15.0)
        return fn(*args, **kwargs)


_LR_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_left", "_right"),
    ("_right", "_left"),
    ("_l", "_r"),
    ("_r", "_l"),
)


def unassemblable_reasons(
    fit: dict[str, Any] | None, motion: dict[str, Any] | None, material: Material
) -> list[str]:
    """Findings that mean the object CANNOT BE ASSEMBLED, not that it is imperfect.

    H3 shipped at ``shipped_with_caveats`` while this engine's own report recorded it not-seated,
    with a floating part and a pin fit its own note called "will bind / not assemble". A caveat
    should mean *real but imperfect*; those are no-build conditions, and the flat
    ``clean``/``caveats`` split had no tier to put them in.

    DELIBERATELY NARROW. Every entry is a STRUCTURED field read whose meaning is unambiguous --
    no prose parsing, no aggregate booleans. Several obvious-looking candidates are excluded
    because their flags do NOT mean "impossible" (each exclusion cost a review round):

    * ``assembly_order.stuck`` -- samples straight extraction along +/-X/Y/Z only, and its own
      contract says failure does not PROVE unassemblability (a rotated insertion may work).
    * ``worst_case["issues"]`` -- flags zero-gap FIXED mates, where zero gap is intentional.
    * ``motion["assembly_fails"]`` as an aggregate -- includes merely rattly print-in-place gaps.
    * ``articulates`` -- False also covers a joint that was never validated (absence of proof).
    * interference / ``assembly.static_clearance`` -- a legitimate PRESS FIT is interpenetration
      in nominal geometry, so blocking on it needs an intended-contact distinction we do not
      have. Left as a caveat until that exists.
    * an OVER-WIDE pin gap -- loose and rattly, but it still assembles.

    Fail-open on missing inputs (absent report -> no reason): this decides what SHIPS, and an
    unrun check must never manufacture a blocker.
    """
    reasons: list[str] = []
    if fit is not None:
        if "seated" in fit and not bool(fit["seated"]):
            reasons.append("the assembly does not seat: parts do not reach their mating pose")
        floating = fit.get("floating_parts") or []
        if floating:
            reasons.append(f"part index(es) {list(floating)} float free, attached to nothing")
    if motion is not None and "material_pin_fit" in motion:
        pin = motion["material_pin_fit"]
        gap = pin.get("measured_clearance_mm")
        lo = material.clearance_mm[0]
        # The BIND branch of materials.clearance_fit, by its own threshold. The over-wide branch
        # is deliberately not mirrored here: a loose joint is ugly, not impossible.
        if isinstance(gap, (int, float)) and math.isfinite(float(gap)) and gap < lo - 0.05:
            reasons.append(
                f"pin fit {float(gap):.2f} mm is below {material.name}'s {lo:.2f} mm slip-fit "
                f"floor -- the parts will bind / not assemble as printed"
            )
    return reasons


def motion_blocked_reasons(motion: dict[str, Any] | None) -> list[str]:
    """Findings that mean the mechanism CANNOT MOVE as declared (ADR-0127).

    A peer of :func:`unassemblable_reasons`, deliberately kept separate: a gearbox whose planet
    gears interpenetrate their own pins at -200 deg still ASSEMBLES. It just cannot turn. Calling
    that ``not_assemblable`` would be a different lie than the one ADR-0126 fixed, so the two
    conditions get two dispositions.

    Measured 2026-07-25: ``certificate.motion`` literally reads ``"fail"`` on 6 of the 18 runs
    this engine ever shipped, every one of them a named pair colliding at a named pose with a
    measured penetration depth -- published in the certificate while the disposition read
    ``shipped_with_caveats``.

    Reads ONE structured field: a per-joint ``assembly.sweep.<a>__<b>`` FAIL that classified
    itself ``evidence == "measured"``. That classification (see ``_sweep_evidence``) is what
    makes this safe to gate on -- it excludes a substituted default range and an unmeasurable
    collision depth, the two ways a sweep FAILs without proving anything.

    NOT read, and why:

    * ``motion["assembly_fails"]`` as an AGGREGATE -- it mixes in merely-rattly
      ``print_in_place_gap`` entries (the ADR-0126 exclusion, unchanged).
    * ``certificate.motion == "fail"`` / ``assembly.range_of_motion`` -- the field the archive
      query surfaced, and the wrong one to gate on. It ALSO reads ``fail`` when no gear check
      ran ("under-specified", absence of proof) and when the plan simply declared no moving
      joint at all -- an object that assembles perfectly and merely does not move.
    * ``assembly.sweep_corner.*`` and the coupled/loop-driven sweeps -- real measurements, but
      their drivers do not return depths yet, so they cannot claim ``MEASURED``. They stay
      caveats until they can (the residual gap in ADR-0127).

    Fail-open on missing inputs, same law as ADR-0126: an unrun check must never manufacture a
    blocker.
    """
    if not motion:
        return []
    reasons: list[str] = []
    for fail in motion.get("assembly_fails") or []:
        if not isinstance(fail, dict):
            continue
        if not str(fail.get("id", "")).startswith("assembly.sweep."):
            continue
        if fail.get("evidence") != "measured":
            continue
        reasons.append(
            f"the mechanism is blocked inside its declared range: {fail.get('message', '')}"
        )
    return reasons


def apply_strict_disposition(
    disposition: str,
    fit: dict[str, Any] | None,
    motion: dict[str, Any] | None,
    material: Material,
) -> tuple[str, list[str]]:
    """Downgrade a SHIPPING disposition to ``not_assemblable`` when the object cannot be built,
    or to ``motion_blocked`` when it builds but cannot move as declared.

    One implementation shared by every shipping door, so a door cannot be guarded by a subtly
    different copy (the L83 incident: the accept-consistency gate refused the early accept while
    a second read shipped the same blocked candidate).

    Enabled by default. ``FORGE_STRICT_DISPOSITION=0`` is the explicit rollback switch.
    It only ever downgrades a disposition that was already going to ship: ``rejected`` /
    ``no_build`` are returned untouched, since an honest rejection must not be relabelled.
    """
    if os.getenv("FORGE_STRICT_DISPOSITION", "1") == "0":
        return disposition, []
    if disposition not in ("shipped", "shipped_with_caveats"):
        return disposition, []
    reasons = unassemblable_reasons(fit, motion, material)
    if reasons:
        # Cannot-build outranks cannot-move: if it never goes together, its motion is moot.
        return "not_assemblable", reasons
    blocked = motion_blocked_reasons(motion)
    if blocked:
        return "motion_blocked", blocked
    return disposition, []


def _mirror_sibling(name_lower: str) -> str | None:
    """The expected mirror-sibling name for an L/R-suffixed part (None if not L/R-named).
    Longer suffixes first so '..._left' is never mis-read as '..._l'."""
    for suf, opp in _LR_SUFFIXES:
        if name_lower.endswith(suf):
            return name_lower[: -len(suf)] + opp
    return None


def _plan_mirror_issues(plan: FunctionalPlan) -> list[str]:
    """Roster symmetry (the H3 asymmetric-shin defect class, plan level): every L/R-named
    part must have its mirror SIBLING present in the plan. A lone 'shin_foot_L' with no
    'shin_foot_R' is a real defect the cards can never recover. This is deliberately a
    PLAN-INTERNAL check -- it does NOT require the part to appear in the architecture's
    mirrored_pairs list (the arch is written before decompose and legitimately anticipates
    a different roster; comparing the two measures cross-call name variance, not a defect).
    Dimensional L==R equality is a GEOMETRY check for the autopsy, not the plan."""
    names = {p.name.lower() for p in plan.parts}
    issues: list[str] = []
    for p in plan.parts:
        sib = _mirror_sibling(p.name.lower())
        if sib is not None and sib not in names:
            issues.append(
                f"mirror: part '{p.name}' is L/R-named but its mirror sibling is absent "
                "from the plan (roster asymmetry)"
            )
    return issues


_BOX_WORD_RE = re.compile(r"\bbox\b", re.IGNORECASE)
_BOX_NEGATION_RE = re.compile(
    r"\b(no|not|never|avoid|without|isn't|non|instead\s+of|rather\s+than)\b", re.IGNORECASE
)


def _card_box_is_final_form(sentence: str) -> bool:
    """True when a card SENTENCE presents a bare box as the final form (a box-limb defect),
    False when 'box' is negated ('no bare box', 'not a box'), is the measurement term
    'bounding box', or is a hyphenated structural term ('box-section')."""
    low = sentence.lower()
    if "bounding box" in low or "box-section" in low or "box section" in low:
        return False
    if not _BOX_WORD_RE.search(sentence):
        return False
    return not _BOX_NEGATION_RE.search(sentence)


def _plan_form_issues(plan: FunctionalPlan) -> list[str]:
    """Form-language check (fidelity doctrine, plan level): a card that presents a bare
    'box' as final form without a prohibition is the boxy-part risk. Sentence-scoped so a
    card that says 'rounded-rectangular loft ... (no bare box)' -- an ENGINEERED negation --
    passes (the P2 pelvis false-positive that this replaces)."""
    issues: list[str] = []
    for p in plan.parts:
        for sentence in re.split(r"(?<=[.;:])\s+", p.prompt or ""):
            if _card_box_is_final_form(sentence):
                issues.append(
                    f"form: card '{p.name}' presents a bare box as final form "
                    f"without a prohibition ({sentence.strip()[:80]})"
                )
                break
    return issues


def _plan_dimension_issues(plan: FunctionalPlan, mat: Material) -> list[str]:
    """Plan-time DIMENSION closure (Wave A, row 54's 90-minute lesson): when BOTH mated
    ports declare a diameter, the pair must differ by a sane running-fit band (a moving
    joint) or be flagged -- caught in milliseconds, before any authoring. Also: the two
    ports of one joint must declare the SAME axis station (coincident origins in the
    shared frame) -- a mismatch is a misdeclared joint that the anchor gate would chase
    per-part much later. Ports without diameters are simply not checked (optional field,
    fail-open by design).

    The fit floor DERIVES from the material (TEST_LOG L100 / the F2 postmortem): F2's
    plan legally declared 0.15 mm piston-pin fits, the parts were authored faithfully to
    them, and the run died 14.9 h later at the strict tier's bind branch -- the same
    number was checkable at plan time. The floor is exactly that branch's threshold,
    diametral: ``2 * (mat.clearance_mm[0] - 0.05)`` (see ``unassemblable_reasons``),
    bounded below by the line-to-line minimum. Replayed over every parseable archived
    plan: catches F2's measured killers (0.15/0.20), passes H4/P2's 0.35 pairs the
    strict tier tolerates, and cannot touch the diameterless pan-platform anchors.
    Direction stays untyped (ports do not know bore-from-pin), so the check reads
    |da - db|; any inversion smaller than the floor is caught by magnitude alone."""
    issues: list[str] = []
    ports_by_ref: dict[str, Port] = {}
    for p in plan.parts:
        for pt in p.ports:
            ports_by_ref[f"{p.name}.{pt.name}"] = pt
    moving = ("revolute", "prismatic", "coaxial_insert")
    for j in plan.joints:
        pa, pb = ports_by_ref.get(j.a), ports_by_ref.get(j.b)
        if pa is None or pb is None:
            continue  # the topology gate owns unresolvable endpoints
        oa, ob = pa.origin, pb.origin
        dist = ((oa.x - ob.x) ** 2 + (oa.y - ob.y) ** 2 + (oa.z - ob.z) ** 2) ** 0.5
        if str(j.type) == "gear_mesh":
            # THE one joint whose stations sit APART by design: a mesh's axes differ by
            # the center distance (the gear-train north star's first iteration declared
            # six correct meshes at exactly the designed 30.0 mm and the same-station
            # rule false-rejected the whole plan). The correct-direction gate for a
            # mesh: a NONZERO center distance between PARALLEL axes; the assembly tier
            # then measures the real built center distance (gear_center_distance).
            if dist < 0.5:
                issues.append(
                    f"joint {j.a} <-> {j.b}: meshed gears cannot share an axis station "
                    "-- a gear mesh needs a nonzero center distance between parallel axes"
                )
            else:
                dot = pa.axis.x * pb.axis.x + pa.axis.y * pb.axis.y + pa.axis.z * pb.axis.z
                na = (pa.axis.x**2 + pa.axis.y**2 + pa.axis.z**2) ** 0.5
                nb = (pb.axis.x**2 + pb.axis.y**2 + pb.axis.z**2) ** 0.5
                if na > 0 and nb > 0 and abs(dot) / (na * nb) < 0.996:  # ~5 deg
                    issues.append(
                        f"joint {j.a} <-> {j.b}: spur gear axes must be PARALLEL to mesh "
                        "-- the declared port axes diverge"
                    )
            continue
        if dist > 2.0:
            issues.append(
                f"joint {j.a} <-> {j.b}: the two mated ports sit {dist:.1f} mm apart in the "
                "shared frame -- mated ports must declare the SAME axis station"
            )
        da, db = pa.diameter_mm, pb.diameter_mm
        if da is None or db is None or str(j.type) not in moving:
            continue
        gap = abs(da - db)
        floor = minimum_running_diametral_clearance_mm(mat)
        if gap < floor - 1e-6:  # epsilon: a pair declared exactly AT the floor passes
            tag = "line-to-line -- " if gap < 0.05 else ""
            issues.append(
                f"joint {j.a} ({da:.2f} mm) <-> {j.b} ({db:.2f} mm): {tag}{gap:.2f} mm "
                f"diametral clearance is below {mat.name}'s printable running fit "
                f"({floor:.2f} mm) -- widen the bore to pin + that clearance, or declare "
                "the joint `fixed` if this is a press/retention fit"
            )
        elif gap > _PLAN_FIT_MAX_MM:
            issues.append(
                f"joint {j.a} ({da:.2f} mm) <-> {j.b} ({db:.2f} mm): {gap:.2f} mm diametral "
                "clearance is beyond any running fit -- the parts will slop/fall out"
            )
    return issues


# Body-dim prose signatures per registry hardware kind: a housing card that really
# pockets the actuator states these numbers (H2's torso card did: "MG996R body 40.7 x
# 19.7"). Deterministic, no LLM -- the audit greps the cards it would grep by hand.
_HW_BODY_SIG: dict[str, tuple[str, ...]] = {"mg996r_servo": ("40.7", "19.7")}
_POCKET_WORDS = ("pocket", "bay", "window", "cavity")
# Multi-word retention tokens on purpose: the bare word "shoulder" false-positives on
# every humanoid shoulder pin (measured against the archived H3 plan -- its
# shoulder_pin card says "shoulder bore" and retains nothing).
_RETENTION_TOKENS = (
    "e-clip",
    "circlip",
    "retaining",
    "retain",
    "push-on cap",
    "push cap",
    "cap screw",
    "threaded end",
    "nut pocket",
    "snap ring",
    "shoulder bolt",
    "shoulder screw",
)


def _plan_engineering_issues(
    plan: FunctionalPlan, hardware: list[Any] | None, arch: MechArchitecture | None = None
) -> list[str]:
    """Deterministic ENGINEERING audit of the plan (FORGE_ENG_AUDIT=1; the founder's
    2026-07-23 bar: engineered parts, not plausible sculpture). v1 checks three named
    decisions a professional plan cannot omit: (1) every DECLARED actuator is housed --
    some card pockets its stated body dims; (2) declared actuators couple through a
    HORN somewhere, never friction on the case; (3) every pin part is axially RETAINED.
    Issues feed the same bounded re-decompose as the topology gate and, unresolved,
    the same honest no_build. Calibration: fires on the archived H3 plan (bare pins,
    servos never housed) as TRUE positives; archived no-hardware plans are vacuous on
    (1)/(2); the flag arms per-arm, never retroactively.

    Retention is TYPED first (TEST_LOG L101): a filled ``PartPlan.retention`` satisfies
    (3) outright -- H5 measured the token grep FALSE-FAILING a plan whose retention was
    fully designed (cards modeled shoulder flanges, press caps existed as parts) because
    the vocabulary drifted from H3's. The token path survives only as the fallback for
    plans that predate the field; its correction quotes the architecture's own retention
    decision when ``Joint.name`` resolves the pin to an arch joint."""
    issues: list[str] = []
    cards = {p.name: (p.prompt or "") for p in plan.parts}
    joined_lower = " ".join(cards.values()).lower()
    for decl in hardware or []:
        sig = _HW_BODY_SIG.get(getattr(decl, "kind", ""), ())
        if not sig:
            continue
        housed = any(
            all(s in card for s in sig) and any(w in card.lower() for w in _POCKET_WORDS)
            for card in cards.values()
        )
        if not housed:
            issues.append(
                f"engineering: declared actuator '{getattr(decl, 'hw_id', '?')}' has no housing "
                f"card stating its body pocket ({' x '.join(sig)} mm + stated air) -- bought "
                "actuators are geometry, not prose"
            )
    if (hardware or []) and "horn" not in joined_lower:
        issues.append(
            "engineering: servos are declared but no card couples a driven link to a servo "
            "HORN (state the horn bore circle) -- torque cannot pass through friction on the case"
        )
    arch_ret = {j.name: j.retention for j in arch.joints} if arch is not None else {}
    for p in plan.parts:
        if "pin" not in p.name.lower():
            continue
        if (getattr(p, "retention", None) or "").strip():
            continue  # typed declaration present -- the field IS the decision (L101)
        own = (cards.get(p.name) or "").lower()
        mates = " ".join(c.lower() for n, c in cards.items() if p.name.lower() in c.lower())
        if not any(t in own or t in mates for t in _RETENTION_TOKENS):
            decided = None
            if arch_ret:  # resolvable only through the named-joint carrier (ADR-0130)
                decided = next(
                    (
                        arch_ret[j.name]
                        for j in plan.joints
                        if j.name in arch_ret
                        and any(r == p.name or r.startswith(p.name + ".") for r in (j.a, j.b))
                    ),
                    None,
                )
            issues.append(
                f"engineering: pin part '{p.name}' declares no `retention` -- "
                + (
                    f'the architecture already decided it: copy "{decided}" into the '
                    "part's `retention` field and MODEL it in the card"
                    if decided
                    else "fill the part's `retention` field with the axial retention the "
                    "card models (shoulder + push-on cap, e-clip groove, threaded end + "
                    "nut pocket)"
                )
            )
    return issues


def _plan_topology_issues(
    plan: FunctionalPlan, hardware_ports: dict[str, set[str]] | None = None
) -> list[str]:
    """Deterministic PLAN-TIME topology gate (the i4-finish audit's headline): a mechanism
    plan whose joint graph is incomplete dooms the run ~90 minutes later, when the sweep
    hits the undeclared parts as phantom static obstacles. Checked in milliseconds instead:

      * every joint endpoint must name a DECLARED part and a port that part DECLARES
        (slug-insensitively, matching ``from_part_results``'s resolution);
      * when the plan declares ANY joints, EVERY part must appear in >= 1 joint -- a part
        in zero joints is kinematically dangling (the live i4 plan had rod3/rod4/piston3/
        piston4 in zero joints; 40% of the mechanism was never going to be validated);
      * a part must not RE-DECLARE bought hardware (same name as a mapped hw_id) -- the
        engine places that hardware itself; a printed twin would be authored on top of it.

    A rigid object (no joints declared) has no joint GRAPH to check; the hardware
    collision rule still applies (a jointless plan can still declare a printed twin)."""
    issues: list[str] = []

    def _norm(s: str) -> str:
        return _slug(s).replace("-", "_")

    ports_by_part = {p.name: {_norm(pt.name) for pt in p.ports} for p in plan.parts}
    # ADR-0124: registry-MAPPED hardware (bought servos) are legal joint endpoints too --
    # `<hw_id>.<port>` validated against the ports the bridge actually exposes. Hardware
    # is exempt from the dangling-part rule (a servo is not a plan part).
    ports_by_part.update(
        {hid: {_norm(p) for p in ps} for hid, ps in (hardware_ports or {}).items()}
    )
    hardware_names = set(hardware_ports or {})
    # H4 (TEST_LOG L98): a plan that re-declared 4 bought servos as printed parts died with
    # a FALSE correction -- the hardware entries above shadow same-named parts, every joint
    # naming them resolves to the hardware branch below, and each twin was reported "in NO
    # joint" while sitting in 2. Name the real defect instead, and exempt the twins from
    # the dangling rule (their joints legitimately resolve to the hardware). Checked BEFORE
    # the no-joints return (review-caught): a jointless plan still authors its twins.
    collided = {p.name for p in plan.parts} & hardware_names
    for name in sorted(collided):
        issues.append(
            f"part '{name}' re-declares bought hardware '{name}' as a printed part -- that "
            f"hardware is placed by the engine, not printed, and its ports "
            f"{sorted(ports_by_part[name])} are already legal joint endpoints. REMOVE "
            f"'{name}' from `parts` and keep the joints that reference it."
        )
    if not plan.joints:
        return issues
    names = sorted(ports_by_part, key=len, reverse=True)  # longest-prefix: names may contain '.'
    in_joints: set[str] = set()
    for j in plan.joints:
        for ref in (j.a, j.b):
            part = next((n for n in names if ref == n or ref.startswith(n + ".")), None)
            if part is None:
                issues.append(f"joint endpoint '{ref}' references a part the plan does not declare")
                continue
            if part in hardware_names:
                port = ref[len(part) + 1 :] if len(ref) > len(part) else ""
                if not port or _norm(port) not in ports_by_part[part]:
                    issues.append(
                        f"joint endpoint '{ref}': hardware '{part}' exposes ports "
                        f"{sorted(ports_by_part[part])}, not '{port or '(none)'}'"
                    )
                continue
            in_joints.add(part)
            port = ref[len(part) + 1 :] if len(ref) > len(part) else ""
            if not port:
                issues.append(f"joint endpoint '{ref}' names no port on part '{part}'")
            elif _norm(port) not in ports_by_part[part]:
                issues.append(
                    f"joint endpoint '{ref}': part '{part}' does not declare port '{port}' "
                    f"(declared: {sorted(ports_by_part[part]) or 'none'}) -- add the port to "
                    "the part (ports allow up to 20 per part)"
                )
    for p in plan.parts:
        if p.name not in in_joints and p.name not in collided:
            issues.append(
                f"part '{p.name}' appears in NO joint -- it is kinematically dangling and its "
                "motion can never be validated. Join it (revolute/prismatic if it moves, "
                "fixed/coaxial_insert if it is rigidly mounted)."
            )
    return issues


def _plan_actuation_issues(plan: FunctionalPlan, arch: MechArchitecture) -> list[str]:
    """ADR-0130: deterministic arch-to-plan ACTUATION reconciliation. P1/P2 declared 11
    actuators and bound zero; H4 prompt-demanded the servo joints verbatim and the plan
    still died -- prose does not bind (instance #7). The binding is typed instead: the
    plan joint realizing an architecture joint carries ``Joint.name``, and each
    ``ActuatorPlacement`` must reconcile:

      * exactly one plan joint named ``placement.joint`` -- REVOLUTE, coupling
        ``<hw_id>.horn`` to the DECLARED driven link (holes 1+2+4 of the L93 spec);
      * a rigid mount joint on ``<hw_id>.mount`` whose housing part is NOT the same
        rigid body as the driven link (fixed/coaxial union-find -- hole 5: a servo
        whose case and horn hang off one link drives nothing);
      * plan-side names unique and drawn from the architecture only (hole 6's plan half).

    Every issue is a copy-actionable correction (law 3: bound at the failure point);
    they ride the same bounded re-decompose as the other plan gates."""
    issues: list[str] = []
    if not arch.actuators:
        return issues

    def _norm(s: str) -> str:
        return _slug(s).replace("-", "_")

    part_names = {p.name for p in plan.parts}
    entities = sorted(part_names | {a.hw_id for a in arch.actuators}, key=len, reverse=True)

    def _entity(ref: str) -> str | None:
        return next((n for n in entities if ref == n or ref.startswith(n + ".")), None)

    def _port_of(ref: str, ent: str) -> str:
        return ref[len(ent) + 1 :] if len(ref) > len(ent) else ""

    # plan-side identity hygiene: unique, and only names the architecture declares
    arch_names = {j.name for j in arch.joints}
    named = [(j.name, j) for j in plan.joints if j.name is not None]
    counts: dict[str, int] = {}
    for n, _j in named:
        counts[n] = counts.get(n, 0) + 1
    for n in sorted(counts):
        if counts[n] > 1:
            issues.append(
                f"joint name '{n}' appears on {counts[n]} plan joints -- a name "
                "identifies ONE joint"
            )
        if n not in arch_names:
            issues.append(
                f"joint name '{n}' is not an architecture joint (declared: {sorted(arch_names)})"
            )
    by_name = {n: j for n, j in named if counts[n] == 1}

    # rigid components over fixed/coaxial_insert -- the codebase's own "rigidly
    # mounted" vocabulary (the dangling-rule message teaches exactly this set)
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        parent[_find(x)] = _find(y)

    for j in plan.joints:
        if j.type in (JointType.FIXED, JointType.COAXIAL_INSERT):
            ea, eb = _entity(j.a), _entity(j.b)
            if ea and eb:
                _union(ea, eb)

    for a in arch.actuators:
        # NO early-out on any single failure (review-caught): the bounded replan runs
        # ONCE, so every placement reports ALL its defects in the same pass -- a staged
        # diagnosis would convert a fixable plan into a no_build on the second gate.
        if a.horn_link not in part_names:
            issues.append(
                f"actuator '{a.hw_id}': declared driven link '{a.horn_link}' is not a plan "
                f"part -- the plan must declare '{a.horn_link}' (the horn couples to it)"
            )
        named_j = by_name.get(a.joint)
        if named_j is None:
            # a "just name it" correction is only complete when the joint is ALREADY the
            # full horn shape (revolute, through <hw>.horn, to the declared link) --
            # review-caught: anything looser stages the diagnosis across two gates.
            near = next(
                (
                    x
                    for x in plan.joints
                    if x.name is None
                    and x.type is JointType.REVOLUTE
                    and any(
                        _entity(r) == a.hw_id and _norm(_port_of(r, a.hw_id)) == "horn"
                        for r in (x.a, x.b)
                    )
                    and {_entity(x.a), _entity(x.b)} == {a.hw_id, a.horn_link}
                ),
                None,
            )
            if near is not None:
                issues.append(
                    f"the joint '{near.a}' <-> '{near.b}' realizes actuated joint "
                    f"'{a.joint}' but carries no name -- set name='{a.joint}' on it"
                )
            else:
                issues.append(
                    f"actuated joint '{a.joint}' (driven by {a.hw_id}, horn on "
                    f"'{a.horn_link}') is not realized: add a revolute joint carrying "
                    f"name='{a.joint}' with a='{a.hw_id}.horn', b='{a.horn_link}.<port>'"
                )
        else:
            ea, eb = _entity(named_j.a), _entity(named_j.b)
            horn_side = named_j.a if ea == a.hw_id else named_j.b if eb == a.hw_id else None
            other_ent = eb if ea == a.hw_id else ea
            if named_j.type is not JointType.REVOLUTE:
                issues.append(
                    f"joint named '{a.joint}' must be revolute (it is '{named_j.type}') -- it "
                    f"realizes the {a.hw_id} horn drive"
                )
            if horn_side is None or _norm(_port_of(horn_side, a.hw_id)) != "horn":
                issues.append(
                    f"joint named '{a.joint}' must couple '{a.hw_id}.horn' (endpoints: "
                    f"'{named_j.a}' <-> '{named_j.b}') -- the horn is the drive interface"
                )
            elif other_ent is not None and other_ent != a.horn_link:
                issues.append(
                    f"joint named '{a.joint}': the horn must couple declared driven link "
                    f"'{a.horn_link}', not '{other_ent}'"
                )
        mounts = [
            x
            for x in plan.joints
            if x.type in (JointType.FIXED, JointType.COAXIAL_INSERT)
            and any(
                _entity(r) == a.hw_id and _norm(_port_of(r, a.hw_id)) == "mount" for r in (x.a, x.b)
            )
        ]
        if not mounts:
            issues.append(
                f"actuator '{a.hw_id}' has no rigid mount: add a fixed joint "
                f"'<housing_part>.<port>' -> '{a.hw_id}.mount' -- house the case on the "
                f"reference side of joint '{a.joint}'"
            )
        for x in mounts:
            ea, eb = _entity(x.a), _entity(x.b)
            m = eb if ea == a.hw_id else ea
            other_ref = x.b if ea == a.hw_id else x.a
            if m is None or m not in part_names:
                # review-caught false PASS: a mount ending on other hardware (or on the
                # actuator itself) anchored nothing, yet counted as mounted.
                issues.append(
                    f"actuator '{a.hw_id}': mount joint '{x.a}' <-> '{x.b}' must land on a "
                    f"printed housing part, not '{other_ref}' -- anchor the case to the "
                    f"structure"
                )
            elif _find(m) == _find(a.horn_link):
                issues.append(
                    f"actuator '{a.hw_id}' is mounted on '{m}', the SAME rigid body as its "
                    f"driven link '{a.horn_link}' (fixed/coaxial chain) -- the servo would "
                    f"drive nothing; mount the case on the reference side of joint '{a.joint}'"
                )
    return issues


def decompose_functional(
    prompt: str,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    brief: DesignBrief | None = None,
    extra_callout: str | None = None,
) -> tuple[FunctionalPlan, LLMUsage]:
    """Decompose a multi-part object into parts in a shared mate frame (one cheap call).

    In CHIEF mode (FORGE_CHIEF=1) the same schema is authored under the chief-engineer
    overlay -- exact build cards, zero drafter freedom -- and the model defaults to
    :data:`BRAIN_MODEL` (the brain; Opus today) instead of SONNET.

    ``brief`` (optional): the whole-object brief. Its visual description IS the acceptance
    judge's reference, so the planner must see it -- row 55's i4 was rejected because the
    brief promised side windows the plan never built (brief-to-plan contract drift)."""
    client = client or make_client()
    chief = _chief_enabled()
    # Overlay flags (PLAN doctrine = the L86 walls; FIDELITY doctrine = the L88 form
    # classes) compose in _decompose_system; all default OFF = byte-identical prompt.
    system = _decompose_system()
    # max_tokens generous: a multi-part plan with per-part ports + joints is a sizeable JSON
    # object, and a truncated response is an honest failure the caller catches. History: a
    # 1500-token cap truncated a 3-part gear plan; 4000 then truncated a robotic-arm plan the
    # moment the pin-is-a-part rule made every revolute joint carry its own pin part (the v2b
    # decompose death). Billing is per ACTUAL token -- headroom only lets long plans complete.
    user = f"Object: {prompt}\n\nDecompose it into separate parts in a shared origin mate frame."
    if brief is not None and getattr(brief, "visual_description", None):
        user += (
            "\n\nTHE JUDGE'S REFERENCE (the finished assembly is ACCEPTED or REJECTED "
            "against this description -- every promised view/window/exposure must exist "
            f"in your cards):\n{brief.visual_description}"
        )
    rules = select_design_rules(prompt)
    if rules:
        user += f"\n\nREFERENCE DESIGN RULES (apply; show the arithmetic in the cards):\n{rules}"
    if extra_callout:
        user += f"\n\n{extra_callout}"
    return client.generate_structured(
        system=system,
        user=user,
        output_model=FunctionalPlan,
        model=model or (BRAIN_MODEL if chief else SONNET),
        max_tokens=PLAN_MAX_TOKENS,
    )


@dataclass
class FunctionalResult:
    """Outcome of a functional multi-part build: the per-part loops + the assembled whole."""

    out_dir: Path
    object_name: str
    parts: list[tuple[str, LoopResult]] = field(default_factory=list)
    assembled_stl: Path | None = None
    assembled_step: Path | None = None
    assembled_glb: Path | None = None
    assembled_render: Path | None = None
    verdict: AcceptanceVerdict | None = None
    disposition: str = "no_build"
    total_cost_usd: float = 0.0
    whole_brief: DesignBrief | None = None
    fit: dict[str, Any] | None = None
    motion: dict[str, Any] | None = None
    failure: str | None = None  # WHY a no_build degraded (the i4 probe's silent-swallow lesson)
    repair_note: str | None = None  # WHY the bounded repair round did/didn't act (row-56 no-fire)
    finish: list[dict[str, Any]] | None = None  # per-part finish decisions (audit P2: 'chose
    # nothing' must be auditable, not indistinguishable from 'silently failed')


_ANCHOR_PAD_MM = 5.0
# Every part gets at least this much budget so a many-part object never starves its late
# parts to $0 (the terse-probe rods never drafted). Notional on the subscription path.
_PER_PART_BUDGET_FLOOR = 0.35
# A part that comes back author_failed/brief_failed with 0 iterations is an intermittent CLI/
# rate-limit blip -- retry it (bounded) with a backoff so the limit recovers.
_AUTHOR_RETRIES = 2
_AUTHOR_BACKOFF_S = float(os.environ.get("FORGE_AUTHOR_BACKOFF_S", "20"))


def _assembled_envelope_issue(
    meshes: list[Any], prompt: str, brief: DesignBrief | None
) -> str | None:
    """Wave A (row 54: 'assembled envelope 163mm silently exceeded the stated 160'):
    compare the ASSEMBLED bbox against the stated envelope budget (parsed from the
    prompt, falling back to the brief text). Per-part checks each passed while the
    assembly quietly outgrew the budget. None = within budget / no budget stated."""
    if not meshes:
        return None
    from forge_mind.agentic.brief import parse_envelope_mm

    budget = parse_envelope_mm(prompt) or (
        parse_envelope_mm(getattr(brief, "visual_description", "") or "") if brief else None
    )
    if budget is None:
        return None
    import numpy as np

    bounds = np.vstack([m.bounds for m in meshes])
    ext = sorted(float(x) for x in (bounds.max(axis=0) - bounds.min(axis=0)))
    limit = sorted(budget)
    over = [(e, b) for e, b in zip(ext, limit, strict=True) if e > b + 1.0]
    if not over:
        return None
    return (
        f"assembled envelope {ext[2]:.0f} x {ext[1]:.0f} x {ext[0]:.0f} mm exceeds the stated "
        f"budget {limit[2]:.0f} x {limit[1]:.0f} x {limit[0]:.0f} mm"
    )


def joint_mobility_counts(plan: FunctionalPlan) -> tuple[int, int]:
    """``(moving, rigid)`` joints. One definition, shared by the mobility note and the motion
    verdict so the prose and the numbers cannot drift apart.

    This exists because they DID drift in the reading: TEST_LOG L89 recorded H3 as "motion 13/13
    joints full-range PASS" when the engine's own message said "All 2 moving joint(s) sweep their
    full range". ``joints_validated`` counts joints EXAMINED, and a fixed joint passes a sweep
    trivially because it has no range -- so 13/13 was true and meant almost nothing. H3 was 2
    moving and 11 rigid: a mannequin logged as a fully articulated humanoid.
    """
    moving = sum(1 for j in plan.joints if str(j.type) in ("revolute", "prismatic"))
    rigid = sum(1 for j in plan.joints if str(j.type) in ("fixed", "coaxial_insert"))
    return moving, rigid


def undriven_dof_lower_bound(plan: FunctionalPlan, hardware: list[Any] | None) -> int:
    """A CONSERVATIVE lower bound on degrees of freedom nothing can drive.

    H3 -- a 350 mm humanoid with 13 joints -- declared ZERO hardware, so every DOF it had was
    undriven, yet the report said ``articulates: true`` and ``motion_status: pass``. Both were
    literally true and materially false: a sweep proves a joint does not COLLIDE, never that
    anything can move it. An unactuated revolute is a free hinge that flops.

    The bound needs no actuator-to-joint mapping (``HardwareDecl`` carries kind/id/transform and
    is not bound to a joint), because it rests on counting alone: you cannot drive N degrees of
    freedom with fewer than N actuators. Since actuators <= total declared hardware,
    ``moving - len(hardware)`` is a valid lower bound on undriven DOF -- it UNDERSTATES the
    problem when some hardware is a bearing or a screw, which is the safe direction for a
    reported number.

    REPORT-TIER, never a gate: a hinge, a clamp, a manual pivot are legitimately undriven, so a
    positive count is information, not a defect. It is the number that separates a mechanism from
    a mannequin -- H3 reads 2, the L85 pan platform reads 0.
    """
    moving, _rigid = joint_mobility_counts(plan)
    return max(0, moving - len(hardware or []))


def _mobility_note(plan: FunctionalPlan) -> str:
    """Advisory planar Gruebler count (Wave A, external research #1): M = 3(n-1) - 2*j1.
    REPORT-TIER ONLY -- multi-loop parallel mechanisms (a 4-cylinder engine) are
    over-constrained by design and still turn, so this number informs, never gates."""
    n = len(plan.parts)
    j1, rigid = joint_mobility_counts(plan)
    m = 3 * (n - 1 - rigid) - 2 * j1 if n > 1 else 0
    return (
        f"planar Gruebler M={m} (n={n}, moving joints={j1}, rigid mounts={rigid}); "
        "M=1 is a clean single-DOF mechanism; M<1 usually means over-constrained-by-design "
        "parallel loops (fine if the coupled sweep passes); M>1 means extra DOFs to drive"
    )


def material_condition_margin_mm(mat: Material) -> float:
    """Worst-case as-printed clearance LOSS per mated pair (Wave B). Keyed on the
    material's PROCESS, per founder direction -- printed parts only:
      FDM: holes print undersized + shafts oversized, ~0.2-0.5 mm combined loss
      (published, calibrated conservative-middle here); RESIN: near-nominal; METAL and
      everything else: 0.0 -- machined metal holds IT7-class tolerances, so nominal
      certification stands and this check is a NO-OP."""
    from forge_mind.materials import ProcessCategory

    if mat.process is ProcessCategory.FDM:
        return 0.25
    if mat.process is ProcessCategory.RESIN:
        return 0.10
    return 0.0


def _worst_case_issues(
    fit: dict[str, Any] | None,
    motion: dict[str, Any] | None,
    mat: Material,
    part_names: list[str],
) -> dict[str, Any] | None:
    """Wave B: the mechanism must still clear AFTER the printer eats the tolerance.
    Deterministic home-pose v1: every touching pair's measured surface gap and every
    measured pin radial clearance must exceed the process's worst-case loss. (Nominal
    checks already passed by the time this runs; this asks 'and as-printed?'.) None =
    no printed-process margin applies (metal/none) -- the check is honestly absent, not
    silently passed."""
    margin = material_condition_margin_mm(mat)
    if margin <= 0.0:
        return None
    issues: list[str] = []
    for pair in (fit or {}).get("pairs", []):
        if not pair.get("touches") or float(pair.get("interference_mm3", 0.0)) > 1.0:
            continue
        gap = float(pair.get("min_gap_mm", 99.0))
        if 0.0 <= gap < margin:
            i, j = pair.get("pair", [0, 0])
            a = part_names[i] if i < len(part_names) else f"part{i}"
            b = part_names[j] if j < len(part_names) else f"part{j}"
            issues.append(
                f"{a} <-> {b}: {gap:.2f} mm gap < {margin:.2f} mm worst-case as-printed "
                "loss -- will bind as printed"
            )
    pin = ((motion or {}).get("checks") or {}).get("pin_clearance")
    if isinstance(pin, dict):
        measured = _parse_mm(pin.get("actual"))
        if measured is not None and measured < margin:
            issues.append(
                f"pin radial clearance {measured:.2f} mm < {margin:.2f} mm worst-case "
                "as-printed loss -- pins will bind as printed"
            )
    return {"margin_mm": margin, "process": str(mat.process), "issues": issues}


def _cell_profile_enabled() -> bool:
    return os.environ.get("FORGE_CELL_PROFILE", "").strip().lower() in ("1", "true", "yes")


def _apply_grasp_tabs(
    part_results: list[tuple[str, LoopResult]],
    out_dir: Path,
    part_ids: list[str],
    mat: Material,
    brief: DesignBrief | None,
) -> list[tuple[str, LoopResult]]:
    """Wave D / PrintCell constitution: 'ForgeMind adds a standardized grasp tab to
    EVERY generated part' -- the multi-part mechanism path never called the (existing,
    fail-closed) emit_grasp_tab. Cell-profile only; a refused tab keeps the clean part
    (the arm handles it as an exception, honestly reported)."""
    if not _cell_profile_enabled():
        return part_results
    from forge_mind.agentic.grasp_tab import emit_grasp_tab

    out: list[tuple[str, LoopResult]] = []
    for ix, (name, lr) in enumerate(part_results):
        if lr.final_stl is None or lr.final_step is None or not lr.final_step.exists():
            out.append((name, lr))
            continue
        tab_dir = out_dir / f"{part_ids[ix]}_grasp_tab"
        res = emit_grasp_tab(lr.final_step, lr.final_stl, tab_dir, material=mat, brief=brief)
        if getattr(res, "ok", False):
            out.append((name, replace(lr, final_stl=res.stl_path, final_step=res.step_path)))
            print(f"grasp-tab: '{name}' tabbed (cell profile)", flush=True)
        else:
            out.append((name, lr))
            print(
                f"grasp-tab: '{name}' REFUSED -- {getattr(res, 'reason', 'unknown')}",
                flush=True,
            )
    return out


def _print_pack(part_results: list[tuple[str, LoopResult]]) -> list[dict[str, Any]] | None:
    """Wave D: per-part recommended print orientation (report-tier). For each of the six
    axis-aligned orientations, the support proxy = total area of down-facing overhang
    faces (normal more than 45 deg below horizontal, off the bed). Least overhang wins.
    Deterministic, advisory -- the slicer gate owns hard printability verdicts."""
    import numpy as np
    import trimesh

    _rot: Any = trimesh.transformations.rotation_matrix
    rows: list[dict[str, Any]] = []
    axes = {
        "+Z down (as modeled)": np.eye(3),
        "-Z down (flipped)": _rot(np.pi, [1, 0, 0])[:3, :3],
        "+X down": _rot(np.pi / 2, [0, 1, 0])[:3, :3],
        "-X down": _rot(-np.pi / 2, [0, 1, 0])[:3, :3],
        "+Y down": _rot(-np.pi / 2, [1, 0, 0])[:3, :3],
        "-Y down": _rot(np.pi / 2, [1, 0, 0])[:3, :3],
    }
    for name, lr in part_results:
        if lr.final_stl is None or not lr.final_stl.exists():
            continue
        try:
            m: Any = trimesh.load(str(lr.final_stl), force="mesh")
            best = None
            for label, rot in axes.items():
                n = m.face_normals @ rot.T
                z = n[:, 2]
                lo_z = (m.triangles_center @ rot.T)[:, 2]
                floor = float(lo_z.min()) + 0.6
                overhang = (z < -0.7071) & (lo_z > floor)  # >45 deg down, off the bed
                area = float(m.area_faces[overhang].sum())
                if best is None or area < best[1]:
                    best = (label, area)
            if best is not None:
                rows.append(
                    {"part": name, "orientation": best[0], "overhang_mm2": round(best[1], 1)}
                )
        except Exception:
            continue
    return rows or None


def _port_anchor_callout(part: PartPlan, lr: LoopResult) -> str | None:
    """A measured callout when a ported part's BUILT geometry is nowhere near its declared
    port origins -- the dropped-shared-frame family (row 55: conrod_3 and piston_4 authored
    at x=0 'local frame' instead of stations 107/147; 5 of 15 parts). Deterministic and
    cheap: every declared port origin must sit inside the part's padded bbox. None = anchored."""
    if not part.ports or lr.final_stl is None or not lr.final_stl.exists():
        return None
    import trimesh

    try:
        m = trimesh.load(str(lr.final_stl), force="mesh")
        lo, hi = m.bounds
    except Exception:
        return None  # unmeasurable here; the assembly-level checks own it
    for port in part.ports:
        o = port.origin
        if not (
            lo[0] - _ANCHOR_PAD_MM <= o.x <= hi[0] + _ANCHOR_PAD_MM
            and lo[1] - _ANCHOR_PAD_MM <= o.y <= hi[1] + _ANCHOR_PAD_MM
            and lo[2] - _ANCHOR_PAD_MM <= o.z <= hi[2] + _ANCHOR_PAD_MM
        ):
            return (
                f"MEASURED PLACEMENT ERROR: the built part spans "
                f"X {lo[0]:.1f}..{hi[0]:.1f}, Y {lo[1]:.1f}..{hi[1]:.1f}, "
                f"Z {lo[2]:.1f}..{hi[2]:.1f} mm, but its declared port '{port.name}' sits at "
                f"({o.x:.1f}, {o.y:.1f}, {o.z:.1f}) -- the part was NOT authored at its "
                "shared-frame station. Rebuild the SAME part AT that station (do not move "
                "the port)."
            )
    return None


def _resume_enabled() -> bool:
    return os.environ.get("FORGE_RESUME_PARTS", "1").strip().lower() not in ("0", "false", "no")


def _card_prompt(part: PartPlan) -> str:
    """The prompt the author actually receives: the card, plus the typed retention
    decision rendered at the last mile (L101: typed carrier -> geometry instruction).
    Byte-identical to the card when the field is unset, so legacy checkpoints keep
    their keys."""
    if (part.retention or "").strip():
        return (
            f"{part.prompt}\n\nAXIAL RETENTION (BINDING -- the architecture's decision; "
            f"MODEL it in this part's geometry): {part.retention}"
        )
    return part.prompt


def _part_key(part: PartPlan, mat: Material) -> str:
    import hashlib

    from forge_mind.agentic.build_cache import _material_prompt_mode, _spec_dims_mode
    from forge_mind.validation_provenance import validation_fingerprint

    # Prompt-shaping flags ride the RESUME key too (review-caught): FORGE_RESUME_PARTS is
    # default-ON, so re-running a functional job in the same out dir loads checkpoints BEFORE
    # the pipeline -- flipping a prompt flag would resume pre-arm geometry and score it as the
    # armed arm. Appended only when a flag is ARMED, so legacy keys stay byte-identical off.
    #
    # KNOWN, BOUNDED GAP (second review-catch, decided deliberately): a checkpoint written
    # while a flag was armed BEFORE this partition existed would carry the unsuffixed hash, so
    # an off-mode rerun could still load it. Bumping the off-mode key would close that -- at the
    # cost of invalidating EVERY archived checkpoint (H5B's 24 parts and every other run's),
    # which are live evidence artifacts. Exposure here is zero: neither flag has ever been
    # armed in any run config (verified by grep across benchmarks/*/launch configs; both
    # shipped dark hours ago). If either is armed before this line changes, bump the off-mode
    # key in the same change.
    suffix = ""
    if _spec_dims_mode():
        suffix += "\x00specdims"
    if _material_prompt_mode():
        suffix += "\x00material"
    identity = f"{_card_prompt(part)}\x00{mat.name}{suffix}\x00{validation_fingerprint()}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _save_part_checkpoint(part_dir: Path, part: PartPlan, lr: LoopResult, mat: Material) -> None:
    """Per-part checkpoint (audit P2): a session boundary killing the process at part 8/10
    used to cost ALL parts on relaunch (FORGE_REUSE_PLAN only saved the plan). Best-effort:
    a checkpoint that cannot be written never fails the run."""
    try:
        from forge_mind.validation_provenance import file_sha256, validation_fingerprint

        artifact_paths = {
            "final_stl": lr.final_stl,
            "final_step": lr.final_step,
            "final_render": lr.final_render,
        }
        payload = {
            "key": _part_key(part, mat),
            "validation_fingerprint": validation_fingerprint(),
            "artifact_sha256": {
                name: file_sha256(Path(path))
                for name, path in artifact_paths.items()
                if path is not None and Path(path).exists()
            },
            "stopped_reason": lr.stopped_reason,
            "final_stl": str(lr.final_stl) if lr.final_stl else None,
            "final_step": str(lr.final_step) if lr.final_step else None,
            "final_render": str(lr.final_render) if lr.final_render else None,
            "hard_ok": (
                bool(lr.validation.hard_ok)
                if lr.validation is not None
                else bool(getattr(lr, "resumed_hard_ok", False))
            ),
            "geometry_passed": bool(lr.geometry_passed),
            "critic_approved": bool(lr.critic_approved),
        }
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / "part_checkpoint.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _load_part_checkpoint(part_dir: Path, part: PartPlan, mat: Material) -> LoopResult | None:
    """Reload a finished part from a prior (killed) run in the SAME out_dir -- 0 LLM calls.
    Keyed on prompt+material so a changed plan can never resurrect a stale body; the STL
    must still exist. FORGE_RESUME_PARTS=0 disables. Fail-open to re-authoring on any doubt."""
    if not _resume_enabled():
        return None
    ck = part_dir / "part_checkpoint.json"
    if not ck.exists():
        return None
    try:
        from forge_mind.validation_provenance import file_sha256, validation_fingerprint

        payload = json.loads(ck.read_text(encoding="utf-8"))
        if payload.get("key") != _part_key(part, mat):
            return None
        if payload.get("validation_fingerprint") != validation_fingerprint():
            return None
        stl = Path(payload["final_stl"]) if payload.get("final_stl") else None
        if stl is None or not stl.exists():
            return None
        step = Path(payload["final_step"]) if payload.get("final_step") else None
        render = Path(payload["final_render"]) if payload.get("final_render") else None
        hashes = payload.get("artifact_sha256") or {}
        paths = {"final_stl": stl, "final_step": step, "final_render": render}
        if "final_stl" not in hashes or set(hashes) - set(paths):
            return None
        for name, expected in hashes.items():
            path = paths[name]
            if path is None or not path.exists() or file_sha256(path) != expected:
                return None
        return LoopResult(
            prompt=part.prompt,
            brief=None,
            final_code=None,
            final_stl=stl,
            final_step=step if step is not None and step.exists() else None,
            final_render=render if render is not None and render.exists() else None,
            validation=None,
            critique=None,
            geometry_passed=bool(payload.get("geometry_passed", False)),
            critic_approved=bool(payload.get("critic_approved", False)),
            stopped_reason=str(payload.get("stopped_reason", "accepted")),
            iterations=[],
            resumed_hard_ok=bool(payload.get("hard_ok", False)),
        )
    except Exception:
        return None


def _deterministic_seat(
    part: PartPlan, lr: LoopResult, part_id: str, out_dir: Path
) -> LoopResult | None:
    """Zero-LLM anchor fix (audit P1): the dominant anchor failure is a part authored in
    its LOCAL frame instead of its shared-frame station -- a pure TRANSLATION error, and
    the gate itself is a free oracle. Try a small candidate set of translations (each
    port origin vs the bbox center, plus their mean); accept the first that puts EVERY
    declared port inside the padded bbox (the same predicate the gate checks), re-export
    STL + STEP, and hand back the seated result. Self-gating: wrong orientation, wrong
    feature size, or wrong multi-port spacing admits no translation that passes all
    ports -> None, and the LLM re-author path stands. The downstream assembly checks
    (fit, coaxial, sweep) still own the real verification either way."""
    if not part.ports or lr.final_stl is None or not lr.final_stl.exists():
        return None
    if lr.final_step is None or not lr.final_step.exists():
        return None  # the STEP rides through assembly/pin/finish; without it, re-author
    import numpy as np
    import trimesh

    try:
        mesh = trimesh.load(str(lr.final_stl), force="mesh")
        lo, hi = (np.asarray(v, dtype=float) for v in mesh.bounds)
    except Exception:
        return None
    center = (lo + hi) / 2.0
    origins = np.array([[p.origin.x, p.origin.y, p.origin.z] for p in part.ports], dtype=float)
    candidates = [o - center for o in origins]
    if len(origins) > 1:
        candidates.insert(0, origins.mean(axis=0) - center)

    def _all_ports_in(t: Any) -> bool:
        tlo, thi = lo + t, hi + t
        return bool(
            np.all(origins >= tlo - _ANCHOR_PAD_MM) and np.all(origins <= thi + _ANCHOR_PAD_MM)
        )

    t = next((c for c in candidates if _all_ports_in(c)), None)
    if t is None:
        return None
    try:
        from build123d import Location, Vector, export_step, export_stl, import_step

        seat_dir = out_dir / f"{part_id}_anchor_seat"
        seat_dir.mkdir(parents=True, exist_ok=True)
        out_stl = seat_dir / lr.final_stl.name
        out_step = seat_dir / lr.final_step.name
        solid = import_step(str(lr.final_step)).moved(
            Location(Vector(float(t[0]), float(t[1]), float(t[2])))
        )
        export_stl(solid, str(out_stl))
        export_step(solid, str(out_step))
        if not (out_stl.exists() and out_step.exists()):
            return None
    except Exception:
        return None
    print(
        f"anchor: '{part.name}' seated deterministically "
        f"(dx={t[0]:+.1f}, dy={t[1]:+.1f}, dz={t[2]:+.1f} mm, 0 LLM calls)",
        flush=True,
    )
    return replace(lr, final_stl=out_stl, final_step=out_step)


def _finish_enabled() -> bool:
    from forge_mind.agentic.detail import finish_enabled

    return finish_enabled()


def _mating_exclusions(part: Any, plan: FunctionalPlan) -> list[str]:
    """Human-readable mating surfaces for a part: each declared port (a bore/pin axis) + the
    joints that touch it. The planner must keep finish clear of these (they set the fit)."""
    ex: list[str] = []
    for p in getattr(part, "ports", []) or []:
        o = p.origin
        ex.append(f"port '{p.name}' at ({o.x:.0f},{o.y:.0f},{o.z:.0f}) (a mating axis/bore)")
    return ex


def _finished_part_not_worse(
    lr: LoopResult, out_stl: Path, out_step: Path
) -> tuple[bool, str, Any]:
    """May the FINISHED geometry replace the clean part? Same gates, same brief, fail-closed.

    The finish stage's ship gate was assembly-fit-only, so a finish op that broke a PART-level
    property (watertightness, a wall floor, an envelope) would have ridden into the report under
    the clean body's stale per-part PASS. This re-runs `validate_authored_part` on the finished
    files and requires NOT-WORSE:

      * always: the functional floor -- one connected watertight solid (`hard_ok`);
      * if the clean part fully PASSED, the finished part must fully pass too.

    A part that was best-at-cap (built, hard_ok, not passed) may stay imperfect -- finish must
    not be reverted for defects it did not cause. No brief (nothing to validate against) or a
    revalidation crash -> (False, why, None): polish is optional, so the clean part is always
    the safe answer.

    Returns ``(ok, why, finished_validation | None)`` -- the fresh validation rides out
    (review-caught) so an accepting caller can STORE it: keeping the clean body's validation on
    finished files left every downstream consumer (reports, mass/cost, cache manifests,
    training labels) describing pre-finish geometry.
    """
    if lr.brief is None:
        return False, "no brief to revalidate against", None
    try:
        from forge_mind.agentic.validate import validate_authored_part

        v = validate_authored_part(out_stl, out_step, lr.brief, author_code=lr.final_code)
    except Exception as exc:  # fail-closed: unknown state never ships polished
        return False, f"revalidation raised {type(exc).__name__}", None
    if not v.hard_ok:
        return False, "finished part is no longer one watertight solid", None
    clean_passed = bool(lr.validation.passed) if lr.validation is not None else False
    if clean_passed and not v.passed:
        return False, "clean part passed validation; finished part does not", None
    return True, "not-worse", v


def _apply_designed_finish(
    part_results: list[tuple[str, LoopResult]],
    plan: FunctionalPlan,
    out_dir: Path,
    part_ids: list[str],
    *,
    client: LLMClientLike | None,
    model: str | None,
) -> tuple[list[tuple[str, LoopResult]], list[dict[str, Any]]]:
    """Plan + apply designed finish per part, then keep the finished set ONLY if the assembly
    fit is unchanged-or-better (else revert). Never raises; a part whose finish fails keeps its
    clean body.

    Returns (part_results, telemetry). The telemetry records EVERY per-part decision -- the
    i4-finish run's lesson: 10 silent empty plans were indistinguishable from 10 silent
    planner failures without re-probing. 'Chose nothing' is now an explicit, auditable row."""
    import trimesh

    from forge_mind.agentic.finish_planner import (
        PartFinishContext,
        finish_step_file,
        fit_not_worse,
        plan_part_finish,
    )

    telemetry: list[dict[str, Any]] = []
    cache: dict[tuple[tuple[str, float], tuple[str, float]], dict[str, Any] | None] = {}
    clean_meshes, clean_keys = _load_part_meshes_keyed(part_results)
    clean_fit = _assembly_fit(clean_meshes, keys=clean_keys, cache=cache)
    finished: list[tuple[str, LoopResult]] = list(part_results)
    applied_ix: list[int] = []
    for ix, (name, lr) in enumerate(part_results):
        if lr.final_stl is None or lr.final_step is None or not lr.final_step.exists():
            telemetry.append(
                {"part": name, "ops": [], "applied": False, "note": "skipped: no STL/STEP"}
            )
            continue
        try:
            m = trimesh.load(str(lr.final_stl), force="mesh")
            lo, hi = m.bounds
        except Exception:
            telemetry.append(
                {"part": name, "ops": [], "applied": False, "note": "skipped: mesh unmeasurable"}
            )
            continue
        part = plan.parts[ix]
        ctx = PartFinishContext(
            name=name,
            role=(part.prompt[:120] if getattr(part, "prompt", None) else name),
            bbox=((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])),
            mating_exclusions=_mating_exclusions(part, plan),
        )
        ops = plan_part_finish(ctx, client=client, model=model)
        print(
            f"finish: '{name}' -> {len(ops)} op(s)"
            + (f" {[o.op for o in ops]}" if ops else " (restraint: left bare)"),
            flush=True,
        )
        if not ops:
            telemetry.append(
                {
                    "part": name,
                    "ops": [],
                    "applied": False,
                    "note": "planner chose no finish (restraint)",
                }
            )
            continue
        fdir = out_dir / f"{part_ids[ix]}_finish"
        fdir.mkdir(parents=True, exist_ok=True)
        out_stl, out_step = fdir / "part.stl", fdir / "part.step"
        if finish_step_file(lr.final_step, ops, out_stl, out_step):
            # REVALIDATE the FINISHED geometry (2026-07-25 review, the production
            # precondition): the fit gate below sees the assembly, but the PER-PART record
            # (`lr.validation`) still describes the CLEAN body -- without this, a finish op
            # that broke a part would ride into the report under a stale per-part PASS.
            # Same gates, same brief; a finished part may not validate WORSE than its clean
            # self did. Any revalidation error keeps the clean part (fail-closed).
            ok, why, fin_v = _finished_part_not_worse(lr, out_stl, out_step)
            if not ok:
                telemetry.append(
                    {
                        "part": name,
                        "ops": [{"op": o.op, "params": o.params} for o in ops],
                        "applied": False,
                        "note": f"reverted: finished geometry revalidation ({why})",
                    }
                )
                continue
            finished[ix] = (
                name,
                replace(
                    lr,
                    final_stl=out_stl,
                    final_step=out_step,
                    # The FINISHED validation replaces the clean body's record (review-caught):
                    # downstream reports/mass/labels must describe the files actually shipped.
                    validation=fin_v if fin_v is not None else lr.validation,
                ),
            )
            applied_ix.append(len(telemetry))
            telemetry.append(
                {
                    "part": name,
                    "ops": [{"op": o.op, "params": o.params} for o in ops],
                    "applied": True,
                    "note": "finish applied; finished geometry revalidated not-worse",
                }
            )
        else:
            telemetry.append(
                {
                    "part": name,
                    "ops": [{"op": o.op, "params": o.params} for o in ops],
                    "applied": False,
                    "note": "finish op(s) failed to apply; clean part kept",
                }
            )
    if not applied_ix:
        return part_results, telemetry
    fin_meshes, fin_keys = _load_part_meshes_keyed(finished)
    finished_fit = _assembly_fit(fin_meshes, keys=fin_keys, cache=cache)
    if fit_not_worse(clean_fit, finished_fit):
        return finished, telemetry
    for tix in applied_ix:  # the ship gate reverted the whole finished set: say so per part
        telemetry[tix]["applied"] = False
        telemetry[tix]["note"] = "reverted: assembly fit would have worsened"
    print("finish: fit gate reverted the finished set (fit would worsen)", flush=True)
    return part_results, telemetry


def _part_coherent(lr: LoopResult) -> bool:
    if lr.validation is not None:
        return lr.validation.hard_ok
    # a checkpoint-resumed part has no in-memory validation; honor its recorded verdict
    return bool(getattr(lr, "resumed_hard_ok", False))


def _assemble_step(
    part_results: list[tuple[str, LoopResult]], out_dir: Path, object_name: str
) -> Path | None:
    """Build a LABELED STEP assembly compound from the per-part STEPs (each already authored in
    the shared origin frame), so the assembly is real CAD with part identity -- not just a fused
    mesh. Best-effort: returns None if no part STEP loads."""
    try:
        from build123d import Compound, export_step, import_step
    except Exception:
        return None
    children: list[Any] = []
    for name, lr in part_results:
        if lr.final_step is None or not lr.final_step.exists():
            continue
        try:
            solid = import_step(str(lr.final_step))
            solid.label = name
            children.append(solid)
        except Exception:
            continue
    if not children:
        return None
    try:
        asm = Compound(children=children)
        asm.label = _slug(object_name)
        step_path = out_dir / "assembled.step"
        export_step(asm, str(step_path))
        return step_path
    except Exception:
        return None


def _load_part_meshes_keyed(
    part_results: list[tuple[str, LoopResult]],
) -> tuple[list[Any], list[tuple[str, float]]]:
    """Load each built part's STL as a trimesh, GUARDED: a part whose STL is missing or will not
    load is dropped (not fatal) and a degenerate (non-Trimesh / empty) load is skipped, so the
    public functional path never raises on a bad mesh -- it degrades to a smaller / empty assembly
    that the disposition then reports honestly (no false-PASS, no crash). FAIL-OPEN per part.

    Also returns one (path, mtime) identity key per loaded mesh: the pair-fit cache key
    (audit P2) -- a repair round swaps a part's FILE, so unchanged pairs re-use their
    measured fit instead of re-running O(n^2) booleans."""
    import trimesh

    meshes: list[Any] = []
    keys: list[tuple[str, float]] = []
    for _name, lr in part_results:
        if lr.final_stl is None or not lr.final_stl.exists():
            continue
        try:
            mesh = trimesh.load(str(lr.final_stl), force="mesh")
        except Exception:
            continue
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0:
            meshes.append(mesh)
            try:
                keys.append((str(lr.final_stl), lr.final_stl.stat().st_mtime))
            except OSError:
                keys.append((str(lr.final_stl), -1.0))
    return meshes, keys


def _load_part_meshes(part_results: list[tuple[str, LoopResult]]) -> list[Any]:
    return _load_part_meshes_keyed(part_results)[0]


def _export_glb(stl_path: Path, glb_path: Path, material: Material | None = None) -> bool:
    """Write a GLB from the assembled STL, carrying the material's colour so a web viewer shows the
    assembly in its real appearance (a PBR material when possible, vertex colour as a robust
    fallback). Mirrors ``result.py::_export_glb`` so the functional artifact path emits the same
    web-ready GLB the single-part path does. Best-effort: returns False on any failure."""
    try:
        import trimesh

        mesh: Any = trimesh.load(str(stl_path), force="mesh")
        if material is not None:
            a = material.appearance
            rgba = [round(255 * c) for c in a.base_color] + [round(255 * a.opacity)]
            try:
                from trimesh.visual.material import PBRMaterial

                mesh.visual = trimesh.visual.TextureVisuals(  # type: ignore[no-untyped-call]
                    material=PBRMaterial(  # type: ignore[no-untyped-call]
                        baseColorFactor=rgba,
                        metallicFactor=float(a.metallic),
                        roughnessFactor=float(a.roughness),
                    )
                )
            except Exception:
                mesh.visual.vertex_colors = rgba
        to_gltf_yup(mesh)  # Z-up kernel mesh -> glTF Y-up (viewer only; the STL stays Z-up)
        mesh.export(str(glb_path))
        return True
    except Exception:
        return False


_PAIRFIT_VERT_CAP_DEFAULT = 8000  # H5B finish grind: 13.7s per 20k-vertex on_surface query
_PAIRFIT_TOUCH_MM = 1.0  # the touch/floating verdict line (a snug mate reads <= this)
_PAIRFIT_AABB_MARGIN_MM = 2.0  # boxes farther apart than this cannot mate (touch threshold is 1mm)


def _pairfit_vert_cap() -> int:
    try:
        return int(os.environ.get("FORGE_PAIRFIT_VERT_CAP", str(_PAIRFIT_VERT_CAP_DEFAULT)))
    except ValueError:
        return _PAIRFIT_VERT_CAP_DEFAULT


def _pair_fit(a: Any, b: Any) -> dict[str, Any] | None:
    """Cheap fit check between ONE pair of parts at the shared origin. Two signals:

    - the minimum SURFACE gap (0 = the parts touch / mate -- including a tenon snug in a socket
      VOID, where material overlap is ~0; a large gap = the parts never meet, i.e. floating).
    - the interference VOLUME ratio (a huge material overlap = one part is buried in the other).

    A defect the assembled render alone can hide. Returns None only if the pair cannot be
    measured at all (so the caller skips it rather than guess -- no false-FAIL)."""
    try:
        import numpy as np
        import trimesh

        # AABB EARLY-OUT (exact): boxes separated by more than the mate margin cannot
        # intersect and cannot touch, so the pair is floating without paying the boolean
        # or the proximity query (the H5B finish grind: all-pairs on_surface over every
        # vertex measured at hours for a 24-part roster). The per-axis separation is a
        # lower bound on the true surface gap; the note states it as such.
        try:
            axis_sep = np.maximum(a.bounds[0] - b.bounds[1], b.bounds[0] - a.bounds[1])
            box_gap = float(np.max(axis_sep))
        except Exception:
            box_gap = -1.0
        if box_gap > _PAIRFIT_AABB_MARGIN_MM:
            return {
                "interference_mm3": 0.0,
                "overlap_ratio": 0.0,
                "min_gap_mm": round(box_gap, 2),
                "seated": False,
                "touches": False,
                "note": f"the two parts are at least {box_gap:.1f}mm apart (bounding boxes "
                "never meet) -- they do not meet (floating); the inserting part should reach "
                "INTO the receiving socket.",
            }
        try:
            inter = float(trimesh.boolean.intersection([a, b]).volume)
        except Exception:
            inter = 0.0
        smaller = min(float(a.volume), float(b.volume))
        ratio = inter / smaller if smaller > 0 else 0.0
        # When the parts share NO material, distinguish a touching mate (a tenon snug in a socket
        # void) from a floating gap via the closest surface distance. The gap VALUE is consumed
        # exactly downstream (clearance_fit's bind margin, _worst_case_issues), so sampling may
        # never replace it -- above the cap it PRUNES instead (branch-and-bound, Codex-caught):
        # a stride sample yields an upper bound on the minimum; a vertex's distance to a's AABB
        # is a lower bound on its distance to a's surface (the surface lies inside the box); any
        # vertex whose lower bound exceeds the sampled upper bound cannot be the minimum. The
        # exact query then runs on the survivors only -- same minimum, proven, at a fraction of
        # the vertex count for the dense-mesh pairs that caused the H5B finish grind.
        try:
            query: Any = trimesh.proximity.ProximityQuery(a)  # type: ignore[no-untyped-call]
            verts = b.vertices
            cap = _pairfit_vert_cap()
            if 0 < cap < len(verts):
                idx = np.linspace(0, len(verts) - 1, cap).astype(int)
                _closest, dists, _tri = query.on_surface(verts[idx])
                ub = float(np.min(dists))
                lo, hi = a.bounds
                outside = np.maximum(np.maximum(lo - verts, verts - hi), 0.0)
                lower = np.sqrt(np.einsum("ij,ij->i", outside, outside))
                keep = lower <= ub + 1e-9
                # The sampled distances are already exact per-point measurements, so the
                # second pass covers only UNSAMPLED survivors and folds ub back in -- the
                # worst (unprunable, fully nested) pair costs the legacy query count, never
                # more (Codex-caught: sample + full re-query regressed nested pairs).
                keep[idx] = False
                rest = verts[keep]
                if len(rest):
                    _closest, dists, _tri = query.on_surface(rest)
                    gap = min(ub, float(np.min(dists)))
                else:
                    gap = ub
            else:
                _closest, dists, _tri = query.on_surface(verts)
                gap = float(np.min(dists))
        except Exception:
            gap = 0.0
        if ratio > 0.6:
            seated, touches, note = (
                False,
                True,
                f"the parts overlap by {inter:.0f} mm^3 ({ratio:.0%} of the smaller part) -- one is "
                "buried too deep into the other.",
            )
        elif inter > 1.0:
            seated, touches, note = (
                True,
                True,
                f"seated: {inter:.0f} mm^3 material overlap ({ratio:.0%} of the smaller part).",
            )
        elif gap <= _PAIRFIT_TOUCH_MM:
            seated, touches, note = (
                True,
                True,
                f"seated: the parts touch (closest surface {gap:.2f}mm) -- a snug mate, no overlap.",
            )
        else:
            seated, touches, note = (
                False,
                False,
                f"the two parts are {gap:.1f}mm apart at the closest point -- they do not meet "
                "(floating); the inserting part should reach INTO the receiving socket.",
            )
        return {
            "interference_mm3": round(inter, 1),
            "overlap_ratio": round(ratio, 3),
            "min_gap_mm": round(gap, 2),
            "seated": seated,
            "touches": touches,  # the two parts make contact (overlap OR snug) -- not floating
            "note": note,
        }
    except Exception:
        return None


def _assembly_fit(
    meshes: list[Any],
    keys: list[tuple[str, float]] | None = None,
    cache: dict[tuple[tuple[str, float], tuple[str, float]], dict[str, Any] | None] | None = None,
    hardware_meshes: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Fit check over ALL part pairs at the shared origin (not just the two largest).

    Two distinct defects the assembled render alone can hide, both caught here:
      * any INTERFERING pair (one part buried in another) -- not just the two biggest bodies.
      * a FLOATING part -- a body that touches NO other body in the assembly (e.g. a 3rd part
        hanging in space). With only the two-largest check, a small mis-placed extra part slipped
        through; the connectivity scan below catches it.

    The returned dict carries the WORST pair's metrics at the top level (back-compat: the
    disposition + report read ``seated`` / ``interference_mm3`` / ``min_gap_mm``), plus the
    full per-pair list and an explicit floating-part list. ``seated`` is True only when every
    part is connected AND no pair interferes. None when fewer than two parts (nothing to mate)."""
    if len(meshes) < 2:
        return None
    n = len(meshes)
    pairs: list[dict[str, Any]] = []
    touches_idx: set[int] = set()
    measured_any = False
    # Track contact so a part touching ANY other counts as seated in the graph (floating = no edge).
    # PAIR-FIT CACHE (audit P2): keyed by both parts' (path, mtime) file identity -- a repair
    # round changes only the swapped part's file, so its pairs miss and everything else hits.
    use_cache = cache is not None and keys is not None and len(keys) == n
    for i in range(n):
        for j in range(i + 1, n):
            ck = (keys[i], keys[j]) if use_cache and keys is not None else None
            if ck is not None and cache is not None and ck in cache:
                cached = cache[ck]
                pf = dict(cached) if cached is not None else None
            else:
                pf = _pair_fit(meshes[i], meshes[j])
                if ck is not None and cache is not None:
                    cache[ck] = dict(pf) if pf is not None else None
            if pf is None:
                continue  # unmeasurable pair: skip rather than fail (no false-FAIL)
            measured_any = True
            pf["pair"] = [i, j]
            pairs.append(pf)
            if pf["touches"]:
                touches_idx.add(i)
                touches_idx.add(j)
    if not measured_any:
        return None
    # L81 measured false-FAIL class: a part mounted ONLY to mapped hardware (the pan-tilt
    # cradle mates the servo flange, no printed neighbour) read 'floating'. Hardware
    # envelope meshes count toward TOUCHES only -- never toward interference verdicts:
    # the envelope is deliberately conservative, so overlap with it is the SWEEP's
    # business (where conservative errs safe), not seating's (where it would err false).
    hardware_mated: list[int] = []
    for i in range(n):
        if i in touches_idx:
            continue
        for hw in hardware_meshes or []:
            pf = _pair_fit(meshes[i], hw)
            if pf is not None and pf["touches"]:
                touches_idx.add(i)
                hardware_mated.append(i)
                break
    floating = [i for i in range(n) if i not in touches_idx]
    # ENCLOSURE (deep-review, the pump-in-a-tube class): a part surrounded by other
    # bodies with DESIGNED clearance touches nothing BY DESIGN (an impeller in a volute,
    # a gear inside a ring). Six axis rays from the candidate's centroid vs the union of
    # the others: >=4 hits = radially/axially surrounded -> seated-by-enclosure, never
    # 'floating' (the chief repair pressure to shove it into a wall dies with it).
    # Interference still fails; a genuinely adrift part (<=3 hits) still floats.
    # Fail-open per candidate.
    enclosed: list[int] = []
    if floating and n > 1:
        import numpy as np
        import trimesh

        for i in list(floating):
            try:
                others = trimesh.util.concatenate([meshes[j] for j in range(n) if j != i])
                lo_i, hi_i = (np.asarray(v, dtype=float) for v in meshes[i].bounds)
                lo_o, hi_o = (np.asarray(v, dtype=float) for v in others.bounds)
                # the candidate must FIT INSIDE the others' extent: an enclosing tube's
                # centroid lies inside its own content, so without this the CONTAINER
                # reads as 'enclosed by' the part it contains (caught by the two-sided
                # test on first run)
                if not (np.all(lo_i >= lo_o - 1e-6) and np.all(hi_i <= hi_o + 1e-6)):
                    continue
                c = (lo_i + hi_i) / 2.0
                dirs = np.array(
                    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                    dtype=float,
                )
                hits = others.ray.intersects_any(
                    ray_origins=np.tile(c, (6, 1)), ray_directions=dirs
                )
                if int(hits.sum()) >= 4:
                    enclosed.append(i)
                    floating.remove(i)
                    # enclosure is a MATE for both sides: the container(s) holding this
                    # part are connected through it, not adrift
                    for j in list(floating):
                        if j == i:
                            continue
                        lo_j, hi_j = (np.asarray(v, dtype=float) for v in meshes[j].bounds)
                        if np.all(lo_j <= lo_i + 1e-6) and np.all(hi_j >= hi_i - 1e-6):
                            floating.remove(j)
            except Exception:
                pass
    interfering = [p for p in pairs if not p["seated"] and p["touches"]]  # buried, not floating
    # The aggregate verdict reads the WORST pair: an interfering pair if any, else the smallest gap.
    worst = min(interfering, key=lambda p: p["overlap_ratio"], default=None) or min(
        pairs, key=lambda p: p["min_gap_mm"]
    )
    if interfering:
        worst = max(interfering, key=lambda p: p["overlap_ratio"])
    seated = not interfering and not floating
    if interfering:
        note = worst["note"]
    elif floating:
        note = (
            f"{len(floating)} part(s) float free -- they touch no other body in the assembly; "
            "every part must mate with at least one neighbour."
        )
    else:
        note = worst["note"]
    return {
        "interference_mm3": worst["interference_mm3"],
        "overlap_ratio": worst["overlap_ratio"],
        "min_gap_mm": worst["min_gap_mm"],
        "seated": seated,
        "floating_parts": floating,
        "enclosed_parts": enclosed,
        "hardware_mated_parts": hardware_mated,
        "pairs": pairs,
        "note": note,
    }


def _validate_motion(
    part_results: list[tuple[str, LoopResult]],
    plan: FunctionalPlan,
    hardware: list[Any] | None = None,
    workdir: Path | None = None,
) -> tuple[dict[str, Any] | None, Any, Any]:
    """When the plan declares joints, run the FULL relational + kinematic tier on the authored
    parts (coaxial + clearance + pin-fit + the 0-180 deg MOTION SWEEP) via the assembly adapter,
    and return ``(verdict, assembly_model, validation_report)`` -- the model + report ride along
    so the articulated ck_joint GLB export can reuse them (the EARNED per-joint sweep levels).

    The verdict is None ONLY when no joints were declared (a rigid object -- nothing to validate).
    When joints WERE declared but could not be assembled / validated (every joint dropped as
    malformed, or validate_assembly raised), it returns an ``articulates=False`` "unvalidated"
    verdict rather than None -- so a declared mechanism whose motion was never proven CANNOT ship
    clean (no-false-PASS). Never raises: a bad declaration degrades to that verdict, the static
    fit check still stands. The model/report are None whenever they could not be built.
    """
    if not plan.joints:
        return None, None, None
    unvalidated: dict[str, Any] = {
        "articulates": False,
        "fully_validated": False,
        "motion_status": "unvalidated",
        "motion_detail": "declared joints could not be assembled or validated",
        "joints_declared": len(plan.joints),
        "joints_validated": 0,
        # joints_validated counts joints EXAMINED, not joints that MOVE -- a fixed joint passes a
        # sweep trivially. Reported side by side so 13/13 can never read as 13 moving again (L89).
        "moving_joints": joint_mobility_counts(plan)[0],
        "rigid_mounts": joint_mobility_counts(plan)[1],
        # A sweep proves a joint does not COLLIDE; it never proves anything can MOVE it. H3 read
        # articulates=true with every DOF undriven. Lower bound, so it understates, never inflates.
        "hardware_declared": len(hardware or []),
        "undriven_dof_min": undriven_dof_lower_bound(plan, hardware),
        "checks": {},
    }
    try:
        from forge_mind.agentic.assembly_adapter import from_part_results

        ports_by_part = {p.name: list(p.ports) for p in plan.parts}
        drop_notes: list[str] = []
        # ADR-0124 deep-build: registry-MAPPED hardware (bought servos) enters the assembly
        # here -- the bridge existed and was tested, but this call site never passed it, so
        # mapped servos were invisible to the mount-pattern and envelope-sweep gates (the R3
        # design's one load-bearing gap). Best-effort per decl: a failed resolution degrades
        # with a note (never-raises law); hardware=None keeps every existing caller
        # byte-identical.
        placed = None
        if hardware:
            from forge_mind.structures.servo_bridge import placed_hardware_for

            placed = []
            for decl in hardware:
                try:
                    placed.append(placed_hardware_for(decl, workdir=workdir))
                except Exception as exc:
                    drop_notes.append(f"hardware {getattr(decl, 'hw_id', '?')} unresolved: {exc}")
        model = from_part_results(
            part_results, plan.joints, ports_by_part, notes=drop_notes, hardware=placed
        )
        if drop_notes:
            # WHY joints vanished (row 58: 12 declared, 0 resolved, zero trace) -- these
            # lines reach the chief via _assembly_failure_lines so the repair can act.
            unvalidated["joint_drop_notes"] = drop_notes
            unvalidated["motion_detail"] = (
                f"{len(drop_notes)} of {len(plan.joints)} declared joints could not be "
                f"resolved: {drop_notes[0]}"
            )
        # A slug collision (two part names -> the same body id) makes the assembly ambiguous:
        # model.body(id) would silently pick one body. Refuse to certify it (no-false-PASS).
        body_ids = [b.id for b in model.bodies]
        if len(model.bodies) < 2 or not model.joints or len(set(body_ids)) != len(body_ids):
            return unvalidated, None, None
    except Exception:
        return unvalidated, None, None
    try:
        from forge_mind.validator.assembly_checks import validate_assembly
        from forge_mind.validator.report import CheckStatus

        report = validate_assembly(model)
    except Exception:
        # The assembly EXISTS but its motion could not be validated: the verdict stays
        # honestly unvalidated, while the model still rides out so the articulated GLB can
        # ship with checked="none" (the viewer shows the not-checked banner verbatim).
        return unvalidated, model, None

    motion = report.check("assembly.range_of_motion")
    relational = {}
    for cid in (
        "assembly.range_of_motion",
        "assembly.pin_coaxial",
        "assembly.pin_clearance",
        "assembly.gear_center_distance",
        "assembly.gear_mesh",
        "assembly.gear_mesh_declared",
    ):
        c = report.check(cid)
        if c is not None:
            relational[cid.split(".", 1)[1]] = {
                "status": c.status.value,
                "message": c.message,
                "actual": c.actual,
            }
    # FULL coverage gate: a mechanism only "articulates" if its sweep PASSED *and* every
    # declared joint was actually validated. A partial joint-drop (one declared joint
    # filtered out as malformed while another swept clean) must NOT ship clean -- the dropped
    # joint's motion was never proven (no-false-PASS).
    swept_ok = motion is not None and motion.status.value == "pass"
    fully_validated = len(model.joints) == len(plan.joints)
    verdict: dict[str, Any] = {
        "articulates": swept_ok and fully_validated,
        "fully_validated": fully_validated,
        "motion_status": motion.status.value if motion is not None else "absent",
        "motion_detail": (motion.actual or motion.message) if motion is not None else None,
        "joints_declared": len(plan.joints),
        "joints_validated": len(model.joints),
        # joints_validated counts joints EXAMINED, not joints that MOVE -- a fixed joint passes a
        # sweep trivially. Reported side by side so 13/13 can never read as 13 moving again (L89).
        "moving_joints": joint_mobility_counts(plan)[0],
        "rigid_mounts": joint_mobility_counts(plan)[1],
        # A sweep proves a joint does not COLLIDE; it never proves anything can MOVE it. H3 read
        # articulates=true with every DOF undriven. Lower bound, so it understates, never inflates.
        "hardware_declared": len(hardware or []),
        "undriven_dof_min": undriven_dof_lower_bound(plan, hardware),
        "checks": relational,
    }
    if drop_notes:
        # WHY joints vanished, on the VALIDATED path too (the north-star smoke reported
        # 9/13 with zero trace -- the notes were attached only to the unvalidated verdict).
        verdict["joint_drop_notes"] = drop_notes
    # THE WHOLE report's FAILs ride the verdict (deep-review: the ship decision was a
    # hand-maintained whitelist, so every new assembly check was born ungated -- the
    # gear bug was one instance; pin_coaxial/print_in_place_gap/static_clearance were
    # computed-but-gating-nothing). New checks now gate + reach the repair round by
    # DEFAULT (fail-closed-everywhere, the constitution's rule).
    # ``evidence`` rides each entry (ADR-0127) so a consumer can tell a MEASURED violation from
    # a fail-closed one. Every gate here is unchanged by it -- they read the LIST, not the tier --
    # but the strict non-shipping tier must fail OPEN on absence of proof, and `status == FAIL`
    # alone cannot distinguish the two. Absent on checks that have not classified themselves.
    verdict["assembly_fails"] = [
        {"id": c.id, "message": c.message, "evidence": c.evidence}
        for c in report.checks
        if c.status is CheckStatus.FAIL
    ]
    return verdict, model, report


def _motion_repair_target(
    report: Any, plan: FunctionalPlan, part_results: list[tuple[str, LoopResult]]
) -> tuple[str, PartPlan, str] | None:
    """The MOVING part to re-author after a failed sweep, with the measured collision text.

    Resolved from the per-joint ``assembly.sweep.<a>__<b>`` FAIL check (the b part is the
    mover by the placement convention); body ids map back to part names the same way the
    output dirs were derived. None when no per-joint sweep failed (nothing measurable to
    repair against) or the mover cannot be resolved -- the honest FAIL then stands."""
    if report is None:
        return None
    fail = next(
        (
            c
            for c in report.checks
            if c.id.startswith("assembly.sweep.") and c.status.value == "fail"
        ),
        None,
    )
    if fail is None:
        return None
    _a_id, _, b_id = fail.id[len("assembly.sweep.") :].partition("__")
    names = [name for name, _lr in part_results]
    by_id = dict(zip(_disambiguate_slugs(names), names, strict=True))
    mover = by_id.get(b_id)
    if mover is None:
        return None
    plan_part = next((p for p in plan.parts if p.name == mover), None)
    if plan_part is None:
        return None
    detail = " ".join(s for s in (fail.message, fail.actual) if s).strip()
    return mover, plan_part, detail


_SHAVE_MAX_DEPTH_MM = 0.6  # deeper collisions are design defects: route to the LLM repair
_SHAVE_CAP_MM = 0.25  # never thin a part by more than one nozzle-ish shell per round


def _parse_sweep_collision(detail: str) -> tuple[str, str, float] | None:
    """(mover_id, static_id, depth_mm) from the sweep's frozen failure grammar
    'X hits Y at V unit (depth D mm)'. None when unparseable / depth unmeasured."""
    import re

    m = re.search(r"(\S+) hits (\S+) at [-\d.]+ \w+ \(depth ([\d.]+) mm\)", detail)
    return (m.group(1), m.group(2), float(m.group(3))) if m else None


def _try_deterministic_shave(
    motion: dict[str, Any],
    part_results: list[tuple[str, LoopResult]],
    out_dir: Path,
) -> tuple[list[tuple[str, LoopResult]], str] | None:
    """Zero-LLM motion repair (audit P1): a SHALLOW sweep collision (<= 0.6 mm -- 'the tab
    is a hair too fat') is fixed by a uniform inward offset of the mover, capped at 0.25 mm,
    instead of a full LLM re-author. The measured depth (now in the sweep detail) sizes the
    shave. Fail-closed: unparseable detail, deep collision, missing STEP, an offset the
    kernel refuses, or a non-watertight result -> None, and the LLM repair path stands.
    The re-assembled fit + sweep still judge the outcome either way."""
    detail = str(motion.get("motion_detail") or "")
    parsed = _parse_sweep_collision(detail)
    if parsed is None:
        return None
    mover_id, _static_id, depth = parsed
    if depth > _SHAVE_MAX_DEPTH_MM:
        return None
    names = [n for n, _lr in part_results]
    by_id = dict(zip(_disambiguate_slugs(names), names, strict=True))
    mover_name = by_id.get(mover_id)
    if mover_name is None:
        return None
    lr = dict(part_results)[mover_name]
    if lr.final_step is None or not lr.final_step.exists():
        return None
    amount = min(_SHAVE_CAP_MM, depth / 2.0 + 0.05)
    try:
        import trimesh
        from build123d import export_step, export_stl, import_step, offset

        shave_dir = out_dir / f"{mover_id}_motion_shave"
        shave_dir.mkdir(parents=True, exist_ok=True)
        out_stl = shave_dir / lr.final_stl.name if lr.final_stl else shave_dir / "part.stl"
        out_step = shave_dir / lr.final_step.name
        solid = offset(import_step(str(lr.final_step)), amount=-amount)
        export_stl(solid, str(out_stl))
        export_step(solid, str(out_step))
        m = trimesh.load(str(out_stl), force="mesh")
        if not (isinstance(m, trimesh.Trimesh) and m.is_watertight and float(m.volume) > 0):
            return None
    except Exception:
        return None
    print(
        f"motion: '{mover_name}' shaved -{amount:.2f} mm deterministically "
        f"(measured overlap {depth:.2f} mm, 0 LLM calls)",
        flush=True,
    )
    shaved = replace(lr, final_stl=out_stl, final_step=out_step)
    note = f"deterministic shave: {mover_name} -{amount:.2f} mm (collision depth {depth:.2f} mm)"
    return [(n, shaved if n == mover_name else r) for n, r in part_results], note


def _motion_repair_prompt(plan_part: PartPlan, detail: str) -> str:
    """The re-author prompt: the part's ORIGINAL spec + the measured collision callout."""
    return (
        f"{plan_part.prompt}\n\n"
        "MOTION-REPAIR ROUND -- the assembled mechanism FAILED its collision sweep, and your "
        f"part is the MOVING body of the failing pair: {detail}\n"
        "Re-model this part to the SAME spec with the MATING geometry adjusted so the moving "
        "pair has at least 1.0 mm of air in EVERY direction except the intended pin-in-bore "
        "contact: shrink the insertion tab / clevis width, inset or chamfer the colliding "
        "faces. Keep every bore exactly at its specified position and diameter, and keep the "
        "part's role, envelope, and all other features unchanged."
    )


def _articulated_glb(model: Any, report: Any, glb_path: Path) -> dict[str, Any] | None:
    """Upgrade the assembly GLB to the frozen CK_JOINT contract: parented per-part nodes
    carrying ``extras.ck_joint`` with the EARNED ``checked`` level (the per-joint
    ``assembly.sweep.*`` checks in ``report``), so the web viewer's articulation rig can
    MOVE the mechanism. Writes atomically (tmp + replace) so a failed export never corrupts
    the flat GLB already on disk. Returns the manifest, or None on any failure (the flat
    GLB stays -- the viewer treats the assembly rigid, which is honest)."""
    if model is None or not getattr(model, "joints", None):
        return None
    tmp = glb_path.with_name(glb_path.name + ".ck.tmp")
    try:
        from forge_mind.viz.assembly_glb import export_assembly_glb

        manifest = export_assembly_glb(model, tmp, report=report)
        tmp.replace(glb_path)
        return manifest
    except Exception:
        import contextlib

        with contextlib.suppress(Exception):
            tmp.unlink()
        return None


def _parse_mm(actual: Any) -> float | None:
    """Pull the leading numeric mm value out of a check ``actual`` like ``'0.250 mm'``.
    None if it does not parse (so the caller skips rather than guess -- no false-FAIL)."""
    if not isinstance(actual, str):
        return None
    head = actual.strip().split()
    try:
        return float(head[0]) if head else None
    except ValueError:
        return None


def _annotate_material_pin_fit(motion: dict[str, Any], material: Material) -> None:
    """MATERIAL pin fit (F6): re-judge the assembly checker's MEASURED radial pin clearance against
    THIS material's slip-fit band (via the F6 ``running_fit_clearance_mm`` helper + ``clearance_fit``)
    instead of the global 0.4mm running-fit constant. Mutates ``motion`` in place, adding a
    ``material_pin_fit`` entry; a hard interference (a pin fatter than its bore -> 0/negative gap)
    classifies as not-ok so the disposition can act on it. PURE OVERLAY: it never relaxes the
    checker's own 0.1mm floor (that gate already ran), it only adds the material-sized verdict.
    No-op when there is no measurable pin clearance (a rigid / hardware-only assembly)."""
    pin = motion.get("checks", {}).get("pin_clearance")
    if not isinstance(pin, dict):
        return
    measured = _parse_mm(pin.get("actual"))
    if measured is None:
        return
    ok, note = clearance_fit(material, measured)
    motion["material_pin_fit"] = {
        "material": material.name,
        "measured_clearance_mm": round(measured, 3),
        "recommended_running_fit_mm": round(running_fit_clearance_mm(material), 3),
        "ok": ok,
        "note": note,
    }


def _reload_mech_arch(
    reuse_arch: str, out_dir: Path, hardware: list[Any] | None
) -> tuple[Any | None, str | None, list[Any] | None]:
    """Reload a saved mech_arch.json (FORGE_REUSE_ARCH) -- the resume half FORGE_REUSE_PLAN
    lacks. The L102 incident: plan reuse skips the arch stage, so a MECH_ARCH run resumed by
    plan alone carries NO HardwareDecls and the reloaded plan's `<servo>.horn` endpoints all
    read unknown-part -> no_build before a single checkpoint reloads. Reloading the typed doc
    re-derives the hardware exactly as the fresh stage does, 0 LLM calls. Fail-open like the
    fresh stage: any failure returns (None, None, hardware) and the run degrades archless.

    Returns (arch_obj, arch_block, hardware)."""
    try:
        from forge_mind.agentic.mech_arch import (
            MechArchitecture,
            arch_context_block,
            hardware_decls_from,
        )

        arch = MechArchitecture.model_validate_json(Path(reuse_arch).read_text(encoding="utf-8"))
        (out_dir / "mech_arch.json").write_text(arch.model_dump_json(indent=2), encoding="utf-8")
        if not hardware:
            hardware = hardware_decls_from(arch)
        print(
            f"mech-arch: RELOADED {len(arch.joints)} joint(s), {len(arch.actuators)} "
            f"actuator(s) from {reuse_arch}; hardware decls "
            f"{'derived' if hardware else 'none'} (0 LLM calls)",
            flush=True,
        )
        return arch, arch_context_block(arch), hardware
    except Exception as exc:
        print(f"mech-arch reload: skipped ({type(exc).__name__}: {exc})", flush=True)
        return None, None, hardware


def produce_functional_part(
    prompt: str,
    out_dir: Path,
    *,
    max_iterations: int = 4,
    budget_usd: float = 2.0,
    n_drafts: int = 2,
    material: Material | None = None,
    client: LLMClientLike | None = None,
    brief: DesignBrief | None = None,
    brief_model: str | None = None,
    author_model: str | None = None,
    critic_model: str | None = None,
    judge_model: str | None = None,
    hardware: list[Any] | None = None,
) -> FunctionalResult:
    """Model a multi-part object as separate bodies + assemble them. Never raises on a stage failure.

    ``hardware`` (ADR-0124): registry-mapped bought components (``HardwareDecl``s -- e.g. the
    MG996R servos of an R3 brief) carried into seating/mount-pattern/motion gates. None (the
    default) is byte-identical to the pre-wire-up behavior.

    ``brief`` (optional): the pre-researched whole-object brief (the judge's reference). When
    given, the whole-object research call is skipped (the caller already paid for it -- e.g.
    ``produce_part`` researches once to route on ``brief.target``); when None, it is researched here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    client = client or make_client()
    total_cost = 0.0

    # 1. research the WHOLE object (the judge's reference) unless the caller passed it, + decompose.
    # BOTH research and decompose are LLM calls that can fail (a truncated / max_tokens response
    # raises a parse error, an API hiccup raises). Neither may escape produce_part (never-raises):
    # degrade to an honest no_build carrying whatever brief exists for the report. (The engine
    # probe found research_brief outside this guard: a truncated brief killed the whole run.)
    whole_brief = brief
    try:
        if whole_brief is None:
            whole_brief, u = _call_with_transient_retry(
                research_brief,
                prompt,
                client=client,
                model=brief_model,
                extra_context=select_design_rules(prompt),
            )
            total_cost += u.cost_usd
        # MECHANICAL ARCHITECTURE stage (FORGE_MECH_ARCH=1, default OFF; the founder's
        # engineered-parts bar): one chief-architect call fixes joints/bearings/
        # retention/actuator placements/fastening BEFORE decompose; its placements
        # DERIVE the HardwareDecls (when the caller passed none), so cards and gates
        # measure against the same document. Fail-open: an arch failure degrades to
        # the archless pipeline; the deterministic plan gates stay the enforcement.
        arch_block: str | None = None
        arch_obj: MechArchitecture | None = None  # the typed doc the actuation gate reads
        reuse = os.environ.get("FORGE_REUSE_PLAN", "").strip()
        # An EXPLICIT saved arch suppresses the fresh stage entirely (review-caught: it was
        # silently ignored when set without FORGE_REUSE_PLAN -- a paid model call would
        # persist a new nondeterministic arch over the requested artifact).
        reuse_arch = os.environ.get("FORGE_REUSE_ARCH", "").strip()
        if os.environ.get("FORGE_MECH_ARCH") == "1" and not reuse and not reuse_arch:
            try:
                from forge_mind.agentic.mech_arch import (
                    arch_context_block,
                    hardware_decls_from,
                    plan_mech_architecture,
                )

                arch, u = _call_with_transient_retry(
                    plan_mech_architecture,
                    prompt,
                    client=client,
                    model=brief_model,
                    brief_text=getattr(whole_brief, "visual_description", None),
                    doctrine=_enabled_overlays(),
                )
                total_cost += u.cost_usd
                (out_dir / "mech_arch.json").write_text(
                    arch.model_dump_json(indent=2), encoding="utf-8"
                )
                arch_block = arch_context_block(arch)
                arch_obj = arch
                if not hardware:
                    hardware = hardware_decls_from(arch)
                print(
                    f"mech-arch: {len(arch.joints)} joint(s), {len(arch.actuators)} "
                    f"actuator(s) placed; hardware decls "
                    f"{'derived' if hardware else 'none'}",
                    flush=True,
                )
            except Exception as exc:
                print(f"mech-arch: skipped ({type(exc).__name__}: {exc})", flush=True)
        # ARCH REUSE (L102 incident): with FORGE_REUSE_PLAN set the fresh-arch stage above is
        # skipped, so a MECH_ARCH run's resume must reload the arch too or its plan gate
        # no_builds on every `<servo>.horn` endpoint (H5B reproduced 8 unknown-part errors).
        # FORGE_REUSE_ARCH=<path> is the explicit override; a MECH_ARCH run reusing a plan
        # defaults to the mech_arch.json SIBLING of the reused plan (review-caught: the
        # one-env recipe must work) -- a stale/mismatched sibling still dies honestly at the
        # plan gate's actuation cross-check.
        if not reuse_arch and reuse and os.environ.get("FORGE_MECH_ARCH") == "1":
            sibling = Path(reuse).parent / "mech_arch.json"
            if sibling.exists():
                reuse_arch = str(sibling)
        if arch_obj is None and reuse_arch:
            arch_obj, arch_block, hardware = _reload_mech_arch(reuse_arch, out_dir, hardware)
        # PLAN REUSE (row 57): every fresh replan re-rolls its own new clearance mistakes,
        # so a re-run of the same prompt should START from the plan that already survived.
        # FORGE_REUSE_PLAN=<path to a saved plan.json> loads it instead of re-decomposing;
        # unset -> decompose as always. The plan is ALWAYS saved to out_dir/plan.json.
        if reuse and Path(reuse).exists():
            plan = FunctionalPlan.model_validate_json(Path(reuse).read_text(encoding="utf-8"))
        else:
            plan, u = _call_with_transient_retry(
                decompose_functional,
                prompt,
                client=client,
                model=brief_model,
                brief=whole_brief,
                extra_callout=arch_block,
            )
            total_cost += u.cost_usd
        # PLAN-TIME TOPOLOGY GATE (deterministic, milliseconds): an incomplete joint graph
        # dooms the run ~90 min later (the i4-finish audit: 4/10 parts dangled in zero
        # joints behind the old 3-port cap and the sweep chased phantoms). One bounded
        # re-decompose with the measured issues; a still-broken plan is an HONEST no_build
        # NOW instead of an untrustworthy verdict after authoring everything (fail-closed).
        plan = _canonicalize_plan_endpoints(plan)  # unambiguous shorthand -> canonical names
        # Frames-only hardware port map (workdir=None -> no envelope compile): the topology
        # gate must know a mapped servo's real ports so `pan_servo.horn` is legal (ADR-0124).
        hw_ports: dict[str, set[str]] = {}
        for _decl in hardware or []:
            try:
                from forge_mind.structures.servo_bridge import placed_hardware_for as _phf

                hw_ports[_decl.hw_id] = {p.name for p in _phf(_decl).ports}
            except Exception:
                hw_ports[getattr(_decl, "hw_id", "?")] = set()
        eng_on = os.environ.get("FORGE_ENG_AUDIT") == "1"
        # Resolved BEFORE the plan gates (was post-gate): the dimension gate's fit floor
        # derives from the material's slip band (L100), so the gate must know it.
        mat = material or resolve_material(whole_brief.material)
        topo_issues = (
            _plan_topology_issues(plan, hw_ports)
            + _plan_dimension_issues(plan, mat)
            + (_plan_actuation_issues(plan, arch_obj) if arch_obj is not None else [])
            + (_plan_engineering_issues(plan, hardware, arch_obj) if eng_on else [])
        )
        if topo_issues and not reuse:
            # L101 postmortem gap: without this line nobody can tell afterwards whether
            # the bounded replan fired or the FIRST plan already carried the final issues.
            print(
                f"plan gate: {len(topo_issues)} issue(s) on the first plan -- one bounded "
                "replan:\n  - " + "\n  - ".join(i[:160] for i in topo_issues[:8]),
                flush=True,
            )
            topo_callout = (
                "PLAN TOPOLOGY ERROR -- your previous plan failed the deterministic plan "
                "gates (joint graph / dimensions / actuation binding / engineering "
                "decisions). Fix EVERY issue "
                "and re-emit the full plan:\n- " + "\n- ".join(topo_issues)
            )
            plan, u = decompose_functional(
                prompt,
                client=client,
                model=brief_model,
                brief=whole_brief,
                extra_callout=(f"{arch_block}\n\n{topo_callout}" if arch_block else topo_callout),
            )
            total_cost += u.cost_usd
            plan = _canonicalize_plan_endpoints(plan)
            topo_issues = (
                _plan_topology_issues(plan, hw_ports)
                + _plan_dimension_issues(plan, mat)
                + (_plan_actuation_issues(plan, arch_obj) if arch_obj is not None else [])
                + (_plan_engineering_issues(plan, hardware, arch_obj) if eng_on else [])
            )
        (out_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        if topo_issues:
            return FunctionalResult(
                out_dir=out_dir,
                object_name=plan.object_name,
                disposition="no_build",
                total_cost_usd=total_cost,
                whole_brief=whole_brief,
                failure="plan topology: " + "; ".join(topo_issues[:6]),
            )
    except Exception as exc:
        return FunctionalResult(
            out_dir=out_dir,
            object_name=whole_brief.object_name if whole_brief else prompt[:60],
            disposition="no_build",
            total_cost_usd=total_cost,
            whole_brief=whole_brief,
            failure=f"{'decompose' if whole_brief else 'research'}: {type(exc).__name__}: {exc}",
        )

    # EXPERT PANEL, plan stage (founder 2026-07-12): mech/materials/thermal/chemical/aero
    # engineers review what must be satisfied BEYOND geometry, BEFORE authoring. The
    # panel surfaces requirements; deterministic code enforces the numeric ones; blocker
    # findings cap the disposition at caveats (never silently shipped clean). Fail-open.
    from forge_mind.agentic.expert_review import run_expert_review

    plan_digest = (
        f"{plan.object_name}: "
        + "; ".join(f"{pp.name}" for pp in plan.parts)
        + f". Assembly: {plan.assembly_note}"
    )
    expert_plan = run_expert_review(
        stage="plan", prompt=prompt, summary=plan_digest, mat=mat, client=client, model=brief_model
    )
    if expert_plan is not None:
        blk = expert_plan["blockers"]
        print(
            f"expert-review(plan): {len(expert_plan['requirements'])} requirement(s), "
            f"{blk} blocker(s)"
            + (
                f" -- {(expert_plan['deterministic_findings'] or expert_plan['requirements'])[0].get('finding', '') or expert_plan['requirements'][0]['requirement']}"
                if blk
                else ""
            ),
            flush=True,
        )
    # EXPERT BLOCKERS GATE THE PLAN (FORGE_EXPERT_REPLAN=1, default OFF): H3's actuation
    # specialist named the exact integration gap ("pins and tenons, no servo integration")
    # and the run proceeded anyway -- review that only narrates is not review. One bounded
    # re-decompose against the blocker findings (the topology-retry pattern); the revised
    # plan must re-pass the deterministic plan gates or the ORIGINAL plan stands (a review
    # round may never replace a gate-clean plan with a gate-broken one). Fail-open on any
    # review/re-plan error, like the panel itself.
    if (
        os.environ.get("FORGE_EXPERT_REPLAN") == "1"
        and expert_plan is not None
        and expert_plan["blockers"] > 0
        and not os.environ.get("FORGE_REUSE_PLAN", "").strip()
    ):
        try:
            blocker_lines = [
                r["requirement"]
                for r in expert_plan["requirements"]
                if r.get("severity") == "blocker"
            ] + [
                str(f.get("finding", ""))
                for f in expert_plan["deterministic_findings"] or []
                if f.get("finding")
            ]
            if blocker_lines:
                replan_callout = (
                    "EXPERT DESIGN REVIEW -- BLOCKERS. The engineering panel reviewed your "
                    "plan and found blocker-severity gaps. Revise the plan so every card "
                    "RESOLVES each blocker in geometry (pockets, mounts, couplings, "
                    "clearances -- not prose), and re-emit the full plan:\n- "
                    + "\n- ".join(blocker_lines[:6])
                )
                replanned, u = decompose_functional(
                    prompt,
                    client=client,
                    model=brief_model,
                    brief=whole_brief,
                    extra_callout=(
                        f"{arch_block}\n\n{replan_callout}" if arch_block else replan_callout
                    ),
                )
                total_cost += u.cost_usd
                replanned = _canonicalize_plan_endpoints(replanned)
                regate = (
                    _plan_topology_issues(replanned, hw_ports)
                    + _plan_dimension_issues(replanned, mat)
                    + (_plan_actuation_issues(replanned, arch_obj) if arch_obj is not None else [])
                    + (
                        _plan_engineering_issues(replanned, hardware, arch_obj)
                        if os.environ.get("FORGE_ENG_AUDIT") == "1"
                        else []
                    )
                )
                if not regate:
                    plan = replanned
                    (out_dir / "plan.json").write_text(
                        plan.model_dump_json(indent=2), encoding="utf-8"
                    )
                    print(
                        f"expert-replan: plan revised against {len(blocker_lines)} blocker(s), "
                        "deterministic gates green",
                        flush=True,
                    )
                else:
                    print(
                        f"expert-replan: revision failed {len(regate)} plan gate(s) -- "
                        "keeping the original plan",
                        flush=True,
                    )
        except Exception as exc:  # fail-open: review rounds never kill a run
            print(f"expert-replan: skipped ({type(exc).__name__}: {exc})", flush=True)

    # 2. author each part through the proven loop (each is a single body -> no fusion-variance).
    #    Disambiguate the per-part dirs: two distinct names that slug-collide (e.g. "Finger" and
    #    "finger") must NOT share one dir, or the second part overwrites the first's STL/STEP. The
    #    assembly adapter re-derives ids the SAME way (_disambiguate_slugs over the parts in order).
    part_ids = _disambiguate_slugs([part.name for part in plan.parts])
    part_results: list[tuple[str, LoopResult]] = []
    doomed: str | None = None  # a failed STRUCTURAL part -> stop authoring (audit P2)

    def _joint_degree(name: str) -> int:
        return sum(
            1
            for j in plan.joints
            for ref in (j.a, j.b)
            if ref == name or ref.startswith(name + ".")
        )

    def _author_one(part: PartPlan, part_id: str, per_part_budget: float) -> LoopResult:
        # CHECKPOINT RESUME (audit P2): a prior run in this out_dir already finished this
        # part -> reload it (0 LLM calls). Session boundaries kill 90-min runs; a part-8
        # kill used to re-author all 10 parts on relaunch.
        resumed = _load_part_checkpoint(out_dir / part_id, part, mat)
        if resumed is not None:
            print(f"resume: '{part.name}' loaded from checkpoint (0 LLM calls)", flush=True)
            return resumed
        # T4 HYBRID AUTHORING (ADR: fable-t4 ruling §A): a declared structural LINK
        # (FORGE_PHYSICS_PARTS + FORGE_PHYSICS_CASE) is physics-authored via author_part instead of
        # the LLM -- lighter, load-verified geometry with a certificate, spliced in resumed-style.
        # Byte-inert unless both envs are set; a non-claimable gate -> None -> the LLM authors it.
        from forge_mind.agentic.hybrid_author import try_hybrid_physics_author

        hybrid = try_hybrid_physics_author(part, part_id, out_dir, mat, client)
        if hybrid is not None:
            return hybrid
        # TRANSIENT-FAILURE RETRY: an author_failed / brief_failed with ZERO iterations is an
        # intermittent LLM/CLI blip (rate limit, empty response), NOT a design failure -- and a
        # single failed part dooms a mechanism (everything joints to the crankcase; the radial
        # lost 9/16 parts this way). Retry such a part up to _AUTHOR_RETRIES times with a short
        # backoff so the limit recovers. A part that AUTHORED but failed validation (iters>0) is
        # a real result, never retried here (the iteration loop already owns that).
        card = _card_prompt(part)
        lr: LoopResult | None = None
        for attempt in range(_AUTHOR_RETRIES + 1):
            lr = run_agentic_pipeline(
                card,
                out_dir / part_id,
                max_iterations=_scaled_iterations(card, max_iterations),
                budget_usd=per_part_budget,
                n_drafts=n_drafts,
                material=mat,
                client=client,
                author_model=author_model,
                critic_model=critic_model,
                brief_model=brief_model,
                # SUBPART call: the multi-part finish stage owns polish (port exclusions +
                # assembly-fit baseline); the in-loop hook must not pre-finish components.
                apply_finish=False,
            )
            transient = lr.stopped_reason in ("author_failed", "brief_failed") and not lr.iterations
            if not transient or attempt == _AUTHOR_RETRIES:
                break
            time.sleep(_AUTHOR_BACKOFF_S)  # let an intermittent rate limit recover
        assert lr is not None
        return lr

    try:
        width = max(1, int(os.environ.get("FORGE_PARALLEL_AUTHORS", "1") or "1"))
    except ValueError:
        width = 1
    if width > 1 and len(plan.parts) > 1:
        # PARALLEL AUTHORING (audit P2, flag-gated, default OFF): parts are independent
        # until assembly, and the LLM subprocess dominates wall-clock (~4 min/part). The
        # in-process geometry stages (OCP validate, pyplot render) are serialized by locks
        # at their own entry points; the ``claude -p`` client is a stateless subprocess per
        # call. Budget slices are fixed at dispatch: an equal share of what remains, never
        # below the per-part floor (the sequential recompute has no parallel analog).
        from concurrent.futures import ThreadPoolExecutor

        per = max(_PER_PART_BUDGET_FLOOR, max(0.0, budget_usd - total_cost) / len(plan.parts))
        with ThreadPoolExecutor(max_workers=width) as pool:
            futures = [
                pool.submit(_author_one, part, pid, per)
                for part, pid in zip(plan.parts, part_ids, strict=True)
            ]
            authored = [f.result() for f in futures]
        for part, lr in zip(plan.parts, authored, strict=True):
            total_cost += lr.total_cost_usd
            part_results.append((part.name, lr))
            if doomed is None and lr.final_stl is None and _joint_degree(part.name) >= 2:
                doomed = part.name
    else:
        remaining_parts = len(plan.parts)
        for part, part_id in zip(plan.parts, part_ids, strict=True):
            # Share ONE generation-wide budget across the parts: each gets a fair slice of
            # whatever is LEFT (research + decompose + earlier parts already spent) -- BUT
            # never below a per-part FLOOR, or a 14-part object starves its late parts to $0
            # and they never even draft (the terse-probe rods: 6 empty dirs). The floor
            # guarantees every part one real attempt; on the subscription path the cost is
            # notional so the floor spends no real money.
            per_part_budget = max(_PER_PART_BUDGET_FLOOR, budget_usd - total_cost) / max(
                1, remaining_parts
            )
            per_part_budget = max(per_part_budget, _PER_PART_BUDGET_FLOOR)
            remaining_parts -= 1
            lr = _author_one(part, part_id, per_part_budget)
            total_cost += lr.total_cost_usd
            part_results.append((part.name, lr))
            # EARLY EXIT (audit P2): a STRUCTURAL part (>= 2 joints reference it) that
            # failed to build dooms the whole mechanism -- authoring the rest, assembling,
            # sweeping, and judging a doomed set burns ~half a run before an inevitable
            # no_build. Stop now; the honest disposition + failure note say why.
            if lr.final_stl is None and _joint_degree(part.name) >= 2:
                doomed = part.name
                break
    if doomed is not None:
        print(
            f"early-exit: structural part '{doomed}' failed to build; "
            f"skipping the remaining work (no_build)",
            flush=True,
        )

    # 2b. PORT-ANCHOR gate (deterministic, the row-55 lesson): a ported part whose built mesh
    #     is nowhere near its declared port origin was authored in a LOCAL frame (5 of 15 i4
    #     parts dropped the shared frame). ONE bounded re-author per misplaced part with the
    #     measured callout -- catching it here saves the chief round for real design defects.
    #     A part still misplaced after its retry stays (assembly checks own the honest FAIL).
    for ix, (name, lr) in enumerate(part_results):
        part = plan.parts[ix]
        callout = _port_anchor_callout(part, lr)
        if callout is None:
            continue
        # FIRST the zero-LLM seat (audit P1): the dominant failure is a pure translation
        # (part authored at local origin); the candidate-translation search is self-gating
        # and costs milliseconds. 4/10 parts on the live i4 each paid a ~4 min re-author
        # for what this fixes for free. The LLM re-author remains the fallback.
        seated_lr = _deterministic_seat(part, lr, part_ids[ix], out_dir)
        if seated_lr is not None and _port_anchor_callout(part, seated_lr) is None:
            part_results[ix] = (name, seated_lr)
            continue
        lr2 = run_agentic_pipeline(
            part.prompt + "\n\n" + callout,
            out_dir / f"{part_ids[ix]}_anchor_fix",
            max_iterations=1,
            budget_usd=max(0.0, budget_usd - total_cost),
            n_drafts=1,
            material=mat,
            client=client,
            author_model=author_model,
            critic_model=critic_model,
            brief_model=brief_model,
            brief=lr.brief,  # the part's own brief: an anchor fix re-authors the SAME part,
            # so re-researching it was a pure duplicate LLM call (audit P1)
            apply_finish=False,  # subpart: the multi-part finish stage owns polish
        )
        total_cost += lr2.total_cost_usd
        if lr2.final_stl is not None and _port_anchor_callout(part, lr2) is None:
            part_results[ix] = (name, lr2)

    # 2b'. CHECKPOINT the anchored parts (audit P2): a relaunch into this out_dir resumes
    #      each finished part for 0 LLM calls instead of re-authoring all of them.
    for ix, (_name, ck_lr) in enumerate(part_results):
        _save_part_checkpoint(out_dir / part_ids[ix], plan.parts[ix], ck_lr, mat)

    # 2c. DESIGNED-FINISH pass (FORGE_FINISH, dark by default): give each part the detail that
    #     makes it look designed (grip, chamfers, fins, labels) WITHOUT touching a mating
    #     surface, then ship the finished parts only if the assembly fit is UNCHANGED-or-better
    #     (else revert to the clean parts). Fail-closed end to end -- polish never ruins a
    #     working part.
    # Wave D / constitution: grasp tabs on EVERY part (cell profile only).
    part_results = _apply_grasp_tabs(part_results, out_dir, part_ids, mat, whole_brief)

    finish_telemetry: list[dict[str, Any]] | None = None
    if _finish_enabled() and doomed is None:
        part_results, finish_telemetry = _apply_designed_finish(
            part_results, plan, out_dir, part_ids, client=client, model=brief_model
        )

    # 3. assemble -> fit -> motion, with ONE bounded MOTION-REPAIR round (the assembly-level
    #    analog of the measured-feedback levers): when the sweep FAILS naming a colliding pair,
    #    the MOVING part is re-authored with that measured callout appended to its ORIGINAL
    #    prompt, the assembly rebuilds, and the sweep re-runs. One round only; a still-failing
    #    sweep stays an honest FAIL -- the repair changes the PART, never any check.
    #    GUARDED as before: a per-part STL that will not load, or a concatenate/export that
    #    raises, degrades to an honest empty-assembly failure; a single bad part is dropped.
    meshes: list[Any] = []
    n_built = 0
    assembled_stl: Path | None = None
    assembled_glb: Path | None = None
    assembled_render: Path | None = None
    fit: dict[str, Any] | None = None
    motion: dict[str, Any] | None = None
    asm_model: Any = None
    asm_report: Any = None
    repair_note: str | None = None
    # a doomed run (structural part failed) skips assembly/motion/judge entirely: the
    # disposition is an inevitable no_build and the remaining stages only burn minutes.
    fit_cache: dict[tuple[tuple[str, float], tuple[str, float]], dict[str, Any] | None] = {}
    mesh_keys: list[tuple[str, float]] = []
    for motion_attempt in range(2) if doomed is None else range(0):
        meshes, mesh_keys = _load_part_meshes_keyed(part_results)
        n_built = len(meshes)
        assembled_stl = None
        assembled_glb = None
        assembled_render = None
        # Resolve mapped-hardware envelopes FIRST (world-placed): they join the seating
        # scan as touch-only mates AND the judged render (the L84 hardware-blind-judge
        # class: two plates hovering around an invisible servo read as broken). Best-
        # effort; never blocks any stage.
        hw_fit_meshes: list[Any] = []
        for _decl in hardware or []:
            try:
                from forge_mind.structures.servo_bridge import placed_hardware_for as _phf2

                _ph = _phf2(_decl, workdir=out_dir)
                if _ph.mesh is not None:
                    _m = _ph.mesh.copy()
                    _m.apply_transform(_ph.transform)
                    hw_fit_meshes.append(_m)
            except Exception:
                pass
        if meshes:
            import trimesh

            try:
                assembled: Any = trimesh.util.concatenate(meshes)
                assembled_stl = out_dir / "assembled.stl"
                assembled.export(str(assembled_stl))
            except Exception:
                assembled_stl = None
            if assembled_stl is not None and assembled_stl.exists():
                # GLB EXPORT: a web-ready coloured GLB (mirrors result.py::_export_glb).
                glb_path = out_dir / "assembled.glb"
                if _export_glb(assembled_stl, glb_path, material=mat):
                    assembled_glb = glb_path
                assembled_render = out_dir / "assembled.png"
                # The JUDGED image shows the WHOLE mechanism: printed parts + hardware
                # envelopes. assembled.stl stays printed-parts-only (the deliverable);
                # the with-hardware STL is a render/viewer artifact.
                render_src = assembled_stl
                if hw_fit_meshes:
                    try:
                        with_hw = trimesh.util.concatenate([assembled, *hw_fit_meshes])
                        with_hw_stl = out_dir / "assembled_with_hardware.stl"
                        with_hw.export(str(with_hw_stl))
                        render_src = with_hw_stl
                    except Exception:
                        render_src = assembled_stl
                try:
                    render_part_to_png(
                        render_src,
                        assembled_render,
                        title=f"{plan.object_name} (assembled, {n_built} parts"
                        + (f" + {len(hw_fit_meshes)} hardware)" if hw_fit_meshes else ")"),
                        caption=prompt,
                    )
                except Exception:
                    assembled_render = None

        # 3b. cheap interference fit check over all pairs (cached: a repair round only
        # re-measures pairs whose part FILE changed -- 45 booleans -> the changed few).
        # Mapped-hardware envelopes (resolved above) join as touch-only mates (the L81
        # cradle false-FAIL class).
        fit = _assembly_fit(meshes, keys=mesh_keys, cache=fit_cache, hardware_meshes=hw_fit_meshes)
        # Material-aware clearance verdict on the WORST non-interfering touching pair: is the
        # measured gap a good slip fit for THIS material? (A hard interference -- a buried pair --
        # has no slip gap to classify; ``seated`` already FAILs it. Only a real touching mate.)
        if fit is not None:
            slip_pairs = [
                p
                for p in fit.get("pairs", [])
                if p.get("touches") and float(p["interference_mm3"]) <= 1.0
            ]
            worst_slip = min(slip_pairs, key=lambda p: float(p["min_gap_mm"]), default=None)
            if worst_slip is not None:
                ok, cnote = clearance_fit(mat, float(worst_slip["min_gap_mm"]))
                fit["clearance"] = {
                    "material": mat.name,
                    "recommended_mm": list(mat.clearance_mm),
                    "ok": ok,
                    "note": cnote,
                }
        # 3c. mechanism validation: when the plan declares joints, reconnect the kinematic spine --
        # run the 0-180 deg motion sweep (+ coaxial/clearance) on the authored parts so a moving
        # assembly is proven to articulate without binding, not just to render.
        # Pass the hardware kwargs ONLY when hardware is declared: the no-hardware call
        # shape stays byte-identical (pre-ADR-0124 monkeypatched stand-ins keep working).
        _hw_kwargs: dict[str, Any] = {"hardware": hardware, "workdir": out_dir} if hardware else {}
        motion, asm_model, asm_report = _validate_motion(part_results, plan, **_hw_kwargs)
        # 3c-bis. ARTICULATED GLB (frozen ck_joint contract): a mechanism that assembled exports
        # parented per-part nodes + extras.ck_joint (checked EARNED from the per-joint sweep
        # checks), replacing the flat concatenated GLB -- the web viewer's articulation rig can
        # now MOVE the agentic assembly. Best-effort: on failure the flat GLB stays (rigid, honest).
        if asm_model is not None:
            glb_target = assembled_glb or (out_dir / "assembled.glb")
            if _articulated_glb(asm_model, asm_report, glb_target) is not None:
                assembled_glb = glb_target
        # 3d. MATERIAL pin fit (F6): the assembly checker proves a pin clears its bore against a
        # global 0.1mm floor; re-judge that SAME measured radial clearance against THIS material's
        # slip-fit band. Pure overlay -- never relaxes the checker's floor.
        if motion is not None:
            _annotate_material_pin_fit(motion, mat)

        # --- the bounded repair decision -------------------------------------
        if motion_attempt == 1:
            break
        # DETERMINISTIC SHAVE first (audit P1): a shallow measured collision is relieved
        # by a capped inward offset of the mover -- zero LLM calls, seconds -- and the
        # re-assembled fit + sweep re-judge it. Deep/unmeasured failures fall through to
        # the chief/legacy repair exactly as before.
        if motion is not None and motion.get("motion_status") == "fail":
            shaved = _try_deterministic_shave(motion, part_results, out_dir)
            if shaved is not None:
                part_results, repair_note = shaved
                continue
        if _chief_enabled():
            # CHIEF mode: route ALL measured assembly failures (interference / floating
            # pairs AND the sweep verdict -- strictly more than the legacy path, which
            # only sees a sweep collision) back to the brain that authored the plan. The
            # chief revises ITS OWN build cards (possibly several parts -- the defect may
            # live in a different part than the symptom); the named parts are re-drafted
            # and the assembly re-measured. Bounded once; the validator stays the judge.
            seated_ok = fit is None or bool(fit.get("seated", True))
            motion_ok = motion is None or bool(motion.get("articulates", True))
            # ANY failing assembly check is a repairable failure (the sweep alone cannot
            # see a too-far gear pair; an off-axis pin can sweep clean) -- the chief gets
            # its bounded repair shot with the failure named.
            asm_ok = motion is None or not motion.get("assembly_fails")
            if seated_ok and motion_ok and asm_ok:
                break
            part_names = [n for n, _lr in part_results]
            failures = _assembly_failure_lines(fit, motion, part_names)
            if not failures:
                repair_note = "repair skipped: measured failure produced no failure lines"
                break
            repair_notes: list[str] = []
            revision = _chief_revise_plan(
                plan,
                failures,
                client=client,
                model=brief_model,
                render_png=assembled_render,
                extent_lines=_part_extent_lines(part_results),
                notes=repair_notes,
            )
            if revision is None:
                repair_note = (
                    repair_notes[0] if repair_notes else "chief revision returned None (unlogged)"
                )
                break
            by_name = {p.name: p for p in plan.parts}
            # Names are matched SLUG-INSENSITIVELY: the chief answers in prose-space
            # ("rod 1") while cards may be slug-space ("rod_1") -- an exact-only match
            # silently discards a valid revision (the row-56 no-fire suspect class).
            slug_to_name = {_slug(p.name): p.name for p in plan.parts}
            accepted = []
            refused: list[str] = []
            for rp in revision.revised_parts:
                canonical = rp.name if rp.name in by_name else slug_to_name.get(_slug(rp.name))
                if canonical is None:
                    repair_notes.append(f"chief revised unknown part '{rp.name}'")
                    continue
                # port contract (north-star smoke): a card that lost a joint-referenced
                # port would silently unresolve those joints at re-validation -- refuse it.
                refusal = _revision_port_guard(plan, rp, canonical)
                if refusal is not None:
                    refused.append(canonical)
                    repair_notes.append(refusal)
                    continue
                accepted.append(rp.model_copy(update={"name": canonical}))
            if not accepted:
                repair_note = "chief revision discarded: " + (
                    f"port contract refused {refused}"
                    if refused
                    else "no revised name matched the plan "
                    f"(revised={[rp.name for rp in revision.revised_parts]}, "
                    f"plan={part_names})"
                )
                break
            repair_note = f"chief revised: {[rp.name for rp in accepted]}" + (
                f"; refused (port contract): {refused}" if refused else ""
            )
            for rp in accepted:
                by_name[rp.name] = rp
            plan = plan.model_copy(update={"parts": [by_name[p.name] for p in plan.parts]})
            ids = dict(zip(part_names, _disambiguate_slugs(part_names), strict=True))
            results_by_name = dict(part_results)
            remaining = len(accepted)
            for rp in accepted:
                slice_budget = max(0.0, budget_usd - total_cost) / max(1, remaining)
                remaining -= 1
                fix_lr = run_agentic_pipeline(
                    rp.prompt,
                    out_dir / f"{ids[rp.name]}_chief_fix",
                    max_iterations=max_iterations,
                    budget_usd=slice_budget,
                    n_drafts=n_drafts,
                    material=mat,
                    client=client,
                    author_model=author_model,
                    critic_model=critic_model,
                    brief_model=brief_model,
                    # the same physical part, revised dimensions: its brief still stands,
                    # so the fix round skips a duplicate research call (audit P1)
                    brief=results_by_name[rp.name].brief,
                    apply_finish=False,  # subpart: the multi-part finish stage owns polish
                )
                total_cost += fix_lr.total_cost_usd
                if (
                    fix_lr.final_stl is not None
                    and fix_lr.validation is not None
                    and fix_lr.validation.hard_ok
                ):
                    # the revision path gets the SAME deterministic anchor treatment as
                    # first-pass parts (north-star iteration 4: the chief's replacement
                    # pins were authored at the ORIGIN and skipped the seat -- three
                    # coincident pins buried in the sun; the 'repair' made the assembly
                    # WORSE). Zero-LLM; a still-misplaced part stays and the assembly
                    # checks own the honest FAIL, exactly like the first pass.
                    if _port_anchor_callout(rp, fix_lr) is not None:
                        seated_fix = _deterministic_seat(
                            rp, fix_lr, f"{ids[rp.name]}_chief_fix", out_dir
                        )
                        if seated_fix is not None and _port_anchor_callout(rp, seated_fix) is None:
                            fix_lr = seated_fix
                    results_by_name[rp.name] = fix_lr
                # a failed re-draft keeps the ORIGINAL part (never swap in a broken body)
            part_results = [(n, results_by_name[n]) for n, _lr in part_results]
            continue

        # LEGACY (flag off, byte-identical): single-mover motion repair on a sweep FAIL.
        if motion is None or bool(motion.get("articulates")):
            break
        if motion.get("motion_status") != "fail":
            break  # unvalidated/absent: no measured collision to repair against
        target = _motion_repair_target(asm_report, plan, part_results)
        if target is None:
            break
        mover_name, mover_plan, detail = target
        mover_id = dict(
            zip(
                [n for n, _lr in part_results],
                _disambiguate_slugs([n for n, _lr in part_results]),
                strict=True,
            )
        )[mover_name]
        repair_lr = run_agentic_pipeline(
            _motion_repair_prompt(mover_plan, detail),
            out_dir / f"{mover_id}_motion_repair",
            max_iterations=max_iterations,
            budget_usd=max(0.0, budget_usd - total_cost),
            n_drafts=n_drafts,
            material=mat,
            client=client,
            author_model=author_model,
            critic_model=critic_model,
            brief_model=brief_model,
            brief=dict(part_results)[mover_name].brief,  # same part, adjusted clearance:
            # reuse its brief instead of re-researching (audit P1)
            apply_finish=False,  # subpart: the multi-part finish stage owns polish
        )
        total_cost += repair_lr.total_cost_usd
        if (
            repair_lr.final_stl is None
            or repair_lr.validation is None
            or not repair_lr.validation.hard_ok
        ):
            break  # repair did not produce a coherent body: keep the original honest FAIL
        part_results = [(n, repair_lr if n == mover_name else lr) for n, lr in part_results]

    # 3d-fin. RE-CHECKPOINT the FINAL part set (validate-vs-ship fix): the 2b' checkpoint was
    # written BEFORE the designed-finish pass, grasp tabs, and the motion-repair loop above.
    # Whenever any of those changed a part's geometry, the persisted checkpoint (what a resume
    # reloads and what m2_score / the deliverable read) diverged from what THIS report actually
    # validated -- a validate-vs-ship inconsistency (measured on the M2 arm: the report certified
    # a shoulder-clear the shipped checkpoint geometry failed). Persist what was validated so the
    # certificate matches the deliverable. Best-effort, same as 2b'.
    for ix, (_name, ck_lr) in enumerate(part_results):
        _save_part_checkpoint(out_dir / part_ids[ix], plan.parts[ix], ck_lr, mat)

    # 3d'. Wave A: assembled-envelope budget + advisory mobility (row 54 #7 / research #1).
    envelope_issue = _assembled_envelope_issue(meshes, prompt, whole_brief)
    worst_case = _worst_case_issues(fit, motion, mat, [n for n, _lr in part_results])
    # Metal DFM tier (founder 2026-07-12): process-keyed manufacturability. METAL -> the
    # machined-metal checks; FDM/RESIN return None here (printability owns those per-part).
    dfm_report = None
    if doomed is None and meshes:
        from forge_mind.validator.manufacturability import dfm_for_material

        dfm_report = dfm_for_material(meshes, mat)
        if dfm_report is not None and dfm_report.findings:
            print(
                f"metal-dfm: {dfm_report.verdict} -- {dfm_report.findings[0].detail}",
                flush=True,
            )
    # Part-FIDELITY tier (founder 2026-07-12, the Merlin valves): is each shipped part a
    # genuine solid, and more than external massing of its named role? Robust
    # watertightness (open boundary = the one blocker; weld artifacts never flag) +
    # name-keyed massing caveat. Report-tier otherwise; mirrors the DFM fold below.
    fidelity_rep = None
    if doomed is None and meshes:
        from forge_mind.validator.part_fidelity import fidelity_report

        stl_names = {
            str(lr.final_stl): name for name, lr in part_results if lr.final_stl is not None
        }
        fidelity_rep = fidelity_report(
            [
                (stl_names.get(key[0], f"part_{i}"), m)
                for i, (m, key) in enumerate(zip(meshes, mesh_keys, strict=True))
            ]
        )
        if fidelity_rep.findings:
            print(
                f"fidelity: {fidelity_rep.verdict} -- "
                f"{fidelity_rep.findings[0].part}: {fidelity_rep.findings[0].detail}",
                flush=True,
            )
    # Wave C: assemblability (CAVEAT-tier -- absence of a straight-line path is a
    # warning to verify by hand, not proof of impossibility). <= 12 parts, fail-open.
    print_pack = _print_pack(part_results) if doomed is None else None
    assembly_order: dict[str, Any] | None = None
    if doomed is None and 3 <= len(meshes) <= 12:
        from forge_mind.validator.insertion import assembly_order_issues

        assembly_order = assembly_order_issues(
            list(zip([n for n, _lr in part_results], meshes, strict=False))
        )
    mobility = _mobility_note(plan) if plan.joints else None
    if motion is not None and mobility:
        motion["mobility"] = mobility

    # 3e. a real labeled STEP compound (part identity), from the FINAL part set.
    assembled_step = _assemble_step(part_results, out_dir, plan.object_name) if meshes else None

    # Aggregate material spec: total material across all parts (each printed/made separately).
    total_volume = sum(float(m.volume) for m in meshes)
    mat_spec = material_spec(mat, total_volume) if meshes else None

    # 4. judge the ASSEMBLED render against the whole-object brief. For a jointed mechanism
    # the render shows the HOME pose (every joint at 0 -- links may stack/fold), so the judge
    # gets that FACT as a render note: it still requires every key feature present, it just
    # stops failing a folded mechanism for not striking the typical working pose.
    verdict: AcceptanceVerdict | None = None
    if assembled_render is not None and assembled_render.exists():
        note = (
            "this is an ARTICULATED assembly rendered at its HOME pose (every joint at 0 "
            "degrees), so the links may stack vertically or fold together rather than strike "
            "the object's typical working pose; judge the presence and form of the parts and "
            "their interfaces, not the pose"
            if plan.joints
            else None
        )
        # OCCLUSION note (deep-review, the pump-in-a-tube class): parts the fit tier
        # measured as ENCLOSED are invisible in this exterior render -- without the note
        # the judge rejects a correct assembly for 'missing' the very part it cannot see
        # (a live case is on record). Their presence/seating was verified geometrically.
        enclosed_idx = (fit or {}).get("enclosed_parts") or []
        if enclosed_idx:
            # indices refer to the MESHES list (which drops unloadable parts) -- map
            # through the mesh keys' stl paths, the same identity the fidelity tier uses
            stl_names = {str(lr.final_stl): n for n, lr in part_results if lr.final_stl is not None}
            shown = ", ".join(
                stl_names.get(mesh_keys[i][0], f"part_{i}")
                if 0 <= i < len(mesh_keys)
                else f"part_{i}"
                for i in enclosed_idx
            )
            occl = (
                f"the part(s) {shown} are ENCLOSED inside other parts and are NOT visible "
                "in this exterior render; their presence and seating were verified "
                "geometrically -- judge only what is visible, and do not report them missing"
            )
            note = f"{note}; ALSO: {occl}" if note else occl
        if hw_fit_meshes:
            hw_note = (
                f"the render includes {len(hw_fit_meshes)} BOUGHT hardware component(s) "
                "(servo envelope blocks) shown as plain solids -- they are purchased "
                "parts, not printed geometry; judge the PRINTED parts and how they "
                "interface the hardware, and do not fault the hardware blocks' plainness"
            )
            note = f"{note}; ALSO: {hw_note}" if note else hw_note
        try:
            verdict, u = acceptance_judge(
                assembled_render, whole_brief, client=client, model=judge_model, render_note=note
            )
            total_cost += u.cost_usd
        except Exception:
            verdict = None

    # 5. honest disposition (no-false-PASS): every part must be a coherent body AND the assembly
    # must read as the object.
    expert_final = None
    if doomed is None:
        final_digest = (
            f"disposition inputs: parts {n_built}/{len(part_results)} built; "
            f"fit seated={None if fit is None else fit.get('seated')}; "
            f"motion={None if motion is None else motion.get('motion_status')}; "
            f"worst_case={None if worst_case is None else worst_case.get('issues')}; "
            f"envelope={envelope_issue}"
        )
        expert_final = run_expert_review(
            stage="final",
            prompt=prompt,
            summary=final_digest,
            mat=mat,
            client=client,
            model=brief_model,
        )
        if expert_final is not None and expert_final["blockers"]:
            print(
                f"expert-review(final): {expert_final['blockers']} blocker(s) flagged",
                flush=True,
            )
    expert_review = (
        {"plan": expert_plan, "final": expert_final}
        if (expert_plan is not None or expert_final is not None)
        else None
    )

    all_built = n_built == len(part_results)
    all_coherent = all_built and all(_part_coherent(lr) for _name, lr in part_results)
    unassemblable_notes: list[str] = []
    if not all_built:
        disposition = "no_build"
    elif verdict is not None and verdict.is_the_object:
        all_passed = all(lr.geometry_passed for _name, lr in part_results)
        seated = fit is None or bool(
            fit["seated"]
        )  # a floating / buried assembly is not a clean ship
        # CLEARANCE FOLD (no-false-PASS): the material-aware slip-fit verdict is computed but was
        # discarded before -- fold it in so a mate that BINDS / interferes (gap outside the
        # material's band) cannot ship clean. Caps at caveats only (never rejects): a usable model
        # whose fit needs a tolerance tweak. FAIL-OPEN -- absent clearance verdict does not cap.
        clearance_ok = fit is None or "clearance" not in fit or bool(fit["clearance"]["ok"])
        # Same for the MATERIAL pin fit: a pin whose measured radial gap is outside this material's
        # running-fit band is not a clean mechanical ship.
        pin_fit_ok = (
            motion is None
            or "material_pin_fit" not in motion
            or bool(motion["material_pin_fit"]["ok"])
        )
        # A declared mechanism must actually ARTICULATE without binding to ship clean
        # (no-false-PASS): a binding/non-moving joint caps the disposition at caveats.
        articulates = motion is None or bool(motion["articulates"])
        # Wave A: a per-part-clean assembly that outgrew the stated envelope is not a
        # clean ship (row 54: 163mm silently exceeded the stated 160).
        envelope_ok = envelope_issue is None
        # Wave B: printed-process worst-case -- nominal-clear is not enough when the
        # printer eats 0.25 mm; a pair that binds AS PRINTED cannot ship clean.
        worst_case_ok = worst_case is None or not worst_case["issues"]
        assemblable_ok = assembly_order is None or not assembly_order.get("stuck")
        # expert-panel blockers (LLM-surfaced + deterministically-confirmed) cap at
        # caveats: unaddressed blocker-class physics never ships as a clean certificate
        expert_ok = not any((er or {}).get("blockers", 0) for er in (expert_plan, expert_final))
        # metal DFM: a blocker (sub-minimum wall) means the part cannot be machined -> not
        # a clean ship; caveats (sharp corners) cap without failing.
        dfm_ok = dfm_report is None or dfm_report.verdict in ("pass", "caveats")
        # part fidelity: an open-boundary non-solid deliverable is a blocker -> not a
        # clean ship; massing / non-manifold caveats cap without failing.
        fidelity_ok = fidelity_rep is None or fidelity_rep.verdict in ("pass", "caveats")
        # the assembly report's OWN verdict gates (deep-review): ANY failing assembly
        # check caps the clean ship -- gears (a too-far pair sweeps clean), pin
        # coaxiality, print-in-place gaps, and every future check, by default.
        gear_ok = motion is None or not motion.get("assembly_fails")
        clean = (
            all_passed
            and seated
            and articulates
            and clearance_ok
            and pin_fit_ok
            and envelope_ok
            and worst_case_ok
            and assemblable_ok
            and expert_ok
            and dfm_ok
            and fidelity_ok
            and gear_ok
        )
        disposition = "shipped" if clean else "shipped_with_caveats"
        # STRICT DISPOSITION (default OFF -> byte-identical): a caveat must not absorb PHYSICAL
        # IMPOSSIBILITY. H3 shipped here at caveats while this same report said not-seated, one
        # part floating, and a pin fit that "will bind / not assemble".
        disposition, unassemblable_notes = apply_strict_disposition(disposition, fit, motion, mat)
    elif verdict is None and all_coherent:
        disposition = "shipped_with_caveats"  # judge unavailable on coherent parts -> caveats
        # The judge-unavailable door reaches caveats WITHOUT consulting fit/motion at all, so it
        # needs the same guard or the tier leaks whenever the judge errors (the L83 lesson).
        disposition, unassemblable_notes = apply_strict_disposition(disposition, fit, motion, mat)
    else:
        disposition = "rejected"

    from forge_mind.validator.manufacturability import scope_certificate

    certificate = scope_certificate(
        disposition=disposition,
        material_process=str(mat.process.value),
        fit_verdict=(None if fit is None else ("seated" if fit.get("seated") else "not-seated")),
        motion_status=(None if motion is None else motion.get("motion_status")),
        worst_case=worst_case,
        dfm=dfm_report,
        expert_blockers=sum((er or {}).get("blockers", 0) for er in (expert_plan, expert_final)),
        fidelity=fidelity_rep,
    )
    print(f"certificate: {certificate['scope']}", flush=True)

    _write_functional_report(
        out_dir,
        prompt,
        plan,
        part_results,
        verdict,
        disposition,
        total_cost,
        fit,
        mat_spec,
        motion,
        finish=finish_telemetry,
        envelope=envelope_issue,
        worst_case=worst_case,
        assembly_order=assembly_order,
        expert_review=expert_review,
        print_pack=print_pack,
        dfm=None if dfm_report is None else dfm_report.as_dict(),
        fidelity=None if fidelity_rep is None else fidelity_rep.as_dict(),
        certificate=certificate,
        unassemblable=unassemblable_notes,
    )
    return FunctionalResult(
        out_dir=out_dir,
        object_name=plan.object_name,
        parts=part_results,
        assembled_stl=assembled_stl,
        assembled_step=assembled_step,
        assembled_glb=assembled_glb,
        assembled_render=assembled_render,
        verdict=verdict,
        disposition=disposition,
        total_cost_usd=total_cost,
        whole_brief=whole_brief,
        fit=fit,
        motion=motion,
        failure=(
            f"structural part '{doomed}' failed to build; remaining work skipped (early exit)"
            if doomed is not None
            else None
        ),
        repair_note=repair_note,
        finish=finish_telemetry,
    )


def _write_functional_report(
    out_dir: Path,
    prompt: str,
    plan: FunctionalPlan,
    parts: list[tuple[str, LoopResult]],
    verdict: AcceptanceVerdict | None,
    disposition: str,
    cost: float,
    fit: dict[str, Any] | None = None,
    material: dict[str, Any] | None = None,
    motion: dict[str, Any] | None = None,
    finish: list[dict[str, Any]] | None = None,
    envelope: str | None = None,
    worst_case: dict[str, Any] | None = None,
    assembly_order: dict[str, Any] | None = None,
    expert_review: dict[str, Any] | None = None,
    print_pack: list[dict[str, Any]] | None = None,
    dfm: dict[str, Any] | None = None,
    fidelity: dict[str, Any] | None = None,
    certificate: dict[str, Any] | None = None,
    unassemblable: list[str] | None = None,
) -> None:
    report = {
        "prompt": prompt,
        "object_name": plan.object_name,
        "mode": "functional",
        "disposition": disposition,
        # A verdict with no evidence is not a verdict: when the strict tier blocks a ship, say
        # exactly which measured condition did it. Empty on every other path.
        "unassemblable_reasons": unassemblable or [],
        "assembly_note": plan.assembly_note,
        "material": material,
        "fit": fit,
        "motion": motion,
        "finish": finish,
        "envelope_issue": envelope,
        "worst_case": worst_case,
        "assembly_order": assembly_order,
        "expert_review": expert_review,
        "print_pack": print_pack,
        "dfm": dfm,
        "fidelity": fidelity,
        "certificate": certificate,
        "parts": [
            {
                "name": name,
                "prompt": next((p.prompt for p in plan.parts if p.name == name), None),
                "built": lr.final_stl is not None,
                "geometry_passed": lr.geometry_passed,
                "hard_ok": (
                    lr.validation.hard_ok
                    if lr.validation
                    else bool(getattr(lr, "resumed_hard_ok", False))
                ),
                "stopped_reason": lr.stopped_reason,
                "iterations": len(lr.iterations),
            }
            for name, lr in parts
        ],
        "acceptance": (
            {
                "is_the_object": verdict.is_the_object,
                "confidence": verdict.confidence,
                "missing_or_wrong": verdict.missing_or_wrong,
                "reason": verdict.reason,
            }
            if verdict is not None
            else None
        ),
        "total_cost_usd": round(cost, 4),
    }
    (out_dir / "functional_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = [
    "FunctionalPlan",
    "FunctionalResult",
    "PartPlan",
    "decompose_functional",
    "produce_functional_part",
]
