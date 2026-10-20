"""The build123d skills cheatsheet cached in the author's system prompt.

Version-correct for build123d 0.7+. Every code idiom in ``VERIFIED_IDIOMS`` was
pulled from the official docs (via context7) and PROVEN by executing it through the
sandbox (``tests/unit/agentic/test_cheatsheet_idioms.py`` runs every one), so the
author reads only API that actually runs on the installed kernel.
``author_skills_block()`` assembles the full reference from the same dict, so the
prose can never drift from the tested snippets.
"""

from __future__ import annotations

# Full, runnable parts (each binds ``result``). The test suite executes every one.
VERIFIED_IDIOMS: dict[str, str] = {
    "builder_basics": (
        "with BuildPart() as bp:\n"
        "    Box(80, 60, 10)            # centered at the origin\n"
        "    Cylinder(radius=8, height=30)   # fused onto the box\n"
        "result = bp.part"
    ),
    "fillet_chamfer": (
        "with BuildPart() as bp:\n"
        "    Box(80, 60, 10)\n"
        "    chamfer(bp.edges().group_by(Axis.Z)[-1], length=4)   # top ring of edges\n"
        "    fillet(bp.edges().filter_by(Axis.Z), radius=5)        # the 4 vertical edges\n"
        "result = bp.part"
    ),
    "safe_edges": (
        "with BuildPart() as bp:\n"
        "    Box(40, 40, 20)\n"
        "# round the 4 vertical edges; if 6 is too big for the local wall, it AUTO-BACKS-OFF\n"
        "# instead of raising and killing the script (returns the part un-rounded if impossible):\n"
        "result = safe_fillet(bp.part, bp.part.edges().filter_by(Axis.Z), radius=6)"
    ),
    "robust_selectors": (
        "with BuildPart() as bp:\n"
        "    Box(60, 60, 8)\n"
        "    with BuildSketch(top_face(bp)):                  # NAME the face -- never faces()[i]\n"
        "        with GridLocations(40, 40, 2, 2):\n"
        "            Circle(clearance_hole_d('M4') / 2)\n"
        "    extrude(amount=-20, mode=Mode.SUBTRACT)          # 4 through holes\n"
        "result = safe_fillet(bp.part, vertical_edges(bp.part), radius=3)  # round the upright corners"
    ),
    "pocket": (
        "with BuildPart() as bp:\n"
        "    Box(120, 200, 12)\n"
        "    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):     # the top face\n"
        "        with GridLocations(0, 90, 1, 2):                  # 2 panels stacked in Y\n"
        "            Rectangle(80, 70)\n"
        "    extrude(amount=-4, mode=Mode.SUBTRACT)                # 4mm-deep recessed panels\n"
        "    fillet(bp.edges(Select.LAST).filter_by(Axis.Z), radius=3)  # soften the panel corners\n"
        "result = bp.part"
    ),
    "through_holes": (
        "with BuildPart() as bp:\n"
        "    Box(80, 60, 10)\n"
        "    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):      # the top face\n"
        "        with GridLocations(50, 30, 2, 2):                  # derive the 4 centres from the COUNT\n"
        "            Circle(2.75)                                   # M5 clearance 5.5mm dia -> r 2.75\n"
        "    extrude(amount=-20, mode=Mode.SUBTRACT)                # cut PAST the far face -> a real THROUGH hole\n"
        "result = bp.part"
    ),
    # ADR-0133 fix 2 (FORGE_ONECALL_FEATURES): one-call hole features. R3 measured the
    # hand-rolled hole stack (BuildSketch+Circle+extrude SUBTRACT) as the density signal
    # (46/135 files, 78 stacks) failing on through-depth arithmetic and stale-face
    # re-selection -- both eliminated BY CONSTRUCTION here. Kernel-verified 2026-07-26
    # (verify_onecall_idioms.json: 1 fused solid each, expected volumes).
    "one_call_holes": (
        "with BuildPart() as bp:\n"
        "    Box(80, 60, 10)\n"
        "    with Locations(bp.faces().sort_by(Axis.Z)[-1]):      # place ON the top face\n"
        "        with GridLocations(50, 30, 2, 2):                # centres from the COUNT\n"
        "            Hole(radius=clearance_hole_d('M5') / 2)      # depth omitted = THROUGH, both faces open\n"
        "result = bp.part"
    ),
    "one_call_counterbore": (
        "cb_d, cb_depth = counterbore_dims('M4')                  # EXACT ISO 4762 head recess\n"
        "with BuildPart() as bp:\n"
        "    Box(70, 30, 12)\n"
        "    with Locations(bp.faces().sort_by(Axis.Z)[-1]):\n"
        "        with GridLocations(40, 0, 2, 1):\n"
        "            CounterBoreHole(\n"
        "                radius=clearance_hole_d('M4') / 2,\n"
        "                counter_bore_radius=cb_d / 2,\n"
        "                counter_bore_depth=cb_depth,\n"
        "            )                                            # bore + recess, ONE call, no re-select\n"
        "result = bp.part\n"
        "# countersunk flat-head: CounterSinkHole(radius=r, counter_sink_radius=csk_r) -- same pattern"
    ),
    "shell": (
        "with BuildPart() as box:\n"
        "    Box(40, 30, 20)\n"
        "    offset(amount=-2, openings=box.faces().sort_by(Axis.Z)[-1])  # hollow, open top, 2mm walls\n"
        "result = box.part"
    ),
    "safe_shell": (
        "with BuildPart() as bp:\n"
        "    Box(40, 30, 25)\n"
        "# hollow to 2mm walls, open top; RAISES a clear message (never returns solid) if\n"
        "# the wall is too thick for the geometry or the result slivers:\n"
        "result = safe_shell(bp.part, thickness=2)"
    ),
    "revolve": (
        "with BuildPart() as part:\n"
        "    with BuildSketch(Plane.XZ):                # half-profile beside the Z axis\n"
        "        with BuildLine():\n"
        "            Polyline((0, 0), (15, 0), (15, 40), (8, 60), (0, 60), (0, 0))\n"
        "        make_face()\n"
        "    revolve(axis=Axis.Z)\n"
        "result = part.part"
    ),
    "loft": (
        "with BuildPart() as part:\n"
        "    with BuildSketch(Plane(origin=(0, 0, 0))):\n"
        "        Circle(20)\n"
        "    with BuildSketch(Plane(origin=(0, 0, 40))):\n"
        "        Rectangle(24, 24)\n"
        "    loft()                                     # blend the stacked sections\n"
        "result = part.part"
    ),
    "sweep": (
        "with BuildPart() as part:\n"
        "    with BuildLine():\n"
        "        l1 = Line((0, 0), (0, 40))\n"
        "        l2 = JernArc(start=l1 @ 1, tangent=l1 % 1, radius=20, arc_size=90)\n"
        "    with BuildSketch(Plane.XZ):                # profile at the path start\n"
        "        Circle(5)\n"
        "    sweep()\n"
        "result = part.part"
    ),
    "wavy_vessel": (
        "import math\n"
        "def wavy_ring(base_r, amp, waves=5, n=64):       # ONE continuous closed wavy ring of points\n"
        "    return [((base_r + amp*math.sin(waves*2*math.pi*i/n))*math.cos(2*math.pi*i/n),\n"
        "             (base_r + amp*math.sin(waves*2*math.pi*i/n))*math.sin(2*math.pi*i/n)) for i in range(n)]\n"
        "with BuildPart() as vase:\n"
        "    # (z, radius, wave_amplitude): amp stays 0 down the body, grows only near the rim\n"
        "    for z, base_r, amp in [(0,30,0),(45,45,0),(90,28,0),(140,40,0),(165,50,9),(185,50,9)]:\n"
        "        with BuildSketch(Plane.XY.offset(z)):\n"
        "            with BuildLine():\n"
        "                Spline(*wavy_ring(base_r, amp), periodic=True)   # closed smooth wavy ring (NO close=; no repeated 1st pt)\n"
        "            make_face()\n"
        "    loft()                                       # ONE solid: a single continuous undulating wall\n"
        "    offset(amount=-4, openings=vase.faces().sort_by(Axis.Z)[-1])  # hollow it, open wavy top\n"
        "    fillet(vase.edges().group_by(Axis.Z)[-1], radius=1.2)         # blunt the rim lip so it is not a knife edge\n"
        "result = vase.part"
    ),
    "grip_loop": (
        "with BuildPart() as bp:\n"
        "    with BuildSketch(Plane.XZ):                 # a graspable loop standing UP in a vertical plane\n"
        "        with BuildLine():                        # OUTER teardrop / D outline (closed)\n"
        "            Spline((-18, 2), (-22, 34), (0, 74), (22, 34), (18, 2), (-18, 2))\n"
        "        make_face()\n"
        "        with BuildLine():                        # INNER outline = the finger APERTURE\n"
        "            Spline((-9, 14), (-11, 34), (0, 56), (11, 34), (9, 14), (-9, 14))\n"
        "        make_face(mode=Mode.SUBTRACT)            # a loop FRAME with a real opening to grip through\n"
        "    extrude(amount=10)                            # the grip width\n"
        "    fillet(bp.edges().filter_by(Axis.Y), radius=2)   # round the grip so it is comfortable\n"
        "result = bp.part"
    ),
    "standard_holes": (
        "with BuildPart() as bp:\n"
        "    Box(60, 60, 6)\n"
        "    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):       # the top face\n"
        "        with GridLocations(40, 40, 2, 2):                   # 4 centres from the COUNT\n"
        "            Circle(clearance_hole_d('M3') / 2)              # EXACT ISO 273 M3 clearance (3.4mm dia)\n"
        "    extrude(amount=-12, mode=Mode.SUBTRACT)                 # cut through both faces\n"
        "result = bp.part"
    ),
    "polar_holes": (
        "with BuildPart() as bp:\n"
        "    Cylinder(radius=30, height=8)                           # a flange disc\n"
        "    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):\n"
        "        with PolarLocations(radius=22, count=6):             # 6 centres evenly on a 44mm bolt circle\n"
        "            Circle(2.25)                                     # M4 clearance 4.5mm dia -> r 2.25\n"
        "    extrude(amount=-16, mode=Mode.SUBTRACT)                  # through ALL the way\n"
        "result = bp.part"
    ),
    "radial_set_screw_holes": (
        "with BuildPart() as bp:\n"
        "    Cylinder(radius=11, height=20)                          # a hub, axis +Z\n"
        "    for ang in (0, 90):                                     # two set-screws 90 deg apart\n"
        "        with Locations(Rot(0, 0, ang) * Pos(11, 0, 0)):     # a point ON the hub OD at this angle\n"
        "            # RADIAL blind tap: rotation=(0,90,0) turns the bore to point inward along the radius;\n"
        "            # nominal dia for a threaded set-screw (thread_hole_d), depth = into the wall, not through\n"
        "            Cylinder(radius=thread_hole_d('M4') / 2, height=12, rotation=(0, 90, 0), mode=Mode.SUBTRACT)\n"
        "result = bp.part"
    ),
    "linear_rib_array": (
        "with BuildPart() as bp:\n"
        "    Box(100, 60, 4)                                          # the base plate (z -2..2)\n"
        "    with BuildSketch(Plane.XY):                              # sketch INSIDE the plate so ribs overlap it\n"
        "        with GridLocations(20, 0, 4, 1):                     # 4 rib centres from the COUNT\n"
        "            Rectangle(3, 50)\n"
        "    extrude(amount=12)                                       # ribs rise THROUGH the top face -> fused\n"
        "result = bp.part"
    ),
    "mirror_feature": (
        "with BuildPart() as bp:\n"
        "    Box(80, 40, 10)\n"
        "    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):\n"
        "        with Locations((30, 10)):\n"
        "            Circle(3)                                        # model ONE instance...\n"
        "        mirror(about=Plane.XZ)                               # ...duplicate across Y\n"
        "        mirror(about=Plane.YZ)                               # ...and across X -> 4 symmetric holes\n"
        "    extrude(amount=-20, mode=Mode.SUBTRACT)\n"
        "result = bp.part"
    ),
    "counterbore_holes": (
        "cb_d, cb_depth = counterbore_dims('M4')                      # EXACT ISO 4762 head recess\n"
        "with BuildPart() as bp:\n"
        "    Box(70, 30, 12)\n"
        "    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):\n"
        "        with GridLocations(40, 0, 2, 1):\n"
        "            Circle(clearance_hole_d('M4') / 2)               # the through bore first\n"
        "    extrude(amount=-24, mode=Mode.SUBTRACT)\n"
        "    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):        # RE-SELECT the face after the boolean\n"
        "        with GridLocations(40, 0, 2, 1):\n"
        "            Circle(cb_d / 2)                                 # the head recess on the same centres\n"
        "    extrude(amount=-cb_depth, mode=Mode.SUBTRACT)\n"
        "result = bp.part"
    ),
    "helix_coil": (
        "with BuildPart() as bp:\n"
        "    with BuildLine():\n"
        "        path = Helix(pitch=12, height=48, radius=12)         # 4 turns\n"
        "    with BuildSketch(Plane(origin=path @ 0, z_dir=path % 0)):  # profile at the path start\n"
        "        Circle(2.5)                                          # the wire cross-section\n"
        "    sweep(is_frenet=True)                                    # one solid coil\n"
        "result = bp.part"
    ),
    "catalog_screw": (
        "from forge_mind.agentic.catalog import catalog_screw   # SELECT a verified part by id\n"
        "# the exact ISO 4762 M5x20 socket-head cap screw -- never hand-modeled\n"
        "result = catalog_screw('shcs_m5x20')"
    ),
    "catalog_bearing": (
        "from forge_mind.agentic.catalog import catalog_bearing\n"
        "# the exact 608 deep-groove ball bearing (bore 8, OD 22, width 7) by designation\n"
        "result = catalog_bearing('608')"
    ),
    "external_iso_thread": (
        "from bd_warehouse.thread import IsoThread            # REAL fastener thread, never swept by hand\n"
        "length = 16.0\n"
        "# default end_finishes ('fade','square') runs ~1 pitch past `length` -- size the core for it\n"
        "thread = IsoThread(major_diameter=8.0, pitch=1.25, length=length, external=True)\n"
        "with BuildPart() as bp:\n"
        "    Cylinder(radius=thread.min_radius, height=length, align=(Align.CENTER, Align.CENTER, Align.MIN))\n"
        "result = bp.part + thread                             # ONE fused solid (bare IsoThread is loose loops)"
    ),
    "internal_iso_thread": (
        "from bd_warehouse.thread import IsoThread\n"
        "length, major = 12.0, 8.0\n"
        "thread = IsoThread(major_diameter=major, pitch=1.25, length=length, external=False,\n"
        "                   end_finishes=('square', 'square'))\n"
        "with BuildPart() as bp:\n"
        "    Box(30, 30, length)\n"
        "    # pre-drill SLIGHTLY UNDER major/2: exactly major/2 null-shapes the union (measured)\n"
        "    Cylinder(radius=major / 2.0 - 0.1, height=length + 2.0, mode=Mode.SUBTRACT)\n"
        "result = bp.part + (Pos(0, 0, -length / 2.0) * thread)"
    ),
    "involute_spur_gear": (
        "from forge_mind.agentic.gears import spur_gear        # REAL involute teeth, never polygon-faked\n"
        "# module x teeth = pitch diameter; mesh two gears at (m*(z1+z2))/2 center distance\n"
        "result = spur_gear(module=1.5, teeth=24, thickness=8, bore=6.0)"
    ),
    "tapered_limb_shell": (
        "# a tapered member (limb / furniture leg / handle / spout): 3 elliptical sections\n"
        "# lofted down the member's axis, root wider than tip -- ONE smooth taper, never a Box\n"
        "with BuildPart() as limb:\n"
        "    for z, rx, ry in [(0, 17, 14), (45, 15, 12), (90, 11, 9)]:   # (station, x-rad, y-rad)\n"
        "        with BuildSketch(Plane.XY.offset(z)):\n"
        "            Ellipse(rx, ry)\n"
        "    loft()                                     # one continuous tapered shell volume\n"
        "result = limb.part"
    ),
    "joint_module": (
        "# a pivot pod (shoulder/elbow/hip/knee, hinge knuckle, gimbal pivot): short cylinder\n"
        "# on the JOINT AXIS with DOMED ends and the pivot bore through -- a designed module\n"
        "with BuildPart() as jm:\n"
        "    Cylinder(radius=14, height=30)                        # module body, axis +Z\n"
        "    with Locations((0, 0, 15)):\n"
        "        Sphere(radius=14)                                 # domed cap, one end\n"
        "    with Locations((0, 0, -15)):\n"
        "        Sphere(radius=14)                                 # domed cap, other end\n"
        "    Cylinder(radius=3.05, height=70, mode=Mode.SUBTRACT)  # 6.1 mm pivot bore, through both domes\n"
        "result = jm.part"
    ),
    "backward_curved_impeller": (
        "from forge_mind.agentic.flow_forms import backward_curved_impeller  # REAL curved blades\n"
        "# blades sweep BACKWARD (default 35 deg) from just outside the hub to the rim --\n"
        "# never flat radial plates. bore is a diameter; blades stand on the disc.\n"
        "result = backward_curved_impeller(35, 4, 12, 7, hub_r=8, hub_h=6, bore=6)"
    ),
    "crank_web_counterweight": (
        "from forge_mind.agentic.flow_forms import crank_web  # crescent-counterweight web\n"
        "# main journal on the ORIGIN axis, crankpin boss at (+throw, 0), counterweight lobe\n"
        "# opposite the pin -- the shape that balances the throw, never a plain disc.\n"
        "result = crank_web(6, 12, main_boss_r=9, pin_boss_r=6, cw_r=16, main_bore=6, pin_bore=5)"
    ),
    "tapered_fin_stack": (
        "from forge_mind.agentic.flow_forms import tapered_fin_stack  # air-cooled fin barrel\n"
        "# fins TAPER root->tip (stronger FDM root, real look; square fins read as toy). The\n"
        "# bore stays OPEN end-to-end -- the helper never closes its own tube.\n"
        "result = tapered_fin_stack(14, 10, 60, 9, fin_r=20, t_root=2.4)"
    ),
    "chamfered_bore_entry": (
        "# EVERY functional bore gets a lead-in chamfer at BOTH entries (assembly lead-in +\n"
        "# elephant-foot immunity). Select the bore's entry edges BY RADIUS, never edges()[i].\n"
        "with BuildPart() as bp:\n"
        "    Cylinder(radius=15, height=12)\n"
        "    Cylinder(radius=4.0, height=14, mode=Mode.SUBTRACT)   # 8 mm bore\n"
        "    bore_edges = bp.edges().filter_by(GeomType.CIRCLE).filter_by(\n"
        "        lambda e: abs(e.radius - 4.0) < 1e-6\n"
        "    )\n"
        "    chamfer(bore_edges, length=0.6)                       # 0.5-1.0 mm lead-in band\n"
        "result = bp.part"
    ),
    "servo_pocket": (
        "# MG996R-class servo bay in a housing wall: body window + the CITED 4-hole lug\n"
        "# pattern (49.5 x 10.0) -- the bought servo drops in and SCREWS down, never glued\n"
        "with BuildPart() as bay:\n"
        "    Box(64, 30, 8)                                        # the housing wall / bulkhead\n"
        "    Box(41.2, 20.2, 20, mode=Mode.SUBTRACT)               # body window: 40.7 x 19.7 + 0.25 air/side\n"
        "    with BuildSketch(bay.faces().sort_by(Axis.Z)[-1]):\n"
        "        with GridLocations(49.5, 10.0, 2, 2):             # the datasheet lug-hole pattern\n"
        "            Circle(4.5 / 2)                               # M4 clearance through the lugs\n"
        "    extrude(amount=-20, mode=Mode.SUBTRACT)\n"
        "result = bay.part"
    ),
}

# Teaching order + the one-line intro for each idiom.
_ORDER: list[tuple[str, str]] = [
    (
        "builder_basics",
        "Wrap construction in `with BuildPart() as bp:`. Consecutive solids FUSE; cut with `mode=Mode.SUBTRACT`. Take the solid with `result = bp.part`. Units are mm; primitives are centered at the origin.",
    ),
    (
        "fillet_chamfer",
        "Select edges, then round/bevel. `group_by(Axis.Z)[-1]` = the top ring; `filter_by(Axis.Z)` = all vertical edges. Do this LAST, after holes/pockets, radius < half the local thickness.",
    ),
    (
        "safe_edges",
        "PREFER `safe_fillet(part, edges, radius)` / `safe_chamfer(part, edges, length)` over the raw `fillet`/`chamfer` for COSMETIC rounding: if the radius exceeds the local wall the raw call RAISES and kills the whole script, but the safe wrapper auto-backs-off the radius and -- if it still cannot -- returns the part un-rounded so a cosmetic round never destroys a valid build. Use the raw `fillet` only when the round is structural and must be exact.",
    ),
    (
        "robust_selectors",
        "NAME the face/edge you cut, never index it. `top_face(part)` / `bottom_face(part)` / `face_facing(part, Axis.X)` (a SIDE face by its normal) / `largest_face(part)` pick a face by position; `vertical_edges(part)` / `top_edges(part)` pick edge sets. A boolean RENUMBERS the B-rep, so call these FRESH right before each cut -- a stale `faces()[3]` silently grabs the wrong face and the hole lands wrong or vanishes.",
    ),
    (
        "safe_shell",
        "HOLLOW a vessel/enclosure with `safe_shell(part, thickness, openings=<face>)` (default opening: the top face). It offsets inward to `thickness` walls and -- unlike safe_fillet -- RAISES a clear, repairable message rather than silently returning a solid when the wall is too thick or the result slivers (a cup must NEVER come back solid). Build the SOLID outer form first, then safe_shell it; never hand-draw an inner+outer two-rail profile.",
    ),
    (
        "pocket",
        "Sketch on a face, replicate with `GridLocations`/`PolarLocations`/`Locations`, then `extrude(amount=-depth, mode=Mode.SUBTRACT)`. This is how to make recessed panels (a paneled door), vent slots, a lid recess.",
    ),
    (
        "through_holes",
        "Bolt / clearance holes: sketch on the face, replicate the centres from the COUNT with `GridLocations`/`PolarLocations` (never copy-paste N circles), then `extrude` PAST the opposite face (amount more negative than the thickness) so every hole goes ALL THE WAY THROUGH and opens on both sides. Standard clearance diameters: M3 3.4, M4 4.5, M5 5.5 mm.",
    ),
    (
        "shell",
        "`offset(amount=-wall, openings=<face(s)>)` hollows a solid to `wall`-thick walls with an opening at the named faces. The way to make enclosures, trays, bottles thin instead of solid.",
    ),
    (
        "revolve",
        "Sketch a half-profile in a plane that CONTAINS the axis (here Plane.XZ beside Axis.Z), close it with `make_face()`, then `revolve(axis=...)`. For bottles, knobs, nozzles, hammer heads.",
    ),
    (
        "loft",
        "Stack `BuildSketch` cross-sections on parallel planes and `loft()` to blend them. For organic transitions, fairings, handles that change section.",
    ),
    (
        "sweep",
        "Define a path with `BuildLine`, a profile with `BuildSketch` at the path start, then `sweep()`. For tubes, handles, hooks, conduits.",
    ),
    (
        "wavy_vessel",
        "WAVY / FLUTED / SCALLOPED / PETAL rim (organic vase, calla-lily cup, flower pot): make the undulation by VARYING THE RADIUS within ONE continuous loft of stacked closed wavy rings -- amp 0 down the body, growing only near the rim -- then shell it. The whole wall is ONE solid. NEVER carve the waves by subtracting angular cutters or building separate petals/staves: that slices the thin wall into disconnected scallops (the mesh splits into N bodies) and leaves knife edges. Close each ring with `Spline(*pts, periodic=True)` (there is NO `close=` arg, and do NOT repeat the first point), and blunt the open rim lip with a small `fillet` so no edge is sub-wall.",
    ),
    (
        "grip_loop",
        "GRIP / PULL / LOOP HANDLE (door pull, drawer pull, teardrop / D handle): a graspable LOOP with a finger APERTURE you wrap your hand THROUGH -- NOT a solid spike, knob, or blob. Build it as a closed OUTER outline minus a closed INNER outline (the opening) on ONE vertical sketch plane (make_face() then make_face(mode=Mode.SUBTRACT)), extrude to the grip width, and fillet the grip edges so it is comfortable. Shape the outline per the brief (teardrop, D, oval, bow). Stand it up and merge its leg ends into the base/body so the whole thing is ONE fused solid.",
    ),
    (
        "standard_holes",
        "STANDARD HARDWARE holes: get the EXACT dimension from the standards helpers, never guess. `clearance_hole_d('M3')` is the ISO 273 bolt clearance (3.4mm); replicate the centres from the COUNT with GridLocations/PolarLocations, then extrude PAST the far face so each hole is a real THROUGH hole. Same for tap_hole_d / counterbore_dims / nut_pocket_dims / bearing_seat_dims -- see the standards reference below. "
        "HOLE-SIZE RULE (measured, bench rows 17-18) -- THREE cases, in priority order: "
        "(1) a diameter the brief STATES as a number ('pivot bore 6 mm diameter') is EXACT -- drill "
        "that number, never a clearance/tap substitute (a stated 6 mm redrilled as M6 clearance 6.6 "
        "is a WRONG part). "
        "(2) a THREADED / tapped / set-screw hole ('M8 threaded bore', 'M4 tapped', 'M3 set-screw') "
        "is modeled at its NOMINAL major diameter: `thread_hole_d('M8')` -> 8.0. The nominal size "
        "IDENTIFIES the thread; model that envelope, NOT the tap drill and NOT the clearance. "
        "(3) a CLEARANCE hole a bolt PASSES THROUGH -> `clearance_hole_d('M3')` (ISO 273, 3.4mm). "
        "Build EVERY brief-named hole including radial and blind set-screw holes -- a dropped "
        "secondary hole fails the part even when the main bore is perfect (bench row 18: pulley + "
        "coupling built the central bore, dropped their 2 set-screw holes).",
    ),
    (
        "polar_holes",
        "BOLT CIRCLE / RADIAL pattern: `PolarLocations(radius, count)` replicates sketch objects evenly on a circle -- flange bolt holes, vent slots, spoke cutouts. Derive the count from the brief; NEVER hand-place radial copies (a hand-placed circle drifts off the true pitch circle).",
    ),
    (
        "linear_rib_array",
        "RIB / FIN ARRAY (heatsink fins, stiffening ribs, card guides): replicate Rectangles with `GridLocations` and extrude so every rib OVERLAPS into the base -- sketch AT or INSIDE the base body and extrude THROUGH its face, so the union fuses into one solid instead of leaving surface-touching fins.",
    ),
    (
        "mirror_feature",
        "MIRROR a feature across a plane: model ONE instance, then `mirror(about=Plane.XZ)` inside the sketch duplicates it; chain two mirrors for 4-quadrant symmetry. For symmetric hole pairs, pockets, or lugs -- build one, mirror the rest, and the symmetry is exact by construction.",
    ),
    (
        "counterbore_holes",
        "COUNTERBORE stack (socket-head screw recess): fetch the EXACT ISO 4762 recess with `counterbore_dims(size)`, cut the clearance bore THROUGH first, then cut the head recess from the SAME face RE-SELECTED after the boolean (a stale face reference cuts in the wrong place). For a COUNTERSINK, chamfer the bore's top edge instead of cutting a cone by hand.",
    ),
    (
        "helix_coil",
        "HELIX / COIL / SPRING / SPIRAL: define a `Helix(pitch, height, radius)` path in BuildLine, sketch the wire profile on a plane AT THE PATH START (`Plane(origin=path @ 0, z_dir=path % 0)`), then `sweep(is_frenet=True)`. Keep pitch > 2x the profile diameter so coils never self-intersect. For FASTENER threads, never sweep by hand -- import bd_warehouse IsoThread (see rules).",
    ),
    (
        "catalog_screw",
        "STANDARD FASTENER from the CATALOG: `from forge_mind.agentic.catalog import catalog_screw` then `catalog_screw('shcs_m5x20')` returns the exact ISO solid by id (shcs/button/csk/hexbolt/setscrew, m2..m8, common lengths). Also catalog_washer / catalog_nut. NEVER model a screw or thread by hand; SELECT a catalog id (a bad id raises with suggestions). Place with Pos(x, y, z) * part.",
    ),
    (
        "catalog_bearing",
        "STANDARD BEARING from the CATALOG: `from forge_mind.agentic.catalog import catalog_bearing` then `catalog_bearing('608')` (or a metric id like 'bearing_dgbb_5x16x5'). NEVER model a bearing by hand; seat it in a bore (use bearing_mount(bearing='608', shaft_d=8, housing_bore_d=22) to check the fit). See the CATALOG reference below.",
    ),
    (
        "radial_set_screw_holes",
        "RADIAL SET-SCREW holes on a hub/collar: Locations(Rot(0,0,ang) * Pos(hub_r, 0, 0)) puts the point ON the OD; rotation=(0,90,0) turns the bore INWARD along the radius; nominal thread_hole_d('M4'), blind depth into the wall. Two at 90 deg is the standard shaft-lock pattern.",
    ),
    (
        "external_iso_thread",
        "EXTERNAL SCREW THREAD (a threaded boss/stud/cap): import bd_warehouse IsoThread(external=True), model the core at thread.min_radius, then `core + thread` -- the SUM is the one fused solid. The default end fade runs ~1 pitch past `length` (plan for it); bare IsoThread is loose loop solids and simple=True is EMPTY geometry.",
    ),
    (
        "internal_iso_thread",
        "INTERNAL THREAD (a nut / tapped bore / jar lid): pre-drill SLIGHTLY UNDER major/2 (exactly major/2 null-shapes the union -- measured), then add IsoThread(external=False) positioned into the bore. Same fused-sum rule as the external idiom.",
    ),
    (
        "involute_spur_gear",
        "SPUR GEAR with REAL involute teeth: `from forge_mind.agentic.gears import spur_gear` then spur_gear(module, teeth, thickness, bore=...) -- one watertight solid, axis +Z. NEVER fake teeth with polygons. Mesh two gears at center distance (module*(z1+z2))/2; tip diameter = module*(teeth+2).",
    ),
    (
        "tapered_limb_shell",
        "TAPERED ORGANIC MEMBER -- a humanoid/figure limb (thigh, shin, arm), a furniture leg, a tool handle, a spout, a blade root, any member whose section changes along its axis: LOFT 3-4 stacked Ellipse / rounded sections along the member's axis, larger at the root, smaller at the tip -- ONE smooth tapered volume that reads as crafted. NEVER ship a limb / leg / handle as a bare Box (a box member reads as a lab prototype); the ellipse stack + loft costs the same and looks designed. Cut pockets / bores into the lofted volume AFTER the loft.",
    ),
    (
        "joint_module",
        "JOINT / PIVOT POD -- a humanoid shoulder / elbow / hip / knee, a hinge knuckle, a tripod or gimbal pivot, the capped end of any moving link: a short Cylinder COAXIAL with the pivot axis, DOMED with a Sphere cap on each end, pivot bore subtracted straight through the axis -- dresses a working joint as a designed module instead of a bare pin through bare plates. Scale the diameter to the member it joins (~24-36 mm on a 350 mm figure).",
    ),
    (
        "servo_pocket",
        "SERVO BAY / actuator pocket (MG996R-class hobby servo): the housing card carries a BODY WINDOW at the stated body dims plus stated air per side (MG996R: 40.7 x 19.7 + 0.25 -> a 41.2 x 20.2 window) AND the DATASHEET lug-hole pattern (MG996R: 4 holes on 49.5 x 10.0, M4 clearance) so the bought servo drops in and SCREWS down -- never glued, never friction-held. Torque leaves through the servo HORN: the driven link carries the horn bore circle, never grips the case. Copy exact registry constants (assembly/servo.py) into the card when they exist.",
    ),
    (
        "backward_curved_impeller",
        "CENTRIFUGAL IMPELLER / blower / fan wheel: `from forge_mind.agentic.flow_forms import backward_curved_impeller` -- disc + hub + N genuinely BACKWARD-CURVED blades (default 35 deg sweep) as ONE watertight solid, axis +Z, bore is a diameter. NEVER build an impeller as flat radial plates on a disc: flat plates read as a toy fan, curve is what makes it a real machine part.",
    ),
    (
        "crank_web_counterweight",
        "CRANKSHAFT WEB: `from forge_mind.agentic.flow_forms import crank_web` -- main-journal boss on the origin axis, crankpin boss at (+throw,0), and a CRESCENT COUNTERWEIGHT lobe opposite the pin, one solid. A real crank web is never a plain disc; the counterweight is what balances the throw and what makes the silhouette read as an engine part. Through-bores for journal stock via main_bore/pin_bore.",
    ),
    (
        "tapered_fin_stack",
        "AIR-COOLED FIN BARREL (cylinder jug, heatsink collar, finned motor case): `from forge_mind.agentic.flow_forms import tapered_fin_stack` -- core tube + evenly-pitched annular fins that TAPER root-to-tip (t_root -> t_tip). Square constant-thickness fins read as toy parts and print weaker; the taper is both the real look and the stronger FDM root. The bore stays OPEN end to end.",
    ),
    (
        "chamfered_bore_entry",
        "EDGE BREAKS ARE NOT OPTIONAL: every FUNCTIONAL BORE carries a 0.5-1.0 mm lead-in chamfer at BOTH entries (assembly lead-in for the pin, elephant-foot immunity on the bed side), selected BY RADIUS via edges().filter_by(GeomType.CIRCLE) -- never by index. Free outer edges of plates/brackets get a 0.5-1.0 mm chamfer or fillet where the printer allows. A part with knife edges everywhere reads as un-engineered and binds on assembly; the finish pass CANNOT do this for you (it is forbidden from touching mating surfaces), so the AUTHOR does it.",
    ),
]

# ADR-0133 fix 2: the one-call swap map -- the ONLY source of the ON-order (law 2:
# derived, never a hand-copied second list). Swapped: the two idioms with direct
# one-call twins; everything else identical. mirror_solid was priced OUT (measured
# ~nil savings; adversarial pricing 2026-07-26).
_ONECALL_SWAPS: dict[str, tuple[str, str]] = {
    "through_holes": (
        "one_call_holes",
        "Bolt / clearance holes are ONE call: place `Hole(radius=...)` on the face via "
        "`Locations`/`GridLocations`/`PolarLocations` (centres from the COUNT, never "
        "copy-pasted). Depth omitted = a real THROUGH hole, open both sides -- no sketch, "
        "no extrude arithmetic, no far-face guesswork. Standard clearance diameters: "
        "M3 3.4, M4 4.5, M5 5.5 mm.",
    ),
    "counterbore_holes": (
        "one_call_counterbore",
        "A counterbore is ONE call: `CounterBoreHole()` cuts the through bore AND the "
        "head recess together on the same centres -- no boolean re-selection, exact ISO "
        "dims from `counterbore_dims()`. Countersunk flat-heads: `CounterSinkHole()`, "
        "same pattern.",
    ),
}
_ONECALL_ORDER: list[tuple[str, str]] = [_ONECALL_SWAPS.get(k, (k, intro)) for k, intro in _ORDER]

_HEADER = (
    "# build123d author cheatsheet (v0.7+, every snippet below is execution-verified)\n"
    "\n"
    "You author ONE part as Python. `from build123d import *` and the Forge-Mind helpers\n"
    "are ALREADY imported -- do not re-import the kernel. Bind the finished solid to a\n"
    "top-level variable named `result`. Build123d is a real BREP/OpenCASCADE kernel:\n"
    "prefer revolve / loft / sweep / shell / fillets over stacks of boxes, so parts come\n"
    "out as thin and detailed as the real object, not blocky approximations."
)

_HELPER_REFERENCE = (
    "## Forge-Mind helpers (already imported -- use these for hardware, don't hand-roll trig)\n"
    "SUBTRACT tools have their open face on z=0 and extend in -Z; `Pos(x, y, z_face) * tool()`\n"
    "puts the opening on a face at height z_face, then subtract it from the body.\n"
    "\n"
    "- `bolt_circle_points(pcd, count, start_angle=0) -> [(x, y), ...]`   # for Locations(*pts)\n"
    "- `bolt_holes(pcd, hole_d, count, depth, start_angle=0) -> Part`     # SUBTRACT tool\n"
    "- `counterbore_tool(thru_d, head_d, head_depth, thru_depth) -> Part` # SUBTRACT tool\n"
    "- `bearing_seat_tool(bore_d, seat_d, seat_depth, thru_depth) -> Part`# SUBTRACT tool\n"
    "- `rounded_box(length, width, height, radius) -> Part`               # finished solid\n"
    "- `radial_pattern(feature, count, axis=Axis.Z, start_angle=0, fuse=True) -> Part`  # array a PRE-BUILT solid around an axis\n"
    "- `linear_pattern(feature, count, spacing, axis=Axis.X, fuse=True) -> Part`        # array a PRE-BUILT solid along an axis\n"
    "- `safe_fillet(part, edges, radius, min_radius=0.2) -> Part`         # fillet that auto-backs-off, never raises\n"
    "- `safe_chamfer(part, edges, length, min_length=0.2) -> Part`        # chamfer that auto-backs-off, never raises\n"
    "- `safe_shell(part, thickness, openings=None) -> Part`               # hollow a vessel; raises clearly on failure, never returns solid\n"
    "- `top_face(part)` / `bottom_face(part)` / `largest_face(part)` -> Face   # NAME a face, never index it\n"
    "- `face_facing(part, axis, positive=True) -> Face`                   # a SIDE face by its normal (Axis.X/Y/Z)\n"
    "- `vertical_edges(part)` / `top_edges(part)` -> edges                # edge sets for fillet/chamfer\n"
    "- `spur_gear(module, teeth, thickness, pressure_angle=20, bore=0, clock_deg=0) -> Part`  # involute\n"
    "- `ring_gear(module, teeth, thickness, pressure_angle=20, rim=5) -> Part`   # internal gear\n"
    "- `center_distance(GearSpec(module, z1), GearSpec(module, z2)) -> float`    # = module*(z1+z2)/2;\n"
    "  it takes two GearSpec OBJECTS, never raw numbers -- or just compute module*(z1+z2)/2 yourself\n"
    "- `mesh_clock_deg(teeth) -> float`  # the clock_deg that meshes a gear placed on +X of a clock-0 gear\n"
    "- `toothed_rim(rim_d, teeth, width) -> Part`  # external tooth BAND that FUSES onto a disc rim\n"
    "  (flywheel starter ring, ratchet, pulley): `result = disc + toothed_rim(disc_od, 120, 12)` --\n"
    "  never hand-draw rim teeth and NEVER skip a brief's named ring gear.\n"
    "NEVER hand-draw gear teeth: call spur_gear / ring_gear (correct involute, Z-CENTERED at the\n"
    "origin like Cylinder -- spans -thickness/2..+thickness/2, so `disc + gear` and a boss placed\n"
    "off the +Z face at thickness/2 line up with NO translation).\n"
    "Two gears MESH only at the SAME module, placed EXACTLY module*(z1+z2)/2 apart (the sum\n"
    "of pitch radii); give the partner `clock_deg=mesh_clock_deg(teeth)` (or a half-tooth phase) so a\n"
    "tooth meets a GAP on the line of centers instead of clashing tip-to-tip.\n"
    "For a NON-gear REPEATED unit (impeller/fan blades, crank throws, spokes, fins, lugs): build ONE\n"
    "unit correctly (offset from the axis), then `radial_pattern(unit, count)` around Axis.Z (or\n"
    "`linear_pattern(unit, count, spacing)` for a row) -- it gets the rotation + fusion right instead\n"
    "of hand-placing N. For a PHASED set (crank throws at 0/180/180/0 deg) use fuse=False and\n"
    "rotate each throw yourself.\n"
    "\n"
    "```python\n"
    "# a 60x60x6 plate with 4 M3 clearance holes on a 40mm circle, drilled from the top:\n"
    "plate = rounded_box(60, 60, 6, 4)\n"
    "result = plate - (Pos(0, 0, 3) * bolt_holes(pcd=40, hole_d=clearance_hole_d('M3'), count=4, depth=10))\n"
    "```"
)

_STANDARDS_REFERENCE = (
    "## Standard hardware dimensions (already imported -- NEVER guess a hole / thread / bearing size)\n"
    "These return the EXACT ISO/DIN mm value; feed them into the geometry helpers above. An unknown\n"
    "size RAISES with the valid list, so you cannot pick a fake one. The validator AUDITS your code:\n"
    "if the brief names a standard part (M3, 608, ...) and you do NOT fetch it through its helper, the\n"
    "part is REFUSED as a hand-modeled dimension. Fastener sizes: M2, M2.5, M3,\n"
    "M4, M5, M6, M8, M10, M12, M14, M16.\n"
    "\n"
    "- `clearance_hole_d(size, fit='normal') -> dia`     # ISO 273 bolt through-hole (close|normal|coarse)\n"
    "- `tap_hole_d(size) -> dia`                          # ISO 261 thread-tapping drill\n"
    "- `counterbore_dims(size) -> (dia, depth)`          # ISO 4762 socket-head-cap-screw head recess\n"
    "- `nut_pocket_dims(size) -> (across_flats, across_corners, depth)`  # ISO 4032 captive hex nut\n"
    "- `bearing_seat_dims(desig, fit='press') -> (shaft_bore_d, seat_d, width)`  # ISO 15 bearing (608, 625, 6800...)\n"
    "- `heat_set_hole_d(size) -> dia`                    # PROVISIONAL: heat-set insert pilot hole\n"
    "- `dowel_hole_d(size, fit='press') -> dia`          # ISO 2338 dowel (sizes 2,3,4,5,6,8 mm)\n"
    "- `magnet_pocket_dims(size) -> (dia, depth)`        # PROVISIONAL: disc magnet (5x2, 6x3, 8x3, 10x3)\n"
    "\n"
    "```python\n"
    "# a plate with four M3 clearance holes + a seat for a 608 bearing, all exact dimensions:\n"
    "shaft_d, seat_d, width = bearing_seat_dims('608')\n"
    "plate = rounded_box(60, 60, 6, 4)\n"
    "plate -= Pos(0, 0, 3) * bolt_holes(pcd=44, hole_d=clearance_hole_d('M3'), count=4, depth=10)\n"
    "result = plate - (Pos(0, 0, 3) * bearing_seat_tool(shaft_d, seat_d, width, 10))\n"
    "```"
)

_CATALOG_REFERENCE = (
    "## Standard-hardware CATALOG (the McMaster brain -- SELECT a verified part, never invent one)\n"
    "`from forge_mind.agentic.catalog import ...`. 650+ deterministic entries (100 bearings, 255\n"
    "screws, 89 nuts, 80 washers, 131 spur gears). Pick an id; a bad id RAISES with the closest real\n"
    "ones. Geometry is the exact ISO/DIN solid.\n"
    "\n"
    "- `catalog_bearing('608')`        # bearing solid by designation\n"
    "- `catalog_screw('shcs_m5x20')`   # shcs|button|csk|hexbolt|setscrew _ m<size> x <length>\n"
    "- `catalog_washer('washer_flat_m5_normal')`   # _normal|_narrow|_wide; `catalog_nut('hexnut_m5')`\n"
    "- `catalog_gear('spur_m1_z24_w4_b6')`         # spur_m<module>_z<teeth>_w<width>_b<bore>\n"
    "- `search_catalog('m5 screw')`, `list_catalog_family('bearing')` to discover ids\n"
    "FIT helpers (return a spec with .ok + computed dims, validate compatibility):\n"
    "- `bearing_mount(bearing='608', shaft_d=8, housing_bore_d=22)`\n"
    "- `screw_stack(screw='shcs_m5x20', washer='washer_flat_m5_normal', nut='hexnut_m5')`\n"
    "- `gear_pair(driver='spur_m1_z24_w4_b6', driven='spur_m1_z40_w4_b10')`  # + gear_pair_geometry(...)\n"
    "\n"
    "```python\n"
    "from forge_mind.agentic.catalog import catalog_bearing, catalog_screw, bearing_mount\n"
    "from build123d import Pos\n"
    "assert bearing_mount(bearing='608', shaft_d=8, housing_bore_d=22).ok   # check the fit first\n"
    "result = Pos(0, 0, 20) * catalog_bearing('608')   # place the verified bearing in the assembly\n"
    "```"
)

_RULES = (
    "## Rules that keep the part manufacturable (the validator enforces these)\n"
    "- ONE connected, watertight solid. No detached blobs, no self-intersections.\n"
    "- BUILD EVERY FEATURE THE BRIEF LISTS -- drop none. A repeated feature means the brief's COUNT of\n"
    "  it (four bolt holes = four holes, six fins = six fins); derive the centres from that count with\n"
    "  GridLocations / PolarLocations, never hand-place a subset. Holes, bores, slots, and mounting\n"
    "  features are STRUCTURAL -- build them in the FIRST pass, not as deferred 'detail'. Before you\n"
    "  finish, COUNT the holes / instances you actually cut against the brief; if any is short, add it,\n"
    "  and make each through-hole cut extend PAST both faces so it opens on both sides. A part that\n"
    "  drops a requested hole is REFUSED (feature-completeness), no matter how good the rest looks.\n"
    "- STANDARD HARDWARE: never GUESS a hole / thread / bearing / nut / dowel / insert size. Call the\n"
    "  standards helpers (clearance_hole_d, tap_hole_d, counterbore_dims, nut_pocket_dims,\n"
    "  bearing_seat_dims, ...) for the exact ISO/DIN value and feed it to the geometry helpers -- the\n"
    "  helper returns the verified dimension, so a guessed number is simply the wrong part. PROVISIONAL\n"
    "  sizes (heat-set, magnet) are flagged; prefer a published-standard size where one exists.\n"
    "- REAL FASTENERS / BEARINGS / NUTS / WASHERS / GEARS: SELECT one from the standard-hardware\n"
    "  CATALOG by id (catalog_screw('shcs_m5x20'), catalog_bearing('608'), catalog_gear(...), ...; see\n"
    "  the CATALOG reference) -- the exact verified solid, never invented. For a raw library part not\n"
    "  in the catalog, import it from bd_warehouse (e.g.\n"
    "  `from bd_warehouse.fastener import SocketHeadCapScrew`;\n"
    "  bd_warehouse.bearing.SingleRowDeepGrooveBallBearing, bd_warehouse.thread.IsoThread) for the\n"
    "  exact ISO/DIN solid -- NEVER hand-model a thread helix or a bearing. (GEARS stay ours.)\n"
    "- A HANDLE / PULL / GRIP is a graspable LOOP with an OPENING your fingers wrap THROUGH\n"
    "  (use grip_loop), NOT a solid spike, knob, or lump -- a pull with no aperture is wrong.\n"
    "- Select faces / edges by AXIS, POSITION, NORMAL, or plane-group (faces().sort_by(Axis.Z)[-1],\n"
    "  edges().filter_by(Axis.Z)), NEVER by a raw list index (faces()[3]): a boolean or fillet\n"
    "  renumbers the topology, so a stale index silently grabs the wrong face and the hole lands in\n"
    "  the wrong place or vanishes. Re-select the face right before you cut it.\n"
    "- To ADD a feature (rib, boss, handle, claw, leg) to a body, OVERLAP IT DEEPLY -- start it well\n"
    "  INSIDE the body so it intersects the wall at EVERY height, not flush against the surface. On a\n"
    "  CURVED / revolved body the wall radius changes with height, so a rib at a fixed radius misses\n"
    "  it and floats free; overlap toward the core, or build the feature INTO the revolve/loft profile.\n"
    "- Every wall, rib, and web >= the brief's min wall (FDM is usually 2-3mm).\n"
    "- Fillet/chamfer with radius < half the local thickness, applied AFTER cuts, or the\n"
    "  kernel raises -- catch failures and back off the radius.\n"
    "- Match the brief's reference dimensions; don't silently shrink or inflate the part.\n"
    "- HOLLOW vessel (glass, vase, cup, bottle): revolve/loft the SOLID outer form, THEN shell it\n"
    "  with offset(amount=-wall, openings=<top face>). NEVER hand-draw a two-rail inner+outer\n"
    "  profile -- it self-intersects (TopoDS::Face fails) or collapses to a near-zero-volume sliver.\n"
    "- WAVY / FLUTED / SCALLOPED / PETAL rim: get the undulation by VARYING THE RADIUS within ONE\n"
    "  continuous revolve/loft (see the wavy_vessel idiom), NEVER by subtracting angular cutters or\n"
    "  building separate petals/staves -- that cuts the thin wall into disconnected scallops and the\n"
    "  mesh single-body check fails even though the kernel still reports one solid.\n"
    "- Sketch primitives: Rectangle, Circle, RegularPolygon(radius, side_count), Polyline,\n"
    "  Line, RadiusArc, Spline; close a BuildLine outline with make_face()."
)


def author_skills_block(one_call: bool | None = None) -> str:
    """Assemble the full author skills reference from the execution-verified idioms.

    ``one_call`` selects the FORGE_ONECALL_FEATURES order (ADR-0133 fix 2): the two
    hole idioms swap to their one-call twins. ``None`` reads the flag; explicit
    True/False lets author.py prebuild both variants at import."""
    use_onecall = onecall_enabled() if one_call is None else one_call
    sections = [_HEADER]
    for key, intro in _ONECALL_ORDER if use_onecall else _ORDER:
        sections.append(
            f"## {key.replace('_', ' ')}\n{intro}\n\n```python\n{VERIFIED_IDIOMS[key]}\n```"
        )
    sections.append(_HELPER_REFERENCE)
    sections.append(_STANDARDS_REFERENCE)
    sections.append(_CATALOG_REFERENCE)
    sections.append(_RULES)
    return "\n\n".join(sections)


# ADR-0122 candidate (FORGE_DOCTRINE, default OFF): the PROACTIVE half of the idiom
# library -- construction doctrine loaded BEFORE authoring. Every rule below is validated
# by a measured incident (L62 thin-wall relational class, L66 carve class, L70 sweep
# ordering, L68 selector-binding crash, gate-v2 lid class) or an execution-verified
# idiom. HISTORY: a static "VERIFIED API RULES" block measured net-zero on ships and
# REGRESSED the pulley class (arms J-N, 2026-07-18) -- that is this lever's null
# hypothesis; it ships OFF and must beat it in its own arm.
CONSTRUCTION_DOCTRINE = """CONSTRUCTION DOCTRINE (each rule prevents a measured failure class):
- BUILD ORDER: base solid -> additive features (bosses/ribs/tubes as their OWN solids,
  fused) -> cuts and bores -> shell -> fillets and chamfers LAST.
- JUNCTIONS (tee/Y/collector): fuse every branch's OUTER solid first, subtract ALL inner
  bores in ONE pass at the end -- never hollow tubes first and overlap them.
- SWEPT FEATURES: build the path FIRST (BuildLine), sketch the section on the path-start
  plane, THEN sweep(). A sweep called before its path/section exists always crashes.
- CUTS: extend every cutting tool ~1 mm past the faces it pierces -- a flush tool leaves
  a zero-thickness skin.
- SPEC DIMENSIONS: every user-stated dimension becomes a NAMED variable used verbatim;
  choose the construction (profile+sweep vs block+features) that lets stated dims stay
  literal instead of derived.
- RESULT: bind ONE fused Part to `result` -- never a ShapeList/selector result.
- SERVO HORN HUBS (the #1 documented field failure -- horn/spline stripping,
  2026-07-23 idiom survey): a horn-mounted hub gets a FULL-DEPTH boss around the center
  bore (>= 4 mm engagement), all four horn screws, and >= 1.8 mm web between screw
  holes and the center bore -- torque flows through the screws, never a friction fit.
- SCREW BOSSES (thread pull-out + boss splitting, the #2/#3 field failures): a boss
  receiving a screw is >= 2x the screw diameter across, blends into its wall with a
  fillet or gusset (never a free-standing tube), and leaves >= 2 mm of material below
  a blind hole's floor.
- LOAD-BEARING BRACKETS (bracket flex, the #4 field failure): a bracket carrying a
  servo or limb load gets ribs or flanges in the LOAD plane -- a flat plate flexes;
  surveyed builds converge on 3-4 mm walls at shoulder/hip/knee-class joints.
- CANTILEVER SNAP FINGERS: uniform thickness 1.2-1.8 mm, length >= 6x thickness, a
  small lead-in chamfer on the lip -- a thick stubby finger snaps off instead of
  flexing."""


def doctrine_enabled() -> bool:
    """OPT-IN via FORGE_DOCTRINE=1 (own flag; off = author prompt byte-identical)."""
    import os

    return os.environ.get("FORGE_DOCTRINE") == "1"


def onecall_enabled() -> bool:
    """OPT-IN via FORGE_ONECALL_FEATURES=1 (ADR-0133 fix 2; off = byte-identical)."""
    import os

    return os.environ.get("FORGE_ONECALL_FEATURES") == "1"


# ADR-0133 fix 3: the fillet-first edge-finishing teaching. Prose at the LAST MILE with
# the measured feedback channel (the sandbox edge-finish event log) as the gate behind
# it -- never in place of it. Appended CALL-TIME in author.py (the _doctrine_block
# pattern), so flag-off prompts stay byte-identical and tests can monkeypatch the env.
EDGE_FINISH_ADDENDUM = (
    "## edge finishing (fillet-first, sized to the wall)\n"
    "COSMETIC edge softening on OUTER edges is fillet-first -- references use rounds; "
    "chamfer stays for BORE LEAD-INS (chamfered_bore_entry) and bed-side edges. THE "
    "TRAP: safe_fillet NEVER errors -- if the radius exceeds the local wall it silently "
    "halves toward nothing or returns the part UN-rounded, and no gate sees it. Make "
    "the FIRST attempt succeed:\n"
    "- size: cosmetic fillet r = 0.5-1 mm; NEVER exceed 1/3 of the smallest adjacent "
    "wall/feature thickness.\n"
    "- scope: fillet NAMED edge sets (vertical_edges(part), edges().group_by(Axis.Z)[-1], "
    "filter_by radius) -- never a broad part.edges() over a part dense with holes/pockets "
    "(that is where fillets fail or degrade).\n"
    "- order: fillet LAST, after every boolean.\n"
    "- keep every brief-stated or structural radius EXACT via raw fillet (it raises "
    "loudly); safe_fillet is for cosmetic rounds only."
)


def edge_finish_enabled() -> bool:
    """OPT-IN via FORGE_EDGE_FINISH=1 (ADR-0133 fix 3; off = byte-identical)."""
    import os

    return os.environ.get("FORGE_EDGE_FINISH") == "1"


__all__ = [
    "CONSTRUCTION_DOCTRINE",
    "EDGE_FINISH_ADDENDUM",
    "VERIFIED_IDIOMS",
    "author_skills_block",
    "doctrine_enabled",
    "edge_finish_enabled",
    "onecall_enabled",
]
