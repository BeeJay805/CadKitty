"""Targeted fallback hints for OCCT kernel failures (lever b's guard half).

OCCT raises are opaque ("StdFail_NotDone") and mildly nondeterministic around fillets and
booleans -- the author sees a wall of traceback and often deletes the whole feature to
dodge it. This module pattern-matches the sandbox's error + traceback to the FAILING OP
and appends the specific, ordered fallback (smaller radius -> narrower selection -> skip
the cosmetic op) so a kernel raise is never a dead end and never answered by deleting a
required feature.

Deterministic string matching only -- no LLM, no guessing beyond the named signature. An
unrecognized failure gets no hint (the generic repair feedback still stands)."""

from __future__ import annotations

# (signature substrings ANY of which must appear in the combined text, hint). Ordered:
# the FIRST matching entry wins, so the more specific op names come before generic ones.
_HINTS: list[tuple[tuple[str, ...], str]] = [
    (
        ("fillet", "chamfer"),
        "the FILLET/CHAMFER failed in the kernel. The radius/length almost certainly exceeds "
        "the local wall or edge thickness, or the edge set changed after a boolean. Fallback, "
        "in order: (1) HALVE the radius; (2) NARROW the edge selection (filter_by / group_by "
        "the exact edges) and apply it LAST, after all cuts; (3) if it still raises, SKIP this "
        "fillet/chamfer entirely -- ship the part without the cosmetic rounding. NEVER delete "
        "the underlying feature to dodge a fillet error.",
    ),
    (
        ("BOPAlgo", "BRepAlgoAPI", "boolean"),
        "a BOOLEAN (union/cut/intersect) failed. The usual causes are exactly-coincident or "
        "tangent faces between the tool and the body, or a zero-thickness sliver where they "
        "barely touch. Fallback: OVERLAP the tool DEEPER (extend it well past the faces it "
        "cuts/joins), or shift it ~0.01mm so no two faces are exactly coplanar, then retry.",
    ),
    (
        # Verified 2026-07-18 vs build123d 0.10.0 on the recorded cable-clip failure: this
        # exact error survived kind=Kind.INTERSECTION unchanged (the message's own suggestion
        # is a dead end there), and the generic shell fallback below never rescued it across
        # 6 identical iterations -- the working escape is to not shell at all.
        ("an alternative kind may resolve",),
        "offset()/shell CANNOT hollow this body -- and switching `kind=` will NOT fix it "
        "(verified). STOP shelling: delete the offset() call and build the thin wall "
        "DIRECTLY instead. A ring/tube/C-profile is ONE sketch -- Circle(outer); "
        "Circle(inner, mode=Mode.SUBTRACT); cut the opening with a Rectangle(mode="
        "Mode.SUBTRACT) -- then extrude(). Every other feature stays.",
    ),
    (
        ("BRepOffset", "offset", "shell"),
        "the SHELL/OFFSET failed. The wall is too thick for the local curvature, or the "
        "opening-face selection is wrong. Fallback: reduce the wall thickness, fillet sharp "
        "internal corners BEFORE shelling, or pick a different opening face.",
    ),
    (
        ("MakeFace", "not closed", "make_face"),
        "the sketch PROFILE did not close into a face. Every BuildLine outline must end where "
        "it started (close the loop exactly; do not repeat the first point of a periodic "
        "Spline) before make_face().",
    ),
    (
        ("sweep", "pipe"),
        "the SWEEP failed. The profile is too large for the path's curvature (it self-"
        "intersects around bends) or the profile plane is not at the path start. Fallback: "
        "shrink the profile, increase the bend/helix radius, sketch the profile on "
        "Plane(origin=path @ 0, z_dir=path % 0), and pass is_frenet=True for twisting paths.",
    ),
    (
        ("Standard_NullObject", "NullObject", "list index out of range"),
        "a SELECTOR matched nothing (or used a stale index): after any boolean/fillet the "
        "topology renumbers. Re-select faces/edges by AXIS or position (faces().sort_by("
        "Axis.Z)[-1]) immediately before each use; never reuse a face variable across a cut.",
    ),
]

# Resource blow-ups arrive as the sandbox's own error_type, not an OCCT signature.
_RESOURCE_HINTS = {
    "timeout": (
        "the build TIMED OUT -- a tessellation or boolean blow-up, not a syntax error. "
        "Simplify the heavy op: fewer loft sections, fewer spline points (<=64 per ring), "
        "fewer pattern instances, smaller fillet scope; avoid intersecting dozens of solids "
        "in one boolean."
    ),
    "memory": (
        "the build exceeded the MEMORY cap -- runaway tessellation or a degenerate boolean. "
        "Same fallback as a timeout: simplify the heavy op (fewer sections / points / "
        "instances), and check no two faces are exactly coincident in a big union."
    ),
    "kernel_crash": (
        "the geometry kernel CRASHED natively (a fault, not a Python error) -- almost always a "
        "chamfer/fillet applied over too many or too-complex edges (a blanket 'break/chamfer ALL "
        "edges' after unioning several solids), or a boolean on coincident/degenerate faces. FIX: "
        "REMOVE any global edge-break; chamfer or fillet ONLY named, simple edges, and do it AFTER "
        "the final boolean, never over a fresh multi-body union. If many overlapping boxes precede "
        "it, make them OVERLAP cleanly (not merely touch) so the fuse is robust, and drop the "
        "global fillet entirely on this attempt. Rebuild the shape more simply: fewer overlapping "
        "solids, no whole-part edge operations."
    ),
}


# API-misuse hints (2026-07-18): the "a plain Python error is self-explanatory" assumption above
# was MEASURED FALSE for local models -- the e2e A/B showed the 30B builder re-authoring the SAME
# hallucinated API unchanged for up to 4 iterations (revolve(angle=), fillet_radius=, Arc,
# imports from forge_mind.*). Each rule below names ONE measured hallucination and the VERIFIED
# build123d 0.10.0 replacement (signatures introspected live, not recalled). First match wins;
# unrecognized errors still get no hint.
_API_HINTS: list[tuple[tuple[str, ...], str]] = [
    (
        # Measured (2026-07-25 impeller smoke S1, iter 1): the author hand-rolled a curved
        # blade as Spline -> offset(side=Side.BOTH) inside BuildSketch and offset threw --
        # then the repair hand-rolled it AGAIN as a loft and died too. The verified helpers
        # exist precisely so a curved-form part never rides this fragile construction.
        # ALL-match with the raising FRAME TERMINATOR "in offset\n": FIVE build123d ops share
        # this exact error text (bounding_box, mirror, offset, scale, split --
        # operations_generic.py, verified in the installed kernel), and only the offset frame
        # earns this rewrite. Two near-misses this discriminator survived review on: the
        # authored SOURCE LINE is NOT in the traceback (the sandbox compiles <authored_part>
        # without linecache) so "offset(" never appears, and bare "in offset" would also match
        # a user helper named e.g. offset_blade -- the trailing newline is what pins the exact
        # frame. Verified on the REAL S1 traceback, not assumed.
        ("objects must be provided", "in offset\n"),
        "offset() found nothing to offset -- inside BuildSketch, offset works on a closed "
        "pending FACE, not on a bare BuildLine curve. Do NOT rebuild the curved wall by hand: "
        "for an impeller/fan/blower use "
        "`from forge_mind.agentic.flow_forms import backward_curved_impeller` "
        "(disc + hub + N genuinely curved blades, ONE watertight solid, e.g. "
        "`result = backward_curved_impeller(35, 4, 12, 8, hub_r=10, hub_h=14, bore=5)`); "
        "a crank web is `crank_web(...)`, a finned barrel is `tapered_fin_stack(...)` -- all "
        "verified on this kernel.",
    ),
    (
        # Measured (breadth sweeps 2026-07-18/19): spur_gear runs invented catalog size
        # variants ('spur_m2.5_z24_w6_b5') three times, terminal once -- the error already
        # carries the real ids in its Did-you-mean list.
        ("unknown catalog part",),
        "that catalog id does not exist -- catalog parts are a FIXED enumeration; NEVER invent "
        "a size variant. Pick EXACTLY one id from the 'Did you mean' list in the error message "
        "(copy it verbatim, including the separators), or if none fits the brief's numbers, "
        "model the feature directly instead of using the catalog.",
    ),
    (
        ("unexpected keyword argument 'angle'",),
        "there is NO `angle` keyword. build123d's revolve signature is "
        "revolve(profiles, axis=Axis.Z, revolution_arc=<degrees>) -- use `revolution_arc=` for "
        "a partial revolve, and pass the axis as `axis=Axis.Z` (or another Axis).",
    ),
    (
        ("fillet_radius", "fillet() missing 1 required positional argument"),
        "fillet takes (edges, radius): `fillet(<edge list>, radius=<r>)` -- there is no "
        "`fillet_radius` keyword. Select edges FIRST, e.g. "
        "`fillet(part.edges().filter_by(Axis.Z), radius=2)`.",
    ),
    (
        ("3D fillet operation takes only Edges",),
        "3D fillet accepts EDGES only -- never a face, solid, or the whole part. Select them: "
        "`fillet(part.edges().group_by(Axis.Z)[-1], radius=<r>)`, and fillet LAST, after all "
        "booleans.",
    ),
    (
        ("name 'Arc' is not defined", "name 'ArcByCenter' is not defined"),
        "`Arc` / `ArcByCenter` do not exist in build123d. The arc classes (inside BuildLine) "
        "are CenterArc(center, radius, start_angle, arc_size), RadiusArc(p1, p2, radius), "
        "ThreePointArc(p1, p2, p3), TangentArc.",
    ),
    # This entry requires BOTH substrings (see _API_ALL_MATCH) -- "cannot import name" alone is
    # too generic to pin the measured forge_mind-helpers hallucination. CORRECTED 2026-07-18
    # (second autopsy): the standards helpers DO exist -- the sandbox star-imports them as bare
    # names; only the invented import PATH fails. The first version of this hint said "no such
    # helpers exist", contradicting the cheatsheet in the same prompt.
    (
        ("cannot import name", "forge_mind"),
        "the standards helpers (clearance_hole_d, tap_hole_d, counterbore_dims, ...) are ALREADY "
        "available as bare names in this environment -- DELETE the import line and call the "
        "function directly. The only import you write is `from build123d import *` (plus math).",
    ),
    (
        ("Polygon.__init__() got an unexpected keyword argument",),
        "Polygon takes only the points (plus rotation/align/mode): `Polygon((0,0), (10,0), "
        "(10,5))` -- it closes itself; there is no `close=` keyword.",
    ),
    # Measured autopsy (e2e_C cable clip): this message comes from INSIDE sweep() when the code
    # passes `path=Axis.Z` -- sweep calls .edges() on the path. The earlier hint blamed edge
    # SELECTION and misdirected the repair for 4 straight iterations; name the sweep case FIRST.
    (
        ("'Axis' object has no attribute",),
        "an Axis was passed where a CURVE is required -- most often `sweep(path=Axis.Z)`. "
        "sweep's path must be an Edge/Wire you drew: keep the path in a variable "
        "(`with BuildLine() as p: Line((0,0,0), (0,0,h))`) and call `sweep(path=p.wire())` -- "
        "or draw a profile and `extrude(amount=h)` instead, which is simpler and equivalent for "
        "a straight path. (If you meant edge SELECTION: Axis has no edges -- use "
        "`part.edges().filter_by(Axis.Z)`.)",
    ),
    # --- SWEEP-frame hints (ADR-0133 fix 1; L100-style archive evidence: 151 archived
    # sweep-frame failures, 112 of which previously got the MISDIRECTING generic hints
    # below -- 103 shell-rejects steered toward "extrude a sketch", 8 null-shapes blamed
    # on revolve. ALL-match on the raising frame terminator "in sweep\n" (the verified
    # "in offset\n" precedent above); the generic entries stay for non-sweep cases.
    # Each discriminator verified on a REAL archived traceback (hook_coat_double_s0
    # iters 01/04/02, benchmarks/2026-07-20_harvest/H1). First-match-wins ordering:
    # these MUST precede the generic BuildPart/Null entries.
    (
        ("BuildPart doesn't accept", "in sweep\n"),
        "sweep() produced a SHELL, not a solid -- the swept SECTION was not ONE closed "
        "planar face. Close the profile outline (end exactly where it started) BEFORE "
        "make_face(); a hollow tube sweeps the OUTER Circle and subtracts the bore after "
        "sweeping, never a hand-drawn open outline. Easiest: "
        "`result = safe_sweep(<path>, radius=<r>)` -- it builds the path-start plane for "
        "you and RAISES clearly instead of emitting a shell.",
    ),
    (
        ("Null TopoDS_Shape", "in sweep\n"),
        "the SWEEP produced EMPTY geometry -- the section self-intersects around the "
        "path's bend (profile too large for the bend radius) or sits off the path start. "
        "Shrink the profile or increase the bend radius; sketch the section on "
        "`Plane(origin=path @ 0, z_dir=path % 0)`, or call "
        "`safe_sweep(<path>, radius=<r>)` which does the plane and check for you. "
        "This is NOT the revolve case.",
    ),
    (
        ("Edges are disconnected", "in sweep\n"),
        "the sweep PATH's segments do not CHAIN -- every Line/Arc must start exactly "
        "where the previous ended: `l2 = Line(l1 @ 1, ...)`, "
        "`JernArc(start=l1 @ 1, tangent=l1 % 1, ...)`. One continuous run, no gaps or "
        "branches; then the section, then sweep().",
    ),
    (
        ("BuildPart doesn't accept",),
        "only 3D objects (and operations that yield them) belong inside BuildPart. A 2D "
        "face/sketch must be extruded: sketch it in a BuildSketch block, then "
        "`extrude(amount=<h>)`; for a thin shell, `thicken` a face -- never add a bare "
        "Shell/Face to BuildPart.",
    ),
    # Verified empirically (2026-07-18): a Plane.XY sketch revolved about Axis.Z yields a
    # ZERO-volume part (vol 0.0, no raise at revolve time) that surfaces later as Null TopoDS;
    # the XZ half-profile revolves correctly. The pulley ran this mistake 4 iterations straight.
    (
        ("Null TopoDS_Shape",),
        "the geometry is EMPTY -- if a revolve precedes this, its profile almost certainly sits "
        "in the rotation plane (a Plane.XY sketch revolved about Axis.Z sweeps to ZERO volume). "
        "A revolve profile must be a CLOSED cross-section drawn on a plane CONTAINING the axis "
        "(Plane.XZ for Axis.Z), entirely on ONE side of it: BuildLine + make_face on Plane.XZ "
        "with x >= the bore radius, then revolve(axis=Axis.Z). For a plain disc or cylinder, "
        "skip revolve entirely -- draw the circles on Plane.XY and extrude(amount=thickness).",
    ),
    (
        ("unsupported operand type(s) for @",),
        "`@` (position-at) works on a CURVE object (a Line/Arc/Wire/Edge), not on the "
        "BuildLine builder: keep the curve in a variable (`seg = Line(a, b)`) and use "
        "`seg @ 0.5`, or use `<builder>.line @ 0.5`.",
    ),
    (
        ("'ShapeList' object has no attribute 'wrapped'",),
        "the exporter needs ONE solid but `result` is bound to a ShapeList (a selector "
        "result like `.solids()` or `.faces()`). Bind the PART itself: `result = bp.part` "
        "in builder mode, or fuse explicit solids (`result = a + b`); to take one solid "
        "from a list, index it (`result = bp.solids()[0]`) -- never export the list.",
    ),
    # The L70 sweep-crash pair: 4/6 manifold generation-3 iterations died on these two
    # signatures while attempting the spec'd swept tubes.
    (
        ("path must be provided",),
        "sweep() found no PATH. In builder mode the path must be a PENDING line built "
        "FIRST, inside the same BuildPart: `with BuildLine() as p: Spline((0,0,0), "
        "(20,0,40), (0,0,80))` THEN `with BuildSketch(Plane(origin=p.line @ 0, z_dir="
        "p.line % 0)): Circle(r)` THEN `sweep()`. In algebra mode pass it explicitly: "
        "`sweep(section, path=path_edge)`. Never call sweep() before the path exists.",
    ),
    (
        ("No sections provided",),
        "sweep()/loft() found no SECTION sketch. Build the profile BEFORE the call: "
        "`with BuildSketch(<plane at the path start>): Circle(r)` then sweep(); for "
        "loft(), build one BuildSketch per station along the axis, then loft(). A sketch "
        "created after the call, or outside the active BuildPart, is invisible to it. "
        "AND if this loft/sweep is a curved BLADE (impeller / fan / blower wall), stop "
        "hand-rolling it -- measured 2026-07-25: the hand construction died twice while "
        "`from forge_mind.agentic.flow_forms import backward_curved_impeller` builds the "
        "whole wheel, curved blades included, as ONE verified watertight solid.",
    ),
]


# Signature tuples whose entry matches only when ALL substrings appear (default is ANY).
_API_ALL_MATCH = {
    ("cannot import name", "forge_mind"),
    ("objects must be provided", "in offset\n"),
    ("BuildPart doesn't accept", "in sweep\n"),
    ("Null TopoDS_Shape", "in sweep\n"),
    ("Edges are disconnected", "in sweep\n"),
}


def api_misuse_hint(message: str | None, traceback_text: str | None) -> str | None:
    """The verified-API correction for a measured hallucination signature, or None."""
    blob = f"{message or ''}\n{traceback_text or ''}"
    if not blob.strip():
        return None
    lowered = blob.lower()
    for signatures, hint in _API_HINTS:
        combine = all if signatures in _API_ALL_MATCH else any
        if combine(sig.lower() in lowered for sig in signatures):
            return hint
    return None


def kernel_failure_hint(
    error_type: str | None, message: str | None, traceback_text: str | None
) -> str | None:
    """The targeted fallback for a sandbox failure, or None when no signature matches."""
    if error_type in _RESOURCE_HINTS:
        return _RESOURCE_HINTS[error_type]
    blob = f"{message or ''}\n{traceback_text or ''}"
    if not blob.strip():
        return None
    lowered = blob.lower()
    # Only OCCT-/kernel-shaped failures get an op hint; a plain Python error (NameError,
    # TypeError) is already self-explanatory in the traceback.
    kernelish = (
        any(s in blob for s in ("StdFail", "Standard_", "BRep", "BOPAlgo", "OCP.", "TopoDS"))
        or "not closed" in lowered
    )
    if not kernelish:
        return None
    for signatures, hint in _HINTS:
        if any(sig.lower() in lowered for sig in signatures):
            return hint
    return None


__all__ = ["api_misuse_hint", "kernel_failure_hint"]
