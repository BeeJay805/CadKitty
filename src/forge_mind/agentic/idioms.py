"""Verified detail idioms (ADR-0117): tiny build123d snippets EXECUTION-VERIFIED against the
installed kernel (2026-07-19, build123d 0.10.0 -- each produced ONE fused solid) that show the
author the working construction for a feature class it keeps omitting or crashing on. They ride
ONLY on triggered findings (a gate finding or a critique-named absent feature) -- the sanctioned
error-triggered channel, never static prompt rules (arms M/N measured those OUT)."""

from __future__ import annotations

# keyword (searched in feature names / critique fixes, lowercased) -> the verified snippet
# The swept-form trigger vocabulary (ADR-0133 fix 1): shared by the error-triggered
# idiom below and the FORGE_SWEEP_IDIOM plan-time promotion in author.py -- one list,
# never two hand-synced copies.
SWEPT_KEYS: tuple[str, ...] = (
    "hook",
    "curve",
    "curved arm",
    "j-shape",
    "runner",
    "tube",
    "swept",
    "sweeping",
)

# The L70 junction vocabulary -- the ONE idiom allowed to shadow the swept idiom at
# plan time (a manifold brief says "tube" too, and the merged-junction construction is
# the load-bearing correction there). Shared with the _IDIOMS key below.
JUNCTION_KEYS: tuple[str, ...] = (
    "junction",
    "collector",
    "manifold",
    "tee",
    "into-1",
    "merge",
    "y-branch",
)

_IDIOMS: dict[tuple[str, ...], str] = {
    ("boss", "standoff", "post", "pattern"): (
        "VERIFIED PATTERN IDIOM (executed, one fused solid): bosses/standoffs are ADDED "
        "cylinders that must START INSIDE the base so the union fuses --\n"
        "    with Locations(Plane.XY.offset(<base_top>)):\n"
        "        with PolarLocations(radius=<r>, count=<n>):  # or GridLocations(dx, dy, nx, ny)\n"
        "            Cylinder(radius=<boss_r>, height=<2x the protrusion>)"
    ),
    ("groove", "pulley", "v-groove", "rim"): (
        "VERIFIED V-GROOVE IDIOM (executed, one solid): cut the groove by revolving a small "
        "triangle AT THE RIM around the part axis --\n"
        "    with BuildSketch(Plane.XZ):\n"
        "        with Locations((<outer_radius>, 0)):\n"
        "            RegularPolygon(<groove_depth>, 3, rotation=90)\n"
        "    revolve(axis=Axis.Z, mode=Mode.SUBTRACT)"
    ),
    # Junction FIRST: a manifold brief mentions both "collector" and "tube(s)", and the
    # merged-junction construction is the more load-bearing correction there (L70: the
    # swept-tube keywords below must not shadow it -- idiom_pointer returns FIRST match).
    JUNCTION_KEYS: (
        "VERIFIED MERGED-JUNCTION IDIOM (executed, one fused solid, uniform walls): where "
        "tubes MEET (tee / Y / 4-into-1 collector), fuse the OUTER solids of EVERY branch "
        "first, then subtract ALL inner bores in ONE pass at the end. Never hollow each tube "
        "separately and overlap them -- the buried tube shells leave sub-mm knife-edge walls "
        "inside the joint (the measured 0.37mm manifold class) --\n"
        "    with BuildPart() as bp:\n"
        "        Cylinder(radius=12, height=80)                        # main run OUTER solid\n"
        "        with Locations((0, 0, 20)):\n"
        "            Cylinder(radius=12, height=60, rotation=(0, 90, 0))  # branch OUTER, fused\n"
        "        Cylinder(radius=9.5, height=80, mode=Mode.SUBTRACT)   # bores LAST, one pass\n"
        "        with Locations((0, 0, 20)):\n"
        "            Cylinder(radius=9.5, height=60, rotation=(0, 90, 0), mode=Mode.SUBTRACT)"
    ),
    # L70 keywords: runner/tube briefs must reach the swept construction (generation 3
    # crashed 4/6 iterations calling sweep() with no pending path/section).
    SWEPT_KEYS: (
        "VERIFIED SWEPT-TUBE IDIOM (executed, one solid): a curved tube/hook/runner is a "
        "profile SWEPT along a path built FIRST, in this exact order --\n"
        "    with BuildLine() as path:\n"
        "        CenterArc((0, 0), <bend_radius>, start_angle=0, arc_size=<degrees>)  # or Spline(...)\n"
        "    with BuildSketch(Plane(origin=path.line @ 0, z_dir=path.line % 0)):\n"
        "        Circle(<outer_radius>)  # hollow tube: subtract the bore AFTER sweeping\n"
        "    sweep()\n"
        "sweep() called before the path/section exist fails with 'path must be provided' / "
        "'No sections provided'. Or skip the ordering entirely: "
        "`result = safe_sweep(path, radius=<r>)` builds the path-start plane for you and "
        "RAISES clearly instead of shipping a shell."
    ),
}


def idiom_pointer(text: str) -> str | None:
    """The FIRST verified idiom whose keywords appear in ``text`` (lowercased), or None.
    One idiom per trigger keeps the feedback tight."""
    lowered = text.lower()
    for keys, snippet in _IDIOMS.items():
        if any(k in lowered for k in keys):
            return snippet
    return None


def swept_form_idiom(text: str) -> str | None:
    """The swept-form idiom IFF ``text`` names a swept form -- else None (ADR-0133).

    ONLY the L70 junction rule may shadow it (a manifold brief says "tube" too, and the
    merged-junction construction is the load-bearing correction there). Review-caught:
    routing through :func:`idiom_pointer`'s GLOBAL first-match order would let any
    earlier idiom (bosses, V-grooves) hijack a mixed-feature swept brief -- so the
    shadow set is exactly ``JUNCTION_KEYS``, nothing else."""
    lowered = text.lower()
    if not any(k in lowered for k in SWEPT_KEYS):
        return None
    if any(k in lowered for k in JUNCTION_KEYS):
        return _IDIOMS[JUNCTION_KEYS]
    return _IDIOMS[SWEPT_KEYS]


__all__ = ["JUNCTION_KEYS", "SWEPT_KEYS", "idiom_pointer", "swept_form_idiom"]
