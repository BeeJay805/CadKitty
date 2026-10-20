"""Hand-coded NEMA 17 motor mount — Phase 0 ground truth.

This is the reference part the PartDSL + Compiler (Phase 1) must reproduce. Any
discrepancy with this STL signals a compiler or DSL design bug.

Design: L-bracket. Motor flange (horizontal) carries the NEMA 17; extrusion
flange (vertical, joined at +Y edge) carries the 2020 extrusion attachment.

Coordinates:
- Origin = center of motor flange, on the print bed (Z=0).
- +Z = up = motor shaft direction post-assembly.
- +Y = rear, toward the extrusion flange.
- +X = lateral; also the extrusion's length direction once attached.

Print orientation: motor flange flat on the bed. Extrusion flange is a single
continuous vertical wall — no overhangs.

Exit criterion (Phase 0): the resulting STL is manifold; has bbox <= 60x60x40mm;
has 4 M3 + 1 shaft hole through the motor flange; has 2 M5 holes through the
extrusion flange.

Run with:
    uv run python examples/handwritten_nema17.py
"""

from __future__ import annotations

from pathlib import Path

from build123d import (  # type: ignore[attr-defined]
    Box,
    Cylinder,
    Pos,
    Rot,
    export_step,
    export_stl,
)

# ----------------------------------------------------------------------------
# Specifications — the answer key
# ----------------------------------------------------------------------------

# NEMA 17 motor face
NEMA17_BOLT_PITCH = 31.0  # mm, center-to-center M3 bolt pattern (square)
NEMA17_M3_CLEARANCE = 3.2  # mm, clearance diameter for M3 bolt
NEMA17_SHAFT_CLEARANCE = 22.0  # mm, central clearance for shaft + raised boss

# 2020 extrusion + M5 T-nut
EXTRUSION_M5_CLEARANCE = 5.2  # mm, clearance diameter for M5 bolt
EXTRUSION_BOLT_SPACING = 20.0  # mm, between T-nuts along slot direction

# Bracket
WALL = 4.0  # mm, all plate thicknesses
# Width 43 (not 42): the M3 holes sit at X=+/-15.5 with a 1.6mm radius, so their
# outer edge is at 17.1mm. A 42mm flange leaves only 3.9mm of land to the edge,
# violating the 4mm-anywhere rule. 43mm gives a 4.4mm web. A real NEMA 17 face
# is 42.3mm, so this is also physically faithful.
MOTOR_FLANGE_W = 43.0  # X span (motor face width)
MOTOR_FLANGE_D = 50.0  # Y span (depth; > 42 so M3 bolt heads clear ext flange)
EXT_FLANGE_W = 43.0  # X span of vertical flange (matches motor flange)
EXT_FLANGE_H = 30.0  # Z span (height); fits 2 M5 holes + margin

# Vertical position of the M5 hole row on the extrusion flange
EXT_HOLE_Z = EXT_FLANGE_H / 2  # centered vertically; one row of T-nuts

# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

# Motor flange: Box centered in X/Y, sitting on the bed (Z=0..WALL).
motor_flange = Pos(0, 0, WALL / 2) * Box(MOTOR_FLANGE_W, MOTOR_FLANGE_D, WALL)

# Extrusion flange: vertical plate flush with the rear edge (+Y) of the motor flange.
# Y center is placed so the flange occupies Y = (rear_edge - WALL) .. rear_edge.
rear_y = MOTOR_FLANGE_D / 2  # rear edge of motor flange (Y=+25)
ext_y_center = rear_y - WALL / 2  # Y=+23
ext_flange = Pos(0, ext_y_center, EXT_FLANGE_H / 2) * Box(EXT_FLANGE_W, WALL, EXT_FLANGE_H)

bracket = motor_flange + ext_flange

# ----------------------------------------------------------------------------
# Holes
# ----------------------------------------------------------------------------

# 4x M3 clearance, NEMA 17 bolt pattern, drilled through Z (motor flange).
half_pitch = NEMA17_BOLT_PITCH / 2
motor_bolt_positions = [
    (-half_pitch, -half_pitch),
    (+half_pitch, -half_pitch),
    (-half_pitch, +half_pitch),
    (+half_pitch, +half_pitch),
]
for hx, hy in motor_bolt_positions:
    bolt_hole = Pos(hx, hy, WALL / 2) * Cylinder(radius=NEMA17_M3_CLEARANCE / 2, height=WALL)
    bracket = bracket - bolt_hole

# Central shaft + boss clearance, also through motor flange.
shaft_hole = Pos(0, 0, WALL / 2) * Cylinder(radius=NEMA17_SHAFT_CLEARANCE / 2, height=WALL)
bracket = bracket - shaft_hole

# 2x M5 clearance through the extrusion flange (axis along Y, spaced along X).
for hx in (-EXTRUSION_BOLT_SPACING / 2, +EXTRUSION_BOLT_SPACING / 2):
    ext_hole = (
        Pos(hx, ext_y_center, EXT_HOLE_Z)
        * Rot(X=90)
        * Cylinder(radius=EXTRUSION_M5_CLEARANCE / 2, height=WALL)
    )
    bracket = bracket - ext_hole


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent
STL_PATH = OUTPUT_DIR / "handwritten_nema17.stl"
STEP_PATH = OUTPUT_DIR / "handwritten_nema17.step"


def main() -> None:
    export_stl(bracket, str(STL_PATH))
    export_step(bracket, str(STEP_PATH))

    bbox = bracket.bounding_box()
    size = bbox.size
    print(f"Wrote {STL_PATH}")
    print(f"Wrote {STEP_PATH}")
    print(f"Bounding box: X={size.X:.2f}  Y={size.Y:.2f}  Z={size.Z:.2f} mm")
    print(f"Volume: {bracket.volume:.2f} mm^3")


if __name__ == "__main__":
    main()
