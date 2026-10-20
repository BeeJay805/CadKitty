"""validate_gcode envelope checks, including G91 relative mode (audit: the dormant G91 escape).

Relative moves were NOT bounds-checked -- an out-of-bounds relative toolpath returned ok=True. The
fix accumulates deltas to an absolute position and enforces the envelope in BOTH modes.
"""

from __future__ import annotations

from forge_mind.slicing import MachineProfile, validate_gcode

_PROFILE = MachineProfile(
    name="test", bounds_mm={"x": (0.0, 100.0), "y": (0.0, 100.0), "z": (0.0, 100.0)}
)


def test_relative_move_out_of_envelope_fails() -> None:
    gcode = "\n".join(
        [
            "M104 S200",  # temperature
            "G90",
            "G1 X10 Y10 Z0.2 E1",  # absolute, in-bounds
            "G91",
            "G1 X200 E5",  # relative +200 -> absolute X=210, OUT of the 100 mm envelope
        ]
    )
    result = validate_gcode(gcode, _PROFILE)
    assert not result.ok
    assert any("210" in e and "outside" in e for e in result.errors)


def test_relative_move_inside_envelope_passes() -> None:
    gcode = "\n".join(
        [
            "M104 S200",
            "G90",
            "G1 X10 Y10 Z0.2 E1",
            "G91",
            "G1 X20 E2",  # relative +20 -> absolute X=30, in-bounds
        ]
    )
    assert validate_gcode(gcode, _PROFILE).ok
