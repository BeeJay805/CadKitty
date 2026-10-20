"""Computer-build hardware facts: cited dimensions of the parts a PC-mod problem interacts with.

The Problem Solver's first real scenario is "mount one more SSD in this case" (ADR-0087 S9). That
needs the engine to KNOW what a 2.5" drive is, what a case fan is — as registry facts, cited and
provisional, exactly like fasteners/bearings/payloads (never a gen-time guess).

Citations (researched 2026-07-05 via web search, the same fail-closed discipline the compiler's
research subsystem will use — ADR-0087 L2; every value here traces to the source below):

  * 2.5" drive — SFF-8201 (2.5" Form Factor Drive Dimensions, SNIA): length 100.20 mm, width
    69.85 mm nominal (70.10 mm max envelope), height classes 7.0 / 9.5 / 15.0 mm, M3x0.5 mounting
    (4 bottom + 4 side holes). The EXACT bottom/side hole COORDINATES live in SFF-8201 Figure 3-3
    and are NOT in the public excerpts — so they are deliberately absent here (``bottom_hole_pitch_mm
    = None``): we cite the envelope + thread + hole count, and never invent the pattern (L2).
  * 3.5" drive — SFF-8301 (3.5" Form Factor Drive Dimensions, SNIA): width 101.60 mm, length
    147.00 mm max, height classes 17.8 / 26.1 / 42.0 mm, 6-32 UNC mounting; the bottom-hole
    longitudinal pitch A13 = 76.20 mm IS cited.
  * Case fans — de-facto standard (Wikipedia "Computer fan"; ARCTIC 120 mm mounting-hole datasheet;
    there is no SFF here): 80->71.5, 92->82.5, 120->105, 140->124.5 mm center-to-center corner-hole
    spacing on an N x N frame, ~25 mm typical thickness, self-tapping ~5 mm-OD fan screws (no formal
    screw standard).

Every entry is ``provisional=True``: real, cited, but NOT physically verified (and, for the drive
hole coordinates, not fully sourced) — so nothing here earns the VERIFIED tier. A physical sample or
the full SFF PDF promotes it via a human git review.
"""

from __future__ import annotations

from pydantic import Field

from forge_mind.dsl.standards.framework import Registry, StandardSpec

_SFF_8201 = "SFF-8201 2.5in Form Factor Drive Dimensions (SNIA); envelope + M3x0.5 mount + hole count cited; exact hole coordinates require SFF-8201 Fig 3-3 (not in public excerpts)"
_SFF_8301 = "SFF-8301 3.5in Form Factor Drive Dimensions (SNIA); envelope + 6-32 UNC mount + A13 bottom-hole pitch 76.20mm cited"
_FAN_SRC = "de-facto PC case-fan standard (Wikipedia 'Computer fan'; ARCTIC 120mm mounting-hole datasheet); no SFF standard; hole spacing widely documented"


class DriveForm(StandardSpec):
    """A storage-drive form factor: the cited envelope a bracket/enclosure must fit + its mounting
    thread. ``bottom_hole_pitch_mm`` is the longitudinal center-to-center pitch of the bottom mount
    holes WHEN it is cited (3.5"); ``None`` means the coordinates are not in a public source and must
    NOT be invented (a mount that bolts to the drive holes then requires the SFF figure or a
    measurement — the compiler surfaces that as an open assumption)."""

    length_mm: float = Field(..., gt=0)
    width_mm: float = Field(..., gt=0, description="Nominal mounting width.")
    max_width_mm: float = Field(
        ..., gt=0, description="Max envelope width (clearance to design to)."
    )
    heights_mm: tuple[float, ...] = Field(..., description="Standard height classes, thin->thick.")
    mount_thread: str = Field(..., description="Mounting screw thread, e.g. 'M3' or '6-32 UNC'.")
    bottom_holes: int = Field(..., ge=0)
    side_holes: int = Field(..., ge=0)
    bottom_hole_pitch_mm: float | None = Field(
        default=None, description="Longitudinal bottom-hole pitch when cited; None = not sourced."
    )


class CaseFan(StandardSpec):
    """A PC case-fan frame: the N x N outer size + the square corner-hole spacing a fan duct or
    shroud must match. Screw is a self-tapping fan screw (no formal standard)."""

    frame_mm: float = Field(..., gt=0, description="Outer width == height of the square frame.")
    hole_spacing_mm: float = Field(
        ..., gt=0, description="Center-to-center of adjacent corner holes."
    )
    thickness_mm: float = Field(..., gt=0, description="Typical frame thickness.")


DRIVE_FORMS: Registry[DriveForm] = Registry(
    "drive_form",
    (
        DriveForm(
            key="2.5in_drive",
            length_mm=100.20,
            width_mm=69.85,
            max_width_mm=70.10,
            heights_mm=(7.0, 9.5, 15.0),
            mount_thread="M3",
            bottom_holes=4,
            side_holes=4,
            bottom_hole_pitch_mm=None,  # SFF-8201 Fig 3-3; not in public excerpts -> never invented
            source=_SFF_8201,
            provisional=True,
        ),
        DriveForm(
            key="3.5in_drive",
            length_mm=147.00,
            width_mm=101.60,
            max_width_mm=101.60,
            heights_mm=(17.8, 26.1, 42.0),
            mount_thread="6-32 UNC",
            bottom_holes=4,
            side_holes=4,
            bottom_hole_pitch_mm=76.20,  # SFF-8301 A13, cited
            source=_SFF_8301,
            provisional=True,
        ),
    ),
)


CASE_FANS: Registry[CaseFan] = Registry(
    "case_fan",
    (
        CaseFan(
            key="fan_80",
            frame_mm=80.0,
            hole_spacing_mm=71.5,
            thickness_mm=25.0,
            source=_FAN_SRC,
            provisional=True,
        ),
        CaseFan(
            key="fan_92",
            frame_mm=92.0,
            hole_spacing_mm=82.5,
            thickness_mm=25.0,
            source=_FAN_SRC,
            provisional=True,
        ),
        CaseFan(
            key="fan_120",
            frame_mm=120.0,
            hole_spacing_mm=105.0,
            thickness_mm=25.0,
            source=_FAN_SRC,
            provisional=True,
        ),
        CaseFan(
            key="fan_140",
            frame_mm=140.0,
            hole_spacing_mm=124.5,
            thickness_mm=25.0,
            source=_FAN_SRC,
            provisional=True,
        ),
    ),
)


def drive_form(key: str) -> DriveForm:
    return DRIVE_FORMS.get(key)


def case_fan(key: str) -> CaseFan:
    return CASE_FANS.get(key)


__all__ = ["CASE_FANS", "DRIVE_FORMS", "CaseFan", "DriveForm", "case_fan", "drive_form"]
