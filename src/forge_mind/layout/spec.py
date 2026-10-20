"""LayoutSpec — a high-level, pattern-based part description (ADR-0004).

The bet: for a KNOWN mechanical part family the LLM should not emit low-level
PartDSL geometry (it makes relational layout mistakes the validator catches but
the reviser can't field-patch). Instead it emits a LayoutSpec — pattern + sizes +
interfaces + wall/print rules + constraints — and a DETERMINISTIC interpreter
turns that into PartDSL with the relational invariants (hole-to-edge land >=
min_wall, envelope, interface placement) satisfied by construction.

This module is data only (the LayoutSpec schema). The interpreter lives in
``layout/interpreter.py``. Exactly ONE pattern is supported for the first spike:
``l_bracket_motor_mount`` (the bench-001 part). More patterns are added only when
a benchmark needs them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.types import Vec3


class Pattern(StrEnum):
    """Supported part-family patterns.

    ``l_bracket_motor_mount`` is implemented end to end. ``electronics_tray`` is the
    board-generic open tray -- first specialised for the Raspberry Pi 4 case
    (bench-002), then shown to handle a generic rectangular PCB through the same
    interpreter (ADR-0007 Stage C). Its interpreter is ``layout/pi4_interpreter.py``.

    ``PI4_CASE`` is kept as a backward-compatible alias of ``ELECTRONICS_TRAY``: it is
    the same enum member, and the serialized value stays ``"pi4_case"`` so existing
    fixtures and benchmark records keep loading unchanged (ADR-0007 Stage E).
    """

    L_BRACKET_MOTOR_MOUNT = "l_bracket_motor_mount"
    ELECTRONICS_TRAY = "pi4_case"
    PI4_CASE = "pi4_case"  # backward-compat alias of ELECTRONICS_TRAY (same enum member)
    ADAPTER_PLATE = "adapter_plate"
    # The unified plate family (ADR-0014): flat_plate + l_bracket_bend base bodies with a
    # list of registry face-interfaces. Collapses l_bracket / adapter_plate / fan_mount.
    COMPOSABLE_PLATE = "composable_plate"
    # A computed-geometry body (ADR-0019): a standalone part whose shape a deterministic
    # builder produces from a categorical ComputedOp pick (first op: faceted_strut).
    COMPUTED_BODY = "computed_body"
    # The first ARM: a computed_body strut fused with a NEMA-17 motor mount at the tip
    # (the north-star "mechanical interface + computed shell", a single rigid part).
    MOTOR_ARM = "motor_arm"


# GussetPolicy was removed in ADR-0009 Task 2: the LLM picked `gusset:
# required` for "NEMA 17 stepper motor" 23/30 times on the decontaminated
# bench-001 prompt, trapping the interpreter against the derived (tightest
# legal) base depth. The fix is to remove the policy field entirely; the
# interpreter now always behaves as the AUTO policy used to (try to fit the
# rib, silently skip if it doesn't). No LLM choice means no LLM mistake.


class BaseDims(BaseModel):
    """The horizontal base plate (carries the motor face). Thickness is derived
    from ``min_wall_mm`` by the interpreter, not specified here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    width_mm: float = Field(..., gt=0, description="X span of the base plate (mm).")
    depth_mm: float = Field(..., gt=0, description="Y span of the base plate (mm).")


class UprightDims(BaseModel):
    """The vertical upright plate (carries the extrusion face). Thickness is
    derived from ``min_wall_mm`` by the interpreter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    width_mm: float = Field(..., gt=0, description="X span of the upright plate (mm).")
    height_mm: float = Field(..., gt=0, description="Z span of the upright plate (mm).")


class Nema17InterfaceSpec(BaseModel):
    """Declared NEMA 17 motor face on the base, facing +Z. The 31mm bolt square,
    M3 clearance, and shaft bore are fixed mechanical facts of the pattern.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="motor_face", min_length=1, max_length=64)


class Extrusion2020InterfaceSpec(BaseModel):
    """Declared 2020-extrusion M5 T-nut face on the upright, facing +Y."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="extrusion_face", min_length=1, max_length=64)
    m5_spacing_mm: float = Field(default=20.0, gt=0, description="Spacing between the 2 M5 holes.")


class LBracketMotorMountSpec(BaseModel):
    """High-level spec for the l_bracket_motor_mount pattern (bench-001).

    One member of the LayoutSpec discriminated union (see layout/layoutspec.py).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: Literal[Pattern.L_BRACKET_MOTOR_MOUNT] = Pattern.L_BRACKET_MOTOR_MOUNT
    name: str = Field(..., min_length=1, max_length=128)
    envelope_mm: Vec3 = Field(..., description="Maximum bounding box the part must fit (mm).")
    material: str = Field(..., min_length=1)
    min_wall_mm: float = Field(..., gt=0, description="Minimum wall/land thickness anywhere (mm).")
    max_overhang_deg: float = Field(..., gt=0, le=90)
    base: BaseDims | None = Field(
        default=None,
        description=(
            "Horizontal base plate dims. OPTIONAL: when omitted (the preferred path), "
            "the interpreter derives the tightest legal dims from min_wall_mm + the "
            "shared NEMA 17 facts. Explicit values are still accepted for backward "
            "compatibility with hand-authored fixtures; the interpreter validates "
            "either way. This split (LLM picks intent fields; structure derives "
            "layout fields) mirrors the ADR-0008 bench-003 fix and removes the "
            "off-by-fence-post failure mode the bench-001 perturbation exposed."
        ),
    )
    upright: UprightDims | None = Field(
        default=None,
        description=(
            "Vertical upright plate dims. OPTIONAL: when omitted (the preferred path), "
            "the interpreter derives the tightest legal dims from min_wall_mm + the "
            "M5 spacing + the shared facts. Explicit values are still accepted; the "
            "interpreter validates either way."
        ),
    )
    # print_orientation is no longer a spec field: l_bracket prints base-down
    # (+Z up) by construction and the interpreter assumes it. Dropping it from
    # the spec also shrinks the structured-output schema bench-001's optional
    # dims pushed past Anthropic's grammar limit.
    nema17: Nema17InterfaceSpec = Field(default_factory=Nema17InterfaceSpec)
    extrusion_2020: Extrusion2020InterfaceSpec = Field(default_factory=Extrusion2020InterfaceSpec)
    # gusset removed in ADR-0009 Task 2; interpreter now always uses AUTO
    # semantics (silently skip rib if it can't fit). No LLM-facing field.


__all__ = [
    "BaseDims",
    "Extrusion2020InterfaceSpec",
    "LBracketMotorMountSpec",
    "Nema17InterfaceSpec",
    "Pattern",
    "UprightDims",
]
