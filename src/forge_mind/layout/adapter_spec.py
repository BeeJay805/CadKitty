"""AdapterPlateSpec — the flat NEMA17 -> 2020 adapter LayoutSpec (ADR-0008).

The third LayoutSpec pattern, alongside ``l_bracket_motor_mount`` (bench-001)
and ``electronics_tray`` (bench-002). A flat adapter plate composes two
already-known interfaces on a single element: NEMA 17 motor face on the +Z
side, 2020 extrusion M5 T-nut pair drilled through to the -Z side. ZERO new
interfaces, ZERO new components, ZERO new validator checks — the bench-003
research thesis is that the architecture composes existing facts cheaply.

This module is data only (the spec schema). The deterministic interpreter
lives in ``layout/adapter_interpreter.py``. The fields the LLM has to fill
are deliberately minimal:

- envelope, material, min_wall, max_overhang from the Blueprint;
- **all four geometry fields are OPTIONAL.** ``plate_width_mm``,
  ``plate_depth_mm``, ``motor_center_x_mm``, ``extrusion_center_x_mm``
  are all derivable from the envelope + ``min_wall_mm`` + the shared
  NEMA17 / M5 facts. The interpreter derives the smallest legal values
  when any of them are omitted (the preferred path). Explicit values
  are still accepted for backward compatibility with the hand-authored
  fixtures + tests; the interpreter validates either way.

  ``plate_width_mm`` / ``plate_depth_mm`` were made optional in
  commit a8d8ec6 (ADR-0008 perturbation-experiment fix). ``motor_center_x_mm``
  / ``extrusion_center_x_mm`` are made optional in this commit
  (ADR-0009 Task 1) for the same reason: the evidence-hardening pass
  showed the LLM picks symmetric large offsets 30/30 trials on the
  decontaminated prompt, blowing the envelope every time. Moving the
  choice into the interpreter deletes the failure class.

- the interface naming and the M5 T-nut spacing are now interpreter
  constants ("motor_face", "extrusion_face", 20.0 mm) rather than spec
  fields. Three reasons: (a) the bench-003 prompt never varies them,
  (b) every prior measurement run emitted the constant values verbatim,
  (c) Anthropic's structured-output grammar limit -- ADR-0009 Task 1
  added two optional offset fields and the schema hit the budget; the
  smallest unlock was to drop these three pass-through fields from the
  LLM-facing surface. ``LBracketMotorMountSpec`` keeps its sub-specs
  because the bench-001 prompt does sometimes vary m5_spacing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.layout.spec import Pattern
from forge_mind.types import Vec3


class AdapterPlateSpec(BaseModel):
    """High-level spec for the ``adapter_plate`` pattern (bench-003).

    One member of the LayoutSpec discriminated union (see
    ``layout/layoutspec.py``). The plate thickness is the spec's
    ``min_wall_mm`` (a flat plate has only one wall to think about); the
    in-plane footprint and the two interface offsets along X are explicit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: Literal[Pattern.ADAPTER_PLATE] = Pattern.ADAPTER_PLATE
    name: str = Field(..., min_length=1, max_length=128)
    envelope_mm: Vec3 = Field(..., description="Maximum bounding box the plate must fit (mm).")
    material: str = Field(..., min_length=1)
    min_wall_mm: float = Field(
        ...,
        gt=0,
        description=(
            "Minimum wall thickness anywhere (mm). On a flat adapter plate this is "
            "ALSO the plate thickness: the geometry has no other walls."
        ),
    )
    max_overhang_deg: float = Field(default=45.0, gt=0, le=90)
    plate_width_mm: float | None = Field(
        default=None,
        gt=0,
        description=(
            "In-plane X span of the plate (mm). OPTIONAL: when omitted, the "
            "interpreter derives the tightest legal value from "
            "motor_center_x_mm + extrusion_center_x_mm + min_wall_mm. The LLM "
            "should leave this null unless a Blueprint feature genuinely "
            "constrains the plate footprint."
        ),
    )
    plate_depth_mm: float | None = Field(
        default=None,
        gt=0,
        description=(
            "In-plane Y span of the plate (mm). OPTIONAL: when omitted, the "
            "interpreter derives the tightest legal value from m5_spacing_mm "
            "+ min_wall_mm + the shared NEMA 17 facts. The LLM should leave "
            "this null unless a Blueprint feature genuinely constrains it."
        ),
    )
    motor_center_x_mm: float | None = Field(
        default=None,
        description=(
            "X offset (mm) of the NEMA 17 face centre from the plate origin "
            "(Y=0). OPTIONAL: when omitted, the interpreter derives the smallest "
            "legal offset pair from envelope + min_wall + the shared interface "
            "facts (motor on +X, T-nut on -X by convention). The LLM should "
            "leave this null unless a Blueprint feature genuinely constrains "
            "the motor position. ADR-0009 Task 1."
        ),
    )
    extrusion_center_x_mm: float | None = Field(
        default=None,
        description=(
            "X offset (mm) of the M5 T-nut pair centre from the plate origin "
            "(Y=0). OPTIONAL: same derivation policy as motor_center_x_mm."
        ),
    )
    fan_component_id: str | None = Field(
        default=None,
        description=(
            "If set, the +Z face hosts an axial FAN (by component id, e.g. "
            "'fan_60mm') instead of a NEMA 17 motor. The fan's bolt square + "
            "airflow bore are facts of the chosen fan -- the interpreter derives "
            "all geometry (plate dims, bolt positions, bore) from them; the LLM "
            "picks only the fan id. Leave null for a motor adapter (bench-003)."
        ),
    )
    # print_orientation, nema17, extrusion_2020 are no longer spec fields:
    # adapter_plate always prints +Z up; interface names are always
    # "motor_face" / "extrusion_face"; m5_spacing is always 20 mm. Dropping
    # them from the spec keeps the structured-output schema under Anthropic's
    # grammar limit -- making the two offsets Optional in ADR-0009 Task 1
    # pushed the GeneratedLayout wrapper over the limit; dropping these
    # three pass-through fields is the smallest unlock.


__all__ = ["AdapterPlateSpec"]
