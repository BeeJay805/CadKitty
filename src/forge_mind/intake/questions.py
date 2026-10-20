"""The clarifying-question bank.

Each builder returns a typed ``Question``. ``questions_for_part_type`` returns the ordered candidate
set for a part family; the planner then DROPS any a known standard or a prior answer already resolves
and CAPS how many it asks at once. Questions carry a ``priority`` (1 physical fit, 2 manufacturability,
3 design preference) so the planner asks the fit-critical ones first.

Only questions that change geometry / fit / interfaces / fasteners / dimensions / printability / a
user-visible design choice live here. No colours, no vague styling, no dimensions a standard provides.
"""

from __future__ import annotations

from collections.abc import Callable

from forge_mind.intake.schema import Option, Question, QuestionType

# --- Priority 1: physical fit ---------------------------------------------------------------------


def q_motor_size() -> Question:
    return Question(
        id="q_motor_size",
        type=QuestionType.DROPDOWN,
        title="What motor is this for?",
        description="The frame size sets the face dimensions and bolt pattern.",
        required=True,
        options=[
            Option(label="NEMA-17 (42 mm)", value="NEMA-17"),
            Option(label="NEMA-23 (56 mm)", value="NEMA-23"),
            Option(label="Other / not a stepper", value="other"),
        ],
        default_value="NEMA-17",
        affects=["interfaces", "holes", "dimensions"],
        reason="The motor frame size determines the face size, hole pattern, and shaft clearance.",
        priority=1,
    )


def q_mounting_surface() -> Question:
    return Question(
        id="q_mounting_surface",
        type=QuestionType.MULTIPLE_CHOICE,
        title="What should this mount attach to?",
        description="This affects the hole pattern and rail interface.",
        required=True,
        options=[
            Option(label="2020 aluminum extrusion", value="2020_extrusion"),
            Option(label="2040 aluminum extrusion", value="2040_extrusion"),
            Option(label="Flat plate", value="flat_plate"),
            Option(label="Wood surface", value="wood_surface"),
            Option(label="Other", value="other"),
        ],
        default_value="2020_extrusion",
        affects=["interfaces", "holes", "fasteners"],
        reason="The mounting surface determines slot geometry and fastener type.",
        priority=1,
    )


def q_fastener_type() -> Question:
    return Question(
        id="q_fastener_type",
        type=QuestionType.DROPDOWN,
        title="What screw type should it use?",
        required=True,
        options=[
            Option(label="M3", value="M3"),
            Option(label="M4", value="M4"),
            Option(label="M5", value="M5"),
            Option(label="1/4-20", value="quarter_20"),
            Option(label="Use recommended", value="recommended"),
        ],
        default_value="recommended",
        affects=["fasteners", "holes"],
        reason="The fastener size sets clearance-hole diameters and the bolt land.",
        priority=1,
    )


def q_object_being_mounted() -> Question:
    return Question(
        id="q_object_being_mounted",
        type=QuestionType.TEXT,
        title="What are you mounting or holding?",
        description="Name the part or device so the bracket fits it (e.g. 'a 30 mm proximity sensor').",
        required=True,
        placeholder="Describe the object being mounted...",
        affects=["dimensions", "interfaces"],
        reason="The mounted object sets the critical envelope and hole pattern.",
        priority=1,
    )


def q_bearing_size() -> Question:
    return Question(
        id="q_bearing_size",
        type=QuestionType.DROPDOWN,
        title="Which bearing does it hold?",
        required=True,
        options=[
            Option(label="608 (8x22x7)", value="608"),
            Option(label="623 (3x10x4)", value="623"),
            Option(label="625 (5x16x5)", value="625"),
            Option(label="Other", value="other"),
        ],
        default_value="608",
        affects=["dimensions", "interfaces"],
        reason="The bearing model sets the seat bore and depth.",
        priority=1,
    )


def q_fan_size() -> Question:
    return Question(
        id="q_fan_size",
        type=QuestionType.DROPDOWN,
        title="What fan size?",
        required=True,
        options=[
            Option(label="40 mm", value="40mm"),
            Option(label="60 mm", value="60mm"),
            Option(label="Other", value="other"),
        ],
        default_value="40mm",
        affects=["dimensions", "interfaces", "holes"],
        reason="The fan frame size sets the bore and bolt pattern.",
        priority=1,
    )


def q_image_scale() -> Question:
    return Question(
        id="q_image_scale",
        type=QuestionType.NUMBER,
        title="What is the width of the part shown in your drawing?",
        description="I can see the shape, but I need one real measurement to scale it correctly.",
        required=True,
        unit="mm",
        min=1.0,
        max=1000.0,
        default_value=40.0,
        affects=["dimensions"],
        reason="A drawing is shape-only; one real dimension lets me scale it without guessing.",
        priority=1,
    )


def q_image_region(
    image_asset_id: str,
    *,
    title: str = "Point to the feature I should anchor to",
    regions: list[Option] | None = None,
) -> Question:
    """Ask the user to indicate a region on an uploaded image (the frontend overlays it on the
    asset). Named ``regions`` make it a pick-list; otherwise the answer is a coordinate string. Takes
    the asset id, so it is built on demand (not a no-arg bank builder)."""
    return Question(
        id="q_image_region",
        type=QuestionType.IMAGE_REGION_SELECT,
        title=title,
        description="Click the feature on your drawing (e.g. the main mounting hole).",
        required=True,
        options=regions or [],
        image_asset_id=image_asset_id,
        affects=["interfaces", "holes", "dimensions"],
        reason="Pointing at the feature removes ambiguity a drawing alone cannot resolve.",
        priority=1,
    )


# --- Priority 2: manufacturability ----------------------------------------------------------------


def q_plate_thickness() -> Question:
    return Question(
        id="q_plate_thickness",
        type=QuestionType.NUMBER,
        title="How thick should the main plate be?",
        description="Use the default if you are 3D printing in PLA/PETG.",
        required=False,
        unit="mm",
        min=2.0,
        max=12.0,
        step=0.5,
        default_value=5.0,
        affects=["strength", "printability", "geometry"],
        reason="Plate thickness trades strength against print time and material.",
        priority=2,
    )


def q_material_process() -> Question:
    return Question(
        id="q_material_process",
        type=QuestionType.DROPDOWN,
        title="How will you make it?",
        required=False,
        options=[
            Option(label="3D print (PLA / PETG)", value="PLA_or_PETG"),
            Option(label="3D print (ABS / ASA)", value="ABS_or_ASA"),
            Option(label="Resin print", value="resin"),
            Option(label="Other", value="other"),
        ],
        default_value="PLA_or_PETG",
        affects=["printability", "strength"],
        reason="The process sets the min wall thickness and overhang rules.",
        priority=2,
    )


def q_supports_ok() -> Question:
    return Question(
        id="q_supports_ok",
        type=QuestionType.BOOLEAN,
        title="Are print supports acceptable?",
        description="If not, I'll favour geometry that prints without supports.",
        required=False,
        default_value=True,
        affects=["printability", "geometry"],
        reason="Avoiding supports constrains overhangs and orientation.",
        priority=2,
    )


# --- Priority 3: design preference ----------------------------------------------------------------


def q_reinforced() -> Question:
    return Question(
        id="q_reinforced",
        type=QuestionType.BOOLEAN,
        title="Add reinforcement (gussets / ribs) where it helps?",
        required=False,
        default_value=True,
        affects=["strength", "geometry"],
        reason="Reinforcement adds stiffness at the cost of size and material.",
        priority=3,
    )


def q_adjustable_slots() -> Question:
    return Question(
        id="q_adjustable_slots",
        type=QuestionType.BOOLEAN,
        title="Use adjustable slots instead of fixed holes?",
        required=False,
        default_value=True,
        affects=["holes", "interfaces"],
        reason="Slots allow position adjustment; fixed holes are more rigid.",
        priority=3,
    )


def q_special_constraints() -> Question:
    return Question(
        id="q_special_constraints",
        type=QuestionType.TEXT,
        title="Any special constraints?",
        description="Example: must clear a belt, avoid a sensor, fit inside a 60 mm space.",
        required=False,
        placeholder="Describe any clearance, fit, or space constraints...",
        affects=["geometry", "constraints"],
        reason="Clearance / envelope constraints change the outline and cutouts.",
        priority=3,
    )


# Ordered candidate questions per part family. The planner filters out resolved ones and caps the
# count, so listing the full relevant set here is safe.
_PLAN: dict[str, list[Callable[[], Question]]] = {
    "motor_mount": [
        q_motor_size,
        q_mounting_surface,
        q_fastener_type,
        q_plate_thickness,
        q_adjustable_slots,
        q_supports_ok,
        q_special_constraints,
    ],
    "bracket": [
        q_object_being_mounted,
        q_mounting_surface,
        q_fastener_type,
        q_plate_thickness,
        q_reinforced,
        q_special_constraints,
    ],
    "adapter_plate": [
        q_mounting_surface,
        q_fastener_type,
        q_plate_thickness,
        q_special_constraints,
    ],
    "bearing_block": [
        q_bearing_size,
        q_mounting_surface,
        q_fastener_type,
        q_plate_thickness,
        q_special_constraints,
    ],
    "fan_mount": [
        q_fan_size,
        q_mounting_surface,
        q_fastener_type,
        q_special_constraints,
    ],
    "enclosure": [
        q_object_being_mounted,
        q_fastener_type,
        q_material_process,
        q_special_constraints,
    ],
    "standoff": [
        q_fastener_type,
        q_special_constraints,
    ],
    "generic": [
        q_object_being_mounted,
        q_mounting_surface,
        q_fastener_type,
        q_plate_thickness,
        q_special_constraints,
    ],
}

# Shape primitives (a plain geometric solid -- cube, plate, cylinder, ...). A minimal, ALL-OPTIONAL
# set: a primitive needs no mount / fastener questions, and its size comes from a stated measurement or
# the scale calibration rather than a prompt. So a fully-specified request like "a 1 meter cube" has no
# required unknowns and generates immediately, while "a cube" (no size) still adds no fit-questions.
_PRIMITIVE_QUESTIONS: list[Callable[[], Question]] = [q_material_process, q_special_constraints]
for _primitive in ("cube", "plate", "cylinder", "tube", "disc", "ring", "sphere", "cone", "block"):
    _PLAN[_primitive] = _PRIMITIVE_QUESTIONS

# Every builder, by id -- so the agent can look one up when re-asking.
ALL_BUILDERS: dict[str, Callable[[], Question]] = {
    b().id: b for builders in _PLAN.values() for b in builders
}
ALL_BUILDERS[q_image_scale().id] = q_image_scale
ALL_BUILDERS[q_material_process().id] = q_material_process
ALL_BUILDERS[q_reinforced().id] = q_reinforced
ALL_BUILDERS[q_adjustable_slots().id] = q_adjustable_slots


def questions_for_part_type(part_type: str) -> list[Question]:
    """The ordered candidate questions for a part family (defaults to the generic set)."""
    builders = _PLAN.get(part_type, _PLAN["generic"])
    return [b() for b in builders]


def build_question(question_id: str) -> Question | None:
    builder = ALL_BUILDERS.get(question_id)
    return builder() if builder is not None else None


__all__ = [
    "ALL_BUILDERS",
    "build_question",
    "q_image_region",
    "q_image_scale",
    "questions_for_part_type",
]
