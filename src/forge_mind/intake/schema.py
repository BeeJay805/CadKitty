"""Typed schemas for the design-intake layer: clarifying questions, answers, the final
``DesignBrief``, and the input assets (uploaded images + ``Draw it!`` mouse sketches).

These are the FRONTEND CONTRACT: the planner returns ``Question`` definitions and the UI renders by
``Question.type`` (no part-specific question types are hardcoded in the UI). The ``DesignBrief`` is
the HANDOFF CONTRACT into the existing CAD pipeline.

Design rules encoded here:
  - every critical dimension records its ``source`` (user / default / standard) so a default is never
    silently passed off as a user-confirmed value;
  - an image/sketch is an intent reference, not exact geometry -- ``scale_confirmed`` defaults False;
  - ``DesignBrief.ready_for_blueprint`` gates generation downstream.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A question default / answer value is one of a few scalar shapes (or a list for multi-select).
ScalarValue = str | float | int | bool
AnswerValue = str | float | int | bool | list[str]


class SessionStatus(StrEnum):
    """Lifecycle of a design session (drives the UI + gates generation)."""

    COLLECTING_INPUT = "collecting_input"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    READY_FOR_BLUEPRINT = "ready_for_blueprint"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"


class QuestionType(StrEnum):
    """The UI control the frontend renders for a question."""

    MULTIPLE_CHOICE = "multiple_choice"  # cards / buttons, one pick
    DROPDOWN = "dropdown"  # select, one pick
    NUMBER = "number"  # number input with unit/min/max/step
    TEXT = "text"  # text / textarea
    BOOLEAN = "boolean"  # yes/no toggle
    CONFIRMATION = "confirmation"  # confirm a fact the planner inferred (e.g. an image reference)
    IMAGE_REGION_SELECT = "image_region_select"  # reserved for later (pick a region on an upload)


class Option(BaseModel):
    """One choice for a multiple_choice / dropdown question."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1, description="Human-facing label.")
    value: str = Field(..., min_length=1, description="Stable machine value stored in the answer.")


class PreviewBehavior(BaseModel):
    """How a dimension question is visualized live in the 3D viewer. Presentation-only: the frontend
    reconstructs a representative feature and morphs it from these fields, so the planner never has to
    describe UI behavior."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="extrusion_thickness | hole_diameter | radius | length | width | depth",
    )
    op_id: str | None = Field(
        default=None, description="Operation this dimension morphs, if known."
    )
    param: str | None = Field(default=None, description="Numeric op parameter the value overrides.")
    axis: str | None = Field(
        default=None, description="Primary axis the dimension acts on (x/y/z)."
    )
    ghost_preview: bool = Field(
        default=True, description="Show a translucent envelope of the range."
    )
    highlight_color: str | None = Field(default=None, description="Hex; defaults to brand accent.")


class Question(BaseModel):
    """A single clarifying question, rendered by ``type``.

    Only the fields relevant to a given ``type`` are populated (``options`` for choice/dropdown;
    ``unit``/``min``/``max``/``step`` for number; ``placeholder`` for text). ``affects`` + ``reason``
    explain WHY it is asked -- the planner must only ask questions that change geometry / fit /
    interfaces / fasteners / dimensions / printability / a user-visible design choice.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ..., min_length=1, max_length=64, description="Stable id, e.g. 'q_fastener_type'."
    )
    type: QuestionType
    title: str = Field(..., min_length=1, description="The question shown to the user.")
    description: str | None = Field(default=None, description="Optional sub-text / guidance.")
    required: bool = Field(default=True)

    # choice / dropdown
    options: list[Option] = Field(default_factory=list)

    # number
    unit: str | None = Field(default=None, description="e.g. 'mm'.")
    min: float | None = None
    max: float | None = None
    step: float | None = None

    # text
    placeholder: str | None = None

    # image_region_select: which uploaded asset the user points at (its regions may be named via
    # `options`, or the frontend returns coordinates as the answer value).
    image_asset_id: str | None = None

    default_value: ScalarValue | None = Field(
        default=None,
        description="A SAFE default; clearly a default until the user confirms (see CriticalDimension.source).",
    )
    affects: list[str] = Field(
        default_factory=list,
        description="What this answer changes, e.g. ['interfaces', 'holes', 'fasteners'].",
    )
    reason: str | None = Field(default=None, description="Why this question matters to the part.")
    # 1 = physical fit, 2 = manufacturability, 3 = design preference. Internal priority the planner
    # asks in order; harmless extra metadata for the frontend.
    priority: int = Field(default=2, ge=1, le=3)

    # --- Build-preview metadata (additive; None on non-geometry questions) -----------------------
    dimension_id: str | None = Field(
        default=None, description="Stable id for the dimension this question sets."
    )
    affected_feature_ids: list[str] = Field(
        default_factory=list, description="Operation/feature ids this dimension controls."
    )
    preview_behavior: PreviewBehavior | None = Field(
        default=None, description="Drives the live 3D dimension preview; None = no preview."
    )

    @model_validator(mode="after")
    def _check_shape(self) -> Question:
        if self.type in (QuestionType.MULTIPLE_CHOICE, QuestionType.DROPDOWN) and not self.options:
            raise ValueError(f"question {self.id}: {self.type} requires non-empty options")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"question {self.id}: min ({self.min}) > max ({self.max})")
        if self.type is QuestionType.IMAGE_REGION_SELECT and not self.image_asset_id:
            raise ValueError(f"question {self.id}: image_region_select requires an image_asset_id")
        return self


class Answer(BaseModel):
    """A user's answer to one question. ``used_default`` records that the user accepted the planner's
    suggested default rather than entering a value (so the brief can mark the dimension's source)."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., min_length=1)
    value: AnswerValue
    used_default: bool = Field(default=False)


# --------------------------------------------------------------------------------------------------
# Input assets: uploaded images/drawings + Draw it! mouse sketches
# --------------------------------------------------------------------------------------------------


class AssetType(StrEnum):
    REFERENCE_IMAGE = "reference_image"
    TECHNICAL_DRAWING = "technical_drawing"
    SCREENSHOT = "screenshot"
    HAND_SKETCH = "hand_sketch"
    MOUSE_SKETCH = "mouse_sketch"
    OTHER = "other"


class Viewpoint(StrEnum):
    """The sketch viewpoint. It tells the planner which TWO axes an in-plane scale calibration applies
    to -- and therefore which axis a single 2D view cannot pin (that one must come from speech/labels).
    ``iso``/``free`` carry no reliable in-plane scale, so dimensions there come from narration."""

    FRONT = "front"  # X (width) + Z (height) in-plane; depth Y unknown
    TOP = "top"  # X (width) + Y (depth) in-plane; height Z unknown
    RIGHT = "right"  # Y (depth) + Z (height) in-plane; width X unknown
    ISO = "iso"  # foreshortened: topology only, no in-plane measurement
    FREE = "free"  # unspecified


class Asset(BaseModel):
    """Metadata for an uploaded image/drawing. The bytes live at ``path`` (or a remote ``url``); this
    record is what the session + planner carry. ``scale_confirmed`` stays False until the user gives a
    real dimension -- pixels are never treated as exact measurements."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    type: AssetType = AssetType.REFERENCE_IMAGE
    filename: str = Field(..., min_length=1)
    content_type: str = Field(default="application/octet-stream")
    size_bytes: int = Field(default=0, ge=0)
    path: str | None = Field(default=None, description="Local/object-store path to the bytes.")
    url: str | None = Field(default=None, description="Remote URL if stored externally.")
    preview_url: str | None = Field(default=None, description="A displayable preview URL.")
    scale_confirmed: bool = Field(
        default=False,
        description="True only once the user supplies a real dimension for the image.",
    )
    notes: str | None = None


class SketchPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    pressure: float | None = Field(default=None, ge=0.0, le=1.0)


class SketchStroke(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    points: list[SketchPoint] = Field(default_factory=list)
    color: str = Field(default="#1C1917")
    width: float = Field(default=3.0, gt=0)
    # The capture surface may type a stroke (pen|line|rect|circle|arrow|text). When present it SEEDS
    # constraint inference (a `rect` gives 4 perpendiculars + 2 equal pairs for free); absent, the
    # detector classifies the point list geometrically. Additive -- legacy strokes carry None.
    type: str | None = Field(default=None)
    label: str | None = Field(
        default=None, description="Optional label the user tied to this stroke."
    )


class SketchAnnotation(BaseModel):
    """A text label the user placed on the sketch (planner reads these as hints, e.g. '60 mm')."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    text: str = Field(..., min_length=1)


class Sketch(BaseModel):
    """The raw ``Draw it!`` payload from the frontend (strokes + annotations on a canvas)."""

    model_config = ConfigDict(extra="forbid")

    canvas_width: int = Field(..., gt=0)
    canvas_height: int = Field(..., gt=0)
    strokes: list[SketchStroke] = Field(default_factory=list)
    annotations: list[SketchAnnotation] = Field(default_factory=list)
    created_at: str | None = None


class SketchAsset(BaseModel):
    """A stored sketch: the raw stroke JSON + a rendered preview (+ optional SVG export). Like an
    image, a sketch is a shape/intention reference -- ``scale_confirmed`` is False until the user
    gives a dimension, so strokes are never converted straight to CAD geometry."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    sketch: Sketch
    preview_path: str | None = None
    preview_url: str | None = None
    svg: str | None = Field(default=None, description="Optional SVG export of the strokes.")
    scale_confirmed: bool = Field(default=False)


# --------------------------------------------------------------------------------------------------
# Multimodal capture context (sketch + speech bundle) -- ADR-0025
# --------------------------------------------------------------------------------------------------


class ScaleRef(BaseModel):
    """A scale calibration: a reference stroke/label is declared to equal a real length ('this line =
    N mm'). Recovers true size for IN-PLANE lengths only (the calibration line is coplanar with what
    it measures); the out-of-plane axis is never recoverable from one view -- it comes from speech."""

    model_config = ConfigDict(extra="forbid")

    mm: float = Field(..., gt=0, description="Real length, in mm, of the reference.")
    ref_label: str | None = Field(default=None, description="Which stroke/label is the reference.")


class CaptureContext(BaseModel):
    """The multimodal bundle that accompanies a sketch/upload (ADR-0025). ``speech_transcript`` is the
    PRIMARY intent channel -- it carries function, rationale, and the numbers a drawing cannot show;
    the sketch supplies form/topology. All fields optional + additive: a text-only or image-only
    request carries none of this, and the planner degrades to exactly today's behavior."""

    model_config = ConfigDict(extra="forbid")

    speech_transcript: str | None = Field(
        default=None, description="Narration captured while drawing; the primary intent source."
    )
    notes: str | None = Field(default=None, description="Freeform note typed alongside the sketch.")
    labels: list[str] = Field(
        default_factory=list, description="On-sketch text labels the user placed (e.g. '60 mm')."
    )
    scale: ScaleRef | None = Field(default=None, description="In-plane scale calibration.")
    viewpoint: Viewpoint | None = Field(
        default=None, description="Sketch viewpoint; fixes which axis a single view cannot pin."
    )


# --------------------------------------------------------------------------------------------------
# Structure-first interpretation: a topology feature list + intentional constraints -- ADR-0025
# --------------------------------------------------------------------------------------------------


class Feature(BaseModel):
    """One topological feature in the structure-first list the planner confirms BEFORE dimensions
    (conditional factorization; Img2CAD ~halved error). Dimensions stay SYMBOLIC here -- names, never
    numbers -- so the model can't invent a value to fill a structural gap. Numbers are bound later
    from speech/labels/scale/defaults."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64, description="Stable id, e.g. 'f_base'.")
    kind: str = Field(
        ...,
        min_length=1,
        description="base_extrude | hole | counterbore | fillet | chamfer | boss | slot | rib | "
        "pocket | shell | gear | revolve (free string; the builder maps it).",
    )
    human_label: str = Field(..., min_length=1, description="Plain-English name shown to the user.")
    relationships: list[str] = Field(
        default_factory=list,
        description="Ties to other features by id, e.g. 'on_face_of:f_base', "
        "'concentric_with:f_boss', 'through_all', 'mirror_of:f_left'.",
    )
    symbolic_dims: list[str] = Field(
        default_factory=list,
        description="Named (NOT numeric) dimensions this feature needs, e.g. 'plate_thickness', "
        "'bore_d'. Bound to numbers downstream.",
    )


class ConstraintKind(StrEnum):
    """Geometric relationships the planner believes are intentional. Priority for removing free
    degrees of freedom (the real accuracy lever): coincident -> h/v -> perp/parallel -> equal ->
    concentric/tangent -> symmetry."""

    SYMMETRY = "symmetry"
    EQUAL = "equal"
    CONCENTRIC = "concentric"
    PERPENDICULAR = "perpendicular"
    PARALLEL = "parallel"
    COINCIDENT = "coincident"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    TANGENT = "tangent"


class ConstrainedState(StrEnum):
    """Advisory degrees-of-freedom verdict for a sketch (AutoConstrain idea, ADR-0025). HEURISTIC and
    REPORTED-only -- a DoF estimate without a real solver can miss redundant/conflicting constraints,
    so it never gates generation; it only flags which primitives still float so the planner can ask or
    the author can pin them. ``OVER`` is reserved for a future solver pass (the heuristic never emits
    it)."""

    UNDER = "under"  # at least one detected primitive has no inferred constraint (floats on edit)
    FULLY = "fully"  # every detected primitive participates in at least one inferred constraint
    OVER = "over"  # redundant / conflicting constraints (needs a real solver to detect)


class InferredConstraint(BaseModel):
    """A geometric relationship the planner believes is INTENTIONAL, so the builder PARAMETERIZES it
    (reuses one variable for two equal edges; shares a center) instead of emitting independent magic
    numbers that break on the first edit. Detected deterministically from strokes (source='sketch')
    or proposed by the vision model (source='vision')."""

    model_config = ConfigDict(extra="forbid")

    kind: ConstraintKind
    refs: list[str] = Field(
        default_factory=list, description="Stroke/feature ids the relation ties together."
    )
    source: str = Field(default="sketch", description="'sketch' (deterministic) | 'vision'.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# --------------------------------------------------------------------------------------------------
# Final DesignBrief: the structured handoff into the existing CAD pipeline
# --------------------------------------------------------------------------------------------------


class DimensionSource(StrEnum):
    """Where a critical dimension came from -- so a default is never read as user-confirmed."""

    USER = "user"
    DEFAULT = "default"
    STANDARD = "standard"
    USER_OR_DEFAULT = "user_or_default"
    IMAGE = "image"
    UNKNOWN = "unknown"


class KnownStandard(BaseModel):
    """A hardware standard the part must conform to, cited (never invented). ``values`` are the public
    facts the brief relies on (e.g. NEMA-17 face 42.3 mm, 31 mm hole pattern, M3 screws)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    values: dict[str, ScalarValue] = Field(default_factory=dict)
    source: str | None = Field(default=None, description="Citation for the values.")


class PlannerState(BaseModel):
    """The planner's accumulated understanding across rounds (persisted on the session). Lets a second
    ``run`` build on the first instead of re-deriving from scratch, and makes the agent's reasoning
    inspectable. It is working memory, not the final contract -- that is ``DesignBrief``."""

    model_config = ConfigDict(extra="forbid")

    classified: bool = Field(default=False)
    part_type: str | None = None
    user_goal: str | None = None
    known_standards: list[KnownStandard] = Field(default_factory=list)
    detected_interfaces: list[str] = Field(default_factory=list)
    detected_fasteners: list[str] = Field(default_factory=list)
    missing_critical: list[str] = Field(
        default_factory=list, description="Critical facts still missing (each maps to a question)."
    )
    notes: list[str] = Field(default_factory=list)
    rounds: int = Field(default=0, ge=0, description="How many planner passes have run.")
    features: list[Feature] = Field(
        default_factory=list,
        description="Structure-first feature list (topology before dimensions).",
    )
    inferred_constraints: list[InferredConstraint] = Field(
        default_factory=list, description="Intentional geometric relationships to preserve."
    )


class RequiredInterface(BaseModel):
    """An interface the part must provide (a mating face / mount / slot)."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., min_length=1, description="e.g. 'NEMA17_face', '2020_extrusion_mount'.")
    fasteners: str | None = Field(default=None, description="e.g. 'M3', 'M5'.")
    clearance: str | None = None
    adjustability: str | None = Field(default=None, description="e.g. 'slotted', 'fixed'.")
    notes: str | None = None


class CriticalDimension(BaseModel):
    """One dimension that affects fit/strength/printability, with its provenance."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    value: float
    unit: str = Field(default="mm")
    source: DimensionSource = DimensionSource.UNKNOWN


class DesignBrief(BaseModel):
    """The structured output the planner produces once enough is known. The existing
    blueprint/PartDSL/CAD/validator pipeline consumes this (via ``brief.brief_to_prompt``).

    ``ready_for_blueprint`` is the generation gate: it must be True (and no required question left
    unanswered, no UNKNOWN-source critical dimension) before generation may start.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1)
    part_name: str = Field(..., min_length=1)
    user_goal: str = Field(..., min_length=1)
    part_type: str = Field(..., min_length=1, description="e.g. 'motor_mount', 'bracket'.")
    manufacturing_process: str = Field(default="fdm_3d_printing")
    material_assumption: str = Field(default="PLA_or_PETG")
    known_standards: list[KnownStandard] = Field(default_factory=list)
    required_interfaces: list[RequiredInterface] = Field(default_factory=list)
    critical_dimensions: list[CriticalDimension] = Field(default_factory=list)
    design_choices: dict[str, ScalarValue] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    # Structure-first interpretation carried into the build prompt (ADR-0025). Both default empty, so
    # a legacy brief renders exactly as before.
    features: list[Feature] = Field(default_factory=list)
    geometry_constraints: list[str] = Field(
        default_factory=list,
        description="Rendered geometric-intent lines, e.g. 'concentric: bore, boss'.",
    )
    intent_narration: str | None = Field(
        default=None,
        description="The maker's spoken narration (primary intent), carried into the build prompt.",
    )
    unknowns: list[str] = Field(
        default_factory=list, description="Anything still unresolved (must be empty to be ready)."
    )
    ready_for_blueprint: bool = Field(default=False)


__all__ = [
    "Answer",
    "AnswerValue",
    "Asset",
    "AssetType",
    "CaptureContext",
    "ConstrainedState",
    "ConstraintKind",
    "CriticalDimension",
    "DesignBrief",
    "DimensionSource",
    "Feature",
    "InferredConstraint",
    "KnownStandard",
    "Option",
    "PlannerState",
    "PreviewBehavior",
    "Question",
    "QuestionType",
    "RequiredInterface",
    "ScalarValue",
    "ScaleRef",
    "SessionStatus",
    "Sketch",
    "SketchAnnotation",
    "SketchAsset",
    "SketchPoint",
    "SketchStroke",
    "Viewpoint",
]
