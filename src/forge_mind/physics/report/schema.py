"""PhysicsReport DTOs — the FROZEN CONTRACT v1 seam with the frontend (web/src/lib/schema.ts).

These models exist ONLY to serialize the physics-validation report the frontend renders. They are a
read-only PROJECTION assembled by ``compose_physics_report`` over an already-finished validator report;
nothing here runs a check, measures geometry, or gates anything (ADR-0056: the validator is the sole
verdict authority). Every model is ``extra="forbid"`` so a drift from the contract fails loudly in a test
rather than silently shipping a malformed report.

Field names + value vocabularies match ``schema.ts`` exactly so the emitted JSON deserializes into the
committed TypeScript types. The one Python-keyword collision (``from``) is handled with a serialization
alias on ``FixChange.from_`` (dump with ``by_alias=True``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.types import Vec3

# Value vocabularies — must match schema.ts exactly.
PhysicsStatus = Literal["pass", "warn", "fail", "info", "not_run"]
PhysicsSeverity = Literal["blocking", "warning", "advisory"]
PhysicsTier = Literal["verified", "computed", "estimated", "advisory"]
PhysicsVerdict = Literal["pass", "warn", "fail", "partial", "not_run"]
Comparator = Literal["<=", ">=", "=="]
Scope = Literal["single_part", "assembly"]
LoadUnit = Literal["N", "kg", "lbf"]


class PhysicsLocator(BaseModel):
    """WHERE a finding sits, in COORDINATES (ADR-0061) — the same Z-up STL frame the viewer converts
    to draw. A SEPARATE type from ``validator.report.Locator`` (so the shared, live ``/check`` locator is
    untouched) that adds the optional ``part_id`` the assembly contract needs. Present only when a finding
    genuinely computed a locus; never fabricated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame: str = "stl_z_up"
    point: Vec3 | None = None
    bbox_min: Vec3 | None = None
    bbox_max: Vec3 | None = None
    lens: str | None = None
    part_id: str | None = None


class Measured(BaseModel):
    """The measured value of a finding (number or short string), with its unit. ``None`` when the check
    did not produce a value (e.g. a ``not_run`` finding) — never a neutral 0 that reads as 'low'."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: int | float | str
    unit: str


class Limit(BaseModel):
    """The cited limit the measured value is compared against, with the standard/formula behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: int | float | str
    unit: str
    comparator: Comparator
    source: str


class FixChange(BaseModel):
    """A concrete parametric change a fix proposes. ``from`` is a Python keyword, aliased."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    target: str
    from_: int | float | str = Field(alias="from")
    to: int | float | str


class Fix(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    change: FixChange | None = None


class Confidence(BaseModel):
    """Authority of a finding's conclusion (NOT the ADR-0040 dimension-provenance axis — see the ADR).
    ``tier`` is keyed off the METHOD that produced the finding. A ``not_run`` finding never carries
    ``verified`` (it didn't run); ``method`` states why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: PhysicsTier
    method: str


class Finding(BaseModel):
    """One LOCATED + MEASURED + ACTIONABLE finding. Never a bare boolean. ``not_run`` is first-class
    (fail-closed): a check that can't run says why in ``reason_not_run``, never a silent pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    suite: str
    label: str
    status: PhysicsStatus
    severity: PhysicsSeverity
    measured: Measured | None = None
    limit: Limit | None = None
    locator: PhysicsLocator | None = None
    explanation: str
    fix: Fix | None = None
    confidence: Confidence
    reason_not_run: str | None = None


class MassG(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    material: str
    grams: float


class Vitals(BaseModel):
    """Always-computed geometric facts (free tier). A field is ``None`` when not computed (slicer-only
    quantities like print time / filament / cost are ``None`` here), never guessed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    volume_mm3: float | None = None
    mass_g: MassG | None = None
    surface_area_mm2: float | None = None
    center_of_mass_mm: Vec3 | None = None
    est_print_time_min: float | None = None
    filament_g: float | None = None
    est_cost_usd: float | None = None


class Suite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    verdict: PhysicsVerdict
    summary: str
    findings: list[Finding]


class IntentLoad(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    magnitude: float
    unit: LoadUnit
    at: PhysicsLocator | None = None
    direction: Vec3 | None = None


class Mate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    spec: str


class Intent(BaseModel):
    """What the user told us about the part's job. Every field optional; absence routes the dependent
    findings to ``not_run`` (never a fabricated default). Slice 1 ships the empty carrier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job: str | None = None
    load: IntentLoad | None = None
    material: str | None = None
    mates_with: list[Mate] | None = None


MotionKind = Literal["spin", "pair_rotation", "coupled_rotation"]
MotionStatus = Literal["free", "arc_limited", "locked", "refused"]
MotionRole = Literal["part", "rotor", "stator", "roller"]


class MotionAxis(BaseModel):
    """A MEASURED rotation axis in the same ``stl_z_up`` frame as every locator — the viewer
    applies its one existing frame conversion. Present only on claims that earned an axis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame: str = "stl_z_up"
    origin: Vec3
    direction: Vec3


class MotionBody(BaseModel):
    """One solid body of the uploaded part (``body_<i>`` in split order — the GLB bake's
    per-body node naming, slice 2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    role: MotionRole
    volume_mm3: float | None = None


class MotionCoupling(BaseModel):
    """A coupled follower (ADR-0086 T3): it turns ``ratio`` times the driver's angle about
    its OWN stated axis — a bearing's rollers orbit the main axis at the measured epicyclic
    ratio; a meshed gear counter-rotates at -z_driver/z_follower. The ratio is MEASURED,
    with the arithmetic in the claim's evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    body: str
    ratio: float
    axis: MotionAxis


class MotionClaim(BaseModel):
    """One validated motion claim (ADR-0086). The viewer is a PLAYER for these: it may
    animate exactly the axis, bodies, and arc stated here and nothing else. ``locked`` and
    ``refused`` claims carry the measured/explicit reason motion is not presented."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: MotionKind
    status: MotionStatus
    axis: MotionAxis | None = None
    moving: list[str]
    static: list[str] = Field(default_factory=list)
    allowed_arc_deg: tuple[float, float] | None = None
    blocked_at_deg: list[float] | None = None
    # coupled_rotation: every mover beyond the driver, with its measured ratio + axis
    couplings: list[MotionCoupling] | None = None
    evidence: list[Finding]
    explanation: str


class MotionReport(BaseModel):
    """The motion block (schema v2, ADR-0086): read-only, advisory, never a verdict input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bodies: list[MotionBody]
    claims: list[MotionClaim]
    method: str


class OverallConfidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: PhysicsTier
    basis: str


class Overall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: PhysicsVerdict
    blocking: int
    warnings: int
    confidence: OverallConfidence


class PartView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    bbox_mm: Vec3


class PhysicsReport(BaseModel):
    """The top-level report. ``overall.verdict`` is a PURE RELABEL of the validator's aggregate verdict
    (see ``compose``), never re-aggregated from these findings — so this projection can never become a
    second verdict authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 2
    scope: Scope
    parts: list[PartView]
    intent: Intent
    vitals: Vitals
    suites: list[Suite]
    overall: Overall
    disclaimers: list[str]
    # v2 (ADR-0086): validated kinematics. None = nothing animates (the v1 shape plus one
    # optional field — existing consumers deserialize unchanged).
    motion: MotionReport | None = None


__all__ = [
    "Comparator",
    "Confidence",
    "Finding",
    "Fix",
    "FixChange",
    "Intent",
    "IntentLoad",
    "Limit",
    "LoadUnit",
    "MassG",
    "Mate",
    "Measured",
    "MotionAxis",
    "MotionBody",
    "MotionClaim",
    "MotionCoupling",
    "MotionKind",
    "MotionReport",
    "MotionRole",
    "MotionStatus",
    "Overall",
    "OverallConfidence",
    "PartView",
    "PhysicsLocator",
    "PhysicsReport",
    "PhysicsSeverity",
    "PhysicsStatus",
    "PhysicsTier",
    "PhysicsVerdict",
    "Scope",
    "Suite",
    "Vitals",
]
