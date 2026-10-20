"""Response + request schema for POST /generate.

These models mirror the frontend's `web/src/lib/schema.ts` EXACTLY (field names + shapes), so the
JSON the browser receives drops straight into the typed client. They are defined standalone here
(NOT imported from the retired `forge_mind.kit`) so the HTTP layer stays decoupled from whatever the
engine does or doesn't produce.

Three response shapes, all returned HTTP 200 and discriminated by `status`:
  - ok      : a generated part (may still be a validation FAIL -- surfaced honestly in `validation`).
  - refused : the request couldn't be grounded into a buildable part. No part.
  - error   : the pipeline failed before a part existed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TierStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tier: str
    status: str  # pass | warn | fail


class CheckView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    status: str
    message: str


class Vec3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    z: float


class StepProofView(BaseModel):
    """STEP-export integrity proof (REPORTED, never gates): re-importing the exported STEP confirms a
    CAD tool reads back the SAME solid -- volume, area, and centroid match the source body within
    tolerance. ``matches`` is the headline; the rel-errors quantify the agreement (ADR-0024).
    The named-B-rep fields (ADR-0037 rung 2, additive) prove the file is real analytic CAD --
    AP schema, B-rep entities, a product named after the part, topology roundtrip -- not a
    volume-matching tessellated lump."""

    model_config = ConfigDict(extra="forbid")
    matches: bool
    volume_mm3: float
    area_mm2: float
    centroid: Vec3
    volume_rel_err: float
    area_rel_err: float
    centroid_dist_mm: float
    schema_id: str | None = None
    is_brep: bool | None = None
    tessellated: bool | None = None
    product_name: str | None = None
    named: bool | None = None
    solid_count: int | None = None
    face_count: int | None = None
    topology_matches: bool | None = None


class ValidationView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    overall: str
    tiers: list[TierStatus] = Field(default_factory=list)
    failed_checks: list[CheckView] = Field(default_factory=list)
    # STEP roundtrip faithfulness proof when a STEP was exported + re-read (ADR-0024); else null.
    step_proof: StepProofView | None = None


class BomLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    component: str
    kind: str  # fastener | tnut | insert | nut | bearing | dowel | magnet | ...
    key: str
    quantity: int
    source: str
    provisional: bool = False
    length_mm: float | None = None
    length_note: str | None = None


class BillOfMaterials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lines: list[BomLine] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list)


class MakeInstructions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    print_orientation: Vec3
    orientation_note: str
    material: str
    recommended_min_wall_mm: float | None = None
    layer_height_mm: float
    wall_note: str
    infill_note: str
    insert_install_note: str | None = None
    steps: list[str] = Field(default_factory=list)
    advisory: bool = True
    caveat: str


class Kit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    part_name: str
    bom: BillOfMaterials
    instructions: MakeInstructions


class Files(BaseModel):
    model_config = ConfigDict(extra="forbid")
    glb: str  # relative path; the frontend prefixes it with NEXT_PUBLIC_API_BASE
    stl: str


class AssemblyPartView(BaseModel):
    """One assembly member for the parts panel: printed (custom CAD) vs bought
    hardware, with the registry citation that grounds a hardware part's dimensions."""

    model_config = ConfigDict(extra="forbid")
    id: str
    printed: bool
    hardware_kind: str | None = None
    source: str | None = None  # the registry citation (hardware only)


class AssemblyJointView(BaseModel):
    """One moving joint: limits + the EARNED motion-verification level (verbatim from
    the validator -- the API layer never recomputes it) + the optional ck_joint v1.1
    mimic coupling (follower_angle = ratio * driver_angle)."""

    model_config = ConfigDict(extra="forbid")
    id: str
    name: str | None = None
    type: str  # revolute | prismatic
    limits: list[float] | None = None
    checked: str  # swept | poses:N | none -- EARNED, displayed verbatim
    mimic_of: str | None = None
    mimic_ratio: float | None = None


class AssemblyView(BaseModel):
    """The assembly manifest, PROJECTED verbatim from the one manifest dict the GLB
    exporter returned plus the gate map (single source -- recomputing any of it in
    the API layer could let two surfaces disagree). None on single-part results."""

    model_config = ConfigDict(extra="forbid")
    parts: list[AssemblyPartView] = Field(default_factory=list)
    joints: list[AssemblyJointView] = Field(default_factory=list)
    gate: dict[str, str] = Field(default_factory=dict)  # check id -> pass|warn|fail|info
    step_note: str | None = None  # what the assembly STEP does/doesn't contain


class PatchView(BaseModel):
    """What a /patch edit changed (ADR-0037): ONE spec field, old -> new. Rides the ok
    payload so the UI can say exactly what moved (and the rest provably didn't)."""

    model_config = ConfigDict(extra="forbid")
    field_path: str
    op: str  # set | increase | decrease
    old_value: float | bool | str | None = None
    new_value: float | bool | str | None = None


class DimensionTierCounts(BaseModel):
    """The dimension-provenance mix (ADR-0040): how many dims earned each tier."""

    model_config = ConfigDict(extra="forbid")
    verified: int = 0
    sourced: int = 0
    stated: int = 0
    estimated: int = 0


class DimensionView(BaseModel):
    """One feature's dimension provenance: the tier it EARNED + the citation that earned it."""

    model_config = ConfigDict(extra="forbid")
    feature: str
    tier: str  # verified | sourced | stated | estimated
    citation: str | None = None
    dims: str | None = None  # the reference dims text, for display


class ConfidenceView(BaseModel):
    """The honest confidence + fitness verdict (ADR-0041) -- COMPUTED from evidence, never
    LLM-composed. ``basis`` is null exactly when validation hard-failed (no basis is claimed;
    ``not_for_manufacture`` already stamps the payload). This block replaces refusal as the
    trust layer: the engine attempts the part; this says how solid the result is."""

    model_config = ConfigDict(extra="forbid")
    overall: str  # high | medium | low
    basis: str | None  # validated | approximate | conceptual | null (failed gates)
    manufacturability: str  # prints_as_is | makeable | printable_as_model | not_makeable_as_shown
    process_note: str | None = None
    dimension_tiers: DimensionTierCounts
    dimensions: list[DimensionView] = Field(default_factory=list)
    verify_dimensions: list[str] = Field(default_factory=list)
    not_fit_for: list[str] = Field(default_factory=list)
    part_class: str = "none"
    summary: str


class ConceptView(BaseModel):
    """The chosen concept (ADR-0048): a closed mechanism + attributed design-proposal prose. There
    is no fitness/performance field -- the same structural guard the confidence block applies."""

    model_config = ConfigDict(extra="forbid")
    concept_id: str
    name: str
    primary_mechanism: str
    secondary_mechanisms: list[str] = Field(default_factory=list)
    approach: str
    obvious_baseline: str
    difference_from_obvious: str
    key_risks: list[str] = Field(default_factory=list)


class FeasibilityView(BaseModel):
    """Feasibility as an ADVISORY assessment (ADR-0048): a categorical verdict + the brain's
    attributed note + the mandatory disclaimer. Never a certifying pass."""

    model_config = ConfigDict(extra="forbid")
    verdict: str
    note: str | None = None
    disclaimer: str


class PhysicalValidationView(BaseModel):
    """The mandatory physical-validation disclosure (ADR-0047): the status + the banner the package
    cannot render without. CAD-valid is never physical proof."""

    model_config = ConfigDict(extra="forbid")
    status: str
    banner: str


class InventionPackageView(BaseModel):
    """The invention-lab package envelope (ADR-0047/0048). Composes ONLY typed/closed data: the
    maturity + evidence labels, the advisory feasibility, the NOT-PHYSICALLY-VALIDATED banner, the
    honest BOM headline, and the chosen concept. No generated fitness prose can enter it."""

    model_config = ConfigDict(extra="forbid")
    part_name: str
    problem_statement: str | None = None
    disposition: str
    maturity: str  # MaturityLabel: concept | prototype_candidate | digitally_validated | ...
    evidence_tier: str  # EvidenceTier: ai_hypothesis | deterministic_check | ...
    bom_summary: str
    feasibility: FeasibilityView
    physical_validation: PhysicalValidationView
    concept: ConceptView | None = None
    # The grounded support-copilot KB (ADR-0050), additive: the deterministic SupportCopilotKB dump,
    # present only when FORGE_INVENTION is on. The frontend passes it verbatim as the `kb` field of
    # POST /invention/support. A pass-through dict (like `spec`) keeps this API layer decoupled from the
    # invention domain types while remaining a stable, documented shape.
    support_kb: dict[str, Any] | None = None


class CertificateView(BaseModel):
    """THE CERTIFICATE (T3 / Fable SS2: the receipt is the product). Rides a physics-authored
    result (FORGE_PHYSICS_ROUTE lane, a load claim) OR -- ADR-0109 -- the DEFAULT agentic result as a
    MANUFACTURABILITY certificate mapped from the validator's per-check verdicts (geometry, NOT a
    load claim; the caveats say so). Simultaneously the trust object, the M4 flywheel label,
    and the audit trail. Every field is computed/echoed, never LLM-composed: the spec echo in
    human units (with the envelope's stated-vs-defaulted provenance), the uniform gate's
    per-check verdicts, the author provenance, the measured numbers, and honest caveats."""

    model_config = ConfigDict(extra="forbid")
    spec_echo: dict[
        str, Any
    ]  # part_name, load_case_echo, load_note_verbatim, envelope + provenance
    checks: dict[str, str]  # per-check verdicts: pass | fail | unproven | n/a
    provenance: str  # which author produced the claim, e.g. "analytic_isection"
    measured: dict[str, Any]  # mass_g, deflection_mm, load_case_echo... (the gate's numbers)
    caveats: list[str] = Field(default_factory=list)
    claimable: bool
    # the monotone fallback trail: every rung tried, its provenance, and whether its gate passed
    attempts: list[dict[str, Any]] = Field(default_factory=list)


class GenerateOk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"] = "ok"
    prompt: str
    gen_path: str
    cost_usd: float
    validation: ValidationView
    kit: Kit
    files: Files
    # How the part was validated: "general" (faithful CAD model, FDM checks OFF) or "print_3d"
    # (FDM printability applied). The frontend labels the result accordingly (ADR-0031). Plain str
    # (not Literal): it is set from the FabricationMode enum's value, an output field for display.
    fabrication_mode: str = "general"
    # The agentic acceptance judge's HONEST outcome (ADR-0035 B8), additive: e.g. "shipped" /
    # "shipped_with_caveats" / "rejected" (agentic dispositions); None on the deterministic
    # standard/implicit lanes (no judge runs there -- the strict gates are the verdict). The
    # server stamps "rejected" payloads not_for_manufacture; the UI labels accordingly.
    disposition: str | None = None
    # The part's PARAMETRIC MODEL (ADR-0037): the LayoutSpec dump when the standard lane built
    # it; None on the agentic/implicit paths. /patch is STATELESS -- the client sends this spec
    # back with an edit and gets a re-derived part where everything else is byte-stable.
    spec: dict[str, Any] | None = None
    # Set on /patch responses: the one field the edit changed (ADR-0037). None on /generate.
    patch: PatchView | None = None
    # The honest confidence + fitness verdict (ADR-0041), additive: per-dimension provenance
    # tiers, manufacturability, and what the part is NOT fit for. None only on legacy cache hits.
    confidence: ConfidenceView | None = None
    # The assembly manifest (ADR-0044, additive): parts (printed vs cited hardware), moving
    # joints with their EARNED checked level + mimic couplings, and the claim-gate check map.
    # None on single-part results -- the frontend's parts/receipts panel keys on this.
    assembly: AssemblyView | None = None
    # The invention-lab package envelope (ADR-0048), additive + FLAG-GATED: present only when
    # FORGE_INVENTION is on. Carries the maturity + evidence labels, the advisory feasibility, the
    # mandatory NOT-PHYSICALLY-VALIDATED banner, and the chosen concept (when the lifecycle ran the
    # front door). None on every existing path -> byte-identical response when the flag is off.
    invention_package: InventionPackageView | None = None
    # THE CERTIFICATE (T3), additive + FLAG-GATED: present only when FORGE_PHYSICS_ROUTE is on AND
    # the physics lane authored the part (a stated load routed analytic/topopt and its uniform gate
    # was claimable). None on every existing path -> byte-identical response when the flag is off.
    certificate: CertificateView | None = None


class GenerateRefused(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["refused"] = "refused"
    reason: str


class GenerateError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["error"] = "error"
    stage: str | None = None
    message: str


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1, max_length=2000)
    # The frontend sends "standard" (faster) | "studio" (more drafts). Optional; defaults standard.
    quality: Literal["standard", "studio"] = "standard"
    # Fabrication target (ADR-0031). Default GENERAL: a faithful CAD model with FDM checks OFF. The
    # client opts into "print_3d" (the deployed server maps its `printable` toggle to this) to apply
    # printability gates + design-for-print.
    fabrication_mode: Literal["print_3d", "general"] = "general"


__all__ = [
    "AssemblyJointView",
    "AssemblyPartView",
    "AssemblyView",
    "BillOfMaterials",
    "BomLine",
    "CheckView",
    "Files",
    "GenerateError",
    "GenerateOk",
    "GenerateRefused",
    "GenerateRequest",
    "Kit",
    "MakeInstructions",
    "TierStatus",
    "ValidationView",
    "Vec3",
]
