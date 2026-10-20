"""Contracts are frozen outside the authoring loop; evidence belongs to one revision."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from forge_mind.workspace.completion_models import (
    FunctionalInterface,
    RequirementDisposition,
    Subsystem,
)
from forge_mind.workspace.inspection_models import InspectionCharacteristic
from forge_mind.workspace.verification_models import (
    VerificationCriterion,
    VerificationSubjectBinding,
)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


PlanName = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")]


class OperationFrame(Record):
    id: PlanName
    description: str = Field(min_length=1)
    part_id: PlanName | None = Field(
        default=None,
        description=(
            "Frame owner. Null (or omitted) declares a shared frame usable across parts. "
            "A named owner permits only operation steps and dimension facts with that same "
            "part_id to reference this frame. Ownership is never inferred or changed automatically."
        ),
    )
    occurrence_path: str | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)


class DimensionFact(Record):
    id: PlanName
    value: float
    source: Literal["customer", "measured_reference", "assumption"]
    source_quote: str = Field(min_length=1)
    units: str = "mm"
    part_id: PlanName | None = None
    frame_id: PlanName | None = Field(
        default=None,
        description=(
            "Existing operation frame ID. Its part_id must be null (shared) or equal this "
            "dimension fact's part_id; a fact without part scope cannot use an owned frame."
        ),
    )
    binding_name: PlanName | None = None
    requirement_ids: list[str] = Field(default_factory=list)
    supersedes_fact_id: PlanName | None = None

    @model_serializer(mode="wrap")
    def legacy_supersession_absence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if self.supersedes_fact_id is None:
            result.pop("supersedes_fact_id", None)
        return result


class OperationStep(Record):
    id: PlanName
    part_id: PlanName
    depends_on: list[PlanName] = Field(default_factory=list)
    feature_ids: list[PlanName] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    binding_names: list[PlanName] = Field(default_factory=list)
    frame_id: PlanName | None = Field(
        default=None,
        description=(
            "Existing operation frame ID. Its part_id must be null (shared) or equal this "
            "step's part_id. A frame owned by another part cannot serve as a shared datum."
        ),
    )
    expected_effect: str = Field(min_length=1)
    inspection_targets: list[str] = Field(default_factory=list)
    kind: Literal["construction", "parameters", "inspection", "delivery"] = "construction"
    referenced_feature_ids: list[PlanName] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def legacy_step_role_absence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if self.kind == "construction":
            result.pop("kind", None)
        if not self.referenced_feature_ids:
            result.pop("referenced_feature_ids", None)
        return result


class OperationPlan(Record):
    version: Literal[1] = 1
    units: Literal["mm"] = "mm"
    part_ids: list[PlanName] = Field(default_factory=list)
    frames: list[OperationFrame] = Field(default_factory=list)
    dimensions: list[DimensionFact] = Field(default_factory=list)
    steps: list[OperationStep] = Field(default_factory=list)
    checkpoint: str = ""
    unresolved: list[str] = Field(default_factory=list)


class DesignPlan(Record):
    objective: str = Field(min_length=1, max_length=2000)
    parts: list[str] = Field(default_factory=list)
    interfaces: list[str] = Field(default_factory=list)
    assembly_sequence: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    subsystems: list[Subsystem] = Field(default_factory=list)
    requirement_dispositions: list[RequirementDisposition] = Field(default_factory=list)
    inspection: list[InspectionCharacteristic] = Field(default_factory=list)
    operation_plan: OperationPlan | None = None
    functional_interfaces: list[FunctionalInterface] = Field(default_factory=list)
    verification_bindings: list[VerificationSubjectBinding] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def legacy_operation_absence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if self.operation_plan is None:
            result.pop("operation_plan", None)
        if not self.functional_interfaces:
            result.pop("functional_interfaces", None)
        if not self.verification_bindings:
            result.pop("verification_bindings", None)
        return result


class Requirement(Record):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    text: str = Field(min_length=1, max_length=2000)
    source_quote: str = Field(min_length=1, max_length=2000)
    source: Literal["request", "explicit", "assumption"] = "request"
    metric: Literal[
        "extent_x",
        "extent_y",
        "extent_z",
        "volume",
        "solid_count",
        "hole_count",
        "hole_diameter",
        "hole_depth",
        "min_wall",
        "unmeasured",
    ] = "unmeasured"
    comparator: Literal["eq", "ge", "le"] = "eq"
    value: float | None = None
    tolerance: float = Field(default=0.0, ge=0)
    part_id: str | None = Field(default=None, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    # Hole selection is separate from diameter conformance, so an undersized hole
    # cannot disappear by falling outside the requested diameter's match window.
    axis: Literal["x", "y", "z"] = "z"
    center: tuple[float, float] | None = None
    position_tolerance: float = Field(default=0.1, gt=0, le=10)
    critical: bool = True
    verification: VerificationCriterion | None = None

    @model_serializer(mode="wrap")
    def legacy_verification_absence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if self.verification is None:
            result.pop("verification", None)
        return result

    @model_validator(mode="after")
    def numeric_contract(self) -> Requirement:
        if self.verification is not None and (
            self.metric != "unmeasured"
            or self.value is not None
            or self.part_id is not None
            or self.center is not None
            or self.tolerance != 0
            or self.comparator != "eq"
        ):
            raise ValueError("Typed verification cannot mix scalar targets or part selectors")
        if self.metric != "unmeasured" and self.value is None:
            raise ValueError("measurable requirements need an explicit value")
        if self.value is not None and self.value < 0:
            raise ValueError("dimensions and counts must be nonnegative")
        if self.metric in {"solid_count", "hole_count"} and (
            self.value is None or not self.value.is_integer() or self.tolerance != 0
        ):
            raise ValueError("counts need an integer value and zero tolerance")
        return self


class Contract(Record):
    requirements: list[Requirement] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> Contract:
        ids = [r.id for r in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement IDs must be unique")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()


class Removal(Record):
    requirement_id: str
    source_quote: str = Field(min_length=1, max_length=2000)


class ContractEdit(Contract):
    removals: list[Removal] = Field(default_factory=list)


class Finding(Record):
    requirement_id: str
    status: Literal["pass", "fail", "unverified"]
    measured: float | None = None
    expected: float | None = None
    method: str
    reason: str
    locator: list[float] | None = None


class Measurement(Record):
    revision_id: str
    contract_hash: str
    artifact_hashes: dict[str, str]
    engine_fingerprint: str
    geometry_valid: bool
    facts: dict[str, object] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    requirements_satisfied: bool = False


class Action(Record):
    kind: Literal[
        "build",
        "document",
        "parameters",
        "inspect",
        "section",
        "restore",
        "finish",
        "catalog",
        "read_part",
        "edit_part",
        "review",
        "template",
        "schema",
        "plan",
        "assembly_edit",
        "feature_edit",
        "motion_check",
        "study",
        "research",
        "fetch_source",
        "source_claim",
        "read_source",
        "source_page",
        "photo_profile",
        "engineering_tool",
        "product_review",
        "propose_interface",
        "validate_interfaces",
        "lab",
        "design_study",
        "read_design",
        "guidance",
        "print_setup",
    ]
    explanation: str = Field(min_length=1, max_length=3000)
    code: str = ""
    document_json: str = ""
    plan: DesignPlan | None = None
    parameters: dict[str, float] = Field(default_factory=dict)
    part_id: str = Field(default="", max_length=64)
    query: str = Field(default="", max_length=2000)
    page_number: int | None = Field(default=None, ge=1, strict=True)
    revision_id: str | None = None
    axis: Literal["x", "y", "z"] = "z"
    offset_mm: float = 0.0
    study_values: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def source_page_target(self) -> Action:
        if self.kind == "source_page" and (
            self.page_number is None or re.fullmatch(r"[a-f0-9]{32}", self.query) is None
        ):
            raise ValueError(
                "source_page requires an owned source artifact ID in query and a one-based page_number"
            )
        return self


class WorkspaceRequest(Record):
    prompt: str = Field(min_length=1)
    contract: Contract | None = None
    max_actions: int | None = Field(default=None, ge=1)
    max_seconds: int | None = Field(default=None, ge=1)
    preview_only: bool = True
    authoring_mode: Literal["hybrid", "native_operations"] | None = None

    @model_serializer(mode="wrap")
    def legacy_authoring_absence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if self.authoring_mode is None:
            result.pop("authoring_mode", None)
        return result


class EditRequest(Record):
    max_actions: int | None = Field(default=None, ge=1)
    max_seconds: int | None = Field(default=None, ge=1)
    instruction: str = Field(default="", max_length=10000)
    parameters: dict[str, float] = Field(default_factory=dict)
    revision_id: str | None = None
    expected_current_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    expected_input_version: int | None = Field(default=None, ge=0)
    contract: Contract | None = None
    preview_only: bool = True
