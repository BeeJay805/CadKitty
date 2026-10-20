"""Author intent for bounded subsystem work; statuses are never author supplied."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from forge_mind.workspace.check_amendment import CriterionAmendment
from forge_mind.workspace.completion_bindings import (
    CHECK_BINDING_DESCRIPTION,
    MATE_BINDING_DESCRIPTION,
    PARTICIPANT_BINDING_DESCRIPTION,
    STATE_BINDING_DESCRIPTION,
)
from forge_mind.workspace.product_tools import (
    AxialCavity,
    Cable,
    Clearance,
    Fastener,
    Holes,
    MountGrid,
    OpenRectangularShell,
    PlanarFacePair,
    ProjectedPair,
    Tolerance,
    Vent,
)


class PreparedCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    # An untagged union emits anyOf with closed objects for the Codex wire schema.
    # model_dump still produces the ordinary request dictionary consumed by tools.
    request: (
        Fastener
        | Tolerance
        | Cable
        | Holes
        | Vent
        | MountGrid
        | Clearance
        | ProjectedPair
        | PlanarFacePair
        | AxialCavity
        | OpenRectangularShell
    )
    reference_revision_id: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{32}$",
        description="Numeric body selectors require this exact current revision ID for completion proof.",
    )
    criterion_amendment: CriterionAmendment | None = Field(
        default=None,
        description=(
            "Required for a new semantic change to this same-ID request. Bind the exact old "
            "request hash and explain its correction basis. Previous failures remain historical; "
            "this metadata grants no acceptance and cannot remove frozen obligations."
        ),
    )

    @model_serializer(mode="wrap")
    def legacy_amendment_absence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if self.criterion_amendment is None:
            result.pop("criterion_amendment", None)
        return result


class Subsystem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    name: str = Field(min_length=1, max_length=160)
    requirement_ids: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    part_ids: list[str] = Field(default_factory=list)
    occurrence_paths: list[str] = Field(default_factory=list)
    checks: list[PreparedCheck] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class RequirementDisposition(BaseModel):
    """Authored proof obligations. References never assert requirement satisfaction."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    check_ids: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    unsupported_checks: list[str] = Field(default_factory=list)
    physical_validation: list[str] = Field(default_factory=list)


InterfaceId = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]{0,63}$")]
InterfacePath = Annotated[str, Field(strict=True, min_length=1, max_length=256)]
InterfaceText = Annotated[str, Field(strict=True, min_length=1, max_length=2000)]
InterfaceName = Annotated[str, Field(strict=True, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")]


class InterfaceFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    occurrence_path: InterfacePath = Field(
        description="Feature owner occurrence; must be in this interface's participant_paths."
    )
    feature_id: Annotated[str, Field(strict=True, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")]


class InterfaceState(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: InterfaceId
    kind: Literal["reference", "motion"] = Field(
        default="reference",
        description=(
            "Reference states require empty or omitted driver_values. Motion states require "
            "at least one driver value, including when the specified driver value is zero."
        ),
    )
    driver_values: dict[InterfaceName, Annotated[float, Field(strict=True)]] = Field(
        default_factory=dict,
        description=(
            "Explicit motion driver settings. A motion state requires a nonempty mapping; "
            "zero is a valid setting and still makes the state motion. A reference state "
            "requires an empty or omitted mapping. Sampled clearance compares the entire "
            "mapping with a verified requested configuration."
        ),
    )

    @model_validator(mode="after")
    def configuration(self) -> InterfaceState:
        if (self.kind == "motion") != bool(self.driver_values):
            raise ValueError("Motion states require driver values; reference states omit them")
        return self


class InterfaceDuty(BaseModel):
    """Requested scope only. No authored field can assert an evidence verdict."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: InterfaceId
    kind: Literal[
        "fit",
        "clearance",
        "connection",
        "retention",
        "holding",
        "load_transfer",
        "travel",
        "service_access",
        "process",
    ] = Field(description="Requested duty only; check_ids describes supported proof bindings.")
    proof_scope: Literal[
        "nominal_fit",
        "static_clearance",
        "sampled_clearance",
        "physical_function",
        "manufacturing_qualification",
    ] = Field(
        description="Requested evidence scope, never an authored verdict or physical qualification."
    )
    check_ids: list[InterfaceId] = Field(
        default_factory=list, description=CHECK_BINDING_DESCRIPTION
    )
    state_ids: list[InterfaceId] = Field(
        default_factory=list, description=STATE_BINDING_DESCRIPTION
    )
    missing_inputs: list[InterfaceText] = Field(default_factory=list)
    unsupported_checks: list[InterfaceText] = Field(default_factory=list)
    physical_validation: list[InterfaceText] = Field(default_factory=list)


class FunctionalInterface(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: InterfaceId
    requirement_ids: list[InterfaceId] = Field(min_length=1)
    participant_paths: list[InterfacePath] = Field(
        min_length=1,
        description=(
            (
                "Unique explicit occurrence participants. Every installed occurrence, required "
                "Feature owner and port owner must be included here. No participant is inferred "
                "from another list, and membership alone supplies no geometric or physical proof."
            )
            + " "
            + PARTICIPANT_BINDING_DESCRIPTION
        ),
    )
    installed_occurrence_paths: list[InterfacePath] = Field(
        default_factory=list,
        description="Unique installed occurrences; each must also appear in participant_paths.",
    )
    required_features: list[InterfaceFeature] = Field(
        default_factory=list,
        description=(
            "Unique occurrence_path/feature_id pairs. Each occurrence_path must appear in "
            "participant_paths; Feature IDs alone do not establish participant membership."
        ),
    )
    port_ids: list[InterfacePath] = Field(
        default_factory=list,
        description=(
            "Unique occurrence_path.port_id references. The owner before the final dot must "
            "appear in participant_paths. A port reference does not add its owner automatically."
        ),
    )
    mate_ids: list[InterfacePath] = Field(
        default_factory=list, description=MATE_BINDING_DESCRIPTION
    )
    states: list[InterfaceState] = Field(default_factory=list)
    duties: list[InterfaceDuty] = Field(min_length=1)

    @model_validator(mode="after")
    def references(self) -> FunctionalInterface:
        lists = (
            self.requirement_ids,
            self.participant_paths,
            self.installed_occurrence_paths,
            self.port_ids,
            self.mate_ids,
            [s.id for s in self.states],
            [d.id for d in self.duties],
            [(f.occurrence_path, f.feature_id) for f in self.required_features],
        )
        if any(len(values) != len(set(values)) for values in lists):
            raise ValueError("Functional interface references must be unique")
        participants = set(self.participant_paths)
        if set(self.installed_occurrence_paths) - participants or any(
            feature.occurrence_path not in participants for feature in self.required_features
        ):
            raise ValueError("Installed occurrences and required Features must be participants")
        if any(port.rpartition(".")[0] not in participants for port in self.port_ids):
            raise ValueError("Interface ports must belong to declared participants")
        states = {state.id for state in self.states}
        for duty in self.duties:
            if set(duty.state_ids) - states:
                raise ValueError("Interface duty references an unknown state")
            if len(duty.check_ids) != len(set(duty.check_ids)) or len(duty.state_ids) != len(
                set(duty.state_ids)
            ):
                raise ValueError("Duty check and state references must be unique")
        return self
