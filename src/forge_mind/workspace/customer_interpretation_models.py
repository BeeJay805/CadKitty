"""Source-linked nominal proposals and host-created interpretation identities.

These records express an interpretation. They do not prove English entailment,
source completeness, manufactured dimensions, or physical performance.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from forge_mind.workspace.verification_models import (
    Name,
    NominalPerforatedBox,
    NominalThroughHolePattern,
    VerificationRecord,
)

RequirementId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
POLICY: Literal["customer_nominal_interpretation_v1"] = "customer_nominal_interpretation_v1"


class NominalFactTerm(VerificationRecord):
    fact_id: Name
    operation: Literal["value", "negative", "half", "negative_half"] = "value"


class NominalFieldBinding(VerificationRecord):
    field: Annotated[str, Field(min_length=1)]
    basis: Literal["source_relation", "design_origin"] = "source_relation"
    terms: list[NominalFactTerm] = Field(default_factory=list)

    @model_validator(mode="after")
    def explicit_operands(self) -> NominalFieldBinding:
        if (self.basis == "source_relation") != bool(self.terms):
            raise ValueError("A source relation needs operands; a design origin has none")
        if len({term.fact_id for term in self.terms}) != len(self.terms):
            raise ValueError("A field cannot repeat a source operand")
        return self


class CustomerNominalTarget(VerificationRecord):
    """A proposed subcheck; target values never replace original requirements."""

    requirement_id: RequirementId
    source_requirement_ids: list[RequirementId] = Field(min_length=1)
    criterion: NominalPerforatedBox | NominalThroughHolePattern
    frame_policy: Literal["centered_body", "source_defined_frame"]
    field_bindings: list[NominalFieldBinding] = Field(min_length=1)
    hole_count_fact_id: Name

    @model_validator(mode="after")
    def unique_references(self) -> CustomerNominalTarget:
        if len(set(self.source_requirement_ids)) != len(self.source_requirement_ids):
            raise ValueError("Source requirement references must be unique")
        fields = [binding.field for binding in self.field_bindings]
        if len(set(fields)) != len(fields):
            raise ValueError("Nominal target field bindings must be unique")
        return self


class CustomerNominalInterpretation(VerificationRecord):
    """Host-created policy context. Never an author or customer verdict."""

    policy: Literal["customer_nominal_interpretation_v1"] = POLICY
    source_sha256: Sha256
    contract_sha256: Sha256
    attribution_sha256: Sha256
    original_requirement_ids: list[RequirementId] = Field(min_length=1)
    original_requirements_sha256: Sha256
    targets: list[CustomerNominalTarget] = Field(min_length=1)
