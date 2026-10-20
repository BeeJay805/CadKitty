"""Domain playbooks (ADR-0087 S4, founder direction 2026-07-05) — structured thinking, registered.

When the requirement tree reaches a sub-problem ("the thermals must be right"), the model must not
free-associate. It lands in a PLAYBOOK — a git-reviewed data structure that says exactly where to
go: which FACTS this domain always needs (and how each closes: registry / research / ask), which
CHECKS it can compile into (a closed vocabulary the validator implements — the model cannot invent a
check), the domain's HONESTY CEILING (what we cannot validate, verbatim, riding the card whenever the
domain is touched — considered is never presented as validated), and advisory HEURISTICS passed to
the engine's prompt (they shape the attempt, never a claim).

This is the registry discipline applied to reasoning: hardware facts live in cited registries,
validation lives in keyed fail-closed checks, and now per-domain thinking lives here. A node whose
domain matches no playbook is handled generically and honestly (unvalidated/ask) and logged as
curation demand — playbooks grow by git review, never by runtime self-authoring.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The CLOSED check vocabulary the compiler may emit (S5 implements them against geometry).
CheckId = Literal["envelope_fits", "keepout_clear", "interface_pattern", "path_cross_section"]

CHECK_IDS: tuple[str, ...] = (
    "envelope_fits",
    "keepout_clear",
    "interface_pattern",
    "path_cross_section",
)


class FactTemplate(BaseModel):
    """One fact a domain always needs, with its closure routes. The compiler resolves in the fixed
    honest order: user-stated -> registry -> research -> (reviewed default | ask) — never invented.

    ``ask_policy`` is the anti-quiz lever (run-4 bench finding — a router bracket was asked fan
    dimensions): ``ask`` = a question when unresolved; ``default`` = close with the reviewed
    default value as a VISIBLE assumption on the card (source="derived", never an anchor tier,
    editable), so low-stakes facts stop being questions; ``leaf_only`` = in the shared vocabulary
    (so the model's naming converges) but never auto-gathered — only an explicit tree leaf asks it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Canonical snake_case fact name, e.g. 'fan_frame_size'.")
    ask: str = Field(..., description="The question to ask the user; may contain {entity}.")
    research_query: str | None = Field(
        default=None, description="Web-research query template ({entity}); None = never researched."
    )
    registry: str | None = Field(
        default=None,
        description="Registry NAME this fact can resolve from (key supplied at compile).",
    )
    registry_attr: str | None = Field(
        default=None,
        description="Attribute on the registry spec carrying the value, e.g. 'width_mm'.",
    )
    unit: str = "mm"
    ask_policy: Literal["ask", "default", "leaf_only"] = "ask"
    default_mm: float | None = Field(
        default=None, description="The reviewed default (ask_policy='default' only)."
    )
    default_note: str = ""
    # N1 identity-first: when the registry KEY is known and this attr is a tuple of standard
    # values (e.g. drive height classes), the question becomes a CHOICE over those cited values
    # instead of a free mm ask — a normal person picks, never measures. ``choice_ask`` is the
    # identity-phrased question for that path (the generic ``ask`` stays payload-agnostic).
    registry_choice_attr: str | None = None
    choice_ask: str = ""


class CheckTemplate(BaseModel):
    """A validator check this domain can compile into, parameterized by named facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: CheckId
    params_from: tuple[str, ...] = Field(..., description="Fact names that parameterize the check.")
    description: str


class DomainPlaybook(BaseModel):
    """The structured method for one requirement domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str
    triggers: tuple[str, ...] = Field(..., description="Keywords that classify a node here.")
    facts: tuple[FactTemplate, ...] = ()
    checks: tuple[CheckTemplate, ...] = ()
    honesty_ceiling: str | None = Field(
        default=None,
        description="What this domain CANNOT validate — verbatim on the card whenever touched.",
    )
    heuristics: tuple[str, ...] = ()


PLAYBOOKS: dict[str, DomainPlaybook] = {
    "fit_envelope": DomainPlaybook(
        domain="fit_envelope",
        triggers=("fit", "envelope", "space", "clearance", "inside", "dimensions"),
        facts=(
            FactTemplate(
                name="available_length",
                ask="How much length (mm) is available where the part will sit in {entity}?",
                research_query="{entity} internal dimensions drive bay length mm",
            ),
            FactTemplate(
                name="available_width",
                ask="How much width (mm) is available where the part will sit in {entity}?",
                research_query="{entity} internal dimensions drive bay width mm",
            ),
            FactTemplate(
                name="available_height",
                ask="How much height (mm) is available where the part will sit in {entity}?",
                research_query="{entity} internal clearance height mm",
            ),
        ),
        checks=(
            CheckTemplate(
                check_id="envelope_fits",
                params_from=("available_length", "available_width", "available_height"),
                description="The built part must fit inside the stated available envelope.",
            ),
        ),
        heuristics=("Design to the stated envelope minus print tolerance; never touch the walls.",),
    ),
    "mount_interface": DomainPlaybook(
        domain="mount_interface",
        triggers=("mount", "screw", "bolt", "attach", "bracket", "hole pattern", "fasten"),
        facts=(
            FactTemplate(
                name="payload_width",
                ask="How wide (mm) is the object being mounted?",
                registry="drive_form",
                registry_attr="width_mm",
            ),
            FactTemplate(
                name="payload_length",
                ask="How long (mm) is the object being mounted?",
                registry="drive_form",
                registry_attr="length_mm",
            ),
            FactTemplate(
                name="payload_thickness",
                ask="How thick (mm) is the object being mounted?",
                registry="drive_form",
                registry_choice_attr="heights_mm",
                choice_ask="How thick is the drive? (standard heights)",
            ),
            FactTemplate(
                name="mount_hole_pitch",
                ask="What is the mounting-hole spacing (mm) on the surface the part attaches to?",
                registry="drive_form",
                registry_attr="bottom_hole_pitch_mm",
            ),
            # The case/surface side of the interface — the run-3/4 bench runs showed the model
            # inventing names for this (case_mount_hole_pitch, case_fixing_hole_pitch); giving it
            # a canonical name converges naming. leaf_only: asked only when the tree names it.
            FactTemplate(
                name="attachment_hole_pitch",
                ask="If the part bolts to EXISTING holes on the case/surface, what is their spacing (mm)?",
                ask_policy="leaf_only",
            ),
        ),
        checks=(
            CheckTemplate(
                check_id="interface_pattern",
                params_from=("mount_hole_pitch",),
                description="The part must carry the stated mounting-hole pattern within tolerance.",
            ),
        ),
        heuristics=(
            "Use registry clearance holes for the stated screw size; never guess a thread.",
            "Support the payload on at least two opposite edges or the full base.",
        ),
    ),
    "thermal_airflow": DomainPlaybook(
        domain="thermal_airflow",
        triggers=("thermal", "cooling", "airflow", "fan", "temperature", "vent", "heat"),
        facts=(
            FactTemplate(
                name="fan_frame_size",
                ask="Which fan serves this region of {entity} (frame size in mm: 80/92/120/140)?",
                research_query="{entity} fan size mm intake",
                registry="case_fan",
                registry_attr="frame_mm",
            ),
            FactTemplate(
                name="airflow_clearance",
                ask="How much free clearance (mm) must stay open in front of that fan?",
                research_query="{entity} fan clearance mm",
            ),
        ),
        checks=(
            CheckTemplate(
                check_id="keepout_clear",
                params_from=("fan_frame_size", "airflow_clearance"),
                description="No part volume inside the fan's projected airflow keepout.",
            ),
            CheckTemplate(
                check_id="path_cross_section",
                params_from=("fan_frame_size",),
                description="The free flow cross-section must not drop below the stated fraction of the fan frame area.",
            ),
        ),
        honesty_ceiling=(
            "Airflow is validated GEOMETRICALLY only — clearance and cross-section. "
            "Temperatures and CFM are NOT simulated."
        ),
        heuristics=(
            "Keep solid faces out of the region directly upstream of an intake fan.",
            "Prefer vented or open geometry inside a stated airflow path.",
            "Orient large flat faces parallel to the flow direction, not across it.",
        ),
    ),
    "cable_routing": DomainPlaybook(
        domain="cable_routing",
        triggers=("cable", "wire", "route", "connector", "plug", "sata", "power lead"),
        facts=(
            FactTemplate(
                name="connector_clearance",
                ask="How much clearance (mm) do the connectors/cables need beyond the payload edge?",
                research_query="{entity} connector clearance cable bend radius mm",
                ask_policy="default",
                default_mm=25.0,
                default_note="typical free space for a connector + cable bend",
            ),
        ),
        checks=(
            CheckTemplate(
                check_id="keepout_clear",
                params_from=("connector_clearance",),
                description="No part volume inside the stated connector/cable keepout.",
            ),
        ),
        honesty_ceiling=(
            "Cable routing is validated as STATED CLEARANCE only — actual cable paths and bend "
            "radii are not simulated."
        ),
        heuristics=(
            "Leave the connector edge of the payload fully open.",
            "Round any edge a cable may rest against.",
        ),
    ),
    "access_service": DomainPlaybook(
        domain="access_service",
        triggers=("access", "remove", "service", "install", "reach", "tool", "screwdriver"),
        facts=(
            FactTemplate(
                name="tool_access_clearance",
                ask="How much straight-line tool access (mm) is needed at the fasteners?",
                ask_policy="default",
                default_mm=25.0,
                default_note="straight screwdriver access at each fastener",
            ),
        ),
        checks=(),
        honesty_ceiling=(
            "Serviceability is a stated-clearance judgment — insertion/removal paths are not "
            "kinematically simulated."
        ),
        heuristics=(
            "Fastener heads must face an open direction after installation.",
            "The payload must be insertable without flexing the printed part beyond a snap-fit.",
        ),
    ),
}


def classify_domain(statement: str) -> str | None:
    """Deterministic classifier assist: the playbook whose trigger appears in the statement.
    Ties resolve to the most trigger hits; None when nothing matches (the model's nomination is
    then judged on its own — and an unknown nomination handles generically)."""
    text = statement.lower()
    best: tuple[int, str] | None = None
    for name, pb in PLAYBOOKS.items():
        hits = sum(1 for t in pb.triggers if t in text)
        if hits and (best is None or hits > best[0]):
            best = (hits, name)
    return best[1] if best else None


__all__ = [
    "CHECK_IDS",
    "PLAYBOOKS",
    "CheckId",
    "CheckTemplate",
    "DomainPlaybook",
    "FactTemplate",
    "classify_domain",
]
