"""Intake projects existing operation semantics, never a second tool catalog."""

from __future__ import annotations

import json
import re
from typing import Any

from forge_mind.workspace.print_setup_models import ALGORITHM, SCOPE, PrintSetupRequest


def validate_preparation_source_scope(source_quote: str) -> None:
    """Accept only a complete artifact clause; unfamiliar extra scope stays open."""
    from forge_mind.workspace.verification_models import DigitalDelivery

    try:
        value = json.loads(source_quote)
        if isinstance(value, dict) and "source_quote" not in value:
            criterion = DigitalDelivery.model_validate({**value, "source_quote": source_quote})
            if criterion.deliverable == "prepared_print_package":
                return
    except (ValueError, TypeError):
        pass
    request = (
        r"(?:(?:please\s+)?(?:give\s+me|deliver|provide|export|prepare|include)|"
        r"I\s+(?:need|want|request))"
    )
    article = r"(?:(?:an?|the|one)\s+)?"
    artifact = r"(?:unsliced\s+)?prepared\s+print\s+package"
    coverage = r"(?:\s+(?:covering|for)\s+(?:everything\s+made|all\s+made(?:\s+(?:parts|definitions|items))?))?"
    if not re.fullmatch(
        rf"(?:{request}\s+)?{article}{artifact}{coverage}[.!]?", source_quote.strip(), re.I
    ):
        raise ValueError(
            "Unsliced placement proof requires a complete artifact-only prepared print clause. "
            "Retain unparsed machine, sliced or physical duties and their real questions separately."
        )


def preparation_capability() -> dict[str, Any]:
    """Describe the same unsliced operation used by print setup and discovery."""
    return {
        "operation": "prepared_print_package",
        "request_schema": "PrintSetupRequest",
        "algorithm": ALGORITHM,
        "required_saved_inputs": [
            name
            for name, field in PrintSetupRequest.model_fields.items()
            if field.is_required() and name != "operation"
        ],
        "required_customer_inputs": [],
        "source_scope": "Complete artifact-only clause; additional source duties require separate retained uncertainty",
        "scope": SCOPE,
        "output": "Separate plate-aligned made-definition STLs with exact occurrence quantities",
        "excluded_outputs": ["sliced_toolpaths", "G-code", "printer_profile", "physical_approval"],
    }
