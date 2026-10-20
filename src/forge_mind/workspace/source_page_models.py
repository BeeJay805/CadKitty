"""Public page selection, independent of any CAD revision or acceptance claim."""

from typing import Any

from pydantic import Field

from forge_mind.workspace.models import Record
from forge_mind.workspace.source_capacity import source_policy


class SourcePageRequest(Record):
    page_number: int = Field(ge=1, strict=True)
    scale: float = Field(default=2.0, gt=0, allow_inf_nan=False, strict=True)


def source_page_schema() -> dict[str, Any]:
    return {
        "request": SourcePageRequest.model_json_schema(),
        "source_capacity": source_policy(),
        "page_numbering": "One-based page numbers in the captured PDF.",
        "source": "An owned immutable PDF source artifact with retained original bytes.",
        "legacy_sources": "A source without retained PDF bytes is unavailable; refetch creates a new source identity.",
        "backend": "Readiness is checked by the bounded worker on request; missing or unsupported backends return explicit errors.",
        "scope": "Source-page text and images are untrusted observations. Reading does not confirm claims or establish CAD acceptance.",
    }
