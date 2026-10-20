"""Pure, paged findings views over evidence verified by the calling transport.

This module performs no ownership, artifact, acceptance or geometry checks.
The caller must retain its existing boundary before projecting a measurement.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from forge_mind.workspace.models import Finding, Measurement, Record

FindingSelection = Literal["actionable", "all"]
IntegrityScope = Literal["evidence_record", "revision_artifact_hashes"]


class EvidenceFinding(Finding):
    """Keep JSON integers exact rather than converting them to binary floats."""

    measured: float | int | None = None
    expected: float | int | None = None
    locator: list[float | int] | None = None


class EvidenceSummary(Record):
    """Exact findings with explicit filter, pagination and inherited integrity scope."""

    projection: Literal["findings_summary"] = "findings_summary"
    revision_id: str
    contract_hash: str
    engine_fingerprint: str
    artifact_hashes: dict[str, str]
    integrity_scope: IntegrityScope
    geometry_valid: bool
    requirements_satisfied: bool
    statuses: FindingSelection
    counts: dict[Literal["pass", "fail", "unverified"], int]
    total_findings: int
    filtered_count: int
    returned_count: int
    offset: int
    limit: int
    next_offset: int | None
    omitted_pass_count: int
    page_complete: bool = Field(
        description="True only if this response includes every finding in the selected filter."
    )
    all_findings_included: bool = Field(
        description="True only if this response includes every finding in the full evidence."
    )
    findings: list[EvidenceFinding]
    errors: list[str]
    caveats: list[str]
    scope: str = (
        "Findings projection only; facts are omitted. Read get_evidence for complete evidence. "
        "page_complete means the selected filter is fully included; all_findings_included "
        "means the full findings list is included. "
        "Integrity scope comes from the calling transport. This view does not accept a revision "
        "or certify manufacture, physical fitness, or complete requirement coverage."
    )


class _Selection(Record):
    expected_revision: str = Field(pattern=r"^[a-f0-9]{32}$", strict=True)
    statuses: FindingSelection
    offset: int = Field(ge=0, strict=True)
    limit: int = Field(ge=1, le=100, strict=True)
    integrity_scope: IntegrityScope


def evidence_summary(
    measurement: dict[str, Any],
    expected_revision: str,
    *,
    statuses: FindingSelection = "actionable",
    offset: int = 0,
    limit: int = 40,
    integrity_scope: IntegrityScope,
) -> EvidenceSummary:
    """Validate and copy an already-verified measurement without changing its meaning.

    Actionable includes fail and unverified findings in their original order.
    Counts cover the full evidence; pagination applies only to the chosen findings.
    Errors and caveats are complete and are never paged or shortened.
    """
    selection = _Selection(
        expected_revision=expected_revision,
        statuses=statuses,
        offset=offset,
        limit=limit,
        integrity_scope=integrity_scope,
    )
    if not isinstance(measurement, dict):
        raise ValueError("Evidence must be a measurement object")
    # Measurement has defaults for several fields. Do not invent absent evidence.
    if Measurement.model_fields.keys() - measurement.keys():
        raise ValueError("Evidence measurement is incomplete")
    try:
        # ConfigDict's finite check does not inspect arbitrary nested facts.
        json.dumps(measurement, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Evidence must contain finite JSON values") from exc
    verified = Measurement.model_validate(measurement, strict=True)
    if verified.revision_id != selection.expected_revision:
        raise ValueError("Evidence does not match the requested revision")
    findings = []
    for row in measurement["findings"]:
        if not isinstance(row, dict) or Finding.model_fields.keys() - row.keys():
            raise ValueError("Evidence finding is incomplete")
        findings.append(EvidenceFinding.model_validate(row, strict=True))
    counts: dict[Literal["pass", "fail", "unverified"], int] = {
        "pass": 0,
        "fail": 0,
        "unverified": 0,
    }
    for finding in findings:
        counts[finding.status] += 1
    selected = [finding for finding in findings if statuses == "all" or finding.status != "pass"]
    page = selected[offset : offset + limit]
    next_offset = offset + len(page) if offset + len(page) < len(selected) else None
    return EvidenceSummary(
        revision_id=verified.revision_id,
        contract_hash=verified.contract_hash,
        engine_fingerprint=verified.engine_fingerprint,
        artifact_hashes=dict(verified.artifact_hashes),
        integrity_scope=selection.integrity_scope,
        geometry_valid=verified.geometry_valid,
        requirements_satisfied=verified.requirements_satisfied,
        statuses=selection.statuses,
        counts=counts,
        total_findings=len(verified.findings),
        filtered_count=len(selected),
        returned_count=len(page),
        offset=selection.offset,
        limit=selection.limit,
        next_offset=next_offset,
        omitted_pass_count=counts["pass"] - sum(f.status == "pass" for f in page),
        page_complete=offset == 0 and len(page) == len(selected),
        all_findings_included=offset == 0 and len(page) == len(verified.findings),
        findings=[finding.model_copy(deep=True) for finding in page],
        errors=list(verified.errors),
        caveats=list(verified.caveats),
    )


def summary_text(summary: EvidenceSummary) -> str:
    """Readable MCP text preserves every page finding, error and caveat exactly."""
    integrity = (
        "Caller verified evidence against its revision record; this view does not rehash artifacts."
        if summary.integrity_scope == "evidence_record"
        else "Caller verified all recorded revision artifact hashes; this view adds no integrity checks."
    )
    lines = [
        f"Evidence findings for revision {summary.revision_id}.",
        f"geometry_valid={summary.geometry_valid}; "
        f"requirements_satisfied={summary.requirements_satisfied}.",
        f"All findings: {summary.total_findings}; pass={summary.counts['pass']}; "
        f"fail={summary.counts['fail']}; unverified={summary.counts['unverified']}.",
        f"Filter={summary.statuses}; filtered={summary.filtered_count}; "
        f"returned={summary.returned_count}; offset={summary.offset}; limit={summary.limit}; "
        f"next_offset={summary.next_offset}; omitted_pass_count={summary.omitted_pass_count}; "
        f"page_complete={summary.page_complete}; "
        f"all_findings_included={summary.all_findings_included}.",
        f"Integrity scope={summary.integrity_scope}; "
        f"contract_hash={summary.contract_hash}; engine_fingerprint={summary.engine_fingerprint}.",
        integrity,
    ]
    for finding in summary.findings:
        # JSON retains exact field meanings and escapes untrusted text without truncation.
        lines.append(json.dumps(finding.model_dump(), allow_nan=False, ensure_ascii=False))
    lines.extend(
        (
            "Errors: " + json.dumps(summary.errors, ensure_ascii=False),
            "Caveats: " + json.dumps(summary.caveats, ensure_ascii=False),
            summary.scope,
        )
    )
    return "\n".join(lines)
