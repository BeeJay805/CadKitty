from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from forge_mind.workspace.evidence_view import evidence_summary, summary_text

RID = "a" * 32


def finding(status: str, number: int) -> dict[str, Any]:
    return {
        "requirement_id": f"check_{number}",
        "status": status,
        "measured": 90.00000460121548 + number,
        "expected": 90.0,
        "method": "native support-distance query",
        "reason": f"Exact finding {number}\nPreserve this second line.",
        "locator": [0.000000002201, 0.0, 7.12345678901234],
    }


def measurement() -> dict[str, Any]:
    return {
        "revision_id": RID,
        "contract_hash": "b" * 64,
        "artifact_hashes": {"part.step": "c" * 64},
        "engine_fingerprint": "d" * 64,
        "geometry_valid": True,
        "requirements_satisfied": False,
        "facts": {"volume_mm3": 123.12345678901234, "private_large_fact": list(range(100))},
        "findings": [
            finding(status, i) for i, status in enumerate(("pass", "fail", "unverified", "fail"))
        ],
        "errors": ["Full error, including\nits second line."],
        "caveats": ["Nominal geometry does not establish physical fit."],
    }


def test_actionable_view_preserves_exact_findings_provenance_and_limits() -> None:
    original = measurement()
    summary = evidence_summary(original, RID, integrity_scope="evidence_record")
    assert [row.model_dump() for row in summary.findings] == original["findings"][1:]
    assert summary.counts == {"pass": 1, "fail": 2, "unverified": 1}
    assert summary.total_findings == 4
    assert summary.filtered_count == summary.returned_count == 3
    assert summary.omitted_pass_count == 1
    assert summary.page_complete
    assert not summary.all_findings_included
    assert summary.next_offset is None
    assert summary.errors == original["errors"]
    assert summary.caveats == original["caveats"]
    assert summary.geometry_valid is True
    assert summary.requirements_satisfied is False
    for key in ("revision_id", "contract_hash", "engine_fingerprint", "artifact_hashes"):
        assert summary.model_dump()[key] == original[key]
    assert "facts" not in summary.model_dump()
    assert "does not accept a revision" in summary.scope


@pytest.mark.parametrize("statuses,expected_count", [("all", 4), ("actionable", 3)])
def test_pages_preserve_order_and_complete_filter_counts(
    statuses: str, expected_count: int
) -> None:
    original = measurement()
    expected = [r for r in original["findings"] if statuses == "all" or r["status"] != "pass"]
    rows = []
    offset = 0
    while True:
        page = evidence_summary(
            original,
            RID,
            statuses=statuses,
            offset=offset,
            limit=2,
            integrity_scope="revision_artifact_hashes",
        )
        rows.extend(row.model_dump() for row in page.findings)
        assert page.counts == {"pass": 1, "fail": 2, "unverified": 1}
        assert page.total_findings == 4
        assert page.filtered_count == expected_count
        assert page.returned_count <= 2
        assert not page.all_findings_included
        assert not page.page_complete
        if page.next_offset is None:
            break
        assert page.next_offset > offset
        offset = page.next_offset
    assert rows == expected


def test_full_page_and_offset_beyond_end_have_explicit_completeness() -> None:
    original = measurement()
    full = evidence_summary(original, RID, statuses="all", integrity_scope="evidence_record")
    assert full.page_complete and full.all_findings_included
    assert full.omitted_pass_count == 0
    empty = evidence_summary(
        original, RID, statuses="all", offset=20, integrity_scope="evidence_record"
    )
    assert empty.findings == []
    assert empty.returned_count == 0 and empty.next_offset is None
    assert empty.filtered_count == 4
    assert not empty.page_complete and not empty.all_findings_included
    assert empty.omitted_pass_count == 1


def test_projection_and_summary_mutations_do_not_change_input() -> None:
    original = measurement()
    before = copy.deepcopy(original)
    summary = evidence_summary(original, RID, integrity_scope="evidence_record")
    summary_text(summary)
    summary.findings[0].reason = "changed"
    summary.findings[0].locator[0] = 42
    summary.errors.append("changed")
    summary.caveats.append("changed")
    summary.artifact_hashes["part.step"] = "changed"
    assert original == before


def test_integer_numbers_and_null_values_remain_exact() -> None:
    original = measurement()
    original["findings"][1].update(
        measured=2**53 + 1, expected=None, locator=[2**53 + 1, 0, 7.12345678901234]
    )
    summary = evidence_summary(original, RID, integrity_scope="evidence_record")
    assert summary.findings[0].model_dump() == original["findings"][1]
    assert isinstance(summary.findings[0].measured, int)
    row = next(
        line for line in summary_text(summary).splitlines() if line.startswith('{"requirement_id"')
    )
    assert json.loads(row) == original["findings"][1]


@pytest.mark.parametrize("failed", [False, True])
def test_empty_evidence_does_not_imply_geometry_or_requirement_success(failed: bool) -> None:
    original = measurement()
    original.update(
        findings=[], artifact_hashes={}, geometry_valid=not failed, requirements_satisfied=False
    )
    summary = evidence_summary(original, RID, integrity_scope="evidence_record")
    assert summary.counts == {"pass": 0, "fail": 0, "unverified": 0}
    assert summary.total_findings == summary.filtered_count == summary.returned_count == 0
    assert summary.page_complete and summary.all_findings_included
    assert summary.artifact_hashes == {}
    assert summary.geometry_valid is not failed
    assert not summary.requirements_satisfied
    assert summary.errors == original["errors"]


def test_text_preserves_page_values_errors_caveats_and_complete_counts() -> None:
    original = measurement()
    summary = evidence_summary(original, RID, limit=1, integrity_scope="evidence_record")
    text = summary_text(summary)
    exact_rows = [
        json.loads(line) for line in text.splitlines() if line.startswith('{"requirement_id"')
    ]
    assert exact_rows == [original["findings"][1]]
    assert "All findings: 4; pass=1; fail=2; unverified=1" in text
    assert "next_offset=1" in text and "omitted_pass_count=1" in text
    assert "all_findings_included=False" in text
    assert "Integrity scope=evidence_record" in text
    assert "Errors: " + json.dumps(original["errors"]) in text
    assert "Caveats: " + json.dumps(original["caveats"]) in text
    assert "get_evidence" in text
    assert original["artifact_hashes"]["part.step"] not in text


def test_wrong_revision_is_rejected() -> None:
    with pytest.raises(ValueError, match="requested revision"):
        evidence_summary(measurement(), "e" * 32, integrity_scope="evidence_record")


def test_no_actionable_findings_does_not_hide_pass_omissions() -> None:
    original = measurement()
    original["findings"] = [finding("pass", 0)]
    summary = evidence_summary(original, RID, integrity_scope="evidence_record")
    assert summary.findings == []
    assert summary.total_findings == summary.omitted_pass_count == 1
    assert summary.filtered_count == summary.returned_count == 0
    assert summary.page_complete and not summary.all_findings_included
    assert not summary.requirements_satisfied


@pytest.mark.parametrize(
    "scope,meaning",
    [
        ("evidence_record", "does not rehash artifacts"),
        ("revision_artifact_hashes", "all recorded revision artifact hashes"),
    ],
)
def test_text_explains_transport_integrity_without_claiming_new_checks(
    scope: str, meaning: str
) -> None:
    summary = evidence_summary(measurement(), RID, integrity_scope=scope)
    text = summary_text(summary)
    assert meaning in text
    assert "selected filter is fully included" in text
    assert "full findings list is included" in text


@pytest.mark.parametrize("missing", list(measurement()))
def test_missing_measurement_fields_are_not_invented(missing: str) -> None:
    original = measurement()
    del original[missing]
    with pytest.raises(ValueError, match="incomplete"):
        evidence_summary(original, RID, integrity_scope="evidence_record")


@pytest.mark.parametrize("missing", list(finding("fail", 0)))
def test_missing_finding_fields_are_not_invented(missing: str) -> None:
    original = measurement()
    del original["findings"][1][missing]
    with pytest.raises(ValueError):
        evidence_summary(original, RID, integrity_scope="evidence_record")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("location", ["measured", "locator", "facts"])
def test_nonfinite_values_are_rejected_even_in_omitted_facts(value: float, location: str) -> None:
    original = measurement()
    if location == "facts":
        original["facts"]["nested"] = {"omitted": [value]}
    elif location == "locator":
        original["findings"][1]["locator"][0] = value
    else:
        original["findings"][1]["measured"] = value
    with pytest.raises(ValueError, match="finite JSON"):
        evidence_summary(original, RID, integrity_scope="evidence_record")


@pytest.mark.parametrize(
    "field,value",
    [
        ("geometry_valid", "false"),
        ("findings", {}),
        ("artifact_hashes", {"part.step": 42}),
        ("errors", [42]),
        ("requirements_satisfied", 1),
        ("unexpected", True),
    ],
)
def test_malformed_measurements_are_rejected(field: str, value: Any) -> None:
    original = measurement()
    original[field] = value
    with pytest.raises(ValidationError):
        evidence_summary(original, RID, integrity_scope="evidence_record")


@pytest.mark.parametrize(
    "arguments",
    [
        {"offset": -1},
        {"offset": True},
        {"limit": 0},
        {"limit": 101},
        {"limit": 2.5},
        {"statuses": "pass"},
        {"integrity_scope": "all_verified"},
        {"expected_revision": "../bad"},
    ],
)
def test_invalid_selection_is_rejected(arguments: dict[str, Any]) -> None:
    values = {"expected_revision": RID, "integrity_scope": "evidence_record", **arguments}
    with pytest.raises(ValidationError):
        evidence_summary(measurement(), **values)
