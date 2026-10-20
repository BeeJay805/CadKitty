"""Exact navigation over reports already loaded by the owned report reader.

This pure module performs no IO, recomputation, ownership or artifact checks.
An overview is a directory, not a complete inventory of nested failures.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

ReportView = Literal["full", "overview", "section"]
ReportSectionName = Literal[
    "production",
    "assembly_definitions",
    "inputs",
    "geometry",
    "references",
    "assembly",
    "tolerances",
    "manufacturing",
    "simulation",
    "mechanism",
    "bom",
    "mass_kg",
    "physical_feedback",
    "assembly_motion",
    "inspection_plan",
    "drawings",
    "section",
    "reviews",
    "open_items",
]
SECTIONS: tuple[ReportSectionName, ...] = (
    "production",
    "assembly_definitions",
    "inputs",
    "geometry",
    "references",
    "assembly",
    "tolerances",
    "manufacturing",
    "simulation",
    "mechanism",
    "bom",
    "mass_kg",
    "physical_feedback",
    "assembly_motion",
    "inspection_plan",
    "drawings",
    "section",
    "reviews",
    "open_items",
)
IDENTITY_FIELDS = (
    "id",
    "revision_id",
    "input_version",
    "input_hash",
    "geometry_hash",
    "engine_fingerprint",
    "frame",
    "at",
)


class _Exact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReportIdentity(_Exact):
    id: str
    revision_id: str
    input_version: int
    input_hash: str
    geometry_hash: str
    engine_fingerprint: str
    frame: str
    at: int | float


class ReportSectionEntry(_Exact):
    name: str
    json_type: Literal["object", "array", "string", "number", "boolean", "null"]
    direct_count: int | None
    count_scope: Literal["direct_object_keys", "direct_array_items", "not_a_collection"]
    value_json_utf8_bytes: int
    included_in_overview: bool
    available_via_section: bool
    full_reader_required: bool


class EngineeringReportOverview(ReportIdentity):
    projection: Literal["engineering_report_overview"] = "engineering_report_overview"
    reviews: list[dict[str, JsonValue]]
    open_items: list[str]
    sections: list[ReportSectionEntry]
    omitted_sections: list[str]
    unavailable_sections: list[str]
    unknown_top_level_fields: list[str]
    nested_failure_inventory_complete: Literal[False] = False
    scope: str = (
        "Exact identity, complete reviews and open_items, and top-level section directory only. "
        "All other section values are omitted. Counts describe direct keys/items, never findings, "
        "failed checks, bodies, pairs or samples. Nested failures, witnesses, errors and coverage "
        "are not inventoried here. Read view=section with a listed available section for its "
        "entire original value, or view=full for the complete report and unknown fields. "
        "No recomputation, verdict change, new integrity check or acceptance. Complete reviews "
        "and open_items can be large; this view is not guaranteed to have a fixed small size."
    )


class EngineeringReportSection(ReportIdentity):
    projection: Literal["engineering_report_section"] = "engineering_report_section"
    section: ReportSectionName
    value: JsonValue
    value_json_utf8_bytes: int
    section_complete: Literal[True] = True
    omitted_sections: list[str]
    scope: str = (
        "The entire original requested top-level section value, with all of its original "
        "fields, witnesses, errors and coverage. Other sections are omitted; this is not the "
        "complete report. No clipping, pagination, recomputation, verdict change, new integrity "
        "check or acceptance. Large sections remain large; view=full retains the full reader."
    )


def validate_report_view(view: ReportView, section: ReportSectionName | None) -> None:
    """Reject invalid navigation before the caller loads a report."""
    if view not in ("full", "overview", "section"):
        raise ValueError("Unknown engineering report view")
    if view == "section":
        if section not in SECTIONS:
            raise ValueError("Section view requires a supported section name")
    elif section is not None:
        raise ValueError("Section is only valid with view=section")


def _json_bytes(value: Any) -> int:
    try:
        return len(
            json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Engineering report must contain finite JSON values") from exc


def _identity(report: dict[str, Any], expected_report_id: str) -> ReportIdentity:
    if not isinstance(report, dict):
        raise ValueError("Engineering report must be an object")
    _json_bytes(report)
    if any(not isinstance(key, str) for key in report):
        raise ValueError("Engineering report must have string keys")
    for key, length in (
        ("id", 32),
        ("revision_id", 32),
        ("input_hash", 64),
        ("geometry_hash", 64),
        ("engine_fingerprint", 64),
    ):
        value = report.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{" + str(length) + "}", value):
            raise ValueError("Engineering report has malformed identity field: " + key)
    if report["id"] != expected_report_id:
        raise ValueError("Engineering report does not match requested report ID")
    if type(report.get("input_version")) is not int or report["input_version"] < 0:
        raise ValueError("Engineering report has malformed input_version")
    if not isinstance(report.get("frame"), str) or not report["frame"]:
        raise ValueError("Engineering report has malformed frame")
    try:
        valid_at = type(report.get("at")) in (int, float) and math.isfinite(report["at"])
    except OverflowError:
        valid_at = False
    if not valid_at:
        raise ValueError("Engineering report has malformed at")
    reviews = report.get("reviews")
    if not isinstance(reviews, list) or any(
        not isinstance(row, dict)
        or any(
            not isinstance(row.get(key), str) or not row[key]
            for key in ("category", "status", "method")
        )
        for row in reviews
    ):
        raise ValueError("Engineering report has malformed reviews")
    if not isinstance(report.get("open_items"), list) or any(
        not isinstance(item, str) for item in report["open_items"]
    ):
        raise ValueError("Engineering report has malformed open_items")
    return ReportIdentity.model_validate({key: report[key] for key in IDENTITY_FIELDS})


def _entry(name: str, value: JsonValue) -> ReportSectionEntry:
    kind: Literal["object", "array", "string", "number", "boolean", "null"]
    count: int | None = None
    scope: Literal["direct_object_keys", "direct_array_items", "not_a_collection"] = (
        "not_a_collection"
    )
    if isinstance(value, dict):
        kind = "object"
        count, scope = len(value), "direct_object_keys"
    elif isinstance(value, list):
        kind = "array"
        count, scope = len(value), "direct_array_items"
    elif value is None:
        kind = "null"
    elif isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, str):
        kind = "string"
    else:
        kind = "number"
    return ReportSectionEntry(
        name=name,
        json_type=kind,
        direct_count=count,
        count_scope=scope,
        value_json_utf8_bytes=_json_bytes(value),
        included_in_overview=name in ("reviews", "open_items"),
        available_via_section=name in SECTIONS,
        full_reader_required=name not in SECTIONS,
    )


def engineering_report_overview(
    report: dict[str, Any], expected_report_id: str
) -> EngineeringReportOverview:
    identity = _identity(report, expected_report_id)
    names = [key for key in report if key not in IDENTITY_FIELDS]
    return EngineeringReportOverview(
        **identity.model_dump(),
        reviews=deepcopy(report["reviews"]),
        open_items=deepcopy(report["open_items"]),
        sections=[_entry(name, report[name]) for name in names],
        omitted_sections=[name for name in names if name not in ("reviews", "open_items")],
        unavailable_sections=[name for name in SECTIONS if name not in report],
        unknown_top_level_fields=[name for name in names if name not in SECTIONS],
    )


def engineering_report_section(
    report: dict[str, Any], expected_report_id: str, section: ReportSectionName
) -> EngineeringReportSection:
    validate_report_view("section", section)
    identity = _identity(report, expected_report_id)
    if section not in report:
        raise ValueError("Requested engineering report section is absent: " + section)
    return EngineeringReportSection(
        **identity.model_dump(),
        section=section,
        value=deepcopy(report[section]),
        value_json_utf8_bytes=_json_bytes(report[section]),
        omitted_sections=[key for key in report if key not in IDENTITY_FIELDS and key != section],
    )


def report_view_text(result: EngineeringReportOverview | EngineeringReportSection) -> str:
    identity = (
        f"Report {result.id}; revision {result.revision_id}; input version {result.input_version}."
    )
    if isinstance(result, EngineeringReportOverview):
        return (
            identity
            + " Complete reviews and open_items are in structured content. Other section values and nested failure details are omitted. Use view=section and a listed section, or view=full."
        )
    return (
        identity
        + f" Entire section {result.section} is in structured content ({result.value_json_utf8_bytes} compact UTF-8 JSON bytes). Other sections are omitted. Use view=full for the complete report."
    )
