"""Offline, evidence-bound trial assessments. Runtime is never human labor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Method = Literal["observed", "instrumented", "self_reported", "synthetic", "model_inference"]
MEASURED = {"observed", "instrumented"}
LIMITATION = (
    "Hashes verify referenced bytes, not the truth or completeness of recorded observations. "
    "Acceptance requires the recorded external assessment; model completion and geometry "
    "validity do not establish it. Task labor savings do not establish staffing reductions."
)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(Record):
    path: Text
    sha256: Sha256


class Provenance(Record):
    actor_id: Text
    method: Method
    evidence: list[Text] = Field(min_length=1)
    note: Text


class Actor(Record):
    id: Text
    role: Text
    kind: Literal["human", "system"]


class CriterionDefinition(Record):
    id: Text
    description: Text


class Acceptance(Record):
    criteria: list[CriterionDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self) -> Acceptance:
        if len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("Acceptance criterion IDs must be unique")
        return self


class Criterion(Record):
    id: Text
    outcome: Literal["passed", "failed", "unassessed"]
    assessment: Provenance


class Coverage(Record):
    status: Literal["complete", "partial", "unknown"]
    provenance: Provenance
    gaps: list[Text]
    empty_confirmed: bool = False

    @model_validator(mode="after")
    def consistent(self) -> Coverage:
        if self.status == "complete" and self.gaps:
            raise ValueError("Complete coverage cannot have gaps")
        return self


class Interval(Record):
    actor_id: Text
    activity: Text
    start_seconds: Seconds
    end_seconds: Seconds
    provenance: Provenance

    @model_validator(mode="after")
    def ordered(self) -> Interval:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Human work intervals must have positive duration")
        return self


class Intervention(Record):
    actor_id: Text
    activity: Text
    reason: Text
    provenance: Provenance


class Accounting(Record):
    boundary: Text
    included_activities: list[Text] = Field(min_length=1)
    excluded_activities: list[Text]

    @model_validator(mode="after")
    def disjoint(self) -> Accounting:
        if set(self.included_activities) & set(self.excluded_activities):
            raise ValueError("An activity cannot be included and excluded")
        return self


class Assessment(Record):
    schema_version: Literal[1] = 1
    assessment_id: Text
    attempt_id: Text
    supersedes: Evidence | None = None
    data_origin: Literal["observed", "self_reported", "synthetic"]
    method: Text
    tools_and_resources: list[Text] = Field(min_length=1)
    evidence: dict[Text, Evidence]
    task_evidence: Text
    acceptance_evidence: Text
    starting_inputs_evidence: Text
    actors: list[Actor] = Field(min_length=1)
    criteria: list[Criterion] = Field(min_length=1)
    accounting: Accounting
    intervals: list[Interval]
    human_time_coverage: Coverage
    interventions: list[Intervention]
    intervention_coverage: Coverage
    machine_runtime_seconds: Seconds | None = None

    @model_validator(mode="after")
    def references(self) -> Assessment:
        actors = {actor.id: actor for actor in self.actors}
        if len(actors) != len(self.actors):
            raise ValueError("Actor IDs must be unique")
        if len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("Criterion results must be unique")
        for ref in (self.task_evidence, self.acceptance_evidence, self.starting_inputs_evidence):
            if ref not in self.evidence:
                raise ValueError(f"Unknown frozen evidence reference: {ref}")
        provenance = [item.assessment for item in self.criteria]
        provenance.extend(
            [self.human_time_coverage.provenance, self.intervention_coverage.provenance]
        )
        work: list[Interval | Intervention] = [*self.intervals, *self.interventions]
        for item in work:
            if item.actor_id not in actors or actors[item.actor_id].kind != "human":
                raise ValueError("Human work must identify a declared human actor")
            if item.activity not in self.accounting.included_activities:
                raise ValueError("Recorded activity is outside the accounting boundary")
            provenance.append(item.provenance)
        for observed in provenance:
            if observed.actor_id not in actors or any(
                ref not in self.evidence for ref in observed.evidence
            ):
                raise ValueError("Provenance must reference declared actors and evidence")
        return self


class Comparison(Record):
    schema_version: Literal[1] = 1
    baseline: Evidence
    candidate: Evidence
    comparable: bool
    assessor: Text
    method: Method
    rationale: Text
    evidence: list[Evidence] = Field(min_length=1)


def strict_json(value: bytes) -> Any:
    """Reject ambiguous keys and nonfinite numbers before interpreting evidence metadata."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"Non-finite JSON number: {value}")

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite JSON number: {value}")
        return result

    return json.loads(
        value,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def verified_path(reference: Evidence, base: Path) -> Path:
    path = base / reference.path
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"Evidence must be a regular file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"Evidence is not a regular file: {path}")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != reference.sha256:
        raise ValueError(f"Evidence hash changed: {path}")
    return path


def verified_bytes(reference: Evidence, base: Path) -> bytes:
    value = verified_path(reference, base).read_bytes()
    if hashlib.sha256(value).hexdigest() != reference.sha256:
        raise ValueError("Evidence changed before JSON parsing")
    return value


def load_assessment(path: Path, expected_hash: str | None = None) -> Assessment:
    value = path.read_bytes()
    if expected_hash is not None and hashlib.sha256(value).hexdigest() != expected_hash:
        raise ValueError("Assessment changed before JSON parsing")
    record = Assessment.model_validate(strict_json(value))
    for reference in record.evidence.values():
        verified_path(reference, path.parent)
    if record.supersedes:
        verified_path(record.supersedes, path.parent)
    manifest = Acceptance.model_validate(
        strict_json(verified_bytes(record.evidence[record.acceptance_evidence], path.parent))
    )
    if {item.id for item in manifest.criteria} != {item.id for item in record.criteria}:
        raise ValueError("Criterion results must cover the entire frozen acceptance manifest")
    for reference in [
        *record.evidence.values(),
        *([record.supersedes] if record.supersedes else []),
    ]:
        reference.path = str((path.parent / reference.path).resolve())
    return record


def coverage_complete(coverage: Coverage, count: int) -> bool:
    return (
        coverage.status == "complete"
        and coverage.provenance.method in MEASURED
        and (count > 0 or coverage.empty_confirmed)
    )


def summarize(record: Assessment) -> dict[str, Any]:
    totals: dict[str, float] = {}
    overlap_actors = []
    for actor in record.actors:
        intervals = sorted(
            (item.start_seconds, item.end_seconds)
            for item in record.intervals
            if item.actor_id == actor.id
        )
        merged: list[tuple[float, float]] = []
        for start, end in intervals:
            if merged and start < merged[-1][1]:
                overlap_actors.append(actor.id)
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        if actor.kind == "human":
            totals[actor.id] = math.fsum(end - start for start, end in merged)
    total = math.fsum(totals.values())
    if not math.isfinite(total):
        raise ValueError("Human time total is not finite")
    reasons = []
    if record.data_origin != "observed":
        reasons.append(f"Data origin is {record.data_origin}, not observed")
    acceptance = record.data_origin == "observed" and all(
        item.outcome == "passed" and item.assessment.method in MEASURED for item in record.criteria
    )
    if not acceptance:
        reasons.append(
            "Frozen acceptance is not fully passed with external observation or instrumentation"
        )
    timing_complete = coverage_complete(record.human_time_coverage, len(record.intervals))
    if not timing_complete:
        reasons.append(
            "Human time coverage is incomplete, unmeasured, or empty without confirmation"
        )
    if any(item.provenance.method not in MEASURED for item in record.intervals):
        reasons.append("Human work intervals include unmeasured timing")
    interventions_complete = coverage_complete(
        record.intervention_coverage, len(record.interventions)
    )
    if not interventions_complete:
        reasons.append(
            "Intervention coverage is incomplete, unmeasured, or empty without confirmation"
        )
    if any(item.provenance.method not in MEASURED for item in record.interventions):
        reasons.append("Interventions include unmeasured records")
    timed_work = {(item.actor_id, item.activity) for item in record.intervals}
    if any((item.actor_id, item.activity) not in timed_work for item in record.interventions):
        reasons.append(
            "Recorded human intervention has no timed interval for its actor and activity"
        )
    return {
        "assessment_id": record.assessment_id,
        "attempt_id": record.attempt_id,
        "data_origin": record.data_origin,
        "method": record.method,
        "tools_and_resources": record.tools_and_resources,
        "actors": [item.model_dump() for item in record.actors],
        "criteria": [item.model_dump() for item in record.criteria],
        "accounting": record.accounting.model_dump(),
        "evidence": {key: value.model_dump() for key, value in record.evidence.items()},
        "frozen_evidence": {
            name: record.evidence[getattr(record, name)].model_dump()
            for name in ("task_evidence", "acceptance_evidence", "starting_inputs_evidence")
        },
        "declared_acceptance_supported": acceptance,
        "recorded_human_seconds": total,
        "intervals": [item.model_dump() for item in record.intervals],
        "human_seconds_by_actor": totals,
        "overlaps_merged_for_actors": sorted(set(overlap_actors)),
        "timing_methods": sorted({item.provenance.method for item in record.intervals}),
        "human_time_coverage": record.human_time_coverage.model_dump(),
        "recorded_interventions": len(record.interventions),
        "interventions": [item.model_dump() for item in record.interventions],
        "intervention_coverage": record.intervention_coverage.model_dump(),
        "intervention_count": len(record.interventions) if interventions_complete else None,
        "machine_runtime_seconds": record.machine_runtime_seconds,
        "measurement_eligible": not reasons,
        "ineligibility_reasons": reasons,
        "labor_savings": None,
        "labor_savings_reasons": ["No matched baseline comparison supplied"],
        "limitations": LIMITATION,
    }


def compare(path: Path) -> dict[str, Any]:
    pair = Comparison.model_validate(strict_json(path.read_bytes()))
    baseline_path = verified_path(pair.baseline, path.parent)
    candidate_path = verified_path(pair.candidate, path.parent)
    for item in pair.evidence:
        verified_path(item, path.parent)
    baseline = load_assessment(baseline_path, pair.baseline.sha256)
    candidate = load_assessment(candidate_path, pair.candidate.sha256)
    for reference in [pair.baseline, pair.candidate, *pair.evidence]:
        reference.path = str((path.parent / reference.path).resolve())
    before, after = summarize(baseline), summarize(candidate)
    reasons = [
        f"{label}: {reason}"
        for label, result in (("baseline", before), ("candidate", after))
        for reason in result["ineligibility_reasons"]
    ]
    if baseline.attempt_id == candidate.attempt_id or pair.baseline.sha256 == pair.candidate.sha256:
        reasons.append("Comparison requires distinct attempts")
    for name in ("task_evidence", "acceptance_evidence", "starting_inputs_evidence"):
        if (
            baseline.evidence[getattr(baseline, name)].sha256
            != candidate.evidence[getattr(candidate, name)].sha256
        ):
            reasons.append(f"Frozen {name} differs")
    if baseline.accounting.model_dump() != candidate.accounting.model_dump():
        reasons.append("Human accounting boundaries differ")
    if not pair.comparable or pair.method not in MEASURED:
        reasons.append("Comparison lacks an observed or instrumented comparability assessment")
    if before["recorded_human_seconds"] <= 0:
        reasons.append("Baseline human labor must be positive")
    savings = None
    if not reasons:
        seconds = before["recorded_human_seconds"] - after["recorded_human_seconds"]
        savings = {
            "human_seconds": seconds,
            "fraction_of_baseline": seconds / before["recorded_human_seconds"],
        }
    return {
        "baseline": before,
        "candidate": after,
        "comparability": pair.model_dump(),
        "labor_savings": savings,
        "labor_savings_reasons": reasons,
        "limitations": LIMITATION,
    }


def write_json(value: Any, output: Path | None) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(content, end="")
    else:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema", help="Print the strict JSON schema")
    schema.add_argument(
        "kind", choices=["assessment", "comparison", "acceptance"], default="assessment", nargs="?"
    )
    for command in ("assess", "compare", "record"):
        sub = commands.add_parser(command)
        sub.add_argument("input", type=Path)
        sub.add_argument("--output", type=Path, required=command == "record")
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            models: dict[str, type[Record]] = {
                "assessment": Assessment,
                "comparison": Comparison,
                "acceptance": Acceptance,
            }
            write_json(models[args.kind].model_json_schema(), None)
        elif args.command == "compare":
            write_json(compare(args.input), args.output)
        else:
            record = load_assessment(args.input)
            if args.command == "record":
                write_json(record.model_dump(), args.output)
            else:
                write_json(summarize(record), args.output)
    except (OSError, ValueError, OverflowError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
