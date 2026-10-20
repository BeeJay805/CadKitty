"""Persistent design experiments with frozen protocols and independently measured candidates."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.design_evaluators import evaluate_design
from forge_mind.workspace.design_search import choose_search_anchor
from forge_mind.workspace.design_study_identity import (
    bind_targets,
    instrument_identity,
    remap_evaluators,
)
from forge_mind.workspace.design_study_models import (
    AcceptStudyCandidate,
    AddStudyCandidate,
    CreateDesignStudy,
    DesignStudySpec,
    SearchDesignStudy,
)
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.lab_records import capture_binding
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import Contract
from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.product_sources import read_sources
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _tables(service: WorkspaceService) -> None:
    with service.store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS design_studies(id TEXT PRIMARY KEY,session TEXT NOT NULL,document TEXT NOT NULL,sha256 TEXT NOT NULL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS design_candidates(id TEXT PRIMARY KEY,study TEXT NOT NULL,input_key TEXT NOT NULL,document TEXT NOT NULL,sha256 TEXT NOT NULL,UNIQUE(study,input_key))"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS design_study_runs(id TEXT PRIMARY KEY,study TEXT NOT NULL,document TEXT NOT NULL,sha256 TEXT NOT NULL)"
        )


def _decode(row: Any) -> dict[str, Any]:
    if row is None:
        raise WorkspaceError("Design study record not found")
    result: dict[str, Any] = json.loads(row[0])
    if digest(result) != row[1]:
        raise WorkspaceError("Design study record integrity check failed")
    return result


def _study(service: WorkspaceService, sid: str, owner: str, study_id: str) -> dict[str, Any]:
    service.store.session_record(sid, owner)
    _tables(service)
    with service.store.connect() as con:
        return _decode(
            con.execute(
                "SELECT document,sha256 FROM design_studies WHERE id=? AND session=?",
                (study_id, sid),
            ).fetchone()
        )


def _candidates(service: WorkspaceService, study_id: str) -> list[dict[str, Any]]:
    with service.store.connect() as con:
        return [
            _decode(r)
            for r in con.execute(
                "SELECT document,sha256 FROM design_candidates WHERE study=? ORDER BY rowid",
                (study_id,),
            )
        ]


def _save_candidate(service: WorkspaceService, candidate: dict[str, Any]) -> None:
    with service.store.connect() as con:
        con.execute(
            "UPDATE design_candidates SET document=?,sha256=? WHERE id=? AND study=?",
            (
                json.dumps(candidate, allow_nan=False),
                digest(candidate),
                candidate["id"],
                candidate["study_id"],
            ),
        )


def _candidate_engine(study: dict[str, Any], revision: dict[str, Any]) -> None:
    if (
        revision.get("measurement", {}).get("engine_fingerprint")
        != study["binding"]["engine_fingerprint"]
    ):
        raise WorkspaceError(
            "Candidate measurement uses different instruments than the frozen study; rebuild the candidate before comparison"
        )


def _fresh(service: WorkspaceService, sid: str, owner: str, study: dict[str, Any]) -> None:
    service.store.session_record(sid, owner)
    engineering = service.store.engineering(sid)
    binding = study["binding"]
    if (
        binding["engine_fingerprint"] != engine_fingerprint()
        or binding["input_version"] != engineering["version"]
        or binding["input_hash"] != digest(engineering["inputs"])
        or binding["source_fingerprint"] != read_sources(service.store, sid, owner)["fingerprint"]
    ):
        raise WorkspaceError(
            "Study instruments, engineering inputs or sources changed; create a new comparison protocol"
        )
    try:
        instruments = instrument_identity(DesignStudySpec.model_validate(study["spec"]))
    except (ValueError, OSError) as exc:
        raise WorkspaceError("Study instrument is unavailable: " + str(exc)) from exc
    if instruments != study["instruments"]:
        raise WorkspaceError("Study external instruments changed; create a new comparison protocol")
    revision = service._load_revision(sid, study["baseline_revision_id"])
    if (
        digest(revision["document"]) != binding["document_hash"]
        or revision["measurement"]["artifact_hashes"]["part.step"] != binding["geometry_hash"]
        or digest(revision.get("contract") or {}) != binding["contract_hash"]
    ):
        raise WorkspaceError("Study baseline changed")


def create_study(
    service: WorkspaceService,
    sid: str,
    owner: str,
    request: CreateDesignStudy,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    binding = capture_binding(service, sid, owner, request.revision_id, request.input_version)
    revision = service._load_revision(sid, request.revision_id)
    if revision["measurement"].get("engine_fingerprint") != engine_fingerprint():
        raise WorkspaceError(
            "Rebuild historical geometry with current instruments before creating a study"
        )
    document = DesignDocument.model_validate(revision["document"])
    known_requirements = {row["id"] for row in revision["contract"].get("requirements", [])}
    for evaluator in request.spec.evaluators:
        if evaluator.kind == "interface_geometry":
            for pair in evaluator.settings["pairs"]:
                if set(pair["requirement_ids"]) - known_requirements:
                    raise WorkspaceError("Interface study refers to an unknown frozen requirement")
    if set(request.spec.variables) - document.variables.keys():
        raise WorkspaceError("Search bounds must name existing editable input variables")
    if any(
        not lo <= document.variables[name] <= hi
        for name, (lo, hi) in request.spec.variables.items()
    ):
        raise WorkspaceError("Baseline parameters must lie inside the frozen search bounds")
    result: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "name": request.spec.name,
        "created_at": time.time(),
        "baseline_revision_id": request.revision_id,
        "input_version": request.input_version,
        "spec": request.spec.model_dump(),
        "spec_hash": digest(request.spec.model_dump()),
        "binding": binding,
        "instruments": instrument_identity(request.spec),
        "targets": bind_targets(
            document,
            service.store.directory(sid, request.revision_id) / "part.step",
            request.spec,
            cancelled,
        ),
        "baseline_parameters": document.variables,
        "scope": "Best feasible evaluated designs under one frozen numerical protocol. Study objectives do not replace original engineering requirements or establish physical qualification.",
    }
    _fresh(service, sid, owner, result)
    key = digest({"document": document.model_dump(), "spec": result["spec_hash"]})
    baseline = _candidate_record(
        result["id"], key, "Baseline", document.variables, request.revision_id
    )
    _tables(service)
    with service.store.connect() as con:
        con.execute(
            "INSERT INTO design_studies VALUES (?,?,?,?)",
            (result["id"], sid, json.dumps(result, allow_nan=False), digest(result)),
        )
        con.execute(
            "INSERT INTO design_candidates VALUES (?,?,?,?,?)",
            (
                baseline["id"],
                result["id"],
                key,
                json.dumps(baseline, allow_nan=False),
                digest(baseline),
            ),
        )
    service.store.event(
        sid, "design_study", "Frozen design study: " + result["name"], study_id=result["id"]
    )
    return read_study(service, sid, owner, result["id"])


def _candidate_record(
    study_id: str, key: str, label: str, parameters: dict[str, float], revision_id: str | None
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "study_id": study_id,
        "input_key": key,
        "label": label,
        "parameters": parameters,
        "revision_id": revision_id,
        "source_revision_id": revision_id,
        "status": "pending",
        "eligible": False,
        "feasible": None,
        "metrics": {},
        "evaluations": [],
        "attempts": [],
        "created_at": time.time(),
    }


def add_candidate(
    service: WorkspaceService, sid: str, owner: str, study_id: str, request: AddStudyCandidate
) -> dict[str, Any]:
    study = _study(service, sid, owner, study_id)
    _fresh(service, sid, owner, study)
    spec = DesignStudySpec.model_validate(study["spec"])
    parameters = dict(study["baseline_parameters"])
    if request.parameters is not None:
        if set(request.parameters) - set(spec.variables):
            raise WorkspaceError("Candidate changes variables outside the frozen search space")
        parameters.update(request.parameters)
        revision_id = None
    else:
        revision = service._load_revision(sid, str(request.revision_id))
        _candidate_engine(study, revision)
        if not revision["measurement"]["geometry_valid"]:
            raise WorkspaceError("Candidate revision has invalid exported geometry")
        if digest(revision.get("contract") or {}) != study["binding"]["contract_hash"]:
            raise WorkspaceError("Candidate must preserve the original acceptance contract")
        parameters = DesignDocument.model_validate(revision.get("document")).variables
        revision_id = request.revision_id
    if any(
        name not in parameters or not lo <= parameters[name] <= hi
        for name, (lo, hi) in spec.variables.items()
    ):
        raise WorkspaceError("Candidate exceeds frozen variable bounds")
    if revision_id:
        candidate_document = revision["document"]
    else:
        baseline = service._load_revision(sid, study["baseline_revision_id"])
        candidate_document = {**baseline["document"], "variables": parameters}
    key = digest({"document": candidate_document, "spec": study["spec_hash"]})
    result = _candidate_record(study_id, key, request.label, parameters, revision_id)
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        existing = con.execute(
            "SELECT document,sha256 FROM design_candidates WHERE study=? AND input_key=?",
            (study_id, key),
        ).fetchone()
        if existing:
            return _decode(existing)
        con.execute(
            "INSERT INTO design_candidates VALUES (?,?,?,?,?)",
            (result["id"], study_id, key, json.dumps(result, allow_nan=False), digest(result)),
        )
    return result


def compare_candidates(
    spec: DesignStudySpec, candidates: list[dict[str, Any]], baseline_id: str
) -> dict[str, Any]:
    eligible = [
        c
        for c in candidates
        if c.get("eligible") is True
        and c.get("feasible") is True
        and c.get("status") == "complete"
        and all(
            isinstance(c.get("metrics", {}).get(o.metric), (int, float))
            and math.isfinite(c["metrics"][o.metric])
            for o in spec.objectives
        )
    ]
    result: dict[str, Any] = {
        "baseline_candidate_id": baseline_id,
        "best_candidate_id": None,
        "pareto_candidate_ids": [],
        "conclusion": "awaiting_evidence",
        "improvement_fraction": None,
        "scope": "Best feasible evaluated candidate, not a global optimum or physical certification.",
    }
    if any(e.kind == "flow" for e in spec.evaluators) and eligible:
        from forge_mind.workspace.flow_comparison import common_flow_validation

        baseline = next((c for c in eligible if c["id"] == baseline_id), None)
        comparable = [
            c
            for c in eligible
            if baseline is not None and common_flow_validation([baseline, c], baseline_id)
        ]
        result["screening_candidate_ids"] = [c["id"] for c in eligible if c not in comparable]
        if comparable and len(comparable) > 1:
            eligible = comparable
        if not common_flow_validation(eligible, baseline_id):
            result.update(
                conclusion="inconclusive",
                reason="Flow screening results need confirmed common mesh and domain levels before ranking",
            )
            return result
        from forge_mind.workspace.flow_comparison import rank_stable

        if baseline is not None and any(
            c["id"] != baseline_id and not rank_stable(c, baseline) for c in eligible
        ):
            result.update(
                conclusion="inconclusive",
                reason="Candidate ordering reverses or lacks evidence across the final controlled flow cross",
            )
            return result
    if not eligible:
        if candidates and all(
            c["status"] in {"complete", "failed", "incomplete"} for c in candidates
        ):
            result["conclusion"] = "no_feasible_candidate"
        return result

    def score(candidate: dict[str, Any]) -> tuple[float, ...]:
        return tuple(
            float(candidate["metrics"][o.metric]) * (1 if o.direction == "minimize" else -1)
            for o in spec.objectives
        )

    baseline = next((c for c in eligible if c["id"] == baseline_id), None)

    def improves(candidate: dict[str, Any]) -> bool:
        if baseline is None or not all(
            a <= b for a, b in zip(score(candidate), score(baseline), strict=True)
        ):
            return False
        if any(e.kind == "flow" for e in spec.evaluators):
            from forge_mind.workspace.flow_comparison import rank_stable

            if not rank_stable(candidate, baseline):
                return False
        return any(
            (baseline["metrics"][o.metric] - candidate["metrics"][o.metric])
            * (1 if o.direction == "minimize" else -1)
            > max(
                abs(baseline["metrics"][o.metric]) * spec.comparison_tolerance,
                sum(
                    c.get("metric_uncertainty", {}).get(o.metric, 0) for c in (baseline, candidate)
                ),
            )
            for o in spec.objectives
        )

    improved = [c for c in eligible if improves(c)]
    result["improved_candidate_ids"] = [c["id"] for c in improved]
    best = min(improved or eligible, key=score)
    result["best_candidate_id"] = best["id"]
    result["pareto_candidate_ids"] = [
        c["id"]
        for c in eligible
        if not any(
            all(a <= b for a, b in zip(score(other), score(c), strict=True))
            and any(a < b for a, b in zip(score(other), score(c), strict=True))
            for other in eligible
        )
    ]
    if baseline is None:
        result["conclusion"] = "inconclusive"
        result["reason"] = "Baseline has no comparable feasible evidence"
        return result
    if best["id"] == baseline_id:
        result["conclusion"] = "baseline_best"
        return result
    metric = spec.objectives[0].metric
    direction = 1 if spec.objectives[0].direction == "minimize" else -1
    improvement = direction * (baseline["metrics"][metric] - best["metrics"][metric])
    scale = max(abs(baseline["metrics"][metric]), 1e-12)
    result["improvement_fraction"] = improvement / scale
    stable = improves(best)
    result["conclusion"] = "improved" if stable else "inconclusive"
    result["reason"] = (
        "Improvement exceeds the frozen tolerance and observed numerical sensitivity"
        if stable
        else "Difference is too small, numerically uncertain, or trades against another objective"
    )
    return result


def candidate_directory(
    service: WorkspaceService, sid: str, study_id: str, candidate_id: str
) -> Path:
    # IDs originate from owned database records, never arbitrary request paths.
    # Globally unique candidate IDs avoid four nested UUIDs and Windows MAX_PATH failures.
    return service.store.root / "studies" / candidate_id


def _verify_evaluations(
    service: WorkspaceService, sid: str, study: dict[str, Any], candidate: dict[str, Any]
) -> None:
    if not candidate.get("revision_id"):
        return
    revisions: dict[str, dict[str, Any]] = {}

    def checked_revision(revision_id: str) -> dict[str, Any]:
        if revision_id not in revisions:
            revision = service._load_revision(sid, revision_id)
            _candidate_engine(study, revision)
            revisions[revision_id] = revision
        return revisions[revision_id]

    checked_revision(candidate["revision_id"])
    receipts = [
        *candidate.get("evaluations", []),
        *(r for a in candidate.get("attempts", []) for r in a.get("previous_evaluations", [])),
    ]
    for receipt in receipts:
        revision = checked_revision(receipt.get("revision_id", candidate["revision_id"]))
        geometry_hash = revision["measurement"]["artifact_hashes"].get("part.step")
        if receipt["geometry_hash"] != geometry_hash or receipt["spec_hash"] != study["spec_hash"]:
            raise WorkspaceError("Evaluation does not belong to this geometry and study protocol")
        directory = (
            candidate_directory(service, sid, study["id"], candidate["id"])
            / receipt["attempt"]
            / receipt["kind"]
        )
        for name, expected in receipt.get("hashes", {}).items():
            path = directory / name
            if (
                path.resolve().is_relative_to(directory.resolve()) is False
                or not regular_file(path)
                or file_sha256(path) != expected
            ):
                raise WorkspaceError("Design evaluation artifact changed: " + name)


def read_study(service: WorkspaceService, sid: str, owner: str, study_id: str) -> dict[str, Any]:
    study = _study(service, sid, owner, study_id)
    candidates = _candidates(service, study_id)
    for candidate in candidates:
        _verify_evaluations(service, sid, study, candidate)
    current, reason = True, None
    try:
        _fresh(service, sid, owner, study)
    except ValueError as exc:
        current, reason = False, str(exc)
    with service.store.connect() as con:
        row = con.execute(
            "SELECT document,sha256 FROM design_study_runs WHERE study=? ORDER BY rowid DESC LIMIT 1",
            (study_id,),
        ).fetchone()
        run = _decode(row) if row else None
        job = (
            con.execute("SELECT state FROM jobs WHERE id=?", (run["job_id"],)).fetchone()
            if run
            else None
        )
    state = run["state"] if run else "draft"
    if state in {"queued", "running"} and (job is None or job[0] not in {"queued", "running"}):
        state = "cancelled" if job and job[0] == "cancelled" else "interrupted"
    comparison = compare_candidates(
        DesignStudySpec.model_validate(study["spec"]),
        candidates,
        candidates[0]["id"] if candidates else "",
    )
    if not current:
        comparison.update(conclusion="inconclusive", reason=reason)
    return {
        **study,
        "candidates": candidates,
        "comparison": comparison,
        "current": current,
        "stale_reason": reason,
        "status": state,
        "job_id": run["job_id"] if run else None,
        "run": run,
    }


def list_studies(service: WorkspaceService, sid: str, owner: str) -> dict[str, Any]:
    service.store.session_record(sid, owner)
    _tables(service)
    with service.store.connect() as con:
        rows = con.execute(
            "SELECT document,sha256 FROM design_studies WHERE session=? ORDER BY rowid DESC", (sid,)
        ).fetchall()
    return {
        "studies": [
            {
                k: _decode(row)[k]
                for k in ("id", "name", "created_at", "baseline_revision_id", "input_version")
            }
            for row in rows
        ]
    }


def enqueue_study(
    service: WorkspaceService,
    sid: str,
    owner: str,
    study_id: str,
    *,
    retry_failed: bool = False,
    search: SearchDesignStudy | None = None,
) -> dict[str, Any]:
    study = _study(service, sid, owner, study_id)
    _fresh(service, sid, owner, study)
    if search and not study["spec"]["variables"]:
        raise WorkspaceError("Automatic search needs frozen variable bounds")
    run_id = uuid.uuid4().hex
    run = {
        "id": run_id,
        "job_id": None,
        "study_id": study_id,
        "state": "queued",
        "search": search.model_dump() if search else None,
        "rounds_completed": 0,
        "created_at": time.time(),
    }
    with service.store.connect() as con:
        con.execute(
            "INSERT INTO design_study_runs VALUES (?,?,?,?)",
            (run_id, study_id, json.dumps(run), digest(run)),
        )
    try:
        job_id = service.store.enqueue(
            sid,
            owner,
            {
                "engineering_operation": "design_study",
                "study_id": study_id,
                "run_id": run_id,
                "retry_failed": retry_failed,
                "search": search.model_dump() if search else None,
                "expected_input_version": study["input_version"],
            },
        )
    except Exception:
        # Queue rejection is not a new run and must not hide the admitted worker.
        with service.store.connect() as con:
            con.execute("DELETE FROM design_study_runs WHERE id=?", (run_id,))
        raise
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        admitted = _decode(
            con.execute(
                "SELECT document,sha256 FROM design_study_runs WHERE id=?", (run_id,)
            ).fetchone()
        )
        admitted["job_id"] = job_id
        con.execute(
            "UPDATE design_study_runs SET document=?,sha256=? WHERE id=?",
            (json.dumps(admitted), digest(admitted), run_id),
        )
    service.start()
    return {"study_id": study_id, "job_id": job_id, "state": "queued"}


def _save_run(service: WorkspaceService, run: dict[str, Any]) -> None:
    with service.store.connect() as con:
        con.execute(
            "UPDATE design_study_runs SET document=?,sha256=? WHERE id=?",
            (json.dumps(run, allow_nan=False), digest(run), run["id"]),
        )


def _evaluate_candidate(
    service: WorkspaceService,
    sid: str,
    owner: str,
    study: dict[str, Any],
    candidate: dict[str, Any],
    cancelled: Callable[[], bool],
    flow_minimum_level: tuple[int, int] = (0, 0),
) -> None:
    from forge_mind.workspace.document_service import build_document

    attempt = uuid.uuid4().hex
    candidate["attempts"].append(
        {
            "id": attempt,
            "started_at": time.time(),
            "previous_status": candidate["status"],
            "previous_evaluations": candidate.get("evaluations", []),
            "previous_metrics": candidate.get("metrics", {}),
            "previous_revision_id": candidate.get("revision_id"),
        }
    )
    candidate.update(
        status="building",
        eligible=False,
        feasible=None,
        metrics={},
        evaluations=[],
        metric_uncertainty={},
    )
    candidate.pop("error", None)
    _save_candidate(service, candidate)
    try:
        _fresh(service, sid, owner, study)
        if cancelled():
            raise WorkspaceError("Design study cancelled")
        if candidate["revision_id"] is None:
            baseline = service._load_revision(sid, study["baseline_revision_id"])
            document = DesignDocument.model_validate(baseline["document"])
            document.variables = dict(candidate["parameters"])
            document = DesignDocument.model_validate(document.model_dump())
            revision = build_document(
                service,
                sid,
                owner,
                document,
                Contract.model_validate(baseline["contract"]),
                baseline["id"],
                "Design study: " + candidate["label"],
                None,
                cancelled,
            )
            candidate["attempts"][-1]["built_revision_id"] = revision["id"]
            candidate["revision_id"] = (
                revision["id"] if revision["measurement"]["geometry_valid"] else None
            )
            _save_candidate(service, candidate)
        else:
            revision = service._load_revision(sid, candidate["revision_id"])
        if not revision["measurement"]["geometry_valid"]:
            raise WorkspaceError("Candidate geometry did not validate")
        _candidate_engine(study, revision)
        candidate["status"] = "evaluating"
        _save_candidate(service, candidate)
        spec = DesignStudySpec.model_validate(study["spec"])
        evaluators = remap_evaluators(
            DesignDocument.model_validate(revision["document"]),
            service.store.directory(sid, revision["id"]) / "part.step",
            spec,
            study["targets"],
            cancelled,
        )
        all_eligible = True
        for evaluator in evaluators:
            if cancelled():
                raise WorkspaceError("Design study cancelled")
            directory = (
                candidate_directory(service, sid, study["id"], candidate["id"])
                / attempt
                / evaluator.kind
            )
            if evaluator.kind == "flow":
                result = evaluate_design(
                    service,
                    sid,
                    revision,
                    evaluator,
                    directory,
                    cancelled,
                    spec.comparison_tolerance,
                    flow_context={
                        "study_id": study["id"],
                        "spec_hash": study["spec_hash"],
                        "revision_id": revision["id"],
                        "target": study["targets"].get("flow"),
                    },
                    flow_minimum_level=flow_minimum_level,
                )
            else:
                result = evaluate_design(
                    service,
                    sid,
                    revision,
                    evaluator,
                    directory,
                    cancelled,
                    spec.comparison_tolerance,
                )
            if cancelled():
                raise WorkspaceError("Design study cancelled before admitting evaluator evidence")
            _fresh(service, sid, owner, study)
            if (
                evaluator.kind == "flow"
                and result.get("provenance", {}).get("solver_sha256")
                != study["instruments"]["flow"]["solver_sha256"]
            ):
                raise WorkspaceError(
                    "Evaluation used a different flow solver than the frozen protocol"
                )
            service._load_revision(sid, revision["id"])
            hashes = {}
            for path in directory.rglob("*"):
                if path.is_dir():
                    continue
                if not regular_file(path):
                    raise WorkspaceError(
                        "Evaluation produced an invalid or oversized artifact: " + path.name
                    )
                hashes[str(path.relative_to(directory)).replace("\\", "/")] = file_sha256(path)
            receipt = {
                "kind": evaluator.kind,
                "revision_id": revision["id"],
                "attempt": attempt,
                "geometry_hash": revision["measurement"]["artifact_hashes"]["part.step"],
                "spec_hash": study["spec_hash"],
                "settings_hash": digest(evaluator.settings),
                "hashes": hashes,
                "result": result,
            }
            candidate["evaluations"].append(receipt)
            candidate["metrics"].update(result.get("metrics", {}))
            all_eligible = all_eligible and result.get("eligible") is True
            for key, value in result.get("metric_uncertainty", {}).items():
                candidate["metric_uncertainty"][evaluator.kind + "." + key] = value
            _save_candidate(service, candidate)
        required = [o.metric for o in spec.objectives] + [c.metric for c in spec.constraints]
        missing = [
            k
            for k in required
            if (k not in candidate["metrics"] or not math.isfinite(candidate["metrics"][k]))
        ]
        if missing:
            failures = [
                str(error)
                for receipt in candidate["evaluations"]
                for error in receipt.get("result", {}).get("errors", [])
            ]
            raise WorkspaceError(
                "Required objective or constraint measurement is missing: "
                + ", ".join(missing)
                + (". Evaluator reported: " + "; ".join(failures) if failures else "")
            )
        checks = [
            {
                **c.model_dump(),
                "measured": candidate["metrics"][c.metric],
                "passed": candidate["metrics"][c.metric] >= c.value
                if c.operator == "ge"
                else candidate["metrics"][c.metric] <= c.value,
            }
            for c in spec.constraints
        ]
        original_failures = [
            f for f in revision["measurement"]["findings"] if f["status"] == "fail"
        ]
        if cancelled():
            raise WorkspaceError("Design study cancelled before candidate finalization")
        candidate.update(
            status="complete" if all_eligible else "incomplete",
            eligible=all_eligible,
            feasible=all(c["passed"] for c in checks) and not original_failures,
            constraints=checks,
            original_requirement_failures=original_failures,
        )
        if not all_eligible:
            candidate["error"] = (
                "One or more evaluators did not produce admissible evidence; this candidate is excluded from ranking"
            )
    except Exception as exc:
        candidate.update(
            status="cancelled" if cancelled() else "failed", eligible=False, error=str(exc)
        )
    candidate["finished_at"] = time.time()
    _save_candidate(service, candidate)


def _confirm_flow_candidates(
    service: WorkspaceService,
    sid: str,
    owner: str,
    study: dict[str, Any],
    cancelled: Callable[[], bool],
) -> None:
    """Screen freely; confirm baseline and nondominated finalists at a shared cross."""
    from forge_mind.workspace.flow_comparison import common_flow_validation, validation_level

    spec = DesignStudySpec.model_validate(study["spec"])
    if not any(e.kind == "flow" for e in spec.evaluators) or cancelled():
        return

    def score(c: dict[str, Any]) -> list[float]:
        return [
            c["metrics"][o.metric] * (1 if o.direction == "minimize" else -1)
            for o in spec.objectives
        ]

    while not cancelled():
        candidates = _candidates(service, study["id"])
        if not candidates or not candidates[0].get("eligible") or not candidates[0].get("feasible"):
            return
        baseline_id = candidates[0]["id"]
        eligible = [c for c in candidates if c.get("eligible") and c.get("feasible")]
        finalists = [
            c
            for c in eligible
            if c["id"] == baseline_id
            or not any(
                all(a <= b for a, b in zip(score(other), score(c), strict=True))
                and any(a < b for a, b in zip(score(other), score(c), strict=True))
                for other in eligible
            )
        ]
        if len(finalists) <= 1:
            return
        levels = [validation_level(c) for c in finalists]
        if any(level is None for level in levels):
            raise WorkspaceError("Flow finalist evidence lacks a confirmed validation protocol")
        if common_flow_validation(finalists, baseline_id):
            return
        floor = (
            max(level[0] for level in levels if level),
            max(level[1] for level in levels if level),
        )
        changed = False
        for candidate in finalists:
            if cancelled():
                return
            if validation_level(candidate) != floor:
                _evaluate_candidate(service, sid, owner, study, candidate, cancelled, floor)
                changed = True
                if not candidate.get("eligible"):
                    return
        if not changed:
            # Different protocol/reference at equal levels is corruption, not a
            # reason to keep recalculating or accept an unmatched comparison.
            raise WorkspaceError("Flow finalists disagree on their frozen validation protocol")


def run_study_job(
    service: WorkspaceService, job: dict[str, Any], cancelled: Callable[[], bool]
) -> None:
    sid, owner, study_id = job["session"], job["owner"], job["study_id"]
    study = _study(service, sid, owner, study_id)
    with service.store.connect() as con:
        run = _decode(
            con.execute(
                "SELECT document,sha256 FROM design_study_runs WHERE id=?", (job["run_id"],)
            ).fetchone()
        )
    run["state"] = "running"
    run["job_id"] = job["id"]
    _save_run(service, run)
    try:
        _fresh(service, sid, owner, study)
        for candidate in _candidates(service, study_id):
            if candidate["status"] == "complete":
                _verify_evaluations(service, sid, study, candidate)
                continue
            if candidate["status"] in {"failed", "incomplete"} and not job.get("retry_failed"):
                continue
            if cancelled():
                break
            _evaluate_candidate(service, sid, owner, study, candidate, cancelled)
        _confirm_flow_candidates(service, sid, owner, study, cancelled)
        if job.get("search") and not cancelled():
            search = SearchDesignStudy.model_validate(job["search"])
            spec = DesignStudySpec.model_validate(study["spec"])
            for round_index in range(search.rounds):
                if cancelled():
                    break
                moved = False
                anchor_unavailable = False
                for name, (lower, upper) in spec.variables.items():
                    if cancelled():
                        break
                    current = read_study(service, sid, owner, study_id)
                    selection = choose_search_anchor(
                        spec, current["candidates"], current["comparison"]
                    )
                    run["search_anchor"] = selection
                    _save_run(service, run)
                    if selection["candidate_id"] is None:
                        run["stop_reason"] = selection["reason"]
                        anchor_unavailable = True
                        break
                    anchor = next(
                        c for c in current["candidates"] if c["id"] == selection["candidate_id"]
                    )
                    # Scale before subtraction to avoid overflowing finite bounds, and
                    # shift the float exponent instead of constructing an enormous int.
                    step = math.ldexp(
                        upper * search.step_fraction - lower * search.step_fraction,
                        -round_index,
                    )
                    for sign in (-1, 1):
                        if cancelled():
                            break
                        values = {k: anchor["parameters"][k] for k in spec.variables}
                        values[name] = min(upper, max(lower, values[name] + sign * step))
                        if values[name] == anchor["parameters"][name]:
                            continue
                        moved = True
                        candidate = add_candidate(
                            service,
                            sid,
                            owner,
                            study_id,
                            AddStudyCandidate(
                                label=f"Search {round_index + 1}: {name}={values[name]:g}",
                                parameters=values,
                            ),
                        )
                        if candidate["status"] != "complete":
                            _evaluate_candidate(service, sid, owner, study, candidate, cancelled)
                    if cancelled():
                        break
                if anchor_unavailable:
                    break
                run["rounds_completed"] = round_index + 1
                _confirm_flow_candidates(service, sid, owner, study, cancelled)
                if not moved and not cancelled():
                    run["stop_reason"] = "Search steps no longer change representable parameters"
                _save_run(service, run)
                if not moved:
                    break
        run["state"] = "cancelled" if cancelled() else "completed"
    except Exception as exc:
        run.update(state="cancelled" if cancelled() else "interrupted", error=str(exc))
        raise
    finally:
        run["finished_at"] = time.time()
        _save_run(service, run)
        service.store.event(sid, "design_study", "Design study " + run["state"], study_id=study_id)


def propose_candidate(
    service: WorkspaceService, sid: str, owner: str, study_id: str, request: AcceptStudyCandidate
) -> dict[str, Any]:
    study = read_study(service, sid, owner, study_id)
    if not study["current"]:
        raise WorkspaceError("Study is stale")
    candidate = next((c for c in study["candidates"] if c["id"] == request.candidate_id), None)
    if candidate is None or not candidate["eligible"] or candidate["feasible"] is not True:
        raise WorkspaceError("Candidate has no feasible admissible evidence")
    if any(e["kind"] == "flow" for e in study["spec"]["evaluators"]):
        from forge_mind.workspace.flow_comparison import common_flow_validation

        baseline = study["candidates"][0]
        if (
            not baseline.get("eligible")
            or not baseline.get("feasible")
            or not common_flow_validation([baseline, candidate], baseline["id"])
        ):
            raise WorkspaceError(
                "Flow candidate needs shared baseline mesh and domain confirmation before proposal"
            )
    revision = service._load_revision(sid, candidate["revision_id"])
    job_id = service.store.enqueue(
        sid,
        owner,
        {
            "document": revision["document"],
            "contract": revision["contract"],
            "revision_id": revision["id"],
            "expected_current_revision": request.expected_current_revision,
            "expected_input_version": study["input_version"],
            "preview_only": True,
            "explanation": "Propose evaluated design: " + candidate["label"],
        },
    )
    service.start()
    return {
        "job_id": job_id,
        "candidate_id": candidate["id"],
        "state": "queued",
        "scope": "Creates a guarded CAD proposal; does not certify physical performance or overwrite the study baseline.",
    }


def field_path(
    service: WorkspaceService, sid: str, owner: str, study_id: str, candidate_id: str
) -> Path:
    study = read_study(service, sid, owner, study_id)
    candidate = next((c for c in study["candidates"] if c["id"] == candidate_id), None)
    if candidate is None:
        raise WorkspaceError("Study candidate not found")
    for receipt in candidate["evaluations"]:
        if receipt["kind"] == "flow" and "surface_field.json" in receipt["hashes"]:
            return (
                candidate_directory(service, sid, study_id, candidate_id)
                / str(receipt["attempt"])
                / "flow"
                / "surface_field.json"
            )
    raise WorkspaceError("This candidate has no retained CFD field")
