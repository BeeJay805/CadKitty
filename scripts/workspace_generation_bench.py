"""Retained actual-model generation trials, with first-build and final evidence.

Use an explicit PYTHONPATH to select a frozen engine. Results are development
evidence; numeric checks alone do not establish full task or physical success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--seconds", type=int, default=480)
    parser.add_argument("--actions", type=int, default=10)
    parser.add_argument("--edit-instruction", default="")
    parser.add_argument("--edit-seconds", type=int, default=180)
    parser.add_argument(
        "--representation",
        choices=("hybrid", "native_operations", "python_features"),
        default="hybrid",
    )
    args = parser.parse_args(argv)
    if args.seconds < 1 or args.actions < 1 or args.edit_seconds < 1:
        parser.error("seconds and actions must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty; retained trials cannot be overwritten")
    return args


def validate_python_control(result: Any) -> None:
    """Refuse cross-representation construction before executing a control response."""
    from pydantic import ValidationError

    from forge_mind.workspace.authoring import AssemblyEdit
    from forge_mind.workspace.document import DesignDocument, PartDefinition
    from forge_mind.workspace.models import Action

    if not isinstance(result, Action):
        return
    if result.kind == "build":
        raise ValueError(
            "Python control requires named document Features, not a whole-program build"
        )
    features = []
    try:
        if result.kind == "document":
            control = DesignDocument.model_validate_json(result.document_json)
            features = (
                [f for p in control.assembly.parts for f in p.features]
                if control.assembly
                else control.features
            )
        elif result.kind == "assembly_edit":
            edit = AssemblyEdit.model_validate_json(result.document_json)
            features = [f for p in edit.parts for f in p.features]
        elif result.kind == "edit_part":
            features = PartDefinition.model_validate_json(result.document_json).features
    except ValidationError:
        # Shared author dispatch refuses this input before CAD and returns repair
        # feedback. The control must not turn that refusal into a model-call error.
        return
    if any(f.kind != "python" for f in features):
        raise ValueError("Python control emitted a native geometry Feature")


def score(
    revision: dict[str, Any] | None,
    requirement_ids: set[str],
    *,
    directory: Path | None = None,
    contract_hash: str | None = None,
    engine_hash: str | None = None,
) -> dict[str, Any]:
    measurement = (revision or {}).get("measurement", {})
    findings = measurement.get("findings", [])
    counts = {key: sum(f.get("requirement_id") == key for f in findings) for key in requirement_ids}
    passed = {
        f["requirement_id"]
        for f in findings
        if f.get("requirement_id") in requirement_ids
        and counts[f["requirement_id"]] == 1
        and f.get("status") == "pass"
    }
    evidence_verified: bool | None = None
    evidence_error = None
    if directory is not None or contract_hash is not None or engine_hash is not None:
        from forge_mind.workspace.measure import verify_artifacts
        from forge_mind.workspace.models import Measurement

        evidence_verified = False
        try:
            measured = Measurement.model_validate(measurement)
            if directory is None or contract_hash is None or engine_hash is None:
                raise ValueError("Artifact, contract and engine bindings are all required")
            if measured.revision_id != (revision or {}).get("id"):
                raise ValueError("Measurement belongs to another revision")
            if (
                measured.contract_hash != contract_hash
                or measured.engine_fingerprint != engine_hash
            ):
                raise ValueError("Measurement contract or engine binding changed")
            if not verify_artifacts(directory, measured):
                raise ValueError("Retained artifact hashes are missing or changed")
            evidence_verified = True
        except (ValueError, OSError) as exc:
            evidence_error = str(exc)
    return {
        "revision_id": (revision or {}).get("id"),
        "geometry_valid": measurement.get("geometry_valid", False),
        "numeric_passed": len(passed),
        "numeric_total": len(requirement_ids),
        "numeric_complete": bool(requirement_ids)
        and measurement.get("geometry_valid", False)
        and passed == requirement_ids
        and evidence_verified is not False,
        "evidence_verified": evidence_verified,
        "evidence_error": evidence_error,
        "score_scope": "Unique numeric findings only; live trials also verify exact artifact/contract/engine binding. Not complete request coverage.",
        "full_task_success": None,
        "physical_readiness": "unverified",
        "findings": findings,
        "errors": measurement.get("errors", []),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ["FORGE_CODEX_MODEL"] = args.model
    os.environ["FORGE_FORBID_API_SPEND"] = "1"
    from scripts.workspace_bench import contract

    from forge_mind.llm.codex_client import CodexClient
    from forge_mind.workspace import agent
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.measure import engine_fingerprint
    from forge_mind.workspace.store import Store

    corpus_bytes = args.corpus.read_bytes()
    cases = json.loads(corpus_bytes)["cases"]
    requested = set(filter(None, args.cases.split(",")))
    known = {case["id"] for case in cases}
    if requested - known:
        raise ValueError(f"Unknown cases: {sorted(requested - known)}")
    cases = [case for case in cases if not requested or case["id"] in requested]
    if not cases:
        raise ValueError("At least one case is required")
    root = Path(agent.__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name: str, value: Any) -> None:
        destination = args.output / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)

    fingerprint = engine_fingerprint()
    (args.output / "corpus.json").write_bytes(corpus_bytes)
    save(
        "provenance.json",
        {
            "configured_model": args.model,
            "model_identity_scope": "CLI request and provider-reported usage; no independent backend attestation",
            "provider": "codex subscription or account credits",
            "mode": "trusted_local",
            "engine_root": str(root),
            "agent_file": agent.__file__,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "contract_helper": {
                "path": str(Path(contract.__code__.co_filename).resolve()),
                "sha256": hashlib.sha256(
                    Path(contract.__code__.co_filename).read_bytes()
                ).hexdigest(),
            },
            "python": {"executable": sys.executable, "version": sys.version},
            "platform": platform.platform(),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ),
            "engine_fingerprint": fingerprint,
            "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "cases": [c["id"] for c in cases],
            "max_actions": args.actions,
            "max_seconds": args.seconds,
            "representation": args.representation,
            "edit_instruction": args.edit_instruction,
            "edit_seconds": args.edit_seconds if args.edit_instruction else None,
            "token_limit_scope": "Codex CLI max_tokens is advisory; wall time and action limits are enforced. Actual usage is retained.",
            "started": time.time(),
        },
    )

    class RecordingClient(CodexClient):
        def __init__(self, calls: list[dict[str, Any]], case_id: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.calls = calls
            self.case_id = case_id

        def _generate(self, *values: Any, **kwargs: Any) -> Any:
            if args.representation == "python_features" and values:
                values = (
                    values[0]
                    + "\nRepresentation control: author all new part geometry using Feature kind=python inside the same named document and variables. Native geometric Feature kinds and whole-program build are forbidden in this control. Assembly placement remains available. Use exact python Feature schema. No hidden native fallback.\n",
                    *values[1:],
                )
            started = time.monotonic()
            record: dict[str, Any] = {"index": len(self.calls) + 1, "configured_model": self.model}
            for key, value in zip(
                ("system", "user", "output_model", "max_tokens"), values, strict=False
            ):
                if key in {"system", "user"}:
                    record[key] = value
            record.update({k: v for k, v in kwargs.items() if k in {"system", "user"}})
            try:
                result, usage = super()._generate(*values, **kwargs)
                record.update(response=result.model_dump(), usage=vars(usage))
                if args.representation == "python_features":
                    validate_python_control(result)
                return result, usage
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                cause = exc.__cause__
                causes = []
                while cause is not None and len(causes) < 4:
                    detail = {"type": type(cause).__name__, "message": str(cause)[:8000]}
                    if callable(getattr(cause, "errors", None)):
                        detail["validation_errors"] = cause.errors(
                            include_input=False, include_context=False, include_url=False
                        )
                    causes.append(detail)
                    cause = cause.__cause__
                record["error_causes"] = causes
                raise
            finally:
                record["input_images"] = list(self.last_input_images)
                record["call_diagnostics"] = getattr(self, "last_call_diagnostics", None)
                record["image_scope"] = (
                    "Images launched by the CLI; no independent model receipt attestation"
                )
                try:
                    record["prepared_image_observation"] = json.loads(record.get("user", "{}")).get(
                        "image_observation"
                    )
                except (TypeError, ValueError):
                    record["prepared_image_observation"] = None
                record["seconds"] = time.monotonic() - started
                self.calls.append(record)
                save(f"{self.case_id}-calls.json", self.calls)

    results = []
    for case in cases:
        calls: list[dict[str, Any]] = []
        started = time.monotonic()
        row: dict[str, Any] = {
            "case": case["id"],
            "state": "harness_error",
            "harness_error": None,
            "trial_kind": "controlled_repair" if "repair_seed_code" in case else "generation",
        }
        store, session, frozen = None, None, None
        supplied_seed = None
        failure: BaseException | None = None
        try:
            if engine_fingerprint() != fingerprint:
                raise RuntimeError(
                    "Engine changed during the trial; freeze source before continuing"
                )
            store = Store(args.output / case["id"])
            frozen = contract(case)
            neutral_plan = case.get("design_plan")
            prompt = case["prompt"] + (
                "\nFrozen detailed design intent: " + json.dumps(neutral_plan, sort_keys=True)
                if neutral_plan
                else ""
            )
            session = store.create(
                "benchmark-owner",
                prompt,
                frozen.model_dump(),
                authoring_mode="native_operations"
                if args.representation == "native_operations"
                else None,
            )
            service = agent.WorkspaceService(
                store, Executor("trusted_local"), partial(RecordingClient, calls, case["id"])
            )
            if "repair_seed_code" in case:
                seed_code = case["repair_seed_code"]
                if not isinstance(seed_code, str) or not seed_code.strip():
                    raise ValueError("repair_seed_code must contain supplied CAD source")
                seed_started = time.monotonic()
                seed = service.build(
                    session["id"],
                    seed_code,
                    frozen,
                    None,
                    "Supplied failing seed for controlled actual-model repair trial",
                    min(args.seconds, 120),
                    lambda: False,
                )
                supplied_seed = {
                    "revision_id": seed["id"],
                    "origin": "Supplied controlled failure; not model-authored or a first-shot generation",
                    "source_sha256": hashlib.sha256(seed_code.encode()).hexdigest(),
                    "seconds": time.monotonic() - seed_started,
                    "geometry_valid": seed["measurement"]["geometry_valid"],
                    "errors": seed["measurement"].get("errors", []),
                    "artifact_hashes": seed["measurement"].get("artifact_hashes", {}),
                }
                save(f"{case['id']}-supplied-seed.json", supplied_seed)
                if supplied_seed["geometry_valid"]:
                    raise ValueError(
                        "Controlled repair seed must fail actual geometry construction"
                    )
            store.enqueue(
                session["id"],
                "benchmark-owner",
                {"max_actions": args.actions, "max_seconds": args.seconds},
            )
            job = store.claim()
            if job is None:
                raise RuntimeError("Benchmark job could not be claimed")
            print(f"START {case['id']} {session['id']}", flush=True)
            service.run_job(job)
            construction = store.get(session["id"], "benchmark-owner")
            save(f"{case['id']}-construction-session.json", construction)
            row["construction_seconds"] = time.monotonic() - started
            row["construction_model_calls"] = len(calls)
            row["construction_usage"] = construction.get("usage", [])
            row["construction_selected_revision"] = construction.get("selected_revision")
            if args.edit_instruction and construction.get("selected_revision"):
                edit_started = time.monotonic()
                calls_before = len(calls)
                store.enqueue(
                    session["id"],
                    "benchmark-owner",
                    {
                        "instruction": args.edit_instruction,
                        "contract": frozen.model_dump(),
                        "revision_id": construction["selected_revision"],
                        "max_seconds": args.edit_seconds,
                        "max_actions": 12,
                    },
                )
                edit_job = store.claim()
                if edit_job is None:
                    raise RuntimeError("Edit phase could not be claimed")
                service.run_job(edit_job)
                edit_session = store.get(session["id"], "benchmark-owner")
                save(f"{case['id']}-edit-session.json", edit_session)
                row["edit"] = {
                    "seconds": time.monotonic() - edit_started,
                    "model_calls": len(calls) - calls_before,
                    "usage": edit_session.get("usage", []),
                    "state": edit_session["state"],
                    "selected_revision": edit_session.get("selected_revision"),
                }
        except BaseException as exc:
            failure = exc
            row["harness_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            final = None
            if store is not None and session is not None:
                try:
                    final = store.get(session["id"], "benchmark-owner")
                    save(f"{case['id']}-session.json", final)
                except Exception as exc:
                    row["snapshot_error"] = f"{type(exc).__name__}: {exc}"
                    if failure is None:
                        failure = exc
            try:
                if (
                    final is not None
                    and frozen is not None
                    and store is not None
                    and session is not None
                ):
                    revisions = final["revisions"]
                    model_revisions = [
                        revision
                        for revision in revisions
                        if supplied_seed is None or revision["id"] != supplied_seed["revision_id"]
                    ]
                    selected = next(
                        (r for r in revisions if r["id"] == final.get("selected_revision")), None
                    )
                    numeric = {r.id for r in frozen.requirements if r.metric != "unmeasured"}

                    scored = {
                        label: score(
                            revision,
                            numeric,
                            directory=store.directory(session["id"], revision["id"])
                            if revision
                            else args.output / "absent-revision",
                            contract_hash=frozen.fingerprint(),
                            engine_hash=fingerprint,
                        )
                        for label, revision in (
                            ("first", model_revisions[0] if model_revisions else None),
                            ("selected", selected),
                        )
                    }
                    row.update(
                        state="harness_error" if failure is not None else final["state"],
                        builds=len(model_revisions),
                        total_revisions=len(revisions),
                        supplied_seed=supplied_seed,
                        **scored,
                        usage=final.get("usage", []),
                        accepted_revision=final.get("accepted_revision"),
                    )
            except Exception as exc:
                row.update(state="harness_error", scoring_error=f"{type(exc).__name__}: {exc}")
                if failure is None:
                    failure = exc
            row.update(
                seconds=time.monotonic() - started,
                model_calls=len(calls),
                source_unchanged=engine_fingerprint() == fingerprint,
            )
            results.append(row)
            save("results.json", results)
            print(json.dumps(row), flush=True)
        if failure is not None:
            raise failure
        if not row["source_unchanged"]:
            raise RuntimeError(
                "Engine changed during the trial; retain but do not compare this run"
            )


if __name__ == "__main__":
    main()
