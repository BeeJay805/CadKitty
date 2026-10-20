"""Exact-identity CFD stage reuse with immutable receipts and cumulative work accounting."""

from __future__ import annotations

import contextlib
import json
import math
import os
import stat
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.document import digest
from forge_mind.workspace.flow_models import FlowSettings
from forge_mind.workspace.flow_solver import solver_executable
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.processing import process_revision, regular_file

_CRASH_RESERVE_SECONDS = 2.0


def _verify_solver_config(directory: Path, settings: FlowSettings, result: dict[str, Any]) -> None:
    """Reconstruct the sole permitted native replay change from retained history."""
    from forge_mind.workspace.flow_solver import convergence_evidence, read_history, solver_config

    expected = solver_config(settings)
    minimum = settings.numerics.force_window
    previous = -1
    replays = result.get("solver_replays", [])
    if not isinstance(replays, list) or (
        replays
        and result.get("replay_policy")
        != "cold replay; delay native checks by one force window; fixed iteration ceiling v1"
    ):
        raise ValueError("Flow solver replay policy is invalid")
    initial = directory / "run.cfg"
    for ordinal, replay in enumerate(replays, 1):
        prefix = f"solver-replay-{ordinal:03d}-"
        archived = replay.get("artifacts", {})
        if not {prefix + "run.cfg", prefix + "history.csv"} <= archived.keys():
            raise ValueError("Flow solver replay evidence is incomplete")
        for name, expected_hash in archived.items():
            if (
                not name.startswith(prefix)
                or result["artifacts"].get(name, {}).get("sha256") != expected_hash
                or file_sha256(_artifact(directory, name)) != expected_hash
            ):
                raise ValueError("Flow solver replay artifact disagrees with manifest")
        config = _artifact(directory, prefix + "run.cfg")
        if ordinal == 1:
            initial = config
        if config.read_text(encoding="utf-8") != expected:
            raise ValueError("Flow solver replay changed the frozen configuration")
        rows = read_history(_artifact(directory, prefix + "history.csv"))
        measured = convergence_evidence(rows, settings)
        last = int(measured["last_iteration"])
        next_minimum = min(settings.numerics.iterations - 1, last + settings.numerics.force_window)
        if (
            digest(measured) != digest(replay.get("convergence", {}))
            or measured["residual_gate"]
            or not measured["force_gate"]
            or not previous < last < settings.numerics.iterations - 1
            or any(
                rows[-1][key] > settings.numerics.residual_log10_max
                for key in measured["worst_window_residuals_log10"]
            )
            or replay.get("next_minimum_iteration") != next_minimum
            or next_minimum <= minimum
        ):
            raise ValueError("Flow solver replay is not justified by retained history")
        expected = expected.replace(
            f"CONV_STARTITER= {minimum}\n", f"CONV_STARTITER= {next_minimum}\n"
        )
        minimum, previous = next_minimum, last
    if (
        (directory / "run.cfg").read_text(encoding="utf-8") != expected
        or result["provenance"].get("initial_config_sha256") != file_sha256(initial)
        or result["convergence"]["last_iteration"] <= previous
    ):
        raise ValueError("Flow solver configuration differs from the frozen stage or replay proof")


def _retry_sharing(operation: Callable[[], Any]) -> None:
    """Windows readers can briefly deny rename/delete; never wait indefinitely."""
    for attempt in range(4):
        try:
            operation()
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.01 * 2**attempt)


def _atomic_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    """Mutable diagnostics replace atomically; evidence paths are created only once."""
    if immutable and path.exists():
        raise ValueError("Flow stage evidence already exists: " + path.name)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            # Atomic exclusive creation on both Windows and POSIX. Remove the
            # temporary hardlink immediately; evidence remains a single-link file.
            _retry_sharing(lambda: os.link(temporary, path))
        else:
            _retry_sharing(lambda: os.replace(temporary, path))
    finally:
        _retry_sharing(lambda: temporary.unlink(missing_ok=True))


def _write_diagnostic(path: Path, value: dict[str, Any]) -> None:
    """Unavailable display telemetry must never interrupt owned numerical work."""
    # Keep the previous atomic snapshot; evidence writers remain fail closed.
    with contextlib.suppress(OSError):
        _atomic_json(path, value)


def _read_diagnostic(path: Path) -> dict[str, Any]:
    try:
        return _json(path)
    except (OSError, ValueError):
        return {}


def _seal(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    _atomic_json(path, {"payload": value, "sha256": digest(value)}, immutable=immutable)


def _json(path: Path) -> dict[str, Any]:
    if not regular_file(path):
        raise ValueError("Flow cache record is missing or not regular: " + path.name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid flow cache record: " + path.name)
    return value


def _unseal(path: Path) -> dict[str, Any]:
    value = _json(path)
    payload = value.get("payload")
    if not isinstance(payload, dict) or digest(payload) != value.get("sha256"):
        raise ValueError("Flow cache record integrity check failed: " + path.name)
    return payload


def _artifact(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()) or not regular_file(path):
        raise ValueError("Invalid flow stage artifact: " + name)
    return path


def _alive(runtime: dict[str, Any]) -> bool:
    if runtime.get("finished_at") is not None:
        return False
    try:
        process = psutil.Process(runtime["pid"])
        return bool(process.create_time() == runtime["process_created_at"] and process.is_running())
    except psutil.AccessDenied:
        return True
    except (KeyError, psutil.NoSuchProcess):
        return False


def _worker_alive(directory: Path) -> bool:
    runtime_path = directory / "runtime.json"
    if _alive(_read_diagnostic(runtime_path)):
        return True
    # A parent can die while Python is importing, before the child writes its
    # first runtime snapshot. The exact owned request path identifies that gap.
    request = (directory / "processing.json").resolve()
    for process in psutil.process_iter(["cmdline"]):
        try:
            command = process.info.get("cmdline") or []
            if (
                "forge_mind.workspace.processing" in command
                and Path(command[-1]).resolve() == request
            ):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return False


def run_flow_stage(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Owned child entrypoint; a complete seal allows safe parent-crash adoption."""
    from forge_mind.workspace.flow_solver import evaluate_flow

    directory = directory.resolve()
    plan = _unseal(directory / "planned.json")
    inputs = plan["inputs"]
    source = Path(payload["source_step"])
    settings = FlowSettings.model_validate(payload["settings"])
    if (
        payload["stage_key"] != plan["key"]
        or digest(inputs) != plan["key"]
        or settings.model_dump(mode="json") != inputs["settings"]
        or file_sha256(source) != inputs["source_sha256"]
        or file_sha256(solver_executable()) != inputs["solver_sha256"]
        or engine_fingerprint() != inputs["engine_fingerprint"]
    ):
        raise ValueError("Flow stage inputs changed before dispatch")
    started = time.monotonic()
    runtime = {
        "pid": os.getpid(),
        "process_created_at": psutil.Process().create_time(),
        "started_at": time.time(),
        "phase": "meshing",
        "elapsed_seconds": 0.0,
    }

    def progress(update: dict[str, Any]) -> None:
        runtime.update(update)
        runtime["phase"] = update.get("stage", runtime["phase"])
        runtime["elapsed_seconds"] = time.monotonic() - started
        runtime["observed_at"] = time.time()
        _write_diagnostic(directory / "runtime.json", runtime)

    progress({})
    try:
        result = evaluate_flow(source, directory / "solve", settings, lambda: False, progress)
        progress({"stage": "postprocessing"})
        if (
            file_sha256(source) != inputs["source_sha256"]
            or file_sha256(solver_executable()) != inputs["solver_sha256"]
            or engine_fingerprint() != inputs["engine_fingerprint"]
        ):
            raise ValueError("Flow stage instruments changed during execution")
        native = _artifact(directory / "solve", "flow_result.json")
        if digest(_json(native)) != digest(result):
            raise ValueError("Native flow result differs from returned stage result")
        runtime["finished_at"] = time.time()
        runtime["elapsed_seconds"] = time.monotonic() - started
        _write_diagnostic(directory / "runtime.json", runtime)
        _seal(
            directory / "child_complete.json",
            {
                "key": plan["key"],
                "inputs_hash": digest(inputs),
                "result_sha256": file_sha256(native),
                "runtime": runtime,
            },
            immutable=True,
        )
        return result
    finally:
        if runtime.get("finished_at") is None:
            runtime.update(
                finished_at=time.time(),
                elapsed_seconds=time.monotonic() - started,
                phase="interrupted",
            )
            _write_diagnostic(directory / "runtime.json", runtime)


class FlowStageCache:
    """Single owned planner per candidate; cache hits verify every retained evidence byte."""

    def __init__(
        self,
        source: Path,
        root: Path,
        identity: dict[str, Any],
        cancelled: Callable[[], bool],
        max_total_seconds: float | None = None,
    ):
        if max_total_seconds is not None and (
            not math.isfinite(max_total_seconds) or max_total_seconds <= 0
        ):
            raise ValueError("Total flow wall-time limit must be finite and positive")
        self.source, self.root = source.resolve(), root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cancelled, self.max_total_seconds = cancelled, max_total_seconds
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.source_hash = file_sha256(self.source)
        self.solver_hash = file_sha256(solver_executable())
        self.engine = engine_fingerprint()
        self.progress_path = self.root / "progress.json"
        self.usage_path = self.root / "usage.json"
        if not self.usage_path.exists() and any(self.root.glob("*/planned.json")):
            raise ValueError("Flow cache work accounting is missing; cannot reset its budget")
        self.usage = _unseal(self.usage_path) if self.usage_path.exists() else {"attempts": {}}
        self.progress = _read_diagnostic(self.progress_path)
        self._active: tuple[str, float, float] | None = None
        self._last_checkpoint = 0.0
        self._reconcile_usage()

    @property
    def elapsed_seconds(self) -> float:
        elapsed = sum(v["seconds"] for v in self.usage["attempts"].values())
        if self._active:
            attempt_id, started, previous = self._active
            elapsed -= self.usage["attempts"][attempt_id]["seconds"]
            elapsed += previous + time.monotonic() - started
        return float(elapsed)

    def _reconcile_usage(self) -> None:
        for directory in self._directories():
            planned = directory / "planned.json"
            if not planned.exists():
                continue
            plan = _unseal(planned)
            record = self.usage["attempts"].setdefault(
                directory.name,
                {
                    "seconds": _CRASH_RESERVE_SECONDS,
                    "state": "interrupted",
                    "key": plan["key"],
                },
            )
            if (
                record["state"] == "running"
                and not (directory / "child_complete.json").exists()
                and _worker_alive(directory)
            ):
                raise ValueError("A prior flow stage is still running; await its completion")
            runtime_path = directory / "runtime.json"
            if record["state"] == "running":
                child_path = directory / "child_complete.json"
                # Completion timing is durable evidence even when readers kept
                # the diagnostic runtime snapshot locked throughout execution.
                runtime = (
                    _unseal(child_path)["runtime"]
                    if child_path.exists()
                    else _read_diagnostic(runtime_path)
                )
                if _alive(runtime):
                    raise ValueError("A prior flow stage is still running; await its completion")
                end = runtime.get("finished_at", runtime.get("observed_at", plan["started_at"]))
                charge = max(0.0, end - plan["started_at"], runtime.get("elapsed_seconds", 0.0))
                if runtime.get("finished_at") is None:
                    charge += _CRASH_RESERVE_SECONDS
                record["seconds"] = max(record["seconds"], charge)
                record["state"] = "interrupted"
        self._save_usage()

    def _directories(self) -> list[Path]:
        directories = []
        for directory in self.root.iterdir():
            if directory.is_dir():
                if (
                    directory.is_symlink()
                    or getattr(directory.lstat(), "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                    or not directory.resolve().is_relative_to(self.root)
                ):
                    raise ValueError("Linked flow cache stage directory is not allowed")
                directories.append(directory)
        return sorted(directories)

    def _save_usage(self) -> None:
        _seal(self.usage_path, self.usage)

    def update_progress(self, update: dict[str, Any] | None = None, **updates: Any) -> None:
        self.progress.update(update or {})
        self.progress.update(updates)
        self.progress["elapsed_seconds"] = self.elapsed_seconds
        self.progress["completed_stages"] = len(
            {r["key"] for r in self.usage["attempts"].values() if r["state"] == "complete"}
        )
        self.progress.setdefault("reused_stages", 0)
        self.progress["scope"] = (
            "Diagnostic execution progress, not numerical eligibility or physical validation"
        )
        self.progress["time_accounting"] = (
            "Cumulative active stage wall time; interrupted recovery retains up to two seconds of conservative heartbeat reserve, excluding downtime"
        )
        _write_diagnostic(self.progress_path, self.progress)

    def _unchanged(self) -> None:
        if (
            file_sha256(self.source) != self.source_hash
            or file_sha256(solver_executable()) != self.solver_hash
            or engine_fingerprint() != self.engine
        ):
            raise ValueError(
                "Flow stage source or instruments changed; start a new frozen protocol"
            )

    def _verify_result(self, directory: Path, key: str, inputs: dict[str, Any]) -> dict[str, Any]:
        child = _unseal(directory / "child_complete.json")
        if child["key"] != key or child["inputs_hash"] != digest(inputs):
            raise ValueError("Flow child receipt belongs to another stage")
        native = _artifact(directory / "solve", "flow_result.json")
        if file_sha256(native) != child["result_sha256"]:
            raise ValueError("Native flow result integrity check failed")
        result = _json(native)
        if result.get("settings") != inputs["settings"]:
            raise ValueError("Flow result settings differ from the frozen stage")
        provenance = result.get("provenance", {})
        if any(
            name in provenance and provenance[name] != inputs[expected]
            for name, expected in (
                ("source_step_sha256", "source_sha256"),
                ("solver_sha256", "solver_sha256"),
            )
        ):
            raise ValueError("Flow result source or solver identity changed")
        for name, expected in result.get("artifacts", {}).items():
            path = _artifact(directory / "solve", name)
            if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
                raise ValueError("Flow stage artifact changed: " + name)
        if result.get("eligible"):
            from forge_mind.workspace.flow_solver import convergence_evidence, read_history

            if provenance.get("solver_sha256") != inputs["solver_sha256"]:
                raise ValueError("Eligible flow stage is missing frozen solver provenance")
            required = {
                "source.step",
                "settings.json",
                "run.cfg",
                "mesh.su2",
                "history.csv",
                "surface_field.json",
            }
            if not required <= result.get("artifacts", {}).keys():
                raise ValueError("Eligible flow stage is missing required artifacts")
            if _json(directory / "solve" / "settings.json") != inputs["settings"]:
                raise ValueError("Native settings file differs from the frozen stage")
            if result.get("status") != "complete" or any(
                type(result.get("metrics", {}).get(k)) not in (int, float)
                or not math.isfinite(result["metrics"][k])
                for k in ("drag_n", "drag_coefficient")
            ):
                raise ValueError("Eligible flow stage has invalid metrics or status")
            for pkey, name in (
                ("source_step_sha256", "source.step"),
                ("settings_sha256", "settings.json"),
                ("config_sha256", "run.cfg"),
                ("mesh_sha256", "mesh.su2"),
            ):
                if provenance.get(pkey) != result["artifacts"][name]["sha256"]:
                    raise ValueError("Flow stage provenance disagrees with artifacts")
            measured = convergence_evidence(
                read_history(directory / "solve" / "history.csv"),
                FlowSettings.model_validate(inputs["settings"]),
            )
            if (
                measured.get("eligible") is not True
                or digest(measured) != digest(result.get("convergence", {}))
                or any(
                    result["metrics"][key] != measured[key]
                    for key in ("drag_n", "drag_coefficient")
                )
                or result.get("wall", {}).get("eligible") is not True
            ):
                raise ValueError(
                    "Eligible flow result disagrees with retained convergence or wall evidence"
                )
            _verify_solver_config(
                directory / "solve", FlowSettings.model_validate(inputs["settings"]), result
            )
        return result

    def _receipt(self, directory: Path, plan: dict[str, Any], result: dict[str, Any]) -> None:
        _seal(
            directory / "receipt.json",
            {
                "key": plan["key"],
                "plan_hash": digest(plan),
                "child_sha256": file_sha256(directory / "child_complete.json"),
                "result_hash": digest(result),
            },
            immutable=True,
        )

    def _checkpoint(self) -> bool:
        if self._active:
            attempt_id, started, previous = self._active
            elapsed = previous + time.monotonic() - started
            if time.monotonic() - self._last_checkpoint >= 1:
                self.usage["attempts"][attempt_id]["seconds"] = elapsed + _CRASH_RESERVE_SECONDS
                self._save_usage()
                self._last_checkpoint = time.monotonic()
                runtime_path = self.root / attempt_id / "runtime.json"
                runtime = _read_diagnostic(runtime_path)
                if isinstance(runtime.get("phase"), str):
                    self.progress["phase"] = runtime["phase"]
                self.update_progress()
        return bool(
            self.cancelled()
            or (
                self.max_total_seconds is not None
                and self.elapsed_seconds >= self.max_total_seconds
            )
        )

    def run(self, settings: FlowSettings, label: str) -> tuple[dict[str, Any], Path, bool]:
        self._unchanged()
        inputs = {
            "identity": self.identity,
            "source_sha256": self.source_hash,
            "solver_sha256": self.solver_hash,
            "engine_fingerprint": self.engine,
            "settings": settings.model_dump(mode="json"),
        }
        key = digest(inputs)
        if self.cancelled():
            raise ValueError("Flow stage cancelled")
        for directory in self._directories():
            if not (directory / "planned.json").exists():
                continue
            plan = _unseal(directory / "planned.json")
            if plan.get("key") != key:
                continue
            if plan.get("inputs") != inputs or digest(plan["inputs"]) != key:
                raise ValueError("Flow stage plan integrity check failed")
            receipt_path = directory / "receipt.json"
            child_path = directory / "child_complete.json"
            if not child_path.exists():
                if _worker_alive(directory):
                    raise ValueError("A prior flow stage is still running; await its completion")
                continue
            result = self._verify_result(directory, key, inputs)
            if receipt_path.exists():
                receipt = _unseal(receipt_path)
                if (
                    receipt.get("key") != key
                    or receipt.get("plan_hash") != digest(plan)
                    or receipt.get("child_sha256") != file_sha256(child_path)
                    or receipt.get("result_hash") != digest(result)
                ):
                    raise ValueError("Flow stage receipt integrity check failed")
            else:
                self._receipt(directory, plan, result)
            if result.get("eligible") is not True or result.get("status") != "complete":
                continue
            self.usage["attempts"][directory.name]["state"] = "complete"
            self._save_usage()
            self.progress["reused_stages"] = self.progress.get("reused_stages", 0) + 1
            self.update_progress(
                stage=label,
                phase="reused",
                stage_directory=str((directory / "solve").relative_to(self.root.parent)).replace(
                    "\\", "/"
                ),
            )
            return result, directory / "solve", True
        if self.max_total_seconds is not None and self.elapsed_seconds >= self.max_total_seconds:
            self.update_progress(phase="budget_exhausted")
            raise ValueError("Flow adaptation exhausted its cumulative wall-time budget")
        attempt_id = uuid.uuid4().hex
        directory = self.root / attempt_id
        directory.mkdir()
        plan = {"key": key, "inputs": inputs, "label": label, "started_at": time.time()}
        _seal(directory / "planned.json", plan, immutable=True)
        self.usage["attempts"][attempt_id] = {
            "seconds": _CRASH_RESERVE_SECONDS,
            "state": "running",
            "key": key,
        }
        self._save_usage()
        self._active = (attempt_id, time.monotonic(), 0.0)
        self.update_progress(
            stage=label,
            phase="meshing",
            stage_directory=str((directory / "solve").relative_to(self.root.parent)).replace(
                "\\", "/"
            ),
        )
        state = "interrupted"
        try:
            remaining = (
                None
                if self.max_total_seconds is None
                else self.max_total_seconds - self.elapsed_seconds
            )
            returned = process_revision(
                directory,
                {
                    "operation": "study_flow_stage",
                    "source_step": str(self.source),
                    "settings": inputs["settings"],
                    "stage_key": key,
                },
                remaining,
                self._checkpoint,
            )
            self._unchanged()
            result = self._verify_result(directory, key, inputs)
            if digest(returned) != digest(result):
                raise ValueError("Returned flow result differs from sealed native result")
            self._receipt(directory, plan, result)
            state = (
                "complete"
                if result.get("eligible") is True and result.get("status") == "complete"
                else "incomplete"
            )
            return result, directory / "solve", False
        except Exception as exc:
            if self.cancelled():
                state = "cancelled"
                raise ValueError("Flow stage cancelled") from exc
            if (
                self.max_total_seconds is not None
                and self.elapsed_seconds >= self.max_total_seconds
            ):
                state = "budget_exhausted"
                raise ValueError(
                    "Flow adaptation exhausted its cumulative wall-time budget"
                ) from exc
            raise
        finally:
            assert self._active is not None
            self.usage["attempts"][attempt_id].update(
                seconds=time.monotonic() - self._active[1], state=state
            )
            self._active = None
            self._save_usage()
            self.update_progress(phase=state)
