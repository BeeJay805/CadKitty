"""Container boundary for public CAD execution; private development is explicit."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from forge_mind.agentic.sandbox import SandboxResult
from forge_mind.workspace.budget import end_time, expired
from forge_mind.workspace.capacity import optional_limit
from forge_mind.workspace.processing import (
    HOSTED_LIMITS,
    MAX_JOB_BYTES,
    bounded_process,
    copy_artifact,
    directory_size,
    regular_file,
)
from forge_mind.workspace.typed_cache import TypedCacheInput, execution_origin, stage_cache
from forge_mind.workspace.worker_threads import docker_thread_arguments

TYPED_ARTIFACTS = {
    "program.py",
    "document.json",
    "part.step",
    "part.stl",
    "part.glb",
    "render.png",
    "tessellation.json",
    "topology.json",
    "assembly.json",
    "parts.json",
    "motion.json",
    "motion-rig.json",
    "render-inspection.png",
    "render-inspection.json",
    "definition-meshes.json",
}


def typed_artifact_name(name: str) -> bool:
    return name in TYPED_ARTIFACTS or bool(
        re.fullmatch(
            r"(?:definition_[A-Za-z][A-Za-z0-9_]{0,63}\.(?:step|stl)|part_[a-f0-9]{32}\.brep)",
            name,
        )
    )


def _typed_worker_log_tail(log: BinaryIO, path: Path) -> str:
    """Read bounded untrusted diagnostics from the original, still-owned log handle."""
    log.flush()
    before = os.fstat(log.fileno())
    if not regular_file(path, HOSTED_LIMITS["file_bytes"]) or not os.path.samestat(
        before, path.lstat()
    ):
        raise ValueError("Worker diagnostic log is missing, linked, oversized or replaced")
    log.seek(max(0, before.st_size - 6400))
    data = log.read(6400)
    after = os.fstat(log.fileno())
    named = path.lstat()
    if (
        not regular_file(path, HOSTED_LIMITS["file_bytes"])
        or not os.path.samestat(after, named)
        or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        or (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        != (named.st_size, named.st_mtime_ns, named.st_ctime_ns)
    ):
        raise ValueError("Worker diagnostic log changed during readback")
    return data.decode("utf-8", errors="replace")[-1600:]


class Executor:
    def __init__(self, mode: str = "docker") -> None:
        if mode not in {"docker", "trusted_local"}:
            raise ValueError("execution must be docker or trusted_local")
        self.mode = mode
        # Deployment supplies this only after a real image/runtime qualification.
        # No environment variable silently promotes an untested typed worker.
        self._typed_container_verified = False
        self._typed_qualified_image: str | None = None
        self._customer_container_verified = False
        self.customer_container_checks: list[str] = []
        self.typed_container_checks: list[str] = []
        self.typed_container_image = os.environ.get(
            "FORGE_WORKSPACE_IMAGE", "forge-workspace-kernel:local"
        )

    def _docker_path(self, discover: Callable[[str], str | None] | None = None) -> str | None:
        return (shutil.which if discover is None else discover)("docker")

    def _inspect_image(self, docker: str) -> str:
        inspected = subprocess.run(
            [docker, "image", "inspect", self.typed_container_image],
            capture_output=True,
            check=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return str(json.loads(inspected.stdout)[0]["Id"])

    @property
    def typed_container_verified(self) -> bool:
        return self._typed_container_verified

    @property
    def customer_container_verified(self) -> bool:
        return self._customer_container_verified

    def qualify_customer_container(self) -> dict[str, Any]:
        """Exercise native saved proof and print workers; no environment grants this."""
        from forge_mind.workspace.customer_qualification import qualify_customer_container

        self._customer_container_verified = False
        self.customer_container_checks = []
        result = qualify_customer_container(self)
        self.customer_container_checks = result["checks"]
        self._customer_container_verified = True
        return result

    def process_observation(
        self,
        directory: Path,
        payload: dict[str, Any],
        timeout: float | None,
        cancelled: Callable[[], bool],
    ) -> dict[str, Any]:
        """Observe staged saved files inside the same qualified native boundary.

        Paths are rewritten into the disposable mount. Original source files
        never become writable worker inputs and are rechecked after execution.
        """
        from forge_mind.validation_provenance import file_sha256

        operations = {
            "nominal_perforated_box",
            "nominal_through_pattern",
            "photo_profile_analyze",
            "photo_profile_compare",
            "nominal_photo_solid",
            "product_tool",
            "section",
        }
        if payload.get("operation") not in operations or "_operation_job" in payload:
            raise ValueError("Unsupported isolated observer operation")
        if self.mode != "docker" or not self.typed_container_verified:
            raise ValueError("Saved-file observer requires qualified isolated execution")
        if timeout is None or timeout <= 0:
            raise ValueError("Hosted saved-file observers require a positive enclosing deadline")
        timeout = min(float(timeout), 600.0)
        docker = self._docker_path()
        if not docker:
            raise ValueError("Docker is required for isolated saved-file observation")
        started = time.monotonic()
        inputs: dict[Path, str] = {}
        staged_inputs: dict[Path, str] = {}
        with tempfile.TemporaryDirectory(prefix="forge-observer-") as temporary:
            scratch = Path(temporary)

            def stage(source: Path, target: Path) -> None:
                if not regular_file(source, HOSTED_LIMITS["file_bytes"]):
                    raise ValueError("Observer input is missing, linked or oversized")
                expected = file_sha256(source)
                copy_artifact(source, target)
                if file_sha256(target) != expected or file_sha256(source) != expected:
                    raise ValueError("Observer source changed during staging")
                inputs[source] = expected
                staged_inputs[target] = expected
                if directory_size(scratch, MAX_JOB_BYTES) > MAX_JOB_BYTES:
                    raise ValueError("Observer input exceeds its aggregate byte limit")

            for source in directory.iterdir():
                if typed_artifact_name(source.name) or (
                    payload["operation"] == "photo_profile_analyze" and source.name == "vision.png"
                ):
                    stage(source, scratch / source.name)
            isolated = dict(payload)
            if payload.get("source"):
                source_directory = Path(payload["source"])
                staged_source = scratch / "source"
                staged_source.mkdir()
                for source in source_directory.iterdir():
                    if typed_artifact_name(source.name):
                        stage(source, staged_source / source.name)
                isolated["source"] = "/work/source"
            (scratch / "processing.json").write_text(
                json.dumps(isolated, allow_nan=False), encoding="utf-8"
            )
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0 or cancelled():
                raise ValueError("Observer cancelled or deadline exhausted while staging")
            self._run_container(
                docker,
                scratch,
                "forge_mind.workspace.processing",
                "processing.json",
                remaining,
                cancelled,
                image=self.typed_container_image,
            )
            if cancelled() or time.monotonic() - started >= timeout:
                raise ValueError("Observer cancelled or deadline exhausted before readback")
            result = scratch / "processed.json"
            if not regular_file(result, HOSTED_LIMITS["result_bytes"]):
                raise ValueError("Saved-file observer returned no bounded regular result")
            data: dict[str, Any] = json.loads(result.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Saved-file observer result must be an object")
            for source, expected in staged_inputs.items():
                if not regular_file(source) or file_sha256(source) != expected:
                    raise ValueError("Observer changed a staged immutable input")
            for source, expected in inputs.items():
                if not regular_file(source) or file_sha256(source) != expected:
                    raise ValueError("Observer original input changed during execution")
            if payload["operation"] == "section" and (scratch / "section.png").exists():
                stage(scratch / "section.png", directory / "section.png")
            (directory / "processed.json").write_text(
                json.dumps(data, allow_nan=False), encoding="utf-8"
            )
            return data

    def run_typed_tool(
        self,
        directory: Path,
        payload: dict[str, Any],
        *,
        module: str,
        timeout: float,
        cancelled: Callable[[], bool],
    ) -> None:
        """Run an allowlisted native observer over its caller's staged inputs.

        The caller verifies the returned files and commits any immutable receipt.
        Public execution keeps the same qualified worker boundary as CAD builds.
        """
        if module not in {"forge_mind.workspace.print_setup"}:
            raise ValueError("Unsupported typed tool module")
        deadline = end_time(timeout)
        assert deadline is not None
        if cancelled() or expired(deadline):
            raise ValueError("Typed tool cancelled or deadline exhausted")
        request = directory / "request.json"
        with request.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False)
        if self.mode == "docker":
            docker = self._docker_path()
            if not self.typed_container_verified or not docker:
                raise ValueError("Typed tool requires qualified isolated execution")
            self._run_container(
                docker,
                directory,
                module,
                request.name,
                max(0.0, deadline - time.monotonic()),
                cancelled,
                image=self.typed_container_image,
            )
        else:
            bounded_process(
                [sys.executable, "-B", "-m", module, str(request.resolve())],
                directory,
                max(0.0, deadline - time.monotonic()),
                cancelled,
                cpu_limit=2,
            )
        if cancelled() or expired(deadline):
            raise ValueError("Typed tool cancelled or deadline exhausted")

    def qualify_typed_container(self) -> dict[str, Any]:
        """Exercise the exact image and controls; this is not a security certification."""
        self._typed_container_verified = False
        self._typed_qualified_image = None
        self._customer_container_verified = False
        self.typed_container_checks = []
        docker = self._docker_path()
        if self.mode != "docker" or not docker:
            raise ValueError("Docker is required for typed worker qualification")
        identity = self._inspect_image(docker)
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", identity):
            raise ValueError("Worker image has no immutable Docker content identity")
        self.typed_container_image = identity
        with tempfile.TemporaryDirectory(prefix="forge-qualification-") as temporary:
            root = Path(temporary)
            probe = root / "probe"
            probe.mkdir()
            (probe / "probe.json").write_text("{}", encoding="utf-8")
            self._run_container(
                docker,
                probe,
                "forge_mind.workspace.container_probe",
                "probe.json",
                30,
                lambda: False,
                image=identity,
            )
            output = probe / "probe-result.json"
            if not regular_file(output, 4096):
                raise ValueError("Isolation probe produced no bounded result")
            checks = json.loads(output.read_text(encoding="utf-8"))
            required = {
                "unprivileged",
                "readonly_root",
                "no_capabilities",
                "no_new_privileges",
                "no_network_route",
            }
            from forge_mind.workspace.container_probe import HOST_PATHS

            observations = checks.get("host_path_observations")
            if observations is None:
                paths_unavailable = checks.get("host_paths_absent") is True
            else:
                paths_unavailable = (
                    checks.get("host_paths_inaccessible") is True
                    and isinstance(observations, dict)
                    and set(observations) == set(HOST_PATHS)
                    and all(
                        isinstance(value, dict)
                        and (
                            value.get("state") == "absent"
                            or (
                                value.get("state") == "denied"
                                and value.get("error") == "PermissionError"
                            )
                        )
                        for value in observations.values()
                    )
                    and checks.get("host_paths_absent")
                    is all(value["state"] == "absent" for value in observations.values())
                )
            allowed = required | {
                "host_paths_absent",
                "host_paths_inaccessible",
                "host_path_observations",
            }
            if (
                not required.issubset(checks)
                or set(checks) - allowed
                or not all(checks[key] is True for key in required)
                or not paths_unavailable
            ):
                raise ValueError("Worker isolation probe failed")
            from forge_mind.workspace.document import DesignDocument
            from forge_mind.workspace.models import Contract

            directory = root / "candidate"
            directory.mkdir()
            document = DesignDocument.model_validate(
                {
                    "name": "Worker qualification box",
                    "features": [
                        {
                            "id": "base",
                            "name": "Base",
                            "kind": "box",
                            "params": {"x": 10, "y": 20, "z": 30},
                        }
                    ],
                    "outputs": ["base"],
                }
            )
            (directory / "document.json").write_text(document.model_dump_json(), encoding="utf-8")
            (directory / "program.py").write_text(
                "# typed qualification candidate\n", encoding="utf-8"
            )
            self._typed_container_verified = True
            try:
                data = self.process_document(
                    directory,
                    {
                        "revision_id": "a" * 32,
                        "contract": Contract().model_dump(),
                        "document": document.model_dump(),
                        "assets": {},
                    },
                    120,
                    lambda: False,
                )
                if not data["geometry_valid"] or not {
                    "part.step",
                    "part.stl",
                    "part.glb",
                    "render.png",
                    "topology.json",
                }.issubset(data["artifact_hashes"]):
                    raise ValueError("Typed worker failed native build/measurement/export smoke")
                from forge_mind.validation_provenance import file_sha256

                inspection = root / "inspection"
                inspection.mkdir()
                copy_artifact(directory / "topology.json", inspection / "topology.json")
                request = {
                    "revision_id": "a" * 32,
                    "occurrence_path": "qualification",
                    "body_index": 0,
                    "topology_sha256": data["artifact_hashes"]["topology.json"],
                    "step_sha256": data["artifact_hashes"]["part.step"],
                    "document_sha256": data["artifact_hashes"]["document.json"],
                }
                (inspection / "request.json").write_text(json.dumps(request), encoding="utf-8")
                self._run_container(
                    docker,
                    inspection,
                    "forge_mind.workspace.occurrence_render",
                    "request.json",
                    30,
                    lambda: False,
                    image=identity,
                )
                receipt_path, image_path = (
                    inspection / "receipt.json",
                    inspection / "inspection.png",
                )
                if not regular_file(receipt_path, 65536) or not regular_file(
                    image_path, 1024 * 1024
                ):
                    raise ValueError("Worker failed occurrence image smoke")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if any(receipt.get(key) != value for key, value in request.items()) or receipt.get(
                    "image"
                ) != {"file": "inspection.png", "sha256": file_sha256(image_path)}:
                    raise ValueError("Worker occurrence image has inconsistent provenance")
                for kind in ("cancel", "timeout"):
                    blocked = root / kind
                    blocked.mkdir()
                    (blocked / "probe.json").write_text('{"sleep": 60}', encoding="utf-8")
                    started = time.monotonic()

                    def stopped(probe_kind: str = kind, probe_start: float = started) -> bool:
                        return probe_kind == "cancel" and time.monotonic() - probe_start > 0.5

                    try:
                        self._run_container(
                            docker,
                            blocked,
                            "forge_mind.workspace.container_probe",
                            "probe.json",
                            0.5 if kind == "timeout" else 30,
                            stopped,
                            image=identity,
                        )
                    except ValueError as exc:
                        expected = "wall-clock" if kind == "timeout" else "cancelled"
                        if expected not in str(exc) or "cleanup failed" in str(exc):
                            raise
                    else:
                        raise ValueError("Worker failed stop/reap probe")
                    if (blocked / "probe-result.json").exists():
                        raise ValueError("Stopped worker returned partial success")
            except BaseException:
                self._typed_container_verified = False
                raise
        self.typed_container_checks = [
            *sorted(required),
            "host_paths_absent"
            if checks.get("host_paths_absent") is True
            else "host_paths_inaccessible",
            "typed_native_measurement_export",
            "occurrence_mesh_image",
            "cancel_reap",
            "deadline_reap",
            "no_partial_success",
        ]
        self._typed_qualified_image = identity
        return {
            "image": identity,
            "checks": self.typed_container_checks,
            "isolation_probe": checks,
            "scope": "Runtime smoke and control probes; adversarial native parser security is untested",
        }

    def process_document(
        self,
        directory: Path,
        payload: dict[str, Any],
        timeout: float | None,
        cancelled: Callable[[], bool],
        *,
        cache_input: TypedCacheInput | None = None,
        origin: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compile and measure typed input in one disposable isolated worker."""
        if origin is not None:
            origin.clear()
        if self.mode != "docker" or not self.typed_container_verified:
            raise ValueError(
                "Typed container execution requires verified image/runtime qualification"
            )
        if timeout is None or not 0 < timeout <= 600:
            raise ValueError("Hosted typed jobs require a deadline of at most 600 seconds")
        docker = self._docker_path()
        if not docker:
            raise ValueError("Docker is required for isolated typed execution")
        if cancelled():
            raise ValueError("Operation cancelled")
        started = time.monotonic()
        image = self.typed_container_image
        qualified_image = self._typed_qualified_image
        if cache_input is not None and qualified_image != image:
            raise ValueError("Typed cache requires the originally qualified image identity")
        with tempfile.TemporaryDirectory(prefix="forge-typed-") as temporary:
            scratch = Path(temporary)
            for name in ("program.py", "document.json"):
                if not regular_file(directory / name, HOSTED_LIMITS["file_bytes"]):
                    raise ValueError("Typed worker input is missing, linked, or oversized")
                copy_artifact(directory / name, scratch / name)
            assets = {}
            for key, source in payload.get("assets", {}).items():
                if not re.fullmatch(r"[a-f0-9]{32}", key):
                    raise ValueError("Invalid owned asset identifier")
                source_path = Path(source)
                if source_path.resolve().parent != directory.resolve():
                    raise ValueError("Typed worker asset must be staged in its owned revision")
                name = "input_" + key + ".step"
                if not regular_file(source_path, HOSTED_LIMITS["file_bytes"]):
                    raise ValueError("Typed worker asset is missing, linked, or oversized")
                copy_artifact(source_path, scratch / name)
                assets[key] = "/work/" + name
                if directory_size(scratch, MAX_JOB_BYTES) > MAX_JOB_BYTES:
                    raise ValueError("Typed worker staged input exceeds its byte limit")
            cache_source = stage_cache(
                cache_input, directory, scratch, image, started + timeout, cancelled
            )
            isolated = {
                key: payload[key]
                for key in ("revision_id", "contract", "document")
                if key in payload
            }
            isolated.update(assets=assets, cache_source=cache_source)
            (scratch / "processing.json").write_text(json.dumps(isolated), encoding="utf-8")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0 or cancelled():
                raise ValueError("Operation cancelled or deadline exhausted while staging")
            self._run_container(
                docker,
                scratch,
                "forge_mind.workspace.processing",
                "processing.json",
                remaining,
                cancelled,
                image=image,
            )
            result = scratch / "processed.json"
            if not regular_file(result, HOSTED_LIMITS["result_bytes"]):
                raise ValueError("Typed container produced no bounded regular measurement")
            data: dict[str, Any] = json.loads(result.read_text(encoding="utf-8"))
            from forge_mind.validation_provenance import file_sha256
            from forge_mind.workspace.models import Contract, Measurement

            measured = Measurement.model_validate(data)
            if measured.revision_id != payload["revision_id"]:
                raise ValueError("Typed worker measurement belongs to another revision")
            if measured.contract_hash != Contract.model_validate(payload["contract"]).fingerprint():
                raise ValueError("Typed worker measurement belongs to another contract")
            # This verifies byte consistency, not independence of worker evidence.
            for name in ("program.py", "document.json"):
                if measured.artifact_hashes.get(name) != file_sha256(directory / name):
                    raise ValueError("Typed worker changed or omitted immutable input evidence")
            for name, digest in measured.artifact_hashes.items():
                if not typed_artifact_name(name):
                    raise ValueError("Invalid worker artifact name")
                source_path = scratch / name
                if not regular_file(source_path, HOSTED_LIMITS["file_bytes"]):
                    raise ValueError("Worker artifact is linked, missing, or oversized")
                if file_sha256(source_path) != digest:
                    raise ValueError("Worker artifact does not match measurement")
                target = directory / name
                if target.exists():
                    if not regular_file(target) or file_sha256(target) != digest:
                        raise ValueError("Typed worker changed immutable input")
                else:
                    copy_artifact(source_path, target)
            if cancelled() or time.monotonic() >= started + timeout:
                raise ValueError("Typed worker cancelled or deadline exhausted during transfer")
            if origin is not None and qualified_image == image:
                origin.update(execution_origin(image, measured, payload.get("document")))
            return measured.model_dump()

    @staticmethod
    def _run_container(
        docker: str,
        directory: Path,
        module: str,
        request: str,
        timeout: float,
        cancelled: Callable[[], bool],
        *,
        image: str,
    ) -> None:
        """Enforce the hosted envelope across compilation, parsing and measurement."""
        if cancelled():
            raise ValueError("Operation cancelled")
        if os.name != "nt":
            directory.chmod(0o777)
        name = "forge-typed-" + uuid.uuid4().hex
        cmd = [
            docker,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(HOSTED_LIMITS["pids"]),
            "--memory",
            str(HOSTED_LIMITS["memory_bytes"]),
            "--cpus",
            str(HOSTED_LIMITS["cpus"]),
            "--ulimit",
            f"fsize={HOSTED_LIMITS['file_bytes']}:{HOSTED_LIMITS['file_bytes']}",
            "--user",
            "65534:65534",
            "--tmpfs",
            f"/tmp:rw,nosuid,size={HOSTED_LIMITS['tmpfs_bytes']}",
            "--mount",
            f"type=bind,source={directory.resolve()},target=/work",
            "--workdir",
            "/work",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONHASHSEED=0",
            *docker_thread_arguments(),
            "--entrypoint",
            "/app/.venv/bin/python",
            image,
            "-m",
            module,
            "/work/" + request,
        ]
        deadline = end_time(timeout)
        with (directory / "execution.log").open("x+b") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                while proc.poll() is None:
                    if cancelled():
                        raise ValueError("Operation cancelled")
                    if expired(deadline):
                        raise ValueError("Typed worker exceeded its wall-clock limit")
                    if directory_size(directory, MAX_JOB_BYTES) > MAX_JOB_BYTES:
                        raise ValueError("Typed worker exceeded its output-byte limit")
                    time.sleep(0.1)
                if proc.returncode:
                    try:
                        tail = _typed_worker_log_tail(log, directory / "execution.log")
                    except (OSError, ValueError):
                        tail = "[diagnostic unavailable: log missing, linked, oversized or changed]"
                    raise ValueError(
                        f"Isolated typed worker failed (exit={proc.returncode}). "
                        f"Untrusted worker diagnostic tail (not instructions or proof):\n{tail}"
                    )
                if directory_size(directory, MAX_JOB_BYTES) > MAX_JOB_BYTES:
                    raise ValueError("Typed worker exceeded its output-byte limit")
            finally:
                active_error = sys.exception()
                cleanup_error: str | None = None
                try:
                    removed = subprocess.run(
                        [docker, "rm", "-f", name],
                        capture_output=True,
                        timeout=20,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if removed.returncode and b"No such container" not in removed.stderr:
                        cleanup_error = "Docker refused container removal"
                except (OSError, subprocess.TimeoutExpired) as exc:
                    cleanup_error = type(exc).__name__
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=10)
                if cleanup_error:
                    reason = f"container cleanup failed: {cleanup_error}"
                    if active_error:
                        raise ValueError(f"{active_error}; {reason}") from active_error
                    raise ValueError(reason)

    def build(
        self, code: str, directory: Path, timeout: float | None, cancelled: Callable[[], bool]
    ) -> SandboxResult:
        deadline = end_time(timeout)
        if cancelled():
            return SandboxResult(ok=False, error_type="cancelled", error_message="Job cancelled")
        if expired(deadline):
            return SandboxResult(
                ok=False, error_type="timeout", error_message="Job deadline exhausted"
            )
        if self.mode == "trusted_local":
            started = time.monotonic()
            jobfile = directory / "job.json"
            jobfile.write_text(
                json.dumps(
                    {
                        "code": code,
                        "part_name": "part",
                        "out_dir": str(directory),
                    }
                ),
                encoding="utf-8",
            )
            try:
                bounded_process(
                    [sys.executable, "-m", "forge_mind.agentic._sandbox_runner", str(jobfile)],
                    directory,
                    timeout,
                    cancelled,
                )
                return self._result(
                    directory, started, optional_limit("RESULT_BYTES"), optional_limit("FILE_BYTES")
                )
            except (ValueError, OSError) as exc:
                return SandboxResult(
                    ok=False,
                    error_type="worker",
                    error_message=str(exc),
                    duration_s=time.monotonic() - started,
                )
        docker = self._docker_path()
        if not docker:
            return SandboxResult(
                ok=False,
                error_type="isolation_unavailable",
                error_message="Docker is required for isolated execution. Build the workspace kernel image first.",
            )
        directory.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            # A disposable worker directory only, never the host-owned revision directory.
            directory.chmod(0o777)
        job = {
            "code": code,
            "part_name": "part",
            "out_dir": "/work",
            "memory_mb": HOSTED_LIMITS["memory_bytes"] // (1024 * 1024),
        }
        (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
        name = f"forge-cad-{uuid.uuid4().hex}"
        cmd = [
            docker,
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(HOSTED_LIMITS["pids"]),
            "--memory",
            str(HOSTED_LIMITS["memory_bytes"]),
            "--cpus",
            str(HOSTED_LIMITS["cpus"]),
            "--ulimit",
            f"fsize={HOSTED_LIMITS['file_bytes']}:{HOSTED_LIMITS['file_bytes']}",
            "--user",
            "65534:65534",
            "--tmpfs",
            f"/tmp:rw,nosuid,size={HOSTED_LIMITS['tmpfs_bytes']}",
            "--mount",
            f"type=bind,source={directory.resolve()},target=/work",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONHASHSEED=0",
            *docker_thread_arguments(),
            os.environ.get("FORGE_WORKSPACE_IMAGE", "forge-workspace-kernel:local"),
            "/app/.venv/bin/python",
            "-m",
            "forge_mind.agentic._sandbox_runner",
            "/work/job.json",
        ]
        started = time.monotonic()
        with (directory / "execution.log").open("wb") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                while proc.poll() is None:
                    if (
                        cancelled()
                        or expired(deadline)
                        or directory_size(directory, MAX_JOB_BYTES) > MAX_JOB_BYTES
                    ):
                        return SandboxResult(
                            ok=False,
                            error_type="cancelled" if cancelled() else "timeout",
                            error_message="Execution stopped at its resource limit",
                        )
                    time.sleep(0.1)
            finally:
                subprocess.run(
                    [docker, "rm", "-f", name],
                    capture_output=True,
                    timeout=20,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=10)
        if directory_size(directory, MAX_JOB_BYTES) > MAX_JOB_BYTES:
            return SandboxResult(
                ok=False,
                error_type="resource_limit",
                error_message="Execution exceeded its hosted output-byte limit",
                duration_s=time.monotonic() - started,
            )
        return self._result(directory, started)

    @staticmethod
    def _result(
        directory: Path,
        started: float,
        maximum: int | None = HOSTED_LIMITS["result_bytes"],
        artifact_maximum: int | None = HOSTED_LIMITS["file_bytes"],
    ) -> SandboxResult:
        result = directory / "result.json"
        if not regular_file(result, maximum):
            return SandboxResult(
                ok=False,
                error_type="harness",
                error_message="Isolated worker produced no result; check kernel image and writable job volume",
            )
        data = json.loads(result.read_text(encoding="utf-8"))
        ok = (
            bool(data.get("ok"))
            and regular_file(directory / "part.stl", artifact_maximum)
            and regular_file(directory / "part.step", artifact_maximum)
        )
        return SandboxResult(
            ok=ok,
            stl_path=directory / "part.stl" if ok else None,
            step_path=directory / "part.step" if ok else None,
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            duration_s=time.monotonic() - started,
        )
