"""Executor adapter that sends every Docker operation to the restricted local runner."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge_mind.agentic.sandbox import SandboxResult
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.hosting import HostingError
from forge_mind.workspace.hosting_runner import OPERATIONS, RunnerClient, RunnerUncertainError
from forge_mind.workspace.processing import HOSTED_LIMITS


class BrokerExecutor(Executor):
    def __init__(
        self,
        client: RunnerClient,
        image: str,
        *,
        on_uncertain: Callable[[RunnerUncertainError], None] | None = None,
    ) -> None:
        super().__init__("docker")
        self.client = client
        self.typed_container_image = image
        self.runner_verified = False
        self.last_runner_receipt: dict[str, Any] | None = None
        self.on_uncertain = on_uncertain

    def _docker_path(self, discover: Callable[[str], str | None] | None = None) -> str | None:
        return "restricted-runner-v1"

    def build(
        self, code: str, directory: Path, timeout: float | None, cancelled: Callable[[], bool]
    ) -> SandboxResult:
        """Cover the historical direct-Docker build path without exposing Docker."""
        started = time.monotonic()
        if timeout is None or not 0 < timeout <= 600:
            raise HostingError("Broker builds require a finite worker deadline at most 600 seconds")
        if cancelled():
            return SandboxResult(ok=False, error_type="cancelled", error_message="Job cancelled")
        directory.mkdir(parents=True, exist_ok=True)
        job = {
            "code": code,
            "part_name": "part",
            "out_dir": "/work",
            "memory_mb": HOSTED_LIMITS["memory_bytes"] // (1024 * 1024),
        }
        (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
        try:
            self._run_container(
                "restricted-runner-v1",
                directory,
                "forge_mind.agentic._sandbox_runner",
                "job.json",
                timeout,
                cancelled,
                image=self.typed_container_image,
            )
        except RunnerUncertainError:
            raise
        except HostingError as exc:
            return SandboxResult(
                ok=False,
                error_type="cancelled" if cancelled() else "worker",
                error_message=str(exc),
                duration_s=time.monotonic() - started,
            )
        return self._result(directory, started)

    def _inspect_image(self, docker: str) -> str:
        self.runner_verified = False
        receipt = self.client.request("identity")
        if (
            receipt.get("version") != 1
            or receipt.get("image") != self.typed_container_image
            or receipt.get("fence") != self.client.fence
            or receipt.get("restricted") is not True
        ):
            raise HostingError("Restricted runner identity differs from the configured authority")
        self.runner_verified = True
        return self.typed_container_image

    # Keep the historical class-call static API on Executor. Instance dispatch
    # deliberately binds the configured broker, and never calls that static API.
    def _run_container(  # type: ignore[override]
        self,
        docker: str,
        directory: Path,
        module: str,
        request: str,
        timeout: float,
        cancelled: Callable[[], bool],
        *,
        image: str,
    ) -> None:
        if image != self.typed_container_image:
            raise HostingError("Broker execution image differs from its frozen identity")
        kind = next((key for key, value in OPERATIONS.items() if value == (module, request)), None)
        if kind is None:
            raise HostingError("Broker operation is not allowlisted")
        started = time.monotonic()
        try:
            receipt = self.client.execute(directory, kind, timeout, cancelled)
        except RunnerUncertainError as exc:
            self.runner_verified = False
            if self.on_uncertain is not None:
                self.on_uncertain(exc)
            raise
        self.last_runner_receipt = receipt
        # Existing callers may use this log filename as bounded diagnostic output.
        log = directory / "execution.log"
        if not log.exists():
            log.write_text(
                json.dumps(
                    {"operation_id": receipt["operation_id"], "seconds": time.monotonic() - started}
                ),
                encoding="utf-8",
            )
