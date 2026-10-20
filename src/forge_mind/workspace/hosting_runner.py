"""Restricted local runner v1. Only this dedicated-UID process can access Docker.

The public process has an opaque staging ticket, never daemon credentials. Durable
launch authority precedes Docker create. An uncertain launch is never repeated.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import socket
import socketserver
import sqlite3
import stat
import struct
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from forge_mind.workspace.hosting import HostingError
from forge_mind.workspace.hosting_backup import inventory, relative_file
from forge_mind.workspace.processing import HOSTED_LIMITS, MAX_JOB_BYTES
from forge_mind.workspace.worker_threads import docker_thread_arguments

OPERATIONS = {
    "process": ("forge_mind.workspace.processing", "processing.json"),
    "print": ("forge_mind.workspace.print_setup", "request.json"),
    "probe": ("forge_mind.workspace.container_probe", "probe.json"),
    "occurrence": ("forge_mind.workspace.occurrence_render", "request.json"),
    "legacy_build": ("forge_mind.agentic._sandbox_runner", "job.json"),
    "access": ("forge_mind.workspace.geometry_access_worker", "request.json"),
    "distance": ("forge_mind.workspace.geometry_distance_worker", "request.json"),
    "neighbors": ("forge_mind.workspace.geometry_neighbors_worker", "request.json"),
    "section": ("forge_mind.workspace.section_render", "request.json"),
    "photo": ("forge_mind.workspace.product_photo_render", "request.json"),
}
_IDENTIFIER = re.compile(r"[a-f0-9]{32}")
_WIRE_MAXIMUM = 4 * 1024 * 1024
_POSIX_OS: Any = os
_UNIX_SOCKET: Any = socket


class RunnerUncertainError(HostingError):
    """Keep compute/launch authority reserved until offline reconciliation."""

    def __init__(self, message: str, *, operation_id: str = "", request_hash: str = "") -> None:
        super().__init__(message)
        self.operation_id, self.request_hash = operation_id, request_hash


class RunnerRejectedError(HostingError):
    """The authenticated runner acknowledged that it committed no launch authority."""


def request_digest(request: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _relative_open(root_fd: int, name: str, flags: int, *, create_parents: bool = False) -> int:
    """Walk each component beneath an already-open ticket, without following links."""
    parts = relative_file(name).parts
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            if create_parents:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, mode=0o770, dir_fd=current)
            child = os.open(
                part, os.O_RDONLY | _POSIX_OS.O_DIRECTORY | _POSIX_OS.O_NOFOLLOW, dir_fd=current
            )
            os.close(current)
            current = child
        descriptor = os.open(parts[-1], flags | _POSIX_OS.O_NOFOLLOW, 0o660, dir_fd=current)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > HOSTED_LIMITS["file_bytes"]
        ):
            os.close(descriptor)
            raise HostingError("Runner ticket contains a linked, special or oversized file")
        return descriptor
    finally:
        os.close(current)


def _copy_input(root_fd: int, name: str, expected: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _relative_open(root_fd, name, os.O_RDONLY | _POSIX_OS.O_NONBLOCK)
    digest, count = hashlib.sha256(), 0
    with os.fdopen(descriptor, "rb") as source, target.open("xb") as copied:
        while block := source.read(1024 * 1024):
            count += len(block)
            if count > HOSTED_LIMITS["file_bytes"]:
                raise HostingError("Staged input grew beyond its file limit")
            digest.update(block)
            copied.write(block)
    if expected != {"bytes": count, "sha256": digest.hexdigest()}:
        raise HostingError("Staged input changed before launch")


@dataclass(frozen=True)
class RunnerPolicy:
    root: Path
    staging_root: Path
    image: str
    fence: str
    gateway_uid: int
    max_seconds: float = 600

    def __post_init__(self) -> None:
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", self.image) or not _IDENTIFIER.fullmatch(
            self.fence
        ):
            raise HostingError("Runner requires an immutable image and operator fence")
        if not self.root.is_absolute() or not self.staging_root.is_absolute():
            raise HostingError("Runner roots must be absolute")
        first, second = self.root.resolve(), self.staging_root.resolve()
        if first == second or first in second.parents or second in first.parents:
            raise HostingError("Runner private state and public staging must be disjoint")
        if not 0 < self.max_seconds <= 600 or self.gateway_uid < 0:
            raise HostingError("Invalid runner resource or caller identity policy")


class DockerBackend:
    """No method accepts an image, module, Docker flag or host path from IPC."""

    def __init__(self, policy: RunnerPolicy) -> None:
        self.policy = policy
        self.docker = shutil.which("docker")
        if not self.docker:
            raise HostingError("Dedicated runner requires Docker")

    def _call(
        self, arguments: list[str], timeout: float = 20
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.docker or "docker", *arguments], capture_output=True, check=True, timeout=timeout
        )

    def identity(self) -> str:
        result = self._call(["image", "inspect", self.policy.image])
        identity = str(json.loads(result.stdout)[0]["Id"])
        if identity != self.policy.image:
            raise HostingError("Runner image differs from the pinned content identity")
        return identity

    def create(self, name: str, directory: Path, kind: str, operation_id: str) -> str:
        module, request = OPERATIONS[kind]
        directory.chmod(0o777)
        arguments = [
            "create",
            "--name",
            name,
            "--label",
            "cadkitty.operation=" + operation_id,
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
            self.policy.image,
            "-m",
            module,
            "/work/" + request,
        ]
        identity = self._call(arguments).stdout.decode().strip()
        if not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise HostingError("Docker did not return a container identity")
        return identity

    def start(self, identity: str) -> None:
        self._call(["start", identity])

    def poll(self, identity: str) -> int | None:
        result = json.loads(self._call(["inspect", identity]).stdout)[0]["State"]
        return None if result["Running"] else int(result["ExitCode"])

    def remove(self, identity: str) -> None:
        self._call(["rm", "-f", identity])

    def reconcile(self, name: str, operation_id: str, expected: str | None) -> bool:
        try:
            observed = json.loads(self._call(["inspect", name]).stdout)[0]
        except subprocess.CalledProcessError as exc:
            if b"No such object" in exc.stderr or b"No such container" in exc.stderr:
                return False
            raise HostingError("Runner cannot establish container absence") from exc
        if (
            observed.get("Image") != self.policy.image
            or observed.get("Config", {}).get("Labels", {}).get("cadkitty.operation")
            != operation_id
            or (expected is not None and observed.get("Id") != expected)
        ):
            raise HostingError("Reconciliation container identity differs from the launch receipt")
        self.remove(str(observed["Id"]))
        return True


class Runner:
    def __init__(self, policy: RunnerPolicy, backend: Any | None = None) -> None:
        self.policy = policy
        self.backend = DockerBackend(policy) if backend is None else backend
        self.policy.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.policy.root.chmod(0o700)
        self.database = self.policy.root / "operations.sqlite3"
        self._lock = threading.RLock()
        self._cancel: dict[str, threading.Event] = {}
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS operations (id TEXT PRIMARY KEY, fence TEXT NOT NULL, request_hash TEXT NOT NULL, receipt TEXT NOT NULL)"
            )
            # Restart does not prove whether Docker create/start completed.
            for row in con.execute("SELECT id,receipt FROM operations").fetchall():
                receipt = json.loads(row[1])
                if receipt["state"] not in {
                    "done",
                    "cancelled",
                    "error",
                    "interrupted",
                    "rejected",
                }:
                    receipt["state"] = "uncertain"
                    con.execute(
                        "UPDATE operations SET receipt=? WHERE id=?", (json.dumps(receipt), row[0])
                    )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.database, timeout=20)
        con.execute("PRAGMA synchronous=FULL")
        return con

    def _read(self, operation_id: str) -> dict[str, Any]:
        with self._connect() as con:
            row = con.execute(
                "SELECT receipt FROM operations WHERE id=? AND fence=?",
                (operation_id, self.policy.fence),
            ).fetchone()
        if row is None:
            raise HostingError("Unknown runner operation")
        result: dict[str, Any] = json.loads(row[0])
        return result

    def _save(self, receipt: dict[str, Any]) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE operations SET receipt=? WHERE id=? AND fence=?",
                (json.dumps(receipt), receipt["operation_id"], self.policy.fence),
            )

    def reconcile(
        self, operation_id: str, *, unaccepted_request_hash: str | None = None
    ) -> dict[str, Any]:
        """Operator-only, with the serving runner stopped; never exposed over IPC."""
        receipt: dict[str, Any]
        if not _IDENTIFIER.fullmatch(operation_id):
            raise HostingError("Invalid reconciliation operation identity")
        with self._connect() as con:
            row = con.execute(
                "SELECT receipt FROM operations WHERE id=?", (operation_id,)
            ).fetchone()
        if row is None:
            if unaccepted_request_hash is None or not re.fullmatch(
                r"[a-f0-9]{64}", unaccepted_request_hash
            ):
                raise HostingError("An unknown operation requires its exact retained request hash")
            found = self.backend.reconcile("cadkitty-" + operation_id, operation_id, None)
            receipt = {
                "version": 1,
                "operation_id": operation_id,
                "fence": self.policy.fence,
                "image": self.policy.image,
                "container_name": "cadkitty-" + operation_id,
                "container_id": None,
                "request_hash": unaccepted_request_hash,
                "state": "interrupted" if found else "rejected",
                "accepted": bool(found),
                "outputs": {},
                "error": "Offline exact nonacceptance/absence reconciliation; no automatic rerun",
            }
            with self._connect() as con:
                con.execute(
                    "INSERT INTO operations VALUES (?,?,?,?)",
                    (operation_id, self.policy.fence, unaccepted_request_hash, json.dumps(receipt)),
                )
            return receipt
        receipt = json.loads(row[0])
        if receipt["fence"] != self.policy.fence or (
            unaccepted_request_hash is not None
            and receipt["request_hash"] != unaccepted_request_hash
        ):
            raise HostingError("Reconciliation differs from retained launch authority")
        if receipt["state"] == "rejected":
            return receipt
        if receipt["state"] != "uncertain":
            raise HostingError("Only an uncertain abandoned launch can be reconciled")
        self.backend.reconcile(receipt["container_name"], operation_id, receipt["container_id"])
        receipt.update(
            state="interrupted",
            outputs={},
            error="Abandoned container reconciled; no automatic rerun",
        )
        self._save(receipt)
        return receipt

    def respond(self, request: dict[str, Any]) -> dict[str, Any]:
        """Classify refusal only while no launch receipt exists under the admission lock."""
        with self._lock:
            try:
                return self.dispatch(request)
            except Exception as exc:
                operation_id = request.get("operation_id")
                if (
                    request.get("command") == "launch"
                    and request.get("version") == 1
                    and request.get("fence") == self.policy.fence
                    and isinstance(operation_id, str)
                    and _IDENTIFIER.fullmatch(operation_id)
                ):
                    with self._connect() as con:
                        accepted = con.execute(
                            "SELECT 1 FROM operations WHERE id=?", (operation_id,)
                        ).fetchone()
                    if accepted is None:
                        return {
                            "version": 1,
                            "operation_id": operation_id,
                            "fence": self.policy.fence,
                            "request_hash": request_digest(request),
                            "state": "rejected",
                            "accepted": False,
                            "preaccept_rejection": True,
                            "error": str(exc)[:300],
                        }
                return {"protocol_error": True, "error": str(exc)[:300]}

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("version") != 1 or request.get("fence") != self.policy.fence:
            raise HostingError("Runner protocol or fencing authority differs")
        command = request.get("command")
        base = {"version", "fence", "command"}
        if command == "identity":
            if set(request) != base:
                raise HostingError("Unexpected runner identity fields")
            return {
                "version": 1,
                "image": self.backend.identity(),
                "fence": self.policy.fence,
                "restricted": True,
            }
        operation_id = request.get("operation_id", "")
        if not isinstance(operation_id, str) or not _IDENTIFIER.fullmatch(operation_id):
            raise HostingError("Invalid runner operation identity")
        if command in {"status", "cancel"}:
            if set(request) != base | {"operation_id"}:
                raise HostingError("Unexpected runner status fields")
            if command == "cancel":
                with self._lock:
                    event = self._cancel.get(operation_id)
                    if event is not None:
                        event.set()
            return self._read(operation_id)
        if command != "launch" or set(request) != base | {
            "operation_id",
            "operation_kind",
            "staging_ticket",
            "inputs",
            "remaining_seconds",
        }:
            raise HostingError("Unsupported runner command or fields")
        kind, ticket, seconds = (
            request["operation_kind"],
            request["staging_ticket"],
            request["remaining_seconds"],
        )
        if (
            not isinstance(kind, str)
            or kind not in OPERATIONS
            or not isinstance(ticket, str)
            or not _IDENTIFIER.fullmatch(ticket)
        ):
            raise HostingError("Unsupported operation kind or staging ticket")
        if (
            type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or not 0 < seconds <= self.policy.max_seconds
        ):
            raise HostingError("Runner work requires a bounded remaining deadline")
        staged = self.policy.staging_root / ticket
        if staged.is_symlink() or not staged.is_dir():
            raise HostingError("Runner staging ticket is missing or linked")
        files = request["inputs"]
        if (
            not isinstance(files, dict)
            or not 1 <= len(files) <= 16384
            or OPERATIONS[kind][1] not in files
        ):
            raise HostingError("Runner requires its bounded input manifest")
        total = 0
        for name, details in files.items():
            relative_file(name)
            if (
                not isinstance(details, dict)
                or set(details) != {"bytes", "sha256"}
                or type(details["bytes"]) is not int
                or not 0 <= details["bytes"] <= HOSTED_LIMITS["file_bytes"]
                or not isinstance(details["sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", details["sha256"])
            ):
                raise HostingError("Invalid runner input manifest entry")
            total += details["bytes"]
        if total > MAX_JOB_BYTES:
            raise HostingError("Runner inputs exceed the aggregate byte limit")
        fingerprint = request_digest(request)
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            prior = con.execute(
                "SELECT request_hash,receipt FROM operations WHERE id=?", (operation_id,)
            ).fetchone()
            if prior is not None:
                if prior[0] != fingerprint:
                    raise HostingError("Runner operation replay changed its authority")
                previous: dict[str, Any] = json.loads(prior[1])
                if previous["state"] == "uncertain":
                    raise HostingError("Uncertain runner launch requires offline reconciliation")
                return previous
            existing = con.execute("SELECT receipt FROM operations").fetchall()
            if any(
                json.loads(row[0])["state"] in {"prepared", "created", "running", "uncertain"}
                for row in existing
            ):
                raise HostingError("Restricted runner already has active or unreconciled work")
            if (
                len(existing) >= 4096
                or shutil.disk_usage(self.policy.root).free < 2 * MAX_JOB_BYTES
            ):
                raise HostingError("Runner retention or free-space policy requires maintenance")
            receipt = {
                "version": 1,
                "operation_id": operation_id,
                "fence": self.policy.fence,
                "state": "prepared",
                "image": self.policy.image,
                "container_name": "cadkitty-" + operation_id,
                "container_id": None,
                "request_hash": fingerprint,
                "outputs": {},
                "error": None,
            }
            con.execute(
                "INSERT INTO operations VALUES (?,?,?,?)",
                (operation_id, self.policy.fence, fingerprint, json.dumps(receipt)),
            )
        # The committed prepared record must exist before any Docker side effect.
        event = threading.Event()
        with self._lock:
            self._cancel[operation_id] = event
        thread = threading.Thread(target=self._execute, args=(request, receipt, event), daemon=True)
        thread.start()
        return receipt

    def _execute(
        self, request: dict[str, Any], receipt: dict[str, Any], cancelled: threading.Event
    ) -> None:
        operation_id = receipt["operation_id"]
        private = self.policy.root / operation_id
        public = self.policy.staging_root / request["staging_ticket"]
        identity: str | None = None
        launch_attempted = False
        ticket_fd: int | None = None
        started = time.monotonic()
        try:
            private.mkdir(mode=0o700)
            ticket_fd = os.open(public, os.O_RDONLY | _POSIX_OS.O_DIRECTORY | _POSIX_OS.O_NOFOLLOW)
            for name in request["inputs"]:
                _copy_input(ticket_fd, name, request["inputs"][name], private / name)
            if inventory(private, maximum=MAX_JOB_BYTES) != request["inputs"]:
                raise HostingError("Staged inputs changed before isolated launch")
            launch_attempted = True
            identity = self.backend.create(
                receipt["container_name"], private, request["operation_kind"], operation_id
            )
            receipt.update(state="created", container_id=identity)
            self._save(receipt)
            if cancelled.is_set() or time.monotonic() - started >= request["remaining_seconds"]:
                raise HostingError("Operation cancelled or deadline exhausted before start")
            self.backend.start(identity)
            receipt["state"] = "running"
            self._save(receipt)
            while (code := self.backend.poll(identity)) is None:
                if cancelled.is_set():
                    raise HostingError("Operation cancelled")
                if time.monotonic() - started >= request["remaining_seconds"]:
                    raise HostingError("Worker exceeded its wall-clock limit")
                inventory(private, maximum=MAX_JOB_BYTES)
                cancelled.wait(0.1)
            if code != 0:
                raise HostingError("Isolated worker exited unsuccessfully")
            if cancelled.is_set() or time.monotonic() - started >= request["remaining_seconds"]:
                raise HostingError("Operation cancelled or deadline exhausted before readback")
            outputs = inventory(private, maximum=MAX_JOB_BYTES)
            for name in outputs:
                source = private / name
                if name in request["inputs"] and outputs[name] != request["inputs"][name]:
                    raise HostingError("Worker modified an immutable input")
                if name not in request["inputs"]:
                    output_fd = _relative_open(
                        ticket_fd, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, create_parents=True
                    )
                    with source.open("rb") as original, os.fdopen(output_fd, "wb") as copied:
                        shutil.copyfileobj(original, copied)
            receipt.update(state="done", outputs=outputs)
        except Exception as exc:
            state = (
                "uncertain"
                if launch_attempted and identity is None
                else "cancelled"
                if cancelled.is_set()
                else "error"
            )
            receipt.update(state=state, outputs={}, error=str(exc)[:300])
        finally:
            try:
                if identity is not None:
                    self.backend.remove(identity)
            except Exception:
                receipt.update(
                    state="uncertain",
                    outputs={},
                    error="Container cleanup requires offline reconciliation",
                )
            self._save(receipt)
            if ticket_fd is not None:
                os.close(ticket_fd)
            with self._lock:
                self._cancel.pop(operation_id, None)


class RunnerClient:
    def __init__(self, socket_path: Path, staging_root: Path, fence: str) -> None:
        self.socket_path, self.staging_root, self.fence = socket_path, staging_root, fence

    def request(self, command: str, **payload: Any) -> dict[str, Any]:
        message = {"version": 1, "fence": self.fence, "command": command, **payload}
        wire = json.dumps(message, allow_nan=False).encode() + b"\n"
        if len(wire) > _WIRE_MAXIMUM:
            raise HostingError("Runner request exceeds its wire limit")
        with socket.socket(_UNIX_SOCKET.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(20)
            channel.connect(str(self.socket_path))
            channel.sendall(wire)
            with channel.makefile("rb") as response:
                data = response.readline(_WIRE_MAXIMUM + 1)
        if len(data) > _WIRE_MAXIMUM or not data.endswith(b"\n"):
            raise HostingError("Invalid bounded runner response")
        return self.decode_response(message, data)

    @staticmethod
    def decode_response(message: dict[str, Any], data: bytes) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(data)
        if not isinstance(result, dict):
            raise HostingError("Malformed runner response")
        if result.get("preaccept_rejection") is True:
            if (
                message.get("command") != "launch"
                or result.get("version") != 1
                or result.get("operation_id") != message.get("operation_id")
                or result.get("fence") != message.get("fence")
                or result.get("request_hash") != request_digest(message)
                or result.get("state") != "rejected"
                or result.get("accepted") is not False
            ):
                raise HostingError("Unbound runner rejection response")
            raise RunnerRejectedError(
                str(result.get("error", "Runner declined launch before acceptance"))
            )
        if "error" in result and result.get("protocol_error"):
            raise HostingError(str(result["error"]))
        return result

    def execute(
        self, directory: Path, kind: str, seconds: float, cancelled: Callable[[], bool]
    ) -> dict[str, Any]:
        ticket, operation_id = uuid.uuid4().hex, uuid.uuid4().hex
        staging = self.staging_root / ticket
        inputs = inventory(directory, maximum=MAX_JOB_BYTES)
        staging.mkdir(mode=0o770)
        for name in inputs:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / name, target)
        message = {
            "version": 1,
            "fence": self.fence,
            "command": "launch",
            "operation_id": operation_id,
            "operation_kind": kind,
            "staging_ticket": ticket,
            "inputs": inputs,
            "remaining_seconds": seconds,
        }
        fingerprint = request_digest(message)

        def uncertain(message: str) -> RunnerUncertainError:
            return RunnerUncertainError(
                message, operation_id=operation_id, request_hash=fingerprint
            )

        def checked(receipt: dict[str, Any]) -> dict[str, Any]:
            if (
                not isinstance(receipt, dict)
                or receipt.get("version") != 1
                or receipt.get("operation_id") != operation_id
                or receipt.get("fence") != self.fence
                or receipt.get("request_hash") != fingerprint
                or receipt.get("state")
                not in {
                    "prepared",
                    "created",
                    "running",
                    "done",
                    "error",
                    "cancelled",
                    "uncertain",
                    "interrupted",
                    "rejected",
                }
            ):
                raise uncertain("Malformed or unbound runner receipt for operation " + operation_id)
            return receipt

        try:
            receipt = checked(
                self.request(
                    "launch",
                    operation_id=operation_id,
                    operation_kind=kind,
                    staging_ticket=ticket,
                    inputs=inputs,
                    remaining_seconds=seconds,
                )
            )
        except RunnerRejectedError:
            raise
        except (OSError, TimeoutError, ValueError) as exc:
            raise uncertain(
                "Runner launch response was lost; reconcile operation " + operation_id
            ) from exc
        deadline = time.monotonic() + seconds
        stopping = False
        stop_deadline = 0.0
        while receipt["state"] not in {
            "done",
            "error",
            "cancelled",
            "uncertain",
            "interrupted",
            "rejected",
        }:
            if cancelled() or time.monotonic() >= deadline:
                if not stopping:
                    try:
                        self.request("cancel", operation_id=operation_id)
                    except (OSError, TimeoutError, ValueError) as exc:
                        raise uncertain(
                            "Runner cancellation response was lost for operation " + operation_id
                        ) from exc
                    stopping, stop_deadline = True, time.monotonic() + 25
                if time.monotonic() >= stop_deadline:
                    raise uncertain(
                        "Runner cleanup remains uncertain for operation " + operation_id
                    )
            time.sleep(0.05)
            try:
                receipt = checked(self.request("status", operation_id=operation_id))
            except (OSError, TimeoutError, ValueError) as exc:
                raise uncertain("Runner status was lost for operation " + operation_id) from exc
        if receipt["state"] == "uncertain":
            raise uncertain("Runner operation requires reconciliation: " + operation_id)
        if receipt["state"] != "done":
            raise HostingError(receipt.get("error") or "Runner operation did not complete")
        if inventory(staging, maximum=MAX_JOB_BYTES) != receipt["outputs"]:
            raise HostingError("Runner outputs differ from their durable receipt")
        for name in receipt["outputs"]:
            if name in inputs:
                continue
            target = directory / name
            if target.exists() or target.is_symlink():
                raise HostingError("Runner output would overwrite an existing file")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / name, target)
        return receipt


def serve(runner: Runner, socket_path: Path) -> None:
    """Linux-only IPC authenticates the dedicated gateway UID with SO_PEERCRED."""
    if os.name != "posix" or not hasattr(socket, "SO_PEERCRED"):
        raise HostingError("Restricted runner requires Linux Unix peer credentials")
    if _POSIX_OS.getuid() == runner.policy.gateway_uid:
        raise HostingError("Runner and public gateway must use distinct dedicated UIDs")

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            self.request.settimeout(20)
            try:
                _, uid, _ = struct.unpack(
                    "3i", self.request.getsockopt(socket.SOL_SOCKET, _UNIX_SOCKET.SO_PEERCRED, 12)
                )
                if uid != runner.policy.gateway_uid:
                    raise HostingError("Runner peer UID is not authorized")
                data = self.rfile.readline(_WIRE_MAXIMUM + 1)
                if len(data) > _WIRE_MAXIMUM or not data.endswith(b"\n"):
                    raise HostingError("Invalid bounded runner request")
                result = runner.respond(json.loads(data))
            except Exception as exc:
                result = {"protocol_error": True, "error": str(exc)[:300]}
            self.wfile.write(json.dumps(result).encode() + b"\n")

    class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
        address_family = _UNIX_SOCKET.AF_UNIX
        daemon_threads = True

    # Existing sockets are deliberately not unlinked: first reconcile the old runner.
    with Server(cast(Any, str(socket_path)), Handler) as server:
        socket_path.chmod(0o660)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--fence", required=True)
    parser.add_argument("--gateway-uid", type=int, required=True)
    parser.add_argument("--reconcile", help="Offline operation ID; does not start a listener")
    parser.add_argument(
        "--unaccepted-request-hash",
        help="Retained exact request SHA for explicit no-record reconciliation",
    )
    args = parser.parse_args()
    policy = RunnerPolicy(args.root, args.staging_root, args.image, args.fence, args.gateway_uid)
    from forge_mind.workspace.hosting import HostLock

    with HostLock(policy.root):
        runner = Runner(policy)
        if args.reconcile:
            print(
                json.dumps(
                    runner.reconcile(
                        args.reconcile, unaccepted_request_hash=args.unaccepted_request_hash
                    )
                )
            )
        else:
            serve(runner, args.socket)


if __name__ == "__main__":
    main()
