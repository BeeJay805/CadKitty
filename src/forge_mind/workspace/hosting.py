"""Explicit single-host policy, process exclusion, and safe readiness facts."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit


class HostingError(ValueError):
    """A hosting authority or configured boundary is unavailable."""


class HostLock:
    """Exclude a serving process and offline maintenance on the same filesystem."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".host.lock"
        self.handle: BinaryIO | None = None

    def __enter__(self) -> HostLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise HostingError("Host lock must not be a symbolic link")
        handle = self.path.open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                flock_api: Any = importlib.import_module("fcntl")
                flock_api.flock(handle.fileno(), flock_api.LOCK_EX | flock_api.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise HostingError("Workspace host is already serving or under maintenance") from exc
        self.handle = handle
        return self

    def __exit__(self, *args: Any) -> None:
        if self.handle is not None:
            if sys.platform == "win32":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                flock_api: Any = importlib.import_module("fcntl")
                flock_api.flock(self.handle.fileno(), flock_api.LOCK_UN)
            self.handle.close()
            self.handle = None


def https_origin(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
    ):
        raise HostingError("A plain HTTPS origin is required")
    return value.rstrip("/")


@dataclass(frozen=True)
class HostConfig:
    root: Path
    spool: Path
    runner_socket: Path
    image: str
    release: str
    engine: str
    origins: tuple[str, ...]
    issuer: str
    owner_active_limit: int = 1
    runner_fence: str = ""
    provider_model: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HostConfig:
        env = os.environ if env is None else env

        def required(name: str) -> str:
            value = env.get(name, "").strip()
            if not value:
                raise HostingError(f"Missing required host setting: {name}")
            return value

        def absolute(name: str) -> Path:
            path = Path(required(name))
            if not path.is_absolute():
                raise HostingError(f"{name} must be absolute")
            return path.resolve()

        image, engine = required("FORGE_WORKSPACE_IMAGE"), required("FORGE_HOST_ENGINE")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
            raise HostingError("Host worker image must use its immutable sha256 identity")
        if not re.fullmatch(r"[a-f0-9]{64}", engine):
            raise HostingError("Host engine must use the frozen engine fingerprint")
        release = required("FORGE_HOST_RELEASE")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", release):
            raise HostingError("Invalid release identity")
        root, spool = absolute("FORGE_WORKSPACE_DIR"), absolute("FORGE_RUNNER_SPOOL")
        if root == spool or root in spool.parents or spool in root.parents:
            raise HostingError("Durable Store and disposable runner spool must be disjoint")
        origins = tuple(
            https_origin(v.strip()) for v in required("FORGE_WORKSPACE_ORIGINS").split(",")
        )
        issuer = https_origin(required("SUPABASE_URL")) + "/auth/v1"
        required("SUPABASE_ANON_KEY")
        limit = int(env.get("FORGE_HOST_OWNER_ACTIVE_LIMIT", "1"))
        if limit <= 0:
            raise HostingError("Owner active limit must be positive")
        fence = required("FORGE_RUNNER_FENCE")
        if not re.fullmatch(r"[a-f0-9]{32}", fence):
            raise HostingError("Runner fence must be an operator-created opaque identity")
        model = required("FORGE_HOST_PROVIDER_MODEL")
        if env.get("FORGE_CODEX_MODEL") != model:
            raise HostingError("Provider selection must match its qualified model exactly")
        if env.get("FORGE_NATIVE_STAGED_PLANNING") != "1":
            raise HostingError("Private customer host requires explicit staged native planning")
        return cls(
            root,
            spool,
            absolute("FORGE_RUNNER_SOCKET"),
            image,
            release,
            engine,
            origins,
            issuer,
            limit,
            fence,
            model,
        )


def readiness(
    config: HostConfig,
    *,
    current_engine: str,
    executor: Any,
    worker_alive: bool,
    provider_qualified: bool = False,
    recovery_qualified: bool = False,
) -> dict[str, Any]:
    """Report observed prerequisites; settings alone never qualify external services."""
    from forge_mind.workspace.customer_projects import hosted_customer_limits

    storage = False
    runner_reconciled = False
    try:
        with sqlite3.connect(f"file:{config.root / 'workspace.sqlite3'}?mode=rw", uri=True) as con:
            storage = con.execute("PRAGMA quick_check").fetchone() == ("ok",)
            if con.execute("SELECT 1 FROM sqlite_master WHERE name='host_runner_holds'").fetchone():
                runner_reconciled = (
                    con.execute("SELECT 1 FROM host_runner_holds LIMIT 1").fetchone() is None
                )
            else:
                runner_reconciled = True
    except sqlite3.Error:
        pass
    checks = {
        "engine": current_engine == config.engine,
        "worker_image": executor.typed_container_image == config.image,
        "isolated_worker": executor.typed_container_verified
        and executor.customer_container_verified,
        "restricted_runner": bool(getattr(executor, "runner_verified", False)),
        "storage": storage,
        "worker_alive": worker_alive,
        "provider": provider_qualified,
        "recovery": recovery_qualified,
        "restore_reconciled": not (config.root / ".restore-required.json").exists(),
        "restore_complete": not (config.root / ".restore-incomplete.json").exists(),
        "runner_reconciled": runner_reconciled,
        "staged_native_planning": os.environ.get("FORGE_NATIVE_STAGED_PLANNING") == "1",
        "provider_model_bound": os.environ.get("FORGE_CODEX_MODEL") == config.provider_model,
        "customer_limits": hosted_customer_limits() is not None,
    }
    return {"ready": all(checks.values()), "release": config.release, "checks": checks}


def provider_credentials() -> bool:
    """Check subscription CLI authentication without generating or exposing secrets."""
    command = shutil.which("codex")
    if command is None:
        return False
    try:
        result = subprocess.run(
            [command, "login", "status"], capture_output=True, timeout=10, check=False
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def qualification_receipt(config: HostConfig) -> dict[str, bool]:
    """Read operator-owned actual-run receipts, distinct from auth/configuration.

    A receipt is an audit boundary, not an independent proof of its author's claim.
    The release owner creates it only after real provider and restore exercises.
    """
    path = config.root / ".host-qualification.json"
    result = {"provider": False, "recovery": False}
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
            return result
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("version") != 1 or any(
            receipt.get(key) != value
            for key, value in {
                "release": config.release,
                "engine": config.engine,
                "image": config.image,
            }.items()
        ):
            return result
        provider = receipt.get("provider", {})
        result["provider"] = provider.get("model") == config.provider_model and all(
            isinstance(provider.get(key), str)
            and re.fullmatch(r"[a-f0-9]{64}", provider[key]) is not None
            for key in ("request_sha256", "response_sha256")
        )
        restore = receipt.get("restore", {})
        result["recovery"] = restore.get("complete") is True and all(
            isinstance(restore.get(key), str)
            and re.fullmatch(r"[a-f0-9]{64}", restore[key]) is not None
            for key in ("snapshot_manifest_sha256", "restored_inventory_sha256")
        )
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return result


def write_json(path: Path, value: Any) -> None:
    """Create a durable receipt before any external side effect."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
