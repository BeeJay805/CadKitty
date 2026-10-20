"""Frozen customer outcome trials. No model verdict is an acceptance oracle.

The local adapter uses existing HTTP routes and the native worker. Importing this
module performs no CAD, model call, service launch, or network request.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from zipfile import ZipFile

MODEL = "gpt-6-astra"
SCHEMA = 1
HARNESS_VERSION = 3
ORACLE_PROTOCOL = "analytic_step_v002"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "benchmarks/customer_outcomes_v1"
SCOPE = (
    "Prospective development corpus, not sampled customer demand. Independent nominal "
    "checks cover only their listed dimensions and finite samples. Digital delivery "
    "and nominal checks do not establish complete request satisfaction, physical fit, "
    "strength, print success, labor savings or hosted readiness."
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: str | bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"Nonfinite number: {value}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)

    def finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Nonfinite number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        if isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def save_new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("Truncated journal; preserve and repair it explicitly before resume")
    rows = [strict_json(line) for line in data.splitlines()]
    previous = "0" * 64
    for index, row in enumerate(rows):
        record = {key: value for key, value in row.items() if key != "event_hash"}
        if row.get("sequence") != index or row.get("previous_hash") != previous:
            raise ValueError("Journal sequence or chain mismatch")
        if sha(encoded(record)) != row.get("event_hash"):
            raise ValueError("Journal content hash mismatch")
        previous = row["event_hash"]
    return rows


class Journal:
    def __init__(self, path: Path):
        self.path = path
        self.rows = read_events(path)
        self.lock = threading.Lock()

    def append(self, kind: str, **values: Any) -> dict[str, Any]:
        with self.lock:
            return self._append(kind, **values)

    def _append(self, kind: str, **values: Any) -> dict[str, Any]:
        row = {
            "sequence": len(self.rows),
            "previous_hash": self.rows[-1]["event_hash"] if self.rows else "0" * 64,
            "kind": kind,
            "at": time.time(),
            **values,
        }
        row["event_hash"] = sha(encoded(row))
        with self.path.open("ab") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        return row


def load_corpus(directory: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = strict_json((directory / "MANIFEST.json").read_bytes())
    if not {"cases.json", "ATTRIBUTION.json"} <= set(manifest["files"]):
        raise ValueError("Case definitions and attribution must be frozen")
    for name, expected in manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe corpus member")
        if sha((directory / name).read_bytes()) != expected:
            raise ValueError(f"Frozen corpus mismatch: {name}")
    document = strict_json((directory / "cases.json").read_bytes())
    cases = document["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_]+", key) for key in ids):
        raise ValueError("Case identities must be unique safe names")
    for case in cases:
        if case["split"] not in {"development", "heldout"}:
            raise ValueError("Unknown corpus split")
        if case["oracle"]["kind"] not in {"analytic_solid_v1", "unassessed_v1", "clarification_v1"}:
            raise ValueError("Unknown oracle; no implicit fallback")
        if case["oracle"]["kind"] == "analytic_solid_v1":
            analytic_spec(case["oracle"])
        if case.get("reference") and case["reference"]["file"] not in manifest["files"]:
            raise ValueError("Reference must be frozen")
    return cases, sha(encoded(manifest))


def source_inventory(root: Path = ROOT) -> dict[str, str]:
    paths = list((root / "src/forge_mind").rglob("*.py"))
    paths += [
        root / name for name in ("pyproject.toml", "uv.lock", "Dockerfile", "Dockerfile.workspace")
    ]
    engine = {
        "engine/" + path.relative_to(root).as_posix(): sha(path.read_bytes())
        for path in sorted(paths)
        if path.is_file()
    }
    for name in ("scripts/customer_outcome_bench.py", "scripts/customer_outcome_report.py"):
        engine["harness/" + name] = sha((ROOT / name).read_bytes())
    return engine


def source_git_metadata(root: Path) -> dict[str, Any]:
    """Git HEAD is optional archive metadata, never file identity or clean-tree proof."""
    redirects = {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
    }
    if redirects.intersection(os.environ):
        raise ValueError("Git source identity redirection is unsupported")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Source root is not a directory")
    if not any(os.path.lexists(parent / ".git") for parent in (root, *root.parents)):
        return {
            "git_head": None,
            "git_metadata": {
                "status": "unavailable_source_archive",
                "worktree_cleanliness": "not_available",
            },
        }
    # A marker means metadata is expected: corrupt/unborn repositories, malformed
    # worktree links and missing executables must remain errors, not archive mode.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        env=environment,
        stderr=subprocess.PIPE,
        timeout=10,
    ).strip()
    if not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", head):
        raise ValueError("Git did not return a commit identity")
    return {
        "git_head": head,
        "git_metadata": {
            "status": "available",
            "scope": "harness_HEAD_only",
            "worktree_cleanliness": "not_checked",
        },
    }


def validate_archive_inventory(adapter: Adapter, captured: dict[str, str]) -> None:
    """An actual archive trial needs exact engine/config/harness bytes, not an empty fallback."""
    if adapter.name != "customer_local":
        return
    required = {
        "engine/src/forge_mind/__init__.py",
        "engine/pyproject.toml",
        "engine/uv.lock",
        "engine/Dockerfile",
        "engine/Dockerfile.workspace",
        "harness/scripts/customer_outcome_bench.py",
        "harness/scripts/customer_outcome_report.py",
    }
    if not required.issubset(captured) or captured != source_inventory(
        Path(getattr(adapter, "source_root", ROOT))
    ):
        raise ValueError("Source archive requires a complete exact file-hash inventory")


def runtime_identity() -> dict[str, Any]:
    packages = {}
    for name in (
        "build123d",
        "cadquery",
        "numpy",
        "pydantic",
        "fastapi",
        "httpx",
        "trimesh",
        "manifold3d",
        "scikit-fem",
        "pytetwild",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def verified_archive(
    data: bytes, receipt: dict[str, Any], *, kind: str = "project"
) -> dict[str, bytes]:
    if len(data) != receipt["bytes"] or sha(data) != receipt["sha256"]:
        raise ValueError("Download differs from receipt")
    with ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or sum(item.file_size for item in infos) > 64 * 1024**2:
            raise ValueError("Duplicate or oversized archive")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unsafe archive member")
        members = {name: archive.read(name) for name in names}
    manifest = strict_json(members["manifest.json"])
    if kind == "project":
        listed, length_key = manifest["files"], "size_bytes"
    elif kind == "print":
        if sha(members["manifest.json"]) != receipt["manifest_sha256"]:
            raise ValueError("Print manifest differs from receipt")
        listed, length_key = manifest["members"], "bytes"
    else:
        raise ValueError("Unknown archive kind")
    for name, record in listed.items():
        raw = members.get(name)
        if raw is None or sha(raw) != record["sha256"] or len(raw) != record[length_key]:
            raise ValueError(f"Archive member mismatch: {name}")
    if set(members) - {"manifest.json"} != set(listed):
        raise ValueError("Unlisted archive member")
    return members


def current_capture_receipt(
    session: dict[str, Any], revision: str, binding: dict[str, Any]
) -> dict[str, Any] | None:
    """Only the normal customer job can establish captured delivery, never this harness."""
    capture = session.get("automatic_delivery") or {}
    if capture.get("status") != "delivered" or capture.get("readback_verified") is not True:
        return None
    if (
        not session.get("job_id")
        or capture.get("job_id") != session["job_id"]
        or capture.get("revision_id") != revision
        or session.get("selected_revision") != revision
        or any(capture.get(key) != value for key, value in binding.items())
    ):
        return None
    receipt = capture.get("receipt") or {}
    if receipt.get("project_id") != session["id"] or receipt.get("revision_id") != revision:
        return None
    return dict(receipt)


def requested_project_files(
    members: dict[str, bytes], receipt: dict[str, Any], *, printable: bool
) -> bool:
    """CAD-only delivery does not require an unrequested mesh or start a new export."""
    if "document.json" not in members or "assembly/part.step" not in members:
        return False
    if not printable:
        return True
    made = receipt.get("made_parts") or []
    return (
        bool(made)
        and receipt.get("include_stl") is True
        and all(
            any(name.endswith(".step") and name in members for name in part.get("files", []))
            and any(name.endswith(".stl") and name in members for name in part.get("files", []))
            for part in made
        )
    )


def analytic_spec(spec: dict[str, Any]) -> tuple[list[float], float, Callable[..., bool]]:
    """Analytic nominal formulas, independent of generated source and validator tags."""
    shape = spec["shape"]
    if shape in {"block", "plate_holes", "cup", "angle", "channel"}:
        x, y, z = spec["size_mm"]
        volume = x * y * z

        def contains(a: float, b: float, c: float) -> bool:
            return 0 < a < x and 0 < b < y and 0 < c < z

        if shape == "plate_holes":
            radius = spec["hole_diameter_mm"] / 2
            centers = spec["hole_centers_mm"]
            volume -= len(centers) * math.pi * radius**2 * z

            def contains(a: float, b: float, c: float) -> bool:
                return (
                    0 < a < x
                    and 0 < b < y
                    and 0 < c < z
                    and all((a - hx) ** 2 + (b - hy) ** 2 > radius**2 for hx, hy in centers)
                )

        elif shape in {"cup", "angle", "channel"}:
            wall = spec["wall_mm"]
            if shape == "cup":
                volume -= (x - 2 * wall) * (y - 2 * wall) * (z - wall)

                def contains(a: float, b: float, c: float) -> bool:
                    return c < wall or a < wall or a > x - wall or b < wall or b > y - wall

            elif shape == "angle":
                volume = x * (y * wall + (z - wall) * wall)

                def contains(a: float, b: float, c: float) -> bool:
                    return b < wall or c < wall

            else:
                volume = x * (y * wall + 2 * (z - wall) * wall)

                def contains(a: float, b: float, c: float) -> bool:
                    return c < wall or b < wall or b > y - wall

        return [x, y, z], volume, contains
    if shape in {"tube", "disk"}:
        diameter, height = spec["diameter_mm"], spec["height_mm"]
        outer = diameter / 2
        inner = spec.get("bore_mm", 0) / 2

        def contains(a: float, b: float, c: float) -> bool:
            distance = (a - outer) ** 2 + (b - outer) ** 2
            return (
                (distance > inner**2 if inner else distance >= 0)
                and distance < outer**2
                and 0 < c < height
            )

        return [diameter, diameter, height], math.pi * (outer**2 - inner**2) * height, contains
    raise ValueError(f"Unsupported analytic shape: {shape}")


def measure_step(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Root-run native adapter only. Reimport delivered STEP, never execute its source."""
    from build123d import import_step

    shape = import_step(str(path))
    size, volume, contains = analytic_spec(spec)
    bounds = shape.bounding_box()
    offset = [bounds.min.X, bounds.min.Y, bounds.min.Z]
    checks = {
        "valid_shape": bool(shape.is_valid),
        "one_solid": len(shape.solids()) == 1,
        "extents": all(
            abs(actual - expected) <= 0.02
            for actual, expected in zip(
                [bounds.size.X, bounds.size.Y, bounds.size.Z], size, strict=True
            )
        ),
        "volume": abs(shape.volume - volume) <= max(0.1, volume * 0.0001),
    }
    # Offset fractions avoid most integer dimension boundaries. This is finite
    # sampling, not exact solid equivalence. Specific bore probes supplement it.
    probes = [
        tuple(size[axis] * (i + 0.37) / 9 for axis, i in enumerate((a, b, c)))
        for a in range(9)
        for b in range(9)
        for c in range(9)
    ]
    if spec["shape"] == "plate_holes":
        probes.extend(
            (x, y, size[2] * f) for x, y in spec["hole_centers_mm"] for f in (0.1, 0.5, 0.9)
        )
    mismatches = []
    for point in probes:
        expected = contains(*point)
        measured = shape.is_inside(tuple(a + b for a, b in zip(point, offset, strict=True)))
        if measured != expected:
            mismatches.append(list(point))
    checks["occupancy_samples"] = not mismatches
    if spec["shape"] in {"plate_holes", "tube"}:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder

        cylinders = []
        for face in shape.faces():
            surface = BRepAdaptor_Surface(face.wrapped, True)
            if surface.GetType() != GeomAbs_Cylinder:
                continue
            cylinder = surface.Cylinder()
            axis, point = cylinder.Axis().Direction(), cylinder.Location()
            span = face.bounding_box()
            if abs(abs(axis.Z()) - 1) < 1e-6:
                cylinders.append(
                    (
                        cylinder.Radius(),
                        point.X() - offset[0],
                        point.Y() - offset[1],
                        span.min.Z - offset[2],
                        span.max.Z - offset[2],
                    )
                )
        radius = (
            spec["hole_diameter_mm"] / 2 if spec["shape"] == "plate_holes" else spec["bore_mm"] / 2
        )
        centers = (
            spec["hole_centers_mm"]
            if spec["shape"] == "plate_holes"
            else [[size[0] / 2, size[1] / 2]]
        )
        matching = [
            row
            for row in cylinders
            if abs(row[0] - radius) < 0.01 and abs(row[3]) < 0.01 and abs(row[4] - size[2]) < 0.01
        ]
        checks["full_span_bore_centers_and_diameters"] = len(matching) == len(centers) and all(
            any(abs(row[1] - x) < 0.01 and abs(row[2] - y) < 0.01 for row in matching)
            for x, y in centers
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "step_sha256": sha(path.read_bytes()),
        "measured_volume_mm3": shape.volume,
        "expected_volume_mm3": volume,
        "sample_count": len(probes),
        "mismatch_count": len(mismatches),
        "first_mismatches_mm": mismatches[:10],
        "scope": "Extents, volume, solid count and finite occupancy in requested axis frame; translation allowed. No exact shape-equivalence or physical claim.",
    }


def assess(case: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """No absent, malformed, synthetic or model-owned evidence can earn a real pass."""
    kind = case["oracle"]["kind"]
    if kind == "clarification_v1":
        nominal = (
            "pass"
            if observation.get("state") == "needs_input"
            and observation.get("questions", 0) > 0
            and observation.get("revisions", 0) == 0
            else "fail"
        )
    elif kind == "unassessed_v1":
        nominal = "unassessed"
    elif kind == "analytic_solid_v1":
        nominal = observation.get("nominal", {}).get("status", "unassessed")
        if nominal not in {"pass", "fail", "unassessed"}:
            raise ValueError("Invalid oracle status")
    else:
        raise ValueError("No oracle registered")
    delivered = all(
        observation.get(key) is True
        for key in ("files_verified", "guide_current", "literal_guide", "revision_current")
    )
    clean = observation.get("unresolved") == 0 and observation.get("required_failed") == 0
    real = observation.get("data_origin") == "actual_customer_pipeline"
    identity = (
        observation.get("actual_model") == MODEL and observation.get("identity_verified") is True
    )
    return {
        "nominal_status": nominal,
        "digital_delivery": delivered,
        "unresolved_clear": clean,
        "actual_identity_verified": identity,
        "nominal_digital_pass": real
        and identity
        and delivered
        and clean
        and nominal == "pass"
        and kind != "clarification_v1",
        "clarification_pass": real
        and identity
        and nominal == "pass"
        and kind == "clarification_v1",
        "full_task_success": None,
        "physical_readiness": "unverified",
        "scope": SCOPE,
    }


class Adapter(Protocol):
    name: str

    def run(
        self, case: dict[str, Any], directory: Path, emit: Callable[..., Any]
    ) -> dict[str, Any]: ...


class FixtureAdapter:
    """Pure harness adapter. Its observations can never count as actual trials."""

    name = "synthetic_fixture"

    def __init__(self, observations: dict[str, dict[str, Any]]):
        self.observations = observations

    def run(
        self, case: dict[str, Any], directory: Path, emit: Callable[..., Any]
    ) -> dict[str, Any]:
        return {**self.observations[case["id"]], "data_origin": "synthetic_fixture"}


class CustomerLocalAdapter:
    name = "customer_local"

    def __init__(
        self,
        corpus: Path,
        *,
        executor: str,
        seconds: int,
        actions: int,
        condition: str,
        source_root: Path = ROOT,
        expected_engine: str | None = None,
        generation_policy: str | None = None,
    ):
        self.corpus, self.executor = corpus, executor
        self.seconds, self.actions, self.condition = seconds, actions, condition
        self.source_root = source_root.resolve()
        self.expected_engine = expected_engine
        self.generation_policy = generation_policy

    def preflight(self) -> None:
        flag = os.environ.get("FORGE_NATIVE_STAGED_PLANNING", "0")
        if self.generation_policy not in {"staged", "legacy"}:
            raise ValueError("Actual trials require an explicit staged or legacy generation policy")
        expected = "1" if self.generation_policy == "staged" else "0"
        if flag != expected:
            raise ValueError(
                "Generation policy differs from FORGE_NATIVE_STAGED_PLANNING; no worker or model was started"
            )

    def run(
        self, case: dict[str, Any], directory: Path, emit: Callable[..., Any]
    ) -> dict[str, Any]:
        self.preflight()
        if os.environ.get("FORGE_FORBID_API_SPEND") != "1":
            raise ValueError("FORGE_FORBID_API_SPEND=1 is mandatory")
        if os.environ.get("FORGE_CODEX_MODEL", MODEL) != MODEL:
            raise ValueError("Actual customer trials require explicit gpt-6-astra")
        os.environ["FORGE_CODEX_MODEL"] = MODEL
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from forge_mind.llm.codex_client import CodexClient
        from forge_mind.workspace.agent import WorkspaceService
        from forge_mind.workspace.api import router
        from forge_mind.workspace.executor import Executor
        from forge_mind.workspace.product_api import product_router
        from forge_mind.workspace.project_access import project_access_router
        from forge_mind.workspace.store import Store

        started = time.monotonic()
        calls: list[dict[str, Any]] = []

        class RecordingClient(CodexClient):
            def _generate(self, *values: Any, **kwargs: Any) -> Any:
                call_started = time.monotonic()
                row: dict[str, Any] = {"index": len(calls), "configured_model": self.model}
                for key, value in zip(("system", "user"), values, strict=False):
                    row[key] = value
                row.update({k: v for k, v in kwargs.items() if k in {"system", "user"}})
                emit("model_call_started", index=row["index"], configured_model=self.model)
                try:
                    result, usage = super()._generate(*values, **kwargs)
                    row.update(response=result.model_dump(), usage=vars(usage))
                    return result, usage
                except BaseException as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    row.update(
                        seconds=time.monotonic() - call_started,
                        diagnostics=self.last_call_diagnostics,
                        input_images=self.last_input_images,
                    )
                    calls.append(row)
                    save_new(directory / f"call-{row['index']:04d}.json", row)
                    emit(
                        "model_call_finished",
                        index=row["index"],
                        seconds=row["seconds"],
                        failed="error" in row,
                    )

        import forge_mind

        if Path(forge_mind.__file__).resolve().parents[2] != self.source_root:
            raise ValueError(
                "Imported engine differs from runner source root; set PYTHONPATH explicitly"
            )
        from forge_mind.workspace.measure import engine_fingerprint

        fingerprint = engine_fingerprint()
        if self.expected_engine and fingerprint != self.expected_engine:
            raise ValueError("Imported engine fingerprint differs from explicit trial target")
        executor = Executor(self.executor)
        if self.executor == "docker":
            emit("worker_qualification_started")
            typed = executor.qualify_typed_container()
            customer = executor.qualify_customer_container()
            save_new(
                directory / "worker-qualification.json", {"typed": typed, "customer": customer}
            )
            emit("worker_qualified", image=executor.typed_container_image)
        service = WorkspaceService(Store(directory / "store"), executor, RecordingClient)
        app = FastAPI()
        owner = lambda: "local-owner"  # noqa: E731
        app.include_router(router(service.store, service, owner))
        app.include_router(project_access_router(service.store, service, owner))
        app.include_router(product_router(service.store, service, owner))
        client = TestClient(app)
        snapshot_index = 0

        def request(method: str, path: str, body: Any = None) -> Any:
            nonlocal snapshot_index
            response = client.request(method, path, **({"json": body} if body is not None else {}))
            snapshot_index += 1
            # Every HTTP response is retained, including refusal and failed transport.
            raw = response.content
            with (directory / f"http-{snapshot_index:05d}.bin").open("xb") as stream:
                stream.write(raw)
            emit(
                "http",
                method=method,
                path=path,
                status=response.status_code,
                file=f"http-{snapshot_index:05d}.bin",
                sha256=sha(raw),
            )
            response.raise_for_status()
            return (
                response.json()
                if response.headers.get("content-type", "").startswith("application/json")
                else raw
            )

        first_cad = None
        questions_seen: set[str] = set()
        sid = None
        result: dict[str, Any] = {
            "data_origin": "actual_customer_pipeline",
            "actual_model": MODEL,
            "condition": self.condition,
            "executor": self.executor,
            "owner_wait_seconds": 0.0,
            "owner_wait_scope": "Automated owner script; no human labor measured",
            "identity_verified": False,
            "files_verified": False,
            "guide_current": False,
            "literal_guide": False,
            "revision_current": False,
            "unresolved": None,
            "required_failed": None,
            "nominal": {"status": "unassessed"},
            "worker": {
                "mode": self.executor,
                "pid": os.getpid(),
                "image": executor.typed_container_image if self.executor == "docker" else None,
                "typed_qualified": executor.typed_container_verified,
                "customer_qualified": executor.customer_container_verified,
            },
            "runtime": runtime_identity(),
            "engine_fingerprint": fingerprint,
        }
        try:
            creation = {
                "request_id": uuid.uuid4().hex,
                "prompt": case["prompt"],
                "approval_required": case.get("approval_required", False),
                "printable": case.get("printable", False),
                "max_seconds": self.seconds,
                "max_actions": self.actions,
            }
            if case.get("reference"):
                ref = case["reference"]
                creation["reference"] = {
                    "filename": Path(ref["file"]).name,
                    "media_type": ref.get("media_type", "image/png"),
                    "content_base64": base64.b64encode(
                        (self.corpus / ref["file"]).read_bytes()
                    ).decode(),
                    "known_dimension": ref.get("known_dimension"),
                }
            save_new(directory / "request.json", creation)
            created = request("POST", "/workspace/customer-projects", creation)
            sid = created["id"]
            prefix = f"/workspace/sessions/{sid}"
            if case.get("recovery") == "lost_creation_response":
                recovered = request("GET", f"/workspace/customer-requests/{creation['request_id']}")
                if recovered["id"] != sid:
                    raise ValueError("Recovery changed project")
                replay = request("POST", "/workspace/customer-projects", creation)
                if replay["id"] != sid:
                    raise ValueError("Creation replay changed project")
                emit("recovery_verified", project_id=sid)
            edited = False
            approved = False
            observed_state = None
            while True:
                session = request("GET", prefix)
                result.update(
                    state=session["state"],
                    project_id=sid,
                    revisions=len(session.get("revisions", [])),
                )
                revision = session.get("selected_revision")
                if (
                    any(
                        row.get("measurement", {}).get("geometry_valid") is True
                        for row in session.get("revisions", [])
                    )
                    and first_cad is None
                ):
                    first_cad = time.monotonic() - started
                    emit("first_cad", seconds=first_cad)
                if observed_state != session["state"]:
                    emit("state", state=session["state"], selected_revision=revision)
                    observed_state = session["state"]
                attribution = session.get("intent_attribution") or {}
                issues = (attribution.get("attribution") or {}).get("issues", [])
                issues += attribution.get("issues", [])
                questions_seen.update(issue["id"] for issue in issues)
                if (
                    session["state"] == "needs_approval"
                    and case.get("approval_required")
                    and not approved
                ):
                    proposal = request("GET", prefix + "/customer-proposal")
                    request(
                        "POST",
                        prefix + "/customer-proposal/approve",
                        {"proposal_hash": proposal["proposal_hash"], "basis": proposal["basis"]},
                    )
                    approved = True
                    continue
                active = session["state"] in {
                    "queued",
                    "running",
                    "extracting",
                    "authoring",
                    "building",
                    "measuring",
                    "checking",
                }
                if active:
                    if (
                        time.monotonic() - started
                        > self.seconds * (2 if case.get("edit") else 1) + 120
                    ):
                        request("POST", prefix + "/cancel")
                        raise TimeoutError("Bounded observation deadline; cancellation requested")
                    time.sleep(0.5)
                    continue
                if case.get("edit") and revision and not edited:
                    result["before_edit_revision"] = revision
                    request(
                        "POST",
                        prefix + "/edit",
                        {
                            "instruction": case["edit"],
                            "revision_id": revision,
                            "max_seconds": self.seconds,
                            "max_actions": self.actions,
                        },
                    )
                    edited = True
                    continue
                break
            save_new(directory / "session.json", session)
            result.update(
                selected_revision=revision,
                questions=len(questions_seen),
                edit_exercised=edited,
                approval_exercised=approved,
            )
            if revision:
                overview = request("GET", prefix + "/overview")
                save_new(directory / "customer-overview.json", overview)
                result["customer_outcome"] = {
                    key: overview.get(key)
                    for key in (
                        "state",
                        "job_outcome_state",
                        "current_completion_status",
                        "selected_revision",
                    )
                }
                guidance = request("GET", prefix + "/guidance")
                save_new(directory / "guidance.json", guidance)
                guide = (guidance.get("record") or {}).get("guidance") or {}
                result["guide_current"] = (
                    guidance.get("status") == "current"
                    and guidance.get("basis", {}).get("revision_id") == revision
                    and guidance.get("instruction_binding", {}).get("status") == "bound"
                )
                result["literal_guide"] = all(
                    isinstance(guide.get(field), str) and guide[field].strip()
                    for field in ("print", "installation", "use_and_removal", "unresolved")
                )
                proof = request("GET", prefix + "/contract-verification")
                save_new(directory / "contract-verification.json", proof)
                findings = proof.get("results", [])
                if not findings:
                    findings = guidance.get("facts", {}).get("requirement_findings", [])
                    result["requirement_status_scope"] = (
                        "Earlier build-time findings; current final proof unavailable"
                    )
                else:
                    result["requirement_status_scope"] = "Current final frozen contract results"
                result["requirement_statuses"] = findings
                result["unresolved"] = (
                    sum(row.get("status") != "pass" for row in findings)
                    if proof.get("results")
                    else None
                )
                result["required_failed"] = sum(row.get("status") == "fail" for row in findings)
                result["requirement_count"] = len(findings)
                from forge_mind.workspace.native_delivery import current_delivery_binding

                binding = current_delivery_binding(service, sid, "local-owner")
                receipt = current_capture_receipt(session, revision, binding)
                result["automatic_delivery_status"] = (session.get("automatic_delivery") or {}).get(
                    "status", "unavailable"
                )
                if receipt is None:
                    result["delivery_scope"] = (
                        "Normal customer workflow has no exact current automatic capture; harness did not manufacture a handoff"
                    )
                    return result
                archive = request(
                    "GET",
                    prefix + f"/project-exports/{receipt['package_id']}/{receipt['sha256']}.zip",
                )
                members = verified_archive(archive, receipt)
                with (directory / "project.zip").open("xb") as stream:
                    stream.write(archive)
                result["file_receipt"] = receipt
                result["files_verified"] = requested_project_files(
                    members, receipt, printable=case.get("printable", False)
                )
                result["print_requested"] = case.get("printable", False)
                result["print_files_verified"] = None
                if case.get("printable"):
                    setup = guidance.get("facts", {}).get("print_setup", {})
                    print_receipt = setup.get("package")
                    result["print_files_verified"] = False
                    if setup.get("status") == "current" and print_receipt:
                        print_data = request(
                            "GET",
                            prefix
                            + f"/print-exports/{print_receipt['package_id']}/{print_receipt['sha256']}",
                        )
                        print_members = verified_archive(print_data, print_receipt, kind="print")
                        with (directory / "print.zip").open("xb") as stream:
                            stream.write(print_data)
                        result["print_files_verified"] = print_receipt.get(
                            "revision_id"
                        ) == revision and any(name.endswith(".stl") for name in print_members)
                        result["print_receipt"] = print_receipt
                    result["files_verified"] &= result["print_files_verified"] is True
                final = request("GET", prefix)
                result["revision_current"] = (
                    final.get("selected_revision") == revision
                    and receipt["revision_id"] == revision
                    and final.get("job_id") == session.get("job_id")
                    and current_delivery_binding(service, sid, "local-owner") == binding
                    and current_capture_receipt(final, revision, binding) == receipt
                )
                if case.get("edit"):
                    result["revision_current"] &= edited and revision != result.get(
                        "before_edit_revision"
                    )
                if case["oracle"]["kind"] == "analytic_solid_v1":
                    steps = [
                        name
                        for name in members
                        if name.startswith("make/") and name.endswith(".step")
                    ]
                    if len(steps) != 1:
                        result["nominal"] = {
                            "status": "fail",
                            "reason": "Single-part oracle requires exactly one make STEP",
                        }
                    else:
                        step = directory / "nominal-input.step"
                        with step.open("xb") as stream:
                            stream.write(members[steps[0]])
                        result["nominal"] = measure_step(step, case["oracle"])
            result["identity_verified"] = (
                bool(calls)
                and all(
                    call.get("configured_model") == MODEL
                    and (call.get("usage") or {}).get("model") == MODEL
                    and (call.get("diagnostics") or {}).get("requested_model") == MODEL
                    and (call.get("diagnostics") or {}).get("process_started") is True
                    and (call.get("diagnostics") or {}).get("outcome") == "success"
                    for call in calls
                    if "error" not in call
                )
                and any("usage" in call for call in calls)
            )
            result["model_identity_scope"] = (
                "Configured CLI invocation and returned usage; not independent provider attestation"
            )
            return result
        finally:
            result["identity_verified"] = (
                bool(calls)
                and any("usage" in call for call in calls)
                and all(
                    call.get("configured_model") == MODEL
                    and (call.get("usage") or {}).get("model") == MODEL
                    and (call.get("diagnostics") or {}).get("requested_model") == MODEL
                    and (call.get("diagnostics") or {}).get("process_started") is True
                    and (call.get("diagnostics") or {}).get("outcome") == "success"
                    for call in calls
                    if "error" not in call
                )
            )
            if sid is not None and result.get("state") in {"running", "queued"}:
                # Cancel only this trial's owned Store; no unrelated worker/process control.
                service.store.cancel(sid, "local-owner")
            service.stop.set()
            if service.worker:
                service.worker.join(timeout=35)
                if service.worker.is_alive():
                    emit("worker_not_quiescent", project_id=sid)
            result["worker_quiescent"] = not service.worker or not service.worker.is_alive()
            result.update(
                first_cad_seconds=first_cad,
                total_seconds=time.monotonic() - started,
                model_calls=len(calls),
                failed_calls=sum("error" in call for call in calls),
                build_retries=max(0, result.get("revisions", 0) - 1),
                retry_scope="Additional saved revisions, not all native attempts; failed model calls reported separately",
                first_cad_scope="First observed revision with valid geometry; polling can add up to 0.5 seconds plus HTTP read time",
            )
            save_new(directory / "adapter-observation.json", result)
            client.close()
            if not result["worker_quiescent"]:
                raise RuntimeError(
                    "Trial worker remains active; preserve runner lock and inspect owned worker"
                )


def run_benchmark(
    output: Path,
    corpus: Path,
    case_ids: list[str],
    adapter: Adapter,
    *,
    split: str = "development",
    max_cases: int = 1,
    condition: str = "cold",
    inventory: Callable[[], dict[str, str]] = source_inventory,
) -> list[dict[str, Any]]:
    cases, corpus_hash = load_corpus(corpus)
    known = {case["id"]: case for case in cases}
    if not case_ids or len(case_ids) > max_cases or not 1 <= max_cases <= 30:
        raise ValueError("Explicit bounded case selection is required (1..30)")
    if len(case_ids) != len(set(case_ids)) or any(key not in known for key in case_ids):
        raise ValueError("Unknown or duplicate case")
    if any(known[key]["split"] != split for key in case_ids):
        raise ValueError("Split mismatch; heldout must be selected explicitly")
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive owner lock prevents journal writers and native workers colliding.
    lock = output / ".runner.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    preserve_lock = False
    try:
        journal = Journal(output / "events.jsonl")
        identity: dict[str, Any] = {
            "schema_version": SCHEMA,
            "corpus_sha256": corpus_hash,
            "source_files": inventory(),
            "case_ids": case_ids,
            "split": split,
            "adapter": adapter.name,
            "condition": condition,
            "actual_model": MODEL,
            "runtime": runtime_identity(),
            "source_root": str(getattr(adapter, "source_root", ROOT)),
            "trial_policy": {
                "seconds": getattr(adapter, "seconds", None),
                "actions": getattr(adapter, "actions", None),
                "executor": getattr(adapter, "executor", None),
            },
            "expected_engine": getattr(adapter, "expected_engine", None),
            "generation_policy": getattr(adapter, "generation_policy", None),
            "delivery_protocol": "customer_automatic_capture_v003",
            "harness_version": HARNESS_VERSION,
            "oracle_protocol": ORACLE_PROTOCOL,
            "staged_native_planning_flag": os.environ.get("FORGE_NATIVE_STAGED_PLANNING", "0"),
            "forge_environment_sha256": sha(
                encoded(
                    {key: value for key, value in os.environ.items() if key.startswith("FORGE_")}
                )
            ),
            "scope": SCOPE,
        }
        if journal.rows:
            if journal.rows[0].get("identity") != identity:
                raise ValueError("Source, corpus, adapter or selection drift; use a new output")
        else:
            git = source_git_metadata(ROOT)
            if git["git_head"] is None:
                validate_archive_inventory(adapter, identity["source_files"])
            journal.append(
                "run",
                identity=identity,
                python=sys.version,
                platform=platform.platform(),
                executable=sys.executable,
                **git,
            )
        finished = {row["case_id"] for row in journal.rows if row["kind"] == "attempt_finished"}
        for key in case_ids:
            if key in finished:
                continue
            if inventory() != identity["source_files"] or load_corpus(corpus)[1] != corpus_hash:
                raise ValueError("Source or corpus changed before trial")
            incomplete = [
                row
                for row in journal.rows
                if row["kind"] == "attempt_started" and row["case_id"] == key
            ]
            for prior in incomplete:
                if not any(
                    row["kind"] == "attempt_abandoned" and row["attempt_id"] == prior["attempt_id"]
                    for row in journal.rows
                ):
                    journal.append(
                        "attempt_abandoned",
                        case_id=key,
                        attempt_id=prior["attempt_id"],
                        reason="Interrupted prior process; retained immutable. New request, never replay uncertain mutations.",
                    )
            attempt_id = uuid.uuid4().hex
            directory = output / attempt_id
            directory.mkdir()
            journal.append("attempt_started", case_id=key, attempt_id=attempt_id)

            def emit(
                kind: str, _case_id: str = key, _attempt_id: str = attempt_id, **values: Any
            ) -> Any:
                return journal.append(kind, case_id=_case_id, attempt_id=_attempt_id, **values)

            start = time.monotonic()
            observation: dict[str, Any] = {
                "data_origin": "actual_customer_pipeline"
                if adapter.name == "customer_local"
                else "synthetic_fixture"
            }
            error = None
            try:
                observation = adapter.run(known[key], directory, emit)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if (directory / "adapter-observation.json").exists():
                    observation = strict_json((directory / "adapter-observation.json").read_bytes())
            try:
                unchanged = (
                    inventory() == identity["source_files"]
                    and load_corpus(corpus)[1] == corpus_hash
                )
            except (ValueError, OSError):
                unchanged = False
            try:
                verdict = assess(known[key], observation)
            except (ValueError, TypeError, AttributeError) as exc:
                error = f"Invalid observation: {type(exc).__name__}: {exc}"
                verdict = assess(known[key], {})
            if error or not unchanged:
                verdict["nominal_digital_pass"] = False
                verdict["clarification_pass"] = False
            result = {
                "case_id": key,
                "attempt_id": attempt_id,
                "split": split,
                "condition": condition,
                "observation": observation,
                "assessment": verdict,
                "error": error,
                "source_unchanged": unchanged,
                "wall_seconds": time.monotonic() - start,
            }
            save_new(directory / "result.json", result)
            journal.append("attempt_finished", **result, result_sha256=sha(encoded(result)))
            if observation.get("worker_quiescent") is False:
                preserve_lock = True
                raise RuntimeError("Worker is still active; retained failure and runner lock")
            if not unchanged:
                raise ValueError("Source or corpus changed during trial; retained as failed")
        return [row for row in journal.rows if row["kind"] == "attempt_finished"]
    finally:
        if not preserve_lock:
            lock.unlink()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="Frozen engine checkout matching explicit PYTHONPATH; harness source stays separate",
    )
    parser.add_argument(
        "--expected-engine",
        help="Exact engine/runtime fingerprint for a matched frozen source image",
    )
    parser.add_argument("--cases", required=True, help="Explicit comma-separated IDs")
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument(
        "--condition",
        choices=("cold", "warm"),
        required=True,
        help="Operator-declared host/cache condition, not inferred",
    )
    parser.add_argument("--executor", choices=("docker", "trusted_local"), default="docker")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--actions", type=int, default=16)
    parser.add_argument(
        "--generation-policy",
        choices=("staged", "legacy"),
        help="Mandatory for actual trials; verifies the existing environment flag without changing it",
    )
    parser.add_argument(
        "--fixtures", type=Path, help="Pure harness test only; never actual outcome evidence"
    )
    args = parser.parse_args(argv)
    if not args.fixtures and args.generation_policy is None:
        parser.error("actual trials require --generation-policy staged|legacy")
    if not 1 <= args.seconds <= 3600 or not 1 <= args.actions <= 100:
        parser.error("seconds must be 1..3600 and actions 1..100")
    adapter: Adapter = (
        FixtureAdapter(strict_json(args.fixtures.read_bytes()))
        if args.fixtures
        else CustomerLocalAdapter(
            args.corpus,
            executor=args.executor,
            seconds=args.seconds,
            actions=args.actions,
            condition=args.condition,
            source_root=args.source_root,
            expected_engine=args.expected_engine,
            generation_policy=args.generation_policy,
        )
    )
    rows = run_benchmark(
        args.output,
        args.corpus,
        args.cases.split(","),
        adapter,
        split=args.split,
        max_cases=args.max_cases,
        condition=args.condition,
        inventory=partial(source_inventory, args.source_root.resolve()),
    )
    print(
        json.dumps(
            {
                "attempts": len(rows),
                "actual_passes": sum(row["assessment"]["nominal_digital_pass"] for row in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
