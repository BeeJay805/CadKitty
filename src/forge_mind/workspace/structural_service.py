"""Queued, immutable component analyses with separate bounded field artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.capacity import enforce_limit, optional_limit
from forge_mind.workspace.document import digest
from forge_mind.workspace.engineering_service import validate_engineering_worker
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.processing import process_revision, regular_file
from forge_mind.workspace.product_sources import read_sources
from forge_mind.workspace.store import WorkspaceError
from forge_mind.workspace.structural_models import STRUCTURAL_SCOPE, StructuralStudy

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService

STUDY_ARTIFACTS = frozenset(
    {"field.json", "coarse/selected.step", "fine/selected.step", "coarse/mesh.npz", "fine/mesh.npz"}
)


def _table(service: WorkspaceService) -> None:
    with service.store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS structural_studies "
            "(id TEXT PRIMARY KEY, session TEXT NOT NULL, document TEXT NOT NULL, digest TEXT NOT NULL, "
            "FOREIGN KEY(session) REFERENCES sessions(id))"
        )


def validate_field(field: dict[str, Any]) -> None:
    """A complete mesh association is mandatory; missing samples never become zero."""
    if field.get("schema_version") != 1 or field.get("frame") != "cad_z_up_mm":
        raise ValueError("Unsupported structural field version or coordinate frame")
    for key, policy in (
        ("positions_mm", "STRUCTURAL_FIELD_POINT_LIMIT"),
        ("triangles", "STRUCTURAL_FIELD_TRIANGLE_LIMIT"),
        ("quadrature_positions_mm", "STRUCTURAL_QUADRATURE_LIMIT"),
        ("volume_positions_mm", "STRUCTURAL_FIELD_POINT_LIMIT"),
        ("tetrahedra", "STRUCTURAL_TET_LIMIT"),
    ):
        values = field.get(key)
        if isinstance(values, (list, tuple)) or (isinstance(values, np.ndarray) and values.ndim):
            enforce_limit(len(values), optional_limit(policy), "structural " + key)
    points = np.asarray(field.get("positions_mm"), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("Structural field has invalid vertex coordinates")
    if not np.isfinite(points).all():
        raise ValueError("Structural field coordinates are nonfinite")
    for name, shape in (
        ("displacements_mm", points.shape),
        ("von_mises_mpa", (len(points),)),
        ("strain_equivalent", (len(points),)),
    ):
        values = np.asarray(field.get(name), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Structural field {name} has missing or nonfinite samples")
        if name != "displacements_mm" and np.any(values < 0):
            raise ValueError(f"Structural field {name} is negative")
    triangles = np.asarray(field.get("triangles"))
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(triangles) < 4
        or triangles.dtype.kind not in "iu"
        or np.any(triangles < 0)
        or np.any(triangles >= len(points))
        or np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0)
    ):
        raise ValueError("Structural field connectivity is invalid")
    boundaries = []
    for name in ("fixed_triangles", "loaded_triangles"):
        values = np.asarray(field.get(name))
        if (
            values.ndim != 1
            or not len(values)
            or values.dtype.kind not in "iu"
            or np.any(values < 0)
            or np.any(values >= len(triangles))
            or len(np.unique(values)) != len(values)
        ):
            raise ValueError(f"Structural field {name} membership is invalid")
        boundaries.append(set(values.tolist()))
    if boundaries[0] & boundaries[1]:
        raise ValueError("Structural field fixed and loaded facets overlap")
    quadrature_keys = ("quadrature_positions_mm", "stress_tensors_mpa", "strain_tensors")
    if any(key in field for key in quadrature_keys):
        if not all(key in field for key in quadrature_keys):
            raise ValueError("Structural quadrature tensor association is incomplete")
        quadrature = np.asarray(field["quadrature_positions_mm"], dtype=float)
        if (
            quadrature.ndim != 2
            or quadrature.shape[1] != 3
            or len(quadrature) < 1
            or not np.isfinite(quadrature).all()
        ):
            raise ValueError("Structural quadrature coordinates are invalid")
        for key in quadrature_keys[1:]:
            tensor = np.asarray(field[key], dtype=float)
            if (
                tensor.shape != (len(quadrature), 3, 3)
                or not np.isfinite(tensor).all()
                or not np.allclose(tensor, tensor.transpose(0, 2, 1), rtol=1e-8, atol=1e-10)
            ):
                raise ValueError("Structural stress/strain tensor association is invalid")
    volume_keys = ("volume_positions_mm", "tetrahedra", "quadrature_element_ids")
    if any(key in field for key in volume_keys):
        if not all(key in field for key in (*volume_keys, *quadrature_keys)):
            raise ValueError("Structural volume mesh association is incomplete")
        volume = np.asarray(field["volume_positions_mm"], dtype=float)
        tetrahedra = np.asarray(field["tetrahedra"])
        element_ids = np.asarray(field["quadrature_element_ids"])
        if (
            volume.ndim != 2
            or volume.shape[1] != 3
            or len(volume) < 4
            or not np.isfinite(volume).all()
            or tetrahedra.ndim != 2
            or tetrahedra.shape[1] != 4
            or len(tetrahedra) < 1
            or tetrahedra.dtype.kind not in "iu"
            or np.any(tetrahedra < 0)
            or np.any(tetrahedra >= len(volume))
            or np.any(np.diff(np.sort(tetrahedra, axis=1), axis=1) == 0)
            or element_ids.shape != (len(field["quadrature_positions_mm"]),)
            or element_ids.dtype.kind not in "iu"
            or np.any(element_ids < 0)
            or np.any(element_ids >= len(tetrahedra))
        ):
            raise ValueError("Structural volume mesh connectivity is invalid")


def study_path(service: WorkspaceService, sid: str, study_id: str) -> Path:
    service.store.directory(sid, study_id)  # validate both internally generated IDs
    return service.store.root / "structural-studies" / sid / study_id


def _read_record(service: WorkspaceService, sid: str, owner: str, study_id: str) -> dict[str, Any]:
    service.store.session_record(sid, owner)
    _table(service)
    with service.store.connect() as con:
        row = con.execute(
            "SELECT document,digest FROM structural_studies WHERE id=? AND session=?",
            (study_id, sid),
        ).fetchone()
    if row is None:
        raise WorkspaceError("Structural study is unavailable; wait for its job to finish")
    enforce_limit(
        len(row[0].encode()), optional_limit("STRUCTURAL_RECEIPT_BYTES"), "structural receipt bytes"
    )
    if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
        raise WorkspaceError("Structural receipt integrity check failed")
    return dict(json.loads(row[0]))


def read_study(service: WorkspaceService, sid: str, owner: str, study_id: str) -> dict[str, Any]:
    result = _read_record(service, sid, owner, study_id)
    session = service.store.session_record(sid, owner)
    session["engineering"] = service.store.engineering(sid)
    revision = service._load_revision(sid, result["revision_id"])
    hashes = revision["measurement"]["artifact_hashes"]
    if (
        hashes.get("part.step") != result["geometry_hash"]
        or hashes.get("topology.json") != result["topology_hash"]
        or digest(result["request"]) != result["request_hash"]
    ):
        raise WorkspaceError("Structural evidence does not match its source geometry or setup")
    if result.get("status") == "computed":
        required = STUDY_ARTIFACTS if result["request"].get("mesher") == "cad" else {"field.json"}
        if not required.issubset(result["hashes"]):
            raise WorkspaceError("Computed structural study is missing required mesh provenance")
    for name, expected in result["hashes"].items():
        if name not in STUDY_ARTIFACTS:
            raise WorkspaceError("Unknown structural artifact")
        path = study_path(service, sid, study_id) / name
        maximum = optional_limit("STRUCTURAL_FIELD_BYTES") if name == "field.json" else None
        if not regular_file(path, maximum) or file_sha256(path) != expected:
            raise WorkspaceError("Structural field artifact changed or is unavailable")
    current = (session.get("proposal") or {}).get("revision_id") or session.get("selected_revision")
    result["matches_current_design"] = current == result["revision_id"]
    result["stale"] = (
        result["engine_fingerprint"] != engine_fingerprint()
        or result["input_version"] != session["engineering"]["version"]
        or result["input_hash"] != digest(session["engineering"]["inputs"])
        or result["source_fingerprint"] != read_sources(service.store, sid, owner)["fingerprint"]
    )
    result["artifact_integrity"] = "verified"
    return result


def list_studies(
    service: WorkspaceService, sid: str, owner: str, rid: str | None = None
) -> list[dict[str, Any]]:
    session = service.store.session_record(sid, owner)
    engineering = service.store.engineering(sid)
    inputs_hash = digest(engineering["inputs"])
    sources_hash = read_sources(service.store, sid, owner)["fingerprint"]
    engine = engine_fingerprint()
    current = (session.get("proposal") or {}).get("revision_id") or session.get("selected_revision")
    _table(service)
    with service.store.connect() as con:
        ids = [
            r[0]
            for r in con.execute(
                "SELECT id FROM structural_studies WHERE session=? "
                "AND (? IS NULL OR json_extract(document,'$.revision_id')=?) "
                "ORDER BY rowid DESC LIMIT 30",
                (sid, rid, rid),
            )
        ]
    results = []
    for study_id in ids:
        result = _read_record(service, sid, owner, study_id)
        result.update(
            matches_current_design=result["revision_id"] == current,
            stale=result["engine_fingerprint"] != engine
            or result["input_version"] != engineering["version"]
            or result["input_hash"] != inputs_hash
            or result["source_fingerprint"] != sources_hash,
            artifact_integrity="Receipt verified; read the study or field to verify artifact bytes",
        )
        results.append(result)
    return results


def field_path(service: WorkspaceService, sid: str, owner: str, study_id: str) -> Path:
    receipt = read_study(service, sid, owner, study_id)
    if "field.json" not in receipt["hashes"]:
        raise WorkspaceError("This study has no computed field")
    return study_path(service, sid, study_id) / "field.json"


def validate_structural_worker(service: WorkspaceService) -> None:
    """Reuse the private solver boundary at admission and execution."""
    validate_engineering_worker(
        service, "Structural worker is private-local until container verification"
    )


def run_structural_job(
    service: WorkspaceService, job: dict[str, Any], cancelled: Callable[[], bool]
) -> None:
    validate_structural_worker(service)
    sid, owner, rid = job["session"], job["owner"], job["revision_id"]
    session = service.store.session_record(sid, owner)
    session["engineering"] = service.store.engineering(sid)
    revision = service._load_revision(sid, rid)
    if not revision["measurement"]["geometry_valid"]:
        raise WorkspaceError("Structural study requires valid exported geometry")
    fingerprint = engine_fingerprint()
    if revision["measurement"]["engine_fingerprint"] != fingerprint:
        raise WorkspaceError("Rebuild this historical geometry before running a structural study")
    if session["engineering"]["version"] != job["input_version"]:
        raise WorkspaceError("Engineering inputs changed before the structural study started")
    study = StructuralStudy.model_validate(job["study"])
    source_state = read_sources(service.store, sid, owner)
    inputs_hash = digest(session["engineering"]["inputs"])
    directory = study_path(service, sid, job["study_id"])
    directory.mkdir(parents=True)
    result = process_revision(
        directory,
        {
            "operation": "structural_study",
            "source": str(service.store.directory(sid, rid)),
            "revision": revision,
            "study": study.model_dump(),
        },
        job.get("max_seconds"),
        cancelled,
    )
    if cancelled():
        raise WorkspaceError("Structural study cancelled before evidence publication")
    service._load_revision(sid, rid)
    if engine_fingerprint() != fingerprint:
        raise WorkspaceError("Structural instruments changed during the study")
    hashes: dict[str, str] = {}
    if result.get("status") == "computed":
        path = directory / "field.json"
        if not regular_file(path, optional_limit("STRUCTURAL_FIELD_BYTES")):
            raise WorkspaceError(
                "Structural worker returned no regular field artifact within operator policy"
            )
        field = json.loads(path.read_text(encoding="utf-8"))
        validate_field(field)
        hashes["field.json"] = file_sha256(path)
    for name in STUDY_ARTIFACTS - {"field.json"}:
        path = directory / name
        if result.get("status") == "computed" and study.mesher == "cad" and not regular_file(path):
            raise WorkspaceError("Computed CAD study is missing required mesh provenance: " + name)
        if path.exists():
            if not regular_file(path):
                raise WorkspaceError("Structural mesh provenance artifact is invalid or oversized")
            hashes[name] = file_sha256(path)
    result.update(
        schema_version=1,
        id=job["study_id"],
        job_id=job["id"],
        revision_id=rid,
        input_version=job["input_version"],
        input_hash=inputs_hash,
        request_hash=digest(study.model_dump()),
        request=study.model_dump(),
        engine_fingerprint=fingerprint,
        geometry_hash=revision["measurement"]["artifact_hashes"]["part.step"],
        topology_hash=revision["measurement"]["artifact_hashes"]["topology.json"],
        source_fingerprint=source_state["fingerprint"],
        source_bindings=[
            {
                "claim_id": claim,
                "status": "confirmed_interpretation"
                if any(
                    c["id"] == claim and c.get("confirmation")
                    for c in source_state.get("claims", [])
                )
                else "unconfirmed_or_missing",
            }
            for claim in study.source_claim_ids
        ],
        qualification_status="unverified",
        scope=STRUCTURAL_SCOPE,
        hashes=hashes,
        at=time.time(),
    )
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    enforce_limit(
        len(encoded.encode()),
        optional_limit("STRUCTURAL_RECEIPT_BYTES"),
        "structural receipt bytes",
    )
    _table(service)
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        active = con.execute(
            "SELECT state,cancel FROM jobs WHERE id=? AND session=?", (job["id"], sid)
        ).fetchone()
        input_row = con.execute(
            "SELECT version,document FROM engineering_inputs WHERE session=?", (sid,)
        ).fetchone()
        from forge_mind.workspace.engineering_models import EngineeringInputs
        from forge_mind.workspace.product_sources import source_fingerprint

        current_inputs = json.loads(input_row[1]) if input_row else EngineeringInputs().model_dump()
        current_record = json.loads(
            con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
        )
        current_design = (current_record.get("proposal") or {}).get(
            "revision_id"
        ) or current_record.get("selected_revision")

        if (
            active is None
            or active[0] != "running"
            or active[1]
            or (input_row[0] if input_row else 0) != job["input_version"]
            or digest(current_inputs) != inputs_hash
            or source_fingerprint(con, sid) != source_state["fingerprint"]
            or current_design != rid
        ):
            raise WorkspaceError(
                "Structural study cancelled or its engineering setup changed before publication"
            )
        con.execute(
            "INSERT INTO structural_studies VALUES (?,?,?,?)",
            (job["study_id"], sid, encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
    service.store.event(
        sid,
        "structural_study",
        "Component study finished: " + study.name,
        study_id=job["study_id"],
        revision_id=rid,
    )
