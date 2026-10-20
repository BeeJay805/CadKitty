"""Cancellable workers and verified artifact transfer under explicit resource policies."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from forge_mind.agentic.sandbox import _ENV_PASSTHROUGH, _kill_tree
from forge_mind.workspace.budget import end_time, expired
from forge_mind.workspace.capacity import completion_policy, optional_limit
from forge_mind.workspace.tessellation import export_verified_tessellation
from forge_mind.workspace.worker_threads import WORKER_THREAD_ENV

# These integers remain the explicit hosted/source-extraction isolation policy.
# Trusted local workers and ordinary artifact transfer use optional operator limits below.
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_JOB_BYTES = 256 * 1024 * 1024
HOSTED_LIMITS = {
    "file_bytes": MAX_FILE_BYTES,
    "job_bytes": MAX_JOB_BYTES,
    "memory_bytes": 2048 * 1024 * 1024,
    "result_bytes": 1024 * 1024,
    "cpus": 1,
    "pids": 64,
    "tmpfs_bytes": 256 * 1024 * 1024,
}


def resource_policy() -> dict[str, Any]:
    """Discover current local limits separately from the hosted isolation envelope."""
    from forge_mind.workspace.source_capacity import source_policy

    names = ("FILE_BYTES", "JOB_BYTES", "RSS_BYTES", "RESULT_BYTES")
    return {
        "local": {
            name.lower(): {
                "value": optional_limit(name),
                "environment": "FORGE_WORKSPACE_" + name,
                "unit": "bytes",
            }
            for name in names
        },
        "structural": {
            name.lower(): {
                "value": optional_limit(name),
                "environment": "FORGE_WORKSPACE_" + name,
                "unit": "bytes" if name.endswith("_BYTES") else "count",
            }
            for name in (
                "STRUCTURAL_TET_LIMIT",
                "STRUCTURAL_DOF_LIMIT",
                "STRUCTURAL_SURFACE_FACE_LIMIT",
                "STRUCTURAL_SURFACE_VERTEX_LIMIT",
                "STRUCTURAL_FIELD_POINT_LIMIT",
                "STRUCTURAL_FIELD_TRIANGLE_LIMIT",
                "STRUCTURAL_QUADRATURE_LIMIT",
                "STRUCTURAL_FIELD_BYTES",
                "STRUCTURAL_RECEIPT_BYTES",
            )
        },
        "unset": "Unset or empty means no configured local limit; configured values must be positive integers.",
        "completion": completion_policy(),
        "source_capture": source_policy(),
        "scope": "Artifact transfer, trusted local execution and workspace postprocessing. Explicit per-job deadlines and cancellation still apply.",
        "hosted": dict(HOSTED_LIMITS),
        "hosted_scope": "Hosted code execution retains its independent container limits, network isolation, read-only root, unprivileged user and process restrictions. Local policies do not raise these limits.",
    }


def regular_file(path: Path, maximum: int | None = None) -> bool:
    configured = optional_limit("FILE_BYTES")
    if configured is not None:
        maximum = configured if maximum is None else min(maximum, configured)
    try:
        info = path.lstat()
        return (
            stat.S_ISREG(info.st_mode)
            and not path.is_symlink()
            and info.st_nlink == 1
            and (maximum is None or info.st_size <= maximum)
        )
    except OSError:
        return False


def copy_artifact(source: Path, target: Path) -> None:
    """Call only after the producing process tree/container has terminated."""
    maximum = optional_limit("FILE_BYTES")
    if not regular_file(source, maximum):
        raise ValueError(
            f"Worker artifact is missing, linked, nonregular, or oversized: {source.name}"
        )
    # Exclusive creation never follows a preexisting destination link.
    with source.open("rb") as reader, target.open("xb") as writer:
        remaining = maximum
        while data := reader.read(
            1024 * 1024 if remaining is None else min(1024 * 1024, remaining + 1)
        ):
            if remaining is not None:
                remaining -= len(data)
                if remaining < 0:
                    raise ValueError("Artifact grew beyond the operator file-byte transfer limit")
            writer.write(data)


def directory_size(directory: Path, stop_after: int | None = None) -> int:
    total = 0
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for name in files:
            try:
                info = (Path(root) / name).lstat()
            except FileNotFoundError:
                # Workers may remove intermediate meshes between enumeration
                # and stat. A vanished file contributes no current disk usage.
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
        if stop_after is not None and total > stop_after:
            break
    return total


def bounded_process(
    command: list[str],
    directory: Path,
    timeout: float | None,
    cancelled: Callable[[], bool],
    *,
    cpu_limit: int | None = None,
) -> None:
    """Always poll cancellation; apply only configured local memory/output policies."""
    rss_limit = optional_limit("RSS_BYTES")
    output_limit = optional_limit("JOB_BYTES")

    def check_output() -> None:
        if output_limit is not None and directory_size(directory, output_limit) > output_limit:
            raise ValueError(
                f"Operation exceeded its operator output-byte limit ({output_limit} bytes; FORGE_WORKSPACE_JOB_BYTES)"
            )

    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    # PYTHONPATH may be relative to the service cwd; the child cwd is the job directory.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONHASHSEED"] = "0"
    for key, value in os.environ.items():
        if key.startswith("FORGE_WORKSPACE_") or key == "CADKITTY_SU2_EXE":
            env[key] = value
    # Apply before child startup, including ordinary processing/render workers.
    # CPU affinity remains a separate policy below.
    env.update(WORKER_THREAD_ENV)
    deadline = end_time(timeout)
    if cancelled() or expired(deadline):
        raise ValueError("Operation cancelled or deadline exhausted")
    check_output()
    with (directory / "worker.log").open("xb") as log:
        proc = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            process = psutil.Process(proc.pid)
            if cpu_limit and hasattr(process, "cpu_affinity"):
                process.cpu_affinity(process.cpu_affinity()[:cpu_limit])
            while proc.poll() is None:
                if cancelled():
                    raise ValueError("Operation cancelled")
                if expired(deadline):
                    raise ValueError("Operation exceeded its wall-clock limit")
                if rss_limit is not None:
                    try:
                        rss = sum(
                            p.memory_info().rss
                            for p in [process, *process.children(recursive=True)]
                        )
                    except psutil.NoSuchProcess:
                        rss = 0
                    if rss > rss_limit:
                        raise ValueError(
                            f"Operation exceeded its operator memory limit ({rss_limit} bytes; FORGE_WORKSPACE_RSS_BYTES)"
                        )
                check_output()
                time.sleep(0.1)
            check_output()
            if proc.returncode:
                log.flush()
                with (directory / "worker.log").open("rb") as reader:
                    reader.seek(max(0, reader.seek(0, os.SEEK_END) - 6400))
                    tail = reader.read(6400).decode("utf-8", errors="replace")[-1600:]
                raise ValueError(f"CAD worker failed (exit={proc.returncode}): {tail}")
        finally:
            _kill_tree(proc.pid)
            proc.wait(timeout=10)


def process_revision(
    directory: Path,
    payload: dict[str, Any],
    timeout: float | None,
    cancelled: Callable[[], bool],
    *,
    operation_job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if operation_job is not None:
        payload = {**payload, "_operation_job": operation_job}
    request = directory / "processing.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    bounded_process(
        [sys.executable, "-m", "forge_mind.workspace.processing", str(request)],
        directory,
        timeout,
        cancelled,
        **(
            {"cpu_limit": 2}
            if payload.get("operation")
            in {
                "motion_check",
                "product_tool",
                "nominal_through_pattern",
                "nominal_perforated_box",
                "structural_study",
                "study_interface_geometry",
                "study_interface_targets",
            }
            else {}
        ),
    )
    result = directory / "processed.json"
    if not regular_file(result, optional_limit("RESULT_BYTES")):
        raise ValueError("Postprocessor returned no regular result within operator resource policy")
    data: dict[str, Any] = json.loads(result.read_text(encoding="utf-8"))
    return data


def _write_definition_mesh_manifest(
    directory: Path,
    revision_id: str,
    selected: dict[str, int],
    result: dict[str, Any],
) -> None:
    """Bind selected local mesh outputs; this never starts CAD or changes STEP facts."""
    if not selected:
        return
    from forge_mind.validation_provenance import file_sha256
    from forge_mind.workspace.definition_meshes import MANIFEST_NAME, DefinitionMeshesManifest

    parts: dict[str, Any] = {}
    mesh_hashes: dict[str, str] = {}
    definitions = result.get("facts", {}).get("definitions", {})
    for part_id, quantity in sorted(selected.items()):
        delivery = definitions.get(part_id, {}).get("mesh_delivery")
        if delivery is None:
            step = directory / f"definition_{part_id}.step"
            delivery = {
                "status": "failed",
                "step_file": step.name,
                "step_sha256": file_sha256(step) if regular_file(step) else None,
                "error": f"Definition {part_id} mesh measurement is unavailable; see revision errors",
            }
        if delivery["status"] == "available":
            name = delivery["stl_file"]
            target = directory / name
            if (
                not regular_file(target)
                or target.stat().st_size != delivery["stl_bytes"]
                or file_sha256(target) != delivery["stl_sha256"]
            ):
                delivery = {
                    "status": "failed",
                    "step_file": delivery["step_file"],
                    "step_sha256": delivery["step_sha256"],
                    "error": f"Definition {part_id} retained STL no longer matches its measured bytes",
                }
            else:
                mesh_hashes[name] = delivery["stl_sha256"]
        parts[part_id] = {"role": "make", "quantity": quantity, **delivery}
    manifest = {
        "schema_version": 1,
        "revision_id": revision_id,
        "document_sha256": file_sha256(directory / "document.json"),
        "frame": "definition_local",
        "units": "mm",
        "parts": parts,
    }
    DefinitionMeshesManifest.model_validate(manifest)
    with (directory / MANIFEST_NAME).open("x", encoding="utf-8") as output:
        json.dump(manifest, output, allow_nan=False, indent=2)
    result["artifact_hashes"].update(mesh_hashes)
    result["artifact_hashes"][MANIFEST_NAME] = file_sha256(directory / MANIFEST_NAME)


def _stage_interface_source(source: Path, directory: Path) -> dict[str, str]:
    """Give OCCT relative names; Python copies and verifies the long source path."""
    from forge_mind.validation_provenance import file_sha256

    hashes = {}
    for name in ("part.step", "document.json"):
        expected = file_sha256(source / name)
        copy_artifact(source / name, directory / name)
        if file_sha256(source / name) != expected or file_sha256(directory / name) != expected:
            raise ValueError("Interface source changed during worker capture")
        hashes[name] = expected
    return hashes


def main(request: Path) -> None:
    payload = json.loads(request.read_text(encoding="utf-8"))
    if "_operation_job" in payload:
        from forge_mind.workspace.operation_recovery import register_child

        register_child(request.resolve().parent, payload.pop("_operation_job"))
    # This entrypoint runs in its own worker process. OCCT's native file APIs
    # can reject Windows absolute paths beyond MAX_PATH even when Python can
    # read them. Keep native output names relative to the owned job directory.
    os.chdir(request.resolve().parent)
    directory = Path(".")
    if payload.get("operation") == "section":
        from forge_mind.workspace.measure import section_revision

        result = section_revision(
            Path(payload["source"]),
            payload["axis"],
            payload["offset_mm"],
            directory / "section.png",
        )
    elif payload.get("operation") == "nominal_perforated_box":
        from forge_mind.workspace.nominal_perforated_box import evaluate_nominal_perforated_box

        result = evaluate_nominal_perforated_box(directory / "part.step", payload["request"])
    elif payload.get("operation") == "nominal_through_pattern":
        from forge_mind.workspace.nominal_through_pattern import evaluate_nominal_through_pattern

        result = evaluate_nominal_through_pattern(directory / "part.step", payload["request"])
    elif payload.get("operation") == "photo_profile_analyze":
        from forge_mind.workspace.photo_profile import analyze_profile

        result = analyze_profile((directory / "vision.png").read_bytes(), payload["request"])
    elif payload.get("operation") == "nominal_photo_solid":
        from forge_mind.workspace.nominal_photo_solid import evaluate_nominal_photo_solid

        result = evaluate_nominal_photo_solid(directory / "part.step", payload["request"])
    elif payload.get("operation") == "photo_profile_compare":
        from forge_mind.workspace.photo_profile_native import compare_saved_profile

        result = compare_saved_profile(directory, payload["observation"], payload["selector"])
    elif payload.get("operation") == "product_tool":
        from forge_mind.workspace.product_tools import run_product_tool

        result = run_product_tool(
            payload["request"], Path(payload["source"]) if payload.get("source") else directory
        )
    elif payload.get("operation") == "motion_check":
        from forge_mind.workspace.motion_check import run_motion_check

        result = run_motion_check(directory, payload)
    elif payload.get("operation") == "engineering":
        from forge_mind.workspace.engineering import run_engineering

        result = run_engineering(directory, payload)
    elif payload.get("operation") == "study_geometry":
        from forge_mind.workspace.design_evaluators import geometry_worker

        result = geometry_worker(Path(payload["source_step"]))
    elif payload.get("operation") == "study_interface_geometry":
        from forge_mind.validation_provenance import file_sha256
        from forge_mind.workspace.interface_geometry import interface_geometry_worker

        source = Path(payload["source"])
        hashes = _stage_interface_source(source, directory)
        result = interface_geometry_worker(directory, payload["settings"])
        if any(file_sha256(source / name) != value for name, value in hashes.items()):
            raise ValueError("Interface source changed during worker measurement")
    elif payload.get("operation") == "study_interface_targets":
        from forge_mind.validation_provenance import file_sha256
        from forge_mind.workspace.document import DesignDocument, digest
        from forge_mind.workspace.interface_geometry import (
            bind_interface_targets,
            validate_interface_targets,
        )

        source = Path(payload["source"])
        expected = payload["artifact_hashes"]
        if any(
            file_sha256(source / name) != expected.get(name)
            for name in ("part.step", "document.json")
        ):
            raise ValueError("Interface target source changed before binding")
        document = DesignDocument.model_validate_json(
            (source / "document.json").read_text(encoding="utf-8")
        )
        if digest(document.model_dump()) != payload["document_hash"]:
            raise ValueError("Interface target document differs from the saved revision")
        if _stage_interface_source(source, directory) != expected:
            raise ValueError("Interface target source changed during worker capture")
        if payload.get("bindings") is None:
            result = bind_interface_targets(document, directory, payload["settings"])
        else:
            validate_interface_targets(
                document, directory, payload["settings"], payload["bindings"]
            )
            result = {"status": "matched"}
        if any(
            file_sha256(source / name) != expected.get(name)
            for name in ("part.step", "document.json")
        ):
            raise ValueError("Interface target source changed during binding")
    elif payload.get("operation") == "study_flow":
        from forge_mind.workspace.design_evaluators import flow_worker

        result = flow_worker(directory, payload)
    elif payload.get("operation") == "study_flow_stage":
        from forge_mind.workspace.flow_stage_cache import run_flow_stage

        result = run_flow_stage(directory, payload)
    elif payload.get("operation") == "structural_study":
        from forge_mind.workspace.structural import run_structural_study

        result = run_structural_study(directory, payload)
    else:
        import trimesh
        from build123d import export_step, export_stl, import_step

        from forge_mind.viz.glb import to_gltf_yup
        from forge_mind.viz.raster import WORKSPACE_PREVIEW_PIXELS
        from forge_mind.workspace.measure import measure_revision
        from forge_mind.workspace.models import Contract

        # STEP is the authoritative artifact. All deliverable meshes and previews are
        # derived here by trusted code, never accepted from the CAD author.
        history = None
        if payload.get("document"):
            from forge_mind.workspace.document import DesignDocument
            from forge_mind.workspace.kernel import compile_document

            shape, history = compile_document(
                DesignDocument.model_validate(payload["document"]),
                {key: Path(value) for key, value in payload.get("assets", {}).items()},
                directory,
                Path(payload["cache_source"]) if payload.get("cache_source") else None,
            )
            export_step(shape, str(directory / "part.step"))
        shape = import_step(str(directory / "part.step"))
        from forge_mind.workspace.topology import topology_index

        topology = topology_index(
            shape,
            assembly=bool(
                payload.get("document", {}).get("mechanism")
                or payload.get("document", {}).get("assembly")
            ),
        )
        (directory / "topology.json").write_text(json.dumps(topology), encoding="utf-8")
        linear, angular, collapsed = export_verified_tessellation(directory)
        scene: Any = trimesh.Scene()
        for index, solid in enumerate(shape.solids()):
            intermediate = directory / f"solid_{index}.stl"
            export_stl(
                solid,
                str(intermediate),
                tolerance=linear,
                angular_tolerance=angular,
            )
            mesh: Any = trimesh.load(str(intermediate), force="mesh")
            to_gltf_yup(mesh)
            mesh.visual.face_colors = (
                [186, 204, 211, 255] if index % 2 == 0 else [170, 196, 183, 255]
            )
            scene.add_geometry(mesh, node_name=f"solid_{index + 1}", geom_name=f"solid_{index + 1}")
            intermediate.unlink()
        scene.export(str(directory / "part.glb"))
        assembly = None
        assembly_record = None
        if history is not None:
            assembly_record = next((h for h in history if h["id"] == "__assembly__"), None)
            if assembly_record:
                from forge_mind.workspace.assembly import assembly_evidence

                assembly = assembly_evidence(
                    DesignDocument.model_validate(payload["document"]), shape, assembly_record
                )
        # The deliverable mesh retains its precision. Six-view rasterization uses
        # a separate coarse tessellation to bound matplotlib's per-face memory.
        from forge_mind.workspace.preview_appearance import render_primary_preview

        preview_bounds, appearance = render_primary_preview(
            directory, shape, payload.get("document"), assembly
        )
        from forge_mind.workspace.definition_meshes import selected_definition_meshes

        selected_meshes = (
            selected_definition_meshes(DesignDocument.model_validate(payload["document"]))
            if payload.get("document")
            else {}
        )
        result = measure_revision(
            directory,
            payload["revision_id"],
            Contract.model_validate(payload["contract"]),
            structured_assembly=bool(payload.get("document", {}).get("assembly")),
            definition_mesh_ids=tuple(selected_meshes),
        ).model_dump()
        result["facts"]["mesh_source"] = "Trusted tessellation of the final exported STEP"
        result["facts"]["render_source"] = (
            "Six orthographic views of every triangle from separate STEP tessellation at "
            "relative linear deflection 0.15 / angular deflection 0.35 rad for display; "
            f"nearest-surface depth is evaluated per pixel at {WORKSPACE_PREVIEW_PIXELS}x{WORKSPACE_PREVIEW_PIXELS} per view before layout. "
            "Features below pixel resolution may not be visible. Deliverable STL uses "
            f"relative linear deflection {linear} / angular deflection {angular} rad and is measured independently; these settings are not manufacturing tolerances"
        )
        result["facts"]["collapsed_export_triangles_removed"] = collapsed
        result["facts"]["preview_appearance"] = appearance
        if history is not None:
            result["facts"]["feature_history"] = history
            if assembly_record and assembly is not None:
                from forge_mind.validation_provenance import file_sha256

                assembly.update(
                    revision_id=payload["revision_id"],
                    engine_fingerprint=result["engine_fingerprint"],
                    step_sha256=file_sha256(directory / "part.step"),
                )
                (directory / "assembly.json").write_text(json.dumps(assembly), encoding="utf-8")
                result["facts"]["assembly"] = assembly
                for name in [
                    "assembly.json",
                    "parts.json",
                    *[v["file"] for v in assembly_record["parts"].values()],
                    *[v["step"] for v in assembly_record["parts"].values()],
                ]:
                    result["artifact_hashes"][name] = file_sha256(directory / name)
                result["facts"]["feature_history"] = [
                    h for h in history if not h["id"].startswith("__")
                ]
            document = DesignDocument.model_validate(payload["document"])
            if document.motion_rig:
                from forge_mind.workspace.motion_rig import bind_rig

                motion_rig = bind_rig(
                    document,
                    result["facts"]["assembly"],
                    revision_id=payload["revision_id"],
                    engine_fingerprint=result["engine_fingerprint"],
                    step_sha256=file_sha256(directory / "part.step"),
                    document_sha256=file_sha256(directory / "document.json"),
                )
                (directory / "motion-rig.json").write_text(
                    json.dumps(motion_rig, allow_nan=False), encoding="utf-8"
                )
                result["facts"]["motion_rig"] = motion_rig
                result["artifact_hashes"]["motion-rig.json"] = file_sha256(
                    directory / "motion-rig.json"
                )
            mechanism_record = next((h for h in history if h["id"] == "__mechanism__"), None)
            if mechanism_record:
                from forge_mind.validation_provenance import file_sha256
                from forge_mind.workspace.mechanism_geometry import bind_exported

                motion = bind_exported(
                    mechanism_record,
                    topology,
                    file_sha256(directory / "part.step"),
                    file_sha256(directory / "document.json"),
                    shape,
                )
                motion["revision_id"] = payload["revision_id"]
                motion["engine_fingerprint"] = result["engine_fingerprint"]
                (directory / "motion.json").write_text(
                    json.dumps(motion, allow_nan=False), encoding="utf-8"
                )
                result["facts"]["mechanism"] = motion
                result["facts"]["feature_history"] = [
                    h for h in history if h["id"] != "__mechanism__"
                ]
                result["artifact_hashes"]["motion.json"] = file_sha256(directory / "motion.json")
        from forge_mind.validation_provenance import file_sha256

        result["artifact_hashes"]["topology.json"] = file_sha256(directory / "topology.json")
        if (directory / "document.json").is_file():
            result["artifact_hashes"]["document.json"] = file_sha256(directory / "document.json")
        _write_definition_mesh_manifest(directory, payload["revision_id"], selected_meshes, result)
        from forge_mind.workspace.preview_inspection import add_inspection_preview

        add_inspection_preview(directory, shape, payload.get("document"), result, preview_bounds)
    (directory / "processed.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
