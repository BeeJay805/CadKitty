"""Add a display-only reference-excluded view after authoritative body binding."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.preview_selection import select_preview_bodies

SCOPE = (
    "Declared reference bodies hidden for inspection; complete view and all validation "
    "retain every body."
)


def add_inspection_preview(
    directory: Path,
    shape: Any,
    document: dict[str, Any] | None,
    result: dict[str, Any],
    frame_bounds: np.ndarray,
) -> dict[str, Any]:
    """Bind optional image bytes to the full STEP and current trusted measurement.

    Only same-invocation assembly evidence is supplied here. Failure to render a
    supplemental view is visible in metadata and never changes validation status.
    """
    from build123d import Compound, export_stl

    from forge_mind.viz.render import render_workspace_preview

    solids = list(shape.solids())
    selection = select_preview_bodies(document, result["facts"].get("assembly"), len(solids))
    details = asdict(selection)
    details.update(
        body_count=len(solids),
        omitted_count=selection.omitted_count,
        omitted_body_indices=[
            i for i in range(len(solids)) if i not in selection.included_body_indices
        ],
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "revision_id": result["revision_id"],
        "engine_fingerprint": result["engine_fingerprint"],
        "step_sha256": file_sha256(directory / "part.step"),
        "document_sha256": file_sha256(directory / "document.json")
        if (directory / "document.json").is_file()
        else None,
        "primary": {"file": "render.png", "sha256": file_sha256(directory / "render.png")},
        "inspection": None,
        "selection": details,
        "render": {
            "linear_deflection": 0.15,
            "angular_deflection": 0.35,
            "frame_bounds_mm": np.asarray(frame_bounds).tolist(),
            "shared_framing": True,
        },
        "scope": SCOPE,
    }
    image = directory / "render-inspection.png"
    intermediate = directory / "preview-inspection.stl"
    metadata = directory / "render-inspection.json"
    for name in (image.name, metadata.name):
        result["artifact_hashes"].pop(name, None)

    def cleanup(path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            receipt.setdefault("cleanup_errors", []).append(f"{path.name}: {type(exc).__name__}")
            return False

    # A locked stale image is never registered or reused as new evidence. Optional
    # display-file cleanup must not turn a valid geometry result into a failure.
    cleared = cleanup(image)
    if selection.status == "reference_excluded" and cleared:
        try:
            if not cleanup(intermediate):
                raise OSError("Previous optional tessellation could not be cleared")
            from OCP.BRepTools import BRepTools

            # OCCT otherwise reuses the fine deliverable mesh. All authoritative
            # exports and measurements are already finished; clear only cached
            # triangulation so the companion matches the fresh full-view mesh.
            BRepTools.Clean_s(shape.wrapped)
            filtered = Compound(children=[solids[i] for i in selection.included_body_indices])
            if not export_stl(filtered, str(intermediate), tolerance=0.15, angular_tolerance=0.35):
                raise ValueError("Optional inspection tessellation export failed")
            render_workspace_preview(
                intermediate,
                image,
                frame_bounds=frame_bounds,
                inspection_label=f"Reference-excluded inspection · {selection.omitted_count} bodies hidden",
            )
            receipt["inspection"] = {"file": image.name, "sha256": file_sha256(image)}
        except Exception as exc:
            cleanup(image)
            receipt["render_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        finally:
            cleanup(intermediate)
    elif not cleared:
        receipt["render_error"] = (
            "Previous optional image could not be cleared; complete view retained."
        )
    try:
        metadata.write_text(json.dumps(receipt, allow_nan=False), encoding="utf-8")
        result["artifact_hashes"][metadata.name] = file_sha256(metadata)
        if receipt["inspection"]:
            result["artifact_hashes"][image.name] = receipt["inspection"]["sha256"]
    except OSError as exc:
        receipt["metadata_error"] = (
            f"{type(exc).__name__}: optional view metadata could not be published"
        )
        receipt["inspection"] = None
    result["facts"]["render_inspection"] = receipt
    return receipt
