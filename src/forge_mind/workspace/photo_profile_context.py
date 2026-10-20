"""Compact model views of full immutable evidence; never a replacement measurement."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from forge_mind.workspace.photo_profile_service import _digest


def photo_profile_context(record: dict[str, Any], sid: str) -> dict[str, Any]:
    """Keep usable geometry and every topology entry without dense display samples."""
    result = record["result"]
    projected = {key: deepcopy(value) for key, value in record.items() if key != "result"}
    view = {
        key: deepcopy(value)
        for key, value in result.items()
        if key not in {"components", "overlay_geometry", "observation_sha256", "comparison_sha256"}
    }
    if "components" in result:
        inventory = []
        for component in result["components"]:
            item = {key: deepcopy(value) for key, value in component.items() if key != "loops"}
            loops = []
            selected = result.get("selected_component") in (None, component["id"])
            for loop in component["loops"]:
                entry = {
                    key: deepcopy(value)
                    for key, value in loop.items()
                    if key not in {"points_px", "points_mm", "pixel_outline", "fit"}
                }
                entry["raw_point_counts"] = {
                    key: len(loop[key]) for key in ("points_px", "points_mm") if key in loop
                }
                if selected:
                    for key in ("fit", "pixel_outline"):
                        if key in loop:
                            entry[key] = deepcopy(loop[key])
                else:
                    entry["fit_kind"] = (loop.get("fit") or {}).get("kind")
                    entry["pixel_outline_point_count"] = len(
                        (loop.get("pixel_outline") or {}).get("points_px", [])
                    )
                loops.append(entry)
            item["loops"] = loops
            inventory.append(item)
        view["components"] = inventory
    if "overlay_geometry" in result:
        overlay = result["overlay_geometry"]
        view["overlay_geometry"] = {
            key: deepcopy(value) for key, value in overlay.items() if key != "saved_boundary_points"
        }
        view["overlay_geometry"]["saved_boundary_points"] = [
            {
                **{key: deepcopy(value) for key, value in loop.items() if key != "points_px"},
                "point_count": len(loop.get("points_px", [])),
                "points_omitted": True,
            }
            for loop in overlay.get("saved_boundary_points", [])
        ]
    projected["result"] = view
    observation_id = record.get("id", record.get("observation_id"))
    request: dict[str, Any] = {"action": "read", "observation_id": observation_id}
    if "comparison_sha256" in result:
        request = {
            "action": "compare",
            "observation_id": observation_id,
            "revision_id": record["revision_id"],
            "selector": record["selector"],
        }
    projected["projection"] = {
        "kind": "compact_photo_profile_context_v1",
        "scope": "Model view only. Full-record/result hashes below reference complete evidence, not this projected JSON. Reduced pixel outlines are display/proposal inputs, never scoring masks.",
        "full_record_sha256": _digest(record),
        "full_result_sha256": _digest(result),
        "observation_sha256": result.get("observation_sha256"),
        "comparison_sha256": result.get("comparison_sha256"),
        "full_read": {
            "method": "POST",
            "path": f"/workspace/sessions/{sid}/photo-profile",
            "body": request,
            "note": "HTTP returns full evidence. Observation read is immutable; explicit compare reruns the fixed read-only observer for these exact identities.",
        },
    }
    return projected
