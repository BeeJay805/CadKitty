"""Explicit optional readback of the saved Astra clamp; root serializes native execution.

Set FORGE_PRINT_SETUP_CLAMP_FIXTURE to the exact saved revision directory. Without
that fixture, this control is skipped and cannot qualify the real clamp geometry.
This is a deterministic saved-artifact readback, not a fresh generation trial.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from forge_mind.workspace.definition_meshes import MANIFEST_NAME
from forge_mind.workspace.print_setup import run_print_setup, sha
from forge_mind.workspace.print_setup_models import PrintSetupRequest


def test_saved_astra_clamp_has_two_measured_oriented_definition_meshes(tmp_path):
    configured = os.environ.get("FORGE_PRINT_SETUP_CLAMP_FIXTURE")
    if not configured:
        pytest.skip(
            "Exact saved Astra clamp fixture was not supplied; no native qualification claimed"
        )
    source = Path(configured)
    before = {name: (source / name).read_bytes() for name in ("document.json", MANIFEST_NAME)}
    manifest = json.loads(before[MANIFEST_NAME])
    assert set(manifest["parts"]) == {"rear_half", "front_half"}
    for part_id in sorted(manifest["parts"]):
        for suffix in ("step", "stl"):
            name = f"definition_{part_id}.{suffix}"
            before[name] = (source / name).read_bytes()
    for name, data in before.items():
        (tmp_path / name).write_bytes(data)
    request = PrintSetupRequest(
        operation="prepare",
        revision_id=manifest["revision_id"],
        part_ids=["front_half", "rear_half"],
        artifact_hashes={name: sha(data) for name, data in before.items()},
    )
    result = run_print_setup(tmp_path, request)
    assert result["status"] == "prepared"
    assert set(result["files"]) == {"print_front_half.stl", "print_rear_half.stl"}
    assert result["physical_readiness"] == "unverified"
    # Existing measured contact witnesses must remain available among candidates.
    # Automatic ranking may choose another valid face; this is not a hidden pose requirement.
    for part_id, axis, coordinate in (("rear_half", 1, 0.0), ("front_half", 2, -11.0)):
        candidates = result["parts"][part_id]["candidates"]
        assert any(
            candidate["status"] == "measured"
            and candidate["plane"]["point_mm"][axis] == pytest.approx(coordinate, abs=1e-7)
            and abs(candidate["plane"]["outward_normal"][axis]) == pytest.approx(1)
            for candidate in candidates
        )
    for name, metadata in result["files"].items():
        delivered = (tmp_path / name).read_bytes()
        assert metadata["sha256"] == sha(delivered) and metadata["bytes"] == len(delivered)
        assert metadata["triangle_inventory_preserved"] is True
        assert metadata["derived_mesh"]["watertight"] is True
        assert metadata["derived_mesh"]["body_count"] == 1
    assert before == {name: (source / name).read_bytes() for name in before}
