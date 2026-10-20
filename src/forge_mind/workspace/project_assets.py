"""Bounded owned import inventory, without content or native geometry verification."""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Any

from forge_mind.workspace.store import Store, WorkspaceError

SCOPE = (
    "Stored owned STEP metadata only; availability and size are filesystem observations, "
    "not current hash verification, native validity, units or interface measurements. "
    "Use submit_document with an import feature and reference production role, then inspect "
    "the measured revision. Building rechecks exact owned input bytes."
)


def _size(path: Path) -> int | None:
    try:
        for parent in path.parents:
            metadata = parent.lstat()
            if stat.S_ISLNK(metadata.st_mode) or getattr(
                metadata, "st_file_attributes", 0
            ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                return None
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or getattr(
            metadata, "st_file_attributes", 0
        ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            return None
        return metadata.st_size
    except OSError:
        return None


def list_project_assets(store: Store, project_id: str, owner: str) -> dict[str, Any]:
    """List stored IDs and cheap file status; never hash, parse or start a job."""
    if not isinstance(project_id, str) or not re.fullmatch(r"[a-f0-9]{32}", project_id):
        raise WorkspaceError("Invalid project identifier")
    session = store.session_record(project_id, owner)
    assets = session.get("assets", [])
    if not isinstance(assets, list) or len(assets) > 100:
        raise WorkspaceError("Imported asset inventory is invalid or exceeds its limit")
    seen: set[str] = set()
    rows = []
    for asset in assets:
        if (
            not isinstance(asset, dict)
            or not isinstance(asset.get("id"), str)
            or not re.fullmatch(r"[a-f0-9]{32}", asset["id"])
            or asset["id"] in seen
            or not isinstance(asset.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", asset["sha256"])
            or not isinstance(asset.get("name"), str)
        ):
            raise WorkspaceError("Imported asset metadata is invalid")
        seen.add(asset["id"])
        size = _size(store.root / "assets" / project_id / (asset["id"] + ".step"))
        rows.append(
            {
                "asset_id": asset["id"],
                "name": asset["name"][:160],
                "sha256": asset["sha256"],
                "size_bytes": size,
                "file_status": "available" if size is not None else "unavailable",
                "measurement_status": "not_measured_by_inventory",
            }
        )
    store.session_record(project_id, owner)
    return {
        "project_id": project_id,
        "assets": rows,
        "content_verification": "stored_metadata_only",
        "scope": SCOPE,
    }
