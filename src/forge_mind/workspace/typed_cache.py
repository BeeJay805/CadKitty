"""Host admission and bounded staging for qualified typed-worker part caches.

This module never imports a native CAD parser. Cache bytes remain untrusted input
to the isolated worker, where the existing geometry checks and exports run again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_mind.workspace.models import Measurement
from forge_mind.workspace.processing import HOSTED_LIMITS, MAX_JOB_BYTES, directory_size

_HEX = re.compile(r"[a-f0-9]{64}")
_IMAGE = re.compile(r"sha256:[a-f0-9]{64}")
_REVISION = re.compile(r"[a-f0-9]{32}")
_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,63}")
_MANIFEST_BYTES = 1024 * 1024
_MAX_PARTS = 4096


@dataclass(frozen=True)
class TypedCacheInput:
    """Internal host descriptor. Never serialize this into the worker request."""

    source: Path
    revision_id: str
    image: str
    manifest_hash: str
    artifact_hashes: tuple[tuple[str, str], ...]


def cacheable_document(value: Any) -> bool:
    """Exclude an entire producer if any editable feature contains opaque Python."""
    # Do not invoke DesignDocument validation here: assembly validation imports
    # native geometry code. The caller owns document validation; admission only
    # makes this conservative, native-free exclusion over its saved JSON.
    if not isinstance(value, dict) or not isinstance(value.get("assembly"), dict):
        return False
    parts = value["assembly"].get("parts")
    features = value.get("features", [])
    if not isinstance(parts, list) or not parts or not isinstance(features, list):
        return False
    feature_groups = [features]
    has_features = bool(features)
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("features"), list):
            return False
        feature_groups.append(part["features"])
        has_features = has_features or bool(part["features"])
    return has_features and all(
        isinstance(feature, dict)
        and isinstance(feature.get("kind"), str)
        and feature["kind"] != "python"
        for group in feature_groups
        for feature in group
    )


def execution_origin(image: str, measured: Measurement, document: Any) -> dict[str, Any]:
    """Called only by the host after a successful qualified Docker result transfer."""
    if (
        _IMAGE.fullmatch(image) is None
        or not measured.geometry_valid
        or measured.errors
        or "parts.json" not in measured.artifact_hashes
        or not cacheable_document(document)
    ):
        return {}
    return {"version": 1, "image": image, "engine": measured.engine_fingerprint}


def parent_cache_input(
    revision: dict[str, Any], source: Path, directory: Path, image: str
) -> TypedCacheInput | None:
    """Admit only an owned, verified parent's original qualified Docker origin.

    The caller first loads the parent through WorkspaceService._load_revision.
    Missing historical origin, another image, and opaque producers are cold builds.
    Once admitted, staging refuses changed bytes rather than hiding corruption.
    """
    origin = revision.get("typed_cache_origin")
    if (
        _IMAGE.fullmatch(image) is None
        or not isinstance(origin, dict)
        or origin.get("version") != 1
        or origin.get("image") != image
    ):
        return None
    measured = Measurement.model_validate(revision["measurement"])
    if (
        not measured.geometry_valid
        or measured.errors
        or origin.get("engine") != measured.engine_fingerprint
        or "parts.json" not in measured.artifact_hashes
        or not cacheable_document(revision.get("document"))
    ):
        return None
    rid = revision["id"]
    if not isinstance(rid, str) or _REVISION.fullmatch(rid) is None:
        raise ValueError("Invalid typed cache parent identity")
    _owned_siblings(source, directory, rid)
    # These are optimization bounds, not design limits. A verified larger parent
    # takes the unchanged cold path before any descriptor is admitted. Reserve
    # its observed output footprint plus one maximum-sized artifact, because
    # cached BREP inputs coexist with the newly materialized output BREP files.
    if len(revision["document"]["assembly"]["parts"]) > _MAX_PARTS:
        return None
    sizes: dict[str, int] = {}
    for name in measured.artifact_hashes:
        if Path(name).name != name:
            raise ValueError("Invalid typed cache parent artifact name")
        value = _not_reparse(source / name)
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise ValueError("Typed cache parent artifact is not a regular single-link file")
        sizes[name] = value.st_size
    cache_bytes = sizes["parts.json"] + sum(
        size for name, size in sizes.items() if re.fullmatch(r"part_[a-f0-9]{32}\.brep", name)
    )
    if (
        sizes["parts.json"] > _MANIFEST_BYTES
        or sum(sizes.values()) + cache_bytes + HOSTED_LIMITS["file_bytes"] > MAX_JOB_BYTES
    ):
        return None
    return TypedCacheInput(
        source=source,
        revision_id=rid,
        image=image,
        manifest_hash=measured.artifact_hashes["parts.json"],
        artifact_hashes=tuple(measured.artifact_hashes.items()),
    )


def _not_reparse(path: Path) -> os.stat_result:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
        raise ValueError("Typed cache paths must not be linked or reparse points")
    return value


def _owned_siblings(source: Path, directory: Path, rid: str) -> None:
    # Revision directories share the exact session revisions directory. Reject
    # linked ancestors too, including junctions that resolve back under the Store.
    for path in (source, directory):
        for ancestor in (path, *path.parents):
            _not_reparse(ancestor)
    if (
        source.name != rid
        or _REVISION.fullmatch(directory.name) is None
        or source == directory
        or source.resolve().parent != directory.resolve().parent
        or source.parent.name != "revisions"
    ):
        raise ValueError("Typed cache must belong to the same session parent revision")


def _check(deadline: float, cancelled: Callable[[], bool]) -> None:
    if cancelled() or time.monotonic() >= deadline:
        raise ValueError("Operation cancelled or deadline exhausted while staging cache")


def _copy(
    source: Path,
    target: Path,
    expected: str,
    maximum: int,
    deadline: float,
    cancelled: Callable[[], bool],
) -> int:
    """Read a bounded ordinary file once; verify both open identity and bytes."""
    _check(deadline, cancelled)
    before = _not_reparse(source)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum
        or _HEX.fullmatch(expected) is None
    ):
        raise ValueError("Typed cache file is missing, linked, oversized, or invalid")
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as reader, target.open("xb") as writer:
        opened = os.fstat(reader.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) or opened.st_nlink != 1:
            raise ValueError("Typed cache file changed before staging")
        while block := reader.read(min(1024 * 1024, maximum - count + 1)):
            _check(deadline, cancelled)
            count += len(block)
            if count > maximum:
                raise ValueError("Typed cache staged input exceeds its byte limit")
            digest.update(block)
            writer.write(block)
        after = _not_reparse(source)
        final = os.fstat(reader.fileno())
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(opened) != _file_identity(final)
            or digest.hexdigest() != expected
        ):
            raise ValueError("Typed cache file changed or does not match parent evidence")
    staged = _not_reparse(target)
    if not stat.S_ISREG(staged.st_mode) or staged.st_nlink != 1 or staged.st_size != count:
        raise ValueError("Typed cache staging did not produce a bounded regular file")
    _check(deadline, cancelled)
    return count


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Typed cache manifest contains duplicate keys")
        result[key] = value
    return result


def stage_cache(
    admitted: TypedCacheInput | None,
    directory: Path,
    scratch: Path,
    image: str,
    deadline: float,
    cancelled: Callable[[], bool],
) -> str | None:
    """Stage only admitted manifest references; expose no host paths to Docker."""
    if admitted is None:
        return None
    if admitted.image != image or _IMAGE.fullmatch(image) is None:
        raise ValueError("Typed cache image changed after admission")
    _owned_siblings(admitted.source, directory, admitted.revision_id)
    _check(deadline, cancelled)
    remaining = MAX_JOB_BYTES - directory_size(scratch, MAX_JOB_BYTES)
    target = scratch / "cache"
    target.mkdir()
    remaining -= _copy(
        admitted.source / "parts.json",
        target / "parts.json",
        admitted.manifest_hash,
        min(_MANIFEST_BYTES, remaining),
        deadline,
        cancelled,
    )
    manifest = json.loads(
        (target / "parts.json").read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    if not isinstance(manifest, dict) or not 0 < len(manifest) <= _MAX_PARTS:
        raise ValueError("Typed cache manifest must contain bounded part entries")
    hashes = dict(admitted.artifact_hashes)
    files: dict[str, tuple[str, str]] = {}
    for part, row in manifest.items():
        _check(deadline, cancelled)
        if (
            _NAME.fullmatch(part) is None
            or not isinstance(row, dict)
            or set(row) != {"key", "file", "sha256", "step"}
            or not isinstance(row["key"], str)
            or _HEX.fullmatch(row["key"]) is None
            or row["file"] != "part_" + row["key"][:32] + ".brep"
            or row["step"] != "definition_" + part + ".step"
            or not isinstance(row["sha256"], str)
            or _HEX.fullmatch(row["sha256"]) is None
            or hashes.get(row["file"]) != row["sha256"]
        ):
            raise ValueError("Typed cache manifest does not match parent artifact evidence")
        identity = (row["key"], row["sha256"])
        if row["file"] in files and files[row["file"]] != identity:
            raise ValueError("Typed cache manifest contains a filename collision")
        files[row["file"]] = identity
    for name, (_, checksum) in files.items():
        remaining -= _copy(
            admitted.source / name,
            target / name,
            checksum,
            min(HOSTED_LIMITS["file_bytes"], remaining),
            deadline,
            cancelled,
        )
    _owned_siblings(admitted.source, directory, admitted.revision_id)
    _check(deadline, cancelled)
    return "/work/cache"
