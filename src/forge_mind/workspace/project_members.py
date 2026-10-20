"""Bounded CAD member views over an already verified captured project ZIP."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import stat
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from forge_mind.workspace.processing import MAX_FILE_BYTES, MAX_JOB_BYTES
from forge_mind.workspace.project_delivery import MAX_PACKAGE_BYTES
from forge_mind.workspace.store import WorkspaceError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkspaceError("Captured CAD files: " + message)


def _hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _safe_name(name: str) -> bool:
    return (
        bool(name)
        and not name.startswith("/")
        and not any(character in name for character in ("\\", ":", "\x00"))
        and all(component not in ("", ".", "..") for component in name.split("/"))
    )


def _json(data: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise WorkspaceError("Captured CAD files: nonfinite JSON value " + value)

    def finite_float(value: str) -> float:
        number = float(value)
        _require(math.isfinite(number), "nonfinite JSON number")
        return number

    value = json.loads(
        data, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float
    )
    _require(isinstance(value, dict), "metadata must be an object")
    assert isinstance(value, dict)
    return value


def _read(archive: ZipFile, item: ZipInfo) -> bytes:
    _require(0 <= item.file_size <= MAX_FILE_BYTES, "member exceeds its byte budget")
    with archive.open(item) as stream:
        data = stream.read(item.file_size + 1)
    _require(len(data) == item.file_size, "member size changed during read")
    return data


def _declaration(value: object, item: ZipInfo) -> dict[str, Any]:
    _require(isinstance(value, dict), "missing member declaration")
    assert isinstance(value, dict)
    _require(
        _hash(value.get("sha256"))
        and type(value.get("size_bytes")) is int
        and value["size_bytes"] == item.file_size,
        "member size or hash declaration is invalid",
    )
    return value


def _bound_metadata(
    archive: ZipFile, items: dict[str, ZipInfo], manifest: dict[str, Any], name: str
) -> dict[str, Any]:
    _require(name in items, "missing " + name)
    declared = _declaration(manifest["files"].get(name), items[name])
    data = _read(archive, items[name])
    _require(hashlib.sha256(data).hexdigest() == declared["sha256"], "changed " + name)
    return _json(data)


def _package_bytes(content: bytes, receipt: dict[str, Any]) -> None:
    _require(isinstance(content, bytes) and isinstance(receipt, dict), "invalid input types")
    _require(
        type(receipt.get("bytes")) is int
        and 0 < len(content) == receipt["bytes"] <= MAX_PACKAGE_BYTES
        and _hash(receipt.get("sha256"))
        and hashlib.sha256(content).hexdigest() == receipt["sha256"],
        "package length or checksum mismatch",
    )


def _inventory(
    receipt: dict[str, Any], archive: ZipFile, *, metadata: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, ZipInfo]]:
    _require(type(receipt.get("include_stl")) is bool, "invalid STL option")
    infos = archive.infolist()
    names = [item.filename for item in infos]
    _require(len(set(names)) == len(names), "duplicate archive member")
    _require(
        all(
            _safe_name(item.filename)
            and item.orig_filename == item.filename
            and not item.is_dir()
            and not item.flag_bits & 1
            and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
            and 0 <= item.file_size <= MAX_FILE_BYTES
            for item in infos
        ),
        "unsafe or oversized archive member",
    )
    _require(sum(item.file_size for item in infos) <= MAX_JOB_BYTES, "aggregate budget exceeded")
    items = {item.filename: item for item in infos}
    _require("manifest.json" in items, "missing manifest")
    manifest = _json(_read(archive, items["manifest.json"]))
    _require(isinstance(manifest.get("files"), dict), "missing manifest inventory")
    _require(set(manifest["files"]) == set(items) - {"manifest.json"}, "inventory mismatch")
    for name, declared in manifest["files"].items():
        _declaration(declared, items[name])
    _require(
        manifest.get("include_assembly_step") is True
        and type(manifest.get("include_stl", False)) is bool
        and manifest.get("include_stl", False) == receipt["include_stl"],
        "package options mismatch",
    )
    assembly = receipt.get("assembly")
    _require(
        isinstance(assembly, dict)
        and assembly == manifest.get("assembly")
        and assembly.get("file") == "assembly/part.step"
        and assembly.get("frame") == "cad_z_up_mm_world"
        and assembly.get("units") == "mm",
        "assembly declaration mismatch",
    )
    assert isinstance(assembly, dict)
    _require(
        assembly["file"] in items
        and assembly.get("sha256") == manifest["files"][assembly["file"]]["sha256"],
        "unbound assembly STEP",
    )
    document = _bound_metadata(archive, items, manifest, "document.json")
    production = _bound_metadata(archive, items, manifest, "production.json")
    if metadata is not None:
        metadata.update(manifest=manifest, document=document, production=production)
    structured = document.get("assembly")
    _require(structured is None or isinstance(structured, dict), "invalid source assembly")
    frame = "definition_local" if structured else "document_local"
    definitions: dict[str, str] = {}
    if structured:
        _require(isinstance(structured.get("parts"), list), "missing source definitions")
        for part in structured["parts"]:
            _require(isinstance(part, dict), "invalid source definition")
            part_id = part.get("id")
            _require(
                isinstance(part_id, str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", part_id) is not None
                and part_id not in definitions,
                "invalid or duplicate source definition",
            )
            setup = part.get("production") or {}
            _require(isinstance(setup, dict), "invalid source production")
            definitions[part_id] = setup.get("role", "unspecified")
    else:
        setup = document.get("production") or {}
        _require(isinstance(setup, dict), "invalid flat production")
        definitions["part"] = setup.get("role", "unspecified")
    _require(isinstance(production.get("make"), list), "missing make inventory")
    wanted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in production["make"]:
        _require(isinstance(row, dict), "invalid make inventory row")
        part_id, quantity = row.get("part_id"), row.get("quantity")
        _require(
            isinstance(part_id, str)
            and part_id not in seen
            and definitions.get(part_id) == "make"
            and row.get("role") == "make"
            and type(quantity) is int
            and quantity > 0,
            "invalid made definition, role or quantity",
        )
        seen.add(part_id)
        stem = "definition_" + part_id if structured else "part"
        wanted.append(
            {
                "part_id": part_id,
                "quantity": quantity,
                "frame": frame,
                "files": [f"make/{stem}.step"]
                + ([f"make/{stem}.stl"] if receipt["include_stl"] else []),
            }
        )
    received = receipt.get("made_parts")
    _require(
        isinstance(received, list)
        and all(
            isinstance(part, dict) and type(part.get("quantity")) is int and part["quantity"] > 0
            for part in received
        )
        and received == wanted,
        "made receipt differs from source inventory",
    )
    rows: list[dict[str, Any]] = [
        {
            "file_id": "assembly-step",
            "archive_path": assembly["file"],
            "format": "step",
            "part_id": None,
            "quantity": 1,
            "frame": assembly["frame"],
            "role": "assembly",
        }
    ]
    for part in wanted:
        for name in part["files"]:
            format_name = name.rsplit(".", 1)[1]
            rows.append(
                {
                    "file_id": f"part-{part['part_id']}-{format_name}",
                    "archive_path": name,
                    "format": format_name,
                    "part_id": part["part_id"],
                    "quantity": part["quantity"],
                    "frame": part["frame"],
                    "role": "make",
                }
            )
    allowed = {row["archive_path"] for row in rows}
    _require(
        {name for name in items if name.startswith(("make/", "assembly/"))} == allowed,
        "missing or unexpected CAD member",
    )
    _require(len({row["file_id"] for row in rows}) == len(rows), "duplicate file identifier")
    for row in rows:
        item = items[row["archive_path"]]
        declared = manifest["files"][item.filename]
        _require(item.file_size > 0, "empty CAD member")
        row.update(sha256=declared["sha256"], bytes=item.file_size)
    return rows, items


def captured_photo_context(content: bytes, package: dict[str, Any]) -> dict[str, Any]:
    """Read bounded captured source/production evidence, without native imports."""
    try:
        _package_bytes(content, package)
        metadata: dict[str, Any] = {}
        with ZipFile(io.BytesIO(content)) as archive:
            rows, items = _inventory(package, archive, metadata=metadata)
            assembly_row = next(row for row in rows if row["role"] == "assembly")
            _check_member(archive, items[assembly_row["archive_path"]], assembly_row)
        document, production = metadata["document"], metadata["production"]
        structured = isinstance(document.get("assembly"), dict)
        definitions = (
            document["assembly"]["parts"]
            if structured
            else [{"id": "part", "production": document.get("production")}]
        )
        roles = {
            part["id"]: (part.get("production") or {}).get("role", "unspecified")
            for part in definitions
        }
        lines = production.get("lines")
        _require(isinstance(lines, list), "missing production occurrence inventory")
        selected: list[dict[str, Any]] = []
        omitted: list[dict[str, Any]] = []
        seen_parts: set[str] = set()
        seen_paths: set[str] = set()
        counts: dict[str, int] = {}
        for line in lines:
            _require(isinstance(line, dict), "invalid production occurrence row")
            part_id, role, quantity = line.get("part_id"), line.get("role"), line.get("quantity")
            paths = line.get("occurrence_paths")
            legacy_buy = (
                roles.get(part_id) == "unspecified"
                and role == "buy"
                and line.get("role_basis") == "unchanged legacy catalog geometry"
                and isinstance(line.get("catalog_id"), str)
                and bool(line["catalog_id"])
            )
            _require(
                part_id in roles
                and part_id not in seen_parts
                and (role == roles[part_id] or legacy_buy)
                and role in ("make", "buy", "reference", "unspecified")
                and type(quantity) is int
                and quantity > 0
                and isinstance(paths, list)
                and len(paths) == quantity,
                "contradictory production occurrence identity or quantity",
            )
            seen_parts.add(part_id)
            for path in paths:
                _require(
                    isinstance(path, str)
                    and 0 < len(path) <= 4096
                    and all(
                        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", segment)
                        for segment in path.split("/")
                    )
                    and path not in seen_paths,
                    "invalid or duplicate occurrence path",
                )
                seen_paths.add(path)
                row = {"occurrence_path": path, "part_id": part_id, "role": role}
                (selected if role == "make" else omitted).append(row)
            if role == "make":
                counts[part_id] = quantity
        for role, key in (
            ("make", "make"),
            ("buy", "buy"),
            ("reference", "reference"),
            ("unspecified", "unresolved"),
        ):
            partition = production.get(key)
            _require(
                isinstance(partition, list)
                and all(
                    isinstance(row, dict) and type(row.get("quantity")) is int for row in partition
                )
                and partition == [row for row in lines if row["role"] == role],
                "production role partition differs from occurrence inventory",
            )
        expected = {row["part_id"]: row["quantity"] for row in package["made_parts"]}
        _require(counts == expected, "made quantities differ from occurrence inventory")
        for key in ("project_id", "revision_id", "package_id"):
            _require(
                isinstance(package.get(key), str)
                and re.fullmatch(r"[a-f0-9]{32}", package[key]) is not None,
                "invalid package identity",
            )
        files = metadata["manifest"]["files"]
        return {
            "version": 1,
            "project_id": package["project_id"],
            "revision_id": package["revision_id"],
            "package_id": package["package_id"],
            "package_sha256": package["sha256"],
            "structured_assembly": structured,
            "captured_document_sha256": files["document.json"]["sha256"],
            "captured_production_sha256": files["production.json"]["sha256"],
            "assembly_step_sha256": assembly_row["sha256"],
            "selected": sorted(selected, key=lambda row: row["occurrence_path"]),
            "omitted": sorted(omitted, key=lambda row: row["occurrence_path"]),
            "make_quantities": [
                {"part_id": key, "quantity": counts[key]} for key in sorted(counts)
            ],
        }
    except (
        BadZipFile,
        OSError,
        KeyError,
        TypeError,
        UnicodeError,
        RecursionError,
        NotImplementedError,
    ) as exc:
        raise WorkspaceError("Captured photo context: malformed archive or declaration") from exc
    except ValueError as exc:
        if isinstance(exc, WorkspaceError):
            raise
        raise WorkspaceError("Captured photo context: malformed metadata") from exc


def _checked_bytes(archive: ZipFile, item: ZipInfo, row: dict[str, Any]) -> bytes:
    data = _read(archive, item)
    _require(hashlib.sha256(data).hexdigest() == row["sha256"], "CAD member checksum mismatch")
    return data


def _check_member(archive: ZipFile, item: ZipInfo, row: dict[str, Any]) -> None:
    digest, count = hashlib.sha256(), 0
    with archive.open(item) as stream:
        while chunk := stream.read(min(1024 * 1024, item.file_size - count + 1)):
            count += len(chunk)
            _require(count <= item.file_size, "member exceeds declared length")
            digest.update(chunk)
    _require(
        count == item.file_size and digest.hexdigest() == row["sha256"],
        "CAD member checksum or length mismatch",
    )


def captured_cad_files(content: bytes, receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact CAD metadata after package validation; never extract or execute."""
    try:
        _package_bytes(content, receipt)
        with ZipFile(io.BytesIO(content)) as archive:
            rows, items = _inventory(receipt, archive)
            for row in rows:
                _check_member(archive, items[row["archive_path"]], row)
            return rows
    except (
        BadZipFile,
        OSError,
        KeyError,
        TypeError,
        UnicodeError,
        RecursionError,
        NotImplementedError,
    ) as exc:
        raise WorkspaceError("Captured CAD files: malformed archive or declaration") from exc
    except ValueError as exc:
        if isinstance(exc, WorkspaceError):
            raise
        raise WorkspaceError("Captured CAD files: malformed metadata") from exc


def read_captured_cad_file(content: bytes, receipt: dict[str, Any], file_id: str) -> bytes:
    """Read one allowed exact member from the same captured package; unknown IDs refuse."""
    try:
        _package_bytes(content, receipt)
        with ZipFile(io.BytesIO(content)) as archive:
            rows, items = _inventory(receipt, archive)
            matches = [row for row in rows if row["file_id"] == file_id]
            _require(isinstance(file_id, str) and len(matches) == 1, "unknown file identifier")
            row = matches[0]
            return _checked_bytes(archive, items[row["archive_path"]], row)
    except (
        BadZipFile,
        OSError,
        KeyError,
        TypeError,
        UnicodeError,
        RecursionError,
        NotImplementedError,
    ) as exc:
        raise WorkspaceError("Captured CAD files: malformed archive or declaration") from exc
    except ValueError as exc:
        if isinstance(exc, WorkspaceError):
            raise
        raise WorkspaceError("Captured CAD files: malformed metadata") from exc
