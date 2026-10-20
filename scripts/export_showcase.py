"""Export reviewed public assets from an exact captured ZIP, without rebuilding CAD.

Run from the repository root with ``python -m scripts.export_showcase --help``.
Operator metadata is a declaration, not proof of rights or successful generation.
Review source and instruction content before publishing the resulting directory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.customer_outcome_bench import sha, strict_json, verified_archive

from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.project_delivery import MAX_PACKAGE_BYTES, validate_project_export
from forge_mind.workspace.project_members import captured_cad_files

_ORIGINS = {"actual_model_generated", "saved_partial", "maintainer_authored"}
_FIELDS = {"case_id", "title", "prompt", "model", "limits", "origin", "rights_owned_or_authorized"}


def _read(path: Path, maximum: int) -> bytes:
    if not regular_file(path, maximum):
        raise ValueError("Showcase input must be a bounded regular file without links")
    with path.open("rb") as stream:
        content = stream.read(maximum + 1)
    if not content or len(content) > maximum:
        raise ValueError("Empty or oversized showcase input")
    return content


def _metadata(content: bytes) -> dict[str, Any]:
    value = strict_json(content)
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("Supply only the explicit public case metadata fields")
    if (
        not isinstance(value["origin"], str)
        or value["origin"] not in _ORIGINS
        or value["rights_owned_or_authorized"] is not True
    ):
        raise ValueError("Declare the input origin and confirm publication authorization")
    if not isinstance(value["case_id"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{0,63}", value["case_id"]
    ):
        raise ValueError("Invalid public case identifier")
    for key, maximum in (("title", 160), ("prompt", 16000), ("model", 100)):
        if not isinstance(value[key], str) or not 0 < len(value[key].strip()) <= maximum:
            raise ValueError("Public title, prompt and model must be bounded explicit text")
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != {"seconds", "actions"}:
        raise ValueError("Declare exact seconds and actions limits")
    if any(item is not None and (type(item) is not int or item <= 0) for item in limits.values()):
        raise ValueError("Limits must be positive integers or explicit null")
    return value


def _guide(content: bytes) -> bytes:
    value = strict_json(content)
    record = value.get("record") or {}
    guidance = record.get("guidance") or {}
    lines = [
        "# Recorded instructions",
        "",
        "These are recorded author instructions, not physical qualification or acceptance.",
        "Separate print-package paths may refer to files outside this showcase.",
        "",
    ]
    for field, title in (
        ("print", "Print"),
        ("installation", "Installation"),
        ("use_and_removal", "Use and removal"),
        ("unresolved", "Unresolved"),
    ):
        text = guidance.get(field)
        if isinstance(text, str) and text.strip():
            lines.extend(["## " + title, "", text, ""])
    if not any(isinstance(guidance.get(key), str) and guidance[key].strip() for key in guidance):
        lines.extend(["No reviewed instruction record is included in this capture.", ""])
    return "\n".join(lines).encode("utf-8")


def export_showcase(
    package: Path, receipt_path: Path, metadata_path: Path, output: Path
) -> dict[str, Any]:
    """Validate everything before creating a new directory; never write to inputs."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Showcase output must be a new directory")
    metadata = _metadata(_read(metadata_path, 64 * 1024))
    receipt = strict_json(_read(receipt_path, 4 * 1024 * 1024))
    content = _read(package, MAX_PACKAGE_BYTES)
    # The member reader rejects links, traversal, duplicates and decompression excess
    # before any member is read into the public export. The shared verifier checks
    # the captured context; the harness then verifies every manifest member hash.
    catalog = captured_cad_files(content, receipt)
    validate_project_export(package, receipt)
    members = verified_archive(content, receipt)
    files = {row["archive_path"]: members[row["archive_path"]] for row in catalog}
    files["document.json"] = members["document.json"]
    guide_source = members.get("customer-guidance.json")
    if guide_source is not None:
        files["GUIDE.md"] = _guide(guide_source)
    readme = [
        "# " + metadata["title"],
        "",
        "Input origin: `" + metadata["origin"] + "` (operator-declared).",
        "Model: `" + metadata["model"] + "` (operator-declared).",
        "",
        "## Original public prompt",
        "",
        metadata["prompt"],
        "",
        "## Scope and limits",
        "",
        "Physical readiness is unverified. Full-task success is unassessed (null).",
        "A saved partial result is not a completed digital delivery.",
        "The rights flag records the operator's declaration; it does not establish third-party rights.",
        "Source is the exact captured document.json. Independent regeneration is not verified.",
        "CAD frames and geometry bytes remain unchanged. No GLB, image or missing format is synthesized.",
        "Recorded guide text can refer to a separate prepared print package that is not included.",
        "Review source and instruction content for private information before publication.",
        "",
        "Declared limits: " + json.dumps(metadata["limits"], sort_keys=True) + ".",
        "",
        "## Files",
        "",
        *[f"- [{name}]({name})" for name in sorted(files)],
        "",
        "manifest.json records exact output hashes and the verified original package checksum.",
        "The original ZIP, private receipt/context and raw history are not redistributed.",
        "",
    ]
    files["README.md"] = "\n".join(readme).encode("utf-8")
    if len({name.casefold() for name in files}) != len(files):
        raise ValueError("Public file names must remain unique on case-insensitive filesystems")
    records: dict[str, Any] = {}
    for name, raw in sorted(files.items()):
        records[name] = {"sha256": sha(raw), "bytes": len(raw)}
        if name in members and raw == members[name]:
            records[name]["source_archive_path"] = name
        elif name == "GUIDE.md" and guide_source is not None:
            records[name].update(
                source_archive_path="customer-guidance.json",
                source_member_sha256=sha(guide_source),
                projection="Only the four recorded instruction strings; no basis, facts or history",
            )
    manifest = {
        "schema_version": 1,
        "case": metadata,
        "package": {"sha256": sha(content), "bytes": len(content)},
        "physical_readiness": "unverified",
        "full_task_success": None,
        "files": records,
    }
    # No replacement, cleanup deletion or Store mutation. A filesystem failure
    # leaves a partial directory and raises; the manifest is written last.
    output.mkdir()
    for name, raw in sorted(files.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(
            (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "receipt", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = export_showcase(args.package, args.receipt, args.metadata, args.output)
    print(json.dumps({"case_id": manifest["case"]["case_id"], "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
