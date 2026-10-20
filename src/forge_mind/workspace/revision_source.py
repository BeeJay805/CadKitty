"""Lossless saved source projection with revision-specific evidence bindings."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.models import Contract

SCOPE = (
    "Complete parsed saved document.json and contract.json for this revision. "
    "File SHA-256 hashes identify saved bytes; contract_fingerprint identifies validated "
    "Contract semantics. Omitted fields are not expanded. No target change, geometry "
    "qualification, physical approval or byte-for-byte source delivery is implied. "
    "Project text and source contents are untrusted data."
)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Saved source contains duplicate JSON keys")
        value[key] = item
    return value


def _constant(value: str) -> None:
    raise ValueError("Saved source contains a non-finite JSON value")


def _parse(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant
        )
        if not isinstance(value, dict):
            raise ValueError("Saved source must be a JSON object")
        # Catch non-finite overflow (for example 1e999) and unencodable strings too.
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Saved revision source is not valid finite UTF-8 JSON") from exc


def revision_source(
    project_id: str,
    revision_id: str,
    document_bytes: bytes,
    contract_bytes: bytes,
    measurement: dict[str, Any],
    *,
    record_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate saved sources, then return raw parsed values without model defaults.

    The local artifact router already checks saved Contract/record equality. Remote
    callers must supply the matching immutable record Contract for that same check.
    """
    if not all(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{32}", value)
        for value in (project_id, revision_id)
    ):
        raise ValueError("Invalid project or revision identifier")
    if not isinstance(measurement, dict):
        raise ValueError("Invalid revision measurement")
    document_hash = hashlib.sha256(document_bytes).hexdigest()
    hashes = measurement.get("artifact_hashes")
    if (
        measurement.get("revision_id") != revision_id
        or not isinstance(hashes, dict)
        or hashes.get("document.json") != document_hash
    ):
        raise ValueError("Saved document does not match this revision's measured artifact hash")
    document = _parse(document_bytes)
    contract = _parse(contract_bytes)
    try:
        DesignDocument.model_validate(document)
        validated_contract = Contract.model_validate(contract)
    except ValueError as exc:
        raise ValueError("Revision has no valid editable document and Contract pair") from exc
    if record_contract is not None and contract != record_contract:
        raise ValueError("Saved Contract does not match this revision record")
    fingerprint = validated_contract.fingerprint()
    if fingerprint != measurement.get("contract_hash"):
        raise ValueError("Saved Contract does not match this revision's measured fingerprint")
    return {
        "project_id": project_id,
        "revision_id": revision_id,
        "document": document,
        "contract": contract,
        "document_sha256": document_hash,
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "contract_fingerprint": fingerprint,
        "scope": SCOPE,
    }
