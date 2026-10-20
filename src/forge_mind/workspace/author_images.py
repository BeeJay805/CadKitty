"""Verified author image inputs and diagnostic submission records, never acceptance."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.preview_selection import select_preview_bodies
from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.store import WorkspaceError

_LOG = logging.getLogger(__name__)
INSPECTION_SCOPE = "Declared reference bodies hidden for inspection; complete view and all validation retain every body."


def with_customer_reference(
    store: Any, sid: str, owner: str, images: AuthorImages, *, multiple: bool
) -> AuthorImages:
    """Retain the exact source on every refinement; never replace a required view."""
    from PIL import Image, ImageDraw, ImageOps

    from forge_mind.workspace.customer_references import reference_image

    photo, binding = reference_image(store, sid, owner)
    components = (*images.images, photo)
    labels = (*images.labels, "Owner visual reference; dimensions remain unverified")
    if len(components) > 6 or sum(map(len, components)) > 64 * 1024 * 1024:
        raise WorkspaceError("Required author images exceed the bounded image envelope")
    source: dict[str, Any] = {
        "customer_reference": binding,
        "prior_source_binding": images.source_binding,
    }
    if multiple or len(components) == 1:
        return AuthorImages(
            components,
            labels,
            images.revision_id,
            "customer_reference_and_" + images.mode,
            images.supplemental_status,
            images.instrument,
            source,
        )
    # A single-image client sees every component on a deterministic labelled sheet.
    # Original component hashes stay in the receipt; sheet pixels are only a view.
    sheet = Image.new("RGB", (1024, 800 * len(components)), "white")
    draw = ImageDraw.Draw(sheet)
    records = []
    for index, (raw, label) in enumerate(zip(components, labels, strict=True)):
        with Image.open(io.BytesIO(raw)) as decoded:
            if decoded.width * decoded.height > 16_000_000:
                raise WorkspaceError("Author image exceeds the contact-sheet pixel bound")
            tile = ImageOps.contain(decoded.convert("RGB"), (1024, 760))
        sheet.paste(tile, ((1024 - tile.width) // 2, index * 800 + 40))
        draw.text((8, index * 800 + 8), f"{index + 1}. {label}", fill="black")
        records.append(
            {
                "index": index + 1,
                "label": label,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    output = io.BytesIO()
    sheet.save(output, format="PNG", optimize=False)
    if output.tell() > 32 * 1024 * 1024:
        raise WorkspaceError("Required contact sheet exceeds the bounded image envelope")
    source["contact_sheet_components"] = records
    return AuthorImages(
        (output.getvalue(),),
        ("Labelled source and required CAD views",),
        images.revision_id,
        "customer_reference_contact_sheet",
        images.supplemental_status,
        images.instrument,
        source,
    )


@dataclass(frozen=True)
class AuthorImages:
    images: tuple[bytes, ...]
    labels: tuple[str, ...]
    revision_id: str | None
    mode: str
    supplemental_status: str
    instrument: dict[str, Any] | None = None
    source_binding: dict[str, Any] | None = None

    def records(self) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "label": label,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
            for index, (data, label) in enumerate(zip(self.images, self.labels, strict=True), 1)
        ]

    def context(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "mode": self.mode,
            "images": self.records(),
            "supplemental_status": self.supplemental_status,
            "instrument": self.instrument,
            "source_binding": self.source_binding,
            "scope": "Images are visual observations; numeric checks and reference geometry remain unchanged.",
        }


def _registered_bytes(directory: Path, name: str, hashes: dict[str, Any]) -> bytes:
    path = directory / name
    if not regular_file(path) or path.resolve().parent != directory.resolve():
        raise ValueError("Artifact path is not an owned revision file")
    data = path.read_bytes()
    if not data or hashlib.sha256(data).hexdigest() != hashes.get(name):
        raise ValueError("Artifact bytes do not match the revision")
    return data


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate metadata field")
        result[key] = value
    return result


def _inspection(
    directory: Path, revision: dict[str, Any], hashes: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    metadata = json.loads(
        _registered_bytes(directory, "render-inspection.json", hashes),
        object_pairs_hook=_unique_object,
    )
    measured = revision["measurement"]
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
        or metadata.get("revision_id") != revision["id"]
        or metadata.get("engine_fingerprint") != measured["engine_fingerprint"]
        or metadata.get("scope") != INSPECTION_SCOPE
        or metadata.get("primary") != {"file": "render.png", "sha256": hashes.get("render.png")}
        or metadata.get("inspection")
        != {"file": "render-inspection.png", "sha256": hashes.get("render-inspection.png")}
    ):
        raise ValueError("Inspection metadata does not bind this revision and its images")
    for field, name in (("step_sha256", "part.step"), ("document_sha256", "document.json")):
        path = directory / name
        expected = hashes.get(name)
        if (
            not isinstance(expected, str)
            or metadata.get(field) != expected
            or not regular_file(path)
            or path.resolve().parent != directory.resolve()
            or file_sha256(path) != expected
        ):
            raise ValueError("Inspection source binding is stale")
    document = json.loads(_registered_bytes(directory, "document.json", hashes))
    if document != revision.get("document"):
        raise ValueError("Inspection document differs from the stored revision")
    facts = measured["facts"]
    count = facts["solid_count"]
    selection = select_preview_bodies(document, facts.get("assembly"), count)
    included = set(selection.included_body_indices)
    recorded = metadata.get("selection")
    if (
        selection.status != "reference_excluded"
        or not isinstance(recorded, dict)
        or recorded.get("status") != selection.status
        or type(recorded.get("body_count")) is not int
        or recorded["body_count"] != count
        or recorded.get("included_body_indices") != list(selection.included_body_indices)
        or recorded.get("omitted_body_indices")
        != [index for index in range(count) if index not in included]
        or recorded.get("omitted_occurrence_ids") != list(selection.omitted_occurrence_ids)
        or type(recorded.get("omitted_count")) is not int
        or recorded["omitted_count"] != selection.omitted_count
    ):
        raise ValueError("Inspection selection differs from the verified body/role mapping")
    for key in ("included_body_indices", "omitted_body_indices"):
        if any(type(index) is not int for index in recorded[key]):
            raise ValueError("Invalid inspection body index")
    instrument = metadata.get("render")
    if (
        not isinstance(instrument, dict)
        or instrument.get("linear_deflection") != 0.15
        or instrument.get("angular_deflection") != 0.35
        or instrument.get("shared_framing") is not True
    ):
        raise ValueError("Unsupported inspection render protocol")
    bounds = instrument.get("frame_bounds_mm")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(not isinstance(row, list) or len(row) != 3 for row in bounds)
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for row in bounds
            for value in row
        )
        or any(lo > hi for lo, hi in zip(*bounds, strict=True))
    ):
        raise ValueError("Invalid inspection framing")
    return _registered_bytes(directory, "render-inspection.png", hashes), {
        key: instrument[key]
        for key in ("linear_deflection", "angular_deflection", "frame_bounds_mm", "shared_framing")
    }


def prepare_author_images(
    directory: Path | None,
    revision: dict[str, Any] | None,
    *,
    section: Path | None = None,
    section_revision_id: str | None = None,
    section_instrument: dict[str, Any] | None = None,
    multiple: bool = False,
) -> AuthorImages:
    rid = revision.get("id") if revision else None
    if section is not None:
        if not regular_file(section):
            raise WorkspaceError("Requested section image is unavailable")
        section_bytes = section.read_bytes()
        if not section_bytes:
            raise WorkspaceError("Requested section image is empty")
        return AuthorImages(
            (section_bytes,),
            ("Requested section view",),
            section_revision_id or rid,
            "section",
            "not_requested",
            section_instrument,
        )
    if directory is None or revision is None:
        return AuthorImages((), (), rid, "none", "no_revision")
    hashes = revision["measurement"]["artifact_hashes"]
    if "render.png" not in hashes:
        return AuthorImages((), (), rid, "none", "no_registered_primary")
    try:
        primary = _registered_bytes(directory, "render.png", hashes)
    except (OSError, ValueError) as exc:
        raise WorkspaceError(
            "Primary render changed or is missing; image evidence is stale"
        ) from exc
    base = ((primary,), ("Complete view: every body",), rid, "complete")
    if not multiple:
        return AuthorImages(*base, "provider_single_image_only")
    try:
        inspection, instrument = _inspection(directory, revision, hashes)
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return AuthorImages(*base, "inspection_unavailable_or_stale")
    return AuthorImages(
        (primary, inspection),
        ("Complete view: every body", INSPECTION_SCOPE),
        rid,
        "complete_and_inspection",
        "reference_excluded",
        instrument,
    )


def _write_receipt(path: Path, record: dict[str, Any], *, create: bool) -> None:
    encoded = json.dumps(record, allow_nan=False, sort_keys=True)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        return
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def invoke_author_images(
    client: Any,
    images: AuthorImages,
    arguments: dict[str, Any],
    *,
    observation_root: Path,
    event: Callable[..., Any],
) -> Any:
    """Call once. Record submission diagnostics without claiming model receipt."""
    if not images.images:
        raise ValueError("An image observation requires prepared image bytes")
    invocation = uuid.uuid4().hex
    path = observation_root / f"{invocation}.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "invocation_id": invocation,
        "revision_id": images.revision_id,
        "images": images.records(),
        "mode": images.mode,
        "supplemental_status": images.supplemental_status,
        "instrument": images.instrument,
        "source_binding": images.source_binding,
        "prepared_at": time.time(),
        "response_status": "unknown",
        "submission_status": "prepared",
        "scope": "Input submission diagnostics only; not proof of model receipt or understanding.",
    }

    def diagnostic(message: str) -> None:
        _LOG.warning(message)
        try:
            event("image_observation_diagnostic", message, invocation_id=invocation)
        except Exception:
            _LOG.warning("Could not persist the image observation diagnostic event")

    created = False
    try:
        _write_receipt(path, record, create=True)
        created = True
    except OSError:
        diagnostic(
            "Image observation receipt could not be created; submission tracking is incomplete"
        )
    record["submission_status"] = "submitted_to_client"
    try:
        if len(images.images) > 1:
            result = client.generate_structured_with_images(
                **arguments, images_png=images.images, image_labels=images.labels
            )
        else:
            result = client.generate_structured_with_image(**arguments, image_png=images.images[0])
        record["response_status"] = "complete"
        return result
    except BaseException as exc:
        record["response_status"] = "failed_or_unknown"
        record["failure_type"] = type(exc).__name__
        raise
    finally:
        record["finished_at"] = time.time()
        try:
            launched = getattr(client, "last_input_images", None)
            if isinstance(launched, (list, tuple)) and launched:
                expected = images.records()
                if len(launched) == len(expected) and all(
                    isinstance(actual, dict)
                    and all(
                        actual.get(key) == requested[key] for key in ("index", "sha256", "bytes")
                    )
                    for actual, requested in zip(launched, expected, strict=True)
                ):
                    record["submission_status"] = "submitted_to_cli"
                else:
                    diagnostic(
                        "CLI image submission metadata did not match the prepared observation"
                    )
        except Exception:
            diagnostic("CLI image submission metadata is unavailable")
        if created:
            try:
                _write_receipt(path, record, create=False)
            except OSError:
                diagnostic(
                    "Image observation receipt could not be finalized; its outcome remains unknown"
                )
        try:
            event("image_observation", "Author image input observation", **record)
        except Exception:
            _LOG.warning("Could not persist the image observation event")
