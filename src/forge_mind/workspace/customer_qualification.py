"""Explicit isolated customer-worker qualification, without model or physical claims."""

from __future__ import annotations

import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.engine_identity import engine_fingerprint
from forge_mind.workspace.models import Contract
from forge_mind.workspace.print_setup_models import PrintSetupRequest
from forge_mind.workspace.processing import copy_artifact

if TYPE_CHECKING:
    from forge_mind.workspace.executor import Executor


def qualify_customer_container(executor: Executor) -> dict[str, Any]:
    """Use actual exported geometry for positive and negative frozen proof controls."""
    if (
        executor.mode != "docker"
        or not executor.typed_container_verified
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", executor.typed_container_image)
    ):
        raise ValueError("Customer qualification requires an immutable qualified typed image")
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "plate",
                        "name": "Qualification plate",
                        "production": {"role": "make", "process": "additive", "deliver_stl": True},
                        "features": [
                            {
                                "id": "blank",
                                "name": "Blank",
                                "kind": "box",
                                "params": {"x": 30, "y": 20, "z": 6},
                            },
                            {
                                "id": "tool",
                                "name": "Bore tool",
                                "kind": "cylinder",
                                "params": {"radius": 2, "height": 8},
                            },
                            {
                                "id": "body",
                                "name": "Through bore",
                                "kind": "cut",
                                "inputs": ["blank", "tool"],
                            },
                        ],
                        "output": "body",
                    }
                ],
                "roots": [{"id": "mounted", "definition": "plate"}],
            },
        }
    )
    checks = []
    records = []
    with tempfile.TemporaryDirectory(prefix="forge-customer-qualification-") as temporary:
        root = Path(temporary)
        candidate = root / "candidate"
        candidate.mkdir()
        (candidate / "document.json").write_text(document.model_dump_json(), encoding="utf-8")
        (candidate / "program.py").write_text(
            "# Deterministic native qualification document\n", encoding="utf-8"
        )
        measured = executor.process_document(
            candidate,
            {
                "revision_id": "a" * 32,
                "contract": Contract().model_dump(),
                "document": document.model_dump(),
                "assets": {},
            },
            180,
            lambda: False,
        )
        if not measured["geometry_valid"] or measured["engine_fingerprint"] != engine_fingerprint():
            raise ValueError(
                "Customer worker geometry or source/runtime fingerprint differs from gateway"
            )
        checks.append("gateway_worker_engine_identity")
        step_hash = file_sha256(candidate / "definition_plate.step")
        for operation, protocol in (
            ("nominal_through_pattern", "nominal_through_hole_pattern_v1"),
            ("nominal_perforated_box", "nominal_perforated_box_v1"),
        ):
            request: dict[str, Any] = {
                "protocol": protocol,
                "subject_id": "plate",
                "part_id": "plate",
                "frame": "definition_local",
                "axis": "z",
                "count_policy": "exact",
                "holes": [
                    {
                        "center_mm": [0.0, 0.0],
                        "diameter_mm": 4.0,
                        "diameter_tolerance_mm": 0.0,
                        "position_tolerance_mm": 0.0,
                    }
                ],
            }
            if operation == "nominal_perforated_box":
                request.update(size_mm=[30.0, 20.0, 6.0], center_mm=[0.0, 0.0, 0.0])
            for expected in ("pass", "fail"):
                query = deepcopy(request)
                if expected == "fail":
                    query["holes"][0]["diameter_mm"] = 5.0
                query["criterion_hash"] = digest(query)
                directory = root / f"{operation}-{expected}"
                directory.mkdir()
                copy_artifact(candidate / "definition_plate.step", directory / "part.step")
                result = executor.process_observation(
                    directory, {"operation": operation, "request": query}, 120, lambda: False
                )
                if (
                    result.get("status") != expected
                    or result.get("method") != protocol
                    or result.get("criterion_hash") != query["criterion_hash"]
                    or (result.get("facts") or {}).get("step_sha256") != step_hash
                    or result.get("evidence_refs")
                    != [{"artifact": "part.step", "sha256": step_hash}]
                ):
                    raise ValueError(f"Isolated {operation} failed its exact {expected} control")
                checks.append(f"{operation}_{expected}")
                records.append({"operation": operation, "expected": expected, "result": result})
        product = root / "product"
        product.mkdir()
        result = executor.process_observation(
            product,
            {
                "operation": "product_tool",
                "source": str(candidate),
                "request": {
                    "tool": "hole_pattern",
                    "units": "mm",
                    "body": {"body_index": 0},
                    "axis": [0.0, 0.0, 1.0],
                    "holes": [{"center": [0.0, 0.0, 0.0], "diameter": 4.0}],
                },
            },
            120,
            lambda: False,
        )
        if result.get("status") != "pass":
            raise ValueError("Isolated product geometry observer failed its known hole pattern")
        checks.append("saved_product_geometry")
        setup = root / "print"
        setup.mkdir()
        for name in (
            "document.json",
            "definition-meshes.json",
            "definition_plate.step",
            "definition_plate.stl",
        ):
            copy_artifact(candidate / name, setup / name)
        print_request = PrintSetupRequest(
            operation="prepare",
            revision_id="a" * 32,
            part_ids=["plate"],
            artifact_hashes=measured["artifact_hashes"],
        )
        executor.run_typed_tool(
            setup,
            print_request.model_dump(mode="json"),
            module="forge_mind.workspace.print_setup",
            timeout=120,
            cancelled=lambda: False,
        )
        from forge_mind.workspace.print_setup_service import _worker_result

        result = _worker_result(setup, print_request)
        if result["status"] != "prepared" or set(result["files"]) != {"print_plate.stl"}:
            raise ValueError("Isolated customer print package failed exact mesh preparation")
        checks.append("measured_prepared_print_mesh")
        records.append({"operation": "print_prepare", "result": result})
    return {
        "image": executor.typed_container_image,
        "engine_fingerprint": engine_fingerprint(),
        "checks": checks,
        "observations": records,
        "scope": "Actual isolated native proof and print worker controls only. Authenticated full jobs, durable hosting, provider credentials, payment and physical manufacture require separate qualification.",
    }
