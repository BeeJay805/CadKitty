"""Draft pure report semantics. Integration freshness checks still required."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from forge_mind.workspace.native_collision_closure import (
    _evaluate_reference,
    reference_collision_context,
)


def example():
    motion = {"rig": {"drivers": [{"id": "open", "reference": 0.0}]}}
    report = {
        "reference_values": {"open": 0.0},
        "reference_completed": True,
        "reference_collisions": [],
        "completed_configurations": [{"open": 0.0}],
        "status": "fail",
        "collisions": [{"values": {"open": 90}, "a": "base", "b": "cap", "overlap_mm3": 8}],
        "coverage": "incomplete",
        "errors": ["Later dynamic pose incomplete"],
    }
    return motion, report, {"base": "make", "cap": "make", "bolt": "buy", "torch": "reference"}


def test_no_current_rig_preserves_existing_closure_without_reading_store():
    assert (
        reference_collision_context(None, "sid", "owner", {}, SimpleNamespace(motion_rig=None))[
            "status"
        ]
        == "not_applicable"
    )


def test_dynamic_failures_remain_visible_but_do_not_block_complete_reference():
    motion, report, roles = example()
    before = deepcopy(report)
    assert _evaluate_reference(report, motion, roles)["status"] == "checked"
    assert report == before


def test_made_reference_overlap_blocks_with_numeric_feedback():
    motion, report, roles = example()
    row = {
        "values": {"open": 0.0},
        "a": "base",
        "b": "cap",
        "overlap_mm3": 195.4,
        "components": [{"volume_mm3": 195.4, "bounds_mm": [[1, 2, 3], [4, 5, 6]]}],
    }
    report["reference_collisions"] = [row]
    result = _evaluate_reference(report, motion, roles)
    assert result["status"] == "needs_work"
    assert result["collisions"] == [row]


@pytest.mark.parametrize("other", ["bolt", "torch"])
def test_bought_and_reference_overlap_are_outside_this_gate(other):
    motion, report, roles = example()
    report["reference_collisions"] = [
        {"values": {"open": 0.0}, "a": "base", "b": other, "overlap_mm3": 195.4}
    ]
    assert _evaluate_reference(report, motion, roles)["status"] == "checked"


@pytest.mark.parametrize(
    "field", ["reference_values", "reference_completed", "reference_collisions"]
)
def test_old_or_partial_reference_reports_cannot_silently_clear(field):
    motion, report, roles = example()
    report.pop(field)
    assert _evaluate_reference(report, motion, roles)["status"] == "unverified"


def test_unspecified_role_cannot_clear_reference_pose():
    motion, report, roles = example()
    roles["cap"] = "unspecified"
    assert _evaluate_reference(report, motion, roles)["status"] == "unverified"


def test_same_definition_occurrences_remain_separate_physical_bodies():
    motion, report, _ = example()
    roles = {"cap_one": "make", "cap_two": "make"}
    report["reference_collisions"] = [
        {"values": {"open": 0.0}, "a": "cap_one", "b": "cap_two", "overlap_mm3": 1}
    ]
    assert _evaluate_reference(report, motion, roles)["status"] == "needs_work"


def saved_report(tmp_path):
    """Exercise artifact identity only; synthetic STEP bytes are not a geometry test."""
    from forge_mind.validation_provenance import file_sha256
    from forge_mind.workspace.document import DesignDocument, digest
    from forge_mind.workspace.measure import engine_fingerprint
    from forge_mind.workspace.motion_rig import bind_rig
    from forge_mind.workspace.store import Store

    store = Store(tmp_path)
    sid = store.create("owner", "test", {})["id"]
    rid = "a" * 32
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "body",
                        "name": "Body",
                        "production": {"role": "make"},
                        "features": [
                            {
                                "id": "stock",
                                "name": "Stock",
                                "kind": "box",
                                "params": {"x": 2, "y": 2, "z": 2},
                            }
                        ],
                        "output": "stock",
                    }
                ],
                "roots": [
                    {"id": "base", "definition": "body"},
                    {"id": "cap", "definition": "body", "position": [3, 0, 0]},
                ],
            },
            "motion_rig": {
                "drivers": [
                    {"id": "open", "name": "Opening", "unit": "mm", "minimum": 0, "maximum": 10}
                ],
                "groups": [
                    {
                        "id": "moving",
                        "name": "Moving",
                        "driver": "open",
                        "kind": "slide",
                        "occurrences": ["cap"],
                    }
                ],
            },
        }
    )
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    (directory / "part.step").write_bytes(b"Synthetic identity fixture; no geometry qualification")
    (directory / "document.json").write_text(document.model_dump_json(), encoding="utf-8")
    hashes = {name: file_sha256(directory / name) for name in ("part.step", "document.json")}
    rows = [
        {"path": p, "definition": "body", "body_index": i} for i, p in enumerate(("base", "cap"))
    ]
    motion = bind_rig(
        document,
        {"occurrences": rows},
        revision_id=rid,
        engine_fingerprint=engine_fingerprint(),
        step_sha256=hashes["part.step"],
        document_sha256=hashes["document.json"],
    )
    revision = {
        "id": rid,
        "parent_id": None,
        "contract": {},
        "document": document.model_dump(),
        "measurement": {
            "geometry_valid": True,
            "artifact_hashes": hashes,
            "facts": {"motion_rig": motion, "assembly": {"occurrences": rows}},
        },
    }
    store.add_revision(sid, revision)
    _, report, _ = example()
    settings = {"samples_per_driver": 7, "include_corners": True, "max_configurations": None}
    report.update(
        {
            key: motion[key]
            for key in ("revision_id", "engine_fingerprint", "step_sha256", "document_sha256")
        },
        settings=settings,
    )
    checkid = "b" * 32
    path = store.root / "motion-checks" / sid / checkid / "check.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    receipt = {
        "id": checkid,
        "revision_id": rid,
        "engine_fingerprint": motion["engine_fingerprint"],
        "key": digest({"motion": motion, "settings": settings}),
        "settings": settings,
        "sha256": file_sha256(path),
    }
    store.update(sid, motion_checks=[receipt])
    return SimpleNamespace(store=store), sid, revision, document, path, receipt


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "report_bytes",
        "step_bytes",
        "document_bytes",
        "engine",
        "key",
        "revision",
        "settings",
        "missing_role",
        "definition_mapping",
    ],
)
def test_current_reference_receipt_rejects_stale_or_changed_evidence(tmp_path, defect):
    service, sid, revision, document, path, receipt = saved_report(tmp_path)
    if defect == "report_bytes":
        path.write_text("{}", encoding="utf-8")
    elif defect in {"step_bytes", "document_bytes"}:
        name = "part.step" if defect == "step_bytes" else "document.json"
        (service.store.directory(sid, revision["id"]) / name).write_text(
            "Changed", encoding="utf-8"
        )
    elif defect in {"engine", "key", "revision"}:
        receipt[
            {"engine": "engine_fingerprint", "key": "key", "revision": "revision_id"}[defect]
        ] = "c" * 64
        service.store.update(sid, motion_checks=[receipt])
    elif defect == "settings":
        receipt["settings"] = {"samples_per_driver": 3}
        service.store.update(sid, motion_checks=[receipt])
    elif defect == "missing_role":
        document.assembly.parts[0].production.role = "unspecified"
    elif defect == "definition_mapping":
        revision["measurement"]["facts"]["assembly"]["occurrences"][0]["definition"] = "missing"
    result = reference_collision_context(service, sid, "owner", revision, document)
    assert result["status"] == ("checked" if defect is None else "unverified")


@pytest.mark.parametrize("defect", ["rig", "role", "occurrence"])
def test_admitted_rigid_review_cannot_be_discarded_or_reclassified(tmp_path, defect):
    service, sid, previous, document, _, _ = saved_report(tmp_path)
    revision = deepcopy(previous)
    revision.update(id="d" * 32, parent_id=previous["id"])
    if defect == "rig":
        document.motion_rig = None
    elif defect == "role":
        document.assembly.parts[0].production.role = "reference"
    else:
        document.assembly.roots.pop()
    result = reference_collision_context(service, sid, "owner", revision, document)
    assert result["status"] == "unverified"
    assert "Retain admitted" in result["reason"] or "Retain the admitted" in result["reason"]


@pytest.mark.parametrize("defect", ["role", "occurrence"])
def test_unfinished_intermediate_revision_cannot_erase_older_review(tmp_path, defect):
    from forge_mind.validation_provenance import file_sha256
    from forge_mind.workspace.native_collision_closure import _retained_review
    from forge_mind.workspace.store import WorkspaceError

    service, sid, first, document, _, _ = saved_report(tmp_path)
    intermediate = deepcopy(first)
    intermediate.update(id="e" * 32, parent_id=first["id"])
    if defect == "role":
        document.assembly.parts[0].production.role = "reference"
    else:
        document.assembly.roots.pop()
        intermediate["measurement"]["facts"]["assembly"]["occurrences"].pop()
    intermediate["document"] = document.model_dump()
    directory = service.store.directory(sid, intermediate["id"])
    directory.mkdir(parents=True)
    (directory / "document.json").write_text(document.model_dump_json(), encoding="utf-8")
    intermediate["measurement"]["artifact_hashes"]["document.json"] = file_sha256(
        directory / "document.json"
    )
    service.store.add_revision(sid, intermediate)
    current = deepcopy(intermediate)
    current.update(id="f" * 32, parent_id=intermediate["id"])
    with pytest.raises(WorkspaceError, match="Retain admitted"):
        _retained_review(service, sid, current, document)


def test_explicit_changed_requirements_start_a_new_review_boundary(tmp_path):
    from tests.unit.workspace.test_operation_authoring_flow import contract

    from forge_mind.workspace.native_collision_closure import _retained_review

    service, sid, previous, document, _, _ = saved_report(tmp_path)
    current = deepcopy(previous)
    current.update(id="d" * 32, parent_id=previous["id"], contract=contract().model_dump())
    document.motion_rig = None
    _retained_review(service, sid, current, document)


def test_guidance_counts_exported_occurrences_without_installing_unused_definitions(tmp_path):
    from forge_mind.workspace.delivery_guidance import _facts

    _, _, revision, document, _, _ = saved_report(tmp_path)
    unused = document.assembly.parts[0].model_copy(deep=True)
    unused.id = "unused_bolt"
    unused.production.role = "buy"
    document.assembly.parts.append(unused)
    revision["document"] = document.model_dump()
    facts = _facts(revision)
    counts = {part["part_id"]: part["exported_occurrence_quantity"] for part in facts["parts"]}
    assert counts == {"body": 2, "unused_bolt": 0}
