from pathlib import Path

import build123d as b
import pytest

from forge_mind.workspace.authoring import AssemblyEdit, apply_assembly_edit, authoring_schema
from forge_mind.workspace.document import Feature, Occurrence, PartDefinition
from forge_mind.workspace.hardware import bill_of_materials, hardware_shape
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.measure import measure_revision, merge_contract_edit
from forge_mind.workspace.models import Contract, Requirement


def initial():
    return apply_assembly_edit(
        None,
        AssemblyEdit(
            parts=[
                PartDefinition(
                    id="base",
                    name="Base",
                    features=[
                        Feature(
                            id="body", name="Body", kind="box", params={"x": 10, "y": 8, "z": 4}
                        )
                    ],
                    output="body",
                )
            ],
            roots=[Occurrence(id="base", definition="base")],
            variables={"travel": 0},
        ),
    )


def test_incremental_assembly_preserves_source_and_reuses_definition():
    doc = initial()
    before = doc.model_dump()
    added = apply_assembly_edit(
        doc, AssemblyEdit(roots=[Occurrence(id="copy", definition="base", position=(20, 0, 0))])
    )
    assert doc.model_dump() == before
    assert len(added.assembly.parts) == 1 and len(added.assembly.roots) == 2
    moved = apply_assembly_edit(
        added, AssemblyEdit(roots=[Occurrence(id="copy", definition="base", position=(30, 0, 0))])
    )
    assert moved.assembly.roots[1].position == (30, 0, 0)
    with pytest.raises(ValueError):
        apply_assembly_edit(doc, AssemblyEdit(roots=[Occurrence(id="bad", definition="missing")]))
    assert doc.model_dump() == before


def test_schema_is_authoritative_and_discoverable():
    assert "assembly_edit" in authoring_schema("")["available"]
    schema = authoring_schema("assembly_edit")["schema"]
    assert schema["additionalProperties"] is False
    assert "PartDefinition" in schema["$defs"]


def test_touching_assembly_mesh_is_checked_per_body_without_hiding_export_changes(tmp_path: Path):
    import trimesh

    shape = b.Compound(
        children=[b.Box(10, 10, 10), b.Box(10, 10, 10).moved(b.Location((10, 0, 0)))]
    )
    b.export_step(shape, tmp_path / "part.step")
    b.export_stl(shape, tmp_path / "part.stl", tolerance=0.01, angular_tolerance=0.07)
    assert not measure_revision(tmp_path, "single", Contract()).geometry_valid
    measured = measure_revision(tmp_path, "assembly", Contract(), structured_assembly=True)
    assert measured.geometry_valid
    assert measured.facts["watertight"] is False
    assert measured.facts["body_mesh_evidence"]["triangle_inventory_matches"]
    mesh = trimesh.load(tmp_path / "part.stl", force="mesh")
    original = mesh.faces.copy()
    for changed in [original[:-1], original[::-1, ::-1], original[[*range(len(original)), 0]]]:
        mesh.faces = changed
        mesh.export(tmp_path / "part.stl")
        assert not measure_revision(
            tmp_path, "changed", Contract(), structured_assembly=True
        ).geometry_valid


def test_triangle_inventory_preserves_orientation_multiplicity_and_exact_coordinates():
    import numpy as np

    from forge_mind.workspace.mesh_evidence import triangle_inventory

    triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    original = triangle_inventory(triangle.copy())
    assert original == triangle_inventory(np.roll(triangle, 1, axis=1))
    assert original != triangle_inventory(triangle[:, ::-1])
    assert original != triangle_inventory(np.concatenate([triangle, triangle]))
    shifted = triangle.copy()
    shifted[0, 0, 0] = 0.000001
    assert original != triangle_inventory(shifted)


def test_named_part_dimension_cannot_borrow_whole_assembly_extent(tmp_path: Path):
    doc = apply_assembly_edit(
        initial(),
        AssemblyEdit(roots=[Occurrence(id="copy", definition="base", position=(20, 0, 0))]),
    )
    shape, _ = compile_document(doc, {}, tmp_path)
    b.export_step(shape, tmp_path / "part.step")
    b.export_stl(shape, tmp_path / "part.stl")
    c = Contract(
        requirements=[
            Requirement(
                id="base_width",
                text="base 30",
                source_quote="base 30",
                source="explicit",
                part_id="base",
                metric="extent_x",
                value=30,
            ),
            Requirement(
                id="missing",
                text="missing 30",
                source_quote="missing 30",
                source="explicit",
                part_id="missing",
                metric="extent_x",
                value=30,
            ),
        ]
    )
    measured = measure_revision(tmp_path, "a" * 32, c)
    assert measured.facts["extents_mm"][0] == pytest.approx(30)
    assert measured.findings[0].status == "fail" and measured.findings[0].measured == pytest.approx(
        10
    )
    assert measured.findings[1].status == "unverified"
    assert measured.facts["definitions"]["base"]["frame"] == "definition_local"


def test_requirement_edit_does_not_confuse_different_parts():
    old = Contract(
        requirements=[
            Requirement(
                id="base_width",
                text="base width",
                source_quote="base 10",
                source="explicit",
                metric="extent_x",
                value=10,
                part_id="base",
            )
        ]
    )
    change = Contract(
        requirements=[
            Requirement(
                id="roller_width",
                text="roller width",
                source_quote="set roller 20",
                metric="extent_x",
                value=20,
                part_id="roller",
            )
        ]
    )
    result = merge_contract_edit(old, change, "set roller 20")
    assert len(result.requirements) == 2 and result.requirements[0].value == 10


def test_catalog_bearing_envelope_is_one_solid_and_explicitly_labeled():
    shape = hardware_shape("bearing_608", envelope=True)
    assert len(shape.solids()) == 1
    assert tuple(shape.bounding_box().size) == pytest.approx((22, 22, 7))
    assert pytest.approx(0) == shape.bounding_box().min.Z
    doc = initial()
    doc.assembly.parts[0].features = [
        Feature(
            id="body",
            name="Bearing envelope",
            kind="hardware",
            catalog_id="bearing_608",
            params={"envelope": 1},
        )
    ]
    compiled, _ = compile_document(doc, {})
    assert len(compiled.solids()) == 1
    assert "envelope" in bill_of_materials(doc)["lines"][0]["geometry_detail"]


def test_static_report_summary_retains_named_collision_witness():
    from forge_mind.workspace.parameter_study import summarize_report

    report = {
        "assembly": {
            "pose_status": "fail",
            "pairs": [{"a": 0, "b": 1, "status": "fail", "overlap_mm3": 5}],
        }
    }
    result = summarize_report(
        report,
        {
            "assembly": {
                "occurrences": [
                    {"body_index": 0, "path": "frame/base"},
                    {"body_index": 1, "path": "moving/slide"},
                ]
            }
        },
    )
    assert result["collisions"][0]["a"] == "frame/base"
    assert result["status"] == "fail"


def test_author_returns_latest_assembly_without_hiding_numeric_regression(tmp_path: Path):
    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Action
    from forge_mind.workspace.store import Store

    actions = iter(
        [
            Action(
                kind="document",
                explanation="Initial base",
                document_json=initial().model_dump_json(),
            ),
            Action(
                kind="assembly_edit",
                explanation="Add requested second body",
                document_json=AssemblyEdit(
                    roots=[Occurrence(id="slide", definition="base", position=(20, 0, 0))],
                ).model_dump_json(),
            ),
            Action(kind="finish", explanation="Both bodies present; envelope requirement fails"),
        ]
    )

    class Client:
        def generate_structured(self, **kwargs):
            return next(actions), LLMUsage("test", 0, 0)

        generate_structured_with_image = generate_structured

    store = Store(tmp_path)
    contract = Contract(
        requirements=[
            Requirement(
                id="width",
                text="10 mm overall width",
                source_quote="10 mm",
                source="explicit",
                metric="extent_x",
                value=10,
            )
        ]
    )
    sid = store.create("owner", "Two bodies within 10 mm overall width", contract.model_dump())[
        "id"
    ]
    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    store.enqueue(sid, "owner", {"max_actions": 3, "max_seconds": 120})
    service.run_job(store.claim())
    session = store.get(sid, "owner")
    assert session["state"] == "needs_review"
    assert session["selected_revision"] != session["best_revision"]
    selected = store.revision(sid, session["selected_revision"])
    assert len(selected["document"]["assembly"]["roots"]) == 2
    assert selected["measurement"]["findings"][0]["status"] == "fail"


def test_parameter_study_measures_actual_displacement_and_keeps_nominal_selection(tmp_path: Path):
    import time

    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.document_service import build_document
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.parameter_study import parameter_study
    from forge_mind.workspace.store import Store

    doc = initial()
    doc = apply_assembly_edit(
        doc,
        AssemblyEdit(roots=[Occurrence(id="slide", definition="base", position=("travel", 20, 0))]),
    )
    store = Store(tmp_path / "store")
    service = WorkspaceService(store, Executor("trusted_local"))
    sid = store.create("owner", "Test real exported displacement", Contract().model_dump())["id"]
    source = build_document(
        service, sid, "owner", doc, Contract(), None, "Source", 120, lambda: False
    )
    assert source["measurement"]["geometry_valid"]
    store.update(sid, selected_revision=source["id"])
    study = parameter_study(
        service,
        sid,
        "owner",
        source["id"],
        Contract(),
        "travel",
        [0, 10],
        time.monotonic() + 180,
        lambda: False,
    )
    assert study["coverage"] == "discrete_only"
    assert study["status"] == "unverified"
    rows = {r["path"]: r for r in study["samples"][1]["occurrences"]}
    assert rows["slide"]["delta_mm"] == pytest.approx([10, 0, 0])
    assert rows["base"]["delta_mm"] == pytest.approx([0, 0, 0])
    assert rows["slide"]["definition_unchanged"]
    assert store.get(sid, "owner")["selected_revision"] == source["id"]
    assert all(
        store.revision(sid, s["revision_id"])["parent_id"] == source["id"] for s in study["samples"]
    )
    from forge_mind.workspace.parameter_study import study_path
    from forge_mind.workspace.store import WorkspaceError

    assert study_path(service, sid, "owner", study["id"]).is_file()
    import hashlib
    import json
    from zipfile import ZipFile

    from forge_mind.workspace.engineering_service import release_package, run_report_job

    run_report_job(
        service,
        {
            "session": sid,
            "owner": "owner",
            "revision_id": source["id"],
            "max_seconds": 120,
        },
        lambda: False,
    )
    with ZipFile(release_package(service, sid, "owner", source["id"])) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert "studies/" + study["id"] + ".json" in manifest["files"]
        for sample in study["samples"]:
            assert "study-revisions/" + sample["revision_id"] + "/part.step" in manifest["files"]
            assert "study-reports/" + sample["report"]["id"] + "/report.json" in manifest["files"]
        assert all(
            hashlib.sha256(archive.read(name)).hexdigest() == digest
            for name, digest in manifest["files"].items()
        )
    cancelled = parameter_study(
        service,
        sid,
        "owner",
        source["id"],
        Contract(),
        "travel",
        [0, 10],
        time.monotonic() + 180,
        lambda: True,
    )
    assert cancelled["coverage"] == "incomplete"
    assert cancelled["status"] != "pass"
    assert len(cancelled["samples"]) == 2
    inputs = store.get(sid, "owner")["engineering"]
    store.save_engineering(sid, "owner", inputs["version"], inputs["inputs"])
    assert study_path(service, sid, "owner", study["id"]).is_file()
    run_report_job(
        service,
        {
            "session": sid,
            "owner": "owner",
            "revision_id": source["id"],
            "max_seconds": 120,
        },
        lambda: False,
    )
    with ZipFile(release_package(service, sid, "owner", source["id"])) as archive:
        assert not any(name.startswith("studies/") for name in archive.namelist())
    path = store.directory(sid, study["samples"][1]["revision_id"]) / "definition_base.step"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(WorkspaceError, match=r"changed|hash|artifact"):
        study_path(service, sid, "owner", study["id"])
