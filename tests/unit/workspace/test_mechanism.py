from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.mechanism import (
    preserved_intent,
    resolve_mechanism,
    solve_mechanism,
    study_mechanism,
)
from forge_mind.workspace.mechanism_models import MechanismTemplate
from forge_mind.workspace.mechanism_templates import template_document


def design(kind: str = "slider_crank") -> DesignDocument:
    return template_document(MechanismTemplate.model_validate({"kind": kind}))


def graph(document: DesignDocument) -> dict:
    assert document.mechanism
    return resolve_mechanism(document.mechanism, document.values())


@pytest.mark.parametrize("kind", ["gearbox", "slider_crank", "four_bar"])
def test_templates_have_one_driven_dof_and_solve_full_sampled_range(kind: str) -> None:
    result = study_mechanism(graph(design(kind)))
    assert result["status"] == "sampled"
    assert len(result["samples"]) == 37
    assert result["current"]["undriven_local_dof"] == 1
    assert all(s["driven_local_dof"] == 0 for s in result["samples"])


def test_slider_travel_matches_independent_closed_form() -> None:
    result = study_mechanism(graph(design()))
    for sample in result["samples"]:
        angle = math.radians(sample["angle_deg"])
        expected = 20 * math.cos(angle) + math.sqrt(65**2 - (20 * math.sin(angle)) ** 2)
        assert sample["poses"]["slider"][0] == pytest.approx(expected, abs=1e-6)
        assert sample["poses"]["slider"][1] == pytest.approx(0, abs=1e-6)


def test_gearbox_shared_variables_keep_centers_and_ratio_consistent() -> None:
    doc = design("gearbox")
    doc.variables.update(module=2, wheel_teeth=40, face_width=10)
    rebuilt = DesignDocument.model_validate(doc.model_dump())
    data = graph(rebuilt)
    result = study_mechanism(data)
    assert result["status"] == "sampled"
    assert rebuilt.values()["center_distance"] == 58
    assert rebuilt.values()["housing_width"] == 214
    assert result["samples"][-1]["poses"]["gear_4"][2] == pytest.approx(360 / (40 / 18) ** 2)


def test_free_body_is_not_solved_by_a_driver_elsewhere() -> None:
    data = graph(design())
    data["joints"] = [j for j in data["joints"] if j["id"] != "rod_slider_pin"]
    result = solve_mechanism(data, 15)
    assert result["status"] == "singular_or_free"
    assert result["driven_local_dof"] > 0


def test_conflicting_joint_and_drive_cannot_earn_solved() -> None:
    data = graph(design())
    data["joints"].append(
        {
            "id": "locked_crank",
            "kind": "fixed",
            "a": "frame",
            "b": "crank",
            "pa": [0, 0],
            "pb": [0, 0],
            "axis": 0,
            "angle": 0,
        }
    )
    result = study_mechanism(data)
    assert result["status"] == "incomplete"
    assert result["current"]["status"] == "unresolved"


def test_collinear_closed_loop_exposes_local_singularity() -> None:
    doc = template_document(
        MechanismTemplate(kind="four_bar", crank=20, rod=50, ground=50, rocker=20)
    )
    data = graph(doc)
    for c in data["components"]:
        c["angle"] = 0
        c["y"] = 0
        c["x"] = 20 if c["id"] == "rod" else 50 if c["id"] == "rocker" else 0
    result = solve_mechanism(data, 0)
    assert result["status"] == "singular_or_free"
    assert result["driven_local_dof"] == 1


@pytest.mark.parametrize("mutation", ["graph", "component", "joint", "general"])
def test_replacement_cannot_silently_remove_mechanism_intent(mutation: str) -> None:
    original = design("gearbox").model_dump()
    candidate = design("gearbox").model_dump()
    if mutation == "graph":
        candidate = design().model_dump()
    elif mutation == "general":
        candidate = {}
    else:
        candidate["mechanism"]["components" if mutation == "component" else "joints"].pop()
    assert not preserved_intent(original, candidate)[0]
    assert preserved_intent(original, original)[0]


def test_component_feature_ownership_is_unique_and_complete() -> None:
    data = design().model_dump()
    data["mechanism"]["components"][1]["feature"] = data["mechanism"]["components"][0]["feature"]
    with pytest.raises(ValidationError, match="unique"):
        DesignDocument.model_validate(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e200, "missing_dimension"])
def test_unbounded_or_unresolved_graph_values_are_rejected(value: object) -> None:
    data = design().model_dump()
    data["mechanism"]["components"][1]["x"] = value
    with pytest.raises((ValueError, ValidationError)):
        DesignDocument.model_validate(data)


def test_derived_variables_reject_cycle_and_zero_divisor() -> None:
    data = design("gearbox").model_dump()
    data["relations"][0]["a"] = "center_distance"
    with pytest.raises(ValidationError, match="ordered"):
        DesignDocument.model_validate(data)
    data = design("gearbox").model_dump()
    data["variables"]["pinion_teeth"] = 0
    with pytest.raises(ValidationError, match="division"):
        DesignDocument.model_validate(data)


@pytest.mark.parametrize("kind", ["slider_crank", "four_bar", "gearbox"])
def test_actual_step_roundtrip_binds_named_components_and_rejects_tampering(
    tmp_path: Path, kind: str
) -> None:
    from build123d import export_step, import_step

    from forge_mind.workspace.kernel import compile_document
    from forge_mind.workspace.mechanism_geometry import bind_exported
    from forge_mind.workspace.topology import topology_index

    doc = design(kind)
    shape, history = compile_document(doc, {})
    path = tmp_path / "mechanism.step"
    export_step(shape, str(path))
    imported = import_step(str(path))
    topology = topology_index(imported, assembly=True)
    result = bind_exported(history[-1], topology, "step-hash", "doc-hash", imported)
    assert len(result["bindings"]) == len(doc.outputs)
    assert len({b["body_index"] for b in result["bindings"]}) == len(doc.outputs)
    imported.children[0].label = "tampered"
    with pytest.raises(ValueError, match="assembly name"):
        bind_exported(history[-1], topology, "step-hash", "doc-hash", imported)


def test_housing_encloses_larger_pinion_and_tracks_later_edits() -> None:
    doc = template_document(MechanismTemplate(kind="gearbox", pinion_teeth=32, wheel_teeth=17))
    assert doc.values()["housing_height"] == 65
    doc.variables["wheel_teeth"] = 35
    assert doc.values()["housing_height"] == 69.5


def test_retained_ids_cannot_change_component_feature_ownership() -> None:
    original = design().model_dump()
    candidate = design().model_dump()
    candidate["mechanism"]["components"][1]["feature"] = "replacement"
    assert not preserved_intent(original, candidate)[0]


def test_actual_intersection_and_declared_overload_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import build123d as b

    from forge_mind.workspace.engineering_models import EngineeringInputs
    from forge_mind.workspace.mechanism_review import mechanism_review

    doc = design("gearbox")
    # A known positive physical intersection exercises the exact same B-rep instrument
    # as tooth contact. This deliberately overlaps two exported-like solids.
    pose = {"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]}
    motion = {
        "bindings": [
            {"component": "a", "feature": "shaft_1", "body_index": 0},
            {"component": "b", "feature": "gear_2", "body_index": 1},
        ],
        "current": {"poses": pose},
        "samples": [{"poses": pose, "status": "solved", "angle_deg": 0}],
        "graph": {"gears": [{"id": "collision", "a": "a", "b": "b"}]},
        "gear_layout": [],
        "status": "sampled",
        "requested_samples": 1,
        "range_deg": [0, 0],
        "method": "test fixture",
        "limitations": [],
    }
    shaft = next(f for f in doc.features if f.kind == "shaft")
    motion["bindings"][0]["feature"] = shaft.id
    inputs = EngineeringInputs.model_validate(
        {
            "mechanism_loads": [
                {
                    "id": "overload",
                    "component": "a",
                    "torque_n_mm": 1e6,
                    "allowable_shear_mpa": 10,
                    "safety_factor": 2,
                    "source": "Explicit test load",
                }
            ]
        }
    )
    motion["samples"][0]["residual_mm_equivalent"] = 0.0
    monkeypatch.setattr(
        "forge_mind.workspace.mechanism_review.contact_poses", lambda _: motion["samples"]
    )
    result = mechanism_review(
        b.Compound(children=[b.Box(10, 10, 10), b.Pos(5, 0, 0) * b.Box(10, 10, 10)]),
        doc,
        motion,
        inputs,
    )
    assert result["tooth_interference"][0]["samples"][0]["overlap_mm3"] == pytest.approx(500)
    assert result["tooth_interference"][0]["status"] == "fail"
    assert result["shaft_loads"][0]["nominal_margin"] < 1
    assert result["shaft_loads"][0]["status"] == result["status"] == "fail"
    motion["samples"][0]["status"] = "unresolved"
    missing = mechanism_review(
        b.Compound(children=[b.Box(10, 10, 10), b.Box(10, 10, 10)]),
        doc,
        motion,
        EngineeringInputs(),
    )
    assert missing["tooth_interference"][0]["coverage"] == "incomplete"
    assert missing["status"] == "unverified"


def test_bounded_revision_binds_motion_and_refuses_modified_artifact(tmp_path: Path) -> None:
    from forge_mind.validation_provenance import file_sha256
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.document_service import build_document
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Contract
    from forge_mind.workspace.store import Store, WorkspaceError

    store = Store(tmp_path)
    contract = Contract(requirements=[])
    sid = store.create("owner", "slider crank", contract.model_dump())["id"]
    service = WorkspaceService(store, Executor("trusted_local"))
    revision = build_document(
        service, sid, "owner", design(), contract, None, "bounded test", 120, lambda: False
    )
    measured = revision["measurement"]
    assert measured["geometry_valid"], measured["errors"]
    directory = store.directory(sid, revision["id"])
    motion = measured["facts"]["mechanism"]
    assert motion["step_sha256"] == file_sha256(directory / "part.step")
    assert motion["document_sha256"] == file_sha256(directory / "document.json")
    assert "motion.json" in measured["artifact_hashes"]
    assert service._load_revision(sid, revision["id"])
    # Edit the unaccepted first candidate through the real preview admission path.
    replacement = DesignDocument.model_validate(
        {
            "features": [
                {"id": "box", "name": "Box", "kind": "box", "params": {"x": 10, "y": 10, "z": 10}}
            ],
            "outputs": ["box"],
        }
    )
    store.enqueue(
        sid,
        "owner",
        {
            "preview_only": True,
            "revision_id": revision["id"],
            "document": replacement.model_dump(),
            "max_seconds": 60,
        },
    )
    job = store.claim()
    assert job
    service.run_job(job)
    proposal = store.get(sid, "owner")["proposal"]
    assert not proposal["eligible"]
    assert any(
        c["id"].startswith("mechanism_intent:") and c["status"] == "fail"
        for c in proposal["protection_checks"]
    )

    (directory / "motion.json").write_text("{}", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="stale evidence"):
        service._load_revision(sid, revision["id"])


def test_contact_angles_do_not_alias_default_pinion_tooth_pitch() -> None:
    from forge_mind.workspace.mechanism_review import contact_poses

    data = graph(design("gearbox"))
    motion = study_mechanism(data)
    motion["graph"] = data
    poses = contact_poses(motion)
    assert len(poses) == 7
    assert all(p["status"] == "solved" for p in poses)
    assert len({round(p["angle_deg"] % 20, 5) for p in poses}) >= 6
    assert poses[0]["angle_deg"] == 0
    assert poses[-1]["angle_deg"] == 360
