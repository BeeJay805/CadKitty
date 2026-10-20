"""Root-owned native qualification for principal-plane geometry and saved evidence."""

import math

import build123d as b
import pytest

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.assembly import assembly_evidence
from forge_mind.workspace.assembly_review import assembly_motion_review
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.mechanism_geometry import bind_exported
from forge_mind.workspace.mechanism_models import MechanismTemplate
from forge_mind.workspace.mechanism_review import mechanism_review
from forge_mind.workspace.mechanism_templates import template_document
from forge_mind.workspace.nominal_boolean import checked_boolean
from forge_mind.workspace.topology import topology_index

pytestmark = pytest.mark.slow
FRAMES = ("planar_xy_mm_deg", "planar_yz_mm_deg", "planar_zx_mm_deg")


def principal_plane(frame):
    return {FRAMES[0]: b.Plane.XY, FRAMES[1]: b.Plane.YZ, FRAMES[2]: b.Plane.ZX}[frame]


@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("kind", ["slider_crank", "four_bar"])
def test_saved_step_placement_and_component_identity_are_rotation_covariant(tmp_path, frame, kind):
    baseline = template_document(MechanismTemplate(kind=kind))
    original, _ = compile_document(baseline, {})
    document = DesignDocument.model_validate(baseline.model_dump())
    document.mechanism.frame = frame
    produced, history = compile_document(document, {})
    expected = principal_plane(frame) * original
    path = tmp_path / "mechanism.step"
    b.export_step(produced, path)
    imported = b.import_step(path)
    assert imported.is_valid and len(imported.solids()) == len(original.solids())
    assert imported.volume == pytest.approx(expected.volume, abs=1e-5)
    assert tuple(imported.bounding_box().min) == pytest.approx(
        tuple(expected.bounding_box().min), abs=1e-5
    )
    assert tuple(imported.bounding_box().max) == pytest.approx(
        tuple(expected.bounding_box().max), abs=1e-5
    )
    assert checked_boolean(imported, expected, "cut").volume <= 1e-5
    assert checked_boolean(expected, imported, "cut").volume <= 1e-5
    topology = topology_index(imported, assembly=True)
    bound = bind_exported(
        history[-1], topology, file_sha256(path), document.fingerprint(), imported
    )
    assert bound["graph"]["frame"] == frame
    assert len({row["body_index"] for row in bound["bindings"]}) == len(document.outputs)
    imported.children[0].label = "unowned_label"
    with pytest.raises(ValueError, match="assembly name"):
        bind_exported(history[-1], topology, file_sha256(path), document.fingerprint(), imported)


def pin_assembly(frame):
    return DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "sleeve",
                        "name": "Sleeve",
                        "output": "ring",
                        "features": [
                            {
                                "id": "outer",
                                "name": "Outer",
                                "kind": "cylinder",
                                "params": {"radius": 3, "height": 2},
                            },
                            {
                                "id": "bore",
                                "name": "Bore",
                                "kind": "cylinder",
                                "params": {"radius": 1.2, "height": 4},
                            },
                            {
                                "id": "ring",
                                "name": "Ring",
                                "kind": "cut",
                                "inputs": ["outer", "bore"],
                            },
                        ],
                        "ports": [{"name": "seat", "kind": "bore", "diameter": 2.4}],
                    },
                    {
                        "id": "shaft",
                        "name": "Shaft",
                        "output": "pin",
                        "features": [
                            {
                                "id": "pin",
                                "name": "Pin",
                                "kind": "cylinder",
                                "params": {"radius": 1, "height": 2},
                            }
                        ],
                        "ports": [{"name": "seat", "kind": "shaft", "diameter": 2}],
                    },
                ],
                "subassemblies": [
                    {
                        "id": "holder",
                        "name": "Offset holder",
                        "children": [
                            {
                                "id": "nested_sleeve",
                                "definition": "sleeve",
                                "position": [2, 1, 0],
                                "rotation": [0, 0, 25],
                            }
                        ],
                    }
                ],
                "roots": [
                    {"id": "ground_root", "definition": "holder"},
                    {"id": "shaft_root", "definition": "shaft"},
                ],
                "mates": [
                    {
                        "id": "pin_fit",
                        "a": "ground_root/nested_sleeve.seat",
                        "b": "shaft_root.seat",
                        "min_clearance": 0.3,
                        "max_clearance": 0.5,
                    }
                ],
            },
            "mechanism": {
                "frame": frame,
                "components": [
                    {
                        "id": "ground",
                        "name": "Ground",
                        "feature": "ground_root",
                        "grounded": True,
                        "x": 10,
                        "y": 20,
                        "z": 5,
                        "angle": 40,
                    },
                    {
                        "id": "pin",
                        "name": "Pin",
                        "feature": "shaft_root",
                        "x": 12,
                        "y": 21,
                        "z": 5,
                        "angle": 30,
                    },
                ],
                "joints": [
                    {
                        "id": "hinge",
                        "kind": "pin",
                        "a": "ground",
                        "b": "pin",
                        "pa": [2, 1],
                        "pb": [0, 0],
                    }
                ],
                "driver": {"component": "pin", "angle": 30, "start": 0, "end": 180, "samples": 3},
            },
        }
    )


@pytest.mark.parametrize("frame", FRAMES)
def test_nested_assembly_ports_and_mates_use_the_same_world_frame(tmp_path, frame):
    document = pin_assembly(frame)
    source, history = compile_document(document, {})
    path = tmp_path / "assembly.step"
    b.export_step(source, path)
    imported = b.import_step(path)
    evidence = assembly_evidence(document, imported, history[-1])
    assert all(row["status"] == "pass" for row in evidence["ports"].values()), evidence
    assert evidence["mates"][0]["status"] == "pass", evidence
    angle = math.radians(40)
    x, y, z = (
        10 + 2 * math.cos(angle) - math.sin(angle),
        20 + 2 * math.sin(angle) + math.cos(angle),
        5,
    )
    expected_origin = {FRAMES[0]: [x, y, z], FRAMES[1]: [z, x, y], FRAMES[2]: [y, z, x]}[frame]
    expected_axis = {FRAMES[0]: [0, 0, 1], FRAMES[1]: [1, 0, 0], FRAMES[2]: [0, 1, 0]}[frame]
    for port in evidence["ports"].values():
        assert port["origin"] == pytest.approx(expected_origin, abs=1e-5)
        assert port["axis"] == pytest.approx(expected_axis, abs=1e-6)
    document.assembly.parts[0].ports[0].origin = (100, 0, 0)
    wrong = assembly_evidence(document, imported, history[-1])
    assert wrong["ports"]["ground_root/nested_sleeve.seat"]["status"] == "fail"
    assert wrong["mates"][0]["status"] == "fail"


@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("collision", [False, True])
def test_saved_body_review_deltas_preserve_collision_witnesses(frame, collision, monkeypatch):
    rest = {"a": [0, 0, 30, 3], "b": [5, 0, 120, 3]}
    sample = {"a": [5 if collision else 10, 0, 120, 3], "b": rest["b"]}
    poses = [{"status": "solved", "angle_deg": 90, "poses": sample, "residual_mm_equivalent": 0}]
    # Independent build123d principal-plane placement, not the new matrix helper.
    solids = [
        principal_plane(frame)
        * b.Pos(pose[0], pose[1], pose[3])
        * b.Rot(0, 0, pose[2])
        * b.Box(2, 2, 2)
        for pose in rest.values()
    ]
    shape = b.Compound(children=solids)
    motion = {
        "current": {"status": "solved", "angle_deg": 0, "poses": rest},
        "samples": poses,
        "graph": {
            "frame": frame,
            "components": [{"id": "a", "grounded": False}, {"id": "b", "grounded": True}],
            "gears": [{"id": "pair", "a": "a", "b": "b"}],
        },
        "bindings": [
            {"component": "a", "feature": "gear_1", "body_index": 0},
            {"component": "b", "feature": "gear_2", "body_index": 1},
        ],
        "gear_layout": [],
        "status": "sampled",
        "requested_samples": 1,
        "range_deg": [0, 90],
        "method": "controlled saved poses",
        "limitations": [],
    }
    monkeypatch.setattr("forge_mind.workspace.mechanism_review.contact_poses", lambda _: poses)
    monkeypatch.setattr("forge_mind.workspace.assembly_review.contact_poses", lambda _: poses)
    document = template_document(MechanismTemplate(kind="gearbox"))
    document.mechanism.frame = frame
    gear_result = mechanism_review(shape, document, motion, EngineeringInputs())
    assembly_result = assembly_motion_review(
        shape,
        motion,
        {"occurrences": [{"body_index": 0, "path": "a"}, {"body_index": 1, "path": "b"}]},
    )
    for result in (gear_result, assembly_result):
        assert result["status"] == ("fail" if collision else "unverified")
    assert assembly_result["coverage"] == "discrete_only" and not assembly_result["errors"]
    overlap = gear_result["tooth_interference"][0]["samples"][0]["overlap_mm3"]
    assert overlap == pytest.approx(8 if collision else 0, abs=1e-6)
    if collision:
        assert assembly_result["collisions"][0]["overlap_mm3"] == pytest.approx(8, abs=1e-6)
