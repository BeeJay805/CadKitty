"""Service volumes must not appear to be manufactured fixture geometry."""

import csv

from forge_mind.workspace.drawings import write_drawings
from forge_mind.workspace.engineering_models import EngineeringInputs


def test_reference_outline_and_scene_bounds_are_separate_from_model(tmp_path):
    topology = {
        "entities": [
            {"kind": "body", "body_index": 0, "min": [0, 0, 0], "max": [20, 10, 4]},
            {"kind": "body", "body_index": 1, "min": [-20, -20, 0], "max": [40, 30, 100]},
            {"kind": "edge", "body_index": 0, "polyline": [[0, 0, 0], [20, 0, 0]]},
            {"kind": "edge", "body_index": 1, "polyline": [[-20, -20, 0], [40, -20, 0]]},
        ]
    }
    revision = {
        "id": "a" * 32,
        "contract": {
            "requirements": [{"id": "safe", "text": "\t=1+1", "value": None, "tolerance": None}]
        },
        "document": {
            "assembly": {
                "parts": [
                    {"id": "fixture", "production": {"role": "make"}},
                    {"id": "tool", "production": {"role": "reference"}},
                ]
            }
        },
        "measurement": {
            "findings": [],
            "facts": {
                "bbox_min": [-20, -20, 0],
                "bbox_max": [40, 30, 100],
                "assembly": {
                    "occurrences": [
                        {"definition": "fixture", "body_index": 0},
                        {"definition": "tool", "body_index": 1},
                    ]
                },
            },
        },
    }
    result = write_drawings(tmp_path, topology, revision, EngineeringInputs())
    assert result["reference_body_count"] == 1
    assert result["non_reference_bounds"] == {"min_mm": [0, 0, 0], "max_mm": [20, 10, 4]}
    svg = (tmp_path / "drawing.svg").read_text()
    assert 'stroke-dasharray="4 3"' in svg
    assert "Scene: 60 x 50 mm" in svg
    assert "Non-reference: 20 x 10 mm" in svg
    assert "Scene dimensions include these references" in svg
    with (tmp_path / "inspection.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["Requirement"] == "'\t=1+1"


def test_single_part_reference_is_not_drawn_as_a_manufactured_part(tmp_path):
    topology = {
        "entities": [
            {"kind": "body", "body_index": 0, "min": [0, 0, 0], "max": [20, 10, 4]},
            {"kind": "edge", "body_index": 0, "polyline": [[0, 0, 0], [20, 0, 0]]},
        ]
    }
    revision = {
        "id": "a" * 32,
        "contract": {"requirements": []},
        "document": {"production": {"role": "reference"}},
        "measurement": {"findings": [], "facts": {"bbox_min": [0, 0, 0], "bbox_max": [20, 10, 4]}},
    }
    result = write_drawings(tmp_path, topology, revision, EngineeringInputs())
    svg = (tmp_path / "drawing.svg").read_text()
    assert result["reference_body_count"] == 1
    assert result["non_reference_bounds"] is None
    assert "Dashed gray: reference" in svg
    assert "Scene: 20 x 10 mm" in svg and "Overall:" not in svg
    assert 'stroke-dasharray="4 3"' in svg
