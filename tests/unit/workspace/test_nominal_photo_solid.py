"""Pure expected-extrusion grammar; actual B-rep qualification belongs to root."""

from copy import deepcopy

import pytest

from forge_mind.workspace.nominal_photo_solid import PhotoSolidRequest, expected_profile_document
from forge_mind.workspace.photo_profile import _proposal


def request(count=1):
    loops = [
        {
            "role": "outer",
            "fit": {"kind": "polygon", "points_mm": [[0, 0], [42, 0], [42, 26], [0, 26], [0, 0]]},
        }
    ]
    if count:
        loops.append(
            {"role": "inner", "fit": {"kind": "circle", "center_mm": [12, 10], "radius_mm": 2}}
        )
    return PhotoSolidRequest.model_validate(
        {
            "protocol": "nominal_photo_solid_v1",
            "criterion_hash": "a" * 64,
            "subject_id": "panel",
            "part_id": "part",
            "frame": "definition_local",
            "target_sha256": "b" * 64,
            "expected_document": _proposal(loops, 4.0),
            "source_constraints": {
                "subject_id": "panel",
                "frame": "definition_local",
                "axis": "z",
                "size_mm": [42.0, 26.0, 4.0],
                "hole_count": count,
                "size_fact_ids": ["width", "height", "depth"],
                "hole_count_fact_id": "count",
            },
        }
    )


@pytest.mark.parametrize("count", [0, 1])
def test_complete_ordinary_extrusion_grammar_accepts_explicit_zero_and_positive(count):
    expected = request(count)
    before = deepcopy(expected.model_dump())
    document = expected_profile_document(expected)
    assert document.outputs and len(document.features) == 2 + 3 * count
    assert expected.model_dump() == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["features"][0]["sketch"].update(plane="XZ"),
        lambda d: d["features"][0]["sketch"].update(offset=1.0),
        lambda d: d["features"][1].update(params={"amount": -4.0}, bindings={}),
        lambda d: d["features"][-1].update(inputs=["loop_1_solid", "loop_0_solid"]),
        lambda d: d.update(outputs=["loop_0_solid"]),
        lambda d: d["features"][0]["sketch"]["points"][1].update(y=2.0),
        lambda d: d["features"].append(
            {"id": "hidden", "name": "hidden", "kind": "python", "code": "raise Exception()"}
        ),
        lambda d: d["variables"].update(thickness_mm=3.0),
    ],
)
def test_hidden_or_changed_expected_construction_cannot_become_authority(mutate):
    value = request()
    mutate(value.expected_document)
    with pytest.raises(ValueError):
        expected_profile_document(value)


def test_wrong_count_source_frame_and_missing_constraint_refuse():
    value = request()
    for change in [{"hole_count": 2}, {"subject_id": "other"}, {"axis": "x"}]:
        changed = value.model_copy(deep=True)
        changed.source_constraints.update(change)
        with pytest.raises(ValueError):
            expected_profile_document(changed)
    changed = value.model_copy(deep=True)
    changed.source_constraints.pop("hole_count_fact_id")
    with pytest.raises(ValueError):
        expected_profile_document(changed)
