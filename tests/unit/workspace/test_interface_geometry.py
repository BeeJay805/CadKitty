"""Local interface instruments and fixed subjects; native execution belongs to root's gate."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace import interface_geometry as subject
from forge_mind.workspace.document import DesignDocument


def settings(*, fixed=()):
    return {
        "pairs": [
            {
                "id": "gap",
                "a": "a",
                "b": "b",
                "basis": "Explicit nominal test assumption.",
                "requirement_ids": ["r1"],
            }
        ],
        "fixed_occurrence_paths": list(fixed),
    }


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "same",
        "blank_path",
        "blank_basis",
        "bad_id",
        "duplicate_id",
        "duplicate_pair",
        "fixed_duplicate",
        "fixed_blank",
        "requirements_duplicate",
        "extra",
    ],
)
def test_settings_refuse_ambiguous_or_unattributed_queries(fault):
    value = settings()
    pair = value["pairs"][0]
    if fault == "empty":
        value["pairs"] = []
    elif fault == "same":
        pair["b"] = pair["a"]
    elif fault == "blank_path":
        pair["a"] = "  "
    elif fault == "blank_basis":
        pair["basis"] = "\n"
    elif fault == "bad_id":
        pair["id"] = "NotLower"
    elif fault == "duplicate_id":
        value["pairs"].append({**pair, "b": "c"})
    elif fault == "duplicate_pair":
        value["pairs"].append({**pair, "id": "other", "a": "b", "b": "a"})
    elif fault == "fixed_duplicate":
        value["fixed_occurrence_paths"] = ["a", "a"]
    elif fault == "fixed_blank":
        value["fixed_occurrence_paths"] = [" "]
    elif fault == "requirements_duplicate":
        pair["requirement_ids"] = ["r1", "r1"]
    else:
        pair["approved"] = True
    with pytest.raises(ValidationError):
        subject.InterfaceGeometrySettings.model_validate(value)


def test_metric_discovery_is_named_and_has_no_added_pair_cap():
    assert subject.interface_metrics(None) == []
    value = {
        "pairs": [
            {"id": f"pair_{n}", "a": f"a{n}", "b": f"b{n}", "basis": "Assumption."}
            for n in range(80)
        ]
    }
    metrics = subject.interface_metrics(value)
    assert len(metrics) == 160
    assert metrics[0]["key"] == "interface_geometry.pair_0_distance_mm"
    assert metrics[-1]["key"] == "interface_geometry.pair_79_overlap_mm3"
    assert {row["unit"] for row in metrics} == {"mm", "mm³"}


def saved(
    tmp_path,
    *,
    b_position=(15, 0, 0),
    b_size=10,
    repeated=False,
    b_definition="second",
    b_role="make",
    extra_reference=False,
    reference_position=(40, 0, 0),
    reference_size=4,
    reference_role="reference",
    shapes=None,
):
    """Save actual STEP and explicit metadata; no generated feature-program claim."""
    import build123d as cad

    from forge_mind.workspace.assembly import occurrence_label

    path = tmp_path / "source"
    path.mkdir(exist_ok=True)
    first = cad.Box(10, 10, 10, align=(cad.Align.MIN,) * 3)
    second = cad.Pos(*b_position) * cad.Box(b_size, b_size, b_size, align=(cad.Align.MIN,) * 3)
    if shapes is not None:
        first, second = shapes
    definition = "first" if repeated else b_definition
    parts = [("first", "make", 10)]
    if not repeated:
        parts.append((definition, b_role, b_size))
    roots = [
        {"id": "a", "definition": "first"},
        {"id": "b", "definition": definition, "position": list(b_position)},
    ]
    bodies = [first, second]
    paths = ["a", "b"]
    if extra_reference:
        parts.append(("third", reference_role, reference_size))
        roots.append(
            {"id": "reference", "definition": "third", "position": list(reference_position)}
        )
        bodies.append(
            cad.Pos(*reference_position)
            * cad.Box(reference_size, reference_size, reference_size, align=(cad.Align.MIN,) * 3)
        )
        paths.append("reference")
    for occurrence, body in zip(paths, bodies, strict=True):
        body.label = occurrence_label(occurrence)
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": name,
                        "name": name,
                        "output": "body",
                        "production": {"role": role},
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"x": size, "y": size, "z": size},
                            }
                        ],
                    }
                    for name, role, size in parts
                ],
                "roots": roots,
            }
        }
    ).model_dump(mode="json")
    (path / "document.json").write_text(json.dumps(document), encoding="utf-8")
    cad.export_step(cad.Compound(children=bodies), path / "part.step")
    return document, path


@pytest.mark.slow
@pytest.mark.parametrize(
    "position,size,distance,overlap",
    [
        ((15, 0, 0), 10, 5, 0),
        ((10, 0, 0), 10, 0, 0),
        ((8, 0, 0), 10, 0, 200),
        ((4, 4, 4), 2, 0, 8),
    ],
)
def test_separation_touch_overlap_and_containment_have_independent_measurements(
    tmp_path, position, size, distance, overlap
):
    _, path = saved(tmp_path, b_position=position, b_size=size)
    before = {name: (path / name).read_bytes() for name in ("document.json", "part.step")}
    result = subject.interface_geometry_worker(path, settings())
    assert result["status"] == "complete" and result["eligible"] is True
    assert result["qualification_status"] == "unverified" and result["scope"] == subject.SCOPE
    assert result["metrics"]["gap_distance_mm"] == pytest.approx(distance, abs=1e-7)
    assert result["metrics"]["gap_overlap_mm3"] == pytest.approx(overlap, abs=1e-7)
    pair = result["pairs"][0]
    assert math.dist(pair["witness_a_mm"], pair["witness_b_mm"]) == pytest.approx(
        distance, abs=1e-6
    )
    assert pair["a_volume_mm3"] == pytest.approx(1000)
    assert pair["b_volume_mm3"] == pytest.approx(size**3)
    assert sum(row["volume_mm3"] for row in pair["overlap_components"]) == pytest.approx(overlap)
    assert {name: (path / name).read_bytes() for name in before} == before


@pytest.mark.slow
def test_repeated_definition_uses_occurrence_placement_not_definition_identity(tmp_path):
    document, path = saved(tmp_path, repeated=True)
    bindings = subject.bind_interface_targets(document, path, settings())
    assert bindings["selected"] == {
        "a": {"definition": "first", "role": "make"},
        "b": {"definition": "first", "role": "make"},
    }
    result = subject.interface_geometry_worker(path, settings())
    assert result["metrics"]["gap_distance_mm"] == pytest.approx(5)
    assert result["pairs"][0]["a_body_index"] != result["pairs"][0]["b_body_index"]


@pytest.mark.slow
def test_all_disconnected_overlap_witnesses_are_retained(tmp_path):
    import build123d as cad

    bridge = cad.Box(10, 2, 2, align=(cad.Align.MIN,) * 3)
    left = cad.Box(2, 10, 2, align=(cad.Align.MIN,) * 3)
    right = cad.Pos(8, 0, 0) * cad.Box(2, 10, 2, align=(cad.Align.MIN,) * 3)
    first = bridge.fuse(left, right)
    second = cad.Pos(0, 5, 0) * cad.Box(10, 2, 2, align=(cad.Align.MIN,) * 3)
    _, path = saved(tmp_path, shapes=(first, second))
    result = subject.interface_geometry_worker(path, settings())
    components = result["pairs"][0]["overlap_components"]
    assert len(components) == 2
    assert sorted(row["volume_mm3"] for row in components) == pytest.approx([8, 8])
    assert result["metrics"]["gap_overlap_mm3"] == pytest.approx(16)


@pytest.mark.slow
def test_unselected_reference_is_frozen_while_working_pair_can_move(tmp_path):
    document, path = saved(tmp_path, extra_reference=True)
    bindings = subject.bind_interface_targets(document, path, settings())
    assert bindings["reference_occurrence_paths"] == ["reference"]
    assert set(bindings["fixed"]) == {"reference"}
    candidate, path = saved(tmp_path, extra_reference=True, b_position=(13, 0, 0))
    subject.validate_interface_targets(candidate, path, settings(), bindings)
    assert subject.interface_geometry_worker(path, settings())["metrics"][
        "gap_distance_mm"
    ] == pytest.approx(3)


@pytest.mark.slow
@pytest.mark.parametrize(
    "change",
    [
        {"b_definition": "retargeted"},
        {"b_role": "buy"},
        {"reference_role": "buy"},
        {"reference_position": (41, 0, 0)},
        {"reference_size": 5},
        {"extra_reference": False},
    ],
)
def test_candidate_cannot_retarget_selected_or_alter_any_baseline_reference(tmp_path, change):
    document, path = saved(tmp_path, extra_reference=True)
    bindings = subject.bind_interface_targets(document, path, settings())
    candidate, path = saved(tmp_path, **{"extra_reference": True, **change})
    with pytest.raises(ValueError, match="frozen target changed"):
        subject.validate_interface_targets(candidate, path, settings(), bindings)


@pytest.mark.slow
def test_explicit_fixed_occurrence_and_binding_integrity_are_enforced(tmp_path):
    document, path = saved(tmp_path)
    frozen_settings = settings(fixed=["b"])
    bindings = subject.bind_interface_targets(document, path, frozen_settings)
    candidate, path = saved(tmp_path, b_position=(13, 0, 0))
    with pytest.raises(ValueError, match="fixed"):
        subject.validate_interface_targets(candidate, path, frozen_settings, bindings)
    document, path = saved(tmp_path)
    for fault in ("selected", "source_hashes", "version"):
        changed = deepcopy(bindings)
        changed[fault] = True if fault == "version" else {}
        with pytest.raises(ValueError):
            subject.validate_interface_targets(document, path, frozen_settings, changed)


@pytest.mark.slow
@pytest.mark.parametrize(
    "fault",
    [
        "missing_path",
        "missing_fixed",
        "missing_label",
        "duplicate_label",
        "corrupt_step",
        "wrong_document",
    ],
)
def test_missing_corrupt_or_relabelled_sources_never_return_scores(tmp_path, fault):
    document, path = saved(tmp_path)
    value = settings()
    if fault == "missing_path":
        value["pairs"][0]["b"] = "unknown"
    elif fault == "missing_fixed":
        value["fixed_occurrence_paths"] = ["unknown"]
    elif fault == "corrupt_step":
        (path / "part.step").write_bytes(b"Not a STEP file")
    elif fault == "wrong_document":
        document["name"] = "Foreign document"
        with pytest.raises(ValueError, match="saved source"):
            subject.bind_interface_targets(document, path, value)
        return
    else:
        import build123d as cad

        from forge_mind.workspace.assembly import occurrence_label

        shape = cad.import_step(path / "part.step")
        for child in shape.children:
            child.label = occurrence_label("a") if fault == "duplicate_label" else "unlabelled"
        cad.export_step(shape, path / "part.step")
    with pytest.raises(ValueError):
        subject.interface_geometry_worker(path, value)


@pytest.mark.slow
def test_source_change_during_measurement_is_not_an_eligible_result(tmp_path, monkeypatch):
    from forge_mind.workspace import geometry_distance_native

    _, path = saved(tmp_path)
    original = geometry_distance_native._distance_result

    def changed(operation):
        result = original(operation)
        with (path / "document.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(geometry_distance_native, "_distance_result", changed)
    with pytest.raises(ValueError, match="source changed"):
        subject.interface_geometry_worker(path, settings())
