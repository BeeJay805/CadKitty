"""Saved-fact projection requirements; byte fixtures do not claim native qualification."""

import hashlib
import json
from copy import deepcopy

import pytest

from forge_mind.workspace.topology_projection import topology_projection


def entity(name, kind, body):
    row = {
        "id": name,
        "kind": kind,
        "signature": name + "-signature",
        "body_signature": f"body-{body}",
        "body_index": body,
        "center": [1, 2, 3],
        "min": [0, 0, 0],
        "max": [2, 4, 6],
        "measure": 12,
        "unit": {"body": "mm^3", "face": "mm^2", "edge": "mm"}[kind],
    }
    if kind == "face":
        row.update(surface="GeomType.PLANE", normal=[0, 0, 1])
    if kind == "edge":
        row.update(curve="GeomType.LINE", polyline=[[0, 0, 0], [2, 4, 6]])
    return row


def sample():
    return {
        "frame": "cad_z_up_mm",
        "selection_scope": "body",
        "topology_counts": {"bodies": 2, "faces": 12, "edges": 24},
        "detail_policy": {"mode": "full_index", "face_edge_picking_available": False},
        "entities": [
            entity("b0", "body", 0),
            entity("f0", "face", 0),
            entity("b1", "body", 1),
            entity("f1", "face", 1),
            entity("e0", "edge", 0),
            entity("f2", "face", 0),
        ],
        "meshes": [{"vertices": [[0, 0, 0]], "triangles": []}],
    }


def encoded(value):
    return json.dumps(value, indent=2).encode()


def test_legacy_minimal_rows_and_complete_entity_records_are_unchanged():
    data = {"entities": [{"id": "old", "polyline": [1], "future_field": "literal"}]}
    assert topology_projection(encoded(data), "r") == {
        "revision_id": "r",
        "total": 1,
        "entities": data["entities"],
        "next_offset": None,
    }
    assert topology_projection(encoded(data), "r", offset=7) == {
        "revision_id": "r",
        "total": 1,
        "entities": [],
        "next_offset": None,
    }


def test_filters_run_before_pagination_and_retain_exact_ids_signatures():
    data = encoded(sample())
    first = topology_projection(data, "r", body_index=0, kind="face", compact=True, limit=1)
    second = topology_projection(
        data, "r", body_index=0, kind="face", compact=True, offset=1, limit=1
    )
    assert [first["entities"][0]["id"], second["entities"][0]["id"]] == ["f0", "f2"]
    assert first["entities"][0]["signature"] == "f0-signature"
    assert first["total"] == second["total"] == 2
    assert first["returned"] == second["returned"] == 1
    assert first["next_offset"] == 1 and second["next_offset"] is None
    assert first["indexed_counts"] == {"body": 1, "face": 2, "edge": 1}
    assert first["topology_counts"] == {"bodies": 2, "faces": 12, "edges": 24}
    assert first["topology_sha256"] == hashlib.sha256(data).hexdigest()
    assert first["frame"] == "cad_z_up_mm"


def test_body_picking_does_not_hide_face_inventory_and_compact_omits_only_display_data():
    value = sample()
    before = deepcopy(value)
    result = topology_projection(encoded(value), "r", compact=True)
    assert result["inventory"]["faces"] == "indexed"
    assert result["detail_policy"] == value["detail_policy"]
    assert result["entities"][1] == value["entities"][1]
    assert result["entities"][4] == {
        k: v for k, v in value["entities"][4].items() if k != "polyline"
    }
    assert "meshes" not in result
    assert value == before
    full = topology_projection(encoded(value), "r", kind="edge")
    assert full["entities"] == [value["entities"][4]]


def test_body_only_inventory_reports_missing_detail_not_absent_faces():
    value = sample()
    value["entities"] = [row for row in value["entities"] if row["kind"] == "body"]
    value["detail_policy"]["mode"] = "body_index"
    result = topology_projection(encoded(value), "r", compact=True, kind="face")
    assert result["total"] == result["returned"] == 0 and result["entities"] == []
    assert result["inventory"]["faces"] == "unavailable"
    assert result["topology_counts"]["faces"] == 12
    assert "does not establish absent geometry" in result["inventory"]["missing_detail"]


@pytest.mark.parametrize(
    "options",
    [
        {"offset": -1},
        {"offset": True},
        {"limit": 0},
        {"limit": 101},
        {"limit": 2.0},
        {"body_index": True},
        {"body_index": -1},
        {"kind": "hole"},
        {"compact": 1},
    ],
)
def test_invalid_requested_pages_and_filters_are_refused(options):
    with pytest.raises(ValueError):
        topology_projection(encoded(sample()), "r", **options)


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(entities="not-list"),
        lambda v: v["entities"].append(deepcopy(v["entities"][0])),
        lambda v: v["entities"][0].update(body_index=True),
        lambda v: v["entities"][0].update(center=[0, 0, float("inf")]),
        lambda v: v["entities"][0].update(measure=True),
        lambda v: v["entities"][0].update(min=[3, 0, 0]),
        lambda v: v.update(frame="guessed-local"),
        lambda v: v["topology_counts"].update(faces=True),
    ],
)
def test_malformed_compact_facts_are_refused(change):
    value = sample()
    change(value)
    with pytest.raises(ValueError):
        topology_projection(encoded(value), "r", compact=True)


@pytest.mark.parametrize("data", [b"[]", b'{"entities":[],"entities":[]}', b"{", b"\xff"])
def test_malformed_json_is_not_summarized(data):
    with pytest.raises(ValueError):
        topology_projection(data, "r", compact=True)


def test_raw_byte_hash_distinguishes_equivalent_json_and_missing_metadata_stays_unknown():
    value = sample()
    value.pop("detail_policy")
    value.pop("topology_counts")
    pretty = topology_projection(encoded(value), "r", compact=True)
    plain = topology_projection(json.dumps(value).encode(), "r", compact=True)
    assert pretty["entities"] == plain["entities"]
    assert pretty["topology_sha256"] != plain["topology_sha256"]
    assert pretty["detail_policy"] is None and pretty["topology_counts"] is None
