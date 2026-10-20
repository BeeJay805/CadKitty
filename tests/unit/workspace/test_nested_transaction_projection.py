"""Saved evidence stays exact; only automatic transaction inventories are projected."""

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_agent_inspection_queries import inspection_case, read_once
from tests.unit.workspace.test_native_handoff_controller import RID

from forge_mind.workspace import document_service
from forge_mind.workspace.agent import assembly_measurement_context
from forge_mind.workspace.agent_memory import recent_context
from forge_mind.workspace.document import digest
from forge_mind.workspace.instruction_context import customer_instruction_context
from forge_mind.workspace.native_review_context import compact_review_context
from forge_mind.workspace.native_stages import deduplicate_context


def dense_facts():
    return {
        "extents_mm": [40, 30, 20],
        "feature_history": [
            {"id": f"feature_{index}", "state": "built", "key": "b" * 64} for index in range(80)
        ],
        "assembly": {
            "status": "unverified",
            "scope": "Function, load and physical clearance remain unverified",
            "mates": [
                {"id": "mate_1", "status": "unverified", "missing_evidence": ["physical test"]}
            ],
            "ports": [{"id": "port_1", "position_mm": [1, 2, 3]}],
            "occurrences": [
                {
                    "path": "root/part",
                    "definition": "part",
                    "volume_mm3": 100,
                    "cylindrical_surfaces": [
                        {
                            "surface_id": f"cylinder_{index}",
                            "radius_mm": 4,
                            "diameter_mm": 8,
                            "axis_point_mm": [0, 0, 0],
                            "axis_direction": [0, 0, 1],
                            "concave": True,
                            "axial_extent_mm": [0, 10],
                        }
                        for index in range(200)
                    ],
                }
            ],
        },
        "mechanism": {
            "samples": [{"poses": {"part": [1, 2, 3, 4]}}],
            "bindings": [{"id": "binding"}],
        },
    }


def receipt():
    return {
        "action": "construction",
        "job_id": "b" * 32,
        "applied": True,
        "plan_applied": True,
        "cad_applied": True,
        "base_revision_id": "c" * 32,
        "candidate_revision_id": RID,
        "revision_id": RID,
        "plan_hash": "d" * 64,
        "plan_version": 2,
        "edit_basis": {"revision_id": "c" * 32, "hash": "e" * 64},
        "coverage": {"status": "unverified", "missing": ["physical load"]},
        "measurement": {
            "revision_id": RID,
            "geometry_valid": True,
            "requirements_satisfied": False,
            "contract_hash": "f" * 64,
            "artifact_hashes": {"part.step": "a" * 64},
            "engine_fingerprint": "0" * 64,
            "findings": [{"status": "unverified", "requirement_id": "load"}],
            "errors": [],
            "caveats": ["Physical behavior unverified"],
            "facts": dense_facts(),
        },
    }


def test_automatic_projection_keeps_receipt_authority_and_unresolved_duties():
    original = {"construction_transaction": receipt()}
    before = deepcopy(original)
    projected = recent_context([original], {})[0]
    saved = original["construction_transaction"]
    current = projected["construction_transaction"]
    assert original == before
    assert {key: value for key, value in current.items() if key != "measurement"} == {
        key: value for key, value in saved.items() if key != "measurement"
    }
    old, new = saved["measurement"], current["measurement"]
    assert {key: value for key, value in new.items() if key != "facts"} == {
        key: value for key, value in old.items() if key != "facts"
    }
    assert new["facts"]["feature_history"]["sha256"] == digest(old["facts"]["feature_history"])
    summary = new["facts"]["assembly"]["occurrences"][0]["cylindrical_surfaces"]
    assert summary["sha256"] == digest(
        old["facts"]["assembly"]["occurrences"][0]["cylindrical_surfaces"]
    )
    assert summary["revision_id"] == RID
    assert new["facts"]["assembly"]["mates"] == old["facts"]["assembly"]["mates"]
    assert new["facts"]["assembly"]["status"] == "unverified"
    assert new["facts"]["mechanism"] == old["facts"]["mechanism"]
    assert len(json.dumps(projected)) < len(json.dumps(original))


@pytest.mark.parametrize(
    "marker",
    [
        {"status": "failed"},
        {"status": "needs_review"},
        {"status": "unknown_future_state"},
        {"errors": ["failed axis"]},
        {"error": "bad axis"},
        {"caveats": ["thin wall"]},
        {"unresolved": ["datum"]},
        {"missing_evidence": ["load"]},
        {"warnings": ["precision"]},
        {"geometry_valid": False},
        {"diagnostics": [{"message": "unknown unresolved duty"}]},
        {"failure_reason": "unknown missing interface"},
    ],
)
def test_failure_unknown_and_unresolved_inventory_records_remain_full(marker):
    row = {"construction_transaction": receipt()}
    row["construction_transaction"]["measurement"]["facts"]["feature_history"].append(marker)
    before = deepcopy(row)
    assert recent_context([row], {}) == [before]
    assert row == before


@pytest.mark.parametrize(
    "change",
    [
        {"applied": False},
        {"cad_applied": False},
        {"status": "failed"},
        {"status": "unknown"},
        {"error": "transaction failed"},
        {"candidate_revision_id": "b" * 32},
    ],
)
def test_failed_or_mismatched_transaction_is_never_projected(change):
    row = {"construction_transaction": {**receipt(), **change}}
    assert recent_context([row], {}) == [row]


def test_unknown_row_and_explicit_inspection_are_never_automatic_projection_candidates():
    row = {"construction_transaction": receipt(), "explicit": True}
    evidence = {"inspection": receipt()["measurement"], "revision_id": RID}
    assert recent_context([row, evidence], {}) == [row, evidence]
    projected = deduplicate_context(
        compact_review_context(
            {"selected": {"id": RID, "measurement": evidence["inspection"]}, "history": [evidence]}
        )
    )
    assert projected["history"] == [evidence]


def saved_dense_case(tmp_path, monkeypatch):
    case = inspection_case(tmp_path, monkeypatch)
    revision = case.service.store.revision(case.sid, RID)
    revision["measurement"]["facts"] = dense_facts()
    (case.service.store.directory(case.sid, RID) / "evidence.json").write_text(
        json.dumps(revision["measurement"]), encoding="utf-8"
    )
    with case.service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), case.sid, RID),
        )
    return case, deepcopy(revision)


@pytest.mark.parametrize("query", ["", "evidence"])
def test_real_dispatch_full_evidence_is_exact_and_empty_query_stays_summary(
    tmp_path, monkeypatch, query
):
    case, revision = saved_dense_case(tmp_path, monkeypatch)
    row, saved = read_once(case, query, revision_id=RID)
    full = revision["measurement"]
    expected = full if query == "evidence" else assembly_measurement_context(full)
    assert row == {"inspection": expected, "revision_id": RID}
    assert assembly_measurement_context(full) != full
    assert case.service.store.revision(case.sid, RID) == revision
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert saved["plan_hash"] == case.before["plan_hash"]
    assert len(case.service.store.revisions(case.sid)) == 1
    if query == "evidence":
        projected = deduplicate_context(
            customer_instruction_context(compact_review_context({"history": [row]}))
        )
        assert projected["history"] == [row]
        automatic = recent_context(
            [{"construction_transaction": {**receipt(), "measurement": full}}], {}
        )[0]["construction_transaction"]["measurement"]
        assert automatic["facts"]["feature_history"]["sha256"] == digest(
            row["inspection"]["facts"]["feature_history"]
        )


def test_evidence_does_not_bypass_saved_artifact_integrity(tmp_path, monkeypatch):
    case, revision = saved_dense_case(tmp_path, monkeypatch)
    program = case.service.store.directory(case.sid, RID) / "program.py"
    original_bytes = program.read_bytes()
    shared_load = case.service._load_revision

    def load(*args, **kwargs):
        try:
            return shared_load(*args, **kwargs)
        finally:
            program.write_bytes(original_bytes)

    monkeypatch.setattr(case.service, "_load_revision", load)

    def tamper():
        program.write_bytes(b"changed")

    row, saved = read_once(case, "evidence", revision_id=RID, before_action=tamper)
    assert "inspection" not in row
    assert "stale" in json.dumps(row).lower() or "changed" in json.dumps(row).lower()
    assert saved.get("accepted_revision") is None
    assert case.service.store.revision(case.sid, RID) == revision


def test_evidence_cannot_read_a_foreign_revision(tmp_path, monkeypatch):
    case, revision = saved_dense_case(tmp_path, monkeypatch)
    row, saved = read_once(case, "evidence", revision_id="9" * 32)
    assert "inspection" not in row
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert case.service.store.revision(case.sid, RID) == revision


@pytest.mark.parametrize("state", ["missing", "malformed", "mismatch", "symlink"])
def test_real_dispatch_refuses_unavailable_or_changed_saved_evidence(tmp_path, monkeypatch, state):
    case, revision = saved_dense_case(tmp_path, monkeypatch)
    path = case.service.store.directory(case.sid, RID) / "evidence.json"

    def change():
        if state == "missing":
            path.unlink()
        elif state == "malformed":
            path.write_text("{broken", encoding="utf-8")
        elif state == "mismatch":
            path.write_text("{}", encoding="utf-8")
        else:
            target = path.with_name("other.json")
            target.write_bytes(path.read_bytes())
            path.unlink()
            try:
                path.symlink_to(target)
            except OSError as exc:
                pytest.skip(f"Symlink creation unavailable: {exc}")

    row, saved = read_once(case, "evidence", revision_id=RID, before_action=change)
    assert "inspection" not in row and "tool_error" in row
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert case.service.store.revision(case.sid, RID) == revision


def evidence_client(case):
    from fastapi import FastAPI, Header
    from fastapi.testclient import TestClient

    from forge_mind.workspace.api import router

    def owner(authorization: str = Header()):
        return authorization

    app = FastAPI()
    app.include_router(router(case.service.store, case.service, owner))
    return TestClient(app)


@pytest.mark.parametrize(
    "state,expected",
    [
        ("normal", 200),
        ("missing", 404),
        ("malformed", 409),
        ("mismatch", 409),
        ("symlink", 404),
    ],
)
def test_http_saved_record_boundary_and_owner_guard(tmp_path, monkeypatch, state, expected):
    case, revision = saved_dense_case(tmp_path, monkeypatch)
    path = case.service.store.directory(case.sid, RID) / "evidence.json"
    if state == "missing":
        path.unlink()
    elif state == "malformed":
        path.write_text("{broken", encoding="utf-8")
    elif state == "mismatch":
        path.write_text("{}", encoding="utf-8")
    elif state == "symlink":
        target = path.with_name("other.json")
        target.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"Symlink creation unavailable: {exc}")
    client = evidence_client(case)
    url = f"/workspace/sessions/{case.sid}/revisions/{RID}/evidence.json"
    assert client.get(url, headers={"Authorization": "foreign"}).status_code == 404
    response = client.get(url, headers={"Authorization": "owner"})
    assert response.status_code == expected
    if expected == 200:
        assert response.json() == revision["measurement"]
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["x-content-type-options"] == "nosniff"


def test_http_evidence_does_not_require_unrelated_cad_or_display_files(tmp_path, monkeypatch):
    case, revision = saved_dense_case(tmp_path, monkeypatch)
    (case.service.store.directory(case.sid, RID) / "part.step").unlink()
    client = evidence_client(case)
    response = client.get(
        f"/workspace/sessions/{case.sid}/revisions/{RID}/evidence.json",
        headers={"Authorization": "owner"},
    )
    assert response.status_code == 200
    assert response.json() == revision["measurement"]


def test_shared_loader_is_limited_to_existing_literal_record_names(tmp_path, monkeypatch):
    case, _ = saved_dense_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Only saved evidence and contract"):
        document_service.read_saved_record(case.service, case.sid, RID, "program.py")


@pytest.mark.parametrize(
    "field,value",
    [
        ("radius_mm", None),
        ("radius_mm", "4"),
        ("radius_mm", True),
        ("radius_mm", []),
        ("radius_mm", {}),
        ("radius_mm", float("inf")),
        ("axis_direction", None),
        ("axis_direction", "axis"),
        ("axis_direction", True),
        ("axis_direction", {}),
        ("axis_direction", []),
        ("axis_direction", [0, 0, 0]),
        ("axis_direction", [0, 0, 2]),
        ("axis_direction", [0, 0, 1e308]),
        ("axial_extent_mm", [10, 0]),
        ("axial_extent_mm", None),
        ("concave", None),
        ("concave", 1),
        ("diagnostics", [{"code": "missing_interface", "message": "unresolved duty"}]),
        ("failure_reason", "unresolved duty"),
    ],
)
def test_unrecognized_or_malformed_surface_inventory_remains_full(field, value):
    row = {"construction_transaction": receipt()}
    surface = row["construction_transaction"]["measurement"]["facts"]["assembly"]["occurrences"][0][
        "cylindrical_surfaces"
    ][0]
    surface[field] = value
    before = deepcopy(row)
    assert recent_context([row], {}) == [before]
    assert row == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", None),
        ("state", []),
        ("state", {}),
        ("state", "failed"),
        ("key", None),
        ("key", 1),
        ("key", []),
        ("key", "not-a-digest"),
        ("key", "g" * 64),
        ("diagnostics", [{"error": "missing"}]),
    ],
)
def test_unrecognized_or_malformed_history_inventory_remains_full(field, value):
    row = {"construction_transaction": receipt()}
    row["construction_transaction"]["measurement"]["facts"]["feature_history"][0][field] = value
    before = deepcopy(row)
    assert recent_context([row], {}) == [before]
    assert row == before


@pytest.mark.parametrize("name", ["evidence.json", "contract.json"])
@pytest.mark.parametrize(
    "expected,changed",
    [
        ({"flag": True}, {"flag": 1}),
        ({"flag": False}, {"flag": 0}),
        ({"count": 1}, {"count": 1.0}),
        ({"number": float("nan")}, {"number": float("nan")}),
        ({"number": float("inf")}, {"number": float("inf")}),
        ({"number": float("-inf")}, {"number": float("-inf")}),
    ],
)
def test_shared_literal_reader_refuses_type_changes_and_nonfinite_records(
    tmp_path, name, expected, changed
):
    from types import SimpleNamespace

    class SavedStore:
        def revision(self, sid, rid):
            assert (sid, rid) == ("owned", RID)
            return {"measurement": expected, "contract": expected}

        def directory(self, sid, rid):
            assert (sid, rid) == ("owned", RID)
            return tmp_path

    path = tmp_path / name
    path.write_text(json.dumps(changed), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        document_service.read_saved_record(SimpleNamespace(store=SavedStore()), "owned", RID, name)
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["evidence.json", "contract.json"])
@pytest.mark.parametrize(
    "state,expected_status",
    [
        ("normal", 200),
        ("malformed", 409),
        ("mismatch", 409),
        ("nonfinite", 409),
        ("type_change", 409),
    ],
)
def test_both_saved_records_keep_exact_http_semantics(
    tmp_path, monkeypatch, name, state, expected_status
):
    case, revision = saved_dense_case(tmp_path, monkeypatch)
    expected = revision["measurement" if name == "evidence.json" else "contract"]
    path = case.service.store.directory(case.sid, RID) / name
    path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    if state == "malformed":
        path.write_text("{broken", encoding="utf-8")
    elif state == "mismatch":
        path.write_text("{}", encoding="utf-8")
    elif state == "nonfinite":
        changed = deepcopy(expected)
        if name == "evidence.json":
            changed["facts"]["volume_mm3"] = float("nan")
        else:
            changed["requirements"][0]["value"] = float("inf")
        path.write_text(json.dumps(changed), encoding="utf-8")
    elif state == "type_change":
        changed = deepcopy(expected)
        if name == "evidence.json":
            changed["geometry_valid"] = 1
        else:
            original = changed["requirements"][0]["value"]
            changed["requirements"][0]["value"] = (
                int(original) if type(original) is float else float(original)
            )
        path.write_text(json.dumps(changed), encoding="utf-8")
    response = evidence_client(case).get(
        f"/workspace/sessions/{case.sid}/revisions/{RID}/{name}", headers={"Authorization": "owner"}
    )
    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json() == expected
        assert document_service.read_saved_record(case.service, case.sid, RID, name) == expected


@pytest.mark.parametrize("channel", ["agent", "http"])
def test_shared_requested_file_link_guard_is_exercised_without_link_privileges(
    tmp_path, monkeypatch, channel
):
    from pathlib import Path

    case, revision = saved_dense_case(tmp_path, monkeypatch)
    evidence = case.service.store.directory(case.sid, RID) / "evidence.json"
    original = Path.is_symlink

    def linked(path):
        return True if path == evidence else original(path)

    monkeypatch.setattr(Path, "is_symlink", linked)
    if channel == "agent":
        row, saved = read_once(case, "evidence", revision_id=RID)
        assert "inspection" not in row and "tool_error" in row
        assert saved["selected_revision"] == RID
    else:
        response = evidence_client(case).get(
            f"/workspace/sessions/{case.sid}/revisions/{RID}/evidence.json",
            headers={"Authorization": "owner"},
        )
        assert response.status_code == 404
    assert case.service.store.revision(case.sid, RID) == revision
