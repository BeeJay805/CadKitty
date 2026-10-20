import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from forge_mind.workspace import product_interfaces as interfaces
from forge_mind.workspace import product_sources as sources
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = Store(tmp_path)
    contract = {"requirements": [{"id": "fan_fit", "text": "Padded fan fits"}]}
    sid = store.create("alice", "fan", contract)["id"]
    rid = "a" * 32
    store.add_revision(sid, {"id": rid, "contract": contract})
    service = SimpleNamespace(store=store)
    monkeypatch.setattr(
        sources,
        "retrieve",
        lambda *args: (
            b"Padded fan 27 mm. Bare fan 25 mm. 2.7 cm.",
            "text/plain",
            "https://vendor.example/a",
        ),
    )
    state = sources.fetch_source(service, sid, "alice", "https://vendor.example/a")
    claim = {
        "source_artifact": state["artifacts"][0]["id"],
        "quantity": 27,
        "unit": "mm",
        "exact_quote": "Padded fan 27 mm.",
        "configuration": "padded",
        "datum": "outer pads",
        "locator": "spec",
    }
    state = sources.propose_claim(store, sid, "alice", claim)
    cid = state["claims"][0]["id"]
    sources.confirm_claim(store, sid, "alice", cid, state["version"])
    payload = {
        "name": "Padded fan interface",
        "reference_revision_id": rid,
        "requirement_ids": ["fan_fit"],
        "configuration": "padded",
        "datum": "Fan axis is world Z; center is assembly origin",
        "request": {
            "tool": "hole_pattern",
            "units": "mm",
            "body": {"body_index": 0},
            "axis": [0.0, 0.0, 1.0],
            "holes": [{"center": [0.0, 0.0, 0.0], "diameter": 27.0}],
        },
        "bindings": [{"claim_id": cid, "request_path": "/holes/0/diameter"}],
    }
    return store, sid, rid, service, payload, claim


def confirm(setup):
    store, sid, _rid, _service, payload, _claim = setup
    state = interfaces.propose_interface(store, sid, "alice", payload, agent=True)
    assert state["records"][0]["confirmation"] is None
    return interfaces.confirm_interface(
        store, sid, "alice", state["records"][0]["id"], state["version"]
    )


def test_persistence_and_explicit_semantic_confirmation(setup):
    store, sid, _rid, _service, payload, _claim = setup
    state = confirm(setup)
    record = state["records"][0]
    assert record["confirmation"]["confirmed_by"] == "alice"
    assert "not source truth" in record["confirmation"]["meaning"]
    assert record["check_status"] == "unverified"
    assert (
        interfaces.read_interfaces(Store(store.root), sid, "alice")["records"] == state["records"]
    )
    with pytest.raises(ValidationError):
        interfaces.propose_interface(
            store, sid, "alice", {**payload, "confirmation": True}, agent=True
        )


def test_wrong_25_vs_27_and_dangling_requirement(setup):
    store, sid, _rid, _service, payload, _claim = setup
    wrong = copy.deepcopy(payload)
    wrong["request"]["holes"][0]["diameter"] = 25.0
    with pytest.raises(WorkspaceError, match="differs"):
        interfaces.propose_interface(store, sid, "alice", wrong)
    with pytest.raises(WorkspaceError, match="missing"):
        interfaces.propose_interface(
            store, sid, "alice", {**payload, "requirement_ids": ["invented"]}
        )


def test_wrong_axis_is_not_silently_source_confirmed(setup):
    store, sid, _rid, _service, payload, _claim = setup
    payload["request"]["axis"] = [1.0, 0.0, 0.0]
    state = interfaces.propose_interface(store, sid, "alice", payload, agent=True)
    assert state["records"][0]["confirmation"] is None
    assert state["records"][0]["check_status"] == "unverified"
    payload["bindings"][0]["request_path"] = "/axis/0"
    with pytest.raises(WorkspaceError, match="not an axis"):
        interfaces.propose_interface(store, sid, "alice", payload)


def test_unconfirmed_source_and_units(setup):
    store, sid, _rid, _service, payload, claim = setup
    state = sources.propose_claim(store, sid, "alice", claim)
    cid = state["claims"][-1]["id"]
    payload["bindings"][0]["claim_id"] = cid
    with pytest.raises(WorkspaceError, match="independently confirmed"):
        interfaces.propose_interface(store, sid, "alice", payload)
    state = sources.propose_claim(store, sid, "alice", {**claim, "unit": "deg"})
    cid = state["claims"][-1]["id"]
    sources.confirm_claim(store, sid, "alice", cid, state["version"])
    payload["bindings"][0]["claim_id"] = cid
    with pytest.raises(WorkspaceError, match="length units"):
        interfaces.propose_interface(store, sid, "alice", payload)


def test_cm_conversion(setup):
    store, sid, _rid, _service, payload, claim = setup
    state = sources.propose_claim(
        store, sid, "alice", {**claim, "quantity": 2.7, "unit": "cm", "exact_quote": "2.7 cm."}
    )
    cid = state["claims"][-1]["id"]
    sources.confirm_claim(store, sid, "alice", cid, state["version"])
    payload["bindings"][0]["claim_id"] = cid
    assert interfaces.propose_interface(store, sid, "alice", payload)["records"]


def test_source_change_invalidates_confirmation(setup):
    store, sid, rid, service, _payload, claim = setup
    confirm(setup)
    sources.propose_claim(store, sid, "alice", claim)
    assert interfaces.read_interfaces(store, sid, "alice", rid)["records"][0]["stale"]
    result = interfaces.validate_interfaces(service, sid, "alice", rid)
    assert result[0]["result"]["status"] == "unverified"
    assert "changed" in result[0]["reason"]


def test_each_revision_reruns_no_inherited_pass(setup, monkeypatch):
    from forge_mind.workspace import product_service
    from forge_mind.workspace.measure import engine_fingerprint

    store, sid, rid, service, payload, _claim = setup
    payload["request"]["body"] = {"occurrence_path": "panel"}
    confirm(setup)
    calls = []

    def tool(_service, _sid, _owner, revision, request, timeout, cancelled):
        from forge_mind.validation_provenance import file_sha256

        calls.append(revision)
        directory = store.directory(sid, revision)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "part.step").write_text("fixture exported artifact")
        return {
            "artifact_hashes": {"part.step": file_sha256(directory / "part.step")},
            "revision_id": revision,
            "request": request,
            "engine_fingerprint": engine_fingerprint(),
            "result": {"status": "pass" if revision == rid else "fail"},
        }

    monkeypatch.setattr(product_service, "run_tool", tool)
    interfaces.validate_interfaces(service, sid, "alice", rid)
    assert (
        interfaces.read_interfaces(store, sid, "alice", rid)["records"][0]["check_status"] == "pass"
    )
    newer = "b" * 32
    store.add_revision(sid, {"id": newer, "contract": store.revision(sid, rid)["contract"]})
    assert (
        interfaces.read_interfaces(store, sid, "alice", newer)["records"][0]["check_status"]
        == "unverified"
    )
    assert (
        interfaces.validate_interfaces(service, sid, "alice", newer)[0]["result"]["status"]
        == "fail"
    )
    assert calls == [rid, newer]


def test_owner_version_and_active_job_guards(setup):
    store, sid, _rid, _service, payload, _claim = setup
    state = interfaces.propose_interface(store, sid, "alice", payload)
    iid = state["records"][0]["id"]
    with pytest.raises(WorkspaceError, match="not found"):
        interfaces.confirm_interface(store, sid, "bob", iid, state["version"])
    with pytest.raises(WorkspaceError, match="version changed"):
        interfaces.confirm_interface(store, sid, "alice", iid, 0)
    with store.connect() as con:
        con.execute(
            "INSERT INTO jobs (id,session,state,payload) VALUES (?,?,?,?)",
            ("job", sid, "running", "{}"),
        )
    interfaces.propose_interface(store, sid, "alice", payload, agent=True)
    with pytest.raises(WorkspaceError, match="active operation"):
        interfaces.confirm_interface(store, sid, "alice", iid, state["version"])


def test_unsourced_requires_explicit_confirmation_and_is_labeled(setup):
    store, sid, _rid, _service, payload, _claim = setup
    payload["bindings"] = []
    state = interfaces.propose_interface(store, sid, "alice", payload, agent=True)
    record = state["records"][0]
    assert record["binding_scope"] == "user-declared"
    assert record["confirmation"] is None
    assert record["check_status"] == "unverified"
    state = interfaces.confirm_interface(store, sid, "alice", record["id"], state["version"])
    assert state["records"][0]["binding_scope"] == "user-declared"
    assert state["records"][0]["check_status"] == "unverified"


def test_contract_change_stales_even_when_old_revision_requested(setup):
    import json

    store, sid, rid, service, _payload, _claim = setup
    confirm(setup)
    with store.connect() as con:
        session = json.loads(
            con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
        )
        session["contract"]["requirements"][0]["text"] = "New limit"
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(session), sid))
    assert interfaces.read_interfaces(store, sid, "alice", rid)["records"][0]["stale"]
    assert (
        interfaces.validate_interfaces(service, sid, "alice", rid)[0]["result"]["status"]
        == "unverified"
    )


def test_event_tampering_detected(setup):
    store, sid, _rid, _service, payload, _claim = setup
    interfaces.propose_interface(store, sid, "alice", payload)
    with store.connect() as con:
        con.execute("UPDATE product_interface_events SET document='{}' WHERE session=?", (sid,))
    with pytest.raises(WorkspaceError, match="integrity"):
        interfaces.read_interfaces(store, sid, "alice")


def test_empty_workspace_can_read_interfaces(tmp_path):
    store = Store(tmp_path)
    sid = store.create("alice", "not extracted yet", None)["id"]
    assert interfaces.read_interfaces(store, sid, "alice") == {"version": 0, "records": []}


def test_worker_failure_is_persisted_unverified(setup, monkeypatch):
    from forge_mind.workspace import product_service

    store, sid, rid, service, _payload, _claim = setup
    confirm(setup)

    def fail(*args):
        raise WorkspaceError("worker timeout")

    monkeypatch.setattr(product_service, "run_tool", fail)
    result = interfaces.validate_interfaces(service, sid, "alice", rid)
    assert result[0]["result"]["status"] == "unverified"
    assert "unavailable" in result[0]["reason"]
    assert (
        interfaces.read_interfaces(store, sid, "alice", rid)["records"][0]["last_check"]["reason"]
        == result[0]["reason"]
    )


def test_mount_pitch_source_conversion(setup, monkeypatch):
    store, sid, _rid, service, payload, _claim = setup
    monkeypatch.setattr(
        sources,
        "retrieve",
        lambda *args: (
            b"Mount pitch 4.1338582677 in",
            "text/plain",
            "https://vendor.example/pitch",
        ),
    )
    state = sources.fetch_source(service, sid, "alice", "https://vendor.example/pitch")
    state = sources.propose_claim(
        store,
        sid,
        "alice",
        {
            "source_artifact": state["artifacts"][-1]["id"],
            "quantity": 4.1338582677,
            "unit": "in",
            "exact_quote": "Mount pitch 4.1338582677 in",
            "configuration": "padded fan",
            "datum": "hole centers",
            "locator": "drawing",
        },
    )
    cid = state["claims"][-1]["id"]
    sources.confirm_claim(store, sid, "alice", cid, state["version"])
    payload["request"] = {
        "tool": "mount_grid",
        "units": "mm",
        "body": {"body_index": 0},
        "origin": [0.0, 0.0, 0.0],
        "axis": [0.0, 0.0, 1.0],
        "x_axis": [1.0, 0.0, 0.0],
        "pitch_x": 105.0,
        "pitch_y": 105.0,
        "columns": 2,
        "rows": 2,
        "diameter": 4.0,
    }
    payload["bindings"] = [{"claim_id": cid, "request_path": "/pitch_x"}]
    assert (
        interfaces.propose_interface(store, sid, "alice", payload)["records"][0]["binding_scope"]
        == "source-bound"
    )
    payload["request"]["pitch_x"] = 104.0
    with pytest.raises(WorkspaceError, match="differs"):
        interfaces.propose_interface(store, sid, "alice", payload)


def test_first_preview_contract_then_rejection_stales(setup, monkeypatch):
    import json

    from forge_mind.workspace import product_service

    store, sid, rid, service, payload, _claim = setup
    with store.connect() as con:
        session = json.loads(
            con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
        )
        session["proposal"] = {
            "revision_id": rid,
            "contract": session["contract"],
            "eligible": True,
            "based_on": None,
            "input_version": 0,
        }
        session["contract"] = None
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(session), sid))
    state = confirm(setup)
    assert not state["records"][0]["stale"]
    monkeypatch.setattr(
        product_service,
        "run_tool",
        lambda *args: {"result": {"status": "pass"}, "revision_id": rid},
    )
    assert (
        interfaces.validate_interfaces(service, sid, "alice", rid)[0]["result"]["status"] == "pass"
    )
    other = "c" * 32
    store.add_revision(sid, {"id": other, "contract": store.revision(sid, rid)["contract"]})
    with pytest.raises(WorkspaceError, match="no longer current"):
        interfaces.propose_interface(
            store, sid, "alice", {**payload, "reference_revision_id": other}
        )
    with store.connect() as con:
        session["proposal"] = None
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(session), sid))
    state = interfaces.read_interfaces(store, sid, "alice", rid)
    assert state["records"] and state["records"][0]["stale"]
    assert (
        interfaces.validate_interfaces(service, sid, "alice", rid)[0]["result"]["status"]
        == "unverified"
    )


def test_indexed_body_cannot_follow_reordered_revision(setup, monkeypatch):
    from forge_mind.workspace import product_service

    store, sid, rid, service, _payload, _claim = setup
    confirm(setup)
    newer = "d" * 32
    store.add_revision(sid, {"id": newer, "contract": store.revision(sid, rid)["contract"]})
    calls = []
    monkeypatch.setattr(product_service, "run_tool", lambda *args: calls.append(args))
    record = interfaces.read_interfaces(store, sid, "alice", newer)["records"][0]
    assert record["stale"]
    assert "Body indices" in record["stale_reason"]
    assert (
        interfaces.validate_interfaces(service, sid, "alice", newer)[0]["result"]["status"]
        == "unverified"
    )
    assert not calls


@pytest.mark.parametrize("mutation", ["changed", "missing"])
def test_standalone_read_revalidates_receipt_artifacts(setup, monkeypatch, mutation):
    from forge_mind.validation_provenance import file_sha256
    from forge_mind.workspace import product_service
    from forge_mind.workspace.measure import engine_fingerprint

    store, sid, rid, service, _payload, _claim = setup
    confirm(setup)
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "part.step"
    path.write_text("original export")
    monkeypatch.setattr(
        product_service,
        "run_tool",
        lambda *args: {
            "revision_id": rid,
            "engine_fingerprint": engine_fingerprint(),
            "artifact_hashes": {"part.step": file_sha256(path)},
            "result": {"status": "pass"},
        },
    )
    interfaces.validate_interfaces(service, sid, "alice", rid)
    assert (
        interfaces.read_interfaces(store, sid, "alice", rid)["records"][0]["check_status"] == "pass"
    )
    if mutation == "changed":
        path.write_text("tampered export")
    else:
        path.unlink()
    record = interfaces.read_interfaces(store, sid, "alice", rid)["records"][0]
    assert record["check_status"] == "unverified"
    assert "artifacts" in record["check_reason"]


@pytest.mark.parametrize(
    "selector_payload",
    [
        {"reservation": {"body_index": 0}},
        {"obstacles": [{"occurrence_path": "a"}, {"body_index": 1}]},
    ],
)
def test_all_selector_locations_are_guarded(selector_payload):
    assert interfaces._indexed_selector(selector_payload)


def test_full_assembly_manifest_over_sixteen_files_remains_current(setup, monkeypatch):
    import json

    import build123d as b

    from forge_mind.validation_provenance import file_sha256
    from forge_mind.workspace import product_service
    from forge_mind.workspace.measure import engine_fingerprint

    store, sid, rid, service, _payload, _claim = setup
    confirm(setup)
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True, exist_ok=True)
    bodies = []
    parts = {}
    for index in range(8):
        body = b.Box(index + 1, 2, 3)
        name = f"component{index}"
        b.export_step(body, directory / f"definition_{name}.step")
        b.export_brep(body, directory / f"definition_{name}.brep")
        parts[name] = {"file": f"definition_{name}.brep", "step": f"definition_{name}.step"}
        bodies.append(body.translate((index * 12, 0, 0)))
    b.export_step(b.Compound(children=bodies), directory / "part.step")
    (directory / "parts.json").write_text(json.dumps(parts))
    (directory / "assembly.json").write_text(json.dumps({"parts": parts}))
    hashes = {path.name: file_sha256(path) for path in directory.iterdir() if path.is_file()}
    assert len(hashes) == 19
    monkeypatch.setattr(
        product_service,
        "run_tool",
        lambda *args: {
            "revision_id": rid,
            "engine_fingerprint": engine_fingerprint(),
            "artifact_hashes": hashes,
            "result": {"status": "pass"},
        },
    )
    interfaces.validate_interfaces(service, sid, "alice", rid)
    assert (
        interfaces.read_interfaces(store, sid, "alice", rid)["records"][0]["check_status"] == "pass"
    )
    (directory / "definition_component7.step").write_text("changed definition")
    assert (
        interfaces.read_interfaces(store, sid, "alice", rid)["records"][0]["check_status"]
        == "unverified"
    )
