"""Native worker/caller retention control for mesh failure diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace


def test_native_worker_failure_feedback_reaches_failed_revision(tmp_path, monkeypatch):
    from forge_mind.workspace import document_service
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.models import Contract
    from forge_mind.workspace.processing import process_revision
    from forge_mind.workspace.store import Store

    # Same tiny line-contact fixture as the direct helper control. The real worker
    # compiles it, writes STEP, reloads STEP and triggers the unchanged mesh guard.
    code = """plate = b.Box(20, 20, 1, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
rib = b.Pos(0, -4, 1) * b.Box(12, 2, 5, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
boss = b.Pos(0, 0, 1) * b.Cylinder(3, 5, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
result = plate + rib + boss
"""
    document = DesignDocument.model_validate(
        {
            "features": [
                {"id": "base", "name": "Tangent boss and rib", "kind": "python", "code": code}
            ],
            "outputs": ["base"],
        }
    )
    contract = Contract()
    store = Store(tmp_path / "store")
    session = store.create("owner", "Small tangency diagnostic control", contract.model_dump())
    service = SimpleNamespace(store=store, executor=SimpleNamespace(mode="trusted_local"))
    calls = []

    def real_process(*args, **kwargs):
        calls.append(args[0])
        return process_revision(*args, **kwargs)

    monkeypatch.setattr(document_service, "process_revision", real_process)
    revision = document_service.build_document(
        service,
        session["id"],
        "owner",
        document,
        contract,
        None,
        "Native diagnostic retention control",
        60,
        lambda: False,
    )
    assert len(calls) == 1
    measurement = revision["measurement"]
    assert measurement["geometry_valid"] is False
    prefix = "STEP tessellation has invalid body topology or volume: "
    matching = [error for error in measurement["errors"] if prefix in error]
    assert len(matching) == 1
    error = matching[0]
    assert "CAD worker failed (exit=1)" in error
    diagnostic_text = error.rsplit(prefix, 1)[1].splitlines()[0]
    diagnostic = json.loads(diagnostic_text)
    assert len(prefix + diagnostic_text) <= 1400
    assert diagnostic["body_index"] == 0
    assert diagnostic["frame"] == "input_step_mm"
    assert "not_watertight" in diagnostic["failed_conditions"]
    assert diagnostic["boundary_edges"]["count"] == 0
    assert diagnostic["nonmanifold_edges"]["count"] > 0
    assert diagnostic["edge_inventory_status"] == "available"
    assert diagnostic["nonmanifold_edges"]["sample"]
    assert store.revision(session["id"], revision["id"])["measurement"] == measurement
    assert json.loads((calls[0] / "evidence.json").read_text())["errors"] == measurement["errors"]
    assert not (calls[0] / "processed.json").exists()
    assert not (calls[0] / "tessellation.json").exists()
