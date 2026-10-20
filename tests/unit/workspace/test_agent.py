from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from build123d import Box, export_step, export_stl

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.measure import measure_revision
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.store import Store

CODE = 'from build123d import Box\nPARAMS = {"width": 10.0, "depth": 8.0, "height": 4.0}\nresult = Box(PARAMS["width"], PARAMS["depth"], PARAMS["height"])\n'


def width_contract(width: float) -> Contract:
    return Contract(
        requirements=[
            Requirement(
                id="width",
                text=f"Width {width} mm",
                source_quote=f"{width} mm",
                source="explicit",
                metric="extent_x",
                value=width,
            )
        ]
    )


def no_model(**kwargs: Any) -> Any:
    class NeverCall:
        def generate_structured(self, **kwargs: Any) -> Any:
            raise AssertionError("A deterministic restore or parameter edit must not call a model")

    return NeverCall()


@pytest.fixture
def seeded(tmp_path: Path) -> tuple[WorkspaceService, str, str]:
    store = Store(tmp_path)
    c = width_contract(10)
    sid = store.create("owner", "10 mm wide block", c.model_dump())["id"]
    rid = uuid.uuid4().hex
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    (directory / "program.py").write_text(CODE, encoding="utf-8")
    export_step(Box(10, 8, 4), directory / "part.step")
    export_stl(Box(10, 8, 4), directory / "part.stl")
    measured = measure_revision(directory, rid, c)
    store.add_revision(
        sid,
        {"id": rid, "code": CODE, "contract": c.model_dump(), "measurement": measured.model_dump()},
    )
    store.update(sid, selected_revision=rid, best_revision=rid)
    return WorkspaceService(store, Executor("trusted_local"), no_model), sid, rid


def run(service: WorkspaceService, sid: str, **payload: Any) -> dict[str, Any]:
    service.store.enqueue(sid, "owner", {"max_actions": 2, "max_seconds": 60, **payload})
    job = service.store.claim()
    assert job
    service.run_job(job)
    return service.store.get(sid, "owner")


def test_restore_does_not_approve_a_new_contract(seeded: tuple[WorkspaceService, str, str]) -> None:
    service, sid, old = seeded
    before = service.store.revision(sid, old)
    service.store.update(sid, contract=width_contract(20).model_dump())
    result = run(service, sid, revision_id=old, restore_only=True)
    assert result["state"] == "needs_review", result["events"]
    assert result["selected_revision"] != old
    assert service.store.revision(sid, old) == before
    latest = result["revisions"][-1]
    assert latest["measurement"]["geometry_valid"], latest["measurement"]["errors"]
    assert latest["measurement"]["findings"][0]["status"] == "fail"
    assert latest["measurement"]["facts"]["mesh_source"].startswith("Trusted tessellation")


def test_parameter_edit_preserves_requested_candidate_and_prior_best(
    seeded: tuple[WorkspaceService, str, str],
) -> None:
    service, sid, old = seeded
    result = run(service, sid, revision_id=old, parameters={"width": 12.0})
    assert result["state"] == "needs_review", result["events"]
    assert result["selected_revision"] != old
    assert result["best_revision"] == old
    latest = result["revisions"][-1]
    assert latest["parameters"]["width"] == 12
    assert latest["measurement"]["findings"][0]["measured"] == pytest.approx(12)
    assert "width" in latest["editable_parameters"]
    assert service.store.revision(sid, old)["code"] == CODE


def test_tampered_revision_cannot_be_restored(seeded: tuple[WorkspaceService, str, str]) -> None:
    service, sid, old = seeded
    (service.store.directory(sid, old) / "program.py").write_text("changed", encoding="utf-8")
    result = run(service, sid, revision_id=old, restore_only=True)
    assert result["state"] == "error"
    assert "stale evidence" in result["events"][-1]["message"]


def test_failed_build_preserves_source_and_unverified_requirements(
    seeded: tuple[WorkspaceService, str, str],
) -> None:
    service, sid, old = seeded
    result = service.build(
        sid,
        "raise ValueError('bad model')",
        width_contract(10),
        old,
        "bad candidate",
        30,
        lambda: False,
    )
    assert not result["measurement"]["geometry_valid"]
    assert result["measurement"]["findings"][0]["status"] == "unverified"
    assert "program.py" in result["measurement"]["artifact_hashes"]
    assert service.store.revision(sid, old)["measurement"]["requirements_satisfied"]


def test_invalid_parameter_candidate_cannot_report_prior_success(
    seeded: tuple[WorkspaceService, str, str],
) -> None:
    service, sid, old = seeded
    result = run(service, sid, revision_id=old, parameters={"width": 0.0})
    assert result["state"] == "needs_review"
    assert result["selected_revision"] != old
    assert result["best_revision"] == old
    assert not result["revisions"][-1]["measurement"]["geometry_valid"]
    # The failed source remains inspectable and can be the basis of a later repair.
    assert service._load_revision(sid, result["selected_revision"])["code"]
