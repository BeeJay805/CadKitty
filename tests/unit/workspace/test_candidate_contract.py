"""Candidate continuation must not erase unresolved targets at queue admission."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import run_document_job
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.store import Store

ACCEPTED = "a" * 32
CANDIDATE = "b" * 32
HISTORICAL = "c" * 32


def contract(width: float, unmeasured: bool = True) -> dict[str, Any]:
    requirements = [
        Requirement(
            id="width",
            text=f"Width {width} mm",
            source_quote=f"Width {width} mm",
            metric="extent_x",
            value=width,
        )
    ]
    if unmeasured:
        requirements.append(
            Requirement(
                id="physical_clamping",
                text="Physical clamping",
                source_quote="Physical clamping",
                metric="unmeasured",
            )
        )
    return Contract(
        requirements=requirements, coverage_notes=["Thread engagement remains unqualified"]
    ).model_dump()


def proposed(store: Store, accepted: bool = False) -> str:
    baseline = contract(60, unmeasured=False) if accepted else None
    session = store.create("owner", "Fixture with unresolved clamping requirements", baseline)
    sid = session["id"]
    store.update(
        sid,
        selected_revision=ACCEPTED if accepted else None,
        accepted_revision=ACCEPTED if accepted else None,
        proposal={
            "revision_id": CANDIDATE,
            "contract": contract(80),
            "based_on": ACCEPTED if accepted else None,
            "input_version": 0,
        },
        instructions=["Preserve accepted baseline"],
    )
    return sid


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("explicit_null", [False, True])
@pytest.mark.parametrize("kind", ["document", "instruction"])
def test_candidate_targets_survive_admission_and_reopen(
    tmp_path: Path,
    accepted: bool,
    explicit_null: bool,
    kind: str,
) -> None:
    store = Store(tmp_path)
    sid = proposed(store, accepted)
    payload: dict[str, Any] = {
        "preview_only": True,
        "revision_id": CANDIDATE,
        kind: {} if kind == "document" else "Adjust jaw opening",
    }
    if explicit_null:
        payload["contract"] = None
    store.enqueue(sid, "owner", payload)
    reopened = Store(tmp_path)
    job = reopened.claim()
    assert job and job["contract"] == contract(80)
    assert job["baseline_contract"] == (contract(60, unmeasured=False) if accepted else None)
    assert job["based_on"] == (ACCEPTED if accepted else None)
    assert reopened.session_record(sid, "owner")["proposal"] is None
    assert payload.get("contract") is None  # Caller payload is not mutated.


@pytest.mark.parametrize("replacement", [contract(90), Contract().model_dump()])
def test_explicit_contract_replacement_is_preserved(
    tmp_path: Path, replacement: dict[str, Any]
) -> None:
    store = Store(tmp_path)
    sid = proposed(store, accepted=True)
    store.enqueue(
        sid, "owner", {"preview_only": True, "revision_id": CANDIDATE, "contract": replacement}
    )
    job = store.claim()
    assert job and job["contract"] == replacement
    assert job["baseline_contract"] == contract(60, unmeasured=False)


@pytest.mark.parametrize("source", [ACCEPTED, HISTORICAL, None])
def test_other_edit_sources_keep_accepted_targets(tmp_path: Path, source: str | None) -> None:
    store = Store(tmp_path)
    sid = proposed(store, accepted=True)
    store.add_revision(sid, {"id": HISTORICAL, "contract": contract(10, unmeasured=False)})
    store.enqueue(sid, "owner", {"preview_only": True, "revision_id": source})
    job = store.claim()
    assert job and "contract" not in job
    assert job["baseline_contract"] == contract(60, unmeasured=False)
    assert store.session_record(sid, "owner")["contract"] == contract(60, unmeasured=False)


def test_real_document_job_receives_failed_and_unmeasured_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store(tmp_path)
    sid = proposed(store)
    captured: list[Contract] = []

    def capture_build(
        service: Any,
        session: str,
        owner: str,
        document: DesignDocument,
        requirements: Contract,
        *args: Any,
    ) -> Any:
        captured.append(requirements)
        # Stop before kernel work; the assertion exercises actual admission,
        # claim, original Contract validation and document-job argument wiring.
        raise RuntimeError("Captured contract before CAD execution")

    monkeypatch.setattr("forge_mind.workspace.document_service.build_document", capture_build)
    store.enqueue(
        sid,
        "owner",
        {"preview_only": True, "revision_id": CANDIDATE, "document": DesignDocument().model_dump()},
    )
    job = store.claim()
    assert job
    service = SimpleNamespace(store=store, stop=threading.Event())
    run_document_job(service, job)
    assert len(captured) == 1
    assert captured[0].model_dump() == contract(80)
    assert captured[0].requirements[0].value == 80
    assert captured[0].requirements[1].metric == "unmeasured"
    assert store.session_record(sid, "owner")["contract"] is None
