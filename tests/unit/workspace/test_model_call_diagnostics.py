"""Call evidence survives transport failure and a fresh workspace read."""

from types import SimpleNamespace

import pytest

from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import agent, author_images
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract
from forge_mind.workspace.store import Store


def run_client(tmp_path, client):
    store = Store(tmp_path)
    sid = store.create("owner", "Review available construction tools", Contract().model_dump())[
        "id"
    ]
    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: client)
    jid = store.enqueue(sid, "owner", {"max_actions": 3, "max_seconds": 120})
    service.run_job(store.claim())
    return Store(tmp_path).get(sid, "owner"), jid


class Client:
    model = "gpt-6.1-sol"
    last_call_diagnostics = None

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def generate_structured(self, **kwargs):
        outcome = next(self.outcomes)
        self.calls += 1
        self.last_call_diagnostics = {
            "call_id": f"controlled-{self.calls}",
            "outcome": outcome,
            "requested_model": self.model,
            "events_bytes": 42,
        }
        if outcome == "timeout":
            raise LLMError("Codex call cancelled or exceeded its wall-clock limit")
        if outcome == "nonzero_exit":
            raise LLMError("Controlled provider failure")
        return Action(kind="finish", explanation="No geometry has been created"), LLMUsage(
            self.model, 0, 0
        )


def test_retry_records_each_attempt_before_retry_event(tmp_path):
    client = Client(["timeout", "success"])
    session, jid = run_client(tmp_path, client)
    events = [e for e in session["events"] if e["kind"] in {"model_call", "model_retry"}]
    assert [e["kind"] for e in events] == ["model_call", "model_retry", "model_call"]
    assert [events[i]["diagnostics"]["outcome"] for i in (0, 2)] == ["timeout", "success"]
    assert events[0]["job_id"] == events[2]["job_id"] == jid
    assert events[2]["message"] == "Assistant response received"
    assert client.calls == 2
    client.last_call_diagnostics["events_bytes"] = 999
    assert events[2]["diagnostics"]["events_bytes"] == 42
    assert not session.get("accepted_revision")


def test_terminal_error_retains_evidence_without_acceptance(tmp_path):
    client = Client(["nonzero_exit"])
    session, _ = run_client(tmp_path, client)
    calls = [e for e in session["events"] if e["kind"] == "model_call"]
    assert len(calls) == 1
    assert calls[0]["diagnostics"]["outcome"] == "nonzero_exit"
    assert session["state"] == "error"
    assert not session.get("accepted_revision")


@pytest.mark.parametrize("outcome", ["success", "nonzero_exit"])
def test_image_action_path_retains_call_evidence(tmp_path, monkeypatch, outcome):
    client = Client([outcome])
    monkeypatch.setattr(
        author_images,
        "prepare_author_images",
        lambda *a, **kw: SimpleNamespace(images=(b"controlled image",), context=lambda: {}),
    )
    invoked = []

    def invoke(actual_client, images, args, **kwargs):
        assert actual_client is client
        invoked.append(images.images)
        return client.generate_structured(**args)

    monkeypatch.setattr(author_images, "invoke_author_images", invoke)
    session, jid = run_client(tmp_path, client)
    assert invoked == [(b"controlled image",)]
    calls = [e for e in session["events"] if e["kind"] == "model_call"]
    assert len(calls) == 1
    assert calls[0]["job_id"] == jid
    assert calls[0]["diagnostics"]["outcome"] == outcome


def test_legacy_client_does_not_fabricate_evidence(tmp_path):
    client = SimpleNamespace(
        generate_structured=lambda **kw: (
            Action(kind="finish", explanation="No geometry created"),
            LLMUsage("test", 0, 0),
        )
    )
    session, _ = run_client(tmp_path, client)
    assert not any(e["kind"] == "model_call" for e in session["events"])


def test_telemetry_write_failure_preserves_timeout_retry(tmp_path, monkeypatch, caplog):
    original = Store.event

    def event(store, sid, kind, message, **data):
        if kind == "model_call":
            raise OSError("private storage sentinel")
        return original(store, sid, kind, message, **data)

    monkeypatch.setattr(Store, "event", event)
    client = Client(["timeout", "success"])
    session, _ = run_client(tmp_path, client)
    assert client.calls == 2
    assert session["state"] == "needs_review"
    assert any(e["kind"] == "model_retry" for e in session["events"])
    assert "Could not persist assistant call diagnostics" in caplog.text
    assert "private storage sentinel" not in caplog.text


@pytest.mark.parametrize("outcomes", [["success"], ["timeout", "success"]])
def test_optional_getter_failure_preserves_result_and_retry(tmp_path, caplog, outcomes):
    class BrokenGetterClient(Client):
        @property
        def last_call_diagnostics(self):
            raise OSError("private diagnostic sentinel")

        @last_call_diagnostics.setter
        def last_call_diagnostics(self, value):
            pass

    client = BrokenGetterClient(outcomes)
    session, _ = run_client(tmp_path, client)
    assert client.calls == len(outcomes)
    assert session["state"] == "needs_review"
    assert "Could not persist assistant call diagnostics" in caplog.text
    assert "private diagnostic sentinel" not in caplog.text
