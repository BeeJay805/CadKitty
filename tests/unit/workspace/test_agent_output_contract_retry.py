"""Scripted correction preserves requests, observations and current owner authority."""

import json
from copy import deepcopy

import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.output_contract import OutputContractError
from forge_mind.workspace import agent, author_images
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, Requirement
from forge_mind.workspace.store import Store


def run_author(
    tmp_path, monkeypatch, *, native=True, image_count=0, fail=True, mutation=None, prepare=None
):
    images = author_images.AuthorImages(
        images=tuple(bytes([index + 1]) for index in range(image_count)),
        labels=tuple(f"View {index}" for index in range(image_count)),
        revision_id=None,
        mode="scripted-correction",
        supplemental_status="none",
    )
    monkeypatch.setattr(author_images, "prepare_author_images", lambda *args, **kwargs: images)
    calls = []
    store = Store(tmp_path)
    contract = Contract(requirements=[Requirement(id="fit", text="A part", source_quote="A part")])
    sid = store.create(
        "owner", "A part", contract.model_dump(), "native_operations" if native else "hybrid"
    )["id"]

    class Author:
        model = "scripted-output-correction"
        timeout_s = None
        last_call_diagnostics = None

        def generate_structured(self, **kwargs):
            calls.append(dict(kwargs))
            index = len(calls)
            self.last_call_diagnostics = {
                "call_id": f"output-attempt-{index}",
                "outcome": "invalid_response" if fail and index == 1 else "success",
            }
            if fail and index == 1:
                if mutation is not None:
                    mutation(store, sid)
                raise OutputContractError(
                    stage="pydantic_validation",
                    schema=kwargs["output_model"].model_json_schema(),
                    violations=[
                        {"loc": ["explanation"], "type": "missing", "input": "SECRET_INPUT_42"}
                    ],
                    retryable=True,
                )
            return Action(
                kind="schema", query="feature:box", explanation="Read syntax once"
            ), LLMUsage(self.model, 0, 0)

        generate_structured_with_image = generate_structured
        generate_structured_with_images = generate_structured

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Author())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(sid, "owner", {"max_seconds": 120, "max_actions": 1})
    job = store.claim()
    assert job is not None
    if prepare is not None:
        prepare(store, sid)
    state = service._run_agent_job(job)
    return store, sid, calls, images, state


@pytest.mark.parametrize("image_count", [0, 1, 2])
@pytest.mark.parametrize("native", [False, True])
def test_every_input_path_keeps_valid_response_at_one_call(
    tmp_path, monkeypatch, native, image_count
):
    store, sid, calls, _, state = run_author(
        tmp_path, monkeypatch, native=native, image_count=image_count, fail=False
    )
    assert state != "error" and len(calls) == 1
    assert (
        len([event for event in store.get(sid, "owner")["events"] if event["kind"] == "action"])
        == 1
    )


@pytest.mark.parametrize("image_count", [0, 1, 2])
@pytest.mark.parametrize("native", [False, True])
def test_correction_uses_original_request_and_separate_attempt_observations(
    tmp_path, monkeypatch, native, image_count
):
    store, sid, calls, images, state = run_author(
        tmp_path, monkeypatch, native=native, image_count=image_count
    )
    eligible = native or image_count == 0
    assert len(calls) == (2 if eligible else 1)
    if eligible:
        assert state != "error"
        assert {key: value for key, value in calls[0].items() if key != "system"} == {
            key: value for key, value in calls[1].items() if key != "system"
        }
        assert calls[1]["system"].startswith(calls[0]["system"])
        assert calls[1]["system"] != calls[0]["system"]
        assert "SECRET_INPUT_42" not in calls[1]["system"]
    events = store.get(sid, "owner")["events"]
    assert len([event for event in events if event["kind"] == "action"]) == int(eligible)
    assert len([event for event in events if event["kind"] == "model_call"]) == len(calls)
    retries = [event for event in events if event["kind"] == "model_retry"]
    assert len(retries) == int(eligible)
    assert all(event["reason"] == "output_contract" for event in retries)
    assert all(
        "timed out" not in event["message"].lower() and "busy" not in event["message"].lower()
        for event in retries
    )
    assert "SECRET_INPUT_42" not in json.dumps(
        [event for event in events if event["kind"] in {"model_call", "model_retry"}]
    )
    if image_count:
        observations = [event for event in events if event["kind"] == "image_observation"]
        assert len(observations) == len(calls)
        assert len({event["invocation_id"] for event in observations}) == len(calls)
        assert all(event["images"] == images.records() for event in observations)
        assert [event["response_status"] for event in observations] == (
            ["failed_or_unknown", "complete"] if eligible else ["failed_or_unknown"]
        )
    assert store.revisions(sid) == []
    assert store.get(sid, "owner").get("accepted_revision") is None


@pytest.mark.parametrize(
    "change",
    ["acceptance", "lease", "source", "contract", "selection", "needs_input", "cancellation"],
)
def test_changed_owner_authority_prevents_corrected_dispatch(tmp_path, monkeypatch, change):
    def mutate(store, sid):
        if change == "acceptance":
            store.update(sid, accepted_revision="a" * 32)
        elif change == "lease":
            with store.connect() as con:
                con.execute("UPDATE jobs SET worker_pid=-1 WHERE session=?", (sid,))
        elif change == "source":
            store.update(sid, prompt="Changed owner source")
        elif change == "contract":
            store.update(sid, contract=Contract().model_dump())
        elif change == "selection":
            store.update(sid, selected_revision="b" * 32)
        elif change == "needs_input":
            store.update(sid, intent_attribution={"status": "needs_input"})
        else:
            store.cancel(sid, "owner")

    _, _, calls, _, state = run_author(tmp_path, monkeypatch, mutation=mutate)
    assert len(calls) == 1
    assert state in {"error", "cancelled"}


def test_changed_workflow_prevents_corrected_dispatch(tmp_path, monkeypatch):
    workflow = {"status": "not_prepared", "construction_authorized": False}
    monkeypatch.setattr(
        "forge_mind.workspace.customer_construction.author_workflow_context",
        lambda *args: deepcopy(workflow),
    )
    _, _, calls, _, state = run_author(
        tmp_path, monkeypatch, mutation=lambda *args: workflow.update(status="stale")
    )
    assert len(calls) == 1 and state == "error"


def test_initial_pending_contract_intake_keeps_existing_valid_behavior(tmp_path, monkeypatch):
    store = Store(tmp_path)
    sid = store.create("owner", "A new part", None)["id"]
    calls = []

    class Author:
        model = "scripted-pending-intake"

        def generate_structured(self, **kwargs):
            calls.append(kwargs)
            value = (
                Contract()
                if kwargs["output_model"] is Contract
                else Action(kind="finish", explanation="No geometry produced")
            )
            return value, LLMUsage(self.model, 0, 0)

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Author())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(sid, "owner", {"max_seconds": 120, "max_actions": 1})
    assert service._run_agent_job(store.claim()) != "error"
    assert len(calls) == 2


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("failure_kind", ["value", "type", "key", "database"])
def test_snapshot_failure_disables_correction_without_blocking_original_call(
    tmp_path, monkeypatch, fail, failure_kind
):
    def prepare(store, sid):
        def unavailable(doc):
            if failure_kind == "database":
                with store.connect() as con:
                    con.execute("SELECT version FROM missing_optional_correction_metadata")
            error = {"value": ValueError, "type": TypeError, "key": KeyError}[failure_kind]
            raise error("Optional recovery snapshot unavailable")

        monkeypatch.setattr(store, "_native_author_binding", unavailable)

    store, sid, calls, _, state = run_author(tmp_path, monkeypatch, fail=fail, prepare=prepare)
    assert len(calls) == 1
    assert (state == "error") == fail
    assert not any(event["kind"] == "model_retry" for event in store.get(sid, "owner")["events"])


@pytest.mark.parametrize("signal", [SystemExit, KeyboardInterrupt])
def test_optional_snapshot_does_not_swallow_process_signals(tmp_path, monkeypatch, signal):
    def prepare(store, sid):
        def interrupted(doc):
            raise signal("Stop the current operation")

        monkeypatch.setattr(store, "_native_author_binding", interrupted)

    with pytest.raises(signal):
        run_author(tmp_path, monkeypatch, prepare=prepare)


def test_authority_change_during_correction_factory_is_checked_before_dispatch(
    tmp_path, monkeypatch
):
    original = agent.response_correction

    def prepare(store, sid):
        def changed(arguments, error):
            corrected = original(arguments, error)
            store.update(sid, prompt="Changed after correction feedback was prepared")
            return corrected

        monkeypatch.setattr(agent, "response_correction", changed)

    _, _, calls, _, state = run_author(tmp_path, monkeypatch, prepare=prepare)
    assert len(calls) == 1 and state == "error"
