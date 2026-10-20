"""Lifecycle evidence controls without provider access or model responses."""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from forge_mind.llm import codex_client as transport
from forge_mind.llm.client import LLMError
from forge_mind.llm.codex_diagnostics import EVENT_LIMIT, EVENT_TYPES, LINE_LIMIT, LOG_LIMIT

SECRET = "SECRET_SENTINEL_not_to_retain"


class Response(BaseModel):
    answer: str


def arguments():
    return {
        "system": SECRET,
        "user": SECRET,
        "output_model": Response,
        "model": "gpt-6.1-sol",
        "max_tokens": 30,
    }


def assert_safe(client):
    record = client.last_call_diagnostics
    assert record is not None
    assert SECRET not in json.dumps(record)
    assert set(record) == {
        "call_id",
        "requested_model",
        "outcome",
        "elapsed_seconds",
        "process_started",
        "returncode",
        "events_bytes",
        "stderr_bytes",
        "answer_bytes",
        "event_counts",
        "events_parse_truncated",
        "diagnostics_complete",
        "collection_error",
        "cleanup_error",
        "output_contract",
    }
    assert set(record["event_counts"]) == {*EVENT_TYPES, "other"}
    assert record["elapsed_seconds"] >= 0
    for key in ("events_bytes", "stderr_bytes", "answer_bytes"):
        assert isinstance(record[key], int) and record[key] >= 0
    return record


@pytest.fixture
def process_control(monkeypatch):
    state = SimpleNamespace(
        answer=b'{"answer":"safe"}',
        events=(
            b'{"type":"thread.started","thread_id":"' + SECRET.encode() + b'"}\n'
            b'{"type":"turn.completed","usage":{"input_tokens":12,"cached_input_tokens":2,"output_tokens":3}}\n'
        ),
        stderr=SECRET.encode(),
        exit_code=0,
        alive=False,
        spawn_error=False,
        wait_error=False,
        roots=[],
        killed=[],
    )
    monkeypatch.setattr(transport, "_command", lambda: ["controlled-cli"])

    def launch(command, **kwargs):
        root = Path(kwargs["cwd"])
        state.roots.append(root)
        assert client.last_call_diagnostics is None
        if state.spawn_error:
            raise OSError(SECRET)
        if state.answer is not None:
            (root / "answer.json").write_bytes(state.answer)
        kwargs["stdout"].write(state.events)
        kwargs["stdout"].flush()
        kwargs["stderr"].write(state.stderr)
        kwargs["stderr"].flush()
        proc = SimpleNamespace(pid=999999, stdin=io.BytesIO(), returncode=None)

        def poll():
            if not state.alive:
                proc.returncode = state.exit_code
            return proc.returncode

        def wait(**kwargs):
            if state.wait_error:
                raise OSError(SECRET)
            return poll()

        proc.poll, proc.wait = poll, wait
        return proc

    def kill(pid):
        state.killed.append(pid)
        if not state.wait_error:
            state.alive, state.exit_code = False, -9

    monkeypatch.setattr(transport.subprocess, "Popen", launch)
    monkeypatch.setattr("forge_mind.agentic.sandbox._kill_tree", kill)
    client = transport.CodexClient()
    client.model = "gpt-6.1-sol"
    return client, state


@pytest.mark.parametrize(
    "outcome", ["success", "invalid_response", "nonzero_exit", "spawn_failure", "output_overflow"]
)
def test_all_terminal_outcomes_retain_only_safe_evidence_before_cleanup(process_control, outcome):
    client, state = process_control
    if outcome == "invalid_response":
        state.answer = ('{"answer":42,"secret":"' + SECRET + '"}').encode()
    elif outcome == "nonzero_exit":
        state.exit_code = 7
        state.events += ('{"type":"turn.failed","error":{"message":"' + SECRET + '"}}\n').encode()
    elif outcome == "spawn_failure":
        state.spawn_error = True
    elif outcome == "output_overflow":
        state.answer = b"x" * 512_001
    if outcome == "success":
        result, usage = client.generate_structured(**arguments())
        assert result.answer == "safe"
        assert (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cost_usd,
        ) == (
            10,
            3,
            2,
            0,
        )
    else:
        with pytest.raises((LLMError, OSError)):
            client.generate_structured(**arguments())
    record = assert_safe(client)
    assert record["outcome"] == outcome
    assert record["process_started"] == (outcome != "spawn_failure")
    assert record["returncode"] == (None if outcome == "spawn_failure" else state.exit_code)
    assert not record["cleanup_error"]
    assert all(not root.exists() for root in state.roots)
    if outcome != "spawn_failure":
        assert record["events_bytes"] == len(state.events)
        assert record["stderr_bytes"] == len(state.stderr)
        assert record["answer_bytes"] == len(state.answer)
        assert record["event_counts"]["thread.started"] == 1
        assert record["event_counts"]["turn.completed"] == 1


@pytest.mark.parametrize("outcome", ["timeout", "cancelled", "output_overflow"])
def test_running_process_failure_collects_after_termination_and_resets_next_call(
    process_control, outcome
):
    client, state = process_control
    client.generate_structured(**arguments())
    previous = client.last_call_diagnostics
    state.alive = True
    if outcome == "timeout":
        client.timeout_s = 0.05
    elif outcome == "cancelled":
        checks = iter((False, True))
        client.cancelled = lambda: next(checks)
    else:
        state.stderr = b"x" * (LOG_LIMIT + 17)
    with pytest.raises(LLMError):
        client.generate_structured(**arguments())
    record = assert_safe(client)
    assert record["outcome"] == outcome
    assert record["call_id"] != previous["call_id"]
    assert record["returncode"] == -9
    assert len(state.killed) == 1
    assert all(not root.exists() for root in state.roots)
    assert record["stderr_bytes"] == len(state.stderr)
    record["event_counts"]["thread.started"] = 99
    assert client.last_call_diagnostics["event_counts"]["thread.started"] == 1


@pytest.mark.parametrize("mode", ["malformed", "long_line", "event_count"])
def test_event_parser_is_bounded_and_never_retains_unknown_types(process_control, mode):
    client, state = process_control
    if mode == "malformed":
        state.events += b'not JSON\n[]\nnull\n{"type":[]}\n'
        state.events += ('{"type":"' + SECRET + '","payload":"' + SECRET + '"}\n').encode()
    elif mode == "long_line":
        state.events += b"x" * (LINE_LIMIT + 1) + b"\n"
    else:
        state.events = b'{"type":"turn.started"}\n' * (EVENT_LIMIT + 1)
    client.generate_structured(**arguments())
    record = assert_safe(client)
    assert record["outcome"] == "success"
    assert not record["diagnostics_complete"]
    assert record["events_parse_truncated"] == (mode != "malformed")
    assert sum(record["event_counts"].values()) <= EVENT_LIMIT
    if mode == "malformed":
        assert record["event_counts"]["other"] == 5


def test_diagnostic_read_failure_does_not_replace_success(process_control, monkeypatch):
    client, _ = process_control
    original = Path.open

    def open_control(path, mode="r", *args, **kwargs):
        if path.name == "events.jsonl" and mode == "rb":
            raise PermissionError(SECRET)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_control)
    result, _ = client.generate_structured(**arguments())
    assert result.answer == "safe"
    record = assert_safe(client)
    assert record["outcome"] == "success"
    assert not record["diagnostics_complete"]
    assert record["collection_error"] == "read_failed"


def test_image_bytes_and_labels_are_not_in_lifecycle_evidence(process_control):
    client, _ = process_control
    client.generate_structured_with_images(
        **arguments(), images_png=[SECRET.encode()], image_labels=[SECRET]
    )
    assert_safe(client)
    assert client.last_input_images[0]["label"] == SECRET


def test_invalid_wire_map_keeps_existing_exception_and_invalid_response_outcome(process_control):
    client, state = process_control

    class MapResponse(BaseModel):
        values: dict[str, float]

    state.answer = b'{"values":[{"key":"width"}]}'
    with pytest.raises(KeyError):
        client.generate_structured(**{**arguments(), "output_model": MapResponse})
    assert assert_safe(client)["outcome"] == "invalid_response"


def test_cleanup_failure_keeps_primary_outcome_and_unknown_returncode(process_control):
    client, state = process_control
    checks = iter((False, True))
    client.cancelled = lambda: next(checks)
    state.alive, state.wait_error = True, True
    with pytest.raises(OSError, match=SECRET):
        client.generate_structured(**arguments())
    record = assert_safe(client)
    assert record["outcome"] == "cancelled"
    assert record["cleanup_error"]
    assert record["returncode"] is None
    assert not record["diagnostics_complete"]
    assert record["collection_error"] == "process_state_unknown"
    assert all(not root.exists() for root in state.roots)


@pytest.mark.parametrize("outcome", ["success", "invalid_response"])
def test_temporary_cleanup_failure_is_visible_without_erasing_collected_outcome(
    process_control, monkeypatch, outcome
):
    client, state = process_control
    if outcome == "invalid_response":
        state.answer = b"not JSON"
    original = transport.tempfile.TemporaryDirectory

    @contextmanager
    def failing_cleanup(**kwargs):
        try:
            with original(**kwargs) as root:
                yield root
        finally:
            raise OSError(SECRET)

    monkeypatch.setattr(transport.tempfile, "TemporaryDirectory", failing_cleanup)
    with pytest.raises(OSError, match=SECRET):
        client.generate_structured(**arguments())
    record = assert_safe(client)
    assert record["outcome"] == outcome
    assert record["cleanup_error"]
    assert record["events_bytes"] == len(state.events)
    assert all(not root.exists() for root in state.roots)


@pytest.mark.parametrize("mode", ["missing_answer", "setup_failure", "cancelled", "input_rejected"])
def test_prelaunch_and_missing_answer_failures_replace_previous_evidence(
    process_control, monkeypatch, mode
):
    client, state = process_control
    client.generate_structured(**arguments())
    previous = client.last_call_diagnostics
    if mode == "setup_failure":

        def missing_command():
            raise LLMError(SECRET)

        monkeypatch.setattr(transport, "_command", missing_command)
    elif mode == "cancelled":
        client.cancelled = lambda: True
    elif mode == "missing_answer":
        state.answer = None
    with pytest.raises(LLMError):
        if mode == "input_rejected":
            client.generate_structured_with_images(**arguments(), images_png=None, image_labels=[])
        else:
            client.generate_structured(**arguments())
    record = assert_safe(client)
    assert record["call_id"] != previous["call_id"]
    assert record["outcome"] == ("invalid_response" if mode == "missing_answer" else mode)
    assert record["process_started"] == (mode == "missing_answer")
    assert all(not root.exists() for root in state.roots)


@pytest.mark.parametrize("outcome", ["success", "timeout"])
def test_real_short_child_retains_evidence_after_temporary_directory_cleanup(monkeypatch, outcome):
    roots = []
    original = transport.subprocess.Popen
    script = (
        "import json,pathlib,sys,time; "
        "root=pathlib.Path.cwd(); "
        "print(json.dumps({'type':'turn.started','secret':sys.argv[1]}),flush=True); "
        "print(sys.argv[1],file=sys.stderr,flush=True); "
        + (
            "time.sleep(30)"
            if outcome == "timeout"
            else "(root/'answer.json').write_text('{\"answer\":\"safe\"}'); sys.stdin.read()"
        )
    )

    def launch(command, **kwargs):
        roots.append(Path(kwargs["cwd"]))
        return original([sys.executable, "-B", "-c", script, SECRET], **kwargs)

    monkeypatch.setattr(transport, "_command", lambda: ["never-invoked-codex"])
    monkeypatch.setattr(transport.subprocess, "Popen", launch)
    client = transport.CodexClient(timeout_s=0.8 if outcome == "timeout" else 5)
    if outcome == "success":
        assert client.generate_structured(**arguments())[0].answer == "safe"
    else:
        with pytest.raises(LLMError, match="wall-clock"):
            client.generate_structured(**arguments())
    record = assert_safe(client)
    assert record["outcome"] == outcome
    assert record["event_counts"]["turn.started"] == 1
    assert record["stderr_bytes"] > 0
    assert record["returncode"] is not None
    assert not record["cleanup_error"]
    assert all(not root.exists() for root in roots)
