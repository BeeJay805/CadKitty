"""Image transport is exercised with process mocks; no provider requests are made."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from forge_mind.llm.client import LLMError

MODULE = Path(__file__).resolve().parents[3] / "src/forge_mind/llm/codex_client.py"
SPEC = importlib.util.spec_from_file_location("codex_image_transport_under_test", MODULE)
assert SPEC and SPEC.loader
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


class Response(BaseModel):
    answer: str


class Prompt(io.BytesIO):
    def close(self):
        self.saved = self.getvalue()
        super().close()


@pytest.fixture
def invocation(monkeypatch):
    launches = []
    state = SimpleNamespace(exit_code=0, answer={"answer": "valid"}, killed=[])
    monkeypatch.setattr(transport, "_command", lambda: ["codex-process-mock"])

    def launch(command, **kwargs):
        images = [
            Path(command[index + 1]) for index, part in enumerate(command) if part == "--image"
        ]
        prompt = Prompt()
        process = SimpleNamespace(
            pid=999999,
            returncode=state.exit_code,
            stdin=prompt,
            poll=lambda: state.exit_code,
            wait=lambda **_: state.exit_code,
        )
        launches.append(
            {
                "command": command,
                "images": [path.read_bytes() for path in images],
                "paths": images,
                "prompt": prompt,
                "kwargs": kwargs,
            }
        )
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps(state.answer), encoding="utf-8"
        )
        kwargs["stdout"].write(
            b'{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":3}}\n'
        )
        kwargs["stdout"].flush()
        return process

    monkeypatch.setattr(transport.subprocess, "Popen", launch)
    monkeypatch.setattr("forge_mind.agentic.sandbox._kill_tree", state.killed.append)
    return launches, state


def arguments():
    return {
        "system": "Trusted system",
        "user": "Task",
        "output_model": Response,
        "model": "gpt-6-astra",
        "max_tokens": 50,
    }


def images(client, **overrides):
    options = {
        "images_png": [b"full-png", b"inspection-png"],
        "image_labels": [
            "Full context: all exported bodies",
            "Inspection: declared references omitted",
        ],
    }
    return client.generate_structured_with_images(**arguments(), **{**options, **overrides})


def test_two_images_in_one_request_keep_order_labels_and_exact_hashes(invocation, monkeypatch):
    launches, _ = invocation
    monkeypatch.setenv("OPENAI_API_KEY", "test-credential")
    client = transport.CodexClient()
    result, usage = images(client)
    assert result.answer == "valid"
    assert usage.input_tokens == 12 and usage.output_tokens == 3
    assert len(launches) == 1
    launched = launches[0]
    assert launched["images"] == [b"full-png", b"inspection-png"]
    assert [item["label"] for item in client.last_input_images] == [
        "Full context: all exported bodies",
        "Inspection: declared references omitted",
    ]
    for index, (content, metadata) in enumerate(
        zip(launched["images"], client.last_input_images, strict=True), start=1
    ):
        assert metadata == {
            "index": index,
            "label": metadata["label"],
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    prompt = launched["prompt"].saved.decode()
    assert json.dumps(client.last_input_images) in prompt
    assert "not validation evidence" in prompt
    assert "test-credential" not in prompt
    assert "OPENAI_API_KEY" not in launched["kwargs"]["env"]
    assert all(not path.exists() for path in launched["paths"])
    assert launched["command"][-1] == "-"
    assert launched["command"].count("--image") == 2
    # Callers cannot mutate the client's submission record through the property.
    client.last_input_images[0]["sha256"] = "changed"
    assert client.last_input_images[0]["sha256"] == hashlib.sha256(b"full-png").hexdigest()


def test_legacy_single_image_and_text_calls_remain_compatible(invocation):
    launches, _ = invocation
    client = transport.CodexClient()
    client.generate_structured_with_image(**arguments(), image_png=b"legacy-image")
    assert launches[0]["images"] == [b"legacy-image"]
    assert launches[0]["paths"][0].name == "input.png"
    assert "Attached image order" not in launches[0]["prompt"].saved.decode()
    assert client.last_input_images == (
        {
            "index": 1,
            "label": "Primary image",
            "sha256": hashlib.sha256(b"legacy-image").hexdigest(),
            "bytes": 12,
        },
    )
    client.generate_structured(**arguments())
    assert launches[1]["images"] == []
    assert client.last_input_images == ()


@pytest.mark.parametrize(
    "options",
    [
        {"images_png": []},
        {"images_png": None},
        {"images_png": [b"full", None]},
        {"images_png": [b"full", b""]},
        {"images_png": [b"full", bytearray(b"mutable")]},
        {"image_labels": ["only one"]},
        {"image_labels": ["full", " "]},
        {"image_labels": ["full", 2]},
    ],
)
def test_missing_image_or_label_never_launches_partial_request(invocation, options):
    launches, _ = invocation
    client = transport.CodexClient()
    with pytest.raises(LLMError):
        images(client, **options)
    assert launches == []
    assert client.last_input_images == ()


def test_supplementary_file_write_failure_never_launches_primary_alone(invocation, monkeypatch):
    launches, _ = invocation
    original = Path.write_bytes

    def write(path, data):
        if path.name == "input-2.png":
            raise OSError("Controlled write failure")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", write)
    client = transport.CodexClient()
    with pytest.raises(OSError, match="Controlled write failure"):
        images(client)
    assert launches == [] and client.last_input_images == ()


@pytest.mark.parametrize("failure", ["provider_exit", "invalid_response"])
def test_failed_launched_call_keeps_input_record_and_never_retries(invocation, failure):
    launches, state = invocation
    if failure == "provider_exit":
        state.exit_code = 1
    else:
        state.answer = {"wrong": "contract"}
    client = transport.CodexClient()
    with pytest.raises(LLMError):
        images(client)
    assert len(launches) == 1 and len(client.last_input_images) == 2


def test_prelaunch_cancellation_clears_prior_submission_metadata(invocation):
    launches, _ = invocation
    client = transport.CodexClient()
    images(client)
    client.cancelled = lambda: True
    with pytest.raises(LLMError, match="cancelled"):
        images(client)
    assert len(launches) == 1
    assert client.last_input_images == ()


def test_cancellation_after_launch_kills_once_without_single_image_retry(invocation):
    launches, state = invocation
    state.exit_code = None
    client = transport.CodexClient(cancelled=lambda: bool(launches))
    with pytest.raises(LLMError, match="cancelled"):
        images(client)
    assert len(launches) == 1
    assert state.killed == [999999]
    assert len(client.last_input_images) == 2


def test_process_creation_failure_has_no_submitted_image_record(invocation, monkeypatch):
    client = transport.CodexClient()

    def fail(*_, **__):
        raise OSError("Controlled launch failure")

    monkeypatch.setattr(transport.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="Controlled launch failure"):
        images(client)
    assert client.last_input_images == ()
