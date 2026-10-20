"""Synthetic file transport controls. No real network, model, native CAD or worker."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import socket
import subprocess
from unittest.mock import Mock

import pytest
from PIL import Image
from tests.unit.workspace.test_customer_mcp import remote as remote
from tests.unit.workspace.test_remote_mcp import call, result

from forge_mind.workspace import customer_photo as photo
from forge_mind.workspace import customer_photo_fetch as fetch
from forge_mind.workspace.customer_photo import (
    ChatGPTPhotoFile,
    create_from_photo,
    photo_capability,
)
from forge_mind.workspace.customer_photo_authority import PROTOCOL
from forge_mind.workspace.store import WorkspaceError

HOST = "files.example.test"
URL = "https://files.example.test/image?secret=confidential"


def png():
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), (20, 40, 60)).save(stream, format="PNG")
    return stream.getvalue()


def file(**changes):
    return ChatGPTPhotoFile(
        download_url=changes.pop("download_url", URL),
        file_id=changes.pop("file_id", "file-photo-1"),
        **changes,
    )


def create(service, **changes):
    return create_from_photo(
        service,
        changes.pop("principal", "alice"),
        request_id=changes.pop("request_id", "a" * 32),
        prompt=changes.pop("prompt", "Make a ring from this photo"),
        file=changes.pop("file", file()),
        known_dimension=None,
        approval_required=True,
        printable=False,
        **changes,
    )


def test_photo_recovery_precedes_fetch_and_never_stores_signed_urls(remote, monkeypatch):
    _client, service = remote
    raw = png()
    fetcher = Mock(return_value=raw)
    monkeypatch.setattr(photo, "fetch_photo", fetcher)
    first = create(service, file=file(mime_type="image/png", file_name="first.png"))
    assert first["customer_photo_identity"]["protocol"] == PROTOCOL
    assert (
        first["customer_photo_identity"]["original_image_sha256"] == hashlib.sha256(raw).hexdigest()
    )
    assert "confidential" not in json.dumps(first)
    assert "download_url" not in json.dumps(first)
    fetcher.side_effect = AssertionError("Recovered creation must never fetch again")
    replay = create(
        service, file=file(download_url="https://unconfigured.example/expired?different=secret")
    )
    assert replay["id"] == first["id"]
    assert fetcher.call_count == 1
    added_metadata = create(service, file=file(mime_type="image/png", file_name="renamed.png"))
    assert added_metadata["id"] == first["id"]
    assert added_metadata["customer_reference"]["filename"] == "first.png"
    with pytest.raises(fetch.PhotoFetchError, match="photo_type_refused"):
        create(service, file=file(mime_type="image/jpeg"))
    assert fetcher.call_count == 1
    with service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        assert "confidential" not in con.execute("SELECT document FROM sessions").fetchone()[0]


@pytest.mark.parametrize(
    "changes", [{"prompt": "Changed requested dimensions"}, {"file": file(file_id="different-id")}]
)
def test_changed_stable_photo_input_conflicts_before_fetch(remote, monkeypatch, changes):
    _client, service = remote
    fetcher = Mock(return_value=png())
    monkeypatch.setattr(photo, "fetch_photo", fetcher)
    create(service)
    with pytest.raises(WorkspaceError, match="different photo"):
        create(service, **changes)
    assert fetcher.call_count == 1


def test_same_external_file_id_is_not_cross_owner_authority(remote, monkeypatch):
    _client, service = remote
    fetcher = Mock(return_value=png())
    monkeypatch.setattr(photo, "fetch_photo", fetcher)
    alice = create(service)
    bob = create(service, principal="bob")
    assert alice["id"] != bob["id"]
    assert fetcher.call_count == 2


def test_original_image_hash_is_bound_in_same_store_admission(remote, monkeypatch):
    _client, service = remote
    monkeypatch.setattr(photo, "fetch_photo", lambda _: png())
    existing = create(service)
    from forge_mind.workspace.customer_projects import CustomerProjectRequest
    from forge_mind.workspace.customer_references import CustomerReference

    request = CustomerProjectRequest(
        request_id="a" * 32,
        prompt=existing["prompt"],
        approval_required=True,
        printable=False,
        reference=CustomerReference(
            filename="reference.png",
            media_type="image/png",
            content_base64=base64.b64encode(png()).decode(),
        ),
    )
    changed = {**existing["customer_photo_identity"], "original_image_sha256": "f" * 64}
    with pytest.raises(ValueError, match="original image"):
        service.store.create_customer_project(
            "alice", request.model_dump(mode="json"), server_metadata=changed
        )


def test_default_unavailable_mime_mismatch_and_failed_fetch_leave_no_admission(remote, monkeypatch):
    _client, service = remote
    monkeypatch.delenv(photo.HOSTS_ENV, raising=False)
    assert photo_capability()["status"] == "unavailable"
    with pytest.raises(fetch.PhotoFetchError):
        create(service)
    assert service.store.list_sessions("alice") == []
    monkeypatch.setattr(photo, "fetch_photo", lambda _: png())
    with pytest.raises(fetch.PhotoFetchError, match="photo_type_refused"):
        create(service, file=file(mime_type="image/jpeg"))
    assert service.store.list_sessions("alice") == []


@pytest.mark.parametrize(
    "url",
    [
        "http://files.example.test/a",
        "https://other.example/a",
        "https://user:pass@files.example.test/a",
        "https://files.example.test:444/a",
        "https://files.example.test/a#fragment",
        "https://files.example.test/a\r\nSecret: yes",
    ],
)
def test_url_boundary_refuses_arbitrary_destinations_without_dns(monkeypatch, url):
    resolver = Mock(side_effect=AssertionError("URL must refuse before DNS"))
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    with pytest.raises(fetch.PhotoFetchError):
        fetch.fetch_image(url, (HOST,))
    assert not resolver.called


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "2002:7f00:1::"],
)
def test_every_dns_answer_must_be_public_and_not_a_tunnel(monkeypatch, address):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_a, **_k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)),
        ],
    )
    with pytest.raises(fetch.PhotoFetchError, match="photo_dns_refused"):
        fetch.public_addresses(HOST)


def test_tls_connect_pins_numeric_ip_but_keeps_original_certificate_hostname(monkeypatch):
    raw = Mock()
    secured = Mock()
    socket_call = Mock(return_value=raw)
    context = Mock()
    context.wrap_socket.return_value = secured
    monkeypatch.setattr(socket, "create_connection", socket_call)
    monkeypatch.setattr(fetch.ssl, "create_default_context", lambda: context)
    connection = fetch.PinnedHTTPSConnection(HOST, "1.1.1.1")
    connection.connect()
    socket_call.assert_called_once_with(("1.1.1.1", 443), timeout=3)
    context.wrap_socket.assert_called_once_with(raw, server_hostname=HOST)
    secured.settimeout.assert_called_once_with(5)


def test_redirect_is_not_followed_and_oversized_stream_is_bounded(monkeypatch):
    monkeypatch.setattr(fetch, "public_addresses", lambda _: ("1.1.1.1",))
    connection = Mock()
    response = Mock(status=302)
    connection.getresponse.return_value = response
    monkeypatch.setattr(fetch, "PinnedHTTPSConnection", lambda *_: connection)
    with pytest.raises(fetch.PhotoFetchError, match="photo_http_refused"):
        fetch.fetch_image(URL, (HOST,))
    assert connection.request.call_count == 1
    monkeypatch.setattr(fetch, "MAX_BYTES", 8)
    response.status = 200
    response.getheader.side_effect = lambda key, default=None: default
    response.read.return_value = b"012345678"
    with pytest.raises(fetch.PhotoFetchError, match="photo_too_large"):
        fetch.fetch_image(URL, (HOST,))
    assert response.read.call_args.args[0] == 9


def test_fetch_child_deadline_kills_reaps_and_never_puts_url_in_argv(monkeypatch):
    monkeypatch.setenv(photo.HOSTS_ENV, HOST)
    process = Mock()
    process.communicate.side_effect = [subprocess.TimeoutExpired("photo_fetch", 10), (b"", None)]
    process.poll.return_value = None
    create_process = Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", create_process)
    with pytest.raises(fetch.PhotoFetchError, match="photo_fetch_deadline") as failure:
        photo.fetch_photo(URL)
    assert "confidential" not in str(failure.value)
    assert URL not in str(create_process.call_args)
    assert URL.encode() in process.communicate.call_args_list[0].kwargs["input"]
    process.kill.assert_called_once()
    assert process.communicate.call_count == 2


def test_official_file_tool_schema_and_metadata_require_only_url_and_id(remote):
    client, _service = remote
    response = client.post(
        "/mcp",
        headers={"Authorization": "Bearer alice", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": {}},
    )
    tools = {item["name"]: item for item in result(response)["tools"]}
    tool = tools["start_customer_project_from_photo"]
    assert tool["_meta"]["openai/fileParams"] == ["file"]
    schema = tool["inputSchema"]
    definition = schema["$defs"]["ChatGPTPhotoFile"]
    assert set(definition["required"]) == {"download_url", "file_id"}
    assert set(definition["properties"]) == {"download_url", "file_id", "mime_type", "file_name"}
    assert all(
        prop["type"] == "string" and "anyOf" not in prop
        for prop in definition["properties"].values()
    )
    assert tool["annotations"]["readOnlyHint"] is False
    assert tool["_meta"]["securitySchemes"] == [
        {"type": "oauth2", "scopes": ["cad:read", "cad:write"]}
    ]


def test_failed_queue_admission_rolls_back_photo_identity_and_request(remote, monkeypatch):
    _client, service = remote
    monkeypatch.setattr(photo, "fetch_photo", lambda _: png())

    def refuse(*_args, **_kwargs):
        raise WorkspaceError("controlled admission refusal")

    monkeypatch.setattr(service.store, "_enqueue_with_connection", refuse)
    with pytest.raises(WorkspaceError, match="controlled admission"):
        create(service)
    with service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM customer_requests").fetchone()[0] == 0


def test_file_metadata_rejects_nested_parameter_names_without_changing_scopes():
    from types import SimpleNamespace

    from mcp.types import ToolAnnotations

    from forge_mind.workspace.connector_auth import scoped_tool

    register = Mock(side_effect=lambda **_kwargs: lambda function: function)
    server = SimpleNamespace(tool=register, connector_tool_scopes={}, connector_file_tools=set())

    def ordinary(file):
        return file

    with pytest.raises(ValueError):
        scoped_tool(
            server, annotations=ToolAnnotations(read_only_hint=False), file_params=("input.file",)
        )
    with pytest.raises(ValueError, match="top-level tool argument"):
        scoped_tool(
            server, annotations=ToolAnnotations(read_only_hint=False), file_params=("missing",)
        )(ordinary)
    scoped_tool(server, annotations=ToolAnnotations(read_only_hint=False), file_params=("file",))(
        ordinary
    )
    assert server.connector_tool_scopes == {"ordinary": True}
    assert server.connector_file_tools == {"ordinary"}
    assert register.call_args.kwargs["meta"] == {
        "openai/fileParams": ["file"],
        "securitySchemes": [{"type": "oauth2", "scopes": ["cad:read", "cad:write"]}],
    }


def test_photo_tool_accepts_official_minimal_file_and_keeps_authenticated_owner(
    remote, monkeypatch
):
    client, service = remote
    fetcher = Mock(return_value=png())
    monkeypatch.setattr(photo, "fetch_photo", fetcher)
    arguments = {
        "request_id": "a" * 32,
        "prompt": "Make a ring",
        "file": {"download_url": URL, "file_id": "file-1"},
        "printable": False,
    }
    assert result(call(client, "readonly", "start_customer_project_from_photo", arguments))[
        "isError"
    ]
    first = result(call(client, "alice", "start_customer_project_from_photo", arguments))
    assert not first.get("isError"), first
    sid = first["structuredContent"]["id"]
    assert (
        service.store.session_record(sid, "alice")["customer_photo_identity"]["protocol"]
        == PROTOCOL
    )
    rotated = {
        **arguments,
        "file": {"download_url": "https://expired.example/now", "file_id": "file-1"},
    }
    replay = result(call(client, "alice", "start_customer_project_from_photo", rotated))
    assert replay["structuredContent"]["id"] == sid
    assert fetcher.call_count == 1


def test_invalid_file_arguments_hide_signed_urls_in_response_and_logs(remote, caplog):
    client, _service = remote
    caplog.set_level(logging.DEBUG)
    base = {
        "request_id": "a" * 32,
        "prompt": "Make a ring",
        "file": {"download_url": URL, "file_id": "file-1"},
    }
    malformed = [
        {**base, "file": {"download_url": [URL], "file_id": "file-1"}},
        {**base, "file": {**base["file"], URL: URL}},
        {**base, "prompt": {"secret": URL}},
        {**base, "known_dimension": {"label": URL, "value": "bad", "unit": "mm"}},
    ]
    for arguments in malformed:
        rejected = result(call(client, "alice", "start_customer_project_from_photo", arguments))
        assert rejected["isError"]
        assert URL not in json.dumps(rejected)
    assert URL not in caplog.text
