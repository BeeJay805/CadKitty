"""Controlled network transport checks for configurable public-source admission."""

import ssl
from types import SimpleNamespace

import pytest

from forge_mind.workspace import product_sources as sources
from forge_mind.workspace.store import WorkspaceError


class Response:
    def __init__(self, body=b"", *, status=200, headers=None, on_read=None):
        self.body = body
        self.status = status
        self.headers = {"Content-Type": "application/pdf", **(headers or {})}
        self.on_read = on_read
        self.requests = []

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, maximum):
        assert 0 < maximum <= 65536
        self.requests.append(maximum)
        if self.on_read:
            self.on_read()
        count = min(maximum, 2)
        value, self.body = self.body[:count], self.body[count:]
        return value


@pytest.fixture
def network(monkeypatch):
    monkeypatch.delenv("FORGE_WORKSPACE_SOURCE_BYTES", raising=False)
    state = SimpleNamespace(
        responses=[], connected=[], resolved=[], requests=[], sockets=[], connections=[], tls=[]
    )

    class Socket:
        def __init__(self):
            self.closed = False
            self.timeouts = []

        def close(self):
            self.closed = True

        def settimeout(self, value):
            self.timeouts.append(value)

        def shutdown(self, how):
            pass

    class Connection:
        sock = None

        def __init__(self, hostname, port, timeout):
            self.hostname, self.port = hostname, port
            self.closed = False
            state.connections.append(self)

        def request(self, method, path, headers):
            state.requests.append((self.hostname, method, path, headers))

        def getresponse(self):
            return state.responses.pop(0)

        def close(self):
            self.closed = True

    def resolve(host, port, *args, **kwargs):
        state.resolved.append(host)
        address = "127.0.0.1" if host == "private.example" else "8.8.8.8"
        return [(2, 1, 6, "", (address, port))]

    def connect(target, **kwargs):
        state.connected.append(target)
        sock = Socket()
        state.sockets.append(sock)
        return sock

    def wrap(sock, server_hostname):
        state.tls.append(server_hostname)
        return sock

    monkeypatch.setattr(sources.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(sources.socket, "create_connection", connect)
    monkeypatch.setattr(sources.http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(
        sources.ssl, "create_default_context", lambda: SimpleNamespace(wrap_socket=wrap)
    )
    return state


@pytest.mark.parametrize("content_length", [None, "1", "5000"])
def test_exact_cap_succeeds_without_trusting_content_length(network, monkeypatch, content_length):
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", "5")
    headers = {} if content_length is None else {"Content-Length": content_length}
    response = Response(b"12345", headers=headers)
    network.responses.append(response)
    raw, mime, final_url = sources.retrieve("https://public.example/catalog?q=1", 20, lambda: False)
    assert (raw, mime, final_url) == (
        b"12345",
        "application/pdf",
        "https://public.example/catalog?q=1",
    )
    assert response.requests[-1] == 1  # Observe EOF without allowing another full chunk.
    assert network.connected == [("8.8.8.8", 443)]
    assert network.tls == ["public.example"]
    assert network.requests[0][2] == "/catalog?q=1"
    assert network.requests[0][3]["Accept-Encoding"] == "identity"
    assert all(item.closed for item in [*network.sockets, *network.connections])


@pytest.mark.parametrize("content_length", [None, "1", "6"])
def test_one_byte_over_cap_is_rejected_and_transport_closed(network, monkeypatch, content_length):
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", "5")
    headers = {} if content_length is None else {"Content-Length": content_length}
    network.responses.append(Response(b"123456", headers=headers))
    with pytest.raises(WorkspaceError, match=r"Source exceeds byte limit \(5 bytes\)"):
        sources.retrieve("https://public.example/catalog", 20, lambda: False)
    assert all(item.closed for item in [*network.sockets, *network.connections])


@pytest.mark.parametrize(
    ("changed", "body", "succeeds"), [("1", b"12345", True), ("1000", b"123456", False)]
)
def test_environment_changes_mid_read_do_not_change_admitted_limit(
    network, monkeypatch, changed, body, succeeds
):
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", "5")
    network.responses.append(
        Response(body, on_read=lambda: monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", changed))
    )
    if succeeds:
        assert sources.retrieve("http://public.example/catalog", 20, lambda: False)[0] == body
    else:
        with pytest.raises(WorkspaceError, match=r"\(5 bytes\)"):
            sources.retrieve("http://public.example/catalog", 20, lambda: False)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "5", float("inf")])
def test_private_limit_rejects_nonpositive_or_noninteger_before_network(network, limit):
    with pytest.raises(WorkspaceError, match="positive integer"):
        sources.retrieve("https://public.example/catalog", 20, lambda: False, max_bytes=limit)
    assert not network.resolved and not network.connected


def test_private_resolved_limit_is_independent_of_environment(network, monkeypatch):
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", "1")
    network.responses.append(Response(b"12345"))
    assert (
        sources.retrieve("https://public.example/catalog", 20, lambda: False, max_bytes=5)[0]
        == b"12345"
    )


def test_redirect_revalidates_destination_and_never_connects_private_address(network, monkeypatch):
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", "8388608")
    network.responses.append(
        Response(status=302, headers={"Location": "http://private.example/secret"})
    )
    with pytest.raises(WorkspaceError, match="not public"):
        sources.retrieve("https://public.example/catalog", 20, lambda: False)
    assert network.resolved == ["public.example", "private.example"]
    assert network.connected == [("8.8.8.8", 443)]
    assert network.sockets[0].closed and network.connections[0].closed


def test_redirect_budget_is_unchanged(network):
    network.responses.extend(Response(status=302, headers={"Location": "/again"}) for _ in range(4))
    with pytest.raises(WorkspaceError, match="redirect limit"):
        sources.retrieve("https://public.example/catalog", 20, lambda: False, max_bytes=8388608)
    assert len(network.connected) == 4
    assert all(item.closed for item in [*network.sockets, *network.connections])


def test_tls_verification_failure_never_sends_request(network, monkeypatch):
    def fail(*args, **kwargs):
        raise ssl.SSLCertVerificationError("controlled invalid certificate")

    monkeypatch.setattr(
        sources.ssl, "create_default_context", lambda: SimpleNamespace(wrap_socket=fail)
    )
    with pytest.raises(ssl.SSLCertVerificationError):
        sources.retrieve("https://public.example/catalog", 20, lambda: False, max_bytes=8388608)
    assert network.tls == [] and network.requests == []
    assert network.sockets[0].closed and network.connections[0].closed


def test_compressed_body_remains_rejected_with_larger_limit(network):
    response = Response(b"compressed", headers={"Content-Encoding": "gzip"})
    network.responses.append(response)
    with pytest.raises(WorkspaceError, match="Compressed source"):
        sources.retrieve("https://public.example/catalog", 20, lambda: False, max_bytes=8388608)
    assert response.requests == []


def test_pre_cancel_keeps_network_unused(network):
    with pytest.raises(WorkspaceError, match="cancelled"):
        sources.retrieve("https://public.example/catalog", 20, lambda: True, max_bytes=8388608)
    assert network.resolved == [] and network.connected == []


@pytest.mark.parametrize(
    "url",
    [
        "file:///private.pdf",
        "http://user:password@public.example/a",
        "http://public.example:8080/a",
        "https://public.example/a\nheader",
    ],
)
def test_unsafe_url_forms_remain_rejected(network, url):
    with pytest.raises(WorkspaceError):
        sources.retrieve(url, 20, lambda: False, max_bytes=8388608)
    assert network.connected == []
