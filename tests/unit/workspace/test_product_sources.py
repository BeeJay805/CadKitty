from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from forge_mind.workspace import product_sources as sources
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    store = Store(tmp_path)
    sid = store.create("alice", "fan", None)["id"]
    service = SimpleNamespace(store=store)
    monkeypatch.setattr(
        sources,
        "retrieve",
        lambda *args: (b"Fan variant A thickness 27 mm", "text/plain", "https://vendor.example/a"),
    )
    return store, sid, service


def fetched(workspace):
    _store, sid, service = workspace
    state = sources.fetch_source(service, sid, "alice", "https://vendor.example/a")
    return {
        "source_artifact": state["artifacts"][0]["id"],
        "quantity": 27,
        "unit": "mm",
        "exact_quote": "variant A thickness 27 mm",
        "configuration": "variant A",
        "datum": "outside front to rear planes",
        "locator": "specification",
    }


def test_claim_is_text_evidence_until_separate_confirmation(workspace):
    store, sid, _ = workspace
    claim = fetched(workspace)
    state = sources.propose_claim(store, sid, "alice", claim)
    assert state["claims"][0]["confirmation"] is None
    assert state["claims"][0]["text_evidence"] is True
    confirmed = sources.confirm_claim(
        store, sid, "alice", state["claims"][0]["id"], state["version"]
    )
    assert confirmed["claims"][0]["confirmation"]["confirmed_by"] == "alice"
    assert confirmed["fingerprint"] != state["fingerprint"]
    assert sources.read_sources(Store(store.root), sid, "alice") == confirmed
    with pytest.raises(WorkspaceError, match="changed"):
        sources.confirm_claim(store, sid, "alice", state["claims"][0]["id"], state["version"])


def test_author_cannot_write_verified_or_confirmation(workspace):
    store, sid, _ = workspace
    claim = fetched(workspace)
    for name in ("verified", "confirmation"):
        with pytest.raises(ValidationError):
            sources.propose_claim(store, sid, "alice", {**claim, name: True})


def test_quote_and_quantity_must_exist_but_configuration_is_unconfirmed(workspace):
    store, sid, _ = workspace
    claim = fetched(workspace)
    for patch in ({"quantity": 28}, {"exact_quote": "not in source"}):
        with pytest.raises(WorkspaceError):
            sources.propose_claim(store, sid, "alice", {**claim, **patch})
    state = sources.propose_claim(
        store, sid, "alice", {**claim, "configuration": "wrong variant B"}
    )
    assert state["claims"][0]["confirmation"] is None


def test_owner_and_job_lock_checked_before_network(workspace, monkeypatch):
    store, sid, service = workspace
    calls = []
    monkeypatch.setattr(sources, "retrieve", lambda *args: calls.append(args))
    with pytest.raises(WorkspaceError, match="not found"):
        sources.fetch_source(service, sid, "bob", "https://vendor.example/a")
    with pytest.raises(WorkspaceError, match="not found"):
        sources.read_sources(store, sid, "bob")
    with store.connect() as con:
        con.execute(
            "INSERT INTO jobs (id,session,state,payload) VALUES (?,?,?,?)",
            ("job", sid, "running", "{}"),
        )
    with pytest.raises(WorkspaceError, match="active operation"):
        sources.fetch_source(service, sid, "alice", "https://vendor.example/a")
    assert calls == []


def test_agent_can_fetch_during_job_but_confirmation_remains_locked(workspace):
    store, sid, service = workspace
    with store.connect() as con:
        con.execute(
            "INSERT INTO jobs (id,session,state,payload) VALUES (?,?,?,?)",
            ("job", sid, "running", "{}"),
        )
    state = sources.fetch_source(service, sid, "alice", "https://vendor.example/a", agent=True)
    claim = {
        "source_artifact": state["artifacts"][0]["id"],
        "quantity": 27,
        "unit": "mm",
        "exact_quote": "27 mm",
        "configuration": "A",
        "datum": "planes",
        "locator": "spec",
    }
    state = sources.propose_claim(store, sid, "alice", claim, agent=True)
    with pytest.raises(WorkspaceError, match="active operation"):
        sources.confirm_claim(store, sid, "alice", state["claims"][0]["id"], state["version"])


def test_artifact_tampering_is_detected(workspace):
    store, sid, _ = workspace
    fetched(workspace)
    with store.connect() as con:
        con.execute(
            "UPDATE product_source_events SET document=replace(document,'27 mm','28 mm') WHERE session=?",
            (sid,),
        )
    with pytest.raises(WorkspaceError, match="integrity"):
        sources.read_sources(store, sid, "alice")


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "192.168.1.1",
        "::1",
        "fd00::1",
        "::ffff:127.0.0.1",
        "0.0.0.0",
    ],
)
def test_private_dns_rejected(monkeypatch, ip):
    monkeypatch.setattr(
        sources.socket, "getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", (ip, 80))]
    )
    with pytest.raises(WorkspaceError, match="not public"):
        sources.public_endpoint("http://vendor.example")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:pass@example.com",
        "http://example.com:8080",
        "https://example.com/\nheader",
    ],
)
def test_unsafe_url_syntax_rejected(url):
    with pytest.raises(WorkspaceError):
        sources.public_endpoint(url)


def test_mixed_public_private_dns_rejected(monkeypatch):
    monkeypatch.setattr(
        sources.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", (ip, 443)) for ip in ["8.8.8.8", "10.0.0.1"]],
    )
    with pytest.raises(WorkspaceError):
        sources.public_endpoint("https://vendor.example")


def test_html_script_removed_and_binary_unsupported():
    text, _ = sources.extract(b"<h1>27 mm</h1><script>evil</script><p>variant A</p>", "text/html")
    assert text == "27 mm variant A"
    assert sources.extract(b"%PDF-1.7", "application/pdf")[0] == ""
    assert "unsupported" in sources.extract(b"%PDF-1.7", "application/pdf")[1]


def test_cancellation_prevents_connect(monkeypatch):
    monkeypatch.setattr(sources, "public_endpoint", lambda *args: pytest.fail("must not resolve"))
    with pytest.raises(WorkspaceError, match="cancelled"):
        sources.retrieve("https://example.com", 20, lambda: True)


def test_retrieve_pins_ip_and_rejects_redirect_to_private(monkeypatch):
    addresses, connected = [], []

    def resolve(host, *args, **kwargs):
        addresses.append(host)
        return [(2, 1, 6, "", ("8.8.8.8" if host == "vendor.example" else "127.0.0.1", 80))]

    class Socket:
        def close(self):
            pass

    class Connection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path, headers):
            pass

        def getresponse(self):
            return SimpleNamespace(
                status=302, getheader=lambda *args: "http://internal.example/secret"
            )

        def close(self):
            pass

    monkeypatch.setattr(sources.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        sources.socket,
        "create_connection",
        lambda target, **kwargs: (connected.append(target), Socket())[1],
    )
    monkeypatch.setattr(sources.http.client, "HTTPConnection", Connection)
    with pytest.raises(WorkspaceError, match="not public"):
        sources.retrieve("http://vendor.example", 20, lambda: False)
    assert connected == [("8.8.8.8", 80)]
    assert addresses == ["vendor.example", "internal.example"]


def test_codex_research_opt_in_keeps_ordinary_author_web_disabled(monkeypatch):
    import io
    import json
    from pathlib import Path

    from forge_mind.llm import codex_client

    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdin = io.BytesIO()
            self.returncode = 0
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text(json.dumps({"findings": "No sources", "urls": [], "unresolved": []}))

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(codex_client, "_command", lambda: ["codex"])
    monkeypatch.setattr(codex_client.subprocess, "Popen", Process)
    client = codex_client.CodexClient()
    client.research_structured(system="", user="", output_model=sources.Discovery)
    client.generate_structured(
        system="", user="", output_model=sources.Discovery, model="ignored", max_tokens=30
    )
    assert 'web_search="live"' in commands[0]
    assert 'web_search="disabled"' in commands[1]
    for command in commands:
        assert "features.shell_tool=false" in command
        assert "--ignore-user-config" in command
        assert command[command.index("--sandbox") + 1] == "read-only"


def test_transaction_fingerprint_matches_read(workspace):
    store, sid, _ = workspace
    with store.connect() as con:
        empty = sources.source_fingerprint(con, sid)
    assert empty == sources.read_sources(store, sid, "alice")["fingerprint"]
    fetched(workspace)
    with store.connect() as con:
        assert (
            sources.source_fingerprint(con, sid)
            == sources.read_sources(store, sid, "alice")["fingerprint"]
        )


def test_project_views_refresh_source_fingerprint_without_session_events(workspace, monkeypatch):
    from forge_mind.workspace.project_access import project_overview

    store, sid, _ = workspace

    def fingerprint():
        expected = sources.read_sources(store, sid, "alice")["fingerprint"]
        assert project_overview(store, sid, "alice")["source_fingerprint"] == expected
        assert store.get(sid, "alice")["source_fingerprint"] == expected
        return expected

    empty = fingerprint()
    claim = fetched(workspace)
    fetched_hash = fingerprint()
    assert fetched_hash != empty
    before = store.session_record(sid, "alice")
    monkeypatch.setattr(store, "event", lambda *args, **kwargs: pytest.fail("Unexpected event"))
    proposed = sources.propose_claim(store, sid, "alice", claim)
    proposed_hash = fingerprint()
    assert proposed_hash == proposed["fingerprint"] and proposed_hash != fetched_hash
    assert store.session_record(sid, "alice") == before
    confirmed = sources.confirm_claim(
        store, sid, "alice", proposed["claims"][0]["id"], proposed["version"]
    )
    confirmed_hash = fingerprint()
    assert confirmed_hash == confirmed["fingerprint"] and confirmed_hash != proposed_hash
    assert store.session_record(sid, "alice") == before


def test_pdf_text_is_extracted_with_page_locator_when_available():
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Variant A thickness 27 mm")
    text, outcome = sources.extract(document.tobytes(), "application/pdf")
    document.close()
    assert "[page 1]" in text
    assert "27 mm" in text
    assert "no OCR" in outcome


def test_confirmation_endpoint_rejects_author_fields(workspace):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from forge_mind.workspace.product_source_api import source_router

    store, sid, service = workspace
    claim = fetched(workspace)
    app = FastAPI()
    app.include_router(source_router(store, service, lambda: "alice"))
    client = TestClient(app)
    base = f"/workspace/sessions/{sid}"
    response = client.post(base + "/source-claims", json={**claim, "verified": True})
    assert response.status_code == 422
    response = client.post(base + "/source-claims", json=claim)
    assert response.status_code == 200
    state = response.json()
    response = client.post(
        base + "/source-claims/" + state["claims"][0]["id"] + "/confirm",
        json={"version": state["version"]},
    )
    assert response.status_code == 200
    assert response.json()["claims"][0]["confirmation"]["confirmed_by"] == "alice"
