import copy
import hashlib
from types import SimpleNamespace

import pytest

from forge_mind.workspace import pdf_render
from forge_mind.workspace import product_sources as sources
from forge_mind.workspace import source_observations as observations
from forge_mind.workspace.store import Store, WorkspaceError


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def render(raw, page_number, *, scale, **kwargs):
    text, image = "Variant A thickness 27 mm", b"controlled image bytes"
    metadata = {
        "schema_version": 1,
        "mode": "source_page",
        "source_sha256": digest(raw),
        "source_bytes": len(raw),
        "source_limit_bytes": kwargs.get("source_limit_bytes", pdf_render.MAX_SOURCE_BYTES),
        "page_number": page_number,
        "page_count": 2,
        "scale": scale,
        "page_bounds": [0, 0, 10, 20],
        "page_size": [10, 20],
        "rotation": 0,
        "crop": [0, 0, 0, 0],
        "bitmap_width": 20,
        "bitmap_height": 40,
        "backend": {"pypdfium2": "test", "pdfium": "test", "pillow": "test", "flags": []},
        "worker_sha256": "0" * 64,
        "text_sha256": digest(text.encode()),
        "text_bytes": len(text.encode()),
        "text_available": True,
        "text_truncated": False,
        "text_method": "pdfium_get_text_range_ucs2_no_ocr",
        "image_sha256": digest(image),
        "image_bytes": len(image),
        "scope": "Controlled test fixture, no real PDF rendering",
    }
    return metadata, text, image


@pytest.fixture
def captured(tmp_path, monkeypatch):
    store = Store(tmp_path)
    sid = store.create("alice", "fixture", None)["id"]
    service = SimpleNamespace(store=store)
    raw = b"%PDF-controlled fixture"
    monkeypatch.setattr(
        sources, "retrieve", lambda *args: (raw, "application/pdf", "https://example.test/a")
    )
    monkeypatch.setattr(pdf_render, "render_pdf_page", render)
    state = sources.fetch_source(service, sid, "alice", "https://example.test/a")
    return store, sid, service, state["artifacts"][0], raw


def observe(captured):
    _, sid, service, artifact, _ = captured
    return observations.observe_source_page(service, sid, "alice", artifact["id"], 1)


def claim(captured, receipt):
    store, sid, _, artifact, _ = captured
    return sources.propose_claim(
        store,
        sid,
        "alice",
        {
            "source_artifact": artifact["id"],
            "source_observation_id": receipt["id"],
            "quantity": 27,
            "unit": "mm",
            "exact_quote": "thickness 27 mm",
            "configuration": "Variant A",
            "datum": "front to rear",
            "locator": "page 1",
        },
    )


def test_manufacturer_tuple_quote_records_only_unconfirmed_page_evidence(captured, monkeypatch):
    def dimension_page(*args, **kwargs):
        metadata, _, image = render(*args, **kwargs)
        text = "Fan dimensions 120x120x25 mm; packaging 210x150x34 mm."
        metadata.update(text_sha256=digest(text.encode()), text_bytes=len(text.encode()))
        return metadata, text, image

    monkeypatch.setattr(pdf_render, "render_pdf_page", dimension_page)
    receipt = observe(captured)
    store, sid, _, artifact, _ = captured
    request = {
        "source_artifact": artifact["id"],
        "source_observation_id": receipt["id"],
        "quantity": 25,
        "unit": "mm",
        "exact_quote": "120x120x25 mm",
        "configuration": "Fan, accessory inclusion unspecified",
        "datum": "Third component interpreted as thickness; faces not specified",
        "locator": "PDF page 1",
    }
    saved = sources.propose_claim(store, sid, "alice", request)["claims"][-1]
    assert saved["quantity"] == 25
    assert saved["confirmation"] is None
    assert saved["source_observation_digest"] == receipt["observation_digest"]
    for patch in ({"quantity": 34}, {"exact_quote": "120 \u00d7 120 \u00d7 25 mm"}):
        with pytest.raises(WorkspaceError):
            sources.propose_claim(store, sid, "alice", {**request, **patch})


def test_capture_exact_bytes_and_page_identity_outside_source_fingerprint(captured):
    store, sid, _, artifact, raw = captured
    before = sources.read_sources(store, sid, "alice")
    assert artifact["text"] == ""
    assert "page" in artifact["outcome"]
    assert (
        store.root / "sessions" / sid / "source-artifacts" / artifact["id"] / "source.pdf"
    ).read_bytes() == raw
    first, second = observe(captured), observe(captured)
    assert first["id"] != second["id"]
    assert first["source_sha256"] == digest(raw)
    assert observations.read_source_observation(store, sid, "alice", first["id"]) == first
    assert (
        observations.observation_image(store, sid, "alice", first["id"])
        == render(raw, 1, scale=2)[2]
    )
    assert sources.read_sources(store, sid, "alice") == before


def test_linked_claim_still_requires_separate_confirmation(captured):
    store, sid, *_ = captured
    receipt = observe(captured)
    state = claim(captured, receipt)
    assert state["claims"][0]["confirmation"] is None
    assert state["claims"][0]["source_observation_digest"] == receipt["observation_digest"]
    confirmed = sources.confirm_claim(
        store, sid, "alice", state["claims"][0]["id"], state["version"]
    )
    assert sources.read_sources(store, sid, "alice") == confirmed


@pytest.mark.parametrize("filename", ["page.txt", "page.png", "manifest.json", "source.pdf"])
@pytest.mark.parametrize("confirmed", [False, True])
def test_linked_evidence_tamper_fails_confirmation_and_later_use(captured, filename, confirmed):
    store, sid, _, artifact, _ = captured
    receipt = observe(captured)
    state = claim(captured, receipt)
    if confirmed:
        sources.confirm_claim(store, sid, "alice", state["claims"][0]["id"], state["version"])
    directory = store.root / "sessions" / sid
    path = (
        directory / "source-artifacts" / artifact["id"] / filename
        if filename == "source.pdf"
        else directory / "source-observations" / receipt["id"] / filename
    )
    path.write_bytes(b"changed")
    with pytest.raises(WorkspaceError):
        sources.read_sources(store, sid, "alice")
    with pytest.raises(WorkspaceError):
        sources.confirm_claim(store, sid, "alice", state["claims"][0]["id"], state["version"])
    with store.connect() as con, pytest.raises(WorkspaceError):
        sources.source_fingerprint(con, sid)


def test_owner_scope_and_missing_observation(captured):
    store, sid, service, artifact, _ = captured
    receipt = observe(captured)
    for action in (
        lambda: observations.observe_source_page(service, sid, "bob", artifact["id"], 1),
        lambda: observations.read_source_observation(store, sid, "bob", receipt["id"]),
        lambda: observations.observation_image(store, sid, "bob", receipt["id"]),
        lambda: claim(captured, {"id": "f" * 32}),
    ):
        with pytest.raises(WorkspaceError):
            action()


def test_bad_worker_binding_never_published(captured, monkeypatch):
    store, sid, *_ = captured

    def wrong(*args, **kwargs):
        metadata, text, image = render(*args, **kwargs)
        metadata["source_sha256"] = "f" * 64
        return metadata, text, image

    monkeypatch.setattr(pdf_render, "render_pdf_page", wrong)
    with pytest.raises(WorkspaceError):
        observe(captured)
    assert not (store.root / "sessions" / sid / "source-observations").exists()


@pytest.mark.parametrize("version", [None, ""])
def test_encoder_version_is_required_provenance(captured, monkeypatch, version):
    def wrong(*args, **kwargs):
        metadata, text, image = render(*args, **kwargs)
        if version is None:
            del metadata["backend"]["pillow"]
        else:
            metadata["backend"]["pillow"] = version
        return metadata, text, image

    monkeypatch.setattr(pdf_render, "render_pdf_page", wrong)
    with pytest.raises(WorkspaceError):
        observe(captured)


def test_legacy_pdf_reports_unavailable(captured):
    store, sid, service, artifact, _ = captured
    legacy = copy.deepcopy(artifact)
    legacy["id"] = "a" * 32
    del legacy["raw_pdf"]
    sources._append(store, sid, "alice", {"kind": "artifact", "data": legacy})
    with pytest.raises(observations.SourceObservationError) as failure:
        observations.observe_source_page(service, sid, "alice", legacy["id"], 1)
    assert failure.value.code == "source_bytes_unavailable"


def test_failed_source_append_cleans_only_new_capture(captured, monkeypatch):
    store, sid, service, artifact, _ = captured

    def reject(*args, **kwargs):
        raise WorkspaceError("controlled rejection")

    monkeypatch.setattr(sources, "_append", reject)
    with pytest.raises(WorkspaceError, match="controlled rejection"):
        sources.fetch_source(service, sid, "alice", "https://example.test/b")
    assert [p.name for p in (store.root / "sessions" / sid / "source-artifacts").iterdir()] == [
        artifact["id"]
    ]


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_ambiguous_manifest_json_rejected(raw):
    with pytest.raises(WorkspaceError):
        observations._json(raw)


@pytest.mark.parametrize("code", ["backend_unavailable", "cancelled", "page_out_of_range"])
def test_worker_failure_propagates_without_publication(captured, monkeypatch, code):
    store, sid, *_ = captured

    def fail(*args, **kwargs):
        raise pdf_render.PdfObservationError(code, "controlled failure")

    monkeypatch.setattr(pdf_render, "render_pdf_page", fail)
    with pytest.raises(observations.SourceObservationError) as failure:
        observe(captured)
    assert failure.value.code == code
    assert not (store.root / "sessions" / sid / "source-observations").exists()


def test_post_worker_cancellation_cleans_unregistered_observation(captured):
    store, sid, service, artifact, _ = captured
    with pytest.raises(observations.SourceObservationError) as failure:
        observations.observe_source_page(
            service, sid, "alice", artifact["id"], 1, cancelled=lambda: True
        )
    assert failure.value.code == "cancelled"
    assert list((store.root / "sessions" / sid / "source-observations").iterdir()) == []


def test_wrong_parent_observation_cannot_be_used_for_claim(captured):
    store, sid, service, _, _ = captured
    receipt = observe(captured)
    second = sources.fetch_source(service, sid, "alice", "https://example.test/b")["artifacts"][-1]
    other = (store, sid, service, second, captured[-1])
    with pytest.raises(WorkspaceError, match="different source"):
        claim(other, receipt)


def test_numeric_check_still_applies_to_derived_quote(captured):
    store, sid, *_ = captured
    state = claim(captured, observe(captured))
    proposed = {
        key: value
        for key, value in state["claims"][0].items()
        if key in sources.DimensionalClaim.model_fields
    }
    proposed["quantity"] = 28
    with pytest.raises(WorkspaceError, match="quantity"):
        sources.propose_claim(store, sid, "alice", proposed)


@pytest.mark.parametrize("filename", ["page.txt", "page.png", "manifest.json", "source.pdf"])
def test_linked_files_are_not_valid_evidence(captured, filename):
    store, sid, _, artifact, _ = captured
    receipt = observe(captured)
    directory = store.root / "sessions" / sid
    path = (
        directory / "source-artifacts" / artifact["id"] / filename
        if filename == "source.pdf"
        else directory / "source-observations" / receipt["id"] / filename
    )
    path.with_name("alias").hardlink_to(path)
    with pytest.raises(WorkspaceError):
        observations.read_source_observation(store, sid, "alice", receipt["id"])


def test_missing_parent_bytes_before_worker(captured, monkeypatch):
    store, sid, _, artifact, _ = captured
    (store.root / "sessions" / sid / "source-artifacts" / artifact["id"] / "source.pdf").unlink()
    monkeypatch.setattr(
        pdf_render,
        "render_pdf_page",
        lambda *args, **kwargs: pytest.fail("must not render changed source"),
    )
    with pytest.raises(WorkspaceError):
        observe(captured)


def test_registry_tamper_rejected(captured):
    store, sid, *_ = captured
    receipt = observe(captured)
    with store.connect() as con:
        con.execute("UPDATE source_observations SET digest=? WHERE session=?", ("0" * 64, sid))
    with pytest.raises(WorkspaceError):
        observations.read_source_observation(store, sid, "alice", receipt["id"])


def test_capture_exists_before_source_event_append(captured, monkeypatch):
    _store, sid, service, _, raw = captured
    append = sources._append

    def checked(store, sid, owner, event, **kwargs):
        artifact = event["data"]
        assert (
            store.root / "sessions" / sid / "source-artifacts" / artifact["id"] / "source.pdf"
        ).read_bytes() == raw
        return append(store, sid, owner, event, **kwargs)

    monkeypatch.setattr(sources, "_append", checked)
    sources.fetch_source(service, sid, "alice", "https://example.test/b")


def test_uncertain_source_commit_preserves_registered_bytes(captured, monkeypatch):
    store, sid, service, *_ = captured
    append = sources._append

    def uncertain(*args, **kwargs):
        append(*args, **kwargs)
        raise OSError("uncertain response")

    monkeypatch.setattr(sources, "_append", uncertain)
    with pytest.raises(OSError, match="uncertain response"):
        sources.fetch_source(service, sid, "alice", "https://example.test/b")
    second = sources.read_sources(store, sid, "alice")["artifacts"][-1]
    assert (
        observations.observe_source_page(service, sid, "alice", second["id"], 1)["source_artifact"]
        == second["id"]
    )


def test_initial_append_failure_without_source_table_cleans_capture(tmp_path, monkeypatch):
    store = Store(tmp_path)
    sid = store.create("alice", "fixture", None)["id"]
    monkeypatch.setattr(
        sources,
        "retrieve",
        lambda *args: (b"%PDF-fixture", "application/pdf", "https://example.test/a"),
    )

    def reject(*args, **kwargs):
        raise WorkspaceError("initial controlled rejection")

    monkeypatch.setattr(sources, "_append", reject)
    with pytest.raises(WorkspaceError, match="initial controlled rejection"):
        sources.fetch_source(SimpleNamespace(store=store), sid, "alice", "https://example.test/a")
    assert list((store.root / "sessions" / sid / "source-artifacts").iterdir()) == []


def test_repeated_page_claims_verified_once_per_read_but_never_cached_across_reads(
    captured, monkeypatch
):
    store, sid, *_ = captured
    receipt = observe(captured)
    claim(captured, receipt)
    claim(captured, receipt)
    original = observations._read_at_root
    calls = []

    def counted(*args):
        calls.append(args[-1])
        return original(*args)

    monkeypatch.setattr(observations, "_read_at_root", counted)
    state = sources.read_sources(store, sid, "alice")
    assert len(state["claims"]) == 2
    assert calls == [receipt["id"]]
    (store.root / "sessions" / sid / "source-observations" / receipt["id"] / "page.txt").write_text(
        "changed"
    )
    with pytest.raises(WorkspaceError):
        sources.read_sources(store, sid, "alice")
    assert calls == [receipt["id"], receipt["id"]]
