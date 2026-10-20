from __future__ import annotations

import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from forge_mind.workspace import typed_cache
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.typed_cache import (
    cacheable_document,
    execution_origin,
    parent_cache_input,
    stage_cache,
)

IMAGE = "sha256:" + "1" * 64
PARENT = "a" * 32
CHILD = "b" * 32
KEY = "c" * 64
BREP = "part_" + KEY[:32] + ".brep"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document() -> dict:
    # JSON only: these controls never construct or import native CAD.
    return {
        "features": [],
        "assembly": {
            "parts": [
                {
                    "id": "block",
                    "name": "Block",
                    "features": [
                        {
                            "id": "body",
                            "name": "Body",
                            "kind": "box",
                            "params": {"x": 10, "y": 5, "z": 4},
                        }
                    ],
                    "output": "body",
                }
            ],
            "roots": [{"id": "block_instance", "definition": "block"}],
        },
    }


@pytest.fixture
def saved(tmp_path):
    revisions = tmp_path / "sessions" / ("d" * 32) / "revisions"
    parent, child = revisions / PARENT, revisions / CHILD
    parent.mkdir(parents=True)
    child.mkdir()
    (parent / BREP).write_bytes(b"untrusted native bytes; parsed only by worker")
    manifest = {
        "block": {
            "key": KEY,
            "sha256": sha((parent / BREP).read_bytes()),
            "file": BREP,
            "step": "definition_block.step",
        }
    }
    (parent / "parts.json").write_text(json.dumps(manifest), encoding="utf-8")
    measured = Measurement(
        revision_id=PARENT,
        contract_hash=Contract().fingerprint(),
        engine_fingerprint="container-engine",
        geometry_valid=True,
        artifact_hashes={name: sha((parent / name).read_bytes()) for name in ("parts.json", BREP)},
    )
    revision = {
        "id": PARENT,
        "document": document(),
        "measurement": measured.model_dump(),
        "typed_cache_origin": execution_origin(IMAGE, measured, document()),
    }
    return revision, parent, child, tmp_path / "scratch"


def admitted(saved):
    revision, parent, child, scratch = saved
    scratch.mkdir()
    value = parent_cache_input(revision, parent, child, IMAGE)
    assert value is not None
    return value, child, scratch


def stage(saved, **kwargs):
    value, child, scratch = admitted(saved)
    return stage_cache(
        value, child, scratch, IMAGE, time.monotonic() + 30, kwargs.get("cancelled", lambda: False)
    )


def test_exact_cache_stage_has_only_verified_references_and_no_parent_metadata(saved):
    revision, parent, child, scratch = saved
    (parent / "private.txt").write_text("not mounted")
    assert stage(saved) == "/work/cache"
    assert {path.name for path in (scratch / "cache").iterdir()} == {"parts.json", BREP}
    for name in ("parts.json", BREP):
        assert (scratch / "cache" / name).read_bytes() == (parent / name).read_bytes()
    assert (parent / "private.txt").exists()
    assert revision["measurement"]["revision_id"] == PARENT
    assert child.is_dir()


@pytest.mark.parametrize(
    "case", ["absent", "tag", "image", "version", "engine", "invalid", "errors", "python"]
)
def test_ineligible_historical_or_python_producer_is_cold(saved, case):
    revision, parent, child, _ = saved
    if case == "absent":
        revision.pop("typed_cache_origin")
    elif case == "tag":
        assert parent_cache_input(revision, parent, child, "forge:latest") is None
        return
    elif case in {"image", "version", "engine"}:
        revision["typed_cache_origin"][case] = "changed"
    elif case == "invalid":
        revision["measurement"]["geometry_valid"] = False
    elif case == "errors":
        revision["measurement"]["errors"] = ["failed"]
    else:
        revision["document"]["assembly"]["parts"].append(
            {"id": "unused", "features": [{"kind": "python"}]}
        )
    assert parent_cache_input(revision, parent, child, IMAGE) is None


@pytest.mark.parametrize("case", ["single", "missing", "top_python", "unused_python"])
def test_whole_producer_exclusion_is_native_free(case):
    raw = document()
    if case == "single":
        raw.pop("assembly")
    elif case == "missing":
        raw["assembly"]["parts"][0].pop("features")
    elif case == "top_python":
        raw["features"] = [{"kind": "python"}]
    else:
        raw["assembly"]["parts"].append({"features": [{"kind": "python"}]})
    assert not cacheable_document(raw)


@pytest.mark.parametrize("case", ["foreign", "self", "wrong_id"])
def test_cache_parent_must_be_same_session_owned_sibling(saved, tmp_path, case):
    revision, parent, child, _ = saved
    if case == "foreign":
        child = tmp_path / "foreign" / "revisions" / CHILD
        child.mkdir(parents=True)
    elif case == "self":
        child = parent
    else:
        revision["id"] = "e" * 32
    with pytest.raises(ValueError, match="same session"):
        parent_cache_input(revision, parent, child, IMAGE)


@pytest.mark.parametrize("name", ["parts.json", BREP])
@pytest.mark.parametrize("case", ["changed", "missing", "hardlink", "directory"])
def test_admitted_cache_integrity_failure_refuses(saved, name, case):
    value, child, scratch = admitted(saved)
    source = value.source / name
    if case == "changed":
        source.write_bytes(b"changed")
    elif case == "hardlink":
        os.link(source, value.source / "extra-link")
    else:
        source.unlink()
        if case == "directory":
            source.mkdir()
    with pytest.raises((ValueError, OSError)):
        stage_cache(value, child, scratch, IMAGE, time.monotonic() + 30, lambda: False)


@pytest.mark.parametrize(
    "case", ["traversal", "key", "sha", "step", "fields", "collision", "duplicate", "not_object"]
)
def test_manifest_admission_refuses_malformed_or_conflicting_entries(saved, case):
    revision, parent, _, _ = saved
    manifest = json.loads((parent / "parts.json").read_text())
    row = manifest["block"]
    if case == "traversal":
        row["file"] = "../private.brep"
    elif case == "key":
        row["key"] = "not a key"
    elif case == "sha":
        row["sha256"] = "f" * 64
    elif case == "step":
        row["step"] = "definition_other.step"
    elif case == "fields":
        row["private_path"] = "unwanted"
    elif case == "collision":
        other = deepcopy(row)
        other.update(key=KEY[:32] + "f" * 32, step="definition_other.step")
        manifest["other"] = other
    elif case == "not_object":
        manifest = []
    data = json.dumps(manifest)
    if case == "duplicate":
        data = '{"block":' + json.dumps(row) + ',"block":' + json.dumps(row) + "}"
    (parent / "parts.json").write_text(data)
    revision["measurement"]["artifact_hashes"]["parts.json"] = sha(data.encode())
    with pytest.raises(ValueError):
        stage(saved)


def test_identical_shared_definition_stages_one_brep(saved):
    revision, parent, _, scratch = saved
    manifest = json.loads((parent / "parts.json").read_text())
    manifest["shared"] = {**manifest["block"], "step": "definition_shared.step"}
    data = json.dumps(manifest).encode()
    (parent / "parts.json").write_bytes(data)
    revision["measurement"]["artifact_hashes"]["parts.json"] = sha(data)
    assert stage(saved) == "/work/cache"
    assert len(list((scratch / "cache").iterdir())) == 2


@pytest.mark.parametrize("limit", ["manifest", "parts", "headroom"])
def test_optimization_limits_choose_cold_before_admission(saved, monkeypatch, limit):
    revision, parent, child, scratch = saved
    original = deepcopy(revision)
    if limit == "manifest":
        monkeypatch.setattr(typed_cache, "_MANIFEST_BYTES", 1)
    elif limit == "parts":
        monkeypatch.setattr(typed_cache, "_MAX_PARTS", 0)
    else:
        monkeypatch.setattr(typed_cache, "MAX_JOB_BYTES", typed_cache.HOSTED_LIMITS["file_bytes"])
    assert parent_cache_input(revision, parent, child, IMAGE) is None
    assert revision == original
    assert not scratch.exists()


@pytest.mark.parametrize("case", ["cancel", "deadline", "file_limit", "job_limit", "image_changed"])
def test_staging_obeys_current_image_limits_and_cancellation(saved, monkeypatch, case):
    value, child, scratch = admitted(saved)
    if case == "file_limit":
        monkeypatch.setitem(typed_cache.HOSTED_LIMITS, "file_bytes", 1)
    if case == "job_limit":
        monkeypatch.setattr(typed_cache, "MAX_JOB_BYTES", 1)
    with pytest.raises(ValueError):
        stage_cache(
            value,
            child,
            scratch,
            "sha256:" + "2" * 64 if case == "image_changed" else IMAGE,
            0 if case == "deadline" else time.monotonic() + 30,
            lambda: case == "cancel",
        )


def test_cancellation_during_copy_refuses_before_worker(saved, monkeypatch):
    value, child, scratch = admitted(saved)
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 4

    with pytest.raises(ValueError, match="cancelled"):
        stage_cache(value, child, scratch, IMAGE, time.monotonic() + 30, cancelled)


def test_reparse_source_is_refused_without_following_it(saved, monkeypatch):
    value, child, scratch = admitted(saved)
    original = typed_cache._not_reparse

    def check(path):
        if path == value.source:
            raise ValueError("reparse point")
        return original(path)

    monkeypatch.setattr(typed_cache, "_not_reparse", check)
    with pytest.raises(ValueError, match="reparse"):
        stage_cache(value, child, scratch, IMAGE, time.monotonic() + 30, lambda: False)


def emit_worker(scratch, payload, *, invalid=False, errors=False):
    (scratch / "part.step").write_bytes(b"fresh export")
    for path in (scratch / "cache").glob("*"):
        (scratch / path.name).write_bytes(path.read_bytes())
    result = Measurement(
        revision_id=CHILD,
        contract_hash=Contract().fingerprint(),
        engine_fingerprint="container-engine",
        geometry_valid=not invalid,
        errors=["failed"] if errors else [],
        artifact_hashes={
            path.name: sha(path.read_bytes())
            for path in scratch.iterdir()
            if path.is_file() and path.name != "processing.json"
        },
    )
    (scratch / "processed.json").write_text(result.model_dump_json())


@pytest.mark.parametrize("case", ["success", "invalid", "errors", "python", "tag", "failed"])
def test_executor_origin_is_host_recorded_only_after_qualified_success(saved, monkeypatch, case):
    value, child, _ = admitted(saved)
    for name in ("program.py", "document.json"):
        (child / name).write_text("owned input")
    engine = Executor()
    engine._typed_container_verified = True
    engine.typed_container_image = IMAGE if case != "tag" else "mutable:tag"
    engine._typed_qualified_image = IMAGE
    monkeypatch.setattr(engine, "_docker_path", lambda: "docker")
    raw = document()
    if case == "python":
        raw["assembly"]["parts"].append({"features": [{"kind": "python"}]})
    payload = {
        "revision_id": CHILD,
        "contract": {},
        "document": raw,
        "assets": {},
        "cache_source": str(value.source),
        "typed_cache_origin": {"image": "forged"},
    }
    calls = []

    def run(docker, scratch, module, request, timeout, cancelled, *, image):
        calls.append(image)
        staged = json.loads((scratch / request).read_text())
        assert set(staged) == {"revision_id", "contract", "document", "assets", "cache_source"}
        assert str(value.source) not in (scratch / request).read_text()
        if case == "tag":
            assert staged["cache_source"] is None
        else:
            assert staged["cache_source"] == "/work/cache"
        if case == "failed":
            raise ValueError("worker failed")
        # A mutable attribute cannot rewrite the identity captured before launch.
        engine.typed_container_image = "sha256:" + "9" * 64
        emit_worker(scratch, staged, invalid=case == "invalid", errors=case == "errors")

    monkeypatch.setattr(engine, "_run_container", run)
    origin = {"stale": True}
    if case == "failed":
        with pytest.raises(ValueError, match="worker failed"):
            engine.process_document(
                child, payload, 30, lambda: False, cache_input=value, origin=origin
            )
    else:
        engine.process_document(
            child,
            payload,
            30,
            lambda: False,
            cache_input=None if case == "tag" else value,
            origin=origin,
        )
    if case == "success":
        assert origin == {"version": 1, "image": IMAGE, "engine": "container-engine"}
        assert (child / "part.step").read_bytes() == b"fresh export"
    else:
        assert origin == {}
    assert len(calls) == 1


def test_admitted_descriptor_is_immutable(saved):
    value, _, _ = admitted(saved)
    with pytest.raises(AttributeError):
        value.image = "mutable"
    # Explicitly copied descriptors still undergo image and sibling checks.
    assert replace(value, image=IMAGE) == value


@pytest.mark.parametrize("mode", ["docker", "trusted_local", "legacy", "python"])
def test_document_service_keeps_host_origin_and_trusted_local_route(saved, monkeypatch, mode):
    from forge_mind.workspace import customer_construction, document_service, photo_generation

    revision, parent, child, _ = saved
    original = deepcopy(revision)
    raw = document()
    if mode == "legacy":
        revision.pop("typed_cache_origin")
    if mode == "python":
        raw["assembly"]["parts"].append({"features": [{"kind": "python"}]})
    native_document = SimpleNamespace(
        assembly=SimpleNamespace(parts=[]),
        features=[],
        variables={},
        model_dump=lambda: deepcopy(raw),
        model_dump_json=lambda **kwargs: json.dumps(raw),
    )
    stored, loaded, calls = [], [], []

    def load(sid, rid):
        loaded.append((sid, rid))
        return revision

    def directory(sid, rid):
        return parent if rid == PARENT else child.parent / rid

    def process(work, payload, timeout, cancelled, **kwargs):
        calls.append((payload, kwargs))
        if mode == "trusted_local":
            assert payload["cache_source"] == str(parent)
            assert kwargs == {}
        else:
            assert payload["cache_source"] is None
            assert (kwargs["cache_input"] is not None) == (mode == "docker")
            if cacheable_document(payload["document"]):
                kwargs["origin"].update(version=1, image=IMAGE, engine="container-engine")
        return Measurement(
            revision_id=payload["revision_id"],
            contract_hash=Contract().fingerprint(),
            engine_fingerprint="container-engine",
            geometry_valid=True,
        ).model_dump()

    service = SimpleNamespace(
        executor=SimpleNamespace(
            mode="trusted_local" if mode == "trusted_local" else "docker",
            typed_container_image=IMAGE,
            process_document=process,
        ),
        store=SimpleNamespace(
            directory=directory,
            get=lambda *args: {"assets": []},
            event=lambda *args, **kwargs: None,
            revision=lambda *args: {"code": "old source"},
            add_revision=lambda sid, item: stored.append(item),
        ),
        _load_revision=load,
    )
    monkeypatch.setattr(photo_generation, "require_grounded", lambda *args: None)
    monkeypatch.setattr(customer_construction, "guard_document", lambda *args: None)
    monkeypatch.setattr(
        document_service, "preflight_document", lambda *args: (None, None, None, {})
    )
    monkeypatch.setattr(document_service, "process_revision", process)
    monkeypatch.setattr(document_service, "engine_fingerprint", lambda: "host-engine")
    result = document_service.build_document(
        service, "d" * 32, "owner", native_document, Contract(), PARENT, "Edit", 30, lambda: False
    )
    assert loaded == [("d" * 32, PARENT)]
    assert len(calls) == len(stored) == 1
    assert ("typed_cache_origin" in result) == (mode in {"docker", "legacy"})
    # Host metadata attaches to the new revision, never retroactively to its parent.
    if mode == "legacy":
        assert "typed_cache_origin" not in revision
    else:
        assert revision == original


def test_native_cache_bridge_preserves_geometry_and_rebuilds_changed_part(tmp_path):
    """Root-only native check; normal pure selection excludes test_native_*.

    This exercises the real existing compiler after the host staging seam, without
    running a container. Docker isolation is qualified separately by the root.
    """
    import build123d as b

    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.kernel import compile_document

    raw = document()
    raw["assembly"]["parts"].append(
        {
            "id": "pin",
            "name": "Pin",
            "features": [
                {
                    "id": "shaft",
                    "name": "Shaft",
                    "kind": "cylinder",
                    "params": {"radius": 2, "height": 6},
                }
            ],
            "output": "shaft",
        }
    )
    raw["assembly"]["roots"].append(
        {"id": "pin_instance", "definition": "pin", "position": [20, 0, 0]}
    )
    doc = DesignDocument.model_validate(raw)
    revisions = tmp_path / "sessions" / ("d" * 32) / "revisions"
    parent, child = revisions / PARENT, revisions / CHILD
    parent.mkdir(parents=True)
    child.mkdir()
    original, _ = compile_document(doc, {}, parent)
    measured = Measurement(
        revision_id=PARENT,
        contract_hash=Contract().fingerprint(),
        engine_fingerprint="native-test",
        geometry_valid=True,
        artifact_hashes={
            path.name: sha(path.read_bytes()) for path in parent.iterdir() if path.is_file()
        },
    )
    revision = {
        "id": PARENT,
        "document": doc.model_dump(),
        "measurement": measured.model_dump(),
        "typed_cache_origin": execution_origin(IMAGE, measured, doc.model_dump()),
    }
    admitted_input = parent_cache_input(revision, parent, child, IMAGE)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert stage_cache(admitted_input, child, scratch, IMAGE, time.monotonic() + 30, lambda: False)
    unchanged = tmp_path / "unchanged"
    unchanged.mkdir()
    reused, history = compile_document(doc, {}, unchanged, scratch / "cache")
    assert history[-1]["reused"] == 2
    assert reused.volume == pytest.approx(original.volume)
    assert [solid.volume for solid in reused.solids()] == pytest.approx(
        [solid.volume for solid in original.solids()]
    )
    assert reused.bounding_box().size == original.bounding_box().size
    assert {path.name for path in unchanged.glob("definition_*.step")} == {
        "definition_block.step",
        "definition_pin.step",
    }
    for path in unchanged.glob("definition_*.step"):
        assert b.import_step(path).is_valid
    changed = tmp_path / "changed"
    changed.mkdir()
    assert doc.assembly
    doc.assembly.parts[0].features[0].params["x"] = 12
    _, history = compile_document(doc, {}, changed, scratch / "cache")
    assert history[-1]["reused"] == 1
    assert history[-1]["built"] == 1
