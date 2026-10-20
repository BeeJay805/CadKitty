"""Native feature refusal diagnostics preserve geometry and worker failure semantics."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

from forge_mind.workspace import program_feature

PREFIX = "Python feature returned invalid geometry: "


def diagnostic(error):
    message = str(error)
    assert message.startswith(PREFIX)
    assert len(message) <= 1400
    value = json.loads(message[len(PREFIX) :])
    assert value["stage"] == "independent_step_reimport"
    assert "source_line" not in value and "cause" not in value
    return value


def fake_author(monkeypatch, *, shape=None, step=None, ok=True):
    calls = []

    def build(self, code, directory, timeout, cancelled):
        calls.append((code, timeout))
        if ok:
            if step is None:
                (directory / "part.step").write_bytes(b"test boundary artifact")
            else:
                shutil.copyfile(step, directory / "part.step")
        return SimpleNamespace(ok=ok, error_message="existing author failure")

    monkeypatch.setattr(program_feature.Executor, "build", build)
    if shape is not None:
        monkeypatch.setattr(program_feature.b, "import_step", lambda path: shape)
    return calls


@pytest.mark.parametrize("valid,count", [(False, 1), (False, 0), (True, 0)])
def test_existing_refusal_conditions_are_distinguished(monkeypatch, valid, count):
    shape = SimpleNamespace(is_valid=valid, solids=lambda: [object()] * count)
    fake_author(monkeypatch, shape=shape)
    with pytest.raises(ValueError) as caught:
        program_feature.compile_program_feature("result = example", {"size": 7.0}, [])
    facts = diagnostic(caught.value)
    assert facts["native_valid"] is valid
    assert facts["solid_count"] == count
    assert facts["shape_type"] == "SimpleNamespace"
    assert facts["inspection_errors"] == {}


def test_success_has_same_return_and_never_reads_failure_diagnostics(monkeypatch):
    shape = SimpleNamespace(is_valid=True, solids=lambda: [object()])
    calls = fake_author(monkeypatch, shape=shape)
    monkeypatch.setattr(
        program_feature,
        "_native_feature_failure_message",
        lambda shape: pytest.fail("Successful feature read failure diagnostics"),
    )
    assert program_feature.compile_program_feature("result = example", {"size": 7.0}, []) is shape
    assert calls == [
        (
            "import build123d as b\n"
            "params = {'size': 7.0}\n"
            "inputs = tuple(b.import_step(name) for name in [])\n"
            "result = example",
            None,
        )
    ]


def test_author_failure_prefix_is_unchanged(monkeypatch):
    fake_author(monkeypatch, ok=False)
    with pytest.raises(
        ValueError, match=r"^Python feature author failed: existing author failure$"
    ):
        program_feature.compile_program_feature("result = example", {}, [])


def test_initial_guard_inspection_failure_is_not_reclassified(monkeypatch):
    class BrokenGuard:
        @property
        def is_valid(self):
            raise RuntimeError("Existing guard read failure")

    fake_author(monkeypatch, shape=BrokenGuard())
    with pytest.raises(RuntimeError, match=r"^Existing guard read failure$"):
        program_feature.compile_program_feature("result = example", {}, [])


def test_diagnostic_reads_cannot_hide_geometry_refusal(monkeypatch):
    class BrokenInspection:
        reads = 0

        @property
        def is_valid(self):
            self.reads += 1
            if self.reads == 1:
                return False  # The unchanged original guard identifies the refusal.
            raise RuntimeError("Diagnostic validity unavailable")

        def solids(self):
            raise OSError("Diagnostic solid inventory unavailable")

    fake_author(monkeypatch, shape=BrokenInspection())
    with pytest.raises(ValueError) as caught:
        program_feature.compile_program_feature("result = example", {}, [])
    facts = diagnostic(caught.value)
    assert facts["native_valid"] is None and facts["solid_count"] is None
    assert facts["inspection_errors"] == {"native_valid": "RuntimeError", "solid_count": "OSError"}


def test_unexpected_diagnostic_failure_keeps_refusal(monkeypatch):
    fake_author(monkeypatch, shape=SimpleNamespace(is_valid=False, solids=lambda: []))

    def broken(shape):
        raise RuntimeError("Unexpected helper failure")

    monkeypatch.setattr(program_feature, "_native_feature_failure_message", broken)
    with pytest.raises(ValueError) as caught:
        program_feature.compile_program_feature("result = example", {}, [])
    facts = diagnostic(caught.value)
    assert facts["inspection_errors"] == {"diagnostic": "unavailable"}
    assert facts["native_valid"] is None and facts["solid_count"] is None


def test_shape_type_is_bounded_and_truncation_is_explicit():
    shape_type = type("Ω" * 200, (), {"is_valid": False, "solids": lambda self: []})
    facts = diagnostic(program_feature._native_feature_failure_message(shape_type()))
    assert facts["shape_type"] == "Ω" * 64
    assert facts["shape_type_truncated"] is True
    assert facts["native_valid"] is False and facts["solid_count"] == 0


@pytest.mark.parametrize("kind", ["face", "shell"])
def test_native_valid_nonsolid_step_remains_refused(tmp_path, monkeypatch, kind):
    import build123d as b

    shape = (
        b.Rectangle(10, 20).faces()[0]
        if kind == "face"
        else b.Shell(b.Box(10, 20, 30).faces()[:-1])
    )
    step = tmp_path / "nonsolid.step"
    b.export_step(shape, step)
    imported = b.import_step(step)
    assert imported.is_valid and not imported.solids()
    fake_author(monkeypatch, step=step)
    with pytest.raises(ValueError) as caught:
        program_feature.compile_program_feature("result = example", {}, [])
    facts = diagnostic(caught.value)
    assert facts["native_valid"] is True and facts["solid_count"] == 0
    assert facts["shape_type"] and facts["inspection_errors"] == {}


def test_native_valid_box_returns_same_positive_solid_geometry():
    shape = program_feature.compile_program_feature("result = b.Box(2,3,4)", {}, [])
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(24)


def test_native_real_worker_tail_retains_nonsolid_refusal(tmp_path, monkeypatch):
    from forge_mind.workspace import document_service
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.models import Contract
    from forge_mind.workspace.processing import process_revision
    from forge_mind.workspace.store import Store

    document = DesignDocument.model_validate(
        {
            "features": [
                {
                    "id": "face",
                    "name": "Nonsolid feature refusal",
                    "kind": "python",
                    "code": "result=b.Rectangle(10,20).faces()[0]",
                }
            ],
            "outputs": ["face"],
        }
    )
    contract = Contract()
    store = Store(tmp_path / "store")
    session = store.create("owner", "Native nonsolid refusal control", contract.model_dump())
    service = SimpleNamespace(store=store, executor=SimpleNamespace(mode="trusted_local"))
    calls = []

    def real_process(*args, **kwargs):
        calls.append(args[0])
        return process_revision(*args, **kwargs)

    monkeypatch.setattr(document_service, "process_revision", real_process)
    revision = document_service.build_document(
        service,
        session["id"],
        "owner",
        document,
        contract,
        None,
        "Native feedback control",
        60,
        lambda: False,
    )
    measurement = revision["measurement"]
    assert measurement["geometry_valid"] is False
    matching = [error for error in measurement["errors"] if PREFIX in error]
    assert len(matching) == 1 and "CAD worker failed (exit=1)" in matching[0]
    error_line = matching[0].rsplit(PREFIX, 1)[1].splitlines()[0]
    facts = diagnostic(PREFIX + error_line)
    assert facts["native_valid"] is True and facts["solid_count"] == 0
    assert facts["inspection_errors"] == {}
    assert store.revision(session["id"], revision["id"])["measurement"] == measurement
    evidence = json.loads((calls[0] / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["errors"] == measurement["errors"]
    assert not (calls[0] / "processed.json").exists()
    assert not (calls[0] / "tessellation.json").exists()
