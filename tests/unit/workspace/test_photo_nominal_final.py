"""Pure final-reader controls; worker doubles do not assert native geometry proof."""

import ast
import sqlite3
import types
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest
from tests.unit.workspace.test_photo_nominal_claims import SOURCE, claim, draft

from forge_mind.workspace.document import digest
from forge_mind.workspace.models import Contract
from forge_mind.workspace.photo_nominal import attach_photo_nominal_claims, photo_source_constraints


def functions():
    path = Path(__file__).parents[3] / "src/forge_mind/workspace/requirement_verification.py"
    # These pure orchestration functions normally import an existing native
    # measurement dependency through Store. No native module is imported here.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_evaluate_photo_geometry", "_photo_basis", "aggregate"}
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    namespace = {
        "digest": digest,
        "Contract": Contract,
        "WorkspaceError": RuntimeError,
        "expired": lambda deadline: False,
        "time_left": lambda deadline: 30.0,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def context(tmp_path, monkeypatch):
    namespace = functions()
    requirement = attach_photo_nominal_claims(
        draft(SOURCE), SOURCE, [claim()]
    ).contract.requirements[0]
    source = tmp_path / "saved"
    source.mkdir()
    (source / "part.step").write_bytes(b"saved STEP test double")
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE sessions(id TEXT, owner TEXT)")
    connection.execute("INSERT INTO sessions VALUES('sid','owner')")

    @contextmanager
    def connect():
        yield connection

    service = types.SimpleNamespace(
        store=types.SimpleNamespace(connect=connect, directory=lambda sid, rid: source),
        executor=types.SimpleNamespace(mode="trusted_local"),
    )
    basis = {
        "revision_id": "rid",
        "step_sha256": "s" * 64,
        "engine_fingerprint": "engine",
        "plan_hash": "plan",
        "job_id": "job",
    }
    target = {
        "part_id": "part",
        "current_basis": basis,
        "target_sha256": "t" * 64,
        "proposal_document": {"frozen": "expected"},
        "source_constraints": photo_source_constraints(requirement.verification),
    }
    closure = {
        "status": "checked",
        "basis": basis,
        "target_sha256": "t" * 64,
        "receipt_sha256": "r" * 64,
    }
    readers = types.ModuleType("forge_mind.workspace.photo_generation")
    readers.read_photo_target = lambda *args: deepcopy(target)
    readers.photo_closure = lambda *args: deepcopy(closure)
    import sys

    monkeypatch.setitem(sys.modules, readers.__name__, readers)
    import forge_mind.workspace.verification_admission as admission

    monkeypatch.setattr(
        admission,
        "resolve_verification_subject",
        lambda *args: {"part_id": "part", "step_artifact": "part.step"},
    )
    calls = []

    def process(directory, payload, seconds, cancelled):
        request = payload["request"]
        calls.append(deepcopy(request))
        return {
            "status": "pass",
            "method": "nominal_photo_solid_v1",
            "criterion_hash": request["criterion_hash"],
            "facts": {
                "step_sha256": "s" * 64,
                "target_sha256": request["target_sha256"],
                "expected_document_sha256": digest(request["expected_document"]),
                "source_constraints": request["source_constraints"],
            },
            "evidence_refs": [{"artifact": "part.step", "sha256": "s" * 64}],
        }

    import shutil

    namespace.update(process_revision=process, copy_artifact=shutil.copy2)
    measured = types.SimpleNamespace(
        artifact_hashes={"part.step": "s" * 64}, engine_fingerprint="engine"
    )

    def evaluate(req=requirement, cache=None):
        return namespace["_evaluate_photo_geometry"](
            service,
            "sid",
            "rid",
            req,
            None,
            None,
            measured,
            tmp_path / f"proof{len(calls)}",
            deadline=None,
            cancelled=lambda: False,
            cache=cache,
        )

    return types.SimpleNamespace(
        namespace=namespace,
        requirement=requirement,
        target=target,
        closure=closure,
        calls=calls,
        evaluate=evaluate,
        connection=connection,
    )


def test_one_worker_per_exact_family_with_individual_frozen_row_hashes(context):
    cache = {}
    first = context.evaluate(cache=cache)
    second_row = context.requirement.model_copy(deep=True)
    second_row.id = "r2"
    second_row.verification.claims = ["rectangular_outline"]
    second = context.evaluate(second_row, cache)
    assert len(context.calls) == 1
    assert first["criterion_hash"] != second["criterion_hash"]
    assert first["facts"]["proved_claims"] != second["facts"]["proved_claims"]
    assert first["facts"]["shared_evaluation_hash"] == second["facts"]["shared_evaluation_hash"]


@pytest.mark.parametrize(
    "field,value", [("status", "comparison_needed"), ("target_sha256", "wrong"), ("basis", {})]
)
def test_missing_failed_or_stale_checked_photo_cannot_use_cache(context, field, value):
    cache = {}
    context.evaluate(cache=cache)
    context.closure[field] = value
    with pytest.raises(RuntimeError):
        context.evaluate(cache=cache)
    assert len(context.calls) == 1


def test_source_constraints_and_exact_definition_identity_still_guard_cache(context):
    cache = {}
    context.evaluate(cache=cache)
    context.target["source_constraints"]["size_mm"][0] = 51.0
    with pytest.raises(ValueError):
        context.evaluate(cache=cache)


def test_shared_worker_fail_or_unknown_never_aggregates_to_pass(context):
    row = context.requirement
    contract = Contract(requirements=[row])
    aggregate = context.namespace["aggregate"]
    for state in ["fail", "unverified"]:
        result = {"requirement_id": row.id, "status": state}
        assert aggregate(contract, [result])[1] is False
    other = row.model_copy(
        update={"id": "physical", "verification": None, "text": "Withstand load"}
    )
    contract = Contract(requirements=[row, other])
    assert aggregate(
        contract,
        [
            {"requirement_id": row.id, "status": "pass"},
            {"requirement_id": "physical", "status": "unverified"},
        ],
    ) == ("unverified", False)


def test_photo_workflow_mutation_invalidates_basis_and_legacy_has_no_new_key(context):
    basis = context.namespace["_photo_basis"]
    connection = context.connection
    session = {"contract": Contract(requirements=[context.requirement]).model_dump()}
    with pytest.raises(RuntimeError):
        basis(connection, "sid", session)
    connection.execute("CREATE TABLE photo_generation(session TEXT, document TEXT, digest TEXT)")
    connection.execute("INSERT INTO photo_generation VALUES('sid','original','digest')")
    before = basis(connection, "sid", session)
    connection.execute("UPDATE photo_generation SET document='changed'")
    assert basis(connection, "sid", session) != before
    assert basis(connection, "sid", {"contract": {}}) == {}
