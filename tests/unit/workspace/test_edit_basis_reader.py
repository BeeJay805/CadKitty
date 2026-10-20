"""Incomplete inspection records remain readable, but cannot supply edit authority."""

import pytest
from tests.unit.workspace.test_author_images import revision_files as revision_files

from forge_mind.workspace.store import Store, WorkspaceError


def test_incomplete_saved_source_reader_rejects_without_mutating_authority(revision_files):
    directory, revision = revision_files
    store = Store(directory / "reader-store")
    sid = store.create(
        "owner", "Inspect retained geometry", {}, authoring_mode="native_operations"
    )["id"]
    store.add_revision(sid, revision)
    store.update(sid, selected_revision=revision["id"])
    before = store.session_record(sid, "owner")
    with pytest.raises(WorkspaceError, match="not a complete typed edit source"):
        store.edit_basis(sid, "owner")
    assert store.session_record(sid, "owner") == before
    with pytest.raises(WorkspaceError, match="workspace not found"):
        store.edit_basis(sid, "foreign-owner")
    assert store.session_record(sid, "owner") == before
