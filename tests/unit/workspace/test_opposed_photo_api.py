"""Pair HTTP dispatch and ownership controls, with no CAD or graphics execution."""

from __future__ import annotations

import pytest
from tests.unit.workspace.test_project_photo_api import capture as base_capture
from tests.unit.workspace.test_project_photo_api import client_for

from forge_mind.workspace import project_photo
from forge_mind.workspace.project_delivery import read_project_export


@pytest.fixture
def capture(tmp_path, monkeypatch):
    return base_capture.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("include_alternate", [None, False, True])
def test_explicit_pair_switch_and_single_total_deadline(capture, monkeypatch, include_alternate):
    service, sid, package, base = capture
    calls = []

    def invoke(kind, svc, project, owner, package_id, digest, **options):
        read_project_export(svc, project, owner, package_id, digest)
        calls.append((kind, options))
        unavailable = {
            "status": "unavailable",
            "code": "execution_unqualified",
            "reason": "controlled optional graphics absence",
            "candidates": ["base", "cap"],
        }
        alternate = {**unavailable, "view": "opposed_upper", "code": "primary_unavailable"}
        return {"primary": unavailable, "alternate": alternate} if kind == "pair" else unavailable

    monkeypatch.setattr(
        project_photo,
        "create_project_photo",
        lambda *args, **kwargs: invoke("single", *args, **kwargs),
    )
    monkeypatch.setattr(
        project_photo,
        "create_project_photo_pair",
        lambda *args, **kwargs: invoke("pair", *args, **kwargs),
    )
    body = {} if include_alternate is None else {"include_alternate": include_alternate}
    with client_for(service, "owner") as client:
        response = client.post(base + "/photo", json=body)
        assert response.status_code == 200
        assert ("primary" in response.json()) is (include_alternate is True)
    assert calls == [
        (
            "pair" if include_alternate else "single",
            {"preset": "plastic", "part_id": None, "timeout": 15},
        )
    ]
    with client_for(service, "foreign") as client:
        assert client.post(base + "/photo", json={"include_alternate": True}).status_code == 409
    assert len(calls) == 1
    assert (
        read_project_export(service, sid, "owner", package["package_id"], package["sha256"])[1]
        == package
    )


@pytest.mark.parametrize("value", [1, 0, "true", None, [], {}])
def test_pair_switch_is_strict_boolean_before_dispatch(capture, monkeypatch, value):
    service, _, _, base = capture
    monkeypatch.setattr(
        project_photo,
        "create_project_photo_pair",
        lambda *a, **k: pytest.fail("invalid request dispatched"),
    )
    with client_for(service, "owner") as client:
        assert client.post(base + "/photo", json={"include_alternate": value}).status_code == 422
