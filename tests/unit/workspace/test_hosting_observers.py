"""Exercise real broker dispatch through every existing observer service, with fake kernels."""

from __future__ import annotations

import pytest

from forge_mind.workspace.hosting_runner import OPERATIONS
from forge_mind.workspace.hosting_runner_executor import BrokerExecutor


@pytest.mark.parametrize("kind", ["access", "distance", "neighbors", "section", "photo"])
def test_observer_services_use_broker_without_docker_binary(tmp_path, monkeypatch, kind):
    from tests.unit.workspace import test_geometry_access_service as access
    from tests.unit.workspace import test_geometry_distance_service as distance
    from tests.unit.workspace import test_product_photo as photo
    from tests.unit.workspace import test_section_inspection as section
    from tests.unit.workspace import test_workspace_geometry_neighbors_service as neighbors

    fixture = {
        "access": access,
        "distance": distance,
        "neighbors": neighbors,
        "section": section,
        "photo": photo,
    }[kind]
    service, _, _, _, calls, worker = fixture.fixture_service(
        tmp_path, monkeypatch, mode="docker", qualified=True
    )

    class Client:
        def execute(self, directory, received_kind, timeout, cancelled):
            assert received_kind == kind
            worker([OPERATIONS[kind][0]], directory, timeout, cancelled)
            return {"operation_id": "a" * 32}

    service.executor = BrokerExecutor(Client(), "qualified@sha256:fixture")
    service.executor._typed_container_verified = True
    service.executor.runner_verified = True
    monkeypatch.setattr(
        "shutil.which", lambda _: pytest.fail("Public process must not discover Docker")
    )
    if kind == "access":
        access.inspect(service)
    elif kind == "distance":
        distance.query(service)
    elif kind == "neighbors":
        neighbors.query(service)
    elif kind == "section":
        section.inspection.inspect_section(service, section.SID, "owner", section.RID, "y", 2)
    else:
        photo.photo.get_product_photo(service, photo.SID, "owner", photo.RID, "plastic")
    assert len(calls["worker"]) == 1


def test_occurrence_service_uses_broker_without_docker_binary(tmp_path, monkeypatch):
    from tests.unit.workspace import test_occurrence_inspection as occurrence

    service, sid, rid, *_ = occurrence.owned_revision.__wrapped__(tmp_path)

    class Client:
        def execute(self, directory, kind, timeout, cancelled):
            assert kind == "occurrence"
            occurrence.fake_render(
                ["python", "-m", OPERATIONS[kind][0], "request.json"], directory, timeout, cancelled
            )
            return {"operation_id": "a" * 32}

    service.executor = BrokerExecutor(Client(), "sha256:" + "b" * 64)
    service.executor._typed_container_verified = True
    service.executor.runner_verified = True
    monkeypatch.setattr(
        "shutil.which", lambda _: pytest.fail("Public process must not discover Docker")
    )
    assert occurrence.inspect_occurrence(service, sid, "alice", rid, "support")["image_base64"]
