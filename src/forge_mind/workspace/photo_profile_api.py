"""One shared owned contract for author, HTTP and MCP photo observations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from forge_mind.workspace.models import Record
from forge_mind.workspace.photo_profile_models import PhotoProfileRequest, PhotoProfileSelector
from forge_mind.workspace.photo_profile_service import (
    analyze_reference,
    compare_reference,
    read_profile,
)


class PhotoProfileAction(Record):
    action: Literal["analyze", "read", "compare"]
    request: PhotoProfileRequest | None = None
    observation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    revision_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    selector: PhotoProfileSelector | None = None

    @model_validator(mode="after")
    def required_fields(self) -> PhotoProfileAction:
        if self.action == "analyze":
            if self.request is None or any((self.observation_id, self.revision_id, self.selector)):
                raise ValueError("analyze requires only request")
        elif self.observation_id is None or self.request is not None:
            raise ValueError("read/compare require observation_id and no request")
        elif self.action == "compare" and (self.revision_id is None or self.selector is None):
            raise ValueError("compare requires exact revision_id and selector")
        elif self.action == "read" and (self.revision_id is not None or self.selector is not None):
            raise ValueError("read takes only observation_id")
        return self


def execute_photo_profile(
    service: Any,
    sid: str,
    owner: str,
    body: PhotoProfileAction | dict[str, Any],
    *,
    timeout: float = 30,
    cancelled: Any = None,
) -> dict[str, Any]:
    body = PhotoProfileAction.model_validate(body)
    if body.action == "analyze":
        assert body.request is not None
        return analyze_reference(
            service, sid, owner, body.request, timeout=timeout, cancelled=cancelled
        )
    assert body.observation_id is not None
    if body.action == "read":
        return read_profile(service, sid, owner, body.observation_id)
    assert body.revision_id is not None and body.selector is not None
    return compare_reference(
        service,
        sid,
        owner,
        body.observation_id,
        body.revision_id,
        body.selector,
        timeout=timeout,
        cancelled=cancelled,
    )
