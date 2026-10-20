"""Topology tagging — how the compiler labels build123d faces/edges so the
validator can query "what's the geometry of op_id=X?" later.

The scheme: every operation registers ``TopologyTag`` entries. Each tag
references the resulting build123d ``Face`` (or list of faces, for ops that
produce more than one canonical face — e.g. a Cylinder has a side face and two
end faces).

The validator does NOT serialize tags through JSON; tags only live in the
in-memory CADModel produced by Compiler. Downstream stages either consume the
CADModel directly or read the validator's report which references op_ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TopologyTag:
    """A handle from an op_id to one face of the resulting build123d body."""

    op_id: str
    role: str  # e.g. "top", "bottom", "side", "instance[0]", "wall"
    face: Any  # build123d.Face; typed as Any to keep this module backend-agnostic.


@dataclass
class TopologyIndex:
    """All topology tags for a compiled PartProgram, keyed by op_id."""

    tags: dict[str, list[TopologyTag]] = field(default_factory=dict)

    def add(self, op_id: str, role: str, face: Any) -> None:
        self.tags.setdefault(op_id, []).append(TopologyTag(op_id=op_id, role=role, face=face))

    def faces_for(self, op_id: str) -> list[Any]:
        """Return the raw build123d faces tagged for an op_id (empty if none)."""
        return [t.face for t in self.tags.get(op_id, [])]

    def roles_for(self, op_id: str) -> list[str]:
        return [t.role for t in self.tags.get(op_id, [])]

    def op_ids(self) -> list[str]:
        return list(self.tags.keys())


__all__ = ["TopologyIndex", "TopologyTag"]
