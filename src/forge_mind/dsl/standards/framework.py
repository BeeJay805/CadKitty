"""Standards-backed hardware fact framework — the ONE mechanism every category uses.

Design goals (the part to get right):
  - ONE typed source of truth for every hardware dimension; the LLM (and the future composition
    engine) picks a component + size + fit, and the registry supplies the exact geometry. No
    dimension is ever guessed.
  - Adding a SIZE = one frozen data row. Adding a CATEGORY = one ``Registry`` instance over a new
    frozen ``StandardSpec`` subclass. No new framework code either way -- categories are data.
  - PROVENANCE is mandatory: every spec carries its governing ``source`` (cited standard or
    manufacturer reference) and a ``provisional`` flag. ``provisional=True`` means the value is
    NOT from a published standard (a manufacturer recommendation or an FDM rule-of-thumb) and must
    be verified before production -- it is surfaced, never hidden.

This module is pure data + lookup (no geometry); placement lives in ``layout.hardware`` and the
conformance gate in ``validator.conformance``, mirroring the existing dsl/layout/validator split.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class Fit(StrEnum):
    """Clearance fit class. The names map to ISO 273's fine / medium / coarse series."""

    CLOSE = "close"  # ISO 273 "fine" (H12)
    NORMAL = "normal"  # ISO 273 "medium" (H13)
    COARSE = "coarse"  # ISO 273 "coarse" (H14)


class StandardSpec(BaseModel):
    """Base for every hardware fact: a key, the cited source, and the provisional flag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(
        ..., min_length=1, description="Lookup key within its category, e.g. 'M3' or '608'."
    )
    source: str = Field(
        ..., min_length=1, description="Governing standard or manufacturer reference (cited)."
    )
    provisional: bool = Field(
        default=False,
        description="True = NOT from a published standard (manufacturer/rule-of-thumb); verify before production.",
    )


SpecT = TypeVar("SpecT", bound=StandardSpec)


class Registry(Generic[SpecT]):
    """A category's frozen, cited fact table resolved by key -- the single mechanism for every
    category (parallel to the board/fan/motor/extrusion registries, generalized)."""

    def __init__(self, name: str, entries: tuple[SpecT, ...]) -> None:
        self.name = name
        self._by_key: dict[str, SpecT] = {}
        for e in entries:
            if e.key in self._by_key:
                raise ValueError(f"duplicate {name} key {e.key!r}")
            self._by_key[e.key] = e

    @staticmethod
    def _norm(key: str | int | float) -> str:
        """Normalize a lookup key: ints and integral floats become their string form.

        The registry is string-keyed ('6', 'M3', '608'); the AGENTIC author naturally
        writes ``dowel_hole_d(6)`` and used to get the gaslighting error "unknown
        dowel_pin 6; known: [... '6' ...]" (caught live on the robotic-arm run). Pure
        coercion -- a coerced key resolves to the SAME cited row, so no-false-PASS is
        untouched; an unknown size still raises with the known list."""
        if isinstance(key, float) and key.is_integer():
            key = int(key)
        return str(key).strip()

    def get(self, key: str | int | float) -> SpecT:
        k = self._norm(key)
        try:
            return self._by_key[k]
        except KeyError:
            # Case-insensitive fallback ('m3' -> 'M3') ONLY when it is unambiguous --
            # the same row a correctly-cased call returns, never a guess between rows.
            ci = [v for kk, v in self._by_key.items() if kk.lower() == k.lower()]
            if len(ci) == 1:
                return ci[0]
            raise ValueError(
                f"unknown {self.name} {key!r}; known: {sorted(self._by_key)}"
            ) from None

    def has(self, key: str | int | float) -> bool:
        k = self._norm(key)
        if k in self._by_key:
            return True
        return sum(1 for kk in self._by_key if kk.lower() == k.lower()) == 1

    def keys(self) -> list[str]:
        return list(self._by_key)

    def all(self) -> tuple[SpecT, ...]:
        return tuple(self._by_key.values())


__all__ = ["Fit", "Registry", "StandardSpec"]
