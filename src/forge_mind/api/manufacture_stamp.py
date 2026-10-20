"""Refusal-honesty stamp: never present a part as manufacturable when the gates said otherwise.

Extracted from ``server/app.py`` so it is TESTABLE. It lived behind a module-level
``import fastapi``, which the local venv does not have, so `tests/unit/server` cannot collect and
this decision -- which controls whether the UI warns a user that a part must not be printed --
shipped verified only by reading. Pure decision logic should not be untestable because of a web
framework import.

It is a DENY-LIST on specific dispositions, which is the trap worth stating: a NEW disposition
string fails OPEN here (no stamp, no warning), while the allow-list consumers elsewhere in the
codebase fail CLOSED on the same unknown string. Both shapes exist in this repo, so neither can be
assumed -- every new non-shipping disposition must be added here explicitly.
"""

from __future__ import annotations

from typing import Any

_NOT_PASSED = (
    "This part did not pass validation -- delivered for inspection only, NOT for manufacture."
)
_UNASSEMBLABLE = (
    "The parts are individually sound but the assembly cannot be built as designed (it does not "
    "seat, a part floats free, or a mating fit binds as printed) -- delivered for inspection "
    "only, NOT for manufacture."
)
_REJECTED = (
    "This part passed the geometry gates but the acceptance review judged it does not match the "
    "request -- delivered for inspection only."
)
_MOTION_BLOCKED = (
    "The parts assemble, but the mechanism cannot move through its declared range: a measured "
    "collision blocks it -- delivered for inspection only, NOT for manufacture."
)


def manufacture_stamp(payload: dict[str, Any]) -> tuple[bool, str | None]:
    """``(not_for_manufacture, warning)`` for a response payload.

    A part that BUILT but did not pass validation, or that the acceptance judge REJECTED against
    the brief (ADR-0035 B8), or whose ASSEMBLY is physically impossible (ADR-0126), or whose
    MECHANISM cannot move through its declared range (ADR-0127) is delivered for inspection and
    clearly stamped. The file is still served -- serve-but-stamp -- so a contract ``status:"ok"``
    can never read as "a good, manufacturable part".

    The `not_assemblable` branch is separate from `rejected` because the reason differs and users
    read the reason: the acceptance review did NOT reject those parts, the physics did. For the
    same reason `motion_blocked` is separate from `not_assemblable` -- those parts DO go together.
    """
    passed = bool((payload.get("validation") or {}).get("passed", True))
    disposition = payload.get("disposition")
    rejected = disposition == "rejected"
    unassemblable = disposition == "not_assemblable"
    motion_blocked = disposition == "motion_blocked"
    if not passed:
        return True, _NOT_PASSED
    if unassemblable:
        return True, _UNASSEMBLABLE
    if motion_blocked:
        return True, _MOTION_BLOCKED
    if rejected:
        return True, _REJECTED
    return False, None


__all__ = ["manufacture_stamp"]
