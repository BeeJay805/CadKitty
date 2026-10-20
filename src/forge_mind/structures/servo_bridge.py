"""The A7 servo bridge (plan §M3): a declared hardware kind becomes a PlacedHardware node.

The load-bearing recon fact (ADR-0096 program notes): the assembly lane has COMPLETE MG996R
validation -- datasheet ports, a collision envelope, cited mount patterns
(``validator.hardware_mounts.patterns_for``) -- but the agentic path never passes ``hardware=``,
so none of it runs there. This bridge closes that gap with NO validator changes: a
``HardwareDecl`` (kind + world transform) resolves through the SAME lane-2 factories
(``assembly.servo``) into the ``PlacedHardware`` the validators already eat. One source of truth
for every servo number; the mount/coaxial/motion gates light up the moment a declaration exists.

Fail-closed: an unknown kind raises (a guessed servo is the false-precision vector -- same rule
as ``structures.hardware.footprint_for``, which serves the 2-D topology domain from the same
registry idea). The collision ENVELOPE compile is best-effort exactly like lane 2's compiler:
a failed envelope degrades to the frames-only shape (the motion gate WARNs, never crashes).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forge_mind.assembly.servo import mg996r_envelope_program, mg996r_ports
from forge_mind.compiler.compile_assembly import PlacedHardware

# kind -> (ports factory, envelope-program factory). The SAME factories lane 2 compiles from.
_HW_REGISTRY: dict[str, tuple[Callable[[], list[Any]], Callable[[], Any]]] = {
    "mg996r_servo": (mg996r_ports, mg996r_envelope_program),
}


def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


@dataclass(frozen=True)
class HardwareDecl:
    """One declared real component in an agentic assembly: WHAT it is and WHERE it sits.

    ``transform`` is the 4x4 world placement of the hardware node's local frame (the servo
    convention: case top at z=0, shaft on +Z -- see ``assembly.servo``). Identity = the shared
    mate frame origin."""

    kind: str
    hw_id: str
    transform: Any = field(default_factory=_identity)  # 4x4; ndarray or nested sequence


def known_hardware_kinds() -> list[str]:
    return sorted(_HW_REGISTRY)


def placed_hardware_for(decl: HardwareDecl, *, workdir: Path | None = None) -> PlacedHardware:
    """Resolve a declaration into the ``PlacedHardware`` the assembly validators consume.

    Ports come from the registry factory (datasheet-cited). The collision envelope is compiled
    best-effort when a workdir is given: success -> the motion gate can sweep hardware-anchored
    joints; failure -> frames-only (WARN'd sweep), exactly lane 2's degradation."""
    try:
        ports_fn, env_fn = _HW_REGISTRY[decl.kind]
    except KeyError:
        known = ", ".join(known_hardware_kinds())
        raise KeyError(f"unknown hardware kind {decl.kind!r}; known: {known}") from None

    hw_mesh: Any = None
    hw_stl: Path | None = None
    if workdir is not None:
        try:
            import trimesh

            from forge_mind.compiler.compile import compile_program

            cad = compile_program(env_fn(), workdir / decl.hw_id)
            loaded = trimesh.load(str(cad.stl_path), force="mesh")
            if isinstance(loaded, trimesh.Trimesh):
                hw_mesh, hw_stl = loaded, Path(cad.stl_path)
        except Exception:  # pragma: no cover - envelope is evidence, not the deliverable
            hw_mesh, hw_stl = None, None

    return PlacedHardware(
        id=decl.hw_id,
        transform=np.asarray(decl.transform, dtype=np.float64),
        ports=ports_fn(),
        mesh=hw_mesh,
        stl_path=hw_stl,
        kind=decl.kind,
    )


__all__ = ["HardwareDecl", "known_hardware_kinds", "placed_hardware_for"]
