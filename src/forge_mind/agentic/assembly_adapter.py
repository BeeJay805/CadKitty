"""Bridge: agentic-authored parts (STL + STEP) -> the proven AssemblyModel.

Phase 3 reconnects the orphaned kinematic spine. The agentic author already emits,
per part, a watertight STL and an editable STEP (the sandbox writes ``<part>.step``).
This adapter wraps those files in the SAME ``AssemblyModel`` / ``PlacedBody`` shape the
DSL compiler produces, so ``validator.assembly_checks.validate_assembly`` -- with its
coaxial + clearance + pin-fit + 0-180 deg MOTION SWEEP -- runs on authored parts.

The one thing an authored body lacks is a ``PartProgram`` (the retired DSL): its
geometry came from arbitrary build123d code, not DSL ops. So ``PlacedBody.program`` is
left None and the validator's program-dependent checks fall back to geometry (rod
detection) or skip (NEMA-hub coupling). The headline motion sweep needs only the mesh +
transform + a declared port, so it runs unchanged.

The functional path authors every part in a SHARED ORIGIN MATE FRAME, so by default each
body carries the IDENTITY transform (parts overlay at the origin as authored). An explicit
``transforms`` map is accepted for the general case (e.g. solver-placed parts in a test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from numpy.typing import NDArray

from forge_mind.assembly.joints import Joint, Port
from forge_mind.compiler.compile_assembly import AssemblyModel, PlacedBody, PlacedHardware

if TYPE_CHECKING:
    from forge_mind.agentic.loop import LoopResult


@dataclass
class AuthoredPart:
    """One agentic-authored part: its mesh STL + (optional) editable STEP + declared ports."""

    id: str
    stl_path: Path
    step_path: Path | None = None
    ports: list[Port] = field(default_factory=list)


def _identity() -> NDArray[np.float64]:
    return np.eye(4, dtype=np.float64)


def _bbox_of(part: object) -> object:
    """A forge_mind BBox from a build123d part's bounding box (mirrors compiler.compile)."""
    from forge_mind.types import BBox, Vec3

    bb = part.bounding_box()  # type: ignore[attr-defined]
    return BBox(
        min=Vec3(x=bb.min.X, y=bb.min.Y, z=bb.min.Z),
        max=Vec3(x=bb.max.X, y=bb.max.Y, z=bb.max.Z),
    )


def cad_from_files(stl_path: Path, step_path: Path | None) -> object | None:
    """Build a ``CADModel`` (B-rep + paths) from an authored STEP, or None if it will not
    import. None makes the validator's cylinder-dependent checks (pin fit) degrade while
    the motion sweep -- which needs no B-rep -- still runs."""
    if step_path is None or not Path(step_path).exists():
        return None
    try:
        from build123d import import_step

        from forge_mind.compiler.compile import CADModel
        from forge_mind.compiler.topology_tags import TopologyIndex

        part = import_step(str(step_path))
    except Exception:
        return None
    try:
        bbox = _bbox_of(part)
        return CADModel(
            part=part,
            stl_path=Path(stl_path),
            step_path=Path(step_path),
            topology_index=TopologyIndex(),
            bounding_box=bbox,  # type: ignore[arg-type]
        )
    except Exception:
        return None


def _load_mesh(stl_path: Path) -> trimesh.Trimesh | None:
    try:
        mesh = trimesh.load(str(stl_path), force="mesh")
    except Exception:
        return None
    return mesh if isinstance(mesh, trimesh.Trimesh) else None


def build_assembly_model(
    authored: list[AuthoredPart],
    joints: list[Joint],
    *,
    transforms: dict[str, NDArray[np.float64]] | None = None,
    hardware: list[PlacedHardware] | None = None,
) -> AssemblyModel:
    """Wrap authored parts (+ declared joints) in the AssemblyModel ``validate_assembly`` eats.

    Each part becomes a ``PlacedBody`` with ``program=None`` (no DSL program) and a ``cad``
    imported from its STEP (None if unimportable). The transform defaults to identity (the
    shared mate frame); pass ``transforms`` to place parts explicitly. A part whose STL will
    not load is dropped (it cannot contribute geometry).

    ``hardware`` (the A7 servo bridge, plan §M3): declared non-printed nodes built by
    ``structures.servo_bridge.placed_hardware_for``. Passing them here is what lights the
    existing hardware-mount / coaxial / hardware-anchored-motion validators on the agentic
    path -- no validator changes, the model simply stops arriving hardware-blind.
    """
    tf = transforms or {}
    bodies: list[PlacedBody] = []
    for part in authored:
        mesh = _load_mesh(part.stl_path)
        if mesh is None:
            continue
        bodies.append(
            PlacedBody(
                id=part.id,
                mesh=mesh,
                transform=tf.get(part.id, _identity()),
                ports=list(part.ports),
                program=None,  # type: ignore[arg-type]  # agentic bodies have no LayoutSpec PartProgram
                cad=cad_from_files(part.stl_path, part.step_path),  # type: ignore[arg-type]
            )
        )
    return AssemblyModel(bodies=bodies, joints=list(joints), hardware=list(hardware or []))


def _reslug_joint(joint: Joint, name_to_id: dict[str, str]) -> Joint:
    """Rewrite a joint's '<part>.<port>' endpoints so the part component is the body id.

    A part NAME may itself contain '.', which would make a blind first-dot split ambiguous,
    so the part is resolved by the LONGEST declared name the endpoint starts with. The
    resulting body id (a slug) is dot-free, so downstream splits are unambiguous."""

    def remap(ref: str) -> str:
        for name in sorted(name_to_id, key=len, reverse=True):
            if ref == name:
                return ref  # a port-less endpoint; the joint filter will drop it
            if ref.startswith(name + "."):
                return f"{name_to_id[name]}.{ref[len(name) + 1 :]}"
        part, sep, port = ref.partition(".")  # unknown part: keep verbatim, will be dropped
        return f"{name_to_id.get(part, part)}.{port}" if sep else ref

    return joint.model_copy(update={"a": remap(joint.a), "b": remap(joint.b)})


def port_norm(s: str) -> str:
    """Separator-blind port-name normalization: 'main axis' == 'main-axis' == 'Main_Axis'
    (_slug keeps hyphens but spaces become underscores, so unify both onto one separator).
    THE port-matching predicate of this module -- a caller reasoning about 'will this
    joint resolve' must use it, never re-derive its own (the north-star adversarial
    review broke a re-implementation on exactly this)."""
    from forge_mind.agentic.loop import _slug

    return _slug(s).replace("-", "_")


def joint_referenced_ports(part_names: list[str], joints: list[Joint]) -> dict[str, set[str]]:
    """Per part NAME: the ``port_norm``-normalized port names the joints reference on it,
    attributed with the SAME longest-declared-prefix logic ``from_part_results`` resolves
    with (dotted part names included) -- so 'this part must declare these ports' is
    definitionally 'these joints resolve'."""
    from forge_mind.agentic.loop import _disambiguate_slugs

    name_to_id = dict(zip(part_names, _disambiguate_slugs(part_names), strict=True))
    id_to_name = {bid: name for name, bid in name_to_id.items()}
    out: dict[str, set[str]] = {}
    for joint in joints:
        rj = _reslug_joint(joint, name_to_id)
        for ref in (rj.a, rj.b):
            part, _, port = ref.partition(".")
            name = id_to_name.get(part)
            if name is not None and port:
                out.setdefault(name, set()).add(port_norm(port))
    return out


def from_part_results(
    part_results: list[tuple[str, LoopResult]],
    joints: list[Joint],
    ports_by_part: dict[str, list[Port]],
    notes: list[str] | None = None,
    *,
    hardware: list[PlacedHardware] | None = None,
) -> AssemblyModel:
    """The functional-path entry point: turn authored ``(name, LoopResult)`` parts + the
    LLM-declared joints/ports into an AssemblyModel in the shared mate frame (identity
    transforms). Names are slugged to body ids consistently with the per-part output dirs.

    Malformed LLM declarations are sanitized, not crashed on: a joint is dropped if it
    references a part that did not build or a port not declared on that part, so a bad joint
    cannot take down the whole functional run. Port references are matched
    SLUG-INSENSITIVELY ('main axis' finds the declared port 'main-axis'): the radial probe
    (row 58) declared 12 joints and the exact-string match dropped ALL of them silently.
    ``notes`` (optional sink) receives one line per dropped joint saying exactly why.
    """
    from forge_mind.agentic.loop import _disambiguate_slugs

    # Disambiguate ids the SAME way produce_functional_part names the per-part output dirs
    # (over the parts IN ORDER), so distinct names that slug-collide get the same unique ids
    # here as on disk -- the bodies stay 1:1 with the authored parts.
    names = [name for name, _lr in part_results]
    name_to_id = dict(zip(names, _disambiguate_slugs(names), strict=True))
    authored: list[AuthoredPart] = []
    declared_ports: dict[str, set[str]] = {}
    for name, lr in part_results:
        if lr.final_stl is None or not lr.final_stl.exists():
            continue
        bid = name_to_id[name]
        ports = list(ports_by_part.get(name, []))
        authored.append(
            AuthoredPart(
                id=bid,
                stl_path=lr.final_stl,
                step_path=lr.final_step,
                ports=ports,
            )
        )
        declared_ports[bid] = {p.name for p in ports}

    built_ids = {p.id for p in authored}
    # The A7 bridge: declared hardware nodes are legitimate joint endpoints. Their DATASHEET
    # ports join the same resolution tables, so 'shoulder_servo.horn' resolves exactly like a
    # printed part's port (and a typo'd hardware port drops with the same note, not a crash).
    for hw in hardware or []:
        built_ids.add(hw.id)
        declared_ports[hw.id] = {p.name for p in hw.ports}

    slug_ports: dict[str, dict[str, str]] = {
        bid: {port_norm(p): p for p in port_names} for bid, port_names in declared_ports.items()
    }

    def _resolve_port(part: str, port: str) -> str | None:
        declared = declared_ports.get(part, set())
        if port in declared:
            return port
        return slug_ports.get(part, {}).get(port_norm(port))

    kept: list[Joint] = []
    for joint in joints:
        rj = _reslug_joint(joint, name_to_id)
        a_part, _, a_port = rj.a.partition(".")
        b_part, _, b_port = rj.b.partition(".")
        if a_part not in built_ids or b_part not in built_ids:
            if notes is not None:
                missing = a_part if a_part not in built_ids else b_part
                notes.append(f"joint {joint.a} <-> {joint.b}: part '{missing}' did not build")
            continue
        a_canon = _resolve_port(a_part, a_port)
        b_canon = _resolve_port(b_part, b_port)
        if a_canon is None or b_canon is None:
            if notes is not None:
                part, port = (a_part, a_port) if a_canon is None else (b_part, b_port)
                notes.append(
                    f"joint {joint.a} <-> {joint.b}: port '{port}' not declared on "
                    f"'{part}' (declared: {sorted(declared_ports.get(part, set()))})"
                )
            continue
        if a_canon != a_port or b_canon != b_port:
            rj = rj.model_copy(update={"a": f"{a_part}.{a_canon}", "b": f"{b_part}.{b_canon}"})
        kept.append(rj)

    return build_assembly_model(authored, kept, hardware=hardware)


__all__ = [
    "AuthoredPart",
    "build_assembly_model",
    "cad_from_files",
    "from_part_results",
    "joint_referenced_ports",
    "port_norm",
]
