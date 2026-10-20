"""Intent-fidelity gate (Step 2): a dropped or mismapped requested feature must REFUSE.

No-false-PASS at the intent level. These are deterministic (no LLM): they build a Blueprint contract
and a PartProgram by hand and assert the gate refuses a drop/mismap while passing a faithful build.
"""

from __future__ import annotations

from forge_mind.blueprint.schema import (
    Blueprint,
    BlueprintVec3,
    FeatureSpec,
    InterfaceSpec,
    RequestedInterface,
)
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature
from forge_mind.validator.fidelity import (
    _CANONICAL,
    built_kinds,
    check_fidelity,
    requested_kinds,
)


def _bp(*interfaces: InterfaceSpec) -> Blueprint:
    return Blueprint(
        name="p",
        summary="s",
        envelope_mm=BlueprintVec3(x=120, y=60, z=20),
        print_orientation=BlueprintVec3(x=0, y=0, z=1),
        material="PLA",
        min_wall_mm=4.0,
        max_overhang_deg=45.0,
        features=[
            FeatureSpec(
                name="body",
                kind="plate",
                purpose="host",
                approx_size_mm=BlueprintVec3(x=120, y=60, z=4),
            )
        ],
        required_interfaces=list(interfaces),
    )


def _iface(ref: RequestedInterface, cid: str | None = None) -> InterfaceSpec:
    return InterfaceSpec(
        name=ref.value, interface_type=ref, on_feature="body", facing="+Z", component_id=cid
    )


def _program(
    *,
    decls: list[InterfaceDeclaration] | None = None,
    hw: list[HardwareFeature] | None = None,
) -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="p"),
        operations=[
            Box(
                op_id="plate",
                mode=OpMode.ADD,
                position=Vec3(x=0, y=0, z=2),
                width=120,
                depth=60,
                height=4,
            )
        ],
        interfaces=decls or [],
        hardware_features=hw or [],
        constraints=Constraints(min_wall_thickness=4.0),
    )


def _decl(t: InterfaceType) -> InterfaceDeclaration:
    return InterfaceDeclaration(
        name=t.value, interface_type=t, location=Vec3(x=0, y=0, z=4), normal=Vec3(x=0, y=0, z=1)
    )


def _hw(category: str, key: str) -> HardwareFeature:
    return HardwareFeature(category=category, key=key, nominal_mm=1.0, where="t")


# --- drops + mismaps REFUSE -------------------------------------------------


def test_dropped_hardware_refuses() -> None:
    # strut+inserts routed to a path that can't host hardware: heat-set insert dropped -> refuse.
    bp = _bp(_iface(RequestedInterface.HEAT_SET_INSERT, "M3"))
    program = _program()  # a bare body, no interfaces built
    msg = check_fidelity(bp, program)
    assert msg is not None
    assert "heat-set insert" in msg


def test_mismapped_type_refuses() -> None:
    # asked for a bearing seat, built a clearance hole instead -> different kind -> refuse.
    bp = _bp(_iface(RequestedInterface.BEARING_SEAT, "608"))
    program = _program(hw=[_hw("clearance_hole", "M3")])
    msg = check_fidelity(bp, program)
    assert msg is not None
    assert "bearing seat" in msg


def test_partial_drop_among_multiple_refuses() -> None:
    # fan plate + M4 holes: fan built, the clearance holes dropped -> refuse on the missing kind.
    bp = _bp(
        _iface(RequestedInterface.FAN_FACE, "fan_60mm"),
        _iface(RequestedInterface.CLEARANCE_HOLE, "M4"),
    )
    program = _program(decls=[_decl(InterfaceType.FAN_FACE)])  # fan built, holes missing
    msg = check_fidelity(bp, program)
    assert msg is not None
    assert "clearance hole" in msg


# --- faithful builds PASS (no false-refuse) ---------------------------------


def test_faithful_composition_passes() -> None:
    bp = _bp(
        _iface(RequestedInterface.BEARING_SEAT, "625"),
        _iface(RequestedInterface.MAGNET_POCKET, "6x3"),
    )
    program = _program(hw=[_hw("bearing_seat", "625"), _hw("magnet_pocket", "6x3")])
    assert check_fidelity(bp, program) is None


def test_bare_member_passes() -> None:
    # No interfaces requested, none built -> a bare strut is faithful, not a refusal.
    assert check_fidelity(_bp(), _program()) is None


def test_extrusion_spelling_coalesces() -> None:
    # blueprint says extrusion_2020_slot; program declares the same slot type -> match.
    bp = _bp(_iface(RequestedInterface.EXTRUSION_2020_SLOT))
    program = _program(decls=[_decl(InterfaceType.EXTRUSION_2020_SLOT)])
    assert check_fidelity(bp, program) is None


def test_pi4_ports_omitted_does_not_false_refuse() -> None:
    # A closed tray legitimately omits the conditional pi4_ports; requesting it must still pass
    # because pi4_mount + pi4_ports coalesce to one "pi4" kind.
    bp = _bp(_iface(RequestedInterface.PI4_MOUNT), _iface(RequestedInterface.PI4_PORTS))
    program = _program(decls=[_decl(InterfaceType.PI4_MOUNT)])  # ports not built
    assert check_fidelity(bp, program) is None


def test_extra_built_interface_is_allowed() -> None:
    # The part may build MORE than requested (e.g. a derived feature) without refusing.
    bp = _bp(_iface(RequestedInterface.NEMA17_FACE))
    program = _program(
        decls=[_decl(InterfaceType.NEMA17_FACE), _decl(InterfaceType.EXTRUSION_2020_SLOT)]
    )
    assert check_fidelity(bp, program) is None


# --- no silent vocabulary gaps (drift guards) -------------------------------


def test_every_requested_vocab_value_is_mapped() -> None:
    # A new RequestedInterface value without a canonical mapping would be a silent false-PASS hole.
    for ref in RequestedInterface:
        assert ref.value in _CANONICAL, f"{ref.value} has no canonical kind"


def test_every_built_vocab_value_is_mapped() -> None:
    # Face declaration types and the hardware categories the composable interpreter emits must all
    # canonicalize, or a real built feature would not match its request.
    from forge_mind.layout.composable_hardware import _CATEGORY

    for t in InterfaceType:
        assert t.value in _CANONICAL, f"declaration {t.value} has no canonical kind"
    for category in _CATEGORY.values():
        assert category in _CANONICAL, f"hardware category {category} has no canonical kind"


def test_requested_and_built_kinds_helpers() -> None:
    bp = _bp(_iface(RequestedInterface.BEARING_SEAT, "608"))
    assert requested_kinds(bp) == {"bearing"}
    assert built_kinds(_program(hw=[_hw("bearing_seat", "608")])) == {"bearing"}
