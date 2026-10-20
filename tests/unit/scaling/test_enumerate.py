"""Step 1 (ADR-0016): deterministic catalog enumeration + pre-trial feasibility prune."""

from __future__ import annotations

from forge_mind.layout.composable_spec import BaseArchetype, Face, InterfaceRef
from forge_mind.scaling.enumerate import (
    FamilyShape,
    Placement,
    enumerate_shapes,
    feasible_shapes,
    is_feasible,
)

_NEMA_T = Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP)
_NEMA_B = Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_BOTTOM)
_EXT_T = Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_TOP)
_EXT_B = Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_BOTTOM)
_FAN60_T = Placement(InterfaceRef.FAN_FACE, Face.PLATE_TOP, "fan_60mm")
_FAN60_B = Placement(InterfaceRef.FAN_FACE, Face.PLATE_BOTTOM, "fan_60mm")
_FAN40_T = Placement(InterfaceRef.FAN_FACE, Face.PLATE_TOP, "fan_40mm")

_ADAPTER = FamilyShape(BaseArchetype.FLAT_PLATE, (_NEMA_T, _EXT_B))
_FAN_MOUNT = FamilyShape(BaseArchetype.FLAT_PLATE, (_FAN60_T, _EXT_B))
_MOTOR_FAN = FamilyShape(BaseArchetype.FLAT_PLATE, (_NEMA_T, _FAN60_B))
_DUAL_MOTOR = FamilyShape(BaseArchetype.FLAT_PLATE, (_NEMA_T, _NEMA_B))
_L_BRACKET = FamilyShape(
    BaseArchetype.L_BRACKET_BEND,
    (
        Placement(InterfaceRef.NEMA17_FACE, Face.BASE_TOP),
        Placement(InterfaceRef.EXTRUSION_2020, Face.UPRIGHT_FRONT),
    ),
)
_FAN40_PLATE = FamilyShape(BaseArchetype.FLAT_PLATE, (_FAN40_T,))
_DUAL_EXT = FamilyShape(BaseArchetype.FLAT_PLATE, (_EXT_T, _EXT_B))


def test_enumeration_is_finite_and_modest() -> None:
    shapes = enumerate_shapes()
    keys = {s.key for s in shapes}
    # The bounded registry vocabulary (ADR-0017: 3 motors x 3 extrusions x 2 fans across faces
    # + pairs) yields a finite catalog -- larger than the pre-registry 36, still bounded
    # (honest: NOT thousands).
    assert 80 <= len(shapes) <= 250
    assert len(keys) == len(shapes)  # no duplicate shapes


def test_dual_extrusion_now_enumerated_and_feasible() -> None:
    # ADR-0017: the extrusion checker is now localized, so two extrusion interfaces ARE
    # certifiable -- the enumerator proposes them (it excluded them pre-localization).
    assert _DUAL_EXT.key in {s.key for s in enumerate_shapes()}
    assert _DUAL_EXT.key in {s.key for s in feasible_shapes()}


def test_registry_hardware_is_enumerated_and_feasible() -> None:
    # ADR-0017 multiplier: new hardware (NEMA 23, 3030) and their combos appear as feasible
    # families -- pure config, no new interpreter/checker code.
    feas = {s.key for s in feasible_shapes()}
    nema23 = FamilyShape(
        BaseArchetype.FLAT_PLATE, (Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP, "nema23"),)
    )
    nema23_on_3030 = FamilyShape(
        BaseArchetype.FLAT_PLATE,
        (
            Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP, "nema23"),
            Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_BOTTOM, "ext_3030"),
        ),
    )
    assert nema23.key in feas, "NEMA 23 plate should be a feasible catalog family"
    assert nema23_on_3030.key in feas, "NEMA 23 + 3030 should be a feasible catalog family"


def test_controls_are_feasible() -> None:
    feasible = {s.key for s in feasible_shapes()}
    for ctrl in (_ADAPTER, _FAN_MOUNT, _MOTOR_FAN, _DUAL_MOTOR, _L_BRACKET):
        assert ctrl.key in feasible, f"{ctrl.key} should survive the feasibility prune"


def test_known_infeasible_is_pruned_pre_trial() -> None:
    # A 40 mm fan's airflow bore can't keep its bolt land at a 4 mm wall (ADR-0011): the
    # deterministic prune drops it for FREE, before any paid trial.
    assert not is_feasible(_FAN40_PLATE)
    assert _FAN40_PLATE.key not in {s.key for s in feasible_shapes()}
    # ...while the 60 mm fan plate is comfortably feasible.
    assert is_feasible(FamilyShape(BaseArchetype.FLAT_PLATE, (_FAN60_T,)))
