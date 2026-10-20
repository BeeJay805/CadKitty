"""Blueprint schema: the interface-modeling root fix (Step 1).

Deterministic guards for the two blueprint root causes the intent-fidelity gate depends on:
(a) a BARE structural member may carry zero required_interfaces (no forced spurious m5_tnut), and
(b) the blueprint vocabulary can NAME every standards hardware feature, so a heat-set insert /
bearing seat / magnet pocket / dowel maps to its OWN ref instead of being mis-mapped to a bolt.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge_mind.blueprint.schema import (
    Blueprint,
    BlueprintVec3,
    FeatureSpec,
    InterfaceSpec,
    RequestedInterface,
)


def _common() -> dict[str, object]:
    return {
        "name": "p",
        "summary": "s",
        "envelope_mm": BlueprintVec3(x=120, y=20, z=20),
        "print_orientation": BlueprintVec3(x=1, y=0, z=0),
        "material": "PLA",
        "min_wall_mm": 3.0,
        "max_overhang_deg": 45.0,
        "features": [
            FeatureSpec(
                name="body",
                kind="strut",
                purpose="span",
                approx_size_mm=BlueprintVec3(x=120, y=14, z=14),
            )
        ],
    }


def test_bare_member_allows_zero_interfaces() -> None:
    # A plain strut carries NO interface -> the empty list is valid (Step 1a: no forced m5_tnut).
    bp = Blueprint(**_common(), required_interfaces=[])  # type: ignore[arg-type]
    assert bp.required_interfaces == []


def test_required_interfaces_field_is_still_required() -> None:
    # Relaxing min_length must not make the key optional: a missing key is still an error (the LLM
    # must consciously emit [] for a bare member, not silently omit it).
    with pytest.raises(ValidationError):
        Blueprint(**_common())  # type: ignore[arg-type]


def test_hardware_vocabulary_is_nameable() -> None:
    # Step 1b: every standards hardware feature has its OWN ref (not mis-mapped to m5_tnut).
    for ref in (
        RequestedInterface.CLEARANCE_HOLE,
        RequestedInterface.COUNTERBORE,
        RequestedInterface.HEAT_SET_INSERT,
        RequestedInterface.CAPTIVE_NUT_POCKET,
        RequestedInterface.BEARING_SEAT,
        RequestedInterface.DOWEL_HOLE,
        RequestedInterface.MAGNET_POCKET,
    ):
        spec = InterfaceSpec(
            name="hw", interface_type=ref, on_feature="body", facing="+Z", component_id="608"
        )
        assert spec.interface_type is ref
        assert spec.component_id == "608"


def test_vocabulary_is_a_superset_of_the_validator_catalog() -> None:
    # The legacy face/board spellings must survive verbatim so existing blueprints/fixtures parse.
    from forge_mind.dsl.interfaces.declaration import InterfaceType

    requested = {r.value for r in RequestedInterface}
    assert {t.value for t in InterfaceType} <= requested
