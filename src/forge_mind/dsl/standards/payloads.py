"""Payload-object facts: cited dimensions of REAL objects a mechanism interacts with.

A gripper that claims to grip "a soda can" must measure its aperture against the CAN'S
cited geometry, not a gen-time guess -- payload objects are registry facts exactly like
fasteners and bearings (the grounded-ingest red line, ADR-0044). v1 carries the one
object the gripper benchmark demands: the 12 oz / 355 ml beverage can.

Citation notes (researched + cross-checked 2026-06-10, grounded-ingest protocol):
  * Body diameter: the US "211" can body = 2 11/16 in = 66.0 mm nominal (Can
    Manufacturers Institute size designation; Crown Holdings "12oz/355ml Standard"
    product spec). Manufacturing tolerance is < 0.5 mm on diameter.
  * Height: the "413" standard body = 4 13/16 in = 122.2 mm nominal (same sources).
  * Masses: filled ~390 g (355 ml of ~1.04 g/ml sugared beverage + ~15 g can);
    empty can ~15 g. These are CONSERVATIVE planning bounds, not a published standard
    (density varies by beverage) -- hence provisional, like every entry here until a
    physical sample is measured. Payload sanity arithmetic must use the FULL mass.
"""

from __future__ import annotations

from pydantic import Field

from forge_mind.dsl.standards.framework import Registry, StandardSpec


class PayloadObject(StandardSpec):
    """A real-world object a mechanism is asked to handle: cited envelope + mass bounds.

    ``grip_diameter_mm`` is the diameter a radial gripper closes on (the body, not the
    narrower lid); ``mass_full_g`` is the conservative planning mass (heaviest state).
    """

    grip_diameter_mm: float = Field(..., gt=0)
    height_mm: float = Field(..., gt=0)
    mass_full_g: float = Field(..., gt=0)
    mass_empty_g: float = Field(..., gt=0)


PAYLOAD_OBJECTS: Registry[PayloadObject] = Registry(
    "payload_object",
    (
        PayloadObject(
            key="beverage_can_355ml",
            grip_diameter_mm=66.0,  # "211" body, 2 11/16 in nominal
            height_mm=122.2,  # "413" body, 4 13/16 in nominal
            mass_full_g=390.0,  # 355 ml + can, conservative upper bound
            mass_empty_g=15.0,
            source=(
                "CMI 211/413 can size designation; Crown Holdings 12oz/355ml Standard "
                "beverage can spec (crowncork.com); masses derived 355ml @ ~1.04 g/ml + can"
            ),
            provisional=True,  # nominal industry figures, no physical sample measured
        ),
    ),
)


def payload_object(key: str) -> PayloadObject:
    """The cited payload-object facts for ``key`` (e.g. 'beverage_can_355ml')."""
    return PAYLOAD_OBJECTS.get(key)


__all__ = ["PAYLOAD_OBJECTS", "PayloadObject", "payload_object"]
