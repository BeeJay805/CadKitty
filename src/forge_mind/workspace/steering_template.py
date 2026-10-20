"""Editable trapezoid steering development reference, without hardware capacity claims."""

from __future__ import annotations

import math

from pydantic import Field

from forge_mind.workspace.document import (
    AssemblySpec,
    DesignDocument,
    Feature,
    InterfaceMate,
    Occurrence,
    PartDefinition,
    PartPort,
    Subassembly,
    VariableRelation,
)
from forge_mind.workspace.mechanism_models import Component, Driver, Mechanism, PlanarJoint
from forge_mind.workspace.models import Record


class SteeringTemplate(Record):
    track_width: float = Field(default=600, ge=400, le=1000)
    wheel_diameter: float = Field(default=200, ge=120, le=300)
    steering_angle: float = Field(default=0, ge=-25, le=25)


def steering_document(setup: SteeringTemplate) -> DesignDocument:
    parts: list[PartDefinition] = []

    def ring(
        pid: str, name: str, diameter: float, bore: float, length: float, wheel: bool = False
    ) -> None:
        parts.append(
            PartDefinition(
                id=pid,
                name=name,
                features=[
                    Feature(
                        id="body",
                        name=name,
                        kind="bearing",
                        params={"diameter": diameter, "bore": bore, "width": length},
                        bindings={"diameter": "wheel_diameter"} if wheel else {},
                    ),
                ],
                output="body",
                ports=[PartPort(name="bore", kind="bore", diameter=bore)],
            )
        )

    ring("wheel", "Wheel envelope", setup.wheel_diameter, 70.4, 60, True)
    ring("hub", "Bored wheel hub", 70, 20.4, 60)
    ring("bearing", "Bearing envelope, unrated", 20, 12.4, 8)
    ring("washer", "Axle washer, retention unresolved", 24, 12.4, 2)
    ring("pivot_washer", "Kingpin washer", 24, 10.4, 2)
    ring("support", "Kingpin support sleeve", 28, 10.4, 16)
    support = parts[-1]
    support.features[0].id = "sleeve"
    support.features.extend(
        [
            Feature(
                id="boss", name="Frame contact pad", kind="box", params={"x": 14, "y": 14, "z": 16}
            ),
            Feature(
                id="pad",
                name="Contact face at frame",
                kind="transform",
                inputs=["boss"],
                params={"y": 19},
            ),
            Feature(
                id="body", name="Supported kingpin sleeve", kind="union", inputs=["sleeve", "pad"]
            ),
        ]
    )
    for pid, diameter, length in [("axle", 12, 116), ("kingpin", 10, 64), ("rod_pin", 10, 28)]:
        parts.append(
            PartDefinition(
                id=pid,
                name=pid.replace("_", " ").title(),
                features=[
                    Feature(
                        id="body",
                        name=pid,
                        kind="shaft",
                        params={"diameter": diameter, "length": length},
                    )
                ],
                output="body",
                ports=[PartPort(name="seat", kind="shaft", diameter=diameter)],
            )
        )
    arm_length = math.hypot(25, 75)
    for pid, angle in [
        ("left_arm", math.degrees(math.atan2(-75, 25))),
        ("right_arm", math.degrees(math.atan2(-75, -25))),
    ]:
        parts.append(
            PartDefinition(
                id=pid,
                name=pid.replace("_", " ").title(),
                features=[
                    Feature(
                        id="link",
                        name="Bored steering arm",
                        kind="link",
                        params={"length": arm_length, "width": 22, "thickness": 10, "bore": 10.4},
                    ),
                    Feature(
                        id="body",
                        name="Arm direction",
                        kind="transform",
                        inputs=["link"],
                        params={"rz": angle},
                    ),
                ],
                output="body",
                ports=[
                    PartPort(name="kingpin", kind="bore", diameter=10.4),
                    PartPort(
                        name="rod",
                        kind="bore",
                        diameter=10.4,
                        origin=(25 if pid == "left_arm" else -25, -75, 0),
                    ),
                ],
            )
        )
    parts.append(
        PartDefinition(
            id="tie_rod",
            name="Bored tie rod",
            features=[
                Feature(
                    id="body",
                    name="Tie rod",
                    kind="link",
                    params={"width": 18, "thickness": 8, "bore": 10.4},
                    bindings={"length": "tie_length"},
                )
            ],
            output="body",
            ports=[
                PartPort(name="left", kind="bore", diameter=10.4),
                PartPort(name="right", kind="bore", diameter=10.4, origin=("tie_length", 0, 0)),
            ],
        )
    )
    parts.append(
        PartDefinition(
            id="crossmember",
            name="Hollow frame crossmember",
            features=[
                Feature(
                    id="outer",
                    name="Tube outside",
                    kind="box",
                    params={"y": 24, "z": 24},
                    bindings={"x": "track_width"},
                ),
                Feature(
                    id="inner",
                    name="Tube inside",
                    kind="box",
                    params={"y": 18, "z": 18},
                    bindings={"x": "tube_cut_length"},
                ),
                Feature(id="body", name="Hollow section", kind="cut", inputs=["outer", "inner"]),
            ],
            output="body",
        )
    )
    # A knuckle is one manufactured solid. The spindle block overlaps the arm
    # during construction and is fused, so the physical assembly has no hidden
    # block/arm intersection and fits between the grounded kingpin supports.
    for side, sign in [("left", -1), ("right", 1)]:
        knuckle = next(p for p in parts if p.id == side + "_arm")
        knuckle.features[-1].id = "arm"
        knuckle.features.extend(
            [
                Feature(
                    id="spindle_blank",
                    name="Spindle support",
                    kind="box",
                    params={"x": 24, "y": 24, "z": 20},
                ),
                Feature(
                    id="spindle",
                    name="Spindle position",
                    kind="transform",
                    inputs=["spindle_blank"],
                    params={"x": sign * 18},
                ),
                Feature(
                    id="axle_hole",
                    name="Axle bore",
                    kind="cylinder",
                    params={"radius": 6.2, "height": 26},
                ),
                Feature(
                    id="axle_axis",
                    name="Axle bore axis",
                    kind="transform",
                    inputs=["axle_hole"],
                    params={"x": sign * 18, "ry": 90},
                ),
                Feature(
                    id="bored_spindle",
                    name="Bored spindle",
                    kind="cut",
                    inputs=["spindle", "axle_axis"],
                ),
                Feature(
                    id="fused",
                    name="One-piece knuckle",
                    kind="union",
                    inputs=["arm", "bored_spindle"],
                ),
            ]
        )
        for i, z in enumerate([-7, 7]):
            knuckle.features.extend(
                [
                    Feature(
                        id=f"washer_blank_{i}",
                        name="Washer relief",
                        kind="cylinder",
                        params={"radius": 12.2, "height": 2.4},
                    ),
                    Feature(
                        id=f"washer_relief_{i}",
                        name="Washer seat clearance",
                        kind="transform",
                        inputs=[f"washer_blank_{i}"],
                        params={"z": z},
                    ),
                ]
            )
        knuckle.features.append(
            Feature(
                id="body",
                name="Finished knuckle bores and washer reliefs",
                kind="cut",
                inputs=["fused", "axle_axis", "washer_relief_0", "washer_relief_1"],
            )
        )
        knuckle.ports.append(
            PartPort(
                name="axle", kind="bore", diameter=12.4, origin=(sign * 18, 0, 0), axis=(1, 0, 0)
            )
        )
    wheel_module = Subassembly(
        id="wheel_module",
        name="Reusable wheel and hub",
        children=[
            Occurrence(id="tire", definition="wheel"),
            Occurrence(id="hub", definition="hub"),
            Occurrence(id="axle", definition="axle"),
            *[
                Occurrence(id=f"bearing_{i}", definition="bearing", position=(0, 0, z))
                for i, z in enumerate([-22, 22])
            ],
            *[
                Occurrence(id=f"washer_{i}", definition="washer", position=(0, 0, z))
                for i, z in enumerate([-32, 32])
            ],
        ],
    )
    knuckles = []
    for side, sign in [("left", -1), ("right", 1)]:
        knuckles.append(
            Subassembly(
                id=side + "_module",
                name=side.title() + " steering knuckle",
                children=[
                    Occurrence(id="arm", definition=side + "_arm"),
                    Occurrence(
                        id="wheel",
                        definition="wheel_module",
                        position=(sign * 64, 0, 0),
                        rotation=(0, 90, 0),
                    ),
                    *[
                        Occurrence(
                            id=f"pivot_washer_{i}", definition="pivot_washer", position=(0, 0, z)
                        )
                        for i, z in enumerate([-7, 7])
                    ],
                    Occurrence(id="rod_pin", definition="rod_pin", position=(-sign * 25, -75, 7)),
                ],
            )
        )
    frame = Subassembly(
        id="frame_module",
        name="Front carrier",
        children=[
            *[
                Occurrence(id=f"tube_{i}", definition="crossmember", position=(0, 38, z))
                for i, z in enumerate([-24, 24])
            ],
            *[
                Occurrence(id=f"kingpin_{side}", definition="kingpin", position=(x, 0, 0))
                for side, x in [("left", "negative_half_track"), ("right", "half_track")]
            ],
            *[
                Occurrence(id=f"support_{side}_{i}", definition="support", position=(x, 0, z))
                for side, x in [("left", "negative_half_track"), ("right", "half_track")]
                for i, z in enumerate([-20, 20])
            ],
        ],
    )
    graph = Mechanism(
        components=[
            Component(id="frame", name="Front carrier", feature="frame", grounded=True),
            Component(
                id="left",
                name="Left knuckle",
                feature="left",
                x="negative_half_track",
                angle="steering_angle",
            ),
            Component(id="right", name="Right knuckle", feature="right", x="half_track"),
            Component(id="tie", name="Tie rod", feature="tie", x="tie_start", y=-75, z=12),
        ],
        joints=[
            PlanarJoint(
                id="left_kingpin", kind="pin", a="frame", b="left", pa=("negative_half_track", 0)
            ),
            PlanarJoint(id="right_kingpin", kind="pin", a="frame", b="right", pa=("half_track", 0)),
            PlanarJoint(id="left_tie", kind="pin", a="left", b="tie", pa=(25, -75)),
            PlanarJoint(
                id="right_tie", kind="pin", a="right", b="tie", pa=(-25, -75), pb=("tie_length", 0)
            ),
        ],
        driver=Driver(component="left", angle="steering_angle", start=-25, end=25, samples=21),
    )
    return DesignDocument(
        name="Front steering assembly",
        variables={
            "track_width": setup.track_width,
            "wheel_diameter": setup.wheel_diameter,
            "steering_angle": setup.steering_angle,
        },
        relations=[
            VariableRelation(name="half_track", operation="divide", a="track_width", b=2),
            VariableRelation(
                name="negative_half_track", operation="multiply", a="half_track", b=-1
            ),
            VariableRelation(name="tie_length", operation="add", a="track_width", b=-50),
            VariableRelation(name="tie_start", operation="add", a="negative_half_track", b=25),
            VariableRelation(name="tube_cut_length", operation="add", a="track_width", b=2),
        ],
        assembly=AssemblySpec(
            parts=parts,
            subassemblies=[wheel_module, *knuckles, frame],
            roots=[
                Occurrence(id="frame", definition="frame_module"),
                Occurrence(id="left", definition="left_module"),
                Occurrence(id="right", definition="right_module"),
                Occurrence(id="tie", definition="tie_rod"),
            ],
            mates=[
                *[
                    InterfaceMate(
                        id=side + "_kingpin",
                        a=f"{side}/arm.kingpin",
                        b=f"frame/kingpin_{side}.seat",
                        min_clearance=0.3,
                        max_clearance=0.5,
                    )
                    for side in ("left", "right")
                ],
                *[
                    InterfaceMate(
                        id=side + "_rod",
                        a=f"{side}/arm.rod",
                        b=f"{side}/rod_pin.seat",
                        axial_tolerance=7,
                        min_clearance=0.3,
                        max_clearance=0.5,
                    )
                    for side in ("left", "right")
                ],
            ],
        ),
        mechanism=graph,
    )
