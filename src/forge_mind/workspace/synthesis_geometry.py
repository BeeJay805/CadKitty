"""Deterministic editable BRep realizations; purchased envelopes are explicitly named."""

from __future__ import annotations

import math
from typing import Any

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.synthesis_models import Candidate, LinkageCandidate, SynthesisRequest


def realize(
    request: SynthesisRequest,
    candidate: Candidate,
    opening: float,
    angle: float,
    angle_range: tuple[float, float],
) -> DesignDocument:
    variables: dict[str, float] = {
        "opening": opening,
        "opening_min": request.task.opening_min_mm,
        "opening_max": request.task.opening_max_mm,
        "jaw_width": 16,
        "jaw_depth": 60,
        "jaw_height": 40,
        "pad_thickness": 4,
        "pad_width": 28,
        "pad_height": 16,
        "guide_radius": 3,
        "guide_clearance": 0.2,
        "guide_spacing": 40,
        "base_thickness": 8,
        "input_angle": angle,
    }
    parts: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    def relation(name: str, operation: str, a: Any, b: Any) -> str:
        relations.append({"name": name, "operation": operation, "a": a, "b": b})
        return name

    def feature(
        identifier: str, kind: str, params: dict[str, Any], inputs: list[str] | None = None
    ) -> dict[str, Any]:
        return {
            "id": identifier,
            "name": identifier.replace("_", " "),
            "kind": kind,
            "params": {k: v for k, v in params.items() if not isinstance(v, str)},
            "bindings": {k: v for k, v in params.items() if isinstance(v, str)},
            "inputs": inputs or [],
        }

    def part(identifier: str, name: str, features: list[dict[str, Any]], role: str = "make") -> str:
        parts.append(
            {
                "id": identifier,
                "name": name,
                "features": features,
                "output": features[-1]["id"],
                "production": {"role": role},
                "source": "Synthesis geometry; material/process unknown"
                if role == "make"
                else "Unrated purchase envelope. Supplier, engagement and load capacity unverified",
            }
        )
        return identifier

    def box(identifier: str, name: str, x: Any, y: Any, z: Any, role: str = "make") -> str:
        return part(identifier, name, [feature("body", "box", {"x": x, "y": y, "z": z})], role)

    def occurrence(
        identifier: str, definition: str, position: list[Any], rotation: list[Any] | None = None
    ) -> None:
        roots.append(
            {
                "id": identifier,
                "definition": definition,
                "position": position,
                "rotation": rotation or [0, 0, 0],
            }
        )

    if isinstance(candidate, LinkageCandidate):
        variables.update(
            crank_length=candidate.crank_mm,
            rod_length=candidate.rod_mm,
            jaw_offset=candidate.jaw_offset_mm,
            link_width=8,
            link_thickness=4,
            pin_bore=4,
            base_length=2 * (candidate.crank_mm + candidate.rod_mm) + 60,
            base_width=2 * candidate.crank_mm + 70,
        )
        relation("jaw_arm_length", "add", "jaw_offset", 16)
        relation("negative_offset", "multiply", "jaw_offset", -1)
        relation("jaw_arm_center", "add", "negative_offset", 16)
        relation("right_center", "divide", "jaw_arm_center", 2)
        relation("left_center", "multiply", "right_center", -1)
        relation("left_crank_angle", "multiply", "input_angle", -1)
        relation("mirror_crank_angle", "add", "left_crank_angle", 180)
        box("frame", "Mounting base", "base_length", "base_width", "base_thickness")
        for name, length in (("crank", "crank_length"), ("rod", "rod_length")):
            part(
                name,
                name.title(),
                [
                    feature(
                        "body",
                        "link",
                        {
                            "length": length,
                            "width": "link_width",
                            "thickness": "link_thickness",
                            "bore": "pin_bore",
                        },
                    )
                ],
            )
        for side in ("left", "right"):
            part(
                side + "_jaw",
                side.title() + " jaw bracket",
                [
                    feature("blank", "box", {"x": "jaw_arm_length", "y": 20, "z": 24}),
                    feature("body", "transform", {"x": side + "_center"}, ["blank"]),
                ],
            )
        box("guide", "Jaw guide rail", "base_length", 6, 24)
        components = []

        def component(identifier: str, definition: str, **pose: Any) -> None:
            occurrence(identifier, definition, [0, 0, 0])
            components.append(
                {
                    "id": identifier,
                    "name": identifier.replace("_", " "),
                    "feature": identifier,
                    **pose,
                }
            )

        r, rod_length, theta = candidate.crank_mm, candidate.rod_mm, math.radians(angle)
        q = math.sqrt(rod_length * rod_length - r * r * math.sin(theta) ** 2)
        x = r * math.cos(theta) + q
        beta = math.degrees(math.atan2(-r * math.sin(theta), q))
        component("frame", "frame", grounded=True, z=-12)
        component("right_crank", "crank", angle="input_angle", z=3)
        component("left_crank", "crank", angle="mirror_crank_angle", z=9)
        component(
            "right_rod", "rod", x=r * math.cos(theta), y=r * math.sin(theta), angle=beta, z=15
        )
        component(
            "left_rod", "rod", x=-r * math.cos(theta), y=r * math.sin(theta), angle=180 - beta, z=21
        )
        component("right_jaw", "right_jaw", x=x, z=38)
        component("left_jaw", "left_jaw", x=-x, z=38)
        component("guide_front", "guide", grounded=True, y=-16, z=38)
        component("guide_rear", "guide", grounded=True, y=16, z=38)
        joints = []
        for side in ("left", "right"):
            joints.extend(
                [
                    {"id": side + "_pivot", "kind": "pin", "a": "frame", "b": side + "_crank"},
                    {
                        "id": side + "_crank_pin",
                        "kind": "pin",
                        "a": side + "_crank",
                        "b": side + "_rod",
                        "pa": ["crank_length", 0],
                    },
                    {
                        "id": side + "_jaw_pin",
                        "kind": "pin",
                        "a": side + "_rod",
                        "b": side + "_jaw",
                        "pa": ["rod_length", 0],
                    },
                    {"id": side + "_guide", "kind": "slider", "a": "frame", "b": side + "_jaw"},
                ]
            )
        return DesignDocument.model_validate(
            {
                "name": candidate.name,
                "variables": variables,
                "relations": relations,
                "assembly": {"parts": parts, "roots": roots},
                "mechanism": {
                    "components": components,
                    "joints": joints,
                    "gears": [
                        {
                            "id": "ideal_synchronizer",
                            "a": "right_crank",
                            "b": "left_crank",
                            "ratio": 1,
                            "phase": 180,
                        }
                    ],
                    "driver": {
                        "component": "right_crank",
                        "angle": "input_angle",
                        "start": min(angle_range),
                        "end": max(angle_range),
                        "samples": request.samples,
                    },
                },
            }
        )

    width = (
        request.task.opening_max_mm + 100
        if candidate.family == "screw"
        else 2 * request.task.opening_max_mm + 120
    )
    radius = candidate.module_mm * candidate.pinion_teeth / 2 if candidate.family == "rack" else 0
    variables.update(
        base_length=width,
        base_width=max(90, 2 * radius + 36),
        shaft_length=width - 20,
        guide_length=width - 20,
        shaft_diameter=candidate.shaft_diameter_mm if candidate.family == "screw" else 8,
        rack_length=request.task.opening_max_mm + 54,
    )
    relation("half_opening", "divide", "opening", 2)
    relation("half_jaw", "divide", "jaw_width", 2)
    relation("jaw_offset", "add", "half_jaw", "pad_thickness")
    relation("jaw_x", "add", "half_opening", "jaw_offset")
    relation("negative_jaw_x", "multiply", "jaw_x", -1)
    relation("nut_x", "add", "jaw_x", 12)
    relation("negative_nut_x", "multiply", "nut_x", -1)
    relation("half_pad", "divide", "pad_thickness", 2)
    relation("pad_x", "add", "half_opening", "half_pad")
    relation("negative_pad_x", "multiply", "pad_x", -1)
    relation("guide_hole", "add", "guide_radius", "guide_clearance")
    relation("shaft_hole_radius", "divide", "shaft_diameter", 2)
    relation("shaft_clearance_radius", "add", "shaft_hole_radius", 0.3)
    relation("half_base", "divide", "base_length", 2)
    relation("end_x", "add", "half_base", -5)
    relation("negative_end_x", "multiply", "end_x", -1)
    box("base", "Mounting base", "base_length", "base_width", "base_thickness")
    occurrence("base", "base", [0, 0, 0])
    guide = part(
        "guide",
        "Unrated guide rod envelope",
        [feature("body", "cylinder", {"radius": "guide_radius", "height": "guide_length"})],
        "buy",
    )
    for name, y in (("front", -20), ("rear", 20)):
        occurrence("guide_" + name, guide, [0, y, 18], [0, 90, 0])
    jaw_features = [
        feature("blank", "box", {"x": "jaw_width", "y": "jaw_depth", "z": "jaw_height"})
    ]
    last = "blank"
    for i, y in enumerate((-20, 0, 20)):
        name = "hole_" + str(i)
        jaw_features += [
            feature(
                name,
                "cylinder",
                {"radius": "shaft_clearance_radius" if y == 0 else "guide_hole", "height": 40},
            ),
            feature(name + "_placed", "transform", {"ry": 90, "y": y, "z": -12}, [name]),
            feature(name + "_cut", "cut", {}, [last, name + "_placed"]),
        ]
        last = name + "_cut"
    part("jaw", "Drilled sliding jaw", jaw_features)
    box("pad", "Replaceable jaw pad", "pad_thickness", "pad_width", "pad_height")
    box("end", "End support plate", 8, "base_width", 26)
    moving = {"left": ["left_jaw", "left_pad"], "right": ["right_jaw", "right_pad"]}
    for side, prefix in (("left", "negative_"), ("right", "")):
        occurrence(side + "_jaw", "jaw", [prefix + "jaw_x", 0, 30])
        occurrence(side + "_pad", "pad", [prefix + "pad_x", 0, 42])
        occurrence(side + "_end", "end", [prefix + "end_x", 0, 17])
    spin_occurrences = []
    if candidate.family == "screw":
        variables["screw_lead"] = candidate.lead_mm
        part(
            "screw",
            "Opposed lead screw PURCHASE ENVELOPE, no threads",
            [feature("body", "shaft", {"diameter": "shaft_diameter", "length": "shaft_length"})],
            "buy",
        )
        occurrence("screw", "screw", [0, 0, 18], [0, 90, 0])
        part(
            "nut",
            "Unrated nut PURCHASE ENVELOPE, no thread engagement",
            [
                feature(
                    "body",
                    "bearing",
                    {
                        "diameter": candidate.shaft_diameter_mm + 6,
                        "bore": candidate.shaft_diameter_mm + 0.2,
                        "width": 8,
                    },
                )
            ],
            "buy",
        )
        for side, prefix in (("left", "negative_"), ("right", "")):
            occurrence(side + "_nut", "nut", [prefix + "nut_x", 0, 18], [0, 90, 0])
            moving[side].append(side + "_nut")
        spin_occurrences = ["screw"]
        spin = {"axis": [1, 0, 0], "pivot": [0, 0, 18], "ratio": 180 / candidate.lead_mm}
    else:
        variables.update(
            gear_module=candidate.module_mm,
            pinion_teeth=float(candidate.pinion_teeth),
            pitch_radius=radius,
        )
        part(
            "pinion",
            "Native spur pinion, mate engagement unverified",
            [
                feature(
                    "body",
                    "gear",
                    {
                        "module": "gear_module",
                        "teeth": "pinion_teeth",
                        "thickness": 6,
                        "bore": 8,
                        "auto_clock": 0,
                    },
                )
            ],
        )
        occurrence("pinion", "pinion", [0, 0, 56])
        box(
            "rack",
            "Rack PURCHASE ENVELOPE, tooth engagement unverified",
            "rack_length",
            6,
            6,
            "buy",
        )
        for side, prefix, sign in (("left", "negative_", 1), ("right", "", -1)):
            occurrence(
                side + "_rack",
                "rack",
                [prefix + "jaw_x", sign * (radius + candidate.module_mm + 3), 56],
            )
            moving[side].append(side + "_rack")
        spin_occurrences = ["pinion"]
        spin = {"axis": [0, 0, 1], "pivot": [0, 0, 56], "ratio": 90 / (math.pi * radius)}
    groups = [
        {
            "id": side,
            "name": side.title() + " jaw",
            "driver": "opening",
            "kind": "slide",
            "axis": [1, 0, 0],
            "ratio": sign * 0.5,
            "occurrences": moving[side],
        }
        for side, sign in (("left", -1), ("right", 1))
    ]
    groups.append(
        {
            "id": "input",
            "name": "Ideal prescribed input coupling",
            "driver": "opening",
            "kind": "rotate",
            "occurrences": spin_occurrences,
            **spin,
        }
    )
    return DesignDocument.model_validate(
        {
            "name": candidate.name,
            "variables": variables,
            "relations": relations,
            "assembly": {"parts": parts, "roots": roots},
            "motion_rig": {
                "drivers": [
                    {
                        "id": "opening",
                        "name": "Jaw opening",
                        "unit": "mm",
                        "minimum": request.task.opening_min_mm,
                        "maximum": request.task.opening_max_mm,
                        "reference": "opening",
                    }
                ],
                "groups": groups,
            },
        }
    )
