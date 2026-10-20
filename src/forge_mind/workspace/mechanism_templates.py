"""Editable reference mechanisms. Dimensions are proposals, never load ratings."""

from __future__ import annotations

import math

from forge_mind.workspace.document import DesignDocument, Feature, VariableRelation
from forge_mind.workspace.mechanism_models import (
    Component,
    Driver,
    GearCoupling,
    Mechanism,
    MechanismTemplate,
    PlanarJoint,
)


def template_document(request: MechanismTemplate) -> DesignDocument:
    features: list[Feature] = []
    components: list[Component] = []
    joints: list[PlanarJoint] = []
    gears: list[GearCoupling] = []
    relations: list[VariableRelation] = []

    def feature(
        fid: str,
        kind: str,
        params: dict[str, float],
        inputs: list[str] | None = None,
        bindings: dict[str, str] | None = None,
    ) -> str:
        features.append(
            Feature.model_validate(
                {
                    "id": fid,
                    "name": fid.replace("_", " ").title(),
                    "kind": kind,
                    "params": params,
                    "inputs": inputs or [],
                    "bindings": bindings or {},
                }
            )
        )
        return fid

    def component(cid: str, fid: str, **kwargs: object) -> None:
        components.append(
            Component.model_validate(
                {"id": cid, "feature": fid, "name": cid.replace("_", " ").title(), **kwargs}
            )
        )

    def pin(
        jid: str,
        a: str,
        other: str,
        pa: tuple[float, float] = (0, 0),
        pb: tuple[float, float] = (0, 0),
    ) -> None:
        joints.append(PlanarJoint(id=jid, kind="pin", a=a, b=other, pa=pa, pb=pb))

    t, shaft, clearance = request.thickness, request.shaft_diameter, request.clearance
    variables = {"input_angle": 15.0}
    if request.kind == "gearbox":
        n, wheel, module = request.pinion_teeth, request.wheel_teeth, request.module
        if shaft + clearance >= module * (n - 2.5):
            raise ValueError(
                "Shaft bore consumes the pinion root; reduce the shaft or increase module"
            )
        distance = module * (n + wheel) / 2
        ratio = wheel / n
        radius = module * max(n, wheel) / 2 + module
        gap = t + 4
        variables.update(
            module=module, pinion_teeth=float(n), wheel_teeth=float(wheel), face_width=t
        )
        for name, operation, a, b in [
            ("tooth_sum", "add", "pinion_teeth", "wheel_teeth"),
            ("pitch_sum", "multiply", "module", "tooth_sum"),
            ("center_distance", "divide", "pitch_sum", 2.0),
            ("output_x", "multiply", "center_distance", 2.0),
            ("stage_ratio", "divide", "wheel_teeth", "pinion_teeth"),
            ("stage_gap", "add", "face_width", 4.0),
            ("largest_teeth", "max", "pinion_teeth", "wheel_teeth"),
            ("wheel_diameter", "multiply", "module", "largest_teeth"),
            ("addenda", "multiply", "module", 2.0),
            ("outside_diameter", "add", "wheel_diameter", "addenda"),
            ("housing_height", "add", "outside_diameter", 14.0),
            ("housing_width", "add", "output_x", "housing_height"),
            ("half_face", "divide", "face_width", 2.0),
            ("back_offset", "multiply", "half_face", -1.0),
            ("rear_z", "add", "back_offset", -9.0),
            ("front_offset", "add", "stage_gap", "half_face"),
            ("front_z", "add", "front_offset", 9.0),
            ("shaft_z", "divide", "stage_gap", 2.0),
            ("double_face", "multiply", "face_width", 2.0),
            ("shaft_length", "add", "double_face", 42.0),
        ]:
            relations.append(
                VariableRelation.model_validate(
                    {"name": name, "operation": operation, "a": a, "b": b}
                )
            )
        # Removable bored plates; dimensional bearing envelopes make no catalog/rating claim.
        for side, z in (("rear", -t / 2 - 9), ("front", gap + t / 2 + 9)):
            base = feature(
                f"{side}_blank",
                "box",
                {"x": 2 * distance + 2 * radius + 14, "y": 2 * radius + 14, "z": 7},
            )
            base = feature(f"{side}_position", "transform", {"x": distance}, [base])
            for i in range(3):
                bore = feature(
                    f"{side}_bore_{i}",
                    "cylinder",
                    {"radius": shaft / 2 + 4 + clearance / 2, "height": 10},
                )
                bore = feature(
                    f"{side}_bore_position_{i}", "transform", {"x": i * distance}, [bore]
                )
                base = feature(f"{side}_cut_{i}", "cut", {}, [base, bore])
                bearing = feature(
                    f"{side}_bearing_{i}",
                    "bearing",
                    {"diameter": shaft + 8, "bore": shaft + clearance, "width": 7},
                )
                component(
                    f"{side}_bearing_{i}",
                    bearing,
                    grounded=True,
                    x=i * distance,
                    z=z,
                    group="Bearing envelopes",
                )
            component(f"{side}_plate", base, grounded=True, z=z, group="Housing")
        for i, (x, z, _teeth) in enumerate(
            ((0, 0, n), (distance, 0, wheel), (distance, gap, n), (2 * distance, gap, wheel))
        ):
            cid = f"gear_{i + 1}"
            fid = feature(
                cid,
                "gear",
                {
                    "bore": shaft + clearance,
                    "auto_clock": 1.0 if i % 2 else 0.0,
                    "shaft_diameter": shaft,
                    "key_width": shaft / 3 + clearance,
                    "key_height": shaft / 3 + clearance,
                },
                bindings={
                    "module": "module",
                    "teeth": "wheel_teeth" if i % 2 else "pinion_teeth",
                    "thickness": "face_width",
                },
            )
            component(
                cid,
                fid,
                x=x,
                z=z,
                angle=(15 if i == 0 else -15 / ratio if i in (1, 2) else 15 / ratio**2),
                group="Transmission",
            )
            pin(f"pivot_{cid}", "rear_plate", cid, (x, 0))
            key = feature(f"key_{i + 1}", "box", {"x": shaft / 3, "y": shaft / 3, "z": t})
            component(key, key, x=x, y=shaft / 2, z=z, group="Keys")
            joints.append(
                PlanarJoint(id=f"key_fixed_{i}", kind="fixed", a=cid, b=key, pa=(0, shaft / 2))
            )
        joints.append(
            PlanarJoint(id="intermediate_shaft_coupling", kind="fixed", a="gear_2", b="gear_3")
        )
        gears.extend(
            [
                GearCoupling(id="stage_1", a="gear_1", b="gear_2", ratio=ratio),
                GearCoupling(id="stage_2", a="gear_3", b="gear_4", ratio=ratio),
            ]
        )
        for i, gear in enumerate(("gear_1", "gear_2", "gear_4")):
            fid = feature(
                f"shaft_{i + 1}",
                "shaft",
                {
                    "diameter": shaft,
                    "length": gap + t + 38,
                    "key_width": shaft / 3 + clearance,
                    "key_height": shaft / 3 + clearance,
                },
            )
            component(fid, fid, x=i * distance, z=gap / 2, group="Shafts")
            joints.append(PlanarJoint(id=f"shaft_fixed_{i}", kind="fixed", a=gear, b=fid))
        for c in components:
            if c.id.startswith("rear_"):
                c.z = "rear_z"
            elif c.id.startswith("front_"):
                c.z = "front_z"
            elif c.id.startswith("shaft_"):
                c.z = "shaft_z"
            if c.x == distance:
                c.x = "center_distance"
            elif c.x == 2 * distance:
                c.x = "output_x"
            if c.z == gap:
                c.z = "stage_gap"
        for j in joints:
            if j.id.startswith("pivot_"):
                if j.pa[0] == distance:
                    j.pa = ("center_distance", 0.0)
                elif j.pa[0] == 2 * distance:
                    j.pa = ("output_x", 0.0)
        for g in gears:
            g.ratio = "stage_ratio"
        for f in features:
            if f.id.endswith("_blank"):
                f.bindings.update(x="housing_width", y="housing_height")
            if f.kind == "shaft":
                f.bindings["length"] = "shaft_length"
            if f.id.startswith("key_"):
                f.bindings["z"] = "face_width"
            if f.kind == "transform" and f.params.get("x") == distance:
                f.bindings["x"] = "center_distance"
            elif f.kind == "transform" and f.params.get("x") == 2 * distance:
                f.bindings["x"] = "output_x"
        driver = Driver(component="gear_1", angle="input_angle", start=0, end=360)
        name = f"Two-stage gearbox {ratio**2:g}:1"
    else:
        crank, rod, ground, rocker = request.crank, request.rod, request.ground, request.rocker
        if request.kind == "slider_crank" and rod <= crank + 2:
            raise ValueError("The reference slider-crank requires rod > crank + 2 mm")
        width = max(shaft + clearance + 4, 8)
        if width >= min(crank, rod, rocker):
            raise ValueError("Link pivot distance must exceed the bore and web width")
        variables.update(crank_length=crank, rod_length=rod, rocker_length=rocker)
        base = feature(
            "frame",
            "box",
            {"x": max(ground, crank + rod) * 2 + 20, "y": max(rocker, crank) * 2 + 20, "z": 5},
        )
        component("frame", base, grounded=True, z=-12, group="Frame")
        for cid, _length, z in (("crank", crank, 0), ("rod", rod, t + 1)):
            fid = feature(
                cid,
                "link",
                {"width": width, "thickness": t, "bore": shaft + clearance},
                bindings={"length": "crank_length" if cid == "crank" else "rod_length"},
            )
            component(
                cid,
                fid,
                x=0 if cid == "crank" else crank * math.cos(math.radians(15)),
                y=0 if cid == "crank" else crank * math.sin(math.radians(15)),
                angle=15 if cid == "crank" else -5,
                z=z,
                group="Linkage",
            )
        pin("crank_pivot", "frame", "crank")
        joints.append(
            PlanarJoint(id="crank_pin", kind="pin", a="crank", b="rod", pa=("crank_length", 0))
        )
        if request.kind == "slider_crank":
            fid = feature("slider", "box", {"x": 18, "y": 16, "z": t})
            hole = feature(
                "slider_bore", "cylinder", {"radius": (shaft + clearance) / 2, "height": t + 2}
            )
            fid = feature("slider_drilled", "cut", {}, [fid, hole])
            component("slider", fid, x=crank + rod, z=0, group="Linkage")
            joints.append(PlanarJoint(id="slider_guide", kind="slider", a="frame", b="slider"))
            joints.append(
                PlanarJoint(
                    id="rod_slider_pin", kind="pin", a="rod", b="slider", pa=("rod_length", 0)
                )
            )
            for sign in (-1, 1):
                fid = feature(
                    "guide_" + ("lower" if sign < 0 else "upper"),
                    "box",
                    {"x": 2 * crank + 32, "y": 5, "z": t + 3},
                )
                component(
                    fid, fid, grounded=True, x=rod, y=sign * (10.5 + clearance), group="Frame"
                )
        else:
            fid = feature(
                "rocker",
                "link",
                {"width": width, "thickness": t, "bore": shaft + clearance},
                bindings={"length": "rocker_length"},
            )
            component("rocker", fid, x=ground, angle=80, z=0, group="Linkage")
            pin("rocker_pivot", "frame", "rocker", (ground, 0))
            joints.append(
                PlanarJoint(
                    id="coupler_pin",
                    kind="pin",
                    a="rod",
                    b="rocker",
                    pa=("rod_length", 0),
                    pb=("rocker_length", 0),
                )
            )
        driver = Driver(component="crank", angle="input_angle", start=15, end=375)
        name = "Slider-crank demonstrator" if request.kind == "slider_crank" else "Four-bar linkage"
    return DesignDocument(
        name=name,
        variables=variables,
        relations=relations,
        features=features,
        outputs=[c.feature for c in components],
        mechanism=Mechanism(components=components, joints=joints, gears=gears, driver=driver),
    )
