"""Reproducible engineering revision of the saved AI idler, not a generation template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge_mind.workspace.document import (
    DesignDocument,
    Feature,
    Occurrence,
    PartDefinition,
)


def revised(source: DesignDocument) -> DesignDocument:
    doc = source.model_copy(deep=True)
    assert doc.assembly
    parts = {p.id: p for p in doc.assembly.parts}
    roots = {r.id: r for r in doc.assembly.roots}
    moving = next(s for s in doc.assembly.subassemblies if s.id == "moving_unit")
    children = {r.id: r for r in moving.children}

    def feature(part: str, name: str) -> Feature:
        return next(f for f in parts[part].features if f.id == name)

    def cut(part: str, name: str, radius: float, height: float, **position: float) -> None:
        definition = parts[part]
        tool = Feature(
            id=name + "_tool",
            name=name,
            kind="cylinder",
            params={"radius": radius, "height": height},
        )
        placed = Feature(
            id=name + "_position", name=name, kind="transform", inputs=[tool.id], params=position
        )
        finish = Feature(
            id=name + "_cut", name=name, kind="cut", inputs=[definition.output, placed.id]
        )
        definition.features.extend([tool, placed, finish])
        definition.output = finish.id

    def hardware(part_id: str, catalog: str, **placement: float) -> PartDefinition:
        definition = PartDefinition(
            id=part_id,
            name=part_id.replace("_", " ").title(),
            source="Purchased "
            + catalog
            + "; catalog boundary geometry, simplified threads. Specify grade and locking process in manufacturing schedule.",
            features=[
                Feature(
                    id="catalog",
                    name=catalog,
                    kind="hardware",
                    catalog_id=catalog,
                    params={"envelope": 1} if catalog.startswith(("button_", "shcs_")) else {},
                )
            ],
            output="catalog",
        )
        if placement:
            definition.features.append(
                Feature(
                    id="position",
                    name="Catalog local frame",
                    kind="transform",
                    inputs=["catalog"],
                    params=placement,
                )
            )
            definition.output = "position"
        return definition

    # Removable supports: four M4x25 screws, 20 mm grip and 5 mm nominal engagement.
    for x in [-34, 34]:
        for y in [-12, 12]:
            name = f"support_{'p' if x > 0 else 'n'}{'p' if y > 0 else 'n'}"
            cut("base", name, 2, 6, x=x, y=y, z=1)
            doc.assembly.roots.append(
                Occurrence(id=name + "_bolt", definition="support_bolt", position=(x, y, 28))
            )
    for y in [-12, 12]:
        cut("guide_support", "bolt_" + ("p" if y > 0 else "n"), 2.25, 22, y=y)
    doc.assembly.parts.append(hardware("support_bolt", "shcs_m4x25"))
    parts[
        "base"
    ].source = "Machined S355 steel plate, 120x80x8. Preserve four machine mounting bores. Four added M4x0.7-6H blind support threads at X=+-34/Y=+-12: 5 mm nominal axial overlap, 6 mm modeled depth; physical tap drill depth and runout must follow drawing. Smooth major-diameter thread representation."
    parts[
        "guide_support"
    ].source = "Machined S355 steel, removable using two M4x25 ISO4762 screws per support; 20 mm grip, 5 mm engagement into base. No welds. Blind guide sockets retain shafts. Park carriage at zero to reach both screw heads. Install left support last and remove it first for guide service."

    feature("guide_support", "guide_socket_tool").params.update(radius=4.025, height=4.5)
    for name in ["socket_positive", "socket_negative"]:
        feature("guide_support", name).params["x"] = -1.75
    for port in parts["guide_support"].ports:
        if port.name.startswith("guide_"):
            port.diameter = 8.05
    for mate in doc.assembly.mates:
        if mate.id.startswith("guide_"):
            mate.max_clearance = 0.06
        if mate.id in {"adjuster_left_support", "adjuster_right_support"}:
            mate.axial_tolerance = 0.35
        if mate.id == "adjuster_carriage_thread":
            mate.axial_tolerance = 15.35

    # Every radial guide ligament is now at least 3.4 mm, with 1.5 mm base clearance.
    feature("carriage", "slider_blank").params.update(y=64, z=16)
    cut("carriage", "roller_saddle", 20.5, 32, rx=-90, z=45)
    feature("carriage", "collar_tool").params["height"] = 4.2
    feature("carriage", "collar_relief_positive").params["x"] = 13
    feature("carriage", "collar_relief_negative").params["x"] = -13
    parts[
        "carriage"
    ].source = "Machined S355 steel carriage: 30x64x16 lower body, Z=17.5; guide centers Y=+-24/Z18, diameter8.2. Minimum nominal guide ligament3.4 mm. M8x1.25-6H integral adjustment thread, smooth major-diameter representation. Pre-thread carriage onto screw before installing final support; removable supports permit service."

    # NSK open 608 published abutments: shaft da>=10, housing Da<=20, fillet<=0.3.
    feature("roller", "relief_tool").params["radius"] = 10
    feature("roller_retainer", "plate_clearance").params["radius"] = 10
    feature("roller_retainer", "plate_blank").params["radius"] = 19.5
    feature("inner_sleeve", "sleeve_outer").params["radius"] = 5
    feature("axle_spacer", "spacer_outer").params["radius"] = 5
    bearing_source = "https://www.nsk.com/in-en/engineering/608-esm-md.html"
    parts["bearing608"].source = (
        "Selected purchased NSK608 OPEN bearing, 8x22x7; manufacturer da_min10, Da_max20, ra_max0.3, C3650N/C0=1370N. "
        + bearing_source
        + ". Catalog ISO608 envelope is dimensional only; no seals or internal raceway geometry."
    )
    parts[
        "roller"
    ].source = "Machined S355 roller, diameter40x30 face. Selected NSK608 OPEN bearings; diameter20 inner shoulder opening gives housing abutment Da20, matching manufacturer's maximum. Bearing bores22, shoulders at Y=+-8. Four M3x0.5-6H retainer holes; nominal smooth thread representation. Lubricate and protect open bearings from contamination."
    parts[
        "roller_retainer"
    ].source = "Machined steel retainer diameter39x1, opening20; outer-ring contact annulus radii10..11 per NSK608 Da<=20. Two ISO7380-1 M3x6 screws. Install retainers and screws before putting roller into fork. Tool approach modeled in that preassembly state."
    parts[
        "inner_sleeve"
    ].source = "Ground steel inner-ring sleeve OD10/ID8.1/L16. Diameter10 shoulder matches NSK608 minimum shaft abutment. Match final sleeve length to measured bearing/roller stack; avoid unintended axial preload."
    parts[
        "axle_spacer"
    ].source = "Ground steel spacer OD10/ID8.1/L3, reused twice; diameter10 matches NSK608 minimum shaft abutment, clear of diameter20 rotating retainer. Match final axial stack before torqueing axle."
    replacement = hardware("retainer_screw", "button_m3x6")
    replacement.ports = parts["retainer_screw"].ports
    doc.assembly.parts[doc.assembly.parts.index(parts["retainer_screw"])] = replacement

    # Real catalog M8 nut dimensions replace the invented five-mm square nut.
    nut = hardware("axle_nut", "hexnut_m8", rx=-90, y=-3.4)
    nut.ports = parts["axle_nut"].ports
    doc.assembly.parts[doc.assembly.parts.index(parts["axle_nut"])] = nut
    children["axle_retention"].position = (0.0, 27.4, 45.0)
    feature("idler_axle", "axle_shank").params["height"] = 57
    feature("idler_axle", "axle_position").params["y"] = 4.5
    for port in parts["idler_axle"].ports:
        if port.name == "nut_thread":
            port.origin = (0.0, 27.4, 0.0)
    parts[
        "idler_axle"
    ].source = "Machined S355 headed axle diameter8, shank Y=-24..33. ISO4032 M8 nut at Y24..30.8, with2.2 mm thread projection. Smooth M8x1.25 thread representation. Counterhold integral12 mm head; use13 mm socket on nut. Match inner-ring sleeve stack to prevent bearing preload."

    doc.variables["screw_endplay"] = 0.25
    doc.variables["screw_float"] = 0.0
    from forge_mind.workspace.document import VariableRelation

    doc.relations.extend(
        [
            VariableRelation(name="retention_x", operation="add", a=41.4, b="screw_endplay"),
            VariableRelation(name="handwheel_x", operation="add", a=44.8, b="screw_endplay"),
            VariableRelation(name="hub_port_x", operation="add", a="handwheel_x", b=10.5),
            VariableRelation(name="negative_wheel_x", operation="multiply", a="handwheel_x", b=-1),
            VariableRelation(name="crosspin_local_x", operation="add", a="negative_wheel_x", b=56),
            VariableRelation(
                name="retention_world_x", operation="add", a="retention_x", b="screw_float"
            ),
            VariableRelation(
                name="handwheel_world_x", operation="add", a="handwheel_x", b="screw_float"
            ),
            VariableRelation(name="crosspin_world_x", operation="add", a=56, b="screw_float"),
        ]
    )
    roots["adjuster"].position = ("screw_float", 0.0, 18.0)
    roots["adjuster_retention"].position = ("retention_world_x", 0.0, 18.0)
    roots["handwheel"].position = ("handwheel_world_x", 0.0, 18.0)
    for port in parts["adjustment_screw"].ports:
        if port.name == "retention_thread":
            port.origin = ("retention_x", 0.0, 0.0)
        if port.name == "handwheel_thread":
            port.origin = ("hub_port_x", 0.0, 0.0)

    # A removable transverse fastener transmits torque positively through the hub.
    cut("adjustment_screw", "crosspin", 1.6, 16, rx=-90, x=56)
    cut("adjustment_handwheel", "crosspin", 1.6, 16, rx=-90)
    feature("adjustment_handwheel", "crosspin_position").bindings["x"] = "crosspin_local_x"
    # Local milled flats support the whole M3 head/nut, instead of tangent seating
    # against the round hub. Keep them away from the wheel and nut bearing end.
    wheel = parts["adjustment_handwheel"]
    for side in [-1, 1]:
        name = "crosspin_flat_" + ("p" if side > 0 else "n")
        wheel.features.extend(
            [
                Feature(id=name, name=name, kind="box", params={"x": 8, "y": 10, "z": 20}),
                Feature(
                    id=name + "_at",
                    name=name,
                    kind="transform",
                    inputs=[name],
                    params={"y": side * 10.0},
                    bindings={"x": "crosspin_local_x"},
                ),
                Feature(
                    id=name + "_cut", name=name, kind="cut", inputs=[wheel.output, name + "_at"]
                ),
            ]
        )
        wheel.output = name + "_cut"
    doc.assembly.parts.extend(
        [hardware("hub_crossbolt", "button_m3x16"), hardware("hub_crossnut", "hexnut_m3")]
    )
    # Trim a sourced 16 mm screw to a drawing-controlled 14 mm finished length.
    # The untrimmed tip hits the base during rotation. BOM marks modified hardware.
    crossbolt = next(p for p in doc.assembly.parts if p.id == "hub_crossbolt")
    crossbolt.features.extend(
        [
            Feature(
                id="tip_tool", name="Trim to 14 mm", kind="box", params={"x": 10, "y": 10, "z": 4}
            ),
            Feature(
                id="tip_at",
                name="Tip cutter",
                kind="transform",
                inputs=["tip_tool"],
                params={"z": -16},
            ),
            Feature(id="trimmed", name="Finished M3x14", kind="cut", inputs=["catalog", "tip_at"]),
        ]
    )
    crossbolt.output = "trimmed"
    crossbolt.source = "ISO7380-1 M3x16 blank, cut and deburr to 14 mm finished under-head length. Nominal thread crest envelope; modified hardware requires drawing-controlled inspection."
    doc.assembly.roots.extend(
        [
            Occurrence(
                id="hub_crossbolt",
                definition="hub_crossbolt",
                position=("crosspin_world_x", -5.0, 18.0),
                rotation=(90, 0, 0),
            ),
            Occurrence(
                id="hub_crossnut",
                definition="hub_crossnut",
                position=("crosspin_world_x", 5.0, 18.0),
                rotation=(-90, 0, 0),
            ),
        ]
    )
    parts[
        "adjustment_screw"
    ].source = "Machined S355 M8x1.25 lead screw, smooth nominal thread. Shoulder faceX30; supportX30..38; ISO4032 nut inner faceX38+screw_endplay. Endplay0.25 nominal, drawing acceptance0.15..0.35. Handwheel uses transverse trimmed M3x14 bolt and M3 nut at X56 for positive torque, not jam friction alone. Pre-thread carriage before fitting left support."
    parts[
        "adjustment_handwheel"
    ].source = "Machined steel handwheel diameter28 and extended hub; threaded on screw, jammed to ISO4032 retention nut, then cross-drilled/bolted at X56 using ISO7380-1 M3x16 trimmed to14 mm and ISO4032 M3 nut. Pin establishes positive bidirectional drive. Set and measure0.25 mm screw endplay before drilling. Use specified removable thread-locking process; torque and physical endplay inspection required."
    doc.name = "Serviceable belt-idler engineering revision"
    return DesignDocument.model_validate(doc.model_dump(warnings="error"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.source.read_text(encoding="utf-8"))
    result = revised(DesignDocument.model_validate(raw.get("document", raw)))
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
