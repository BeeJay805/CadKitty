"""On-demand CAD construction knowledge, using the existing document and kernel.

Examples are starting points, not a geometry catalog or a supported-shape boundary.
Native execution tests exercise the retained documents, including their bindings.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
from textwrap import dedent
from typing import Any

from forge_mind.workspace.parametric_guidance import PARAMETRIC_DESIGN_GUIDANCE

# Parameter names match kernel.compile_document. This is guidance, not a second parser.
OPERATIONS: dict[str, dict[str, Any]] = {
    "box": {
        "inputs": "none",
        "required": ["x", "y", "z"],
        "meaning": "Positive full dimensions; centered at origin.",
    },
    "cylinder": {
        "inputs": "none",
        "required": ["radius", "height"],
        "meaning": "Positive radius and height; centered at origin, axis Z.",
    },
    "cone": {
        "inputs": "none",
        "required": ["bottom_radius", "top_radius", "height"],
        "meaning": "Cone or conical frustum centered at origin, axis Z. bottom_radius is at -height/2; top_radius is at +height/2. Radii must be nonnegative with at least one positive; height must be positive. Equal radii produce a cylinder. Transform and cut to make countersinks or tapered openings.",
    },
    "torus": {
        "inputs": "none",
        "required": ["major_radius", "minor_radius"],
        "meaning": "Full circular ring centered at origin, axis Z. major_radius is the distance from the axis to the tube center; minor_radius is the tube radius. Require 0 < minor_radius < major_radius. Outside radius is their sum; inside radius is their difference. Transform and cut to make rounded ring grooves.",
    },
    "sphere": {
        "inputs": "none",
        "required": ["radius"],
        "meaning": "Positive radius; centered at origin.",
    },
    "profile": {
        "inputs": "none",
        "required": ["radius_x", "radius_y"],
        "optional": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "meaning": "Ellipse face in local XY; rotate XYZ degrees, then translate.",
    },
    "rounded_profile": {
        "inputs": "none",
        "required": ["width", "height", "radius"],
        "optional": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "meaning": "Rounded rectangle face centered in local XY. width and height are full outside dimensions. All three dimensions must be positive, with radius < min(width, height) / 2 after bindings resolve. Rotate XYZ degrees, then translate. Intermediate section for loft or sweep; it is not a solid output.",
    },
    "curve": {
        "inputs": "none",
        "meaning": "3D interpolating spline. Supply points, at least two distinct XYZ tuples. Each coordinate is a number or a variable/relation name.",
    },
    "loft": {
        "inputs": "two or more profile or rounded_profile sections in section order",
        "optional": {"ruled": 0},
        "meaning": "Solid between sections. ruled=1 uses straight section transitions; 0 uses smooth transitions.",
    },
    "sweep": {
        "inputs": "profile or rounded_profile first, continuous curve second",
        "optional": {"frenet": 0},
        "meaning": "Sweep one face along one wire. Place the section at the path start, normal to its tangent. frenet=1 changes frame transport.",
    },
    "python": {
        "inputs": "optional earlier feature shapes, copied in input order",
        "meaning": "Retained build123d Python source in code. b, params and inputs are supplied. Bind dimensions into params through bindings. Assign final CAD to result. All installed build123d geometry is available; this is not limited to these examples.",
        "cautions": [
            "Never write artifacts yourself; the worker exports result.",
            "Python input shapes arrive by STEP copy; do not reuse guessed topology indices or old selection signatures.",
            "Internal operations remain source-editable; they are not claimed as native feature history.",
            "Let required operations raise errors; a silent fillet fallback can hide missing design intent.",
        ],
    },
    "sketch": {
        "inputs": "none",
        "meaning": "Supply sketch points, line/circle entities and optional constraints. XY/XZ/YZ plane plus offset. Lines must form closed loops. Separate nested loops do not imply holes: cut inner solids explicitly.",
    },
    "extrude": {
        "inputs": "one sketch",
        "required": ["amount"],
        "optional": {"taper": 0},
        "meaning": "Signed extrusion along each face normal; zero cannot produce solid geometry. taper is a draft angle in degrees, strictly between -90 and 90. Positive taper narrows the section away from its source; negative taper widens it. Excessive taper or height can collapse a section and fail the build.",
    },
    "revolve": {
        "inputs": "one sketch",
        "optional": {"angle": 360, "ox": 0, "oy": 0, "oz": 0, "ax": 0, "ay": 1, "az": 0},
        "meaning": "Angle in (0,360] degrees about axis through o with direction a. The sketch profile must not cross its revolution axis.",
    },
    "union": {
        "inputs": "two or more solids",
        "meaning": "Fuse input solids in order. Disconnected results can remain multiple solids; a custom assembly part requires one solid.",
    },
    "cut": {
        "inputs": "target first, one or more tool solids next",
        "meaning": "Subtract each tool. Extend through-hole tools beyond both target faces to avoid blind cuts and coincident ends.",
    },
    "intersect": {
        "inputs": "two or more solids",
        "meaning": "Keep common volume. An empty intersection fails rather than returning fabricated geometry.",
    },
    "fillet": {
        "inputs": "one solid",
        "required": ["radius"],
        "meaning": "Positive radius. selection contains exact edge signatures from this input feature stage, not guessed indices or the final body.",
    },
    "chamfer": {
        "inputs": "one solid",
        "required": ["length"],
        "meaning": "Positive length; exact input-stage edge signatures in selection.",
    },
    "shell": {
        "inputs": "one single-solid feature",
        "required": ["thickness"],
        "meaning": "Positive inward wall thickness; selection names opening face signatures at the input stage.",
    },
    "pattern": {
        "inputs": "one solid",
        "required": ["count"],
        "optional": {"x": 0, "y": 0, "z": 0, "angle": 0},
        "meaning": "Positive integer count. Instance i is rotated i*angle about local Z then translated i*(x,y,z). Returns copies, not an automatic fused part. Use explicit transformed Features and union for a connected part. Patterns can provide repeated cut tools.",
    },
    "mirror": {
        "inputs": "one solid",
        "optional": {"plane": 0},
        "meaning": "plane 0=YZ, 1=XZ, 2=XY. Produces mirrored geometry, not original plus mirror.",
    },
    "transform": {
        "inputs": "one solid feature",
        "optional": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "meaning": "Rotate XYZ Euler degrees about origin, then translate in millimeters.",
    },
    "import": {
        "inputs": "none",
        "meaning": "asset_id identifies an owned STEP asset. Optional params.solid_index is a nonnegative integer within that asset.",
    },
    "hardware": {
        "inputs": "none",
        "optional": {"envelope": 0},
        "meaning": "catalog_id identifies sourced hardware. envelope=1 uses nominal clearance geometry for supported bearings/screws. Inspect dimensions/origin first; threads and physical engagement are not implied.",
    },
    "gear": {
        "inputs": "none",
        "required": ["module", "teeth", "thickness"],
        "optional": {"bore": 0, "pressure_angle": 20, "clock": 0, "auto_clock": 0},
        "meaning": "Positive module/thickness; integer teeth >=17; pressure_angle 20..25 degrees. Optional key_width/key_height/shaft_diameter add a keyway. Native involute construction is a convenience, not the boundary for other tooth forms.",
    },
    "shaft": {
        "inputs": "none",
        "required": ["diameter", "length"],
        "optional": {"bore": 0},
        "meaning": "Centered Z-axis cylinder, optional bore smaller than diameter. Optional key_width/key_height cut a keyway.",
    },
    "bearing": {
        "inputs": "none",
        "required": ["diameter", "width"],
        "optional": {"bore": 0},
        "meaning": "Centered Z-axis ring. Dimensional envelope, not a rated bearing. Bore must be smaller than outside diameter.",
    },
    "link": {
        "inputs": "none",
        "required": ["length", "width", "thickness", "bore"],
        "meaning": "Local pivots (0,0) and (length,0); bore < width < length; Z-centered thickness.",
    },
}


def _python_document(name: str, parameters: dict[str, float], source: str) -> dict[str, Any]:
    return {
        "name": name,
        "variables": parameters,
        "features": [
            {
                "id": "body",
                "name": name,
                "kind": "python",
                "bindings": {key: key for key in parameters},
                "code": dedent(source).strip(),
            }
        ],
        "outputs": ["body"],
    }


RECIPES: dict[str, dict[str, Any]] = {
    "surface_transitions": {
        "summary": "Hollow tangent elbow with fixed straight endpoint collars and an editable bend radius.",
        "techniques": ["tangent line/arc/line spine", "annular sweep", "fixed endpoint interfaces"],
        "checks": [
            "One connected hollow solid; two complete annular end faces",
            "Inlet (0,0,0) along +Z; outlet (reach,0,rise) along +X",
            "Bend-radius edits preserve both straight collars and outer envelope",
            "Nominal CAD passage only; no flow, strength, print or physical-fit qualification",
        ],
        "notes": [
            "The line and quarter-circle tangents agree at each join (G1 continuity). Curvature changes at the joins; this is not G2 continuity.",
            "A smooth loft or rounded edge alone does not guarantee tangent continuity where independent solids meet. Choose a construction that preserves the intended interfaces.",
            "The path starts normal to the annular section. Its fixed endpoints set the interfaces; changing bend radius changes the straight-run lengths instead of moving attachments.",
            "Use the external Python namespace b, params and inputs. Required operations must raise on failure; do not silently return an unrounded substitute.",
        ],
        "document": _python_document(
            "Tangent hollow elbow",
            {
                "reach": 42,
                "rise": 36,
                "outer_radius": 6,
                "wall": 2,
                "bend_radius": 16,
                "collar_length": 8,
            },
            """
            from math import sqrt
            reach, rise, outer, wall, bend, collar = (
                params[k] for k in ('reach', 'rise', 'outer_radius', 'wall',
                                   'bend_radius', 'collar_length'))
            if not (0 < wall < outer < bend and collar > 0
                    and bend + collar < min(reach, rise)):
                raise ValueError('Use 0 < wall < outer_radius < bend_radius, positive collar_length, and bend_radius + collar_length < min(reach, rise)')
            inlet = (0, 0, 0)
            bend_start = (0, 0, rise-bend)
            bend_mid = (bend-bend/sqrt(2), 0, rise-bend+bend/sqrt(2))
            bend_end = (bend, 0, rise)
            outlet = (reach, 0, rise)
            spine = b.Wire([
                b.Edge.make_line(inlet, bend_start),
                b.Edge.make_three_point_arc(bend_start, bend_mid, bend_end),
                b.Edge.make_line(bend_end, outlet),
            ])
            # Put the circular seam opposite the bend direction for STEP booleans.
            section = b.Rot(0, 0, 180) * (b.Circle(outer) - b.Circle(outer-wall))
            result = b.sweep(section, path=spine)
        """,
        ),
    },
    "native_python_detail": {
        "summary": "Add rounded through slots to a native plate with a bound Python detail feature.",
        "techniques": ["native/Python composition", "copied inputs", "slots", "bindings"],
        "checks": [
            "One solid; two through slots",
            "Native base and retained detail source remain editable",
        ],
        "document": {
            "name": "Slotted plate",
            "variables": {
                "width": 60,
                "depth": 30,
                "thickness": 6,
                "slot_length": 16,
                "slot_width": 5,
            },
            "features": [
                {
                    "id": "blank",
                    "name": "Native plate",
                    "kind": "box",
                    "bindings": {"x": "width", "y": "depth", "z": "thickness"},
                },
                {
                    "id": "body",
                    "name": "Rounded slots",
                    "kind": "python",
                    "inputs": ["blank"],
                    "bindings": {
                        "width": "width",
                        "thickness": "thickness",
                        "slot_length": "slot_length",
                        "slot_width": "slot_width",
                    },
                    "code": dedent("""
                     result = inputs[0]
                     t = params['thickness']
                     slot = b.extrude(b.SlotOverall(params['slot_length'], params['slot_width']), amount=t+2)
                     for x in (-params['width']/4, params['width']/4):
                         result = result - b.Pos(x,0,-t/2-1) * slot
                 """).strip(),
                },
            ],
            "outputs": ["body"],
        },
    },
    "rounded_enclosure": {
        "summary": "Open rounded enclosure: cavity, base, corner posts and blind pilot holes.",
        "techniques": ["rounded profiles", "boolean cavities", "bosses", "blind holes"],
        "checks": [
            "One solid; outer width/depth/height",
            "Open top; bottom wall and blind hole floors remain",
        ],
        "document": _python_document(
            "Rounded enclosure",
            {"width": 80, "depth": 60, "height": 28, "wall": 2.5, "corner": 8, "pilot_radius": 1.4},
            """
            w, d, h, t, r = (params[k] for k in ('width', 'depth', 'height', 'wall', 'corner'))
            if not 0 < t < r < min(w, d) / 2 or h <= 2*t:
                raise ValueError('Use 0 < wall < corner < half the smaller side and height > 2*wall')
            result = b.extrude(b.RectangleRounded(w, d, r), amount=h)
            cavity = b.Pos(0, 0, t) * b.extrude(b.RectangleRounded(w-2*t, d-2*t, r-t), amount=h)
            result = result - cavity
            boss_radius = t + params['pilot_radius']
            for x in (-w/2+r, w/2-r):
                for y in (-d/2+r, d/2-r):
                    boss = b.Pos(x, y, h/2) * b.Cylinder(boss_radius, h)
                    pilot = b.Pos(x, y, (h+t)/2) * b.Cylinder(params['pilot_radius'], h-t)
                    result = (result + boss) - pilot
        """,
        ),
    },
    "swept_tube": {
        "summary": "Hollow curved tube with an annular section normal to the initial spline tangent.",
        "techniques": ["3D spline", "section frame", "annular sweep"],
        "checks": ["One solid; two open ends", "Positive wall; curved B-spline faces"],
        "document": _python_document(
            "Curved tube",
            {"radius": 7, "wall": 2, "rise": 50, "reach": 30},
            """
            r, t, h, reach = (params[k] for k in ('radius', 'wall', 'rise', 'reach'))
            if not 0 < t < r or h <= 0 or reach <= 0:
                raise ValueError('Use 0 < wall < radius and positive rise/reach')
            path = b.Spline((0,0,0), (0,0,0.4*h), (reach/3,0,0.8*h), (reach,0,h))
            plane = b.Plane(origin=path@0, z_dir=path%0)
            section = plane * (b.Circle(r) - b.Circle(r-t))
            result = b.sweep(section, path=path)
        """,
        ),
    },
    "twisted_impeller": {
        "summary": "Six twisted loft blades joined to a hub with a through shaft bore.",
        "techniques": ["section loft", "twist", "radial fused array", "through bore"],
        "checks": [
            "One connected solid; six blades",
            "Through bore and genuinely curved blade faces",
        ],
        "document": _python_document(
            "Twisted impeller",
            {"height": 15, "twist": 40, "bore_radius": 3, "hub_radius": 9},
            """
            h, twist, bore, hub = (params[k] for k in ('height', 'twist', 'bore_radius', 'hub_radius'))
            if h <= 0 or not 0 < bore < hub or hub <= 6:
                raise ValueError('Use positive height, 0 < bore_radius < hub_radius and hub_radius > 6')
            sections = [b.Pos(0,0,h*f) * b.Rot(0,0,twist*f) * b.Pos(16,0,0) * b.RectangleRounded(22,3,1)
                        for f in (0, 1/3, 2/3, 1)]
            blade = b.loft(sections)
            result = b.Pos(0,0,h/2) * b.Cylinder(hub,h)
            for angle in range(0,360,60):
                result = result + b.Rot(0,0,angle) * blade
            result = result - b.Pos(0,0,h/2) * b.Cylinder(bore,h+2)
        """,
        ),
    },
    "ribbed_bracket": {
        "summary": "L bracket with two triangular ribs and two base through holes.",
        "techniques": ["plane-oriented polygon", "gussets", "fused components", "through holes"],
        "checks": [
            "One solid; full width/depth/height",
            "Ribs join both walls; mounting holes pass through base",
        ],
        "document": _python_document(
            "Ribbed bracket",
            {"width": 70, "depth": 50, "height": 40, "wall": 4, "hole_radius": 2.5},
            """
            w, d, h, t = (params[k] for k in ('width', 'depth', 'height', 'wall'))
            if t <= 0 or min(w,d,h) <= 4*t:
                raise ValueError('Use positive wall and dimensions greater than 4*wall')
            result = b.Pos(0,0,t/2) * b.Box(w,d,t)
            result = result + b.Pos(0,-d/2+t/2,h/2) * b.Box(w,t,h)
            rib_profile = b.Plane.YZ * b.Polygon((-d/2+t,t), (d/2-t,t), (-d/2+t,h-t), align=None)
            rib = b.extrude(rib_profile, amount=t)
            for x in (-w/4,w/4):
                result = result + b.Pos(x-t/2,0,0) * rib
            for x in (-w/2+2*t,w/2-2*t):
                result = result - b.Pos(x,d/4,t/2) * b.Cylinder(params['hole_radius'],t+2)
        """,
        ),
    },
    "mounting_plate": {
        "summary": "Native feature graph: centered plate with a variable-bound through bore.",
        "techniques": ["native features", "bindings", "boolean cut"],
        "checks": ["One solid; width/depth/thickness", "Through bore radius follows the variable"],
        "document": {
            "name": "Mounting plate",
            "variables": {"width": 60, "depth": 40, "thickness": 6, "bore_radius": 5},
            "relations": [{"name": "tool_height", "operation": "add", "a": "thickness", "b": 2}],
            "features": [
                {
                    "id": "blank",
                    "name": "Plate blank",
                    "kind": "box",
                    "bindings": {"x": "width", "y": "depth", "z": "thickness"},
                },
                {
                    "id": "bore",
                    "name": "Through bore tool",
                    "kind": "cylinder",
                    "bindings": {"radius": "bore_radius", "height": "tool_height"},
                },
                {"id": "body", "name": "Bored plate", "kind": "cut", "inputs": ["blank", "bore"]},
            ],
            "outputs": ["body"],
        },
    },
    "countersunk_plate": {
        "summary": "Native plate with a through bore and an editable included countersink angle.",
        "techniques": ["named degree relations", "cone", "bindings", "boolean cut"],
        "checks": [
            "One solid; complete cylindrical bore and conical recess",
            "Changing included_angle changes recess depth while both radii stay unchanged",
            "Inspect the exported cone angle; source bindings are not measured geometry",
        ],
        "document": {
            "name": "Countersunk plate",
            "variables": {
                "width": 24,
                "thickness": 8,
                "bore_radius": 2,
                "rim_radius": 5,
                "included_angle": 90,
            },
            "relations": [
                {"name": "half_angle", "operation": "divide", "a": "included_angle", "b": 2},
                {"name": "slope", "operation": "tan_deg", "a": "half_angle"},
                {
                    "name": "radial_change",
                    "operation": "subtract",
                    "a": "rim_radius",
                    "b": "bore_radius",
                },
                {"name": "sink_depth", "operation": "divide", "a": "radial_change", "b": "slope"},
                {"name": "half_depth", "operation": "divide", "a": "sink_depth", "b": 2},
                {"name": "top_z", "operation": "divide", "a": "thickness", "b": 2},
                {"name": "sink_z", "operation": "subtract", "a": "top_z", "b": "half_depth"},
                {"name": "tool_height", "operation": "add", "a": "thickness", "b": 2},
            ],
            "features": [
                {
                    "id": "blank",
                    "name": "Plate blank",
                    "kind": "box",
                    "bindings": {"x": "width", "y": "width", "z": "thickness"},
                },
                {
                    "id": "bore",
                    "name": "Through bore",
                    "kind": "cylinder",
                    "bindings": {"radius": "bore_radius", "height": "tool_height"},
                },
                {
                    "id": "sink",
                    "name": "Countersink tool",
                    "kind": "cone",
                    "bindings": {
                        "bottom_radius": "bore_radius",
                        "top_radius": "rim_radius",
                        "height": "sink_depth",
                    },
                },
                {
                    "id": "placed_sink",
                    "name": "Placed countersink",
                    "kind": "transform",
                    "inputs": ["sink"],
                    "bindings": {"z": "sink_z"},
                },
                {
                    "id": "body",
                    "name": "Countersunk plate",
                    "kind": "cut",
                    "inputs": ["blank", "bore", "placed_sink"],
                },
            ],
            "outputs": ["body"],
        },
    },
    "loft_transition": {
        "summary": "Native smooth solid loft through variable-bound elliptical stations.",
        "techniques": ["native sections", "smooth loft", "bindings"],
        "checks": [
            "One solid; height 60 mm",
            "B-spline side faces; station dimensions change on edits",
        ],
        "document": {
            "name": "Loft transition",
            "variables": {"waist": 12},
            "features": [
                {
                    "id": "inlet",
                    "name": "Inlet",
                    "kind": "profile",
                    "params": {"radius_x": 24, "radius_y": 16},
                },
                {
                    "id": "waist",
                    "name": "Waist",
                    "kind": "profile",
                    "params": {"radius_y": 10, "z": 30},
                    "bindings": {"radius_x": "waist"},
                },
                {
                    "id": "outlet",
                    "name": "Outlet",
                    "kind": "profile",
                    "params": {"radius_x": 18, "radius_y": 14, "z": 60},
                },
                {
                    "id": "body",
                    "name": "Smooth transition",
                    "kind": "loft",
                    "inputs": ["inlet", "waist", "outlet"],
                },
            ],
            "outputs": ["body"],
        },
    },
    "rounded_transition": {
        "summary": "Bound rounded inlet lofted to a circular neck with an unchanged outlet interface.",
        "techniques": ["rounded sections", "composed native lofts", "bindings"],
        "checks": [
            "One solid; height 44 mm; inlet outside dimensions 36 by 20 mm",
            "Inlet corner radius 4 mm; circular neck radius 9 mm",
            "Editing inlet_width changes the transition while the circular neck stays unchanged",
            "Solid construction example; no hollow passage or physical fit is established",
        ],
        "document": {
            "name": "Rounded transition",
            "variables": {"inlet_width": 36, "corner_radius": 4, "outlet_radius": 9},
            "features": [
                {
                    "id": "inlet",
                    "name": "Rounded inlet",
                    "kind": "rounded_profile",
                    "params": {"height": 20},
                    "bindings": {"width": "inlet_width", "radius": "corner_radius"},
                },
                {
                    "id": "outlet",
                    "name": "Circular outlet interface",
                    "kind": "profile",
                    "params": {"z": 32},
                    "bindings": {"radius_x": "outlet_radius", "radius_y": "outlet_radius"},
                },
                {
                    "id": "transition",
                    "name": "Rounded to circular transition",
                    "kind": "loft",
                    "inputs": ["inlet", "outlet"],
                    "params": {"ruled": 1},
                },
                {
                    "id": "neck_end",
                    "name": "Circular neck end",
                    "kind": "profile",
                    "params": {"z": 44},
                    "bindings": {"radius_x": "outlet_radius", "radius_y": "outlet_radius"},
                },
                {
                    "id": "neck",
                    "name": "Straight circular neck",
                    "kind": "loft",
                    "inputs": ["outlet", "neck_end"],
                    "params": {"ruled": 1},
                },
                {
                    "id": "body",
                    "name": "Complete solid transition",
                    "kind": "union",
                    "inputs": ["transition", "neck"],
                },
            ],
            "outputs": ["body"],
        },
    },
}


def construction_reference(topic: str = "") -> dict[str, Any]:
    """Return a small index or one complete editable example, never evaluate source."""
    common = {
        "units": "mm",
        "build123d_version": version("build123d"),
        "scope": "Construction syntax and examples. A successful CAD build is not physical qualification or complete request coverage.",
    }
    if topic == "relations":
        from forge_mind.workspace.document import VariableRelation

        return {
            **common,
            "name": "relations",
            "schema": VariableRelation.model_json_schema(),
            "binary": ["add", "subtract", "multiply", "divide", "min", "max", "atan2_deg"],
            "unary": ["sin_deg", "cos_deg", "tan_deg", "sqrt"],
            "rules": "Relations are ordered named arithmetic, never evaluated code. Operands are finite numbers or earlier variable/relation names. Binary operations require b. Unary operations omit b or use literal zero; a named or nonzero b is rejected. A derived name cannot overwrite a variable or earlier relation.",
            "angles": "sin_deg, cos_deg and tan_deg consume degrees. atan2_deg returns degrees with a=y and b=x. tan_deg rejects angles within 1e-10 degrees of odd multiples of 90. atan2_deg rejects the zero vector. sqrt requires a>=0; divide requires b!=0. All results must be finite.",
            "example": "construction:countersunk_plate retains a native cone whose depth follows the included-angle relation. This syntax does not establish physical fit or manufacturing tolerance.",
        }
    if topic == "mechanism":
        from forge_mind.workspace.mechanism_models import Mechanism

        return {
            **common,
            "name": "mechanism",
            "schema": Mechanism.model_json_schema(),
            "output_binding": "For a simple document, Component.feature names a top-level output feature. For an assembly, it names a unique assembly.roots occurrence ID, not a PartDefinition ID or a part-local output feature. Give every output/root exactly one component and give components unique IDs and feature values. Reuse a part definition through distinct roots; its local feature IDs can repeat across definitions/instances.",
            "placement": "With a mechanism, every assembly root position and rotation must resolve to zero. The component pose places that root once. Keep internal child occurrences in their relative frames. Group geometry that moves together, or stationary hardware, under a subassembly root when appropriate; each root still needs one component.",
            "coordinates": "The graph solves local planar millimetres and degrees. frame selects a proper CAD placement: planar_xy_mm_deg (default) maps local x/y/z to world X/Y/Z; planar_yz_mm_deg maps to Y/Z/X; planar_zx_mm_deg maps to Z/X/Y. Local pose order stays [x,y,angle,z]. Grounded x/y/angle set the fixed local pose; moving values seed the solver. Component z is fixed depth along the selected plane normal. Joint pa/pb remain local XY component points; slider axis is a local planar angle. Component/root geometry and internal child placements are local before frame placement. Changing frame moves that local design; convert existing world-authored geometry and poses to preserve world intent. World envelope axes and source requirements do not rotate. No out-of-plane motion, physical joint construction or spatial dynamics is solved.",
            "proof_limits": "Mechanism joints and gears describe ideal coupled rigid-body constraints. A solved graph or sampled motion report does not prove native joint correspondence, full continuous travel, retention, contact mechanics or manufacture. motion_rig prescribes independent rigid motion; it does not prove a coupled mechanism. Use measured geometry and engineering reports for their stated checks and retain unverified outcomes.",
            "mapping_example": {
                "scope": "Structural ID and frame mapping only; this is not a complete CAD document or physical joint design.",
                "roots": [
                    {
                        "id": "fixed_root",
                        "definition": "shared_part",
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0],
                    },
                    {
                        "id": "moving_root",
                        "definition": "shared_part",
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0],
                    },
                ],
                "mechanism": {
                    "components": [
                        {"id": "fixed", "name": "Fixed", "feature": "fixed_root", "grounded": True},
                        {"id": "moving", "name": "Moving", "feature": "moving_root"},
                    ],
                    "joints": [
                        {
                            "id": "pivot",
                            "kind": "pin",
                            "a": "fixed",
                            "b": "moving",
                            "pa": [0, 0],
                            "pb": [0, 0],
                        }
                    ],
                    "driver": {
                        "component": "moving",
                        "angle": 0,
                        "start": 0,
                        "end": 360,
                        "samples": 37,
                    },
                },
            },
        }
    if topic in RECIPES:
        return {**common, "name": topic, **deepcopy(RECIPES[topic])}
    return {
        **common,
        "name": "construction",
        "parametric_design": PARAMETRIC_DESIGN_GUIDANCE,
        "representations": {
            "mechanism": "For coupled planar assemblies, query construction:mechanism for the typed graph, root ownership, placement and local-frame rules. Prescribed motion_rig is a different intent.",
            "native": "Named features and parameter bindings provide explicit editable history. Query feature:<kind> for parameters and input order.",
            "python": "For custom profiles, surfaces, threads or other advanced forms, use a named python feature with bound params and retained source. Native features can consume its solid output.",
            "assembly": "Reuse PartDefinition objects with one connected output solid per custom part, then place Occurrence objects. Query assembly_edit or part for schema. Do not replace the whole document to change one part.",
        },
        "operations": list(OPERATIONS),
        "named_relations": {
            "query": "construction:relations",
            "example": "construction:countersunk_plate",
        },
        "recipes": [
            {
                "id": key,
                "summary": value["summary"],
                "techniques": value["techniques"],
                "query": f"construction:{key}",
            }
            for key, value in RECIPES.items()
        ],
        "usage": "These examples teach composable techniques; adapt them or write arbitrary build123d. Fetch one construction:<id> for its complete document.",
        **({"unknown_topic": topic} if topic else {}),
    }


def feature_reference(kind: str) -> dict[str, Any]:
    if kind not in OPERATIONS:
        return {
            "unknown_kind": kind,
            "available": list(OPERATIONS),
            "usage": "Query feature:<kind>",
        }
    return {
        "name": kind,
        **deepcopy(OPERATIONS[kind]),
        "parameter_rules": "params holds literal numeric values. bindings maps parameter keys to document variable/relation names and overrides params. Bindings are names, not expressions. Earlier feature IDs supply inputs in order.",
        "scope": "Kernel syntax guidance; material, strength, function and manufacture require their own evidence.",
    }
