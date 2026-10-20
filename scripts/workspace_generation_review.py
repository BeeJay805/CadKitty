"""Prospective development checks on reimported STEP, separate from engine verdicts.

The checks intentionally expose their sampling, numeric allowances and missing
physical scope. They never evaluate model Python or accept a workspace revision.
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
from pathlib import Path

import build123d as b
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder


def close(a, expected, allowance=1e-5):
    return abs(a - expected) <= allowance


def boolean_solid_volume(result):
    """Measure known boolean return forms; empty/contact geometry has no solid volume."""
    if result is None:
        return 0.0
    if isinstance(result, b.ShapeList):
        return sum(boolean_solid_volume(item) for item in result)
    if not isinstance(result, b.Shape):
        raise TypeError(f"Unsupported boolean result type: {type(result).__name__}")
    volume = result.volume if result.solids() else 0.0
    if not math.isfinite(volume) or volume < 0:
        raise ValueError("Boolean solid volume must be finite and nonnegative")
    return volume


def cylinders(shape):
    rows = []
    for face in shape.faces():
        surface = BRepAdaptor_Surface(face.wrapped, True)
        if surface.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = surface.Cylinder()
        axis, point = cylinder.Axis().Direction(), cylinder.Location()
        bounds = face.bounding_box()
        if close(abs(axis.Z()), 1):
            rows.append(
                {
                    "radius": cylinder.Radius(),
                    "x": point.X(),
                    "y": point.Y(),
                    "z_min": bounds.min.Z,
                    "z_max": bounds.max.Z,
                }
            )
    return rows


def cylinder_pattern(shape, radius, centers, z_min, z_max):
    matched = [
        row
        for row in cylinders(shape)
        if close(row["radius"], radius)
        and close(row["z_min"], z_min)
        and close(row["z_max"], z_max)
    ]
    return len(matched) == len(centers) and {
        (round(row["x"], 5), round(row["y"], 5)) for row in matched
    } == set(centers)


def angular_intervals(solid, radius, z, resolution=0.25):
    count = round(360 / resolution)
    occupancy = [
        solid.is_inside(
            (
                radius * math.cos(math.radians(i * resolution)),
                radius * math.sin(math.radians(i * resolution)),
                z,
            )
        )
        for i in range(count)
    ]
    starts = [i for i in range(count) if occupancy[i] and not occupancy[(i - 1) % count]]
    intervals = []
    for start in starts:
        width = 1
        while width < count and occupancy[(start + width) % count]:
            width += 1
        intervals.append(
            {
                "center_deg": ((start + (width - 1) / 2) * resolution) % 360,
                "arc_width_mm": width * resolution * math.pi / 180 * radius,
            }
        )
    centers = sorted(row["center_deg"] for row in intervals)
    spacing = [(centers[(i + 1) % len(centers)] - value) % 360 for i, value in enumerate(centers)]
    phase = (
        math.degrees(cmath.phase(sum(cmath.exp(8j * math.radians(value)) for value in centers)))
        / 8
        % 45
        if centers
        else None
    )
    return {
        "z": z,
        "radius": radius,
        "resolution_deg": resolution,
        "intervals": intervals,
        "spacing_deg": spacing,
        "eightfold_phase_deg": phase,
        "occupied_fraction": sum(occupancy) / count,
    }


def check(case_id, step, document=None):
    shape = b.import_step(str(step))
    solids = list(shape.solids())
    bounds = shape.bounding_box()
    checks = {"valid_positive_volume_export": bool(shape.is_valid and shape.volume > 0)}
    observations = {
        "bounds": {"minimum": list(bounds.min), "maximum": list(bounds.max)},
        "solid_count": len(solids),
        "volume_mm3": shape.volume,
    }
    if case_id == "development_twisted_impeller":
        checks["one_connected_solid"] = len(solids) == 1
        checks["90x90x24_bounds_Z0_datum"] = all(
            close(v, t)
            for v, t in zip(
                [
                    bounds.min.X,
                    bounds.min.Y,
                    bounds.min.Z,
                    bounds.max.X,
                    bounds.max.Y,
                    bounds.max.Z,
                ],
                [-45, -45, 0, 45, 45, 24],
                strict=True,
            )
        )
        radial_envelope = b.Pos(0, 0, bounds.center().Z) * b.Cylinder(45, bounds.size.Z + 2)
        outside = shape - radial_envelope
        outside_volume = boolean_solid_volume(outside)
        observations["material_outside_radius45_volume_mm3"] = outside_volume
        checks["no_material_outside_tip_radius45"] = outside_volume < 1e-5
        if len(solids) == 1:
            solid = solids[0]
            profiles = [angular_intervals(solid, 32, z) for z in (0.1, 12, 23.9)]
            observations["blade_sections"] = profiles
            checks["eight_blades_all_three_sections"] = all(
                len(row["intervals"]) == 8 for row in profiles
            )
            checks["uniform45deg_spacing"] = (
                all(close(value, 45, 1) for row in profiles for value in row["spacing_deg"])
                and checks["eight_blades_all_three_sections"]
            )
            checks["nominal3mm_sections_sampling_allowance0_35"] = (
                all(
                    close(value["arc_width_mm"], 3, 0.35)
                    for row in profiles
                    for value in row["intervals"]
                )
                and checks["eight_blades_all_three_sections"]
            )
            if all(row["eightfold_phase_deg"] is not None for row in profiles):
                twist = (
                    profiles[-1]["eightfold_phase_deg"] - profiles[0]["eightfold_phase_deg"]
                ) % 45
                observations["sampled_phase_change_deg"] = twist
                checks["correct_positive30deg_total_twist"] = close(twist, 30 * 23.8 / 24, 1)
            else:
                checks["correct_positive30deg_total_twist"] = False
            checks["actual8mm_full_span_bore"] = cylinder_pattern(solid, 4, [(0, 0)], 0, 24)
            checks["sampled_axis_void"] = all(
                not solid.is_inside((0, 0, z)) for z in (0.1, 4, 8, 12, 16, 20, 23.9)
            )
            checks["hub_annulus_present"] = all(
                solid.is_inside((8 * math.cos(math.radians(a)), 8 * math.sin(math.radians(a)), 12))
                for a in range(0, 360, 30)
            )
            rim = angular_intervals(solid, 43.5, 12)
            observations["outer_tip_section"] = rim
            checks["no_continuous_outer_rim_at_midheight"] = (
                len(rim["intervals"]) == 8 and rim["occupied_fraction"] < 0.3
            )
            checks["curved_BSpline_faces"] = any(
                "BSpline" in str(BRepAdaptor_Surface(face.wrapped, True).GetType())
                for face in solid.faces()
            )
    elif case_id == "development_enclosure_assembly":
        checks["three_displayed_solids"] = len(solids) == 3
        checks["108x76x40_4_assembly_bounds"] = all(
            close(v, t)
            for v, t in zip(
                [
                    bounds.min.X,
                    bounds.min.Y,
                    bounds.min.Z,
                    bounds.max.X,
                    bounds.max.Y,
                    bounds.max.Z,
                ],
                [-54, -38, 0, 54, 38, 40.4],
                strict=True,
            )
        )
        by_role = {}
        for name, z_min, z_max in (
            ("base", 0, 36),
            ("lid", 36.4, 40.4),
            ("pcb_reference", 15, 16.6),
        ):
            matches = [
                solid
                for solid in solids
                if close(solid.bounding_box().min.Z, z_min)
                and close(solid.bounding_box().max.Z, z_max)
            ]
            checks[f"unique_{name}_geometric_identity"] = len(matches) == 1
            if len(matches) == 1:
                by_role[name] = matches[0]
        if len(by_role) == 3:
            base, lid, pcb = (by_role[key] for key in ("base", "lid", "pcb_reference"))
            intersections = {}
            for left, right in (
                ("base", "lid"),
                ("base", "pcb_reference"),
                ("lid", "pcb_reference"),
            ):
                intersection = by_role[left] & by_role[right]
                intersections[f"{left}/{right}"] = boolean_solid_volume(intersection)
            observations["pairwise_intersection_volume_mm3"] = intersections
            checks["no_positive_volume_interference"] = all(
                volume < 1e-6 for volume in intersections.values()
            )
            checks["PCB_reference_exact84x48x1_6"] = all(
                close(v, t) for v, t in zip(pcb.bounding_box().size, (84, 48, 1.6), strict=True)
            )
            checks["floor_and_open_cavity_samples"] = base.is_inside((0, 0, 1.5)) and all(
                not base.is_inside((0, 0, z)) for z in (4, 18, 34)
            )
            screw_centers = [(x, y) for x in (-44, 44) for y in (-28, 28)]
            pcb_centers = [(x, y) for x in (-36, 36) for y in (-20, 20)]
            checks["all_four_actual_lid_clearance_holes"] = cylinder_pattern(
                lid, 1.7, screw_centers, 36.4, 40.4
            )
            checks["all_four_actual_base_blind_pilots"] = cylinder_pattern(
                base, 1.25, screw_centers, 28, 36
            )
            checks["base_pilot_floors_retained"] = all(
                base.is_inside((x, y, 27.5))
                and not base.is_inside((x, y, 28.1))
                and not base.is_inside((x, y, 35.9))
                for x, y in screw_centers
            )
            checks["actual_PCB_blind_pilots"] = cylinder_pattern(base, 1.4, pcb_centers, 7, 15)
            checks["PCB_pilot_floors_and_seating_annuli"] = all(
                base.is_inside((x, y, 6.5))
                and not base.is_inside((x, y, 7.1))
                and not base.is_inside((x, y, 14.9))
                and base.is_inside((x + 2, y, 14.9))
                for x, y in pcb_centers
            )
            checks["requested_cable_window_samples"] = (
                all(
                    not base.is_inside((x, 36.5, z))
                    for x in (-7.9, 0, 7.9)
                    for z in (6.1, 11, 15.9)
                )
                and base.is_inside((8.2, 36.5, 11))
                and base.is_inside((0, 36.5, 5.8))
            )
            checks["eight_requested_vent_window_samples"] = all(
                not base.is_inside((52.5, y, z))
                for y in (-21, -15, -9, -3, 3, 9, 15, 21)
                for z in (21.1, 27, 32.9)
            )
            checks["vent_webs_retained"] = all(
                base.is_inside((52.5, y, 27)) for y in (-18, -12, -6, 0, 6, 12, 18)
            )
            checks["nominal3mm_side_wall_samples"] = (
                not base.is_inside((50.8, 0, 18))
                and base.is_inside((51.2, 0, 18))
                and not base.is_inside((0, 34.8, 18))
                and base.is_inside((0, 35.2, 18))
            )
        if document:
            parts = {part["id"]: part for part in document.get("assembly", {}).get("parts", [])}
            checks["stable_part_ids_and_declared_production_roles"] = (
                all(name in parts for name in ("base", "lid", "pcb_reference"))
                and all(
                    parts[name].get("production", {}).get("role") == "make"
                    and parts[name]["production"].get("process") == "additive"
                    and (parts[name]["production"].get("material") or {}).get("name") == "PETG"
                    for name in ("base", "lid")
                )
                and parts["pcb_reference"].get("production", {}).get("role") == "reference"
            )
        else:
            checks["stable_part_ids_and_declared_production_roles"] = False
            observations["production_roles"] = (
                "Not checked: supply retained document; cannot infer roles from STEP"
            )
    elif case_id == "repair_filleted_block":
        checks["one_connected_solid"] = len(solids) == 1
        checks["centered40x30x8_bounds"] = all(
            close(v, t)
            for v, t in zip(
                [
                    bounds.min.X,
                    bounds.min.Y,
                    bounds.min.Z,
                    bounds.max.X,
                    bounds.max.Y,
                    bounds.max.Z,
                ],
                [-20, -15, -4, 20, 15, 4],
                strict=True,
            )
        )
        checks["four_actual_R2_corner_cylinders"] = cylinder_pattern(
            shape, 2, [(x, y) for x in (-18, 18) for y in (-13, 13)], -4, 4
        )
        checks["complete_rounded_profile_volume"] = close(
            shape.volume, (40 * 30 - (4 - math.pi) * 2**2) * 8, 1e-4
        )
    else:
        raise ValueError("Unknown prospective case")
    return {
        "case": case_id,
        "step_sha256": hashlib.sha256(step.read_bytes()).hexdigest(),
        "checks": checks,
        "observations": observations,
        "declared_export_checks_pass": all(checks.values()),
        "full_task_success": None,
        "physical_readiness": "unverified",
        "parameter_regeneration": "not_exercised",
        "manufacturing_export_exclusion": "not_exercised",
        "scope": "Independent development STEP checks with explicit sampling. Not a general validator, manufacturing acceptance, source applicability check or physical qualification.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--document", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.expected_sha256
        and hashlib.sha256(args.step.read_bytes()).hexdigest() != args.expected_sha256
    ):
        raise ValueError("Retained STEP hash changed")
    doc = json.loads(args.document.read_text()) if args.document else None
    result = check(args.case, args.step, doc)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["checks"]))
