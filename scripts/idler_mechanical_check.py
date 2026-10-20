"""Independent, nominal STEP witnesses for the idler review specification.

Limits and probe positions belong to the review, never read from author features.
This is a design-specific acceptance fixture, not a general engineering certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import build123d as b

from forge_mind.workspace.assembly import named_solids
from forge_mind.workspace.engineering import overlap


def cylinder(
    radius: float, length: float, position: tuple[float, float, float], axis: str = "z"
) -> Any:
    rotation = {"x": (0, 90, 0), "y": (-90, 0, 0), "z": (0, 0, 0)}[axis]
    return b.Pos(*position) * b.Rot(*rotation) * b.Cylinder(radius, length)


def annulus(
    outer: float, inner: float, length: float, position: tuple[float, float, float], axis: str
) -> Any:
    return cylinder(outer, length, position, axis) - cylinder(inner, length + 0.2, position, axis)


def mechanical_checks(step: Path) -> dict[str, Any]:
    shape = b.import_step(step)

    def path(label: str) -> str:
        remaining = label.removeprefix("ck_occ_")
        segments = []
        while remaining:
            length, rest = remaining.split("_", 1)
            size = int(length)
            segments.append(rest[:size])
            remaining = rest[size:].removeprefix("_")
        return "/".join(segments)

    solids = {path(k): v for k, v in named_solids(shape).items()}
    required = {
        "base_fixed",
        "support_left",
        "support_right",
        "moving/carrier",
        "moving/idler",
        "adjuster",
        "adjuster_retention",
        "handwheel",
        "hub_crossbolt",
        "hub_crossnut",
    }
    if not required <= solids.keys():
        raise ValueError(
            "Missing required exported occurrence identities: "
            + str(sorted(required - solids.keys()))
        )
    checks: list[dict[str, Any]] = []

    def probe(name: str, tool: Any, body: str, contains: bool = False) -> None:
        volume = overlap(solids[body], tool)
        error = abs(tool.volume - volume) if contains else volume
        checks.append(
            {
                "id": name,
                "status": "pass" if error <= 1e-5 else "fail",
                "body": body,
                "error_volume_mm3": error,
                "probe_volume_mm3": tool.volume,
                "method": "Exported STEP contains fixed material witness"
                if contains
                else "Exported STEP clears fixed probe",
            }
        )

    for x, y in itertools.product([-50, 50], [-30, 30]):
        probe(f"mount_through_{x}_{y}", cylinder(3.299, 10, (x, y, 4)), "base_fixed")
    for y in [-24, 24]:
        probe(f"guide_open_{y}", cylinder(4.099, 29.8, (0, y, 18), "x"), "moving/carrier")
        probe(
            f"guide_wall_3mm_{y}",
            annulus(7.1, 4.101, 29.8, (0, y, 18), "x"),
            "moving/carrier",
            True,
        )
    probe("roller_radial_clearance_0_5", cylinder(20.499, 30, (0, 0, 45), "y"), "moving/carrier")
    for x, y in itertools.product([-34, 34], [-12, 12]):
        bolt = f"support_{'p' if x > 0 else 'n'}{'p' if y > 0 else 'n'}_bolt"
        probe(f"support_shank_{x}_{y}", cylinder(1.999, 24.9, (x, y, 15.5)), bolt, True)
        probe(f"support_bottom_clearance_{x}_{y}", cylinder(1.999, 0.9, (x, y, 2.5)), "base_fixed")
        probe(
            f"support_thread_host_{x}_{y}",
            annulus(3.0, 2.001, 4.8, (x, y, 5.5), "z"),
            "base_fixed",
            True,
        )
        probe(
            f"support_head_seat_{x}_{y}",
            annulus(3.4, 2.251, 0.1, (x, y, 27.9), "z"),
            "support_right" if x > 0 else "support_left",
            True,
        )
        checks.append(
            {
                "id": f"support_nominal_overlap_{x}_{y}",
                "status": "pass" if abs(solids[bolt].bounding_box().min.Z - 3) < 1e-5 else "fail",
                "nominal_overlap_mm": 8 - solids[bolt].bounding_box().min.Z,
                "effective_thread_engagement": "unverified",
            }
        )
    for sign, side in [(-1, "negative"), (1, "positive")]:
        probe(
            f"outer_shoulder_{side}",
            annulus(10.95, 10.05, 0.1, (0, sign * 7.9, 45), "y"),
            "moving/idler",
            True,
        )
        probe(
            f"outer_retainer_{side}",
            annulus(10.95, 10.05, 0.1, (0, sign * 15.1, 45), "y"),
            f"moving/retainer_{side}",
            True,
        )
        probe(
            f"inner_spacer_{side}",
            annulus(4.95, 4.15, 0.1, (0, sign * 15.1, 45), "y"),
            f"moving/spacer_{side}",
            True,
        )
        probe(
            f"retainer_inner_clearance_{side}",
            cylinder(9.999, 1.2, (0, sign * 15.5, 45), "y"),
            f"moving/retainer_{side}",
        )

    # Derive endplay from the two exported capturing faces, not the input value.
    nut_min = solids["adjuster_retention"].bounding_box().min.X
    support_bounds = solids["support_right"].bounding_box()
    shoulder = solids["adjuster"] & annulus(5.9, 4.2, 20, (30, 0, 18), "x")
    if shoulder is None or shoulder.volume <= 0:
        raise ValueError("No exported screw capture shoulder")
    shoulder_max = shoulder.bounding_box().max.X
    left_gap = support_bounds.min.X - shoulder_max
    right_gap = nut_min - support_bounds.max.X
    endplay = left_gap + right_gap
    checks.append(
        {
            "id": "screw_endplay",
            "status": "pass"
            if 0.15 - 1e-6 <= endplay <= 0.35 + 1e-6 and min(left_gap, right_gap) >= -1e-6
            else "fail",
            "measured_mm": endplay,
            "allowed_mm": [0.15, 0.35],
            "capture_face_gaps_mm": [left_gap, right_gap],
            "method": "Both exported capture-face gaps: shoulder/support and support/nut",
        }
    )
    probe("shoulder_capture", annulus(5.9, 4.2, 0.1, (29.9, 0, 18), "x"), "adjuster", True)
    for y in [-5, 5]:
        # Flat bearing pads under the M3 head/nut, excluding their cross-bore.
        probe(
            f"crosspin_flat_seat_{y}",
            annulus(2.85, 1.61, 0.05, (56, y - math.copysign(0.04, y), 18), "y"),
            "handwheel",
            True,
        )
    probe(
        "crosspin_transverse_shank", cylinder(1.499, 13.8, (56, 2, 18), "y"), "hub_crossbolt", True
    )
    probe("crosspin_shaft_host", annulus(2.8, 1.601, 2, (56, 0, 18), "y"), "adjuster", True)
    for y in [-4.7, 4.7]:
        probe(
            f"crosspin_hub_host_{y}", annulus(2.5, 1.601, 0.4, (56, y, 18), "y"), "handwheel", True
        )

    # Full 360-degree conservative envelopes for the transverse fasteners, plus
    # the screw's entire axial play. A bounding-box corner overestimates radius.
    rotation = []
    for name in ["hub_crossbolt", "hub_crossnut"]:
        bounds = solids[name].bounding_box()
        radius = math.hypot(
            max(abs(bounds.min.Y), abs(bounds.max.Y)),
            max(abs(bounds.min.Z - 18), abs(bounds.max.Z - 18)),
        )
        xmin, xmax = bounds.min.X - 0.35, bounds.max.X
        swept = cylinder(radius, xmax - xmin, ((xmin + xmax) / 2, 0, 18), "x")
        hits = [
            {"body": target, "overlap_mm3": overlap(swept, solids[target])}
            for target in ["base_fixed", "support_right", "support_left", "moving/carrier"]
        ]
        rotation.append(
            {
                "body": name,
                "radius_bound_mm": radius,
                "base_clearance_lower_bound_mm": 10 - radius,
                "status": "pass" if all(h["overlap_mm3"] <= 1e-5 for h in hits) else "unverified",
                "witnesses": hits,
            }
        )

    # Fixed approach envelopes include the whole straight insertion corridor.
    # Driver/socket dimensions below are tool requirements, not a purchased-tool claim.
    tools = []

    def access(name: str, tool: Any, absent: set[str]) -> None:
        hits = [
            {"body": body, "overlap_mm3": overlap(tool, solid)}
            for body, solid in solids.items()
            if body not in absent
        ]
        tools.append(
            {
                "id": name,
                "status": "pass" if all(h["overlap_mm3"] <= 1e-5 for h in hits) else "fail",
                "absent_at_stage": sorted(absent),
                "hits": [h for h in hits if h["overlap_mm3"] > 1e-5],
                "method": "Entire specified tool approach volume intersects exported stationary STEP",
            }
        )

    for x, y in itertools.product([-34, 34], [-12, 12]):
        bolt = f"support_{'p' if x > 0 else 'n'}{'p' if y > 0 else 'n'}_bolt"
        access(
            f"support_driver_{x}_{y}",
            cylinder(2, 30, (x, y, 47)) + cylinder(8, 50, (x, y, 87)),
            {bolt},
        )
    roller_group = {
        n for n in solids if n.startswith(("moving/idler", "moving/bearing_", "moving/retainer"))
    }
    for sign, suffix in [(1, "p"), (-1, "n")]:
        for x, second in [(16, "p"), (-16, "n")]:
            target = f"moving/retainer_screw_{suffix}{second}"
            access(
                target + "_preassembly_driver",
                cylinder(1.2, 40, (x, sign * 37.65, 45), "y"),
                set(solids) - roller_group | {target},
            )
    # Include all possible hex-drive material; only the target nut is excluded.
    # The central bore clears the projecting screw throughout the approach.
    socket = annulus(9.5, 4.2, 33.8, (55.15, 0, 18), "x")
    access(
        "retention_pass_through_socket",
        socket,
        {"handwheel", "hub_crossbolt", "hub_crossnut", "adjuster_retention"},
    )

    # Disassembly order: left support removed, guides withdrawn, carriage/screw
    # assembly withdrawn to the left. These are sampled demonstrations only.
    sequence = []
    absent = {
        "support_nn_bolt",
        "support_np_bolt",
        "handwheel",
        "hub_crossbolt",
        "hub_crossnut",
        "adjuster_retention",
    }
    steps = [
        ({"support_left"}, -12.0),
        ({"guide_positive", "guide_negative"}, -100.0),
        ({n for n in solids if n.startswith("moving/")} | {"adjuster"}, -120.0),
    ]
    for group, travel in steps:
        hits = []
        for index in range(21):
            dx = travel * index / 20
            for moving in group:
                placed = b.Pos(dx, 0, 0) * solids[moving]
                for fixed in solids.keys() - group - absent:
                    volume = overlap(placed, solids[fixed])
                    if volume > 1e-5:
                        hits.append(
                            {
                                "moving": moving,
                                "fixed": fixed,
                                "translation_x_mm": dx,
                                "overlap_mm3": volume,
                            }
                        )
        sequence.append(
            {
                "moving": sorted(group),
                "travel_x_mm": travel,
                "samples": 21,
                "status": "fail" if hits else "unverified",
                "sampled_clear": not hits,
                "hits": hits[:20],
                "scope": "Sampled nominal translation; threads represented by envelopes; not certified continuous/helical assembly",
            }
        )
        absent |= group
    failed = any(c["status"] == "fail" for c in checks + tools + sequence)
    return {
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "step_sha256": hashlib.sha256(step.read_bytes()).hexdigest(),
        "status": "fail" if failed else "needs_review",
        "checks": checks,
        "rotation_bounds": rotation,
        "tool_access": tools,
        "disassembly": sequence,
        "scope": "Nominal geometry witnesses only. Manufacturing fits, full threads, preload, strength, fatigue and physical assembly remain unverified.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = mechanical_checks(args.step)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "failed": [
                    c["id"]
                    for c in result["checks"] + result["tool_access"]
                    if c["status"] == "fail"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
