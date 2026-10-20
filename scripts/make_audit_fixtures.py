"""Gated-class fixtures for the seeded-defect audit (ADR-0083 v2 req 4).

The audit's moat classes (blind_where_through, dropped_named_feature) were corpus-limited to n=2
because few real corpus parts have clean through-bores / cleanly-deletable named features. These
hand-authored fixtures are dense in exactly those: multiple THROUGH clearance holes (blind-able),
side-effect feature blocks (dataflow-safe deletable), stated dims, position params, and a wall.

Built LOCALLY via the sandbox ($0, NO LLM) into a corpus dir the audit's loader reads
(brief.json + program.py + part.stl/step + a results.json marking each PASSED). Deterministic;
re-runnable. NOT a benchmark -- purely audit raw material, so it lives under benchmarks/results/.

Run:  uv run --no-sync python scripts/make_audit_fixtures.py
Then: uv run --no-sync python scripts/audit_validator.py "benchmarks/results/{arbitrary_*,audit_fixtures_v1}"
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge_mind.agentic.brief import BriefFeature, DesignBrief  # noqa: E402
from forge_mind.agentic.sandbox import run_build123d_code  # noqa: E402

_OUT = REPO / "benchmarks" / "results" / "audit_fixtures_v1"


@dataclass(frozen=True)
class Fixture:
    label: str
    prompt: str
    brief: DesignBrief
    code: str


def _f(
    name: str, purpose: str, dims: str, method: str, count: int = 1, standard: str | None = None
) -> BriefFeature:
    return BriefFeature(
        name=name,
        purpose=purpose,
        approx_dims_mm=dims,
        method=method,
        count=count,
        standard=standard,
    )


def _brief(name: str, env: str, feats: list[BriefFeature], wall: float = 3.0) -> DesignBrief:
    return DesignBrief(
        object_name=name,
        summary=f"a {name}",
        visual_description=f"a {name}: a flat body with through holes",
        overall_envelope_mm=env,
        features=feats,
        min_wall_mm=wall,
        print_orientation_note="largest flat face down",
        target="cosmetic",
    )


# Each code binds `result`; holes are THROUGH (depth = thickness + margin) so the confirmer sees
# them open end-to-end and the blind operator can shorten them. Feature blocks are side-effect
# `with ...:` (bind nothing) so the dataflow-safe deletion can drop them cleanly.

_FIXTURES: list[Fixture] = [
    Fixture(
        label="fx_mounting_plate",
        prompt=(
            "Design a single-piece flat mounting plate: 100 mm long x 60 mm wide x 6 mm thick; "
            "a 12 mm central drive bore through the plate; 4 M5 clearance holes (5.4 mm) on a "
            "40 mm bolt circle."
        ),
        brief=_brief(
            "mounting plate",
            "100 L x 60 W x 6 H mm",
            [
                _f("central drive bore", "shaft clearance", "12 mm dia through", "bore"),
                _f("bolt holes", "fastening", "5.4 mm dia, 40 mm bcd", "bolt_holes", count=4),
            ],
        ),
        code=(
            "from build123d import *\n"
            'PARAMS = {"plate_len": 100.0, "plate_wid": 60.0, "plate_thk": 6.0, '
            '"drive_bore_d": 12.0, "bolt_bcd": 40.0, "n_bolts": 4, "wall_min": 3.0}\n'
            "with BuildPart() as part:\n"
            '    Box(PARAMS["plate_len"], PARAMS["plate_wid"], PARAMS["plate_thk"])\n'
            "    with Locations((0, 0, 0)):\n"
            '        Hole(radius=PARAMS["drive_bore_d"] / 2, depth=PARAMS["plate_thk"] + 4)\n'
            '    with PolarLocations(PARAMS["bolt_bcd"] / 2, PARAMS["n_bolts"]):\n'
            '        Hole(radius=2.7, depth=PARAMS["plate_thk"] + 4)\n'
            "result = part.part\n"
        ),
    ),
    Fixture(
        label="fx_linkage_bar",
        prompt=(
            "Design a single-piece linkage bar: 120 mm long x 24 mm wide x 8 mm thick; a 10 mm "
            "pivot bore through each end, 100 mm apart; 3 M4 clearance holes (4.4 mm) along the "
            "centre line for lightening."
        ),
        brief=_brief(
            "linkage bar",
            "120 L x 24 W x 8 H mm",
            [
                _f("pivot bore", "pin joint", "10 mm dia through, 2 ends", "bore", count=2),
                _f("lightening holes", "mass reduction", "4.4 mm dia", "bolt_holes", count=3),
            ],
        ),
        code=(
            "from build123d import *\n"
            'PARAMS = {"bar_len": 120.0, "bar_wid": 24.0, "bar_thk": 8.0, "pivot_bore_d": 10.0, '
            '"pivot_span": 100.0, "n_light": 3, "wall_min": 3.0}\n'
            "with BuildPart() as part:\n"
            '    Box(PARAMS["bar_len"], PARAMS["bar_wid"], PARAMS["bar_thk"])\n'
            '    with Locations((-PARAMS["pivot_span"] / 2, 0, 0), (PARAMS["pivot_span"] / 2, 0, 0)):\n'
            '        Hole(radius=PARAMS["pivot_bore_d"] / 2, depth=PARAMS["bar_thk"] + 4)\n'
            '    with GridLocations(20, 0, PARAMS["n_light"], 1):\n'
            '        Hole(radius=2.2, depth=PARAMS["bar_thk"] + 4)\n'
            "result = part.part\n"
        ),
    ),
    Fixture(
        label="fx_pillow_block",
        prompt=(
            "Design a single-piece pillow block: 80 mm long x 40 mm wide x 30 mm tall; a 22 mm "
            "bearing bore through the length; 4 M6 clearance holes (6.6 mm) through the base."
        ),
        brief=_brief(
            "pillow block",
            "80 L x 40 W x 30 H mm",
            [
                _f("bearing bore", "bearing seat", "22 mm dia through", "bore"),
                _f("base mounting holes", "fastening", "6.6 mm dia", "bolt_holes", count=4),
            ],
            wall=4.0,
        ),
        code=(
            "from build123d import *\n"
            'PARAMS = {"blk_len": 80.0, "blk_wid": 40.0, "blk_hgt": 30.0, "bearing_bore_d": 22.0, '
            '"hole_dx": 60.0, "hole_dy": 24.0, "wall_min": 4.0}\n'
            "with BuildPart() as part:\n"
            '    Box(PARAMS["blk_len"], PARAMS["blk_wid"], PARAMS["blk_hgt"])\n'
            "    with Locations(Plane.XZ):\n"
            '        Hole(radius=PARAMS["bearing_bore_d"] / 2, depth=PARAMS["blk_wid"] + 4)\n'
            '    with Locations(Plane.XY.offset(PARAMS["blk_hgt"] / 2)):\n'
            '        with GridLocations(PARAMS["hole_dx"], PARAMS["hole_dy"], 2, 2):\n'
            '            Hole(radius=3.3, depth=PARAMS["blk_hgt"] + 4)\n'
            "result = part.part\n"
        ),
    ),
    Fixture(
        label="fx_cover_plate",
        prompt=(
            "Design a single-piece cover plate: 90 mm long x 70 mm wide x 5 mm thick; a 20 mm "
            "central access bore through the plate; 4 M4 clearance holes (4.4 mm), one near each "
            "corner; 2 M3 vent holes (3.4 mm) on the centre line."
        ),
        brief=_brief(
            "cover plate",
            "90 L x 70 W x 5 H mm",
            [
                _f("access bore", "cable access", "20 mm dia through", "bore"),
                _f("corner mounting holes", "fastening", "4.4 mm dia", "bolt_holes", count=4),
                _f("vent holes", "airflow", "3.4 mm dia", "bolt_holes", count=2),
            ],
        ),
        code=(
            "from build123d import *\n"
            'PARAMS = {"cov_len": 90.0, "cov_wid": 70.0, "cov_thk": 5.0, "access_bore_d": 20.0, '
            '"corner_dx": 74.0, "corner_dy": 54.0, "vent_pitch": 30.0, "wall_min": 3.0}\n'
            "with BuildPart() as part:\n"
            '    Box(PARAMS["cov_len"], PARAMS["cov_wid"], PARAMS["cov_thk"])\n'
            "    with Locations((0, 0, 0)):\n"
            '        Hole(radius=PARAMS["access_bore_d"] / 2, depth=PARAMS["cov_thk"] + 4)\n'
            '    with GridLocations(PARAMS["corner_dx"], PARAMS["corner_dy"], 2, 2):\n'
            '        Hole(radius=2.2, depth=PARAMS["cov_thk"] + 4)\n'
            '    with GridLocations(PARAMS["vent_pitch"], 0, 2, 1):\n'
            '        Hole(radius=1.7, depth=PARAMS["cov_thk"] + 4)\n'
            "result = part.part\n"
        ),
    ),
    Fixture(
        label="fx_spacer_rail",
        prompt=(
            "Design a single-piece spacer rail: 140 mm long x 20 mm wide x 10 mm thick; 5 M3 "
            "through holes (3.4 mm) evenly spaced along the centre line, 30 mm apart."
        ),
        brief=_brief(
            "spacer rail",
            "140 L x 20 W x 10 H mm",
            [
                _f("through holes", "fastening", "3.4 mm dia, 30 mm pitch", "bolt_holes", count=5),
            ],
        ),
        code=(
            "from build123d import *\n"
            'PARAMS = {"rail_len": 140.0, "rail_wid": 20.0, "rail_thk": 10.0, "n_holes": 5, '
            '"pitch": 30.0, "wall_min": 3.0}\n'
            "with BuildPart() as part:\n"
            '    Box(PARAMS["rail_len"], PARAMS["rail_wid"], PARAMS["rail_thk"])\n'
            '    with GridLocations(PARAMS["pitch"], 0, PARAMS["n_holes"], 1):\n'
            '        Hole(radius=1.7, depth=PARAMS["rail_thk"] + 4)\n'
            "result = part.part\n"
        ),
    ),
]


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for fx in _FIXTURES:
        job = _OUT / f"{fx.label}_v0"
        job.mkdir(parents=True, exist_ok=True)
        sb = run_build123d_code(fx.code, job, part_name="part")
        ok = sb.ok and sb.stl_path is not None and sb.step_path is not None
        print(f"{'OK ' if ok else 'XX '} {fx.label}: {sb.error_type or 'built'}", flush=True)
        if not ok:
            continue
        (job / "brief.json").write_text(fx.brief.model_dump_json(indent=2), encoding="utf-8")
        (job / "program.py").write_text(fx.code, encoding="utf-8")
        records.append({"label": fx.label, "variant": 0, "passed": True, "prompt": fx.prompt})
    (_OUT / "results.json").write_text(
        json.dumps({"benchmark": "audit_fixtures_v1", "results": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\n{len(records)}/{len(_FIXTURES)} fixtures built -> {_OUT}", flush=True)
    return 0 if len(records) == len(_FIXTURES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
