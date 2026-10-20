"""Procedural prompt generator for the 4090 flywheel harvest (founder-directed 2026-07-14).

Emits a large, DIVERSE, DIFFICULTY-TIERED bank of natural-language part specs to feed the
reject-sampling harvest (scripts/harvest.py). Diversity + labels beat raw count, so each part
FAMILY is a parametric template sampled over realistic ranges (dims, thread sizes, hole counts,
patterns), tagged with:
  * family  -- the part class (for balanced curation + per-class pass-rate tracking)
  * tier    -- 1 easy / 2 medium / 3 hard: steers the curriculum (harvest easy-heavy first,
               widen as the model's pass-rate climbs -- the reject filter self-curricula the rest)
  * features-- the declared features the validator/judge should find (intent, for reward shaping)

Deterministic (seeded) so a re-run reproduces the same bank; pass a different --seed for more.
Output: benchmarks/harvest/prompts.jsonl  (one row per prompt).

Run:  python scripts/gen_prompts.py [count] [--seed N] [--out PATH]
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

# ---- parametric part families -------------------------------------------------------------
# Each builder takes a seeded RNG and returns (prompt_text, [features]). Ranges are chosen to be
# printable and realistic (FDM, mm). Tier reflects PLANNING + BUILDING difficulty.

R = random.Random


def _mm(lo: int, hi: int, rng: R, step: int = 1) -> int:
    return rng.randrange(lo, hi + 1, step)


def _choice(rng: R, *xs: str) -> str:
    return rng.choice(xs)


def f_washer(rng: R) -> tuple[str, list[str]]:
    idd = _mm(4, 20, rng)
    od = idd + _mm(6, 20, rng, 2)
    h = _mm(2, 8, rng)
    ch = _choice(rng, "chamfered", "square")
    return (
        f"a washer / spacer, {idd} mm inner diameter, {od} mm outer diameter, {h} mm thick, "
        f"{ch} edges",
        ["through bore", "annular ring"],
    )


def f_standoff(rng: R) -> tuple[str, list[str]]:
    m = _choice(rng, "M3", "M4", "M5")
    ln = _mm(8, 30, rng)
    shape = _choice(rng, "hex", "round")
    return (
        f"a {shape} threaded standoff, {m} thread, {ln} mm long",
        [f"{m} thread", f"{shape} body"],
    )


def f_plate(rng: R) -> tuple[str, list[str]]:
    w = _mm(30, 90, rng, 5)
    d = _mm(30, 90, rng, 5)
    t = _mm(3, 8, rng)
    n = _choice(rng, "four corner", "two", "six")
    m = _choice(rng, "M3", "M4", "M5")
    return (
        f"a mounting plate, {w} x {d} x {t} mm, with {n} {m} mounting holes",
        ["flat plate", f"{m} holes"],
    )


def f_block_hole(rng: R) -> tuple[str, list[str]]:
    w = _mm(20, 60, rng, 5)
    h = _mm(15, 40, rng, 5)
    bore = _mm(5, 20, rng)
    return (
        f"a rectangular block, {w} x {w} x {h} mm, with a {bore} mm bore through the center",
        ["solid block", "center bore"],
    )


def f_lbracket(rng: R) -> tuple[str, list[str]]:
    a = _mm(30, 60, rng, 5)
    t = _mm(3, 6, rng)
    m = _choice(rng, "M4", "M5")
    return (
        f"a right-angle L-bracket, {a} x {a} mm legs, {t} mm thick, two {m} holes per leg",
        ["right angle", f"{m} holes both faces"],
    )


def f_flange(rng: R) -> tuple[str, list[str]]:
    od = _mm(50, 90, rng, 5)
    bore = _mm(20, 40, rng, 5)
    n = _choice(rng, "four", "six", "eight")
    bcd = od - _mm(10, 20, rng, 2)
    m = _choice(rng, "M5", "M6")
    return (
        f"a pipe flange, {od} mm outer diameter, {bore} mm bore, {n} {m} bolt holes on a "
        f"{bcd} mm bolt circle",
        ["central bore", f"{n} bolt holes on BCD"],
    )


def f_knob(rng: R) -> tuple[str, list[str]]:
    dia = _mm(20, 45, rng, 5)
    shaft = _choice(rng, "6 mm D-shaft", "1/4 inch D-shaft", "5 mm round shaft")
    return (
        f"a control knob, {dia} mm diameter, knurled grip, for a {shaft}",
        ["knurled grip", "shaft bore"],
    )


def f_hook(rng: R) -> tuple[str, list[str]]:
    load = _choice(rng, "utility", "coat", "cable")
    m = _choice(rng, "M4", "M5")
    return (
        f"a wall-mounted {load} hook, screw-mounted with two {m} countersunk holes",
        ["hook curve", f"{m} mounting holes"],
    )


def f_lid(rng: R) -> tuple[str, list[str]]:
    w = _mm(50, 90, rng, 5)
    d = _mm(40, 80, rng, 5)
    return (
        f"an enclosure lid, {w} x {d} mm, 3 mm thick, with a 2 mm recessed lip that seats "
        f"into the box",
        ["recessed lip", "flat lid"],
    )


def f_clip(rng: R) -> tuple[str, list[str]]:
    dia = _mm(4, 12, rng)
    m = _choice(rng, "M3", "M4")
    return (
        f"a cable clip for a {dia} mm cable, screw-mounted with one {m} hole",
        ["cable channel", f"{m} hole"],
    )


def f_vented(rng: R) -> tuple[str, list[str]]:
    w = _mm(60, 100, rng, 10)
    d = _mm(40, 80, rng, 10)
    pat = _choice(rng, "hex", "slot", "round-hole")
    return (
        f"a {w} x {d} mm vented panel, 3 mm thick, {pat} ventilation pattern, four corner holes",
        [f"{pat} vents", "corner holes"],
    )


def f_pulley(rng: R) -> tuple[str, list[str]]:
    od = _mm(30, 60, rng, 5)
    bore = _mm(5, 12, rng)
    kind = _choice(rng, "V-belt", "flat-belt", "round-belt")
    return (
        f"a {kind} pulley, {od} mm outer diameter, {bore} mm bore, one M4 set screw",
        ["belt groove", "bore", "set screw"],
    )


def f_gear(rng: R) -> tuple[str, list[str]]:
    mod = _choice(rng, "1", "1.5", "2")
    teeth = _mm(12, 40, rng)
    bore = _mm(4, 10, rng)
    key = _choice(rng, " with a keyway", "")
    return (
        f"a spur gear, module {mod}, {teeth} teeth, {bore} mm bore{key}",
        ["involute teeth", "bore"] + (["keyway"] if key else []),
    )


def f_herringbone(rng: R) -> tuple[str, list[str]]:
    mod = _choice(rng, "1.5", "2")
    teeth = _mm(20, 36, rng)
    bore = _mm(6, 12, rng)
    return (
        f"a herringbone gear, module {mod}, {teeth} teeth, {bore} mm bore with a keyway",
        ["double-helical teeth", "bore", "keyway"],
    )


def f_snapfit(rng: R) -> tuple[str, list[str]]:
    w = _mm(50, 80, rng, 5)
    d = _mm(35, 60, rng, 5)
    h = _mm(20, 35, rng, 5)
    return (
        f"a snap-fit enclosure base, {w} x {d} x {h} mm, four internal M3 screw bosses",
        ["snap-fit lip", "M3 bosses"],
    )


def f_hinge(rng: R) -> tuple[str, list[str]]:
    w = _mm(30, 60, rng, 5)
    pin = _mm(3, 6, rng)
    return (
        f"a hinge leaf, {w} mm wide, {pin} mm pin diameter, three knuckles, two M3 mounting holes",
        ["interleaving knuckles", "pin bore", "M3 holes"],
    )


def f_servo(rng: R) -> tuple[str, list[str]]:
    arms = _choice(rng, "single", "two", "four")
    return (
        f"a servo horn adapter from a 25-tooth spline to a 4 mm D-shaft, {arms} arm",
        ["spline hub", "D-shaft output", f"{arms} arm"],
    )


def f_cap(rng: R) -> tuple[str, list[str]]:
    dia = _mm(24, 40, rng, 2)
    thr = _choice(rng, "M28x2", "M30x2", "M24x1.5")
    return (
        f"a threaded bottle cap, {dia} mm diameter, {thr} internal thread, knurled grip",
        [f"{thr} thread", "knurled grip"],
    )


def f_tslot(rng: R) -> tuple[str, list[str]]:
    ext = _choice(rng, "2020", "3030", "4040")
    m = _choice(rng, "M4", "M5", "M6")
    return (
        f"a drop-in T-slot nut for {ext} aluminium extrusion, {m} thread",
        ["T-profile", f"{m} thread"],
    )


FAMILIES: list[tuple[str, int, Callable[[R], tuple[str, list[str]]]]] = [
    # tier 1 -- easy (simple prisms/bores)
    ("washer", 1, f_washer),
    ("standoff", 1, f_standoff),
    ("plate", 1, f_plate),
    ("block_hole", 1, f_block_hole),
    ("lbracket", 1, f_lbracket),
    # tier 2 -- medium (patterns, grooves, lips, single mechanism feature)
    ("flange", 2, f_flange),
    ("knob", 2, f_knob),
    ("hook", 2, f_hook),
    ("lid", 2, f_lid),
    ("clip", 2, f_clip),
    ("vented", 2, f_vented),
    ("pulley", 2, f_pulley),
    # tier 3 -- hard (gears/threads/assemblies/splines)
    ("spur_gear", 3, f_gear),
    ("herringbone", 3, f_herringbone),
    ("snapfit", 3, f_snapfit),
    ("hinge", 3, f_hinge),
    ("servo_horn", 3, f_servo),
    ("bottle_cap", 3, f_cap),
    ("tslot_nut", 3, f_tslot),
]

# curriculum weighting: sample easy heavier at first (harvest.py can re-weight later)
TIER_WEIGHT = {1: 3, 2: 2, 3: 1}


def generate(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    weighted: list[tuple[str, int, Callable[[R], tuple[str, list[str]]]]] = []
    for fam in FAMILIES:
        weighted += [fam] * TIER_WEIGHT[fam[1]]
    out: list[dict] = []
    seen: set[str] = set()
    tries = 0
    while len(out) < count and tries < count * 20:
        tries += 1
        name, tier, fn = rng.choice(weighted)
        prompt, feats = fn(rng)
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append(
            {
                "id": f"{name}_{len(out):04d}",
                "family": name,
                "tier": tier,
                "prompt": prompt,
                "features": feats,
            }
        )
    return out


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1200
    seed = 7
    out_path = Path("benchmarks/harvest/prompts.jsonl")
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
        if a == "--out" and i + 1 < len(args):
            out_path = Path(args[i + 1])
    rows = generate(count, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter

    byt = Counter(r["tier"] for r in rows)
    byf = Counter(r["family"] for r in rows)
    print(f"wrote {len(rows)} prompts -> {out_path}")
    print("by tier:", dict(sorted(byt.items())))
    print("by family:", dict(byf.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
