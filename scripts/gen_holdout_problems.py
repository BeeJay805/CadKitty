"""Seeded, parametric HELD-OUT problem generator -- the overfit oracle (ADR-0048 lab).

An autonomous measure->improve loop that tunes against the FROZEN benchmarks will MEMORIZE them, and
the gains will be fake. This emits FRESH problems the loop never trained on, DETERMINISTICALLY from a
--seed (same seed -> byte-identical output) so a snapshot is reproducible AND the loop can ROTATE to a
new seed each cycle (so it cannot memorize even the held-out set over many iterations).

Two shapes, two benches -- each the EXACT shape the existing loader reads, unchanged:
  - problems.json             -- invention LifecycleProblem shape (forge_mind.invention.rubric.load_problems):
                                 {version, description, problems:[{id, tier, problem, min_concepts}]}.
  - arbitrary_holdout_v1.json -- CAD breadth shape (scripts/bench_arbitrary.py loader):
                                 {version, ..., prompts:[{id, class, prompt, paraphrases, expect}]}.

DISJOINT from the frozen sets (arbitrary_v1 + invention_v1 / _normal_v1 / _hard_v1): a `holdout_` id
namespace, distinct templates/params, and an explicit assertion that no id collides and no generated
text is a near-duplicate (token Jaccard) of any frozen prompt. The CAD `expect` rubrics are
upper-bound bbox + clearance-hole counts the bench re-measures against the produced artifacts.

Writes <out>/{problems.json, arbitrary_holdout_v1.json, MANIFEST.sha256}. Offline, deterministic, $0.

Run (freeze the v1 snapshot):
  uv run --no-sync python scripts/gen_holdout_problems.py --seed 1337 --out benchmarks/holdout_v1
Run (rotate -- a fresh held-out set the loop has never seen; never commit or train on it):
  uv run --no-sync python scripts/gen_holdout_problems.py --seed 4242 --out benchmarks/results/holdout_rot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from forge_mind.invention.rubric import load_problems

_FROZEN_CAD = Path("benchmarks/arbitrary_v1/arbitrary_v1.json")
_FROZEN_INVENTION = (
    Path("benchmarks/invention_v1/problems.json"),
    Path("benchmarks/invention_normal_v1/problems.json"),
    Path("benchmarks/invention_hard_v1/problems.json"),
)
_CLEARANCE = {"M3": 3.4, "M4": 4.4, "M5": 5.4}  # mm; matches arbitrary_v1 (M5->5.4, M4->4.4)
_NEAR_DUP_JACCARD = 0.7  # reject a generated text this token-similar to any FROZEN prompt
_PARAPHRASE_RULE = (
    "P1 reorders the clauses of the base prompt without changing any number, unit, or component "
    "name. P2 swaps phrasing/register without changing any number, unit, or component name. Each "
    "variant runs once; a prompt's score is the MINIMUM across its variants."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9.]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def _paraphrases(thing: str, clauses: list[str]) -> list[str]:
    """P1 reorders clauses (numbers/units/names verbatim); P2 swaps register/phrasing only."""
    p1 = "; ".join([*clauses[1:], clauses[0]]) + f" -- design this as a single-piece {thing}."
    p2 = (
        (f"Build me a one-body {thing}: " + "; ".join(clauses) + ".")
        .replace("evenly spaced", "equally spaced")
        .replace("centre line", "center line")
    )
    return [p1, p2]


def _bbox_uniform(maxdim: float) -> list[int]:
    """An orientation-invariant upper bound: no axis of the produced part exceeds the largest
    nominal dimension, so a uniform cube + margin can never false-fail on the engine's orientation."""
    m = int(maxdim) + 20
    return [m, m, m]


def _assemble(
    idx: int,
    word: str,
    verb: str,
    thing: str,
    clauses: list[str],
    holes: list[dict[str, Any]],
    maxdim: float,
) -> dict[str, Any]:
    expect: dict[str, Any] = {
        "min_parts": 1,
        "watertight": True,
        "named_step": True,
        "bbox_max_mm": _bbox_uniform(maxdim),
    }
    if holes:
        expect["holes"] = holes
    return {
        "id": f"holdout_{word}_{idx:02d}",
        "class": word,
        "prompt": f"{verb} a single-piece {thing}: " + "; ".join(clauses) + ".",
        "paraphrases": _paraphrases(thing, clauses),
        "expect": expect,
    }


# --- CAD part families (parametric; disjoint from arbitrary_v1's L/gusset/pipe/hex parts) --------


def _cad_bracket(rng: random.Random, idx: int) -> dict[str, Any]:
    length = rng.choice([80, 100, 120, 140, 160])
    width = rng.choice([25, 30, 35, 40])
    thick = rng.choice([4, 5, 6])
    size = rng.choice(["M4", "M5"])
    dia = _CLEARANCE[size]
    n = rng.choice([2, 3, 4])
    margin = rng.choice([12, 15, 20])
    clauses = [
        f"{length} mm long x {width} mm wide x {thick} mm thick",
        f"{n} {size} clearance holes ({dia} mm) evenly spaced along the centre line",
        f"the two end holes {margin} mm from each end",
    ]
    holes = [{"diameter_mm": dia, "tol_mm": 0.5, "min_count": n}]
    return _assemble(idx, "bracket", "Design", "flat mounting bar bracket", clauses, holes, length)


def _cad_plate(rng: random.Random, idx: int) -> dict[str, Any]:
    w = rng.choice([40, 50, 60, 70])
    d = rng.choice([40, 50, 60])
    t = rng.choice([4, 5, 6])
    size = rng.choice(["M4", "M5"])
    dia = _CLEARANCE[size]
    inset = rng.choice([8, 10, 12])
    clauses = [
        f"{w} x {d} x {t} mm",
        f"four {size} clearance holes ({dia} mm)",
        f"one hole {inset} mm in from each corner",
    ]
    holes = [{"diameter_mm": dia, "tol_mm": 0.5, "min_count": 4}]
    return _assemble(idx, "plate", "Model", "rectangular mounting plate", clauses, holes, max(w, d))


def _cad_mount(rng: random.Random, idx: int) -> dict[str, Any]:
    s = rng.choice([42, 50, 56, 60])
    t = rng.choice([4, 5])
    bore = rng.choice([8, 16, 22])
    pitch = s - rng.choice([12, 16, 20])
    clauses = [
        f"{s} x {s} x {t} mm",
        f"a {bore} mm central bore",
        f"four M3 clearance holes (3.4 mm) in a {pitch} x {pitch} mm square pattern centred on the plate",
    ]
    holes = [
        {"diameter_mm": 3.4, "tol_mm": 0.5, "min_count": 4},
        {"diameter_mm": float(bore), "tol_mm": 1.0, "min_count": 1},
    ]
    return _assemble(idx, "mount", "Design", "motor mount plate", clauses, holes, s)


def _cad_standoff(rng: random.Random, idx: int) -> dict[str, Any]:
    n = rng.choice([3, 4, 5])
    od = rng.choice([6, 7, 8])
    h = rng.choice([6, 8, 10, 12])
    pitch = rng.choice([15, 18, 20])
    base_w = rng.choice([14, 16, 18])
    length = pitch * (n - 1) + od + 8
    clauses = [
        f"a base bar {length} mm long x {base_w} mm wide x 3 mm thick",
        f"{n} cylindrical standoffs each {od} mm in diameter and {h} mm tall, spaced {pitch} mm apart",
        "an M3 through-hole (3.4 mm) down the centre of every standoff",
    ]
    holes = [{"diameter_mm": 3.4, "tol_mm": 0.5, "min_count": n}]
    thing = f"row of {n} PCB standoffs on a common base bar"
    return _assemble(idx, "standoff", "Make", thing, clauses, holes, length)


# --- HARD CAD families (multiple hole groups = the "secondary-hole drop" + bolt-cluster failure
# classes measured on arbitrary_v1: bracket_pipe_clamp 6-hole, cam_indexing multi-group,
# mech_star_coupling secondary holes). All holes are AXIAL (through one face) so the bench's
# concave-cylinder measurement counts them reliably -- a CORRECT part satisfies the rubric, and the
# hardness is completeness (getting EVERY group), not a rubric the engine can never meet. These push
# pass@1 into the ~40-70% band where a pass@1->pass@8 lift is actually visible. ------------------


def _cad_flange(rng: random.Random, idx: int) -> dict[str, Any]:
    od = rng.choice([70, 80, 90, 100])
    t = rng.choice([6, 8, 10])
    bore = rng.choice([20, 25, 30, 35])
    n = rng.choice([6, 8])
    size = rng.choice(["M4", "M5"])
    dia = _CLEARANCE[size]
    pcd = od - rng.choice([16, 20, 24])
    clauses = [
        f"a round flange {od} mm in outer diameter and {t} mm thick",
        f"a {bore} mm central bore through the axis",
        f"{n} {size} clearance holes ({dia} mm) equally spaced on a {pcd} mm bolt circle",
    ]
    holes = [
        {"diameter_mm": float(bore), "tol_mm": 1.5, "min_count": 1},
        {"diameter_mm": dia, "tol_mm": 0.5, "min_count": n},
    ]
    return _assemble(idx, "flange", "Model", "round bolt flange", clauses, holes, od)


def _cad_indexer(rng: random.Random, idx: int) -> dict[str, Any]:
    od = rng.choice([70, 80, 90])
    t = rng.choice([5, 6, 8])
    bore = rng.choice([10, 12, 16])
    n_index = rng.choice([6, 8, 12])
    n_mount = rng.choice([3, 4])
    clauses = [
        f"a round index plate {od} mm in outer diameter and {t} mm thick",
        f"a {bore} mm central bore through the axis",
        f"{n_index} Ø3 mm index holes equally spaced on a ring near the rim, all through the thickness",
        f"{n_mount} M4 clearance holes (4.4 mm) equally spaced on an inner ring for mounting",
    ]
    holes = [
        {"diameter_mm": float(bore), "tol_mm": 1.5, "min_count": 1},
        {"diameter_mm": 3.0, "tol_mm": 0.5, "min_count": n_index},
        {"diameter_mm": 4.4, "tol_mm": 0.5, "min_count": n_mount},
    ]
    return _assemble(idx, "indexer", "Design", "rotary index disc", clauses, holes, od)


def _cad_manifold(rng: random.Random, idx: int) -> dict[str, Any]:
    length = rng.choice([70, 80, 90])
    width = rng.choice([40, 45, 50])
    t = rng.choice([12, 15, 18])
    d1 = rng.choice([10, 12])
    d2 = rng.choice([16, 20])
    clauses = [
        f"a rectangular manifold plate {length} mm long x {width} mm wide x {t} mm thick",
        f"two Ø{d1} mm bores and one Ø{d2} mm bore, all through the {t} mm thickness, spaced along the length",
        "four M4 clearance holes (4.4 mm), one near each corner",
    ]
    holes = [
        {"diameter_mm": float(d1), "tol_mm": 1.0, "min_count": 2},
        {"diameter_mm": float(d2), "tol_mm": 1.0, "min_count": 1},
        {"diameter_mm": 4.4, "tol_mm": 0.5, "min_count": 4},
    ]
    return _assemble(idx, "manifold", "Model", "manifold plate", clauses, holes, length)


_EASY_BUILDERS = [_cad_bracket, _cad_plate, _cad_mount, _cad_standoff]
_HARD_BUILDERS = [_cad_flange, _cad_indexer, _cad_manifold]


def _gen_cad(rng: random.Random, n: int, difficulty: str = "easy") -> list[dict[str, Any]]:
    if difficulty == "hard":
        builders = _HARD_BUILDERS
    elif difficulty == "mixed":
        builders = _EASY_BUILDERS + _HARD_BUILDERS
    else:
        builders = _EASY_BUILDERS
    return [builders[i % len(builders)](rng, i) for i in range(n)]


# --- invention-style PROBLEM statements (LifecycleProblem shape; disjoint from invention_*) -------


def _problem(idx: int, tier: str, min_concepts: int, text: str) -> dict[str, Any]:
    return {
        "id": f"holdout_inv_{idx:02d}",
        "tier": tier,
        "problem": text,
        "min_concepts": min_concepts,
    }


def _inv_anti_slide(rng: random.Random, idx: int) -> dict[str, Any]:
    obj = rng.choice(["coffee mug", "water bottle", "travel cup"])
    surf = rng.choice(["tilted standing desk", "sloped drafting table"])
    return _problem(
        idx,
        "easy",
        2,
        f"My {obj} keeps sliding around on my {surf}. I want a 3D-printable way to stop it from "
        "sliding off without clamping anything to the surface or leaving adhesive residue.",
    )


def _inv_cable(rng: random.Random, idx: int) -> dict[str, Any]:
    n = rng.choice(["three", "four"])
    return _problem(
        idx,
        "normal",
        2,
        f"A tangle of {n} charger cables hangs behind my monitor and slips down behind the desk. I "
        "need a 3D-printable way to gather and route them up to the desktop with no screws or adhesive.",
    )


def _inv_tablet(rng: random.Random, idx: int) -> dict[str, Any]:
    place = rng.choice(["kitchen counter", "workbench"])
    return _problem(
        idx,
        "normal",
        2,
        f"I follow recipes and guides on a tablet at my {place} and want it held at a readable angle, "
        "raised off the surface so spills run under it, and wipeable -- 3D-printable with no hardware.",
    )


def _inv_keyboard(rng: random.Random, idx: int) -> dict[str, Any]:
    obj = rng.choice(["mechanical keyboard", "trackpad"])
    return _problem(
        idx,
        "easy",
        2,
        f"My {obj} creeps across the desk while I type. I want a 3D-printable way to keep it from "
        "sliding that grips the desk without screws and leaves no residue when removed.",
    )


def _inv_dryrack(rng: random.Random, idx: int) -> dict[str, Any]:
    obj = rng.choice(["freshly washed 3D-printed parts", "small painted miniatures"])
    return _problem(
        idx,
        "normal",
        3,
        f"I need somewhere to air-dry {obj} on my desk so they don't touch each other or the surface, "
        "drips are caught, and it folds flat when not in use -- 3D-printable, no hardware.",
    )


def _inv_bits(rng: random.Random, idx: int) -> dict[str, Any]:
    obj = rng.choice(["hex keys", "screwdriver bits"])
    return _problem(
        idx,
        "easy",
        2,
        f"My {obj} roll around loose in a drawer and I can never find the size I want. I want a "
        "3D-printable desk organizer that holds them upright, sorted by size, with no hardware.",
    )


def _gen_invention(rng: random.Random, n: int) -> list[dict[str, Any]]:
    templates = [_inv_anti_slide, _inv_cable, _inv_tablet, _inv_keyboard, _inv_dryrack, _inv_bits]
    return [templates[i % len(templates)](rng, i) for i in range(n)]


# --- disjointness + load verification ------------------------------------------------------------


def _frozen_corpus() -> tuple[set[str], list[set[str]]]:
    ids: set[str] = set()
    texts: list[str] = []
    cad = json.loads(_FROZEN_CAD.read_text(encoding="utf-8"))
    for p in cad["prompts"]:
        ids.add(p["id"])
        texts.append(p["prompt"])
        texts.extend(p.get("paraphrases", []))
    for path in _FROZEN_INVENTION:
        for prob in load_problems(path):
            ids.add(prob.id)
            texts.append(prob.problem)
    return ids, [_tokens(t) for t in texts]


def _assert_disjoint(cad: list[dict[str, Any]], inv: list[dict[str, Any]]) -> None:
    frozen_ids, frozen_tok = _frozen_corpus()
    items = [(c["id"], c["prompt"]) for c in cad] + [(p["id"], p["problem"]) for p in inv]
    ids = [i for i, _ in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids within the holdout set")
    clash = sorted(set(ids) & frozen_ids)
    if clash:
        raise SystemExit(f"id overlap with the frozen sets: {clash}")
    seen: list[str] = []
    for _id, text in items:
        if text in seen:
            raise SystemExit(f"duplicate prompt within the holdout set: {text[:60]!r}")
        seen.append(text)
        tok = _tokens(text)
        worst = max((_jaccard(tok, ft) for ft in frozen_tok), default=0.0)
        if worst > _NEAR_DUP_JACCARD:
            raise SystemExit(
                f"near-duplicate of a frozen prompt (jaccard {worst:.2f}): {text[:60]!r}"
            )


def _verify_loadable(out: Path) -> tuple[int, int]:
    """Prove BOTH benches can load the snapshot, the way their real loaders do."""
    probs = load_problems(out / "problems.json")  # invention bench loader (raises on a bad shape)
    if not probs:
        raise SystemExit("problems.json loaded empty")
    cad = json.loads((out / "arbitrary_holdout_v1.json").read_text(encoding="utf-8"))
    if not isinstance(cad.get("prompts"), list) or not cad["prompts"]:
        raise SystemExit("CAD holdout has no prompts (the CAD bench loader would reject it)")
    for p in cad["prompts"]:
        if "id" not in p or "prompt" not in p:
            raise SystemExit(f"CAD prompt missing id/prompt: {p}")
    # the MANIFEST must pin the actual bytes (the CAD bench refuses on a hash mismatch)
    want = {}
    for line in (out / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, name = line.split(None, 1)
            want[name.strip()] = digest
    for name, digest in want.items():
        if _sha256(out / name) != digest:
            raise SystemExit(f"MANIFEST hash mismatch for {name}")
    return len(probs), len(cad["prompts"])


def _write(
    out: Path,
    seed: int,
    cad: list[dict[str, Any]],
    inv: list[dict[str, Any]],
    difficulty: str = "easy",
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cad_doc = {
        "version": "holdout_v1",
        "generated_by": "scripts/gen_holdout_problems.py (seeded parametric generator; deterministic)",
        "seed": seed,
        "difficulty": difficulty,
        "holdout": "HELD-OUT oracle -- never train or tune against this; measure true generalization only.",
        "paraphrase_rule": _PARAPHRASE_RULE,
        "prompts": cad,
    }
    inv_doc = {
        "version": "holdout_v1",
        "description": (
            "HELD-OUT invention-lifecycle problems (LifecycleProblem shape) for true-generalization "
            "measurement. GENERATED + seeded (deterministic). Never train or tune against it."
        ),
        "seed": seed,
        "problems": inv,
    }
    (out / "arbitrary_holdout_v1.json").write_text(
        json.dumps(cad_doc, indent=2) + "\n", encoding="utf-8"
    )
    (out / "problems.json").write_text(json.dumps(inv_doc, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# holdout_v1 -- HELD-OUT problem oracle, GENERATED (not authored). Do NOT train/tune against it.",
        f"# regenerate byte-identically: uv run python scripts/gen_holdout_problems.py --seed {seed} --difficulty {difficulty} --out {out.as_posix()}",
        f"# seed: {seed}  difficulty: {difficulty}",
        f"{_sha256(out / 'arbitrary_holdout_v1.json')}  arbitrary_holdout_v1.json",
        f"{_sha256(out / 'problems.json')}  problems.json",
    ]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a seeded, held-out problem oracle.")
    ap.add_argument(
        "--seed", type=int, default=1337, help="PRNG seed; same seed -> byte-identical."
    )
    ap.add_argument("--n-cad", type=int, default=12, help="Number of CAD part prompts.")
    ap.add_argument("--n-invention", type=int, default=6, help="Number of invention problems.")
    ap.add_argument(
        "--difficulty",
        choices=["easy", "hard", "mixed"],
        default="easy",
        help="easy = simple prismatic families (default; the engine nails these, pass@1~100%); "
        "hard = multi-hole-group families (flange bolt-circle, index disc, manifold -- the "
        "secondary-hole-drop failure classes, pass@1~40-70% so a pass@8 lift is measurable); "
        "mixed = both.",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("benchmarks/holdout_v1"), help="Output directory."
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cad = _gen_cad(rng, args.n_cad, args.difficulty)
    inv = _gen_invention(rng, args.n_invention)
    _assert_disjoint(cad, inv)
    _write(args.out, args.seed, cad, inv, args.difficulty)
    n_inv, n_cad = _verify_loadable(args.out)

    print(f"seed={args.seed}  difficulty={args.difficulty}  ->  {args.out}")
    print(f"  arbitrary_holdout_v1.json: {n_cad} CAD prompts (loadable by the CAD bench loader)")
    print(f"  problems.json:             {n_inv} invention problems (load_problems OK)")
    print(f"  MANIFEST.sha256:           written (pins both files; seed {args.seed})")
    print("disjoint from arbitrary_v1 + invention_v1/_normal_v1/_hard_v1: OK (ids + near-dup text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
