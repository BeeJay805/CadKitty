"""Autopsy instrument: measure a humanoid run's proportions + form class against the
FORGE_FIDELITY_DOCTRINE anthropometry table (350 mm figure).

Usage: python measure_humanoid_proportions.py <run_dir>
Reads plan.json (joint stations from the cards' stated Z numbers are NOT parsed --
plans differ; we measure the BUILT artifacts) and each accepted part's final STL:
per-part bbox extents + world Z spans + a box-likeness score (bbox volume / mesh
volume: ~1.0 = a filled box; lofted/tapered/domed forms score higher). Prints a
table; the reviewer judges against the doctrine's stated stations."""

import json
import sys
from pathlib import Path

import trimesh

TABLE = {  # the doctrine's stated set for H=350 (segment: expected mm)
    "head height": 45.0,
    "torso height": 120.0,
    "thigh (hip->knee)": 75.0,
    "shank+foot (knee->sole)": 105.0,
    "upper arm": 55.0,
    "forearm": 60.0,
}

run = Path(sys.argv[1])
plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
print(f"object: {plan.get('object_name')}  parts: {len(plan.get('parts', []))}")
print(f"{'part':28s} {'ext X x Y x Z (mm)':26s} {'Z span':16s} {'boxness':7s}")
rows = []
for part in plan.get("parts", []):
    name = part.get("name")
    stl = None
    pdir = run / name
    if pdir.is_dir():
        cands = sorted(pdir.rglob("*.stl"), key=lambda p: p.stat().st_mtime)
        stl = cands[-1] if cands else None
    if stl is None:
        print(f"{name:28s} NO STL")
        continue
    m = trimesh.load(str(stl), force="mesh")
    e = m.extents
    bbox_vol = float(e[0] * e[1] * e[2])
    vol = float(abs(m.volume)) if m.is_volume else float("nan")
    boxness = vol / bbox_vol if bbox_vol > 0 and vol == vol else float("nan")
    rows.append((name, e, m.bounds[0][2], m.bounds[1][2], boxness))
    print(
        f"{name:28s} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f}    "
        f"{m.bounds[0][2]:7.1f}..{m.bounds[1][2]:7.1f}  {boxness:5.2f}"
    )

if rows:
    zmax = max(r[3] for r in rows)
    zmin = min(r[2] for r in rows)
    print(f"\nassembled Z span: {zmin:.1f} .. {zmax:.1f}  (height {zmax - zmin:.1f} mm)")
print("\nDOCTRINE TABLE (350 mm):", json.dumps(TABLE))
print(
    "boxness ~1.00 = solid rectangular box; a tapered loft ~0.6-0.8; a domed capsule"
    " ~0.6-0.7; shells lower. Judge limb parts: majority must NOT read ~1.0."
)
