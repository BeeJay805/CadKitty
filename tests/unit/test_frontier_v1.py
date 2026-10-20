"""frontier_v1 freeze contract (ADR-0012 E2 / W3.1).

The bench is only trustworthy if (a) the manifest pins the content, (b) the corpus is
self-describing (print-mode + ADR-0085 semantics), (c) every rubric number traces
verbatim to its own prompt (a rubric the prompt never stated manufactures false-FAILs),
and (d) hole groups are tolerance-disjoint (overlapping groups make counts ambiguous).
No LLM runs here — this is the instrument contract, not the measurement.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "frontier_v1"


def _corpus() -> dict:
    return json.loads((BENCH / "frontier_v1.json").read_text(encoding="utf-8"))


def _frozen_sha(path: Path) -> str:
    # Normalize CRLF so Windows working copies and Linux CI hash identically.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_manifest_pins_the_frozen_content() -> None:
    lines = [
        ln
        for ln in (BENCH / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert lines, "empty manifest"
    for ln in lines:
        digest, name = ln.split(maxsplit=1)
        assert _frozen_sha(BENCH / name.strip()) == digest, (
            f"{name}: frozen content changed — that is a NEW bench version (frontier_v2)"
        )


def test_corpus_is_print_mode_and_self_describing() -> None:
    from forge_mind.fabrication import FabricationMode

    d = _corpus()
    assert FabricationMode(d["fabrication_mode"]) is FabricationMode.PRINT_3D
    assert d["hole_semantics"] == "penetrations"
    assert d["bbox_semantics"] == "sorted"
    prompts = d["prompts"]
    assert len(prompts) == 14
    assert len({p["id"] for p in prompts}) == len(prompts), "duplicate prompt ids"
    for p in prompts:
        exp = p["expect"]
        assert exp["min_parts"] == 1 and exp["watertight"] and exp["named_step"], p["id"]
        assert len(exp["bbox_max_mm"]) == 3, p["id"]


def test_every_rubric_number_is_stated_in_its_own_prompt() -> None:
    # Verbatim arbitrary_v1 imports keep their own frozen rubric (M-designator diameters);
    # the self-consistency rule binds all content authored FOR frontier_v1.
    for p in _corpus()["prompts"]:
        if "verbatim" in p.get("source", ""):
            continue
        text = p["prompt"]
        stated = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mm", text)}
        for h in p["expect"].get("holes", []):
            d = float(h["diameter_mm"])
            assert d in stated, (
                f"{p['id']}: expected hole {d} mm is not stated verbatim in the prompt"
            )


def test_hole_groups_are_tolerance_disjoint() -> None:
    for p in _corpus()["prompts"]:
        if "verbatim" in p.get("source", ""):
            continue  # frozen v1 rubrics predate this rule and may not be edited
        holes = p["expect"].get("holes", [])
        for i, a in enumerate(holes):
            for b in holes[i + 1 :]:
                da, db = float(a["diameter_mm"]), float(b["diameter_mm"])
                ta = float(a.get("tol_mm", 0.4))
                tb = float(b.get("tol_mm", 0.4))
                assert abs(da - db) > ta + tb, (
                    f"{p['id']}: hole groups {da} and {db} overlap within tolerance — "
                    "counts would be ambiguous"
                )


def test_runner_accepts_the_mode_passthrough() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from bench_arbitrary import _run_variant

    params = inspect.signature(_run_variant).parameters
    assert "fabrication_mode" in params, "bench_arbitrary._run_variant lost the mode kwarg"
    from forge_mind.fabrication import FabricationMode

    assert params["fabrication_mode"].default is FabricationMode.GENERAL, (
        "existing GENERAL-mode callers must be unaffected"
    )
