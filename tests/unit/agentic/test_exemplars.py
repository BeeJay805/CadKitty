"""Exemplar memory (lever e, ADR-0045): harvest hygiene, deterministic retrieval,
and the env-gated author hook (money-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.agentic.exemplars import (
    ExemplarRecord,
    benchmark_deny_list,
    exemplars_block,
    harvest_bench_run,
    load_all,
    retrieve,
    store,
)

_CODE = "from build123d import *\nwith BuildPart() as p:\n    Box(10, 10, 10)\n"


def _bench_run(tmp_path: Path, records: list[dict]) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({"results": records}), encoding="utf-8")
    for r in records:
        job = run / f"{r['label']}_v{r['variant']}"
        job.mkdir()
        (job / "program.py").write_text(_CODE, encoding="utf-8")
        (job / "brief.json").write_text(
            json.dumps({"object_name": r["label"].replace("_", " ")}), encoding="utf-8"
        )
    return run


def test_harvests_only_passing_builds(tmp_path: Path) -> None:
    run = _bench_run(
        tmp_path,
        [
            {"label": "garden_hook", "variant": 0, "passed": True, "prompt": "a garden hook"},
            {"label": "bad_clip", "variant": 0, "passed": False, "prompt": "a clip"},
        ],
    )
    lib = tmp_path / "lib"
    assert harvest_bench_run(run, lib, deny=set()) == 1
    names = [r.object_name for r in load_all(lib)]
    assert names == ["garden hook"]


def test_benchmark_prompts_never_enter_the_library(tmp_path: Path) -> None:
    """The contamination guard: a PASSING build of a frozen-benchmark prompt (base or
    paraphrase) is REFUSED at harvest -- the library may never contain the test set."""
    deny = benchmark_deny_list(Path.cwd())
    assert deny, "the frozen benchmark must be present in-repo for this guard test"
    bench_prompt = next(iter(deny))
    run = _bench_run(
        tmp_path,
        [
            {"label": "from_bench", "variant": 0, "passed": True, "prompt": bench_prompt},
            {"label": "seed_part", "variant": 0, "passed": True, "prompt": "a violin mute"},
        ],
    )
    lib = tmp_path / "lib"
    assert harvest_bench_run(run, lib) == 1  # the bench prompt was refused
    assert [r.object_name for r in load_all(lib)] == ["seed part"]


def test_retrieval_is_deterministic_and_similarity_ranked(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    for name, prompt in (
        ("wall hook", "a wall hook for a 12 mm rail"),
        ("pipe clamp", "a pipe clamp for 25 mm pipe"),
        ("pen cup", "a hexagonal pen cup"),
    ):
        store(
            ExemplarRecord(
                object_name=name, prompt=prompt, code=_CODE, source="t", harvested_at="t"
            ),
            lib,
        )
    a = retrieve("a hook to hang on a wall rail", lib, k=2)
    b = retrieve("a hook to hang on a wall rail", lib, k=2)
    assert [r.object_name for r in a] == [r.object_name for r in b]  # deterministic
    assert a and a[0].object_name == "wall hook"  # similarity ranked


def test_seed_prompts_are_disjoint_from_the_benchmark() -> None:
    """The harvest seed list may never collide with the frozen benchmark (the deny
    guard would refuse them anyway; this pins the seed FILE itself stays clean)."""
    from forge_mind.agentic.exemplars import _is_denied

    seeds_file = Path("scripts/exemplar_seeds.txt")
    assert seeds_file.exists()
    seeds = [
        line.strip()
        for line in seeds_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(seeds) >= 12
    deny = benchmark_deny_list()
    assert deny, "frozen benchmark must be present"
    colliding = [s for s in seeds if _is_denied(s, deny)]
    assert not colliding, f"seed prompts collide with the benchmark: {colliding}"


def test_author_block_is_env_gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib = tmp_path / "lib"
    store(
        ExemplarRecord(
            object_name="wall hook",
            prompt="a wall hook",
            code=_CODE,
            source="t",
            harvested_at="t",
        ),
        lib,
    )
    monkeypatch.delenv("FORGE_AUTHOR_EXEMPLARS", raising=False)
    assert exemplars_block("a wall hook") == ""  # OFF by default: byte-identical prompts
    monkeypatch.setenv("FORGE_AUTHOR_EXEMPLARS", str(lib))
    block = exemplars_block("a wall hook")
    assert "REFERENCE BUILDS" in block and "BuildPart" in block
    monkeypatch.setenv("FORGE_AUTHOR_EXEMPLARS", str(tmp_path / "missing"))
    assert exemplars_block("a wall hook") == ""  # empty/missing library degrades silently


def test_author_initial_carries_the_block_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook in author_initial: flag off -> the user prompt has no reference block;
    flag on -> it does. Captured via a sentinel client (no LLM)."""
    from forge_mind.agentic.author import AuthoredCode, author_initial
    from forge_mind.agentic.brief import DesignBrief
    from forge_mind.llm.cost import LLMUsage

    captured: dict[str, str] = {}

    class _Sentinel:
        def generate_structured(self, *, system, user, output_model, model, max_tokens, **kw):
            captured["user"] = user
            return (
                AuthoredCode(approach="x", code="print('hi')"),
                LLMUsage(model=model, input_tokens=1, output_tokens=1, cost_usd=0.0),
            )

    from forge_mind.agentic.brief import BriefFeature

    brief = DesignBrief(
        object_name="wall hook",
        summary="a wall hook",
        visual_description="a J-shaped hook on a back plate",
        overall_envelope_mm="40 x 20 x 60 (hand-sized wall hook)",
        features=[
            BriefFeature(
                name="hook",
                purpose="carry the hung item",
                approx_dims_mm="J profile 40 tall",
                method="sweep",
            )
        ],
        min_wall_mm=2.0,
        print_orientation_note="back plate down",
    )
    lib = tmp_path / "lib"
    store(
        ExemplarRecord(
            object_name="wall hook",
            prompt="a wall hook",
            code=_CODE,
            source="t",
            harvested_at="t",
        ),
        lib,
    )
    monkeypatch.delenv("FORGE_AUTHOR_EXEMPLARS", raising=False)
    author_initial(brief, client=_Sentinel())
    assert "REFERENCE BUILDS" not in captured["user"]
    monkeypatch.setenv("FORGE_AUTHOR_EXEMPLARS", str(lib))
    author_initial(brief, client=_Sentinel())
    assert "REFERENCE BUILDS" in captured["user"]
