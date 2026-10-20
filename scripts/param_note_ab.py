"""B6 isolation A/B (ADR-0106): does the Stage-1 PARAMS-contract NOTE alone degrade a
REFERENCE-LESS baseline? — the one regression vector the post-loop validator gate cannot see.

The council's decisive finding: `FORGE_PARAM_OPT` drove BOTH the author note (global, changes every
part's authored output) and the polish (validator-gated, never-worse). The split (ADR-0106) lets us
run the note WITHOUT the polish. `author_initial` never fetches a reference, so an author-level A/B
is inherently reference-less — precisely the majority case where the note's effect was unmeasured.

For each fixed brief, author it n times with the note OFF (arm A) vs ON (arm B) and record whether
the code BUILDS a watertight solid + basic geometry, plus whether arm B actually emitted a PARAMS
dict (proof the note took effect). If B ~ A on build/watertight rate, the note is harmless on
reference-less parts and the reference-less extension is safe; if B < A, keep the note reference-
gated or off. $0 on the Opus subscription (FORGE_LLM_PROVIDER=claude_code).

Run: FORGE_LLM_PROVIDER=claude_code python -u scripts/param_note_ab.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# Reference-LESS everyday parts WITH clearly STATED dims (the founder's revert-trigger is
# stated-dim drift): each prompt's envelope is the expected bbox -> drift = |bbox - stated|.
PROMPTS = [
    (
        "A wall-mount L-bracket, 80 x 80 x 40 mm, 5 mm thick, four M5 through-holes, a corner gusset.",
        (80.0, 80.0, 40.0),
    ),
    (
        "A NEMA-17 stepper motor mount plate, 60 mm square, 5 mm thick, the 31 mm bolt circle plus a "
        "22 mm center bore, two side slots for cable ties.",
        (60.0, 60.0, 5.0),
    ),
    (
        "A rectangular enclosure lid, 100 x 70 x 4 mm, a 2 mm lip that drops in, four countersunk "
        "corner screw holes.",
        (100.0, 70.0, 4.0),
    ),
]
N = 2  # trials per arm per part (dampens LLM nondeterminism; n=2 is directional, not conclusive)


def _bbox_drift(volume_extents: Any, stated: tuple[float, float, float]) -> float | None:
    """Max % deviation of the sorted bbox extents from the sorted stated dims -- the stated-dim
    drift proxy (the founder's revert-trigger). None if the mesh has no extents."""
    if volume_extents is None:
        return None
    got = sorted(float(x) for x in volume_extents)
    want = sorted(stated)
    return max(abs(g - w) / w * 100.0 for g, w in zip(got, want, strict=True))


def _measure(code: str, out: Path) -> dict[str, Any]:
    from forge_mind.agentic.sandbox import run_build123d_code

    res = run_build123d_code(code, out, part_name="part")
    built = bool(getattr(res, "ok", False))
    watertight = False
    volume = None
    extents = None
    if built:
        try:
            import trimesh

            stl = getattr(res, "stl", None) or (out / "part.stl")
            m: Any = trimesh.load(str(stl), force="mesh")
            watertight = bool(m.is_watertight)
            volume = round(float(abs(m.volume)), 1)
            extents = tuple(round(float(e), 2) for e in m.extents)
        except Exception:
            pass
    return {
        "built": built,
        "watertight": watertight,
        "volume_mm3": volume,
        "extents": extents,
        "code_chars": len(code),
        "has_params": "PARAMS" in code and "=" in code,
        "error": getattr(res, "error_type", None),
    }


def main() -> None:
    from forge_mind.agentic.author import author_initial
    from forge_mind.agentic.brief import research_brief
    from forge_mind.llm.claude_code_client import ClaudeCodeClient
    from forge_mind.llm.provider import make_client

    os.environ["FORGE_CLAUDE_EFFORT"] = "low"  # fast + throttle-resilient; equal across both arms
    client = (
        ClaudeCodeClient(timeout_s=900)
        if os.environ.get("FORGE_LLM_PROVIDER") == "claude_code"
        else make_client()
    )
    out = Path("benchmarks/results/param_note_ab")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for pi, (prompt, stated) in enumerate(PROMPTS):
        print(
            f"\n=== part {pi + 1}/{len(PROMPTS)}: {prompt[:55]}... (stated {stated}) ===",
            flush=True,
        )
        brief, _ = research_brief(prompt, client=client)
        print(f"  brief: {brief.object_name} | {len(brief.features)} features", flush=True)
        for arm, note in (("A_note_off", "0"), ("B_note_on", "1")):
            os.environ["FORGE_PARAM_OPT_NOTE"] = note
            os.environ["FORGE_PARAM_OPT_POLISH"] = "0"  # isolate the NOTE; never polish here
            for t in range(N):
                try:
                    t0 = time.perf_counter()
                    authored, _ = author_initial(brief, client=client)
                    dt = round(time.perf_counter() - t0, 1)
                    m = _measure(authored.code, out / f"p{pi}_{arm}_t{t}")
                except Exception as e:  # throttle / API blip: record, keep going
                    dt = None
                    m = {"built": False, "error": repr(e)[:100], "has_params": None}
                drift = _bbox_drift(m.get("extents"), stated)
                row = {"part": pi, "arm": arm, "trial": t, "author_s": dt, "drift_pct": drift, **m}
                rows.append(row)
                dstr = "-" if drift is None else f"{drift:.1f}%"
                print(
                    f"  [{arm} t{t}] built={m.get('built')} watertight={m.get('watertight')} "
                    f"extents={m.get('extents')} dim_drift={dstr} params={m.get('has_params')} "
                    f"s={dt} err={m.get('error')}",
                    flush=True,
                )

    # summary: build/watertight rate AND stated-dim drift per arm (the founder's revert-trigger)
    print("\n==== NOTE A/B SUMMARY (reference-less; arm B = note ON, polish OFF) ====", flush=True)
    for arm in ("A_note_off", "B_note_on"):
        a = [r for r in rows if r["arm"] == arm]
        nb = sum(1 for r in a if r.get("built"))
        nw = sum(1 for r in a if r.get("watertight"))
        npm = sum(1 for r in a if r.get("has_params"))
        drifts = [r["drift_pct"] for r in a if r.get("drift_pct") is not None]
        mean_d = f"{sum(drifts) / len(drifts):.1f}%" if drifts else "-"
        max_d = f"{max(drifts):.1f}%" if drifts else "-"
        print(
            f"  {arm}: built {nb}/{len(a)}  watertight {nw}/{len(a)}  PARAMS {npm}/{len(a)}  "
            f"stated-dim drift mean {mean_d} max {max_d}"
        )
    print(
        "\nread: B drift ~ A (and both small)  =>  the note does NOT worsen stated-dim adherence "
        "-> the founder's revert-trigger is not tripped; the live blanket-ON stands."
    )


if __name__ == "__main__":
    main()
