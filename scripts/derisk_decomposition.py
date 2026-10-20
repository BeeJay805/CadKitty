"""Step 0 de-risk: how much free-text COMPOSITION does the CURRENT path already support?

The composable path already accepts a base body + an interface list (ComposablePlateSpec). This
feeds ~6 compositional prompts end-to-end through the existing pipeline (intent -> blueprint ->
router -> interpret -> validate) and dumps, per prompt: the routed path, validated?, the refusal /
fallback reason, and the FEATURES actually authored (hardware_features + face interface decls) so we
can see what composed vs what was lost (wrong base / dropped feature / no recipe). Decides whether
the "brain" is a prompt/intent enhancement or a real new recipe-authoring step.

Run: uv run python scripts/derisk_decomposition.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.pipeline import run_pipeline

# (label, prompt, expected feature gist) — genuinely multi-feature, spanning families.
PROMPTS: list[tuple[str, str]] = [
    (
        "fan+holes",
        "A flat plate to mount a 60mm fan with two M4 clearance holes for cable tie-downs. PLA, 4mm walls.",
    ),
    (
        "bearing+2xM3",
        "A bracket with a 608 bearing seat and two M3 clearance holes. PLA, 4mm walls.",
    ),
    ("strut+inserts", "A strut with a heat-set insert at each end. PLA, 4mm min wall."),
    (
        "3-feature-plate",
        "A flat plate with a 625 bearing seat, a 6x3 magnet pocket, and four M3 clearance holes. PLA, 4mm walls.",
    ),
    (
        "motor+hw",
        "A NEMA 17 motor mount plate with two extra M5 clearance holes and a 5x2 magnet pocket. PLA, 4mm walls.",
    ),
    ("pi+fan", "A plate to mount a Raspberry Pi and a 60mm fan. PLA, 3mm walls."),
]


def _features(program: object) -> str:
    if program is None:
        return "(no program)"
    hw = [f"{f.category}:{f.key}" for f in getattr(program, "hardware_features", [])]
    decls = [d.interface_type.value for d in getattr(program, "interfaces", [])]
    return f"hardware={hw} faces={decls} ops={len(getattr(program, 'operations', []))}"


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="derisk_decomp_"))
    total = 0.0
    print("=== Step 0: current-path composition on compositional prompts ===")
    for i, (label, prompt) in enumerate(PROMPTS):
        r = run_pipeline(prompt, out / f"p{i}", use_layoutspec=True)
        total += r.total_cost_usd
        status = "VALIDATED" if (r.gen_path == "layoutspec" and r.passed) else "LOST"
        print(f"\n[{label}] {status}  route={r.gen_path} passed={r.passed}")
        print(f"   features: {_features(r.program)}")
        if r.refusal:
            print(f"   refusal: {r.refusal[:120]}")
        if r.fallback_reason:
            print(f"   fallback: {r.fallback_reason[:120]}")
    print(f"\n  TOTAL SPEND: ${total:.3f}")


if __name__ == "__main__":
    main()
