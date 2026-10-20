"""B5 showcase — a natural rated-part request -> a LEGIBLE certified part (no LLM, $0, reproducible).

Rides the SHIPPED physics-route machinery (T2/T3/B2a): it faithfully mirrors ``_generate_physics``
AFTER extraction, so it exercises the whole authoring + certificate stack without the LLM
extractor (the live extractor half is proven separately -- TEST_LOG row 86). A hand-built
``ExtractedRequirements`` for a shelf bracket ("holds N kg") ->

    to_spec (B2a: a declared load with no stated bound gets a labeled span/250 serviceability
             default) -> route -> author_part (the closed-form / topology authors; the LLM floor
             is injected as a REFUSAL) -> monotone fallback -> the uniform verify_full gate ->
             _physics_certificate -> render_certificate_html.

Prints the verdict and writes ``certificate.html`` -- the same one-page receipt the server serves.

Usage:  python scripts/rated_part_certificate_demo.py <payload_kg> <out_dir> [bound_mm]
        bound_mm omitted / "-" -> B2a defaults span/250; else the stated bound is used verbatim.

WINDOWS SPAWN GUARD (TEST_LOG row 83 lesson, do not remove): the FEA runs in a spawned
subprocess; on Windows ``spawn`` re-imports ``__main__``, so ALL execution MUST stay under
``if __name__ == "__main__":`` -- otherwise every FEA child re-runs the authoring and races the
certificate write (the symptom is phantom NOT-CLAIMABLE certificates).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["FORGE_PHYSICS_ROUTE"] = "1"
os.environ.setdefault("FORGE_FEA_TIMEOUT_S", "500")  # batch budget, not the live server's 30 s
os.environ["FORGE_TRAIN_LOG"] = "0"


def main() -> None:
    from forge_mind.api.core import _physics_certificate, render_certificate_html
    from forge_mind.structures.dispatch import author_part, refusing_author
    from forge_mind.structures.extract import ExtractedRequirements, to_spec

    payload_kg = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("rated_part_demo")
    stated_bound = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    out.mkdir(parents=True, exist_ok=True)

    sag = f", without sagging more than {stated_bound:g} mm" if stated_bound else ""
    prompt = (
        f"a wall-mount shelf bracket, about 140 x 40 x 60 mm, that holds {payload_kg:g} kg{sag}"
    )
    extracted = ExtractedRequirements(
        part_name="shelf bracket",
        envelope_mm=(140.0, 40.0, 60.0),
        has_declared_load=True,
        payload_kg=payload_kg,
        deflection_bound_mm=stated_bound,  # None -> B2a default span/250; else stated (verbatim)
        material="pla",
        load_note=f"holds {payload_kg:g} kg{sag}",
    )
    spec, extraction = to_spec(extracted, default_envelope_mm=(150.0, 30.0, 60.0))
    print(f"prompt: {prompt}")
    print(
        f"spec: envelope={spec.envelope_mm} load_N={spec.load.force_n:.2f} "
        f"bound_mm={spec.deflection_bound_mm} ({extraction['bound_provenance']}) "
        f"components={len(spec.components)}"
    )

    t0 = time.time()
    outcome = author_part(
        spec,
        out / "physics",
        authors={"agentic": refusing_author("agentic", "showcase: physics authors only")},
        part_name="shelf bracket",
    )
    dt = time.time() - t0
    final = outcome["final"]
    gate = outcome["attempts"][-1]["gate"]
    print(f"\nauthored in {dt:.1f}s  provenance={final.provenance}  claimable={gate['claimable']}")
    print(f"attempts: {[(a['rung'], a['gate']['claimable']) for a in outcome['attempts']]}")
    print(f"checks: {gate['checks']}")
    print(f"measured: {gate.get('measured')}")

    cert = _physics_certificate(
        extraction, gate, final.provenance, spec.ref_mass_g, outcome["attempts"]
    )
    (out / "certificate.html").write_text(render_certificate_html(cert, prompt), encoding="utf-8")
    print(f"\nVERDICT: {'CLAIMABLE' if cert.claimable else 'NOT CLAIMABLE'}")
    print(f"cert: {out / 'certificate.html'}")


if __name__ == "__main__":
    main()
