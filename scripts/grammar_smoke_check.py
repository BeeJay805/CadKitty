"""Grammar-compile smoke-check CLI (ADR-0020, the durable meta-fix).

Thin wrapper over ``forge_mind.layout.router.check_grammar``. ADR-0019 shipped a latent break
because the offline char-count proxy was UNFAITHFUL and the ``not llm`` suite is BLIND to
compiled-grammar errors (they surface only on real API calls). This is the cheap, reusable
real-API check: it asks every structured-output schema the router emits (the classifier + each
per-path spec) to generate with a tiny max_tokens and asserts none is rejected with "the
compiled grammar is too large" (a rejected compile costs ~$0).

The same check is wired as an opt-in pytest: tests/integration/test_grammar_smoke.py (marked
``llm``, so it is deselected by the default ``-m "not llm"`` suite but runnable on demand).

Run: uv run python scripts/grammar_smoke_check.py   (exit 0 = all fit, 1 = something over)
"""

from __future__ import annotations

import sys

from forge_mind.layout.router import check_grammar


def main() -> int:
    results = check_grammar()
    all_fit = all(fits for _label, fits, _detail in results)
    for label, fits, detail in results:
        print(f"  {'FIT ' if fits else 'OVER'}  {label:<24} {detail}")
    print(f"\n{'ALL SCHEMAS FIT' if all_fit else 'SOME SCHEMAS OVER THE GRAMMAR CEILING'}")
    return 0 if all_fit else 1


if __name__ == "__main__":
    sys.exit(main())
