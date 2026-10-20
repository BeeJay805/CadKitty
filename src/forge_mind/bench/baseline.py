"""Baseline naive runner — the falsification test.

A single LLM call writes raw CAD output with no pipeline (no DSL, no interface
library, no structured validation beyond mesh checks). If this ever matches our
pipeline's pass rate on a benchmark, the "structure beats scale" thesis is in
trouble and we want to know early.

Scaffolded in Phase 3, implemented in Phase 8 when the LLM stages exist. Until
then it raises so the harness can list it but never silently produce empty runs.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.bench.runners import Runner
from forge_mind.bench.types import TrialOutput


class BaselineNaiveRunner(Runner):
    """Single-call, no-pipeline baseline. Not implemented until Phase 8."""

    name = "baseline_naive"

    def run_trial(self, prompt: str, trial_dir: Path, trial_index: int) -> TrialOutput:
        raise NotImplementedError(
            "BaselineNaiveRunner is implemented in Phase 8 (needs the LLM client). "
            "It will issue one model call to emit raw CAD, then mesh-validate only."
        )


__all__ = ["BaselineNaiveRunner"]
