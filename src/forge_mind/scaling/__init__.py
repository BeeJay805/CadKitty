"""Automated scaling engine (ADR-0016) -- SAFE LAYER ONLY.

Combination search over the bounded composable plate spec: enumerate feasible family
shapes deterministically, test each through a robustness battery, bank the high-confidence
passers and flag the uncertain ones. The risky layer (LLM-authored checkers/archetypes)
is NOT here and is deferred by name. The $-budget scale-run is print-gated (see
``config.SCALING_PRINT_CALIBRATED`` + ``engine.require_print_calibration``).
"""

from __future__ import annotations

__all__: list[str] = []
