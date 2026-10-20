"""Best-of-N generation + auto-repair — a reliability wrapper around the pipeline.

Measurement/orchestration only: this package does NOT change any pipeline stage,
prompt schema, validator, interpreter, or geometry. It runs the EXISTING pipeline
up to ``n`` times, keeps the FIRST candidate that both validates and honours intent
fidelity (``gen_path == "layoutspec" and passed``), and — between non-winning
attempts — derives a TARGETED repair hint from the result's own diagnostics
(refusal / failed checks / fallback reason) to re-prompt the next attempt.
"""

from forge_mind.generation.best_of_n import (
    AttemptLog,
    BestOfNResult,
    generate_best_of_n,
)

__all__ = ["AttemptLog", "BestOfNResult", "generate_best_of_n"]
