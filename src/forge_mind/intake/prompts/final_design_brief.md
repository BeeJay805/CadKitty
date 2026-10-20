# Final DesignBrief

Once enough is known, the intake produces a structured `DesignBrief` — the handoff contract the
existing blueprint / PartDSL / CAD / validator pipeline consumes. It is assembled deterministically
from the classified state, the cited standards, and the user's answers (plus safe, marked defaults).

A brief is **ready_for_blueprint** only when:
- every required question is answered (or safely defaulted and marked),
- every critical dimension and interface is resolved or comes from a standard,
- there are no remaining `unknowns`.

Each critical dimension records a `source`:
- `user` — the user gave it
- `standard` — it comes from a cited standard (e.g. NEMA-17 shaft clearance)
- `default` — a safe default the system chose (clearly marked; never presented as user-confirmed)
- `image` — derived from a drawing AFTER the user confirmed a real dimension

Generation is BLOCKED unless the brief is ready, no required answer is missing, and no critical
dimension is of unknown source. The brief is then rendered to a precise, standards-resolved prompt and
handed to the existing pipeline — which still runs the full CAD compiler and validator. The planner
never bypasses validation.
