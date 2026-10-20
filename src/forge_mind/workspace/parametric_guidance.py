"""Shared authoring advice, not a document rule or acceptance instrument."""

PARAMETRIC_DESIGN_GUIDANCE = (
    "Before the first build, expose meaningful requested dimensions and component interfaces "
    "as named document variables. Preserve required values, including reference-part interfaces. "
    "Include useful clearance and edge-offset inputs. Derive dependent sizes and coordinates "
    "across manufactured and reference parts from shared variables or ordered relations, "
    "rather than repeat numeric literals. Bind native and Python feature parameters to those "
    "names. Python features must read bound params values; occurrence positions can use "
    "variable or relation names. Keep unrelated dimensions independent. Incidental constants "
    "remain useful outside intended controls. Name inputs by design intent. Explain meaningful "
    "dimensions that remain source-only. Variable changes rebuild geometry; they do not amend "
    "contract targets or prove fit, complete editability, or physical suitability."
)
