"""Shared fitted-part dialogue advice; no document rule or approval gate."""

FITTED_PART_INTENT_GUIDANCE = (
    "For a part that fits an existing object, distinguish overall size from the surfaces and clearances that determine fit. "
    "Ask only for missing measurements that change support, contact, access, installation or removal. "
    "Explain where to measure and which units to use in plain language. "
    "Do not infer exact dimensions from pixels. "
    "If the user cannot measure a needed region, offer a useful draft with explicit limits or a small physical check. "
    "Retain supplied facts and assumptions separately. "
    "Respect the user's requested proposal and approval flow. "
    "Use the conversation photo or drawing to identify contact surfaces and intended use; retain its interpreted intent and measurement assumptions. "
    "For print delivery, prepare_print_files measures and saves separate plate-aligned made STL files. "
    "Prepare the current design revision before updating guidance, then read get_guidance again and bind instructions to the returned setup. "
    "Export original CAD with export_project and deliver the print package separately. "
    "prepare_physical_pilot returns a short bench packet; actual specimen outcomes remain pending until observed."
)
