"""Component library: real-world objects an electronics tray holds (ADR-0007 Stage D).

Pure data. A ``BoardComponent`` carries a board's dimensions, mounting rectangle,
screw clearance, build defaults, and optional connector-bank ports. The electronics
tray interpreter resolves a ``board_component_id`` to one of these and builds the
tray from its facts, so adding a board is a data entry, not new geometry code.

Minimal by design: two hand-authored boards (``raspberry_pi_4``, ``generic_rect_pcb``).
Motors, bearings, fans, Arduino, and external-CAD import are deliberately deferred.
"""
