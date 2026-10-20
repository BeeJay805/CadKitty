You configure a single drone/robot ARM as an ArmSpec: a structural strut that carries a
NEMA-17 motor mount at its tip — ONE rigid printed part (the "mechanical interface + computed
shell" fusion). A deterministic builder turns your CATEGORICAL choices into exact geometry; you
place NO coordinates and invent NO numbers.

Fill `cross_section` (the strut shape):

- `hexagonal` — a stiff, light hex strut; the default arm member, when stiffness-for-its-weight
  matters.
- `rectangular` — a rectangular strut; use when the arm must present flat faces (to stack or
  mate flush along its length).

Fill `reach` (how far the arm spans — the overall length class):

- `compact` — a short arm (a tight/compact build).
- `standard` — a normal-reach arm; the default when reach is unspecified.
- `long` — a long-reach / large arm.

Copy `material`, `min_wall_mm`, and `max_overhang_deg` straight from the Blueprint.

Do NOT set `envelope_mm` — leave it null. The arm's whole envelope (the span from `reach`, and
the lateral + height from the NEMA-17 mount facts) is DERIVED deterministically; the long axis is
X and the motor mounts at the +X tip. The interpreter also derives the strut length and
cross-section size and the mount pad. You choose only the two categories above; supply no numbers.
