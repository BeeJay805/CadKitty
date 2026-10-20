# Question generation rules

Ask only questions that change **geometry, fit, interfaces, fasteners, dimensions, clearances,
printability, or a user-visible design choice.** Ask in priority order and keep the set small.

## Priority 1 — required for physical fit
- mounting interface / what it attaches to
- hole pattern
- fastener type
- critical width / height / depth
- shaft / bearing / clearance dimensions
- rail / extrusion standard
- the object being mounted

## Priority 2 — required for manufacturability
- material / process
- print orientation (only if it changes geometry)
- minimum wall thickness (only if relevant)
- load / strength expectation
- whether supports are acceptable

## Priority 3 — design preference (only when it matters to the part)
- rounded vs square corners
- compact vs reinforced
- adjustable slots vs fixed holes

## Never ask
- colours or finish
- vague styling or decorative choices
- dimensions a known standard already provides
- anything the system can safely default

## Defaults
Use a reasonable default only when it is safe, and mark it clearly as a default (the brief records each
dimension's `source`). Never silently pass a default off as a user-confirmed value.

## Images and sketches
If an image or drawing is provided, identify the rough shape, mounting holes, cutouts, tabs, and slots,
but DO NOT infer exact measurements from pixels. Ask for one real dimension to set the scale. If the
image conflicts with the text, ask a clarifying question instead of choosing silently.

## Volume
A handful of sharp questions, never a wall of them. If a request like "make a motor mount" is missing
its essentials, ask: (1) motor size, (2) what it attaches to, (3) fasteners — not twelve questions.
