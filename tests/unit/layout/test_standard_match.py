"""Routing regression corpus for the standard-parts lane matcher (ADR-0035 A1).

This file is the versioned ROUTING CONTRACT: which prompt families enter the deterministic
lane (and which never do) is asserted here, so a vocabulary change is a reviewed diff, not a
silent behavior shift. Three sections:

1. RECALL -- prompts naming registry hardware + a buildable form MUST route.
2. PRECISION -- adversarial negatives (bare years, embedded numbers, the hardware itself,
   creative objects) MUST fall through to the agentic engine.
3. COVERAGE (ADR-0035 decision c) -- every reachable registry component id must have a
   vocabulary token: a new registry row FAILS here until a routing decision is made.
"""

from __future__ import annotations

import pytest

from forge_mind.layout.standard_match import standard_match

# ------------------------------------------------------------------------------ recall ---

RECALL_CORPUS = [
    # motors -- all registry sizes, all spellings
    "a NEMA 17 motor mount plate with the 31 mm bolt circle",
    "a nema17 bracket for a camera slider",
    "a NEMA-23 motor mounting plate 6 mm thick",
    "a nema 14 adapter plate",
    "a stepper motor mount for my CNC",
    "an arm for a NEMA 17 with a 100 mm reach",
    # extrusions -- all registry profiles + synonyms
    "a corner bracket for 2020 aluminum extrusion",
    "a 2020 extrusion end cap mounting plate",
    "a 2040 extrusion joining plate",
    "a 3030 profile mounting bracket",
    "a v-slot rail bracket with M5 hardware",
    "an aluminium extrusion mounting block",
    # fans -- the bench-006 family that never routed before
    "a 60mm fan mount that bolts to a NEMA 17",
    "a 60 mm fan grille",
    "a 40mm fan shroud",
    "a fan adapter from 60mm to 40mm",
    # bearings -- the composition-engine family that never routed before
    "a 608 bearing block",
    "a mounting plate with a 608 bearing seat and four M4 holes",
    "a 625 bearing holder",
    "a flanged housing for a 6000 bearing",
    "a pillow block for a 688 bearing",
    # boards
    "a raspberry pi 4 case with vent slots",
    "a raspberry pi tray",
    "a pi4 enclosure with port cutouts",
    "an arduino tray with standoffs",
    "a pcb holder for a 60x40 board",
    "a circuit board mounting plate",
    # combinations
    "an adapter plate from a NEMA-17 face to four M5 holes on a 50 mm square",
    "a 60 mm fan shroud that bolts to a 2020 aluminum extrusion with M5 hardware",
    "a bracket joining a 2020 extrusion to a NEMA 23 stepper",
]


@pytest.mark.parametrize("prompt", RECALL_CORPUS)
def test_recall(prompt: str) -> None:
    assert standard_match(prompt) is True, f"should route to the deterministic lane: {prompt!r}"


# --------------------------------------------------------------------------- precision ---

PRECISION_CORPUS = [
    # no standard ecosystem at all
    "a right-angle bracket 40 mm x 40 mm with two M4 clearance holes per leg",
    "a 25 mm knob with a 6 mm D-shaft bore and a knurled rim",
    "a flower vase with a wavy rim",
    "a hammer with a curved claw",
    "a phone stand with a 60 degree angle",
    "a desk organizer tray",
    "a hinge with a 3mm pin",
    "a gear with 24 teeth",
    "",
    # the HARDWARE itself, not a part for it (no form token)
    "a NEMA-17 stepper motor",
    "a 608 bearing",
    "a raspberry pi 4",
    "a 60mm fan",
    # numeric tokens without their hardware noun (ADR-0035 decision c)
    "built in 2020, a phone stand",
    "a 1625 mm long rod with end caps",  # 625 embedded in 1625 must not read as a bearing
    "a 2040 piece puzzle tray",  # 2040 without extrusion/profile/rail does not route
    "my 3030 address plaque",
    # near-miss vocabulary
    "a load-bearing wall hook",  # 'load-bearing' is not bearing hardware (and no form token)
    "a standard letter tray",  # 'standard' must not read as 'stand'
    "a standoff measurement chart",  # 'standoff' must not read as 'stand'
]


@pytest.mark.parametrize("prompt", PRECISION_CORPUS)
def test_precision(prompt: str) -> None:
    assert standard_match(prompt) is False, f"must fall through to the agentic engine: {prompt!r}"


# ------------------------------------------------------------- ADR-0034 snapshot (no loss) ---

ADR_0034_POSITIVES = [
    # every family the ORIGINAL 12x8 substring matcher routed must still route
    "a nema 17 mount",
    "a nema17 plate",
    "a nema-17 bracket",
    "a nema 23 adapter",
    "a nema23 case",
    "a nema-23 mount",
    "a 2020 extrusion bracket",
    "a 2020 aluminum tray",
    "a 2020 aluminium enclosure",
    "a raspberry pi shroud",
    "a pi 4 case",
    "a pi4 tray",
]


@pytest.mark.parametrize("prompt", ADR_0034_POSITIVES)
def test_adr_0034_recall_preserved(prompt: str) -> None:
    assert standard_match(prompt) is True, f"ADR-0034 recall regression: {prompt!r}"


# ------------------------------------------------------------------ registry coverage ---


def test_every_motor_component_has_a_token() -> None:
    from forge_mind.dsl.components.registry import known_motor_component_ids

    for cid in known_motor_component_ids():
        size = cid.removeprefix("nema")
        assert standard_match(f"a nema {size} mount"), f"no vocabulary for motor {cid!r}"


def test_every_extrusion_component_has_a_token() -> None:
    from forge_mind.dsl.components.registry import known_extrusion_component_ids

    for cid in known_extrusion_component_ids():
        size = cid.removeprefix("ext_")
        assert standard_match(f"a {size} extrusion bracket"), f"no vocabulary for {cid!r}"


def test_every_fan_component_has_a_token() -> None:
    from forge_mind.dsl.components.registry import known_fan_component_ids

    for cid in known_fan_component_ids():
        size = cid.removeprefix("fan_")
        assert standard_match(f"a {size} fan mount"), f"no vocabulary for fan {cid!r}"


def test_every_board_component_has_a_token() -> None:
    from forge_mind.dsl.components.registry import known_board_component_ids

    spellings = {"raspberry_pi_4": "a raspberry pi case", "generic_rect_pcb": "a pcb tray"}
    for cid in known_board_component_ids():
        assert cid in spellings, f"new board component {cid!r}: add a routing-vocabulary decision"
        assert standard_match(spellings[cid]), f"no vocabulary for board {cid!r}"


def test_every_bearing_designator_routes_with_its_noun() -> None:
    from forge_mind.dsl.standards import ALL_REGISTRIES

    designators = ALL_REGISTRIES["bearing"].keys()  # Registry API (not a dict)
    assert designators
    for key in designators:
        assert standard_match(f"a {key} bearing block"), f"no vocabulary for bearing {key!r}"
        assert not standard_match(f"a {key} block"), f"bare designator {key!r} must NOT route"
