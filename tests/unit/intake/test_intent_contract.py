"""The intent contract (ADR-0036): the fabrication toggle finally reaches the planner, the
classifier's ungrounded type guess can no longer rename a part ("a hammer" -> "block"), and the
typed brief renders into the STATED-FACTS block the agentic engine binds to."""

from __future__ import annotations

from forge_mind.intake.brief import (
    assemble_brief,
    brief_to_context,
    brief_to_prompt,
)
from forge_mind.intake.schema import PlannerState
from forge_mind.intake.session import DesignSession
from forge_mind.intake.standards import detect_standards

# ------------------------------------------------------------------- fabrication ---


def _session(text: str, fabrication: str | None = None) -> DesignSession:
    s = DesignSession()
    s.original_text_prompt = text
    s.fabrication = fabrication
    return s


def test_legacy_session_keeps_fdm_defaults() -> None:
    """No toggle ever sent (fabrication=None): byte-compatible with today -- FDM process + the
    printability constraints. Old clients see zero change."""
    brief = assemble_brief(_session("a 40 mm bracket"), PlannerState(), ready=True)
    assert brief.manufacturing_process == "fdm_3d_printing"
    assert "Must be 3D printable" in brief.constraints


def test_general_session_drops_printability_claims() -> None:
    """THE BOARD-REPORTED BUG: with the 3D-print toggle OFF the brief still claimed 'Must be 3D
    printable' and manufacturing_process='fdm_3d_printing'. A general session is now honest."""
    brief = assemble_brief(_session("a 40 mm bracket", "general"), PlannerState(), ready=True)
    assert brief.manufacturing_process == "general_fabrication"
    assert all("printable" not in c.lower() for c in brief.constraints)
    assert all("overhang" not in c.lower() for c in brief.constraints)
    # ... and the rendered prompt manufactures honestly.
    assert "general fabrication" in brief_to_prompt(brief)


def test_print3d_session_keeps_printability() -> None:
    brief = assemble_brief(_session("a 40 mm bracket", "print_3d"), PlannerState(), ready=True)
    assert brief.manufacturing_process == "fdm_3d_printing"
    assert "Must be 3D printable" in brief.constraints


def test_service_add_input_stores_fabrication(tmp_path: object) -> None:
    from forge_mind.intake.agent import PlannerAgent
    from forge_mind.intake.service import IntakeService
    from forge_mind.intake.session import InMemorySessionStore

    svc = IntakeService(store=InMemorySessionStore(), asset_dir=tmp_path, agent=PlannerAgent())  # type: ignore[arg-type]
    session = svc.create_session()
    svc.add_input(session.id, text="a thing", printable=False)
    assert svc.get_session(session.id).fabrication == "general"
    svc.add_input(session.id, printable=True)
    assert svc.get_session(session.id).fabrication == "print_3d"
    svc.add_input(session.id, text="more detail")  # printable not sent -> unchanged
    assert svc.get_session(session.id).fabrication == "print_3d"


# ----------------------------------------------------------------- label grounding ---


def test_ungrounded_type_never_renames_the_part() -> None:
    """The label-consistency bug: the closed-vocabulary classifier maps 'a hammer' to 'block';
    the part must keep the USER'S noun, and the prompt must not open with 'A block:'."""
    state = PlannerState(part_type="block")
    brief = assemble_brief(_session("a hammer with a curved claw"), state, ready=True)
    assert "hammer" in brief.part_name
    assert "block" not in brief.part_name
    prompt = brief_to_prompt(brief)
    assert not prompt.lower().startswith("a block:")
    assert prompt.startswith("a hammer with a curved claw")


def test_grounded_type_keeps_its_prefix_and_slug() -> None:
    """A type whose own words appear in the goal is evidence, not a guess -- behavior unchanged."""
    state = PlannerState(part_type="bracket")
    brief = assemble_brief(_session("a wall bracket for a shelf"), state, ready=True)
    assert brief.part_name == "bracket"
    assert brief_to_prompt(brief).startswith("A bracket: a wall bracket")


def test_standards_ground_their_types() -> None:
    """A NEMA mention genuinely IS a motor mount: the standards-derived type stays grounded even
    when the word 'motor' is absent from the goal."""
    state = PlannerState(
        part_type="motor_mount", known_standards=detect_standards("a NEMA-17 holder")
    )
    brief = assemble_brief(_session("a NEMA-17 holder"), state, ready=True)
    assert brief.part_name == "nema17_motor_mount"


def test_shape_primitive_stays_grounded() -> None:
    # The ADR-0028/0032 behavior: "a 1 m cube" classifies as cube and keeps it (the word appears).
    state = PlannerState(part_type="cube")
    brief = assemble_brief(_session("a 1 m cube"), state, ready=True)
    assert brief.part_name == "cube"


# ------------------------------------------------------------------ typed context ---


def test_brief_to_context_renders_typed_fields() -> None:
    state = PlannerState(
        part_type="motor_mount", known_standards=detect_standards("a NEMA-17 mount")
    )
    brief = assemble_brief(_session("a NEMA-17 mount", "general"), state, ready=True)
    ctx = brief_to_context(brief)
    assert ctx.startswith("STATED FACTS")
    assert "- object: a NEMA-17 mount" in ctx
    assert "general fabrication" in ctx
    assert "NEMA-17" in ctx  # the standard line
    assert "interface: nema17_face" in ctx
    # Dimensions carry provenance so the engine knows what is user-stated vs defaulted.
    assert "(source:" in ctx
