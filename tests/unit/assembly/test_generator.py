"""AssemblyGraphGenerator converter: LLM counts -> derived interleave (no LLM, no API).

Covers the deterministic half of the layer-4 front -- the part the live probe
showed is load-bearing. The LLM picks knuckle COUNTS + topology; the converter
derives centered interleaved positions, clearance notches, and the pin (no radius).
The live LLM path is exercised by the probe (see bench-004 README), not in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.assembly.generator import (
    LlmAssembly,
    LlmFlatLink,
    LlmJoint,
    LlmKnuckledPlate,
    LlmRod,
    _interleave_positions,
    to_assembly_graph,
)
from forge_mind.assembly.plan import FunctionalPlan, PlanPart, PlanRelation
from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import compile_assembly
from forge_mind.validator.assembly_checks import validate_assembly

# Each case compiles + validates a full multi-knuckle hinge / linkage assembly (~50s total) --
# the second-heaviest unit module. Marked slow to keep the inner loop fast; still in the full
# gate (`-m "not llm"`) and CI.
pytestmark = pytest.mark.slow


def _hinge(count_a: int, count_b: int) -> LlmAssembly:
    def plate(pid: str, n: int) -> LlmKnuckledPlate:
        return LlmKnuckledPlate(
            id=pid,
            knuckle_count=n,
            length_x=40.0,
            width_y=18.0,
            thickness=4.0,
            knuckle_outer_r=5.0,
            bore_r=2.6,
            knuckle_width=8.0,
        )

    return LlmAssembly(
        plan=FunctionalPlan(
            summary="two-leaf hinge",
            parts=[
                PlanPart(id="leaf_a", role="fixed_leaf"),
                PlanPart(id="leaf_b", role="moving_leaf"),
                PlanPart(id="pin", role="pivot_pin"),
            ],
            relations=[PlanRelation(type="rotates_about", moving="leaf_b", reference="leaf_a")],
        ),
        knuckled_plates=[plate("leaf_a", count_a), plate("leaf_b", count_b)],
        rods=[LlmRod(id="pin")],
        joints=[
            LlmJoint(type="coaxial_insert", a="leaf_a.bore", b="pin.shaft"),
            LlmJoint(type="revolute", a="leaf_a.bore", b="leaf_b.bore", range_deg=[0.0, 180.0]),
        ],
        root="leaf_a",
    )


def test_interleave_larger_leaf_takes_outer_slots() -> None:
    pos_a, pos_b = _interleave_positions(2, 3, 10.0)
    assert len(pos_a) == 2 and len(pos_b) == 3
    allpos = pos_a + pos_b
    # The 3-count leaf owns both outer ends -> clean B,A,B,A,B alternation, centered.
    assert min(allpos) in pos_b and max(allpos) in pos_b
    assert abs(sum(allpos)) < 1e-9


def test_converter_derives_interleave_without_llm_coordinates() -> None:
    graph = to_assembly_graph(_hinge(2, 3))
    leaf_a = next(n for n in graph.nodes if n.id == "leaf_a")
    leaf_b = next(n for n in graph.nodes if n.id == "leaf_b")
    pin = next(n for n in graph.nodes if n.id == "pin")
    # Counts derive the knuckles; each leaf is notched once per PARTNER knuckle (room
    # for the partner's barrels). leafB's positions are negated for its 180-deg
    # placement, so compare counts, not ordered position lists.
    assert len(leaf_a.params["knuckle_centers_x"]) == 2
    assert len(leaf_b.params["knuckle_centers_x"]) == 3
    assert len(leaf_a.params["clearance_notches_x"]) == 3  # one per leafB knuckle
    assert len(leaf_b.params["clearance_notches_x"]) == 2  # one per leafA knuckle
    # The pin carries no radius: the solver derives it from the bore + fit.
    assert "radius" not in pin.params


@pytest.mark.slow
@pytest.mark.parametrize(
    ("count_a", "count_b"),
    # Unequal, EQUAL (2+2, 1+1), and |a-b|>1 (2+4) -- all interleave correctly now that
    # leafB's local positions are negated for its 180-deg placement (regression: equal
    # counts used to collide at rest).
    [(2, 1), (2, 3), (3, 2), (2, 2), (1, 1), (2, 4)],
)
def test_derived_hinge_validates_for_all_counts(count_a: int, count_b: int, tmp_path: Path) -> None:
    report = validate_assembly(
        compile_assembly(solve(to_assembly_graph(_hinge(count_a, count_b))), tmp_path)
    )
    assert report.passed, report.failed_check_ids()


def _linkage() -> LlmAssembly:
    def link(pid: str) -> LlmFlatLink:
        return LlmFlatLink(id=pid, length_x=40.0, width_y=14.0, thickness=4.0, bore_r=2.6)

    return LlmAssembly(
        plan=FunctionalPlan(
            summary="2-bar linkage",
            parts=[
                PlanPart(id="base", role="ground"),
                PlanPart(id="bar1", role="link"),
                PlanPart(id="bar2", role="link"),
            ],
            relations=[
                PlanRelation(type="rotates_about", moving="bar1", reference="base"),
                PlanRelation(type="rotates_about", moving="bar2", reference="bar1"),
            ],
        ),
        flat_links=[link("base"), link("bar1"), link("bar2")],
        rods=[LlmRod(id="pin1"), LlmRod(id="pin2")],
        joints=[
            LlmJoint(type="coaxial_insert", a="base.pivot_b", b="pin1.shaft"),
            LlmJoint(type="revolute", a="base.pivot_b", b="bar1.pivot_a", range_deg=[0.0, 90.0]),
            LlmJoint(type="coaxial_insert", a="bar1.pivot_b", b="pin2.shaft"),
            LlmJoint(type="revolute", a="bar1.pivot_b", b="bar2.pivot_a", range_deg=[0.0, 90.0]),
        ],
        root="base",
    )


def test_converter_coerces_interference_pin_to_running_fit() -> None:
    # "interlocking knuckles" lures the LLM into an interference coaxial fit, which
    # sizes the derived pin fatter than its bore -> assembly.pin_clearance fails. A
    # functional pivot pin must turn, so the fit class is derived, not the LLM's to
    # pick: interference on a rod-in-bore insert is coerced to running_fit.
    emitted = _hinge(2, 1)
    bad = emitted.model_copy(
        update={
            "joints": [
                LlmJoint(
                    type="coaxial_insert",
                    a="leaf_a.bore",
                    b="pin.shaft",
                    clearance_class="interference",
                ),
                LlmJoint(type="revolute", a="leaf_a.bore", b="leaf_b.bore", range_deg=[0.0, 180.0]),
            ]
        }
    )
    graph = to_assembly_graph(bad)
    insert = next(j for j in graph.joints if j.type.value == "coaxial_insert")
    assert insert.clearance_class == "running_fit"


def test_converter_infers_align_mate_for_chained_links() -> None:
    # A revolute between two flat_links is a chain pivot -> parallel "align" stacking
    # (a revolute between knuckled_plates stays "oppose"). The LLM doesn't set the mate.
    graph = to_assembly_graph(_linkage())
    revolutes = [j for j in graph.joints if j.type.value == "revolute"]
    assert revolutes and all(j.mate == "align" for j in revolutes)


@pytest.mark.slow
def test_derived_linkage_validates(tmp_path: Path) -> None:
    report = validate_assembly(compile_assembly(solve(to_assembly_graph(_linkage())), tmp_path))
    assert report.passed, report.failed_check_ids()
