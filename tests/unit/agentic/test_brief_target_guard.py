"""The lone-graspable downgrade guard (the knob misroute): a graspable object that IS the whole
request must stay ONE body, while genuinely multi-part objects keep target='functional'. Measured:
the bench's 25mm knob routed to the multi-part assembler in 2 of 3 trials and went 0/3."""

from __future__ import annotations

from forge_mind.agentic.brief import BriefFeature, DesignBrief, _downgrade_lone_graspable


def _brief(name: str, target: str, feature_names: list[str]) -> DesignBrief:
    return DesignBrief(
        object_name=name,
        summary="x",
        visual_description="x",
        overall_envelope_mm="25 x 25 x 20",
        features=[
            BriefFeature(name=fn, purpose="p", approx_dims_mm="10", method="revolve")
            for fn in feature_names
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat down",
        target=target,  # type: ignore[arg-type]
    )


def test_lone_knob_is_downgraded_to_cosmetic() -> None:
    # the measured misroute: object IS the graspable thing, no second part anywhere
    brief = _brief("control knob", "functional", ["d-shaft bore", "knurled rim", "knob body"])
    assert _downgrade_lone_graspable(brief).target == "cosmetic"


def test_hammer_stays_functional() -> None:
    # a real multi-part tool: the NAME is not graspable (its handle is just one feature)
    brief = _brief("claw hammer", "functional", ["striking face", "claw", "handle"])
    assert _downgrade_lone_graspable(brief).target == "functional"


def test_pull_with_backplate_stays_functional() -> None:
    # graspable name BUT a genuine second part it mounts on -> the head+handle structure is real
    brief = _brief("door pull", "functional", ["grip loop", "backplate"])
    assert _downgrade_lone_graspable(brief).target == "functional"


def test_cosmetic_is_untouched() -> None:
    brief = _brief("control knob", "cosmetic", ["d-shaft bore", "knurled rim"])
    assert _downgrade_lone_graspable(brief).target == "cosmetic"
