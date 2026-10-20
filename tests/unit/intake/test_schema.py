"""Question schema validation + DesignBrief shape."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge_mind.intake.schema import (
    CriticalDimension,
    DesignBrief,
    DimensionSource,
    Option,
    Question,
    QuestionType,
)


def test_multiple_choice_requires_options() -> None:
    with pytest.raises(ValidationError):
        Question(id="q", type=QuestionType.MULTIPLE_CHOICE, title="pick")


def test_dropdown_requires_options() -> None:
    with pytest.raises(ValidationError):
        Question(id="q", type=QuestionType.DROPDOWN, title="pick")


def test_number_min_greater_than_max_rejected() -> None:
    with pytest.raises(ValidationError):
        Question(id="q", type=QuestionType.NUMBER, title="thickness", min=12.0, max=2.0)


def test_valid_number_question_with_default() -> None:
    q = Question(
        id="q_plate_thickness",
        type=QuestionType.NUMBER,
        title="How thick?",
        required=False,
        unit="mm",
        min=2.0,
        max=12.0,
        step=0.5,
        default_value=5.0,
        affects=["strength", "printability"],
    )
    assert q.default_value == 5.0
    assert q.priority == 2  # default priority


def test_valid_multiple_choice() -> None:
    q = Question(
        id="q_mount",
        type=QuestionType.MULTIPLE_CHOICE,
        title="Attach to?",
        options=[
            Option(label="2020", value="2020_extrusion"),
            Option(label="Flat", value="flat_plate"),
        ],
        default_value="2020_extrusion",
    )
    assert len(q.options) == 2
    assert q.required is True


def test_critical_dimension_default_source_unknown() -> None:
    d = CriticalDimension(name="plate_thickness", value=5.0)
    assert d.unit == "mm"
    assert d.source == DimensionSource.UNKNOWN


def test_design_brief_round_trip() -> None:
    brief = DesignBrief(
        session_id="s1",
        part_name="nema17_2020_motor_mount",
        user_goal="Mount a NEMA-17 to a 2020 rail.",
        part_type="motor_mount",
        critical_dimensions=[
            CriticalDimension(name="plate_thickness", value=5.0, source=DimensionSource.DEFAULT)
        ],
        design_choices={"reinforced": True, "rounded_edges": True},
        ready_for_blueprint=True,
    )
    again = DesignBrief.model_validate(brief.model_dump(mode="json"))
    assert again == brief
    assert again.critical_dimensions[0].source == DimensionSource.DEFAULT


def test_image_region_select_requires_asset_id() -> None:
    with pytest.raises(ValidationError):
        Question(id="q_region", type=QuestionType.IMAGE_REGION_SELECT, title="point to it")


def test_valid_image_region_select() -> None:
    q = Question(
        id="q_region",
        type=QuestionType.IMAGE_REGION_SELECT,
        title="point to it",
        image_asset_id="img1",
    )
    assert q.image_asset_id == "img1"
