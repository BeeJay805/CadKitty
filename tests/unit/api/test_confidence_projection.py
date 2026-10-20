"""ADR-0041 projection: the confidence block rides GenerateOk for the agentic path,
derived from the result's evidence (provenance + gates + judge). Money-free."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from forge_mind.agentic.provenance import DimensionProvenance, EstimateBasis, Tier
from forge_mind.api.core import _agentic_confidence


def _result(
    *,
    records: list[DimensionProvenance],
    geometry_passed: bool,
    hard_ok: bool,
    accepted: bool | None,
    judge_conf: str | None,
    name: str,
) -> Any:
    verdict = (
        SimpleNamespace(is_the_object=accepted, confidence=judge_conf)
        if accepted is not None
        else None
    )
    return SimpleNamespace(
        provenance=records,
        verdict=verdict,
        loop=SimpleNamespace(
            geometry_passed=geometry_passed,
            validation=SimpleNamespace(hard_ok=hard_ok),
            brief=SimpleNamespace(object_name=name, summary=name),
        ),
    )


def _crank_records() -> list[DimensionProvenance]:
    return [
        DimensionProvenance(
            feature="main journals",
            dims_text="52mm dia",
            tier=Tier.ESTIMATED,
            mechanism="default_estimated",
            estimate_basis=EstimateBasis(method="anchor_scaled", anchor="1NZ class"),
        )
    ]


def test_agentic_confidence_view_carries_dimensions_and_label() -> None:
    res = _result(
        records=_crank_records(),
        geometry_passed=True,
        hard_ok=True,
        accepted=True,
        judge_conf="high",
        name="automotive crankshaft",
    )
    view = _agentic_confidence(res, "print_3d")
    assert view.basis == "conceptual"
    assert view.overall == "medium"
    assert view.manufacturability == "printable_as_model"
    assert view.dimensions and view.dimensions[0].feature == "main journals"
    assert view.dimensions[0].tier == "estimated"
    assert view.verify_dimensions == ["main journals"]
    assert "rotating service" in " ".join(view.not_fit_for)


def test_agentic_confidence_caveats_path_caps_medium() -> None:
    res = _result(
        records=[
            DimensionProvenance(
                feature="holes",
                dims_text="4x M3",
                tier=Tier.VERIFIED,
                mechanism="registry_resolution",
                citation="ISO 273",
            )
        ],
        geometry_passed=False,
        hard_ok=True,  # coherent solid, a tier flagged it -> gates "warn"
        accepted=True,
        judge_conf="high",
        name="bracket",
    )
    view = _agentic_confidence(res, "print_3d")
    assert view.overall == "medium"
    assert view.basis == "validated"  # the dims ARE grounded; the gates capped confidence
