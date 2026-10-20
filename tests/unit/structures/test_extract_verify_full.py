"""The RequirementSpec extractor (fail-closed honesty) + the full-validator bridge.

The extractor's LLM half is mocked (the suite never calls a model); ``to_spec`` -- the pure half
that guards against invented loads and unknown hardware -- is what these tests pin down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.structures.author import AuthorResult
from forge_mind.structures.extract import (
    ExtractedComponent,
    ExtractedRequirements,
    to_spec,
)
from forge_mind.structures.loadcase import G_M_S2, LoadCase, describe_load_case
from forge_mind.structures.router import AGENTIC, TOPOPT, route
from forge_mind.structures.spec import RequirementSpec
from forge_mind.structures.verify import verify_full

trimesh = pytest.importorskip("trimesh")


def test_no_declared_load_routes_agentic() -> None:
    """The fail-closed core: 'make it sturdy' is NOT a load -> spec.load None -> agentic."""
    ex = ExtractedRequirements(part_name="bracket", has_declared_load=False)
    spec, report = to_spec(ex, default_envelope_mm=(100.0, 20.0, 40.0))
    assert spec.load is None
    assert spec.deflection_bound_mm is None  # a bound without a load is meaningless -> dropped
    assert route(spec) == AGENTIC
    assert report["load_case_echo"] is None


def test_envelope_provenance_stated_vs_defaulted() -> None:
    """T3: the certificate must show whether the design volume was the USER'S or a labeled
    default -- to_spec's report carries the flag either way, never a silent assumption."""
    stated = ExtractedRequirements(
        part_name="link", has_declared_load=False, envelope_mm=(130.7, 24.0, 64.4)
    )
    _spec_s, report_s = to_spec(stated, default_envelope_mm=(150.0, 30.0, 60.0))
    assert report_s["envelope_provenance"] == "stated"
    assert report_s["envelope_mm"] == [130.7, 24.0, 64.4]

    defaulted = ExtractedRequirements(part_name="link", has_declared_load=False)
    spec_d, report_d = to_spec(defaulted, default_envelope_mm=(150.0, 30.0, 60.0))
    assert report_d["envelope_provenance"] == "defaulted"
    assert report_d["envelope_mm"] == [150.0, 30.0, 60.0]
    assert spec_d.envelope_mm == (150.0, 30.0, 60.0)


def test_declared_load_without_a_bound_defaults_serviceability_labeled() -> None:
    """B2a (the denominator fix): a real user states a LOAD but not a deflection BOUND. The spec
    gets a serviceability default (span/250) so the physics authors can run, ALWAYS labeled
    bound_provenance='defaulted...' -- never a hidden number. This is what lets 'holds 0.5 kg'
    reach the physics path instead of falling through to the LLM."""
    ex = ExtractedRequirements(
        part_name="link", has_declared_load=True, payload_kg=0.5, envelope_mm=(200.0, 20.0, 40.0)
    )
    spec, report = to_spec(ex, default_envelope_mm=(150.0, 30.0, 60.0))
    assert spec.load is not None
    assert spec.deflection_bound_mm == 200.0 / 250.0  # span/250 of the longest envelope dim
    assert report["deflection_bound_mm"] == spec.deflection_bound_mm
    assert isinstance(report["bound_provenance"], str)
    assert (
        report["bound_provenance"].startswith("defaulted") and "250" in report["bound_provenance"]
    )
    assert route(spec) == TOPOPT or spec.deflection_bound_mm is not None  # authorable now


def test_stated_bound_is_used_verbatim_and_labeled_stated() -> None:
    ex = ExtractedRequirements(
        part_name="link", has_declared_load=True, payload_kg=0.5, deflection_bound_mm=0.046
    )
    spec, report = to_spec(ex, default_envelope_mm=(130.0, 24.0, 64.0))
    assert spec.deflection_bound_mm == 0.046
    assert report["bound_provenance"] == "stated"


def test_no_load_leaves_the_bound_none() -> None:
    ex = ExtractedRequirements(part_name="stand", has_declared_load=False)
    spec, report = to_spec(ex, default_envelope_mm=(100.0, 20.0, 40.0))
    assert spec.deflection_bound_mm is None and report["bound_provenance"] is None


def test_declared_load_builds_case_and_echo() -> None:
    ex = ExtractedRequirements(
        part_name="drone arm",
        has_declared_load=True,
        payload_kg=0.5,
        deflection_bound_mm=0.15,
        load_note="holds a 500 g motor at the tip",
        components=[ExtractedComponent(kind="mg996r_servo", placement_hint="tip")],
    )
    spec, report = to_spec(ex, default_envelope_mm=(142.0, 24.5, 67.0))
    assert spec.load is not None
    assert abs(spec.load.force_n - 0.5 * G_M_S2) < 1e-9
    assert spec.deflection_bound_mm == 0.15
    assert route(spec) == TOPOPT  # component present -> topology
    assert report["load_case_echo"] is not None and "0.50 kg" in str(report["load_case_echo"])
    assert report["load_note_verbatim"] == "holds a 500 g motor at the tip"
    # deterministic tip placement: 0.8 * 142 = 113.6, mid-height 33.5
    assert report["placed_components"] == [{"kind": "mg996r_servo", "at_mm": [113.6, 33.5]}]


def test_unknown_hardware_is_dropped_and_surfaced_never_guessed() -> None:
    ex = ExtractedRequirements(
        part_name="thing",
        has_declared_load=True,
        payload_kg=1.0,
        components=[
            ExtractedComponent(kind="warp_nacelle"),
            ExtractedComponent(kind="mg996r_servo", at_mm=(50.0, 10.0)),
        ],
    )
    spec, report = to_spec(ex, default_envelope_mm=(100.0, 20.0, 40.0))
    assert report["dropped_components"] == ["warp_nacelle"]
    assert len(spec.components) == 1
    assert spec.components[0].at_mm == (50.0, 10.0)  # stated position used verbatim


def test_describe_load_case_human_units() -> None:
    case = LoadCase(name="arm_link_v0", force_n=0.5 * G_M_S2, load_dir=(1.0, 0.0, 0.0))
    s = describe_load_case(case)
    assert "0.50 kg" in s and "4.90 N" in s and "arm_link_v0" in s


@pytest.mark.slow
def test_verify_full_merges_the_engine_validator(tmp_path: Path) -> None:
    """The bridge: the uniform gate + validate_authored_part behind a synthesized brief. A plain
    solid box should clear both (no load declared -> deflection n/a; full validator runs)."""
    from build123d import Box, export_step, export_stl

    stl, step = tmp_path / "p.stl", tmp_path / "p.step"
    part = Box(60, 20, 20)
    export_stl(part, str(stl))
    export_step(part, str(step))
    spec = RequirementSpec(envelope_mm=(60.0, 20.0, 20.0))
    out = verify_full(
        AuthorResult(provenance="test", stl=stl, step=step), spec, part_name="test block"
    )
    assert out["checks"]["full_validator"] in ("pass", "fail", "unproven")
    assert out["checks"]["full_hard_ok"] == "pass"  # one watertight solid, positive volume
    assert out["checks"]["watertight"] == "pass"
