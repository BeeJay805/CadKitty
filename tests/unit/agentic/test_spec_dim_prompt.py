"""spec_dims -> author prompt rendering (the dead-carrier fix): the typed per-feature dims
the brief already produces must reach the author verbatim, HARD-worded, for every role."""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from forge_mind.agentic.brief import SpecDim, SpecFeature
from forge_mind.agentic.spec_dim_prompt import spec_dim_lines, spec_dim_prompt_enabled
from forge_mind.fabrication import FabricationMode


def _feature(
    dims: list[SpecDim], approx: str = "~60 long shell, 2.8 walls, 6 pin, 32 span, 45 deg draft"
) -> SpecFeature:
    return SpecFeature(
        name="arm_shell_loft",
        purpose="carries the shoulder torque",
        approx_dims_mm=approx,
        method="lofted shell between end profiles",
        feature_id="arm_shell_loft",
        spec_dims=dims,
    )


def test_the_measured_killer_case_renders() -> None:
    """H5B upper_arm_r's exact carrier: thickness 2.8 of 'shell wall' — died unseen."""
    f = _feature([SpecDim(role="thickness", value_mm=2.8, of="shell wall")])
    lines = spec_dim_lines(f)
    assert len(lines) == 1
    assert "thickness" in lines[0] and "2.8 mm" in lines[0] and "shell wall" in lines[0]
    assert "HARD" in lines[0]


def test_every_role_renders_not_just_thickness() -> None:
    f = _feature(
        [
            SpecDim(role="diameter", value_mm=6.0, of="pin journal"),
            SpecDim(role="length", value_mm=32.0, of="bearing span"),
        ]
    )
    lines = spec_dim_lines(f)
    assert len(lines) == 2
    assert any("diameter" in ln and "6 mm" in ln for ln in lines)
    assert any("length" in ln and "32 mm" in ln for ln in lines)


def test_angle_role_renders_degrees_not_mm() -> None:
    """SpecDim.value_mm is documented degrees for role=angle (review-caught)."""
    f = _feature([SpecDim(role="angle", value_mm=45.0, of="draft")])
    (line,) = spec_dim_lines(f)
    assert "45 deg" in line and "45 mm" not in line


def test_untraceable_spec_dim_is_never_promoted() -> None:
    """The 'second liar' law (review-caught): a spec_dim whose number is absent from the
    feature's own prose must not become a HARD instruction."""
    f = _feature([SpecDim(role="thickness", value_mm=50.0, of="shell wall")])  # prose has no 50
    assert spec_dim_lines(f) == []


def test_featureless_and_malformed_dims_render_nothing() -> None:
    assert spec_dim_lines(_feature([])) == []
    assert spec_dim_lines(object()) == []  # a plain Feature without spec_dims


def test_flag_default_off(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_SPEC_DIM_PROMPT", raising=False)
    assert spec_dim_prompt_enabled() is False
    monkeypatch.setenv("FORGE_SPEC_DIM_PROMPT", "1")
    assert spec_dim_prompt_enabled() is True


def _spec_brief() -> SpecBrief:  # noqa: F821 - forward name for readability
    from forge_mind.agentic.brief import SpecBrief

    return SpecBrief(
        object_name="upper arm shell",
        summary="tapered arm shell between shoulder and elbow",
        visual_description="smooth tapered limb shell",
        overall_envelope_mm="34 x 34 x 85 mm",
        min_wall_mm=2.8,
        material="PLA",
        target="cosmetic",
        print_orientation_note="upright",
        features=[
            _feature([SpecDim(role="thickness", value_mm=2.8, of="shell wall")], approx="2.8 walls")
        ],
    )


def test_material_line_states_the_density_only_when_the_registry_knows_it() -> None:
    """L103's dominant mass cause. The honesty guard matters as much as the lever:
    `resolve_material` silently falls back to PLA, so an unknown alloy must NOT be given a
    confident density — a wrong density is worse than the silence it replaces."""
    from forge_mind.agentic.spec_dim_prompt import material_line

    steel = material_line("steel")
    assert steel is not None and "7.85 g/cm^3" in steel and "volume = mass / density" in steel
    alu = material_line("aluminium")  # registry alias
    assert alu is not None and "2.7 g/cm^3" in alu

    unknown = material_line("6061-T6 aerospace billet")
    assert unknown is not None
    assert "do NOT assume one" in unknown and "g/cm^3" not in unknown  # never a guessed density
    assert material_line(None) is None and material_line("  ") is None


def test_author_prompt_renders_the_material_only_when_armed(monkeypatch) -> None:
    from forge_mind.agentic.author import _brief_to_user

    brief = _spec_brief()
    monkeypatch.delenv("FORGE_MATERIAL_PROMPT", raising=False)
    off = _brief_to_user(brief)
    assert "Material:" not in off
    monkeypatch.setenv("FORGE_MATERIAL_PROMPT", "1")
    on = _brief_to_user(brief)
    assert "Material: PLA" in on and "1.24 g/cm^3" in on
    # the flag adds exactly one line and moves nothing else
    assert [ln for ln in on.splitlines() if "Material:" not in ln] == off.splitlines()


def test_target_mass_is_a_typed_field_that_survives_research() -> None:
    """L103's other half: a stated target mass used to reach the author only if the researcher
    kept it in prose (ppp0103 did and scored -0%; ppp0106 did not and scored +187%)."""
    from forge_mind.agentic.brief import DesignBrief

    assert "target_mass_g" in DesignBrief.model_fields
    brief = _spec_brief().model_copy(update={"target_mass_g": 96.13})
    round_tripped = type(brief).model_validate_json(brief.model_dump_json())
    assert round_tripped.target_mass_g == 96.13
    # optional by default, so every legacy brief still validates
    assert _spec_brief().target_mass_g is None
    # and never zero/negative (a "0 g target" is a researcher error, not a spec)
    payload = _spec_brief().model_dump()
    with pytest.raises(ValidationError):
        type(brief).model_validate({**payload, "target_mass_g": 0.0})
    with pytest.raises(ValidationError):
        type(brief).model_validate({**payload, "target_mass_g": -5.0})


def test_target_volume_line_converts_mass_to_a_buildable_volume() -> None:
    from forge_mind.agentic.spec_dim_prompt import target_volume_line

    line = target_volume_line(96.13, "aluminum")  # 96.13 / 2.70 = 35.6 cm^3
    assert line is not None and "35.6 cm^3" in line and "35604 mm^3" in line
    # small targets must not print two DIFFERENT targets (review-caught: 1 g of steel showed
    # "0.1 cm^3" beside "127 mm^3", and 0.2 g showed "0.0 cm^3" beside "25 mm^3")
    tiny = target_volume_line(0.2, "steel")
    assert tiny is not None and "0.0 cm^3" not in tiny and "0.0255 cm^3" in tiny
    assert "96.13 g" in line and "2.7 g/cm^3" in line
    # review-caught honesty: density x volume is a SOLID-equivalent basis, so the line must not
    # promise the printed mass (partial infill weighs less)
    assert "SOLID" in line and "infill" in line

    steel = target_volume_line(310.0, "steel")  # 310 / 7.85 = 39.5 cm^3
    assert steel is not None and "39.5 cm^3" in steel


def test_the_researcher_is_told_to_convert_units_to_grams() -> None:
    """Review-caught: 'copy it verbatim' plus a grams-typed field turns '2 kg' into 2 g."""
    from forge_mind.agentic.brief import DesignBrief

    desc = DesignBrief.model_fields["target_mass_g"].description or ""
    assert "GRAMS" in desc and "CONVERT" in desc
    assert "2 kg" in desc and "2000" in desc  # the conversion is shown, not just demanded
    # review-caught: the scalar carries no equality-vs-bound marker, so a CAP must stay out of
    # it -- "under 100 g" is a limit, and rendering it as a build-to target would invert intent
    assert "BOUND" in desc and "under 100 g" in desc


def test_the_cache_key_separates_different_target_masses() -> None:
    """Review-caught: a pre-researched brief can carry a mass the prompt never states, and the
    cache lookup returns before the brief is rendered -- so without this the second request is
    served the first mass's geometry."""
    from forge_mind.agentic.build_cache import compute_key

    def key(mass: float | None) -> str:
        return compute_key(
            "a machined metal bracket",
            material=None,
            n_drafts=1,
            max_iterations=3,
            fabrication_mode=FabricationMode.GENERAL,
            author_model="sonnet",
            critic_model=None,
            brief_model=None,
            target_mass_g=mass,
        )

    assert key(96.13) != key(310.0) != key(None)
    assert key(96.13) == key(96.13)  # and it stays stable for the same target


def test_canary_staged_mass_reaches_the_calls_that_can_act_on_it() -> None:
    """CANARY, not a behavioral test (stated plainly): `staged_initial` drives sandbox builds
    and LLM calls, so exercising it here is impractical -- this trips if the routing changes.

    The rule, arrived at over three review rounds: the BASE keeps the target (it authors the
    body-class features that ARE the bulk, and `_stage_feedback` forbids later stages from
    resizing existing geometry, so nothing else can act on a mass); MIDDLE stages have it
    cleared (one chasing the whole volume while adding a single feature oversizes itself); the
    FINAL stage keeps it to true the part up."""
    src = pathlib.Path(__file__).resolve().parents[3] / "src/forge_mind/agentic/staged.py"
    text = src.read_text(encoding="utf-8")
    base = text.split("base_brief = brief.model_copy(", 1)[1].split(")", 1)[0]
    assert "target_mass_g" not in base  # the base is NOT stripped
    stage = text.split("stage_brief = (", 1)[1].split("\n        for _attempt", 1)[0]
    assert '"target_mass_g": None' in stage and "ix == len(stage_feats)" in stage


def test_target_volume_line_refuses_to_guess() -> None:
    """No stated mass, no material, or an unknown density -> no volume claim. A guessed density
    would yield a confidently wrong volume, which is worse than no target."""
    from forge_mind.agentic.spec_dim_prompt import target_volume_line

    assert target_volume_line(None, "steel") is None
    assert target_volume_line(0.0, "steel") is None
    assert target_volume_line(100.0, None) is None
    unknown = target_volume_line(100.0, "6061-T6 aerospace billet")
    assert unknown is not None and "cm^3" not in unknown and "do NOT derive" in unknown


def test_author_prompt_carries_both_mass_halves_only_when_armed(monkeypatch) -> None:
    from forge_mind.agentic.author import _brief_to_user

    brief = _spec_brief().model_copy(update={"target_mass_g": 62.0, "material": "PLA"})
    monkeypatch.delenv("FORGE_MATERIAL_PROMPT", raising=False)
    assert "TARGET MASS" not in _brief_to_user(brief)
    monkeypatch.setenv("FORGE_MATERIAL_PROMPT", "1")
    on = _brief_to_user(brief)
    assert "Material: PLA" in on and "TARGET MASS 62 g" in on and "50.0 cm^3" in on


def test_prompt_shaping_flags_partition_the_build_cache(monkeypatch) -> None:
    """Review-caught: a cache hit returns BEFORE authoring, so an unpartitioned prompt flag
    lets an A/B be served pre-arm geometry and scored as if the arm ran. (The spec_dims flag
    shipped at 50e134c without this partition.)"""
    from forge_mind.agentic.build_cache import compute_key

    def key() -> str:
        return compute_key(
            "a bracket",
            material=None,
            n_drafts=1,
            max_iterations=3,
            fabrication_mode=FabricationMode.PRINT_3D,
            author_model="sonnet",
            critic_model=None,
            brief_model=None,
        )

    for flag in ("FORGE_SPEC_DIM_PROMPT", "FORGE_MATERIAL_PROMPT"):
        monkeypatch.delenv(flag, raising=False)
        off = key()
        monkeypatch.setenv(flag, "1")
        assert key() != off, f"{flag} must partition the cache"
        # ...and through the flag's OWN predicate, not a bare == "1" (review-caught: the
        # helpers accept true/yes/padding, so a hand-typed value would arm the prompt while
        # leaving the key unpartitioned -- serving off-mode geometry to an armed A/B).
        monkeypatch.setenv(flag, " TRUE ")
        assert key() != off, f"{flag}=TRUE must partition the cache too"
        monkeypatch.delenv(flag, raising=False)
        assert key() == off  # and flipping back returns to the same partition


def test_prompt_shaping_flags_partition_the_resume_checkpoint_key(monkeypatch) -> None:
    """Same hazard one layer down: FORGE_RESUME_PARTS is default-ON, so a functional re-run
    loads part checkpoints BEFORE the pipeline. Legacy keys must stay identical when off."""
    from forge_mind.agentic.functional import PartPlan, _part_key
    from forge_mind.materials import MATERIALS

    part = PartPlan(name="bracket", prompt="an L bracket")
    mat = MATERIALS["pla"]
    for flag in ("FORGE_SPEC_DIM_PROMPT", "FORGE_MATERIAL_PROMPT"):
        monkeypatch.delenv(flag, raising=False)
    legacy = _part_key(part, mat)
    for flag in ("FORGE_SPEC_DIM_PROMPT", "FORGE_MATERIAL_PROMPT"):
        monkeypatch.setenv(flag, "1")
        assert _part_key(part, mat) != legacy, f"{flag} must partition resume checkpoints"
        monkeypatch.delenv(flag, raising=False)
    assert _part_key(part, mat) == legacy  # flag-off keys are byte-identical to legacy


def test_author_prompt_renders_spec_lines_only_when_armed(monkeypatch) -> None:
    """The wiring contract: byte-identical prompt with the flag off; HARD lines under the
    feature with it on (the dark-introduction law)."""
    from forge_mind.agentic.author import _brief_to_user

    brief = _spec_brief()
    monkeypatch.delenv("FORGE_SPEC_DIM_PROMPT", raising=False)
    off = _brief_to_user(brief)
    assert "SPEC thickness" not in off
    monkeypatch.setenv("FORGE_SPEC_DIM_PROMPT", "1")
    on = _brief_to_user(brief)
    assert "SPEC thickness of the shell wall: 2.8 mm" in on
    assert on.replace("\n" + "\n".join(spec_dim_lines(brief.features[0])), "") == off
