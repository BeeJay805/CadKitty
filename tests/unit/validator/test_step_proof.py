"""STEP roundtrip integrity proof (ISO 10303-49 spirit): the exported STEP must re-import to the SAME
solid (volume / area / centroid), and the file must BE a real named B-rep (ADR-0037 rung 2) -- a
recognised AP schema, analytic B-rep entities, a product named after the part, topology roundtrip.
The proof is REPORTED, never gates, and never raises."""

from __future__ import annotations

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.run import validate
from forge_mind.validator.step_proof import (
    StepFileFacts,
    inspect_step_text,
    verify_step_roundtrip,
)


def _program() -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="plate_one_hole"),
        operations=[
            Box(op_id="plate", width=40.0, depth=20.0, height=4.0, position=Vec3(z=2.0)),
            Cylinder(
                op_id="h",
                mode=OpMode.SUBTRACT,
                radius=1.6,
                height=4.0,
                position=Vec3(x=-10.0, z=2.0),
            ),
        ],
    )


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):  # type: ignore[no-untyped-def]
    program = _program()
    model = compile_program(program, tmp_path_factory.mktemp("step_proof"))
    return model, program


def test_roundtrip_matches_for_a_clean_part(compiled) -> None:  # type: ignore[no-untyped-def]
    model, _ = compiled
    proof = verify_step_roundtrip(model.part, model.step_path)
    assert proof is not None
    assert proof.matches, f"STEP did not round-trip: {proof}"
    assert proof.volume_rel_err < 1e-3
    assert proof.area_rel_err < 1e-3
    assert proof.centroid_dist_mm < 1e-2
    assert proof.volume_mm3 > 0.0


def test_missing_step_returns_none(compiled, tmp_path) -> None:  # type: ignore[no-untyped-def]
    model, _ = compiled
    assert verify_step_roundtrip(model.part, tmp_path / "does_not_exist.step") is None


def test_validate_attaches_the_step_proof(compiled) -> None:  # type: ignore[no-untyped-def]
    model, program = compiled
    report = validate(model, program)
    assert report.step_proof is not None
    assert report.step_proof.matches


# ---------------------------------------------------------------------------
# Named-B-rep fidelity (ADR-0037 rung 2).
# ---------------------------------------------------------------------------


def test_export_is_a_named_brep(compiled) -> None:  # type: ignore[no-untyped-def]
    """The shipped STEP is real analytic CAD: AP214, B-rep entities, no tessellation, the
    PRODUCT named after the part, and the topology round-trips exactly."""
    model, _ = compiled
    proof = verify_step_roundtrip(model.part, model.step_path)
    assert proof is not None and proof.matches
    assert proof.schema_id == "AP214"
    assert proof.is_brep is True
    assert proof.tessellated is False
    assert proof.product_name == "plate_one_hole"  # the engineer's CAD tree shows the part
    assert proof.named is True
    assert proof.solid_count == 1
    assert proof.face_count == len(model.part.faces())
    assert proof.topology_matches is True


def test_compile_labels_the_body(compiled) -> None:  # type: ignore[no-untyped-def]
    model, program = compiled
    assert model.part.label == program.metadata.name


def test_inspect_step_text_classifies_schemas_and_naming() -> None:
    named_brep = (
        "ISO-10303-21;\nHEADER;\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n"
        "#10 = PRODUCT('my_bracket','my_bracket','',(#2));\n"
        "#20 = MANIFOLD_SOLID_BREP('',#21);\n#22 = ADVANCED_FACE('',(#23),#24,.T.);\n"
        "ENDSEC;\nEND-ISO-10303-21;\n"
    )
    facts = inspect_step_text(named_brep)
    assert facts == StepFileFacts(
        schema_id="AP214", is_brep=True, tessellated=False, product_name="my_bracket", named=True
    )

    unnamed = named_brep.replace("'my_bracket','my_bracket'", "'SOLID',''")
    assert inspect_step_text(unnamed).named is False

    ap242 = named_brep.replace(
        "'AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'",
        "'AP242_MANAGED_MODEL_BASED_3D_ENGINEERING { 1 0 10303 442 1 1 4 }'",
    )
    assert inspect_step_text(ap242).schema_id == "AP242"


def test_inspect_step_text_flags_a_tessellated_lump() -> None:
    lump = (
        "ISO-10303-21;\nHEADER;\n"
        "FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING { 1 0 10303 442 1 1 4 }'));\n"
        "ENDSEC;\nDATA;\n#10 = PRODUCT('SOLID','SOLID','',(#2));\n"
        "#30 = TESSELLATED_SHAPE_REPRESENTATION('',(#31),#32);\n"
        "#31 = TRIANGULATED_FACE('',#33,3,(1.0),$,((1,2,3)));\n"
        "ENDSEC;\nEND-ISO-10303-21;\n"
    )
    facts = inspect_step_text(lump)
    assert facts.is_brep is False
    assert facts.tessellated is True
    assert facts.named is False


def test_lump_census_demotes_matches_even_when_volume_agrees(
    compiled, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """No-false-PASS for the proof itself: a volume-matching file whose census says
    'tessellated, no B-rep' must NOT read as a faithful STEP."""
    import forge_mind.validator.step_proof as sp

    model, _ = compiled
    monkeypatch.setattr(
        sp,
        "inspect_step_text",
        lambda text: StepFileFacts(
            schema_id="AP242", is_brep=False, tessellated=True, product_name="SOLID", named=False
        ),
    )
    proof = verify_step_roundtrip(model.part, model.step_path)
    assert proof is not None
    assert proof.volume_rel_err < 1e-3  # layer 1 (mass props) still agrees...
    assert proof.matches is False  # ...but the lump census demotes the headline
    assert proof.is_brep is False and proof.tessellated is True


def test_census_failure_never_demotes(compiled, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """An unreadable census leaves the fidelity fields None and layer 1 stands alone --
    a false refusal here would be worse than an honest 'couldn't prove it'."""
    import forge_mind.validator.step_proof as sp

    model, _ = compiled

    def _boom(text: str) -> StepFileFacts:
        raise RuntimeError("census exploded")

    monkeypatch.setattr(sp, "inspect_step_text", _boom)
    proof = verify_step_roundtrip(model.part, model.step_path)
    assert proof is not None
    assert proof.matches is True
    assert proof.schema_id is None and proof.is_brep is None and proof.named is None
