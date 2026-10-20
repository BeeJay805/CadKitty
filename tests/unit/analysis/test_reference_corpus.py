"""The frozen reference corpus: dimensionally-true ground truth for shape scoring.

Founder direction 2026-07-25: "build the reference corpus first" -- the missing half of CADSmith's
method. Their IoU/Chamfer numbers only exist because Text2CAD-Bench ships reference models; we had
the metric code (`analysis/shape_similarity.py`, written and never wired) and no references.

The references are PARAMETRIC (archetype + exact mm params compiled by the engine's own builders),
not fetched artist meshes, because `shape_similarity` works in absolute millimetres: a Sketchfab
model has arbitrary scale, is not watertight, and is not reproducible across runs.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import trimesh

from forge_mind.analysis.reference_corpus import (
    compile_reference_corpus,
    compile_reference_part,
    corpus_manifest,
    load_reference_spec,
    score_against_reference,
)

_CORPUS = Path(__file__).resolve().parents[3] / "benchmarks" / "reference_corpus_v1"
_SPEC = _CORPUS / "reference.parts.json"

# Measured 2026-07-25 on the frozen spec. Exact, not approximate: these archetypes are
# deterministic, so a change here means the builders moved and the corpus must be re-frozen.
_EXPECTED_EXTENTS = {
    "pin": (25.0, 5.0, 5.0),
    "bushing": (8.0, 8.0, 8.0),
    "link": (40.0, 12.0, 4.0),
    "hinge_leaf": (40.0, 23.0, 10.0),
    "arm": (42.0, 14.0, 8.0),
}


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    return compile_reference_corpus(_SPEC, tmp_path_factory.mktemp("refcorpus"))


def test_the_frozen_spec_is_complete_and_compiles(parts) -> None:
    """Every node builds. THE REASON THIS TEST EXISTS: the two pre-existing
    `benchmarks/*/reference.assembly.json` artifacts do NOT compile -- their `rod` node declares
    only `length_x` and omits the required `radius`, so they were stale as well as unconsumed.
    A reference corpus that cannot be rebuilt is not frozen, it is folklore."""
    assert len(parts) == 5
    assert {p.node_id for p in parts} == set(_EXPECTED_EXTENTS)


def test_every_reference_is_watertight_and_dimensionally_exact(parts) -> None:
    """Watertight is required for volumetric IoU; exact mm is required for Chamfer to mean
    anything (the metric is deliberately NOT scale-normalized)."""
    for p in parts:
        mesh = trimesh.load(str(p.stl), force="mesh")
        assert mesh.is_watertight, f"{p.node_id} is not watertight -- IoU would be undefined"
        assert mesh.volume > 0
        expected = _EXPECTED_EXTENTS[p.node_id]
        for got, want in zip(mesh.extents, expected, strict=True):
            assert got == pytest.approx(want, abs=0.01), (
                f"{p.node_id}: {mesh.extents} != {expected}"
            )


def test_a_reference_scores_perfectly_against_itself(parts) -> None:
    """End-to-end wiring check: identity must be the metric's fixed point, or every other number
    it reports is uninterpretable."""
    part = next(p for p in parts if p.node_id == "hinge_leaf")
    score = score_against_reference(part.stl, part)
    assert score.volumetric_iou == pytest.approx(1.0)
    assert score.f1_at_threshold == pytest.approx(1.0)
    assert score.chamfer_distance_mm == pytest.approx(0.0, abs=1e-6)


def test_the_metric_rejects_a_featureless_approximation(parts) -> None:
    """CALIBRATION, and the reason this corpus is worth having: the previous shape metric
    saturated at 0.85-0.95 for every part ("a box and a sphere score similar"), so no lever could
    be measured. On FEATURED parts this triple has real dynamic range -- a solid bounding box of
    the hinge leaf scores far from 1.

    See test_the_metric_is_weak_on_simple_plates for where it does NOT, which is pinned so the
    limitation cannot be forgotten.
    """
    part = next(p for p in parts if p.node_id == "hinge_leaf")
    ref = trimesh.load(str(part.stl), force="mesh")
    box = trimesh.creation.box(extents=ref.extents)
    box.apply_translation(ref.bounds.mean(axis=0))
    box_stl = part.stl.parent / "hinge_leaf_bbox.stl"
    box.export(str(box_stl))

    score = score_against_reference(box_stl, part)
    assert score.volumetric_iou < 0.5, score
    assert score.f1_at_threshold < 0.6, score
    assert score.chamfer_distance_mm > 1.0, score


def test_the_metric_is_weak_on_simple_plates(parts) -> None:
    """THE HONEST LIMITATION, pinned so nobody quotes this corpus as a general shape gate.

    A solid plate with NO holes scores IoU ~0.90 / F1 ~0.96 / Chamfer ~0.08 mm against the same
    plate WITH two bores -- the holes are too small a volume fraction for IoU to notice. So on
    L2 plate-with-holes parts (the commonest real shape) this triple is nearly blind to missing
    features, and feature-presence checks remain the gate that catches them.

    If this assertion ever FAILS because the scores got worse, the metric improved -- re-read it.
    """
    part = next(p for p in parts if p.node_id == "link")
    ref = trimesh.load(str(part.stl), force="mesh")
    box = trimesh.creation.box(extents=ref.extents)
    box.apply_translation(ref.bounds.mean(axis=0))
    box_stl = part.stl.parent / "link_bbox.stl"
    box.export(str(box_stl))

    score = score_against_reference(box_stl, part)
    assert score.volumetric_iou > 0.85, "a featureless plate still scores high -- the limitation"
    assert score.f1_at_threshold > 0.90, "F1@1mm is the weakest of the three signals"


def test_the_manifest_hashes_inputs_and_expanded_program_but_not_the_mesh(parts) -> None:
    """The frozen record carries BOTH hashes and the tier. Mesh bytes legitimately change when the
    CAD kernel is upgraded, and that must not read as a corpus change -- but a change in what the
    builder actually emits must."""
    manifest = corpus_manifest(parts)
    assert [p["id"] for p in manifest["parts"]] == sorted(_EXPECTED_EXTENTS)
    for entry in manifest["parts"]:
        assert len(entry["spec_sha256"]) == 64
        assert len(entry["program_sha256"]) == 64
        assert len(entry["geometry_sha256"]) == 64
        assert entry["tier"], "tier must ride into the manifest -- the corpus is read per-tier"
    # Deterministic: same spec, same hashes, independent of where it was compiled.
    assert corpus_manifest(parts) == manifest


def test_a_builder_change_that_preserves_inputs_still_breaks_the_freeze(parts) -> None:
    """THE HOLE THIS CLOSES. `spec_sha256` binds only the authored inputs, so a change to an
    archetype BUILDER slips straight past it: if `flat_link` stopped cutting its bores, archetype
    and params would be byte-identical, extents and watertightness unchanged, and every
    longitudinal score would silently use a different criterion.

    Simulated by dropping an operation from the expanded program -- exactly what such a builder
    regression would produce. `spec_sha256` cannot see it; `program_sha256` must.
    """
    link = next(p for p in parts if p.node_id == "link")
    assert len(link.program.operations) > 1, "fixture premise: the link has features to lose"

    debored = link.program.model_copy(update={"operations": link.program.operations[:1]})
    regressed = replace(link, program=debored)

    assert regressed.spec_sha256 == link.spec_sha256, "inputs unchanged -- the blind spot"
    assert regressed.program_sha256 != link.program_sha256, "the expanded program must be bound"


def test_tier_rides_through_compilation(parts) -> None:
    """The doc prescribes per-tier reporting; dropping tier on compile would force every caller to
    re-parse the spec to do it."""
    assert {p.node_id: p.tier for p in parts} == {
        "pin": "L1_primitive",
        "bushing": "L1_primitive",
        "link": "L2_plate_with_holes",
        "hinge_leaf": "L3_interleaved_features",
        "arm": "L4_hub_and_arm",
    }


def test_the_committed_manifest_matches_the_committed_spec(parts) -> None:
    """The freeze itself: if the spec is edited without re-freezing, this fails. That is the whole
    point of pinning -- a corpus that drifts silently cannot support a comparison across runs."""
    frozen = json.loads((_CORPUS / "MANIFEST.sha256.json").read_text(encoding="utf-8"))
    assert corpus_manifest(parts) == frozen, "spec changed without re-freezing the manifest"


def test_an_unknown_archetype_raises_rather_than_shrinking_the_corpus(tmp_path) -> None:
    """Silently dropping an unbuildable reference would shrink the corpus AND flatter the score."""
    with pytest.raises(ValueError, match="unknown archetype"):
        compile_reference_part("not_a_real_archetype", "x", {}, tmp_path)


def test_spec_loading_rejects_an_empty_node_list(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no 'nodes'"):
        load_reference_spec(bad)


def test_the_program_hash_ignores_fields_the_compiler_never_reads(parts) -> None:
    """The freeze must not cry wolf. Hashing the WHOLE program would drag in
    metadata.description, interfaces and hardware_features -- editing a description would report
    corpus drift and force a pointless re-freeze. A freeze that fires on non-changes gets ignored,
    which is worse than one gap."""
    link = next(p for p in parts if p.node_id == "link")
    renamed = replace(
        link,
        program=link.program.model_copy(
            update={
                "metadata": link.program.metadata.model_copy(update={"description": "reworded"})
            }
        ),
    )
    assert renamed.program_sha256 == link.program_sha256, "a description edit is not corpus drift"


def test_the_geometry_hash_is_read_from_the_brep_not_the_stl(parts) -> None:
    """The third hash covers what the first two cannot: a COMPILER semantics change leaves the
    program identical and the geometry different.

    IT MUST COME FROM THE B-REP. A first version hashed the tessellated mesh and claimed to survive
    re-tessellation; review measured that it does not -- the same rod solid exported at fine vs
    default tolerance gives 490.855 vs 490.670 mm^3 and 5.000 vs 4.998 mm, both far above the 1e-3
    quantum, so a mesh-derived hash fires on tessellation-only changes. This test pins the property
    that correction bought: the hash does not move when the STL bytes do.
    """
    link = next(p for p in parts if p.node_id == "link")
    baseline = link.geometry_sha256

    # Perturb the STL on disk however you like: the hash came from the compiled solid and is
    # already fixed, so nothing about the tessellation can move it. Restored afterwards -- the
    # fixture is module-scoped and later tests read the same file.
    original = link.stl.read_bytes()
    try:
        coarse = trimesh.load(str(link.stl), force="mesh").copy()
        coarse.apply_scale(0.5)
        coarse.export(str(link.stl))
        assert link.geometry_sha256 == baseline, "the hash must not depend on the STL at all"
    finally:
        link.stl.write_bytes(original)

    # It is still sensitive to a genuinely different solid: recompiling a 1%-smaller link moves it.
    smaller = dict(link.params)
    smaller["length_x"] = float(smaller["length_x"]) * 0.99
    other = compile_reference_part("flat_link", "link_small", smaller, link.stl.parent)
    assert other.geometry_sha256 != baseline, "a real dimensional change must move the hash"


def test_the_geometry_hash_sees_a_pure_translation() -> None:
    """The scorer does NO centering, so a solid that is merely MOVED is a different criterion.
    Hashing bbox SIZE alone would miss it and leave the freeze passing while every score shifted."""
    from forge_mind.analysis.reference_corpus import _brep_geometry_sha256

    class _Vec:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _BBox:
        def __init__(self, size, lo):
            self.size, self.min = size, lo

    class _Cad:
        def __init__(self, volume, size, lo):
            self.part = type("P", (), {"volume": volume})()
            self.bounding_box = _BBox(size, lo)

    size, volume = _Vec(40.0, 12.0, 4.0), 1750.2
    here = _brep_geometry_sha256(_Cad(volume, size, _Vec(0.0, 0.0, 0.0)))
    moved = _brep_geometry_sha256(_Cad(volume, size, _Vec(10.0, 0.0, 0.0)))
    assert here != moved, "same size and volume, different place -- must be different criterion"
    assert _brep_geometry_sha256(_Cad(volume, size, _Vec(0.0, 0.0, 0.0))) == here


def test_renaming_an_operation_is_not_corpus_drift(parts) -> None:
    """op_id is excluded from the program hash: this corpus consumes only the solid, so a rename
    emits an identical B-rep. A freeze that blocks harmless builder refactors gets worked around."""
    link = next(p for p in parts if p.node_id == "link")
    renamed_ops = [
        op.model_copy(update={"op_id": f"renamed_{i}"})
        for i, op in enumerate(link.program.operations)
    ]
    renamed = replace(link, program=link.program.model_copy(update={"operations": renamed_ops}))
    assert renamed.program_sha256 == link.program_sha256
