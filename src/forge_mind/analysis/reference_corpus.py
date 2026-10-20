"""Frozen reference corpus — the independent ground truth `shape_similarity` needs to mean anything.

`analysis.shape_similarity` measures Chamfer / volumetric-IoU / F1 **in absolute millimetres**,
deliberately unnormalized, because we care about dimensional accuracy and not just shape agreement
after a hidden scale fit. That only works against a reference whose millimetres are TRUE. This
module builds those references.

## Why not the reference spine we already have

`agentic.reference_lib` fetches an artist model from Sketchfab per run, which is right for the
vision critic (it wants a picture of the real object) and wrong for this metric:

* **Arbitrary scale.** An artist's crankshaft is whatever size the artist modelled. Chamfer in mm
  against it is meaningless.
* **Not watertight.** `shape_signature` already records this ("the Sketchfab reference is never
  watertight"), which rules out volumetric IoU.
* **Not reproducible.** A live search can return a different model tomorrow, so a score is not
  comparable across runs — fatal for a benchmark.

## What a reference is here

A **parametric spec**: archetype + exact millimetre parameters, compiled deterministically through
the same `assembly.archetypes` builders the engine itself uses. That gives a watertight,
dimensionally-true, reproducible mesh with no network, no token, and no licensing question.

Authoring the spec by hand is the point, not a compromise: the reference must be an acceptance
criterion fixed INDEPENDENTLY of what the generator produces. A spec written first and SHA-pinned
is exactly that — which is also why this unblocks the frozen-acceptance-manifest problem that
stalled the mirror-pair gate (`docs/research/2026-07-24-mirror-pair-defect-class.md`).

## Scope of this slice

PER-PART references only. Each node compiles on its own, so no joint/placement solver is involved
and no assembly-level pose can confound the score. Assembly-level comparison needs placement solved
from joints and is deliberately left out.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_mind.analysis.shape_similarity import ShapeSimilarity, compare_meshes
from forge_mind.assembly.archetypes import build_node
from forge_mind.compiler.compile import compile_program

# Nodes whose archetype is not a printed part builder (bought hardware). They have no parametric
# body to compile, so they are skipped rather than failing the whole spec.
_NON_PRINTED = frozenset({"hardware"})


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _brep_geometry_sha256(cad: Any, *, quantum: float = 1e-3) -> str:
    """Quantized invariants of the COMPILED B-REP: exact volume + bounding-box size.

    The third hash exists because ``spec_sha256`` and ``program_sha256`` both stop at the program:
    if the COMPILER's semantics change -- a primitive's dimensions, an orientation convention, a
    boolean rule -- the program is identical and the geometry is not.

    Read from the B-rep (``CADModel.part.volume``, ``CADModel.bounding_box``) and NEVER from the
    STL, which is the correction that matters here. An earlier version of this hashed the
    tessellated mesh and claimed to survive re-tessellation; it does not. Measured: exporting the
    same rod solid at fine vs default tolerance gives 490.855 vs 490.670 mm^3 and 5.000 vs 4.998 mm
    -- both far above this quantum, so a mesh-derived hash fires on tessellation-only changes,
    exactly the false drift it was supposed to avoid. B-rep volume is tessellation-independent.

    Known gap, stated rather than papered over: an internal change that preserves BOTH volume and
    bounding box slips through. The three hashes narrow the window; they do not seal it.
    """
    size = cad.bounding_box.size
    lo = cad.bounding_box.min
    return _sha256_json(
        {
            "volume_mm3": round(float(cad.part.volume) / quantum) * quantum,
            "bbox_mm": [round(float(v) / quantum) * quantum for v in (size.x, size.y, size.z)],
            # WORLD POSITION, not just size: `compare_meshes` does no centering and no scale fit,
            # so a solid that is merely TRANSLATED is a different criterion. Size alone cannot see
            # that, which would leave the freeze passing while every score shifted.
            "bbox_min_mm": [round(float(v) / quantum) * quantum for v in (lo.x, lo.y, lo.z)],
        }
    )


@dataclass(frozen=True)
class ReferencePart:
    """One compiled reference part: its id, tier, the STL, and the three hashes that freeze it."""

    node_id: str
    archetype: str
    tier: str
    params: dict[str, Any]
    program: Any  # PartProgram — the EXPANDED operations, kept so the freeze can bind them
    stl: Path
    # Quantized B-REP invariants, computed at compile time (see _brep_geometry_sha256). A field
    # rather than a property because it must come from the compiled solid, not from re-reading
    # the tessellated STL -- that distinction is the whole point of it.
    geometry_sha256: str

    @property
    def spec_sha256(self) -> str:
        """Hash of the authored INPUTS (archetype + params) -- what a human wrote down."""
        return _sha256_json({"archetype": self.archetype, "params": self.params})

    @property
    def program_sha256(self) -> str:
        """Hash of the GEOMETRY-AFFECTING half of the expanded PartProgram: the ordered operations
        plus the unit system the compiler reads them in.

        Why this hash exists: ``spec_sha256`` binds only the inputs, so a change to an archetype
        BUILDER slips straight past it. If ``flat_link`` stopped cutting its bores tomorrow, the
        archetype and params would be byte-identical, the extents and watertightness unchanged, and
        every longitudinal score would silently be measured against a different criterion.

        Why it is deliberately NARROW: hashing the whole program would drag in fields the compiler
        never reads -- ``metadata.description``, ``interfaces``, ``hardware_features`` -- so editing
        a description would report corpus drift and force a pointless re-freeze. A freeze that cries
        wolf gets ignored, which is worse than one gap.
        """
        return _sha256_json(
            {
                "units": self.program.metadata.units,
                # ``op_id`` is excluded: this corpus consumes only the solid, never the topology
                # index, so renaming an operation emits an identical B-rep and must not read as
                # corpus drift. If a rename ever DID break an intra-program reference, the
                # geometry would change and ``geometry_sha256`` catches it -- the narrow hash is
                # backstopped, not trusted alone.
                "operations": [
                    {k: v for k, v in op.model_dump(mode="json").items() if k != "op_id"}
                    for op in self.program.operations
                ],
            }
        )


def load_reference_spec(spec_path: Path) -> list[dict[str, Any]]:
    """The node list from a ``reference.assembly.json`` (``{nodes, joints, root}``).

    Joints and root are ignored here by design: this slice scores parts, not poses.
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"{spec_path}: no 'nodes' in reference spec")
    return nodes


def compile_reference_part(
    archetype: str,
    node_id: str,
    params: dict[str, Any],
    out_dir: Path,
    *,
    tier: str = "untiered",
) -> ReferencePart:
    """Compile ONE reference node to a watertight STL via the engine's own archetype builders.

    Raises on an unknown archetype or a failed build -- a reference that cannot be built is not a
    reference, and silently dropping it would quietly shrink the corpus (and flatter the score).
    """
    program, _ports = build_node(archetype, node_id, params)
    cad = compile_program(program, Path(out_dir) / node_id)
    return ReferencePart(
        node_id=node_id,
        archetype=archetype,
        tier=tier,
        params=dict(params),
        program=program,
        stl=Path(cad.stl_path),
        geometry_sha256=_brep_geometry_sha256(cad),
    )


def compile_reference_corpus(spec_path: Path, out_dir: Path) -> list[ReferencePart]:
    """Compile every PRINTED node of a reference spec. Non-printed hardware nodes are skipped.

    ``tier`` rides through: the corpus is meant to be read PER TIER (a mean across L1-L4 hides
    exactly the signal the tiers exist to expose), so dropping it here would force every caller to
    re-parse the spec to do the reporting this corpus is for.
    """
    parts: list[ReferencePart] = []
    for node in load_reference_spec(spec_path):
        archetype = str(node.get("archetype", ""))
        if archetype in _NON_PRINTED:
            continue
        parts.append(
            compile_reference_part(
                archetype,
                str(node["id"]),
                dict(node.get("params") or {}),
                Path(out_dir),
                tier=str(node.get("tier", "untiered")),
            )
        )
    return parts


def score_against_reference(
    generated_stl: Path, reference: ReferencePart, **kwargs: Any
) -> ShapeSimilarity:
    """Chamfer (mm) / volumetric IoU / F1@threshold for a generated part vs its reference.

    Thin wrapper so callers never have to remember which argument is the reference -- getting that
    backwards silently reports a different (non-symmetric-in-meaning) number for F1 precision and
    recall.
    """
    return compare_meshes(generated_stl, reference.stl, **kwargs)


def corpus_manifest(parts: list[ReferencePart]) -> dict[str, Any]:
    """The freeze record: one entry per part, hashing BOTH the authored inputs and the expanded
    program, and carrying the tier the corpus is meant to be reported by.

    Written next to the corpus so a later run can prove it scored against the same criterion.
    Matches the repo's SHA-pinned `MANIFEST` convention for frozen benchmark sets.
    """
    return {
        "version": 3,
        "parts": [
            {
                "id": p.node_id,
                "archetype": p.archetype,
                "tier": p.tier,
                "params": p.params,
                "spec_sha256": p.spec_sha256,
                "program_sha256": p.program_sha256,
                "geometry_sha256": p.geometry_sha256,
            }
            for p in sorted(parts, key=lambda x: x.node_id)
        ],
    }


__all__ = [
    "ReferencePart",
    "compile_reference_corpus",
    "compile_reference_part",
    "corpus_manifest",
    "load_reference_spec",
    "score_against_reference",
]
