"""The real LLM trial runner (ADR-0016 Step 3): a decontaminated functional Blueprint ->
the LayoutSpecGenerator picks the composable MENU (base + interfaces) -> the ENGINE owns
the geometry (derives dims) -> interpret + validate, with each interface independently
certified (ADR-0015). The LLM only ever picks the categorical menu; code derives geometry.

A trial PASSES iff the LLM emitted THIS family's menu (reachability) AND the engine-built
part validates with every interface certified. The validate is deterministic per
(shape, perturbation), so it is cached -- only the stochastic LLM menu pick runs each trial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from forge_mind.blueprint.schema import (
    Blueprint,
    BlueprintVec3,
    FeatureSpec,
    InterfaceSpec,
    RequestedInterface,
)
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import BaseArchetype, ComposablePlateSpec, Face, InterfaceRef
from forge_mind.layout.generator import LayoutSpecGenerator
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.llm.client import LLMError
from forge_mind.scaling.engine import Perturbation, TrialResult
from forge_mind.scaling.enumerate import FamilyShape, build_spec

_FACE_FACING = {
    Face.PLATE_TOP: "+Z",
    Face.PLATE_BOTTOM: "-Z",
    Face.BASE_TOP: "+Z",
    Face.UPRIGHT_FRONT: "+Y",
}
_REF_TYPE = {
    InterfaceRef.NEMA17_FACE: RequestedInterface.NEMA17_FACE,
    InterfaceRef.FAN_FACE: RequestedInterface.FAN_FACE,
    InterfaceRef.EXTRUSION_2020: RequestedInterface.EXTRUSION_2020_SLOT,
}
_REF_NOUN = {
    InterfaceRef.NEMA17_FACE: "a NEMA 17 motor",
    InterfaceRef.FAN_FACE: "an axial fan",
    InterfaceRef.EXTRUSION_2020: "a 2020 extrusion (M5 T-nut)",
}
# Per-component nouns so the decontaminated Blueprint NAMES the variant (ADR-0017), letting
# the LLM dispatch the right component_id. Decontaminated still: it names the COMPONENT (the
# request -- "a NEMA 23 motor"), never the part's geometry (dims/offsets/bolt positions). The
# legacy defaults (nema17 -> None, ext_2020 -> None) keep the generic _REF_NOUN wording.
_VARIANT_NOUN = {
    "nema14": "a NEMA 14 motor",
    "nema23": "a NEMA 23 motor",
    "ext_2040": "a 2040 extrusion (M5 T-nut)",
    "ext_3030": "a 3030 extrusion (M6 T-nut)",
    "fan_40mm": "a 40 mm axial fan",
    "fan_60mm": "a 60 mm axial fan",
}


def _noun(placement) -> str:  # type: ignore[no-untyped-def]
    """Name the interface's COMPONENT for the decontaminated request: a registry variant by
    its component_id (NEMA 23, 3030, 60 mm fan) so the LLM picks the right id; the legacy
    default (component_id None) keeps the generic noun. Unknown id -> generic (safe)."""
    if placement.component_id is not None and placement.component_id in _VARIANT_NOUN:
        return _VARIANT_NOUN[placement.component_id]
    return _REF_NOUN[placement.ref]


# Paraphrase templates (decontaminated: name the FUNCTION + interfaces, never geometry).
_SUMMARIES = [
    "A {base} that mounts {items}.",
    "{Base_cap} bracket carrying {items}; derive the layout to fit.",
    "Design a {base} part that holds {items} together on one piece.",
]


def _base_phrase(base: BaseArchetype) -> str:
    return "flat plate" if base is BaseArchetype.FLAT_PLATE else "right-angle L-bracket"


def build_blueprint(shape: FamilyShape, pert: Perturbation, paraphrase_idx: int) -> Blueprint:
    """A decontaminated functional Blueprint for a family at a perturbation point. Names the
    interfaces + facings (the request); states NO dims/offsets/bolt positions (the answer)."""
    base_phrase = _base_phrase(shape.base)
    items = ", ".join(_noun(p) for p in shape.placements)
    tmpl = _SUMMARIES[paraphrase_idx % len(_SUMMARIES)]
    summary = tmpl.format(base=base_phrase, Base_cap=base_phrase.capitalize(), items=items)

    if shape.base is BaseArchetype.FLAT_PLATE:
        features = [
            FeatureSpec(
                name="plate",
                kind="plate",
                purpose="Carries the face interfaces.",
                approx_size_mm=BlueprintVec3(x=80.0, y=60.0, z=pert.min_wall_mm),
            )
        ]
        feat_for = dict.fromkeys(shape.placements, "plate")
    else:
        features = [
            FeatureSpec(
                name="base_plate",
                kind="plate",
                purpose="Carries the motor face.",
                approx_size_mm=BlueprintVec3(x=43.0, y=50.0, z=pert.min_wall_mm),
            ),
            FeatureSpec(
                name="upright",
                kind="plate",
                purpose="Carries the extrusion face.",
                approx_size_mm=BlueprintVec3(x=43.0, y=pert.min_wall_mm, z=30.0),
            ),
        ]
        feat_for = {
            p: ("base_plate" if p.ref is InterfaceRef.NEMA17_FACE else "upright")
            for p in shape.placements
        }

    required = []
    for i, p in enumerate(shape.placements):
        nm = _REF_TYPE[p.ref].value + (f"_{i + 1}" if i else "")
        required.append(
            InterfaceSpec(
                name=nm,
                interface_type=_REF_TYPE[p.ref],
                on_feature=feat_for[p],
                facing=_FACE_FACING[p.face],
            )
        )
    env = pert.envelope_mm
    return Blueprint(
        name=shape.key,
        summary=summary,
        envelope_mm=BlueprintVec3(x=env.x, y=env.y, z=env.z),
        print_orientation=BlueprintVec3(x=0.0, y=0.0, z=1.0),
        material="PLA",
        min_wall_mm=pert.min_wall_mm,
        max_overhang_deg=45.0,
        features=features,
        required_interfaces=required,
    )


def _menu(spec: ComposablePlateSpec) -> tuple:  # type: ignore[type-arg]
    return (spec.base, frozenset((p.ref, p.face, p.component_id) for p in spec.interfaces))


def _shape_menu(shape: FamilyShape) -> tuple:  # type: ignore[type-arg]
    return (shape.base, frozenset((p.ref, p.face, p.component_id) for p in shape.placements))


@dataclass
class LlmTrialRunner:
    """Reachability runner: the LLM picks the menu; the engine derives geometry + validates.
    Validation is cached per (shape, perturbation) -- only the LLM menu pick runs per trial."""

    out_dir: Path
    generator: LayoutSpecGenerator = field(default_factory=LayoutSpecGenerator)
    _validate_cache: dict[tuple[str, str], tuple[bool, bool]] = field(default_factory=dict)
    _feasible_cache: dict[tuple[str, str], bool] = field(default_factory=dict)

    def feasible(self, shape: FamilyShape, *, perturbation: Perturbation) -> bool:
        key = (shape.key, perturbation.label)
        if key not in self._feasible_cache:
            spec = build_spec(
                shape,
                name=shape.key,
                envelope_mm=perturbation.envelope_mm,
                min_wall_mm=perturbation.min_wall_mm,
            )
            try:
                interpret_composable_plate(spec)
                self._feasible_cache[key] = True
            except LayoutError:
                self._feasible_cache[key] = False
        return self._feasible_cache[key]

    def __call__(
        self, shape: FamilyShape, *, perturbation: Perturbation, paraphrase_idx: int
    ) -> TrialResult:
        bp = build_blueprint(shape, perturbation, paraphrase_idx)
        try:
            spec, usage = self.generator(bp)
        except LLMError as exc:
            return TrialResult(False, False, 0.0, f"llm error: {exc}")
        cost = usage.cost_usd
        if not isinstance(spec, ComposablePlateSpec) or _menu(spec) != _shape_menu(shape):
            return TrialResult(False, False, cost, "LLM picked the wrong menu/spec")
        passed, certified = self._validate(shape, perturbation)
        return TrialResult(passed and certified, certified, cost, "")

    def _validate(self, shape: FamilyShape, pert: Perturbation) -> tuple[bool, bool]:
        key = (shape.key, pert.label)
        if key in self._validate_cache:
            return self._validate_cache[key]
        complete: ComposablePlateSpec = build_spec(
            shape, name=shape.key, envelope_mm=pert.envelope_mm, min_wall_mm=pert.min_wall_mm
        )
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{shape.key}_{pert.label}")
        result = run_layout_pipeline(complete, self.out_dir / slug)
        report = result.report
        passed = bool(result.passed and report is not None)
        certified = bool(
            passed
            and report is not None
            and len({c.id.split(".")[1] for c in report.checks if c.id.startswith("interfaces.")})
            == len(complete.interfaces)
        )
        self._validate_cache[key] = (passed, certified)
        return passed, certified


__all__ = ["LlmTrialRunner", "build_blueprint"]
