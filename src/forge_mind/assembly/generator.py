"""AssemblyGraphGenerator — part request -> AssemblyGraph via the LLM (ADR-0010, layer-4).

The LLM composes a part from the generic archetype catalog (knuckled_plate, rod,
tube) wired by typed joints (coaxial_insert, revolute, prismatic, fixed). It chooses
the topology (which archetypes, knuckle counts, which ports mate via which joints)
and coarse sizes. It does NOT emit world coordinates, individual knuckle positions,
the pin/piston radius, or the slide travel: those *relational* metrics are derived
(interleave from counts; pin span from the knuckles; piston length + travel from the
tube; rod radius from the bore). "LLM picks categorical, code derives relational."
A hinge and a linear slide both come out of the same catalog -- no object-templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode
from forge_mind.assembly.plan import FunctionalPlan, plan_graph_issues, plan_issues
from forge_mind.config import LAYOUTSPEC_MODEL
from forge_mind.dsl.interfaces.nema17_shaft import SHAFT_BORE_RADIUS_MM as _SHAFT_BORE_RADIUS_MM
from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

_PROMPT_PATH = Path(__file__).parent / "prompts" / "assembly_v1.md"
_KNUCKLE_GAP_MM = 1.0  # axial clearance between adjacent interleaved knuckles
_PISTON_LEN_FRAC = 0.45  # piston length as a fraction of the tube length
_SLIDE_MARGIN_MM = 4.0  # keep the piston this far from sliding off the tube end
_MIN_BARREL_WALL_MM = 1.5  # a barrel/tube wall must be at least this much thicker than its bore


class LlmKnuckledPlate(BaseModel):
    """A plate with knuckles on one edge. The LLM picks the count + coarse sizes;
    the interleaved knuckle positions and clearance notches are derived."""

    model_config = ConfigDict(extra="forbid")

    id: str
    knuckle_count: int = Field(..., ge=1, le=8, description="How many knuckles this leaf has.")
    length_x: float = Field(
        ..., description="Plate length along the hinge axis (a floor; widened to fit the knuckles)."
    )
    width_y: float
    thickness: float
    knuckle_outer_r: float
    bore_r: float
    knuckle_width: float


class LlmRod(BaseModel):
    """A pin/piston. Length and radius are both derived (length from the knuckle span
    or the tube; radius from the bore + fit)."""

    model_config = ConfigDict(extra="forbid")

    id: str


class LlmTube(BaseModel):
    """A hollow sleeve a rod slides/pivots through. The LLM picks the size."""

    model_config = ConfigDict(extra="forbid")

    id: str
    length_x: float
    outer_r: float
    bore_r: float


class LlmFlatLink(BaseModel):
    """A flat bar with a pivot bore at each end -- a link in a planar linkage."""

    model_config = ConfigDict(extra="forbid")

    id: str
    length_x: float
    width_y: float
    thickness: float
    bore_r: float


class LlmOutputArm(BaseModel):
    """A driven robot-arm link: a beam + a shaft hub that grips the motor shaft. The
    LLM picks the coarse beam size only; the gripping bore, set-screw boss, and hub
    dimensions are DERIVED from the shared NEMA 17 shaft facts (it never sizes the
    coupling -- the ADR-0009 'LLM picks categorical, code derives relational' rule)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    length_x: float = Field(..., description="Beam reach from the hub (mm).")
    width_y: float
    thickness: float


class LlmMotor(BaseModel):
    """A NEMA 17 motor: a NON-PRINTED hardware node. The LLM only names it -- every
    fact (bbox, face + shaft frames coaxial by fact) comes from the registry. It emits
    no geometry and no size for the motor."""

    model_config = ConfigDict(extra="forbid")

    id: str


class LlmJoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["coaxial_insert", "revolute", "prismatic", "fixed"]
    a: str = Field(..., description="Reference port '<part_id>.<port>'.")
    b: str = Field(..., description="Mated/moving port '<part_id>.<port>'.")
    range_deg: list[float] = Field(
        default_factory=list, description="[lo, hi] for a revolute joint, else empty."
    )
    clearance_class: Literal["running_fit", "loose_fit", "interference"] = "running_fit"


class LlmAssembly(BaseModel):
    """LLM output: a functional plan, then the geometry that realizes it."""

    model_config = ConfigDict(extra="forbid")

    plan: FunctionalPlan = Field(
        ...,
        description="State the functional decomposition FIRST: the parts + roles, and what "
        "rotates about / slides along / is fixed to what. The geometry below must realize it.",
    )
    knuckled_plates: list[LlmKnuckledPlate] = Field(default_factory=list)
    rods: list[LlmRod] = Field(default_factory=list)
    tubes: list[LlmTube] = Field(default_factory=list)
    flat_links: list[LlmFlatLink] = Field(default_factory=list)
    output_arms: list[LlmOutputArm] = Field(default_factory=list)
    motors: list[LlmMotor] = Field(default_factory=list)
    joints: list[LlmJoint]
    root: str = Field(..., description="Part placed at world identity (the fixed member).")


def _interleave_positions(
    count_a: int, count_b: int, pitch: float
) -> tuple[list[float], list[float]]:
    """Centered, alternating knuckle X positions for an interleaved pair.

    e.g. (2, 1) -> A at the two outer slots, B in the centre slot; (2, 1) the larger
    leaf takes the outer slots so the knuckles alternate cleanly.
    """
    pattern: list[str] = []
    # The larger-count leaf takes the outer slots so the knuckles alternate cleanly
    # (e.g. (2,3) -> B,A,B,A,B, not A,B,A,B,B). Equal counts start with A.
    a_left, b_left, turn_a = count_a, count_b, count_a >= count_b
    while a_left or b_left:
        if turn_a and a_left:
            pattern.append("a")
            a_left -= 1
        elif not turn_a and b_left:
            pattern.append("b")
            b_left -= 1
        elif a_left:
            pattern.append("a")
            a_left -= 1
        else:
            pattern.append("b")
            b_left -= 1
        turn_a = not turn_a
    total = count_a + count_b
    centered = [(i - (total - 1) / 2.0) * pitch for i in range(total)]
    pos_a = [centered[i] for i, t in enumerate(pattern) if t == "a"]
    pos_b = [centered[i] for i, t in enumerate(pattern) if t == "b"]
    return pos_a, pos_b


def _plate_node(
    plate: LlmKnuckledPlate, centers: list[float], notches: list[float], kw: float, span: float
) -> PartNode:
    outer_r = max(
        plate.knuckle_outer_r, plate.bore_r + _MIN_BARREL_WALL_MM
    )  # barrel thicker than its bore
    length_x = max(plate.length_x, span + 2.0 * outer_r)  # ensure the plate contains its knuckles
    return PartNode(
        id=plate.id,
        archetype="knuckled_plate",
        params={
            "length_x": length_x,
            "width_y": plate.width_y,
            "thickness": plate.thickness,
            "knuckle_centers_x": centers,
            "knuckle_outer_r": outer_r,
            "bore_r": plate.bore_r,
            "knuckle_width": kw,
            "clearance_notches_x": notches,
        },
    )


def to_assembly_graph(emitted: LlmAssembly) -> AssemblyGraph:
    """Map the typed LLM output to the real AssemblyGraph, DERIVING all relational
    metric (interleave layout, pin span, piston length + travel) from the chosen
    counts and sizes. The LLM never states a coordinate, a knuckle position, a pin
    radius, or a slide travel."""
    plates = {p.id: p for p in emitted.knuckled_plates}
    tubes = {t.id: t for t in emitted.tubes}
    links = {fl.id: fl for fl in emitted.flat_links}
    rod_ids = {r.id for r in emitted.rods}
    rod_length: dict[str, float] = {}
    nodes: list[PartNode] = []

    # --- hinge: interleave the two plates joined by the revolute pivot ---
    placed: set[str] = set()
    pin_length = 25.0
    for joint in emitted.joints:
        a, b = joint.a.split(".", 1)[0], joint.b.split(".", 1)[0]
        if joint.type == "revolute" and a in plates and b in plates and a not in placed:
            pa, pb = plates[a], plates[b]
            kw = min(pa.knuckle_width, pb.knuckle_width)
            pitch = kw + _KNUCKLE_GAP_MM
            pos_a, pos_b = _interleave_positions(pa.knuckle_count, pb.knuckle_count, pitch)
            all_pos = pos_a + pos_b
            span = (max(all_pos) - min(all_pos)) + kw if all_pos else kw
            # FLUSH-RECESSED, never overhanging: the cleared region (knuckle bores +
            # clearance notches) ends at the outer knuckle faces. The old span+4 pin
            # poked ~2 mm per side THROUGH the un-notched plate edge -- a real
            # cannot-assemble defect the chain gate's static-clearance check caught
            # (pin interior 0.25-1.0 mm inside both leaves at |x| ~ span/2+).
            pin_length = span - 0.5
            nodes.append(_plate_node(pa, pos_a, pos_b, kw, span))
            # leafB is placed by the solver with a 180-deg rotation about the pivot
            # (x -> -x), so its LOCAL knuckle + notch positions must be negated to land
            # at the intended interleaved world positions. (Symmetric sets are a no-op;
            # asymmetric ones -- e.g. equal counts -- would otherwise overlap leafA.)
            nodes.append(_plate_node(pb, [-x for x in pos_b], [-x for x in pos_a], kw, span))
            placed.update({a, b})

    # remaining plates: centre their own knuckles, no interleave partner
    for p in emitted.knuckled_plates:
        if p.id in placed:
            continue
        pitch = p.knuckle_width + _KNUCKLE_GAP_MM
        centers = [(i - (p.knuckle_count - 1) / 2.0) * pitch for i in range(p.knuckle_count)]
        span = (max(centers) - min(centers)) + p.knuckle_width if centers else p.knuckle_width
        nodes.append(_plate_node(p, centers, [], p.knuckle_width, span))

    # --- tubes + slide travel: derive piston length + travel from the tube ---
    slide_travel: dict[int, tuple[float, float]] = {}
    for t in emitted.tubes:
        tube_outer_r = max(t.outer_r, t.bore_r + _MIN_BARREL_WALL_MM)  # wall thicker than its bore
        nodes.append(
            PartNode(
                id=t.id,
                archetype="tube",
                params={"length_x": t.length_x, "outer_r": tube_outer_r, "bore_r": t.bore_r},
            )
        )
    for fl in emitted.flat_links:
        end_margin = fl.bore_r + _MIN_BARREL_WALL_MM  # the bore clears the bar end
        link_width = max(
            fl.width_y, 2.0 * fl.bore_r + 2.0 * _MIN_BARREL_WALL_MM
        )  # bore fits across the width
        link_length = max(
            fl.length_x, 2.0 * (fl.bore_r + end_margin)
        )  # both bores fit, none reach an end
        nodes.append(
            PartNode(
                id=fl.id,
                archetype="flat_link",
                params={
                    "length_x": link_length,
                    "width_y": link_width,
                    "thickness": max(fl.thickness, 2.5),
                    "bore_r": fl.bore_r,
                    "end_margin": end_margin,
                },
            )
        )
    # --- motors: non-printed hardware nodes (frames + facts; no geometry) ---
    for m in emitted.motors:
        nodes.append(
            PartNode(
                id=m.id, archetype="hardware", printed=False, params={"hardware": "nema17_motor"}
            )
        )
    # --- output arms: the LLM's coarse beam + a DERIVED shaft hub (grip bore +
    # set-screw boss). Hub wall >= the barrel-wall floor around the grip bore. ---
    for arm in emitted.output_arms:
        hub_outer_r = _SHAFT_BORE_RADIUS_MM + _MIN_BARREL_WALL_MM + 1.0  # boss wall past the bore
        nodes.append(
            PartNode(
                id=arm.id,
                archetype="output_arm",
                params={
                    "length_x": max(arm.length_x, 3.0 * hub_outer_r),  # beam clears the hub
                    "width_y": max(arm.width_y, 2.0 * hub_outer_r),
                    "thickness": max(arm.thickness, 3.0),
                    "hub_outer_r": hub_outer_r,
                    "hub_height": max(8.0, 2.0 * hub_outer_r),
                },
            )
        )

    for idx, joint in enumerate(emitted.joints):
        a, b = joint.a.split(".", 1)[0], joint.b.split(".", 1)[0]
        if joint.type == "coaxial_insert" and b in rod_ids:
            # a linkage pin spans two stacked links; a hinge pin spans the knuckle interleave
            rod_length[b] = (2.0 * links[a].thickness + 8.0) if a in links else pin_length
        if joint.type == "prismatic" and a in tubes and b in rod_ids:
            tube = tubes[a]
            piston_len = max(6.0, tube.length_x * _PISTON_LEN_FRAC)
            rod_length[b] = piston_len
            travel_hi = max(2.0, tube.length_x - piston_len - 2.0 * _SLIDE_MARGIN_MM)
            slide_travel[idx] = (0.0, travel_hi)

    # rods, with derived lengths
    for r in emitted.rods:
        nodes.append(
            PartNode(id=r.id, archetype="rod", params={"length_x": rod_length.get(r.id, 25.0)})
        )

    # joints, with derived ranges
    joints: list[Joint] = []
    for idx, joint in enumerate(emitted.joints):
        ja, jb = joint.a.split(".", 1)[0], joint.b.split(".", 1)[0]
        rng_deg = (
            (float(joint.range_deg[0]), float(joint.range_deg[1]))
            if len(joint.range_deg) == 2
            else None
        )
        rng_mm = slide_travel.get(idx)
        # chained links stack (parallel "align"); everything else opposes (anti-parallel)
        mate: Literal["oppose", "align"] = (
            "align" if (joint.type == "revolute" and ja in links and jb in links) else "oppose"
        )
        # A rod inserted/slid into a bore is a FUNCTIONAL pin/piston: it must turn or
        # slide, so an interference fit is structurally wrong (the derived radius would
        # exceed the bore). The fit class that sizes the pin is therefore derived, not
        # the LLM's to pick -- coerce interference to running_fit. (Wordings like
        # "interlocking knuckles" otherwise lure the model into interference and fail
        # assembly.pin_clearance.) Graph-level fixtures + the reviser still set any
        # class directly; this only governs the LLM surface.
        clearance_class = joint.clearance_class
        if (
            joint.type in ("coaxial_insert", "prismatic")
            and jb in rod_ids
            and clearance_class == "interference"
        ):
            clearance_class = "running_fit"
        joints.append(
            Joint(
                type=JointType(joint.type),
                a=joint.a,
                b=joint.b,
                range_deg=rng_deg,
                range_mm=rng_mm,
                clearance_class=clearance_class,
                mate=mate,
            )
        )

    return AssemblyGraph(nodes=nodes, joints=joints, root=emitted.root)


class AssemblyGraphGenerator:
    """Stage: a part request -> AssemblyGraph. The LLM picks topology + counts + coarse sizes."""

    def __init__(self, client: LLMClientLike | None = None, model: str | None = None) -> None:
        self._client = client or make_client()
        self._system = _PROMPT_PATH.read_text(encoding="utf-8")
        self._model = model or LAYOUTSPEC_MODEL

    def __call__(self, request: str) -> tuple[AssemblyGraph, LLMUsage]:
        user = (
            "Compose the following part from the archetype catalog. Choose the "
            "archetypes, the topology, the knuckle counts, and coarse sizes; do not "
            "write any world coordinate, knuckle position, pin/piston radius, or "
            "slide travel.\n\n"
            f"Request:\n{request}\n"
        )
        emitted, usage = self._client.generate_structured(
            system=self._system,
            user=user,
            output_model=LlmAssembly,
            model=self._model,
            max_tokens=2000,
        )
        graph = to_assembly_graph(emitted)
        # The functional plan is load-bearing: it must be coherent AND realized by the
        # geometry. A structure that contradicts its stated function is a generation
        # failure (caught here, before solve/compile).
        issues = plan_issues(emitted.plan) + plan_graph_issues(emitted.plan, graph)
        if issues:
            raise LLMError("functional plan not realized by the geometry: " + "; ".join(issues))
        return graph, usage


__all__ = [
    "AssemblyGraphGenerator",
    "LlmAssembly",
    "LlmFlatLink",
    "LlmJoint",
    "LlmKnuckledPlate",
    "LlmMotor",
    "LlmOutputArm",
    "LlmRod",
    "LlmTube",
    "to_assembly_graph",
]
