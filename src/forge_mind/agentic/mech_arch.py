"""Mechanical architecture stage (FORGE_MECH_ARCH=1, default OFF): the interface
control document an engineering team writes BEFORE part design.

The founder's 2026-07-23 bar ("engineered parts like a team of engineers would do it")
failed at H3 precisely here: with no architecture, the free plan placed decorative
pins and dropped the servos entirely. This stage makes Fable the chief mechanical
architect: one typed call that fixes, per joint, actuated-vs-passive, axis, station,
bearing, and retention; places each actuator in the shared frame (these placements
DERIVE the HardwareDecls, so the envelope/motion/servo-fit gates measure against the
same architecture the cards were written to); and fixes the fastening system, seam
strategy, wiring route, and mirrored pairs for the whole object.

Flag OFF = byte-identical pipeline (the stage is never called). The stage itself is
fail-open at the call site: an architecture that cannot be produced degrades to the
archless pipeline; the deterministic plan gates remain the enforcement."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge_mind.config import PLAN_MAX_TOKENS, SONNET
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike

AxisDir = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

_DIR_VEC: dict[str, np.ndarray] = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}


class MechJoint(BaseModel):
    """One joint's engineering decisions, fixed before any part card exists."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Joint name, e.g. 'shoulder_L'.")
    kind: Literal["actuated", "passive"]
    axis_dir: AxisDir = Field(..., description="Joint axis direction in the shared frame.")
    station_mm: tuple[float, float, float] = Field(
        ..., description="A point the joint axis passes through, shared frame, mm."
    )
    bearing: str = Field(
        ...,
        description="The bearing decision BY NAME: journal dia + bore land + running "
        "clearance class (e.g. 'Ø6.0 pin journal in Ø6.35 bore land, running fit 0.35').",
    )
    retention: str = Field(
        ...,
        description="The axial retention BY NAME (e.g. 'printed shoulder + push-on cap', "
        "'e-clip groove 0.9 wide', 'threaded end + nut pocket'). Never empty for a pin.",
    )


class ActuatorPlacement(BaseModel):
    """One bought actuator placed in the shared frame; DERIVES a HardwareDecl."""

    model_config = ConfigDict(extra="forbid")

    joint: str = Field(..., description="The joint this actuator drives (a MechJoint name).")
    hw_id: str = Field(..., description="Unique id, e.g. 'servo_shoulder_L'.")
    kind: str = Field(default="mg996r_servo", description="Registry hardware kind.")
    position_mm: tuple[float, float, float] = Field(
        ...,
        description="Placement of the hardware local origin (servo: case top center, "
        "shaft on local +Z), shared frame, mm.",
    )
    shaft_dir: AxisDir = Field(..., description="World direction the output shaft points.")
    horn_link: str = Field(..., description="The driven part that couples to the HORN.")


class MechArchitecture(BaseModel):
    """The whole-object interface control document. Cards are written AGAINST this."""

    model_config = ConfigDict(extra="forbid")

    object_name: str
    # Pinned EQUAL to FunctionalPlan.joints (72), not to a hand-counted figure. The arch joint
    # graph is never richer than the plan's, so matching the plan cap makes this dependency
    # structural instead of a number somebody has to remember to bump. It was missed twice:
    # 24 (flat-six needed 23, fit by ONE) and then 32 (a flat-ten needs ~37).
    #
    # Getting this wrong FAILS SILENTLY, which is why it is pinned by a test: the call site
    # wraps this stage in try/except and degrades to the archless pipeline, so an over-cap
    # architecture does not error -- it just prints "mech-arch: skipped" and the object quietly
    # loses its binding bearings, retention and actuator placements.
    joints: list[MechJoint] = Field(..., min_length=1, max_length=72)
    actuators: list[ActuatorPlacement] = Field(default_factory=list, max_length=12)
    fastening_system: str = Field(
        ...,
        description="ONE consistent system: screw standard, boss/insert policy, "
        "e.g. 'M3 self-tappers into 4.0 bosses everywhere; M4 only at servo lugs'.",
    )
    seam_strategy: str = Field(
        ...,
        description="Where shells split and how seams register + fasten "
        "(tongue-and-groove, lip, screw boss count).",
    )
    wiring_route: str = Field(..., description="Named route from each actuator bay to the bus.")
    mirrored_pairs: list[list[str]] = Field(
        default_factory=list,
        description="Part-name pairs that must be exact sagittal mirrors, e.g. "
        "[['thigh_L','thigh_R']].",
    )

    @model_validator(mode="after")
    def _reconciled(self) -> MechArchitecture:
        """ADR-0130 (holes 3+6): the invariants every downstream count relies on. All five
        archived real arches (F1, F2, H4, P1, P2) satisfy these; a violating arch fails the
        structured call, whose claude_code retry carries the first ~300 chars of this
        message -- keep messages short and front-loaded. Still-invalid degrades archless
        (the stage's existing fail-open), never into untrustworthy counts."""
        names = [j.name for j in self.joints]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"duplicate joint name(s) {dup}: every MechJoint.name must be unique")
        ids = [a.hw_id for a in self.actuators]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate hw_id(s) {dup}: every ActuatorPlacement needs its own id")
        actuated = {j.name for j in self.joints if j.kind == "actuated"}
        for a in self.actuators:
            if a.joint not in set(names):
                raise ValueError(
                    f"actuator '{a.hw_id}' drives undeclared joint '{a.joint}' -- "
                    "declare the joint or fix the reference"
                )
            if a.joint not in actuated:
                raise ValueError(
                    f"actuator '{a.hw_id}' drives '{a.joint}' which is kind=passive -- "
                    "mark that joint actuated or re-target the actuator"
                )
        placed = [a.joint for a in self.actuators]
        dup = sorted({j for j in placed if placed.count(j) > 1})
        if dup:
            raise ValueError(
                f"joint(s) {dup} carry more than one actuator placement -- exactly one each"
            )
        unplaced = sorted(actuated - set(placed))
        if unplaced:
            raise ValueError(
                f"actuated joint(s) {unplaced} have no actuator placement -- "
                "place an actuator on each or mark them passive"
            )
        return self


_ARCH_SYSTEM = """You are the CHIEF MECHANICAL ARCHITECT of a professional product-design \
team. Before any part is drawn, you fix the mechanical architecture the whole team builds \
against -- exactly what a real engineering org freezes in an interface control document.

Given the product request (and the researched brief), emit the MechArchitecture:
- JOINTS: every articulation, actuated or passive, with its axis direction, the exact \
station (a point on the axis, shared frame, mm), the BEARING decision by name (journal \
diameter, bore land, running clearance class), and the axial RETENTION by name. A pin \
with no retention is forbidden.
- ACTUATORS: place every bought actuator ON its joint axis with a real position and shaft \
direction (the servo local convention: case top at z=0, output shaft on local +Z). The \
horn couples to the named driven link. An actuated joint whose servo cannot physically \
fit at its station is YOUR error -- move the joint or re-route the drive.
- FASTENING SYSTEM: one consistent screw standard + boss/insert policy for the whole \
object. SEAM STRATEGY: where shells split, how seams register and screw together. \
WIRING ROUTE: how each actuator's cable reaches the electronics bay.
- MIRRORED PAIRS: name every left/right pair that must be exact sagittal mirrors.

Use the shared frame the briefs use (Z up from the ground plane). Respect any stated \
anthropometry or proportion table exactly -- stations are load-bearing numbers, not \
suggestions. Be concrete: every field carries numbers a drafter can copy verbatim."""


def plan_mech_architecture(
    prompt: str,
    *,
    client: LLMClientLike,
    model: str | None = None,
    brief_text: str | None = None,
    doctrine: str | None = None,
) -> tuple[MechArchitecture, LLMUsage]:
    """One typed chief-architect call. ``doctrine`` (the enabled decompose overlays)
    rides the system prompt so the architecture obeys the same form/engineering rules
    the cards will be audited against."""
    system = _ARCH_SYSTEM + (doctrine or "")
    user = f"Product request:\n{prompt.strip()}\n"
    if brief_text:
        user += f"\nResearched brief (the judge's reference):\n{brief_text.strip()}\n"
    user += "\nEmit the MechArchitecture."
    return client.generate_structured(
        system=system,
        user=user,
        output_model=MechArchitecture,
        model=model or SONNET,
        max_tokens=PLAN_MAX_TOKENS,
    )


def arch_transform(position_mm: tuple[float, float, float], shaft_dir: str) -> np.ndarray:
    """4x4 world placement mapping the hardware LOCAL frame (shaft on local +Z) to a
    world position + shaft direction. Axis-aligned directions only (the six tokens the
    schema admits) -- deterministic, no gimbal ambiguity."""
    z = _DIR_VEC[shaft_dir]
    # Any world vector not parallel to z seeds a right-handed basis.
    seed = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(seed, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2] = x, y, z
    t[:3, 3] = np.asarray(position_mm, dtype=np.float64)
    return t


def hardware_decls_from(arch: MechArchitecture) -> list[Any]:
    """Derive the HardwareDecls the gates consume from the architecture's placements --
    the cards and the instruments measure against the SAME document."""
    from forge_mind.structures.servo_bridge import HardwareDecl

    return [
        HardwareDecl(
            kind=a.kind,
            hw_id=a.hw_id,
            transform=arch_transform(a.position_mm, a.shaft_dir),
        )
        for a in arch.actuators
    ]


def arch_context_block(arch: MechArchitecture) -> str:
    """The binding context every decompose call carries when the stage is on: the plan
    must REALIZE this architecture, and the deterministic gates audit exactly it."""
    joints = "\n".join(
        f"- {j.name} ({j.kind}): axis {j.axis_dir} through {tuple(j.station_mm)}; "
        f"bearing {j.bearing}; retention {j.retention}"
        for j in arch.joints
    )
    actuators = (
        "\n".join(
            f"- {a.hw_id} ({a.kind}) drives {a.joint}: local origin at {tuple(a.position_mm)}, "
            f"shaft {a.shaft_dir}, horn couples to '{a.horn_link}'"
            for a in arch.actuators
        )
        or "- none"
    )
    # ADR-0130, law 2 (structure binds): the exact typed rows the plan must contain, per
    # actuator, with the joint identity to echo -- the deterministic actuation gate
    # reconciles precisely these. H4 measured the prose version insufficient.
    contract = "\n".join(
        f"- for {a.hw_id}: a fixed joint '<housing_part>.<port>' -> '{a.hw_id}.mount' "
        f"(the housing part must NOT be '{a.horn_link}' or anything rigidly fixed to it), "
        f"AND a revolute joint carrying name='{a.joint}' with a='{a.hw_id}.horn', "
        f"b='{a.horn_link}.<port>'. Do NOT declare '{a.hw_id}' in `parts` -- it is bought "
        "hardware the engine places."
        for a in arch.actuators
    )
    pairs = ", ".join("/".join(p) for p in arch.mirrored_pairs) or "none"
    return (
        "THE MECHANICAL ARCHITECTURE (BINDING -- every card realizes it in geometry; "
        "the deterministic gates audit exactly these decisions):\n"
        f"JOINTS:\n{joints}\n"
        "Every plan part named '*pin*' copies its joint's retention decision VERBATIM "
        "into the part's `retention` field; the card then MODELS it (L101 -- the field "
        "is gate-checked, prose is not).\n"
        f"ACTUATORS (bought, registry-mapped; house the body, seat the lugs, couple the "
        f"horn -- never friction on the case):\n{actuators}\n"
        + (f"PLAN CONTRACT per actuator (typed, gate-enforced):\n{contract}\n" if contract else "")
        + f"FASTENING SYSTEM: {arch.fastening_system}\n"
        f"SEAM STRATEGY: {arch.seam_strategy}\n"
        f"WIRING ROUTE: {arch.wiring_route}\n"
        f"MIRRORED PAIRS (exact sagittal mirrors): {pairs}"
    )


__all__ = [
    "ActuatorPlacement",
    "MechArchitecture",
    "MechJoint",
    "arch_context_block",
    "arch_transform",
    "hardware_decls_from",
    "plan_mech_architecture",
]
