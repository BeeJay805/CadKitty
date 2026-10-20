"""ComposablePlateInterpreter — ComposablePlateSpec -> PartProgram (ADR-0014).

ONE interpreter for the whole plate family. The EMIT is unified: place the base body
(``flat_plate`` or ``l_bracket_bend``), then loop the ``InterfacePlacement`` list calling
the existing ``place_*`` helpers, then emit the ``InterfaceDeclaration``s + metadata +
constraints. The GEOMETRY math (offsets, dims, feasibility, the L-bracket gusset) is
DELEGATED to the proven legacy resolvers (``adapter_interpreter`` / ``fan_mount`` /
``interpreter``), reused verbatim so a migrated pattern is byte-identical to its legacy
output and the proven benches cannot regress (ADR-0014 Q4/Q5).

New plate families that reuse a known interface combo are pure config; a genuinely new
combo needs explicit ``base_dims`` + offsets (zero new interpreter code) or, later, a new
derive (earned). The LLM only ever picks a menu; code derives every coordinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from forge_mind.dsl.components.registry import get_extrusion_component, get_motor_component
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.fan import FAN_BOLT_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.nema17 import BOLT_PITCH_MM as _NEMA17_BOLT_PITCH
from forge_mind.dsl.interfaces.nema17 import M3_CLEARANCE_RADIUS_MM as _M3_R
from forge_mind.dsl.interfaces.nema17 import SHAFT_BORE_RADIUS_MM as _SHAFT_R
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout import composable_hardware as _hw
from forge_mind.layout.adapter_interpreter import (
    _check_feasible as _adapter_check_feasible,
)
from forge_mind.layout.adapter_interpreter import (
    _resolve_adapter_offsets,
    _resolve_plate_dims,
)
from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.layout.envelope import grow_envelope
from forge_mind.layout.fan_mount import (
    _check_feasible as _fan_check_feasible,
)
from forge_mind.layout.fan_mount import (
    _resolve_offsets as _fan_resolve_offsets,
)
from forge_mind.layout.fan_mount import (
    _resolve_plate_dims as _fan_resolve_plate_dims,
)
from forge_mind.layout.fan_mount import (
    resolve_fan_facts,
)
from forge_mind.layout.interfaces import TNutPair, place_fan_face, place_m5_pair, place_nema17_face
from forge_mind.layout.interpreter import (
    InfeasibleRequestError,
    LayoutError,
    _gusset_op,
    _resolve_l_bracket_dims,
)
from forge_mind.layout.interpreter import (
    _check_feasible as _l_bracket_check_feasible,
)
from forge_mind.layout.placement import LayoutPattern, derive_positions
from forge_mind.layout.rules import edge_land
from forge_mind.layout.spec import (
    Extrusion2020InterfaceSpec,
    LBracketMotorMountSpec,
    Nema17InterfaceSpec,
)
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature

_DEFAULT_M5_SPACING_MM = 20.0
_SEP_MARGIN_MM = 6.0  # comfortable centre-to-centre slack beyond the no-overlap floor (derive)
# Float-noise tolerance for the derive-stage land/overlap pre-filter (NOT the validator): a
# derived plate sized to exactly min_wall land can read 4e-16 under it. The real no-false-PASS
# gate is the min_wall validator (with its own 0.2 mm tolerance); this only avoids float rejects.
_FEAS_EPS_MM = 1e-6
_DEFAULT_NAMES = {
    InterfaceRef.NEMA17_FACE: "motor_face",
    InterfaceRef.FAN_FACE: "fan_face",
    InterfaceRef.EXTRUSION_2020: "extrusion_face",
}
# The registry ids that ARE the legacy default hardware: a spec may name them explicitly
# instead of leaving component_id null, and that must behave identically (byte-equivalent)
# to null -- legacy delegate, no extra params (ADR-0017, Codex).
_LEGACY_MOTOR_ID = "nema17"
_LEGACY_EXTRUSION_ID = "ext_2020"

# In-plane (x, y) centre of a placement on the plate. Face/legacy paths always use y=0 (the
# historical centerline), so they stay byte-identical; only the patterned hardware path uses y != 0.
_Off = tuple[float, float]
# A bearing seat is the part's central axis -> it (and the face mounts) sit on the centerline and the
# mounting hardware arranges AROUND them. Everything else is "mounting" hardware (holes/inserts/etc).
_CENTRAL_HARDWARE = frozenset({InterfaceRef.BEARING_SEAT})


def _is_legacy_motor(component_id: str | None) -> bool:
    """True for the legacy NEMA 17 default (null OR an explicit ``nema17`` id)."""
    return component_id is None or component_id == _LEGACY_MOTOR_ID


def _is_legacy_extrusion(component_id: str | None) -> bool:
    """True for the legacy 2020 default (null OR an explicit ``ext_2020`` id)."""
    return component_id is None or component_id == _LEGACY_EXTRUSION_ID


def _is_default_spacing(ext: InterfacePlacement) -> bool:
    """The adapter/fan legacy derive + feasibility hardcode 20mm M5 spacing. Only DELEGATE
    to them when the spacing matches; a non-default spacing falls through to the generic
    explicit-dims path (which sizes + validates against the actual spacing), so a valid
    deep plate with e.g. 40mm spacing is accepted, not rejected (Codex)."""
    return ext.spacing_mm is None or ext.spacing_mm == _DEFAULT_M5_SPACING_MM


_LBRACKET_TO_FLAT_FACE = {Face.BASE_TOP: Face.PLATE_TOP, Face.UPRIGHT_FRONT: Face.PLATE_BOTTOM}


def _is_l_bracket_combo(spec: ComposablePlateSpec) -> bool:
    """True iff the spec is the ONE combo the l_bracket derive supports: a legacy NEMA 17 face + a
    legacy 2020 extrusion, exactly two interfaces (the right-angle motor mount). Any other combo on
    an l_bracket base is the LLM mis-picking the base -- the part is really a flat plate."""
    nema = _find(spec, InterfaceRef.NEMA17_FACE)
    ext = _find(spec, InterfaceRef.EXTRUSION_2020)
    return (
        nema is not None
        and ext is not None
        and len(spec.interfaces) == 2
        and _is_legacy_motor(nema.component_id)
        and _is_legacy_extrusion(ext.component_id)
    )


def _rebased_to_flat(spec: ComposablePlateSpec) -> ComposablePlateSpec:
    """Re-express a mis-picked l_bracket spec as the flat_plate it really is: base -> flat_plate and
    each l_bracket face mapped to the slab (base_top -> plate_top, upright_front -> plate_bottom;
    plate faces unchanged), so the flat_plate derive + grow build + validate it instead of rejecting."""
    interfaces = [
        p.model_copy(update={"face": _LBRACKET_TO_FLAT_FACE.get(p.face, p.face)})
        for p in spec.interfaces
    ]
    return spec.model_copy(update={"base": BaseArchetype.FLAT_PLATE, "interfaces": interfaces})


def interpret_composable_plate(spec: ComposablePlateSpec) -> PartProgram:
    """Deterministically realise a ComposablePlateSpec as a validated PartProgram.

    The base body is DERIVED from the interfaces (overriding the LLM, like the envelope): the
    l_bracket derive supports only the NEMA 17 + 2020 right-angle mount, so any other combo on an
    l_bracket base is RE-BASED to the flat_plate it actually is. Raises ``LayoutError`` only for a
    genuinely infeasible spec or an unsupported interface/face combo.
    """
    w = spec.min_wall_mm
    if spec.base is BaseArchetype.L_BRACKET_BEND and _is_l_bracket_combo(spec):
        return _interpret_l_bracket(spec, w)
    if spec.base is BaseArchetype.L_BRACKET_BEND:
        # The LLM picked l_bracket for a combo it cannot host (anything but NEMA 17 + 2020). The part
        # is really a flat plate with those interfaces -> RE-BASE to flat_plate (correct, don't reject
        # -> a silent fallback), remapping the l_bracket faces. The l_bracket derive already OWNS its
        # placement (it ignores the LLM's faces), so deriving the base here is the same spirit. The
        # re-based part is built with its OWN hardware (e.g. a real NEMA 23), never mis-built as NEMA 17.
        return _interpret_flat_plate(_rebased_to_flat(spec), w)
    walls = _BASE_WALLS.get(spec.base)
    if walls is not None:  # the walled-base grammar: flat_plate floor + a configurable wall set
        return _interpret_walled_base(spec, w, walls)
    if spec.base is BaseArchetype.ROUND_PLATE:
        return _interpret_round_plate(spec, w)
    if spec.base is BaseArchetype.FLAT_PLATE:
        return _interpret_flat_plate(spec, w)
    raise LayoutError(f"composable_plate: unsupported base archetype {spec.base!r}")


# ---------------------------------------------------------------------------
# flat_plate
# ---------------------------------------------------------------------------


def _find(spec: ComposablePlateSpec, ref: InterfaceRef) -> InterfacePlacement | None:
    return next((p for p in spec.interfaces if p.ref is ref), None)


def _flat_plate_geometry(spec: ComposablePlateSpec, w: float) -> tuple[float, float, list[_Off]]:
    """(plate_width, plate_depth, [(x, y) offset per placement]). Delegates offsets + dims +
    feasibility to the proven adapter/fan resolvers for known combos; a new combo needs
    explicit base_dims + every offset_mm (zero new derive code). Offsets are 2D: face/legacy
    paths return y=0 (the historical centerline, byte-identical); the composed-hardware path
    arranges mounting features in a 2D pattern (corner/grid/row/perimeter)."""
    bd = spec.base_dims
    nema = _find(spec, InterfaceRef.NEMA17_FACE)
    fan = _find(spec, InterfaceRef.FAN_FACE)
    ext = _find(spec, InterfaceRef.EXTRUSION_2020)
    bd_w = bd.width_mm if bd is not None else None
    bd_d = bd.depth_mm if bd is not None else None

    if (
        fan is not None
        and ext is not None
        and nema is None
        and len(spec.interfaces)
        == 2  # the delegate maps EXACTLY these two by id (ADR-0060 slice C)
        and _is_legacy_extrusion(ext.component_id)  # component ext (M6/3030) -> generic
        and _is_default_spacing(ext)
    ):
        legacy = AdapterPlateSpec(
            name=spec.name,
            envelope_mm=spec.envelope_mm,
            material=spec.material,
            min_wall_mm=w,
            max_overhang_deg=spec.max_overhang_deg,
            plate_width_mm=bd_w,
            plate_depth_mm=bd_d,
            motor_center_x_mm=fan.offset_mm,
            extrusion_center_x_mm=ext.offset_mm,
            fan_component_id=fan.component_id,
        )
        bolt_square, bore_r = resolve_fan_facts(fan.component_id or "")
        try:
            _fan_check_feasible(legacy, bolt_square, bore_r)
        except LayoutError:
            pass  # too small for the legacy fan fast-path -> fall through to the generic derive (grows)
        else:
            fan_x, ext_x = _fan_resolve_offsets(legacy, bolt_square, bore_r)
            width, depth = _fan_resolve_plate_dims(legacy, bolt_square, bore_r)
            chosen = {id(fan): fan_x, id(ext): ext_x}
            return width, depth, [(chosen[id(p)], 0.0) for p in spec.interfaces]

    if (
        nema is not None
        and ext is not None
        and fan is None
        and len(spec.interfaces)
        == 2  # the delegate maps EXACTLY these two by id (ADR-0060 slice C)
        and _is_legacy_motor(nema.component_id)  # NEMA 14/23 -> generic (per-component derive)
        and _is_legacy_extrusion(ext.component_id)
        and _is_default_spacing(ext)
    ):
        legacy = AdapterPlateSpec(
            name=spec.name,
            envelope_mm=spec.envelope_mm,
            material=spec.material,
            min_wall_mm=w,
            max_overhang_deg=spec.max_overhang_deg,
            plate_width_mm=bd_w,
            plate_depth_mm=bd_d,
            motor_center_x_mm=nema.offset_mm,
            extrusion_center_x_mm=ext.offset_mm,
        )
        try:
            _adapter_check_feasible(legacy)
        except LayoutError:
            pass  # too small for the legacy adapter fast-path -> fall through to the generic derive (grows)
        else:
            motor_x, ext_x = _resolve_adapter_offsets(legacy)
            width, depth = _resolve_plate_dims(legacy)
            chosen = {id(nema): motor_x, id(ext): ext_x}
            return width, depth, [(chosen[id(p)], 0.0) for p in spec.interfaces]

    # New / unknown combo (e.g. NEMA 23, 3030): no LEGACY delegate, so the geometry is
    # DERIVED here from each placement's per-component half-extent (ADR-0017) -- the LLM
    # leaves base_dims + offsets null and code derives them. base_dims is the mode selector:
    #   - null  -> derive dims AND offsets (the LLM path);
    #   - given -> config path: use the explicit dims, require every offset explicit.
    # Either way the generic edge-land/overlap/envelope check runs (validate before geometry).
    offsets_opt = [p.offset_mm for p in spec.interfaces]
    any_explicit = bd_w is not None or bd_d is not None or any(o is not None for o in offsets_opt)
    offsets: list[_Off]
    if not any_explicit:
        # Pure derive (the LLM path): code derives dims + offsets from per-component extents.
        width, depth, offsets = _derive_geometry(spec.interfaces, w, spec.layout)
    else:
        # Config path: explicit geometry must be COMPLETE -- never silently discard a partial
        # explicit dim/offset (Codex). All-or-nothing: width+depth AND every offset, or null. The
        # escape hatch stays a 1-D centerline (y=0); patterns are for the derive path.
        if bd_w is None or bd_d is None or any(o is None for o in offsets_opt):
            raise LayoutError(
                "composable flat_plate: partial explicit geometry is not supported; provide a "
                "COMPLETE layout (base_dims width_mm + depth_mm AND every placement offset_mm) "
                "or leave ALL of them null to derive."
            )
        width, depth, offsets = bd_w, bd_d, [(o, 0.0) for o in offsets_opt if o is not None]
    _check_flat_plate_feasible(spec, w, width, depth, offsets)
    return width, depth, offsets


def _derive_geometry(
    interfaces: list[InterfacePlacement], w: float, layout: LayoutPattern | None
) -> tuple[float, float, list[_Off]]:
    """Derive (width, depth, [(x, y) per placement]) for the pure-derive (LLM) path.

    FACE-only combos (NEMA / fan / extrusion, the 102-catalog and benches) keep the historical
    centerline ROW arrangement EXACTLY (``_derive_flat_geometry`` wrapped to y=0) -> byte-identical.
    When composed HARDWARE is present, mounting features arrange in a 2-D PATTERN (corner default)
    around any central feature (a bearing seat / a face mount), so a four-hole plate is a bolt
    pattern, not a line. A single feature stays centered -> hardware SINGLES are byte-identical too."""
    if not any(p.ref in _hw.HARDWARE_REFS for p in interfaces):
        width, depth, xs = _derive_flat_geometry(interfaces, w)  # unchanged face/row helper
        return width, depth, [(x, 0.0) for x in xs]
    return _derive_patterned(interfaces, w, layout)


def _footprint_reach(placement: InterfacePlacement, w: float) -> float:
    """The radius of the smallest disk that contains a feature's whole footprint, from its centre.

    A FACE mount (NEMA / fan) carries its bolt holes at the CORNERS of its square, so its real reach
    is the DIAGONAL (hypot), not the axis half-extent -- mounting features around it must clear those
    corner bolts, and a round rim must clear them too. A circular hardware seat / hole (a bearing,
    a clearance hole) reaches only its radius (hx == hy), so max() is exact there."""
    hx, hy = _interface_half_extent(placement, w)
    return max(hx, hy) if placement.ref in _hw.HARDWARE_REFS else math.hypot(hx, hy)


def _central_radius(placement: InterfacePlacement, x_offset: float, w: float) -> float:
    """Bounding in-plane radius of a centered/centerline feature from the origin (so mounting
    features arranged AROUND it actually clear it)."""
    return abs(x_offset) + _footprint_reach(placement, w)


def _round_reach(placement: InterfacePlacement, w: float) -> float:
    """Reach to the CIRCULAR rim (round-plate edge-land + disc sizing). A RECTANGULAR boss (the
    captive-nut pocket = a box) reaches its CORNER DIAGONAL, not its axis footprint, so the disc
    must clear that corner; circular hardware + face mounts keep the flat ``_footprint_reach``. This
    is round-specific on purpose -- the flat plate (straight edges, axis-aligned boxes) is unchanged
    and stays byte-identical."""
    if placement.ref in _hw.HARDWARE_REFS:
        return _hw.hardware_rim_reach(placement, w)
    return _footprint_reach(placement, w)


def _derive_patterned(
    interfaces: list[InterfacePlacement], w: float, layout: LayoutPattern | None
) -> tuple[float, float, list[_Off]]:
    """Lay central features (face mounts + a bearing seat) on the centerline and arrange the
    remaining mounting hardware in the chosen pattern around them. Deterministic; the existing
    edge-land / overlap check still gates validity."""
    central = [
        p for p in interfaces if p.ref not in _hw.HARDWARE_REFS or p.ref in _CENTRAL_HARDWARE
    ]
    central_ids = {id(p) for p in central}  # id, not value: identical placements must stay distinct
    peripheral = [p for p in interfaces if id(p) not in central_ids]
    pos: dict[int, _Off] = {}

    central_radius = 0.0
    if central:
        _cw, _cd, cxs = _derive_flat_geometry(central, w)  # current centerline spread (y=0)
        for p, x in zip(central, cxs, strict=True):
            pos[id(p)] = (x, 0.0)
            central_radius = max(central_radius, _central_radius(p, x, w))

    if peripheral:
        n_total = len(central) + len(peripheral)
        pattern = layout or (LayoutPattern.CENTERED if n_total <= 1 else LayoutPattern.CORNER)
        r_max = max(_interface_half_extent(p, w)[0] for p in peripheral)
        positions = derive_positions(
            pattern,
            len(peripheral),
            r_max,
            min_wall=w,
            central_radius=central_radius,
            slack=_SEP_MARGIN_MM,
        )
        for p, xy in zip(peripheral, positions, strict=True):
            pos[id(p)] = xy

    offsets = [pos[id(p)] for p in interfaces]
    extents = [_interface_half_extent(p, w) for p in interfaces]
    half_w = max(abs(ox) + hx for (ox, _oy), (hx, _hy) in zip(offsets, extents, strict=True)) + w
    half_d = max(abs(oy) + hy for (_ox, oy), (_hx, hy) in zip(offsets, extents, strict=True)) + w
    return (2.0 * half_w, 2.0 * half_d, offsets)


def _resolve_motor_facts(component_id: str | None) -> tuple[float, float, float]:
    """(bolt_pitch, bolt_clearance_radius, pilot_bore_radius) for a motor placement.
    ``None`` -> the legacy NEMA 17 facts (byte-identical default path, no registry lookup);
    an explicit id -> the MotorComponent (NEMA 14/23). Unknown id -> LayoutError (ADR-0017)."""
    if component_id is None:
        return (_NEMA17_BOLT_PITCH, _M3_R, _SHAFT_R)
    try:
        m = get_motor_component(component_id)
    except ValueError as exc:
        raise LayoutError(str(exc)) from None
    return (m.mount.bolt_pitch_mm, m.mount.bolt_clearance_radius_mm, m.mount.pilot_bore_radius_mm)


def _resolve_extrusion_facts(placement: InterfacePlacement) -> tuple[float, float]:
    """(bolt_clearance_radius, tnut_spacing) for an extrusion placement. ``component_id``
    None -> legacy M5 + 20 mm (byte-identical default); an explicit id -> the
    ExtrusionComponent (M6 + 30 mm for 3030). An explicit ``spacing_mm`` always wins, so a
    defaulted component extrusion records its OWN default spacing, not the legacy 20 mm."""
    if placement.component_id is None:
        clearance, default_spacing = M5_CLEARANCE_RADIUS_MM, _DEFAULT_M5_SPACING_MM
    else:
        try:
            e = get_extrusion_component(placement.component_id)
        except ValueError as exc:
            raise LayoutError(str(exc)) from None
        clearance, default_spacing = e.slot.bolt_clearance_radius_mm, e.slot.default_tnut_spacing_mm
    spacing = placement.spacing_mm if placement.spacing_mm is not None else default_spacing
    return clearance, spacing


def _interface_half_extent(
    placement: InterfacePlacement, min_wall: float = 0.0
) -> tuple[float, float]:
    """(x, y) in-plane half-extent of an interface's cluster from its centre -- used by
    the generic fallback feasibility (the proven combos use the legacy resolvers). Reads
    the per-component facts so a NEMA 23 / 3030 sizes correctly; the default (None) path
    reproduces the legacy NEMA 17 / M5 extents (byte-identical). Hardware features (composition
    engine) report their boss/hole footprint so the SAME collision/edge-land check covers them."""
    if placement.ref in _hw.HARDWARE_REFS:
        r = _hw.hardware_footprint_radius(placement, min_wall)
        return (r, r)
    if placement.ref is InterfaceRef.NEMA17_FACE:
        pitch, bolt_r, pilot_r = _resolve_motor_facts(placement.component_id)
        h = max(pitch / 2.0 + bolt_r, pilot_r)  # default NEMA 17: max(17.1, 11) = 17.1
        return (h, h)
    if placement.ref is InterfaceRef.FAN_FACE:
        bolt_square, bore_r = resolve_fan_facts(placement.component_id or "")
        h = max(bolt_square / 2.0 + FAN_BOLT_CLEARANCE_RADIUS_MM, bore_r)
        return (h, h)
    if placement.ref is InterfaceRef.EXTRUSION_2020:
        clearance, spacing = _resolve_extrusion_facts(placement)
        return (clearance, spacing / 2.0 + clearance)
    raise LayoutError(f"composable flat_plate: unsupported interface ref {placement.ref!r}")


def _derive_flat_geometry(
    interfaces: list[InterfacePlacement], min_wall_mm: float
) -> tuple[float, float, list[float]]:
    """Derive (plate_width, plate_depth, [offset per placement]) for a flat_plate from each
    placement's per-component half-extent (ADR-0017) -- the LLM-path 'code derives geometry'
    half for combos with no legacy delegate (NEMA 23, 3030, ...). Spreads the interfaces
    along X with a comfortable separation and sizes the plate so every interface keeps a
    >= min_wall land. The scaling engine's $0 probe (scaling/enumerate.py) reuses this, so
    the engine probes the SAME geometry the interpreter builds."""
    halfs = [_interface_half_extent(p, min_wall_mm) for p in interfaces]
    n = len(interfaces)
    if n == 1:
        offsets = [0.0]
    else:
        # Spread symmetrically: cumulative centres separated by (hx_i + hx_{i+1} + wall + slack).
        centres: list[float] = [0.0]
        for i in range(1, n):
            gap = halfs[i - 1][0] + halfs[i][0] + min_wall_mm + _SEP_MARGIN_MM
            centres.append(centres[-1] + gap)
        mid = (centres[0] + centres[-1]) / 2.0
        offsets = [c - mid for c in centres]
    half_w = max(abs(o) + hx for o, (hx, _hy) in zip(offsets, halfs, strict=True)) + min_wall_mm
    half_d = max(hy for _hx, hy in halfs) + min_wall_mm
    return (2.0 * half_w, 2.0 * half_d, offsets)


def _check_flat_plate_feasible(
    spec: ComposablePlateSpec, w: float, width: float, depth: float, offsets: list[_Off]
) -> None:
    """Edge-land (per interface, both axes) + pairwise no-overlap -- the part-INTERNAL lands the
    legacy interpreters guarantee. The envelope is NOT rejected here: it is an LLM estimate that the
    caller GROWS to the derived footprint (see ``envelope.grow_envelope``); only these internal
    lands are hard constraints (a real no-false-PASS check, unaffected by envelope growth).

    Offsets are 2-D (x, y). For the historical centerline paths (y=0) the X/Y edge tests and the
    centre-distance overlap reduce EXACTLY to the former 1-D arithmetic -> byte/behaviour identical;
    the patterned hardware path exercises the full 2-D form."""
    half_w, half_d = width / 2.0, depth / 2.0
    extents = [_interface_half_extent(p, w) for p in spec.interfaces]
    problems: list[str] = []
    for placement, (ox, oy), (hx, hy) in zip(spec.interfaces, offsets, extents, strict=True):
        if edge_land(half_w, abs(ox), hx) < w - _FEAS_EPS_MM:
            problems.append(
                f"{placement.ref.value} at ({ox:.1f},{oy:.1f}): X hole-to-edge land < min_wall {w}mm."
            )
        if edge_land(half_d, abs(oy), hy) < w - _FEAS_EPS_MM:
            problems.append(
                f"{placement.ref.value} at ({ox:.1f},{oy:.1f}): Y hole-to-edge land < min_wall {w}mm."
            )
        if placement.ref is InterfaceRef.FAN_FACE:
            # Fan-internal: the airflow bore must keep >= min_wall to the corner bolts (ADR-0011).
            # This is a GENUINE physical infeasibility of the fan itself -- no envelope grow or layout
            # change fixes it -> refuse honestly (InfeasibleRequestError), never a fallback that
            # builds a wrong part.
            bolt_square, bore_r = resolve_fan_facts(placement.component_id or "")
            land = (bolt_square / 2.0) * math.sqrt(2) - FAN_BOLT_CLEARANCE_RADIUS_MM - bore_r
            if land < w:
                raise InfeasibleRequestError(
                    f"this fan's airflow bore leaves only {land:.2f}mm to its corner bolts at "
                    f"{w:.0f}mm min wall (need >= {w:.0f}mm) -- physically too small to build."
                )
    n = len(spec.interfaces)
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(offsets[i][0] - offsets[j][0], offsets[i][1] - offsets[j][1])
            gap = dist - (extents[i][0] + extents[j][0])
            if gap < w - _FEAS_EPS_MM:
                problems.append(
                    f"{spec.interfaces[i].ref.value}/{spec.interfaces[j].ref.value} overlap: "
                    f"land {gap:.2f}mm < min_wall {w}mm."
                )
    if problems:
        raise LayoutError("infeasible composable flat_plate spec: " + "; ".join(problems))


def _flat_plate_required(width: float, depth: float, w: float, ops: list[Operation]) -> Vec3:
    """The part's true bbox extent, for envelope growth: the plate footprint (width x depth -- every
    interface sits within it with a min_wall land) plus the tallest +Z feature (hardware bosses rise
    above the plate; the part is Z-stacked up from z=0, so the extent is the highest ADD-op top)."""
    tops = [
        op.position.z + op.height / 2.0
        for op in ops
        if op.mode is OpMode.ADD and isinstance(op, (Box, Cylinder))
    ]
    return Vec3(x=width, y=depth, z=max(tops, default=w))


def _interpret_flat_plate(spec: ComposablePlateSpec, w: float) -> PartProgram:
    # Same-radius coexistence (NEMA 23 M5 + 2020/2040 M5, dual same-size extrusion) no longer
    # needs an up-front guard: the extrusion checker is now localized (ADR-0017), so each
    # interface's holes are counted in its own region. The only residual is the exotic
    # mixed-tiny-spacing same-radius case (a conservative false-fail, pinned in
    # test_interface_localization.py) -- never a mis-certify.
    width, depth, offsets = _flat_plate_geometry(spec, w)
    plate_mid_z = w / 2.0
    plate_top_z = w

    ops: list[Operation] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=plate_mid_z),
            width=width,
            depth=depth,
            height=w,
        )
    ]
    feat_ops, decls, hw_features = _emit_placements(
        spec.interfaces, offsets, w, plate_mid_z, plate_top_z
    )
    ops.extend(feat_ops)
    # GROW the (LLM-estimated) envelope to the derived footprint instead of rejecting it; honour a
    # user hard max if one was plumbed onto the spec. The part still validates against this bbox.
    eff_env = grow_envelope(
        spec.envelope_mm, _flat_plate_required(width, depth, w, ops), spec.max_envelope_mm
    )
    return _program(spec, w, ops, decls, hw_features, envelope=eff_env)


def _emit_placements(
    interfaces: list[InterfacePlacement], offsets: list[_Off], w: float, mid_z: float, top_z: float
) -> tuple[list[Operation], list[InterfaceDeclaration], list[HardwareFeature]]:
    """Place every interface at its derived (x, y) on a flat floor top: the place_* ops, the face
    declarations (verified by the interface tier), and the registry HardwareFeatures (verified by the
    A4 conformance check). Shared by the flat plate and the round plate -- the floor SHAPE differs, the
    feature placement is identical (everything sits on the floor top). Repeated refs get suffixed
    op-ids/names (the first occurrence keeps the legacy ids so single-ref migrations stay byte-identical)."""
    ops: list[Operation] = []
    decls: list[InterfaceDeclaration] = []
    hw_features: list[HardwareFeature] = []
    seen: dict[InterfaceRef, int] = {}
    for placement, off in zip(interfaces, offsets, strict=True):
        occurrence = seen.get(placement.ref, 0)
        seen[placement.ref] = occurrence + 1
        p_ops, decl = _place_on_flat(placement, off, w, mid_z, top_z, occurrence)
        ops.extend(p_ops)
        if decl is not None:  # a face interface -> declaration (verified by the interface tier)
            decls.append(decl)
        elif placement.ref in _hw.HARDWARE_REFS:
            # A hardware feature emits no face declaration; it is verified by the A4 conformance check,
            # so carry its (category, key, fit, built dimension) -- the SAME registry value the geometry
            # used, so it conforms by construction. This is what makes the conformance gate actually run.
            cat, key, fit, nominal = _hw.hardware_feature_of(placement)
            hw_features.append(
                HardwareFeature(
                    category=cat,
                    key=key,
                    fit=fit,
                    nominal_mm=nominal,
                    where=placement.name or placement.ref.value,
                )
            )
    return ops, decls, hw_features


# ---------------------------------------------------------------------------
# round_plate — a CIRCULAR floor (a round plate / disc mount / flange / lid). The round analog of
# the flat plate: it hosts the SAME hardware vocabulary (place_hardware_on_flat + A4 conformance are
# floor-shape-agnostic) and composes through the engine the same way. Two genuinely-new pieces: a
# CIRCULAR edge-land rule (distance to the round rim, min_wall margin) and a BOLT-CIRCLE layout
# pattern (the round analog of the 2x2 corner). The floor is a Cylinder (reusing the primitive).
# ---------------------------------------------------------------------------


def _derive_round_positions(
    interfaces: list[InterfacePlacement], w: float, layout: LayoutPattern | None
) -> list[_Off]:
    """In-plane (x, y) centres for a round plate: a single primary (a bearing seat / a face mount)
    sits centered; the remaining mounting features arrange on a BOLT CIRCLE around it (the round
    default). Mirrors the flat plate's central/peripheral split; only the default pattern differs."""
    central = [
        p for p in interfaces if p.ref not in _hw.HARDWARE_REFS or p.ref in _CENTRAL_HARDWARE
    ]
    pos: dict[int, _Off] = {}
    if len(central) == 1:  # one primary -> centered; everything else rings around it
        center_p = central[0]
        peripheral = [p for p in interfaces if p is not center_p]
        central_radius = _footprint_reach(center_p, w)
        pos[id(center_p)] = (0.0, 0.0)
    else:  # no single primary (or several) -> arrange everything on the ring
        peripheral = list(interfaces)
        central_radius = 0.0
    if peripheral:
        n_total = len(interfaces)
        pattern = layout or (LayoutPattern.CENTERED if n_total <= 1 else LayoutPattern.BOLT_CIRCLE)
        r_max = max(_interface_half_extent(p, w)[0] for p in peripheral)
        positions = derive_positions(
            pattern,
            len(peripheral),
            r_max,
            min_wall=w,
            central_radius=central_radius,
            slack=_SEP_MARGIN_MM,
        )
        for p, xy in zip(peripheral, positions, strict=True):
            pos[id(p)] = xy
    return [pos[id(p)] for p in interfaces]


def _check_round_feasible(
    spec: ComposablePlateSpec, w: float, plate_radius: float, offsets: list[_Off]
) -> None:
    """The CIRCULAR edge-land + pairwise no-overlap -- the round analog of _check_flat_plate_feasible,
    a real no-false-PASS gate (never false-pass a rim-crowded feature). Each feature must keep
    >= min_wall between its outermost point (centre distance + footprint reach) and the round rim;
    pairwise centres must not crowd; a fan's internal bore-to-bolt land is still a hard infeasibility."""
    # The RIM land uses the corner-aware reach (a rectangular boss reaches its diagonal); the
    # pairwise check below keeps the circular footprint reach (feature-to-feature spacing, unchanged).
    rim_reaches = [_round_reach(p, w) for p in spec.interfaces]
    reaches = [_footprint_reach(p, w) for p in spec.interfaces]
    problems: list[str] = []
    for placement, (ox, oy), reach in zip(spec.interfaces, offsets, rim_reaches, strict=True):
        rim_land = plate_radius - (math.hypot(ox, oy) + reach)
        if rim_land < w - _FEAS_EPS_MM:
            problems.append(
                f"{placement.ref.value} at ({ox:.1f},{oy:.1f}): rim land {rim_land:.2f}mm < min_wall {w}mm."
            )
        if placement.ref is InterfaceRef.FAN_FACE:
            bolt_square, bore_r = resolve_fan_facts(placement.component_id or "")
            land = (bolt_square / 2.0) * math.sqrt(2) - FAN_BOLT_CLEARANCE_RADIUS_MM - bore_r
            if land < w:
                raise InfeasibleRequestError(
                    f"this fan's airflow bore leaves only {land:.2f}mm to its corner bolts at "
                    f"{w:.0f}mm min wall (need >= {w:.0f}mm) -- physically too small to build."
                )
    n = len(spec.interfaces)
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(offsets[i][0] - offsets[j][0], offsets[i][1] - offsets[j][1])
            gap = dist - (reaches[i] + reaches[j])
            if gap < w - _FEAS_EPS_MM:
                problems.append(
                    f"{spec.interfaces[i].ref.value}/{spec.interfaces[j].ref.value} overlap: "
                    f"land {gap:.2f}mm < min_wall {w}mm."
                )
    if problems:
        raise LayoutError("infeasible composable round_plate spec: " + "; ".join(problems))


def _interpret_round_plate(spec: ComposablePlateSpec, w: float) -> PartProgram:
    """A round plate: a circular floor (a Cylinder) that hosts the hardware exactly like the flat
    plate, but sized + gated by a CIRCULAR edge-land and laid out on a bolt circle by default."""
    offsets = _derive_round_positions(spec.interfaces, w, spec.layout)
    # Corner-aware reach: a rectangular boss (captive-nut pocket) must clear the rim by its DIAGONAL,
    # not its axis footprint (Fix 3); circular features are unchanged, so non-captive-nut round parts
    # stay bit-identical.
    extents = [_round_reach(p, w) for p in spec.interfaces]
    # Size the disc so every feature keeps a >= min_wall land to the rim (the grow-don't-reject rule).
    plate_radius = (
        max(
            (math.hypot(ox, oy) + reach for (ox, oy), reach in zip(offsets, extents, strict=True)),
            default=0.0,
        )
        + w
    )
    _check_round_feasible(spec, w, plate_radius, offsets)
    plate_mid_z, plate_top_z = w / 2.0, w
    ops: list[Operation] = [
        Cylinder(
            op_id="plate",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=plate_mid_z),
            radius=plate_radius,
            height=w,
        )
    ]
    feat_ops, decls, hw_features = _emit_placements(
        spec.interfaces, offsets, w, plate_mid_z, plate_top_z
    )
    ops.extend(feat_ops)
    diameter = 2.0 * plate_radius
    feat_top = max(
        (
            op.position.z + op.height / 2.0
            for op in ops
            if op.mode is OpMode.ADD and isinstance(op, (Box, Cylinder))
        ),
        default=w,
    )
    eff_env = grow_envelope(
        spec.envelope_mm, Vec3(x=diameter, y=diameter, z=feat_top), spec.max_envelope_mm
    )
    return _program(spec, w, ops, decls, hw_features, envelope=eff_env)


# ---------------------------------------------------------------------------
# Walled-base grammar — the flat_plate floor + a configurable set of walls/uprights.
# open_box, angle_bracket, U-channel, tray, partial enclosure are all DATA (a WallConfig),
# interpreted by ONE function. The floor is always _interpret_flat_plate (unchanged), so the full
# hardware vocabulary + layout patterns + edge-land + A4 conformance apply to every walled base for
# free. A feature kept >= min_wall off the floor edge automatically clears every wall (walls sit on
# / outside the floor edge). Byte-equivalence is preserved EXACTLY: a full perimeter (all 4 edges)
# emits the watertight outer-minus-cavity ring (op-ids box_outer/box_cavity) the legacy open_box
# emitted; a partial wall set emits additive edge boxes (the legacy angle_bracket upright, op-id
# bracket_upright + its M5 flange). Only the floor-slab vs ring encoding differs by mode.
# ---------------------------------------------------------------------------


class _Edge(StrEnum):
    """The four in-plane edges of the rectangular floor a wall can sit on."""

    REAR = "rear"  # +Y
    FRONT = "front"  # -Y
    LEFT = "left"  # -X
    RIGHT = "right"  # +X


_EDGE_ORDER = (_Edge.REAR, _Edge.FRONT, _Edge.LEFT, _Edge.RIGHT)  # deterministic op-emit order
_ALL_EDGES = frozenset(_Edge)
# Below this floor width an M5 mounting pair cannot keep min_wall in a flange -> a plain wall.
_FLANGE_MIN_WIDTH_MM = 34.0


@dataclass(frozen=True)
class WallConfig:
    """A base body AS DATA: which floor edges carry a wall, the wall height, an optional default
    mounting flange, and op-id overrides for byte-equivalence with the legacy bespoke interpreters.

    ``wall_height_mm`` is the wall extent ABOVE the floor in full-ring mode (open_box: a wall ring
    z in [w, w+h]); in per-edge mode it is the wall's TOTAL height from the bed (the angle_bracket
    upright z in [0, h]). The two modes differ only in how the floor is encoded (a full ring is the
    watertight outer-minus-cavity solid; partial walls are additive boxes on a bare slab) -- both
    reuse the SAME flat_plate floor + features."""

    edges: frozenset[_Edge]
    wall_height_mm: float
    flange_edge: _Edge | None = None
    edge_op_ids: dict[_Edge, str] = field(default_factory=dict)


# The base-body registry: each entry is pure DATA. New prismatic bodies are added here, not in code.
_BASE_WALLS: dict[BaseArchetype, WallConfig] = {
    # open_box: a watertight perimeter ring 15mm above the floor (the legacy open_box, byte-identical).
    BaseArchetype.OPEN_BOX: WallConfig(edges=_ALL_EDGES, wall_height_mm=15.0),
    # angle_bracket: a single rear upright flange (the legacy angle_bracket, byte-identical via the
    # bracket_upright op-id + the default M5 flange pair).
    BaseArchetype.ANGLE_BRACKET: WallConfig(
        edges=frozenset({_Edge.REAR}),
        wall_height_mm=30.0,
        flange_edge=_Edge.REAR,
        edge_op_ids={_Edge.REAR: "bracket_upright"},
    ),
    # New prismatic bodies -- PURE DATA, zero new interpreter code (the grammar above realises them):
    BaseArchetype.U_CHANNEL: WallConfig(  # a channel / rail: two opposite side walls
        edges=frozenset({_Edge.LEFT, _Edge.RIGHT}), wall_height_mm=20.0
    ),
    BaseArchetype.TRAY: WallConfig(  # a deeper 4-wall ring than open_box (config-only variation)
        edges=_ALL_EDGES, wall_height_mm=30.0
    ),
    BaseArchetype.PARTIAL_ENCLOSURE: WallConfig(  # three walls, open on the front
        edges=frozenset({_Edge.REAR, _Edge.LEFT, _Edge.RIGHT}), wall_height_mm=25.0
    ),
}


def _edge_wall(
    edge: _Edge,
    width: float,
    depth: float,
    w: float,
    height: float,
    op_id: str,
    edges: frozenset[_Edge],
) -> Box:
    """One additive wall box on a floor edge, OUTSIDE the floor footprint, z in [0, height]. A
    rear/front wall EXTENDS in X to overlap any present left/right wall (and a left/right wall extends
    in Y toward a present rear/front wall), so ADJACENT walls fuse into a watertight corner instead of
    leaving a gap. A lone wall (e.g. the angle_bracket rear upright) gets no extension -> it stays
    byte-identical to the legacy single-wall geometry."""
    if edge in (_Edge.REAR, _Edge.FRONT):
        x_lo = -width / 2.0 - (w if _Edge.LEFT in edges else 0.0)
        x_hi = width / 2.0 + (w if _Edge.RIGHT in edges else 0.0)
        y = (depth / 2.0 + w / 2.0) * (1.0 if edge is _Edge.REAR else -1.0)
        return Box(
            op_id=op_id,
            mode=OpMode.ADD,
            position=Vec3(x=(x_lo + x_hi) / 2.0, y=y, z=height / 2.0),
            width=x_hi - x_lo,
            depth=w,
            height=height,
        )
    y_lo = -depth / 2.0 - (w if _Edge.FRONT in edges else 0.0)
    y_hi = depth / 2.0 + (w if _Edge.REAR in edges else 0.0)
    x = (width / 2.0 + w / 2.0) * (-1.0 if edge is _Edge.LEFT else 1.0)
    return Box(
        op_id=op_id,
        mode=OpMode.ADD,
        position=Vec3(x=x, y=(y_lo + y_hi) / 2.0, z=height / 2.0),
        width=w,
        depth=y_hi - y_lo,
        height=height,
    )


def _edge_flange(edge: _Edge, depth: float, w: float, height: float) -> list[Operation]:
    """A default M5 mounting pair drilled through a rear/front wall (spread along X, drilled in Y) --
    the legacy angle_bracket flange. Only the rear/front axis is supported; a left/right flange would
    drill in X and is not needed by any current base."""
    if edge not in (_Edge.REAR, _Edge.FRONT):
        raise LayoutError(
            f"walled_base: a mounting flange is only supported on rear/front, not {edge}"
        )
    y = (depth / 2.0 + w / 2.0) * (1.0 if edge is _Edge.REAR else -1.0)
    return place_m5_pair(
        Vec3(x=0.0, y=y, z=(w + height) / 2.0),
        _DEFAULT_M5_SPACING_MM,
        w,
        orientation=TNutPair.ALONG_X_DRILL_Y,
        left_op_id="flange_bolt_left",
        right_op_id="flange_bolt_right",
        clearance_radius_mm=M5_CLEARANCE_RADIUS_MM,
    )


def _walled_required(
    width: float, depth: float, w: float, config: WallConfig, feat_top: float
) -> Vec3:
    """The part bbox for a PER-EDGE walled base: the floor grown by a wall thickness on each walled
    edge, and tall enough for the walls + any feature boss."""
    x = (
        width
        + (w if _Edge.LEFT in config.edges else 0.0)
        + (w if _Edge.RIGHT in config.edges else 0.0)
    )
    y = (
        depth
        + (w if _Edge.REAR in config.edges else 0.0)
        + (w if _Edge.FRONT in config.edges else 0.0)
    )
    return Vec3(x=x, y=y, z=max(config.wall_height_mm, feat_top))


def _interpret_walled_base(spec: ComposablePlateSpec, w: float, config: WallConfig) -> PartProgram:
    """Realise a walled base from its WallConfig: the flat_plate floor + the configured walls. The
    floor + every feature is byte-identical to the flat plate the same spec would build (additive)."""
    floor_prog = _interpret_flat_plate(spec, w)
    ops = list(floor_prog.operations)
    plate = ops[0]
    if not isinstance(plate, Box):  # defensive: the flat_plate floor is always op 0 (the slab)
        raise LayoutError("walled_base: expected a flat_plate floor slab as the first operation")
    width, depth = plate.width, plate.depth
    features = ops[1:]
    feat_top = _flat_plate_required(width, depth, w, ops).z  # tallest +Z feature (may exceed walls)

    if config.edges == _ALL_EDGES:
        # FULL PERIMETER: the watertight ring -- the floor extends under the walls (full footprint)
        # and the wall ring is an outer solid minus the top cavity (clean corners). Byte-identical to
        # the legacy open_box: op-ids box_outer / box_cavity, the floor slab folded into box_outer.
        wall_h = config.wall_height_mm
        height = w + wall_h
        fw, fd = width + 2.0 * w, depth + 2.0 * w
        base_ops: list[Operation] = [
            Box(
                op_id="box_outer",
                mode=OpMode.ADD,
                position=Vec3(x=0.0, y=0.0, z=height / 2.0),
                width=fw,
                depth=fd,
                height=height,
            ),
            Box(
                op_id="box_cavity",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=0.0, y=0.0, z=w + wall_h / 2.0),
                width=width,
                depth=depth,
                height=wall_h,
            ),
        ]
        new_ops = base_ops + features
        required = Vec3(x=fw, y=fd, z=max(height, feat_top))
    else:
        # PARTIAL WALLS: the bare flat slab + additive edge walls (open form). Byte-identical to the
        # legacy angle_bracket when edges={rear} + flange (op-id bracket_upright + the M5 flange pair).
        walls = [
            _edge_wall(
                e,
                width,
                depth,
                w,
                config.wall_height_mm,
                config.edge_op_ids.get(e, f"wall_{e.value}"),
                config.edges,
            )
            for e in _EDGE_ORDER
            if e in config.edges
        ]
        new_ops = [plate, *walls, *features]
        if config.flange_edge is not None and width >= _FLANGE_MIN_WIDTH_MM:
            new_ops += _edge_flange(config.flange_edge, depth, w, config.wall_height_mm)
        required = _walled_required(width, depth, w, config, feat_top)

    eff_env = grow_envelope(spec.envelope_mm, required, spec.max_envelope_mm)
    return _program(
        spec,
        w,
        new_ops,
        list(floor_prog.interfaces),
        list(floor_prog.hardware_features),
        envelope=eff_env,
    )


def _suffixed(base: str, occurrence: int) -> str:
    """Legacy op-id prefix for the first occurrence of a ref; suffixed for repeats
    (``motor_bolt`` -> ``motor_bolt_2`` for the second). Preserves byte-equivalence:
    every migrated pattern is single-ref, so occurrence is always 0 -> legacy id."""
    return base if occurrence == 0 else f"{base}_{occurrence + 1}"


def _placement_name(placement: InterfacePlacement, occurrence: int) -> str:
    """An explicit name wins; else the default, suffixed only on a repeated ref so each
    interface gets a distinct declaration name (and thus distinct check ids)."""
    if placement.name is not None:
        return placement.name
    base = _DEFAULT_NAMES[placement.ref]
    return base if occurrence == 0 else f"{base}_{occurrence + 1}"


def _place_on_flat(
    placement: InterfacePlacement,
    off: _Off,
    w: float,
    mid_z: float,
    top_z: float,
    occurrence: int = 0,
) -> tuple[list[Operation], InterfaceDeclaration | None]:
    """Emit the placement ops + declaration for one interface on a flat plate. The face
    sets the declaration plane/normal; the ref sets the place_* call + interface type;
    ``occurrence`` namespaces op-ids/name for a repeated ref (0 = legacy ids). ``off`` is the
    in-plane (x, y) centre (face mounts are always y=0 -> byte-identical to the former x-only path).

    Hardware features (composition engine) emit geometry but NO InterfaceDeclaration -- they are
    verified by the A4 conformance check, not a face checker -- so they return (ops, None)."""
    ox, oy = off
    if placement.ref in _hw.HARDWARE_REFS:
        return _hw.place_hardware_on_flat(placement, off, w, occurrence), None
    center = Vec3(x=ox, y=oy, z=mid_z)
    if placement.face is Face.PLATE_TOP:
        decl_loc, decl_normal = Vec3(x=ox, y=oy, z=top_z), Vec3(x=0.0, y=0.0, z=1.0)
    elif placement.face is Face.PLATE_BOTTOM:
        decl_loc, decl_normal = Vec3(x=ox, y=oy, z=0.0), Vec3(x=0.0, y=0.0, z=-1.0)
    else:
        raise LayoutError(f"face {placement.face!r} is not valid on a flat_plate base")
    name = _placement_name(placement, occurrence)

    if placement.ref is InterfaceRef.NEMA17_FACE:
        pitch, bolt_r, pilot_r = _resolve_motor_facts(placement.component_id)
        ops = place_nema17_face(
            center,
            w,
            bolt_op_prefix=_suffixed("motor_bolt", occurrence),
            shaft_op_id=_suffixed("shaft_clearance", occurrence),
            bolt_pitch_mm=pitch,
            bolt_clearance_radius_mm=bolt_r,
            pilot_bore_radius_mm=pilot_r,
        )
        # Omit params for the legacy default (null OR an explicit nema17 id) so the
        # declaration stays byte-identical; a registry motor (NEMA 14/23) carries its facts
        # so the checker certifies the right pattern (ADR-0017 omit-param-for-default).
        motor_params: dict[str, float | int | str | bool] = (
            {}
            if _is_legacy_motor(placement.component_id)
            else {
                "bolt_pitch": pitch,
                "bolt_clearance_radius": bolt_r,
                "pilot_bore_radius": pilot_r,
            }
        )
        decl = InterfaceDeclaration(
            name=name,
            interface_type=InterfaceType.NEMA17_FACE,
            location=decl_loc,
            normal=decl_normal,
            params=motor_params,
        )
        return ops, decl
    if placement.ref is InterfaceRef.FAN_FACE:
        bolt_square, bore_r = resolve_fan_facts(placement.component_id or "")
        ops = place_fan_face(
            center,
            w,
            bolt_square_mm=bolt_square,
            airflow_bore_radius_mm=bore_r,
            bolt_op_prefix=_suffixed("fan_bolt", occurrence),
            bore_op_id=_suffixed("airflow_bore", occurrence),
        )
        decl = InterfaceDeclaration(
            name=name,
            interface_type=InterfaceType.FAN_FACE,
            location=decl_loc,
            normal=decl_normal,
            params={"bolt_square": bolt_square, "airflow_bore_radius": bore_r},
        )
        return ops, decl
    if placement.ref is InterfaceRef.EXTRUSION_2020:
        clearance, spacing = _resolve_extrusion_facts(placement)
        ops = place_m5_pair(
            center,
            spacing,
            w,
            orientation=TNutPair.ALONG_Y_DRILL_Z,
            left_op_id=_suffixed("ext_bolt_back", occurrence),
            right_op_id=_suffixed("ext_bolt_front", occurrence),
            clearance_radius_mm=clearance,
        )
        # spacing is always present (legacy); add bolt_clearance_radius only for a non-legacy
        # extrusion (M6/3030) so the default 2020 declaration stays byte-identical (ADR-0017).
        ext_params: dict[str, float | int | str | bool] = {"spacing": spacing}
        if not _is_legacy_extrusion(placement.component_id):
            ext_params["bolt_clearance_radius"] = clearance
        decl = InterfaceDeclaration(
            name=name,
            interface_type=InterfaceType.EXTRUSION_2020_SLOT,
            location=decl_loc,
            normal=decl_normal,
            params=ext_params,
        )
        return ops, decl
    raise LayoutError(f"composable flat_plate: unsupported interface ref {placement.ref!r}")


# ---------------------------------------------------------------------------
# l_bracket_bend
# ---------------------------------------------------------------------------


def _interpret_l_bracket(spec: ComposablePlateSpec, w: float) -> PartProgram:
    nema = _find(spec, InterfaceRef.NEMA17_FACE)
    ext = _find(spec, InterfaceRef.EXTRUSION_2020)
    if nema is None or ext is None or len(spec.interfaces) != 2:
        raise LayoutError(
            "composable l_bracket_bend supports exactly a nema17_face (base) + an "
            "extrusion_2020 (upright); other combos are not yet a proven derive."
        )
    if not _is_legacy_motor(nema.component_id) or not _is_legacy_extrusion(ext.component_id):
        raise LayoutError(
            "composable l_bracket_bend does not yet support non-legacy component hardware "
            "(NEMA 14/23, 2040/3030); only the NEMA 17 + 2020 combo. Use a flat_plate base "
            "for other hardware (ADR-0017)."
        )
    spacing = ext.spacing_mm or _DEFAULT_M5_SPACING_MM
    bd = spec.base_dims

    # Construct the legacy spec so feasibility + dims + the gusset are the PROVEN code.
    from forge_mind.layout.spec import BaseDims, UprightDims

    explicit_base = (
        BaseDims(width_mm=bd.base_width_mm, depth_mm=bd.base_depth_mm)
        if bd is not None and bd.base_width_mm is not None and bd.base_depth_mm is not None
        else None
    )
    explicit_up = (
        UprightDims(width_mm=bd.upright_width_mm, height_mm=bd.upright_height_mm)
        if bd is not None and bd.upright_width_mm is not None and bd.upright_height_mm is not None
        else None
    )
    legacy = LBracketMotorMountSpec(
        name=spec.name,
        envelope_mm=spec.envelope_mm,
        material=spec.material,
        min_wall_mm=w,
        max_overhang_deg=spec.max_overhang_deg,
        base=explicit_base,
        upright=explicit_up,
        nema17=Nema17InterfaceSpec(name=nema.name or "motor_face"),
        extrusion_2020=Extrusion2020InterfaceSpec(
            name=ext.name or "extrusion_face", m5_spacing_mm=spacing
        ),
    )
    # GROW the envelope to the derived l_bracket bbox instead of rejecting an under-sized one (the
    # dims are envelope-independent). A feasible envelope is >= the bbox, so grown == original ->
    # byte-equ (the gusset + max_bbox keep using it); a user hard max is honoured.
    base, up = _resolve_l_bracket_dims(legacy)
    bbox = Vec3(x=max(base.width_mm, up.width_mm), y=base.depth_mm, z=max(up.height_mm, w))
    grown_env = grow_envelope(spec.envelope_mm, bbox, spec.max_envelope_mm)
    legacy = legacy.model_copy(update={"envelope_mm": grown_env})
    _l_bracket_check_feasible(legacy)

    rear_y = base.depth_mm / 2.0
    ext_y_center = rear_y - w / 2.0
    ext_front_y = rear_y - w
    hole_z = up.height_mm / 2.0

    ops: list[Operation] = [
        Box(
            op_id="motor_flange",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=w / 2.0),
            width=base.width_mm,
            depth=base.depth_mm,
            height=w,
        ),
        Box(
            op_id="ext_flange",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=ext_y_center, z=up.height_mm / 2.0),
            width=up.width_mm,
            depth=w,
            height=up.height_mm,
        ),
    ]
    gusset = _gusset_op(legacy, ext_front_y)
    if gusset is not None:
        ops.append(gusset)

    # CANONICAL placement: an L-bracket motor mount has exactly one sensible arrangement
    # -- the motor on the base (+Z), the extrusion on the upright (+Y). That face mapping
    # is a fact of the archetype, NOT a real DOF, so the interpreter owns it (ADR-0009:
    # code derives relational placement). We ignore the face the LLM picked for these two
    # interfaces: the reliability probe showed the LLM sometimes swaps them, and there is
    # no valid swapped L-bracket to honour. Order is fixed (nema then m5) for byte-equiv.
    ops.extend(place_nema17_face(Vec3(x=0.0, y=0.0, z=w / 2.0), w))
    ops.extend(
        place_m5_pair(
            Vec3(x=0.0, y=ext_y_center, z=hole_z),
            spacing,
            w,
            orientation=TNutPair.ALONG_X_DRILL_Y,
        )
    )
    decls = [
        InterfaceDeclaration(
            name=nema.name or "motor_face",
            interface_type=InterfaceType.NEMA17_FACE,
            location=Vec3(x=0.0, y=0.0, z=w / 2.0),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
        ),
        InterfaceDeclaration(
            name=ext.name or "extrusion_face",
            interface_type=InterfaceType.EXTRUSION_2020_SLOT,
            location=Vec3(x=0.0, y=ext_y_center, z=hole_z),
            normal=Vec3(x=0.0, y=1.0, z=0.0),
            params={"spacing": spacing},
        ),
    ]
    return _program(spec, w, ops, decls, envelope=grown_env)


# ---------------------------------------------------------------------------
# shared emit
# ---------------------------------------------------------------------------


def _program(
    spec: ComposablePlateSpec,
    w: float,
    ops: list[Operation],
    decls: list[InterfaceDeclaration],
    hw_features: list[HardwareFeature] | None = None,
    envelope: Vec3 | None = None,
) -> PartProgram:
    metadata = Metadata(
        name=spec.name,
        units="mm",
        print_orientation=Vec3(x=0.0, y=0.0, z=1.0),  # the plate family always prints +Z up
    )
    # max_bbox is the GROWN envelope when the caller derived one (flat_plate); the l_bracket path
    # passes None -> the spec envelope, unchanged (byte-identical).
    constraints = Constraints(
        min_wall_thickness=w,
        max_overhang_deg=spec.max_overhang_deg,
        max_bbox=envelope if envelope is not None else spec.envelope_mm,
    )
    # hardware_features defaults to [] (the l_bracket path + every face-only plate), so those parts
    # serialize byte-identically to before this field existed.
    return PartProgram(
        metadata=metadata,
        operations=ops,
        interfaces=decls,
        hardware_features=hw_features or [],
        constraints=constraints,
    )


__all__ = ["interpret_composable_plate"]
