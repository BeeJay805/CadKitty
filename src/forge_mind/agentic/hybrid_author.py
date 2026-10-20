"""T4 (Fable ruling §A): hybrid physics-authoring inside the functional pipeline.

For a declared STRUCTURAL LINK, physics-author the part with the closed-form / topology authors
(via the T2 ``author_part``) instead of the LLM -- lighter, load-verified geometry with a
certificate, spliced into the multi-part run as a resumed-style ``LoopResult``. Gated on TWO envs
(both required; absent -> byte-inert, the LLM authors everything as today):

  FORGE_PHYSICS_PARTS="upper_arm,forearm"          # part-dir slugs eligible for physics authoring
  FORGE_PHYSICS_CASE="payload_kg=0.5;bound_mm=0.046;hy=24;wx=64"  # the OPERATOR-DECLARED case

The load case is DECLARED by the operator (the rows-73-77 benchmark pattern -- how M1 beat the
SO-101 reference), never invented from the part prompt (a link's prompt states geometry, not a
load). The certificate/train-row echo ``case_provenance: declared-by-operator``.

ORIENTATION: the analytic link builds its LENGTH along +X (``sections.build_link``: "Long axis
X"); a plan's link ports may span any axis, and ``_deterministic_seat`` is translation-only. So
the authored geometry is rotated so its +X length axis aligns with the part's port-to-port vector
BEFORE it is returned (identity when the ports already span X, e.g. the M2-arm v4 plan). Fail-open
everywhere: any gap -> None, and the LLM author runs unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from forge_mind.agentic.loop import LoopResult
from forge_mind.structures.loadcase import G_M_S2, LoadCase
from forge_mind.structures.spec import RequirementSpec


def _targets() -> set[str]:
    return {s.strip() for s in os.environ.get("FORGE_PHYSICS_PARTS", "").split(",") if s.strip()}


def _parse_case(raw: str) -> dict[str, float]:
    """'payload_kg=0.5;bound_mm=0.046;hy=24;wx=64' -> {k: float}. Malformed pairs are skipped."""
    out: dict[str, float] = {}
    for pair in raw.split(";"):
        if "=" in pair:
            k, _, v = pair.partition("=")
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                continue
    return out


def _port_span(part: Any) -> tuple[float, tuple[float, float, float]] | None:
    """(length, unit port-to-port vector) from a link's two bore ports, or None (not a 2-port
    link). Deterministic -- the length is the distance between the two bore origins."""
    ports = list(getattr(part, "ports", []) or [])
    bores = [p for p in ports if "bore" in p.name.lower()] or ports
    if len(bores) < 2:
        return None
    a = np.array([bores[0].origin.x, bores[0].origin.y, bores[0].origin.z], dtype=float)
    b = np.array([bores[1].origin.x, bores[1].origin.y, bores[1].origin.z], dtype=float)
    v = b - a
    length = float(np.linalg.norm(v))
    if length < 1.0:
        return None
    u = v / length
    return length, (float(u[0]), float(u[1]), float(u[2]))


def _synthesize_spec(
    part: Any, length: float, case: dict[str, float], material_id: str
) -> RequirementSpec:
    """The DECLARED spec for this link: envelope from the port span + the case's section (hy, wx),
    load + bound from the operator-declared case. No invented numbers; components are none (a bare
    structural span -> the router sends it to the analytic author)."""
    hy = case.get("hy", 24.0)
    wx = case.get("wx", 64.0)
    return RequirementSpec(
        envelope_mm=(length, hy, wx),  # analytic reads (length, hy, wx)
        material_id=material_id,
        load=LoadCase(
            name=f"{part.name.replace(' ', '_')}_declared", force_n=case["payload_kg"] * G_M_S2
        ),
        deflection_bound_mm=case["bound_mm"],
    )


def _rotation_x_to(v_unit: tuple[float, float, float]) -> np.ndarray:
    """A 4x4 that rotates +X onto ``v_unit`` (Rodrigues; identity when already +X, 180deg when
    anti-parallel). The authored link's length axis is +X."""
    x = np.array([1.0, 0.0, 0.0])
    v = np.array(v_unit, dtype=float)
    c = float(np.dot(x, v))
    if c > 1.0 - 1e-9:
        return np.eye(4)
    if c < -1.0 + 1e-9:  # anti-parallel: 180deg about any perpendicular axis (Z)
        r = np.diag([-1.0, -1.0, 1.0])
    else:
        axis = np.cross(x, v)
        axis = axis / np.linalg.norm(axis)
        s = float(np.linalg.norm(np.cross(x, v)))
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        r = np.eye(3) + s * k + (1 - c) * (k @ k)
    m = np.eye(4)
    m[:3, :3] = r
    return m


def _oriented(
    stl: Path, step: Path | None, v_unit: tuple[float, float, float], out_dir: Path
) -> tuple[Path, Path | None]:
    """Rotate the authored geometry so its +X length axis aligns with ``v_unit`` and re-export
    (identity -> return the originals). Fail-soft: on any export error return the originals."""
    m = _rotation_x_to(v_unit)
    if np.allclose(m, np.eye(4)):
        return stl, step
    try:
        import trimesh

        out_dir.mkdir(parents=True, exist_ok=True)
        mesh: Any = trimesh.load(str(stl), force="mesh")
        mesh.apply_transform(m)
        out_stl = out_dir / "hybrid_oriented.stl"
        mesh.export(str(out_stl))
        out_step: Path | None = None
        if step is not None and Path(step).exists():
            from build123d import Location, export_step, import_step

            solid = import_step(str(step)).moved(Location(m))
            out_step = out_dir / "hybrid_oriented.step"
            export_step(solid, str(out_step))
        return out_stl, out_step
    except Exception:
        return stl, step


def try_hybrid_physics_author(
    part: Any, part_id: str, out_dir: Path, mat: Any, client: Any = None
) -> LoopResult | None:
    """The hook: physics-author ``part`` if it is a declared structural link. Returns a
    resumed-style LoopResult carrying the (oriented) physics geometry when the uniform gate is
    CLAIMABLE, else None (the LLM author runs). Byte-inert unless both envs are set; fail-open."""
    raw_case = os.environ.get("FORGE_PHYSICS_CASE", "").strip()
    if part_id not in _targets() and part.name not in _targets():
        return None
    if not raw_case:
        return None
    try:
        from forge_mind.structures.dispatch import author_part, refusing_author

        case = _parse_case(raw_case)
        if "payload_kg" not in case or "bound_mm" not in case:
            return None
        span = _port_span(part)
        if span is None:
            return None  # not a resolvable 2-port link -> the LLM authors it
        length, v_unit = span
        material_id = getattr(mat, "name", None) or "pla"
        spec = _synthesize_spec(part, length, case, material_id.lower())
        outcome = author_part(
            spec,
            out_dir / f"{part_id}_hybrid",
            authors={
                "agentic": refusing_author(
                    "agentic", "hybrid link authoring: the LLM authors non-physics parts"
                )
            },
            part_name=part.name,
        )
        final = outcome["final"]
        gate = outcome["attempts"][-1]["gate"]
        # T6 flywheel label (fail-open): the method-SELECTION signal for this in-pipeline part.
        try:
            import dataclasses as _dc

            from forge_mind.train_log import log_selection

            log_selection(
                part.prompt,
                provenance=final.provenance,
                spec=_dc.asdict(spec),
                checks=dict(gate["checks"]),
                claimable=bool(gate["claimable"]),
                material=material_id,
                object_name=part.name,
                source="functional_hybrid",
            )
        except Exception:
            pass
        if not gate["claimable"] or final.stl is None:
            print(
                f"hybrid: '{part.name}' not claimable by physics "
                f"({[a['rung'] for a in outcome['attempts']]}); LLM authors it",
                flush=True,
            )
            return None
        stl, step = _oriented(Path(final.stl), final.step, v_unit, out_dir / f"{part_id}_hybrid")
        print(
            f"hybrid: '{part.name}' physics-authored ({final.provenance}, "
            f"case declared-by-operator); 0 LLM calls",
            flush=True,
        )
        return LoopResult(
            prompt=part.prompt,
            brief=None,
            final_code=None,
            final_stl=stl,
            final_step=step,
            final_render=None,
            validation=None,
            critique=None,
            geometry_passed=True,
            critic_approved=True,
            stopped_reason="accepted",
            iterations=[],
            resumed_hard_ok=bool(gate["checks"].get("full_hard_ok") == "pass"),
        )
    except Exception:  # fail-open: any gap -> the LLM author runs unchanged
        return None


__all__ = ["try_hybrid_physics_author"]
