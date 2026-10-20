"""STEP export integrity proof (ISO 10303-49 spirit).

We hand the user an editable B-rep STEP alongside the STL (ADR-0024). This module answers the only
question that makes that STEP trustworthy: **will a downstream CAD tool read back the SAME solid we
exported?** Two layers of evidence (ADR-0037 rung 2):

1. **Mass-properties roundtrip** — re-import the exported STEP and compare volume / surface area /
   centroid against the source body (numerical-noise tolerances).
2. **Named-B-rep fidelity** — the file itself must BE real CAD: a recognised AP schema, analytic
   B-rep entities (``MANIFOLD_SOLID_BREP`` + ``ADVANCED_FACE``), no tessellated geometry, a PRODUCT
   named after the part, and a topology roundtrip (solid + face counts source == re-import). A
   volume-matching tessellated lump fails layer 2 even though it passes layer 1.

REPORTED-only, never gates, and never raises (returns ``None`` on any failure) — the STL the user
prints is unaffected, and a false refusal here would be worse than an honest "couldn't prove it".
Layer-2 signals therefore only DEMOTE ``matches`` when they affirmatively detect a problem; an
unreadable file leaves them ``None`` and layer 1 stands alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_mind.types import Vec3
from forge_mind.validator.report import StepProof

# A faithful B-rep roundtrip matches to numerical noise; these absorb OCCT format noise while still
# catching a real drop (a missing feature shifts volume/area/centroid far beyond them).
_VOLUME_REL_TOL = 1e-3
_AREA_REL_TOL = 1e-3
_CENTROID_ABS_TOL_MM = 1e-2

# Read cap for the file-census pass: our parts are tens of KB; anything bigger is read in part
# (the header + early DATA section carry the schema/product lines the census needs).
_MAX_CENSUS_BYTES = 8_000_000

# PRODUCT names OCCT emits when nothing named the shape — present means NOT named after the part.
_DEFAULT_PRODUCT_NAMES = frozenset({"", "SOLID", "COMPOUND", "SHELL", "Open CASCADE Model"})


@dataclass(frozen=True)
class StepFileFacts:
    """The text-level census of one STEP file (pure; see ``inspect_step_text``)."""

    schema_id: str | None
    is_brep: bool
    tessellated: bool
    product_name: str | None
    named: bool


def inspect_step_text(text: str) -> StepFileFacts:
    """Census the ISO-10303-21 text: schema, B-rep vs tessellated entities, product naming.

    Pure + deterministic (regex token counts on the file the user actually downloads — the same
    evidence a receiving CAD kernel parses)."""
    schema_match = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", text)
    schema_id: str | None = None
    if schema_match:
        raw = schema_match.group(1)
        if "10303 214" in raw or "AUTOMOTIVE_DESIGN" in raw:
            schema_id = "AP214"
        elif "10303 242" in raw or "AP242" in raw:
            schema_id = "AP242"
        elif "10303 203" in raw or "CONFIG_CONTROL_DESIGN" in raw:
            schema_id = "AP203"
        else:
            schema_id = raw.split()[0] if raw else None

    solids = len(re.findall(r"MANIFOLD_SOLID_BREP", text))
    faces = len(re.findall(r"ADVANCED_FACE", text))
    tessellated = bool(re.search(r"TESSELLATED|TRIANGULATED", text))
    is_brep = solids >= 1 and faces >= 1

    product = re.search(r"PRODUCT\s*\(\s*'([^']*)'", text)
    product_name = product.group(1) if product else None
    named = product_name is not None and product_name not in _DEFAULT_PRODUCT_NAMES

    return StepFileFacts(
        schema_id=schema_id,
        is_brep=is_brep,
        tessellated=tessellated,
        product_name=product_name,
        named=named,
    )


def _props(shape: Any) -> tuple[float, float, Vec3]:
    """Volume (mm^3), surface area (mm^2), and centroid (mm) of a build123d shape."""
    c = shape.center()
    return float(shape.volume), float(shape.area), Vec3(x=float(c.X), y=float(c.Y), z=float(c.Z))


def _topology(shape: Any) -> tuple[int, int] | None:
    try:
        return len(shape.solids()), len(shape.faces())
    except Exception:
        return None


def verify_step_roundtrip(part: Any, step_path: Path) -> StepProof | None:
    """Re-import ``step_path`` and prove it is the SAME named B-rep as ``part``.

    Returns a :class:`StepProof` (with ``matches`` set), or ``None`` if the STEP is missing or could
    not be re-imported. Never raises — REPORTED, never a gate. build123d is imported lazily so the
    validator's import graph stays light.
    """
    try:
        from build123d import import_step

        if not Path(step_path).exists():
            return None
        src_v, src_a, src_c = _props(part)
        reimported = import_step(str(step_path))
        re_v, re_a, re_c = _props(reimported)
    except Exception:
        return None

    v_err = abs(re_v - src_v) / src_v if src_v else (0.0 if re_v == 0.0 else 1.0)
    a_err = abs(re_a - src_a) / src_a if src_a else (0.0 if re_a == 0.0 else 1.0)
    c_dist = ((re_c.x - src_c.x) ** 2 + (re_c.y - src_c.y) ** 2 + (re_c.z - src_c.z) ** 2) ** 0.5
    mass_props_ok = (
        v_err <= _VOLUME_REL_TOL and a_err <= _AREA_REL_TOL and c_dist <= _CENTROID_ABS_TOL_MM
    )

    # Layer 2 — named-B-rep fidelity. Best-effort: a census/topology failure leaves the fields
    # None, which never demotes (no new false-FAIL from an unreadable file).
    facts: StepFileFacts | None = None
    try:
        with open(step_path, encoding="utf-8", errors="replace") as fh:
            facts = inspect_step_text(fh.read(_MAX_CENSUS_BYTES))
    except Exception:
        facts = None

    solid_count: int | None = None
    face_count: int | None = None
    topology_matches: bool | None = None
    src_topo = _topology(part)
    re_topo = _topology(reimported)
    if re_topo is not None:
        solid_count, face_count = re_topo
        if src_topo is not None:
            topology_matches = src_topo == re_topo

    # Demote on AFFIRMATIVE problems only: a tessellated/no-brep file or a topology drift is not
    # a faithful B-rep, however well the volume agrees.
    matches = mass_props_ok
    if facts is not None and (not facts.is_brep or facts.tessellated):
        matches = False
    if topology_matches is False:
        matches = False

    return StepProof(
        matches=matches,
        volume_mm3=re_v,
        area_mm2=re_a,
        centroid=re_c,
        volume_rel_err=v_err,
        area_rel_err=a_err,
        centroid_dist_mm=c_dist,
        schema_id=facts.schema_id if facts is not None else None,
        is_brep=facts.is_brep if facts is not None else None,
        tessellated=facts.tessellated if facts is not None else None,
        product_name=facts.product_name if facts is not None else None,
        named=facts.named if facts is not None else None,
        solid_count=solid_count,
        face_count=face_count,
        topology_matches=topology_matches,
    )


__all__ = ["StepFileFacts", "inspect_step_text", "verify_step_roundtrip"]
