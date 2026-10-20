"""Audit the author's build123d code for STANDARD-dimension fidelity (ADR-0009 enforcement).

The research brief tags a feature with a ``standard`` designation (``M3``, ``608``, ...) precisely so
the author uses the EXACT ISO/DIN dimension the standards helpers hand it, never a guess. This audit
confirms it actually did: it AST-parses the authored code, collects which standards helpers were
called and with which designation, then

  * CITES each fetched standard from the SAME registry the helper + the validator's conformance gate
    use ("M3 clearance hole: 3.40 mm (ISO 273)") -- the verifiable "cited standard" a senior engineer
    asks for; and
  * FAILS when the brief DECLARED a standard the code never fetched -- the hand-modeled-dimension
    bypass that "undoes the whole pitch" (one wrong M3 hole). Refusing is correct here: a guessed
    standard dimension is exactly the silent-wrong-part the no-false-PASS rule forbids.

Deterministic, no geometry, no LLM, no sandbox round-trip -- it reads the code the parent already
holds. Designation-level match (did the author fetch ANYTHING standard for ``M3``?) is deliberate:
the point is that the registry, not the model, set the number.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from forge_mind.dsl.standards import ALL_REGISTRIES
from forge_mind.validator.conformance import _CATEGORIES

# author-facing helper name -> the conformance category (so a cite uses the same registry resolver the
# gate does). Keep in lockstep with forge_mind.agentic.standards_helpers.
_HELPER_CATEGORY: dict[str, str] = {
    "clearance_hole_d": "clearance_hole",
    "tap_hole_d": "tapped_hole",
    # The cheatsheet TEACHES thread_hole_d for threaded/set-screw holes (ADR-0081, the
    # radial_set_screw_holes idiom) -- the auditor refusing it made the engine's left hand
    # reject what its right hand instructed (mech_star: permanent engine_passed=False on a
    # correct part; engine deep review 2026-07-04).
    "thread_hole_d": "threaded_hole",
    "counterbore_dims": "counterbore",
    "nut_pocket_dims": "captive_nut_pocket",
    "bearing_seat_dims": "bearing_seat",
    "heat_set_hole_d": "heat_set_insert",
    "dowel_hole_d": "dowel_hole",
    "magnet_pocket_dims": "magnet_pocket",
}


@dataclass(frozen=True)
class StandardUse:
    """One standards-helper call found in the authored code."""

    helper: str
    category: str
    designation: str


def _norm(designation: str) -> str:
    """Normalize a designation for matching: 'm3' -> 'M3'; everything else stripped, as-is."""
    d = designation.strip()
    if len(d) >= 2 and d[0] in "mM" and d[1].isdigit():
        return "M" + d[1:]
    return d


def _resolvable(designation: str) -> bool:
    """True if SOME standards registry can supply this designation (a helper exists that returns a
    real value for it). When NOTHING can supply it, ordering the author to "call the helper" is a
    poison-pill: the helper RAISES on an unknown size and crashes the build, so the author oscillates
    between a refused hand-model and a crashing helper call and the loop never converges (the measured
    M10 crankshaft trap, ADR-0067). The real fix is to add the size to a registry (done for M10-M16);
    this guard keeps any remaining absent/exotic size from creating an unsatisfiable instruction."""
    nd = _norm(designation)
    for reg in ALL_REGISTRIES.values():
        try:
            if reg.get(nd) is not None:
                return True
        except (KeyError, ValueError):
            continue
    return False


def _call_name(func: ast.expr) -> str:
    """The bare function name of a call target, whether ``f(...)`` or ``mod.f(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def fetched_standards(code: str) -> list[StandardUse]:
    """Every standards-helper call in the authored code with a literal string designation."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    uses: list[StandardUse] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        category = _HELPER_CATEGORY.get(_call_name(node.func))
        if category is None or not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            uses.append(
                StandardUse(helper=_call_name(node.func), category=category, designation=arg0.value)
            )
    return uses


def cite_standard(category: str, designation: str) -> str | None:
    """A human citation ('clearance hole M3: 3.40 mm (ISO 273)') for a fetched standard, else None."""
    entry = _CATEGORIES.get(category)
    if entry is None:
        return None
    reg_name, resolver = entry
    try:
        value = resolver(designation, None)
        spec = ALL_REGISTRIES[reg_name].get(designation)
    except (ValueError, KeyError):
        return None
    if spec is None:
        return None
    return f"{category.replace('_', ' ')} {designation}: {value:.2f} mm ({spec.source})"


def audit_standards(code: str, declared: list[str]) -> tuple[list[str], list[str]]:
    """Audit authored ``code`` against the brief's ``declared`` standard designations.

    Returns ``(cites, fails)``:
      * ``cites`` -- the verified ISO/DIN dimensions the code FETCHED (the cited-standard proof);
      * ``fails`` -- one message per declared standard the code did NOT fetch (a hand-modeled value).
    """
    uses = fetched_standards(code)
    fetched = {_norm(u.designation) for u in uses}

    cites: list[str] = []
    for u in uses:
        cite = cite_standard(u.category, u.designation)
        if cite is not None:
            cites.append(cite)

    fails: list[str] = []
    seen: set[str] = set()
    for d in declared:
        nd = _norm(d)
        if nd in seen or nd in fetched:
            seen.add(nd)
            continue
        seen.add(nd)
        # Only DEMAND the helper for a size a registry can actually supply. Refusing a hand-modeled
        # value for an UNSUPPLIED size (nothing to fetch) would force a helper call that crashes the
        # build -- an unsatisfiable, loop-poisoning instruction (ADR-0067). Fail-closed still holds:
        # supplied sizes are enforced; unsupplied ones fall through to the geometry gates.
        if not _resolvable(d):
            continue
        fails.append(
            f"the brief specifies the standard part {d!r}, but the authored code never fetched its "
            f"ISO/DIN dimension via the standards helper -- a hand-modeled (likely non-conformant) "
            f"value, which is refused. Call the helper, e.g. clearance_hole_d({d!r})."
        )
    return cites, fails


__all__ = ["StandardUse", "audit_standards", "cite_standard", "fetched_standards"]
